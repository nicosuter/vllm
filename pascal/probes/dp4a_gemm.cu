// Prototype W4A8 GEMM on dp4a, checked for correctness and timed against the
// path it would replace.
//
// Standalone on purpose. Integrating a new linear kernel into vLLM costs
// registration, an MPLinearKernel, weight-transform plumbing and a rebuild
// cycle; none of that is worth paying until the kernel is known to beat what it
// replaces. So this compiles with nvcc alone and answers one question: at
// prefill and large-batch shapes, is int4-weight x int8-activation on dp4a
// faster than reconstruct-to-fp16 plus cublasGemmEx?
//
// What it replaces: above MAX_Q_GEMM_ROWS the exllama path dequantizes the
// entire weight matrix to fp16 and calls cuBLAS. That cuBLAS call, after this
// fork switched it to CUBLAS_COMPUTE_32F, already runs at ~88% of the card's
// fp32 peak, so the bar is high. dp4a's advantage is a 4.00x higher roof
// (pascal/probes/dp4a_rate.cu), no reconstruct pass at all, and half the weight
// traffic since the int4 stays packed.
//
// Numerics. Weights are symmetric uint4b8: stored 0..15, value q-8, with an
// fp16 scale per group of 32 along K. Activations are quantized per row to int8
// with an fp32 scale, which is what vLLM's dynamic_scaled_int8_quant already
// produces. Because a K-group is exactly 32 elements and the weight scale is
// constant across it, the dot product accumulates in int32 within a group and
// converts once per group:
//
//     out[m][n] = a_scale[m] * sum_g ( w_scale[g][n] * sum_{k in g} a[m][k]*w[k][n] )
//
// The inner sum cannot overflow int32: 32 terms of at most 127*8.
//
//   nvcc -arch=sm_61 -O3 -lcublas -o dp4a_gemm dp4a_gemm.cu && ./dp4a_gemm

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

#define GROUP 32          // K elements sharing one weight scale
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64        // two groups per tile: halves the __syncthreads count
#define GROUPS_PER_TILE (BLOCK_K / GROUP)
#define THREADS 256
#define TM 4              // 4x4. 8x4 was tried and lost: 64 accumulators plus
#define TN 4              // addressing spills occupancy, and BLOCK_M=128 wastes
                          // half the tile at batch 64.

#define CHECK(x)                                                          \
  do {                                                                    \
    cudaError_t e = (x);                                                  \
    if (e != cudaSuccess) {                                               \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

// A: [M, K] int8 row-major.  a_scale: [M] fp32.
// B: [K/8, N] int32, eight 4-bit values along K per word, low nibble first.
// w_scale: [K/GROUP, N] fp16.  C: [M, N] fp16.
__global__ void w4a8_dp4a_gemm(const int8_t* __restrict__ A,
                               const uint32_t* __restrict__ B,
                               const half* __restrict__ w_scale,
                               const float* __restrict__ a_scale,
                               half* __restrict__ C, int M, int N, int K) {
  const int tid = threadIdx.x;
  const int block_m = blockIdx.y * BLOCK_M;
  const int block_n = blockIdx.x * BLOCK_N;

  // Thread's TM x TN output tile within the block tile.
  const int tm = (tid / (BLOCK_N / TN)) * TM;
  const int tn = (tid % (BLOCK_N / TN)) * TN;

  // Both tiles are stored k-contiguous so a thread's four consecutive K values
  // are one 32-bit load rather than four strided byte loads. For sB that means
  // transposing to [n][k], which is also the natural write order when unpacking
  // an int32 of eight consecutive K values for a single column.
  //
  // The +4 padding breaks bank conflicts: with a 32-byte row stride, threads
  // reading consecutive n land on banks 8 apart and collide 4 ways.
  __shared__ int8_t sA[BLOCK_M][BLOCK_K + 4];
  __shared__ int8_t sB[BLOCK_N][BLOCK_K + 4];

  float acc[TM][TN] = {};

  for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
    // Stage A: BLOCK_M x BLOCK_K int8, 8 bytes per thread per pass.
    for (int idx = tid * 8; idx < BLOCK_M * BLOCK_K; idx += THREADS * 8) {
      const int r = idx / BLOCK_K, c = idx % BLOCK_K;
      const int gr = block_m + r;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        sA[r][c + j] = (gr < M) ? A[(size_t)gr * K + k0 + c + j] : 0;
      }
    }

    // Stage B: unpack BLOCK_K x BLOCK_N int4 into int8, subtracting the
    // symmetric bias of 8 here so the inner loop is a plain signed dot product.
    // Each int32 holds 8 consecutive K values for one column.
    for (int idx = tid; idx < (BLOCK_K / 8) * BLOCK_N; idx += THREADS) {
      const int kw = idx / BLOCK_N;      // which run of 8 along K
      const int c = idx % BLOCK_N;
      const int gn = block_n + c;
      const uint32_t packed =
          (gn < N) ? B[(size_t)(k0 / 8 + kw) * N + gn] : 0x88888888u;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        sB[c][kw * 8 + j] = (int8_t)((int)((packed >> (4 * j)) & 0xF) - 8);
      }
    }
    __syncthreads();

    // One tile spans GROUPS_PER_TILE weight-scale groups. int32 accumulates
    // within a group, where the scale is constant, and converts once at its
    // end -- so widening the tile costs nothing in conversions and saves the
    // __syncthreads pair that a group-sized tile pays for every 32 elements.
#pragma unroll
    for (int gi = 0; gi < GROUPS_PER_TILE; gi++) {
      int iacc[TM][TN] = {};
#pragma unroll
      for (int kk = gi * GROUP; kk < (gi + 1) * GROUP; kk += 4) {
        int a4[TM], b4[TN];
#pragma unroll
        for (int i = 0; i < TM; i++) {
          a4[i] = *reinterpret_cast<const int*>(&sA[tm + i][kk]);
        }
#pragma unroll
        for (int j = 0; j < TN; j++) {
          b4[j] = *reinterpret_cast<const int*>(&sB[tn + j][kk]);
        }
#pragma unroll
        for (int i = 0; i < TM; i++) {
#pragma unroll
          for (int j = 0; j < TN; j++) {
            iacc[i][j] = __dp4a(a4[i], b4[j], iacc[i][j]);
          }
        }
      }

      const int g = k0 / GROUP + gi;
#pragma unroll
      for (int j = 0; j < TN; j++) {
        const int gn = block_n + tn + j;
        const float ws =
            (gn < N) ? __half2float(w_scale[(size_t)g * N + gn]) : 0.0f;
#pragma unroll
        for (int i = 0; i < TM; i++) {
          acc[i][j] = fmaf((float)iacc[i][j], ws, acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; i++) {
    const int gm = block_m + tm + i;
    if (gm >= M) continue;
    const float as = a_scale[gm];
#pragma unroll
    for (int j = 0; j < TN; j++) {
      const int gn = block_n + tn + j;
      if (gn < N) C[(size_t)gm * N + gn] = __float2half(acc[i][j] * as);
    }
  }
}

// Reference: the same arithmetic in fp64 on the host, so a disagreement points
// at the kernel rather than at accumulated fp32 error.
static void reference(const std::vector<int8_t>& A, const std::vector<uint32_t>& B,
                      const std::vector<float>& ws, const std::vector<float>& as,
                      std::vector<float>& C, int M, int N, int K) {
  const int groups = K / GROUP;
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      double sum = 0.0;
      for (int g = 0; g < groups; g++) {
        long long isum = 0;
        for (int k = g * GROUP; k < (g + 1) * GROUP; k++) {
          const uint32_t packed = B[(size_t)(k / 8) * N + n];
          const int w = (int)((packed >> (4 * (k % 8))) & 0xF) - 8;
          isum += (long long)A[(size_t)m * K + k] * w;
        }
        sum += (double)isum * ws[(size_t)g * N + n];
      }
      C[(size_t)m * N + n] = (float)(sum * as[m]);
    }
  }
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

  struct Shape { int m, k, n; const char* label; };
  std::vector<Shape> shapes = {
      {64, 2048, 2048, "batch64  qkv/o "},
      {512, 2048, 2048, "prefill512 o   "},
      {512, 2048, 6144, "prefill512 mlp "},
      {2048, 2048, 6144, "prefill2048 mlp"},
  };

  cublasHandle_t cub;
  cublasCreate(&cub);

  // Correctness on a small shape first: a fast wrong kernel is worthless.
  {
    const int M = 64, K = 256, N = 64;
    std::vector<int8_t> hA((size_t)M * K);
    std::vector<uint32_t> hB((size_t)(K / 8) * N);
    std::vector<float> hWs((size_t)(K / GROUP) * N), hAs(M);
    srand(1);
    for (auto& v : hA) v = (int8_t)(rand() % 255 - 127);
    for (auto& v : hB) v = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
    for (auto& v : hWs) v = 0.01f + 0.02f * (rand() / (float)RAND_MAX);
    for (auto& v : hAs) v = 0.005f + 0.01f * (rand() / (float)RAND_MAX);

    std::vector<half> hWsH(hWs.size());
    for (size_t i = 0; i < hWs.size(); i++) hWsH[i] = __float2half(hWs[i]);

    int8_t* dA; uint32_t* dB; half *dWs, *dC; float* dAs;
    CHECK(cudaMalloc(&dA, hA.size()));
    CHECK(cudaMalloc(&dB, hB.size() * 4));
    CHECK(cudaMalloc(&dWs, hWsH.size() * 2));
    CHECK(cudaMalloc(&dAs, hAs.size() * 4));
    CHECK(cudaMalloc(&dC, (size_t)M * N * 2));
    CHECK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dWs, hWsH.data(), hWsH.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dAs, hAs.data(), hAs.size() * 4, cudaMemcpyHostToDevice));

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    w4a8_dp4a_gemm<<<grid, THREADS>>>(dA, dB, dWs, dAs, dC, M, N, K);
    CHECK(cudaDeviceSynchronize());

    std::vector<half> hC((size_t)M * N);
    CHECK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost));

    // Reference uses the fp16-rounded scales the kernel actually sees.
    std::vector<float> wsSeen(hWs.size());
    for (size_t i = 0; i < hWs.size(); i++) wsSeen[i] = __half2float(hWsH[i]);
    std::vector<float> ref((size_t)M * N);
    reference(hA, hB, wsSeen, hAs, ref, M, N, K);

    double max_rel = 0.0, denom = 0.0;
    for (size_t i = 0; i < ref.size(); i++) denom = fmax(denom, fabs(ref[i]));
    for (size_t i = 0; i < ref.size(); i++)
      max_rel = fmax(max_rel, fabs(__half2float(hC[i]) - ref[i]) / fmax(denom, 1e-6));
    printf("correctness (M=%d K=%d N=%d): max_rel_err = %.3e  %s\n\n", M, K, N,
           max_rel, max_rel < 2e-3 ? "PASS" : "FAIL");
    if (max_rel >= 2e-3) { printf("kernel is wrong; timings withheld\n"); return 1; }

    cudaFree(dA); cudaFree(dB); cudaFree(dWs); cudaFree(dAs); cudaFree(dC);
  }

  printf("%-17s %12s %12s %9s %10s\n", "shape", "dp4a ms", "cuBLAS ms",
         "speedup", "dp4a TOP/s");
  printf("%s\n", "----------------------------------------------------------------------");

  for (const auto& s : shapes) {
    int8_t* dA; uint32_t* dB; half *dWs, *dC; float* dAs;
    CHECK(cudaMalloc(&dA, (size_t)s.m * s.k));
    CHECK(cudaMalloc(&dB, (size_t)(s.k / 8) * s.n * 4));
    CHECK(cudaMalloc(&dWs, (size_t)(s.k / GROUP) * s.n * 2));
    CHECK(cudaMalloc(&dAs, (size_t)s.m * 4));
    CHECK(cudaMalloc(&dC, (size_t)s.m * s.n * 2));
    CHECK(cudaMemset(dA, 1, (size_t)s.m * s.k));
    CHECK(cudaMemset(dB, 0x37, (size_t)(s.k / 8) * s.n * 4));
    CHECK(cudaMemset(dWs, 0x2c, (size_t)(s.k / GROUP) * s.n * 2));
    CHECK(cudaMemset(dAs, 0, (size_t)s.m * 4));

    dim3 grid((s.n + BLOCK_N - 1) / BLOCK_N, (s.m + BLOCK_M - 1) / BLOCK_M);
    auto launch_dp4a = [&] {
      w4a8_dp4a_gemm<<<grid, THREADS>>>(dA, dB, dWs, dAs, dC, s.m, s.n, s.k);
    };

    // fp16 operands for the cuBLAS comparison: this is what the reconstruct
    // path hands it, minus the reconstruct itself, so cuBLAS is being flattered.
    half *fA, *fB, *fC;
    CHECK(cudaMalloc(&fA, (size_t)s.m * s.k * 2));
    CHECK(cudaMalloc(&fB, (size_t)s.k * s.n * 2));
    CHECK(cudaMalloc(&fC, (size_t)s.m * s.n * 2));
    CHECK(cudaMemset(fA, 0x3c, (size_t)s.m * s.k * 2));
    CHECK(cudaMemset(fB, 0x3c, (size_t)s.k * s.n * 2));
    const float alpha = 1.0f, beta = 0.0f;
    auto launch_cublas = [&] {
      cublasGemmEx(cub, CUBLAS_OP_N, CUBLAS_OP_N, s.n, s.m, s.k, &alpha, fB,
                   CUDA_R_16F, s.n, fA, CUDA_R_16F, s.k, &beta, fC, CUDA_R_16F,
                   s.n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };

    auto timeit = [&](auto&& fn) {
      for (int i = 0; i < 5; i++) fn();       // warm: this card idles at 164 MHz
      CHECK(cudaDeviceSynchronize());
      cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
      float best = 1e30f;
      for (int r = 0; r < 7; r++) {
        cudaEventRecord(a); fn(); cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
      }
      cudaEventDestroy(a); cudaEventDestroy(b);
      return best;
    };

    const float ms_d = timeit(launch_dp4a);
    const float ms_c = timeit(launch_cublas);
    const double ops = 2.0 * s.m * s.n * s.k;
    printf("%-17s %12.3f %12.3f %8.2fx %10.2f\n", s.label, ms_d, ms_c,
           ms_c / ms_d, ops / (ms_d * 1e9));

    cudaFree(dA); cudaFree(dB); cudaFree(dWs); cudaFree(dAs); cudaFree(dC);
    cudaFree(fA); cudaFree(fB); cudaFree(fC);
  }

  printf(
      "\ncuBLAS here is timed without the reconstruct pass that really precedes\n"
      "it, so its column is optimistic by a fixed per-forward cost the dp4a\n"
      "path does not pay at all.\n");
  cublasDestroy(cub);
  return 0;
}
