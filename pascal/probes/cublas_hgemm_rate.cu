// Compares cublasHgemm against cublasGemmEx with fp32 compute, at prefill
// shapes, on the target card.
//
// Decode was fixed by moving exllama's own arithmetic to fp32, but prefill does
// not run that code at all. Above MAX_Q_GEMM_ROWS (50 rows) gemm_half_q_half_cuda
// reconstructs the full fp16 weight matrix and calls cublasHgemm -- which
// requests half *compute*, not merely half storage. If cuBLAS honours that on
// sm_61 it runs on the same 1/56-rate FP16x2 unit, and prefill (plus any batch
// above 50) is paying the identical penalty one level up.
//
// cublasGemmEx with CUBLAS_COMPUTE_32F keeps fp16 inputs and outputs while
// accumulating in fp32, so it is a drop-in that is also strictly more accurate.
// Whether it is faster is a question about what cuBLAS actually emits, which is
// why this measures instead of assuming.
//
//   nvcc -arch=sm_61 -O3 -lcublas -o cublas_rate cublas_hgemm_rate.cu && ./cublas_rate

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

struct Shape {
  int m, k, n;
  const char* label;
};

static float time_it(cublasHandle_t h, const Shape& s, bool use_ex,
                     const half* a, const half* b, half* c) {
  const half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
  const float alpha_f = 1.0f, beta_f = 0.0f;

  auto launch = [&]() {
    if (use_ex) {
      // fp16 in/out, fp32 accumulate.
      cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, s.n, s.m, s.k, &alpha_f, b,
                   CUDA_R_16F, s.n, a, CUDA_R_16F, s.k, &beta_f, c, CUDA_R_16F,
                   s.n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    } else {
      // What q_gemm.cu calls today: half compute.
      cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, s.n, s.m, s.k, &alpha_h, b, s.n,
                  a, s.k, &beta_h, c, s.n);
    }
  };

  for (int i = 0; i < 3; i++) launch();  // warm up / autotune
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float best = 1e30f;
  for (int r = 0; r < 5; r++) {
    cudaEventRecord(start);
    launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    if (ms < best) best = ms;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return best;
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

  cublasHandle_t h;
  cublasCreate(&h);

  // Qwen3.5-2B: hidden 2048, intermediate 6144. Row counts span the
  // reconstruct threshold (50) into real prefill widths.
  std::vector<Shape> shapes = {
      {64, 2048, 2048, "batch64  qkv/o  "},
      {128, 2048, 6144, "batch128 mlp up "},
      {512, 2048, 2048, "prefill512 o    "},
      {512, 2048, 6144, "prefill512 mlp  "},
      {2048, 2048, 6144, "prefill2048 mlp "},
  };

  printf("%-18s %10s %10s %8s\n", "shape", "Hgemm ms", "GemmEx ms", "speedup");
  printf("%s\n", "--------------------------------------------------------");

  for (const auto& s : shapes) {
    half *a, *b, *c;
    cudaMalloc(&a, (size_t)s.m * s.k * sizeof(half));
    cudaMalloc(&b, (size_t)s.k * s.n * sizeof(half));
    cudaMalloc(&c, (size_t)s.m * s.n * sizeof(half));
    cudaMemset(a, 0x3c, (size_t)s.m * s.k * sizeof(half));
    cudaMemset(b, 0x3c, (size_t)s.k * s.n * sizeof(half));

    const float ms_h = time_it(h, s, false, a, b, c);
    const float ms_e = time_it(h, s, true, a, b, c);
    printf("%-18s %10.3f %10.3f %7.2fx\n", s.label, ms_h, ms_e, ms_h / ms_e);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  }

  printf(
      "\nIf GemmEx wins, q_gemm.cu's cublasHgemm call is costing prefill and\n"
      "any batch above MAX_Q_GEMM_ROWS the same 1/56 fp16 penalty that the\n"
      "decode kernel was just fixed for.\n");

  cublasDestroy(h);
  return 0;
}
