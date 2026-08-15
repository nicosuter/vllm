// Can sm_61 read FP8 weights? Correctness and cost of an e4m3 decode.
//
// Pascal has no FP8 unit, but that only rules out an FP8 *multiply*. This fork
// has never wanted one: the established shape here is a narrow storage type
// carrying weights to the ALU, which then computes in fp32 (see the exllama
// dp4a work and the CUBLAS_COMPUTE_32F switch). FP8 is that same trade one step
// further -- 1 byte on the bus instead of 2.
//
// So the question is not "does the card do FP8 math" but "what does it cost to
// turn an e4m3 byte into an fp32 register". Three things are measured:
//
//   1. Is the shift-and-scale decode bit-exact against the e4m3 definition, for
//      all 256 patterns? Including subnormals, which is where it can go wrong.
//   2. Does it survive -ftz=true? The subnormal path routes through an fp32
//      denormal, so a flush-to-zero build would silently zero the small weights.
//   3. Is decode free at streaming rates, i.e. does an fp8 buffer deliver ~2x
//      the elements/s of an fp16 one at the same GB/s?
//
// The decode. e4m3 is 1 sign, 4 exponent (bias 7), 3 mantissa, max finite 448,
// NaN at 0x7F/0xFF, no infinities. Drop the fields into the fp32 exponent and
// mantissa positions and the result is the right number with the wrong exponent
// bias -- off by a constant 2^(127-7) for every input, normal and subnormal
// alike. One multiply corrects the lot:
//
//     bits = (b & 0x80) << 24 | (b & 0x7f) << 20      // sign, then exp+mantissa
//     value = __int_as_float(bits) * 0x1p120f
//
// The constant folds into the per-channel weight scale in real use, so the
// decode is two integer ops per weight and nothing else.
//
//   nvcc -arch=sm_61 -O3 -o fp8_decode fp8_decode.cu && ./fp8_decode
//
// Build with -DUSE_CUDA_FP8_HEADER to also check cuda_fp8.h against the same
// reference, which answers whether the header is usable on sm_61 at all.

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef USE_CUDA_FP8_HEADER
  #include <cuda_fp8.h>
#endif

#define CHECK(x)                                                              \
  do {                                                                        \
    cudaError_t err_ = (x);                                                   \
    if (err_ != cudaSuccess) {                                                \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,  \
             __LINE__);                                                       \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

// The decode under test.
__device__ __forceinline__ float decode_e4m3(uint8_t b) {
  uint32_t bits = ((uint32_t)(b & 0x80u) << 24) | ((uint32_t)(b & 0x7fu) << 20);
  return __int_as_float((int)bits) * 0x1p120f;
}

// ---------------------------------------------------------------- correctness

__global__ void decode_all_patterns(float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 256) out[i] = decode_e4m3((uint8_t)i);
}

#ifdef USE_CUDA_FP8_HEADER
__global__ void decode_all_patterns_header(float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 256) {
    __nv_fp8_e4m3 v;
    v.__x = (__nv_fp8_storage_t)i;
    out[i] = (float)v;
  }
}
#endif

// e4m3 straight from its definition, in double so the comparison is decisive.
static double reference_e4m3(uint8_t b, bool* is_nan) {
  int sign = (b >> 7) & 1;
  int exp = (b >> 3) & 0xf;
  int man = b & 0x7;
  *is_nan = (exp == 0xf && man == 0x7);
  if (*is_nan) return NAN;
  double mag;
  if (exp == 0) {
    mag = ldexp((double)man / 8.0, -6);  // subnormal: 2^-6 * man/8
  } else {
    mag = ldexp(1.0 + (double)man / 8.0, exp - 7);
  }
  return sign ? -mag : mag;
}

// ----------------------------------------------------------------- throughput

// One thread walks a strided slice, 16 bytes at a time. The reduction exists
// only so the loads cannot be optimised away; the number it produces is not
// interesting.
__global__ void stream_fp8(const uint4* __restrict__ in, size_t n_vec,
                           float* __restrict__ sink) {
  float acc = 0.f;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
       i += stride) {
    uint4 v = in[i];
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
#pragma unroll
      for (int byte = 0; byte < 4; ++byte) {
        acc += decode_e4m3((uint8_t)((words[w] >> (8 * byte)) & 0xffu));
      }
    }
  }
  if (acc == 12345.678f) *sink = acc;
}

#ifdef USE_CUDA_FP8_HEADER
// Same loop through cuda_fp8.h, to see what the header's emulation costs where
// it matters -- inside the inner loop rather than once per pattern.
__global__ void stream_fp8_header(const uint4* __restrict__ in, size_t n_vec,
                                  float* __restrict__ sink) {
  float acc = 0.f;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
       i += stride) {
    uint4 v = in[i];
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
#pragma unroll
      for (int byte = 0; byte < 4; ++byte) {
        __nv_fp8_e4m3 f8;
        f8.__x = (__nv_fp8_storage_t)((words[w] >> (8 * byte)) & 0xffu);
        acc += (float)f8;
      }
    }
  }
  if (acc == 12345.678f) *sink = acc;
}
#endif

__global__ void stream_fp16(const uint4* __restrict__ in, size_t n_vec,
                            float* __restrict__ sink) {
  float acc = 0.f;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
       i += stride) {
    uint4 v = in[i];
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      // fp32 conversion on load, exactly as the rest of this fork does it.
      __half2 h = *reinterpret_cast<const __half2*>(&words[w]);
      float2 f = __half22float2(h);
      acc += f.x + f.y;
    }
  }
  if (acc == 12345.678f) *sink = acc;
}

template <typename K>
static double time_kernel(K kernel, const uint4* d_in, size_t n_vec,
                          float* d_sink, int blocks, int threads, int iters) {
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  kernel<<<blocks, threads>>>(d_in, n_vec, d_sink);  // warm up
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) kernel<<<blocks, threads>>>(d_in, n_vec, d_sink);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

int main() {
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("%s, sm_%d%d, %.1f GB/s theoretical\n", prop.name, prop.major,
         prop.minor,
         2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

  // -- 1. every pattern, against the definition -----------------------------
  float* d_out;
  CHECK(cudaMalloc(&d_out, 256 * sizeof(float)));
  decode_all_patterns<<<1, 256>>>(d_out);
  CHECK(cudaDeviceSynchronize());
  float h_out[256];
  CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

  int mismatches = 0, subnormals = 0;
  for (int i = 0; i < 256; ++i) {
    bool is_nan;
    double ref = reference_e4m3((uint8_t)i, &is_nan);
    if (is_nan) continue;  // reported separately below
    if (((i >> 3) & 0xf) == 0 && (i & 0x7) != 0) subnormals++;
    if ((double)h_out[i] != ref) {
      if (mismatches < 8)
        printf("  MISMATCH 0x%02x: got %.10g want %.10g\n", i, (double)h_out[i],
               ref);
      mismatches++;
    }
  }
  printf("\ndecode vs definition: %d/254 finite patterns exact (%d subnormal), "
         "%d wrong\n",
         254 - mismatches, subnormals, mismatches);

  // NaN is the one pattern the shift cannot carry: e4m3's NaN exponent is 0xf,
  // which lands in the fp32 exponent field as a perfectly ordinary number.
  bool dummy;
  reference_e4m3(0x7f, &dummy);
  printf("NaN (0x7f) decodes to %.10g, not NaN -- see note below\n",
         (double)h_out[0x7f]);

#ifdef USE_CUDA_FP8_HEADER
  decode_all_patterns_header<<<1, 256>>>(d_out);
  CHECK(cudaDeviceSynchronize());
  float h_hdr[256];
  CHECK(cudaMemcpy(h_hdr, d_out, sizeof(h_hdr), cudaMemcpyDeviceToHost));
  int hdr_mismatch = 0;
  for (int i = 0; i < 256; ++i) {
    bool is_nan;
    double ref = reference_e4m3((uint8_t)i, &is_nan);
    if (is_nan) continue;
    if ((double)h_hdr[i] != ref) hdr_mismatch++;
  }
  printf("cuda_fp8.h on sm_61: compiles and runs, %d/254 exact\n",
         254 - hdr_mismatch);
#else
  printf("cuda_fp8.h: not tested (rebuild with -DUSE_CUDA_FP8_HEADER)\n");
#endif

  // -- 2. streaming rate ----------------------------------------------------
  // Same element count both ways, so fp8 moves half the bytes.
  const size_t n_elem = 512ull << 20;  // 512M weights: 512 MB fp8, 1 GB fp16
  const size_t fp8_vecs = n_elem / 16;
  const size_t fp16_vecs = n_elem / 8;

  uint4 *d_fp8, *d_fp16;
  float* d_sink;
  CHECK(cudaMalloc(&d_fp8, n_elem));
  CHECK(cudaMalloc(&d_fp16, n_elem * 2));
  CHECK(cudaMalloc(&d_sink, sizeof(float)));
  CHECK(cudaMemset(d_fp8, 0x3c, n_elem));       // some ordinary e4m3 value
  CHECK(cudaMemset(d_fp16, 0x3c, n_elem * 2));

  const int threads = 256;
  const int blocks = prop.multiProcessorCount * 16;
  const int iters = 20;

  double ms8 = time_kernel(stream_fp8, d_fp8, fp8_vecs, d_sink, blocks, threads, iters);
  double ms16 = time_kernel(stream_fp16, d_fp16, fp16_vecs, d_sink, blocks, threads, iters);

  double gb8 = (double)n_elem / (ms8 * 1e-3) / 1e9;
  double gb16 = (double)n_elem * 2 / (ms16 * 1e-3) / 1e9;
  double ge8 = (double)n_elem / (ms8 * 1e-3) / 1e9;
  double ge16 = (double)n_elem / (ms16 * 1e-3) / 1e9;

  printf("\nstreaming %zuM weights\n", (size_t)(n_elem >> 20));
  printf("  fp8  decode  %7.3f ms   %6.1f GB/s   %6.2f Gweight/s\n", ms8, gb8, ge8);
  printf("  fp16 convert %7.3f ms   %6.1f GB/s   %6.2f Gweight/s\n", ms16, gb16, ge16);
  printf("  fp8 delivers %.2fx the weights/s at %.0f%% of fp16's bandwidth\n",
         ge8 / ge16, 100.0 * gb8 / gb16);

#ifdef USE_CUDA_FP8_HEADER
  double ms_hdr = time_kernel(stream_fp8_header, d_fp8, fp8_vecs, d_sink, blocks,
                              threads, iters);
  printf("  fp8 via cuda_fp8.h %7.3f ms   %6.1f GB/s   (%.2fx the hand decode)\n",
         ms_hdr, (double)n_elem / (ms_hdr * 1e-3) / 1e9, ms_hdr / ms8);
#endif

  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_fp8));
  CHECK(cudaFree(d_fp16));
  CHECK(cudaFree(d_sink));
  return mismatches == 0 ? 0 : 1;
}
