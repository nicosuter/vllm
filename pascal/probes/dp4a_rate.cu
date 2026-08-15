// Measures INT8 dp4a throughput against fp32 FFMA on the target card.
//
// dp4a is the one piece of tensor-core-like throughput SM 6.1 has: a single
// instruction taking two packed int8x4 operands and accumulating their dot
// product into int32. Decode on this fork is memory-bound and cannot use it,
// but prefill and batched serving are compute-bound, and there dp4a is the only
// way to raise the roof above fp32 FFMA.
//
// Whether it is worth writing a W4A8 GEMM depends entirely on the ratio, which
// is quoted from spec sheets far more often than it is measured. NVIDIA's
// figure implies about 4x fp32 for GP104. This checks.
//
// Both loops are serially dependent through their accumulator, so what is
// measured is instruction throughput rather than memory or ILP. Results are
// written out under a branch that never fires, to defeat dead-code elimination.
//
//   nvcc -arch=sm_61 -O3 -o dp4a_rate dp4a_rate.cu && ./dp4a_rate

#include <cstdio>
#include <cuda_runtime.h>

#define ITERS 100000

__global__ void dp4a_bench(int* out, int a, int b) {
  int acc = 0;
#pragma unroll 16
  for (int i = 0; i < ITERS; i++) {
    acc = __dp4a(a, b, acc);
  }
  if (threadIdx.x == 0xFFFF) *out = acc;
}

__global__ void ffma_bench(float* out, float a, float b) {
  float acc = 0.0f;
#pragma unroll 16
  for (int i = 0; i < ITERS; i++) {
    acc = fmaf(a, b, acc);
  }
  if (threadIdx.x == 0xFFFF) *out = acc;
}

// int32 IMAD, for context: it is what a hand-rolled int8 path would fall back
// to if dp4a were unavailable, so it separates "int8 is fast" from "dp4a is
// fast".
__global__ void imad_bench(int* out, int a, int b) {
  int acc = 1;
#pragma unroll 16
  for (int i = 0; i < ITERS; i++) {
    // The accumulator must be an *operand* of the multiply, not just the
    // addend. With `a * b + acc` the product is loop-invariant, nvcc folds it
    // to a constant and strength-reduces the whole loop to a single multiply,
    // which reports absurd throughput rather than IMAD's.
    acc = acc * b + a;
  }
  if (threadIdx.x == 0xFFFF) *out = acc;
}

template <typename F>
static float time_kernel(F launch, int reps = 5) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  launch();
  cudaDeviceSynchronize();
  float best = 1e30f;
  for (int r = 0; r < reps; r++) {
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
  printf("device: %s (sm_%d%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);

  const int blocks = prop.multiProcessorCount * 8;
  const int threads = 256;
  const double lanes = double(blocks) * threads;

  int *d_i, *d_i2;
  float* d_f;
  cudaMalloc(&d_i, sizeof(int));
  cudaMalloc(&d_i2, sizeof(int));
  cudaMalloc(&d_f, sizeof(float));

  // 0x01010101 = four int8 ones; the values are irrelevant to timing.
  const int a = 0x01010101, b = 0x02020202;

  float ms_d = time_kernel([&] { dp4a_bench<<<blocks, threads>>>(d_i, a, b); });
  float ms_f = time_kernel([&] { ffma_bench<<<blocks, threads>>>(d_f, 1.0001f, 0.9999f); });
  float ms_i = time_kernel([&] { imad_bench<<<blocks, threads>>>(d_i2, 3, 5); });

  // dp4a does 4 multiply-accumulates per instruction = 8 ops; fmaf and imad do
  // 1 each = 2 ops. Counting this way makes the numbers comparable as useful
  // arithmetic rather than as instructions retired.
  const double dp4a_ops = lanes * ITERS * 8.0;
  const double scalar_ops = lanes * ITERS * 2.0;

  const double g_d = dp4a_ops / (ms_d * 1e6);
  const double g_f = scalar_ops / (ms_f * 1e6);
  const double g_i = scalar_ops / (ms_i * 1e6);

  printf("  INT8 __dp4a : %8.2f ms  %8.1f GOP/s\n", ms_d, g_d);
  printf("  fp32 fmaf   : %8.2f ms  %8.1f GFLOP/s\n", ms_f, g_f);
  printf("  int32 imad  : %8.2f ms  %8.1f GOP/s\n", ms_i, g_i);
  printf("\n  dp4a / fp32 : %.2fx\n", g_d / g_f);
  printf("  dp4a / imad : %.2fx\n", g_d / g_i);
  printf(
      "\n  This ratio is the ceiling a W4A8 kernel could raise prefill and\n"
      "  batched decode to. Batch-1 decode is memory-bound and cannot use it.\n");

  cudaFree(d_i);
  cudaFree(d_i2);
  cudaFree(d_f);
  return 0;
}
