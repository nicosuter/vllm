// Measures native fp16 (HFMA2) throughput against fp32 (FFMA) on the target
// card.
//
// This exists because the fork's hottest arithmetic -- exllama's dot22_8_f,
// which every quantized linear layer runs -- accumulates with __hfma2, and the
// decision to rewrite it in fp32 rests entirely on GP10x fp16 being slow. That
// is a spec-sheet claim (one FP16x2 unit per SM on GP102/104/106, versus real
// 2x throughput on GP100), and it is worth confirming on the actual silicon
// before rewriting a kernel on the strength of it.
//
// Both loops are serially dependent so the measurement is instruction
// throughput and not memory or ILP. The accumulator is written out to stop the
// compiler deleting the loop.
//
//   nvcc -arch=sm_61 -O3 -o fp16_rate fp16_rate.cu && ./fp16_rate

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define ITERS 100000

__global__ void hfma2_bench(half2* out, half2 a, half2 b) {
  half2 acc = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll 16
  for (int i = 0; i < ITERS; i++) {
    acc = __hfma2(a, b, acc);
  }
  if (threadIdx.x == 0xFFFF) *out = acc;  // never taken; defeats DCE
}

__global__ void ffma_bench(float2* out, float2 a, float2 b) {
  float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll 16
  for (int i = 0; i < ITERS; i++) {
    acc.x = fmaf(a.x, b.x, acc.x);
    acc.y = fmaf(a.y, b.y, acc.y);
  }
  if (threadIdx.x == 0xFFFF) *out = acc;
}

template <typename F>
float time_kernel(F launch, int reps = 5) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  launch();  // warm up: first launch pays module load
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
  printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);

  // Enough blocks to fill every SM several times over.
  const int blocks = prop.multiProcessorCount * 8;
  const int threads = 256;
  const double total_threads = double(blocks) * threads;

  half2* d_h2;
  float2* d_f2;
  cudaMalloc(&d_h2, sizeof(half2));
  cudaMalloc(&d_f2, sizeof(float2));

  half2 ha = __floats2half2_rn(1.0001f, 0.9999f);
  half2 hb = __floats2half2_rn(0.9999f, 1.0001f);
  float2 fa = make_float2(1.0001f, 0.9999f);
  float2 fb = make_float2(0.9999f, 1.0001f);

  float ms_h = time_kernel(
      [&] { hfma2_bench<<<blocks, threads>>>(d_h2, ha, hb); });
  float ms_f = time_kernel(
      [&] { ffma_bench<<<blocks, threads>>>(d_f2, fa, fb); });

  // Each iteration is 2 FMAs (2 lanes) = 4 flops, either way.
  const double flops = total_threads * ITERS * 4.0;
  const double gf_h = flops / (ms_h * 1e6);
  const double gf_f = flops / (ms_f * 1e6);

  printf("\n  fp16 (__hfma2) : %8.2f ms  %8.1f GFLOP/s\n", ms_h, gf_h);
  printf("  fp32 (fmaf)    : %8.2f ms  %8.1f GFLOP/s\n", ms_f, gf_f);
  printf("\n  fp16/fp32 throughput ratio: %.4f  (1/%.1f)\n", gf_h / gf_f,
         gf_f / gf_h);
  printf(
      "\n  If the ratio is well below 1, converting exllama's dot22_8_f to\n"
      "  fp32 should speed up every quantized linear layer by roughly the\n"
      "  reciprocal, minus the two cvt instructions per half2.\n");

  cudaFree(d_h2);
  cudaFree(d_f2);
  return 0;
}
