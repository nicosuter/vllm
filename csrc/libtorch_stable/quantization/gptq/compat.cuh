/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _compat_cuh
#define _compat_cuh

namespace vllm {
namespace gptq {
// atomicAdd for half types, to support CC < 7.x

__device__ __forceinline__ void atomicAdd_half(half* address, half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    __half_raw hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    half tmpres = __hadd(hsum, val);
    hsum = __half_raw(tmpres);
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}

// atomicAdd for half2 types

__device__ __forceinline__ void atomicAdd_half2(half2* address, half2 val) {
  unsigned int* address_as_ui = (unsigned int*)address;
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    half2 old_val = *((half2*)&old);
    half2 new_val = __hadd2(old_val, val);
    old = atomicCAS(address_as_ui, assumed, *((unsigned int*)&new_val));
  } while (assumed != old);
}

//
#if defined(__CUDA_ARCH__) || \
    (defined(USE_ROCM) && (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR) < 713)
  #if __CUDA_ARCH__ < 700 || defined(USE_ROCM)

__device__ __forceinline__ void atomicAdd(half* address, half val) {
  atomicAdd_half(address, val);
}

    #if __CUDA_ARCH__ < 600 || defined(USE_ROCM)
__device__ __forceinline__ void atomicAdd(half2* address, half2 val) {
  atomicAdd_half2(address, val);
}
    #endif

  #endif
#endif

// GP102/104/106/107 (sm_61) and GP10B (sm_62) have a single FP16x2 unit per SM.
// pascal/probes/fp16_rate.cu measures __hfma2 on a GTX 1070 Ti at 139.9 GFLOP/s
// against 7844.7 GFLOP/s for fmaf -- a ratio of 1/56 -- so native half2 is the
// slowest available way to do arithmetic on this hardware.
//
// GP100 (sm_60) is deliberately excluded: it has genuine 2x fp16 throughput and
// wants the half2 paths. This is why the guard names the two architectures
// rather than testing __CUDA_ARCH__ < 700.
//
// Defined here because both q_gemm.cu and qdq_4.cuh need it, and compat.cuh is
// included before qdq_4.cuh by every translation unit that uses either.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 610 || __CUDA_ARCH__ == 620)
  #define VLLM_GPTQ_SLOW_NATIVE_FP16 1
#endif

}  // namespace gptq
}  // namespace vllm
#endif
