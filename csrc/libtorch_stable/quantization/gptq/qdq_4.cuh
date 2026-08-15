/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _qdq_4_cuh
#define _qdq_4_cuh

#include "qdq_util.cuh"

namespace vllm {
namespace gptq {
// Permutation:
//
// 77775555 33331111  66664444 22220000

__forceinline__ __device__ void shuffle_4bit_8(uint32_t* q, int stride) {
  uint32_t qa = q[0];
  uint32_t qb = 0;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t qa0 = qa & 0x0f;
    uint32_t qa1 = (qa & 0xf0) >> 4;
    qa >>= 8;
    qb |= (qa1 << (i * 4 + 16));
    qb |= (qa0 << (i * 4));
  }
  q[0] = qb;
}

__forceinline__ __device__ void dequant_4bit_8(const uint32_t q_0,
                                               half2 (&dq)[4], int stride,
                                               const uint32_t zero) {
  const uint32_t c0 = 0x64006400;
  const half y16_ = __float2half_rn(1.0f / 16.0f);
  const half2 y16 = __halves2half2(y16_, y16_);
  const half_uint16 z1_(0xe400 | zero);  // half(-1024.0f - zero);
  const half z16_ = __hsub(__int2half_rn(-64), __int2half_rn(zero));
  const half2 z1 = __half2half2(z1_.as_half);
  const half2 z16 = __half2half2(z16_);

  uint32_t qa = q_0;
  half2_uint32 q0((qa & 0x000f000f) | c0);  // half2(q[ 0], q[ 1])      + 1024
  half2_uint32 q1((qa & 0x00f000f0) | c0);  // half2(q[ 2], q[ 3]) * 16 + 1024
  qa >>= 8;
  half2_uint32 q2((qa & 0x000f000f) | c0);  // half2(q[ 4], q[ 5])      + 1024
  half2_uint32 q3((qa & 0x00f000f0) | c0);  // half2(q[ 6], q[ 7]) * 16 + 1024

  dq[0] = __hadd2(q0.as_half2, z1);
  dq[1] = __hfma2(q1.as_half2, y16, z16);
  dq[2] = __hadd2(q2.as_half2, z1);
  dq[3] = __hfma2(q3.as_half2, y16, z16);
}

__forceinline__ __device__ void dequant_4bit_8_prep_zero_scale(
    const uint32_t zero, const half scale, half2 (&z1z16)[2],
    half2 (&y1y16)[2]) {
  half_uint16 z1(0xe400 | zero);  // half(-1024.0f - zero);
  half z16 = __hsub(__int2half_rn(-64), __int2half_rn(zero));

  half2 scale2 = __half2half2(scale);

  z1z16[0] = __hmul2(scale2, __half2half2(z1.as_half));
  z1z16[1] = __hmul2(scale2, __half2half2(z16));

  const half y1 = __float2half_rn(1.0f);
  const half y16 = __float2half_rn(1.0f / 16.0f);

  y1y16[0] = __hmul2(scale2, __half2half2(y1));
  y1y16[1] = __hmul2(scale2, __half2half2(y16));
}

__forceinline__ __device__ void dequant_4bit_8_prep_zero(const uint32_t zero,
                                                         half2 (&z1z16)[2],
                                                         half2 (&y1y16)[2]) {
  half_uint16 z1(0xe400 | zero);  // half(-1024.0f - zero);
  half z16 = __hsub(__int2half_rn(-64), __int2half_rn(zero));

  z1z16[0] = __half2half2(z1.as_half);
  z1z16[1] = __half2half2(z16);

  const half y1 = __float2half_rn(1.0f);
  const half y16 = __float2half_rn(1.0f / 16.0f);

  y1y16[0] = __half2half2(y1);
  y1y16[1] = __half2half2(y16);
}

__forceinline__ __device__ void dequant_4bit_8_gptq(const uint32_t q_0,
                                                    half2 (&dq)[4],
                                                    half2 (&z1z16)[2],
                                                    half2 (&y1y16)[2],
                                                    int stride, bool scaled) {
  const uint32_t c0 = 0x64006400;

  uint32_t qa = q_0;
  half2_uint32 q0((qa & 0x000f000f) |
                  c0);  // half2( q[0]      + 1024, q[1]      + 1024 )
  half2_uint32 q1((qa & 0x00f000f0) |
                  c0);  // half2( q[2] * 16 + 1024, q[3] * 16 + 1024 )
  qa >>= 8;
  half2_uint32 q2((qa & 0x000f000f) |
                  c0);  // half2( q[4]      + 1024, q[5]      + 1024 )
  half2_uint32 q3((qa & 0x00f000f0) |
                  c0);  // half2( q[6] * 16 + 1024, q[7] * 16 + 1024 )

  if (scaled) {
    dq[0] = __hfma2(q0.as_half2, y1y16[0],
                    z1z16[0]);  // half2( q[0] * s - z * s, q[1] * s - z * s)
    dq[1] = __hfma2(q1.as_half2, y1y16[1],
                    z1z16[1]);  // half2( q[2] * s - z * s, q[3] * s - z * s)
    dq[2] = __hfma2(q2.as_half2, y1y16[0], z1z16[0]);
    dq[3] = __hfma2(q3.as_half2, y1y16[1], z1z16[1]);
  } else {
    dq[0] = __hadd2(q0.as_half2, z1z16[0]);  // half2( q[0] - z, q[1] - z )
    dq[1] = __hfma2(q1.as_half2, y1y16[1],
                    z1z16[1]);               // half2( q[2] - z, q[3] - z )
    dq[2] = __hadd2(q2.as_half2, z1z16[0]);  // half2( q[4] - z, q[5] - z )
    dq[3] = __hfma2(q3.as_half2, y1y16[1],
                    z1z16[1]);  // half2( q[6] - z, q[7] - z )
  }
}
#ifdef VLLM_GPTQ_SLOW_NATIVE_FP16

// fp32 counterpart of dequant_4bit_8_gptq's unscaled path, for architectures
// where __hfma2 runs at 1/56 the fp32 rate. The four half2 ops this replaces are
// the dominant arithmetic left in gemm_half_q_half_gptq_4bit_kernel once
// dot22_8_f accumulates in fp32.
//
// The bit trick is kept, because it is free: ORing a 4-bit nibble into the
// mantissa of the half constant 1024.0 (0x6400) produces half(q + 1024) with no
// conversion instruction. What changes is only what happens afterwards --
// __half22float2 lowers to a full-rate cvt pair, and the affine step becomes
// FFMA instead of HFMA2.
//
// Algebraically identical to the unscaled branch above:
//   lanes 0,1,4,5 hold q + 1024        -> q - zero        = f + (-1024 - zero)
//   lanes 2,3,6,7 hold q * 16 + 1024   -> q - zero        = f/16 + (-64 - zero)
// which is the same pair of constants (z1, z16) that
// dequant_4bit_8_prep_zero builds, just in fp32 and without the half2 packing.
__forceinline__ __device__ void dequant_4bit_8_gptq_f(const uint32_t q_0,
                                                      float (&dq)[8],
                                                      const float zero) {
  const uint32_t c0 = 0x64006400;

  uint32_t qa = q_0;
  half2_uint32 q0((qa & 0x000f000f) | c0);  // half2( q[0] + 1024, q[1] + 1024 )
  half2_uint32 q1((qa & 0x00f000f0) | c0);  // half2( q[2]*16 + 1024, q[3]*.. )
  qa >>= 8;
  half2_uint32 q2((qa & 0x000f000f) | c0);  // half2( q[4] + 1024, q[5] + 1024 )
  half2_uint32 q3((qa & 0x00f000f0) | c0);  // half2( q[6]*16 + 1024, q[7]*.. )

  const float2 f0 = __half22float2(q0.as_half2);
  const float2 f1 = __half22float2(q1.as_half2);
  const float2 f2 = __half22float2(q2.as_half2);
  const float2 f3 = __half22float2(q3.as_half2);

  const float b1 = -1024.0f - zero;
  const float b16 = -64.0f - zero;

  dq[0] = f0.x + b1;
  dq[1] = f0.y + b1;
  dq[2] = fmaf(f1.x, 0.0625f, b16);
  dq[3] = fmaf(f1.y, 0.0625f, b16);
  dq[4] = f2.x + b1;
  dq[5] = f2.y + b1;
  dq[6] = fmaf(f3.x, 0.0625f, b16);
  dq[7] = fmaf(f3.y, 0.0625f, b16);
}

#endif  // VLLM_GPTQ_SLOW_NATIVE_FP16

}  // namespace gptq
}  // namespace vllm

#endif
