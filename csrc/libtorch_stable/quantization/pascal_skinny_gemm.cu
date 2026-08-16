// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// A small-M dense fp16 GEMM for sm_61.
//
// y[M, N] = x[M, K] @ w[N, K]^T, for the decode shapes of a dense fp16 model.
// Decode is a bandwidth problem -- the step reads every weight exactly once --
// so the only question is what fraction of the card's bandwidth the GEMM
// reaches. cuBLAS reaches it at M=1 and then falls off a cliff:
//
//   Qwen3-VL-Embedding-2B shapes on a GTX 1070 Ti, microseconds per call,
//   against a measured 212 GB/s achievable (pascal/probes/fp16_linear_rate.py).
//   `floor` is the weight bytes divided by that bandwidth:
//
//     shape                 floor    M=1    M=2    M=4    M=8   M=16   M=32
//     qkv     2048->4096       79     82    186    188    191    195    129
//     o       2048->2048       40     43     93     94     98    102     82
//     gate_up 2048->12288     237    234    552    552    555    555    310
//     down    6144->2048      119    128    486    482    490    500    315
//     lm_head 2048->151936   2933   2832   6000   6021   6038   6141  11020
//
// At M=1 cuBLAS picks a gemv and lands on the roofline. At M=2 it switches to a
// tensor-core-shaped tile that this card cannot feed and loses 2.2-4.1x, and it
// stays there through M=16. On a dense fp16 model that cliff *is* the batching
// curve: batch 4 measured 2.5x the step time of batch 1 for 4x the sequences.
// `lm_head` never recovers -- at M=32 it is 3.8x off its floor.
//
// This kernel, same shapes, same units:
//
//     qkv                       78     78     80     89    128    240
//     o                         40     41     42     48     72    130
//     gate_up                  228    229    231    246    349    682
//     down                     116    117    119    135    185    339
//     lm_head                 2792   2799   2813   2892   3899   8757
//
// Three things make the difference, all of them sm_61 facts:
//
//   1. fp16 is a storage format only. __hfma2 runs at 1/56 the fp32 rate on
//      GP104 (pascal/probes/fp16_rate.cu), so every value converts to float on
//      the way into the FMA and the accumulator is float. Same finding that
//      gave the exllama kernel its 1.73x and the int4 dequantize its 4.03x.
//
//   2. The weight crosses the memory bus exactly once. All M columns stay live
//      while a weight tile sits in registers, which is why this is one kernel
//      and not M gemv calls -- those would read the weight M times and lose to
//      cuBLAS outright.
//
//   3. The FMA nest is ordered for independent accumulator chains, not for
//      locality. An FMA takes ~6 cycles to land, and `acc` is both an operand
//      and the result, so the innermost loop has to walk at least six distinct
//      accumulators before it returns to any one of them. Two earlier revisions
//      got this wrong in different ways and both sat at 8-15% of FMA peak
//      regardless of shape, tiling or occupancy.
//
// Layout: one warp owns R output rows and all M columns. Lane L reads eight
// contiguous halves (one float4) at k = L*8, striding by 256, so the 32 lanes of
// a warp cover 512 contiguous bytes of one weight row per instruction. The
// weight tile converts to float once per k-chunk and is reused across every
// column of x.
//
// R and MT are the two knobs, and they trade against the register file:
//
//   R  output rows per warp. Also the count of accumulators that share a staged
//      weight, so raising it cuts how often each warp re-reads x.
//   MT columns of x staged in registers at once. R*MT is the number of
//      independent chains the inner loop walks, which is what has to cover FMA
//      latency; R*M_MAX + 8*R + 8*MT is roughly what has to fit in registers.
//
// The pairs below are measured, not derived -- see the tables in
// pascal/probes/skinny_gemm_sweep.py. The pattern they follow is that small M
// wants MT=M and R=1 (chains come free, registers stay tiny) while M=16 needs
// R=4 MT=4 to reach 16 chains without 128 accumulators.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kKPerLane = 8;  // one float4 of halves

template <int R, int M_MAX, int MT>
__global__ __launch_bounds__(kWarp* kWarpsPerBlock) void pascal_skinny_gemm_kernel(
    const __half* __restrict__ X, const __half* __restrict__ W,
    __half* __restrict__ Y, int M, int N, int K) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x >> 5;
  const int n0 = (blockIdx.x * kWarpsPerBlock + warp) * R;
  if (n0 >= N) return;

  float acc[R][M_MAX];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) acc[r][m] = 0.0f;

  for (int k = lane * kKPerLane; k < K; k += kWarp * kKPerLane) {
    float wf[R][kKPerLane];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = n0 + r;
      float4 raw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (n < N) {
        raw = *reinterpret_cast<const float4*>(W + static_cast<size_t>(n) * K + k);
      }
      const __half* h = reinterpret_cast<const __half*>(&raw);
#pragma unroll
      for (int j = 0; j < kKPerLane; ++j) wf[r][j] = __half2float(h[j]);
    }

#pragma unroll
    for (int mt = 0; mt < M_MAX; mt += MT) {
      if (mt >= M) break;
      float xf[MT][kKPerLane];
#pragma unroll
      for (int mm = 0; mm < MT; ++mm) {
        const int m = mt + mm;
        if (m >= M) {
          // Past the end of the batch: contributes zero and is never stored.
#pragma unroll
          for (int j = 0; j < kKPerLane; ++j) xf[mm][j] = 0.0f;
          continue;
        }
        const float4 raw =
            *reinterpret_cast<const float4*>(X + static_cast<size_t>(m) * K + k);
        const __half* h = reinterpret_cast<const __half*>(&raw);
#pragma unroll
        for (int j = 0; j < kKPerLane; ++j) xf[mm][j] = __half2float(h[j]);
      }

      // k outermost: the innermost pair walks R*MT distinct accumulators before
      // returning to any one of them.
#pragma unroll
      for (int j = 0; j < kKPerLane; ++j)
#pragma unroll
        for (int mm = 0; mm < MT; ++mm)
#pragma unroll
          for (int r = 0; r < R; ++r)
            acc[r][mt + mm] = fmaf(wf[r][j], xf[mm][j], acc[r][mt + mm]);
    }
  }

#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) {
      if (m >= M) break;
#pragma unroll
      for (int off = kWarp >> 1; off > 0; off >>= 1)
        acc[r][m] += __shfl_down_sync(0xffffffffu, acc[r][m], off);
    }

  if (lane != 0) return;
#pragma unroll
  for (int m = 0; m < M_MAX; ++m) {
    if (m >= M) break;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = n0 + r;
      if (n < N) Y[static_cast<size_t>(m) * N + n] = __float2half(acc[r][m]);
    }
  }
}

template <int R, int M_MAX, int MT>
void launch(const __half* x, const __half* w, __half* y, int M, int N, int K,
            cudaStream_t stream) {
  const int rowsPerBlock = kWarpsPerBlock * R;
  pascal_skinny_gemm_kernel<R, M_MAX, MT>
      <<<dim3((N + rowsPerBlock - 1) / rowsPerBlock), dim3(kWarp * kWarpsPerBlock), 0,
         stream>>>(x, w, y, M, N, K);
}

}  // namespace

// out[M, N] = x[M, K] @ weight[N, K]^T, fp16 throughout, fp32 accumulate.
void pascal_skinny_gemm(torch::stable::Tensor& out,
                        torch::stable::Tensor const& x,
                        torch::stable::Tensor const& weight) {
  STD_TORCH_CHECK(out.dim() == 2 && x.dim() == 2 && weight.dim() == 2,
                  "pascal_skinny_gemm: all tensors must be 2-D");
  STD_TORCH_CHECK(out.is_cuda() && x.is_cuda() && weight.is_cuda(),
                  "pascal_skinny_gemm: all tensors must be CUDA tensors");
  STD_TORCH_CHECK(out.is_contiguous() && x.is_contiguous() && weight.is_contiguous(),
                  "pascal_skinny_gemm: all tensors must be contiguous");
  STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Half &&
                      weight.scalar_type() == torch::headeronly::ScalarType::Half &&
                      out.scalar_type() == torch::headeronly::ScalarType::Half,
                  "pascal_skinny_gemm: fp16 only");

  const int M = x.size(0);
  const int K = x.size(1);
  const int N = weight.size(0);
  STD_TORCH_CHECK(weight.size(1) == K, "pascal_skinny_gemm: K mismatch");
  STD_TORCH_CHECK(out.size(0) == M && out.size(1) == N,
                  "pascal_skinny_gemm: out must be [M, N]");
  STD_TORCH_CHECK(K % kKPerLane == 0,
                  "pascal_skinny_gemm: K must be a multiple of 8");
  STD_TORCH_CHECK(M >= 1 && M <= 32, "pascal_skinny_gemm: M must be in [1, 32]");

  if (N == 0) return;

  const torch::stable::accelerator::DeviceGuard device_guard(x.get_device_index());
  auto stream = get_current_cuda_stream(x.get_device_index());

  const auto* xp = reinterpret_cast<const __half*>(x.const_data_ptr());
  const auto* wp = reinterpret_cast<const __half*>(weight.const_data_ptr());
  auto* yp = reinterpret_cast<__half*>(out.mutable_data_ptr());

  if (M <= 1) {
    launch<4, 1, 1>(xp, wp, yp, M, N, K, stream);
  } else if (M <= 2) {
    launch<1, 2, 2>(xp, wp, yp, M, N, K, stream);
  } else if (M <= 4) {
    launch<1, 4, 4>(xp, wp, yp, M, N, K, stream);
  } else if (M <= 8) {
    launch<2, 8, 2>(xp, wp, yp, M, N, K, stream);
  } else if (M <= 16) {
    launch<4, 16, 4>(xp, wp, yp, M, N, K, stream);
  } else {
    launch<2, 32, 8>(xp, wp, yp, M, N, K, stream);
  }
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("pascal_skinny_gemm", TORCH_BOX(&pascal_skinny_gemm));
}
