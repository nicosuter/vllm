// A small-M fp16 GEMM for sm_61, in two layouts.
//
// y[M, N] = x[M, K] @ w^T, with M small (decode) and the weight far larger than
// the activation. cuBLAS reaches the memory roofline here at M=1 and loses
// 2.2-4.1x at M=2..16 (pascal/probes/fp16_linear_rate.py), which on a dense fp16
// model is most of the decode step.
//
// Two sm_61 facts shape both kernels:
//
//   1. fp16 is a storage format only. __hfma2 runs at 1/56 the fp32 rate on
//      GP104, so every value is converted to float on the way into the FMA and
//      the accumulator is float. Same finding as the exllama kernel's 1.73x and
//      the int4 dequantize's 4.03x.
//
//   2. The weight must cross the memory bus exactly once, which forces all M
//      columns to be live while a weight tile is in registers. That is why this
//      is one kernel and not M gemv calls.
//
// What separates the two kernels is which operand is broadcast.
//
// `nk` takes vLLM's native [N, K] weight. A warp owns R output rows and strides
// k, so the activation is indexed by the same k the weight is -- every warp must
// re-read the whole of x. Total activation traffic is (N/R) * M * K, which at
// M=16 is 268 MB against a 16.8 MB weight and pins the kernel at ~1.1 TB/s of L2.
// Measured: 96-104% of the bandwidth floor at M<=4, falling to 34% at M=16.
//
// `kn` takes the weight transposed to [K, N]. A thread owns two adjacent output
// columns and strides k, so x[m][k] is the *same address for every thread* -- a
// broadcast, served once per block from L2. Activation traffic drops to
// (N / columns-per-block) * M * K, about 1 MB, and the kernel stops caring how
// large M is. The transpose is done once at load time and replaces the original,
// so it costs no extra memory.
//
// The [K, N] layout is also why cuBLAS's own `torch.mm(x, w.t().contiguous())`
// beats `torch.nn.functional.linear(x, w)` by 1.3-2.5x on this card: same
// mechanism, in a kernel that was tuned for hardware this old.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace {

constexpr int kWarp = 32;
constexpr int kKPerLane = 8;  // one float4 of halves

// ---------------------------------------------------------------- [N, K] ----

constexpr int kNkWarpsPerBlock = 4;

// MT can never exceed M_MAX, or the staged tile runs off the accumulator array.
#define MT_CLAMP(m_max, mt) ((mt) > (m_max) ? (m_max) : (mt))

// MT is how many columns of x are staged in registers at once. It sets how many
// *independent* accumulator chains the innermost loop walks -- MT*R of them --
// which is the number that has to cover the ~6-cycle FMA latency. At M<=4 that
// came free from R=8. At M=16 the accumulator budget forces R down to 2, and
// with MT=1 the inner loop alternates between two chains and stalls four cycles
// in six: measured 0.63 TFMA/s, 15% of this card's peak, at every R and every
// shape. Staging four columns instead of one puts it back to eight chains for
// 32 more registers.
template <int R, int M_MAX, int MT, typename XT>
__global__ __launch_bounds__(kWarp* kNkWarpsPerBlock) void skinny_gemm_nk_kernel(
    const XT* __restrict__ X, const __half* __restrict__ W,
    __half* __restrict__ Y, int M, int N, int K) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x >> 5;
  const int n0 = (blockIdx.x * kNkWarpsPerBlock + warp) * R;
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
#pragma unroll
          for (int j = 0; j < kKPerLane; ++j) xf[mm][j] = 0.0f;
          continue;
        }
        const XT* xp = X + static_cast<size_t>(m) * K + k;
        if constexpr (sizeof(XT) == 4) {
          const float4 xa = *reinterpret_cast<const float4*>(xp);
          const float4 xb = *reinterpret_cast<const float4*>(xp + 4);
          const float tmp[kKPerLane] = {xa.x, xa.y, xa.z, xa.w,
                                        xb.x, xb.y, xb.z, xb.w};
#pragma unroll
          for (int j = 0; j < kKPerLane; ++j) xf[mm][j] = tmp[j];
        } else {
          const float4 raw = *reinterpret_cast<const float4*>(xp);
          const __half* h = reinterpret_cast<const __half*>(&raw);
#pragma unroll
          for (int j = 0; j < kKPerLane; ++j) xf[mm][j] = __half2float(h[j]);
        }
      }
      // j outermost, so the innermost loop walks MT*R distinct accumulators
      // before returning to any one of them.
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

// ---------------------------------------------------------------- [K, N] ----

constexpr int kKnWarps = 4;          // warps per block, one k-slice each
constexpr int kKnColsPerThread = 2;  // one half2 load
constexpr int kKnColsPerWarp = kWarp * kKnColsPerThread;
constexpr int kKnUnroll = 4;         // one float4 of activation

// The n dimension alone cannot fill this GPU in a k-major layout: at two
// columns per thread, N=2048 is 1024 threads against the ~39,000 the card wants
// resident. The first version of this kernel ignored that and ran 4 blocks over
// 19 SMs, which cost it 8x. So k is split across blocks as well, and the partial
// sums meet in an fp32 accumulator through atomicAdd. When one slice is enough
// -- large N, e.g. lm_head -- the accumulator is skipped and the kernel writes
// fp16 directly.
template <int M_MAX, bool SPLIT_K>
__global__ __launch_bounds__(kWarp* kKnWarps) void skinny_gemm_kn_kernel(
    const float* __restrict__ X, const __half* __restrict__ W, void* __restrict__ Yv,
    int M, int N, int K, int kSlices) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x >> 5;
  const int n = blockIdx.x * kKnColsPerWarp + lane * kKnColsPerThread;
  const int slice = blockIdx.y * kKnWarps + warp;

  const int groups = K / kKnUnroll;
  const int perSlice = (groups + kSlices - 1) / kSlices;
  const int kBeg = slice * perSlice * kKnUnroll;
  const int kEnd = min(kBeg + perSlice * kKnUnroll, K);
  if (kBeg >= kEnd || n >= N) return;

  float acc[kKnColsPerThread][M_MAX];
#pragma unroll
  for (int c = 0; c < kKnColsPerThread; ++c)
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) acc[c][m] = 0.0f;

  const bool full = (n + kKnColsPerThread) <= N;
  for (int k = kBeg; k < kEnd; k += kKnUnroll) {
    // Converted once per k-group and reused across every column of x.
    float wf[kKnUnroll][kKnColsPerThread];
#pragma unroll
    for (int u = 0; u < kKnUnroll; ++u) {
      const __half* row = W + static_cast<size_t>(k + u) * N;
      if (full) {
        const float2 pair = __half22float2(*reinterpret_cast<const __half2*>(row + n));
        wf[u][0] = pair.x;
        wf[u][1] = pair.y;
      } else {
#pragma unroll
        for (int c = 0; c < kKnColsPerThread; ++c)
          wf[u][c] = (n + c) < N ? __half2float(row[n + c]) : 0.0f;
      }
    }

    // Every thread in the block reads these same addresses, so each costs one
    // broadcast rather than 128 loads.
    float xf[M_MAX][kKnUnroll];
#pragma unroll
    for (int m = 0; m < M_MAX; ++m) {
      if (m >= M) break;
      const float4 xv =
          *reinterpret_cast<const float4*>(X + static_cast<size_t>(m) * K + k);
      xf[m][0] = xv.x;
      xf[m][1] = xv.y;
      xf[m][2] = xv.z;
      xf[m][3] = xv.w;
    }

    // k innermost would make every FMA wait on the one before it: acc[c][m] is
    // both operand and result, and an FMA takes ~6 cycles to land. With k
    // outermost the innermost loop walks 2*M *distinct* accumulators before it
    // comes back to any of them, which is enough independent chains to keep the
    // pipe full. The first version of this kernel had the loops the other way
    // round and ran at 7.6% of FMA peak because of it.
#pragma unroll
    for (int u = 0; u < kKnUnroll; ++u)
#pragma unroll
      for (int m = 0; m < M_MAX; ++m) {
        if (m >= M) break;
#pragma unroll
        for (int c = 0; c < kKnColsPerThread; ++c)
          acc[c][m] = fmaf(wf[u][c], xf[m][u], acc[c][m]);
      }
  }

#pragma unroll
  for (int m = 0; m < M_MAX; ++m) {
    if (m >= M) break;
#pragma unroll
    for (int c = 0; c < kKnColsPerThread; ++c) {
      if ((n + c) >= N) continue;
      const size_t idx = static_cast<size_t>(m) * N + n + c;
      if (SPLIT_K) {
        atomicAdd(static_cast<float*>(Yv) + idx, acc[c][m]);
      } else {
        static_cast<__half*>(Yv)[idx] = __float2half(acc[c][m]);
      }
    }
  }
}

__global__ void kn_finalize_kernel(const float* __restrict__ src,
                                   __half* __restrict__ dst, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

// ------------------------------------------------------------- dispatch ----

template <int R, int M_MAX, int MT, typename XT>
void launch_nk(const XT* x, const __half* w, __half* y, int M, int N, int K) {
  const int rowsPerBlock = kNkWarpsPerBlock * R;
  skinny_gemm_nk_kernel<R, M_MAX, MT, XT>
      <<<dim3((N + rowsPerBlock - 1) / rowsPerBlock), dim3(kWarp * kNkWarpsPerBlock),
         0, at::cuda::getCurrentCUDAStream()>>>(x, w, y, M, N, K);
}

template <int M_MAX, int MT, typename XT>
bool dispatch_r(int r, const XT* x, const __half* w, __half* y, int M, int N,
                int K) {
  switch (r) {
    case 1: launch_nk<1, M_MAX, MT, XT>(x, w, y, M, N, K); return true;
    case 2: launch_nk<2, M_MAX, MT, XT>(x, w, y, M, N, K); return true;
    case 4: launch_nk<4, M_MAX, MT, XT>(x, w, y, M, N, K); return true;
    case 8: launch_nk<8, M_MAX, MT, XT>(x, w, y, M, N, K); return true;
    default: return false;
  }
}

template <int M_MAX, typename XT>
bool dispatch_rm(int r, int mt, const XT* x, const __half* w, __half* y, int M,
                 int N, int K) {
  const int mtc = MT_CLAMP(M_MAX, mt);
  switch (mtc) {
    case 1: return dispatch_r<M_MAX, 1, XT>(r, x, w, y, M, N, K);
    case 2: return dispatch_r<M_MAX, 2, XT>(r, x, w, y, M, N, K);
    case 4: return dispatch_r<M_MAX, 4, XT>(r, x, w, y, M, N, K);
    case 8: return dispatch_r<M_MAX, 8, XT>(r, x, w, y, M, N, K);
    default: return false;
  }
}

// Enough warps to fill the card: 19 SMs, and a streaming kernel this light wants
// several blocks each. Anything the n dimension cannot supply comes from k.
constexpr int kKnTargetWarps = 640;

int kn_slices(int N, int K, int override) {
  if (override > 0) return override;
  const int colBlocks = (N + kKnColsPerWarp - 1) / kKnColsPerWarp;
  int slices = (kKnTargetWarps + colBlocks - 1) / colBlocks;
  slices = std::min(slices, K / kKnUnroll);
  return std::max(slices, 1);
}

template <int M_MAX>
void launch_kn(const float* x, const __half* w, at::Tensor& y, int M, int N, int K,
               int slicesOverride) {
  const int colBlocks = (N + kKnColsPerWarp - 1) / kKnColsPerWarp;
  int slices = kn_slices(N, K, slicesOverride);
  // gridDim.y carries kKnWarps slices per block, so round up to a whole block.
  const int gridY = (slices + kKnWarps - 1) / kKnWarps;
  slices = gridY * kKnWarps;
  auto stream = at::cuda::getCurrentCUDAStream();

  if (kn_slices(N, K, slicesOverride) == 1) {
    // One slice covers all of k, so there is nothing to reduce: write fp16
    // straight out and skip the fp32 accumulator entirely. One warp per block
    // here, since the block's other warps would have no slice to take.
    skinny_gemm_kn_kernel<M_MAX, false>
        <<<dim3(colBlocks, 1), dim3(kWarp), 0, stream>>>(x, w, y.data_ptr(), M, N, K, 1);
    return;
  }
  at::Tensor acc = at::zeros({M, N}, y.options().dtype(at::kFloat));
  skinny_gemm_kn_kernel<M_MAX, true><<<dim3(colBlocks, gridY), dim3(kWarp * kKnWarps),
                                       0, stream>>>(
      x, w, acc.data_ptr(), M, N, K, slices);
  const size_t total = static_cast<size_t>(M) * N;
  kn_finalize_kernel<<<dim3((total + 255) / 256), dim3(256), 0, stream>>>(
      acc.data_ptr<float>(), reinterpret_cast<__half*>(y.data_ptr<at::Half>()), total);
}

at::Tensor to_float_2d(const at::Tensor& x) {
  return x.scalar_type() == at::kFloat ? x.contiguous() : x.to(at::kFloat).contiguous();
}

}  // namespace

// w is [N, K]; R defaults to the accumulator budget R*M <= 64.
at::Tensor skinny_gemm_nk(at::Tensor x, at::Tensor w, int64_t r_override,
                          bool half_x, int64_t mt_override) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && w.is_contiguous(), "bad inputs");
  TORCH_CHECK(w.scalar_type() == at::kHalf && x.dim() == 2 && w.dim() == 2, "bad inputs");
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K && K % kKPerLane == 0, "bad shape");
  TORCH_CHECK(M <= 32, "M > 32 not supported by the [N, K] kernel");

  at::Tensor y = at::empty({M, N}, w.options());
  const int r = r_override > 0 ? static_cast<int>(r_override)
                               : (M <= 4 ? 8 : (M <= 8 ? 4 : 2));
  const int mt = mt_override > 0 ? static_cast<int>(mt_override) : 1;
  const __half* wp = reinterpret_cast<const __half*>(w.data_ptr<at::Half>());
  __half* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

  bool ok;
  if (half_x) {
    // Read the activation as fp16 and convert in the inner loop: half the bytes
    // through L1, at the price of 8*M extra conversions per k-chunk.
    const at::Tensor xh = x.to(at::kHalf).contiguous();
    const __half* xp = reinterpret_cast<const __half*>(xh.data_ptr<at::Half>());
    if (M <= 1) ok = dispatch_rm<1>(r, mt, xp, wp, yp, M, N, K);
    else if (M <= 2) ok = dispatch_rm<2>(r, mt, xp, wp, yp, M, N, K);
    else if (M <= 4) ok = dispatch_rm<4>(r, mt, xp, wp, yp, M, N, K);
    else if (M <= 8) ok = dispatch_rm<8>(r, mt, xp, wp, yp, M, N, K);
    else if (M <= 16) ok = dispatch_rm<16>(r, mt, xp, wp, yp, M, N, K);
    else ok = dispatch_rm<32>(r, mt, xp, wp, yp, M, N, K);
  } else {
    const at::Tensor xf = to_float_2d(x);
    const float* xp = xf.data_ptr<float>();
    if (M <= 1) ok = dispatch_rm<1>(r, mt, xp, wp, yp, M, N, K);
    else if (M <= 2) ok = dispatch_rm<2>(r, mt, xp, wp, yp, M, N, K);
    else if (M <= 4) ok = dispatch_rm<4>(r, mt, xp, wp, yp, M, N, K);
    else if (M <= 8) ok = dispatch_rm<8>(r, mt, xp, wp, yp, M, N, K);
    else if (M <= 16) ok = dispatch_rm<16>(r, mt, xp, wp, yp, M, N, K);
    else ok = dispatch_rm<32>(r, mt, xp, wp, yp, M, N, K);
  }
  TORCH_CHECK(ok, "unsupported R=", r);
  return y;
}

// w is [K, N], i.e. already transposed and made contiguous at load time.
at::Tensor skinny_gemm_kn(at::Tensor x, at::Tensor w, int64_t slices) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && w.is_contiguous(), "bad inputs");
  TORCH_CHECK(w.scalar_type() == at::kHalf && x.dim() == 2 && w.dim() == 2, "bad inputs");
  const int M = x.size(0), K = x.size(1), N = w.size(1);
  TORCH_CHECK(w.size(0) == K, "K mismatch");
  TORCH_CHECK(K % kKnUnroll == 0, "K must be a multiple of 4");
  TORCH_CHECK(M <= 32, "M > 32 not supported");

  const at::Tensor xf = to_float_2d(x);
  at::Tensor y = at::empty({M, N}, w.options());
  const float* xp = xf.data_ptr<float>();
  const __half* wp = reinterpret_cast<const __half*>(w.data_ptr<at::Half>());
  const int s = static_cast<int>(slices);

  if (M <= 1) launch_kn<1>(xp, wp, y, M, N, K, s);
  else if (M <= 2) launch_kn<2>(xp, wp, y, M, N, K, s);
  else if (M <= 4) launch_kn<4>(xp, wp, y, M, N, K, s);
  else if (M <= 8) launch_kn<8>(xp, wp, y, M, N, K, s);
  else if (M <= 16) launch_kn<16>(xp, wp, y, M, N, K, s);
  else launch_kn<32>(xp, wp, y, M, N, K, s);
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("skinny_gemm_nk", &skinny_gemm_nk, "small-M fp16 GEMM, [N, K] weight",
        py::arg("x"), py::arg("w"), py::arg("r") = 0, py::arg("half_x") = false,
        py::arg("mt") = 0);
  m.def("skinny_gemm_kn", &skinny_gemm_kn, "small-M fp16 GEMM, [K, N] weight",
        py::arg("x"), py::arg("w"), py::arg("slices") = 0);
}
