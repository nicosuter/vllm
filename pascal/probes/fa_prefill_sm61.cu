// FlashAttention for sm_61 prefill. **Correct, and not fast enough to ship.**
//
// Kept because it is a working baseline for the next attempt and because the
// reason it loses is a property of the hardware rather than of this code.
//
// Measured against vLLM's Triton kernel as this fork now tunes it (fp32 dot
// operands, BLOCK_M=32), milliseconds per layer, 16 query heads / 8 KV heads /
// head_dim 128 on a GTX 1070 Ti:
//
//     n      triton   flash BR=32   flash BR=64   flash BR=128
//     512     1.199         1.654         1.257    will not launch
//     1024    4.245           ---         4.094    will not launch
//
// The kernel is bound by the ratio of FMAs to shared-memory reads in the two
// dots. A thread owning RT rows and CT columns does RT*CT FMAs per RT+CT reads,
// so the ratio is 2 at RT=CT=4 and 4 at RT=CT=8 -- and 8 is where it would beat
// Triton. RT=8 needs BR=128, whose query tile and score buffer come to 64 KiB
// against Pascal's 48 KiB per-block limit, so that configuration cannot be
// launched at all.
//
// Occupancy is not the explanation, which was worth checking: BR=32 fits in
// 28 KiB and admits three blocks per SM instead of two, and it is *slower*
// (1.654 against 1.257), because halving RT halves the FMA-to-read ratio.
//
// Beating Triton here therefore needs a different data flow, not a bigger tile
// -- something that keeps more of Q in registers across the KV sweep instead of
// re-reading it from shared memory for every d. That is a redesign, and this
// file is where it should start from.
//
// Prefill attention is the largest single inefficiency left in this fork: 105.9
// ms of a ~635 ms 1024-token prefill, running at 13.9% of the card's 8.19
// TFLOP/s even after `tl.dot` was given float operands. Triton's sub-sm_70
// lowering is doing the arithmetic as an FMA loop, which is correct and slow.
//
// Flash attention is an algorithm, not a hardware feature. Online softmax and
// tiling are arithmetic; what FA2 and FA3 need tensor cores and `cp.async` for
// is speed, not correctness. So the same schedule works here with fp32
// accumulation and ordinary loads -- and it keeps the O(n) memory that makes it
// worth having in the first place, instead of the [heads, n, n] score matrix a
// bmm-based implementation would materialise (33 MiB per layer at n=1024).
//
// Layouts match vLLM's so that integration is a call, not a conversion:
//
//   q            [num_tokens, num_q_heads, D]
//   key_cache    [num_blocks, block_size, num_kv_heads, D]   (paged)
//   value_cache  [num_blocks, block_size, num_kv_heads, D]
//   out          [num_tokens, num_q_heads, D]
//   block_table  [num_seqs, max_blocks_per_seq]
//   cu_seqlens_q [num_seqs + 1]
//   seqused_k    [num_seqs]
//
// Chunked prefill is handled by the usual alignment: a query at local position
// i attends to key positions up to `kv_len - q_len + i`, so a chunk that arrives
// after a cached prefix still masks correctly.
//
// Scheduling. One block owns BR query rows of one head of one sequence, and
// sweeps the keys in tiles of BC. Threads are laid out so that the rows a thread
// owns are the same for the scores and for the output accumulator -- that is
// what lets the running max and sum live in registers instead of shared memory,
// and it makes the softmax reduction a shuffle across eight adjacent lanes.
//
//   thread t owns rows [ (t/8)*RT, +RT )   and, for scores, cols [ (t%8)*CT, +CT )
//                                          and, for output,  dims [ (t%8)*DT, +DT )
//
// The two dots are register-blocked for the same reason the GEMM is: an FMA
// takes ~6 cycles to land and the accumulator is also an operand, so the inner
// loop has to walk enough distinct accumulators to cover it. RT*CT and RT*DT are
// both >= 16 here.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>

namespace {

constexpr int kThreads = 128;
constexpr int kColGroups = 8;               // threads spanning the column axis
constexpr int kRowGroups = kThreads / kColGroups;  // 16

template <int D, int BR, int BC>
__global__ __launch_bounds__(kThreads) void fa_prefill_kernel(
    const __half* __restrict__ Q, const __half* __restrict__ KC,
    const __half* __restrict__ VC, __half* __restrict__ O,
    const int* __restrict__ cu_seqlens_q, const int* __restrict__ seqused_k,
    const int* __restrict__ block_table, int block_table_stride, int block_size,
    int num_q_heads, int num_kv_heads, float scale) {
  constexpr int RT = BR / kRowGroups;  // rows per thread
  constexpr int CT = BC / kColGroups;  // score columns per thread
  constexpr int DT = D / kColGroups;   // output dims per thread

  const int qtile = blockIdx.x;
  const int qh = blockIdx.y;
  const int seq = blockIdx.z;

  const int q_start = cu_seqlens_q[seq];
  const int q_len = cu_seqlens_q[seq + 1] - q_start;
  const int row0 = qtile * BR;
  if (row0 >= q_len) return;

  const int kv_len = seqused_k[seq];
  const int ctx = kv_len - q_len;  // keys already cached before this chunk
  const int kvh = qh / (num_q_heads / num_kv_heads);

  const int tid = threadIdx.x;
  const int rg = tid / kColGroups;  // 0..15
  const int cg = tid % kColGroups;  // 0..7

  __shared__ __half qs[BR][D];
  __shared__ __half ks[BC][D];
  __shared__ __half vs[BC][D];
  __shared__ float ps[BR][BC];

  // ---- load the query tile ------------------------------------------------
  for (int i = tid; i < BR * D; i += kThreads) {
    const int r = i / D, d = i % D;
    const int tok = row0 + r;
    qs[r][d] = tok < q_len
                   ? Q[((size_t)(q_start + tok) * num_q_heads + qh) * D + d]
                   : __float2half(0.0f);
  }

  float acc[RT][DT];
  float m[RT], l[RT];
#pragma unroll
  for (int i = 0; i < RT; ++i) {
    m[i] = -FLT_MAX;
    l[i] = 0.0f;
#pragma unroll
    for (int j = 0; j < DT; ++j) acc[i][j] = 0.0f;
  }
  __syncthreads();

  // A query at local position i attends to key positions <= ctx + i, so the
  // last row of this tile bounds the whole tile.
  const int kv_hi = min(kv_len, ctx + row0 + BR);

  for (int k0 = 0; k0 < kv_hi; k0 += BC) {
    // ---- load one key/value tile out of the paged cache -------------------
    for (int i = tid; i < BC * D; i += kThreads) {
      const int t = i / D, d = i % D;
      const int p = k0 + t;
      __half kv = __float2half(0.0f), vv = __float2half(0.0f);
      if (p < kv_hi) {
        const int blk = block_table[seq * block_table_stride + p / block_size];
        const size_t off =
            (((size_t)blk * block_size + (p % block_size)) * num_kv_heads + kvh) * D + d;
        kv = KC[off];
        vv = VC[off];
      }
      ks[t][d] = kv;
      vs[t][d] = vv;
    }
    __syncthreads();

    // ---- S = Q K^T --------------------------------------------------------
    float s[RT][CT];
#pragma unroll
    for (int i = 0; i < RT; ++i)
#pragma unroll
      for (int j = 0; j < CT; ++j) s[i][j] = 0.0f;

    // Two d at a time, read as half2. The inner nest is RT*CT FMAs against
    // RT+CT loads and conversions, so at RT=CT=4 that is one non-FMA
    // instruction per two FMAs -- halving the load count is most of what can be
    // done without a bigger register tile than sm_61 has registers for.
#pragma unroll 4
    for (int d = 0; d < D; d += 2) {
      float2 qv[RT], kvv[CT];
#pragma unroll
      for (int i = 0; i < RT; ++i)
        qv[i] = __half22float2(*reinterpret_cast<const __half2*>(&qs[rg * RT + i][d]));
#pragma unroll
      for (int j = 0; j < CT; ++j)
        kvv[j] = __half22float2(*reinterpret_cast<const __half2*>(&ks[cg * CT + j][d]));
#pragma unroll
      for (int i = 0; i < RT; ++i)
#pragma unroll
        for (int j = 0; j < CT; ++j) {
          s[i][j] = fmaf(qv[i].x, kvv[j].x, s[i][j]);
          s[i][j] = fmaf(qv[i].y, kvv[j].y, s[i][j]);
        }
    }

    // ---- mask, then the online-softmax rescale ----------------------------
    float mtile[RT];
#pragma unroll
    for (int i = 0; i < RT; ++i) {
      const int qpos = ctx + row0 + rg * RT + i;
      float best = -FLT_MAX;
#pragma unroll
      for (int j = 0; j < CT; ++j) {
        const int kpos = k0 + cg * CT + j;
        s[i][j] = (kpos <= qpos && kpos < kv_len && (row0 + rg * RT + i) < q_len)
                      ? s[i][j] * scale
                      : -FLT_MAX;
        best = fmaxf(best, s[i][j]);
      }
      mtile[i] = best;
    }

    // The eight threads sharing a row group are adjacent lanes, so the row
    // reduction is a shuffle and never touches shared memory.
#pragma unroll
    for (int i = 0; i < RT; ++i) {
#pragma unroll
      for (int off = kColGroups / 2; off > 0; off >>= 1)
        mtile[i] = fmaxf(mtile[i], __shfl_xor_sync(0xffffffffu, mtile[i], off));
    }

    float rescale[RT];
#pragma unroll
    for (int i = 0; i < RT; ++i) {
      const float mnew = fmaxf(m[i], mtile[i]);
      rescale[i] = (m[i] == -FLT_MAX) ? 0.0f : __expf(m[i] - mnew);
      m[i] = mnew;
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < CT; ++j) {
        const float e = (s[i][j] == -FLT_MAX) ? 0.0f : __expf(s[i][j] - mnew);
        ps[rg * RT + i][cg * CT + j] = e;
        sum += e;
      }
#pragma unroll
      for (int off = kColGroups / 2; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
      l[i] = l[i] * rescale[i] + sum;
#pragma unroll
      for (int j = 0; j < DT; ++j) acc[i][j] *= rescale[i];
    }
    __syncthreads();

    // ---- O += P V ---------------------------------------------------------
#pragma unroll 4
    for (int t = 0; t < BC; ++t) {
      float pv[RT];
#pragma unroll
      for (int i = 0; i < RT; ++i) pv[i] = ps[rg * RT + i][t];
      float2 vvv[DT / 2];
#pragma unroll
      for (int j = 0; j < DT / 2; ++j)
        vvv[j] = __half22float2(
            *reinterpret_cast<const __half2*>(&vs[t][cg * DT + 2 * j]));
#pragma unroll
      for (int i = 0; i < RT; ++i)
#pragma unroll
        for (int j = 0; j < DT / 2; ++j) {
          acc[i][2 * j] = fmaf(pv[i], vvv[j].x, acc[i][2 * j]);
          acc[i][2 * j + 1] = fmaf(pv[i], vvv[j].y, acc[i][2 * j + 1]);
        }
    }
    __syncthreads();
  }

  // ---- epilogue -----------------------------------------------------------
#pragma unroll
  for (int i = 0; i < RT; ++i) {
    const int r = row0 + rg * RT + i;
    if (r >= q_len) continue;
    const float inv = (l[i] > 0.0f) ? 1.0f / l[i] : 0.0f;
    __half* dst = O + ((size_t)(q_start + r) * num_q_heads + qh) * D;
#pragma unroll
    for (int j = 0; j < DT; ++j)
      dst[cg * DT + j] = __float2half(acc[i][j] * inv);
  }
}

}  // namespace

void fa_prefill(at::Tensor q, at::Tensor key_cache, at::Tensor value_cache,
                at::Tensor out, at::Tensor cu_seqlens_q, at::Tensor seqused_k,
                at::Tensor block_table, int64_t max_seqlen_q, double scale,
                int64_t br) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf, "fp16 cuda q expected");
  TORCH_CHECK(q.dim() == 3 && key_cache.dim() == 4, "bad shapes");
  const int num_q_heads = q.size(1);
  const int D = q.size(2);
  const int block_size = key_cache.size(1);
  const int num_kv_heads = key_cache.size(2);
  const int num_seqs = seqused_k.size(0);
  TORCH_CHECK(D == 128 || D == 64, "head_size 64 or 128 only");
  TORCH_CHECK(num_q_heads % num_kv_heads == 0, "GQA mismatch");

  // BR trades the register tile against occupancy. At BR=64 the block needs
  // 40 KB of shared memory, and GP104's 96 KB per SM then admits only two
  // blocks -- eight warps, which is not enough to hide shared-memory latency.
  // BR=32 halves the query tile and the score buffer to 28 KB for three blocks,
  // at the cost of half the rows per thread.
  const int BR = br > 0 ? int(br) : 64;
  constexpr int BC = 32;
  const int qtiles = (int(max_seqlen_q) + BR - 1) / BR;
  const dim3 grid(qtiles, num_q_heads, num_seqs);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto* qp = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
  auto* kp = reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>());
  auto* vp = reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>());
  auto* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());

#define LAUNCH(DD, BRR)                                                       \
  fa_prefill_kernel<DD, BRR, BC><<<grid, kThreads, 0, stream>>>(               \
      qp, kp, vp, op, cu_seqlens_q.data_ptr<int>(), seqused_k.data_ptr<int>(), \
      block_table.data_ptr<int>(), block_table.stride(0), block_size,          \
      num_q_heads, num_kv_heads, float(scale))

  if (D == 128) {
    if (BR == 32) LAUNCH(128, 32);
    else if (BR == 128) LAUNCH(128, 128);
    else LAUNCH(128, 64);
  } else {
    if (BR == 32) LAUNCH(64, 32);
    else LAUNCH(64, 64);
  }
#undef LAUNCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa_prefill", &fa_prefill, "flash attention prefill for sm_61",
        py::arg("q"), py::arg("key_cache"), py::arg("value_cache"), py::arg("out"),
        py::arg("cu_seqlens_q"), py::arg("seqused_k"), py::arg("block_table"),
        py::arg("max_seqlen_q"), py::arg("scale"), py::arg("br") = 0);
}
