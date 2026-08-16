"""A small-M fp16 GEMM for sm_61 in Triton. Kept as a recorded negative result.

Triton was the obvious first move: `tl.dot` is proven bit-exact on this card
(see the probe results in the README), it needs no nvcc rebuild, and the fork
already carries Triton kernels behind `direct_register_custom_op`.

It does not work. Below sm_70 Triton lowers `tl.dot` to its FMA path, and that
path runs at roughly **8% of this card's FMA peak** -- 0.70 TFLOP/s against 8.19.
Measured against the same shapes the CUDA kernel now serves:

    shape   M    cuBLAS    triton   of roofline
    qkv     1      83 us    383 us         21%
    qkv     16    192 us    339 us         23%
    down    16    488 us    611 us         19%

Correct to 2.5e-4, and about a third of cuBLAS's speed on a path where cuBLAS is
itself 2.4x off the memory roofline. The shipped kernel is CUDA:
`csrc/libtorch_stable/quantization/pascal_skinny_gemm.cu`.

`fp16_linear_rate.py` found that cuBLAS reaches the memory roofline at M=1 and
then loses 2.2-4.1x the moment M reaches 2, staying flat and wrong all the way to
M=16. Decode at batch>1 is ~90% dense fp16 GEMM, so that cliff *is* the batching
curve.

The fix has to obey three sm_61 facts:

  * fp16 must never be the accumulate type (`__hfma2` is 1/64 rate), so the
    accumulator is fp32 -- which is what `tl.dot(..., out_dtype=tl.float32)`
    lowers to on the FMA path Triton selects below sm_70.
  * the weight must be read exactly once from DRAM. That means every column of
    the activation has to be resident while a weight tile is in registers, which
    is why this is one kernel over all M rather than M gemv calls.
  * activations are tiny (M*K*2 bytes, <= 200 KiB here) so the re-read per output
    tile lands in the 2 MiB L2 and never reaches DRAM.

    python pascal/probes/skinny_gemm.py
"""

from __future__ import annotations

import argparse
import json

import torch
import triton
import triton.language as tl

# Qwen3-VL-Embedding-2B's text stack, as (label, K, N).
SHAPES = [
    ("qkv", 2048, 4096),
    ("o", 2048, 2048),
    ("gate_up", 2048, 12288),
    ("down", 6144, 2048),
    ("lm_head", 2048, 151936),
]


def _configs():
    out = []
    for block_n in (32, 64, 128):
        for block_k in (32, 64, 128):
            for warps in (2, 4, 8):
                for stages in (1, 2, 3):
                    out.append(
                        triton.Config(
                            {"BLOCK_N": block_n, "BLOCK_K": block_k},
                            num_warps=warps,
                            num_stages=stages,
                        )
                    )
    return out


@triton.autotune(configs=_configs(), key=["N", "K", "BLOCK_M"])
@triton.jit
def _skinny_gemm_kernel(
    X,
    W,
    Y,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = W + offs_n[:, None] * stride_wn + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        if EVEN_K:
            x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        else:
            k_mask = offs_k < K - k0
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def skinny_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y[M, N] = x[M, K] @ w[N, K].T, with M small."""
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    # tl.dot needs at least 16 on every axis; padding M costs arithmetic that
    # this kernel has to spare, since it is bandwidth-bound on the weight.
    block_m = 16 if m <= 16 else triton.next_power_of_2(m)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)  # noqa: E731
    _skinny_gemm_kernel[grid](
        x, w, y, m, n, k,
        x.stride(0), w.stride(0), y.stride(0),
        BLOCK_M=block_m,
        # Every K here is a multiple of 128 and BLOCK_K never exceeds it, so
        # this is a conservative test for "no tail iteration".
        EVEN_K=(k % 128 == 0),
    )
    return y


def timeit(fn, warmup: int = 5, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/skinny_gemm.json")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--shapes", nargs="+", default=[s[0] for s in SHAPES])
    args = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  sm_{props.major}{props.minor}")

    # Same reference as fp16_linear_rate.py, so the two tables are comparable.
    ref = torch.randn(256 * 1024 * 1024, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(ref)
    t_copy = timeit(lambda: dst.copy_(ref), iters=20)
    ceiling = 2 * ref.numel() * 2 / t_copy / 1e9
    del ref, dst
    torch.cuda.empty_cache()
    print(f"achievable bandwidth (copy): {ceiling:.1f} GB/s\n")

    results = {"bandwidth_GB_s": round(ceiling, 1), "shapes": {}}
    print(f"{'shape':<9} {'M':>4} {'cuBLAS us':>10} {'triton us':>10} {'speedup':>8} "
          f"{'GB/s':>7} {'%roof':>7} {'rel_err':>10}")
    print("-" * 74)

    for label, K, N in SHAPES:
        if label not in args.shapes:
            continue
        w = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.05
        wbytes = N * K * 2
        w32 = w.float()
        rows = []
        for m in args.batches:
            x = torch.randn(m, K, device="cuda", dtype=torch.float16) * 0.05

            y_ref = (x.float() @ w32.t())
            y_tri = skinny_gemm(x, w)
            rel = ((y_tri.float() - y_ref).abs().max() / y_ref.abs().max()).item()

            t_cublas = timeit(lambda: torch.nn.functional.linear(x, w))
            t_triton = timeit(lambda: skinny_gemm(x, w))
            gbs = wbytes / t_triton / 1e9
            rows.append({
                "M": m,
                "cublas_us": round(t_cublas * 1e6, 2),
                "triton_us": round(t_triton * 1e6, 2),
                "speedup": round(t_cublas / t_triton, 3),
                "GB_s": round(gbs, 1),
                "pct_roofline": round(100 * gbs / ceiling, 1),
                "rel_err": rel,
            })
            print(f"{label:<9} {m:>4} {t_cublas * 1e6:>10.2f} {t_triton * 1e6:>10.2f} "
                  f"{t_cublas / t_triton:>7.2f}x {gbs:>7.1f} {100 * gbs / ceiling:>6.1f}% "
                  f"{rel:>10.2e}")
        results["shapes"][label] = {"K": K, "N": N, "rows": rows}
        del w, w32
        torch.cuda.empty_cache()

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
