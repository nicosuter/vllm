"""How fast is `tl.dot` on sm_61, and does the input dtype change it?

Two kernels in this fork are limited by the same thing: below sm_70 Triton
lowers `tl.dot` to an FMA loop rather than to an MMA instruction, and both the
Triton GEMM prototype (`skinny_gemm.py`) and vLLM's attention kernel land near
**0.7-1.0 TFLOP/s against this card's 8.19**.

Before writing attention in CUDA it is worth knowing whether that ceiling is the
lowering itself or something about how it is being fed. `tl.dot` with fp16
inputs has to convert on the way in; with fp32 inputs it does not, and fp32 is
the compute type on this card anyway. If fp32 inputs are markedly faster, the
attention kernel can cast its tiles and stay in Triton.

    python pascal/probes/tl_dot_rate.py
"""

from __future__ import annotations

import argparse
import json

import torch
import triton
import triton.language as tl


@triton.jit
def _dot_kernel(
    A, B, C, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    CAST_F32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_n[:, None] * K + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if CAST_F32:
            # The operands are still fp16 in memory and in shared memory; only
            # the values handed to tl.dot are float. If this recovers the fp32
            # rate, the fix for every tl.dot on this card is one cast.
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc.to(C.dtype.element_ty))


def timeit(fn, warmup: int = 5, iters: int = 20) -> float:
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
    ap.add_argument("--out", default="/work/tl_dot_rate.json")
    ap.add_argument("--size", type=int, default=2048)
    args = ap.parse_args()

    peak = 8.19
    n = args.size
    flops = 2 * n * n * n
    results = {}

    print(f"square matmul {n}x{n}x{n} = {flops / 1e9:.1f} GFLOP, "
          f"fp32 FMA peak {peak} TFLOP/s\n")
    print(f"{'dtype':>8} {'BM':>4} {'BN':>4} {'BK':>4} {'warps':>6} {'ms':>8} "
          f"{'TFLOP/s':>8} {'%peak':>7}")
    print("-" * 60)

    for dtype in (torch.float16, torch.float32):
        a = torch.randn(n, n, device="cuda", dtype=dtype) * 0.05
        b = torch.randn(n, n, device="cuda", dtype=dtype) * 0.05
        c = torch.empty(n, n, device="cuda", dtype=dtype)

        # cuBLAS at the same shape, as the thing to beat.
        t = timeit(lambda: torch.mm(a, b.t()))
        tf = flops / t / 1e12
        print(f"{str(dtype).split('.')[-1]:>8} {'cuBLAS':>19} {t * 1e3:>8.2f} "
              f"{tf:>8.3f} {100 * tf / peak:>6.1f}%")
        results[f"{dtype}_cublas"] = {"ms": round(t * 1e3, 3), "TFLOP_s": round(tf, 3)}

        casts = (False, True) if dtype == torch.float16 else (False,)
        for cast in casts:
          for bm, bn, bk, warps in [(64, 64, 32, 4), (64, 64, 64, 4),
                                    (128, 64, 32, 4), (64, 128, 32, 4),
                                    (128, 128, 32, 8), (32, 64, 64, 4)]:
            tag = ("f16->f32" if cast else str(dtype).split(".")[-1])
            try:
                fn = lambda bm=bm, bn=bn, bk=bk, warps=warps, cast=cast: _dot_kernel[
                    (triton.cdiv(n, bm), triton.cdiv(n, bn))
                ](a, b, c, n, n, n, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                  CAST_F32=cast, num_warps=warps)
                fn()
                torch.cuda.synchronize()
            except Exception as exc:
                print(f"{tag:>8} {bm:>4} {bn:>4} {bk:>4} "
                      f"{warps:>6}   {type(exc).__name__}")
                continue
            t = timeit(fn)
            tf = flops / t / 1e12
            results[f"{tag}_{bm}_{bn}_{bk}_{warps}"] = {
                "ms": round(t * 1e3, 3), "TFLOP_s": round(tf, 3),
                "pct_peak": round(100 * tf / peak, 1),
            }
            print(f"{tag:>8} {bm:>4} {bn:>4} {bk:>4} {warps:>6} "
                  f"{t * 1e3:>8.2f} {tf:>8.3f} {100 * tf / peak:>6.1f}%")
        del a, b, c
        torch.cuda.empty_cache()
        print()

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
