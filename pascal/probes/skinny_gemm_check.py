"""Build and measure the sm_61 small-M fp16 GEMMs against cuBLAS.

Compiles `skinny_gemm_sm61.cu` as a standalone extension so the kernels can be
iterated on without rebuilding vLLM, checks them against an fp32 reference, and
times them at this model's shapes.

Four candidates, all computing the same product:

    linear   torch.nn.functional.linear(x, w)   -- what vLLM calls today, [N, K]
    xw       torch.mm(x, w.t().contiguous())    -- cuBLAS on a [K, N] weight
    nk       our kernel on the [N, K] weight
    kn       our kernel on the [K, N] weight

The bar is `xw`, not `linear`: transposing the weight once at load time is free
and already buys cuBLAS 1.3-2.5x at M>=2, so a kernel is only worth carrying if
it beats that.

    python pascal/probes/skinny_gemm_check.py
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.cpp_extension import load

SHAPES = [
    ("qkv", 2048, 4096),
    ("o", 2048, 2048),
    ("gate_up", 2048, 12288),
    ("down", 6144, 2048),
    ("lm_head", 2048, 151936),
]


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
    ap.add_argument("--out", default="/work/skinny_gemm_check.json")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--shapes", nargs="+", default=[s[0] for s in SHAPES])
    ap.add_argument("--src", default=os.path.join(os.path.dirname(__file__),
                                                  "skinny_gemm_sm61.cu"))
    args = ap.parse_args()

    ext = load(
        name="skinny_gemm_sm61",
        sources=[args.src],
        # No --use_fast_math: it implies -ftz=true, which flushes fp16
        # subnormals to zero on the way through the fp32 convert. The fp8 probe
        # already caught that as a silent miscompile; the same trap applies here.
        #
        # `-Xptxas -v` as two tokens makes nvcc hand ptxas the -std flag torch
        # appends afterwards, and ptxas rejects it; the joined form does not.
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_61,code=sm_61",
                           "--ptxas-options=-v"],
        verbose=True,
    )

    ref = torch.randn(256 * 1024 * 1024, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(ref)
    ceiling = 2 * ref.numel() * 2 / timeit(lambda: dst.copy_(ref), iters=20) / 1e9
    del ref, dst
    torch.cuda.empty_cache()
    print(f"\nachievable bandwidth (copy): {ceiling:.1f} GB/s\n")

    results = {"bandwidth_GB_s": round(ceiling, 1), "shapes": {}}
    print(f"{'shape':<9} {'M':>3} {'floor':>7} {'linear':>8} {'xw':>8} {'nk f32':>8} "
          f"{'nk f16':>8} {'kn':>8} {'best %fl':>9} {'err nk16':>9}")
    print("-" * 96)

    for label, K, N in SHAPES:
        if label not in args.shapes:
            continue
        w = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.05
        wt = w.t().contiguous()  # [K, N]
        w32 = w.float()
        floor_us = N * K * 2 / ceiling / 1e9 * 1e6
        rows = []
        for m in args.batches:
            x = torch.randn(m, K, device="cuda", dtype=torch.float16) * 0.05

            y_ref = x.float() @ w32.t()
            scale = y_ref.abs().max().item()

            def relerr(y):
                return ((y.float() - y_ref).abs().max() / scale).item()

            err_kn = relerr(ext.skinny_gemm_kn(x, wt))
            err_bl = relerr(torch.nn.functional.linear(x, w))
            err_nk = relerr(ext.skinny_gemm_nk(x, w)) if m <= 16 else float("nan")
            err_nkh = (relerr(ext.skinny_gemm_nk(x, w, 0, True)) if m <= 16
                       else float("nan"))

            t_lin = timeit(lambda: torch.nn.functional.linear(x, w))
            t_xw = timeit(lambda: torch.mm(x, wt))
            t_nk = timeit(lambda: ext.skinny_gemm_nk(x, w)) if m <= 16 else float("nan")
            t_nkh = (timeit(lambda: ext.skinny_gemm_nk(x, w, 0, True))
                     if m <= 16 else float("nan"))
            t_kn = timeit(lambda: ext.skinny_gemm_kn(x, wt))
            best_cublas = min(t_lin, t_xw)

            rows.append({
                "M": m,
                "floor_us": round(floor_us, 1),
                "linear_us": round(t_lin * 1e6, 2),
                "xw_us": round(t_xw * 1e6, 2),
                "nk_us": round(t_nk * 1e6, 2),
                "nk_half_x_us": round(t_nkh * 1e6, 2),
                "rel_err_nk_half_x": err_nkh,
                "kn_us": round(t_kn * 1e6, 2),
                "kn_pct_floor": round(100 * floor_us / (t_kn * 1e6), 1),
                "kn_speedup_vs_best_cublas": round(best_cublas / t_kn, 3),
                "kn_speedup_vs_linear": round(t_lin / t_kn, 3),
                "rel_err_kn": err_kn,
                "rel_err_nk": err_nk,
                "rel_err_cublas": err_bl,
            })
            best_ours = min(v for v in (t_nk, t_nkh, t_kn) if v == v)
            print(f"{label:<9} {m:>3} {floor_us:>7.1f} {t_lin * 1e6:>8.1f} "
                  f"{t_xw * 1e6:>8.1f} {t_nk * 1e6:>8.1f} {t_nkh * 1e6:>8.1f} "
                  f"{t_kn * 1e6:>8.1f} {100 * floor_us / (best_ours * 1e6):>8.1f}% "
                  f"{err_nkh:>9.2e}")

        results["shapes"][label] = {"K": K, "N": N, "rows": rows}
        del w, wt, w32
        torch.cuda.empty_cache()

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
