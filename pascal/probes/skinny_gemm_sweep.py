"""Sweep R for the [N, K] kernel at the batch sizes cuBLAS still owns.

M<=8 is settled: the kernel is at 90-105% of the bandwidth floor and shipped.
M=9..64 is not, and it is where the money now is -- at batch 32 cuBLAS spends
11.0 ms on lm_head alone against a 2.9 ms floor. R trades accumulators
(R*M of them) against how many times each warp re-reads the activation, so the
best R moves with M and there is no substitute for measuring it.

    python pascal/probes/skinny_gemm_sweep.py
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
R_VALUES = [1, 2, 4, 8]
MT_VALUES = [1, 2, 4, 8]


def timeit(fn, warmup: int = 5, iters: int = 30) -> float:
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
    ap.add_argument("--out", default="/work/skinny_sweep.json")
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--src", default=os.path.join(os.path.dirname(__file__),
                                                  "skinny_gemm_sm61.cu"))
    args = ap.parse_args()

    ext = load(name="skinny_gemm_sm61", sources=[args.src],
               extra_cuda_cflags=["-O3", "-gencode", "arch=compute_61,code=sm_61"],
               verbose=False)

    ref = torch.randn(256 * 1024 * 1024, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(ref)
    ceiling = 2 * ref.numel() * 2 / timeit(lambda: dst.copy_(ref), iters=20) / 1e9
    del ref, dst
    torch.cuda.empty_cache()
    print(f"achievable bandwidth (copy): {ceiling:.1f} GB/s\n")

    results = {"bandwidth_GB_s": round(ceiling, 1), "shapes": {}}
    head = (f"{'shape':<9} {'M':>3} {'floor':>7} {'linear':>8} {'transp':>8} "
            f"{'ours':>8} config    {'%floor':>7} {'gain':>6}")
    print(head)
    print("-" * len(head))

    for label, K, N in SHAPES:
        w = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.05
        w32 = w.float()
        floor_us = N * K * 2 / ceiling / 1e9 * 1e6
        rows = []
        for m in args.batches:
            x = torch.randn(m, K, device="cuda", dtype=torch.float16) * 0.05
            y_ref = x.float() @ w32.t()
            scale = y_ref.abs().max().item()

            t_lin = timeit(lambda: torch.nn.functional.linear(x, w))
            t_tr = timeit(lambda: torch.mm(w, x.t()).t().contiguous())
            t_r, err = {}, {}
            for r in R_VALUES:
                for mt in MT_VALUES:
                    if mt > m:
                        continue
                    y = ext.skinny_gemm_nk(x, w, r, True, mt)
                    e = ((y.float() - y_ref).abs().max() / scale).item()
                    if e > 5e-3:
                        print(f"    WRONG r={r} mt={mt} rel={e:.2e}")
                    err[f"r{r}m{mt}"] = e
                    t_r[(r, mt)] = timeit(
                        lambda r=r, mt=mt: ext.skinny_gemm_nk(x, w, r, True, mt))
            best_r = min(t_r, key=t_r.get)
            best = min(t_r[best_r], t_lin, t_tr)
            rows.append({
                "M": m, "floor_us": round(floor_us, 1),
                "linear_us": round(t_lin * 1e6, 2),
                "transposed_us": round(t_tr * 1e6, 2),
                "by_R_MT_us": {f"r{r}m{mt}": round(v * 1e6, 2)
                               for (r, mt), v in t_r.items()},
                "rel_err": err,
                "best_R": best_r[0], "best_MT": best_r[1],
                "gain_vs_cublas": round(min(t_lin, t_tr) / t_r[best_r], 3),
            })
            print(f"{label:<9} {m:>3} {floor_us:>7.1f} {t_lin * 1e6:>8.1f} "
                  f"{t_tr * 1e6:>8.1f} {t_r[best_r] * 1e6:>8.1f} "
                  f"R={best_r[0]} MT={best_r[1]} "
                  f"{100 * floor_us / (best * 1e6):>6.1f}% "
                  f"{min(t_lin, t_tr) / t_r[best_r]:>5.2f}x")
        results["shapes"][label] = {"K": K, "N": N, "rows": rows}
        del w, w32
        torch.cuda.empty_cache()

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
