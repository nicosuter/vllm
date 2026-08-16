"""Can the cuBLAS small-M cliff be dodged without writing a kernel?

`fp16_linear_rate.py` showed cuBLAS at the roofline for M=1 and 2.2-4.1x off it
for M=2..16 -- and, oddly, *faster* at M=32 than at M=16. That non-monotonicity
says the loss is kernel selection, not arithmetic, so it is worth asking whether
the selection can simply be steered.

Four ways to ask for the same product:

  linear      y[M,N] = x[M,K] @ w[N,K].T          -- what vLLM calls today
  transposed  y.T    = w[N,K] @ x[M,K].T          -- the same GEMM with M and N
                                                     swapped in cuBLAS's frame
  padded      linear() on M rounded up to a size cuBLAS handles well, sliced
  fp32        the fp32 GEMM, which reads twice the bytes but hits a different
              (and on this card much better exercised) kernel family

    python pascal/probes/cublas_variants.py
"""

from __future__ import annotations

import argparse
import json

import torch

SHAPES = [
    ("qkv", 2048, 4096),
    ("o", 2048, 2048),
    ("gate_up", 2048, 12288),
    ("down", 6144, 2048),
]

PAD_TO = [8, 16, 32, 64, 128]


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
    ap.add_argument("--out", default="/work/cublas_variants.json")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = ap.parse_args()

    ref = torch.randn(256 * 1024 * 1024, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(ref)
    ceiling = 2 * ref.numel() * 2 / timeit(lambda: dst.copy_(ref), iters=20) / 1e9
    del ref, dst
    torch.cuda.empty_cache()
    print(f"achievable bandwidth (copy): {ceiling:.1f} GB/s\n")

    results = {"bandwidth_GB_s": round(ceiling, 1), "shapes": {}}

    for label, K, N in SHAPES:
        w = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.05
        wt = w.t().contiguous()  # [K, N], for the transposed dispatch
        w32 = w.float()
        floor_us = N * K * 2 / ceiling / 1e9 * 1e6
        print(f"== {label}  K={K} N={N}   bandwidth floor {floor_us:.1f} us")
        rows = []
        for m in args.batches:
            x = torch.randn(m, K, device="cuda", dtype=torch.float16) * 0.05
            entry = {"M": m, "floor_us": round(floor_us, 1)}

            entry["linear_us"] = round(
                timeit(lambda: torch.nn.functional.linear(x, w)) * 1e6, 2
            )
            entry["transposed_us"] = round(
                timeit(lambda: torch.mm(w, x.t())) * 1e6, 2
            )
            entry["xw_us"] = round(timeit(lambda: torch.mm(x, wt)) * 1e6, 2)
            entry["fp32_us"] = round(
                timeit(lambda: torch.nn.functional.linear(x.float(), w32)) * 1e6, 2
            )

            for pad in PAD_TO:
                if pad <= m:
                    continue
                xp = torch.zeros(pad, K, device="cuda", dtype=torch.float16)
                xp[:m] = x
                entry[f"pad{pad}_us"] = round(
                    timeit(lambda: torch.nn.functional.linear(xp, w)[:m]) * 1e6, 2
                )

            best = min(
                (v, k) for k, v in entry.items() if k.endswith("_us") and k != "floor_us"
            )
            entry["best"] = best[1]
            entry["best_speedup"] = round(entry["linear_us"] / best[0], 3)
            entry["best_pct_floor"] = round(100 * floor_us / best[0], 1)
            rows.append(entry)

            cells = " ".join(
                f"{k[:-3]}={v:.0f}"
                for k, v in entry.items()
                if k.endswith("_us") and k != "floor_us"
            )
            print(f"  M={m:<4} {cells}   -> {best[1][:-3]} "
                  f"({entry['best_speedup']:.2f}x, {entry['best_pct_floor']:.0f}% of floor)")

        results["shapes"][label] = {"K": K, "N": N, "floor_us": floor_us, "rows": rows}
        del w, wt, w32
        torch.cuda.empty_cache()

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
