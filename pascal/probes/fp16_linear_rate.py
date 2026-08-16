"""How close is cuBLAS fp16 to the memory roofline at this model's shapes?

Decode on a dense fp16 model is a bandwidth problem: every step reads every
weight exactly once and does two flops per byte-pair. So the only question that
matters is what fraction of the card's achievable bandwidth the GEMM call
reaches. Anything well below it is headroom for a hand-written kernel; anything
at it means decode is already done and the work belongs elsewhere.

The shapes are Qwen3-VL-Embedding-2B's text stack, which is what a decode step
actually touches:

    qkv   2048 -> 4096   (fused q 2048 + k 1024 + v 1024)
    o     2048 -> 2048
    gate_up 2048 -> 12288 (fused)
    down  6144 -> 2048
    lm_head 2048 -> 151936

    python pascal/probes/fp16_linear_rate.py
"""

from __future__ import annotations

import argparse
import json

import torch

# (label, K, N) — N is the output width, K the reduction depth, matching
# torch.nn.functional.linear(x[M, K], w[N, K]).
SHAPES = [
    ("qkv", 2048, 4096),
    ("o", 2048, 2048),
    ("gate_up", 2048, 12288),
    ("down", 6144, 2048),
    ("lm_head", 2048, 151936),
]

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]


def timeit(fn, warmup: int = 5, iters: int = 50) -> float:
    """Median-ish device time in seconds, via CUDA events."""
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


def achievable_bandwidth() -> dict:
    """What the card actually delivers, as opposed to what the spec sheet says.

    Three shapes of traffic, because a GEMV is a pure streaming read of the
    weight and the copy number (read + write) is the one usually quoted.
    """
    n = 256 * 1024 * 1024  # 512 MiB of fp16
    a = torch.randn(n, device="cuda", dtype=torch.float16)
    b = torch.empty_like(a)

    t_copy = timeit(lambda: b.copy_(a), iters=20)
    t_read = timeit(lambda: torch.sum(a, dtype=torch.float32), iters=20)

    return {
        "copy_GB_s": round(2 * a.numel() * 2 / t_copy / 1e9, 1),
        "read_GB_s": round(a.numel() * 2 / t_read / 1e9, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/fp16_linear_rate.json")
    ap.add_argument("--batches", type=int, nargs="+", default=BATCHES)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  sm_{props.major}{props.minor}  {props.total_memory / 2**30:.1f} GiB")

    bw = achievable_bandwidth()
    print(f"achievable bandwidth: copy {bw['copy_GB_s']} GB/s   read {bw['read_GB_s']} GB/s")
    ceiling = bw["read_GB_s"]

    results = {"device": props.name, "bandwidth": bw, "shapes": {}}

    print()
    print(f"{'shape':<9} {'M':>4} {'us':>9} {'GB/s':>8} {'%roof':>7} {'TFLOP/s':>9}")
    print("-" * 52)
    for label, K, N in SHAPES:
        w = torch.randn(N, K, device="cuda", dtype=torch.float16)
        wbytes = N * K * 2
        rows = []
        for m in args.batches:
            x = torch.randn(m, K, device="cuda", dtype=torch.float16)
            t = timeit(lambda: torch.nn.functional.linear(x, w))
            gbs = wbytes / t / 1e9
            tflops = 2 * m * N * K / t / 1e12
            rows.append(
                {
                    "M": m,
                    "us": round(t * 1e6, 2),
                    "GB_s": round(gbs, 1),
                    "pct_roofline": round(100 * gbs / ceiling, 1),
                    "TFLOP_s": round(tflops, 3),
                }
            )
            print(
                f"{label:<9} {m:>4} {t * 1e6:>9.2f} {gbs:>8.1f} {100 * gbs / ceiling:>6.1f}% {tflops:>9.3f}"
            )
        results["shapes"][label] = {"K": K, "N": N, "rows": rows}
        del w
        torch.cuda.empty_cache()

    # The number that matters: a whole decode step's weight traffic, if every
    # layer ran at the rate measured above.
    print()
    per_layer = 2048 * 4096 + 2048 * 2048 + 2048 * 12288 + 6144 * 2048
    decode_params = 28 * per_layer + 2048 * 151936
    decode_bytes = decode_params * 2
    print(f"decode-resident weights: {decode_params / 1e9:.3f} G params  "
          f"= {decode_bytes / 1e9:.3f} GB")
    print(f"roofline decode at batch 1: {ceiling * 1e9 / decode_bytes:.1f} tok/s")
    results["decode_bytes"] = decode_bytes
    results["roofline_tok_s"] = round(ceiling * 1e9 / decode_bytes, 1)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
