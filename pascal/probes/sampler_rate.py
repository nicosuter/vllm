"""Why does sampling one token cost 333 us on this card?

`profile_decode.py` puts `_gumbel_sample_kernel` at 0.344 ms of an 18.1 ms
decode step -- 1.9%, and the largest single item in the tail once the GEMM is at
the memory roofline. The work it does is one argmax over a 151,936-entry fp32
logit row: 608 KiB, which at this card's measured 223 GB/s is **2.7 us**. It is
taking 120x that.

At batch 1 the launch is grid=(1, 149) with BLOCK_SIZE=1024, so 149 blocks over
19 SMs. That is a single wave of very small blocks, and the shape of the answer
decides the fix: a launch-bound kernel wants a bigger block, a reduction-bound
one wants different warps, and a kernel that is simply slow on this architecture
wants replacing with torch.argmax on the greedy path.

    python pascal/probes/sampler_rate.py
"""

from __future__ import annotations

import argparse
import json

import torch

VOCAB = 151936


def timeit(fn, warmup: int = 10, iters: int = 100) -> float:
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
    ap.add_argument("--out", default="/work/sampler_rate.json")
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--vocab", type=int, default=VOCAB)
    args = ap.parse_args()

    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    nbytes = args.vocab * 4
    print(f"vocab {args.vocab}, {nbytes / 1024:.0f} KiB of fp32 logits per token")
    print(f"read-bound floor at 223 GB/s: {nbytes / 223e9 * 1e6:.1f} us\n")

    results = {}
    print(f"{'tokens':>7} {'impl':<24} {'us':>9} {'GB/s':>8}")
    print("-" * 52)
    for n in args.tokens:
        logits = torch.randn(n, args.vocab, device="cuda", dtype=torch.float32)
        idx_map = torch.zeros(n, device="cuda", dtype=torch.int32)
        temperature = torch.zeros(1, device="cuda", dtype=torch.float32)  # greedy
        seed = torch.zeros(1, device="cuda", dtype=torch.int64)
        pos = torch.zeros(n, device="cuda", dtype=torch.int64)

        rows = {}

        def record(label, fn):
            t = timeit(fn)
            rows[label] = {"us": round(t * 1e6, 2),
                           "GB_s": round(n * nbytes / t / 1e9, 1)}
            print(f"{n:>7} {label:<24} {t * 1e6:>9.2f} "
                  f"{n * nbytes / t / 1e9:>8.1f}")

        record("gumbel_sample", lambda: gumbel_sample(
            logits, idx_map, temperature, seed, pos, apply_temperature=True))
        record("torch.argmax", lambda: torch.argmax(logits, dim=-1))
        record("torch.max", lambda: torch.max(logits, dim=-1))

        # Agreement, since a faster argmax is only useful if it is the same one.
        g = gumbel_sample(logits, idx_map, temperature, seed, pos,
                          apply_temperature=True)
        a = torch.argmax(logits, dim=-1)
        agree = bool((g.flatten().long() == a.flatten().long()).all().item())
        rows["greedy_agrees_with_argmax"] = agree
        print(f"{'':>7} {'-> same token as argmax':<24} {str(agree):>9}")

        results[n] = rows
        del logits
        torch.cuda.empty_cache()
        print()

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
