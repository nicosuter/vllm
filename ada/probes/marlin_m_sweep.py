"""Is Marlin's batch-1 shortfall a bad kernel, and what do extra rows cost?

In the profile, Gemma4's int4 Marlin kernels move ~722 GB/s while cuBLAS's fp16
GEMV manages ~958 on the same card. That looked like a 25% per-byte deficit and
the case for writing a batch-1 W4A16 GEMV. It is not. Measured directly, Marlin
reaches **867 GB/s against cuBLAS's 954** on the same access pattern -- 91%, not
75%. The kernel is close to fine and the in-model gap is launch granularity: the
model's launches read 10-13 MB each and carry a fixed per-launch cost.

The second question matters more. Holding weights fixed and sweeping M shows what
an extra row costs, which is the price of speculative decoding through the
quantized layers.

## Two traps, both walked into while writing this

**L2 is huge on a 4090 -- 72 MiB, larger than every weight matrix in Gemma4.**
Any weight smaller than that is served from cache on the second iteration of a
benchmark loop, and the numbers are nonsense: 26 MB of int4 "read" in 11.2 us is
2.3 TB/s, more than twice the card's HBM bandwidth. In the model those weights
still stream from HBM, because a layer's weights are evicted long before that
layer runs again. So the shapes here are deliberately much larger than the
model's. The cuBLAS control makes the boundary visible: it reports 1400-2238
GB/s below L2 and settles to 952-956 above it.

L2 size is queried, never hardcoded: a 4090 has 72 MiB and a 3090 Ti has 6 MiB,
so a constant for either mislabels the other. An earlier version assumed the
4090's and told the 3090 Ti that a 67 MB working set was cache-resident when it
was not.

**Synchronising per iteration measures the launch, not the kernel.** The first
version timed one call between two events and synchronised each time. It
reported a flat ~18 us for every shape including a 1.1 MB weight and a 6.5 MB
one -- a 6x difference in bytes with no difference in time. Launches are
asynchronous; queue many and divide.

    python ada/probes/marlin_m_sweep.py
"""

from __future__ import annotations

import argparse
import sys


# Anything at or below L2 is cache-resident under a benchmark loop and its
# bandwidth number is meaningless. This is emphatically not a constant: a 4090
# has 72 MiB, a 3090 Ti has 6 MiB, so hardcoding either mislabels the other.
# Query the device rather than assume.
def _l2_bytes() -> int:
    import torch

    return torch.cuda.get_device_properties(0).L2_cache_size


# Deliberately larger than any shape in Gemma4, to get past L2. K is the
# reduction dim, N the output dim.
HBM_SHAPES: list[tuple[int, int]] = [
    (4096, 65536),
    (4096, 131072),
    (8192, 131072),
]

# The M sweep runs on one HBM-bound shape.
SWEEP_SHAPE = (4096, 131072)
MS = [1, 2, 4, 8, 16, 32, 64]


def bench(fn, iters: int, warmup: int = 8, reps: int = 4) -> float:
    """Best per-call time in ms, amortising launch overhead across `iters`."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def main() -> int:
    import torch

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )
    from vllm.scalar_type import scalar_types

    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=32, help="Gemma4 uses 32")
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    dev = torch.device("cuda")
    dtype = torch.float16
    qtype = scalar_types.uint4b8  # symmetric int4
    ws = marlin_make_workspace_new(dev)
    l2 = _l2_bytes()
    print(f"{torch.cuda.get_device_name(0)}, L2 = {l2 / 1024**2:.0f} MiB")
    print("Shapes below are much larger than Gemma4's on purpose: smaller ones")
    print("sit in L2 and report bandwidth the memory system cannot deliver.\n")

    def quantized(K, N):
        b = torch.randn((K, N), dtype=dtype, device=dev) / K**0.5
        w_ref, q_w, s, g_idx, sort_idx, _ = marlin_quantize(
            b, qtype, args.group_size, act_order=False
        )
        del b, w_ref
        torch.cuda.empty_cache()
        return q_w, s, g_idx, sort_idx

    print("marlin int4, M=1")
    print(f"    {'K':>6} {'N':>7} {'MB':>8} {'us':>9} {'GB/s':>7}")
    for K, N in HBM_SHAPES:
        q_w, s, g_idx, sort_idx = quantized(K, N)
        a = torch.randn((1, K), dtype=dtype, device=dev)
        out = torch.empty((1, N), dtype=dtype, device=dev)
        nbytes = K * N / 2 + (K * N / args.group_size) * 2

        def call(a=a, out=out, q_w=q_w, s=s, g=g_idx, si=sort_idx, N=N, K=K):
            return ops.marlin_gemm(
                a,
                out,
                q_w,
                None,
                s,
                None,
                None,
                None,
                g,
                si,
                ws,
                qtype,
                1,
                N,
                K,
                is_k_full=True,
                use_atomic_add=False,
                use_fp32_reduce=True,
                is_zp_float=False,
            )

        ms = bench(call, args.iters)
        cached = " (IN L2 -- ignore)" if nbytes <= l2 else ""
        print(
            f"    {K:>6} {N:>7} {nbytes / 1e6:8.1f} {ms * 1000:9.1f} "
            f"{nbytes / 1e6 / ms:7.0f}{cached}"
        )
        del q_w, s
        torch.cuda.empty_cache()

    print("\ncublas fp16 control, M=1 -- shows where L2 stops flattering the result")
    print(f"    {'K':>6} {'N':>7} {'MB':>8} {'us':>9} {'GB/s':>7}")
    for K, N in [(4096, 8192), (4096, 32768), (4096, 65536)]:
        w = torch.randn((N, K), dtype=dtype, device=dev)
        a = torch.randn((1, K), dtype=dtype, device=dev)
        nbytes = K * N * 2

        def call(a=a, w=w):
            return torch.nn.functional.linear(a, w)

        ms = bench(call, args.iters)
        cached = " (IN L2 -- ignore)" if nbytes <= l2 else ""
        print(
            f"    {K:>6} {N:>7} {nbytes / 1e6:8.1f} {ms * 1000:9.1f} "
            f"{nbytes / 1e6 / ms:7.0f}{cached}"
        )
        del w
        torch.cuda.empty_cache()

    K, N = SWEEP_SHAPE
    q_w, s, g_idx, sort_idx = quantized(K, N)
    nbytes = K * N / 2 + (K * N / args.group_size) * 2
    print(f"\nrows are free up to? marlin int4, K={K} N={N}, {nbytes / 1e6:.0f} MB")
    print(f"    {'M':>4} {'us':>9} {'GB/s':>7} {'vs M=1':>8}")
    base = None
    for M in MS:
        a = torch.randn((M, K), dtype=dtype, device=dev)
        out = torch.empty((M, N), dtype=dtype, device=dev)

        def call(a=a, out=out, q_w=q_w, s=s, g=g_idx, si=sort_idx, M=M, N=N, K=K):
            return ops.marlin_gemm(
                a,
                out,
                q_w,
                None,
                s,
                None,
                None,
                None,
                g,
                si,
                ws,
                qtype,
                M,
                N,
                K,
                is_k_full=True,
                use_atomic_add=False,
                use_fp32_reduce=True,
                is_zp_float=False,
            )

        ms = bench(call, args.iters)
        base = base or ms
        print(f"    {M:>4} {ms * 1000:9.1f} {nbytes / 1e6 / ms:7.0f} {ms / base:7.2f}x")

    print(
        "\nFlat time across M means the kernel is paying for the weight read and\n"
        "nothing else, so every row after the first is free. That is the price of\n"
        "a speculative-decode draft token through the quantized layers."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
