"""Price the TP=2 all-reduce at the shapes a batch-1 decode actually uses.

The 3090 Ti pair has no P2P (`nvidia-smi topo -p2p r` reports NS both ways), no
NVLink bridge fitted, and sm_86 has no symmetric-memory support, so vLLM's
custom all-reduce, QUICK_REDUCE and SYMM_MEM paths are all unavailable and every
reduction falls through to PYNCCL across the host bridge. The engine says so at
startup:

    SymmMemCommunicator: Device capability 8.6 not supported
    Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'
      out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER',
      'AITER_CUSTOM', 'CUSTOM', 'SYMM_MEM', 'PYNCCL']

A tensor-parallel decode does two all-reduces per layer. On the 27B that is
2 x 64 = 128 per forward, and MTP with three speculative tokens runs four
forwards per step, so a step pays for up to 512 of them. At batch 1 each one
carries hidden_size * dtype bytes -- 10 KB in bf16 -- which is far too small to
amortise anything, so this is a pure latency question, not a bandwidth one.

The number that matters is therefore microseconds per small all-reduce, not
GB/s. This probe measures it at the real shapes and multiplies out.

    torchrun --nproc-per-node 2 ampere/probes/allreduce_rate.py
    python ampere/probes/allreduce_rate.py            # spawns two ranks itself

Needs both cards, so it cannot run while a TP=2 deployment holds them.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Qwen3.5-27B: hidden 5120, 64 layers, 2 all-reduces per layer, 4 forwards per
# MTP step at num_speculative_tokens=3.
HIDDEN = 5120
LAYERS = 64
REDUCES_PER_LAYER = 2
FORWARDS_PER_MTP_STEP = 4


def bench(rank: int, world: int, args: argparse.Namespace) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"world={world}  device={torch.cuda.get_device_name(0)}")
        head = f"{'tokens':>8} {'bytes':>10} {'us p50':>9} {'us p99':>9}"
        print(f"{head} {'ms/MTP step':>13}")
        print("-" * 54)

    for tokens in [int(t) for t in args.token_counts.split(",")]:
        buf = torch.ones(tokens, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{rank}")

        for _ in range(args.warmup):
            dist.all_reduce(buf)
        torch.cuda.synchronize()

        # CUDA events rather than wall clock: at ~10 KB the reduction is short
        # enough that Python overhead would dominate a perf_counter reading.
        samples = []
        for _ in range(args.iters):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            dist.all_reduce(buf)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0)  # ms -> us

        samples.sort()
        p50 = statistics.median(samples)
        p99 = samples[min(len(samples) - 1, int(0.99 * len(samples)))]
        per_step_ms = p50 * LAYERS * REDUCES_PER_LAYER * FORWARDS_PER_MTP_STEP / 1000.0

        dist.barrier()
        if rank == 0:
            print(
                f"{tokens:>8} {tokens * HIDDEN * 2:>10} "
                f"{p50:>9.1f} {p99:>9.1f} {per_step_ms:>13.2f}"
            )

    if rank == 0:
        print(
            f"\n'ms/MTP step' extrapolates the batch-1 figure to "
            f"{LAYERS * REDUCES_PER_LAYER * FORWARDS_PER_MTP_STEP} reductions "
            "-- 2 per layer,\n"
            f"{LAYERS} layers, {FORWARDS_PER_MTP_STEP} forwards per step. "
            "Compare it against the 62.5 ms/token\nbaseline: if it is a small "
            "fraction, the interconnect is not the problem and an\nNVLink "
            "bridge would not repay itself."
        )
    dist.destroy_process_group()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--token-counts",
        default="1,4,8,32,256,2048",
        help="1 is batch-1 decode, 4 is an MTP verify batch, the large ones "
        "are prefill chunks where bandwidth starts to matter",
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    if torch.cuda.device_count() < 2:
        print(
            f"needs 2 GPUs, found {torch.cuda.device_count()}. This hypothesis "
            "is about the link between the pair and cannot be probed on one "
            "card.",
            file=sys.stderr,
        )
        return 1

    if "RANK" in os.environ:  # launched by torchrun
        bench(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), args)
        return 0

    mp.spawn(bench, args=(2, args), nprocs=2, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
