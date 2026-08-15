"""Break a Gemma4 step down by CUDA kernel, and say how much of it is idle.

Gemma4-26B-A4B looks like a model where attention should matter: vLLM forces
every one of its 30 layers onto Triton attention because 5 of them have
head_dim 512, which FlashAttention cannot take without FA4 (Hopper only). The
obvious patch is to split the backends per layer kind and give the 25
head_dim-256 sliding-window layers FlashAttention back.

Whether that is worth writing depends on how much of a step attention actually
is, and the arithmetic says: at decode, possibly very little. The model is MoE
with 4B active parameters, but its embedding table is ``[262144, 2816]`` in fp16
and ``tie_word_embeddings`` is true, so the lm_head is 1.48 GB read in full on
every single decode step. Against roughly 0.80 GB of top-8-of-128 expert weights
and ~0.4 GB of attention projections, the output projection alone looks like over
half the traffic, and the 25 sliding layers are bounded to a 1024-token window
that keeps their attention cheap.

Two things this measures that a kernel table alone cannot.

**Idle time.** Summing kernel durations answers "which kernel is slowest" but not
"is the GPU even busy". A plausible failure mode at batch 1 is that every kernel
is fast and the card spends most of the step waiting for the CPU to launch the
next one. Those two worlds want opposite fixes -- a better kernel versus better
graph coverage -- and they are indistinguishable in a table of summed durations.
So this walks the trace and reports the union of busy intervals against the
wall-clock span, per step.

Note that the expert GEMMs are *not* a launch-count problem: MoE Marlin is a
grouped GEMM (`moe/marlin_moe_wna16/marlin_template.h` reads `expert_id` per
block), so `marlin_moe.py` issues exactly two `moe_wna16_marlin_gemm` calls per
layer regardless of top-k -- about 60 per token, not 240.

**Prefill separately from decode.** Attention scales with context and the batch-1
decode figure says nothing about it, which the previous version of this probe
noted and then did not act on. Rather than trying to segment one mixed trace,
run it twice:

    # decode-dominated: one prefill forward out of 129
    python ada/probes/decode_profile.py --model /models/awq --max-tokens 128

    # prefill-dominated: production's mean prompt, one decode forward
    python ada/probes/decode_profile.py --model /models/awq \
        --prompt-tokens 3926 --max-tokens 1 --label prefill

Buckets are matched by kernel name, which is fragile across versions; the
per-kernel table is printed too so a mis-bucketed kernel is visible rather than
silently folded into "everything else".
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import sys

PROMPT = (
    "Explain how a copy-on-write filesystem keeps snapshots cheap, and what "
    "that costs at read time."
)

# Filler for --prompt-tokens. Prose rather than a repeated token so the
# tokenizer produces a realistic ratio and the attention kernels see a normal
# distribution of positions.
FILLER = (
    "The allocator groups objects by size class, which keeps fragmentation "
    "bounded but costs a lookup on every free. Compaction runs opportunistically "
    "when a class crosses its occupancy threshold, and the collector treats a "
    "partially compacted class as immovable until the cycle completes. "
)

# Substring -> bucket. Order matters: first match wins, so the specific
# quantized-GEMM names are listed before the generic gemm/cutlass catch-all
# that would otherwise swallow them.
BUCKETS: list[tuple[str, str]] = [
    ("marlin", "MoE + linear (Marlin W4A16)"),
    ("moe", "MoE routing/gather"),
    ("topk", "MoE routing/gather"),
    ("unified_attention", "attention (Triton)"),
    ("attn", "attention (Triton)"),
    ("flash", "attention (FlashAttention)"),
    ("rms_norm", "norms + RoPE"),
    ("rotary", "norms + RoPE"),
    ("norm", "norms + RoPE"),
    ("embedding", "embedding / lm_head"),
    ("gemv", "embedding / lm_head"),
    # The vocab projection at batch 1 is a tall-skinny GEMM against the tied
    # 262144x2816 fp16 table; it lands in cuBLAS rather than Marlin because the
    # embedding is not quantized. This is the bucket the hypothesis is about.
    ("gemm", "embedding / lm_head"),
    ("cutlass", "embedding / lm_head"),
    ("sampl", "sampling"),
    ("softmax", "sampling"),
    ("memcpy", "memcpy/memset"),
    ("memset", "memcpy/memset"),
]

DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def bucket_of(name: str) -> str:
    low = name.lower()
    for needle, bucket in BUCKETS:
        if needle in low:
            return bucket
    return "everything else"


def build_prompt(target_tokens: int, model: str) -> str:
    """Grow FILLER until the tokenizer reports at least target_tokens."""
    if target_tokens <= 0:
        return PROMPT
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    text = PROMPT + " "
    while len(tok(text).input_ids) < target_tokens:
        text += FILLER
    n = len(tok(text).input_ids)
    print(f"synthesized prompt: {n} tokens (asked for {target_tokens})")
    return text


def run_profile(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    prompt = build_prompt(args.prompt_tokens, args.model)

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_frac,
        kv_cache_dtype=args.kv_cache_dtype,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=False,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        profiler_config={"profiler": "torch", "torch_profiler_dir": args.out_dir},
    )

    # Warm up outside the profile: Triton's per-shape JIT is large enough to
    # swamp every real kernel in the trace if it lands inside it. Warm up on the
    # same prompt so prefill hits the same shapes the measured run will.
    llm.generate(
        [prompt], SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False
    )

    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        ignore_eos=True,
    )
    llm.start_profile()
    llm.generate([prompt], params, use_tqdm=False)
    llm.stop_profile()
    del llm  # stop_profile writes asynchronously; teardown flushes it


def union_busy(intervals: list[tuple[float, float]]) -> float:
    """Total time at least one device event was in flight.

    Kernels on different streams overlap, so summing durations overstates
    occupancy -- which is the opposite of the error we care about here, where
    the question is how much of the step the card was doing nothing at all.
    """
    if not intervals:
        return 0.0
    intervals.sort()
    busy = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:
            busy += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    busy += cur_end - cur_start
    return busy


def summarize(out_dir: str, top: int, steps: int, label: str) -> int:
    traces = glob.glob(os.path.join(out_dir, "*.pt.trace.json*"))
    if not traces:
        print(f"no trace written to {out_dir}", file=sys.stderr)
        return 1
    trace_path = max(traces, key=os.path.getmtime)
    print(f"trace: {trace_path}\n")

    opener = gzip.open if trace_path.endswith(".gz") else open
    with opener(trace_path, "rt") as fh:
        trace = json.load(fh)

    by_name: dict[str, float] = collections.defaultdict(float)
    counts: dict[str, int] = collections.defaultdict(int)
    intervals: list[tuple[float, float]] = []
    total = 0.0
    first_ts = float("inf")
    last_ts = 0.0
    for ev in trace.get("traceEvents", []):
        if ev.get("cat") not in DEVICE_CATS:
            continue
        dur = float(ev.get("dur", 0.0))
        ts = float(ev.get("ts", 0.0))
        name = ev.get("name", "?")
        by_name[name] += dur
        counts[name] += 1
        intervals.append((ts, ts + dur))
        first_ts = min(first_ts, ts)
        last_ts = max(last_ts, ts + dur)
        total += dur

    if total == 0.0:
        print("trace contains no device events", file=sys.stderr)
        return 1

    launches = sum(counts.values())
    span = last_ts - first_ts
    busy = union_busy(intervals)
    idle = span - busy

    print(f"total device time: {total / 1000:.1f} ms across {launches} launches\n")

    print(f"{'us total':>11} {'%':>6} {'count':>7}  kernel")
    print("-" * 96)
    for name, dur in sorted(by_name.items(), key=lambda kv: -kv[1])[:top]:
        print(f"{dur:11.0f} {100 * dur / total:5.1f}% {counts[name]:7d}  {name[:64]}")

    agg: dict[str, float] = collections.defaultdict(float)
    agg_n: dict[str, int] = collections.defaultdict(int)
    for name, dur in by_name.items():
        b = bucket_of(name)
        agg[b] += dur
        agg_n[b] += counts[name]

    print("\n" + "=" * 96)
    print(f"{'ms':>10} {'%':>7} {'launches':>10}  bucket")
    print("-" * 96)
    for b, dur in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"{dur / 1000:10.1f} {100 * dur / total:6.1f}% {agg_n[b]:10d}  {b}")

    # The decisive table. If idle dominates, no kernel rewrite in the bucket
    # list above can pay for itself and the target is per-step launch overhead.
    print("\n" + "=" * 96)
    print(f"occupancy ({label})")
    print("-" * 96)
    print(f"{'wall span':>22}: {span / 1000:9.2f} ms")
    print(
        f"{'device busy (union)':>22}: {busy / 1000:9.2f} ms  {100 * busy / span:5.1f}%"
    )
    print(f"{'device idle':>22}: {idle / 1000:9.2f} ms  {100 * idle / span:5.1f}%")
    if steps > 0:
        print(f"\n{'per step':>22}: {steps} steps")
        print(f"{'wall':>22}: {span / steps / 1000:9.3f} ms")
        print(f"{'busy':>22}: {busy / steps / 1000:9.3f} ms")
        print(f"{'idle':>22}: {idle / steps / 1000:9.3f} ms")
        print(f"{'launches':>22}: {launches / steps:9.1f}")

    attn = agg.get("attention (Triton)", 0.0) + agg.get(
        "attention (FlashAttention)", 0.0
    )
    print(
        f"\nAttention is {100 * attn / total:.1f}% of device time in this run. Only "
        "25 of 30 layers are\neligible for the per-layer-kind backend split, so its "
        f"ceiling here is about {100 * attn * 25 / 30 / total:.1f}%.\nA "
        "decode-dominated run says nothing about prefill; use --prompt-tokens."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/awq")
    ap.add_argument("--out-dir", default="/work/profile")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="synthesize a prompt of about this many tokens; 0 uses the short one",
    )
    ap.add_argument("--label", default="decode")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-batched-tokens", type=int, default=3072)
    ap.add_argument("--max-num-seqs", type=int, default=10)
    ap.add_argument("--kv-cache-dtype", default="fp8")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.summarize_only:
        run_profile(args)
    # One prefill forward plus max_tokens-1 decode forwards; for the
    # decode-dominated run the prefill is a rounding error on the divisor.
    return summarize(args.out_dir, args.top, args.max_tokens, args.label)


if __name__ == "__main__":
    sys.exit(main())
