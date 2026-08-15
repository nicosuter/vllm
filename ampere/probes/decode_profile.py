"""Break a Qwen3.5-27B decode step down by CUDA kernel, across both ranks.

The ampere README's headline number -- 62.5 ms inter-token, against a ~4 ms/token
bandwidth roofline -- is a 15x that nothing on the hypothesis list explains. H1
(idle clocks) was measured false. H2 (the CUDA-graph downgrade) is worth 1.42x on
a proxy. H3 (all-reduce over the host bridge) has since been measured at roughly
4% of the step, which retires it: halving a 4% cost buys 2%. So the gap is still
unaccounted for, and the 62.5 ms itself is known-contaminated -- 230 samples, all
of them four-token health-check generations against an idle engine, short enough
that per-request fixed costs dominate.

This replaces it with a number taken under sustained decode on the pair, and
splits that number by kernel.

Two things it measures that a kernel table alone cannot.

**Idle time.** Summing kernel durations answers "which kernel is slowest" but not
"is the card even busy". At batch 1 across 64 layers, with TP=2 adding a second
process worth of launch overhead and MTP adding three draft forwards per step,
the plausible failure mode is that every kernel is fast and the GPU spends the
step waiting. That world wants graph coverage, not a better kernel, and the two
are indistinguishable in a table of summed durations. So this walks the trace and
reports the union of busy intervals against the wall-clock span.

**Both ranks.** vLLM's profiler is constructed per worker
(`vllm/profiler/wrapper.py`, instantiated in the engine-core process), so TP=2
writes one trace per rank. A rank that spends its time blocked in an all-reduce
waiting for the other rank looks busy in NCCL and idle everywhere else, and
reading only rank 0 hides that. Both are summarized and their occupancy compared.

    python ampere/probes/decode_profile.py --model /models/awq --tp 2

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

FILLER = (
    "The allocator groups objects by size class, which keeps fragmentation "
    "bounded but costs a lookup on every free. Compaction runs opportunistically "
    "when a class crosses its occupancy threshold, and the collector treats a "
    "partially compacted class as immovable until the cycle completes. "
)

# Substring -> bucket, first match wins. Order is load-bearing twice over:
# the NCCL kernels are matched before anything else so a collective is never
# folded into a compute bucket, and the GDN-specific names are matched before
# the generic "attn" that would otherwise swallow them.
BUCKETS: list[tuple[str, str]] = [
    ("nccl", "all-reduce (NCCL)"),
    ("allreduce", "all-reduce (NCCL)"),
    ("all_reduce", "all-reduce (NCCL)"),
    # Gated DeltaNet linear attention: 3 of every 4 layers. Triton kernels out
    # of the vendored flash-linear-attention tree, plus the conv that feeds them.
    ("causal_conv1d", "GDN linear attention"),
    ("chunk_gated", "GDN linear attention"),
    ("chunk_delta", "GDN linear attention"),
    ("delta_rule", "GDN linear attention"),
    ("recurrent", "GDN linear attention"),
    ("fused_gdn", "GDN linear attention"),
    ("gdn", "GDN linear attention"),
    # Full attention: every 4th layer.
    ("flash", "full attention"),
    ("unified_attention", "full attention"),
    ("paged", "full attention"),
    ("attn", "full attention"),
    # AWQ-INT4 backbone. This is the bucket the GEMV thesis is about.
    ("marlin", "linear (AWQ-INT4 Marlin)"),
    ("awq", "linear (AWQ-INT4 Marlin)"),
    ("gptq", "linear (AWQ-INT4 Marlin)"),
    ("rms_norm", "norms + RoPE"),
    ("rotary", "norms + RoPE"),
    ("norm", "norms + RoPE"),
    # The vocab projection is a TP-sharded 248320x5120 bf16 tensor -- not
    # quantized, so it lands in cuBLAS rather than Marlin.
    ("gemm", "lm_head / logits"),
    ("cutlass", "lm_head / logits"),
    ("gemv", "lm_head / logits"),
    ("logits", "lm_head / logits"),
    ("sampl", "sampling"),
    ("argmax", "sampling"),
    ("softmax", "sampling"),
    ("topk", "sampling"),
    ("elementwise", "elementwise / misc"),
    ("vectorized", "elementwise / misc"),
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
    if target_tokens <= 0:
        return PROMPT
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    text = PROMPT + " "
    while len(tok(text).input_ids) < target_tokens:
        text += FILLER
    print(f"synthesized prompt: {len(tok(text).input_ids)} tokens")
    return text


def run_profile(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    prompt = build_prompt(args.prompt_tokens, args.model)

    kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_frac,
        kv_cache_dtype=args.kv_cache_dtype,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=False,
        trust_remote_code=True,
        disable_custom_all_reduce=True,
        profiler_config={"profiler": "torch", "torch_profiler_dir": args.out_dir},
    )
    # The deployment runs MTP with 3 draft tokens. Profiling without it would
    # measure a model nobody serves, but it is worth being able to turn off:
    # with spec decode on, one "step" is three draft forwards plus a target
    # forward, and attributing device time per generated token gets murky.
    if args.spec_tokens > 0:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.spec_tokens,
        }

    llm = LLM(**kwargs)

    # Warm up outside the profile: Triton JIT for the GDN kernels is large
    # enough to swamp every real kernel in the trace if it lands inside it.
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
    occupancy -- the opposite of the error that matters here, where the question
    is how much of the step the card did nothing at all.
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


def load_device_events(trace_path: str):
    opener = gzip.open if trace_path.endswith(".gz") else open
    with opener(trace_path, "rt") as fh:
        trace = json.load(fh)

    by_name: dict[str, float] = collections.defaultdict(float)
    counts: dict[str, int] = collections.defaultdict(int)
    intervals: list[tuple[float, float]] = []
    for ev in trace.get("traceEvents", []):
        if ev.get("cat") not in DEVICE_CATS:
            continue
        dur = float(ev.get("dur", 0.0))
        ts = float(ev.get("ts", 0.0))
        by_name[ev.get("name", "?")] += dur
        counts[ev.get("name", "?")] += 1
        intervals.append((ts, ts + dur))
    return by_name, counts, intervals


def summarize_one(trace_path: str, top: int, tokens: int, tag: str) -> dict | None:
    by_name, counts, intervals = load_device_events(trace_path)
    total = sum(by_name.values())
    if total == 0.0:
        print(f"[{tag}] trace contains no device events", file=sys.stderr)
        return None

    launches = sum(counts.values())
    span = max(e for _, e in intervals) - min(s for s, _ in intervals)
    busy = union_busy(list(intervals))

    print(f"\n{'#' * 96}\n# {tag}: {trace_path}\n{'#' * 96}")
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

    print("\n" + "-" * 96)
    print(f"{'wall span':>22}: {span / 1000:9.2f} ms")
    print(
        f"{'device busy (union)':>22}: {busy / 1000:9.2f} ms  {100 * busy / span:5.1f}%"
    )
    print(
        f"{'device idle':>22}: {(span - busy) / 1000:9.2f} ms  "
        f"{100 * (span - busy) / span:5.1f}%"
    )
    if tokens > 0:
        print(f"\nper generated token ({tokens} tokens):")
        print(f"{'wall':>22}: {span / tokens / 1000:9.3f} ms")
        print(f"{'busy':>22}: {busy / tokens / 1000:9.3f} ms")
        print(f"{'idle':>22}: {(span - busy) / tokens / 1000:9.3f} ms")
        print(f"{'launches':>22}: {launches / tokens:9.1f}")

    return {"tag": tag, "span": span, "busy": busy, "agg": dict(agg), "total": total}


def summarize(out_dir: str, top: int, tokens: int) -> int:
    traces = sorted(glob.glob(os.path.join(out_dir, "*.pt.trace.json*")))
    if not traces:
        print(f"no trace written to {out_dir}", file=sys.stderr)
        return 1

    results = []
    for i, path in enumerate(traces):
        r = summarize_one(path, top, tokens, tag=f"rank {i}")
        if r:
            results.append(r)

    if not results:
        return 1

    if len(results) > 1:
        print("\n" + "=" * 96)
        print("cross-rank occupancy")
        print("-" * 96)
        print(f"{'rank':>8} {'busy %':>9} {'all-reduce %':>14}")
        for r in results:
            ar = r["agg"].get("all-reduce (NCCL)", 0.0)
            print(
                f"{r['tag']:>8} {100 * r['busy'] / r['span']:8.1f}% "
                f"{100 * ar / r['total']:13.1f}%"
            )
        print(
            "\nA rank that is busy only in NCCL is waiting for its peer, not "
            "working. If the\ntwo ranks disagree sharply, the step is bounded by "
            "whichever is slower and the\nper-rank compute numbers cannot be read "
            "as the cost of the step."
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/awq")
    ap.add_argument("--out-dir", default="/work/profile")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--spec-tokens", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt-tokens", type=int, default=0)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=6)
    ap.add_argument("--kv-cache-dtype", default="fp8")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.summarize_only:
        run_profile(args)
    return summarize(args.out_dir, args.top, args.max_tokens)


if __name__ == "__main__":
    sys.exit(main())
