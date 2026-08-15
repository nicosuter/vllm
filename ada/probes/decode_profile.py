"""Break a Gemma4 decode step down by CUDA kernel, to decide where work goes.

Gemma4-26B-A4B looks like a model where attention should matter: vLLM forces
every one of its 30 layers onto Triton attention because 5 of them have
head_dim 512, which FlashAttention cannot take without FA4 (Hopper only). The
obvious patch is to split the backends per layer kind and give the 25
head_dim-256 sliding-window layers FlashAttention back.

Whether that is worth writing depends on how much of a decode step attention
actually is, and the arithmetic says: possibly very little. The model is MoE
with 4B active parameters, but its embedding table is
``[262144, 2816]`` in fp16 and ``tie_word_embeddings`` is true, so the lm_head
is 1.48 GB read in full on every single decode step. Against roughly 0.80 GB of
top-8-of-128 expert weights and ~0.4 GB of attention projections, the output
projection alone looks like over half the traffic, and the 25 sliding layers are
bounded to a 1024-token window that keeps their attention cheap.

If that holds, the backend split is a prefill and long-context change, not a
decode-throughput one, and should be sold and measured as such. Guessing from
FLOP counts is how the Qwen work twice measured the wrong configuration, so this
measures instead.

    python ada/probes/decode_profile.py --model /models/awq

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


def bucket_of(name: str) -> str:
    low = name.lower()
    for needle, bucket in BUCKETS:
        if needle in low:
            return bucket
    return "everything else"


def run_profile(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

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
    # swamp every real kernel in the trace if it lands inside it.
    llm.generate(
        [PROMPT], SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False
    )

    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        ignore_eos=True,
    )
    llm.start_profile()
    llm.generate([PROMPT], params, use_tqdm=False)
    llm.stop_profile()
    del llm  # stop_profile writes asynchronously; teardown flushes it


def summarize(out_dir: str, top: int) -> int:
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
    total = 0.0
    for ev in trace.get("traceEvents", []):
        if ev.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = float(ev.get("dur", 0.0))
        name = ev.get("name", "?")
        by_name[name] += dur
        counts[name] += 1
        total += dur

    if total == 0.0:
        print("trace contains no device events", file=sys.stderr)
        return 1

    launches = sum(counts.values())
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

    attn = agg.get("attention (Triton)", 0.0) + agg.get(
        "attention (FlashAttention)", 0.0
    )
    print(
        f"\nAttention is {100 * attn / total:.1f}% of device time at batch 1. That is "
        "the ceiling on\nwhat a per-layer-kind backend split can return here, and "
        "only 25 of 30 layers are\neligible, so the realistic decode ceiling is "
        f"about {100 * attn * 25 / 30 / total:.1f}%. Re-run with a long prompt "
        "before\nconcluding anything about prefill, where attention scales with "
        "context and this\nbatch-1 figure does not apply."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/awq")
    ap.add_argument("--out-dir", default="/work/profile")
    ap.add_argument("--max-tokens", type=int, default=32)
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
    return summarize(args.out_dir, args.top)


if __name__ == "__main__":
    sys.exit(main())
