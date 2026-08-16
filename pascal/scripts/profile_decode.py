"""Break a decode step down by CUDA kernel, to decide where optimisation goes.

After the fp32 rewrite of exllama's dot22_8_f, the quantized linear layers are
no longer obviously the bottleneck, and the next move (dp4a? a better attention
kernel? nothing?) depends on what is actually consuming the remaining time.
Guessing from FLOP counts is how the fp16 problem stayed hidden for as long as
it did, so this measures instead.

vLLM v1 runs the model in a separate EngineCore process, so wrapping generate()
in torch.profiler locally captures nothing; the engine's own profiler hooks have
to be used. Note that VLLM_TORCH_PROFILER_DIR no longer enables them -- the
worker now refuses start_profile() unless profiler_config is passed to the
engine.

    python pascal/scripts/profile_decode.py --model /work/models/Qwen3.5-2B-AWQ-4bit
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import sys

PROMPT = "Explain how a CPU scheduler decides which thread to run next."


def run_profile(model: str, out_dir: str, max_tokens: int,
                dtype: str = "auto", backend: str = "none",
                prompt_tokens: int = 0) -> None:
    if backend == "inductor":
        # Same lift the probe uses, so there is one definition of it.
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probes"))
        from inductor_sm61 import lift_inductor_floor

        lift_inductor_floor()

    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    llm = LLM(
        model=model,
        # "auto", not "float16". On Pascal supported_dtypes is
        # [float16, float32], so auto still picks fp16 for ordinary models --
        # but vLLM refuses fp16 outright for the families in
        # _FLOAT16_NOT_SUPPORTED_MODELS (gemma2, gemma3, gemma3_text, glm4,
        # "numerical instability"), and for those auto falls back to fp32 while
        # a hardcoded "float16" raises ValueError instead. bf16 is never
        # selectable here, which is the whole point.
        dtype=dtype,
        enforce_eager=False,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        compilation_config=CompilationConfig(
            # Graph capture stays off in every arm: it collapses the kernel
            # boundaries this script exists to read.
            mode=(
                CompilationMode.NONE if backend == "none"
                else CompilationMode.VLLM_COMPILE
            ),
            backend=("" if backend == "none" else backend),
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        profiler_config={"profiler": "torch", "torch_profiler_dir": out_dir},
    )

    params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, min_tokens=max_tokens, ignore_eos=True
    )

    # A synthetic prompt of an exact length, for isolating prefill. Salted per
    # call because prefix caching would otherwise serve the profiled run from
    # the warm-up's blocks and the trace would contain no prefill at all.
    def make_prompt(salt: int):
        if not prompt_tokens:
            return PROMPT
        head = [1000 + ((salt * 7919 + i) % 50000) for i in range(16)]
        return {"prompt_token_ids": head + [100] * (prompt_tokens - 16)}

    # Warm up outside the profile so Triton's per-shape JIT does not land in the
    # trace and swamp every real kernel.
    llm.generate([make_prompt(1)], SamplingParams(temperature=0.0, max_tokens=8),
                 use_tqdm=False)

    llm.start_profile()
    llm.generate([make_prompt(2)], params, use_tqdm=False)
    llm.stop_profile()

    # The worker flushes the trace on stop_profile, but the write is async.
    del llm


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

    # 'kernel' is the actual device execution; 'gpu_memcpy'/'gpu_memset' are
    # separated out because they are bandwidth, not compute, and the two want
    # different fixes.
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

    print(f"total device time: {total / 1000:.1f} ms across {sum(counts.values())} launches\n")
    print(f"{'us total':>12} {'%':>6} {'count':>7}  kernel")
    print("-" * 100)
    for name, dur in sorted(by_name.items(), key=lambda kv: -kv[1])[:top]:
        print(f"{dur:12.0f} {100 * dur / total:5.1f}% {counts[name]:7d}  {name[:70]}")

    # Buckets, because "which kernel is slowest" is rarely the question. What
    # matters is how much of the step is the linear layers -- whose floor is
    # known exactly, being the weight bytes over the memory bandwidth -- and how
    # much is the tail around them, which is where the batch-1 gap to llama.cpp
    # lives once the GEMM is at the roofline.
    def bucket(pred) -> float:
        return sum(d for n, d in by_name.items() if pred(n.lower()))

    gemm = bucket(lambda n: "skinny_gemm" in n or "gemm" in n or "gemv" in n
                  or "gemm_half_q_half" in n or "cutlass" in n or "sgemm" in n)
    attn = bucket(lambda n: "attention" in n or "gated_delta" in n or "conv1d" in n)
    norm = bucket(lambda n: "rms_norm" in n or "layernorm" in n or "typeconvert" in n)
    rope = bucket(lambda n: "rope" in n or "rotary" in n)
    sample = bucket(lambda n: "gumbel" in n or "sampl" in n or "topk" in n
                    or "penalt" in n)
    tail = total - gemm - attn - norm - rope - sample

    print("\n" + "-" * 100)
    for label, val in (("linear layers (GEMM)", gemm), ("attention", attn),
                       ("norms", norm), ("rope", rope), ("sampling", sample),
                       ("everything else", tail)):
        print(f"{label:<24} {val / 1000:8.1f} ms  {100 * val / total:5.1f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="/work/profile")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="feed a synthetic prompt of exactly this many tokens; with "
        "--max-tokens 1 the trace is almost entirely prefill",
    )
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument(
        "--backend",
        default="none",
        choices=["none", "eager", "inductor"],
        help="compile backend to profile; inductor lifts the sm_70 floor first "
        "(see pascal/probes/inductor_sm61.py)",
    )
    ap.add_argument(
        "--summarize-only",
        action="store_true",
        help="skip the run and re-read the newest trace in --out-dir",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.summarize_only:
        run_profile(args.model, args.out_dir, args.max_tokens, args.dtype,
                    args.backend, args.prompt_tokens)
    return summarize(args.out_dir, args.top)


if __name__ == "__main__":
    sys.exit(main())
