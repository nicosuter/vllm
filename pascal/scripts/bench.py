"""Decode-throughput benchmark for the Pascal fork.

Exists to answer one question the v1 numbers left open: how much of the 11.7
tok/s is arithmetic and how much is kernel-launch overhead. Every v1 measurement
was taken with `--enforce-eager`, which disables torch.compile *and* CUDA
graphs. Those are separable, and only the first is actually impossible here:

  * `mode` selects torch.compile. Anything but NONE routes through inductor,
    which raises GPUTooOldForTriton below sm_70 regardless of what Triton
    itself supports.
  * `cudagraph_mode` selects graph capture, which is pure launch-overhead
    elimination and involves no compiler at all.

So the interesting configuration is mode=NONE with graphs on. PIECEWISE is not
reachable -- it needs splitting_ops from piecewise compilation, i.e. inductor --
which leaves FULL and FULL_DECODE_ONLY.

Throughput is measured by slope rather than by dividing tokens by wall time.
A single generate() call also pays prefill, sampler setup and detokenisation,
and on a card this slow those are not noise. Running two output lengths and
taking (t_long - t_short) / (n_long - n_short) cancels everything that does not
scale with the number of decode steps, which is exactly the quantity CUDA graphs
are supposed to move.

    python pascal/scripts/bench.py --model /work/models/Qwen3.5-2B-AWQ-4bit \
        --graphs none full_decode_only full
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

# Long enough to be a realistic prefill, short enough that prefill is not the
# whole run. Content is irrelevant; only its token count is.
PROMPT = (
    "Write a detailed technical explanation of how a modern operating system "
    "kernel schedules threads across multiple CPU cores, covering run queues, "
    "load balancing, priority inversion and the tradeoffs between throughput "
    "and latency in the design of a preemptive scheduler."
)


def build_llm(model: str, graphs: str, mtp: int, gpu_frac: float,
              dtype: str = "auto"):
    from vllm import LLM
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    # mode=NONE unconditionally: inductor cannot run on this hardware, so the
    # only variable under test is graph capture.
    compilation_config = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode={
            "none": CUDAGraphMode.NONE,
            "full": CUDAGraphMode.FULL,
            "full_decode_only": CUDAGraphMode.FULL_DECODE_ONLY,
        }[graphs],
    )

    kwargs = {}
    if mtp:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": mtp,
        }

    return LLM(
        model=model,
        # "auto", not "float16". On Pascal supported_dtypes is
        # [float16, float32], so auto still picks fp16 for ordinary models --
        # but vLLM refuses fp16 outright for the families in
        # _FLOAT16_NOT_SUPPORTED_MODELS (gemma2, gemma3, gemma3_text, glm4,
        # "numerical instability"), and for those auto falls back to fp32 while
        # a hardcoded "float16" raises ValueError instead. bf16 is never
        # selectable here, which is the whole point.
        dtype=dtype,
        # Not enforce_eager: that would force cudagraph_mode=NONE and overwrite
        # the setting this benchmark exists to vary.
        enforce_eager=False,
        gpu_memory_utilization=gpu_frac,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        compilation_config=compilation_config,
        **kwargs,
    )


def time_generate(llm, n_tokens: int, reps: int, batch: int = 1) -> float:
    """Median wall time for generating exactly n_tokens from `batch` prompts.

    The prompts are made distinct by prefix so prefix caching cannot collapse
    the batch into one sequence and flatter the result.
    """
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0.0,
        max_tokens=n_tokens,
        min_tokens=n_tokens,  # with ignore_eos, pins the step count exactly
        ignore_eos=True,
    )
    prompts = [f"({i}) {PROMPT}" for i in range(batch)]
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        times.append(time.perf_counter() - start)
        for out in outs:
            produced = len(out.outputs[0].token_ids)
            if produced != n_tokens:
                raise RuntimeError(f"expected {n_tokens} tokens, got {produced}")
    return statistics.median(times)


def time_prefill(llm, n_prompt_tokens: int, reps: int) -> float:
    """Median wall time to prefill n_prompt_tokens and emit exactly one token.

    Fed as token ids rather than text so the prompt length is exact; tokenising
    a string of "about the right size" would put the independent variable of the
    slope under the tokeniser's control.
    """
    from vllm import SamplingParams

    # Token 100 is arbitrary but safely inside every vocab here, and content
    # cannot matter: prefill cost depends on length, not on what was said.
    prompt = {"prompt_token_ids": [100] * n_prompt_tokens}
    params = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1, ignore_eos=True)

    times = []
    for _ in range(reps):
        start = time.perf_counter()
        llm.generate([prompt], params, use_tqdm=False)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def bench_prefill(llm, short: int, long: int, reps: int) -> dict:
    """Prefill cost per prompt token, isolated by the same slope trick.

    Prefill is not a rounding error even at short context: it runs a different
    kernel path entirely. Above MAX_Q_GEMM_ROWS (50) exllama stops using its
    fused GEMM and switches to reconstruct-then-cuBLAS, so decode numbers say
    nothing about it.
    """
    time_prefill(llm, short, 1)  # discard: per-shape JIT

    t_short = time_prefill(llm, short, reps)
    t_long = time_prefill(llm, long, reps)

    per_token = (t_long - t_short) / (long - short)
    return {
        "t_short_s": round(t_short, 4),
        "t_long_s": round(t_long, 4),
        "prefill_tok_s": round(1.0 / per_token, 1),
        "us_per_prompt_token": round(per_token * 1e6, 2),
    }


def bench_one(llm, short: int, long: int, reps: int, batch: int = 1) -> dict:
    # Discarded: the first call pays Triton's per-shape JIT and, with graphs on,
    # any capture that was deferred past init.
    time_generate(llm, short, 1, batch)

    t_short = time_generate(llm, short, reps, batch)
    t_long = time_generate(llm, long, reps, batch)

    # One "step" advances every sequence in the batch by one token, so the
    # slope is per-step regardless of batch, and throughput is batch/step.
    # This is the whole reason batching pays here: the step reads the weights
    # once no matter how many sequences share it.
    per_step = (t_long - t_short) / (long - short)
    return {
        "batch": batch,
        "t_short_s": round(t_short, 4),
        "t_long_s": round(t_long, 4),
        "decode_tok_s": round(batch / per_step, 2),
        "ms_per_step": round(per_step * 1000, 3),
        # Kept for comparability with the v1 README, which reported vLLM's own
        # end-to-end figure.
        "end_to_end_tok_s": round(batch * long / t_long, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--graphs",
        nargs="+",
        default=["none"],
        choices=["none", "full", "full_decode_only"],
        help="cudagraph modes to measure; one engine is built per mode",
    )
    ap.add_argument("--mtp", type=int, default=0)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--short", type=int, default=16)
    ap.add_argument("--long", type=int, default=144)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--prefill",
        action="store_true",
        help="also measure prefill cost per prompt token (different kernel path)",
    )
    ap.add_argument("--prefill-short", type=int, default=128)
    ap.add_argument("--prefill-long", type=int, default=1024)
    ap.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=[1],
        help="batch sizes to sweep; one engine serves them all",
    )
    ap.add_argument("--out", default="/work/bench_result.json")
    args = ap.parse_args()

    results = {}
    for graphs in args.graphs:
        print(f"\n{'=' * 72}\ncudagraph_mode={graphs}  mtp={args.mtp}\n{'=' * 72}",
              flush=True)
        try:
            llm = build_llm(args.model, graphs, args.mtp, args.gpu_frac, args.dtype)
            # One engine serves every batch size, since only the request shape
            # differs; rebuilding per batch would pay engine init each time.
            if len(args.batch) == 1:
                results[graphs] = bench_one(
                    llm, args.short, args.long, args.reps, args.batch[0]
                )
            else:
                results[graphs] = {
                    f"batch{b}": bench_one(llm, args.short, args.long, args.reps, b)
                    for b in args.batch
                }
            if args.prefill:
                results[graphs]["prefill"] = bench_prefill(
                    llm, args.prefill_short, args.prefill_long, args.reps
                )
            print(json.dumps(results[graphs], indent=1), flush=True)
        except Exception as exc:  # noqa: BLE001 - a mode failing is a result
            results[graphs] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"FAILED: {results[graphs]['error']}", flush=True)
        finally:
            # Each mode needs its own engine, and the card holds only one.
            llm = None
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    flat = {}
    for mode, r in results.items():
        if "error" in r or "decode_tok_s" in r:
            flat[mode] = r
        else:  # batch sweep: one row per batch size
            for key, sub in r.items():
                if isinstance(sub, dict) and "decode_tok_s" in sub:
                    flat[f"{mode}/{key}"] = sub
    for mode, r in flat.items():
        if "error" in r:
            print(f"  {mode:<18} FAILED  {r['error'][:90]}")
        else:
            line = (
                f"  {mode:<18} {r['decode_tok_s']:>7.2f} tok/s decode  "
                f"({r['ms_per_step']:.2f} ms/step)  "
                f"e2e {r['end_to_end_tok_s']:.2f} tok/s"
            )
            if "prefill" in r:
                line += (
                    f"  |  prefill {r['prefill']['prefill_tok_s']:.0f} tok/s "
                    f"({r['prefill']['us_per_prompt_token']:.1f} us/tok)"
                )
            print(line)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
