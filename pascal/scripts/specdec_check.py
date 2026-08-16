"""Speculative decoding on Pascal: does it agree, and what does it actually buy?

Speculative decoding is the one algorithmic lever that gets past the memory
roofline at batch 1, because a decode step reads all 3.44 GB of weights once
whatever M is. Verifying k drafted tokens is one weight read, not k. What made
it not worth having on this card before is that cuBLAS charged 2.5x for M=4
(see `csrc/libtorch_stable/quantization/pascal_skinny_gemm.cu`); with that fixed,
M=4 costs 1.11x of M=1 and the arithmetic changes completely.

Two things have to be measured separately, and conflating them is how
speculative decoding gets oversold:

  * **Cost** -- milliseconds per forward step at each k. This is a property of
    the engine and the card, it is what this fork can improve, and it is
    workload-independent.
  * **Acceptance** -- how many of the k drafted tokens survive verification.
    This is a property of the *workload and model*, not of the engine, and it
    is where a flattering benchmark hides. A model stuck in a degenerate loop
    accepts everything; that number means nothing.

Throughput is the product, so this reports all three and never the last alone.

Correctness first. With greedy sampling, speculative decoding is an exactness
claim: the output must be **token-for-token identical** to running without it,
because rejection sampling only accepts a draft token when it is the one the
target model would have produced. Any divergence is a bug in the rejection
sampler, which on this card means a Pascal bug.

    python pascal/scripts/specdec_check.py --model /work/models/Qwen3.5-2B-AWQ-4bit
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

# Two workloads, because they bracket what acceptance can be.
#
# `copy` is the friendly end: the answer is largely present in the prompt, which
# is exactly the case n-gram drafting was built for.
#
# `open` is the honest end: continuation with nothing to copy from, where an
# n-gram drafter has little to go on and speculative decoding has to pay its
# overhead out of a thinner acceptance rate.
DOC = (
    "The Rhaetian Railway operates a metre-gauge network in the Swiss canton of "
    "Graubunden. Its Albula and Bernina lines were inscribed as a UNESCO World "
    "Heritage Site in 2008. The Landwasser Viaduct, completed in 1902, curves "
    "directly into a tunnel portal in the rock face. The Bernina line climbs to "
    "2253 metres at Ospizio Bernina without rack assistance, which makes it one "
    "of the steepest adhesion railways in the world."
)

WORKLOADS = {
    "copy": (
        "Answer using the exact wording of the passage.\n\n"
        + DOC
        + "\n\nQuestion: Describe the Landwasser Viaduct and the Bernina line, "
        "quoting the passage exactly.\nAnswer:"
    ),
    "open": (
        "Write a detailed technical explanation of how a modern operating system "
        "kernel schedules threads across multiple CPU cores, covering run queues, "
        "load balancing, priority inversion and the tradeoffs between throughput "
        "and latency in the design of a preemptive scheduler."
    ),
}


def build_llm(model: str, k: int, gpu_frac: float, dtype: str, method: str,
              draft_model: str | None, enforce_eager: bool):
    from vllm import LLM

    kwargs = {}
    if k > 0:
        cfg: dict = {"method": method, "num_speculative_tokens": k}
        if method == "ngram":
            # A 2-token match is the shortest that is worth trusting; longer
            # windows raise precision and lower hit rate.
            cfg.update({"prompt_lookup_min": 2, "prompt_lookup_max": 4})
        if draft_model:
            cfg["model"] = draft_model
        kwargs["speculative_config"] = cfg

    return LLM(
        model=model,
        dtype=dtype,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_frac,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        # get_metrics() asserts on this. Without it the acceptance rate comes
        # back None, and a speculative-decoding speedup without its acceptance
        # rate is not a measurement.
        disable_log_stats=False,
        **kwargs,
    )


def spec_counters(llm) -> dict:
    """Drafts, drafted tokens and accepted tokens, from the engine's own metrics.

    Returned as cumulative totals; the caller differences them around a run.
    """
    wanted = {
        "vllm:spec_decode_num_drafts": "drafts",
        "vllm:spec_decode_num_draft_tokens": "draft_tokens",
        "vllm:spec_decode_num_accepted_tokens": "accepted_tokens",
    }
    out = {v: 0.0 for v in wanted.values()}
    try:
        for metric in llm.get_metrics():
            name = getattr(metric, "name", "")
            if name in wanted:
                value = getattr(metric, "value", None)
                if value is None:  # counters may arrive as sample lists
                    samples = getattr(metric, "samples", []) or []
                    value = sum(getattr(s, "value", 0.0) for s in samples)
                out[wanted[name]] = float(value)
    except Exception as exc:  # loud: the acceptance rate is half the result
        print(f"  [warn] could not read spec-decode metrics: "
              f"{type(exc).__name__}: {exc}", flush=True)
    return out


def generate(llm, prompt: str, n_tokens: int):
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0.0, max_tokens=n_tokens, min_tokens=n_tokens, ignore_eos=True
    )
    return llm.generate([prompt], params, use_tqdm=False)


def measure(llm, prompt: str, short: int, long: int, reps: int) -> dict:
    """Seconds per output token, by the same slope trick the rest of the fork uses.

    The slope survives speculative decoding unchanged: it divides wall time by
    *output tokens*, and stays honest however many forward steps produced them.
    What it no longer measures is a forward step, so the field is not called one.
    """
    generate(llm, prompt, short)  # discard: per-shape JIT and any deferred capture

    def timed(n):
        times = []
        for _ in range(reps):
            start = time.perf_counter()
            outs = generate(llm, prompt, n)
            times.append(time.perf_counter() - start)
        return statistics.median(times), outs[0].outputs[0].token_ids

    before = spec_counters(llm)
    t_short, _ = timed(short)
    t_long, tokens = timed(long)
    after = spec_counters(llm)

    per_token = (t_long - t_short) / (long - short)
    drafts = after["drafts"] - before["drafts"]
    draft_tokens = after["draft_tokens"] - before["draft_tokens"]
    accepted = after["accepted_tokens"] - before["accepted_tokens"]

    return {
        "decode_tok_s": round(1.0 / per_token, 2),
        "ms_per_token": round(per_token * 1000, 3),
        "drafts": int(drafts),
        "draft_tokens": int(draft_tokens),
        "accepted_tokens": int(accepted),
        # Per-token acceptance, and the mean number of tokens a forward step
        # yields -- 1.0 means speculative decoding bought nothing at all.
        "acceptance_rate": round(accepted / draft_tokens, 3) if draft_tokens else None,
        "accepted_per_draft": round(accepted / drafts, 3) if drafts else None,
        "tokens": list(tokens),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", default="ngram",
                    choices=["ngram", "suffix", "draft_model", "mtp"])
    ap.add_argument("--draft-model", default=None)
    ap.add_argument("--spec-tokens", type=int, nargs="+", default=[0, 1, 2, 3, 5],
                    help="k values to measure; 0 is the no-speculation baseline")
    ap.add_argument("--workloads", nargs="+", default=["copy", "open"],
                    choices=sorted(WORKLOADS))
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--short", type=int, default=16)
    ap.add_argument("--long", type=int, default=112)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--out", default="/work/specdec_check.json")
    args = ap.parse_args()

    results: dict = {"model": args.model, "method": args.method, "by_k": {}}
    baseline_tokens: dict[str, list[int]] = {}

    for k in args.spec_tokens:
        print(f"\n{'=' * 72}\nnum_speculative_tokens={k}"
              f"{' (baseline)' if k == 0 else ''}\n{'=' * 72}", flush=True)
        entry: dict = {}
        llm = None
        try:
            llm = build_llm(args.model, k, args.gpu_frac, args.dtype, args.method,
                            args.draft_model, args.enforce_eager)
            for name in args.workloads:
                row = measure(llm, WORKLOADS[name], args.short, args.long, args.reps)
                tokens = row.pop("tokens")
                if k == 0:
                    baseline_tokens[name] = tokens
                    row["matches_baseline"] = True
                else:
                    row["matches_baseline"] = tokens == baseline_tokens.get(name)
                    if not row["matches_baseline"]:
                        base = baseline_tokens.get(name) or []
                        first = next(
                            (i for i, (a, b) in enumerate(zip(tokens, base)) if a != b),
                            min(len(tokens), len(base)),
                        )
                        row["first_divergence"] = first
                # A model looping on a few tokens accepts every draft, which
                # makes any speedup a measurement of the loop. Flag it rather
                # than quietly reporting a flattering number.
                uniq = len(set(tokens))
                row["distinct_tokens"] = uniq
                row["degenerate"] = uniq <= max(4, len(tokens) // 10)
                entry[name] = row
                print(f"  {name:<5} {row['decode_tok_s']:>7.2f} tok/s  "
                      f"accept={row['acceptance_rate']}  "
                      f"tokens/step={row['accepted_per_draft']}  "
                      f"identical={row['matches_baseline']}"
                      f"{'  DEGENERATE OUTPUT' if row['degenerate'] else ''}",
                      flush=True)
        except Exception as exc:  # a k that will not run is a result
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {entry['error'][:160]}", flush=True)
        finally:
            del llm
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        results["by_k"][k] = entry

    print(f"\n{'=' * 72}\nSUMMARY  ({args.method})\n{'=' * 72}")
    print(f"{'k':>3} {'workload':<6} {'tok/s':>8} {'vs k=0':>8} {'accept':>7} "
          f"{'tok/step':>9} {'identical':>10} note")
    exact = True
    for name in args.workloads:
        base = results["by_k"].get(0, {}).get(name, {}).get("decode_tok_s")
        for k in args.spec_tokens:
            row = results["by_k"].get(k, {}).get(name)
            if not row:
                continue
            speedup = f"{row['decode_tok_s'] / base:.2f}x" if base else "-"
            if not row.get("matches_baseline", True):
                exact = False
            print(f"{k:>3} {name:<6} {row['decode_tok_s']:>8.2f} {speedup:>8} "
                  f"{str(row['acceptance_rate']):>7} "
                  f"{str(row['accepted_per_draft']):>9} "
                  f"{str(row.get('matches_baseline')):>10} "
                  f"{'degenerate' if row.get('degenerate') else ''}")

    results["greedy_exact"] = exact
    print(f"\ngreedy output identical to the k=0 baseline everywhere: {exact}")
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {args.out}")
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
