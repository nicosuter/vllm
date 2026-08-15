"""A/B the forced-Triton default against a per-layer-kind backend split.

`Gemma4Config.verify_and_update_config` sets `attention_config.backend =
TRITON_ATTN` for the whole model whenever head_dim != global_head_dim and FA4 is
unavailable. On Gemma4-26B-A4B that is head_dim 256 for 25 sliding-window layers
and 512 for 5 full-attention layers, and FA4 is Hopper-only, so on an RTX 4090
all 30 layers go to Triton because 5 of them have to.

FlashAttention takes head_size <= 256 without FA4 (`flash_attn.py:171`), so the
25 sliding layers -- 83% of the model -- are eligible for it and are on Triton
only for their neighbours' sake. The upstream rationale is explicitly about
Hopper ("avoid the mixed FA3/FA4 penalty"), which is not the situation here.

No patch is needed to test this. `Gemma4Config` only assigns
`attention_config.backend`, and `selector.py` resolves `backend_per_kind` ahead
of `backend`, so a per-kind override reaches the sliding group while the full
group still falls back to the forced Triton. If the split wins, the patch is to
make Gemma4Config do this itself on non-Hopper; if it does not, the forcing is
fine and the hypothesis dies cheaply.

Measures decode and prefill separately, because they are not the same bet.
Sliding-window layers are capped at a 1024-token window, so their decode
attention is cheap regardless of kernel; prefill is where attention scales with
the prompt and where this should pay if it pays anywhere.

    python ada/probes/attn_backend_split.py --model /models/awq
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

RESULT_PREFIX = "ADA_PROBE_RESULT "

FILLER = (
    "A copy-on-write filesystem never overwrites a live block: it writes the "
    "new version elsewhere and re-points the tree, so a snapshot is just a "
    "retained root. "
)


def build_prompt(tokenizer, target_tokens: int) -> str:
    text = FILLER
    while len(tokenizer(text).input_ids) < target_tokens:
        text += FILLER
    return tokenizer.decode(tokenizer(text).input_ids[:target_tokens])


def child(args: argparse.Namespace) -> int:
    from vllm import LLM, SamplingParams

    kwargs: dict = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_frac,
        kv_cache_dtype=args.kv_cache_dtype,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=False,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        # Off deliberately. Production runs with it on, but here every timed
        # prefill would be served out of the previous run's blocks and the
        # measurement would be of the cache, not of the attention kernel under
        # test. Isolating the kernel is the whole point.
        enable_prefix_caching=False,
    )
    if args.split:
        # full_attention is left to fall through to whatever Gemma4Config
        # forced, so this arm changes exactly one thing: the 25 eligible
        # layers. Anything else would confound the comparison.
        kwargs["attention_config"] = {
            "backend_per_kind": {"sliding_window": "FLASH_ATTN"}
        }

    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()

    llm.generate(
        ["warm up the kernels"],
        SamplingParams(temperature=0.0, max_tokens=8),
        use_tqdm=False,
    )

    result: dict = {"split": args.split}

    # Prefill: one token out, so wall time is dominated by processing the
    # prompt. This is the arm the hypothesis actually predicts.
    for plen in [int(x) for x in args.prompt_lens.split(",")]:
        prompt = build_prompt(tokenizer, plen)
        one = SamplingParams(temperature=0.0, max_tokens=1)
        llm.generate([prompt], one, use_tqdm=False)  # shape warm-up, untimed
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            llm.generate([prompt], one, use_tqdm=False)
            times.append(time.perf_counter() - t0)
        times.sort()
        result[f"prefill_{plen}_ms"] = round(times[len(times) // 2] * 1000, 2)

    # Decode: fixed long generation from a short prompt.
    dparams = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        ignore_eos=True,
    )
    lat = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        out = llm.generate(
            ["Count slowly and describe each step."], dparams, use_tqdm=False
        )
        lat.append((time.perf_counter() - t0) / len(out[0].outputs[0].token_ids) * 1000)
    lat.sort()
    result["decode_ms_per_tok"] = round(lat[len(lat) // 2], 3)
    result["decode_tok_s"] = round(1000 / lat[len(lat) // 2], 1)

    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return 0


def run_one(args: argparse.Namespace, split: bool) -> dict:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--child",
        "--model",
        args.model,
        "--prompt-lens",
        args.prompt_lens,
        "--max-tokens",
        str(args.max_tokens),
        "--reps",
        str(args.reps),
        "--gpu-frac",
        str(args.gpu_frac),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
    ]
    if split:
        cmd.append("--split")

    label = "per-kind split" if split else "forced Triton (default)"
    print(f"==> {label}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    blob = proc.stdout + proc.stderr

    record: dict = {"label": label, "split": split}
    for line in blob.splitlines():
        if line.startswith(RESULT_PREFIX):
            record.update(json.loads(line[len(RESULT_PREFIX) :]))

    backends = sorted(
        set(
            re.findall(
                r"Using (?:AttentionBackendEnum\.)?([A-Z_]+)\b[^\n]*backend", blob
            )
        )
    )
    record["backends"] = backends

    # The split arm is only meaningful if FlashAttention actually appears. A
    # silently-ignored override measuring the default twice is the exact trap
    # the Qwen probes fell into.
    if split and not any("FLASH" in b for b in backends):
        record["error"] = f"split requested but backends resolved to {backends}"
        print(f"    REJECTED: {record['error']}", flush=True)
        return record
    if "decode_tok_s" not in record:
        record["error"] = f"child exited {proc.returncode}"
        tail = [ln for ln in blob.splitlines() if ln.strip()][-10:]
        print("    FAILED:\n      " + "\n      ".join(tail), flush=True)
        return record

    print(f"    backends={backends} decode={record['decode_tok_s']} tok/s", flush=True)
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/awq")
    ap.add_argument(
        "--prompt-lens",
        default="1024,4096,16384",
        help="prefill lengths; 4096 is near the production mean prompt",
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-num-batched-tokens", type=int, default=3072)
    ap.add_argument("--max-num-seqs", type=int, default=10)
    ap.add_argument("--kv-cache-dtype", default="fp8")
    ap.add_argument("--out", default="")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--split", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        return child(args)

    records = [run_one(args, False), run_one(args, True)]
    plens = [int(x) for x in args.prompt_lens.split(",")]

    print("\n" + "-" * 78)
    cols = "".join(f"{'pf' + str(p):>11}" for p in plens)
    print(f"{'arm':>24}{cols}{'decode tok/s':>14}")
    print("-" * 78)
    for r in records:
        if "error" in r:
            print(f"{r['label']:>24}{'FAILED':>11}")
            continue
        cells = "".join(f"{r.get(f'prefill_{p}_ms', 0):>11.1f}" for p in plens)
        print(f"{r['label']:>24}{cells}{r['decode_tok_s']:>14}")

    ok = [r for r in records if "error" not in r]
    if len(ok) == 2:
        base, split = ok
        print()
        for p in plens:
            b, s = base[f"prefill_{p}_ms"], split[f"prefill_{p}_ms"]
            print(f"  prefill {p:>6}: {b / s:.2f}x")
        print(f"  decode      : {split['decode_tok_s'] / base['decode_tok_s']:.2f}x")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(records, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
