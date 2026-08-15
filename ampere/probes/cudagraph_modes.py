"""Measure what FlashInfer's CUDA-graph downgrade costs under MTP spec decode.

On sm_86 the engine picks FlashInfer for full attention and then, because spec
decode is on, quietly drops from FULL_AND_PIECEWISE to PIECEWISE:

    CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
    attention backend FlashInferBackend (support: UNIFORM_SINGLE_TOKEN_DECODE);
    setting cudagraph_mode=PIECEWISE

That is not a tunable. `FlashInferMetadataBuilder.get_cudagraph_support` only
returns `UNIFORM_BATCH` — the level `vllm/config/compilation.py` requires to
keep full graphs when `uniform_decode_query_len > 1` — if `can_use_trtllm_attention`
holds, and trtllm-gen is Hopper/Blackwell only. So on consumer Ampere,
FlashInfer plus spec decode structurally forfeits full CUDA graphs, for as long
as both are selected together.

`TRITON_ATTN` declares `AttentionCGSupport.ALWAYS` and is in the candidate list
the selector prints. Backend selection is a plain priority sort
(`vllm/platforms/cuda.py`) and never considers this consequence.

This probe runs the same model twice, once per backend, and reports the
cudagraph mode each one settles on and what a token costs. If the gap is small,
the selector is fine as it is and H2 is closed.

    python ampere/probes/cudagraph_modes.py --model /work/models/Qwen3.5-2B-AWQ-4bit

Run it under sustained generation, not a short burst: on an idle card a burst
measures the clock ramp (see clock_ramp.py) rather than the graph mode.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

PROMPT = (
    "Write a short technical explanation of how a write-ahead log keeps a "
    "database consistent across a crash."
)

RESULT_PREFIX = "AMPERE_PROBE_RESULT "


def child(args: argparse.Namespace) -> int:
    from vllm import LLM, SamplingParams

    kwargs: dict = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_frac,
        enforce_eager=False,
        trust_remote_code=True,
        # Production parity, and not incidental: fp8 KV changes which backends
        # are candidates at all. Omitting it made a first run of this probe
        # select FLASH_ATTN on both arms and report a comparison of nothing.
        kv_cache_dtype=args.kv_cache_dtype,
        # The 2B carries a vision tower it never uses here. Profiling it costs
        # memory that the KV cache wants and adds nothing to a decode result.
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    if args.backend != "auto":
        # AttentionBackendEnum is a plain Enum, so a bare string is accepted by
        # the dataclass and then quietly ignored -- the first run of this probe
        # asked for TRITON_ATTN and measured FLASH_ATTN twice. Pass the member.
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        kwargs["attention_backend"] = AttentionBackendEnum[args.backend]
    if args.spec_tokens > 0:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": args.spec_tokens,
        }

    llm = LLM(**kwargs)

    # Warm up: first-call Triton JIT and the clock ramp both belong outside the
    # measurement, and both are large enough to invert the result if included.
    llm.generate(
        [PROMPT], SamplingParams(temperature=0.0, max_tokens=32), use_tqdm=False
    )

    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        ignore_eos=True,
    )

    latencies = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        out = llm.generate([PROMPT], params, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        produced = len(out[0].outputs[0].token_ids)
        latencies.append(elapsed / max(produced, 1) * 1000.0)

    latencies.sort()
    result = {
        "backend": args.backend,
        "spec_tokens": args.spec_tokens,
        "ms_per_token_best": round(latencies[0], 3),
        "ms_per_token_median": round(latencies[len(latencies) // 2], 3),
        "tok_per_s_median": round(1000.0 / latencies[len(latencies) // 2], 1),
    }

    # Best effort: the authoritative value lives in the engine-core process, so
    # the parent also greps the log. Never let this fail the run.
    try:
        cfg = llm.llm_engine.vllm_config.compilation_config
        result["cudagraph_mode_attr"] = str(cfg.cudagraph_mode)
    except Exception:  # noqa: BLE001
        result["cudagraph_mode_attr"] = None

    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return 0


def run_one(args: argparse.Namespace, backend: str) -> dict:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--child",
        "--model",
        args.model,
        "--backend",
        backend,
        "--spec-tokens",
        str(args.spec_tokens),
        "--max-tokens",
        str(args.max_tokens),
        "--reps",
        str(args.reps),
        "--gpu-frac",
        str(args.gpu_frac),
        "--max-model-len",
        str(args.max_model_len),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
    ]
    print(f"==> {backend}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    blob = proc.stdout + proc.stderr

    record: dict = {"backend": backend}
    for line in blob.splitlines():
        if line.startswith(RESULT_PREFIX):
            record.update(json.loads(line[len(RESULT_PREFIX) :]))

    selected = re.search(
        r"Using (\S+) attention backend out of potential backends: (\[[^\]]*\])", blob
    )
    if selected:
        record["selected"] = selected.group(1)
        record["candidates"] = selected.group(2)

    downgrade = re.search(r"setting cudagraph_mode=(\w+)", blob)
    record["downgraded_to"] = downgrade.group(1) if downgrade else None

    # A probe that measures a backend other than the one it asked for is worse
    # than no probe, because the number looks legitimate. Refuse it.
    if backend != "auto" and record.get("selected") not in (None, backend):
        record["error"] = f"asked for {backend}, engine selected {record['selected']}"
        print(f"    REJECTED: {record['error']}", flush=True)
        return record

    if "ms_per_token_median" not in record:
        record["error"] = f"child exited {proc.returncode}"
        tail = [ln for ln in blob.splitlines() if ln.strip()][-8:]
        print("    FAILED:\n      " + "\n      ".join(tail), flush=True)
    else:
        print(
            f"    {record['tok_per_s_median']} tok/s  "
            f"(downgrade: {record['downgraded_to'] or 'none'})",
            flush=True,
        )
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--backends",
        default="auto,TRITON_ATTN",
        help="comma-separated; 'auto' lets the selector choose, which is the "
        "case being questioned",
    )
    ap.add_argument("--spec-tokens", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument(
        "--kv-cache-dtype",
        default="fp8",
        help="production runs fp8; it changes the candidate backend set, so "
        "'auto' means something different without it",
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--backend", default="auto", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        return child(args)

    records = [run_one(args, b.strip()) for b in args.backends.split(",") if b.strip()]

    print("\n" + "-" * 78)
    print(
        f"{'backend':>14} {'selected':>14} {'downgrade':>12} {'tok/s':>9} {'ms/tok':>9}"
    )
    print("-" * 78)
    for r in records:
        if "error" in r:
            print(f"{r['backend']:>14} {'-':>14} {'-':>12} {'FAILED':>9} {'-':>9}")
            continue
        print(
            f"{r['backend']:>14} {r.get('selected', '?'):>14} "
            f"{str(r.get('downgraded_to') or 'none'):>12} "
            f"{r['tok_per_s_median']:>9} {r['ms_per_token_median']:>9}"
        )

    ok = [r for r in records if "error" not in r]
    if len(ok) >= 2:
        base, best = ok[0], max(ok, key=lambda r: r["tok_per_s_median"])
        if best is not base:
            speedup = best["tok_per_s_median"] / base["tok_per_s_median"]
            print(
                f"\n{best['backend']} is {speedup:.2f}x {base['backend']} at batch 1 "
                f"with {args.spec_tokens} speculative tokens."
            )
        else:
            print(
                f"\nNo backend beat {base['backend']}; the selector's choice stands "
                "and H2 is closed."
            )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(records, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
