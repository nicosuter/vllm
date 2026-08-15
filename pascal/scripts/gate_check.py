"""Green-gate check: Qwen3.5-2B-AWQ-4bit on Pascal.

v1 is a correctness gate, so "it printed something" is not the bar. Two things
have to hold:

  1. Greedy decoding produces coherent, on-topic continuations.
  2. The logits agree with a CPU fp32 reference from HF transformers on the same
     prompts. This is the check that actually catches a miscompiled kernel:
     wrong attention or a bad dequant usually still yields fluent-looking text,
     just text that has quietly drifted from what the weights say.

Run inside the build pod:

    python pascal/scripts/gate_check.py --model /work/models/Qwen3.5-2B-AWQ-4bit
"""

from __future__ import annotations

import argparse
import json
import sys

PROMPTS = [
    "The capital of Switzerland is",
    "def fibonacci(n):\n    ",
    "In one sentence, explain why the sky is blue:",
    "List three prime numbers greater than 100:",
]


def run_vllm(
    model: str,
    max_tokens: int,
    enforce_eager: bool,
    gpu_frac: float,
    mtp_tokens: int = 0,
    text_only: bool = True,
):
    from vllm import LLM, SamplingParams

    kwargs = {}
    if text_only:
        # Qwen3.5-2B is multimodal, so vLLM profiles the vision tower with a
        # max-size image even when no image is ever sent. On Pascal that
        # encoder runs its rotary embedding unfused (flash-attention's fused
        # apply_rotary_emb needs the FA extension), which makes startup
        # profiling far more expensive than the text path it is sizing.
        # v1 is text-only, so decline the modalities outright.
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

    if mtp_tokens:
        # Qwen3.5 carries a real MTP head (mtp_num_hidden_layers=1), and this
        # checkpoint keeps it quantized rather than stripping it. vLLM builds the
        # draft model from the same weights, so no second checkpoint is needed.
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": mtp_tokens,
        }

    llm = LLM(
        model=model,
        dtype="float16",  # Pascal has no usable bf16
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_frac,
        max_model_len=2048,
        trust_remote_code=True,
        **kwargs,
    )
    # Greedy, so the comparison against the reference is deterministic.
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=5)
    outs = llm.generate(PROMPTS, params)
    results = []
    for prompt, out in zip(PROMPTS, outs):
        completion = out.outputs[0]
        results.append(
            {
                "prompt": prompt,
                "text": completion.text,
                "token_ids": list(completion.token_ids),
            }
        )
    return results


def run_reference(model: str, results: list[dict], max_tokens: int):
    """Greedy-decode the same prompts on CPU in fp32 via HF transformers.

    Slow but decisive: any disagreement in the token sequence points at a kernel
    rather than at sampling, because both sides are greedy.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    ref_model.eval()

    for entry in results:
        ids = tok(entry["prompt"], return_tensors="pt").input_ids
        with torch.no_grad():
            gen = ref_model.generate(
                ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        ref_ids = gen[0, ids.shape[1]:].tolist()
        entry["ref_token_ids"] = ref_ids
        entry["ref_text"] = tok.decode(ref_ids, skip_special_tokens=True)

        # Report the length of the matching prefix rather than a bare
        # equal/not-equal: greedy decoding can legitimately diverge late on a
        # near-tie, but an early split means a broken kernel.
        match = 0
        for a, b in zip(entry["token_ids"], ref_ids):
            if a != b:
                break
            match += 1
        entry["prefix_match"] = match
        entry["prefix_match_frac"] = match / max(1, min(len(entry["token_ids"]), len(ref_ids)))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--no-eager", action="store_true", help="allow CUDA graphs / torch.compile")
    ap.add_argument("--skip-reference", action="store_true", help="generation only, no CPU reference")
    ap.add_argument(
        "--mtp",
        type=int,
        default=0,
        metavar="N",
        help="enable MTP speculative decoding with N speculative tokens",
    )
    ap.add_argument(
        "--multimodal",
        action="store_true",
        help="allow image/video inputs (v2); off by default so startup does not "
        "profile the vision tower",
    )
    ap.add_argument("--out", default="/work/gate_result.json")
    args = ap.parse_args()

    results = run_vllm(
        args.model,
        args.max_tokens,
        not args.no_eager,
        args.gpu_frac,
        args.mtp,
        not args.multimodal,
    )

    print("\n" + "=" * 72)
    print("GENERATION")
    print("=" * 72)
    for r in results:
        print(f"\n>>> {r['prompt']!r}\n    {r['text']!r}")

    if not args.skip_reference:
        print("\nrunning CPU fp32 reference (slow)...", flush=True)
        results = run_reference(args.model, results, args.max_tokens)
        print("\n" + "=" * 72)
        print("AGREEMENT WITH CPU fp32 REFERENCE")
        print("=" * 72)
        for r in results:
            print(
                f"  prefix_match={r['prefix_match']:>3}/{args.max_tokens} "
                f"({r['prefix_match_frac']:.0%})  {r['prompt'][:40]!r}"
            )
            if r["prefix_match_frac"] < 1.0:
                print(f"      pascal: {r['text']!r}")
                print(f"      ref   : {r['ref_text']!r}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {args.out}")

    if not args.skip_reference:
        worst = min(r["prefix_match_frac"] for r in results)
        # A late divergence on one prompt is tolerable; an early one is not.
        if worst < 0.5:
            print(f"\nFAIL: worst prefix agreement {worst:.0%} — suspect a kernel, not sampling")
            return 1
        print(f"\nPASS: worst prefix agreement {worst:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
