"""Measure what the hybrid layout's forced block size costs prefix reuse.

Qwen3.5 interleaves GDN linear attention with full attention, and the allocator
hands both a single uniform page. One attention block therefore has to hold at
least one mamba state, which `Platform.check_and_update_config` enforces:

    attn_block_size = kernel_block_alignment * cdiv(
        mamba_page_size, kernel_block_alignment * attn_page_size_1_token)

On the 27B that lands at 1600 tokens, and vLLM says so at startup:

    Setting attention block size to 1600 tokens to ensure that attention page
    size is >= mamba page size.
    Padding mamba page size by 0.25% ...
    Add 3 padding layers, may waste at most 6.25% KV cache memory

Prefix caching is block-granular, so a 1600-token block means a shared prefix is
only reused down to the last whole 1600 tokens, and any sequence occupies a
whole multiple of 1600 no matter how short it is. Both effects are bounded
rather than proportional, which is exactly why they deserve measuring instead of
asserting: "prefix caching is broken" and "prefix caching wastes up to 1599
tokens per request" call for very different amounts of work.

    python ampere/probes/prefix_granularity.py --model /work/models/Qwen3.5-2B-AWQ-4bit

Reports, per prefix length, how many tokens the engine actually reports as
cached on a repeat request. A staircase means block quantization; a diagonal
means it is finer than the block and H4 is closed.
"""

from __future__ import annotations

import argparse
import contextlib
import sys

FILLER = (
    "The scheduler assigns each runnable thread a virtual deadline and picks "
    "the smallest. Preemption happens when a newly woken thread has an earlier "
    "deadline than the running one. "
)


def build_prompt(tokenizer, target_tokens: int, salt: str = "") -> str:
    """A prompt whose token count is as close to target as repetition allows."""
    text = salt + FILLER
    while len(tokenizer(text).input_ids) < target_tokens:
        text += FILLER
    ids = tokenizer(text).input_ids[:target_tokens]
    return tokenizer.decode(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--lengths",
        default="400,800,1600,2400,3200,4800,6400",
        help="prefix lengths in tokens; span the suspected block size so the "
        "staircase, if there is one, has visible steps",
    )
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_frac,
        enable_prefix_caching=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )

    block_size = None
    with contextlib.suppress(Exception):
        block_size = llm.llm_engine.vllm_config.cache_config.block_size
    print(f"\nresolved block_size: {block_size}\n")

    tokenizer = llm.get_tokenizer()
    one_token = SamplingParams(temperature=0.0, max_tokens=1)

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    header = f"{'prefix':>8} {'cached':>8} {'cached %':>9}"
    print(f"{header} {'blocks held':>12} {'waste':>8}")
    print("-" * 52)

    for length in lengths:
        # A distinct salt per length keeps each case from being served out of
        # the previous case's blocks, which would report reuse that a real
        # first-time prefix never gets.
        prompt = build_prompt(tokenizer, length, salt=f"[case {length}] ")

        # First pass populates the cache; its own hit count is meaningless.
        llm.generate([prompt], one_token, use_tqdm=False)
        # Second pass is the measurement: same prefix, so everything the engine
        # is capable of reusing, it reuses.
        out = llm.generate([prompt], one_token, use_tqdm=False)

        cached = out[0].num_cached_tokens
        actual = len(out[0].prompt_token_ids)
        if cached is None:
            print(f"{actual:>8} {'n/a':>8} {'-':>9} {'-':>12} {'-':>8}")
            continue

        pct = 100.0 * cached / max(actual, 1)
        if block_size:
            held = -(-actual // block_size) * block_size
            waste = held - actual
            print(f"{actual:>8} {cached:>8} {pct:>8.1f}% {held:>12} {waste:>8}")
        else:
            print(f"{actual:>8} {cached:>8} {pct:>8.1f}% {'?':>12} {'?':>8}")

    print(
        "\n'cached' is what the engine reports reusing on an identical repeat "
        "prompt.\n'blocks held' is what the allocator reserves for it; the gap "
        "is KV paid for and\nnot used, on every sequence regardless of length."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
