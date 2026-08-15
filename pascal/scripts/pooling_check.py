"""Embedding and reranking checks on Pascal.

Generative models can be judged by reading their output; embeddings cannot. A
vector of finite floats proves nothing — a broken kernel produces those happily.
So the check is behavioural: semantically related pairs must score above
unrelated ones, by a margin, on inputs where the ordering is not in doubt.

    python pascal/scripts/pooling_check.py --model /work/models/Qwen3-VL-Embedding-2B --mode embed
    python pascal/scripts/pooling_check.py --model /work/models/Qwen3-VL-Reranker-2B  --mode score
"""

from __future__ import annotations

import argparse
import sys

QUERY = "How do I bake sourdough bread at home?"

# Ordered by expected relevance. The check is that the model reproduces this
# ordering, not that it hits any particular score.
DOCS = [
    "Mix flour and water, let the starter ferment, then shape the loaf and bake "
    "it in a hot Dutch oven until the crust is dark.",
    "Bread making relies on yeast fermentation to leaven dough before baking.",
    "The 1976 Montreal Olympics were the first to be held in Canada.",
]

EMBED_TEXTS = [
    "a photograph of a cat sitting in the snow",
    "a picture of a kitten outdoors in winter",
    "quarterly earnings exceeded analyst expectations",
]


def cosine(a, b):
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / max(na * nb, 1e-9)


def run_embed(model: str, gpu_frac: float, max_len: int, max_batched: int) -> int:
    from vllm import LLM

    llm = LLM(
        model=model,
        runner="pooling",
        dtype="float16",  # Pascal has no usable bf16
        enforce_eager=True,
        gpu_memory_utilization=gpu_frac,
        max_model_len=max_len,
        max_num_batched_tokens=max_batched,
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=True,
    )
    outs = llm.embed(EMBED_TEXTS, use_tqdm=False)
    vecs = [list(o.outputs.embedding) for o in outs]

    print(f"\nembedding dim: {len(vecs[0])}")
    for t, v in zip(EMBED_TEXTS, vecs):
        finite = all(x == x and abs(x) != float("inf") for x in v)
        print(f"  finite={finite}  |v|={sum(x * x for x in v) ** 0.5:.4f}  {t[:52]!r}")

    if not all(all(x == x for x in v) for v in vecs):
        print("\nFAIL: embeddings contain NaN")
        return 1

    near = cosine(vecs[0], vecs[1])
    far_a = cosine(vecs[0], vecs[2])
    far_b = cosine(vecs[1], vecs[2])
    print(f"\n  cos(cat-snow, kitten-winter) = {near:.4f}   <- related")
    print(f"  cos(cat-snow, earnings)      = {far_a:.4f}")
    print(f"  cos(kitten-winter, earnings) = {far_b:.4f}")

    # A working encoder puts the paraphrase pair clearly above the unrelated
    # pairs. Requiring a margin rather than a bare inequality keeps this from
    # passing on noise.
    if near <= max(far_a, far_b) + 0.05:
        print("\nFAIL: related pair not separated from unrelated pairs")
        return 1
    print(f"\nPASS: related pair leads by {near - max(far_a, far_b):.4f}")
    return 0


def run_score(model: str, gpu_frac: float, max_len: int, max_batched: int) -> int:
    from vllm import LLM

    llm = LLM(
        model=model,
        runner="pooling",
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=gpu_frac,
        max_model_len=max_len,
        max_num_batched_tokens=max_batched,
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=True,
    )
    outs = llm.score(QUERY, DOCS, use_tqdm=False)
    scores = [float(o.outputs.score) for o in outs]

    print(f"\nquery: {QUERY!r}")
    for s, d in zip(scores, DOCS):
        print(f"  {s: .5f}  {d[:64]!r}")

    if any(s != s for s in scores):
        print("\nFAIL: scores contain NaN")
        return 1
    # The third document is about the Olympics. If it outranks either bread
    # document the ranking head is not working.
    if scores[2] >= min(scores[0], scores[1]):
        print("\nFAIL: unrelated document outranks a relevant one")
        return 1
    print(f"\nPASS: relevant docs lead unrelated by {min(scores[0], scores[1]) - scores[2]:.5f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["embed", "score"], required=True)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument(
        "--max-batched-tokens",
        type=int,
        default=2048,
        help="vLLM profiles a forward pass at this width before serving. The "
        "default 8192 is sized for tensor-core hardware; on a 1070 Ti it makes "
        "startup dominate the run without changing what is being tested.",
    )
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    if args.mode == "embed":
        return run_embed(args.model, args.gpu_frac, args.max_len, args.max_batched_tokens)
    return run_score(args.model, args.gpu_frac, args.max_len, args.max_batched_tokens)


if __name__ == "__main__":
    sys.exit(main())
