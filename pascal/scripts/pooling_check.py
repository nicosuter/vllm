"""Embedding and reranking checks on Pascal.

Generative models can be judged by reading their output; embeddings and scores
cannot. A vector of finite floats proves nothing — a broken kernel produces
those happily. So both checks here are behavioural, on inputs where the right
answer is not in doubt.

    python pascal/scripts/pooling_check.py --model /work/models/Qwen3-VL-Embedding-2B --mode embed
    python pascal/scripts/pooling_check.py --model /work/models/Qwen3-VL-Reranker-2B  --mode rerank-logits
"""

from __future__ import annotations

import argparse
import math
import sys

EMBED_TEXTS = [
    "a photograph of a cat sitting in the snow",
    "a picture of a kitten outdoors in winter",
    "quarterly earnings exceeded analyst expectations",
]

# A cross-encoder is tested by discrimination, not by a fixed ordering of one
# query's documents.
#
# An earlier fixture used one query and asserted that two "relevant" documents
# both outrank an unrelated one. The model scored the sourdough method 1.0 and
# everything else 0.0 — including "Bread making relies on yeast fermentation",
# which states a fact but does not tell you how to bake anything. That is
# defensible behaviour from a reranker; the assertion was wrong, not the model.
#
# Two queries from unrelated domains, each with exactly one document that
# answers it, avoids the judgement call: a working reranker must prefer its own
# document for each query, which no constant scoring function can satisfy.
RERANK_CASES = [
    (
        "How do I bake sourdough bread at home?",
        "Mix flour and water, let the starter ferment, then shape the loaf and "
        "bake it in a hot Dutch oven until the crust is dark.",
    ),
    (
        "What causes the aurora borealis?",
        "Charged particles from the solar wind excite oxygen and nitrogen in the "
        "upper atmosphere, which emit light as they return to their ground state.",
    ),
]

# From the model's own 1_LogitScore/config.json.
YES_TOKEN_ID = 9693
NO_TOKEN_ID = 2152

RERANK_INSTRUCTION = (
    "Given a search query, retrieve relevant candidates that answer the query."
)


def cosine(a, b):
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

    # Requiring a margin rather than a bare inequality keeps this from passing
    # on noise.
    if near <= max(far_a, far_b) + 0.05:
        print("\nFAIL: related pair not separated from unrelated pairs")
        return 1
    print(f"\nPASS: related pair leads by {near - max(far_a, far_b):.4f}")
    return 0


def run_rerank_logits(model: str, gpu_frac: float, max_len: int, max_batched: int) -> int:
    """Score with the reranker the way the model itself does: yes/no logits.

    vLLM cannot use this model's scoring head. The head is not a linear layer —
    1_LogitScore/ holds only {"true_token_id": 9693, "false_token_id": 2152},
    so the relevance score is the LM logit of "yes" against "no", and the
    checkpoint contains no classifier tensors at all. Driving that through
    vLLM's score() API needs a *ForSequenceClassification architecture with
    classifier_from_token, and vLLM implements those only for Bert, GPT2,
    Llama, Jamba, ModernBert and Roberta — there is no Qwen3-VL variant.

    That is a vLLM gap, not a Pascal one: it would fail identically on an H100.
    Running the model generatively and reading the two logits exercises exactly
    the same kernels and yields the model's real score.

    Loading it as a pooling model instead is worse than useless here: vLLM
    resolves `--convert auto` to "embed", and score() then returns the cosine
    between query and document embeddings. That still ranks roughly correctly,
    which is what makes it a trap — the numbers bunch into a narrow band and
    nothing announces that the ranking head was never involved.
    """
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(
        model=model,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=gpu_frac,
        max_model_len=max_len,
        max_num_batched_tokens=max_batched,
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=True,
    )

    def build_prompt(query: str, doc: str) -> str:
        messages = [
            {"role": "system", "content": RERANK_INSTRUCTION},
            {"role": "user", "content": f"Query: {query}"},
            {"role": "user", "content": f"Candidate: {doc}"},
        ]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    queries = [q for q, _ in RERANK_CASES]
    docs = [d for _, d in RERANK_CASES]
    prompts = [build_prompt(q, d) for q in queries for d in docs]

    # Restricting to the two tokens makes the logprobs deterministic to read;
    # otherwise "yes"/"no" can fall outside the returned top-k.
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=2,
        allowed_token_ids=[YES_TOKEN_ID, NO_TOKEN_ID],
    )
    outs = llm.generate(prompts, params)

    scores = []
    for out in outs:
        lp = out.outputs[0].logprobs[0]
        lp_yes = lp[YES_TOKEN_ID].logprob if YES_TOKEN_ID in lp else -1e9
        lp_no = lp[NO_TOKEN_ID].logprob if NO_TOKEN_ID in lp else -1e9
        scores.append(math.exp(lp_yes) / (math.exp(lp_yes) + math.exp(lp_no)))

    n = len(RERANK_CASES)
    grid = [scores[i * n : (i + 1) * n] for i in range(n)]

    print("\nP(yes) for each query x document pair:")
    header = "".join(f"   doc{j} " for j in range(n))
    print(f"  {'query':<32}{header}")
    for i, row in enumerate(grid):
        print(f"  {queries[i][:32]:<32}" + "".join(f"  {v:.3f}" for v in row))

    if any(v != v for row in grid for v in row):
        print("\nFAIL: scores contain NaN")
        return 1

    for i, row in enumerate(grid):
        best = max(range(n), key=lambda j: row[j])
        if best != i:
            print(f"\nFAIL: query {i} prefers doc{best}, expected doc{i}")
            return 1

    margin = min(
        grid[i][i] - max(v for j, v in enumerate(grid[i]) if j != i) for i in range(n)
    )
    print(f"\nPASS: every query prefers its own document, worst margin {margin:.5f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["embed", "rerank-logits"], required=True)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument(
        "--max-batched-tokens",
        type=int,
        default=2048,
        help="vLLM profiles a forward pass at this width before serving. The "
        "default 8192 is sized for tensor-core hardware; on a 1070 Ti it makes "
        "startup dominate the run without changing what is being tested.",
    )
    args = ap.parse_args()

    if args.mode == "embed":
        return run_embed(
            args.model, args.gpu_frac, args.max_len, args.max_batched_tokens
        )
    return run_rerank_logits(
        args.model, args.gpu_frac, args.max_len, args.max_batched_tokens
    )


if __name__ == "__main__":
    sys.exit(main())
