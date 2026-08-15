# Model coverage

Results for individual models on the GTX 1070 Ti. Split out of the main
README, which is about the fork itself rather than about which checkpoints
were tried on it.

Quantized variants are substituted wherever the headline model ships
unquantized, because 8 GB does not hold fp16 weights plus a KV cache.

| Requested | Tested as | Size | Status |
|---|---|---|---|
| `cyankiwi/Qwen3.5-2B-AWQ-4bit` | as requested | 2.4 GB | **Green gate.** 80.5 tok/s batch 1, 600 tok/s batch 32 |
| `ibm-granite/granite-4.1-3b` | `cyankiwi/granite-4.1-3b-AWQ-INT4` | 2.3 GB | **Works.** 16.8 tok/s, measured before the kernel work |
| `Qwen/Qwen3-VL-Embedding-2B` | as requested, fp16 | 4.3 GB | **Works.** dim 2048, related pair leads by 0.52 cosine |
| `Qwen/Qwen3-VL-Reranker-2B` | as requested, fp16 | 4.3 GB | **Works** via yes/no logits; vLLM's score() path cannot load it |
| `google/gemma-4-E2B-it-qat-q4_0-unquantized` | `google/gemma-4-E2B-it-qat-w4a16-ct` | 8.3 GB | **Does not fit.** Three blockers cleared, OOM remains; see below |
| `RedHatAI/Qwen2.5-1.5B-Instruct-FP8-dynamic` | as requested | 2.2 GB | **Works.** 57.5 tok/s batch 1; first FP8 checkpoint on this card. 2.98 GiB resident — see the FP8 section in the main README |
| `Qwen/Qwen3-0.6B-FP8` | as requested | 0.7 GB | **Does not load.** Block-wise `[128,128]` fp8; no block-scaled kernel exists below capability 8.9. Declines cleanly. Use a per-channel `-FP8-dynamic` checkpoint instead |

### Gemma 4 E2B does not fit this card, and the reason is structural

Three separate blockers were found and cleared, and the fourth is the one that
stops it. Recorded in order, because each had to be removed to see the next.

**1. It does not fit in any quantization.** `embed_tokens_per_layer` is 4.375
GiB and stays unquantized in every variant, so Google's W4A16 QAT release is
still 7.745 GiB:

```
  4.375 GiB  model.language_model.embed_tokens_per_layer   <- unquantized
  0.977 GiB  model.language_model.layers                   <- INT4
  0.750 GiB  lm_head.weight
  0.750 GiB  model.language_model.embed_tokens
  0.568 GiB  model.audio_tower
  0.312 GiB  model.vision_tower
```

**2. transformers incompatibility.** Cleared by pinning 5.8.1 plus two code
fixes — see `get_maybe_per_layer_attr`. Not a Pascal problem.

**3. A bf16 activation meeting an fp16 weight.** Gemma 4 is any-to-any, so
declining only image and video still builds and profiles the **audio** tower,
whose weights are in the quantization ignore list and therefore keep the
checkpoint's bfloat16. They then meet fp16 weights:
`expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != c10::Half`.
Declining audio as well fixes it and drops the load from 6.96 to 6.39 GiB.

This one *is* Pascal-shaped: only Pascal is forced to convert the model to fp16,
so only Pascal exercises the path where a bf16 island survives.

**4. Out of memory, and `cpu_offload_gb` does not help.** 6.39 GiB of weights
plus KV cache and activations exceeds 7.92 GiB. The load reports **6.39 GiB with
and without** `--cpu-offload-gb 3.0`, so the offload is not reducing this
model's resident footprint — its weights evidently do not go through the path
that offload wraps.

Making Gemma 4 E2B work here means offloading the per-layer embeddings
specifically, which is what Gemma-3n's design intends: they are a per-token
gather, cheap to keep in host RAM and cheap to transfer. That is real
engineering, not a flag, and it is the honest next step rather than something
this fork currently does.

### The reranker works, but not through vLLM's scoring API

`Qwen3-VL-Reranker-2B` has no scoring head to load. `1_LogitScore/` contains
only `{"true_token_id": 9693, "false_token_id": 2152}` — tokens that decode to
`"yes"` and `"no"` — and the checkpoint holds no classifier tensors at all. The
relevance score *is* the LM logit of yes against no.

vLLM cannot drive that through `score()`. Doing so needs a
`*ForSequenceClassification` architecture with `classifier_from_token`, and vLLM
implements those for Bert, GPT2, Llama, Jamba, ModernBert and Roberta only —
there is no Qwen3-VL variant. `--convert classify` gets as far as building a head
and then fails with `Scoring API is only enabled for num_labels == 1`. **This is
a vLLM gap, not a Pascal one; it would fail the same way on an H100.**

Running the model generatively and reading the two logits exercises the same
kernels and gives the real score:

```
query                              doc0    doc1
How do I bake sourdough bread at    1.000   0.000
What causes the aurora borealis?    0.000   1.000
```

Worth recording the trap, because it nearly passed silently: with
`--convert auto`, vLLM resolves the reranker to **embed** and `score()` returns
the cosine between query and document *embeddings*. That still ranks roughly
correctly — 0.894 / 0.883 / 0.868 in the first attempt here — so it looks like a
pass. The giveaway is the compression: those are cosines of related English
text, and the ranking head was never involved.

## A note on the throughput figures

Only the gate model has been re-measured since the fp32 kernel work (see the
main README). The granite figure predates it and is left as recorded rather than
extrapolated -- the speedup is a property of the exllama W4A16 path, which
granite also uses, but claiming a number that was never measured would be worse
than an out-of-date one.

MTP is no longer worth enabling. It was 1.83x when a decode step cost 55 ms;
after the step fell to ~13 ms there is far less per-step cost to amortise while
the drafter's own overhead is unchanged, and it now measures 68.4 tok/s against
72.8 without.
