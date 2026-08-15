# vLLM for consumer Ada (sm_89)

This directory targets one deployment: **Gemma4-26B-A4B AWQ-INT4, single card,
on an RTX 4090**. Like `ampere/`, it is a tuning effort and not a portability
one — sm_89 is well supported and the model runs today.

The sibling directories are different problems on the same branch: `ampere/` is
Qwen3.5-27B on a 3090 Ti pair (TP=2, hybrid GDN, spec decode), and the `pascal`
branch is sm_61 portability. They share no code.

## The card and the model

| | |
|---|---|
| GPU | 1× RTX 4090, sm_89, 24 GiB, ~1008 GB/s, power-capped to 350 W of 450 W |
| Host | 28 vCPU but only **27.6 GiB RAM**, 87% of it already requested |
| Model | Gemma4-26B-A4B, MoE, 128 experts, top-8, ~4B active of 26B |
| Layers | 30: **25 sliding_attention** (head_dim 256, window 1024) + **5 full_attention** (head_dim 512) |
| Quant | compressed-tensors W4A16, group 32, symmetric → Marlin |
| Not quantized | `embed_tokens` `[262144, 2816]` **fp16**, and `tie_word_embeddings: true` |
| KV | fp8, 4.55 GiB, 374,591 tokens, **1.43x** concurrency at 262k context |

The two structural facts that drive everything below: the output projection is
an unquantized 1.48 GB tensor read on every decode step, and 5 layers with an
unusual head dimension dictate the attention backend for the other 25.

## Measured baseline

From the production engine's own metrics (nightly `0.26.1rc1.dev602`, which is
ahead of this branch's `v0.27.1` base — see "Version skew"):

| Metric | Value |
|---|---|
| Inter-token latency | mean 7.9 ms → **~127 tok/s single stream** |
| TTFT | mean 206 ms, p99 5 s |
| Prompt length | mean 3,926 tokens |
| Iteration size | **p50 = 1 token, mean 2.23** — 75% of forwards carry ≤1 token |
| Prefix cache | 51% hit rate over 223,775 queries |

That iteration-size distribution is the single most important number here. This
is not a throughput workload; it is almost pure single-stream decode. Batching
work is close to worthless, and anything that reduces bytes-read-per-step or
per-step overhead is worth its weight.

**Decode roofline.** Per step at batch 1, approximately:

| component | bytes |
|---|---|
| lm_head (tied `embed_tokens`, fp16, unquantized) | **1.48 GB** |
| MoE experts (8 of 128, 30 layers, ~4.5 bits effective) | ~0.80 GB |
| attention projections | ~0.4 GB |
| **total** | **~2.7 GB** |

At 1008 GB/s that is a 373 tok/s ceiling against 127 measured — **34% of
memory bandwidth**, so there is roughly 2.5x of headroom before the card is the
limit. And the output projection alone is over half the traffic.

## Hypotheses

**A1 — The tied fp16 lm_head is the majority of every decode step.** 1.48 GB of
~2.7 GB. Quantizing it to int8 would cut the step to ~1.95 GB (a **+38%**
roofline) and free ~740 MiB, which is another **+16%** of KV cache on a model
whose concurrency is 1.43x. `final_logit_softcapping: 30.0` already bounds the
logit range, which is favourable.

This is a checkpoint change and belongs to whoever owns the quantization, not to
this branch. What this branch owes them is the measurement rather than the
arithmetic: `probes/decode_profile.py` sizes the vocab projection against
everything else, so the decision is made on a number.

**A2 — 25 of 30 layers are on Triton for their neighbours' sake.**
`Gemma4Config.verify_and_update_config` forces `TRITON_ATTN` model-wide when
`head_dim != global_head_dim` and FA4 is unavailable. FA4 is Hopper-only, so on
a 4090 all 30 layers go to Triton because 5 of them have head_dim 512.
FlashAttention accepts head_size ≤ 256 without FA4, so the 25 sliding layers are
eligible and excluded only by the blanket policy — whose stated rationale
("avoid the mixed FA3/FA4 penalty") is about Hopper.

*Expect this to be a prefill win, not a decode win.* The sliding layers are
capped at a 1024-token window, so their decode attention is cheap whichever
kernel runs it; prefill is where attention scales with the 3,926-token mean
prompt. `probes/attn_backend_split.py` measures both separately, and needs no
patch to do it: `Gemma4Config` only sets `attention_config.backend`, while
`selector.py` resolves `backend_per_kind` ahead of it.

**A3 — KV cache headroom is left on the table.** The engine's own startup log
says `--kv-cache-memory=5746727424` (5.35 GiB) would fully use the requested
budget against the 4.55 GiB actually taken: **+18% KV for free** at unchanged
`--gpu-memory-utilization`. Configuration, not a patch; reported here because
KV is the binding constraint on concurrency.

**A4 — Marlin MoE at batch 1.** 128 experts, top-8, 30 layers means 240 expert
GEMMs per token, each 2816×704 — small enough that launch and tail effects may
dominate. Only worth opening if `decode_profile.py` shows the Marlin bucket
large relative to its bandwidth share. Deep work; last in line.

## Version skew

Production runs a nightly (`0.26.1rc1.dev602+g65b7662d3`); this branch is based
on `v0.27.1`. `Gemma4Config` carries the same forcing logic in both, differing
only in how it reads the head dimensions, so A2 transfers. Anything else found
here should be re-checked against the nightly before it reaches the deployment.

## Running the probes

Unlike `ampere/`, this needs no model download: the checkpoint already lives on
a cluster PVC and the overlay mounts it read-only. That claim is ReadWriteOnce,
so the serving deployment has to be scaled to zero for the volume as well as for
the GPU. See `k8s/README.md`.
