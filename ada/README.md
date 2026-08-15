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

## Measured on the card: the decode step, by kernel

Taken on the 4090 itself in the production nightly (`65b7662d3`), batch 1, 128
decode tokens, `kv_cache_dtype=fp8`. This replaces an earlier roofline estimate
that was wrong in a way worth stating plainly: **it omitted the dense MLP**, and
so understated traffic by 40% and overstated the available headroom.

**The card is never idle.** Device-busy time is 5.741 ms/step; the same
generation with the profiler off runs at 5.723–5.751 ms/token. Those agree to
within half a percent, so essentially the whole step is the GPU computing, and
the 6% "idle" the profiler reports is the profiler's own overhead. Full CUDA
graphs are captured (`Capturing CUDA graphs (decode, FULL)`), so Gemma never hits
the PIECEWISE downgrade that costs `ampere/` 1.42x. **There is no launch-overhead
problem here and no graph work worth doing.** Decode is bandwidth-bound, full
stop.

| component | kernel | per step | bytes | achieved |
|---|---|---|---|---|
| **lm_head** (fp16, tied) | cuBLAS `gemvx` | 1.541 ms | 1.476 GB | **958 GB/s** |
| **dense MLP** gate+up (fp16) | cuBLAS `gemvx` | 0.830 ms | 0.714 GB | 860 GB/s |
| **dense MLP** down (fp16) | cuBLAS `gemvx` | 0.445 ms | 0.357 GB | 802 GB/s |
| MoE experts (int4) | `marlin_moe_wna16` | 1.112 ms | 0.803 GB | 722 GB/s |
| attention proj (int4) | `marlin` | 0.865 ms | ~0.624 GB | ~721 GB/s |
| MoE routing/gather | | 0.228 ms | | |
| attention | `kernel_unified_attention` | 0.149 ms | | |
| **total** | | **5.72 ms** | **~3.97 GB** | **694 GB/s** |

**What to measure achieved bandwidth against.** A device-to-device copy on this
pod runs at 919 GB/s, but a copy is read+write and every kernel above is nearly
pure read, so copy understates the ceiling for them — which is why lm_head
appears to beat it. The honest reference is lm_head itself: cuBLAS's fp16 GEMV,
on exactly this access pattern, at **958 GB/s**. Percentages below are against
that.

| kernel | GB/s | % of 958 |
|---|---|---|
| lm_head (fp16) | 958 | **100% — the reference** |
| dense MLP gate+up (fp16) | 860 | 90% |
| dense MLP down (fp16) | 802 | 84% |
| MoE experts (**int4**) | 722 | **75%** |
| attention proj (**int4**) | 721 | **75%** |

The split is by dtype, not by layer. Every fp16 kernel reaches 84–100% of what a
good GEMV does on this card; both int4 Marlin kernels sit at 75%. **lm_head is at
the roofline and no kernel can improve it** — only reading fewer bytes can. Why
Marlin loses the other 25% is *not* measured: dequantisation ALU, scale traffic,
layout shuffles and low occupancy at M=1 are all candidates and picking between
them needs a microbenchmark that sweeps M.

So the real roofline is ~3.97 GB/step → **232 tok/s**, and 175 measured is **75%
of achievable bandwidth**, not the 34% the old estimate implied. The headroom is
1.3x, not 2.5x, and it is not where the estimate said it was.

**16.5% of the step reads no weights.** The five kernels above account for 4.793
ms of the 5.741 ms step. The remaining 0.948 ms is routing (0.228), attention
(0.149), norms, elementwise and memcpy, spread over 674 launches per step. Byte
accounting misses it entirely, and it is the same size as the whole Marlin gap.

**Compute is ~1% utilised.** About 4.7B active parameters per token (4B backbone
plus the 738M vocab projection) is ~9.4 GFLOP, and 9.4 GFLOP in 5.72 ms is 1.6
TFLOPS against roughly 165 TFLOPS of fp16 tensor-core peak. Charging Marlin for
all eight rows of its `m_block_size_8` tile still leaves it under 8%. Two things
follow. Any kernel that spends arithmetic to save bandwidth is free here, so if
Marlin's 25% gap is dequantisation cost it is recoverable. And the classic way to
convert idle FLOPs into tokens is speculative decoding, which this deployment
does not run — see A6.

The attention-projection byte count is derived from config shapes (heterogeneous
head dims, `attention_k_eq_v`, `v_proj` present in only 25 of 30 layers) and is
the one row to re-derive before quoting it.

## Hypotheses

**A1 — The tied fp16 lm_head. CONFIRMED, and it is 27% of the step, not the
majority.** 1.541 ms of 5.72, reading 1.476 GB at 958 GB/s. It is the single
largest kernel and it is already at the roofline, so the only way to make it
cheaper is to read fewer bytes. `final_logit_softcapping: 30.0` already bounds
the logit range, which is favourable.

**A1b — the dense MLP is also fp16, and nobody had noticed. NEW.** Every layer
carries a dense MLP alongside its routed experts (`intermediate_size: 2112`
beside `moe_intermediate_size: 704`, `enable_moe_block: true`), and the
checkpoint leaves all three of its projections unquantized:

```
layers.N.mlp.gate_proj   {'F16': 30}          <- fp16
layers.N.mlp.up_proj     {'F16': 30}          <- fp16
layers.N.mlp.down_proj   {'F16': 30}          <- fp16
layers.N.self_attn.*     {'I64','I32','F16'}  <- packed int4
layers.N.experts.*       {'I64','I32','F16'}  <- packed int4
```

That is **1.07 GB per step**, comparable to lm_head's 1.48, and 22% of decode
time. It is not a deliberate exclusion: `quantization_config.ignore` names only
the vision tower, and `config_groups` targets `Linear` at 4 bits.

Together A1 and A1b mean **2.55 GB of the 3.97 GB read per step — 64% — is fp16
weights that the rest of the checkpoint already demonstrates can be int4.**
Quantizing both would take the step to roughly 2.1 GB, close to a 2x on a
decode that is otherwise at 75% of achievable bandwidth. This is by a wide
margin the largest result on this branch.

Both are checkpoint changes and belong to whoever owns the quantization, not to
this branch. What this branch owes them is the measurement rather than the
arithmetic, and `probes/decode_profile.py` now supplies it.

**A5 — Marlin at M=1 leaves ~25% of bandwidth on the table. Real, and small.**
Both int4 kernels run at ~722 GB/s where cuBLAS's fp16 GEMV manages 958 on the
same card. Lifting Marlin to ~90% of that would save about 0.4 ms of a 5.72 ms
step — **roughly 7%**. That is the honest size of the "write a better batch-1
W4A16 GEMV" idea here, an order of magnitude smaller than A1+A1b.

Before writing anything, sweep M over 1, 2, 4, 8 and 16 on these shapes. If time
is flat across that range the kernel is bandwidth- or latency-bound and the gap
is addressable; if it rises with M the tile is already doing real work and it is
not. That sweep is cheap, needs no patch, and decides whether A5 exists.

Two things that idea should not be sold on, because both were checked and are
false. MoE Marlin is a *grouped* GEMM — `marlin_moe_wna16/marlin_template.h`
selects `expert_id` per block — so `marlin_moe.py` issues two launches per layer
regardless of top-k, about 60 per token rather than the 240 a per-expert reading
suggests; there is no per-expert launch overhead. And "a GEMM-shaped kernel
wastes work at M=1" is true about arithmetic and irrelevant to cost: an 8-row
tile reads the same weight bytes as a 1-row tile, the operation is bandwidth-
bound, and Marlin already narrows to `m_block_size_8` below M=8. The gap is real
but it has to be argued from the 722 GB/s, not from tile shapes.

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

**Measured, and the expectation held — but the prize is far smaller than
"25 of 30 layers" suggests.** The forcing is confirmed on the real deployment:
`Using AttentionBackendEnum.TRITON_ATTN backend` is logged once per attention
group, so both the sliding and the full layers are on Triton. Attention is:

| | attention share of device time | A2 ceiling (25/30 layers) |
|---|---|---|
| decode, batch 1 | 2.6% | 2.2% |
| prefill, 3,958-token prompt | 16.7% | 13.9% |

The 6x difference between the two confirms A2 is a prefill change. But one
3,958-token prefill costs only **12.6 ms of device time**, so 13.9% of it is
about **1.8 ms** — against a production TTFT of 206 ms, that is **under 1%**.
TTFT on this deployment is dominated by scheduling and queueing, not by
attention arithmetic.

So A2 is correct, cheap, and nearly worthless here. It stays on the list because
it is the one change that is genuinely this branch's to make rather than the
quantization owner's, and because the ceiling grows with prompt length — but it
should not be sold as a headline.

**A3 — KV cache headroom is left on the table.** The engine's own startup log
says `--kv-cache-memory=5746727424` (5.35 GiB) would fully use the requested
budget against the 4.55 GiB actually taken: **+18% KV for free** at unchanged
`--gpu-memory-utilization`. Configuration, not a patch; reported here because
KV is the binding constraint on concurrency.

**A4 — Marlin MoE at batch 1. CLOSED, the premise was false.** This said 240
expert GEMMs per token, small enough that launch and tail effects might dominate.
There are about 60. MoE Marlin is a grouped GEMM: `marlin_moe_wna16` reads
`expert_id` per block, so `marlin_moe.py` issues one launch for `w1` and one for
`w2` per layer regardless of top-k. There is no per-expert launch overhead, and
the measured occupancy leaves no room for a launch problem of any kind. What
remains of this hypothesis is A5, which is about bandwidth rather than launches.

**A6 — Nothing is using the compute, and there is no speculative decoding. NEW,
and probably the largest engine-level item here.** Decode runs at roughly 1% of
the card's fp16 tensor-core throughput (~1.6 TFLOPS of ~165), because batch-1
decode reads ~3.97 GB to do ~9.4 GFLOP. The arithmetic units are idle for
essentially the whole step.

Speculative decoding is the standard way to spend idle FLOPs on tokens, and this
deployment does not run it — there is no `--speculative-config` in its args,
while the sibling Qwen deployment runs MTP with three draft tokens. On a model
this bandwidth-bound the drafting cost lands almost entirely in compute nobody is
using, so the acceptance rate is close to a pure win.

The obvious caveats before this becomes a change: Gemma4 ships no MTP head, so it
needs either a draft model (which costs weights, and weights are the scarce
resource here — see A1b) or an EAGLE-style head trained for it; and `ampere/`'s
H2 is a warning that spec decode can silently cost more than it returns, since a
one-layer draft head there removed full CUDA graphs from all 64 target layers.
Gemma currently captures `FULL` graphs, and that must be re-checked, not assumed,
after any spec-decode change.

## Version skew: v0.27.1 is the wrong base for this model

This branch was based on upstream's latest *release*, which is the right default
and happens to be wrong here. Three findings, all from file content and a
measured failure rather than from commit ancestry (the local clone is shallow,
so `merge-base` cannot be trusted and was not used):

**1. The stock `v0.27.1` release image cannot load this checkpoint.** It ships
`transformers 5.15.0`, whose heterogeneity integration raises on per-layer
attributes:

```
AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute and
may vary across layers. Access it via the individual layer configs instead
```

v0.27.1's `model_arch_config_convertor.get_head_size()` does a bare
`getattr(self.hf_text_config, "head_dim", 0)`, and `requirements/common.txt`
pins only `transformers >= 5.5.3`, so the release image installs a transformers
that its own code cannot survive on a heterogeneous model.

**2. Pinning `transformers==5.8.1` fixes it** — the same pin the `pascal` branch
arrived at independently from the same v0.27.1 base. The model then loads (15.55
GiB, `MarlinExperts`) and everything here runs.

**3. Production is running different code in exactly the function A2 patches.**
`Gemma4Config` on v0.27.1 reads two scalars off the text config:

```python
head_dim = getattr(hf_text_config, "head_dim", None)
global_head_dim = getattr(hf_text_config, "global_head_dim", None)
```

while upstream `main` reads them per layer, which is why it is immune to (1):

```python
head_dims = {layer_types[i]: arch_config[i].head_size ...}
```

The deployment logs `heterogeneous head dimensions {'sliding_attention': 256,
'full_attention': 512}` — a dict, matching `main`'s message and not v0.27.1's
`(head_dim=%d, global_head_dim=%d)`. So production runs the per-layer version.

**Consequence.** Measurements here are made on a transformers pin production
does not use, against a `Gemma4Config` production does not run. The A2 result
should still transfer, because the forcing *behaviour* is identical in both
versions — both end at `attention_config.backend = TRITON_ATTN` — but the patch
itself must be written against whatever base the deployment actually ships, and
that is not this one. Before A2 becomes a deployment change, rebase `ada/` onto
the nightly the Gemma pod runs (`65b7662d3`) or onto `main`.

`ampere/` is unaffected: Qwen3.5 does not have heterogeneous head dimensions, and
that deployment genuinely runs `v0.27.1`.

## Running the probes

Unlike `ampere/`, this needs no model download: the checkpoint already lives on
a cluster PVC and the overlay mounts it read-only. That claim is ReadWriteOnce,
so the serving deployment has to be scaled to zero for the volume as well as for
the GPU. See `k8s/README.md`.
