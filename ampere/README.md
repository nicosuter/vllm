# vLLM for consumer Ampere (sm_86)

This branch tunes vLLM for one deployment: **Qwen3.5-27B AWQ-INT4, TP=2, on a
pair of RTX 3090 Ti**. It is not a portability fork — sm_86 has been supported
upstream for years and everything here already runs. It exists because the
defaults upstream picks for this hardware are chosen for datacenter parts, and
several of them degrade silently rather than failing.

The sibling `pascal` branch is a different problem (making sm_61 work at all)
and shares no code with this one.

## The card and the box

| | |
|---|---|
| GPU | 2× RTX 3090 Ti, sm_86, 24 GiB, ~1008 GB/s each |
| Interconnect | PCIe, both cards on the same host bridge (`PHB`) |
| P2P | **Not supported** (`nvidia-smi topo -p2p r` → `NS` both directions) |
| NVLink | Links present, **no bridge fitted** (`nvidia-smi nvlink -s` → "all links are inActive") |
| FP8 | No tensor-core FP8 on sm_86 — fp8 weights are a bandwidth win only, never a math win |
| Idle state | `nvidia-smi` reports pstate P8 / 210 MHz between requests. This is a sampling artifact, not a tax — see H1. |

The two facts that shape everything below: there is no fast path between the two
cards, and sm_86 cannot do FP8 math.

## Measured baseline

From the production engine's own Prometheus metrics, vLLM v0.27.1, TP=2, MTP
spec decode with `num_speculative_tokens=3`, `kv_cache_dtype=fp8`:

| Metric | Value |
|---|---|
| Inter-token latency | mean 62.5 ms, p50 bucket 75 ms → **~16 tok/s single stream** |
| TTFT | mean 0.90 s at 1,173-token mean prompt |
| MTP acceptance | 478 / 690 draft tokens (69%), 2.08 accepted per draft step |
| Weights | 12.19 GiB per GPU |
| KV cache | fp8, 7.2 GiB, 405,664 tokens, 1.55x concurrency at 262k context |
| External prefix cache | 36,364 queries, **0 hits** |

**Treat the 62.5 ms as unproven.** It comes from 230 samples, all of them
four-token health-check generations against an otherwise idle engine. A card in
P8 runs at 210 MHz against a 2100 MHz ceiling; a short request that never
triggers a clock ramp would produce roughly this number all on its own. Ruling
that out is the first job of the harness here, because a 10x clock deficit and
a 10x kernel problem look identical from the outside, and only one of them is
worth writing code for.

For reference, the bandwidth roofline for a batch-1 decode step: ~8.8 GB read
per target forward per GPU (backbone at 4 bits plus a TP-sharded 248320×5120
bf16 lm_head), plus three cheap MTP draft forwards, ≈ 12.6 ms per step over
3.08 tokens ≈ **4 ms/token**. Measured is 15x that. Somewhere between "the card
was asleep" and "a kernel is wrong" lies the answer, and this directory exists
to find out which.

## Hypotheses, ranked

Each is falsifiable and none has been tested yet.

**H1 — Idle clocks. CLOSED, false.** Measured with `probes/clock_ramp.py` on an
idle RTX 4090 (see "Where these were measured"), 8192³ bf16 matmul after 25 s of
idle:

```
idle state: P8, 210 MHz, 26 W
first iteration cold :    14.88 ms
steady state (hot)   :     7.46 ms
first-iteration tax  :     1.99x
ramp to within 10%   :       22 ms of continuous work
```

The card is at full clocks 22 ms into a request, and the whole ramp costs about
7 ms once. Against a production TTFT near 0.9 s and hundreds of decode tokens
that is a rounding error, not a 15x. It cannot explain the baseline.

Two things worth keeping. `nvidia-smi` reported `P8, 210 MHz` *while the burst
was running* — the pstate you read between synchronisation points is the idle
one, so "the GPUs sit at P8" is an artifact of when the sample lands and not
evidence about anything. And the sm_89 card is a stand-in; if the 3090 Ti pair
ever shows a materially different ramp, this reopens.

**H2 — Full CUDA graphs are silently disabled. REPRODUCED, and worse than
stated: it cannot be worked around from the command line.**

The downgrade reproduces on the proxy model as soon as `kv_cache_dtype=fp8` is
set, which is what makes FlashInfer the selection in the first place:

```
Using FLASHINFER attention backend out of potential backends: ['FLASHINFER' ...
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
  attention backend FlashInferBackend (support: UNIFORM_SINGLE_TOKEN_DECODE)
setting cudagraph_mode=PIECEWISE
```

The obvious workaround does not work. Forcing `--attention-backend TRITON_ATTN`
produces **both** of these in one run:

```
Using AttentionBackendEnum.TRITON_ATTN backend.
Using FLASHINFER attention backend out of potential backends: ['FLASHINFER' ...
```

One attention group honours the override; a second one still auto-selects
FlashInfer. Because `compilation.py` gates on `min_cg_support` — the *minimum*
across every group — that single escapee drops the whole model to `PIECEWISE`
anyway. Measured throughput confirms it: 237.3 tok/s (auto), 233.7 (FLASHINFER),
229.2 (TRITON_ATTN) — three arms that all ran `PIECEWISE`, so what they compare
is noise, not graph modes.

So the cost of the downgrade is **still unmeasured**, and the fix is not the one
originally proposed. Making the selector prefer a full-graph-capable backend
would not have helped, because the group that escapes is not going through the
override at all.

**The escapee is the MTP drafter, and it is deliberate.** Re-running with
`--spec-tokens 0` leaves exactly one backend line and no downgrade:

```
Using AttentionBackendEnum.TRITON_ATTN backend
```

`vllm/v1/spec_decode/llm_base_proposer.py:1292` says why:

```python
# Note (matt): Never inherit the attention backend from base, because there are
# many opportunities for incompatibility, so we always independently autoselect
# unless explicitly specified in the speculative config.
base = replace(base, attention_config=replace(
    base.attention_config, backend=spec_cfg.attention_backend))
```

The drafter throws away the target's backend by design and auto-selects, which
on this hardware means FlashInfer. Since `min_cg_support` is a minimum across
groups, the drafter alone decides the CUDA-graph mode for the whole model.

**Neither override alone is enough — the target and the drafter must both be
set.** Setting only `SpeculativeConfig.attention_backend` leaves the target on
FlashInfer; setting only `--attention-backend` leaves the drafter on it. Either
survivor forces `PIECEWISE`. With both set, the downgrade disappears and full
graphs are captured:

```
16 Capturing CUDA graphs (decode, FULL)
10 Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)
 1 Using AttentionBackendEnum.TRITON_ATTN backend
```

| configuration | graph mode | tok/s | ms/token |
|---|---|---|---|
| default (both auto) | PIECEWISE | 234.5 | 4.265 |
| target only | PIECEWISE | 229.2 | 4.363 |
| drafter only | PIECEWISE | 242.3 | 4.127 |
| **both TRITON_ATTN** | **FULL** | **333.0** | **3.003** |

**1.42x on single-stream decode**, batch 1, three speculative tokens — from
configuration, with no engine change. The three PIECEWISE rows agree within
noise, which is the control: they differ only in which backend ran, and that
barely matters. What matters is the graph mode.

Measured on sm_89 with the 2B proxy, so the number does not transfer. The
direction should, and probably understates the 27B: that model has 64 layers to
this one's 24, and TP=2 adds a second process worth of launch overhead per
step — both of which are exactly what full graphs remove.

What makes this fork-worthy rather than a config note: the failure is silent,
inverted, and not fixable by the obvious knob. A one-layer draft head, chosen
for being cheap, removes full CUDA graphs from all 64 layers of the target; the
only trace is a warning naming a backend the user never selected; and a user who
reads that warning and forces the backend it names still gets `PIECEWISE`,
because the override does not reach both config trees. Candidate change: when
spec decode is on, make the drafter's independent auto-selection prefer a
backend whose `AttentionCGSupport` is at least `UNIFORM_BATCH`, and fail loudly
rather than downgrading when the user has forced one that is not.

*Also learned:* `llm.llm_engine.vllm_config.compilation_config.cudagraph_mode`
reported `FULL_AND_PIECEWISE` in the parent while the engine core was running
`PIECEWISE`. The downgrade happens in the engine-core process and the parent's
copy never sees it. Parse the log; do not trust the attribute.

**H3 — TP=2 all-reduce over the host bridge.** No P2P, no NVLink, no symmetric
memory on sm_86, so the custom all-reduce and symm-mem paths are all unavailable
and every reduction falls to PYNCCL through the CPU — two per layer, 64 layers,
four forwards per step.
*Probe:* `probes/allreduce_rate.py`. *Fix if true:* the honest fix is an NVLink
bridge. Failing that, a 2-GPU staged reduce or a quantized-payload reduce.

**H4 — Prefix reuse is quantized to the forced block size. CONFIRMED.**

`Platform.check_and_update_config` raises the attention block size until one
attention block holds a whole mamba state. The 2B proxy lands on 544 tokens; the
production 27B lands on 1600. Measured reuse on an identical repeat prompt:

```
resolved block_size: 544
prefix   cached  cached %  blocks held    waste
   400        0      0.0%          544      144
   800      544     68.0%         1088      288
  1600     1088     68.0%         1632       32
  2400     2176     90.7%         2720      320
  3200     2720     85.0%         3264       64
  4800     4352     90.7%         4896       96
  6400     5984     93.5%         6528      128
```

Every `cached` value is an exact multiple of 544, so reuse is strictly
block-quantized, and the trailing partial block is never cached.

The consequence is not the one to reach for first. Wasted KV is small (32–320
tokens held and unused). What matters is the top row: **a prompt shorter than
one block gets zero reuse — not partial, none.** Scaled to the 27B's 1600-token
block, every request under 1600 tokens can never hit the prefix cache at all,
and production's mean prompt on that model is 1,173 tokens. The median request
is structurally excluded.

That also reframes the "36,364 external prefix-cache queries, 0 hits" figure:
worth re-checking against prompt length before blaming the offload connector.

*Fix:* decouple the attention block size from the mamba page size for hybrid
models, so the two live in separately-sized pools. This is the largest change on
the list and the one most likely to be worth upstreaming. Cheap partial
mitigation to measure first: whether a smaller `mamba_block_size` under
`mamba_cache_mode=all` buys back granularity without costing GDN kernel
throughput.

## Where these were measured

The 3090 Ti pair serves production and was unavailable, so the first round ran
on the single RTX 4090 that normally serves Gemma, via the loaner overlay
described in `k8s/README.md`. That card is **sm_89, not sm_86**, and the
difference is not cosmetic: it has FP8 tensor cores, a different FlashAttention
support matrix, and a different set of candidate attention backends for the same
model.

So the rule for anything measured there: **mechanism transfers, magnitude does
not.** Whether the selector picks a backend that forfeits full CUDA graphs, and
whether the hybrid layout forces a 1600-token block, are properties of the model
and of vLLM's own logic. How many microseconds that costs is a property of the
card, and has to be re-measured on the pair before it goes in a commit message.

The proxy model is `cyankiwi/Qwen3.5-2B-AWQ-4bit`: same architecture as the
27B — `Qwen3_5ForConditionalGeneration`, GDN linear attention interleaved with
full attention at interval 4, `mtp_num_hidden_layers: 1`, `head_dim` 256 — at a
size that fits one card.

One trap already caught, recorded so it is not walked into twice: the first run
of `cudagraph_modes.py` omitted `kv_cache_dtype=fp8` and passed the backend
override as a bare string. fp8 changes which backends are candidates, and
`AttentionBackendEnum` is a plain `Enum` that silently ignores a string, so both
arms selected `FLASH_ATTN` and the probe cheerfully reported a comparison of a
configuration against itself. The probe now refuses a run whose selected backend
is not the one requested.

## Method

Measure first. The `pascal` branch found a 4x by profiling a decode step by
kernel rather than reasoning about FLOPs, and the same discipline applies here
with more force, because the headline number on this branch is contaminated by
a known confounder (H1) that costs nothing to eliminate.

Order of work: H1 (exclude the confounder) → re-measure the baseline under
sustained load with locked clocks → profile by kernel → then, and only then,
pick between H2, H3 and H4 on the evidence.

## Layout

```
ampere/
  k8s/          dev pod holding both cards, plus a local overlay you supply
  probes/       one falsifiable question each, cheap to run
  scripts/      environment setup, decode profile, benchmark
```

Unlike `pascal`, the first increment needs **no compiler**: H1 through H4 are
Python-level, so the dev pod runs the stock `vllm/vllm-openai:v0.27.1-x86_64`
image with this tree installed over it. A full CUDA build is only required if
H3 turns into kernel work; `scripts/build.sh` covers that case.
