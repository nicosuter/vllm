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
| Idle state | Both cards sit at **pstate P8, 210 MHz, ~20 W** between requests |

The three facts that shape everything below: there is no fast path between the
two cards, sm_86 cannot do FP8 math, and the cards are usually asleep.

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

**H1 — Idle clocks.** The engine goes fully idle between requests and the cards
drop to P8/210 MHz. Interactive traffic pays the ramp on every request.
*Probe:* `probes/clock_ramp.py`. *Fix if true:* lock minimum SM clocks; this is
deployment config, not a fork change, but it must be excluded before anything
else is believed.

**H2 — Full CUDA graphs are silently disabled.** vLLM picks FlashInfer
(`Using FLASHINFER ... out of potential backends: ['FLASHINFER', 'TRITON_ATTN']`),
then discovers FlashInfer only declares `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`
and downgrades `FULL_AND_PIECEWISE` → `PIECEWISE` because spec decode is on
(`vllm/config/compilation.py`). `TRITON_ATTN` declares `AttentionCGSupport.ALWAYS`
and was an available candidate. With 64 layers and four forwards per MTP step at
batch 1, the launch overhead this gives back is not small.
*Probe:* `probes/cudagraph_modes.py`. *Fix if true:* make backend selection
prefer a full-graph-capable backend when spec decode is enabled, instead of
picking one and then quietly degrading the graph mode.

**H3 — TP=2 all-reduce over the host bridge.** No P2P, no NVLink, no symmetric
memory on sm_86, so the custom all-reduce and symm-mem paths are all unavailable
and every reduction falls to PYNCCL through the CPU — two per layer, 64 layers,
four forwards per step.
*Probe:* `probes/allreduce_rate.py`. *Fix if true:* the honest fix is an NVLink
bridge. Failing that, a 2-GPU staged reduce or a quantized-payload reduce.

**H4 — Prefix caching is defeated by the mamba page size.** The hybrid layout
forces attention block size to 1600 tokens so the attention page is at least as
large as the mamba page (`vllm/v1/kv_cache_interface.py`), then pads three
layers for a further ≤6.25% KV waste. Prefix matches can only land on 1600-token
boundaries. The production engine has taken 36,364 external prefix-cache
queries and served zero hits.
*Fix if true:* decouple attention block size from the mamba page size for hybrid
models. This is the largest change on the list and the one most likely to be
worth upstreaming.

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
