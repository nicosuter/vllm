# vllm-pascal

A one-off hard fork of vLLM that runs modern model paths on Pascal (SM 6.1).

This is **not** a maintained track of upstream. It is shipped once and refreshed
rarely, when a new model family is wanted. Everything here optimizes for being
easy to re-apply over a future upstream tag, not for being continuously merged.

## Green gate

`cyankiwi/Qwen3.5-2B-AWQ-4bit` generates correct text on a GTX 1070 Ti, with MTP
speculative decoding working.

- **v1 — correctness.** Text-only. Output must be coherent, and the kernels must
  agree numerically with independent implementations of the same mathematics.
- **v2 — speed, and image input.** Both are commitments, not maybes.

## Fork layout

Base is upstream vLLM **v0.27.1** (commit `6e448d0ea9`), on branch `pascal`.
Upstream is kept as the `upstream` remote:

```bash
git fetch upstream --tags        # to see what a refresh would involve
```

Our own files live under `pascal/` so they stay identifiable inside the vendored
tree. Edits to vLLM's own sources are made in place, as a hard fork should.

## The hardware

One GTX 1070 Ti in node `<node>` of the <cluster> cluster (Talos, driver
580.167.08). GP104, SM 6.1, 8 GB VRAM. The node has 28 vCPU and 24 GB RAM.

Schedule onto it with:

```yaml
runtimeClassName: nvidia
nodeSelector:
  kubernetes.io/hostname: <node>
resources:
  limits:
    nvidia.com/gpu: 1
```

### What SM 6.1 cannot do

These constraints drive nearly every decision in this fork:

| Constraint | Consequence |
|---|---|
| No tensor cores | FlashAttention, FlashInfer, Marlin, Machete, CUTLASS SM80+ and all FP8 paths are unavailable and get compiled out |
| No bfloat16 | Everything is fp16 storage with fp32 compute |
| fp16 arithmetic at 1/64 rate (HFMA2 on GP104) | fp16 is a *storage* format only; it must never become the compute type |
| INT8 `dp4a`/`dp2a` **is** available | The fast path worth reaching for in v2 |
| CUDA 13 dropped sm_61 codegen | Toolkit pinned to CUDA 12.6 |

### Why the toolchain is pinned where it is

PyTorch builds Pascal cubins in exactly one wheel channel. From
`.ci/manywheel/build_cuda.sh`, with upstream's own comment:

```
12.6) TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;..."  # Only 12.6 includes legacy Maxwell/Pascal/Volta
12.8) ...7.5;8.0;8.6;9.0;10.0;12.0
13.0) ...7.5;8.0;8.6;9.0;10.0;12.0+PTX
```

The list has `6.0` but not `6.1`. That is fine: CUDA guarantees a cubin built for
`sm_X.y` runs on `sm_X.z` whenever `z >= y`, so the `sm_60` cubins execute on this
`sm_61` card. This is what lets us avoid building PyTorch from source — install
from the cu126 index and nothing else.

## The load-bearing unknown: Triton

vLLM has **447 `@triton.jit` kernels across 180 files**, with **228 `tl.dot` call
sites**. Triton is not optional, and `tl.dot` is the risky part because it is what
normally lowers to tensor-core MMA.

This matters most for the gate model. Qwen3.5-2B is a hybrid: **18 of its 24
layers are Gated DeltaNet**, implemented only as Triton in
`vllm/third_party/flash_linear_attention/` — 6089 lines, 29 kernels, 42 `tl.dot`
sites across 7 files. Plus `causal_conv1d` (2 kernels, no `tl.dot`).

Triton does still carry a non-tensor-core lowering for `tl.dot`
(`lib/Conversion/TritonGPUToLLVM/DotOpToLLVM/FMA.cpp`). Whether the NVIDIA
backend selects it below `sm_70` is the question that sets the scope of this
whole project:

- **If `tl.dot` works on sm_61** — the FLA kernels are reusable and this is a
  build-system and backend-selection problem.
- **If it does not** — 29 Gated DeltaNet kernels need hand-written SM 6.1 CUDA,
  and the project roughly doubles.

That is an empirical question, so it is answered empirically. See
`pascal/probes/triton_probe.py` and the results recorded below.

## Probe

```bash
kubectl apply -f pascal/k8s/probe-job.yaml
kubectl -n vllm-pascal create configmap triton-probe-src \
  --from-file=triton_probe.py=pascal/probes/triton_probe.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n vllm-pascal logs -l job-name=triton-probe -f
```

### Results

Run 2026-08-15 on `<node>`, torch `2.13.0+cu126`, triton `3.7.1`:

```
torch.arch_list           PASS  GTX 1070 Ti sm_61; arch_list has sm_50/60/70/75/80/86/90; usable=['sm_60']
torch.basic_op            PASS  fp32 matmul max_abs_err=2.136e-04
torch.fp16_matmul         PASS  fp16 matmul max_rel_err=3.704e-04
torch.bf16_rejected       PASS  bf16 matmul ran and returned finite values
torch.fp16_vs_fp32_speed  PASS  fp32=6.32 TF/s  fp16=6.60 TF/s  ratio=1.04
triton.elementwise        PASS  max_abs_err=0.000e+00
triton.tl_dot.fp32        PASS  rel_err=0.000e+00
triton.tl_dot.fp16        PASS  rel_err=0.000e+00
```

Four conclusions, all of which shrink the project:

1. **Stock cu126 wheels run here.** `sm_60` cubins execute on this `sm_61` card as
   the CUDA compatibility rule promises. No PyTorch source build.

2. **`tl.dot` works on sm_61, exactly.** Triton selects its FMA lowering below
   `sm_70` and returns bit-exact results for fp16 and fp32. This is the big one:
   the 29 Gated DeltaNet kernels in `flash_linear_attention/` do not need to be
   rewritten in CUDA, and neither do the other ~420 Triton kernels in vLLM.
   **This is a build-system and backend-selection problem, not a kernel-rewrite
   problem.**

3. **fp16 is not crippled through torch's GEMM path.** fp16 measures *faster*
   than fp32 (ratio 1.04), so cuBLAS is already accumulating in fp32 rather than
   issuing HFMA2. The 1/64 native-fp16 rate is a trap only if we hand-write a
   kernel that makes fp16 the compute type. fp32 hits 6.32 TF/s, about 78% of
   this card's 8.1 TFLOPS peak.

4. **bf16 degrades rather than crashes.** It runs, emulated, and returns finite
   values. So bf16 is a performance bug on this card, not a correctness cliff —
   worth forcing to fp16, but it will not blow up if one slips through.

Triton needs a C toolchain at runtime to build its driver shim, which is why the
probe image is `python:3.12` and not `-slim`.

## Building

```bash
kubectl apply -f pascal/k8s/build-pod.yaml
kubectl -n vllm-pascal exec -i pascal-build -- bash -s < pascal/scripts/setup-build-env.sh
kubectl -n vllm-pascal exec -i pascal-build -- bash -s < pascal/scripts/build.sh
```

`TORCH_CUDA_ARCH_LIST=6.1` does more than pick codegen: it is what makes CMake
and `setup.py` drop FlashAttention, Marlin, Machete, CUTLASS SM80+, QuTLASS and
the FP8 paths, since each already selects itself out by target arch.

### What the fork had to change to compile

| Change | Why |
|---|---|
| `CUDA_SUPPORTED_ARCHS` gains `6.1` | The CUDA<12.8 branch is the only one that can carry it; CUDA 13 dropped sm_61 codegen |
| MoE W4A16 fp16 path compiled out below sm_70 | `atomicAdd(__half*)` is sm_70+. Upstream already does this for bf16 below sm_80, so it extends that guard |
| vllm-flash-attn subproject skipped | FA2 is sm_80+, FA3 sm_90+. ~200 dead translation units, and the selector falls through to Triton on ImportError |
| `setup.py` FA extensions gated the same way | Otherwise `cmake --build` is asked for `_vllm_fa2_C`, which CMake never defined |
| `requirements/pascal.txt` replaces `cuda.txt` | `cuda.txt` pins torch from the default index (cu128, no Pascal cubins) and pulls flashinfer, tilelang and two cu13-only CUTLASS packages |

Environment gotchas, all encoded in `setup-build-env.sh`:

- Ubuntu 24.04's cargo is 1.75; vLLM's workspace manifest needs `resolver = "3"`,
  so rustup is required.
- The Rust `vllm-server` crate builds prost definitions and needs `protoc`.
- `setuptools-scm` derives the version from git, so a tarball checkout needs a
  seeded repo and tag or the build aborts before compiling anything.

### How the quantized path resolves on Pascal

The gate model is compressed-tensors W4A16, asymmetric, group size 128. Marlin
is sm_75+, Machete sm_90, CUTLASS W4A8 sm_90 — none available. But
`TritonW4A16LinearKernel` reports `get_min_capability() == 0` ("Triton handles
capability checks itself") and accepts `scalar_types.uint4` (asymmetric with
explicit zeros) at group sizes `[-1, 32, 64, 128, 256]`. So the checkpoint's
exact quantization lands on a Triton kernel that we have measured working on
this card.

Attention resolves the same way: `TritonAttentionBackend.supports_compute_capability`
returns `True` unconditionally, and `gdn_attn` handles the 18 Gated DeltaNet
layers.

### The runtime changes, and the one that was not obvious

Building was the easy half. Four separate places assumed hardware we do not have
and killed the engine before or during generation:

| Change | Why |
|---|---|
| `fa_utils` imports FA lazily | `model_executor.layers.attention` pulls it in unconditionally, so a build without the FA extension made *every* model architecture uninspectable |
| `topk_topp_sampler` treats missing flashinfer as a reason, not an error | It imported flashinfer purely to decide whether to use it |
| compressed-tensors capability floors lowered to 60 | 70 (config) and 75 (wNa16) both date from when Marlin was the only wNa16 kernel |
| rotary embedding falls back to `forward_native` | Its only CUDA-specific ingredient is flash-attention's fused `apply_rotary_emb` |

The one worth remembering: **`torch.compile` has a stricter floor than Triton.**
Inductor raises `GPUTooOldForTriton` below sm_70 regardless of what Triton
itself supports — so an error that names Triton as unsupported appears on a card
where Triton demonstrably works, and where vLLM had already compiled and run
hundreds of Triton kernels. It surfaced late, mid-generation, inside the
sampler's `@torch.compile`d `batched_count_greater_than`, well after startup and
weight loading had succeeded. `enforce_eager` does not cover it: that disables
compilation of the *model*, not of standalone decorated helpers.

Fourteen sites read `current_platform.simple_compile_backend` at import time to
decorate functions, so the fix belongs on the platform, once:
`CudaPlatform.simple_compile_backend = "eager"` below sm_70.

## Model coverage

Everything here is tested on the same single GTX 1070 Ti. Quantized variants are
substituted wherever the headline model ships unquantized, because 8 GB does not
hold bf16 weights plus a KV cache — and because compressed-tensors W4A16 lands on
the Triton kernel already verified on sm_61.

| Requested | Tested as | Size | Status |
|---|---|---|---|
| `cyankiwi/Qwen3.5-2B-AWQ-4bit` | as requested | 2.4 GB | **Green gate.** 11.7 tok/s, 21.4 with MTP |
| `ibm-granite/granite-4.1-3b` | `cyankiwi/granite-4.1-3b-AWQ-INT4` | 2.3 GB | **Works.** 16.8 tok/s |
| `Qwen/Qwen3-VL-Embedding-2B` | as requested, fp16 | 4.3 GB | **Works.** dim 2048, related pair leads by 0.52 cosine |
| `Qwen/Qwen3-VL-Reranker-2B` | as requested, fp16 | 4.3 GB | **Works** via yes/no logits; vLLM's score() path cannot load it |
| `google/gemma-4-E2B-it-qat-q4_0-unquantized` | `google/gemma-4-E2B-it-qat-w4a16-ct` | 8.3 GB | needs `cpu_offload_gb`; see below |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | — | 3.9 GB | **Not supported by vLLM** (arch absent upstream too) |

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

### Text-to-speech does not run on vLLM, on any GPU

Both TTS references were checked and neither is a Pascal problem — vLLM has no
audio-generation path whatsoever.

**`Qwen/Qwen3-TTS-12Hz-1.7B-Base`** declares `Qwen3TTSForConditionalGeneration` /
`qwen3_tts`. That architecture appears **nowhere in vLLM**, not in our v0.27.1
base and not in upstream `main` either. VRAM is not the limit: at 3.9 GB it
would fit this card with room to spare, and the 0.6B variant more so. The
architecture is simply unimplemented, so the smaller model does not help.

**`hexgrad/Kokoro-82M`** (dropped from scope, recorded because it was checked) is
further out still: it is not a transformer LM at all. Its `config.json` describes
StyleTTS2 — an `istftnet` vocoder, a PLBERT text encoder, a duration predictor,
`style_dim`/`n_mels` — ships a single `.pth`, and declares no `architectures`,
no `model_type`, and no `library_name`.

The natural next guess, that vLLM's "omni" models might provide a way in, does
not work either. Every omni entry is a *thinker*:
`Qwen2_5OmniThinkerForConditionalGeneration`,
`Qwen3OmniMoeThinkerForConditionalGeneration`. `qwen2_5_omni_thinker.py` is
explicit when loading weights:

```python
loader = AutoWeightsLoader(self, skip_prefixes=["talker.", "token2wav."])
```

The talker and token2wav stacks — the parts that emit audio — are skipped
outright. vLLM's audio support is uniformly audio **in**, text **out**
(`qwen2_audio`, `granite_speech`, `kimi_audio`, `qwen3_asr`, ...).

Serving TTS on this card means a different runtime, not a different quantization.
Qwen3-TTS runs under `transformers`; Kokoro under its own `kokoro` package or
ONNX. Both are small enough that a 1070 Ti handles them without a serving engine.

## Status

**v1 green gate: met, including MTP.** `cyankiwi/Qwen3.5-2B-AWQ-4bit` generates
correct text on the GTX 1070 Ti, and MTP speculative decoding works.

```
>>> 'The capital of Switzerland is'
    ' Bern. ... Bern is the capital city of Switzerland,'
>>> 'def fibonacci(n):\n    '
    ' if n == 0:\n         return 0\n     elif n == 1:\n         return 1\n     else:\n         return fibonacci'
>>> 'In one sentence, explain why the sky is blue:'
    'The sky appears blue because the short wavelengths of sunlight are scattered
     more efficiently by air molecules than the longer wavelengths, ... known as Ray[leigh]'
>>> 'List three prime numbers greater than 100:'
    ' 101, 103, 107.'
```

Measured on the card, text-only, `--enforce-eager`, no performance work yet:

| | without MTP | with MTP (`--mtp 1`) |
|---|---|---|
| Output throughput | 11.71 tok/s | **21.39 tok/s** (1.83×) |
| Weights on GPU | 1.83 GiB | 1.88 GiB |
| KV cache | 3.7 GiB / **173,494 tokens** | |
| Attention backend | `TRITON_ATTN` | |
| Compile backend | `eager` | |

MTP needs no second checkpoint: vLLM resolves the architecture to `Qwen3_5MTP`
and builds the drafter from the same weights, which is why this checkpoint
mattered — it is the one that kept `mtp.*` quantized instead of stripping it.
The extra ~50 MiB is the MTP head.

The KV cache figure is worth noting: there was never any need for the
short-context compromise that an 8 GB card seems to imply. At 2.4 GB of INT4
weights, the card has room to spare — including for the v2 vision tower.

First startup is slow (many minutes) because Triton autotunes and compiles every
kernel for sm_61 — 250+ cubins, ~300 MB.

Two things about that cache are worth knowing, both learned by measuring rather
than assuming:

- **It is not persistent by default.** Triton writes to `~/.triton`, which on a
  container is the overlay filesystem and vanishes with the pod. `build-pod.yaml`
  therefore sets `TRITON_CACHE_DIR=/work/.triton` so it lands on the PVC.
- **It is shape-specialized, so a warm start is not a free start.** Re-running
  the same command still compiles: changing sequence length, batch shape, or
  enabling MTP produces new specializations. Measured: an identical repeat of
  the gate run took **195 s** end to end (13.7 tok/s), against many minutes cold.
  Faster, not instant.

### Kernel correctness

Coherent text is not proof: a miscompiled kernel usually still reads fine. Each
kernel is therefore checked against an independent implementation of the same
mathematics (`pascal/probes/kernel_check.py`):

```
w4a16.triton_vs_fp32_dequant  PASS  rel_err=3.918e-04  (M=8 K=2048 N=512 group=128)
gdn.chunked_vs_recurrent      PASS  rel_err=8.591e-04  (T=128 H=4 K=64 V=64)
```

Both are at fp16 rounding. The first covers every linear layer in the model —
the kernel Pascal reaches only because Marlin and Machete opt out. The second is
a real cross-check rather than a kernel compared against itself, since prefill
uses the chunked form and decode the recurrent one.

The originally intended check, greedy decoding under HF transformers on CPU in
fp32, is unavailable: transformers rejects this checkpoint with `strategy group
requires group_size to be set to a positive value`, a disagreement between the
checkpoint's `quantization_config` and the installed `compressed-tensors`
validation. That is a library-version problem on the reference side and says
nothing about Pascal.
