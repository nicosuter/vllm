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

One GTX 1070 Ti: GP104, SM 6.1, 8 GB VRAM, NVIDIA driver 580.167.08. The host
has 28 vCPU and 24 GB RAM, which is what sets `MAX_JOBS=8` for the build (nvcc
peaks near 2 GB per translation unit).

The manifests in `pascal/k8s/` request a GPU generically:

```yaml
runtimeClassName: nvidia
resources:
  limits:
    nvidia.com/gpu: 1
```

Pinning to a particular node or storage class is deployment-specific and is
supplied by a local kustomize overlay that is not committed. See
`pascal/k8s/README.md`.

### What SM 6.1 cannot do

These constraints drive nearly every decision in this fork:

| Constraint | Consequence |
|---|---|
| No tensor cores | FlashAttention, FlashInfer, Marlin, Machete, CUTLASS SM80+ and every FP8 *GEMM* are unavailable and get compiled out. FP8 *checkpoints* still load — see below |
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
kubectl apply -k pascal/k8s/local
kubectl -n vllm-pascal create configmap triton-probe-src \
  --from-file=triton_probe.py=pascal/probes/triton_probe.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n vllm-pascal logs -l job-name=triton-probe -f
```

### Results

Run 2026-08-15 on the GTX 1070 Ti, torch `2.13.0+cu126`, triton `3.7.1`:

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
kubectl apply -k pascal/k8s/local
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

The gate model is compressed-tensors W4A16 `pack-quantized`, **symmetric, group
size 32** (from the checkpoint's own `quantization_config`). Marlin is sm_75+,
Machete sm_90, CUTLASS W4A8 sm_90 — none available.

What it actually lands on is **`ExllamaLinearKernel`**, which the engine reports
at startup:

```
[compressed_tensors_wNa16.py:146] Using ExllamaLinearKernel for CompressedTensorsWNA16
```

That is a hand-written CUDA kernel declaring `get_min_capability() == 60`, i.e.
Pascal-era by construction. Symmetric int4 is `scalar_types.uint4b8`, which is
in its `SUPPORTED_QUANT_TYPES`, and it requires fp16 activations, which is what
this fork forces anyway.

> An earlier revision of this file claimed the path resolved to
> `TritonW4A16LinearKernel` at group size 128, asymmetric. All three details
> were wrong. That kernel's docstring describes it as "Triton-based W4A16 GEMM
> kernel for ROCm MI300"; it is never selected here. The distinction matters
> because it moves the hot loop from Triton to nvcc-compiled CUDA, which is what
> made the fp16-rate problem below findable at all.

Attention resolves differently, and does go through Triton:
`TritonAttentionBackend.supports_compute_capability` returns `True`
unconditionally, and `gdn_attn` handles the 18 Gated DeltaNet layers.

Attention resolves the same way: `TritonAttentionBackend.supports_compute_capability`
returns `True` unconditionally, and `gdn_attn` handles the 18 Gated DeltaNet
layers.

### FP8 checkpoints load, because e4m3 is a storage format

"No FP8 hardware" rules out an FP8 *multiply*, which this fork was never going
to use — there are no tensor cores to issue one from. It says nothing about
reading the bytes. e4m3 is 1 sign, 4 exponent (bias 7), 3 mantissa; dropping
those fields into the fp32 positions gives the right number with the wrong
exponent bias, off by a constant `2^120` for every input. One multiply corrects
all of them.

`pascal/probes/fp8_decode.cu` checks that on the card and measures what it costs:

```
decode vs definition: 254/254 finite patterns exact (14 subnormal), 0 wrong
  fp8  decode    2.412 ms    222.6 GB/s   222.61 Gweight/s
  fp16 convert   4.806 ms    223.4 GB/s   111.71 Gweight/s
  fp8 delivers 1.99x the weights/s at 100% of fp16's bandwidth
```

Two results worth keeping. `cuda_fp8.h` does compile and run correctly on
sm_61 — NVIDIA ships a software path — but it is 1.10x slower in the inner loop
(2.645 ms, 203 GB/s) than moving the fields by hand. And **`-ftz=true` or
`--use_fast_math` silently zeroes all 14 subnormal patterns**: the decode routes
a subnormal through fp32, and a flush-to-zero build discards it without
complaining. That is a miscompile that produces plausible weights.

What is actually wired up is `PascalFP8ScaledMMLinearKernel`, last in the
candidate list behind Humming and Marlin, claiming only capabilities below 7.5.
It decodes once at load and hands fp16 to cuBLAS, which this fork already runs
with `CUBLAS_COMPUTE_32F`. Two capability floors moved from 75 to 60 to let the
choice happen at all — the same argument as `wNa16` above: 75 described the best
kernel of its day, not the format.

Which FP8 actually runs, then. Per-tensor and per-channel scales do;
**block-wise `[128, 128]` does not**, and that covers Qwen's own `-FP8`
releases, whose `quantization_config` carries `weight_block_size`. Block quant
routes to a separate candidate list this kernel is not in, and none of its
members can run here. That used to surface as a Triton JIT error four layers
down — `type fp8e4nv not supported in this architecture` — because
`TritonFp8BlockScaledMMKernel` claimed every CUDA device without checking. It
now declines below 7.5, so the failure reads as intended:

```
ValueError: Failed to find a kernel that can implement the ScaledMM linear layer. Reasons:
  CutlassFp8BlockScaledMMKernel The device compute capability of 61 is not supported..
  TritonFp8BlockScaledMMKernel Triton has no fp8e4nv below compute capability 7.5..
```

Prefer the `-FP8-dynamic` checkpoints (RedHatAI and the rest of the
compressed-tensors family), which are per-channel.

Lowering the `fp8` floor has one consequence worth naming: `Fp8Config` also
covers MoE, and the capability check in `vllm/config/vllm.py` is a single gate
for the whole method. An FP8 **MoE** checkpoint now gets past it and fails
further in, at `select_fp8_moe_backend`'s `NotImplementedError`, instead of at
the clean "not supported for the current GPU". Both are loud; only the message
got worse. Linear is what was made to work.

Because vLLM already routes W8A8 FP8 to `CompressedTensorsW8A16Fp8` below
capability 8.9, an ordinary `FP8-dynamic` checkpoint lands there without
knowing anything about this card. All 112 linear layers of
`Qwen2.5-1.5B-Instruct-FP8-dynamic` resolve to it, and the activation scale is
dropped rather than honoured — quantizing activations only pays against an fp8
MMA, and there is none to aim at.

**The byte saving is not banked yet.** Decoding at load means a 2.09 GiB fp8
checkpoint occupies 2.98 GiB resident, the footprint of the fp16 model, and
decode runs at **57.5 tok/s at batch 1** (17.39 ms/step) against the gate
model's 80.5 tok/s on int4 — the gap is weight bytes, which is the whole story
at batch 1. Keeping the byte narrow needs a fused decode-GEMM; the probe above
exists to show that is worth building, and it is not built.

#### The gate metric does not survive this model, and neither does fp16

`gate_check.py` reports **FAIL, worst prefix agreement 0%** here. That is not a
kernel fault. Run the identical comparison with *transformers' own fp16* on this
card instead of vLLM and it fails the same way:

| | worst prefix agreement | verdict |
|---|---|---|
| vLLM + `PascalFP8ScaledMMLinearKernel` | 0% | FAIL |
| HF transformers fp16, same card, same weights | 25% | FAIL |

Both diverge in the same place — the ordering of two distractors in a multiple
choice list, a genuine coin flip. Greedy prefix agreement measures fp16 against
fp32 on a model full of near-ties, and 32 tokens of it compounds a per-token
disagreement into a sequence one.

Per-token logits are the measurement that survives, over 102 positions:

| | top-1 agreement with CPU fp32 | mean abs logprob difference |
|---|---|---|
| HF fp16 on this card | 96.1% | 0.0735 |
| **vLLM + `PascalFP8ScaledMMLinearKernel`** | **95.1%** | **0.0408** |

The kernel tracks the fp32 reference at least as closely as ordinary fp16
inference does, and its logprobs are closer to it, because vLLM keeps more of
the surrounding arithmetic in fp32. Separately, the dequantized weight was
compared against the checkpoint directly and is bit-identical in fp32; the only
loss is the fp16 store, at 5.8e-5 mean relative error.

Worth recording because it nearly went the other way: an earlier run of this
comparison read `prompt_logprobs=0`, which returns the *actual* token's logprob
rather than the argmax. Comparing that against the reference's argmax scored
46% and looked exactly like a broken kernel.

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

### Dropping `enforce_eager` — what actually blocked it, and what it was worth

Inductor is what has the sm_70 floor. Dynamo does not, so `VLLM_COMPILE` without
inductor is tracing only and should run here. What stopped it was narrower: two
**raw Triton launches inside the traced region**. vLLM compiles with
`fullgraph=True`, so neither could graph-break, and both reached
`triton/runtime/jit.py`'s `driver.active.get_current_stream(device)` — whose
`_cuda_getCurrentRawStream` returns an `int`, which dynamo will not put in a
graph:

| Site | Reached via |
|---|---|
| `RMSNormGated.forward_cuda` → flash-linear-attention's `rmsnorm_fn` | Gated DeltaNet's output projection |
| `MRotaryEmbedding.forward_cuda` → `triton_mrope` | 2-D positions, i.e. multimodal rope |

Both are now registered with `direct_register_custom_op`, so dynamo emits one
opaque call and never sees the launch. Above sm_70 inductor owns Triton launches
and emits that stream call itself, which is why nobody upstream meets this.
`layernorm_guard.py` records that the FLA kernel used to sit behind an
`autograd.Function` that dynamo would not trace into, and that the wrapper was
removed as inference-only — removing it is what exposed the launch.

The gate model now runs without `enforce_eager` and produces output **identical
token for token** to the eager path.

**It buys nothing.** With graphs held at `full`, on the gate model:

| | ms/step | decode tok/s |
|---|---|---|
| `mode=NONE` | 12.37 | 80.82 |
| `mode=VLLM_COMPILE` | 12.46 | 80.27 |

Tracing without a compiler backend has nothing to optimise, so this is a
correctness fix and not a performance one — `enforce_eager` is no longer
*needed*, rather than newly worth dropping. The +10.5% above is CUDA graphs,
which never involved the compiler at all. `bench.py --compile` exists to keep
that separable.

### The sm_70 floor is one PTX hint wide, and inductor is worth 1.18x

The natural next question is why we cannot have a compiler backend at all. It
turns out we can: `pascal/probes/inductor_sm61.py` runs inductor on this card.

The floor is three `major >= 7` checks in torch — `has_triton`'s
`cuda_extra_check`, `CudaInterface.is_triton_capable`, and a re-raise in the
inductor scheduler. Lifting them is not enough on its own, and what stops it is
much smaller than an architecture gap:

```
ptxas error: Modifier '.evict_last' on 'ld' requires .target sm_70 or higher
```

Inductor asks for cache-eviction hints on loads. They are advisory — they tell
L1 what to discard first and change no result — so dropping them at Triton's
`_str_to_eviction_policy`, the single funnel every policy passes through, costs
a cache hint and nothing else. That is the whole of it: **the floor is one
optional decoration wide.**

One wrinkle worth recording, because it looks like a partial failure rather than
a plumbing problem: inductor compiles in worker *subprocesses*, which do not
inherit an in-process patch. 108 ptxas errors become 12, not 0.
`TORCHINDUCTOR_WORKER_START=fork` makes the workers inherit it, which beats
serialising compilation with `TORCHINDUCTOR_COMPILE_THREADS=1`.

What it is worth, graphs held at `full`, same slope measurement as `bench.py`:

| backend | ms/step | decode tok/s |
|---|---|---|
| `eager` | 12.46 | 80.28 |
| **`inductor`** | **10.55** | **94.76** |

**1.18x**, and the fastest decode recorded on this card.

Where it comes from, by `profile_decode.py --backend none` against
`--backend inductor` (graph capture off in both, since it collapses the kernel
boundaries this reads):

| | eager | inductor |
|---|---|---|
| total device time | 414.2 ms | 358.9 ms |
| kernel launches | 37,818 | **15,716** |
| cuBLAS GEMV | 154.1 ms | 154.0 ms |
| exllama quantized GEMM | 149.0 ms | 148.6 ms |
| attention + GDN + conv1d | 29.9 ms | 28.8 ms |

Every GEMM is untouched, which is the point: inductor does not go near the two
kernels that own 73% of the time. What disappears is the small stuff around
them. `unrolled_elementwise_kernel` alone was 20.8 ms across **5,466** launches;
with the reductions, the rsqrt, `act_and_mul` and half a dozen flavours of
`vectorized_elementwise`, roughly 80 ms of elementwise and reduction traffic
becomes about 10 ms of `triton_poi_fused_*` and `triton_red_fused_*`. Launches
more than halve.

So the earlier guess was wrong twice over. The 2.8% figure recorded above
answers "is attention the bottleneck", and reusing it to predict fusion headroom
was a category error: the fusable traffic was never attention, it was the ~20%
of device time sitting in "everything else" behind the GEMMs. Fewer launches
also matters here for the same reason CUDA graphs did — at 12 ms/step there is
not much step to hide launch overhead in.

Not bit-identical, and it should not be expected to be: fusion reassociates
reductions. Against the eager path's own greedy output, worst prefix agreement
was **97%** — a single token differing at position 31 of 32 on one prompt, with
the other three identical. A second run at `COMPILE_THREADS=1` scored 100%, so
the disagreement moves between runs, which is what autotuning picking different
configs looks like. Late divergence on a near-tie is what the gate already
tolerates; an early one would not be.

Wired in `vllm/platforms/cuda.py` alongside the existing sm_70 block, with no
switch of its own. `enforce_eager` already means "compile nothing", and turns
graph capture off with it, so an eager run never reaches this backend and never
pays for it. A non-eager run gets compilation and capture *together*, which is
the only combination worth having — capture without compilation is the middle
state this fork sat in, and it leaves the 1.18x on the table for nothing.

The two platform attributes are deliberately split: `simple_compile_backend`
stays `eager`, because that is what the fourteen `@torch.compile` helper sites
read and nothing measured says they need inductor, while `get_compile_backend()`
— which the *model* compile resolves through — returns `inductor`. The lift
fails closed: if triton or the torch internals it patches move, it reports so
and the backend stays eager rather than half-applying.

So non-eager changes the answers slightly, and `--enforce-eager` is the way out.
Verified through the default path with no explicit `compilation_config`: the
gate model reports `backend: inductor`, zero ptxas errors, 94.88 tok/s at
10.54 ms/step, and greedy output matching the eager reference;
`Qwen2.5-1.5B-Instruct-FP8-dynamic` likewise loads and generates correctly.

## Model coverage

Which checkpoints run on this card, and why the ones that do not fail:
[`pascal/MODELS.md`](MODELS.md).

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

| | |
|---|---|
| Weights on GPU | 1.83 GiB (1.88 with MTP) |
| KV cache | 3.7 GiB / **173,494 tokens** |
| Attention backend | `TRITON_ATTN` |
| Compile backend | `inductor` unless `--enforce-eager` (helpers stay `eager`) |
| Quantized GEMM | `ExllamaLinearKernel` |

### Performance

Decode throughput, batch 1, measured by `pascal/scripts/bench.py`:

| | ms/step | decode tok/s | vs baseline |
|---|---|---|---|
| as ported | 55.34 | 18.07 | — |
| fp32 `dot22_8_f` | 32.00 | 31.25 | 1.73× |
| + fp32 int4 dequant | 13.73 | 72.82 | 4.03× |
| + CUDA graphs (`full`), no compiler | 12.37 | 80.82 | 4.47× |
| **+ inductor, i.e. any non-eager run** | **10.54** | **94.88** | **5.25×** |

The third row is a waypoint rather than a configuration: graph capture without
compilation is what this fork could reach before inductor ran, and there is no
reason to run it now. Eager or the last row.

Both changes are the same finding applied twice: `__hfma2` runs at 1/56 the
fp32 rate on this card (`pascal/probes/fp16_rate.cu`), and the exllama GEMM --
which is 75% of device time and every quantized linear layer -- was built
entirely out of it. Nothing upstream guards this because no other supported
architecture is penalised for native fp16.

A third instance of the same defect lives in prefill, which shares none of
that code: above `MAX_Q_GEMM_ROWS` the exllama path reconstructs an fp16 weight
matrix and calls cuBLAS. It called `cublasHgemm`, which asks for half *compute*,
and cuBLAS honours that here:

| shape | `cublasHgemm` | `cublasGemmEx` fp32 | |
|---|---|---|---|
| prefill512 mlp | 100.50 ms | 1.87 ms | 53.8× |
| prefill2048 mlp | 383.03 ms | 8.13 ms | 47.1× |

**Not attention.** Triton attention, Gated DeltaNet and conv1d together are 2.8%
of device time. The Triton FMA lowering that this fork was expected to live or
die by costs almost nothing on the decode path.

**CUDA graphs: measure them per regime.** At 55 ms/step they were worth nothing
(55.34 / 56.07 / 55.55 for `none` / `full_decode_only` / `full`, with 35 graphs
genuinely captured — capture works on sm_61). At 13.7 ms/step the same ~1.3 ms
of launch overhead is worth **+10.5%** (12.43 ms/step, 80.5 tok/s), and
`full_decode_only` captures all of it. The conclusion inverted without the
hardware changing; only the denominator did.

**MTP inverted too.** It was 1.83× against the original baseline and is now a
regression (68.4 tok/s against 72.8). Speculative decoding amortises per-step
cost, and there is 4× less of it to amortise while the drafter's overhead is
unchanged.

### Batching and prefill

Aggregate decode throughput, since a step reads the weights once regardless of
how many sequences share it — and the KV cache holds 173k tokens, so capacity is
not the limit:

| batch | ms/step | decode tok/s |
|---|---|---|
| 1 | 13.95 | 71.7 |
| 4 | 18.71 | 213.7 |
| 16 | 30.00 | 533.4 |
| 32 | 53.36 | 599.7 |
| 48 | 68.72 | 698.5 |
| 64 | 84.26 | 759.6 |
| 128 | 151.66 | 844.0 |

Measuring this exposed a third fix. At upstream's `MAX_Q_GEMM_ROWS = 50`, batch
64 ran *slower* than batch 48 (532 against 700): crossing the threshold
abandons the fused kernel for reconstruct-then-cuBLAS, whose whole-matrix
dequantize is a fixed per-forward cost that does not amortise until far larger
batches. That constant was chosen for hardware where the reconstruct is cheap
beside a tensor-core GEMM, and both sides of the comparison had just moved here
by very different factors. Raising it to 256 removes the cliff and is worth
**+42.7%** at batch 64 and **+10.9%** at batch 128.

Prefill, measured separately because it runs the reconstruct path at any prompt
length worth the name: **1302.9 tok/s** (767.5 µs per prompt token) against a
~383 µs/token compute roofline. Unlike decode, prefill is compute-bound — which
is the one place `dp4a` would still have headroom.

Throughput is measured by slope -- two output lengths, differenced -- so prefill
and setup cancel rather than being smeared into the number. The v1 figure of
11.71 tok/s quoted earlier in this project was vLLM's own end-to-end number over
four prompts and is not comparable; 18.07 is the same build measured this way.

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
