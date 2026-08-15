# vllm-pascal

A one-off hard fork of vLLM that runs modern model paths on Pascal (SM 6.1).

This is **not** a maintained track of upstream. It is shipped once and refreshed
rarely, when a new model family is wanted. Everything here optimizes for being
easy to re-apply over a future upstream tag, not for being continuously merged.

## Green gate

`cyankiwi/Qwen3.5-2B-AWQ-4bit` generates correct text on a GTX 1070 Ti, with MTP
speculative decoding working.

- **v1 — correctness.** Text-only. Output must be coherent and must agree
  numerically with an HF transformers reference on the same prompts.
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

_Pending first run._
