# Pascal (SM 6.1) support

Run modern vLLM model paths on Pascal GPUs, as a one-off fork rather than a
maintained track of upstream.

## Requirement: The fork targets sm_61 and drops what sm_61 cannot run

vLLM SHALL build for compute capability 6.1, and kernel families requiring newer
hardware SHALL be excluded from that build rather than compiled and left
unreachable.

Excluded, with cause:

- FlashAttention (FA2 is sm_80+, FA3 sm_90+) — skipped in CMake and `setup.py`
- Marlin (sm_75+), Machete (sm_90), CUTLASS W4A8 (sm_90) — self-excluded by arch
- QuTLASS — requires CUDA 12.8+
- MoE W4A16 fp16 — `atomicAdd(__half*)` is sm_70+

#### Scenario: Building with a single Pascal arch
- **WHEN** `TORCH_CUDA_ARCH_LIST=6.1` is set
- **THEN** the build completes with no compile errors
- **AND** no FlashAttention translation units are compiled
- **AND** `vllm._C_stable_libtorch` imports on a device reporting `(6, 1)`

#### Scenario: Building for tensor-core hardware is unaffected
- **WHEN** the arch list contains any capability >= 8.0, or is unset
- **THEN** FlashAttention is built exactly as upstream would build it

## Requirement: The toolchain is pinned to what still emits Pascal code

The fork SHALL depend on CUDA 12.6 and the cu126 PyTorch wheel channel.

CUDA 13 dropped sm_61 codegen. cu126 is the only PyTorch channel still shipping
Maxwell/Pascal/Volta cubins; its arch list contains `6.0` and not `6.1`, which
suffices because CUDA runs an `sm_X.y` cubin on `sm_X.z` whenever `z >= y`.

`requirements/pascal.txt` SHALL replace `requirements/cuda.txt`, which pins torch
from the default index and pulls tensor-core and CUDA-13-only libraries.

#### Scenario: Verifying the wheel can run here
- **WHEN** `torch.cuda.get_arch_list()` is queried on the target device
- **THEN** it contains an arch with the same major version and a minor version
  no greater than the device's

## Requirement: Absent accelerator libraries degrade, they do not abort

Optional tensor-core libraries SHALL NOT be imported at module scope in a way
that prevents startup when absent.

This is not a theoretical concern: an unguarded `vllm.vllm_flash_attn` import in
`fa_utils` made every model architecture uninspectable, and an unguarded
`flashinfer` import in the sampler killed engine initialization.

#### Scenario: Selecting an attention backend without FlashAttention
- **WHEN** the FA extension is not built
- **THEN** the selector reports FlashAttention invalid and selects `TRITON_ATTN`
- **AND** engine startup succeeds

## Requirement: Quantization capability floors reflect the kernel that will run

A quantization scheme SHALL NOT be rejected for a capability floor belonging to
a kernel that is not the one selected.

`TritonW4A16LinearKernel` reports `get_min_capability() == 0` and supports
`uint4`/`uint4b8` at group sizes `[-1, 32, 64, 128, 256]`, so wNa16 runs on
Pascal even though Marlin does not.

#### Scenario: Loading a compressed-tensors W4A16 checkpoint on Pascal
- **WHEN** the checkpoint is asymmetric int4, group size 128
- **THEN** the model loads and `TritonW4A16LinearKernel` is selected
- **AND** schemes that genuinely need newer hardware still fail, naming the
  scheme rather than the container format

## Requirement: torch.compile is disabled below sm_70

`current_platform.simple_compile_backend` SHALL be `"eager"` on devices below
compute capability 7.0.

Inductor applies a floor that is stricter than Triton's own: it raises
`GPUTooOldForTriton` below sm_70 even though hand-written Triton kernels compile
and run correctly on sm_61. Fourteen call sites read this attribute at import
time to decorate functions, and `enforce_eager` does not cover them because it
disables compilation of the model, not of standalone helpers.

#### Scenario: Sampling with logprobs on Pascal
- **WHEN** a request asks for logprobs, invoking the compiled
  `batched_count_greater_than`
- **THEN** generation completes without `GPUTooOldForTriton`

## Requirement: Correctness is demonstrated, not inferred from fluency

The gate SHALL check kernel outputs numerically, because a miscompiled kernel
typically still produces fluent text that has drifted from the weights.

Each kernel is compared against an independent implementation of the same
mathematics: the Triton W4A16 kernel against explicit fp32 dequantize-then-matmul,
and the chunked gated delta rule against the fused recurrent form.

#### Scenario: Green gate
- **WHEN** `cyankiwi/Qwen3.5-2B-AWQ-4bit` is served text-only on a GTX 1070 Ti
- **THEN** greedy decoding produces coherent, factually correct continuations
- **AND** MTP speculative decoding works via `speculative_config` method `mtp`
