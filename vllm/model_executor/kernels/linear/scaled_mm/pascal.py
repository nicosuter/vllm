# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Static128BlockSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


class PascalFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """Weight-only FP8 for cards with neither FP8 units nor tensor cores.

    FP8 Marlin already covers "no FP8 hardware" down to capability 7.5, but it
    gets its speed from `mma.sync`, so it stops where tensor cores do. Below
    that the checkpoint is still perfectly readable: e4m3 is a storage encoding,
    and nothing about decoding it needs an FP8 unit. `pascal/probes/fp8_decode.cu`
    measures the decode on this card at 1.99x the weights per second of fp16
    while holding the same 223 GB/s -- i.e. free, because the load is what costs.

    What this kernel does not do is keep the weights narrow. It decodes once at
    load and hands fp16 to cuBLAS, which this fork already runs with
    CUBLAS_COMPUTE_32F: fp16 storage, fp32 compute, the same trade made
    everywhere else here. So an FP8 checkpoint loads and runs correctly, at the
    footprint of the fp16 model rather than of the fp8 one. Spending the byte
    saving needs a fused decode-GEMM, and the probe exists to show that is worth
    building. It is not built yet.

    Activations stay fp16. Quantizing them to fp8 only pays against an fp8 MMA
    instruction, and there is none here to aim at -- so this is W8A16 even when
    the checkpoint was serialized W8A8. vLLM already routes it that way below
    capability 8.9 (see CompressedTensorsW8A8Fp8's fallback), which is why the
    activation scale is dropped rather than honoured.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        if compute_capability is None:
            compute_capability = current_platform.get_device_capability().to_int()
        # A floor of last resort. Every candidate ahead of this one needs 7.5 or
        # more, so claiming anything at or above that would take work off a
        # tensor-core kernel that does it faster.
        if compute_capability >= 75:
            return (
                False,
                "capability 7.5 and up is served by FP8 Marlin or a tensor-core "
                "kernel; this one is for cards below that.",
            )
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key == kFp8Static128BlockSym:
            return (
                False,
                "block-wise fp8 scales are not implemented here; per-tensor and "
                "per-channel are.",
            )
        if c.input_dtype not in (torch.float16, torch.float32):
            return False, f"needs fp16 or fp32 activations, got {c.input_dtype}."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # torch's own e4m3 cast is exact on sm_61 -- every one of the 254 finite
        # patterns, subnormals included, checked against the format definition
        # in pascal/probes/fp8_decode.cu. It needs no FP8 hardware because it is
        # a field move, not a conversion.
        weight = layer.weight.data.to(torch.float32)

        # Callers hand non-block fp8 weights over as (K, N), so a per-channel
        # scale indexes the second axis. A per-tensor scale has one element and
        # the same reshape broadcasts it unchanged.
        scale = layer.weight_scale.data.to(torch.float32).reshape(1, -1)
        weight = weight * scale

        # Transpose here, once, because F.linear wants (N, K); leaving it to
        # apply_weights would repeat it every step.
        replace_parameter(
            layer, "weight", weight.t().contiguous().to(self.config.input_dtype)
        )

        # The scale is spent. Leaving it behind invites a second application by
        # anything that goes looking for it.
        replace_parameter(
            layer,
            "weight_scale",
            torch.ones(1, dtype=torch.float32, device=weight.device),
        )
        layer.input_scale = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight.to(x.dtype), bias)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # Unused: apply_weights is overridden, because the activation is never
        # quantized on this path.
        raise NotImplementedError
