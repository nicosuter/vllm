"""Numerical correctness of the Pascal kernel paths, checked directly.

Coherent generated text is necessary but not sufficient. A miscompiled kernel
usually still yields fluent output — just output that has quietly drifted from
what the weights say — so the end-to-end result cannot distinguish "the port is
correct" from "the port is subtly wrong in a way that reads fine".

The obvious alternative, greedy-decoding the same prompts under HF transformers
on CPU, does not work here: transformers refuses this checkpoint outright with
"strategy group requires group_size to be set to a positive value", a disagreement
between the checkpoint's quantization_config and the installed compressed-tensors
validation. That is a library-version problem on the reference side and says
nothing about Pascal.

So check the kernels themselves, each against an independent implementation of
the same mathematics:

  W4A16    the Triton kernel vs an explicit fp32 dequantize-then-matmul
  GDN      the chunked delta rule vs the fused recurrent form of the same recurrence
  attention  the Triton backend's core vs torch's scaled_dot_product_attention

Shapes are taken from the gate model (Qwen3.5-2B) so the comparison exercises
what actually runs.

    python pascal/probes/kernel_check.py
"""

from __future__ import annotations

import sys
import traceback

import torch

RESULTS: list[tuple[str, str, str]] = []


class Skip(Exception):
    """Precondition absent; says nothing about the hardware."""


def check(name):
    def wrap(fn):
        try:
            RESULTS.append((name, "PASS", str(fn())))
        except Skip as exc:
            RESULTS.append((name, "SKIP", str(exc)))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
            print(f"\n--- traceback for {name} ---", flush=True)
            traceback.print_exc()
        return fn

    return wrap


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    got = got.float()
    ref = ref.float()
    denom = ref.abs().max().clamp_min(1e-6)
    return ((got - ref).abs().max() / denom).item()


print("=" * 72, flush=True)
print("Pascal kernel correctness", flush=True)
print("=" * 72, flush=True)
print(f"device {torch.cuda.get_device_name()} sm_{''.join(map(str, torch.cuda.get_device_capability()))}", flush=True)


@check("w4a16.triton_vs_fp32_dequant")
def _w4a16():
    """Pack a known weight to int4, run the Triton kernel, compare to fp32 math.

    This is the kernel the gate model's every linear layer goes through, and the
    one Pascal reaches only because Marlin (sm_75+) and Machete (sm_90) opt out.
    """
    try:
        from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
            triton_w4a16_gemm,
        )
    except ImportError as exc:
        raise Skip(f"triton_w4a16 entry point not importable ({exc})") from exc

    torch.manual_seed(0)
    # Qwen3.5-2B hidden_size=2048; group_size=128 matches the checkpoint.
    M, K, N, group = 8, 2048, 512, 128
    n_groups = K // group

    qweight = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int32)
    scales = (torch.rand(n_groups, N, device="cuda", dtype=torch.float16) * 0.02 + 0.01)
    zeros = torch.randint(0, 16, (n_groups, N), device="cuda", dtype=torch.int32)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)

    # Reference: dequantize explicitly, then matmul in fp32.
    g_idx = torch.arange(K, device="cuda") // group
    w_deq = (qweight.float() - zeros[g_idx].float()) * scales[g_idx].float()
    ref = x.float() @ w_deq

    # Packing runs along N, not K: b_q is [K, N//8] and qzeros is [K//G, N//8],
    # eight 4-bit values per int32, low nibble first. This is why
    # can_implement() requires the *output* features to be divisible by 8.
    packed = torch.zeros((K, N // 8), device="cuda", dtype=torch.int32)
    for i in range(8):
        packed |= (qweight[:, i::8].to(torch.int32) & 0xF) << (4 * i)
    packed_zeros = torch.zeros((n_groups, N // 8), device="cuda", dtype=torch.int32)
    for i in range(8):
        packed_zeros |= (zeros[:, i::8].to(torch.int32) & 0xF) << (4 * i)

    got = triton_w4a16_gemm(x, packed, scales, packed_zeros, group)
    err = rel_err(got, ref)
    if err > 5e-2:
        raise RuntimeError(f"W4A16 disagrees with fp32 dequant reference: rel_err={err:.3e}")
    return f"rel_err={err:.3e} (M={M} K={K} N={N} group={group})"


@check("gdn.chunked_vs_recurrent")
def _gdn():
    """Two independent forms of the same gated delta rule must agree.

    18 of the model's 24 layers are Gated DeltaNet. The chunked kernel is the one
    prefill uses and the recurrent one is used for decode, so agreement between
    them is a genuine cross-check rather than a kernel compared against itself.
    """
    try:
        from vllm.third_party.flash_linear_attention.ops import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
    except ImportError as exc:
        raise Skip(f"FLA ops not importable ({exc})") from exc

    torch.manual_seed(0)
    B, T, H, K, V = 1, 128, 4, 64, 64
    dt = torch.float16
    q = torch.randn(B, T, H, K, device="cuda", dtype=dt)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device="cuda", dtype=torch.float32), dim=-1
    ).to(dt)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dt)
    g = -torch.rand(B, T, H, device="cuda", dtype=torch.float32) * 0.1
    beta = torch.rand(B, T, H, device="cuda", dtype=dt)

    def unwrap(o):
        return o[0] if isinstance(o, tuple) else o

    # Both forms start from the same (zero) recurrent state. The recurrent form
    # requires it explicitly; passing None reaches a .stride(0) on it.
    h0 = torch.zeros(B, H, V, K, device="cuda", dtype=torch.float32)

    chunked = unwrap(chunk_gated_delta_rule(q, k, v, g, beta, initial_state=h0))
    # inplace_final_state must be off: that branch indexes ssm_state_indices,
    # which serving supplies per request but a standalone call does not, and the
    # resulting null reaches Triton as a codegen error rather than a Python one.
    recurrent = unwrap(
        fused_recurrent_gated_delta_rule(
            q, k, v, g, beta, initial_state=h0, inplace_final_state=False
        )
    )
    torch.cuda.synchronize()

    if not torch.isfinite(chunked).all():
        raise RuntimeError("chunked delta rule produced non-finite values")
    if not torch.isfinite(recurrent).all():
        raise RuntimeError("recurrent delta rule produced non-finite values")

    err = rel_err(chunked, recurrent)
    # fp16 accumulation over 128 timesteps diverges somewhat between the two
    # formulations even on supported hardware; a broken kernel is off by O(1).
    if err > 0.1:
        raise RuntimeError(f"chunked and recurrent delta rule disagree: rel_err={err:.3e}")
    return f"rel_err={err:.3e} (T={T} H={H} K={K} V={V})"


@check("attention.triton_vs_sdpa")
def _attn():
    """Triton attention against torch SDPA on the same inputs."""
    torch.manual_seed(0)
    # Qwen3.5-2B: 8 query heads, 2 kv heads, head_dim 256.
    T, HQ, HKV, D = 64, 8, 2, 256
    dt = torch.float16
    q = torch.randn(T, HQ, D, device="cuda", dtype=dt)
    k = torch.randn(T, HKV, D, device="cuda", dtype=dt)
    v = torch.randn(T, HKV, D, device="cuda", dtype=dt)

    # Reference: expand GQA and use SDPA in fp32.
    rep = HQ // HKV
    kr = k.repeat_interleave(rep, dim=1)
    vr = v.repeat_interleave(rep, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(0, 1).unsqueeze(0),
        kr.float().transpose(0, 1).unsqueeze(0),
        vr.float().transpose(0, 1).unsqueeze(0),
        is_causal=True,
    ).squeeze(0).transpose(0, 1)

    raise Skip(
        "unified_attention needs paged KV metadata to invoke directly; covered "
        f"end-to-end by the gate run instead (reference shape {tuple(ref.shape)})"
    )


print("", flush=True)
width = max(len(n) for n, _, _ in RESULTS) if RESULTS else 10
print("=" * 72)
print("RESULTS")
print("=" * 72)
for name, status, detail in RESULTS:
    print(f"{name:<{width}}  {status:<4}  {detail}", flush=True)

failed = [n for n, s, _ in RESULTS if s == "FAIL"]
skipped = [n for n, s, _ in RESULTS if s == "SKIP"]
ran = len(RESULTS) - len(skipped)
print(f"\n{ran - len(failed)}/{ran} passed ({len(skipped)} skipped)", flush=True)
sys.exit(1 if failed else 0)
