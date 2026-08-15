"""Can sm_86 dequantize an fp8 KV cache in software, exactly and cheaply?

This is the gate on the only route to full CUDA graphs on the 3090 Ti pair.

The chain that forces `PIECEWISE` today: `kv_cache_dtype=fp8` leaves only
`FLASHINFER` and `TRITON_ATTN` as candidates on sm_86; `triton_attn.py:521`
refuses fp8 below SM89 because the kernel's dequantisation is
``data.to(tl.float32)`` on an fp8 value, which compiles to a `cvt` Ampere does
not have; and FlashInfer declares only `UNIFORM_SINGLE_TOKEN_DECODE`, which is
not enough for spec decode. FlashInfer cannot be fixed here -- its
`UNIFORM_BATCH` path needs trtllm-gen, and `supports_trtllm_attention()` is
explicit that only SM90 and SM100+ have it. Triton attention, by contrast,
already declares `AttentionCGSupport.ALWAYS`.

So the whole question is whether the *conversion* can be done without the
instruction. e4m3 is eight bits with a known layout, and reconstructing it with
integer arithmetic is ordinary bit manipulation that Ampere can do. Whether
Triton will compile that on sm_86 -- in particular whether it will load fp8
memory at all and let it be bitcast to `uint8` -- is not obvious, and is what
this measures.

Cost is not expected to matter: decode on this pair spends ~7 ms of a 15 ms
step computing, and Gemma's profile put arithmetic at ~1% of tensor-core
throughput. Spending a handful of integer ops per KV element to buy a capability
is the same trade as everywhere else on these boxes. It is still measured rather
than assumed.

Exactness is the part that must not be hand-waved: this is a numerics change on
the attention path, so the probe checks every one of the 256 possible byte
values against PyTorch's own conversion rather than sampling.

    python ampere/probes/fp8_dequant_sm86.py
"""

from __future__ import annotations

import sys


def build_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _e4m3_to_f32(b):
        """Reconstruct float32 from raw e4m3 (fn) bits, without a hardware cvt.

        e4m3fn: 1 sign, 4 exponent (bias 7), 3 mantissa, no infinities.
        Normals assemble directly into float32 bits, since the exponent
        rebiases as ``e - 7 + 127 == e + 120`` and the mantissa left-shifts
        from 3 bits to 23. Subnormals (``e == 0``) carry no implicit leading
        one and are worth ``m * 2**-9``, which is cheaper to build in floating
        point than to renormalise by hand.
        """
        u = b.to(tl.uint32)
        sign = (u >> 7) & 1
        exp = (u >> 3) & 0xF
        man = u & 0x7

        normal_bits = ((exp + 120) << 23) | (man << 20)
        normal = normal_bits.to(tl.float32, bitcast=True)
        sub = man.to(tl.float32) * 0.001953125  # 2**-9

        val = tl.where(exp == 0, sub, normal)
        return tl.where(sign == 1, -val, val)

    @triton.jit
    def sw_dequant_kernel(src, dst, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        raw = tl.load(src + offs, mask=mask, other=0)
        tl.store(dst + offs, _e4m3_to_f32(raw), mask=mask)

    @triton.jit
    def hw_dequant_kernel(src, dst, n, BLOCK: tl.constexpr):
        """The path vLLM uses today; needs a cvt that sm_86 lacks."""
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        raw = tl.load(src + offs, mask=mask, other=0.0)
        tl.store(dst + offs, raw.to(tl.float32), mask=mask)

    return sw_dequant_kernel, hw_dequant_kernel


def main() -> int:
    import torch

    sw_kernel, hw_kernel = build_kernels()

    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability(0)
    print(f"{torch.cuda.get_device_name(0)}, sm_{cap[0]}{cap[1]}")

    # Every representable byte, so the check is exhaustive rather than sampled.
    all_bytes = torch.arange(256, dtype=torch.uint8, device=dev)
    as_fp8 = all_bytes.view(torch.float8_e4m3fn)
    reference = as_fp8.to(torch.float32)
    finite = torch.isfinite(reference)

    out = torch.empty(256, dtype=torch.float32, device=dev)
    n = 256
    grid = (1,)
    try:
        sw_kernel[grid](all_bytes, out, n, BLOCK=256)
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"SOFTWARE PATH FAILED TO COMPILE: {type(exc).__name__}: {exc}")
        return 1

    mism = (out[finite] != reference[finite]).sum().item()
    print(
        f"software dequant: {256 - int(finite.sum())} non-finite skipped, "
        f"{mism} mismatches out of {int(finite.sum())} finite values"
    )
    if mism:
        bad = (out != reference) & finite
        idx = bad.nonzero()[:5].flatten().tolist()
        for i in idx:
            got, want = out[i].item(), reference[i].item()
            print(f"    byte 0x{i:02x}: got {got!r} want {want!r}")
        return 1
    print("software dequant is bit-exact on every finite e4m3 value")

    # Does the path vLLM uses today actually fail here, or is the gate
    # conservative? Worth knowing before proposing to relax it.
    hw_out = torch.empty(256, dtype=torch.float32, device=dev)
    try:
        hw_kernel[grid](as_fp8, hw_out, n, BLOCK=256)
        torch.cuda.synchronize()
        hw_mism = (hw_out[finite] != reference[finite]).sum().item()
        print(
            f"hardware cvt path: COMPILED AND RAN on sm_{cap[0]}{cap[1]}, "
            f"{hw_mism} mismatches -- the SM89 gate may be conservative"
        )
    except Exception as exc:
        print(
            f"hardware cvt path: fails as expected -- "
            f"{type(exc).__name__}: {str(exc)[:120]}"
        )

    # Throughput, against a plain copy of the same bytes.
    big = torch.randint(0, 255, (64 * 1024 * 1024,), dtype=torch.uint8, device=dev)
    big_out = torch.empty(big.numel(), dtype=torch.float32, device=dev)
    nb = big.numel()
    g = ((nb + 1023) // 1024,)

    def timed(fn, iters=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    ms = timed(lambda: sw_kernel[g](big, big_out, nb, BLOCK=1024))
    # read 1 byte + write 4 per element
    gbs = nb * 5 / (ms / 1000) / 1e9
    print(
        f"\nsoftware dequant over {nb / 1e6:.0f}M values: {ms:.3f} ms, "
        f"{gbs:.0f} GB/s of traffic -- bandwidth-bound means the integer ops "
        f"are free"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
