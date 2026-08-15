"""Capability probe for SM 6.1 (Pascal / GP104).

Answers, on real hardware, the questions that decide how much of vLLM has to be
rewritten rather than merely rebuilt:

  1. Do stock PyTorch cu126 wheels carry code this GPU can run at all?
     (arch list should contain sm_60; CUDA runs an sm_X.y cubin on sm_X.z, z>=y)
  2. Does Triton emit working sm_61 code for plain elementwise kernels?
  3. Does `tl.dot` work?  Triton keeps a non-tensor-core FMA lowering
     (lib/Conversion/TritonGPUToLLVM/DotOpToLLVM/FMA.cpp), but whether the
     NVIDIA backend selects it below sm_70 is the crux: 18 of Qwen3.5-2B's 24
     layers are Gated DeltaNet, implemented only as `tl.dot`-heavy Triton.
  4. How badly does fp16 arithmetic actually hurt?  GP104 runs HFMA2 at 1/64
     rate, so if torch dispatches fp16 matmul to fp32 SMs we are fine, and if it
     dispatches to native fp16 we must force fp32 compute everywhere.

Every check is isolated: one failure must not mask the others, because the point
is to produce a complete picture in a single run.
"""

import os
import sys
import traceback

# Keep failures legible rather than aborting the process on the first CUDA error.
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

RESULTS: list[tuple[str, str, str]] = []


class Skip(Exception):
    """Raised by a probe whose precondition is absent, e.g. vLLM not installed.

    Distinct from failure: a skipped probe says nothing about the hardware, and
    must not be counted against it.
    """


def check(name):
    """Run a probe, recording PASS/FAIL/SKIP plus detail, never raising."""

    def wrap(fn):
        try:
            detail = fn()
            RESULTS.append((name, "PASS", str(detail)))
        except Skip as exc:
            RESULTS.append((name, "SKIP", str(exc)))
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not judge
            RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
            print(f"\n--- traceback for {name} ---", flush=True)
            traceback.print_exc()
        return fn

    return wrap


print("=" * 72, flush=True)
print("Pascal SM 6.1 capability probe", flush=True)
print("=" * 72, flush=True)

import torch  # noqa: E402

print(f"torch            {torch.__version__}", flush=True)
print(f"torch.version.cuda {torch.version.cuda}", flush=True)
print(f"cuda available   {torch.cuda.is_available()}", flush=True)


@check("torch.arch_list")
def _arch_list():
    archs = torch.cuda.get_arch_list()
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    major, minor = cap
    # CUDA binary compatibility: an sm_X.y cubin runs on sm_X.z when z >= y.
    usable = [
        a
        for a in archs
        if a.startswith("sm_")
        and a[3:].isdigit()
        and int(a[3]) == major
        and int(a[4:]) <= minor
    ]
    print(f"\ndevice           {name} (sm_{major}{minor})", flush=True)
    print(f"arch_list        {archs}", flush=True)
    print(f"usable for this  {usable}", flush=True)
    if not usable:
        raise RuntimeError(
            f"no arch in {archs} is binary-compatible with sm_{major}{minor}; "
            "this wheel cannot run here and torch must be built from source"
        )
    return f"{name} sm_{major}{minor}; usable={usable}"


@check("torch.basic_op")
def _basic():
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = (a @ b).float()
    ref = (a.cpu() @ b.cpu()).float()
    err = (c.cpu() - ref).abs().max().item()
    if not torch.isfinite(c).all():
        raise RuntimeError("fp32 matmul produced non-finite values")
    return f"fp32 matmul max_abs_err={err:.3e}"


@check("torch.fp16_matmul")
def _fp16():
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    c = a @ b
    ref = (a.float().cpu() @ b.float().cpu())
    err = (c.float().cpu() - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    if not torch.isfinite(c).all():
        raise RuntimeError("fp16 matmul produced non-finite values")
    return f"fp16 matmul max_rel_err={rel:.3e}"


@check("torch.bf16_rejected")
def _bf16():
    # Pascal has no bf16. We assert the failure mode is clean so vLLM's dtype
    # fallback has something predictable to key on.
    try:
        a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        c = a @ a
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(c).all())
        return f"bf16 matmul unexpectedly ran (finite={finite})"
    except Exception as exc:  # noqa: BLE001
        return f"bf16 correctly unsupported: {type(exc).__name__}"


@check("torch.fp16_vs_fp32_speed")
def _speed():
    import time

    def bench(dtype, n=2048, iters=20):
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        return 2 * n**3 / dt / 1e12  # TFLOP/s

    fp32 = bench(torch.float32)
    fp16 = bench(torch.float16)
    # If fp16 lands anywhere near 1/64 of fp32, torch is using native HFMA2 and
    # every fp16 GEMM path in vLLM has to be forced to fp32 compute.
    verdict = "fp16 OK (fp32 compute)" if fp16 > fp32 * 0.5 else "fp16 CRIPPLED"
    return f"fp32={fp32:.2f} TF/s fp16={fp16:.2f} TF/s ratio={fp16 / fp32:.2f} -> {verdict}"


# --------------------------------------------------------------------------
# Triton
# --------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    print(f"triton           {triton.__version__}", flush=True)
    HAVE_TRITON = True
except Exception as exc:  # noqa: BLE001
    print(f"triton           UNAVAILABLE: {exc}", flush=True)
    HAVE_TRITON = False

if HAVE_TRITON:

    @triton.jit
    def _add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

    @check("triton.elementwise")
    def _tl_add():
        n = 8192
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        o = torch.empty_like(x)
        _add_kernel[(triton.cdiv(n, 256),)](x, y, o, n, BLOCK=256)
        torch.cuda.synchronize()
        err = (o - (x + y)).abs().max().item()
        if err > 1e-5:
            raise RuntimeError(f"elementwise mismatch {err}")
        return f"max_abs_err={err:.3e}"

    @triton.jit
    def _matmul_kernel(
        a_ptr, b_ptr, c_ptr, M, N, K,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(
                a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :],
                mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :],
                mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)
        tl.store(
            c_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    def _run_dot(dtype, M=128, N=128, K=128):
        a = torch.randn(M, K, device="cuda", dtype=dtype)
        b = torch.randn(K, N, device="cuda", dtype=dtype)
        c = torch.empty(M, N, device="cuda", dtype=torch.float32)
        _matmul_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
            a, b, c, M, N, K, BM=64, BN=64, BK=32
        )
        torch.cuda.synchronize()
        ref = a.float().cpu() @ b.float().cpu()
        rel = (c.cpu() - ref).abs().max().item() / ref.abs().max().item()
        if rel > 1e-2:
            raise RuntimeError(f"tl.dot result mismatch rel={rel}")
        return f"rel_err={rel:.3e}"

    @check("triton.tl_dot.fp32")
    def _dot_fp32():
        return _run_dot(torch.float32)

    @check("triton.tl_dot.fp16")
    def _dot_fp16():
        # This is the one that matters most: FLA's Gated DeltaNet kernels are
        # fp16/bf16 tl.dot throughout.
        return _run_dot(torch.float16)


# --------------------------------------------------------------------------
# The real kernels: vLLM's vendored FLA + causal_conv1d
# --------------------------------------------------------------------------


@check("vllm.causal_conv1d")
def _conv():
    try:
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    except ImportError as exc:
        raise Skip(f"vLLM not importable yet ({exc})") from exc

    batch, dim, seqlen, width = 1, 64, 32, 4
    x = torch.randn(batch, dim, seqlen, device="cuda", dtype=torch.float16)
    w = torch.randn(dim, width, device="cuda", dtype=torch.float16)
    out = causal_conv1d_fn(x, w, None)
    torch.cuda.synchronize()
    if not torch.isfinite(out).all():
        raise RuntimeError("causal_conv1d produced non-finite values")
    return f"ok shape={tuple(out.shape)}"


@check("vllm.fla.chunk_gated_delta_rule")
def _gdn():
    try:
        from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
    except ImportError as exc:
        raise Skip(f"vLLM not importable yet ({exc})") from exc

    B, T, H, K, V = 1, 64, 4, 64, 64
    dt = torch.float16
    q = torch.randn(B, T, H, K, device="cuda", dtype=dt)
    k = torch.randn(B, T, H, K, device="cuda", dtype=dt)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dt)
    g = torch.rand(B, T, H, device="cuda", dtype=torch.float32).log()
    beta = torch.rand(B, T, H, device="cuda", dtype=dt)
    out = chunk_gated_delta_rule(q, k, v, g, beta)
    if isinstance(out, tuple):
        out = out[0]
    torch.cuda.synchronize()
    if not torch.isfinite(out).all():
        raise RuntimeError("chunk_gated_delta_rule produced non-finite values")
    return f"ok shape={tuple(out.shape)}"


print("\n" + "=" * 72, flush=True)
print("RESULTS", flush=True)
print("=" * 72, flush=True)
width = max(len(n) for n, _, _ in RESULTS)
for name, status, detail in RESULTS:
    print(f"{name:<{width}}  {status:<4}  {detail}", flush=True)

failed = [n for n, s, _ in RESULTS if s == "FAIL"]
skipped = [n for n, s, _ in RESULTS if s == "SKIP"]
ran = len(RESULTS) - len(skipped)
print(f"\n{ran - len(failed)}/{ran} passed ({len(skipped)} skipped)", flush=True)
if failed:
    print(f"failed: {', '.join(failed)}", flush=True)
# The probe's job is to report, not to gate a pipeline; always exit 0 so the
# full result table survives to the pod logs.
sys.exit(0)
