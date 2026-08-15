"""Measure what an idle-clock ramp costs a short request.

This probe exists to kill a confounder before anyone writes a kernel. The
production engine's inter-token latency is 62.5 ms, roughly 15x the bandwidth
roofline, and that is exactly the kind of number that starts an expensive
investigation. But the cards it was measured on sit at pstate P8 and 210 MHz
between requests, against a 2100 MHz ceiling. A 10x clock deficit and a 10x
kernel problem are indistinguishable from the outside.

So: run the same kernel twice. Once on a card that has been idle long enough to
drop to P8, once on a card already at full clocks. The ratio is the tax an
interactive workload pays for arriving at a sleeping GPU, and it has nothing to
do with vLLM.

    python ampere/probes/clock_ramp.py
    python ampere/probes/clock_ramp.py --idle-seconds 30 --device 0

If the cold/hot ratio is large, re-measure every other number on this branch
with clocks locked (`nvidia-smi -lgc <min>,<max>`) before believing it.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch


def nvml_handle(device: int):
    """Return an NVML handle for the *physical* card behind cuda:<device>.

    CUDA_VISIBLE_DEVICES makes the two numbering schemes disagree, and reading
    clocks off the wrong card is a silent way to draw the opposite conclusion.
    Matching on UUID is the only mapping that survives it.
    """
    try:
        import pynvml
    except ImportError:
        return None, None

    pynvml.nvmlInit()
    uuid = torch.cuda.get_device_properties(device).uuid
    want = str(uuid).replace("GPU-", "").lower()
    for idx in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        got = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(got, bytes):
            got = got.decode()
        if got.replace("GPU-", "").lower() == want:
            return pynvml, handle
    return pynvml, None


def read_state(pynvml, handle) -> tuple[int, int, float]:
    """(sm_clock_mhz, pstate, power_w); zeros when NVML is unavailable."""
    if pynvml is None or handle is None:
        return 0, 0, 0.0
    sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
    pstate = pynvml.nvmlDeviceGetPerformanceState(handle)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    return sm, pstate, power


def timed_burst(a: torch.Tensor, b: torch.Tensor, iters: int) -> list[float]:
    """Per-iteration wall time in ms, synchronising each one.

    Deliberately not the usual "sync once around N iterations": the whole point
    is the shape of the first few milliseconds, which an aggregate hides.
    """
    out = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.mm(a, b)
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--idle-seconds",
        type=float,
        default=20.0,
        help="how long to leave the card alone before the cold run; the driver "
        "needs on the order of ten seconds to settle into P8",
    )
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument(
        "--size",
        type=int,
        default=8192,
        help="square matmul dimension; large enough to be clock-bound rather "
        "than launch-bound",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1

    torch.cuda.set_device(args.device)
    pynvml, handle = nvml_handle(args.device)
    if handle is None:
        print(
            "warning: NVML unavailable or UUID unmatched; timings still valid, "
            "clock columns will read zero\n",
            file=sys.stderr,
        )

    name = torch.cuda.get_device_name(args.device)
    a = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16)

    # Warm up allocator, cuBLAS handle and kernel selection, so the cold run
    # below measures clocks and not one-time setup.
    timed_burst(a, b, 5)

    print(f"device: {name} (cuda:{args.device})")
    print(f"matmul: {args.size}^3 bf16\n")

    print(f"idling {args.idle_seconds:.0f}s to let the card drop to P8 ...")
    torch.cuda.synchronize()
    time.sleep(args.idle_seconds)
    sm_idle, pstate_idle, power_idle = read_state(pynvml, handle)
    print(f"  idle state: P{pstate_idle}, {sm_idle} MHz, {power_idle:.0f} W\n")

    cold = timed_burst(a, b, args.iters)
    sm_hot, pstate_hot, power_hot = read_state(pynvml, handle)
    print(f"  after burst: P{pstate_hot}, {sm_hot} MHz, {power_hot:.0f} W\n")

    # Second run with the card already awake, for the ratio.
    hot = timed_burst(a, b, args.iters)

    steady = statistics.median(hot)
    print(f"{'iter':>6} {'cold ms':>10} {'hot ms':>10} {'cold/hot':>10}")
    print("-" * 40)
    for i in range(min(12, args.iters)):
        print(f"{i:6d} {cold[i]:10.2f} {hot[i]:10.2f} {cold[i] / hot[i]:10.2f}")

    first = cold[0]
    # How long until a cold card is within 10% of its own steady state -- the
    # number that decides whether a short request ever gets full clocks.
    ramp_ms, elapsed = None, 0.0
    for duration in cold:
        elapsed += duration
        if duration <= steady * 1.10:
            ramp_ms = elapsed
            break

    print("\n" + "-" * 40)
    print(f"first iteration cold : {first:8.2f} ms")
    print(f"steady state (hot)   : {steady:8.2f} ms")
    print(f"first-iteration tax  : {first / steady:8.2f}x")
    if ramp_ms is None:
        print(f"ramp to within 10%   :   never, in {elapsed:.0f} ms of continuous work")
    else:
        print(f"ramp to within 10%   : {ramp_ms:8.0f} ms of continuous work")

    print(
        "\nA request shorter than the ramp never sees full clocks. Compare that "
        "ramp against\nthe production TTFT (~0.9 s) and decode duration before "
        "attributing anything to vLLM."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
