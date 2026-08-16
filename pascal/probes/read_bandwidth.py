"""What a pure streaming read actually costs on this card.

Every decode roofline in this project divides the weight bytes by a bandwidth
figure, so that figure had better be the right one. Two earlier candidates were
not:

  * `torch.sum` on fp16 reports 67 GB/s. That measures torch's reduction kernel,
    not the memory system.
  * a device-to-device copy reports 212 GB/s, but a copy is a read *and* a write.
    A GEMM only reads its weight, and a read-only stream can beat a copy.

The skinny GEMM already measured 223 GB/s of weight traffic in a real decode
step, i.e. above the copy figure, which is the tell that the copy number is the
wrong ceiling. This probe pins the read ceiling directly: 128-bit loads, one
accumulate per element, enough blocks to fill the card, and no writes worth
counting.

    python pascal/probes/read_bandwidth.py
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import torch
from torch.utils.cpp_extension import load_inline

SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

// One float4 (eight halves) per lane per step, grid-strided. The accumulate is
// there only so the loads cannot be optimised away; it is one FMA per eight
// values, far below anything that could bind before memory does.
__global__ void read_stream_kernel(const __half* __restrict__ src,
                                   float* __restrict__ sink, size_t n4) {
  float acc = 0.0f;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const float4 raw = reinterpret_cast<const float4*>(src)[i];
    const __half* h = reinterpret_cast<const __half*>(&raw);
    acc += __half2float(h[0]) + __half2float(h[7]);
  }
  if (acc == 1234.5678f) sink[0] = acc;  // never taken; defeats dead-code removal
}

void read_stream(at::Tensor src, at::Tensor sink, int64_t blocks, int64_t threads) {
  const size_t n4 = src.numel() / 8;
  read_stream_kernel<<<dim3((unsigned)blocks), dim3((unsigned)threads), 0,
                       at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(src.data_ptr<at::Half>()),
      sink.data_ptr<float>(), n4);
}
"""


def timeit(fn, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/read_bandwidth.json")
    ap.add_argument("--mib", type=int, default=1024)
    args = ap.parse_args()

    ext = load_inline(
        name="read_bandwidth_sm61",
        cpp_sources="void read_stream(at::Tensor, at::Tensor, int64_t, int64_t);",
        cuda_sources=SOURCE,
        functions=["read_stream"],
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_61,code=sm_61"],
        build_directory=tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp")),
        verbose=False,
    )

    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    peak = props.memory_clock_rate * 1e3 * (props.memory_bus_width / 8) * 2 / 1e9
    print(f"{props.name}  {sms} SMs  spec peak {peak:.1f} GB/s")

    n = args.mib * 1024 * 1024 // 2
    src = torch.randn(n, device="cuda", dtype=torch.float16)
    sink = torch.zeros(1, device="cuda", dtype=torch.float32)
    nbytes = n * 2

    dst = torch.empty_like(src)
    t = timeit(lambda: dst.copy_(src))
    copy_gbs = 2 * nbytes / t / 1e9
    del dst
    torch.cuda.empty_cache()
    print(f"copy (read+write): {copy_gbs:>7.1f} GB/s")

    best = 0.0
    best_cfg = None
    print(f"\n{'blocks':>8} {'threads':>8} {'GB/s':>8} {'% spec':>8}")
    for threads in (128, 256, 512):
        for per_sm in (1, 2, 4, 8, 16):
            blocks = sms * per_sm
            t = timeit(lambda b=blocks, th=threads: ext.read_stream(src, sink, b, th))
            gbs = nbytes / t / 1e9
            if gbs > best:
                best, best_cfg = gbs, (blocks, threads)
            print(f"{blocks:>8} {threads:>8} {gbs:>8.1f} {100 * gbs / peak:>7.1f}%")

    print(f"\nread ceiling: {best:.1f} GB/s at blocks={best_cfg[0]} "
          f"threads={best_cfg[1]}  ({100 * best / peak:.1f}% of spec)")

    decode_bytes = (28 * (2048 * 4096 + 2048 * 2048 + 2048 * 12288 + 6144 * 2048)
                    + 2048 * 151936) * 2
    print(f"\ndecode-resident weights: {decode_bytes / 1e9:.3f} GB")
    print(f"batch-1 GEMM floor at the read ceiling: "
          f"{decode_bytes / best / 1e9 * 1e3:.2f} ms")
    print(f"batch-1 decode ceiling with a 2.7 ms tail: "
          f"{1.0 / (decode_bytes / best / 1e9 + 0.0027):.1f} tok/s")

    with open(args.out, "w") as fh:
        json.dump({
            "spec_peak_GB_s": round(peak, 1),
            "copy_GB_s": round(copy_gbs, 1),
            "read_GB_s": round(best, 1),
            "best_blocks": best_cfg[0],
            "best_threads": best_cfg[1],
            "decode_bytes": decode_bytes,
        }, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
