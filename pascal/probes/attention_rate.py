"""How much of prefill attention is Triton's sub-sm_70 `tl.dot`, and what else could run it.

`profile_decode.py --prompt-tokens 1024 --max-tokens 1` puts
`kernel_unified_attention` at **164.9 ms of a 672.6 ms prefill**, 24.5% of the
whole thing, across 28 launches. The arithmetic it has to do is fixed and small:

    per layer, causal, n=1024, 16 q heads, head_dim 128
      QK^T   n*n*d*h     = 2.15 GFLOP   (halved for the causal triangle)
      P V    n*n*d*h     = 2.15 GFLOP
    28 layers                = 120 GFLOP

120 GFLOP in 164.9 ms is **0.73 TFLOP/s**, 9% of this card's 8.19 TFLOP/s. That
is the same number `skinny_gemm.py` measured for Triton's FMA lowering of
`tl.dot`, which is what Triton selects below sm_70 -- so prefill attention is
paying the same tax the GEMM used to.

The question this probe answers is whether the ordinary batched GEMM path is
better, since cuBLAS reaches 5-7 TFLOP/s at these shapes
(`fp16_linear_rate.py`). Materialising the [h, n, n] score matrix costs memory
that flash-style attention exists to avoid -- 16*1024*1024*2 = 33 MiB per layer
here -- but at this context length that is affordable, and 33 MiB at 223 GB/s is
0.15 ms against the 5.9 ms per layer being paid now.

    python pascal/probes/attention_rate.py
"""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

# Qwen3-VL-Embedding-2B's text attention: 16 query heads, 8 KV heads, head_dim
# 128. GQA group size 2.
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 128
N_LAYERS = 28

# Measured by profile_decode.py on this card, microseconds per layer.
TRITON_PREFILL_US = {1024: 164852 / 28}


def timeit(fn, warmup: int = 3, iters: int = 10) -> float:
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


def explicit_attention(q, k, v, scale):
    """(q @ k^T) -> softmax -> (@ v), i.e. two cuBLAS batched GEMMs.

    q is [h, n, d]; k and v are [h, n, d] with the KV heads already repeated.
    The causal mask is added rather than applied by slicing, so the GEMMs stay
    dense -- which wastes half the arithmetic and is still the point of the
    comparison, since it is the *rate* that differs by an order of magnitude.
    """
    scores = torch.bmm(q, k.transpose(1, 2)) * scale
    n = q.shape[1]
    mask = torch.full((n, n), float("-inf"), device=q.device, dtype=scores.dtype)
    scores += torch.triu(mask, diagonal=1)
    return torch.bmm(F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype), v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/attention_rate.json")
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048])
    args = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    peak = 8.19  # TFLOP/s, fp32 FMA, GP104 at boost
    print(f"{props.name}  fp32 FMA peak ~{peak} TFLOP/s\n")

    results = {}
    print(f"{'n':>6} {'impl':<22} {'ms/layer':>9} {'TFLOP/s':>8} {'%peak':>7} "
          f"{'28 layers':>10} {'rel_err':>9}")
    print("-" * 78)

    for n in args.lengths:
        scale = 1.0 / math.sqrt(HEAD_DIM)
        q = torch.randn(N_HEADS, n, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.1
        kv_shape = (N_KV_HEADS, n, HEAD_DIM)
        k_kv = torch.randn(kv_shape, device="cuda", dtype=torch.float16) * 0.1
        v_kv = torch.randn(kv_shape, device="cuda", dtype=torch.float16) * 0.1
        rep = N_HEADS // N_KV_HEADS
        k = k_kv.repeat_interleave(rep, dim=0).contiguous()
        v = v_kv.repeat_interleave(rep, dim=0).contiguous()

        # Causal attention does half the work of the dense form; count the half.
        flops = 2 * (2 * n * n * HEAD_DIM * N_HEADS) / 2

        ref = explicit_attention(q.float(), k.float(), v.float(), scale)

        rows = {}

        def record(label, fn):
            out = fn()
            rel = ((out.float() - ref).abs().max() / ref.abs().max()).item()
            t = timeit(fn)
            tf = flops / t / 1e12
            rows[label] = {
                "ms_per_layer": round(t * 1e3, 3),
                "TFLOP_s": round(tf, 3),
                "pct_peak": round(100 * tf / peak, 1),
                "ms_28_layers": round(t * 1e3 * N_LAYERS, 1),
                "rel_err": rel,
            }
            print(f"{n:>6} {label:<22} {t * 1e3:>9.3f} {tf:>8.3f} {100 * tf / peak:>6.1f}% "
                  f"{t * 1e3 * N_LAYERS:>10.1f} {rel:>9.2e}")

        record("bmm + softmax", lambda: explicit_attention(q, k, v, scale))
        record("sdpa", lambda: F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True,
            scale=scale).squeeze(0))

        if n in TRITON_PREFILL_US:
            t = TRITON_PREFILL_US[n] / 1e6
            tf = flops / t / 1e12
            rows["triton (measured in vLLM)"] = {
                "ms_per_layer": round(t * 1e3, 3),
                "TFLOP_s": round(tf, 3),
                "pct_peak": round(100 * tf / peak, 1),
                "ms_28_layers": round(t * 1e3 * N_LAYERS, 1),
            }
            print(f"{n:>6} {'triton (in vLLM)':<22} {t * 1e3:>9.3f} {tf:>8.3f} "
                  f"{100 * tf / peak:>6.1f}% {t * 1e3 * N_LAYERS:>10.1f}")

        results[n] = rows
        print()
        del q, k, v, k_kv, v_kv, ref
        torch.cuda.empty_cache()

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
