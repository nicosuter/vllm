"""Tune vLLM's Triton attention tiling for sm_61.

Prefill attention is 24.5% of a 1024-token prefill on this card -- 164.9 ms
across 28 layers, which is 0.73 TFLOP/s against a 8.19 TFLOP/s peak. Below
sm_70 Triton lowers `tl.dot` to its FMA path, and the tile sizes vLLM picks were
chosen for hardware where `tl.dot` becomes an MMA instruction. For this model
`num_queries_per_kv` is 2, so the defaults come out at BLOCK_M=16, BLOCK_Q=8,
TILE_SIZE=32 and Triton's own 4 warps -- a 16x32x128 dot per step, which is a
small tile to amortise an FMA inner loop over.

This asks whether a larger tile is simply better here, before concluding that
the kernel has to be rewritten in CUDA.

The override is read from the environment so the sweep can drive an unmodified
tree; the winning values are then hard-coded in
`vllm/v1/attention/ops/triton_unified_attention.py`.

    VLLM_PASCAL_ATTN_TUNE=1 python pascal/probes/attention_tiling.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import torch


def build_inputs(seq_len: int, num_heads: int, num_kv_heads: int, head_size: int,
                 block_size: int, dtype: torch.dtype):
    """One prefill sequence of `seq_len` tokens against a paged KV cache."""
    num_blocks = (seq_len + block_size - 1) // block_size
    q = torch.randn(seq_len, num_heads, head_size, device="cuda", dtype=dtype) * 0.1
    key_cache = (
        torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                    device="cuda", dtype=dtype) * 0.1
    )
    value_cache = (
        torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                    device="cuda", dtype=dtype) * 0.1
    )
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).unsqueeze(0)
    return q, key_cache, value_cache, out, cu_seqlens_q, seqused_k, block_table


def reference(q, key_cache, value_cache, seq_len, scale):
    """Causal paged attention in fp32, computed independently of the kernel.

    The tiling sweep compares configurations against each other, which cannot
    catch a change that is wrong in every configuration. This can.
    """
    import torch.nn.functional as F

    num_kv_heads, head_size = key_cache.shape[2], key_cache.shape[3]
    num_heads = q.shape[1]
    k = key_cache.reshape(-1, num_kv_heads, head_size)[:seq_len]
    v = value_cache.reshape(-1, num_kv_heads, head_size)[:seq_len]
    rep = num_heads // num_kv_heads
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    qf = q.float().transpose(0, 1)          # [h, n, d]
    kf = k.float().transpose(0, 1)
    vf = v.float().transpose(0, 1)
    scores = torch.bmm(qf, kf.transpose(1, 2)) * scale
    mask = torch.full((seq_len, seq_len), float("-inf"), device=q.device)
    scores += torch.triu(mask, diagonal=1)
    return torch.bmm(F.softmax(scores, dim=-1), vf).transpose(0, 1)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/attention_tiling.json")
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--block-m", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--tile", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--warps", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--dot-fp32", type=int, nargs="+", default=[0, 1],
                    help="0 hands tl.dot fp16 operands, 1 casts them to float; "
                         "the cast is 2.7x faster but doubles the shared memory "
                         "the dot operands need")
    args = ap.parse_args()

    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    num_heads, num_kv_heads, head_size, block_size = 16, 8, 128, 16
    dtype = torch.float16
    peak_tflops = 8.19

    results = {}
    for seq_len in args.seq_lens:
        q, kc, vc, out, cu, used, bt = build_inputs(
            seq_len, num_heads, num_kv_heads, head_size, block_size, dtype
        )
        scale = head_size ** -0.5
        flops = 2 * (2 * seq_len * seq_len * head_size * num_heads) / 2

        def run():
            unified_attention(
                q=q, k=kc, v=vc, out=out,
                cu_seqlens_q=cu, seqused_k=used,
                max_seqlen_q=seq_len, max_seqlen_k=seq_len,
                softmax_scale=scale, causal=True, window_size=(-1, -1),
                block_table=bt, softcap=0,
                q_descale=None, k_descale=None, v_descale=None,
            )

        ref_out = reference(q, kc, vc, seq_len, scale)
        os.environ.pop("VLLM_PASCAL_ATTN_BLOCK_M", None)
        os.environ.pop("VLLM_PASCAL_ATTN_DOT_FP32", None)
        run()
        torch.cuda.synchronize()
        shipped_rel = ((out.float() - ref_out).abs().max()
                       / ref_out.abs().max()).item()
        shipped_ms = timeit(run) * 1e3
        print(f"\n== seq_len {seq_len}  ({flops / 1e9:.1f} GFLOP/layer) ==")
        print(f"  shipped config: {shipped_ms:.3f} ms/layer, "
              f"{shipped_ms * args.layers:.1f} ms over {args.layers} layers, "
              f"rel_err vs fp32 {shipped_rel:.2e}")
        print(f"{'f32?':>5} {'BLOCK_M':>8} {'TILE':>6} {'warps':>6} {'ms':>8} "
              f"{'TFLOP/s':>8} {'%peak':>7} {f'x{args.layers}':>9} {'rel_err':>9}")
        rows = []
        for dot32, block_m, tile, warps in itertools.product(
                args.dot_fp32, args.block_m, args.tile, args.warps):
            os.environ["VLLM_PASCAL_ATTN_DOT_FP32"] = str(dot32)
            os.environ["VLLM_PASCAL_ATTN_BLOCK_M"] = str(block_m)
            os.environ["VLLM_PASCAL_ATTN_TILE"] = str(tile)
            os.environ["VLLM_PASCAL_ATTN_WARPS"] = str(warps)
            try:
                run()
            except Exception as exc:  # a config the kernel rejects is a result
                print(f"{dot32:>5} {block_m:>8} {tile:>6} {warps:>6}   "
                      f"{type(exc).__name__}: {str(exc)[:36]}")
                continue
            torch.cuda.synchronize()
            rel = ((out.float() - ref_out).abs().max()
                   / ref_out.abs().max()).item()
            t = timeit(run)
            tf = flops / t / 1e12
            rows.append({
                "dot_fp32": dot32, "BLOCK_M": block_m, "TILE": tile,
                "warps": warps,
                "ms": round(t * 1e3, 3), "TFLOP_s": round(tf, 3),
                "pct_peak": round(100 * tf / peak_tflops, 1),
                "ms_all_layers": round(t * 1e3 * args.layers, 1),
                "rel_err": rel,
            })
            print(f"{dot32:>5} {block_m:>8} {tile:>6} {warps:>6} {t * 1e3:>8.3f} "
                  f"{tf:>8.3f} {100 * tf / peak_tflops:>6.1f}% "
                  f"{t * 1e3 * args.layers:>9.1f} {rel:>9.2e}")

        rows.sort(key=lambda r: r["ms"])
        if rows:
            b = rows[0]
            print(f"  best: dot_fp32={b['dot_fp32']} BLOCK_M={b['BLOCK_M']} "
                  f"TILE={b['TILE']} warps={b['warps']}  {b['ms']:.3f} ms  "
                  f"({b['ms_all_layers']:.1f} ms over {args.layers} layers)")
        results[seq_len] = rows
        del q, kc, vc, out
        torch.cuda.empty_cache()

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
