"""Check and time the sm_61 flash-attention prefill kernel.

Compares three implementations of the same causal paged attention at this
model's shape (16 query heads, 8 KV heads, head_dim 128):

    triton   vLLM's `unified_attention`, as this fork now tunes it
    flash    `fa_prefill_sm61.cu`
    fp32 ref an independent torch implementation, as the correctness bar

The number that matters is TFLOP/s against this card's 8.19. Triton reaches
13.9% of that even with float operands, which is what makes a CUDA kernel worth
writing: the algorithm needs nothing sm_61 lacks.

    python pascal/probes/fa_prefill_check.py
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

N_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE = 16, 8, 128, 16
N_LAYERS = 28
PEAK_TFLOPS = 8.19


def build_inputs(seq_len: int, dtype=torch.float16):
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    q = torch.randn(seq_len, N_HEADS, HEAD_DIM, device="cuda", dtype=dtype) * 0.1
    kc = torch.randn(num_blocks, BLOCK_SIZE, N_KV_HEADS, HEAD_DIM,
                     device="cuda", dtype=dtype) * 0.1
    vc = torch.randn(num_blocks, BLOCK_SIZE, N_KV_HEADS, HEAD_DIM,
                     device="cuda", dtype=dtype) * 0.1
    out = torch.empty_like(q)
    cu = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
    used = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    bt = torch.arange(num_blocks, device="cuda", dtype=torch.int32).unsqueeze(0)
    return q, kc, vc, out, cu, used, bt


def reference(q, kc, vc, seq_len, scale):
    k = kc.reshape(-1, N_KV_HEADS, HEAD_DIM)[:seq_len]
    v = vc.reshape(-1, N_KV_HEADS, HEAD_DIM)[:seq_len]
    rep = N_HEADS // N_KV_HEADS
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    qf = q.float().transpose(0, 1)
    scores = torch.bmm(qf, k.float().transpose(0, 1).transpose(1, 2)) * scale
    mask = torch.full((seq_len, seq_len), float("-inf"), device=q.device)
    scores += torch.triu(mask, diagonal=1)
    return torch.bmm(F.softmax(scores, dim=-1),
                     v.float().transpose(0, 1)).transpose(0, 1)


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
    ap.add_argument("--out", default="/work/fa_prefill_check.json")
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--src", default=os.path.join(os.path.dirname(__file__),
                                                  "fa_prefill_sm61.cu"))
    args = ap.parse_args()

    ext = load(
        name="fa_prefill_sm61",
        sources=[args.src],
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_61,code=sm_61",
                           "--ptxas-options=-v"],
        verbose=True,
    )
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    results = {}
    print(f"\n{'n':>6} {'impl':<14} {'ms/layer':>9} {'TFLOP/s':>8} {'%peak':>7} "
          f"{f'x{N_LAYERS}':>9} {'rel_err':>10}")
    print("-" * 62)
    for n in args.seq_lens:
        q, kc, vc, out, cu, used, bt = build_inputs(n)
        scale = HEAD_DIM ** -0.5
        flops = 2 * (2 * n * n * HEAD_DIM * N_HEADS) / 2
        ref = reference(q, kc, vc, n, scale)
        scale_ref = ref.abs().max()

        def triton_run():
            unified_attention(
                q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, seqused_k=used,
                max_seqlen_q=n, max_seqlen_k=n, softmax_scale=scale, causal=True,
                window_size=(-1, -1), block_table=bt, softcap=0,
                q_descale=None, k_descale=None, v_descale=None)

        def flash_run(br=0):
            ext.fa_prefill(q, kc, vc, out, cu, used, bt, n, scale, br)

        rows = {}
        impls = [("triton", triton_run)]
        # BR=128 would give the register tile that actually pays, but its query
        # tile and score buffer need 64 KiB and Pascal caps a block at 48 KiB, so
        # it cannot launch at all. That is the finding, not an omission.
        impls += [(f"flash BR={b}", (lambda b=b: flash_run(b))) for b in (32, 64)]
        for label, fn in impls:
            # Zeroed first: `out` is shared between the two implementations, so a
            # kernel that silently failed to launch would otherwise be scored on
            # its predecessor's output and look perfect.
            out.zero_()
            fn()
            torch.cuda.synchronize()
            rel = ((out.float() - ref).abs().max() / scale_ref).item()
            assert out.abs().max().item() > 0, f"{label} wrote nothing"
            t = timeit(fn)
            tf = flops / t / 1e12
            rows[label] = {"ms": round(t * 1e3, 3), "TFLOP_s": round(tf, 3),
                           "pct_peak": round(100 * tf / PEAK_TFLOPS, 1),
                           "ms_all_layers": round(t * 1e3 * N_LAYERS, 1),
                           "rel_err": rel}
            print(f"{n:>6} {label:<14} {t * 1e3:>9.3f} {tf:>8.3f} "
                  f"{100 * tf / PEAK_TFLOPS:>6.1f}% {t * 1e3 * N_LAYERS:>9.1f} "
                  f"{rel:>10.2e}")
        best = min((v["ms"], k) for k, v in rows.items() if k != "triton")
        print(f"{'':>6} {'best ' + best[1]:<14} "
              f"{rows['triton']['ms'] / best[0]:>5.2f}x vs triton")
        results[n] = rows
        del q, kc, vc, out, ref
        torch.cuda.empty_cache()
        print()

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
