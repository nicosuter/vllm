"""Scan a checkpoint for values fp16 cannot represent.

Pascal has no bfloat16, so this fork converts models to fp16. The risk in that
is range, not precision: bf16 carries 8 exponent bits (the reach of fp32) while
fp16 carries 5, so anything above 65504 becomes inf and anything below ~6.1e-5
falls into subnormals or flushes to zero. bf16 is in fact the *less* precise
format of the two (7 mantissa bits against 10); only the exponent is the
problem.

Whether that is a real problem or a theoretical one is a property of the
checkpoint, so it is measured rather than argued about. Runs on CPU and reads
tensors one at a time, so it needs no GPU and does not need the model to fit.

    python pascal/probes/bf16_range.py /work/models/some-model

Thresholds reported per tensor:

    overflow    |x| > 65504            -> inf in fp16
    subnormal   0 < |x| < 6.104e-5     -> representable but with reduced mantissa
    flush       0 < |x| < 5.960e-8     -> rounds to zero in fp16

Subnormals are counted separately from flushes because they are usually benign:
a weight that small contributes nothing to a dot product against normal
activations. Overflow is the one that corrupts output.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-05
FP16_MIN_SUBNORMAL = 5.960464477539063e-08


def scan(path: str, top: int, all_dtypes: bool) -> int:
    import torch
    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        print(f"no .safetensors under {path}", file=sys.stderr)
        return 1

    rows = []
    totals = {"overflow": 0, "subnormal": 0, "flush": 0, "elements": 0}
    dtypes: dict[str, int] = {}

    for f in files:
        with safe_open(f, framework="pt", device="cpu") as fh:
            for name in fh.keys():
                t = fh.get_tensor(name)
                dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
                if not t.is_floating_point():
                    continue
                if not all_dtypes and t.dtype not in (torch.bfloat16, torch.float32):
                    # fp16 tensors are already in range by construction, and
                    # integer tensors are packed quantized weights whose scales
                    # carry the magnitude.
                    continue

                # Chunked, because t.float().abs() on a 248320x2048 embedding
                # materialises two multi-GB copies and gets the process killed
                # inside a container memory limit. The counts are additive, so
                # slicing costs nothing but a loop.
                flat = t.reshape(-1)
                n = flat.numel()
                over = sub = flush = 0
                max_abs = 0.0
                CHUNK = 8 << 20
                for i in range(0, n, CHUNK):
                    x = flat[i : i + CHUNK].float().abs_()
                    nonzero = x > 0
                    over += int((x > FP16_MAX).sum())
                    sub += int((nonzero & (x < FP16_MIN_NORMAL)).sum())
                    flush += int((nonzero & (x < FP16_MIN_SUBNORMAL)).sum())
                    m = float(x.max()) if x.numel() else 0.0
                    if m > max_abs:
                        max_abs = m
                    del x, nonzero

                totals["overflow"] += over
                totals["subnormal"] += sub
                totals["flush"] += flush
                totals["elements"] += n

                if over or flush:
                    rows.append(
                        {
                            "tensor": name,
                            "dtype": str(t.dtype),
                            "max_abs": max_abs,
                            "overflow": over,
                            "flush": flush,
                            "elements": n,
                        }
                    )

    print(f"scanned {len(files)} shard(s) under {path}")
    print("tensor dtypes present:", ", ".join(f"{k}x{v}" for k, v in sorted(dtypes.items())))
    print(
        f"\nelements checked: {totals['elements']:,}\n"
        f"  overflow  (> {FP16_MAX:g}):        {totals['overflow']:,}\n"
        f"  subnormal (< {FP16_MIN_NORMAL:.3e}):  {totals['subnormal']:,}\n"
        f"  flush     (< {FP16_MIN_SUBNORMAL:.3e}):  {totals['flush']:,}"
    )

    if rows:
        rows.sort(key=lambda r: (-r["overflow"], -r["flush"]))
        print(f"\n{'overflow':>10} {'flush':>12} {'max_abs':>12}  tensor")
        print("-" * 100)
        for r in rows[:top]:
            print(
                f"{r['overflow']:>10,} {r['flush']:>12,} {r['max_abs']:>12.4g}  "
                f"{r['tensor'][:60]}"
            )

    if totals["overflow"]:
        print(
            "\nOverflow present: a plain bf16 -> fp16 cast will produce inf here.\n"
            "The fix is a scale, not a wider float -- quantizing these layers "
            "attaches a per-channel scale that puts the values back in range, "
            "which is why a quantized checkpoint sidesteps this entirely."
        )
    else:
        print(
            "\nNo overflow: every value is inside fp16's range, so the cast is "
            "safe for the weights. This says nothing about activations, which "
            "can still exceed it at runtime."
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a checkpoint directory")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument(
        "--all-dtypes",
        action="store_true",
        help="also scan tensors already stored as fp16 (normally pointless)",
    )
    args = ap.parse_args()
    return scan(args.model, args.top, args.all_dtypes)


if __name__ == "__main__":
    sys.exit(main())
