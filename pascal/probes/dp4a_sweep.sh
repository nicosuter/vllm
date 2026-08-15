#!/usr/bin/env bash
# Sweeps tile configurations for the W4A8 dp4a GEMM.
#
# Hand-tuning established that registers, not arithmetic intensity or memory
# latency, bind this kernel: every attempt to widen the micro-tile or prefetch
# into registers lost. That points at the CUTLASS-shaped design -- fewer threads
# each doing much more work, trading occupancy for instruction-level parallelism
# -- which is a different point in the space rather than a further tweak, and
# worth searching rather than guessing.
#
#   bash pascal/probes/dp4a_sweep.sh
#
# Prints TOP/s at prefill2048 (the largest shape, where the kernel is most
# compute-bound and the roof matters most) and the speedup over cuBLAS.

set -u
SRC=${SRC:-/work/vllm-pascal/pascal/probes/dp4a_gemm.cu}
OUT=${OUT:-/work/dp4a_sweep}

# BLOCK_M BLOCK_N BLOCK_K THREADS TM TN
# The constraint is (BLOCK_M/TM) * (BLOCK_N/TN) == THREADS, enforced by a
# static_assert in the kernel, so bad rows fail loudly at compile time.
CONFIGS="
64 64 128 256 4 4
64 64 64 256 4 4
64 128 128 256 4 8
128 64 128 256 8 4
64 128 64 128 8 8
128 128 64 256 8 8
128 128 128 256 8 8
128 128 128 1024 4 4
64 64 128 128 8 4
"

# Register cap is swept alongside the tile, because it is not independent of it:
# uncapped, this kernel takes 126 registers and gets 2 blocks per SM, and an
# earlier sweep that left it uncapped was measuring the compiler's allocation
# decisions rather than the tile. -maxrregcount=80 alone was worth 1.41x -> 1.80x.
REGCAPS=${REGCAPS:-"96 80 72"}

printf '%-28s %5s %10s %10s %9s\n' "config (M N K thr TM TN)" "reg" "TOP/s" "ms" "vs cuBLAS"
printf '%s\n' "--------------------------------------------------------------"

echo "$CONFIGS" | while read -r bm bn bk th tm tn; do
  [ -z "${bm:-}" ] && continue
  tag="$bm $bn $bk $th $tm $tn"
  for rc in $REGCAPS; do
    if ! nvcc -arch=sm_61 -O3 --extended-lambda -lcublas -maxrregcount=$rc \
         -DBLOCK_M=$bm -DBLOCK_N=$bn -DBLOCK_K=$bk \
         -DTHREADS=$th -DTM=$tm -DTN=$tn \
         -o "$OUT" "$SRC" 2>/tmp/nvcc.err; then
      printf '%-28s %5s %10s\n' "$tag" "$rc" "COMPILE"
      continue
    fi
    res=$("$OUT" 2>/dev/null)
    if ! echo "$res" | grep -q PASS; then
      printf '%-28s %5s %10s\n' "$tag" "$rc" "WRONG"
      continue
    fi
    line=$(echo "$res" | grep "prefill2048")
    ms=$(echo "$line" | awk '{print $3}')
    sp=$(echo "$line" | awk '{print $5}')
    top=$(echo "$line" | awk '{print $6}')
    printf '%-28s %5s %10s %10s %9s\n' "$tag" "$rc" "$top" "$ms" "$sp"
  done
done
