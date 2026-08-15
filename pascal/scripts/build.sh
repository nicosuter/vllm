#!/usr/bin/env bash
# Build the vLLM C extensions for Pascal, inside the build pod.
#
#   kubectl -n vllm-pascal exec -i pascal-build -- bash -s < pascal/scripts/build.sh
#
# Run pascal/scripts/setup-build-env.sh first.
set -euo pipefail

WORK=/work
SRC="$WORK/vllm-pascal"
LOG="$WORK/build.log"

# shellcheck disable=SC1091
. "$WORK/venv/bin/activate"
export RUSTUP_HOME="$WORK/rustup" CARGO_HOME="$WORK/cargo" PATH="$WORK/cargo/bin:$PATH"

# 6.1 only. Setting a single arch is not just an optimization: it is what makes
# CMake and setup.py drop FlashAttention, Marlin, Machete, CUTLASS SM80+, the
# FP8 paths and QuTLASS, none of which can run on this card.
export TORCH_CUDA_ARCH_LIST="6.1"
export VLLM_TARGET_DEVICE=cuda
export CCACHE_DIR="$WORK/.ccache"

# nvcc peaks near 2 GB per translation unit; 8 against this node's 24 GB is as
# far as it goes without swapping.
export MAX_JOBS="${MAX_JOBS:-8}"
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"

cd "$SRC"
echo "==> building (log: $LOG)"
if python setup.py build_ext --inplace > "$LOG" 2>&1; then
  echo "==> build OK"
else
  status=$?
  echo "==> build FAILED (exit $status)"
  echo "--- distinct errors ---"
  grep -oE "error: .*" "$LOG" | sort -u | head -30
  echo "--- tail ---"
  tail -30 "$LOG"
  exit "$status"
fi

echo "==> extensions built"
find "$SRC/vllm" -maxdepth 2 -name "*.so" -printf "    %p\n"

echo "==> import check"
cd "$SRC"
python - <<'PY'
import torch  # noqa: F401  (must precede the extension import)

# The CUDA kernels live in _C_stable_libtorch, not _C: vLLM moved them to the
# libtorch-stable ABI. Probing the old name made a successful build report
# failure, which is worse than not checking at all.
import vllm._C_stable_libtorch  # noqa: F401
import vllm._moe_C_stable_libtorch  # noqa: F401

# Cheap proof the quantized path is actually wired up, since that is the only
# GEMM this fork can use for W4A16.
from vllm import _custom_ops as ops

assert hasattr(ops, "gptq_gemm"), "gptq_gemm missing from _custom_ops"
print("extensions imported; gptq_gemm present")
PY
