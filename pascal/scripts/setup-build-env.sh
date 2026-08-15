#!/usr/bin/env bash
# Prepare a build environment for the Pascal fork, inside the build pod.
#
#   kubectl -n vllm-pascal exec -i pascal-build -- bash -s < pascal/scripts/setup-build-env.sh
#
# Idempotent: re-running it skips work that is already done. Everything lands
# under /work so it survives pod restarts on the PVC.
set -euo pipefail

WORK=/work
SRC="$WORK/vllm-pascal"

echo "==> system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# Ubuntu's cargo (1.75) cannot parse vLLM's workspace manifest, which uses
# resolver = "3" and needs cargo >= 1.84, so it is deliberately not installed
# here; rustup provides the toolchain below.
# protobuf-compiler: the Rust vllm-server crate builds prost definitions and
# fails with "Could not find protoc" without it.
apt-get install -y -qq python3.12 python3.12-venv python3-pip git ccache cmake \
  ninja-build curl protobuf-compiler >/dev/null

echo "==> rust toolchain (rustup, not apt)"
export RUSTUP_HOME="$WORK/rustup" CARGO_HOME="$WORK/cargo"
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain stable >/dev/null
fi
export PATH="$CARGO_HOME/bin:$PATH"
cargo --version

echo "==> python venv + torch (cu126)"
[ -d "$WORK/venv" ] || python3 -m venv "$WORK/venv"
# shellcheck disable=SC1091
. "$WORK/venv/bin/activate"
pip install -q --upgrade pip wheel setuptools

# cu126 is the last PyTorch channel shipping Maxwell/Pascal/Volta cubins. Its
# arch list is "5.0;6.0;7.0;..." — note it has 6.0 and not 6.1, which is fine:
# CUDA runs an sm_X.y cubin on sm_X.z whenever z >= y, so sm_60 code executes on
# this sm_61 card. Installing torch from any other index silently produces a
# build that cannot launch a single kernel here.
pip install -q --index-url https://download.pytorch.org/whl/cu126 torch==2.13.0 torchvision==0.28.0

echo "==> build dependencies"
pip install -q "cmake>=3.26.1" ninja "packaging>=24.2" "setuptools>=77.0.3,<81.0.0" \
  "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" wheel jinja2 numpy

echo "==> sanity"
nvcc --version | tail -2
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arch_list", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
PY

# setuptools-scm reads the version from git metadata. The source is copied into
# the pod as a tarball, so give it a repository and the upstream base tag,
# otherwise the build aborts before compiling anything.
if [ ! -d "$SRC/.git" ]; then
  echo "==> seeding git metadata for setuptools-scm"
  git config --global user.email "build@vllm-pascal.local"
  git config --global user.name "pascal build"
  git config --global --add safe.directory "$SRC"
  git -C "$SRC" init -q
  git -C "$SRC" add -A
  git -C "$SRC" commit -qm "vllm-pascal source snapshot"
  git -C "$SRC" tag v0.27.1
fi

echo "==> ready. build with:"
echo "    pascal/scripts/build.sh"
