# vLLM built for Pascal (SM 6.1).
#
# Not derived from docker/Dockerfile: upstream's image builds for tensor-core
# architectures, installs requirements/cuda.txt and pulls torch from the default
# index, and every one of those is wrong here. The differences that matter are
# the same three the fork exists for -- CUDA 12.6, the cu126 torch channel, and
# TORCH_CUDA_ARCH_LIST=6.1 -- so this is written directly rather than patched.
#
#   docker build -f pascal/docker/Dockerfile -t vllm-pascal .
#
# Expect a long build. Compiling the extensions for sm_61 is the bulk of it, and
# unlike upstream there is no precompiled wheel to fall back on.

# 12.6 is the last toolkit line that still emits sm_61 code; CUDA 13 dropped it.
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04
FROM ${CUDA_DEVEL_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=6.1
ENV VLLM_TARGET_DEVICE=cuda

# Ubuntu's cargo (1.75) cannot parse vLLM's workspace manifest, which uses
# resolver = "3" and needs cargo >= 1.84, so rustup provides the toolchain.
# protobuf-compiler: the vllm-server crate builds prost definitions and fails
# with "Could not find protoc" without it.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev python3-pip \
      git ccache cmake ninja-build curl ca-certificates protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
ENV PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain stable

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH=$VIRTUAL_ENV/bin:$PATH

# cu126 is the only PyTorch channel still shipping Maxwell/Pascal/Volta cubins.
# Its arch list has 6.0 but not 6.1, which is fine: CUDA runs an sm_X.y cubin on
# sm_X.z whenever z >= y, so sm_60 code executes on this sm_61 card. Installing
# torch from any other index silently produces a build that cannot launch a
# single kernel here, which is why it is pinned and installed before anything
# that might resolve it differently.
RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
       torch==2.13.0 torchvision==0.28.0

RUN pip install --no-cache-dir "cmake>=3.26.1" ninja "packaging>=24.2" \
      "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" \
      wheel jinja2 numpy

WORKDIR /src
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/pascal.txt

COPY . .

# setuptools-scm derives the version from git metadata, which a build context
# may not carry. Supplying it explicitly avoids a failure that has nothing to do
# with the code.
ARG VLLM_VERSION=0.27.2.dev0+pascal
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VLLM_VERSION}

# nvcc peaks near 2 GB per translation unit, so MAX_JOBS is memory-bound rather
# than core-bound. Override it to match the builder.
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}
ENV CCACHE_DIR=/ccache

# TORCH_CUDA_ARCH_LIST=6.1 does more than select codegen: it is what makes
# CMake and setup.py drop FlashAttention, Marlin, Machete, CUTLASS SM80+,
# QuTLASS and the FP8 paths, each of which selects itself out by target arch.
RUN --mount=type=cache,target=/ccache \
    pip install --no-cache-dir --no-build-isolation -e . -v

# torchaudio and torchcodec arrive transitively (transformers 5.x pulls them for audio/video)
# from the DEFAULT index, so they are cu13 wheels whose .so files want libcudart.so.13 while
# everything here is cu126. requirements/pascal.txt already calls torchcodec out of scope; this
# is what makes that true of the built image rather than only of the requirements file.
#
# Uninstalling rather than pinning is deliberate: neither is needed to serve text, and both
# libraries are ALREADY optional to vLLM -- vllm/multimodal/video.py wraps the torchcodec
# import and falls back to a PlaceholderModule. That guard catches (ImportError, RuntimeError)
# and torch.ops.load_library raises OSError, so a present-but-broken wheel is strictly worse
# than an absent one: absent degrades, broken kills the process at import.
RUN pip uninstall -y torchaudio torchcodec || true

FROM ${CUDA_RUNTIME_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
# gcc and the Python headers are not optional here, and the reason is the same one that makes
# the note further down true: with FlashAttention compiled out at arch 6.1, vLLM falls back to
# the Triton attention backend, and Triton builds its launcher stubs with a C compiler AT RUN
# TIME. Without these, every model dies in profile_run with "Failed to find C compiler" -- so
# `vllm serve` could not start at all, for any checkpoint. It was not caught earlier because
# everything in pascal/MODELS.md was measured inside the build pod, which is the devel image
# and carries a full toolchain.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev libgomp1 gcc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /src /src
ENV PATH=/opt/venv/bin:$PATH
ENV VIRTUAL_ENV=/opt/venv
WORKDIR /src

# Triton compiles and autotunes every kernel for sm_61 on first run, which takes
# many minutes and 300+ MB. Point the cache somewhere a volume can be mounted so
# a container restart does not pay it again.
ENV TRITON_CACHE_DIR=/cache/triton
ENV HF_HOME=/cache/hf
VOLUME ["/cache"]

ENTRYPOINT ["vllm"]
CMD ["--help"]
