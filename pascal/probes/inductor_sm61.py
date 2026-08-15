"""Inductor on sm_61: what the sm_70 floor is actually made of, and what lifting
it is worth.

vLLM has always run here with `mode=NONE`, on the understanding that torch.compile
cannot work below sm_70 -- inductor raises GPUTooOldForTriton. That is true, and
it hides something: Triton itself works fine on this card, which is why hundreds
of its kernels already run, attention included.

Three gates in torch key on device major >= 7:

    torch/utils/_triton.py             has_triton()'s cuda_extra_check
    torch/_dynamo/device_interface.py  CudaInterface.is_triton_capable
    torch/_inductor/scheduler.py       re-raises when has_triton() is False

Lifting them is not sufficient on its own. Inductor asks for cache-eviction
hints on loads, and ptxas rejects them:

    ptxas error: Modifier '.evict_last' on 'ld' requires .target sm_70 or higher

Those hints are advisory -- they tell L1 what to discard first and change no
result -- so dropping them at Triton's frontend costs a cache hint and nothing
else. `_str_to_eviction_policy` is the single funnel every policy passes through.

The last piece is that inductor compiles in worker subprocesses by default, which
do not inherit an in-process patch: 108 ptxas errors become 12, not 0. Forked
workers do inherit it, hence TORCHINDUCTOR_WORKER_START=fork rather than
serialising compilation with TORCHINDUCTOR_COMPILE_THREADS=1.

    TORCHINDUCTOR_WORKER_START=fork VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        python pascal/probes/inductor_sm61.py /work/models/Qwen3.5-2B-AWQ-4bit \
            [eager|inductor] [reference.json]

One backend per process: vLLM does not release the engine's device memory on
del, so a second engine in the same process starts with 1.68 of 7.92 GiB.
"""

from __future__ import annotations

import json
import sys

import torch.utils._triton as _triton_utils
from torch._dynamo.device_interface import CudaInterface


def lift_inductor_floor() -> None:
    """Make inductor usable on sm_61. Call before importing vllm."""
    _triton_utils.has_triton = lambda: True
    CudaInterface.is_triton_capable = staticmethod(lambda device=None: True)

    from triton.language import semantic as tl_semantic

    tl_semantic.TritonSemantic._str_to_eviction_policy = lambda self, eviction_policy: (
        tl_semantic.ir.EVICTION_POLICY.NORMAL
    )


def main() -> int:
    model = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "inductor"
    reference = sys.argv[3] if len(sys.argv) > 3 else None

    if backend == "inductor":
        lift_inductor_floor()

    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode

    sys.path.insert(0, "pascal/scripts")
    import bench

    llm = LLM(
        model=model,
        dtype="auto",
        enforce_eager=False,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        trust_remote_code=True,
        # The gate model is multimodal, and profiling the vision tower with a
        # max-size image costs far more here than the text path it is sizing.
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            # get_compile_backend() reads simple_compile_backend, which this
            # fork pins to eager below sm_70, so inductor has to be asked for.
            backend=backend,
            cudagraph_mode=CUDAGraphMode.FULL,
        ),
    )

    # Same slope measurement as pascal/scripts/bench.py, so the number is
    # directly comparable to what that already records.
    result = bench.bench_one(llm, 16, 144, 3, 1)
    print(
        f"RESULT backend={backend} {result['decode_tok_s']:.2f} tok/s decode "
        f"({result['ms_per_step']:.2f} ms/step)",
        flush=True,
    )

    if reference is None:
        return 0

    # Fusion is allowed to reassociate, so compare against the path already
    # checked against the CPU fp32 reference rather than trusting it.
    with open(reference) as fh:
        ref = json.load(fh)
    outs = llm.generate(
        [r["prompt"] for r in ref], SamplingParams(temperature=0.0, max_tokens=32)
    )
    worst = 1.0
    print(f"\n{'=' * 72}\nagreement with the eager path, greedy\n{'=' * 72}")
    for r, o in zip(ref, outs):
        got, want = list(o.outputs[0].token_ids), r["token_ids"]
        match = 0
        for a, b in zip(got, want):
            if a != b:
                break
            match += 1
        n = min(len(got), len(want))
        worst = min(worst, match / max(1, n))
        print(
            f"  prefix_match={match:>3}/{n} ({match / max(1, n):>4.0%})  "
            f"{r['prompt'][:40]!r}"
        )
    print(f"\nworst: {worst:.0%}")
    return 0 if worst == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
