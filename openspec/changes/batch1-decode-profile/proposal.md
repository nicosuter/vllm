# Profile batch-1 decode by kernel, on both production cards

## Why

Every performance number this fork currently holds is unusable for choosing what
to build next.

`ampere/README.md`'s headline — 62.5 ms inter-token against a ~4 ms/token
bandwidth roofline — comes from 230 samples that were all four-token health-check
generations against an otherwise idle engine, short enough that per-request fixed
costs dominate whatever the steady-state decode rate is. The README says so
itself and asks for it to be replaced.

`ada/README.md`'s hypotheses were measured on a **proxy card** (sm_89 standing in
for sm_86) and a **proxy model** (a 2B Qwen for the 27B), under the rule that
mechanism transfers but magnitude does not. No magnitude from that round can
justify a kernel.

Meanwhile three candidate directions have already died on inspection, which is
the argument for measuring before building rather than after:

- **FA3/FA4 backport.** `fa_utils.py:163-168` gates FA3 to `major == 9` and FA4
  to `major == 10`. The gate is hardware: FA3 is built on TMA, `wgmma`, and warp
  specialization with async barriers; FA4 adds `tcgen05`. sm_86 and sm_89 have
  `cp.async` and `mma.sync`, which FA2 already uses.
- **All-reduce compression.** Measured independently at ~4% of the step. Halving
  it buys 2%.
- **Per-expert MoE launch overhead.** Does not exist. MoE Marlin is a grouped
  GEMM (`moe/marlin_moe_wna16/marlin_template.h:543` selects `expert_id` per
  block), so `marlin_moe.py` issues two launches per layer regardless of top-k.

What is left is a genuine open question that no existing probe can answer:
**at batch 1, is the GPU compute-bound in the quantized-weight kernels, or is it
idle waiting for the CPU to launch the next one?** Those two worlds want opposite
fixes — a better kernel versus better graph coverage — and a table of summed
kernel durations, which is all the current probe produces, cannot distinguish
them.

## What changes

Both probes gain an occupancy measurement and are run on the real cards.

- `ada/probes/decode_profile.py` — add union-of-busy-intervals against wall-clock
  span, per step; separate prefill from decode by running twice rather than
  segmenting one mixed trace.
- `ampere/probes/decode_profile.py` — new. Same occupancy measurement, plus
  per-rank summaries for TP=2, because a rank blocked in an all-reduce looks busy
  in NCCL and idle everywhere else, and reading only rank 0 hides that.
- Both READMEs get their contaminated baselines replaced with measured
  per-bucket tables.

No engine or kernel code changes in this change. Its entire output is evidence.

## What would falsify this

The change is about producing a measurement, so it fails by producing an
unusable one:

- **Bucketing is wrong.** Buckets match on kernel-name substrings, which is
  fragile across versions. The per-kernel table is printed alongside so a
  mis-bucketed kernel is visible rather than folded silently into "everything
  else". If "everything else" is large, the table is not trustworthy.
- **JIT contaminates the trace.** Triton autotuning a kernel per shape inside the
  profiled region would swamp every real kernel. Mitigated by warming up on the
  same prompt outside `start_profile`, and detectable as absurd first-launch
  durations in the per-kernel table.
- **The occupancy figure is an artifact of the profiler.** `torch.profiler` adds
  per-launch overhead that inflates apparent idle time. If measured idle is
  large, it must be corroborated — by CUDA-graph replay counts, or by comparing
  against wall-clock tok/s outside the profiler — before any conclusion is drawn
  from it.

## Non-goals

- Changing runtime flags, live deployment config, or the quantization.
- Writing any kernel. That decision belongs to the next change, and only if the
  numbers here support one.
