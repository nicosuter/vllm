# Tasks

Every task producing a number states the card it was taken on. Numbers from a
proxy card or proxy model are directional only and never justify a patch.

## Harness

- [x] Add `union_busy` interval merge and occupancy reporting to
      `ada/probes/decode_profile.py`. *(unit-tested on 8 cases: empty, single,
      disjoint, overlapping, nested, unsorted, touching, chain+far)*
- [x] Split prefill from decode via `--prompt-tokens` / `--max-tokens` rather
      than segmenting one mixed trace.
- [x] Write `ampere/probes/decode_profile.py` with per-rank summaries for TP=2
      and Qwen3.5-specific buckets (GDN, full attention, AWQ Marlin, NCCL).
- [x] Both files pass `ruff check` and `ruff format --line-length 88`.
- [x] Confirm the probes actually execute. `profiler_config` exists on nightly
      `65b7662d3` with the `profiler` and `torch_profiler_dir` fields the probe
      passes; the API-skew risk did not materialise.

## Cluster access

- [x] Back up `ds/gemma4-26b-memory-reservation` and write `RESTORE.sh` before
      touching anything.
- [x] Scale `deploy/vllm-gemma4-26b-a4b` to 0. *(Verified idle first: no engine
      throughput lines since its 19:43 restart.)*
- [x] Park the memory-reservation DaemonSet.
- [x] Discover why nothing held: an **ApplicationSet** owns the Argo Application
      and regenerates its `automated` sync policy, so Application-level patches
      are reverted. Peel order is appset → app → workload; `OPEN-WINDOW.sh` now
      aborts if auto-sync is still set rather than proceeding into a doomed
      scale-down.
- [x] Bring up `ada-dev`. *(Had to move it to the `inference` namespace: PVCs
      are namespace-scoped and the checkpoint claim lives there.)*

## Measure — Gemma4-26B-A4B, RTX 4090 (sm_89)

*All numbers taken on the real 4090 (`w0-bo-k1`, compute capability 8.9, 350 W
cap) in the production nightly `0.26.1rc1.dev602+g65b7662d3`.*

- [x] Decode-dominated run: 5.741 ms/step device-busy, 674 launches/step.
- [x] Prefill-dominated run at 3,958 tokens: 12.6 ms device time, attention
      16.7% versus 2.6% at decode.
- [x] Corroborate occupancy outside the profiler: 5.723–5.751 ms/token with
      profiling off versus 5.741 ms device-busy — agreement to within 0.5%, so
      the reported 6% idle is profiler overhead and real occupancy is ~100%.
- [x] Calibrate against achievable rather than spec bandwidth: 919 GB/s measured
      on a large device-to-device copy, 91% of the 1008 GB/s spec.
- [x] Restore: `RESTORE.sh gemma`. Argo auto-sync and appset policy confirmed
      back. **Pending: `2/2 Running` and `/health` 200.**

## Findings — Gemma

- [x] **Not launch-bound.** Full CUDA graphs captured; ~100% occupancy.
- [x] **The README roofline was wrong**: it omitted the dense MLP. 3.97 GB/step,
      not 2.7; ceiling 232 tok/s, not 373; 75% of achievable, not 34%.
- [x] **New: the dense MLP is unquantized fp16** (1.07 GB/step) in a checkpoint
      whose attention and 128 experts are int4. With lm_head, 64% of bytes read
      per step is fp16 that could be int4.
- [x] **lm_head is at the roofline** (958 GB/s vs 919 measured ceiling) — no
      kernel can help it.
- [x] **A5 sized, then closed.** The in-model 722 GB/s looked like a 25% per-byte
      deficit. Measured directly past L2, Marlin reaches 856–872 GB/s against
      cuBLAS fp16's 855–952 — **~91%**. No per-byte headroom; a batch-1 GEMV
      wins nothing. The in-model figure is launch granularity (10–13 MB per
      launch plus fixed cost), set by model structure.
- [x] **Two benchmark traps recorded** in `marlin_m_sweep.py`: the 4090's 72 MiB
      L2 exceeds every Gemma4 weight matrix, so the model's own shapes measure
      cache (26 MB at an impossible 2.3 TB/s); and per-iteration synchronisation
      measures launch overhead, reporting a flat ~18 µs for both a 1.1 MB and a
      6.5 MB weight.
- [x] **Rows 2–32 are free** through the quantized layers (M=8 is 0.94× of M=1;
      only M=64 rises). A draft token costs nothing there, so the whole price of
      spec decode is the proposer — and `ngram`/`ngram_gpu` cost zero weights,
      which is the one form that survives the VRAM objection.

## Measure — Qwen3.5-27B, 2× RTX 3090 Ti (sm_86)

**Blocked on permission for the Qwen stack.** `ampere/k8s/local/` is built and
validated (2 GPUs, `w0.srv4.k1`, checkpoint mounted read-only, 40Gi scratch
claim); Qwen has no reservation DaemonSet and its node has room for the 58Gi pod.
The window needs the same appset → app → deployment peel.

- [ ] Same two runs at TP=2 with MTP-3, matching the deployment.
- [ ] Compare rank 0 and rank 1 occupancy. If they disagree sharply, the step is
      bounded by the slower rank and per-rank compute numbers cannot be read as
      the cost of the step.
- [ ] Re-measure sustained-load inter-token latency to replace the contaminated
      62.5 ms.
- [ ] Re-check H2's 1.42x CUDA-graph result at sm_86 and 64 layers.
- [ ] Restore and verify.

## Land

- [x] Replace `ada/README.md`'s baseline with the measured tables, keeping the
      falsified hypotheses on the record rather than deleting them.
- [x] Record the go/no-go on a batch-1 GEMV kernel with the number that decides
      it: **no-go**, Marlin is at ~91% of cuBLAS.
- [ ] Same for `ampere/README.md`, once the Qwen pair is measured.
