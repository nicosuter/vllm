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
- [ ] Confirm the probes actually execute. `profiler_config={...}` is v0.27.1
      API and the Gemma pod runs nightly `65b7662d3`; the kwarg may differ.
      **Not yet verified — linting is not execution.**

## Cluster access

- [x] Back up `ds/gemma4-26b-memory-reservation` and write `RESTORE.sh` before
      touching anything.
- [x] Scale `deploy/vllm-gemma4-26b-a4b` to 0. *(Verified idle first: no engine
      throughput lines since its 19:43 restart.)*
- [ ] Park the memory-reservation DaemonSet. It holds 20Gi of `w0-bo-k1`'s
      ~26.5Gi allocatable, so the dev pod stays Pending without this.
      **Blocked: `kubectl patch` trips the permission classifier.**
- [ ] Bring up `ada-dev` via `kubectl apply -k ada/k8s/local`.

## Measure — Gemma4-26B-A4B, RTX 4090 (sm_89)

- [ ] Decode-dominated run: `--max-tokens 128`.
- [ ] Prefill-dominated run: `--prompt-tokens 3926 --max-tokens 1`, matching
      production's mean prompt.
- [ ] Corroborate the occupancy figure against wall-clock tok/s measured outside
      the profiler, so the idle number is not a profiler artifact.
- [ ] Restore: run `RESTORE.sh`, confirm `2/2 Running` and `/health` 200.

## Measure — Qwen3.5-27B, 2x RTX 3090 Ti (sm_86)

- [ ] Same two runs at TP=2 with MTP-3, matching the deployment.
- [ ] Compare rank 0 and rank 1 occupancy. If they disagree sharply, the step is
      bounded by the slower rank and per-rank compute numbers cannot be read as
      the cost of the step.
- [ ] Re-measure sustained-load inter-token latency to replace the contaminated
      62.5 ms.
- [ ] Restore and verify.

## Land

- [ ] Replace the baseline sections of both READMEs with the measured tables,
      keeping the falsified hypotheses on the record rather than deleting them.
- [ ] Record the go/no-go on a batch-1 GEMV kernel, with the bandwidth-saturation
      number that decides it.
