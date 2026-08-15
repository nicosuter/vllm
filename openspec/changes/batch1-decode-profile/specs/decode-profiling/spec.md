# decode-profiling

## ADDED Requirements

### Requirement: Separate device-busy time from device-idle time

A profile SHALL report how much of the measured wall-clock span the GPU was
executing at least one device event, and how much it was executing none.

Summed kernel durations answer "which kernel is slowest" but not "was the card
even busy". At batch 1 those are different questions with opposite fixes: slow
kernels want a better kernel, an idle card wants better CUDA-graph coverage. A
probe that cannot distinguish them cannot justify either.

#### Scenario: Kernels overlap across streams

- **WHEN** device events from different streams overlap in time
- **THEN** the busy figure counts the union of their intervals, not the sum of
  their durations
- **AND** the reported busy time never exceeds the wall-clock span

#### Scenario: The card is idle between launches

- **WHEN** consecutive device events are separated by gaps
- **THEN** those gaps are reported as idle time, both in total and per generated
  token
- **AND** the launch count per generated token is reported alongside, so idle
  time can be attributed to launch overhead

#### Scenario: No device events were captured

- **WHEN** a trace contains no kernel, memcpy, or memset events
- **THEN** the probe reports the failure and exits non-zero rather than printing
  a table of zeros

### Requirement: Attribute device time to named buckets without hiding misses

A profile SHALL group kernels into model-meaningful buckets and SHALL make
mis-bucketed kernels visible rather than silently absorbing them.

Buckets match on kernel-name substrings, which is fragile across vLLM versions.
A bucketing error that is invisible produces a confident wrong conclusion.

#### Scenario: A kernel matches no known bucket

- **WHEN** a kernel name matches no bucket substring
- **THEN** it is attributed to an explicit "everything else" bucket
- **AND** the per-kernel table is printed alongside the bucket table, so the
  unmatched kernel is inspectable by name

#### Scenario: A collective could be mistaken for compute

- **WHEN** an NCCL kernel is bucketed
- **THEN** it is matched before any compute bucket, so a collective is never
  folded into a compute total

### Requirement: Report every rank under tensor parallelism

A profile of a tensor-parallel deployment SHALL summarize each rank and compare
their occupancy.

vLLM constructs its profiler per worker, so TP=2 writes one trace per rank. A
rank blocked in an all-reduce waiting for its peer appears busy in NCCL and idle
everywhere else; reading only rank 0 hides that the step is bounded by the other
rank.

#### Scenario: Two ranks report different occupancy

- **WHEN** more than one rank trace is present
- **THEN** each rank is summarized separately
- **AND** a cross-rank table reports each rank's busy percentage and all-reduce
  share

### Requirement: Measure prefill and decode separately

A profile SHALL NOT present a figure drawn from a mixed prefill-and-decode run as
though it characterized either phase.

Attention cost scales with context; a batch-1 decode figure says nothing about
prefill, and a long-prompt figure says nothing about decode.

#### Scenario: Characterizing decode

- **WHEN** the decode phase is the subject
- **THEN** the run generates many tokens from a short prompt, so a single prefill
  forward is a rounding error against the decode forwards

#### Scenario: Characterizing prefill

- **WHEN** the prefill phase is the subject
- **THEN** the run uses a prompt of a caller-specified length and generates one
  token
- **AND** the synthesized prompt's actual tokenized length is reported, not the
  requested one

### Requirement: Keep JIT compilation out of the measured region

A profile SHALL warm up on the same prompt shape it will measure, outside the
profiled region.

Triton compiles and autotunes a kernel per shape on first use. Inside the
profiled region that cost swamps every real kernel and the resulting table is
meaningless.

#### Scenario: First use of a shape

- **WHEN** a profiled run begins
- **THEN** a warm-up generation on the same prompt has already completed before
  profiling starts
