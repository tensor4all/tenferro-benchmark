# Result Layout and Metadata

`suite_id` identifies the workload. `target_profile` identifies the hardware
profile where the suite was run.

Current target profiles:

- `mac-cpu`
- `amd-cpu`
- `nvidia-gpu`

Latest human-facing reports are kept under `result/<target_profile>/...`:

```text
result/
  mac-cpu/
    cpu/einsum.md
    cpu/cpu_ops.md
    cpu/linalg_jvp_vjp.md
    cpu/permutation.md
  amd-cpu/
    cpu/einsum.md
    cpu/cpu_ops.md
    cpu/linalg_jvp_vjp.md
    cpu/permutation.md
    cpu/small_work.md
  nvidia-gpu/
    gpu/dense.md
    gpu/einsum.md
    gpu/sparse.md
    gpu/linalg_jvp_vjp.md
    gpu/permutation.md
```

Raw run data is written under `data/results/<target_profile>/<suite_id>/<timestamp>/`:

```text
data/results/mac-cpu/cpu/einsum/<timestamp>/run.yaml
data/results/mac-cpu/cpu/permutation/<timestamp>/run.yaml
data/results/amd-cpu/cpu/einsum/<timestamp>/run.yaml
data/results/amd-cpu/cpu/small_work/<timestamp>/run.yaml
data/results/nvidia-gpu/gpu/dense/<timestamp>/run.yaml
data/results/nvidia-gpu/gpu/permutation/<timestamp>/run.yaml
```

Historical report files are not maintained. Use git history to inspect older
latest reports.

`run.yaml` records:

- `target_profile`
- `suite_id`
- suite file path
- timestamp
- tenferro-rs path, commit, dirty state, and features
- host CPU/OS metadata
- controlled thread environment variables
- BLAS provider for tenferro
- PyTorch/JAX provider metadata
- CUDA metadata for GPU runs where available

The benchmark repository commit is intentionally not stored in `run.yaml`; the
result file's git history already records it. A dirty `extern/tenferro-rs`
checkout may be inspected by correctness-only runs, but cannot provide timing
provenance; the receipt records dirty/unknown state and rejects promotion.

The small-work driver freezes `run.yaml` before measured children, archives each
child's stdout/stderr, and writes raw records under the standard timestamped
path. A busy-host correctness smoke (which never emits timing samples) is:

```bash
uv run python scripts/run_small_work.py \
  --suite benchmarks/cpu/small_work.yaml \
  --case-binary target/release/small_work_case \
  --correctness-only --output /tmp/small-work-correctness.json
```

For the eight size-4 cases only, set the exact-ID filter before launch:

```bash
SMALL_WORK_CASE_FILTER='add_f64_concrete_fresh,add_f64_concrete-fresh_size4_dependent10,add_f64_concrete-shared_size4_single,add_f64_concrete-shared_size4_dependent10,add_f64_eager-no-ad_size4_single,add_f64_eager-no-ad_size4_dependent10,add_f64_eager_active_ad,add_f64_eager-ad_size4_dependent10' \\
uv run python scripts/run_small_work.py --suite benchmarks/cpu/small_work.yaml \\
  --case-binary target/release/small_work_case --correctness-only \\
  --output /tmp/small-work-n4-correctness.json
```

An explicit `SMALL_WORK_CASE_FILTER` selects a bounded campaign: it still runs
all validation and receipt checks and archives the selected-case `report.md`,
`results.jsonl`, `run.yaml`, and `run.json`, but never replaces the full-suite
latest `result/<target_profile>/cpu/small_work.md`. A valid filtered timing run
is still reported as `READY`; omit the filter to publish the latest full-suite
report.

A complete public campaign that exceeds the frozen process-median CoV limit is
`INCONCLUSIVE`: all samples/statistics remain in the archive, but the report hides
unusable timing values and latest is unchanged. Missing/malformed samples,
under-duration samples and correctness failures remain `FAILED`, even alongside
noise. No threshold is relaxed or repetition removed. As with unavailable
resources, `INCONCLUSIVE` exits zero; automation must inspect the status, not just
the exit code. `READY` is single-campaign readiness, not paired performance acceptance.

### Passive component machine observations

Component **timing diagnostics** opt into the existing serial child runner's
read-only observer. It samples selected CPUs' sysfs `scaling_cur_freq` (kHz) and
available thermal/hwmon temperature inputs (millidegrees Celsius) while waiting
for each timed child. No sudo, settings changes, helper service or extra thread
is used. Ordinary suite runs, correctness and allocation diagnostics do not opt in.

`commands[].machine_observations` records a 100ms wait-poll interval, monotonic read
start/end timestamps, source paths, null/error readings and at most 1024 samples.
Sensor-read time is additional: this is not a guaranteed 100ms sampling cadence.
Short children explicitly report `boundary_only`; `interior` means observations
while awaiting command completion, **not** synchronization with individual timed
samples. If the cap is reached, `truncated` is set and the final observation is
retained. Missing sensors are not invented; hwmon/thermal paths are not asserted
to identify a particular CPU. Kernel-reported frequency is not an instantaneous
instruction-level clock measurement.

Polling keeps one original command deadline, preserves complete child output,
and uses existing process-group cleanup on timeout or observer failure. Sensor
reads can add overhead or contend with the child; no zero-overhead claim is made.
Observed campaigns must freeze the same harness on both revisions and cannot be
pooled with older unobserved campaigns. This is diagnostic context only: during-run
**competing-load validation**, public-suite integration and #96 performance
acceptance remain unfinished. Neither thresholds nor publication gates are relaxed.

### Crate-owned component correctness smoke

Use the lib-test executable discovered by Cargo preparation (not an ordinary
`--bin` target):

```bash
uv run python scripts/run_small_work.py --component-probe \
  --artifact-kind lib-test --case-binary "$PROBE_BINARY" \
  --correctness-only --output /tmp/component-correctness.json
```

This exports the probe's contracts and checks every exported case/stage, saving
`run.json` and child output under `data/results/<target_profile>/cpu/small_work_components/<timestamp>/`.
It does not collect timings or update latest reports. Without a preparation
receipt, provenance is explicitly unverified, even if correctness passes. To
verify provenance, pass `--preparation-receipt` with the matching
`--tenferro-dir`, `--build-project`, `--cargo-package`, and `--cargo-target-name`.
Add `--allocation-iterations 2` to run a separate allocation child for each
exported case/stage after correctness succeeds. The result is
`ALLOCATION_DIAGNOSTIC`, not timing acceptance. Counters cover caller-thread
allocation calls and requested-byte traffic, excluding fixture setup and
worker/provider/native allocations; they do not report retained memory or latency.
For a component latency diagnostic, omit `--correctness-only`, pass `--suite
benchmarks/cpu/small_work.yaml` for its sampling/resource protocol, and supply the
verified release lib-test receipt and its package/target selectors. The component
case/stage list comes from the library export, not the public suite's case list.
Calibration children and predeclared warmups are archived but excluded from the
reported measured samples; independent processes use the same calibrated iteration
count. Source/artifact and resource checks run before and after collection, and
affinity is restored even on failure. Excess process variability or unavailable
resources are INCONCLUSIVE. Successful collection is TIMING_DIAGNOSTIC, not an
accepted baseline/candidate comparison; no latest table is published. Release
readiness and an actual accepted comparison remain separate requirements.

### Small-work dry-run

Preview the ordinary suite's selected cases and current resource eligibility without
building or executing a benchmark binary:

```bash
uv run python scripts/run_small_work.py --suite benchmarks/cpu/small_work.yaml \
  --dry-run --observation-window 2 --output /tmp/small-work-preview.json
```

This uses the same `SMALL_WORK_CASE_FILTER`, suite `resource_policy`, and optional
`--cpuset` as collection. Alternatively, repeat `--changed-path` with library-relative
paths to reuse the selected library's canonical dependency selector:

```bash
uv run python scripts/run_small_work.py --suite benchmarks/cpu/small_work.yaml \
  --dry-run --changed-path crates/tenferro-einsum/src/concrete.rs
```

This runs the existing inventory checker, not a benchmark, against a clean
`--tenferro-dir` (default `extern/tenferro-rs`). It selects **all** suite variants
of every selected canonical ID and archives `change_selection.json` with the
source identity and missing contract IDs. Do not combine changed paths with
`SMALL_WORK_CASE_FILTER`. Missing contracts or an empty selection are INCONCLUSIVE;
normal collection refuses to execute a partial matrix. Changed-path runs remain
selected-case archives and cannot replace latest reports. This is a per-checkout
selector, not yet a baseline/candidate comparison command.

The JSON lists case/contract IDs and resource reasons;
unavailable or unsupported resources remain INCONCLUSIVE. No affinity settings
are changed, no latest report is published, and no sudo is required. JSON is printed
and archived under `--results-root`; `--output` optionally writes another copy.
A preview does not validate canonical binding, artifact provenance, correctness,
allocation or timing and is never a readiness result. Do not combine `--dry-run`
with preparation, component, correctness-only, fixture or command execution modes.

### Measured collection

Measured collection requires a valid live idle-core resource selection and a
Cargo-owned preparation receipt. Prepare the binary explicitly before timing:

```bash
uv run python scripts/run_small_work.py --prepare \
  --case-binary target/release/small_work_case --profile release \
  --features cpu-faer --output /tmp/small-work-preparation.json
```

Pass that receipt with `--preparation-receipt` to measured collection. When
used together with `--prepare`, the same existing option supplies prior proof
for a cached (`fresh: true`) artifact; omit it and cached preparation fails
closed. The receipt records Cargo JSON compiler-artifact/build-finished evidence, source and
lock/config/toolchain hashes, resolved dependency paths/features (including
foreign mutable path checkouts), and the executable SHA-256. The driver
re-collects and compares those identities before launch and promotion; cached
`fresh: true` artifacts require a matching verified prior receipt. Relevant
compiler/config overrides are captured without storing their contents and
unsupported overrides fail closed. Correctness-only mode may run with unverified
provenance but never promotes timing evidence. Synthetic fixtures are never
preparation evidence.
