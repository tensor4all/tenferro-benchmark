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
