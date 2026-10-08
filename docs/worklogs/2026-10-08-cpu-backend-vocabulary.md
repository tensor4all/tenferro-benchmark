# CPU backend vocabulary migration (tenferro-rs #2004 consumer)

Record of the change that moves this harness onto the tenferro-rs revision that
contains #2004 (upstream PR #2026, merge commit `b3f472962`). The harness could
not even resolve dependencies against that revision before this change: its
`tenferro-cpu-tprims` path dependency points at a crate #2004 deleted.

## Upstream contract this harness now targets

- A build compiles exactly one CPU backend: `native` (default, faer kernels and
  tenferro's thread pool) or `blas` (cpueinsum contractions and tlinalg linear
  algebra on a vendor BLAS/LAPACK). Enabling both is an upstream error.
- Vendor selections are `blas-openblas`, `blas-accelerate`, `blas-mkl`.
- `CpuBackendKind`, `CpuBackend::with_kind`, `CpuBackend::with_threads_and_kind`,
  `CpuBackend::execution_info()`, the provider bundle/injection API and the
  batch-policy API are gone; `tenferro_cpu::cpu_provider_id()` reports the
  compiled backend.

## Vocabulary mapping

| before | after | forwards to |
| --- | --- | --- |
| `default = ["cpu-faer"]` | `default = ["native"]` | `tenferro-{ad,einsum,fft,linalg,runtime}/native`, `tenferro-cpu/native` |
| `cpu-faer` | `native` | same |
| `cpu-blas` | `blas` | `.../blas` on the same crates |
| `system-openblas` | `blas-openblas` | `blas` + the installed OpenBLAS link |
| `system-accelerate` | `blas-accelerate` | `blas` + the Accelerate framework link |
| `system-mkl` | `blas-mkl` | `blas` + the installed oneMKL link |

`TENFERRO_CPU_FEATURES` takes `native`, `blas-openblas`, `blas-accelerate` or
`blas-mkl`. `TENFERRO_CPU_BACKEND_KIND` is gone: there is nothing to select at
runtime, and the compiled provider id (or its short `faer`/`blas` form from
`tenferro_einsum_benchmark::compiled_cpu_provider()`) is recorded instead of an
environment echo. Every vendor-selecting cargo invocation passes
`--no-default-features`, otherwise `native` and `blas` would both be enabled.

## Decisions

1. **Linking stays consumer-owned.** The vendor features select upstream's
   provider-neutral `blas` feature and link the *installed* library through the
   existing `build.rs` (`OPENBLAS_ROOT`, `MKLROOT`, Accelerate framework). They
   deliberately do not enable upstream `provider-src`, which would build and
   link its own library copy and break the provider-matched comparison the
   AGENTS.md CPU policy requires. Verified: the BLAS build's `ldd` resolves
   `/lib/x86_64-linux-gnu/libopenblas.so.0`, and the `native` build links no
   vendor BLAS.
2. **Capabilities that #2004 removed are retired here, not reimplemented
   downstream.** `src/cpu_provider.rs` and the `tprims` feature/dependency (the
   provider-injection comparison and its shape corpus), the provider-spy
   `cpu_route_diagnostic` binary, the policy half of
   `src/bench_support/batch_route.rs`, `scripts/run_route_diagnostic.sh` and the
   `scripts/route_contract.py` verdict tooling are gone. A `policy` override and
   a requested provider the build does not contain now report `unsupported`
   instead of being silently measured on the wrong backend.
3. **No derived backend-kind metadata.** The compiled provider id and the
   recorded cargo feature set are the authorities for what a run measured.

## Files

- `Cargo.toml`, `build.rs`: feature map, vendor mutual-exclusion check, link
  configuration, and the surviving cross-revision probe
  (`tenferro_session_einsum`; the lane-cost probe is gone with `batch_policy.rs`).
- Rust: `src/lib.rs`, `src/main.rs`, `src/bench_support/batch_route.rs`,
  `src/bin/{benchmark_cpu_fft,benchmark_cpu_public_api,benchmark_cpu_session,cpu_matmul_diagnostic,publication_gate,small_work_case}.rs`,
  `examples/cpu_gap_mwe.rs`.
- Scripts, CI, devcontainer, docs, shell/python tests: the vocabulary sweep, with
  `scripts/format_cpu_ops_results.py` still normalizing the *historical* labels
  (`cpu-faer` / `system-*`) so older raw runs under `data/results/**` re-format
  unchanged.
- `.github/workflows/ci.yml`: the recorded compatible revision moves from
  `5cf78c7ec` (pre-#2004) to `b3f472962` (this migration's target).

## Review round (independent post-review of the finished diff)

The finished diff was reviewed independently before the PR. Findings that were
fixed:

- A batched/stream `--case` run declared `providers: ["faer"]` in the case
  manifest but timed whatever backend the build had, so a BLAS build produced a
  passed measurement under a `[faer]` case key. The runner now takes the case's
  declared provider and reports `unsupported` when the build does not contain it
  (verified in both directions: a `native` build asked for `blas` and a
  `blas-openblas` build asked for `faer` both emit the unsupported row and
  measure nothing).
- `detect_regressions.py` lost the comparison-safety checks with the route
  contract: it compares shared rows' `provider` and `worker_count` again, and
  `confirm` rejects arms whose rounds recorded different CPU features.
- `run_cpu_ops.sh` now defaults `PUBLICATION_GATE_FEATURES` to the run's
  `TENFERRO_CPU_FEATURES` (an explicit override still wins), so a run cannot
  describe its CPU-ops rows with a provider other than the one it selected.
- `run_publication_gate.sh` validates the feature name once instead of five
  identical cargo arms, and the unsupported-provider message names a selection
  that exists (`native` / a concrete `blas-*` feature).
- The `#1946 B6` sequence in `docs/regression-detection.md` is marked historical
  with the older harness requirement stated, because those revisions predate the
  feature names the current harness has.

Dismissed with reason: the duplicate `-lopenblas` link argument in `build.rs`
(is harmless, and the adjacent comment records why the bin-specific argument
exists), the duplicated `dot_general_read` diagnostic rows (pre-existing), and
retiring the `probe_tenferro_apis`/`src/tensornetwork.rs` cross-revision
fallback (still the harness's only revision-adaptive switch; a separate cleanup).

## Verification (no timing collection)

- `cargo check -j 16 --all-targets` — clean (`native`).
- `OPENBLAS_ROOT=/usr cargo check -j 16 --all-targets --no-default-features --features blas-openblas` — clean.
- `MKLROOT=<stub> cargo check --all-targets --no-default-features --features blas-mkl` —
  type-check only; no oneMKL is installed on this host, so MKL linking is unverified.
- `cargo build --bins --examples` for `native` and `blas-openblas`, plus `ldd`.
- `cargo test --lib` and `cargo test --bins` (43 tests) in both real lanes.
- The non-timing shell suites under `tests/` and the three detector python
  suites; the CI `validate` steps reproduced locally (15 suite YAMLs,
  `py_compile`, layout test).
- Not run: any `run_all.sh` / suite collection (no timing data was produced).

Pre-existing, unchanged by this work: `tests/test_instance_colmajor_metadata.sh`
fails on the list-shaped `data/instances/session_matrix.json`, and
`tests/test_mac_gpu_permutation_profile.sh` needs PyYAML in the plain `python3`
environment.

## Follow-ups

- `tensor4all-rs` consumer migration (pin plus feature forwarding).
- Remove the temporary `cpu-faer` alias in tenferro-rs
  `docs/tutorial-code/Cargo.toml` now that the default-branch workflows use
  `cuda,native`.
- Restoring the tprims provider comparison needs a supported provider seam
  upstream; it cannot be rebuilt from this harness.
