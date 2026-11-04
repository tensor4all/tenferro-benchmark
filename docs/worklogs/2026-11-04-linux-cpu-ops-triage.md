# Linux CPU nightly `cpu_ops` triage

Date: 2026-11-04
Run triaged: [41687773887](https://github.com/tensor4all/tenferro-benchmark/actions/runs/41687773887) (2026-11-03, `main` @ `0baddde`)
Cross-checked: 41494612805 (11-02), 41333350090 (11-01), 41259861008 (11-01 dispatch), 41101626722 (10-30), 41226254979 (10-31), 40975831961 (10-29, the last green run)

This document only records what happened. No code changed with it.

## What the job actually reports, and in which order

The failing step prints two different errors, and only the second one names
`cpu_ops`:

1. `Case status t1: executed=228 failed=94 skipped=0 missing=0` — and the same
   `failed=94` at `t4`. The 94 rows are the CPU `perf_issues` suite.
2. `CPU collection failed at line 16: ./scripts/run_all.sh 1 4` — `run_perf_issues.sh`
   ends with `exit "$failed"`, so 94 failed rows give the suite exit 1, and
   `run_all.sh` runs under `set -euo pipefail`.
3. `ValueError: CPU workload selection does not match declared nightly inventory: cpu/cpu_ops`
   from `scripts/ci/cpu_ingestion_adapter.py:52`, after the collection step already
   failed.

So `cpu_ops` in the final message is the *packaging* consequence, not the first
failure: `cpu_ops` sorts before `einsum` in `sorted(identities(...))`, so the
adapter names it first when the run record does not match the declared
inventory. The 188 `failed` lines the run log shows are 94 cases × 2 thread
counts, all from `perf_issues`.

The same final `ValueError` appears on the mac nightly, where the collection dies
earlier for an unrelated reason (empty-array expansion in `scripts/thread_env.sh`
on bash 3.2). The shared final message is not evidence of a shared cause.

## The 94 failing cases

All 94 have exactly one recorded error, and it is the same for every arm:

```text
FileNotFoundError(2, 'No such file or directory')
```

The set is stable from 10-30 (the first nightly after #286 put `perf_issues` into
`run_all.sh`) through 11-03, identical at `t1` and `t4`:

| nightly | `perf_issues` counts (t1) | failing set |
|---|---|---|
| 20261008_063605 (committed record) | expected 153, failed 0 | — |
| 20261031_190716 | expected 322, executed 228, failed 94 | same 94 |
| 20261103 (this run) | expected 322, executed 228, failed 94 | same 94 |

### Root cause

`scripts/cpu_followup_cases.py` runs the independent Rust MWE through

```python
os.environ.get('CPU_FOLLOWUP_BIN', str(ROOT/'target/followup-current/release/cpu-followup-mwe'))
```

and only `scripts/run_cpu_followup.sh --prepare` builds that binary
(`CARGO_TARGET_DIR=$PWD/target/followup-current`, `mwe/cpu_followup`). Nothing on
the nightly path runs `--prepare`: `run_all.sh` calls `run_perf_issues.sh` at
lines 107-109 and 431-433, and neither the workflow nor the devcontainer build
step builds `mwe/cpu_followup`. So the binary is absent, `subprocess.run` raises
`FileNotFoundError`, and `collect()` records it as the row's `error` verbatim.

The same error reproduces on this Mac against the current `main`:

```text
$ python3 -c 'import cpu_followup_cases as c; c.execute("activation.erf","eager-shared",1,0,2000000)'
FileNotFoundError(2, 'No such file or directory')
```

### Why the reference arms fail too

`run()` in `cpu_followup_cases.py` executes the reference arm
(`reference_path`) first, then the Rust arm (`rust_path`), then the arm under
collection. For every one of these cases `rust_path` is `eager-shared` or
`direct-shared`, both of which resolve to the missing binary, so the row records
that error. On a host where the reference arm cannot run either, its own error is
what the row records instead (verified on a host without numpy: the `jax`
reference fails first with `ModuleNotFoundError`). That is why the failing set
spans all arms, and it is *not* evidence that PyTorch/JAX/Julia are broken.

Verified separately: the PyTorch reference runs fine on this machine.

```text
$ .venv/bin/python mwe/cpu_followup/reference.py activation.erf pytorch 4 0
{"case_id": "activation.erf", "path": "pytorch-eager", "threads": 4, ...}   # exit 0
```

The 10 `activation_*` rows are in the set for the same reason: those cases carry
`followup_mwe` with `rust_path: eager-shared`, so `collect()` routes them through
`cpu_followup_cases.run()` too. The `_session-single-call` and `_eager-composed`
arms of the same activation cases pass because they use the `perf_issue_case`
binary that `run_perf_issues.sh` does build.

## Classification table

Every row below is one of the 94, at both thread counts. `arm runner` is the
executable the arm's branch of `collect()` reaches first.

| case | arm | arm runner | recorded error | class |
|---|---|---|---|---|
| `activation_erf_f32_eager-single-call_1024x64` | eager-single-call | cpu-followup-mwe (Rust) | `FileNotFoundError(2, 'No such file or directory')` | arm definition points at an unbuilt binary |
| `activation_erf_f32_pytorch_1024x64` | pytorch | cpu-followup-mwe (Rust), via `rust_path` | same | same, reference arm blocked by the Rust precondition |
| `activation_sigmoid_f32_eager-single-call_1024x64` | eager-single-call | cpu-followup-mwe | same | same |
| `activation_sigmoid_f32_pytorch_1024x64` | pytorch | cpu-followup-mwe via `rust_path` | same | same |
| `activation_silu_f32_eager-single-call_1024x64` | eager-single-call | cpu-followup-mwe | same | same |
| `activation_silu_f32_pytorch_1024x64` | pytorch | cpu-followup-mwe via `rust_path` | same | same |
| `activation_softplus_f32_eager-single-call_1024x64` | eager-single-call | cpu-followup-mwe | same | same |
| `activation_softplus_f32_pytorch_1024x64` | pytorch | cpu-followup-mwe via `rust_path` | same | same |
| `activation_gelu_tanh_f32_eager-single-call_1024x64` | eager-single-call | cpu-followup-mwe | same | same |
| `activation_gelu_tanh_f32_pytorch_1024x64` | pytorch | cpu-followup-mwe via `rust_path` | same | same |
| `followup_real_tanh_direct-shared` / `_jax` | direct-shared / jax | cpu-followup-mwe | same | same |
| `followup_real_log1p_direct-shared` / `_pytorch` | direct-shared / pytorch | cpu-followup-mwe | same | same |
| `followup_real_cos_direct-shared` / `_pytorch` | direct-shared / pytorch | cpu-followup-mwe | same | same |
| `followup_real_sin_direct-shared` / `_pytorch` | direct-shared / pytorch | cpu-followup-mwe | same | same |
| `followup_real_exp_direct-shared` / `_jax` | direct-shared / jax | cpu-followup-mwe | same | same |
| `followup_real_expm1_direct-shared` / `_jax` | direct-shared / jax | cpu-followup-mwe | same | same |
| `followup_real_pow_direct-shared` / `_pytorch` | direct-shared / pytorch | cpu-followup-mwe | same | same |
| `followup_real_log_direct-shared` / `_pytorch` | direct-shared / pytorch | cpu-followup-mwe | same | same |
| `followup_complex_exp_direct-shared` / `_julia-base` | direct-shared / julia-base | cpu-followup-mwe | same | same |
| `followup_real_reduce_{max,min,prod,sum}_*_direct-shared` (8) | direct-shared | cpu-followup-mwe | same | same |
| `followup_real_reduce_{max,min}_*_julia-base` (7) | julia-base | cpu-followup-mwe via `rust_path` | same | same |
| `followup_complex_norm_fro_direct-shared` / `_jax` | direct-shared / jax | cpu-followup-mwe | same | same |
| `followup_index_gather_direct-shared` / `_jax` | direct-shared / jax | cpu-followup-mwe | same | same |
| `followup_index_scatter_direct-shared` / `_pytorch` | direct-shared / pytorch | cpu-followup-mwe | same | same |
| `followup_index_dynamic_update_slice_direct-shared` / `_jax` | direct-shared / jax | cpu-followup-mwe | same | same |
| `followup_structural_{broadcast_in_dim,transpose,extract_diagonal}_direct-shared` / `_pytorch`, `_jax` | direct-shared / reference | cpu-followup-mwe | same | same |
| `followup_perm_*_materialize` (8) | materialize | cpu-followup-mwe | same | same |
| `followup_perm_*_strided-rs` (8) | strided-rs | cpu-followup-mwe | same | same |
| `followup_fft_{rfft,irfft,fft,ifft}_direct-shared` (4) | direct-shared | cpu-followup-mwe | same | same |
| `followup_fft_{rfft,irfft,fft,ifft}_mkl-dfti` (4) | mkl-dfti | cpu-followup-mwe via `rust_path` | same | same |
| `followup_metadata_{transpose,slice,reshape}_view_metadata` (3) | metadata | cpu-followup-mwe | same | same |
| `followup_metadata_*_metadata-host-{dyn,static}` (6) | metadata-host-dyn / -static | cpu-followup-mwe | same | same |
| `followup_metadata_*_julia-base` (3) | julia-base | cpu-followup-mwe via `rust_path` | same | same |

By backend: `tenferro-rs` 50, `pytorch-cpu` 15, `julia-base` 10, `strided-rs` 8,
`jax-cpu` 7, `mkl-dfti` 4. By arm (`execution_path`): `direct-shared` 28,
`pytorch` 15, `julia-base` 10, `materialize` 8, `strided-rs` 8, `jax-compiled` 7,
`eager-shared` 5, `mkl-dfti-cached` 4, `metadata` 3, `metadata-host-dyn` 3,
`metadata-host-static` 3.

Class summary: **one class, 94/94 rows, zero rows in any other class.** No row is
an environment difference, a reference-script incompatibility, or a measurement
defect. There is no library regression: the rows never ran a measurement.

Two things this triage cannot decide, because they were never reached:

- whether the `julia-base`, `jax` and `mkl-dfti` reference runners work inside
  the MKL devcontainer;
- whether the Rust MWE builds and validates against the pinned tenferro revision
  under `system-mkl`.

Neither is evidence of a bug; both become observable as soon as the binary
exists.

## The `cpu_ops` inventory mismatch is a second, separate failure

The successful 10-29 run (40975831961) shipped `ingestion.json` with
`cpu/cpu_ops` = **892** expected identities, and its `cpu_ops` CSV held 446 rows
per thread. The 11-03 run's `cpu_ops` CSV holds **496** rows per thread
(212 `cpu_ops` + 284 `grad_*` linalg rows), i.e. 992 identities against the
declared 892, and 568 linalg identities against the declared 300.

So the packaging rejection is not purely "the suite is missing because the
collection aborted": the row set this run produced no longer matches
`benchmarks/cpu/ingestion-selection.json`, which is a checked-in file with no
generator in the repository. Which of these is the trigger is unproven here,
because packaging failed before writing `ingestion.json` for this run and the
artifact therefore has no bundle document to compare. It needs its own
investigation; it should not be folded into the `perf_issues` fix.

## Can one PR fix it?

The 94 rows: yes. The change is small and touches three places:

1. `scripts/run_perf_issues.sh` (or `run_all.sh` before its `perf_issues` call):
   build `mwe/cpu_followup` into `target/followup-current/release/` when the
   `perf_issues` coverage includes `cpu_followup_mwe` cases, with the same
   `CARGO_TARGET_DIR` that `scripts/cpu_followup_cases.py` defaults to. Reusing
   `run_cpu_followup.sh --prepare` keeps one build path instead of adding a
   second.
2. `scripts/cpu_followup_cases.py`: report a missing binary as a named,
   actionable error (the command it tried and how to build it) instead of the
   bare `FileNotFoundError`, so the next occurrence is not a 94-row mystery.
3. A test that the collector's default binary path is the one `--prepare`
   produces (`tests/test_cpu_followup_run_record.sh` already builds both halves
   separately; it asserts the path, not that the nightly reaches it).

Estimate: under ~60 lines of shell/Python plus one test, no schema or report
change. The second failure (the `cpu_ops` inventory) does not belong in it.

The mac nightly is affected by the same 94 rows: `plan()` in
`scripts/benchmark_perf_issues.py` does not filter on the case `profiles` field,
and `run_all.sh` runs `perf_issues` on macOS too, where `mwe/cpu_followup`'s
default `system-mkl` feature cannot build. Fixing the bash 3.2 array expansion
alone therefore does not make the mac nightly green — it moves the failure to
this one.
