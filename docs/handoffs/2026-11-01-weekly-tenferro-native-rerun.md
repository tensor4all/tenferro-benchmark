# 20261030 weekly tenferro: native rerun handoff

## Outcome

The 20261030 weekly tenferro attempt and its first native retry exited nonzero. The relevant followup failures were **environment/runner setup failures, not observed tenferro kernel correctness failures**: the dedicated binary lacked an MKL loader path/RPATH, the Julia default was a devcontainer-only path, and the MKL DFTI reference expected `MKLROOT` while the build environment deliberately unsets it. Harness #289 subsequently added host-path fallbacks. The original failures remain in their records; they are not retroactively changed to passes.

The perf_issues quick/standard rerun is complete **only after substituting the separately measured FFT rows by `(case_id, threads)`**: 898 passed, 0 failed, 0 missing, 898 nonempty sampled rows, 13,470 samples. This is a composite across six raw directories, **not** a single clean collector invocation. The 24 original failed FFT rows remain in the three full-row sources; the scoped FFT sources select only those eight case IDs per thread. Never count both versions or publish the three incomplete full-row sources as independently clean runs.

| threads | full-row source: passed / failed | FFT replacement source: selected / passed / failed / missing | composite |
|---|---|---|---|
| 1 | [`20261031_210328`](../../data/results/zen5-cpu/cpu/perf_issues/20261031_210328/) 314 / 8 | [`20261101_084100`](../../data/results/zen5-cpu/cpu/perf_issues/20261101_084100/) 8 / 8 / 0 / 0 | 322 passed |
| 4 | [`20261031_211155`](../../data/results/zen5-cpu/cpu/perf_issues/20261031_211155/) 292 / 8 | [`20261101_084129`](../../data/results/zen5-cpu/cpu/perf_issues/20261101_084129/) 8 / 8 / 0 / 0 | 300 passed |
| 8 | [`20261031_211858`](../../data/results/zen5-cpu/cpu/perf_issues/20261031_211858/) 268 / 8 | [`20261101_084159`](../../data/results/zen5-cpu/cpu/perf_issues/20261101_084159/) 8 / 8 / 0 / 0 | 276 passed |

The full-row sources measured tenferro-rs `a9f494cc4679387c75af698b299c316fd2295fe4` with harness `5220eed2525b73a328ef426226d299c741c41aa8`; the FFT replacement used the same recorded tenferro checkout and harness `41db23777fb8781bbb7be5c457ac47b9216e2e3e` after #289. The FFT binary was pre-existing, not rebuilt during replacement; its native build provenance cannot be inferred from the checkout metadata alone. Different collection times, harness revisions and setup must remain visible; this is not an A/A or performance-regression comparison. The original native session_matrix run [`20261031_202928`](../../data/results/zen5-cpu/cpu/session_matrix/20261031_202928/) has 99/99 faer rows passed with 15 nonempty samples and the requested 1/4/8 worker counts; it also records 42 unsupported BLAS rows. In that native retry other agents were observed on 12-15, SMT siblings of 0-3, so the session_matrix host-isolation caveat remains.

**Scope limit:** 898/898 refers to perf_issues, not every weekly suite. The native rerun's [`public_api/20261101_044737`](../../data/results/zen5-cpu/cpu/public_api/20261101_044737/) still records one 8T PyTorch `lstsq` failure (rank-deficient matrix); its cause is not established here. The original weekly attempt also reported 1T tenferro `lstsq` failures and 8T PyTorch failure, and overhead_smoke/permutation were not reached. The native `run_all.sh` exited 1 at perf_issues, so those later suites were not run. This raw-evidence PR is not a claim that the entire weekly run passed. No fresh timing was collected for this report.

## Pending

The FFT `MKLROOT` failure is resolved for these recorded cases: the validated collector passed `MKLROOT` **only to the `mkl-dfti` reference subprocess**. The build and collector parents keep it unset; do not export it globally, or the build can pick up the wrong `-liomp5` linkage. The recorded FFT run used the existing `target/followup-current/release/cpu-followup-mwe` and did not build. #289 also supplies an installed-oneAPI fallback, but the run below explicitly scopes the variable and does not rely on it.

Reproducible reference-only command pattern, from this repository, after the normal separate observer pinning and idle gates (the selected case list below is exactly the measured eight):

```bash
export BENCH_INSTANCE=followup_fft_irfft_direct-shared,followup_fft_irfft_mkl-dfti,followup_fft_rfft_direct-shared,followup_fft_rfft_mkl-dfti,followup_fft_fft_direct-shared,followup_fft_fft_mkl-dfti,followup_fft_ifft_direct-shared,followup_fft_ifft_mkl-dfti
# Run one row at a time AFTER configure_cpu_thread_env <n> and the idle gate.
# Each $OUT must be a new empty run directory and N must be 1, 4 or 8.
env -u MKLROOT -u MKL_DIR -u BENCHMARK_ALLOW_BUSY_HOST \
  BENCHMARK_TARGET_PROFILE=zen5-cpu TENFERRO_CPU_FEATURES=native \
  PUBLICATION_GATE_FEATURES=native PUBLICATION_GATE_PROFILE=full \
  BENCHMARK_RAW_ONLY=1 OPENBLAS_ROOT=/usr BENCH_COVERAGE=quick BENCH_EFFORT=standard \
  JULIA=/home/shinaoka/.juliaup/bin/julia \
  LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2026.2/lib:/opt/intel/oneapi/compiler/2026.2/lib \
  .venv/bin/python - "$N" "$OUT" <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import cpu_followup_cases as followup
import benchmark_perf_issues as suite
assert 'MKLROOT' not in os.environ
original = followup.command
def reference_command(case_id, path, threads, runs):
    argv = original(case_id, path, threads, runs)
    return ['/usr/bin/env', 'MKLROOT=/opt/intel/oneapi/mkl/2026.2', *argv] if path == 'mkl-dfti' else argv
followup.command = reference_command
n, output = int(sys.argv[1]), Path(sys.argv[2])
_, cases, _, _, _, selected, _ = suite.plan('cpu', n)
assert len(selected) == 8 and all(cases[c].get('followup_mwe', cases[c]['params']).get('reference_path') == 'mkl-dfti' for c in selected)
raise SystemExit(suite.collect(Path('target/release/perf_issue_case'), output / f'samples_t{n}.jsonl', n))
PY
```

The actual run additionally collected `run_t*.yaml`, `cpu_info.md` and finalized `case_status.json` and `wall_command.json` through the existing repository scripts. Its pre-row measured/observer CPU sets were 0 / 4-11,16-23 (1T), 0-3 / 4-11,16-23 (4T), and 4-11 / 0-3,12-15 (8T); each shared idle gate included SMT siblings. Existing results were checked unchanged by SHA256. No report, summary, Pages index or historical status was rewritten. The `public_api` 8T failure remains a separate unresolved result, not explained away by the followup environment fixes.
