# M5 shared-session benchmark refresh

Benchmark code `38550899679f8d2c1972b4055caa5eb939054c26` implements the
[short-operation sampling contract](short-operation-sessions.md) on tenferro
`a793c2e95693f053722fbff8d0db25722336c21f`.

The code was pushed before collection. CPU runs use native macOS on Apple M5
Max. Inputs, session entry, graph/runtime preparation and compilation are outside
timing; output lifetimes extend past timer stop. Accelerate is the normal BLAS
provider, with faer explicitly selected as an additional session-matrix reference.
JAX now receives `PJRT_NPROC` as well as the existing library thread settings.

Collection command, with the Julia executable on PATH:

```bash
export PATH="/Users/hiroshi/.juliaup/bin:$PATH"
export BENCHMARK_COMMIT=38550899679f8d2c1972b4055caa5eb939054c26
BENCHMARK_TARGET_PROFILE=mac-cpu PUBLICATION_GATE_PROFILE=full \
  FFT_BENCH_LENGTHS=1024,16384,1048576 RUN_PERMUTATION_SUITE=0 \
  ./scripts/run_all.sh 1 4
```

Timing is sequential. The previously collected permutation suite is unchanged.
The standard linalg JVP/VJP report now contains the larger-matrix cases.
Old isolated small-matrix rows remain only in historical raw diagnostics;
they are not current shared-session results. Ordinary eager/trace APIs
cannot accept the borrowed shared session, so unsupported short routes are
explicit diagnostics rather than being relabeled as shared-session measurements.

Validation before collection: 25 CPU Rust unit tests across the affected runners,
6 timing/lifetime tests, 10 public-API harness checks, 3 small-work harness checks,
90 CPU linalg AD integration rows, thread metadata and suite/report contracts.
CUDA trace runners compile with the prepared execution path; GPU timings were
not collected in this refresh.

The Julia view correction and explicit provider execution-mode metadata use
benchmark code `4c63ae9`. The saved
[follow-up collection script](../data/results/mac-cpu/cpu/public_api/20260913_155000/collection.sh)
recollects only the three Julia views per thread, then the session-matrix suite.
The public-API report combines unchanged full-run rows with these six replacement
rows; its [source manifest](../data/results/mac-cpu/cpu/public_api/20260913_155000/source_manifest.json)
records the inputs and override order. Original measurements are preserved.

Generated results and raw runs:

- [Einsum](../result/mac-cpu/cpu/einsum.md), [CPU ops](../result/mac-cpu/cpu/cpu_ops.md),
  [large-matrix JVP/VJP](../result/mac-cpu/cpu/linalg_jvp_vjp.md):
  [1 thread](../data/results/mac-cpu/cpu/einsum/20260913_152441/),
  [4 threads](../data/results/mac-cpu/cpu/einsum/20260913_153332/).
  CPU ops: 666 successful rows at each thread count.
- [FFT](../result/mac-cpu/cpu/fft.md):
  [raw run](../data/results/mac-cpu/cpu/fft/20260913_153932/).
  64 successful and 80 explicitly skipped rows at each thread count; short
  non-shared routes are among the skips.
- [Public API](../result/mac-cpu/cpu/public_api.md):
  [full run](../data/results/mac-cpu/cpu/public_api/20260913_154011/),
  [Julia view correction](../data/results/mac-cpu/cpu/public_api/20260913_155000/).
  429 successful and 18 unsupported rows at each thread count.
- [Small work](../result/mac-cpu/cpu/small_work.md):
  [raw run](../data/results/mac-cpu/cpu/small_work/20260913_064413/).
  All 25 shared-session cases passed at both thread counts.
- [Session matrices](../result/mac-cpu/cpu/session_matrix.md):
  [raw run with execution modes](../data/results/mac-cpu/cpu/session_matrix/20260913_065008/).
  All 60 case/provider/thread combinations passed full-output or residual checks.
  The preceding [run](../data/results/mac-cpu/cpu/session_matrix/20260913_064429/)
  is also retained; it predates recording the runtime execution-mode field.

The session-matrix results expose a tenferro issue: shared BLAS sessions retain
`ProviderDefaultExclusive` mode and still enter the executor per operation.
Faer uses the already-entered context. These results are the baseline for a
separate tenferro-rs correction; the benchmark does not bypass that behavior.
