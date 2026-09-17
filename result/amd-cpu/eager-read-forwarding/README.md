# Eager borrowed-read forwarding: 1T instruction evidence

Candidate `2fd8163e568a232072348e8939d534713f3b3a98` removes blanket input
materialization before existing standard-operation `_read` hooks. Owned-only
operations and dtype promotion retain their explicit boundaries. CPU kernels,
SIMD code, and the linked BLAS overwrite implementation are unchanged.

## Protocol and result

`protocol.json` was written before collection. Three complete independent
baseline/candidate N1/N3 pairs were collected for every case, sequentially,
with one fixed warmup. All 36 invocations completed successfully. The metric
is whole-process Callgrind Ir differencing for warmed eager operations, **not
kernel-only cost, wall-clock time, bandwidth, or a native speedup estimate**.
Fixed preparation cancels; per-repeat harness work remains. Identical binary
paths share a cached measurement, hence divisor 2. LM's two distinct paths give
divisor 4 and a path mean, not an individual-path result.

| Workload | Baseline median Ir | Candidate median Ir | Paired reduction range |
|---|---:|---:|---:|
| multiply 2048² | 11,106,156.5 | 5,309,045 | 52.180–52.215% |
| GEMM1024 control | 524,967,422.5 | 523,500,184 | 0.278–0.281% |
| LM3 12d path mean control | 968,053,178.5 | 950,507,132.5 | 1.812–1.814% |

The predeclared primary ≥20% reduction and both ≤1% regression controls pass.
Vendor `dgemm_kernel_HASWELL` self-Ir is identical in every baseline/candidate
pair: 512,386,632 for GEMM1024 and 444,650,842 for LM. This does not establish
unchanged wall-clock kernel cost.

For multiply, two 32 MiB input payload copies are eliminated. Their 128 MiB
combined logical read/write traffic is arithmetic accounting, not measured
DRAM traffic. A recording regression observes two baseline materializations
versus zero after this change, with nonzero transposed inputs and unchanged
source values. Canonical packing still dominates remaining LM copy work; this
change does not claim to remove all eager or extension-boundary copies.

## Provenance and reproduction

- Baseline tenferro: `57bee76de1a752782bdd985f332050422f885372` (the prior
  BLAS-full-overwrite candidate, **not** its older baseline).
- Candidate tenferro: `2fd8163e568a232072348e8939d534713f3b3a98`.
- Both use strided `17e05ffb168d0f826529ea2c3aceeb4ec851448b`, with local
  overrides; benchmark executable source is `976c4b9a459f94086c714ec0f6836978a7bbdfc0`.
- Existing user-authorized OpenBLAS Docker image, Rust 1.98.1 / LLVM 22.1.8,
  OpenBLAS 0.3.26 LP64 pthreads, Valgrind 3.22.0, CPU16 (L3 CPUs16–23).
  The build and analysis image IDs match. Image/library/binary hashes, thread
  settings and host snapshots are in `raw/environment.txt`.
- `configure_cpu_thread_env 1`, `TENFERRO_MODE=eager`, BLAS selected,
  decomposer disabled. A separate diagnostic intercepted GEMM1024 calls and
  verified `openblas_get_num_threads()==1` for both binaries. The interceptor
  from `../blas-full-overwrite/verify_threads.c` was absent from measurements.
- The host still had active Julia workloads. No quiet-host native comparison
  was attempted, and profiler-emitted time columns must not be used as one.

Build each revision from the benchmark checkout with its matching local
strided override, preserving the baseline binary before switching source:

```sh
RUSTC_WRAPPER= CARGO_PROFILE_RELEASE_DEBUG=1 cargo build -j 16 \
  --config /tmp/local-patch.toml --release --no-default-features \
  --features system-openblas --bin tenferro-einsum-benchmark
```

Keep the embedded benchmark fixture path resolvable. The three fixtures and
Cargo.lock/local patch used here are retained under `raw/`. Inside the Docker
workspace, run `reproduce.sh` with `BASELINE`, `CANDIDATE`, and a fresh `OUT`.
It reproduces the collection loops (original collection used one invocation
per pair). Put its profile/log/annotation outputs in `raw/` for `summarize.py`;
the parser accepts plain or gzip artifacts and checks run/thread metadata.
The script is syntax-checked; its equivalent collection loops were executed.

## Correctness and remaining gates

- 91 debug library tests, 354 functional integration tests, 174 doctests pass.
- Docker release library: 91 tests pass. The focused new regression also passes
  release Memcheck with zero errors; leak checking was disabled.
- Profile fixtures are zero-input execution smoke checks, **not** independent
  numerical validation. Nonzero correctness comes from the regression suites.
- The UI integration wrapper still fails: `--config` dependency patches are
  not inherited by trybuild. Temporary manifest patches resolve dependency
  compilation, revealing two expected-stderr mismatches (underline formatting
  and a kache-remapped path). No unrelated snapshots were blessed; temporary
  manifest edits were restored. Raw failure logs are retained.
- Full repository validation, 4T/AD performance validation, native acceptance,
  and override-free integration remain pending. Publication and dependency-pin
  integration remain deliberately last. This is an optimization candidate,
  not a fully validated release.
