# Routine GPU wall clock on the A100: what was removed, what is left

Companion to [../gpu-routine-timing.md](../gpu-routine-timing.md), which carries
the measured tables (per-suite cold and warm walls, coverage, compile totals, the
reconstruction note). This handoff records the change, the evidence behind it, and
the conclusion that the remaining wall clock does not need optimizing.

## The change

`scripts/run_gpu_suite.sh` exported no `TENFERRO_CUDA_ARTIFACT_CACHE`, so every
case of every tenferro arm opened its engine or traced runtime
(`src/bin/benchmark_gpu_rust.rs:1215` and `:1219` for the eager arm, `:1674` and
`:1686` for the traced arm) and compiled its modules into process memory from
cold, charged to `input_generation_transfer_reference_and_preparation` (`:1312`)
and to the sampler probe (`:656-668`, recorded at `:714`). The public-API
collector had already been given a persistent cache directory (#350); the routine
suites had not.

The discovery now lives once, in `scripts/gpu_preflight.sh` as
`configure_cuda_artifact_cache`, called by `configure_gpu_bench_env` and by
`scripts/run_gpu_suite.sh`, so the routine, AD, throughput and permutation
collectors share one directory and one code path.

Nothing about the measurement changed: no sample count, warmup count, timing
scope, validation threshold or phase definition, and no timer or dependency was
added. Compilation stays outside every timed region in both regimes.

Measured on the A100, same host, same binary, same profile, 698 of 714 rows: the
summed per-suite wall went from 3967.3 s (cache disabled) to 897.9 s (cache warm),
and a first, partly cold pass measured 1010.6 s. Per-case medians did not move
outside run-to-run noise: comparing two runs under identical warm conditions, the
tenferro arms move by median 0.6 % with four cases beyond 10 %; before/after they
move by median 0.6 % with two cases beyond 10 %.

## What is left, and why it should be left alone

The warm run's 897.9 s, partitioned without overlaps:

| Item | Seconds |
|---|---:|
| Recorded case walls, of which | 458.1 |
| — the timed regions themselves | 281.6 |
| — untimed preparation around each case | 86.3 |
| — JAX per-case jit (one distinct shape per case) | 45.1 |
| — validation, probe, warmup, calibration, backend init, sampler overhead | 45.1 |
| Per-suite fixture generation and copy calibration | 114.4 |
| `permutation` arms (their records carry medians only, no case wall) | 154.2 |
| `linalg_ad_latency` and `linalg_jvp_vjp` (no case wall recorded) | 66.6 |
| Backend-release sleeps, deliberate (`GPU_BENCH_BACKEND_SLEEP=2`) | 34.0 |
| Process startup and imports, per-suite glue, accounting, report writes | 70.6 |
| **Total** | **897.9** |

That is a harness whose remaining cost is the measurement and deliberate waiting:
281.6 s of timed regions it exists to report, one fixture generation per suite
that every arm then reads, a per-case JIT no cache can share, and a 2 s
allocator-release margin between framework processes. **There is nothing further
worth optimizing here.** The one lever worth pulling — the cold compile — is
pulled, and the next reader should not spend a day looking for another in fixtures
or process startup: both are unchanged between the two regimes (fixture generation
117.8 s cold vs 114.7 s warm, measured per suite from its fixture manifest
timestamp; startup and report glue ~34 s vs ~29 s).

## What the untimed preparation contains

Measured with throwaway timers that were removed before the change was committed,
warm cache, three cases (seconds):

| Case | Arm | Generation | H2D | CPU reference | Plan |
|---|---|---:|---:|---:|---:|
| `dense_batched_matmul_f64_b16_32` | eager | 0.000 | 0.011 | 0.000 | 0.010 |
| `dense_matmul_f64_3072` | eager | 0.164 | 0.082 | 0.090 | 0.015 |
| `dense_qr_f64_1536` | eager | 0.022 | 0.024 | 0.012 | 0.010 |
| `dense_matmul_f64_3072` | traced | 0.237 (graph) | in graph | — | < 0.001 |
| `dense_matmul_f64_3072` | torch | 0.360 | 0.009 | 0.113 | 0.000 |
| `dense_matmul_f64_3072` | jax | 0.365 | 0.091 | 0.162 | 0.711 (jit) |

The routine suites generate their inputs from a seed; there is no fixture file to
read, so "fixture read" is zero here and fixture generation for the public-API
families is the 114.4 s above.

## Duplicated work: what was checked

- **Compiled per case, per arm.** Fixed, as above.
- **The same fixture generated more than once.** Yes: every arm regenerates the
  same seeded inputs in its own process, the two Python arms each recompute the
  same double-precision reference (`scripts/gpu_framework_runner.py`, the
  `_to_*_cpu` branch), and the Rust eager arm builds its CPU reference inputs a
  second time (`src/bin/benchmark_gpu_rust.rs:1815` and `:1925`). It is now worth
  0.02 to 1.7 s per case across all four arms. Sharing it needs materialised
  fixtures and references for the routine families too — the mechanism the
  public-API families already use, with `fixture_ms` of 0.01 to 0.06 s per case —
  and that is a design change to how a routine arm obtains its inputs, not a
  duplicate-work fix. Left as a decision; the measured ceiling is small.
- **The same output downloaded twice.** No: `src/bin/benchmark_gpu_rust.rs:2471-2481`
  downloads each output once and compares that host tensor, and the public-API
  verifier fingerprints the one download it takes (`fingerprint_compares` counts
  passes over it, not extra transfers).
- **Validation recomputing what the sampler had.** No: the CPU reference is the
  reference and is computed once per case, after sampling.
- **A backend re-initialised per case.** The engine and runtime are still opened
  per case, deliberately: with a warm cache that costs 0.02 to 0.37 s and keeps a
  case's lazy initialization inside its own untimed window.

## Reproducing

Raw runs stay on the collection host:
`~/work/wallclock/data/results/nvidia-gpu/gpu/<suite>/<stamp>/` (the host is
reclaiming disk, so these may disappear), with the durable copy of the records,
reports and metadata at `~/work/wallclock-results/{20261105_0820,0920,0940}/`.
The cold run's records, metadata and reports for `20261105_0820` are also
committed under `data/results/nvidia-gpu/gpu/`. The 4.08 GB of generated fixtures
and sparse caches are regenerable and are not kept.
