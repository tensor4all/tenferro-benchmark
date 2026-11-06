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

The warm run's 897.9 s, partitioned without overlaps: 458.1 s of recorded case
walls (281.6 s of them the timed regions), 154.2 s of `permutation` arms,
114.4 s of per-suite fixture generation and copy calibration, 66.6 s of the two
AD suites whose records carry no case wall, 34.0 s of deliberate backend-release
sleeps and 70.6 s of process startup and report glue. The totals are the
[budget reference's](../../benchmarks/gpu/routine_budget.yaml) `cache_conditions`
figure; the partition is not republished here, and the warm run directories were
not retained under `data/results/`.

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
warm cache, three cases: several tenths of a second per case for input generation
and H2D, one CPU reference of note (`dense_matmul_f64_3072`, 0.090 s eager and
0.113 s torch) and JAX's per-case jit as the largest single item at 0.711 s. Those
timers left no stored record, so the per-case figures are not republished here;
their window is the untimed case wall that each record still carries as
`overhead.untimed_ms` in the stored [cold
run](../../data/results/nvidia-gpu/gpu/routine_selection/20261105_0820).

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
