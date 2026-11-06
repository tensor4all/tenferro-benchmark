# GPU routine timing on the A100

The first complete collection of the single routine set
(`benchmarks/gpu/manifests/routine.yaml`: 180 problems, 55 backend arms,
714 planned rows) on the maintained `nvidia-gpu` device. This is the
measurement that replaces the RTX 3060 reference in
[`routine_budget.yaml`](../benchmarks/gpu/routine_budget.yaml).

Raw run: `data/results/nvidia-gpu/gpu/routine_selection/20261105_0820`, with
each suite's records under `data/results/nvidia-gpu/gpu/<suite>/20261105_0820`.

| Field | Value |
|---|---|
| Device | NVIDIA A100 80GB PCIe (`GPU-530977e1-4968-9283-4129-9fbec3e66542`) |
| Driver / CUDA | 580.178.04; runtime and toolkit 12.6 (nvidia-smi reports CUDA 13.0) |
| Harness commit | `18ae0c5` (as recorded in the run metadata) |
| tenferro-rs | `a000e5c4` (pin), features `cuda`, clean |
| Effort / arms | `standard`; trace, eager, PyTorch and JAX, plus the suite-specific AD and permutation arms |
| CUDA artifact cache | `TENFERRO_CUDA_ARTIFACT_CACHE=none`: every process compiled cold; no persistent cubins |
| Routine controls | not selected (`GPU_BENCH_ROUTINE_CONTROLS=0`) |
| Python | torch 2.12.0+cu130; jax/jaxlib 0.10.1 |
| Threads and repetitions | OMP/OPENBLAS/RAYON/MATMUL = 1; `BENCH_WARMUPS=3`, `BENCH_RUNS=15` |

The run was collected on the wall-bottleneck branch at `18ae0c5`. Its only
change relative to main is where the routine path discovers the persistent
artifact cache, and `cache=none` disables that discovery, so this run measures
the same cache-free path as main at `7760de3`.

## Wall clock

The budget's `script_seconds` is the **sum of the suites' own collection
clocks** (`collection.suite_seconds` in each suite's `run.yaml`):
**3967.3 s**. The enclosing command elapsed (15:31-17:18, 107 minutes) is not
used because it includes a 29-minute stall while the Python interpreter was
unavailable (16:14-16:43) and command/report overhead outside the suite clocks.

| Suite | collection s | rows / expected |
|---|---:|---:|
| dense | 653.544 | 28 / 28 |
| einsum | 69.085 | 4 / 4 |
| sparse | 74.244 | 2 / 2 |
| elementwise | 203.924 | 8 / 8 |
| linalg_ad_latency | 648.795 | 30 / 30 |
| linalg_jvp_vjp | 642.107 | 30 / 30 |
| permutation | 262.402 | 44 / 60 |
| lstsq | 71.369 | 4 / 4 |
| linalg_public_api_throughput | 196.441 | 64 / 64 |
| fft | 74.808 | 16 / 16 |
| nn | 156.629 | 44 / 44 |
| structural | 291.558 | 64 / 64 |
| isolated_ops | 215.226 | 132 / 132 |
| reuse_contraction | 230.612 | 224 / 224 |
| tensornetwork | 176.540 | 4 / 4 |
| **Total** | **3967.285** | **698 / 714** |

## The same set with the artifact cache warm

`script_seconds` above is the cache-free path: with
`TENFERRO_CUDA_ARTIFACT_CACHE=none` every arm compiled its modules in process,
which is what a machine with an empty cache pays. The persistent artifact cache is
merged now (#350), and the same set on the same device measured:

| Regime | Run | Summed suite clocks | Enclosing command | Rows |
|---|---|---:|---:|---:|
| cache disabled (`none`) | `20261105_0820` | 3967.3 s | not recorded as one clock | 698 / 714 |
| cache warm, first pass partly cold | `20261105_0920` | 1010.6 s | 1044 s | 698 / 714 |
| cache warm, fully warm | `20261105_0940` | 897.9 s | 930 s | 698 / 714 |

Device, effort, arms, rows and tenferro-rs `a000e5c4` are the same in all three.
The cold run ran at harness `18ae0c5`; the warm runs at `9517d09`, the same tree
plus a guard for an unset `HOME` in the cache discovery, so the pair differs only
in whether the discovery runs. The enclosing command is the launcher's own clock:
a run with the 16 structural permutation rows absent exits non-zero from the
coverage accounting before `wall_command_complete.json` is written, so the script
clock cannot be read from the run directory.

Per suite, the same two runs: cold sums to **3967.3 s** and warm to **897.9 s**
across the 15 suites, which is the [budget reference](../benchmarks/gpu/routine_budget.yaml)
(`script_seconds` and `cache_conditions`). Each suite's own
`collection_seconds_including_build_and_backend_release_delays` is in its `run.yaml`
under the stored cold run
[`20261105_0820`](../data/results/nvidia-gpu/gpu/routine_selection/20261105_0820);
the per-suite figures are not republished in this note.

Arithmetic, because this field has been misread twice: every cell is that
suite's own
`collection_seconds_including_build_and_backend_release_delays` from its
`wall_command.json`, which measures one suite's window inside one suite's loop
iteration, and each column is the sum of the 15 values. It is never a difference
of `command_seconds_through_this_suite`, which is cumulative from the start of the
script that produced it and means nothing across an invocation boundary. The cold
column is summed across the three invocations that stamp needed; the warm column
is a single invocation.

The two figures are not interchangeable. The budget keeps `script_seconds` at
**3967.3 s** because that is the cache-free reference — what a fresh pod pays on
its first suite, and the only regime reproducible on an empty cache. The warm
figure, **897.9 s** of summed suite clocks (930 s enclosing), is what the
persistent cache buys every later invocation; the budget states it in
`cache_conditions`, which the generated workflow report prints. The published
`routine_workflow.md` for this run predates the cache and will carry the second
regime from the next collection.

Of the two warm passes, the fully warm 897.9 s is the one to quote as the steady
state. The 1010.6 s pass was partly cold only because three earlier probe runs had
already compiled a few dense modules, so it is neither the cold reference nor the
steady state; a scheduled run that follows any earlier run on the same cache sees
the `20261105_0940` shape.

## What the two regimes explain

Warm minus cold is **3069.4 s** (3967.3 - 897.9), or 2956.7 s against the partly
cold first pass. The recorded case-phase fields for the cold run total 1067 s —
preparation 618.5, probe/init 384.8, `compile` 44.8, validation 18.9 — so most of
the ~2900 s that the phase accounting did not explain is per-process compilation
and lazy initialization that a warm cache removes, not fixture generation or
process startup. The largest single block is not in the phase fields at all: the
two AD suites record no case phases and fall from 1290.9 s to 66.6 s. Per case the
same effect: `dense_batched_matmul_f64_b16_32` eager preparation 40.06 s to
0.02 s, `dense_matmul_f64_3072` 39.86 s to 0.37 s, `dense_qr_f64_1536` traced
probe/init 53.58 s to 7.57 s.

What does not move between the regimes: per-suite fixture generation and copy
calibration (117.8 s cold, 114.7 s warm, each measured per suite from its fixture
manifest timestamp), the 34 s of backend-release sleeps, and the delegated suites'
process startup and report glue (~34 s cold, ~29 s warm). There is therefore
little left to optimize in fixtures or process startup: the warm run is the timed
regions, one fixture generation per suite, JAX's per-case jit, and the harness's
own clock-ramp and calibration margins.

## Coverage

The 714 planned rows split into 698 recorded rows and 16 structurally absent
permutation rows: 570 executed (successful timings with retained samples and
numerical validation), 128 `unsupported` capability decisions with a recorded
reason, 16 missing, and no `failed`, `duplicates` or `unexpected` rows. Nine
executed rows carry the descriptive noise flag (sample standard deviation over
10% of the mean); it is not a failure.

The 16 missing rows are all in `gpu/permutation` and are structural, not
failures:

| Class | Rows | Cases and arms |
|---|---:|---|
| Non-memcpy arms on a pure memcpy pattern | 5 | `memcpy_24d_64x2`: tenferro-cuda-transpose, tenferro-cuda-to-contiguous, tenferro-cuda-destination-reuse, cutensor, pytorch-cuda |
| `memcpy-d2d` does not participate | 9 | seven non-memcpy patterns: `transpose_2d_32768_16384`, `transpose_3d_1024_1024_512_201`, `transpose_3d_1024_1024_512_102`, `rotation_6d_64_32_32_32_16_16`, `reverse_23d_128x2`, `reverse_18d_3`, `cyclic_18d_3`; plus `tn_light_415_24d_scattered_to_colmajor_gpu` and `tn_light_415_24d_contiguous_same_perm_gpu` |
| Scattered 24d layout unsupported by the arm | 2 | `tn_light_415_24d_scattered_to_colmajor_gpu`: tenferro-cuda-transpose, tenferro-cuda-destination-reuse |

Because these missing pairs never count as measured coverage, the coverage
report for this run says "Measured coverage complete: no" even though every
recorded row passed.

## Compile totals

With the cache disabled, the public-API Rust arms compiled 52 modules across 34
rows. The run's own phase accounting records 44.799 s in the `compile` phase.
The arms' `compile_time_ms` values sum to 623.845 s, but that field is the
untimed case wall of the case that compiled (fixtures and verification
excluded), not NVRTC time alone.

With the cache warm the same phase field measures 45.1 s, so what it records is
JAX's per-case jit, which no artifact cache can remove; the NVRTC compile appears
in `preparation` (618.5 s cold, 86.3 s warm) and `probe/init` (384.8 s cold, 3.9 s
warm) instead. This field is not where the cache's effect becomes visible.

## Reconstruction note

The `tensornetwork` suite was re-collected at the same timestamp after its first
collection; the stored run carries the replacement (176.540 s). That re-run also
overwrote the shared stamp's `selection.json` and `case_status.json` with its
own four-row view, so both files in the stored run were regenerated by the
repository's own `gpu_routine_selection.py` from the stored per-suite
`records.jsonl` (714 planned, 570 executed, 128 unsupported, 16 missing, 0
failed/duplicates/unexpected). `selection.json` names the stored materialized
suite files. The full durable copy, including 4.08 GB of generated fixtures and
sparse caches, remains at `/home/shinaoka/work/wallclock-results/20261105_0820`
on primerose; those binary fixtures are not committed.
