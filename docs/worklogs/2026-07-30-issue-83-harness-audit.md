# Issue 83 Harness Validity Worklog

Date: 2026-07-30

## Scope

- Replace the diagonal `eigvalsh`/Cholesky fixture with an equivalent dense,
  deterministic SPD fixture in the Rust, PyTorch, and JAX runners.
- Add separate automatic report warnings for cross-backend spreads and
  operation/shape physical-bound violations across the full public API suite.
- Add a CUDA destination-reuse row through the public production
  `CudaBackend::copy_read_into` dispatch.

## Root Cause

The old `spd` helpers constructed an exactly diagonal matrix. PyTorch and the
tenferro trace path selected a diagonal-specialized or constant path, so their
`eigvalsh` cells did not measure a representative dense symmetric eigensolve.
The CPU timing boundaries were already synchronous; missing synchronization
was not the cause.

The replacement fixture is a small symmetric dense perturbation of a positive
diagonal. It is strictly diagonally dominant, so it remains positive definite
without charging any backend for fixture construction inside the timed region.
The Rust, PyTorch, and JAX fixtures are pinned to the same known values for
`n=3, seed=7`, not only to structural SPD properties.

The first destination-reuse implementation was invalid: it called the
benchmark's raw cuTENSOR helper under a tenferro backend label, duplicating the
`cutensor` control without going through `CudaBackend` dispatch, error handling,
or cache ownership. Post-#1509 tenferro does expose
`TensorStructural::copy_read_into`. Its CUDA contract is compact source to
arbitrary non-overlapping destination view, so the corrected benchmark copies
the compact source into an inverse-permuted view of a caller-owned compact
output. The production implementation currently uses the native CUDA copy
kernel, not the cuTENSOR permutation cache.

## Test-First Reproduction

Before the implementation:

- the Rust fixture test failed because it found zero nonzero off-diagonal
  elements;
- the PyTorch fixture test found zero nonzero off-diagonal elements;
- the JAX helper did not accept the fixture seed and also produced a diagonal
  matrix;
- a synthetic copy of the implausible `eigvalsh` row produced no report
  warning;
- the GPU suite had no destination-reuse backend.

Review follow-up added two further RED cases:

- equal 0.100 ms `eigvalsh` cells produced no warning because the original
  check only compared backends with each other;
- the physical model did not parse output shapes, right-hand-side counts, or
  concatenated extents and did not cover einsum, dot-general,
  triangular-solve, or Frobenius-norm work;
- the destination-reuse source-contract test showed that the row still shared
  the raw cuTENSOR helper and incorrectly participated in the scattered-source
  pattern;
- the destination row's per-call mutable-view construction was inside timing
  while the documentation claimed that all setup was outside timing;
- the formatter module documentation and runner status line omitted the
  destination-reuse backend.

All of those regression tests pass after the implementation.

## Focused CPU Measurement

The release binary used a dedicated target directory:

`/home/shinaoka/tensor4all/.target/benchmark-issue83-cpu`

Measurements ran sequentially with seven samples and three warmups. The
one-thread run was bound to CPU 60; the four-thread run was bound to CPUs
56-59. The Rust, PyTorch, and JAX runners used the
`cpu/linalg_uncovered`/`eigvalsh` filters.

| threads | tenferro direct | tenferro trace | PyTorch | JAX |
|---:|---:|---:|---:|---:|
| 1 | 35.389 ms | 17.549 ms | 16.675 ms | 33.927 ms |
| 4 | 20.887 ms | 16.231 ms | 9.422 ms | 23.250 ms |

For comparison, the previous report contained approximately 0.21 ms for
tenferro trace and 0.27 ms for PyTorch at one thread. The dense fixture removes
those implausible cells; all four implementations now measure the same
operation-complexity class.

## GPU Measurement

The CUDA binary used a separate release target:

`/home/shinaoka/tensor4all/.target/benchmark-issue83-gpu`

A focused run on an NVIDIA A100 80GB PCIe used
`transpose_2d_32768_16384`, three warmups, seven measured runs, and CPUs 52-55
for the host process. All four Rust participants passed the exact-value
correctness gate. The destination allocation and inverse permutation are
one-time setup. `copy_read_into` consumes its mutable view, so the timed public
API dispatch includes O(rank) destination-view metadata reconstruction and
device synchronization.

| backend | allocation per call | median | p25 | p75 |
|---|---:|---:|---:|---:|
| tenferro CUDA transpose | yes | 5.603 ms | 5.580 ms | 5.616 ms |
| tenferro CUDA to_contiguous | yes | 5.596 ms | 5.590 ms | 5.603 ms |
| tenferro CUDA copy_read_into | no | 25.893 ms | 25.822 ms | 25.989 ms |
| direct cuTENSOR | no | 5.403 ms | 5.400 ms | 5.407 ms |

The corrected destination-reuse record reports `per_call_allocation=false`,
`correctness=passed`, and names public `CudaBackend::copy_read_into` in its
notes. It is 4.6x slower than allocation-inclusive `to_contiguous` on this
seven-sample median because the production caller-owned path uses a different
native kernel. Consequently this row is a valid public API comparison but does
not isolate allocation cost. A future allocation-only comparison requires a
caller-owned cuTENSOR permutation API in tenferro; the benchmark must not
simulate one through private or duplicated raw FFI.

A second A100 run used the non-self-inverse 3D permutation `[2,0,1]` to verify
the inverse-permuted destination-view mapping, again with three warmups and
seven measured runs. Exact correctness passed for all four Rust participants.

| backend | allocation per call | median | p25 | p75 |
|---|---:|---:|---:|---:|
| tenferro CUDA transpose | yes | 5.512 ms | 5.476 ms | 6.660 ms |
| tenferro CUDA to_contiguous | yes | 5.542 ms | 5.443 ms | 5.767 ms |
| tenferro CUDA copy_read_into | no | 84.692 ms | 84.643 ms | 84.749 ms |
| direct cuTENSOR | no | 5.340 ms | 5.336 ms | 5.343 ms |

Improving the production caller-owned path belongs in tenferro-rs, where the
backend can route compact-source/strided-destination copies through its owned
cuTENSOR cache and typed error policy. It is not a benchmark-harness
optimization.

## Plausibility Model

The report now keeps two independent audits:

- Cross-backend spread flags a successful cell more than 10x faster than the
  slowest successful implementation in the same row.
- Physical-bound audit parses dense numeric shapes plus output (`->`), RHS
  (`rhs=`), and concatenation (`+`) decorations. It derives a guaranteed
  minimum logical byte count for every materializing public API row and
  conservative scalar-operation counts for reductions, Frobenius norms,
  matrix products, triangular solves, and dense linalg. The matrix-product
  model uses one scalar multiply-accumulate per output/inner-index tuple;
  triangular solve uses `n^2 * rhs / 2`; reductions and `norm_fro` use one
  scalar operation per input element. These intentionally undercount real and
  complex arithmetic.
- Static and dynamic slice rows now declare their exact output extents in all
  three runners. The formatter retains exact compatibility values for the
  checked-in pre-change report; unknown legacy slice shapes remain unmodeled
  instead of estimating a potentially too-large touched region.
- The current full report has 86 unique workloads. Eighty-two materializing or
  compute workloads have a nonzero byte or FLOP model. The only unmodeled rows
  are the four intentional metadata-only views:
  `broadcast_in_dim_view`, `reshape_view`, `slice_view`, and `transpose_view`.
- The audit uses explicit ceilings of 100 GB/s and 50 GFLOP/s per requested
  CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster
  than the resulting lower bound is flagged even when every backend reports
  the same shortcut.

For `eigvalsh` 512x512, the model uses 134,217,728 FLOPs. A synthetic
all-backend 0.100 ms row implies 1342.2 GFLOP/s and is flagged by the physical
audit despite having no backend spread. The corrected focused measurements
produce no warning in either audit. Regression tests also pin the conservative
work estimates for `einsum_ij_jk_ik` 1024x1024 (1,073,741,824),
complex `dot_general` 640x640 (262,144,000), `triangular_solve`
4096x4096 with 64 RHS columns (536,870,912), and `norm_fro` 2048x2048
(4,194,304).

## Verification

- `uv run python tests/test_public_api_harness_validity.py`
- `cargo test --bin benchmark_cpu_public_api --no-default-features --features cpu-faer spd_fixture_is_dense`
- `bash tests/test_gpu_permutation_destination_reuse.sh`
- `uv run python scripts/validate_benchmark_suite.py benchmarks/cpu/public_api.yaml benchmarks/gpu/permutation.yaml`
- `bash tests/test_gpu_permutation_patterns.sh`
- `bash tests/test_permutation_result_schema.sh`
- `uv run bash tests/test_mac_gpu_permutation_profile.sh`
- `bash tests/test_suite_result_layout.sh`
- `bash tests/test_run_all_docs_outputs.sh`
- `bash tests/test_markdown_formatters_no_blank_eof.sh`
- `uv run python -m py_compile` for the changed Python files
- `git diff --check`
- A100 focused run with `BENCH_RUNS=7`, `BENCH_WARMUPS=3`, and
  `PATTERN_ID=transpose_2d_32768_16384`
- A100 non-self-inverse permutation run with `BENCH_RUNS=7`,
  `BENCH_WARMUPS=3`, and
  `PATTERN_ID=transpose_3d_1024_1024_512_201`

`cargo fmt --all -- --check` also inspects pre-existing unformatted code in
`examples/svd_backward_panic_mwe.rs`, `src/bin/cpu_matmul_diagnostic.rs`, and
unchanged sections of `src/bin/benchmark_gpu_permutation.rs`. It therefore
does not pass on the base revision. The Rust additions in this change were
formatted directly; unrelated formatting was left untouched.

`uv run ruff format --check` was not available because `ruff` is not installed
in this repository's environment. Python syntax and behavior were instead
checked by `py_compile` and the focused Python test suite.
