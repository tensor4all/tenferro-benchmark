# GPU Backend Comparison Design

## Context

This repository currently benchmarks `tenferro-rs` against LibTorch C++,
PyTorch Python, and JAX through CPU-focused runners. The main paths are:

- Rust `tenferro-einsum-benchmark` for trace and eager einsum execution.
- `scripts/benchmark_python.py` for PyTorch and JAX einsum baselines.
- `cpp/benchmark_libtorch.cpp` for LibTorch einsum baselines.
- `scripts/benchmark_cpu_ops_python.py`, `cpp/benchmark_cpu_ops_libtorch.cpp`,
  and `src/bin/publication_gate.rs` for PR884-style dense CPU ops.
- `scripts/run_all.sh` as the orchestration and Markdown reporting entry point.

The `spec4gpu/` directory is a useful independent concept draft, but it does
not match the current benchmark repo directly. Its single-problem schema does
not validate its suite YAML files, and its `glbench` command assumes a new
implementation rather than the runner/reporting pipeline already present here.

`extern/tenferro-rs` can perform GPU computation. CUDA support is provided by
the feature-gated `tenferro-gpu` CubeCL backend and is available through the
concrete tensor, eager, and traced execution surfaces. It uses explicit
CPU/GPU transfer, does not silently fall back to CPU, and returns errors for
unsupported CUDA op/dtype combinations. Current documented CUDA coverage
includes dense elementwise operations, `dot_general`, and dense linalg
operations such as `qr`, `solve`, `svd`, and `eigh` for supported floating and
complex dtypes. General nonsymmetric `eig`, selected pivoting operations,
ROCm/HIP, and some integer/complex analytic gaps remain unsupported.

## Goal

Add a GPU benchmark design based on the current codebase that can compare:

- `tenferro-rs` CUDA trace mode.
- `tenferro-rs` CUDA eager mode.
- PyTorch Python CUDA.
- LibTorch C++ CUDA.
- JAX CUDA.
- Vendor CUDA baselines such as cuBLASLt, CUTLASS, cuSOLVER, cuSPARSE, and
  Ginkgo.

All runners must consume the same problem suite schema and emit the same result
schema. The schema should cover `matmul`, `batched_matmul`, `einsum`, `qr`,
`solve`, `svd`, `eigh`, `spmv`, and `spmm` from the start. Backends that cannot
run a problem must emit a structured `unsupported` result rather than
disappearing from the report.

## Non-Goals

The first implementation is not expected to produce complete measured values
for every operation/backend pair. Sparse and vendor integrations may start as
contract-complete runners that emit `unsupported` or `not_configured` records.

This design does not replace the current CPU benchmark reports. Existing CPU
scripts and result files should continue to work while the GPU suite is added
beside them.

This design does not introduce MLPerf-style governance, distributed multi-node
execution, full autotuning, or training/inference workload benchmarks.

## Recommended Approach

Use the existing benchmark repository as the owner of orchestration, data
layout, result storage, and reports. Treat `spec4gpu/` as reference material,
not as an implementation directory to extend directly.

Add a new GPU benchmark layer with these directories:

```text
benchmarks/gpu/
  dense.yaml
  sparse.yaml
  einsum.yaml
schemas/
  benchmark-suite.schema.json
  benchmark-result.schema.json
scripts/
  validate_benchmark_suite.py
  run_gpu_suite.sh
  format_gpu_results.py
src/bin/
  gpu_benchmark_tenferro.rs
cpp/
  benchmark_gpu_libtorch.cpp
  benchmark_vendor_cuda.cpp
```

Python CUDA support should initially live in a new `scripts/benchmark_gpu_python.py`
instead of expanding `scripts/benchmark_python.py`. This keeps CPU behavior
stable and lets GPU timing, device selection, and synchronization evolve
without fragile mode flags in the current CPU runner.

## Problem Suite Schema

The schema validates a suite file, not only one problem. A suite has:

- `suite_id`: stable suite name.
- `schema_version`: initially `1`.
- `description`: human-readable purpose.
- `defaults`: optional run, device, dtype, verification, and layout defaults.
- `problems`: array of problem records.

Each problem has:

- `id`: stable result join key.
- `family`: `dense`, `sparse`, or `einsum`.
- `op`: `matmul`, `batched_matmul`, `einsum`, `qr`, `solve`, `svd`, `eigh`,
  `spmv`, or `spmm`.
- `dtype`: logical operand/result dtype plus accumulation dtype where relevant.
- `layout`: explicit layout semantics. For tenferro problems this must be
  able to represent column-major tensors; for PyTorch/JAX/LibTorch it must
  represent row-major contiguous tensors or explicit transpose/stride choices.
- `data`: deterministic generator, seed, conditioning model, and matrix source.
- `run`: warmups, repeats, optional minimum runtime, device ordinal, and timing
  mode.
- `verify`: reference policy and tolerances.
- `backend_candidates`: backend ids expected to attempt the problem.

Operation-specific fields are grouped under names matching the operation:

```yaml
op: matmul
matmul:
  m: 2048
  n: 2048
  k: 2048
  transpose_a: false
  transpose_b: false
```

```yaml
op: batched_matmul
batched_matmul:
  batch: 1024
  m: 32
  n: 32
  k: 32
  batch_layout: leading
```

```yaml
op: einsum
einsum:
  format_rowmajor: "ij,jk->ik"
  format_colmajor: "ij,jk->ik"
  shapes_rowmajor: [[256, 256], [256, 256]]
  shapes_colmajor: [[256, 256], [256, 256]]
  path:
    strategy: opt_flops
    pairs: [[0, 1]]
```

```yaml
op: spmv
sparse:
  source: suitesparse
  group: HB
  name: bcspwr10
  storage: csr
  rows: 5300
  cols: 5300
  nnz: 21842
```

The existing `data/instances/*.json` files remain valid legacy einsum inputs.
The GPU suite may reference them through generated `op: einsum` suite entries
or through a converter script. This avoids forcing the current CPU runners to
change format immediately.

## Backend IDs

Backend ids are stable strings in result records:

- `tenferro-cuda-trace`
- `tenferro-cuda-eager`
- `pytorch-cuda`
- `libtorch-cuda`
- `jax-cuda`
- `cublaslt`
- `cutlass`
- `cusolver`
- `cusparse`
- `ginkgo`

CPU ids remain unchanged in existing reports. If a future unified CPU/GPU
report is needed, CPU ids should be explicit: `tenferro-cpu-trace`,
`tenferro-cpu-eager`, `pytorch-cpu`, `libtorch-cpu`, and `jax-cpu`.

## Result Schema

Results are emitted as JSON Lines. One backend attempt for one problem emits
one record. This makes partial runs, crashed backends, and incremental report
generation easier than a single large JSON file.

Required fields:

- `schema_version`
- `suite_id`
- `problem_id`
- `op`
- `backend`
- `status`
- `timing`
- `verification`
- `environment`
- `execution`

`status` values:

- `ok`
- `unsupported`
- `not_configured`
- `compile_failed`
- `runtime_failed`
- `oom`
- `verification_failed`
- `cpu_fallback`

`timing` fields:

- `warmup_runs`
- `timed_runs`
- `compile_time_ms`
- `first_run_ms`
- `median_ms`
- `min_ms`
- `p95_ms`
- `iqr_ms`
- `timing_scope`

`timing_scope` is one of:

- `steady_state_device_only`
- `steady_state_host_api_plus_device_sync`
- `steady_state_host_api_plus_device_sync_with_download`
- `compile_plus_first_run`

The default comparison table uses `steady_state_host_api_plus_device_sync`
because it is implementable across all frameworks and includes framework
dispatch overhead. Vendor CUDA runners may additionally record
`steady_state_device_only` measured with CUDA events.

`verification` fields:

- `status`: `passed`, `failed`, `skipped`, or `not_applicable`
- `reference_backend`
- `max_abs_error`
- `max_rel_error`
- `residual`
- `rtol`
- `atol`

`environment` fields:

- `hostname`
- `timestamp_utc`
- `os`
- `gpu_name`
- `gpu_uuid`
- `gpu_memory_bytes`
- `driver_version`
- `cuda_version`
- `cudnn_version`
- `framework_version`
- `tenferro_rs_commit`
- `benchmark_repo_commit`
- `env`

`execution` fields:

- `device`
- `device_ordinal`
- `execution_path`
- `synchronization`
- `layout`
- `dtype`
- `notes`
- `unsupported_reason`

## Runner Contracts

Every runner accepts:

```bash
runner --suite benchmarks/gpu/dense.yaml \
       --backend BACKEND_ID \
       --output data/results/gpu_BACKEND_ID_TIMESTAMP.jsonl \
       --problem-filter optional_problem_id
```

Each runner must:

- Validate the suite enough to reject malformed inputs before timing.
- Emit one JSONL result for every attempted problem.
- Emit `unsupported` for unsupported operations, dtypes, layouts, or sparse
  formats.
- Emit `not_configured` when required CUDA libraries or GPU devices are absent.
- Keep input generation, host-to-device upload, and device-to-host download
  outside steady-state timing unless the problem explicitly requests transfer
  timing.
- Use deterministic input data matching the suite seed and generator.
- Record the execution path and synchronization method.

## Tenferro CUDA Runner

The Rust runner should be a new binary, `gpu_benchmark_tenferro`, to avoid
destabilizing the existing CPU `tenferro-einsum-benchmark` binary.

Trace mode uses:

- Host tensors generated with `Tensor::from_vec_col_major`.
- `tenferro_gpu::cubecl::upload_tensor` to create CUDA-resident inputs.
- `TracedTensor::from_tensor_concrete_shape` to bind CUDA inputs.
- `GraphExecutor<CubeclBackend>` for execution.
- `tenferro_einsum::register_runtime` for einsum.
- `tenferro_linalg` extension registration for linalg operations when needed.

Eager mode uses:

- `CubeclBackend::new(device_ordinal)`.
- `EagerRuntime::with_cuda_backend`.
- Explicitly uploaded CUDA tensors wrapped in `EagerTensor`.
- `tenferro_einsum::eager_tensor` for einsum contractions.
- `tenferro_linalg` eager helpers or backend methods for linalg.

Synchronization should avoid measuring downloads. If `tenferro-gpu` exposes a
public stream synchronize helper by implementation time, use it. Otherwise the
runner may use the raw CUDA stream exposed by `CubeclRuntime::raw_cuda_stream`
plus CUDA runtime bindings, and should record that method. Downloading only to
synchronize is allowed for an initial smoke implementation but must be recorded
as `timing_scope: steady_state_host_api_plus_device_sync_with_download` and
excluded from fair report columns.

Unsupported tenferro cases should be explicit. For example, general `eig`,
ROCm, unsupported sparse operations, missing cuTENSOR/cuSOLVER/cuBLAS, or
unsupported dtypes produce structured non-`ok` records.

## PyTorch Python CUDA Runner

The Python GPU runner uses `torch` tensors on `cuda:{device_ordinal}`.

Synchronization:

- Warmups call the operation and then `torch.cuda.synchronize()`.
- Timed runs start a host timer, call the operation, call
  `torch.cuda.synchronize()`, and stop the host timer.
- Optional device-only timing can use CUDA events later.

`torch.compile` is not folded into `pytorch-cuda`. It should be a later
backend variant such as `pytorch-cuda-compile`, with compile time separated
from steady-state timing.

CPU fallback must be detected where possible. If an operation returns a CPU
tensor, the result status is `cpu_fallback`.

## LibTorch C++ CUDA Runner

The C++ GPU runner should be separate from the current CPU
`benchmark_libtorch.cpp`. It uses LibTorch CUDA tensors and CMake configuration
that does not require the current OpenBLAS-linked LibTorch package.

Synchronization:

- Use `cudaDeviceSynchronize()` or LibTorch CUDA stream synchronization after
  each warmup/timed operation.
- Record CUDA, LibTorch, and GPU metadata.

The CPU runner's OpenBLAS linkage checks stay in place for CPU comparisons but
must not be required for CUDA LibTorch runs.

## JAX CUDA Runner

The JAX runner uses `jax.numpy` arrays on the default GPU device or a selected
GPU device where supported.

Synchronization:

- Use `block_until_ready()` for every warmup and timed run.
- Separate compilation time and steady-state timing for `jit` variants.

The base `jax-cuda` backend should be non-`jit` or the default JAX execution
mode chosen by the runner, with the execution path recorded. A later
`jax-cuda-jit` variant can make compilation behavior explicit.

## Vendor CUDA Runner

Vendor baselines are separate backend ids, not framework variants.

Initial responsibilities:

- `cublaslt`: dense `matmul` and selected `batched_matmul`.
- `cutlass`: dense `matmul`, grouped or batched GEMM where configured.
- `cusolver`: `qr`, `solve`, `svd`, `eigh`.
- `cusparse`: `spmv`, `spmm` for CSR matrices.
- `ginkgo`: sparse `spmv`, `spmm`, and future iterative solvers.

The first implementation may emit `not_configured` for missing SDKs or
`unsupported` for unimplemented operations. This still validates the schema,
orchestration, and reports before all native wrappers exist.

## Verification

Small dense problems use CPU fp64 references. Larger dense problems use
backend-to-reference comparison or residual checks:

- `matmul`, `batched_matmul`, and `einsum`: max absolute and relative error.
- `solve`: `||AX - B|| / (||A|| ||X|| + ||B||)`.
- `qr`: `||A - QR|| / ||A||` and `||Q^TQ - I||`.
- `svd`: `||A - U diag(S) Vh|| / ||A||`.
- `eigh`: `||AV - V diag(w)|| / ||A||` and orthogonality.
- `spmv` and `spmm`: CPU sparse reference or vendor sparse baseline.

Verification runs outside steady-state timing. For initial implementation,
verification may be limited to smoke-sized problems but must record
`verification.status: skipped` and a reason for large problems.

## Reporting

GPU JSONL files are stored under `data/results/` with names such as:

```text
gpu_tenferro-cuda-trace_20260528_120000.jsonl
gpu_pytorch-cuda_20260528_120000.jsonl
```

`scripts/format_gpu_results.py` generates:

- `data/results/gpu_results_TIMESTAMP.md`
- `result/gpu-benchmark-results.md`

The report groups by suite and operation, then shows one row per problem and
one column per backend. Non-`ok` statuses render as short status labels such as
`unsupported`, `not configured`, or `verification failed`. Full reasons remain
in the JSONL.

Existing `result/einsum-results.md` and `result/cpu-benchmark-results.md` are
not changed by the GPU formatter.

## Orchestration

Add `scripts/run_gpu_suite.sh` as the GPU entry point. It should:

1. Validate suite files.
2. Resolve `extern/tenferro-rs` commit.
3. Build the tenferro CUDA runner with `--no-default-features --features cuda`.
4. Run enabled backend runners, continuing after backend-specific failures.
5. Emit JSONL for every backend and problem where possible.
6. Format Markdown/CSV summaries from JSONL.

Environment controls:

- `GPU_BENCH_SUITE`: comma-separated suite paths or logical suite names.
- `GPU_BENCH_BACKENDS`: comma-separated backend ids.
- `GPU_BENCH_DEVICE`: CUDA device ordinal, default `0`.
- `GPU_BENCH_PROBLEM`: optional single problem id.
- `GPU_BENCH_RUNS`: timed run count.
- `GPU_BENCH_WARMUPS`: warmup count.
- `CUDA_PATH`, `LD_LIBRARY_PATH`, `TENFERRO_CUTENSOR_PATH`,
  `TENFERRO_CUSOLVER_PATH`, and `TENFERRO_CUBLAS_PATH` for tenferro CUDA.

The existing devcontainer currently sets `USE_CUDA=0` and does not install a
CUDA toolkit. GPU benchmark documentation should therefore target a CUDA-capable
host, a separate GPU devcontainer, or a self-hosted GPU CI runner.

## Testing Strategy

Schema and formatter tests should run without a GPU:

- Validate representative dense, sparse, and einsum suite YAML.
- Validate representative `ok`, `unsupported`, and `not_configured` result
  records.
- Test JSONL formatting into Markdown columns.
- Test that missing backends still produce stable report tables.

Runner smoke tests should be split:

- CPU-only unit tests for suite parsing and deterministic data generation.
- CUDA ignored tests for tenferro `matmul`, `einsum`, and one linalg operation.
- Optional self-hosted GPU CI workflow for nightly or manual full GPU runs.

Existing CPU checks remain useful after benchmark script changes:

```bash
cargo metadata --no-deps --format-version 1
bash tests/test_run_all_docs_outputs.sh
bash tests/test_extern_dependency_paths.sh
bash tests/test_run_all_rust_bin_selection.sh
bash tests/test_native_batch_layout_labels.sh
```

New non-GPU checks should be added for the GPU schema and formatter.

## Implementation Phases

Phase 1: Schema and report contract.

- Add suite and result schemas.
- Add sample dense/einsum/sparse suites.
- Add suite validation script.
- Add JSONL formatter.
- Add tests for validation and formatting.

Phase 2: tenferro CUDA and PyTorch CUDA measured MVP.

- Add `gpu_benchmark_tenferro`.
- Add `benchmark_gpu_python.py` for `pytorch-cuda`.
- Support `matmul`, `batched_matmul`, and `einsum`.
- Emit `unsupported` for linalg/sparse/vendor gaps.

Phase 3: JAX CUDA and LibTorch CUDA.

- Add `jax-cuda` to the Python GPU runner.
- Add `benchmark_gpu_libtorch.cpp`.
- Support dense matmul/einsum and selected linalg where straightforward.

Phase 4: Dense linalg and vendor baselines.

- Add tenferro/PyTorch/JAX/LibTorch support for `qr`, `solve`, `svd`, and
  `eigh`.
- Add cuBLASLt/cuSOLVER runner support or structured `not_configured` records.

Phase 5: Sparse suite.

- Add SuiteSparse metadata/download/cache contract.
- Add `spmv` and `spmm` problem support.
- Add cuSPARSE/Ginkgo runners or structured unsupported records.

## Risks and Open Decisions

Tenferro GPU synchronization needs a download-free public helper or a runner
local CUDA stream synchronization wrapper. Until that exists, fair steady-state
timing must avoid using download as the synchronization mechanism in published
comparison columns.

JAX execution paths can compile lazily. Reports must distinguish first-run,
compile, and steady-state timing so JAX numbers are not mixed with PyTorch or
vendor steady-state numbers incorrectly.

LibTorch CPU and CUDA should use separate CMake targets because the CPU target
requires OpenBLAS-linked LibTorch while CUDA does not.

Sparse support should be schema-complete early but measurement-complete later.
The report must make unsupported sparse backend cells explicit.

The broad B-scope creates many operation/backend combinations. The first
implementation should privilege contract stability and honest statuses over
partial hidden behavior.
