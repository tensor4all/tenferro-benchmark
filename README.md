# tenferro-benchmark

Timing policy: [timing policy revision 2](docs/timing-policy.md); short operations use
[many operations in one session](docs/short-operation-sessions.md).

Benchmark suite for [tenferro-rs](https://github.com/tensor4all/tenferro-rs),
comparing tenferro against PyTorch, JAX, Julia, HPTT, cuTENSOR, and other
reference backends on CPU and GPU workloads.

## Latest Results

The published pages cover only the latest hardware generation, so older data
is never mistaken for current: **`zen5-cpu`** (AMD Ryzen AI 9 HX 470 workstation,
pinned 1/4/8-thread rows; see [AGENTS.md](AGENTS.md#zen5-cpu-workflow-pinned-workstation-rows)) and
**`nvidia-gpu`** (A100). Older profiles (`amd-cpu`, `mac-cpu`, `mac-gpu`) are
kept in `result/` and git history but are not linked from here or from the
Pages site.

- Live pages, generated in CI on every deploy: <https://tensor4all.org/tenferro-benchmark/>
  (landing page, per-suite pages, and a per-commit
  [history](https://tensor4all.org/tenferro-benchmark/history.html)).
- Source of truth (never duplicated into this README): the tracked reports under
  [`result/zen5-cpu/`](result/zen5-cpu/) and [`result/nvidia-gpu/`](result/nvidia-gpu/).
  Each file is the latest report for one `target_profile` x `suite_id` pair;
  the table below is generated from them
  (`scripts/generate_readme_results.py`, checked by CI).

<!-- results:begin -->
| Target profile | Suite | Report |
|---|---|---|
| `zen5-cpu` | `cpu/cpu_ops` | [result/zen5-cpu/cpu/cpu_ops.md](result/zen5-cpu/cpu/cpu_ops.md) |
| `zen5-cpu` | `cpu/einsum` | [result/zen5-cpu/cpu/einsum.md](result/zen5-cpu/cpu/einsum.md) |
| `zen5-cpu` | `cpu/fft` | [result/zen5-cpu/cpu/fft.md](result/zen5-cpu/cpu/fft.md) |
| `zen5-cpu` | `cpu/linalg_jvp_vjp` | [result/zen5-cpu/cpu/linalg_jvp_vjp.md](result/zen5-cpu/cpu/linalg_jvp_vjp.md) |
| `zen5-cpu` | `cpu/overhead_smoke` | [result/zen5-cpu/cpu/overhead_smoke.md](result/zen5-cpu/cpu/overhead_smoke.md) |
| `zen5-cpu` | `cpu/permutation` | [result/zen5-cpu/cpu/permutation.md](result/zen5-cpu/cpu/permutation.md) |
| `zen5-cpu` | `cpu/public_api` | [result/zen5-cpu/cpu/public_api.md](result/zen5-cpu/cpu/public_api.md) |
| `zen5-cpu` | `cpu/session_matrix` | [result/zen5-cpu/cpu/session_matrix.md](result/zen5-cpu/cpu/session_matrix.md) |
| `zen5-cpu` | `cpu/small_work` | [result/zen5-cpu/cpu/small_work.md](result/zen5-cpu/cpu/small_work.md) |
| `nvidia-gpu` | `gpu/command_wall` | [result/nvidia-gpu/gpu/command_wall.md](result/nvidia-gpu/gpu/command_wall.md) |
| `nvidia-gpu` | `gpu/compare_lifetime` | [result/nvidia-gpu/gpu/compare_lifetime.md](result/nvidia-gpu/gpu/compare_lifetime.md) |
| `nvidia-gpu` | `gpu/compare_lifetime_counters` | [result/nvidia-gpu/gpu/compare_lifetime_counters.md](result/nvidia-gpu/gpu/compare_lifetime_counters.md) |
| `nvidia-gpu` | `gpu/dense` | [result/nvidia-gpu/gpu/dense.md](result/nvidia-gpu/gpu/dense.md) |
| `nvidia-gpu` | `gpu/dense_throughput` | [result/nvidia-gpu/gpu/dense_throughput.md](result/nvidia-gpu/gpu/dense_throughput.md) |
| `nvidia-gpu` | `gpu/einsum` | [result/nvidia-gpu/gpu/einsum.md](result/nvidia-gpu/gpu/einsum.md) |
| `nvidia-gpu` | `gpu/elementwise` | [result/nvidia-gpu/gpu/elementwise.md](result/nvidia-gpu/gpu/elementwise.md) |
| `nvidia-gpu` | `gpu/fft` | [result/nvidia-gpu/gpu/fft.md](result/nvidia-gpu/gpu/fft.md) |
| `nvidia-gpu` | `gpu/isolated_ops` | [result/nvidia-gpu/gpu/isolated_ops.md](result/nvidia-gpu/gpu/isolated_ops.md) |
| `nvidia-gpu` | `gpu/linalg_ad_latency` | [result/nvidia-gpu/gpu/linalg_ad_latency.md](result/nvidia-gpu/gpu/linalg_ad_latency.md) |
| `nvidia-gpu` | `gpu/linalg_ad_throughput` | [result/nvidia-gpu/gpu/linalg_ad_throughput.md](result/nvidia-gpu/gpu/linalg_ad_throughput.md) |
| `nvidia-gpu` | `gpu/linalg_capability` | [result/nvidia-gpu/gpu/linalg_capability.md](result/nvidia-gpu/gpu/linalg_capability.md) |
| `nvidia-gpu` | `gpu/linalg_jvp_vjp` | [result/nvidia-gpu/gpu/linalg_jvp_vjp.md](result/nvidia-gpu/gpu/linalg_jvp_vjp.md) |
| `nvidia-gpu` | `gpu/linalg_public_api` | [result/nvidia-gpu/gpu/linalg_public_api.md](result/nvidia-gpu/gpu/linalg_public_api.md) |
| `nvidia-gpu` | `gpu/lstsq` | [result/nvidia-gpu/gpu/lstsq.md](result/nvidia-gpu/gpu/lstsq.md) |
| `nvidia-gpu` | `gpu/nn` | [result/nvidia-gpu/gpu/nn.md](result/nvidia-gpu/gpu/nn.md) |
| `nvidia-gpu` | `gpu/perf_issues` | [result/nvidia-gpu/gpu/perf_issues.md](result/nvidia-gpu/gpu/perf_issues.md) |
| `nvidia-gpu` | `gpu/perf_issues_aa` | [result/nvidia-gpu/gpu/perf_issues_aa.md](result/nvidia-gpu/gpu/perf_issues_aa.md) |
| `nvidia-gpu` | `gpu/perf_issues_paired` | [result/nvidia-gpu/gpu/perf_issues_paired.md](result/nvidia-gpu/gpu/perf_issues_paired.md) |
| `nvidia-gpu` | `gpu/permutation` | [result/nvidia-gpu/gpu/permutation.md](result/nvidia-gpu/gpu/permutation.md) |
| `nvidia-gpu` | `gpu/reuse_contraction` | [result/nvidia-gpu/gpu/reuse_contraction.md](result/nvidia-gpu/gpu/reuse_contraction.md) |
| `nvidia-gpu` | `gpu/routine_command_wall` | [result/nvidia-gpu/gpu/routine_command_wall.md](result/nvidia-gpu/gpu/routine_command_wall.md) |
| `nvidia-gpu` | `gpu/routine_controls` | [result/nvidia-gpu/gpu/routine_controls.md](result/nvidia-gpu/gpu/routine_controls.md) |
| `nvidia-gpu` | `gpu/routine_coverage` | [result/nvidia-gpu/gpu/routine_coverage.md](result/nvidia-gpu/gpu/routine_coverage.md) |
| `nvidia-gpu` | `gpu/routine_workflow` | [result/nvidia-gpu/gpu/routine_workflow.md](result/nvidia-gpu/gpu/routine_workflow.md) |
| `nvidia-gpu` | `gpu/shape_stream` | [result/nvidia-gpu/gpu/shape_stream.md](result/nvidia-gpu/gpu/shape_stream.md) |
| `nvidia-gpu` | `gpu/sparse` | [result/nvidia-gpu/gpu/sparse.md](result/nvidia-gpu/gpu/sparse.md) |
| `nvidia-gpu` | `gpu/structural` | [result/nvidia-gpu/gpu/structural.md](result/nvidia-gpu/gpu/structural.md) |
| `nvidia-gpu` | `gpu/tensornetwork` | [result/nvidia-gpu/gpu/tensornetwork.md](result/nvidia-gpu/gpu/tensornetwork.md) |
<!-- results:end -->

The [CUDA public API suites](docs/gpu-public-api-suite.md) add FFT, NN, isolated
operations, structural operations, linear algebra and output reuse comparisons
with PyTorch and JAX. Their family reports live under `result/nvidia-gpu/gpu/`.

Raw runs (per-timestamp `run.yaml` + machine-readable outputs) are written to
`data/results/<target_profile>/<suite_id>/<timestamp>/`; the tracked report in
`result/` is generated from the explicitly selected raw runs. See
[docs/results.md](docs/results.md) for the full layout.

A live dashboard is published to GitHub Pages at
<https://tensor4all.org/tenferro-benchmark/> (`scripts/generate_dashboard.py`).
It reads machine-readable `result/<profile>/<suite>.summary.json` sidecars
written next to each maintained report (schema `summary-schema-v1`), shows the
latest-generation profiles only (`zen5-cpu`, `nvidia-gpu`), and flags per-case anomalies. A CPU
sidecar is written by `scripts/regenerate_report_summaries.py --publish`, which
derives both the page and its summary from explicitly selected run directories;
`--check` re-derives every maintained page from the runs its summary records and
fails on a byte difference. Collection uploads raw
bundles; validated ingestion commits the published subset; Pages renders only
committed results. See [result ingestion and recovery](docs/result-ingestion.md).

## Prerequisites

All profiles:

- Rust toolchain (`cargo`).
- [uv](https://docs.astral.sh/uv/) for the Python environment (`uv sync`
  creates `.venv` from `pyproject.toml`).
- External checkouts under `extern/` (tenferro-rs and problem
  data), fetched by `./scripts/setup_extern_deps.sh`. Only tenferro-rs is
  pinned by revision, in `pins/tenferro-rs.rev`: `TENFERRO_UPDATE=1` checks
  out that revision (`TENFERRO_REF=<ref>` overrides the pin); without it the
  script reuses or clones the checkout as it is. strided-perm / strided-view /
  threadteam come from crates.io and are recorded in `run.yaml`
  (`resolved_deps`).

For `mac-cpu`, run the "All profiles" commands directly on the host. For the
devcontainer profiles (`amd-cpu` / `nvidia-gpu`), run them
**inside the corresponding devcontainer** (`devcontainer exec ... bash -lc
'uv sync && ./scripts/setup_extern_deps.sh'`) — the setup script expects the
container's `OPENBLAS_ROOT` / `MKLROOT` environment, and the container's
`.venv` must be built with the container's wheels (see the GPU note below).

Profile-specific:

- `zen5-cpu`: the AMD Ryzen AI 9 HX 470 workstation; thread rows 1/4/8 run under
  `taskset` (see AGENTS.md).
- `mac-cpu`: runs natively (no Docker); tenferro uses Accelerate.
- `amd-cpu`: the [devcontainer CLI](https://github.com/devcontainers/cli)
  and Docker; tenferro defaults to OpenBLAS, oneMKL is optional.
- `nvidia-gpu`: the CUDA devcontainer under `.devcontainer/cuda/`.
- `cpu/permutation`, `cpu/public_api`, and `cpu/einsum` suites: Julia on
  `PATH` (e.g. [juliaup](https://github.com/JuliaLang/juliaup) or
  `brew install julia`) for the `julia-base`/`strided-jl` columns
  (`cpu/permutation`/`cpu/public_api`) and the `omeinsum-jl` column
  (`cpu/einsum`); the repo `Project.toml` pulls in JSON.jl, LinearAlgebra
  (stdlib), Strided.jl, and OMEinsum.jl via `Pkg.instantiate`. Without
  `julia`, those columns are skipped with a warning. For the HPTT
  column (present in the tracked latest `cpu/permutation` reports), also
  install cmake plus a C++ toolchain (macOS: `brew install cmake`) and pass
  `PERMUTATION_EXTRA_FEATURES=hptt`, because the `hptt` Cargo feature builds
  the vendored HPTT C++ library.

Workflow guides per platform:
[macOS CPU](docs/macos-cpu.md) ·
[Linux CPU devcontainer](docs/linux-devcontainer.md) ·
[NVIDIA GPU devcontainer](docs/gpu-devcontainer.md).

## Running the Benchmarks

### CPU einsum + ops (macOS, native)

`scripts/run_all.sh [NUM_THREADS]` runs the CPU einsum suite
(tenferro trace/eager vs PyTorch/JAX) plus the CPU ops microbenchmarks
(primal linalg, JVP/VJP, eager backward), and regenerates
`result/<target_profile>/cpu/{einsum,cpu_ops,linalg_jvp_vjp}.md`.
Passing multiple thread counts runs those main suites once per thread count,
then runs the FFT, public API, and permutation suites once over the same
thread-count list, regenerating all tracked CPU reports.

```bash
uv sync
./scripts/setup_extern_deps.sh
BENCHMARK_TARGET_PROFILE=mac-cpu ./scripts/run_all.sh 1
BENCHMARK_TARGET_PROFILE=mac-cpu ./scripts/run_all.sh 4
```

To regenerate all tracked `result/mac-cpu/cpu/*.md` reports in one sequential
orchestration, including FFT, public API, and permutation at 1T and 4T, run:

```bash
uv sync
./scripts/setup_extern_deps.sh
# Julia on PATH; for HPTT: brew install cmake (and a C++ toolchain)
PERMUTATION_EXTRA_FEATURES=hptt \
BENCHMARK_TARGET_PROFILE=mac-cpu ./scripts/run_all.sh 1 4
```

Quick smoke (single small instance, one run, no warmup):

```bash
BENCHMARK_TARGET_PROFILE=mac-cpu \
BENCH_INSTANCE=bin_matmul_256 \
BENCH_RUNS=1 \
BENCH_WARMUPS=0 \
PUBLICATION_GATE_SUITE=small \
  ./scripts/run_all.sh 1
```

Useful environment variables: `BENCH_INSTANCE` (restrict to one einsum
instance), `BENCH_RUNS` / `BENCH_WARMUPS` (iteration counts),
`TENFERRO_CPU_FEATURES` (`native` or a BLAS provider: `blas-openblas`,
`blas-mkl`, `blas-accelerate`; **Linux defaults to `blas-openblas`**,
macOS defaults to `blas-accelerate`), `RUN_FFT_SUITE=0`, `RUN_PUBLIC_API_SUITE=0`, `RUN_SMALL_WORK_SUITE=0`,
`RUN_PERF_ISSUES_SUITE=0`, and `RUN_PERMUTATION_SUITE=0`
(skip one of the follow-up suites in a multi-thread-count `run_all.sh`
invocation; HPTT still needs `PERMUTATION_EXTRA_FEATURES=hptt`).

`BENCHMARK_RAW_ONLY=1` records a run and stops: every CPU runner closes the run
record after collection (the per-thread `case_status` files are merged into
`case_status.json` and `wall_command.json` reports the command time through
collection), and then renders nothing — no run report, no summary and no
`result/` page is written or replaced. Use it when the recordings are published
separately: `uv run python scripts/regenerate_report_summaries.py --publish
data/results/<profile>/cpu/<suite>/<timestamp>` renders the selected runs and
writes `result/<profile>/cpu/<suite>.md` plus its `.summary.json`, and
`--check` verifies every published page against the runs its summary names.

Note: a full or smoke `run_all.sh` invocation **overwrites** the tracked
latest reports under `result/<target_profile>/`. If you only ran a smoke
subset, restore them before committing
(`git checkout -- result/`).

### CPU einsum + ops (Linux devcontainer)

Same runner, executed inside the devcontainer from the host:

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash -lc '
  BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh 1'
```

### Small-work public API suite (`cpu/small_work`)

Reuses the 154 cases from `feat/95-small-work` (`38b9a83`): F64
add/einsum/solve/gather/reduce_sum and C64 einsum. Twelve further f64 einsum
cases carry explicit `subscripts` and `operand_shapes` and guard tenferro-rs
CPU fixes: `abcd,dbef->acef` at extent 4 (#1897, canonical-fallback fixed
cost), `ax,asb->xsb` at D=4, s=2 (#1899, prepared einsum dispatch) and
`ij,jk->ik` with 95x95 times 95x1 (#1904, in-session tiny GEMM). Each runs as
prepared `execute` (`prepared-repeat`), prepared `execute_into` into a
preallocated destination (`prepared-into-repeat`), the plain in-session
`dot_general_read_into` (`dot-general-into-shared`) and, as a diagnostic,
ordinary einsum (`concrete-shared`). Shared/prepared operation routes use one clock interval for at least 1024 operations.
Fresh-session, eager and compiled per-call routes are opt-in diagnostics.
No cross-library equivalents or automatic performance gates are added.

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash -lc '
  TENFERRO_CPU_FEATURES=blas-mkl \
  BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_small_work.sh 1 4'

# Selected case (also overwrites the latest report; run the full suite last):
devcontainer exec --workspace-folder . bash -lc '
  TENFERRO_CPU_FEATURES=blas-mkl \
  BENCH_INSTANCE=add_f64_concrete_fresh ./scripts/run_small_work.sh 1'
```

Follow the existing checkout-freshness policy in [AGENTS.md](AGENTS.md).
For a worktree whose Git directory is not mounted in the container, pass
`--remote-env BENCHMARK_COMMIT="$(git rev-parse HEAD)"` to `devcontainer exec`.
The CLI can also be invoked as `npx --yes @devcontainers/cli`.

`BENCH_INSTANCE` accepts comma-separated case IDs; list them using
`python scripts/suite_instances.py --suite-file benchmarks/cpu/small_work.yaml --format lines`.
The suite and `cpu/session_matrix` run by default in `run_all.sh` calls
(disable with `RUN_SMALL_WORK_SUITE=0`). Standalone collection avoids
rerunning other suites.

Numerical values, solve residuals, and applicable AD gradients are checked
outside timing. The report records batch-normalized median/IQR in ns, CoV,
chain totals versus per-operation normalization, and timing boundaries.
Preparation is excluded from prepared execution rows and measured separately.
Noisy rows remain visible. Failed cases have no latency and the runner exits
nonzero. Raw samples and metadata are retained under
`data/results/<target_profile>/cpu/small_work/<timestamp>/`; the generated
latest report is `result/<target_profile>/cpu/small_work.md`.

### Performance-issue workloads (`cpu/perf_issues`, `gpu/perf_issues`)

Every open tenferro-rs performance issue gets its workload added as cases when
the issue is opened, with the issue number recorded next to each case
(`issues` in `data/instances/perf_issues.json` /
`data/instances/gpu_perf_issues.json`, or the suite description / instance
`intent` where an existing suite already covers it). A performance-fix PR
reports numbers from those cases. There is no automated audit, gate or
scheduled run. Each family carries its reference arm(s) (faer direct, host
sgemm, PyTorch, cudarc memcpy, the eager or batched counterpart, ...);
`reference_for` names the compared case. Cases are built only from public
tenferro-rs APIs (`CpuEngine` with a caller-owned `Pool`, a reused eager
session, and public concrete/eager/prepared operations). Steady-state rows,
per-call session-entry diagnostics, eager AD workflows, counter runs, GPU
allocation-state diagnostics and the GPU first-call diagnostic are reported in
separate sections.

Edit `scripts/generate_perf_issue_cases.py` (it also writes the #1865/#1863
`cpu/einsum` instances), rerun it and bump the manifest version. Collection:

```bash
devcontainer exec --workspace-folder . bash -lc '
  TENFERRO_CPU_FEATURES=blas-mkl BENCHMARK_TARGET_PROFILE=amd-cpu \
  ./scripts/run_perf_issues.sh 1 4'
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_perf_issues.sh'
```

`BENCH_COVERAGE=full` adds the diagnostics, `BENCH_EFFORT=scan` is a
low-repetition screen, `BENCH_INSTANCE` filters case IDs and
`PERF_ISSUES_CORRECTNESS_ONLY=1` executes and validates every case without
timing. Raw runs go to `data/results/<target_profile>/{cpu,gpu}/perf_issues/<timestamp>/`.

For benchmark issues #247/#252/#253, first run the selected cases with
`PERF_ISSUES_CORRECTNESS_ONLY=1` on an A100 in the CUDA devcontainer, then
repeat the same selection without that flag, sequentially. Set `BENCH_INSTANCE`
to the comma-separated literal case IDs (including references) from
`data/instances/gpu_perf_issues.json` whose `kind` is `linalg_readback_latency`,
`prepared_decode` or `standard_batched_matmul`.
These are latency rows. The runner rebuilds the binary and records
`GPU_BENCH_CLOCK_RAMP_MS` (default 300); fixture preparation, compilation,
validation and output readback are outside the operation timer. Reports remain
under `result/nvidia-gpu/gpu/perf_issues.md`.

| tenferro-rs issue | Cases | Suite |
|---|---|---|
| #1897, #1899, #1904 | generic einsum cases (`*_abcd-dbef-acef_d4`, `*_ax-asb-xsb_d4s2`, `*_ij-jk-ik_m95k95n1`) | `cpu/small_work` |
| #2000, #1884 | `batched_lu_factor` / `batched_lu_solve` 1024x{8,16}, `cpu/linalg_batch_families` | `cpu/public_api` |
| #1803 B/C | `batched_solve`, `*_backward` rows | `cpu/cpu_ops` |
| #2186 | prepared-trace `grad_sum_{qr,eigh,lu}_{jvp,vjp}`, `grad_sum_{svd_s,solve}_{jvp,vjp,backward}`, `grad_sum_matmul_backward`, `grad_sum_batched_{matmul,solve}_backward` rows (1/4-thread runs) | `cpu/cpu_ops` |
| #1865 | `bin_omeinsum_{matmul_10x10,batched_matmul_8x8_batch_4,high_d_12x12_contract_4_batch_4}` | `cpu/einsum` |
| #1863 | `bin_permuted_r4_abcd_dbef_acef_d32` (1/4(/8)-thread runs) | `cpu/einsum` |
| #1992, #2003, #1995 | `decode_proj_f32_*`, `decode_block_copy_volume_f32_d1024_len8` | `cpu/perf_issues` |
| #1900 | `gemm_{c64,f64}_mm256_t1` | `cpu/perf_issues` |
| #1615 | `conj_dot_{f64,c64}_*` | `cpu/perf_issues` |
| #2007 | `small_solve_f64_*` | `cpu/perf_issues` |
| #1803 A | `eager_backward_matmul2x2_f64_leaves*` | `cpu/perf_issues` |
| #1990 | `tanh_chain_f32_*` | `cpu/perf_issues` |
| #2006 (+ #2010 PR-B1 single call) | `{layer_norm,rms_norm}_f32_*` | `cpu/perf_issues` |
| #1975 (#2010 PR-B1) | `activation_{erf,sigmoid,silu,softplus,gelu,gelu_tanh}_f32_*` | `cpu/perf_issues` |
| #1976 (#2010 PR-B1) | `{softmax,log_softmax,masked_softmax}_f32_*`, `reduce_mean_f32_*` | `cpu/perf_issues` |
| #2008 (#2010 PR-B1) | `take_along_axis_rows_f64_*` | `cpu/perf_issues` |
| #1885 §4.1 | `small_contraction_abcd-dbef-acef_f64_*_d4` | `cpu/perf_issues` |
| #2180 | `metadata_transpose_view_f64_rank8_t*` and Julia metadata reference | `cpu/perf_issues` |
| #2197 | `prepared_dot_general_f32_decode_len8*` (prepared, eager, PyTorch) | `gpu/perf_issues` |
| #2193 | `standard_batched_matmul_f64_b16_32_cublas*` (trailing-batch fast path, batch-outer fallback, PyTorch) | `gpu/perf_issues` |
| #2192, #2204, #2195 | `{solve,qr}_readback_latency_f64_n*` and PyTorch references | `gpu/perf_issues` |
| #2009 | `transfer_{up,down}_f64_*` (the cudarc arms are an output-reuse lower bound: they reuse one destination while the tenferro arms allocate) | `gpu/perf_issues` |
| #1887 | `alloc_zero_f64_*` | `gpu/perf_issues` |
| #1885 §3.1-§3.4 | `small_blocks_gemm_*`, `batched_{qr,svd}_*`, `qr_live_buffers_*`, `first_call_transpose_*` | `gpu/perf_issues` |
| #2078 | `raw_scope_entry_f64_{empty-idle,empty-behind-queued-gemm,sync-behind-queued-gemm}` | `gpu/perf_issues` |
| #2079 | `eager_op_f32_{add,dot_general}_decode_*` | `gpu/perf_issues` |
| #2080 | `gather_embedding_f32_vocab{8192,65536,248320}_*` | `gpu/perf_issues` |
| #2081 | `pool_retention_f64_*` (allocation-state diagnostic) | `gpu/perf_issues` |
| #2082 | `reshape_constant_f32_64M_{constant,op-output}` | `gpu/perf_issues` |
| #2083 | `vendor_handle_cublas_f64_{create-destroy,reused}` | `gpu/perf_issues` |
| #1924 | `vendor_churn_f64_cycles{8,64}` (state diagnostic: process RSS per cycle, the leak the issue documents, plus device free bytes) | `gpu/perf_issues` |
| #1839, #1849 | `svd_driver_c64_n{64,256,1024}_{auto,gesvdj,gesvd,xgesvdp}` | `gpu/perf_issues` |
| #1846, #1852, #1923 | `eigh_batched_{f64,c64}_n{16,32,128}_{auto,syevd,syevj}` | `gpu/perf_issues` |
| #1898 | `hadamard_f32_1024x1024_{dot-general,multiply}` | `gpu/perf_issues` |
| #1873, #1805 | `contraction_plan_churn_f64_{distinct-plans,repeated-plan}` | `gpu/perf_issues` |
| #1949 | `dot_general_destination_f64_{owned-dst,borrowed-dst}` | `gpu/perf_issues` |
| #1925 | `borrowed_input_gemm_f64_n{8,32}_{owned-input,view-input}` | `gpu/perf_issues` |
| #2192 | `dense_solve_f64_512_rhs16` | `gpu/dense` |
| #2178 | `typed_elemwise_f64_add_{bcast,permuted_view}_counter`, `permute_materialize_f64_r4_counter`, `fused_elemwise_f64_serial_counter`, `reduce_axis_f64_counter` (allocation counters, 1/2 threads, host-loop references) | `cpu/perf_issues`, trend in `cpu/overhead_smoke` |

### Overhead smoke (`cpu/overhead_smoke`)

A small nightly trend view for the allocation-free provider work of
tenferro-rs #2178 (tenferro-benchmark #197): ten small public calls (stride-0
and permuted-view add, rank-4 transpose, traced fused `a*b+c` at backend
thread budgets 1 and 2, axis reduction, concatenate, prepared einsum at n=2/4)
on one run thread. Eight rows measure steady-state operations in entered
sessions; two separately named `run_prepared` diagnostics include runtime-internal
session entry/exit and are reported in a separate table. Each row reports ns/op
and allocs/op. Allocation counts come from the pooled steady state of the matching
`cpu/perf_issues` counter case; compare matching fixtures, thread budgets and
execution scopes across nights.
Nothing gates here (the gates live in each repository's PR CI).
`scripts/run_all.sh` runs it once after the other suites
(`RUN_OVERHEAD_SMOKE_SUITE=0` skips it); standalone:

```bash
devcontainer exec --workspace-folder . bash -lc '
  TENFERRO_CPU_FEATURES=blas-mkl BENCHMARK_TARGET_PROFILE=amd-cpu \
  ./scripts/run_overhead_smoke.sh'
BENCHMARK_TARGET_PROFILE=mac-cpu ./scripts/run_overhead_smoke.sh
```

Edit `scripts/generate_overhead_smoke_cases.py` and rerun it to change the
rows. Raw runs go to `data/results/<target_profile>/cpu/overhead_smoke/<timestamp>/`;
the latest report is `result/<target_profile>/cpu/overhead_smoke.md`.

### CPU permutation suite (`cpu/permutation`)

A standalone materialize/copy-kernel benchmark comparing tenferro-rs
`to_contiguous` against strided-rs, HPTT, Julia Base, and Strided.jl. An
internal untimed odometer implementation provides the correctness reference.
Every timed call includes fresh destination allocation for a common
end-to-end materialization comparison. Spec:
[docs/permutation-suite.md](docs/permutation-suite.md).

```bash
uv sync
./scripts/setup_extern_deps.sh
# Multiple thread counts are measured sequentially in one run.
# Without PERMUTATION_EXTRA_FEATURES=hptt the HPTT column is omitted (`-`);
# without `julia` on PATH the Julia columns are omitted. Tracked latest
# reports include both.
PERMUTATION_EXTRA_FEATURES=hptt \
BENCHMARK_TARGET_PROFILE=mac-cpu ./scripts/run_permutation.sh 1 4
```

For a quick trial run, the suite honors `PATTERN_ID` (restrict to one
pattern), `BENCH_RUNS`, and `BENCH_WARMUPS`:

```bash
BENCHMARK_TARGET_PROFILE=mac-cpu \
PATTERN_ID=transpose_2d_2048 \
BENCH_RUNS=1 \
BENCH_WARMUPS=0 \
  ./scripts/run_permutation.sh 1
```

This writes `result/<target_profile>/cpu/permutation.md`. Pattern definitions
live in `data/instances/permutation_patterns.json` and are read by both the
Rust and Julia runners; result records are validated against
`schemas/permutation-result.schema.json`. The suite is also included by
`./scripts/run_all.sh 1 4` unless `RUN_PERMUTATION_SUITE=0` is set.

### GPU suites (CUDA devcontainer)

The repo `.venv` is **shared between the CPU and CUDA devcontainers** (it
lives in the bind-mounted workspace). A CPU-side `uv sync` — including the
CPU devcontainer's own post-create hook — replaces the CUDA wheels with CPU
ones. The GPU Python runners then **silently skip** `pytorch-cuda` /
`jax-cuda` (they exit 0 and the report is simply missing those columns)
rather than failing. Before collecting GPU results, install the CUDA Python
backends inside the container and verify they see the GPU:

```bash
devcontainer up --workspace-folder . --config .devcontainer/cuda/devcontainer.json

# One-time, and again after ANY CPU-side `uv sync`:
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc '
    (uv sync --frozen || uv sync)
    uv pip install "torch>=2.12.0" --extra-index-url https://download.pytorch.org/whl/cu126
    uv pip install "jax[cuda12]"
    ./scripts/setup_extern_deps.sh
    uv run python -c "import torch, jax; assert torch.cuda.is_available(); jax.devices(\"cuda\"); print(\"CUDA OK:\", torch.__version__)"'
```

If `nvidia-smi` fails inside a previously created container ("Failed to
initialize NVML" / `CUDA_ERROR_NO_DEVICE`) while the host GPU is fine,
`docker restart <container>` usually restores GPU access — no rebuild needed.

The routine set is dense, einsum, sparse, tensor-network, elementwise and both
linalg AD suites; `docs/gpu-routine-coverage.md` lists every case and the row
cost.

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_suite.sh'

# One suite alone (a probe, not a routine refresh):
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_SUITE=benchmarks/gpu/linalg_jvp_vjp.yaml ./scripts/run_gpu_suite.sh'

# gpu/permutation (standalone, like cpu/permutation):
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_permutation.sh'
```

`gpu/permutation` honors the same quick-trial variables as the CPU suite
(`PATTERN_ID`, `BENCH_RUNS`, `BENCH_WARMUPS`), plus `GPU_BENCH_DEVICE` for
the CUDA ordinal:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu \
    PATTERN_ID=transpose_2d_2048 BENCH_RUNS=1 BENCH_WARMUPS=0 \
    ./scripts/run_gpu_permutation.sh'
```

After any GPU run, **check the generated report for the full backend set**
(the [Comparison Backends](#comparison-backends) section lists what each
suite compares; for `gpu/permutation` that is two tenferro columns, cuTENSOR,
PyTorch CUDA, JAX CUDA, and memcpy-d2d). A column that is `-` on every row
means that backend's runner skipped — usually the CUDA-wheels issue above.
Like `run_all.sh`, these scripts **overwrite** the tracked latest reports
under `result/nvidia-gpu/`; after a trial or partial run, restore them before
committing (`git checkout -- result/`).

The GPU tensor network benchmark uses problem data from
[`extern/TensorNetworkBenchmarks/`](extern/TensorNetworkBenchmarks/), based on
the upstream [TensorNetworkBenchmarks](https://github.com/TensorBFS/TensorNetworkBenchmarks)
repository; see [docs/tensornetwork-gpu.md](docs/tensornetwork-gpu.md).

## Measurement Policy

- Benchmarks are always run **sequentially** — never multiple benchmark
  processes at once, including different thread-count variants of the same
  suite (see [AGENTS.md](AGENTS.md)).
- Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` /
  `JULIA_NUM_THREADS`. CPU pinning is unavailable on macOS; on Linux the
  devcontainer convention applies no `taskset`/`numactl` pinning either. The
  effective thread environment is recorded in each run's `run.yaml`.
- Every run records provenance (target profile, suite, tenferro-rs commit,
  CPU/GPU info) in `data/results/.../run.yaml`.
- A provider's thread count does not prove which CPUs its threads may use:
  BLAS/LAPACK provider threads inherit the CPU mask of the thread that creates
  them. At least one run per lane records the observed per-thread
  `Cpus_allowed_list` groups with
  [`scripts/cpu_provider_affinity_check.py`](scripts/cpu_provider_affinity_check.py)
  (`--output FILE.json -- COMMAND`), which also reports whether the run was
  signalled. A 4-thread provider whose threads are all confined to one CPU is a
  configuration failure, not a result.

## Comparison Backends

CPU einsum reports compare `tenferro-trace`, `tenferro-eager`, `pytorch-cpu`,
`jax-cpu`, and `omeinsum-jl` (Julia; see [docs/einsum-suite.md](docs/einsum-suite.md#backends)).
GPU reports compare `tenferro-cuda-trace`,
`tenferro-cuda-eager`, `pytorch-cuda`, `jax-cuda`, and vendor-specific CUDA
backends where meaningful. The `cpu/permutation` suite has its own backend
set (tenferro-rs `to_contiguous`, HPTT, strided-rs, Julia Base, Strided.jl,
memcpy); `gpu/permutation` compares tenferro CUDA transpose paths
against cuTENSOR, PyTorch/JAX CUDA, and a device-to-device memcpy baseline.

The `cpu/public_api` suite additionally compares two Julia columns,
`julia-base` (natural Base/LinearAlgebra spellings, e.g. `permutedims!`,
`cholesky`, `eigen`, `gather`/`slice` indexing, `reshape`/`transpose`/`@view`
metadata-only views, broadcast-into/`mul!`/`copyto!` output reuse, and
complex Base/LinearAlgebra spellings) and `strided-jl` (natural
[Strided.jl](https://github.com/Jutho/Strided.jl) `@strided` fused-broadcast
spellings), populated only where each spelling naturally applies:
`strided-jl` covers the elementwise/chain/transpose rows, the elementwise
`cpu/output_reuse` `_into` rows, and the elementwise `cpu/complex` rows
(conj/mul/div/exp/log); reductions and dense linalg have no natural
Strided.jl spelling, so those stay `julia-base`-only. `julia-base` also now
covers `cpu/indexing_layout`, `cpu/view_metadata`, `cpu/output_reuse`, and
`cpu/complex`, with a few rows left missing where Julia Base has no natural
spelling: `pad` (edge padding), `broadcast_in_dim_view` (a zero-stride
broadcast array view), and `extract_diagonal` (batched diagonal extraction).
Julia is column-major like tenferro-rs, so these columns need no
PyTorch/JAX-style layout reconstruction to preserve the same logical fixture
values. **Attribution**: Strided.jl is prior art for tenferro-rs' strided-rs
kernel layer (both implement strided-array views and fused, cache-blocked
elementwise/permutation kernels); the `strided-jl` column exists to make that
lineage visible in the comparison, not merely to add another backend.

C++ Torch/LibTorch runners are intentionally removed; PyTorch Python is the
ATen comparison backend. The PyTorch CPU provider is detected at run time and
recorded in `run.yaml` and generated reports. The default Linux image uses a
binary PyTorch wheel; the separate [provider-matched OpenBLAS image](docs/linux-devcontainer.md#provider-matched-openblas-image)
source-builds PyTorch against the same OpenBLAS as tenferro.

### CPU provider pairs

A CPU result compares two provider stacks, and each side is fixed by a
different mechanism:

| Side | Selection | Linux default |
|---|---|---|
| tenferro | `TENFERRO_CPU_FEATURES` (`native`, `blas-openblas`, `blas-mkl`) | OpenBLAS from `OPENBLAS_ROOT` (`/opt/openblas`) |
| PyTorch CPU | Build-time choice inside the wheel | wheel-bundled Intel MKL |

Consequences to state in a report instead of assuming provider identity:

- A `blas-openblas` lane is **tenferro=OpenBLAS vs PyTorch=bundled MKL**, not
  OpenBLAS on both sides. The provider-matched image above is the only supported
  way to align them, and it applies to OpenBLAS only.
- A `blas-mkl` lane compares **different MKL builds**: the system oneAPI MKL
  that tenferro links against, and the older MKL inside the wheel (reported by
  `torch.__config__`). Both sides say "MKL" but they are not the same library
  or OpenMP runtime, and replacing the wheel's MKL by preloading the system one
  is not supported.
- Record both sides with `run.yaml` (tenferro features and BLAS implementation)
  plus the reference's own provider report (`torch.__config__` BLAS/MKL/OpenMP
  lines), so a reader can see the pair that was actually measured.

## Documentation

- [macOS CPU workflow](docs/macos-cpu.md): native run, no Docker, Accelerate.
- [Linux CPU devcontainer workflow](docs/linux-devcontainer.md):
  Docker/devcontainer, OpenBLAS default, optional oneMKL, detected PyTorch
  provider.
- [NVIDIA GPU devcontainer workflow](docs/gpu-devcontainer.md): CUDA
  devcontainer.
- [Einsum suite and instance selection](docs/einsum-suite.md): source
  benchmark, selection rules, diagnostic cases, path strategies.
- [CPU permutation suite](docs/permutation-suite.md): materialize-kernel
  benchmark spec (patterns, backends, fair-comparison guards).
- [GPU permutation suite](docs/gpu-permutation-suite.md): CUDA port of the
  permutation suite.
- [GPU tensor network suite](docs/tensornetwork-gpu.md):
  TensorNetworkBenchmarks parity on CUDA.
- [Result layout and metadata](docs/results.md): `target_profile`,
  `suite_id`, `run.yaml`, latest reports.
- [Architecture terminology](docs/architecture.md): suite, runner, backend,
  strategy, target profile.
- [PyTorch einsum dispatch notes](docs/pytorch-einsum-dispatch.md): PyTorch
  source investigation notes.

## Development Checks

Run these after changing benchmark scripts or schemas:

```bash
uv run python scripts/validate_benchmark_suite.py benchmarks/cpu/einsum.yaml benchmarks/cpu/permutation.yaml
uv run python scripts/validate_benchmark_suite.py benchmarks/gpu/dense.yaml benchmarks/gpu/einsum.yaml benchmarks/gpu/sparse.yaml benchmarks/gpu/tensornetwork.yaml
bash tests/test_suite_result_layout.sh
bash tests/test_run_all_docs_outputs.sh
bash tests/test_clean_extern_deps.sh
bash tests/test_setup_extern_tenferro_checkout.sh
bash tests/test_permutation_result_schema.sh
cmake -S cpp -B build/cpp-plan-test
cmake --build build/cpp-plan-test --target einsum_plan_test
ctest --test-dir build/cpp-plan-test --output-on-failure
```

## License

MIT
