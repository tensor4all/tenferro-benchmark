# A100 GPU refresh against tenferro-rs origin/main

## Scope and environment

Refreshed all eight maintained NVIDIA reports under `result/nvidia-gpu/gpu/`
using clean tenferro-rs `40e24634f9aa814db82efd9faae5e99a8fcee8c4`
(`origin/main` fetched before collection). The benchmark base is `dd129e7`.
The only benchmark code change replaces five obsolete `Tensor::F64` uses in
the permutation runner with the current typed-access API; timing boundaries
and workloads are unchanged.

Measurements ran sequentially in a separate CUDA devcontainer-image container
on the NVIDIA A100 80GB PCIe, driver 580.126.09, toolkit 12.9, Rust 1.97.1,
and PyTorch 2.12.0+cu130. CPU/provider thread environment was explicitly 1T;
`torch.get_num_threads()` returned 1. No CPU-affinity pinning was applied.
The existing Python environment and vendor volumes were reused read-only.
Container builds did not use the host's kache (whose metadata integrity check
still failed after repair).

Inputs, GPU uploads, runtimes and trace compilation/preparation are outside
sampling; host dispatch and device synchronization are inside. Public API
session entry may remain inside. Short cases use the existing single-call
sampling and are latency diagnostics, not GPU-throughput or batched
shared-session measurements. See each retained `run.yaml` for timing policy,
thread environment and software/hardware provenance.

## Evidence

Canonical JSONL and run metadata are retained under
`data/results/nvidia-gpu/gpu/<suite>/20260919_084000/`, except tensor-network,
which uses `20260919_083200/`. These are explicitly supplied run IDs; actual
collection timestamps are in `run.yaml`. Generated reports are the source of
truth, not duplicated here.

The first tensor-network attempt had a missing external-data link; all three
backends were rerun after restoring access to the pinned fixture. The rejected
attempt remains local at `tensornetwork/20260919_084000/` and is not used in the
published report. No successful timing row was selectively replaced.

The selected runs contain 267 records: 201 `ok`, 61 `unsupported`, and five
`not_configured`; no runtime failure, verification failure or OOM remains.
All `ok` records report passed verification/correctness. CUTLASS and Ginkgo
remain unavailable through their optional wrappers, as in the previous reports;
this refresh does not repair those integrations. The sparse tenferro backends
remain unsupported. GPU AD has trace and PyTorch columns, not eager AD.

Validation: three release GPU binaries built; GPU contract, linalg formatter
and runner, permutation destination-reuse and pattern tests passed. CPU
permutation schema, suite layout and documentation-output tests also passed.
Tensor-network contract and eight timing-boundary tests passed. Selected run
metadata and general JSONL passed the repository validators. GPU permutation
uses its separate legacy record layout (the CLI permutation schema is CPU-only):
all 44 rows are unique, successful, correctness-passed, with finite positive
medians and ordered quartiles, and reproduce the report exactly. Formatting
and diff whitespace checks passed. Existing upstream unsafe-operation warnings
remain; no library source was changed.

## Reproduction

In the CUDA container, with `extern/tenferro-rs` detached at the commit above,
`extern/strided-rs` at `dc0a8e03286c61a84d56446b5cc2c53295f75d76`, and the
tracked TensorNetworkBenchmarks fixture accessible:

```bash
export CARGO_BUILD_JOBS=16 MAX_JOBS=16 UV_NO_SYNC=1
export CUDA_HOME=/usr/local/cuda OPENBLAS_ROOT=/opt/openblas
export TENFERRO_CUTENSOR_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
export TENFERRO_CUBLAS_PATH=/usr/local/cuda/lib64/libcublas.so
export TENFERRO_CUSOLVER_PATH=/usr/local/cuda/lib64/libcusolver.so
source scripts/thread_env.sh
configure_cpu_thread_env 1
export BENCHMARK_TARGET_PROFILE=nvidia-gpu
export GPU_BENCH_TIMESTAMP=20260919_084000 BENCHMARK_TIMESTAMP=20260919_084000
./scripts/run_gpu_suite.sh
GPU_BENCH_SUITE=benchmarks/gpu/elementwise.yaml ./scripts/run_gpu_suite.sh
./scripts/run_gpu_linalg_jvp_vjp.sh
./scripts/run_gpu_linalg_ad_latency.sh
./scripts/run_gpu_permutation.sh
GPU_BENCH_TIMESTAMP=20260919_083200 \
  GPU_BENCH_SUITE=benchmarks/gpu/tensornetwork.yaml ./scripts/run_gpu_suite.sh
```

For the requested follow-up comparison, use matching successful problem/backend
rows and the ratio `tenferro median / PyTorch median`, with 2x as the descriptive
threshold for a conspicuous slowdown. Separate the small latency suite from
GPU-sized workloads; the threshold is not a statistical-significance test.
