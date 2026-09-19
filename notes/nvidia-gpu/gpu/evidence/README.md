# Evidence bundle (2026-09-19)

Small snapshots preserved for future checkouts; generated publication results
remain under `result/`. Archived patches below are **historical experimental
snapshots, not the final merged source**.

- [CubeCL final validation and merge](cubecl-pr-validation.md).
- [tenferro/Cubek integration validation and merge](tenferro-integration-validation.md).
- [Declared post-merge refresh protocol](integration-refresh-protocol.md).

- [Materialization protocol](materialization-protocol.md),
  [all paired medians/IQRs](materialization-summary.json),
  [source patch](materialization.patch), [original artifact hashes](materialization-hashes.txt).
  Apply the patch to tenferro `40e24634f9aa814db82efd9faae5e99a8fcee8c4`.
- [Sleep counterfactual protocol](sleep-protocol.md),
  [all paired results including regressions](sleep-summary.json),
  [kernel/profile summary](sleep-profiles.json). The only experimental change
  was CubeCL `SLEEP_STEP_SERVER` from 150 us to zero; do not adopt busy polling.
- [Notification design and second protocol](wake-design.md),
  [all paired results including regressions](wake-summary.json),
  [kernel/profile summary](wake-profiles.json),
  [source and test patch](wake.patch), [original artifact hashes](wake-hashes.txt).
  Apply this patch to Tensor4all's CubeCL fork at
  `5939d8e3b1c479ed6db31de5f01cb6d432178f2d`, **not** on top of the zero-sleep
  counterfactual. Upstream source is available at
  <https://github.com/tensor4all/cubecl>; its Apache-2.0/MIT licensing applies.
- [Small-case diagnostic protocol](small-protocol.md),
  [placement intervention protocol](placement-protocol.md),
  [all placement pairs and sampled CPU pairs](placement-summary.json).

Reproduction environment: `bench-gpu-20260919` CUDA container, A100 80GB PCIe,
CUDA toolkit 12.9, driver 580.126.09, Rust 1.97.1. Benchmark source `9a5b104`.
Release builds; Cargo `-j 16`; provider 1T via `scripts/thread_env.sh`.
CubeCL service threads are additional threads, not provider workers. Benchmark
execution includes device completion but excludes input/setup/download work.
Use `src/bin/benchmark_gpu_rust.rs` and suite YAML with the case lists and
warmup/sample counts declared in each protocol; small BMM adds batch=1 and 32
variants of `dense_batched_matmul_f64_b1024_256`, keeping m=n=k=256.

The protocols and original hash manifests retain their original local paths.
Those refer to `data/diagnostics/` in the investigation worktree, not to files
implicitly included here. Large executables, full Nsight/OSRT databases and
thread trajectories remain local and are **not** part of this portable bundle.
Summaries are retained evidence, not a substitute for obtaining fresh raw traces
when further attribution is needed. Do not assume the named container survives.

For the copy-count result, ten eager 1m chains retained 80 add/80 multiply/80
tanh kernels while permutation kernels changed 403 -> 3. This removes forty
per-expression copies while preserving three setup copies; profiled wall times
were not used for the performance comparison.
