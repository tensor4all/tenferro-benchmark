# Benchmark Architecture

## Terms

`suite`
: A workload collection identified by `suite_id`, such as `cpu/einsum` or
`gpu/dense`. A suite says what to measure.

`target_profile`
: A hardware/software bucket for latest reports. Current profiles are
`mac-cpu`, `amd-cpu`, and `nvidia-gpu`.

`runner`
: A script or binary that measures one implementation family. Examples:
`scripts/run_all_rust.sh`, `scripts/benchmark_python.py`, and
`src/bin/benchmark_gpu_rust.rs`.

`backend`
: A measured implementation column, such as `tenferro-eager`, `pytorch-cpu`,
or `jax-cuda`.

`strategy`
: An einsum contraction-path strategy, such as `opt_flops` or `opt_size`.
Strategies are comparable only when the same instance and same path semantics
are used.

`run.yaml`
: Run-level metadata for one suite execution. It records target profile,
environment, tenferro commit and dirty state, BLAS/provider information, and
suite identity.

`report.md`
: The run-local human report for one timestamp.

`result/<target_profile>/...`
: The tracked latest report for one target profile and suite. It is overwritten
by a new run for the same target profile.

## Backend Policy

CPU:

- `tenferro-trace`
- `tenferro-eager`
- `pytorch-cpu`
- `jax-cpu`

GPU:

- `tenferro-cuda-trace`
- `tenferro-cuda-eager`
- `pytorch-cuda`
- `jax-cuda`
- vendor-specific CUDA backends

C++ Torch/LibTorch is not a backend in this repository. PyTorch Python is the
ATen comparison backend.
