# macOS CPU Workflow

macOS CPU benchmarks run natively. Do not use Docker for the standard Mac CPU
path.

Target profile:

```bash
export BENCHMARK_TARGET_PROFILE=mac-cpu
```

BLAS policy:

- tenferro BLAS-backed runs use `system-accelerate`.
- PyTorch uses the installed Python wheel provider and records it in metadata.
- JAX is reported as XLA CPU, not as an Accelerate/OpenBLAS/MKL row.

Setup:

```bash
uv sync
./scripts/setup_extern_deps.sh
```

cmake is not needed for this workflow. It is only required when opting into
the `hptt` Cargo feature of the cpu/permutation suite (`brew install cmake`);
see [permutation-suite.md](permutation-suite.md).

Quick smoke:

```bash
BENCHMARK_TARGET_PROFILE=mac-cpu \
BENCH_INSTANCE=bin_matmul_256 \
BENCH_RUNS=1 \
BENCH_WARMUPS=0 \
PUBLICATION_GATE_SUITE=small \
  ./scripts/run_all.sh 1
```

Normal single-thread run:

```bash
BENCHMARK_TARGET_PROFILE=mac-cpu ./scripts/run_all.sh 1
```

Normal multi-thread run:

```bash
BENCHMARK_TARGET_PROFILE=mac-cpu ./scripts/run_all.sh 4
```

Latest reports:

- `result/mac-cpu/cpu/einsum.md`
- `result/mac-cpu/cpu/cpu_ops.md`
- `result/mac-cpu/cpu/linalg_jvp_vjp.md`

Raw run data:

- `data/results/mac-cpu/cpu/einsum/<timestamp>/run.yaml`
- `data/results/mac-cpu/cpu/einsum/<timestamp>/report.md`

