# Einsum Benchmark Results

Latest run: `./scripts/run_all.sh 4`.

This file collects the latest einsum benchmark table for each thread count under `data/results/`.

- tenferro-rs commit: `a5364e224ebed721cc0711c9ab16e47f7faaa2e5`

## Threads: 1

- Timestamp: `20260526_180729`
- Source table: `data/results/results_t1_20260526_180729.md`

Logs:

- `data/results/tenferro_trace_t1_20260526_180729.log`
- `data/results/tenferro_eager_t1_20260526_180729.log`
- `data/results/libtorch_cpu_t1_20260526_180729.log`
- `data/results/pytorch_cpu_t1_20260526_180729.log`
- `data/results/jax_cpu_t1_20260526_180729.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_matmul_256 | 1.723 ± 0.518 | 1.645 ± 0.495 | 0.722 ± 0.056 | **0.112 ± 0.003** | 0.309 ± 0.042 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_matmul_256 | 1.067 ± 0.108 | 1.067 ± 0.137 | 0.765 ± 0.063 | **0.111 ± 0.001** | 0.323 ± 0.031 |


## Threads: 4

- Timestamp: `20260526_180925`
- Source table: `data/results/results_t4_20260526_180925.md`

Logs:

- `data/results/tenferro_trace_t4_20260526_180925.log`
- `data/results/tenferro_eager_t4_20260526_180925.log`
- `data/results/libtorch_cpu_t4_20260526_180925.log`
- `data/results/pytorch_cpu_t4_20260526_180925.log`
- `data/results/jax_cpu_t4_20260526_180925.log`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_matmul_256 | 0.838 ± 0.159 | 0.675 ± 0.302 | 0.254 ± 0.010 | **0.112 ± 0.003** | 0.312 ± 0.073 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---:|---:|---:|---:|---:|
| bin_matmul_256 | 0.568 ± 0.125 | 0.547 ± 0.066 | 0.253 ± 0.005 | **0.112 ± 0.002** | 0.302 ± 0.025 |

