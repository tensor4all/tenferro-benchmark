# Focused Einsum Lazy Value Retest

- Date: `2026-06-05`
- Threads: `1`
- tenferro-rs commit: `a41744e07ecc0168fdee94155cad04599dd376b0`
- tenferro CPU features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- Timing: median +/- IQR of 15 runs, 3 warmups
- Scope: `str_nw_mera_open_26` and final-permutation-heavy TensorNetworkBenchmarks cases
- Note: `tenferro-trace` uses the `TensorValue` output API so terminal transpose/permutation outputs can remain lazy owned views.

## Results

#### Strategy: opt_flops

Median +/- IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| str_nw_mera_open_26 | 134.145 +/- 0.725 | 191.657 +/- 1.824 | 127.356 +/- 1.664 | **67.188 +/- 0.976** |
| tensornetwork_permutation_focus_step409_316 | 119.190 +/- 1.167 | 191.778 +/- 9.452 | 158.892 +/- 2.225 | **75.918 +/- 6.546** |
| tensornetwork_permutation_light_415 | 120.966 +/- 0.986 | 156.441 +/- 4.003 | 139.257 +/- 1.348 | **82.913 +/- 8.711** |

#### Strategy: opt_size

Median +/- IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU dot) (ms) |
|---|---:|---:|---:|---:|
| str_nw_mera_open_26 | 136.873 +/- 0.517 | 194.141 +/- 2.540 | 132.982 +/- 0.898 | **73.837 +/- 2.031** |
| tensornetwork_permutation_focus_step409_316 | 119.190 +/- 1.167 | 191.778 +/- 9.452 | 158.892 +/- 2.225 | **75.918 +/- 6.546** |
| tensornetwork_permutation_light_415 | 120.966 +/- 0.986 | 156.441 +/- 4.003 | 139.257 +/- 1.348 | **82.913 +/- 8.711** |

## Raw Logs

- `data/results/tenferro_trace_t1_lazyvalue_str_nw_mera_open_26_20260605_202937.log`
- `data/results/tenferro_eager_t1_lazyvalue_str_nw_mera_open_26_20260605_202937.log`
- `data/results/pytorch_cpu_t1_lazyvalue_str_nw_mera_open_26_20260605_202937.log`
- `data/results/jax_cpu_t1_lazyvalue_str_nw_mera_open_26_20260605_202937.log`
- `data/results/tenferro_trace_t1_lazyvalue_tensornetwork_permutation_light_415_20260605_203007.log`
- `data/results/tenferro_eager_t1_lazyvalue_tensornetwork_permutation_light_415_20260605_203007.log`
- `data/results/pytorch_cpu_t1_lazyvalue_tensornetwork_permutation_light_415_20260605_203007.log`
- `data/results/jax_cpu_t1_lazyvalue_tensornetwork_permutation_light_415_20260605_203007.log`
- `data/results/tenferro_trace_t1_lazyvalue_tensornetwork_permutation_focus_step409_316_20260605_203019.log`
- `data/results/tenferro_eager_t1_lazyvalue_tensornetwork_permutation_focus_step409_316_20260605_203019.log`
- `data/results/pytorch_cpu_t1_lazyvalue_tensornetwork_permutation_focus_step409_316_20260605_203019.log`
- `data/results/jax_cpu_t1_lazyvalue_tensornetwork_permutation_focus_step409_316_20260605_203019.log`
