# CPU large-matrix eager AD: gap survey and fixes

- Suite: `cpu/large_ad`
- Target profile: `amd-cpu`
- Provider: system MKL, one backend worker thread, `MKL_NUM_THREADS=1`
- Affinity: pinned to CPU 1 for every row
- Thread budget: 1
- tenferro-rs revision measured: `5782859f` (the tree that became #1828), plus the
  `#1835` eigvalsh change measured separately after it merged
- Reference: PyTorch 2.12.0+cpu, `torch.set_num_threads(1)`

## Commands

```sh
cargo build -j 16 --release --features system-mkl --bin publication_gate

export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/compiler/latest/lib
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_DYNAMIC=FALSE
export PUBLICATION_GATE_PROFILE=full PUBLICATION_GATE_SUITE=large \
       PUBLICATION_GATE_TENFERRO_MODE=both BENCH_RUNS=15 BENCH_WARMUPS=3

# Interleave the two backends per repetition so the reference is not inflated by drift.
for rep in 1 2 3; do
  taskset -c 1 target/release/publication_gate > fin$rep-rust.csv 2> fin$rep-rust.stderr
  taskset -c 1 .venv/bin/python scripts/benchmark_cpu_ops_python.py \
    --backend pytorch-cpu --num-threads 1 --output fin$rep-torch.csv
done
```

The `full` profile now covers 64/128/256/512/1024 for the large-matrix primal and
AD rows in both the Rust gate and the Python runner; that size extension is part of
this change.

## Scope

Timings are **forward + sum + backward** for the eager rows, not backward alone.
Eager includes its own recording/AD preparation inside the timing; the prepared
trace builds and compiles the derivative graph outside the timing and the timed
region is one `run_prepared` call. Input construction and sessions are outside
timing in both cases, and outputs are retained until the clock stops. Cross-backend
ratios come from interleaved runs, because two sequential suite runs drift more than
the effects measured here.

## Results

Median of three interleaved repetitions, 15 samples and 3 warmups each.

| operation | phase | shape | eager ms | trace ms | Torch ms | eager/Torch |
|---|---|---|---:|---:|---:|---:|
| `matmul` | primal | 128x128 | 0.155 | 0.146 | 0.200 | 0.77x |
| `matmul` | primal | 256x256 | 1.119 | 0.786 | 1.089 | 1.03x |
| `matmul` | primal | 512x512 | 7.442 | 5.656 | 6.941 | 1.07x |
| `matmul` | primal | 1024x1024 | 53.912 | 43.147 | 53.226 | 1.01x |
| `matmul_rect` | primal | 256x1024 * 1024x256 | 3.381 | 3.472 | 3.419 | 0.99x |
| `matmul_rect` | primal | 1024x256 * 256x1024 | 12.157 | 12.317 | 13.194 | 0.92x |
| `svd` | primal | 64x64 | 0.626 | 0.614 | 0.649 | 0.96x |
| `svd` | primal | 128x128 | 2.626 | 2.611 | 2.529 | 1.04x |
| `svd` | primal | 256x256 | 13.943 | 11.918 | 14.707 | 0.95x |
| `svd` | primal | 512x512 | 91.391 | 78.582 | 96.974 | 0.94x |
| `svd` | primal | 1024x1024 | 496.685 | 531.127 | 573.071 | 0.87x |
| `qr` | primal | 64x64 | 0.111 | 0.108 | 0.112 | 0.99x |
| `qr` | primal | 128x128 | 0.558 | 0.464 | 0.584 | 0.96x |
| `qr` | primal | 256x256 | 3.182 | 2.655 | 3.356 | 0.95x |
| `qr` | primal | 512x512 | 21.731 | 18.863 | 23.547 | 0.92x |
| `qr` | primal | 1024x1024 | 120.038 | 117.875 | 128.080 | 0.94x |
| `eigh` | primal | 64x64 | 0.321 | 0.278 | 0.330 | 0.97x |
| `eigh` | primal | 128x128 | 1.390 | 1.163 | 1.333 | 1.04x |
| `eigh` | primal | 256x256 | 7.648 | 6.351 | 7.522 | 1.02x |
| `eigh` | primal | 512x512 | 40.155 | 40.882 | 45.367 | 0.89x |
| `eigh` | primal | 1024x1024 | 297.875 | 251.507 | 259.070 | 1.15x |
| `solve` | primal | 64x64,rhs=16 | 0.071 | 0.097 | 0.066 | 1.08x |
| `solve` | primal | 64x64,rhs=1 | 0.059 | 0.074 | 0.048 | 1.21x |
| `solve` | primal | 64x64,rhs=64 | 0.125 | 0.149 | 0.121 | 1.04x |
| `solve` | primal | 128x128,rhs=64 | 0.339 | 0.304 | 0.340 | 1.00x |
| `solve` | primal | 128x128,rhs=16 | 0.195 | 0.197 | 0.216 | 0.90x |
| `solve` | primal | 128x128,rhs=1 | 0.175 | 0.142 | 0.150 | 1.16x |
| `solve` | primal | 256x256,rhs=16 | 0.838 | 0.815 | 0.950 | 0.88x |
| `solve` | primal | 256x256,rhs=1 | 0.683 | 0.733 | 0.801 | 0.85x |
| `solve` | primal | 256x256,rhs=64 | 1.213 | 1.032 | 1.280 | 0.95x |
| `solve` | primal | 512x512,rhs=16 | 4.160 | 4.553 | 5.557 | 0.75x |
| `solve` | primal | 512x512,rhs=64 | 5.422 | 4.890 | 6.705 | 0.81x |
| `solve` | primal | 512x512,rhs=1 | 3.558 | 4.255 | 5.183 | 0.69x |
| `solve` | primal | 1024x1024,rhs=1 | 24.992 | 22.617 | 36.142 | 0.69x |
| `solve` | primal | 1024x1024,rhs=16 | 25.127 | 23.971 | 37.328 | 0.67x |
| `solve` | primal | 1024x1024,rhs=64 | 29.266 | 26.953 | 41.021 | 0.71x |
| `grad_sum_matmul` | primal | 64x64 | 0.046 | 0.037 | 0.027 | 1.69x |
| `grad_sum_matmul` | primal | 128x128 | 0.171 | 0.160 | 0.142 | 1.20x |
| `grad_sum_matmul` | primal | 256x256 | 0.944 | 0.783 | 0.906 | 1.04x |
| `grad_sum_matmul` | primal | 512x512 | 6.725 | 6.548 | 6.944 | 0.97x |
| `grad_sum_matmul` | primal | 1024x1024 | 51.614 | 46.676 | 51.012 | 1.01x |
| `grad_sum_matmul_backward` | backward | 64x64 | 0.320 | 0.129 | 0.138 | 2.33x |
| `grad_sum_matmul_backward` | backward | 128x128 | 0.687 | 0.505 | 0.479 | 1.44x |
| `grad_sum_matmul_backward` | backward | 256x256 | 2.993 | 2.339 | 2.768 | 1.08x |
| `grad_sum_matmul_backward` | backward | 512x512 | 18.786 | 19.508 | 20.488 | 0.92x |
| `grad_sum_matmul_backward` | backward | 1024x1024 | 141.814 | 147.108 | 151.254 | 0.94x |
| `grad_sum_svd_s_backward` | backward | 64x64 | 0.884 | 0.609 | 0.787 | 1.12x |
| `grad_sum_svd_s_backward` | backward | 128x128 | 3.191 | 2.644 | 3.102 | 1.03x |
| `grad_sum_svd_s_backward` | backward | 256x256 | 15.972 | 13.328 | 16.301 | 0.98x |
| `grad_sum_svd_s_backward` | backward | 512x512 | 94.485 | 96.003 | 106.058 | 0.89x |
| `grad_sum_svd_s_backward` | backward | 1024x1024 | 575.980 | 577.247 | 639.680 | 0.90x |
| `grad_sum_solve_backward` | backward | 64x64,rhs=1 | 0.430 | 0.174 | 0.144 | 2.97x |
| `grad_sum_solve_backward` | backward | 128x128,rhs=1 | 0.614 | 0.285 | 0.278 | 2.21x |
| `grad_sum_solve_backward` | backward | 256x256,rhs=1 | 1.294 | 0.830 | 0.975 | 1.33x |
| `grad_sum_solve_backward` | backward | 512x512,rhs=1 | 5.153 | 4.253 | 5.517 | 0.93x |
| `grad_sum_solve_backward` | backward | 1024x1024,rhs=1 | 40.472 | 25.401 | 35.154 | 1.15x |
| `grad_sum_svd_s_jvp` | jvp | 256x256 | - | 13.290 | 19.309 | - |
| `grad_sum_svd_s_jvp` | jvp | 512x512 | - | 89.714 | 125.656 | - |
| `grad_sum_svd_s_jvp` | jvp | 1024x1024 | - | 569.768 | 846.715 | - |
| `grad_sum_svd_s_vjp` | vjp | 256x256 | - | 13.824 | 16.728 | - |
| `grad_sum_svd_s_vjp` | vjp | 512x512 | - | 92.648 | 102.852 | - |
| `grad_sum_svd_s_vjp` | vjp | 1024x1024 | - | 583.408 | 610.775 | - |
| `grad_sum_qr_jvp` | jvp | 256x256 | - | 6.767 | 7.754 | - |
| `grad_sum_qr_jvp` | jvp | 512x512 | - | 47.922 | 51.488 | - |
| `grad_sum_qr_jvp` | jvp | 1024x1024 | - | 307.620 | 359.045 | - |
| `grad_sum_qr_vjp` | vjp | 256x256 | - | 6.489 | 7.677 | - |
| `grad_sum_qr_vjp` | vjp | 512x512 | - | 46.586 | 49.181 | - |
| `grad_sum_qr_vjp` | vjp | 1024x1024 | - | 307.302 | 360.033 | - |
| `grad_sum_eigh_jvp` | jvp | 256x256 | - | 8.145 | 10.675 | - |
| `grad_sum_eigh_jvp` | jvp | 512x512 | - | 62.156 | 63.780 | - |
| `grad_sum_eigh_jvp` | jvp | 1024x1024 | - | 378.166 | 463.654 | - |
| `grad_sum_eigh_vjp` | vjp | 256x256 | - | 9.688 | 8.786 | - |
| `grad_sum_eigh_vjp` | vjp | 512x512 | - | 58.568 | 50.272 | - |
| `grad_sum_eigh_vjp` | vjp | 1024x1024 | - | 379.309 | 332.450 | - |
| `grad_sum_lu_jvp` | jvp | 256x256 | - | 4.664 | 10.344 | - |
| `grad_sum_lu_jvp` | jvp | 512x512 | - | 31.661 | 69.995 | - |
| `grad_sum_lu_jvp` | jvp | 1024x1024 | - | 241.292 | 555.989 | - |
| `grad_sum_lu_vjp` | vjp | 256x256 | - | 4.534 | 5.525 | - |
| `grad_sum_lu_vjp` | vjp | 512x512 | - | 34.715 | 38.451 | - |
| `grad_sum_lu_vjp` | vjp | 1024x1024 | - | 224.337 | 322.493 | - |
| `grad_sum_solve_jvp` | jvp | 256x256,rhs=1 | - | 0.823 | 1.249 | - |
| `grad_sum_solve_jvp` | jvp | 512x512,rhs=1 | - | 4.519 | 6.462 | - |
| `grad_sum_solve_jvp` | jvp | 1024x1024,rhs=1 | - | 25.405 | 36.689 | - |
| `grad_sum_solve_vjp` | vjp | 256x256,rhs=1 | - | 0.848 | 1.744 | - |
| `grad_sum_solve_vjp` | vjp | 512x512,rhs=1 | - | 4.503 | 10.171 | - |
| `grad_sum_solve_vjp` | vjp | 1024x1024,rhs=1 | - | 28.978 | 66.182 | - |

## Provider floor

Measured inside the benchmark binary with an `LD_PRELOAD` probe at n=1024, so the
kernel's own overhead is separable from its LAPACK call:

| call | real decomposition | row | row/call |
|---|---:|---:|---:|
| `dgesdd_` (svd) | 500.3 ms | 497.9 ms eager / 529.9 ms trace | within noise |
| `dsyevd_` (eigh) | 266.9 ms | 273.9 ms eager / 265.6 ms trace | +2.6% / -0.5% |
| `dgetrf_` (solve) | 21.6 ms | 25.0 ms eager (includes `dgetrs`) | - |

Every primal row is within about 3% of the provider call, so the remaining cost is
the decomposition itself. `dsyevr` (MRRR) is not an alternative here: 373.6 ms
against 292.0 ms for `dsyevd` (divide and conquer) at n=1024 for all eigenpairs.

## Changes that came out of this survey

| change | measured effect at 1024 |
|---|---|
| eager reverse mode binds declared residuals instead of replaying producer graphs (#1828) | eager SVD forward+backward 1047.8 -> 576.5 ms; decomposition count 2 -> 1 per workflow |
| tracked eager solve reuses saved LU/pivots and the saved solution (#1828) | eager solve forward+backward 101.4 -> ~40 ms; `dgetrf` count 3 -> 1 per workflow |
| eigvalsh pulls the eigenvalue cotangent back with one matmul instead of two (#1835) | `grad_sum_eigh` jvp 1.10x, vjp 1.20x; 1024-class `dgemm` 2 -> 1 per workflow |
| LAPACK scratch comes from the session buffer pool (#1831, not from this survey's branch) | LU overhead above the `dgetrf` floor 6.6 -> ~1 ms |

The eigvalsh pair, interleaved A/B with both rows in one process:

| row | before ms | after ms |
|---|---:|---:|
| `grad_sum_eigh` jvp | 358.5 / 365.4 | 321.0 / 326.6 / 335.2 |
| `grad_sum_eigh` vjp | 381.3 / 368.3 | 312.3 / 304.1 / 329.3 |

## Remaining, not addressed

- Small-size eager AD rows still carry a fixed per-backward cost (64x64 solve
  backward 2.97x and matmul backward 2.33x versus Torch, about 0.2-0.3 ms in
  absolute terms). This is latency, not large-matrix throughput.
- Eager rows remain marginally slower than their own prepared-trace counterparts.
  A dedicated clean-context probe of the n=1024 solve row measured eager 27.0-27.9 ms
  against trace 24.6 ms, and running both linalg operations inside one entered
  session was not faster than two separate entries once the pooled workspace was
  warm. The much larger eager solve numbers that appear inside a full suite run
  (up to about 45 ms) are a memory/buffer-pool context effect, so that row's
  absolute value is not a stable artifact.
- The eigenvalue-only workspace zeroing suspected during the eigvalsh analysis was
  measured at 0.332 ms for the 16.8 MB `dsyevd` workspace and left unchanged.

## Evidence

Raw samples, protocol scripts, provider probes and source snapshots are kept under
`data/results/amd-cpu/cpu/large-ad/` in the benchmark checkout; that tree is
git-ignored by policy. Files of interest: `fin{1,2,3}-{rust,torch}.{csv,jsonl}`,
`eighAB-{beforeA,beforeB,afterA,afterB,afterC}.{csv,stderr}`, `sym-floor.{csv,stderr}`,
`eigh_ab.sh`, `gemm_probe.c`, `sym_probe.c`, `syev_compare.c`, `getrf_probe.c`.

## Post-fix multi-thread (1T/4T) provider survey

The 1T survey above remains valid: with one thread the domain is a single CPU,
so worker affinity and provider threading are identical before and after the
change below. This section records the multi-thread case, which is what exposed
the problem and what changed.

**Change measured:** tenferro-rs `67fdd861` (*fix(cpu): confine workers to the
domain CPU set*, PR #1847). Before it, each Rayon worker was pinned to one CPU;
BLAS/LAPACK provider threads inherit the creating thread's mask, so an MKL
provider team was confined to that single CPU. After it, every worker is
confined to the *whole* domain CPU set, which provider threads inherit.

**Protocol (declared before the run):** suite `cpu/large_ad`, profile `full`,
`PUBLICATION_GATE_TENFERRO_MODE=both`, 15 samples / 3 warmups, one paired
repetition, runs strictly sequential. 1T uses `taskset -c 10`; 4T uses
`taskset -c 10-13` with `RAYON_NUM_THREADS` and the lane's provider thread count
set to 4. Lanes: `system-openblas` (Linux default), `system-mkl`, `cpu-faer`;
PyTorch reference from the wheel at the same thread counts. Every run exited 0
and produced a stable row count (142 tenferro rows, 86 torch rows).

Measured on the same EPYC 7713P host as the 1T survey, with other users'
background load present; treat small ratios as indicative and the >2x ratios as
structural.

## Tenferro (median over repetitions)

| lane | op | phase | shape | backend | 1T ms | 4T ms | 1T/4T |
|---|---|---|---|---|---:|---:|---:|
| openblas | matmul | primal | 1024x1024 | system-openblas | 46.7190 | 15.4292 | 3.03x |
| openblas | matmul | primal | 1024x1024 | tenferro-trace | 43.6013 | 13.4306 | 3.25x |
| openblas | solve | primal | 1024x1024,rhs=1 | system-openblas | 19.9311 | 9.0047 | 2.21x |
| openblas | solve | primal | 1024x1024,rhs=1 | tenferro-trace | 20.4275 | 9.8173 | 2.08x |
| openblas | svd | primal | 512x512 | system-openblas | 76.3302 | 61.7716 | 1.24x |
| openblas | svd | primal | 512x512 | tenferro-trace | 74.1779 | 61.0661 | 1.21x |
| openblas | qr | primal | 1024x1024 | system-openblas | 108.7955 | 60.4444 | 1.80x |
| openblas | qr | primal | 1024x1024 | tenferro-trace | 104.4175 | 60.7791 | 1.72x |
| openblas | eigh | primal | 512x512 | system-openblas | 33.8549 | 25.4732 | 1.33x |
| openblas | eigh | primal | 512x512 | tenferro-trace | 32.0955 | 24.9880 | 1.28x |
| openblas | grad_sum_matmul | backward | 1024x1024 | system-openblas | 134.8579 | 47.7888 | 2.82x |
| openblas | grad_sum_matmul | backward | 1024x1024 | tenferro-trace | 131.5840 | 46.0072 | 2.86x |
| openblas | grad_sum_solve | backward | 512x512,rhs=1 | system-openblas | 4.3164 | 6.2420 | 0.69x |
| openblas | grad_sum_solve | backward | 512x512,rhs=1 | tenferro-trace | 4.0845 | 3.9638 | 1.03x |
| mkl | matmul | primal | 1024x1024 | system-mkl | 45.3613 | 14.3271 | 3.17x |
| mkl | matmul | primal | 1024x1024 | tenferro-trace | 42.5349 | 14.1502 | 3.01x |
| mkl | solve | primal | 1024x1024,rhs=1 | system-mkl | 21.1182 | 8.1394 | 2.59x |
| mkl | solve | primal | 1024x1024,rhs=1 | tenferro-trace | 23.1322 | 9.0823 | 2.55x |
| mkl | svd | primal | 512x512 | system-mkl | 77.8446 | 49.6160 | 1.57x |
| mkl | svd | primal | 512x512 | tenferro-trace | 76.2904 | 48.6871 | 1.57x |
| mkl | qr | primal | 1024x1024 | system-mkl | 110.6092 | 44.3276 | 2.50x |
| mkl | qr | primal | 1024x1024 | tenferro-trace | 106.4600 | 44.6143 | 2.39x |
| mkl | eigh | primal | 512x512 | system-mkl | 39.8198 | 18.0857 | 2.20x |
| mkl | eigh | primal | 512x512 | tenferro-trace | 39.5824 | 17.6730 | 2.24x |
| mkl | grad_sum_matmul | backward | 1024x1024 | system-mkl | 134.3094 | 49.7724 | 2.70x |
| mkl | grad_sum_matmul | backward | 1024x1024 | tenferro-trace | 135.4312 | 45.2106 | 3.00x |
| mkl | grad_sum_solve | backward | 512x512,rhs=1 | system-mkl | 4.3443 | 4.4488 | 0.98x |
| mkl | grad_sum_solve | backward | 512x512,rhs=1 | tenferro-trace | 4.0255 | 2.5872 | 1.56x |
| faer | matmul | primal | 1024x1024 | cpu-faer | 46.0752 | 16.2134 | 2.84x |
| faer | matmul | primal | 1024x1024 | tenferro-trace | 43.3313 | 12.9339 | 3.35x |
| faer | solve | primal | 1024x1024,rhs=1 | cpu-faer | 23.4754 | 16.1267 | 1.46x |
| faer | solve | primal | 1024x1024,rhs=1 | tenferro-trace | 21.2965 | 12.1594 | 1.75x |
| faer | svd | primal | 512x512 | cpu-faer | 67.1881 | 46.6861 | 1.44x |
| faer | svd | primal | 512x512 | tenferro-trace | 73.1089 | 47.0378 | 1.55x |
| faer | qr | primal | 1024x1024 | cpu-faer | 99.7632 | 52.6982 | 1.89x |
| faer | qr | primal | 1024x1024 | tenferro-trace | 83.3347 | 35.5487 | 2.34x |
| faer | eigh | primal | 512x512 | cpu-faer | 31.9801 | 22.6534 | 1.41x |
| faer | eigh | primal | 512x512 | tenferro-trace | 34.6162 | 20.5102 | 1.69x |
| faer | grad_sum_matmul | backward | 1024x1024 | cpu-faer | 153.7467 | 48.9755 | 3.14x |
| faer | grad_sum_matmul | backward | 1024x1024 | tenferro-trace | 164.3217 | 46.8266 | 3.51x |
| faer | grad_sum_solve | backward | 512x512,rhs=1 | cpu-faer | 4.1583 | 4.0788 | 1.02x |
| faer | grad_sum_solve | backward | 512x512,rhs=1 | tenferro-trace | 4.0327 | 3.2117 | 1.26x |

## Provider-matched torch reference (median over repetitions)

| op | phase | shape | 1T ms | 4T ms | 1T/4T |
|---|---|---|---:|---:|---:|
| matmul | primal | 1024x1024 | 47.8838 | 21.3867 | 2.24x |
| solve | primal | 1024x1024,rhs=1 | 30.6087 | 11.2586 | 2.72x |
| svd | primal | 512x512 | 80.8440 | 63.4895 | 1.27x |
| qr | primal | 1024x1024 | 116.4051 | 53.8617 | 2.16x |
| eigh | primal | 512x512 | 38.2568 | 21.4116 | 1.79x |
| grad_sum_matmul | backward | 1024x1024 | 138.5150 | 55.1711 | 2.51x |
| grad_sum_solve | backward | 512x512,rhs=1 | 4.6684 | 2.8643 | 1.63x |

## Run validity

- all runs exited 0 with a stable row count

### Affinity guard (4T, one run per lane)

`scripts/cpu_provider_affinity_check.py` records `Cpus_allowed_list` per thread
while a run executes. A provider thread count does not prove where the threads
may run, so this is the regression observable for the failure above.

| lane | run status | tenferro worker masks | provider thread masks |
|---|---|---|---|
| `system-openblas` | exit 0 | ['10-13'] | no provider threads observed |
| `system-mkl` | exit 0 | ['10-13'] | {'openmp_worker': ['10-13']} |
| `cpu-faer` | exit 0 | ['10-13'] | no provider threads observed |

Before the fix the same probe showed `openmp_worker` at `cpus=10` (one CPU,
inherited from the worker pinned to CPU 10). It now reports `10-13`, i.e. the
whole admitted domain.

### Controlled before/after (same binary, MKL, 4T)

A dedicated A/B run toggles only the worker mask (per-worker single CPU vs
domain set) with the same binary and interleaved execution, so it isolates the
cause from build, harness and drift:

| operation | per-worker single CPU | domain CPU set |
|---|---:|---:|
| `dgemm` 1024x1024 | 48.0 ms | 13.3 ms |
| `solve` 1024x1024 rhs=1 | 136.5 ms | 6.6 ms |
| `svd` 512x512 | 116.2 ms | 30.5 ms |
| `qr` 1024x1024 | 321.8 ms | 86.1 ms |
| `eigh` 512x512 | 84.0 ms | 18.4 ms |
| elementwise/transpose (native) | 8.8 / 21.8 ms | 8.8 / 21.9 ms |

The native rows show the change costs nothing outside the provider path; a
separate paired run at 4 workers over 16 CPUs across two CCXs kept every
faer-backed row within +-4% with mixed signs, and 2 ms affinity sampling showed
0-0.2 worker migrations/s without a provider.

### Provider pairs measured here

| lane | tenferro side | PyTorch side |
|---|---|---|
| `system-openblas` | system OpenBLAS 0.3.26 (`/opt/openblas`, DYNAMIC_ARCH/Zen, MAX_THREADS=64) | wheel-bundled Intel MKL 2024.2 |
| `system-mkl` | system oneAPI MKL (`MKLROOT`) | wheel-bundled Intel MKL 2024.2 |
| `cpu-faer` | faer (no system BLAS) | wheel-bundled Intel MKL 2024.2 |

The `system-mkl` lane therefore compares two different MKL builds, and no lane
makes both sides use the same library except the provider-matched OpenBLAS image.
See [CPU provider pairs](../../../README.md#cpu-provider-pairs).

### Reproduce

```sh
# tenferro lanes (Linux default is system-openblas; use system-mkl / cpu-faer for the others)
cargo build -j 16 --release --features system-openblas --bin publication_gate
export PUBLICATION_GATE_PROFILE=full PUBLICATION_GATE_SUITE=large \
       PUBLICATION_GATE_TENFERRO_MODE=both BENCH_RUNS=15 BENCH_WARMUPS=3
taskset -c 10-13 env RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  LD_LIBRARY_PATH=/opt/openblas/lib ./target/release/publication_gate > run-4t.csv

# PyTorch reference at the same thread count
.venv/bin/python scripts/benchmark_cpu_ops_python.py --backend pytorch-cpu \
  --num-threads 4 --runs 15 --warmups 3 --output run-4t-torch.csv

# affinity guard: which CPUs were worker and provider threads actually allowed to use
python3 scripts/cpu_provider_affinity_check.py --output guard-4t.json -- \
  taskset -c 10-13 env RAYON_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib \
  ./target/release/publication_gate
```

Raw CSVs, the guard JSON, the campaign driver and the summarizer are kept under
`data/results/amd-cpu/cpu/large-ad/postfix4t/` (local, git-ignored).
