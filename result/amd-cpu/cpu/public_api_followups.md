# CPU Public API Benchmark Results

- Suite: `cpu/public_api_followups`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api_followups.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api_followups/20261013_082720/run.yaml`
- Timestamp: `20261013_082720`
- Selection: thirteen API follow-ups previously measured in perf_issues; this report preserves the complete Public API report. Julia columns are unselected.
- Invocation prefix: `PUBLIC_API_FOLLOWUPS_ONLY=1 TENFERRO_CPU_FEATURES=blas-mkl BENCHMARK_TARGET_PROFILE=amd-cpu`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api_followups/20261013_082720`.

- tenferro-rs commit: `8b13220c1e0397a29a27be65557c24105352a35b`

## CPU Information

- Model: `Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `16`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `1`
- NUMA nodes: `unknown`
- Python platform: `Linux-7.0.14-linuxkit-x86_64-with-glibc2.39`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/amd-cpu/cpu/public_api_followups/20261013_082720/run_t1.yaml`
- OMP_NUM_THREADS: `1`
- OMP_THREAD_LIMIT: `1`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `1`
- OPENBLAS_NUM_THREADS: `1`
- GOTO_NUM_THREADS: `1`
- MKL_NUM_THREADS: `1`
- VECLIB_MAXIMUM_THREADS: `1`
- VECLIB_NUM_THREADS: `1`
- NUMEXPR_NUM_THREADS: `1`
- BLIS_NUM_THREADS: `1`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=1`
- JULIA_NUM_THREADS: `1`

### Threads: 4

- Run metadata: `data/results/amd-cpu/cpu/public_api_followups/20261013_082720/run_t4.yaml`
- OMP_NUM_THREADS: `4`
- OMP_THREAD_LIMIT: `4`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `4`
- OPENBLAS_NUM_THREADS: `4`
- GOTO_NUM_THREADS: `4`
- MKL_NUM_THREADS: `4`
- VECLIB_MAXIMUM_THREADS: `4`
- VECLIB_NUM_THREADS: `4`
- NUMEXPR_NUM_THREADS: `4`
- BLIS_NUM_THREADS: `4`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=4`
- JULIA_NUM_THREADS: `4`

## Timing Discipline

- These thirteen APIs already have AMD perf_issues measurements; this collection adds dedicated standard public-operation cases and JAX comparisons rather than claiming they were previously unmeasured.
- All cases use deterministic F32 logical matrices in native framework layouts. Direct and eager reuse entered sessions; inputs, masks, indices, runtime construction, initialization, JIT preparation and numerical validation are outside timing, including zero warmups.
- The small 1024x64 cases batch 64 operations and the 4096x256 control cases batch eight, retaining every output through clock stop. Batch duration and normalized per-operation timing are recorded.
- Axis operations reduce or normalize axis 1. LayerNorm and RMSNorm are non-affine with eps=1e-5; affine variants are not claimed as covered. JAX standardize(algorithm=stable) provides LayerNorm, and standardize(mean=0, algorithm=stable) provides non-affine RMSNorm directly.
- gelu requests the exact erf formulation; gelu_tanh requests the tanh approximation. Softplus inputs stay below the PyTorch threshold transition. Masked softmax requests zero masked outputs; JAX uses its where argument and the missing direct PyTorch contract is unsupported.
- Each route is validated against independent double-precision scalar/NumPy references. Operation-created outputs are timed; validation copies and output destruction are untimed. Eager capture is inactive and inputs are untracked.
- Compiled trace has no public borrowed-session execution API, so its session creation is excluded through explicit unsupported rows. Julia columns are unselected.
- Provider/configuration/linked-library and thread observations are recorded in run_t1.yaml and run_t4.yaml. The CPU model above identifies this Docker host; these results are not a speedup comparison to historical AMD hardware.

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.278 ± 0.002 | 0.281 ± 0.010 | unsupported | 0.097 ± 0.002 | 0.088 ± 0.004 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 4.407 ± 0.033 | 4.424 ± 0.084 | unsupported | 1.124 ± 0.021 | 0.728 ± 0.082 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.320 ± 0.009 | 0.330 ± 0.005 | unsupported | 0.035 ± 0.010 | 0.083 ± 0.016 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 1.609 ± 0.070 | 1.615 ± 0.055 | unsupported | 0.449 ± 0.033 | 0.657 ± 0.086 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.386 ± 0.003 | 0.386 ± 0.003 | unsupported | 0.118 ± 0.003 | 0.262 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 6.162 ± 0.028 | 6.156 ± 0.038 | unsupported | 1.243 ± 0.017 | 1.716 ± 0.036 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.346 ± 0.005 | 0.381 ± 0.004 | unsupported | 0.046 ± 0.006 | 0.247 ± 0.006 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 2.021 ± 0.088 | 2.007 ± 0.073 | unsupported | 0.475 ± 0.032 | 0.886 ± 0.029 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 1.100 ± 0.005 | 1.101 ± 0.002 | unsupported | 0.245 ± 0.003 | 0.184 ± 0.017 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 17.634 ± 0.018 | 17.622 ± 0.066 | unsupported | 3.446 ± 0.055 | 1.047 ± 0.018 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.810 ± 0.008 | 0.708 ± 0.006 | unsupported | 0.071 ± 0.003 | 0.238 ± 0.023 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 5.170 ± 0.231 | 5.285 ± 0.132 | unsupported | 1.034 ± 0.038 | 0.708 ± 0.032 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.098 ± 0.004 | 0.120 ± 0.003 | unsupported | 0.139 ± 0.003 | 0.208 ± 0.018 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 1.379 ± 0.173 | 0.941 ± 0.038 | unsupported | 1.353 ± 0.022 | 4.170 ± 0.064 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 1.727 ± 0.028 | 1.292 ± 0.017 | unsupported | 0.044 ± 0.002 | 0.296 ± 0.007 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 3.204 ± 0.111 | 2.216 ± 0.073 | unsupported | 0.252 ± 0.020 | 1.986 ± 0.043 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.138 ± 0.006 | 0.155 ± 0.006 | unsupported | 0.107 ± 0.001 | 0.437 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 2.034 ± 0.115 | 1.826 ± 0.200 | unsupported | 1.347 ± 0.048 | 4.454 ± 0.021 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 1.747 ± 0.025 | 1.424 ± 0.019 | unsupported | 0.039 ± 0.003 | 0.545 ± 0.019 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 3.467 ± 0.259 | 2.696 ± 0.162 | unsupported | 0.335 ± 0.038 | 2.057 ± 0.068 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.174 ± 0.006 | 0.226 ± 0.007 | unsupported | unsupported | 0.457 ± 0.002 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 2.935 ± 0.113 | 2.683 ± 0.059 | unsupported | unsupported | 5.149 ± 0.058 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 2.084 ± 0.024 | 1.591 ± 0.020 | unsupported | unsupported | 0.565 ± 0.013 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 4.343 ± 0.106 | 3.036 ± 0.309 | unsupported | unsupported | 2.700 ± 0.137 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.016 ± 0.000 | 0.027 ± 0.000 | unsupported | 0.017 ± 0.000 | 0.118 ± 0.009 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.104 ± 0.002 | 0.132 ± 0.007 | unsupported | 0.100 ± 0.006 | 1.818 ± 0.014 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.427 ± 0.013 | 0.412 ± 0.006 | unsupported | 0.012 ± 0.000 | 0.117 ± 0.009 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.557 ± 0.047 | 0.550 ± 0.072 | unsupported | 0.035 ± 0.005 | 0.793 ± 0.032 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.053 ± 0.003 | 0.068 ± 0.003 | unsupported | 0.096 ± 0.003 | 0.186 ± 0.012 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.685 ± 0.083 | 0.483 ± 0.015 | unsupported | 0.947 ± 0.071 | 2.603 ± 0.035 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.993 ± 0.030 | 0.775 ± 0.080 | unsupported | 0.072 ± 0.004 | 0.268 ± 0.005 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 1.731 ± 0.159 | 1.276 ± 0.080 | unsupported | 0.579 ± 0.057 | 1.468 ± 0.079 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.623 ± 0.003 | 0.625 ± 0.004 | unsupported | 0.091 ± 0.003 | 0.176 ± 0.008 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 9.972 ± 0.043 | 9.991 ± 0.031 | unsupported | 1.052 ± 0.026 | 1.123 ± 0.031 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.453 ± 0.028 | 0.446 ± 0.005 | unsupported | 0.049 ± 0.006 | 0.239 ± 0.036 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 2.807 ± 0.065 | 2.801 ± 0.116 | unsupported | 0.430 ± 0.029 | 0.721 ± 0.050 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.539 ± 0.003 | 0.537 ± 0.002 | unsupported | 0.095 ± 0.002 | 0.179 ± 0.008 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 8.601 ± 0.040 | 8.588 ± 0.018 | unsupported | 1.100 ± 0.033 | 1.109 ± 0.038 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.499 ± 0.022 | 0.497 ± 0.012 | unsupported | 0.048 ± 0.004 | 0.231 ± 0.013 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 2.929 ± 0.041 | 2.877 ± 0.088 | unsupported | 0.439 ± 0.035 | 0.719 ± 0.038 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.134 ± 0.006 | 0.149 ± 0.007 | unsupported | 0.115 ± 0.003 | 0.433 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 2.092 ± 0.108 | 1.836 ± 0.122 | unsupported | 1.066 ± 0.017 | 4.419 ± 0.038 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 1.656 ± 0.020 | 1.308 ± 0.010 | unsupported | 0.036 ± 0.001 | 0.548 ± 0.011 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 3.348 ± 0.078 | 2.433 ± 0.170 | unsupported | 0.278 ± 0.019 | 2.043 ± 0.082 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 1.339 ± 0.003 | 1.338 ± 0.003 | unsupported | 0.223 ± 0.002 | 0.363 ± 0.003 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 21.448 ± 0.042 | 21.452 ± 0.053 | unsupported | 3.089 ± 0.025 | 3.209 ± 0.026 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.972 ± 0.007 | 0.870 ± 0.059 | unsupported | 0.117 ± 0.003 | 0.313 ± 0.013 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 6.404 ± 0.088 | 6.489 ± 0.152 | unsupported | 0.911 ± 0.034 | 1.233 ± 0.035 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.986 ± 0.016 | 1.046 ± 0.016 | unsupported | 0.594 ± 0.002 | 0.310 ± 0.004 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 27.570 ± 0.130 | 23.247 ± 0.370 | unsupported | 9.601 ± 0.030 | 2.362 ± 0.036 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 6.048 ± 0.858 | 4.698 ± 0.104 | unsupported | 0.328 ± 0.010 | 0.266 ± 0.004 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 24.923 ± 0.604 | 19.177 ± 0.424 | unsupported | 2.824 ± 0.105 | 1.040 ± 0.032 | - | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.641610 | 88150.16 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.180811 | 96575.17 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 17.782092 | 277845.19 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 18.006302 | 281348.47 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.825645 | 728205.62 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.988598 | 1123574.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 35.254317 | 4406789.62 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 35.390573 | 4423821.62 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.333626 | 83337.91 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.245260 | 35082.19 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 20.486829 | 320106.70 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 21.119958 | 329999.34 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.259093 | 657386.62 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.591452 | 448931.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 12.874344 | 1609293.00 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 12.916199 | 1614524.88 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.773596 | 262087.44 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.573367 | 118333.86 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 24.673221 | 385519.08 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 24.692230 | 385816.09 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.731283 | 1716410.38 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.947280 | 1243410.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 49.293862 | 6161732.75 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 49.244368 | 6155546.00 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.811898 | 247060.91 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.965070 | 46329.22 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 22.144276 | 346004.31 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 24.369834 | 380778.66 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.090117 | 886264.62 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.800202 | 475025.25 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.168970 | 2021121.25 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.055910 | 2006988.75 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.775852 | 183997.69 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 15.663986 | 244749.78 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 70.399989 | 1099999.83 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 70.444089 | 1100688.89 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.376688 | 1047086.00 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 27.567602 | 3445950.25 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 141.069320 | 17633665.00 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 140.979283 | 17622410.38 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.247534 | 238242.72 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.526339 | 70724.05 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 51.828793 | 809824.89 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 45.313098 | 708017.16 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.665448 | 708181.00 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.270912 | 1033864.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 41.358827 | 5169853.38 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 42.278196 | 5284774.50 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.313721 | 208026.89 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.880294 | 138754.59 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.246984 | 97609.12 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.668105 | 119814.14 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 33.363255 | 4170406.88 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.823223 | 1352902.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.029890 | 1378736.25 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.531271 | 941408.88 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 18.913392 | 295521.75 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.805644 | 43838.19 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 110.533092 | 1727079.56 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 82.701620 | 1292212.81 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 15.885031 | 1985628.88 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.019986 | 252498.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 25.629527 | 3203690.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 17.728705 | 2216088.12 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 27.966037 | 436969.33 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.823204 | 106612.56 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.835119 | 138048.73 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.946540 | 155414.69 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 35.629850 | 4453731.25 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.773305 | 1346663.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 16.274184 | 2034273.00 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.610312 | 1826289.00 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 34.896439 | 545256.86 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.475371 | 38677.67 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 111.828751 | 1747324.23 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 91.139074 | 1424048.03 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.452850 | 2056606.25 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.678764 | 334845.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 27.733514 | 3466689.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 21.565616 | 2695702.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 29.258531 | 457164.55 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.118174 | 173721.47 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 14.438206 | 225596.97 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 41.190792 | 5148849.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.483377 | 2935422.12 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 21.465856 | 2683232.00 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 36.135777 | 564621.52 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 133.379399 | 2084053.11 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 101.840192 | 1591253.00 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 21.600567 | 2700070.88 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 34.740217 | 4342527.12 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 24.287720 | 3035965.00 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.536684 | 117760.69 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.090459 | 17038.42 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.038832 | 16231.75 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.717623 | 26837.86 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.541880 | 1817735.00 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.801214 | 100151.75 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.832489 | 104061.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.053734 | 131716.75 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.479322 | 116864.41 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.786948 | 12296.06 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 27.345613 | 427275.20 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 26.340332 | 411567.69 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.347525 | 793440.62 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.277794 | 34724.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.454999 | 556874.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.400848 | 550106.00 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.878333 | 185598.95 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.120167 | 95627.61 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.415278 | 53363.72 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.357090 | 68079.53 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.824069 | 2603008.62 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.574638 | 946829.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.476184 | 684523.00 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.867476 | 483434.50 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 17.152891 | 268013.92 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.594977 | 71796.52 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 63.526839 | 992606.86 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 49.599848 | 774997.62 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.744892 | 1468111.50 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.629063 | 578632.88 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 13.851612 | 1731451.50 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.204560 | 1275570.00 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.260837 | 175950.58 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.801005 | 90640.70 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 39.891753 | 623308.64 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 40.031680 | 625495.00 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.984676 | 1123084.50 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.412121 | 1051515.12 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 79.776095 | 9972011.88 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 79.931000 | 9991375.00 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.301538 | 239086.53 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.140378 | 49068.41 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 28.994963 | 453046.30 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 28.569514 | 446398.66 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.768494 | 721061.75 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.437352 | 429669.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 22.459567 | 2807445.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 22.408818 | 2801102.25 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.465551 | 179149.23 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.070222 | 94847.22 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 34.481614 | 538775.22 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 34.392930 | 537389.53 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.875127 | 1109390.88 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.802189 | 1100273.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 68.805681 | 8600710.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 68.703421 | 8587927.62 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.789222 | 231081.59 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.069389 | 47959.20 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 31.917931 | 498717.67 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 31.800133 | 496877.08 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.751726 | 718965.75 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.515942 | 439492.75 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 23.435644 | 2929455.50 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 23.012387 | 2876548.38 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 27.692358 | 432693.09 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.385283 | 115395.05 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.586587 | 134165.42 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.509995 | 148593.67 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 35.348484 | 4418560.50 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.531971 | 1066496.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 16.738406 | 2092300.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.687276 | 1835909.50 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 35.096263 | 548379.11 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.285963 | 35718.17 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 105.971991 | 1655812.36 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 83.716026 | 1308062.91 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.341417 | 2042677.12 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.226989 | 278373.62 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 26.786024 | 3348253.00 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 19.461950 | 2432743.75 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 23.242831 | 363169.23 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 14.299518 | 223429.97 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 85.699147 | 1339049.17 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 85.660642 | 1338447.53 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 25.669285 | 3208660.62 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 24.712032 | 3089004.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 171.583927 | 21447990.88 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 171.617235 | 21452154.38 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 20.015314 | 312739.28 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.518035 | 117469.30 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 62.211305 | 972051.64 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 55.660170 | 869690.16 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.861557 | 1232694.62 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.286861 | 910857.62 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 51.233352 | 6404169.00 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 51.914330 | 6489291.25 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 19.836816 | 309950.25 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 38.019108 | 594048.56 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 63.120307 | 986254.80 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 66.948453 | 1046069.58 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.894897 | 2361862.12 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 76.808384 | 9601048.00 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 220.560179 | 27570022.38 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 185.977643 | 23247205.38 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 17.045139 | 266330.30 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 21.023240 | 328488.12 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 387.073073 | 6048016.77 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 300.662186 | 4697846.66 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.317934 | 1039741.75 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 22.592731 | 2824091.38 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 199.382729 | 24922841.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 153.418057 | 19177257.12 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/neural_public_api/gelu_tanh` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 16.8x faster than the slowest successful cell (`tenferro-direct`, 17.634 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`tenferro-direct`, 0.810 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 39.4x faster than the slowest successful cell (`tenferro-direct`, 1.727 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 12.7x faster than the slowest successful cell (`tenferro-direct`, 3.204 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 45.2x faster than the slowest successful cell (`tenferro-direct`, 1.747 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`tenferro-direct`, 3.467 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=1, shape=`4096x256`): `pytorch-cpu` is 18.1x faster than the slowest successful cell (`jax-cpu`, 1.818 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=1, shape=`4096x256`): `tenferro-direct` is 17.5x faster than the slowest successful cell (`jax-cpu`, 1.818 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=1, shape=`4096x256`): `tenferro-eager` is 13.8x faster than the slowest successful cell (`jax-cpu`, 1.818 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 34.7x faster than the slowest successful cell (`tenferro-direct`, 0.427 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 22.8x faster than the slowest successful cell (`jax-cpu`, 0.793 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 13.8x faster than the slowest successful cell (`tenferro-direct`, 0.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`tenferro-direct`, 0.499 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 46.4x faster than the slowest successful cell (`tenferro-direct`, 1.656 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`tenferro-direct`, 3.348 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 11.7x faster than the slowest successful cell (`tenferro-direct`, 27.570 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 22.7x faster than the slowest successful cell (`tenferro-direct`, 6.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 18.4x faster than the slowest successful cell (`tenferro-direct`, 6.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 24.0x faster than the slowest successful cell (`tenferro-direct`, 24.923 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.
