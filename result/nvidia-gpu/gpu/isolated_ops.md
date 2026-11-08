# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **124**. Unsupported cells: **8**, including **6** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1478.0 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261108_115237/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**11** of 33) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_f64_n1048576 † | 31.009 (0.561) | 96.480 (1.184) | 12.095 (0.218) | 93.911 (6.093) |
| sub_f64_n1048576 † | 31.274 (0.608) | 92.817 (0.665) | 12.249 (0.158) | 102.980 (8.702) |
| mul_f64_n1048576 † | 31.504 (0.667) | 93.146 (0.833) | 12.297 (0.190) | 81.995 (1.990) |
| div_f64_n1048576 | 31.182 (0.386) | 96.909 (1.442) | 13.156 (0.138) | 88.060 (8.194) |
| rem_f64_n1048576 | 62.378 (1.701) | 119.416 (6.636) | 12.197 (0.257) | 73.099 (4.257) |
| pow_f64_n1048576 | 65.366 (2.459) | 123.608 (5.731) | 29.520 (0.271) | 83.040 (8.531) |
| maximum_f64_n1048576 | 51.104 (0.838) | 103.903 (0.695) | 12.273 (0.213) | 80.707 (1.450) |
| minimum_f64_n1048576 | 51.199 (0.678) | 107.744 (1.110) | 12.236 (0.201) | 92.758 (14.073) |
| neg_f64_n1048576 | 25.526 (1.601) | 55.710 (0.514) | 7.994 (0.066) | 81.355 (1.951) |
| abs_f64_n1048576 | 24.943 (0.195) | 55.616 (0.495) | 8.431 (0.650) | 69.611 (1.642) |
| sign_f64_n1048576 | 25.046 (0.349) | 56.029 (0.642) | 7.979 (1.709) | 81.329 (12.946) |
| conj_c64_n1048576 | 39.742 (0.797) | 84.856 (0.552) | 24.293 (0.156) | 70.409 (5.209) |
| exp_f64_n1048576 | 26.066 (0.594) | 58.922 (0.284) | 8.384 (0.109) | 74.794 (1.701) |
| expm1_f64_n1048576 | 26.728 (0.328) | 60.146 (1.054) | 10.085 (0.301) | 68.904 (2.212) |
| log_f64_n1048576 | 29.204 (0.470) | 60.163 (0.445) | 12.720 (0.110) | 78.334 (2.903) |
| log1p_f64_n1048576 | 29.886 (0.599) | 60.118 (0.726) | 13.269 (0.119) | 67.140 (1.211) |
| erf_f64_n1048576 | 32.964 (0.949) | 61.632 (0.515) | 14.540 (0.036) | 91.203 (1.465) |
| sin_f64_n1048576 | 29.495 (0.822) | 58.903 (0.349) | 11.311 (0.072) | 67.789 (1.764) |
| cos_f64_n1048576 | 29.068 (0.736) | 59.013 (0.591) | 11.385 (0.104) | 69.360 (2.453) |
| tanh_f64_n1048576 | 27.052 (0.849) | 56.036 (0.778) | 8.390 (0.108) | 70.374 (1.799) |
| sqrt_f64_n1048576 | 25.475 (0.315) | 55.653 (0.746) | 8.028 (0.063) | 71.694 (4.062) |
| rsqrt_f64_n1048576 | 25.434 (0.421) | 55.465 (0.691) | 8.022 (0.081) | 105.255 (4.717) |
| compare_f64_n1048576 † | 22.083 (0.238) | 86.855 (0.784) | 8.106 (0.129) | 73.308 (1.790) |
| where_select_f64_n1048576 | 32.696 (0.482) | 122.638 (6.321) | 13.982 (0.077) | 77.553 (2.372) |
| clamp_f64_n1048576 † | 88.036 (0.480) | 163.048 (1.381) | 10.390 (0.078) | 82.121 (2.245) |
| reduce_sum_f64_n1048576 † | 36.991 (0.266) | 74.909 (0.456) | 7.310 (0.029) | 90.705 (3.783) |
| reduce_prod_f64_n1048576 † | 36.662 (0.136) | 71.831 (0.598) | 7.344 (0.042) | 85.887 (5.278) |
| reduce_max_f64_n1048576 † | 38.099 (0.347) | 71.497 (0.395) | 7.326 (0.017) | 78.632 (8.172) |
| reduce_min_f64_n1048576 † | 37.802 (0.248) | 74.777 (2.189) | 7.403 (0.084) | 82.753 (10.986) |
| reduce_sum_squares_f64_n1048576 † | 36.860 (0.264) | 71.481 (0.411) | unsupported | unsupported |
| reduce_mean_f64_n1048576 † | unsupported | unsupported | 7.178 (0.060) | 80.141 (11.098) |
| scale_real_f64_n1048576 | unsupported | unsupported | 9.647 (0.106) | 85.753 (3.527) |
| scale_complex_c64_n1048576 | unsupported | unsupported | 24.511 (0.183) | 77.925 (4.663) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 2.228 | 16777216 | 0.163 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.270 | 16777216 | 1.346 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.798 | 16777216 | 0.455 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.780 | 16777216 | 0.204 | 0 |
| add_f64_n1048576 | jax-cuda | 4 | 15 | 0.376 | 25165824 | 0.181 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.387 | 25165824 | 1.408 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.992 | 25165824 | 0.549 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.544 | 25165824 | 0.176 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 2.628 | 16777232 | 0.138 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.332 | 16777232 | 1.093 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.817 | 16777232 | 0.129 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.609 | 16777232 | 0.070 | 0 |
| compare_f64_n1048576 | jax-cuda | 32 | 15 | 2.346 | 17825792 | 0.165 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 8 | 15 | 0.065 | 17825792 | 1.488 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.827 | 17825792 | 0.546 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.779 | 17825792 | 0.139 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 1.127 | 33554432 | 0.322 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.389 | 33554432 | 0.935 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.636 | 33554432 | 0.571 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.358 | 33554432 | 0.268 | 0 |
| cos_f64_n1048576 | jax-cuda | 32 | 15 | 2.220 | 16777216 | 0.164 | 5.905e-08 |
| cos_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.364 | 16777216 | 0.997 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.930 | 16777216 | 0.391 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.888 | 16777216 | 0.192 | 5.905e-08 |
| div_f64_n1048576 | jax-cuda | 32 | 15 | 2.818 | 25165824 | 0.193 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.421 | 25165824 | 1.294 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.998 | 25165824 | 0.546 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.101 | 25165824 | 0.176 | 0 |
| erf_f64_n1048576 | jax-cuda | 32 | 15 | 2.919 | 16777216 | 0.124 | 8.536e-08 |
| erf_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.465 | 16777216 | 0.781 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.055 | 16777216 | 0.344 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.972 | 16777216 | 0.184 | 8.536e-08 |
| exp_f64_n1048576 | jax-cuda | 32 | 15 | 2.393 | 16777216 | 0.152 | 1.095e-07 |
| exp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.268 | 16777216 | 1.354 | 1.095e-07 |
| exp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.834 | 16777216 | 0.435 | 1.095e-07 |
| exp_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.885 | 16777216 | 0.193 | 1.095e-07 |
| expm1_f64_n1048576 | jax-cuda | 32 | 15 | 2.205 | 16777216 | 0.165 | 7.39e-08 |
| expm1_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.323 | 16777216 | 1.126 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.855 | 16777216 | 0.425 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.925 | 16777216 | 0.189 | 7.39e-08 |
| log1p_f64_n1048576 | jax-cuda | 32 | 15 | 2.148 | 16777216 | 0.169 | 1.11e-07 |
| log1p_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.425 | 16777216 | 0.855 | 1.11e-07 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.956 | 16777216 | 0.380 | 1.11e-07 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.924 | 16777216 | 0.189 | 1.11e-07 |
| log_f64_n1048576 | jax-cuda | 32 | 15 | 2.507 | 16777216 | 0.145 | 4.308e-08 |
| log_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.407 | 16777216 | 0.892 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.935 | 16777216 | 0.389 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.925 | 16777216 | 0.189 | 4.308e-08 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 2.583 | 25165824 | 0.211 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.393 | 25165824 | 1.387 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.635 | 25165824 | 0.333 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.325 | 25165824 | 0.164 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 2.968 | 25165824 | 0.184 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.392 | 25165824 | 1.392 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.638 | 25165824 | 0.333 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.448 | 25165824 | 0.158 | 0 |
| mul_f64_n1048576 | jax-cuda | 16 | 15 | 1.312 | 25165824 | 0.208 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.393 | 25165824 | 1.385 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.008 | 25165824 | 0.540 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.981 | 25165824 | 0.183 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 2.603 | 16777216 | 0.140 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.256 | 16777216 | 1.420 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.817 | 16777216 | 0.445 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.783 | 16777216 | 0.204 | 0 |
| pow_f64_n1048576 | jax-cuda | 32 | 15 | 2.657 | 25165824 | 0.205 | 1.48e-07 |
| pow_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.945 | 25165824 | 0.577 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.092 | 25165824 | 0.260 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.955 | 25165824 | 0.138 | 1.48e-07 |
| reduce_max_f64_n1048576 | jax-cuda | 32 | 15 | 2.516 | 8396800 | 0.072 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.751 | 8396800 | 0.775 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.438 | 8396800 | 0.149 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.288 | 8396800 | 0.079 | 0 |
| reduce_mean_f64_n1048576 | jax-cuda | 32 | 15 | 2.565 | 8396800 | 0.071 | 6.5e-09 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.675 | 8396800 | 0.791 | 3.9e-09 |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_min_f64_n1048576 | jax-cuda | 32 | 15 | 2.648 | 8396800 | 0.069 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 256 | 15 | 1.895 | 8396800 | 0.767 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.419 | 8396800 | 0.150 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.393 | 8396800 | 0.076 | 0 |
| reduce_prod_f64_n1048576 | jax-cuda | 32 | 15 | 2.748 | 8396800 | 0.066 | 4.436e-06 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.760 | 8396800 | 0.774 | 4.216e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.346 | 8396800 | 0.155 | 3.55e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.299 | 8396800 | 0.079 | 3.55e-06 |
| reduce_sum_f64_n1048576 | jax-cuda | 32 | 15 | 2.903 | 8396800 | 0.063 | 4.449e-06 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 256 | 15 | 1.871 | 8396800 | 0.777 | 2.961e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.367 | 8396800 | 0.154 | 3.391e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.397 | 8396800 | 0.076 | 3.391e-06 |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.359 | 8396800 | 0.154 | 3.294e-07 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.287 | 8396800 | 0.079 | 3.294e-07 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 2.339 | 25165824 | 0.233 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.390 | 25165824 | 1.396 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.996 | 25165824 | 0.273 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.821 | 25165824 | 0.143 | 0 |
| rsqrt_f64_n1048576 | jax-cuda | 32 | 15 | 3.368 | 16777216 | 0.108 | 6.504e-08 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.257 | 16777216 | 1.415 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.814 | 16777216 | 0.446 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.775 | 16777216 | 0.205 | 6.504e-08 |
| scale_complex_c64_n1048576 | jax-cuda | 16 | 15 | 1.247 | 33554448 | 0.291 | 4.129e-08 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.392 | 33554448 | 0.926 | 0 |
| scale_complex_c64_n1048576 | tenferro-cuda-direct | — | — | — | 33554448 | unavailable | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554448 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_f64_n1048576 | jax-cuda | 32 | 15 | 2.744 | 16777224 | 0.132 | 0 |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.309 | 16777224 | 1.177 | 0 |
| scale_real_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777224 | unavailable | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_real_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777224 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 2.603 | 16777216 | 0.140 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.255 | 16777216 | 1.423 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.801 | 16777216 | 0.453 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.793 | 16777216 | 0.203 | 0 |
| sin_f64_n1048576 | jax-cuda | 32 | 15 | 2.169 | 16777216 | 0.167 | 4.153e-08 |
| sin_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.362 | 16777216 | 1.004 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.944 | 16777216 | 0.385 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.885 | 16777216 | 0.193 | 4.153e-08 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 2.294 | 16777216 | 0.158 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.257 | 16777216 | 1.414 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.815 | 16777216 | 0.446 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.781 | 16777216 | 0.204 | 0 |
| sub_f64_n1048576 | jax-cuda | 8 | 15 | 0.824 | 25165824 | 0.165 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.392 | 25165824 | 1.390 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.001 | 25165824 | 0.544 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.970 | 25165824 | 0.183 | 0 |
| tanh_f64_n1048576 | jax-cuda | 32 | 15 | 2.252 | 16777216 | 0.161 | 4.437e-08 |
| tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.268 | 16777216 | 1.353 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.866 | 16777216 | 0.420 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.793 | 16777216 | 0.203 | 4.437e-08 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 2.482 | 26214400 | 0.229 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.447 | 26214400 | 1.269 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.046 | 26214400 | 0.542 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.924 | 26214400 | 0.145 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261108_115237/gpu_isolated_ops.yaml
timestamp: '2026-11-08T11:52:37+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: cf44136e09063a2ca09d2fe0dbb38e8849cb4522
  dirty: false
resolved_deps:
  strided-perm:
  - 0.8.0
  - 0.12.0
  strided-view:
  - 0.7.1
  threadteam:
  - 0.1.0
  - 0.5.1
host:
  cpu: AMD EPYC 7763 64-Core Processor
  logical_cpus: 252
  power:
    smt:
      active: '0'
      control: notsupported
  numa:
    nodes: 1
    cpulists:
      '0': 0-251
  smt_siblings:
  - '0'
  - '1'
  - '2'
  - '3'
  - '4'
  - '5'
  - '6'
  - '7'
  - '8'
  - '9'
  - '10'
  - '11'
  - '12'
  - '13'
  - '14'
  - '15'
  - '16'
  - '17'
  - '18'
  - '19'
  - '20'
  - '21'
  - '22'
  - '23'
  - '24'
  - '25'
  - '26'
  - '27'
  - '28'
  - '29'
  - '30'
  - '31'
  - '32'
  - '33'
  - '34'
  - '35'
  - '36'
  - '37'
  - '38'
  - '39'
  - '40'
  - '41'
  - '42'
  - '43'
  - '44'
  - '45'
  - '46'
  - '47'
  - '48'
  - '49'
  - '50'
  - '51'
  - '52'
  - '53'
  - '54'
  - '55'
  - '56'
  - '57'
  - '58'
  - '59'
  - '60'
  - '61'
  - '62'
  - '63'
  - '64'
  - '65'
  - '66'
  - '67'
  - '68'
  - '69'
  - '70'
  - '71'
  - '72'
  - '73'
  - '74'
  - '75'
  - '76'
  - '77'
  - '78'
  - '79'
  - '80'
  - '81'
  - '82'
  - '83'
  - '84'
  - '85'
  - '86'
  - '87'
  - '88'
  - '89'
  - '90'
  - '91'
  - '92'
  - '93'
  - '94'
  - '95'
  - '96'
  - '97'
  - '98'
  - '99'
  - '100'
  - '101'
  - '102'
  - '103'
  - '104'
  - '105'
  - '106'
  - '107'
  - '108'
  - '109'
  - '110'
  - '111'
  - '112'
  - '113'
  - '114'
  - '115'
  - '116'
  - '117'
  - '118'
  - '119'
  - '120'
  - '121'
  - '122'
  - '123'
  - '124'
  - '125'
  - '126'
  - '127'
  - '128'
  - '129'
  - '130'
  - '131'
  - '132'
  - '133'
  - '134'
  - '135'
  - '136'
  - '137'
  - '138'
  - '139'
  - '140'
  - '141'
  - '142'
  - '143'
  - '144'
  - '145'
  - '146'
  - '147'
  - '148'
  - '149'
  - '150'
  - '151'
  - '152'
  - '153'
  - '154'
  - '155'
  - '156'
  - '157'
  - '158'
  - '159'
  - '160'
  - '161'
  - '162'
  - '163'
  - '164'
  - '165'
  - '166'
  - '167'
  - '168'
  - '169'
  - '170'
  - '171'
  - '172'
  - '173'
  - '174'
  - '175'
  - '176'
  - '177'
  - '178'
  - '179'
  - '180'
  - '181'
  - '182'
  - '183'
  - '184'
  - '185'
  - '186'
  - '187'
  - '188'
  - '189'
  - '190'
  - '191'
  - '192'
  - '193'
  - '194'
  - '195'
  - '196'
  - '197'
  - '198'
  - '199'
  - '200'
  - '201'
  - '202'
  - '203'
  - '204'
  - '205'
  - '206'
  - '207'
  - '208'
  - '209'
  - '210'
  - '211'
  - '212'
  - '213'
  - '214'
  - '215'
  - '216'
  - '217'
  - '218'
  - '219'
  - '220'
  - '221'
  - '222'
  - '223'
  - '224'
  - '225'
  - '226'
  - '227'
  - '228'
  - '229'
  - '230'
  - '231'
  - '232'
  - '233'
  - '234'
  - '235'
  - '236'
  - '237'
  - '238'
  - '239'
  - '240'
  - '241'
  - '242'
  - '243'
  - '244'
  - '245'
  - '246'
  - '247'
  - '248'
  - '249'
  - '250'
  - '251'
  l3_domains:
  - 0-125
  - 126-251
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: 941a9a84f64c
  os: Linux-6.8.0-138-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-251
    l3_domains:
    - 0-125
    - 126-251
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_NUM_THREADS: '1'
    MATMUL_NUM_THREADS: '1'
    CUDA_HOME: /opt/tenferro-benchmark-ci/cuda
    CUDA_PATH: /opt/tenferro-benchmark-ci/cuda
    TENFERRO_CUTENSOR_PATH: /opt/tenferro-benchmark-ci/cutensor/lib/libcutensor.so.2
    GPU_RUNTIME_SHA256: ecc740128a444e2b7d8a80ea018e5fbeb2da956dc2669f05886d8d7c97b49c67
    UV_PROJECT_ENVIRONMENT: /opt/tenferro-benchmark-ci/venv
    BENCH_RUNS: '15'
    BENCH_WARMUPS: '3'
timing_policy:
  version: 2
  scope: operation execution and completion synchronization
  setup: fixtures, H2D, runtime/session/handles, plans and compilation outside every
    timer
  cleanup: returned outputs retained until after timer stops
  minimum_untimed_priming_runs: 1
  short_operation_sampling: adaptive batches target 2 ms, at most 4096 operations
    and 256 MiB retained output; output-reuse destinations prepared before timing
  trace_execution: 'tenferro trace excluded: this suite compares immediate per-op
    calls, while trace executes a prepared graph in one caller-held session (run_prepared_in);
    JAX JIT compilation outside timing'
  setup_diagnostics: false
blas:
  implementation: none
python_backends:
  pytorch:
    available: true
    version: 2.11.0+cu128
    file: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/torch/__init__.py
    provider: mkl
    blas_info: mkl
    lapack_info: mkl
    config: "PyTorch built with:\n  - GCC 13.3\n  - C++ Version: 201703\n  - Intel(R)\
      \ oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R)\
      \ 64 architecture applications\n  - Intel(R) MKL-DNN v3.10.2 (Git Hash f1d471933dc852f956fd05389f9313c7148783d5)\n\
      \  - OpenMP 201511 (a.k.a. OpenMP 4.5)\n  - LAPACK is enabled (usually provided\
      \ by MKL)\n  - NNPACK is enabled\n  - CPU capability usage: AVX2\n  - CUDA Runtime\
      \ 12.8\n  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120\n\
      \  - CuDNN 91.9  (built against CUDA 12.9)\n  - Magma 2.6.1\n  - Build settings:\
      \ BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=70d99e998b4955e0049d13a98d77ae1b14db1f45,\
      \ CUDA_VERSION=12.8, CUDNN_VERSION=9.19.0, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++,\
      \ CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO\
      \ -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_FBGEMM -DUSE_MSLK -DUSE_PYTORCH_QNNPACK\
      \ -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED\
      \ -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct\
      \ -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas\
      \ -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow\
      \ -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized\
      \ -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference\
      \ -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1,\
      \ PERF_WITH_AVX2=1, TORCH_VERSION=2.11.0, USE_CUDA=ON, USE_CUDNN=ON, USE_CUSPARSELT=1,\
      \ USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF,\
      \ USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF,\
      \ USE_XCCL=OFF, USE_XPU=OFF, \n"
    library: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
    linked_libraries:
    - /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1
  jax:
    available: true
    version: 0.10.1
    jaxlib_version: 0.10.1
    backend: gpu
    file: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/jaxlib/__init__.py
    provider: xla_gpu
    dot_backend: xla_gpu
    lapack_provider: none_detected
    library: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/jaxlib/_ifrt_proxy.so
    linked_libraries: []
julia:
  available: false
  reason: julia not found on PATH
cuda:
  device_ordinal: 0
  device_name: NVIDIA A100 80GB PCIe
  device_uuid: GPU-e8de7039-b03e-2d5e-3147-bf92141dbc74
  device_memory_bytes: 85899345920
  driver_version: 595.91.07
  runtime_version: '12.8'
  cuda_version: '13.2'
  toolkit_version: null
  toolkit_root: /opt/tenferro-benchmark-ci/cuda
  cutensor_version: 2.2.0
  cusolver_version: 11.7.3
  cudnn_version: '91900'
  artifact_cache:
    path: /root/.cache/tenferro-benchmark/cuda-artifacts/12.8.93/sm_80
    regime: warm
gpu_memory_calibration:
  kind: device_copy
  method: torch.Tensor.copy_ between two contiguous device buffers, one copy per sample,
    device synchronized before and after the clock
  device_ordinal: 0
  device_name: NVIDIA A100 80GB PCIe
  buffer_bytes: 67108864
  traffic_bytes_per_copy: 134217728
  traffic_model: one read plus one write of buffer_bytes per copy
  warmups: 3
  sample_ms:
  - 0.092614
  - 0.095258
  - 0.091071
  - 0.091252
  - 0.09053
  - 0.090621
  - 0.089538
  - 0.09018
  - 0.091292
  - 0.089678
  - 0.090812
  - 0.09003
  - 0.093025
  - 0.091541
  - 0.089829
  median_ms: 0.090812
  bytes_per_second: 1477973483680.5708
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-08T12:07:58.936423+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-e8de7039-b03e-2d5e-3147-bf92141dbc74
  cuda_visible_devices: GPU-e8de7039-b03e-2d5e-3147-bf92141dbc74
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh isolated_ops
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 33
  selected_cases: 33
  sha256:
    .devcontainer/cuda/devcontainer.json: 0bf1465bbd1a4c9f896d5320838f2d5a488632a59af7f9b59cc050e95e0b8e09
    .devcontainer/cuda/Dockerfile: 578d9815e5145aa1651ac7e93833468c345ee8bf5e95bfa7717551580ae9feca
    src/bin/benchmark_gpu_public_api.rs: 28b921283d595842fc599670df7fadfa2b226573a287cb9fd90799300ca81e6c
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 949eadd6272c17ba4e380a228547a03608d328a5ef852eb95294731c288f24d9
    src/bin/benchmark_gpu_public_api/plan.rs: 70fce8a6e9ed7518779a51a9d8692e010fa1e470e95eee0ce5276f595737fde8
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: a8bbf8985bbc58c435e3c4dda7135161a141a90a10fa50d4f95128397ce949cb
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 32aa2ab45453502fc16479f429191aa24f867c8d5ad5f59d5a2752a3da413ee5
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 257228952e48c61fa1714d0cb2d6fdfdb9b0cf3898f59307ba9e72e513aa1b50
    data/results/nvidia-gpu/gpu/isolated_ops/20261108_115237/fixtures/manifest.json: 3c1a75f9c57996b228c04da1d4cbf34e4137a074e44ddca9f1c37e5824580bd8
    target/release/benchmark_gpu_public_api: 62f92e0c72d6fb46531ee8106f112567dfd32ddb482255a7cc41ba6819929786
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T12:08:26Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T12:08:18Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T12:08:06Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T12:08:11Z'
  compile:
    case_wall_seconds: 18.393
    compiling_processes: 0
    modules_compiled: 0
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 0.0
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| add_f64_n1048576 | tenferro-cuda-direct | add_f64_n1048576 untimed 222.4 ms is 1497% of the 14.9 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-direct | sub_f64_n1048576 untimed 153.1 ms is 1005% of the 15.2 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-direct | mul_f64_n1048576 untimed 139.7 ms is 904% of the 15.5 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-direct | div_f64_n1048576 untimed 139.2 ms is 931% of the 14.9 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-direct | rem_f64_n1048576 untimed 156.7 ms is 520% of the 30.1 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-direct | pow_f64_n1048576 untimed 135.9 ms is 429% of the 31.7 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-direct | maximum_f64_n1048576 untimed 138.0 ms is 562% of the 24.6 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-direct | minimum_f64_n1048576 untimed 144.7 ms is 590% of the 24.5 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-direct | neg_f64_n1048576 untimed 123.0 ms is 961% of the 12.8 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-direct | abs_f64_n1048576 untimed 124.5 ms is 1040% of the 12.0 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-direct | sign_f64_n1048576 untimed 131.8 ms is 1095% of the 12.0 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-direct | conj_c64_n1048576 untimed 185.1 ms is 1924% of the 9.6 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-direct | exp_f64_n1048576 untimed 122.7 ms is 965% of the 12.7 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-direct | expm1_f64_n1048576 untimed 125.7 ms is 979% of the 12.8 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-direct | log_f64_n1048576 untimed 122.6 ms is 858% of the 14.3 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-direct | log1p_f64_n1048576 untimed 132.8 ms is 926% of the 14.3 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-direct | erf_f64_n1048576 untimed 140.3 ms is 894% of the 15.7 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-direct | sin_f64_n1048576 untimed 125.5 ms is 862% of the 14.6 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-direct | cos_f64_n1048576 untimed 128.2 ms is 920% of the 13.9 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-direct | tanh_f64_n1048576 untimed 135.7 ms is 1040% of the 13.0 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-direct | sqrt_f64_n1048576 untimed 131.8 ms is 1076% of the 12.3 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | rsqrt_f64_n1048576 untimed 125.7 ms is 1018% of the 12.3 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-direct | compare_f64_n1048576 untimed 105.3 ms is 248% of the 42.5 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-direct | where_select_f64_n1048576 untimed 124.7 ms is 793% of the 15.7 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-direct | clamp_f64_n1048576 untimed 135.7 ms is 321% of the 42.3 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | reduce_sum_f64_n1048576 untimed 28.2 ms is 79% of the 35.5 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | reduce_prod_f64_n1048576 untimed 11.7 ms is 33% of the 35.2 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | reduce_max_f64_n1048576 untimed 10.0 ms is 27% of the 36.7 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | reduce_min_f64_n1048576 untimed 12.9 ms is 36% of the 36.3 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_f64_n1048576 untimed 8.8 ms is 25% of the 35.4 ms timed region (budget 20%) |
| add_f64_n1048576 | tenferro-cuda-eager | add_f64_n1048576 untimed 239.7 ms is 1030% of the 23.3 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-eager | sub_f64_n1048576 untimed 114.8 ms is 256% of the 44.8 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-eager | mul_f64_n1048576 untimed 136.6 ms is 305% of the 44.8 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-eager | div_f64_n1048576 untimed 131.5 ms is 279% of the 47.1 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-eager | rem_f64_n1048576 untimed 143.1 ms is 246% of the 58.3 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-eager | pow_f64_n1048576 untimed 129.7 ms is 216% of the 60.0 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-eager | maximum_f64_n1048576 untimed 141.0 ms is 283% of the 49.9 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-eager | minimum_f64_n1048576 untimed 130.5 ms is 250% of the 52.2 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-eager | neg_f64_n1048576 untimed 116.6 ms is 436% of the 26.8 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-eager | abs_f64_n1048576 untimed 105.6 ms is 395% of the 26.8 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-eager | sign_f64_n1048576 untimed 111.5 ms is 415% of the 26.9 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-eager | conj_c64_n1048576 untimed 158.1 ms is 776% of the 20.4 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-eager | exp_f64_n1048576 untimed 116.7 ms is 411% of the 28.4 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-eager | expm1_f64_n1048576 untimed 98.6 ms is 337% of the 29.3 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-eager | log_f64_n1048576 untimed 100.9 ms is 350% of the 28.8 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-eager | log1p_f64_n1048576 untimed 102.4 ms is 355% of the 28.9 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-eager | erf_f64_n1048576 untimed 108.2 ms is 365% of the 29.6 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-eager | sin_f64_n1048576 untimed 98.1 ms is 347% of the 28.3 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-eager | cos_f64_n1048576 untimed 98.1 ms is 346% of the 28.3 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-eager | tanh_f64_n1048576 untimed 99.6 ms is 370% of the 26.9 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-eager | sqrt_f64_n1048576 untimed 96.9 ms is 362% of the 26.8 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | rsqrt_f64_n1048576 untimed 102.3 ms is 383% of the 26.7 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-eager | compare_f64_n1048576 untimed 70.7 ms is 170% of the 41.7 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-eager | where_select_f64_n1048576 untimed 132.0 ms is 220% of the 60.0 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-eager | clamp_f64_n1048576 untimed 116.3 ms is 298% of the 39.1 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | reduce_sum_f64_n1048576 untimed 11.1 ms is 31% of the 36.2 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | reduce_prod_f64_n1048576 untimed 13.7 ms is 40% of the 34.5 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | reduce_max_f64_n1048576 untimed 8.9 ms is 26% of the 34.3 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | reduce_min_f64_n1048576 untimed 8.8 ms is 24% of the 36.4 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | reduce_sum_squares_f64_n1048576 untimed 11.9 ms is 35% of the 34.4 ms timed region (budget 20%) |
| add_f64_n1048576 | pytorch-cuda | add_f64_n1048576 untimed 257.1 ms is 4422% of the 5.8 ms timed region (budget 20%) |
| sub_f64_n1048576 | pytorch-cuda | sub_f64_n1048576 untimed 90.0 ms is 1537% of the 5.9 ms timed region (budget 20%) |
| mul_f64_n1048576 | pytorch-cuda | mul_f64_n1048576 untimed 82.2 ms is 1393% of the 5.9 ms timed region (budget 20%) |
| div_f64_n1048576 | pytorch-cuda | div_f64_n1048576 untimed 101.7 ms is 1612% of the 6.3 ms timed region (budget 20%) |
| rem_f64_n1048576 | pytorch-cuda | rem_f64_n1048576 untimed 107.0 ms is 1827% of the 5.9 ms timed region (budget 20%) |
| pow_f64_n1048576 | pytorch-cuda | pow_f64_n1048576 untimed 120.3 ms is 841% of the 14.3 ms timed region (budget 20%) |
| maximum_f64_n1048576 | pytorch-cuda | maximum_f64_n1048576 untimed 104.5 ms is 1721% of the 6.1 ms timed region (budget 20%) |
| minimum_f64_n1048576 | pytorch-cuda | minimum_f64_n1048576 untimed 83.1 ms is 1412% of the 5.9 ms timed region (budget 20%) |
| neg_f64_n1048576 | pytorch-cuda | neg_f64_n1048576 untimed 94.3 ms is 2444% of the 3.9 ms timed region (budget 20%) |
| abs_f64_n1048576 | pytorch-cuda | abs_f64_n1048576 untimed 90.3 ms is 2232% of the 4.0 ms timed region (budget 20%) |
| sign_f64_n1048576 | pytorch-cuda | sign_f64_n1048576 untimed 86.4 ms is 1977% of the 4.4 ms timed region (budget 20%) |
| conj_c64_n1048576 | pytorch-cuda | conj_c64_n1048576 untimed 157.5 ms is 2691% of the 5.9 ms timed region (budget 20%) |
| exp_f64_n1048576 | pytorch-cuda | exp_f64_n1048576 untimed 110.8 ms is 2741% of the 4.0 ms timed region (budget 20%) |
| expm1_f64_n1048576 | pytorch-cuda | expm1_f64_n1048576 untimed 91.8 ms is 1818% of the 5.1 ms timed region (budget 20%) |
| log_f64_n1048576 | pytorch-cuda | log_f64_n1048576 untimed 88.6 ms is 1438% of the 6.2 ms timed region (budget 20%) |
| log1p_f64_n1048576 | pytorch-cuda | log1p_f64_n1048576 untimed 84.0 ms is 1313% of the 6.4 ms timed region (budget 20%) |
| erf_f64_n1048576 | pytorch-cuda | erf_f64_n1048576 untimed 93.0 ms is 1332% of the 7.0 ms timed region (budget 20%) |
| sin_f64_n1048576 | pytorch-cuda | sin_f64_n1048576 untimed 91.6 ms is 1685% of the 5.4 ms timed region (budget 20%) |
| cos_f64_n1048576 | pytorch-cuda | cos_f64_n1048576 untimed 95.6 ms is 1746% of the 5.5 ms timed region (budget 20%) |
| tanh_f64_n1048576 | pytorch-cuda | tanh_f64_n1048576 untimed 90.7 ms is 2150% of the 4.2 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | pytorch-cuda | sqrt_f64_n1048576 untimed 82.9 ms is 2152% of the 3.9 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | pytorch-cuda | rsqrt_f64_n1048576 untimed 74.0 ms is 1913% of the 3.9 ms timed region (budget 20%) |
| compare_f64_n1048576 | pytorch-cuda | compare_f64_n1048576 untimed 88.2 ms is 8978% of the 1.0 ms timed region (budget 20%) |
| where_select_f64_n1048576 | pytorch-cuda | where_select_f64_n1048576 untimed 106.4 ms is 1582% of the 6.7 ms timed region (budget 20%) |
| clamp_f64_n1048576 | pytorch-cuda | clamp_f64_n1048576 untimed 79.2 ms is 1577% of the 5.0 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | pytorch-cuda | reduce_sum_f64_n1048576 untimed 31.3 ms is 111% of the 28.2 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | pytorch-cuda | reduce_prod_f64_n1048576 untimed 31.3 ms is 55% of the 56.5 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | pytorch-cuda | reduce_max_f64_n1048576 untimed 29.1 ms is 52% of the 56.3 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | pytorch-cuda | reduce_min_f64_n1048576 untimed 22.5 ms is 78% of the 28.8 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | pytorch-cuda | reduce_mean_f64_n1048576 untimed 32.6 ms is 59% of the 55.2 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | pytorch-cuda | scale_real_f64_n1048576 untimed 88.4 ms is 1886% of the 4.7 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | pytorch-cuda | scale_complex_c64_n1048576 untimed 185.1 ms is 3145% of the 5.9 ms timed region (budget 20%) |
| add_f64_n1048576 | jax-cuda | add_f64_n1048576 untimed 459.7 ms is 8184% of the 5.6 ms timed region (budget 20%) |
| sub_f64_n1048576 | jax-cuda | sub_f64_n1048576 untimed 136.8 ms is 1114% of the 12.3 ms timed region (budget 20%) |
| mul_f64_n1048576 | jax-cuda | mul_f64_n1048576 untimed 119.7 ms is 610% of the 19.6 ms timed region (budget 20%) |
| div_f64_n1048576 | jax-cuda | div_f64_n1048576 untimed 136.9 ms is 332% of the 41.3 ms timed region (budget 20%) |
| rem_f64_n1048576 | jax-cuda | rem_f64_n1048576 untimed 152.9 ms is 444% of the 34.5 ms timed region (budget 20%) |
| pow_f64_n1048576 | jax-cuda | pow_f64_n1048576 untimed 178.4 ms is 439% of the 40.6 ms timed region (budget 20%) |
| maximum_f64_n1048576 | jax-cuda | maximum_f64_n1048576 untimed 107.9 ms is 278% of the 38.8 ms timed region (budget 20%) |
| minimum_f64_n1048576 | jax-cuda | minimum_f64_n1048576 untimed 143.8 ms is 336% of the 42.8 ms timed region (budget 20%) |
| neg_f64_n1048576 | jax-cuda | neg_f64_n1048576 untimed 126.7 ms is 325% of the 39.0 ms timed region (budget 20%) |
| abs_f64_n1048576 | jax-cuda | abs_f64_n1048576 untimed 124.7 ms is 372% of the 33.5 ms timed region (budget 20%) |
| sign_f64_n1048576 | jax-cuda | sign_f64_n1048576 untimed 134.9 ms is 358% of the 37.7 ms timed region (budget 20%) |
| conj_c64_n1048576 | jax-cuda | conj_c64_n1048576 untimed 250.1 ms is 1446% of the 17.3 ms timed region (budget 20%) |
| exp_f64_n1048576 | jax-cuda | exp_f64_n1048576 untimed 149.9 ms is 415% of the 36.1 ms timed region (budget 20%) |
| expm1_f64_n1048576 | jax-cuda | expm1_f64_n1048576 untimed 139.5 ms is 419% of the 33.3 ms timed region (budget 20%) |
| log_f64_n1048576 | jax-cuda | log_f64_n1048576 untimed 149.2 ms is 397% of the 37.6 ms timed region (budget 20%) |
| log1p_f64_n1048576 | jax-cuda | log1p_f64_n1048576 untimed 172.5 ms is 533% of the 32.4 ms timed region (budget 20%) |
| erf_f64_n1048576 | jax-cuda | erf_f64_n1048576 untimed 154.5 ms is 352% of the 43.8 ms timed region (budget 20%) |
| sin_f64_n1048576 | jax-cuda | sin_f64_n1048576 untimed 184.3 ms is 565% of the 32.6 ms timed region (budget 20%) |
| cos_f64_n1048576 | jax-cuda | cos_f64_n1048576 untimed 177.3 ms is 532% of the 33.3 ms timed region (budget 20%) |
| tanh_f64_n1048576 | jax-cuda | tanh_f64_n1048576 untimed 139.1 ms is 412% of the 33.7 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | jax-cuda | sqrt_f64_n1048576 untimed 146.9 ms is 426% of the 34.4 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | jax-cuda | rsqrt_f64_n1048576 untimed 135.9 ms is 271% of the 50.2 ms timed region (budget 20%) |
| compare_f64_n1048576 | jax-cuda | compare_f64_n1048576 untimed 100.2 ms is 285% of the 35.2 ms timed region (budget 20%) |
| where_select_f64_n1048576 | jax-cuda | where_select_f64_n1048576 untimed 162.4 ms is 433% of the 37.5 ms timed region (budget 20%) |
| clamp_f64_n1048576 | jax-cuda | clamp_f64_n1048576 untimed 147.7 ms is 376% of the 39.3 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | jax-cuda | reduce_sum_f64_n1048576 untimed 207.8 ms is 484% of the 42.9 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | jax-cuda | reduce_prod_f64_n1048576 untimed 198.2 ms is 466% of the 42.5 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | jax-cuda | reduce_max_f64_n1048576 untimed 305.2 ms is 781% of the 39.1 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | jax-cuda | reduce_min_f64_n1048576 untimed 94.4 ms is 237% of the 39.9 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | jax-cuda | reduce_mean_f64_n1048576 untimed 83.0 ms is 215% of the 38.6 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | jax-cuda | scale_real_f64_n1048576 untimed 149.3 ms is 365% of the 40.9 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | jax-cuda | scale_complex_c64_n1048576 untimed 226.6 ms is 1204% of the 18.8 ms timed region (budget 20%) |