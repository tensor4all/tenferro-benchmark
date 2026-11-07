# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **124**. Unsupported cells: **8**, including **6** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1419.7 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261107_201728/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**14** of 33) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_f64_n1048576 † | 37.627 (0.396) | 102.344 (0.387) | 12.155 (0.039) | 67.641 (4.127) |
| sub_f64_n1048576 | 37.517 (0.212) | 101.371 (0.411) | 12.134 (0.083) | 66.329 (1.872) |
| mul_f64_n1048576 | 37.518 (0.309) | 102.236 (0.698) | 12.139 (0.078) | 60.612 (2.688) |
| div_f64_n1048576 | 37.693 (0.152) | 102.229 (0.272) | 13.122 (0.081) | 57.502 (1.996) |
| rem_f64_n1048576 † | 80.802 (0.766) | 138.557 (0.839) | 12.009 (0.062) | 53.587 (1.619) |
| pow_f64_n1048576 † | 80.168 (0.177) | 134.490 (0.975) | 35.053 (0.047) | 46.073 (11.084) |
| maximum_f64_n1048576 † | 63.520 (0.547) | 131.975 (0.519) | 12.068 (0.116) | 56.791 (8.154) |
| minimum_f64_n1048576 † | 63.356 (0.660) | 132.310 (1.230) | 12.105 (0.123) | 48.735 (2.287) |
| neg_f64_n1048576 | 32.255 (0.141) | 69.712 (0.360) | 9.635 (0.107) | 49.617 (2.664) |
| abs_f64_n1048576 | 32.207 (0.121) | 69.209 (0.334) | 9.827 (0.229) | 42.448 (1.188) |
| sign_f64_n1048576 | 32.430 (0.138) | 69.716 (0.274) | 9.542 (0.064) | 46.020 (1.547) |
| conj_c64_n1048576 | 46.398 (0.346) | 96.840 (0.319) | 25.386 (0.299) | 52.782 (5.312) |
| exp_f64_n1048576 | 33.135 (0.178) | 70.610 (0.467) | 10.016 (0.097) | 45.542 (0.969) |
| expm1_f64_n1048576 | 34.486 (0.125) | 71.553 (0.517) | 11.980 (0.064) | 47.156 (1.292) |
| log_f64_n1048576 | 37.246 (0.183) | 75.135 (0.132) | 15.249 (0.040) | 45.557 (1.226) |
| log1p_f64_n1048576 | 37.177 (0.076) | 74.740 (0.326) | 15.888 (0.052) | 44.194 (1.727) |
| erf_f64_n1048576 | 39.816 (0.348) | 77.670 (0.556) | 17.523 (0.049) | 61.539 (1.779) |
| sin_f64_n1048576 | 35.432 (0.157) | 73.827 (0.284) | 13.641 (0.059) | 45.171 (0.936) |
| cos_f64_n1048576 | 35.513 (0.157) | 73.739 (0.309) | 13.689 (0.071) | 45.076 (3.057) |
| tanh_f64_n1048576 | 33.286 (0.162) | 70.423 (0.345) | 10.110 (0.051) | 45.379 (0.758) |
| sqrt_f64_n1048576 | 32.830 (0.171) | 69.893 (0.382) | 9.653 (0.051) | 45.863 (1.131) |
| rsqrt_f64_n1048576 | 32.584 (0.129) | 69.963 (0.291) | 9.741 (0.057) | 66.778 (1.572) |
| compare_f64_n1048576 † | 28.658 (0.205) | 107.148 (1.015) | 8.663 (0.034) | 46.701 (1.836) |
| where_select_f64_n1048576 † | 39.917 (0.125) | 148.699 (0.632) | 14.331 (0.166) | 59.008 (3.829) |
| clamp_f64_n1048576 † | 109.764 (0.494) | 197.802 (2.024) | 12.084 (0.034) | 56.155 (3.885) |
| reduce_sum_f64_n1048576 † | 49.394 (0.191) | 90.826 (0.930) | 9.407 (0.071) | 111.527 (9.510) |
| reduce_prod_f64_n1048576 † | 48.148 (0.362) | 89.774 (0.214) | 8.879 (0.050) | 109.609 (10.868) |
| reduce_max_f64_n1048576 † | 42.776 (0.210) | 89.834 (0.286) | 9.404 (0.441) | 107.728 (5.966) |
| reduce_min_f64_n1048576 † | 42.884 (0.198) | 89.990 (0.352) | 9.156 (0.071) | 107.449 (3.455) |
| reduce_sum_squares_f64_n1048576 † | 42.195 (0.162) | 79.555 (10.789) | unsupported | unsupported |
| reduce_mean_f64_n1048576 † | unsupported | unsupported | 9.415 (0.268) | 59.843 (3.458) |
| scale_real_f64_n1048576 | unsupported | unsupported | 9.675 (0.220) | 57.694 (0.751) |
| scale_complex_c64_n1048576 | unsupported | unsupported | 25.717 (0.204) | 56.385 (4.901) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 1.358 | 16777216 | 0.278 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.314 | 16777216 | 1.203 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.031 | 16777216 | 0.367 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.215 | 16777216 | 0.171 | 0 |
| add_f64_n1048576 | jax-cuda | 32 | 15 | 2.165 | 25165824 | 0.262 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.389 | 25165824 | 1.458 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.204 | 25165824 | 0.471 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.637 | 25165824 | 0.173 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 1.797 | 16777232 | 0.210 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.387 | 16777232 | 0.978 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.512 | 16777232 | 0.108 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.165 | 16777232 | 0.060 | 0 |
| compare_f64_n1048576 | jax-cuda | 64 | 15 | 2.989 | 17825792 | 0.269 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.554 | 17825792 | 1.449 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 3.668 | 17825792 | 0.438 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.429 | 17825792 | 0.117 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 0.845 | 33554432 | 0.448 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.406 | 33554432 | 0.931 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.742 | 33554432 | 0.509 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.549 | 33554432 | 0.244 | 0 |
| cos_f64_n1048576 | jax-cuda | 32 | 15 | 1.442 | 16777216 | 0.262 | 5.905e-08 |
| cos_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.438 | 16777216 | 0.863 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.136 | 16777216 | 0.333 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.360 | 16777216 | 0.160 | 5.905e-08 |
| div_f64_n1048576 | jax-cuda | 32 | 15 | 1.840 | 25165824 | 0.308 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.420 | 25165824 | 1.351 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.206 | 25165824 | 0.470 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.271 | 25165824 | 0.173 | 0 |
| erf_f64_n1048576 | jax-cuda | 32 | 15 | 1.969 | 16777216 | 0.192 | 8.536e-08 |
| erf_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.561 | 16777216 | 0.674 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.274 | 16777216 | 0.297 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.485 | 16777216 | 0.152 | 8.536e-08 |
| exp_f64_n1048576 | jax-cuda | 32 | 15 | 1.457 | 16777216 | 0.259 | 1.095e-07 |
| exp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.321 | 16777216 | 1.180 | 1.095e-07 |
| exp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.060 | 16777216 | 0.357 | 1.095e-07 |
| exp_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.260 | 16777216 | 0.167 | 1.095e-07 |
| expm1_f64_n1048576 | jax-cuda | 32 | 15 | 1.509 | 16777216 | 0.251 | 7.39e-08 |
| expm1_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.383 | 16777216 | 0.986 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.104 | 16777216 | 0.343 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.290 | 16777216 | 0.165 | 7.39e-08 |
| log1p_f64_n1048576 | jax-cuda | 32 | 15 | 1.414 | 16777216 | 0.267 | 1.11e-07 |
| log1p_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.508 | 16777216 | 0.744 | 1.11e-07 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.190 | 16777216 | 0.318 | 1.11e-07 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.392 | 16777216 | 0.158 | 1.11e-07 |
| log_f64_n1048576 | jax-cuda | 32 | 15 | 1.458 | 16777216 | 0.259 | 4.308e-08 |
| log_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.488 | 16777216 | 0.775 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.192 | 16777216 | 0.317 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.404 | 16777216 | 0.157 | 4.308e-08 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 1.817 | 25165824 | 0.312 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.386 | 25165824 | 1.469 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.033 | 25165824 | 0.279 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.112 | 25165824 | 0.134 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 1.560 | 25165824 | 0.364 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.387 | 25165824 | 1.464 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.027 | 25165824 | 0.280 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.117 | 25165824 | 0.134 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 1.940 | 25165824 | 0.292 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.388 | 25165824 | 1.460 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.201 | 25165824 | 0.472 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.272 | 25165824 | 0.173 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 1.588 | 16777216 | 0.238 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.308 | 16777216 | 1.227 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.032 | 16777216 | 0.366 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.231 | 16777216 | 0.170 | 0 |
| pow_f64_n1048576 | jax-cuda | 32 | 15 | 1.474 | 25165824 | 0.385 | 1.48e-07 |
| pow_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.122 | 25165824 | 0.506 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.565 | 25165824 | 0.221 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.152 | 25165824 | 0.132 | 1.48e-07 |
| reduce_max_f64_n1048576 | jax-cuda | 32 | 15 | 3.447 | 8396800 | 0.055 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.407 | 8396800 | 0.629 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.738 | 8396800 | 0.138 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.875 | 8396800 | 0.066 | 0 |
| reduce_mean_f64_n1048576 | jax-cuda | 64 | 15 | 3.830 | 8396800 | 0.099 | 6.5e-09 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.410 | 8396800 | 0.628 | 3.9e-09 |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_min_f64_n1048576 | jax-cuda | 32 | 15 | 3.438 | 8396800 | 0.055 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.344 | 8396800 | 0.646 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.745 | 8396800 | 0.138 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.880 | 8396800 | 0.066 | 0 |
| reduce_prod_f64_n1048576 | jax-cuda | 32 | 15 | 3.507 | 8396800 | 0.054 | 4.436e-06 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.273 | 8396800 | 0.666 | 4.216e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.081 | 8396800 | 0.123 | 3.55e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.873 | 8396800 | 0.066 | 3.55e-06 |
| reduce_sum_f64_n1048576 | jax-cuda | 32 | 15 | 3.569 | 8396800 | 0.053 | 4.449e-06 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.408 | 8396800 | 0.629 | 2.961e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.161 | 8396800 | 0.120 | 3.391e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.906 | 8396800 | 0.065 | 3.391e-06 |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.701 | 8396800 | 0.140 | 3.294e-07 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.546 | 8396800 | 0.074 | 3.294e-07 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 1.715 | 25165824 | 0.331 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.384 | 25165824 | 1.476 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.586 | 25165824 | 0.219 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.217 | 25165824 | 0.128 | 0 |
| rsqrt_f64_n1048576 | jax-cuda | 32 | 15 | 2.137 | 16777216 | 0.177 | 6.504e-08 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.312 | 16777216 | 1.213 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.043 | 16777216 | 0.363 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.239 | 16777216 | 0.169 | 6.504e-08 |
| scale_complex_c64_n1048576 | jax-cuda | 16 | 15 | 0.902 | 33554448 | 0.419 | 4.129e-08 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.411 | 33554448 | 0.919 | 0 |
| scale_complex_c64_n1048576 | tenferro-cuda-direct | — | — | — | 33554448 | unavailable | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554448 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_f64_n1048576 | jax-cuda | 32 | 15 | 1.846 | 16777224 | 0.205 | 0 |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.310 | 16777224 | 1.221 | 0 |
| scale_real_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777224 | unavailable | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_real_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777224 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 1.473 | 16777216 | 0.257 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.305 | 16777216 | 1.238 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.038 | 16777216 | 0.364 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.231 | 16777216 | 0.170 | 0 |
| sin_f64_n1048576 | jax-cuda | 32 | 15 | 1.445 | 16777216 | 0.262 | 4.153e-08 |
| sin_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.437 | 16777216 | 0.866 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.134 | 16777216 | 0.334 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.362 | 16777216 | 0.160 | 4.153e-08 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 1.468 | 16777216 | 0.258 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.309 | 16777216 | 1.224 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.051 | 16777216 | 0.360 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.237 | 16777216 | 0.169 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 2.123 | 25165824 | 0.267 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.388 | 25165824 | 1.461 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.201 | 25165824 | 0.472 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.244 | 25165824 | 0.175 | 0 |
| tanh_f64_n1048576 | jax-cuda | 32 | 15 | 1.452 | 16777216 | 0.260 | 4.437e-08 |
| tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.324 | 16777216 | 1.169 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.065 | 16777216 | 0.355 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.254 | 16777216 | 0.168 | 4.437e-08 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 1.888 | 26214400 | 0.313 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.459 | 26214400 | 1.288 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.277 | 26214400 | 0.463 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.379 | 26214400 | 0.124 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261107_201728/gpu_isolated_ops.yaml
timestamp: '2026-11-07T20:17:28+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: ffd352bdb9f98537be747101bc278db8a50a1595
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: e7860ca369657a92a929026378d2a4c11adaae16
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
  cpu: AMD EPYC 7742 64-Core Processor
  logical_cpus: 256
  power:
    cpufreq_boost: '1'
    governor: schedutil
    scaling_max_freq_khz: '2250000'
    smt:
      active: '1'
      control: 'on'
  numa:
    nodes: 8
    cpulists:
      '0': 0-15,128-143
      '1': 16-31,144-159
      '2': 32-47,160-175
      '3': 48-63,176-191
      '4': 64-79,192-207
      '5': 80-95,208-223
      '6': 96-111,224-239
      '7': 112-127,240-255
  smt_siblings:
  - 0,128
  - 1,129
  - 2,130
  - 3,131
  - 4,132
  - 5,133
  - 6,134
  - 7,135
  - 8,136
  - 9,137
  - 10,138
  - 11,139
  - 12,140
  - 13,141
  - 14,142
  - 15,143
  - 16,144
  - 17,145
  - 18,146
  - 19,147
  - 20,148
  - 21,149
  - 22,150
  - 23,151
  - 24,152
  - 25,153
  - 26,154
  - 27,155
  - 28,156
  - 29,157
  - 30,158
  - 31,159
  - 32,160
  - 33,161
  - 34,162
  - 35,163
  - 36,164
  - 37,165
  - 38,166
  - 39,167
  - 40,168
  - 41,169
  - 42,170
  - 43,171
  - 44,172
  - 45,173
  - 46,174
  - 47,175
  - 48,176
  - 49,177
  - 50,178
  - 51,179
  - 52,180
  - 53,181
  - 54,182
  - 55,183
  - 56,184
  - 57,185
  - 58,186
  - 59,187
  - 60,188
  - 61,189
  - 62,190
  - 63,191
  - 64,192
  - 65,193
  - 66,194
  - 67,195
  - 68,196
  - 69,197
  - 70,198
  - 71,199
  - 72,200
  - 73,201
  - 74,202
  - 75,203
  - 76,204
  - 77,205
  - 78,206
  - 79,207
  - 80,208
  - 81,209
  - 82,210
  - 83,211
  - 84,212
  - 85,213
  - 86,214
  - 87,215
  - 88,216
  - 89,217
  - 90,218
  - 91,219
  - 92,220
  - 93,221
  - 94,222
  - 95,223
  - 96,224
  - 97,225
  - 98,226
  - 99,227
  - 100,228
  - 101,229
  - 102,230
  - 103,231
  - 104,232
  - 105,233
  - 106,234
  - 107,235
  - 108,236
  - 109,237
  - 110,238
  - 111,239
  - 112,240
  - 113,241
  - 114,242
  - 115,243
  - 116,244
  - 117,245
  - 118,246
  - 119,247
  - 120,248
  - 121,249
  - 122,250
  - 123,251
  - 124,252
  - 125,253
  - 126,254
  - 127,255
  l3_domains:
  - 0-3,128-131
  - 4-7,132-135
  - 8-11,136-139
  - 12-15,140-143
  - 16-19,144-147
  - 20-23,148-151
  - 24-27,152-155
  - 28-31,156-159
  - 32-35,160-163
  - 36-39,164-167
  - 40-43,168-171
  - 44-47,172-175
  - 48-51,176-179
  - 52-55,180-183
  - 56-59,184-187
  - 60-63,188-191
  - 64-67,192-195
  - 68-71,196-199
  - 72-75,200-203
  - 76-79,204-207
  - 80-83,208-211
  - 84-87,212-215
  - 88-91,216-219
  - 92-95,220-223
  - 96-99,224-227
  - 100-103,228-231
  - 104-107,232-235
  - 108-111,236-239
  - 112-115,240-243
  - 116-119,244-247
  - 120-123,248-251
  - 124-127,252-255
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: 86889f3412c3
  os: Linux-6.8.0-138-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-255
    l3_domains:
    - 0-3,128-131
    - 4-7,132-135
    - 8-11,136-139
    - 12-15,140-143
    - 16-19,144-147
    - 20-23,148-151
    - 24-27,152-155
    - 28-31,156-159
    - 32-35,160-163
    - 36-39,164-167
    - 40-43,168-171
    - 44-47,172-175
    - 48-51,176-179
    - 52-55,180-183
    - 56-59,184-187
    - 60-63,188-191
    - 64-67,192-195
    - 68-71,196-199
    - 72-75,200-203
    - 76-79,204-207
    - 80-83,208-211
    - 84-87,212-215
    - 88-91,216-219
    - 92-95,220-223
    - 96-99,224-227
    - 100-103,228-231
    - 104-107,232-235
    - 108-111,236-239
    - 112-115,240-243
    - 116-119,244-247
    - 120-123,248-251
    - 124-127,252-255
    cpu_boost: enabled
    scaling_governor: schedutil
    scaling_driver: acpi-cpufreq
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
  device_name: NVIDIA A100-SXM4-80GB
  device_uuid: GPU-3f509591-db26-6474-253e-fcb29d2b374b
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
  device_name: NVIDIA A100-SXM4-80GB
  buffer_bytes: 67108864
  traffic_bytes_per_copy: 134217728
  traffic_model: one read plus one write of buffer_bytes per copy
  warmups: 3
  sample_ms:
  - 0.09522
  - 0.094539
  - 0.094529
  - 0.097274
  - 0.09506
  - 0.093376
  - 0.094409
  - 0.097223
  - 0.096723
  - 0.094428
  - 0.095691
  - 0.09525
  - 0.093647
  - 0.093116
  - 0.093397
  median_ms: 0.094539
  bytes_per_second: 1419707506954.8018
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-07T20:35:34.037384+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-3f509591-db26-6474-253e-fcb29d2b374b
  cuda_visible_devices: GPU-3f509591-db26-6474-253e-fcb29d2b374b
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
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 1919e33cc5f2d0208901837f80cb7c0600426462f998be6dd933062d9f6bb345
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 788b9fde4f8ca4162921baef1203b2e31bb747ec30327a425de3442cf872df52
    data/results/nvidia-gpu/gpu/isolated_ops/20261107_201728/fixtures/manifest.json: 3c1a75f9c57996b228c04da1d4cbf34e4137a074e44ddca9f1c37e5824580bd8
    target/release/benchmark_gpu_public_api: 94cf356a747431a654002f07d7e16069b1849c82e215e2fe5d75a94346f4ae17
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-07T20:37:29Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-07T20:36:44Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-07T20:35:58Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-07T20:36:03Z'
  compile:
    case_wall_seconds: 88.017
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
| add_f64_n1048576 | tenferro-cuda-direct | add_f64_n1048576 untimed 154.9 ms is 853% of the 18.2 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-direct | sub_f64_n1048576 untimed 111.2 ms is 617% of the 18.0 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-direct | mul_f64_n1048576 untimed 111.3 ms is 618% of the 18.0 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-direct | div_f64_n1048576 untimed 115.0 ms is 633% of the 18.2 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-direct | rem_f64_n1048576 untimed 115.9 ms is 298% of the 39.0 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-direct | pow_f64_n1048576 untimed 111.1 ms is 288% of the 38.5 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-direct | maximum_f64_n1048576 untimed 108.1 ms is 353% of the 30.7 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-direct | minimum_f64_n1048576 untimed 108.5 ms is 356% of the 30.5 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-direct | neg_f64_n1048576 untimed 103.9 ms is 672% of the 15.5 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-direct | abs_f64_n1048576 untimed 98.5 ms is 632% of the 15.6 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-direct | sign_f64_n1048576 untimed 103.2 ms is 657% of the 15.7 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-direct | conj_c64_n1048576 untimed 147.4 ms is 1326% of the 11.1 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-direct | exp_f64_n1048576 untimed 103.1 ms is 648% of the 15.9 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-direct | expm1_f64_n1048576 untimed 104.5 ms is 630% of the 16.6 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-direct | log_f64_n1048576 untimed 99.1 ms is 554% of the 17.9 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-direct | log1p_f64_n1048576 untimed 103.6 ms is 580% of the 17.9 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-direct | erf_f64_n1048576 untimed 108.9 ms is 570% of the 19.1 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-direct | sin_f64_n1048576 untimed 99.0 ms is 582% of the 17.0 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-direct | cos_f64_n1048576 untimed 99.9 ms is 586% of the 17.0 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-direct | tanh_f64_n1048576 untimed 101.0 ms is 630% of the 16.0 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-direct | sqrt_f64_n1048576 untimed 98.8 ms is 627% of the 15.8 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | rsqrt_f64_n1048576 untimed 104.1 ms is 665% of the 15.6 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-direct | compare_f64_n1048576 untimed 79.9 ms is 144% of the 55.3 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-direct | where_select_f64_n1048576 untimed 99.6 ms is 517% of the 19.3 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-direct | clamp_f64_n1048576 untimed 107.9 ms is 204% of the 52.9 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | reduce_sum_f64_n1048576 untimed 20.8 ms is 44% of the 47.6 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | reduce_prod_f64_n1048576 untimed 12.6 ms is 28% of the 45.3 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | reduce_max_f64_n1048576 untimed 10.5 ms is 26% of the 41.1 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | reduce_min_f64_n1048576 untimed 10.6 ms is 26% of the 41.3 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_f64_n1048576 untimed 10.6 ms is 26% of the 40.6 ms timed region (budget 20%) |
| add_f64_n1048576 | tenferro-cuda-eager | add_f64_n1048576 untimed 174.3 ms is 709% of the 24.6 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-eager | sub_f64_n1048576 untimed 90.3 ms is 185% of the 48.7 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-eager | mul_f64_n1048576 untimed 105.7 ms is 215% of the 49.1 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-eager | div_f64_n1048576 untimed 116.8 ms is 236% of the 49.5 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-eager | rem_f64_n1048576 untimed 106.8 ms is 321% of the 33.2 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-eager | pow_f64_n1048576 untimed 103.8 ms is 320% of the 32.5 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-eager | maximum_f64_n1048576 untimed 102.8 ms is 324% of the 31.7 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-eager | minimum_f64_n1048576 untimed 102.6 ms is 322% of the 31.9 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-eager | neg_f64_n1048576 untimed 92.8 ms is 277% of the 33.5 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-eager | abs_f64_n1048576 untimed 80.5 ms is 242% of the 33.3 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-eager | sign_f64_n1048576 untimed 80.8 ms is 241% of the 33.5 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-eager | conj_c64_n1048576 untimed 128.2 ms is 550% of the 23.3 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-eager | exp_f64_n1048576 untimed 93.6 ms is 275% of the 34.0 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-eager | expm1_f64_n1048576 untimed 80.5 ms is 234% of the 34.4 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-eager | log_f64_n1048576 untimed 80.6 ms is 223% of the 36.1 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-eager | log1p_f64_n1048576 untimed 84.7 ms is 237% of the 35.8 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-eager | erf_f64_n1048576 untimed 90.2 ms is 241% of the 37.5 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-eager | sin_f64_n1048576 untimed 80.7 ms is 227% of the 35.6 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-eager | cos_f64_n1048576 untimed 81.5 ms is 230% of the 35.4 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-eager | tanh_f64_n1048576 untimed 82.8 ms is 243% of the 34.0 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-eager | sqrt_f64_n1048576 untimed 80.0 ms is 237% of the 33.8 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | rsqrt_f64_n1048576 untimed 85.8 ms is 255% of the 33.7 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-eager | compare_f64_n1048576 untimed 61.6 ms is 119% of the 51.9 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-eager | where_select_f64_n1048576 untimed 103.7 ms is 291% of the 35.7 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-eager | clamp_f64_n1048576 untimed 95.2 ms is 200% of the 47.7 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | reduce_sum_f64_n1048576 untimed 12.1 ms is 28% of the 43.8 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | reduce_prod_f64_n1048576 untimed 10.6 ms is 25% of the 43.1 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | reduce_max_f64_n1048576 untimed 10.6 ms is 25% of the 43.3 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | reduce_min_f64_n1048576 untimed 10.6 ms is 25% of the 43.2 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | reduce_sum_squares_f64_n1048576 untimed 10.7 ms is 27% of the 39.9 ms timed region (budget 20%) |
| add_f64_n1048576 | pytorch-cuda | add_f64_n1048576 untimed 1660.6 ms is 28446% of the 5.8 ms timed region (budget 20%) |
| sub_f64_n1048576 | pytorch-cuda | sub_f64_n1048576 untimed 1418.4 ms is 24229% of the 5.9 ms timed region (budget 20%) |
| mul_f64_n1048576 | pytorch-cuda | mul_f64_n1048576 untimed 1385.2 ms is 23638% of the 5.9 ms timed region (budget 20%) |
| div_f64_n1048576 | pytorch-cuda | div_f64_n1048576 untimed 1370.6 ms is 21746% of the 6.3 ms timed region (budget 20%) |
| rem_f64_n1048576 | pytorch-cuda | rem_f64_n1048576 untimed 1381.0 ms is 23948% of the 5.8 ms timed region (budget 20%) |
| pow_f64_n1048576 | pytorch-cuda | pow_f64_n1048576 untimed 1392.2 ms is 8185% of the 17.0 ms timed region (budget 20%) |
| maximum_f64_n1048576 | pytorch-cuda | maximum_f64_n1048576 untimed 1371.4 ms is 23688% of the 5.8 ms timed region (budget 20%) |
| minimum_f64_n1048576 | pytorch-cuda | minimum_f64_n1048576 untimed 1389.0 ms is 23880% of the 5.8 ms timed region (budget 20%) |
| neg_f64_n1048576 | pytorch-cuda | neg_f64_n1048576 untimed 1360.6 ms is 29036% of the 4.7 ms timed region (budget 20%) |
| abs_f64_n1048576 | pytorch-cuda | abs_f64_n1048576 untimed 1393.8 ms is 29351% of the 4.7 ms timed region (budget 20%) |
| sign_f64_n1048576 | pytorch-cuda | sign_f64_n1048576 untimed 1405.4 ms is 30601% of the 4.6 ms timed region (budget 20%) |
| conj_c64_n1048576 | pytorch-cuda | conj_c64_n1048576 untimed 1405.6 ms is 23107% of the 6.1 ms timed region (budget 20%) |
| exp_f64_n1048576 | pytorch-cuda | exp_f64_n1048576 untimed 1383.6 ms is 28661% of the 4.8 ms timed region (budget 20%) |
| expm1_f64_n1048576 | pytorch-cuda | expm1_f64_n1048576 untimed 1362.4 ms is 22980% of the 5.9 ms timed region (budget 20%) |
| log_f64_n1048576 | pytorch-cuda | log_f64_n1048576 untimed 1411.4 ms is 19278% of the 7.3 ms timed region (budget 20%) |
| log1p_f64_n1048576 | pytorch-cuda | log1p_f64_n1048576 untimed 1364.4 ms is 17895% of the 7.6 ms timed region (budget 20%) |
| erf_f64_n1048576 | pytorch-cuda | erf_f64_n1048576 untimed 1369.3 ms is 16255% of the 8.4 ms timed region (budget 20%) |
| sin_f64_n1048576 | pytorch-cuda | sin_f64_n1048576 untimed 1362.1 ms is 20762% of the 6.6 ms timed region (budget 20%) |
| cos_f64_n1048576 | pytorch-cuda | cos_f64_n1048576 untimed 1414.7 ms is 21447% of the 6.6 ms timed region (budget 20%) |
| tanh_f64_n1048576 | pytorch-cuda | tanh_f64_n1048576 untimed 1369.3 ms is 28163% of the 4.9 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | pytorch-cuda | sqrt_f64_n1048576 untimed 1361.9 ms is 29329% of the 4.6 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | pytorch-cuda | rsqrt_f64_n1048576 untimed 1410.8 ms is 30140% of the 4.7 ms timed region (budget 20%) |
| compare_f64_n1048576 | pytorch-cuda | compare_f64_n1048576 untimed 1361.6 ms is 16365% of the 8.3 ms timed region (budget 20%) |
| where_select_f64_n1048576 | pytorch-cuda | where_select_f64_n1048576 untimed 1380.8 ms is 20089% of the 6.9 ms timed region (budget 20%) |
| clamp_f64_n1048576 | pytorch-cuda | clamp_f64_n1048576 untimed 1397.4 ms is 24087% of the 5.8 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | pytorch-cuda | reduce_sum_f64_n1048576 untimed 35.8 ms is 99% of the 36.2 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | pytorch-cuda | reduce_prod_f64_n1048576 untimed 22.5 ms is 65% of the 34.3 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | pytorch-cuda | reduce_max_f64_n1048576 untimed 24.5 ms is 66% of the 37.1 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | pytorch-cuda | reduce_min_f64_n1048576 untimed 24.6 ms is 69% of the 35.4 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | pytorch-cuda | reduce_mean_f64_n1048576 untimed 26.3 ms is 71% of the 37.2 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | pytorch-cuda | scale_real_f64_n1048576 untimed 1390.9 ms is 29791% of the 4.7 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | pytorch-cuda | scale_complex_c64_n1048576 untimed 1548.1 ms is 25068% of the 6.2 ms timed region (budget 20%) |
| add_f64_n1048576 | jax-cuda | add_f64_n1048576 untimed 1824.3 ms is 5594% of the 32.6 ms timed region (budget 20%) |
| sub_f64_n1048576 | jax-cuda | sub_f64_n1048576 untimed 1456.8 ms is 4561% of the 31.9 ms timed region (budget 20%) |
| mul_f64_n1048576 | jax-cuda | mul_f64_n1048576 untimed 1441.5 ms is 4793% of the 30.1 ms timed region (budget 20%) |
| div_f64_n1048576 | jax-cuda | div_f64_n1048576 untimed 1458.4 ms is 5243% of the 27.8 ms timed region (budget 20%) |
| rem_f64_n1048576 | jax-cuda | rem_f64_n1048576 untimed 1527.5 ms is 5901% of the 25.9 ms timed region (budget 20%) |
| pow_f64_n1048576 | jax-cuda | pow_f64_n1048576 untimed 1477.4 ms is 6274% of the 23.5 ms timed region (budget 20%) |
| maximum_f64_n1048576 | jax-cuda | maximum_f64_n1048576 untimed 1447.2 ms is 5614% of the 25.8 ms timed region (budget 20%) |
| minimum_f64_n1048576 | jax-cuda | minimum_f64_n1048576 untimed 1432.6 ms is 6108% of the 23.5 ms timed region (budget 20%) |
| neg_f64_n1048576 | jax-cuda | neg_f64_n1048576 untimed 1422.4 ms is 6052% of the 23.5 ms timed region (budget 20%) |
| abs_f64_n1048576 | jax-cuda | abs_f64_n1048576 untimed 1432.6 ms is 7031% of the 20.4 ms timed region (budget 20%) |
| sign_f64_n1048576 | jax-cuda | sign_f64_n1048576 untimed 1454.4 ms is 6661% of the 21.8 ms timed region (budget 20%) |
| conj_c64_n1048576 | jax-cuda | conj_c64_n1048576 untimed 1501.4 ms is 12040% of the 12.5 ms timed region (budget 20%) |
| exp_f64_n1048576 | jax-cuda | exp_f64_n1048576 untimed 1492.6 ms is 6893% of the 21.7 ms timed region (budget 20%) |
| expm1_f64_n1048576 | jax-cuda | expm1_f64_n1048576 untimed 1469.3 ms is 6404% of the 22.9 ms timed region (budget 20%) |
| log_f64_n1048576 | jax-cuda | log_f64_n1048576 untimed 1496.2 ms is 6866% of the 21.8 ms timed region (budget 20%) |
| log1p_f64_n1048576 | jax-cuda | log1p_f64_n1048576 untimed 1481.6 ms is 6996% of the 21.2 ms timed region (budget 20%) |
| erf_f64_n1048576 | jax-cuda | erf_f64_n1048576 untimed 1462.1 ms is 4929% of the 29.7 ms timed region (budget 20%) |
| sin_f64_n1048576 | jax-cuda | sin_f64_n1048576 untimed 1496.4 ms is 6779% of the 22.1 ms timed region (budget 20%) |
| cos_f64_n1048576 | jax-cuda | cos_f64_n1048576 untimed 1537.5 ms is 7079% of the 21.7 ms timed region (budget 20%) |
| tanh_f64_n1048576 | jax-cuda | tanh_f64_n1048576 untimed 1462.2 ms is 6662% of the 22.0 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | jax-cuda | sqrt_f64_n1048576 untimed 1459.5 ms is 6666% of the 21.9 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | jax-cuda | rsqrt_f64_n1048576 untimed 1474.3 ms is 4596% of the 32.1 ms timed region (budget 20%) |
| compare_f64_n1048576 | jax-cuda | compare_f64_n1048576 untimed 1448.4 ms is 3172% of the 45.7 ms timed region (budget 20%) |
| where_select_f64_n1048576 | jax-cuda | where_select_f64_n1048576 untimed 1474.9 ms is 5232% of the 28.2 ms timed region (budget 20%) |
| clamp_f64_n1048576 | jax-cuda | clamp_f64_n1048576 untimed 1470.5 ms is 5491% of the 26.8 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | jax-cuda | reduce_sum_f64_n1048576 untimed 301.2 ms is 553% of the 54.5 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | jax-cuda | reduce_prod_f64_n1048576 untimed 302.0 ms is 520% of the 58.1 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | jax-cuda | reduce_max_f64_n1048576 untimed 435.8 ms is 830% of the 52.5 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | jax-cuda | reduce_min_f64_n1048576 untimed 151.3 ms is 282% of the 53.7 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | jax-cuda | reduce_mean_f64_n1048576 untimed 134.7 ms is 231% of the 58.2 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | jax-cuda | scale_real_f64_n1048576 untimed 1441.0 ms is 5181% of the 27.8 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | jax-cuda | scale_complex_c64_n1048576 untimed 1487.0 ms is 10896% of the 13.6 ms timed region (budget 20%) |