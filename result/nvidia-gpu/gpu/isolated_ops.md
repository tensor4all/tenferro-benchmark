# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **124**. Unsupported cells: **8**, including **6** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1412.2 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261121_082311/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**9** of 33) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_f64_n1048576 † | 22.965 (0.313) | 73.153 (0.742) | 12.913 (0.328) | 62.725 (1.102) |
| sub_f64_n1048576 | 22.907 (0.184) | 73.371 (0.792) | 12.070 (0.265) | 63.923 (0.468) |
| mul_f64_n1048576 | 22.877 (0.249) | 73.390 (1.019) | 12.380 (0.040) | 67.042 (1.294) |
| div_f64_n1048576 | 22.749 (0.150) | 73.263 (0.614) | 13.048 (0.190) | 68.030 (11.030) |
| rem_f64_n1048576 | 41.123 (0.229) | 94.985 (1.477) | 12.281 (0.156) | 58.822 (0.965) |
| pow_f64_n1048576 | 43.579 (0.393) | 95.924 (0.480) | 29.308 (0.038) | 66.228 (1.222) |
| maximum_f64_n1048576 | 29.438 (0.255) | 81.245 (0.638) | 12.304 (0.246) | 67.626 (1.489) |
| minimum_f64_n1048576 | 29.584 (0.242) | 81.264 (0.986) | 12.350 (0.190) | 68.908 (1.424) |
| neg_f64_n1048576 | 18.724 (0.276) | 42.223 (0.423) | 7.955 (0.069) | 65.048 (5.550) |
| abs_f64_n1048576 | 18.804 (0.334) | 42.084 (0.303) | 7.947 (0.041) | 65.932 (1.300) |
| sign_f64_n1048576 | 19.154 (0.309) | 42.553 (0.430) | 7.938 (0.052) | 66.460 (10.050) |
| conj_c64_n1048576 | 34.868 (0.277) | 64.191 (0.938) | 25.241 (0.184) | 65.124 (2.476) |
| exp_f64_n1048576 | 19.578 (0.110) | 42.589 (0.715) | 8.310 (0.050) | 66.171 (2.606) |
| expm1_f64_n1048576 | 20.445 (0.182) | 43.832 (0.826) | 9.880 (0.178) | 66.497 (1.359) |
| log_f64_n1048576 | 22.350 (0.203) | 46.462 (0.675) | 12.666 (0.097) | 67.078 (2.795) |
| log1p_f64_n1048576 | 22.302 (0.500) | 46.301 (0.443) | 13.094 (0.047) | 65.683 (2.015) |
| erf_f64_n1048576 | 25.152 (0.135) | 48.446 (0.582) | 14.493 (0.030) | 84.984 (5.856) |
| sin_f64_n1048576 | 21.476 (0.188) | 44.641 (0.336) | 11.274 (0.030) | 64.523 (1.717) |
| cos_f64_n1048576 | 22.853 (1.370) | 45.116 (0.744) | 11.340 (0.052) | 52.338 (1.923) |
| tanh_f64_n1048576 | 20.899 (0.483) | 42.289 (0.651) | 8.234 (0.045) | 53.754 (0.880) |
| sqrt_f64_n1048576 | 20.040 (0.187) | 42.320 (0.375) | 8.008 (0.050) | 57.104 (0.905) |
| rsqrt_f64_n1048576 | 19.764 (0.197) | 41.904 (0.512) | 7.967 (0.042) | 77.907 (0.992) |
| compare_f64_n1048576 † | 18.681 (0.172) | 69.387 (0.585) | 7.443 (0.089) | 58.673 (7.847) |
| where_select_f64_n1048576 | 24.567 (0.135) | 97.957 (0.496) | 14.248 (0.105) | 75.534 (1.240) |
| clamp_f64_n1048576 † | 68.104 (0.116) | 124.807 (1.368) | 10.342 (0.047) | 58.991 (3.556) |
| reduce_sum_f64_n1048576 † | 25.777 (0.135) | 48.309 (0.458) | 7.191 (0.011) | 126.829 (1.568) |
| reduce_prod_f64_n1048576 † | 26.739 (1.379) | 48.743 (1.605) | 7.275 (0.007) | 127.337 (2.312) |
| reduce_max_f64_n1048576 † | 26.624 (0.080) | 49.366 (0.604) | 7.328 (0.009) | 127.742 (2.186) |
| reduce_min_f64_n1048576 † | 27.474 (1.066) | 49.610 (0.989) | 7.348 (0.006) | 122.902 (8.448) |
| reduce_sum_squares_f64_n1048576 † | 25.552 (0.263) | 48.767 (0.417) | unsupported | unsupported |
| reduce_mean_f64_n1048576 † | unsupported | unsupported | 7.145 (0.004) | 130.147 (30.493) |
| scale_real_f64_n1048576 | unsupported | unsupported | 9.584 (0.059) | 75.133 (5.550) |
| scale_complex_c64_n1048576 | unsupported | unsupported | 25.656 (0.070) | 77.029 (2.990) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 2.110 | 16777216 | 0.180 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.254 | 16777216 | 1.495 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.602 | 16777216 | 0.632 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.347 | 16777216 | 0.282 | 0 |
| add_f64_n1048576 | jax-cuda | 32 | 15 | 2.007 | 25165824 | 0.284 | 0 |
| add_f64_n1048576 | pytorch-cuda | 8 | 15 | 0.103 | 25165824 | 1.380 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.735 | 25165824 | 0.776 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.170 | 25165824 | 0.244 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 1.888 | 16777232 | 0.201 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.331 | 16777232 | 1.149 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.179 | 16777232 | 0.174 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.997 | 16777232 | 0.095 | 0 |
| compare_f64_n1048576 | jax-cuda | 64 | 15 | 3.755 | 17825792 | 0.215 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.119 | 17825792 | 1.696 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.391 | 17825792 | 0.676 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.220 | 17825792 | 0.182 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 1.042 | 33554432 | 0.365 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.404 | 33554432 | 0.941 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.558 | 33554432 | 0.681 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.027 | 33554432 | 0.370 | 0 |
| cos_f64_n1048576 | jax-cuda | 32 | 15 | 1.675 | 16777216 | 0.227 | 5.905e-08 |
| cos_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.363 | 16777216 | 1.048 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.731 | 16777216 | 0.520 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.444 | 16777216 | 0.263 | 5.905e-08 |
| div_f64_n1048576 | jax-cuda | 32 | 15 | 2.177 | 25165824 | 0.262 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.418 | 25165824 | 1.366 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.728 | 25165824 | 0.783 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.344 | 25165824 | 0.243 | 0 |
| erf_f64_n1048576 | jax-cuda | 32 | 15 | 2.719 | 16777216 | 0.140 | 8.536e-08 |
| erf_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.464 | 16777216 | 0.820 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.805 | 16777216 | 0.472 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.550 | 16777216 | 0.245 | 8.536e-08 |
| exp_f64_n1048576 | jax-cuda | 32 | 15 | 2.117 | 16777216 | 0.180 | 1.095e-07 |
| exp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.266 | 16777216 | 1.430 | 1.095e-07 |
| exp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.626 | 16777216 | 0.607 | 1.095e-07 |
| exp_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.363 | 16777216 | 0.279 | 1.095e-07 |
| expm1_f64_n1048576 | jax-cuda | 32 | 15 | 2.128 | 16777216 | 0.179 | 7.39e-08 |
| expm1_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.316 | 16777216 | 1.202 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.654 | 16777216 | 0.581 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.403 | 16777216 | 0.271 | 7.39e-08 |
| log1p_f64_n1048576 | jax-cuda | 32 | 15 | 2.102 | 16777216 | 0.181 | 1.11e-07 |
| log1p_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.419 | 16777216 | 0.907 | 1.11e-07 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.714 | 16777216 | 0.533 | 1.11e-07 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.482 | 16777216 | 0.257 | 1.11e-07 |
| log_f64_n1048576 | jax-cuda | 32 | 15 | 2.147 | 16777216 | 0.177 | 4.308e-08 |
| log_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.405 | 16777216 | 0.938 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.715 | 16777216 | 0.532 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.487 | 16777216 | 0.256 | 4.308e-08 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 2.164 | 25165824 | 0.264 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.394 | 25165824 | 1.448 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.942 | 25165824 | 0.605 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.600 | 25165824 | 0.219 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 2.205 | 25165824 | 0.259 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.395 | 25165824 | 1.443 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.947 | 25165824 | 0.602 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.600 | 25165824 | 0.219 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 2.145 | 25165824 | 0.266 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.396 | 25165824 | 1.439 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.732 | 25165824 | 0.779 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.348 | 25165824 | 0.243 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 2.082 | 16777216 | 0.183 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.255 | 16777216 | 1.493 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.599 | 16777216 | 0.635 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.351 | 16777216 | 0.281 | 0 |
| pow_f64_n1048576 | jax-cuda | 32 | 15 | 2.119 | 25165824 | 0.269 | 1.48e-07 |
| pow_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.938 | 25165824 | 0.608 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.395 | 25165824 | 0.409 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.070 | 25165824 | 0.186 | 1.48e-07 |
| reduce_max_f64_n1048576 | jax-cuda | 16 | 15 | 2.044 | 8396800 | 0.047 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.752 | 8396800 | 0.811 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 3.408 | 8396800 | 0.223 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.159 | 8396800 | 0.120 | 0 |
| reduce_mean_f64_n1048576 | jax-cuda | 16 | 15 | 2.082 | 8396800 | 0.046 | 6.5e-09 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.658 | 8396800 | 0.832 | 3.9e-09 |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_min_f64_n1048576 | jax-cuda | 16 | 15 | 1.966 | 8396800 | 0.048 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.762 | 8396800 | 0.809 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 3.517 | 8396800 | 0.216 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.175 | 8396800 | 0.120 | 0 |
| reduce_prod_f64_n1048576 | jax-cuda | 16 | 15 | 2.037 | 8396800 | 0.047 | 4.436e-06 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.725 | 8396800 | 0.817 | 4.216e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 3.423 | 8396800 | 0.222 | 4.438e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.120 | 8396800 | 0.122 | 4.438e-06 |
| reduce_sum_f64_n1048576 | jax-cuda | 16 | 15 | 2.029 | 8396800 | 0.047 | 4.449e-06 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.682 | 8396800 | 0.827 | 2.961e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 3.300 | 8396800 | 0.231 | 3.305e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.092 | 8396800 | 0.123 | 3.305e-06 |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 3.271 | 8396800 | 0.233 | 4.939e-07 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.121 | 8396800 | 0.122 | 4.939e-07 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 1.882 | 25165824 | 0.303 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.393 | 25165824 | 1.451 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.316 | 25165824 | 0.433 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.040 | 25165824 | 0.188 | 0 |
| rsqrt_f64_n1048576 | jax-cuda | 32 | 15 | 2.493 | 16777216 | 0.152 | 6.504e-08 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.255 | 16777216 | 1.491 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.632 | 16777216 | 0.601 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.341 | 16777216 | 0.284 | 6.504e-08 |
| scale_complex_c64_n1048576 | jax-cuda | 16 | 15 | 1.232 | 33554448 | 0.308 | 4.129e-08 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.410 | 33554448 | 0.926 | 0 |
| scale_complex_c64_n1048576 | tenferro-cuda-direct | — | — | — | 33554448 | unavailable | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554448 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_f64_n1048576 | jax-cuda | 32 | 15 | 2.404 | 16777224 | 0.158 | 0 |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.307 | 16777224 | 1.240 | 0 |
| scale_real_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777224 | unavailable | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_real_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777224 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 2.127 | 16777216 | 0.179 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.254 | 16777216 | 1.497 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.613 | 16777216 | 0.620 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.362 | 16777216 | 0.279 | 0 |
| sin_f64_n1048576 | jax-cuda | 32 | 15 | 2.065 | 16777216 | 0.184 | 4.153e-08 |
| sin_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.361 | 16777216 | 1.054 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.687 | 16777216 | 0.553 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.429 | 16777216 | 0.266 | 4.153e-08 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 1.827 | 16777216 | 0.208 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.256 | 16777216 | 1.483 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.641 | 16777216 | 0.593 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.354 | 16777216 | 0.281 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 2.046 | 25165824 | 0.279 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.386 | 25165824 | 1.476 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.733 | 25165824 | 0.778 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.348 | 25165824 | 0.243 | 0 |
| tanh_f64_n1048576 | jax-cuda | 32 | 15 | 1.720 | 16777216 | 0.221 | 4.437e-08 |
| tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.263 | 16777216 | 1.443 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.669 | 16777216 | 0.568 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.353 | 16777216 | 0.281 | 4.437e-08 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 2.417 | 26214400 | 0.246 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.456 | 26214400 | 1.303 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.786 | 26214400 | 0.756 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.135 | 26214400 | 0.189 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261121_082311/gpu_isolated_ops.yaml
timestamp: '2026-11-21T08:23:11+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: ce9125d0cff4ee40841120759b5df45252c27f22
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: a741cd656e4d51e77d8369154bf789e5f0147f5c
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
  cpu: AMD EPYC 7543 32-Core Processor
  logical_cpus: 128
  power:
    cpufreq_boost: '1'
    governor: schedutil
    scaling_max_freq_khz: '2800000'
    smt:
      active: '1'
      control: 'on'
  numa:
    nodes: 2
    cpulists:
      '0': 0-31,64-95
      '1': 32-63,96-127
  smt_siblings:
  - 0,64
  - 1,65
  - 2,66
  - 3,67
  - 4,68
  - 5,69
  - 6,70
  - 7,71
  - 8,72
  - 9,73
  - 10,74
  - 11,75
  - 12,76
  - 13,77
  - 14,78
  - 15,79
  - 16,80
  - 17,81
  - 18,82
  - 19,83
  - 20,84
  - 21,85
  - 22,86
  - 23,87
  - 24,88
  - 25,89
  - 26,90
  - 27,91
  - 28,92
  - 29,93
  - 30,94
  - 31,95
  - 32,96
  - 33,97
  - 34,98
  - 35,99
  - 36,100
  - 37,101
  - 38,102
  - 39,103
  - 40,104
  - 41,105
  - 42,106
  - 43,107
  - 44,108
  - 45,109
  - 46,110
  - 47,111
  - 48,112
  - 49,113
  - 50,114
  - 51,115
  - 52,116
  - 53,117
  - 54,118
  - 55,119
  - 56,120
  - 57,121
  - 58,122
  - 59,123
  - 60,124
  - 61,125
  - 62,126
  - 63,127
  l3_domains:
  - 0-3,64-67
  - 4-7,68-71
  - 8-11,72-75
  - 12-15,76-79
  - 16-19,80-83
  - 20-23,84-87
  - 24-27,88-91
  - 28-31,92-95
  - 32-35,96-99
  - 36-39,100-103
  - 40-43,104-107
  - 44-47,108-111
  - 48-51,112-115
  - 52-55,116-119
  - 56-59,120-123
  - 60-63,124-127
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: 7894e3c40841
  os: Linux-6.8.0-138-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-127
    l3_domains:
    - 0-3,64-67
    - 4-7,68-71
    - 8-11,72-75
    - 12-15,76-79
    - 16-19,80-83
    - 20-23,84-87
    - 24-27,88-91
    - 28-31,92-95
    - 32-35,96-99
    - 36-39,100-103
    - 40-43,104-107
    - 44-47,108-111
    - 48-51,112-115
    - 52-55,116-119
    - 56-59,120-123
    - 60-63,124-127
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
    GPU_RUNTIME_SHA256: 3d603f753ad72a63b928ce8e1b63847b8a21f94f0919276124993a6548126185
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
  device_uuid: GPU-98d8618e-77c8-160a-92f2-117e34fd43df
  device_memory_bytes: 85899345920
  driver_version: 595.112.07
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
  - 0.093701
  - 0.093641
  - 0.095661
  - 0.09642
  - 0.09259
  - 0.095541
  - 0.094881
  - 0.094611
  - 0.095651
  - 0.095041
  - 0.093701
  - 0.093681
  - 0.096531
  - 0.09567
  - 0.095751
  median_ms: 0.095041
  bytes_per_second: 1412208709925.1902
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-21T08:35:21.319924+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-98d8618e-77c8-160a-92f2-117e34fd43df
  cuda_visible_devices: GPU-98d8618e-77c8-160a-92f2-117e34fd43df
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh isolated_ops
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 33
  selected_cases: 33
  sha256:
    .devcontainer/cuda/devcontainer.json: 402f3763533ea1531d0e400193395302213fd1b151a42baaeaebeafca3acabcd
    .devcontainer/cuda/Dockerfile: 6eff822595fa44f53a6fd9eebe0f4e3cc9e28362d9cfa8cac977a21b10a823f2
    src/bin/benchmark_gpu_public_api.rs: a5ed803c89dc3085b18ed3220a6d484facf179c9b55fe13042beb19d83090196
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 3d5bedcc3620f75cab2687fdf49210df324021d043b23aec6e169642ceff15ba
    src/bin/benchmark_gpu_public_api/multi.rs: 950cb1b3c1a7cca37665b58e3de074bd906dafb8c84b17a51c27ce797a73afa8
    src/bin/benchmark_gpu_public_api/plan.rs: 016e752c59edefc3ded1405c36325e8680736860cb65954919ea4cf0f3eed061
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: e9b588e54fd067e570b557a4b06e40f0b9b432d665f50b64ee7f2ff7f461754b
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 88038508963fbc13303fb54b3898808e4303131321cabedb9f38255941ead438
    src/bin/benchmark_gpu_public_api/state.rs: e1cc2781ce7392154f32cc19372c9fc41ae29765aedc544c7ae7bf02f873c12b
    src/bin/benchmark_gpu_public_api/structural.rs: 10850458a640d11e79262c215072482ee463bc4ca0604f40d82ae56f345f6564
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: d1ff68f86496d991bcbc730138fe6e3076001b3f23cc1a94c44b96f6857cb8ed
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 7995bcf409258185e07210969a738896f64860b67df251fcd1100d11128a5431
    data/results/nvidia-gpu/gpu/isolated_ops/20261121_082311/fixtures/manifest.json: 3c1a75f9c57996b228c04da1d4cbf34e4137a074e44ddca9f1c37e5824580bd8
    target/release/benchmark_gpu_public_api: c67dd59529f9c6644708d8bc0a5a5fabc82f29fff70f5532c20f6a3066720813
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:35:42Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:35:35Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:35:26Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:35:30Z'
  compile:
    case_wall_seconds: 14.221
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
| add_f64_n1048576 | tenferro-cuda-direct | add_f64_n1048576 untimed 137.1 ms is 1243% of the 11.0 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-direct | sub_f64_n1048576 untimed 76.2 ms is 693% of the 11.0 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-direct | mul_f64_n1048576 untimed 84.6 ms is 767% of the 11.0 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-direct | div_f64_n1048576 untimed 84.8 ms is 776% of the 10.9 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-direct | rem_f64_n1048576 untimed 94.9 ms is 471% of the 20.2 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-direct | pow_f64_n1048576 untimed 76.1 ms is 363% of the 21.0 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-direct | maximum_f64_n1048576 untimed 92.4 ms is 653% of the 14.2 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-direct | minimum_f64_n1048576 untimed 85.9 ms is 605% of the 14.2 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-direct | neg_f64_n1048576 untimed 78.2 ms is 864% of the 9.1 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-direct | abs_f64_n1048576 untimed 66.7 ms is 733% of the 9.1 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-direct | sign_f64_n1048576 untimed 66.3 ms is 721% of the 9.2 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-direct | conj_c64_n1048576 untimed 103.1 ms is 1233% of the 8.4 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-direct | exp_f64_n1048576 untimed 80.5 ms is 857% of the 9.4 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-direct | expm1_f64_n1048576 untimed 71.5 ms is 727% of the 9.8 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-direct | log_f64_n1048576 untimed 67.6 ms is 628% of the 10.7 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-direct | log1p_f64_n1048576 untimed 78.3 ms is 709% of the 11.0 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-direct | erf_f64_n1048576 untimed 100.4 ms is 830% of the 12.1 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-direct | sin_f64_n1048576 untimed 77.8 ms is 751% of the 10.4 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-direct | cos_f64_n1048576 untimed 86.4 ms is 766% of the 11.3 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-direct | tanh_f64_n1048576 untimed 86.3 ms is 859% of the 10.1 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-direct | sqrt_f64_n1048576 untimed 81.1 ms is 844% of the 9.6 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | rsqrt_f64_n1048576 untimed 91.1 ms is 960% of the 9.5 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-direct | compare_f64_n1048576 untimed 57.8 ms is 161% of the 35.9 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-direct | where_select_f64_n1048576 untimed 92.2 ms is 781% of the 11.8 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-direct | clamp_f64_n1048576 untimed 103.9 ms is 318% of the 32.7 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | reduce_sum_f64_n1048576 untimed 13.0 ms is 26% of the 49.5 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | reduce_prod_f64_n1048576 untimed 12.9 ms is 25% of the 51.4 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | reduce_max_f64_n1048576 untimed 14.5 ms is 28% of the 51.1 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | reduce_min_f64_n1048576 untimed 11.8 ms is 22% of the 52.7 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_f64_n1048576 untimed 15.5 ms is 32% of the 49.1 ms timed region (budget 20%) |
| add_f64_n1048576 | tenferro-cuda-eager | add_f64_n1048576 untimed 147.5 ms is 829% of the 17.8 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-eager | sub_f64_n1048576 untimed 92.0 ms is 258% of the 35.6 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-eager | mul_f64_n1048576 untimed 73.9 ms is 207% of the 35.7 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-eager | div_f64_n1048576 untimed 74.8 ms is 212% of the 35.2 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-eager | rem_f64_n1048576 untimed 78.4 ms is 171% of the 45.7 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-eager | pow_f64_n1048576 untimed 84.8 ms is 184% of the 46.1 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-eager | maximum_f64_n1048576 untimed 76.3 ms is 195% of the 39.0 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-eager | minimum_f64_n1048576 untimed 77.4 ms is 197% of the 39.2 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-eager | neg_f64_n1048576 untimed 67.2 ms is 325% of the 20.7 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-eager | abs_f64_n1048576 untimed 78.7 ms is 390% of the 20.2 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-eager | sign_f64_n1048576 untimed 85.1 ms is 417% of the 20.4 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-eager | conj_c64_n1048576 untimed 107.5 ms is 695% of the 15.5 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-eager | exp_f64_n1048576 untimed 80.8 ms is 393% of the 20.5 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-eager | expm1_f64_n1048576 untimed 77.4 ms is 364% of the 21.2 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-eager | log_f64_n1048576 untimed 75.0 ms is 335% of the 22.4 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-eager | log1p_f64_n1048576 untimed 82.5 ms is 372% of the 22.2 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-eager | erf_f64_n1048576 untimed 96.3 ms is 412% of the 23.4 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-eager | sin_f64_n1048576 untimed 73.2 ms is 338% of the 21.7 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-eager | cos_f64_n1048576 untimed 73.7 ms is 335% of the 22.0 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-eager | tanh_f64_n1048576 untimed 80.0 ms is 392% of the 20.4 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-eager | sqrt_f64_n1048576 untimed 72.8 ms is 358% of the 20.3 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | rsqrt_f64_n1048576 untimed 84.1 ms is 419% of the 20.1 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-eager | compare_f64_n1048576 untimed 51.5 ms is 155% of the 33.3 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-eager | where_select_f64_n1048576 untimed 87.5 ms is 186% of the 47.1 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-eager | clamp_f64_n1048576 untimed 68.9 ms is 230% of the 30.0 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | reduce_sum_f64_n1048576 untimed 11.5 ms is 25% of the 46.5 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | reduce_prod_f64_n1048576 untimed 10.5 ms is 22% of the 47.5 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | reduce_min_f64_n1048576 untimed 10.0 ms is 21% of the 48.0 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | reduce_sum_squares_f64_n1048576 untimed 10.0 ms is 21% of the 47.0 ms timed region (budget 20%) |
| add_f64_n1048576 | pytorch-cuda | add_f64_n1048576 untimed 207.9 ms is 13423% of the 1.5 ms timed region (budget 20%) |
| sub_f64_n1048576 | pytorch-cuda | sub_f64_n1048576 untimed 74.4 ms is 1285% of the 5.8 ms timed region (budget 20%) |
| mul_f64_n1048576 | pytorch-cuda | mul_f64_n1048576 untimed 65.0 ms is 1095% of the 5.9 ms timed region (budget 20%) |
| div_f64_n1048576 | pytorch-cuda | div_f64_n1048576 untimed 69.0 ms is 1073% of the 6.4 ms timed region (budget 20%) |
| rem_f64_n1048576 | pytorch-cuda | rem_f64_n1048576 untimed 80.8 ms is 1373% of the 5.9 ms timed region (budget 20%) |
| pow_f64_n1048576 | pytorch-cuda | pow_f64_n1048576 untimed 90.8 ms is 645% of the 14.1 ms timed region (budget 20%) |
| maximum_f64_n1048576 | pytorch-cuda | maximum_f64_n1048576 untimed 73.6 ms is 1246% of the 5.9 ms timed region (budget 20%) |
| minimum_f64_n1048576 | pytorch-cuda | minimum_f64_n1048576 untimed 53.9 ms is 909% of the 5.9 ms timed region (budget 20%) |
| neg_f64_n1048576 | pytorch-cuda | neg_f64_n1048576 untimed 63.2 ms is 1653% of the 3.8 ms timed region (budget 20%) |
| abs_f64_n1048576 | pytorch-cuda | abs_f64_n1048576 untimed 75.3 ms is 1972% of the 3.8 ms timed region (budget 20%) |
| sign_f64_n1048576 | pytorch-cuda | sign_f64_n1048576 untimed 70.3 ms is 1843% of the 3.8 ms timed region (budget 20%) |
| conj_c64_n1048576 | pytorch-cuda | conj_c64_n1048576 untimed 94.6 ms is 1554% of the 6.1 ms timed region (budget 20%) |
| exp_f64_n1048576 | pytorch-cuda | exp_f64_n1048576 untimed 92.4 ms is 2313% of the 4.0 ms timed region (budget 20%) |
| expm1_f64_n1048576 | pytorch-cuda | expm1_f64_n1048576 untimed 64.5 ms is 1352% of the 4.8 ms timed region (budget 20%) |
| log_f64_n1048576 | pytorch-cuda | log_f64_n1048576 untimed 66.6 ms is 1096% of the 6.1 ms timed region (budget 20%) |
| log1p_f64_n1048576 | pytorch-cuda | log1p_f64_n1048576 untimed 60.5 ms is 963% of the 6.3 ms timed region (budget 20%) |
| erf_f64_n1048576 | pytorch-cuda | erf_f64_n1048576 untimed 71.3 ms is 393% of the 18.1 ms timed region (budget 20%) |
| sin_f64_n1048576 | pytorch-cuda | sin_f64_n1048576 untimed 67.2 ms is 1242% of the 5.4 ms timed region (budget 20%) |
| cos_f64_n1048576 | pytorch-cuda | cos_f64_n1048576 untimed 70.9 ms is 1283% of the 5.5 ms timed region (budget 20%) |
| tanh_f64_n1048576 | pytorch-cuda | tanh_f64_n1048576 untimed 74.6 ms is 1890% of the 3.9 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | pytorch-cuda | sqrt_f64_n1048576 untimed 61.4 ms is 1594% of the 3.8 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | pytorch-cuda | rsqrt_f64_n1048576 untimed 56.7 ms is 1479% of the 3.8 ms timed region (budget 20%) |
| compare_f64_n1048576 | pytorch-cuda | compare_f64_n1048576 untimed 68.1 ms is 3764% of the 1.8 ms timed region (budget 20%) |
| where_select_f64_n1048576 | pytorch-cuda | where_select_f64_n1048576 untimed 84.0 ms is 1229% of the 6.8 ms timed region (budget 20%) |
| clamp_f64_n1048576 | pytorch-cuda | clamp_f64_n1048576 untimed 54.7 ms is 1101% of the 5.0 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | pytorch-cuda | reduce_sum_f64_n1048576 untimed 29.4 ms is 53% of the 55.6 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | pytorch-cuda | reduce_prod_f64_n1048576 untimed 22.8 ms is 41% of the 56.0 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | pytorch-cuda | reduce_max_f64_n1048576 untimed 23.4 ms is 41% of the 56.6 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | pytorch-cuda | reduce_min_f64_n1048576 untimed 24.1 ms is 43% of the 56.4 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | pytorch-cuda | reduce_mean_f64_n1048576 untimed 25.1 ms is 46% of the 54.9 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | pytorch-cuda | scale_real_f64_n1048576 untimed 65.4 ms is 1422% of the 4.6 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | pytorch-cuda | scale_complex_c64_n1048576 untimed 110.8 ms is 1798% of the 6.2 ms timed region (budget 20%) |
| add_f64_n1048576 | jax-cuda | add_f64_n1048576 untimed 316.2 ms is 1053% of the 30.0 ms timed region (budget 20%) |
| sub_f64_n1048576 | jax-cuda | sub_f64_n1048576 untimed 119.3 ms is 389% of the 30.7 ms timed region (budget 20%) |
| mul_f64_n1048576 | jax-cuda | mul_f64_n1048576 untimed 97.8 ms is 304% of the 32.1 ms timed region (budget 20%) |
| div_f64_n1048576 | jax-cuda | div_f64_n1048576 untimed 126.6 ms is 397% of the 31.9 ms timed region (budget 20%) |
| rem_f64_n1048576 | jax-cuda | rem_f64_n1048576 untimed 152.5 ms is 540% of the 28.2 ms timed region (budget 20%) |
| pow_f64_n1048576 | jax-cuda | pow_f64_n1048576 untimed 118.5 ms is 370% of the 32.1 ms timed region (budget 20%) |
| maximum_f64_n1048576 | jax-cuda | maximum_f64_n1048576 untimed 113.1 ms is 341% of the 33.1 ms timed region (budget 20%) |
| minimum_f64_n1048576 | jax-cuda | minimum_f64_n1048576 untimed 93.7 ms is 280% of the 33.5 ms timed region (budget 20%) |
| neg_f64_n1048576 | jax-cuda | neg_f64_n1048576 untimed 104.7 ms is 324% of the 32.3 ms timed region (budget 20%) |
| abs_f64_n1048576 | jax-cuda | abs_f64_n1048576 untimed 114.4 ms is 362% of the 31.6 ms timed region (budget 20%) |
| sign_f64_n1048576 | jax-cuda | sign_f64_n1048576 untimed 126.9 ms is 361% of the 35.1 ms timed region (budget 20%) |
| conj_c64_n1048576 | jax-cuda | conj_c64_n1048576 untimed 169.4 ms is 1070% of the 15.8 ms timed region (budget 20%) |
| exp_f64_n1048576 | jax-cuda | exp_f64_n1048576 untimed 129.3 ms is 406% of the 31.9 ms timed region (budget 20%) |
| expm1_f64_n1048576 | jax-cuda | expm1_f64_n1048576 untimed 132.2 ms is 415% of the 31.9 ms timed region (budget 20%) |
| log_f64_n1048576 | jax-cuda | log_f64_n1048576 untimed 137.3 ms is 427% of the 32.2 ms timed region (budget 20%) |
| log1p_f64_n1048576 | jax-cuda | log1p_f64_n1048576 untimed 150.3 ms is 474% of the 31.7 ms timed region (budget 20%) |
| erf_f64_n1048576 | jax-cuda | erf_f64_n1048576 untimed 134.0 ms is 329% of the 40.7 ms timed region (budget 20%) |
| sin_f64_n1048576 | jax-cuda | sin_f64_n1048576 untimed 169.5 ms is 545% of the 31.1 ms timed region (budget 20%) |
| cos_f64_n1048576 | jax-cuda | cos_f64_n1048576 untimed 152.1 ms is 593% of the 25.6 ms timed region (budget 20%) |
| tanh_f64_n1048576 | jax-cuda | tanh_f64_n1048576 untimed 107.8 ms is 416% of the 25.9 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | jax-cuda | sqrt_f64_n1048576 untimed 112.7 ms is 412% of the 27.4 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | jax-cuda | rsqrt_f64_n1048576 untimed 109.0 ms is 289% of the 37.7 ms timed region (budget 20%) |
| compare_f64_n1048576 | jax-cuda | compare_f64_n1048576 untimed 97.2 ms is 158% of the 61.3 ms timed region (budget 20%) |
| where_select_f64_n1048576 | jax-cuda | where_select_f64_n1048576 untimed 108.0 ms is 297% of the 36.4 ms timed region (budget 20%) |
| clamp_f64_n1048576 | jax-cuda | clamp_f64_n1048576 untimed 108.9 ms is 361% of the 30.2 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | jax-cuda | reduce_sum_f64_n1048576 untimed 213.8 ms is 713% of the 30.0 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | jax-cuda | reduce_prod_f64_n1048576 untimed 218.3 ms is 704% of the 31.0 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | jax-cuda | reduce_max_f64_n1048576 untimed 305.1 ms is 997% of the 30.6 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | jax-cuda | reduce_min_f64_n1048576 untimed 89.6 ms is 308% of the 29.1 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | jax-cuda | reduce_mean_f64_n1048576 untimed 104.8 ms is 362% of the 28.9 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | jax-cuda | scale_real_f64_n1048576 untimed 132.4 ms is 371% of the 35.7 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | jax-cuda | scale_complex_c64_n1048576 untimed 162.2 ms is 877% of the 18.5 ms timed region (budget 20%) |