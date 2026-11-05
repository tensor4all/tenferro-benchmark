# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **124**. Unsupported cells: **8**, including **6** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1408.2 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261105_0820/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_f64_n1048576 | 31.368 (0.281) | 88.809 (0.305) | 12.610 (0.147) | 60.991 (0.985) |
| sub_f64_n1048576 | 31.655 (0.201) | 89.308 (0.682) | 12.640 (0.077) | 62.727 (2.427) |
| mul_f64_n1048576 | 31.382 (0.303) | 89.578 (0.737) | 12.814 (0.087) | 62.815 (0.844) |
| div_f64_n1048576 | 31.625 (0.214) | 89.590 (0.227) | 13.366 (0.092) | 47.278 (0.428) |
| rem_f64_n1048576 | 60.531 (0.980) | 113.071 (0.380) | 12.306 (0.182) | 48.346 (0.882) |
| pow_f64_n1048576 | 63.244 (0.568) | 115.577 (0.590) | 28.955 (0.048) | 52.769 (0.759) |
| maximum_f64_n1048576 | 51.733 (0.573) | 100.716 (0.629) | 12.045 (0.250) | 50.047 (0.441) |
| minimum_f64_n1048576 | 51.355 (0.914) | 101.345 (0.793) | 12.083 (0.177) | 51.059 (0.814) |
| neg_f64_n1048576 | 24.303 (0.219) | 53.196 (0.324) | 7.855 (0.065) | 47.972 (0.855) |
| abs_f64_n1048576 | 24.348 (0.130) | 52.976 (0.247) | 7.826 (0.099) | 42.559 (0.559) |
| sign_f64_n1048576 | 24.441 (0.138) | 53.306 (0.457) | 7.734 (0.039) | 42.744 (0.299) |
| conj_c64_n1048576 | 40.315 (0.240) | 84.614 (0.419) | 25.380 (0.438) | 44.920 (1.613) |
| exp_f64_n1048576 | 25.270 (0.386) | 54.233 (0.189) | 8.299 (0.046) | 40.486 (0.465) |
| expm1_f64_n1048576 | 26.016 (0.137) | 54.858 (0.315) | 9.710 (0.075) | 40.222 (0.294) |
| log_f64_n1048576 | 28.398 (0.264) | 57.841 (0.415) | 12.536 (0.041) | 40.209 (0.187) |
| log1p_f64_n1048576 | 28.344 (0.120) | 57.574 (0.292) | 12.769 (0.056) | 40.398 (0.162) |
| erf_f64_n1048576 | 30.451 (0.169) | 59.368 (0.157) | 14.305 (0.071) | 64.633 (1.201) |
| sin_f64_n1048576 | 26.722 (0.114) | 56.463 (0.244) | 10.002 (0.060) | 41.772 (0.519) |
| cos_f64_n1048576 | 26.848 (0.108) | 56.484 (0.241) | 10.075 (0.058) | 41.397 (0.896) |
| tanh_f64_n1048576 | 25.162 (0.071) | 54.307 (0.803) | 8.094 (0.041) | 40.354 (0.493) |
| sqrt_f64_n1048576 | 24.812 (0.157) | 53.659 (0.258) | 7.875 (0.033) | 40.451 (0.406) |
| rsqrt_f64_n1048576 | 24.572 (0.092) | 53.624 (0.297) | 7.761 (0.068) | 62.620 (0.781) |
| compare_f64_n1048576 | 21.830 (0.066) | 82.840 (0.375) | 6.814 (0.016) | 42.140 (0.376) |
| where_select_f64_n1048576 | 32.382 (0.106) | 117.483 (0.851) | 13.504 (0.123) | 50.917 (1.383) |
| clamp_f64_n1048576 | 86.003 (0.314) | 155.736 (0.302) | 10.289 (0.028) | 44.981 (1.458) |
| reduce_sum_f64_n1048576 | 62.623 (0.251) | 93.546 (0.681) | 7.058 (0.010) | 47.669 (0.400) |
| reduce_prod_f64_n1048576 | 62.572 (0.501) | 93.571 (0.210) | 7.096 (0.004) | 90.278 (35.830) |
| reduce_max_f64_n1048576 | 62.665 (0.306) | 93.115 (0.677) | 7.192 (0.007) | 54.824 (8.680) |
| reduce_min_f64_n1048576 | 62.507 (0.208) | 93.785 (0.550) | 7.231 (0.007) | 86.801 (39.019) |
| reduce_sum_squares_f64_n1048576 | 62.408 (0.225) | 93.240 (0.768) | unsupported | unsupported |
| reduce_mean_f64_n1048576 | unsupported | unsupported | 6.928 (0.027) | 53.202 (11.660) |
| scale_real_f64_n1048576 | unsupported | unsupported | 9.504 (0.076) | 50.849 (0.829) |
| scale_complex_c64_n1048576 | unsupported | unsupported | 25.719 (0.167) | 49.610 (0.782) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 1.362 | 16777216 | 0.280 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.250 | 16777216 | 1.522 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.779 | 16777216 | 0.489 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.695 | 16777216 | 0.225 | 0 |
| add_f64_n1048576 | jax-cuda | 32 | 15 | 1.952 | 25165824 | 0.293 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.404 | 25165824 | 1.417 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.004 | 25165824 | 0.570 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.842 | 25165824 | 0.201 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 1.439 | 16777232 | 0.265 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.329 | 16777232 | 1.158 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.752 | 16777232 | 0.139 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.492 | 16777232 | 0.077 | 0 |
| compare_f64_n1048576 | jax-cuda | 64 | 15 | 2.697 | 17825792 | 0.300 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.436 | 17825792 | 1.858 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.794 | 17825792 | 0.580 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.651 | 17825792 | 0.153 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 0.719 | 33554432 | 0.530 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.406 | 33554432 | 0.939 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.645 | 33554432 | 0.591 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.354 | 33554432 | 0.282 | 0 |
| cos_f64_n1048576 | jax-cuda | 32 | 15 | 1.325 | 16777216 | 0.288 | 5.905e-08 |
| cos_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.322 | 16777216 | 1.182 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.859 | 16777216 | 0.444 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.807 | 16777216 | 0.211 | 5.905e-08 |
| div_f64_n1048576 | jax-cuda | 32 | 15 | 1.513 | 25165824 | 0.378 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.428 | 25165824 | 1.337 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.012 | 25165824 | 0.565 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.867 | 25165824 | 0.199 | 0 |
| erf_f64_n1048576 | jax-cuda | 32 | 15 | 2.068 | 16777216 | 0.184 | 8.536e-08 |
| erf_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.458 | 16777216 | 0.833 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.974 | 16777216 | 0.391 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.900 | 16777216 | 0.201 | 8.536e-08 |
| exp_f64_n1048576 | jax-cuda | 32 | 15 | 1.296 | 16777216 | 0.294 | 1.095e-07 |
| exp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.266 | 16777216 | 1.436 | 1.095e-07 |
| exp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.809 | 16777216 | 0.471 | 1.095e-07 |
| exp_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.735 | 16777216 | 0.220 | 1.095e-07 |
| expm1_f64_n1048576 | jax-cuda | 32 | 15 | 1.287 | 16777216 | 0.296 | 7.39e-08 |
| expm1_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.311 | 16777216 | 1.227 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.833 | 16777216 | 0.458 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.755 | 16777216 | 0.217 | 7.39e-08 |
| log1p_f64_n1048576 | jax-cuda | 32 | 15 | 1.293 | 16777216 | 0.295 | 1.11e-07 |
| log1p_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.409 | 16777216 | 0.933 | 1.11e-07 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.907 | 16777216 | 0.420 | 1.11e-07 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.842 | 16777216 | 0.207 | 1.11e-07 |
| log_f64_n1048576 | jax-cuda | 32 | 15 | 1.287 | 16777216 | 0.296 | 4.308e-08 |
| log_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.401 | 16777216 | 0.950 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.909 | 16777216 | 0.420 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.851 | 16777216 | 0.206 | 4.308e-08 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 1.602 | 25165824 | 0.357 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.385 | 25165824 | 1.484 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.655 | 25165824 | 0.345 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.223 | 25165824 | 0.177 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 1.634 | 25165824 | 0.350 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.387 | 25165824 | 1.479 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.643 | 25165824 | 0.348 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.243 | 25165824 | 0.176 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 2.010 | 25165824 | 0.285 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.410 | 25165824 | 1.395 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.004 | 25165824 | 0.569 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.866 | 25165824 | 0.200 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 1.535 | 16777216 | 0.248 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.251 | 16777216 | 1.517 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.778 | 16777216 | 0.490 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.702 | 16777216 | 0.224 | 0 |
| pow_f64_n1048576 | jax-cuda | 32 | 15 | 1.689 | 25165824 | 0.339 | 1.48e-07 |
| pow_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.927 | 25165824 | 0.617 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.024 | 25165824 | 0.283 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.698 | 25165824 | 0.155 | 1.48e-07 |
| reduce_max_f64_n1048576 | jax-cuda | 32 | 15 | 1.754 | 8396800 | 0.109 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.682 | 8396800 | 0.829 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.005 | 8396800 | 0.095 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.980 | 8396800 | 0.064 | 0 |
| reduce_mean_f64_n1048576 | jax-cuda | 32 | 15 | 1.702 | 8396800 | 0.112 | 6.5e-09 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.547 | 8396800 | 0.861 | 3.9e-09 |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_min_f64_n1048576 | jax-cuda | 32 | 15 | 2.778 | 8396800 | 0.069 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.702 | 8396800 | 0.825 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 4.000 | 8396800 | 0.095 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.001 | 8396800 | 0.064 | 0 |
| reduce_prod_f64_n1048576 | jax-cuda | 32 | 15 | 2.889 | 8396800 | 0.066 | 4.436e-06 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.633 | 8396800 | 0.840 | 4.216e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 4.005 | 8396800 | 0.095 | 3.55e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.994 | 8396800 | 0.064 | 3.55e-06 |
| reduce_sum_f64_n1048576 | jax-cuda | 64 | 15 | 3.051 | 8396800 | 0.125 | 4.449e-06 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.614 | 8396800 | 0.845 | 2.961e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.004 | 8396800 | 0.095 | 3.391e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.993 | 8396800 | 0.064 | 3.391e-06 |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.994 | 8396800 | 0.096 | 3.294e-07 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.984 | 8396800 | 0.064 | 3.294e-07 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 1.547 | 25165824 | 0.370 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.394 | 25165824 | 1.452 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.937 | 25165824 | 0.295 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.618 | 25165824 | 0.158 | 0 |
| rsqrt_f64_n1048576 | jax-cuda | 32 | 15 | 2.004 | 16777216 | 0.190 | 6.504e-08 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.248 | 16777216 | 1.535 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.786 | 16777216 | 0.485 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.716 | 16777216 | 0.222 | 6.504e-08 |
| scale_complex_c64_n1048576 | jax-cuda | 16 | 15 | 0.794 | 33554448 | 0.480 | 4.129e-08 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.411 | 33554448 | 0.927 | 0 |
| scale_complex_c64_n1048576 | tenferro-cuda-direct | — | — | — | 33554448 | unavailable | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554448 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_f64_n1048576 | jax-cuda | 32 | 15 | 1.627 | 16777224 | 0.234 | 0 |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.304 | 16777224 | 1.254 | 0 |
| scale_real_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777224 | unavailable | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_real_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777224 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 1.368 | 16777216 | 0.279 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.248 | 16777216 | 1.540 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.782 | 16777216 | 0.487 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.706 | 16777216 | 0.224 | 0 |
| sin_f64_n1048576 | jax-cuda | 32 | 15 | 1.337 | 16777216 | 0.285 | 4.153e-08 |
| sin_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.320 | 16777216 | 1.191 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.855 | 16777216 | 0.446 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.807 | 16777216 | 0.211 | 4.153e-08 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 1.294 | 16777216 | 0.295 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.252 | 16777216 | 1.513 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.794 | 16777216 | 0.480 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.717 | 16777216 | 0.222 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 2.007 | 25165824 | 0.285 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.404 | 25165824 | 1.414 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.013 | 25165824 | 0.565 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.858 | 25165824 | 0.200 | 0 |
| tanh_f64_n1048576 | jax-cuda | 32 | 15 | 1.291 | 16777216 | 0.295 | 4.437e-08 |
| tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.259 | 16777216 | 1.472 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.805 | 16777216 | 0.473 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.738 | 16777216 | 0.219 | 4.437e-08 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 1.629 | 26214400 | 0.366 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.432 | 26214400 | 1.379 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.036 | 26214400 | 0.575 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.759 | 26214400 | 0.158 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: /home/shinaoka/work/wallclock/data/results/nvidia-gpu/gpu/routine_selection/20261105_0820/gpu_isolated_ops.yaml
timestamp: '2026-11-05T08:02:00+00:00'
tenferro_rs:
  path: /home/shinaoka/work/wallclock/extern/tenferro-rs
  commit: a000e5c4929d5bebb91e783c329b27106baed12e
  dirty: false
  features:
  - cuda
  resolved_path: /home/shinaoka/work/wallclock/extern/tenferro-rs
harness:
  path: /home/shinaoka/work/wallclock
  commit: 18ae0c54c3f6cc4493988efe87a65148334a5c77
  dirty: false
resolved_deps:
  strided-perm:
  - 0.8.0
  - 0.10.0
  strided-view:
  - 0.7.0
  threadteam:
  - 0.1.0
  - 0.3.0
host:
  cpu: AMD EPYC 7713P 64-Core Processor
  logical_cpus: 64
  power:
    cpufreq_boost: '1'
    governor: schedutil
    scaling_max_freq_khz: '2000000'
    smt:
      active: '0'
      control: notsupported
  numa:
    nodes: 1
    cpulists:
      '0': 0-63
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
  l3_domains:
  - 0-7
  - 8-15
  - 16-23
  - 24-31
  - 32-39
  - 40-47
  - 48-55
  - 56-63
build:
  toolchain:
    rustc: 'rustc 1.99.1 (58cf8f9c3 2026-10-13)

      binary: rustc

      commit-hash: 58cf8f9c35b355199727627d07ae34a4e2846d67

      commit-date: 2026-10-13

      host: x86_64-unknown-linux-gnu

      release: 1.99.1

      LLVM version: 23.1.1

      '
    rustc_release: 1.99.1
    rustflags: null
    target_cpu: null
environment:
  container: none
  hostname: primerose
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-63
    l3_domains:
    - 0-7
    - 8-15
    - 16-23
    - 24-31
    - 32-39
    - 40-47
    - 48-55
    - 56-63
    cpu_boost: enabled
    scaling_governor: schedutil
    scaling_driver: acpi-cpufreq
  env:
    OMP_NUM_THREADS: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_NUM_THREADS: '1'
    MATMUL_NUM_THREADS: '1'
    CUDA_HOME: /usr/local/cuda-12.6
    CUDA_PATH: /usr/local/cuda-12.6
    TENFERRO_CUTENSOR_PATH: /usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
    UV_PROJECT_ENVIRONMENT: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv
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
    version: 2.12.0+cu130
    file: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/torch/__init__.py
    provider: mkl
    blas_info: mkl
    lapack_info: mkl
    config: "PyTorch built with:\n  - GCC 13.3\n  - C++ Version: 202002\n  - Intel(R)\
      \ oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R)\
      \ 64 architecture applications\n  - Intel(R) MKL-DNN v3.11.2 (Git Hash 03c022d3ffdcee958cfacbe720048e725fdf644c)\n\
      \  - OpenMP 201511 (a.k.a. OpenMP 4.5)\n  - LAPACK is enabled (usually provided\
      \ by MKL)\n  - NNPACK is enabled\n  - CPU capability usage: AVX2\n  - CUDA Runtime\
      \ 13.0\n  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120\n\
      \  - CuDNN 92.7  (built against CUDA 12.9)\n    - Built with CuDNN 92.0\n  -\
      \ Magma 2.6.1\n  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=7661cd9c6b841b62b7f411aa52ec51f05457263b,\
      \ CUDA_FLAGS= -DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS -Xfatbin -compress-all\
      \ -DONNX_NAMESPACE=onnx_torch -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80\
      \ -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode\
      \ arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120 -Xcudafe\
      \ --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl\
      \ --expt-relaxed-constexpr --expt-extended-lambda -Xfatbin -compress-all --threads\
      \ 2 -compress-mode=size -Wno-deprecated-gpu-targets --expt-extended-lambda -DCUB_WRAPPED_NAMESPACE=at_cuda_detail\
      \ -DDISABLE_CUSPARSE_DEPRECATED -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__\
      \ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__\
      \ -DC10_NODEPRECATED, CUDA_VERSION=13.0, CUDNN_VERSION=9.20.0, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++,\
      \ CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO\
      \ -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_FBGEMM -DUSE_MSLK -DUSE_PYTORCH_QNNPACK\
      \ -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED\
      \ -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct\
      \ -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas\
      \ -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow\
      \ -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized\
      \ -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference\
      \ -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1,\
      \ PERF_WITH_AVX2=1, TORCH_VERSION=2.12.0, USE_CUDA=ON, USE_CUDNN=ON, USE_CUSPARSELT=1,\
      \ USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF,\
      \ USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF,\
      \ USE_XCCL=OFF, USE_XPU=OFF, \n"
    library: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/torch/lib/libtorch_cpu.so
    linked_libraries:
    - /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/torch/lib/libgomp.so.1
  jax:
    available: true
    version: 0.10.1
    jaxlib_version: 0.10.1
    backend: gpu
    file: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/jaxlib/__init__.py
    provider: xla_gpu
    dot_backend: xla_gpu
    lapack_provider: none_detected
    library: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/jaxlib/_ifrt_proxy.so
    linked_libraries: []
julia:
  available: false
  reason: julia not found on PATH
cuda:
  device_ordinal: 0
  device_name: NVIDIA A100 80GB PCIe
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  device_memory_bytes: 85899345920
  driver_version: 580.178.04
  runtime_version: '12.6'
  cuda_version: '13.0'
  toolkit_version: '12.6'
  toolkit_root: /usr/local/cuda-12.6
  cutensor_version: 2.8.0
  cusolver_version: 11.7.1
  cudnn_version: '92700'
  artifact_cache:
    path: null
    regime: 'off'
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
  - 0.095663
  - 0.090694
  - 0.097913
  - 0.095314
  - 0.092533
  - 0.096813
  - 0.095974
  - 0.093363
  - 0.095714
  - 0.096513
  - 0.093944
  - 0.094553
  - 0.095644
  - 0.094563
  - 0.094854
  median_ms: 0.095314
  bytes_per_second: 1408163837421.575
  torch_version: 2.12.0+cu130
  cuda_version: '13.0'
  timestamp: '2026-11-05T07:59:44.994500+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  cuda_visible_devices: GPU-530977e1-4968-9283-4129-9fbec3e66542
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
    src/bin/benchmark_gpu_public_api.rs: 4e8f3c519e9210dbc8725dac34da23e543bf462d10c4a202c61e80b59ed108dd
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 949eadd6272c17ba4e380a228547a03608d328a5ef852eb95294731c288f24d9
    src/bin/benchmark_gpu_public_api/plan.rs: 70fce8a6e9ed7518779a51a9d8692e010fa1e470e95eee0ce5276f595737fde8
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: a8bbf8985bbc58c435e3c4dda7135161a141a90a10fa50d4f95128397ce949cb
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /home/shinaoka/work/wallclock/scripts/benchmark_gpu_public_api.py: 1919e33cc5f2d0208901837f80cb7c0600426462f998be6dd933062d9f6bb345
    /home/shinaoka/work/wallclock/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 31587e7a83440d99417be00110c4dd12c5cc1143e620bf24b1c03fe5f76bfb5e
    data/results/nvidia-gpu/gpu/isolated_ops/20261105_0820/fixtures/manifest.json: 3c1a75f9c57996b228c04da1d4cbf34e4137a074e44ddca9f1c37e5824580bd8
    target/release/benchmark_gpu_public_api: efd5ab70e2c4de27c458d808d308b8bf3415012551220c291451a2591d08880f
  reference_numpy_version: 2.4.2
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| add_f64_n1048576 | tenferro-cuda-direct | add_f64_n1048576 untimed 247.0 ms is 1640% of the 15.1 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-direct | sub_f64_n1048576 untimed 78.1 ms is 514% of the 15.2 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-direct | mul_f64_n1048576 untimed 93.8 ms is 618% of the 15.2 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-direct | div_f64_n1048576 untimed 93.6 ms is 610% of the 15.3 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-direct | rem_f64_n1048576 untimed 1097.3 ms is 3789% of the 29.0 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-direct | pow_f64_n1048576 untimed 75.8 ms is 249% of the 30.4 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-direct | maximum_f64_n1048576 untimed 71.3 ms is 287% of the 24.8 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-direct | minimum_f64_n1048576 untimed 72.4 ms is 294% of the 24.6 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-direct | neg_f64_n1048576 untimed 64.3 ms is 547% of the 11.8 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-direct | abs_f64_n1048576 untimed 69.8 ms is 596% of the 11.7 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-direct | sign_f64_n1048576 untimed 69.1 ms is 588% of the 11.8 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-direct | conj_c64_n1048576 untimed 108.2 ms is 1116% of the 9.7 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-direct | exp_f64_n1048576 untimed 61.9 ms is 509% of the 12.2 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-direct | expm1_f64_n1048576 untimed 70.3 ms is 563% of the 12.5 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-direct | log_f64_n1048576 untimed 68.4 ms is 497% of the 13.8 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-direct | log1p_f64_n1048576 untimed 72.5 ms is 533% of the 13.6 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-direct | erf_f64_n1048576 untimed 77.8 ms is 531% of the 14.6 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-direct | sin_f64_n1048576 untimed 68.1 ms is 531% of the 12.8 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-direct | cos_f64_n1048576 untimed 67.9 ms is 527% of the 12.9 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-direct | tanh_f64_n1048576 untimed 71.9 ms is 595% of the 12.1 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-direct | sqrt_f64_n1048576 untimed 67.5 ms is 567% of the 11.9 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | rsqrt_f64_n1048576 untimed 72.8 ms is 617% of the 11.8 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-direct | compare_f64_n1048576 untimed 56.0 ms is 134% of the 41.9 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-direct | where_select_f64_n1048576 untimed 75.5 ms is 486% of the 15.5 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-direct | clamp_f64_n1048576 untimed 47200.3 ms is 114315% of the 41.3 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | reduce_sum_f64_n1048576 untimed 7142.4 ms is 23778% of the 30.0 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | reduce_prod_f64_n1048576 untimed 12.1 ms is 20% of the 60.2 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | reduce_max_f64_n1048576 untimed 8.0 ms is 27% of the 30.1 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_f64_n1048576 untimed 12.1 ms is 20% of the 59.9 ms timed region (budget 20%) |
| add_f64_n1048576 | tenferro-cuda-eager | add_f64_n1048576 untimed 46025.8 ms is 107904% of the 42.7 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-eager | sub_f64_n1048576 untimed 38.9 ms is 90% of the 43.1 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-eager | mul_f64_n1048576 untimed 38.1 ms is 88% of the 43.1 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-eager | div_f64_n1048576 untimed 37.7 ms is 88% of the 43.0 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-eager | rem_f64_n1048576 untimed 1052.3 ms is 1939% of the 54.3 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-eager | pow_f64_n1048576 untimed 43.1 ms is 78% of the 55.5 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-eager | maximum_f64_n1048576 untimed 39.8 ms is 82% of the 48.3 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-eager | minimum_f64_n1048576 untimed 39.0 ms is 80% of the 48.7 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-eager | neg_f64_n1048576 untimed 34.4 ms is 134% of the 25.7 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-eager | abs_f64_n1048576 untimed 33.5 ms is 132% of the 25.4 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-eager | sign_f64_n1048576 untimed 33.6 ms is 131% of the 25.7 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-eager | conj_c64_n1048576 untimed 57.2 ms is 281% of the 20.3 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-eager | exp_f64_n1048576 untimed 35.7 ms is 137% of the 26.1 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-eager | expm1_f64_n1048576 untimed 34.9 ms is 132% of the 26.4 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-eager | log_f64_n1048576 untimed 33.8 ms is 121% of the 27.9 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-eager | log1p_f64_n1048576 untimed 39.0 ms is 141% of the 27.7 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-eager | erf_f64_n1048576 untimed 44.4 ms is 156% of the 28.5 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-eager | sin_f64_n1048576 untimed 34.8 ms is 128% of the 27.2 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-eager | cos_f64_n1048576 untimed 34.3 ms is 126% of the 27.1 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-eager | tanh_f64_n1048576 untimed 37.1 ms is 142% of the 26.1 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-eager | sqrt_f64_n1048576 untimed 34.2 ms is 133% of the 25.8 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | rsqrt_f64_n1048576 untimed 39.9 ms is 155% of the 25.9 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-eager | compare_f64_n1048576 untimed 19.6 ms is 49% of the 40.0 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-eager | where_select_f64_n1048576 untimed 41.6 ms is 74% of the 56.4 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-eager | clamp_f64_n1048576 untimed 34.4 ms is 92% of the 37.4 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | reduce_sum_f64_n1048576 untimed 7045.1 ms is 15673% of the 45.0 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | reduce_prod_f64_n1048576 untimed 10.1 ms is 23% of the 44.9 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | reduce_max_f64_n1048576 untimed 9.3 ms is 21% of the 44.8 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | reduce_min_f64_n1048576 untimed 9.6 ms is 21% of the 44.9 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | reduce_sum_squares_f64_n1048576 untimed 9.3 ms is 21% of the 45.0 ms timed region (budget 20%) |
| add_f64_n1048576 | pytorch-cuda | add_f64_n1048576 untimed 1386.4 ms is 22833% of the 6.1 ms timed region (budget 20%) |
| sub_f64_n1048576 | pytorch-cuda | sub_f64_n1048576 untimed 1238.4 ms is 20410% of the 6.1 ms timed region (budget 20%) |
| mul_f64_n1048576 | pytorch-cuda | mul_f64_n1048576 untimed 1213.7 ms is 19701% of the 6.2 ms timed region (budget 20%) |
| div_f64_n1048576 | pytorch-cuda | div_f64_n1048576 untimed 1208.5 ms is 18839% of the 6.4 ms timed region (budget 20%) |
| rem_f64_n1048576 | pytorch-cuda | rem_f64_n1048576 untimed 1215.7 ms is 20606% of the 5.9 ms timed region (budget 20%) |
| pow_f64_n1048576 | pytorch-cuda | pow_f64_n1048576 untimed 1228.7 ms is 8847% of the 13.9 ms timed region (budget 20%) |
| maximum_f64_n1048576 | pytorch-cuda | maximum_f64_n1048576 untimed 1233.0 ms is 21313% of the 5.8 ms timed region (budget 20%) |
| minimum_f64_n1048576 | pytorch-cuda | minimum_f64_n1048576 untimed 1300.7 ms is 22504% of the 5.8 ms timed region (budget 20%) |
| neg_f64_n1048576 | pytorch-cuda | neg_f64_n1048576 untimed 1337.8 ms is 35438% of the 3.8 ms timed region (budget 20%) |
| abs_f64_n1048576 | pytorch-cuda | abs_f64_n1048576 untimed 1323.4 ms is 35114% of the 3.8 ms timed region (budget 20%) |
| sign_f64_n1048576 | pytorch-cuda | sign_f64_n1048576 untimed 1294.2 ms is 34856% of the 3.7 ms timed region (budget 20%) |
| conj_c64_n1048576 | pytorch-cuda | conj_c64_n1048576 untimed 1371.1 ms is 22487% of the 6.1 ms timed region (budget 20%) |
| exp_f64_n1048576 | pytorch-cuda | exp_f64_n1048576 untimed 1238.0 ms is 31047% of the 4.0 ms timed region (budget 20%) |
| expm1_f64_n1048576 | pytorch-cuda | expm1_f64_n1048576 untimed 1219.7 ms is 26086% of the 4.7 ms timed region (budget 20%) |
| log_f64_n1048576 | pytorch-cuda | log_f64_n1048576 untimed 1223.7 ms is 20348% of the 6.0 ms timed region (budget 20%) |
| log1p_f64_n1048576 | pytorch-cuda | log1p_f64_n1048576 untimed 1197.4 ms is 19525% of the 6.1 ms timed region (budget 20%) |
| erf_f64_n1048576 | pytorch-cuda | erf_f64_n1048576 untimed 1239.6 ms is 18072% of the 6.9 ms timed region (budget 20%) |
| sin_f64_n1048576 | pytorch-cuda | sin_f64_n1048576 untimed 1273.7 ms is 26524% of the 4.8 ms timed region (budget 20%) |
| cos_f64_n1048576 | pytorch-cuda | cos_f64_n1048576 untimed 1306.3 ms is 26995% of the 4.8 ms timed region (budget 20%) |
| tanh_f64_n1048576 | pytorch-cuda | tanh_f64_n1048576 untimed 1355.7 ms is 34892% of the 3.9 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | pytorch-cuda | sqrt_f64_n1048576 untimed 1326.8 ms is 35110% of the 3.8 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | pytorch-cuda | rsqrt_f64_n1048576 untimed 1295.3 ms is 34725% of the 3.7 ms timed region (budget 20%) |
| compare_f64_n1048576 | pytorch-cuda | compare_f64_n1048576 untimed 1224.8 ms is 18702% of the 6.5 ms timed region (budget 20%) |
| where_select_f64_n1048576 | pytorch-cuda | where_select_f64_n1048576 untimed 1307.6 ms is 20161% of the 6.5 ms timed region (budget 20%) |
| clamp_f64_n1048576 | pytorch-cuda | clamp_f64_n1048576 untimed 1271.5 ms is 25753% of the 4.9 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | pytorch-cuda | reduce_sum_f64_n1048576 untimed 33.3 ms is 61% of the 54.2 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | pytorch-cuda | reduce_prod_f64_n1048576 untimed 23.8 ms is 44% of the 54.5 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | pytorch-cuda | reduce_max_f64_n1048576 untimed 25.4 ms is 46% of the 55.4 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | pytorch-cuda | reduce_min_f64_n1048576 untimed 28.2 ms is 51% of the 55.5 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | pytorch-cuda | reduce_mean_f64_n1048576 untimed 29.5 ms is 55% of the 53.3 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | pytorch-cuda | scale_real_f64_n1048576 untimed 1241.5 ms is 27117% of the 4.6 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | pytorch-cuda | scale_complex_c64_n1048576 untimed 1312.6 ms is 21246% of the 6.2 ms timed region (budget 20%) |
| add_f64_n1048576 | jax-cuda | add_f64_n1048576 untimed 1540.2 ms is 5267% of the 29.2 ms timed region (budget 20%) |
| sub_f64_n1048576 | jax-cuda | sub_f64_n1048576 untimed 1264.0 ms is 4117% of the 30.7 ms timed region (budget 20%) |
| mul_f64_n1048576 | jax-cuda | mul_f64_n1048576 untimed 1254.9 ms is 4154% of the 30.2 ms timed region (budget 20%) |
| div_f64_n1048576 | jax-cuda | div_f64_n1048576 untimed 1182.7 ms is 5212% of the 22.7 ms timed region (budget 20%) |
| rem_f64_n1048576 | jax-cuda | rem_f64_n1048576 untimed 1244.5 ms is 5349% of the 23.3 ms timed region (budget 20%) |
| pow_f64_n1048576 | jax-cuda | pow_f64_n1048576 untimed 1223.9 ms is 4824% of the 25.4 ms timed region (budget 20%) |
| maximum_f64_n1048576 | jax-cuda | maximum_f64_n1048576 untimed 1220.2 ms is 5082% of the 24.0 ms timed region (budget 20%) |
| minimum_f64_n1048576 | jax-cuda | minimum_f64_n1048576 untimed 1220.2 ms is 4980% of the 24.5 ms timed region (budget 20%) |
| neg_f64_n1048576 | jax-cuda | neg_f64_n1048576 untimed 1251.7 ms is 5433% of the 23.0 ms timed region (budget 20%) |
| abs_f64_n1048576 | jax-cuda | abs_f64_n1048576 untimed 1281.1 ms is 6273% of the 20.4 ms timed region (budget 20%) |
| sign_f64_n1048576 | jax-cuda | sign_f64_n1048576 untimed 1309.3 ms is 6369% of the 20.6 ms timed region (budget 20%) |
| conj_c64_n1048576 | jax-cuda | conj_c64_n1048576 untimed 1367.2 ms is 12626% of the 10.8 ms timed region (budget 20%) |
| exp_f64_n1048576 | jax-cuda | exp_f64_n1048576 untimed 1235.5 ms is 6293% of the 19.6 ms timed region (budget 20%) |
| expm1_f64_n1048576 | jax-cuda | expm1_f64_n1048576 untimed 1210.8 ms is 6261% of the 19.3 ms timed region (budget 20%) |
| log_f64_n1048576 | jax-cuda | log_f64_n1048576 untimed 1218.5 ms is 6297% of the 19.3 ms timed region (budget 20%) |
| log1p_f64_n1048576 | jax-cuda | log1p_f64_n1048576 untimed 1263.3 ms is 6522% of the 19.4 ms timed region (budget 20%) |
| erf_f64_n1048576 | jax-cuda | erf_f64_n1048576 untimed 1283.7 ms is 4168% of the 30.8 ms timed region (budget 20%) |
| sin_f64_n1048576 | jax-cuda | sin_f64_n1048576 untimed 1307.3 ms is 6518% of the 20.1 ms timed region (budget 20%) |
| cos_f64_n1048576 | jax-cuda | cos_f64_n1048576 untimed 1322.7 ms is 6661% of the 19.9 ms timed region (budget 20%) |
| tanh_f64_n1048576 | jax-cuda | tanh_f64_n1048576 untimed 1225.5 ms is 6337% of the 19.3 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | jax-cuda | sqrt_f64_n1048576 untimed 1254.3 ms is 6458% of the 19.4 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | jax-cuda | rsqrt_f64_n1048576 untimed 1198.7 ms is 3968% of the 30.2 ms timed region (budget 20%) |
| compare_f64_n1048576 | jax-cuda | compare_f64_n1048576 untimed 1199.9 ms is 2967% of the 40.4 ms timed region (budget 20%) |
| where_select_f64_n1048576 | jax-cuda | where_select_f64_n1048576 untimed 1252.3 ms is 5150% of the 24.3 ms timed region (budget 20%) |
| clamp_f64_n1048576 | jax-cuda | clamp_f64_n1048576 untimed 1264.0 ms is 5838% of the 21.7 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | jax-cuda | reduce_sum_f64_n1048576 untimed 257.3 ms is 563% of the 45.7 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | jax-cuda | reduce_prod_f64_n1048576 untimed 255.4 ms is 697% of the 36.7 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | jax-cuda | reduce_max_f64_n1048576 untimed 355.4 ms is 1330% of the 26.7 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | jax-cuda | reduce_min_f64_n1048576 untimed 125.9 ms is 367% of the 34.3 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | jax-cuda | reduce_mean_f64_n1048576 untimed 112.0 ms is 417% of the 26.9 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | jax-cuda | scale_real_f64_n1048576 untimed 1249.7 ms is 5134% of the 24.3 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | jax-cuda | scale_complex_c64_n1048576 untimed 1285.9 ms is 10830% of the 11.9 ms timed region (budget 20%) |