# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **130**. Unsupported cells: **2**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1405.8 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261128_082346/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**6** of 33) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_f64_n1048576 | 13.421 (0.473) | 13.437 (0.174) | 12.195 (0.132) | 40.695 (2.742) |
| sub_f64_n1048576 | 13.676 (0.469) | 13.472 (0.373) | 12.404 (0.185) | 44.200 (1.603) |
| mul_f64_n1048576 | 13.389 (0.307) | 13.528 (0.423) | 12.314 (0.153) | 47.593 (3.093) |
| div_f64_n1048576 | 13.303 (0.276) | 13.298 (0.269) | 13.503 (0.178) | 43.943 (1.927) |
| rem_f64_n1048576 | 12.532 (0.069) | 12.533 (0.075) | 12.407 (0.129) | 44.788 (2.330) |
| pow_f64_n1048576 | 25.828 (0.065) | 25.780 (0.043) | 29.108 (0.054) | 47.607 (1.337) |
| maximum_f64_n1048576 | 12.518 (0.099) | 12.462 (0.088) | 12.401 (0.143) | 47.353 (2.097) |
| minimum_f64_n1048576 | 12.509 (0.103) | 12.454 (0.071) | 12.492 (0.120) | 42.802 (2.555) |
| neg_f64_n1048576 | 8.807 (0.075) | 8.758 (0.038) | 7.750 (0.067) | 43.041 (1.715) |
| abs_f64_n1048576 | 8.864 (0.059) | 8.831 (0.050) | 7.837 (0.070) | 43.997 (2.243) |
| sign_f64_n1048576 | 9.007 (0.055) | 8.963 (0.068) | 7.744 (0.040) | 42.563 (1.252) |
| conj_c64_n1048576 | 25.290 (0.284) | 25.199 (0.229) | 25.327 (0.122) | 40.087 (3.512) |
| exp_f64_n1048576 | 9.472 (0.036) | 9.410 (0.033) | 8.160 (0.076) | 41.029 (2.038) |
| expm1_f64_n1048576 | 10.347 (0.045) | 10.320 (0.036) | 9.870 (0.075) | 40.511 (1.153) |
| log_f64_n1048576 | 12.181 (0.026) | 12.097 (0.025) | 12.602 (0.048) | 37.421 (3.568) |
| log1p_f64_n1048576 | 12.173 (0.031) | 12.107 (0.027) | 13.139 (0.082) | 40.755 (0.935) |
| erf_f64_n1048576 | 14.178 (0.025) | 14.097 (0.031) | 14.349 (0.052) | 56.846 (1.526) |
| sin_f64_n1048576 | 10.829 (0.034) | 10.767 (0.024) | 11.103 (0.034) | 40.200 (1.972) |
| cos_f64_n1048576 | 10.974 (0.026) | 10.900 (0.054) | 11.120 (0.066) | 42.107 (1.730) |
| tanh_f64_n1048576 | 9.482 (0.038) | 9.454 (0.027) | 8.242 (0.059) | 41.499 (1.656) |
| sqrt_f64_n1048576 | 9.272 (0.051) | 9.206 (0.044) | 7.794 (0.056) | 41.438 (1.258) |
| rsqrt_f64_n1048576 | 9.145 (0.032) | 9.117 (0.024) | 7.872 (0.033) | 54.015 (2.360) |
| compare_f64_n1048576 † | 7.478 (0.013) | 7.404 (0.014) | 7.008 (0.024) | 40.647 (0.839) |
| where_select_f64_n1048576 | 13.651 (0.057) | 13.654 (0.049) | 14.299 (0.072) | 45.273 (1.813) |
| clamp_f64_n1048576 | 31.955 (0.195) | 46.164 (0.266) | 10.246 (0.041) | 41.502 (1.336) |
| reduce_sum_f64_n1048576 † | 14.726 (0.015) | 14.706 (0.011) | 7.056 (0.006) | 41.284 (1.470) |
| reduce_prod_f64_n1048576 † | 14.694 (0.011) | 14.637 (0.012) | 7.182 (0.014) | 44.177 (1.964) |
| reduce_max_f64_n1048576 † | 15.480 (0.012) | 15.441 (0.015) | 7.254 (0.006) | 44.609 (1.614) |
| reduce_min_f64_n1048576 † | 15.502 (0.020) | 15.444 (0.012) | 7.243 (0.004) | 42.068 (1.580) |
| reduce_sum_squares_f64_n1048576 | 14.691 (0.022) | 14.568 (0.012) | unsupported | unsupported |
| reduce_mean_f64_n1048576 † | 20.668 (0.019) | 20.543 (0.038) | 7.013 (0.003) | 43.876 (1.809) |
| scale_real_f64_n1048576 | 12.795 (0.052) | 12.746 (0.033) | 9.526 (0.059) | 48.653 (3.942) |
| scale_complex_c64_n1048576 | 26.673 (0.243) | 26.673 (0.191) | 25.572 (0.231) | 49.081 (4.267) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 1.408 | 16777216 | 0.271 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.251 | 16777216 | 1.523 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.284 | 16777216 | 1.346 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.283 | 16777216 | 1.351 | 0 |
| add_f64_n1048576 | jax-cuda | 32 | 15 | 1.302 | 25165824 | 0.440 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.390 | 25165824 | 1.468 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.429 | 25165824 | 1.334 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.430 | 25165824 | 1.332 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 1.328 | 16777232 | 0.288 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.328 | 16777232 | 1.165 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.023 | 16777232 | 0.373 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.477 | 16777232 | 0.259 | 0 |
| compare_f64_n1048576 | jax-cuda | 64 | 15 | 2.601 | 17825792 | 0.312 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.448 | 17825792 | 1.809 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 1.914 | 17825792 | 1.696 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 256 | 15 | 1.896 | 17825792 | 1.713 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 0.641 | 33554432 | 0.595 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.405 | 33554432 | 0.942 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.405 | 33554432 | 0.944 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.403 | 33554432 | 0.947 | 0 |
| cos_f64_n1048576 | jax-cuda | 32 | 15 | 1.347 | 16777216 | 0.283 | 5.905e-08 |
| cos_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.356 | 16777216 | 1.073 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.351 | 16777216 | 1.088 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.349 | 16777216 | 1.095 | 5.905e-08 |
| div_f64_n1048576 | jax-cuda | 32 | 15 | 1.406 | 25165824 | 0.407 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.432 | 25165824 | 1.326 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.426 | 25165824 | 1.346 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.426 | 25165824 | 1.346 | 0 |
| erf_f64_n1048576 | jax-cuda | 32 | 15 | 1.819 | 16777216 | 0.210 | 8.536e-08 |
| erf_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.459 | 16777216 | 0.832 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.454 | 16777216 | 0.842 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.451 | 16777216 | 0.847 | 8.536e-08 |
| exp_f64_n1048576 | jax-cuda | 32 | 15 | 1.313 | 16777216 | 0.291 | 1.093e-07 |
| exp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.261 | 16777216 | 1.462 | 1.093e-07 |
| exp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.303 | 16777216 | 1.260 | 1.093e-07 |
| exp_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.301 | 16777216 | 1.268 | 1.093e-07 |
| expm1_f64_n1048576 | jax-cuda | 32 | 15 | 1.296 | 16777216 | 0.295 | 7.39e-08 |
| expm1_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.316 | 16777216 | 1.209 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.331 | 16777216 | 1.153 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.330 | 16777216 | 1.156 | 7.39e-08 |
| log1p_f64_n1048576 | jax-cuda | 32 | 15 | 1.304 | 16777216 | 0.293 | 1.106e-07 |
| log1p_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.420 | 16777216 | 0.908 | 1.106e-07 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.390 | 16777216 | 0.980 | 1.106e-07 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.387 | 16777216 | 0.986 | 1.106e-07 |
| log_f64_n1048576 | jax-cuda | 32 | 15 | 1.197 | 16777216 | 0.319 | 4.308e-08 |
| log_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.403 | 16777216 | 0.947 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.390 | 16777216 | 0.980 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.387 | 16777216 | 0.987 | 4.308e-08 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 1.515 | 25165824 | 0.378 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.397 | 25165824 | 1.444 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.401 | 25165824 | 1.430 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.399 | 25165824 | 1.436 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 1.370 | 25165824 | 0.418 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.400 | 25165824 | 1.433 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.400 | 25165824 | 1.431 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.399 | 25165824 | 1.437 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 1.523 | 25165824 | 0.376 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.394 | 25165824 | 1.454 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.428 | 25165824 | 1.337 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.433 | 25165824 | 1.323 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 1.377 | 16777216 | 0.277 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.248 | 16777216 | 1.540 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.282 | 16777216 | 1.355 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.280 | 16777216 | 1.363 | 0 |
| pow_f64_n1048576 | jax-cuda | 32 | 15 | 1.523 | 25165824 | 0.376 | 1.48e-07 |
| pow_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.931 | 25165824 | 0.615 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.826 | 25165824 | 0.693 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.825 | 25165824 | 0.694 | 1.48e-07 |
| reduce_max_f64_n1048576 | jax-cuda | 64 | 15 | 2.855 | 8396800 | 0.134 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.714 | 8396800 | 0.823 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 3.963 | 8396800 | 0.386 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 256 | 15 | 3.953 | 8396800 | 0.387 | 0 |
| reduce_mean_f64_n1048576 | jax-cuda | 64 | 15 | 2.808 | 8396800 | 0.136 | 6.5e-09 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.591 | 8396800 | 0.852 | 3.9e-09 |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.646 | 8396800 | 0.289 | 3.894e-09 |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 2.630 | 8396800 | 0.291 | 3.894e-09 |
| reduce_min_f64_n1048576 | jax-cuda | 64 | 15 | 2.692 | 8396800 | 0.142 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.709 | 8396800 | 0.825 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 3.969 | 8396800 | 0.385 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 256 | 15 | 3.954 | 8396800 | 0.387 | 0 |
| reduce_prod_f64_n1048576 | jax-cuda | 64 | 15 | 2.827 | 8396800 | 0.135 | 4.436e-06 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.677 | 8396800 | 0.832 | 4.216e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 3.762 | 8396800 | 0.406 | 4.438e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 256 | 15 | 3.747 | 8396800 | 0.408 | 4.438e-06 |
| reduce_sum_f64_n1048576 | jax-cuda | 64 | 15 | 2.642 | 8396800 | 0.145 | 4.449e-06 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.613 | 8396800 | 0.846 | 2.961e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 3.770 | 8396800 | 0.406 | 3.305e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 256 | 15 | 3.765 | 8396800 | 0.406 | 3.305e-06 |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 3.761 | 8396800 | 0.407 | 4.939e-07 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 256 | 15 | 3.729 | 8396800 | 0.410 | 4.939e-07 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 1.433 | 25165824 | 0.400 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.397 | 25165824 | 1.443 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.401 | 25165824 | 1.428 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.401 | 25165824 | 1.428 | 0 |
| rsqrt_f64_n1048576 | jax-cuda | 32 | 15 | 1.728 | 16777216 | 0.221 | 6.504e-08 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.252 | 16777216 | 1.516 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.293 | 16777216 | 1.305 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.292 | 16777216 | 1.309 | 6.504e-08 |
| scale_complex_c64_n1048576 | jax-cuda | 16 | 15 | 0.785 | 33554448 | 0.486 | 4.129e-08 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.409 | 33554448 | 0.933 | 0 |
| scale_complex_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.427 | 33554448 | 0.895 | 4.129e-08 |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.427 | 33554448 | 0.895 | 4.129e-08 |
| scale_real_f64_n1048576 | jax-cuda | 32 | 15 | 1.557 | 16777224 | 0.245 | 0 |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.305 | 16777224 | 1.253 | 0 |
| scale_real_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.409 | 16777224 | 0.933 | 0 |
| scale_real_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.408 | 16777224 | 0.936 | 0 |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 1.362 | 16777216 | 0.280 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.248 | 16777216 | 1.541 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.288 | 16777216 | 1.325 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.287 | 16777216 | 1.332 | 0 |
| sin_f64_n1048576 | jax-cuda | 32 | 15 | 1.286 | 16777216 | 0.297 | 4.153e-08 |
| sin_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.355 | 16777216 | 1.075 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.347 | 16777216 | 1.102 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.345 | 16777216 | 1.108 | 4.153e-08 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 1.326 | 16777216 | 0.288 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.249 | 16777216 | 1.531 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.297 | 16777216 | 1.287 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.295 | 16777216 | 1.296 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 1.414 | 25165824 | 0.405 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.397 | 25165824 | 1.443 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.438 | 25165824 | 1.309 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.431 | 25165824 | 1.329 | 0 |
| tanh_f64_n1048576 | jax-cuda | 32 | 15 | 1.328 | 16777216 | 0.288 | 4.437e-08 |
| tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.264 | 16777216 | 1.448 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.303 | 16777216 | 1.259 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.303 | 16777216 | 1.262 | 4.437e-08 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 1.449 | 26214400 | 0.412 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.458 | 26214400 | 1.304 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.437 | 26214400 | 1.366 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.437 | 26214400 | 1.366 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261128_082346/gpu_isolated_ops.yaml
timestamp: '2026-11-28T08:23:46+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 28b35e0408e272b66d7460e3fa12213799c543f9
  dirty: false
resolved_deps:
  strided-perm:
  - 0.8.0
  - 0.15.1
  strided-view:
  - 0.7.1
  - 0.8.1
  threadteam:
  - 0.1.0
  - 0.6.1
host:
  cpu: AMD EPYC 9554 64-Core Processor
  logical_cpus: 252
  power:
    smt:
      active: '0'
      control: notsupported
  numa:
    nodes: 2
    cpulists:
      '0': 0-125
      '1': 126-251
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
  hostname: 32b786430f5d
  os: Linux-6.8.0-149-generic-x86_64-with-glibc2.39
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
    GPU_RUNTIME_SHA256: a261fcdef53ecfd3885b6d565dc834faef59287ce2da88c3d6754be96741f84d
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
      \ by MKL)\n  - NNPACK is enabled\n  - CPU capability usage: AVX512\n  - CUDA\
      \ Runtime 12.8\n  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120\n\
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
  device_uuid: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
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
  - 0.096045
  - 0.091508
  - 0.096766
  - 0.096094
  - 0.092448
  - 0.095535
  - 0.095223
  - 0.094362
  - 0.096455
  - 0.096105
  - 0.093761
  - 0.095063
  - 0.096797
  - 0.095473
  - 0.094283
  median_ms: 0.095473
  bytes_per_second: 1405818692195.6992
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-28T08:36:17.796735+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
  cuda_visible_devices: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
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
    src/bin/benchmark_gpu_public_api.rs: f807adb35a2c03383fde11227a47dffbf0ec5cc08c45ce298f7d9aa6341bbb67
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: e8121a793ab45e565d150633b7e9a473b4d4f2c457ae36baed344f2c809fcad5
    src/bin/benchmark_gpu_public_api/multi.rs: 950cb1b3c1a7cca37665b58e3de074bd906dafb8c84b17a51c27ce797a73afa8
    src/bin/benchmark_gpu_public_api/plan.rs: 108edbad472e42b1a734552826d6d233085b581a3484cc4444471435160a6fd8
    src/bin/benchmark_gpu_public_api/read_layout.rs: f51757a34c90ce68e6fae0994df74169ed0438022031465349c7379c7b2c85e0
    src/bin/benchmark_gpu_public_api/reuse.rs: 53bee3baaeba93f4e1ee57d0d1b9553c189edd51d1940ea630ad0dbaa05b11eb
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 88038508963fbc13303fb54b3898808e4303131321cabedb9f38255941ead438
    src/bin/benchmark_gpu_public_api/state.rs: e1cc2781ce7392154f32cc19372c9fc41ae29765aedc544c7ae7bf02f873c12b
    src/bin/benchmark_gpu_public_api/structural.rs: 09b3cbf0dc0579978344c1aa738f6367112a0c25236924441f690a59de3fb80e
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: d1ff68f86496d991bcbc730138fe6e3076001b3f23cc1a94c44b96f6857cb8ed
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 7995bcf409258185e07210969a738896f64860b67df251fcd1100d11128a5431
    data/results/nvidia-gpu/gpu/isolated_ops/20261128_082346/fixtures/manifest.json: 2fac9150044c06526d293df21bf67d626eeab8bffea134d443f55bee346f38dd
    target/release/benchmark_gpu_public_api: 54dda9d9f7540806d7ff5a5d68e1312bc33b0d58871d8e3e9467a819e7c59318
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:36:41Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:36:31Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:36:23Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:36:26Z'
  compile:
    case_wall_seconds: 15.913
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
| add_f64_n1048576 | tenferro-cuda-direct | add_f64_n1048576 untimed 143.8 ms is 2210% of the 6.5 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-direct | sub_f64_n1048576 untimed 80.7 ms is 1227% of the 6.6 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-direct | mul_f64_n1048576 untimed 90.5 ms is 1408% of the 6.4 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-direct | div_f64_n1048576 untimed 90.8 ms is 1417% of the 6.4 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-direct | rem_f64_n1048576 untimed 93.3 ms is 1548% of the 6.0 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-direct | pow_f64_n1048576 untimed 75.6 ms is 609% of the 12.4 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-direct | maximum_f64_n1048576 untimed 87.3 ms is 1453% of the 6.0 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-direct | minimum_f64_n1048576 untimed 108.1 ms is 1796% of the 6.0 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-direct | neg_f64_n1048576 untimed 80.8 ms is 1905% of the 4.2 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-direct | abs_f64_n1048576 untimed 67.1 ms is 1576% of the 4.3 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-direct | sign_f64_n1048576 untimed 67.2 ms is 1554% of the 4.3 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-direct | conj_c64_n1048576 untimed 98.4 ms is 1620% of the 6.1 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-direct | exp_f64_n1048576 untimed 79.6 ms is 1748% of the 4.6 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-direct | expm1_f64_n1048576 untimed 68.1 ms is 1371% of the 5.0 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-direct | log_f64_n1048576 untimed 67.3 ms is 1151% of the 5.9 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-direct | log1p_f64_n1048576 untimed 70.8 ms is 1211% of the 5.8 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-direct | erf_f64_n1048576 untimed 78.0 ms is 1145% of the 6.8 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-direct | sin_f64_n1048576 untimed 67.1 ms is 1291% of the 5.2 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-direct | cos_f64_n1048576 untimed 67.3 ms is 1276% of the 5.3 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-direct | tanh_f64_n1048576 untimed 71.3 ms is 1561% of the 4.6 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-direct | sqrt_f64_n1048576 untimed 66.9 ms is 1506% of the 4.4 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | rsqrt_f64_n1048576 untimed 73.9 ms is 1684% of the 4.4 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-direct | compare_f64_n1048576 untimed 50.3 ms is 175% of the 28.7 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-direct | where_select_f64_n1048576 untimed 79.9 ms is 1217% of the 6.6 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-direct | clamp_f64_n1048576 untimed 90.0 ms is 587% of the 15.3 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | reduce_sum_f64_n1048576 untimed 13.5 ms is 24% of the 56.6 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | reduce_prod_f64_n1048576 untimed 13.3 ms is 24% of the 56.4 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | reduce_max_f64_n1048576 untimed 12.8 ms is 21% of the 59.5 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | reduce_min_f64_n1048576 untimed 12.7 ms is 21% of the 59.5 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_f64_n1048576 untimed 12.3 ms is 22% of the 56.4 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | reduce_mean_f64_n1048576 untimed 9.1 ms is 23% of the 39.7 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | tenferro-cuda-direct | scale_real_f64_n1048576 untimed 70.9 ms is 1155% of the 6.1 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | tenferro-cuda-direct | scale_complex_c64_n1048576 untimed 103.4 ms is 1613% of the 6.4 ms timed region (budget 20%) |
| add_f64_n1048576 | tenferro-cuda-eager | add_f64_n1048576 untimed 154.2 ms is 2399% of the 6.4 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-eager | sub_f64_n1048576 untimed 77.2 ms is 1188% of the 6.5 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-eager | mul_f64_n1048576 untimed 68.6 ms is 1051% of the 6.5 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-eager | div_f64_n1048576 untimed 68.9 ms is 1072% of the 6.4 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-eager | rem_f64_n1048576 untimed 79.1 ms is 1310% of the 6.0 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-eager | pow_f64_n1048576 untimed 77.5 ms is 626% of the 12.4 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-eager | maximum_f64_n1048576 untimed 68.7 ms is 1147% of the 6.0 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-eager | minimum_f64_n1048576 untimed 67.5 ms is 1122% of the 6.0 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-eager | neg_f64_n1048576 untimed 71.2 ms is 1692% of the 4.2 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-eager | abs_f64_n1048576 untimed 79.6 ms is 1874% of the 4.2 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-eager | sign_f64_n1048576 untimed 65.7 ms is 1527% of the 4.3 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-eager | conj_c64_n1048576 untimed 102.6 ms is 1694% of the 6.1 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-eager | exp_f64_n1048576 untimed 74.2 ms is 1641% of the 4.5 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-eager | expm1_f64_n1048576 untimed 66.9 ms is 1352% of the 5.0 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-eager | log_f64_n1048576 untimed 66.3 ms is 1141% of the 5.8 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-eager | log1p_f64_n1048576 untimed 70.1 ms is 1206% of the 5.8 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-eager | erf_f64_n1048576 untimed 76.9 ms is 1134% of the 6.8 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-eager | sin_f64_n1048576 untimed 66.2 ms is 1281% of the 5.2 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-eager | cos_f64_n1048576 untimed 66.1 ms is 1253% of the 5.3 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-eager | tanh_f64_n1048576 untimed 69.8 ms is 1534% of the 4.5 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-eager | sqrt_f64_n1048576 untimed 66.2 ms is 1496% of the 4.4 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | rsqrt_f64_n1048576 untimed 72.6 ms is 1660% of the 4.4 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-eager | compare_f64_n1048576 untimed 50.8 ms is 179% of the 28.5 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-eager | where_select_f64_n1048576 untimed 72.9 ms is 1112% of the 6.6 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-eager | clamp_f64_n1048576 untimed 65.7 ms is 297% of the 22.1 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | reduce_sum_f64_n1048576 untimed 13.8 ms is 24% of the 56.5 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | reduce_prod_f64_n1048576 untimed 13.6 ms is 24% of the 56.2 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | reduce_max_f64_n1048576 untimed 13.1 ms is 22% of the 59.3 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | reduce_min_f64_n1048576 untimed 13.1 ms is 22% of the 59.3 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | reduce_sum_squares_f64_n1048576 untimed 12.7 ms is 23% of the 56.2 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | reduce_mean_f64_n1048576 untimed 9.3 ms is 24% of the 39.5 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | tenferro-cuda-eager | scale_real_f64_n1048576 untimed 66.0 ms is 1078% of the 6.1 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | scale_complex_c64_n1048576 untimed 109.9 ms is 1723% of the 6.4 ms timed region (budget 20%) |
| add_f64_n1048576 | pytorch-cuda | add_f64_n1048576 untimed 170.0 ms is 2908% of the 5.8 ms timed region (budget 20%) |
| sub_f64_n1048576 | pytorch-cuda | sub_f64_n1048576 untimed 60.7 ms is 1019% of the 6.0 ms timed region (budget 20%) |
| mul_f64_n1048576 | pytorch-cuda | mul_f64_n1048576 untimed 56.1 ms is 954% of the 5.9 ms timed region (budget 20%) |
| div_f64_n1048576 | pytorch-cuda | div_f64_n1048576 untimed 60.8 ms is 939% of the 6.5 ms timed region (budget 20%) |
| rem_f64_n1048576 | pytorch-cuda | rem_f64_n1048576 untimed 75.0 ms is 1260% of the 6.0 ms timed region (budget 20%) |
| pow_f64_n1048576 | pytorch-cuda | pow_f64_n1048576 untimed 85.7 ms is 613% of the 14.0 ms timed region (budget 20%) |
| maximum_f64_n1048576 | pytorch-cuda | maximum_f64_n1048576 untimed 66.0 ms is 1106% of the 6.0 ms timed region (budget 20%) |
| minimum_f64_n1048576 | pytorch-cuda | minimum_f64_n1048576 untimed 43.3 ms is 721% of the 6.0 ms timed region (budget 20%) |
| neg_f64_n1048576 | pytorch-cuda | neg_f64_n1048576 untimed 52.4 ms is 1408% of the 3.7 ms timed region (budget 20%) |
| abs_f64_n1048576 | pytorch-cuda | abs_f64_n1048576 untimed 57.5 ms is 1528% of the 3.8 ms timed region (budget 20%) |
| sign_f64_n1048576 | pytorch-cuda | sign_f64_n1048576 untimed 55.8 ms is 1498% of the 3.7 ms timed region (budget 20%) |
| conj_c64_n1048576 | pytorch-cuda | conj_c64_n1048576 untimed 77.4 ms is 1274% of the 6.1 ms timed region (budget 20%) |
| exp_f64_n1048576 | pytorch-cuda | exp_f64_n1048576 untimed 53.7 ms is 1371% of the 3.9 ms timed region (budget 20%) |
| expm1_f64_n1048576 | pytorch-cuda | expm1_f64_n1048576 untimed 57.1 ms is 1207% of the 4.7 ms timed region (budget 20%) |
| log_f64_n1048576 | pytorch-cuda | log_f64_n1048576 untimed 62.4 ms is 1029% of the 6.1 ms timed region (budget 20%) |
| log1p_f64_n1048576 | pytorch-cuda | log1p_f64_n1048576 untimed 58.8 ms is 933% of the 6.3 ms timed region (budget 20%) |
| erf_f64_n1048576 | pytorch-cuda | erf_f64_n1048576 untimed 65.8 ms is 955% of the 6.9 ms timed region (budget 20%) |
| sin_f64_n1048576 | pytorch-cuda | sin_f64_n1048576 untimed 60.9 ms is 1142% of the 5.3 ms timed region (budget 20%) |
| cos_f64_n1048576 | pytorch-cuda | cos_f64_n1048576 untimed 63.7 ms is 1193% of the 5.3 ms timed region (budget 20%) |
| tanh_f64_n1048576 | pytorch-cuda | tanh_f64_n1048576 untimed 55.5 ms is 1398% of the 4.0 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | pytorch-cuda | sqrt_f64_n1048576 untimed 58.9 ms is 1559% of the 3.8 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | pytorch-cuda | rsqrt_f64_n1048576 untimed 68.6 ms is 1812% of the 3.8 ms timed region (budget 20%) |
| compare_f64_n1048576 | pytorch-cuda | compare_f64_n1048576 untimed 56.2 ms is 835% of the 6.7 ms timed region (budget 20%) |
| where_select_f64_n1048576 | pytorch-cuda | where_select_f64_n1048576 untimed 78.9 ms is 1149% of the 6.9 ms timed region (budget 20%) |
| clamp_f64_n1048576 | pytorch-cuda | clamp_f64_n1048576 untimed 50.1 ms is 1019% of the 4.9 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | pytorch-cuda | reduce_sum_f64_n1048576 untimed 30.0 ms is 55% of the 54.2 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | pytorch-cuda | reduce_prod_f64_n1048576 untimed 21.4 ms is 39% of the 55.2 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | pytorch-cuda | reduce_max_f64_n1048576 untimed 22.7 ms is 41% of the 55.8 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | pytorch-cuda | reduce_min_f64_n1048576 untimed 22.9 ms is 41% of the 55.6 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | pytorch-cuda | reduce_mean_f64_n1048576 untimed 24.1 ms is 45% of the 53.9 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | pytorch-cuda | scale_real_f64_n1048576 untimed 59.3 ms is 1297% of the 4.6 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | pytorch-cuda | scale_complex_c64_n1048576 untimed 103.3 ms is 1682% of the 6.1 ms timed region (budget 20%) |
| add_f64_n1048576 | jax-cuda | add_f64_n1048576 untimed 494.1 ms is 2509% of the 19.7 ms timed region (budget 20%) |
| sub_f64_n1048576 | jax-cuda | sub_f64_n1048576 untimed 285.0 ms is 1340% of the 21.3 ms timed region (budget 20%) |
| mul_f64_n1048576 | jax-cuda | mul_f64_n1048576 untimed 118.6 ms is 518% of the 22.9 ms timed region (budget 20%) |
| div_f64_n1048576 | jax-cuda | div_f64_n1048576 untimed 249.8 ms is 1170% of the 21.4 ms timed region (budget 20%) |
| rem_f64_n1048576 | jax-cuda | rem_f64_n1048576 untimed 310.1 ms is 1437% of the 21.6 ms timed region (budget 20%) |
| pow_f64_n1048576 | jax-cuda | pow_f64_n1048576 untimed 129.5 ms is 564% of the 23.0 ms timed region (budget 20%) |
| maximum_f64_n1048576 | jax-cuda | maximum_f64_n1048576 untimed 245.0 ms is 1067% of the 22.9 ms timed region (budget 20%) |
| minimum_f64_n1048576 | jax-cuda | minimum_f64_n1048576 untimed 119.9 ms is 575% of the 20.8 ms timed region (budget 20%) |
| neg_f64_n1048576 | jax-cuda | neg_f64_n1048576 untimed 217.4 ms is 1051% of the 20.7 ms timed region (budget 20%) |
| abs_f64_n1048576 | jax-cuda | abs_f64_n1048576 untimed 243.9 ms is 1169% of the 20.9 ms timed region (budget 20%) |
| sign_f64_n1048576 | jax-cuda | sign_f64_n1048576 untimed 257.5 ms is 1252% of the 20.6 ms timed region (budget 20%) |
| conj_c64_n1048576 | jax-cuda | conj_c64_n1048576 untimed 292.5 ms is 2991% of the 9.8 ms timed region (budget 20%) |
| exp_f64_n1048576 | jax-cuda | exp_f64_n1048576 untimed 251.3 ms is 1265% of the 19.9 ms timed region (budget 20%) |
| expm1_f64_n1048576 | jax-cuda | expm1_f64_n1048576 untimed 261.1 ms is 1339% of the 19.5 ms timed region (budget 20%) |
| log_f64_n1048576 | jax-cuda | log_f64_n1048576 untimed 265.3 ms is 1458% of the 18.2 ms timed region (budget 20%) |
| log1p_f64_n1048576 | jax-cuda | log1p_f64_n1048576 untimed 248.7 ms is 1259% of the 19.8 ms timed region (budget 20%) |
| erf_f64_n1048576 | jax-cuda | erf_f64_n1048576 untimed 235.4 ms is 866% of the 27.2 ms timed region (budget 20%) |
| sin_f64_n1048576 | jax-cuda | sin_f64_n1048576 untimed 284.3 ms is 1460% of the 19.5 ms timed region (budget 20%) |
| cos_f64_n1048576 | jax-cuda | cos_f64_n1048576 untimed 258.4 ms is 1277% of the 20.2 ms timed region (budget 20%) |
| tanh_f64_n1048576 | jax-cuda | tanh_f64_n1048576 untimed 223.1 ms is 1117% of the 20.0 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | jax-cuda | sqrt_f64_n1048576 untimed 257.8 ms is 1294% of the 19.9 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | jax-cuda | rsqrt_f64_n1048576 untimed 234.8 ms is 899% of the 26.1 ms timed region (budget 20%) |
| compare_f64_n1048576 | jax-cuda | compare_f64_n1048576 untimed 149.3 ms is 382% of the 39.1 ms timed region (budget 20%) |
| where_select_f64_n1048576 | jax-cuda | where_select_f64_n1048576 untimed 228.9 ms is 1057% of the 21.7 ms timed region (budget 20%) |
| clamp_f64_n1048576 | jax-cuda | clamp_f64_n1048576 untimed 249.3 ms is 1265% of the 19.7 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | jax-cuda | reduce_sum_f64_n1048576 untimed 145.6 ms is 365% of the 39.9 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | jax-cuda | reduce_prod_f64_n1048576 untimed 143.0 ms is 336% of the 42.5 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | jax-cuda | reduce_max_f64_n1048576 untimed 222.4 ms is 521% of the 42.6 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | jax-cuda | reduce_min_f64_n1048576 untimed 71.7 ms is 177% of the 40.4 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | jax-cuda | reduce_mean_f64_n1048576 untimed 61.3 ms is 145% of the 42.3 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | jax-cuda | scale_real_f64_n1048576 untimed 260.2 ms is 1117% of the 23.3 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | jax-cuda | scale_complex_c64_n1048576 untimed 335.2 ms is 2857% of the 11.7 ms timed region (budget 20%) |