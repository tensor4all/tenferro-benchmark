# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **130**. Unsupported cells: **2**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1444.0 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261126_062950/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**7** of 33) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_f64_n1048576 † | 13.052 (0.364) | 13.143 (0.289) | 12.043 (0.300) | 65.164 (1.148) |
| sub_f64_n1048576 | 12.866 (0.244) | 12.975 (0.197) | 11.647 (0.213) | 64.992 (2.328) |
| mul_f64_n1048576 | 12.892 (0.230) | 13.093 (0.396) | 11.961 (0.184) | 61.499 (3.324) |
| div_f64_n1048576 | 12.815 (0.338) | 13.115 (0.443) | 13.133 (0.160) | 59.674 (3.141) |
| rem_f64_n1048576 | 12.293 (0.028) | 12.457 (0.061) | 11.857 (0.217) | 56.897 (2.123) |
| pow_f64_n1048576 | 25.632 (0.032) | 25.797 (0.032) | 29.223 (0.129) | 64.211 (2.605) |
| maximum_f64_n1048576 | 12.200 (0.049) | 12.367 (0.038) | 11.925 (0.242) | 57.807 (0.843) |
| minimum_f64_n1048576 | 12.224 (0.042) | 12.452 (0.184) | 11.653 (0.158) | 57.422 (0.804) |
| neg_f64_n1048576 | 8.674 (0.048) | 8.808 (0.047) | 7.587 (0.044) | 54.698 (1.152) |
| abs_f64_n1048576 | 8.694 (0.050) | 8.835 (0.039) | 9.418 (4.210) | 51.207 (2.100) |
| sign_f64_n1048576 | 8.863 (0.037) | 9.052 (0.068) | 7.604 (0.052) | 54.001 (1.639) |
| conj_c64_n1048576 | 23.928 (0.059) | 24.058 (0.100) | 24.080 (0.228) | 52.207 (3.548) |
| exp_f64_n1048576 | 9.345 (0.024) | 9.453 (0.046) | 8.178 (0.063) | 53.608 (2.426) |
| expm1_f64_n1048576 | 10.228 (0.050) | 10.374 (0.031) | 9.861 (0.163) | 55.383 (2.056) |
| log_f64_n1048576 | 12.054 (0.015) | 12.173 (0.025) | 12.453 (0.066) | 51.287 (0.911) |
| log1p_f64_n1048576 | 12.055 (0.056) | 12.189 (0.015) | 13.081 (0.051) | 53.199 (2.053) |
| erf_f64_n1048576 | 14.050 (0.011) | 14.155 (0.027) | 14.287 (0.027) | 77.103 (2.543) |
| sin_f64_n1048576 | 10.678 (0.038) | 10.823 (0.028) | 10.930 (0.034) | 53.576 (1.074) |
| cos_f64_n1048576 | 10.842 (0.152) | 10.958 (0.022) | 10.959 (0.030) | 54.219 (0.796) |
| tanh_f64_n1048576 | 9.360 (0.028) | 9.539 (0.038) | 8.167 (0.039) | 52.925 (1.644) |
| sqrt_f64_n1048576 | 9.096 (0.057) | 9.269 (0.032) | 7.686 (0.050) | 52.679 (3.683) |
| rsqrt_f64_n1048576 | 9.019 (0.044) | 9.170 (0.065) | 7.697 (0.048) | 75.081 (3.753) |
| compare_f64_n1048576 † | 7.364 (0.012) | 7.475 (0.010) | 6.965 (0.119) | 54.034 (1.542) |
| where_select_f64_n1048576 | 13.280 (0.063) | 13.477 (0.062) | 13.812 (0.138) | 58.277 (1.197) |
| clamp_f64_n1048576 | 31.219 (0.108) | 45.139 (0.186) | 10.273 (0.045) | 51.837 (2.850) |
| reduce_sum_f64_n1048576 † | 14.640 (0.030) | 14.704 (0.027) | 7.109 (0.010) | 116.010 (2.617) |
| reduce_prod_f64_n1048576 † | 14.535 (0.015) | 14.659 (0.021) | 7.184 (0.007) | 113.898 (2.453) |
| reduce_max_f64_n1048576 † | 15.423 (0.014) | 15.679 (0.279) | 7.262 (0.014) | 112.556 (14.647) |
| reduce_min_f64_n1048576 † | 15.460 (0.015) | 15.574 (0.021) | 7.282 (0.012) | 114.624 (2.094) |
| reduce_sum_squares_f64_n1048576 | 14.526 (0.017) | 14.890 (0.280) | unsupported | unsupported |
| reduce_mean_f64_n1048576 † | 20.379 (0.099) | 24.123 (0.810) | 7.053 (0.047) | 125.840 (3.107) |
| scale_real_f64_n1048576 | 12.675 (0.025) | 12.937 (0.041) | 9.504 (0.066) | 61.951 (2.910) |
| scale_complex_c64_n1048576 | 25.376 (0.244) | 25.591 (0.190) | 24.481 (0.188) | 60.422 (3.124) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 1.639 | 16777216 | 0.227 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.301 | 16777216 | 1.234 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.278 | 16777216 | 1.336 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.283 | 16777216 | 1.315 | 0 |
| add_f64_n1048576 | jax-cuda | 16 | 15 | 1.043 | 25165824 | 0.267 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.385 | 25165824 | 1.447 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.418 | 25165824 | 1.335 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.421 | 25165824 | 1.326 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 1.659 | 16777232 | 0.224 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.329 | 16777232 | 1.131 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.999 | 16777232 | 0.372 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.444 | 16777232 | 0.257 | 0 |
| compare_f64_n1048576 | jax-cuda | 64 | 15 | 3.458 | 17825792 | 0.228 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.446 | 17825792 | 1.772 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 1.885 | 17825792 | 1.676 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 256 | 15 | 1.914 | 17825792 | 1.652 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 0.835 | 33554432 | 0.445 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.385 | 33554432 | 0.965 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.383 | 33554432 | 0.971 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.385 | 33554432 | 0.966 | 0 |
| cos_f64_n1048576 | jax-cuda | 32 | 15 | 1.735 | 16777216 | 0.214 | 5.905e-08 |
| cos_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.351 | 16777216 | 1.060 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.347 | 16777216 | 1.072 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.351 | 16777216 | 1.060 | 5.905e-08 |
| div_f64_n1048576 | jax-cuda | 32 | 15 | 1.910 | 25165824 | 0.292 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.420 | 25165824 | 1.327 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.410 | 25165824 | 1.360 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.420 | 25165824 | 1.329 | 0 |
| erf_f64_n1048576 | jax-cuda | 32 | 15 | 2.467 | 16777216 | 0.151 | 8.536e-08 |
| erf_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.457 | 16777216 | 0.813 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.450 | 16777216 | 0.827 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.453 | 16777216 | 0.821 | 8.536e-08 |
| exp_f64_n1048576 | jax-cuda | 32 | 15 | 1.715 | 16777216 | 0.217 | 1.095e-07 |
| exp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.262 | 16777216 | 1.421 | 1.095e-07 |
| exp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.299 | 16777216 | 1.243 | 1.095e-07 |
| exp_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.302 | 16777216 | 1.229 | 1.095e-07 |
| expm1_f64_n1048576 | jax-cuda | 32 | 15 | 1.772 | 16777216 | 0.210 | 7.39e-08 |
| expm1_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.316 | 16777216 | 1.178 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.327 | 16777216 | 1.136 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.332 | 16777216 | 1.120 | 7.39e-08 |
| log1p_f64_n1048576 | jax-cuda | 32 | 15 | 1.702 | 16777216 | 0.218 | 1.11e-07 |
| log1p_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.419 | 16777216 | 0.888 | 1.11e-07 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.386 | 16777216 | 0.964 | 1.11e-07 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.390 | 16777216 | 0.953 | 1.11e-07 |
| log_f64_n1048576 | jax-cuda | 32 | 15 | 1.641 | 16777216 | 0.227 | 4.308e-08 |
| log_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.398 | 16777216 | 0.933 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.386 | 16777216 | 0.964 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.390 | 16777216 | 0.954 | 4.308e-08 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 1.850 | 25165824 | 0.301 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.382 | 25165824 | 1.461 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.390 | 25165824 | 1.429 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.396 | 25165824 | 1.409 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 1.838 | 25165824 | 0.304 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.373 | 25165824 | 1.496 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.391 | 25165824 | 1.426 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.398 | 25165824 | 1.400 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 1.968 | 25165824 | 0.283 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.383 | 25165824 | 1.457 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.413 | 25165824 | 1.352 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.419 | 25165824 | 1.331 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 1.750 | 16777216 | 0.212 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.243 | 16777216 | 1.531 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.278 | 16777216 | 1.339 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.282 | 16777216 | 1.319 | 0 |
| pow_f64_n1048576 | jax-cuda | 32 | 15 | 2.055 | 25165824 | 0.271 | 1.48e-07 |
| pow_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.935 | 25165824 | 0.596 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.820 | 25165824 | 0.680 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.825 | 25165824 | 0.676 | 1.48e-07 |
| reduce_max_f64_n1048576 | jax-cuda | 32 | 15 | 3.602 | 8396800 | 0.052 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.718 | 8396800 | 0.801 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 3.948 | 8396800 | 0.377 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 2.007 | 8396800 | 0.371 | 0 |
| reduce_mean_f64_n1048576 | jax-cuda | 16 | 15 | 2.013 | 8396800 | 0.046 | 6.5e-09 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.611 | 8396800 | 0.824 | 3.9e-09 |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.609 | 8396800 | 0.285 | 3.894e-09 |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 3.088 | 8396800 | 0.241 | 3.894e-09 |
| reduce_min_f64_n1048576 | jax-cuda | 16 | 15 | 1.834 | 8396800 | 0.051 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.728 | 8396800 | 0.799 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 3.958 | 8396800 | 0.376 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 256 | 15 | 3.987 | 8396800 | 0.373 | 0 |
| reduce_prod_f64_n1048576 | jax-cuda | 32 | 15 | 3.645 | 8396800 | 0.051 | 4.436e-06 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.678 | 8396800 | 0.809 | 4.216e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 3.721 | 8396800 | 0.400 | 4.438e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 256 | 15 | 3.753 | 8396800 | 0.397 | 4.438e-06 |
| reduce_sum_f64_n1048576 | jax-cuda | 16 | 15 | 1.856 | 8396800 | 0.050 | 4.449e-06 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.640 | 8396800 | 0.818 | 2.961e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 1.874 | 8396800 | 0.397 | 3.305e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 256 | 15 | 3.764 | 8396800 | 0.395 | 3.305e-06 |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 3.719 | 8396800 | 0.400 | 4.939e-07 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 256 | 15 | 3.812 | 8396800 | 0.391 | 4.939e-07 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 1.821 | 25165824 | 0.306 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.379 | 25165824 | 1.470 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.393 | 25165824 | 1.418 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.399 | 25165824 | 1.399 | 0 |
| rsqrt_f64_n1048576 | jax-cuda | 32 | 15 | 2.403 | 16777216 | 0.155 | 6.504e-08 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.246 | 16777216 | 1.510 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.289 | 16777216 | 1.288 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.293 | 16777216 | 1.267 | 6.504e-08 |
| scale_complex_c64_n1048576 | jax-cuda | 16 | 15 | 0.967 | 33554448 | 0.385 | 4.129e-08 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.392 | 33554448 | 0.949 | 0 |
| scale_complex_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.406 | 33554448 | 0.916 | 4.129e-08 |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.409 | 33554448 | 0.908 | 4.129e-08 |
| scale_real_f64_n1048576 | jax-cuda | 32 | 15 | 1.982 | 16777224 | 0.188 | 0 |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.304 | 16777224 | 1.223 | 0 |
| scale_real_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.406 | 16777224 | 0.917 | 0 |
| scale_real_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.414 | 16777224 | 0.898 | 0 |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 1.728 | 16777216 | 0.215 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.243 | 16777216 | 1.528 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.284 | 16777216 | 1.311 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.290 | 16777216 | 1.284 | 0 |
| sin_f64_n1048576 | jax-cuda | 32 | 15 | 1.714 | 16777216 | 0.217 | 4.153e-08 |
| sin_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.350 | 16777216 | 1.063 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.342 | 16777216 | 1.088 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.346 | 16777216 | 1.074 | 4.153e-08 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 1.686 | 16777216 | 0.221 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.246 | 16777216 | 1.512 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.291 | 16777216 | 1.277 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.297 | 16777216 | 1.253 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 2.080 | 25165824 | 0.268 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.373 | 25165824 | 1.496 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.412 | 25165824 | 1.355 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.415 | 25165824 | 1.343 | 0 |
| tanh_f64_n1048576 | jax-cuda | 32 | 15 | 1.694 | 16777216 | 0.220 | 4.437e-08 |
| tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.261 | 16777216 | 1.423 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.300 | 16777216 | 1.241 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.305 | 16777216 | 1.218 | 4.437e-08 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 1.865 | 26214400 | 0.312 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.442 | 26214400 | 1.314 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.425 | 26214400 | 1.367 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.431 | 26214400 | 1.347 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261126_062950/gpu_isolated_ops.yaml
timestamp: '2026-11-26T06:29:50+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: cc9867403ef65b20357ed79bc78ec5fc2864c4a9
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
  hostname: d73a3a75b493
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
  device_uuid: GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd
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
  - 0.09295
  - 0.09296
  - 0.09342
  - 0.091731
  - 0.09234
  - 0.09306
  - 0.09191
  - 0.09348
  - 0.092651
  - 0.091
  - 0.09348
  - 0.09138
  - 0.09302
  - 0.093511
  - 0.09229
  median_ms: 0.09295
  bytes_per_second: 1443977708445.401
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-26T06:44:18.932675+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd
  cuda_visible_devices: GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd
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
    data/results/nvidia-gpu/gpu/isolated_ops/20261126_062950/fixtures/manifest.json: 9342befcf71769632efb829705a2bf6a1ac7bd64a0ea94297a7cc47d6532ee2c
    target/release/benchmark_gpu_public_api: 3ff693b2ae7411a94a3e3337550156543b79bbe33a2720b00f7fb8181bdf1fd4
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:44:41Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:44:34Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:44:25Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:44:28Z'
  compile:
    case_wall_seconds: 14.597
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
| add_f64_n1048576 | tenferro-cuda-direct | add_f64_n1048576 untimed 164.6 ms is 2627% of the 6.3 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-direct | sub_f64_n1048576 untimed 87.5 ms is 1409% of the 6.2 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-direct | mul_f64_n1048576 untimed 95.9 ms is 1542% of the 6.2 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-direct | div_f64_n1048576 untimed 93.0 ms is 1500% of the 6.2 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-direct | rem_f64_n1048576 untimed 99.6 ms is 1690% of the 5.9 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-direct | pow_f64_n1048576 untimed 80.3 ms is 652% of the 12.3 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-direct | maximum_f64_n1048576 untimed 99.5 ms is 1699% of the 5.9 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-direct | minimum_f64_n1048576 untimed 97.1 ms is 1653% of the 5.9 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-direct | neg_f64_n1048576 untimed 86.0 ms is 2064% of the 4.2 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-direct | abs_f64_n1048576 untimed 75.9 ms is 1814% of the 4.2 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-direct | sign_f64_n1048576 untimed 72.7 ms is 1706% of the 4.3 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-direct | conj_c64_n1048576 untimed 106.9 ms is 1862% of the 5.7 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-direct | exp_f64_n1048576 untimed 89.4 ms is 1993% of the 4.5 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-direct | expm1_f64_n1048576 untimed 76.1 ms is 1549% of the 4.9 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-direct | log_f64_n1048576 untimed 77.7 ms is 1343% of the 5.8 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-direct | log1p_f64_n1048576 untimed 86.9 ms is 1487% of the 5.8 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-direct | erf_f64_n1048576 untimed 98.6 ms is 1431% of the 6.9 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-direct | sin_f64_n1048576 untimed 77.9 ms is 1520% of the 5.1 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-direct | cos_f64_n1048576 untimed 79.6 ms is 1500% of the 5.3 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-direct | tanh_f64_n1048576 untimed 80.5 ms is 1793% of the 4.5 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-direct | sqrt_f64_n1048576 untimed 74.9 ms is 1716% of the 4.4 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | rsqrt_f64_n1048576 untimed 86.3 ms is 1993% of the 4.3 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-direct | compare_f64_n1048576 untimed 58.6 ms is 207% of the 28.3 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-direct | where_select_f64_n1048576 untimed 88.1 ms is 1381% of the 6.4 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-direct | clamp_f64_n1048576 untimed 94.8 ms is 632% of the 15.0 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | reduce_sum_f64_n1048576 untimed 12.2 ms is 43% of the 28.2 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | reduce_prod_f64_n1048576 untimed 16.0 ms is 29% of the 56.1 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | reduce_max_f64_n1048576 untimed 15.9 ms is 26% of the 59.9 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | reduce_min_f64_n1048576 untimed 15.2 ms is 25% of the 59.6 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_f64_n1048576 untimed 14.9 ms is 27% of the 55.8 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | reduce_mean_f64_n1048576 untimed 10.5 ms is 27% of the 39.7 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | tenferro-cuda-direct | scale_real_f64_n1048576 untimed 71.5 ms is 1175% of the 6.1 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | tenferro-cuda-direct | scale_complex_c64_n1048576 untimed 109.9 ms is 1801% of the 6.1 ms timed region (budget 20%) |
| add_f64_n1048576 | tenferro-cuda-eager | add_f64_n1048576 untimed 160.2 ms is 2526% of the 6.3 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-eager | sub_f64_n1048576 untimed 81.7 ms is 1306% of the 6.3 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-eager | mul_f64_n1048576 untimed 74.0 ms is 1171% of the 6.3 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-eager | div_f64_n1048576 untimed 73.8 ms is 1169% of the 6.3 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-eager | rem_f64_n1048576 untimed 81.4 ms is 1361% of the 6.0 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-eager | pow_f64_n1048576 untimed 80.5 ms is 650% of the 12.4 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-eager | maximum_f64_n1048576 untimed 72.8 ms is 1226% of the 5.9 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-eager | minimum_f64_n1048576 untimed 80.5 ms is 1326% of the 6.1 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-eager | neg_f64_n1048576 untimed 74.5 ms is 1762% of the 4.2 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-eager | abs_f64_n1048576 untimed 71.8 ms is 1690% of the 4.2 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-eager | sign_f64_n1048576 untimed 69.8 ms is 1481% of the 4.7 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-eager | conj_c64_n1048576 untimed 112.6 ms is 1950% of the 5.8 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-eager | exp_f64_n1048576 untimed 82.6 ms is 1820% of the 4.5 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-eager | expm1_f64_n1048576 untimed 72.5 ms is 1456% of the 5.0 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-eager | log_f64_n1048576 untimed 71.9 ms is 1230% of the 5.8 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-eager | log1p_f64_n1048576 untimed 80.3 ms is 1373% of the 5.8 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-eager | erf_f64_n1048576 untimed 94.5 ms is 1391% of the 6.8 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-eager | sin_f64_n1048576 untimed 68.9 ms is 1325% of the 5.2 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-eager | cos_f64_n1048576 untimed 73.7 ms is 1402% of the 5.3 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-eager | tanh_f64_n1048576 untimed 77.3 ms is 1690% of the 4.6 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-eager | sqrt_f64_n1048576 untimed 72.1 ms is 1621% of the 4.4 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | rsqrt_f64_n1048576 untimed 87.0 ms is 1978% of the 4.4 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-eager | compare_f64_n1048576 untimed 53.0 ms is 184% of the 28.7 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-eager | where_select_f64_n1048576 untimed 76.1 ms is 1173% of the 6.5 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-eager | clamp_f64_n1048576 untimed 84.1 ms is 388% of the 21.7 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | reduce_sum_f64_n1048576 untimed 17.0 ms is 30% of the 56.7 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | reduce_prod_f64_n1048576 untimed 26.6 ms is 47% of the 56.3 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | reduce_max_f64_n1048576 untimed 12.2 ms is 40% of the 30.7 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | reduce_min_f64_n1048576 untimed 22.6 ms is 38% of the 59.9 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | reduce_sum_squares_f64_n1048576 untimed 15.4 ms is 27% of the 58.1 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | reduce_mean_f64_n1048576 untimed 16.6 ms is 36% of the 46.4 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | tenferro-cuda-eager | scale_real_f64_n1048576 untimed 85.2 ms is 1330% of the 6.4 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | scale_complex_c64_n1048576 untimed 152.7 ms is 2490% of the 6.1 ms timed region (budget 20%) |
| add_f64_n1048576 | pytorch-cuda | add_f64_n1048576 untimed 212.6 ms is 3696% of the 5.8 ms timed region (budget 20%) |
| sub_f64_n1048576 | pytorch-cuda | sub_f64_n1048576 untimed 74.8 ms is 1352% of the 5.5 ms timed region (budget 20%) |
| mul_f64_n1048576 | pytorch-cuda | mul_f64_n1048576 untimed 74.4 ms is 1302% of the 5.7 ms timed region (budget 20%) |
| div_f64_n1048576 | pytorch-cuda | div_f64_n1048576 untimed 81.3 ms is 1286% of the 6.3 ms timed region (budget 20%) |
| rem_f64_n1048576 | pytorch-cuda | rem_f64_n1048576 untimed 91.0 ms is 1600% of the 5.7 ms timed region (budget 20%) |
| pow_f64_n1048576 | pytorch-cuda | pow_f64_n1048576 untimed 103.5 ms is 737% of the 14.1 ms timed region (budget 20%) |
| maximum_f64_n1048576 | pytorch-cuda | maximum_f64_n1048576 untimed 84.1 ms is 1469% of the 5.7 ms timed region (budget 20%) |
| minimum_f64_n1048576 | pytorch-cuda | minimum_f64_n1048576 untimed 64.7 ms is 1154% of the 5.6 ms timed region (budget 20%) |
| neg_f64_n1048576 | pytorch-cuda | neg_f64_n1048576 untimed 71.7 ms is 1945% of the 3.7 ms timed region (budget 20%) |
| abs_f64_n1048576 | pytorch-cuda | abs_f64_n1048576 untimed 69.7 ms is 1413% of the 4.9 ms timed region (budget 20%) |
| sign_f64_n1048576 | pytorch-cuda | sign_f64_n1048576 untimed 63.5 ms is 1740% of the 3.7 ms timed region (budget 20%) |
| conj_c64_n1048576 | pytorch-cuda | conj_c64_n1048576 untimed 109.5 ms is 1897% of the 5.8 ms timed region (budget 20%) |
| exp_f64_n1048576 | pytorch-cuda | exp_f64_n1048576 untimed 86.1 ms is 2184% of the 3.9 ms timed region (budget 20%) |
| expm1_f64_n1048576 | pytorch-cuda | expm1_f64_n1048576 untimed 67.9 ms is 1427% of the 4.8 ms timed region (budget 20%) |
| log_f64_n1048576 | pytorch-cuda | log_f64_n1048576 untimed 84.3 ms is 1406% of the 6.0 ms timed region (budget 20%) |
| log1p_f64_n1048576 | pytorch-cuda | log1p_f64_n1048576 untimed 65.9 ms is 1049% of the 6.3 ms timed region (budget 20%) |
| erf_f64_n1048576 | pytorch-cuda | erf_f64_n1048576 untimed 74.8 ms is 1091% of the 6.9 ms timed region (budget 20%) |
| sin_f64_n1048576 | pytorch-cuda | sin_f64_n1048576 untimed 73.4 ms is 1393% of the 5.3 ms timed region (budget 20%) |
| cos_f64_n1048576 | pytorch-cuda | cos_f64_n1048576 untimed 77.9 ms is 1481% of the 5.3 ms timed region (budget 20%) |
| tanh_f64_n1048576 | pytorch-cuda | tanh_f64_n1048576 untimed 59.2 ms is 1509% of the 3.9 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | pytorch-cuda | sqrt_f64_n1048576 untimed 69.0 ms is 1866% of the 3.7 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | pytorch-cuda | rsqrt_f64_n1048576 untimed 67.4 ms is 1822% of the 3.7 ms timed region (budget 20%) |
| compare_f64_n1048576 | pytorch-cuda | compare_f64_n1048576 untimed 67.3 ms is 995% of the 6.8 ms timed region (budget 20%) |
| where_select_f64_n1048576 | pytorch-cuda | where_select_f64_n1048576 untimed 94.6 ms is 1428% of the 6.6 ms timed region (budget 20%) |
| clamp_f64_n1048576 | pytorch-cuda | clamp_f64_n1048576 untimed 59.6 ms is 1208% of the 4.9 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | pytorch-cuda | reduce_sum_f64_n1048576 untimed 29.5 ms is 53% of the 55.6 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | pytorch-cuda | reduce_prod_f64_n1048576 untimed 24.2 ms is 44% of the 55.2 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | pytorch-cuda | reduce_max_f64_n1048576 untimed 25.6 ms is 46% of the 55.9 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | pytorch-cuda | reduce_min_f64_n1048576 untimed 26.1 ms is 46% of the 56.3 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | pytorch-cuda | reduce_mean_f64_n1048576 untimed 28.9 ms is 51% of the 56.3 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | pytorch-cuda | scale_real_f64_n1048576 untimed 246.1 ms is 5390% of the 4.6 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | pytorch-cuda | scale_complex_c64_n1048576 untimed 358.8 ms is 6110% of the 5.9 ms timed region (budget 20%) |
| add_f64_n1048576 | jax-cuda | add_f64_n1048576 untimed 340.3 ms is 2187% of the 15.6 ms timed region (budget 20%) |
| sub_f64_n1048576 | jax-cuda | sub_f64_n1048576 untimed 125.6 ms is 397% of the 31.6 ms timed region (budget 20%) |
| mul_f64_n1048576 | jax-cuda | mul_f64_n1048576 untimed 118.6 ms is 393% of the 30.2 ms timed region (budget 20%) |
| div_f64_n1048576 | jax-cuda | div_f64_n1048576 untimed 93.7 ms is 327% of the 28.6 ms timed region (budget 20%) |
| rem_f64_n1048576 | jax-cuda | rem_f64_n1048576 untimed 169.9 ms is 618% of the 27.5 ms timed region (budget 20%) |
| pow_f64_n1048576 | jax-cuda | pow_f64_n1048576 untimed 133.1 ms is 437% of the 30.5 ms timed region (budget 20%) |
| maximum_f64_n1048576 | jax-cuda | maximum_f64_n1048576 untimed 110.7 ms is 398% of the 27.8 ms timed region (budget 20%) |
| minimum_f64_n1048576 | jax-cuda | minimum_f64_n1048576 untimed 117.6 ms is 420% of the 28.0 ms timed region (budget 20%) |
| neg_f64_n1048576 | jax-cuda | neg_f64_n1048576 untimed 102.6 ms is 391% of the 26.2 ms timed region (budget 20%) |
| abs_f64_n1048576 | jax-cuda | abs_f64_n1048576 untimed 106.4 ms is 434% of the 24.5 ms timed region (budget 20%) |
| sign_f64_n1048576 | jax-cuda | sign_f64_n1048576 untimed 129.2 ms is 497% of the 26.0 ms timed region (budget 20%) |
| conj_c64_n1048576 | jax-cuda | conj_c64_n1048576 untimed 159.9 ms is 1260% of the 12.7 ms timed region (budget 20%) |
| exp_f64_n1048576 | jax-cuda | exp_f64_n1048576 untimed 121.4 ms is 469% of the 25.9 ms timed region (budget 20%) |
| expm1_f64_n1048576 | jax-cuda | expm1_f64_n1048576 untimed 128.9 ms is 488% of the 26.4 ms timed region (budget 20%) |
| log_f64_n1048576 | jax-cuda | log_f64_n1048576 untimed 140.6 ms is 570% of the 24.7 ms timed region (budget 20%) |
| log1p_f64_n1048576 | jax-cuda | log1p_f64_n1048576 untimed 145.8 ms is 574% of the 25.4 ms timed region (budget 20%) |
| erf_f64_n1048576 | jax-cuda | erf_f64_n1048576 untimed 128.4 ms is 347% of the 37.0 ms timed region (budget 20%) |
| sin_f64_n1048576 | jax-cuda | sin_f64_n1048576 untimed 162.2 ms is 627% of the 25.9 ms timed region (budget 20%) |
| cos_f64_n1048576 | jax-cuda | cos_f64_n1048576 untimed 155.9 ms is 597% of the 26.1 ms timed region (budget 20%) |
| tanh_f64_n1048576 | jax-cuda | tanh_f64_n1048576 untimed 125.0 ms is 490% of the 25.5 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | jax-cuda | sqrt_f64_n1048576 untimed 118.5 ms is 465% of the 25.5 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | jax-cuda | rsqrt_f64_n1048576 untimed 113.5 ms is 312% of the 36.4 ms timed region (budget 20%) |
| compare_f64_n1048576 | jax-cuda | compare_f64_n1048576 untimed 89.3 ms is 171% of the 52.4 ms timed region (budget 20%) |
| where_select_f64_n1048576 | jax-cuda | where_select_f64_n1048576 untimed 113.4 ms is 397% of the 28.6 ms timed region (budget 20%) |
| clamp_f64_n1048576 | jax-cuda | clamp_f64_n1048576 untimed 120.7 ms is 477% of the 25.3 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | jax-cuda | reduce_sum_f64_n1048576 untimed 209.3 ms is 753% of the 27.8 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | jax-cuda | reduce_prod_f64_n1048576 untimed 217.3 ms is 409% of the 53.1 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | jax-cuda | reduce_max_f64_n1048576 untimed 304.1 ms is 591% of the 51.5 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | jax-cuda | reduce_min_f64_n1048576 untimed 106.8 ms is 385% of the 27.8 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | jax-cuda | reduce_mean_f64_n1048576 untimed 96.9 ms is 326% of the 29.7 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | jax-cuda | scale_real_f64_n1048576 untimed 134.4 ms is 453% of the 29.7 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | jax-cuda | scale_complex_c64_n1048576 untimed 151.1 ms is 1025% of the 14.7 ms timed region (budget 20%) |