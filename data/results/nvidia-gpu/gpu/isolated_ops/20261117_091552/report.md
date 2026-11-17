# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **124**. Unsupported cells: **8**, including **6** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1426.8 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261117_091552/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**13** of 33) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_f64_n1048576 † | 31.955 (7.785) | 91.187 (28.785) | 12.093 (0.109) | 63.472 (3.703) |
| sub_f64_n1048576 | 27.852 (4.196) | 93.088 (16.769) | 12.193 (2.183) | 66.078 (6.996) |
| mul_f64_n1048576 | 27.936 (4.216) | 103.235 (19.488) | 12.375 (2.386) | 65.595 (3.375) |
| div_f64_n1048576 | 31.137 (4.590) | 98.701 (18.643) | 13.224 (0.326) | 64.421 (2.342) |
| rem_f64_n1048576 † | 66.872 (8.010) | 136.567 (8.884) | 12.221 (2.503) | 63.607 (3.077) |
| pow_f64_n1048576 | 64.938 (7.186) | 127.348 (9.210) | 36.459 (1.446) | 69.065 (1.958) |
| maximum_f64_n1048576 | 50.025 (15.764) | 120.583 (15.043) | 12.077 (0.112) | 76.860 (9.464) |
| minimum_f64_n1048576 | 47.628 (6.858) | 116.421 (7.690) | 12.183 (2.083) | 69.874 (4.715) |
| neg_f64_n1048576 | 25.155 (1.821) | 56.459 (10.572) | 10.513 (3.388) | 66.616 (1.623) |
| abs_f64_n1048576 | 25.221 (3.883) | 57.977 (3.258) | 9.720 (0.840) | 62.576 (5.812) |
| sign_f64_n1048576 | 25.308 (2.785) | 63.896 (10.561) | 9.562 (0.031) | 56.811 (3.526) |
| conj_c64_n1048576 | 39.847 (10.239) | 83.534 (15.329) | 25.254 (0.313) | 65.336 (61.342) |
| exp_f64_n1048576 | 26.884 (7.393) | 60.976 (11.052) | 12.226 (3.108) | 51.432 (4.510) |
| expm1_f64_n1048576 | 27.455 (8.902) | 63.674 (7.829) | 12.084 (2.515) | 55.350 (4.844) |
| log_f64_n1048576 | 30.065 (7.932) | 66.182 (2.934) | 15.411 (0.123) | 53.724 (2.627) |
| log1p_f64_n1048576 | 33.786 (7.744) | 69.513 (10.973) | 16.203 (0.831) | 57.657 (1.080) |
| erf_f64_n1048576 | 36.337 (3.390) | 71.293 (13.931) | 17.757 (1.886) | 79.066 (3.015) |
| sin_f64_n1048576 | 31.927 (8.990) | 65.313 (12.657) | 13.838 (2.235) | 57.252 (5.656) |
| cos_f64_n1048576 | 28.105 (0.237) | 65.418 (9.158) | 13.774 (0.062) | 58.131 (2.783) |
| tanh_f64_n1048576 † | 29.898 (3.235) | 60.979 (12.452) | 10.013 (0.035) | 63.230 (10.412) |
| sqrt_f64_n1048576 | 25.573 (4.607) | 62.185 (7.417) | 9.636 (0.086) | 61.123 (2.209) |
| rsqrt_f64_n1048576 † | 26.128 (1.767) | 61.920 (5.739) | 9.721 (0.647) | 96.380 (4.659) |
| compare_f64_n1048576 † | 26.393 (2.363) | 104.692 (17.042) | 8.734 (0.271) | 64.023 (2.201) |
| where_select_f64_n1048576 † | 30.227 (5.563) | 138.661 (14.798) | 14.286 (1.299) | 68.970 (7.459) |
| clamp_f64_n1048576 † | 89.577 (22.855) | 179.424 (24.306) | 12.262 (1.225) | 61.776 (1.305) |
| reduce_sum_f64_n1048576 † | 35.709 (2.626) | 70.374 (5.085) | 8.882 (0.204) | 64.361 (1.539) |
| reduce_prod_f64_n1048576 † | 36.224 (2.258) | 68.751 (7.552) | 8.909 (0.022) | 64.587 (3.197) |
| reduce_max_f64_n1048576 † | 38.149 (4.711) | 69.471 (12.433) | 8.462 (0.065) | 58.241 (1.907) |
| reduce_min_f64_n1048576 † | 39.090 (3.786) | 70.625 (2.280) | 8.478 (0.074) | 63.159 (1.173) |
| reduce_sum_squares_f64_n1048576 † | 38.115 (2.367) | 68.000 (5.155) | unsupported | unsupported |
| reduce_mean_f64_n1048576 † | unsupported | unsupported | 8.214 (0.095) | 66.137 (7.391) |
| scale_real_f64_n1048576 | unsupported | unsupported | 9.474 (0.091) | 70.908 (1.998) |
| scale_complex_c64_n1048576 | unsupported | unsupported | 24.966 (0.800) | 66.858 (2.678) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 2.002 | 16777216 | 0.188 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.311 | 16777216 | 1.210 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.807 | 16777216 | 0.466 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.855 | 16777216 | 0.203 | 0 |
| add_f64_n1048576 | jax-cuda | 32 | 15 | 2.031 | 25165824 | 0.278 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.387 | 25165824 | 1.459 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.023 | 25165824 | 0.552 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.459 | 25165824 | 0.193 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 1.977 | 16777232 | 0.190 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.392 | 16777232 | 0.959 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.866 | 16777232 | 0.131 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.871 | 16777232 | 0.066 | 0 |
| compare_f64_n1048576 | jax-cuda | 32 | 15 | 2.049 | 17825792 | 0.195 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.559 | 17825792 | 1.430 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 3.378 | 17825792 | 0.473 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.350 | 17825792 | 0.119 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 1.045 | 33554432 | 0.360 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.404 | 33554432 | 0.931 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.638 | 33554432 | 0.590 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.337 | 33554432 | 0.282 | 0 |
| cos_f64_n1048576 | jax-cuda | 32 | 15 | 1.860 | 16777216 | 0.202 | 5.905e-08 |
| cos_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.441 | 16777216 | 0.854 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.899 | 16777216 | 0.418 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.093 | 16777216 | 0.180 | 5.905e-08 |
| div_f64_n1048576 | jax-cuda | 32 | 15 | 2.061 | 25165824 | 0.274 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.423 | 25165824 | 1.334 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.996 | 25165824 | 0.566 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.158 | 25165824 | 0.179 | 0 |
| erf_f64_n1048576 | jax-cuda | 32 | 15 | 2.530 | 16777216 | 0.149 | 8.536e-08 |
| erf_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.568 | 16777216 | 0.662 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.163 | 16777216 | 0.324 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.281 | 16777216 | 0.165 | 8.536e-08 |
| exp_f64_n1048576 | jax-cuda | 32 | 15 | 1.646 | 16777216 | 0.229 | 1.095e-07 |
| exp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.391 | 16777216 | 0.962 | 1.095e-07 |
| exp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.860 | 16777216 | 0.437 | 1.095e-07 |
| exp_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.951 | 16777216 | 0.193 | 1.095e-07 |
| expm1_f64_n1048576 | jax-cuda | 32 | 15 | 1.771 | 16777216 | 0.212 | 7.39e-08 |
| expm1_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.387 | 16777216 | 0.973 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.879 | 16777216 | 0.428 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.038 | 16777216 | 0.185 | 7.39e-08 |
| log1p_f64_n1048576 | jax-cuda | 32 | 15 | 1.845 | 16777216 | 0.204 | 1.11e-07 |
| log1p_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.518 | 16777216 | 0.726 | 1.11e-07 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.081 | 16777216 | 0.348 | 1.11e-07 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.224 | 16777216 | 0.169 | 1.11e-07 |
| log_f64_n1048576 | jax-cuda | 32 | 15 | 1.719 | 16777216 | 0.219 | 4.308e-08 |
| log_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.493 | 16777216 | 0.763 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.962 | 16777216 | 0.391 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.118 | 16777216 | 0.178 | 4.308e-08 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 2.460 | 25165824 | 0.229 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.386 | 25165824 | 1.460 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.601 | 25165824 | 0.353 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.859 | 25165824 | 0.146 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 2.236 | 25165824 | 0.252 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.390 | 25165824 | 1.448 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.524 | 25165824 | 0.370 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.725 | 25165824 | 0.152 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 2.099 | 25165824 | 0.269 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.396 | 25165824 | 1.425 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.894 | 25165824 | 0.631 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.304 | 25165824 | 0.171 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 2.132 | 16777216 | 0.177 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.336 | 16777216 | 1.118 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.805 | 16777216 | 0.467 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.807 | 16777216 | 0.208 | 0 |
| pow_f64_n1048576 | jax-cuda | 32 | 15 | 2.210 | 25165824 | 0.255 | 1.48e-07 |
| pow_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.167 | 25165824 | 0.484 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.078 | 25165824 | 0.272 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 4.075 | 25165824 | 0.139 | 1.48e-07 |
| reduce_max_f64_n1048576 | jax-cuda | 32 | 15 | 1.864 | 8396800 | 0.101 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.166 | 8396800 | 0.695 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.442 | 8396800 | 0.154 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.223 | 8396800 | 0.085 | 0 |
| reduce_mean_f64_n1048576 | jax-cuda | 32 | 15 | 2.116 | 8396800 | 0.089 | 6.5e-09 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.103 | 8396800 | 0.716 | 3.9e-09 |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_min_f64_n1048576 | jax-cuda | 32 | 15 | 2.021 | 8396800 | 0.093 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.170 | 8396800 | 0.694 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.502 | 8396800 | 0.151 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.260 | 8396800 | 0.083 | 0 |
| reduce_prod_f64_n1048576 | jax-cuda | 32 | 15 | 2.067 | 8396800 | 0.091 | 4.436e-06 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.281 | 8396800 | 0.661 | 4.216e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.318 | 8396800 | 0.162 | 4.438e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.200 | 8396800 | 0.086 | 4.438e-06 |
| reduce_sum_f64_n1048576 | jax-cuda | 32 | 15 | 2.060 | 8396800 | 0.091 | 4.449e-06 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.274 | 8396800 | 0.663 | 2.961e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.285 | 8396800 | 0.165 | 3.305e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.252 | 8396800 | 0.084 | 3.305e-06 |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.439 | 8396800 | 0.154 | 4.939e-07 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.176 | 8396800 | 0.087 | 4.939e-07 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 2.035 | 25165824 | 0.277 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.391 | 25165824 | 1.443 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.140 | 25165824 | 0.264 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.185 | 25165824 | 0.129 | 0 |
| rsqrt_f64_n1048576 | jax-cuda | 4 | 15 | 0.386 | 16777216 | 0.122 | 6.504e-08 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.311 | 16777216 | 1.210 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.836 | 16777216 | 0.450 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.981 | 16777216 | 0.190 | 6.504e-08 |
| scale_complex_c64_n1048576 | jax-cuda | 16 | 15 | 1.070 | 33554448 | 0.352 | 4.129e-08 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.399 | 33554448 | 0.942 | 0 |
| scale_complex_c64_n1048576 | tenferro-cuda-direct | — | — | — | 33554448 | unavailable | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554448 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_f64_n1048576 | jax-cuda | 32 | 15 | 2.269 | 16777224 | 0.166 | 0 |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.303 | 16777224 | 1.241 | 0 |
| scale_real_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777224 | unavailable | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_real_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777224 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 1.818 | 16777216 | 0.207 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.306 | 16777216 | 1.230 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.810 | 16777216 | 0.465 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.045 | 16777216 | 0.184 | 0 |
| sin_f64_n1048576 | jax-cuda | 32 | 15 | 1.832 | 16777216 | 0.205 | 4.153e-08 |
| sin_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.443 | 16777216 | 0.850 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.022 | 16777216 | 0.368 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.090 | 16777216 | 0.180 | 4.153e-08 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 1.956 | 16777216 | 0.192 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.308 | 16777216 | 1.220 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.818 | 16777216 | 0.460 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.990 | 16777216 | 0.189 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 2.114 | 25165824 | 0.267 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.390 | 25165824 | 1.447 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.891 | 25165824 | 0.633 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.979 | 25165824 | 0.189 | 0 |
| tanh_f64_n1048576 | jax-cuda | 16 | 15 | 1.012 | 16777216 | 0.186 | 4.437e-08 |
| tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.320 | 16777216 | 1.174 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.957 | 16777216 | 0.393 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.951 | 16777216 | 0.193 | 4.437e-08 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 2.207 | 26214400 | 0.266 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.457 | 26214400 | 1.286 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.967 | 26214400 | 0.608 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.219 | 26214400 | 0.133 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261117_091552/gpu_isolated_ops.yaml
timestamp: '2026-11-17T09:15:52+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: ce9125d0cff4ee40841120759b5df45252c27f22
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 2e63ea288fb120038b7a88288485bf2cb7e76866
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
  cpu: AMD EPYC 7713 64-Core Processor
  logical_cpus: 128
  power:
    cpufreq_boost: '1'
    governor: schedutil
    scaling_max_freq_khz: '2000000'
    smt:
      active: '0'
      control: notsupported
  numa:
    nodes: 2
    cpulists:
      '0': 0-63
      '1': 64-127
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
  l3_domains:
  - 0-7
  - 8-15
  - 16-23
  - 24-31
  - 32-39
  - 40-47
  - 48-55
  - 56-63
  - 64-71
  - 72-79
  - 80-87
  - 88-95
  - 96-103
  - 104-111
  - 112-119
  - 120-127
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: b8e17af19547
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
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
    - 0-7
    - 8-15
    - 16-23
    - 24-31
    - 32-39
    - 40-47
    - 48-55
    - 56-63
    - 64-71
    - 72-79
    - 80-87
    - 88-95
    - 96-103
    - 104-111
    - 112-119
    - 120-127
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
    GPU_RUNTIME_SHA256: f876805d95374305db5787a72cee40a6848d0fc64ee71d7b306c264971c36869
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
  device_name: NVIDIA A100-SXM4-80GB
  device_uuid: GPU-704abe6c-29a2-539e-7ccd-01f9de4eed3b
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
  - 0.11819
  - 0.09682
  - 0.09527
  - 0.091259
  - 0.0933
  - 0.09448
  - 0.09514
  - 0.09167
  - 0.09503
  - 0.094499
  - 0.09345
  - 0.09282
  - 0.09372
  - 0.09348
  - 0.09407
  median_ms: 0.09407
  bytes_per_second: 1426785670245.5618
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-17T09:38:20.901987+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-704abe6c-29a2-539e-7ccd-01f9de4eed3b
  cuda_visible_devices: GPU-704abe6c-29a2-539e-7ccd-01f9de4eed3b
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
    data/results/nvidia-gpu/gpu/isolated_ops/20261117_091552/fixtures/manifest.json: 3c1a75f9c57996b228c04da1d4cbf34e4137a074e44ddca9f1c37e5824580bd8
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
        finished_at: '2026-11-17T09:39:04Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-17T09:38:41Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-17T09:38:30Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-17T09:38:35Z'
  compile:
    case_wall_seconds: 30.722
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
| add_f64_n1048576 | tenferro-cuda-direct | add_f64_n1048576 untimed 156.8 ms is 1030% of the 15.2 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-direct | sub_f64_n1048576 untimed 91.5 ms is 622% of the 14.7 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-direct | mul_f64_n1048576 untimed 111.2 ms is 751% of the 14.8 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-direct | div_f64_n1048576 untimed 108.0 ms is 571% of the 18.9 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-direct | rem_f64_n1048576 untimed 112.8 ms is 344% of the 32.8 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-direct | pow_f64_n1048576 untimed 102.2 ms is 303% of the 33.7 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-direct | maximum_f64_n1048576 untimed 104.3 ms is 412% of the 25.3 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-direct | minimum_f64_n1048576 untimed 104.8 ms is 439% of the 23.9 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-direct | neg_f64_n1048576 untimed 97.6 ms is 763% of the 12.8 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-direct | abs_f64_n1048576 untimed 82.3 ms is 632% of the 13.0 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-direct | sign_f64_n1048576 untimed 80.0 ms is 601% of the 13.3 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-direct | conj_c64_n1048576 untimed 116.5 ms is 1050% of the 11.1 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-direct | exp_f64_n1048576 untimed 95.5 ms is 667% of the 14.3 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-direct | expm1_f64_n1048576 untimed 86.5 ms is 556% of the 15.6 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-direct | log_f64_n1048576 untimed 81.5 ms is 504% of the 16.2 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-direct | log1p_f64_n1048576 untimed 96.3 ms is 588% of the 16.4 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-direct | erf_f64_n1048576 untimed 106.0 ms is 589% of the 18.0 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-direct | sin_f64_n1048576 untimed 80.9 ms is 493% of the 16.4 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-direct | cos_f64_n1048576 untimed 80.3 ms is 556% of the 14.4 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-direct | tanh_f64_n1048576 untimed 90.7 ms is 624% of the 14.5 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-direct | sqrt_f64_n1048576 untimed 89.8 ms is 656% of the 13.7 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | rsqrt_f64_n1048576 untimed 106.9 ms is 783% of the 13.6 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-direct | compare_f64_n1048576 untimed 60.7 ms is 112% of the 54.0 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-direct | where_select_f64_n1048576 untimed 98.5 ms is 577% of the 17.1 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-direct | clamp_f64_n1048576 untimed 111.1 ms is 239% of the 46.5 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | reduce_sum_f64_n1048576 untimed 10.0 ms is 28% of the 35.6 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | reduce_prod_f64_n1048576 untimed 9.8 ms is 28% of the 35.5 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | reduce_max_f64_n1048576 untimed 9.7 ms is 26% of the 37.6 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | reduce_min_f64_n1048576 untimed 9.4 ms is 24% of the 38.6 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_f64_n1048576 untimed 8.3 ms is 21% of the 38.9 ms timed region (budget 20%) |
| add_f64_n1048576 | tenferro-cuda-eager | add_f64_n1048576 untimed 172.5 ms is 701% of the 24.6 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-eager | sub_f64_n1048576 untimed 107.6 ms is 219% of the 49.2 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-eager | mul_f64_n1048576 untimed 91.5 ms is 170% of the 53.8 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-eager | div_f64_n1048576 untimed 113.8 ms is 215% of the 53.0 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-eager | rem_f64_n1048576 untimed 118.0 ms is 349% of the 33.8 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-eager | pow_f64_n1048576 untimed 94.3 ms is 151% of the 62.3 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-eager | maximum_f64_n1048576 untimed 89.1 ms is 151% of the 58.9 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-eager | minimum_f64_n1048576 untimed 90.1 ms is 161% of the 55.9 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-eager | neg_f64_n1048576 untimed 81.0 ms is 273% of the 29.6 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-eager | abs_f64_n1048576 untimed 85.5 ms is 297% of the 28.8 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-eager | sign_f64_n1048576 untimed 88.8 ms is 280% of the 31.7 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-eager | conj_c64_n1048576 untimed 113.1 ms is 543% of the 20.8 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-eager | exp_f64_n1048576 untimed 90.8 ms is 272% of the 33.4 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-eager | expm1_f64_n1048576 untimed 87.4 ms is 272% of the 32.1 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-eager | log_f64_n1048576 untimed 88.7 ms is 269% of the 33.0 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-eager | log1p_f64_n1048576 untimed 97.3 ms is 261% of the 37.3 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-eager | erf_f64_n1048576 untimed 109.3 ms is 296% of the 37.0 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-eager | sin_f64_n1048576 untimed 86.8 ms is 257% of the 33.7 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-eager | cos_f64_n1048576 untimed 88.1 ms is 272% of the 32.4 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-eager | tanh_f64_n1048576 untimed 92.8 ms is 280% of the 33.1 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-eager | sqrt_f64_n1048576 untimed 86.7 ms is 280% of the 31.0 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | rsqrt_f64_n1048576 untimed 101.8 ms is 330% of the 30.9 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-eager | compare_f64_n1048576 untimed 59.1 ms is 116% of the 51.0 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-eager | where_select_f64_n1048576 untimed 101.1 ms is 303% of the 33.4 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-eager | clamp_f64_n1048576 untimed 78.2 ms is 172% of the 45.4 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | reduce_sum_f64_n1048576 untimed 10.4 ms is 30% of the 34.8 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | reduce_prod_f64_n1048576 untimed 8.6 ms is 25% of the 34.1 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | reduce_max_f64_n1048576 untimed 8.8 ms is 25% of the 34.8 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | reduce_min_f64_n1048576 untimed 8.9 ms is 26% of the 34.6 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | reduce_sum_squares_f64_n1048576 untimed 8.6 ms is 25% of the 34.6 ms timed region (budget 20%) |
| add_f64_n1048576 | pytorch-cuda | add_f64_n1048576 untimed 244.1 ms is 4158% of the 5.9 ms timed region (budget 20%) |
| sub_f64_n1048576 | pytorch-cuda | sub_f64_n1048576 untimed 102.0 ms is 1604% of the 6.4 ms timed region (budget 20%) |
| mul_f64_n1048576 | pytorch-cuda | mul_f64_n1048576 untimed 227.8 ms is 3561% of the 6.4 ms timed region (budget 20%) |
| div_f64_n1048576 | pytorch-cuda | div_f64_n1048576 untimed 103.2 ms is 1603% of the 6.4 ms timed region (budget 20%) |
| rem_f64_n1048576 | pytorch-cuda | rem_f64_n1048576 untimed 119.0 ms is 1725% of the 6.9 ms timed region (budget 20%) |
| pow_f64_n1048576 | pytorch-cuda | pow_f64_n1048576 untimed 149.0 ms is 810% of the 18.4 ms timed region (budget 20%) |
| maximum_f64_n1048576 | pytorch-cuda | maximum_f64_n1048576 untimed 106.6 ms is 1760% of the 6.1 ms timed region (budget 20%) |
| minimum_f64_n1048576 | pytorch-cuda | minimum_f64_n1048576 untimed 84.2 ms is 1317% of the 6.4 ms timed region (budget 20%) |
| neg_f64_n1048576 | pytorch-cuda | neg_f64_n1048576 untimed 85.9 ms is 1433% of the 6.0 ms timed region (budget 20%) |
| abs_f64_n1048576 | pytorch-cuda | abs_f64_n1048576 untimed 87.1 ms is 1607% of the 5.4 ms timed region (budget 20%) |
| sign_f64_n1048576 | pytorch-cuda | sign_f64_n1048576 untimed 89.9 ms is 1913% of the 4.7 ms timed region (budget 20%) |
| conj_c64_n1048576 | pytorch-cuda | conj_c64_n1048576 untimed 123.1 ms is 2022% of the 6.1 ms timed region (budget 20%) |
| exp_f64_n1048576 | pytorch-cuda | exp_f64_n1048576 untimed 117.9 ms is 2070% of the 5.7 ms timed region (budget 20%) |
| expm1_f64_n1048576 | pytorch-cuda | expm1_f64_n1048576 untimed 89.9 ms is 1393% of the 6.5 ms timed region (budget 20%) |
| log_f64_n1048576 | pytorch-cuda | log_f64_n1048576 untimed 94.2 ms is 1216% of the 7.8 ms timed region (budget 20%) |
| log1p_f64_n1048576 | pytorch-cuda | log1p_f64_n1048576 untimed 88.0 ms is 1091% of the 8.1 ms timed region (budget 20%) |
| erf_f64_n1048576 | pytorch-cuda | erf_f64_n1048576 untimed 114.9 ms is 965% of the 11.9 ms timed region (budget 20%) |
| sin_f64_n1048576 | pytorch-cuda | sin_f64_n1048576 untimed 105.7 ms is 1471% of the 7.2 ms timed region (budget 20%) |
| cos_f64_n1048576 | pytorch-cuda | cos_f64_n1048576 untimed 111.9 ms is 1691% of the 6.6 ms timed region (budget 20%) |
| tanh_f64_n1048576 | pytorch-cuda | tanh_f64_n1048576 untimed 173.5 ms is 3547% of the 4.9 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | pytorch-cuda | sqrt_f64_n1048576 untimed 91.9 ms is 1831% of the 5.0 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | pytorch-cuda | rsqrt_f64_n1048576 untimed 84.9 ms is 1701% of the 5.0 ms timed region (budget 20%) |
| compare_f64_n1048576 | pytorch-cuda | compare_f64_n1048576 untimed 91.6 ms is 956% of the 9.6 ms timed region (budget 20%) |
| where_select_f64_n1048576 | pytorch-cuda | where_select_f64_n1048576 untimed 106.0 ms is 864% of the 12.3 ms timed region (budget 20%) |
| clamp_f64_n1048576 | pytorch-cuda | clamp_f64_n1048576 untimed 72.9 ms is 1157% of the 6.3 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | pytorch-cuda | reduce_sum_f64_n1048576 untimed 32.7 ms is 95% of the 34.3 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | pytorch-cuda | reduce_prod_f64_n1048576 untimed 23.2 ms is 68% of the 34.3 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | pytorch-cuda | reduce_max_f64_n1048576 untimed 27.6 ms is 84% of the 32.7 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | pytorch-cuda | reduce_min_f64_n1048576 untimed 25.8 ms is 79% of the 32.7 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | pytorch-cuda | reduce_mean_f64_n1048576 untimed 26.9 ms is 80% of the 33.5 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | pytorch-cuda | scale_real_f64_n1048576 untimed 77.5 ms is 1658% of the 4.7 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | pytorch-cuda | scale_complex_c64_n1048576 untimed 129.9 ms is 2133% of the 6.1 ms timed region (budget 20%) |
| add_f64_n1048576 | jax-cuda | add_f64_n1048576 untimed 462.3 ms is 1240% of the 37.3 ms timed region (budget 20%) |
| sub_f64_n1048576 | jax-cuda | sub_f64_n1048576 untimed 358.0 ms is 1119% of the 32.0 ms timed region (budget 20%) |
| mul_f64_n1048576 | jax-cuda | mul_f64_n1048576 untimed 155.7 ms is 497% of the 31.3 ms timed region (budget 20%) |
| div_f64_n1048576 | jax-cuda | div_f64_n1048576 untimed 339.7 ms is 1097% of the 31.0 ms timed region (budget 20%) |
| rem_f64_n1048576 | jax-cuda | rem_f64_n1048576 untimed 398.4 ms is 1053% of the 37.8 ms timed region (budget 20%) |
| pow_f64_n1048576 | jax-cuda | pow_f64_n1048576 untimed 158.3 ms is 442% of the 35.8 ms timed region (budget 20%) |
| maximum_f64_n1048576 | jax-cuda | maximum_f64_n1048576 untimed 312.2 ms is 836% of the 37.3 ms timed region (budget 20%) |
| minimum_f64_n1048576 | jax-cuda | minimum_f64_n1048576 untimed 159.9 ms is 465% of the 34.4 ms timed region (budget 20%) |
| neg_f64_n1048576 | jax-cuda | neg_f64_n1048576 untimed 277.5 ms is 870% of the 31.9 ms timed region (budget 20%) |
| abs_f64_n1048576 | jax-cuda | abs_f64_n1048576 untimed 329.2 ms is 942% of the 34.9 ms timed region (budget 20%) |
| sign_f64_n1048576 | jax-cuda | sign_f64_n1048576 untimed 575.1 ms is 2124% of the 27.1 ms timed region (budget 20%) |
| conj_c64_n1048576 | jax-cuda | conj_c64_n1048576 untimed 542.2 ms is 1976% of the 27.4 ms timed region (budget 20%) |
| exp_f64_n1048576 | jax-cuda | exp_f64_n1048576 untimed 144.8 ms is 561% of the 25.8 ms timed region (budget 20%) |
| expm1_f64_n1048576 | jax-cuda | expm1_f64_n1048576 untimed 650.7 ms is 2437% of the 26.7 ms timed region (budget 20%) |
| log_f64_n1048576 | jax-cuda | log_f64_n1048576 untimed 263.1 ms is 1019% of the 25.8 ms timed region (budget 20%) |
| log1p_f64_n1048576 | jax-cuda | log1p_f64_n1048576 untimed 155.3 ms is 564% of the 27.5 ms timed region (budget 20%) |
| erf_f64_n1048576 | jax-cuda | erf_f64_n1048576 untimed 159.5 ms is 413% of the 38.6 ms timed region (budget 20%) |
| sin_f64_n1048576 | jax-cuda | sin_f64_n1048576 untimed 465.0 ms is 1703% of the 27.3 ms timed region (budget 20%) |
| cos_f64_n1048576 | jax-cuda | cos_f64_n1048576 untimed 176.1 ms is 629% of the 28.0 ms timed region (budget 20%) |
| tanh_f64_n1048576 | jax-cuda | tanh_f64_n1048576 untimed 125.8 ms is 637% of the 19.8 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | jax-cuda | sqrt_f64_n1048576 untimed 135.6 ms is 458% of the 29.6 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | jax-cuda | rsqrt_f64_n1048576 untimed 139.8 ms is 2317% of the 6.0 ms timed region (budget 20%) |
| compare_f64_n1048576 | jax-cuda | compare_f64_n1048576 untimed 111.1 ms is 323% of the 34.4 ms timed region (budget 20%) |
| where_select_f64_n1048576 | jax-cuda | where_select_f64_n1048576 untimed 124.7 ms is 331% of the 37.7 ms timed region (budget 20%) |
| clamp_f64_n1048576 | jax-cuda | clamp_f64_n1048576 untimed 676.8 ms is 2269% of the 29.8 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | jax-cuda | reduce_sum_f64_n1048576 untimed 187.2 ms is 601% of the 31.2 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | jax-cuda | reduce_prod_f64_n1048576 untimed 192.9 ms is 623% of the 31.0 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | jax-cuda | reduce_max_f64_n1048576 untimed 287.8 ms is 1024% of the 28.1 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | jax-cuda | reduce_min_f64_n1048576 untimed 90.1 ms is 296% of the 30.4 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | jax-cuda | reduce_mean_f64_n1048576 untimed 94.1 ms is 274% of the 34.4 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | jax-cuda | scale_real_f64_n1048576 untimed 5755.2 ms is 16986% of the 33.9 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | jax-cuda | scale_complex_c64_n1048576 untimed 5159.7 ms is 32137% of the 16.1 ms timed region (budget 20%) |