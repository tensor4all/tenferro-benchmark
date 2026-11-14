# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **124**. Unsupported cells: **8**, including **6** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **674.8 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261114_082852/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**31** of 33) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| add_f64_n1048576 † | 49.273 (5.491) | 161.525 (17.984) | 17.321 (1.624) | 309.118 (61.495) |
| sub_f64_n1048576 † | 57.034 (10.860) | 163.865 (42.621) | 13.479 (0.690) | 335.308 (116.914) |
| mul_f64_n1048576 † | 61.827 (96.106) | 202.086 (39.565) | 14.317 (2.968) | 246.540 (27.570) |
| div_f64_n1048576 † | 119.271 (130.542) | 288.445 (98.756) | 14.640 (0.345) | 139.909 (58.833) |
| rem_f64_n1048576 † | 76.354 (4.675) | 222.270 (216.421) | 17.408 (5.138) | 264.105 (54.738) |
| pow_f64_n1048576 † | 76.118 (8.582) | 211.148 (47.885) | 30.704 (0.716) | 1729.828 (629.829) |
| maximum_f64_n1048576 † | 61.242 (3.217) | 178.375 (25.056) | 14.510 (3.195) | 124.492 (11.332) |
| minimum_f64_n1048576 † | 69.064 (3.623) | 179.475 (48.252) | 13.706 (0.384) | 172.595 (29.648) |
| neg_f64_n1048576 † | 44.506 (8.638) | 89.685 (15.199) | 9.271 (0.738) | 217.852 (115.683) |
| abs_f64_n1048576 † | 40.735 (3.915) | 87.048 (2.981) | 9.804 (4.754) | 270.146 (99.943) |
| sign_f64_n1048576 † | 42.446 (5.963) | 84.133 (11.519) | 9.053 (2.584) | 957.663 (229.778) |
| conj_c64_n1048576 | 69.544 (13.130) | 146.003 (18.517) | 44.541 (37.836) | 252.895 (62.028) |
| exp_f64_n1048576 † | 58.911 (12.742) | 174.516 (47.152) | 19.119 (12.576) | 151.503 (35.960) |
| expm1_f64_n1048576 † | 72.487 (57.209) | 120.825 (17.037) | 10.869 (1.213) | 165.227 (48.627) |
| log_f64_n1048576 † | 103.610 (33.052) | 101.383 (6.072) | 14.003 (0.943) | 214.203 (43.328) |
| log1p_f64_n1048576 † | 60.823 (19.027) | 123.584 (10.389) | 13.951 (0.292) | 185.394 (53.018) |
| erf_f64_n1048576 † | 49.238 (4.640) | 156.130 (58.317) | 15.505 (0.886) | 162.626 (37.260) |
| sin_f64_n1048576 † | 47.867 (5.526) | 202.802 (69.309) | 12.493 (1.432) | 282.578 (105.460) |
| cos_f64_n1048576 † | 468.774 (722.858) | 146.876 (101.946) | 12.283 (1.628) | 206.032 (64.027) |
| tanh_f64_n1048576 † | 46.567 (11.607) | 107.695 (38.203) | 9.734 (4.422) | 152.271 (68.497) |
| sqrt_f64_n1048576 † | 39.888 (3.579) | 90.628 (12.144) | 8.861 (0.511) | 407.524 (295.848) |
| rsqrt_f64_n1048576 † | 39.539 (5.888) | 101.514 (14.475) | 8.774 (0.729) | 247.378 (40.735) |
| compare_f64_n1048576 † | 46.130 (7.124) | 139.246 (5.299) | 22.297 (12.947) | 238.089 (26.138) |
| where_select_f64_n1048576 † | 85.206 (84.159) | 301.889 (94.286) | 15.610 (1.960) | 127.635 (37.959) |
| clamp_f64_n1048576 † | 170.448 (22.175) | 309.083 (26.906) | 13.078 (4.701) | 149.775 (38.744) |
| reduce_sum_f64_n1048576 † | 114.260 (95.263) | 133.650 (29.534) | 7.093 (0.163) | 1530.362 (821.054) |
| reduce_prod_f64_n1048576 † | 64.725 (4.535) | 121.489 (12.763) | 7.197 (0.055) | 539.593 (249.753) |
| reduce_max_f64_n1048576 † | 78.536 (20.092) | 107.294 (4.308) | 7.358 (0.268) | 219.162 (176.838) |
| reduce_min_f64_n1048576 † | 69.317 (15.474) | 111.336 (13.370) | 7.428 (0.128) | 974.985 (295.987) |
| reduce_sum_squares_f64_n1048576 † | 118.591 (77.349) | 106.560 (3.396) | unsupported | unsupported |
| reduce_mean_f64_n1048576 † | unsupported | unsupported | 7.189 (1.833) | 377.387 (127.583) |
| scale_real_f64_n1048576 † | unsupported | unsupported | 11.017 (1.702) | 809.321 (1005.574) |
| scale_complex_c64_n1048576 | unsupported | unsupported | 57.479 (69.684) | 121.140 (5.360) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_f64_n1048576 | jax-cuda | 8 | 15 | 2.161 | 16777216 | 0.092 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.314 | 16777216 | 2.536 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.304 | 16777216 | 0.610 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.786 | 16777216 | 0.286 | 0 |
| add_f64_n1048576 | jax-cuda | 4 | 15 | 1.236 | 25165824 | 0.121 | 0 |
| add_f64_n1048576 | pytorch-cuda | 8 | 15 | 0.139 | 25165824 | 2.153 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.788 | 25165824 | 0.757 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.584 | 25165824 | 0.231 | 0 |
| clamp_f64_n1048576 | jax-cuda | 16 | 15 | 2.396 | 16777232 | 0.166 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.418 | 16777232 | 1.901 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.727 | 16777232 | 0.146 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.473 | 16777232 | 0.080 | 0 |
| compare_f64_n1048576 | jax-cuda | 8 | 15 | 1.905 | 17825792 | 0.111 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.713 | 17825792 | 1.185 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.952 | 17825792 | 0.573 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.228 | 17825792 | 0.190 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 4.046 | 33554432 | 0.197 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.713 | 33554432 | 1.116 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.113 | 33554432 | 0.715 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.336 | 33554432 | 0.341 | 0 |
| cos_f64_n1048576 | jax-cuda | 8 | 15 | 1.648 | 16777216 | 0.121 | 5.905e-08 |
| cos_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.393 | 16777216 | 2.024 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 0.469 | 16777216 | 0.053 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 4.700 | 16777216 | 0.169 | 5.905e-08 |
| div_f64_n1048576 | jax-cuda | 16 | 15 | 2.239 | 25165824 | 0.267 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.468 | 25165824 | 2.547 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.908 | 25165824 | 0.313 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.308 | 25165824 | 0.129 | 0 |
| erf_f64_n1048576 | jax-cuda | 8 | 15 | 1.301 | 16777216 | 0.153 | 8.536e-08 |
| erf_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.496 | 16777216 | 1.604 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.576 | 16777216 | 0.505 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.498 | 16777216 | 0.159 | 8.536e-08 |
| exp_f64_n1048576 | jax-cuda | 16 | 15 | 2.424 | 16777216 | 0.164 | 1.093e-07 |
| exp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.612 | 16777216 | 1.300 | 1.093e-07 |
| exp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.885 | 16777216 | 0.422 | 1.093e-07 |
| exp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.792 | 16777216 | 0.142 | 1.093e-07 |
| expm1_f64_n1048576 | jax-cuda | 16 | 15 | 2.644 | 16777216 | 0.150 | 7.39e-08 |
| expm1_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.348 | 16777216 | 2.288 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.320 | 16777216 | 0.343 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.866 | 16777216 | 0.206 | 7.39e-08 |
| log1p_f64_n1048576 | jax-cuda | 16 | 15 | 2.966 | 16777216 | 0.134 | 1.106e-07 |
| log1p_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.446 | 16777216 | 1.782 | 1.106e-07 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.946 | 16777216 | 0.409 | 1.106e-07 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.977 | 16777216 | 0.201 | 1.106e-07 |
| log_f64_n1048576 | jax-cuda | 16 | 15 | 3.427 | 16777216 | 0.116 | 4.308e-08 |
| log_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.448 | 16777216 | 1.776 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.658 | 16777216 | 0.240 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.244 | 16777216 | 0.245 | 4.308e-08 |
| maximum_f64_n1048576 | jax-cuda | 16 | 15 | 1.992 | 25165824 | 0.300 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.464 | 25165824 | 2.570 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.960 | 25165824 | 0.609 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.854 | 25165824 | 0.209 | 0 |
| minimum_f64_n1048576 | jax-cuda | 16 | 15 | 2.762 | 25165824 | 0.216 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.439 | 25165824 | 2.721 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.210 | 25165824 | 0.540 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.872 | 25165824 | 0.208 | 0 |
| mul_f64_n1048576 | jax-cuda | 8 | 15 | 1.972 | 25165824 | 0.151 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.458 | 25165824 | 2.605 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.978 | 25165824 | 0.603 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.233 | 25165824 | 0.185 | 0 |
| neg_f64_n1048576 | jax-cuda | 16 | 15 | 3.486 | 16777216 | 0.114 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.297 | 16777216 | 2.682 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.424 | 16777216 | 0.559 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.870 | 16777216 | 0.277 | 0 |
| pow_f64_n1048576 | jax-cuda | 1 | 15 | 1.730 | 25165824 | 0.022 | 1.48e-07 |
| pow_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.983 | 25165824 | 1.215 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.436 | 25165824 | 0.490 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.378 | 25165824 | 0.177 | 1.48e-07 |
| reduce_max_f64_n1048576 | jax-cuda | 8 | 15 | 1.753 | 8396800 | 0.057 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.768 | 8396800 | 1.691 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.513 | 8396800 | 0.158 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.433 | 8396800 | 0.116 | 0 |
| reduce_mean_f64_n1048576 | jax-cuda | 8 | 15 | 3.019 | 8396800 | 0.033 | 6.5e-09 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.681 | 8396800 | 1.731 | 3.9e-09 |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_min_f64_n1048576 | jax-cuda | 1 | 15 | 0.975 | 8396800 | 0.013 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.803 | 8396800 | 1.675 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.218 | 8396800 | 0.180 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.563 | 8396800 | 0.112 | 0 |
| reduce_prod_f64_n1048576 | jax-cuda | 2 | 15 | 1.079 | 8396800 | 0.023 | 4.436e-06 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.685 | 8396800 | 1.729 | 4.216e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.071 | 8396800 | 0.192 | 3.55e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.944 | 8396800 | 0.102 | 3.55e-06 |
| reduce_sum_f64_n1048576 | jax-cuda | 1 | 15 | 1.530 | 8396800 | 0.008 | 4.449e-06 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.632 | 8396800 | 1.754 | 2.961e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 0.457 | 8396800 | 0.109 | 3.391e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 4.277 | 8396800 | 0.093 | 3.391e-06 |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 0.474 | 8396800 | 0.105 | 3.294e-07 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.410 | 8396800 | 0.117 | 3.294e-07 |
| rem_f64_n1048576 | jax-cuda | 8 | 15 | 2.113 | 25165824 | 0.141 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.557 | 25165824 | 2.142 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.443 | 25165824 | 0.488 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 1.778 | 25165824 | 0.168 | 0 |
| rsqrt_f64_n1048576 | jax-cuda | 8 | 15 | 1.979 | 16777216 | 0.101 | 6.504e-08 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.281 | 16777216 | 2.834 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.265 | 16777216 | 0.629 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.624 | 16777216 | 0.245 | 6.504e-08 |
| scale_complex_c64_n1048576 | jax-cuda | 16 | 15 | 1.938 | 33554448 | 0.410 | 4.129e-08 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.920 | 33554448 | 0.865 | 0 |
| scale_complex_c64_n1048576 | tenferro-cuda-direct | — | — | — | 33554448 | unavailable | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554448 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_f64_n1048576 | jax-cuda | 1 | 15 | 0.809 | 16777224 | 0.031 | 0 |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.353 | 16777224 | 2.257 | 0 |
| scale_real_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777224 | unavailable | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_real_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777224 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| sign_f64_n1048576 | jax-cuda | 1 | 15 | 0.958 | 16777216 | 0.026 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.290 | 16777216 | 2.746 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.358 | 16777216 | 0.586 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.692 | 16777216 | 0.296 | 0 |
| sin_f64_n1048576 | jax-cuda | 4 | 15 | 1.130 | 16777216 | 0.088 | 4.153e-08 |
| sin_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.400 | 16777216 | 1.990 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.532 | 16777216 | 0.519 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.245 | 16777216 | 0.123 | 4.153e-08 |
| sqrt_f64_n1048576 | jax-cuda | 4 | 15 | 1.630 | 16777216 | 0.061 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.284 | 16777216 | 2.806 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.276 | 16777216 | 0.623 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.900 | 16777216 | 0.274 | 0 |
| sub_f64_n1048576 | jax-cuda | 4 | 15 | 1.341 | 25165824 | 0.111 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.431 | 25165824 | 2.767 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.825 | 25165824 | 0.654 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.622 | 25165824 | 0.228 | 0 |
| tanh_f64_n1048576 | jax-cuda | 16 | 15 | 2.436 | 16777216 | 0.163 | 4.437e-08 |
| tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.311 | 16777216 | 2.554 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.490 | 16777216 | 0.534 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.446 | 16777216 | 0.231 | 4.437e-08 |
| where_select_f64_n1048576 | jax-cuda | 16 | 15 | 2.042 | 26214400 | 0.304 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.500 | 26214400 | 2.489 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 2 | 15 | 0.170 | 26214400 | 0.456 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.415 | 26214400 | 0.129 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261114_082852/gpu_isolated_ops.yaml
timestamp: '2026-11-14T08:28:52+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 76cb137693f0b16de2556cc47d32686deaa05863
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
  cpu: AMD EPYC-Genoa Processor
  logical_cpus: 116
  power:
    smt:
      active: '1'
      control: 'on'
  numa:
    nodes: 1
    cpulists:
      '0': 0-115
  smt_siblings:
  - 0-1
  - 2-3
  - 4-5
  - 6-7
  - 8-9
  - 10-11
  - 12-13
  - 14-15
  - 16-17
  - 18-19
  - 20-21
  - 22-23
  - 24-25
  - 26-27
  - 28-29
  - 30-31
  - 32-33
  - 34-35
  - 36-37
  - 38-39
  - 40-41
  - 42-43
  - 44-45
  - 46-47
  - 48-49
  - 50-51
  - 52-53
  - 54-55
  - 56-57
  - 58-59
  - 60-61
  - 62-63
  - 64-65
  - 66-67
  - 68-69
  - 70-71
  - 72-73
  - 74-75
  - 76-77
  - 78-79
  - 80-81
  - 82-83
  - 84-85
  - 86-87
  - 88-89
  - 90-91
  - 92-93
  - 94-95
  - 96-97
  - 98-99
  - 100-101
  - 102-103
  - 104-105
  - 106-107
  - 108-109
  - 110-111
  - 112-113
  - 114-115
  l3_domains:
  - 0-115
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: dba65816ff33
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-115
    l3_domains:
    - 0-115
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
  device_uuid: GPU-c70851ec-1874-5bb5-8a0a-85e4eed17212
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
  - 0.285232
  - 0.150019
  - 0.478273
  - 0.309537
  - 0.311492
  - 0.155886
  - 0.205333
  - 0.199467
  - 0.274057
  - 0.198349
  - 0.195276
  - 0.181588
  - 0.196114
  - 0.166501
  - 0.198908
  median_ms: 0.198908
  bytes_per_second: 674772900034.1866
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-14T08:48:21.525556+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-c70851ec-1874-5bb5-8a0a-85e4eed17212
  cuda_visible_devices: GPU-c70851ec-1874-5bb5-8a0a-85e4eed17212
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
    src/bin/benchmark_gpu_public_api.rs: 3c723c1d0c1658d8499a7713ced3b30357447f661dae37fca7fa4ce2c72de804
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 512b659c731722a152dbdf2e41569974aa84442dfc1f739afbe234e08c71c93b
    src/bin/benchmark_gpu_public_api/plan.rs: 70fce8a6e9ed7518779a51a9d8692e010fa1e470e95eee0ce5276f595737fde8
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: a8bbf8985bbc58c435e3c4dda7135161a141a90a10fa50d4f95128397ce949cb
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: d1ff68f86496d991bcbc730138fe6e3076001b3f23cc1a94c44b96f6857cb8ed
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: b0e1cdcefc95a1f2170739eeffbc2e9600eb3acd363566b605458401b04ec37e
    data/results/nvidia-gpu/gpu/isolated_ops/20261114_082852/fixtures/manifest.json: 3c1a75f9c57996b228c04da1d4cbf34e4137a074e44ddca9f1c37e5824580bd8
    target/release/benchmark_gpu_public_api: 25f583eab659e7cc2c88a104c8203f0c2f400fd4316bfed75f479ee99d3f16bb
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-14T08:48:54Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-14T08:48:41Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-14T08:48:29Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-14T08:48:34Z'
  compile:
    case_wall_seconds: 21.094
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
| add_f64_n1048576 | tenferro-cuda-direct | add_f64_n1048576 untimed 132.2 ms is 1094% of the 12.1 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-direct | sub_f64_n1048576 untimed 127.1 ms is 255% of the 49.8 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-direct | mul_f64_n1048576 untimed 128.1 ms is 187% of the 68.6 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-direct | div_f64_n1048576 untimed 107.9 ms is 319% of the 33.8 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-direct | rem_f64_n1048576 untimed 116.5 ms is 319% of the 36.6 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-direct | pow_f64_n1048576 untimed 80.1 ms is 215% of the 37.3 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-direct | maximum_f64_n1048576 untimed 98.9 ms is 335% of the 29.5 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-direct | minimum_f64_n1048576 untimed 103.4 ms is 313% of the 33.0 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-direct | neg_f64_n1048576 untimed 78.7 ms is 362% of the 21.8 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-direct | abs_f64_n1048576 untimed 67.4 ms is 344% of the 19.6 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-direct | sign_f64_n1048576 untimed 71.1 ms is 331% of the 21.5 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-direct | conj_c64_n1048576 untimed 156.2 ms is 879% of the 17.8 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-direct | exp_f64_n1048576 untimed 129.0 ms is 282% of the 45.8 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-direct | expm1_f64_n1048576 untimed 73.5 ms is 119% of the 62.0 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-direct | log_f64_n1048576 untimed 129.8 ms is 484% of the 26.8 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-direct | log1p_f64_n1048576 untimed 87.9 ms is 186% of the 47.3 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-direct | erf_f64_n1048576 untimed 105.5 ms is 438% of the 24.1 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-direct | sin_f64_n1048576 untimed 94.9 ms is 266% of the 35.7 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-direct | cos_f64_n1048576 untimed 84.4 ms is 870% of the 9.7 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-direct | tanh_f64_n1048576 untimed 75.9 ms is 165% of the 46.0 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-direct | sqrt_f64_n1048576 untimed 89.4 ms is 452% of the 19.8 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | rsqrt_f64_n1048576 untimed 113.8 ms is 579% of the 19.7 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-direct | compare_f64_n1048576 untimed 52.6 ms is 113% of the 46.5 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-direct | where_select_f64_n1048576 untimed 148.2 ms is 3021% of the 4.9 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-direct | clamp_f64_n1048576 untimed 129.3 ms is 162% of the 79.7 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | reduce_sum_f64_n1048576 untimed 21.2 ms is 188% of the 11.2 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | reduce_prod_f64_n1048576 untimed 8.9 ms is 27% of the 32.6 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | reduce_min_f64_n1048576 untimed 9.2 ms is 25% of the 37.3 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | reduce_sum_squares_f64_n1048576 untimed 25.9 ms is 331% of the 7.8 ms timed region (budget 20%) |
| add_f64_n1048576 | tenferro-cuda-eager | add_f64_n1048576 untimed 153.9 ms is 385% of the 39.9 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-eager | sub_f64_n1048576 untimed 80.7 ms is 185% of the 43.5 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-eager | mul_f64_n1048576 untimed 78.3 ms is 160% of the 48.9 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-eager | div_f64_n1048576 untimed 164.5 ms is 507% of the 32.5 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-eager | rem_f64_n1048576 untimed 81.8 ms is 231% of the 35.4 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-eager | pow_f64_n1048576 untimed 106.7 ms is 178% of the 59.9 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-eager | maximum_f64_n1048576 untimed 81.5 ms is 186% of the 43.8 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-eager | minimum_f64_n1048576 untimed 86.1 ms is 130% of the 66.3 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-eager | neg_f64_n1048576 untimed 80.2 ms is 170% of the 47.3 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-eager | abs_f64_n1048576 untimed 92.8 ms is 222% of the 41.8 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-eager | sign_f64_n1048576 untimed 101.7 ms is 157% of the 64.7 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-eager | conj_c64_n1048576 untimed 120.0 ms is 330% of the 36.4 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-eager | exp_f64_n1048576 untimed 104.4 ms is 241% of the 43.3 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-eager | expm1_f64_n1048576 untimed 85.0 ms is 151% of the 56.1 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-eager | log_f64_n1048576 untimed 77.4 ms is 156% of the 49.5 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-eager | log1p_f64_n1048576 untimed 98.6 ms is 336% of the 29.4 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-eager | erf_f64_n1048576 untimed 113.1 ms is 177% of the 63.9 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-eager | sin_f64_n1048576 untimed 85.2 ms is 123% of the 69.4 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-eager | cos_f64_n1048576 untimed 100.8 ms is 90% of the 112.1 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-eager | tanh_f64_n1048576 untimed 125.4 ms is 162% of the 77.2 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-eager | sqrt_f64_n1048576 untimed 75.0 ms is 114% of the 65.6 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | rsqrt_f64_n1048576 untimed 93.2 ms is 277% of the 33.6 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-eager | compare_f64_n1048576 untimed 70.6 ms is 211% of the 33.5 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-eager | where_select_f64_n1048576 untimed 202.6 ms is 573% of the 35.3 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-eager | clamp_f64_n1048576 untimed 129.0 ms is 351% of the 36.7 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | reduce_sum_f64_n1048576 untimed 14.4 ms is 22% of the 66.0 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | reduce_prod_f64_n1048576 untimed 8.8 ms is 30% of the 29.1 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | reduce_max_f64_n1048576 untimed 11.7 ms is 23% of the 51.2 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | reduce_min_f64_n1048576 untimed 11.9 ms is 21% of the 55.8 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | reduce_sum_squares_f64_n1048576 untimed 11.3 ms is 22% of the 51.3 ms timed region (budget 20%) |
| add_f64_n1048576 | pytorch-cuda | add_f64_n1048576 untimed 235.8 ms is 10855% of the 2.2 ms timed region (budget 20%) |
| sub_f64_n1048576 | pytorch-cuda | sub_f64_n1048576 untimed 70.8 ms is 1040% of the 6.8 ms timed region (budget 20%) |
| mul_f64_n1048576 | pytorch-cuda | mul_f64_n1048576 untimed 67.6 ms is 907% of the 7.5 ms timed region (budget 20%) |
| div_f64_n1048576 | pytorch-cuda | div_f64_n1048576 untimed 98.4 ms is 1355% of the 7.3 ms timed region (budget 20%) |
| rem_f64_n1048576 | pytorch-cuda | rem_f64_n1048576 untimed 95.6 ms is 1124% of the 8.5 ms timed region (budget 20%) |
| pow_f64_n1048576 | pytorch-cuda | pow_f64_n1048576 untimed 144.6 ms is 922% of the 15.7 ms timed region (budget 20%) |
| maximum_f64_n1048576 | pytorch-cuda | maximum_f64_n1048576 untimed 92.8 ms is 1280% of the 7.2 ms timed region (budget 20%) |
| minimum_f64_n1048576 | pytorch-cuda | minimum_f64_n1048576 untimed 51.4 ms is 752% of the 6.8 ms timed region (budget 20%) |
| neg_f64_n1048576 | pytorch-cuda | neg_f64_n1048576 untimed 62.0 ms is 1315% of the 4.7 ms timed region (budget 20%) |
| abs_f64_n1048576 | pytorch-cuda | abs_f64_n1048576 untimed 67.9 ms is 1251% of the 5.4 ms timed region (budget 20%) |
| sign_f64_n1048576 | pytorch-cuda | sign_f64_n1048576 untimed 59.6 ms is 1190% of the 5.0 ms timed region (budget 20%) |
| conj_c64_n1048576 | pytorch-cuda | conj_c64_n1048576 untimed 121.7 ms is 554% of the 22.0 ms timed region (budget 20%) |
| exp_f64_n1048576 | pytorch-cuda | exp_f64_n1048576 untimed 136.3 ms is 763% of the 17.9 ms timed region (budget 20%) |
| expm1_f64_n1048576 | pytorch-cuda | expm1_f64_n1048576 untimed 81.7 ms is 1472% of the 5.6 ms timed region (budget 20%) |
| log_f64_n1048576 | pytorch-cuda | log_f64_n1048576 untimed 76.6 ms is 1120% of the 6.8 ms timed region (budget 20%) |
| log1p_f64_n1048576 | pytorch-cuda | log1p_f64_n1048576 untimed 98.5 ms is 1448% of the 6.8 ms timed region (budget 20%) |
| erf_f64_n1048576 | pytorch-cuda | erf_f64_n1048576 untimed 80.6 ms is 1034% of the 7.8 ms timed region (budget 20%) |
| sin_f64_n1048576 | pytorch-cuda | sin_f64_n1048576 untimed 71.9 ms is 1118% of the 6.4 ms timed region (budget 20%) |
| cos_f64_n1048576 | pytorch-cuda | cos_f64_n1048576 untimed 90.7 ms is 1449% of the 6.3 ms timed region (budget 20%) |
| tanh_f64_n1048576 | pytorch-cuda | tanh_f64_n1048576 untimed 120.9 ms is 2101% of the 5.8 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | pytorch-cuda | sqrt_f64_n1048576 untimed 69.3 ms is 1508% of the 4.6 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | pytorch-cuda | rsqrt_f64_n1048576 untimed 67.8 ms is 1522% of the 4.5 ms timed region (budget 20%) |
| compare_f64_n1048576 | pytorch-cuda | compare_f64_n1048576 untimed 68.3 ms is 686% of the 9.9 ms timed region (budget 20%) |
| where_select_f64_n1048576 | pytorch-cuda | where_select_f64_n1048576 untimed 84.1 ms is 1073% of the 7.8 ms timed region (budget 20%) |
| clamp_f64_n1048576 | pytorch-cuda | clamp_f64_n1048576 untimed 59.4 ms is 880% of the 6.7 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | pytorch-cuda | reduce_sum_f64_n1048576 untimed 30.8 ms is 56% of the 55.1 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | pytorch-cuda | reduce_prod_f64_n1048576 untimed 25.7 ms is 46% of the 55.6 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | pytorch-cuda | reduce_max_f64_n1048576 untimed 26.8 ms is 47% of the 56.7 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | pytorch-cuda | reduce_min_f64_n1048576 untimed 26.3 ms is 46% of the 57.1 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | pytorch-cuda | reduce_mean_f64_n1048576 untimed 34.3 ms is 50% of the 69.2 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | pytorch-cuda | scale_real_f64_n1048576 untimed 62.1 ms is 1109% of the 5.6 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | pytorch-cuda | scale_complex_c64_n1048576 untimed 164.5 ms is 871% of the 18.9 ms timed region (budget 20%) |
| add_f64_n1048576 | jax-cuda | add_f64_n1048576 untimed 580.7 ms is 2933% of the 19.8 ms timed region (budget 20%) |
| sub_f64_n1048576 | jax-cuda | sub_f64_n1048576 untimed 229.2 ms is 654% of the 35.1 ms timed region (budget 20%) |
| mul_f64_n1048576 | jax-cuda | mul_f64_n1048576 untimed 197.9 ms is 619% of the 32.0 ms timed region (budget 20%) |
| div_f64_n1048576 | jax-cuda | div_f64_n1048576 untimed 212.9 ms is 564% of the 37.7 ms timed region (budget 20%) |
| rem_f64_n1048576 | jax-cuda | rem_f64_n1048576 untimed 329.3 ms is 981% of the 33.6 ms timed region (budget 20%) |
| pow_f64_n1048576 | jax-cuda | pow_f64_n1048576 untimed 333.4 ms is 1151% of the 29.0 ms timed region (budget 20%) |
| maximum_f64_n1048576 | jax-cuda | maximum_f64_n1048576 untimed 204.6 ms is 650% of the 31.5 ms timed region (budget 20%) |
| minimum_f64_n1048576 | jax-cuda | minimum_f64_n1048576 untimed 229.5 ms is 587% of the 39.1 ms timed region (budget 20%) |
| neg_f64_n1048576 | jax-cuda | neg_f64_n1048576 untimed 207.1 ms is 336% of the 61.7 ms timed region (budget 20%) |
| abs_f64_n1048576 | jax-cuda | abs_f64_n1048576 untimed 216.4 ms is 626% of the 34.6 ms timed region (budget 20%) |
| sign_f64_n1048576 | jax-cuda | sign_f64_n1048576 untimed 203.7 ms is 1414% of the 14.4 ms timed region (budget 20%) |
| conj_c64_n1048576 | jax-cuda | conj_c64_n1048576 untimed 276.9 ms is 383% of the 72.4 ms timed region (budget 20%) |
| exp_f64_n1048576 | jax-cuda | exp_f64_n1048576 untimed 229.9 ms is 656% of the 35.0 ms timed region (budget 20%) |
| expm1_f64_n1048576 | jax-cuda | expm1_f64_n1048576 untimed 319.8 ms is 854% of the 37.4 ms timed region (budget 20%) |
| log_f64_n1048576 | jax-cuda | log_f64_n1048576 untimed 226.7 ms is 428% of the 52.9 ms timed region (budget 20%) |
| log1p_f64_n1048576 | jax-cuda | log1p_f64_n1048576 untimed 283.9 ms is 593% of the 47.9 ms timed region (budget 20%) |
| erf_f64_n1048576 | jax-cuda | erf_f64_n1048576 untimed 261.3 ms is 1245% of the 21.0 ms timed region (budget 20%) |
| sin_f64_n1048576 | jax-cuda | sin_f64_n1048576 untimed 332.5 ms is 1630% of the 20.4 ms timed region (budget 20%) |
| cos_f64_n1048576 | jax-cuda | cos_f64_n1048576 untimed 262.4 ms is 952% of the 27.6 ms timed region (budget 20%) |
| tanh_f64_n1048576 | jax-cuda | tanh_f64_n1048576 untimed 266.2 ms is 705% of the 37.7 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | jax-cuda | sqrt_f64_n1048576 untimed 227.6 ms is 787% of the 28.9 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | jax-cuda | rsqrt_f64_n1048576 untimed 236.6 ms is 803% of the 29.5 ms timed region (budget 20%) |
| compare_f64_n1048576 | jax-cuda | compare_f64_n1048576 untimed 147.9 ms is 512% of the 28.9 ms timed region (budget 20%) |
| where_select_f64_n1048576 | jax-cuda | where_select_f64_n1048576 untimed 244.6 ms is 745% of the 32.8 ms timed region (budget 20%) |
| clamp_f64_n1048576 | jax-cuda | clamp_f64_n1048576 untimed 193.1 ms is 532% of the 36.3 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | jax-cuda | reduce_sum_f64_n1048576 untimed 476.0 ms is 2047% of the 23.3 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | jax-cuda | reduce_prod_f64_n1048576 untimed 374.9 ms is 2158% of the 17.4 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | jax-cuda | reduce_max_f64_n1048576 untimed 494.2 ms is 1005% of the 49.2 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | jax-cuda | reduce_min_f64_n1048576 untimed 264.4 ms is 1512% of the 17.5 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | jax-cuda | reduce_mean_f64_n1048576 untimed 202.3 ms is 400% of the 50.6 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | jax-cuda | scale_real_f64_n1048576 untimed 264.1 ms is 787% of the 33.6 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | jax-cuda | scale_complex_c64_n1048576 untimed 274.4 ms is 936% of the 29.3 ms timed region (budget 20%) |