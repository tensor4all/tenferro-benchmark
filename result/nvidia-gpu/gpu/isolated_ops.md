# gpu/isolated_ops: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261013_102419/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_f32_n1024 | 8.340 (0.549) | 11.203 (1.034) | 9.887 (0.085) | 109.446 (3.641) | 81.410 (1.836) |
| add_f64_n1024 | 8.770 (0.794) | 15.230 (0.153) | 10.142 (0.312) | 104.639 (20.100) | 82.175 (11.752) |
| sub_f32_n1024 | 8.613 (0.538) | 18.172 (0.889) | 9.904 (0.077) | 87.920 (9.874) | 80.256 (6.531) |
| sub_f64_n1024 | 8.993 (0.837) | 20.466 (1.137) | 10.380 (0.298) | 84.359 (8.436) | 77.508 (7.826) |
| mul_f32_n1024 | 9.275 (0.946) | 21.644 (0.679) | 9.893 (0.207) | 83.931 (8.489) | 79.127 (5.092) |
| mul_f64_n1024 | 9.078 (0.722) | 21.865 (1.266) | 9.709 (0.240) | 84.997 (4.897) | 76.956 (5.394) |
| div_f32_n1024 | 8.707 (0.674) | 26.517 (3.607) | 9.855 (0.581) | 79.458 (6.492) | 76.361 (6.983) |
| div_f64_n1024 | 9.279 (0.731) | 25.796 (3.515) | 9.789 (0.318) | 82.558 (5.266) | 78.768 (5.140) |
| rem_f32_n1024 | 8.430 (0.728) | 178.598 (8.508) | 9.843 (0.116) | 96.633 (6.901) | 87.067 (5.138) |
| rem_f64_n1024 | 8.587 (0.959) | 228.652 (19.727) | 9.863 (0.118) | 109.351 (13.995) | 95.546 (18.890) |
| pow_f32_n1024 | 8.683 (0.766) | 266.562 (12.362) | 10.149 (0.264) | 142.519 (8.679) | 100.303 (23.155) |
| pow_f64_n1024 | 9.746 (0.217) | 294.700 (31.436) | 30.950 (0.015) | 171.893 (16.755) | 104.135 (15.527) |
| maximum_f32_n1024 | 8.815 (1.049) | 203.609 (16.906) | 9.907 (0.596) | 160.531 (28.715) | 93.755 (6.408) |
| maximum_f64_n1024 | 8.999 (0.783) | 239.027 (14.262) | 9.749 (0.228) | 147.402 (25.251) | 96.857 (5.448) |
| minimum_f32_n1024 | 9.097 (0.760) | 275.978 (8.517) | 10.092 (0.469) | 143.118 (21.678) | 96.959 (7.348) |
| minimum_f64_n1024 | 8.896 (0.852) | 302.490 (9.451) | 9.649 (0.127) | 135.328 (12.649) | 98.149 (8.673) |
| neg_f32_n1024 | 8.246 (0.702) | 18.747 (0.115) | 9.220 (0.101) | 125.933 (9.653) | 94.233 (8.287) |
| neg_f64_n1024 | 8.322 (0.566) | 19.132 (0.259) | 9.336 (0.275) | 120.487 (10.125) | 92.923 (4.970) |
| abs_f32_n1024 | 6.877 (1.049) | 187.154 (2.575) | 11.944 (0.110) | 92.082 (16.021) | 95.023 (9.494) |
| abs_f64_n1024 | 6.451 (1.031) | 127.551 (3.187) | 11.925 (0.087) | 97.649 (10.304) | 88.052 (11.752) |
| sign_f32_n1024 | 5.943 (0.693) | 136.922 (2.521) | 9.172 (0.103) | 100.492 (15.690) | 95.564 (11.581) |
| sign_f64_n1024 | 5.317 (0.089) | 143.911 (1.898) | 9.387 (0.119) | 99.341 (13.599) | 90.511 (9.559) |
| conj_c32_n1024 | 5.219 (0.481) | 145.515 (4.657) | 10.713 (0.134) | 96.446 (9.037) | 93.912 (2.968) |
| conj_c64_n1024 | 6.329 (0.629) | 152.375 (4.342) | 11.060 (0.308) | 94.276 (7.214) | 92.589 (1.152) |
| exp_f32_n1024 | 7.536 (0.448) | 18.998 (0.223) | 8.685 (0.172) | 91.539 (13.288) | 89.150 (1.247) |
| exp_f64_n1024 | 7.814 (0.216) | 19.794 (0.427) | 9.315 (0.290) | 95.507 (12.721) | 95.729 (3.814) |
| expm1_f32_n1024 | 8.082 (0.747) | 20.094 (0.350) | 9.225 (0.088) | 97.036 (11.805) | 96.865 (9.377) |
| expm1_f64_n1024 | 7.943 (0.089) | 19.173 (0.294) | 9.410 (0.107) | 101.182 (13.201) | 99.019 (15.646) |
| log_f32_n1024 | 6.890 (0.961) | 20.065 (0.192) | 8.860 (0.126) | 100.984 (11.458) | 99.761 (16.110) |
| log_f64_n1024 | 6.309 (0.515) | 20.380 (0.673) | 10.833 (0.010) | 120.654 (6.489) | 101.843 (10.690) |
| log1p_f32_n1024 | 5.672 (0.392) | 19.828 (0.425) | 8.992 (0.096) | 131.256 (6.433) | 96.419 (8.812) |
| log1p_f64_n1024 | 5.585 (0.343) | 20.391 (0.481) | 11.209 (0.026) | 141.811 (2.721) | 100.188 (11.969) |
| erf_f32_n1024 | 5.249 (0.362) | 19.632 (0.421) | 9.865 (0.498) | 253.843 (8.409) | 115.701 (18.308) |
| erf_f64_n1024 | 5.655 (0.107) | 19.584 (0.560) | 15.752 (0.014) | 249.960 (16.578) | 126.135 (24.749) |
| sin_f32_n1024 | 5.565 (0.132) | 19.991 (0.361) | 9.557 (0.057) | 171.728 (18.825) | 158.084 (28.176) |
| sin_f64_n1024 | 5.462 (0.248) | 19.762 (0.358) | 9.303 (0.400) | 155.643 (4.723) | 159.263 (15.370) |
| cos_f32_n1024 | 5.808 (0.229) | 19.517 (0.345) | 8.938 (0.269) | 169.810 (10.891) | 165.102 (22.960) |
| cos_f64_n1024 | 5.838 (0.541) | 20.420 (0.218) | 9.419 (0.298) | 162.802 (7.553) | 162.292 (22.528) |
| tanh_f32_n1024 | 5.341 (0.181) | 20.331 (0.416) | 9.841 (0.551) | 162.893 (24.347) | 149.120 (23.831) |
| tanh_f64_n1024 | 5.447 (0.322) | 20.397 (0.665) | 9.551 (0.123) | 155.635 (26.142) | 151.999 (32.548) |
| sqrt_f32_n1024 | 5.329 (0.351) | 19.585 (0.417) | 9.495 (0.258) | 150.610 (26.476) | 142.682 (23.982) |
| sqrt_f64_n1024 | 5.368 (0.151) | 20.163 (0.242) | 9.521 (0.129) | 143.718 (20.762) | 137.854 (19.622) |
| rsqrt_f32_n1024 | 5.256 (0.244) | 20.079 (0.531) | 9.134 (0.079) | 175.874 (40.154) | 135.174 (19.100) |
| rsqrt_f64_n1024 | 5.439 (0.426) | 20.715 (0.405) | 9.352 (0.393) | 174.180 (37.185) | 129.482 (9.772) |
| compare_f32_n1024 | unsupported | 348.666 (4.704) | 9.047 (0.082) | 104.783 (17.080) | 127.261 (17.181) |
| compare_f64_n1024 | unsupported | 350.611 (20.382) | 9.096 (0.141) | 99.948 (10.216) | 94.742 (14.688) |
| where_select_f32_n1024 | 9.136 (0.154) | 452.531 (16.376) | 13.591 (0.546) | 125.359 (12.819) | 100.725 (11.989) |
| where_select_f64_n1024 | 9.581 (0.636) | 501.838 (34.365) | 13.106 (0.143) | 130.465 (13.144) | 100.799 (11.609) |
| clamp_f32_n1024 | 152.244 (5.753) | 1027.912 (78.706) | 11.656 (0.149) | 100.323 (12.213) | 101.946 (10.855) |
| clamp_f64_n1024 | 151.790 (2.821) | 1075.370 (11.644) | 11.189 (0.379) | 103.579 (12.988) | 97.373 (11.855) |
| reduce_sum_f32_n1024 | 78.513 (1.044) | 593.796 (13.149) | 14.809 (0.259) | 99.436 (1.657) | 92.823 (3.194) |
| reduce_sum_f64_n1024 | 80.746 (2.041) | 587.031 (21.887) | 15.161 (0.141) | 167.419 (10.170) | 89.901 (1.369) |
| reduce_prod_f32_n1024 | 81.297 (1.238) | 562.452 (10.916) | 13.813 (0.487) | 177.651 (5.489) | 136.761 (6.488) |
| reduce_prod_f64_n1024 | 81.306 (1.139) | 604.142 (33.336) | 13.570 (0.333) | 141.851 (10.803) | 154.137 (4.825) |
| reduce_max_f32_n1024 | 81.720 (7.795) | 565.410 (32.786) | 14.221 (0.398) | 108.424 (16.579) | 134.362 (10.400) |
| reduce_max_f64_n1024 | 81.951 (2.263) | 234.469 (6.612) | 14.116 (0.232) | 167.643 (10.219) | 154.980 (4.817) |
| reduce_min_f32_n1024 | 81.485 (4.788) | 239.818 (5.159) | 14.365 (0.526) | 99.120 (1.640) | 91.145 (1.858) |
| reduce_min_f64_n1024 | 81.653 (2.747) | 242.264 (5.677) | 14.312 (0.275) | 111.281 (11.228) | 99.974 (8.013) |
| reduce_sum_squares_f32_n1024 | 81.836 (1.384) | 264.214 (2.488) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1024 | 81.193 (1.018) | 267.672 (2.421) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1024 | unsupported | unsupported | 14.602 (0.345) | 99.486 (3.037) | 98.667 (1.348) |
| reduce_mean_f64_n1024 | unsupported | unsupported | 15.290 (0.265) | 106.097 (12.156) | 97.302 (0.762) |
| add_f32_n1048576 | 44.099 (0.316) | 44.969 (0.398) | 39.757 (0.083) | 110.544 (0.688) | 97.882 (1.330) |
| add_f64_n1048576 | 82.745 (0.179) | 86.113 (0.584) | 77.776 (0.156) | 114.605 (3.137) | 91.432 (0.778) |
| sub_f32_n1048576 | 43.250 (0.047) | 44.604 (0.038) | 39.756 (0.086) | 108.788 (1.763) | 98.881 (0.650) |
| sub_f64_n1048576 | 82.572 (0.094) | 84.630 (1.238) | 77.749 (0.077) | 106.210 (3.195) | 83.139 (4.593) |
| mul_f32_n1048576 | 43.288 (0.078) | 43.262 (0.352) | 39.777 (0.118) | 107.444 (1.833) | 73.218 (6.092) |
| mul_f64_n1048576 | 82.836 (0.089) | 83.017 (0.366) | 77.825 (0.193) | 107.736 (1.191) | 83.395 (1.094) |
| div_f32_n1048576 | 43.283 (0.076) | 43.214 (0.384) | 39.979 (0.031) | 97.663 (0.766) | 74.644 (2.430) |
| div_f64_n1048576 | 94.112 (0.327) | 96.319 (0.378) | 94.061 (0.084) | 123.833 (1.084) | 122.108 (0.374) |
| rem_f32_n1048576 | 42.240 (0.292) | 98.217 (0.318) | 39.883 (0.022) | 98.543 (0.566) | 84.259 (8.007) |
| rem_f64_n1048576 | 145.902 (0.199) | 250.622 (0.325) | 78.180 (0.081) | 97.846 (11.057) | 83.317 (1.144) |
| pow_f32_n1048576 | 41.948 (0.150) | 98.091 (0.312) | 41.575 (0.115) | 121.463 (0.478) | 77.203 (3.732) |
| pow_f64_n1048576 | 1015.147 (0.380) | 1118.565 (2.122) | 1013.225 (0.431) | 1319.352 (9.384) | 1277.651 (15.032) |
| maximum_f32_n1048576 | 42.256 (0.066) | 98.525 (0.246) | 39.884 (0.017) | 109.278 (6.719) | 92.177 (15.523) |
| maximum_f64_n1048576 | 81.383 (0.402) | 188.187 (0.434) | 78.430 (0.057) | 100.391 (2.388) | 83.575 (0.777) |
| minimum_f32_n1048576 | 42.246 (0.193) | 98.566 (0.462) | 39.906 (0.127) | 107.243 (0.567) | 73.858 (1.078) |
| minimum_f64_n1048576 | 81.298 (0.158) | 187.892 (0.230) | 78.455 (0.089) | 104.642 (2.222) | 83.876 (0.842) |
| neg_f32_n1048576 | 29.346 (0.215) | 29.606 (0.181) | 27.330 (0.041) | 103.384 (1.753) | 70.920 (2.091) |
| neg_f64_n1048576 | 56.109 (0.252) | 57.409 (0.544) | 53.028 (0.043) | 85.309 (5.567) | 70.485 (1.549) |
| abs_f32_n1048576 | 29.131 (0.223) | 56.386 (0.226) | 27.490 (0.076) | 70.270 (1.636) | 72.034 (1.204) |
| abs_f64_n1048576 | 56.148 (0.510) | 107.514 (0.135) | 53.187 (0.081) | 66.302 (2.096) | 68.886 (2.019) |
| sign_f32_n1048576 | 29.095 (0.051) | 56.343 (0.136) | 27.390 (0.042) | 71.539 (1.757) | 70.089 (0.944) |
| sign_f64_n1048576 | 55.973 (0.380) | 107.735 (0.246) | 53.214 (0.079) | 66.255 (4.556) | 66.188 (2.296) |
| conj_c32_n1048576 | 55.649 (0.407) | 107.888 (0.164) | 53.069 (0.144) | 66.582 (2.189) | 67.275 (0.549) |
| conj_c64_n1048576 | 108.973 (0.856) | 209.282 (0.153) | 104.054 (0.107) | 115.972 (1.409) | 116.874 (2.672) |
| exp_f32_n1048576 | 29.499 (0.058) | 29.599 (0.099) | 27.406 (0.047) | 70.333 (3.107) | 70.902 (2.839) |
| exp_f64_n1048576 | 183.808 (0.218) | 185.940 (0.299) | 179.700 (0.103) | 203.983 (1.626) | 204.029 (2.008) |
| expm1_f32_n1048576 | 29.468 (0.113) | 29.778 (0.325) | 27.518 (0.078) | 94.707 (16.882) | 79.255 (14.452) |
| expm1_f64_n1048576 | 227.511 (1.181) | 229.630 (0.171) | 223.921 (0.279) | 265.995 (7.886) | 264.459 (8.608) |
| log_f32_n1048576 | 29.582 (0.064) | 29.726 (0.200) | 27.477 (0.017) | 87.614 (14.286) | 84.242 (11.807) |
| log_f64_n1048576 | 353.218 (0.529) | 355.652 (2.603) | 348.805 (0.140) | 404.644 (14.423) | 399.217 (14.281) |
| log1p_f32_n1048576 | 29.411 (0.061) | 29.953 (0.313) | 27.486 (0.103) | 99.957 (10.853) | 97.244 (6.014) |
| log1p_f64_n1048576 | 364.964 (0.232) | 367.279 (0.844) | 361.009 (0.158) | 426.025 (12.477) | 420.672 (5.773) |
| erf_f32_n1048576 | 29.357 (0.065) | 29.698 (0.137) | 27.458 (0.045) | 165.004 (36.213) | 124.426 (13.874) |
| erf_f64_n1048576 | 533.127 (0.525) | 535.841 (0.670) | 527.866 (0.510) | 682.789 (16.127) | 673.317 (18.254) |
| sin_f32_n1048576 | 29.532 (0.196) | 29.947 (0.242) | 27.561 (0.045) | 134.451 (8.576) | 130.575 (22.362) |
| sin_f64_n1048576 | 201.079 (0.438) | 203.139 (0.719) | 210.961 (0.198) | 239.521 (1.234) | 253.320 (1.647) |
| cos_f32_n1048576 | 29.588 (0.192) | 29.753 (0.174) | 27.584 (0.102) | 124.201 (10.230) | 101.726 (13.867) |
| cos_f64_n1048576 | 201.035 (0.712) | 202.676 (0.844) | 210.492 (0.156) | 239.526 (1.640) | 237.200 (1.335) |
| tanh_f32_n1048576 | 29.533 (0.245) | 29.632 (0.042) | 27.423 (0.035) | 100.505 (11.574) | 96.980 (3.352) |
| tanh_f64_n1048576 | 171.100 (0.713) | 172.811 (0.934) | 166.650 (0.167) | 190.346 (1.927) | 190.518 (1.928) |
| sqrt_f32_n1048576 | 29.571 (0.027) | 29.601 (0.032) | 27.464 (0.037) | 93.852 (2.413) | 96.600 (4.287) |
| sqrt_f64_n1048576 | 94.006 (0.300) | 95.020 (0.363) | 91.612 (0.037) | 103.268 (1.373) | 102.816 (1.073) |
| rsqrt_f32_n1048576 | 29.195 (0.037) | 29.564 (0.101) | 27.421 (0.033) | 102.865 (8.086) | 94.365 (1.212) |
| rsqrt_f64_n1048576 | 62.694 (0.119) | 64.047 (0.488) | 61.353 (0.040) | 95.637 (1.425) | 91.468 (0.620) |
| compare_f32_n1048576 | 32.329 (0.136) | 87.793 (0.401) | 30.441 (0.016) | 72.725 (3.063) | 95.982 (0.530) |
| compare_f64_n1048576 | 57.010 (0.066) | 164.910 (0.566) | 55.973 (0.035) | 67.703 (2.788) | 99.533 (0.664) |
| where_select_f32_n1048576 | 45.734 (0.050) | 118.671 (0.625) | 43.035 (0.044) | 102.920 (1.164) | 95.826 (2.898) |
| where_select_f64_n1048576 | 84.990 (0.397) | 208.795 (0.152) | 80.988 (0.156) | 89.034 (6.461) | 95.830 (5.223) |
| clamp_f32_n1048576 | 334.033 (9.284) | 461.339 (54.861) | 28.298 (0.020) | 72.372 (1.405) | 100.180 (3.548) |
| clamp_f64_n1048576 | 342.949 (16.261) | 468.790 (19.200) | 62.302 (0.340) | 82.898 (1.291) | 95.240 (5.051) |
| reduce_sum_f32_n1048576 | 153.821 (10.831) | 282.901 (6.440) | 16.944 (1.089) | 122.790 (10.832) | 120.079 (19.013) |
| reduce_sum_f64_n1048576 | 115.748 (4.246) | 269.990 (2.650) | 32.432 (0.020) | 168.519 (10.952) | 154.984 (3.195) |
| reduce_prod_f32_n1048576 | 113.290 (0.985) | 270.676 (4.303) | 16.137 (0.026) | 156.017 (2.154) | 107.430 (10.106) |
| reduce_prod_f64_n1048576 | 113.019 (1.591) | 272.647 (4.002) | 32.215 (0.014) | 162.215 (2.865) | 149.638 (2.787) |
| reduce_max_f32_n1048576 | 113.323 (0.877) | 272.310 (5.109) | 16.538 (0.041) | 158.758 (2.341) | 145.824 (1.605) |
| reduce_max_f64_n1048576 | 113.380 (1.682) | 271.568 (1.508) | 41.688 (0.029) | 155.127 (1.684) | 147.812 (1.267) |
| reduce_min_f32_n1048576 | 113.580 (1.208) | 272.429 (2.889) | 16.556 (0.197) | 162.737 (16.628) | 146.558 (13.148) |
| reduce_min_f64_n1048576 | 114.126 (1.038) | 274.160 (5.475) | 41.675 (0.017) | 168.145 (6.480) | 151.024 (8.266) |
| reduce_sum_squares_f32_n1048576 | 112.912 (1.271) | 270.665 (5.280) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1048576 | 113.335 (1.091) | 275.328 (3.507) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1048576 | unsupported | unsupported | 16.219 (0.044) | 171.464 (24.975) | 160.522 (22.672) |
| reduce_mean_f64_n1048576 | unsupported | unsupported | 32.865 (0.034) | 148.465 (2.730) | 169.903 (22.300) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.135 | 0 |
| add_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.245 | 0 |
| sub_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.205 | 0 |
| sub_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.302 | 0 |
| mul_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.374 | 0 |
| mul_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.324 | 0 |
| div_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.229 | 0 |
| div_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.375 | 0 |
| rem_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.158 | 0 |
| rem_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.198 | 0 |
| pow_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.223 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.495 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.257 | 0 |
| maximum_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.304 | 0 |
| minimum_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.329 | 0 |
| minimum_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.277 | 0 |
| neg_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.111 | 0 |
| neg_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.130 | 0 |
| abs_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.521 | 0 |
| abs_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 3.303 | 0 |
| sign_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.043 | 0 |
| sign_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.722 | 0 |
| conj_c32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.672 | 0 |
| conj_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 0.810 | 0 |
| exp_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 1.929 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.000 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.069 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.033 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.527 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 3.230 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.904 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.859 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.687 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.895 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.849 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.796 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.973 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.989 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.734 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.789 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.729 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.748 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.691 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.785 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; steady-state sampling is not established: consecutive batch durations increase in repeated runs 20261013_101359 and 20261013_102038. Retained as diagnostics pending allocator/lifetime investigation. |
| compare_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; steady-state sampling is not established: consecutive batch durations increase in repeated runs 20261013_101359 and 20261013_102038. Retained as diagnostics pending allocator/lifetime investigation. |
| where_select_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.339 | 0 |
| where_select_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.453 | 0 |
| clamp_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.436 | 0 |
| clamp_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.429 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.512 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.584 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.601 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.602 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.615 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.622 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.608 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.613 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.619 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.598 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.822 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.648 | 0 |
| sub_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.768 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.642 | 0 |
| mul_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.770 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.651 | 0 |
| div_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.770 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.012 | 0 |
| rem_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.703 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.334 | 0 |
| pow_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.685 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.030 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.704 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.604 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.704 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.602 | 0 |
| neg_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.878 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.795 | 0 |
| abs_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.864 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.797 | 0 |
| sign_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.862 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.791 | 0 |
| conj_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.781 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.744 | 0 |
| exp_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.888 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.941 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.886 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.640 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.893 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.826 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.882 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.920 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.879 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.133 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.890 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.217 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.894 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.217 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.890 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.738 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.893 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.008 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.868 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.006 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.069 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.649 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.927 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.720 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.672 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.744 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.461 | 0.001812 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.704 | 2.148e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.625 | 0.005478 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.617 | 3.659e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.626 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.628 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.635 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.652 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.613 | 0.0003528 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.627 | 3.293e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1024 | tenferro-cuda-eager | 256 | 15 | 2.868 | 0 |
| add_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 1.949 | 0 |
| sub_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.326 | 0 |
| sub_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.620 | 0 |
| mul_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.770 | 0 |
| mul_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.799 | 0 |
| div_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.394 | 0 |
| div_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.302 | 0 |
| rem_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.858 | 0 |
| rem_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.658 | 0 |
| pow_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.132 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.358 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.258 | 0 |
| maximum_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.824 | 0 |
| minimum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.208 | 0 |
| minimum_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.420 | 0 |
| neg_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.400 | 0 |
| neg_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.449 | 0 |
| abs_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.994 | 0 |
| abs_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.041 | 0 |
| sign_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.191 | 0 |
| sign_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.303 | 0 |
| conj_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.328 | 0 |
| conj_c64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.438 | 0 |
| exp_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.432 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.534 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.572 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.454 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.568 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.609 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.538 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.610 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.513 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.507 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.559 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.530 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.498 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.614 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.602 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.611 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.507 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.581 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.570 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.652 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.789 | 0 |
| compare_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.805 | 0 |
| where_select_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.620 | 0 |
| where_select_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 4.015 | 0 |
| clamp_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 4.112 | 0 |
| clamp_f64_n1024 | tenferro-cuda-eager | 2 | 15 | 2.151 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.375 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.348 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.250 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.417 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.262 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 1.876 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 1.919 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.876 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.114 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.141 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.878 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.756 | 0 |
| sub_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.855 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.708 | 0 |
| mul_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.769 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.657 | 0 |
| div_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.766 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.082 | 0 |
| rem_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.143 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.005 | 0 |
| pow_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.139 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-eager | 2 | 15 | 2.237 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.153 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.011 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.154 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.006 | 0 |
| neg_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.895 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.837 | 0 |
| abs_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.609 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.440 | 0 |
| sign_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.606 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.448 | 0 |
| conj_c32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.452 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.349 | 0 |
| exp_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.894 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.975 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.906 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.674 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.902 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.845 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.917 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.938 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.901 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.143 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.917 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.250 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.904 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.243 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.896 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.765 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.894 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.041 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.892 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.050 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.809 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.639 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.797 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.341 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.691 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.750 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.263 | 0.001812 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.160 | 2.148e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.165 | 0.005478 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.181 | 3.659e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.178 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.173 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.179 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.193 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.165 | 0.0003528 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.203 | 3.293e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1024 | pytorch-cuda | 256 | 15 | 2.531 | 0 |
| add_f64_n1024 | pytorch-cuda | 256 | 15 | 2.596 | 0 |
| sub_f32_n1024 | pytorch-cuda | 256 | 15 | 2.535 | 0 |
| sub_f64_n1024 | pytorch-cuda | 256 | 15 | 2.657 | 0 |
| mul_f32_n1024 | pytorch-cuda | 256 | 15 | 2.533 | 0 |
| mul_f64_n1024 | pytorch-cuda | 256 | 15 | 2.485 | 0 |
| div_f32_n1024 | pytorch-cuda | 256 | 15 | 2.523 | 0 |
| div_f64_n1024 | pytorch-cuda | 256 | 15 | 2.506 | 0 |
| rem_f32_n1024 | pytorch-cuda | 256 | 15 | 2.520 | 0 |
| rem_f64_n1024 | pytorch-cuda | 256 | 15 | 2.525 | 0 |
| pow_f32_n1024 | pytorch-cuda | 256 | 15 | 2.598 | 0.000117 |
| pow_f64_n1024 | pytorch-cuda | 128 | 15 | 3.962 | 1.48e-07 |
| maximum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.536 | 0 |
| maximum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.496 | 0 |
| minimum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.584 | 0 |
| minimum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.470 | 0 |
| neg_f32_n1024 | pytorch-cuda | 256 | 15 | 2.360 | 0 |
| neg_f64_n1024 | pytorch-cuda | 256 | 15 | 2.390 | 0 |
| abs_f32_n1024 | pytorch-cuda | 256 | 15 | 3.058 | 0 |
| abs_f64_n1024 | pytorch-cuda | 256 | 15 | 3.053 | 0 |
| sign_f32_n1024 | pytorch-cuda | 256 | 15 | 2.348 | 0 |
| sign_f64_n1024 | pytorch-cuda | 256 | 15 | 2.403 | 0 |
| conj_c32_n1024 | pytorch-cuda | 256 | 15 | 2.743 | 0 |
| conj_c64_n1024 | pytorch-cuda | 256 | 15 | 2.831 | 0 |
| exp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.223 | 0.0001272 |
| exp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.385 | 1.095e-07 |
| expm1_f32_n1024 | pytorch-cuda | 256 | 15 | 2.362 | 7.911e-05 |
| expm1_f64_n1024 | pytorch-cuda | 256 | 15 | 2.409 | 7.39e-08 |
| log_f32_n1024 | pytorch-cuda | 256 | 15 | 2.268 | 7.945e-05 |
| log_f64_n1024 | pytorch-cuda | 256 | 15 | 2.773 | 4.231e-08 |
| log1p_f32_n1024 | pytorch-cuda | 256 | 15 | 2.302 | 0.0001189 |
| log1p_f64_n1024 | pytorch-cuda | 256 | 15 | 2.869 | 1.108e-07 |
| erf_f32_n1024 | pytorch-cuda | 256 | 15 | 2.526 | 7.914e-05 |
| erf_f64_n1024 | pytorch-cuda | 128 | 15 | 2.016 | 8.536e-08 |
| sin_f32_n1024 | pytorch-cuda | 256 | 15 | 2.447 | 4.348e-05 |
| sin_f64_n1024 | pytorch-cuda | 256 | 15 | 2.382 | 4.153e-08 |
| cos_f32_n1024 | pytorch-cuda | 256 | 15 | 2.288 | 6.342e-05 |
| cos_f64_n1024 | pytorch-cuda | 256 | 15 | 2.411 | 5.905e-08 |
| tanh_f32_n1024 | pytorch-cuda | 256 | 15 | 2.519 | 9.107e-05 |
| tanh_f64_n1024 | pytorch-cuda | 256 | 15 | 2.445 | 4.434e-08 |
| sqrt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.431 | 0 |
| sqrt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.437 | 0 |
| rsqrt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.338 | 6.983e-05 |
| rsqrt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.394 | 6.504e-08 |
| compare_f32_n1024 | pytorch-cuda | 256 | 15 | 2.316 | 0 |
| compare_f64_n1024 | pytorch-cuda | 256 | 15 | 2.329 | 0 |
| where_select_f32_n1024 | pytorch-cuda | 256 | 15 | 3.479 | 0 |
| where_select_f64_n1024 | pytorch-cuda | 256 | 15 | 3.355 | 0 |
| clamp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.984 | 0 |
| clamp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.864 | 0 |
| reduce_sum_f32_n1024 | pytorch-cuda | 128 | 15 | 1.896 | 0.0002742 |
| reduce_sum_f64_n1024 | pytorch-cuda | 256 | 15 | 3.881 | 1.06e-07 |
| reduce_prod_f32_n1024 | pytorch-cuda | 256 | 15 | 3.536 | 0.0003579 |
| reduce_prod_f64_n1024 | pytorch-cuda | 256 | 15 | 3.474 | 4.441e-07 |
| reduce_max_f32_n1024 | pytorch-cuda | 256 | 15 | 3.641 | 0 |
| reduce_max_f64_n1024 | pytorch-cuda | 256 | 15 | 3.614 | 0 |
| reduce_min_f32_n1024 | pytorch-cuda | 256 | 15 | 3.678 | 0 |
| reduce_min_f64_n1024 | pytorch-cuda | 256 | 15 | 3.664 | 0 |
| reduce_sum_squares_f32_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | pytorch-cuda | 256 | 15 | 3.738 | 1.439e-05 |
| reduce_mean_f64_n1024 | pytorch-cuda | 256 | 15 | 3.914 | 6.71e-09 |
| add_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.489 | 0 |
| sub_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| mul_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.546 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.490 | 0 |
| div_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.559 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.010 | 0 |
| rem_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.502 | 0 |
| pow_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.661 | 0.0001577 |
| pow_f64_n1048576 | pytorch-cuda | 2 | 15 | 2.026 | 1.48e-07 |
| maximum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.510 | 0 |
| minimum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.554 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.511 | 0 |
| neg_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.749 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.697 | 0 |
| abs_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.759 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.702 | 0 |
| sign_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.753 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.703 | 0 |
| conj_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.698 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.665 | 0 |
| exp_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.754 | 0.0001272 |
| exp_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.875 | 1.095e-07 |
| expm1_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.761 | 7.915e-05 |
| expm1_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.583 | 7.39e-08 |
| log_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.758 | 7.946e-05 |
| log_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.790 | 4.308e-08 |
| log1p_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.759 | 0.000119 |
| log1p_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.888 | 1.11e-07 |
| erf_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.757 | 7.923e-05 |
| erf_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.111 | 8.536e-08 |
| sin_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.764 | 4.348e-05 |
| sin_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.375 | 4.153e-08 |
| cos_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.765 | 6.342e-05 |
| cos_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.368 | 5.905e-08 |
| tanh_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.755 | 9.135e-05 |
| tanh_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.666 | 4.437e-08 |
| sqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.758 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.932 | 0 |
| rsqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.755 | 6.983e-05 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.963 | 6.504e-08 |
| compare_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.948 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.582 | 0 |
| where_select_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.754 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.592 | 0 |
| clamp_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.811 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.994 | 0 |
| reduce_sum_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.169 | 0.002948 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.076 | 2.961e-06 |
| reduce_prod_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.066 | 0.002623 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.062 | 4.216e-06 |
| reduce_max_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.117 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.668 | 0 |
| reduce_min_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.119 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.667 | 0 |
| reduce_sum_squares_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.076 | 2.947e-06 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.103 | 3.9e-09 |
| add_f32_n1024 | jax-cuda | 32 | 15 | 3.502 | 0 |
| add_f64_n1024 | jax-cuda | 32 | 15 | 3.348 | 0 |
| sub_f32_n1024 | jax-cuda | 32 | 15 | 2.813 | 0 |
| sub_f64_n1024 | jax-cuda | 32 | 15 | 2.699 | 0 |
| mul_f32_n1024 | jax-cuda | 32 | 15 | 2.686 | 0 |
| mul_f64_n1024 | jax-cuda | 32 | 15 | 2.720 | 0 |
| div_f32_n1024 | jax-cuda | 32 | 15 | 2.543 | 4.875e-05 |
| div_f64_n1024 | jax-cuda | 32 | 15 | 2.642 | 0 |
| rem_f32_n1024 | jax-cuda | 32 | 15 | 3.092 | 0 |
| rem_f64_n1024 | jax-cuda | 32 | 15 | 3.499 | 0 |
| pow_f32_n1024 | jax-cuda | 16 | 15 | 2.280 | 0.000117 |
| pow_f64_n1024 | jax-cuda | 16 | 15 | 2.750 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda | 16 | 15 | 2.568 | 0 |
| maximum_f64_n1024 | jax-cuda | 16 | 15 | 2.358 | 0 |
| minimum_f32_n1024 | jax-cuda | 16 | 15 | 2.290 | 0 |
| minimum_f64_n1024 | jax-cuda | 16 | 15 | 2.165 | 0 |
| neg_f32_n1024 | jax-cuda | 16 | 15 | 2.015 | 0 |
| neg_f64_n1024 | jax-cuda | 16 | 15 | 1.928 | 0 |
| abs_f32_n1024 | jax-cuda | 32 | 15 | 2.947 | 0 |
| abs_f64_n1024 | jax-cuda | 32 | 15 | 3.125 | 0 |
| sign_f32_n1024 | jax-cuda | 32 | 15 | 3.216 | 0 |
| sign_f64_n1024 | jax-cuda | 32 | 15 | 3.179 | 0 |
| conj_c32_n1024 | jax-cuda | 32 | 15 | 3.086 | 0 |
| conj_c64_n1024 | jax-cuda | 32 | 15 | 3.017 | 0 |
| exp_f32_n1024 | jax-cuda | 32 | 15 | 2.929 | 0.0001272 |
| exp_f64_n1024 | jax-cuda | 32 | 15 | 3.056 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda | 32 | 15 | 3.105 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda | 32 | 15 | 3.238 | 7.39e-08 |
| log_f32_n1024 | jax-cuda | 32 | 15 | 3.231 | 7.945e-05 |
| log_f64_n1024 | jax-cuda | 16 | 15 | 1.930 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda | 16 | 15 | 2.100 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda | 16 | 15 | 2.269 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda | 8 | 15 | 2.031 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda | 8 | 15 | 2.000 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda | 16 | 15 | 2.748 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda | 16 | 15 | 2.490 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda | 16 | 15 | 2.717 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda | 16 | 15 | 2.605 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda | 16 | 15 | 2.606 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda | 16 | 15 | 2.490 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda | 16 | 15 | 2.410 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda | 16 | 15 | 2.299 | 0 |
| rsqrt_f32_n1024 | jax-cuda | 16 | 15 | 2.814 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda | 16 | 15 | 2.787 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda | 32 | 15 | 3.353 | 0 |
| compare_f64_n1024 | jax-cuda | 32 | 15 | 3.198 | 0 |
| where_select_f32_n1024 | jax-cuda | 16 | 15 | 2.006 | 0 |
| where_select_f64_n1024 | jax-cuda | 16 | 15 | 2.087 | 0 |
| clamp_f32_n1024 | jax-cuda | 32 | 15 | 3.210 | 0 |
| clamp_f64_n1024 | jax-cuda | 32 | 15 | 3.315 | 0 |
| reduce_sum_f32_n1024 | jax-cuda | 32 | 15 | 3.182 | 0.0001139 |
| reduce_sum_f64_n1024 | jax-cuda | 16 | 15 | 2.679 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda | 16 | 15 | 2.842 | 0.0003579 |
| reduce_prod_f64_n1024 | jax-cuda | 16 | 15 | 2.270 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda | 32 | 15 | 3.470 | 0 |
| reduce_max_f64_n1024 | jax-cuda | 16 | 15 | 2.682 | 0 |
| reduce_min_f32_n1024 | jax-cuda | 32 | 15 | 3.172 | 0 |
| reduce_min_f64_n1024 | jax-cuda | 32 | 15 | 3.561 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda | 32 | 15 | 3.184 | 7.204e-06 |
| reduce_mean_f64_n1024 | jax-cuda | 32 | 15 | 3.395 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda | 16 | 15 | 1.769 | 0 |
| add_f64_n1048576 | jax-cuda | 16 | 15 | 1.834 | 0 |
| sub_f32_n1048576 | jax-cuda | 32 | 15 | 3.481 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 3.399 | 0 |
| mul_f32_n1048576 | jax-cuda | 32 | 15 | 3.438 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 3.448 | 0 |
| div_f32_n1048576 | jax-cuda | 32 | 15 | 3.125 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda | 32 | 15 | 3.963 | 0 |
| rem_f32_n1048576 | jax-cuda | 32 | 15 | 3.153 | 0 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 3.131 | 0 |
| pow_f32_n1048576 | jax-cuda | 32 | 15 | 3.887 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda | 2 | 15 | 2.639 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda | 32 | 15 | 3.497 | 0 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 3.212 | 0 |
| minimum_f32_n1048576 | jax-cuda | 32 | 15 | 3.432 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 3.349 | 0 |
| neg_f32_n1048576 | jax-cuda | 32 | 15 | 3.308 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 2.730 | 0 |
| abs_f32_n1048576 | jax-cuda | 32 | 15 | 2.249 | 0 |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 2.122 | 0 |
| sign_f32_n1048576 | jax-cuda | 32 | 15 | 2.289 | 0 |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 2.120 | 0 |
| conj_c32_n1048576 | jax-cuda | 32 | 15 | 2.131 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 1.856 | 0 |
| exp_f32_n1048576 | jax-cuda | 32 | 15 | 2.251 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda | 16 | 15 | 3.264 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda | 32 | 15 | 3.031 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda | 8 | 15 | 2.128 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda | 32 | 15 | 2.804 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda | 8 | 15 | 3.237 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda | 32 | 15 | 3.199 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda | 8 | 15 | 3.408 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda | 16 | 15 | 2.640 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda | 4 | 15 | 2.731 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda | 16 | 15 | 2.151 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda | 16 | 15 | 3.832 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda | 16 | 15 | 1.987 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda | 16 | 15 | 3.832 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda | 32 | 15 | 3.216 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda | 16 | 15 | 3.046 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda | 32 | 15 | 3.003 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 3.305 | 0 |
| rsqrt_f32_n1048576 | jax-cuda | 32 | 15 | 3.292 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda | 32 | 15 | 3.060 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda | 32 | 15 | 2.327 | 0 |
| compare_f64_n1048576 | jax-cuda | 32 | 15 | 2.166 | 0 |
| where_select_f32_n1048576 | jax-cuda | 32 | 15 | 3.293 | 0 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 2.849 | 0 |
| clamp_f32_n1048576 | jax-cuda | 32 | 15 | 2.316 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 2.653 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda | 32 | 15 | 3.929 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda | 16 | 15 | 2.696 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda | 16 | 15 | 2.496 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda | 16 | 15 | 2.595 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda | 16 | 15 | 2.540 | 0 |
| reduce_max_f64_n1048576 | jax-cuda | 16 | 15 | 2.482 | 0 |
| reduce_min_f32_n1048576 | jax-cuda | 16 | 15 | 2.604 | 0 |
| reduce_min_f64_n1048576 | jax-cuda | 16 | 15 | 2.690 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda | 16 | 15 | 2.743 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda | 16 | 15 | 2.375 | 6.5e-09 |
| add_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.605 | 0 |
| add_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.630 | 0 |
| sub_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.568 | 0 |
| sub_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.480 | 0 |
| mul_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.532 | 0 |
| mul_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.463 | 0 |
| div_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.444 | 4.875e-05 |
| div_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.521 | 0 |
| rem_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.786 | 0 |
| rem_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.057 | 0 |
| pow_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.210 | 0.000117 |
| pow_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.332 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.000 | 0 |
| maximum_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.099 | 0 |
| minimum_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.103 | 0 |
| minimum_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.141 | 0 |
| neg_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.015 | 0 |
| neg_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.974 | 0 |
| abs_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.041 | 0 |
| abs_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.818 | 0 |
| sign_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.058 | 0 |
| sign_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.896 | 0 |
| conj_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.005 | 0 |
| conj_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.963 | 0 |
| exp_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.853 | 0.0001272 |
| exp_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.063 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.100 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.169 | 7.39e-08 |
| log_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.192 | 7.945e-05 |
| log_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.259 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.085 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.206 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.851 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.018 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.529 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.548 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.642 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.597 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.386 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.432 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.283 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.206 | 0 |
| rsqrt_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.163 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.072 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.036 | 0 |
| compare_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.032 | 0 |
| where_select_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.223 | 0 |
| where_select_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.226 | 0 |
| clamp_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.262 | 0 |
| clamp_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.116 | 0 |
| reduce_sum_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.970 | 0.0001139 |
| reduce_sum_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.877 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.188 | 0.0003579 |
| reduce_prod_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.466 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.150 | 0 |
| reduce_max_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.480 | 0 |
| reduce_min_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.917 | 0 |
| reduce_min_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.199 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.157 | 7.204e-06 |
| reduce_mean_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.114 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.132 | 0 |
| add_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.926 | 0 |
| sub_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.164 | 0 |
| sub_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.660 | 0 |
| mul_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.343 | 0 |
| mul_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.669 | 0 |
| div_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.389 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.907 | 0 |
| rem_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.696 | 0 |
| rem_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.666 | 0 |
| pow_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.470 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda-jit | 2 | 15 | 2.555 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.950 | 0 |
| maximum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.674 | 0 |
| minimum_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.363 | 0 |
| minimum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.684 | 0 |
| neg_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.269 | 0 |
| neg_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.256 | 0 |
| abs_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.305 | 0 |
| abs_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.204 | 0 |
| sign_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.243 | 0 |
| sign_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.118 | 0 |
| conj_c32_n1048576 | jax-cuda-jit | 32 | 15 | 2.153 | 0 |
| conj_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.870 | 0 |
| exp_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.269 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.264 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.536 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.116 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.696 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.194 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.112 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.365 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.991 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda-jit | 4 | 15 | 2.693 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.089 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.027 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.255 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.795 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.103 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.048 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.091 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.290 | 0 |
| rsqrt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.020 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.927 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.071 | 0 |
| compare_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.185 | 0 |
| where_select_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.066 | 0 |
| where_select_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.067 | 0 |
| clamp_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.206 | 0 |
| clamp_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.048 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.843 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.480 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.438 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.394 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.333 | 0 |
| reduce_max_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.365 | 0 |
| reduce_min_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.345 | 0 |
| reduce_min_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.416 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.568 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.718 | 6.5e-09 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: benchmarks/gpu/isolated_ops.yaml
timestamp: '2026-10-13T10:24:19Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 8b13220c1e0397a29a27be65557c24105352a35b
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: c3cb4e87a0640e621c7aa098a9bb3fd560bd9ef3
  dirty: false
environment:
  hostname: 33ed5e46905b
  os: Linux-6.8.0-146-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_ROOT: /opt/openblas
    OPENBLAS_NUM_THREADS: '1'
    CUDA_HOME: /usr/local/cuda
    CUDA_PATH: /usr/local/cuda
    TENFERRO_CUTENSOR_PATH: /usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
    BENCH_RUNS: '15'
    BENCH_WARMUPS: '3'
    USE_CUDA: '1'
timing_policy:
  version: 2
  scope: operation execution and completion synchronization
  setup: fixtures, H2D, runtime/session/handles, plans and compilation outside every
    timer
  cleanup: returned outputs retained until after timer stops
  minimum_untimed_priming_runs: 1
  short_operation_sampling: adaptive batches target 2 ms, at most 4096 operations
    and 256 MiB retained output; output-reuse destinations prepared before timing
  trace_execution: 'tenferro trace excluded: run_prepared cannot borrow the entered
    session; JAX JIT compilation outside timing'
  setup_diagnostics: false
blas:
  implementation: none
python_backends:
  pytorch:
    available: true
    version: 2.12.0+cu130
    file: /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/__init__.py
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
    library: /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
    linked_libraries:
    - /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1
  jax:
    available: true
    version: 0.10.1
    jaxlib_version: 0.10.1
    backend: gpu
    file: /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/jaxlib/__init__.py
    provider: xla_gpu
    dot_backend: xla_gpu
    lapack_provider: none_detected
    library: /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/jaxlib/_ifrt_proxy.so
    linked_libraries: []
julia:
  available: false
  reason: julia not found on PATH
cuda:
  device_ordinal: 0
  device_name: NVIDIA GeForce RTX 3060
  device_uuid: GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e
  device_memory_bytes: 12884901888
  driver_version: 580.178.04
  runtime_version: '12.9'
  cuda_version: '13.0'
  cudnn_version: '92700'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e
  cuda_visible_devices: GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh isolated_ops
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 124
  selected_cases: 124
  sha256:
    .devcontainer/cuda/devcontainer.json: 6d3b10caecad15c0a513120ce6282ebb13c65be81c40e29f78a329439c50f083
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: f29f1ad0dd99e8e5d24fffc68b6fe30673b0c42d34b5fe4a5557b7f774c2af15
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 86f2856411d902d27dcc5c8e66415eb933e057e04b9fad86be55569004572a36
    src/bin/benchmark_gpu_public_api/structural.rs: b5d780e867a5748be4f29f245d367833a8cd5ee0ebd440926d8b3003b432b0e4
    src/bin/benchmark_gpu_public_api/linalg.rs: 7d4e29355a973f3c60c2af53fd6ceac597d85fe21f9ab5f6bf164d0afe0a6566
    src/bin/benchmark_gpu_public_api/reuse.rs: ad72b565373904275128d7beb8673bada08d922e8f310be6cedb0d2d8187898f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 24cd5cec53eaecc8b6b4a22a1cd69e8cbe430ab6c9945d48df4c03d3b2655999
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 0a9e8d89405fc802e392e047d8b7c587e38ab1850d831735dd53fe13e87dabd3
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/isolated_ops/20261013_102419/fixtures/manifest.json: f9dceb1bda660eb7a0e78a78de2e7b3c95eef9afbae30d0cea8ff499ab4089c8
    target/release/benchmark_gpu_public_api: 91380afb8faffb97ca96d131ef98cc8e2656aababe594e82d65a07c9bb1bed38
  reference_numpy_version: 2.4.2
```
