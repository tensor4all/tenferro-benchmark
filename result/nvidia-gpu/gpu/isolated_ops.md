# gpu/isolated_ops: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261013_151634/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_f32_n1024 | 13.632 (0.517) | 22.338 (1.605) | 8.764 (0.298) | 89.695 (3.030) | 85.793 (2.576) |
| add_f64_n1024 | 16.746 (1.157) | 27.076 (2.985) | 8.844 (0.201) | 88.352 (12.821) | 83.070 (7.998) |
| sub_f32_n1024 | 16.736 (0.950) | 27.919 (1.375) | 8.842 (0.081) | 89.385 (10.502) | 80.501 (9.382) |
| sub_f64_n1024 | 17.894 (0.671) | 30.147 (1.745) | 8.773 (0.087) | 85.992 (6.330) | 80.442 (7.215) |
| mul_f32_n1024 | 18.153 (0.246) | 34.056 (1.246) | 8.406 (0.249) | 85.698 (9.723) | 80.624 (6.351) |
| mul_f64_n1024 | 18.248 (0.149) | 33.563 (1.786) | 8.543 (0.116) | 85.696 (12.415) | 81.425 (7.319) |
| div_f32_n1024 | 18.567 (0.318) | 33.094 (1.467) | 8.731 (0.146) | 80.036 (5.935) | 80.147 (3.244) |
| div_f64_n1024 | 18.148 (0.182) | 34.034 (0.620) | 8.430 (0.125) | 83.062 (6.723) | 82.311 (4.175) |
| rem_f32_n1024 | 17.435 (0.118) | 206.045 (11.778) | 8.627 (0.555) | 99.581 (8.513) | 96.500 (8.093) |
| rem_f64_n1024 | 17.232 (0.153) | 237.932 (23.547) | 8.824 (0.185) | 102.773 (21.118) | 125.532 (8.307) |
| pow_f32_n1024 | 17.348 (0.140) | 204.745 (18.789) | 8.898 (0.106) | 153.684 (14.335) | 135.314 (7.392) |
| pow_f64_n1024 | 18.178 (0.238) | 245.949 (23.753) | 30.933 (0.011) | 201.643 (34.468) | 167.724 (17.702) |
| maximum_f32_n1024 | 16.998 (0.158) | 294.042 (12.193) | 8.763 (0.102) | 166.089 (37.336) | 148.144 (25.529) |
| maximum_f64_n1024 | 17.261 (0.120) | 227.297 (17.758) | 8.780 (0.341) | 146.558 (15.709) | 134.791 (20.118) |
| minimum_f32_n1024 | 17.115 (0.213) | 270.960 (8.211) | 8.656 (0.129) | 140.068 (17.033) | 125.693 (18.428) |
| minimum_f64_n1024 | 17.170 (0.135) | 284.942 (10.023) | 8.428 (0.153) | 122.056 (12.173) | 106.883 (13.175) |
| neg_f32_n1024 | 13.338 (1.679) | 20.178 (0.257) | 7.940 (0.058) | 107.202 (13.223) | 95.447 (5.359) |
| neg_f64_n1024 | 12.529 (1.598) | 21.466 (0.542) | 8.377 (0.689) | 106.503 (9.791) | 96.992 (3.558) |
| abs_f32_n1024 | 10.971 (1.098) | 149.821 (20.830) | 10.901 (0.644) | 97.996 (9.528) | 97.322 (9.621) |
| abs_f64_n1024 | 10.875 (1.424) | 145.617 (4.724) | 10.300 (0.750) | 97.638 (8.265) | 93.817 (0.608) |
| sign_f32_n1024 | 10.256 (1.445) | 163.790 (3.965) | 7.965 (0.046) | 95.374 (7.249) | 97.127 (4.272) |
| sign_f64_n1024 | 9.907 (1.267) | 170.904 (2.605) | 8.335 (0.279) | 99.238 (10.495) | 102.069 (12.076) |
| conj_c32_n1024 | 9.682 (1.173) | 177.017 (4.657) | 10.014 (0.144) | 97.781 (9.036) | 97.143 (5.519) |
| conj_c64_n1024 | 10.974 (0.412) | 141.432 (5.628) | 10.451 (0.067) | 96.531 (7.475) | 91.675 (7.274) |
| exp_f32_n1024 | 11.455 (1.410) | 19.945 (0.261) | 7.909 (0.160) | 97.802 (13.528) | 98.198 (11.291) |
| exp_f64_n1024 | 10.569 (1.525) | 21.468 (0.518) | 8.056 (0.100) | 102.088 (13.968) | 101.023 (10.285) |
| expm1_f32_n1024 | 10.393 (1.382) | 21.270 (0.787) | 8.052 (0.154) | 99.750 (13.697) | 97.597 (12.181) |
| expm1_f64_n1024 | 9.700 (1.199) | 21.576 (0.264) | 8.120 (0.049) | 122.075 (20.757) | 102.565 (15.614) |
| log_f32_n1024 | 9.599 (0.957) | 21.252 (0.670) | 7.890 (0.038) | 140.610 (18.620) | 101.279 (10.035) |
| log_f64_n1024 | 9.252 (0.840) | 22.823 (1.411) | 10.891 (0.007) | 146.231 (17.971) | 99.112 (13.023) |
| log1p_f32_n1024 | 9.202 (1.104) | 21.522 (1.058) | 7.945 (0.091) | 151.255 (26.975) | 100.220 (12.511) |
| log1p_f64_n1024 | 9.192 (0.833) | 22.116 (0.284) | 11.239 (0.017) | 156.589 (7.368) | 107.016 (11.906) |
| erf_f32_n1024 | 8.671 (0.934) | 21.478 (0.842) | 8.533 (0.139) | 246.102 (5.956) | 122.672 (16.147) |
| erf_f64_n1024 | 9.227 (0.881) | 21.751 (0.654) | 15.801 (0.029) | 252.819 (11.464) | 133.504 (29.328) |
| sin_f32_n1024 | 8.728 (0.646) | 21.557 (0.420) | 8.671 (0.394) | 167.811 (14.881) | 141.356 (8.995) |
| sin_f64_n1024 | 9.696 (0.877) | 21.936 (0.518) | 8.781 (0.415) | 161.629 (5.089) | 158.444 (21.950) |
| cos_f32_n1024 | 9.340 (0.762) | 21.730 (0.608) | 7.942 (0.082) | 165.945 (13.807) | 169.797 (18.798) |
| cos_f64_n1024 | 9.311 (1.195) | 21.877 (0.595) | 8.231 (0.135) | 162.819 (8.454) | 165.194 (15.796) |
| tanh_f32_n1024 | 9.226 (0.784) | 21.124 (0.717) | 8.092 (0.077) | 173.447 (18.623) | 150.651 (31.742) |
| tanh_f64_n1024 | 9.273 (1.013) | 21.463 (0.677) | 8.364 (0.034) | 159.989 (24.577) | 106.788 (12.836) |
| sqrt_f32_n1024 | 8.944 (0.982) | 21.651 (0.786) | 8.382 (0.137) | 145.554 (37.535) | 98.898 (7.335) |
| sqrt_f64_n1024 | 9.350 (0.813) | 22.245 (0.761) | 8.288 (0.071) | 138.353 (26.246) | 101.215 (14.349) |
| rsqrt_f32_n1024 | 8.935 (0.898) | 21.740 (0.630) | 7.927 (0.128) | 247.903 (4.497) | 96.070 (9.247) |
| rsqrt_f64_n1024 | 8.841 (1.011) | 22.657 (0.626) | 7.944 (0.042) | 183.752 (60.678) | 100.791 (10.344) |
| compare_f32_n1024 | unsupported | 433.939 (3.581) | 8.024 (0.069) | 136.159 (16.328) | 99.367 (5.546) |
| compare_f64_n1024 | unsupported | 437.788 (54.437) | 7.996 (0.097) | 129.680 (7.097) | 97.429 (2.982) |
| where_select_f32_n1024 | 18.739 (1.763) | 459.575 (24.317) | 11.847 (0.064) | 137.870 (9.929) | 101.838 (8.670) |
| where_select_f64_n1024 | 18.609 (2.255) | 518.524 (37.714) | 12.244 (0.195) | 134.922 (16.563) | 101.285 (10.904) |
| clamp_f32_n1024 | 51.944 (0.564) | 1024.040 (49.559) | 10.116 (0.207) | 125.927 (10.912) | 101.169 (9.696) |
| clamp_f64_n1024 | 51.981 (3.369) | 1064.645 (58.443) | 10.376 (0.158) | 106.792 (15.502) | 103.803 (12.049) |
| reduce_sum_f32_n1024 | 22.363 (0.156) | 258.702 (1.771) | 13.615 (0.182) | 156.867 (3.262) | 151.653 (20.979) |
| reduce_sum_f64_n1024 | 22.590 (0.419) | 256.984 (2.979) | 13.343 (0.242) | 138.851 (5.820) | 172.286 (13.372) |
| reduce_prod_f32_n1024 | 21.325 (0.379) | 214.926 (4.508) | 12.004 (0.881) | 177.221 (16.051) | 166.719 (7.817) |
| reduce_prod_f64_n1024 | 21.316 (1.606) | 219.338 (4.611) | 11.861 (0.259) | 167.068 (8.565) | 167.149 (7.559) |
| reduce_max_f32_n1024 | 21.679 (1.537) | 217.330 (4.376) | 13.264 (0.381) | 186.218 (8.276) | 171.730 (12.272) |
| reduce_max_f64_n1024 | 22.416 (1.860) | 201.274 (7.150) | 12.856 (0.243) | 115.393 (15.213) | 164.960 (4.337) |
| reduce_min_f32_n1024 | 20.335 (1.970) | 221.648 (5.761) | 13.009 (0.558) | 127.335 (5.717) | 143.524 (32.572) |
| reduce_min_f64_n1024 | 22.588 (2.024) | 188.930 (3.177) | 13.213 (0.191) | 96.415 (0.619) | 150.850 (29.064) |
| reduce_sum_squares_f32_n1024 | 21.801 (0.672) | 203.482 (7.139) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1024 | 21.669 (0.352) | 198.451 (4.033) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1024 | unsupported | unsupported | 13.220 (0.159) | 107.427 (8.522) | 141.175 (26.866) |
| reduce_mean_f64_n1024 | unsupported | unsupported | 14.498 (0.525) | 104.027 (0.358) | 145.585 (29.385) |
| add_f32_n1048576 | 47.757 (0.463) | 49.118 (1.200) | 39.746 (0.123) | 112.441 (1.508) | 104.051 (3.075) |
| add_f64_n1048576 | 82.593 (0.360) | 85.701 (0.607) | 77.778 (0.083) | 108.087 (2.614) | 98.363 (2.736) |
| sub_f32_n1048576 | 43.458 (0.053) | 44.120 (0.103) | 39.733 (0.027) | 106.276 (1.936) | 99.859 (1.680) |
| sub_f64_n1048576 | 82.346 (0.104) | 85.113 (0.228) | 77.812 (0.173) | 105.075 (0.910) | 94.054 (1.065) |
| mul_f32_n1048576 | 43.548 (0.062) | 44.200 (0.147) | 39.764 (0.028) | 106.510 (1.135) | 100.281 (1.630) |
| mul_f64_n1048576 | 82.303 (0.106) | 85.254 (0.239) | 77.696 (0.049) | 104.927 (1.172) | 94.233 (0.727) |
| div_f32_n1048576 | 43.593 (0.106) | 44.197 (0.092) | 39.967 (0.026) | 85.906 (10.471) | 85.197 (13.105) |
| div_f64_n1048576 | 94.949 (0.105) | 99.799 (0.524) | 94.290 (0.100) | 129.142 (5.647) | 122.497 (0.673) |
| rem_f32_n1048576 | 42.696 (0.676) | 97.788 (0.343) | 39.896 (0.053) | 85.413 (7.906) | 85.046 (8.567) |
| rem_f64_n1048576 | 80.836 (0.592) | 186.502 (0.379) | 78.182 (0.057) | 83.067 (0.687) | 85.732 (3.692) |
| pow_f32_n1048576 | 42.213 (0.148) | 97.580 (0.485) | 41.551 (0.122) | 97.994 (7.735) | 79.551 (7.741) |
| pow_f64_n1048576 | 1005.032 (1.234) | 1118.097 (1.436) | 1016.682 (0.652) | 1281.850 (17.973) | 1261.743 (14.673) |
| maximum_f32_n1048576 | 42.800 (0.537) | 97.793 (0.226) | 39.892 (0.044) | 124.585 (15.610) | 89.362 (13.801) |
| maximum_f64_n1048576 | 81.696 (0.529) | 186.935 (0.196) | 78.421 (0.057) | 102.089 (4.729) | 84.065 (0.610) |
| minimum_f32_n1048576 | 42.641 (0.475) | 97.737 (0.309) | 39.893 (0.070) | 103.455 (3.908) | 71.594 (1.545) |
| minimum_f64_n1048576 | 81.816 (0.275) | 187.160 (0.359) | 78.410 (0.060) | 97.348 (3.958) | 84.110 (0.551) |
| neg_f32_n1048576 | 29.473 (0.153) | 29.556 (0.217) | 27.322 (0.058) | 77.855 (5.908) | 71.006 (2.647) |
| neg_f64_n1048576 | 56.092 (0.484) | 56.746 (0.274) | 53.034 (0.141) | 75.696 (1.912) | 67.072 (2.574) |
| abs_f32_n1048576 | 29.223 (0.300) | 56.542 (0.210) | 27.416 (0.057) | 70.393 (2.374) | 74.820 (2.640) |
| abs_f64_n1048576 | 56.189 (0.563) | 107.104 (0.238) | 53.111 (0.052) | 66.840 (3.615) | 66.821 (0.737) |
| sign_f32_n1048576 | 29.565 (0.357) | 56.121 (0.248) | 27.378 (0.110) | 69.791 (2.443) | 71.008 (1.013) |
| sign_f64_n1048576 | 56.556 (0.447) | 107.157 (0.233) | 53.382 (0.235) | 64.149 (1.235) | 67.092 (0.585) |
| conj_c32_n1048576 | 56.130 (0.369) | 107.598 (0.081) | 52.988 (0.043) | 65.363 (1.536) | 66.844 (0.884) |
| conj_c64_n1048576 | 109.929 (0.276) | 209.219 (0.223) | 104.063 (0.095) | 117.358 (3.316) | 115.559 (2.223) |
| exp_f32_n1048576 | 29.793 (0.311) | 29.768 (0.416) | 27.388 (0.044) | 72.054 (3.111) | 70.763 (1.520) |
| exp_f64_n1048576 | 184.609 (0.850) | 185.930 (0.290) | 180.521 (0.177) | 203.991 (2.555) | 202.379 (2.917) |
| expm1_f32_n1048576 | 29.697 (0.534) | 29.818 (0.205) | 27.532 (0.134) | 98.197 (14.807) | 88.950 (16.835) |
| expm1_f64_n1048576 | 227.957 (1.107) | 230.140 (1.180) | 224.716 (0.142) | 263.272 (2.727) | 265.125 (5.985) |
| log_f32_n1048576 | 29.791 (0.256) | 29.527 (0.152) | 27.470 (0.025) | 99.367 (10.323) | 81.809 (11.664) |
| log_f64_n1048576 | 352.017 (0.443) | 355.148 (0.502) | 350.128 (0.103) | 393.410 (2.641) | 389.873 (7.116) |
| log1p_f32_n1048576 | 29.499 (0.367) | 29.641 (0.240) | 27.471 (0.022) | 100.912 (11.751) | 101.695 (6.150) |
| log1p_f64_n1048576 | 364.412 (1.076) | 367.105 (1.012) | 362.581 (0.198) | 409.285 (13.420) | 403.007 (1.621) |
| erf_f32_n1048576 | 29.702 (0.410) | 29.609 (0.227) | 27.480 (0.064) | 157.136 (28.247) | 102.749 (9.985) |
| erf_f64_n1048576 | 530.219 (1.137) | 533.671 (0.592) | 529.867 (0.218) | 675.901 (23.764) | 654.899 (15.337) |
| sin_f32_n1048576 | 29.618 (0.349) | 29.641 (0.232) | 27.559 (0.045) | 99.191 (16.788) | 128.244 (9.840) |
| sin_f64_n1048576 | 201.212 (1.035) | 203.485 (0.903) | 211.874 (0.252) | 237.837 (0.491) | 239.718 (1.139) |
| cos_f32_n1048576 | 29.725 (0.452) | 29.946 (0.400) | 27.568 (0.053) | 135.732 (16.239) | 119.744 (20.465) |
| cos_f64_n1048576 | 201.483 (1.328) | 202.912 (0.902) | 211.321 (0.130) | 254.831 (1.114) | 238.727 (1.575) |
| tanh_f32_n1048576 | 29.581 (0.222) | 29.545 (0.246) | 27.419 (0.041) | 95.906 (6.680) | 98.229 (4.868) |
| tanh_f64_n1048576 | 171.125 (1.337) | 173.072 (0.893) | 167.284 (0.066) | 192.102 (1.660) | 190.356 (1.397) |
| sqrt_f32_n1048576 | 29.493 (0.366) | 29.448 (0.139) | 27.433 (0.018) | 91.423 (1.994) | 97.202 (2.319) |
| sqrt_f64_n1048576 | 94.262 (0.508) | 95.100 (0.327) | 91.903 (0.072) | 103.290 (0.986) | 103.527 (1.471) |
| rsqrt_f32_n1048576 | 29.463 (0.209) | 29.462 (0.159) | 27.410 (0.089) | 139.317 (6.031) | 96.401 (0.529) |
| rsqrt_f64_n1048576 | 62.945 (0.175) | 64.006 (0.692) | 61.602 (0.121) | 139.390 (0.962) | 92.135 (1.180) |
| compare_f32_n1048576 | 32.274 (0.291) | 88.131 (1.178) | 30.465 (0.189) | 96.838 (0.921) | 99.277 (0.631) |
| compare_f64_n1048576 | 57.186 (0.102) | 164.747 (0.589) | 55.957 (0.027) | 99.028 (0.661) | 101.250 (0.631) |
| where_select_f32_n1048576 | 46.342 (0.185) | 117.674 (0.484) | 43.032 (0.056) | 106.856 (1.011) | 101.546 (3.568) |
| where_select_f64_n1048576 | 85.492 (0.402) | 206.521 (0.250) | 80.972 (0.187) | 88.036 (3.527) | 95.125 (1.639) |
| clamp_f32_n1048576 | 87.787 (1.307) | 528.311 (11.993) | 28.296 (0.018) | 77.242 (1.265) | 101.138 (0.654) |
| clamp_f64_n1048576 | 163.332 (1.399) | 524.279 (21.423) | 62.210 (0.021) | 82.549 (1.242) | 97.336 (5.003) |
| reduce_sum_f32_n1048576 | 40.670 (0.290) | 68.928 (0.601) | 16.276 (0.025) | 126.189 (1.529) | 118.669 (18.049) |
| reduce_sum_f64_n1048576 | 47.954 (0.599) | 95.475 (0.290) | 32.426 (0.038) | 156.710 (2.738) | 152.996 (1.656) |
| reduce_prod_f32_n1048576 | 44.551 (0.317) | 69.652 (0.981) | 16.160 (0.050) | 149.809 (1.886) | 149.391 (2.976) |
| reduce_prod_f64_n1048576 | 51.275 (0.584) | 95.393 (0.381) | 32.200 (0.023) | 167.529 (3.709) | 154.428 (1.425) |
| reduce_max_f32_n1048576 | 65.341 (0.593) | 73.909 (0.200) | 16.561 (0.068) | 152.715 (11.527) | 153.599 (4.518) |
| reduce_max_f64_n1048576 | 78.719 (0.681) | 112.632 (0.187) | 41.662 (0.015) | 156.246 (2.305) | 149.992 (1.923) |
| reduce_min_f32_n1048576 | 65.256 (0.550) | 73.958 (0.272) | 16.543 (0.050) | 189.006 (46.740) | 157.079 (16.498) |
| reduce_min_f64_n1048576 | 78.741 (0.536) | 112.696 (0.225) | 41.658 (0.016) | 206.310 (28.652) | 162.908 (4.641) |
| reduce_sum_squares_f32_n1048576 | 43.529 (0.381) | 73.074 (0.509) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1048576 | 53.483 (0.495) | 96.634 (0.370) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1048576 | unsupported | unsupported | 16.274 (0.030) | 208.817 (40.305) | 170.364 (23.156) |
| reduce_mean_f64_n1048576 | unsupported | unsupported | 32.869 (0.028) | 172.452 (25.292) | 175.874 (21.848) |
| reduce_sum_full_f32_n1024 | 24.068 (1.652) | 218.731 (16.613) | 11.641 (0.151) | 201.495 (45.331) | 171.915 (8.608) |
| reduce_prod_full_f32_n1024 | 21.638 (1.092) | 253.150 (8.801) | 10.300 (0.153) | 162.781 (2.336) | 159.483 (6.490) |
| reduce_max_full_f32_n1024 | 17.510 (1.336) | 261.228 (3.567) | 11.476 (0.124) | 186.012 (10.536) | 169.369 (3.010) |
| reduce_min_full_f32_n1024 | 13.139 (0.272) | 260.112 (9.687) | 11.581 (0.369) | 182.791 (23.683) | 168.820 (23.378) |
| compare_eq_f32_n1024 | unsupported | 529.964 (48.381) | 8.172 (0.389) | 155.088 (26.306) | 151.530 (26.043) |
| compare_lt_f32_n1024 | unsupported | 567.358 (54.529) | 8.271 (0.168) | 133.145 (13.925) | 135.725 (9.675) |
| compare_le_f32_n1024 | unsupported | 556.182 (38.974) | 8.680 (0.324) | 131.843 (11.490) | 125.470 (16.313) |
| compare_ge_f32_n1024 | unsupported | 601.660 (37.646) | 8.216 (0.197) | 105.007 (15.036) | 97.795 (13.090) |
| reduce_sum_full_f64_n1024 | 22.257 (0.179) | 196.568 (3.209) | 11.113 (0.303) | 168.669 (8.271) | 97.328 (7.169) |
| reduce_prod_full_f64_n1024 | 21.794 (1.686) | 195.241 (2.182) | 10.022 (0.517) | 131.845 (9.584) | 145.321 (2.396) |
| reduce_max_full_f64_n1024 | 19.447 (1.280) | 195.336 (2.982) | 11.067 (0.279) | 109.031 (16.295) | 132.508 (3.440) |
| reduce_min_full_f64_n1024 | 16.485 (0.893) | 196.843 (1.018) | 11.379 (0.096) | 88.356 (1.726) | 87.562 (0.946) |
| compare_eq_f64_n1024 | unsupported | 529.452 (67.412) | 8.032 (0.170) | 86.153 (0.949) | 97.021 (4.655) |
| compare_lt_f64_n1024 | unsupported | 495.603 (5.941) | 8.414 (0.200) | 87.541 (5.361) | 99.330 (7.406) |
| compare_le_f64_n1024 | unsupported | 493.777 (14.038) | 8.617 (0.061) | 89.901 (5.240) | 98.062 (2.970) |
| compare_ge_f64_n1024 | unsupported | 494.653 (27.117) | 8.531 (0.154) | 96.882 (4.163) | 97.148 (6.180) |
| reduce_sum_full_f32_n1048576 | 2531.256 (3.856) | 2648.025 (14.479) | 18.455 (0.043) | 105.832 (18.542) | 156.005 (5.667) |
| reduce_prod_full_f32_n1048576 | 2521.403 (4.051) | 2617.686 (10.354) | 18.425 (0.041) | 104.250 (14.223) | 162.608 (2.204) |
| reduce_max_full_f32_n1048576 | 9775.548 (1.758) | 9864.613 (3.996) | 18.755 (0.065) | 126.517 (22.371) | 163.300 (5.426) |
| reduce_min_full_f32_n1048576 | 9775.366 (2.585) | 9865.783 (5.580) | 18.866 (0.031) | 164.637 (29.287) | 170.102 (22.400) |
| compare_eq_f32_n1048576 | 33.101 (0.340) | 88.150 (0.665) | 30.328 (0.073) | 101.184 (10.754) | 123.958 (11.546) |
| compare_lt_f32_n1048576 | 33.344 (0.165) | 87.977 (0.248) | 30.282 (0.074) | 97.355 (0.431) | 99.584 (1.219) |
| compare_le_f32_n1048576 | 33.330 (0.250) | 88.023 (0.344) | 30.311 (0.045) | 97.736 (0.517) | 98.678 (0.520) |
| compare_ge_f32_n1048576 | 33.235 (0.326) | 87.689 (0.626) | 30.318 (0.082) | 97.415 (1.755) | 94.486 (2.294) |
| reduce_sum_full_f64_n1048576 | 3029.004 (6.415) | 3094.611 (1.615) | 37.445 (0.029) | 120.551 (15.889) | 116.969 (10.501) |
| reduce_prod_full_f64_n1048576 | 3022.743 (6.283) | 3095.502 (1.619) | 37.530 (0.158) | 145.635 (5.614) | 160.395 (7.079) |
| reduce_max_full_f64_n1048576 | 12422.772 (9.079) | 12537.915 (6.308) | 44.585 (0.053) | 165.673 (9.458) | 163.019 (17.507) |
| reduce_min_full_f64_n1048576 | 12420.187 (2.567) | 12533.115 (6.306) | 44.599 (0.031) | 129.586 (2.225) | 136.263 (2.038) |
| compare_eq_f64_n1048576 | 57.219 (0.168) | 164.386 (0.151) | 56.712 (0.034) | 102.188 (4.693) | 104.205 (15.940) |
| compare_lt_f64_n1048576 | 57.273 (0.116) | 164.461 (0.408) | 55.993 (0.035) | 99.856 (1.103) | 95.979 (2.524) |
| compare_le_f64_n1048576 | 57.301 (0.206) | 164.429 (0.540) | 55.979 (0.060) | 98.858 (1.723) | 91.561 (2.885) |
| compare_ge_f64_n1048576 | 57.185 (0.208) | 164.756 (0.518) | 55.953 (0.030) | 100.136 (0.714) | 92.164 (2.536) |
| reduce_mean_full_f32_n1024 | unsupported | unsupported | 11.797 (0.192) | 97.293 (10.736) | 98.752 (9.022) |
| reduce_sum_squares_full_f32_n1024 | 11.214 (0.126) | 204.248 (10.876) | 15.042 (0.378) | 152.347 (3.279) | 145.211 (2.815) |
| reduce_mean_full_f64_n1024 | unsupported | unsupported | 11.686 (0.150) | 155.244 (7.450) | 104.239 (12.661) |
| reduce_sum_squares_full_f64_n1024 | 9.223 (0.364) | 218.717 (19.977) | 14.736 (0.269) | 159.304 (3.750) | 98.070 (6.362) |
| reduce_mean_full_f32_n1048576 | unsupported | unsupported | 18.453 (0.064) | 184.669 (21.639) | 126.561 (6.909) |
| reduce_sum_squares_full_f32_n1048576 | 2548.037 (5.595) | 2674.131 (10.729) | 17.659 (0.084) | 103.909 (11.957) | 166.666 (7.927) |
| reduce_mean_full_f64_n1048576 | unsupported | unsupported | 37.480 (0.162) | 147.874 (8.622) | 116.415 (12.762) |
| reduce_sum_squares_full_f64_n1048576 | 3292.989 (2.791) | 3372.057 (4.991) | 33.237 (0.020) | 108.441 (9.705) | 106.099 (1.282) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 1.745 | 0 |
| add_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.143 | 0 |
| sub_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.142 | 0 |
| sub_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.290 | 0 |
| mul_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.324 | 0 |
| mul_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.336 | 0 |
| div_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.377 | 0 |
| div_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.323 | 0 |
| rem_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.232 | 0 |
| rem_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.206 | 0 |
| pow_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.221 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.327 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.176 | 0 |
| maximum_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.209 | 0 |
| minimum_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.191 | 0 |
| minimum_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.198 | 0 |
| neg_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.414 | 0 |
| neg_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.208 | 0 |
| abs_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.809 | 0 |
| abs_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.784 | 0 |
| sign_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.626 | 0 |
| sign_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.536 | 0 |
| conj_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.478 | 0 |
| conj_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.405 | 0 |
| exp_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.933 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.706 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.661 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.483 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.457 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.369 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.356 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.353 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.220 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.362 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.234 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.482 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.391 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.384 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.362 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.374 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.290 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.393 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.287 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.263 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; steady-state sampling is not established: consecutive batch durations increase in repeated runs 20261013_101359 and 20261013_102038. Retained as diagnostics pending allocator/lifetime investigation. |
| compare_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; steady-state sampling is not established: consecutive batch durations increase in repeated runs 20261013_101359 and 20261013_102038. Retained as diagnostics pending allocator/lifetime investigation. |
| where_select_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.399 | 0 |
| where_select_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.382 | 0 |
| clamp_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.324 | 0 |
| clamp_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.327 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.431 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.446 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.730 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.728 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.775 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.869 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.603 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.891 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.791 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.774 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.528 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.643 | 0 |
| sub_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.781 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.635 | 0 |
| mul_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.787 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.634 | 0 |
| div_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.790 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.038 | 0 |
| rem_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.733 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.587 | 0 |
| pow_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.702 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.010 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.739 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.614 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.729 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.618 | 0 |
| neg_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.886 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.795 | 0 |
| abs_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.870 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.798 | 0 |
| sign_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.892 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.810 | 0 |
| conj_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.796 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.759 | 0 |
| exp_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.907 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.954 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.901 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.647 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.907 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.816 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.888 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.915 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.901 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.121 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.896 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.219 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.902 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.224 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.893 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.738 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.888 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.016 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.886 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.014 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.066 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.660 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.966 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.736 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.809 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.613 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.603 | 0.001812 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.069 | 2.148e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.851 | 0.005478 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.282 | 3.659e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.091 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.519 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.088 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.520 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.786 | 0.0003528 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.423 | 3.293e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_full_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 0.770 | 0.0001333 |
| reduce_prod_full_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.385 | 0.003334 |
| reduce_max_full_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.241 | 0 |
| reduce_min_full_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.364 | 0 |
| compare_eq_f32_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| compare_lt_f32_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| compare_le_f32_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| compare_ge_f32_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| reduce_sum_full_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.424 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.790 | 0 |
| reduce_max_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.489 | 0 |
| reduce_min_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.110 | 0 |
| compare_eq_f64_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| compare_lt_f64_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| compare_le_f64_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| compare_ge_f64_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| reduce_sum_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.531 | 0.07061 |
| reduce_prod_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.521 | 0.1957 |
| reduce_max_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 9.776 | 0 |
| reduce_min_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 9.775 | 0 |
| compare_eq_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.118 | 0 |
| compare_lt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.134 | 0 |
| compare_le_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.133 | 0 |
| compare_ge_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.127 | 0 |
| reduce_sum_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.029 | 0.001745 |
| reduce_prod_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.023 | 6.451e-05 |
| reduce_max_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 12.423 | 0 |
| reduce_min_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 12.420 | 0 |
| compare_eq_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.662 | 0 |
| compare_lt_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.665 | 0 |
| compare_le_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.667 | 0 |
| compare_ge_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.660 | 0 |
| reduce_mean_full_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.871 | 0.0001766 |
| reduce_mean_full_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.361 | 0 |
| reduce_mean_full_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.548 | 0.001429 |
| reduce_mean_full_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.293 | 9.982e-07 |
| add_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.859 | 0 |
| add_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.466 | 0 |
| sub_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.574 | 0 |
| sub_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.859 | 0 |
| mul_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.180 | 0 |
| mul_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.148 | 0 |
| div_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.118 | 0 |
| div_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.178 | 0 |
| rem_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.297 | 0 |
| rem_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.807 | 0 |
| pow_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.276 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.935 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.352 | 0 |
| maximum_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.637 | 0 |
| minimum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.168 | 0 |
| minimum_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.280 | 0 |
| neg_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.583 | 0 |
| neg_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.748 | 0 |
| abs_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.397 | 0 |
| abs_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.330 | 0 |
| sign_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.621 | 0 |
| sign_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.734 | 0 |
| conj_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.832 | 0 |
| conj_c64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.263 | 0 |
| exp_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.553 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.748 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.723 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.762 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.720 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.921 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.755 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.831 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.749 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.784 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.759 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.808 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.781 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.800 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.704 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.747 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.771 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.847 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.783 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.900 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.472 | 0 |
| compare_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.502 | 0 |
| where_select_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.677 | 0 |
| where_select_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 4.148 | 0 |
| clamp_f32_n1024 | tenferro-cuda-eager | 2 | 15 | 2.048 | 0 |
| clamp_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 4.259 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.070 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.056 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.439 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.509 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.477 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.220 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.546 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.023 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 1.628 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.175 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.572 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.742 | 0 |
| sub_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.824 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.724 | 0 |
| mul_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.829 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.728 | 0 |
| div_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.829 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.194 | 0 |
| rem_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.129 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.984 | 0 |
| pow_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.123 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-eager | 2 | 15 | 2.236 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.129 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.991 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.128 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.995 | 0 |
| neg_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.892 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.816 | 0 |
| abs_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.809 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.427 | 0 |
| sign_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.592 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.429 | 0 |
| conj_c32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.443 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.348 | 0 |
| exp_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.905 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.975 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.908 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.682 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.890 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.841 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.897 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.937 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.895 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.135 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.897 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.256 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.917 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.247 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.891 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.769 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.885 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.043 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.886 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.048 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.820 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.636 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.766 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.304 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.113 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.097 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.206 | 0.001812 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.055 | 2.148e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.229 | 0.005478 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.053 | 3.659e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.365 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.604 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.367 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.606 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.338 | 0.0003528 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.092 | 3.293e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_full_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.500 | 0.0001333 |
| reduce_prod_full_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 4.050 | 0.003334 |
| reduce_max_full_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.090 | 0 |
| reduce_min_full_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.081 | 0 |
| compare_eq_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.120 | 0 |
| compare_lt_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.269 | 0 |
| compare_le_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.225 | 0 |
| compare_ge_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.407 | 0 |
| reduce_sum_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.145 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.124 | 0 |
| reduce_max_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.125 | 0 |
| reduce_min_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.149 | 0 |
| compare_eq_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.118 | 0 |
| compare_lt_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.965 | 0 |
| compare_le_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 1.975 | 0 |
| compare_ge_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.957 | 0 |
| reduce_sum_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.648 | 0.07061 |
| reduce_prod_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.618 | 0.1957 |
| reduce_max_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 9.865 | 0 |
| reduce_min_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 9.866 | 0 |
| compare_eq_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.821 | 0 |
| compare_lt_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.815 | 0 |
| compare_le_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.817 | 0 |
| compare_ge_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.806 | 0 |
| reduce_sum_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.095 | 0.001745 |
| reduce_prod_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.096 | 6.451e-05 |
| reduce_max_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 12.538 | 0 |
| reduce_min_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 12.533 | 0 |
| compare_eq_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.630 | 0 |
| compare_lt_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.631 | 0 |
| compare_le_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.631 | 0 |
| compare_ge_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.636 | 0 |
| reduce_mean_full_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.268 | 0.0001766 |
| reduce_mean_full_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.499 | 0 |
| reduce_mean_full_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.674 | 0.001429 |
| reduce_mean_full_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.372 | 9.982e-07 |
| add_f32_n1024 | pytorch-cuda | 256 | 15 | 2.244 | 0 |
| add_f64_n1024 | pytorch-cuda | 256 | 15 | 2.264 | 0 |
| sub_f32_n1024 | pytorch-cuda | 256 | 15 | 2.264 | 0 |
| sub_f64_n1024 | pytorch-cuda | 256 | 15 | 2.246 | 0 |
| mul_f32_n1024 | pytorch-cuda | 256 | 15 | 2.152 | 0 |
| mul_f64_n1024 | pytorch-cuda | 256 | 15 | 2.187 | 0 |
| div_f32_n1024 | pytorch-cuda | 256 | 15 | 2.235 | 0 |
| div_f64_n1024 | pytorch-cuda | 256 | 15 | 2.158 | 0 |
| rem_f32_n1024 | pytorch-cuda | 256 | 15 | 2.209 | 0 |
| rem_f64_n1024 | pytorch-cuda | 256 | 15 | 2.259 | 0 |
| pow_f32_n1024 | pytorch-cuda | 256 | 15 | 2.278 | 0.000117 |
| pow_f64_n1024 | pytorch-cuda | 128 | 15 | 3.959 | 1.48e-07 |
| maximum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.243 | 0 |
| maximum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.248 | 0 |
| minimum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.216 | 0 |
| minimum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.158 | 0 |
| neg_f32_n1024 | pytorch-cuda | 256 | 15 | 2.033 | 0 |
| neg_f64_n1024 | pytorch-cuda | 256 | 15 | 2.144 | 0 |
| abs_f32_n1024 | pytorch-cuda | 256 | 15 | 2.791 | 0 |
| abs_f64_n1024 | pytorch-cuda | 256 | 15 | 2.637 | 0 |
| sign_f32_n1024 | pytorch-cuda | 256 | 15 | 2.039 | 0 |
| sign_f64_n1024 | pytorch-cuda | 256 | 15 | 2.134 | 0 |
| conj_c32_n1024 | pytorch-cuda | 256 | 15 | 2.564 | 0 |
| conj_c64_n1024 | pytorch-cuda | 256 | 15 | 2.676 | 0 |
| exp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.025 | 0.0001272 |
| exp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.062 | 1.095e-07 |
| expm1_f32_n1024 | pytorch-cuda | 256 | 15 | 2.061 | 7.911e-05 |
| expm1_f64_n1024 | pytorch-cuda | 256 | 15 | 2.079 | 7.39e-08 |
| log_f32_n1024 | pytorch-cuda | 256 | 15 | 2.020 | 7.945e-05 |
| log_f64_n1024 | pytorch-cuda | 256 | 15 | 2.788 | 4.231e-08 |
| log1p_f32_n1024 | pytorch-cuda | 256 | 15 | 2.034 | 0.0001189 |
| log1p_f64_n1024 | pytorch-cuda | 256 | 15 | 2.877 | 1.108e-07 |
| erf_f32_n1024 | pytorch-cuda | 256 | 15 | 2.184 | 7.914e-05 |
| erf_f64_n1024 | pytorch-cuda | 128 | 15 | 2.022 | 8.536e-08 |
| sin_f32_n1024 | pytorch-cuda | 256 | 15 | 2.220 | 4.348e-05 |
| sin_f64_n1024 | pytorch-cuda | 256 | 15 | 2.248 | 4.153e-08 |
| cos_f32_n1024 | pytorch-cuda | 256 | 15 | 2.033 | 6.342e-05 |
| cos_f64_n1024 | pytorch-cuda | 256 | 15 | 2.107 | 5.905e-08 |
| tanh_f32_n1024 | pytorch-cuda | 256 | 15 | 2.071 | 9.107e-05 |
| tanh_f64_n1024 | pytorch-cuda | 256 | 15 | 2.141 | 4.434e-08 |
| sqrt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.146 | 0 |
| sqrt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.122 | 0 |
| rsqrt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.029 | 6.983e-05 |
| rsqrt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.034 | 6.504e-08 |
| compare_f32_n1024 | pytorch-cuda | 256 | 15 | 2.054 | 0 |
| compare_f64_n1024 | pytorch-cuda | 256 | 15 | 2.047 | 0 |
| where_select_f32_n1024 | pytorch-cuda | 256 | 15 | 3.033 | 0 |
| where_select_f64_n1024 | pytorch-cuda | 256 | 15 | 3.135 | 0 |
| clamp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.590 | 0 |
| clamp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.656 | 0 |
| reduce_sum_f32_n1024 | pytorch-cuda | 256 | 15 | 3.485 | 0.0002742 |
| reduce_sum_f64_n1024 | pytorch-cuda | 256 | 15 | 3.416 | 1.06e-07 |
| reduce_prod_f32_n1024 | pytorch-cuda | 256 | 15 | 3.073 | 0.0003579 |
| reduce_prod_f64_n1024 | pytorch-cuda | 256 | 15 | 3.036 | 4.441e-07 |
| reduce_max_f32_n1024 | pytorch-cuda | 256 | 15 | 3.396 | 0 |
| reduce_max_f64_n1024 | pytorch-cuda | 256 | 15 | 3.291 | 0 |
| reduce_min_f32_n1024 | pytorch-cuda | 256 | 15 | 3.330 | 0 |
| reduce_min_f64_n1024 | pytorch-cuda | 256 | 15 | 3.383 | 0 |
| reduce_sum_squares_f32_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | pytorch-cuda | 256 | 15 | 3.384 | 1.439e-05 |
| reduce_mean_f64_n1024 | pytorch-cuda | 256 | 15 | 3.711 | 6.71e-09 |
| add_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.489 | 0 |
| sub_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.543 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.490 | 0 |
| mul_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.545 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| div_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.558 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.017 | 0 |
| rem_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.502 | 0 |
| pow_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.659 | 0.0001577 |
| pow_f64_n1048576 | pytorch-cuda | 2 | 15 | 2.033 | 1.48e-07 |
| maximum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.509 | 0 |
| minimum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.509 | 0 |
| neg_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.749 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.697 | 0 |
| abs_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.755 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.700 | 0 |
| sign_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.752 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.708 | 0 |
| conj_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.696 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.665 | 0 |
| exp_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.753 | 0.0001272 |
| exp_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.888 | 1.095e-07 |
| expm1_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.762 | 7.915e-05 |
| expm1_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.595 | 7.39e-08 |
| log_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.758 | 7.946e-05 |
| log_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.801 | 4.308e-08 |
| log1p_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.758 | 0.000119 |
| log1p_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.901 | 1.11e-07 |
| erf_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.759 | 7.923e-05 |
| erf_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.119 | 8.536e-08 |
| sin_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.764 | 4.348e-05 |
| sin_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.390 | 4.153e-08 |
| cos_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.764 | 6.342e-05 |
| cos_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.381 | 5.905e-08 |
| tanh_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.755 | 9.135e-05 |
| tanh_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.677 | 4.437e-08 |
| sqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.941 | 0 |
| rsqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.754 | 6.983e-05 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.971 | 6.504e-08 |
| compare_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.950 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.581 | 0 |
| where_select_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.754 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.591 | 0 |
| clamp_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.811 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.991 | 0 |
| reduce_sum_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.083 | 0.002948 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.075 | 2.961e-06 |
| reduce_prod_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.068 | 0.002623 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.061 | 4.216e-06 |
| reduce_max_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.120 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.666 | 0 |
| reduce_min_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.118 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.666 | 0 |
| reduce_sum_squares_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.083 | 2.947e-06 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.104 | 3.9e-09 |
| reduce_sum_full_f32_n1024 | pytorch-cuda | 256 | 15 | 2.980 | 0.0002666 |
| reduce_prod_full_f32_n1024 | pytorch-cuda | 256 | 15 | 2.637 | 0.0025 |
| reduce_max_full_f32_n1024 | pytorch-cuda | 256 | 15 | 2.938 | 0 |
| reduce_min_full_f32_n1024 | pytorch-cuda | 256 | 15 | 2.965 | 0 |
| compare_eq_f32_n1024 | pytorch-cuda | 256 | 15 | 2.092 | 0 |
| compare_lt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.117 | 0 |
| compare_le_f32_n1024 | pytorch-cuda | 256 | 15 | 2.222 | 0 |
| compare_ge_f32_n1024 | pytorch-cuda | 256 | 15 | 2.103 | 0 |
| reduce_sum_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.845 | 6.208e-07 |
| reduce_prod_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.566 | 1.663e-06 |
| reduce_max_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.833 | 0 |
| reduce_min_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.913 | 0 |
| compare_eq_f64_n1024 | pytorch-cuda | 256 | 15 | 2.056 | 0 |
| compare_lt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.154 | 0 |
| compare_le_f64_n1024 | pytorch-cuda | 256 | 15 | 2.206 | 0 |
| compare_ge_f64_n1024 | pytorch-cuda | 256 | 15 | 2.184 | 0 |
| reduce_sum_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.362 | 0.00522 |
| reduce_prod_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.358 | 0.5197 |
| reduce_max_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.401 | 0 |
| reduce_min_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.415 | 0 |
| compare_eq_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.882 | 0 |
| compare_lt_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.876 | 0 |
| compare_le_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.880 | 0 |
| compare_ge_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.881 | 0 |
| reduce_sum_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.396 | 0.0004425 |
| reduce_prod_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.402 | 0.0001269 |
| reduce_max_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.853 | 0 |
| reduce_min_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.854 | 0 |
| compare_eq_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.630 | 0 |
| compare_lt_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.584 | 0 |
| compare_le_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.583 | 0 |
| compare_ge_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.581 | 0 |
| reduce_mean_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.020 | 0.0002906 |
| reduce_sum_squares_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.851 | 0 |
| reduce_mean_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.992 | 8.614e-07 |
| reduce_sum_squares_full_f64_n1024 | pytorch-cuda | 128 | 15 | 1.886 | 0 |
| reduce_mean_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.362 | 0.00522 |
| reduce_sum_squares_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.260 | 0 |
| reduce_mean_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.399 | 0.001223 |
| reduce_sum_squares_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.127 | 1.664e-07 |
| add_f32_n1024 | jax-cuda | 32 | 15 | 2.870 | 0 |
| add_f64_n1024 | jax-cuda | 32 | 15 | 2.827 | 0 |
| sub_f32_n1024 | jax-cuda | 32 | 15 | 2.860 | 0 |
| sub_f64_n1024 | jax-cuda | 32 | 15 | 2.752 | 0 |
| mul_f32_n1024 | jax-cuda | 32 | 15 | 2.742 | 0 |
| mul_f64_n1024 | jax-cuda | 32 | 15 | 2.742 | 0 |
| div_f32_n1024 | jax-cuda | 32 | 15 | 2.561 | 4.875e-05 |
| div_f64_n1024 | jax-cuda | 32 | 15 | 2.658 | 0 |
| rem_f32_n1024 | jax-cuda | 32 | 15 | 3.187 | 0 |
| rem_f64_n1024 | jax-cuda | 32 | 15 | 3.289 | 0 |
| pow_f32_n1024 | jax-cuda | 16 | 15 | 2.459 | 0.000117 |
| pow_f64_n1024 | jax-cuda | 16 | 15 | 3.226 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda | 16 | 15 | 2.657 | 0 |
| maximum_f64_n1024 | jax-cuda | 16 | 15 | 2.345 | 0 |
| minimum_f32_n1024 | jax-cuda | 16 | 15 | 2.241 | 0 |
| minimum_f64_n1024 | jax-cuda | 16 | 15 | 1.953 | 0 |
| neg_f32_n1024 | jax-cuda | 32 | 15 | 3.430 | 0 |
| neg_f64_n1024 | jax-cuda | 32 | 15 | 3.408 | 0 |
| abs_f32_n1024 | jax-cuda | 32 | 15 | 3.136 | 0 |
| abs_f64_n1024 | jax-cuda | 32 | 15 | 3.124 | 0 |
| sign_f32_n1024 | jax-cuda | 32 | 15 | 3.052 | 0 |
| sign_f64_n1024 | jax-cuda | 32 | 15 | 3.176 | 0 |
| conj_c32_n1024 | jax-cuda | 32 | 15 | 3.129 | 0 |
| conj_c64_n1024 | jax-cuda | 32 | 15 | 3.089 | 0 |
| exp_f32_n1024 | jax-cuda | 32 | 15 | 3.130 | 0.0001272 |
| exp_f64_n1024 | jax-cuda | 32 | 15 | 3.267 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda | 32 | 15 | 3.192 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda | 16 | 15 | 1.953 | 7.39e-08 |
| log_f32_n1024 | jax-cuda | 16 | 15 | 2.250 | 7.945e-05 |
| log_f64_n1024 | jax-cuda | 16 | 15 | 2.340 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda | 16 | 15 | 2.420 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda | 16 | 15 | 2.505 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda | 8 | 15 | 1.969 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda | 8 | 15 | 2.023 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda | 16 | 15 | 2.685 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda | 16 | 15 | 2.586 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda | 16 | 15 | 2.655 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda | 16 | 15 | 2.605 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda | 16 | 15 | 2.775 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda | 16 | 15 | 2.560 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda | 16 | 15 | 2.329 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda | 16 | 15 | 2.214 | 0 |
| rsqrt_f32_n1024 | jax-cuda | 8 | 15 | 1.983 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda | 16 | 15 | 2.940 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda | 16 | 15 | 2.179 | 0 |
| compare_f64_n1024 | jax-cuda | 16 | 15 | 2.075 | 0 |
| where_select_f32_n1024 | jax-cuda | 16 | 15 | 2.206 | 0 |
| where_select_f64_n1024 | jax-cuda | 16 | 15 | 2.159 | 0 |
| clamp_f32_n1024 | jax-cuda | 16 | 15 | 2.015 | 0 |
| clamp_f64_n1024 | jax-cuda | 32 | 15 | 3.417 | 0 |
| reduce_sum_f32_n1024 | jax-cuda | 16 | 15 | 2.510 | 0.0002742 |
| reduce_sum_f64_n1024 | jax-cuda | 16 | 15 | 2.222 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda | 16 | 15 | 2.836 | 0.0003579 |
| reduce_prod_f64_n1024 | jax-cuda | 16 | 15 | 2.673 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda | 16 | 15 | 2.979 | 0 |
| reduce_max_f64_n1024 | jax-cuda | 32 | 15 | 3.693 | 0 |
| reduce_min_f32_n1024 | jax-cuda | 16 | 15 | 2.037 | 0 |
| reduce_min_f64_n1024 | jax-cuda | 32 | 15 | 3.085 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda | 32 | 15 | 3.438 | 1.439e-05 |
| reduce_mean_f64_n1024 | jax-cuda | 32 | 15 | 3.329 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda | 16 | 15 | 1.799 | 0 |
| add_f64_n1048576 | jax-cuda | 16 | 15 | 1.729 | 0 |
| sub_f32_n1048576 | jax-cuda | 32 | 15 | 3.401 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 3.362 | 0 |
| mul_f32_n1048576 | jax-cuda | 32 | 15 | 3.408 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 3.358 | 0 |
| div_f32_n1048576 | jax-cuda | 32 | 15 | 2.749 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda | 16 | 15 | 2.066 | 0 |
| rem_f32_n1048576 | jax-cuda | 32 | 15 | 2.733 | 0 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 2.658 | 0 |
| pow_f32_n1048576 | jax-cuda | 32 | 15 | 3.136 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda | 2 | 15 | 2.564 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda | 16 | 15 | 1.993 | 0 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 3.267 | 0 |
| minimum_f32_n1048576 | jax-cuda | 32 | 15 | 3.311 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 3.115 | 0 |
| neg_f32_n1048576 | jax-cuda | 32 | 15 | 2.491 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 2.422 | 0 |
| abs_f32_n1048576 | jax-cuda | 32 | 15 | 2.253 | 0 |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 2.139 | 0 |
| sign_f32_n1048576 | jax-cuda | 32 | 15 | 2.233 | 0 |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 2.053 | 0 |
| conj_c32_n1048576 | jax-cuda | 32 | 15 | 2.092 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 1.878 | 0 |
| exp_f32_n1048576 | jax-cuda | 32 | 15 | 2.306 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda | 16 | 15 | 3.264 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda | 32 | 15 | 3.142 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda | 8 | 15 | 2.106 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda | 32 | 15 | 3.180 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda | 8 | 15 | 3.147 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda | 32 | 15 | 3.229 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda | 8 | 15 | 3.274 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda | 16 | 15 | 2.514 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda | 4 | 15 | 2.704 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda | 32 | 15 | 3.174 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda | 16 | 15 | 3.805 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda | 16 | 15 | 2.172 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda | 8 | 15 | 2.039 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda | 32 | 15 | 3.069 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda | 16 | 15 | 3.074 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda | 32 | 15 | 2.926 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 3.305 | 0 |
| rsqrt_f32_n1048576 | jax-cuda | 16 | 15 | 2.229 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda | 16 | 15 | 2.230 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda | 32 | 15 | 3.099 | 0 |
| compare_f64_n1048576 | jax-cuda | 32 | 15 | 3.169 | 0 |
| where_select_f32_n1048576 | jax-cuda | 32 | 15 | 3.419 | 0 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 2.817 | 0 |
| clamp_f32_n1048576 | jax-cuda | 32 | 15 | 2.472 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 2.642 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda | 16 | 15 | 2.019 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda | 16 | 15 | 2.507 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda | 16 | 15 | 2.397 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda | 16 | 15 | 2.680 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda | 16 | 15 | 2.443 | 0 |
| reduce_max_f64_n1048576 | jax-cuda | 16 | 15 | 2.500 | 0 |
| reduce_min_f32_n1048576 | jax-cuda | 16 | 15 | 3.024 | 0 |
| reduce_min_f64_n1048576 | jax-cuda | 16 | 15 | 3.301 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda | 16 | 15 | 3.341 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda | 16 | 15 | 2.759 | 6.5e-09 |
| reduce_sum_full_f32_n1024 | jax-cuda | 16 | 15 | 3.224 | 0 |
| reduce_prod_full_f32_n1024 | jax-cuda | 16 | 15 | 2.604 | 0.001072 |
| reduce_max_full_f32_n1024 | jax-cuda | 16 | 15 | 2.976 | 0 |
| reduce_min_full_f32_n1024 | jax-cuda | 16 | 15 | 2.925 | 0 |
| compare_eq_f32_n1024 | jax-cuda | 16 | 15 | 2.481 | 0 |
| compare_lt_f32_n1024 | jax-cuda | 16 | 15 | 2.130 | 0 |
| compare_le_f32_n1024 | jax-cuda | 16 | 15 | 2.109 | 0 |
| compare_ge_f32_n1024 | jax-cuda | 32 | 15 | 3.360 | 0 |
| reduce_sum_full_f64_n1024 | jax-cuda | 16 | 15 | 2.699 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | jax-cuda | 16 | 15 | 2.110 | 1.774e-06 |
| reduce_max_full_f64_n1024 | jax-cuda | 32 | 15 | 3.489 | 0 |
| reduce_min_full_f64_n1024 | jax-cuda | 32 | 15 | 2.827 | 0 |
| compare_eq_f64_n1024 | jax-cuda | 32 | 15 | 2.757 | 0 |
| compare_lt_f64_n1024 | jax-cuda | 32 | 15 | 2.801 | 0 |
| compare_le_f64_n1024 | jax-cuda | 32 | 15 | 2.877 | 0 |
| compare_ge_f64_n1024 | jax-cuda | 32 | 15 | 3.100 | 0 |
| reduce_sum_full_f32_n1048576 | jax-cuda | 32 | 15 | 3.387 | 0.003668 |
| reduce_prod_full_f32_n1048576 | jax-cuda | 32 | 15 | 3.336 | 0.5327 |
| reduce_max_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.024 | 0 |
| reduce_min_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.634 | 0 |
| compare_eq_f32_n1048576 | jax-cuda | 32 | 15 | 3.238 | 0 |
| compare_lt_f32_n1048576 | jax-cuda | 32 | 15 | 3.115 | 0 |
| compare_le_f32_n1048576 | jax-cuda | 32 | 15 | 3.128 | 0 |
| compare_ge_f32_n1048576 | jax-cuda | 32 | 15 | 3.117 | 0 |
| reduce_sum_full_f64_n1048576 | jax-cuda | 32 | 15 | 3.858 | 0.0006452 |
| reduce_prod_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.330 | 0.0001105 |
| reduce_max_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.651 | 0 |
| reduce_min_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.073 | 0 |
| compare_eq_f64_n1048576 | jax-cuda | 32 | 15 | 3.270 | 0 |
| compare_lt_f64_n1048576 | jax-cuda | 32 | 15 | 3.195 | 0 |
| compare_le_f64_n1048576 | jax-cuda | 32 | 15 | 3.163 | 0 |
| compare_ge_f64_n1048576 | jax-cuda | 32 | 15 | 3.204 | 0 |
| reduce_mean_full_f32_n1024 | jax-cuda | 32 | 15 | 3.113 | 0 |
| reduce_sum_squares_full_f32_n1024 | jax-cuda | 16 | 15 | 2.438 | 0.0001766 |
| reduce_mean_full_f64_n1024 | jax-cuda | 16 | 15 | 2.484 | 5.169e-07 |
| reduce_sum_squares_full_f64_n1024 | jax-cuda | 16 | 15 | 2.549 | 0 |
| reduce_mean_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.955 | 0.003668 |
| reduce_sum_squares_full_f32_n1048576 | jax-cuda | 32 | 15 | 3.325 | 0 |
| reduce_mean_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.366 | 0.001783 |
| reduce_sum_squares_full_f64_n1048576 | jax-cuda | 32 | 15 | 3.470 | 0 |
| add_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.745 | 0 |
| add_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.658 | 0 |
| sub_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.576 | 0 |
| sub_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.574 | 0 |
| mul_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.580 | 0 |
| mul_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.606 | 0 |
| div_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.565 | 4.875e-05 |
| div_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.634 | 0 |
| rem_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.088 | 0 |
| rem_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.009 | 0 |
| pow_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.165 | 0.000117 |
| pow_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.684 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.370 | 0 |
| maximum_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.157 | 0 |
| minimum_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.011 | 0 |
| minimum_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.420 | 0 |
| neg_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.054 | 0 |
| neg_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.104 | 0 |
| abs_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.114 | 0 |
| abs_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.002 | 0 |
| sign_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.108 | 0 |
| sign_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.266 | 0 |
| conj_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.109 | 0 |
| conj_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.934 | 0 |
| exp_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.142 | 0.0001272 |
| exp_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.233 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.123 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.282 | 7.39e-08 |
| log_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.241 | 7.945e-05 |
| log_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.172 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.207 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.425 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.963 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.136 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.262 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.535 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.717 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.643 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.410 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.417 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.165 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.239 | 0 |
| rsqrt_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.074 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.225 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.180 | 0 |
| compare_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.118 | 0 |
| where_select_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.259 | 0 |
| where_select_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.241 | 0 |
| clamp_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.237 | 0 |
| clamp_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.322 | 0 |
| reduce_sum_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.426 | 0.0002742 |
| reduce_sum_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.757 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.667 | 0.0003579 |
| reduce_prod_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.674 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.748 | 0 |
| reduce_max_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.639 | 0 |
| reduce_min_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.296 | 0 |
| reduce_min_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.414 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.259 | 1.439e-05 |
| reduce_mean_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.329 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.665 | 0 |
| add_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.574 | 0 |
| sub_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.195 | 0 |
| sub_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.010 | 0 |
| mul_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.209 | 0 |
| mul_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.015 | 0 |
| div_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.726 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.920 | 0 |
| rem_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.721 | 0 |
| rem_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.743 | 0 |
| pow_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.546 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda-jit | 2 | 15 | 2.523 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.860 | 0 |
| maximum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.690 | 0 |
| minimum_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.291 | 0 |
| minimum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.692 | 0 |
| neg_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.272 | 0 |
| neg_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.146 | 0 |
| abs_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.394 | 0 |
| abs_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.138 | 0 |
| sign_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.272 | 0 |
| sign_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.147 | 0 |
| conj_c32_n1048576 | jax-cuda-jit | 32 | 15 | 2.139 | 0 |
| conj_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.849 | 0 |
| exp_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.264 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.238 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.846 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.121 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.618 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.119 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.254 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.224 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.288 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda-jit | 4 | 15 | 2.620 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.052 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.835 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.916 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.820 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.143 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.046 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.110 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.313 | 0 |
| rsqrt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.085 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.948 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.177 | 0 |
| compare_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.240 | 0 |
| where_select_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.249 | 0 |
| where_select_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.044 | 0 |
| clamp_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.236 | 0 |
| clamp_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.115 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.797 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.448 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.390 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.471 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.458 | 0 |
| reduce_max_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.400 | 0 |
| reduce_min_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.513 | 0 |
| reduce_min_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.607 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.726 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.814 | 6.5e-09 |
| reduce_sum_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.751 | 0 |
| reduce_prod_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.552 | 0.001072 |
| reduce_max_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.710 | 0 |
| reduce_min_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.701 | 0 |
| compare_eq_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.424 | 0 |
| compare_lt_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.172 | 0 |
| compare_le_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.008 | 0 |
| compare_ge_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.129 | 0 |
| reduce_sum_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.114 | 4.966e-07 |
| reduce_prod_full_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.325 | 4.435e-07 |
| reduce_max_full_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.120 | 0 |
| reduce_min_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.802 | 0 |
| compare_eq_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.105 | 0 |
| compare_lt_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.179 | 0 |
| compare_le_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.138 | 0 |
| compare_ge_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.109 | 0 |
| reduce_sum_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.496 | 0.003668 |
| reduce_prod_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.602 | 0.5327 |
| reduce_max_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.613 | 0 |
| reduce_min_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.722 | 0 |
| compare_eq_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.983 | 0 |
| compare_lt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.187 | 0 |
| compare_le_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.158 | 0 |
| compare_ge_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.024 | 0 |
| reduce_sum_full_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.743 | 0.0006452 |
| reduce_prod_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.566 | 0.0001122 |
| reduce_max_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.608 | 0 |
| reduce_min_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.180 | 0 |
| compare_eq_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.335 | 0 |
| compare_lt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.071 | 0 |
| compare_le_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.930 | 0 |
| compare_ge_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.949 | 0 |
| reduce_mean_full_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.160 | 0 |
| reduce_sum_squares_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.323 | 0.0001766 |
| reduce_mean_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.336 | 6.891e-07 |
| reduce_sum_squares_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.138 | 0 |
| reduce_mean_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.025 | 0.003668 |
| reduce_sum_squares_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.667 | 0 |
| reduce_mean_full_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.725 | 0.001783 |
| reduce_sum_squares_full_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.395 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: benchmarks/gpu/isolated_ops.yaml
timestamp: '2026-10-13T15:16:34Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: bc70342b20e7ff9257c64cc046c8a90294f012e6
  dirty: false
environment:
  hostname: 737f43342de3
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
  expected_cases: 164
  selected_cases: 164
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: ef0cec562e6a48dc151cd120b3f916c38ac96663a1228e351018d99aa625d87d
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/structural.rs: 8121b12b5fed3b0acbc71e2218856173d53a5acc93092b93c7fd3309d4f45962
    src/bin/benchmark_gpu_public_api/linalg.rs: 84162923fa0d0772c88ec9c4eab169ad3a75e5d16ff76961d1541670dd8565c1
    src/bin/benchmark_gpu_public_api/reuse.rs: ad72b565373904275128d7beb8673bada08d922e8f310be6cedb0d2d8187898f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: fc27901bbb61c4c56c7a8417090f56f4312928eb502942cebeda7042505eb4c9
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: d9d56faa7d44345e6d3317bc837d84173ad0d7b0bc1e829b776fd41ec68fb1aa
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/isolated_ops/20261013_151634/fixtures/manifest.json: 88b010a4ee77306097e4ab814923590b5ccabbffa49a8de78946318d24463533
    target/release/benchmark_gpu_public_api: 1f7a23d62012e216687550eb87baaf8cd287c945991f2c1a30df70d669c11c62
  reference_numpy_version: 2.4.2
```
