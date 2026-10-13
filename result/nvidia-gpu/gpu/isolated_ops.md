# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **818**. Unsupported cells: **62**, including **38** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261013_201022/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_f32_n1024 | 12.088 (1.526) | 24.136 (1.465) | 9.793 (0.294) | 108.475 (3.969) | 97.265 (2.167) |
| add_f64_n1024 | 14.230 (0.845) | 27.234 (3.586) | 9.758 (0.166) | 103.871 (20.659) | 95.508 (16.179) |
| sub_f32_n1024 | 16.059 (1.224) | 30.627 (1.253) | 9.908 (0.116) | 92.601 (15.906) | 86.024 (11.369) |
| sub_f64_n1024 | 17.881 (1.060) | 36.280 (1.937) | 9.815 (0.099) | 87.705 (5.773) | 80.908 (9.582) |
| mul_f32_n1024 | 17.472 (0.267) | 37.392 (1.236) | 9.678 (0.391) | 86.463 (3.770) | 79.467 (8.991) |
| mul_f64_n1024 | 17.937 (0.241) | 37.234 (1.248) | 9.592 (0.367) | 84.731 (6.584) | 78.151 (5.756) |
| div_f32_n1024 | 17.842 (0.256) | 37.061 (1.251) | 9.555 (0.195) | 78.123 (2.542) | 78.817 (9.827) |
| div_f64_n1024 | 17.938 (0.138) | 36.544 (1.275) | 9.270 (0.118) | 78.665 (8.186) | 82.198 (7.409) |
| rem_f32_n1024 | 16.636 (0.239) | 222.967 (10.703) | 9.821 (0.777) | 94.868 (1.233) | 94.979 (10.143) |
| rem_f64_n1024 | 16.736 (0.127) | 252.304 (20.994) | 9.706 (0.398) | 105.897 (9.157) | 105.023 (23.336) |
| pow_f32_n1024 | 17.160 (0.222) | 222.733 (17.655) | 9.928 (0.098) | 139.188 (5.151) | 124.072 (6.040) |
| pow_f64_n1024 | 17.634 (0.106) | 258.942 (22.921) | 30.911 (0.010) | 181.624 (5.683) | 144.462 (4.947) |
| maximum_f32_n1024 | 16.787 (0.116) | 311.480 (11.803) | 10.293 (0.125) | 156.131 (28.940) | 130.372 (19.137) |
| maximum_f64_n1024 | 16.948 (0.336) | 238.220 (16.601) | 10.172 (0.264) | 131.210 (13.960) | 126.308 (17.009) |
| minimum_f32_n1024 | 17.056 (0.331) | 285.818 (7.294) | 10.049 (0.497) | 139.898 (22.110) | 101.981 (16.375) |
| minimum_f64_n1024 | 17.035 (0.234) | 294.207 (6.779) | 9.578 (0.079) | 137.306 (11.033) | 98.134 (8.344) |
| neg_f32_n1024 | 13.235 (1.606) | 22.343 (0.420) | 9.301 (0.163) | 131.185 (10.416) | 91.637 (0.573) |
| neg_f64_n1024 | 11.953 (1.544) | 23.389 (0.474) | 9.230 (0.090) | 127.960 (12.272) | 91.569 (0.360) |
| abs_f32_n1024 | 10.909 (1.275) | 161.212 (25.581) | 11.647 (0.616) | 99.439 (16.180) | 93.278 (2.397) |
| abs_f64_n1024 | 10.107 (1.039) | 152.292 (3.719) | 11.994 (0.544) | 92.149 (5.196) | 93.380 (4.872) |
| sign_f32_n1024 | 10.090 (0.987) | 178.088 (4.747) | 9.368 (0.431) | 96.223 (10.155) | 94.293 (7.707) |
| sign_f64_n1024 | 10.034 (0.888) | 179.175 (4.174) | 9.135 (0.141) | 91.024 (12.707) | 97.844 (7.026) |
| conj_c32_n1024 | 9.218 (0.969) | 187.553 (5.132) | 10.831 (0.078) | 94.749 (9.526) | 92.115 (0.706) |
| conj_c64_n1024 | 10.892 (0.249) | 147.308 (3.601) | 12.107 (0.860) | 93.544 (6.475) | 95.319 (7.315) |
| exp_f32_n1024 | 11.681 (1.524) | 22.319 (0.406) | 9.253 (0.460) | 96.051 (10.264) | 92.306 (2.715) |
| exp_f64_n1024 | 11.311 (1.667) | 22.386 (0.471) | 9.075 (0.142) | 103.102 (9.622) | 97.843 (12.609) |
| expm1_f32_n1024 | 10.188 (1.118) | 23.360 (0.663) | 9.413 (0.413) | 98.799 (13.303) | 91.132 (12.721) |
| expm1_f64_n1024 | 9.920 (1.092) | 24.313 (0.781) | 9.297 (0.111) | 97.837 (20.788) | 100.146 (14.433) |
| log_f32_n1024 | 9.723 (0.814) | 23.705 (0.786) | 8.938 (0.070) | 101.024 (16.840) | 98.234 (14.742) |
| log_f64_n1024 | 9.427 (0.811) | 24.120 (1.155) | 10.913 (0.008) | 93.624 (12.742) | 103.895 (21.061) |
| log1p_f32_n1024 | 9.344 (0.874) | 23.633 (1.363) | 8.901 (0.121) | 98.618 (14.801) | 101.553 (17.100) |
| log1p_f64_n1024 | 9.116 (0.903) | 23.948 (0.582) | 11.272 (0.002) | 106.895 (16.707) | 99.632 (17.089) |
| erf_f32_n1024 | 8.673 (0.743) | 24.285 (1.476) | 8.948 (0.154) | 192.775 (26.225) | 99.265 (15.101) |
| erf_f64_n1024 | 8.987 (0.716) | 23.645 (0.761) | 15.852 (0.007) | 184.268 (49.365) | 126.280 (19.030) |
| sin_f32_n1024 | 8.787 (0.723) | 24.139 (1.101) | 9.232 (0.038) | 155.784 (15.756) | 148.016 (16.847) |
| sin_f64_n1024 | 8.973 (0.851) | 23.631 (1.155) | 9.027 (0.224) | 163.637 (3.814) | 162.850 (15.840) |
| cos_f32_n1024 | 8.915 (0.775) | 23.195 (0.787) | 9.119 (0.066) | 167.461 (7.630) | 159.517 (22.267) |
| cos_f64_n1024 | 8.991 (0.856) | 23.946 (0.672) | 8.910 (0.174) | 156.748 (6.503) | 159.836 (23.355) |
| tanh_f32_n1024 | 8.670 (0.847) | 24.068 (0.765) | 8.960 (0.092) | 173.868 (28.796) | 141.913 (32.896) |
| tanh_f64_n1024 | 9.038 (0.685) | 24.180 (1.123) | 9.100 (0.260) | 136.977 (9.182) | 142.720 (31.865) |
| sqrt_f32_n1024 | 8.792 (0.822) | 23.566 (0.908) | 9.247 (0.090) | 141.444 (13.265) | 126.835 (12.225) |
| sqrt_f64_n1024 | 8.910 (0.808) | 24.228 (0.565) | 9.320 (0.125) | 140.436 (14.025) | 135.693 (17.448) |
| rsqrt_f32_n1024 | 8.521 (0.922) | 23.899 (1.185) | 9.461 (0.118) | 185.536 (48.754) | 127.172 (15.658) |
| rsqrt_f64_n1024 | 9.007 (0.730) | 24.492 (0.929) | 8.955 (0.098) | 173.153 (42.508) | 124.087 (23.395) |
| compare_f32_n1024 | unsupported | 451.065 (4.940) | 9.607 (0.688) | 104.498 (16.063) | 124.132 (14.069) |
| compare_f64_n1024 | unsupported | 447.452 (49.810) | 9.313 (0.476) | 99.160 (9.418) | 100.795 (13.426) |
| where_select_f32_n1024 | 18.298 (1.116) | 474.125 (18.443) | 12.438 (0.487) | 107.985 (10.714) | 100.034 (10.758) |
| where_select_f64_n1024 | 18.392 (1.264) | 512.738 (31.698) | 12.550 (0.157) | 108.504 (9.928) | 101.333 (9.414) |
| clamp_f32_n1024 | 49.297 (3.892) | 1001.524 (26.362) | 10.767 (0.146) | 99.515 (9.405) | 101.393 (7.345) |
| clamp_f64_n1024 | 50.185 (1.982) | 924.019 (48.359) | 10.964 (0.087) | 100.357 (9.967) | 102.218 (10.691) |
| reduce_sum_f32_n1024 | 22.333 (0.288) | 221.724 (4.888) | 13.996 (0.149) | 169.087 (12.953) | 90.832 (1.081) |
| reduce_sum_f64_n1024 | 22.371 (0.437) | 215.328 (3.464) | 14.011 (0.401) | 163.867 (2.513) | 91.709 (3.845) |
| reduce_prod_f32_n1024 | 21.264 (0.190) | 236.269 (8.038) | 12.334 (0.312) | 127.183 (4.727) | 157.358 (19.217) |
| reduce_prod_f64_n1024 | 21.322 (0.714) | 186.274 (3.508) | 12.419 (0.125) | 168.197 (6.555) | 129.490 (12.327) |
| reduce_max_f32_n1024 | 22.355 (1.225) | 202.901 (13.227) | 13.471 (0.172) | 173.800 (7.788) | 166.352 (15.299) |
| reduce_max_f64_n1024 | 24.271 (1.335) | 201.026 (7.413) | 13.356 (0.072) | 128.187 (1.693) | 108.502 (18.083) |
| reduce_min_f32_n1024 | 21.838 (1.174) | 219.642 (5.733) | 13.601 (0.138) | 155.135 (23.607) | 101.337 (13.816) |
| reduce_min_f64_n1024 | 22.389 (1.401) | 215.816 (9.657) | 13.705 (0.398) | 94.467 (0.916) | 85.917 (0.544) |
| reduce_sum_squares_f32_n1024 | 21.743 (0.666) | 208.606 (3.683) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1024 | 21.267 (0.632) | 205.283 (4.708) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1024 | unsupported | unsupported | 15.124 (0.659) | 128.964 (9.374) | 97.486 (0.384) |
| reduce_mean_f64_n1024 | unsupported | unsupported | 14.907 (1.041) | 137.631 (8.602) | 93.822 (1.021) |
| add_f32_n1048576 | 48.318 (0.350) | 50.066 (0.488) | 39.852 (0.117) | 106.367 (2.339) | 98.066 (0.313) |
| add_f64_n1048576 | 83.106 (0.450) | 86.031 (1.036) | 77.744 (0.064) | 107.941 (5.042) | 92.040 (1.229) |
| sub_f32_n1048576 | 43.761 (0.129) | 44.495 (0.161) | 39.733 (0.069) | 105.592 (4.570) | 78.744 (10.021) |
| sub_f64_n1048576 | 82.724 (0.057) | 83.431 (0.573) | 77.768 (0.082) | 103.697 (2.463) | 83.480 (1.354) |
| mul_f32_n1048576 | 43.639 (0.056) | 43.346 (0.170) | 39.744 (0.103) | 106.489 (1.520) | 70.322 (4.640) |
| mul_f64_n1048576 | 82.533 (0.078) | 83.214 (0.093) | 77.703 (0.103) | 98.386 (4.303) | 83.954 (1.526) |
| div_f32_n1048576 | 43.579 (0.091) | 43.562 (0.415) | 39.969 (0.029) | 89.956 (3.189) | 70.636 (4.307) |
| div_f64_n1048576 | 95.535 (0.049) | 96.131 (0.404) | 94.618 (0.065) | 129.077 (0.631) | 122.370 (0.827) |
| rem_f32_n1048576 | 42.388 (0.118) | 97.174 (0.260) | 39.875 (0.041) | 121.990 (16.326) | 109.752 (17.018) |
| rem_f64_n1048576 | 80.696 (0.442) | 186.444 (0.128) | 78.207 (0.073) | 98.767 (6.920) | 101.534 (24.244) |
| pow_f32_n1048576 | 42.179 (0.243) | 97.284 (0.132) | 41.556 (0.080) | 127.780 (11.056) | 101.236 (6.609) |
| pow_f64_n1048576 | 1006.214 (0.587) | 1119.652 (0.747) | 1020.684 (0.378) | 1277.259 (12.570) | 1299.607 (11.528) |
| maximum_f32_n1048576 | 42.403 (0.449) | 97.630 (0.227) | 39.882 (0.056) | 121.229 (10.787) | 117.053 (14.996) |
| maximum_f64_n1048576 | 81.647 (0.395) | 186.784 (0.259) | 78.425 (0.125) | 102.293 (4.966) | 92.711 (1.190) |
| minimum_f32_n1048576 | 42.660 (0.302) | 97.558 (0.358) | 39.886 (0.038) | 104.530 (0.830) | 98.149 (0.737) |
| minimum_f64_n1048576 | 81.636 (0.336) | 186.764 (0.173) | 78.431 (0.128) | 99.199 (9.188) | 93.069 (2.216) |
| neg_f32_n1048576 | 29.737 (0.389) | 29.745 (0.754) | 27.307 (0.026) | 97.259 (1.184) | 94.011 (0.571) |
| neg_f64_n1048576 | 56.142 (0.417) | 57.527 (0.250) | 53.034 (0.042) | 104.204 (0.385) | 93.282 (1.038) |
| abs_f32_n1048576 | 29.286 (0.571) | 56.309 (0.768) | 27.400 (0.047) | 90.401 (0.440) | 79.725 (11.369) |
| abs_f64_n1048576 | 56.283 (0.398) | 107.473 (0.208) | 53.106 (0.053) | 90.398 (1.167) | 67.234 (3.779) |
| sign_f32_n1048576 | 29.473 (0.151) | 56.095 (0.255) | 27.337 (0.052) | 90.217 (2.995) | 71.349 (2.422) |
| sign_f64_n1048576 | 56.349 (0.485) | 107.469 (0.173) | 53.221 (0.189) | 93.010 (1.614) | 66.839 (1.728) |
| conj_c32_n1048576 | 55.873 (0.483) | 107.925 (0.112) | 53.147 (0.216) | 89.372 (6.821) | 66.540 (1.082) |
| conj_c64_n1048576 | 109.874 (0.658) | 209.328 (0.123) | 104.060 (0.069) | 118.389 (1.384) | 116.758 (5.073) |
| exp_f32_n1048576 | 29.668 (0.290) | 29.812 (0.189) | 27.461 (0.139) | 86.474 (2.114) | 71.687 (5.409) |
| exp_f64_n1048576 | 184.461 (0.555) | 187.927 (0.443) | 181.034 (0.123) | 204.473 (1.597) | 201.017 (1.172) |
| expm1_f32_n1048576 | 29.812 (0.343) | 29.750 (0.138) | 27.498 (0.072) | 95.193 (6.487) | 78.426 (12.076) |
| expm1_f64_n1048576 | 228.123 (0.097) | 232.179 (0.844) | 225.279 (0.068) | 265.254 (2.519) | 264.836 (4.557) |
| log_f32_n1048576 | 29.571 (0.167) | 29.658 (0.173) | 27.478 (0.046) | 95.862 (9.084) | 80.034 (11.744) |
| log_f64_n1048576 | 352.368 (0.407) | 357.147 (0.380) | 351.140 (0.261) | 390.854 (4.407) | 394.283 (18.466) |
| log1p_f32_n1048576 | 29.893 (0.461) | 29.871 (0.151) | 27.481 (0.039) | 122.439 (14.855) | 94.489 (3.802) |
| log1p_f64_n1048576 | 364.187 (0.381) | 369.429 (0.572) | 363.479 (0.141) | 413.345 (10.287) | 420.759 (8.208) |
| erf_f32_n1048576 | 29.635 (0.376) | 29.852 (0.039) | 27.450 (0.041) | 154.108 (18.815) | 130.632 (21.013) |
| erf_f64_n1048576 | 530.118 (0.462) | 536.566 (0.714) | 531.429 (0.361) | 688.523 (17.228) | 667.086 (18.651) |
| sin_f32_n1048576 | 29.883 (0.339) | 29.896 (0.124) | 27.604 (0.047) | 97.478 (19.610) | 139.518 (15.104) |
| sin_f64_n1048576 | 201.308 (1.566) | 204.697 (0.465) | 212.456 (0.151) | 253.093 (1.423) | 238.201 (0.488) |
| cos_f32_n1048576 | 29.973 (0.557) | 29.777 (0.106) | 27.571 (0.039) | 103.408 (18.541) | 101.643 (13.549) |
| cos_f64_n1048576 | 200.743 (0.999) | 204.304 (0.635) | 211.833 (0.090) | 252.701 (1.175) | 237.953 (1.878) |
| tanh_f32_n1048576 | 29.670 (0.335) | 29.633 (0.027) | 27.419 (0.024) | 95.513 (7.027) | 93.667 (2.490) |
| tanh_f64_n1048576 | 171.210 (1.217) | 174.386 (0.252) | 167.752 (0.057) | 191.174 (2.603) | 188.844 (0.808) |
| sqrt_f32_n1048576 | 29.538 (0.202) | 29.805 (0.236) | 27.443 (0.048) | 94.659 (5.519) | 98.032 (5.288) |
| sqrt_f64_n1048576 | 93.929 (0.520) | 95.808 (0.303) | 92.261 (0.082) | 102.097 (1.030) | 102.632 (1.831) |
| rsqrt_f32_n1048576 | 29.507 (0.040) | 29.603 (0.077) | 27.408 (0.027) | 130.525 (4.373) | 95.119 (1.599) |
| rsqrt_f64_n1048576 | 62.950 (0.152) | 64.341 (0.329) | 61.712 (0.087) | 138.598 (4.205) | 89.904 (0.729) |
| compare_f32_n1048576 | 32.283 (0.235) | 87.718 (0.213) | 30.485 (0.121) | 95.097 (3.218) | 98.275 (0.515) |
| compare_f64_n1048576 | 57.155 (0.151) | 164.928 (0.109) | 56.043 (0.105) | 84.475 (2.422) | 100.491 (0.394) |
| where_select_f32_n1048576 | 46.320 (0.214) | 117.745 (0.293) | 43.012 (0.032) | 101.294 (3.409) | 98.368 (1.539) |
| where_select_f64_n1048576 | 85.215 (0.350) | 206.918 (0.505) | 80.931 (0.078) | 101.793 (4.749) | 95.205 (2.865) |
| clamp_f32_n1048576 | 87.041 (0.641) | 516.631 (13.423) | 28.290 (0.029) | 93.592 (2.400) | 102.675 (1.074) |
| clamp_f64_n1048576 | 163.348 (1.402) | 520.348 (20.044) | 62.404 (0.043) | 88.746 (1.828) | 97.378 (1.122) |
| reduce_sum_f32_n1048576 | 40.355 (0.384) | 67.301 (1.612) | 16.248 (0.029) | 133.964 (1.174) | 153.916 (5.798) |
| reduce_sum_f64_n1048576 | 48.082 (0.603) | 95.304 (0.293) | 32.417 (0.029) | 156.801 (2.186) | 162.970 (3.630) |
| reduce_prod_f32_n1048576 | 44.927 (0.455) | 72.323 (0.945) | 16.131 (0.025) | 155.302 (3.094) | 153.230 (4.123) |
| reduce_prod_f64_n1048576 | 51.640 (0.298) | 95.411 (0.504) | 32.213 (0.030) | 165.395 (1.609) | 155.875 (2.554) |
| reduce_max_f32_n1048576 | 66.485 (0.980) | 74.058 (0.095) | 16.549 (0.028) | 160.678 (1.490) | 150.964 (2.824) |
| reduce_max_f64_n1048576 | 78.345 (0.416) | 112.193 (0.221) | 41.783 (0.013) | 160.890 (2.142) | 152.063 (1.595) |
| reduce_min_f32_n1048576 | 65.087 (0.542) | 74.073 (0.247) | 16.536 (0.013) | 161.302 (15.245) | 162.189 (18.238) |
| reduce_min_f64_n1048576 | 77.470 (0.520) | 112.435 (0.123) | 41.779 (0.020) | 162.254 (5.123) | 163.259 (12.254) |
| reduce_sum_squares_f32_n1048576 | 42.665 (0.290) | 85.477 (2.345) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1048576 | 52.536 (0.651) | 97.130 (0.411) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1048576 | unsupported | unsupported | 16.262 (0.015) | 172.735 (14.500) | 180.923 (25.609) |
| reduce_mean_f64_n1048576 | unsupported | unsupported | 32.870 (0.015) | 169.946 (23.319) | 156.133 (22.556) |
| reduce_sum_full_f32_n1024 | 24.571 (1.393) | 218.832 (28.881) | 11.754 (0.225) | 135.332 (8.672) | 133.488 (6.662) |
| reduce_prod_full_f32_n1024 | 19.336 (1.534) | 253.025 (5.499) | 10.595 (0.629) | 128.645 (3.272) | 99.819 (7.286) |
| reduce_max_full_f32_n1024 | 16.808 (0.789) | 256.653 (3.418) | 12.354 (0.656) | 184.266 (12.270) | 146.998 (4.424) |
| reduce_min_full_f32_n1024 | 15.810 (0.583) | 257.238 (3.585) | 11.879 (0.081) | 187.606 (25.471) | 169.644 (25.389) |
| compare_eq_f32_n1024 | unsupported | 519.176 (20.492) | 9.266 (0.065) | 156.244 (32.759) | 138.942 (12.048) |
| compare_lt_f32_n1024 | unsupported | 505.873 (13.841) | 9.307 (0.288) | 133.690 (8.874) | 104.240 (13.908) |
| compare_le_f32_n1024 | unsupported | 511.854 (13.534) | 9.740 (0.404) | 121.247 (20.168) | 101.521 (3.097) |
| compare_ge_f32_n1024 | unsupported | 503.198 (13.163) | 9.421 (0.200) | 127.716 (11.850) | 100.949 (4.938) |
| reduce_sum_full_f64_n1024 | 23.704 (0.432) | 198.244 (5.299) | 11.939 (0.106) | 178.373 (5.027) | 93.680 (1.008) |
| reduce_prod_full_f64_n1024 | 21.323 (1.130) | 199.401 (6.915) | 10.448 (0.239) | 176.953 (8.546) | 93.479 (2.449) |
| reduce_max_full_f64_n1024 | 19.626 (0.980) | 194.398 (2.550) | 11.596 (0.113) | 149.785 (5.155) | 92.605 (1.039) |
| reduce_min_full_f64_n1024 | 15.905 (1.198) | 194.559 (4.360) | 11.891 (0.331) | 96.667 (0.706) | 93.000 (1.200) |
| compare_eq_f64_n1024 | unsupported | 484.471 (19.227) | 9.593 (0.042) | 100.043 (4.822) | 100.435 (3.081) |
| compare_lt_f64_n1024 | unsupported | 479.616 (7.344) | 9.189 (0.246) | 100.073 (8.580) | 100.405 (3.500) |
| compare_le_f64_n1024 | unsupported | 518.798 (50.854) | 9.344 (0.116) | 96.174 (12.001) | 111.683 (7.069) |
| compare_ge_f64_n1024 | unsupported | 503.859 (25.411) | 9.455 (0.239) | 97.696 (6.714) | 103.642 (7.425) |
| reduce_sum_full_f32_n1048576 | 2533.377 (3.894) | 2661.580 (24.177) | 18.507 (0.323) | 144.493 (4.411) | 103.882 (10.877) |
| reduce_prod_full_f32_n1048576 | 2524.106 (8.383) | 2621.956 (5.565) | 18.441 (0.038) | 128.844 (2.327) | 106.375 (2.153) |
| reduce_max_full_f32_n1048576 | 9794.935 (3.072) | 9887.060 (16.227) | 18.734 (0.071) | 177.986 (4.827) | 157.441 (8.584) |
| reduce_min_full_f32_n1048576 | 9795.316 (2.051) | 9905.093 (13.764) | 18.873 (0.070) | 201.286 (27.310) | 151.848 (20.686) |
| compare_eq_f32_n1048576 | 32.712 (0.076) | 120.023 (4.316) | 30.333 (0.097) | 122.552 (8.661) | 122.739 (15.109) |
| compare_lt_f32_n1048576 | 32.753 (0.107) | 103.423 (3.599) | 30.367 (0.086) | 100.320 (1.126) | 100.997 (0.326) |
| compare_le_f32_n1048576 | 32.919 (0.487) | 95.781 (10.698) | 30.262 (0.027) | 96.416 (3.453) | 103.088 (0.913) |
| compare_ge_f32_n1048576 | 33.218 (0.291) | 87.564 (0.304) | 30.269 (0.012) | 95.017 (3.355) | 102.393 (1.318) |
| reduce_sum_full_f64_n1048576 | 3037.319 (1.363) | 3102.715 (1.842) | 37.460 (0.023) | 150.477 (2.779) | 117.130 (14.383) |
| reduce_prod_full_f64_n1048576 | 3028.771 (2.840) | 3104.041 (2.924) | 37.496 (0.029) | 181.232 (6.477) | 130.282 (3.358) |
| reduce_max_full_f64_n1048576 | 12449.414 (3.943) | 12553.851 (6.128) | 44.723 (0.089) | 176.247 (13.691) | 151.084 (13.880) |
| reduce_min_full_f64_n1048576 | 12449.511 (2.939) | 12553.369 (4.200) | 44.712 (0.028) | 147.756 (2.467) | 110.022 (14.952) |
| compare_eq_f64_n1048576 | 57.162 (0.194) | 164.654 (0.335) | 56.712 (0.039) | 104.410 (11.603) | 90.683 (26.246) |
| compare_lt_f64_n1048576 | 57.181 (0.170) | 164.608 (0.194) | 55.982 (0.039) | 99.949 (1.386) | 69.468 (1.483) |
| compare_le_f64_n1048576 | 57.149 (0.241) | 164.579 (0.423) | 55.978 (0.149) | 101.402 (2.390) | 69.979 (1.360) |
| compare_ge_f64_n1048576 | 57.209 (0.255) | 164.582 (0.134) | 55.968 (0.041) | 99.622 (0.843) | 69.182 (2.221) |
| reduce_mean_full_f32_n1024 | unsupported | unsupported | 12.141 (0.284) | 105.009 (0.786) | 75.311 (3.959) |
| reduce_sum_squares_full_f32_n1024 | 12.183 (1.329) | 253.180 (3.373) | 16.131 (1.129) | 156.976 (6.448) | 95.804 (4.702) |
| reduce_mean_full_f64_n1024 | unsupported | unsupported | 12.406 (0.276) | 190.177 (13.983) | 86.535 (0.750) |
| reduce_sum_squares_full_f64_n1024 | 8.969 (0.142) | 253.362 (18.960) | 14.972 (0.248) | 166.772 (5.233) | 77.023 (0.666) |
| reduce_mean_full_f32_n1048576 | unsupported | unsupported | 18.467 (0.082) | 114.806 (4.515) | 82.290 (0.897) |
| reduce_sum_squares_full_f32_n1048576 | 2545.288 (2.812) | 2687.929 (10.387) | 17.666 (0.022) | 107.628 (2.862) | 80.514 (0.858) |
| reduce_mean_full_f64_n1048576 | unsupported | unsupported | 37.501 (0.038) | 151.245 (19.672) | 91.588 (0.602) |
| reduce_sum_squares_full_f64_n1048576 | 3300.154 (2.745) | 3379.213 (3.078) | 33.261 (0.034) | 170.258 (3.929) | 106.039 (15.531) |
| scale_real_f32_n1024 | unsupported | unsupported | 10.613 (0.256) | 155.579 (38.192) | 95.181 (3.351) |
| scale_real_f64_n1024 | unsupported | unsupported | 10.404 (0.177) | 176.260 (22.865) | 99.038 (0.378) |
| scale_real_c32_n1024 | unsupported | unsupported | 10.287 (0.163) | 141.319 (32.087) | 107.323 (9.216) |
| scale_real_c64_n1024 | unsupported | unsupported | 10.341 (0.084) | 141.803 (14.601) | 103.524 (4.424) |
| scale_complex_c32_n1024 | unsupported | unsupported | 10.236 (0.253) | 119.629 (12.321) | 102.841 (4.192) |
| scale_complex_c64_n1024 | unsupported | unsupported | 10.152 (0.087) | 109.214 (2.472) | 100.302 (3.639) |
| scale_real_f32_n1048576 | unsupported | unsupported | 28.134 (0.068) | 110.376 (1.606) | 102.709 (2.212) |
| scale_real_f64_n1048576 | unsupported | unsupported | 53.704 (0.076) | 110.460 (0.774) | 104.265 (1.089) |
| scale_real_c32_n1048576 | unsupported | unsupported | 53.663 (0.039) | 111.825 (1.467) | 103.468 (0.766) |
| scale_real_c64_n1048576 | unsupported | unsupported | 104.834 (0.333) | 121.007 (6.222) | 115.998 (1.242) |
| scale_complex_c32_n1048576 | unsupported | unsupported | 53.677 (0.029) | 110.934 (0.740) | 103.516 (1.459) |
| scale_complex_c64_n1048576 | unsupported | unsupported | 104.894 (0.337) | 115.823 (0.533) | 117.227 (1.126) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.094 | 0 |
| add_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.821 | 0 |
| sub_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.056 | 0 |
| sub_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.289 | 0 |
| mul_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.236 | 0 |
| mul_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.296 | 0 |
| div_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.284 | 0 |
| div_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.296 | 0 |
| rem_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.129 | 0 |
| rem_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.142 | 0 |
| pow_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.197 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.257 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.149 | 0 |
| maximum_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.169 | 0 |
| minimum_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.183 | 0 |
| minimum_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.181 | 0 |
| neg_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.388 | 0 |
| neg_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.060 | 0 |
| abs_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.793 | 0 |
| abs_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.588 | 0 |
| sign_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.583 | 0 |
| sign_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.569 | 0 |
| conj_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.360 | 0 |
| conj_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.394 | 0 |
| exp_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.990 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.896 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.608 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.540 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.489 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.413 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.392 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.334 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.220 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.301 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.249 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.297 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.282 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.302 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.219 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.314 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.251 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.281 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.181 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.306 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| where_select_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.342 | 0 |
| where_select_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.354 | 0 |
| clamp_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.155 | 0 |
| clamp_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.212 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.429 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.432 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.722 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.729 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.861 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.107 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.795 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.866 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.783 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.722 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.546 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.659 | 0 |
| sub_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.801 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.647 | 0 |
| mul_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.793 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.641 | 0 |
| div_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.789 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.057 | 0 |
| rem_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.713 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.582 | 0 |
| pow_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.699 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.012 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.714 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.613 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.730 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.612 | 0 |
| neg_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.903 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.797 | 0 |
| abs_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.874 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.801 | 0 |
| sign_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.886 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.803 | 0 |
| conj_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.788 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.758 | 0 |
| exp_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.899 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.951 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.908 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.650 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.893 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.819 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.913 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.913 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.897 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.120 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.912 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.221 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.918 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.212 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.899 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.739 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.890 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.006 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.888 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.014 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.066 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.658 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.964 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.727 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.785 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.614 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.583 | 0.001812 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.077 | 2.148e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.875 | 0.005478 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.305 | 3.659e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.128 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.507 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.083 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.479 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.731 | 0.0003528 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.362 | 3.293e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_full_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 0.786 | 0.0001333 |
| reduce_prod_full_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.238 | 0.003334 |
| reduce_max_full_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.151 | 0 |
| reduce_min_full_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.024 | 0 |
| compare_eq_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_lt_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_le_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_ge_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| reduce_sum_full_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.517 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.729 | 0 |
| reduce_max_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.512 | 0 |
| reduce_min_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.036 | 0 |
| compare_eq_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_lt_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_le_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_ge_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| reduce_sum_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.533 | 0.07061 |
| reduce_prod_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.524 | 0.1957 |
| reduce_max_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 9.795 | 0 |
| reduce_min_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 9.795 | 0 |
| compare_eq_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.094 | 0 |
| compare_lt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.096 | 0 |
| compare_le_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.107 | 0 |
| compare_ge_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.126 | 0 |
| reduce_sum_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.037 | 0.001745 |
| reduce_prod_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.029 | 6.451e-05 |
| reduce_max_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 12.449 | 0 |
| reduce_min_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 12.450 | 0 |
| compare_eq_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.658 | 0 |
| compare_lt_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.660 | 0 |
| compare_le_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.658 | 0 |
| compare_ge_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.661 | 0 |
| reduce_mean_full_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.119 | 0.0001766 |
| reduce_mean_full_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.296 | 0 |
| reduce_mean_full_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.545 | 0.001429 |
| reduce_mean_full_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.300 | 9.982e-07 |
| scale_real_f32_n1024 | tenferro-cuda-direct | — | — | — | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_real_f64_n1024 | tenferro-cuda-direct | — | — | — | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_real_c32_n1024 | tenferro-cuda-direct | — | — | — | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_real_c64_n1024 | tenferro-cuda-direct | — | — | — | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_complex_c32_n1024 | tenferro-cuda-direct | — | — | — | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_complex_c64_n1024 | tenferro-cuda-direct | — | — | — | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_real_f32_n1048576 | tenferro-cuda-direct | — | — | — | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_real_f64_n1048576 | tenferro-cuda-direct | — | — | — | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_real_c32_n1048576 | tenferro-cuda-direct | — | — | — | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_real_c64_n1048576 | tenferro-cuda-direct | — | — | — | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_complex_c32_n1048576 | tenferro-cuda-direct | — | — | — | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| scale_complex_c64_n1048576 | tenferro-cuda-direct | — | — | — | Public scalar-scale API creates and uploads the scalar internally; no preuploaded scalar argument exists. Numerically verified only; tensor-tensor mul is a distinct public route. |
| add_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.089 | 0 |
| add_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.486 | 0 |
| sub_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.920 | 0 |
| sub_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.322 | 0 |
| mul_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.393 | 0 |
| mul_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.383 | 0 |
| div_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.372 | 0 |
| div_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.339 | 0 |
| rem_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.567 | 0 |
| rem_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 4.037 | 0 |
| pow_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.564 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 4.143 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.492 | 0 |
| maximum_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.812 | 0 |
| minimum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.287 | 0 |
| minimum_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.354 | 0 |
| neg_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.860 | 0 |
| neg_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.994 | 0 |
| abs_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.579 | 0 |
| abs_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.437 | 0 |
| sign_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.849 | 0 |
| sign_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.867 | 0 |
| conj_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.001 | 0 |
| conj_c64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.357 | 0 |
| exp_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.857 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.865 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.990 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.112 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.034 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.087 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.025 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.065 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.108 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.027 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.090 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.025 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.969 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.065 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.081 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.095 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.016 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.101 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.059 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.135 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.609 | 0 |
| compare_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.580 | 0 |
| where_select_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.793 | 0 |
| where_select_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.051 | 0 |
| clamp_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 4.006 | 0 |
| clamp_f64_n1024 | tenferro-cuda-eager | 2 | 15 | 1.848 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 1.774 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 1.723 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.780 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.980 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.246 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.216 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.514 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.453 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.338 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.285 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.602 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.753 | 0 |
| sub_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.848 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.670 | 0 |
| mul_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.774 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.663 | 0 |
| div_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.788 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.076 | 0 |
| rem_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.110 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.983 | 0 |
| pow_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.113 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-eager | 2 | 15 | 2.239 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.124 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.989 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.122 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.988 | 0 |
| neg_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.904 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.841 | 0 |
| abs_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.604 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.439 | 0 |
| sign_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.590 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.439 | 0 |
| conj_c32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.454 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.349 | 0 |
| exp_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.908 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.007 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.904 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.715 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.898 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.857 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.912 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.955 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.911 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.146 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.913 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.275 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.906 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.269 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.897 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.790 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.908 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.066 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.895 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.059 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.807 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.639 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.768 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.311 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.067 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.081 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.154 | 0.001812 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.050 | 2.148e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.314 | 0.005478 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.053 | 3.659e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.370 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.590 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.370 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.598 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.735 | 0.0003528 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.108 | 3.293e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_full_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.501 | 0.0001333 |
| reduce_prod_full_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 4.048 | 0.003334 |
| reduce_max_full_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.053 | 0 |
| reduce_min_full_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.058 | 0 |
| compare_eq_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.077 | 0 |
| compare_lt_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.023 | 0 |
| compare_le_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.047 | 0 |
| compare_ge_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.013 | 0 |
| reduce_sum_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.172 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.190 | 0 |
| reduce_max_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.110 | 0 |
| reduce_min_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.113 | 0 |
| compare_eq_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 1.938 | 0 |
| compare_lt_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.837 | 0 |
| compare_le_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.075 | 0 |
| compare_ge_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.015 | 0 |
| reduce_sum_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.662 | 0.07061 |
| reduce_prod_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.622 | 0.1957 |
| reduce_max_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 9.887 | 0 |
| reduce_min_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 9.905 | 0 |
| compare_eq_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.841 | 0 |
| compare_lt_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.310 | 0 |
| compare_le_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.065 | 0 |
| compare_ge_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.802 | 0 |
| reduce_sum_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.103 | 0.001745 |
| reduce_prod_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.104 | 6.451e-05 |
| reduce_max_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 12.554 | 0 |
| reduce_min_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 12.553 | 0 |
| compare_eq_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.634 | 0 |
| compare_lt_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.634 | 0 |
| compare_le_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.633 | 0 |
| compare_ge_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.633 | 0 |
| reduce_mean_full_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.025 | 0.0001766 |
| reduce_mean_full_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 4.054 | 0 |
| reduce_mean_full_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.688 | 0.001429 |
| reduce_mean_full_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.379 | 9.982e-07 |
| scale_real_f32_n1024 | tenferro-cuda-eager | — | — | — | Current dfbdf6b CUDA probe 20261013_200158: EagerSession.scale creates a host scalar through constant_from, then mul fails in to_contiguous_read (expected CubeCL GPU tensor view, got host tensor). No public scale parameter can replace the internal scalar; execution unsupported, not numerically verified. |
| scale_real_f64_n1024 | tenferro-cuda-eager | — | — | — | Current dfbdf6b CUDA probe 20261013_200158: EagerSession.scale creates a host scalar through constant_from, then mul fails in to_contiguous_read (expected CubeCL GPU tensor view, got host tensor). No public scale parameter can replace the internal scalar; execution unsupported, not numerically verified. |
| scale_real_c32_n1024 | tenferro-cuda-eager | — | — | — | Current dfbdf6b CUDA probe 20261013_200158: EagerSession.scale creates a host scalar through constant_from, then mul fails in to_contiguous_read (expected CubeCL GPU tensor view, got host tensor). No public scale parameter can replace the internal scalar; execution unsupported, not numerically verified. |
| scale_real_c64_n1024 | tenferro-cuda-eager | — | — | — | Current dfbdf6b CUDA probe 20261013_200158: EagerSession.scale creates a host scalar through constant_from, then mul fails in to_contiguous_read (expected CubeCL GPU tensor view, got host tensor). No public scale parameter can replace the internal scalar; execution unsupported, not numerically verified. |
| scale_complex_c32_n1024 | tenferro-cuda-eager | — | — | — | Current dfbdf6b CUDA probe 20261013_200158: EagerSession.scale creates a host scalar through constant_from, then mul fails in to_contiguous_read (expected CubeCL GPU tensor view, got host tensor). No public scale parameter can replace the internal scalar; execution unsupported, not numerically verified. |
| scale_complex_c64_n1024 | tenferro-cuda-eager | — | — | — | Current dfbdf6b CUDA probe 20261013_200158: EagerSession.scale creates a host scalar through constant_from, then mul fails in to_contiguous_read (expected CubeCL GPU tensor view, got host tensor). No public scale parameter can replace the internal scalar; execution unsupported, not numerically verified. |
| scale_real_f32_n1048576 | tenferro-cuda-eager | — | — | — | Current dfbdf6b CUDA probe 20261013_200158: EagerSession.scale creates a host scalar through constant_from, then mul fails in to_contiguous_read (expected CubeCL GPU tensor view, got host tensor). No public scale parameter can replace the internal scalar; execution unsupported, not numerically verified. |
| scale_real_f64_n1048576 | tenferro-cuda-eager | — | — | — | Current dfbdf6b CUDA probe 20261013_200158: EagerSession.scale creates a host scalar through constant_from, then mul fails in to_contiguous_read (expected CubeCL GPU tensor view, got host tensor). No public scale parameter can replace the internal scalar; execution unsupported, not numerically verified. |
| scale_real_c32_n1048576 | tenferro-cuda-eager | — | — | — | Current dfbdf6b CUDA probe 20261013_200158: EagerSession.scale creates a host scalar through constant_from, then mul fails in to_contiguous_read (expected CubeCL GPU tensor view, got host tensor). No public scale parameter can replace the internal scalar; execution unsupported, not numerically verified. |
| scale_real_c64_n1048576 | tenferro-cuda-eager | — | — | — | Current dfbdf6b CUDA probe 20261013_200158: EagerSession.scale creates a host scalar through constant_from, then mul fails in to_contiguous_read (expected CubeCL GPU tensor view, got host tensor). No public scale parameter can replace the internal scalar; execution unsupported, not numerically verified. |
| scale_complex_c32_n1048576 | tenferro-cuda-eager | — | — | — | Current dfbdf6b CUDA probe 20261013_200158: EagerSession.scale creates a host scalar through constant_from, then mul fails in to_contiguous_read (expected CubeCL GPU tensor view, got host tensor). No public scale parameter can replace the internal scalar; execution unsupported, not numerically verified. |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | — | — | — | Current dfbdf6b CUDA probe 20261013_200158: EagerSession.scale creates a host scalar through constant_from, then mul fails in to_contiguous_read (expected CubeCL GPU tensor view, got host tensor). No public scale parameter can replace the internal scalar; execution unsupported, not numerically verified. |
| add_f32_n1024 | pytorch-cuda | 256 | 15 | 2.507 | 0 |
| add_f64_n1024 | pytorch-cuda | 256 | 15 | 2.498 | 0 |
| sub_f32_n1024 | pytorch-cuda | 256 | 15 | 2.536 | 0 |
| sub_f64_n1024 | pytorch-cuda | 256 | 15 | 2.513 | 0 |
| mul_f32_n1024 | pytorch-cuda | 256 | 15 | 2.478 | 0 |
| mul_f64_n1024 | pytorch-cuda | 256 | 15 | 2.456 | 0 |
| div_f32_n1024 | pytorch-cuda | 256 | 15 | 2.446 | 0 |
| div_f64_n1024 | pytorch-cuda | 256 | 15 | 2.373 | 0 |
| rem_f32_n1024 | pytorch-cuda | 256 | 15 | 2.514 | 0 |
| rem_f64_n1024 | pytorch-cuda | 256 | 15 | 2.485 | 0 |
| pow_f32_n1024 | pytorch-cuda | 256 | 15 | 2.542 | 0.000117 |
| pow_f64_n1024 | pytorch-cuda | 128 | 15 | 3.957 | 1.48e-07 |
| maximum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.635 | 0 |
| maximum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.604 | 0 |
| minimum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.573 | 0 |
| minimum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.452 | 0 |
| neg_f32_n1024 | pytorch-cuda | 256 | 15 | 2.381 | 0 |
| neg_f64_n1024 | pytorch-cuda | 256 | 15 | 2.363 | 0 |
| abs_f32_n1024 | pytorch-cuda | 256 | 15 | 2.982 | 0 |
| abs_f64_n1024 | pytorch-cuda | 256 | 15 | 3.071 | 0 |
| sign_f32_n1024 | pytorch-cuda | 256 | 15 | 2.398 | 0 |
| sign_f64_n1024 | pytorch-cuda | 256 | 15 | 2.339 | 0 |
| conj_c32_n1024 | pytorch-cuda | 256 | 15 | 2.773 | 0 |
| conj_c64_n1024 | pytorch-cuda | 256 | 15 | 3.099 | 0 |
| exp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.369 | 0.0001272 |
| exp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.323 | 1.095e-07 |
| expm1_f32_n1024 | pytorch-cuda | 256 | 15 | 2.410 | 7.911e-05 |
| expm1_f64_n1024 | pytorch-cuda | 256 | 15 | 2.380 | 7.39e-08 |
| log_f32_n1024 | pytorch-cuda | 256 | 15 | 2.288 | 7.945e-05 |
| log_f64_n1024 | pytorch-cuda | 256 | 15 | 2.794 | 4.231e-08 |
| log1p_f32_n1024 | pytorch-cuda | 256 | 15 | 2.279 | 0.0001189 |
| log1p_f64_n1024 | pytorch-cuda | 256 | 15 | 2.886 | 1.108e-07 |
| erf_f32_n1024 | pytorch-cuda | 256 | 15 | 2.291 | 7.914e-05 |
| erf_f64_n1024 | pytorch-cuda | 128 | 15 | 2.029 | 8.536e-08 |
| sin_f32_n1024 | pytorch-cuda | 256 | 15 | 2.363 | 4.348e-05 |
| sin_f64_n1024 | pytorch-cuda | 256 | 15 | 2.311 | 4.153e-08 |
| cos_f32_n1024 | pytorch-cuda | 256 | 15 | 2.334 | 6.342e-05 |
| cos_f64_n1024 | pytorch-cuda | 256 | 15 | 2.281 | 5.905e-08 |
| tanh_f32_n1024 | pytorch-cuda | 256 | 15 | 2.294 | 9.107e-05 |
| tanh_f64_n1024 | pytorch-cuda | 256 | 15 | 2.330 | 4.434e-08 |
| sqrt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.367 | 0 |
| sqrt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.386 | 0 |
| rsqrt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.422 | 6.983e-05 |
| rsqrt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.292 | 6.504e-08 |
| compare_f32_n1024 | pytorch-cuda | 256 | 15 | 2.459 | 0 |
| compare_f64_n1024 | pytorch-cuda | 256 | 15 | 2.384 | 0 |
| where_select_f32_n1024 | pytorch-cuda | 256 | 15 | 3.184 | 0 |
| where_select_f64_n1024 | pytorch-cuda | 256 | 15 | 3.213 | 0 |
| clamp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.756 | 0 |
| clamp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.807 | 0 |
| reduce_sum_f32_n1024 | pytorch-cuda | 256 | 15 | 3.583 | 0.0002742 |
| reduce_sum_f64_n1024 | pytorch-cuda | 256 | 15 | 3.587 | 1.06e-07 |
| reduce_prod_f32_n1024 | pytorch-cuda | 256 | 15 | 3.157 | 0.0003579 |
| reduce_prod_f64_n1024 | pytorch-cuda | 256 | 15 | 3.179 | 4.441e-07 |
| reduce_max_f32_n1024 | pytorch-cuda | 256 | 15 | 3.449 | 0 |
| reduce_max_f64_n1024 | pytorch-cuda | 256 | 15 | 3.419 | 0 |
| reduce_min_f32_n1024 | pytorch-cuda | 256 | 15 | 3.482 | 0 |
| reduce_min_f64_n1024 | pytorch-cuda | 256 | 15 | 3.509 | 0 |
| reduce_sum_squares_f32_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | pytorch-cuda | 256 | 15 | 3.872 | 1.439e-05 |
| reduce_mean_f64_n1024 | pytorch-cuda | 128 | 15 | 1.908 | 6.71e-09 |
| add_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.551 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| sub_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.543 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.489 | 0 |
| mul_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.487 | 0 |
| div_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.558 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.028 | 0 |
| rem_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.552 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.503 | 0 |
| pow_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.660 | 0.0001577 |
| pow_f64_n1048576 | pytorch-cuda | 2 | 15 | 2.041 | 1.48e-07 |
| maximum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.552 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.510 | 0 |
| minimum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.510 | 0 |
| neg_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.748 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.697 | 0 |
| abs_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.754 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.699 | 0 |
| sign_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.750 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.703 | 0 |
| conj_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.701 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.665 | 0 |
| exp_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.757 | 0.0001272 |
| exp_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.897 | 1.095e-07 |
| expm1_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.760 | 7.915e-05 |
| expm1_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.604 | 7.39e-08 |
| log_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.759 | 7.946e-05 |
| log_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.809 | 4.308e-08 |
| log1p_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.759 | 0.000119 |
| log1p_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.908 | 1.11e-07 |
| erf_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.757 | 7.923e-05 |
| erf_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.126 | 8.536e-08 |
| sin_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.767 | 4.348e-05 |
| sin_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.399 | 4.153e-08 |
| cos_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.765 | 6.342e-05 |
| cos_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.389 | 5.905e-08 |
| tanh_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.755 | 9.135e-05 |
| tanh_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.684 | 4.437e-08 |
| sqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.952 | 0 |
| rsqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.754 | 6.983e-05 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.975 | 6.504e-08 |
| compare_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.951 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.587 | 0 |
| where_select_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.753 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.590 | 0 |
| clamp_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.811 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.997 | 0 |
| reduce_sum_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.080 | 0.002948 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.075 | 2.961e-06 |
| reduce_prod_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.065 | 0.002623 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.062 | 4.216e-06 |
| reduce_max_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.118 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.674 | 0 |
| reduce_min_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.117 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.674 | 0 |
| reduce_sum_squares_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.082 | 2.947e-06 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.104 | 3.9e-09 |
| reduce_sum_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.009 | 0.0002666 |
| reduce_prod_full_f32_n1024 | pytorch-cuda | 256 | 15 | 2.712 | 0.0025 |
| reduce_max_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.163 | 0 |
| reduce_min_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.041 | 0 |
| compare_eq_f32_n1024 | pytorch-cuda | 256 | 15 | 2.372 | 0 |
| compare_lt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.383 | 0 |
| compare_le_f32_n1024 | pytorch-cuda | 256 | 15 | 2.493 | 0 |
| compare_ge_f32_n1024 | pytorch-cuda | 256 | 15 | 2.412 | 0 |
| reduce_sum_full_f64_n1024 | pytorch-cuda | 256 | 15 | 3.056 | 6.208e-07 |
| reduce_prod_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.675 | 1.663e-06 |
| reduce_max_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.969 | 0 |
| reduce_min_full_f64_n1024 | pytorch-cuda | 256 | 15 | 3.044 | 0 |
| compare_eq_f64_n1024 | pytorch-cuda | 256 | 15 | 2.456 | 0 |
| compare_lt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.352 | 0 |
| compare_le_f64_n1024 | pytorch-cuda | 256 | 15 | 2.392 | 0 |
| compare_ge_f64_n1024 | pytorch-cuda | 256 | 15 | 2.420 | 0 |
| reduce_sum_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.369 | 0.00522 |
| reduce_prod_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.360 | 0.5197 |
| reduce_max_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.398 | 0 |
| reduce_min_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.416 | 0 |
| compare_eq_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.883 | 0 |
| compare_lt_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.887 | 0 |
| compare_le_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.874 | 0 |
| compare_ge_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.874 | 0 |
| reduce_sum_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.397 | 0.0004425 |
| reduce_prod_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.400 | 0.0001269 |
| reduce_max_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.862 | 0 |
| reduce_min_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.862 | 0 |
| compare_eq_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.630 | 0 |
| compare_lt_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.583 | 0 |
| compare_le_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.583 | 0 |
| compare_ge_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.582 | 0 |
| reduce_mean_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.108 | 0.0002906 |
| reduce_sum_squares_full_f32_n1024 | pytorch-cuda | 128 | 15 | 2.065 | 0 |
| reduce_mean_full_f64_n1024 | pytorch-cuda | 256 | 15 | 3.176 | 8.614e-07 |
| reduce_sum_squares_full_f64_n1024 | pytorch-cuda | 256 | 15 | 3.833 | 0 |
| reduce_mean_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.364 | 0.00522 |
| reduce_sum_squares_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.261 | 0 |
| reduce_mean_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.400 | 0.001223 |
| reduce_sum_squares_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.129 | 1.664e-07 |
| scale_real_f32_n1024 | pytorch-cuda | 256 | 15 | 2.717 | 0 |
| scale_real_f64_n1024 | pytorch-cuda | 256 | 15 | 2.663 | 0 |
| scale_real_c32_n1024 | pytorch-cuda | 256 | 15 | 2.633 | 0 |
| scale_real_c64_n1024 | pytorch-cuda | 256 | 15 | 2.647 | 0 |
| scale_complex_c32_n1024 | pytorch-cuda | 256 | 15 | 2.620 | 4.666e-05 |
| scale_complex_c64_n1024 | pytorch-cuda | 256 | 15 | 2.599 | 0 |
| scale_real_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.801 | 0 |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.719 | 0 |
| scale_real_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.717 | 0 |
| scale_real_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.677 | 0 |
| scale_complex_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.718 | 4.516e-05 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.678 | 0 |
| add_f32_n1024 | jax-cuda | 32 | 15 | 3.471 | 0 |
| add_f64_n1024 | jax-cuda | 32 | 15 | 3.324 | 0 |
| sub_f32_n1024 | jax-cuda | 32 | 15 | 2.963 | 0 |
| sub_f64_n1024 | jax-cuda | 32 | 15 | 2.807 | 0 |
| mul_f32_n1024 | jax-cuda | 32 | 15 | 2.767 | 0 |
| mul_f64_n1024 | jax-cuda | 32 | 15 | 2.711 | 0 |
| div_f32_n1024 | jax-cuda | 32 | 15 | 2.500 | 4.875e-05 |
| div_f64_n1024 | jax-cuda | 32 | 15 | 2.517 | 0 |
| rem_f32_n1024 | jax-cuda | 32 | 15 | 3.036 | 0 |
| rem_f64_n1024 | jax-cuda | 32 | 15 | 3.389 | 0 |
| pow_f32_n1024 | jax-cuda | 16 | 15 | 2.227 | 0.000117 |
| pow_f64_n1024 | jax-cuda | 16 | 15 | 2.906 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda | 16 | 15 | 2.498 | 0 |
| maximum_f64_n1024 | jax-cuda | 16 | 15 | 2.099 | 0 |
| minimum_f32_n1024 | jax-cuda | 16 | 15 | 2.238 | 0 |
| minimum_f64_n1024 | jax-cuda | 16 | 15 | 2.197 | 0 |
| neg_f32_n1024 | jax-cuda | 16 | 15 | 2.099 | 0 |
| neg_f64_n1024 | jax-cuda | 16 | 15 | 2.047 | 0 |
| abs_f32_n1024 | jax-cuda | 32 | 15 | 3.182 | 0 |
| abs_f64_n1024 | jax-cuda | 32 | 15 | 2.949 | 0 |
| sign_f32_n1024 | jax-cuda | 32 | 15 | 3.079 | 0 |
| sign_f64_n1024 | jax-cuda | 32 | 15 | 2.913 | 0 |
| conj_c32_n1024 | jax-cuda | 32 | 15 | 3.032 | 0 |
| conj_c64_n1024 | jax-cuda | 32 | 15 | 2.993 | 0 |
| exp_f32_n1024 | jax-cuda | 32 | 15 | 3.074 | 0.0001272 |
| exp_f64_n1024 | jax-cuda | 32 | 15 | 3.299 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda | 32 | 15 | 3.162 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda | 32 | 15 | 3.131 | 7.39e-08 |
| log_f32_n1024 | jax-cuda | 32 | 15 | 3.233 | 7.945e-05 |
| log_f64_n1024 | jax-cuda | 32 | 15 | 2.996 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda | 32 | 15 | 3.156 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda | 32 | 15 | 3.421 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda | 16 | 15 | 3.084 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda | 16 | 15 | 2.948 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda | 16 | 15 | 2.493 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda | 16 | 15 | 2.618 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda | 16 | 15 | 2.679 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda | 16 | 15 | 2.508 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda | 16 | 15 | 2.782 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda | 16 | 15 | 2.192 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda | 16 | 15 | 2.263 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda | 16 | 15 | 2.247 | 0 |
| rsqrt_f32_n1024 | jax-cuda | 16 | 15 | 2.969 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda | 16 | 15 | 2.770 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda | 32 | 15 | 3.344 | 0 |
| compare_f64_n1024 | jax-cuda | 32 | 15 | 3.173 | 0 |
| where_select_f32_n1024 | jax-cuda | 32 | 15 | 3.456 | 0 |
| where_select_f64_n1024 | jax-cuda | 32 | 15 | 3.472 | 0 |
| clamp_f32_n1024 | jax-cuda | 32 | 15 | 3.184 | 0 |
| clamp_f64_n1024 | jax-cuda | 32 | 15 | 3.211 | 0 |
| reduce_sum_f32_n1024 | jax-cuda | 16 | 15 | 2.705 | 0.0002742 |
| reduce_sum_f64_n1024 | jax-cuda | 16 | 15 | 2.622 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda | 16 | 15 | 2.035 | 0.0003574 |
| reduce_prod_f64_n1024 | jax-cuda | 16 | 15 | 2.691 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda | 16 | 15 | 2.781 | 0 |
| reduce_max_f64_n1024 | jax-cuda | 16 | 15 | 2.051 | 0 |
| reduce_min_f32_n1024 | jax-cuda | 16 | 15 | 2.482 | 0 |
| reduce_min_f64_n1024 | jax-cuda | 32 | 15 | 3.023 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda | 16 | 15 | 2.063 | 1.439e-05 |
| reduce_mean_f64_n1024 | jax-cuda | 16 | 15 | 2.202 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda | 16 | 15 | 1.702 | 0 |
| add_f64_n1048576 | jax-cuda | 16 | 15 | 1.727 | 0 |
| sub_f32_n1048576 | jax-cuda | 32 | 15 | 3.379 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 3.318 | 0 |
| mul_f32_n1048576 | jax-cuda | 32 | 15 | 3.408 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 3.148 | 0 |
| div_f32_n1048576 | jax-cuda | 32 | 15 | 2.879 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda | 16 | 15 | 2.065 | 0 |
| rem_f32_n1048576 | jax-cuda | 16 | 15 | 1.952 | 0 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 3.161 | 0 |
| pow_f32_n1048576 | jax-cuda | 16 | 15 | 2.044 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda | 2 | 15 | 2.555 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda | 16 | 15 | 1.940 | 0 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 3.273 | 0 |
| minimum_f32_n1048576 | jax-cuda | 32 | 15 | 3.345 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 3.174 | 0 |
| neg_f32_n1048576 | jax-cuda | 32 | 15 | 3.112 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 3.335 | 0 |
| abs_f32_n1048576 | jax-cuda | 32 | 15 | 2.893 | 0 |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 2.893 | 0 |
| sign_f32_n1048576 | jax-cuda | 32 | 15 | 2.887 | 0 |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 2.976 | 0 |
| conj_c32_n1048576 | jax-cuda | 32 | 15 | 2.860 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 1.894 | 0 |
| exp_f32_n1048576 | jax-cuda | 32 | 15 | 2.767 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda | 16 | 15 | 3.272 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda | 32 | 15 | 3.046 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda | 8 | 15 | 2.122 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda | 32 | 15 | 3.068 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda | 8 | 15 | 3.127 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda | 16 | 15 | 1.959 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda | 8 | 15 | 3.307 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda | 16 | 15 | 2.466 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda | 4 | 15 | 2.754 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda | 32 | 15 | 3.119 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda | 8 | 15 | 2.025 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda | 32 | 15 | 3.309 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda | 8 | 15 | 2.022 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda | 32 | 15 | 3.056 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda | 16 | 15 | 3.059 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda | 32 | 15 | 3.029 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 3.267 | 0 |
| rsqrt_f32_n1048576 | jax-cuda | 16 | 15 | 2.088 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda | 16 | 15 | 2.218 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda | 32 | 15 | 3.043 | 0 |
| compare_f64_n1048576 | jax-cuda | 32 | 15 | 2.703 | 0 |
| where_select_f32_n1048576 | jax-cuda | 32 | 15 | 3.241 | 0 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 3.257 | 0 |
| clamp_f32_n1048576 | jax-cuda | 32 | 15 | 2.995 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 2.840 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda | 16 | 15 | 2.143 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda | 16 | 15 | 2.509 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda | 16 | 15 | 2.485 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda | 16 | 15 | 2.646 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda | 16 | 15 | 2.571 | 0 |
| reduce_max_f64_n1048576 | jax-cuda | 16 | 15 | 2.574 | 0 |
| reduce_min_f32_n1048576 | jax-cuda | 16 | 15 | 2.581 | 0 |
| reduce_min_f64_n1048576 | jax-cuda | 16 | 15 | 2.596 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda | 16 | 15 | 2.764 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda | 16 | 15 | 2.719 | 6.5e-09 |
| reduce_sum_full_f32_n1024 | jax-cuda | 16 | 15 | 2.165 | 0.0002666 |
| reduce_prod_full_f32_n1024 | jax-cuda | 16 | 15 | 2.058 | 0.001786 |
| reduce_max_full_f32_n1024 | jax-cuda | 16 | 15 | 2.948 | 0 |
| reduce_min_full_f32_n1024 | jax-cuda | 16 | 15 | 3.002 | 0 |
| compare_eq_f32_n1024 | jax-cuda | 16 | 15 | 2.500 | 0 |
| compare_lt_f32_n1024 | jax-cuda | 16 | 15 | 2.139 | 0 |
| compare_le_f32_n1024 | jax-cuda | 16 | 15 | 1.940 | 0 |
| compare_ge_f32_n1024 | jax-cuda | 16 | 15 | 2.043 | 0 |
| reduce_sum_full_f64_n1024 | jax-cuda | 16 | 15 | 2.854 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | jax-cuda | 16 | 15 | 2.831 | 4.435e-07 |
| reduce_max_full_f64_n1024 | jax-cuda | 16 | 15 | 2.397 | 0 |
| reduce_min_full_f64_n1024 | jax-cuda | 32 | 15 | 3.093 | 0 |
| compare_eq_f64_n1024 | jax-cuda | 32 | 15 | 3.201 | 0 |
| compare_lt_f64_n1024 | jax-cuda | 32 | 15 | 3.202 | 0 |
| compare_le_f64_n1024 | jax-cuda | 32 | 15 | 3.078 | 0 |
| compare_ge_f64_n1024 | jax-cuda | 32 | 15 | 3.126 | 0 |
| reduce_sum_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.312 | 0.003872 |
| reduce_prod_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.062 | 0.5323 |
| reduce_max_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.848 | 0 |
| reduce_min_full_f32_n1048576 | jax-cuda | 16 | 15 | 3.221 | 0 |
| compare_eq_f32_n1048576 | jax-cuda | 16 | 15 | 1.961 | 0 |
| compare_lt_f32_n1048576 | jax-cuda | 32 | 15 | 3.210 | 0 |
| compare_le_f32_n1048576 | jax-cuda | 32 | 15 | 3.085 | 0 |
| compare_ge_f32_n1048576 | jax-cuda | 32 | 15 | 3.041 | 0 |
| reduce_sum_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.408 | 0.0006452 |
| reduce_prod_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.900 | 0.0001122 |
| reduce_max_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.820 | 0 |
| reduce_min_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.364 | 0 |
| compare_eq_f64_n1048576 | jax-cuda | 32 | 15 | 3.341 | 0 |
| compare_lt_f64_n1048576 | jax-cuda | 32 | 15 | 3.198 | 0 |
| compare_le_f64_n1048576 | jax-cuda | 32 | 15 | 3.245 | 0 |
| compare_ge_f64_n1048576 | jax-cuda | 32 | 15 | 3.188 | 0 |
| reduce_mean_full_f32_n1024 | jax-cuda | 32 | 15 | 3.360 | 0.0002906 |
| reduce_sum_squares_full_f32_n1024 | jax-cuda | 16 | 15 | 2.512 | 0.0001766 |
| reduce_mean_full_f64_n1024 | jax-cuda | 16 | 15 | 3.043 | 5.169e-07 |
| reduce_sum_squares_full_f64_n1024 | jax-cuda | 16 | 15 | 2.668 | 0 |
| reduce_mean_full_f32_n1048576 | jax-cuda | 32 | 15 | 3.674 | 0.003872 |
| reduce_sum_squares_full_f32_n1048576 | jax-cuda | 32 | 15 | 3.444 | 0 |
| reduce_mean_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.420 | 0.001783 |
| reduce_sum_squares_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.724 | 0 |
| scale_real_f32_n1024 | jax-cuda | 16 | 15 | 2.489 | 0 |
| scale_real_f64_n1024 | jax-cuda | 16 | 15 | 2.820 | 0 |
| scale_real_c32_n1024 | jax-cuda | 16 | 15 | 2.261 | 0 |
| scale_real_c64_n1024 | jax-cuda | 16 | 15 | 2.269 | 0 |
| scale_complex_c32_n1024 | jax-cuda | 16 | 15 | 1.914 | 7.929e-05 |
| scale_complex_c64_n1024 | jax-cuda | 32 | 15 | 3.495 | 3.753e-08 |
| scale_real_f32_n1048576 | jax-cuda | 32 | 15 | 3.532 | 0 |
| scale_real_f64_n1048576 | jax-cuda | 32 | 15 | 3.535 | 0 |
| scale_real_c32_n1048576 | jax-cuda | 32 | 15 | 3.578 | 0 |
| scale_real_c64_n1048576 | jax-cuda | 16 | 15 | 1.936 | 0 |
| scale_complex_c32_n1048576 | jax-cuda | 32 | 15 | 3.550 | 6.186e-05 |
| scale_complex_c64_n1048576 | jax-cuda | 16 | 15 | 1.853 | 4.129e-08 |
| add_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.112 | 0 |
| add_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.056 | 0 |
| sub_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.753 | 0 |
| sub_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.589 | 0 |
| mul_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.543 | 0 |
| mul_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.501 | 0 |
| div_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.522 | 4.875e-05 |
| div_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.630 | 0 |
| rem_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.039 | 0 |
| rem_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.361 | 0 |
| pow_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.985 | 0.000117 |
| pow_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.311 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.086 | 0 |
| maximum_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.021 | 0 |
| minimum_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.263 | 0 |
| minimum_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.140 | 0 |
| neg_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.932 | 0 |
| neg_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.930 | 0 |
| abs_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.985 | 0 |
| abs_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.988 | 0 |
| sign_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.017 | 0 |
| sign_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.131 | 0 |
| conj_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.948 | 0 |
| conj_c64_n1024 | jax-cuda-jit | 32 | 15 | 3.050 | 0 |
| exp_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.954 | 0.0001272 |
| exp_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.131 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.916 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.205 | 7.39e-08 |
| log_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.143 | 7.945e-05 |
| log_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.325 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.250 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.188 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.176 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.020 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.368 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.606 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.552 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.557 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.271 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.284 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.029 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.171 | 0 |
| rsqrt_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.035 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.985 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.986 | 0 |
| compare_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.225 | 0 |
| where_select_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.201 | 0 |
| where_select_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.243 | 0 |
| clamp_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.245 | 0 |
| clamp_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.271 | 0 |
| reduce_sum_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.907 | 0.0001139 |
| reduce_sum_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.935 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.518 | 0.0003579 |
| reduce_prod_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.072 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.662 | 0 |
| reduce_max_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.472 | 0 |
| reduce_min_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.243 | 0 |
| reduce_min_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.749 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.120 | 7.204e-06 |
| reduce_mean_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.002 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.138 | 0 |
| add_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.945 | 0 |
| sub_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.520 | 0 |
| sub_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.671 | 0 |
| mul_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.250 | 0 |
| mul_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.687 | 0 |
| div_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.260 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.916 | 0 |
| rem_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.756 | 0 |
| rem_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.249 | 0 |
| pow_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.240 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda-jit | 2 | 15 | 2.599 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.873 | 0 |
| maximum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.967 | 0 |
| minimum_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.141 | 0 |
| minimum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.978 | 0 |
| neg_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.008 | 0 |
| neg_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.985 | 0 |
| abs_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.551 | 0 |
| abs_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.152 | 0 |
| sign_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.283 | 0 |
| sign_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.139 | 0 |
| conj_c32_n1048576 | jax-cuda-jit | 32 | 15 | 2.129 | 0 |
| conj_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.868 | 0 |
| exp_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.294 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.216 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.510 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.119 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.561 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.154 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.024 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.366 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.090 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda-jit | 4 | 15 | 2.668 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.232 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.811 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.253 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.807 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.997 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.022 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.137 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.284 | 0 |
| rsqrt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.044 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.877 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.145 | 0 |
| compare_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.216 | 0 |
| where_select_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.148 | 0 |
| where_select_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.047 | 0 |
| clamp_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.286 | 0 |
| clamp_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.116 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.463 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.608 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.452 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.494 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.415 | 0 |
| reduce_max_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.433 | 0 |
| reduce_min_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.595 | 0 |
| reduce_min_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.612 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.895 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.498 | 6.5e-09 |
| reduce_sum_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.136 | 0.0002666 |
| reduce_prod_full_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.194 | 0.001786 |
| reduce_max_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.352 | 0 |
| reduce_min_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.714 | 0 |
| compare_eq_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.223 | 0 |
| compare_lt_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.336 | 0 |
| compare_le_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.249 | 0 |
| compare_ge_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.230 | 0 |
| reduce_sum_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.998 | 4.966e-07 |
| reduce_prod_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.991 | 1.774e-06 |
| reduce_max_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.963 | 0 |
| reduce_min_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.976 | 0 |
| compare_eq_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.214 | 0 |
| compare_lt_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.213 | 0 |
| compare_le_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.787 | 0 |
| compare_ge_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.317 | 0 |
| reduce_sum_full_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.324 | 0.003872 |
| reduce_prod_full_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.404 | 0.5323 |
| reduce_max_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.519 | 0 |
| reduce_min_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.430 | 0 |
| compare_eq_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.964 | 0 |
| compare_lt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.232 | 0 |
| compare_le_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.649 | 0 |
| compare_ge_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.277 | 0 |
| reduce_sum_full_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.748 | 0.0006452 |
| reduce_prod_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.085 | 0.0001105 |
| reduce_max_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.417 | 0 |
| reduce_min_full_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.521 | 0 |
| compare_eq_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.902 | 0 |
| compare_lt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.223 | 0 |
| compare_le_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.239 | 0 |
| compare_ge_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.214 | 0 |
| reduce_mean_full_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.410 | 0.0002906 |
| reduce_sum_squares_full_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.066 | 0.0001766 |
| reduce_mean_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.769 | 6.891e-07 |
| reduce_sum_squares_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.465 | 0 |
| reduce_mean_full_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.633 | 0.003872 |
| reduce_sum_squares_full_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.576 | 0 |
| reduce_mean_full_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.931 | 0.001783 |
| reduce_sum_squares_full_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.393 | 0 |
| scale_real_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.046 | 0 |
| scale_real_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.169 | 0 |
| scale_real_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.434 | 0 |
| scale_real_c64_n1024 | jax-cuda-jit | 32 | 15 | 3.313 | 0 |
| scale_complex_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.291 | 7.929e-05 |
| scale_complex_c64_n1024 | jax-cuda-jit | 32 | 15 | 3.210 | 3.753e-08 |
| scale_real_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.287 | 0 |
| scale_real_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.336 | 0 |
| scale_real_c32_n1048576 | jax-cuda-jit | 32 | 15 | 3.311 | 0 |
| scale_real_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.856 | 0 |
| scale_complex_c32_n1048576 | jax-cuda-jit | 32 | 15 | 3.313 | 6.186e-05 |
| scale_complex_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.876 | 4.129e-08 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: benchmarks/gpu/isolated_ops.yaml
timestamp: '2026-10-13T20:10:23Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: ad8e1351f2f0fda84ca875ec592c1282c9638bb5
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
  expected_cases: 176
  selected_cases: 176
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: e04b88277edb1bb3202f50da23a7b0d359a81675b9fe8a8a3692b1595961aebe
    src/bin/benchmark_gpu_public_api/plan.rs: 3d0f10f390ebd0e08a40deb5a64e49e086dcd41b9a777d3a208c6846ed6b2fac
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: da168bf8b7c30f748f62839cf0bf7cce8f5bccf9c37b1c1d993fd70c466f2f76
    src/bin/benchmark_gpu_public_api/multi.rs: 1b477fef78260cf28aa31e515d57ab411f6241b91ffb30d625939538fdbc6bab
    src/bin/benchmark_gpu_public_api/lifetime.rs: c687a8b0329f6d3073073671b5d246acbb1f4dc3f983d1430da897a908ef1745
    src/bin/benchmark_gpu_public_api/read_layout.rs: 184d4508d0729f531ee8107e3b70aa6b77c5a9e952c2dc43606a868cd97631a2
    src/bin/benchmark_gpu_public_api/structural.rs: b8fd99ceb5c80fd3474974b0ba9aa57833ad6bd25383af3091c11c0b6ab94f6d
    src/bin/benchmark_gpu_public_api/linalg.rs: 5f5c4afc161f3d6e267861511d0d5871be9192d9968c95b59a73c9dc5a64e22b
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: 1dc14ac94fac700aa696d0ada04315685e3245738f9514de3d7989b8cb59152b
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 3faaaf5ae74a2ef9c91888b8fd8e566d1cf83b24d2b385cc1cd7b87d9d80c6fb
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: a591b03f182b50f2b15ffd0d2d4c359225aae96aa01d541e0ca3f95eac4b739d
    scripts/run_gpu_public_api.sh: dd5e4cc47e85fcb5afdc56d460136b2a399a6a7d6a62005de8ba94afb0119ecc
    data/results/nvidia-gpu/gpu/isolated_ops/20261013_201022/fixtures/manifest.json: 525424c0b2b9293434058269c2ec84bb255f11a6a75b802994780f3d0fd8ebb7
    target/release/benchmark_gpu_public_api: dd32d3055d16480b3dc368d5231fc6778eac2537cb00285228b0fa6593d3f068
  reference_numpy_version: 2.4.2
```
