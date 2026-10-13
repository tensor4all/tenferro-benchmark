# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **1078**. Unsupported cells: **62**, including **50** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261013_212123/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

I32/I64 outputs require exact integer equality without conversion to floating point. Division and remainder truncate toward zero; axis-1 sum preserves the input dtype. Arithmetic fixtures avoid overflow, and I64 unary/order fixtures include values above 2**53 to detect loss of integer precision. A zero verification value in these rows means exact equality.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_f32_n1024 | 13.131 (0.523) | 22.355 (0.968) | 9.233 (0.088) | 109.834 (1.215) | 105.996 (6.015) |
| add_f64_n1024 | 16.008 (0.977) | 30.123 (1.967) | 9.208 (0.097) | 129.984 (30.726) | 95.887 (17.350) |
| sub_f32_n1024 | 17.169 (1.044) | 27.553 (1.048) | 9.348 (0.277) | 127.580 (14.435) | 89.928 (15.592) |
| sub_f64_n1024 | 17.190 (1.199) | 25.755 (3.643) | 9.502 (0.068) | 131.777 (9.006) | 83.395 (10.606) |
| mul_f32_n1024 | 17.896 (1.090) | 25.648 (1.256) | 9.031 (0.503) | 126.842 (12.577) | 77.094 (3.468) |
| mul_f64_n1024 | 18.280 (1.041) | 28.196 (1.798) | 9.220 (0.109) | 121.564 (15.331) | 76.004 (9.172) |
| div_f32_n1024 | 18.391 (0.991) | 27.197 (3.241) | 9.033 (0.186) | 98.182 (16.359) | 77.325 (4.876) |
| div_f64_n1024 | 18.898 (0.684) | 26.403 (2.666) | 8.710 (0.099) | 96.937 (13.771) | 79.003 (7.972) |
| rem_f32_n1024 | 17.588 (0.116) | 165.899 (16.215) | 9.056 (0.137) | 108.720 (16.236) | 96.476 (14.447) |
| rem_f64_n1024 | 17.805 (0.238) | 233.120 (23.532) | 9.001 (0.081) | 129.316 (2.593) | 106.838 (18.564) |
| pow_f32_n1024 | 17.365 (0.293) | 209.174 (18.502) | 9.502 (0.108) | 190.786 (48.171) | 94.179 (18.704) |
| pow_f64_n1024 | 18.419 (0.536) | 248.366 (20.309) | 30.954 (0.005) | 193.178 (32.289) | 107.049 (15.523) |
| maximum_f32_n1024 | 17.562 (0.224) | 299.740 (10.388) | 9.240 (0.080) | 163.552 (35.700) | 97.816 (10.954) |
| maximum_f64_n1024 | 17.743 (0.175) | 226.652 (20.499) | 9.423 (0.411) | 139.022 (32.703) | 97.074 (11.181) |
| minimum_f32_n1024 | 18.051 (0.210) | 275.180 (10.582) | 9.684 (0.157) | 136.587 (13.084) | 90.457 (2.437) |
| minimum_f64_n1024 | 17.753 (0.123) | 289.878 (12.068) | 9.349 (0.304) | 132.697 (9.527) | 96.778 (4.462) |
| neg_f32_n1024 | 13.554 (1.955) | 20.395 (0.731) | 8.615 (0.295) | 126.851 (8.299) | 93.891 (7.969) |
| neg_f64_n1024 | 12.742 (1.963) | 20.120 (0.638) | 8.371 (0.102) | 127.244 (11.821) | 91.928 (3.207) |
| abs_f32_n1024 | 11.377 (1.196) | 146.150 (34.859) | 11.321 (0.859) | 95.630 (16.811) | 95.761 (8.843) |
| abs_f64_n1024 | 10.829 (1.096) | 134.308 (2.931) | 10.700 (0.135) | 91.897 (11.557) | 95.424 (9.533) |
| sign_f32_n1024 | 10.132 (1.077) | 137.339 (4.101) | 9.877 (1.941) | 91.035 (16.192) | 95.457 (10.217) |
| sign_f64_n1024 | 9.862 (1.054) | 142.882 (6.635) | 8.819 (0.928) | 96.715 (14.069) | 95.207 (12.778) |
| conj_c32_n1024 | 9.161 (0.742) | 141.342 (3.098) | 9.925 (0.102) | 93.532 (9.992) | 93.462 (9.407) |
| conj_c64_n1024 | 10.426 (0.778) | 124.109 (4.811) | 10.356 (0.157) | 91.526 (9.241) | 87.461 (10.077) |
| exp_f32_n1024 | 11.569 (1.928) | 19.594 (0.581) | 8.497 (0.439) | 93.633 (12.945) | 89.953 (11.542) |
| exp_f64_n1024 | 11.174 (1.312) | 18.809 (2.352) | 8.488 (0.085) | 92.646 (4.715) | 93.963 (6.287) |
| expm1_f32_n1024 | 9.828 (0.835) | 18.391 (1.347) | 8.828 (0.257) | 95.258 (9.289) | 97.476 (13.532) |
| expm1_f64_n1024 | 9.795 (0.681) | 18.320 (1.524) | 8.646 (0.320) | 99.076 (11.485) | 98.989 (10.165) |
| log_f32_n1024 | 9.330 (0.983) | 19.100 (1.180) | 8.553 (0.077) | 93.511 (12.486) | 98.460 (7.223) |
| log_f64_n1024 | 9.477 (0.946) | 20.054 (1.718) | 10.899 (0.005) | 100.570 (16.017) | 98.149 (10.067) |
| log1p_f32_n1024 | 8.862 (0.763) | 21.593 (0.964) | 8.800 (0.262) | 100.517 (17.416) | 96.031 (14.359) |
| log1p_f64_n1024 | 9.371 (0.515) | 22.055 (1.170) | 11.256 (0.007) | 106.773 (18.288) | 102.922 (22.118) |
| erf_f32_n1024 | 8.859 (0.666) | 21.603 (1.088) | 8.686 (0.125) | 184.893 (23.720) | 105.272 (19.392) |
| erf_f64_n1024 | 9.073 (0.749) | 21.949 (1.088) | 15.832 (0.006) | 183.594 (50.796) | 128.699 (20.719) |
| sin_f32_n1024 | 8.505 (0.754) | 22.149 (1.044) | 8.291 (0.174) | 157.633 (11.505) | 141.562 (9.874) |
| sin_f64_n1024 | 9.110 (1.066) | 22.185 (0.563) | 8.447 (0.475) | 156.739 (3.665) | 154.227 (20.658) |
| cos_f32_n1024 | 8.747 (0.588) | 21.560 (1.349) | 8.739 (0.393) | 163.673 (10.931) | 148.205 (20.521) |
| cos_f64_n1024 | 8.991 (0.849) | 21.128 (0.956) | 8.775 (0.683) | 147.925 (5.503) | 164.038 (13.224) |
| tanh_f32_n1024 | 8.674 (0.876) | 22.362 (0.834) | 9.589 (0.184) | 163.167 (20.056) | 151.111 (28.060) |
| tanh_f64_n1024 | 9.431 (0.539) | 22.717 (0.952) | 9.816 (0.214) | 136.924 (33.625) | 134.932 (22.610) |
| sqrt_f32_n1024 | 8.679 (0.567) | 23.743 (0.764) | 8.654 (0.238) | 145.062 (16.136) | 131.480 (14.727) |
| sqrt_f64_n1024 | 9.074 (0.772) | 22.614 (1.068) | 8.601 (0.153) | 136.089 (12.170) | 121.892 (32.755) |
| rsqrt_f32_n1024 | 8.899 (0.895) | 22.375 (0.335) | 8.532 (0.138) | 186.642 (44.134) | 131.889 (24.678) |
| rsqrt_f64_n1024 | 8.757 (0.207) | 22.401 (0.483) | 8.362 (0.094) | 178.439 (45.172) | 137.207 (15.195) |
| compare_f32_n1024 | unsupported | 428.809 (4.243) | 8.543 (0.157) | 102.253 (19.976) | 127.588 (15.221) |
| compare_f64_n1024 | unsupported | 438.778 (20.777) | 8.396 (0.107) | 93.713 (11.177) | 123.163 (17.940) |
| where_select_f32_n1024 | 20.770 (1.875) | 672.207 (37.094) | 11.911 (0.451) | 105.651 (9.952) | 128.971 (17.529) |
| where_select_f64_n1024 | 19.720 (1.377) | 698.652 (27.423) | 12.553 (0.444) | 104.894 (9.238) | 125.681 (12.334) |
| clamp_f32_n1024 | 52.733 (1.348) | 1141.543 (25.436) | 10.087 (0.241) | 97.615 (9.653) | 129.999 (12.368) |
| clamp_f64_n1024 | 52.653 (2.273) | 1122.666 (52.049) | 10.246 (0.108) | 97.475 (11.218) | 122.783 (17.783) |
| reduce_sum_f32_n1024 | 21.912 (0.292) | 217.025 (2.686) | 13.872 (0.175) | 164.832 (8.716) | 95.874 (10.551) |
| reduce_sum_f64_n1024 | 23.043 (0.296) | 212.307 (3.468) | 13.757 (0.412) | 128.913 (6.886) | 90.966 (3.505) |
| reduce_prod_f32_n1024 | 21.433 (0.350) | 230.203 (4.497) | 12.173 (0.068) | 97.443 (19.106) | 147.121 (17.894) |
| reduce_prod_f64_n1024 | 21.818 (1.390) | 223.856 (3.000) | 12.594 (0.239) | 93.856 (0.688) | 168.535 (9.341) |
| reduce_max_f32_n1024 | 21.795 (1.525) | 223.384 (5.012) | 13.449 (0.791) | 159.920 (14.157) | 160.927 (12.879) |
| reduce_max_f64_n1024 | 21.885 (1.795) | 212.442 (5.816) | 13.476 (0.325) | 128.058 (2.930) | 102.686 (17.023) |
| reduce_min_f32_n1024 | 24.373 (0.380) | 196.596 (3.608) | 13.545 (0.498) | 151.250 (17.362) | 100.637 (18.334) |
| reduce_min_f64_n1024 | 23.664 (0.516) | 196.299 (2.419) | 14.186 (0.646) | 90.236 (1.825) | 87.712 (1.379) |
| reduce_sum_squares_f32_n1024 | 21.643 (0.164) | 214.784 (2.983) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1024 | 22.546 (0.333) | 215.879 (8.450) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1024 | unsupported | unsupported | 13.477 (0.380) | 104.924 (12.101) | 93.071 (2.955) |
| reduce_mean_f64_n1024 | unsupported | unsupported | 14.131 (0.316) | 99.475 (0.498) | 94.387 (2.512) |
| add_f32_n1048576 | 48.191 (0.948) | 49.044 (0.541) | 39.752 (0.036) | 106.856 (1.016) | 93.552 (2.367) |
| add_f64_n1048576 | 83.554 (0.637) | 94.938 (1.339) | 77.775 (0.046) | 102.024 (1.312) | 87.768 (2.995) |
| sub_f32_n1048576 | 43.961 (0.049) | 44.350 (0.118) | 39.743 (0.029) | 102.534 (3.306) | 77.367 (4.994) |
| sub_f64_n1048576 | 82.865 (0.082) | 85.796 (0.218) | 77.753 (0.052) | 99.623 (3.920) | 87.038 (1.993) |
| mul_f32_n1048576 | 43.725 (0.110) | 44.386 (0.172) | 39.768 (0.045) | 103.811 (1.698) | 77.425 (4.809) |
| mul_f64_n1048576 | 82.736 (0.220) | 85.968 (0.268) | 77.677 (0.059) | 97.507 (0.857) | 83.471 (0.528) |
| div_f32_n1048576 | 43.822 (0.088) | 44.656 (0.121) | 39.989 (0.071) | 92.230 (1.068) | 75.112 (2.344) |
| div_f64_n1048576 | 95.989 (0.067) | 102.318 (1.028) | 94.412 (0.130) | 130.278 (1.911) | 122.022 (0.404) |
| rem_f32_n1048576 | 42.527 (0.208) | 97.835 (0.308) | 40.089 (0.097) | 94.138 (7.584) | 81.496 (5.815) |
| rem_f64_n1048576 | 81.358 (0.462) | 186.712 (0.128) | 78.193 (0.060) | 92.967 (6.289) | 83.246 (0.451) |
| pow_f32_n1048576 | 42.623 (0.114) | 97.524 (0.155) | 41.595 (0.110) | 124.307 (8.702) | 82.478 (13.077) |
| pow_f64_n1048576 | 1013.376 (0.579) | 1119.725 (0.560) | 1016.976 (0.554) | 1297.357 (15.465) | 1271.144 (19.033) |
| maximum_f32_n1048576 | 42.793 (0.288) | 97.757 (0.341) | 39.901 (0.026) | 124.922 (16.964) | 116.892 (17.177) |
| maximum_f64_n1048576 | 81.566 (0.462) | 186.779 (0.183) | 78.481 (0.136) | 98.134 (0.382) | 84.318 (1.472) |
| minimum_f32_n1048576 | 42.772 (0.301) | 97.741 (0.265) | 39.910 (0.045) | 102.449 (2.499) | 73.149 (6.452) |
| minimum_f64_n1048576 | 81.913 (0.418) | 186.738 (0.145) | 78.405 (0.071) | 97.897 (4.633) | 84.441 (0.438) |
| neg_f32_n1048576 | 29.577 (0.449) | 29.491 (0.357) | 27.337 (0.055) | 96.195 (2.590) | 75.100 (1.770) |
| neg_f64_n1048576 | 56.109 (0.359) | 57.244 (0.233) | 53.039 (0.046) | 103.770 (0.601) | 68.824 (1.247) |
| abs_f32_n1048576 | 29.587 (0.282) | 56.168 (0.183) | 27.435 (0.102) | 86.085 (0.500) | 72.638 (1.954) |
| abs_f64_n1048576 | 55.906 (0.238) | 107.152 (0.359) | 53.159 (0.080) | 82.943 (1.067) | 72.641 (1.611) |
| sign_f32_n1048576 | 29.772 (0.490) | 56.286 (0.227) | 27.355 (0.063) | 78.664 (9.629) | 73.954 (3.597) |
| sign_f64_n1048576 | 56.305 (0.626) | 107.112 (0.218) | 53.226 (0.038) | 64.772 (5.670) | 68.422 (2.261) |
| conj_c32_n1048576 | 55.549 (0.577) | 107.476 (0.067) | 52.985 (0.125) | 63.736 (1.932) | 67.898 (1.356) |
| conj_c64_n1048576 | 109.057 (0.671) | 209.396 (0.346) | 104.061 (0.111) | 116.463 (2.032) | 116.168 (0.557) |
| exp_f32_n1048576 | 29.632 (0.371) | 29.658 (0.169) | 27.397 (0.022) | 67.940 (6.598) | 74.843 (4.390) |
| exp_f64_n1048576 | 184.420 (0.469) | 187.067 (1.145) | 180.537 (0.078) | 204.443 (2.569) | 202.662 (2.432) |
| expm1_f32_n1048576 | 29.732 (0.222) | 29.759 (0.131) | 27.490 (0.033) | 88.538 (16.388) | 88.561 (14.391) |
| expm1_f64_n1048576 | 228.496 (0.298) | 230.618 (0.690) | 224.873 (0.294) | 268.406 (4.030) | 261.829 (1.919) |
| log_f32_n1048576 | 29.556 (0.245) | 29.818 (0.197) | 27.479 (0.023) | 92.568 (10.559) | 89.457 (15.346) |
| log_f64_n1048576 | 353.727 (0.322) | 355.560 (0.429) | 350.337 (0.401) | 408.912 (15.175) | 391.425 (10.606) |
| log1p_f32_n1048576 | 29.956 (0.263) | 29.810 (0.202) | 27.520 (0.091) | 95.339 (15.361) | 117.251 (17.341) |
| log1p_f64_n1048576 | 365.638 (1.231) | 367.678 (0.677) | 362.489 (0.173) | 418.291 (8.367) | 411.590 (9.840) |
| erf_f32_n1048576 | 29.777 (0.334) | 29.785 (0.239) | 27.488 (0.042) | 148.576 (18.642) | 134.564 (22.822) |
| erf_f64_n1048576 | 532.347 (1.446) | 534.378 (1.699) | 529.767 (0.220) | 675.802 (8.457) | 659.197 (18.516) |
| sin_f32_n1048576 | 29.791 (0.307) | 29.732 (0.140) | 27.557 (0.043) | 135.247 (14.618) | 128.916 (22.537) |
| sin_f64_n1048576 | 201.797 (0.754) | 204.210 (0.773) | 211.735 (0.125) | 239.573 (1.846) | 256.448 (2.546) |
| cos_f32_n1048576 | 29.679 (0.159) | 29.931 (0.152) | 27.591 (0.036) | 126.227 (14.483) | 127.306 (14.240) |
| cos_f64_n1048576 | 201.065 (0.808) | 203.188 (0.663) | 211.302 (0.107) | 256.032 (2.773) | 237.795 (0.900) |
| tanh_f32_n1048576 | 29.559 (0.271) | 29.904 (0.185) | 27.437 (0.028) | 86.962 (3.199) | 98.673 (7.944) |
| tanh_f64_n1048576 | 171.377 (0.829) | 174.571 (3.093) | 167.338 (0.108) | 190.578 (1.521) | 190.035 (1.803) |
| sqrt_f32_n1048576 | 29.682 (0.318) | 29.814 (0.045) | 27.600 (0.150) | 91.297 (6.928) | 90.868 (2.284) |
| sqrt_f64_n1048576 | 94.248 (0.152) | 95.472 (0.304) | 91.923 (0.041) | 104.077 (0.912) | 102.663 (0.723) |
| rsqrt_f32_n1048576 | 29.735 (0.453) | 29.780 (0.177) | 27.398 (0.029) | 131.471 (3.189) | 93.688 (2.725) |
| rsqrt_f64_n1048576 | 62.946 (0.241) | 64.108 (0.342) | 61.582 (0.064) | 134.141 (2.737) | 88.394 (2.306) |
| compare_f32_n1048576 | 32.137 (0.248) | 87.789 (0.312) | 30.543 (0.112) | 88.027 (2.398) | 94.614 (2.923) |
| compare_f64_n1048576 | 57.232 (0.172) | 164.845 (0.243) | 55.956 (0.038) | 86.333 (2.401) | 97.716 (2.238) |
| where_select_f32_n1048576 | 46.597 (0.111) | 117.655 (0.419) | 43.022 (0.024) | 100.324 (2.386) | 99.874 (4.899) |
| where_select_f64_n1048576 | 85.244 (0.297) | 206.534 (0.472) | 80.972 (0.103) | 98.232 (5.266) | 96.288 (3.897) |
| clamp_f32_n1048576 | 87.505 (0.955) | 603.795 (10.212) | 28.342 (0.082) | 92.319 (0.537) | 99.503 (3.798) |
| clamp_f64_n1048576 | 163.120 (1.271) | 558.269 (17.577) | 62.245 (0.220) | 89.316 (3.643) | 96.569 (3.329) |
| reduce_sum_f32_n1048576 | 40.274 (0.257) | 64.919 (1.078) | 16.275 (0.028) | 157.198 (2.116) | 131.431 (0.943) |
| reduce_sum_f64_n1048576 | 47.985 (0.716) | 95.485 (0.174) | 32.426 (0.022) | 158.682 (1.569) | 153.449 (8.857) |
| reduce_prod_f32_n1048576 | 43.882 (0.638) | 77.426 (1.205) | 16.157 (0.036) | 153.371 (2.562) | 156.639 (3.947) |
| reduce_prod_f64_n1048576 | 51.307 (0.516) | 95.433 (0.591) | 32.215 (0.074) | 153.473 (2.006) | 158.767 (3.716) |
| reduce_max_f32_n1048576 | 66.242 (0.481) | 76.802 (0.714) | 16.566 (0.027) | 155.069 (2.236) | 157.849 (4.319) |
| reduce_max_f64_n1048576 | 75.649 (3.173) | 112.789 (0.061) | 41.717 (0.035) | 150.490 (2.544) | 157.492 (4.774) |
| reduce_min_f32_n1048576 | 64.998 (0.412) | 76.284 (0.310) | 16.626 (0.205) | 159.564 (8.949) | 163.609 (15.563) |
| reduce_min_f64_n1048576 | 78.090 (0.526) | 112.325 (0.098) | 41.695 (0.034) | 165.483 (6.653) | 156.924 (6.263) |
| reduce_sum_squares_f32_n1048576 | 43.573 (0.562) | 83.233 (2.929) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1048576 | 63.716 (0.549) | 97.130 (0.310) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1048576 | unsupported | unsupported | 16.281 (0.034) | 164.476 (14.773) | 170.096 (19.828) |
| reduce_mean_f64_n1048576 | unsupported | unsupported | 32.855 (0.029) | 174.905 (17.000) | 150.486 (11.583) |
| reduce_sum_full_f32_n1024 | 15.224 (0.244) | 405.214 (52.615) | 12.550 (2.294) | 139.144 (8.436) | 160.524 (8.789) |
| reduce_prod_full_f32_n1024 | 14.824 (0.692) | 478.291 (18.367) | 10.520 (0.200) | 169.088 (10.945) | 173.415 (9.244) |
| reduce_max_full_f32_n1024 | 12.717 (0.149) | 481.436 (25.347) | 12.210 (0.135) | 156.322 (2.112) | 168.681 (4.070) |
| reduce_min_full_f32_n1024 | 12.659 (0.192) | 317.884 (19.171) | 12.249 (0.254) | 167.027 (14.606) | 170.590 (18.810) |
| compare_eq_f32_n1024 | unsupported | 593.013 (77.267) | 8.701 (0.442) | 149.071 (25.500) | 146.272 (28.265) |
| compare_lt_f32_n1024 | unsupported | 589.971 (89.225) | 8.645 (0.151) | 138.107 (22.445) | 132.910 (14.662) |
| compare_le_f32_n1024 | unsupported | 562.842 (53.434) | 8.715 (0.111) | 129.244 (19.518) | 121.962 (12.291) |
| compare_ge_f32_n1024 | unsupported | 546.053 (50.098) | 8.770 (0.792) | 124.662 (17.790) | 100.617 (12.925) |
| reduce_sum_full_f64_n1024 | 24.790 (0.407) | 186.230 (4.436) | 12.529 (0.400) | 101.522 (9.966) | 94.868 (5.788) |
| reduce_prod_full_f64_n1024 | 20.834 (1.830) | 184.791 (4.750) | 10.756 (0.072) | 97.726 (7.901) | 88.948 (4.851) |
| reduce_max_full_f64_n1024 | 18.935 (1.774) | 185.811 (1.544) | 12.366 (0.165) | 97.961 (5.534) | 85.703 (2.019) |
| reduce_min_full_f64_n1024 | 15.775 (1.376) | 187.662 (3.722) | 12.864 (0.712) | 85.480 (1.042) | 83.170 (2.258) |
| compare_eq_f64_n1024 | unsupported | 586.432 (41.946) | 9.194 (0.333) | 85.099 (5.354) | 92.997 (1.502) |
| compare_lt_f64_n1024 | unsupported | 583.678 (60.294) | 8.533 (0.295) | 81.555 (8.053) | 98.445 (7.192) |
| compare_le_f64_n1024 | unsupported | 586.880 (47.534) | 8.642 (0.240) | 86.851 (7.805) | 99.839 (7.336) |
| compare_ge_f64_n1024 | unsupported | 567.235 (55.264) | 8.820 (0.097) | 90.357 (7.020) | 99.542 (6.570) |
| reduce_sum_full_f32_n1048576 | 2546.618 (5.038) | 2650.226 (8.840) | 18.473 (0.032) | 105.986 (2.193) | 166.736 (10.447) |
| reduce_prod_full_f32_n1048576 | 2536.433 (5.898) | 2612.151 (4.348) | 18.441 (0.065) | 168.995 (6.706) | 161.488 (6.138) |
| reduce_max_full_f32_n1048576 | 9844.627 (5.935) | 9860.647 (4.838) | 18.751 (0.023) | 164.361 (3.397) | 162.038 (4.561) |
| reduce_min_full_f32_n1048576 | 9847.470 (7.957) | 9860.911 (4.150) | 18.873 (0.071) | 179.325 (23.065) | 181.039 (23.806) |
| compare_eq_f32_n1048576 | 33.337 (0.110) | 88.146 (0.549) | 30.292 (0.025) | 98.915 (9.751) | 119.145 (13.598) |
| compare_lt_f32_n1048576 | 33.297 (0.351) | 88.622 (1.843) | 30.281 (0.017) | 90.414 (0.780) | 94.901 (2.861) |
| compare_le_f32_n1048576 | 33.340 (0.426) | 88.252 (0.440) | 30.283 (0.021) | 91.165 (0.564) | 95.284 (2.663) |
| compare_ge_f32_n1048576 | 32.445 (0.332) | 87.942 (0.515) | 30.292 (0.009) | 91.744 (0.581) | 96.078 (3.207) |
| reduce_sum_full_f64_n1048576 | 3046.124 (2.218) | 3094.774 (0.903) | 37.483 (0.114) | 102.888 (2.595) | 109.883 (19.249) |
| reduce_prod_full_f64_n1048576 | 3045.410 (3.530) | 3094.376 (2.112) | 37.489 (0.031) | 153.608 (12.597) | 127.151 (2.978) |
| reduce_max_full_f64_n1048576 | 12521.546 (6.782) | 12527.971 (5.099) | 44.618 (0.052) | 166.200 (20.784) | 150.072 (13.651) |
| reduce_min_full_f64_n1048576 | 12523.027 (5.746) | 12529.672 (3.626) | 44.606 (0.023) | 132.862 (1.821) | 131.724 (1.988) |
| compare_eq_f64_n1048576 | 57.169 (0.153) | 164.538 (0.322) | 56.876 (0.064) | 91.090 (9.809) | 101.531 (8.518) |
| compare_lt_f64_n1048576 | 57.184 (0.197) | 164.541 (0.387) | 56.050 (0.091) | 86.295 (3.435) | 98.709 (2.943) |
| compare_le_f64_n1048576 | 57.207 (0.334) | 164.626 (0.445) | 56.035 (0.103) | 91.312 (3.590) | 93.574 (2.707) |
| compare_ge_f64_n1048576 | 57.105 (0.212) | 164.708 (0.390) | 55.965 (0.025) | 92.888 (3.091) | 91.210 (0.714) |
| reduce_mean_full_f32_n1024 | unsupported | unsupported | 12.077 (0.148) | 97.833 (1.043) | 97.926 (10.371) |
| reduce_sum_squares_full_f32_n1024 | 10.818 (0.306) | 233.741 (16.234) | 14.784 (0.365) | 86.273 (0.720) | 92.653 (2.545) |
| reduce_mean_full_f64_n1024 | unsupported | unsupported | 12.891 (0.333) | 100.164 (6.169) | 95.666 (3.113) |
| reduce_sum_squares_full_f64_n1024 | 9.191 (0.188) | 255.477 (24.485) | 14.323 (0.254) | 90.783 (3.300) | 155.729 (11.861) |
| reduce_mean_full_f32_n1048576 | unsupported | unsupported | 18.454 (0.072) | 104.911 (2.198) | 169.274 (30.349) |
| reduce_sum_squares_full_f32_n1048576 | 2558.246 (4.333) | 2681.517 (9.237) | 17.690 (0.022) | 101.144 (2.398) | 161.439 (2.156) |
| reduce_mean_full_f64_n1048576 | unsupported | unsupported | 37.564 (0.190) | 115.553 (6.888) | 106.543 (1.877) |
| reduce_sum_squares_full_f64_n1048576 | 3326.284 (2.303) | 3377.128 (1.899) | 33.300 (0.023) | 138.919 (4.480) | 126.253 (2.236) |
| scale_real_f32_n1024 | unsupported | unsupported | 10.278 (0.118) | 151.962 (34.443) | 121.645 (19.337) |
| scale_real_f64_n1024 | unsupported | unsupported | 10.186 (0.250) | 129.538 (19.535) | 127.083 (16.240) |
| scale_real_c32_n1024 | unsupported | unsupported | 10.409 (0.162) | 128.037 (29.473) | 102.398 (14.046) |
| scale_real_c64_n1024 | unsupported | unsupported | 10.482 (0.536) | 127.308 (14.106) | 98.947 (10.425) |
| scale_complex_c32_n1024 | unsupported | unsupported | 10.234 (0.266) | 101.280 (1.332) | 98.573 (8.116) |
| scale_complex_c64_n1024 | unsupported | unsupported | 10.205 (0.190) | 100.683 (0.176) | 99.928 (7.288) |
| scale_real_f32_n1048576 | unsupported | unsupported | 28.177 (0.114) | 103.330 (1.150) | 76.259 (4.420) |
| scale_real_f64_n1048576 | unsupported | unsupported | 53.736 (0.032) | 105.774 (0.767) | 70.344 (1.827) |
| scale_real_c32_n1048576 | unsupported | unsupported | 53.700 (0.035) | 105.325 (1.266) | 69.984 (0.633) |
| scale_real_c64_n1048576 | unsupported | unsupported | 104.810 (0.211) | 115.822 (0.762) | 116.359 (0.579) |
| scale_complex_c32_n1048576 | unsupported | unsupported | 53.686 (0.068) | 105.153 (0.586) | 70.735 (2.231) |
| scale_complex_c64_n1048576 | unsupported | unsupported | 104.921 (0.441) | 116.702 (2.261) | 116.551 (0.392) |
| add_i32_n1024 | 8.774 (0.581) | 12.506 (1.305) | 9.369 (0.136) | 94.954 (1.974) | 77.255 (1.832) |
| sub_i32_n1024 | 9.716 (0.933) | 12.164 (1.245) | 9.810 (0.263) | 97.772 (1.885) | 77.198 (2.440) |
| mul_i32_n1024 | 10.656 (1.034) | 12.102 (1.215) | 9.295 (0.126) | 102.552 (7.948) | 77.357 (4.968) |
| div_i32_n1024 | 67.073 (0.826) | 355.644 (31.945) | 10.397 (0.062) | 147.628 (7.738) | 78.635 (2.626) |
| rem_i32_n1024 | 76.121 (0.368) | 414.253 (8.393) | 9.450 (0.076) | 95.610 (13.657) | 80.200 (3.296) |
| maximum_i32_n1024 | 18.274 (0.720) | 294.785 (3.919) | 9.566 (0.223) | 97.080 (1.556) | 78.137 (5.266) |
| minimum_i32_n1024 | 18.320 (0.147) | 297.062 (4.054) | 9.565 (0.471) | 98.705 (0.531) | 78.692 (5.512) |
| neg_i32_n1024 | 16.436 (0.266) | 20.060 (0.159) | 8.674 (0.174) | 99.600 (5.544) | 76.074 (8.167) |
| abs_i32_n1024 | 16.806 (0.163) | 165.807 (4.076) | 11.093 (0.120) | 86.161 (0.470) | 75.574 (5.013) |
| sign_i32_n1024 | 16.571 (0.209) | 160.488 (2.379) | 8.729 (0.186) | 89.323 (4.911) | 76.854 (8.210) |
| reduce_sum_i32_n1024 | 23.509 (0.527) | 147.868 (2.425) | 15.366 (0.064) | 96.710 (2.548) | 74.312 (1.911) |
| reduce_max_i32_n1024 | 24.141 (0.963) | 143.201 (3.010) | 13.680 (0.282) | 90.583 (0.683) | 74.292 (3.359) |
| reduce_min_i32_n1024 | 24.759 (0.612) | 142.611 (1.557) | 14.804 (0.398) | 92.511 (3.827) | 77.596 (4.113) |
| add_i64_n1024 | 20.733 (0.702) | 27.797 (0.606) | 10.027 (0.661) | 102.958 (7.212) | 79.476 (6.384) |
| sub_i64_n1024 | 20.976 (0.188) | 25.719 (1.285) | 9.670 (0.135) | 103.932 (8.609) | 80.752 (6.633) |
| mul_i64_n1024 | 21.013 (0.498) | 26.385 (1.351) | 9.320 (0.189) | 97.296 (1.705) | 78.272 (6.279) |
| div_i64_n1024 | 101.845 (3.395) | 432.395 (31.027) | 10.584 (0.157) | 150.781 (7.716) | 83.028 (2.468) |
| rem_i64_n1024 | 95.472 (0.442) | 398.966 (10.528) | 9.171 (0.108) | 100.320 (16.513) | 89.254 (10.057) |
| maximum_i64_n1024 | 18.899 (0.204) | 273.861 (4.659) | 9.313 (0.132) | 103.415 (11.207) | 82.407 (4.409) |
| minimum_i64_n1024 | 18.954 (0.464) | 274.200 (3.266) | 9.192 (0.368) | 103.564 (9.122) | 78.617 (6.080) |
| neg_i64_n1024 | 16.989 (0.227) | 19.932 (0.216) | 8.854 (0.093) | 100.113 (6.617) | 76.186 (9.274) |
| abs_i64_n1024 | 16.562 (0.116) | 153.520 (4.710) | 11.291 (0.161) | 87.378 (6.881) | 79.839 (10.540) |
| sign_i64_n1024 | 16.268 (0.136) | 149.172 (1.431) | 9.168 (0.391) | 87.453 (2.833) | 78.167 (8.881) |
| reduce_sum_i64_n1024 | 22.826 (0.589) | 211.931 (5.472) | 15.808 (0.622) | 152.833 (2.730) | 138.076 (11.356) |
| reduce_max_i64_n1024 | 22.207 (0.559) | 210.343 (16.687) | 14.219 (0.587) | 113.656 (21.363) | 160.656 (8.882) |
| reduce_min_i64_n1024 | 23.030 (0.718) | 223.526 (9.289) | 14.023 (0.286) | 159.671 (5.504) | 110.758 (19.564) |
| add_i32_n1048576 | 44.924 (0.672) | 47.039 (0.994) | 39.775 (0.061) | 159.455 (26.457) | 98.929 (6.140) |
| sub_i32_n1048576 | 44.626 (0.293) | 44.877 (0.499) | 39.747 (0.052) | 131.056 (29.723) | 114.835 (12.795) |
| mul_i32_n1048576 | 44.392 (0.154) | 44.399 (0.158) | 39.770 (0.032) | 143.393 (31.300) | 116.899 (16.883) |
| div_i32_n1048576 | 98.310 (5.627) | 181.813 (3.196) | 39.997 (0.090) | 185.277 (46.842) | 124.312 (19.796) |
| rem_i32_n1048576 | 87.378 (0.916) | 177.956 (0.719) | 40.049 (0.103) | 134.161 (14.414) | 131.979 (21.880) |
| maximum_i32_n1048576 | 43.687 (0.102) | 101.891 (0.566) | 39.746 (0.042) | 134.470 (20.871) | 125.233 (16.626) |
| minimum_i32_n1048576 | 43.638 (0.115) | 105.645 (1.022) | 39.799 (0.084) | 135.215 (19.408) | 123.394 (17.596) |
| neg_i32_n1048576 | 30.419 (0.049) | 29.695 (0.085) | 27.335 (0.010) | 131.000 (22.196) | 116.969 (15.170) |
| abs_i32_n1048576 | 30.439 (0.101) | 57.221 (0.277) | 27.357 (0.032) | 121.572 (20.713) | 97.250 (9.779) |
| sign_i32_n1048576 | 30.336 (0.043) | 57.187 (0.269) | 27.372 (0.024) | 125.983 (21.535) | 111.141 (12.109) |
| reduce_sum_i32_n1048576 | 41.929 (0.592) | 71.935 (0.854) | 15.958 (0.307) | 95.025 (1.989) | 91.740 (0.791) |
| reduce_max_i32_n1048576 | 43.512 (0.673) | 73.924 (0.695) | 16.284 (0.018) | 89.569 (1.596) | 92.074 (1.676) |
| reduce_min_i32_n1048576 | 45.159 (0.618) | 77.589 (0.981) | 16.297 (0.119) | 89.817 (1.093) | 90.946 (2.493) |
| add_i64_n1048576 | 83.403 (0.556) | 85.836 (0.339) | 77.741 (0.064) | 128.728 (3.812) | 91.538 (2.919) |
| sub_i64_n1048576 | 83.040 (0.180) | 85.579 (0.294) | 77.915 (0.186) | 140.525 (2.537) | 83.008 (0.659) |
| mul_i64_n1048576 | 83.209 (0.153) | 85.634 (0.113) | 77.741 (0.072) | 150.070 (5.531) | 83.209 (0.259) |
| div_i64_n1048576 | 141.913 (1.280) | 285.776 (3.773) | 80.402 (0.057) | 211.949 (13.450) | 90.666 (2.997) |
| rem_i64_n1048576 | 136.822 (1.147) | 282.002 (2.096) | 80.109 (0.081) | 164.120 (11.379) | 115.727 (2.687) |
| maximum_i64_n1048576 | 82.379 (0.246) | 193.681 (1.333) | 77.601 (0.087) | 159.825 (15.735) | 102.247 (8.537) |
| minimum_i64_n1048576 | 82.380 (0.135) | 191.876 (0.661) | 77.612 (0.031) | 162.762 (7.697) | 99.432 (7.537) |
| neg_i64_n1048576 | 58.315 (0.717) | 57.051 (0.501) | 52.958 (0.062) | 121.900 (21.187) | 109.060 (10.027) |
| abs_i64_n1048576 | 58.177 (0.387) | 109.850 (0.463) | 53.113 (0.063) | 113.099 (15.847) | 113.720 (11.877) |
| sign_i64_n1048576 | 56.232 (0.089) | 109.668 (0.449) | 52.987 (0.067) | 129.198 (22.698) | 118.999 (14.347) |
| reduce_sum_i64_n1048576 | 47.673 (0.651) | 102.424 (1.177) | 28.790 (0.038) | 159.386 (2.208) | 152.131 (1.621) |
| reduce_max_i64_n1048576 | 49.425 (0.537) | 104.430 (1.074) | 28.850 (0.052) | 154.341 (1.852) | 150.173 (2.341) |
| reduce_min_i64_n1048576 | 51.345 (0.438) | 102.890 (0.760) | 28.848 (0.025) | 158.432 (2.931) | 147.887 (2.240) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 1.681 | 0 |
| add_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.049 | 0 |
| sub_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.198 | 0 |
| sub_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.200 | 0 |
| mul_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.291 | 0 |
| mul_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.340 | 0 |
| div_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.354 | 0 |
| div_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.419 | 0 |
| rem_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.251 | 0 |
| rem_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.279 | 0 |
| pow_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.223 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.358 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.248 | 0 |
| maximum_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.271 | 0 |
| minimum_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.311 | 0 |
| minimum_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.272 | 0 |
| neg_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.470 | 0 |
| neg_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.262 | 0 |
| abs_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.913 | 0 |
| abs_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.772 | 0 |
| sign_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.594 | 0 |
| sign_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.525 | 0 |
| conj_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.345 | 0 |
| conj_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.335 | 0 |
| exp_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.962 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.860 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.516 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.508 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.388 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.426 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.269 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.399 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.268 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.323 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.177 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.332 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.239 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.302 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.221 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.414 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.222 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.323 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.278 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.242 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| where_select_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.659 | 0 |
| where_select_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.524 | 0 |
| clamp_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.375 | 0 |
| clamp_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.370 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.402 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.475 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.743 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.793 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.790 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.801 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.120 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.029 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.770 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.886 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.542 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.674 | 0 |
| sub_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.813 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.652 | 0 |
| mul_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.798 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.648 | 0 |
| div_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.805 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.072 | 0 |
| rem_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.722 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.603 | 0 |
| pow_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.728 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.027 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.739 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.610 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.737 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.621 | 0 |
| neg_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.893 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.795 | 0 |
| abs_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.894 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.789 | 0 |
| sign_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.905 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.802 | 0 |
| conj_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.778 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.745 | 0 |
| exp_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.896 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.951 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.903 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.656 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.892 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.830 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.917 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.925 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.906 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.129 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.907 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.229 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.899 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.217 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.892 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.742 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.900 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.016 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.903 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.014 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.057 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.663 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.982 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.728 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.800 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.610 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.578 | 0.001812 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.071 | 2.148e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.808 | 0.005478 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.284 | 3.659e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.120 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.421 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.080 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.499 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.789 | 0.0003528 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.039 | 3.293e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_full_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 0.974 | 0.0001333 |
| reduce_prod_full_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 1.897 | 0.003334 |
| reduce_max_full_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.255 | 0 |
| reduce_min_full_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.241 | 0 |
| compare_eq_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_lt_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_le_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_ge_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| reduce_sum_full_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.587 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.667 | 0 |
| reduce_max_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.424 | 0 |
| reduce_min_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.019 | 0 |
| compare_eq_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_lt_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_le_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_ge_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| reduce_sum_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.547 | 0.07061 |
| reduce_prod_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.536 | 0.1957 |
| reduce_max_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 9.845 | 0 |
| reduce_min_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 9.847 | 0 |
| compare_eq_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.134 | 0 |
| compare_lt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.131 | 0 |
| compare_le_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.134 | 0 |
| compare_ge_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.076 | 0 |
| reduce_sum_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.046 | 0.001745 |
| reduce_prod_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.045 | 6.451e-05 |
| reduce_max_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 12.522 | 0 |
| reduce_min_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 12.523 | 0 |
| compare_eq_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.659 | 0 |
| compare_lt_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.660 | 0 |
| compare_le_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.661 | 0 |
| compare_ge_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.655 | 0 |
| reduce_mean_full_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.769 | 0.0001766 |
| reduce_mean_full_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.353 | 0 |
| reduce_mean_full_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.558 | 0.001429 |
| reduce_mean_full_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.326 | 9.982e-07 |
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
| add_i32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.246 | 0 |
| sub_i32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.487 | 0 |
| mul_i32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.728 | 0 |
| div_i32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.146 | 0 |
| rem_i32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.436 | 0 |
| maximum_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.339 | 0 |
| minimum_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.345 | 0 |
| neg_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.104 | 0 |
| abs_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.151 | 0 |
| sign_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.121 | 0 |
| reduce_sum_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.009 | 0 |
| reduce_max_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.090 | 0 |
| reduce_min_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.169 | 0 |
| add_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.654 | 0 |
| sub_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.685 | 0 |
| mul_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.690 | 0 |
| div_i64_n1024 | tenferro-cuda-direct | 32 | 15 | 3.259 | 0 |
| rem_i64_n1024 | tenferro-cuda-direct | 32 | 15 | 3.055 | 0 |
| maximum_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.419 | 0 |
| minimum_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.426 | 0 |
| neg_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.175 | 0 |
| abs_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.120 | 0 |
| sign_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.082 | 0 |
| reduce_sum_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.922 | 0 |
| reduce_max_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.843 | 0 |
| reduce_min_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.948 | 0 |
| add_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.875 | 0 |
| sub_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.856 | 0 |
| mul_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.841 | 0 |
| div_i32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.146 | 0 |
| rem_i32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.796 | 0 |
| maximum_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.796 | 0 |
| minimum_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.793 | 0 |
| neg_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.947 | 0 |
| abs_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.948 | 0 |
| sign_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.941 | 0 |
| reduce_sum_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.683 | 0 |
| reduce_max_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.785 | 0 |
| reduce_min_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.890 | 0 |
| add_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.669 | 0 |
| sub_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.657 | 0 |
| mul_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.663 | 0 |
| div_i64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.271 | 0 |
| rem_i64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.189 | 0 |
| maximum_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.636 | 0 |
| minimum_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.636 | 0 |
| neg_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.866 | 0 |
| abs_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.862 | 0 |
| sign_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.799 | 0 |
| reduce_sum_i64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.051 | 0 |
| reduce_max_i64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.163 | 0 |
| reduce_min_i64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.286 | 0 |
| add_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.861 | 0 |
| add_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 1.928 | 0 |
| sub_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.527 | 0 |
| sub_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.297 | 0 |
| mul_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.283 | 0 |
| mul_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.609 | 0 |
| div_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.481 | 0 |
| div_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.380 | 0 |
| rem_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.654 | 0 |
| rem_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.730 | 0 |
| pow_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.347 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.974 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.398 | 0 |
| maximum_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.626 | 0 |
| minimum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.201 | 0 |
| minimum_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.319 | 0 |
| neg_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.611 | 0 |
| neg_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.575 | 0 |
| abs_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.338 | 0 |
| abs_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.149 | 0 |
| sign_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.197 | 0 |
| sign_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.286 | 0 |
| conj_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.261 | 0 |
| conj_c64_n1024 | tenferro-cuda-eager | 32 | 15 | 3.971 | 0 |
| exp_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.508 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.408 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.354 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.345 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.445 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.567 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.764 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.823 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.765 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.810 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.835 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.840 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.760 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.704 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.862 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.908 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.039 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.895 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.864 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.867 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.430 | 0 |
| compare_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.510 | 0 |
| where_select_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.689 | 0 |
| where_select_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.795 | 0 |
| clamp_f32_n1024 | tenferro-cuda-eager | 2 | 15 | 2.283 | 0 |
| clamp_f64_n1024 | tenferro-cuda-eager | 2 | 15 | 2.245 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 1.736 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 1.698 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.683 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.582 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.574 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.399 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.146 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.141 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.437 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.454 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.569 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.519 | 0 |
| sub_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.838 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.745 | 0 |
| mul_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.841 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.751 | 0 |
| div_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.858 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.274 | 0 |
| rem_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.131 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.987 | 0 |
| pow_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.121 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-eager | 2 | 15 | 2.239 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.128 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.988 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.128 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.988 | 0 |
| neg_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.887 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.832 | 0 |
| abs_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.595 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.429 | 0 |
| sign_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.602 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.428 | 0 |
| conj_c32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.439 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.350 | 0 |
| exp_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.898 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.993 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.905 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.690 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.908 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.844 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.908 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.941 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.906 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.138 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.903 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.267 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.916 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.251 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.914 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.793 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.908 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.055 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.906 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.051 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.809 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.638 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.765 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.305 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.415 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.233 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.077 | 0.001812 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.056 | 2.148e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.478 | 0.005478 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.054 | 3.659e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.458 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.609 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.441 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.594 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.663 | 0.0003528 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.108 | 3.293e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_full_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.242 | 0.0001333 |
| reduce_prod_full_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.826 | 0.003334 |
| reduce_max_full_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.851 | 0 |
| reduce_min_full_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.543 | 0 |
| compare_eq_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.372 | 0 |
| compare_lt_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.360 | 0 |
| compare_le_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.251 | 0 |
| compare_ge_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.184 | 0 |
| reduce_sum_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.980 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.957 | 0 |
| reduce_max_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.973 | 0 |
| reduce_min_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.003 | 0 |
| compare_eq_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.346 | 0 |
| compare_lt_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.335 | 0 |
| compare_le_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.348 | 0 |
| compare_ge_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.269 | 0 |
| reduce_sum_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.650 | 0.07061 |
| reduce_prod_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.612 | 0.1957 |
| reduce_max_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 9.861 | 0 |
| reduce_min_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 9.861 | 0 |
| compare_eq_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.821 | 0 |
| compare_lt_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.836 | 0 |
| compare_le_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.824 | 0 |
| compare_ge_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.814 | 0 |
| reduce_sum_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.095 | 0.001745 |
| reduce_prod_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.094 | 6.451e-05 |
| reduce_max_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 12.528 | 0 |
| reduce_min_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 12.530 | 0 |
| compare_eq_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.633 | 0 |
| compare_lt_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.633 | 0 |
| compare_le_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.634 | 0 |
| compare_ge_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.635 | 0 |
| reduce_mean_full_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.740 | 0.0001766 |
| reduce_mean_full_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 4.088 | 0 |
| reduce_mean_full_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.682 | 0.001429 |
| reduce_mean_full_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.377 | 9.982e-07 |
| scale_real_f32_n1024 | tenferro-cuda-eager | — | — | — | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_f64_n1024 | tenferro-cuda-eager | — | — | — | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_c32_n1024 | tenferro-cuda-eager | — | — | — | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_c64_n1024 | tenferro-cuda-eager | — | — | — | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_complex_c32_n1024 | tenferro-cuda-eager | — | — | — | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_complex_c64_n1024 | tenferro-cuda-eager | — | — | — | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_f32_n1048576 | tenferro-cuda-eager | — | — | — | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_f64_n1048576 | tenferro-cuda-eager | — | — | — | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_c32_n1048576 | tenferro-cuda-eager | — | — | — | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_c64_n1048576 | tenferro-cuda-eager | — | — | — | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_complex_c32_n1048576 | tenferro-cuda-eager | — | — | — | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | — | — | — | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| add_i32_n1024 | tenferro-cuda-eager | 256 | 15 | 3.201 | 0 |
| sub_i32_n1024 | tenferro-cuda-eager | 256 | 15 | 3.114 | 0 |
| mul_i32_n1024 | tenferro-cuda-eager | 256 | 15 | 3.098 | 0 |
| div_i32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.845 | 0 |
| rem_i32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.314 | 0 |
| maximum_i32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.358 | 0 |
| minimum_i32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.376 | 0 |
| neg_i32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.568 | 0 |
| abs_i32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.653 | 0 |
| sign_i32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.568 | 0 |
| reduce_sum_i32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.366 | 0 |
| reduce_max_i32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.291 | 0 |
| reduce_min_i32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.282 | 0 |
| add_i64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.558 | 0 |
| sub_i64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.292 | 0 |
| mul_i64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.377 | 0 |
| div_i64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.459 | 0 |
| rem_i64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.192 | 0 |
| maximum_i64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.191 | 0 |
| minimum_i64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.194 | 0 |
| neg_i64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.551 | 0 |
| abs_i64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.456 | 0 |
| sign_i64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.387 | 0 |
| reduce_sum_i64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.391 | 0 |
| reduce_max_i64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.365 | 0 |
| reduce_min_i64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.576 | 0 |
| add_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.010 | 0 |
| sub_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.872 | 0 |
| mul_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.842 | 0 |
| div_i32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.909 | 0 |
| rem_i32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.847 | 0 |
| maximum_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.261 | 0 |
| minimum_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.381 | 0 |
| neg_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.900 | 0 |
| abs_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.662 | 0 |
| sign_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.660 | 0 |
| reduce_sum_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.302 | 0 |
| reduce_max_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.366 | 0 |
| reduce_min_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.483 | 0 |
| add_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.747 | 0 |
| sub_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.739 | 0 |
| mul_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.740 | 0 |
| div_i64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.286 | 0 |
| rem_i64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.256 | 0 |
| maximum_i64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.099 | 0 |
| minimum_i64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.070 | 0 |
| neg_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.826 | 0 |
| abs_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.515 | 0 |
| sign_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.509 | 0 |
| reduce_sum_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.278 | 0 |
| reduce_max_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.342 | 0 |
| reduce_min_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.292 | 0 |
| add_f32_n1024 | pytorch-cuda | 256 | 15 | 2.364 | 0 |
| add_f64_n1024 | pytorch-cuda | 256 | 15 | 2.357 | 0 |
| sub_f32_n1024 | pytorch-cuda | 256 | 15 | 2.393 | 0 |
| sub_f64_n1024 | pytorch-cuda | 256 | 15 | 2.433 | 0 |
| mul_f32_n1024 | pytorch-cuda | 256 | 15 | 2.312 | 0 |
| mul_f64_n1024 | pytorch-cuda | 256 | 15 | 2.360 | 0 |
| div_f32_n1024 | pytorch-cuda | 256 | 15 | 2.313 | 0 |
| div_f64_n1024 | pytorch-cuda | 256 | 15 | 2.230 | 0 |
| rem_f32_n1024 | pytorch-cuda | 256 | 15 | 2.318 | 0 |
| rem_f64_n1024 | pytorch-cuda | 256 | 15 | 2.304 | 0 |
| pow_f32_n1024 | pytorch-cuda | 256 | 15 | 2.433 | 0.000117 |
| pow_f64_n1024 | pytorch-cuda | 128 | 15 | 3.962 | 1.48e-07 |
| maximum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.365 | 0 |
| maximum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.412 | 0 |
| minimum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.479 | 0 |
| minimum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.393 | 0 |
| neg_f32_n1024 | pytorch-cuda | 256 | 15 | 2.206 | 0 |
| neg_f64_n1024 | pytorch-cuda | 256 | 15 | 2.143 | 0 |
| abs_f32_n1024 | pytorch-cuda | 256 | 15 | 2.898 | 0 |
| abs_f64_n1024 | pytorch-cuda | 256 | 15 | 2.739 | 0 |
| sign_f32_n1024 | pytorch-cuda | 256 | 15 | 2.528 | 0 |
| sign_f64_n1024 | pytorch-cuda | 256 | 15 | 2.258 | 0 |
| conj_c32_n1024 | pytorch-cuda | 256 | 15 | 2.541 | 0 |
| conj_c64_n1024 | pytorch-cuda | 256 | 15 | 2.651 | 0 |
| exp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.175 | 0.0001272 |
| exp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.173 | 1.095e-07 |
| expm1_f32_n1024 | pytorch-cuda | 256 | 15 | 2.260 | 7.911e-05 |
| expm1_f64_n1024 | pytorch-cuda | 256 | 15 | 2.213 | 7.39e-08 |
| log_f32_n1024 | pytorch-cuda | 256 | 15 | 2.190 | 7.945e-05 |
| log_f64_n1024 | pytorch-cuda | 256 | 15 | 2.790 | 4.231e-08 |
| log1p_f32_n1024 | pytorch-cuda | 256 | 15 | 2.253 | 0.0001189 |
| log1p_f64_n1024 | pytorch-cuda | 256 | 15 | 2.882 | 1.108e-07 |
| erf_f32_n1024 | pytorch-cuda | 256 | 15 | 2.224 | 7.914e-05 |
| erf_f64_n1024 | pytorch-cuda | 128 | 15 | 2.027 | 8.536e-08 |
| sin_f32_n1024 | pytorch-cuda | 256 | 15 | 2.123 | 4.348e-05 |
| sin_f64_n1024 | pytorch-cuda | 256 | 15 | 2.162 | 4.153e-08 |
| cos_f32_n1024 | pytorch-cuda | 256 | 15 | 2.237 | 6.342e-05 |
| cos_f64_n1024 | pytorch-cuda | 256 | 15 | 2.246 | 5.905e-08 |
| tanh_f32_n1024 | pytorch-cuda | 256 | 15 | 2.455 | 9.107e-05 |
| tanh_f64_n1024 | pytorch-cuda | 256 | 15 | 2.513 | 4.434e-08 |
| sqrt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.216 | 0 |
| sqrt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.202 | 0 |
| rsqrt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.184 | 6.983e-05 |
| rsqrt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.141 | 6.504e-08 |
| compare_f32_n1024 | pytorch-cuda | 256 | 15 | 2.187 | 0 |
| compare_f64_n1024 | pytorch-cuda | 256 | 15 | 2.149 | 0 |
| where_select_f32_n1024 | pytorch-cuda | 256 | 15 | 3.049 | 0 |
| where_select_f64_n1024 | pytorch-cuda | 256 | 15 | 3.214 | 0 |
| clamp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.582 | 0 |
| clamp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.623 | 0 |
| reduce_sum_f32_n1024 | pytorch-cuda | 256 | 15 | 3.551 | 0.0002742 |
| reduce_sum_f64_n1024 | pytorch-cuda | 256 | 15 | 3.522 | 1.06e-07 |
| reduce_prod_f32_n1024 | pytorch-cuda | 256 | 15 | 3.116 | 0.0003579 |
| reduce_prod_f64_n1024 | pytorch-cuda | 256 | 15 | 3.224 | 4.441e-07 |
| reduce_max_f32_n1024 | pytorch-cuda | 256 | 15 | 3.443 | 0 |
| reduce_max_f64_n1024 | pytorch-cuda | 256 | 15 | 3.450 | 0 |
| reduce_min_f32_n1024 | pytorch-cuda | 256 | 15 | 3.468 | 0 |
| reduce_min_f64_n1024 | pytorch-cuda | 256 | 15 | 3.632 | 0 |
| reduce_sum_squares_f32_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | pytorch-cuda | 256 | 15 | 3.450 | 1.439e-05 |
| reduce_mean_f64_n1024 | pytorch-cuda | 256 | 15 | 3.618 | 6.71e-09 |
| add_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.489 | 0 |
| sub_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| mul_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.545 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| div_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.559 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.021 | 0 |
| rem_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.566 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.502 | 0 |
| pow_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.662 | 0.0001577 |
| pow_f64_n1048576 | pytorch-cuda | 2 | 15 | 2.034 | 1.48e-07 |
| maximum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.554 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.511 | 0 |
| minimum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.554 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.509 | 0 |
| neg_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.750 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.697 | 0 |
| abs_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.701 | 0 |
| sign_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.751 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.703 | 0 |
| conj_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.696 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.665 | 0 |
| exp_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.753 | 0.0001272 |
| exp_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.889 | 1.095e-07 |
| expm1_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.759 | 7.915e-05 |
| expm1_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.598 | 7.39e-08 |
| log_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.759 | 7.946e-05 |
| log_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.803 | 4.308e-08 |
| log1p_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.761 | 0.000119 |
| log1p_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.900 | 1.11e-07 |
| erf_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.759 | 7.923e-05 |
| erf_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.119 | 8.536e-08 |
| sin_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.764 | 4.348e-05 |
| sin_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.388 | 4.153e-08 |
| cos_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.766 | 6.342e-05 |
| cos_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.381 | 5.905e-08 |
| tanh_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 9.135e-05 |
| tanh_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.677 | 4.437e-08 |
| sqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.766 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.942 | 0 |
| rsqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.753 | 6.983e-05 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.971 | 6.504e-08 |
| compare_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.955 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.581 | 0 |
| where_select_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.753 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.591 | 0 |
| clamp_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.814 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.992 | 0 |
| reduce_sum_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.083 | 0.002948 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.075 | 2.961e-06 |
| reduce_prod_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.068 | 0.002623 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.062 | 4.216e-06 |
| reduce_max_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.120 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.670 | 0 |
| reduce_min_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.128 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.669 | 0 |
| reduce_sum_squares_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.084 | 2.947e-06 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.103 | 3.9e-09 |
| reduce_sum_full_f32_n1024 | pytorch-cuda | 128 | 15 | 1.606 | 0.0002666 |
| reduce_prod_full_f32_n1024 | pytorch-cuda | 256 | 15 | 2.693 | 0.0025 |
| reduce_max_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.126 | 0 |
| reduce_min_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.136 | 0 |
| compare_eq_f32_n1024 | pytorch-cuda | 256 | 15 | 2.228 | 0 |
| compare_lt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.213 | 0 |
| compare_le_f32_n1024 | pytorch-cuda | 256 | 15 | 2.231 | 0 |
| compare_ge_f32_n1024 | pytorch-cuda | 256 | 15 | 2.245 | 0 |
| reduce_sum_full_f64_n1024 | pytorch-cuda | 256 | 15 | 3.208 | 6.208e-07 |
| reduce_prod_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.754 | 1.663e-06 |
| reduce_max_full_f64_n1024 | pytorch-cuda | 256 | 15 | 3.166 | 0 |
| reduce_min_full_f64_n1024 | pytorch-cuda | 256 | 15 | 3.293 | 0 |
| compare_eq_f64_n1024 | pytorch-cuda | 256 | 15 | 2.354 | 0 |
| compare_lt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.184 | 0 |
| compare_le_f64_n1024 | pytorch-cuda | 256 | 15 | 2.212 | 0 |
| compare_ge_f64_n1024 | pytorch-cuda | 256 | 15 | 2.258 | 0 |
| reduce_sum_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.365 | 0.00522 |
| reduce_prod_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.360 | 0.5197 |
| reduce_max_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.400 | 0 |
| reduce_min_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.416 | 0 |
| compare_eq_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.877 | 0 |
| compare_lt_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.876 | 0 |
| compare_le_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.876 | 0 |
| compare_ge_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.877 | 0 |
| reduce_sum_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.399 | 0.0004425 |
| reduce_prod_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.399 | 0.0001269 |
| reduce_max_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.856 | 0 |
| reduce_min_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.855 | 0 |
| compare_eq_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.640 | 0 |
| compare_lt_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.587 | 0 |
| compare_le_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.586 | 0 |
| compare_ge_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.582 | 0 |
| reduce_mean_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.092 | 0.0002906 |
| reduce_sum_squares_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.785 | 0 |
| reduce_mean_full_f64_n1024 | pytorch-cuda | 256 | 15 | 3.300 | 8.614e-07 |
| reduce_sum_squares_full_f64_n1024 | pytorch-cuda | 256 | 15 | 3.667 | 0 |
| reduce_mean_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.362 | 0.00522 |
| reduce_sum_squares_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.264 | 0 |
| reduce_mean_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.404 | 0.001223 |
| reduce_sum_squares_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.131 | 1.664e-07 |
| scale_real_f32_n1024 | pytorch-cuda | 256 | 15 | 2.631 | 0 |
| scale_real_f64_n1024 | pytorch-cuda | 256 | 15 | 2.608 | 0 |
| scale_real_c32_n1024 | pytorch-cuda | 256 | 15 | 2.665 | 0 |
| scale_real_c64_n1024 | pytorch-cuda | 256 | 15 | 2.683 | 0 |
| scale_complex_c32_n1024 | pytorch-cuda | 256 | 15 | 2.620 | 4.666e-05 |
| scale_complex_c64_n1024 | pytorch-cuda | 256 | 15 | 2.612 | 0 |
| scale_real_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.803 | 0 |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.720 | 0 |
| scale_real_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.718 | 0 |
| scale_real_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.677 | 0 |
| scale_complex_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.718 | 4.516e-05 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.679 | 0 |
| add_i32_n1024 | pytorch-cuda | 256 | 15 | 2.399 | 0 |
| sub_i32_n1024 | pytorch-cuda | 256 | 15 | 2.511 | 0 |
| mul_i32_n1024 | pytorch-cuda | 256 | 15 | 2.379 | 0 |
| div_i32_n1024 | pytorch-cuda | 256 | 15 | 2.662 | 0 |
| rem_i32_n1024 | pytorch-cuda | 256 | 15 | 2.419 | 0 |
| maximum_i32_n1024 | pytorch-cuda | 256 | 15 | 2.449 | 0 |
| minimum_i32_n1024 | pytorch-cuda | 256 | 15 | 2.449 | 0 |
| neg_i32_n1024 | pytorch-cuda | 256 | 15 | 2.221 | 0 |
| abs_i32_n1024 | pytorch-cuda | 256 | 15 | 2.840 | 0 |
| sign_i32_n1024 | pytorch-cuda | 256 | 15 | 2.235 | 0 |
| reduce_sum_i32_n1024 | pytorch-cuda | 128 | 15 | 1.967 | 0 |
| reduce_max_i32_n1024 | pytorch-cuda | 256 | 15 | 3.502 | 0 |
| reduce_min_i32_n1024 | pytorch-cuda | 256 | 15 | 3.790 | 0 |
| add_i64_n1024 | pytorch-cuda | 256 | 15 | 2.567 | 0 |
| sub_i64_n1024 | pytorch-cuda | 256 | 15 | 2.475 | 0 |
| mul_i64_n1024 | pytorch-cuda | 256 | 15 | 2.386 | 0 |
| div_i64_n1024 | pytorch-cuda | 256 | 15 | 2.710 | 0 |
| rem_i64_n1024 | pytorch-cuda | 256 | 15 | 2.348 | 0 |
| maximum_i64_n1024 | pytorch-cuda | 256 | 15 | 2.384 | 0 |
| minimum_i64_n1024 | pytorch-cuda | 256 | 15 | 2.353 | 0 |
| neg_i64_n1024 | pytorch-cuda | 256 | 15 | 2.267 | 0 |
| abs_i64_n1024 | pytorch-cuda | 256 | 15 | 2.890 | 0 |
| sign_i64_n1024 | pytorch-cuda | 256 | 15 | 2.347 | 0 |
| reduce_sum_i64_n1024 | pytorch-cuda | 128 | 15 | 2.023 | 0 |
| reduce_max_i64_n1024 | pytorch-cuda | 256 | 15 | 3.640 | 0 |
| reduce_min_i64_n1024 | pytorch-cuda | 256 | 15 | 3.590 | 0 |
| add_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.546 | 0 |
| sub_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| mul_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.545 | 0 |
| div_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.560 | 0 |
| rem_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.563 | 0 |
| maximum_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| minimum_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.547 | 0 |
| neg_i32_n1048576 | pytorch-cuda | 64 | 15 | 1.749 | 0 |
| abs_i32_n1048576 | pytorch-cuda | 64 | 15 | 1.751 | 0 |
| sign_i32_n1048576 | pytorch-cuda | 64 | 15 | 1.752 | 0 |
| reduce_sum_i32_n1048576 | pytorch-cuda | 128 | 15 | 2.043 | 0 |
| reduce_max_i32_n1048576 | pytorch-cuda | 128 | 15 | 2.084 | 0 |
| reduce_min_i32_n1048576 | pytorch-cuda | 128 | 15 | 2.086 | 0 |
| add_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| sub_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.493 | 0 |
| mul_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| div_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.573 | 0 |
| rem_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.563 | 0 |
| maximum_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.483 | 0 |
| minimum_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| neg_i64_n1048576 | pytorch-cuda | 32 | 15 | 1.695 | 0 |
| abs_i64_n1048576 | pytorch-cuda | 32 | 15 | 1.700 | 0 |
| sign_i64_n1048576 | pytorch-cuda | 32 | 15 | 1.696 | 0 |
| reduce_sum_i64_n1048576 | pytorch-cuda | 128 | 15 | 3.685 | 0 |
| reduce_max_i64_n1048576 | pytorch-cuda | 128 | 15 | 3.693 | 0 |
| reduce_min_i64_n1048576 | pytorch-cuda | 128 | 15 | 3.693 | 0 |
| add_f32_n1024 | jax-cuda | 16 | 15 | 1.757 | 0 |
| add_f64_n1024 | jax-cuda | 16 | 15 | 2.080 | 0 |
| sub_f32_n1024 | jax-cuda | 16 | 15 | 2.041 | 0 |
| sub_f64_n1024 | jax-cuda | 16 | 15 | 2.108 | 0 |
| mul_f32_n1024 | jax-cuda | 16 | 15 | 2.029 | 0 |
| mul_f64_n1024 | jax-cuda | 16 | 15 | 1.945 | 0 |
| div_f32_n1024 | jax-cuda | 32 | 15 | 3.142 | 4.875e-05 |
| div_f64_n1024 | jax-cuda | 32 | 15 | 3.102 | 0 |
| rem_f32_n1024 | jax-cuda | 32 | 15 | 3.479 | 0 |
| rem_f64_n1024 | jax-cuda | 16 | 15 | 2.069 | 0 |
| pow_f32_n1024 | jax-cuda | 16 | 15 | 3.053 | 0.000117 |
| pow_f64_n1024 | jax-cuda | 16 | 15 | 3.091 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda | 16 | 15 | 2.617 | 0 |
| maximum_f64_n1024 | jax-cuda | 16 | 15 | 2.224 | 0 |
| minimum_f32_n1024 | jax-cuda | 16 | 15 | 2.185 | 0 |
| minimum_f64_n1024 | jax-cuda | 16 | 15 | 2.123 | 0 |
| neg_f32_n1024 | jax-cuda | 16 | 15 | 2.030 | 0 |
| neg_f64_n1024 | jax-cuda | 16 | 15 | 2.036 | 0 |
| abs_f32_n1024 | jax-cuda | 32 | 15 | 3.060 | 0 |
| abs_f64_n1024 | jax-cuda | 32 | 15 | 2.941 | 0 |
| sign_f32_n1024 | jax-cuda | 32 | 15 | 2.913 | 0 |
| sign_f64_n1024 | jax-cuda | 32 | 15 | 3.095 | 0 |
| conj_c32_n1024 | jax-cuda | 32 | 15 | 2.993 | 0 |
| conj_c64_n1024 | jax-cuda | 32 | 15 | 2.929 | 0 |
| exp_f32_n1024 | jax-cuda | 32 | 15 | 2.996 | 0.0001272 |
| exp_f64_n1024 | jax-cuda | 32 | 15 | 2.965 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda | 32 | 15 | 3.048 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda | 32 | 15 | 3.170 | 7.39e-08 |
| log_f32_n1024 | jax-cuda | 32 | 15 | 2.992 | 7.945e-05 |
| log_f64_n1024 | jax-cuda | 32 | 15 | 3.218 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda | 32 | 15 | 3.217 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda | 32 | 15 | 3.417 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda | 16 | 15 | 2.958 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda | 16 | 15 | 2.938 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda | 16 | 15 | 2.522 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda | 16 | 15 | 2.508 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda | 16 | 15 | 2.619 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda | 16 | 15 | 2.367 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda | 16 | 15 | 2.611 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda | 16 | 15 | 2.191 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda | 16 | 15 | 2.321 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda | 16 | 15 | 2.177 | 0 |
| rsqrt_f32_n1024 | jax-cuda | 16 | 15 | 2.986 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda | 16 | 15 | 2.855 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda | 32 | 15 | 3.272 | 0 |
| compare_f64_n1024 | jax-cuda | 32 | 15 | 2.999 | 0 |
| where_select_f32_n1024 | jax-cuda | 32 | 15 | 3.381 | 0 |
| where_select_f64_n1024 | jax-cuda | 32 | 15 | 3.357 | 0 |
| clamp_f32_n1024 | jax-cuda | 32 | 15 | 3.124 | 0 |
| clamp_f64_n1024 | jax-cuda | 32 | 15 | 3.119 | 0 |
| reduce_sum_f32_n1024 | jax-cuda | 16 | 15 | 2.637 | 0.0002742 |
| reduce_sum_f64_n1024 | jax-cuda | 16 | 15 | 2.063 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda | 32 | 15 | 3.118 | 0.0003574 |
| reduce_prod_f64_n1024 | jax-cuda | 32 | 15 | 3.003 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda | 16 | 15 | 2.559 | 0 |
| reduce_max_f64_n1024 | jax-cuda | 16 | 15 | 2.049 | 0 |
| reduce_min_f32_n1024 | jax-cuda | 16 | 15 | 2.420 | 0 |
| reduce_min_f64_n1024 | jax-cuda | 32 | 15 | 2.888 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda | 32 | 15 | 3.358 | 1.439e-05 |
| reduce_mean_f64_n1024 | jax-cuda | 32 | 15 | 3.183 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda | 16 | 15 | 1.710 | 0 |
| add_f64_n1048576 | jax-cuda | 16 | 15 | 1.632 | 0 |
| sub_f32_n1048576 | jax-cuda | 32 | 15 | 3.281 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 3.188 | 0 |
| mul_f32_n1048576 | jax-cuda | 32 | 15 | 3.322 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 3.120 | 0 |
| div_f32_n1048576 | jax-cuda | 32 | 15 | 2.951 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda | 16 | 15 | 2.084 | 0 |
| rem_f32_n1048576 | jax-cuda | 32 | 15 | 3.012 | 0 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 2.975 | 0 |
| pow_f32_n1048576 | jax-cuda | 16 | 15 | 1.989 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda | 2 | 15 | 2.595 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda | 16 | 15 | 1.999 | 0 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 3.140 | 0 |
| minimum_f32_n1048576 | jax-cuda | 32 | 15 | 3.278 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 3.133 | 0 |
| neg_f32_n1048576 | jax-cuda | 32 | 15 | 3.078 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 3.321 | 0 |
| abs_f32_n1048576 | jax-cuda | 32 | 15 | 2.755 | 0 |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 2.654 | 0 |
| sign_f32_n1048576 | jax-cuda | 32 | 15 | 2.517 | 0 |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 2.073 | 0 |
| conj_c32_n1048576 | jax-cuda | 32 | 15 | 2.040 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 1.863 | 0 |
| exp_f32_n1048576 | jax-cuda | 32 | 15 | 2.174 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda | 16 | 15 | 3.271 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda | 32 | 15 | 2.833 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda | 8 | 15 | 2.147 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda | 32 | 15 | 2.962 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda | 8 | 15 | 3.271 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda | 32 | 15 | 3.051 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda | 8 | 15 | 3.346 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda | 16 | 15 | 2.377 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda | 4 | 15 | 2.703 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda | 16 | 15 | 2.164 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda | 16 | 15 | 3.833 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda | 16 | 15 | 2.020 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda | 8 | 15 | 2.048 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda | 32 | 15 | 2.783 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda | 16 | 15 | 3.049 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda | 32 | 15 | 2.922 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 3.330 | 0 |
| rsqrt_f32_n1048576 | jax-cuda | 16 | 15 | 2.104 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda | 16 | 15 | 2.146 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda | 32 | 15 | 2.817 | 0 |
| compare_f64_n1048576 | jax-cuda | 32 | 15 | 2.763 | 0 |
| where_select_f32_n1048576 | jax-cuda | 32 | 15 | 3.210 | 0 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 3.143 | 0 |
| clamp_f32_n1048576 | jax-cuda | 32 | 15 | 2.954 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 2.858 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda | 16 | 15 | 2.515 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda | 16 | 15 | 2.539 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda | 16 | 15 | 2.454 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda | 16 | 15 | 2.456 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda | 16 | 15 | 2.481 | 0 |
| reduce_max_f64_n1048576 | jax-cuda | 16 | 15 | 2.408 | 0 |
| reduce_min_f32_n1048576 | jax-cuda | 16 | 15 | 2.553 | 0 |
| reduce_min_f64_n1048576 | jax-cuda | 16 | 15 | 2.648 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda | 16 | 15 | 2.632 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda | 16 | 15 | 2.798 | 6.5e-09 |
| reduce_sum_full_f32_n1024 | jax-cuda | 16 | 15 | 2.226 | 0.0002666 |
| reduce_prod_full_f32_n1024 | jax-cuda | 16 | 15 | 2.705 | 0.001072 |
| reduce_max_full_f32_n1024 | jax-cuda | 16 | 15 | 2.501 | 0 |
| reduce_min_full_f32_n1024 | jax-cuda | 16 | 15 | 2.672 | 0 |
| compare_eq_f32_n1024 | jax-cuda | 16 | 15 | 2.385 | 0 |
| compare_lt_f32_n1024 | jax-cuda | 16 | 15 | 2.210 | 0 |
| compare_le_f32_n1024 | jax-cuda | 16 | 15 | 2.068 | 0 |
| compare_ge_f32_n1024 | jax-cuda | 16 | 15 | 1.995 | 0 |
| reduce_sum_full_f64_n1024 | jax-cuda | 32 | 15 | 3.249 | 4.966e-07 |
| reduce_prod_full_f64_n1024 | jax-cuda | 32 | 15 | 3.127 | 1.774e-06 |
| reduce_max_full_f64_n1024 | jax-cuda | 32 | 15 | 3.135 | 0 |
| reduce_min_full_f64_n1024 | jax-cuda | 32 | 15 | 2.735 | 0 |
| compare_eq_f64_n1024 | jax-cuda | 32 | 15 | 2.723 | 0 |
| compare_lt_f64_n1024 | jax-cuda | 32 | 15 | 2.610 | 0 |
| compare_le_f64_n1024 | jax-cuda | 32 | 15 | 2.779 | 0 |
| compare_ge_f64_n1024 | jax-cuda | 32 | 15 | 2.891 | 0 |
| reduce_sum_full_f32_n1048576 | jax-cuda | 32 | 15 | 3.392 | 0.003872 |
| reduce_prod_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.704 | 0.5327 |
| reduce_max_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.630 | 0 |
| reduce_min_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.869 | 0 |
| compare_eq_f32_n1048576 | jax-cuda | 32 | 15 | 3.165 | 0 |
| compare_lt_f32_n1048576 | jax-cuda | 32 | 15 | 2.893 | 0 |
| compare_le_f32_n1048576 | jax-cuda | 32 | 15 | 2.917 | 0 |
| compare_ge_f32_n1048576 | jax-cuda | 32 | 15 | 2.936 | 0 |
| reduce_sum_full_f64_n1048576 | jax-cuda | 32 | 15 | 3.292 | 0.0006452 |
| reduce_prod_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.458 | 0.0001105 |
| reduce_max_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.659 | 0 |
| reduce_min_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.126 | 0 |
| compare_eq_f64_n1048576 | jax-cuda | 32 | 15 | 2.915 | 0 |
| compare_lt_f64_n1048576 | jax-cuda | 32 | 15 | 2.761 | 0 |
| compare_le_f64_n1048576 | jax-cuda | 32 | 15 | 2.922 | 0 |
| compare_ge_f64_n1048576 | jax-cuda | 32 | 15 | 2.972 | 0 |
| reduce_mean_full_f32_n1024 | jax-cuda | 32 | 15 | 3.131 | 0.0002906 |
| reduce_sum_squares_full_f32_n1024 | jax-cuda | 32 | 15 | 2.761 | 0 |
| reduce_mean_full_f64_n1024 | jax-cuda | 32 | 15 | 3.205 | 6.891e-07 |
| reduce_sum_squares_full_f64_n1024 | jax-cuda | 32 | 15 | 2.905 | 0 |
| reduce_mean_full_f32_n1048576 | jax-cuda | 32 | 15 | 3.357 | 0.003872 |
| reduce_sum_squares_full_f32_n1048576 | jax-cuda | 32 | 15 | 3.237 | 0 |
| reduce_mean_full_f64_n1048576 | jax-cuda | 32 | 15 | 3.698 | 0.001783 |
| reduce_sum_squares_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.223 | 0 |
| scale_real_f32_n1024 | jax-cuda | 16 | 15 | 2.431 | 0 |
| scale_real_f64_n1024 | jax-cuda | 16 | 15 | 2.073 | 0 |
| scale_real_c32_n1024 | jax-cuda | 16 | 15 | 2.049 | 0 |
| scale_real_c64_n1024 | jax-cuda | 16 | 15 | 2.037 | 0 |
| scale_complex_c32_n1024 | jax-cuda | 32 | 15 | 3.241 | 7.929e-05 |
| scale_complex_c64_n1024 | jax-cuda | 32 | 15 | 3.222 | 3.753e-08 |
| scale_real_f32_n1048576 | jax-cuda | 32 | 15 | 3.307 | 0 |
| scale_real_f64_n1048576 | jax-cuda | 32 | 15 | 3.385 | 0 |
| scale_real_c32_n1048576 | jax-cuda | 32 | 15 | 3.370 | 0 |
| scale_real_c64_n1048576 | jax-cuda | 16 | 15 | 1.853 | 0 |
| scale_complex_c32_n1048576 | jax-cuda | 32 | 15 | 3.365 | 6.186e-05 |
| scale_complex_c64_n1048576 | jax-cuda | 16 | 15 | 1.867 | 4.129e-08 |
| add_i32_n1024 | jax-cuda | 32 | 15 | 3.039 | 0 |
| sub_i32_n1024 | jax-cuda | 32 | 15 | 3.129 | 0 |
| mul_i32_n1024 | jax-cuda | 32 | 15 | 3.282 | 0 |
| div_i32_n1024 | jax-cuda | 16 | 15 | 2.362 | 0 |
| rem_i32_n1024 | jax-cuda | 32 | 15 | 3.060 | 0 |
| maximum_i32_n1024 | jax-cuda | 32 | 15 | 3.107 | 0 |
| minimum_i32_n1024 | jax-cuda | 32 | 15 | 3.159 | 0 |
| neg_i32_n1024 | jax-cuda | 32 | 15 | 3.187 | 0 |
| abs_i32_n1024 | jax-cuda | 32 | 15 | 2.757 | 0 |
| sign_i32_n1024 | jax-cuda | 32 | 15 | 2.858 | 0 |
| reduce_sum_i32_n1024 | jax-cuda | 32 | 15 | 3.095 | 0 |
| reduce_max_i32_n1024 | jax-cuda | 32 | 15 | 2.899 | 0 |
| reduce_min_i32_n1024 | jax-cuda | 32 | 15 | 2.960 | 0 |
| add_i64_n1024 | jax-cuda | 32 | 15 | 3.295 | 0 |
| sub_i64_n1024 | jax-cuda | 32 | 15 | 3.326 | 0 |
| mul_i64_n1024 | jax-cuda | 32 | 15 | 3.113 | 0 |
| div_i64_n1024 | jax-cuda | 16 | 15 | 2.412 | 0 |
| rem_i64_n1024 | jax-cuda | 32 | 15 | 3.210 | 0 |
| maximum_i64_n1024 | jax-cuda | 32 | 15 | 3.309 | 0 |
| minimum_i64_n1024 | jax-cuda | 32 | 15 | 3.314 | 0 |
| neg_i64_n1024 | jax-cuda | 32 | 15 | 3.204 | 0 |
| abs_i64_n1024 | jax-cuda | 32 | 15 | 2.796 | 0 |
| sign_i64_n1024 | jax-cuda | 32 | 15 | 2.798 | 0 |
| reduce_sum_i64_n1024 | jax-cuda | 16 | 15 | 2.445 | 0 |
| reduce_max_i64_n1024 | jax-cuda | 32 | 15 | 3.637 | 0 |
| reduce_min_i64_n1024 | jax-cuda | 16 | 15 | 2.555 | 0 |
| add_i32_n1048576 | jax-cuda | 16 | 15 | 2.551 | 0 |
| sub_i32_n1048576 | jax-cuda | 16 | 15 | 2.097 | 0 |
| mul_i32_n1048576 | jax-cuda | 16 | 15 | 2.294 | 0 |
| div_i32_n1048576 | jax-cuda | 16 | 15 | 2.964 | 0 |
| rem_i32_n1048576 | jax-cuda | 16 | 15 | 2.147 | 0 |
| maximum_i32_n1048576 | jax-cuda | 16 | 15 | 2.152 | 0 |
| minimum_i32_n1048576 | jax-cuda | 16 | 15 | 2.163 | 0 |
| neg_i32_n1048576 | jax-cuda | 16 | 15 | 2.096 | 0 |
| abs_i32_n1048576 | jax-cuda | 16 | 15 | 1.945 | 0 |
| sign_i32_n1048576 | jax-cuda | 16 | 15 | 2.016 | 0 |
| reduce_sum_i32_n1048576 | jax-cuda | 32 | 15 | 3.041 | 0 |
| reduce_max_i32_n1048576 | jax-cuda | 32 | 15 | 2.866 | 0 |
| reduce_min_i32_n1048576 | jax-cuda | 32 | 15 | 2.874 | 0 |
| add_i64_n1048576 | jax-cuda | 16 | 15 | 2.060 | 0 |
| sub_i64_n1048576 | jax-cuda | 16 | 15 | 2.248 | 0 |
| mul_i64_n1048576 | jax-cuda | 16 | 15 | 2.401 | 0 |
| div_i64_n1048576 | jax-cuda | 16 | 15 | 3.391 | 0 |
| rem_i64_n1048576 | jax-cuda | 16 | 15 | 2.626 | 0 |
| maximum_i64_n1048576 | jax-cuda | 16 | 15 | 2.557 | 0 |
| minimum_i64_n1048576 | jax-cuda | 16 | 15 | 2.604 | 0 |
| neg_i64_n1048576 | jax-cuda | 16 | 15 | 1.950 | 0 |
| abs_i64_n1048576 | jax-cuda | 16 | 15 | 1.810 | 0 |
| sign_i64_n1048576 | jax-cuda | 16 | 15 | 2.067 | 0 |
| reduce_sum_i64_n1048576 | jax-cuda | 16 | 15 | 2.550 | 0 |
| reduce_max_i64_n1048576 | jax-cuda | 16 | 15 | 2.469 | 0 |
| reduce_min_i64_n1048576 | jax-cuda | 16 | 15 | 2.535 | 0 |
| add_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.392 | 0 |
| add_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.068 | 0 |
| sub_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.878 | 0 |
| sub_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.669 | 0 |
| mul_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.467 | 0 |
| mul_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.432 | 0 |
| div_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.474 | 4.875e-05 |
| div_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.528 | 0 |
| rem_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.087 | 0 |
| rem_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.419 | 0 |
| pow_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.014 | 0.000117 |
| pow_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.426 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.130 | 0 |
| maximum_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.106 | 0 |
| minimum_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.895 | 0 |
| minimum_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.097 | 0 |
| neg_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.005 | 0 |
| neg_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.942 | 0 |
| abs_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.064 | 0 |
| abs_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.054 | 0 |
| sign_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.055 | 0 |
| sign_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.047 | 0 |
| conj_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.991 | 0 |
| conj_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.799 | 0 |
| exp_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.878 | 0.0001272 |
| exp_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.007 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.119 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.168 | 7.39e-08 |
| log_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.151 | 7.945e-05 |
| log_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.141 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.073 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.293 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.369 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.059 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.265 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.468 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.371 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.625 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.418 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.159 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.104 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.950 | 0 |
| rsqrt_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.110 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.195 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.041 | 0 |
| compare_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.971 | 0 |
| where_select_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.064 | 0 |
| where_select_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.011 | 0 |
| clamp_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.080 | 0 |
| clamp_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.965 | 0 |
| reduce_sum_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.068 | 0.0001139 |
| reduce_sum_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.911 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.354 | 0.0003579 |
| reduce_prod_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.697 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.575 | 0 |
| reduce_max_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.286 | 0 |
| reduce_min_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.220 | 0 |
| reduce_min_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.807 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.978 | 7.204e-06 |
| reduce_mean_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.020 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.994 | 0 |
| add_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.809 | 0 |
| sub_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.476 | 0 |
| sub_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.785 | 0 |
| mul_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.478 | 0 |
| mul_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.671 | 0 |
| div_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.404 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.905 | 0 |
| rem_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.608 | 0 |
| rem_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.664 | 0 |
| pow_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.639 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda-jit | 2 | 15 | 2.542 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.870 | 0 |
| maximum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.698 | 0 |
| minimum_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.341 | 0 |
| minimum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.702 | 0 |
| neg_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.403 | 0 |
| neg_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.202 | 0 |
| abs_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.324 | 0 |
| abs_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.325 | 0 |
| sign_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.367 | 0 |
| sign_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.190 | 0 |
| conj_c32_n1048576 | jax-cuda-jit | 32 | 15 | 2.173 | 0 |
| conj_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.859 | 0 |
| exp_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.395 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.243 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.834 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.095 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.863 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.131 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.876 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.293 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.153 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda-jit | 4 | 15 | 2.637 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.063 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.052 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.037 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.805 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.158 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.041 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.908 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.285 | 0 |
| rsqrt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.998 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.829 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.028 | 0 |
| compare_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.127 | 0 |
| where_select_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.196 | 0 |
| where_select_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.081 | 0 |
| clamp_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.184 | 0 |
| clamp_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.090 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.103 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.455 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.506 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.540 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.526 | 0 |
| reduce_max_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.520 | 0 |
| reduce_min_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.618 | 0 |
| reduce_min_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.511 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.722 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.408 | 6.5e-09 |
| reduce_sum_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.568 | 0 |
| reduce_prod_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.775 | 0.001072 |
| reduce_max_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.699 | 0 |
| reduce_min_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.729 | 0 |
| compare_eq_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.340 | 0 |
| compare_lt_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.127 | 0 |
| compare_le_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.951 | 0 |
| compare_ge_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.220 | 0 |
| reduce_sum_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.036 | 4.966e-07 |
| reduce_prod_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.846 | 1.774e-06 |
| reduce_max_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.742 | 0 |
| reduce_min_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.661 | 0 |
| compare_eq_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.976 | 0 |
| compare_lt_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.150 | 0 |
| compare_le_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.195 | 0 |
| compare_ge_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.185 | 0 |
| reduce_sum_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.668 | 0.003668 |
| reduce_prod_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.584 | 0.5327 |
| reduce_max_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.593 | 0 |
| reduce_min_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.897 | 0 |
| compare_eq_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.906 | 0 |
| compare_lt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.037 | 0 |
| compare_le_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.049 | 0 |
| compare_ge_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.074 | 0 |
| reduce_sum_full_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.516 | 0.0006452 |
| reduce_prod_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.034 | 0.0001105 |
| reduce_max_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.401 | 0 |
| reduce_min_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.108 | 0 |
| compare_eq_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.249 | 0 |
| compare_lt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.159 | 0 |
| compare_le_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.994 | 0 |
| compare_ge_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.919 | 0 |
| reduce_mean_full_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.134 | 0 |
| reduce_sum_squares_full_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.965 | 0 |
| reduce_mean_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.061 | 6.891e-07 |
| reduce_sum_squares_full_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.492 | 0 |
| reduce_mean_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.708 | 0.003668 |
| reduce_sum_squares_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.583 | 0 |
| reduce_mean_full_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.409 | 0.001783 |
| reduce_sum_squares_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.020 | 0 |
| scale_real_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.946 | 0 |
| scale_real_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.033 | 0 |
| scale_real_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.277 | 0 |
| scale_real_c64_n1024 | jax-cuda-jit | 32 | 15 | 3.166 | 0 |
| scale_complex_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.154 | 7.929e-05 |
| scale_complex_c64_n1024 | jax-cuda-jit | 32 | 15 | 3.198 | 3.753e-08 |
| scale_real_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.440 | 0 |
| scale_real_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.251 | 0 |
| scale_real_c32_n1048576 | jax-cuda-jit | 32 | 15 | 2.239 | 0 |
| scale_real_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.862 | 0 |
| scale_complex_c32_n1048576 | jax-cuda-jit | 32 | 15 | 2.264 | 6.186e-05 |
| scale_complex_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.865 | 4.129e-08 |
| add_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.472 | 0 |
| sub_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.470 | 0 |
| mul_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.475 | 0 |
| div_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.516 | 0 |
| rem_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.566 | 0 |
| maximum_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.500 | 0 |
| minimum_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.518 | 0 |
| neg_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.434 | 0 |
| abs_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.418 | 0 |
| sign_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.459 | 0 |
| reduce_sum_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.378 | 0 |
| reduce_max_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.377 | 0 |
| reduce_min_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.483 | 0 |
| add_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.543 | 0 |
| sub_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.584 | 0 |
| mul_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.505 | 0 |
| div_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.657 | 0 |
| rem_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.856 | 0 |
| maximum_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.637 | 0 |
| minimum_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.516 | 0 |
| neg_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.438 | 0 |
| abs_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.555 | 0 |
| sign_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.501 | 0 |
| reduce_sum_i64_n1024 | jax-cuda-jit | 16 | 15 | 2.209 | 0 |
| reduce_max_i64_n1024 | jax-cuda-jit | 16 | 15 | 2.570 | 0 |
| reduce_min_i64_n1024 | jax-cuda-jit | 32 | 15 | 3.544 | 0 |
| add_i32_n1048576 | jax-cuda-jit | 32 | 15 | 3.166 | 0 |
| sub_i32_n1048576 | jax-cuda-jit | 16 | 15 | 1.837 | 0 |
| mul_i32_n1048576 | jax-cuda-jit | 16 | 15 | 1.870 | 0 |
| div_i32_n1048576 | jax-cuda-jit | 16 | 15 | 1.989 | 0 |
| rem_i32_n1048576 | jax-cuda-jit | 16 | 15 | 2.112 | 0 |
| maximum_i32_n1048576 | jax-cuda-jit | 16 | 15 | 2.004 | 0 |
| minimum_i32_n1048576 | jax-cuda-jit | 16 | 15 | 1.974 | 0 |
| neg_i32_n1048576 | jax-cuda-jit | 16 | 15 | 1.871 | 0 |
| abs_i32_n1048576 | jax-cuda-jit | 32 | 15 | 3.112 | 0 |
| sign_i32_n1048576 | jax-cuda-jit | 16 | 15 | 1.778 | 0 |
| reduce_sum_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.936 | 0 |
| reduce_max_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.946 | 0 |
| reduce_min_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.910 | 0 |
| add_i64_n1048576 | jax-cuda-jit | 32 | 15 | 2.929 | 0 |
| sub_i64_n1048576 | jax-cuda-jit | 32 | 15 | 2.656 | 0 |
| mul_i64_n1048576 | jax-cuda-jit | 32 | 15 | 2.663 | 0 |
| div_i64_n1048576 | jax-cuda-jit | 32 | 15 | 2.901 | 0 |
| rem_i64_n1048576 | jax-cuda-jit | 32 | 15 | 3.703 | 0 |
| maximum_i64_n1048576 | jax-cuda-jit | 32 | 15 | 3.272 | 0 |
| minimum_i64_n1048576 | jax-cuda-jit | 32 | 15 | 3.182 | 0 |
| neg_i64_n1048576 | jax-cuda-jit | 16 | 15 | 1.745 | 0 |
| abs_i64_n1048576 | jax-cuda-jit | 16 | 15 | 1.820 | 0 |
| sign_i64_n1048576 | jax-cuda-jit | 16 | 15 | 1.904 | 0 |
| reduce_sum_i64_n1048576 | jax-cuda-jit | 16 | 15 | 2.434 | 0 |
| reduce_max_i64_n1048576 | jax-cuda-jit | 16 | 15 | 2.403 | 0 |
| reduce_min_i64_n1048576 | jax-cuda-jit | 16 | 15 | 2.366 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: benchmarks/gpu/isolated_ops.yaml
timestamp: '2026-10-13T21:21:23Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: e04d33dd2ab9f4e392b8cd63d6df2b43efdb54cb
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 49f248a55b9080907f84efc5a19edeef4c74de31
  dirty: true
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
  expected_cases: 228
  selected_cases: 228
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 19d46353cb2f682dcda76ef3aa8d7733f39d4843aa38ef6231fd496ec575d6a9
    src/bin/benchmark_gpu_public_api/plan.rs: 3d0f10f390ebd0e08a40deb5a64e49e086dcd41b9a777d3a208c6846ed6b2fac
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: da168bf8b7c30f748f62839cf0bf7cce8f5bccf9c37b1c1d993fd70c466f2f76
    src/bin/benchmark_gpu_public_api/multi.rs: 1b477fef78260cf28aa31e515d57ab411f6241b91ffb30d625939538fdbc6bab
    src/bin/benchmark_gpu_public_api/lifetime.rs: c687a8b0329f6d3073073671b5d246acbb1f4dc3f983d1430da897a908ef1745
    src/bin/benchmark_gpu_public_api/read_layout.rs: 70b1ed632ad52db5654c3d9f3df9fdb20a1e8c48894dbdcb24ce055757f5f7eb
    src/bin/benchmark_gpu_public_api/structural.rs: 6b6c5f3c2e771d527dbab1ae4e9dd0354bd1bb88acbe4fb1292ce8b68b408240
    src/bin/benchmark_gpu_public_api/linalg.rs: 5f5c4afc161f3d6e267861511d0d5871be9192d9968c95b59a73c9dc5a64e22b
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: 1dc14ac94fac700aa696d0ada04315685e3245738f9514de3d7989b8cb59152b
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: c5f9507e8971ae14acd540d6c35b305c35b222431ec5aad9830a88bdc32975b0
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: dd5e4cc47e85fcb5afdc56d460136b2a399a6a7d6a62005de8ba94afb0119ecc
    data/results/nvidia-gpu/gpu/isolated_ops/20261013_212123/fixtures/manifest.json: b477c4a11634759024723d4cd13d3a4490980e3e97ff932845296400cca874db
    target/release/benchmark_gpu_public_api: cffb781f4b953aacb5db9e5d0a86a57c6b088cc5161a0e5e384b322e7066a0f5
  reference_numpy_version: 2.4.2
```
