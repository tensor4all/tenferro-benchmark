# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **1078**. Unsupported cells: **62**, including **50** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261013_204849/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

I32/I64 outputs require exact integer equality without conversion to floating point. Division and remainder truncate toward zero; axis-1 sum preserves the input dtype. Arithmetic fixtures avoid overflow, and I64 unary/order fixtures include values above 2**53 to detect loss of integer precision. A zero verification value in these rows means exact equality.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_f32_n1024 | 11.752 (1.208) | 17.610 (1.103) | 9.122 (0.105) | 88.095 (3.228) | 101.526 (2.337) |
| add_f64_n1024 | 12.559 (0.629) | 20.417 (1.833) | 9.277 (0.118) | 95.343 (6.944) | 98.786 (18.456) |
| sub_f32_n1024 | 11.753 (1.500) | 21.644 (2.105) | 9.181 (0.335) | 91.719 (2.313) | 86.496 (11.099) |
| sub_f64_n1024 | 11.222 (3.994) | 22.508 (1.877) | 9.218 (0.141) | 90.725 (11.507) | 79.356 (4.466) |
| mul_f32_n1024 | 10.336 (0.748) | 24.535 (1.729) | 9.126 (0.322) | 90.327 (9.892) | 79.930 (5.480) |
| mul_f64_n1024 | 9.777 (0.926) | 26.856 (1.129) | 9.012 (0.125) | 91.018 (6.519) | 77.607 (6.103) |
| div_f32_n1024 | 9.452 (0.939) | 27.862 (2.879) | 9.226 (0.067) | 84.798 (6.037) | 79.847 (4.981) |
| div_f64_n1024 | 9.455 (1.091) | 27.635 (1.748) | 8.785 (0.051) | 83.343 (2.335) | 81.327 (6.645) |
| rem_f32_n1024 | 8.855 (0.678) | 197.934 (16.794) | 9.225 (0.106) | 97.322 (10.521) | 81.580 (1.275) |
| rem_f64_n1024 | 9.133 (0.567) | 246.162 (21.783) | 9.345 (0.245) | 108.767 (12.944) | 95.890 (12.589) |
| pow_f32_n1024 | 9.004 (0.837) | 214.673 (18.698) | 9.328 (0.152) | 169.108 (21.420) | 101.237 (13.388) |
| pow_f64_n1024 | 9.833 (0.396) | 255.940 (17.256) | 30.929 (0.005) | 200.496 (27.220) | 109.080 (15.675) |
| maximum_f32_n1024 | 8.922 (0.688) | 308.788 (14.691) | 8.929 (0.287) | 152.155 (25.946) | 96.812 (14.663) |
| maximum_f64_n1024 | 8.859 (0.556) | 228.867 (16.814) | 9.204 (0.158) | 147.353 (16.830) | 87.034 (9.599) |
| minimum_f32_n1024 | 9.039 (0.876) | 281.444 (7.943) | 8.888 (0.043) | 143.529 (15.628) | 82.645 (7.991) |
| minimum_f64_n1024 | 9.140 (0.709) | 290.145 (9.688) | 9.072 (0.443) | 141.567 (15.961) | 79.597 (6.597) |
| neg_f32_n1024 | 8.081 (0.445) | 22.258 (0.589) | 8.377 (0.142) | 137.070 (9.834) | 77.717 (4.824) |
| neg_f64_n1024 | 8.185 (0.475) | 22.618 (0.190) | 9.045 (0.534) | 114.871 (0.416) | 77.956 (5.594) |
| abs_f32_n1024 | 8.381 (0.414) | 157.442 (23.864) | 10.689 (0.127) | 106.886 (6.887) | 78.934 (7.734) |
| abs_f64_n1024 | 8.382 (1.046) | 146.365 (0.826) | 10.630 (0.662) | 103.012 (0.780) | 80.169 (10.009) |
| sign_f32_n1024 | 8.630 (0.654) | 168.721 (8.943) | 8.558 (0.087) | 107.468 (7.382) | 82.760 (6.711) |
| sign_f64_n1024 | 8.508 (0.813) | 175.497 (4.501) | 8.753 (0.086) | 103.169 (8.808) | 80.668 (13.116) |
| conj_c32_n1024 | 8.677 (1.107) | 168.380 (14.304) | 9.954 (0.174) | 107.960 (8.709) | 81.094 (8.080) |
| conj_c64_n1024 | 10.767 (0.683) | 132.115 (7.960) | 10.245 (0.158) | 104.376 (0.910) | 78.278 (2.476) |
| exp_f32_n1024 | 11.254 (1.124) | 21.161 (0.303) | 8.244 (0.302) | 94.608 (8.239) | 79.426 (4.475) |
| exp_f64_n1024 | 10.317 (1.076) | 21.310 (1.503) | 8.175 (0.075) | 98.827 (9.213) | 83.129 (7.761) |
| expm1_f32_n1024 | 10.183 (0.840) | 21.090 (0.979) | 8.539 (0.131) | 92.756 (15.532) | 84.695 (9.391) |
| expm1_f64_n1024 | 9.655 (0.904) | 22.126 (1.492) | 8.457 (0.101) | 102.437 (19.479) | 84.082 (7.064) |
| log_f32_n1024 | 9.772 (0.584) | 23.172 (0.926) | 8.603 (0.348) | 102.869 (17.442) | 83.325 (7.043) |
| log_f64_n1024 | 9.718 (1.012) | 24.986 (1.324) | 10.851 (0.007) | 106.962 (16.143) | 91.955 (6.883) |
| log1p_f32_n1024 | 9.128 (0.775) | 22.934 (0.584) | 8.741 (0.089) | 102.548 (16.892) | 90.310 (10.860) |
| log1p_f64_n1024 | 9.173 (0.951) | 23.417 (0.763) | 11.226 (0.004) | 131.841 (9.301) | 100.033 (9.187) |
| erf_f32_n1024 | 9.248 (0.998) | 23.564 (0.500) | 8.515 (0.325) | 217.547 (39.192) | 98.186 (14.439) |
| erf_f64_n1024 | 8.964 (0.639) | 22.973 (1.276) | 15.756 (0.008) | 200.903 (56.410) | 95.605 (15.434) |
| sin_f32_n1024 | 8.722 (0.760) | 22.834 (0.772) | 8.734 (0.224) | 137.913 (4.735) | 91.248 (6.879) |
| sin_f64_n1024 | 9.055 (0.873) | 22.948 (1.448) | 8.833 (0.082) | 161.261 (5.252) | 108.267 (8.200) |
| cos_f32_n1024 | 8.859 (0.788) | 23.215 (0.964) | 7.914 (0.058) | 168.143 (11.025) | 134.840 (9.408) |
| cos_f64_n1024 | 8.155 (0.235) | 21.664 (1.608) | 8.026 (0.082) | 149.532 (4.304) | 157.631 (16.119) |
| tanh_f32_n1024 | 8.568 (0.756) | 23.371 (1.758) | 8.698 (0.256) | 146.270 (21.485) | 149.591 (26.514) |
| tanh_f64_n1024 | 8.520 (0.647) | 23.934 (0.773) | 9.248 (0.575) | 160.818 (28.957) | 143.543 (17.738) |
| sqrt_f32_n1024 | 8.650 (1.008) | 23.965 (0.856) | 8.889 (0.460) | 141.256 (25.997) | 141.996 (26.046) |
| sqrt_f64_n1024 | 8.563 (0.785) | 24.324 (0.691) | 9.155 (0.355) | 148.052 (26.889) | 130.955 (9.135) |
| rsqrt_f32_n1024 | 9.032 (0.903) | 23.905 (0.920) | 9.169 (0.329) | 189.812 (53.002) | 130.922 (9.196) |
| rsqrt_f64_n1024 | 8.698 (0.225) | 23.395 (0.727) | 8.513 (0.108) | 182.627 (45.175) | 133.113 (12.746) |
| compare_f32_n1024 | unsupported | 448.491 (5.038) | 8.816 (0.255) | 103.464 (16.307) | 124.571 (11.984) |
| compare_f64_n1024 | unsupported | 447.031 (23.859) | 8.866 (0.109) | 94.402 (1.451) | 105.301 (14.073) |
| where_select_f32_n1024 | 19.308 (1.032) | 659.044 (41.918) | 12.147 (0.146) | 105.511 (0.382) | 105.542 (12.726) |
| where_select_f64_n1024 | 19.293 (2.455) | 698.978 (24.784) | 12.637 (0.265) | 109.019 (9.469) | 120.165 (13.622) |
| clamp_f32_n1024 | 51.315 (2.890) | 1129.521 (15.001) | 10.431 (0.128) | 99.781 (11.007) | 124.966 (15.959) |
| clamp_f64_n1024 | 51.303 (2.998) | 641.717 (281.728) | 10.436 (0.180) | 100.947 (10.123) | 129.152 (9.405) |
| reduce_sum_f32_n1024 | 22.255 (0.332) | 191.741 (3.568) | 14.746 (0.470) | 97.525 (0.595) | 156.512 (18.844) |
| reduce_sum_f64_n1024 | 22.598 (0.372) | 192.442 (5.202) | 14.107 (0.224) | 99.629 (3.298) | 167.515 (8.105) |
| reduce_prod_f32_n1024 | 21.300 (0.916) | 210.547 (5.934) | 11.937 (0.163) | 102.308 (14.927) | 173.254 (12.252) |
| reduce_prod_f64_n1024 | 20.434 (1.240) | 220.701 (5.462) | 11.786 (0.182) | 154.776 (5.045) | 180.799 (6.771) |
| reduce_max_f32_n1024 | 20.940 (1.810) | 199.979 (29.741) | 12.938 (0.492) | 125.664 (2.822) | 107.760 (21.107) |
| reduce_max_f64_n1024 | 22.724 (2.683) | 196.149 (4.333) | 12.502 (0.272) | 110.095 (9.695) | 95.453 (1.286) |
| reduce_min_f32_n1024 | 20.653 (1.544) | 214.257 (6.181) | 13.408 (0.324) | 109.384 (14.438) | 93.920 (0.457) |
| reduce_min_f64_n1024 | 22.549 (1.296) | 214.236 (2.384) | 13.236 (0.563) | 97.250 (2.371) | 94.295 (0.602) |
| reduce_sum_squares_f32_n1024 | 21.280 (0.952) | 230.776 (3.407) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1024 | 21.665 (0.354) | 220.766 (25.237) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1024 | unsupported | unsupported | 13.098 (0.150) | 102.642 (0.661) | 104.943 (11.028) |
| reduce_mean_f64_n1024 | unsupported | unsupported | 13.071 (0.899) | 103.127 (0.649) | 125.334 (9.051) |
| add_f32_n1048576 | 48.536 (0.272) | 49.465 (0.412) | 39.765 (0.048) | 108.638 (0.493) | 103.140 (1.782) |
| add_f64_n1048576 | 83.269 (0.349) | 96.700 (0.995) | 77.800 (0.133) | 97.177 (3.147) | 98.898 (2.126) |
| sub_f32_n1048576 | 43.740 (0.124) | 45.132 (0.139) | 39.709 (0.031) | 82.749 (2.166) | 102.491 (0.851) |
| sub_f64_n1048576 | 82.759 (0.078) | 86.878 (0.260) | 77.710 (0.066) | 82.415 (0.534) | 97.997 (0.699) |
| mul_f32_n1048576 | 43.684 (0.081) | 44.741 (0.143) | 39.742 (0.029) | 83.499 (2.434) | 85.132 (6.437) |
| mul_f64_n1048576 | 82.803 (0.101) | 87.302 (0.588) | 77.675 (0.184) | 82.489 (0.691) | 83.101 (0.474) |
| div_f32_n1048576 | 43.699 (0.144) | 45.337 (0.488) | 39.949 (0.034) | 70.882 (2.047) | 78.933 (1.498) |
| div_f64_n1048576 | 95.158 (0.111) | 96.916 (0.469) | 94.225 (0.082) | 123.077 (2.119) | 122.727 (1.345) |
| rem_f32_n1048576 | 42.486 (0.128) | 97.915 (0.323) | 39.883 (0.069) | 80.490 (5.530) | 85.583 (4.084) |
| rem_f64_n1048576 | 80.660 (0.314) | 186.884 (0.414) | 78.270 (0.114) | 83.127 (1.342) | 87.389 (3.652) |
| pow_f32_n1048576 | 42.170 (0.243) | 97.872 (0.795) | 41.544 (0.076) | 121.124 (9.821) | 82.092 (3.065) |
| pow_f64_n1048576 | 1003.170 (0.760) | 1118.770 (1.146) | 1016.531 (0.465) | 1311.180 (20.285) | 1258.495 (10.221) |
| maximum_f32_n1048576 | 42.363 (0.141) | 98.115 (0.557) | 39.867 (0.023) | 119.870 (12.676) | 121.108 (16.758) |
| maximum_f64_n1048576 | 81.597 (0.368) | 187.110 (0.315) | 78.391 (0.122) | 103.915 (6.022) | 83.550 (0.829) |
| minimum_f32_n1048576 | 42.356 (0.072) | 98.375 (0.674) | 39.864 (0.060) | 116.264 (4.530) | 79.467 (2.312) |
| minimum_f64_n1048576 | 81.154 (0.231) | 187.115 (0.217) | 78.385 (0.092) | 128.498 (2.108) | 83.658 (0.700) |
| neg_f32_n1048576 | 29.657 (0.324) | 29.796 (0.076) | 27.290 (0.027) | 111.905 (3.135) | 74.798 (1.708) |
| neg_f64_n1048576 | 56.374 (0.486) | 57.586 (0.709) | 52.979 (0.052) | 110.094 (3.377) | 69.816 (2.085) |
| abs_f32_n1048576 | 29.362 (0.180) | 56.189 (0.529) | 27.385 (0.020) | 99.838 (2.987) | 74.289 (0.507) |
| abs_f64_n1048576 | 56.191 (0.289) | 107.113 (0.284) | 53.124 (0.071) | 102.724 (1.711) | 71.936 (2.477) |
| sign_f32_n1048576 | 29.436 (0.054) | 56.369 (0.917) | 27.435 (0.110) | 103.555 (0.507) | 74.308 (2.360) |
| sign_f64_n1048576 | 56.391 (0.356) | 107.484 (0.447) | 53.177 (0.067) | 99.490 (15.941) | 75.278 (4.512) |
| conj_c32_n1048576 | 55.563 (0.361) | 107.628 (0.258) | 52.981 (0.099) | 73.917 (2.012) | 68.720 (1.683) |
| conj_c64_n1048576 | 109.623 (0.775) | 209.371 (0.206) | 104.033 (0.089) | 113.208 (1.478) | 116.605 (2.303) |
| exp_f32_n1048576 | 29.637 (0.075) | 29.909 (0.199) | 27.348 (0.024) | 75.698 (1.781) | 73.703 (4.610) |
| exp_f64_n1048576 | 183.169 (0.519) | 187.780 (0.620) | 180.251 (0.100) | 198.868 (2.304) | 203.017 (2.915) |
| expm1_f32_n1048576 | 29.693 (0.258) | 29.981 (0.105) | 27.470 (0.036) | 92.659 (13.831) | 88.604 (12.482) |
| expm1_f64_n1048576 | 227.693 (0.838) | 232.415 (0.817) | 224.394 (0.107) | 253.810 (2.430) | 265.978 (9.758) |
| log_f32_n1048576 | 29.748 (0.151) | 30.129 (0.126) | 27.460 (0.027) | 102.078 (19.185) | 84.861 (13.629) |
| log_f64_n1048576 | 351.705 (0.494) | 357.147 (1.856) | 349.781 (0.140) | 404.656 (9.791) | 398.527 (9.020) |
| log1p_f32_n1048576 | 29.471 (0.062) | 29.881 (0.073) | 27.470 (0.095) | 96.749 (11.925) | 101.943 (5.899) |
| log1p_f64_n1048576 | 363.276 (0.426) | 368.030 (0.392) | 362.117 (0.256) | 417.158 (12.414) | 422.314 (12.355) |
| erf_f32_n1048576 | 29.664 (0.241) | 29.949 (0.113) | 27.653 (0.194) | 170.219 (26.481) | 101.585 (8.368) |
| erf_f64_n1048576 | 529.020 (0.587) | 534.737 (0.618) | 529.294 (0.269) | 691.118 (12.979) | 659.027 (12.694) |
| sin_f32_n1048576 | 29.532 (0.270) | 29.966 (0.077) | 27.540 (0.032) | 151.898 (33.648) | 136.259 (18.954) |
| sin_f64_n1048576 | 200.970 (1.110) | 204.324 (1.139) | 211.546 (0.118) | 238.237 (0.799) | 239.772 (0.792) |
| cos_f32_n1048576 | 29.619 (0.240) | 30.199 (0.247) | 27.572 (0.047) | 130.407 (9.816) | 128.925 (15.961) |
| cos_f64_n1048576 | 199.800 (1.125) | 204.708 (0.849) | 211.072 (0.102) | 237.610 (0.573) | 239.203 (1.330) |
| tanh_f32_n1048576 | 29.832 (0.232) | 30.016 (0.224) | 27.473 (0.069) | 111.549 (12.159) | 118.959 (16.654) |
| tanh_f64_n1048576 | 170.196 (0.939) | 174.134 (1.026) | 167.155 (0.047) | 190.919 (1.259) | 191.456 (2.392) |
| sqrt_f32_n1048576 | 29.643 (0.038) | 29.926 (0.106) | 27.438 (0.114) | 103.501 (7.871) | 100.080 (3.676) |
| sqrt_f64_n1048576 | 93.513 (0.273) | 95.667 (0.670) | 91.863 (0.044) | 103.455 (4.864) | 102.510 (1.145) |
| rsqrt_f32_n1048576 | 29.534 (0.254) | 29.851 (0.099) | 27.391 (0.037) | 151.050 (0.902) | 99.022 (1.058) |
| rsqrt_f64_n1048576 | 62.391 (0.297) | 64.456 (0.546) | 61.523 (0.054) | 139.520 (4.963) | 94.939 (1.024) |
| compare_f32_n1048576 | 32.199 (0.193) | 88.558 (1.717) | 30.432 (0.057) | 94.668 (1.123) | 101.887 (1.073) |
| compare_f64_n1048576 | 57.333 (0.100) | 164.908 (0.330) | 55.956 (0.024) | 94.699 (2.196) | 105.201 (0.511) |
| where_select_f32_n1048576 | 46.312 (0.217) | 118.209 (0.687) | 43.005 (0.078) | 120.651 (2.311) | 103.540 (2.602) |
| where_select_f64_n1048576 | 85.266 (0.497) | 206.879 (0.541) | 80.898 (0.059) | 117.218 (4.389) | 97.845 (2.869) |
| clamp_f32_n1048576 | 87.936 (1.131) | 560.522 (13.558) | 28.275 (0.024) | 95.224 (0.276) | 106.056 (4.440) |
| clamp_f64_n1048576 | 162.811 (0.979) | 551.144 (10.849) | 62.184 (0.048) | 92.926 (8.746) | 99.061 (4.155) |
| reduce_sum_f32_n1048576 | 40.716 (0.412) | 62.301 (0.169) | 16.217 (0.223) | 165.544 (5.400) | 150.847 (20.923) |
| reduce_sum_f64_n1048576 | 48.239 (0.548) | 94.968 (0.268) | 32.405 (0.024) | 184.823 (6.305) | 159.109 (3.416) |
| reduce_prod_f32_n1048576 | 43.669 (0.646) | 69.738 (3.163) | 16.110 (0.015) | 158.610 (2.372) | 156.310 (2.131) |
| reduce_prod_f64_n1048576 | 50.616 (1.502) | 95.121 (0.296) | 32.188 (0.038) | 157.030 (1.987) | 156.752 (3.000) |
| reduce_max_f32_n1048576 | 65.848 (1.112) | 74.156 (0.226) | 16.528 (0.018) | 151.778 (1.427) | 155.627 (2.116) |
| reduce_max_f64_n1048576 | 79.598 (0.196) | 112.305 (0.123) | 41.605 (0.018) | 151.423 (1.497) | 157.934 (4.309) |
| reduce_min_f32_n1048576 | 64.769 (0.660) | 74.140 (0.302) | 16.532 (0.087) | 157.999 (9.728) | 175.599 (16.122) |
| reduce_min_f64_n1048576 | 78.002 (0.490) | 112.358 (0.165) | 41.621 (0.025) | 164.944 (9.532) | 161.701 (16.867) |
| reduce_sum_squares_f32_n1048576 | 43.783 (0.525) | 74.138 (0.316) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1048576 | 53.483 (0.582) | 96.760 (0.237) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1048576 | unsupported | unsupported | 16.288 (0.161) | 187.331 (19.796) | 151.649 (24.586) |
| reduce_mean_f64_n1048576 | unsupported | unsupported | 32.862 (0.118) | 181.584 (13.006) | 165.658 (26.401) |
| reduce_sum_full_f32_n1024 | 25.621 (1.799) | 163.252 (6.692) | 14.021 (1.001) | 169.915 (6.720) | 171.234 (5.969) |
| reduce_prod_full_f32_n1024 | 20.367 (1.224) | 179.788 (10.100) | 10.348 (0.099) | 127.926 (9.618) | 139.923 (10.883) |
| reduce_max_full_f32_n1024 | 16.591 (0.841) | 197.744 (9.128) | 11.481 (0.132) | 128.341 (6.083) | 172.450 (13.133) |
| reduce_min_full_f32_n1024 | 12.985 (0.419) | 197.153 (5.420) | 11.795 (0.206) | 94.382 (1.214) | 174.693 (28.158) |
| compare_eq_f32_n1024 | unsupported | 426.533 (9.459) | 8.763 (0.095) | 94.245 (3.893) | 147.708 (26.444) |
| compare_lt_f32_n1024 | unsupported | 430.100 (14.312) | 8.767 (0.062) | 94.018 (6.445) | 132.697 (19.241) |
| compare_le_f32_n1024 | unsupported | 426.361 (9.909) | 8.603 (0.040) | 90.226 (11.978) | 107.035 (14.961) |
| compare_ge_f32_n1024 | unsupported | 448.684 (37.476) | 8.681 (0.206) | 97.103 (3.695) | 100.901 (4.944) |
| reduce_sum_full_f64_n1024 | 23.307 (0.399) | 193.769 (2.628) | 11.840 (0.188) | 160.109 (4.030) | 165.156 (11.362) |
| reduce_prod_full_f64_n1024 | 22.162 (1.442) | 195.064 (3.346) | 9.996 (0.100) | 142.607 (6.382) | 77.842 (2.868) |
| reduce_max_full_f64_n1024 | 19.550 (1.619) | 200.048 (14.165) | 11.098 (0.253) | 158.058 (2.297) | 137.750 (7.229) |
| reduce_min_full_f64_n1024 | 15.760 (0.967) | 192.154 (4.126) | 11.475 (0.194) | 165.965 (5.828) | 166.760 (10.937) |
| compare_eq_f64_n1024 | unsupported | 420.061 (6.340) | 8.771 (0.155) | 160.817 (31.059) | 152.215 (20.106) |
| compare_lt_f64_n1024 | unsupported | 422.230 (6.101) | 8.811 (0.163) | 140.274 (11.177) | 136.440 (16.276) |
| compare_le_f64_n1024 | unsupported | 423.296 (3.816) | 8.988 (0.474) | 122.255 (15.931) | 127.947 (10.395) |
| compare_ge_f64_n1024 | unsupported | 423.911 (9.039) | 8.833 (0.112) | 95.283 (13.300) | 102.921 (14.254) |
| reduce_sum_full_f32_n1048576 | 2530.885 (3.835) | 2646.727 (10.262) | 18.439 (0.047) | 171.614 (6.938) | 166.163 (10.306) |
| reduce_prod_full_f32_n1048576 | 2569.349 (6.204) | 2615.031 (8.480) | 18.401 (0.084) | 165.390 (28.738) | 137.146 (37.196) |
| reduce_max_full_f32_n1048576 | 9976.084 (176.694) | 9862.392 (8.424) | 18.712 (0.033) | 146.418 (5.707) | 158.070 (12.256) |
| reduce_min_full_f32_n1048576 | 9777.576 (3.474) | 9850.622 (10.642) | 18.834 (0.041) | 105.533 (1.708) | 167.818 (23.868) |
| compare_eq_f32_n1048576 | 32.853 (0.104) | 87.906 (0.376) | 30.265 (0.031) | 93.919 (2.725) | 124.178 (14.319) |
| compare_lt_f32_n1048576 | 32.837 (0.084) | 87.665 (0.537) | 30.268 (0.043) | 94.629 (0.839) | 96.745 (0.927) |
| compare_le_f32_n1048576 | 32.024 (0.145) | 87.965 (1.064) | 30.259 (0.052) | 94.807 (0.577) | 96.578 (3.893) |
| compare_ge_f32_n1048576 | 31.974 (0.200) | 87.840 (0.485) | 30.262 (0.026) | 95.270 (1.112) | 97.179 (1.185) |
| reduce_sum_full_f64_n1048576 | 3015.889 (1.176) | 3093.969 (1.852) | 37.373 (0.049) | 142.557 (2.051) | 166.205 (11.429) |
| reduce_prod_full_f64_n1048576 | 3015.698 (2.706) | 3093.533 (2.366) | 37.466 (0.017) | 150.349 (11.216) | 157.865 (26.071) |
| reduce_max_full_f64_n1048576 | 12415.891 (5.500) | 12529.911 (4.944) | 44.564 (0.029) | 162.735 (2.328) | 166.343 (7.990) |
| reduce_min_full_f64_n1048576 | 12413.521 (3.856) | 12530.836 (2.219) | 44.596 (0.065) | 168.363 (8.248) | 171.803 (15.765) |
| compare_eq_f64_n1048576 | 57.405 (0.322) | 164.569 (0.174) | 56.677 (0.024) | 133.871 (18.268) | 118.875 (10.445) |
| compare_lt_f64_n1048576 | 57.333 (0.140) | 164.562 (0.188) | 55.952 (0.022) | 88.800 (3.540) | 100.943 (3.169) |
| compare_le_f64_n1048576 | 57.299 (0.128) | 164.564 (0.370) | 55.974 (0.063) | 91.711 (0.528) | 95.977 (2.310) |
| compare_ge_f64_n1048576 | 57.368 (0.291) | 164.796 (0.778) | 55.946 (0.061) | 92.570 (3.817) | 98.910 (0.362) |
| reduce_mean_full_f32_n1024 | unsupported | unsupported | 12.728 (0.282) | 98.244 (4.326) | 102.328 (10.941) |
| reduce_sum_squares_full_f32_n1024 | 11.236 (0.600) | 145.827 (5.930) | 14.806 (0.757) | 94.626 (2.222) | 156.569 (16.379) |
| reduce_mean_full_f64_n1024 | unsupported | unsupported | 11.685 (0.192) | 139.197 (4.766) | 171.092 (24.055) |
| reduce_sum_squares_full_f64_n1024 | 9.087 (0.561) | 139.833 (5.780) | 14.372 (0.352) | 163.084 (5.451) | 164.486 (6.305) |
| reduce_mean_full_f32_n1048576 | unsupported | unsupported | 18.432 (0.042) | 177.739 (22.841) | 166.688 (26.790) |
| reduce_sum_squares_full_f32_n1048576 | 2541.102 (3.842) | 2657.842 (14.017) | 17.611 (0.056) | 164.989 (4.977) | 159.867 (6.369) |
| reduce_mean_full_f64_n1048576 | unsupported | unsupported | 37.428 (0.070) | 190.189 (16.055) | 171.887 (20.322) |
| reduce_sum_squares_full_f64_n1048576 | 3294.637 (1.355) | 3379.363 (3.102) | 33.189 (0.020) | 164.499 (5.050) | 162.842 (6.238) |
| scale_real_f32_n1024 | unsupported | unsupported | 9.894 (0.072) | 166.129 (32.563) | 144.414 (31.159) |
| scale_real_f64_n1024 | unsupported | unsupported | 9.835 (0.326) | 149.288 (33.132) | 131.142 (23.186) |
| scale_real_c32_n1024 | unsupported | unsupported | 9.938 (0.140) | 136.447 (31.495) | 127.992 (10.689) |
| scale_real_c64_n1024 | unsupported | unsupported | 10.098 (0.122) | 140.000 (16.560) | 100.768 (17.098) |
| scale_complex_c32_n1024 | unsupported | unsupported | 9.913 (0.058) | 120.439 (14.971) | 99.660 (11.453) |
| scale_complex_c64_n1024 | unsupported | unsupported | 10.522 (0.434) | 103.710 (2.642) | 96.913 (10.016) |
| scale_real_f32_n1048576 | unsupported | unsupported | 28.123 (0.016) | 105.781 (0.424) | 97.641 (3.171) |
| scale_real_f64_n1048576 | unsupported | unsupported | 53.692 (0.049) | 106.793 (0.376) | 98.419 (0.606) |
| scale_real_c32_n1048576 | unsupported | unsupported | 53.647 (0.083) | 107.440 (1.444) | 95.516 (3.849) |
| scale_real_c64_n1048576 | unsupported | unsupported | 104.724 (0.111) | 117.850 (1.839) | 118.174 (2.174) |
| scale_complex_c32_n1048576 | unsupported | unsupported | 53.688 (0.112) | 108.347 (0.655) | 97.146 (0.886) |
| scale_complex_c64_n1048576 | unsupported | unsupported | 104.732 (0.058) | 116.123 (0.600) | 117.173 (2.204) |
| add_i32_n1024 | 8.367 (0.357) | 14.036 (1.594) | 9.324 (0.198) | 103.306 (0.785) | 98.602 (0.424) |
| sub_i32_n1024 | 9.266 (0.555) | 13.197 (0.634) | 9.336 (0.064) | 102.860 (0.359) | 86.006 (12.520) |
| mul_i32_n1024 | 10.061 (0.726) | 13.240 (0.490) | 9.535 (0.164) | 106.275 (9.246) | 81.834 (10.812) |
| div_i32_n1024 | 85.402 (2.283) | 321.545 (15.926) | 10.086 (0.147) | 156.718 (23.030) | 82.198 (8.206) |
| rem_i32_n1024 | 88.885 (1.679) | 395.943 (1.633) | 9.363 (0.083) | 100.933 (14.848) | 83.574 (7.250) |
| maximum_i32_n1024 | 18.208 (0.190) | 266.047 (7.227) | 9.277 (0.138) | 110.520 (11.790) | 81.001 (6.046) |
| minimum_i32_n1024 | 18.197 (0.186) | 262.568 (10.730) | 9.397 (0.156) | 105.306 (8.554) | 80.039 (6.409) |
| neg_i32_n1024 | 16.438 (0.133) | 21.366 (0.576) | 8.699 (0.102) | 103.222 (8.926) | 77.732 (5.778) |
| abs_i32_n1024 | 15.858 (0.205) | 132.048 (0.779) | 11.752 (0.345) | 94.091 (7.785) | 78.677 (7.592) |
| sign_i32_n1024 | 15.909 (0.199) | 132.223 (4.004) | 9.067 (0.261) | 93.134 (8.779) | 79.727 (6.179) |
| reduce_sum_i32_n1024 | 23.093 (0.558) | 137.162 (1.749) | 15.958 (0.218) | 98.030 (3.131) | 69.920 (3.178) |
| reduce_max_i32_n1024 | 23.808 (0.690) | 135.736 (2.592) | 13.966 (0.257) | 93.181 (0.530) | 74.477 (4.492) |
| reduce_min_i32_n1024 | 24.488 (0.839) | 136.409 (2.358) | 13.563 (0.069) | 97.414 (3.557) | 73.765 (4.017) |
| add_i64_n1024 | 20.132 (0.357) | 31.324 (0.709) | 9.664 (0.236) | 107.756 (6.574) | 77.522 (3.964) |
| sub_i64_n1024 | 20.076 (0.191) | 31.479 (0.216) | 9.904 (0.181) | 108.492 (8.528) | 79.558 (4.380) |
| mul_i64_n1024 | 19.735 (0.120) | 31.384 (0.460) | 9.120 (0.112) | 105.588 (5.831) | 78.365 (4.291) |
| div_i64_n1024 | 105.230 (2.958) | 365.727 (12.899) | 10.056 (0.146) | 164.788 (22.310) | 83.256 (4.583) |
| rem_i64_n1024 | 98.601 (0.301) | 355.245 (16.390) | 9.911 (0.525) | 100.787 (9.835) | 86.702 (15.941) |
| maximum_i64_n1024 | 18.207 (0.245) | 245.079 (3.644) | 9.809 (0.536) | 106.775 (2.482) | 80.678 (3.941) |
| minimum_i64_n1024 | 18.365 (0.155) | 212.722 (1.891) | 9.306 (0.064) | 108.231 (2.169) | 76.110 (2.372) |
| neg_i64_n1024 | 16.404 (0.106) | 21.058 (0.187) | 8.726 (0.081) | 101.956 (3.449) | 73.107 (2.763) |
| abs_i64_n1024 | 15.969 (0.205) | 129.261 (1.741) | 11.368 (0.166) | 90.071 (1.173) | 72.712 (1.824) |
| sign_i64_n1024 | 16.004 (0.286) | 124.677 (1.910) | 9.094 (0.219) | 92.316 (1.664) | 72.184 (1.495) |
| reduce_sum_i64_n1024 | 23.237 (0.264) | 136.992 (1.522) | 15.761 (0.300) | 129.535 (2.198) | 68.590 (1.803) |
| reduce_max_i64_n1024 | 24.004 (0.724) | 137.350 (2.860) | 13.081 (0.163) | 140.093 (2.190) | 68.521 (1.448) |
| reduce_min_i64_n1024 | 23.736 (0.759) | 137.277 (1.502) | 13.630 (0.611) | 140.100 (3.189) | 103.690 (14.753) |
| add_i32_n1048576 | 44.583 (0.481) | 47.172 (1.026) | 39.837 (0.035) | 152.167 (22.985) | 94.010 (11.448) |
| sub_i32_n1048576 | 43.975 (0.138) | 45.048 (0.425) | 39.824 (0.046) | 126.417 (20.729) | 76.734 (10.452) |
| mul_i32_n1048576 | 43.921 (0.069) | 44.782 (0.132) | 39.919 (0.086) | 124.956 (13.383) | 73.045 (1.107) |
| div_i32_n1048576 | 95.975 (1.469) | 185.470 (2.175) | 39.959 (0.008) | 155.751 (21.763) | 76.203 (5.806) |
| rem_i32_n1048576 | 90.506 (0.601) | 181.789 (1.556) | 40.033 (0.168) | 125.128 (20.951) | 91.596 (9.593) |
| maximum_i32_n1048576 | 43.589 (0.175) | 101.963 (0.325) | 39.740 (0.038) | 143.066 (22.946) | 85.453 (15.458) |
| minimum_i32_n1048576 | 43.527 (0.160) | 102.217 (0.401) | 39.760 (0.035) | 136.458 (20.032) | 83.115 (17.177) |
| neg_i32_n1048576 | 30.359 (0.030) | 30.600 (0.046) | 27.346 (0.027) | 132.291 (18.437) | 78.341 (7.683) |
| abs_i32_n1048576 | 30.234 (0.099) | 58.267 (0.105) | 27.319 (0.034) | 116.607 (14.543) | 78.757 (5.342) |
| sign_i32_n1048576 | 30.568 (0.109) | 58.284 (0.086) | 27.351 (0.027) | 122.517 (13.829) | 79.587 (9.263) |
| reduce_sum_i32_n1048576 | 41.705 (1.293) | 75.541 (0.790) | 16.115 (0.121) | 94.300 (2.365) | 77.792 (3.498) |
| reduce_max_i32_n1048576 | 45.153 (0.975) | 79.210 (1.324) | 16.244 (0.025) | 89.514 (2.508) | 70.982 (0.849) |
| reduce_min_i32_n1048576 | 46.056 (0.724) | 83.141 (0.994) | 16.254 (0.032) | 92.321 (2.290) | 70.732 (1.954) |
| add_i64_n1048576 | 84.930 (1.128) | 87.716 (0.926) | 77.755 (0.064) | 141.694 (5.144) | 82.671 (0.376) |
| sub_i64_n1048576 | 83.727 (0.170) | 86.816 (0.359) | 77.726 (0.089) | 145.765 (5.044) | 82.730 (0.255) |
| mul_i64_n1048576 | 83.622 (0.221) | 86.895 (0.263) | 77.680 (0.044) | 153.101 (10.032) | 82.785 (0.536) |
| div_i64_n1048576 | 147.131 (1.730) | 292.231 (3.162) | 80.393 (0.061) | 190.269 (28.496) | 89.720 (2.562) |
| rem_i64_n1048576 | 140.984 (1.197) | 291.442 (3.084) | 80.081 (0.064) | 166.355 (2.245) | 115.966 (1.595) |
| maximum_i64_n1048576 | 82.250 (0.081) | 197.708 (0.352) | 77.579 (0.068) | 149.537 (6.539) | 91.728 (6.006) |
| minimum_i64_n1048576 | 82.341 (0.130) | 197.720 (0.406) | 77.583 (0.049) | 145.429 (4.578) | 87.067 (5.877) |
| neg_i64_n1048576 | 54.881 (0.097) | 56.025 (0.255) | 52.952 (0.114) | 128.712 (10.326) | 78.288 (7.353) |
| abs_i64_n1048576 | 55.284 (0.088) | 110.110 (0.466) | 53.091 (0.113) | 115.938 (16.893) | 75.727 (7.441) |
| sign_i64_n1048576 | 54.530 (0.156) | 110.109 (0.365) | 52.975 (0.046) | 120.707 (15.805) | 79.229 (6.214) |
| reduce_sum_i64_n1048576 | 45.490 (0.565) | 102.175 (1.516) | 28.701 (0.029) | 168.276 (3.528) | 144.844 (2.734) |
| reduce_max_i64_n1048576 | 46.224 (0.577) | 102.493 (1.546) | 28.814 (0.012) | 156.951 (1.924) | 158.587 (4.968) |
| reduce_min_i64_n1048576 | 46.318 (0.634) | 103.463 (1.474) | 28.817 (0.023) | 157.664 (3.202) | 158.370 (4.742) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.008 | 0 |
| add_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.608 | 0 |
| sub_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.009 | 0 |
| sub_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.873 | 0 |
| mul_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.646 | 0 |
| mul_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.503 | 0 |
| div_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.420 | 0 |
| div_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.421 | 0 |
| rem_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.267 | 0 |
| rem_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.338 | 0 |
| pow_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.305 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.517 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.284 | 0 |
| maximum_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.268 | 0 |
| minimum_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.314 | 0 |
| minimum_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.340 | 0 |
| neg_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.069 | 0 |
| neg_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.095 | 0 |
| abs_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.146 | 0 |
| abs_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.146 | 0 |
| sign_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.209 | 0 |
| sign_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.178 | 0 |
| conj_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.221 | 0 |
| conj_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.378 | 0 |
| exp_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.881 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.641 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.607 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.472 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.502 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.488 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.337 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.348 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.367 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.295 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.233 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.318 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.268 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.088 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.193 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.181 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.214 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.192 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.312 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.227 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| where_select_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.471 | 0 |
| where_select_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.469 | 0 |
| clamp_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.284 | 0 |
| clamp_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.283 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.424 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.446 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.726 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.616 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.680 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.909 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.644 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.886 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.724 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.773 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.553 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.665 | 0 |
| sub_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.799 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.648 | 0 |
| mul_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.796 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.650 | 0 |
| div_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.797 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.045 | 0 |
| rem_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.719 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.581 | 0 |
| pow_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.699 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.006 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.711 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.611 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.711 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.597 | 0 |
| neg_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.898 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.804 | 0 |
| abs_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.879 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.798 | 0 |
| sign_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.884 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.805 | 0 |
| conj_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.778 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.754 | 0 |
| exp_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.897 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.931 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.900 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.643 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.904 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.814 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.886 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.906 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.898 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.116 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.890 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.216 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.896 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.197 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.909 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.723 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.897 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.992 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.890 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.997 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.061 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.669 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.964 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.729 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.814 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.605 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.606 | 0.001812 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.087 | 2.148e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.795 | 0.005478 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.239 | 3.659e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.107 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.547 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.073 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.496 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.802 | 0.0003528 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.423 | 3.293e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_full_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 0.820 | 0.0001333 |
| reduce_prod_full_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.303 | 0.003334 |
| reduce_max_full_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.124 | 0 |
| reduce_min_full_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.324 | 0 |
| compare_eq_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_lt_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_le_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_ge_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| reduce_sum_full_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.492 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.837 | 0 |
| reduce_max_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.502 | 0 |
| reduce_min_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.017 | 0 |
| compare_eq_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_lt_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_le_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| compare_ge_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. Latest GPU1 diagnostic 20261013_192206 on dfbdf6b reproduced Gt F32 last/first-ten batch growth of 2.516–2.796 in three independent processes. Other directions share the allocation path but are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md; excluded pending issue 107 instrumentation, not a numerical-support failure. |
| reduce_sum_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.531 | 0.07061 |
| reduce_prod_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.569 | 0.1957 |
| reduce_max_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 9.976 | 0 |
| reduce_min_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 9.778 | 0 |
| compare_eq_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.103 | 0 |
| compare_lt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.102 | 0 |
| compare_le_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.050 | 0 |
| compare_ge_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.046 | 0 |
| reduce_sum_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.016 | 0.001745 |
| reduce_prod_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.016 | 6.451e-05 |
| reduce_max_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 12.416 | 0 |
| reduce_min_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 12.414 | 0 |
| compare_eq_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.674 | 0 |
| compare_lt_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.669 | 0 |
| compare_le_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.667 | 0 |
| compare_ge_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.672 | 0 |
| reduce_mean_full_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.876 | 0.0001766 |
| reduce_mean_full_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.326 | 0 |
| reduce_mean_full_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.541 | 0.001429 |
| reduce_mean_full_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.295 | 9.982e-07 |
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
| add_i32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.142 | 0 |
| sub_i32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.372 | 0 |
| mul_i32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.576 | 0 |
| div_i32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.733 | 0 |
| rem_i32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.844 | 0 |
| maximum_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.331 | 0 |
| minimum_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.329 | 0 |
| neg_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.104 | 0 |
| abs_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.030 | 0 |
| sign_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.036 | 0 |
| reduce_sum_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.956 | 0 |
| reduce_max_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.047 | 0 |
| reduce_min_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.134 | 0 |
| add_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.577 | 0 |
| sub_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.570 | 0 |
| mul_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.526 | 0 |
| div_i64_n1024 | tenferro-cuda-direct | 32 | 15 | 3.367 | 0 |
| rem_i64_n1024 | tenferro-cuda-direct | 32 | 15 | 3.155 | 0 |
| maximum_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.331 | 0 |
| minimum_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.351 | 0 |
| neg_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.100 | 0 |
| abs_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.044 | 0 |
| sign_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.049 | 0 |
| reduce_sum_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.974 | 0 |
| reduce_max_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.073 | 0 |
| reduce_min_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.038 | 0 |
| add_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.853 | 0 |
| sub_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.814 | 0 |
| mul_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.811 | 0 |
| div_i32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.071 | 0 |
| rem_i32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.896 | 0 |
| maximum_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.790 | 0 |
| minimum_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.786 | 0 |
| neg_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.943 | 0 |
| abs_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.935 | 0 |
| sign_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.956 | 0 |
| reduce_sum_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.669 | 0 |
| reduce_max_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.890 | 0 |
| reduce_min_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.948 | 0 |
| add_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.718 | 0 |
| sub_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.679 | 0 |
| mul_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.676 | 0 |
| div_i64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.354 | 0 |
| rem_i64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.256 | 0 |
| maximum_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.632 | 0 |
| minimum_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.635 | 0 |
| neg_i64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.512 | 0 |
| abs_i64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.538 | 0 |
| sign_i64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.490 | 0 |
| reduce_sum_i64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.911 | 0 |
| reduce_max_i64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.958 | 0 |
| reduce_min_i64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.964 | 0 |
| add_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.254 | 0 |
| add_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.613 | 0 |
| sub_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.770 | 0 |
| sub_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.881 | 0 |
| mul_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.141 | 0 |
| mul_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.438 | 0 |
| div_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.566 | 0 |
| div_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.537 | 0 |
| rem_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.167 | 0 |
| rem_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.939 | 0 |
| pow_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.435 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 4.095 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.470 | 0 |
| maximum_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.662 | 0 |
| minimum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.252 | 0 |
| minimum_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.321 | 0 |
| neg_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.849 | 0 |
| neg_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.895 | 0 |
| abs_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.519 | 0 |
| abs_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.342 | 0 |
| sign_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.700 | 0 |
| sign_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.808 | 0 |
| conj_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.694 | 0 |
| conj_c64_n1024 | tenferro-cuda-eager | 32 | 15 | 4.228 | 0 |
| exp_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.709 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.728 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.700 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.832 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.966 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.198 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.936 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.997 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.016 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.941 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.923 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.937 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.971 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.773 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.992 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.063 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.068 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.113 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.060 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.995 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.588 | 0 |
| compare_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.576 | 0 |
| where_select_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.636 | 0 |
| where_select_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.796 | 0 |
| clamp_f32_n1024 | tenferro-cuda-eager | 2 | 15 | 2.259 | 0 |
| clamp_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.567 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.068 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.079 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.369 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.531 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.200 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.138 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.428 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.428 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 1.846 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.532 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.583 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.547 | 0 |
| sub_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.888 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.780 | 0 |
| mul_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.863 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.794 | 0 |
| div_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.902 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.101 | 0 |
| rem_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.133 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.990 | 0 |
| pow_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.132 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-eager | 2 | 15 | 2.238 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.140 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.994 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.148 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.994 | 0 |
| neg_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.907 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.843 | 0 |
| abs_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.596 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.428 | 0 |
| sign_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.608 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.439 | 0 |
| conj_c32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.444 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.350 | 0 |
| exp_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.914 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.004 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.919 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.719 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.928 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.857 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.912 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.944 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.917 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.139 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.918 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.269 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.933 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.275 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.921 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.786 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.915 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.061 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.910 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.063 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.834 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.639 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.783 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.310 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.242 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.205 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.987 | 0.001812 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.039 | 2.148e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.232 | 0.005478 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.044 | 3.659e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.373 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.594 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.372 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.595 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.372 | 0.0003528 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.096 | 3.293e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_full_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.612 | 0.0001333 |
| reduce_prod_full_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.877 | 0.003334 |
| reduce_max_full_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.164 | 0 |
| reduce_min_full_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.154 | 0 |
| compare_eq_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.412 | 0 |
| compare_lt_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.441 | 0 |
| compare_le_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.411 | 0 |
| compare_ge_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.589 | 0 |
| reduce_sum_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.100 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.121 | 0 |
| reduce_max_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.201 | 0 |
| reduce_min_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.074 | 0 |
| compare_eq_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.360 | 0 |
| compare_lt_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.378 | 0 |
| compare_le_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.386 | 0 |
| compare_ge_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.391 | 0 |
| reduce_sum_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.647 | 0.07061 |
| reduce_prod_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.615 | 0.1957 |
| reduce_max_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 9.862 | 0 |
| reduce_min_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 9.851 | 0 |
| compare_eq_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.813 | 0 |
| compare_lt_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.805 | 0 |
| compare_le_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.815 | 0 |
| compare_ge_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.811 | 0 |
| reduce_sum_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.094 | 0.001745 |
| reduce_prod_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.094 | 6.451e-05 |
| reduce_max_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 12.530 | 0 |
| reduce_min_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 12.531 | 0 |
| compare_eq_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.633 | 0 |
| compare_lt_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.633 | 0 |
| compare_le_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.633 | 0 |
| compare_ge_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.637 | 0 |
| reduce_mean_full_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.333 | 0.0001766 |
| reduce_mean_full_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.237 | 0 |
| reduce_mean_full_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.658 | 0.001429 |
| reduce_mean_full_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.379 | 9.982e-07 |
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
| add_i32_n1024 | tenferro-cuda-eager | 256 | 15 | 3.593 | 0 |
| sub_i32_n1024 | tenferro-cuda-eager | 256 | 15 | 3.378 | 0 |
| mul_i32_n1024 | tenferro-cuda-eager | 256 | 15 | 3.389 | 0 |
| div_i32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.572 | 0 |
| rem_i32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.168 | 0 |
| maximum_i32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.128 | 0 |
| minimum_i32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.101 | 0 |
| neg_i32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.735 | 0 |
| abs_i32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.113 | 0 |
| sign_i32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.116 | 0 |
| reduce_sum_i32_n1024 | tenferro-cuda-eager | 32 | 15 | 4.389 | 0 |
| reduce_max_i32_n1024 | tenferro-cuda-eager | 32 | 15 | 4.344 | 0 |
| reduce_min_i32_n1024 | tenferro-cuda-eager | 32 | 15 | 4.365 | 0 |
| add_i64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.005 | 0 |
| sub_i64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.015 | 0 |
| mul_i64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.009 | 0 |
| div_i64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.926 | 0 |
| rem_i64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.842 | 0 |
| maximum_i64_n1024 | tenferro-cuda-eager | 8 | 15 | 1.961 | 0 |
| minimum_i64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.404 | 0 |
| neg_i64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.695 | 0 |
| abs_i64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.068 | 0 |
| sign_i64_n1024 | tenferro-cuda-eager | 16 | 15 | 1.995 | 0 |
| reduce_sum_i64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.192 | 0 |
| reduce_max_i64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.198 | 0 |
| reduce_min_i64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.196 | 0 |
| add_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.019 | 0 |
| sub_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.883 | 0 |
| mul_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.866 | 0 |
| div_i32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.968 | 0 |
| rem_i32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.909 | 0 |
| maximum_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.263 | 0 |
| minimum_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.271 | 0 |
| neg_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.958 | 0 |
| abs_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.729 | 0 |
| sign_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.730 | 0 |
| reduce_sum_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.417 | 0 |
| reduce_max_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.535 | 0 |
| reduce_min_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.661 | 0 |
| add_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.807 | 0 |
| sub_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.778 | 0 |
| mul_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.781 | 0 |
| div_i64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.338 | 0 |
| rem_i64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.332 | 0 |
| maximum_i64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.163 | 0 |
| minimum_i64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.164 | 0 |
| neg_i64_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.586 | 0 |
| abs_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.524 | 0 |
| sign_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.523 | 0 |
| reduce_sum_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.270 | 0 |
| reduce_max_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.280 | 0 |
| reduce_min_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.311 | 0 |
| add_f32_n1024 | pytorch-cuda | 256 | 15 | 2.335 | 0 |
| add_f64_n1024 | pytorch-cuda | 256 | 15 | 2.375 | 0 |
| sub_f32_n1024 | pytorch-cuda | 256 | 15 | 2.350 | 0 |
| sub_f64_n1024 | pytorch-cuda | 256 | 15 | 2.360 | 0 |
| mul_f32_n1024 | pytorch-cuda | 256 | 15 | 2.336 | 0 |
| mul_f64_n1024 | pytorch-cuda | 256 | 15 | 2.307 | 0 |
| div_f32_n1024 | pytorch-cuda | 256 | 15 | 2.362 | 0 |
| div_f64_n1024 | pytorch-cuda | 256 | 15 | 2.249 | 0 |
| rem_f32_n1024 | pytorch-cuda | 256 | 15 | 2.362 | 0 |
| rem_f64_n1024 | pytorch-cuda | 256 | 15 | 2.392 | 0 |
| pow_f32_n1024 | pytorch-cuda | 256 | 15 | 2.388 | 0.000117 |
| pow_f64_n1024 | pytorch-cuda | 128 | 15 | 3.959 | 1.48e-07 |
| maximum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.286 | 0 |
| maximum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.356 | 0 |
| minimum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.275 | 0 |
| minimum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.322 | 0 |
| neg_f32_n1024 | pytorch-cuda | 256 | 15 | 2.145 | 0 |
| neg_f64_n1024 | pytorch-cuda | 256 | 15 | 2.315 | 0 |
| abs_f32_n1024 | pytorch-cuda | 256 | 15 | 2.736 | 0 |
| abs_f64_n1024 | pytorch-cuda | 256 | 15 | 2.721 | 0 |
| sign_f32_n1024 | pytorch-cuda | 256 | 15 | 2.191 | 0 |
| sign_f64_n1024 | pytorch-cuda | 256 | 15 | 2.241 | 0 |
| conj_c32_n1024 | pytorch-cuda | 256 | 15 | 2.548 | 0 |
| conj_c64_n1024 | pytorch-cuda | 256 | 15 | 2.623 | 0 |
| exp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.110 | 0.0001272 |
| exp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.093 | 1.095e-07 |
| expm1_f32_n1024 | pytorch-cuda | 256 | 15 | 2.186 | 7.911e-05 |
| expm1_f64_n1024 | pytorch-cuda | 256 | 15 | 2.165 | 7.39e-08 |
| log_f32_n1024 | pytorch-cuda | 256 | 15 | 2.202 | 7.945e-05 |
| log_f64_n1024 | pytorch-cuda | 256 | 15 | 2.778 | 4.231e-08 |
| log1p_f32_n1024 | pytorch-cuda | 256 | 15 | 2.238 | 0.0001189 |
| log1p_f64_n1024 | pytorch-cuda | 256 | 15 | 2.874 | 1.108e-07 |
| erf_f32_n1024 | pytorch-cuda | 256 | 15 | 2.180 | 7.914e-05 |
| erf_f64_n1024 | pytorch-cuda | 128 | 15 | 2.017 | 8.536e-08 |
| sin_f32_n1024 | pytorch-cuda | 256 | 15 | 2.236 | 4.348e-05 |
| sin_f64_n1024 | pytorch-cuda | 256 | 15 | 2.261 | 4.153e-08 |
| cos_f32_n1024 | pytorch-cuda | 256 | 15 | 2.026 | 6.342e-05 |
| cos_f64_n1024 | pytorch-cuda | 256 | 15 | 2.055 | 5.905e-08 |
| tanh_f32_n1024 | pytorch-cuda | 256 | 15 | 2.227 | 9.107e-05 |
| tanh_f64_n1024 | pytorch-cuda | 256 | 15 | 2.367 | 4.434e-08 |
| sqrt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.276 | 0 |
| sqrt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.344 | 0 |
| rsqrt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.347 | 6.983e-05 |
| rsqrt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.179 | 6.504e-08 |
| compare_f32_n1024 | pytorch-cuda | 256 | 15 | 2.257 | 0 |
| compare_f64_n1024 | pytorch-cuda | 256 | 15 | 2.270 | 0 |
| where_select_f32_n1024 | pytorch-cuda | 256 | 15 | 3.110 | 0 |
| where_select_f64_n1024 | pytorch-cuda | 256 | 15 | 3.235 | 0 |
| clamp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.670 | 0 |
| clamp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.672 | 0 |
| reduce_sum_f32_n1024 | pytorch-cuda | 256 | 15 | 3.775 | 0.0002742 |
| reduce_sum_f64_n1024 | pytorch-cuda | 256 | 15 | 3.611 | 1.06e-07 |
| reduce_prod_f32_n1024 | pytorch-cuda | 256 | 15 | 3.056 | 0.0003579 |
| reduce_prod_f64_n1024 | pytorch-cuda | 256 | 15 | 3.017 | 4.441e-07 |
| reduce_max_f32_n1024 | pytorch-cuda | 128 | 15 | 1.656 | 0 |
| reduce_max_f64_n1024 | pytorch-cuda | 256 | 15 | 3.201 | 0 |
| reduce_min_f32_n1024 | pytorch-cuda | 128 | 15 | 1.716 | 0 |
| reduce_min_f64_n1024 | pytorch-cuda | 256 | 15 | 3.388 | 0 |
| reduce_sum_squares_f32_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | pytorch-cuda | 256 | 15 | 3.353 | 1.439e-05 |
| reduce_mean_f64_n1024 | pytorch-cuda | 256 | 15 | 3.346 | 6.71e-09 |
| add_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.545 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.490 | 0 |
| sub_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.541 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.487 | 0 |
| mul_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.543 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| div_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.557 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.015 | 0 |
| rem_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.505 | 0 |
| pow_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.659 | 0.0001577 |
| pow_f64_n1048576 | pytorch-cuda | 2 | 15 | 2.033 | 1.48e-07 |
| maximum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.551 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.508 | 0 |
| minimum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.551 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.508 | 0 |
| neg_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.747 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.695 | 0 |
| abs_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.753 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.700 | 0 |
| sign_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.702 | 0 |
| conj_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.695 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.665 | 0 |
| exp_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.750 | 0.0001272 |
| exp_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.884 | 1.095e-07 |
| expm1_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.758 | 7.915e-05 |
| expm1_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.590 | 7.39e-08 |
| log_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.757 | 7.946e-05 |
| log_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.798 | 4.308e-08 |
| log1p_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.758 | 0.000119 |
| log1p_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.897 | 1.11e-07 |
| erf_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.770 | 7.923e-05 |
| erf_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.117 | 8.536e-08 |
| sin_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.763 | 4.348e-05 |
| sin_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.385 | 4.153e-08 |
| cos_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.765 | 6.342e-05 |
| cos_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.377 | 5.905e-08 |
| tanh_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.758 | 9.135e-05 |
| tanh_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.674 | 4.437e-08 |
| sqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.940 | 0 |
| rsqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.753 | 6.983e-05 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.969 | 6.504e-08 |
| compare_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.948 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.581 | 0 |
| where_select_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.752 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.589 | 0 |
| clamp_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.810 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.990 | 0 |
| reduce_sum_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.076 | 0.002948 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.074 | 2.961e-06 |
| reduce_prod_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.062 | 0.002623 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.060 | 4.216e-06 |
| reduce_max_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.116 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.663 | 0 |
| reduce_min_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.116 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.664 | 0 |
| reduce_sum_squares_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.085 | 2.947e-06 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.103 | 3.9e-09 |
| reduce_sum_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.589 | 0.0002666 |
| reduce_prod_full_f32_n1024 | pytorch-cuda | 256 | 15 | 2.649 | 0.0025 |
| reduce_max_full_f32_n1024 | pytorch-cuda | 256 | 15 | 2.939 | 0 |
| reduce_min_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.020 | 0 |
| compare_eq_f32_n1024 | pytorch-cuda | 256 | 15 | 2.243 | 0 |
| compare_lt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.244 | 0 |
| compare_le_f32_n1024 | pytorch-cuda | 256 | 15 | 2.202 | 0 |
| compare_ge_f32_n1024 | pytorch-cuda | 256 | 15 | 2.222 | 0 |
| reduce_sum_full_f64_n1024 | pytorch-cuda | 256 | 15 | 3.031 | 6.208e-07 |
| reduce_prod_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.559 | 1.663e-06 |
| reduce_max_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.841 | 0 |
| reduce_min_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.938 | 0 |
| compare_eq_f64_n1024 | pytorch-cuda | 256 | 15 | 2.245 | 0 |
| compare_lt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.256 | 0 |
| compare_le_f64_n1024 | pytorch-cuda | 256 | 15 | 2.301 | 0 |
| compare_ge_f64_n1024 | pytorch-cuda | 256 | 15 | 2.261 | 0 |
| reduce_sum_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.360 | 0.00522 |
| reduce_prod_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.355 | 0.5197 |
| reduce_max_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.395 | 0 |
| reduce_min_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.411 | 0 |
| compare_eq_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.874 | 0 |
| compare_lt_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.874 | 0 |
| compare_le_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.873 | 0 |
| compare_ge_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.874 | 0 |
| reduce_sum_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.392 | 0.0004425 |
| reduce_prod_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.398 | 0.0001269 |
| reduce_max_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.852 | 0 |
| reduce_min_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.854 | 0 |
| compare_eq_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.627 | 0 |
| compare_lt_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.581 | 0 |
| compare_le_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.582 | 0 |
| compare_ge_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.581 | 0 |
| reduce_mean_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.258 | 0.0002906 |
| reduce_sum_squares_full_f32_n1024 | pytorch-cuda | 128 | 15 | 1.895 | 0 |
| reduce_mean_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.991 | 8.614e-07 |
| reduce_sum_squares_full_f64_n1024 | pytorch-cuda | 256 | 15 | 3.679 | 0 |
| reduce_mean_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.359 | 0.00522 |
| reduce_sum_squares_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.254 | 0 |
| reduce_mean_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.395 | 0.001223 |
| reduce_sum_squares_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.124 | 1.664e-07 |
| scale_real_f32_n1024 | pytorch-cuda | 256 | 15 | 2.533 | 0 |
| scale_real_f64_n1024 | pytorch-cuda | 256 | 15 | 2.518 | 0 |
| scale_real_c32_n1024 | pytorch-cuda | 256 | 15 | 2.544 | 0 |
| scale_real_c64_n1024 | pytorch-cuda | 256 | 15 | 2.585 | 0 |
| scale_complex_c32_n1024 | pytorch-cuda | 256 | 15 | 2.538 | 4.666e-05 |
| scale_complex_c64_n1024 | pytorch-cuda | 256 | 15 | 2.694 | 0 |
| scale_real_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.800 | 0 |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.718 | 0 |
| scale_real_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.717 | 0 |
| scale_real_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.676 | 0 |
| scale_complex_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.718 | 4.516e-05 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.676 | 0 |
| add_i32_n1024 | pytorch-cuda | 256 | 15 | 2.387 | 0 |
| sub_i32_n1024 | pytorch-cuda | 256 | 15 | 2.390 | 0 |
| mul_i32_n1024 | pytorch-cuda | 256 | 15 | 2.441 | 0 |
| div_i32_n1024 | pytorch-cuda | 256 | 15 | 2.582 | 0 |
| rem_i32_n1024 | pytorch-cuda | 256 | 15 | 2.397 | 0 |
| maximum_i32_n1024 | pytorch-cuda | 256 | 15 | 2.375 | 0 |
| minimum_i32_n1024 | pytorch-cuda | 256 | 15 | 2.406 | 0 |
| neg_i32_n1024 | pytorch-cuda | 256 | 15 | 2.227 | 0 |
| abs_i32_n1024 | pytorch-cuda | 256 | 15 | 3.009 | 0 |
| sign_i32_n1024 | pytorch-cuda | 256 | 15 | 2.321 | 0 |
| reduce_sum_i32_n1024 | pytorch-cuda | 128 | 15 | 2.043 | 0 |
| reduce_max_i32_n1024 | pytorch-cuda | 128 | 15 | 1.788 | 0 |
| reduce_min_i32_n1024 | pytorch-cuda | 256 | 15 | 3.472 | 0 |
| add_i64_n1024 | pytorch-cuda | 256 | 15 | 2.474 | 0 |
| sub_i64_n1024 | pytorch-cuda | 256 | 15 | 2.536 | 0 |
| mul_i64_n1024 | pytorch-cuda | 256 | 15 | 2.335 | 0 |
| div_i64_n1024 | pytorch-cuda | 256 | 15 | 2.574 | 0 |
| rem_i64_n1024 | pytorch-cuda | 256 | 15 | 2.537 | 0 |
| maximum_i64_n1024 | pytorch-cuda | 256 | 15 | 2.511 | 0 |
| minimum_i64_n1024 | pytorch-cuda | 256 | 15 | 2.382 | 0 |
| neg_i64_n1024 | pytorch-cuda | 256 | 15 | 2.234 | 0 |
| abs_i64_n1024 | pytorch-cuda | 256 | 15 | 2.910 | 0 |
| sign_i64_n1024 | pytorch-cuda | 256 | 15 | 2.328 | 0 |
| reduce_sum_i64_n1024 | pytorch-cuda | 128 | 15 | 2.017 | 0 |
| reduce_max_i64_n1024 | pytorch-cuda | 256 | 15 | 3.349 | 0 |
| reduce_min_i64_n1024 | pytorch-cuda | 256 | 15 | 3.489 | 0 |
| add_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.550 | 0 |
| sub_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.549 | 0 |
| mul_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.555 | 0 |
| div_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.557 | 0 |
| rem_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.562 | 0 |
| maximum_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.543 | 0 |
| minimum_i32_n1048576 | pytorch-cuda | 64 | 15 | 2.545 | 0 |
| neg_i32_n1048576 | pytorch-cuda | 64 | 15 | 1.750 | 0 |
| abs_i32_n1048576 | pytorch-cuda | 64 | 15 | 1.748 | 0 |
| sign_i32_n1048576 | pytorch-cuda | 64 | 15 | 1.750 | 0 |
| reduce_sum_i32_n1048576 | pytorch-cuda | 128 | 15 | 2.063 | 0 |
| reduce_max_i32_n1048576 | pytorch-cuda | 128 | 15 | 2.079 | 0 |
| reduce_min_i32_n1048576 | pytorch-cuda | 128 | 15 | 2.080 | 0 |
| add_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| sub_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.487 | 0 |
| mul_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| div_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.573 | 0 |
| rem_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.563 | 0 |
| maximum_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.483 | 0 |
| minimum_i64_n1048576 | pytorch-cuda | 32 | 15 | 2.483 | 0 |
| neg_i64_n1048576 | pytorch-cuda | 32 | 15 | 1.694 | 0 |
| abs_i64_n1048576 | pytorch-cuda | 32 | 15 | 1.699 | 0 |
| sign_i64_n1048576 | pytorch-cuda | 32 | 15 | 1.695 | 0 |
| reduce_sum_i64_n1048576 | pytorch-cuda | 128 | 15 | 3.674 | 0 |
| reduce_max_i64_n1048576 | pytorch-cuda | 128 | 15 | 3.688 | 0 |
| reduce_min_i64_n1048576 | pytorch-cuda | 128 | 15 | 3.689 | 0 |
| add_f32_n1024 | jax-cuda | 32 | 15 | 2.819 | 0 |
| add_f64_n1024 | jax-cuda | 32 | 15 | 3.051 | 0 |
| sub_f32_n1024 | jax-cuda | 32 | 15 | 2.935 | 0 |
| sub_f64_n1024 | jax-cuda | 32 | 15 | 2.903 | 0 |
| mul_f32_n1024 | jax-cuda | 32 | 15 | 2.890 | 0 |
| mul_f64_n1024 | jax-cuda | 32 | 15 | 2.913 | 0 |
| div_f32_n1024 | jax-cuda | 32 | 15 | 2.714 | 4.875e-05 |
| div_f64_n1024 | jax-cuda | 32 | 15 | 2.667 | 0 |
| rem_f32_n1024 | jax-cuda | 32 | 15 | 3.114 | 0 |
| rem_f64_n1024 | jax-cuda | 32 | 15 | 3.481 | 0 |
| pow_f32_n1024 | jax-cuda | 16 | 15 | 2.706 | 0.000117 |
| pow_f64_n1024 | jax-cuda | 16 | 15 | 3.208 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda | 16 | 15 | 2.434 | 0 |
| maximum_f64_n1024 | jax-cuda | 16 | 15 | 2.358 | 0 |
| minimum_f32_n1024 | jax-cuda | 16 | 15 | 2.296 | 0 |
| minimum_f64_n1024 | jax-cuda | 16 | 15 | 2.265 | 0 |
| neg_f32_n1024 | jax-cuda | 16 | 15 | 2.193 | 0 |
| neg_f64_n1024 | jax-cuda | 32 | 15 | 3.676 | 0 |
| abs_f32_n1024 | jax-cuda | 32 | 15 | 3.420 | 0 |
| abs_f64_n1024 | jax-cuda | 32 | 15 | 3.296 | 0 |
| sign_f32_n1024 | jax-cuda | 32 | 15 | 3.439 | 0 |
| sign_f64_n1024 | jax-cuda | 32 | 15 | 3.301 | 0 |
| conj_c32_n1024 | jax-cuda | 32 | 15 | 3.455 | 0 |
| conj_c64_n1024 | jax-cuda | 32 | 15 | 3.340 | 0 |
| exp_f32_n1024 | jax-cuda | 32 | 15 | 3.027 | 0.0001272 |
| exp_f64_n1024 | jax-cuda | 32 | 15 | 3.162 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda | 32 | 15 | 2.968 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda | 32 | 15 | 3.278 | 7.39e-08 |
| log_f32_n1024 | jax-cuda | 32 | 15 | 3.292 | 7.945e-05 |
| log_f64_n1024 | jax-cuda | 32 | 15 | 3.423 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda | 32 | 15 | 3.282 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda | 16 | 15 | 2.109 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda | 16 | 15 | 3.481 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda | 16 | 15 | 3.214 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda | 16 | 15 | 2.207 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda | 16 | 15 | 2.580 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda | 16 | 15 | 2.690 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda | 16 | 15 | 2.393 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda | 16 | 15 | 2.340 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda | 16 | 15 | 2.573 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda | 16 | 15 | 2.260 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda | 16 | 15 | 2.369 | 0 |
| rsqrt_f32_n1024 | jax-cuda | 16 | 15 | 3.037 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda | 16 | 15 | 2.922 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda | 32 | 15 | 3.311 | 0 |
| compare_f64_n1024 | jax-cuda | 32 | 15 | 3.021 | 0 |
| where_select_f32_n1024 | jax-cuda | 32 | 15 | 3.376 | 0 |
| where_select_f64_n1024 | jax-cuda | 32 | 15 | 3.489 | 0 |
| clamp_f32_n1024 | jax-cuda | 32 | 15 | 3.193 | 0 |
| clamp_f64_n1024 | jax-cuda | 32 | 15 | 3.230 | 0 |
| reduce_sum_f32_n1024 | jax-cuda | 32 | 15 | 3.121 | 0.0001139 |
| reduce_sum_f64_n1024 | jax-cuda | 32 | 15 | 3.188 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda | 32 | 15 | 3.274 | 0.0003574 |
| reduce_prod_f64_n1024 | jax-cuda | 16 | 15 | 2.476 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda | 16 | 15 | 2.011 | 0 |
| reduce_max_f64_n1024 | jax-cuda | 32 | 15 | 3.523 | 0 |
| reduce_min_f32_n1024 | jax-cuda | 32 | 15 | 3.500 | 0 |
| reduce_min_f64_n1024 | jax-cuda | 32 | 15 | 3.112 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda | 32 | 15 | 3.285 | 7.204e-06 |
| reduce_mean_f64_n1024 | jax-cuda | 32 | 15 | 3.300 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda | 16 | 15 | 1.738 | 0 |
| add_f64_n1048576 | jax-cuda | 16 | 15 | 1.555 | 0 |
| sub_f32_n1048576 | jax-cuda | 32 | 15 | 2.648 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 2.637 | 0 |
| mul_f32_n1048576 | jax-cuda | 32 | 15 | 2.672 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 2.640 | 0 |
| div_f32_n1048576 | jax-cuda | 32 | 15 | 2.268 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda | 32 | 15 | 3.938 | 0 |
| rem_f32_n1048576 | jax-cuda | 32 | 15 | 2.576 | 0 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 2.660 | 0 |
| pow_f32_n1048576 | jax-cuda | 16 | 15 | 1.938 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda | 2 | 15 | 2.622 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda | 16 | 15 | 1.918 | 0 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 3.325 | 0 |
| minimum_f32_n1048576 | jax-cuda | 32 | 15 | 3.720 | 0 |
| minimum_f64_n1048576 | jax-cuda | 16 | 15 | 2.056 | 0 |
| neg_f32_n1048576 | jax-cuda | 32 | 15 | 3.581 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 3.523 | 0 |
| abs_f32_n1048576 | jax-cuda | 32 | 15 | 3.195 | 0 |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 3.287 | 0 |
| sign_f32_n1048576 | jax-cuda | 32 | 15 | 3.314 | 0 |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 3.184 | 0 |
| conj_c32_n1048576 | jax-cuda | 32 | 15 | 2.365 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 1.811 | 0 |
| exp_f32_n1048576 | jax-cuda | 32 | 15 | 2.422 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda | 16 | 15 | 3.182 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda | 32 | 15 | 2.965 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda | 8 | 15 | 2.030 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda | 32 | 15 | 3.267 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda | 8 | 15 | 3.237 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda | 32 | 15 | 3.096 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda | 8 | 15 | 3.337 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda | 16 | 15 | 2.724 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda | 4 | 15 | 2.764 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda | 16 | 15 | 2.430 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda | 16 | 15 | 3.812 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda | 16 | 15 | 2.087 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda | 16 | 15 | 3.802 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda | 32 | 15 | 3.570 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda | 16 | 15 | 3.055 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda | 32 | 15 | 3.312 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 3.311 | 0 |
| rsqrt_f32_n1048576 | jax-cuda | 16 | 15 | 2.417 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda | 16 | 15 | 2.232 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda | 32 | 15 | 3.029 | 0 |
| compare_f64_n1048576 | jax-cuda | 32 | 15 | 3.030 | 0 |
| where_select_f32_n1048576 | jax-cuda | 32 | 15 | 3.861 | 0 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 3.751 | 0 |
| clamp_f32_n1048576 | jax-cuda | 32 | 15 | 3.047 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 2.974 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda | 16 | 15 | 2.649 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda | 16 | 15 | 2.957 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda | 16 | 15 | 2.538 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda | 16 | 15 | 2.512 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda | 16 | 15 | 2.428 | 0 |
| reduce_max_f64_n1048576 | jax-cuda | 16 | 15 | 2.423 | 0 |
| reduce_min_f32_n1048576 | jax-cuda | 16 | 15 | 2.528 | 0 |
| reduce_min_f64_n1048576 | jax-cuda | 16 | 15 | 2.639 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda | 16 | 15 | 2.997 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda | 16 | 15 | 2.905 | 6.5e-09 |
| reduce_sum_full_f32_n1024 | jax-cuda | 16 | 15 | 2.719 | 0 |
| reduce_prod_full_f32_n1024 | jax-cuda | 16 | 15 | 2.047 | 0.001786 |
| reduce_max_full_f32_n1024 | jax-cuda | 16 | 15 | 2.053 | 0 |
| reduce_min_full_f32_n1024 | jax-cuda | 32 | 15 | 3.020 | 0 |
| compare_eq_f32_n1024 | jax-cuda | 32 | 15 | 3.016 | 0 |
| compare_lt_f32_n1024 | jax-cuda | 32 | 15 | 3.009 | 0 |
| compare_le_f32_n1024 | jax-cuda | 32 | 15 | 2.887 | 0 |
| compare_ge_f32_n1024 | jax-cuda | 32 | 15 | 3.107 | 0 |
| reduce_sum_full_f64_n1024 | jax-cuda | 16 | 15 | 2.562 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | jax-cuda | 16 | 15 | 2.282 | 1.774e-06 |
| reduce_max_full_f64_n1024 | jax-cuda | 16 | 15 | 2.529 | 0 |
| reduce_min_full_f64_n1024 | jax-cuda | 16 | 15 | 2.655 | 0 |
| compare_eq_f64_n1024 | jax-cuda | 16 | 15 | 2.573 | 0 |
| compare_lt_f64_n1024 | jax-cuda | 16 | 15 | 2.244 | 0 |
| compare_le_f64_n1024 | jax-cuda | 16 | 15 | 1.956 | 0 |
| compare_ge_f64_n1024 | jax-cuda | 32 | 15 | 3.049 | 0 |
| reduce_sum_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.746 | 0.003668 |
| reduce_prod_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.646 | 0.5323 |
| reduce_max_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.343 | 0 |
| reduce_min_full_f32_n1048576 | jax-cuda | 32 | 15 | 3.377 | 0 |
| compare_eq_f32_n1048576 | jax-cuda | 32 | 15 | 3.005 | 0 |
| compare_lt_f32_n1048576 | jax-cuda | 32 | 15 | 3.028 | 0 |
| compare_le_f32_n1048576 | jax-cuda | 32 | 15 | 3.034 | 0 |
| compare_ge_f32_n1048576 | jax-cuda | 32 | 15 | 3.049 | 0 |
| reduce_sum_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.281 | 0.0006452 |
| reduce_prod_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.406 | 0.0001105 |
| reduce_max_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.604 | 0 |
| reduce_min_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.694 | 0 |
| compare_eq_f64_n1048576 | jax-cuda | 16 | 15 | 2.142 | 0 |
| compare_lt_f64_n1048576 | jax-cuda | 32 | 15 | 2.842 | 0 |
| compare_le_f64_n1048576 | jax-cuda | 32 | 15 | 2.935 | 0 |
| compare_ge_f64_n1048576 | jax-cuda | 32 | 15 | 2.962 | 0 |
| reduce_mean_full_f32_n1024 | jax-cuda | 32 | 15 | 3.144 | 0 |
| reduce_sum_squares_full_f32_n1024 | jax-cuda | 32 | 15 | 3.028 | 0 |
| reduce_mean_full_f64_n1024 | jax-cuda | 16 | 15 | 2.227 | 5.169e-07 |
| reduce_sum_squares_full_f64_n1024 | jax-cuda | 16 | 15 | 2.609 | 0 |
| reduce_mean_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.844 | 0.003668 |
| reduce_sum_squares_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.640 | 0 |
| reduce_mean_full_f64_n1048576 | jax-cuda | 16 | 15 | 3.043 | 0.001783 |
| reduce_sum_squares_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.632 | 0 |
| scale_real_f32_n1024 | jax-cuda | 16 | 15 | 2.658 | 0 |
| scale_real_f64_n1024 | jax-cuda | 16 | 15 | 2.389 | 0 |
| scale_real_c32_n1024 | jax-cuda | 16 | 15 | 2.183 | 0 |
| scale_real_c64_n1024 | jax-cuda | 16 | 15 | 2.240 | 0 |
| scale_complex_c32_n1024 | jax-cuda | 16 | 15 | 1.927 | 7.929e-05 |
| scale_complex_c64_n1024 | jax-cuda | 32 | 15 | 3.319 | 3.753e-08 |
| scale_real_f32_n1048576 | jax-cuda | 32 | 15 | 3.385 | 0 |
| scale_real_f64_n1048576 | jax-cuda | 32 | 15 | 3.417 | 0 |
| scale_real_c32_n1048576 | jax-cuda | 32 | 15 | 3.438 | 0 |
| scale_real_c64_n1048576 | jax-cuda | 16 | 15 | 1.886 | 0 |
| scale_complex_c32_n1048576 | jax-cuda | 32 | 15 | 3.467 | 6.186e-05 |
| scale_complex_c64_n1048576 | jax-cuda | 16 | 15 | 1.858 | 4.129e-08 |
| add_i32_n1024 | jax-cuda | 32 | 15 | 3.306 | 0 |
| sub_i32_n1024 | jax-cuda | 32 | 15 | 3.292 | 0 |
| mul_i32_n1024 | jax-cuda | 32 | 15 | 3.401 | 0 |
| div_i32_n1024 | jax-cuda | 16 | 15 | 2.507 | 0 |
| rem_i32_n1024 | jax-cuda | 32 | 15 | 3.230 | 0 |
| maximum_i32_n1024 | jax-cuda | 32 | 15 | 3.537 | 0 |
| minimum_i32_n1024 | jax-cuda | 32 | 15 | 3.370 | 0 |
| neg_i32_n1024 | jax-cuda | 32 | 15 | 3.303 | 0 |
| abs_i32_n1024 | jax-cuda | 32 | 15 | 3.011 | 0 |
| sign_i32_n1024 | jax-cuda | 32 | 15 | 2.980 | 0 |
| reduce_sum_i32_n1024 | jax-cuda | 32 | 15 | 3.137 | 0 |
| reduce_max_i32_n1024 | jax-cuda | 32 | 15 | 2.982 | 0 |
| reduce_min_i32_n1024 | jax-cuda | 32 | 15 | 3.117 | 0 |
| add_i64_n1024 | jax-cuda | 32 | 15 | 3.448 | 0 |
| sub_i64_n1024 | jax-cuda | 32 | 15 | 3.472 | 0 |
| mul_i64_n1024 | jax-cuda | 32 | 15 | 3.379 | 0 |
| div_i64_n1024 | jax-cuda | 16 | 15 | 2.637 | 0 |
| rem_i64_n1024 | jax-cuda | 32 | 15 | 3.225 | 0 |
| maximum_i64_n1024 | jax-cuda | 32 | 15 | 3.417 | 0 |
| minimum_i64_n1024 | jax-cuda | 32 | 15 | 3.463 | 0 |
| neg_i64_n1024 | jax-cuda | 32 | 15 | 3.263 | 0 |
| abs_i64_n1024 | jax-cuda | 32 | 15 | 2.882 | 0 |
| sign_i64_n1024 | jax-cuda | 32 | 15 | 2.954 | 0 |
| reduce_sum_i64_n1024 | jax-cuda | 16 | 15 | 2.073 | 0 |
| reduce_max_i64_n1024 | jax-cuda | 16 | 15 | 2.241 | 0 |
| reduce_min_i64_n1024 | jax-cuda | 16 | 15 | 2.242 | 0 |
| add_i32_n1048576 | jax-cuda | 16 | 15 | 2.435 | 0 |
| sub_i32_n1048576 | jax-cuda | 16 | 15 | 2.023 | 0 |
| mul_i32_n1048576 | jax-cuda | 16 | 15 | 1.999 | 0 |
| div_i32_n1048576 | jax-cuda | 16 | 15 | 2.492 | 0 |
| rem_i32_n1048576 | jax-cuda | 16 | 15 | 2.002 | 0 |
| maximum_i32_n1048576 | jax-cuda | 16 | 15 | 2.289 | 0 |
| minimum_i32_n1048576 | jax-cuda | 16 | 15 | 2.183 | 0 |
| neg_i32_n1048576 | jax-cuda | 16 | 15 | 2.117 | 0 |
| abs_i32_n1048576 | jax-cuda | 16 | 15 | 1.866 | 0 |
| sign_i32_n1048576 | jax-cuda | 16 | 15 | 1.960 | 0 |
| reduce_sum_i32_n1048576 | jax-cuda | 32 | 15 | 3.018 | 0 |
| reduce_max_i32_n1048576 | jax-cuda | 32 | 15 | 2.864 | 0 |
| reduce_min_i32_n1048576 | jax-cuda | 32 | 15 | 2.954 | 0 |
| add_i64_n1048576 | jax-cuda | 16 | 15 | 2.267 | 0 |
| sub_i64_n1048576 | jax-cuda | 16 | 15 | 2.332 | 0 |
| mul_i64_n1048576 | jax-cuda | 16 | 15 | 2.450 | 0 |
| div_i64_n1048576 | jax-cuda | 16 | 15 | 3.044 | 0 |
| rem_i64_n1048576 | jax-cuda | 16 | 15 | 2.662 | 0 |
| maximum_i64_n1048576 | jax-cuda | 16 | 15 | 2.393 | 0 |
| minimum_i64_n1048576 | jax-cuda | 16 | 15 | 2.327 | 0 |
| neg_i64_n1048576 | jax-cuda | 16 | 15 | 2.059 | 0 |
| abs_i64_n1048576 | jax-cuda | 16 | 15 | 1.855 | 0 |
| sign_i64_n1048576 | jax-cuda | 16 | 15 | 1.931 | 0 |
| reduce_sum_i64_n1048576 | jax-cuda | 16 | 15 | 2.692 | 0 |
| reduce_max_i64_n1048576 | jax-cuda | 16 | 15 | 2.511 | 0 |
| reduce_min_i64_n1048576 | jax-cuda | 16 | 15 | 2.523 | 0 |
| add_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.249 | 0 |
| add_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.161 | 0 |
| sub_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.768 | 0 |
| sub_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.539 | 0 |
| mul_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.558 | 0 |
| mul_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.483 | 0 |
| div_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.555 | 4.875e-05 |
| div_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.602 | 0 |
| rem_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.611 | 0 |
| rem_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.068 | 0 |
| pow_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.240 | 0.000117 |
| pow_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.491 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.098 | 0 |
| maximum_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.785 | 0 |
| minimum_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.645 | 0 |
| minimum_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.547 | 0 |
| neg_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.487 | 0 |
| neg_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.495 | 0 |
| abs_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.526 | 0 |
| abs_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.565 | 0 |
| sign_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.648 | 0 |
| sign_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.581 | 0 |
| conj_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.595 | 0 |
| conj_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.505 | 0 |
| exp_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.542 | 0.0001272 |
| exp_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.660 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.710 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.691 | 7.39e-08 |
| log_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.666 | 7.945e-05 |
| log_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.943 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.890 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.201 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.142 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.059 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.920 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.465 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.157 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.522 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.393 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.297 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.272 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.095 | 0 |
| rsqrt_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.095 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.130 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.993 | 0 |
| compare_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.370 | 0 |
| where_select_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.377 | 0 |
| where_select_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.923 | 0 |
| clamp_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.999 | 0 |
| clamp_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.066 | 0 |
| reduce_sum_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.504 | 0.0002742 |
| reduce_sum_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.680 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.772 | 0.0003579 |
| reduce_prod_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.893 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.448 | 0 |
| reduce_max_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.054 | 0 |
| reduce_min_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.005 | 0 |
| reduce_min_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.017 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.358 | 1.439e-05 |
| reduce_mean_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.005 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.650 | 0 |
| add_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.582 | 0 |
| sub_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.280 | 0 |
| sub_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.136 | 0 |
| mul_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.724 | 0 |
| mul_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.659 | 0 |
| div_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.526 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.927 | 0 |
| rem_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.739 | 0 |
| rem_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.796 | 0 |
| pow_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.627 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda-jit | 2 | 15 | 2.517 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.938 | 0 |
| maximum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.674 | 0 |
| minimum_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.543 | 0 |
| minimum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.677 | 0 |
| neg_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.394 | 0 |
| neg_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.234 | 0 |
| abs_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.377 | 0 |
| abs_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.302 | 0 |
| sign_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.378 | 0 |
| sign_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.409 | 0 |
| conj_c32_n1048576 | jax-cuda-jit | 32 | 15 | 2.199 | 0 |
| conj_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.866 | 0 |
| exp_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.359 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.248 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.835 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.128 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.716 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.188 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.262 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.379 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.251 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda-jit | 4 | 15 | 2.636 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.180 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.836 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.063 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.827 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.903 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.063 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.203 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.280 | 0 |
| rsqrt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.169 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.038 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.260 | 0 |
| compare_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.366 | 0 |
| where_select_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.313 | 0 |
| where_select_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.131 | 0 |
| clamp_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.394 | 0 |
| clamp_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.170 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.414 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.546 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.501 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.508 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.490 | 0 |
| reduce_max_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.527 | 0 |
| reduce_min_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.810 | 0 |
| reduce_min_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.587 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.426 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.651 | 6.5e-09 |
| reduce_sum_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.740 | 0 |
| reduce_prod_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.239 | 0.001786 |
| reduce_max_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.759 | 0 |
| reduce_min_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.795 | 0 |
| compare_eq_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.363 | 0 |
| compare_lt_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.123 | 0 |
| compare_le_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.425 | 0 |
| compare_ge_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.229 | 0 |
| reduce_sum_full_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.642 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.491 | 1.774e-06 |
| reduce_max_full_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.204 | 0 |
| reduce_min_full_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.668 | 0 |
| compare_eq_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.435 | 0 |
| compare_lt_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.183 | 0 |
| compare_le_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.047 | 0 |
| compare_ge_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.293 | 0 |
| reduce_sum_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.659 | 0.003668 |
| reduce_prod_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.194 | 0.5323 |
| reduce_max_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.529 | 0 |
| reduce_min_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.685 | 0 |
| compare_eq_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.987 | 0 |
| compare_lt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.096 | 0 |
| compare_le_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.090 | 0 |
| compare_ge_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.110 | 0 |
| reduce_sum_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.659 | 0.0006452 |
| reduce_prod_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.526 | 0.0001105 |
| reduce_max_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.661 | 0 |
| reduce_min_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.749 | 0 |
| compare_eq_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.902 | 0 |
| compare_lt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.230 | 0 |
| compare_le_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.071 | 0 |
| compare_ge_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.165 | 0 |
| reduce_mean_full_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.274 | 0 |
| reduce_sum_squares_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.505 | 0.0001766 |
| reduce_mean_full_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.737 | 5.169e-07 |
| reduce_sum_squares_full_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.632 | 0 |
| reduce_mean_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.667 | 0.003668 |
| reduce_sum_squares_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.558 | 0 |
| reduce_mean_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.750 | 0.001783 |
| reduce_sum_squares_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.605 | 0 |
| scale_real_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.311 | 0 |
| scale_real_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.098 | 0 |
| scale_real_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.048 | 0 |
| scale_real_c64_n1024 | jax-cuda-jit | 32 | 15 | 3.225 | 0 |
| scale_complex_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.189 | 7.929e-05 |
| scale_complex_c64_n1024 | jax-cuda-jit | 32 | 15 | 3.101 | 3.753e-08 |
| scale_real_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.125 | 0 |
| scale_real_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.149 | 0 |
| scale_real_c32_n1048576 | jax-cuda-jit | 32 | 15 | 3.057 | 0 |
| scale_real_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.891 | 0 |
| scale_complex_c32_n1048576 | jax-cuda-jit | 32 | 15 | 3.109 | 6.186e-05 |
| scale_complex_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.875 | 4.129e-08 |
| add_i32_n1024 | jax-cuda-jit | 32 | 15 | 3.155 | 0 |
| sub_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.752 | 0 |
| mul_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.619 | 0 |
| div_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.630 | 0 |
| rem_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.674 | 0 |
| maximum_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.592 | 0 |
| minimum_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.561 | 0 |
| neg_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.487 | 0 |
| abs_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.518 | 0 |
| sign_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.551 | 0 |
| reduce_sum_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.237 | 0 |
| reduce_max_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.383 | 0 |
| reduce_min_i32_n1024 | jax-cuda-jit | 32 | 15 | 2.360 | 0 |
| add_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.481 | 0 |
| sub_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.546 | 0 |
| mul_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.508 | 0 |
| div_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.664 | 0 |
| rem_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.774 | 0 |
| maximum_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.582 | 0 |
| minimum_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.436 | 0 |
| neg_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.339 | 0 |
| abs_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.327 | 0 |
| sign_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.310 | 0 |
| reduce_sum_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.195 | 0 |
| reduce_max_i64_n1024 | jax-cuda-jit | 32 | 15 | 2.193 | 0 |
| reduce_min_i64_n1024 | jax-cuda-jit | 32 | 15 | 3.318 | 0 |
| add_i32_n1048576 | jax-cuda-jit | 32 | 15 | 3.008 | 0 |
| sub_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.455 | 0 |
| mul_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.337 | 0 |
| div_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.438 | 0 |
| rem_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.931 | 0 |
| maximum_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.735 | 0 |
| minimum_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.660 | 0 |
| neg_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.507 | 0 |
| abs_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.520 | 0 |
| sign_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.547 | 0 |
| reduce_sum_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.489 | 0 |
| reduce_max_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.271 | 0 |
| reduce_min_i32_n1048576 | jax-cuda-jit | 32 | 15 | 2.263 | 0 |
| add_i64_n1048576 | jax-cuda-jit | 32 | 15 | 2.645 | 0 |
| sub_i64_n1048576 | jax-cuda-jit | 32 | 15 | 2.647 | 0 |
| mul_i64_n1048576 | jax-cuda-jit | 32 | 15 | 2.649 | 0 |
| div_i64_n1048576 | jax-cuda-jit | 32 | 15 | 2.871 | 0 |
| rem_i64_n1048576 | jax-cuda-jit | 32 | 15 | 3.711 | 0 |
| maximum_i64_n1048576 | jax-cuda-jit | 32 | 15 | 2.935 | 0 |
| minimum_i64_n1048576 | jax-cuda-jit | 32 | 15 | 2.786 | 0 |
| neg_i64_n1048576 | jax-cuda-jit | 32 | 15 | 2.505 | 0 |
| abs_i64_n1048576 | jax-cuda-jit | 32 | 15 | 2.423 | 0 |
| sign_i64_n1048576 | jax-cuda-jit | 32 | 15 | 2.535 | 0 |
| reduce_sum_i64_n1048576 | jax-cuda-jit | 16 | 15 | 2.317 | 0 |
| reduce_max_i64_n1048576 | jax-cuda-jit | 16 | 15 | 2.537 | 0 |
| reduce_min_i64_n1048576 | jax-cuda-jit | 16 | 15 | 2.534 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: benchmarks/gpu/isolated_ops.yaml
timestamp: '2026-10-13T20:48:50Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: fbd735abbfd5bcf0453f151472b274a71249f115
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 8fd6c139d57c50ed5856819c79c656fabaacd765
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
  expected_cases: 228
  selected_cases: 228
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 8e6162a34710e411c12fb248e8995579ef2300fe8cf744850dec06805ebc12fa
    src/bin/benchmark_gpu_public_api/plan.rs: 3d0f10f390ebd0e08a40deb5a64e49e086dcd41b9a777d3a208c6846ed6b2fac
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: da168bf8b7c30f748f62839cf0bf7cce8f5bccf9c37b1c1d993fd70c466f2f76
    src/bin/benchmark_gpu_public_api/multi.rs: 1b477fef78260cf28aa31e515d57ab411f6241b91ffb30d625939538fdbc6bab
    src/bin/benchmark_gpu_public_api/lifetime.rs: c687a8b0329f6d3073073671b5d246acbb1f4dc3f983d1430da897a908ef1745
    src/bin/benchmark_gpu_public_api/read_layout.rs: 70b1ed632ad52db5654c3d9f3df9fdb20a1e8c48894dbdcb24ce055757f5f7eb
    src/bin/benchmark_gpu_public_api/structural.rs: b8fd99ceb5c80fd3474974b0ba9aa57833ad6bd25383af3091c11c0b6ab94f6d
    src/bin/benchmark_gpu_public_api/linalg.rs: 5f5c4afc161f3d6e267861511d0d5871be9192d9968c95b59a73c9dc5a64e22b
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: 1dc14ac94fac700aa696d0ada04315685e3245738f9514de3d7989b8cb59152b
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 23ea8b8eba83128c689e8c09385f2a4b9485ddf94470c4aa74005130dfa2d0ae
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: df4ae210f2cb3fb7e90df7a2cb5f69c71a7c2f87338b56972ac5f23ff67ecddb
    scripts/run_gpu_public_api.sh: dd5e4cc47e85fcb5afdc56d460136b2a399a6a7d6a62005de8ba94afb0119ecc
    data/results/nvidia-gpu/gpu/isolated_ops/20261013_204849/fixtures/manifest.json: b477c4a11634759024723d4cd13d3a4490980e3e97ff932845296400cca874db
    target/release/benchmark_gpu_public_api: 4d689fb71cc34615be8a0e28f4a32a31325a3c5a71b30747036c1a91ddeb4aee
  reference_numpy_version: 2.4.2
```
