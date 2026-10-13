# gpu/isolated_ops: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261013_133737/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_f32_n1024 | 14.966 (1.726) | 22.359 (1.477) | 8.468 (0.562) | 103.399 (3.444) | 84.963 (3.922) |
| add_f64_n1024 | 16.335 (1.122) | 27.275 (2.060) | 8.733 (0.319) | 92.551 (20.563) | 83.325 (11.141) |
| sub_f32_n1024 | 16.177 (1.331) | 29.346 (2.317) | 8.845 (0.116) | 86.361 (15.572) | 81.383 (8.488) |
| sub_f64_n1024 | 17.358 (1.121) | 30.684 (2.409) | 8.757 (0.097) | 84.711 (12.310) | 81.862 (10.251) |
| mul_f32_n1024 | 17.372 (0.505) | 32.431 (1.038) | 9.102 (0.138) | 81.058 (4.628) | 81.990 (6.255) |
| mul_f64_n1024 | 17.647 (0.154) | 33.478 (0.566) | 8.963 (0.070) | 79.674 (9.333) | 80.822 (9.050) |
| div_f32_n1024 | 19.201 (0.462) | 34.016 (1.016) | 8.676 (0.103) | 74.715 (6.460) | 80.271 (5.754) |
| div_f64_n1024 | 17.286 (0.154) | 34.121 (0.905) | 8.393 (0.093) | 81.345 (10.232) | 80.699 (2.267) |
| rem_f32_n1024 | 16.822 (0.213) | 212.454 (9.481) | 8.455 (0.163) | 91.529 (8.202) | 97.338 (8.111) |
| rem_f64_n1024 | 16.971 (0.155) | 247.256 (20.262) | 8.539 (0.276) | 100.261 (15.448) | 128.011 (7.289) |
| pow_f32_n1024 | 16.807 (0.168) | 209.665 (18.294) | 8.922 (0.101) | 140.203 (5.188) | 151.230 (23.143) |
| pow_f64_n1024 | 17.708 (0.206) | 252.411 (20.778) | 30.902 (0.009) | 185.617 (35.780) | 170.476 (20.420) |
| maximum_f32_n1024 | 16.649 (0.287) | 303.036 (10.149) | 8.591 (0.166) | 140.365 (27.833) | 146.604 (24.809) |
| maximum_f64_n1024 | 16.822 (0.140) | 234.988 (17.189) | 8.576 (0.158) | 135.951 (27.532) | 135.873 (16.936) |
| minimum_f32_n1024 | 16.701 (0.149) | 277.679 (8.419) | 8.646 (0.070) | 133.705 (15.870) | 128.343 (8.873) |
| minimum_f64_n1024 | 16.767 (0.184) | 290.384 (14.378) | 8.498 (0.138) | 130.598 (10.382) | 104.952 (14.054) |
| neg_f32_n1024 | 12.994 (1.905) | 20.528 (0.617) | 8.209 (0.092) | 117.612 (20.228) | 99.495 (10.018) |
| neg_f64_n1024 | 12.578 (1.467) | 21.046 (0.632) | 8.187 (0.180) | 123.634 (9.380) | 99.622 (8.089) |
| abs_f32_n1024 | 11.096 (1.374) | 154.653 (22.866) | 10.958 (0.691) | 95.946 (17.444) | 99.860 (7.187) |
| abs_f64_n1024 | 10.702 (1.429) | 149.518 (5.346) | 10.437 (0.288) | 88.070 (13.349) | 99.954 (6.640) |
| sign_f32_n1024 | 10.334 (1.211) | 170.937 (3.921) | 8.081 (0.065) | 97.970 (13.674) | 101.312 (8.847) |
| sign_f64_n1024 | 10.361 (1.021) | 173.581 (4.228) | 8.420 (0.351) | 95.629 (12.893) | 95.835 (10.210) |
| conj_c32_n1024 | 9.660 (0.897) | 176.032 (3.219) | 9.860 (0.077) | 90.346 (11.543) | 101.775 (11.031) |
| conj_c64_n1024 | 11.395 (0.763) | 146.285 (3.147) | 9.792 (0.139) | 91.252 (10.265) | 100.866 (11.269) |
| exp_f32_n1024 | 11.514 (1.262) | 20.985 (0.408) | 8.249 (0.082) | 93.922 (8.435) | 100.335 (7.365) |
| exp_f64_n1024 | 11.486 (1.373) | 21.957 (0.674) | 7.970 (0.090) | 96.763 (12.212) | 96.612 (1.516) |
| expm1_f32_n1024 | 10.196 (1.112) | 21.847 (0.628) | 8.758 (0.322) | 100.232 (12.076) | 96.138 (0.619) |
| expm1_f64_n1024 | 10.250 (1.094) | 21.477 (0.611) | 8.200 (0.170) | 97.838 (12.861) | 102.150 (8.307) |
| log_f32_n1024 | 10.056 (0.755) | 21.534 (0.851) | 7.979 (0.499) | 97.846 (20.222) | 102.243 (14.815) |
| log_f64_n1024 | 10.166 (1.047) | 24.081 (0.831) | 10.885 (0.009) | 101.940 (15.352) | 108.800 (18.690) |
| log1p_f32_n1024 | 9.256 (0.343) | 21.831 (0.725) | 8.275 (0.099) | 95.996 (14.099) | 103.186 (11.657) |
| log1p_f64_n1024 | 10.895 (1.358) | 21.572 (0.580) | 11.272 (0.004) | 105.770 (17.844) | 127.977 (3.966) |
| erf_f32_n1024 | 10.390 (1.422) | 21.793 (0.558) | 8.008 (0.085) | 192.697 (25.030) | 150.471 (15.877) |
| erf_f64_n1024 | 10.116 (1.069) | 20.824 (1.057) | 15.841 (0.007) | 210.944 (38.864) | 145.973 (39.302) |
| sin_f32_n1024 | 10.496 (1.510) | 20.621 (1.033) | 7.896 (0.067) | 140.518 (5.348) | 156.508 (22.567) |
| sin_f64_n1024 | 10.424 (1.387) | 23.542 (1.077) | 7.913 (0.144) | 153.654 (8.263) | 167.460 (21.245) |
| cos_f32_n1024 | 9.737 (1.120) | 22.178 (0.573) | 7.884 (0.062) | 160.603 (12.385) | 171.614 (25.813) |
| cos_f64_n1024 | 9.276 (0.507) | 22.083 (0.804) | 7.907 (0.120) | 157.644 (8.739) | 161.448 (18.325) |
| tanh_f32_n1024 | 8.870 (0.859) | 21.745 (0.668) | 7.948 (0.392) | 146.815 (19.166) | 166.273 (30.790) |
| tanh_f64_n1024 | 9.036 (0.994) | 21.838 (0.626) | 8.648 (0.253) | 151.508 (27.468) | 151.170 (24.911) |
| sqrt_f32_n1024 | 9.049 (0.687) | 21.348 (0.411) | 8.483 (0.332) | 142.004 (26.962) | 138.468 (19.781) |
| sqrt_f64_n1024 | 9.515 (0.809) | 21.249 (1.149) | 7.875 (0.565) | 137.568 (27.005) | 133.078 (8.668) |
| rsqrt_f32_n1024 | 8.733 (0.875) | 21.584 (1.120) | 8.201 (0.107) | 176.362 (48.182) | 106.369 (18.152) |
| rsqrt_f64_n1024 | 9.155 (0.920) | 22.055 (0.824) | 7.984 (0.084) | 179.064 (48.692) | 103.595 (14.643) |
| compare_f32_n1024 | unsupported | 438.765 (3.347) | 8.518 (0.508) | 102.639 (19.256) | 103.699 (6.699) |
| compare_f64_n1024 | unsupported | 445.294 (52.537) | 8.100 (0.264) | 94.712 (11.027) | 102.047 (4.010) |
| where_select_f32_n1024 | 18.442 (1.230) | 466.644 (23.469) | 11.849 (0.114) | 102.153 (10.329) | 105.070 (10.097) |
| where_select_f64_n1024 | 18.416 (1.823) | 503.050 (26.802) | 11.780 (0.088) | 102.774 (6.631) | 104.926 (8.344) |
| clamp_f32_n1024 | 49.780 (0.877) | 1028.439 (62.171) | 10.084 (0.272) | 92.566 (8.033) | 104.710 (10.168) |
| clamp_f64_n1024 | 53.348 (0.651) | 1086.941 (73.010) | 9.937 (0.185) | 96.081 (8.150) | 121.980 (12.352) |
| reduce_sum_f32_n1024 | 23.884 (0.469) | 264.683 (2.199) | 13.026 (0.340) | 94.288 (4.071) | 97.793 (5.538) |
| reduce_sum_f64_n1024 | 24.160 (0.478) | 264.577 (2.342) | 13.145 (0.171) | 94.721 (3.226) | 93.629 (3.771) |
| reduce_prod_f32_n1024 | 22.613 (0.548) | 217.091 (30.781) | 11.441 (0.148) | 94.623 (3.654) | 140.145 (4.139) |
| reduce_prod_f64_n1024 | 23.637 (0.769) | 215.988 (4.874) | 11.632 (0.105) | 130.826 (1.271) | 160.144 (4.328) |
| reduce_max_f32_n1024 | 24.433 (0.195) | 209.521 (4.611) | 12.698 (0.194) | 161.626 (3.054) | 174.973 (12.312) |
| reduce_max_f64_n1024 | 23.881 (0.717) | 213.914 (4.360) | 12.722 (0.383) | 115.031 (21.280) | 107.727 (15.774) |
| reduce_min_f32_n1024 | 24.755 (0.788) | 221.911 (5.093) | 12.582 (0.380) | 126.828 (7.196) | 100.348 (18.044) |
| reduce_min_f64_n1024 | 24.762 (0.660) | 192.866 (27.144) | 12.631 (0.262) | 91.505 (1.133) | 91.542 (0.799) |
| reduce_sum_squares_f32_n1024 | 24.358 (0.815) | 201.606 (4.155) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1024 | 22.131 (0.333) | 205.543 (2.536) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1024 | unsupported | unsupported | 13.575 (0.672) | 98.795 (0.415) | 100.114 (0.871) |
| reduce_mean_f64_n1024 | unsupported | unsupported | 13.139 (0.119) | 101.197 (0.411) | 101.188 (0.879) |
| add_f32_n1048576 | 48.286 (0.680) | 49.112 (0.823) | 39.764 (0.050) | 106.137 (2.241) | 100.031 (1.039) |
| add_f64_n1048576 | 92.934 (2.055) | 85.155 (0.377) | 77.770 (0.025) | 102.728 (1.780) | 99.103 (1.093) |
| sub_f32_n1048576 | 44.056 (0.132) | 44.107 (0.146) | 39.727 (0.029) | 98.770 (1.755) | 80.544 (5.143) |
| sub_f64_n1048576 | 83.564 (0.216) | 85.070 (0.239) | 77.741 (0.057) | 99.235 (2.925) | 83.636 (1.267) |
| mul_f32_n1048576 | 44.045 (0.141) | 44.415 (0.143) | 39.744 (0.046) | 102.492 (1.409) | 78.024 (3.146) |
| mul_f64_n1048576 | 83.517 (0.383) | 85.591 (0.302) | 77.699 (0.075) | 97.165 (3.567) | 83.031 (0.522) |
| div_f32_n1048576 | 44.030 (0.248) | 44.660 (0.387) | 39.978 (0.015) | 88.146 (0.496) | 73.262 (5.599) |
| div_f64_n1048576 | 96.796 (0.114) | 100.479 (0.622) | 94.513 (0.160) | 128.708 (1.556) | 127.797 (6.409) |
| rem_f32_n1048576 | 42.814 (0.444) | 98.186 (0.750) | 39.879 (0.029) | 116.433 (15.802) | 83.640 (9.882) |
| rem_f64_n1048576 | 81.311 (0.396) | 186.534 (0.229) | 78.276 (0.136) | 97.044 (25.263) | 86.286 (2.957) |
| pow_f32_n1048576 | 42.420 (0.295) | 97.851 (0.254) | 41.609 (0.149) | 134.812 (21.380) | 82.822 (8.791) |
| pow_f64_n1048576 | 1017.101 (0.597) | 1121.224 (1.560) | 1018.529 (0.474) | 1321.406 (6.222) | 1270.822 (16.820) |
| maximum_f32_n1048576 | 42.645 (0.156) | 98.100 (0.981) | 39.891 (0.027) | 124.057 (20.330) | 120.320 (10.542) |
| maximum_f64_n1048576 | 81.744 (0.347) | 187.099 (0.674) | 78.441 (0.175) | 122.140 (7.351) | 95.046 (1.190) |
| minimum_f32_n1048576 | 42.647 (0.183) | 97.756 (0.255) | 39.888 (0.026) | 103.902 (2.804) | 99.212 (0.370) |
| minimum_f64_n1048576 | 81.662 (0.230) | 186.947 (0.307) | 78.452 (0.106) | 93.600 (2.735) | 96.350 (3.672) |
| neg_f32_n1048576 | 29.749 (0.283) | 29.698 (0.154) | 27.316 (0.025) | 97.812 (0.850) | 96.292 (0.533) |
| neg_f64_n1048576 | 56.392 (0.330) | 56.954 (0.219) | 53.008 (0.067) | 96.233 (0.950) | 93.227 (17.718) |
| abs_f32_n1048576 | 29.478 (0.380) | 56.079 (0.146) | 27.425 (0.045) | 86.195 (2.258) | 71.753 (0.740) |
| abs_f64_n1048576 | 56.139 (0.321) | 107.132 (0.118) | 53.124 (0.063) | 86.779 (1.321) | 67.754 (2.529) |
| sign_f32_n1048576 | 29.286 (0.395) | 56.084 (0.246) | 27.374 (0.053) | 85.348 (0.326) | 71.984 (0.980) |
| sign_f64_n1048576 | 56.348 (0.269) | 107.055 (0.136) | 53.192 (0.059) | 86.410 (0.732) | 68.317 (2.043) |
| conj_c32_n1048576 | 55.939 (0.431) | 107.757 (0.322) | 52.952 (0.036) | 88.049 (2.907) | 69.997 (2.702) |
| conj_c64_n1048576 | 109.745 (0.696) | 209.293 (0.351) | 104.002 (0.052) | 118.487 (1.592) | 115.985 (2.010) |
| exp_f32_n1048576 | 30.050 (0.876) | 29.667 (0.223) | 27.410 (0.031) | 85.682 (0.690) | 74.953 (2.789) |
| exp_f64_n1048576 | 185.795 (0.738) | 187.477 (1.163) | 180.850 (0.089) | 203.540 (0.906) | 204.836 (1.598) |
| expm1_f32_n1048576 | 29.744 (0.556) | 29.814 (0.115) | 27.495 (0.071) | 92.162 (10.429) | 92.813 (15.826) |
| expm1_f64_n1048576 | 229.413 (0.890) | 230.826 (0.884) | 225.022 (0.088) | 266.195 (9.843) | 264.796 (4.723) |
| log_f32_n1048576 | 29.733 (0.379) | 29.683 (0.126) | 27.480 (0.050) | 93.040 (11.986) | 87.000 (13.850) |
| log_f64_n1048576 | 354.654 (0.268) | 357.120 (0.341) | 350.802 (0.158) | 406.342 (12.162) | 391.779 (15.692) |
| log1p_f32_n1048576 | 29.750 (0.225) | 29.800 (0.136) | 27.479 (0.038) | 92.469 (11.587) | 101.438 (10.617) |
| log1p_f64_n1048576 | 366.283 (0.517) | 370.075 (1.130) | 363.226 (0.368) | 428.055 (5.870) | 414.382 (19.343) |
| erf_f32_n1048576 | 29.585 (0.492) | 29.631 (0.061) | 27.550 (0.171) | 147.066 (42.794) | 121.697 (10.829) |
| erf_f64_n1048576 | 533.727 (0.449) | 535.391 (0.859) | 530.351 (0.174) | 688.809 (20.926) | 666.898 (18.635) |
| sin_f32_n1048576 | 29.679 (0.316) | 29.727 (0.241) | 27.543 (0.077) | 141.163 (14.433) | 132.222 (24.320) |
| sin_f64_n1048576 | 201.943 (0.449) | 204.925 (1.319) | 212.225 (0.100) | 253.567 (3.957) | 238.426 (1.758) |
| cos_f32_n1048576 | 29.783 (0.345) | 30.113 (0.372) | 27.577 (0.048) | 127.767 (14.053) | 125.764 (17.723) |
| cos_f64_n1048576 | 201.337 (0.519) | 203.213 (1.077) | 211.749 (0.104) | 238.583 (1.452) | 253.548 (2.462) |
| tanh_f32_n1048576 | 29.638 (0.382) | 29.675 (0.276) | 27.414 (0.026) | 92.379 (8.739) | 97.757 (5.938) |
| tanh_f64_n1048576 | 172.078 (0.913) | 173.997 (1.032) | 167.568 (0.054) | 190.390 (2.101) | 190.752 (2.180) |
| sqrt_f32_n1048576 | 29.653 (0.261) | 29.592 (0.103) | 27.456 (0.066) | 85.973 (3.183) | 96.963 (4.830) |
| sqrt_f64_n1048576 | 94.456 (0.486) | 95.294 (0.434) | 92.126 (0.030) | 102.979 (1.650) | 102.465 (1.693) |
| rsqrt_f32_n1048576 | 29.326 (0.425) | 29.568 (0.083) | 27.397 (0.045) | 129.102 (0.548) | 93.738 (2.541) |
| rsqrt_f64_n1048576 | 63.039 (0.473) | 64.082 (0.123) | 61.698 (0.027) | 140.958 (5.845) | 91.198 (0.784) |
| compare_f32_n1048576 | 32.165 (0.253) | 87.833 (0.632) | 30.448 (0.026) | 90.987 (0.509) | 98.453 (2.634) |
| compare_f64_n1048576 | 57.177 (0.137) | 164.684 (0.511) | 55.952 (0.048) | 92.083 (3.122) | 102.735 (0.835) |
| where_select_f32_n1048576 | 46.459 (0.279) | 118.013 (0.389) | 43.035 (0.030) | 102.282 (0.695) | 101.101 (1.962) |
| where_select_f64_n1048576 | 85.338 (0.214) | 206.891 (0.748) | 80.927 (0.173) | 97.196 (0.551) | 95.304 (0.666) |
| clamp_f32_n1048576 | 87.545 (0.881) | 490.537 (24.259) | 28.279 (0.026) | 90.814 (1.700) | 102.751 (2.717) |
| clamp_f64_n1048576 | 163.187 (1.706) | 503.549 (42.630) | 62.336 (0.041) | 82.227 (0.327) | 94.890 (1.251) |
| reduce_sum_f32_n1048576 | 40.583 (0.387) | 68.403 (0.613) | 16.285 (0.061) | 121.161 (15.020) | 155.060 (6.002) |
| reduce_sum_f64_n1048576 | 47.972 (1.028) | 95.212 (0.425) | 32.437 (0.036) | 156.002 (2.963) | 151.062 (3.438) |
| reduce_prod_f32_n1048576 | 45.324 (0.652) | 70.153 (0.332) | 16.142 (0.025) | 160.761 (4.125) | 151.313 (1.753) |
| reduce_prod_f64_n1048576 | 52.053 (0.314) | 95.557 (0.358) | 32.226 (0.019) | 152.374 (2.092) | 161.747 (3.991) |
| reduce_max_f32_n1048576 | 65.673 (0.403) | 74.907 (0.661) | 16.554 (0.015) | 153.573 (2.386) | 152.030 (3.547) |
| reduce_max_f64_n1048576 | 77.278 (0.337) | 112.710 (0.180) | 41.744 (0.016) | 159.430 (3.570) | 157.718 (2.428) |
| reduce_min_f32_n1048576 | 65.111 (0.369) | 86.545 (0.724) | 16.571 (0.034) | 176.782 (6.294) | 149.138 (12.147) |
| reduce_min_f64_n1048576 | 77.968 (0.437) | 112.433 (0.326) | 41.745 (0.031) | 169.719 (3.750) | 151.731 (13.984) |
| reduce_sum_squares_f32_n1048576 | 43.308 (0.497) | 66.589 (0.796) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1048576 | 52.978 (0.523) | 96.286 (0.457) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1048576 | unsupported | unsupported | 16.345 (0.137) | 172.920 (24.422) | 163.369 (26.786) |
| reduce_mean_f64_n1048576 | unsupported | unsupported | 32.871 (0.029) | 163.628 (15.578) | 173.117 (21.889) |
| reduce_sum_full_f32_n1024 | 19.261 (2.519) | 190.102 (55.619) | 11.758 (0.108) | 172.093 (6.773) | 173.612 (9.312) |
| reduce_prod_full_f32_n1024 | 16.292 (0.507) | 273.831 (11.087) | 10.155 (0.087) | 163.924 (2.896) | 135.197 (8.503) |
| reduce_max_full_f32_n1024 | 13.335 (0.304) | 277.713 (9.609) | 11.505 (0.183) | 166.479 (5.112) | 169.076 (12.921) |
| reduce_min_full_f32_n1024 | 12.410 (0.229) | 281.411 (4.527) | 10.839 (0.295) | 166.129 (9.516) | 170.984 (23.427) |
| compare_eq_f32_n1024 | unsupported | 569.790 (69.349) | 7.894 (0.193) | 149.400 (30.334) | 142.065 (26.420) |
| compare_lt_f32_n1024 | unsupported | 566.372 (68.338) | 8.264 (0.090) | 135.865 (26.979) | 119.779 (25.018) |
| compare_le_f32_n1024 | unsupported | 580.889 (46.512) | 8.435 (0.078) | 128.970 (12.041) | 115.594 (20.693) |
| compare_ge_f32_n1024 | unsupported | 630.543 (71.895) | 8.123 (0.205) | 101.653 (18.630) | 95.476 (16.871) |
| reduce_sum_full_f64_n1024 | 21.975 (0.298) | 274.044 (4.151) | 11.393 (0.142) | 156.129 (6.580) | 162.423 (16.310) |
| reduce_prod_full_f64_n1024 | 20.512 (1.020) | 273.774 (5.364) | 10.262 (0.196) | 160.844 (3.628) | 138.090 (10.271) |
| reduce_max_full_f64_n1024 | 18.106 (1.093) | 273.481 (5.247) | 11.053 (0.143) | 141.431 (2.477) | 109.909 (15.782) |
| reduce_min_full_f64_n1024 | 15.659 (0.495) | 274.695 (8.205) | 10.999 (0.267) | 91.307 (0.820) | 91.243 (0.288) |
| compare_eq_f64_n1024 | unsupported | 565.200 (40.880) | 8.294 (0.063) | 90.343 (5.316) | 99.374 (2.950) |
| compare_lt_f64_n1024 | unsupported | 566.817 (58.802) | 8.314 (0.162) | 91.998 (8.189) | 101.454 (6.830) |
| compare_le_f64_n1024 | unsupported | 571.363 (64.017) | 8.607 (0.096) | 90.021 (6.574) | 100.366 (8.347) |
| compare_ge_f64_n1024 | unsupported | 591.189 (65.439) | 8.082 (0.130) | 86.470 (3.253) | 96.834 (10.127) |
| reduce_sum_full_f32_n1048576 | 2540.898 (5.054) | 2649.814 (12.890) | 18.581 (0.129) | 163.576 (7.194) | 152.308 (5.533) |
| reduce_prod_full_f32_n1048576 | 2528.350 (5.685) | 2618.567 (8.678) | 18.436 (0.116) | 119.166 (22.291) | 145.340 (13.075) |
| reduce_max_full_f32_n1048576 | 9816.299 (4.637) | 9872.136 (5.754) | 18.848 (0.066) | 162.841 (3.485) | 159.246 (1.766) |
| reduce_min_full_f32_n1048576 | 9817.952 (6.512) | 9872.515 (7.886) | 18.971 (0.128) | 181.671 (19.018) | 170.365 (25.086) |
| compare_eq_f32_n1048576 | 32.697 (0.189) | 93.700 (0.948) | 30.288 (0.017) | 95.274 (13.133) | 122.975 (16.532) |
| compare_lt_f32_n1048576 | 32.383 (0.310) | 93.591 (1.058) | 30.286 (0.015) | 90.785 (1.343) | 99.763 (0.704) |
| compare_le_f32_n1048576 | 32.385 (0.251) | 93.618 (1.599) | 30.284 (0.065) | 90.335 (0.725) | 95.348 (1.258) |
| compare_ge_f32_n1048576 | 32.353 (0.209) | 87.631 (0.302) | 30.294 (0.037) | 90.987 (1.638) | 98.891 (1.032) |
| reduce_sum_full_f64_n1048576 | 3034.038 (2.745) | 3095.259 (5.694) | 37.455 (0.045) | 93.832 (1.612) | 159.047 (5.613) |
| reduce_prod_full_f64_n1048576 | 3034.066 (2.790) | 3096.591 (2.879) | 37.488 (0.055) | 132.557 (1.084) | 163.093 (23.578) |
| reduce_max_full_f64_n1048576 | 12489.376 (6.020) | 12538.688 (3.699) | 44.713 (0.057) | 159.879 (16.049) | 165.393 (19.355) |
| reduce_min_full_f64_n1048576 | 12483.861 (3.888) | 12535.940 (6.517) | 44.700 (0.033) | 169.053 (22.949) | 146.146 (3.514) |
| compare_eq_f64_n1048576 | 57.314 (0.294) | 164.720 (0.474) | 56.850 (0.053) | 96.946 (8.565) | 108.136 (7.103) |
| compare_lt_f64_n1048576 | 57.298 (0.080) | 164.601 (0.250) | 56.024 (0.076) | 84.497 (1.410) | 102.978 (1.485) |
| compare_le_f64_n1048576 | 57.331 (0.193) | 164.725 (0.651) | 55.984 (0.038) | 92.800 (2.246) | 103.052 (0.958) |
| compare_ge_f64_n1048576 | 57.333 (0.237) | 164.423 (0.256) | 56.078 (0.139) | 93.947 (2.281) | 102.850 (0.624) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.831 | 0 |
| add_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.091 | 0 |
| sub_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.071 | 0 |
| sub_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.222 | 0 |
| mul_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.224 | 0 |
| mul_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.259 | 0 |
| div_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.458 | 0 |
| div_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.213 | 0 |
| rem_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.153 | 0 |
| rem_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.172 | 0 |
| pow_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.151 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.267 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.131 | 0 |
| maximum_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.153 | 0 |
| minimum_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.138 | 0 |
| minimum_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.146 | 0 |
| neg_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.326 | 0 |
| neg_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.220 | 0 |
| abs_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.840 | 0 |
| abs_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.740 | 0 |
| sign_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.646 | 0 |
| sign_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.652 | 0 |
| conj_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.473 | 0 |
| conj_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.459 | 0 |
| exp_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.947 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.940 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.610 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.624 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.574 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.603 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.370 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.789 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.660 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.590 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.687 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.669 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.493 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.375 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.271 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.313 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.317 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.436 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.236 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.344 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; steady-state sampling is not established: consecutive batch durations increase in repeated runs 20261013_101359 and 20261013_102038. Retained as diagnostics pending allocator/lifetime investigation. |
| compare_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; steady-state sampling is not established: consecutive batch durations increase in repeated runs 20261013_101359 and 20261013_102038. Retained as diagnostics pending allocator/lifetime investigation. |
| where_select_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.361 | 0 |
| where_select_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.357 | 0 |
| clamp_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.186 | 0 |
| clamp_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.414 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.529 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.546 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.894 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.026 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.127 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.057 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.169 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.170 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.118 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.833 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.545 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.487 | 0 |
| sub_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.820 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.674 | 0 |
| mul_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.819 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.673 | 0 |
| div_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.818 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.097 | 0 |
| rem_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.740 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.602 | 0 |
| pow_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.715 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.034 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.729 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.616 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.729 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.613 | 0 |
| neg_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.904 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.805 | 0 |
| abs_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.887 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.796 | 0 |
| sign_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.874 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.803 | 0 |
| conj_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.790 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.756 | 0 |
| exp_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.923 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.973 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.904 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.671 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.903 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.837 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.904 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.930 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.893 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.135 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.899 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.231 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.906 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.221 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.897 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.753 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.898 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.023 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.877 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.017 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.059 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.659 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.973 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.731 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.801 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.611 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.597 | 0.001812 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.070 | 2.148e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.901 | 0.005478 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.331 | 3.659e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.102 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.473 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.084 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.495 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.772 | 0.0003528 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.391 | 3.293e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_full_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.233 | 0.0001333 |
| reduce_prod_full_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.085 | 0.003334 |
| reduce_max_full_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.414 | 0 |
| reduce_min_full_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.177 | 0 |
| compare_eq_f32_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| compare_lt_f32_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| compare_le_f32_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| compare_ge_f32_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| reduce_sum_full_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.406 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.626 | 0 |
| reduce_max_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.318 | 0 |
| reduce_min_full_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.004 | 0 |
| compare_eq_f64_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| compare_lt_f64_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| compare_le_f64_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| compare_ge_f64_n1024 | tenferro-cuda-direct | — | — | — | Small comparison allocation path requires steady-state lifetime validation; known Gt drift is tracked separately from numerical support. |
| reduce_sum_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.541 | 0.07061 |
| reduce_prod_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.528 | 0.1957 |
| reduce_max_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 9.816 | 0 |
| reduce_min_full_f32_n1048576 | tenferro-cuda-direct | 1 | 15 | 9.818 | 0 |
| compare_eq_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.093 | 0 |
| compare_lt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.073 | 0 |
| compare_le_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.073 | 0 |
| compare_ge_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.071 | 0 |
| reduce_sum_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.034 | 0.001745 |
| reduce_prod_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 3.034 | 6.451e-05 |
| reduce_max_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 12.489 | 0 |
| reduce_min_full_f64_n1048576 | tenferro-cuda-direct | 1 | 15 | 12.484 | 0 |
| compare_eq_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.668 | 0 |
| compare_lt_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.667 | 0 |
| compare_le_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.669 | 0 |
| compare_ge_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.669 | 0 |
| add_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.862 | 0 |
| add_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.491 | 0 |
| sub_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.756 | 0 |
| sub_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.928 | 0 |
| mul_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.076 | 0 |
| mul_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.143 | 0 |
| div_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.177 | 0 |
| div_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.184 | 0 |
| rem_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.399 | 0 |
| rem_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.956 | 0 |
| pow_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.355 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 4.039 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.424 | 0 |
| maximum_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.760 | 0 |
| minimum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.221 | 0 |
| minimum_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.323 | 0 |
| neg_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.628 | 0 |
| neg_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.694 | 0 |
| abs_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.474 | 0 |
| abs_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.392 | 0 |
| sign_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.735 | 0 |
| sign_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.777 | 0 |
| conj_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.817 | 0 |
| conj_c64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.341 | 0 |
| exp_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.686 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.810 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.796 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.749 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.756 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.082 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.794 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.761 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.789 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.666 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.639 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.013 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.839 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.827 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.783 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.795 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.732 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.720 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.763 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.823 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.510 | 0 |
| compare_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.562 | 0 |
| where_select_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.733 | 0 |
| where_select_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 4.024 | 0 |
| clamp_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 4.114 | 0 |
| clamp_f64_n1024 | tenferro-cuda-eager | 2 | 15 | 2.174 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.117 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.117 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 1.737 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.456 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.352 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.423 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.551 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.086 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.226 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.289 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.572 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.725 | 0 |
| sub_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.823 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.722 | 0 |
| mul_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.843 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.739 | 0 |
| div_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.858 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.215 | 0 |
| rem_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.142 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.985 | 0 |
| pow_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.131 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-eager | 2 | 15 | 2.242 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.139 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.994 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.128 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.991 | 0 |
| neg_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.901 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.823 | 0 |
| abs_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.589 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.428 | 0 |
| sign_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.589 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.426 | 0 |
| conj_c32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.448 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.349 | 0 |
| exp_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.899 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.000 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.908 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.693 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.900 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.857 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.907 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.961 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.896 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.142 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.903 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.279 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.927 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.251 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.899 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.784 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.894 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.049 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.892 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.051 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.811 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.635 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.776 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.310 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.924 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 4.028 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.189 | 0.001812 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.047 | 2.148e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.245 | 0.005478 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.058 | 3.659e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.397 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.607 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.769 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.598 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.131 | 0.0003528 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.081 | 3.293e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_full_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.042 | 0.0001333 |
| reduce_prod_full_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 4.381 | 0.003334 |
| reduce_max_full_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.222 | 0 |
| reduce_min_full_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.251 | 0 |
| compare_eq_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.279 | 0 |
| compare_lt_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.265 | 0 |
| compare_le_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.324 | 0 |
| compare_ge_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 2.522 | 0 |
| reduce_sum_full_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.192 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.190 | 0 |
| reduce_max_full_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.188 | 0 |
| reduce_min_full_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.198 | 0 |
| compare_eq_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.261 | 0 |
| compare_lt_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.267 | 0 |
| compare_le_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.285 | 0 |
| compare_ge_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.365 | 0 |
| reduce_sum_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.650 | 0.07061 |
| reduce_prod_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 2.619 | 0.1957 |
| reduce_max_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 9.872 | 0 |
| reduce_min_full_f32_n1048576 | tenferro-cuda-eager | 1 | 15 | 9.873 | 0 |
| compare_eq_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.998 | 0 |
| compare_lt_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.995 | 0 |
| compare_le_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.996 | 0 |
| compare_ge_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.804 | 0 |
| reduce_sum_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.095 | 0.001745 |
| reduce_prod_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 3.097 | 6.451e-05 |
| reduce_max_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 12.539 | 0 |
| reduce_min_full_f64_n1048576 | tenferro-cuda-eager | 1 | 15 | 12.536 | 0 |
| compare_eq_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.636 | 0 |
| compare_lt_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.634 | 0 |
| compare_le_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.636 | 0 |
| compare_ge_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.631 | 0 |
| add_f32_n1024 | pytorch-cuda | 256 | 15 | 2.168 | 0 |
| add_f64_n1024 | pytorch-cuda | 256 | 15 | 2.236 | 0 |
| sub_f32_n1024 | pytorch-cuda | 256 | 15 | 2.264 | 0 |
| sub_f64_n1024 | pytorch-cuda | 256 | 15 | 2.242 | 0 |
| mul_f32_n1024 | pytorch-cuda | 256 | 15 | 2.330 | 0 |
| mul_f64_n1024 | pytorch-cuda | 256 | 15 | 2.294 | 0 |
| div_f32_n1024 | pytorch-cuda | 256 | 15 | 2.221 | 0 |
| div_f64_n1024 | pytorch-cuda | 256 | 15 | 2.149 | 0 |
| rem_f32_n1024 | pytorch-cuda | 256 | 15 | 2.165 | 0 |
| rem_f64_n1024 | pytorch-cuda | 256 | 15 | 2.186 | 0 |
| pow_f32_n1024 | pytorch-cuda | 256 | 15 | 2.284 | 0.000117 |
| pow_f64_n1024 | pytorch-cuda | 128 | 15 | 3.955 | 1.48e-07 |
| maximum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.199 | 0 |
| maximum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.195 | 0 |
| minimum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.213 | 0 |
| minimum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.175 | 0 |
| neg_f32_n1024 | pytorch-cuda | 256 | 15 | 2.102 | 0 |
| neg_f64_n1024 | pytorch-cuda | 256 | 15 | 2.096 | 0 |
| abs_f32_n1024 | pytorch-cuda | 256 | 15 | 2.805 | 0 |
| abs_f64_n1024 | pytorch-cuda | 256 | 15 | 2.672 | 0 |
| sign_f32_n1024 | pytorch-cuda | 256 | 15 | 2.069 | 0 |
| sign_f64_n1024 | pytorch-cuda | 256 | 15 | 2.155 | 0 |
| conj_c32_n1024 | pytorch-cuda | 256 | 15 | 2.524 | 0 |
| conj_c64_n1024 | pytorch-cuda | 256 | 15 | 2.507 | 0 |
| exp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.112 | 0.0001272 |
| exp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.040 | 1.095e-07 |
| expm1_f32_n1024 | pytorch-cuda | 256 | 15 | 2.242 | 7.911e-05 |
| expm1_f64_n1024 | pytorch-cuda | 256 | 15 | 2.099 | 7.39e-08 |
| log_f32_n1024 | pytorch-cuda | 256 | 15 | 2.043 | 7.945e-05 |
| log_f64_n1024 | pytorch-cuda | 256 | 15 | 2.787 | 4.231e-08 |
| log1p_f32_n1024 | pytorch-cuda | 256 | 15 | 2.118 | 0.0001189 |
| log1p_f64_n1024 | pytorch-cuda | 256 | 15 | 2.886 | 1.108e-07 |
| erf_f32_n1024 | pytorch-cuda | 256 | 15 | 2.050 | 7.914e-05 |
| erf_f64_n1024 | pytorch-cuda | 128 | 15 | 2.028 | 8.536e-08 |
| sin_f32_n1024 | pytorch-cuda | 256 | 15 | 2.021 | 4.348e-05 |
| sin_f64_n1024 | pytorch-cuda | 256 | 15 | 2.026 | 4.153e-08 |
| cos_f32_n1024 | pytorch-cuda | 256 | 15 | 2.018 | 6.342e-05 |
| cos_f64_n1024 | pytorch-cuda | 256 | 15 | 2.024 | 5.905e-08 |
| tanh_f32_n1024 | pytorch-cuda | 256 | 15 | 2.035 | 9.107e-05 |
| tanh_f64_n1024 | pytorch-cuda | 256 | 15 | 2.214 | 4.434e-08 |
| sqrt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.172 | 0 |
| sqrt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.016 | 0 |
| rsqrt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.100 | 6.983e-05 |
| rsqrt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.044 | 6.504e-08 |
| compare_f32_n1024 | pytorch-cuda | 256 | 15 | 2.181 | 0 |
| compare_f64_n1024 | pytorch-cuda | 256 | 15 | 2.074 | 0 |
| where_select_f32_n1024 | pytorch-cuda | 256 | 15 | 3.033 | 0 |
| where_select_f64_n1024 | pytorch-cuda | 256 | 15 | 3.016 | 0 |
| clamp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.581 | 0 |
| clamp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.544 | 0 |
| reduce_sum_f32_n1024 | pytorch-cuda | 256 | 15 | 3.335 | 0.0002742 |
| reduce_sum_f64_n1024 | pytorch-cuda | 256 | 15 | 3.365 | 1.06e-07 |
| reduce_prod_f32_n1024 | pytorch-cuda | 256 | 15 | 2.929 | 0.0003579 |
| reduce_prod_f64_n1024 | pytorch-cuda | 256 | 15 | 2.978 | 4.441e-07 |
| reduce_max_f32_n1024 | pytorch-cuda | 256 | 15 | 3.251 | 0 |
| reduce_max_f64_n1024 | pytorch-cuda | 256 | 15 | 3.257 | 0 |
| reduce_min_f32_n1024 | pytorch-cuda | 256 | 15 | 3.221 | 0 |
| reduce_min_f64_n1024 | pytorch-cuda | 256 | 15 | 3.234 | 0 |
| reduce_sum_squares_f32_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | pytorch-cuda | 256 | 15 | 3.475 | 1.439e-05 |
| reduce_mean_f64_n1024 | pytorch-cuda | 256 | 15 | 3.364 | 6.71e-09 |
| add_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.545 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.489 | 0 |
| sub_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.543 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| mul_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| div_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.559 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.024 | 0 |
| rem_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.552 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.505 | 0 |
| pow_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.663 | 0.0001577 |
| pow_f64_n1048576 | pytorch-cuda | 2 | 15 | 2.037 | 1.48e-07 |
| maximum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.510 | 0 |
| minimum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.510 | 0 |
| neg_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.748 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.696 | 0 |
| abs_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.755 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.700 | 0 |
| sign_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.752 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.702 | 0 |
| conj_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.694 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.664 | 0 |
| exp_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.754 | 0.0001272 |
| exp_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.894 | 1.095e-07 |
| expm1_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.760 | 7.915e-05 |
| expm1_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.600 | 7.39e-08 |
| log_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.759 | 7.946e-05 |
| log_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.806 | 4.308e-08 |
| log1p_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.759 | 0.000119 |
| log1p_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.906 | 1.11e-07 |
| erf_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.763 | 7.923e-05 |
| erf_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.121 | 8.536e-08 |
| sin_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.763 | 4.348e-05 |
| sin_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.396 | 4.153e-08 |
| cos_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.765 | 6.342e-05 |
| cos_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.388 | 5.905e-08 |
| tanh_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.755 | 9.135e-05 |
| tanh_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.681 | 4.437e-08 |
| sqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.757 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.948 | 0 |
| rsqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.753 | 6.983e-05 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.974 | 6.504e-08 |
| compare_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.949 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.581 | 0 |
| where_select_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.754 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.590 | 0 |
| clamp_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.810 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.995 | 0 |
| reduce_sum_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.085 | 0.002948 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.076 | 2.961e-06 |
| reduce_prod_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.066 | 0.002623 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.062 | 4.216e-06 |
| reduce_max_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.119 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.672 | 0 |
| reduce_min_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.121 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.672 | 0 |
| reduce_sum_squares_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.092 | 2.947e-06 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.104 | 3.9e-09 |
| reduce_sum_full_f32_n1024 | pytorch-cuda | 256 | 15 | 3.010 | 0.0002666 |
| reduce_prod_full_f32_n1024 | pytorch-cuda | 256 | 15 | 2.600 | 0.0025 |
| reduce_max_full_f32_n1024 | pytorch-cuda | 256 | 15 | 2.945 | 0 |
| reduce_min_full_f32_n1024 | pytorch-cuda | 256 | 15 | 2.775 | 0 |
| compare_eq_f32_n1024 | pytorch-cuda | 256 | 15 | 2.021 | 0 |
| compare_lt_f32_n1024 | pytorch-cuda | 256 | 15 | 2.116 | 0 |
| compare_le_f32_n1024 | pytorch-cuda | 256 | 15 | 2.159 | 0 |
| compare_ge_f32_n1024 | pytorch-cuda | 256 | 15 | 2.079 | 0 |
| reduce_sum_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.917 | 6.208e-07 |
| reduce_prod_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.627 | 1.663e-06 |
| reduce_max_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.830 | 0 |
| reduce_min_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.816 | 0 |
| compare_eq_f64_n1024 | pytorch-cuda | 256 | 15 | 2.123 | 0 |
| compare_lt_f64_n1024 | pytorch-cuda | 256 | 15 | 2.128 | 0 |
| compare_le_f64_n1024 | pytorch-cuda | 256 | 15 | 2.203 | 0 |
| compare_ge_f64_n1024 | pytorch-cuda | 256 | 15 | 2.069 | 0 |
| reduce_sum_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.378 | 0.00522 |
| reduce_prod_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.360 | 0.5197 |
| reduce_max_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.412 | 0 |
| reduce_min_full_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.428 | 0 |
| compare_eq_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.877 | 0 |
| compare_lt_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.877 | 0 |
| compare_le_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.876 | 0 |
| compare_ge_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.878 | 0 |
| reduce_sum_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.397 | 0.0004425 |
| reduce_prod_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.399 | 0.0001269 |
| reduce_max_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.862 | 0 |
| reduce_min_full_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.861 | 0 |
| compare_eq_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.638 | 0 |
| compare_lt_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.586 | 0 |
| compare_le_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.583 | 0 |
| compare_ge_f64_n1048576 | pytorch-cuda | 64 | 15 | 3.589 | 0 |
| add_f32_n1024 | jax-cuda | 32 | 15 | 3.309 | 0 |
| add_f64_n1024 | jax-cuda | 32 | 15 | 2.962 | 0 |
| sub_f32_n1024 | jax-cuda | 32 | 15 | 2.764 | 0 |
| sub_f64_n1024 | jax-cuda | 32 | 15 | 2.711 | 0 |
| mul_f32_n1024 | jax-cuda | 32 | 15 | 2.594 | 0 |
| mul_f64_n1024 | jax-cuda | 32 | 15 | 2.550 | 0 |
| div_f32_n1024 | jax-cuda | 32 | 15 | 2.391 | 4.875e-05 |
| div_f64_n1024 | jax-cuda | 32 | 15 | 2.603 | 0 |
| rem_f32_n1024 | jax-cuda | 32 | 15 | 2.929 | 0 |
| rem_f64_n1024 | jax-cuda | 32 | 15 | 3.208 | 0 |
| pow_f32_n1024 | jax-cuda | 16 | 15 | 2.243 | 0.000117 |
| pow_f64_n1024 | jax-cuda | 16 | 15 | 2.970 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda | 16 | 15 | 2.246 | 0 |
| maximum_f64_n1024 | jax-cuda | 16 | 15 | 2.175 | 0 |
| minimum_f32_n1024 | jax-cuda | 16 | 15 | 2.139 | 0 |
| minimum_f64_n1024 | jax-cuda | 16 | 15 | 2.090 | 0 |
| neg_f32_n1024 | jax-cuda | 16 | 15 | 1.882 | 0 |
| neg_f64_n1024 | jax-cuda | 16 | 15 | 1.978 | 0 |
| abs_f32_n1024 | jax-cuda | 32 | 15 | 3.070 | 0 |
| abs_f64_n1024 | jax-cuda | 32 | 15 | 2.818 | 0 |
| sign_f32_n1024 | jax-cuda | 32 | 15 | 3.135 | 0 |
| sign_f64_n1024 | jax-cuda | 32 | 15 | 3.060 | 0 |
| conj_c32_n1024 | jax-cuda | 32 | 15 | 2.891 | 0 |
| conj_c64_n1024 | jax-cuda | 32 | 15 | 2.920 | 0 |
| exp_f32_n1024 | jax-cuda | 32 | 15 | 3.006 | 0.0001272 |
| exp_f64_n1024 | jax-cuda | 32 | 15 | 3.096 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda | 32 | 15 | 3.207 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda | 32 | 15 | 3.131 | 7.39e-08 |
| log_f32_n1024 | jax-cuda | 32 | 15 | 3.131 | 7.945e-05 |
| log_f64_n1024 | jax-cuda | 32 | 15 | 3.262 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda | 32 | 15 | 3.072 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda | 32 | 15 | 3.385 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda | 16 | 15 | 3.083 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda | 16 | 15 | 3.375 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda | 16 | 15 | 2.248 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda | 16 | 15 | 2.458 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda | 16 | 15 | 2.570 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda | 16 | 15 | 2.522 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda | 16 | 15 | 2.349 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda | 16 | 15 | 2.424 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda | 16 | 15 | 2.272 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda | 16 | 15 | 2.201 | 0 |
| rsqrt_f32_n1024 | jax-cuda | 16 | 15 | 2.822 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda | 16 | 15 | 2.865 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda | 32 | 15 | 3.284 | 0 |
| compare_f64_n1024 | jax-cuda | 32 | 15 | 3.031 | 0 |
| where_select_f32_n1024 | jax-cuda | 32 | 15 | 3.269 | 0 |
| where_select_f64_n1024 | jax-cuda | 32 | 15 | 3.289 | 0 |
| clamp_f32_n1024 | jax-cuda | 32 | 15 | 2.962 | 0 |
| clamp_f64_n1024 | jax-cuda | 32 | 15 | 3.075 | 0 |
| reduce_sum_f32_n1024 | jax-cuda | 32 | 15 | 3.017 | 0.0001139 |
| reduce_sum_f64_n1024 | jax-cuda | 32 | 15 | 3.031 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda | 32 | 15 | 3.028 | 0.0003574 |
| reduce_prod_f64_n1024 | jax-cuda | 16 | 15 | 2.093 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda | 16 | 15 | 2.586 | 0 |
| reduce_max_f64_n1024 | jax-cuda | 32 | 15 | 3.681 | 0 |
| reduce_min_f32_n1024 | jax-cuda | 16 | 15 | 2.029 | 0 |
| reduce_min_f64_n1024 | jax-cuda | 32 | 15 | 2.928 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda | 32 | 15 | 3.161 | 7.204e-06 |
| reduce_mean_f64_n1024 | jax-cuda | 32 | 15 | 3.238 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda | 16 | 15 | 1.698 | 0 |
| add_f64_n1048576 | jax-cuda | 16 | 15 | 1.644 | 0 |
| sub_f32_n1048576 | jax-cuda | 32 | 15 | 3.161 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 3.176 | 0 |
| mul_f32_n1048576 | jax-cuda | 32 | 15 | 3.280 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 3.109 | 0 |
| div_f32_n1048576 | jax-cuda | 32 | 15 | 2.821 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda | 16 | 15 | 2.059 | 0 |
| rem_f32_n1048576 | jax-cuda | 16 | 15 | 1.863 | 0 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 3.105 | 0 |
| pow_f32_n1048576 | jax-cuda | 16 | 15 | 2.157 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda | 2 | 15 | 2.643 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda | 16 | 15 | 1.985 | 0 |
| maximum_f64_n1048576 | jax-cuda | 16 | 15 | 1.954 | 0 |
| minimum_f32_n1048576 | jax-cuda | 32 | 15 | 3.325 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 2.995 | 0 |
| neg_f32_n1048576 | jax-cuda | 32 | 15 | 3.130 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 3.079 | 0 |
| abs_f32_n1048576 | jax-cuda | 32 | 15 | 2.758 | 0 |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 2.777 | 0 |
| sign_f32_n1048576 | jax-cuda | 32 | 15 | 2.731 | 0 |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 2.765 | 0 |
| conj_c32_n1048576 | jax-cuda | 32 | 15 | 2.818 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 1.896 | 0 |
| exp_f32_n1048576 | jax-cuda | 32 | 15 | 2.742 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda | 16 | 15 | 3.257 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda | 32 | 15 | 2.949 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda | 8 | 15 | 2.130 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda | 32 | 15 | 2.977 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda | 8 | 15 | 3.251 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda | 32 | 15 | 2.959 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda | 8 | 15 | 3.424 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda | 16 | 15 | 2.353 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda | 4 | 15 | 2.755 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda | 16 | 15 | 2.259 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda | 8 | 15 | 2.029 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda | 16 | 15 | 2.044 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda | 16 | 15 | 3.817 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda | 32 | 15 | 2.956 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda | 16 | 15 | 3.046 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda | 32 | 15 | 2.751 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 3.295 | 0 |
| rsqrt_f32_n1048576 | jax-cuda | 16 | 15 | 2.066 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda | 16 | 15 | 2.255 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda | 32 | 15 | 2.912 | 0 |
| compare_f64_n1048576 | jax-cuda | 32 | 15 | 2.947 | 0 |
| where_select_f32_n1048576 | jax-cuda | 32 | 15 | 3.273 | 0 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 3.110 | 0 |
| clamp_f32_n1048576 | jax-cuda | 32 | 15 | 2.906 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 2.631 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda | 32 | 15 | 3.877 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda | 16 | 15 | 2.496 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda | 16 | 15 | 2.572 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda | 16 | 15 | 2.438 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda | 16 | 15 | 2.457 | 0 |
| reduce_max_f64_n1048576 | jax-cuda | 16 | 15 | 2.551 | 0 |
| reduce_min_f32_n1048576 | jax-cuda | 16 | 15 | 2.829 | 0 |
| reduce_min_f64_n1048576 | jax-cuda | 16 | 15 | 2.716 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda | 16 | 15 | 2.767 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda | 16 | 15 | 2.618 | 6.5e-09 |
| reduce_sum_full_f32_n1024 | jax-cuda | 16 | 15 | 2.753 | 0 |
| reduce_prod_full_f32_n1024 | jax-cuda | 16 | 15 | 2.623 | 0.001072 |
| reduce_max_full_f32_n1024 | jax-cuda | 16 | 15 | 2.664 | 0 |
| reduce_min_full_f32_n1024 | jax-cuda | 16 | 15 | 2.658 | 0 |
| compare_eq_f32_n1024 | jax-cuda | 16 | 15 | 2.390 | 0 |
| compare_lt_f32_n1024 | jax-cuda | 16 | 15 | 2.174 | 0 |
| compare_le_f32_n1024 | jax-cuda | 16 | 15 | 2.064 | 0 |
| compare_ge_f32_n1024 | jax-cuda | 32 | 15 | 3.253 | 0 |
| reduce_sum_full_f64_n1024 | jax-cuda | 16 | 15 | 2.498 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | jax-cuda | 16 | 15 | 2.574 | 4.435e-07 |
| reduce_max_full_f64_n1024 | jax-cuda | 16 | 15 | 2.263 | 0 |
| reduce_min_full_f64_n1024 | jax-cuda | 32 | 15 | 2.922 | 0 |
| compare_eq_f64_n1024 | jax-cuda | 32 | 15 | 2.891 | 0 |
| compare_lt_f64_n1024 | jax-cuda | 32 | 15 | 2.944 | 0 |
| compare_le_f64_n1024 | jax-cuda | 32 | 15 | 2.881 | 0 |
| compare_ge_f64_n1024 | jax-cuda | 32 | 15 | 2.767 | 0 |
| reduce_sum_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.617 | 0.003668 |
| reduce_prod_full_f32_n1048576 | jax-cuda | 32 | 15 | 3.813 | 0.5327 |
| reduce_max_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.605 | 0 |
| reduce_min_full_f32_n1048576 | jax-cuda | 16 | 15 | 2.907 | 0 |
| compare_eq_f32_n1048576 | jax-cuda | 32 | 15 | 3.049 | 0 |
| compare_lt_f32_n1048576 | jax-cuda | 32 | 15 | 2.905 | 0 |
| compare_le_f32_n1048576 | jax-cuda | 32 | 15 | 2.891 | 0 |
| compare_ge_f32_n1048576 | jax-cuda | 32 | 15 | 2.912 | 0 |
| reduce_sum_full_f64_n1048576 | jax-cuda | 32 | 15 | 3.003 | 0.0006452 |
| reduce_prod_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.121 | 0.0001122 |
| reduce_max_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.558 | 0 |
| reduce_min_full_f64_n1048576 | jax-cuda | 16 | 15 | 2.705 | 0 |
| compare_eq_f64_n1048576 | jax-cuda | 32 | 15 | 3.102 | 0 |
| compare_lt_f64_n1048576 | jax-cuda | 32 | 15 | 2.704 | 0 |
| compare_le_f64_n1048576 | jax-cuda | 32 | 15 | 2.970 | 0 |
| compare_ge_f64_n1048576 | jax-cuda | 32 | 15 | 3.006 | 0 |
| add_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.719 | 0 |
| add_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.666 | 0 |
| sub_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.604 | 0 |
| sub_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.620 | 0 |
| mul_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.624 | 0 |
| mul_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.586 | 0 |
| div_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.569 | 4.875e-05 |
| div_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.582 | 0 |
| rem_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.115 | 0 |
| rem_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.048 | 0 |
| pow_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.420 | 0.000117 |
| pow_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.728 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.346 | 0 |
| maximum_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.174 | 0 |
| minimum_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.053 | 0 |
| minimum_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.358 | 0 |
| neg_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.184 | 0 |
| neg_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.188 | 0 |
| abs_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.196 | 0 |
| abs_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.199 | 0 |
| sign_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.242 | 0 |
| sign_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.067 | 0 |
| conj_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.257 | 0 |
| conj_c64_n1024 | jax-cuda-jit | 32 | 15 | 3.228 | 0 |
| exp_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.211 | 0.0001272 |
| exp_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.092 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.076 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.269 | 7.39e-08 |
| log_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.272 | 7.945e-05 |
| log_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.482 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.302 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.048 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.408 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.336 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.504 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.679 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.746 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.583 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.660 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.419 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.215 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.129 | 0 |
| rsqrt_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.404 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.315 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.318 | 0 |
| compare_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.266 | 0 |
| where_select_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.362 | 0 |
| where_select_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.358 | 0 |
| clamp_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.351 | 0 |
| clamp_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.952 | 0 |
| reduce_sum_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.129 | 0.0001139 |
| reduce_sum_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.996 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.242 | 0.0003579 |
| reduce_prod_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.562 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.800 | 0 |
| reduce_max_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.447 | 0 |
| reduce_min_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.211 | 0 |
| reduce_min_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.929 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.204 | 7.204e-06 |
| reduce_mean_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.238 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.601 | 0 |
| add_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.586 | 0 |
| sub_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.577 | 0 |
| sub_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.676 | 0 |
| mul_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.497 | 0 |
| mul_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.657 | 0 |
| div_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.344 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.045 | 0 |
| rem_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.676 | 0 |
| rem_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.761 | 0 |
| pow_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.650 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda-jit | 2 | 15 | 2.542 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.925 | 0 |
| maximum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.041 | 0 |
| minimum_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.175 | 0 |
| minimum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.083 | 0 |
| neg_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.081 | 0 |
| neg_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.983 | 0 |
| abs_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.296 | 0 |
| abs_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.168 | 0 |
| sign_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.304 | 0 |
| sign_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.186 | 0 |
| conj_c32_n1048576 | jax-cuda-jit | 32 | 15 | 2.240 | 0 |
| conj_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.856 | 0 |
| exp_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.399 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.277 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.970 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.118 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.784 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.134 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.246 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.315 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.947 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda-jit | 4 | 15 | 2.668 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.116 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.815 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.012 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.028 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.128 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.052 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.103 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.279 | 0 |
| rsqrt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.000 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.918 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.150 | 0 |
| compare_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.288 | 0 |
| where_select_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.235 | 0 |
| where_select_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.050 | 0 |
| clamp_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.288 | 0 |
| clamp_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.036 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.481 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.417 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.421 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.588 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.432 | 0 |
| reduce_max_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.523 | 0 |
| reduce_min_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.386 | 0 |
| reduce_min_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.428 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.614 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.770 | 6.5e-09 |
| reduce_sum_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.778 | 0 |
| reduce_prod_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.163 | 0.001786 |
| reduce_max_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.705 | 0 |
| reduce_min_full_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.736 | 0 |
| compare_eq_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.273 | 0 |
| compare_lt_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.916 | 0 |
| compare_le_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.850 | 0 |
| compare_ge_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.055 | 0 |
| reduce_sum_full_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.599 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.209 | 1.774e-06 |
| reduce_max_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.517 | 0 |
| reduce_min_full_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.920 | 0 |
| compare_eq_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.180 | 0 |
| compare_lt_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.247 | 0 |
| compare_le_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.212 | 0 |
| compare_ge_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.099 | 0 |
| reduce_sum_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.437 | 0.003668 |
| reduce_prod_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.325 | 0.5323 |
| reduce_max_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.548 | 0 |
| reduce_min_full_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.726 | 0 |
| compare_eq_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.968 | 0 |
| compare_lt_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.192 | 0 |
| compare_le_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.051 | 0 |
| compare_ge_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.165 | 0 |
| reduce_sum_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.545 | 0.0006452 |
| reduce_prod_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.609 | 0.0001105 |
| reduce_max_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.646 | 0 |
| reduce_min_full_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.338 | 0 |
| compare_eq_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.460 | 0 |
| compare_lt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.295 | 0 |
| compare_le_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.298 | 0 |
| compare_ge_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.291 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: benchmarks/gpu/isolated_ops.yaml
timestamp: '2026-10-13T13:37:38Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 2858dcda46699788bc1df84d923a95ed93ef6469
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
  expected_cases: 156
  selected_cases: 156
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 249184323d09433d1fbf401284e39fa797c0252dceb20351bbcc151551d1b32c
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/structural.rs: 382fb5ffb6d86ac1a4a66723376808e9c95c8a8731570a5617f2ffed2a6e40a7
    src/bin/benchmark_gpu_public_api/linalg.rs: 84162923fa0d0772c88ec9c4eab169ad3a75e5d16ff76961d1541670dd8565c1
    src/bin/benchmark_gpu_public_api/reuse.rs: ad72b565373904275128d7beb8673bada08d922e8f310be6cedb0d2d8187898f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 4d05b9facc0b7e5a3ff31146fc42f98d680010264c6606a3154e9da2dfa6a0cc
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 71cb26510a35647e7aadc46827ca00d06a06e283eacfcba8ebe93d2e8e8add86
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/isolated_ops/20261013_133737/fixtures/manifest.json: 2cfba0631852f4503adbc24a8831452b0bbcb955241b3b0e2dc51ec206fbcefd
    target/release/benchmark_gpu_public_api: b505a2bdb3d0c5a7bffd1390af659a154753ea998ec68becd4779e137c9c7150
  reference_numpy_version: 2.4.2
```
