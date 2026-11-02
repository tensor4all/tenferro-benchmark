# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **1078**. Unsupported cells: **62**, including **50** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261102_210120/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

I32/I64 outputs require exact integer equality without conversion to floating point. Division and remainder truncate toward zero; axis-1 sum preserves the input dtype. Arithmetic fixtures avoid overflow, and I64 unary/order fixtures include values above 2**53 to detect loss of integer precision. A zero verification value in these rows means exact equality.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_f32_n1024 | 15.763 (0.119) | 52.865 (0.321) | 4.649 (0.014) | 49.855 (0.221) | 42.778 (0.395) |
| add_f64_n1024 | 16.436 (0.040) | 55.383 (0.387) | 4.781 (0.520) | 49.556 (0.293) | 42.472 (0.345) |
| sub_f32_n1024 | 15.971 (0.215) | 51.671 (1.252) | 5.047 (0.688) | 49.827 (0.287) | 42.541 (0.340) |
| sub_f64_n1024 | 16.682 (0.410) | 55.433 (0.211) | 4.766 (0.559) | 49.747 (0.501) | 42.571 (0.168) |
| mul_f32_n1024 | 16.062 (0.088) | 53.094 (0.736) | 4.652 (0.345) | 49.606 (0.300) | 41.874 (0.071) |
| mul_f64_n1024 | 16.799 (0.203) | 52.935 (2.497) | 4.678 (0.554) | 49.665 (0.151) | 41.884 (0.334) |
| div_f32_n1024 | 15.790 (0.146) | 50.447 (3.812) | 4.666 (0.317) | 42.065 (0.232) | 42.198 (0.280) |
| div_f64_n1024 | 16.584 (0.062) | 53.828 (2.292) | 4.791 (0.583) | 42.328 (0.281) | 42.359 (0.215) |
| rem_f32_n1024 | 35.950 (0.234) | 71.852 (4.955) | 4.683 (0.271) | 41.992 (0.280) | 42.315 (0.335) |
| rem_f64_n1024 | 36.225 (0.731) | 73.423 (0.785) | 5.006 (0.425) | 42.299 (0.222) | 42.113 (0.229) |
| pow_f32_n1024 | 24.407 (2.070) | 61.171 (1.087) | 5.119 (0.535) | 55.475 (0.323) | 42.015 (0.199) |
| pow_f64_n1024 | 25.739 (0.638) | 65.482 (0.980) | 7.390 (0.008) | 55.698 (0.264) | 42.240 (0.155) |
| maximum_f32_n1024 | 23.897 (1.576) | 59.230 (0.408) | 5.426 (0.106) | 49.605 (0.790) | 42.142 (0.123) |
| maximum_f64_n1024 | 28.589 (1.429) | 64.068 (1.161) | 4.900 (0.488) | 49.883 (0.225) | 42.038 (0.642) |
| minimum_f32_n1024 | 23.731 (0.535) | 60.273 (1.188) | 5.480 (0.068) | 50.110 (0.265) | 42.089 (0.171) |
| minimum_f64_n1024 | 28.289 (1.283) | 63.906 (0.994) | 5.502 (0.660) | 49.802 (0.133) | 42.206 (0.392) |
| neg_f32_n1024 | 14.216 (0.120) | 31.682 (1.688) | 5.053 (0.664) | 47.548 (0.206) | 39.655 (0.244) |
| neg_f64_n1024 | 14.474 (0.167) | 33.466 (0.231) | 5.235 (0.363) | 47.421 (0.205) | 39.787 (0.245) |
| abs_f32_n1024 | 13.874 (0.044) | 34.134 (0.268) | 5.314 (0.319) | 40.200 (0.247) | 39.838 (0.373) |
| abs_f64_n1024 | 14.483 (0.098) | 35.916 (0.146) | 5.301 (0.027) | 40.074 (0.362) | 39.454 (0.228) |
| sign_f32_n1024 | 13.936 (0.042) | 33.944 (0.329) | 4.856 (0.620) | 39.980 (0.170) | 39.593 (0.439) |
| sign_f64_n1024 | 14.489 (0.165) | 35.897 (0.206) | 4.526 (0.154) | 40.036 (0.333) | 39.820 (0.390) |
| conj_c32_n1024 | 14.716 (0.119) | 35.654 (0.147) | 5.349 (0.497) | 40.583 (0.409) | 39.645 (0.154) |
| conj_c64_n1024 | 14.803 (0.155) | 35.777 (0.181) | 5.197 (0.183) | 40.608 (0.208) | 39.599 (0.206) |
| exp_f32_n1024 | 13.872 (0.102) | 34.263 (0.658) | 4.397 (0.015) | 40.215 (0.207) | 39.790 (0.359) |
| exp_f64_n1024 | 14.774 (0.053) | 35.621 (0.170) | 4.383 (0.428) | 40.547 (0.329) | 39.991 (0.408) |
| expm1_f32_n1024 | 14.344 (0.046) | 34.062 (0.263) | 4.607 (0.475) | 40.130 (0.227) | 39.581 (0.202) |
| expm1_f64_n1024 | 14.931 (0.318) | 36.115 (0.134) | 4.872 (0.491) | 40.440 (0.261) | 39.680 (0.318) |
| log_f32_n1024 | 14.186 (0.837) | 33.770 (0.392) | 5.086 (0.021) | 40.309 (0.120) | 39.703 (0.329) |
| log_f64_n1024 | 15.311 (0.531) | 36.399 (0.129) | 4.462 (0.020) | 40.264 (0.375) | 40.074 (0.213) |
| log1p_f32_n1024 | 13.942 (0.809) | 34.153 (0.290) | 4.705 (0.369) | 39.962 (0.177) | 39.686 (0.356) |
| log1p_f64_n1024 | 14.860 (0.404) | 36.890 (0.127) | 4.521 (0.012) | 40.075 (0.199) | 39.744 (0.432) |
| erf_f32_n1024 | 13.913 (0.378) | 33.960 (0.249) | 4.982 (0.576) | 63.459 (0.322) | 39.929 (0.383) |
| erf_f64_n1024 | 15.163 (0.360) | 36.367 (0.207) | 4.642 (0.478) | 63.429 (0.820) | 40.074 (0.313) |
| sin_f32_n1024 | 14.459 (0.820) | 32.373 (0.867) | 4.417 (0.329) | 40.246 (0.100) | 39.971 (0.266) |
| sin_f64_n1024 | 16.322 (0.055) | 36.998 (0.084) | 5.122 (0.015) | 40.518 (0.261) | 40.374 (0.242) |
| cos_f32_n1024 | 15.201 (0.035) | 32.515 (0.427) | 4.893 (0.547) | 40.158 (0.346) | 39.827 (0.412) |
| cos_f64_n1024 | 16.417 (0.326) | 34.123 (0.159) | 5.146 (0.020) | 39.912 (0.194) | 40.654 (0.349) |
| tanh_f32_n1024 | 15.469 (0.177) | 31.375 (0.303) | 4.446 (0.296) | 39.549 (0.218) | 39.621 (0.411) |
| tanh_f64_n1024 | 16.469 (0.055) | 33.465 (0.124) | 4.973 (0.723) | 39.618 (0.258) | 39.806 (0.223) |
| sqrt_f32_n1024 | 15.340 (0.087) | 31.552 (0.277) | 5.081 (0.020) | 39.535 (0.155) | 39.619 (0.309) |
| sqrt_f64_n1024 | 16.242 (0.159) | 33.186 (0.114) | 4.356 (0.571) | 39.707 (0.235) | 39.765 (0.114) |
| rsqrt_f32_n1024 | 15.420 (0.035) | 31.625 (0.406) | 4.473 (0.090) | 64.146 (0.162) | 39.776 (0.284) |
| rsqrt_f64_n1024 | 15.906 (0.383) | 33.351 (0.150) | 5.094 (0.008) | 64.371 (0.215) | 39.740 (0.203) |
| compare_f32_n1024 | unsupported | 49.123 (0.168) | 5.271 (0.005) | 41.822 (0.211) | 41.880 (0.220) |
| compare_f64_n1024 | unsupported | 50.990 (0.233) | 4.572 (0.513) | 41.886 (0.207) | 42.287 (0.253) |
| where_select_f32_n1024 | 16.026 (0.175) | 71.034 (0.672) | 5.960 (0.475) | 48.729 (0.308) | 43.058 (0.463) |
| where_select_f64_n1024 | 16.807 (0.095) | 74.765 (0.375) | 6.551 (0.585) | 48.717 (0.189) | 43.339 (0.180) |
| clamp_f32_n1024 | 50.287 (0.369) | 110.879 (3.173) | 5.935 (0.638) | 43.358 (0.188) | 43.157 (0.331) |
| clamp_f64_n1024 | 52.704 (0.219) | 114.785 (0.736) | 6.155 (0.019) | 43.318 (0.211) | 43.521 (0.312) |
| reduce_sum_f32_n1024 | 34.680 (0.149) | 54.505 (0.362) | 6.298 (0.896) | 45.188 (0.691) | 39.617 (0.464) |
| reduce_sum_f64_n1024 | 35.515 (0.131) | 56.833 (0.206) | 6.455 (0.534) | 43.762 (0.967) | 39.692 (0.235) |
| reduce_prod_f32_n1024 | 33.779 (0.069) | 54.830 (0.187) | 6.681 (0.106) | 44.567 (0.218) | 39.954 (0.190) |
| reduce_prod_f64_n1024 | 34.535 (0.412) | 56.443 (0.187) | 7.174 (0.749) | 44.623 (0.324) | 39.924 (0.102) |
| reduce_max_f32_n1024 | 38.262 (0.209) | 61.655 (0.299) | 7.718 (0.577) | 44.562 (0.264) | 39.573 (0.411) |
| reduce_max_f64_n1024 | 39.848 (0.918) | 60.971 (0.113) | 7.637 (0.992) | 44.548 (0.207) | 39.977 (0.137) |
| reduce_min_f32_n1024 | 38.894 (0.254) | 62.457 (0.258) | 7.239 (1.105) | 44.361 (0.306) | 39.832 (0.227) |
| reduce_min_f64_n1024 | 38.387 (2.175) | 60.503 (0.226) | 6.766 (1.203) | 44.151 (0.285) | 40.083 (0.335) |
| reduce_sum_squares_f32_n1024 | 31.519 (0.101) | 54.290 (0.435) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1024 | 32.245 (0.101) | 56.805 (2.311) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1024 | unsupported | unsupported | 6.493 (0.912) | 47.454 (0.263) | 43.330 (0.178) |
| reduce_mean_f64_n1024 | unsupported | unsupported | 7.079 (0.487) | 48.036 (0.210) | 43.232 (0.278) |
| add_f32_n1048576 | 22.765 (0.142) | 68.744 (0.235) | 5.654 (0.327) | 50.838 (0.205) | 43.653 (0.305) |
| add_f64_n1048576 | 34.540 (0.261) | 95.539 (3.676) | 12.779 (0.266) | 50.913 (0.409) | 44.191 (0.275) |
| sub_f32_n1048576 | 22.766 (0.051) | 69.538 (0.200) | 5.709 (0.574) | 50.349 (0.199) | 43.901 (0.198) |
| sub_f64_n1048576 | 34.553 (0.238) | 93.797 (0.331) | 13.142 (0.140) | 51.132 (0.625) | 43.786 (0.362) |
| mul_f32_n1048576 | 22.890 (0.658) | 68.805 (0.164) | 6.591 (0.239) | 50.429 (0.231) | 43.436 (0.335) |
| mul_f64_n1048576 | 34.523 (1.016) | 94.069 (0.546) | 12.894 (0.096) | 50.939 (0.476) | 43.930 (0.266) |
| div_f32_n1048576 | 22.849 (0.865) | 70.862 (2.874) | 5.656 (0.039) | 43.095 (0.186) | 43.755 (0.312) |
| div_f64_n1048576 | 34.625 (0.348) | 94.640 (2.841) | 13.814 (0.211) | 43.414 (0.344) | 43.766 (0.402) |
| rem_f32_n1048576 | 63.624 (3.588) | 93.825 (0.936) | 5.857 (0.938) | 42.665 (0.327) | 43.486 (0.473) |
| rem_f64_n1048576 | 89.663 (1.143) | 111.969 (0.912) | 12.628 (0.185) | 42.889 (0.300) | 43.971 (0.417) |
| pow_f32_n1048576 | 61.292 (0.769) | 86.697 (0.325) | 11.033 (0.025) | 56.037 (0.291) | 43.344 (0.342) |
| pow_f64_n1048576 | 92.087 (0.762) | 115.954 (1.064) | 29.260 (0.068) | 54.526 (0.919) | 41.929 (0.645) |
| maximum_f32_n1048576 | 50.892 (1.147) | 83.104 (0.368) | 6.843 (0.334) | 50.101 (0.293) | 43.424 (0.423) |
| maximum_f64_n1048576 | 79.158 (1.126) | 102.756 (1.210) | 12.346 (0.080) | 50.556 (0.440) | 44.288 (0.463) |
| minimum_f32_n1048576 | 51.015 (0.850) | 82.185 (0.532) | 6.905 (0.137) | 50.103 (0.247) | 43.301 (0.529) |
| minimum_f64_n1048576 | 80.158 (1.298) | 100.413 (0.926) | 12.744 (0.123) | 50.451 (0.184) | 44.066 (0.546) |
| neg_f32_n1048576 | 22.557 (0.149) | 46.728 (0.315) | 6.686 (0.215) | 48.415 (0.318) | 40.985 (0.275) |
| neg_f64_n1048576 | 28.196 (0.035) | 57.343 (0.193) | 7.891 (0.085) | 48.514 (0.372) | 41.664 (0.540) |
| abs_f32_n1048576 | 22.394 (0.276) | 48.649 (0.051) | 7.678 (0.531) | 40.708 (0.191) | 40.998 (0.214) |
| abs_f64_n1048576 | 28.193 (0.030) | 56.782 (0.264) | 8.094 (0.074) | 40.914 (0.360) | 41.560 (0.291) |
| sign_f32_n1048576 | 22.851 (0.138) | 49.252 (0.153) | 6.005 (0.438) | 40.515 (0.237) | 41.036 (0.163) |
| sign_f64_n1048576 | 28.190 (0.029) | 56.919 (0.159) | 7.879 (0.065) | 41.102 (0.439) | 41.466 (0.279) |
| conj_c32_n1048576 | 28.201 (0.024) | 57.310 (0.357) | 7.978 (0.119) | 41.418 (0.318) | 41.462 (0.383) |
| conj_c64_n1048576 | 57.120 (0.047) | 123.488 (0.826) | 25.463 (0.187) | 41.738 (1.113) | 42.189 (1.174) |
| exp_f32_n1048576 | 22.725 (0.129) | 48.619 (0.115) | 5.208 (0.075) | 40.814 (0.185) | 41.571 (0.417) |
| exp_f64_n1048576 | 28.524 (0.168) | 55.413 (0.162) | 8.430 (0.081) | 41.260 (0.203) | 41.520 (0.397) |
| expm1_f32_n1048576 | 23.698 (0.126) | 47.584 (0.125) | 5.871 (0.018) | 40.785 (0.294) | 41.226 (0.259) |
| expm1_f64_n1048576 | 29.307 (0.085) | 56.985 (0.163) | 9.956 (0.039) | 41.166 (0.286) | 41.254 (0.473) |
| log_f32_n1048576 | 23.749 (0.095) | 47.453 (0.069) | 5.610 (0.025) | 40.580 (0.228) | 41.150 (0.421) |
| log_f64_n1048576 | 31.990 (0.237) | 59.457 (0.103) | 12.744 (0.037) | 41.011 (0.363) | 41.367 (0.429) |
| log1p_f32_n1048576 | 23.440 (0.224) | 46.579 (1.687) | 5.614 (0.029) | 40.881 (0.220) | 41.217 (0.350) |
| log1p_f64_n1048576 | 32.489 (0.135) | 59.859 (1.635) | 13.177 (0.059) | 41.249 (0.379) | 41.516 (0.282) |
| erf_f32_n1048576 | 23.293 (0.076) | 47.402 (0.203) | 5.281 (0.088) | 63.914 (0.380) | 41.245 (0.232) |
| erf_f64_n1048576 | 33.277 (0.206) | 63.670 (0.202) | 14.450 (0.029) | 64.443 (0.920) | 41.367 (0.159) |
| sin_f32_n1048576 | 22.291 (0.139) | 47.268 (0.094) | 5.870 (0.025) | 40.788 (0.460) | 41.402 (0.262) |
| sin_f64_n1048576 | 28.922 (1.078) | 59.456 (0.271) | 10.152 (0.041) | 41.579 (0.341) | 41.647 (0.445) |
| cos_f32_n1048576 | 23.746 (0.053) | 49.655 (0.554) | 5.827 (0.039) | 41.471 (0.243) | 41.229 (0.346) |
| cos_f64_n1048576 | 29.562 (0.455) | 58.962 (0.221) | 10.152 (0.076) | 41.575 (0.415) | 41.561 (0.427) |
| tanh_f32_n1048576 | 23.127 (0.072) | 48.940 (0.112) | 5.271 (0.082) | 40.812 (0.204) | 41.092 (0.335) |
| tanh_f64_n1048576 | 28.488 (0.319) | 58.027 (0.550) | 8.337 (0.041) | 41.392 (0.314) | 41.709 (0.580) |
| sqrt_f32_n1048576 | 22.751 (0.064) | 48.715 (0.593) | 5.217 (0.072) | 40.670 (0.310) | 41.155 (0.356) |
| sqrt_f64_n1048576 | 28.200 (0.022) | 57.360 (0.240) | 7.966 (0.055) | 41.105 (0.283) | 41.585 (0.350) |
| rsqrt_f32_n1048576 | 21.648 (0.087) | 49.728 (0.538) | 5.286 (0.100) | 65.820 (0.394) | 41.044 (0.369) |
| rsqrt_f64_n1048576 | 28.143 (0.025) | 58.230 (0.157) | 7.985 (0.069) | 65.491 (0.441) | 41.291 (0.374) |
| compare_f32_n1048576 | 19.187 (0.097) | 71.275 (0.273) | 5.347 (0.098) | 43.112 (0.230) | 43.564 (0.423) |
| compare_f64_n1048576 | 20.107 (0.090) | 83.440 (0.517) | 6.939 (0.018) | 42.816 (0.252) | 43.548 (0.234) |
| where_select_f32_n1048576 | 24.573 (0.131) | 94.859 (2.759) | 6.931 (0.145) | 49.643 (0.124) | 44.538 (0.647) |
| where_select_f64_n1048576 | 35.463 (0.215) | 113.514 (2.335) | 13.631 (0.038) | 50.117 (0.481) | 44.984 (0.537) |
| clamp_f32_n1048576 | 66.531 (0.201) | 127.168 (3.057) | 7.638 (0.025) | 44.336 (0.160) | 44.783 (0.208) |
| clamp_f64_n1048576 | 87.473 (2.239) | 150.953 (0.523) | 10.376 (0.020) | 44.973 (0.334) | 45.036 (0.410) |
| reduce_sum_f32_n1048576 | 57.720 (3.443) | 86.748 (0.472) | 6.427 (0.489) | 46.421 (0.163) | 41.047 (0.528) |
| reduce_sum_f64_n1048576 | 63.184 (0.557) | 90.585 (1.142) | 7.143 (0.089) | 89.708 (38.821) | 41.132 (0.213) |
| reduce_prod_f32_n1048576 | 57.116 (3.110) | 90.493 (0.798) | 6.175 (0.013) | 53.389 (11.107) | 41.039 (0.202) |
| reduce_prod_f64_n1048576 | 61.609 (0.174) | 93.271 (0.189) | 7.070 (0.010) | 71.131 (38.589) | 41.134 (0.175) |
| reduce_max_f32_n1048576 | 59.530 (0.415) | 88.178 (3.405) | 6.378 (0.132) | 45.784 (0.284) | 41.172 (0.254) |
| reduce_max_f64_n1048576 | 62.409 (0.313) | 94.491 (1.065) | 7.266 (0.006) | 45.430 (21.145) | 41.059 (6.798) |
| reduce_min_f32_n1048576 | 59.009 (0.177) | 85.671 (3.534) | 6.382 (0.014) | 45.126 (0.042) | 41.441 (0.225) |
| reduce_min_f64_n1048576 | 61.937 (0.371) | 88.934 (2.181) | 7.261 (0.007) | 45.365 (0.256) | 41.025 (0.223) |
| reduce_sum_squares_f32_n1048576 | 60.541 (0.262) | 89.191 (0.493) | unsupported | unsupported | unsupported |
| reduce_sum_squares_f64_n1048576 | 62.962 (0.380) | 94.604 (0.174) | unsupported | unsupported | unsupported |
| reduce_mean_f32_n1048576 | unsupported | unsupported | 6.430 (0.453) | 48.232 (0.261) | 44.204 (0.148) |
| reduce_mean_f64_n1048576 | unsupported | unsupported | 7.002 (0.021) | 48.064 (0.522) | 44.695 (0.407) |
| reduce_sum_full_f32_n1024 | 42.357 (0.198) | 64.756 (0.332) | 6.024 (0.916) | 44.404 (0.230) | 39.638 (0.342) |
| reduce_prod_full_f32_n1024 | 42.913 (0.156) | 64.921 (0.304) | 5.602 (0.492) | 43.598 (0.407) | 39.569 (0.203) |
| reduce_max_full_f32_n1024 | 43.426 (0.062) | 65.613 (0.502) | 6.369 (0.551) | 44.116 (0.196) | 39.593 (0.170) |
| reduce_min_full_f32_n1024 | 40.599 (1.558) | 64.218 (0.114) | 5.544 (0.098) | 44.080 (0.128) | 39.475 (0.085) |
| compare_eq_f32_n1024 | unsupported | 52.704 (0.184) | 4.612 (0.420) | 41.648 (0.162) | 41.775 (0.179) |
| compare_lt_f32_n1024 | unsupported | 52.418 (1.361) | 4.610 (0.183) | 41.341 (0.382) | 41.971 (0.173) |
| compare_le_f32_n1024 | unsupported | 52.532 (0.219) | 5.283 (0.621) | 41.616 (0.210) | 42.114 (0.234) |
| compare_ge_f32_n1024 | unsupported | 53.469 (0.369) | 4.663 (0.372) | 41.503 (0.327) | 42.018 (0.185) |
| reduce_sum_full_f64_n1024 | 42.626 (2.164) | 60.795 (1.590) | 5.670 (0.072) | 43.251 (0.189) | 39.840 (0.234) |
| reduce_prod_full_f64_n1024 | 39.479 (0.133) | 60.528 (2.209) | 5.110 (0.028) | 44.248 (0.207) | 40.073 (0.409) |
| reduce_max_full_f64_n1024 | 41.363 (0.121) | 62.801 (3.822) | 5.532 (0.161) | 45.145 (0.174) | 41.329 (0.284) |
| reduce_min_full_f64_n1024 | 40.609 (0.134) | 62.954 (4.009) | 5.582 (0.432) | 45.637 (0.570) | 41.258 (0.337) |
| compare_eq_f64_n1024 | unsupported | 55.327 (2.985) | 4.648 (0.018) | 41.579 (0.367) | 42.338 (0.340) |
| compare_lt_f64_n1024 | unsupported | 53.569 (2.791) | 4.623 (0.278) | 41.604 (0.425) | 42.004 (0.251) |
| compare_le_f64_n1024 | unsupported | 51.497 (1.742) | 4.631 (0.011) | 42.577 (0.188) | 42.189 (0.160) |
| compare_ge_f64_n1024 | unsupported | 51.144 (2.349) | 5.169 (0.667) | 42.433 (0.180) | 42.466 (0.187) |
| reduce_sum_full_f32_n1048576 | 56.374 (3.193) | 81.086 (0.294) | 9.652 (0.173) | 49.990 (0.190) | 45.533 (0.270) |
| reduce_prod_full_f32_n1048576 | 54.902 (2.768) | 84.047 (2.453) | 9.665 (0.062) | 50.068 (0.411) | 45.107 (0.264) |
| reduce_max_full_f32_n1048576 | 79.248 (3.172) | 107.026 (1.348) | 10.262 (0.027) | 49.567 (0.210) | 45.108 (0.111) |
| reduce_min_full_f32_n1048576 | 81.678 (0.283) | 110.183 (0.152) | 10.255 (0.016) | 49.671 (0.164) | 45.212 (0.170) |
| compare_eq_f32_n1048576 | 19.123 (0.152) | 70.183 (2.130) | 5.414 (0.031) | 43.524 (0.450) | 43.666 (0.533) |
| compare_lt_f32_n1048576 | 20.224 (0.092) | 68.319 (3.367) | 4.805 (0.013) | 43.206 (0.277) | 43.629 (0.236) |
| compare_le_f32_n1048576 | 20.697 (0.331) | 68.783 (2.763) | 4.820 (0.012) | 43.195 (0.297) | 43.660 (0.394) |
| compare_ge_f32_n1048576 | 20.357 (0.360) | 71.934 (0.437) | 4.832 (0.194) | 43.268 (0.374) | 43.463 (0.379) |
| reduce_sum_full_f64_n1048576 | 58.411 (0.614) | 88.297 (0.259) | 10.968 (0.015) | 50.482 (0.300) | 45.574 (0.288) |
| reduce_prod_full_f64_n1048576 | 57.230 (0.192) | 88.483 (0.201) | 11.038 (0.028) | 49.841 (0.256) | 45.648 (0.222) |
| reduce_max_full_f64_n1048576 | 85.185 (0.320) | 116.860 (0.975) | 11.538 (0.019) | 49.572 (4.655) | 45.752 (0.304) |
| reduce_min_full_f64_n1048576 | 84.493 (0.420) | 116.319 (0.505) | 11.540 (0.021) | 50.131 (0.623) | 45.570 (0.196) |
| compare_eq_f64_n1048576 | 21.841 (0.059) | 80.742 (2.861) | 6.788 (0.008) | 43.077 (0.316) | 43.409 (0.243) |
| compare_lt_f64_n1048576 | 21.618 (0.316) | 80.627 (1.469) | 6.781 (0.019) | 43.225 (0.303) | 43.621 (0.153) |
| compare_le_f64_n1048576 | 21.588 (0.120) | 81.484 (2.484) | 6.768 (0.009) | 43.181 (0.119) | 43.617 (0.317) |
| compare_ge_f64_n1048576 | 21.226 (0.064) | 80.959 (2.195) | 6.778 (0.012) | 43.060 (0.231) | 43.803 (0.361) |
| reduce_mean_full_f32_n1024 | unsupported | unsupported | 6.155 (0.842) | 46.749 (0.194) | 42.716 (0.185) |
| reduce_sum_squares_full_f32_n1024 | 43.324 (0.795) | 60.819 (2.252) | 7.915 (0.698) | 40.270 (0.330) | 39.569 (0.249) |
| reduce_mean_full_f64_n1024 | unsupported | unsupported | 5.462 (0.030) | 47.461 (0.305) | 42.951 (0.189) |
| reduce_sum_squares_full_f64_n1024 | 44.006 (0.150) | 65.877 (0.271) | 7.882 (0.268) | 42.265 (0.172) | 39.758 (0.295) |
| reduce_mean_full_f32_n1048576 | unsupported | unsupported | 9.648 (0.313) | 51.440 (0.119) | 47.854 (0.234) |
| reduce_sum_squares_full_f32_n1048576 | 56.943 (0.487) | 85.411 (0.161) | 9.675 (0.015) | 47.837 (0.306) | 45.017 (0.203) |
| reduce_mean_full_f64_n1048576 | unsupported | unsupported | 11.032 (0.029) | 52.833 (0.225) | 48.406 (0.369) |
| reduce_sum_squares_full_f64_n1048576 | 54.546 (0.251) | 85.066 (2.146) | 11.036 (0.014) | 48.644 (0.231) | 45.544 (0.129) |
| scale_real_f32_n1024 | unsupported | unsupported | 5.792 (0.012) | 49.287 (0.160) | 42.437 (0.193) |
| scale_real_f64_n1024 | unsupported | unsupported | 5.801 (0.219) | 49.331 (0.154) | 42.321 (0.476) |
| scale_real_c32_n1024 | unsupported | unsupported | 5.352 (0.763) | 49.519 (0.244) | 42.757 (0.160) |
| scale_real_c64_n1024 | unsupported | unsupported | 5.013 (0.259) | 49.386 (0.159) | 42.655 (0.302) |
| scale_complex_c32_n1024 | unsupported | unsupported | 5.007 (0.114) | 49.006 (0.503) | 42.215 (0.232) |
| scale_complex_c64_n1024 | unsupported | unsupported | 5.788 (0.786) | 49.249 (0.101) | 42.296 (0.288) |
| scale_real_f32_n1048576 | unsupported | unsupported | 7.248 (0.028) | 50.204 (0.251) | 43.740 (0.411) |
| scale_real_f64_n1048576 | unsupported | unsupported | 9.502 (0.039) | 51.100 (0.299) | 43.984 (0.484) |
| scale_real_c32_n1048576 | unsupported | unsupported | 9.457 (0.083) | 51.275 (0.341) | 44.270 (0.462) |
| scale_real_c64_n1048576 | unsupported | unsupported | 25.774 (0.262) | 50.810 (0.635) | 44.345 (1.155) |
| scale_complex_c32_n1048576 | unsupported | unsupported | 9.520 (0.070) | 52.040 (0.430) | 44.566 (0.182) |
| scale_complex_c64_n1048576 | unsupported | unsupported | 25.857 (0.197) | 51.619 (0.713) | 44.053 (2.249) |
| add_i32_n1024 | 16.017 (0.200) | 53.512 (0.197) | 5.472 (0.042) | 49.373 (0.436) | 42.510 (0.504) |
| sub_i32_n1024 | 16.215 (0.128) | 52.029 (3.777) | 5.505 (0.038) | 49.077 (0.232) | 42.016 (0.107) |
| mul_i32_n1024 | 15.903 (0.200) | 52.790 (1.117) | 5.401 (0.525) | 49.380 (0.705) | 42.384 (0.385) |
| div_i32_n1024 | 36.867 (0.284) | 77.505 (0.630) | 5.901 (0.074) | 69.283 (0.569) | 42.009 (0.326) |
| rem_i32_n1024 | 36.043 (0.990) | 74.918 (2.666) | 5.079 (0.779) | 42.239 (0.186) | 42.566 (0.277) |
| maximum_i32_n1024 | 24.437 (0.130) | 60.150 (1.079) | 4.727 (0.465) | 48.597 (0.169) | 42.151 (0.451) |
| minimum_i32_n1024 | 24.470 (0.118) | 61.699 (0.231) | 5.010 (0.635) | 49.115 (0.131) | 42.255 (0.155) |
| neg_i32_n1024 | 15.677 (0.204) | 34.226 (2.137) | 4.927 (0.703) | 46.666 (0.239) | 39.390 (0.250) |
| abs_i32_n1024 | 15.641 (0.108) | 34.303 (0.394) | 5.237 (0.341) | 39.607 (0.318) | 39.509 (0.284) |
| sign_i32_n1024 | 15.721 (0.110) | 34.141 (0.599) | 4.507 (0.465) | 39.846 (0.259) | 39.673 (0.358) |
| reduce_sum_i32_n1024 | 34.970 (0.152) | 55.317 (0.679) | 6.708 (1.048) | 44.384 (0.680) | 38.839 (0.268) |
| reduce_max_i32_n1024 | 34.424 (0.200) | 55.370 (0.172) | 5.805 (0.336) | 43.123 (0.112) | 38.703 (0.277) |
| reduce_min_i32_n1024 | 34.196 (0.108) | 54.963 (0.288) | 5.995 (0.282) | 43.404 (0.124) | 38.733 (0.188) |
| add_i64_n1024 | 16.569 (0.054) | 54.912 (0.387) | 4.660 (0.445) | 49.825 (0.231) | 42.129 (0.157) |
| sub_i64_n1024 | 16.933 (0.095) | 52.142 (0.365) | 4.681 (0.273) | 49.774 (0.216) | 42.331 (0.155) |
| mul_i64_n1024 | 16.337 (0.053) | 51.947 (0.222) | 4.675 (0.138) | 49.856 (0.176) | 42.426 (0.238) |
| div_i64_n1024 | 37.169 (0.170) | 74.583 (1.807) | 5.446 (0.012) | 69.232 (0.512) | 42.316 (0.184) |
| rem_i64_n1024 | 37.105 (0.833) | 77.051 (4.434) | 5.414 (0.089) | 42.087 (0.171) | 42.480 (0.279) |
| maximum_i64_n1024 | 25.332 (0.094) | 63.970 (0.311) | 5.046 (0.510) | 49.615 (0.192) | 42.429 (0.304) |
| minimum_i64_n1024 | 25.211 (0.208) | 64.637 (0.659) | 4.710 (0.011) | 49.646 (0.205) | 42.197 (0.173) |
| neg_i64_n1024 | 16.204 (0.163) | 35.864 (0.649) | 5.260 (0.013) | 47.285 (0.313) | 39.979 (0.189) |
| abs_i64_n1024 | 15.328 (0.637) | 35.838 (0.118) | 5.278 (0.188) | 39.956 (0.196) | 39.656 (0.241) |
| sign_i64_n1024 | 15.220 (0.741) | 33.700 (0.106) | 4.526 (0.108) | 39.971 (0.303) | 39.756 (0.350) |
| reduce_sum_i64_n1024 | 34.580 (1.822) | 55.117 (1.624) | 6.444 (0.564) | 45.791 (0.258) | 39.718 (0.080) |
| reduce_max_i64_n1024 | 32.938 (1.001) | 56.638 (0.193) | 6.933 (0.636) | 44.384 (0.124) | 39.652 (0.135) |
| reduce_min_i64_n1024 | 32.298 (0.235) | 56.478 (0.155) | 6.796 (0.723) | 44.537 (0.232) | 39.865 (0.369) |
| add_i32_n1048576 | 22.839 (0.500) | 75.606 (0.245) | 5.624 (0.033) | 50.472 (0.177) | 43.545 (0.279) |
| sub_i32_n1048576 | 22.933 (0.060) | 74.967 (0.418) | 5.684 (0.063) | 50.194 (0.166) | 43.681 (0.300) |
| mul_i32_n1048576 | 23.012 (0.103) | 74.517 (2.820) | 5.652 (0.087) | 50.533 (0.337) | 43.604 (0.202) |
| div_i32_n1048576 | 66.076 (0.384) | 100.909 (1.535) | 6.495 (0.028) | 70.695 (0.656) | 43.584 (0.328) |
| rem_i32_n1048576 | 65.360 (0.644) | 96.995 (1.849) | 6.756 (0.016) | 43.127 (0.287) | 43.580 (0.229) |
| maximum_i32_n1048576 | 55.732 (0.559) | 85.559 (0.506) | 5.690 (0.100) | 50.206 (0.291) | 43.551 (0.291) |
| minimum_i32_n1048576 | 56.071 (0.386) | 84.825 (0.663) | 5.630 (0.061) | 50.226 (0.224) | 43.708 (0.137) |
| neg_i32_n1048576 | 21.734 (0.066) | 49.556 (1.638) | 5.288 (0.102) | 48.799 (0.259) | 40.951 (0.146) |
| abs_i32_n1048576 | 21.762 (0.100) | 49.510 (1.159) | 6.222 (0.114) | 41.106 (0.436) | 40.819 (0.112) |
| sign_i32_n1048576 | 22.338 (0.152) | 50.157 (1.269) | 5.344 (0.117) | 40.726 (0.237) | 41.214 (0.268) |
| reduce_sum_i32_n1048576 | 57.803 (1.044) | 90.246 (0.492) | 7.529 (0.088) | 45.457 (0.268) | 40.245 (0.161) |
| reduce_max_i32_n1048576 | 57.550 (0.795) | 91.443 (0.261) | 6.937 (0.036) | 44.383 (0.806) | 40.288 (0.211) |
| reduce_min_i32_n1048576 | 58.890 (1.796) | 90.434 (0.310) | 7.264 (0.060) | 51.836 (7.287) | 40.145 (0.161) |
| add_i64_n1048576 | 34.564 (0.252) | 98.019 (0.425) | 12.830 (0.173) | 56.692 (7.211) | 43.780 (0.353) |
| sub_i64_n1048576 | 35.478 (0.323) | 98.962 (0.380) | 12.862 (0.118) | 54.413 (4.389) | 43.750 (0.473) |
| mul_i64_n1048576 | 35.488 (0.181) | 98.789 (1.597) | 12.998 (0.157) | 57.212 (6.411) | 43.818 (0.445) |
| div_i64_n1048576 | 94.048 (0.710) | 121.009 (0.593) | 16.767 (0.063) | 71.306 (3.348) | 42.505 (0.412) |
| rem_i64_n1048576 | 93.306 (0.881) | 121.903 (1.061) | 16.376 (0.042) | 43.819 (5.498) | 44.041 (0.501) |
| maximum_i64_n1048576 | 80.445 (0.437) | 213.557 (0.959) | 12.372 (0.111) | 56.462 (9.317) | 43.995 (0.492) |
| minimum_i64_n1048576 | 79.838 (0.613) | 214.341 (1.276) | 12.581 (0.085) | 52.580 (2.994) | 43.813 (0.312) |
| neg_i64_n1048576 | 28.164 (0.027) | 56.617 (2.003) | 7.779 (0.069) | 52.422 (5.215) | 41.072 (0.343) |
| abs_i64_n1048576 | 28.193 (0.084) | 56.397 (1.857) | 7.912 (0.030) | 52.599 (3.792) | 41.578 (0.222) |
| sign_i64_n1048576 | 28.268 (0.203) | 58.629 (0.184) | 7.786 (0.082) | 53.925 (1.441) | 41.321 (0.263) |
| reduce_sum_i64_n1048576 | 62.854 (0.709) | 94.351 (0.813) | 7.740 (0.201) | 55.097 (5.359) | 41.693 (0.598) |
| reduce_max_i64_n1048576 | 62.165 (0.729) | 93.850 (0.760) | 7.169 (0.007) | 45.838 (8.853) | 41.261 (0.531) |
| reduce_min_i64_n1048576 | 61.342 (0.256) | 94.735 (0.348) | 7.233 (0.014) | 64.153 (15.117) | 41.810 (0.512) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.018 | 0 |
| add_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.104 | 0 |
| sub_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.044 | 0 |
| sub_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.135 | 0 |
| mul_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.056 | 0 |
| mul_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.150 | 0 |
| div_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.021 | 0 |
| div_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.123 | 0 |
| rem_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.301 | 0 |
| rem_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.318 | 0 |
| pow_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.124 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.295 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.059 | 0 |
| maximum_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.659 | 0 |
| minimum_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.038 | 0 |
| minimum_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.621 | 0 |
| neg_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.639 | 0 |
| neg_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.705 | 0 |
| abs_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.552 | 0 |
| abs_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.708 | 0 |
| sign_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.568 | 0 |
| sign_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.709 | 0 |
| conj_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.767 | 0 |
| conj_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.895 | 0 |
| exp_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.551 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.891 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.672 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.911 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.632 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.920 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.569 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.804 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.562 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.941 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.701 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.089 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.891 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.101 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 1.980 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.108 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.927 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.079 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.947 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.036 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. GPU1 diagnostic 20261014_010820 on 9655b4e0 (published CubeCL provider) reproduced initial Gt F32 last/first-ten batch growth of 2.902, 3.554 and 2.710 in three independent processes. Other directions are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md and compare_lifetime_counters.md. Keep excluded from steady-state timing; upstream tenferro-rs #1928 investigation remains open, not a numerical-support failure. |
| compare_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. GPU1 diagnostic 20261014_010820 on 9655b4e0 (published CubeCL provider) reproduced initial Gt F32 last/first-ten batch growth of 2.902, 3.554 and 2.710 in three independent processes. Other directions are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md and compare_lifetime_counters.md. Keep excluded from steady-state timing; upstream tenferro-rs #1928 investigation remains open, not a numerical-support failure. |
| where_select_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.051 | 0 |
| where_select_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.151 | 0 |
| clamp_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.218 | 0 |
| clamp_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.373 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.220 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.273 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.162 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.210 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.449 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.550 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.489 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.457 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.017 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.064 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.457 | 0 |
| add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.105 | 0 |
| sub_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.457 | 0 |
| sub_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.106 | 0 |
| mul_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.465 | 0 |
| mul_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.105 | 0 |
| div_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.462 | 0 |
| div_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.108 | 0 |
| rem_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 4.072 | 0 |
| rem_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.869 | 0 |
| pow_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.923 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.947 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.257 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.533 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.265 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.565 | 0 |
| neg_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.444 | 0 |
| neg_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.902 | 0 |
| abs_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.433 | 0 |
| abs_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.902 | 0 |
| sign_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.462 | 0 |
| sign_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.902 | 0 |
| conj_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.902 | 0 |
| conj_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.914 | 0 |
| exp_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.454 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.913 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.517 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.938 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.520 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.024 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.500 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.040 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.491 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.065 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.427 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.926 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.520 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.946 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.480 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.912 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.456 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.902 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.385 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.901 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.456 | 0 |
| compare_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.574 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.573 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.135 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.129 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.799 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.694 | 0.001926 |
| reduce_sum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.022 | 3.391e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.655 | 0.004885 |
| reduce_prod_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.943 | 3.55e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.810 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.997 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.777 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.982 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.875 | 0.0003531 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.015 | 3.294e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_full_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.711 | 0.0001333 |
| reduce_prod_full_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.746 | 0.002738 |
| reduce_max_full_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.779 | 0 |
| reduce_min_full_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.598 | 0 |
| compare_eq_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. GPU1 diagnostic 20261014_010820 on 9655b4e0 (published CubeCL provider) reproduced initial Gt F32 last/first-ten batch growth of 2.902, 3.554 and 2.710 in three independent processes. Other directions are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md and compare_lifetime_counters.md. Keep excluded from steady-state timing; upstream tenferro-rs #1928 investigation remains open, not a numerical-support failure. |
| compare_lt_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. GPU1 diagnostic 20261014_010820 on 9655b4e0 (published CubeCL provider) reproduced initial Gt F32 last/first-ten batch growth of 2.902, 3.554 and 2.710 in three independent processes. Other directions are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md and compare_lifetime_counters.md. Keep excluded from steady-state timing; upstream tenferro-rs #1928 investigation remains open, not a numerical-support failure. |
| compare_le_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. GPU1 diagnostic 20261014_010820 on 9655b4e0 (published CubeCL provider) reproduced initial Gt F32 last/first-ten batch growth of 2.902, 3.554 and 2.710 in three independent processes. Other directions are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md and compare_lifetime_counters.md. Keep excluded from steady-state timing; upstream tenferro-rs #1928 investigation remains open, not a numerical-support failure. |
| compare_ge_f32_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. GPU1 diagnostic 20261014_010820 on 9655b4e0 (published CubeCL provider) reproduced initial Gt F32 last/first-ten batch growth of 2.902, 3.554 and 2.710 in three independent processes. Other directions are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md and compare_lifetime_counters.md. Keep excluded from steady-state timing; upstream tenferro-rs #1928 investigation remains open, not a numerical-support failure. |
| reduce_sum_full_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.728 | 6.208e-07 |
| reduce_prod_full_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.527 | 5.544e-07 |
| reduce_max_full_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.647 | 0 |
| reduce_min_full_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.599 | 0 |
| compare_eq_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. GPU1 diagnostic 20261014_010820 on 9655b4e0 (published CubeCL provider) reproduced initial Gt F32 last/first-ten batch growth of 2.902, 3.554 and 2.710 in three independent processes. Other directions are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md and compare_lifetime_counters.md. Keep excluded from steady-state timing; upstream tenferro-rs #1928 investigation remains open, not a numerical-support failure. |
| compare_lt_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. GPU1 diagnostic 20261014_010820 on 9655b4e0 (published CubeCL provider) reproduced initial Gt F32 last/first-ten batch growth of 2.902, 3.554 and 2.710 in three independent processes. Other directions are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md and compare_lifetime_counters.md. Keep excluded from steady-state timing; upstream tenferro-rs #1928 investigation remains open, not a numerical-support failure. |
| compare_le_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. GPU1 diagnostic 20261014_010820 on 9655b4e0 (published CubeCL provider) reproduced initial Gt F32 last/first-ten batch growth of 2.902, 3.554 and 2.710 in three independent processes. Other directions are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md and compare_lifetime_counters.md. Keep excluded from steady-state timing; upstream tenferro-rs #1928 investigation remains open, not a numerical-support failure. |
| compare_ge_f64_n1024 | tenferro-cuda-direct | — | — | — | Numerically valid; small Bool allocation lifetime remains nonstationary. GPU1 diagnostic 20261014_010820 on 9655b4e0 (published CubeCL provider) reproduced initial Gt F32 last/first-ten batch growth of 2.902, 3.554 and 2.710 in three independent processes. Other directions are not individually asserted to have that growth. See result/nvidia-gpu/gpu/compare_lifetime.md and compare_lifetime_counters.md. Keep excluded from steady-state timing; upstream tenferro-rs #1928 investigation remains open, not a numerical-support failure. |
| reduce_sum_full_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.608 | 0.005627 |
| reduce_prod_full_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.514 | 0.7454 |
| reduce_max_full_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.536 | 0 |
| reduce_min_full_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.614 | 0 |
| compare_eq_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.448 | 0 |
| compare_lt_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.589 | 0 |
| compare_le_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.649 | 0 |
| compare_ge_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.606 | 0 |
| reduce_sum_full_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.738 | 0.0003449 |
| reduce_prod_full_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.663 | 1.617e-05 |
| reduce_max_full_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.726 | 0 |
| reduce_min_full_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.704 | 0 |
| compare_eq_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.796 | 0 |
| compare_lt_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.767 | 0 |
| compare_le_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.763 | 0 |
| compare_ge_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.717 | 0 |
| reduce_mean_full_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.773 | 0.0001766 |
| reduce_mean_full_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.816 | 1.645e-07 |
| reduce_mean_full_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.644 | 0 |
| reduce_mean_full_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.491 | 0 |
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
| add_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.050 | 0 |
| sub_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.076 | 0 |
| mul_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.036 | 0 |
| div_i32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.359 | 0 |
| rem_i32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.307 | 0 |
| maximum_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.128 | 0 |
| minimum_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.132 | 0 |
| neg_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.007 | 0 |
| abs_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.002 | 0 |
| sign_i32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.012 | 0 |
| reduce_sum_i32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.238 | 0 |
| reduce_max_i32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.203 | 0 |
| reduce_min_i32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.189 | 0 |
| add_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.121 | 0 |
| sub_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.167 | 0 |
| mul_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.091 | 0 |
| div_i64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.379 | 0 |
| rem_i64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.375 | 0 |
| maximum_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.243 | 0 |
| minimum_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.227 | 0 |
| neg_i64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.074 | 0 |
| abs_i64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.924 | 0 |
| sign_i64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.896 | 0 |
| reduce_sum_i64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.213 | 0 |
| reduce_max_i64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.108 | 0 |
| reduce_min_i64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.067 | 0 |
| add_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.462 | 0 |
| sub_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.468 | 0 |
| mul_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.473 | 0 |
| div_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 4.229 | 0 |
| rem_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 4.183 | 0 |
| maximum_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.567 | 0 |
| minimum_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.589 | 0 |
| neg_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.391 | 0 |
| abs_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.393 | 0 |
| sign_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.430 | 0 |
| reduce_sum_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.699 | 0 |
| reduce_max_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.683 | 0 |
| reduce_min_i32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.769 | 0 |
| add_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.106 | 0 |
| sub_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.135 | 0 |
| mul_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.136 | 0 |
| div_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.010 | 0 |
| rem_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.986 | 0 |
| maximum_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.574 | 0 |
| minimum_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.555 | 0 |
| neg_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.901 | 0 |
| abs_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.902 | 0 |
| sign_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.905 | 0 |
| reduce_sum_i64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.011 | 0 |
| reduce_max_i64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.979 | 0 |
| reduce_min_i64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.926 | 0 |
| add_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.383 | 0 |
| add_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.545 | 0 |
| sub_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.307 | 0 |
| sub_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.548 | 0 |
| mul_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.398 | 0 |
| mul_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.388 | 0 |
| div_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.229 | 0 |
| div_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.445 | 0 |
| rem_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.299 | 0 |
| rem_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.350 | 0 |
| pow_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 1.957 | 0.000117 |
| pow_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.095 | 1.48e-07 |
| maximum_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.791 | 0 |
| maximum_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.050 | 0 |
| minimum_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.857 | 0 |
| minimum_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 4.090 | 0 |
| neg_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.028 | 0 |
| neg_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.142 | 0 |
| abs_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.185 | 0 |
| abs_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.299 | 0 |
| sign_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.172 | 0 |
| sign_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.297 | 0 |
| conj_c32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.282 | 0 |
| conj_c64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.290 | 0 |
| exp_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.193 | 0.0001272 |
| exp_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.280 | 1.095e-07 |
| expm1_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.180 | 7.911e-05 |
| expm1_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.311 | 7.39e-08 |
| log_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.161 | 7.945e-05 |
| log_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.330 | 4.231e-08 |
| log1p_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.186 | 0.0001189 |
| log1p_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.361 | 1.108e-07 |
| erf_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.173 | 7.914e-05 |
| erf_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.327 | 8.536e-08 |
| sin_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.072 | 4.348e-05 |
| sin_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.368 | 4.153e-08 |
| cos_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.081 | 6.342e-05 |
| cos_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.184 | 5.905e-08 |
| tanh_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.008 | 9.107e-05 |
| tanh_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.142 | 4.434e-08 |
| sqrt_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.019 | 0 |
| sqrt_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.124 | 0 |
| rsqrt_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.024 | 6.983e-05 |
| rsqrt_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.134 | 6.504e-08 |
| compare_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.144 | 0 |
| compare_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.263 | 0 |
| where_select_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.273 | 0 |
| where_select_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.392 | 0 |
| clamp_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 3.548 | 0 |
| clamp_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 3.673 | 0 |
| reduce_sum_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.488 | 0.0002334 |
| reduce_sum_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.637 | 3.25e-07 |
| reduce_prod_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.509 | 0.0004765 |
| reduce_prod_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.612 | 0 |
| reduce_max_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.946 | 0 |
| reduce_max_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.902 | 0 |
| reduce_min_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 1.999 | 0 |
| reduce_min_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.872 | 0 |
| reduce_sum_squares_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.475 | 0.0002711 |
| reduce_sum_squares_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.636 | 2.474e-07 |
| reduce_mean_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| add_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.200 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.057 | 0 |
| sub_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.225 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.002 | 0 |
| mul_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.202 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.010 | 0 |
| div_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.268 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.028 | 0 |
| rem_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.002 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.792 | 0 |
| pow_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.774 | 0.0001577 |
| pow_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.855 | 1.48e-07 |
| maximum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.659 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.644 | 0 |
| minimum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.630 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.607 | 0 |
| neg_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.495 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.835 | 0 |
| abs_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.114 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.817 | 0 |
| sign_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.152 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.821 | 0 |
| conj_c32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.834 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.976 | 0 |
| exp_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.112 | 0.0001272 |
| exp_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.773 | 1.095e-07 |
| expm1_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.045 | 7.915e-05 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.824 | 7.39e-08 |
| log_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.037 | 7.946e-05 |
| log_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.903 | 4.308e-08 |
| log1p_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.491 | 0.000119 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.915 | 1.11e-07 |
| erf_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.034 | 7.923e-05 |
| erf_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.037 | 8.536e-08 |
| sin_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.513 | 4.348e-05 |
| sin_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.903 | 4.153e-08 |
| cos_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.178 | 6.342e-05 |
| cos_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.887 | 5.905e-08 |
| tanh_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.132 | 9.135e-05 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.857 | 4.437e-08 |
| sqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.118 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.836 | 0 |
| rsqrt_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.183 | 6.983e-05 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.863 | 6.504e-08 |
| compare_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.281 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.670 | 0 |
| where_select_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.035 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.816 | 0 |
| clamp_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.035 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.415 | 0 |
| reduce_sum_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.776 | 0.001926 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.899 | 3.391e-06 |
| reduce_prod_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.896 | 0.004885 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.985 | 3.55e-06 |
| reduce_max_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.822 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.024 | 0 |
| reduce_min_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.741 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.846 | 0 |
| reduce_sum_squares_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.854 | 0.0003531 |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.027 | 3.294e-07 |
| reduce_mean_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_full_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.072 | 0.0001333 |
| reduce_prod_full_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.077 | 0.002738 |
| reduce_max_full_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.100 | 0 |
| reduce_min_full_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.055 | 0 |
| compare_eq_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.373 | 0 |
| compare_lt_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.355 | 0 |
| compare_le_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.362 | 0 |
| compare_ge_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.422 | 0 |
| reduce_sum_full_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.891 | 6.208e-07 |
| reduce_prod_full_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.874 | 5.544e-07 |
| reduce_max_full_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.010 | 0 |
| reduce_min_full_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.015 | 0 |
| compare_eq_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.541 | 0 |
| compare_lt_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.428 | 0 |
| compare_le_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.296 | 0 |
| compare_ge_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.273 | 0 |
| reduce_sum_full_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.595 | 0.005627 |
| reduce_prod_full_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.689 | 0.7454 |
| reduce_max_full_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.425 | 0 |
| reduce_min_full_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.526 | 0 |
| compare_eq_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.246 | 0 |
| compare_lt_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.186 | 0 |
| compare_le_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.201 | 0 |
| compare_ge_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.302 | 0 |
| reduce_sum_full_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.825 | 0.0003449 |
| reduce_prod_full_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.831 | 1.617e-05 |
| reduce_max_full_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.740 | 0 |
| reduce_min_full_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.722 | 0 |
| compare_eq_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.584 | 0 |
| compare_lt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.580 | 0 |
| compare_le_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.607 | 0 |
| compare_ge_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.591 | 0 |
| reduce_mean_full_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.892 | 0.0001766 |
| reduce_mean_full_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.108 | 1.645e-07 |
| reduce_mean_full_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.733 | 0 |
| reduce_mean_full_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_sum_squares_full_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.722 | 0 |
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
| add_i32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.425 | 0 |
| sub_i32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.330 | 0 |
| mul_i32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.379 | 0 |
| div_i32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.480 | 0 |
| rem_i32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.397 | 0 |
| maximum_i32_n1024 | tenferro-cuda-eager | 32 | 15 | 1.925 | 0 |
| minimum_i32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.949 | 0 |
| neg_i32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.190 | 0 |
| abs_i32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.195 | 0 |
| sign_i32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.185 | 0 |
| reduce_sum_i32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.540 | 0 |
| reduce_max_i32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.544 | 0 |
| reduce_min_i32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.518 | 0 |
| add_i64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.514 | 0 |
| sub_i64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.337 | 0 |
| mul_i64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.325 | 0 |
| div_i64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.387 | 0 |
| rem_i64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.466 | 0 |
| maximum_i64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.047 | 0 |
| minimum_i64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.068 | 0 |
| neg_i64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.295 | 0 |
| abs_i64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.294 | 0 |
| sign_i64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.157 | 0 |
| reduce_sum_i64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.528 | 0 |
| reduce_max_i64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.625 | 0 |
| reduce_min_i64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.615 | 0 |
| add_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.419 | 0 |
| sub_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.399 | 0 |
| mul_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.385 | 0 |
| div_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.229 | 0 |
| rem_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.104 | 0 |
| maximum_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.738 | 0 |
| minimum_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.714 | 0 |
| neg_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.172 | 0 |
| abs_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.169 | 0 |
| sign_i32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.210 | 0 |
| reduce_sum_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.888 | 0 |
| reduce_max_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.926 | 0 |
| reduce_min_i32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.894 | 0 |
| add_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.137 | 0 |
| sub_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.167 | 0 |
| mul_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.161 | 0 |
| div_i64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.936 | 0 |
| rem_i64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.950 | 0 |
| maximum_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 6.834 | 0 |
| minimum_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 6.859 | 0 |
| neg_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.812 | 0 |
| abs_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.805 | 0 |
| sign_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.876 | 0 |
| reduce_sum_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.019 | 0 |
| reduce_max_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.003 | 0 |
| reduce_min_i64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.032 | 0 |
| add_f32_n1024 | pytorch-cuda | 512 | 15 | 2.380 | 0 |
| add_f64_n1024 | pytorch-cuda | 512 | 15 | 2.448 | 0 |
| sub_f32_n1024 | pytorch-cuda | 512 | 15 | 2.584 | 0 |
| sub_f64_n1024 | pytorch-cuda | 512 | 15 | 2.440 | 0 |
| mul_f32_n1024 | pytorch-cuda | 512 | 15 | 2.382 | 0 |
| mul_f64_n1024 | pytorch-cuda | 512 | 15 | 2.395 | 0 |
| div_f32_n1024 | pytorch-cuda | 512 | 15 | 2.389 | 0 |
| div_f64_n1024 | pytorch-cuda | 512 | 15 | 2.453 | 0 |
| rem_f32_n1024 | pytorch-cuda | 512 | 15 | 2.398 | 0 |
| rem_f64_n1024 | pytorch-cuda | 512 | 15 | 2.563 | 0 |
| pow_f32_n1024 | pytorch-cuda | 512 | 15 | 2.621 | 0.000117 |
| pow_f64_n1024 | pytorch-cuda | 512 | 15 | 3.784 | 1.48e-07 |
| maximum_f32_n1024 | pytorch-cuda | 512 | 15 | 2.778 | 0 |
| maximum_f64_n1024 | pytorch-cuda | 512 | 15 | 2.509 | 0 |
| minimum_f32_n1024 | pytorch-cuda | 512 | 15 | 2.806 | 0 |
| minimum_f64_n1024 | pytorch-cuda | 512 | 15 | 2.817 | 0 |
| neg_f32_n1024 | pytorch-cuda | 512 | 15 | 2.587 | 0 |
| neg_f64_n1024 | pytorch-cuda | 512 | 15 | 2.681 | 0 |
| abs_f32_n1024 | pytorch-cuda | 512 | 15 | 2.721 | 0 |
| abs_f64_n1024 | pytorch-cuda | 512 | 15 | 2.714 | 0 |
| sign_f32_n1024 | pytorch-cuda | 512 | 15 | 2.486 | 0 |
| sign_f64_n1024 | pytorch-cuda | 512 | 15 | 2.317 | 0 |
| conj_c32_n1024 | pytorch-cuda | 512 | 15 | 2.739 | 0 |
| conj_c64_n1024 | pytorch-cuda | 512 | 15 | 2.661 | 0 |
| exp_f32_n1024 | pytorch-cuda | 512 | 15 | 2.251 | 0.0001272 |
| exp_f64_n1024 | pytorch-cuda | 512 | 15 | 2.244 | 1.095e-07 |
| expm1_f32_n1024 | pytorch-cuda | 512 | 15 | 2.359 | 7.911e-05 |
| expm1_f64_n1024 | pytorch-cuda | 512 | 15 | 2.495 | 7.39e-08 |
| log_f32_n1024 | pytorch-cuda | 512 | 15 | 2.604 | 7.945e-05 |
| log_f64_n1024 | pytorch-cuda | 512 | 15 | 2.285 | 4.231e-08 |
| log1p_f32_n1024 | pytorch-cuda | 512 | 15 | 2.409 | 0.0001189 |
| log1p_f64_n1024 | pytorch-cuda | 512 | 15 | 2.315 | 1.108e-07 |
| erf_f32_n1024 | pytorch-cuda | 512 | 15 | 2.551 | 7.914e-05 |
| erf_f64_n1024 | pytorch-cuda | 512 | 15 | 2.377 | 8.536e-08 |
| sin_f32_n1024 | pytorch-cuda | 512 | 15 | 2.261 | 4.348e-05 |
| sin_f64_n1024 | pytorch-cuda | 512 | 15 | 2.622 | 4.153e-08 |
| cos_f32_n1024 | pytorch-cuda | 512 | 15 | 2.505 | 6.342e-05 |
| cos_f64_n1024 | pytorch-cuda | 512 | 15 | 2.635 | 5.905e-08 |
| tanh_f32_n1024 | pytorch-cuda | 512 | 15 | 2.276 | 9.107e-05 |
| tanh_f64_n1024 | pytorch-cuda | 512 | 15 | 2.546 | 4.434e-08 |
| sqrt_f32_n1024 | pytorch-cuda | 512 | 15 | 2.601 | 0 |
| sqrt_f64_n1024 | pytorch-cuda | 512 | 15 | 2.230 | 0 |
| rsqrt_f32_n1024 | pytorch-cuda | 512 | 15 | 2.290 | 6.983e-05 |
| rsqrt_f64_n1024 | pytorch-cuda | 512 | 15 | 2.608 | 6.504e-08 |
| compare_f32_n1024 | pytorch-cuda | 512 | 15 | 2.699 | 0 |
| compare_f64_n1024 | pytorch-cuda | 512 | 15 | 2.341 | 0 |
| where_select_f32_n1024 | pytorch-cuda | 512 | 15 | 3.052 | 0 |
| where_select_f64_n1024 | pytorch-cuda | 512 | 15 | 3.354 | 0 |
| clamp_f32_n1024 | pytorch-cuda | 512 | 15 | 3.039 | 0 |
| clamp_f64_n1024 | pytorch-cuda | 512 | 15 | 3.151 | 0 |
| reduce_sum_f32_n1024 | pytorch-cuda | 512 | 15 | 3.225 | 0.0002742 |
| reduce_sum_f64_n1024 | pytorch-cuda | 512 | 15 | 3.305 | 1.06e-07 |
| reduce_prod_f32_n1024 | pytorch-cuda | 512 | 15 | 3.421 | 0.0003579 |
| reduce_prod_f64_n1024 | pytorch-cuda | 512 | 15 | 3.673 | 4.441e-07 |
| reduce_max_f32_n1024 | pytorch-cuda | 256 | 15 | 1.976 | 0 |
| reduce_max_f64_n1024 | pytorch-cuda | 512 | 15 | 3.910 | 0 |
| reduce_min_f32_n1024 | pytorch-cuda | 256 | 15 | 1.853 | 0 |
| reduce_min_f64_n1024 | pytorch-cuda | 512 | 15 | 3.464 | 0 |
| reduce_sum_squares_f32_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | pytorch-cuda | 256 | 15 | 1.662 | 1.439e-05 |
| reduce_mean_f64_n1024 | pytorch-cuda | 512 | 15 | 3.625 | 6.71e-09 |
| add_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.362 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.409 | 0 |
| sub_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.365 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.421 | 0 |
| mul_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.422 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.413 | 0 |
| div_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.362 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.442 | 0 |
| rem_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.375 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.404 | 0 |
| pow_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.706 | 0.0001577 |
| pow_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.936 | 1.48e-07 |
| maximum_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.438 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.395 | 0 |
| minimum_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.442 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.408 | 0 |
| neg_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.428 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.253 | 0 |
| abs_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.491 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.259 | 0 |
| sign_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.384 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.252 | 0 |
| conj_c32_n1048576 | pytorch-cuda | 32 | 15 | 0.255 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.407 | 0 |
| exp_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.333 | 0.0001272 |
| exp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.270 | 1.095e-07 |
| expm1_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.376 | 7.915e-05 |
| expm1_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.319 | 7.39e-08 |
| log_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.359 | 7.946e-05 |
| log_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.408 | 4.308e-08 |
| log1p_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.359 | 0.000119 |
| log1p_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.422 | 1.11e-07 |
| erf_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.338 | 7.923e-05 |
| erf_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.462 | 8.536e-08 |
| sin_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.376 | 4.348e-05 |
| sin_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.325 | 4.153e-08 |
| cos_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.373 | 6.342e-05 |
| cos_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.325 | 5.905e-08 |
| tanh_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.337 | 9.135e-05 |
| tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.267 | 4.437e-08 |
| sqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.334 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.255 | 0 |
| rsqrt_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.338 | 6.983e-05 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.256 | 6.504e-08 |
| compare_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.342 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 128 | 15 | 0.888 | 0 |
| where_select_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.444 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.436 | 0 |
| clamp_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.489 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.332 | 0 |
| reduce_sum_f32_n1048576 | pytorch-cuda | 512 | 15 | 3.291 | 0.002948 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.657 | 2.961e-06 |
| reduce_prod_f32_n1048576 | pytorch-cuda | 512 | 15 | 3.162 | 0.002623 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.620 | 4.216e-06 |
| reduce_max_f32_n1048576 | pytorch-cuda | 512 | 15 | 3.265 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.720 | 0 |
| reduce_min_f32_n1048576 | pytorch-cuda | 512 | 15 | 3.268 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.718 | 0 |
| reduce_sum_squares_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | pytorch-cuda | 512 | 15 | 3.292 | 2.947e-06 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.585 | 3.9e-09 |
| reduce_sum_full_f32_n1024 | pytorch-cuda | 512 | 15 | 3.084 | 0.0002666 |
| reduce_prod_full_f32_n1024 | pytorch-cuda | 512 | 15 | 2.868 | 0.0025 |
| reduce_max_full_f32_n1024 | pytorch-cuda | 512 | 15 | 3.261 | 0 |
| reduce_min_full_f32_n1024 | pytorch-cuda | 512 | 15 | 2.838 | 0 |
| compare_eq_f32_n1024 | pytorch-cuda | 512 | 15 | 2.361 | 0 |
| compare_lt_f32_n1024 | pytorch-cuda | 512 | 15 | 2.360 | 0 |
| compare_le_f32_n1024 | pytorch-cuda | 512 | 15 | 2.705 | 0 |
| compare_ge_f32_n1024 | pytorch-cuda | 512 | 15 | 2.388 | 0 |
| reduce_sum_full_f64_n1024 | pytorch-cuda | 512 | 15 | 2.903 | 6.208e-07 |
| reduce_prod_full_f64_n1024 | pytorch-cuda | 512 | 15 | 2.616 | 1.663e-06 |
| reduce_max_full_f64_n1024 | pytorch-cuda | 512 | 15 | 2.833 | 0 |
| reduce_min_full_f64_n1024 | pytorch-cuda | 512 | 15 | 2.858 | 0 |
| compare_eq_f64_n1024 | pytorch-cuda | 512 | 15 | 2.380 | 0 |
| compare_lt_f64_n1024 | pytorch-cuda | 512 | 15 | 2.367 | 0 |
| compare_le_f64_n1024 | pytorch-cuda | 512 | 15 | 2.371 | 0 |
| compare_ge_f64_n1024 | pytorch-cuda | 512 | 15 | 2.646 | 0 |
| reduce_sum_full_f32_n1048576 | pytorch-cuda | 256 | 15 | 2.471 | 0.004049 |
| reduce_prod_full_f32_n1048576 | pytorch-cuda | 256 | 15 | 2.474 | 0.9873 |
| reduce_max_full_f32_n1048576 | pytorch-cuda | 256 | 15 | 2.627 | 0 |
| reduce_min_full_f32_n1048576 | pytorch-cuda | 256 | 15 | 2.625 | 0 |
| compare_eq_f32_n1048576 | pytorch-cuda | 256 | 15 | 1.386 | 0 |
| compare_lt_f32_n1048576 | pytorch-cuda | 256 | 15 | 1.230 | 0 |
| compare_le_f32_n1048576 | pytorch-cuda | 256 | 15 | 1.234 | 0 |
| compare_ge_f32_n1048576 | pytorch-cuda | 256 | 15 | 1.237 | 0 |
| reduce_sum_full_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.808 | 0.001102 |
| reduce_prod_full_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.826 | 0.0001989 |
| reduce_max_full_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.954 | 0 |
| reduce_min_full_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.954 | 0 |
| compare_eq_f64_n1048576 | pytorch-cuda | 256 | 15 | 1.738 | 0 |
| compare_lt_f64_n1048576 | pytorch-cuda | 256 | 15 | 1.736 | 0 |
| compare_le_f64_n1048576 | pytorch-cuda | 256 | 15 | 1.733 | 0 |
| compare_ge_f64_n1048576 | pytorch-cuda | 256 | 15 | 1.735 | 0 |
| reduce_mean_full_f32_n1024 | pytorch-cuda | 512 | 15 | 3.151 | 0.0002906 |
| reduce_sum_squares_full_f32_n1024 | pytorch-cuda | 256 | 15 | 2.026 | 0 |
| reduce_mean_full_f64_n1024 | pytorch-cuda | 512 | 15 | 2.797 | 8.614e-07 |
| reduce_sum_squares_full_f64_n1024 | pytorch-cuda | 256 | 15 | 2.018 | 0 |
| reduce_mean_full_f32_n1048576 | pytorch-cuda | 256 | 15 | 2.470 | 0.004049 |
| reduce_sum_squares_full_f32_n1048576 | pytorch-cuda | 256 | 15 | 2.477 | 0 |
| reduce_mean_full_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.824 | 0.003047 |
| reduce_sum_squares_full_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.825 | 0 |
| scale_real_f32_n1024 | pytorch-cuda | 512 | 15 | 2.965 | 0 |
| scale_real_f64_n1024 | pytorch-cuda | 512 | 15 | 2.970 | 0 |
| scale_real_c32_n1024 | pytorch-cuda | 512 | 15 | 2.740 | 0 |
| scale_real_c64_n1024 | pytorch-cuda | 512 | 15 | 2.567 | 0 |
| scale_complex_c32_n1024 | pytorch-cuda | 512 | 15 | 2.564 | 4.666e-05 |
| scale_complex_c64_n1024 | pytorch-cuda | 512 | 15 | 2.963 | 0 |
| scale_real_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.464 | 0 |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.304 | 0 |
| scale_real_c32_n1048576 | pytorch-cuda | 32 | 15 | 0.303 | 0 |
| scale_real_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.412 | 0 |
| scale_complex_c32_n1048576 | pytorch-cuda | 32 | 15 | 0.305 | 4.516e-05 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.414 | 0 |
| add_i32_n1024 | pytorch-cuda | 512 | 15 | 2.801 | 0 |
| sub_i32_n1024 | pytorch-cuda | 512 | 15 | 2.818 | 0 |
| mul_i32_n1024 | pytorch-cuda | 512 | 15 | 2.765 | 0 |
| div_i32_n1024 | pytorch-cuda | 512 | 15 | 3.021 | 0 |
| rem_i32_n1024 | pytorch-cuda | 512 | 15 | 2.600 | 0 |
| maximum_i32_n1024 | pytorch-cuda | 512 | 15 | 2.420 | 0 |
| minimum_i32_n1024 | pytorch-cuda | 512 | 15 | 2.565 | 0 |
| neg_i32_n1024 | pytorch-cuda | 512 | 15 | 2.523 | 0 |
| abs_i32_n1024 | pytorch-cuda | 512 | 15 | 2.682 | 0 |
| sign_i32_n1024 | pytorch-cuda | 512 | 15 | 2.307 | 0 |
| reduce_sum_i32_n1024 | pytorch-cuda | 512 | 15 | 3.435 | 0 |
| reduce_max_i32_n1024 | pytorch-cuda | 512 | 15 | 2.972 | 0 |
| reduce_min_i32_n1024 | pytorch-cuda | 512 | 15 | 3.069 | 0 |
| add_i64_n1024 | pytorch-cuda | 512 | 15 | 2.386 | 0 |
| sub_i64_n1024 | pytorch-cuda | 512 | 15 | 2.397 | 0 |
| mul_i64_n1024 | pytorch-cuda | 512 | 15 | 2.394 | 0 |
| div_i64_n1024 | pytorch-cuda | 512 | 15 | 2.788 | 0 |
| rem_i64_n1024 | pytorch-cuda | 512 | 15 | 2.772 | 0 |
| maximum_i64_n1024 | pytorch-cuda | 512 | 15 | 2.584 | 0 |
| minimum_i64_n1024 | pytorch-cuda | 512 | 15 | 2.412 | 0 |
| neg_i64_n1024 | pytorch-cuda | 512 | 15 | 2.693 | 0 |
| abs_i64_n1024 | pytorch-cuda | 512 | 15 | 2.702 | 0 |
| sign_i64_n1024 | pytorch-cuda | 512 | 15 | 2.317 | 0 |
| reduce_sum_i64_n1024 | pytorch-cuda | 512 | 15 | 3.299 | 0 |
| reduce_max_i64_n1024 | pytorch-cuda | 512 | 15 | 3.549 | 0 |
| reduce_min_i64_n1024 | pytorch-cuda | 512 | 15 | 3.480 | 0 |
| add_i32_n1048576 | pytorch-cuda | 64 | 15 | 0.360 | 0 |
| sub_i32_n1048576 | pytorch-cuda | 64 | 15 | 0.364 | 0 |
| mul_i32_n1048576 | pytorch-cuda | 64 | 15 | 0.362 | 0 |
| div_i32_n1048576 | pytorch-cuda | 64 | 15 | 0.416 | 0 |
| rem_i32_n1048576 | pytorch-cuda | 64 | 15 | 0.432 | 0 |
| maximum_i32_n1048576 | pytorch-cuda | 64 | 15 | 0.364 | 0 |
| minimum_i32_n1048576 | pytorch-cuda | 64 | 15 | 0.360 | 0 |
| neg_i32_n1048576 | pytorch-cuda | 64 | 15 | 0.338 | 0 |
| abs_i32_n1048576 | pytorch-cuda | 64 | 15 | 0.398 | 0 |
| sign_i32_n1048576 | pytorch-cuda | 64 | 15 | 0.342 | 0 |
| reduce_sum_i32_n1048576 | pytorch-cuda | 512 | 15 | 3.855 | 0 |
| reduce_max_i32_n1048576 | pytorch-cuda | 512 | 15 | 3.552 | 0 |
| reduce_min_i32_n1048576 | pytorch-cuda | 512 | 15 | 3.719 | 0 |
| add_i64_n1048576 | pytorch-cuda | 32 | 15 | 0.411 | 0 |
| sub_i64_n1048576 | pytorch-cuda | 32 | 15 | 0.412 | 0 |
| mul_i64_n1048576 | pytorch-cuda | 32 | 15 | 0.416 | 0 |
| div_i64_n1048576 | pytorch-cuda | 32 | 15 | 0.537 | 0 |
| rem_i64_n1048576 | pytorch-cuda | 32 | 15 | 0.524 | 0 |
| maximum_i64_n1048576 | pytorch-cuda | 32 | 15 | 0.396 | 0 |
| minimum_i64_n1048576 | pytorch-cuda | 32 | 15 | 0.403 | 0 |
| neg_i64_n1048576 | pytorch-cuda | 32 | 15 | 0.249 | 0 |
| abs_i64_n1048576 | pytorch-cuda | 32 | 15 | 0.253 | 0 |
| sign_i64_n1048576 | pytorch-cuda | 32 | 15 | 0.249 | 0 |
| reduce_sum_i64_n1048576 | pytorch-cuda | 256 | 15 | 1.981 | 0 |
| reduce_max_i64_n1048576 | pytorch-cuda | 512 | 15 | 3.670 | 0 |
| reduce_min_i64_n1048576 | pytorch-cuda | 512 | 15 | 3.703 | 0 |
| add_f32_n1024 | jax-cuda | 64 | 15 | 3.191 | 0 |
| add_f64_n1024 | jax-cuda | 64 | 15 | 3.172 | 0 |
| sub_f32_n1024 | jax-cuda | 64 | 15 | 3.189 | 0 |
| sub_f64_n1024 | jax-cuda | 64 | 15 | 3.184 | 0 |
| mul_f32_n1024 | jax-cuda | 64 | 15 | 3.175 | 0 |
| mul_f64_n1024 | jax-cuda | 64 | 15 | 3.179 | 0 |
| div_f32_n1024 | jax-cuda | 64 | 15 | 2.692 | 4.875e-05 |
| div_f64_n1024 | jax-cuda | 64 | 15 | 2.709 | 0 |
| rem_f32_n1024 | jax-cuda | 64 | 15 | 2.687 | 0 |
| rem_f64_n1024 | jax-cuda | 64 | 15 | 2.707 | 0 |
| pow_f32_n1024 | jax-cuda | 64 | 15 | 3.550 | 0.000117 |
| pow_f64_n1024 | jax-cuda | 64 | 15 | 3.565 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda | 64 | 15 | 3.175 | 0 |
| maximum_f64_n1024 | jax-cuda | 64 | 15 | 3.193 | 0 |
| minimum_f32_n1024 | jax-cuda | 64 | 15 | 3.207 | 0 |
| minimum_f64_n1024 | jax-cuda | 64 | 15 | 3.187 | 0 |
| neg_f32_n1024 | jax-cuda | 64 | 15 | 3.043 | 0 |
| neg_f64_n1024 | jax-cuda | 64 | 15 | 3.035 | 0 |
| abs_f32_n1024 | jax-cuda | 64 | 15 | 2.573 | 0 |
| abs_f64_n1024 | jax-cuda | 64 | 15 | 2.565 | 0 |
| sign_f32_n1024 | jax-cuda | 64 | 15 | 2.559 | 0 |
| sign_f64_n1024 | jax-cuda | 64 | 15 | 2.562 | 0 |
| conj_c32_n1024 | jax-cuda | 64 | 15 | 2.597 | 0 |
| conj_c64_n1024 | jax-cuda | 64 | 15 | 2.599 | 0 |
| exp_f32_n1024 | jax-cuda | 64 | 15 | 2.574 | 0.0001272 |
| exp_f64_n1024 | jax-cuda | 64 | 15 | 2.595 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda | 64 | 15 | 2.568 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda | 64 | 15 | 2.588 | 7.39e-08 |
| log_f32_n1024 | jax-cuda | 64 | 15 | 2.580 | 7.945e-05 |
| log_f64_n1024 | jax-cuda | 64 | 15 | 2.577 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda | 64 | 15 | 2.558 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda | 64 | 15 | 2.565 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda | 32 | 15 | 2.031 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda | 32 | 15 | 2.030 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda | 64 | 15 | 2.576 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda | 64 | 15 | 2.593 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda | 64 | 15 | 2.570 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda | 64 | 15 | 2.554 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda | 64 | 15 | 2.531 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda | 64 | 15 | 2.536 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda | 64 | 15 | 2.530 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda | 64 | 15 | 2.541 | 0 |
| rsqrt_f32_n1024 | jax-cuda | 32 | 15 | 2.053 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda | 32 | 15 | 2.060 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda | 64 | 15 | 2.677 | 0 |
| compare_f64_n1024 | jax-cuda | 64 | 15 | 2.681 | 0 |
| where_select_f32_n1024 | jax-cuda | 64 | 15 | 3.119 | 0 |
| where_select_f64_n1024 | jax-cuda | 64 | 15 | 3.118 | 0 |
| clamp_f32_n1024 | jax-cuda | 64 | 15 | 2.775 | 0 |
| clamp_f64_n1024 | jax-cuda | 64 | 15 | 2.772 | 0 |
| reduce_sum_f32_n1024 | jax-cuda | 64 | 15 | 2.892 | 0.0002742 |
| reduce_sum_f64_n1024 | jax-cuda | 64 | 15 | 2.801 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda | 64 | 15 | 2.852 | 0.0003579 |
| reduce_prod_f64_n1024 | jax-cuda | 64 | 15 | 2.856 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda | 64 | 15 | 2.852 | 0 |
| reduce_max_f64_n1024 | jax-cuda | 64 | 15 | 2.851 | 0 |
| reduce_min_f32_n1024 | jax-cuda | 64 | 15 | 2.839 | 0 |
| reduce_min_f64_n1024 | jax-cuda | 64 | 15 | 2.826 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda | 64 | 15 | 3.037 | 1.439e-05 |
| reduce_mean_f64_n1024 | jax-cuda | 64 | 15 | 3.074 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda | 64 | 15 | 3.254 | 0 |
| add_f64_n1048576 | jax-cuda | 32 | 15 | 1.629 | 0 |
| sub_f32_n1048576 | jax-cuda | 64 | 15 | 3.222 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 1.636 | 0 |
| mul_f32_n1048576 | jax-cuda | 64 | 15 | 3.227 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 1.630 | 0 |
| div_f32_n1048576 | jax-cuda | 64 | 15 | 2.758 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda | 32 | 15 | 1.389 | 0 |
| rem_f32_n1048576 | jax-cuda | 64 | 15 | 2.731 | 0 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 1.372 | 0 |
| pow_f32_n1048576 | jax-cuda | 64 | 15 | 3.586 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda | 32 | 15 | 1.745 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda | 64 | 15 | 3.206 | 0 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 1.618 | 0 |
| minimum_f32_n1048576 | jax-cuda | 64 | 15 | 3.207 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 1.614 | 0 |
| neg_f32_n1048576 | jax-cuda | 64 | 15 | 3.099 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 1.552 | 0 |
| abs_f32_n1048576 | jax-cuda | 64 | 15 | 2.605 | 0 |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 1.309 | 0 |
| sign_f32_n1048576 | jax-cuda | 64 | 15 | 2.593 | 0 |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 1.315 | 0 |
| conj_c32_n1048576 | jax-cuda | 32 | 15 | 1.325 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 0.668 | 0 |
| exp_f32_n1048576 | jax-cuda | 64 | 15 | 2.612 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda | 32 | 15 | 1.320 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda | 64 | 15 | 2.610 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda | 32 | 15 | 1.317 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda | 64 | 15 | 2.597 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda | 32 | 15 | 1.312 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda | 64 | 15 | 2.616 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda | 32 | 15 | 1.320 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda | 32 | 15 | 2.045 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda | 32 | 15 | 2.062 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda | 64 | 15 | 2.610 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda | 32 | 15 | 1.331 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda | 64 | 15 | 2.654 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda | 32 | 15 | 1.330 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda | 64 | 15 | 2.612 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda | 32 | 15 | 1.325 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda | 64 | 15 | 2.603 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 1.315 | 0 |
| rsqrt_f32_n1048576 | jax-cuda | 32 | 15 | 2.106 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda | 32 | 15 | 2.096 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda | 64 | 15 | 2.759 | 0 |
| compare_f64_n1048576 | jax-cuda | 64 | 15 | 2.740 | 0 |
| where_select_f32_n1048576 | jax-cuda | 64 | 15 | 3.177 | 0 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 1.604 | 0 |
| clamp_f32_n1048576 | jax-cuda | 64 | 15 | 2.837 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 1.439 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda | 64 | 15 | 2.971 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda | 32 | 15 | 2.871 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda | 32 | 15 | 1.708 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda | 32 | 15 | 2.276 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda | 64 | 15 | 2.930 | 0 |
| reduce_max_f64_n1048576 | jax-cuda | 64 | 15 | 2.908 | 0 |
| reduce_min_f32_n1048576 | jax-cuda | 64 | 15 | 2.888 | 0 |
| reduce_min_f64_n1048576 | jax-cuda | 64 | 15 | 2.903 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda | 64 | 15 | 3.087 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda | 64 | 15 | 3.076 | 6.5e-09 |
| reduce_sum_full_f32_n1024 | jax-cuda | 64 | 15 | 2.842 | 0 |
| reduce_prod_full_f32_n1024 | jax-cuda | 64 | 15 | 2.790 | 0.001786 |
| reduce_max_full_f32_n1024 | jax-cuda | 64 | 15 | 2.823 | 0 |
| reduce_min_full_f32_n1024 | jax-cuda | 64 | 15 | 2.821 | 0 |
| compare_eq_f32_n1024 | jax-cuda | 64 | 15 | 2.665 | 0 |
| compare_lt_f32_n1024 | jax-cuda | 64 | 15 | 2.646 | 0 |
| compare_le_f32_n1024 | jax-cuda | 64 | 15 | 2.663 | 0 |
| compare_ge_f32_n1024 | jax-cuda | 64 | 15 | 2.656 | 0 |
| reduce_sum_full_f64_n1024 | jax-cuda | 64 | 15 | 2.768 | 4.966e-07 |
| reduce_prod_full_f64_n1024 | jax-cuda | 64 | 15 | 2.832 | 4.435e-07 |
| reduce_max_full_f64_n1024 | jax-cuda | 64 | 15 | 2.889 | 0 |
| reduce_min_full_f64_n1024 | jax-cuda | 64 | 15 | 2.921 | 0 |
| compare_eq_f64_n1024 | jax-cuda | 64 | 15 | 2.661 | 0 |
| compare_lt_f64_n1024 | jax-cuda | 64 | 15 | 2.663 | 0 |
| compare_le_f64_n1024 | jax-cuda | 64 | 15 | 2.725 | 0 |
| compare_ge_f64_n1024 | jax-cuda | 64 | 15 | 2.716 | 0 |
| reduce_sum_full_f32_n1048576 | jax-cuda | 64 | 15 | 3.199 | 0.003668 |
| reduce_prod_full_f32_n1048576 | jax-cuda | 64 | 15 | 3.204 | 0.5323 |
| reduce_max_full_f32_n1048576 | jax-cuda | 64 | 15 | 3.172 | 0 |
| reduce_min_full_f32_n1048576 | jax-cuda | 64 | 15 | 3.179 | 0 |
| compare_eq_f32_n1048576 | jax-cuda | 64 | 15 | 2.786 | 0 |
| compare_lt_f32_n1048576 | jax-cuda | 64 | 15 | 2.765 | 0 |
| compare_le_f32_n1048576 | jax-cuda | 64 | 15 | 2.764 | 0 |
| compare_ge_f32_n1048576 | jax-cuda | 64 | 15 | 2.769 | 0 |
| reduce_sum_full_f64_n1048576 | jax-cuda | 64 | 15 | 3.231 | 0.0006452 |
| reduce_prod_full_f64_n1048576 | jax-cuda | 64 | 15 | 3.190 | 0.0001122 |
| reduce_max_full_f64_n1048576 | jax-cuda | 64 | 15 | 3.173 | 0 |
| reduce_min_full_f64_n1048576 | jax-cuda | 64 | 15 | 3.208 | 0 |
| compare_eq_f64_n1048576 | jax-cuda | 64 | 15 | 2.757 | 0 |
| compare_lt_f64_n1048576 | jax-cuda | 64 | 15 | 2.766 | 0 |
| compare_le_f64_n1048576 | jax-cuda | 64 | 15 | 2.764 | 0 |
| compare_ge_f64_n1048576 | jax-cuda | 64 | 15 | 2.756 | 0 |
| reduce_mean_full_f32_n1024 | jax-cuda | 64 | 15 | 2.992 | 0 |
| reduce_sum_squares_full_f32_n1024 | jax-cuda | 64 | 15 | 2.577 | 0 |
| reduce_mean_full_f64_n1024 | jax-cuda | 64 | 15 | 3.037 | 6.891e-07 |
| reduce_sum_squares_full_f64_n1024 | jax-cuda | 64 | 15 | 2.705 | 0 |
| reduce_mean_full_f32_n1048576 | jax-cuda | 64 | 15 | 3.292 | 0.003668 |
| reduce_sum_squares_full_f32_n1048576 | jax-cuda | 64 | 15 | 3.062 | 0 |
| reduce_mean_full_f64_n1048576 | jax-cuda | 64 | 15 | 3.381 | 0.001783 |
| reduce_sum_squares_full_f64_n1048576 | jax-cuda | 64 | 15 | 3.113 | 0 |
| scale_real_f32_n1024 | jax-cuda | 64 | 15 | 3.154 | 0 |
| scale_real_f64_n1024 | jax-cuda | 64 | 15 | 3.157 | 0 |
| scale_real_c32_n1024 | jax-cuda | 64 | 15 | 3.169 | 0 |
| scale_real_c64_n1024 | jax-cuda | 64 | 15 | 3.161 | 0 |
| scale_complex_c32_n1024 | jax-cuda | 64 | 15 | 3.136 | 7.929e-05 |
| scale_complex_c64_n1024 | jax-cuda | 64 | 15 | 3.152 | 3.753e-08 |
| scale_real_f32_n1048576 | jax-cuda | 64 | 15 | 3.213 | 0 |
| scale_real_f64_n1048576 | jax-cuda | 32 | 15 | 1.635 | 0 |
| scale_real_c32_n1048576 | jax-cuda | 32 | 15 | 1.641 | 0 |
| scale_real_c64_n1048576 | jax-cuda | 16 | 15 | 0.813 | 0 |
| scale_complex_c32_n1048576 | jax-cuda | 32 | 15 | 1.665 | 6.186e-05 |
| scale_complex_c64_n1048576 | jax-cuda | 16 | 15 | 0.826 | 4.129e-08 |
| add_i32_n1024 | jax-cuda | 64 | 15 | 3.160 | 0 |
| sub_i32_n1024 | jax-cuda | 64 | 15 | 3.141 | 0 |
| mul_i32_n1024 | jax-cuda | 64 | 15 | 3.160 | 0 |
| div_i32_n1024 | jax-cuda | 32 | 15 | 2.217 | 0 |
| rem_i32_n1024 | jax-cuda | 64 | 15 | 2.703 | 0 |
| maximum_i32_n1024 | jax-cuda | 64 | 15 | 3.110 | 0 |
| minimum_i32_n1024 | jax-cuda | 64 | 15 | 3.143 | 0 |
| neg_i32_n1024 | jax-cuda | 64 | 15 | 2.987 | 0 |
| abs_i32_n1024 | jax-cuda | 64 | 15 | 2.535 | 0 |
| sign_i32_n1024 | jax-cuda | 64 | 15 | 2.550 | 0 |
| reduce_sum_i32_n1024 | jax-cuda | 64 | 15 | 2.841 | 0 |
| reduce_max_i32_n1024 | jax-cuda | 64 | 15 | 2.760 | 0 |
| reduce_min_i32_n1024 | jax-cuda | 64 | 15 | 2.778 | 0 |
| add_i64_n1024 | jax-cuda | 64 | 15 | 3.189 | 0 |
| sub_i64_n1024 | jax-cuda | 64 | 15 | 3.186 | 0 |
| mul_i64_n1024 | jax-cuda | 64 | 15 | 3.191 | 0 |
| div_i64_n1024 | jax-cuda | 32 | 15 | 2.215 | 0 |
| rem_i64_n1024 | jax-cuda | 64 | 15 | 2.694 | 0 |
| maximum_i64_n1024 | jax-cuda | 64 | 15 | 3.175 | 0 |
| minimum_i64_n1024 | jax-cuda | 64 | 15 | 3.177 | 0 |
| neg_i64_n1024 | jax-cuda | 64 | 15 | 3.026 | 0 |
| abs_i64_n1024 | jax-cuda | 64 | 15 | 2.557 | 0 |
| sign_i64_n1024 | jax-cuda | 64 | 15 | 2.558 | 0 |
| reduce_sum_i64_n1024 | jax-cuda | 64 | 15 | 2.931 | 0 |
| reduce_max_i64_n1024 | jax-cuda | 64 | 15 | 2.841 | 0 |
| reduce_min_i64_n1024 | jax-cuda | 64 | 15 | 2.850 | 0 |
| add_i32_n1048576 | jax-cuda | 64 | 15 | 3.230 | 0 |
| sub_i32_n1048576 | jax-cuda | 64 | 15 | 3.212 | 0 |
| mul_i32_n1048576 | jax-cuda | 64 | 15 | 3.234 | 0 |
| div_i32_n1048576 | jax-cuda | 32 | 15 | 2.262 | 0 |
| rem_i32_n1048576 | jax-cuda | 64 | 15 | 2.760 | 0 |
| maximum_i32_n1048576 | jax-cuda | 64 | 15 | 3.213 | 0 |
| minimum_i32_n1048576 | jax-cuda | 64 | 15 | 3.214 | 0 |
| neg_i32_n1048576 | jax-cuda | 64 | 15 | 3.123 | 0 |
| abs_i32_n1048576 | jax-cuda | 64 | 15 | 2.631 | 0 |
| sign_i32_n1048576 | jax-cuda | 64 | 15 | 2.606 | 0 |
| reduce_sum_i32_n1048576 | jax-cuda | 64 | 15 | 2.909 | 0 |
| reduce_max_i32_n1048576 | jax-cuda | 64 | 15 | 2.841 | 0 |
| reduce_min_i32_n1048576 | jax-cuda | 64 | 15 | 3.317 | 0 |
| add_i64_n1048576 | jax-cuda | 32 | 15 | 1.814 | 0 |
| sub_i64_n1048576 | jax-cuda | 32 | 15 | 1.741 | 0 |
| mul_i64_n1048576 | jax-cuda | 32 | 15 | 1.831 | 0 |
| div_i64_n1048576 | jax-cuda | 32 | 15 | 2.282 | 0 |
| rem_i64_n1048576 | jax-cuda | 32 | 15 | 1.402 | 0 |
| maximum_i64_n1048576 | jax-cuda | 32 | 15 | 1.807 | 0 |
| minimum_i64_n1048576 | jax-cuda | 32 | 15 | 1.683 | 0 |
| neg_i64_n1048576 | jax-cuda | 32 | 15 | 1.678 | 0 |
| abs_i64_n1048576 | jax-cuda | 32 | 15 | 1.683 | 0 |
| sign_i64_n1048576 | jax-cuda | 32 | 15 | 1.726 | 0 |
| reduce_sum_i64_n1048576 | jax-cuda | 64 | 15 | 3.526 | 0 |
| reduce_max_i64_n1048576 | jax-cuda | 64 | 15 | 2.934 | 0 |
| reduce_min_i64_n1048576 | jax-cuda | 32 | 15 | 2.053 | 0 |
| add_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.738 | 0 |
| add_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.718 | 0 |
| sub_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.723 | 0 |
| sub_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.725 | 0 |
| mul_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.680 | 0 |
| mul_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.681 | 0 |
| div_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.701 | 4.875e-05 |
| div_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.711 | 0 |
| rem_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.708 | 0 |
| rem_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.695 | 0 |
| pow_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.689 | 0.000117 |
| pow_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.703 | 1.48e-07 |
| maximum_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.697 | 0 |
| maximum_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.690 | 0 |
| minimum_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.694 | 0 |
| minimum_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.701 | 0 |
| neg_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.538 | 0 |
| neg_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.546 | 0 |
| abs_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.550 | 0 |
| abs_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.525 | 0 |
| sign_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.534 | 0 |
| sign_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.548 | 0 |
| conj_c32_n1024 | jax-cuda-jit | 64 | 15 | 2.537 | 0 |
| conj_c64_n1024 | jax-cuda-jit | 64 | 15 | 2.534 | 0 |
| exp_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.547 | 0.0001272 |
| exp_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.559 | 1.095e-07 |
| expm1_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.533 | 7.911e-05 |
| expm1_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.540 | 7.39e-08 |
| log_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.541 | 7.945e-05 |
| log_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.565 | 4.231e-08 |
| log1p_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.540 | 0.0001189 |
| log1p_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.544 | 1.108e-07 |
| erf_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.555 | 9.523e-05 |
| erf_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.565 | 8.536e-08 |
| sin_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.558 | 4.348e-05 |
| sin_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.584 | 4.153e-08 |
| cos_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.549 | 6.342e-05 |
| cos_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.602 | 5.905e-08 |
| tanh_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.536 | 9.325e-05 |
| tanh_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.548 | 4.434e-08 |
| sqrt_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.536 | 0.000119 |
| sqrt_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.545 | 0 |
| rsqrt_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.546 | 6.983e-05 |
| rsqrt_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.543 | 6.504e-08 |
| compare_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.680 | 0 |
| compare_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.706 | 0 |
| where_select_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.756 | 0 |
| where_select_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.774 | 0 |
| clamp_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.762 | 0 |
| clamp_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.785 | 0 |
| reduce_sum_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.535 | 0.0002742 |
| reduce_sum_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.540 | 1.06e-07 |
| reduce_prod_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.557 | 0.0003579 |
| reduce_prod_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.555 | 4.441e-07 |
| reduce_max_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.533 | 0 |
| reduce_max_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.559 | 0 |
| reduce_min_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.549 | 0 |
| reduce_min_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.565 | 0 |
| reduce_sum_squares_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.773 | 1.439e-05 |
| reduce_mean_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.767 | 6.71e-09 |
| add_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.794 | 0 |
| add_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.414 | 0 |
| sub_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.810 | 0 |
| sub_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.401 | 0 |
| mul_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.780 | 0 |
| mul_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.406 | 0 |
| div_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.800 | 4.875e-05 |
| div_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.401 | 0 |
| rem_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.783 | 0 |
| rem_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.407 | 0 |
| pow_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.774 | 0.0001577 |
| pow_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.342 | 1.48e-07 |
| maximum_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.779 | 0 |
| maximum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.417 | 0 |
| minimum_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.771 | 0 |
| minimum_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.410 | 0 |
| neg_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.623 | 0 |
| neg_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.333 | 0 |
| abs_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.624 | 0 |
| abs_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.330 | 0 |
| sign_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.626 | 0 |
| sign_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.327 | 0 |
| conj_c32_n1048576 | jax-cuda-jit | 32 | 15 | 1.327 | 0 |
| conj_c64_n1048576 | jax-cuda-jit | 16 | 15 | 0.675 | 0 |
| exp_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.661 | 0.0001272 |
| exp_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.329 | 1.095e-07 |
| expm1_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.638 | 7.915e-05 |
| expm1_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.320 | 7.39e-08 |
| log_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.634 | 7.946e-05 |
| log_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.324 | 4.308e-08 |
| log1p_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.638 | 0.000119 |
| log1p_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.329 | 1.11e-07 |
| erf_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.640 | 0.0001214 |
| erf_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.324 | 8.536e-08 |
| sin_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.650 | 4.348e-05 |
| sin_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.333 | 4.153e-08 |
| cos_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.639 | 6.342e-05 |
| cos_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.330 | 5.905e-08 |
| tanh_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.630 | 9.508e-05 |
| tanh_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.335 | 4.437e-08 |
| sqrt_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.634 | 0.0001192 |
| sqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.331 | 0 |
| rsqrt_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.627 | 6.983e-05 |
| rsqrt_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.321 | 6.504e-08 |
| compare_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.788 | 0 |
| compare_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.787 | 0 |
| where_select_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.850 | 0 |
| where_select_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.439 | 0 |
| clamp_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.866 | 0 |
| clamp_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.441 | 0 |
| reduce_sum_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.627 | 0.001309 |
| reduce_sum_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.632 | 4.449e-06 |
| reduce_prod_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.626 | 0.004172 |
| reduce_prod_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.633 | 4.436e-06 |
| reduce_max_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.635 | 0 |
| reduce_max_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.628 | 0 |
| reduce_min_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.652 | 0 |
| reduce_min_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.626 | 0 |
| reduce_sum_squares_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_mean_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.829 | 1.86e-06 |
| reduce_mean_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.861 | 6.5e-09 |
| reduce_sum_full_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.537 | 0 |
| reduce_prod_full_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.532 | 0.001072 |
| reduce_max_full_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.534 | 0 |
| reduce_min_full_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.526 | 0 |
| compare_eq_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.674 | 0 |
| compare_lt_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.686 | 0 |
| compare_le_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.695 | 0 |
| compare_ge_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.689 | 0 |
| reduce_sum_full_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.550 | 3.725e-07 |
| reduce_prod_full_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.565 | 4.435e-07 |
| reduce_max_full_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.645 | 0 |
| reduce_min_full_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.641 | 0 |
| compare_eq_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.710 | 0 |
| compare_lt_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.688 | 0 |
| compare_le_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.700 | 0 |
| compare_ge_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.718 | 0 |
| reduce_sum_full_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.914 | 0.003668 |
| reduce_prod_full_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.887 | 0.5327 |
| reduce_max_full_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.887 | 0 |
| reduce_min_full_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.894 | 0 |
| compare_eq_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.795 | 0 |
| compare_lt_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.792 | 0 |
| compare_le_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.794 | 0 |
| compare_ge_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.782 | 0 |
| reduce_sum_full_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.917 | 0.0006452 |
| reduce_prod_full_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.921 | 0.0001122 |
| reduce_max_full_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.928 | 0 |
| reduce_min_full_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.916 | 0 |
| compare_eq_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.778 | 0 |
| compare_lt_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.792 | 0 |
| compare_le_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.791 | 0 |
| compare_ge_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.803 | 0 |
| reduce_mean_full_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.734 | 0 |
| reduce_sum_squares_full_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.532 | 0.0001766 |
| reduce_mean_full_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.749 | 5.169e-07 |
| reduce_sum_squares_full_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.544 | 0 |
| reduce_mean_full_f32_n1048576 | jax-cuda-jit | 64 | 15 | 3.063 | 0.003668 |
| reduce_sum_squares_full_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.881 | 0 |
| reduce_mean_full_f64_n1048576 | jax-cuda-jit | 64 | 15 | 3.098 | 0.001783 |
| reduce_sum_squares_full_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.915 | 0 |
| scale_real_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.716 | 0 |
| scale_real_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.709 | 0 |
| scale_real_c32_n1024 | jax-cuda-jit | 64 | 15 | 2.736 | 0 |
| scale_real_c64_n1024 | jax-cuda-jit | 64 | 15 | 2.730 | 0 |
| scale_complex_c32_n1024 | jax-cuda-jit | 64 | 15 | 2.702 | 7.929e-05 |
| scale_complex_c64_n1024 | jax-cuda-jit | 64 | 15 | 2.707 | 3.753e-08 |
| scale_real_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.799 | 0 |
| scale_real_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.407 | 0 |
| scale_real_c32_n1048576 | jax-cuda-jit | 32 | 15 | 1.417 | 0 |
| scale_real_c64_n1048576 | jax-cuda-jit | 16 | 15 | 0.710 | 0 |
| scale_complex_c32_n1048576 | jax-cuda-jit | 32 | 15 | 1.426 | 6.186e-05 |
| scale_complex_c64_n1048576 | jax-cuda-jit | 16 | 15 | 0.705 | 4.129e-08 |
| add_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.721 | 0 |
| sub_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.689 | 0 |
| mul_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.713 | 0 |
| div_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.689 | 0 |
| rem_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.724 | 0 |
| maximum_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.698 | 0 |
| minimum_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.704 | 0 |
| neg_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.521 | 0 |
| abs_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.529 | 0 |
| sign_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.539 | 0 |
| reduce_sum_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.486 | 0 |
| reduce_max_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.477 | 0 |
| reduce_min_i32_n1024 | jax-cuda-jit | 64 | 15 | 2.479 | 0 |
| add_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.696 | 0 |
| sub_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.709 | 0 |
| mul_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.715 | 0 |
| div_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.708 | 0 |
| rem_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.719 | 0 |
| maximum_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.715 | 0 |
| minimum_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.701 | 0 |
| neg_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.559 | 0 |
| abs_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.538 | 0 |
| sign_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.544 | 0 |
| reduce_sum_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.542 | 0 |
| reduce_max_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.538 | 0 |
| reduce_min_i64_n1024 | jax-cuda-jit | 64 | 15 | 2.551 | 0 |
| add_i32_n1048576 | jax-cuda-jit | 64 | 15 | 2.787 | 0 |
| sub_i32_n1048576 | jax-cuda-jit | 64 | 15 | 2.796 | 0 |
| mul_i32_n1048576 | jax-cuda-jit | 64 | 15 | 2.791 | 0 |
| div_i32_n1048576 | jax-cuda-jit | 64 | 15 | 2.789 | 0 |
| rem_i32_n1048576 | jax-cuda-jit | 64 | 15 | 2.789 | 0 |
| maximum_i32_n1048576 | jax-cuda-jit | 64 | 15 | 2.787 | 0 |
| minimum_i32_n1048576 | jax-cuda-jit | 64 | 15 | 2.797 | 0 |
| neg_i32_n1048576 | jax-cuda-jit | 64 | 15 | 2.621 | 0 |
| abs_i32_n1048576 | jax-cuda-jit | 64 | 15 | 2.612 | 0 |
| sign_i32_n1048576 | jax-cuda-jit | 64 | 15 | 2.638 | 0 |
| reduce_sum_i32_n1048576 | jax-cuda-jit | 64 | 15 | 2.576 | 0 |
| reduce_max_i32_n1048576 | jax-cuda-jit | 64 | 15 | 2.578 | 0 |
| reduce_min_i32_n1048576 | jax-cuda-jit | 64 | 15 | 2.569 | 0 |
| add_i64_n1048576 | jax-cuda-jit | 32 | 15 | 1.401 | 0 |
| sub_i64_n1048576 | jax-cuda-jit | 32 | 15 | 1.400 | 0 |
| mul_i64_n1048576 | jax-cuda-jit | 32 | 15 | 1.402 | 0 |
| div_i64_n1048576 | jax-cuda-jit | 32 | 15 | 1.360 | 0 |
| rem_i64_n1048576 | jax-cuda-jit | 32 | 15 | 1.409 | 0 |
| maximum_i64_n1048576 | jax-cuda-jit | 32 | 15 | 1.408 | 0 |
| minimum_i64_n1048576 | jax-cuda-jit | 32 | 15 | 1.402 | 0 |
| neg_i64_n1048576 | jax-cuda-jit | 32 | 15 | 1.314 | 0 |
| abs_i64_n1048576 | jax-cuda-jit | 32 | 15 | 1.330 | 0 |
| sign_i64_n1048576 | jax-cuda-jit | 32 | 15 | 1.322 | 0 |
| reduce_sum_i64_n1048576 | jax-cuda-jit | 64 | 15 | 2.668 | 0 |
| reduce_max_i64_n1048576 | jax-cuda-jit | 64 | 15 | 2.641 | 0 |
| reduce_min_i64_n1048576 | jax-cuda-jit | 64 | 15 | 2.676 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: benchmarks/gpu/isolated_ops.yaml
timestamp: '2026-11-02T21:01:20Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 433121a7756fdb4f109b02f0422d3714a04514c6
  dirty: false
  features:
  - cuda
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-batched-gemm
harness:
  path: /workspaces/tenferro-benchmark
  commit: 8987033651c18a03e47e16cdd365df807b77b6be
  dirty: true
resolved_deps:
  strided-perm:
  - 0.8.0
  - 0.10.0
  strided-view:
  - 0.7.0
  threadteam:
  - 0.1.0
  - 0.3.0
host:
  cpu: AMD EPYC 7713P 64-Core Processor
  logical_cpus: 64
  power:
    cpufreq_boost: '1'
    governor: schedutil
    scaling_max_freq_khz: '2000000'
    smt:
      active: '0'
      control: notsupported
  numa:
    nodes: 1
    cpulists:
      '0': 0-63
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
  l3_domains:
  - 0-7
  - 8-15
  - 16-23
  - 24-31
  - 32-39
  - 40-47
  - 48-55
  - 56-63
build:
  toolchain:
    rustc: 'rustc 1.99.1 (58cf8f9c3 2026-10-13)

      binary: rustc

      commit-hash: 58cf8f9c35b355199727627d07ae34a4e2846d67

      commit-date: 2026-10-13

      host: x86_64-unknown-linux-gnu

      release: 1.99.1

      LLVM version: 23.1.1

      '
    rustc_release: 1.99.1
    rustflags: null
    target_cpu: null
environment:
  hostname: 0fe295fad79b
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-7
    l3_domains:
    - 0-7
    - 8-15
    - 16-23
    - 24-31
    - 32-39
    - 40-47
    - 48-55
    - 56-63
    cpu_boost: enabled
    scaling_governor: schedutil
    scaling_driver: acpi-cpufreq
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
  trace_execution: 'tenferro trace excluded: this suite compares immediate per-op
    calls, while trace executes a prepared graph in one caller-held session (run_prepared_in);
    JAX JIT compilation outside timing'
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
  device_name: NVIDIA A100 80GB PCIe
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  device_memory_bytes: 85899345920
  driver_version: 580.178.04
  runtime_version: '12.9'
  cuda_version: '13.0'
  cudnn_version: '92700'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  cuda_visible_devices: GPU-530977e1-4968-9283-4129-9fbec3e66542
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh isolated_ops
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 228
  selected_cases: 228
  sha256:
    .devcontainer/cuda/devcontainer.json: 0bf1465bbd1a4c9f896d5320838f2d5a488632a59af7f9b59cc050e95e0b8e09
    .devcontainer/cuda/Dockerfile: 578d9815e5145aa1651ac7e93833468c345ee8bf5e95bfa7717551580ae9feca
    src/bin/benchmark_gpu_public_api.rs: 1a5f3469fc9ffde64a79e05eb9853632366855ece3280ba83e2ba21fe3fb2cdc
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 291e63a517b6f7108527254a7f83751420aa1d4e73395aa60f0691312b2a3773
    src/bin/benchmark_gpu_public_api/plan.rs: 2d87b9d73bf25df8797133cf0a8dc8b1fa974fc3bee055ad92e42802c9ebf643
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: e11a863aee7a85822c474eb771b45ec402e21dbe92173d178c42119ab8ca307e
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: ddc112ea5d568a0399378f2be8b31c91b3144fe7d286e8094d27ca5ccda959bf
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 9cbcb473cba55b1605bab07fd3ff14247859fd7fe9e73d5378dd70108313256b
    data/results/nvidia-gpu/gpu/isolated_ops/20261102_210120/fixtures/manifest.json: be1f51be2eb6db82510c5c0ca85229894cb6cebf4e1510320056fddd71017675
    target/release/benchmark_gpu_public_api: c0a0be40dd246cd4cb7ca59cac19bacd1549a4f3a593aea91e887cf41f6e4aa6
  reference_numpy_version: 2.4.2
```
