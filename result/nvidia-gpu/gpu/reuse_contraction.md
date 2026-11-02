# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **1068**. Unsupported cells: **572**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261102_211407/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

Unit-coefficient AXPBY uses native JAX Ref.addupdate with caller-owned fixed buffers. Its synchronization includes one GPU scalar completion token per destination and block_until_ready; tokens are retained through clock stop and included in the logical retention bound. Helper compilation and seed restoration are untimed. Full Ref snapshots are used only for untimed inspection. This scalar-token synchronization cost is included in the reported latency; it performs no full output copy or D2H transfer inside timing.

Generalized batch dot preserves native output order: tenferro [m,n,b], PyTorch/JAX [b,m,n]. Input permutation views and dimension-number configurations are prepared before timing. Output axes are normalized only on downloaded CPU arrays for verification; no timed output transpose or copy is added.

Prepared einsum uses the public ConcreteEinsumPlan for ik,kj->ij, built once outside execution and sampling. Native matrix multiplication/reuse/accumulation references match the selected mathematical and output-ownership contracts; they do not assert an equivalent prepared-plan interface. Transposed input views/values, read/write handles and accumulation seed restoration are untimed.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | 11.417 (0.080) | unsupported | 4.455 (0.009) | unsupported | unsupported |
| add_into_f64_n1024 | 11.315 (0.144) | unsupported | 4.525 (0.101) | unsupported | unsupported |
| add_read_into_f32_n1024 | 11.857 (0.439) | unsupported | 4.471 (0.008) | unsupported | unsupported |
| add_read_into_f64_n1024 | 11.352 (0.049) | unsupported | 4.474 (0.011) | unsupported | unsupported |
| sub_into_f32_n1024 | 11.771 (0.025) | unsupported | 4.540 (0.010) | unsupported | unsupported |
| sub_into_f64_n1024 | 12.127 (0.035) | unsupported | 4.511 (0.115) | unsupported | unsupported |
| sub_read_into_f32_n1024 | 11.831 (0.203) | unsupported | 4.574 (0.009) | unsupported | unsupported |
| sub_read_into_f64_n1024 | 12.194 (0.041) | unsupported | 4.504 (0.011) | unsupported | unsupported |
| mul_into_f32_n1024 | 11.931 (0.046) | unsupported | 4.506 (0.012) | unsupported | unsupported |
| mul_into_f64_n1024 | 11.768 (0.077) | unsupported | 4.599 (0.095) | unsupported | unsupported |
| mul_read_into_f32_n1024 | 11.854 (0.071) | unsupported | 4.632 (0.010) | unsupported | unsupported |
| mul_read_into_f64_n1024 | 12.077 (0.036) | unsupported | 4.625 (0.012) | unsupported | unsupported |
| div_into_f32_n1024 | 11.766 (0.066) | unsupported | 4.483 (0.006) | unsupported | unsupported |
| div_into_f64_n1024 | 11.865 (0.252) | unsupported | 4.502 (0.022) | unsupported | unsupported |
| div_read_into_f32_n1024 | 11.702 (0.228) | unsupported | 4.475 (0.007) | unsupported | unsupported |
| div_read_into_f64_n1024 | 12.075 (0.322) | unsupported | 4.479 (0.011) | unsupported | unsupported |
| neg_into_f32_n1024 | 11.660 (0.020) | unsupported | 4.421 (0.006) | unsupported | unsupported |
| neg_into_f64_n1024 | 11.667 (0.066) | unsupported | 4.438 (0.006) | unsupported | unsupported |
| neg_read_into_f32_n1024 | 11.637 (0.034) | unsupported | 4.422 (0.015) | unsupported | unsupported |
| neg_read_into_f64_n1024 | 11.753 (0.033) | unsupported | 4.416 (0.010) | unsupported | unsupported |
| conj_into_c32_n1024 | 11.611 (0.065) | unsupported | 4.692 (0.006) | unsupported | unsupported |
| conj_into_c64_n1024 | 11.811 (0.341) | unsupported | 4.878 (0.108) | unsupported | unsupported |
| conj_read_into_c32_n1024 | 11.661 (0.057) | unsupported | 4.589 (0.011) | unsupported | unsupported |
| conj_read_into_c64_n1024 | 11.744 (0.084) | unsupported | 4.791 (0.112) | unsupported | unsupported |
| add_into_f32_n1048576 | 16.445 (0.068) | unsupported | 5.566 (0.024) | unsupported | unsupported |
| add_into_f64_n1048576 | 21.300 (0.218) | unsupported | 12.571 (0.137) | unsupported | unsupported |
| add_read_into_f32_n1048576 | 16.503 (0.041) | unsupported | 5.542 (0.049) | unsupported | unsupported |
| add_read_into_f64_n1048576 | 21.387 (0.173) | unsupported | 12.470 (0.101) | unsupported | unsupported |
| sub_into_f32_n1048576 | 16.511 (0.061) | unsupported | 5.544 (0.020) | unsupported | unsupported |
| sub_into_f64_n1048576 | 21.297 (0.423) | unsupported | 12.600 (0.127) | unsupported | unsupported |
| sub_read_into_f32_n1048576 | 15.884 (0.140) | unsupported | 5.532 (0.027) | unsupported | unsupported |
| sub_read_into_f64_n1048576 | 21.527 (0.200) | unsupported | 12.511 (0.119) | unsupported | unsupported |
| mul_into_f32_n1048576 | 15.819 (0.128) | unsupported | 5.542 (0.040) | unsupported | unsupported |
| mul_into_f64_n1048576 | 21.006 (0.329) | unsupported | 12.731 (0.162) | unsupported | unsupported |
| mul_read_into_f32_n1048576 | 16.473 (0.084) | unsupported | 5.547 (0.030) | unsupported | unsupported |
| mul_read_into_f64_n1048576 | 21.065 (0.318) | unsupported | 12.625 (0.142) | unsupported | unsupported |
| div_into_f32_n1048576 | 15.669 (0.371) | unsupported | 5.569 (0.035) | unsupported | unsupported |
| div_into_f64_n1048576 | 21.444 (0.302) | unsupported | 13.250 (0.220) | unsupported | unsupported |
| div_read_into_f32_n1048576 | 16.197 (0.091) | unsupported | 5.568 (0.028) | unsupported | unsupported |
| div_read_into_f64_n1048576 | 21.474 (0.123) | unsupported | 13.275 (0.098) | unsupported | unsupported |
| neg_into_f32_n1048576 | 15.440 (0.052) | unsupported | 4.930 (0.022) | unsupported | unsupported |
| neg_into_f64_n1048576 | 17.773 (0.330) | unsupported | 7.620 (0.079) | unsupported | unsupported |
| neg_read_into_f32_n1048576 | 15.529 (0.121) | unsupported | 4.915 (0.019) | unsupported | unsupported |
| neg_read_into_f64_n1048576 | 17.418 (0.148) | unsupported | 7.710 (0.069) | unsupported | unsupported |
| conj_into_c32_n1048576 | 17.950 (0.366) | unsupported | 7.715 (0.050) | unsupported | unsupported |
| conj_into_c64_n1048576 | 33.325 (0.315) | unsupported | 25.208 (0.249) | unsupported | unsupported |
| conj_read_into_c32_n1048576 | 17.412 (0.446) | unsupported | 7.686 (0.079) | unsupported | unsupported |
| conj_read_into_c64_n1048576 | 33.226 (0.595) | unsupported | 25.164 (0.259) | unsupported | unsupported |
| tensordot_f32_n1024 | 22.277 (0.106) | 23.305 (0.914) | 26.106 (0.135) | 116.147 (0.681) | 41.539 (0.367) |
| tensordot_f64_n1024 | 22.000 (0.431) | 26.356 (0.267) | 26.310 (0.278) | 114.855 (0.415) | 41.432 (0.295) |
| tensordot_c32_n1024 | 23.122 (0.348) | 28.193 (0.102) | 25.913 (0.096) | 130.242 (0.372) | 53.043 (0.342) |
| tensordot_c64_n1024 | 22.937 (0.221) | 28.020 (0.195) | 26.104 (0.256) | 132.492 (0.547) | 55.546 (0.341) |
| vdot_read_f32_n1024 | 78.206 (0.601) | unsupported | 9.325 (0.085) | 43.355 (0.360) | 41.168 (0.464) |
| vdot_read_f64_n1024 | 79.442 (0.155) | unsupported | 9.344 (0.033) | 43.408 (0.186) | 41.725 (0.181) |
| vdot_read_c32_n1024 | 75.769 (1.677) | unsupported | 9.374 (0.032) | 43.354 (0.435) | 42.936 (0.125) |
| vdot_read_c64_n1024 | 75.114 (2.974) | unsupported | 9.392 (0.030) | 44.452 (0.255) | 44.108 (0.448) |
| norm_squared_read_f32_n1024 | 42.894 (2.172) | unsupported | 9.172 (0.035) | 43.055 (1.703) | 40.012 (0.308) |
| norm_squared_read_f64_n1024 | 41.107 (1.912) | unsupported | 9.054 (0.026) | 38.462 (4.031) | 40.239 (0.212) |
| tensordot_f32_n262144 | 44.180 (1.863) | 48.923 (0.086) | 27.701 (0.089) | 116.579 (0.693) | 49.866 (4.053) |
| tensordot_f64_n262144 | 42.705 (1.946) | 47.201 (0.236) | 26.884 (0.397) | 131.952 (8.787) | 62.877 (7.086) |
| tensordot_c32_n262144 | 78.441 (1.312) | 83.170 (1.045) | 102.553 (1.011) | 138.613 (0.595) | 67.411 (0.678) |
| tensordot_c64_n262144 | 96.463 (0.301) | 100.224 (0.529) | 79.117 (0.760) | 141.301 (1.009) | 75.518 (0.186) |
| vdot_read_f32_n262144 | 81.263 (4.582) | unsupported | 9.471 (0.035) | 46.962 (0.254) | 57.548 (4.597) |
| vdot_read_f64_n262144 | 87.559 (1.577) | unsupported | 9.859 (0.030) | 48.093 (0.450) | 54.669 (5.940) |
| vdot_read_c32_n262144 | 88.727 (0.226) | unsupported | 9.790 (0.008) | 47.722 (0.251) | 53.366 (4.371) |
| vdot_read_c64_n262144 | 93.021 (2.302) | unsupported | 13.223 (0.019) | 48.732 (0.280) | 48.831 (0.487) |
| norm_squared_read_f32_n262144 | 44.576 (2.285) | unsupported | 9.354 (0.046) | 47.356 (0.179) | 45.093 (0.167) |
| norm_squared_read_f64_n262144 | 45.556 (0.223) | unsupported | 9.594 (0.017) | 48.320 (0.209) | 44.756 (0.444) |
| axpby_read_into_accum_f32_n1024 | 20.341 (0.476) | unsupported | 5.229 (0.046) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1024 | 20.642 (0.289) | unsupported | 5.169 (0.015) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1024 | 20.432 (0.304) | unsupported | 5.102 (0.006) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1024 | 20.586 (0.558) | unsupported | 5.268 (0.018) | unsupported | unsupported |
| axpby_read_into_accum_f32_n1048576 | 28.849 (0.336) | unsupported | 8.506 (0.126) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1048576 | 31.356 (0.157) | unsupported | 16.182 (0.122) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1048576 | 31.735 (0.204) | unsupported | 16.252 (0.144) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1048576 | 76.351 (0.538) | unsupported | 36.222 (0.149) | unsupported | unsupported |
| dot_general_read_into_f32_s16 | 14.318 (0.123) | unsupported | 9.714 (0.061) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s16 | 15.345 (0.084) | unsupported | 10.807 (0.096) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s16 | 15.445 (0.045) | unsupported | 10.756 (0.060) | unsupported | unsupported |
| dot_general_read_into_f64_s16 | 14.104 (1.160) | unsupported | 9.702 (0.048) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s16 | 13.899 (0.612) | unsupported | 11.332 (0.117) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s16 | 15.018 (0.090) | unsupported | 11.382 (0.034) | unsupported | unsupported |
| dot_general_read_into_c32_s16 | 14.651 (0.076) | unsupported | 9.774 (0.176) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s16 | 15.941 (0.039) | unsupported | 10.751 (0.158) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s16 | 15.675 (0.044) | unsupported | 10.813 (0.035) | unsupported | unsupported |
| dot_general_read_into_c64_s16 | 14.731 (0.021) | unsupported | 10.222 (0.044) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s16 | 15.710 (0.069) | unsupported | 11.458 (0.098) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s16 | 15.844 (0.076) | unsupported | 11.372 (0.093) | unsupported | unsupported |
| dot_general_read_into_f32_s256 | 21.492 (0.077) | unsupported | 12.300 (0.018) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s256 | 22.657 (0.084) | unsupported | 13.265 (0.082) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s256 | 22.930 (0.123) | unsupported | 13.277 (0.042) | unsupported | unsupported |
| dot_general_read_into_f64_s256 | 16.874 (0.834) | unsupported | 9.947 (0.046) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s256 | 18.571 (0.613) | unsupported | 10.192 (1.349) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s256 | 19.757 (0.788) | unsupported | 11.554 (1.798) | unsupported | unsupported |
| dot_general_read_into_c32_s256 | 25.781 (0.061) | unsupported | 18.183 (0.015) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s256 | 27.237 (0.074) | unsupported | 19.011 (0.031) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s256 | 27.330 (0.087) | unsupported | 19.007 (0.022) | unsupported | unsupported |
| dot_general_read_into_c64_s256 | 27.547 (0.092) | unsupported | 16.873 (0.024) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s256 | 28.932 (0.590) | unsupported | 18.181 (0.012) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s256 | 28.662 (0.580) | unsupported | 18.257 (0.036) | unsupported | unsupported |
| add_strided_read_f32_n1024 | 49.716 (0.237) | unsupported | 5.107 (0.710) | 49.796 (0.238) | 42.587 (0.324) |
| add_strided_read_f64_n1024 | 51.612 (2.630) | unsupported | 5.854 (0.159) | 49.652 (0.380) | 42.736 (0.229) |
| mul_strided_read_f32_n1024 | 47.898 (2.077) | unsupported | 5.587 (0.424) | 49.924 (0.307) | 42.771 (0.255) |
| mul_strided_read_f64_n1024 | 48.932 (0.602) | unsupported | 5.031 (0.085) | 49.461 (0.097) | 42.233 (0.208) |
| neg_strided_read_f32_n1024 | 30.449 (0.936) | unsupported | 4.810 (0.874) | 47.750 (0.222) | 39.743 (0.331) |
| neg_strided_read_f64_n1024 | 34.011 (1.479) | unsupported | 4.764 (0.488) | 47.258 (0.193) | 39.886 (0.333) |
| exp_strided_read_f32_n1024 | 33.356 (0.112) | unsupported | 5.175 (0.486) | 39.605 (0.493) | 40.120 (0.169) |
| exp_strided_read_f64_n1024 | 32.825 (0.653) | unsupported | 4.777 (0.132) | 40.100 (0.253) | 40.002 (0.088) |
| tanh_strided_read_f32_n1024 | 32.579 (0.270) | unsupported | 4.712 (0.341) | 39.883 (0.377) | 40.220 (0.246) |
| tanh_strided_read_f64_n1024 | 32.715 (0.424) | unsupported | 4.767 (0.122) | 39.901 (0.111) | 40.232 (0.226) |
| reduce_sum_strided_read_f32_n1024 | 48.353 (1.436) | unsupported | 7.506 (0.028) | 45.510 (0.391) | 40.335 (0.147) |
| reduce_sum_strided_read_f64_n1024 | 52.532 (2.617) | unsupported | 8.084 (0.019) | 46.096 (0.141) | 40.132 (0.150) |
| conj_strided_read_c32_n1024 | 34.394 (0.126) | unsupported | 6.490 (0.271) | 40.444 (0.604) | 40.400 (0.214) |
| conj_strided_read_c64_n1024 | 34.964 (0.104) | unsupported | 6.657 (0.027) | 40.217 (0.189) | 40.614 (0.173) |
| add_strided_read_f32_n1048576 | 84.927 (0.151) | unsupported | 5.832 (0.132) | 51.302 (0.283) | 44.483 (0.204) |
| add_strided_read_f64_n1048576 | 102.383 (0.317) | unsupported | 12.578 (0.067) | 51.265 (0.336) | 44.424 (0.174) |
| mul_strided_read_f32_n1048576 | 84.857 (0.304) | unsupported | 5.736 (0.342) | 51.018 (0.374) | 44.620 (0.246) |
| mul_strided_read_f64_n1048576 | 102.477 (0.326) | unsupported | 12.644 (0.120) | 52.046 (0.557) | 44.365 (0.629) |
| neg_strided_read_f32_n1048576 | 53.486 (0.948) | unsupported | 4.974 (0.027) | 49.922 (0.294) | 41.894 (0.206) |
| neg_strided_read_f64_n1048576 | 68.906 (1.476) | unsupported | 7.834 (0.081) | 50.060 (1.513) | 41.909 (0.291) |
| exp_strided_read_f32_n1048576 | 56.016 (0.606) | unsupported | 5.536 (0.155) | 41.311 (0.190) | 41.818 (0.259) |
| exp_strided_read_f64_n1048576 | 65.430 (0.297) | unsupported | 8.315 (0.054) | 41.923 (0.328) | 42.060 (0.803) |
| tanh_strided_read_f32_n1048576 | 54.068 (0.131) | unsupported | 5.145 (0.596) | 41.530 (0.216) | 42.228 (0.253) |
| tanh_strided_read_f64_n1048576 | 65.722 (0.341) | unsupported | 8.236 (0.050) | 41.735 (0.384) | 42.545 (0.531) |
| reduce_sum_strided_read_f32_n1048576 | 95.986 (0.487) | unsupported | 11.990 (0.020) | 47.572 (0.376) | 42.198 (0.158) |
| reduce_sum_strided_read_f64_n1048576 | 100.640 (0.448) | unsupported | 13.348 (0.015) | 47.118 (0.337) | 42.115 (0.516) |
| conj_strided_read_c32_n1048576 | 68.136 (1.588) | unsupported | 7.915 (0.054) | 41.810 (0.266) | 42.631 (0.556) |
| conj_strided_read_c64_n1048576 | 96.413 (2.060) | unsupported | 25.235 (0.253) | 40.863 (0.881) | 42.425 (1.846) |
| axpby_overwrite_f32_n1024 | 19.637 (0.721) | unsupported | 5.181 (0.581) | unsupported | unsupported |
| axpby_lerp_f32_n1024 | 20.111 (0.691) | unsupported | 5.501 (0.812) | unsupported | unsupported |
| axpby_unit_add_f32_n1024 | 19.652 (0.636) | unsupported | 4.369 (0.137) | 221.941 (1.048) | 162.904 (0.497) |
| axpby_overwrite_f64_n1024 | 20.287 (1.023) | unsupported | 4.851 (0.422) | unsupported | unsupported |
| axpby_lerp_f64_n1024 | 19.681 (0.295) | unsupported | 4.846 (0.334) | unsupported | unsupported |
| axpby_unit_add_f64_n1024 | 19.691 (0.242) | unsupported | 4.421 (0.237) | 234.279 (1.339) | 166.150 (0.913) |
| axpby_overwrite_c32_n1024 | 19.900 (0.470) | unsupported | 5.808 (0.532) | unsupported | unsupported |
| axpby_lerp_c32_n1024 | 19.824 (0.530) | unsupported | 4.298 (0.318) | unsupported | unsupported |
| axpby_unit_add_c32_n1024 | 20.414 (0.279) | unsupported | 4.524 (0.547) | 225.051 (0.979) | 165.739 (0.466) |
| axpby_overwrite_c64_n1024 | 20.974 (0.581) | unsupported | 5.097 (0.381) | unsupported | unsupported |
| axpby_lerp_c64_n1024 | 20.773 (0.634) | unsupported | 5.445 (0.007) | unsupported | unsupported |
| axpby_unit_add_c64_n1024 | 20.335 (0.855) | unsupported | 4.616 (0.014) | 224.108 (0.564) | 166.389 (1.081) |
| axpby_overwrite_f32_n1048576 | 28.115 (0.146) | unsupported | 5.103 (0.058) | unsupported | unsupported |
| axpby_lerp_f32_n1048576 | 28.434 (0.111) | unsupported | 8.682 (0.054) | unsupported | unsupported |
| axpby_unit_add_f32_n1048576 | 28.539 (0.546) | unsupported | 8.552 (0.182) | 224.812 (1.090) | 166.386 (0.954) |
| axpby_overwrite_f64_n1048576 | 31.305 (0.863) | unsupported | 8.016 (0.058) | unsupported | unsupported |
| axpby_lerp_f64_n1048576 | 31.430 (0.224) | unsupported | 16.935 (0.152) | unsupported | unsupported |
| axpby_unit_add_f64_n1048576 | 31.123 (0.076) | unsupported | 16.147 (0.127) | 225.107 (0.907) | 166.249 (0.758) |
| axpby_overwrite_c32_n1048576 | 31.666 (0.193) | unsupported | 7.973 (0.067) | unsupported | unsupported |
| axpby_lerp_c32_n1048576 | 32.047 (0.241) | unsupported | 17.519 (0.111) | unsupported | unsupported |
| axpby_unit_add_c32_n1048576 | 31.589 (0.278) | unsupported | 16.152 (0.086) | 223.694 (20.806) | 165.451 (0.606) |
| axpby_overwrite_c64_n1048576 | 77.554 (0.334) | unsupported | 25.412 (0.318) | unsupported | unsupported |
| axpby_lerp_c64_n1048576 | 77.328 (0.553) | unsupported | 38.512 (0.145) | unsupported | unsupported |
| axpby_unit_add_c64_n1048576 | 77.221 (0.913) | unsupported | 36.121 (0.199) | 199.325 (23.327) | 165.976 (0.764) |
| dot_general_batch_f32_s32 | 20.382 (1.362) | 24.844 (0.066) | 16.798 (0.057) | 83.089 (0.336) | 41.128 (0.330) |
| dot_general_batch_f64_s32 | 21.518 (2.163) | 25.613 (0.319) | 18.104 (0.062) | 82.312 (0.449) | 41.369 (0.251) |
| dot_general_batch_c32_s32 | 22.620 (1.311) | 27.061 (0.498) | 16.671 (0.135) | 101.862 (0.543) | 60.165 (0.261) |
| dot_general_batch_c64_s32 | 21.480 (2.180) | 27.168 (0.073) | 19.053 (1.551) | 86.609 (8.919) | 62.372 (0.577) |
| dot_general_multiple_f32_s32 | 31.753 (0.094) | 38.829 (0.313) | 36.142 (0.169) | 90.018 (0.274) | 48.408 (0.198) |
| dot_general_multiple_f64_s32 | 29.938 (1.057) | 35.906 (0.178) | 36.365 (0.236) | 91.587 (0.187) | 48.071 (0.221) |
| dot_general_multiple_c32_s32 | 28.903 (2.113) | 34.037 (0.209) | 36.749 (0.141) | 106.713 (0.548) | 61.832 (0.546) |
| dot_general_multiple_c64_s32 | 31.967 (2.422) | 38.252 (0.313) | 36.705 (0.150) | 106.416 (0.526) | 64.136 (0.481) |
| dot_general_batch_f32_s256 | 44.711 (0.089) | 47.539 (1.346) | 34.782 (0.024) | 99.442 (4.265) | 42.843 (0.248) |
| dot_general_batch_f64_s256 | 25.765 (1.620) | 30.105 (0.318) | 22.140 (0.117) | 84.547 (0.457) | 41.019 (0.291) |
| dot_general_batch_c32_s256 | 37.645 (0.733) | 40.956 (0.982) | 33.671 (0.032) | 103.508 (0.575) | 61.053 (0.297) |
| dot_general_batch_c64_s256 | 34.986 (1.766) | 39.339 (2.250) | 35.003 (0.065) | 106.794 (0.767) | 63.708 (0.666) |
| dot_general_multiple_f32_s256 | 40.595 (1.952) | 44.300 (0.229) | 36.740 (0.191) | 87.673 (3.041) | 49.188 (0.265) |
| dot_general_multiple_f64_s256 | 39.387 (0.220) | 43.168 (1.915) | 37.047 (0.249) | 91.808 (1.458) | 62.820 (20.810) |
| dot_general_multiple_c32_s256 | 53.757 (0.751) | 59.465 (2.455) | 40.103 (0.243) | 107.706 (1.840) | 62.079 (0.737) |
| dot_general_multiple_c64_s256 | 48.345 (1.768) | 51.260 (2.385) | 38.838 (1.679) | 110.803 (0.560) | 67.174 (0.414) |
| dot_conj_l0r0_c32_n1024 | 19.513 (0.579) | 21.412 (1.067) | 8.230 (0.192) | 42.264 (2.829) | 43.348 (0.146) |
| dot_conj_l1r0_c32_n1024 | 20.437 (0.865) | 22.981 (0.213) | 9.401 (0.503) | 43.469 (2.833) | 43.174 (0.156) |
| dot_conj_l0r1_c32_n1024 | 20.280 (0.732) | 21.582 (0.195) | 9.438 (0.029) | 43.622 (0.296) | 43.148 (0.305) |
| dot_conj_l1r1_c32_n1024 | 21.168 (1.176) | 21.565 (0.396) | unsupported | unsupported | unsupported |
| dot_conj_l0r0_c64_n1024 | 20.763 (1.191) | 22.356 (0.046) | 8.257 (0.023) | 44.801 (0.483) | 43.891 (0.148) |
| dot_conj_l1r0_c64_n1024 | 22.140 (0.123) | 23.100 (1.043) | 9.182 (1.094) | 44.992 (0.157) | 44.481 (0.174) |
| dot_conj_l0r1_c64_n1024 | 21.892 (0.684) | 22.487 (0.141) | 9.239 (0.047) | 45.266 (0.285) | 44.459 (0.122) |
| dot_conj_l1r1_c64_n1024 | 22.343 (0.354) | 22.452 (0.060) | unsupported | unsupported | unsupported |
| dot_conj_l0r0_c32_n262144 | 33.879 (0.225) | 35.311 (0.129) | 9.758 (0.011) | 48.231 (0.155) | 47.457 (0.277) |
| dot_conj_l1r0_c32_n262144 | 37.757 (0.067) | 39.577 (0.157) | 9.771 (0.023) | 48.438 (0.270) | 47.650 (0.241) |
| dot_conj_l0r1_c32_n262144 | 38.340 (1.119) | 39.637 (0.229) | 9.746 (0.020) | 48.578 (0.187) | 47.392 (0.314) |
| dot_conj_l1r1_c32_n262144 | 37.801 (0.947) | 41.201 (0.175) | unsupported | unsupported | unsupported |
| dot_conj_l0r0_c64_n262144 | 34.637 (0.102) | 36.432 (0.195) | 13.226 (0.011) | 49.106 (0.201) | 48.497 (0.278) |
| dot_conj_l1r0_c64_n262144 | 35.141 (0.713) | 36.435 (0.128) | 13.195 (0.005) | 49.650 (0.387) | 48.895 (0.333) |
| dot_conj_l0r1_c64_n262144 | 36.121 (0.080) | 36.461 (0.618) | 13.211 (0.011) | 50.092 (0.263) | 48.398 (0.210) |
| dot_conj_l1r1_c64_n262144 | 34.713 (0.164) | 36.512 (0.508) | unsupported | unsupported | unsupported |
| einsum_plan_owned_f32_s32 | 19.410 (0.509) | unsupported | 8.473 (0.185) | 40.663 (0.386) | 41.097 (0.294) |
| einsum_plan_read_f32_s32 | 19.247 (0.119) | unsupported | 8.489 (0.028) | 41.062 (0.316) | 41.152 (0.099) |
| einsum_plan_strided_read_f32_s32 | 59.786 (2.240) | unsupported | 8.396 (0.037) | 52.801 (14.026) | 41.537 (0.126) |
| einsum_plan_into_f32_s32 | 14.782 (0.083) | unsupported | 7.172 (0.291) | unsupported | unsupported |
| einsum_plan_read_into_f32_s32 | 15.074 (0.144) | unsupported | 7.001 (0.035) | unsupported | unsupported |
| einsum_plan_read_into_accum_f32_s32 | 15.925 (0.169) | unsupported | 9.145 (0.605) | unsupported | unsupported |
| einsum_plan_strided_read_into_f32_s32 | 55.452 (0.309) | unsupported | 6.906 (0.193) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_f32_s32 | 56.769 (0.241) | unsupported | 8.950 (0.058) | unsupported | unsupported |
| einsum_plan_owned_f64_s32 | 21.295 (0.107) | unsupported | 10.000 (0.952) | 40.208 (0.230) | 41.494 (0.444) |
| einsum_plan_read_f64_s32 | 21.371 (0.318) | unsupported | 9.900 (0.414) | 40.835 (0.281) | 41.224 (0.354) |
| einsum_plan_strided_read_f64_s32 | 61.620 (0.432) | unsupported | 8.715 (0.312) | 41.222 (4.861) | 41.690 (0.212) |
| einsum_plan_into_f64_s32 | 14.525 (0.038) | unsupported | 7.774 (0.472) | unsupported | unsupported |
| einsum_plan_read_into_f64_s32 | 14.692 (0.186) | unsupported | 7.749 (0.368) | unsupported | unsupported |
| einsum_plan_read_into_accum_f64_s32 | 15.657 (0.060) | unsupported | 9.574 (0.088) | unsupported | unsupported |
| einsum_plan_strided_read_into_f64_s32 | 57.632 (0.880) | unsupported | 7.486 (0.312) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_f64_s32 | 59.877 (0.231) | unsupported | 9.829 (0.321) | unsupported | unsupported |
| einsum_plan_owned_c32_s32 | 20.893 (0.040) | unsupported | 10.031 (0.572) | 53.113 (0.420) | 53.357 (0.295) |
| einsum_plan_read_c32_s32 | 21.079 (0.327) | unsupported | 10.718 (1.437) | 53.812 (0.361) | 53.470 (0.240) |
| einsum_plan_strided_read_c32_s32 | 59.046 (2.313) | unsupported | 10.016 (1.423) | 53.317 (0.256) | 53.434 (0.208) |
| einsum_plan_into_c32_s32 | 14.381 (0.439) | unsupported | 8.906 (0.794) | unsupported | unsupported |
| einsum_plan_read_into_c32_s32 | 15.455 (0.774) | unsupported | 9.268 (0.780) | unsupported | unsupported |
| einsum_plan_read_into_accum_c32_s32 | 16.300 (0.642) | unsupported | 10.925 (2.061) | unsupported | unsupported |
| einsum_plan_strided_read_into_c32_s32 | 52.728 (0.965) | unsupported | 8.528 (1.047) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_c32_s32 | 55.480 (2.591) | unsupported | 11.720 (1.135) | unsupported | unsupported |
| einsum_plan_owned_c64_s32 | 19.934 (0.514) | unsupported | 11.132 (2.199) | 55.451 (0.127) | 55.427 (0.143) |
| einsum_plan_read_c64_s32 | 20.365 (1.538) | unsupported | 10.923 (0.702) | 55.894 (0.247) | 55.391 (0.176) |
| einsum_plan_strided_read_c64_s32 | 61.486 (3.951) | unsupported | 11.420 (1.140) | 56.028 (0.161) | 55.805 (0.237) |
| einsum_plan_into_c64_s32 | 14.131 (0.318) | unsupported | 8.135 (0.769) | unsupported | unsupported |
| einsum_plan_read_into_c64_s32 | 14.645 (0.617) | unsupported | 8.697 (0.748) | unsupported | unsupported |
| einsum_plan_read_into_accum_c64_s32 | 16.460 (0.101) | unsupported | 11.508 (0.815) | unsupported | unsupported |
| einsum_plan_strided_read_into_c64_s32 | 58.816 (0.562) | unsupported | 9.817 (0.800) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_c64_s32 | 57.492 (4.484) | unsupported | 12.827 (1.767) | unsupported | unsupported |
| einsum_plan_owned_f32_s256 | 21.771 (0.769) | unsupported | 10.008 (1.133) | 41.466 (0.461) | 42.816 (0.271) |
| einsum_plan_read_f32_s256 | 21.715 (1.323) | unsupported | 10.288 (1.115) | 41.806 (0.231) | 43.091 (0.366) |
| einsum_plan_strided_read_f32_s256 | 65.118 (3.621) | unsupported | 10.647 (0.627) | 41.720 (0.176) | 42.943 (0.263) |
| einsum_plan_into_f32_s256 | 17.219 (0.099) | unsupported | 9.300 (1.275) | unsupported | unsupported |
| einsum_plan_read_into_f32_s256 | 17.339 (0.150) | unsupported | 8.142 (1.354) | unsupported | unsupported |
| einsum_plan_read_into_accum_f32_s256 | 19.919 (0.108) | unsupported | 10.641 (0.019) | unsupported | unsupported |
| einsum_plan_strided_read_into_f32_s256 | 56.259 (0.218) | unsupported | 7.900 (0.709) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_f32_s256 | 58.107 (2.191) | unsupported | 10.539 (0.231) | unsupported | unsupported |
| einsum_plan_owned_f64_s256 | 21.554 (1.121) | unsupported | 9.042 (0.966) | 41.852 (0.197) | 42.833 (0.280) |
| einsum_plan_read_f64_s256 | 21.697 (0.889) | unsupported | 10.601 (0.064) | 42.246 (0.211) | 42.813 (0.257) |
| einsum_plan_strided_read_f64_s256 | 60.311 (0.424) | unsupported | 9.312 (1.097) | 48.174 (8.777) | 48.683 (4.793) |
| einsum_plan_into_f64_s256 | 16.403 (0.163) | unsupported | 8.027 (0.373) | unsupported | unsupported |
| einsum_plan_read_into_f64_s256 | 16.472 (0.269) | unsupported | 8.949 (1.100) | unsupported | unsupported |
| einsum_plan_read_into_accum_f64_s256 | 18.454 (0.255) | unsupported | 11.931 (0.142) | unsupported | unsupported |
| einsum_plan_strided_read_into_f64_s256 | 60.358 (0.209) | unsupported | 9.454 (0.051) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_f64_s256 | 64.664 (4.700) | unsupported | 12.123 (0.339) | unsupported | unsupported |
| einsum_plan_owned_c32_s256 | 30.940 (1.165) | unsupported | 15.253 (0.024) | 51.882 (0.292) | 61.376 (3.407) |
| einsum_plan_read_c32_s256 | 30.396 (0.172) | unsupported | 15.262 (0.018) | 53.095 (0.182) | 55.886 (7.239) |
| einsum_plan_strided_read_c32_s256 | 73.873 (1.042) | unsupported | 15.177 (0.018) | 52.665 (0.306) | 54.082 (2.944) |
| einsum_plan_into_c32_s256 | 24.351 (0.098) | unsupported | 15.238 (0.011) | unsupported | unsupported |
| einsum_plan_read_into_c32_s256 | 24.460 (0.036) | unsupported | 15.247 (0.016) | unsupported | unsupported |
| einsum_plan_read_into_accum_c32_s256 | 26.870 (0.137) | unsupported | 16.468 (0.029) | unsupported | unsupported |
| einsum_plan_strided_read_into_c32_s256 | 69.053 (0.482) | unsupported | 15.182 (0.012) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_c32_s256 | 66.050 (2.732) | unsupported | 16.312 (0.024) | unsupported | unsupported |
| einsum_plan_owned_c64_s256 | 26.312 (0.974) | unsupported | 17.503 (0.007) | 59.582 (0.205) | 70.452 (7.859) |
| einsum_plan_read_c64_s256 | 26.298 (0.435) | unsupported | 17.435 (0.010) | 60.208 (0.305) | 59.810 (4.465) |
| einsum_plan_strided_read_c64_s256 | 66.281 (4.824) | unsupported | 17.406 (0.011) | 60.437 (0.275) | 62.776 (5.843) |
| einsum_plan_into_c64_s256 | 21.232 (0.040) | unsupported | 17.394 (0.008) | unsupported | unsupported |
| einsum_plan_read_into_c64_s256 | 21.913 (0.259) | unsupported | 17.422 (0.009) | unsupported | unsupported |
| einsum_plan_read_into_accum_c64_s256 | 22.739 (0.231) | unsupported | 17.952 (0.019) | unsupported | unsupported |
| einsum_plan_strided_read_into_c64_s256 | 62.246 (2.552) | unsupported | 17.323 (0.010) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_c64_s256 | 64.821 (1.519) | unsupported | 17.897 (0.012) | unsupported | unsupported |
| sub_strided_read_f32_n1024 | 47.590 (0.208) | unsupported | 5.742 (0.010) | 50.057 (0.255) | 47.088 (2.824) |
| sub_strided_read_f64_n1024 | 53.200 (1.820) | unsupported | 5.677 (0.032) | 50.142 (0.339) | 52.353 (4.155) |
| div_strided_read_f32_n1024 | 49.504 (2.205) | unsupported | 5.666 (0.111) | 42.563 (0.155) | 51.299 (3.812) |
| div_strided_read_f64_n1024 | 48.990 (1.507) | unsupported | 5.816 (0.013) | 42.820 (0.270) | 49.531 (3.214) |
| rem_strided_read_f32_n1024 | 69.188 (1.269) | unsupported | 5.964 (0.052) | 42.514 (0.267) | 43.027 (0.197) |
| rem_strided_read_f64_n1024 | 75.543 (4.549) | unsupported | 5.838 (0.041) | 42.800 (0.245) | 42.844 (0.332) |
| pow_strided_read_f32_n1024 | 59.048 (0.660) | unsupported | 5.895 (0.013) | 56.275 (0.212) | 43.318 (0.195) |
| pow_strided_read_f64_n1024 | 64.498 (2.642) | unsupported | 7.251 (0.007) | 56.194 (0.147) | 43.069 (0.225) |
| maximum_strided_read_f32_n1024 | 58.664 (0.882) | unsupported | 5.858 (0.029) | 50.247 (0.221) | 43.156 (0.158) |
| maximum_strided_read_f64_n1024 | 61.205 (1.371) | unsupported | 5.848 (0.111) | 50.159 (0.202) | 42.905 (0.270) |
| minimum_strided_read_f32_n1024 | 60.150 (0.744) | unsupported | 5.971 (0.025) | 50.587 (0.254) | 43.095 (0.135) |
| minimum_strided_read_f64_n1024 | 62.536 (1.197) | unsupported | 5.937 (0.037) | 50.583 (0.186) | 43.013 (0.252) |
| abs_strided_read_f32_n1024 | 32.537 (0.398) | unsupported | 6.960 (0.058) | 40.440 (0.218) | 40.415 (0.354) |
| abs_strided_read_f64_n1024 | 34.680 (0.073) | unsupported | 7.050 (0.014) | 40.447 (0.117) | 40.325 (0.312) |
| sign_strided_read_f32_n1024 | 32.697 (0.172) | unsupported | 5.520 (0.132) | 40.668 (0.268) | 40.310 (0.213) |
| sign_strided_read_f64_n1024 | 35.117 (0.134) | unsupported | 5.430 (0.028) | 40.468 (0.181) | 40.337 (0.311) |
| expm1_strided_read_f32_n1024 | 33.558 (0.090) | unsupported | 5.587 (0.021) | 40.241 (0.248) | 40.219 (0.227) |
| expm1_strided_read_f64_n1024 | 34.913 (0.177) | unsupported | 5.501 (0.035) | 40.621 (0.247) | 40.022 (0.327) |
| log_strided_read_f32_n1024 | 32.797 (0.116) | unsupported | 5.328 (0.050) | 40.435 (0.259) | 40.113 (0.184) |
| log_strided_read_f64_n1024 | 35.023 (0.122) | unsupported | 5.329 (0.112) | 40.628 (0.242) | 40.116 (0.159) |
| log1p_strided_read_f32_n1024 | 32.770 (0.103) | unsupported | 5.325 (0.017) | 40.688 (0.218) | 40.074 (0.226) |
| log1p_strided_read_f64_n1024 | 34.956 (0.523) | unsupported | 5.476 (0.069) | 40.585 (0.312) | 40.200 (0.246) |
| erf_strided_read_f32_n1024 | 33.388 (0.109) | unsupported | 5.529 (0.059) | 62.910 (0.352) | 40.279 (0.224) |
| erf_strided_read_f64_n1024 | 35.729 (0.157) | unsupported | 5.445 (0.009) | 63.086 (0.501) | 40.288 (0.553) |
| sin_strided_read_f32_n1024 | 33.467 (0.087) | unsupported | 5.447 (0.111) | 40.343 (0.161) | 40.287 (0.245) |
| sin_strided_read_f64_n1024 | 35.632 (0.093) | unsupported | 5.530 (0.016) | 40.540 (0.278) | 40.646 (0.409) |
| cos_strided_read_f32_n1024 | 33.509 (0.141) | unsupported | 5.451 (0.027) | 40.192 (0.258) | 40.192 (0.272) |
| cos_strided_read_f64_n1024 | 34.951 (0.120) | unsupported | 5.342 (0.011) | 40.765 (0.170) | 40.597 (0.176) |
| sqrt_strided_read_f32_n1024 | 31.409 (0.865) | unsupported | 5.364 (0.091) | 40.202 (0.267) | 40.017 (0.249) |
| sqrt_strided_read_f64_n1024 | 34.799 (0.164) | unsupported | 5.465 (0.012) | 40.393 (0.257) | 40.119 (0.366) |
| rsqrt_strided_read_f32_n1024 | 31.383 (0.691) | unsupported | 5.544 (0.024) | 65.075 (0.273) | 40.232 (0.245) |
| rsqrt_strided_read_f64_n1024 | 35.239 (0.231) | unsupported | 5.460 (0.035) | 65.203 (0.236) | 40.170 (0.305) |
| reduce_prod_strided_read_f32_n1024 | 50.209 (1.982) | unsupported | 7.330 (0.006) | 46.146 (0.339) | 40.231 (0.279) |
| reduce_prod_strided_read_f64_n1024 | 50.751 (2.089) | unsupported | 8.134 (0.012) | 46.295 (0.297) | 38.961 (0.234) |
| reduce_max_strided_read_f32_n1024 | 52.082 (0.798) | unsupported | 8.079 (0.010) | 46.298 (0.300) | 40.340 (0.165) |
| reduce_max_strided_read_f64_n1024 | 51.765 (1.969) | unsupported | 8.664 (0.006) | 46.208 (0.154) | 40.349 (0.210) |
| reduce_min_strided_read_f32_n1024 | 50.249 (1.869) | unsupported | 8.074 (0.008) | 46.387 (0.144) | 40.126 (0.224) |
| reduce_min_strided_read_f64_n1024 | 50.995 (2.230) | unsupported | 8.664 (0.010) | 46.114 (0.199) | 40.067 (0.296) |
| reduce_sum_squares_strided_read_f32_n1024 | 48.216 (1.207) | unsupported | 13.687 (0.064) | 837.101 (6.785) | 39.994 (0.207) |
| reduce_sum_squares_strided_read_f64_n1024 | 49.583 (0.232) | unsupported | 13.960 (0.112) | 837.153 (2.768) | 39.997 (0.146) |
| sub_strided_read_f32_n1048576 | 85.038 (0.878) | unsupported | 5.758 (0.056) | 51.525 (0.243) | 44.178 (0.103) |
| sub_strided_read_f64_n1048576 | 101.090 (0.893) | unsupported | 12.665 (0.152) | 51.892 (0.384) | 44.418 (0.398) |
| div_strided_read_f32_n1048576 | 84.445 (0.638) | unsupported | 5.706 (0.106) | 44.085 (0.207) | 44.575 (0.158) |
| div_strided_read_f64_n1048576 | 101.234 (1.891) | unsupported | 13.573 (0.259) | 44.114 (0.249) | 44.481 (0.282) |
| rem_strided_read_f32_n1048576 | 105.681 (1.287) | unsupported | 5.831 (0.082) | 43.324 (0.429) | 43.848 (0.309) |
| rem_strided_read_f64_n1048576 | 126.499 (0.993) | unsupported | 12.180 (0.093) | 44.719 (0.456) | 44.283 (0.415) |
| pow_strided_read_f32_n1048576 | 98.683 (0.913) | unsupported | 10.878 (0.027) | 58.015 (1.194) | 44.469 (0.147) |
| pow_strided_read_f64_n1048576 | 127.967 (0.919) | unsupported | 29.108 (0.048) | 56.488 (7.599) | 42.645 (0.351) |
| maximum_strided_read_f32_n1048576 | 94.253 (0.886) | unsupported | 5.866 (0.102) | 51.275 (0.336) | 44.295 (0.239) |
| maximum_strided_read_f64_n1048576 | 115.078 (0.765) | unsupported | 12.149 (0.148) | 52.082 (0.270) | 44.388 (0.452) |
| minimum_strided_read_f32_n1048576 | 94.441 (0.497) | unsupported | 5.861 (0.087) | 51.395 (4.532) | 43.865 (0.220) |
| minimum_strided_read_f64_n1048576 | 113.904 (0.388) | unsupported | 12.160 (0.118) | 46.496 (7.978) | 44.556 (0.575) |
| abs_strided_read_f32_n1048576 | 53.062 (0.217) | unsupported | 7.046 (0.097) | 39.462 (6.615) | 42.084 (0.420) |
| abs_strided_read_f64_n1048576 | 64.992 (0.121) | unsupported | 7.979 (0.091) | 41.555 (0.433) | 41.980 (0.331) |
| sign_strided_read_f32_n1048576 | 53.653 (0.449) | unsupported | 5.526 (0.112) | 41.191 (0.367) | 42.049 (0.226) |
| sign_strided_read_f64_n1048576 | 64.958 (0.337) | unsupported | 7.877 (0.038) | 34.172 (0.569) | 42.015 (0.267) |
| expm1_strided_read_f32_n1048576 | 54.725 (0.163) | unsupported | 5.845 (0.033) | 41.323 (0.203) | 42.065 (0.416) |
| expm1_strided_read_f64_n1048576 | 67.414 (0.268) | unsupported | 9.957 (0.054) | 41.890 (0.280) | 42.050 (0.450) |
| log_strided_read_f32_n1048576 | 54.515 (0.239) | unsupported | 5.568 (0.078) | 41.597 (0.212) | 42.199 (0.202) |
| log_strided_read_f64_n1048576 | 69.150 (0.160) | unsupported | 12.734 (0.031) | 41.569 (0.215) | 42.076 (0.292) |
| log1p_strided_read_f32_n1048576 | 54.467 (0.200) | unsupported | 5.511 (0.025) | 41.180 (0.181) | 41.678 (0.252) |
| log1p_strided_read_f64_n1048576 | 69.877 (0.207) | unsupported | 13.150 (0.052) | 41.624 (0.174) | 41.818 (0.275) |
| erf_strided_read_f32_n1048576 | 54.448 (0.214) | unsupported | 5.178 (0.368) | 64.151 (0.281) | 41.359 (0.197) |
| erf_strided_read_f64_n1048576 | 71.213 (0.239) | unsupported | 14.372 (0.062) | 64.434 (0.473) | 42.209 (0.251) |
| sin_strided_read_f32_n1048576 | 52.726 (0.415) | unsupported | 5.814 (0.020) | 41.311 (0.178) | 41.627 (0.265) |
| sin_strided_read_f64_n1048576 | 66.956 (1.573) | unsupported | 10.107 (0.042) | 41.862 (0.306) | 42.118 (0.372) |
| cos_strided_read_f32_n1048576 | 53.491 (1.423) | unsupported | 5.813 (0.036) | 41.227 (0.208) | 41.833 (0.231) |
| cos_strided_read_f64_n1048576 | 66.180 (0.801) | unsupported | 10.199 (0.053) | 41.520 (0.270) | 41.922 (0.557) |
| sqrt_strided_read_f32_n1048576 | 52.764 (0.753) | unsupported | 5.510 (0.117) | 41.310 (0.142) | 41.866 (0.322) |
| sqrt_strided_read_f64_n1048576 | 65.587 (1.254) | unsupported | 7.928 (0.055) | 41.493 (0.417) | 42.005 (0.316) |
| rsqrt_strided_read_f32_n1048576 | 53.341 (0.272) | unsupported | 5.471 (0.097) | 65.846 (0.207) | 41.898 (0.185) |
| rsqrt_strided_read_f64_n1048576 | 65.170 (0.244) | unsupported | 7.919 (0.023) | 66.072 (0.380) | 41.869 (0.278) |
| reduce_prod_strided_read_f32_n1048576 | 91.253 (1.404) | unsupported | 12.108 (0.011) | 47.757 (0.399) | 40.588 (0.160) |
| reduce_prod_strided_read_f64_n1048576 | 99.378 (0.282) | unsupported | 13.305 (0.009) | 47.493 (0.371) | 41.981 (0.246) |
| reduce_max_strided_read_f32_n1048576 | 92.022 (0.829) | unsupported | 12.072 (0.011) | 47.718 (0.442) | 41.873 (0.296) |
| reduce_max_strided_read_f64_n1048576 | 96.421 (0.282) | unsupported | 13.405 (0.009) | 47.951 (0.161) | 42.485 (0.117) |
| reduce_min_strided_read_f32_n1048576 | 91.186 (0.298) | unsupported | 12.066 (0.035) | 47.561 (0.239) | 42.595 (0.375) |
| reduce_min_strided_read_f64_n1048576 | 96.484 (1.014) | unsupported | 13.408 (0.023) | 47.791 (0.239) | 42.287 (0.355) |
| reduce_sum_squares_strided_read_f32_n1048576 | 91.054 (3.443) | unsupported | 17.096 (1.634) | 836.490 (4.064) | 41.935 (0.388) |
| reduce_sum_squares_strided_read_f64_n1048576 | 99.669 (1.751) | unsupported | 20.500 (0.019) | 842.788 (3.958) | 42.435 (0.231) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.923 | 0 |
| add_into_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.897 | 0 |
| add_read_into_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.035 | 0 |
| add_read_into_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.906 | 0 |
| sub_into_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.013 | 0 |
| sub_into_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.105 | 0 |
| sub_read_into_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.029 | 0 |
| sub_read_into_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.122 | 0 |
| mul_into_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.054 | 0 |
| mul_into_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.012 | 0 |
| mul_read_into_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.035 | 0 |
| mul_read_into_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.092 | 0 |
| div_into_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.012 | 0 |
| div_into_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.037 | 0 |
| div_read_into_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.996 | 0 |
| div_read_into_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.091 | 0 |
| neg_into_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.985 | 0 |
| neg_into_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.987 | 0 |
| neg_read_into_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.979 | 0 |
| neg_read_into_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.009 | 0 |
| conj_into_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.972 | 0 |
| conj_into_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.024 | 0 |
| conj_read_into_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.985 | 0 |
| conj_read_into_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.006 | 0 |
| add_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.052 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.682 | 0 |
| add_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.056 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.684 | 0 |
| sub_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.057 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.681 | 0 |
| sub_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.017 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.689 | 0 |
| mul_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.012 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.672 | 0 |
| mul_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.054 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.674 | 0 |
| div_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.003 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.686 | 0 |
| div_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.037 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.687 | 0 |
| neg_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 0.988 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.569 | 0 |
| neg_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 0.994 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.557 | 0 |
| conj_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.574 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.533 | 0 |
| conj_read_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.557 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.532 | 0 |
| tensordot_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.851 | 0.000577 |
| tensordot_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.816 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.960 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.936 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.503 | 0 |
| vdot_read_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.542 | 1.722e-07 |
| vdot_read_c32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.425 | 2.301e-05 |
| vdot_read_c64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.404 | 3.435e-07 |
| norm_squared_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.745 | 0.0001766 |
| norm_squared_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.631 | 1.645e-07 |
| tensordot_f32_n262144 | tenferro-cuda-direct | 64 | 15 | 2.828 | 0.01125 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 64 | 15 | 2.733 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-direct | 32 | 15 | 2.510 | 0.01203 |
| tensordot_c64_n262144 | tenferro-cuda-direct | 32 | 15 | 3.087 | 7.381e-06 |
| vdot_read_f32_n262144 | tenferro-cuda-direct | 32 | 15 | 2.600 | 0 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 32 | 15 | 2.802 | 2.995e-06 |
| vdot_read_c32_n262144 | tenferro-cuda-direct | 32 | 15 | 2.839 | 0 |
| vdot_read_c64_n262144 | tenferro-cuda-direct | 32 | 15 | 2.977 | 8.766e-06 |
| norm_squared_read_f32_n262144 | tenferro-cuda-direct | 64 | 15 | 2.853 | 0.0001786 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 64 | 15 | 2.916 | 1.664e-07 |
| axpby_read_into_accum_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.604 | 0 |
| axpby_read_into_accum_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.642 | 0 |
| axpby_read_into_accum_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.615 | 0 |
| axpby_read_into_accum_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.635 | 0 |
| axpby_read_into_accum_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.846 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.003 | 0 |
| axpby_read_into_accum_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.016 | 0 |
| axpby_read_into_accum_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.222 | 0 |
| dot_general_read_into_f32_s16 | tenferro-cuda-direct | 256 | 15 | 3.665 | 0.0001207 |
| dot_general_read_into_accum_f32_s16 | tenferro-cuda-direct | 256 | 15 | 3.928 | 9.558e-05 |
| dot_general_read_into_accum_cached_f32_s16 | tenferro-cuda-direct | 128 | 15 | 1.977 | 9.558e-05 |
| dot_general_read_into_f64_s16 | tenferro-cuda-direct | 256 | 15 | 3.611 | 0 |
| dot_general_read_into_accum_f64_s16 | tenferro-cuda-direct | 256 | 15 | 3.558 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | tenferro-cuda-direct | 256 | 15 | 3.845 | 0 |
| dot_general_read_into_c32_s16 | tenferro-cuda-direct | 256 | 15 | 3.751 | 0.0003615 |
| dot_general_read_into_accum_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.040 | 0.0001632 |
| dot_general_read_into_accum_cached_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.006 | 0.0001632 |
| dot_general_read_into_c64_s16 | tenferro-cuda-direct | 256 | 15 | 3.771 | 2.703e-07 |
| dot_general_read_into_accum_c64_s16 | tenferro-cuda-direct | 128 | 15 | 2.011 | 1.774e-07 |
| dot_general_read_into_accum_cached_c64_s16 | tenferro-cuda-direct | 128 | 15 | 2.028 | 1.774e-07 |
| dot_general_read_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.751 | 0.0003909 |
| dot_general_read_into_accum_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.900 | 0.0002334 |
| dot_general_read_into_accum_cached_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.935 | 0.0002334 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.160 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.377 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.529 | 0 |
| dot_general_read_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.300 | 0.001457 |
| dot_general_read_into_accum_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.486 | 0.0008947 |
| dot_general_read_into_accum_cached_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.498 | 0.0008947 |
| dot_general_read_into_c64_s256 | tenferro-cuda-direct | 128 | 15 | 3.526 | 2.474e-06 |
| dot_general_read_into_accum_c64_s256 | tenferro-cuda-direct | 128 | 15 | 3.703 | 1.654e-06 |
| dot_general_read_into_accum_cached_c64_s256 | tenferro-cuda-direct | 128 | 15 | 3.669 | 1.654e-06 |
| add_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.182 | 0 |
| add_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.303 | 0 |
| mul_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.065 | 0 |
| mul_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.132 | 0 |
| neg_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.949 | 0 |
| neg_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.177 | 0 |
| exp_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.135 | 0.0001272 |
| exp_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.101 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 4.170 | 9.107e-05 |
| tanh_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.094 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.095 | 0.0001208 |
| reduce_sum_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.362 | 4.202e-07 |
| conj_strided_read_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.201 | 0 |
| conj_strided_read_c64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.238 | 0 |
| add_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.718 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.638 | 0 |
| mul_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.715 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.640 | 0 |
| neg_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.712 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.205 | 0 |
| exp_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.793 | 0.0001272 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.094 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.730 | 9.135e-05 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.103 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.072 | 0.003592 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.220 | 4.724e-06 |
| conj_strided_read_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.180 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.543 | 0 |
| axpby_overwrite_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.514 | 0 |
| axpby_lerp_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.574 | 0.0001233 |
| axpby_unit_add_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.515 | 0 |
| axpby_overwrite_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.597 | 0 |
| axpby_lerp_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.519 | 0 |
| axpby_unit_add_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.520 | 0 |
| axpby_overwrite_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.547 | 0 |
| axpby_lerp_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.537 | 0.0001233 |
| axpby_unit_add_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.613 | 0 |
| axpby_overwrite_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.685 | 0 |
| axpby_lerp_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.659 | 0 |
| axpby_unit_add_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.603 | 0 |
| axpby_overwrite_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.799 | 0 |
| axpby_lerp_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.820 | 0.0001247 |
| axpby_unit_add_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.826 | 0 |
| axpby_overwrite_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.002 | 0 |
| axpby_lerp_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.006 | 0 |
| axpby_unit_add_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.996 | 0 |
| axpby_overwrite_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.013 | 0 |
| axpby_lerp_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.026 | 0.0001245 |
| axpby_unit_add_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.011 | 0 |
| axpby_overwrite_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.241 | 0 |
| axpby_lerp_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.237 | 0 |
| axpby_unit_add_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.236 | 0 |
| dot_general_batch_f32_s32 | tenferro-cuda-direct | 128 | 15 | 2.609 | 0.0002227 |
| dot_general_batch_f64_s32 | tenferro-cuda-direct | 128 | 15 | 2.754 | 1.326e-07 |
| dot_general_batch_c32_s32 | tenferro-cuda-direct | 128 | 15 | 2.895 | 0.0004376 |
| dot_general_batch_c64_s32 | tenferro-cuda-direct | 128 | 15 | 2.749 | 4.296e-07 |
| dot_general_multiple_f32_s32 | tenferro-cuda-direct | 64 | 15 | 2.032 | 0.0002162 |
| dot_general_multiple_f64_s32 | tenferro-cuda-direct | 128 | 15 | 3.832 | 1.632e-07 |
| dot_general_multiple_c32_s32 | tenferro-cuda-direct | 128 | 15 | 3.700 | 0.0002794 |
| dot_general_multiple_c64_s32 | tenferro-cuda-direct | 64 | 15 | 2.046 | 4.303e-07 |
| dot_general_batch_f32_s256 | tenferro-cuda-direct | 64 | 15 | 2.862 | 0.0008561 |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | 128 | 15 | 3.298 | 7.463e-07 |
| dot_general_batch_c32_s256 | tenferro-cuda-direct | 64 | 15 | 2.409 | 0.001749 |
| dot_general_batch_c64_s256 | tenferro-cuda-direct | 64 | 15 | 2.239 | 1.665e-06 |
| dot_general_multiple_f32_s256 | tenferro-cuda-direct | 64 | 15 | 2.598 | 0.001306 |
| dot_general_multiple_f64_s256 | tenferro-cuda-direct | 64 | 15 | 2.521 | 1.803e-06 |
| dot_general_multiple_c32_s256 | tenferro-cuda-direct | 64 | 15 | 3.440 | 0.0022 |
| dot_general_multiple_c64_s256 | tenferro-cuda-direct | 64 | 15 | 3.094 | 9.057e-07 |
| dot_conj_l0r0_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.498 | 0.0003083 |
| dot_conj_l1r0_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.616 | 0.0002905 |
| dot_conj_l0r1_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.596 | 0.0002905 |
| dot_conj_l1r1_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.709 | 0.0003083 |
| dot_conj_l0r0_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.658 | 4.927e-07 |
| dot_conj_l1r0_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.834 | 1.807e-07 |
| dot_conj_l0r1_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.802 | 1.807e-07 |
| dot_conj_l1r1_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.860 | 4.927e-07 |
| dot_conj_l0r0_c32_n262144 | tenferro-cuda-direct | 64 | 15 | 2.168 | 0.005822 |
| dot_conj_l1r0_c32_n262144 | tenferro-cuda-direct | 64 | 15 | 2.416 | 0.005619 |
| dot_conj_l0r1_c32_n262144 | tenferro-cuda-direct | 64 | 15 | 2.454 | 0.005619 |
| dot_conj_l1r1_c32_n262144 | tenferro-cuda-direct | 64 | 15 | 2.419 | 0.005822 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | 64 | 15 | 2.217 | 1.212e-05 |
| dot_conj_l1r0_c64_n262144 | tenferro-cuda-direct | 64 | 15 | 2.249 | 1.516e-05 |
| dot_conj_l0r1_c64_n262144 | tenferro-cuda-direct | 64 | 15 | 2.312 | 1.516e-05 |
| dot_conj_l1r1_c64_n262144 | tenferro-cuda-direct | 64 | 15 | 2.222 | 1.212e-05 |
| einsum_plan_owned_f32_s32 | tenferro-cuda-direct | 128 | 15 | 2.485 | 0.0002018 |
| einsum_plan_read_f32_s32 | tenferro-cuda-direct | 128 | 15 | 2.464 | 0.0002018 |
| einsum_plan_strided_read_f32_s32 | tenferro-cuda-direct | 64 | 15 | 3.826 | 0.0002018 |
| einsum_plan_into_f32_s32 | tenferro-cuda-direct | 256 | 15 | 3.784 | 0.0002018 |
| einsum_plan_read_into_f32_s32 | tenferro-cuda-direct | 256 | 15 | 3.859 | 0.0002018 |
| einsum_plan_read_into_accum_f32_s32 | tenferro-cuda-direct | 128 | 15 | 2.038 | 0.0001134 |
| einsum_plan_strided_read_into_f32_s32 | tenferro-cuda-direct | 64 | 15 | 3.549 | 0.0002018 |
| einsum_plan_strided_read_into_accum_f32_s32 | tenferro-cuda-direct | 64 | 15 | 3.633 | 0.0001134 |
| einsum_plan_owned_f64_s32 | tenferro-cuda-direct | 128 | 15 | 2.726 | 0 |
| einsum_plan_read_f64_s32 | tenferro-cuda-direct | 128 | 15 | 2.736 | 0 |
| einsum_plan_strided_read_f64_s32 | tenferro-cuda-direct | 32 | 15 | 1.972 | 0 |
| einsum_plan_into_f64_s32 | tenferro-cuda-direct | 256 | 15 | 3.718 | 0 |
| einsum_plan_read_into_f64_s32 | tenferro-cuda-direct | 256 | 15 | 3.761 | 0 |
| einsum_plan_read_into_accum_f64_s32 | tenferro-cuda-direct | 256 | 15 | 4.008 | 0 |
| einsum_plan_strided_read_into_f64_s32 | tenferro-cuda-direct | 64 | 15 | 3.688 | 0 |
| einsum_plan_strided_read_into_accum_f64_s32 | tenferro-cuda-direct | 64 | 15 | 3.832 | 0 |
| einsum_plan_owned_c32_s32 | tenferro-cuda-direct | 128 | 15 | 2.674 | 0.0004148 |
| einsum_plan_read_c32_s32 | tenferro-cuda-direct | 128 | 15 | 2.698 | 0.0004148 |
| einsum_plan_strided_read_c32_s32 | tenferro-cuda-direct | 64 | 15 | 3.779 | 0.0004148 |
| einsum_plan_into_c32_s32 | tenferro-cuda-direct | 256 | 15 | 3.682 | 0.0004148 |
| einsum_plan_read_into_c32_s32 | tenferro-cuda-direct | 256 | 15 | 3.957 | 0.0004148 |
| einsum_plan_read_into_accum_c32_s32 | tenferro-cuda-direct | 256 | 15 | 4.173 | 0.0001409 |
| einsum_plan_strided_read_into_c32_s32 | tenferro-cuda-direct | 64 | 15 | 3.375 | 0.0004148 |
| einsum_plan_strided_read_into_accum_c32_s32 | tenferro-cuda-direct | 64 | 15 | 3.551 | 0.0001409 |
| einsum_plan_owned_c64_s32 | tenferro-cuda-direct | 128 | 15 | 2.552 | 1.978e-07 |
| einsum_plan_read_c64_s32 | tenferro-cuda-direct | 128 | 15 | 2.607 | 1.978e-07 |
| einsum_plan_strided_read_c64_s32 | tenferro-cuda-direct | 64 | 15 | 3.935 | 1.978e-07 |
| einsum_plan_into_c64_s32 | tenferro-cuda-direct | 256 | 15 | 3.618 | 1.978e-07 |
| einsum_plan_read_into_c64_s32 | tenferro-cuda-direct | 256 | 15 | 3.749 | 1.978e-07 |
| einsum_plan_read_into_accum_c64_s32 | tenferro-cuda-direct | 128 | 15 | 2.107 | 1.074e-07 |
| einsum_plan_strided_read_into_c64_s32 | tenferro-cuda-direct | 64 | 15 | 3.764 | 1.978e-07 |
| einsum_plan_strided_read_into_accum_c64_s32 | tenferro-cuda-direct | 64 | 15 | 3.680 | 1.074e-07 |
| einsum_plan_owned_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.787 | 0.0009425 |
| einsum_plan_read_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.780 | 0.0009425 |
| einsum_plan_strided_read_f32_s256 | tenferro-cuda-direct | 64 | 15 | 4.168 | 0.0009425 |
| einsum_plan_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.204 | 0.0009425 |
| einsum_plan_read_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.219 | 0.0009425 |
| einsum_plan_read_into_accum_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.550 | 0.0003475 |
| einsum_plan_strided_read_into_f32_s256 | tenferro-cuda-direct | 64 | 15 | 3.601 | 0.0009425 |
| einsum_plan_strided_read_into_accum_f32_s256 | tenferro-cuda-direct | 64 | 15 | 3.719 | 0.0003475 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.759 | 0 |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.777 | 0 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | 32 | 15 | 1.930 | 0 |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.100 | 0 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.108 | 0 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | 128 | 15 | 2.362 | 0 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | 64 | 15 | 3.863 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | 32 | 15 | 2.069 | 0 |
| einsum_plan_owned_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.960 | 0.001882 |
| einsum_plan_read_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.891 | 0.001882 |
| einsum_plan_strided_read_c32_s256 | tenferro-cuda-direct | 32 | 15 | 2.364 | 0.001882 |
| einsum_plan_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.117 | 0.001882 |
| einsum_plan_read_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.131 | 0.001882 |
| einsum_plan_read_into_accum_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.439 | 0.0006637 |
| einsum_plan_strided_read_into_c32_s256 | tenferro-cuda-direct | 32 | 15 | 2.210 | 0.001882 |
| einsum_plan_strided_read_into_accum_c32_s256 | tenferro-cuda-direct | 32 | 15 | 2.114 | 0.0006637 |
| einsum_plan_owned_c64_s256 | tenferro-cuda-direct | 128 | 15 | 3.368 | 1.972e-06 |
| einsum_plan_read_c64_s256 | tenferro-cuda-direct | 128 | 15 | 3.366 | 1.972e-06 |
| einsum_plan_strided_read_c64_s256 | tenferro-cuda-direct | 32 | 15 | 2.121 | 1.972e-06 |
| einsum_plan_into_c64_s256 | tenferro-cuda-direct | 128 | 15 | 2.718 | 1.972e-06 |
| einsum_plan_read_into_c64_s256 | tenferro-cuda-direct | 128 | 15 | 2.805 | 1.972e-06 |
| einsum_plan_read_into_accum_c64_s256 | tenferro-cuda-direct | 128 | 15 | 2.911 | 8.162e-07 |
| einsum_plan_strided_read_into_c64_s256 | tenferro-cuda-direct | 32 | 15 | 1.992 | 1.972e-06 |
| einsum_plan_strided_read_into_accum_c64_s256 | tenferro-cuda-direct | 32 | 15 | 2.074 | 8.162e-07 |
| sub_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.046 | 0 |
| sub_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.405 | 0 |
| div_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.168 | 0 |
| div_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.135 | 0 |
| rem_strided_read_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.214 | 0 |
| rem_strided_read_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.417 | 0 |
| pow_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.779 | 0.000117 |
| pow_strided_read_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.064 | 1.48e-07 |
| maximum_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.755 | 0 |
| maximum_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.917 | 0 |
| minimum_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.850 | 0 |
| minimum_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 4.002 | 0 |
| abs_strided_read_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 4.165 | 0 |
| abs_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.220 | 0 |
| sign_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.093 | 0 |
| sign_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.247 | 0 |
| expm1_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.148 | 7.911e-05 |
| expm1_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.234 | 7.39e-08 |
| log_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.099 | 7.945e-05 |
| log_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.241 | 4.231e-08 |
| log1p_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.097 | 0.0001189 |
| log1p_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.237 | 1.108e-07 |
| erf_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.137 | 7.914e-05 |
| erf_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.287 | 8.536e-08 |
| sin_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.142 | 4.348e-05 |
| sin_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.280 | 4.153e-08 |
| cos_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.145 | 6.342e-05 |
| cos_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.237 | 5.905e-08 |
| sqrt_strided_read_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 4.020 | 0 |
| sqrt_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.227 | 0 |
| rsqrt_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.009 | 6.983e-05 |
| rsqrt_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.255 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.213 | 0.0004769 |
| reduce_prod_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.248 | 5.55e-07 |
| reduce_max_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.333 | 0 |
| reduce_max_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.313 | 0 |
| reduce_min_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.216 | 0 |
| reduce_min_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.264 | 0 |
| reduce_sum_squares_strided_read_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.086 | 0.0001524 |
| reduce_sum_squares_strided_read_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.173 | 2.804e-07 |
| sub_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.721 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.617 | 0 |
| div_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.702 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.620 | 0 |
| rem_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.382 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.024 | 0 |
| pow_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.158 | 0.0001577 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.047 | 1.48e-07 |
| maximum_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.016 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.841 | 0 |
| minimum_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.022 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.822 | 0 |
| abs_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.698 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.080 | 0 |
| sign_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.717 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.079 | 0 |
| expm1_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.751 | 7.915e-05 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.157 | 7.39e-08 |
| log_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.744 | 7.946e-05 |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.213 | 4.308e-08 |
| log1p_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.743 | 0.000119 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.236 | 1.11e-07 |
| erf_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.742 | 7.923e-05 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.279 | 8.536e-08 |
| sin_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.687 | 4.348e-05 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.143 | 4.153e-08 |
| cos_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.712 | 6.342e-05 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.118 | 5.905e-08 |
| sqrt_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.688 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.099 | 0 |
| rsqrt_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.707 | 6.983e-05 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.085 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.920 | 0.008463 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.180 | 4.661e-06 |
| reduce_max_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.945 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.085 | 0 |
| reduce_min_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.918 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.087 | 0 |
| reduce_sum_squares_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.914 | 0.0003552 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.189 | 3.312e-07 |
| add_into_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| tensordot_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.983 | 0.000577 |
| tensordot_f64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.374 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.609 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-eager | 128 | 15 | 3.587 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| tensordot_f32_n262144 | tenferro-cuda-eager | 64 | 15 | 3.131 | 0.01125 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 64 | 15 | 3.021 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-eager | 32 | 15 | 2.661 | 0.01203 |
| tensordot_c64_n262144 | tenferro-cuda-eager | 32 | 15 | 3.207 | 7.381e-06 |
| vdot_read_f32_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_f64_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c32_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c64_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f32_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f64_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| axpby_read_into_accum_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f32_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c32_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c64_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| dot_general_read_into_f32_s16 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f32_s16 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f32_s16 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s16 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s16 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s16 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_c32_s16 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_c32_s16 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_c32_s16 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_c64_s16 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_c64_s16 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_c64_s16 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f32_s256 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f32_s256 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f32_s256 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_c32_s256 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_c32_s256 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_c32_s256 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_c64_s256 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_c64_s256 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_c64_s256 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| add_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| add_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| mul_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| mul_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| neg_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| neg_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| exp_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| exp_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tanh_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tanh_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| conj_strided_read_c32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| conj_strided_read_c64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| add_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| add_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| mul_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| mul_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| neg_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| neg_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| exp_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| exp_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tanh_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| conj_strided_read_c32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| conj_strided_read_c64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| axpby_overwrite_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_lerp_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_unit_add_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_overwrite_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_lerp_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_unit_add_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_overwrite_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_lerp_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_unit_add_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_overwrite_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_lerp_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_unit_add_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_overwrite_f32_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_lerp_f32_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_unit_add_f32_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_overwrite_f64_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_lerp_f64_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_unit_add_f64_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_overwrite_c32_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_lerp_c32_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_unit_add_c32_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_overwrite_c64_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_lerp_c64_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| axpby_unit_add_c64_n1048576 | tenferro-cuda-eager | — | — | — | No matching public caller-owned output API. |
| dot_general_batch_f32_s32 | tenferro-cuda-eager | 128 | 15 | 3.180 | 0.0002227 |
| dot_general_batch_f64_s32 | tenferro-cuda-eager | 128 | 15 | 3.278 | 1.326e-07 |
| dot_general_batch_c32_s32 | tenferro-cuda-eager | 128 | 15 | 3.464 | 0.0004376 |
| dot_general_batch_c64_s32 | tenferro-cuda-eager | 128 | 15 | 3.478 | 4.296e-07 |
| dot_general_multiple_f32_s32 | tenferro-cuda-eager | 64 | 15 | 2.485 | 0.0002162 |
| dot_general_multiple_f64_s32 | tenferro-cuda-eager | 64 | 15 | 2.298 | 1.632e-07 |
| dot_general_multiple_c32_s32 | tenferro-cuda-eager | 64 | 15 | 2.178 | 0.0002794 |
| dot_general_multiple_c64_s32 | tenferro-cuda-eager | 64 | 15 | 2.448 | 4.303e-07 |
| dot_general_batch_f32_s256 | tenferro-cuda-eager | 64 | 15 | 3.043 | 0.0008561 |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | 64 | 15 | 1.927 | 7.463e-07 |
| dot_general_batch_c32_s256 | tenferro-cuda-eager | 64 | 15 | 2.621 | 0.001749 |
| dot_general_batch_c64_s256 | tenferro-cuda-eager | 64 | 15 | 2.518 | 1.665e-06 |
| dot_general_multiple_f32_s256 | tenferro-cuda-eager | 64 | 15 | 2.835 | 0.001306 |
| dot_general_multiple_f64_s256 | tenferro-cuda-eager | 64 | 15 | 2.763 | 1.803e-06 |
| dot_general_multiple_c32_s256 | tenferro-cuda-eager | 32 | 15 | 1.903 | 0.0022 |
| dot_general_multiple_c64_s256 | tenferro-cuda-eager | 64 | 15 | 3.281 | 9.057e-07 |
| dot_conj_l0r0_c32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.741 | 0.0003083 |
| dot_conj_l1r0_c32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.942 | 0.0002905 |
| dot_conj_l0r1_c32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.762 | 0.0002905 |
| dot_conj_l1r1_c32_n1024 | tenferro-cuda-eager | 128 | 15 | 2.760 | 0.0003083 |
| dot_conj_l0r0_c64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.862 | 4.927e-07 |
| dot_conj_l1r0_c64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.957 | 1.807e-07 |
| dot_conj_l0r1_c64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.878 | 1.807e-07 |
| dot_conj_l1r1_c64_n1024 | tenferro-cuda-eager | 128 | 15 | 2.874 | 4.927e-07 |
| dot_conj_l0r0_c32_n262144 | tenferro-cuda-eager | 64 | 15 | 2.260 | 0.005822 |
| dot_conj_l1r0_c32_n262144 | tenferro-cuda-eager | 64 | 15 | 2.533 | 0.005619 |
| dot_conj_l0r1_c32_n262144 | tenferro-cuda-eager | 64 | 15 | 2.537 | 0.005619 |
| dot_conj_l1r1_c32_n262144 | tenferro-cuda-eager | 64 | 15 | 2.637 | 0.005822 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | 64 | 15 | 2.332 | 1.212e-05 |
| dot_conj_l1r0_c64_n262144 | tenferro-cuda-eager | 64 | 15 | 2.332 | 1.516e-05 |
| dot_conj_l0r1_c64_n262144 | tenferro-cuda-eager | 64 | 15 | 2.333 | 1.516e-05 |
| dot_conj_l1r1_c64_n262144 | tenferro-cuda-eager | 64 | 15 | 2.337 | 1.212e-05 |
| einsum_plan_owned_f32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_f32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_f32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_into_f32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_f32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_f32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_f32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_f32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_f64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_f64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_f64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_into_f64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_f64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_f64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_f64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_f64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_c32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_c32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_c32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_into_c32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_c32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_c32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_c32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_c32_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_c64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_c64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_c64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_into_c64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_c64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_c64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_c64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_c64_s32 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_f32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_f32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_f32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_into_f32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_f32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_f32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_f32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_f32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_f64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_f64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_into_f64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_c32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_c32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_c32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_into_c32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_c32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_c32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_c32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_c32_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_owned_c64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_c64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_c64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_into_c64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_c64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_read_into_accum_c64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_c64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| einsum_plan_strided_read_into_accum_c64_s256 | tenferro-cuda-eager | — | — | — | ConcreteEinsumPlan accepts concrete Tensor/TensorRead; no equivalent public prepared EagerTensor plan execution API is established. |
| sub_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sub_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| div_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| div_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rem_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rem_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| pow_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| pow_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| maximum_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| maximum_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| minimum_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| minimum_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| abs_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| abs_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sign_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sign_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| expm1_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| expm1_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log1p_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log1p_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| erf_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| erf_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sin_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sin_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| cos_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| cos_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sqrt_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sqrt_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rsqrt_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rsqrt_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_prod_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_prod_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_max_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_max_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_min_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_min_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_squares_strided_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_squares_strided_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sub_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sub_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| div_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| div_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rem_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rem_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| pow_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| pow_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| maximum_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| minimum_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| abs_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| abs_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sign_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sign_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| expm1_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log1p_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| erf_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| erf_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sin_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sin_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| cos_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| cos_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sqrt_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rsqrt_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_prod_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_max_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_min_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_squares_strided_read_f32_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-eager | — | — | — | No equivalent public borrowed TensorRead route; an owned EagerTensor is a different input representation. |
| add_into_f32_n1024 | pytorch-cuda | 512 | 15 | 2.281 | 0 |
| add_into_f64_n1024 | pytorch-cuda | 512 | 15 | 2.317 | 0 |
| add_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 2.289 | 0 |
| add_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 2.291 | 0 |
| sub_into_f32_n1024 | pytorch-cuda | 512 | 15 | 2.325 | 0 |
| sub_into_f64_n1024 | pytorch-cuda | 512 | 15 | 2.310 | 0 |
| sub_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 2.342 | 0 |
| sub_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 2.306 | 0 |
| mul_into_f32_n1024 | pytorch-cuda | 512 | 15 | 2.307 | 0 |
| mul_into_f64_n1024 | pytorch-cuda | 512 | 15 | 2.355 | 0 |
| mul_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 2.372 | 0 |
| mul_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 2.368 | 0 |
| div_into_f32_n1024 | pytorch-cuda | 512 | 15 | 2.295 | 0 |
| div_into_f64_n1024 | pytorch-cuda | 512 | 15 | 2.305 | 0 |
| div_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 2.291 | 0 |
| div_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 2.293 | 0 |
| neg_into_f32_n1024 | pytorch-cuda | 512 | 15 | 2.263 | 0 |
| neg_into_f64_n1024 | pytorch-cuda | 512 | 15 | 2.272 | 0 |
| neg_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 2.264 | 0 |
| neg_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 2.261 | 0 |
| conj_into_c32_n1024 | pytorch-cuda | 512 | 15 | 2.402 | 0 |
| conj_into_c64_n1024 | pytorch-cuda | 512 | 15 | 2.498 | 0 |
| conj_read_into_c32_n1024 | pytorch-cuda | 512 | 15 | 2.349 | 0 |
| conj_read_into_c64_n1024 | pytorch-cuda | 512 | 15 | 2.453 | 0 |
| add_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.356 | 0 |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.402 | 0 |
| add_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.355 | 0 |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.399 | 0 |
| sub_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.355 | 0 |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.403 | 0 |
| sub_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.354 | 0 |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.400 | 0 |
| mul_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.355 | 0 |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.407 | 0 |
| mul_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.355 | 0 |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.404 | 0 |
| div_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.356 | 0 |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.424 | 0 |
| div_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.356 | 0 |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.425 | 0 |
| neg_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.316 | 0 |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.244 | 0 |
| neg_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.315 | 0 |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.247 | 0 |
| conj_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 0.247 | 0 |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.403 | 0 |
| conj_read_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 0.246 | 0 |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.403 | 0 |
| tensordot_f32_n1024 | pytorch-cuda | 128 | 15 | 3.342 | 0.000577 |
| tensordot_f64_n1024 | pytorch-cuda | 128 | 15 | 3.368 | 0 |
| tensordot_c32_n1024 | pytorch-cuda | 128 | 15 | 3.317 | 0.0005344 |
| tensordot_c64_n1024 | pytorch-cuda | 128 | 15 | 3.341 | 0 |
| vdot_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.387 | 0 |
| vdot_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.392 | 1.722e-07 |
| vdot_read_c32_n1024 | pytorch-cuda | 256 | 15 | 2.400 | 0 |
| vdot_read_c64_n1024 | pytorch-cuda | 256 | 15 | 2.404 | 3.455e-07 |
| norm_squared_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.348 | 0 |
| norm_squared_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.318 | 0 |
| tensordot_f32_n262144 | pytorch-cuda | 128 | 15 | 3.546 | 0.01125 |
| tensordot_f64_n262144 | pytorch-cuda | 128 | 15 | 3.441 | 6.714e-06 |
| tensordot_c32_n262144 | pytorch-cuda | 32 | 15 | 3.282 | 0.00718 |
| tensordot_c64_n262144 | pytorch-cuda | 32 | 15 | 2.532 | 7.381e-06 |
| vdot_read_f32_n262144 | pytorch-cuda | 256 | 15 | 2.425 | 0 |
| vdot_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.524 | 2.995e-06 |
| vdot_read_c32_n262144 | pytorch-cuda | 256 | 15 | 2.506 | 0 |
| vdot_read_c64_n262144 | pytorch-cuda | 256 | 15 | 3.385 | 8.772e-06 |
| norm_squared_read_f32_n262144 | pytorch-cuda | 256 | 15 | 2.395 | 0.0001786 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 256 | 15 | 2.456 | 1.664e-07 |
| axpby_read_into_accum_f32_n1024 | pytorch-cuda | 512 | 15 | 2.677 | 0 |
| axpby_read_into_accum_f64_n1024 | pytorch-cuda | 512 | 15 | 2.647 | 0 |
| axpby_read_into_accum_c32_n1024 | pytorch-cuda | 512 | 15 | 2.612 | 0 |
| axpby_read_into_accum_c64_n1024 | pytorch-cuda | 512 | 15 | 2.697 | 0 |
| axpby_read_into_accum_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.544 | 0 |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.518 | 0 |
| axpby_read_into_accum_c32_n1048576 | pytorch-cuda | 32 | 15 | 0.520 | 0 |
| axpby_read_into_accum_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.580 | 0 |
| dot_general_read_into_f32_s16 | pytorch-cuda | 256 | 15 | 2.487 | 0.0001207 |
| dot_general_read_into_accum_f32_s16 | pytorch-cuda | 256 | 15 | 2.767 | 9.558e-05 |
| dot_general_read_into_accum_cached_f32_s16 | pytorch-cuda | 256 | 15 | 2.754 | 9.558e-05 |
| dot_general_read_into_f64_s16 | pytorch-cuda | 256 | 15 | 2.484 | 0 |
| dot_general_read_into_accum_f64_s16 | pytorch-cuda | 256 | 15 | 2.901 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | pytorch-cuda | 256 | 15 | 2.914 | 0 |
| dot_general_read_into_c32_s16 | pytorch-cuda | 256 | 15 | 2.502 | 0.0001647 |
| dot_general_read_into_accum_c32_s16 | pytorch-cuda | 256 | 15 | 2.752 | 0.0001125 |
| dot_general_read_into_accum_cached_c32_s16 | pytorch-cuda | 256 | 15 | 2.768 | 0.0001125 |
| dot_general_read_into_c64_s16 | pytorch-cuda | 256 | 15 | 2.617 | 1.373e-07 |
| dot_general_read_into_accum_c64_s16 | pytorch-cuda | 256 | 15 | 2.933 | 7.37e-08 |
| dot_general_read_into_accum_cached_c64_s16 | pytorch-cuda | 256 | 15 | 2.911 | 7.37e-08 |
| dot_general_read_into_f32_s256 | pytorch-cuda | 256 | 15 | 3.149 | 0.0003909 |
| dot_general_read_into_accum_f32_s256 | pytorch-cuda | 256 | 15 | 3.396 | 0.0002334 |
| dot_general_read_into_accum_cached_f32_s256 | pytorch-cuda | 256 | 15 | 3.399 | 0.0002334 |
| dot_general_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.546 | 0 |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 2.609 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 256 | 15 | 2.958 | 0 |
| dot_general_read_into_c32_s256 | pytorch-cuda | 128 | 15 | 2.327 | 0.001008 |
| dot_general_read_into_accum_c32_s256 | pytorch-cuda | 128 | 15 | 2.433 | 0.0005527 |
| dot_general_read_into_accum_cached_c32_s256 | pytorch-cuda | 128 | 15 | 2.433 | 0.0005527 |
| dot_general_read_into_c64_s256 | pytorch-cuda | 128 | 15 | 2.160 | 1.175e-06 |
| dot_general_read_into_accum_c64_s256 | pytorch-cuda | 128 | 15 | 2.327 | 6.523e-07 |
| dot_general_read_into_accum_cached_c64_s256 | pytorch-cuda | 128 | 15 | 2.337 | 6.523e-07 |
| add_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.615 | 0 |
| add_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.997 | 0 |
| mul_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.860 | 0 |
| mul_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.576 | 0 |
| neg_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.463 | 0 |
| neg_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.439 | 0 |
| exp_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.650 | 0.0001272 |
| exp_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.446 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.412 | 9.107e-05 |
| tanh_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.441 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 3.843 | 0.0005233 |
| reduce_sum_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.069 | 3.504e-07 |
| conj_strided_read_c32_n1024 | pytorch-cuda | 512 | 15 | 3.323 | 0 |
| conj_strided_read_c64_n1024 | pytorch-cuda | 512 | 15 | 3.408 | 0 |
| add_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.373 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.402 | 0 |
| mul_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.367 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.405 | 0 |
| neg_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.318 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.251 | 0 |
| exp_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.354 | 0.0001272 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.266 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.329 | 9.135e-05 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.264 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | pytorch-cuda | 256 | 15 | 3.069 | 0.002693 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.417 | 5.935e-06 |
| conj_strided_read_c32_n1048576 | pytorch-cuda | 32 | 15 | 0.253 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.404 | 0 |
| axpby_overwrite_f32_n1024 | pytorch-cuda | 512 | 15 | 2.652 | 0 |
| axpby_lerp_f32_n1024 | pytorch-cuda | 512 | 15 | 2.817 | 0.0001189 |
| axpby_unit_add_f32_n1024 | pytorch-cuda | 512 | 15 | 2.237 | 0 |
| axpby_overwrite_f64_n1024 | pytorch-cuda | 512 | 15 | 2.484 | 0 |
| axpby_lerp_f64_n1024 | pytorch-cuda | 512 | 15 | 2.481 | 1.161e-07 |
| axpby_unit_add_f64_n1024 | pytorch-cuda | 512 | 15 | 2.264 | 0 |
| axpby_overwrite_c32_n1024 | pytorch-cuda | 512 | 15 | 2.974 | 0 |
| axpby_lerp_c32_n1024 | pytorch-cuda | 512 | 15 | 2.201 | 0.0001189 |
| axpby_unit_add_c32_n1024 | pytorch-cuda | 512 | 15 | 2.316 | 0 |
| axpby_overwrite_c64_n1024 | pytorch-cuda | 512 | 15 | 2.610 | 0 |
| axpby_lerp_c64_n1024 | pytorch-cuda | 512 | 15 | 2.788 | 1.161e-07 |
| axpby_unit_add_c64_n1024 | pytorch-cuda | 512 | 15 | 2.363 | 0 |
| axpby_overwrite_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.327 | 0 |
| axpby_lerp_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.556 | 0.0001192 |
| axpby_unit_add_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.547 | 0 |
| axpby_overwrite_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.256 | 0 |
| axpby_lerp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.542 | 1.161e-07 |
| axpby_unit_add_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.517 | 0 |
| axpby_overwrite_c32_n1048576 | pytorch-cuda | 32 | 15 | 0.255 | 0 |
| axpby_lerp_c32_n1048576 | pytorch-cuda | 32 | 15 | 0.561 | 0.0001191 |
| axpby_unit_add_c32_n1048576 | pytorch-cuda | 32 | 15 | 0.517 | 0 |
| axpby_overwrite_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.407 | 0 |
| axpby_lerp_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.616 | 1.159e-07 |
| axpby_unit_add_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.578 | 0 |
| dot_general_batch_f32_s32 | pytorch-cuda | 128 | 15 | 2.150 | 0.0002227 |
| dot_general_batch_f64_s32 | pytorch-cuda | 128 | 15 | 2.317 | 1.326e-07 |
| dot_general_batch_c32_s32 | pytorch-cuda | 128 | 15 | 2.134 | 0.0003092 |
| dot_general_batch_c64_s32 | pytorch-cuda | 128 | 15 | 2.439 | 2.791e-07 |
| dot_general_multiple_f32_s32 | pytorch-cuda | 64 | 15 | 2.313 | 0.0001628 |
| dot_general_multiple_f64_s32 | pytorch-cuda | 64 | 15 | 2.327 | 2.466e-07 |
| dot_general_multiple_c32_s32 | pytorch-cuda | 64 | 15 | 2.352 | 0.0004058 |
| dot_general_multiple_c64_s32 | pytorch-cuda | 64 | 15 | 2.349 | 4.129e-07 |
| dot_general_batch_f32_s256 | pytorch-cuda | 64 | 15 | 2.226 | 0.0008561 |
| dot_general_batch_f64_s256 | pytorch-cuda | 128 | 15 | 2.834 | 7.463e-07 |
| dot_general_batch_c32_s256 | pytorch-cuda | 64 | 15 | 2.155 | 0.001286 |
| dot_general_batch_c64_s256 | pytorch-cuda | 64 | 15 | 2.240 | 8.893e-07 |
| dot_general_multiple_f32_s256 | pytorch-cuda | 64 | 15 | 2.351 | 0.00133 |
| dot_general_multiple_f64_s256 | pytorch-cuda | 64 | 15 | 2.371 | 2.584e-06 |
| dot_general_multiple_c32_s256 | pytorch-cuda | 64 | 15 | 2.567 | 0.001059 |
| dot_general_multiple_c64_s256 | pytorch-cuda | 64 | 15 | 2.486 | 2.776e-06 |
| dot_conj_l0r0_c32_n1024 | pytorch-cuda | 256 | 15 | 2.107 | 0.0001028 |
| dot_conj_l1r0_c32_n1024 | pytorch-cuda | 256 | 15 | 2.407 | 0.0004339 |
| dot_conj_l0r1_c32_n1024 | pytorch-cuda | 256 | 15 | 2.416 | 0.0004457 |
| dot_conj_l1r1_c32_n1024 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n1024 | pytorch-cuda | 256 | 15 | 2.114 | 4.333e-07 |
| dot_conj_l1r0_c64_n1024 | pytorch-cuda | 256 | 15 | 2.351 | 2.122e-07 |
| dot_conj_l0r1_c64_n1024 | pytorch-cuda | 256 | 15 | 2.365 | 2.847e-07 |
| dot_conj_l1r1_c64_n1024 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c32_n262144 | pytorch-cuda | 256 | 15 | 2.498 | 0.004445 |
| dot_conj_l1r0_c32_n262144 | pytorch-cuda | 256 | 15 | 2.501 | 0.007446 |
| dot_conj_l0r1_c32_n262144 | pytorch-cuda | 256 | 15 | 2.495 | 0.007339 |
| dot_conj_l1r1_c32_n262144 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | 256 | 15 | 3.386 | 1.366e-05 |
| dot_conj_l1r0_c64_n262144 | pytorch-cuda | 256 | 15 | 3.378 | 1.886e-05 |
| dot_conj_l0r1_c64_n262144 | pytorch-cuda | 256 | 15 | 3.382 | 2.282e-05 |
| dot_conj_l1r1_c64_n262144 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| einsum_plan_owned_f32_s32 | pytorch-cuda | 256 | 15 | 2.169 | 0.0002018 |
| einsum_plan_read_f32_s32 | pytorch-cuda | 256 | 15 | 2.173 | 0.0002018 |
| einsum_plan_strided_read_f32_s32 | pytorch-cuda | 256 | 15 | 2.149 | 0.0002018 |
| einsum_plan_into_f32_s32 | pytorch-cuda | 512 | 15 | 3.672 | 0.0002018 |
| einsum_plan_read_into_f32_s32 | pytorch-cuda | 512 | 15 | 3.584 | 0.0002018 |
| einsum_plan_read_into_accum_f32_s32 | pytorch-cuda | 256 | 15 | 2.341 | 0.0001134 |
| einsum_plan_strided_read_into_f32_s32 | pytorch-cuda | 512 | 15 | 3.536 | 0.0002018 |
| einsum_plan_strided_read_into_accum_f32_s32 | pytorch-cuda | 256 | 15 | 2.291 | 0.0001134 |
| einsum_plan_owned_f64_s32 | pytorch-cuda | 256 | 15 | 2.560 | 0 |
| einsum_plan_read_f64_s32 | pytorch-cuda | 256 | 15 | 2.534 | 0 |
| einsum_plan_strided_read_f64_s32 | pytorch-cuda | 256 | 15 | 2.231 | 0 |
| einsum_plan_into_f64_s32 | pytorch-cuda | 512 | 15 | 3.980 | 0 |
| einsum_plan_read_into_f64_s32 | pytorch-cuda | 256 | 15 | 1.984 | 0 |
| einsum_plan_read_into_accum_f64_s32 | pytorch-cuda | 256 | 15 | 2.451 | 0 |
| einsum_plan_strided_read_into_f64_s32 | pytorch-cuda | 512 | 15 | 3.833 | 0 |
| einsum_plan_strided_read_into_accum_f64_s32 | pytorch-cuda | 256 | 15 | 2.516 | 0 |
| einsum_plan_owned_c32_s32 | pytorch-cuda | 256 | 15 | 2.568 | 0.0002342 |
| einsum_plan_read_c32_s32 | pytorch-cuda | 256 | 15 | 2.744 | 0.0002342 |
| einsum_plan_strided_read_c32_s32 | pytorch-cuda | 256 | 15 | 2.564 | 0.0002342 |
| einsum_plan_into_c32_s32 | pytorch-cuda | 256 | 15 | 2.280 | 0.0002342 |
| einsum_plan_read_into_c32_s32 | pytorch-cuda | 256 | 15 | 2.373 | 0.0002342 |
| einsum_plan_read_into_accum_c32_s32 | pytorch-cuda | 256 | 15 | 2.797 | 0.0001157 |
| einsum_plan_strided_read_into_c32_s32 | pytorch-cuda | 256 | 15 | 2.183 | 0.0002342 |
| einsum_plan_strided_read_into_accum_c32_s32 | pytorch-cuda | 256 | 15 | 3.000 | 0.0001157 |
| einsum_plan_owned_c64_s32 | pytorch-cuda | 256 | 15 | 2.850 | 1.981e-07 |
| einsum_plan_read_c64_s32 | pytorch-cuda | 256 | 15 | 2.796 | 1.981e-07 |
| einsum_plan_strided_read_c64_s32 | pytorch-cuda | 256 | 15 | 2.924 | 1.981e-07 |
| einsum_plan_into_c64_s32 | pytorch-cuda | 256 | 15 | 2.082 | 1.981e-07 |
| einsum_plan_read_into_c64_s32 | pytorch-cuda | 256 | 15 | 2.226 | 1.981e-07 |
| einsum_plan_read_into_accum_c64_s32 | pytorch-cuda | 256 | 15 | 2.946 | 1.097e-07 |
| einsum_plan_strided_read_into_c64_s32 | pytorch-cuda | 256 | 15 | 2.513 | 1.981e-07 |
| einsum_plan_strided_read_into_accum_c64_s32 | pytorch-cuda | 256 | 15 | 3.284 | 1.097e-07 |
| einsum_plan_owned_f32_s256 | pytorch-cuda | 256 | 15 | 2.562 | 0.0009425 |
| einsum_plan_read_f32_s256 | pytorch-cuda | 256 | 15 | 2.634 | 0.0009425 |
| einsum_plan_strided_read_f32_s256 | pytorch-cuda | 256 | 15 | 2.726 | 0.0009425 |
| einsum_plan_into_f32_s256 | pytorch-cuda | 256 | 15 | 2.381 | 0.0009425 |
| einsum_plan_read_into_f32_s256 | pytorch-cuda | 256 | 15 | 2.084 | 0.0009425 |
| einsum_plan_read_into_accum_f32_s256 | pytorch-cuda | 256 | 15 | 2.724 | 0.0003475 |
| einsum_plan_strided_read_into_f32_s256 | pytorch-cuda | 512 | 15 | 4.045 | 0.0009425 |
| einsum_plan_strided_read_into_accum_f32_s256 | pytorch-cuda | 256 | 15 | 2.698 | 0.0003475 |
| einsum_plan_owned_f64_s256 | pytorch-cuda | 256 | 15 | 2.315 | 0 |
| einsum_plan_read_f64_s256 | pytorch-cuda | 256 | 15 | 2.714 | 0 |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | 256 | 15 | 2.384 | 0 |
| einsum_plan_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.055 | 0 |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.291 | 0 |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 3.054 | 0 |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | 256 | 15 | 2.420 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | 256 | 15 | 3.104 | 0 |
| einsum_plan_owned_c32_s256 | pytorch-cuda | 256 | 15 | 3.905 | 0.001524 |
| einsum_plan_read_c32_s256 | pytorch-cuda | 256 | 15 | 3.907 | 0.001524 |
| einsum_plan_strided_read_c32_s256 | pytorch-cuda | 256 | 15 | 3.885 | 0.001524 |
| einsum_plan_into_c32_s256 | pytorch-cuda | 256 | 15 | 3.901 | 0.001524 |
| einsum_plan_read_into_c32_s256 | pytorch-cuda | 256 | 15 | 3.903 | 0.001524 |
| einsum_plan_read_into_accum_c32_s256 | pytorch-cuda | 128 | 15 | 2.108 | 0.0004028 |
| einsum_plan_strided_read_into_c32_s256 | pytorch-cuda | 256 | 15 | 3.887 | 0.001524 |
| einsum_plan_strided_read_into_accum_c32_s256 | pytorch-cuda | 128 | 15 | 2.088 | 0.0004028 |
| einsum_plan_owned_c64_s256 | pytorch-cuda | 128 | 15 | 2.240 | 8.397e-07 |
| einsum_plan_read_c64_s256 | pytorch-cuda | 128 | 15 | 2.232 | 8.397e-07 |
| einsum_plan_strided_read_c64_s256 | pytorch-cuda | 128 | 15 | 2.228 | 8.397e-07 |
| einsum_plan_into_c64_s256 | pytorch-cuda | 128 | 15 | 2.226 | 8.397e-07 |
| einsum_plan_read_into_c64_s256 | pytorch-cuda | 128 | 15 | 2.230 | 8.397e-07 |
| einsum_plan_read_into_accum_c64_s256 | pytorch-cuda | 128 | 15 | 2.298 | 4.364e-07 |
| einsum_plan_strided_read_into_c64_s256 | pytorch-cuda | 128 | 15 | 2.217 | 8.397e-07 |
| einsum_plan_strided_read_into_accum_c64_s256 | pytorch-cuda | 128 | 15 | 2.291 | 4.364e-07 |
| sub_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.940 | 0 |
| sub_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.907 | 0 |
| div_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.901 | 0 |
| div_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.978 | 0 |
| rem_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 3.054 | 0 |
| rem_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.989 | 0 |
| pow_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 3.018 | 0.000117 |
| pow_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 3.712 | 1.48e-07 |
| maximum_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.999 | 0 |
| maximum_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.994 | 0 |
| minimum_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 3.057 | 0 |
| minimum_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 3.040 | 0 |
| abs_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 3.564 | 0 |
| abs_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 3.609 | 0 |
| sign_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.826 | 0 |
| sign_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.780 | 0 |
| expm1_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.861 | 7.911e-05 |
| expm1_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.817 | 7.39e-08 |
| log_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.728 | 7.945e-05 |
| log_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.728 | 4.231e-08 |
| log1p_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.726 | 0.0001189 |
| log1p_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.804 | 1.108e-07 |
| erf_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.831 | 7.914e-05 |
| erf_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.788 | 8.536e-08 |
| sin_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.789 | 4.348e-05 |
| sin_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.832 | 4.153e-08 |
| cos_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.791 | 6.342e-05 |
| cos_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.735 | 5.905e-08 |
| sqrt_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.746 | 0 |
| sqrt_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.798 | 0 |
| rsqrt_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 2.839 | 6.983e-05 |
| rsqrt_strided_read_f64_n1024 | pytorch-cuda | 512 | 15 | 2.796 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1024 | pytorch-cuda | 512 | 15 | 3.753 | 0.0005958 |
| reduce_prod_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.082 | 4.44e-07 |
| reduce_max_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.068 | 0 |
| reduce_max_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.218 | 0 |
| reduce_min_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.067 | 0 |
| reduce_min_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.218 | 0 |
| reduce_sum_squares_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 3.504 | 0.0001524 |
| reduce_sum_squares_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 3.574 | 2.836e-07 |
| sub_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.368 | 0 |
| sub_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.405 | 0 |
| div_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.365 | 0 |
| div_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.434 | 0 |
| rem_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.373 | 0 |
| rem_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.390 | 0 |
| pow_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.696 | 0.0001577 |
| pow_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.931 | 1.48e-07 |
| maximum_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.375 | 0 |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.389 | 0 |
| minimum_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.375 | 0 |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.389 | 0 |
| abs_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.451 | 0 |
| abs_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.255 | 0 |
| sign_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.354 | 0 |
| sign_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.252 | 0 |
| expm1_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.374 | 7.915e-05 |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.319 | 7.39e-08 |
| log_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.356 | 7.946e-05 |
| log_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.407 | 4.308e-08 |
| log1p_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.353 | 0.000119 |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.421 | 1.11e-07 |
| erf_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.331 | 7.923e-05 |
| erf_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.460 | 8.536e-08 |
| sin_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.372 | 4.348e-05 |
| sin_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.323 | 4.153e-08 |
| cos_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.372 | 6.342e-05 |
| cos_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.326 | 5.905e-08 |
| sqrt_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.353 | 0 |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.254 | 0 |
| rsqrt_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.350 | 6.983e-05 |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.253 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1048576 | pytorch-cuda | 256 | 15 | 3.100 | 0.009062 |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.406 | 4.106e-06 |
| reduce_max_strided_read_f32_n1048576 | pytorch-cuda | 256 | 15 | 3.090 | 0 |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.432 | 0 |
| reduce_min_strided_read_f32_n1048576 | pytorch-cuda | 256 | 15 | 3.089 | 0 |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | 256 | 15 | 3.432 | 0 |
| reduce_sum_squares_strided_read_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.188 | 0.0001778 |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.624 | 3.309e-07 |
| add_into_f32_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f32_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f32_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f32_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f32_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f32_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f32_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f32_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f32_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f32_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c32_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c32_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1024 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f32_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f32_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f32_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f32_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f32_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f32_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f32_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f32_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f32_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f32_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c32_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c32_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | jax-cuda | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| tensordot_f32_n1024 | jax-cuda | 32 | 15 | 3.717 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda | 32 | 15 | 3.675 | 0 |
| tensordot_c32_n1024 | jax-cuda | 16 | 15 | 2.084 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda | 16 | 15 | 2.120 | 0 |
| vdot_read_f32_n1024 | jax-cuda | 64 | 15 | 2.775 | 0 |
| vdot_read_f64_n1024 | jax-cuda | 64 | 15 | 2.778 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda | 64 | 15 | 2.775 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda | 64 | 15 | 2.845 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda | 64 | 15 | 2.756 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda | 64 | 15 | 2.462 | 0 |
| tensordot_f32_n262144 | jax-cuda | 32 | 15 | 3.731 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda | 16 | 15 | 2.111 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda | 16 | 15 | 2.218 | 0.01974 |
| tensordot_c64_n262144 | jax-cuda | 16 | 15 | 2.261 | 6.383e-06 |
| vdot_read_f32_n262144 | jax-cuda | 64 | 15 | 3.006 | 0 |
| vdot_read_f64_n262144 | jax-cuda | 64 | 15 | 3.078 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda | 64 | 15 | 3.054 | 0 |
| vdot_read_c64_n262144 | jax-cuda | 64 | 15 | 3.119 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda | 64 | 15 | 3.031 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda | 64 | 15 | 3.092 | 1.664e-07 |
| axpby_read_into_accum_f32_n1024 | jax-cuda | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f64_n1024 | jax-cuda | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_c32_n1024 | jax-cuda | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_c64_n1024 | jax-cuda | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_c32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_c64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| dot_general_read_into_f32_s16 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f32_s16 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f32_s16 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s16 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s16 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s16 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_c32_s16 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_c32_s16 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_c32_s16 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_c64_s16 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_c64_s16 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_c64_s16 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f32_s256 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f32_s256 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f32_s256 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s256 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_c32_s256 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_c32_s256 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_c32_s256 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_c64_s256 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_c64_s256 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_c64_s256 | jax-cuda | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| add_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 3.187 | 0 |
| add_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 3.178 | 0 |
| mul_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 3.195 | 0 |
| mul_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 3.165 | 0 |
| neg_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 3.056 | 0 |
| neg_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 3.025 | 0 |
| exp_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.535 | 0.0001272 |
| exp_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.566 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.553 | 9.325e-05 |
| tanh_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.554 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.913 | 0.0002267 |
| reduce_sum_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.950 | 3.268e-07 |
| conj_strided_read_c32_n1024 | jax-cuda | 64 | 15 | 2.588 | 0 |
| conj_strided_read_c64_n1024 | jax-cuda | 64 | 15 | 2.574 | 0 |
| add_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 1.642 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.640 | 0 |
| mul_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 3.265 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.665 | 0 |
| neg_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 3.195 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.602 | 0 |
| exp_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 2.644 | 0.0001272 |
| exp_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.342 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 2.658 | 9.508e-05 |
| tanh_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.336 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 3.045 | 0.003094 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.016 | 7.159e-06 |
| conj_strided_read_c32_n1048576 | jax-cuda | 32 | 15 | 1.338 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 0.654 | 0 |
| axpby_overwrite_f32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1024 | jax-cuda | 16 | 15 | 3.551 | 0 |
| axpby_overwrite_f64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1024 | jax-cuda | 8 | 15 | 1.874 | 0 |
| axpby_overwrite_c32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1024 | jax-cuda | 16 | 15 | 3.601 | 0 |
| axpby_overwrite_c64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1024 | jax-cuda | 16 | 15 | 3.586 | 0 |
| axpby_overwrite_f32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1048576 | jax-cuda | 16 | 15 | 3.597 | 0 |
| axpby_overwrite_f64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1048576 | jax-cuda | 16 | 15 | 3.602 | 0 |
| axpby_overwrite_c32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1048576 | jax-cuda | 16 | 15 | 3.579 | 0 |
| axpby_overwrite_c64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1048576 | jax-cuda | 15 | 15 | 2.990 | 0 |
| dot_general_batch_f32_s32 | jax-cuda | 32 | 15 | 2.659 | 0.0002227 |
| dot_general_batch_f64_s32 | jax-cuda | 32 | 15 | 2.634 | 1.326e-07 |
| dot_general_batch_c32_s32 | jax-cuda | 32 | 15 | 3.260 | 0.0004354 |
| dot_general_batch_c64_s32 | jax-cuda | 32 | 15 | 2.772 | 2.524e-07 |
| dot_general_multiple_f32_s32 | jax-cuda | 32 | 15 | 2.881 | 0.000269 |
| dot_general_multiple_f64_s32 | jax-cuda | 32 | 15 | 2.931 | 2.466e-07 |
| dot_general_multiple_c32_s32 | jax-cuda | 32 | 15 | 3.415 | 0.0004771 |
| dot_general_multiple_c64_s32 | jax-cuda | 32 | 15 | 3.405 | 4.129e-07 |
| dot_general_batch_f32_s256 | jax-cuda | 32 | 15 | 3.182 | 0.0008561 |
| dot_general_batch_f64_s256 | jax-cuda | 32 | 15 | 2.706 | 7.463e-07 |
| dot_general_batch_c32_s256 | jax-cuda | 32 | 15 | 3.312 | 0.001359 |
| dot_general_batch_c64_s256 | jax-cuda | 32 | 15 | 3.417 | 1.869e-06 |
| dot_general_multiple_f32_s256 | jax-cuda | 32 | 15 | 2.806 | 0.0025 |
| dot_general_multiple_f64_s256 | jax-cuda | 32 | 15 | 2.938 | 2.584e-06 |
| dot_general_multiple_c32_s256 | jax-cuda | 32 | 15 | 3.447 | 0.002465 |
| dot_general_multiple_c64_s256 | jax-cuda | 32 | 15 | 3.546 | 1.805e-06 |
| dot_conj_l0r0_c32_n1024 | jax-cuda | 64 | 15 | 2.705 | 0.0002298 |
| dot_conj_l1r0_c32_n1024 | jax-cuda | 64 | 15 | 2.782 | 0.0002522 |
| dot_conj_l0r1_c32_n1024 | jax-cuda | 64 | 15 | 2.792 | 0.0002522 |
| dot_conj_l1r1_c32_n1024 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n1024 | jax-cuda | 64 | 15 | 2.867 | 3.644e-07 |
| dot_conj_l1r0_c64_n1024 | jax-cuda | 64 | 15 | 2.879 | 2.767e-07 |
| dot_conj_l0r1_c64_n1024 | jax-cuda | 64 | 15 | 2.897 | 2.767e-07 |
| dot_conj_l1r1_c64_n1024 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c32_n262144 | jax-cuda | 64 | 15 | 3.087 | 0.01276 |
| dot_conj_l1r0_c32_n262144 | jax-cuda | 64 | 15 | 3.100 | 0.0002732 |
| dot_conj_l0r1_c32_n262144 | jax-cuda | 64 | 15 | 3.109 | 0.0002732 |
| dot_conj_l1r1_c32_n262144 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda | 64 | 15 | 3.143 | 1.221e-05 |
| dot_conj_l1r0_c64_n262144 | jax-cuda | 64 | 15 | 3.178 | 1.586e-05 |
| dot_conj_l0r1_c64_n262144 | jax-cuda | 64 | 15 | 3.206 | 1.586e-05 |
| dot_conj_l1r1_c64_n262144 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| einsum_plan_owned_f32_s32 | jax-cuda | 64 | 15 | 2.602 | 0.0002018 |
| einsum_plan_read_f32_s32 | jax-cuda | 64 | 15 | 2.628 | 0.0002018 |
| einsum_plan_strided_read_f32_s32 | jax-cuda | 32 | 15 | 1.690 | 0.0002018 |
| einsum_plan_into_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f64_s32 | jax-cuda | 64 | 15 | 2.573 | 0 |
| einsum_plan_read_f64_s32 | jax-cuda | 64 | 15 | 2.613 | 0 |
| einsum_plan_strided_read_f64_s32 | jax-cuda | 64 | 15 | 2.638 | 0 |
| einsum_plan_into_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c32_s32 | jax-cuda | 64 | 15 | 3.399 | 0.0002532 |
| einsum_plan_read_c32_s32 | jax-cuda | 64 | 15 | 3.444 | 0.0002532 |
| einsum_plan_strided_read_c32_s32 | jax-cuda | 64 | 15 | 3.412 | 0.0002532 |
| einsum_plan_into_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c64_s32 | jax-cuda | 64 | 15 | 3.549 | 3.753e-07 |
| einsum_plan_read_c64_s32 | jax-cuda | 64 | 15 | 3.577 | 3.753e-07 |
| einsum_plan_strided_read_c64_s32 | jax-cuda | 64 | 15 | 3.586 | 3.753e-07 |
| einsum_plan_into_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f32_s256 | jax-cuda | 64 | 15 | 2.654 | 0.00118 |
| einsum_plan_read_f32_s256 | jax-cuda | 64 | 15 | 2.676 | 0.00118 |
| einsum_plan_strided_read_f32_s256 | jax-cuda | 64 | 15 | 2.670 | 0.00118 |
| einsum_plan_into_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f64_s256 | jax-cuda | 64 | 15 | 2.679 | 0 |
| einsum_plan_read_f64_s256 | jax-cuda | 64 | 15 | 2.704 | 0 |
| einsum_plan_strided_read_f64_s256 | jax-cuda | 32 | 15 | 1.542 | 0 |
| einsum_plan_into_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c32_s256 | jax-cuda | 64 | 15 | 3.320 | 0.002017 |
| einsum_plan_read_c32_s256 | jax-cuda | 64 | 15 | 3.398 | 0.002017 |
| einsum_plan_strided_read_c32_s256 | jax-cuda | 64 | 15 | 3.371 | 0.002017 |
| einsum_plan_into_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c64_s256 | jax-cuda | 64 | 15 | 3.813 | 1.855e-06 |
| einsum_plan_read_c64_s256 | jax-cuda | 64 | 15 | 3.853 | 1.855e-06 |
| einsum_plan_strided_read_c64_s256 | jax-cuda | 64 | 15 | 3.868 | 1.855e-06 |
| einsum_plan_into_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| sub_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 3.204 | 0 |
| sub_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 3.209 | 0 |
| div_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.724 | 4.875e-05 |
| div_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.740 | 0 |
| rem_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.721 | 0 |
| rem_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.739 | 0 |
| pow_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 3.602 | 0.000117 |
| pow_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 3.596 | 1.48e-07 |
| maximum_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 3.216 | 0 |
| maximum_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 3.210 | 0 |
| minimum_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 3.238 | 0 |
| minimum_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 3.237 | 0 |
| abs_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.588 | 0 |
| abs_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.589 | 0 |
| sign_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.603 | 0 |
| sign_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.590 | 0 |
| expm1_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.575 | 7.911e-05 |
| expm1_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.600 | 7.39e-08 |
| log_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.588 | 7.945e-05 |
| log_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.600 | 4.231e-08 |
| log1p_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.604 | 0.0001189 |
| log1p_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.597 | 1.108e-07 |
| erf_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 2.013 | 9.523e-05 |
| erf_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 2.019 | 8.536e-08 |
| sin_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.582 | 4.348e-05 |
| sin_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.595 | 4.153e-08 |
| cos_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.572 | 6.342e-05 |
| cos_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.609 | 5.905e-08 |
| sqrt_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.573 | 0.000119 |
| sqrt_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.585 | 0 |
| rsqrt_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 2.082 | 6.983e-05 |
| rsqrt_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 2.086 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.953 | 0.0004768 |
| reduce_prod_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.963 | 6.662e-07 |
| reduce_max_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.963 | 0 |
| reduce_max_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.957 | 0 |
| reduce_min_strided_read_f32_n1024 | jax-cuda | 64 | 15 | 2.969 | 0 |
| reduce_min_strided_read_f64_n1024 | jax-cuda | 64 | 15 | 2.951 | 0 |
| reduce_sum_squares_strided_read_f32_n1024 | jax-cuda | 4 | 15 | 3.348 | 0.0001522 |
| reduce_sum_squares_strided_read_f64_n1024 | jax-cuda | 4 | 15 | 3.349 | 1.418e-07 |
| sub_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 3.298 | 0 |
| sub_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.661 | 0 |
| div_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 2.821 | 4.875e-05 |
| div_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.412 | 0 |
| rem_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 2.773 | 0 |
| rem_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.431 | 0 |
| pow_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 3.713 | 0.0001577 |
| pow_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.808 | 1.48e-07 |
| maximum_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 3.282 | 0 |
| maximum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.667 | 0 |
| minimum_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 3.289 | 0 |
| minimum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.488 | 0 |
| abs_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 2.526 | 0 |
| abs_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.330 | 0 |
| sign_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 2.636 | 0 |
| sign_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.094 | 0 |
| expm1_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 2.645 | 7.915e-05 |
| expm1_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.340 | 7.39e-08 |
| log_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 2.662 | 7.946e-05 |
| log_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.330 | 4.308e-08 |
| log1p_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 2.636 | 0.000119 |
| log1p_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.332 | 1.11e-07 |
| erf_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 2.053 | 0.0001214 |
| erf_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.062 | 8.536e-08 |
| sin_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 2.644 | 4.348e-05 |
| sin_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.340 | 4.153e-08 |
| cos_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 2.639 | 6.342e-05 |
| cos_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.329 | 5.905e-08 |
| sqrt_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 2.644 | 0.0001192 |
| sqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 1.328 | 0 |
| rsqrt_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 2.107 | 6.983e-05 |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.114 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 3.056 | 0.007035 |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.040 | 4.995e-06 |
| reduce_max_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 3.054 | 0 |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.069 | 0 |
| reduce_min_strided_read_f32_n1048576 | jax-cuda | 64 | 15 | 3.044 | 0 |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | 64 | 15 | 3.059 | 0 |
| reduce_sum_squares_strided_read_f32_n1048576 | jax-cuda | 4 | 15 | 3.346 | 0.0001778 |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | 4 | 15 | 3.371 | 3.311e-07 |
| add_into_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c32_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c32_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1024 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_into_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| add_read_into_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_into_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| sub_read_into_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_into_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| mul_read_into_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_into_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| div_read_into_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_into_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| neg_read_into_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c32_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_into_c64_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c32_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| conj_read_into_c64_n1048576 | jax-cuda-jit | — | — | — | No equivalent public output-reuse API; allocation-returning operations and JAX donation are not substitutes. |
| tensordot_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.658 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.652 | 0 |
| tensordot_c32_n1024 | jax-cuda-jit | 64 | 15 | 3.395 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda-jit | 64 | 15 | 3.555 | 0 |
| vdot_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.635 | 0 |
| vdot_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.670 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda-jit | 64 | 15 | 2.748 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda-jit | 64 | 15 | 2.823 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.561 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.575 | 0 |
| tensordot_f32_n262144 | jax-cuda-jit | 64 | 15 | 3.191 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda-jit | 32 | 15 | 2.012 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda-jit | 32 | 15 | 2.157 | 0.01974 |
| tensordot_c64_n262144 | jax-cuda-jit | 32 | 15 | 2.417 | 6.383e-06 |
| vdot_read_f32_n262144 | jax-cuda-jit | 32 | 15 | 1.842 | 0 |
| vdot_read_f64_n262144 | jax-cuda-jit | 64 | 15 | 3.499 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda-jit | 64 | 15 | 3.415 | 0 |
| vdot_read_c64_n262144 | jax-cuda-jit | 64 | 15 | 3.125 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda-jit | 64 | 15 | 2.886 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda-jit | 64 | 15 | 2.864 | 1.664e-07 |
| axpby_read_into_accum_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_c32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_c64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_c32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| axpby_read_into_accum_c64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled Ref update for alpha=0.5/beta=1; public Ref.addupdate has no alpha parameter. |
| dot_general_read_into_f32_s16 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f32_s16 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f32_s16 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s16 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s16 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s16 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_c32_s16 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_c32_s16 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_c32_s16 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_c64_s16 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_c64_s16 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_c64_s16 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f32_s256 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f32_s256 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f32_s256 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_f64_s256 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_f64_s256 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_f64_s256 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_c32_s256 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_c32_s256 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_c32_s256 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_c64_s256 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_c64_s256 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| dot_general_read_into_accum_cached_c64_s256 | jax-cuda-jit | — | — | — | No matching public caller-owned output API; allocation-returning dot and donation are different contracts. |
| add_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.726 | 0 |
| add_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.735 | 0 |
| mul_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.737 | 0 |
| mul_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.703 | 0 |
| neg_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.544 | 0 |
| neg_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.553 | 0 |
| exp_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.568 | 0.0001272 |
| exp_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.560 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.574 | 9.325e-05 |
| tanh_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.575 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.581 | 0.0002267 |
| reduce_sum_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.568 | 3.268e-07 |
| conj_strided_read_c32_n1024 | jax-cuda-jit | 64 | 15 | 2.586 | 0 |
| conj_strided_read_c64_n1024 | jax-cuda-jit | 64 | 15 | 2.599 | 0 |
| add_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.847 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.422 | 0 |
| mul_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.856 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.420 | 0 |
| neg_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.681 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.341 | 0 |
| exp_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.676 | 0.0001272 |
| exp_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.346 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.703 | 9.508e-05 |
| tanh_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.361 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.701 | 0.003094 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.695 | 7.159e-06 |
| conj_strided_read_c32_n1048576 | jax-cuda-jit | 32 | 15 | 1.364 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda-jit | 16 | 15 | 0.679 | 0 |
| axpby_overwrite_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.606 | 0 |
| axpby_overwrite_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.658 | 0 |
| axpby_overwrite_c32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.652 | 0 |
| axpby_overwrite_c64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.662 | 0 |
| axpby_overwrite_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.662 | 0 |
| axpby_overwrite_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.660 | 0 |
| axpby_overwrite_c32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1048576 | jax-cuda-jit | 16 | 15 | 2.647 | 0 |
| axpby_overwrite_c64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1048576 | jax-cuda-jit | 15 | 15 | 2.490 | 0 |
| dot_general_batch_f32_s32 | jax-cuda-jit | 64 | 15 | 2.632 | 0.0002227 |
| dot_general_batch_f64_s32 | jax-cuda-jit | 64 | 15 | 2.648 | 1.326e-07 |
| dot_general_batch_c32_s32 | jax-cuda-jit | 64 | 15 | 3.851 | 0.0004354 |
| dot_general_batch_c64_s32 | jax-cuda-jit | 32 | 15 | 1.996 | 5.436e-07 |
| dot_general_multiple_f32_s32 | jax-cuda-jit | 64 | 15 | 3.098 | 0.000269 |
| dot_general_multiple_f64_s32 | jax-cuda-jit | 64 | 15 | 3.077 | 2.466e-07 |
| dot_general_multiple_c32_s32 | jax-cuda-jit | 32 | 15 | 1.979 | 0.0004771 |
| dot_general_multiple_c64_s32 | jax-cuda-jit | 32 | 15 | 2.052 | 4.129e-07 |
| dot_general_batch_f32_s256 | jax-cuda-jit | 64 | 15 | 2.742 | 0.0008561 |
| dot_general_batch_f64_s256 | jax-cuda-jit | 64 | 15 | 2.625 | 7.463e-07 |
| dot_general_batch_c32_s256 | jax-cuda-jit | 64 | 15 | 3.907 | 0.001359 |
| dot_general_batch_c64_s256 | jax-cuda-jit | 32 | 15 | 2.039 | 1.869e-06 |
| dot_general_multiple_f32_s256 | jax-cuda-jit | 64 | 15 | 3.148 | 0.0025 |
| dot_general_multiple_f64_s256 | jax-cuda-jit | 32 | 15 | 2.010 | 2.584e-06 |
| dot_general_multiple_c32_s256 | jax-cuda-jit | 32 | 15 | 1.987 | 0.001197 |
| dot_general_multiple_c64_s256 | jax-cuda-jit | 32 | 15 | 2.150 | 1.805e-06 |
| dot_conj_l0r0_c32_n1024 | jax-cuda-jit | 64 | 15 | 2.774 | 0.0002298 |
| dot_conj_l1r0_c32_n1024 | jax-cuda-jit | 64 | 15 | 2.763 | 0.0002522 |
| dot_conj_l0r1_c32_n1024 | jax-cuda-jit | 64 | 15 | 2.761 | 0.0002522 |
| dot_conj_l1r1_c32_n1024 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n1024 | jax-cuda-jit | 64 | 15 | 2.809 | 3.644e-07 |
| dot_conj_l1r0_c64_n1024 | jax-cuda-jit | 64 | 15 | 2.847 | 2.767e-07 |
| dot_conj_l0r1_c64_n1024 | jax-cuda-jit | 64 | 15 | 2.845 | 2.767e-07 |
| dot_conj_l1r1_c64_n1024 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c32_n262144 | jax-cuda-jit | 64 | 15 | 3.037 | 0.01276 |
| dot_conj_l1r0_c32_n262144 | jax-cuda-jit | 64 | 15 | 3.050 | 0.0002732 |
| dot_conj_l0r1_c32_n262144 | jax-cuda-jit | 64 | 15 | 3.033 | 0.0002732 |
| dot_conj_l1r1_c32_n262144 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda-jit | 64 | 15 | 3.104 | 1.221e-05 |
| dot_conj_l1r0_c64_n262144 | jax-cuda-jit | 64 | 15 | 3.129 | 1.586e-05 |
| dot_conj_l0r1_c64_n262144 | jax-cuda-jit | 64 | 15 | 3.097 | 1.586e-05 |
| dot_conj_l1r1_c64_n262144 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| einsum_plan_owned_f32_s32 | jax-cuda-jit | 64 | 15 | 2.630 | 0.0002018 |
| einsum_plan_read_f32_s32 | jax-cuda-jit | 64 | 15 | 2.634 | 0.0002018 |
| einsum_plan_strided_read_f32_s32 | jax-cuda-jit | 64 | 15 | 2.658 | 0.0002018 |
| einsum_plan_into_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f64_s32 | jax-cuda-jit | 64 | 15 | 2.656 | 0 |
| einsum_plan_read_f64_s32 | jax-cuda-jit | 64 | 15 | 2.638 | 0 |
| einsum_plan_strided_read_f64_s32 | jax-cuda-jit | 64 | 15 | 2.668 | 0 |
| einsum_plan_into_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c32_s32 | jax-cuda-jit | 64 | 15 | 3.415 | 0.0002532 |
| einsum_plan_read_c32_s32 | jax-cuda-jit | 64 | 15 | 3.422 | 0.0002532 |
| einsum_plan_strided_read_c32_s32 | jax-cuda-jit | 64 | 15 | 3.420 | 0.0002532 |
| einsum_plan_into_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c64_s32 | jax-cuda-jit | 64 | 15 | 3.547 | 1.981e-07 |
| einsum_plan_read_c64_s32 | jax-cuda-jit | 64 | 15 | 3.545 | 1.981e-07 |
| einsum_plan_strided_read_c64_s32 | jax-cuda-jit | 64 | 15 | 3.572 | 1.981e-07 |
| einsum_plan_into_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f32_s256 | jax-cuda-jit | 64 | 15 | 2.740 | 0.00118 |
| einsum_plan_read_f32_s256 | jax-cuda-jit | 64 | 15 | 2.758 | 0.00118 |
| einsum_plan_strided_read_f32_s256 | jax-cuda-jit | 64 | 15 | 2.748 | 0.00118 |
| einsum_plan_into_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f64_s256 | jax-cuda-jit | 64 | 15 | 2.741 | 0 |
| einsum_plan_read_f64_s256 | jax-cuda-jit | 64 | 15 | 2.740 | 0 |
| einsum_plan_strided_read_f64_s256 | jax-cuda-jit | 64 | 15 | 3.116 | 0 |
| einsum_plan_into_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c32_s256 | jax-cuda-jit | 64 | 15 | 3.928 | 0.002017 |
| einsum_plan_read_c32_s256 | jax-cuda-jit | 32 | 15 | 1.788 | 0.002017 |
| einsum_plan_strided_read_c32_s256 | jax-cuda-jit | 64 | 15 | 3.461 | 0.002017 |
| einsum_plan_into_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c64_s256 | jax-cuda-jit | 32 | 15 | 2.254 | 2.186e-06 |
| einsum_plan_read_c64_s256 | jax-cuda-jit | 64 | 15 | 3.828 | 2.186e-06 |
| einsum_plan_strided_read_c64_s256 | jax-cuda-jit | 32 | 15 | 2.009 | 2.186e-06 |
| einsum_plan_into_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| sub_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 3.014 | 0 |
| sub_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 3.351 | 0 |
| div_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 3.283 | 4.875e-05 |
| div_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 3.170 | 0 |
| rem_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.754 | 0 |
| rem_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.742 | 0 |
| pow_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.772 | 0.000117 |
| pow_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.756 | 1.48e-07 |
| maximum_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.762 | 0 |
| maximum_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.746 | 0 |
| minimum_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.758 | 0 |
| minimum_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.753 | 0 |
| abs_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.587 | 0 |
| abs_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.581 | 0 |
| sign_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.580 | 0 |
| sign_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.582 | 0 |
| expm1_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.574 | 7.911e-05 |
| expm1_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.561 | 7.39e-08 |
| log_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.567 | 7.945e-05 |
| log_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.567 | 4.231e-08 |
| log1p_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.565 | 0.0001189 |
| log1p_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.573 | 1.108e-07 |
| erf_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.578 | 9.523e-05 |
| erf_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.578 | 8.536e-08 |
| sin_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.578 | 4.348e-05 |
| sin_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.601 | 4.153e-08 |
| cos_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.572 | 6.342e-05 |
| cos_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.598 | 5.905e-08 |
| sqrt_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.561 | 0.000119 |
| sqrt_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.568 | 0 |
| rsqrt_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.575 | 6.983e-05 |
| rsqrt_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.571 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.575 | 0.0004768 |
| reduce_prod_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.494 | 4.441e-07 |
| reduce_max_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.582 | 0 |
| reduce_max_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.582 | 0 |
| reduce_min_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.568 | 0 |
| reduce_min_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.564 | 0 |
| reduce_sum_squares_strided_read_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.560 | 0.0001522 |
| reduce_sum_squares_strided_read_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.560 | 1.418e-07 |
| sub_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.827 | 0 |
| sub_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.421 | 0 |
| div_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.853 | 4.875e-05 |
| div_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.423 | 0 |
| rem_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.806 | 0 |
| rem_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.417 | 0 |
| pow_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.846 | 0.0001577 |
| pow_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.365 | 1.48e-07 |
| maximum_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.835 | 0 |
| maximum_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.420 | 0 |
| minimum_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.807 | 0 |
| minimum_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.426 | 0 |
| abs_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.693 | 0 |
| abs_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.343 | 0 |
| sign_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.691 | 0 |
| sign_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.344 | 0 |
| expm1_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.692 | 7.915e-05 |
| expm1_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.346 | 7.39e-08 |
| log_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.701 | 7.946e-05 |
| log_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.346 | 4.308e-08 |
| log1p_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.667 | 0.000119 |
| log1p_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.338 | 1.11e-07 |
| erf_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.647 | 0.0001214 |
| erf_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.351 | 8.536e-08 |
| sin_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.664 | 4.348e-05 |
| sin_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.348 | 4.153e-08 |
| cos_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.677 | 6.342e-05 |
| cos_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.341 | 5.905e-08 |
| sqrt_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.679 | 0.0001192 |
| sqrt_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.344 | 0 |
| rsqrt_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.681 | 6.983e-05 |
| rsqrt_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.340 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.598 | 0.006678 |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.687 | 4.995e-06 |
| reduce_max_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.680 | 0 |
| reduce_max_strided_read_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.719 | 0 |
| reduce_min_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.726 | 0 |
| reduce_min_strided_read_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.706 | 0 |
| reduce_sum_squares_strided_read_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.684 | 0.0001778 |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda-jit | 64 | 15 | 2.716 | 3.311e-07 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: benchmarks/gpu/reuse_contraction.yaml
timestamp: '2026-11-02T21:14:08Z'
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
    ./scripts/run_gpu_public_api.sh reuse_contraction
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 328
  selected_cases: 328
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
    data/results/nvidia-gpu/gpu/reuse_contraction/20261102_211407/fixtures/manifest.json: 0dee7d09b26142623e40e8d1517ebf92104039edf94ef7a83c73cea2da9830ae
    target/release/benchmark_gpu_public_api: c0a0be40dd246cd4cb7ca59cac19bacd1549a4f3a593aea91e887cf41f6e4aa6
  reference_numpy_version: 2.4.2
```
