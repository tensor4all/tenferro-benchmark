# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **424**. Unsupported cells: **336**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261013_183707/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

Unit-coefficient AXPBY uses native JAX Ref.addupdate with caller-owned fixed buffers. Its synchronization includes one GPU scalar completion token per destination and block_until_ready; tokens are retained through clock stop and included in the logical retention bound. Helper compilation and seed restoration are untimed. Full Ref snapshots are used only for untimed inspection. This scalar-token synchronization cost is included in the reported latency; it performs no full output copy or D2H transfer inside timing.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | 8.074 (0.281) | unsupported | 7.890 (0.429) | unsupported | unsupported |
| add_into_f64_n1024 | 5.437 (0.121) | unsupported | 7.000 (0.119) | unsupported | unsupported |
| add_read_into_f32_n1024 | 5.513 (0.138) | unsupported | 6.825 (0.047) | unsupported | unsupported |
| add_read_into_f64_n1024 | 5.707 (0.285) | unsupported | 7.417 (0.394) | unsupported | unsupported |
| sub_into_f32_n1024 | 5.827 (0.329) | unsupported | 7.025 (0.037) | unsupported | unsupported |
| sub_into_f64_n1024 | 5.546 (0.079) | unsupported | 7.212 (0.151) | unsupported | unsupported |
| sub_read_into_f32_n1024 | 5.595 (0.161) | unsupported | 7.445 (0.264) | unsupported | unsupported |
| sub_read_into_f64_n1024 | 5.613 (0.194) | unsupported | 7.153 (0.061) | unsupported | unsupported |
| mul_into_f32_n1024 | 5.535 (0.376) | unsupported | 7.226 (0.157) | unsupported | unsupported |
| mul_into_f64_n1024 | 5.679 (0.251) | unsupported | 7.152 (0.128) | unsupported | unsupported |
| mul_read_into_f32_n1024 | 5.723 (0.195) | unsupported | 7.064 (0.171) | unsupported | unsupported |
| mul_read_into_f64_n1024 | 5.699 (0.065) | unsupported | 7.331 (0.395) | unsupported | unsupported |
| div_into_f32_n1024 | 45.698 (2.651) | unsupported | 7.433 (0.254) | unsupported | unsupported |
| div_into_f64_n1024 | 44.671 (0.647) | unsupported | 7.034 (0.227) | unsupported | unsupported |
| div_read_into_f32_n1024 | 42.129 (1.796) | unsupported | 7.009 (0.107) | unsupported | unsupported |
| div_read_into_f64_n1024 | 44.600 (0.611) | unsupported | 6.961 (0.352) | unsupported | unsupported |
| neg_into_f32_n1024 | 5.113 (0.477) | unsupported | 6.802 (0.141) | unsupported | unsupported |
| neg_into_f64_n1024 | 4.920 (0.107) | unsupported | 6.586 (0.276) | unsupported | unsupported |
| neg_read_into_f32_n1024 | 4.863 (0.105) | unsupported | 6.768 (0.132) | unsupported | unsupported |
| neg_read_into_f64_n1024 | 5.081 (0.240) | unsupported | 6.702 (0.031) | unsupported | unsupported |
| conj_into_c32_n1024 | 39.708 (1.380) | unsupported | 7.179 (0.062) | unsupported | unsupported |
| conj_into_c64_n1024 | 184.272 (5.425) | unsupported | 7.233 (0.112) | unsupported | unsupported |
| conj_read_into_c32_n1024 | 39.045 (0.392) | unsupported | 7.363 (0.228) | unsupported | unsupported |
| conj_read_into_c64_n1024 | 173.904 (17.568) | unsupported | 7.162 (0.043) | unsupported | unsupported |
| add_into_f32_n1048576 | 43.189 (0.178) | unsupported | 39.779 (0.025) | unsupported | unsupported |
| add_into_f64_n1048576 | 83.285 (0.696) | unsupported | 77.741 (0.075) | unsupported | unsupported |
| add_read_into_f32_n1048576 | 42.815 (0.263) | unsupported | 39.753 (0.081) | unsupported | unsupported |
| add_read_into_f64_n1048576 | 82.508 (0.522) | unsupported | 77.630 (0.076) | unsupported | unsupported |
| sub_into_f32_n1048576 | 42.783 (0.245) | unsupported | 39.695 (0.030) | unsupported | unsupported |
| sub_into_f64_n1048576 | 82.314 (0.589) | unsupported | 77.628 (0.042) | unsupported | unsupported |
| sub_read_into_f32_n1048576 | 42.870 (0.161) | unsupported | 39.685 (0.073) | unsupported | unsupported |
| sub_read_into_f64_n1048576 | 82.549 (0.429) | unsupported | 77.640 (0.146) | unsupported | unsupported |
| mul_into_f32_n1048576 | 42.670 (0.196) | unsupported | 39.699 (0.069) | unsupported | unsupported |
| mul_into_f64_n1048576 | 82.579 (0.540) | unsupported | 77.564 (0.071) | unsupported | unsupported |
| mul_read_into_f32_n1048576 | 42.885 (0.248) | unsupported | 39.641 (0.066) | unsupported | unsupported |
| mul_read_into_f64_n1048576 | 82.841 (0.663) | unsupported | 77.547 (0.048) | unsupported | unsupported |
| div_into_f32_n1048576 | 70.140 (0.137) | unsupported | 39.989 (0.083) | unsupported | unsupported |
| div_into_f64_n1048576 | 151.519 (0.152) | unsupported | 94.320 (0.138) | unsupported | unsupported |
| div_read_into_f32_n1048576 | 70.095 (0.110) | unsupported | 39.921 (0.073) | unsupported | unsupported |
| div_read_into_f64_n1048576 | 151.289 (0.091) | unsupported | 94.379 (0.134) | unsupported | unsupported |
| neg_into_f32_n1048576 | 29.582 (0.144) | unsupported | 27.266 (0.119) | unsupported | unsupported |
| neg_into_f64_n1048576 | 57.139 (0.347) | unsupported | 52.865 (0.040) | unsupported | unsupported |
| neg_read_into_f32_n1048576 | 29.767 (0.262) | unsupported | 27.225 (0.069) | unsupported | unsupported |
| neg_read_into_f64_n1048576 | 57.187 (0.570) | unsupported | 52.881 (0.081) | unsupported | unsupported |
| conj_into_c32_n1048576 | 107.865 (0.066) | unsupported | 52.788 (0.036) | unsupported | unsupported |
| conj_into_c64_n1048576 | 208.900 (0.296) | unsupported | 103.691 (0.183) | unsupported | unsupported |
| conj_read_into_c32_n1048576 | 107.658 (0.246) | unsupported | 52.986 (0.308) | unsupported | unsupported |
| conj_read_into_c64_n1048576 | 208.986 (0.369) | unsupported | 103.709 (0.076) | unsupported | unsupported |
| tensordot_f32_n1024 | 29.659 (0.975) | 42.568 (2.343) | 53.072 (1.121) | 191.654 (6.033) | 72.664 (1.579) |
| tensordot_f64_n1024 | 28.408 (1.504) | 44.968 (1.134) | 53.802 (0.902) | 163.211 (3.060) | 70.319 (2.019) |
| tensordot_c32_n1024 | 29.764 (1.413) | 40.875 (1.319) | 52.491 (1.344) | 174.915 (2.784) | 81.251 (1.057) |
| tensordot_c64_n1024 | 123.985 (13.004) | 39.174 (1.974) | 55.240 (1.968) | 186.030 (7.057) | 87.937 (3.171) |
| vdot_read_f32_n1024 | 157.885 (9.381) | unsupported | 15.509 (0.115) | 107.581 (6.019) | 108.599 (13.270) |
| vdot_read_f64_n1024 | 169.307 (4.567) | unsupported | 14.946 (0.127) | 105.179 (14.595) | 88.383 (7.943) |
| vdot_read_c32_n1024 | 165.619 (3.050) | unsupported | 14.891 (0.240) | 96.728 (5.253) | 147.542 (18.083) |
| vdot_read_c64_n1024 | 196.381 (7.749) | unsupported | 14.841 (0.293) | 117.069 (7.589) | 159.050 (28.222) |
| norm_squared_read_f32_n1024 | 181.439 (12.086) | unsupported | 15.351 (0.219) | 87.835 (11.201) | 159.471 (8.560) |
| norm_squared_read_f64_n1024 | 167.888 (17.758) | unsupported | 15.248 (0.450) | 74.211 (2.495) | 161.298 (5.463) |
| tensordot_f32_n262144 | 66.954 (5.470) | 47.113 (1.123) | 62.853 (1.687) | 187.476 (19.773) | 133.380 (30.823) |
| tensordot_f64_n262144 | 1613.992 (4.925) | 1629.722 (12.018) | 1633.117 (1.367) | 1689.483 (17.486) | 1611.153 (19.140) |
| tensordot_c32_n262144 | 140.629 (0.271) | 142.173 (0.431) | 161.356 (2.905) | 343.869 (22.771) | 159.176 (1.675) |
| tensordot_c64_n262144 | 5019.582 (4.133) | 5057.571 (7.021) | 5284.572 (6.388) | 5932.029 (32.399) | 5385.454 (18.288) |
| vdot_read_f32_n262144 | 167.937 (4.687) | unsupported | 15.567 (0.558) | 193.807 (5.290) | 166.097 (18.368) |
| vdot_read_f64_n262144 | 187.637 (6.568) | unsupported | 20.298 (0.061) | 184.882 (12.845) | 137.119 (4.884) |
| vdot_read_c32_n262144 | 191.763 (4.082) | unsupported | 17.350 (0.052) | 145.912 (33.682) | 169.107 (22.243) |
| vdot_read_c64_n262144 | 191.774 (3.122) | unsupported | 33.842 (0.026) | 191.923 (19.519) | 177.928 (22.194) |
| norm_squared_read_f32_n262144 | 192.416 (10.007) | unsupported | 15.648 (0.271) | 154.646 (32.667) | 167.599 (9.734) |
| norm_squared_read_f64_n262144 | 191.305 (5.521) | unsupported | 15.500 (0.844) | 121.376 (16.707) | 140.304 (11.727) |
| axpby_read_into_accum_f32_n1024 | 10.246 (0.604) | unsupported | 8.718 (0.257) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1024 | 10.219 (0.156) | unsupported | 8.770 (0.057) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1024 | 9.630 (0.235) | unsupported | 8.868 (0.136) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1024 | 9.666 (0.320) | unsupported | 8.801 (0.299) | unsupported | unsupported |
| axpby_read_into_accum_f32_n1048576 | 273.182 (0.248) | unsupported | 39.749 (0.037) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1048576 | 534.261 (0.463) | unsupported | 77.715 (0.042) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1048576 | 168.812 (0.235) | unsupported | 77.665 (0.040) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1048576 | 2017.124 (0.443) | unsupported | 153.549 (0.188) | unsupported | unsupported |
| dot_general_read_into_f32_s16 | 23.052 (0.363) | unsupported | 20.637 (0.402) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s16 | 22.822 (0.663) | unsupported | 24.097 (0.354) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s16 | 22.962 (0.668) | unsupported | 24.238 (0.194) | unsupported | unsupported |
| dot_general_read_into_f64_s16 | 23.178 (0.547) | unsupported | 21.014 (0.529) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s16 | 24.143 (1.114) | unsupported | 25.520 (1.079) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s16 | 24.389 (0.802) | unsupported | 25.094 (0.421) | unsupported | unsupported |
| dot_general_read_into_c32_s16 | 23.809 (1.311) | unsupported | 20.607 (0.293) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s16 | 23.995 (0.605) | unsupported | 24.061 (0.418) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s16 | 23.632 (1.317) | unsupported | 23.706 (0.200) | unsupported | unsupported |
| dot_general_read_into_c64_s16 | 23.582 (1.032) | unsupported | 22.247 (0.166) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s16 | 24.713 (0.979) | unsupported | 27.331 (1.063) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s16 | 24.233 (0.778) | unsupported | 27.581 (2.157) | unsupported | unsupported |
| dot_general_read_into_f32_s256 | 22.998 (0.553) | unsupported | 20.779 (0.339) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s256 | 22.372 (0.559) | unsupported | 25.075 (1.839) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s256 | 24.728 (1.056) | unsupported | 24.744 (0.114) | unsupported | unsupported |
| dot_general_read_into_f64_s256 | 262.946 (0.120) | unsupported | 260.441 (15.499) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s256 | 275.138 (10.679) | unsupported | 262.010 (10.124) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s256 | 318.211 (11.115) | unsupported | 261.966 (5.115) | unsupported | unsupported |
| dot_general_read_into_c32_s256 | 28.497 (0.535) | unsupported | 26.975 (0.049) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s256 | 30.183 (0.315) | unsupported | 27.781 (0.053) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s256 | 29.831 (0.262) | unsupported | 27.799 (0.415) | unsupported | unsupported |
| dot_general_read_into_c64_s256 | 683.097 (0.577) | unsupported | 974.654 (0.246) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s256 | 684.722 (0.894) | unsupported | 979.943 (0.653) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s256 | 684.194 (0.218) | unsupported | 980.150 (0.411) | unsupported | unsupported |
| add_strided_read_f32_n1024 | 307.486 (4.341) | unsupported | 10.239 (0.101) | 154.098 (30.464) | 152.444 (23.569) |
| add_strided_read_f64_n1024 | 266.102 (21.509) | unsupported | 10.627 (0.614) | 175.331 (27.599) | 165.510 (23.411) |
| mul_strided_read_f32_n1024 | 352.785 (12.525) | unsupported | 10.444 (0.730) | 158.866 (40.227) | 148.639 (30.863) |
| mul_strided_read_f64_n1024 | 324.147 (3.975) | unsupported | 10.177 (0.291) | 142.607 (26.300) | 131.312 (19.495) |
| neg_strided_read_f32_n1024 | 204.173 (5.003) | unsupported | 9.548 (0.159) | 128.150 (9.163) | 122.665 (15.199) |
| neg_strided_read_f64_n1024 | 178.761 (2.204) | unsupported | 9.984 (0.739) | 103.105 (13.348) | 119.007 (10.368) |
| exp_strided_read_f32_n1024 | 205.900 (2.654) | unsupported | 8.994 (0.120) | 94.580 (9.891) | 101.760 (18.824) |
| exp_strided_read_f64_n1024 | 189.453 (7.893) | unsupported | 9.145 (0.100) | 94.174 (11.777) | 101.082 (13.483) |
| tanh_strided_read_f32_n1024 | 206.454 (5.526) | unsupported | 9.314 (0.110) | 92.719 (8.972) | 90.837 (5.887) |
| tanh_strided_read_f64_n1024 | 188.734 (4.382) | unsupported | 9.232 (0.046) | 96.498 (12.737) | 96.799 (10.267) |
| reduce_sum_strided_read_f32_n1024 | 396.092 (48.446) | unsupported | 15.709 (0.232) | 93.620 (1.821) | 145.319 (8.141) |
| reduce_sum_strided_read_f64_n1024 | 318.136 (12.144) | unsupported | 14.614 (0.157) | 94.632 (3.432) | 123.884 (12.132) |
| conj_strided_read_c32_n1024 | 167.937 (4.378) | unsupported | 11.581 (0.301) | 134.838 (26.390) | 145.990 (30.309) |
| conj_strided_read_c64_n1024 | 250.270 (5.338) | unsupported | 11.801 (0.038) | 151.816 (28.683) | 158.263 (32.939) |
| add_strided_read_f32_n1048576 | 98.071 (0.441) | unsupported | 39.784 (0.040) | 130.103 (11.216) | 120.325 (10.836) |
| add_strided_read_f64_n1048576 | 188.163 (0.243) | unsupported | 77.820 (0.117) | 104.908 (1.613) | 101.789 (5.345) |
| mul_strided_read_f32_n1048576 | 97.428 (0.186) | unsupported | 39.803 (0.039) | 101.276 (1.777) | 97.242 (2.008) |
| mul_strided_read_f64_n1048576 | 188.010 (0.248) | unsupported | 77.873 (0.177) | 97.486 (0.935) | 89.971 (1.756) |
| neg_strided_read_f32_n1048576 | 56.033 (0.116) | unsupported | 27.431 (0.148) | 98.060 (0.651) | 94.863 (0.642) |
| neg_strided_read_f64_n1048576 | 108.116 (0.210) | unsupported | 53.094 (0.190) | 101.847 (0.949) | 94.831 (0.501) |
| exp_strided_read_f32_n1048576 | 56.225 (0.304) | unsupported | 27.408 (0.027) | 86.111 (0.280) | 93.423 (0.726) |
| exp_strided_read_f64_n1048576 | 233.417 (0.216) | unsupported | 180.945 (0.088) | 205.425 (1.832) | 204.172 (1.719) |
| tanh_strided_read_f32_n1048576 | 56.196 (0.109) | unsupported | 27.448 (0.021) | 90.297 (3.291) | 93.732 (3.078) |
| tanh_strided_read_f64_n1048576 | 220.366 (0.429) | unsupported | 167.753 (0.134) | 191.217 (2.817) | 190.668 (1.771) |
| reduce_sum_strided_read_f32_n1048576 | 186.415 (5.101) | unsupported | 24.280 (0.382) | 158.241 (2.552) | 91.024 (0.307) |
| reduce_sum_strided_read_f64_n1048576 | 92.842 (0.413) | unsupported | 36.975 (0.058) | 156.006 (3.171) | 150.509 (3.199) |
| conj_strided_read_c32_n1048576 | 114.228 (0.145) | unsupported | 53.062 (0.026) | 92.811 (7.435) | 98.489 (3.240) |
| conj_strided_read_c64_n1048576 | 214.076 (0.237) | unsupported | 104.194 (0.050) | 116.485 (0.718) | 118.168 (0.774) |
| axpby_overwrite_f32_n1024 | 10.238 (0.160) | unsupported | 11.958 (0.819) | unsupported | unsupported |
| axpby_lerp_f32_n1024 | 10.304 (0.136) | unsupported | 9.715 (0.278) | unsupported | unsupported |
| axpby_unit_add_f32_n1024 | 10.505 (0.178) | unsupported | 8.888 (0.601) | 339.948 (44.152) | 253.032 (55.935) |
| axpby_overwrite_f64_n1024 | 10.576 (0.627) | unsupported | 11.528 (0.191) | unsupported | unsupported |
| axpby_lerp_f64_n1024 | 10.312 (0.355) | unsupported | 9.515 (0.115) | unsupported | unsupported |
| axpby_unit_add_f64_n1024 | 10.505 (0.276) | unsupported | 8.827 (0.063) | 342.810 (28.258) | 247.476 (45.962) |
| axpby_overwrite_c32_n1024 | 10.049 (0.421) | unsupported | 11.966 (0.144) | unsupported | unsupported |
| axpby_lerp_c32_n1024 | 10.269 (0.561) | unsupported | 8.076 (0.127) | unsupported | unsupported |
| axpby_unit_add_c32_n1024 | 9.969 (0.201) | unsupported | 8.717 (0.080) | 351.895 (53.808) | 242.127 (33.911) |
| axpby_overwrite_c64_n1024 | 9.871 (0.169) | unsupported | 11.937 (0.084) | unsupported | unsupported |
| axpby_lerp_c64_n1024 | 9.886 (0.311) | unsupported | 10.877 (0.004) | unsupported | unsupported |
| axpby_unit_add_c64_n1024 | 9.850 (0.284) | unsupported | 9.179 (0.484) | 345.915 (41.513) | 258.514 (49.079) |
| axpby_overwrite_f32_n1048576 | 304.750 (0.526) | unsupported | 27.495 (0.091) | unsupported | unsupported |
| axpby_lerp_f32_n1048576 | 273.441 (0.173) | unsupported | 39.776 (0.040) | unsupported | unsupported |
| axpby_unit_add_f32_n1048576 | 273.241 (0.233) | unsupported | 39.712 (0.029) | 354.629 (13.707) | 242.868 (15.477) |
| axpby_overwrite_f64_n1048576 | 339.425 (0.772) | unsupported | 53.202 (0.087) | unsupported | unsupported |
| axpby_lerp_f64_n1048576 | 534.138 (0.182) | unsupported | 78.053 (0.074) | unsupported | unsupported |
| axpby_unit_add_f64_n1048576 | 535.122 (1.413) | unsupported | 77.715 (0.045) | 359.485 (13.386) | 248.096 (10.665) |
| axpby_overwrite_c32_n1048576 | 156.292 (0.616) | unsupported | 53.151 (0.066) | unsupported | unsupported |
| axpby_lerp_c32_n1048576 | 168.985 (0.152) | unsupported | 77.946 (0.054) | unsupported | unsupported |
| axpby_unit_add_c32_n1048576 | 168.898 (0.099) | unsupported | 77.675 (0.054) | 381.447 (34.216) | 249.836 (24.622) |
| axpby_overwrite_c64_n1048576 | 1351.534 (1.185) | unsupported | 104.431 (0.114) | unsupported | unsupported |
| axpby_lerp_c64_n1048576 | 2145.159 (1.067) | unsupported | 341.844 (0.148) | unsupported | unsupported |
| axpby_unit_add_c64_n1048576 | 2144.992 (1.024) | unsupported | 153.537 (0.166) | 345.084 (23.354) | 257.063 (19.208) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
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
| tensordot_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.325 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.250 | 0 |
| tensordot_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.600 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.814 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.475 | 0 |
| vdot_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.828 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.361 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.545 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.552 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.581 | 0 |
| tensordot_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.134 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda-jit | 2 | 15 | 3.222 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.547 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda-jit | 1 | 15 | 5.385 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.658 | 0 |
| vdot_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.194 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.706 | 0 |
| vdot_read_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.847 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.682 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.245 | 1.664e-07 |
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
| add_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.439 | 0 |
| add_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.648 | 0 |
| mul_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.378 | 0 |
| mul_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.101 | 0 |
| neg_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.963 | 0 |
| neg_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.904 | 0 |
| exp_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.256 | 0.0001272 |
| exp_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.235 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.907 | 9.325e-05 |
| tanh_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.098 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.325 | 0.0002267 |
| reduce_sum_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.982 | 4.202e-07 |
| conj_strided_read_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.336 | 0 |
| conj_strided_read_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.532 | 0 |
| add_strided_read_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.925 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.629 | 0 |
| mul_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.112 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.879 | 0 |
| neg_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.036 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.035 | 0 |
| exp_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.990 | 0.0001272 |
| exp_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.267 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.999 | 9.508e-05 |
| tanh_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.051 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.913 | 0.002078 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.408 | 7.159e-06 |
| conj_strided_read_c32_n1048576 | jax-cuda-jit | 32 | 15 | 3.152 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.891 | 0 |
| axpby_overwrite_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1024 | jax-cuda-jit | 8 | 15 | 2.024 | 0 |
| axpby_overwrite_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1024 | jax-cuda-jit | 8 | 15 | 1.980 | 0 |
| axpby_overwrite_c32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1024 | jax-cuda-jit | 8 | 15 | 1.937 | 0 |
| axpby_overwrite_c64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1024 | jax-cuda-jit | 8 | 15 | 2.068 | 0 |
| axpby_overwrite_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1048576 | jax-cuda-jit | 8 | 15 | 1.943 | 0 |
| axpby_overwrite_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1048576 | jax-cuda-jit | 8 | 15 | 1.985 | 0 |
| axpby_overwrite_c32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1048576 | jax-cuda-jit | 8 | 15 | 1.999 | 0 |
| axpby_overwrite_c64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1048576 | jax-cuda-jit | 8 | 15 | 2.057 | 0 |
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
| tensordot_f32_n1024 | jax-cuda | 16 | 15 | 3.066 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda | 16 | 15 | 2.611 | 0 |
| tensordot_c32_n1024 | jax-cuda | 16 | 15 | 2.799 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda | 16 | 15 | 2.976 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda | 32 | 15 | 3.443 | 0 |
| vdot_read_f64_n1024 | jax-cuda | 32 | 15 | 3.366 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda | 32 | 15 | 3.095 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda | 32 | 15 | 3.746 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda | 32 | 15 | 2.811 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda | 32 | 15 | 2.375 | 0 |
| tensordot_f32_n262144 | jax-cuda | 16 | 15 | 3.000 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda | 2 | 15 | 3.379 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda | 8 | 15 | 2.751 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda | 1 | 15 | 5.932 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda | 16 | 15 | 3.101 | 0 |
| vdot_read_f64_n262144 | jax-cuda | 16 | 15 | 2.958 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda | 16 | 15 | 2.335 | 0 |
| vdot_read_c64_n262144 | jax-cuda | 16 | 15 | 3.071 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda | 16 | 15 | 2.474 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda | 16 | 15 | 1.942 | 1.664e-07 |
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
| add_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.466 | 0 |
| add_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.805 | 0 |
| mul_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.542 | 0 |
| mul_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.282 | 0 |
| neg_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.050 | 0 |
| neg_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 3.299 | 0 |
| exp_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 3.027 | 0.0001272 |
| exp_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 3.014 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 2.967 | 9.325e-05 |
| tanh_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 3.088 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 2.996 | 0.0002267 |
| reduce_sum_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 3.028 | 4.202e-07 |
| conj_strided_read_c32_n1024 | jax-cuda | 16 | 15 | 2.157 | 0 |
| conj_strided_read_c64_n1024 | jax-cuda | 16 | 15 | 2.429 | 0 |
| add_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.082 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 1.679 | 0 |
| mul_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.241 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.120 | 0 |
| neg_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.138 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.259 | 0 |
| exp_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 2.756 | 0.0001272 |
| exp_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.287 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 2.890 | 9.508e-05 |
| tanh_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.059 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.532 | 0.003094 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.496 | 7.159e-06 |
| conj_strided_read_c32_n1048576 | jax-cuda | 32 | 15 | 2.970 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 1.864 | 0 |
| axpby_overwrite_f32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1024 | jax-cuda | 8 | 15 | 2.720 | 0 |
| axpby_overwrite_f64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1024 | jax-cuda | 8 | 15 | 2.742 | 0 |
| axpby_overwrite_c32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1024 | jax-cuda | 8 | 15 | 2.815 | 0 |
| axpby_overwrite_c64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1024 | jax-cuda | 8 | 15 | 2.767 | 0 |
| axpby_overwrite_f32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1048576 | jax-cuda | 8 | 15 | 2.837 | 0 |
| axpby_overwrite_f64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1048576 | jax-cuda | 8 | 15 | 2.876 | 0 |
| axpby_overwrite_c32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1048576 | jax-cuda | 8 | 15 | 3.052 | 0 |
| axpby_overwrite_c64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1048576 | jax-cuda | 8 | 15 | 2.761 | 0 |
| add_into_f32_n1024 | pytorch-cuda | 256 | 15 | 2.020 | 0 |
| add_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.584 | 0 |
| add_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.495 | 0 |
| add_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.798 | 0 |
| sub_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.597 | 0 |
| sub_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.692 | 0 |
| sub_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.812 | 0 |
| sub_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.663 | 0 |
| mul_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.700 | 0 |
| mul_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.662 | 0 |
| mul_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.617 | 0 |
| mul_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.754 | 0 |
| div_into_f32_n1024 | pytorch-cuda | 256 | 15 | 1.903 | 0 |
| div_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.601 | 0 |
| div_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.589 | 0 |
| div_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.564 | 0 |
| neg_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.483 | 0 |
| neg_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.372 | 0 |
| neg_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.465 | 0 |
| neg_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.431 | 0 |
| conj_into_c32_n1024 | pytorch-cuda | 512 | 15 | 3.676 | 0 |
| conj_into_c64_n1024 | pytorch-cuda | 512 | 15 | 3.703 | 0 |
| conj_read_into_c32_n1024 | pytorch-cuda | 512 | 15 | 3.770 | 0 |
| conj_read_into_c64_n1024 | pytorch-cuda | 512 | 15 | 3.667 | 0 |
| add_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.546 | 0 |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| add_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| sub_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.540 | 0 |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| sub_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.540 | 0 |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| mul_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.541 | 0 |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.482 | 0 |
| mul_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.537 | 0 |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.482 | 0 |
| div_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.559 | 0 |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.018 | 0 |
| div_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.555 | 0 |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.020 | 0 |
| neg_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.745 | 0 |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.692 | 0 |
| neg_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.742 | 0 |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.692 | 0 |
| conj_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.689 | 0 |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.659 | 0 |
| conj_read_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.696 | 0 |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.659 | 0 |
| tensordot_f32_n1024 | pytorch-cuda | 64 | 15 | 3.397 | 0.0008859 |
| tensordot_f64_n1024 | pytorch-cuda | 64 | 15 | 3.443 | 0 |
| tensordot_c32_n1024 | pytorch-cuda | 64 | 15 | 3.359 | 0.0005344 |
| tensordot_c64_n1024 | pytorch-cuda | 64 | 15 | 3.535 | 0 |
| vdot_read_f32_n1024 | pytorch-cuda | 128 | 15 | 1.985 | 0 |
| vdot_read_f64_n1024 | pytorch-cuda | 128 | 15 | 1.913 | 1.722e-07 |
| vdot_read_c32_n1024 | pytorch-cuda | 256 | 15 | 3.812 | 0 |
| vdot_read_c64_n1024 | pytorch-cuda | 256 | 15 | 3.799 | 3.455e-07 |
| norm_squared_read_f32_n1024 | pytorch-cuda | 256 | 15 | 3.930 | 0 |
| norm_squared_read_f64_n1024 | pytorch-cuda | 256 | 15 | 3.904 | 0 |
| tensordot_f32_n262144 | pytorch-cuda | 32 | 15 | 2.011 | 0.006191 |
| tensordot_f64_n262144 | pytorch-cuda | 2 | 15 | 3.266 | 6.714e-06 |
| tensordot_c32_n262144 | pytorch-cuda | 16 | 15 | 2.582 | 0.005717 |
| tensordot_c64_n262144 | pytorch-cuda | 1 | 15 | 5.285 | 4.401e-06 |
| vdot_read_f32_n262144 | pytorch-cuda | 128 | 15 | 1.993 | 0 |
| vdot_read_f64_n262144 | pytorch-cuda | 128 | 15 | 2.598 | 2.995e-06 |
| vdot_read_c32_n262144 | pytorch-cuda | 128 | 15 | 2.221 | 0 |
| vdot_read_c64_n262144 | pytorch-cuda | 64 | 15 | 2.166 | 8.762e-06 |
| norm_squared_read_f32_n262144 | pytorch-cuda | 128 | 15 | 2.003 | 0.0001786 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 128 | 15 | 1.984 | 0 |
| axpby_read_into_accum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.232 | 0 |
| axpby_read_into_accum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.245 | 0 |
| axpby_read_into_accum_c32_n1024 | pytorch-cuda | 256 | 15 | 2.270 | 0 |
| axpby_read_into_accum_c64_n1024 | pytorch-cuda | 256 | 15 | 2.253 | 0 |
| axpby_read_into_accum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.487 | 0 |
| axpby_read_into_accum_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.485 | 0 |
| axpby_read_into_accum_c64_n1048576 | pytorch-cuda | 16 | 15 | 2.457 | 0 |
| dot_general_read_into_f32_s16 | pytorch-cuda | 128 | 15 | 2.642 | 8.645e-05 |
| dot_general_read_into_accum_f32_s16 | pytorch-cuda | 128 | 15 | 3.084 | 7.907e-05 |
| dot_general_read_into_accum_cached_f32_s16 | pytorch-cuda | 128 | 15 | 3.103 | 7.907e-05 |
| dot_general_read_into_f64_s16 | pytorch-cuda | 128 | 15 | 2.690 | 0 |
| dot_general_read_into_accum_f64_s16 | pytorch-cuda | 128 | 15 | 3.267 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | pytorch-cuda | 128 | 15 | 3.212 | 0 |
| dot_general_read_into_c32_s16 | pytorch-cuda | 128 | 15 | 2.638 | 0.0001647 |
| dot_general_read_into_accum_c32_s16 | pytorch-cuda | 128 | 15 | 3.080 | 0.0001125 |
| dot_general_read_into_accum_cached_c32_s16 | pytorch-cuda | 128 | 15 | 3.034 | 0.0001125 |
| dot_general_read_into_c64_s16 | pytorch-cuda | 128 | 15 | 2.848 | 1.373e-07 |
| dot_general_read_into_accum_c64_s16 | pytorch-cuda | 128 | 15 | 3.498 | 7.37e-08 |
| dot_general_read_into_accum_cached_c64_s16 | pytorch-cuda | 64 | 15 | 1.765 | 7.37e-08 |
| dot_general_read_into_f32_s256 | pytorch-cuda | 128 | 15 | 2.660 | 0.0005542 |
| dot_general_read_into_accum_f32_s256 | pytorch-cuda | 128 | 15 | 3.210 | 0.0003617 |
| dot_general_read_into_accum_cached_f32_s256 | pytorch-cuda | 128 | 15 | 3.167 | 0.0003617 |
| dot_general_read_into_f64_s256 | pytorch-cuda | 8 | 15 | 2.084 | 0 |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 8 | 15 | 2.096 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 8 | 15 | 2.096 | 0 |
| dot_general_read_into_c32_s256 | pytorch-cuda | 128 | 15 | 3.453 | 0.0008017 |
| dot_general_read_into_accum_c32_s256 | pytorch-cuda | 128 | 15 | 3.556 | 0.000499 |
| dot_general_read_into_accum_cached_c32_s256 | pytorch-cuda | 128 | 15 | 3.558 | 0.000499 |
| dot_general_read_into_c64_s256 | pytorch-cuda | 4 | 15 | 3.899 | 1.175e-06 |
| dot_general_read_into_accum_c64_s256 | pytorch-cuda | 4 | 15 | 3.920 | 6.523e-07 |
| dot_general_read_into_accum_cached_c64_s256 | pytorch-cuda | 4 | 15 | 3.921 | 6.523e-07 |
| add_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.621 | 0 |
| add_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.721 | 0 |
| mul_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.674 | 0 |
| mul_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.605 | 0 |
| neg_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.444 | 0 |
| neg_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.556 | 0 |
| exp_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.302 | 0.0001272 |
| exp_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.341 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.384 | 9.107e-05 |
| tanh_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.363 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 4.022 | 0.0005233 |
| reduce_sum_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 3.741 | 3.504e-07 |
| conj_strided_read_c32_n1024 | pytorch-cuda | 256 | 15 | 2.965 | 0 |
| conj_strided_read_c64_n1024 | pytorch-cuda | 256 | 15 | 3.021 | 0 |
| add_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.546 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.490 | 0 |
| mul_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.547 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.492 | 0 |
| neg_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.699 | 0 |
| exp_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.754 | 0.0001272 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.895 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.757 | 9.135e-05 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.684 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.108 | 0.002693 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.366 | 5.935e-06 |
| conj_strided_read_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.698 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.667 | 0 |
| axpby_overwrite_f32_n1024 | pytorch-cuda | 256 | 15 | 3.061 | 0 |
| axpby_lerp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.487 | 0.0001189 |
| axpby_unit_add_f32_n1024 | pytorch-cuda | 256 | 15 | 2.275 | 0 |
| axpby_overwrite_f64_n1024 | pytorch-cuda | 256 | 15 | 2.951 | 0 |
| axpby_lerp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.436 | 1.161e-07 |
| axpby_unit_add_f64_n1024 | pytorch-cuda | 256 | 15 | 2.260 | 0 |
| axpby_overwrite_c32_n1024 | pytorch-cuda | 256 | 15 | 3.063 | 0 |
| axpby_lerp_c32_n1024 | pytorch-cuda | 256 | 15 | 2.068 | 0.0001189 |
| axpby_unit_add_c32_n1024 | pytorch-cuda | 256 | 15 | 2.232 | 0 |
| axpby_overwrite_c64_n1024 | pytorch-cuda | 256 | 15 | 3.056 | 0 |
| axpby_lerp_c64_n1024 | pytorch-cuda | 256 | 15 | 2.785 | 1.161e-07 |
| axpby_unit_add_c64_n1024 | pytorch-cuda | 256 | 15 | 2.350 | 0 |
| axpby_overwrite_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.760 | 0 |
| axpby_lerp_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.546 | 0.0001192 |
| axpby_unit_add_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.542 | 0 |
| axpby_overwrite_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.702 | 0 |
| axpby_lerp_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.498 | 1.161e-07 |
| axpby_unit_add_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.487 | 0 |
| axpby_overwrite_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.701 | 0 |
| axpby_lerp_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.494 | 0.0001191 |
| axpby_unit_add_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| axpby_overwrite_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.671 | 0 |
| axpby_lerp_c64_n1048576 | pytorch-cuda | 8 | 15 | 2.735 | 1.159e-07 |
| axpby_unit_add_c64_n1048576 | pytorch-cuda | 16 | 15 | 2.457 | 0 |
| add_into_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.067 | 0 |
| add_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.784 | 0 |
| add_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.823 | 0 |
| add_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.922 | 0 |
| sub_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.984 | 0 |
| sub_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.839 | 0 |
| sub_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.864 | 0 |
| sub_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.874 | 0 |
| mul_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.834 | 0 |
| mul_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.908 | 0 |
| mul_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.930 | 0 |
| mul_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.918 | 0 |
| div_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.925 | 0 |
| div_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.859 | 0 |
| div_read_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.696 | 0 |
| div_read_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.854 | 0 |
| neg_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.618 | 0 |
| neg_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.519 | 0 |
| neg_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.490 | 0 |
| neg_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.602 | 0 |
| conj_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.541 | 0 |
| conj_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.948 | 0 |
| conj_read_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.499 | 0 |
| conj_read_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.782 | 0 |
| add_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.764 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.665 | 0 |
| add_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.740 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.640 | 0 |
| sub_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.738 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.634 | 0 |
| sub_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.744 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.642 | 0 |
| mul_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.731 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.643 | 0 |
| mul_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.745 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.651 | 0 |
| div_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.244 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.424 | 0 |
| div_read_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.243 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.421 | 0 |
| neg_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.893 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.828 | 0 |
| neg_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.905 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.830 | 0 |
| conj_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.452 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.342 | 0 |
| conj_read_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.445 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.344 | 0 |
| tensordot_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.796 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.636 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.810 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-direct | 32 | 15 | 3.968 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.526 | 0 |
| vdot_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.709 | 1.722e-07 |
| vdot_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.650 | 0 |
| vdot_read_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.142 | 3.455e-07 |
| norm_squared_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.903 | 0 |
| norm_squared_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.686 | 0 |
| tensordot_f32_n262144 | tenferro-cuda-direct | 32 | 15 | 2.143 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 2 | 15 | 3.228 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.250 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-direct | 1 | 15 | 5.020 | 1.446e-05 |
| vdot_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.687 | 0 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.002 | 2.995e-06 |
| vdot_read_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 3.068 | 0 |
| vdot_read_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.068 | 8.762e-06 |
| norm_squared_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 3.079 | 0.0001786 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.061 | 0 |
| axpby_read_into_accum_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.623 | 0 |
| axpby_read_into_accum_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.616 | 0 |
| axpby_read_into_accum_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.465 | 0 |
| axpby_read_into_accum_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.474 | 0 |
| axpby_read_into_accum_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.185 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.137 | 0 |
| axpby_read_into_accum_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.701 | 0 |
| axpby_read_into_accum_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.017 | 0 |
| dot_general_read_into_f32_s16 | tenferro-cuda-direct | 128 | 15 | 2.951 | 8.645e-05 |
| dot_general_read_into_accum_f32_s16 | tenferro-cuda-direct | 128 | 15 | 2.921 | 7.907e-05 |
| dot_general_read_into_accum_cached_f32_s16 | tenferro-cuda-direct | 128 | 15 | 2.939 | 7.907e-05 |
| dot_general_read_into_f64_s16 | tenferro-cuda-direct | 128 | 15 | 2.967 | 0 |
| dot_general_read_into_accum_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.090 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.122 | 0 |
| dot_general_read_into_c32_s16 | tenferro-cuda-direct | 128 | 15 | 3.048 | 0.0003615 |
| dot_general_read_into_accum_c32_s16 | tenferro-cuda-direct | 128 | 15 | 3.071 | 0.0001632 |
| dot_general_read_into_accum_cached_c32_s16 | tenferro-cuda-direct | 128 | 15 | 3.025 | 0.0001632 |
| dot_general_read_into_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.018 | 1.204e-07 |
| dot_general_read_into_accum_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.163 | 6.975e-08 |
| dot_general_read_into_accum_cached_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.102 | 6.975e-08 |
| dot_general_read_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.944 | 0.0005542 |
| dot_general_read_into_accum_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.864 | 0.0003617 |
| dot_general_read_into_accum_cached_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.165 | 0.0003617 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.104 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.201 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.546 | 0 |
| dot_general_read_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.648 | 0.001457 |
| dot_general_read_into_accum_c32_s256 | tenferro-cuda-direct | 64 | 15 | 1.932 | 0.0008947 |
| dot_general_read_into_accum_cached_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.818 | 0.0008947 |
| dot_general_read_into_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.732 | 2.092e-06 |
| dot_general_read_into_accum_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.739 | 1.318e-06 |
| dot_general_read_into_accum_cached_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.737 | 1.318e-06 |
| add_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.460 | 0 |
| add_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 4.258 | 0 |
| mul_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.822 | 0 |
| mul_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.593 | 0 |
| neg_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.267 | 0 |
| neg_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.860 | 0 |
| exp_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.294 | 0.0001272 |
| exp_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.031 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.303 | 9.107e-05 |
| tanh_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.020 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 3.169 | 0.0001208 |
| reduce_sum_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.545 | 4.202e-07 |
| conj_strided_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.687 | 0 |
| conj_strided_read_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 4.004 | 0 |
| add_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.138 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.011 | 0 |
| mul_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.118 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.008 | 0 |
| neg_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.586 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.460 | 0 |
| exp_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.598 | 0.0001272 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.735 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.597 | 9.135e-05 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.526 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.983 | 0.002489 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.971 | 3.227e-06 |
| conj_strided_read_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.655 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.425 | 0 |
| axpby_overwrite_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.621 | 0 |
| axpby_lerp_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.638 | 0 |
| axpby_unit_add_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.689 | 0 |
| axpby_overwrite_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.708 | 0 |
| axpby_lerp_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.640 | 1.155e-07 |
| axpby_unit_add_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.689 | 0 |
| axpby_overwrite_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.572 | 0 |
| axpby_lerp_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.629 | 0 |
| axpby_unit_add_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.552 | 0 |
| axpby_overwrite_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.527 | 0 |
| axpby_lerp_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.531 | 1.155e-07 |
| axpby_unit_add_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.522 | 0 |
| axpby_overwrite_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.438 | 0 |
| axpby_lerp_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.188 | 0 |
| axpby_unit_add_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.186 | 0 |
| axpby_overwrite_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.715 | 0 |
| axpby_lerp_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.137 | 1.16e-07 |
| axpby_unit_add_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.140 | 0 |
| axpby_overwrite_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.501 | 0 |
| axpby_lerp_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.704 | 0 |
| axpby_unit_add_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.702 | 0 |
| axpby_overwrite_c64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.703 | 0 |
| axpby_lerp_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.145 | 1.159e-07 |
| axpby_unit_add_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.145 | 0 |
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
| tensordot_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.724 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.878 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.616 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.507 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| tensordot_f32_n262144 | tenferro-cuda-eager | 64 | 15 | 3.015 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 2 | 15 | 3.259 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-eager | 16 | 15 | 2.275 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-eager | 1 | 15 | 5.058 | 1.446e-05 |
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

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: benchmarks/gpu/reuse_contraction.yaml
timestamp: '2026-10-13T18:37:08Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 89a0095302c7664bbd0c8a91d513a0c1338ea662
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
    ./scripts/run_gpu_public_api.sh reuse_contraction
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 152
  selected_cases: 152
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 841a675fd8e0291513cc2659d53ac8c1f996cc84212497f6dc913b0f53354976
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/multi.rs: 1b477fef78260cf28aa31e515d57ab411f6241b91ffb30d625939538fdbc6bab
    src/bin/benchmark_gpu_public_api/read_layout.rs: bd7ce4a42576d1937a7048b7b002cac4838a7cd3c53628bc9aa6aebb0c5c1128
    src/bin/benchmark_gpu_public_api/structural.rs: 8121b12b5fed3b0acbc71e2218856173d53a5acc93092b93c7fd3309d4f45962
    src/bin/benchmark_gpu_public_api/linalg.rs: 19cd0d5e23b63d2b525611bffa02a62dbe4c4ed77dd2604ff379fa80130e392e
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: 62173a8bb61beaa98f93468de4db2930ff1f95520f91e7aa3d00f43da3ab6b15
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 7481dd1755c2772445c81f8fa3f986ff45621a1ee683b11e22b68add331ced88
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 2d4a81e918d1fac64d4fc943e097d853ab7e6f72f44a05ad36a73f0b43a7ec9f
    scripts/run_gpu_public_api.sh: dd5e4cc47e85fcb5afdc56d460136b2a399a6a7d6a62005de8ba94afb0119ecc
    data/results/nvidia-gpu/gpu/reuse_contraction/20261013_183707/fixtures/manifest.json: 215704bbfbb1b7f71d8776470a4d7659f840963e2b36877a8378de81b10bc3d6
    target/release/benchmark_gpu_public_api: ce2112f54cc14ccef9adb958f9ea45e87264e03612834c1fc0c1523f64125455
  reference_numpy_version: 2.4.2
```
