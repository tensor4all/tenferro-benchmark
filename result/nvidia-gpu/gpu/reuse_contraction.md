# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **572**. Unsupported cells: **348**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261013_185827/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

Unit-coefficient AXPBY uses native JAX Ref.addupdate with caller-owned fixed buffers. Its synchronization includes one GPU scalar completion token per destination and block_until_ready; tokens are retained through clock stop and included in the logical retention bound. Helper compilation and seed restoration are untimed. Full Ref snapshots are used only for untimed inspection. This scalar-token synchronization cost is included in the reported latency; it performs no full output copy or D2H transfer inside timing.

Generalized batch dot preserves native output order: tenferro [m,n,b], PyTorch/JAX [b,m,n]. Input permutation views and dimension-number configurations are prepared before timing. Output axes are normalized only on downloaded CPU arrays for verification; no timed output transpose or copy is added.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | 7.031 (0.765) | unsupported | 6.421 (0.129) | unsupported | unsupported |
| add_into_f64_n1024 | 5.303 (0.170) | unsupported | 6.577 (0.058) | unsupported | unsupported |
| add_read_into_f32_n1024 | 5.472 (0.046) | unsupported | 6.397 (0.048) | unsupported | unsupported |
| add_read_into_f64_n1024 | 5.423 (0.190) | unsupported | 6.645 (0.171) | unsupported | unsupported |
| sub_into_f32_n1024 | 5.596 (0.305) | unsupported | 7.245 (0.160) | unsupported | unsupported |
| sub_into_f64_n1024 | 5.390 (0.100) | unsupported | 7.292 (0.046) | unsupported | unsupported |
| sub_read_into_f32_n1024 | 5.531 (0.167) | unsupported | 7.233 (0.205) | unsupported | unsupported |
| sub_read_into_f64_n1024 | 5.444 (0.128) | unsupported | 7.288 (0.112) | unsupported | unsupported |
| mul_into_f32_n1024 | 5.465 (0.180) | unsupported | 6.569 (0.055) | unsupported | unsupported |
| mul_into_f64_n1024 | 5.638 (0.454) | unsupported | 6.534 (0.354) | unsupported | unsupported |
| mul_read_into_f32_n1024 | 5.380 (0.193) | unsupported | 6.523 (0.079) | unsupported | unsupported |
| mul_read_into_f64_n1024 | 5.349 (0.135) | unsupported | 6.474 (0.187) | unsupported | unsupported |
| div_into_f32_n1024 | 42.436 (2.184) | unsupported | 6.724 (0.100) | unsupported | unsupported |
| div_into_f64_n1024 | 41.038 (1.014) | unsupported | 6.796 (0.180) | unsupported | unsupported |
| div_read_into_f32_n1024 | 41.788 (0.971) | unsupported | 6.742 (0.162) | unsupported | unsupported |
| div_read_into_f64_n1024 | 40.966 (0.889) | unsupported | 6.781 (0.069) | unsupported | unsupported |
| neg_into_f32_n1024 | 4.870 (0.175) | unsupported | 5.964 (0.089) | unsupported | unsupported |
| neg_into_f64_n1024 | 4.845 (0.141) | unsupported | 5.991 (0.019) | unsupported | unsupported |
| neg_read_into_f32_n1024 | 4.856 (0.142) | unsupported | 5.945 (0.070) | unsupported | unsupported |
| neg_read_into_f64_n1024 | 4.732 (0.165) | unsupported | 6.000 (0.069) | unsupported | unsupported |
| conj_into_c32_n1024 | 37.972 (1.606) | unsupported | 6.710 (0.145) | unsupported | unsupported |
| conj_into_c64_n1024 | 167.168 (10.159) | unsupported | 6.850 (0.231) | unsupported | unsupported |
| conj_read_into_c32_n1024 | 41.618 (1.335) | unsupported | 7.117 (0.301) | unsupported | unsupported |
| conj_read_into_c64_n1024 | 163.699 (3.731) | unsupported | 6.885 (0.314) | unsupported | unsupported |
| add_into_f32_n1048576 | 43.540 (0.255) | unsupported | 39.698 (0.102) | unsupported | unsupported |
| add_into_f64_n1048576 | 83.793 (0.739) | unsupported | 77.773 (0.314) | unsupported | unsupported |
| add_read_into_f32_n1048576 | 43.356 (0.509) | unsupported | 39.649 (0.079) | unsupported | unsupported |
| add_read_into_f64_n1048576 | 82.478 (0.492) | unsupported | 77.609 (0.052) | unsupported | unsupported |
| sub_into_f32_n1048576 | 42.715 (0.162) | unsupported | 39.682 (0.051) | unsupported | unsupported |
| sub_into_f64_n1048576 | 82.044 (0.514) | unsupported | 77.632 (0.107) | unsupported | unsupported |
| sub_read_into_f32_n1048576 | 42.770 (0.151) | unsupported | 39.690 (0.065) | unsupported | unsupported |
| sub_read_into_f64_n1048576 | 81.857 (0.265) | unsupported | 77.597 (0.048) | unsupported | unsupported |
| mul_into_f32_n1048576 | 42.956 (0.382) | unsupported | 39.646 (0.045) | unsupported | unsupported |
| mul_into_f64_n1048576 | 82.731 (0.778) | unsupported | 77.649 (0.272) | unsupported | unsupported |
| mul_read_into_f32_n1048576 | 42.715 (0.208) | unsupported | 39.675 (0.064) | unsupported | unsupported |
| mul_read_into_f64_n1048576 | 82.005 (0.403) | unsupported | 77.536 (0.152) | unsupported | unsupported |
| div_into_f32_n1048576 | 70.103 (0.132) | unsupported | 39.998 (0.144) | unsupported | unsupported |
| div_into_f64_n1048576 | 151.346 (0.105) | unsupported | 94.370 (0.163) | unsupported | unsupported |
| div_read_into_f32_n1048576 | 70.169 (0.225) | unsupported | 39.901 (0.060) | unsupported | unsupported |
| div_read_into_f64_n1048576 | 151.412 (0.113) | unsupported | 94.373 (0.108) | unsupported | unsupported |
| neg_into_f32_n1048576 | 29.615 (0.255) | unsupported | 27.220 (0.009) | unsupported | unsupported |
| neg_into_f64_n1048576 | 56.810 (0.210) | unsupported | 52.859 (0.035) | unsupported | unsupported |
| neg_read_into_f32_n1048576 | 29.654 (0.171) | unsupported | 27.210 (0.032) | unsupported | unsupported |
| neg_read_into_f64_n1048576 | 56.886 (0.523) | unsupported | 52.874 (0.062) | unsupported | unsupported |
| conj_into_c32_n1048576 | 107.948 (0.166) | unsupported | 52.806 (0.044) | unsupported | unsupported |
| conj_into_c64_n1048576 | 209.046 (0.397) | unsupported | 103.766 (0.275) | unsupported | unsupported |
| conj_read_into_c32_n1048576 | 107.865 (0.079) | unsupported | 52.935 (0.140) | unsupported | unsupported |
| conj_read_into_c64_n1048576 | 209.263 (0.577) | unsupported | 103.900 (0.482) | unsupported | unsupported |
| tensordot_f32_n1024 | 27.750 (0.412) | 65.019 (1.931) | 50.071 (1.315) | 179.584 (5.896) | 81.766 (3.410) |
| tensordot_f64_n1024 | 29.184 (2.463) | 43.614 (0.504) | 53.600 (0.466) | 152.970 (7.505) | 72.053 (2.608) |
| tensordot_c32_n1024 | 28.790 (1.401) | 40.770 (1.761) | 50.997 (0.863) | 168.462 (2.440) | 86.987 (2.603) |
| tensordot_c64_n1024 | 113.397 (2.099) | 43.494 (0.584) | 51.980 (0.541) | 175.955 (3.335) | 92.981 (2.890) |
| vdot_read_f32_n1024 | 139.584 (6.270) | unsupported | 15.212 (0.341) | 65.453 (1.358) | 88.072 (1.418) |
| vdot_read_f64_n1024 | 155.229 (6.418) | unsupported | 15.378 (0.242) | 72.122 (1.762) | 90.227 (4.399) |
| vdot_read_c32_n1024 | 162.345 (4.480) | unsupported | 14.760 (0.128) | 100.597 (3.030) | 136.523 (5.703) |
| vdot_read_c64_n1024 | 184.130 (3.668) | unsupported | 15.352 (0.150) | 91.438 (2.172) | 156.953 (12.956) |
| norm_squared_read_f32_n1024 | 170.043 (6.726) | unsupported | 14.976 (0.081) | 72.772 (6.913) | 155.881 (6.825) |
| norm_squared_read_f64_n1024 | 157.279 (2.964) | unsupported | 14.992 (0.255) | 66.209 (2.907) | 157.558 (6.494) |
| tensordot_f32_n262144 | 71.271 (5.240) | 47.316 (0.389) | 57.848 (0.236) | 160.496 (5.223) | 84.856 (2.548) |
| tensordot_f64_n262144 | 1673.892 (1.608) | 1619.263 (15.993) | 1633.626 (2.815) | 1704.003 (32.416) | 1588.509 (32.563) |
| tensordot_c32_n262144 | 141.299 (0.064) | 141.539 (0.351) | 158.622 (1.816) | 316.532 (12.577) | 160.645 (2.610) |
| tensordot_c64_n262144 | 5031.346 (2.547) | 5031.509 (5.682) | 5279.912 (3.907) | 5602.758 (27.938) | 5359.708 (58.958) |
| vdot_read_f32_n262144 | 162.616 (4.130) | unsupported | 15.568 (0.485) | 166.691 (3.010) | 145.634 (34.846) |
| vdot_read_f64_n262144 | 181.782 (8.333) | unsupported | 20.314 (0.031) | 158.959 (2.958) | 169.578 (3.724) |
| vdot_read_c32_n262144 | 182.469 (3.418) | unsupported | 17.427 (0.044) | 163.466 (1.839) | 172.320 (27.334) |
| vdot_read_c64_n262144 | 182.252 (2.960) | unsupported | 33.889 (0.036) | 166.027 (10.382) | 172.552 (19.056) |
| norm_squared_read_f32_n262144 | 178.683 (5.614) | unsupported | 15.181 (0.608) | 135.840 (28.673) | 89.327 (1.506) |
| norm_squared_read_f64_n262144 | 179.431 (7.369) | unsupported | 15.034 (0.148) | 123.502 (18.449) | 148.305 (2.108) |
| axpby_read_into_accum_f32_n1024 | 12.053 (3.651) | unsupported | 7.715 (0.060) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1024 | 11.788 (0.375) | unsupported | 7.867 (0.111) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1024 | 11.392 (0.545) | unsupported | 7.758 (0.059) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1024 | 11.522 (0.289) | unsupported | 7.804 (0.042) | unsupported | unsupported |
| axpby_read_into_accum_f32_n1048576 | 273.215 (0.394) | unsupported | 39.846 (0.163) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1048576 | 534.729 (0.291) | unsupported | 77.946 (0.195) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1048576 | 169.002 (0.248) | unsupported | 77.720 (0.131) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1048576 | 2015.446 (1.707) | unsupported | 153.534 (0.242) | unsupported | unsupported |
| dot_general_read_into_f32_s16 | 22.768 (0.529) | unsupported | 19.643 (0.143) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s16 | 24.666 (1.196) | unsupported | 23.567 (0.212) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s16 | 23.110 (0.359) | unsupported | 23.642 (0.268) | unsupported | unsupported |
| dot_general_read_into_f64_s16 | 26.299 (0.418) | unsupported | 20.693 (0.520) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s16 | 25.918 (0.523) | unsupported | 25.534 (0.731) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s16 | 25.573 (0.344) | unsupported | 25.406 (0.213) | unsupported | unsupported |
| dot_general_read_into_c32_s16 | 23.271 (0.597) | unsupported | 19.731 (0.180) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s16 | 23.391 (1.107) | unsupported | 23.888 (1.094) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s16 | 23.184 (0.812) | unsupported | 24.051 (1.392) | unsupported | unsupported |
| dot_general_read_into_c64_s16 | 26.228 (0.823) | unsupported | 20.994 (0.191) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s16 | 25.595 (0.449) | unsupported | 26.425 (2.289) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s16 | 25.220 (0.600) | unsupported | 25.306 (0.328) | unsupported | unsupported |
| dot_general_read_into_f32_s256 | 23.472 (0.876) | unsupported | 19.926 (0.421) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s256 | 23.433 (0.383) | unsupported | 25.244 (0.891) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s256 | 23.394 (0.491) | unsupported | 24.599 (0.250) | unsupported | unsupported |
| dot_general_read_into_f64_s256 | 263.282 (10.713) | unsupported | 249.019 (5.758) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s256 | 262.065 (12.885) | unsupported | 270.999 (15.015) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s256 | 271.953 (19.969) | unsupported | 281.010 (10.490) | unsupported | unsupported |
| dot_general_read_into_c32_s256 | 28.659 (0.386) | unsupported | 26.946 (0.092) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s256 | 29.853 (0.144) | unsupported | 27.750 (0.149) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s256 | 29.851 (0.079) | unsupported | 27.794 (0.077) | unsupported | unsupported |
| dot_general_read_into_c64_s256 | 683.178 (0.481) | unsupported | 974.692 (0.858) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s256 | 685.133 (2.885) | unsupported | 980.135 (0.620) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s256 | 684.675 (0.723) | unsupported | 980.125 (0.657) | unsupported | unsupported |
| add_strided_read_f32_n1024 | 284.586 (7.418) | unsupported | 10.143 (0.390) | 126.664 (15.758) | 130.749 (17.995) |
| add_strided_read_f64_n1024 | 246.157 (18.396) | unsupported | 9.933 (0.308) | 132.300 (31.125) | 153.033 (20.605) |
| mul_strided_read_f32_n1024 | 325.203 (5.292) | unsupported | 10.453 (0.117) | 137.401 (29.483) | 137.150 (24.367) |
| mul_strided_read_f64_n1024 | 304.688 (3.413) | unsupported | 10.787 (0.236) | 134.082 (15.071) | 125.272 (7.576) |
| neg_strided_read_f32_n1024 | 192.557 (2.948) | unsupported | 9.357 (0.190) | 126.892 (12.831) | 115.589 (15.812) |
| neg_strided_read_f64_n1024 | 166.900 (4.513) | unsupported | 9.744 (0.277) | 103.277 (14.692) | 96.860 (15.118) |
| exp_strided_read_f32_n1024 | 192.156 (2.933) | unsupported | 9.158 (0.135) | 94.216 (16.094) | 95.670 (6.713) |
| exp_strided_read_f64_n1024 | 167.800 (3.526) | unsupported | 9.458 (0.249) | 98.345 (12.382) | 97.420 (17.465) |
| tanh_strided_read_f32_n1024 | 192.569 (2.879) | unsupported | 9.282 (0.092) | 94.443 (12.201) | 97.681 (15.663) |
| tanh_strided_read_f64_n1024 | 165.471 (2.507) | unsupported | 9.201 (0.061) | 98.477 (12.535) | 97.634 (9.433) |
| reduce_sum_strided_read_f32_n1024 | 332.853 (12.691) | unsupported | 14.462 (0.115) | 94.977 (6.203) | 103.023 (17.529) |
| reduce_sum_strided_read_f64_n1024 | 291.171 (4.451) | unsupported | 15.016 (0.569) | 92.680 (4.665) | 86.621 (0.790) |
| conj_strided_read_c32_n1024 | 164.964 (8.451) | unsupported | 11.538 (0.457) | 94.198 (24.194) | 99.708 (19.314) |
| conj_strided_read_c64_n1024 | 239.274 (12.449) | unsupported | 11.848 (0.227) | 149.818 (23.865) | 123.265 (15.922) |
| add_strided_read_f32_n1048576 | 98.214 (0.351) | unsupported | 39.800 (0.026) | 128.933 (15.724) | 100.720 (16.161) |
| add_strided_read_f64_n1048576 | 188.492 (0.159) | unsupported | 77.832 (0.145) | 118.476 (3.198) | 91.473 (5.882) |
| mul_strided_read_f32_n1048576 | 97.638 (0.329) | unsupported | 39.785 (0.019) | 101.879 (2.175) | 93.756 (0.634) |
| mul_strided_read_f64_n1048576 | 188.471 (0.310) | unsupported | 77.764 (0.059) | 94.612 (4.570) | 89.163 (0.521) |
| neg_strided_read_f32_n1048576 | 56.295 (0.125) | unsupported | 27.356 (0.037) | 91.108 (1.944) | 91.126 (0.253) |
| neg_strided_read_f64_n1048576 | 108.412 (0.073) | unsupported | 53.089 (0.033) | 97.490 (1.535) | 93.047 (0.515) |
| exp_strided_read_f32_n1048576 | 56.429 (0.233) | unsupported | 27.436 (0.028) | 86.416 (2.266) | 91.119 (0.403) |
| exp_strided_read_f64_n1048576 | 233.547 (0.217) | unsupported | 181.190 (0.506) | 204.368 (1.630) | 204.392 (1.899) |
| tanh_strided_read_f32_n1048576 | 56.274 (0.136) | unsupported | 27.432 (0.038) | 87.449 (2.360) | 92.941 (1.145) |
| tanh_strided_read_f64_n1048576 | 220.469 (0.250) | unsupported | 167.644 (0.065) | 189.841 (1.856) | 188.733 (1.634) |
| reduce_sum_strided_read_f32_n1048576 | 178.765 (6.653) | unsupported | 21.463 (0.160) | 151.234 (3.000) | 152.089 (4.763) |
| reduce_sum_strided_read_f64_n1048576 | 93.045 (0.273) | unsupported | 36.980 (0.067) | 155.874 (3.296) | 145.474 (4.419) |
| conj_strided_read_c32_n1048576 | 114.318 (0.186) | unsupported | 53.046 (0.081) | 96.615 (7.763) | 98.281 (3.761) |
| conj_strided_read_c64_n1048576 | 213.986 (0.185) | unsupported | 104.290 (0.249) | 116.944 (0.848) | 115.692 (0.435) |
| axpby_overwrite_f32_n1024 | 10.999 (0.217) | unsupported | 12.070 (1.087) | unsupported | unsupported |
| axpby_lerp_f32_n1024 | 10.927 (0.236) | unsupported | 9.082 (0.072) | unsupported | unsupported |
| axpby_unit_add_f32_n1024 | 11.218 (0.416) | unsupported | 7.675 (0.248) | 335.702 (24.129) | 238.254 (23.662) |
| axpby_overwrite_f64_n1024 | 11.376 (0.402) | unsupported | 11.306 (0.461) | unsupported | unsupported |
| axpby_lerp_f64_n1024 | 11.705 (0.491) | unsupported | 8.761 (0.141) | unsupported | unsupported |
| axpby_unit_add_f64_n1024 | 11.645 (0.378) | unsupported | 7.781 (0.048) | 332.766 (29.074) | 242.578 (17.806) |
| axpby_overwrite_c32_n1024 | 11.442 (0.360) | unsupported | 11.563 (0.305) | unsupported | unsupported |
| axpby_lerp_c32_n1024 | 11.761 (0.434) | unsupported | 7.469 (0.449) | unsupported | unsupported |
| axpby_unit_add_c32_n1024 | 11.803 (0.415) | unsupported | 7.818 (0.399) | 328.607 (23.211) | 237.872 (28.303) |
| axpby_overwrite_c64_n1024 | 11.783 (0.949) | unsupported | 11.742 (0.206) | unsupported | unsupported |
| axpby_lerp_c64_n1024 | 12.052 (0.290) | unsupported | 10.906 (0.008) | unsupported | unsupported |
| axpby_unit_add_c64_n1024 | 11.966 (0.198) | unsupported | 7.880 (0.031) | 351.825 (18.183) | 245.322 (22.998) |
| axpby_overwrite_f32_n1048576 | 304.856 (0.885) | unsupported | 27.433 (0.055) | unsupported | unsupported |
| axpby_lerp_f32_n1048576 | 273.312 (0.349) | unsupported | 39.806 (0.046) | unsupported | unsupported |
| axpby_unit_add_f32_n1048576 | 273.281 (0.439) | unsupported | 39.742 (0.032) | 362.856 (22.797) | 251.477 (17.222) |
| axpby_overwrite_f64_n1048576 | 339.627 (0.539) | unsupported | 53.192 (0.065) | unsupported | unsupported |
| axpby_lerp_f64_n1048576 | 534.973 (0.296) | unsupported | 78.068 (0.050) | unsupported | unsupported |
| axpby_unit_add_f64_n1048576 | 535.027 (0.409) | unsupported | 77.815 (0.144) | 366.416 (13.806) | 252.174 (14.256) |
| axpby_overwrite_c32_n1048576 | 156.397 (0.899) | unsupported | 53.188 (0.362) | unsupported | unsupported |
| axpby_lerp_c32_n1048576 | 163.434 (0.054) | unsupported | 77.986 (0.237) | unsupported | unsupported |
| axpby_unit_add_c32_n1048576 | 163.415 (0.114) | unsupported | 77.689 (0.098) | 368.547 (19.440) | 251.404 (12.841) |
| axpby_overwrite_c64_n1048576 | 1303.873 (0.837) | unsupported | 104.490 (0.119) | unsupported | unsupported |
| axpby_lerp_c64_n1048576 | 2067.090 (1.162) | unsupported | 341.863 (0.253) | unsupported | unsupported |
| axpby_unit_add_c64_n1048576 | 2067.074 (1.239) | unsupported | 153.588 (0.166) | 369.107 (29.365) | 247.496 (15.138) |
| dot_general_batch_f32_s32 | 123.062 (2.014) | 53.304 (1.977) | 42.263 (1.986) | 112.826 (4.819) | 64.366 (1.346) |
| dot_general_batch_f64_s32 | 131.502 (1.779) | 53.314 (1.750) | 42.695 (1.488) | 115.149 (7.227) | 65.346 (1.742) |
| dot_general_batch_c32_s32 | 128.995 (7.517) | 50.358 (2.107) | 40.135 (1.422) | 142.352 (1.467) | 92.554 (10.776) |
| dot_general_batch_c64_s32 | 146.600 (4.049) | 50.214 (0.872) | 41.548 (0.594) | 158.583 (14.968) | 95.367 (12.153) |
| dot_general_multiple_f32_s32 | 164.768 (2.603) | 57.064 (2.389) | 75.496 (0.862) | 163.067 (3.016) | 76.375 (1.803) |
| dot_general_multiple_f64_s32 | 157.722 (3.188) | 57.606 (2.672) | 79.048 (1.713) | 127.524 (6.538) | 71.239 (1.882) |
| dot_general_multiple_c32_s32 | 160.736 (2.133) | 56.196 (0.873) | 75.272 (0.688) | 138.967 (6.495) | 95.977 (13.892) |
| dot_general_multiple_c64_s32 | 159.066 (1.579) | 54.799 (0.906) | 77.037 (0.583) | 167.852 (28.901) | 105.174 (18.207) |
| dot_general_batch_f32_s256 | 55.569 (3.614) | 42.735 (0.486) | 41.887 (0.635) | 118.295 (7.431) | 68.988 (2.792) |
| dot_general_batch_f64_s256 | 351.960 (0.615) | 331.369 (0.780) | 420.055 (0.503) | 323.482 (7.848) | 319.617 (13.342) |
| dot_general_batch_c32_s256 | 41.958 (0.980) | 44.218 (0.884) | 54.205 (0.043) | 198.725 (33.613) | 150.646 (35.763) |
| dot_general_batch_c64_s256 | 1101.947 (1.073) | 1101.125 (0.727) | 1333.710 (0.901) | 1444.584 (18.065) | 1336.433 (55.438) |
| dot_general_multiple_f32_s256 | 47.865 (0.478) | 49.915 (1.374) | 78.800 (4.043) | 185.272 (59.672) | 93.948 (1.554) |
| dot_general_multiple_f64_s256 | 331.967 (0.291) | 334.815 (0.879) | 339.315 (9.627) | 218.529 (2.667) | 216.640 (1.440) |
| dot_general_multiple_c32_s256 | 45.483 (0.483) | 46.244 (0.658) | 83.438 (0.631) | 326.594 (15.242) | 200.796 (46.212) |
| dot_general_multiple_c64_s256 | 827.118 (0.794) | 827.126 (0.763) | 1014.603 (0.766) | 1141.623 (23.894) | 1059.581 (14.766) |
| dot_conj_l0r0_c32_n1024 | 158.967 (2.471) | 34.923 (7.534) | 14.641 (0.233) | 159.380 (4.550) | 162.730 (5.693) |
| dot_conj_l1r0_c32_n1024 | 171.505 (4.446) | 121.851 (3.132) | 14.891 (0.225) | 142.262 (32.485) | 155.886 (15.749) |
| dot_conj_l0r1_c32_n1024 | 170.610 (8.334) | 116.536 (2.851) | 15.117 (0.933) | 95.331 (18.206) | 155.790 (12.837) |
| dot_conj_l1r1_c32_n1024 | 165.146 (4.517) | 109.986 (1.259) | unsupported | unsupported | unsupported |
| dot_conj_l0r0_c64_n1024 | 235.885 (4.886) | 31.482 (3.554) | 14.785 (0.852) | 114.655 (11.914) | 171.754 (17.090) |
| dot_conj_l1r0_c64_n1024 | 164.704 (5.407) | 108.339 (0.961) | 15.118 (0.351) | 87.648 (11.662) | 165.422 (14.574) |
| dot_conj_l0r1_c64_n1024 | 164.840 (4.110) | 104.055 (1.956) | 15.296 (0.445) | 75.035 (3.373) | 172.869 (14.239) |
| dot_conj_l1r1_c64_n1024 | 164.956 (5.767) | 102.908 (1.723) | unsupported | unsupported | unsupported |
| dot_conj_l0r0_c32_n262144 | 669.575 (14.337) | 617.522 (1.859) | 17.628 (0.185) | 100.542 (8.976) | 170.506 (22.086) |
| dot_conj_l1r0_c32_n262144 | 176.697 (3.375) | 92.631 (1.923) | 17.620 (0.086) | 80.798 (12.585) | 172.592 (25.501) |
| dot_conj_l0r1_c32_n262144 | 181.606 (9.670) | 94.970 (7.422) | 17.699 (0.207) | 72.825 (3.407) | 175.036 (24.043) |
| dot_conj_l1r1_c32_n262144 | 180.962 (5.568) | 99.546 (5.023) | unsupported | unsupported | unsupported |
| dot_conj_l0r0_c64_n262144 | 946.894 (10.422) | 899.341 (3.073) | 33.871 (0.019) | 113.632 (10.875) | 178.690 (19.588) |
| dot_conj_l1r0_c64_n262144 | 174.841 (8.658) | 102.088 (0.436) | 33.939 (0.024) | 99.084 (3.630) | 177.394 (22.344) |
| dot_conj_l0r1_c64_n262144 | 177.106 (3.758) | 102.648 (2.542) | 33.804 (0.077) | 93.299 (1.931) | 177.401 (22.070) |
| dot_conj_l1r1_c64_n262144 | 175.811 (3.202) | 102.299 (0.304) | unsupported | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.600 | 0 |
| add_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.715 | 0 |
| add_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.802 | 0 |
| add_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.776 | 0 |
| sub_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.865 | 0 |
| sub_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.759 | 0 |
| sub_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.832 | 0 |
| sub_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.787 | 0 |
| mul_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.798 | 0 |
| mul_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.887 | 0 |
| mul_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.755 | 0 |
| mul_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.739 | 0 |
| div_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.716 | 0 |
| div_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.626 | 0 |
| div_read_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.674 | 0 |
| div_read_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.622 | 0 |
| neg_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.493 | 0 |
| neg_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.480 | 0 |
| neg_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.486 | 0 |
| neg_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.423 | 0 |
| conj_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.430 | 0 |
| conj_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.675 | 0 |
| conj_read_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.664 | 0 |
| conj_read_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.619 | 0 |
| add_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.787 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.681 | 0 |
| add_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.775 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.639 | 0 |
| sub_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.734 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.625 | 0 |
| sub_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.737 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.619 | 0 |
| mul_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.749 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.647 | 0 |
| mul_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.734 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.624 | 0 |
| div_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.243 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.422 | 0 |
| div_read_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.245 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.423 | 0 |
| neg_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.895 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.818 | 0 |
| neg_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.898 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.820 | 0 |
| conj_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.454 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.345 | 0 |
| conj_read_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.452 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.348 | 0 |
| tensordot_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.552 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.735 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.685 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-direct | 32 | 15 | 3.629 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.233 | 0 |
| vdot_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.484 | 1.722e-07 |
| vdot_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.598 | 0 |
| vdot_read_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.946 | 3.455e-07 |
| norm_squared_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.721 | 0 |
| norm_squared_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.516 | 0 |
| tensordot_f32_n262144 | tenferro-cuda-direct | 32 | 15 | 2.281 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 2 | 15 | 3.348 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.261 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-direct | 1 | 15 | 5.031 | 1.446e-05 |
| vdot_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.602 | 0 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.909 | 2.995e-06 |
| vdot_read_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.920 | 0 |
| vdot_read_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.916 | 8.762e-06 |
| norm_squared_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.859 | 0.0001786 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.871 | 0 |
| axpby_read_into_accum_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.085 | 0 |
| axpby_read_into_accum_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.018 | 0 |
| axpby_read_into_accum_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.916 | 0 |
| axpby_read_into_accum_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.950 | 0 |
| axpby_read_into_accum_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.186 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.139 | 0 |
| axpby_read_into_accum_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.704 | 0 |
| axpby_read_into_accum_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.015 | 0 |
| dot_general_read_into_f32_s16 | tenferro-cuda-direct | 128 | 15 | 2.914 | 8.645e-05 |
| dot_general_read_into_accum_f32_s16 | tenferro-cuda-direct | 128 | 15 | 3.157 | 7.907e-05 |
| dot_general_read_into_accum_cached_f32_s16 | tenferro-cuda-direct | 128 | 15 | 2.958 | 7.907e-05 |
| dot_general_read_into_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.366 | 0 |
| dot_general_read_into_accum_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.317 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.273 | 0 |
| dot_general_read_into_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.979 | 0.0003615 |
| dot_general_read_into_accum_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.994 | 0.0001632 |
| dot_general_read_into_accum_cached_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.968 | 0.0001632 |
| dot_general_read_into_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.357 | 1.204e-07 |
| dot_general_read_into_accum_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.276 | 6.975e-08 |
| dot_general_read_into_accum_cached_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.228 | 6.975e-08 |
| dot_general_read_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.004 | 0.0005542 |
| dot_general_read_into_accum_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.999 | 0.0003617 |
| dot_general_read_into_accum_cached_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.994 | 0.0003617 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.106 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.097 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.176 | 0 |
| dot_general_read_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.668 | 0.001457 |
| dot_general_read_into_accum_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.821 | 0.0008947 |
| dot_general_read_into_accum_cached_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.821 | 0.0008947 |
| dot_general_read_into_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.733 | 2.092e-06 |
| dot_general_read_into_accum_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.741 | 1.318e-06 |
| dot_general_read_into_accum_cached_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.739 | 1.318e-06 |
| add_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.277 | 0 |
| add_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.939 | 0 |
| mul_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.602 | 0 |
| mul_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.438 | 0 |
| neg_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.081 | 0 |
| neg_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.670 | 0 |
| exp_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.075 | 0.0001272 |
| exp_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.685 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.081 | 9.107e-05 |
| tanh_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.648 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.663 | 0.0001208 |
| reduce_sum_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.329 | 4.202e-07 |
| conj_strided_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.639 | 0 |
| conj_strided_read_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.828 | 0 |
| add_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.143 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.016 | 0 |
| mul_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.124 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.016 | 0 |
| neg_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.603 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.469 | 0 |
| exp_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.611 | 0.0001272 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.737 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.602 | 9.135e-05 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.528 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.860 | 0.002489 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.977 | 3.227e-06 |
| conj_strided_read_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.658 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.424 | 0 |
| axpby_overwrite_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.816 | 0 |
| axpby_lerp_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.797 | 0 |
| axpby_unit_add_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.872 | 0 |
| axpby_overwrite_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.912 | 0 |
| axpby_lerp_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.996 | 1.155e-07 |
| axpby_unit_add_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.981 | 0 |
| axpby_overwrite_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.929 | 0 |
| axpby_lerp_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.011 | 0 |
| axpby_unit_add_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.021 | 0 |
| axpby_overwrite_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.016 | 0 |
| axpby_lerp_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.085 | 1.155e-07 |
| axpby_unit_add_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.063 | 0 |
| axpby_overwrite_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.439 | 0 |
| axpby_lerp_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.186 | 0 |
| axpby_unit_add_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.186 | 0 |
| axpby_overwrite_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.717 | 0 |
| axpby_lerp_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.140 | 1.16e-07 |
| axpby_unit_add_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.140 | 0 |
| axpby_overwrite_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.502 | 0 |
| axpby_lerp_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.615 | 0 |
| axpby_unit_add_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.615 | 0 |
| axpby_overwrite_c64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.608 | 0 |
| axpby_lerp_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.067 | 1.159e-07 |
| axpby_unit_add_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.067 | 0 |
| dot_general_batch_f32_s32 | tenferro-cuda-direct | 32 | 15 | 3.938 | 0.0002227 |
| dot_general_batch_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.104 | 1.326e-07 |
| dot_general_batch_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.064 | 0.0004376 |
| dot_general_batch_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.346 | 4.296e-07 |
| dot_general_multiple_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.636 | 0.0002162 |
| dot_general_multiple_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.524 | 1.632e-07 |
| dot_general_multiple_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.572 | 0.0002794 |
| dot_general_multiple_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.545 | 4.303e-07 |
| dot_general_batch_f32_s256 | tenferro-cuda-direct | 32 | 15 | 1.778 | 0.0008561 |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.816 | 7.463e-07 |
| dot_general_batch_c32_s256 | tenferro-cuda-direct | 64 | 15 | 2.685 | 0.001749 |
| dot_general_batch_c64_s256 | tenferro-cuda-direct | 2 | 15 | 2.204 | 1.665e-06 |
| dot_general_multiple_f32_s256 | tenferro-cuda-direct | 64 | 15 | 3.063 | 0.002418 |
| dot_general_multiple_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.656 | 1.519e-06 |
| dot_general_multiple_c32_s256 | tenferro-cuda-direct | 64 | 15 | 2.911 | 0.0022 |
| dot_general_multiple_c64_s256 | tenferro-cuda-direct | 4 | 15 | 3.308 | 2.187e-06 |
| dot_conj_l0r0_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.543 | 0.0004142 |
| dot_conj_l1r0_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.744 | 0.0002905 |
| dot_conj_l0r1_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.730 | 0.0002905 |
| dot_conj_l1r1_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.642 | 0.0003083 |
| dot_conj_l0r0_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.774 | 3.737e-07 |
| dot_conj_l1r0_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.635 | 1.807e-07 |
| dot_conj_l0r1_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.637 | 1.807e-07 |
| dot_conj_l1r1_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.639 | 4.927e-07 |
| dot_conj_l0r0_c32_n262144 | tenferro-cuda-direct | 4 | 15 | 2.678 | 0.02908 |
| dot_conj_l1r0_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.827 | 0.005619 |
| dot_conj_l0r1_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.906 | 0.005619 |
| dot_conj_l1r1_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.895 | 0.005822 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | 4 | 15 | 3.788 | 1.986e-05 |
| dot_conj_l1r0_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.797 | 1.516e-05 |
| dot_conj_l0r1_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.834 | 1.516e-05 |
| dot_conj_l1r1_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.813 | 1.212e-05 |
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
| tensordot_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.081 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.791 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.609 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.784 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| tensordot_f32_n262144 | tenferro-cuda-eager | 64 | 15 | 3.028 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 2 | 15 | 3.239 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-eager | 16 | 15 | 2.265 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-eager | 1 | 15 | 5.032 | 1.446e-05 |
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
| dot_general_batch_f32_s32 | tenferro-cuda-eager | 64 | 15 | 3.411 | 0.0002227 |
| dot_general_batch_f64_s32 | tenferro-cuda-eager | 64 | 15 | 3.412 | 1.326e-07 |
| dot_general_batch_c32_s32 | tenferro-cuda-eager | 64 | 15 | 3.223 | 0.0004376 |
| dot_general_batch_c64_s32 | tenferro-cuda-eager | 64 | 15 | 3.214 | 4.296e-07 |
| dot_general_multiple_f32_s32 | tenferro-cuda-eager | 64 | 15 | 3.652 | 0.0002162 |
| dot_general_multiple_f64_s32 | tenferro-cuda-eager | 64 | 15 | 3.687 | 1.632e-07 |
| dot_general_multiple_c32_s32 | tenferro-cuda-eager | 64 | 15 | 3.597 | 0.0002794 |
| dot_general_multiple_c64_s32 | tenferro-cuda-eager | 64 | 15 | 3.507 | 4.303e-07 |
| dot_general_batch_f32_s256 | tenferro-cuda-eager | 64 | 15 | 2.735 | 0.0008561 |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | 8 | 15 | 2.651 | 7.463e-07 |
| dot_general_batch_c32_s256 | tenferro-cuda-eager | 64 | 15 | 2.830 | 0.001749 |
| dot_general_batch_c64_s256 | tenferro-cuda-eager | 2 | 15 | 2.202 | 1.665e-06 |
| dot_general_multiple_f32_s256 | tenferro-cuda-eager | 64 | 15 | 3.195 | 0.002418 |
| dot_general_multiple_f64_s256 | tenferro-cuda-eager | 8 | 15 | 2.679 | 1.519e-06 |
| dot_general_multiple_c32_s256 | tenferro-cuda-eager | 64 | 15 | 2.960 | 0.0022 |
| dot_general_multiple_c64_s256 | tenferro-cuda-eager | 4 | 15 | 3.309 | 2.187e-06 |
| dot_conj_l0r0_c32_n1024 | tenferro-cuda-eager | 128 | 15 | 4.470 | 0.0004142 |
| dot_conj_l1r0_c32_n1024 | tenferro-cuda-eager | 32 | 15 | 3.899 | 0.0002905 |
| dot_conj_l0r1_c32_n1024 | tenferro-cuda-eager | 32 | 15 | 3.729 | 0.0002905 |
| dot_conj_l1r1_c32_n1024 | tenferro-cuda-eager | 32 | 15 | 3.520 | 0.0003083 |
| dot_conj_l0r0_c64_n1024 | tenferro-cuda-eager | 128 | 15 | 4.030 | 3.737e-07 |
| dot_conj_l1r0_c64_n1024 | tenferro-cuda-eager | 32 | 15 | 3.467 | 1.807e-07 |
| dot_conj_l0r1_c64_n1024 | tenferro-cuda-eager | 32 | 15 | 3.330 | 1.807e-07 |
| dot_conj_l1r1_c64_n1024 | tenferro-cuda-eager | 32 | 15 | 3.293 | 4.927e-07 |
| dot_conj_l0r0_c32_n262144 | tenferro-cuda-eager | 4 | 15 | 2.470 | 0.02908 |
| dot_conj_l1r0_c32_n262144 | tenferro-cuda-eager | 32 | 15 | 2.964 | 0.005619 |
| dot_conj_l0r1_c32_n262144 | tenferro-cuda-eager | 32 | 15 | 3.039 | 0.005619 |
| dot_conj_l1r1_c32_n262144 | tenferro-cuda-eager | 32 | 15 | 3.185 | 0.005822 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | 4 | 15 | 3.597 | 1.986e-05 |
| dot_conj_l1r0_c64_n262144 | tenferro-cuda-eager | 32 | 15 | 3.267 | 1.516e-05 |
| dot_conj_l0r1_c64_n262144 | tenferro-cuda-eager | 32 | 15 | 3.285 | 1.516e-05 |
| dot_conj_l1r1_c64_n262144 | tenferro-cuda-eager | 32 | 15 | 3.274 | 1.212e-05 |
| add_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.287 | 0 |
| add_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.367 | 0 |
| add_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.275 | 0 |
| add_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.402 | 0 |
| sub_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.709 | 0 |
| sub_into_f64_n1024 | pytorch-cuda | 256 | 15 | 1.867 | 0 |
| sub_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.703 | 0 |
| sub_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.731 | 0 |
| mul_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.363 | 0 |
| mul_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.345 | 0 |
| mul_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.340 | 0 |
| mul_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.315 | 0 |
| div_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.443 | 0 |
| div_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.480 | 0 |
| div_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.452 | 0 |
| div_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.472 | 0 |
| neg_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.054 | 0 |
| neg_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.067 | 0 |
| neg_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.044 | 0 |
| neg_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.072 | 0 |
| conj_into_c32_n1024 | pytorch-cuda | 512 | 15 | 3.436 | 0 |
| conj_into_c64_n1024 | pytorch-cuda | 512 | 15 | 3.507 | 0 |
| conj_read_into_c32_n1024 | pytorch-cuda | 512 | 15 | 3.644 | 0 |
| conj_read_into_c64_n1024 | pytorch-cuda | 256 | 15 | 1.763 | 0 |
| add_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.541 | 0 |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.489 | 0 |
| add_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.538 | 0 |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.483 | 0 |
| sub_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.540 | 0 |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| sub_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.540 | 0 |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.483 | 0 |
| mul_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.537 | 0 |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.485 | 0 |
| mul_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.539 | 0 |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.481 | 0 |
| div_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.560 | 0 |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.020 | 0 |
| div_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.554 | 0 |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.020 | 0 |
| neg_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.742 | 0 |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.691 | 0 |
| neg_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.741 | 0 |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.692 | 0 |
| conj_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.690 | 0 |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.660 | 0 |
| conj_read_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.694 | 0 |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.662 | 0 |
| tensordot_f32_n1024 | pytorch-cuda | 64 | 15 | 3.205 | 0.0008859 |
| tensordot_f64_n1024 | pytorch-cuda | 64 | 15 | 3.430 | 0 |
| tensordot_c32_n1024 | pytorch-cuda | 64 | 15 | 3.264 | 0.0005344 |
| tensordot_c64_n1024 | pytorch-cuda | 64 | 15 | 3.327 | 0 |
| vdot_read_f32_n1024 | pytorch-cuda | 128 | 15 | 1.947 | 0 |
| vdot_read_f64_n1024 | pytorch-cuda | 128 | 15 | 1.968 | 1.722e-07 |
| vdot_read_c32_n1024 | pytorch-cuda | 256 | 15 | 3.778 | 0 |
| vdot_read_c64_n1024 | pytorch-cuda | 256 | 15 | 3.930 | 3.455e-07 |
| norm_squared_read_f32_n1024 | pytorch-cuda | 128 | 15 | 1.917 | 0 |
| norm_squared_read_f64_n1024 | pytorch-cuda | 128 | 15 | 1.919 | 0 |
| tensordot_f32_n262144 | pytorch-cuda | 64 | 15 | 3.702 | 0.006191 |
| tensordot_f64_n262144 | pytorch-cuda | 2 | 15 | 3.267 | 6.714e-06 |
| tensordot_c32_n262144 | pytorch-cuda | 16 | 15 | 2.538 | 0.005717 |
| tensordot_c64_n262144 | pytorch-cuda | 1 | 15 | 5.280 | 4.401e-06 |
| vdot_read_f32_n262144 | pytorch-cuda | 128 | 15 | 1.993 | 0 |
| vdot_read_f64_n262144 | pytorch-cuda | 128 | 15 | 2.600 | 2.995e-06 |
| vdot_read_c32_n262144 | pytorch-cuda | 128 | 15 | 2.231 | 0 |
| vdot_read_c64_n262144 | pytorch-cuda | 64 | 15 | 2.169 | 8.762e-06 |
| norm_squared_read_f32_n262144 | pytorch-cuda | 256 | 15 | 3.886 | 0.0001786 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 128 | 15 | 1.924 | 0 |
| axpby_read_into_accum_f32_n1024 | pytorch-cuda | 512 | 15 | 3.950 | 0 |
| axpby_read_into_accum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.014 | 0 |
| axpby_read_into_accum_c32_n1024 | pytorch-cuda | 256 | 15 | 1.986 | 0 |
| axpby_read_into_accum_c64_n1024 | pytorch-cuda | 256 | 15 | 1.998 | 0 |
| axpby_read_into_accum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.550 | 0 |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.494 | 0 |
| axpby_read_into_accum_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.487 | 0 |
| axpby_read_into_accum_c64_n1048576 | pytorch-cuda | 16 | 15 | 2.457 | 0 |
| dot_general_read_into_f32_s16 | pytorch-cuda | 128 | 15 | 2.514 | 8.645e-05 |
| dot_general_read_into_accum_f32_s16 | pytorch-cuda | 128 | 15 | 3.017 | 7.907e-05 |
| dot_general_read_into_accum_cached_f32_s16 | pytorch-cuda | 128 | 15 | 3.026 | 7.907e-05 |
| dot_general_read_into_f64_s16 | pytorch-cuda | 128 | 15 | 2.649 | 0 |
| dot_general_read_into_accum_f64_s16 | pytorch-cuda | 128 | 15 | 3.268 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | pytorch-cuda | 128 | 15 | 3.252 | 0 |
| dot_general_read_into_c32_s16 | pytorch-cuda | 128 | 15 | 2.526 | 0.0001647 |
| dot_general_read_into_accum_c32_s16 | pytorch-cuda | 128 | 15 | 3.058 | 0.0001125 |
| dot_general_read_into_accum_cached_c32_s16 | pytorch-cuda | 128 | 15 | 3.079 | 0.0001125 |
| dot_general_read_into_c64_s16 | pytorch-cuda | 128 | 15 | 2.687 | 1.373e-07 |
| dot_general_read_into_accum_c64_s16 | pytorch-cuda | 128 | 15 | 3.382 | 7.37e-08 |
| dot_general_read_into_accum_cached_c64_s16 | pytorch-cuda | 128 | 15 | 3.239 | 7.37e-08 |
| dot_general_read_into_f32_s256 | pytorch-cuda | 128 | 15 | 2.551 | 0.0005542 |
| dot_general_read_into_accum_f32_s256 | pytorch-cuda | 128 | 15 | 3.231 | 0.0003617 |
| dot_general_read_into_accum_cached_f32_s256 | pytorch-cuda | 128 | 15 | 3.149 | 0.0003617 |
| dot_general_read_into_f64_s256 | pytorch-cuda | 8 | 15 | 1.992 | 0 |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 8 | 15 | 2.168 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 8 | 15 | 2.248 | 0 |
| dot_general_read_into_c32_s256 | pytorch-cuda | 128 | 15 | 3.449 | 0.0008017 |
| dot_general_read_into_accum_c32_s256 | pytorch-cuda | 128 | 15 | 3.552 | 0.000499 |
| dot_general_read_into_accum_cached_c32_s256 | pytorch-cuda | 128 | 15 | 3.558 | 0.000499 |
| dot_general_read_into_c64_s256 | pytorch-cuda | 4 | 15 | 3.899 | 1.175e-06 |
| dot_general_read_into_accum_c64_s256 | pytorch-cuda | 4 | 15 | 3.921 | 6.523e-07 |
| dot_general_read_into_accum_cached_c64_s256 | pytorch-cuda | 4 | 15 | 3.921 | 6.523e-07 |
| add_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.597 | 0 |
| add_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.543 | 0 |
| mul_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.676 | 0 |
| mul_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.761 | 0 |
| neg_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.395 | 0 |
| neg_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.494 | 0 |
| exp_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.345 | 0.0001272 |
| exp_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.421 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.376 | 9.107e-05 |
| tanh_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.356 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | pytorch-cuda | 128 | 15 | 1.851 | 0.0005233 |
| reduce_sum_strided_read_f64_n1024 | pytorch-cuda | 128 | 15 | 1.922 | 3.504e-07 |
| conj_strided_read_c32_n1024 | pytorch-cuda | 256 | 15 | 2.954 | 0 |
| conj_strided_read_c64_n1024 | pytorch-cuda | 256 | 15 | 3.033 | 0 |
| add_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.547 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.491 | 0 |
| mul_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.546 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| neg_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.751 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.699 | 0 |
| exp_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 0.0001272 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.899 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 9.135e-05 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.682 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.747 | 0.002693 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.367 | 5.935e-06 |
| conj_strided_read_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.697 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.669 | 0 |
| axpby_overwrite_f32_n1024 | pytorch-cuda | 256 | 15 | 3.090 | 0 |
| axpby_lerp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.325 | 0.0001189 |
| axpby_unit_add_f32_n1024 | pytorch-cuda | 512 | 15 | 3.930 | 0 |
| axpby_overwrite_f64_n1024 | pytorch-cuda | 256 | 15 | 2.894 | 0 |
| axpby_lerp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.243 | 1.161e-07 |
| axpby_unit_add_f64_n1024 | pytorch-cuda | 512 | 15 | 3.984 | 0 |
| axpby_overwrite_c32_n1024 | pytorch-cuda | 256 | 15 | 2.960 | 0 |
| axpby_lerp_c32_n1024 | pytorch-cuda | 512 | 15 | 3.824 | 0.0001189 |
| axpby_unit_add_c32_n1024 | pytorch-cuda | 256 | 15 | 2.001 | 0 |
| axpby_overwrite_c64_n1024 | pytorch-cuda | 256 | 15 | 3.006 | 0 |
| axpby_lerp_c64_n1024 | pytorch-cuda | 256 | 15 | 2.792 | 1.161e-07 |
| axpby_unit_add_c64_n1024 | pytorch-cuda | 256 | 15 | 2.017 | 0 |
| axpby_overwrite_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 0 |
| axpby_lerp_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.548 | 0.0001192 |
| axpby_unit_add_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.543 | 0 |
| axpby_overwrite_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.702 | 0 |
| axpby_lerp_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.498 | 1.161e-07 |
| axpby_unit_add_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.490 | 0 |
| axpby_overwrite_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.702 | 0 |
| axpby_lerp_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.496 | 0.0001191 |
| axpby_unit_add_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| axpby_overwrite_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.672 | 0 |
| axpby_lerp_c64_n1048576 | pytorch-cuda | 8 | 15 | 2.735 | 1.159e-07 |
| axpby_unit_add_c64_n1048576 | pytorch-cuda | 16 | 15 | 2.457 | 0 |
| dot_general_batch_f32_s32 | pytorch-cuda | 64 | 15 | 2.705 | 0.0002227 |
| dot_general_batch_f64_s32 | pytorch-cuda | 64 | 15 | 2.732 | 1.326e-07 |
| dot_general_batch_c32_s32 | pytorch-cuda | 64 | 15 | 2.569 | 0.0003092 |
| dot_general_batch_c64_s32 | pytorch-cuda | 64 | 15 | 2.659 | 2.791e-07 |
| dot_general_multiple_f32_s32 | pytorch-cuda | 32 | 15 | 2.416 | 0.000269 |
| dot_general_multiple_f64_s32 | pytorch-cuda | 32 | 15 | 2.530 | 2.466e-07 |
| dot_general_multiple_c32_s32 | pytorch-cuda | 32 | 15 | 2.409 | 0.0004058 |
| dot_general_multiple_c64_s32 | pytorch-cuda | 32 | 15 | 2.465 | 4.129e-07 |
| dot_general_batch_f32_s256 | pytorch-cuda | 64 | 15 | 2.681 | 0.0008561 |
| dot_general_batch_f64_s256 | pytorch-cuda | 8 | 15 | 3.360 | 7.463e-07 |
| dot_general_batch_c32_s256 | pytorch-cuda | 64 | 15 | 3.469 | 0.001286 |
| dot_general_batch_c64_s256 | pytorch-cuda | 2 | 15 | 2.667 | 8.893e-07 |
| dot_general_multiple_f32_s256 | pytorch-cuda | 32 | 15 | 2.522 | 0.00133 |
| dot_general_multiple_f64_s256 | pytorch-cuda | 8 | 15 | 2.715 | 2.584e-06 |
| dot_general_multiple_c32_s256 | pytorch-cuda | 32 | 15 | 2.670 | 0.001059 |
| dot_general_multiple_c64_s256 | pytorch-cuda | 2 | 15 | 2.029 | 2.776e-06 |
| dot_conj_l0r0_c32_n1024 | pytorch-cuda | 256 | 15 | 3.748 | 0.0001028 |
| dot_conj_l1r0_c32_n1024 | pytorch-cuda | 256 | 15 | 3.812 | 0.0004339 |
| dot_conj_l0r1_c32_n1024 | pytorch-cuda | 256 | 15 | 3.870 | 0.0004457 |
| dot_conj_l1r1_c32_n1024 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n1024 | pytorch-cuda | 256 | 15 | 3.785 | 4.333e-07 |
| dot_conj_l1r0_c64_n1024 | pytorch-cuda | 256 | 15 | 3.870 | 2.122e-07 |
| dot_conj_l0r1_c64_n1024 | pytorch-cuda | 128 | 15 | 1.958 | 2.847e-07 |
| dot_conj_l1r1_c64_n1024 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c32_n262144 | pytorch-cuda | 128 | 15 | 2.256 | 0.003079 |
| dot_conj_l1r0_c32_n262144 | pytorch-cuda | 128 | 15 | 2.255 | 0.006155 |
| dot_conj_l0r1_c32_n262144 | pytorch-cuda | 128 | 15 | 2.266 | 0.006499 |
| dot_conj_l1r1_c32_n262144 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | 64 | 15 | 2.168 | 1.455e-05 |
| dot_conj_l1r0_c64_n262144 | pytorch-cuda | 64 | 15 | 2.172 | 2.05e-05 |
| dot_conj_l0r1_c64_n262144 | pytorch-cuda | 64 | 15 | 2.163 | 1.888e-05 |
| dot_conj_l1r1_c64_n262144 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
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
| tensordot_f32_n1024 | jax-cuda | 16 | 15 | 2.873 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda | 16 | 15 | 2.448 | 0 |
| tensordot_c32_n1024 | jax-cuda | 16 | 15 | 2.695 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda | 16 | 15 | 2.815 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda | 32 | 15 | 2.094 | 0 |
| vdot_read_f64_n1024 | jax-cuda | 32 | 15 | 2.308 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda | 32 | 15 | 3.219 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda | 32 | 15 | 2.926 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda | 32 | 15 | 2.329 | 0 |
| norm_squared_read_f64_n1024 | jax-cuda | 32 | 15 | 2.119 | 0 |
| tensordot_f32_n262144 | jax-cuda | 16 | 15 | 2.568 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda | 2 | 15 | 3.408 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda | 8 | 15 | 2.532 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda | 1 | 15 | 5.603 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda | 16 | 15 | 2.667 | 0 |
| vdot_read_f64_n262144 | jax-cuda | 16 | 15 | 2.543 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda | 16 | 15 | 2.615 | 0 |
| vdot_read_c64_n262144 | jax-cuda | 16 | 15 | 2.656 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda | 16 | 15 | 2.173 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda | 16 | 15 | 1.976 | 1.664e-07 |
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
| add_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.027 | 0 |
| add_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.117 | 0 |
| mul_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.198 | 0 |
| mul_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.145 | 0 |
| neg_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.030 | 0 |
| neg_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 3.305 | 0 |
| exp_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 3.015 | 0.0001272 |
| exp_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 3.147 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 3.022 | 9.325e-05 |
| tanh_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 3.151 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 3.039 | 0.0002267 |
| reduce_sum_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 2.966 | 4.202e-07 |
| conj_strided_read_c32_n1024 | jax-cuda | 32 | 15 | 3.014 | 0 |
| conj_strided_read_c64_n1024 | jax-cuda | 16 | 15 | 2.397 | 0 |
| add_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.063 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 1.896 | 0 |
| mul_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.260 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.028 | 0 |
| neg_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 2.915 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.120 | 0 |
| exp_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 2.765 | 0.0001272 |
| exp_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.270 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 2.798 | 9.508e-05 |
| tanh_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.037 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.420 | 0.003094 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.494 | 7.159e-06 |
| conj_strided_read_c32_n1048576 | jax-cuda | 32 | 15 | 3.092 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 1.871 | 0 |
| axpby_overwrite_f32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1024 | jax-cuda | 8 | 15 | 2.686 | 0 |
| axpby_overwrite_f64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1024 | jax-cuda | 8 | 15 | 2.662 | 0 |
| axpby_overwrite_c32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1024 | jax-cuda | 8 | 15 | 2.629 | 0 |
| axpby_overwrite_c64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1024 | jax-cuda | 8 | 15 | 2.815 | 0 |
| axpby_overwrite_f32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1048576 | jax-cuda | 8 | 15 | 2.903 | 0 |
| axpby_overwrite_f64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1048576 | jax-cuda | 8 | 15 | 2.931 | 0 |
| axpby_overwrite_c32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1048576 | jax-cuda | 8 | 15 | 2.948 | 0 |
| axpby_overwrite_c64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1048576 | jax-cuda | 8 | 15 | 2.953 | 0 |
| dot_general_batch_f32_s32 | jax-cuda | 32 | 15 | 3.610 | 0.0002227 |
| dot_general_batch_f64_s32 | jax-cuda | 32 | 15 | 3.685 | 1.326e-07 |
| dot_general_batch_c32_s32 | jax-cuda | 16 | 15 | 2.278 | 0.0003092 |
| dot_general_batch_c64_s32 | jax-cuda | 16 | 15 | 2.537 | 2.524e-07 |
| dot_general_multiple_f32_s32 | jax-cuda | 16 | 15 | 2.609 | 0.000269 |
| dot_general_multiple_f64_s32 | jax-cuda | 16 | 15 | 2.040 | 2.466e-07 |
| dot_general_multiple_c32_s32 | jax-cuda | 16 | 15 | 2.223 | 0.0004771 |
| dot_general_multiple_c64_s32 | jax-cuda | 16 | 15 | 2.686 | 6.063e-07 |
| dot_general_batch_f32_s256 | jax-cuda | 16 | 15 | 1.893 | 0.0008561 |
| dot_general_batch_f64_s256 | jax-cuda | 8 | 15 | 2.588 | 7.463e-07 |
| dot_general_batch_c32_s256 | jax-cuda | 16 | 15 | 3.180 | 0.001359 |
| dot_general_batch_c64_s256 | jax-cuda | 2 | 15 | 2.889 | 8.493e-07 |
| dot_general_multiple_f32_s256 | jax-cuda | 16 | 15 | 2.964 | 0.0025 |
| dot_general_multiple_f64_s256 | jax-cuda | 16 | 15 | 3.496 | 2.584e-06 |
| dot_general_multiple_c32_s256 | jax-cuda | 8 | 15 | 2.613 | 0.002465 |
| dot_general_multiple_c64_s256 | jax-cuda | 2 | 15 | 2.283 | 1.805e-06 |
| dot_conj_l0r0_c32_n1024 | jax-cuda | 16 | 15 | 2.550 | 0.0002298 |
| dot_conj_l1r0_c32_n1024 | jax-cuda | 16 | 15 | 2.276 | 0.0002522 |
| dot_conj_l0r1_c32_n1024 | jax-cuda | 32 | 15 | 3.051 | 0.0002522 |
| dot_conj_l1r1_c32_n1024 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n1024 | jax-cuda | 32 | 15 | 3.669 | 3.644e-07 |
| dot_conj_l1r0_c64_n1024 | jax-cuda | 32 | 15 | 2.805 | 2.767e-07 |
| dot_conj_l0r1_c64_n1024 | jax-cuda | 32 | 15 | 2.401 | 2.767e-07 |
| dot_conj_l1r1_c64_n1024 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c32_n262144 | jax-cuda | 32 | 15 | 3.217 | 0.01276 |
| dot_conj_l1r0_c32_n262144 | jax-cuda | 32 | 15 | 2.586 | 0.0002732 |
| dot_conj_l0r1_c32_n262144 | jax-cuda | 32 | 15 | 2.330 | 0.0002732 |
| dot_conj_l1r1_c32_n262144 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda | 32 | 15 | 3.636 | 1.221e-05 |
| dot_conj_l1r0_c64_n262144 | jax-cuda | 32 | 15 | 3.171 | 1.586e-05 |
| dot_conj_l0r1_c64_n262144 | jax-cuda | 32 | 15 | 2.986 | 1.586e-05 |
| dot_conj_l1r1_c64_n262144 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
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
| tensordot_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.617 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.306 | 0 |
| tensordot_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.784 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.975 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.818 | 0 |
| vdot_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.887 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.184 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.511 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.494 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.521 | 0 |
| tensordot_f32_n262144 | jax-cuda-jit | 32 | 15 | 2.715 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda-jit | 2 | 15 | 3.177 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.570 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda-jit | 1 | 15 | 5.360 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.330 | 0 |
| vdot_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.713 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.757 | 0 |
| vdot_read_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.761 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda-jit | 32 | 15 | 2.858 | 0.0001786 |
| norm_squared_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.373 | 1.664e-07 |
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
| add_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.092 | 0 |
| add_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.449 | 0 |
| mul_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.194 | 0 |
| mul_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.004 | 0 |
| neg_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.849 | 0 |
| neg_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.100 | 0 |
| exp_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.061 | 0.0001272 |
| exp_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.117 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.126 | 9.325e-05 |
| tanh_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.124 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.297 | 0.0002267 |
| reduce_sum_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.772 | 4.202e-07 |
| conj_strided_read_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.191 | 0 |
| conj_strided_read_c64_n1024 | jax-cuda-jit | 16 | 15 | 1.972 | 0 |
| add_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.223 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.927 | 0 |
| mul_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.000 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.853 | 0 |
| neg_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.916 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.978 | 0 |
| exp_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.916 | 0.0001272 |
| exp_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.270 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.974 | 9.508e-05 |
| tanh_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.020 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.433 | 0.003094 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.328 | 7.159e-06 |
| conj_strided_read_c32_n1048576 | jax-cuda-jit | 32 | 15 | 3.145 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.851 | 0 |
| axpby_overwrite_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1024 | jax-cuda-jit | 8 | 15 | 1.906 | 0 |
| axpby_overwrite_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1024 | jax-cuda-jit | 8 | 15 | 1.941 | 0 |
| axpby_overwrite_c32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1024 | jax-cuda-jit | 8 | 15 | 1.903 | 0 |
| axpby_overwrite_c64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1024 | jax-cuda-jit | 8 | 15 | 1.963 | 0 |
| axpby_overwrite_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1048576 | jax-cuda-jit | 8 | 15 | 2.012 | 0 |
| axpby_overwrite_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.017 | 0 |
| axpby_overwrite_c32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1048576 | jax-cuda-jit | 8 | 15 | 2.011 | 0 |
| axpby_overwrite_c64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1048576 | jax-cuda-jit | 8 | 15 | 1.980 | 0 |
| dot_general_batch_f32_s32 | jax-cuda-jit | 32 | 15 | 2.060 | 0.0002227 |
| dot_general_batch_f64_s32 | jax-cuda-jit | 32 | 15 | 2.091 | 1.326e-07 |
| dot_general_batch_c32_s32 | jax-cuda-jit | 32 | 15 | 2.962 | 0.0004354 |
| dot_general_batch_c64_s32 | jax-cuda-jit | 32 | 15 | 3.052 | 2.524e-07 |
| dot_general_multiple_f32_s32 | jax-cuda-jit | 32 | 15 | 2.444 | 0.000269 |
| dot_general_multiple_f64_s32 | jax-cuda-jit | 32 | 15 | 2.280 | 2.466e-07 |
| dot_general_multiple_c32_s32 | jax-cuda-jit | 32 | 15 | 3.071 | 0.0004771 |
| dot_general_multiple_c64_s32 | jax-cuda-jit | 32 | 15 | 3.366 | 6.063e-07 |
| dot_general_batch_f32_s256 | jax-cuda-jit | 32 | 15 | 2.208 | 0.0008561 |
| dot_general_batch_f64_s256 | jax-cuda-jit | 8 | 15 | 2.557 | 7.463e-07 |
| dot_general_batch_c32_s256 | jax-cuda-jit | 16 | 15 | 2.410 | 0.001359 |
| dot_general_batch_c64_s256 | jax-cuda-jit | 2 | 15 | 2.673 | 8.493e-07 |
| dot_general_multiple_f32_s256 | jax-cuda-jit | 32 | 15 | 3.006 | 0.0025 |
| dot_general_multiple_f64_s256 | jax-cuda-jit | 16 | 15 | 3.466 | 2.584e-06 |
| dot_general_multiple_c32_s256 | jax-cuda-jit | 16 | 15 | 3.213 | 0.002465 |
| dot_general_multiple_c64_s256 | jax-cuda-jit | 2 | 15 | 2.119 | 1.805e-06 |
| dot_conj_l0r0_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.604 | 0.0002298 |
| dot_conj_l1r0_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.494 | 0.0002522 |
| dot_conj_l0r1_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.493 | 0.0002522 |
| dot_conj_l1r1_c32_n1024 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.748 | 3.644e-07 |
| dot_conj_l1r0_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.647 | 2.767e-07 |
| dot_conj_l0r1_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.766 | 2.767e-07 |
| dot_conj_l1r1_c64_n1024 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.728 | 0.01276 |
| dot_conj_l1r0_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.761 | 0.0002732 |
| dot_conj_l0r1_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.801 | 0.0002732 |
| dot_conj_l1r1_c32_n262144 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.859 | 1.221e-05 |
| dot_conj_l1r0_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.838 | 1.586e-05 |
| dot_conj_l0r1_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.838 | 1.586e-05 |
| dot_conj_l1r1_c64_n262144 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: benchmarks/gpu/reuse_contraction.yaml
timestamp: '2026-10-13T18:58:27Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 8ff5abd8cfc00509617fef3711e8a63ffc49f815
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
  expected_cases: 184
  selected_cases: 184
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 42d1d21f59c8a48814609b5730b976228f5509cb419fa387d4ccd57ff7f71d7a
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/multi.rs: 1b477fef78260cf28aa31e515d57ab411f6241b91ffb30d625939538fdbc6bab
    src/bin/benchmark_gpu_public_api/read_layout.rs: bd7ce4a42576d1937a7048b7b002cac4838a7cd3c53628bc9aa6aebb0c5c1128
    src/bin/benchmark_gpu_public_api/structural.rs: 10e3645d2027fb543f7d8134a00390dc985904e29743f65f4777c3fe533614b3
    src/bin/benchmark_gpu_public_api/linalg.rs: 5f5c4afc161f3d6e267861511d0d5871be9192d9968c95b59a73c9dc5a64e22b
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: 62173a8bb61beaa98f93468de4db2930ff1f95520f91e7aa3d00f43da3ab6b15
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 720c8ebbee03788b000ef3f040eebee865c632d88a6c583f375cf1286862f882
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: c5348ce4a026c5c09f5d1af0d1002abfe27376662c9d89a93a91d9c0634429cb
    scripts/run_gpu_public_api.sh: dd5e4cc47e85fcb5afdc56d460136b2a399a6a7d6a62005de8ba94afb0119ecc
    data/results/nvidia-gpu/gpu/reuse_contraction/20261013_185827/fixtures/manifest.json: 4f97bec592af21f8afe5e3319a926477e8d6b625c9cd38435d6ff9f3a5fe6dd0
    target/release/benchmark_gpu_public_api: 8f7d00111d9d7beb1de2adee6e7825475974ba1a6a10de4f7170a902d5f3d5c9
  reference_numpy_version: 2.4.2
```
