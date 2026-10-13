# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **748**. Unsupported cells: **492**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261013_191556/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

Unit-coefficient AXPBY uses native JAX Ref.addupdate with caller-owned fixed buffers. Its synchronization includes one GPU scalar completion token per destination and block_until_ready; tokens are retained through clock stop and included in the logical retention bound. Helper compilation and seed restoration are untimed. Full Ref snapshots are used only for untimed inspection. This scalar-token synchronization cost is included in the reported latency; it performs no full output copy or D2H transfer inside timing.

Generalized batch dot preserves native output order: tenferro [m,n,b], PyTorch/JAX [b,m,n]. Input permutation views and dimension-number configurations are prepared before timing. Output axes are normalized only on downloaded CPU arrays for verification; no timed output transpose or copy is added.

Prepared einsum uses the public ConcreteEinsumPlan for ik,kj->ij, built once outside execution and sampling. Native matrix multiplication/reuse/accumulation references match the selected mathematical and output-ownership contracts; they do not assert an equivalent prepared-plan interface. Transposed input views/values, read/write handles and accumulation seed restoration are untimed.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | 7.036 (0.800) | unsupported | 6.928 (0.078) | unsupported | unsupported |
| add_into_f64_n1024 | 5.166 (0.259) | unsupported | 7.469 (0.359) | unsupported | unsupported |
| add_read_into_f32_n1024 | 5.162 (0.152) | unsupported | 6.775 (0.126) | unsupported | unsupported |
| add_read_into_f64_n1024 | 5.158 (0.152) | unsupported | 7.077 (0.044) | unsupported | unsupported |
| sub_into_f32_n1024 | 5.469 (0.263) | unsupported | 7.679 (0.489) | unsupported | unsupported |
| sub_into_f64_n1024 | 5.213 (0.115) | unsupported | 7.516 (0.374) | unsupported | unsupported |
| sub_read_into_f32_n1024 | 5.771 (0.144) | unsupported | 7.951 (0.275) | unsupported | unsupported |
| sub_read_into_f64_n1024 | 5.294 (0.089) | unsupported | 8.274 (0.306) | unsupported | unsupported |
| mul_into_f32_n1024 | 5.675 (0.110) | unsupported | 7.086 (0.108) | unsupported | unsupported |
| mul_into_f64_n1024 | 5.607 (0.192) | unsupported | 7.098 (0.101) | unsupported | unsupported |
| mul_read_into_f32_n1024 | 5.583 (0.195) | unsupported | 7.017 (0.191) | unsupported | unsupported |
| mul_read_into_f64_n1024 | 5.210 (0.157) | unsupported | 6.956 (0.035) | unsupported | unsupported |
| div_into_f32_n1024 | 53.065 (3.517) | unsupported | 6.857 (0.110) | unsupported | unsupported |
| div_into_f64_n1024 | 41.906 (0.848) | unsupported | 7.120 (0.460) | unsupported | unsupported |
| div_read_into_f32_n1024 | 41.997 (0.850) | unsupported | 6.901 (0.056) | unsupported | unsupported |
| div_read_into_f64_n1024 | 41.219 (0.613) | unsupported | 6.920 (0.061) | unsupported | unsupported |
| neg_into_f32_n1024 | 4.841 (0.204) | unsupported | 6.686 (0.099) | unsupported | unsupported |
| neg_into_f64_n1024 | 4.676 (0.236) | unsupported | 6.913 (0.425) | unsupported | unsupported |
| neg_read_into_f32_n1024 | 4.904 (0.164) | unsupported | 6.545 (0.196) | unsupported | unsupported |
| neg_read_into_f64_n1024 | 4.795 (0.138) | unsupported | 6.780 (0.136) | unsupported | unsupported |
| conj_into_c32_n1024 | 40.292 (0.504) | unsupported | 7.056 (0.070) | unsupported | unsupported |
| conj_into_c64_n1024 | 192.652 (12.446) | unsupported | 7.001 (0.167) | unsupported | unsupported |
| conj_read_into_c32_n1024 | 38.569 (0.610) | unsupported | 7.333 (0.544) | unsupported | unsupported |
| conj_read_into_c64_n1024 | 176.198 (9.832) | unsupported | 6.927 (0.119) | unsupported | unsupported |
| add_into_f32_n1048576 | 43.490 (0.322) | unsupported | 39.790 (0.020) | unsupported | unsupported |
| add_into_f64_n1048576 | 84.460 (0.956) | unsupported | 77.731 (0.058) | unsupported | unsupported |
| add_read_into_f32_n1048576 | 42.867 (0.273) | unsupported | 39.701 (0.040) | unsupported | unsupported |
| add_read_into_f64_n1048576 | 82.645 (0.477) | unsupported | 77.610 (0.053) | unsupported | unsupported |
| sub_into_f32_n1048576 | 42.838 (0.198) | unsupported | 39.708 (0.041) | unsupported | unsupported |
| sub_into_f64_n1048576 | 82.346 (0.377) | unsupported | 77.650 (0.289) | unsupported | unsupported |
| sub_read_into_f32_n1048576 | 42.715 (0.279) | unsupported | 39.695 (0.080) | unsupported | unsupported |
| sub_read_into_f64_n1048576 | 82.452 (0.525) | unsupported | 77.613 (0.035) | unsupported | unsupported |
| mul_into_f32_n1048576 | 42.716 (0.330) | unsupported | 39.860 (0.170) | unsupported | unsupported |
| mul_into_f64_n1048576 | 82.649 (0.655) | unsupported | 77.571 (0.063) | unsupported | unsupported |
| mul_read_into_f32_n1048576 | 42.798 (0.248) | unsupported | 39.652 (0.072) | unsupported | unsupported |
| mul_read_into_f64_n1048576 | 82.445 (0.332) | unsupported | 77.544 (0.095) | unsupported | unsupported |
| div_into_f32_n1048576 | 70.106 (0.195) | unsupported | 39.893 (0.031) | unsupported | unsupported |
| div_into_f64_n1048576 | 151.383 (0.176) | unsupported | 94.473 (0.182) | unsupported | unsupported |
| div_read_into_f32_n1048576 | 70.109 (0.108) | unsupported | 39.905 (0.098) | unsupported | unsupported |
| div_read_into_f64_n1048576 | 151.328 (0.554) | unsupported | 94.471 (0.091) | unsupported | unsupported |
| neg_into_f32_n1048576 | 29.554 (0.140) | unsupported | 27.244 (0.093) | unsupported | unsupported |
| neg_into_f64_n1048576 | 57.225 (0.757) | unsupported | 52.874 (0.061) | unsupported | unsupported |
| neg_read_into_f32_n1048576 | 29.602 (0.225) | unsupported | 27.232 (0.033) | unsupported | unsupported |
| neg_read_into_f64_n1048576 | 57.104 (0.180) | unsupported | 52.865 (0.070) | unsupported | unsupported |
| conj_into_c32_n1048576 | 107.856 (0.102) | unsupported | 52.825 (0.046) | unsupported | unsupported |
| conj_into_c64_n1048576 | 209.031 (0.320) | unsupported | 103.648 (0.127) | unsupported | unsupported |
| conj_read_into_c32_n1048576 | 107.748 (0.182) | unsupported | 52.817 (0.039) | unsupported | unsupported |
| conj_read_into_c64_n1048576 | 208.836 (0.279) | unsupported | 103.663 (0.133) | unsupported | unsupported |
| tensordot_f32_n1024 | 27.250 (1.370) | 38.206 (0.829) | 53.533 (0.587) | 169.968 (3.009) | 86.456 (2.196) |
| tensordot_f64_n1024 | 30.270 (2.276) | 41.154 (0.434) | 54.605 (0.494) | 172.183 (10.405) | 68.115 (4.634) |
| tensordot_c32_n1024 | 28.167 (0.217) | 36.592 (0.949) | 51.598 (0.600) | 187.768 (9.552) | 79.217 (2.131) |
| tensordot_c64_n1024 | 116.525 (9.075) | 40.918 (0.858) | 55.386 (2.326) | 212.479 (7.466) | 85.499 (0.839) |
| vdot_read_f32_n1024 | 145.338 (6.259) | unsupported | 15.956 (1.018) | 69.887 (3.100) | 126.759 (1.904) |
| vdot_read_f64_n1024 | 159.740 (3.426) | unsupported | 15.853 (0.374) | 86.663 (3.782) | 155.491 (5.432) |
| vdot_read_c32_n1024 | 160.488 (2.042) | unsupported | 15.571 (0.407) | 92.834 (3.180) | 158.273 (12.762) |
| vdot_read_c64_n1024 | 181.917 (2.749) | unsupported | 15.342 (0.097) | 127.722 (1.422) | 171.114 (22.028) |
| norm_squared_read_f32_n1024 | 169.191 (2.468) | unsupported | 15.883 (0.473) | 128.018 (8.764) | 129.232 (8.535) |
| norm_squared_read_f64_n1024 | 158.760 (7.938) | unsupported | 16.710 (0.649) | 90.532 (9.470) | 155.662 (4.986) |
| tensordot_f32_n262144 | 72.068 (5.557) | 46.246 (1.892) | 66.933 (3.049) | 204.190 (2.744) | 84.597 (2.509) |
| tensordot_f64_n262144 | 1619.460 (2.675) | 1620.424 (15.721) | 1636.507 (1.846) | 1666.559 (25.397) | 1600.375 (18.126) |
| tensordot_c32_n262144 | 141.270 (2.679) | 141.667 (0.282) | 158.891 (2.619) | 369.471 (32.538) | 161.072 (2.183) |
| tensordot_c64_n262144 | 5031.248 (2.751) | 5063.196 (7.170) | 5288.922 (1.696) | 5865.198 (20.812) | 5350.316 (29.671) |
| vdot_read_f32_n262144 | 170.704 (10.644) | unsupported | 15.898 (0.185) | 195.868 (8.320) | 167.186 (7.872) |
| vdot_read_f64_n262144 | 188.747 (6.590) | unsupported | 20.286 (0.031) | 182.515 (1.900) | 162.445 (8.891) |
| vdot_read_c32_n262144 | 186.408 (4.507) | unsupported | 17.379 (0.038) | 193.911 (24.277) | 165.515 (25.634) |
| vdot_read_c64_n262144 | 184.485 (3.070) | unsupported | 33.867 (0.027) | 203.821 (26.318) | 160.566 (20.234) |
| norm_squared_read_f32_n262144 | 182.219 (3.124) | unsupported | 15.827 (0.084) | 138.561 (20.346) | 156.042 (9.994) |
| norm_squared_read_f64_n262144 | 183.050 (5.150) | unsupported | 15.681 (0.266) | 113.894 (4.147) | 149.597 (3.289) |
| axpby_read_into_accum_f32_n1024 | 10.798 (2.674) | unsupported | 8.055 (0.269) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1024 | 10.004 (0.287) | unsupported | 8.028 (0.086) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1024 | 10.558 (0.180) | unsupported | 8.195 (0.146) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1024 | 11.140 (0.365) | unsupported | 8.059 (0.055) | unsupported | unsupported |
| axpby_read_into_accum_f32_n1048576 | 273.390 (0.310) | unsupported | 39.727 (0.022) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1048576 | 534.196 (0.509) | unsupported | 77.724 (0.025) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1048576 | 168.810 (0.196) | unsupported | 77.717 (0.154) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1048576 | 2145.223 (1.404) | unsupported | 153.571 (0.058) | unsupported | unsupported |
| dot_general_read_into_f32_s16 | 22.403 (0.500) | unsupported | 21.774 (0.171) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s16 | 23.446 (0.948) | unsupported | 25.528 (0.353) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s16 | 22.275 (0.820) | unsupported | 27.004 (1.632) | unsupported | unsupported |
| dot_general_read_into_f64_s16 | 25.793 (1.344) | unsupported | 23.737 (0.400) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s16 | 25.236 (1.007) | unsupported | 27.987 (0.583) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s16 | 25.407 (1.325) | unsupported | 28.011 (0.266) | unsupported | unsupported |
| dot_general_read_into_c32_s16 | 22.619 (1.165) | unsupported | 24.935 (1.194) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s16 | 22.055 (0.549) | unsupported | 27.889 (1.276) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s16 | 22.170 (0.954) | unsupported | 27.838 (0.698) | unsupported | unsupported |
| dot_general_read_into_c64_s16 | 24.343 (0.779) | unsupported | 23.654 (0.402) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s16 | 23.890 (1.496) | unsupported | 27.794 (0.598) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s16 | 25.186 (0.803) | unsupported | 27.635 (0.242) | unsupported | unsupported |
| dot_general_read_into_f32_s256 | 21.002 (0.867) | unsupported | 22.806 (0.434) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s256 | 23.557 (0.716) | unsupported | 26.915 (0.541) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s256 | 23.505 (0.637) | unsupported | 27.365 (1.847) | unsupported | unsupported |
| dot_general_read_into_f64_s256 | 263.387 (6.177) | unsupported | 267.255 (12.444) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s256 | 298.164 (19.840) | unsupported | 301.781 (15.049) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s256 | 308.262 (15.041) | unsupported | 311.747 (5.401) | unsupported | unsupported |
| dot_general_read_into_c32_s256 | 28.327 (0.314) | unsupported | 27.011 (0.200) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s256 | 30.145 (0.217) | unsupported | 30.481 (0.394) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s256 | 29.827 (0.050) | unsupported | 30.182 (0.321) | unsupported | unsupported |
| dot_general_read_into_c64_s256 | 684.326 (3.119) | unsupported | 975.562 (0.245) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s256 | 684.655 (0.365) | unsupported | 982.370 (1.075) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s256 | 684.359 (0.552) | unsupported | 981.890 (0.578) | unsupported | unsupported |
| add_strided_read_f32_n1024 | 283.767 (9.115) | unsupported | 9.913 (0.064) | 155.476 (28.805) | 130.256 (21.982) |
| add_strided_read_f64_n1024 | 250.844 (12.843) | unsupported | 9.990 (0.197) | 191.402 (35.922) | 154.360 (23.707) |
| mul_strided_read_f32_n1024 | 334.414 (3.728) | unsupported | 9.984 (0.160) | 159.113 (27.264) | 130.519 (21.865) |
| mul_strided_read_f64_n1024 | 308.759 (4.604) | unsupported | 9.774 (0.158) | 146.884 (14.511) | 118.960 (15.527) |
| neg_strided_read_f32_n1024 | 195.937 (3.585) | unsupported | 9.588 (0.454) | 128.888 (9.184) | 97.564 (16.328) |
| neg_strided_read_f64_n1024 | 168.879 (2.941) | unsupported | 10.023 (0.442) | 129.372 (8.824) | 92.485 (9.100) |
| exp_strided_read_f32_n1024 | 196.128 (4.548) | unsupported | 8.927 (0.086) | 126.398 (6.346) | 94.449 (12.006) |
| exp_strided_read_f64_n1024 | 172.722 (23.907) | unsupported | 9.019 (0.059) | 141.308 (20.712) | 95.455 (13.258) |
| tanh_strided_read_f32_n1024 | 201.791 (5.638) | unsupported | 9.576 (0.111) | 142.883 (16.200) | 90.482 (9.789) |
| tanh_strided_read_f64_n1024 | 170.765 (5.852) | unsupported | 9.222 (0.108) | 155.156 (23.150) | 90.550 (18.103) |
| reduce_sum_strided_read_f32_n1024 | 334.412 (4.586) | unsupported | 15.511 (0.200) | 188.191 (24.446) | 80.855 (1.067) |
| reduce_sum_strided_read_f64_n1024 | 298.310 (10.316) | unsupported | 15.241 (0.365) | 172.340 (5.415) | 147.674 (12.894) |
| conj_strided_read_c32_n1024 | 163.475 (2.835) | unsupported | 11.855 (0.350) | 162.746 (29.227) | 155.384 (29.099) |
| conj_strided_read_c64_n1024 | 238.401 (4.569) | unsupported | 12.551 (0.907) | 167.632 (27.136) | 143.752 (25.916) |
| add_strided_read_f32_n1048576 | 98.264 (0.528) | unsupported | 39.925 (0.091) | 162.242 (10.580) | 99.768 (12.535) |
| add_strided_read_f64_n1048576 | 188.500 (0.169) | unsupported | 77.784 (0.048) | 151.147 (3.707) | 91.006 (4.751) |
| mul_strided_read_f32_n1048576 | 97.896 (0.238) | unsupported | 39.780 (0.027) | 134.924 (1.812) | 92.223 (0.514) |
| mul_strided_read_f64_n1048576 | 188.152 (0.177) | unsupported | 77.736 (0.031) | 138.055 (4.864) | 87.907 (0.904) |
| neg_strided_read_f32_n1048576 | 56.082 (0.093) | unsupported | 27.359 (0.048) | 130.822 (2.463) | 81.927 (1.571) |
| neg_strided_read_f64_n1048576 | 108.142 (0.106) | unsupported | 53.073 (0.028) | 133.826 (3.318) | 91.871 (0.625) |
| exp_strided_read_f32_n1048576 | 56.193 (0.059) | unsupported | 27.484 (0.126) | 104.606 (9.935) | 90.088 (0.422) |
| exp_strided_read_f64_n1048576 | 233.685 (0.350) | unsupported | 181.122 (0.070) | 198.685 (1.419) | 205.011 (1.889) |
| tanh_strided_read_f32_n1048576 | 56.279 (0.097) | unsupported | 27.459 (0.023) | 95.251 (11.296) | 88.922 (0.549) |
| tanh_strided_read_f64_n1048576 | 220.560 (0.305) | unsupported | 167.810 (0.129) | 184.874 (1.060) | 190.417 (2.006) |
| reduce_sum_strided_read_f32_n1048576 | 180.875 (6.567) | unsupported | 22.780 (0.140) | 138.034 (1.593) | 87.350 (1.622) |
| reduce_sum_strided_read_f64_n1048576 | 92.878 (0.370) | unsupported | 36.996 (0.042) | 204.022 (18.482) | 144.754 (0.998) |
| conj_strided_read_c32_n1048576 | 114.380 (0.134) | unsupported | 53.064 (0.076) | 133.156 (4.993) | 93.714 (5.189) |
| conj_strided_read_c64_n1048576 | 214.078 (0.225) | unsupported | 104.147 (0.075) | 131.203 (2.142) | 116.105 (0.536) |
| axpby_overwrite_f32_n1024 | 10.984 (0.267) | unsupported | 12.460 (0.205) | unsupported | unsupported |
| axpby_lerp_f32_n1024 | 10.666 (0.322) | unsupported | 9.799 (0.119) | unsupported | unsupported |
| axpby_unit_add_f32_n1024 | 11.092 (0.667) | unsupported | 8.045 (0.099) | 351.798 (51.021) | 241.406 (35.892) |
| axpby_overwrite_f64_n1024 | 10.370 (0.092) | unsupported | 12.434 (0.464) | unsupported | unsupported |
| axpby_lerp_f64_n1024 | 10.240 (0.146) | unsupported | 9.672 (0.100) | unsupported | unsupported |
| axpby_unit_add_f64_n1024 | 10.264 (0.381) | unsupported | 8.346 (0.167) | 351.181 (48.583) | 232.173 (20.067) |
| axpby_overwrite_c32_n1024 | 10.515 (0.201) | unsupported | 12.645 (0.210) | unsupported | unsupported |
| axpby_lerp_c32_n1024 | 10.838 (0.600) | unsupported | 8.388 (0.082) | unsupported | unsupported |
| axpby_unit_add_c32_n1024 | 10.542 (0.349) | unsupported | 8.693 (0.286) | 349.171 (12.039) | 253.744 (36.979) |
| axpby_overwrite_c64_n1024 | 11.046 (0.202) | unsupported | 12.686 (0.290) | unsupported | unsupported |
| axpby_lerp_c64_n1024 | 11.835 (0.215) | unsupported | 10.929 (0.003) | unsupported | unsupported |
| axpby_unit_add_c64_n1024 | 11.313 (0.974) | unsupported | 8.246 (0.160) | 348.181 (16.584) | 248.409 (41.350) |
| axpby_overwrite_f32_n1048576 | 304.845 (0.860) | unsupported | 27.447 (0.050) | unsupported | unsupported |
| axpby_lerp_f32_n1048576 | 273.125 (0.098) | unsupported | 39.813 (0.082) | unsupported | unsupported |
| axpby_unit_add_f32_n1048576 | 273.518 (0.411) | unsupported | 39.758 (0.085) | 363.768 (16.026) | 241.100 (21.350) |
| axpby_overwrite_f64_n1048576 | 338.859 (0.414) | unsupported | 53.196 (0.032) | unsupported | unsupported |
| axpby_lerp_f64_n1048576 | 534.120 (0.219) | unsupported | 78.063 (0.050) | unsupported | unsupported |
| axpby_unit_add_f64_n1048576 | 534.930 (0.444) | unsupported | 77.719 (0.058) | 367.515 (16.212) | 245.614 (17.304) |
| axpby_overwrite_c32_n1048576 | 156.930 (0.478) | unsupported | 53.189 (0.049) | unsupported | unsupported |
| axpby_lerp_c32_n1048576 | 168.902 (0.180) | unsupported | 77.971 (0.050) | unsupported | unsupported |
| axpby_unit_add_c32_n1048576 | 168.810 (0.333) | unsupported | 77.682 (0.031) | 375.145 (30.242) | 243.776 (15.446) |
| axpby_overwrite_c64_n1048576 | 1352.022 (2.398) | unsupported | 104.513 (0.158) | unsupported | unsupported |
| axpby_lerp_c64_n1048576 | 2136.773 (1.223) | unsupported | 342.120 (0.292) | unsupported | unsupported |
| axpby_unit_add_c64_n1048576 | 2136.078 (1.036) | unsupported | 153.563 (0.111) | 377.599 (24.145) | 250.356 (9.251) |
| dot_general_batch_f32_s32 | 111.428 (3.249) | 44.197 (0.484) | 45.239 (1.262) | 124.945 (4.427) | 70.348 (4.534) |
| dot_general_batch_f64_s32 | 121.507 (1.873) | 47.587 (0.791) | 45.620 (0.436) | 119.794 (3.234) | 66.053 (2.447) |
| dot_general_batch_c32_s32 | 141.641 (16.555) | 44.879 (0.832) | 46.597 (2.216) | 152.867 (16.148) | 89.757 (4.621) |
| dot_general_batch_c64_s32 | 153.625 (2.446) | 62.654 (14.190) | 48.991 (1.517) | 165.269 (17.592) | 99.112 (1.903) |
| dot_general_multiple_f32_s32 | 165.644 (3.374) | 52.062 (1.099) | 81.510 (0.528) | 145.949 (2.377) | 95.049 (2.672) |
| dot_general_multiple_f64_s32 | 159.293 (3.427) | 52.336 (1.410) | 83.155 (0.686) | 128.895 (3.855) | 98.729 (0.740) |
| dot_general_multiple_c32_s32 | 160.884 (2.685) | 52.150 (0.537) | 80.757 (1.627) | 170.333 (28.958) | 153.817 (15.168) |
| dot_general_multiple_c64_s32 | 162.917 (4.129) | 53.312 (0.336) | 83.460 (0.446) | 189.515 (8.365) | 162.167 (17.057) |
| dot_general_batch_f32_s256 | 53.431 (6.967) | 41.945 (0.530) | 46.394 (0.745) | 125.532 (3.843) | 78.525 (2.888) |
| dot_general_batch_f64_s256 | 350.231 (0.688) | 333.652 (0.211) | 420.817 (0.326) | 326.231 (16.424) | 316.499 (5.508) |
| dot_general_batch_c32_s256 | 41.886 (0.093) | 45.085 (3.175) | 54.303 (0.382) | 257.320 (33.001) | 163.313 (31.123) |
| dot_general_batch_c64_s256 | 1166.011 (65.244) | 1112.180 (1.481) | 1337.233 (1.158) | 1429.059 (14.472) | 1367.600 (13.181) |
| dot_general_multiple_f32_s256 | 48.197 (1.354) | 47.688 (0.863) | 82.551 (0.452) | 162.473 (0.641) | 121.490 (42.481) |
| dot_general_multiple_f64_s256 | 332.010 (0.338) | 334.012 (0.061) | 339.903 (0.415) | 219.403 (5.160) | 212.189 (2.109) |
| dot_general_multiple_c32_s256 | 45.078 (0.221) | 44.633 (1.115) | 89.691 (3.652) | 320.416 (42.689) | 189.487 (50.316) |
| dot_general_multiple_c64_s256 | 826.745 (1.052) | 833.583 (1.181) | 1017.759 (0.465) | 1152.344 (18.318) | 1059.671 (7.458) |
| dot_conj_l0r0_c32_n1024 | 160.464 (2.706) | 35.662 (5.177) | 15.465 (0.675) | 165.619 (4.091) | 177.765 (25.557) |
| dot_conj_l1r0_c32_n1024 | 166.364 (1.059) | 162.795 (18.057) | 15.506 (0.166) | 161.493 (41.099) | 166.923 (7.411) |
| dot_conj_l0r1_c32_n1024 | 178.813 (5.693) | 140.831 (2.816) | 15.489 (0.218) | 105.367 (7.750) | 181.423 (26.781) |
| dot_conj_l1r1_c32_n1024 | 167.498 (4.440) | 132.745 (2.900) | unsupported | unsupported | unsupported |
| dot_conj_l0r0_c64_n1024 | 235.261 (4.261) | 38.871 (3.484) | 15.277 (0.094) | 119.771 (16.173) | 180.375 (34.408) |
| dot_conj_l1r0_c64_n1024 | 166.313 (4.099) | 155.765 (7.464) | 15.420 (0.095) | 105.635 (4.884) | 185.242 (28.681) |
| dot_conj_l0r1_c64_n1024 | 176.730 (10.761) | 144.761 (5.883) | 15.575 (0.136) | 103.758 (0.495) | 191.607 (27.617) |
| dot_conj_l1r1_c64_n1024 | 179.289 (4.003) | 141.203 (2.257) | unsupported | unsupported | unsupported |
| dot_conj_l0r0_c32_n262144 | 668.856 (7.853) | 627.676 (3.593) | 17.469 (0.025) | 123.108 (13.418) | 193.360 (32.402) |
| dot_conj_l1r0_c32_n262144 | 176.038 (4.864) | 111.070 (1.017) | 17.390 (0.036) | 104.866 (2.636) | 164.727 (7.396) |
| dot_conj_l0r1_c32_n262144 | 179.321 (8.597) | 110.502 (2.103) | 17.507 (0.016) | 104.151 (0.541) | 158.466 (25.910) |
| dot_conj_l1r1_c32_n262144 | 178.979 (4.451) | 111.316 (1.523) | unsupported | unsupported | unsupported |
| dot_conj_l0r0_c64_n262144 | 949.726 (7.967) | 910.899 (1.168) | 33.907 (0.103) | 145.191 (1.240) | 153.945 (4.020) |
| dot_conj_l1r0_c64_n262144 | 176.431 (3.698) | 111.276 (2.163) | 33.945 (0.037) | 127.067 (16.941) | 173.594 (31.360) |
| dot_conj_l0r1_c64_n262144 | 184.350 (14.406) | 110.431 (1.352) | 33.812 (0.039) | 107.215 (4.629) | 176.105 (22.538) |
| dot_conj_l1r1_c64_n262144 | 181.911 (2.679) | 112.221 (1.342) | unsupported | unsupported | unsupported |
| einsum_plan_owned_f32_s32 | 175.054 (5.209) | unsupported | 23.747 (0.222) | 93.335 (0.791) | 104.144 (4.492) |
| einsum_plan_read_f32_s32 | 172.160 (3.863) | unsupported | 23.642 (0.204) | 93.750 (2.234) | 91.482 (10.901) |
| einsum_plan_strided_read_f32_s32 | 187.224 (1.540) | unsupported | 25.423 (0.158) | 92.279 (0.508) | 86.993 (2.122) |
| einsum_plan_into_f32_s32 | 26.068 (0.821) | unsupported | 20.181 (0.928) | unsupported | unsupported |
| einsum_plan_read_into_f32_s32 | 23.342 (0.932) | unsupported | 19.335 (0.844) | unsupported | unsupported |
| einsum_plan_read_into_accum_f32_s32 | 24.818 (1.219) | unsupported | 26.456 (0.265) | unsupported | unsupported |
| einsum_plan_strided_read_into_f32_s32 | 20.515 (1.641) | unsupported | 21.255 (0.501) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_f32_s32 | 24.539 (2.256) | unsupported | 28.737 (0.407) | unsupported | unsupported |
| einsum_plan_owned_f64_s32 | 122.864 (5.464) | unsupported | 25.514 (0.302) | 91.440 (1.987) | 82.301 (1.114) |
| einsum_plan_read_f64_s32 | 146.825 (11.242) | unsupported | 25.555 (0.186) | 91.186 (2.365) | 83.140 (1.398) |
| einsum_plan_strided_read_f64_s32 | 164.279 (3.113) | unsupported | 26.641 (0.601) | 147.172 (3.083) | 66.982 (1.290) |
| einsum_plan_into_f64_s32 | 25.549 (0.670) | unsupported | 21.383 (0.591) | unsupported | unsupported |
| einsum_plan_read_into_f64_s32 | 22.555 (0.402) | unsupported | 20.377 (0.733) | unsupported | unsupported |
| einsum_plan_read_into_accum_f64_s32 | 26.663 (1.114) | unsupported | 27.767 (0.360) | unsupported | unsupported |
| einsum_plan_strided_read_into_f64_s32 | 23.990 (0.874) | unsupported | 21.665 (1.271) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_f64_s32 | 26.785 (0.601) | unsupported | 29.862 (0.995) | unsupported | unsupported |
| einsum_plan_owned_c32_s32 | 122.277 (7.402) | unsupported | 23.178 (0.113) | 140.068 (18.735) | 79.614 (1.838) |
| einsum_plan_read_c32_s32 | 139.592 (12.748) | unsupported | 23.391 (0.156) | 121.712 (9.177) | 79.786 (1.794) |
| einsum_plan_strided_read_c32_s32 | 152.099 (1.906) | unsupported | 23.539 (0.227) | 186.162 (19.128) | 98.305 (17.698) |
| einsum_plan_into_c32_s32 | 24.035 (0.230) | unsupported | 19.547 (1.874) | unsupported | unsupported |
| einsum_plan_read_into_c32_s32 | 23.400 (0.931) | unsupported | 18.270 (0.372) | unsupported | unsupported |
| einsum_plan_read_into_accum_c32_s32 | 24.520 (1.194) | unsupported | 26.372 (0.247) | unsupported | unsupported |
| einsum_plan_strided_read_into_c32_s32 | 25.776 (0.403) | unsupported | 18.765 (0.148) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_c32_s32 | 25.064 (1.041) | unsupported | 26.666 (0.286) | unsupported | unsupported |
| einsum_plan_owned_c64_s32 | 123.327 (7.599) | unsupported | 24.892 (0.193) | 141.178 (23.631) | 86.761 (2.292) |
| einsum_plan_read_c64_s32 | 141.556 (17.965) | unsupported | 25.031 (0.378) | 122.905 (4.415) | 86.448 (0.896) |
| einsum_plan_strided_read_c64_s32 | 165.184 (3.023) | unsupported | 26.566 (2.275) | 184.447 (36.448) | 118.902 (15.756) |
| einsum_plan_into_c64_s32 | 22.894 (1.638) | unsupported | 20.043 (0.278) | unsupported | unsupported |
| einsum_plan_read_into_c64_s32 | 23.773 (0.731) | unsupported | 20.753 (0.866) | unsupported | unsupported |
| einsum_plan_read_into_accum_c64_s32 | 25.176 (1.358) | unsupported | 28.404 (1.273) | unsupported | unsupported |
| einsum_plan_strided_read_into_c64_s32 | 28.582 (1.690) | unsupported | 22.549 (0.683) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_c64_s32 | 26.926 (0.997) | unsupported | 29.594 (1.077) | unsupported | unsupported |
| einsum_plan_owned_f32_s256 | 38.518 (0.750) | unsupported | 25.279 (0.233) | 75.215 (5.212) | 71.995 (1.310) |
| einsum_plan_read_f32_s256 | 38.548 (0.817) | unsupported | 24.520 (0.295) | 75.790 (2.184) | 71.787 (1.415) |
| einsum_plan_strided_read_f32_s256 | 43.371 (0.873) | unsupported | 26.737 (0.320) | 82.276 (4.605) | 71.963 (1.289) |
| einsum_plan_into_f32_s256 | 25.269 (0.969) | unsupported | 19.210 (0.292) | unsupported | unsupported |
| einsum_plan_read_into_f32_s256 | 23.082 (0.661) | unsupported | 18.836 (0.117) | unsupported | unsupported |
| einsum_plan_read_into_accum_f32_s256 | 24.998 (0.567) | unsupported | 26.734 (0.145) | unsupported | unsupported |
| einsum_plan_strided_read_into_f32_s256 | 24.903 (0.643) | unsupported | 21.220 (0.460) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_f32_s256 | 26.338 (1.027) | unsupported | 29.081 (0.427) | unsupported | unsupported |
| einsum_plan_owned_f64_s256 | 174.124 (5.238) | unsupported | 167.652 (2.489) | 108.257 (0.995) | 109.109 (1.760) |
| einsum_plan_read_f64_s256 | 166.284 (0.113) | unsupported | 167.542 (2.480) | 108.623 (1.075) | 107.937 (1.274) |
| einsum_plan_strided_read_f64_s256 | 166.938 (0.186) | unsupported | 164.750 (2.881) | 159.513 (2.172) | 108.614 (2.005) |
| einsum_plan_into_f64_s256 | 165.474 (0.074) | unsupported | 167.092 (2.569) | unsupported | unsupported |
| einsum_plan_read_into_f64_s256 | 165.592 (0.120) | unsupported | 166.333 (2.551) | unsupported | unsupported |
| einsum_plan_read_into_accum_f64_s256 | 167.054 (0.123) | unsupported | 169.484 (0.146) | unsupported | unsupported |
| einsum_plan_strided_read_into_f64_s256 | 166.021 (0.140) | unsupported | 164.243 (4.944) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_f64_s256 | 167.448 (0.164) | unsupported | 164.093 (6.251) | unsupported | unsupported |
| einsum_plan_owned_c32_s256 | 38.993 (0.511) | unsupported | 23.209 (0.563) | 133.274 (15.036) | 86.525 (4.021) |
| einsum_plan_read_c32_s256 | 39.420 (0.495) | unsupported | 23.365 (0.382) | 122.444 (4.007) | 81.604 (1.509) |
| einsum_plan_strided_read_c32_s256 | 45.120 (1.986) | unsupported | 24.796 (0.766) | 162.892 (19.188) | 112.790 (11.269) |
| einsum_plan_into_c32_s256 | 26.990 (0.667) | unsupported | 18.412 (0.369) | unsupported | unsupported |
| einsum_plan_read_into_c32_s256 | 25.403 (0.459) | unsupported | 18.290 (0.342) | unsupported | unsupported |
| einsum_plan_read_into_accum_c32_s256 | 25.690 (0.389) | unsupported | 25.961 (0.098) | unsupported | unsupported |
| einsum_plan_strided_read_into_c32_s256 | 29.673 (1.587) | unsupported | 20.792 (0.386) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_c32_s256 | 28.992 (0.851) | unsupported | 28.232 (0.493) | unsupported | unsupported |
| einsum_plan_owned_c64_s256 | 420.525 (4.060) | unsupported | 493.943 (0.110) | 402.166 (11.045) | 418.507 (19.981) |
| einsum_plan_read_c64_s256 | 418.683 (5.529) | unsupported | 493.813 (0.101) | 402.268 (11.116) | 436.100 (20.549) |
| einsum_plan_strided_read_c64_s256 | 411.974 (6.749) | unsupported | 493.982 (0.239) | 410.325 (12.962) | 437.066 (14.236) |
| einsum_plan_into_c64_s256 | 417.556 (3.791) | unsupported | 493.153 (0.120) | unsupported | unsupported |
| einsum_plan_read_into_c64_s256 | 417.947 (4.011) | unsupported | 493.337 (0.199) | unsupported | unsupported |
| einsum_plan_read_into_accum_c64_s256 | 420.882 (3.849) | unsupported | 503.161 (0.779) | unsupported | unsupported |
| einsum_plan_strided_read_into_c64_s256 | 409.851 (8.047) | unsupported | 493.408 (0.144) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_c64_s256 | 409.061 (4.613) | unsupported | 502.965 (0.223) | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.603 | 0 |
| add_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.645 | 0 |
| add_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.643 | 0 |
| add_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.641 | 0 |
| sub_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.800 | 0 |
| sub_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.669 | 0 |
| sub_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.955 | 0 |
| sub_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.710 | 0 |
| mul_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.906 | 0 |
| mul_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.871 | 0 |
| mul_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.859 | 0 |
| mul_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.667 | 0 |
| div_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.396 | 0 |
| div_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.682 | 0 |
| div_read_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.688 | 0 |
| div_read_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.638 | 0 |
| neg_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.478 | 0 |
| neg_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.394 | 0 |
| neg_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.511 | 0 |
| neg_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.455 | 0 |
| conj_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.579 | 0 |
| conj_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.082 | 0 |
| conj_read_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.468 | 0 |
| conj_read_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.819 | 0 |
| add_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.783 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.703 | 0 |
| add_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.743 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.645 | 0 |
| sub_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.742 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.635 | 0 |
| sub_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.734 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.638 | 0 |
| mul_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.734 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.645 | 0 |
| mul_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.739 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.638 | 0 |
| div_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.243 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.422 | 0 |
| div_read_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.243 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.421 | 0 |
| neg_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.891 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.831 | 0 |
| neg_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.895 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.827 | 0 |
| conj_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.451 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.344 | 0 |
| conj_read_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.448 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.341 | 0 |
| tensordot_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.488 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.875 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.605 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-direct | 32 | 15 | 3.729 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.325 | 0 |
| vdot_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.556 | 1.722e-07 |
| vdot_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.568 | 0 |
| vdot_read_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.911 | 3.455e-07 |
| norm_squared_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.707 | 0 |
| norm_squared_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.540 | 0 |
| tensordot_f32_n262144 | tenferro-cuda-direct | 32 | 15 | 2.306 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 2 | 15 | 3.239 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.260 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-direct | 1 | 15 | 5.031 | 1.446e-05 |
| vdot_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.731 | 0 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.020 | 2.995e-06 |
| vdot_read_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.983 | 0 |
| vdot_read_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.952 | 8.762e-06 |
| norm_squared_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.916 | 0.0001786 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.929 | 0 |
| axpby_read_into_accum_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.764 | 0 |
| axpby_read_into_accum_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.561 | 0 |
| axpby_read_into_accum_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.703 | 0 |
| axpby_read_into_accum_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.852 | 0 |
| axpby_read_into_accum_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.187 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.137 | 0 |
| axpby_read_into_accum_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.701 | 0 |
| axpby_read_into_accum_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.145 | 0 |
| dot_general_read_into_f32_s16 | tenferro-cuda-direct | 128 | 15 | 2.868 | 8.645e-05 |
| dot_general_read_into_accum_f32_s16 | tenferro-cuda-direct | 128 | 15 | 3.001 | 7.907e-05 |
| dot_general_read_into_accum_cached_f32_s16 | tenferro-cuda-direct | 128 | 15 | 2.851 | 7.907e-05 |
| dot_general_read_into_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.301 | 0 |
| dot_general_read_into_accum_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.230 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.252 | 0 |
| dot_general_read_into_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.895 | 0.0003615 |
| dot_general_read_into_accum_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.823 | 0.0001632 |
| dot_general_read_into_accum_cached_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.838 | 0.0001632 |
| dot_general_read_into_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.116 | 1.204e-07 |
| dot_general_read_into_accum_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.058 | 6.975e-08 |
| dot_general_read_into_accum_cached_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.224 | 6.975e-08 |
| dot_general_read_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.688 | 0.0005542 |
| dot_general_read_into_accum_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.015 | 0.0003617 |
| dot_general_read_into_accum_cached_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.009 | 0.0003617 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.107 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.385 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.466 | 0 |
| dot_general_read_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.626 | 0.001457 |
| dot_general_read_into_accum_c32_s256 | tenferro-cuda-direct | 64 | 15 | 1.929 | 0.0008947 |
| dot_general_read_into_accum_cached_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.818 | 0.0008947 |
| dot_general_read_into_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.737 | 2.092e-06 |
| dot_general_read_into_accum_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.739 | 1.318e-06 |
| dot_general_read_into_accum_cached_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.737 | 1.318e-06 |
| add_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.270 | 0 |
| add_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 4.013 | 0 |
| mul_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.675 | 0 |
| mul_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.470 | 0 |
| neg_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.135 | 0 |
| neg_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.702 | 0 |
| exp_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.138 | 0.0001272 |
| exp_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.764 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.229 | 9.107e-05 |
| tanh_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.732 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.675 | 0.0001208 |
| reduce_sum_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.386 | 4.202e-07 |
| conj_strided_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.616 | 0 |
| conj_strided_read_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.814 | 0 |
| add_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.144 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.016 | 0 |
| mul_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.133 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.010 | 0 |
| neg_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.589 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.461 | 0 |
| exp_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.596 | 0.0001272 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.739 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.602 | 9.135e-05 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.529 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.894 | 0.002489 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.972 | 3.227e-06 |
| conj_strided_read_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.660 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.425 | 0 |
| axpby_overwrite_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.812 | 0 |
| axpby_lerp_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.730 | 0 |
| axpby_unit_add_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.840 | 0 |
| axpby_overwrite_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.655 | 0 |
| axpby_lerp_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.622 | 1.155e-07 |
| axpby_unit_add_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.628 | 0 |
| axpby_overwrite_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.692 | 0 |
| axpby_lerp_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.775 | 0 |
| axpby_unit_add_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.699 | 0 |
| axpby_overwrite_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.828 | 0 |
| axpby_lerp_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.030 | 1.155e-07 |
| axpby_unit_add_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.896 | 0 |
| axpby_overwrite_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.439 | 0 |
| axpby_lerp_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.185 | 0 |
| axpby_unit_add_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.188 | 0 |
| axpby_overwrite_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.711 | 0 |
| axpby_lerp_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.136 | 1.16e-07 |
| axpby_unit_add_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.140 | 0 |
| axpby_overwrite_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.511 | 0 |
| axpby_lerp_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.702 | 0 |
| axpby_unit_add_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.701 | 0 |
| axpby_overwrite_c64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.704 | 0 |
| axpby_lerp_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.137 | 1.159e-07 |
| axpby_unit_add_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.136 | 0 |
| dot_general_batch_f32_s32 | tenferro-cuda-direct | 32 | 15 | 3.566 | 0.0002227 |
| dot_general_batch_f64_s32 | tenferro-cuda-direct | 16 | 15 | 1.944 | 1.326e-07 |
| dot_general_batch_c32_s32 | tenferro-cuda-direct | 32 | 15 | 4.533 | 0.0004376 |
| dot_general_batch_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.458 | 4.296e-07 |
| dot_general_multiple_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.650 | 0.0002162 |
| dot_general_multiple_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.549 | 1.632e-07 |
| dot_general_multiple_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.574 | 0.0002794 |
| dot_general_multiple_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.607 | 4.303e-07 |
| dot_general_batch_f32_s256 | tenferro-cuda-direct | 64 | 15 | 3.420 | 0.0008561 |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.802 | 7.463e-07 |
| dot_general_batch_c32_s256 | tenferro-cuda-direct | 64 | 15 | 2.681 | 0.001749 |
| dot_general_batch_c64_s256 | tenferro-cuda-direct | 2 | 15 | 2.332 | 1.665e-06 |
| dot_general_multiple_f32_s256 | tenferro-cuda-direct | 64 | 15 | 3.085 | 0.002418 |
| dot_general_multiple_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.656 | 1.519e-06 |
| dot_general_multiple_c32_s256 | tenferro-cuda-direct | 64 | 15 | 2.885 | 0.0022 |
| dot_general_multiple_c64_s256 | tenferro-cuda-direct | 4 | 15 | 3.307 | 2.187e-06 |
| dot_conj_l0r0_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.567 | 0.0004142 |
| dot_conj_l1r0_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.662 | 0.0002905 |
| dot_conj_l0r1_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.861 | 0.0002905 |
| dot_conj_l1r1_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.680 | 0.0003083 |
| dot_conj_l0r0_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.764 | 3.737e-07 |
| dot_conj_l1r0_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.661 | 1.807e-07 |
| dot_conj_l0r1_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.828 | 1.807e-07 |
| dot_conj_l1r1_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.869 | 4.927e-07 |
| dot_conj_l0r0_c32_n262144 | tenferro-cuda-direct | 4 | 15 | 2.675 | 0.02908 |
| dot_conj_l1r0_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.817 | 0.005619 |
| dot_conj_l0r1_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.869 | 0.005619 |
| dot_conj_l1r1_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.864 | 0.005822 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | 4 | 15 | 3.799 | 1.986e-05 |
| dot_conj_l1r0_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.823 | 1.516e-05 |
| dot_conj_l0r1_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.950 | 1.516e-05 |
| dot_conj_l1r1_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.911 | 1.212e-05 |
| einsum_plan_owned_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.801 | 0.0002018 |
| einsum_plan_read_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.755 | 0.0002018 |
| einsum_plan_strided_read_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.996 | 0.0002018 |
| einsum_plan_into_f32_s32 | tenferro-cuda-direct | 128 | 15 | 3.337 | 0.0002018 |
| einsum_plan_read_into_f32_s32 | tenferro-cuda-direct | 128 | 15 | 2.988 | 0.0002018 |
| einsum_plan_read_into_accum_f32_s32 | tenferro-cuda-direct | 128 | 15 | 3.177 | 0.0001134 |
| einsum_plan_strided_read_into_f32_s32 | tenferro-cuda-direct | 128 | 15 | 2.626 | 0.0002018 |
| einsum_plan_strided_read_into_accum_f32_s32 | tenferro-cuda-direct | 128 | 15 | 3.141 | 0.0001134 |
| einsum_plan_owned_f64_s32 | tenferro-cuda-direct | 32 | 15 | 3.932 | 0 |
| einsum_plan_read_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.349 | 0 |
| einsum_plan_strided_read_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.628 | 0 |
| einsum_plan_into_f64_s32 | tenferro-cuda-direct | 128 | 15 | 3.270 | 0 |
| einsum_plan_read_into_f64_s32 | tenferro-cuda-direct | 128 | 15 | 2.887 | 0 |
| einsum_plan_read_into_accum_f64_s32 | tenferro-cuda-direct | 128 | 15 | 3.413 | 0 |
| einsum_plan_strided_read_into_f64_s32 | tenferro-cuda-direct | 128 | 15 | 3.071 | 0 |
| einsum_plan_strided_read_into_accum_f64_s32 | tenferro-cuda-direct | 128 | 15 | 3.428 | 0 |
| einsum_plan_owned_c32_s32 | tenferro-cuda-direct | 32 | 15 | 3.913 | 0.0004148 |
| einsum_plan_read_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.233 | 0.0004148 |
| einsum_plan_strided_read_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.434 | 0.0004148 |
| einsum_plan_into_c32_s32 | tenferro-cuda-direct | 128 | 15 | 3.076 | 0.0004148 |
| einsum_plan_read_into_c32_s32 | tenferro-cuda-direct | 128 | 15 | 2.995 | 0.0004148 |
| einsum_plan_read_into_accum_c32_s32 | tenferro-cuda-direct | 128 | 15 | 3.139 | 0.0001409 |
| einsum_plan_strided_read_into_c32_s32 | tenferro-cuda-direct | 128 | 15 | 3.299 | 0.0004148 |
| einsum_plan_strided_read_into_accum_c32_s32 | tenferro-cuda-direct | 128 | 15 | 3.208 | 0.0001409 |
| einsum_plan_owned_c64_s32 | tenferro-cuda-direct | 32 | 15 | 3.946 | 1.978e-07 |
| einsum_plan_read_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.265 | 1.978e-07 |
| einsum_plan_strided_read_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.643 | 1.978e-07 |
| einsum_plan_into_c64_s32 | tenferro-cuda-direct | 128 | 15 | 2.930 | 1.978e-07 |
| einsum_plan_read_into_c64_s32 | tenferro-cuda-direct | 128 | 15 | 3.043 | 1.978e-07 |
| einsum_plan_read_into_accum_c64_s32 | tenferro-cuda-direct | 128 | 15 | 3.223 | 1.074e-07 |
| einsum_plan_strided_read_into_c64_s32 | tenferro-cuda-direct | 128 | 15 | 3.658 | 1.978e-07 |
| einsum_plan_strided_read_into_accum_c64_s32 | tenferro-cuda-direct | 128 | 15 | 3.447 | 1.074e-07 |
| einsum_plan_owned_f32_s256 | tenferro-cuda-direct | 64 | 15 | 2.465 | 0.0009425 |
| einsum_plan_read_f32_s256 | tenferro-cuda-direct | 64 | 15 | 2.467 | 0.0009425 |
| einsum_plan_strided_read_f32_s256 | tenferro-cuda-direct | 64 | 15 | 2.776 | 0.0009425 |
| einsum_plan_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.234 | 0.0009425 |
| einsum_plan_read_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.954 | 0.0009425 |
| einsum_plan_read_into_accum_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.200 | 0.0003475 |
| einsum_plan_strided_read_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.188 | 0.0009425 |
| einsum_plan_strided_read_into_accum_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.371 | 0.0003475 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.786 | 0 |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.661 | 0 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.671 | 0 |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.648 | 0 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.649 | 0 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.673 | 0 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.656 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.679 | 0 |
| einsum_plan_owned_c32_s256 | tenferro-cuda-direct | 64 | 15 | 2.496 | 0.001996 |
| einsum_plan_read_c32_s256 | tenferro-cuda-direct | 64 | 15 | 2.523 | 0.001996 |
| einsum_plan_strided_read_c32_s256 | tenferro-cuda-direct | 64 | 15 | 2.888 | 0.001996 |
| einsum_plan_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.455 | 0.001996 |
| einsum_plan_read_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.252 | 0.001996 |
| einsum_plan_read_into_accum_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.288 | 0.0006165 |
| einsum_plan_strided_read_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.798 | 0.001996 |
| einsum_plan_strided_read_into_accum_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.711 | 0.0006165 |
| einsum_plan_owned_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.364 | 2.139e-06 |
| einsum_plan_read_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.349 | 2.139e-06 |
| einsum_plan_strided_read_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.296 | 2.139e-06 |
| einsum_plan_into_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.340 | 2.139e-06 |
| einsum_plan_read_into_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.344 | 2.139e-06 |
| einsum_plan_read_into_accum_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.367 | 9.602e-07 |
| einsum_plan_strided_read_into_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.279 | 2.139e-06 |
| einsum_plan_strided_read_into_accum_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.272 | 9.602e-07 |
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
| tensordot_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.445 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.634 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.342 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.619 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| tensordot_f32_n262144 | tenferro-cuda-eager | 64 | 15 | 2.960 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 2 | 15 | 3.241 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-eager | 16 | 15 | 2.267 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-eager | 1 | 15 | 5.063 | 1.446e-05 |
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
| dot_general_batch_f32_s32 | tenferro-cuda-eager | 64 | 15 | 2.829 | 0.0002227 |
| dot_general_batch_f64_s32 | tenferro-cuda-eager | 64 | 15 | 3.046 | 1.326e-07 |
| dot_general_batch_c32_s32 | tenferro-cuda-eager | 64 | 15 | 2.872 | 0.0004376 |
| dot_general_batch_c64_s32 | tenferro-cuda-eager | 64 | 15 | 4.010 | 4.296e-07 |
| dot_general_multiple_f32_s32 | tenferro-cuda-eager | 64 | 15 | 3.332 | 0.0002162 |
| dot_general_multiple_f64_s32 | tenferro-cuda-eager | 64 | 15 | 3.350 | 1.632e-07 |
| dot_general_multiple_c32_s32 | tenferro-cuda-eager | 64 | 15 | 3.338 | 0.0002794 |
| dot_general_multiple_c64_s32 | tenferro-cuda-eager | 64 | 15 | 3.412 | 4.303e-07 |
| dot_general_batch_f32_s256 | tenferro-cuda-eager | 64 | 15 | 2.685 | 0.0008561 |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | 8 | 15 | 2.669 | 7.463e-07 |
| dot_general_batch_c32_s256 | tenferro-cuda-eager | 64 | 15 | 2.885 | 0.001749 |
| dot_general_batch_c64_s256 | tenferro-cuda-eager | 2 | 15 | 2.224 | 1.665e-06 |
| dot_general_multiple_f32_s256 | tenferro-cuda-eager | 64 | 15 | 3.052 | 0.002418 |
| dot_general_multiple_f64_s256 | tenferro-cuda-eager | 8 | 15 | 2.672 | 1.519e-06 |
| dot_general_multiple_c32_s256 | tenferro-cuda-eager | 64 | 15 | 2.856 | 0.0022 |
| dot_general_multiple_c64_s256 | tenferro-cuda-eager | 4 | 15 | 3.334 | 2.187e-06 |
| dot_conj_l0r0_c32_n1024 | tenferro-cuda-eager | 128 | 15 | 4.565 | 0.0004142 |
| dot_conj_l1r0_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.605 | 0.0002905 |
| dot_conj_l0r1_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.253 | 0.0002905 |
| dot_conj_l1r1_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.124 | 0.0003083 |
| dot_conj_l0r0_c64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.488 | 3.737e-07 |
| dot_conj_l1r0_c64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.492 | 1.807e-07 |
| dot_conj_l0r1_c64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.316 | 1.807e-07 |
| dot_conj_l1r1_c64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.259 | 4.927e-07 |
| dot_conj_l0r0_c32_n262144 | tenferro-cuda-eager | 4 | 15 | 2.511 | 0.02908 |
| dot_conj_l1r0_c32_n262144 | tenferro-cuda-eager | 32 | 15 | 3.554 | 0.005619 |
| dot_conj_l0r1_c32_n262144 | tenferro-cuda-eager | 32 | 15 | 3.536 | 0.005619 |
| dot_conj_l1r1_c32_n262144 | tenferro-cuda-eager | 32 | 15 | 3.562 | 0.005822 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | 4 | 15 | 3.644 | 1.986e-05 |
| dot_conj_l1r0_c64_n262144 | tenferro-cuda-eager | 32 | 15 | 3.561 | 1.516e-05 |
| dot_conj_l0r1_c64_n262144 | tenferro-cuda-eager | 32 | 15 | 3.534 | 1.516e-05 |
| dot_conj_l1r1_c64_n262144 | tenferro-cuda-eager | 32 | 15 | 3.591 | 1.212e-05 |
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
| add_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.547 | 0 |
| add_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.824 | 0 |
| add_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.469 | 0 |
| add_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.624 | 0 |
| sub_into_f32_n1024 | pytorch-cuda | 256 | 15 | 1.966 | 0 |
| sub_into_f64_n1024 | pytorch-cuda | 256 | 15 | 1.924 | 0 |
| sub_read_into_f32_n1024 | pytorch-cuda | 256 | 15 | 2.035 | 0 |
| sub_read_into_f64_n1024 | pytorch-cuda | 256 | 15 | 2.118 | 0 |
| mul_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.628 | 0 |
| mul_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.634 | 0 |
| mul_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.593 | 0 |
| mul_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.561 | 0 |
| div_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.511 | 0 |
| div_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.645 | 0 |
| div_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.533 | 0 |
| div_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.543 | 0 |
| neg_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.423 | 0 |
| neg_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.539 | 0 |
| neg_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.351 | 0 |
| neg_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.471 | 0 |
| conj_into_c32_n1024 | pytorch-cuda | 512 | 15 | 3.613 | 0 |
| conj_into_c64_n1024 | pytorch-cuda | 512 | 15 | 3.585 | 0 |
| conj_read_into_c32_n1024 | pytorch-cuda | 512 | 15 | 3.754 | 0 |
| conj_read_into_c64_n1024 | pytorch-cuda | 512 | 15 | 3.547 | 0 |
| add_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.547 | 0 |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.487 | 0 |
| add_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.541 | 0 |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| sub_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.541 | 0 |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.485 | 0 |
| sub_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.540 | 0 |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| mul_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.551 | 0 |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.482 | 0 |
| mul_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.538 | 0 |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.481 | 0 |
| div_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.023 | 0 |
| div_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.554 | 0 |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.023 | 0 |
| neg_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.744 | 0 |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.692 | 0 |
| neg_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.743 | 0 |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.692 | 0 |
| conj_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.690 | 0 |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.658 | 0 |
| conj_read_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.690 | 0 |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.659 | 0 |
| tensordot_f32_n1024 | pytorch-cuda | 64 | 15 | 3.426 | 0.0008859 |
| tensordot_f64_n1024 | pytorch-cuda | 64 | 15 | 3.495 | 0 |
| tensordot_c32_n1024 | pytorch-cuda | 64 | 15 | 3.302 | 0.0005344 |
| tensordot_c64_n1024 | pytorch-cuda | 64 | 15 | 3.545 | 0 |
| vdot_read_f32_n1024 | pytorch-cuda | 128 | 15 | 2.042 | 0 |
| vdot_read_f64_n1024 | pytorch-cuda | 128 | 15 | 2.029 | 1.722e-07 |
| vdot_read_c32_n1024 | pytorch-cuda | 128 | 15 | 1.993 | 0 |
| vdot_read_c64_n1024 | pytorch-cuda | 256 | 15 | 3.928 | 3.455e-07 |
| norm_squared_read_f32_n1024 | pytorch-cuda | 128 | 15 | 2.033 | 0 |
| norm_squared_read_f64_n1024 | pytorch-cuda | 128 | 15 | 2.139 | 0 |
| tensordot_f32_n262144 | pytorch-cuda | 32 | 15 | 2.142 | 0.006191 |
| tensordot_f64_n262144 | pytorch-cuda | 2 | 15 | 3.273 | 6.714e-06 |
| tensordot_c32_n262144 | pytorch-cuda | 16 | 15 | 2.542 | 0.005717 |
| tensordot_c64_n262144 | pytorch-cuda | 1 | 15 | 5.289 | 4.401e-06 |
| vdot_read_f32_n262144 | pytorch-cuda | 128 | 15 | 2.035 | 0 |
| vdot_read_f64_n262144 | pytorch-cuda | 128 | 15 | 2.597 | 2.995e-06 |
| vdot_read_c32_n262144 | pytorch-cuda | 128 | 15 | 2.224 | 0 |
| vdot_read_c64_n262144 | pytorch-cuda | 64 | 15 | 2.167 | 8.762e-06 |
| norm_squared_read_f32_n262144 | pytorch-cuda | 128 | 15 | 2.026 | 0.0001786 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 128 | 15 | 2.007 | 0 |
| axpby_read_into_accum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.062 | 0 |
| axpby_read_into_accum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.055 | 0 |
| axpby_read_into_accum_c32_n1024 | pytorch-cuda | 256 | 15 | 2.098 | 0 |
| axpby_read_into_accum_c64_n1024 | pytorch-cuda | 256 | 15 | 2.063 | 0 |
| axpby_read_into_accum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.543 | 0 |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.487 | 0 |
| axpby_read_into_accum_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.487 | 0 |
| axpby_read_into_accum_c64_n1048576 | pytorch-cuda | 16 | 15 | 2.457 | 0 |
| dot_general_read_into_f32_s16 | pytorch-cuda | 128 | 15 | 2.787 | 8.645e-05 |
| dot_general_read_into_accum_f32_s16 | pytorch-cuda | 128 | 15 | 3.268 | 7.907e-05 |
| dot_general_read_into_accum_cached_f32_s16 | pytorch-cuda | 128 | 15 | 3.456 | 7.907e-05 |
| dot_general_read_into_f64_s16 | pytorch-cuda | 128 | 15 | 3.038 | 0 |
| dot_general_read_into_accum_f64_s16 | pytorch-cuda | 128 | 15 | 3.582 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | pytorch-cuda | 128 | 15 | 3.585 | 0 |
| dot_general_read_into_c32_s16 | pytorch-cuda | 128 | 15 | 3.192 | 0.0001647 |
| dot_general_read_into_accum_c32_s16 | pytorch-cuda | 64 | 15 | 1.785 | 0.0001125 |
| dot_general_read_into_accum_cached_c32_s16 | pytorch-cuda | 128 | 15 | 3.563 | 0.0001125 |
| dot_general_read_into_c64_s16 | pytorch-cuda | 128 | 15 | 3.028 | 1.373e-07 |
| dot_general_read_into_accum_c64_s16 | pytorch-cuda | 128 | 15 | 3.558 | 7.37e-08 |
| dot_general_read_into_accum_cached_c64_s16 | pytorch-cuda | 128 | 15 | 3.537 | 7.37e-08 |
| dot_general_read_into_f32_s256 | pytorch-cuda | 128 | 15 | 2.919 | 0.0005542 |
| dot_general_read_into_accum_f32_s256 | pytorch-cuda | 128 | 15 | 3.445 | 0.0003617 |
| dot_general_read_into_accum_cached_f32_s256 | pytorch-cuda | 128 | 15 | 3.503 | 0.0003617 |
| dot_general_read_into_f64_s256 | pytorch-cuda | 16 | 15 | 4.276 | 0 |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 8 | 15 | 2.414 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 8 | 15 | 2.494 | 0 |
| dot_general_read_into_c32_s256 | pytorch-cuda | 128 | 15 | 3.457 | 0.0008017 |
| dot_general_read_into_accum_c32_s256 | pytorch-cuda | 128 | 15 | 3.902 | 0.000499 |
| dot_general_read_into_accum_cached_c32_s256 | pytorch-cuda | 128 | 15 | 3.863 | 0.000499 |
| dot_general_read_into_c64_s256 | pytorch-cuda | 4 | 15 | 3.902 | 1.175e-06 |
| dot_general_read_into_accum_c64_s256 | pytorch-cuda | 4 | 15 | 3.929 | 6.523e-07 |
| dot_general_read_into_accum_cached_c64_s256 | pytorch-cuda | 4 | 15 | 3.928 | 6.523e-07 |
| add_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.538 | 0 |
| add_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.558 | 0 |
| mul_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.556 | 0 |
| mul_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.502 | 0 |
| neg_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.455 | 0 |
| neg_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.566 | 0 |
| exp_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.285 | 0.0001272 |
| exp_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.309 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.451 | 9.107e-05 |
| tanh_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.361 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | pytorch-cuda | 128 | 15 | 1.985 | 0.0005233 |
| reduce_sum_strided_read_f64_n1024 | pytorch-cuda | 128 | 15 | 1.951 | 3.504e-07 |
| conj_strided_read_c32_n1024 | pytorch-cuda | 256 | 15 | 3.035 | 0 |
| conj_strided_read_c64_n1024 | pytorch-cuda | 256 | 15 | 3.213 | 0 |
| add_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.555 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.489 | 0 |
| mul_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.546 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| neg_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.751 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.698 | 0 |
| exp_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.759 | 0.0001272 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.898 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.757 | 9.135e-05 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.685 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.916 | 0.002693 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.368 | 5.935e-06 |
| conj_strided_read_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.698 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.666 | 0 |
| axpby_overwrite_f32_n1024 | pytorch-cuda | 256 | 15 | 3.190 | 0 |
| axpby_lerp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.508 | 0.0001189 |
| axpby_unit_add_f32_n1024 | pytorch-cuda | 256 | 15 | 2.060 | 0 |
| axpby_overwrite_f64_n1024 | pytorch-cuda | 256 | 15 | 3.183 | 0 |
| axpby_lerp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.476 | 1.161e-07 |
| axpby_unit_add_f64_n1024 | pytorch-cuda | 256 | 15 | 2.137 | 0 |
| axpby_overwrite_c32_n1024 | pytorch-cuda | 256 | 15 | 3.237 | 0 |
| axpby_lerp_c32_n1024 | pytorch-cuda | 256 | 15 | 2.147 | 0.0001189 |
| axpby_unit_add_c32_n1024 | pytorch-cuda | 256 | 15 | 2.225 | 0 |
| axpby_overwrite_c64_n1024 | pytorch-cuda | 256 | 15 | 3.248 | 0 |
| axpby_lerp_c64_n1024 | pytorch-cuda | 256 | 15 | 2.798 | 1.161e-07 |
| axpby_unit_add_c64_n1024 | pytorch-cuda | 256 | 15 | 2.111 | 0 |
| axpby_overwrite_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.757 | 0 |
| axpby_lerp_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.548 | 0.0001192 |
| axpby_unit_add_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| axpby_overwrite_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.702 | 0 |
| axpby_lerp_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.498 | 1.161e-07 |
| axpby_unit_add_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.487 | 0 |
| axpby_overwrite_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.702 | 0 |
| axpby_lerp_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.495 | 0.0001191 |
| axpby_unit_add_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| axpby_overwrite_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.672 | 0 |
| axpby_lerp_c64_n1048576 | pytorch-cuda | 8 | 15 | 2.737 | 1.159e-07 |
| axpby_unit_add_c64_n1048576 | pytorch-cuda | 16 | 15 | 2.457 | 0 |
| dot_general_batch_f32_s32 | pytorch-cuda | 64 | 15 | 2.895 | 0.0002227 |
| dot_general_batch_f64_s32 | pytorch-cuda | 64 | 15 | 2.920 | 1.326e-07 |
| dot_general_batch_c32_s32 | pytorch-cuda | 64 | 15 | 2.982 | 0.0003092 |
| dot_general_batch_c64_s32 | pytorch-cuda | 64 | 15 | 3.135 | 2.791e-07 |
| dot_general_multiple_f32_s32 | pytorch-cuda | 32 | 15 | 2.608 | 0.000269 |
| dot_general_multiple_f64_s32 | pytorch-cuda | 32 | 15 | 2.661 | 2.466e-07 |
| dot_general_multiple_c32_s32 | pytorch-cuda | 32 | 15 | 2.584 | 0.0004058 |
| dot_general_multiple_c64_s32 | pytorch-cuda | 32 | 15 | 2.671 | 4.129e-07 |
| dot_general_batch_f32_s256 | pytorch-cuda | 64 | 15 | 2.969 | 0.0008561 |
| dot_general_batch_f64_s256 | pytorch-cuda | 8 | 15 | 3.367 | 7.463e-07 |
| dot_general_batch_c32_s256 | pytorch-cuda | 64 | 15 | 3.475 | 0.001286 |
| dot_general_batch_c64_s256 | pytorch-cuda | 2 | 15 | 2.674 | 8.893e-07 |
| dot_general_multiple_f32_s256 | pytorch-cuda | 32 | 15 | 2.642 | 0.00133 |
| dot_general_multiple_f64_s256 | pytorch-cuda | 8 | 15 | 2.719 | 2.584e-06 |
| dot_general_multiple_c32_s256 | pytorch-cuda | 32 | 15 | 2.870 | 0.001059 |
| dot_general_multiple_c64_s256 | pytorch-cuda | 2 | 15 | 2.036 | 2.776e-06 |
| dot_conj_l0r0_c32_n1024 | pytorch-cuda | 128 | 15 | 1.980 | 0.0001028 |
| dot_conj_l1r0_c32_n1024 | pytorch-cuda | 128 | 15 | 1.985 | 0.0004339 |
| dot_conj_l0r1_c32_n1024 | pytorch-cuda | 128 | 15 | 1.983 | 0.0004457 |
| dot_conj_l1r1_c32_n1024 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n1024 | pytorch-cuda | 128 | 15 | 1.955 | 4.333e-07 |
| dot_conj_l1r0_c64_n1024 | pytorch-cuda | 128 | 15 | 1.974 | 2.122e-07 |
| dot_conj_l0r1_c64_n1024 | pytorch-cuda | 128 | 15 | 1.994 | 2.847e-07 |
| dot_conj_l1r1_c64_n1024 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c32_n262144 | pytorch-cuda | 128 | 15 | 2.236 | 0.003079 |
| dot_conj_l1r0_c32_n262144 | pytorch-cuda | 128 | 15 | 2.226 | 0.006155 |
| dot_conj_l0r1_c32_n262144 | pytorch-cuda | 128 | 15 | 2.241 | 0.006499 |
| dot_conj_l1r1_c32_n262144 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | 64 | 15 | 2.170 | 1.455e-05 |
| dot_conj_l1r0_c64_n262144 | pytorch-cuda | 64 | 15 | 2.172 | 2.05e-05 |
| dot_conj_l0r1_c64_n262144 | pytorch-cuda | 64 | 15 | 2.164 | 1.888e-05 |
| dot_conj_l1r1_c64_n262144 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| einsum_plan_owned_f32_s32 | pytorch-cuda | 128 | 15 | 3.040 | 0.0002018 |
| einsum_plan_read_f32_s32 | pytorch-cuda | 128 | 15 | 3.026 | 0.0002018 |
| einsum_plan_strided_read_f32_s32 | pytorch-cuda | 128 | 15 | 3.254 | 0.0002018 |
| einsum_plan_into_f32_s32 | pytorch-cuda | 128 | 15 | 2.583 | 0.0002018 |
| einsum_plan_read_into_f32_s32 | pytorch-cuda | 128 | 15 | 2.475 | 0.0002018 |
| einsum_plan_read_into_accum_f32_s32 | pytorch-cuda | 128 | 15 | 3.386 | 0.0001134 |
| einsum_plan_strided_read_into_f32_s32 | pytorch-cuda | 128 | 15 | 2.721 | 0.0002018 |
| einsum_plan_strided_read_into_accum_f32_s32 | pytorch-cuda | 128 | 15 | 3.678 | 0.0001134 |
| einsum_plan_owned_f64_s32 | pytorch-cuda | 128 | 15 | 3.266 | 0 |
| einsum_plan_read_f64_s32 | pytorch-cuda | 128 | 15 | 3.271 | 0 |
| einsum_plan_strided_read_f64_s32 | pytorch-cuda | 128 | 15 | 3.410 | 0 |
| einsum_plan_into_f64_s32 | pytorch-cuda | 128 | 15 | 2.737 | 0 |
| einsum_plan_read_into_f64_s32 | pytorch-cuda | 128 | 15 | 2.608 | 0 |
| einsum_plan_read_into_accum_f64_s32 | pytorch-cuda | 128 | 15 | 3.554 | 0 |
| einsum_plan_strided_read_into_f64_s32 | pytorch-cuda | 128 | 15 | 2.773 | 0 |
| einsum_plan_strided_read_into_accum_f64_s32 | pytorch-cuda | 128 | 15 | 3.822 | 0 |
| einsum_plan_owned_c32_s32 | pytorch-cuda | 128 | 15 | 2.967 | 0.0002342 |
| einsum_plan_read_c32_s32 | pytorch-cuda | 128 | 15 | 2.994 | 0.0002342 |
| einsum_plan_strided_read_c32_s32 | pytorch-cuda | 128 | 15 | 3.013 | 0.0002342 |
| einsum_plan_into_c32_s32 | pytorch-cuda | 128 | 15 | 2.502 | 0.0002342 |
| einsum_plan_read_into_c32_s32 | pytorch-cuda | 128 | 15 | 2.339 | 0.0002342 |
| einsum_plan_read_into_accum_c32_s32 | pytorch-cuda | 128 | 15 | 3.376 | 0.0001157 |
| einsum_plan_strided_read_into_c32_s32 | pytorch-cuda | 128 | 15 | 2.402 | 0.0002342 |
| einsum_plan_strided_read_into_accum_c32_s32 | pytorch-cuda | 64 | 15 | 1.707 | 0.0001157 |
| einsum_plan_owned_c64_s32 | pytorch-cuda | 128 | 15 | 3.186 | 1.981e-07 |
| einsum_plan_read_c64_s32 | pytorch-cuda | 128 | 15 | 3.204 | 1.981e-07 |
| einsum_plan_strided_read_c64_s32 | pytorch-cuda | 128 | 15 | 3.400 | 1.981e-07 |
| einsum_plan_into_c64_s32 | pytorch-cuda | 128 | 15 | 2.566 | 1.981e-07 |
| einsum_plan_read_into_c64_s32 | pytorch-cuda | 128 | 15 | 2.656 | 1.981e-07 |
| einsum_plan_read_into_accum_c64_s32 | pytorch-cuda | 64 | 15 | 1.818 | 1.097e-07 |
| einsum_plan_strided_read_into_c64_s32 | pytorch-cuda | 128 | 15 | 2.886 | 1.981e-07 |
| einsum_plan_strided_read_into_accum_c64_s32 | pytorch-cuda | 128 | 15 | 3.788 | 1.097e-07 |
| einsum_plan_owned_f32_s256 | pytorch-cuda | 128 | 15 | 3.236 | 0.0009425 |
| einsum_plan_read_f32_s256 | pytorch-cuda | 128 | 15 | 3.139 | 0.0009425 |
| einsum_plan_strided_read_f32_s256 | pytorch-cuda | 128 | 15 | 3.422 | 0.0009425 |
| einsum_plan_into_f32_s256 | pytorch-cuda | 128 | 15 | 2.459 | 0.0009425 |
| einsum_plan_read_into_f32_s256 | pytorch-cuda | 128 | 15 | 2.411 | 0.0009425 |
| einsum_plan_read_into_accum_f32_s256 | pytorch-cuda | 128 | 15 | 3.422 | 0.0003475 |
| einsum_plan_strided_read_into_f32_s256 | pytorch-cuda | 128 | 15 | 2.716 | 0.0009425 |
| einsum_plan_strided_read_into_accum_f32_s256 | pytorch-cuda | 128 | 15 | 3.722 | 0.0003475 |
| einsum_plan_owned_f64_s256 | pytorch-cuda | 16 | 15 | 2.682 | 0 |
| einsum_plan_read_f64_s256 | pytorch-cuda | 16 | 15 | 2.681 | 0 |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | 16 | 15 | 2.636 | 0 |
| einsum_plan_into_f64_s256 | pytorch-cuda | 16 | 15 | 2.673 | 0 |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | 16 | 15 | 2.661 | 0 |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | 16 | 15 | 2.712 | 0 |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | 16 | 15 | 2.628 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | 16 | 15 | 2.625 | 0 |
| einsum_plan_owned_c32_s256 | pytorch-cuda | 128 | 15 | 2.971 | 0.001598 |
| einsum_plan_read_c32_s256 | pytorch-cuda | 128 | 15 | 2.991 | 0.001598 |
| einsum_plan_strided_read_c32_s256 | pytorch-cuda | 128 | 15 | 3.174 | 0.001598 |
| einsum_plan_into_c32_s256 | pytorch-cuda | 128 | 15 | 2.357 | 0.001598 |
| einsum_plan_read_into_c32_s256 | pytorch-cuda | 128 | 15 | 2.341 | 0.001598 |
| einsum_plan_read_into_accum_c32_s256 | pytorch-cuda | 128 | 15 | 3.323 | 0.0006293 |
| einsum_plan_strided_read_into_c32_s256 | pytorch-cuda | 128 | 15 | 2.661 | 0.001598 |
| einsum_plan_strided_read_into_accum_c32_s256 | pytorch-cuda | 128 | 15 | 3.614 | 0.0006293 |
| einsum_plan_owned_c64_s256 | pytorch-cuda | 8 | 15 | 3.952 | 8.397e-07 |
| einsum_plan_read_c64_s256 | pytorch-cuda | 8 | 15 | 3.951 | 8.397e-07 |
| einsum_plan_strided_read_c64_s256 | pytorch-cuda | 8 | 15 | 3.952 | 8.397e-07 |
| einsum_plan_into_c64_s256 | pytorch-cuda | 8 | 15 | 3.945 | 8.397e-07 |
| einsum_plan_read_into_c64_s256 | pytorch-cuda | 8 | 15 | 3.947 | 8.397e-07 |
| einsum_plan_read_into_accum_c64_s256 | pytorch-cuda | 4 | 15 | 2.013 | 4.364e-07 |
| einsum_plan_strided_read_into_c64_s256 | pytorch-cuda | 8 | 15 | 3.947 | 8.397e-07 |
| einsum_plan_strided_read_into_accum_c64_s256 | pytorch-cuda | 4 | 15 | 2.012 | 4.364e-07 |
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
| tensordot_f32_n1024 | jax-cuda | 16 | 15 | 2.719 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda | 16 | 15 | 2.755 | 0 |
| tensordot_c32_n1024 | jax-cuda | 16 | 15 | 3.004 | 0.0005344 |
| tensordot_c64_n1024 | jax-cuda | 16 | 15 | 3.400 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda | 32 | 15 | 2.236 | 0 |
| vdot_read_f64_n1024 | jax-cuda | 32 | 15 | 2.773 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda | 32 | 15 | 2.971 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda | 16 | 15 | 2.044 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda | 16 | 15 | 2.048 | 0 |
| norm_squared_read_f64_n1024 | jax-cuda | 32 | 15 | 2.897 | 0 |
| tensordot_f32_n262144 | jax-cuda | 16 | 15 | 3.267 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda | 2 | 15 | 3.333 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda | 8 | 15 | 2.956 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda | 1 | 15 | 5.865 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda | 16 | 15 | 3.134 | 0 |
| vdot_read_f64_n262144 | jax-cuda | 16 | 15 | 2.920 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda | 16 | 15 | 3.103 | 0 |
| vdot_read_c64_n262144 | jax-cuda | 16 | 15 | 3.261 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda | 16 | 15 | 2.217 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda | 32 | 15 | 3.645 | 1.664e-07 |
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
| add_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.488 | 0 |
| add_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 3.062 | 0 |
| mul_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.546 | 0 |
| mul_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.350 | 0 |
| neg_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.062 | 0 |
| neg_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.070 | 0 |
| exp_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.022 | 0.0001272 |
| exp_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.261 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.286 | 9.325e-05 |
| tanh_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.482 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 3.011 | 0.0002267 |
| reduce_sum_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.757 | 3.268e-07 |
| conj_strided_read_c32_n1024 | jax-cuda | 16 | 15 | 2.604 | 0 |
| conj_strided_read_c64_n1024 | jax-cuda | 16 | 15 | 2.682 | 0 |
| add_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.596 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.418 | 0 |
| mul_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.159 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.209 | 0 |
| neg_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.093 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.141 | 0 |
| exp_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.347 | 0.0001272 |
| exp_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.179 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.048 | 9.508e-05 |
| tanh_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.958 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.209 | 0.003094 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.264 | 7.159e-06 |
| conj_strided_read_c32_n1048576 | jax-cuda | 16 | 15 | 2.130 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 2.099 | 0 |
| axpby_overwrite_f32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1024 | jax-cuda | 8 | 15 | 2.814 | 0 |
| axpby_overwrite_f64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1024 | jax-cuda | 8 | 15 | 2.809 | 0 |
| axpby_overwrite_c32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1024 | jax-cuda | 8 | 15 | 2.793 | 0 |
| axpby_overwrite_c64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1024 | jax-cuda | 8 | 15 | 2.785 | 0 |
| axpby_overwrite_f32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1048576 | jax-cuda | 8 | 15 | 2.910 | 0 |
| axpby_overwrite_f64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1048576 | jax-cuda | 8 | 15 | 2.940 | 0 |
| axpby_overwrite_c32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1048576 | jax-cuda | 8 | 15 | 3.001 | 0 |
| axpby_overwrite_c64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1048576 | jax-cuda | 8 | 15 | 3.021 | 0 |
| dot_general_batch_f32_s32 | jax-cuda | 16 | 15 | 1.999 | 0.0002227 |
| dot_general_batch_f64_s32 | jax-cuda | 32 | 15 | 3.833 | 1.326e-07 |
| dot_general_batch_c32_s32 | jax-cuda | 16 | 15 | 2.446 | 0.0004354 |
| dot_general_batch_c64_s32 | jax-cuda | 16 | 15 | 2.644 | 2.524e-07 |
| dot_general_multiple_f32_s32 | jax-cuda | 16 | 15 | 2.335 | 0.000269 |
| dot_general_multiple_f64_s32 | jax-cuda | 16 | 15 | 2.062 | 2.466e-07 |
| dot_general_multiple_c32_s32 | jax-cuda | 16 | 15 | 2.725 | 0.0004771 |
| dot_general_multiple_c64_s32 | jax-cuda | 16 | 15 | 3.032 | 6.063e-07 |
| dot_general_batch_f32_s256 | jax-cuda | 16 | 15 | 2.009 | 0.0008561 |
| dot_general_batch_f64_s256 | jax-cuda | 8 | 15 | 2.610 | 7.463e-07 |
| dot_general_batch_c32_s256 | jax-cuda | 8 | 15 | 2.059 | 0.001359 |
| dot_general_batch_c64_s256 | jax-cuda | 2 | 15 | 2.858 | 8.493e-07 |
| dot_general_multiple_f32_s256 | jax-cuda | 16 | 15 | 2.600 | 0.0025 |
| dot_general_multiple_f64_s256 | jax-cuda | 16 | 15 | 3.510 | 2.584e-06 |
| dot_general_multiple_c32_s256 | jax-cuda | 8 | 15 | 2.563 | 0.002465 |
| dot_general_multiple_c64_s256 | jax-cuda | 2 | 15 | 2.305 | 1.805e-06 |
| dot_conj_l0r0_c32_n1024 | jax-cuda | 16 | 15 | 2.650 | 0.0002298 |
| dot_conj_l1r0_c32_n1024 | jax-cuda | 16 | 15 | 2.584 | 0.0002522 |
| dot_conj_l0r1_c32_n1024 | jax-cuda | 32 | 15 | 3.372 | 0.0002522 |
| dot_conj_l1r1_c32_n1024 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n1024 | jax-cuda | 32 | 15 | 3.833 | 3.644e-07 |
| dot_conj_l1r0_c64_n1024 | jax-cuda | 32 | 15 | 3.380 | 2.767e-07 |
| dot_conj_l0r1_c64_n1024 | jax-cuda | 32 | 15 | 3.320 | 2.767e-07 |
| dot_conj_l1r1_c64_n1024 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c32_n262144 | jax-cuda | 32 | 15 | 3.939 | 0.01276 |
| dot_conj_l1r0_c32_n262144 | jax-cuda | 32 | 15 | 3.356 | 0.0002732 |
| dot_conj_l0r1_c32_n262144 | jax-cuda | 32 | 15 | 3.333 | 0.0002732 |
| dot_conj_l1r1_c32_n262144 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda | 16 | 15 | 2.323 | 1.221e-05 |
| dot_conj_l1r0_c64_n262144 | jax-cuda | 16 | 15 | 2.033 | 1.586e-05 |
| dot_conj_l0r1_c64_n262144 | jax-cuda | 32 | 15 | 3.431 | 1.586e-05 |
| dot_conj_l1r1_c64_n262144 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| einsum_plan_owned_f32_s32 | jax-cuda | 32 | 15 | 2.987 | 0.0002018 |
| einsum_plan_read_f32_s32 | jax-cuda | 32 | 15 | 3.000 | 0.0002018 |
| einsum_plan_strided_read_f32_s32 | jax-cuda | 32 | 15 | 2.953 | 0.0002018 |
| einsum_plan_into_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f64_s32 | jax-cuda | 32 | 15 | 2.926 | 0 |
| einsum_plan_read_f64_s32 | jax-cuda | 32 | 15 | 2.918 | 0 |
| einsum_plan_strided_read_f64_s32 | jax-cuda | 16 | 15 | 2.355 | 0 |
| einsum_plan_into_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c32_s32 | jax-cuda | 16 | 15 | 2.241 | 0.0002532 |
| einsum_plan_read_c32_s32 | jax-cuda | 16 | 15 | 1.947 | 0.0002532 |
| einsum_plan_strided_read_c32_s32 | jax-cuda | 16 | 15 | 2.979 | 0.0002532 |
| einsum_plan_into_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c64_s32 | jax-cuda | 16 | 15 | 2.259 | 3.753e-07 |
| einsum_plan_read_c64_s32 | jax-cuda | 16 | 15 | 1.966 | 3.753e-07 |
| einsum_plan_strided_read_c64_s32 | jax-cuda | 16 | 15 | 2.951 | 3.753e-07 |
| einsum_plan_into_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f32_s256 | jax-cuda | 32 | 15 | 2.407 | 0.00118 |
| einsum_plan_read_f32_s256 | jax-cuda | 32 | 15 | 2.425 | 0.00118 |
| einsum_plan_strided_read_f32_s256 | jax-cuda | 32 | 15 | 2.633 | 0.00118 |
| einsum_plan_into_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f64_s256 | jax-cuda | 32 | 15 | 3.464 | 0 |
| einsum_plan_read_f64_s256 | jax-cuda | 32 | 15 | 3.476 | 0 |
| einsum_plan_strided_read_f64_s256 | jax-cuda | 16 | 15 | 2.552 | 0 |
| einsum_plan_into_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c32_s256 | jax-cuda | 16 | 15 | 2.132 | 0.002017 |
| einsum_plan_read_c32_s256 | jax-cuda | 16 | 15 | 1.959 | 0.002017 |
| einsum_plan_strided_read_c32_s256 | jax-cuda | 16 | 15 | 2.606 | 0.002017 |
| einsum_plan_into_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c64_s256 | jax-cuda | 8 | 15 | 3.217 | 1.855e-06 |
| einsum_plan_read_c64_s256 | jax-cuda | 8 | 15 | 3.218 | 1.855e-06 |
| einsum_plan_strided_read_c64_s256 | jax-cuda | 8 | 15 | 3.283 | 1.855e-06 |
| einsum_plan_into_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
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
| tensordot_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.767 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.180 | 0 |
| tensordot_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.535 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.736 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.028 | 0 |
| vdot_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.488 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.532 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.738 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.068 | 0 |
| norm_squared_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.491 | 0 |
| tensordot_f32_n262144 | jax-cuda-jit | 32 | 15 | 2.707 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda-jit | 2 | 15 | 3.201 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.577 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda-jit | 1 | 15 | 5.350 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.675 | 0 |
| vdot_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.599 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.648 | 0 |
| vdot_read_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.569 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.497 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.394 | 1.664e-07 |
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
| add_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.084 | 0 |
| add_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.470 | 0 |
| mul_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.088 | 0 |
| mul_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.903 | 0 |
| neg_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.122 | 0 |
| neg_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.960 | 0 |
| exp_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.022 | 0.0001272 |
| exp_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.055 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.895 | 9.325e-05 |
| tanh_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.898 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.587 | 0.0002267 |
| reduce_sum_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.363 | 3.268e-07 |
| conj_strided_read_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.486 | 0 |
| conj_strided_read_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.300 | 0 |
| add_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.193 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.912 | 0 |
| mul_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.951 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.813 | 0 |
| neg_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.622 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.940 | 0 |
| exp_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.883 | 0.0001272 |
| exp_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.280 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.846 | 9.508e-05 |
| tanh_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.047 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.795 | 0.002078 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.316 | 7.159e-06 |
| conj_strided_read_c32_n1048576 | jax-cuda-jit | 32 | 15 | 2.999 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.858 | 0 |
| axpby_overwrite_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1024 | jax-cuda-jit | 8 | 15 | 1.931 | 0 |
| axpby_overwrite_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1024 | jax-cuda-jit | 8 | 15 | 1.857 | 0 |
| axpby_overwrite_c32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1024 | jax-cuda-jit | 8 | 15 | 2.030 | 0 |
| axpby_overwrite_c64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1024 | jax-cuda-jit | 8 | 15 | 1.987 | 0 |
| axpby_overwrite_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1048576 | jax-cuda-jit | 8 | 15 | 1.929 | 0 |
| axpby_overwrite_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1048576 | jax-cuda-jit | 8 | 15 | 1.965 | 0 |
| axpby_overwrite_c32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1048576 | jax-cuda-jit | 8 | 15 | 1.950 | 0 |
| axpby_overwrite_c64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1048576 | jax-cuda-jit | 8 | 15 | 2.003 | 0 |
| dot_general_batch_f32_s32 | jax-cuda-jit | 32 | 15 | 2.251 | 0.0002227 |
| dot_general_batch_f64_s32 | jax-cuda-jit | 32 | 15 | 2.114 | 1.326e-07 |
| dot_general_batch_c32_s32 | jax-cuda-jit | 32 | 15 | 2.872 | 0.0003092 |
| dot_general_batch_c64_s32 | jax-cuda-jit | 32 | 15 | 3.172 | 2.524e-07 |
| dot_general_multiple_f32_s32 | jax-cuda-jit | 32 | 15 | 3.042 | 0.000269 |
| dot_general_multiple_f64_s32 | jax-cuda-jit | 32 | 15 | 3.159 | 2.466e-07 |
| dot_general_multiple_c32_s32 | jax-cuda-jit | 16 | 15 | 2.461 | 0.0004771 |
| dot_general_multiple_c64_s32 | jax-cuda-jit | 16 | 15 | 2.595 | 6.063e-07 |
| dot_general_batch_f32_s256 | jax-cuda-jit | 32 | 15 | 2.513 | 0.0008561 |
| dot_general_batch_f64_s256 | jax-cuda-jit | 8 | 15 | 2.532 | 7.463e-07 |
| dot_general_batch_c32_s256 | jax-cuda-jit | 16 | 15 | 2.613 | 0.001359 |
| dot_general_batch_c64_s256 | jax-cuda-jit | 2 | 15 | 2.735 | 8.493e-07 |
| dot_general_multiple_f32_s256 | jax-cuda-jit | 32 | 15 | 3.888 | 0.0025 |
| dot_general_multiple_f64_s256 | jax-cuda-jit | 16 | 15 | 3.395 | 2.584e-06 |
| dot_general_multiple_c32_s256 | jax-cuda-jit | 16 | 15 | 3.032 | 0.002465 |
| dot_general_multiple_c64_s256 | jax-cuda-jit | 2 | 15 | 2.119 | 1.805e-06 |
| dot_conj_l0r0_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.844 | 0.0002298 |
| dot_conj_l1r0_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.671 | 0.0002522 |
| dot_conj_l0r1_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.903 | 0.0002522 |
| dot_conj_l1r1_c32_n1024 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.886 | 3.644e-07 |
| dot_conj_l1r0_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.964 | 2.767e-07 |
| dot_conj_l0r1_c64_n1024 | jax-cuda-jit | 16 | 15 | 3.066 | 2.767e-07 |
| dot_conj_l1r1_c64_n1024 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c32_n262144 | jax-cuda-jit | 16 | 15 | 3.094 | 0.01276 |
| dot_conj_l1r0_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.636 | 0.0002732 |
| dot_conj_l0r1_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.535 | 0.0002732 |
| dot_conj_l1r1_c32_n262144 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.463 | 1.221e-05 |
| dot_conj_l1r0_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.778 | 1.586e-05 |
| dot_conj_l0r1_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.818 | 1.586e-05 |
| dot_conj_l1r1_c64_n262144 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| einsum_plan_owned_f32_s32 | jax-cuda-jit | 32 | 15 | 3.333 | 0.0002018 |
| einsum_plan_read_f32_s32 | jax-cuda-jit | 32 | 15 | 2.927 | 0.0002018 |
| einsum_plan_strided_read_f32_s32 | jax-cuda-jit | 32 | 15 | 2.784 | 0.0002018 |
| einsum_plan_into_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f64_s32 | jax-cuda-jit | 32 | 15 | 2.634 | 0 |
| einsum_plan_read_f64_s32 | jax-cuda-jit | 32 | 15 | 2.660 | 0 |
| einsum_plan_strided_read_f64_s32 | jax-cuda-jit | 32 | 15 | 2.143 | 0 |
| einsum_plan_into_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c32_s32 | jax-cuda-jit | 32 | 15 | 2.548 | 0.0002532 |
| einsum_plan_read_c32_s32 | jax-cuda-jit | 32 | 15 | 2.553 | 0.0002532 |
| einsum_plan_strided_read_c32_s32 | jax-cuda-jit | 32 | 15 | 3.146 | 0.0002532 |
| einsum_plan_into_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c64_s32 | jax-cuda-jit | 32 | 15 | 2.776 | 3.753e-07 |
| einsum_plan_read_c64_s32 | jax-cuda-jit | 32 | 15 | 2.766 | 3.753e-07 |
| einsum_plan_strided_read_c64_s32 | jax-cuda-jit | 16 | 15 | 1.902 | 3.753e-07 |
| einsum_plan_into_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f32_s256 | jax-cuda-jit | 32 | 15 | 2.304 | 0.00118 |
| einsum_plan_read_f32_s256 | jax-cuda-jit | 32 | 15 | 2.297 | 0.00118 |
| einsum_plan_strided_read_f32_s256 | jax-cuda-jit | 32 | 15 | 2.303 | 0.00118 |
| einsum_plan_into_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f64_s256 | jax-cuda-jit | 32 | 15 | 3.491 | 0 |
| einsum_plan_read_f64_s256 | jax-cuda-jit | 32 | 15 | 3.454 | 0 |
| einsum_plan_strided_read_f64_s256 | jax-cuda-jit | 32 | 15 | 3.476 | 0 |
| einsum_plan_into_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c32_s256 | jax-cuda-jit | 32 | 15 | 2.769 | 0.002017 |
| einsum_plan_read_c32_s256 | jax-cuda-jit | 32 | 15 | 2.611 | 0.002017 |
| einsum_plan_strided_read_c32_s256 | jax-cuda-jit | 16 | 15 | 1.805 | 0.002017 |
| einsum_plan_into_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c64_s256 | jax-cuda-jit | 8 | 15 | 3.348 | 1.855e-06 |
| einsum_plan_read_c64_s256 | jax-cuda-jit | 8 | 15 | 3.489 | 1.855e-06 |
| einsum_plan_strided_read_c64_s256 | jax-cuda-jit | 8 | 15 | 3.497 | 1.855e-06 |
| einsum_plan_into_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: benchmarks/gpu/reuse_contraction.yaml
timestamp: '2026-10-13T19:15:56Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: d1413daf4d0b8289c71c88ee5a86bf38f4c05091
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
  expected_cases: 248
  selected_cases: 248
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: e77bc557a55496d5e6a806a97bd96201aefe59ce0f495ff81041cde417509347
    src/bin/benchmark_gpu_public_api/plan.rs: 3d0f10f390ebd0e08a40deb5a64e49e086dcd41b9a777d3a208c6846ed6b2fac
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/multi.rs: 1b477fef78260cf28aa31e515d57ab411f6241b91ffb30d625939538fdbc6bab
    src/bin/benchmark_gpu_public_api/read_layout.rs: 184d4508d0729f531ee8107e3b70aa6b77c5a9e952c2dc43606a868cd97631a2
    src/bin/benchmark_gpu_public_api/structural.rs: 10e3645d2027fb543f7d8134a00390dc985904e29743f65f4777c3fe533614b3
    src/bin/benchmark_gpu_public_api/linalg.rs: 5f5c4afc161f3d6e267861511d0d5871be9192d9968c95b59a73c9dc5a64e22b
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: 1dc14ac94fac700aa696d0ada04315685e3245738f9514de3d7989b8cb59152b
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 3faaaf5ae74a2ef9c91888b8fd8e566d1cf83b24d2b385cc1cd7b87d9d80c6fb
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 8653471e75fc408e8382470f31455e42d78ddc6f4d982087ba91c130026bd9d3
    scripts/run_gpu_public_api.sh: dd5e4cc47e85fcb5afdc56d460136b2a399a6a7d6a62005de8ba94afb0119ecc
    data/results/nvidia-gpu/gpu/reuse_contraction/20261013_191556/fixtures/manifest.json: 423fb21f2d48263df6204ca9d596f75a186f6e7ce183905cafeea892082bd8d1
    target/release/benchmark_gpu_public_api: 344477d9397419a46ef770502449c9ede75f091c022b514088415770b059ce78
  reference_numpy_version: 2.4.2
```
