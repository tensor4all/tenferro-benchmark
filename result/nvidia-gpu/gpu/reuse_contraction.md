# gpu/reuse_contraction: CUDA public API benchmark

Measured cells: **1068**. Unsupported cells: **572**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261013_202222/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

Unit-coefficient AXPBY uses native JAX Ref.addupdate with caller-owned fixed buffers. Its synchronization includes one GPU scalar completion token per destination and block_until_ready; tokens are retained through clock stop and included in the logical retention bound. Helper compilation and seed restoration are untimed. Full Ref snapshots are used only for untimed inspection. This scalar-token synchronization cost is included in the reported latency; it performs no full output copy or D2H transfer inside timing.

Generalized batch dot preserves native output order: tenferro [m,n,b], PyTorch/JAX [b,m,n]. Input permutation views and dimension-number configurations are prepared before timing. Output axes are normalized only on downloaded CPU arrays for verification; no timed output transpose or copy is added.

Prepared einsum uses the public ConcreteEinsumPlan for ik,kj->ij, built once outside execution and sampling. Native matrix multiplication/reuse/accumulation references match the selected mathematical and output-ownership contracts; they do not assert an equivalent prepared-plan interface. Transposed input views/values, read/write handles and accumulation seed restoration are untimed.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | 7.229 (0.161) | unsupported | 6.550 (0.233) | unsupported | unsupported |
| add_into_f64_n1024 | 6.791 (0.981) | unsupported | 7.151 (0.143) | unsupported | unsupported |
| add_read_into_f32_n1024 | 5.534 (0.319) | unsupported | 6.742 (0.219) | unsupported | unsupported |
| add_read_into_f64_n1024 | 5.388 (0.072) | unsupported | 6.667 (0.238) | unsupported | unsupported |
| sub_into_f32_n1024 | 7.304 (0.935) | unsupported | 6.726 (0.312) | unsupported | unsupported |
| sub_into_f64_n1024 | 7.149 (0.830) | unsupported | 6.836 (0.197) | unsupported | unsupported |
| sub_read_into_f32_n1024 | 5.447 (0.077) | unsupported | 6.800 (0.299) | unsupported | unsupported |
| sub_read_into_f64_n1024 | 5.385 (0.091) | unsupported | 6.753 (0.232) | unsupported | unsupported |
| mul_into_f32_n1024 | 7.297 (0.893) | unsupported | 7.546 (0.480) | unsupported | unsupported |
| mul_into_f64_n1024 | 6.866 (0.555) | unsupported | 7.259 (0.550) | unsupported | unsupported |
| mul_read_into_f32_n1024 | 5.465 (0.092) | unsupported | 7.352 (0.560) | unsupported | unsupported |
| mul_read_into_f64_n1024 | 5.458 (0.026) | unsupported | 6.736 (0.244) | unsupported | unsupported |
| div_into_f32_n1024 | 51.886 (9.715) | unsupported | 6.623 (0.279) | unsupported | unsupported |
| div_into_f64_n1024 | 47.040 (0.431) | unsupported | 6.814 (0.234) | unsupported | unsupported |
| div_read_into_f32_n1024 | 41.637 (1.090) | unsupported | 6.692 (0.377) | unsupported | unsupported |
| div_read_into_f64_n1024 | 42.231 (0.879) | unsupported | 6.764 (0.338) | unsupported | unsupported |
| neg_into_f32_n1024 | 7.210 (0.261) | unsupported | 6.252 (0.197) | unsupported | unsupported |
| neg_into_f64_n1024 | 5.763 (0.446) | unsupported | 6.182 (0.262) | unsupported | unsupported |
| neg_read_into_f32_n1024 | 4.785 (0.073) | unsupported | 6.273 (0.091) | unsupported | unsupported |
| neg_read_into_f64_n1024 | 4.947 (0.350) | unsupported | 6.160 (0.188) | unsupported | unsupported |
| conj_into_c32_n1024 | 45.229 (0.424) | unsupported | 7.280 (0.145) | unsupported | unsupported |
| conj_into_c64_n1024 | 197.125 (16.203) | unsupported | 7.356 (0.384) | unsupported | unsupported |
| conj_read_into_c32_n1024 | 38.873 (0.911) | unsupported | 7.307 (0.499) | unsupported | unsupported |
| conj_read_into_c64_n1024 | 160.737 (5.344) | unsupported | 7.108 (0.035) | unsupported | unsupported |
| add_into_f32_n1048576 | 44.889 (0.620) | unsupported | 39.743 (0.038) | unsupported | unsupported |
| add_into_f64_n1048576 | 84.630 (0.669) | unsupported | 77.670 (0.032) | unsupported | unsupported |
| add_read_into_f32_n1048576 | 42.848 (0.236) | unsupported | 39.751 (0.032) | unsupported | unsupported |
| add_read_into_f64_n1048576 | 82.722 (0.658) | unsupported | 77.586 (0.069) | unsupported | unsupported |
| sub_into_f32_n1048576 | 42.950 (0.446) | unsupported | 39.644 (0.056) | unsupported | unsupported |
| sub_into_f64_n1048576 | 82.740 (0.635) | unsupported | 77.578 (0.102) | unsupported | unsupported |
| sub_read_into_f32_n1048576 | 42.948 (0.226) | unsupported | 39.649 (0.032) | unsupported | unsupported |
| sub_read_into_f64_n1048576 | 82.580 (0.523) | unsupported | 77.547 (0.053) | unsupported | unsupported |
| mul_into_f32_n1048576 | 42.811 (0.267) | unsupported | 39.659 (0.032) | unsupported | unsupported |
| mul_into_f64_n1048576 | 82.775 (0.566) | unsupported | 77.523 (0.088) | unsupported | unsupported |
| mul_read_into_f32_n1048576 | 42.867 (0.190) | unsupported | 39.850 (0.129) | unsupported | unsupported |
| mul_read_into_f64_n1048576 | 83.482 (0.584) | unsupported | 77.518 (0.040) | unsupported | unsupported |
| div_into_f32_n1048576 | 70.156 (0.109) | unsupported | 39.867 (0.069) | unsupported | unsupported |
| div_into_f64_n1048576 | 151.429 (0.172) | unsupported | 93.931 (0.105) | unsupported | unsupported |
| div_read_into_f32_n1048576 | 70.165 (0.230) | unsupported | 39.921 (0.050) | unsupported | unsupported |
| div_read_into_f64_n1048576 | 151.289 (0.199) | unsupported | 93.972 (0.112) | unsupported | unsupported |
| neg_into_f32_n1048576 | 29.654 (0.171) | unsupported | 27.193 (0.041) | unsupported | unsupported |
| neg_into_f64_n1048576 | 57.163 (0.247) | unsupported | 52.823 (0.039) | unsupported | unsupported |
| neg_read_into_f32_n1048576 | 29.720 (0.078) | unsupported | 27.222 (0.166) | unsupported | unsupported |
| neg_read_into_f64_n1048576 | 57.676 (0.488) | unsupported | 52.819 (0.047) | unsupported | unsupported |
| conj_into_c32_n1048576 | 107.928 (0.121) | unsupported | 52.820 (0.105) | unsupported | unsupported |
| conj_into_c64_n1048576 | 209.249 (0.160) | unsupported | 103.759 (0.266) | unsupported | unsupported |
| conj_read_into_c32_n1048576 | 107.738 (0.355) | unsupported | 52.789 (0.049) | unsupported | unsupported |
| conj_read_into_c64_n1048576 | 208.904 (0.393) | unsupported | 103.679 (0.109) | unsupported | unsupported |
| tensordot_f32_n1024 | 27.334 (4.813) | 38.305 (1.764) | 50.093 (1.320) | 264.705 (6.118) | 100.246 (7.056) |
| tensordot_f64_n1024 | 39.845 (2.179) | 42.693 (1.484) | 53.642 (1.352) | 166.457 (11.426) | 71.836 (0.967) |
| tensordot_c32_n1024 | 35.730 (0.234) | 39.725 (1.616) | 50.361 (0.859) | 183.406 (6.410) | 85.269 (1.201) |
| tensordot_c64_n1024 | 133.513 (1.424) | 42.109 (1.204) | 54.830 (0.877) | 186.650 (9.209) | 96.860 (3.080) |
| vdot_read_f32_n1024 | 150.002 (2.728) | unsupported | 15.788 (0.200) | 104.718 (12.884) | 94.722 (2.233) |
| vdot_read_f64_n1024 | 149.887 (0.505) | unsupported | 15.514 (0.133) | 91.653 (9.693) | 80.490 (1.116) |
| vdot_read_c32_n1024 | 150.050 (2.334) | unsupported | 15.099 (0.236) | 84.188 (2.324) | 96.342 (4.554) |
| vdot_read_c64_n1024 | 172.863 (1.074) | unsupported | 15.383 (0.138) | 120.590 (14.841) | 109.642 (18.062) |
| norm_squared_read_f32_n1024 | 160.772 (1.665) | unsupported | 15.647 (0.123) | 85.357 (9.745) | 146.943 (14.051) |
| norm_squared_read_f64_n1024 | 149.225 (1.825) | unsupported | 15.781 (0.714) | 74.794 (5.442) | 132.740 (6.889) |
| tensordot_f32_n262144 | 47.703 (0.261) | 46.807 (1.227) | 62.268 (2.757) | 163.846 (2.631) | 90.340 (9.845) |
| tensordot_f64_n262144 | 1620.364 (1.259) | 1630.637 (2.234) | 1628.357 (5.165) | 1701.892 (28.517) | 1570.333 (41.085) |
| tensordot_c32_n262144 | 141.914 (1.879) | 142.341 (0.988) | 159.194 (8.296) | 327.250 (16.311) | 170.052 (15.561) |
| tensordot_c64_n262144 | 5032.362 (2.774) | 5066.478 (2.478) | 5258.257 (4.906) | 5643.406 (44.203) | 5404.210 (23.924) |
| vdot_read_f32_n262144 | 184.330 (6.212) | unsupported | 16.594 (0.911) | 161.830 (25.045) | 169.856 (14.099) |
| vdot_read_f64_n262144 | 174.049 (0.594) | unsupported | 20.228 (0.015) | 160.708 (3.194) | 163.234 (5.915) |
| vdot_read_c32_n262144 | 173.463 (1.693) | unsupported | 17.448 (0.127) | 167.121 (10.199) | 176.704 (20.664) |
| vdot_read_c64_n262144 | 174.434 (0.789) | unsupported | 33.751 (0.024) | 155.293 (2.494) | 152.076 (5.423) |
| norm_squared_read_f32_n262144 | 171.862 (1.751) | unsupported | 15.726 (0.208) | 139.321 (25.281) | 174.342 (12.749) |
| norm_squared_read_f64_n262144 | 171.710 (1.918) | unsupported | 16.148 (0.538) | 121.203 (15.418) | 166.350 (3.955) |
| axpby_read_into_accum_f32_n1024 | 9.812 (0.158) | unsupported | 7.515 (0.192) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1024 | 10.346 (0.389) | unsupported | 7.786 (0.544) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1024 | 10.033 (0.283) | unsupported | 7.660 (0.120) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1024 | 10.114 (0.578) | unsupported | 7.782 (0.680) | unsupported | unsupported |
| axpby_read_into_accum_f32_n1048576 | 273.466 (0.345) | unsupported | 39.717 (0.080) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1048576 | 534.765 (0.693) | unsupported | 77.679 (0.036) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1048576 | 168.901 (0.268) | unsupported | 77.631 (0.065) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1048576 | 2144.375 (1.174) | unsupported | 153.460 (0.130) | unsupported | unsupported |
| dot_general_read_into_f32_s16 | 22.330 (0.465) | unsupported | 18.605 (0.129) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s16 | 23.135 (1.081) | unsupported | 22.528 (0.505) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s16 | 22.770 (0.779) | unsupported | 22.365 (0.268) | unsupported | unsupported |
| dot_general_read_into_f64_s16 | 27.598 (0.846) | unsupported | 20.899 (0.658) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s16 | 26.342 (0.484) | unsupported | 24.893 (0.828) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s16 | 26.606 (1.269) | unsupported | 24.937 (0.825) | unsupported | unsupported |
| dot_general_read_into_c32_s16 | 22.199 (0.462) | unsupported | 21.783 (1.795) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s16 | 22.095 (0.447) | unsupported | 26.417 (1.520) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s16 | 22.227 (0.491) | unsupported | 24.614 (0.322) | unsupported | unsupported |
| dot_general_read_into_c64_s16 | 26.998 (0.934) | unsupported | 21.772 (0.547) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s16 | 24.806 (0.533) | unsupported | 25.708 (0.427) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s16 | 24.279 (0.475) | unsupported | 25.931 (0.304) | unsupported | unsupported |
| dot_general_read_into_f32_s256 | 23.004 (1.181) | unsupported | 19.054 (0.185) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s256 | 22.605 (0.555) | unsupported | 22.309 (0.471) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s256 | 23.843 (1.283) | unsupported | 22.473 (0.299) | unsupported | unsupported |
| dot_general_read_into_f64_s256 | 262.887 (0.212) | unsupported | 248.242 (5.254) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s256 | 260.343 (10.694) | unsupported | 270.026 (10.049) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s256 | 260.081 (13.910) | unsupported | 279.880 (14.937) | unsupported | unsupported |
| dot_general_read_into_c32_s256 | 28.297 (0.310) | unsupported | 26.822 (0.060) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s256 | 29.856 (0.098) | unsupported | 27.776 (1.117) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s256 | 29.809 (0.035) | unsupported | 28.497 (1.477) | unsupported | unsupported |
| dot_general_read_into_c64_s256 | 683.214 (0.466) | unsupported | 975.577 (0.441) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s256 | 684.419 (0.107) | unsupported | 981.851 (0.598) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s256 | 684.492 (0.357) | unsupported | 981.670 (0.170) | unsupported | unsupported |
| add_strided_read_f32_n1024 | 281.693 (9.380) | unsupported | 10.588 (0.677) | 162.905 (30.353) | 173.660 (21.376) |
| add_strided_read_f64_n1024 | 245.091 (22.601) | unsupported | 10.453 (0.340) | 172.708 (33.507) | 156.446 (17.125) |
| mul_strided_read_f32_n1024 | 330.078 (4.062) | unsupported | 10.376 (0.302) | 147.964 (27.295) | 144.043 (21.594) |
| mul_strided_read_f64_n1024 | 307.438 (2.618) | unsupported | 10.392 (0.289) | 145.239 (22.006) | 129.480 (13.616) |
| neg_strided_read_f32_n1024 | 191.764 (2.985) | unsupported | 11.253 (1.771) | 132.601 (10.820) | 121.938 (14.218) |
| neg_strided_read_f64_n1024 | 172.451 (4.576) | unsupported | 9.936 (0.768) | 128.482 (12.318) | 120.508 (14.573) |
| exp_strided_read_f32_n1024 | 197.949 (10.752) | unsupported | 9.169 (0.358) | 102.103 (16.659) | 123.720 (10.196) |
| exp_strided_read_f64_n1024 | 169.227 (3.648) | unsupported | 11.210 (1.048) | 92.623 (16.292) | 132.044 (8.703) |
| tanh_strided_read_f32_n1024 | 194.133 (10.104) | unsupported | 9.949 (0.605) | 93.853 (8.689) | 129.673 (7.698) |
| tanh_strided_read_f64_n1024 | 178.223 (8.916) | unsupported | 9.772 (0.618) | 102.275 (12.961) | 134.586 (8.091) |
| reduce_sum_strided_read_f32_n1024 | 333.512 (5.629) | unsupported | 15.678 (1.033) | 158.434 (10.250) | 154.291 (12.489) |
| reduce_sum_strided_read_f64_n1024 | 295.116 (5.278) | unsupported | 15.678 (1.174) | 185.273 (11.672) | 174.348 (12.505) |
| conj_strided_read_c32_n1024 | 162.503 (6.547) | unsupported | 11.860 (0.284) | 158.094 (18.023) | 165.952 (37.510) |
| conj_strided_read_c64_n1024 | 235.359 (5.521) | unsupported | 12.307 (0.580) | 166.098 (28.790) | 152.207 (25.180) |
| add_strided_read_f32_n1048576 | 97.936 (0.500) | unsupported | 39.790 (0.121) | 122.246 (3.174) | 127.379 (10.045) |
| add_strided_read_f64_n1048576 | 188.814 (0.115) | unsupported | 77.795 (0.232) | 123.086 (4.661) | 105.369 (7.448) |
| mul_strided_read_f32_n1048576 | 98.222 (0.197) | unsupported | 39.784 (0.088) | 111.440 (1.022) | 99.022 (0.565) |
| mul_strided_read_f64_n1048576 | 188.969 (0.166) | unsupported | 77.755 (0.124) | 107.564 (2.215) | 93.253 (0.353) |
| neg_strided_read_f32_n1048576 | 58.860 (0.763) | unsupported | 27.337 (0.120) | 103.881 (2.381) | 95.526 (0.622) |
| neg_strided_read_f64_n1048576 | 108.445 (0.070) | unsupported | 53.090 (0.192) | 106.309 (0.546) | 98.716 (0.553) |
| exp_strided_read_f32_n1048576 | 56.760 (1.262) | unsupported | 27.417 (0.046) | 94.209 (0.554) | 95.691 (0.879) |
| exp_strided_read_f64_n1048576 | 233.598 (0.189) | unsupported | 181.276 (0.084) | 203.522 (2.311) | 203.002 (2.840) |
| tanh_strided_read_f32_n1048576 | 56.405 (0.189) | unsupported | 27.424 (0.030) | 93.997 (3.127) | 81.918 (9.659) |
| tanh_strided_read_f64_n1048576 | 220.444 (0.088) | unsupported | 167.929 (0.117) | 191.543 (1.731) | 187.899 (2.544) |
| reduce_sum_strided_read_f32_n1048576 | 260.447 (12.249) | unsupported | 21.620 (0.689) | 99.766 (0.680) | 124.956 (2.083) |
| reduce_sum_strided_read_f64_n1048576 | 92.980 (0.273) | unsupported | 36.984 (0.137) | 160.653 (4.863) | 148.140 (1.219) |
| conj_strided_read_c32_n1048576 | 114.419 (0.163) | unsupported | 53.022 (0.057) | 98.276 (3.370) | 103.003 (5.374) |
| conj_strided_read_c64_n1048576 | 214.139 (0.526) | unsupported | 104.099 (0.035) | 116.097 (2.530) | 117.810 (1.663) |
| axpby_overwrite_f32_n1024 | 10.125 (0.135) | unsupported | 11.797 (0.096) | unsupported | unsupported |
| axpby_lerp_f32_n1024 | 10.300 (0.425) | unsupported | 9.032 (0.153) | unsupported | unsupported |
| axpby_unit_add_f32_n1024 | 10.272 (0.124) | unsupported | 7.602 (0.112) | 371.358 (58.176) | 246.903 (13.153) |
| axpby_overwrite_f64_n1024 | 10.396 (0.242) | unsupported | 11.171 (0.216) | unsupported | unsupported |
| axpby_lerp_f64_n1024 | 10.944 (0.509) | unsupported | 9.010 (0.219) | unsupported | unsupported |
| axpby_unit_add_f64_n1024 | 10.499 (0.233) | unsupported | 7.832 (0.059) | 354.758 (17.225) | 250.228 (33.572) |
| axpby_overwrite_c32_n1024 | 10.241 (0.542) | unsupported | 11.831 (0.294) | unsupported | unsupported |
| axpby_lerp_c32_n1024 | 11.062 (1.098) | unsupported | 7.914 (0.211) | unsupported | unsupported |
| axpby_unit_add_c32_n1024 | 10.645 (0.343) | unsupported | 8.298 (0.347) | 365.464 (46.924) | 256.248 (30.105) |
| axpby_overwrite_c64_n1024 | 10.658 (0.929) | unsupported | 11.682 (0.192) | unsupported | unsupported |
| axpby_lerp_c64_n1024 | 10.337 (0.871) | unsupported | 10.831 (0.004) | unsupported | unsupported |
| axpby_unit_add_c64_n1024 | 10.284 (0.230) | unsupported | 7.788 (0.121) | 365.502 (26.366) | 266.048 (37.185) |
| axpby_overwrite_f32_n1048576 | 304.878 (1.078) | unsupported | 27.370 (0.034) | unsupported | unsupported |
| axpby_lerp_f32_n1048576 | 273.231 (0.270) | unsupported | 39.769 (0.026) | unsupported | unsupported |
| axpby_unit_add_f32_n1048576 | 273.293 (0.474) | unsupported | 39.719 (0.034) | 368.308 (28.825) | 258.411 (14.279) |
| axpby_overwrite_f64_n1048576 | 339.214 (0.873) | unsupported | 53.181 (0.213) | unsupported | unsupported |
| axpby_lerp_f64_n1048576 | 534.848 (0.215) | unsupported | 77.983 (0.074) | unsupported | unsupported |
| axpby_unit_add_f64_n1048576 | 534.933 (0.321) | unsupported | 77.675 (0.067) | 372.168 (25.857) | 248.933 (8.033) |
| axpby_overwrite_c32_n1048576 | 156.683 (0.706) | unsupported | 53.111 (0.065) | unsupported | unsupported |
| axpby_lerp_c32_n1048576 | 168.870 (0.127) | unsupported | 77.917 (0.048) | unsupported | unsupported |
| axpby_unit_add_c32_n1048576 | 168.922 (0.081) | unsupported | 77.641 (0.057) | 390.207 (18.403) | 250.136 (9.261) |
| axpby_overwrite_c64_n1048576 | 1271.869 (2.393) | unsupported | 104.405 (0.081) | unsupported | unsupported |
| axpby_lerp_c64_n1048576 | 2016.922 (0.894) | unsupported | 340.508 (0.181) | unsupported | unsupported |
| axpby_unit_add_c64_n1048576 | 2016.996 (1.505) | unsupported | 153.434 (0.128) | 379.782 (20.310) | 251.610 (18.820) |
| dot_general_batch_f32_s32 | 204.996 (15.494) | 47.786 (3.182) | 39.563 (0.392) | 128.374 (7.802) | 70.627 (0.971) |
| dot_general_batch_f64_s32 | 184.912 (5.469) | 50.836 (1.273) | 45.739 (2.320) | 125.612 (3.480) | 73.844 (2.170) |
| dot_general_batch_c32_s32 | 169.587 (3.419) | 50.005 (1.309) | 43.824 (1.836) | 160.066 (18.700) | 96.303 (3.707) |
| dot_general_batch_c64_s32 | 173.527 (1.782) | 53.072 (2.028) | 44.395 (0.911) | 174.528 (10.557) | 98.197 (5.357) |
| dot_general_multiple_f32_s32 | 205.256 (6.499) | 54.818 (2.777) | 76.757 (5.367) | 130.617 (5.880) | 86.362 (1.980) |
| dot_general_multiple_f64_s32 | 205.344 (1.361) | 53.051 (2.096) | 78.846 (0.575) | 161.578 (1.173) | 78.585 (1.529) |
| dot_general_multiple_c32_s32 | 203.816 (3.869) | 52.769 (2.238) | 76.200 (0.309) | 226.190 (35.613) | 101.689 (12.706) |
| dot_general_multiple_c64_s32 | 202.935 (5.178) | 53.382 (3.092) | 79.565 (0.626) | 300.262 (42.376) | 105.932 (10.966) |
| dot_general_batch_f32_s256 | 63.284 (4.350) | 40.633 (1.085) | 40.938 (1.521) | 127.994 (4.797) | 79.115 (5.575) |
| dot_general_batch_f64_s256 | 333.191 (0.854) | 333.742 (0.340) | 418.346 (0.200) | 323.057 (9.936) | 313.510 (8.343) |
| dot_general_batch_c32_s256 | 42.252 (0.356) | 41.612 (1.880) | 53.961 (0.039) | 280.941 (31.443) | 162.369 (29.195) |
| dot_general_batch_c64_s256 | 1104.938 (2.186) | 1109.526 (2.030) | 1329.440 (1.879) | 1466.003 (10.892) | 1390.769 (9.654) |
| dot_general_multiple_f32_s256 | 50.611 (0.195) | 48.693 (1.816) | 75.875 (0.405) | 332.950 (8.585) | 104.711 (0.245) |
| dot_general_multiple_f64_s256 | 332.583 (1.236) | 334.116 (0.291) | 338.095 (0.243) | 218.073 (4.343) | 211.381 (2.282) |
| dot_general_multiple_c32_s256 | 47.458 (7.770) | 47.552 (1.948) | 88.152 (4.500) | 328.310 (20.632) | 191.872 (30.659) |
| dot_general_multiple_c64_s256 | 825.582 (2.111) | 832.635 (2.249) | 1010.838 (0.789) | 1139.160 (12.554) | 1060.934 (22.476) |
| dot_conj_l0r0_c32_n1024 | 208.325 (1.435) | 34.434 (4.721) | 15.782 (0.149) | 86.718 (4.852) | 168.122 (8.219) |
| dot_conj_l1r0_c32_n1024 | 176.801 (28.056) | 157.445 (10.486) | 15.502 (0.549) | 80.058 (4.104) | 171.706 (11.219) |
| dot_conj_l0r1_c32_n1024 | 175.854 (16.572) | 136.885 (3.429) | 15.677 (0.245) | 73.740 (2.729) | 169.856 (12.211) |
| dot_conj_l1r1_c32_n1024 | 179.301 (3.904) | 126.516 (5.970) | unsupported | unsupported | unsupported |
| dot_conj_l0r0_c64_n1024 | 263.111 (6.028) | 37.132 (2.776) | 15.902 (0.718) | 97.630 (2.569) | 173.731 (23.321) |
| dot_conj_l1r0_c64_n1024 | 172.165 (2.175) | 147.805 (5.133) | 15.300 (0.222) | 92.985 (0.773) | 161.225 (17.623) |
| dot_conj_l0r1_c64_n1024 | 172.198 (4.092) | 142.421 (3.031) | 15.102 (0.174) | 93.997 (0.492) | 173.588 (18.682) |
| dot_conj_l1r1_c64_n1024 | 171.848 (3.157) | 137.171 (1.389) | unsupported | unsupported | unsupported |
| dot_conj_l0r0_c32_n262144 | 665.943 (8.131) | 625.881 (2.999) | 17.406 (0.040) | 138.436 (1.374) | 173.427 (27.433) |
| dot_conj_l1r0_c32_n262144 | 187.209 (4.772) | 105.822 (2.859) | 17.319 (0.018) | 117.805 (19.809) | 176.877 (27.558) |
| dot_conj_l0r1_c32_n262144 | 186.587 (3.955) | 105.173 (4.152) | 17.431 (0.048) | 102.319 (2.031) | 183.699 (17.297) |
| dot_conj_l1r1_c32_n262144 | 187.557 (6.888) | 106.148 (2.108) | unsupported | unsupported | unsupported |
| dot_conj_l0r0_c64_n262144 | 950.374 (4.016) | 909.763 (0.583) | 33.790 (0.047) | 150.941 (3.611) | 187.279 (30.821) |
| dot_conj_l1r0_c64_n262144 | 183.494 (5.658) | 106.193 (2.659) | 33.835 (0.032) | 132.887 (17.077) | 182.533 (28.936) |
| dot_conj_l0r1_c64_n262144 | 188.375 (7.988) | 105.191 (2.809) | 33.699 (0.033) | 115.739 (10.388) | 149.810 (26.394) |
| dot_conj_l1r1_c64_n262144 | 189.226 (5.768) | 106.883 (3.801) | unsupported | unsupported | unsupported |
| einsum_plan_owned_f32_s32 | 172.334 (8.220) | unsupported | 21.485 (0.138) | 85.794 (0.725) | 97.227 (3.146) |
| einsum_plan_read_f32_s32 | 172.165 (2.543) | unsupported | 21.304 (0.580) | 85.933 (1.095) | 95.428 (1.087) |
| einsum_plan_strided_read_f32_s32 | 186.312 (2.371) | unsupported | 23.471 (0.487) | 150.999 (1.588) | 95.073 (1.291) |
| einsum_plan_into_f32_s32 | 32.744 (0.626) | unsupported | 15.414 (0.151) | unsupported | unsupported |
| einsum_plan_read_into_f32_s32 | 26.309 (2.217) | unsupported | 15.454 (1.020) | unsupported | unsupported |
| einsum_plan_read_into_accum_f32_s32 | 25.859 (2.190) | unsupported | 23.353 (0.785) | unsupported | unsupported |
| einsum_plan_strided_read_into_f32_s32 | 27.553 (1.384) | unsupported | 16.990 (0.098) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_f32_s32 | 26.351 (1.895) | unsupported | 24.454 (0.221) | unsupported | unsupported |
| einsum_plan_owned_f64_s32 | 127.373 (6.121) | unsupported | 23.715 (0.415) | 86.443 (0.995) | 86.854 (1.005) |
| einsum_plan_read_f64_s32 | 148.127 (19.908) | unsupported | 23.947 (0.610) | 85.704 (1.783) | 86.443 (1.003) |
| einsum_plan_strided_read_f64_s32 | 168.789 (3.571) | unsupported | 25.603 (0.548) | 159.089 (4.849) | 88.828 (1.879) |
| einsum_plan_into_f64_s32 | 27.720 (0.869) | unsupported | 17.883 (0.857) | unsupported | unsupported |
| einsum_plan_read_into_f64_s32 | 27.714 (0.735) | unsupported | 17.919 (0.164) | unsupported | unsupported |
| einsum_plan_read_into_accum_f64_s32 | 27.766 (1.131) | unsupported | 25.014 (0.342) | unsupported | unsupported |
| einsum_plan_strided_read_into_f64_s32 | 30.179 (1.701) | unsupported | 18.443 (0.347) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_f64_s32 | 28.035 (1.798) | unsupported | 25.759 (0.729) | unsupported | unsupported |
| einsum_plan_owned_c32_s32 | 125.387 (7.492) | unsupported | 22.464 (0.809) | 154.487 (34.895) | 102.305 (0.512) |
| einsum_plan_read_c32_s32 | 135.570 (3.131) | unsupported | 22.203 (0.151) | 125.376 (3.793) | 102.489 (0.555) |
| einsum_plan_strided_read_c32_s32 | 154.395 (6.928) | unsupported | 22.933 (1.825) | 155.864 (3.904) | 132.154 (23.203) |
| einsum_plan_into_c32_s32 | 23.560 (0.566) | unsupported | 17.377 (0.963) | unsupported | unsupported |
| einsum_plan_read_into_c32_s32 | 24.599 (1.455) | unsupported | 16.816 (0.272) | unsupported | unsupported |
| einsum_plan_read_into_accum_c32_s32 | 25.746 (3.018) | unsupported | 23.385 (0.246) | unsupported | unsupported |
| einsum_plan_strided_read_into_c32_s32 | 26.037 (1.852) | unsupported | 17.115 (0.548) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_c32_s32 | 25.108 (0.832) | unsupported | 23.782 (0.366) | unsupported | unsupported |
| einsum_plan_owned_c64_s32 | 125.570 (3.104) | unsupported | 25.270 (0.629) | 131.752 (19.397) | 121.907 (3.884) |
| einsum_plan_read_c64_s32 | 125.740 (15.557) | unsupported | 25.395 (1.906) | 124.247 (3.187) | 111.454 (2.109) |
| einsum_plan_strided_read_c64_s32 | 158.776 (9.933) | unsupported | 29.206 (2.030) | 193.081 (25.820) | 135.230 (19.327) |
| einsum_plan_into_c64_s32 | 26.064 (0.412) | unsupported | 20.573 (1.741) | unsupported | unsupported |
| einsum_plan_read_into_c64_s32 | 25.717 (1.150) | unsupported | 19.295 (0.726) | unsupported | unsupported |
| einsum_plan_read_into_accum_c64_s32 | 26.354 (1.124) | unsupported | 25.844 (0.444) | unsupported | unsupported |
| einsum_plan_strided_read_into_c64_s32 | 27.744 (1.628) | unsupported | 19.759 (0.101) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_c64_s32 | 27.556 (0.911) | unsupported | 27.103 (0.467) | unsupported | unsupported |
| einsum_plan_owned_f32_s256 | 37.152 (1.865) | unsupported | 22.288 (1.050) | 74.009 (11.460) | 80.207 (1.617) |
| einsum_plan_read_f32_s256 | 36.680 (0.430) | unsupported | 22.391 (0.834) | 72.898 (3.820) | 78.238 (1.770) |
| einsum_plan_strided_read_f32_s256 | 42.811 (2.543) | unsupported | 24.131 (0.280) | 146.255 (3.001) | 79.916 (1.795) |
| einsum_plan_into_f32_s256 | 25.891 (2.321) | unsupported | 15.738 (0.513) | unsupported | unsupported |
| einsum_plan_read_into_f32_s256 | 24.965 (0.393) | unsupported | 15.894 (0.489) | unsupported | unsupported |
| einsum_plan_read_into_accum_f32_s256 | 25.907 (2.071) | unsupported | 22.341 (0.417) | unsupported | unsupported |
| einsum_plan_strided_read_into_f32_s256 | 28.460 (2.226) | unsupported | 18.052 (0.735) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_f32_s256 | 24.954 (1.716) | unsupported | 25.501 (0.974) | unsupported | unsupported |
| einsum_plan_owned_f64_s256 | 171.489 (2.818) | unsupported | 164.243 (3.365) | 108.526 (1.674) | 106.901 (0.331) |
| einsum_plan_read_f64_s256 | 172.953 (3.839) | unsupported | 164.113 (2.536) | 108.263 (0.594) | 106.504 (0.331) |
| einsum_plan_strided_read_f64_s256 | 169.652 (5.280) | unsupported | 161.962 (2.453) | 146.669 (1.909) | 106.855 (0.515) |
| einsum_plan_into_f64_s256 | 165.553 (0.349) | unsupported | 163.606 (2.533) | unsupported | unsupported |
| einsum_plan_read_into_f64_s256 | 165.794 (0.436) | unsupported | 163.659 (2.516) | unsupported | unsupported |
| einsum_plan_read_into_accum_f64_s256 | 167.160 (0.173) | unsupported | 163.509 (4.932) | unsupported | unsupported |
| einsum_plan_strided_read_into_f64_s256 | 166.017 (0.133) | unsupported | 163.226 (2.510) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_f64_s256 | 167.411 (0.117) | unsupported | 148.423 (5.021) | unsupported | unsupported |
| einsum_plan_owned_c32_s256 | 39.181 (1.451) | unsupported | 22.307 (0.243) | 112.905 (16.967) | 91.955 (3.204) |
| einsum_plan_read_c32_s256 | 38.828 (2.879) | unsupported | 22.527 (1.325) | 91.080 (7.981) | 90.793 (4.668) |
| einsum_plan_strided_read_c32_s256 | 44.859 (2.849) | unsupported | 24.301 (0.739) | 158.866 (9.795) | 123.942 (14.105) |
| einsum_plan_into_c32_s256 | 26.365 (1.256) | unsupported | 17.705 (0.625) | unsupported | unsupported |
| einsum_plan_read_into_c32_s256 | 26.498 (1.503) | unsupported | 17.491 (0.044) | unsupported | unsupported |
| einsum_plan_read_into_accum_c32_s256 | 25.963 (1.721) | unsupported | 23.511 (0.317) | unsupported | unsupported |
| einsum_plan_strided_read_into_c32_s256 | 30.189 (1.194) | unsupported | 18.377 (0.534) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_c32_s256 | 29.387 (1.064) | unsupported | 27.234 (1.415) | unsupported | unsupported |
| einsum_plan_owned_c64_s256 | 420.532 (3.766) | unsupported | 491.614 (0.433) | 396.417 (3.862) | 424.536 (13.920) |
| einsum_plan_read_c64_s256 | 422.423 (3.800) | unsupported | 491.188 (0.203) | 398.774 (3.381) | 432.076 (9.220) |
| einsum_plan_strided_read_c64_s256 | 415.749 (4.316) | unsupported | 491.507 (0.193) | 395.805 (7.914) | 408.817 (15.489) |
| einsum_plan_into_c64_s256 | 421.131 (3.935) | unsupported | 490.504 (0.160) | unsupported | unsupported |
| einsum_plan_read_into_c64_s256 | 421.173 (0.418) | unsupported | 490.728 (0.059) | unsupported | unsupported |
| einsum_plan_read_into_accum_c64_s256 | 420.801 (5.216) | unsupported | 494.870 (0.187) | unsupported | unsupported |
| einsum_plan_strided_read_into_c64_s256 | 413.357 (7.661) | unsupported | 490.760 (0.139) | unsupported | unsupported |
| einsum_plan_strided_read_into_accum_c64_s256 | 413.165 (4.296) | unsupported | 501.309 (1.910) | unsupported | unsupported |
| sub_strided_read_f32_n1024 | 316.501 (8.714) | unsupported | 11.704 (0.969) | 176.760 (37.872) | 152.569 (33.883) |
| sub_strided_read_f64_n1024 | 328.799 (46.408) | unsupported | 10.621 (0.194) | 153.139 (25.488) | 138.819 (14.462) |
| div_strided_read_f32_n1024 | 369.957 (12.132) | unsupported | 10.935 (0.872) | 123.867 (4.988) | 122.406 (15.993) |
| div_strided_read_f64_n1024 | 334.192 (13.152) | unsupported | 10.444 (0.540) | 108.342 (20.065) | 128.804 (14.279) |
| rem_strided_read_f32_n1024 | 371.343 (11.360) | unsupported | 10.182 (0.147) | 106.532 (13.981) | 106.896 (11.765) |
| rem_strided_read_f64_n1024 | 326.010 (8.443) | unsupported | 10.025 (0.290) | 123.184 (2.556) | 131.551 (11.274) |
| pow_strided_read_f32_n1024 | 368.332 (10.207) | unsupported | 10.150 (0.220) | 188.763 (30.325) | 148.074 (22.226) |
| pow_strided_read_f64_n1024 | 328.384 (13.574) | unsupported | 29.158 (0.008) | 190.984 (34.739) | 157.282 (16.652) |
| maximum_strided_read_f32_n1024 | 378.399 (16.863) | unsupported | 10.127 (0.202) | 155.649 (18.123) | 156.041 (30.957) |
| maximum_strided_read_f64_n1024 | 322.162 (13.272) | unsupported | 9.944 (0.107) | 138.801 (10.385) | 135.267 (21.629) |
| minimum_strided_read_f32_n1024 | 368.535 (7.686) | unsupported | 10.540 (0.566) | 131.759 (19.430) | 124.863 (8.732) |
| minimum_strided_read_f64_n1024 | 334.371 (10.282) | unsupported | 10.547 (0.238) | 127.891 (16.891) | 98.862 (10.374) |
| abs_strided_read_f32_n1024 | 192.201 (7.862) | unsupported | 13.585 (0.267) | 103.061 (11.850) | 94.229 (7.215) |
| abs_strided_read_f64_n1024 | 167.257 (1.930) | unsupported | 13.490 (0.291) | 99.499 (9.754) | 98.942 (7.873) |
| sign_strided_read_f32_n1024 | 191.481 (3.791) | unsupported | 9.183 (0.136) | 100.649 (15.695) | 94.464 (6.896) |
| sign_strided_read_f64_n1024 | 177.202 (2.636) | unsupported | 9.204 (0.134) | 100.450 (9.505) | 98.886 (10.851) |
| expm1_strided_read_f32_n1024 | 203.719 (12.100) | unsupported | 9.408 (0.224) | 100.257 (10.305) | 99.275 (10.718) |
| expm1_strided_read_f64_n1024 | 186.330 (5.885) | unsupported | 9.651 (0.133) | 103.262 (8.779) | 97.168 (11.637) |
| log_strided_read_f32_n1024 | 202.884 (7.967) | unsupported | 9.319 (0.545) | 102.536 (13.625) | 99.123 (14.898) |
| log_strided_read_f64_n1024 | 179.336 (3.541) | unsupported | 10.930 (0.028) | 128.510 (9.219) | 104.885 (18.277) |
| log1p_strided_read_f32_n1024 | 200.810 (5.069) | unsupported | 10.051 (0.529) | 142.520 (15.925) | 95.225 (8.331) |
| log1p_strided_read_f64_n1024 | 179.291 (5.419) | unsupported | 11.187 (0.006) | 159.720 (19.642) | 127.454 (7.690) |
| erf_strided_read_f32_n1024 | 205.719 (5.903) | unsupported | 9.395 (0.072) | 250.086 (3.162) | 153.612 (29.817) |
| erf_strided_read_f64_n1024 | 179.398 (5.847) | unsupported | 15.683 (0.005) | 248.884 (5.447) | 151.248 (24.785) |
| sin_strided_read_f32_n1024 | 202.807 (6.514) | unsupported | 9.505 (0.177) | 151.217 (12.115) | 157.744 (22.714) |
| sin_strided_read_f64_n1024 | 180.856 (5.207) | unsupported | 9.475 (0.197) | 166.711 (4.869) | 163.704 (15.068) |
| cos_strided_read_f32_n1024 | 211.657 (11.713) | unsupported | 9.020 (0.080) | 170.233 (15.086) | 165.566 (22.032) |
| cos_strided_read_f64_n1024 | 178.418 (5.053) | unsupported | 9.195 (0.310) | 162.356 (7.625) | 163.083 (11.205) |
| sqrt_strided_read_f32_n1024 | 203.027 (7.313) | unsupported | 9.302 (0.105) | 161.812 (28.362) | 148.452 (26.021) |
| sqrt_strided_read_f64_n1024 | 190.634 (8.276) | unsupported | 9.399 (0.344) | 155.318 (23.690) | 135.807 (16.234) |
| rsqrt_strided_read_f32_n1024 | 202.375 (2.095) | unsupported | 9.603 (0.684) | 191.339 (55.902) | 121.867 (21.917) |
| rsqrt_strided_read_f64_n1024 | 180.178 (6.712) | unsupported | 9.337 (0.141) | 179.048 (39.350) | 102.644 (22.375) |
| reduce_prod_strided_read_f32_n1024 | 345.934 (20.935) | unsupported | 13.573 (0.323) | 166.693 (14.252) | 137.176 (3.328) |
| reduce_prod_strided_read_f64_n1024 | 335.277 (14.984) | unsupported | 13.090 (0.184) | 141.902 (7.851) | 174.014 (15.980) |
| reduce_max_strided_read_f32_n1024 | 352.020 (13.991) | unsupported | 14.775 (0.222) | 170.863 (3.289) | 164.119 (7.337) |
| reduce_max_strided_read_f64_n1024 | 335.191 (18.576) | unsupported | 14.798 (0.221) | 130.675 (3.488) | 112.405 (15.696) |
| reduce_min_strided_read_f32_n1024 | 343.073 (11.576) | unsupported | 14.763 (0.177) | 149.190 (10.640) | 101.829 (17.882) |
| reduce_min_strided_read_f64_n1024 | 331.544 (12.062) | unsupported | 15.477 (0.858) | 95.478 (2.233) | 91.349 (0.483) |
| reduce_sum_squares_strided_read_f32_n1024 | 341.912 (10.686) | unsupported | 27.517 (1.678) | 1825.010 (242.256) | 137.621 (3.150) |
| reduce_sum_squares_strided_read_f64_n1024 | 337.267 (13.559) | unsupported | 27.667 (0.388) | 1789.226 (255.072) | 118.637 (10.102) |
| sub_strided_read_f32_n1048576 | 99.459 (1.450) | unsupported | 39.751 (0.056) | 110.285 (1.145) | 99.860 (4.113) |
| sub_strided_read_f64_n1048576 | 189.746 (0.138) | unsupported | 77.752 (0.028) | 102.341 (3.828) | 89.779 (1.689) |
| div_strided_read_f32_n1048576 | 98.389 (0.177) | unsupported | 39.970 (0.020) | 98.941 (1.086) | 97.651 (0.351) |
| div_strided_read_f64_n1048576 | 205.483 (0.129) | unsupported | 94.368 (0.116) | 129.352 (2.548) | 129.718 (2.391) |
| rem_strided_read_f32_n1048576 | 98.415 (0.309) | unsupported | 39.984 (0.103) | 125.302 (14.460) | 103.085 (9.837) |
| rem_strided_read_f64_n1048576 | 189.022 (0.279) | unsupported | 78.186 (0.039) | 104.618 (12.741) | 97.005 (10.615) |
| pow_strided_read_f32_n1048576 | 97.678 (0.245) | unsupported | 41.598 (0.104) | 131.623 (17.555) | 98.821 (2.964) |
| pow_strided_read_f64_n1048576 | 1113.512 (1.109) | unsupported | 1015.219 (0.518) | 1310.798 (13.968) | 1293.271 (10.900) |
| maximum_strided_read_f32_n1048576 | 98.035 (0.679) | unsupported | 39.933 (0.197) | 122.861 (11.206) | 98.967 (4.063) |
| maximum_strided_read_f64_n1048576 | 188.969 (0.198) | unsupported | 78.461 (0.074) | 106.760 (2.911) | 92.000 (0.564) |
| minimum_strided_read_f32_n1048576 | 97.769 (0.236) | unsupported | 39.891 (0.045) | 107.624 (0.916) | 97.091 (1.048) |
| minimum_strided_read_f64_n1048576 | 188.999 (0.297) | unsupported | 78.442 (0.087) | 102.346 (3.131) | 91.205 (0.646) |
| abs_strided_read_f32_n1048576 | 56.161 (0.100) | unsupported | 27.443 (0.115) | 92.644 (0.489) | 93.997 (0.867) |
| abs_strided_read_f64_n1048576 | 107.864 (0.320) | unsupported | 53.180 (0.036) | 92.907 (1.900) | 95.995 (0.372) |
| sign_strided_read_f32_n1048576 | 56.176 (0.231) | unsupported | 27.373 (0.047) | 95.176 (0.605) | 94.319 (0.514) |
| sign_strided_read_f64_n1048576 | 107.855 (0.179) | unsupported | 53.250 (0.166) | 89.876 (4.139) | 95.136 (1.200) |
| expm1_strided_read_f32_n1048576 | 56.534 (0.279) | unsupported | 27.483 (0.026) | 94.191 (0.808) | 94.545 (0.503) |
| expm1_strided_read_f64_n1048576 | 280.485 (0.648) | unsupported | 224.196 (0.059) | 269.835 (4.476) | 264.925 (4.281) |
| log_strided_read_f32_n1048576 | 56.547 (0.314) | unsupported | 27.486 (0.046) | 95.559 (7.119) | 94.754 (1.942) |
| log_strided_read_f64_n1048576 | 404.460 (0.705) | unsupported | 349.674 (0.111) | 397.294 (12.728) | 403.942 (7.440) |
| log1p_strided_read_f32_n1048576 | 56.611 (0.293) | unsupported | 27.469 (0.028) | 95.800 (3.350) | 95.552 (2.905) |
| log1p_strided_read_f64_n1048576 | 416.865 (0.966) | unsupported | 361.793 (0.136) | 407.997 (13.160) | 421.288 (5.639) |
| erf_strided_read_f32_n1048576 | 56.390 (0.207) | unsupported | 27.448 (0.058) | 144.525 (10.043) | 100.919 (5.262) |
| erf_strided_read_f64_n1048576 | 583.933 (0.663) | unsupported | 528.709 (0.182) | 675.572 (7.597) | 662.626 (15.373) |
| sin_strided_read_f32_n1048576 | 56.483 (0.127) | unsupported | 27.567 (0.045) | 120.175 (8.126) | 99.250 (9.081) |
| sin_strided_read_f64_n1048576 | 253.346 (0.457) | unsupported | 211.516 (0.128) | 238.030 (2.450) | 237.441 (0.579) |
| cos_strided_read_f32_n1048576 | 56.425 (0.355) | unsupported | 27.719 (0.181) | 101.597 (9.802) | 92.318 (2.476) |
| cos_strided_read_f64_n1048576 | 253.085 (0.819) | unsupported | 211.062 (0.060) | 237.765 (1.147) | 252.104 (2.775) |
| sqrt_strided_read_f32_n1048576 | 56.307 (0.185) | unsupported | 27.437 (0.028) | 93.955 (1.494) | 93.034 (2.597) |
| sqrt_strided_read_f64_n1048576 | 148.072 (0.598) | unsupported | 91.877 (0.152) | 102.910 (1.141) | 102.990 (0.929) |
| rsqrt_strided_read_f32_n1048576 | 56.197 (0.184) | unsupported | 27.397 (0.047) | 134.684 (1.050) | 91.887 (2.746) |
| rsqrt_strided_read_f64_n1048576 | 115.344 (0.212) | unsupported | 61.503 (0.063) | 137.575 (3.515) | 88.101 (4.645) |
| reduce_prod_strided_read_f32_n1048576 | 198.079 (10.490) | unsupported | 19.722 (0.333) | 148.385 (1.777) | 127.901 (1.894) |
| reduce_prod_strided_read_f64_n1048576 | 92.857 (0.235) | unsupported | 36.415 (0.050) | 169.250 (2.690) | 156.768 (3.188) |
| reduce_max_strided_read_f32_n1048576 | 212.272 (14.960) | unsupported | 21.581 (0.651) | 102.024 (1.714) | 150.444 (2.797) |
| reduce_max_strided_read_f64_n1048576 | 123.111 (0.266) | unsupported | 42.267 (0.244) | 157.812 (2.882) | 157.173 (4.155) |
| reduce_min_strided_read_f32_n1048576 | 208.178 (13.924) | unsupported | 23.878 (1.543) | 95.228 (2.062) | 155.471 (10.324) |
| reduce_min_strided_read_f64_n1048576 | 123.116 (0.194) | unsupported | 42.343 (0.256) | 165.600 (8.288) | 159.840 (5.199) |
| reduce_sum_squares_strided_read_f32_n1048576 | 209.887 (12.128) | unsupported | 45.151 (0.188) | 2388.683 (63.812) | 89.752 (2.707) |
| reduce_sum_squares_strided_read_f64_n1048576 | 94.771 (0.628) | unsupported | 90.321 (0.515) | 2290.031 (17.528) | 147.886 (3.278) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.701 | 0 |
| add_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 3.477 | 0 |
| add_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.833 | 0 |
| add_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.759 | 0 |
| sub_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.740 | 0 |
| sub_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 3.660 | 0 |
| sub_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.789 | 0 |
| sub_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.757 | 0 |
| mul_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.736 | 0 |
| mul_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 3.516 | 0 |
| mul_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.798 | 0 |
| mul_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.795 | 0 |
| div_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.321 | 0 |
| div_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.011 | 0 |
| div_read_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.665 | 0 |
| div_read_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.703 | 0 |
| neg_into_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 1.846 | 0 |
| neg_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.950 | 0 |
| neg_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.450 | 0 |
| neg_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.533 | 0 |
| conj_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.895 | 0 |
| conj_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.154 | 0 |
| conj_read_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.488 | 0 |
| conj_read_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.572 | 0 |
| add_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.873 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.708 | 0 |
| add_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.742 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.647 | 0 |
| sub_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.749 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.648 | 0 |
| sub_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.749 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.643 | 0 |
| mul_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.740 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.649 | 0 |
| mul_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.743 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.671 | 0 |
| div_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.245 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.423 | 0 |
| div_read_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.245 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.421 | 0 |
| neg_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.898 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.829 | 0 |
| neg_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.902 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.846 | 0 |
| conj_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.454 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.348 | 0 |
| conj_read_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.448 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.342 | 0 |
| tensordot_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.499 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.550 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.287 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.136 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.400 | 0 |
| vdot_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.398 | 1.722e-07 |
| vdot_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.401 | 0 |
| vdot_read_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.766 | 3.455e-07 |
| norm_squared_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.572 | 0 |
| norm_squared_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.388 | 0 |
| tensordot_f32_n262144 | tenferro-cuda-direct | 64 | 15 | 3.053 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 2 | 15 | 3.241 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.271 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-direct | 1 | 15 | 5.032 | 1.446e-05 |
| vdot_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.949 | 0 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.785 | 2.995e-06 |
| vdot_read_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.775 | 0 |
| vdot_read_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.791 | 8.762e-06 |
| norm_squared_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.750 | 0.0001786 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.747 | 0 |
| axpby_read_into_accum_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.512 | 0 |
| axpby_read_into_accum_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.648 | 0 |
| axpby_read_into_accum_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.569 | 0 |
| axpby_read_into_accum_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.589 | 0 |
| axpby_read_into_accum_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.188 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.139 | 0 |
| axpby_read_into_accum_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.702 | 0 |
| axpby_read_into_accum_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.144 | 0 |
| dot_general_read_into_f32_s16 | tenferro-cuda-direct | 128 | 15 | 2.858 | 8.645e-05 |
| dot_general_read_into_accum_f32_s16 | tenferro-cuda-direct | 128 | 15 | 2.961 | 7.907e-05 |
| dot_general_read_into_accum_cached_f32_s16 | tenferro-cuda-direct | 128 | 15 | 2.915 | 7.907e-05 |
| dot_general_read_into_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.533 | 0 |
| dot_general_read_into_accum_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.372 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.406 | 0 |
| dot_general_read_into_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.841 | 0.0003615 |
| dot_general_read_into_accum_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.828 | 0.0001632 |
| dot_general_read_into_accum_cached_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.845 | 0.0001632 |
| dot_general_read_into_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.456 | 1.204e-07 |
| dot_general_read_into_accum_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.175 | 6.975e-08 |
| dot_general_read_into_accum_cached_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.108 | 6.975e-08 |
| dot_general_read_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.944 | 0.0005542 |
| dot_general_read_into_accum_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.893 | 0.0003617 |
| dot_general_read_into_accum_cached_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.052 | 0.0003617 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.103 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.083 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.081 | 0 |
| dot_general_read_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.622 | 0.001457 |
| dot_general_read_into_accum_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.822 | 0.0008947 |
| dot_general_read_into_accum_cached_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.816 | 0.0008947 |
| dot_general_read_into_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.733 | 2.092e-06 |
| dot_general_read_into_accum_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.738 | 1.318e-06 |
| dot_general_read_into_accum_cached_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.738 | 1.318e-06 |
| add_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.254 | 0 |
| add_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.921 | 0 |
| mul_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.641 | 0 |
| mul_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.460 | 0 |
| neg_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.068 | 0 |
| neg_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.759 | 0 |
| exp_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.167 | 0.0001272 |
| exp_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.708 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.106 | 9.107e-05 |
| tanh_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.852 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.668 | 0.0001208 |
| reduce_sum_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.361 | 4.202e-07 |
| conj_strided_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.600 | 0 |
| conj_strided_read_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.766 | 0 |
| add_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.134 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.021 | 0 |
| mul_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.143 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.024 | 0 |
| neg_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.767 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.470 | 0 |
| exp_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.633 | 0.0001272 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.738 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.610 | 9.135e-05 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.527 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.084 | 0.002489 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.975 | 3.227e-06 |
| conj_strided_read_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.661 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.426 | 0 |
| axpby_overwrite_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.592 | 0 |
| axpby_lerp_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.637 | 0 |
| axpby_unit_add_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.630 | 0 |
| axpby_overwrite_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.661 | 0 |
| axpby_lerp_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.802 | 1.155e-07 |
| axpby_unit_add_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.688 | 0 |
| axpby_overwrite_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.622 | 0 |
| axpby_lerp_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.832 | 0 |
| axpby_unit_add_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.725 | 0 |
| axpby_overwrite_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.728 | 0 |
| axpby_lerp_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.646 | 1.155e-07 |
| axpby_unit_add_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.633 | 0 |
| axpby_overwrite_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.439 | 0 |
| axpby_lerp_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.186 | 0 |
| axpby_unit_add_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.186 | 0 |
| axpby_overwrite_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.714 | 0 |
| axpby_lerp_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.139 | 1.16e-07 |
| axpby_unit_add_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.140 | 0 |
| axpby_overwrite_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.507 | 0 |
| axpby_lerp_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.702 | 0 |
| axpby_unit_add_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.703 | 0 |
| axpby_overwrite_c64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.544 | 0 |
| axpby_lerp_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.017 | 1.159e-07 |
| axpby_unit_add_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.017 | 0 |
| dot_general_batch_f32_s32 | tenferro-cuda-direct | 16 | 15 | 3.280 | 0.0002227 |
| dot_general_batch_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.959 | 1.326e-07 |
| dot_general_batch_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.713 | 0.0004376 |
| dot_general_batch_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.776 | 4.296e-07 |
| dot_general_multiple_f32_s32 | tenferro-cuda-direct | 16 | 15 | 3.284 | 0.0002162 |
| dot_general_multiple_f64_s32 | tenferro-cuda-direct | 16 | 15 | 3.286 | 1.632e-07 |
| dot_general_multiple_c32_s32 | tenferro-cuda-direct | 16 | 15 | 3.261 | 0.0002794 |
| dot_general_multiple_c64_s32 | tenferro-cuda-direct | 16 | 15 | 3.247 | 4.303e-07 |
| dot_general_batch_f32_s256 | tenferro-cuda-direct | 32 | 15 | 2.025 | 0.0008561 |
| dot_general_batch_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.666 | 7.463e-07 |
| dot_general_batch_c32_s256 | tenferro-cuda-direct | 64 | 15 | 2.704 | 0.001749 |
| dot_general_batch_c64_s256 | tenferro-cuda-direct | 2 | 15 | 2.210 | 1.665e-06 |
| dot_general_multiple_f32_s256 | tenferro-cuda-direct | 64 | 15 | 3.239 | 0.002418 |
| dot_general_multiple_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.661 | 1.519e-06 |
| dot_general_multiple_c32_s256 | tenferro-cuda-direct | 64 | 15 | 3.037 | 0.0022 |
| dot_general_multiple_c64_s256 | tenferro-cuda-direct | 4 | 15 | 3.302 | 2.187e-06 |
| dot_conj_l0r0_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.333 | 0.0004142 |
| dot_conj_l1r0_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.829 | 0.0002905 |
| dot_conj_l0r1_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.814 | 0.0002905 |
| dot_conj_l1r1_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.869 | 0.0003083 |
| dot_conj_l0r0_c64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.105 | 3.737e-07 |
| dot_conj_l1r0_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.755 | 1.807e-07 |
| dot_conj_l0r1_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.755 | 1.807e-07 |
| dot_conj_l1r1_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.750 | 4.927e-07 |
| dot_conj_l0r0_c32_n262144 | tenferro-cuda-direct | 4 | 15 | 2.664 | 0.02908 |
| dot_conj_l1r0_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.995 | 0.005619 |
| dot_conj_l0r1_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.985 | 0.005619 |
| dot_conj_l1r1_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 3.001 | 0.005822 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-direct | 4 | 15 | 3.801 | 1.986e-05 |
| dot_conj_l1r0_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.936 | 1.516e-05 |
| dot_conj_l0r1_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.014 | 1.516e-05 |
| dot_conj_l1r1_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.028 | 1.212e-05 |
| einsum_plan_owned_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.757 | 0.0002018 |
| einsum_plan_read_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.755 | 0.0002018 |
| einsum_plan_strided_read_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.981 | 0.0002018 |
| einsum_plan_into_f32_s32 | tenferro-cuda-direct | 64 | 15 | 2.096 | 0.0002018 |
| einsum_plan_read_into_f32_s32 | tenferro-cuda-direct | 128 | 15 | 3.368 | 0.0002018 |
| einsum_plan_read_into_accum_f32_s32 | tenferro-cuda-direct | 128 | 15 | 3.310 | 0.0001134 |
| einsum_plan_strided_read_into_f32_s32 | tenferro-cuda-direct | 128 | 15 | 3.527 | 0.0002018 |
| einsum_plan_strided_read_into_accum_f32_s32 | tenferro-cuda-direct | 128 | 15 | 3.373 | 0.0001134 |
| einsum_plan_owned_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.038 | 0 |
| einsum_plan_read_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.370 | 0 |
| einsum_plan_strided_read_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.701 | 0 |
| einsum_plan_into_f64_s32 | tenferro-cuda-direct | 128 | 15 | 3.548 | 0 |
| einsum_plan_read_into_f64_s32 | tenferro-cuda-direct | 64 | 15 | 1.774 | 0 |
| einsum_plan_read_into_accum_f64_s32 | tenferro-cuda-direct | 128 | 15 | 3.554 | 0 |
| einsum_plan_strided_read_into_f64_s32 | tenferro-cuda-direct | 128 | 15 | 3.863 | 0 |
| einsum_plan_strided_read_into_accum_f64_s32 | tenferro-cuda-direct | 128 | 15 | 3.588 | 0 |
| einsum_plan_owned_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.006 | 0.0004148 |
| einsum_plan_read_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.169 | 0.0004148 |
| einsum_plan_strided_read_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.470 | 0.0004148 |
| einsum_plan_into_c32_s32 | tenferro-cuda-direct | 128 | 15 | 3.016 | 0.0004148 |
| einsum_plan_read_into_c32_s32 | tenferro-cuda-direct | 128 | 15 | 3.149 | 0.0004148 |
| einsum_plan_read_into_accum_c32_s32 | tenferro-cuda-direct | 128 | 15 | 3.296 | 0.0001409 |
| einsum_plan_strided_read_into_c32_s32 | tenferro-cuda-direct | 128 | 15 | 3.333 | 0.0004148 |
| einsum_plan_strided_read_into_accum_c32_s32 | tenferro-cuda-direct | 128 | 15 | 3.214 | 0.0001409 |
| einsum_plan_owned_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.009 | 1.978e-07 |
| einsum_plan_read_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.012 | 1.978e-07 |
| einsum_plan_strided_read_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.540 | 1.978e-07 |
| einsum_plan_into_c64_s32 | tenferro-cuda-direct | 128 | 15 | 3.336 | 1.978e-07 |
| einsum_plan_read_into_c64_s32 | tenferro-cuda-direct | 128 | 15 | 3.292 | 1.978e-07 |
| einsum_plan_read_into_accum_c64_s32 | tenferro-cuda-direct | 128 | 15 | 3.373 | 1.074e-07 |
| einsum_plan_strided_read_into_c64_s32 | tenferro-cuda-direct | 64 | 15 | 1.776 | 1.978e-07 |
| einsum_plan_strided_read_into_accum_c64_s32 | tenferro-cuda-direct | 128 | 15 | 3.527 | 1.074e-07 |
| einsum_plan_owned_f32_s256 | tenferro-cuda-direct | 64 | 15 | 2.378 | 0.0009425 |
| einsum_plan_read_f32_s256 | tenferro-cuda-direct | 64 | 15 | 2.347 | 0.0009425 |
| einsum_plan_strided_read_f32_s256 | tenferro-cuda-direct | 64 | 15 | 2.740 | 0.0009425 |
| einsum_plan_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.314 | 0.0009425 |
| einsum_plan_read_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.195 | 0.0009425 |
| einsum_plan_read_into_accum_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.316 | 0.0003475 |
| einsum_plan_strided_read_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.643 | 0.0009425 |
| einsum_plan_strided_read_into_accum_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.194 | 0.0003475 |
| einsum_plan_owned_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.744 | 0 |
| einsum_plan_read_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.767 | 0 |
| einsum_plan_strided_read_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.714 | 0 |
| einsum_plan_into_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.649 | 0 |
| einsum_plan_read_into_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.653 | 0 |
| einsum_plan_read_into_accum_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.675 | 0 |
| einsum_plan_strided_read_into_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.656 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | tenferro-cuda-direct | 16 | 15 | 2.679 | 0 |
| einsum_plan_owned_c32_s256 | tenferro-cuda-direct | 64 | 15 | 2.508 | 0.001996 |
| einsum_plan_read_c32_s256 | tenferro-cuda-direct | 64 | 15 | 2.485 | 0.001996 |
| einsum_plan_strided_read_c32_s256 | tenferro-cuda-direct | 64 | 15 | 2.871 | 0.001996 |
| einsum_plan_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.375 | 0.001996 |
| einsum_plan_read_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.392 | 0.001996 |
| einsum_plan_read_into_accum_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.323 | 0.0006165 |
| einsum_plan_strided_read_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.864 | 0.001996 |
| einsum_plan_strided_read_into_accum_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.762 | 0.0006165 |
| einsum_plan_owned_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.364 | 2.139e-06 |
| einsum_plan_read_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.379 | 2.139e-06 |
| einsum_plan_strided_read_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.326 | 2.139e-06 |
| einsum_plan_into_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.369 | 2.139e-06 |
| einsum_plan_read_into_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.369 | 2.139e-06 |
| einsum_plan_read_into_accum_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.366 | 9.602e-07 |
| einsum_plan_strided_read_into_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.307 | 2.139e-06 |
| einsum_plan_strided_read_into_accum_c64_s256 | tenferro-cuda-direct | 8 | 15 | 3.305 | 9.602e-07 |
| sub_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.532 | 0 |
| sub_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.630 | 0 |
| div_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.960 | 0 |
| div_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.674 | 0 |
| rem_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.971 | 0 |
| rem_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.608 | 0 |
| pow_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.947 | 0.000117 |
| pow_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.627 | 1.48e-07 |
| maximum_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 3.027 | 0 |
| maximum_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.577 | 0 |
| minimum_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.948 | 0 |
| minimum_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.675 | 0 |
| abs_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.075 | 0 |
| abs_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.676 | 0 |
| sign_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.064 | 0 |
| sign_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.835 | 0 |
| expm1_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.260 | 7.911e-05 |
| expm1_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.981 | 7.39e-08 |
| log_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.246 | 7.945e-05 |
| log_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.869 | 4.231e-08 |
| log1p_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.213 | 0.0001189 |
| log1p_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.869 | 1.108e-07 |
| erf_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.292 | 7.914e-05 |
| erf_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.870 | 8.536e-08 |
| sin_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.245 | 4.348e-05 |
| sin_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.894 | 4.153e-08 |
| cos_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.387 | 6.342e-05 |
| cos_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.855 | 5.905e-08 |
| sqrt_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.248 | 0 |
| sqrt_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.050 | 0 |
| rsqrt_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.238 | 6.983e-05 |
| rsqrt_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.883 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.767 | 0.0004769 |
| reduce_prod_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.682 | 5.55e-07 |
| reduce_max_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.816 | 0 |
| reduce_max_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.682 | 0 |
| reduce_min_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.745 | 0 |
| reduce_min_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.652 | 0 |
| reduce_sum_squares_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.735 | 0.0001524 |
| reduce_sum_squares_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.698 | 2.804e-07 |
| sub_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.183 | 0 |
| sub_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.036 | 0 |
| div_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.148 | 0 |
| div_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.288 | 0 |
| rem_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.149 | 0 |
| rem_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.024 | 0 |
| pow_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.126 | 0.0001577 |
| pow_strided_read_f64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.227 | 1.48e-07 |
| maximum_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.137 | 0 |
| maximum_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.024 | 0 |
| minimum_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.129 | 0 |
| minimum_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.024 | 0 |
| abs_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.594 | 0 |
| abs_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.452 | 0 |
| sign_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.595 | 0 |
| sign_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.451 | 0 |
| expm1_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.618 | 7.915e-05 |
| expm1_strided_read_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.244 | 7.39e-08 |
| log_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.619 | 7.946e-05 |
| log_strided_read_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 3.236 | 4.308e-08 |
| log1p_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.623 | 0.000119 |
| log1p_strided_read_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 3.335 | 1.11e-07 |
| erf_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.609 | 7.923e-05 |
| erf_strided_read_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.336 | 8.536e-08 |
| sin_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.615 | 4.348e-05 |
| sin_strided_read_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.027 | 4.153e-08 |
| cos_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.611 | 6.342e-05 |
| cos_strided_read_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.025 | 5.905e-08 |
| sqrt_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.604 | 0 |
| sqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.369 | 0 |
| rsqrt_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.597 | 6.983e-05 |
| rsqrt_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.691 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.169 | 0.008881 |
| reduce_prod_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.971 | 4.833e-06 |
| reduce_max_strided_read_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.396 | 0 |
| reduce_max_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.940 | 0 |
| reduce_min_strided_read_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.331 | 0 |
| reduce_min_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.940 | 0 |
| reduce_sum_squares_strided_read_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.358 | 0.0001778 |
| reduce_sum_squares_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.033 | 3.312e-07 |
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
| tensordot_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.452 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.732 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.542 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.695 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| tensordot_f32_n262144 | tenferro-cuda-eager | 64 | 15 | 2.996 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 2 | 15 | 3.261 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-eager | 16 | 15 | 2.277 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-eager | 1 | 15 | 5.066 | 1.446e-05 |
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
| dot_general_batch_f32_s32 | tenferro-cuda-eager | 64 | 15 | 3.058 | 0.0002227 |
| dot_general_batch_f64_s32 | tenferro-cuda-eager | 64 | 15 | 3.254 | 1.326e-07 |
| dot_general_batch_c32_s32 | tenferro-cuda-eager | 64 | 15 | 3.200 | 0.0004376 |
| dot_general_batch_c64_s32 | tenferro-cuda-eager | 64 | 15 | 3.397 | 4.296e-07 |
| dot_general_multiple_f32_s32 | tenferro-cuda-eager | 64 | 15 | 3.508 | 0.0002162 |
| dot_general_multiple_f64_s32 | tenferro-cuda-eager | 64 | 15 | 3.395 | 1.632e-07 |
| dot_general_multiple_c32_s32 | tenferro-cuda-eager | 64 | 15 | 3.377 | 0.0002794 |
| dot_general_multiple_c64_s32 | tenferro-cuda-eager | 64 | 15 | 3.416 | 4.303e-07 |
| dot_general_batch_f32_s256 | tenferro-cuda-eager | 64 | 15 | 2.601 | 0.0008561 |
| dot_general_batch_f64_s256 | tenferro-cuda-eager | 8 | 15 | 2.670 | 7.463e-07 |
| dot_general_batch_c32_s256 | tenferro-cuda-eager | 64 | 15 | 2.663 | 0.001749 |
| dot_general_batch_c64_s256 | tenferro-cuda-eager | 2 | 15 | 2.219 | 1.665e-06 |
| dot_general_multiple_f32_s256 | tenferro-cuda-eager | 64 | 15 | 3.116 | 0.002418 |
| dot_general_multiple_f64_s256 | tenferro-cuda-eager | 8 | 15 | 2.673 | 1.519e-06 |
| dot_general_multiple_c32_s256 | tenferro-cuda-eager | 64 | 15 | 3.043 | 0.0022 |
| dot_general_multiple_c64_s256 | tenferro-cuda-eager | 4 | 15 | 3.331 | 2.187e-06 |
| dot_conj_l0r0_c32_n1024 | tenferro-cuda-eager | 128 | 15 | 4.408 | 0.0004142 |
| dot_conj_l1r0_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.519 | 0.0002905 |
| dot_conj_l0r1_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.190 | 0.0002905 |
| dot_conj_l1r1_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.024 | 0.0003083 |
| dot_conj_l0r0_c64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.376 | 3.737e-07 |
| dot_conj_l1r0_c64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.365 | 1.807e-07 |
| dot_conj_l0r1_c64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.279 | 1.807e-07 |
| dot_conj_l1r1_c64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.195 | 4.927e-07 |
| dot_conj_l0r0_c32_n262144 | tenferro-cuda-eager | 4 | 15 | 2.504 | 0.02908 |
| dot_conj_l1r0_c32_n262144 | tenferro-cuda-eager | 32 | 15 | 3.386 | 0.005619 |
| dot_conj_l0r1_c32_n262144 | tenferro-cuda-eager | 32 | 15 | 3.366 | 0.005619 |
| dot_conj_l1r1_c32_n262144 | tenferro-cuda-eager | 32 | 15 | 3.397 | 0.005822 |
| dot_conj_l0r0_c64_n262144 | tenferro-cuda-eager | 4 | 15 | 3.639 | 1.986e-05 |
| dot_conj_l1r0_c64_n262144 | tenferro-cuda-eager | 32 | 15 | 3.398 | 1.516e-05 |
| dot_conj_l0r1_c64_n262144 | tenferro-cuda-eager | 32 | 15 | 3.366 | 1.516e-05 |
| dot_conj_l1r1_c64_n262144 | tenferro-cuda-eager | 32 | 15 | 3.420 | 1.212e-05 |
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
| add_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.353 | 0 |
| add_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.662 | 0 |
| add_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.452 | 0 |
| add_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.413 | 0 |
| sub_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.444 | 0 |
| sub_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.500 | 0 |
| sub_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.482 | 0 |
| sub_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.457 | 0 |
| mul_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.864 | 0 |
| mul_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.716 | 0 |
| mul_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.764 | 0 |
| mul_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.449 | 0 |
| div_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.391 | 0 |
| div_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.489 | 0 |
| div_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.427 | 0 |
| div_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.463 | 0 |
| neg_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.201 | 0 |
| neg_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.165 | 0 |
| neg_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.212 | 0 |
| neg_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.154 | 0 |
| conj_into_c32_n1024 | pytorch-cuda | 512 | 15 | 3.728 | 0 |
| conj_into_c64_n1024 | pytorch-cuda | 512 | 15 | 3.766 | 0 |
| conj_read_into_c32_n1024 | pytorch-cuda | 512 | 15 | 3.741 | 0 |
| conj_read_into_c64_n1024 | pytorch-cuda | 256 | 15 | 1.820 | 0 |
| add_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.485 | 0 |
| add_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.483 | 0 |
| sub_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.537 | 0 |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.483 | 0 |
| sub_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.538 | 0 |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.482 | 0 |
| mul_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.538 | 0 |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.481 | 0 |
| mul_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.550 | 0 |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.481 | 0 |
| div_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.551 | 0 |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.006 | 0 |
| div_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.555 | 0 |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.007 | 0 |
| neg_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.740 | 0 |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.690 | 0 |
| neg_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.742 | 0 |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.690 | 0 |
| conj_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.690 | 0 |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.660 | 0 |
| conj_read_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.689 | 0 |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.659 | 0 |
| tensordot_f32_n1024 | pytorch-cuda | 64 | 15 | 3.206 | 0.0008859 |
| tensordot_f64_n1024 | pytorch-cuda | 64 | 15 | 3.433 | 0 |
| tensordot_c32_n1024 | pytorch-cuda | 64 | 15 | 3.223 | 0.0005344 |
| tensordot_c64_n1024 | pytorch-cuda | 64 | 15 | 3.509 | 0 |
| vdot_read_f32_n1024 | pytorch-cuda | 128 | 15 | 2.021 | 0 |
| vdot_read_f64_n1024 | pytorch-cuda | 128 | 15 | 1.986 | 1.722e-07 |
| vdot_read_c32_n1024 | pytorch-cuda | 256 | 15 | 3.865 | 0 |
| vdot_read_c64_n1024 | pytorch-cuda | 128 | 15 | 1.969 | 3.455e-07 |
| norm_squared_read_f32_n1024 | pytorch-cuda | 128 | 15 | 2.003 | 0 |
| norm_squared_read_f64_n1024 | pytorch-cuda | 256 | 15 | 4.040 | 0 |
| tensordot_f32_n262144 | pytorch-cuda | 32 | 15 | 1.993 | 0.006191 |
| tensordot_f64_n262144 | pytorch-cuda | 2 | 15 | 3.257 | 6.714e-06 |
| tensordot_c32_n262144 | pytorch-cuda | 16 | 15 | 2.547 | 0.005717 |
| tensordot_c64_n262144 | pytorch-cuda | 1 | 15 | 5.258 | 4.401e-06 |
| vdot_read_f32_n262144 | pytorch-cuda | 128 | 15 | 2.124 | 0 |
| vdot_read_f64_n262144 | pytorch-cuda | 128 | 15 | 2.589 | 2.995e-06 |
| vdot_read_c32_n262144 | pytorch-cuda | 128 | 15 | 2.233 | 0 |
| vdot_read_c64_n262144 | pytorch-cuda | 64 | 15 | 2.160 | 8.762e-06 |
| norm_squared_read_f32_n262144 | pytorch-cuda | 128 | 15 | 2.013 | 0.0001786 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 128 | 15 | 2.067 | 0 |
| axpby_read_into_accum_f32_n1024 | pytorch-cuda | 256 | 15 | 1.924 | 0 |
| axpby_read_into_accum_f64_n1024 | pytorch-cuda | 256 | 15 | 1.993 | 0 |
| axpby_read_into_accum_c32_n1024 | pytorch-cuda | 512 | 15 | 3.922 | 0 |
| axpby_read_into_accum_c64_n1024 | pytorch-cuda | 512 | 15 | 3.984 | 0 |
| axpby_read_into_accum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.542 | 0 |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| axpby_read_into_accum_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| axpby_read_into_accum_c64_n1048576 | pytorch-cuda | 16 | 15 | 2.455 | 0 |
| dot_general_read_into_f32_s16 | pytorch-cuda | 128 | 15 | 2.381 | 8.645e-05 |
| dot_general_read_into_accum_f32_s16 | pytorch-cuda | 128 | 15 | 2.884 | 7.907e-05 |
| dot_general_read_into_accum_cached_f32_s16 | pytorch-cuda | 128 | 15 | 2.863 | 7.907e-05 |
| dot_general_read_into_f64_s16 | pytorch-cuda | 128 | 15 | 2.675 | 0 |
| dot_general_read_into_accum_f64_s16 | pytorch-cuda | 128 | 15 | 3.186 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | pytorch-cuda | 128 | 15 | 3.192 | 0 |
| dot_general_read_into_c32_s16 | pytorch-cuda | 128 | 15 | 2.788 | 0.0001647 |
| dot_general_read_into_accum_c32_s16 | pytorch-cuda | 128 | 15 | 3.381 | 0.0001125 |
| dot_general_read_into_accum_cached_c32_s16 | pytorch-cuda | 128 | 15 | 3.151 | 0.0001125 |
| dot_general_read_into_c64_s16 | pytorch-cuda | 128 | 15 | 2.787 | 1.373e-07 |
| dot_general_read_into_accum_c64_s16 | pytorch-cuda | 128 | 15 | 3.291 | 7.37e-08 |
| dot_general_read_into_accum_cached_c64_s16 | pytorch-cuda | 64 | 15 | 1.660 | 7.37e-08 |
| dot_general_read_into_f32_s256 | pytorch-cuda | 128 | 15 | 2.439 | 0.0005542 |
| dot_general_read_into_accum_f32_s256 | pytorch-cuda | 128 | 15 | 2.856 | 0.0003617 |
| dot_general_read_into_accum_cached_f32_s256 | pytorch-cuda | 128 | 15 | 2.877 | 0.0003617 |
| dot_general_read_into_f64_s256 | pytorch-cuda | 8 | 15 | 1.986 | 0 |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 8 | 15 | 2.160 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 8 | 15 | 2.239 | 0 |
| dot_general_read_into_c32_s256 | pytorch-cuda | 128 | 15 | 3.433 | 0.0008017 |
| dot_general_read_into_accum_c32_s256 | pytorch-cuda | 128 | 15 | 3.555 | 0.000499 |
| dot_general_read_into_accum_cached_c32_s256 | pytorch-cuda | 128 | 15 | 3.648 | 0.000499 |
| dot_general_read_into_c64_s256 | pytorch-cuda | 4 | 15 | 3.902 | 1.175e-06 |
| dot_general_read_into_accum_c64_s256 | pytorch-cuda | 4 | 15 | 3.927 | 6.523e-07 |
| dot_general_read_into_accum_cached_c64_s256 | pytorch-cuda | 4 | 15 | 3.927 | 6.523e-07 |
| add_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.711 | 0 |
| add_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.676 | 0 |
| mul_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.656 | 0 |
| mul_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.660 | 0 |
| neg_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.881 | 0 |
| neg_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.544 | 0 |
| exp_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.347 | 0.0001272 |
| exp_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.870 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.547 | 9.107e-05 |
| tanh_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.502 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | pytorch-cuda | 128 | 15 | 2.007 | 0.0005233 |
| reduce_sum_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 4.013 | 3.504e-07 |
| conj_strided_read_c32_n1024 | pytorch-cuda | 256 | 15 | 3.036 | 0 |
| conj_strided_read_c64_n1024 | pytorch-cuda | 256 | 15 | 3.150 | 0 |
| add_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.547 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.489 | 0 |
| mul_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.546 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| neg_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.750 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.699 | 0 |
| exp_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.755 | 0.0001272 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.900 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.755 | 9.135e-05 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.687 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.767 | 0.002693 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.367 | 5.935e-06 |
| conj_strided_read_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.697 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.666 | 0 |
| axpby_overwrite_f32_n1024 | pytorch-cuda | 256 | 15 | 3.020 | 0 |
| axpby_lerp_f32_n1024 | pytorch-cuda | 256 | 15 | 2.312 | 0.0001189 |
| axpby_unit_add_f32_n1024 | pytorch-cuda | 512 | 15 | 3.892 | 0 |
| axpby_overwrite_f64_n1024 | pytorch-cuda | 256 | 15 | 2.860 | 0 |
| axpby_lerp_f64_n1024 | pytorch-cuda | 256 | 15 | 2.306 | 1.161e-07 |
| axpby_unit_add_f64_n1024 | pytorch-cuda | 256 | 15 | 2.005 | 0 |
| axpby_overwrite_c32_n1024 | pytorch-cuda | 256 | 15 | 3.029 | 0 |
| axpby_lerp_c32_n1024 | pytorch-cuda | 512 | 15 | 4.052 | 0.0001189 |
| axpby_unit_add_c32_n1024 | pytorch-cuda | 512 | 15 | 4.248 | 0 |
| axpby_overwrite_c64_n1024 | pytorch-cuda | 256 | 15 | 2.991 | 0 |
| axpby_lerp_c64_n1024 | pytorch-cuda | 256 | 15 | 2.773 | 1.161e-07 |
| axpby_unit_add_c64_n1024 | pytorch-cuda | 256 | 15 | 1.994 | 0 |
| axpby_overwrite_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.752 | 0 |
| axpby_lerp_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.545 | 0.0001192 |
| axpby_unit_add_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.542 | 0 |
| axpby_overwrite_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.702 | 0 |
| axpby_lerp_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.495 | 1.161e-07 |
| axpby_unit_add_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| axpby_overwrite_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.700 | 0 |
| axpby_lerp_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.493 | 0.0001191 |
| axpby_unit_add_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.485 | 0 |
| axpby_overwrite_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.670 | 0 |
| axpby_lerp_c64_n1048576 | pytorch-cuda | 8 | 15 | 2.724 | 1.159e-07 |
| axpby_unit_add_c64_n1048576 | pytorch-cuda | 16 | 15 | 2.455 | 0 |
| dot_general_batch_f32_s32 | pytorch-cuda | 64 | 15 | 2.532 | 0.0002227 |
| dot_general_batch_f64_s32 | pytorch-cuda | 64 | 15 | 2.927 | 1.326e-07 |
| dot_general_batch_c32_s32 | pytorch-cuda | 64 | 15 | 2.805 | 0.0003092 |
| dot_general_batch_c64_s32 | pytorch-cuda | 64 | 15 | 2.841 | 2.791e-07 |
| dot_general_multiple_f32_s32 | pytorch-cuda | 32 | 15 | 2.456 | 0.000269 |
| dot_general_multiple_f64_s32 | pytorch-cuda | 32 | 15 | 2.523 | 2.466e-07 |
| dot_general_multiple_c32_s32 | pytorch-cuda | 32 | 15 | 2.438 | 0.0004058 |
| dot_general_multiple_c64_s32 | pytorch-cuda | 32 | 15 | 2.546 | 4.129e-07 |
| dot_general_batch_f32_s256 | pytorch-cuda | 64 | 15 | 2.620 | 0.0008561 |
| dot_general_batch_f64_s256 | pytorch-cuda | 8 | 15 | 3.347 | 7.463e-07 |
| dot_general_batch_c32_s256 | pytorch-cuda | 64 | 15 | 3.454 | 0.001286 |
| dot_general_batch_c64_s256 | pytorch-cuda | 2 | 15 | 2.659 | 8.893e-07 |
| dot_general_multiple_f32_s256 | pytorch-cuda | 32 | 15 | 2.428 | 0.00133 |
| dot_general_multiple_f64_s256 | pytorch-cuda | 8 | 15 | 2.705 | 2.584e-06 |
| dot_general_multiple_c32_s256 | pytorch-cuda | 32 | 15 | 2.821 | 0.001059 |
| dot_general_multiple_c64_s256 | pytorch-cuda | 2 | 15 | 2.022 | 2.776e-06 |
| dot_conj_l0r0_c32_n1024 | pytorch-cuda | 128 | 15 | 2.020 | 0.0001028 |
| dot_conj_l1r0_c32_n1024 | pytorch-cuda | 256 | 15 | 3.968 | 0.0004339 |
| dot_conj_l0r1_c32_n1024 | pytorch-cuda | 256 | 15 | 4.013 | 0.0004457 |
| dot_conj_l1r1_c32_n1024 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n1024 | pytorch-cuda | 128 | 15 | 2.035 | 4.333e-07 |
| dot_conj_l1r0_c64_n1024 | pytorch-cuda | 256 | 15 | 3.917 | 2.122e-07 |
| dot_conj_l0r1_c64_n1024 | pytorch-cuda | 128 | 15 | 1.933 | 2.847e-07 |
| dot_conj_l1r1_c64_n1024 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c32_n262144 | pytorch-cuda | 128 | 15 | 2.228 | 0.003079 |
| dot_conj_l1r0_c32_n262144 | pytorch-cuda | 128 | 15 | 2.217 | 0.006155 |
| dot_conj_l0r1_c32_n262144 | pytorch-cuda | 128 | 15 | 2.231 | 0.006499 |
| dot_conj_l1r1_c32_n262144 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n262144 | pytorch-cuda | 64 | 15 | 2.163 | 1.455e-05 |
| dot_conj_l1r0_c64_n262144 | pytorch-cuda | 64 | 15 | 2.165 | 2.05e-05 |
| dot_conj_l0r1_c64_n262144 | pytorch-cuda | 64 | 15 | 2.157 | 1.888e-05 |
| dot_conj_l1r1_c64_n262144 | pytorch-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| einsum_plan_owned_f32_s32 | pytorch-cuda | 128 | 15 | 2.750 | 0.0002018 |
| einsum_plan_read_f32_s32 | pytorch-cuda | 128 | 15 | 2.727 | 0.0002018 |
| einsum_plan_strided_read_f32_s32 | pytorch-cuda | 128 | 15 | 3.004 | 0.0002018 |
| einsum_plan_into_f32_s32 | pytorch-cuda | 128 | 15 | 1.973 | 0.0002018 |
| einsum_plan_read_into_f32_s32 | pytorch-cuda | 128 | 15 | 1.978 | 0.0002018 |
| einsum_plan_read_into_accum_f32_s32 | pytorch-cuda | 128 | 15 | 2.989 | 0.0001134 |
| einsum_plan_strided_read_into_f32_s32 | pytorch-cuda | 128 | 15 | 2.175 | 0.0002018 |
| einsum_plan_strided_read_into_accum_f32_s32 | pytorch-cuda | 128 | 15 | 3.130 | 0.0001134 |
| einsum_plan_owned_f64_s32 | pytorch-cuda | 128 | 15 | 3.035 | 0 |
| einsum_plan_read_f64_s32 | pytorch-cuda | 128 | 15 | 3.065 | 0 |
| einsum_plan_strided_read_f64_s32 | pytorch-cuda | 128 | 15 | 3.277 | 0 |
| einsum_plan_into_f64_s32 | pytorch-cuda | 128 | 15 | 2.289 | 0 |
| einsum_plan_read_into_f64_s32 | pytorch-cuda | 128 | 15 | 2.294 | 0 |
| einsum_plan_read_into_accum_f64_s32 | pytorch-cuda | 128 | 15 | 3.202 | 0 |
| einsum_plan_strided_read_into_f64_s32 | pytorch-cuda | 128 | 15 | 2.361 | 0 |
| einsum_plan_strided_read_into_accum_f64_s32 | pytorch-cuda | 128 | 15 | 3.297 | 0 |
| einsum_plan_owned_c32_s32 | pytorch-cuda | 128 | 15 | 2.875 | 0.0002342 |
| einsum_plan_read_c32_s32 | pytorch-cuda | 128 | 15 | 2.842 | 0.0002342 |
| einsum_plan_strided_read_c32_s32 | pytorch-cuda | 128 | 15 | 2.935 | 0.0002342 |
| einsum_plan_into_c32_s32 | pytorch-cuda | 128 | 15 | 2.224 | 0.0002342 |
| einsum_plan_read_into_c32_s32 | pytorch-cuda | 128 | 15 | 2.152 | 0.0002342 |
| einsum_plan_read_into_accum_c32_s32 | pytorch-cuda | 128 | 15 | 2.993 | 0.0001157 |
| einsum_plan_strided_read_into_c32_s32 | pytorch-cuda | 128 | 15 | 2.191 | 0.0002342 |
| einsum_plan_strided_read_into_accum_c32_s32 | pytorch-cuda | 128 | 15 | 3.044 | 0.0001157 |
| einsum_plan_owned_c64_s32 | pytorch-cuda | 128 | 15 | 3.235 | 1.981e-07 |
| einsum_plan_read_c64_s32 | pytorch-cuda | 128 | 15 | 3.251 | 1.981e-07 |
| einsum_plan_strided_read_c64_s32 | pytorch-cuda | 128 | 15 | 3.738 | 1.981e-07 |
| einsum_plan_into_c64_s32 | pytorch-cuda | 128 | 15 | 2.633 | 1.981e-07 |
| einsum_plan_read_into_c64_s32 | pytorch-cuda | 128 | 15 | 2.470 | 1.981e-07 |
| einsum_plan_read_into_accum_c64_s32 | pytorch-cuda | 128 | 15 | 3.308 | 1.097e-07 |
| einsum_plan_strided_read_into_c64_s32 | pytorch-cuda | 128 | 15 | 2.529 | 1.981e-07 |
| einsum_plan_strided_read_into_accum_c64_s32 | pytorch-cuda | 128 | 15 | 3.469 | 1.097e-07 |
| einsum_plan_owned_f32_s256 | pytorch-cuda | 128 | 15 | 2.853 | 0.0009425 |
| einsum_plan_read_f32_s256 | pytorch-cuda | 128 | 15 | 2.866 | 0.0009425 |
| einsum_plan_strided_read_f32_s256 | pytorch-cuda | 128 | 15 | 3.089 | 0.0009425 |
| einsum_plan_into_f32_s256 | pytorch-cuda | 128 | 15 | 2.014 | 0.0009425 |
| einsum_plan_read_into_f32_s256 | pytorch-cuda | 128 | 15 | 2.034 | 0.0009425 |
| einsum_plan_read_into_accum_f32_s256 | pytorch-cuda | 128 | 15 | 2.860 | 0.0003475 |
| einsum_plan_strided_read_into_f32_s256 | pytorch-cuda | 128 | 15 | 2.311 | 0.0009425 |
| einsum_plan_strided_read_into_accum_f32_s256 | pytorch-cuda | 128 | 15 | 3.264 | 0.0003475 |
| einsum_plan_owned_f64_s256 | pytorch-cuda | 16 | 15 | 2.628 | 0 |
| einsum_plan_read_f64_s256 | pytorch-cuda | 16 | 15 | 2.626 | 0 |
| einsum_plan_strided_read_f64_s256 | pytorch-cuda | 16 | 15 | 2.591 | 0 |
| einsum_plan_into_f64_s256 | pytorch-cuda | 16 | 15 | 2.618 | 0 |
| einsum_plan_read_into_f64_s256 | pytorch-cuda | 16 | 15 | 2.619 | 0 |
| einsum_plan_read_into_accum_f64_s256 | pytorch-cuda | 16 | 15 | 2.616 | 0 |
| einsum_plan_strided_read_into_f64_s256 | pytorch-cuda | 16 | 15 | 2.612 | 0 |
| einsum_plan_strided_read_into_accum_f64_s256 | pytorch-cuda | 16 | 15 | 2.375 | 0 |
| einsum_plan_owned_c32_s256 | pytorch-cuda | 128 | 15 | 2.855 | 0.001598 |
| einsum_plan_read_c32_s256 | pytorch-cuda | 128 | 15 | 2.883 | 0.001598 |
| einsum_plan_strided_read_c32_s256 | pytorch-cuda | 128 | 15 | 3.110 | 0.001598 |
| einsum_plan_into_c32_s256 | pytorch-cuda | 128 | 15 | 2.266 | 0.001598 |
| einsum_plan_read_into_c32_s256 | pytorch-cuda | 128 | 15 | 2.239 | 0.001598 |
| einsum_plan_read_into_accum_c32_s256 | pytorch-cuda | 128 | 15 | 3.009 | 0.0006293 |
| einsum_plan_strided_read_into_c32_s256 | pytorch-cuda | 128 | 15 | 2.352 | 0.001598 |
| einsum_plan_strided_read_into_accum_c32_s256 | pytorch-cuda | 128 | 15 | 3.486 | 0.0006293 |
| einsum_plan_owned_c64_s256 | pytorch-cuda | 8 | 15 | 3.933 | 8.397e-07 |
| einsum_plan_read_c64_s256 | pytorch-cuda | 8 | 15 | 3.930 | 8.397e-07 |
| einsum_plan_strided_read_c64_s256 | pytorch-cuda | 8 | 15 | 3.932 | 8.397e-07 |
| einsum_plan_into_c64_s256 | pytorch-cuda | 8 | 15 | 3.924 | 8.397e-07 |
| einsum_plan_read_into_c64_s256 | pytorch-cuda | 8 | 15 | 3.926 | 8.397e-07 |
| einsum_plan_read_into_accum_c64_s256 | pytorch-cuda | 8 | 15 | 3.959 | 4.364e-07 |
| einsum_plan_strided_read_into_c64_s256 | pytorch-cuda | 8 | 15 | 3.926 | 8.397e-07 |
| einsum_plan_strided_read_into_accum_c64_s256 | pytorch-cuda | 4 | 15 | 2.005 | 4.364e-07 |
| sub_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.996 | 0 |
| sub_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.719 | 0 |
| div_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.799 | 0 |
| div_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.674 | 0 |
| rem_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.607 | 0 |
| rem_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.567 | 0 |
| pow_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.599 | 0.000117 |
| pow_strided_read_f64_n1024 | pytorch-cuda | 128 | 15 | 3.732 | 1.48e-07 |
| maximum_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.592 | 0 |
| maximum_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.546 | 0 |
| minimum_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.698 | 0 |
| minimum_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.700 | 0 |
| abs_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 3.478 | 0 |
| abs_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 3.453 | 0 |
| sign_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.351 | 0 |
| sign_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.356 | 0 |
| expm1_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.408 | 7.911e-05 |
| expm1_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.471 | 7.39e-08 |
| log_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.386 | 7.945e-05 |
| log_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.798 | 4.231e-08 |
| log1p_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.573 | 0.0001189 |
| log1p_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.864 | 1.108e-07 |
| erf_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.405 | 7.914e-05 |
| erf_strided_read_f64_n1024 | pytorch-cuda | 128 | 15 | 2.007 | 8.536e-08 |
| sin_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.433 | 4.348e-05 |
| sin_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.426 | 4.153e-08 |
| cos_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.309 | 6.342e-05 |
| cos_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.354 | 5.905e-08 |
| sqrt_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.381 | 0 |
| sqrt_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.406 | 0 |
| rsqrt_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.458 | 6.983e-05 |
| rsqrt_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.390 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 3.475 | 0.0005958 |
| reduce_prod_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 3.351 | 4.44e-07 |
| reduce_max_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 3.782 | 0 |
| reduce_max_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 3.788 | 0 |
| reduce_min_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 3.779 | 0 |
| reduce_min_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 3.962 | 0 |
| reduce_sum_squares_strided_read_f32_n1024 | pytorch-cuda | 128 | 15 | 3.522 | 0.0001524 |
| reduce_sum_squares_strided_read_f64_n1024 | pytorch-cuda | 128 | 15 | 3.541 | 2.836e-07 |
| sub_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| sub_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| div_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.558 | 0 |
| div_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.020 | 0 |
| rem_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.559 | 0 |
| rem_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.502 | 0 |
| pow_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.662 | 0.0001577 |
| pow_strided_read_f64_n1048576 | pytorch-cuda | 2 | 15 | 2.030 | 1.48e-07 |
| maximum_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.556 | 0 |
| maximum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.511 | 0 |
| minimum_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| minimum_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.510 | 0 |
| abs_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 0 |
| abs_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.702 | 0 |
| sign_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.752 | 0 |
| sign_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.704 | 0 |
| expm1_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.759 | 7.915e-05 |
| expm1_strided_read_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.587 | 7.39e-08 |
| log_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.759 | 7.946e-05 |
| log_strided_read_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.797 | 4.308e-08 |
| log1p_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.758 | 0.000119 |
| log1p_strided_read_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.894 | 1.11e-07 |
| erf_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.757 | 7.923e-05 |
| erf_strided_read_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.115 | 8.536e-08 |
| sin_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.764 | 4.348e-05 |
| sin_strided_read_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.384 | 4.153e-08 |
| cos_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.774 | 6.342e-05 |
| cos_strided_read_f64_n1048576 | pytorch-cuda | 16 | 15 | 3.377 | 5.905e-08 |
| sqrt_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 0 |
| sqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.940 | 0 |
| rsqrt_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.753 | 6.983e-05 |
| rsqrt_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.968 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.524 | 0.009062 |
| reduce_prod_strided_read_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.331 | 4.106e-06 |
| reduce_max_strided_read_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.762 | 0 |
| reduce_max_strided_read_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.705 | 0 |
| reduce_min_strided_read_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.056 | 0 |
| reduce_min_strided_read_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.710 | 0 |
| reduce_sum_squares_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.890 | 0.0001778 |
| reduce_sum_squares_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.890 | 3.309e-07 |
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
| tensordot_f32_n1024 | jax-cuda | 8 | 15 | 2.118 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda | 16 | 15 | 2.663 | 0 |
| tensordot_c32_n1024 | jax-cuda | 16 | 15 | 2.935 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda | 16 | 15 | 2.986 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda | 32 | 15 | 3.351 | 0 |
| vdot_read_f64_n1024 | jax-cuda | 32 | 15 | 2.933 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda | 32 | 15 | 2.694 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda | 32 | 15 | 3.859 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda | 32 | 15 | 2.731 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda | 32 | 15 | 2.393 | 0 |
| tensordot_f32_n262144 | jax-cuda | 16 | 15 | 2.622 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda | 2 | 15 | 3.404 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda | 8 | 15 | 2.618 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda | 1 | 15 | 5.643 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda | 16 | 15 | 2.589 | 0 |
| vdot_read_f64_n262144 | jax-cuda | 16 | 15 | 2.571 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda | 16 | 15 | 2.674 | 0 |
| vdot_read_c64_n262144 | jax-cuda | 16 | 15 | 2.485 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda | 16 | 15 | 2.229 | 0.0001786 |
| norm_squared_read_f64_n262144 | jax-cuda | 16 | 15 | 1.939 | 1.664e-07 |
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
| add_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.606 | 0 |
| add_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.763 | 0 |
| mul_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.367 | 0 |
| mul_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.324 | 0 |
| neg_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.122 | 0 |
| neg_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.056 | 0 |
| exp_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 3.267 | 0.0001272 |
| exp_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 2.964 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 3.003 | 9.325e-05 |
| tanh_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 3.273 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.535 | 0.0002267 |
| reduce_sum_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.964 | 3.268e-07 |
| conj_strided_read_c32_n1024 | jax-cuda | 16 | 15 | 2.530 | 0 |
| conj_strided_read_c64_n1024 | jax-cuda | 16 | 15 | 2.658 | 0 |
| add_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 1.956 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 1.969 | 0 |
| mul_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.566 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.442 | 0 |
| neg_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.324 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.402 | 0 |
| exp_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.015 | 0.0001272 |
| exp_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.256 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.008 | 9.508e-05 |
| tanh_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.065 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.193 | 0.002078 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.570 | 7.159e-06 |
| conj_strided_read_c32_n1048576 | jax-cuda | 32 | 15 | 3.145 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 1.858 | 0 |
| axpby_overwrite_f32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1024 | jax-cuda | 8 | 15 | 2.971 | 0 |
| axpby_overwrite_f64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1024 | jax-cuda | 8 | 15 | 2.838 | 0 |
| axpby_overwrite_c32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1024 | jax-cuda | 8 | 15 | 2.924 | 0 |
| axpby_overwrite_c64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1024 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1024 | jax-cuda | 8 | 15 | 2.924 | 0 |
| axpby_overwrite_f32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1048576 | jax-cuda | 8 | 15 | 2.946 | 0 |
| axpby_overwrite_f64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1048576 | jax-cuda | 8 | 15 | 2.977 | 0 |
| axpby_overwrite_c32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1048576 | jax-cuda | 8 | 15 | 3.122 | 0 |
| axpby_overwrite_c64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1048576 | jax-cuda | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1048576 | jax-cuda | 8 | 15 | 3.038 | 0 |
| dot_general_batch_f32_s32 | jax-cuda | 16 | 15 | 2.054 | 0.0002227 |
| dot_general_batch_f64_s32 | jax-cuda | 16 | 15 | 2.010 | 1.326e-07 |
| dot_general_batch_c32_s32 | jax-cuda | 16 | 15 | 2.561 | 0.0004354 |
| dot_general_batch_c64_s32 | jax-cuda | 16 | 15 | 2.792 | 2.524e-07 |
| dot_general_multiple_f32_s32 | jax-cuda | 16 | 15 | 2.090 | 0.000269 |
| dot_general_multiple_f64_s32 | jax-cuda | 16 | 15 | 2.585 | 2.466e-07 |
| dot_general_multiple_c32_s32 | jax-cuda | 8 | 15 | 1.810 | 0.0004771 |
| dot_general_multiple_c64_s32 | jax-cuda | 8 | 15 | 2.402 | 6.063e-07 |
| dot_general_batch_f32_s256 | jax-cuda | 16 | 15 | 2.048 | 0.0008561 |
| dot_general_batch_f64_s256 | jax-cuda | 8 | 15 | 2.584 | 7.463e-07 |
| dot_general_batch_c32_s256 | jax-cuda | 8 | 15 | 2.248 | 0.001359 |
| dot_general_batch_c64_s256 | jax-cuda | 2 | 15 | 2.932 | 8.493e-07 |
| dot_general_multiple_f32_s256 | jax-cuda | 8 | 15 | 2.664 | 0.00133 |
| dot_general_multiple_f64_s256 | jax-cuda | 16 | 15 | 3.489 | 2.584e-06 |
| dot_general_multiple_c32_s256 | jax-cuda | 8 | 15 | 2.626 | 0.002465 |
| dot_general_multiple_c64_s256 | jax-cuda | 2 | 15 | 2.278 | 1.805e-06 |
| dot_conj_l0r0_c32_n1024 | jax-cuda | 32 | 15 | 2.775 | 0.0002298 |
| dot_conj_l1r0_c32_n1024 | jax-cuda | 32 | 15 | 2.562 | 0.0002522 |
| dot_conj_l0r1_c32_n1024 | jax-cuda | 32 | 15 | 2.360 | 0.0002522 |
| dot_conj_l1r1_c32_n1024 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n1024 | jax-cuda | 32 | 15 | 3.124 | 3.644e-07 |
| dot_conj_l1r0_c64_n1024 | jax-cuda | 32 | 15 | 2.976 | 2.767e-07 |
| dot_conj_l0r1_c64_n1024 | jax-cuda | 32 | 15 | 3.008 | 2.767e-07 |
| dot_conj_l1r1_c64_n1024 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c32_n262144 | jax-cuda | 16 | 15 | 2.215 | 0.01276 |
| dot_conj_l1r0_c32_n262144 | jax-cuda | 16 | 15 | 1.885 | 0.0002732 |
| dot_conj_l0r1_c32_n262144 | jax-cuda | 32 | 15 | 3.274 | 0.0002732 |
| dot_conj_l1r1_c32_n262144 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda | 16 | 15 | 2.415 | 1.221e-05 |
| dot_conj_l1r0_c64_n262144 | jax-cuda | 16 | 15 | 2.126 | 1.586e-05 |
| dot_conj_l0r1_c64_n262144 | jax-cuda | 16 | 15 | 1.852 | 1.586e-05 |
| dot_conj_l1r1_c64_n262144 | jax-cuda | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| einsum_plan_owned_f32_s32 | jax-cuda | 32 | 15 | 2.745 | 0.0002018 |
| einsum_plan_read_f32_s32 | jax-cuda | 32 | 15 | 2.750 | 0.0002018 |
| einsum_plan_strided_read_f32_s32 | jax-cuda | 16 | 15 | 2.416 | 0.0002018 |
| einsum_plan_into_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f64_s32 | jax-cuda | 32 | 15 | 2.766 | 0 |
| einsum_plan_read_f64_s32 | jax-cuda | 32 | 15 | 2.743 | 0 |
| einsum_plan_strided_read_f64_s32 | jax-cuda | 16 | 15 | 2.545 | 0 |
| einsum_plan_into_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c32_s32 | jax-cuda | 16 | 15 | 2.472 | 0.0002532 |
| einsum_plan_read_c32_s32 | jax-cuda | 16 | 15 | 2.006 | 0.0002532 |
| einsum_plan_strided_read_c32_s32 | jax-cuda | 16 | 15 | 2.494 | 0.0002532 |
| einsum_plan_into_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c32_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c64_s32 | jax-cuda | 16 | 15 | 2.108 | 3.753e-07 |
| einsum_plan_read_c64_s32 | jax-cuda | 16 | 15 | 1.988 | 3.753e-07 |
| einsum_plan_strided_read_c64_s32 | jax-cuda | 16 | 15 | 3.089 | 3.753e-07 |
| einsum_plan_into_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c64_s32 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f32_s256 | jax-cuda | 32 | 15 | 2.368 | 0.00118 |
| einsum_plan_read_f32_s256 | jax-cuda | 32 | 15 | 2.333 | 0.00118 |
| einsum_plan_strided_read_f32_s256 | jax-cuda | 16 | 15 | 2.340 | 0.00118 |
| einsum_plan_into_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f64_s256 | jax-cuda | 32 | 15 | 3.473 | 0 |
| einsum_plan_read_f64_s256 | jax-cuda | 32 | 15 | 3.464 | 0 |
| einsum_plan_strided_read_f64_s256 | jax-cuda | 16 | 15 | 2.347 | 0 |
| einsum_plan_into_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c32_s256 | jax-cuda | 16 | 15 | 1.806 | 0.002017 |
| einsum_plan_read_c32_s256 | jax-cuda | 32 | 15 | 2.915 | 0.002017 |
| einsum_plan_strided_read_c32_s256 | jax-cuda | 16 | 15 | 2.542 | 0.002017 |
| einsum_plan_into_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c32_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c64_s256 | jax-cuda | 8 | 15 | 3.171 | 1.855e-06 |
| einsum_plan_read_c64_s256 | jax-cuda | 8 | 15 | 3.190 | 1.855e-06 |
| einsum_plan_strided_read_c64_s256 | jax-cuda | 8 | 15 | 3.166 | 1.855e-06 |
| einsum_plan_into_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c64_s256 | jax-cuda | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| sub_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.828 | 0 |
| sub_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.450 | 0 |
| div_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 1.982 | 4.875e-05 |
| div_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 3.467 | 0 |
| rem_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 3.409 | 0 |
| rem_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 1.971 | 0 |
| pow_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 3.020 | 0.000117 |
| pow_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 3.056 | 1.48e-07 |
| maximum_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.490 | 0 |
| maximum_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.221 | 0 |
| minimum_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.108 | 0 |
| minimum_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.046 | 0 |
| abs_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 3.298 | 0 |
| abs_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 3.184 | 0 |
| sign_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 3.221 | 0 |
| sign_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 3.214 | 0 |
| expm1_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 3.208 | 7.911e-05 |
| expm1_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 3.304 | 7.39e-08 |
| log_strided_read_f32_n1024 | jax-cuda | 32 | 15 | 3.281 | 7.945e-05 |
| log_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.056 | 4.231e-08 |
| log1p_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.280 | 0.0001189 |
| log1p_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.556 | 1.108e-07 |
| erf_strided_read_f32_n1024 | jax-cuda | 8 | 15 | 2.001 | 9.523e-05 |
| erf_strided_read_f64_n1024 | jax-cuda | 8 | 15 | 1.991 | 8.536e-08 |
| sin_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.419 | 4.348e-05 |
| sin_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.667 | 4.153e-08 |
| cos_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.724 | 6.342e-05 |
| cos_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.598 | 5.905e-08 |
| sqrt_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.589 | 0.000119 |
| sqrt_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.485 | 0 |
| rsqrt_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 3.061 | 6.983e-05 |
| rsqrt_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.865 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.667 | 0.0004768 |
| reduce_prod_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.270 | 4.441e-07 |
| reduce_max_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.734 | 0 |
| reduce_max_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.091 | 0 |
| reduce_min_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.387 | 0 |
| reduce_min_strided_read_f64_n1024 | jax-cuda | 32 | 15 | 3.055 | 0 |
| reduce_sum_squares_strided_read_f32_n1024 | jax-cuda | 2 | 15 | 3.650 | 0.0001522 |
| reduce_sum_squares_strided_read_f64_n1024 | jax-cuda | 2 | 15 | 3.578 | 1.418e-07 |
| sub_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.529 | 0 |
| sub_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.275 | 0 |
| div_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.166 | 4.875e-05 |
| div_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.070 | 0 |
| rem_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.005 | 0 |
| rem_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.348 | 0 |
| pow_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.106 | 0.0001577 |
| pow_strided_read_f64_n1048576 | jax-cuda | 2 | 15 | 2.622 | 1.48e-07 |
| maximum_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 1.966 | 0 |
| maximum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.416 | 0 |
| minimum_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.444 | 0 |
| minimum_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.275 | 0 |
| abs_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 2.965 | 0 |
| abs_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.973 | 0 |
| sign_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.046 | 0 |
| sign_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 2.876 | 0 |
| expm1_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.014 | 7.915e-05 |
| expm1_strided_read_f64_n1048576 | jax-cuda | 8 | 15 | 2.159 | 7.39e-08 |
| log_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.058 | 7.946e-05 |
| log_strided_read_f64_n1048576 | jax-cuda | 8 | 15 | 3.178 | 4.308e-08 |
| log1p_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.066 | 0.000119 |
| log1p_strided_read_f64_n1048576 | jax-cuda | 8 | 15 | 3.264 | 1.11e-07 |
| erf_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.312 | 0.0001214 |
| erf_strided_read_f64_n1048576 | jax-cuda | 4 | 15 | 2.702 | 8.536e-08 |
| sin_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 1.923 | 4.348e-05 |
| sin_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.808 | 4.153e-08 |
| cos_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.251 | 6.342e-05 |
| cos_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.804 | 5.905e-08 |
| sqrt_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.007 | 0.0001192 |
| sqrt_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.293 | 0 |
| rsqrt_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.155 | 6.983e-05 |
| rsqrt_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.201 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.374 | 0.007035 |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.708 | 4.995e-06 |
| reduce_max_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.265 | 0 |
| reduce_max_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.525 | 0 |
| reduce_min_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.047 | 0 |
| reduce_min_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.650 | 0 |
| reduce_sum_squares_strided_read_f32_n1048576 | jax-cuda | 1 | 15 | 2.389 | 0.0001778 |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda | 1 | 15 | 2.290 | 3.311e-07 |
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
| tensordot_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.208 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.299 | 0 |
| tensordot_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.729 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda-jit | 32 | 15 | 3.100 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.031 | 0 |
| vdot_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.576 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.083 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda-jit | 32 | 15 | 3.509 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.351 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.124 | 0 |
| tensordot_f32_n262144 | jax-cuda-jit | 32 | 15 | 2.891 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda-jit | 2 | 15 | 3.141 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.721 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda-jit | 1 | 15 | 5.404 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.718 | 0 |
| vdot_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.612 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.827 | 0 |
| vdot_read_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.433 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.789 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.662 | 1.664e-07 |
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
| add_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.779 | 0 |
| add_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.503 | 0 |
| mul_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.305 | 0 |
| mul_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.072 | 0 |
| neg_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.951 | 0 |
| neg_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.928 | 0 |
| exp_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.980 | 0.0001272 |
| exp_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.113 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.075 | 9.325e-05 |
| tanh_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.153 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.469 | 0.0002267 |
| reduce_sum_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.790 | 3.268e-07 |
| conj_strided_read_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.655 | 0 |
| conj_strided_read_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.435 | 0 |
| add_strided_read_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.038 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.686 | 0 |
| mul_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.169 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.984 | 0 |
| neg_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.057 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.159 | 0 |
| exp_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.062 | 0.0001272 |
| exp_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.248 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.621 | 9.508e-05 |
| tanh_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.006 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.999 | 0.003094 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.370 | 7.159e-06 |
| conj_strided_read_c32_n1048576 | jax-cuda-jit | 32 | 15 | 3.296 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.885 | 0 |
| axpby_overwrite_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1024 | jax-cuda-jit | 8 | 15 | 1.975 | 0 |
| axpby_overwrite_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1024 | jax-cuda-jit | 8 | 15 | 2.002 | 0 |
| axpby_overwrite_c32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1024 | jax-cuda-jit | 8 | 15 | 2.050 | 0 |
| axpby_overwrite_c64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1024 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1024 | jax-cuda-jit | 8 | 15 | 2.128 | 0 |
| axpby_overwrite_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f32_n1048576 | jax-cuda-jit | 8 | 15 | 2.067 | 0 |
| axpby_overwrite_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_f64_n1048576 | jax-cuda-jit | 8 | 15 | 1.991 | 0 |
| axpby_overwrite_c32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c32_n1048576 | jax-cuda-jit | 8 | 15 | 2.001 | 0 |
| axpby_overwrite_c64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_lerp_c64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call scaled caller-owned Ref update with these alpha/beta values; functional outputs and donation are different contracts. |
| axpby_unit_add_c64_n1048576 | jax-cuda-jit | 8 | 15 | 2.013 | 0 |
| dot_general_batch_f32_s32 | jax-cuda-jit | 32 | 15 | 2.260 | 0.0002227 |
| dot_general_batch_f64_s32 | jax-cuda-jit | 32 | 15 | 2.363 | 1.326e-07 |
| dot_general_batch_c32_s32 | jax-cuda-jit | 32 | 15 | 3.082 | 0.0004354 |
| dot_general_batch_c64_s32 | jax-cuda-jit | 32 | 15 | 3.142 | 2.524e-07 |
| dot_general_multiple_f32_s32 | jax-cuda-jit | 32 | 15 | 2.764 | 0.000269 |
| dot_general_multiple_f64_s32 | jax-cuda-jit | 32 | 15 | 2.515 | 2.466e-07 |
| dot_general_multiple_c32_s32 | jax-cuda-jit | 32 | 15 | 3.254 | 0.0004771 |
| dot_general_multiple_c64_s32 | jax-cuda-jit | 32 | 15 | 3.390 | 6.063e-07 |
| dot_general_batch_f32_s256 | jax-cuda-jit | 32 | 15 | 2.532 | 0.0008561 |
| dot_general_batch_f64_s256 | jax-cuda-jit | 8 | 15 | 2.508 | 7.463e-07 |
| dot_general_batch_c32_s256 | jax-cuda-jit | 16 | 15 | 2.598 | 0.001359 |
| dot_general_batch_c64_s256 | jax-cuda-jit | 2 | 15 | 2.782 | 8.493e-07 |
| dot_general_multiple_f32_s256 | jax-cuda-jit | 32 | 15 | 3.351 | 0.0025 |
| dot_general_multiple_f64_s256 | jax-cuda-jit | 16 | 15 | 3.382 | 2.584e-06 |
| dot_general_multiple_c32_s256 | jax-cuda-jit | 16 | 15 | 3.070 | 0.002465 |
| dot_general_multiple_c64_s256 | jax-cuda-jit | 2 | 15 | 2.122 | 1.805e-06 |
| dot_conj_l0r0_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.690 | 0.0002298 |
| dot_conj_l1r0_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.747 | 0.0002522 |
| dot_conj_l0r1_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.718 | 0.0002522 |
| dot_conj_l1r1_c32_n1024 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.780 | 3.644e-07 |
| dot_conj_l1r0_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.580 | 2.767e-07 |
| dot_conj_l0r1_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.777 | 2.767e-07 |
| dot_conj_l1r1_c64_n1024 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.775 | 0.01276 |
| dot_conj_l1r0_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.830 | 0.0002732 |
| dot_conj_l0r1_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.939 | 0.0002732 |
| dot_conj_l1r1_c32_n262144 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| dot_conj_l0r0_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.996 | 1.221e-05 |
| dot_conj_l1r0_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.921 | 1.586e-05 |
| dot_conj_l0r1_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.397 | 1.586e-05 |
| dot_conj_l1r1_c64_n262144 | jax-cuda-jit | — | — | — | No verified single public both-conjugated dot operation; conjugate-then-dot or conjugating the result is a different operation. |
| einsum_plan_owned_f32_s32 | jax-cuda-jit | 32 | 15 | 3.111 | 0.0002018 |
| einsum_plan_read_f32_s32 | jax-cuda-jit | 32 | 15 | 3.054 | 0.0002018 |
| einsum_plan_strided_read_f32_s32 | jax-cuda-jit | 32 | 15 | 3.042 | 0.0002018 |
| einsum_plan_into_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f64_s32 | jax-cuda-jit | 32 | 15 | 2.779 | 0 |
| einsum_plan_read_f64_s32 | jax-cuda-jit | 32 | 15 | 2.766 | 0 |
| einsum_plan_strided_read_f64_s32 | jax-cuda-jit | 32 | 15 | 2.842 | 0 |
| einsum_plan_into_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c32_s32 | jax-cuda-jit | 32 | 15 | 3.274 | 0.0002532 |
| einsum_plan_read_c32_s32 | jax-cuda-jit | 32 | 15 | 3.280 | 0.0002532 |
| einsum_plan_strided_read_c32_s32 | jax-cuda-jit | 16 | 15 | 2.114 | 0.0002532 |
| einsum_plan_into_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c32_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c64_s32 | jax-cuda-jit | 16 | 15 | 1.951 | 3.753e-07 |
| einsum_plan_read_c64_s32 | jax-cuda-jit | 32 | 15 | 3.567 | 3.753e-07 |
| einsum_plan_strided_read_c64_s32 | jax-cuda-jit | 16 | 15 | 2.164 | 3.753e-07 |
| einsum_plan_into_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c64_s32 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f32_s256 | jax-cuda-jit | 32 | 15 | 2.567 | 0.00118 |
| einsum_plan_read_f32_s256 | jax-cuda-jit | 32 | 15 | 2.504 | 0.00118 |
| einsum_plan_strided_read_f32_s256 | jax-cuda-jit | 32 | 15 | 2.557 | 0.00118 |
| einsum_plan_into_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_f64_s256 | jax-cuda-jit | 32 | 15 | 3.421 | 0 |
| einsum_plan_read_f64_s256 | jax-cuda-jit | 32 | 15 | 3.408 | 0 |
| einsum_plan_strided_read_f64_s256 | jax-cuda-jit | 32 | 15 | 3.419 | 0 |
| einsum_plan_into_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_f64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c32_s256 | jax-cuda-jit | 32 | 15 | 2.943 | 0.002017 |
| einsum_plan_read_c32_s256 | jax-cuda-jit | 32 | 15 | 2.905 | 0.002017 |
| einsum_plan_strided_read_c32_s256 | jax-cuda-jit | 16 | 15 | 1.983 | 0.002017 |
| einsum_plan_into_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c32_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_owned_c64_s256 | jax-cuda-jit | 8 | 15 | 3.396 | 1.855e-06 |
| einsum_plan_read_c64_s256 | jax-cuda-jit | 8 | 15 | 3.457 | 1.855e-06 |
| einsum_plan_strided_read_c64_s256 | jax-cuda-jit | 8 | 15 | 3.271 | 1.855e-06 |
| einsum_plan_into_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_read_into_accum_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| einsum_plan_strided_read_into_accum_c64_s256 | jax-cuda-jit | — | — | — | No matching single public matrix multiplication with caller-owned output; Ref.set(matmul(...)) and donation change the operation. |
| sub_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.441 | 0 |
| sub_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.221 | 0 |
| div_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.958 | 4.875e-05 |
| div_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.061 | 0 |
| rem_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.421 | 0 |
| rem_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.105 | 0 |
| pow_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.369 | 0.000117 |
| pow_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.517 | 1.48e-07 |
| maximum_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.497 | 0 |
| maximum_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.164 | 0 |
| minimum_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.998 | 0 |
| minimum_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.164 | 0 |
| abs_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.015 | 0 |
| abs_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.166 | 0 |
| sign_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.023 | 0 |
| sign_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.164 | 0 |
| expm1_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.177 | 7.911e-05 |
| expm1_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.109 | 7.39e-08 |
| log_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.172 | 7.945e-05 |
| log_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.356 | 4.231e-08 |
| log1p_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.047 | 0.0001189 |
| log1p_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.039 | 1.108e-07 |
| erf_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.458 | 9.523e-05 |
| erf_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.420 | 8.536e-08 |
| sin_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.524 | 4.348e-05 |
| sin_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.619 | 4.153e-08 |
| cos_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.649 | 6.342e-05 |
| cos_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.609 | 5.905e-08 |
| sqrt_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.375 | 0.000119 |
| sqrt_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.173 | 0 |
| rsqrt_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.950 | 6.983e-05 |
| rsqrt_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.285 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.195 | 0.0004768 |
| reduce_prod_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.784 | 6.662e-07 |
| reduce_max_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.626 | 0 |
| reduce_max_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.597 | 0 |
| reduce_min_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.259 | 0 |
| reduce_min_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.923 | 0 |
| reduce_sum_squares_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.202 | 0.0001522 |
| reduce_sum_squares_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.898 | 1.42e-07 |
| sub_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.196 | 0 |
| sub_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.873 | 0 |
| div_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.125 | 4.875e-05 |
| div_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.075 | 0 |
| rem_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.299 | 0 |
| rem_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.104 | 0 |
| pow_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.162 | 0.0001577 |
| pow_strided_read_f64_n1048576 | jax-cuda-jit | 2 | 15 | 2.587 | 1.48e-07 |
| maximum_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.167 | 0 |
| maximum_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.944 | 0 |
| minimum_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.107 | 0 |
| minimum_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.919 | 0 |
| abs_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.008 | 0 |
| abs_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.072 | 0 |
| sign_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.018 | 0 |
| sign_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.044 | 0 |
| expm1_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.025 | 7.915e-05 |
| expm1_strided_read_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.119 | 7.39e-08 |
| log_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.032 | 7.946e-05 |
| log_strided_read_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.232 | 4.308e-08 |
| log1p_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.058 | 0.000119 |
| log1p_strided_read_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.370 | 1.11e-07 |
| erf_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.229 | 0.0001214 |
| erf_strided_read_f64_n1048576 | jax-cuda-jit | 4 | 15 | 2.651 | 8.536e-08 |
| sin_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.176 | 4.348e-05 |
| sin_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.799 | 4.153e-08 |
| cos_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.954 | 6.342e-05 |
| cos_strided_read_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.017 | 5.905e-08 |
| sqrt_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.977 | 0.0001192 |
| sqrt_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.296 | 0 |
| rsqrt_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.940 | 6.983e-05 |
| rsqrt_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.819 | 6.504e-08 |
| reduce_prod_strided_read_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.046 | 0.007035 |
| reduce_prod_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.508 | 4.995e-06 |
| reduce_max_strided_read_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.407 | 0 |
| reduce_max_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.515 | 0 |
| reduce_min_strided_read_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.488 | 0 |
| reduce_min_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.557 | 0 |
| reduce_sum_squares_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.872 | 0.0001778 |
| reduce_sum_squares_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.366 | 3.311e-07 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: benchmarks/gpu/reuse_contraction.yaml
timestamp: '2026-10-13T20:27:59Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: fbd735abbfd5bcf0453f151472b274a71249f115
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 6b1ac3203310fe07a586b76bc19248d00a12ab0c
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
  expected_cases: 328
  selected_cases: 328
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: e04b88277edb1bb3202f50da23a7b0d359a81675b9fe8a8a3692b1595961aebe
    src/bin/benchmark_gpu_public_api/plan.rs: 3d0f10f390ebd0e08a40deb5a64e49e086dcd41b9a777d3a208c6846ed6b2fac
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: da168bf8b7c30f748f62839cf0bf7cce8f5bccf9c37b1c1d993fd70c466f2f76
    src/bin/benchmark_gpu_public_api/multi.rs: 1b477fef78260cf28aa31e515d57ab411f6241b91ffb30d625939538fdbc6bab
    src/bin/benchmark_gpu_public_api/lifetime.rs: c687a8b0329f6d3073073671b5d246acbb1f4dc3f983d1430da897a908ef1745
    src/bin/benchmark_gpu_public_api/read_layout.rs: 70b1ed632ad52db5654c3d9f3df9fdb20a1e8c48894dbdcb24ce055757f5f7eb
    src/bin/benchmark_gpu_public_api/structural.rs: b8fd99ceb5c80fd3474974b0ba9aa57833ad6bd25383af3091c11c0b6ab94f6d
    src/bin/benchmark_gpu_public_api/linalg.rs: 5f5c4afc161f3d6e267861511d0d5871be9192d9968c95b59a73c9dc5a64e22b
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: 1dc14ac94fac700aa696d0ada04315685e3245738f9514de3d7989b8cb59152b
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 3faaaf5ae74a2ef9c91888b8fd8e566d1cf83b24d2b385cc1cd7b87d9d80c6fb
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 1ec80b23a4e12325f074f26149b3a21fd9d50f1e9ad70d447db24c471a069b55
    scripts/run_gpu_public_api.sh: dd5e4cc47e85fcb5afdc56d460136b2a399a6a7d6a62005de8ba94afb0119ecc
    data/results/nvidia-gpu/gpu/reuse_contraction/20261013_202222/fixtures/manifest.json: 0dee7d09b26142623e40e8d1517ebf92104039edf94ef7a83c73cea2da9830ae
    target/release/benchmark_gpu_public_api: 1e7435f93580ec070b55a03ff6a2063ecf15fecdb0f6489947a353fe1371e378
  reference_numpy_version: 2.4.2
```
