# gpu/reuse_contraction: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261013_170413/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | 6.588 (0.662) | unsupported | 6.560 (0.227) | unsupported | unsupported |
| add_into_f64_n1024 | 5.043 (0.125) | unsupported | 6.781 (0.071) | unsupported | unsupported |
| add_read_into_f32_n1024 | 5.568 (0.310) | unsupported | 6.963 (0.338) | unsupported | unsupported |
| add_read_into_f64_n1024 | 5.099 (0.391) | unsupported | 6.714 (0.113) | unsupported | unsupported |
| sub_into_f32_n1024 | 5.392 (0.409) | unsupported | 7.038 (0.070) | unsupported | unsupported |
| sub_into_f64_n1024 | 5.367 (0.301) | unsupported | 6.952 (0.205) | unsupported | unsupported |
| sub_read_into_f32_n1024 | 5.264 (0.119) | unsupported | 7.079 (0.123) | unsupported | unsupported |
| sub_read_into_f64_n1024 | 5.418 (0.205) | unsupported | 6.973 (0.166) | unsupported | unsupported |
| mul_into_f32_n1024 | 5.549 (0.357) | unsupported | 7.029 (0.421) | unsupported | unsupported |
| mul_into_f64_n1024 | 5.312 (0.316) | unsupported | 7.051 (0.259) | unsupported | unsupported |
| mul_read_into_f32_n1024 | 5.506 (0.090) | unsupported | 6.747 (0.167) | unsupported | unsupported |
| mul_read_into_f64_n1024 | 5.356 (0.145) | unsupported | 6.790 (0.206) | unsupported | unsupported |
| div_into_f32_n1024 | 41.940 (0.317) | unsupported | 6.797 (0.140) | unsupported | unsupported |
| div_into_f64_n1024 | 43.077 (1.420) | unsupported | 6.854 (0.379) | unsupported | unsupported |
| div_read_into_f32_n1024 | 44.614 (2.980) | unsupported | 7.641 (0.563) | unsupported | unsupported |
| div_read_into_f64_n1024 | 41.599 (0.843) | unsupported | 6.574 (0.092) | unsupported | unsupported |
| neg_into_f32_n1024 | 5.155 (0.389) | unsupported | 6.609 (0.036) | unsupported | unsupported |
| neg_into_f64_n1024 | 5.351 (0.249) | unsupported | 6.298 (0.039) | unsupported | unsupported |
| neg_read_into_f32_n1024 | 5.122 (0.170) | unsupported | 6.545 (0.019) | unsupported | unsupported |
| neg_read_into_f64_n1024 | 4.891 (0.214) | unsupported | 6.175 (0.021) | unsupported | unsupported |
| conj_into_c32_n1024 | 42.631 (4.213) | unsupported | 7.250 (0.100) | unsupported | unsupported |
| conj_into_c64_n1024 | 188.689 (12.090) | unsupported | 7.347 (0.159) | unsupported | unsupported |
| conj_read_into_c32_n1024 | 38.525 (1.259) | unsupported | 7.750 (0.130) | unsupported | unsupported |
| conj_read_into_c64_n1024 | 182.087 (13.717) | unsupported | 7.661 (0.302) | unsupported | unsupported |
| add_into_f32_n1048576 | 43.166 (0.248) | unsupported | 39.797 (0.059) | unsupported | unsupported |
| add_into_f64_n1048576 | 82.723 (1.033) | unsupported | 77.719 (0.059) | unsupported | unsupported |
| add_read_into_f32_n1048576 | 42.867 (0.463) | unsupported | 39.793 (0.044) | unsupported | unsupported |
| add_read_into_f64_n1048576 | 83.191 (0.712) | unsupported | 77.682 (0.172) | unsupported | unsupported |
| sub_into_f32_n1048576 | 42.786 (0.254) | unsupported | 39.657 (0.077) | unsupported | unsupported |
| sub_into_f64_n1048576 | 82.360 (0.478) | unsupported | 77.630 (0.043) | unsupported | unsupported |
| sub_read_into_f32_n1048576 | 43.054 (0.467) | unsupported | 39.691 (0.085) | unsupported | unsupported |
| sub_read_into_f64_n1048576 | 82.393 (0.492) | unsupported | 77.640 (0.137) | unsupported | unsupported |
| mul_into_f32_n1048576 | 42.774 (0.093) | unsupported | 39.648 (0.056) | unsupported | unsupported |
| mul_into_f64_n1048576 | 82.935 (0.633) | unsupported | 77.564 (0.056) | unsupported | unsupported |
| mul_read_into_f32_n1048576 | 43.000 (0.526) | unsupported | 39.641 (0.079) | unsupported | unsupported |
| mul_read_into_f64_n1048576 | 82.532 (0.691) | unsupported | 77.548 (0.103) | unsupported | unsupported |
| div_into_f32_n1048576 | 70.179 (0.192) | unsupported | 39.913 (0.027) | unsupported | unsupported |
| div_into_f64_n1048576 | 151.450 (0.198) | unsupported | 94.393 (0.185) | unsupported | unsupported |
| div_read_into_f32_n1048576 | 70.180 (0.143) | unsupported | 39.996 (0.125) | unsupported | unsupported |
| div_read_into_f64_n1048576 | 151.435 (0.237) | unsupported | 94.484 (0.185) | unsupported | unsupported |
| neg_into_f32_n1048576 | 29.565 (0.200) | unsupported | 27.248 (0.082) | unsupported | unsupported |
| neg_into_f64_n1048576 | 57.043 (0.313) | unsupported | 52.919 (0.150) | unsupported | unsupported |
| neg_read_into_f32_n1048576 | 29.617 (0.140) | unsupported | 27.246 (0.065) | unsupported | unsupported |
| neg_read_into_f64_n1048576 | 56.994 (0.585) | unsupported | 52.887 (0.085) | unsupported | unsupported |
| conj_into_c32_n1048576 | 107.956 (0.043) | unsupported | 52.855 (0.090) | unsupported | unsupported |
| conj_into_c64_n1048576 | 209.305 (0.333) | unsupported | 103.713 (0.097) | unsupported | unsupported |
| conj_read_into_c32_n1048576 | 107.877 (0.069) | unsupported | 52.835 (0.063) | unsupported | unsupported |
| conj_read_into_c64_n1048576 | 209.031 (0.440) | unsupported | 103.696 (0.125) | unsupported | unsupported |
| tensordot_f32_n1024 | 30.019 (1.663) | 36.884 (0.888) | 47.151 (1.745) | 194.965 (8.173) | 80.396 (2.310) |
| tensordot_f64_n1024 | 30.113 (1.147) | 40.202 (0.954) | 55.206 (1.113) | 164.130 (4.683) | 77.559 (3.969) |
| tensordot_c32_n1024 | 30.513 (1.690) | 37.665 (1.465) | 47.714 (0.229) | 178.375 (2.772) | 90.320 (4.831) |
| tensordot_c64_n1024 | 122.732 (3.227) | 40.295 (1.069) | 52.274 (2.201) | 199.436 (9.799) | 92.499 (2.308) |
| vdot_read_f32_n1024 | 151.180 (4.914) | unsupported | 15.114 (0.124) | 86.856 (2.580) | 95.502 (2.747) |
| vdot_read_f64_n1024 | 164.977 (5.934) | unsupported | 14.905 (0.232) | 150.236 (2.579) | 90.355 (9.768) |
| vdot_read_c32_n1024 | 167.110 (4.040) | unsupported | 14.506 (0.382) | 158.124 (6.522) | 113.738 (13.321) |
| vdot_read_c64_n1024 | 192.563 (3.378) | unsupported | 14.490 (0.427) | 178.814 (14.431) | 147.216 (6.398) |
| norm_squared_read_f32_n1024 | 184.227 (6.709) | unsupported | 15.283 (0.870) | 148.405 (31.401) | 160.556 (10.452) |
| norm_squared_read_f64_n1024 | 167.252 (5.087) | unsupported | 14.940 (0.142) | 114.532 (19.127) | 130.997 (3.867) |
| tensordot_f32_n262144 | 73.843 (6.185) | 46.563 (1.115) | 55.661 (1.531) | 167.852 (4.419) | 136.566 (29.808) |
| tensordot_f64_n262144 | 1620.780 (3.272) | 1626.599 (3.140) | 1633.110 (2.888) | 1985.258 (72.693) | 1598.072 (12.313) |
| tensordot_c32_n262144 | 141.500 (2.694) | 142.419 (1.140) | 159.906 (7.728) | 300.165 (13.405) | 170.356 (10.932) |
| tensordot_c64_n262144 | 5057.954 (3.521) | 5051.912 (2.363) | 5278.158 (4.869) | 5658.205 (55.616) | 5718.921 (25.851) |
| vdot_read_f32_n262144 | 176.122 (7.134) | unsupported | 15.412 (0.851) | 175.365 (4.142) | 177.726 (15.013) |
| vdot_read_f64_n262144 | 193.633 (3.729) | unsupported | 20.301 (0.084) | 161.876 (24.164) | 165.266 (7.554) |
| vdot_read_c32_n262144 | 194.893 (3.609) | unsupported | 17.470 (0.092) | 139.020 (1.956) | 174.226 (22.586) |
| vdot_read_c64_n262144 | 193.111 (1.925) | unsupported | 33.929 (0.105) | 162.578 (5.688) | 191.835 (21.265) |
| norm_squared_read_f32_n262144 | 191.094 (3.519) | unsupported | 15.634 (0.866) | 143.482 (34.119) | 175.812 (12.961) |
| norm_squared_read_f64_n262144 | 191.077 (5.479) | unsupported | 15.060 (0.263) | 116.990 (14.614) | 160.636 (6.776) |
| axpby_read_into_accum_f32_n1024 | 12.044 (0.500) | unsupported | 7.814 (0.124) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1024 | 11.011 (0.587) | unsupported | 7.942 (0.348) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1024 | 11.106 (0.617) | unsupported | 7.883 (0.126) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1024 | 11.508 (0.755) | unsupported | 8.584 (0.749) | unsupported | unsupported |
| axpby_read_into_accum_f32_n1048576 | 273.131 (0.213) | unsupported | 39.734 (0.021) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1048576 | 535.046 (0.131) | unsupported | 77.748 (0.056) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1048576 | 168.996 (0.256) | unsupported | 77.679 (0.117) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1048576 | 2144.240 (2.254) | unsupported | 153.587 (0.262) | unsupported | unsupported |
| dot_general_read_into_f32_s16 | 23.870 (0.860) | unsupported | 18.602 (0.240) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s16 | 24.008 (1.238) | unsupported | 22.840 (1.283) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s16 | 24.659 (2.106) | unsupported | 22.449 (0.243) | unsupported | unsupported |
| dot_general_read_into_f64_s16 | 27.889 (0.570) | unsupported | 22.090 (0.273) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s16 | 26.647 (1.840) | unsupported | 26.421 (1.251) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s16 | 27.445 (0.683) | unsupported | 25.853 (0.373) | unsupported | unsupported |
| dot_general_read_into_c32_s16 | 24.368 (0.794) | unsupported | 20.768 (0.406) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s16 | 24.397 (0.892) | unsupported | 24.512 (0.193) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s16 | 24.243 (0.586) | unsupported | 24.179 (0.437) | unsupported | unsupported |
| dot_general_read_into_c64_s16 | 27.101 (0.833) | unsupported | 22.592 (0.939) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s16 | 26.655 (1.518) | unsupported | 26.196 (0.326) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s16 | 28.113 (1.758) | unsupported | 26.259 (0.298) | unsupported | unsupported |
| dot_general_read_into_f32_s256 | 25.071 (0.272) | unsupported | 19.415 (1.306) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s256 | 25.041 (1.006) | unsupported | 22.544 (0.866) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s256 | 24.756 (0.240) | unsupported | 22.493 (0.470) | unsupported | unsupported |
| dot_general_read_into_f64_s256 | 263.106 (0.164) | unsupported | 247.017 (5.111) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s256 | 271.481 (10.803) | unsupported | 281.121 (21.072) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s256 | 270.331 (19.815) | unsupported | 271.286 (14.940) | unsupported | unsupported |
| dot_general_read_into_c32_s256 | 29.313 (0.794) | unsupported | 27.071 (0.136) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s256 | 30.160 (0.163) | unsupported | 27.740 (0.071) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s256 | 30.005 (0.210) | unsupported | 27.896 (0.387) | unsupported | unsupported |
| dot_general_read_into_c64_s256 | 687.844 (0.629) | unsupported | 983.473 (2.683) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s256 | 689.033 (0.447) | unsupported | 979.885 (1.359) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s256 | 689.366 (0.832) | unsupported | 979.944 (0.294) | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.373 | 0 |
| add_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.582 | 0 |
| add_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.851 | 0 |
| add_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.611 | 0 |
| sub_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.761 | 0 |
| sub_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.748 | 0 |
| sub_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.695 | 0 |
| sub_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.774 | 0 |
| mul_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.841 | 0 |
| mul_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.720 | 0 |
| mul_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.819 | 0 |
| mul_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.743 | 0 |
| div_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.684 | 0 |
| div_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.757 | 0 |
| div_read_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.855 | 0 |
| div_read_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.662 | 0 |
| neg_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.639 | 0 |
| neg_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.740 | 0 |
| neg_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.622 | 0 |
| neg_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.504 | 0 |
| conj_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.728 | 0 |
| conj_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.019 | 0 |
| conj_read_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.466 | 0 |
| conj_read_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.913 | 0 |
| add_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.763 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.647 | 0 |
| add_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.744 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.662 | 0 |
| sub_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.738 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.636 | 0 |
| sub_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.755 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.637 | 0 |
| mul_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.738 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.654 | 0 |
| mul_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.752 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.641 | 0 |
| div_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.246 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.423 | 0 |
| div_read_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.246 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.423 | 0 |
| neg_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.892 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.825 | 0 |
| neg_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.896 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.824 | 0 |
| conj_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.455 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.349 | 0 |
| conj_read_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.452 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.344 | 0 |
| tensordot_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.842 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.854 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.953 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 1.964 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.419 | 0 |
| vdot_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.640 | 1.722e-07 |
| vdot_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.674 | 0 |
| vdot_read_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.081 | 3.455e-07 |
| norm_squared_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.948 | 0 |
| norm_squared_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.676 | 0 |
| tensordot_f32_n262144 | tenferro-cuda-direct | 32 | 15 | 2.363 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 2 | 15 | 3.242 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.264 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-direct | 1 | 15 | 5.058 | 1.446e-05 |
| vdot_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.818 | 0 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.098 | 2.995e-06 |
| vdot_read_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 3.118 | 0 |
| vdot_read_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.090 | 8.762e-06 |
| norm_squared_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 3.057 | 0.0001786 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.057 | 0 |
| axpby_read_into_accum_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 1.542 | 0 |
| axpby_read_into_accum_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.819 | 0 |
| axpby_read_into_accum_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.843 | 0 |
| axpby_read_into_accum_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.946 | 0 |
| axpby_read_into_accum_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.185 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.140 | 0 |
| axpby_read_into_accum_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.704 | 0 |
| axpby_read_into_accum_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.144 | 0 |
| dot_general_read_into_f32_s16 | tenferro-cuda-direct | 128 | 15 | 3.055 | 8.645e-05 |
| dot_general_read_into_accum_f32_s16 | tenferro-cuda-direct | 128 | 15 | 3.073 | 7.907e-05 |
| dot_general_read_into_accum_cached_f32_s16 | tenferro-cuda-direct | 128 | 15 | 3.156 | 7.907e-05 |
| dot_general_read_into_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.570 | 0 |
| dot_general_read_into_accum_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.411 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.513 | 0 |
| dot_general_read_into_c32_s16 | tenferro-cuda-direct | 128 | 15 | 3.119 | 0.0003615 |
| dot_general_read_into_accum_c32_s16 | tenferro-cuda-direct | 128 | 15 | 3.123 | 0.0001632 |
| dot_general_read_into_accum_cached_c32_s16 | tenferro-cuda-direct | 128 | 15 | 3.103 | 0.0001632 |
| dot_general_read_into_c64_s16 | tenferro-cuda-direct | 64 | 15 | 1.734 | 1.204e-07 |
| dot_general_read_into_accum_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.412 | 6.975e-08 |
| dot_general_read_into_accum_cached_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.599 | 6.975e-08 |
| dot_general_read_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.209 | 0.0005542 |
| dot_general_read_into_accum_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.205 | 0.0003617 |
| dot_general_read_into_accum_cached_f32_s256 | tenferro-cuda-direct | 128 | 15 | 3.169 | 0.0003617 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.105 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.172 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.163 | 0 |
| dot_general_read_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.752 | 0.001457 |
| dot_general_read_into_accum_c32_s256 | tenferro-cuda-direct | 64 | 15 | 1.930 | 0.0008947 |
| dot_general_read_into_accum_cached_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.841 | 0.0008947 |
| dot_general_read_into_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.751 | 2.092e-06 |
| dot_general_read_into_accum_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.756 | 1.318e-06 |
| dot_general_read_into_accum_cached_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.757 | 1.318e-06 |
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
| tensordot_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.361 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.573 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.411 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.579 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| tensordot_f32_n262144 | tenferro-cuda-eager | 64 | 15 | 2.980 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 2 | 15 | 3.253 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-eager | 16 | 15 | 2.279 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-eager | 1 | 15 | 5.052 | 1.446e-05 |
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
| add_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.359 | 0 |
| add_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.472 | 0 |
| add_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.565 | 0 |
| add_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.437 | 0 |
| sub_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.603 | 0 |
| sub_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.560 | 0 |
| sub_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.625 | 0 |
| sub_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.570 | 0 |
| mul_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.599 | 0 |
| mul_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.610 | 0 |
| mul_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.454 | 0 |
| mul_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.476 | 0 |
| div_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.480 | 0 |
| div_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.509 | 0 |
| div_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.912 | 0 |
| div_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.366 | 0 |
| neg_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.384 | 0 |
| neg_into_f64_n1024 | pytorch-cuda | 256 | 15 | 1.612 | 0 |
| neg_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.351 | 0 |
| neg_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.161 | 0 |
| conj_into_c32_n1024 | pytorch-cuda | 512 | 15 | 3.712 | 0 |
| conj_into_c64_n1024 | pytorch-cuda | 512 | 15 | 3.761 | 0 |
| conj_read_into_c32_n1024 | pytorch-cuda | 256 | 15 | 1.984 | 0 |
| conj_read_into_c64_n1024 | pytorch-cuda | 512 | 15 | 3.922 | 0 |
| add_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.547 | 0 |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.487 | 0 |
| add_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.547 | 0 |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| sub_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.538 | 0 |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| sub_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.540 | 0 |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| mul_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.537 | 0 |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.482 | 0 |
| mul_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.537 | 0 |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.482 | 0 |
| div_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.554 | 0 |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.021 | 0 |
| div_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.560 | 0 |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.023 | 0 |
| neg_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.744 | 0 |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.693 | 0 |
| neg_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.744 | 0 |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.692 | 0 |
| conj_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.691 | 0 |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.659 | 0 |
| conj_read_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.691 | 0 |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.659 | 0 |
| tensordot_f32_n1024 | pytorch-cuda | 64 | 15 | 3.018 | 0.0008859 |
| tensordot_f64_n1024 | pytorch-cuda | 32 | 15 | 1.767 | 0 |
| tensordot_c32_n1024 | pytorch-cuda | 64 | 15 | 3.054 | 0.0005344 |
| tensordot_c64_n1024 | pytorch-cuda | 64 | 15 | 3.346 | 0 |
| vdot_read_f32_n1024 | pytorch-cuda | 256 | 15 | 3.869 | 0 |
| vdot_read_f64_n1024 | pytorch-cuda | 256 | 15 | 3.816 | 1.722e-07 |
| vdot_read_c32_n1024 | pytorch-cuda | 256 | 15 | 3.714 | 0 |
| vdot_read_c64_n1024 | pytorch-cuda | 256 | 15 | 3.710 | 3.455e-07 |
| norm_squared_read_f32_n1024 | pytorch-cuda | 256 | 15 | 3.912 | 0 |
| norm_squared_read_f64_n1024 | pytorch-cuda | 128 | 15 | 1.912 | 0 |
| tensordot_f32_n262144 | pytorch-cuda | 64 | 15 | 3.562 | 0.006191 |
| tensordot_f64_n262144 | pytorch-cuda | 2 | 15 | 3.266 | 6.714e-06 |
| tensordot_c32_n262144 | pytorch-cuda | 16 | 15 | 2.559 | 0.005717 |
| tensordot_c64_n262144 | pytorch-cuda | 1 | 15 | 5.278 | 4.401e-06 |
| vdot_read_f32_n262144 | pytorch-cuda | 256 | 15 | 3.945 | 0 |
| vdot_read_f64_n262144 | pytorch-cuda | 128 | 15 | 2.599 | 2.995e-06 |
| vdot_read_c32_n262144 | pytorch-cuda | 128 | 15 | 2.236 | 0 |
| vdot_read_c64_n262144 | pytorch-cuda | 64 | 15 | 2.171 | 8.762e-06 |
| norm_squared_read_f32_n262144 | pytorch-cuda | 128 | 15 | 2.001 | 0.0001786 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 256 | 15 | 3.855 | 0 |
| axpby_read_into_accum_f32_n1024 | pytorch-cuda | 256 | 15 | 2.000 | 0 |
| axpby_read_into_accum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.033 | 0 |
| axpby_read_into_accum_c32_n1024 | pytorch-cuda | 256 | 15 | 2.018 | 0 |
| axpby_read_into_accum_c64_n1024 | pytorch-cuda | 256 | 15 | 2.198 | 0 |
| axpby_read_into_accum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.543 | 0 |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| axpby_read_into_accum_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| axpby_read_into_accum_c64_n1048576 | pytorch-cuda | 16 | 15 | 2.457 | 0 |
| dot_general_read_into_f32_s16 | pytorch-cuda | 128 | 15 | 2.381 | 8.645e-05 |
| dot_general_read_into_accum_f32_s16 | pytorch-cuda | 128 | 15 | 2.923 | 7.907e-05 |
| dot_general_read_into_accum_cached_f32_s16 | pytorch-cuda | 128 | 15 | 2.873 | 7.907e-05 |
| dot_general_read_into_f64_s16 | pytorch-cuda | 128 | 15 | 2.828 | 0 |
| dot_general_read_into_accum_f64_s16 | pytorch-cuda | 128 | 15 | 3.382 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | pytorch-cuda | 128 | 15 | 3.309 | 0 |
| dot_general_read_into_c32_s16 | pytorch-cuda | 128 | 15 | 2.658 | 0.0001647 |
| dot_general_read_into_accum_c32_s16 | pytorch-cuda | 128 | 15 | 3.138 | 0.0001125 |
| dot_general_read_into_accum_cached_c32_s16 | pytorch-cuda | 128 | 15 | 3.095 | 0.0001125 |
| dot_general_read_into_c64_s16 | pytorch-cuda | 128 | 15 | 2.892 | 1.373e-07 |
| dot_general_read_into_accum_c64_s16 | pytorch-cuda | 128 | 15 | 3.353 | 7.37e-08 |
| dot_general_read_into_accum_cached_c64_s16 | pytorch-cuda | 128 | 15 | 3.361 | 7.37e-08 |
| dot_general_read_into_f32_s256 | pytorch-cuda | 128 | 15 | 2.485 | 0.0005542 |
| dot_general_read_into_accum_f32_s256 | pytorch-cuda | 128 | 15 | 2.886 | 0.0003617 |
| dot_general_read_into_accum_cached_f32_s256 | pytorch-cuda | 128 | 15 | 2.879 | 0.0003617 |
| dot_general_read_into_f64_s256 | pytorch-cuda | 16 | 15 | 3.952 | 0 |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 8 | 15 | 2.249 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 8 | 15 | 2.170 | 0 |
| dot_general_read_into_c32_s256 | pytorch-cuda | 128 | 15 | 3.465 | 0.0008017 |
| dot_general_read_into_accum_c32_s256 | pytorch-cuda | 128 | 15 | 3.551 | 0.000499 |
| dot_general_read_into_accum_cached_c32_s256 | pytorch-cuda | 128 | 15 | 3.571 | 0.000499 |
| dot_general_read_into_c64_s256 | pytorch-cuda | 2 | 15 | 1.967 | 1.175e-06 |
| dot_general_read_into_accum_c64_s256 | pytorch-cuda | 4 | 15 | 3.920 | 6.523e-07 |
| dot_general_read_into_accum_cached_c64_s256 | pytorch-cuda | 4 | 15 | 3.920 | 6.523e-07 |
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
| tensordot_f32_n1024 | jax-cuda | 16 | 15 | 3.119 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda | 16 | 15 | 2.626 | 0 |
| tensordot_c32_n1024 | jax-cuda | 16 | 15 | 2.854 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda | 16 | 15 | 3.191 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda | 32 | 15 | 2.779 | 0 |
| vdot_read_f64_n1024 | jax-cuda | 16 | 15 | 2.404 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda | 16 | 15 | 2.530 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda | 16 | 15 | 2.861 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda | 16 | 15 | 2.374 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda | 16 | 15 | 1.833 | 0 |
| tensordot_f32_n262144 | jax-cuda | 16 | 15 | 2.686 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda | 1 | 15 | 1.985 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda | 8 | 15 | 2.401 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda | 1 | 15 | 5.658 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda | 16 | 15 | 2.806 | 0 |
| vdot_read_f64_n262144 | jax-cuda | 16 | 15 | 2.590 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda | 16 | 15 | 2.224 | 0 |
| vdot_read_c64_n262144 | jax-cuda | 16 | 15 | 2.601 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda | 16 | 15 | 2.296 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda | 16 | 15 | 1.872 | 1.664e-07 |
| axpby_read_into_accum_f32_n1024 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f64_n1024 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c32_n1024 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c64_n1024 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f32_n1048576 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f64_n1048576 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c32_n1048576 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c64_n1048576 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
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
| tensordot_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.573 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.482 | 0 |
| tensordot_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.890 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.960 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.056 | 0 |
| vdot_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.891 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.640 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.355 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.569 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.096 | 0 |
| tensordot_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.185 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda-jit | 2 | 15 | 3.196 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.726 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda-jit | 1 | 15 | 5.719 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.844 | 0 |
| vdot_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.644 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.788 | 0 |
| vdot_read_c64_n262144 | jax-cuda-jit | 16 | 15 | 3.069 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.813 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.570 | 1.664e-07 |
| axpby_read_into_accum_f32_n1024 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f64_n1024 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c32_n1024 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c64_n1024 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f32_n1048576 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f64_n1048576 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c32_n1048576 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c64_n1048576 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
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

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: benchmarks/gpu/reuse_contraction.yaml
timestamp: '2026-10-13T17:04:14Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: c845efeeb96ea0868029fb2f9549852d2bed92a7
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
  expected_cases: 100
  selected_cases: 100
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: a081ed5568476b259df15e2e3c2b6c631bcfecfce4d4fb4fd43178409e36fac3
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/multi.rs: 5efc881b0010ed15d35aff3a42eca76847ab445a117ebd69bc9580d58c116dfa
    src/bin/benchmark_gpu_public_api/structural.rs: 8121b12b5fed3b0acbc71e2218856173d53a5acc93092b93c7fd3309d4f45962
    src/bin/benchmark_gpu_public_api/linalg.rs: 84162923fa0d0772c88ec9c4eab169ad3a75e5d16ff76961d1541670dd8565c1
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: a305c36801d5bee0d9843ab6d224918d67b5330006ef997a37d12a36bc95ef7f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: a30fcf2eb1777e04017dda9cf0a6664d53a0ddae256c79ccc49855c8018b50f3
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: ef5b07dd6e46e5f9b10e123273404eec276645721cc1855127764484ac58cabc
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/reuse_contraction/20261013_170413/fixtures/manifest.json: a17e6dea900749fcfabd3c4ea2f191a4b136e73cf66a2ea669ee6e4b41c539d7
    target/release/benchmark_gpu_public_api: 53889c545da2fd398beb2f0348abd6faf5acdf47462b156ab0803a29cb5371b8
  reference_numpy_version: 2.4.2
```
