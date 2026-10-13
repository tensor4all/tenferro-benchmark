# gpu/reuse_contraction: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261013_171503/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Transposed borrowed-read cases use prepared native CUDA views in tenferro and PyTorch. JAX compares the same logical values in its native representation; its columns do not claim an equivalent public strided-view contract. View creation and input conversion are outside timing.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | 6.733 (0.739) | unsupported | 6.257 (0.043) | unsupported | unsupported |
| add_into_f64_n1024 | 5.266 (0.155) | unsupported | 6.646 (0.193) | unsupported | unsupported |
| add_read_into_f32_n1024 | 5.336 (0.218) | unsupported | 6.215 (0.376) | unsupported | unsupported |
| add_read_into_f64_n1024 | 5.328 (0.256) | unsupported | 6.470 (0.088) | unsupported | unsupported |
| sub_into_f32_n1024 | 5.390 (0.245) | unsupported | 6.661 (0.299) | unsupported | unsupported |
| sub_into_f64_n1024 | 5.329 (0.268) | unsupported | 6.665 (0.059) | unsupported | unsupported |
| sub_read_into_f32_n1024 | 5.538 (0.223) | unsupported | 6.610 (0.063) | unsupported | unsupported |
| sub_read_into_f64_n1024 | 5.831 (0.325) | unsupported | 6.553 (0.092) | unsupported | unsupported |
| mul_into_f32_n1024 | 5.475 (0.310) | unsupported | 6.571 (0.069) | unsupported | unsupported |
| mul_into_f64_n1024 | 5.444 (0.138) | unsupported | 6.653 (0.090) | unsupported | unsupported |
| mul_read_into_f32_n1024 | 5.488 (0.105) | unsupported | 6.594 (0.149) | unsupported | unsupported |
| mul_read_into_f64_n1024 | 5.409 (0.114) | unsupported | 6.975 (0.356) | unsupported | unsupported |
| div_into_f32_n1024 | 40.018 (0.316) | unsupported | 6.506 (0.167) | unsupported | unsupported |
| div_into_f64_n1024 | 39.887 (2.226) | unsupported | 6.618 (0.218) | unsupported | unsupported |
| div_read_into_f32_n1024 | 39.060 (0.475) | unsupported | 6.619 (0.078) | unsupported | unsupported |
| div_read_into_f64_n1024 | 40.369 (0.917) | unsupported | 6.668 (0.050) | unsupported | unsupported |
| neg_into_f32_n1024 | 5.105 (0.300) | unsupported | 6.162 (0.159) | unsupported | unsupported |
| neg_into_f64_n1024 | 5.026 (0.135) | unsupported | 5.904 (0.167) | unsupported | unsupported |
| neg_read_into_f32_n1024 | 4.894 (0.088) | unsupported | 6.195 (0.293) | unsupported | unsupported |
| neg_read_into_f64_n1024 | 4.944 (0.180) | unsupported | 5.888 (0.026) | unsupported | unsupported |
| conj_into_c32_n1024 | 36.331 (0.660) | unsupported | 6.983 (0.346) | unsupported | unsupported |
| conj_into_c64_n1024 | 174.524 (3.043) | unsupported | 6.736 (0.175) | unsupported | unsupported |
| conj_read_into_c32_n1024 | 37.003 (0.515) | unsupported | 7.717 (0.174) | unsupported | unsupported |
| conj_read_into_c64_n1024 | 174.464 (5.431) | unsupported | 7.550 (0.322) | unsupported | unsupported |
| add_into_f32_n1048576 | 43.200 (0.246) | unsupported | 39.690 (0.034) | unsupported | unsupported |
| add_into_f64_n1048576 | 84.034 (0.793) | unsupported | 77.676 (0.142) | unsupported | unsupported |
| add_read_into_f32_n1048576 | 42.680 (0.201) | unsupported | 39.665 (0.072) | unsupported | unsupported |
| add_read_into_f64_n1048576 | 82.089 (0.537) | unsupported | 77.621 (0.071) | unsupported | unsupported |
| sub_into_f32_n1048576 | 42.672 (0.254) | unsupported | 39.650 (0.064) | unsupported | unsupported |
| sub_into_f64_n1048576 | 82.024 (0.535) | unsupported | 77.624 (0.115) | unsupported | unsupported |
| sub_read_into_f32_n1048576 | 42.687 (0.190) | unsupported | 39.722 (0.122) | unsupported | unsupported |
| sub_read_into_f64_n1048576 | 82.257 (0.511) | unsupported | 77.607 (0.054) | unsupported | unsupported |
| mul_into_f32_n1048576 | 42.628 (0.322) | unsupported | 39.642 (0.014) | unsupported | unsupported |
| mul_into_f64_n1048576 | 82.360 (0.350) | unsupported | 77.631 (0.507) | unsupported | unsupported |
| mul_read_into_f32_n1048576 | 42.894 (0.266) | unsupported | 39.682 (0.061) | unsupported | unsupported |
| mul_read_into_f64_n1048576 | 82.735 (0.590) | unsupported | 77.543 (0.094) | unsupported | unsupported |
| div_into_f32_n1048576 | 70.231 (0.222) | unsupported | 39.897 (0.026) | unsupported | unsupported |
| div_into_f64_n1048576 | 151.363 (0.168) | unsupported | 93.808 (0.131) | unsupported | unsupported |
| div_read_into_f32_n1048576 | 70.143 (0.111) | unsupported | 39.884 (0.083) | unsupported | unsupported |
| div_read_into_f64_n1048576 | 151.404 (0.481) | unsupported | 93.897 (0.237) | unsupported | unsupported |
| neg_into_f32_n1048576 | 29.473 (0.117) | unsupported | 27.207 (0.031) | unsupported | unsupported |
| neg_into_f64_n1048576 | 56.876 (0.289) | unsupported | 52.888 (0.058) | unsupported | unsupported |
| neg_read_into_f32_n1048576 | 29.915 (0.536) | unsupported | 27.230 (0.042) | unsupported | unsupported |
| neg_read_into_f64_n1048576 | 56.926 (0.272) | unsupported | 52.851 (0.042) | unsupported | unsupported |
| conj_into_c32_n1048576 | 107.909 (0.122) | unsupported | 52.822 (0.066) | unsupported | unsupported |
| conj_into_c64_n1048576 | 209.269 (0.429) | unsupported | 103.791 (0.295) | unsupported | unsupported |
| conj_read_into_c32_n1048576 | 107.966 (0.122) | unsupported | 52.812 (0.065) | unsupported | unsupported |
| conj_read_into_c64_n1048576 | 209.177 (0.459) | unsupported | 103.713 (0.079) | unsupported | unsupported |
| tensordot_f32_n1024 | 26.257 (0.320) | 38.971 (0.723) | 50.069 (1.513) | 180.461 (5.751) | 82.727 (1.925) |
| tensordot_f64_n1024 | 30.047 (0.835) | 43.739 (0.902) | 53.660 (0.497) | 155.057 (5.119) | 71.829 (1.982) |
| tensordot_c32_n1024 | 26.533 (0.689) | 40.147 (1.392) | 51.781 (1.468) | 170.394 (12.359) | 86.112 (1.231) |
| tensordot_c64_n1024 | 117.445 (2.375) | 44.838 (1.596) | 56.293 (2.296) | 171.499 (3.193) | 90.044 (1.072) |
| vdot_read_f32_n1024 | 141.050 (2.900) | unsupported | 14.986 (0.303) | 66.921 (2.602) | 92.516 (2.177) |
| vdot_read_f64_n1024 | 147.938 (6.689) | unsupported | 14.958 (0.258) | 68.563 (1.472) | 157.562 (7.085) |
| vdot_read_c32_n1024 | 161.401 (11.162) | unsupported | 15.177 (0.813) | 84.408 (1.649) | 172.917 (9.439) |
| vdot_read_c64_n1024 | 178.029 (3.501) | unsupported | 14.807 (0.341) | 132.039 (1.663) | 179.800 (22.352) |
| norm_squared_read_f32_n1024 | 165.754 (4.565) | unsupported | 14.862 (0.181) | 115.014 (16.465) | 158.350 (4.042) |
| norm_squared_read_f64_n1024 | 152.762 (2.593) | unsupported | 14.762 (0.081) | 84.843 (12.690) | 141.674 (11.510) |
| tensordot_f32_n262144 | 68.610 (2.709) | 48.350 (0.925) | 57.406 (0.548) | 154.134 (2.003) | 78.675 (20.922) |
| tensordot_f64_n262144 | 1639.704 (2.650) | 1617.152 (1.030) | 1628.282 (11.804) | 1710.935 (70.447) | 1606.569 (5.942) |
| tensordot_c32_n262144 | 140.504 (3.157) | 141.098 (0.203) | 158.718 (7.756) | 327.344 (18.656) | 161.878 (2.726) |
| tensordot_c64_n262144 | 5035.104 (7.078) | 5032.446 (4.348) | 5251.280 (7.119) | 5653.732 (242.995) | 5371.144 (20.231) |
| vdot_read_f32_n262144 | 398.120 (32.139) | unsupported | 15.788 (0.254) | 159.104 (22.820) | 166.384 (29.769) |
| vdot_read_f64_n262144 | 301.016 (15.355) | unsupported | 20.285 (0.014) | 158.419 (19.354) | 165.645 (15.044) |
| vdot_read_c32_n262144 | 283.680 (15.413) | unsupported | 17.402 (0.020) | 148.856 (1.906) | 159.337 (26.392) |
| vdot_read_c64_n262144 | 268.538 (8.554) | unsupported | 33.827 (0.026) | 193.699 (12.232) | 173.707 (20.339) |
| norm_squared_read_f32_n262144 | 246.342 (8.236) | unsupported | 15.281 (0.246) | 137.529 (25.073) | 153.062 (19.419) |
| norm_squared_read_f64_n262144 | 248.776 (1.928) | unsupported | 15.066 (0.503) | 107.310 (5.075) | 164.287 (6.990) |
| axpby_read_into_accum_f32_n1024 | 10.098 (0.219) | unsupported | 7.402 (0.194) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1024 | 10.629 (0.404) | unsupported | 7.566 (0.114) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1024 | 11.267 (0.563) | unsupported | 7.301 (0.201) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1024 | 10.653 (0.179) | unsupported | 8.241 (0.415) | unsupported | unsupported |
| axpby_read_into_accum_f32_n1048576 | 273.288 (0.489) | unsupported | 39.733 (0.058) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1048576 | 534.179 (0.321) | unsupported | 77.698 (0.079) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1048576 | 168.891 (0.213) | unsupported | 77.747 (0.191) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1048576 | 2144.725 (1.552) | unsupported | 153.612 (0.340) | unsupported | unsupported |
| dot_general_read_into_f32_s16 | 21.636 (0.368) | unsupported | 19.001 (0.193) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s16 | 21.467 (0.275) | unsupported | 23.293 (0.432) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s16 | 21.747 (0.331) | unsupported | 23.678 (1.252) | unsupported | unsupported |
| dot_general_read_into_f64_s16 | 23.998 (0.613) | unsupported | 21.779 (0.659) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s16 | 24.442 (0.937) | unsupported | 25.603 (0.301) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s16 | 22.582 (1.007) | unsupported | 26.899 (1.564) | unsupported | unsupported |
| dot_general_read_into_c32_s16 | 21.316 (0.533) | unsupported | 22.183 (0.284) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s16 | 21.576 (0.622) | unsupported | 25.858 (0.171) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s16 | 21.619 (0.881) | unsupported | 26.299 (0.371) | unsupported | unsupported |
| dot_general_read_into_c64_s16 | 24.077 (1.082) | unsupported | 23.424 (0.085) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s16 | 24.480 (0.702) | unsupported | 27.104 (0.366) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s16 | 24.377 (0.880) | unsupported | 27.744 (1.725) | unsupported | unsupported |
| dot_general_read_into_f32_s256 | 22.591 (0.875) | unsupported | 19.782 (0.227) | unsupported | unsupported |
| dot_general_read_into_accum_f32_s256 | 21.456 (0.385) | unsupported | 23.732 (0.571) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f32_s256 | 21.567 (0.611) | unsupported | 23.814 (0.327) | unsupported | unsupported |
| dot_general_read_into_f64_s256 | 262.773 (0.174) | unsupported | 296.837 (29.780) | unsupported | unsupported |
| dot_general_read_into_accum_f64_s256 | 308.140 (19.122) | unsupported | 328.877 (10.016) | unsupported | unsupported |
| dot_general_read_into_accum_cached_f64_s256 | 317.920 (9.993) | unsupported | 319.158 (9.854) | unsupported | unsupported |
| dot_general_read_into_c32_s256 | 28.164 (0.378) | unsupported | 26.829 (0.324) | unsupported | unsupported |
| dot_general_read_into_accum_c32_s256 | 29.896 (0.270) | unsupported | 32.612 (1.512) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c32_s256 | 29.817 (0.087) | unsupported | 30.157 (2.401) | unsupported | unsupported |
| dot_general_read_into_c64_s256 | 683.019 (0.504) | unsupported | 968.377 (0.358) | unsupported | unsupported |
| dot_general_read_into_accum_c64_s256 | 684.617 (0.593) | unsupported | 974.204 (0.319) | unsupported | unsupported |
| dot_general_read_into_accum_cached_c64_s256 | 684.433 (0.617) | unsupported | 974.204 (0.186) | unsupported | unsupported |
| add_strided_read_f32_n1024 | 282.361 (10.336) | unsupported | 9.865 (0.495) | 125.023 (6.959) | 154.333 (20.090) |
| add_strided_read_f64_n1024 | 242.084 (20.679) | unsupported | 9.945 (0.112) | 158.278 (30.840) | 159.917 (17.931) |
| mul_strided_read_f32_n1024 | 335.864 (9.679) | unsupported | 9.963 (0.186) | 169.938 (19.597) | 136.980 (28.584) |
| mul_strided_read_f64_n1024 | 303.034 (2.550) | unsupported | 10.433 (0.612) | 166.830 (35.148) | 123.853 (29.686) |
| neg_strided_read_f32_n1024 | 193.720 (8.279) | unsupported | 9.489 (0.215) | 152.125 (14.098) | 121.885 (15.337) |
| neg_strided_read_f64_n1024 | 177.107 (3.055) | unsupported | 9.267 (0.174) | 146.007 (24.045) | 117.589 (15.550) |
| exp_strided_read_f32_n1024 | 194.925 (3.492) | unsupported | 8.714 (0.108) | 137.480 (9.487) | 103.148 (20.018) |
| exp_strided_read_f64_n1024 | 169.929 (5.516) | unsupported | 8.779 (0.102) | 147.441 (19.096) | 97.905 (17.830) |
| tanh_strided_read_f32_n1024 | 193.457 (2.470) | unsupported | 8.950 (0.132) | 149.865 (14.985) | 96.528 (10.159) |
| tanh_strided_read_f64_n1024 | 168.692 (2.302) | unsupported | 8.981 (0.212) | 159.833 (30.342) | 99.643 (15.977) |
| reduce_sum_strided_read_f32_n1024 | 337.605 (20.018) | unsupported | 14.471 (0.204) | 198.054 (53.448) | 151.877 (11.407) |
| reduce_sum_strided_read_f64_n1024 | 294.514 (5.048) | unsupported | 14.554 (0.223) | 185.820 (11.748) | 163.377 (8.695) |
| conj_strided_read_c32_n1024 | 160.845 (6.623) | unsupported | 11.503 (0.171) | 188.979 (32.876) | 156.104 (27.549) |
| conj_strided_read_c64_n1024 | 242.868 (8.523) | unsupported | 11.762 (0.705) | 184.639 (36.299) | 144.487 (33.661) |
| add_strided_read_f32_n1048576 | 98.248 (0.311) | unsupported | 39.804 (0.049) | 166.182 (25.899) | 118.405 (7.752) |
| add_strided_read_f64_n1048576 | 188.536 (0.203) | unsupported | 77.800 (0.054) | 135.755 (3.505) | 99.145 (3.928) |
| mul_strided_read_f32_n1048576 | 97.893 (0.429) | unsupported | 39.787 (0.041) | 123.820 (0.396) | 92.527 (2.181) |
| mul_strided_read_f64_n1048576 | 188.504 (0.106) | unsupported | 77.724 (0.063) | 126.757 (0.970) | 86.069 (1.727) |
| neg_strided_read_f32_n1048576 | 56.350 (0.114) | unsupported | 27.475 (0.183) | 117.173 (0.886) | 87.133 (2.416) |
| neg_strided_read_f64_n1048576 | 108.378 (0.069) | unsupported | 53.139 (0.176) | 118.195 (0.384) | 85.409 (3.203) |
| exp_strided_read_f32_n1048576 | 56.323 (0.079) | unsupported | 27.432 (0.059) | 85.154 (5.429) | 83.670 (1.757) |
| exp_strided_read_f64_n1048576 | 233.512 (0.308) | unsupported | 179.933 (0.076) | 198.823 (2.226) | 204.403 (1.870) |
| tanh_strided_read_f32_n1048576 | 56.278 (0.089) | unsupported | 27.451 (0.047) | 83.071 (10.943) | 88.340 (2.187) |
| tanh_strided_read_f64_n1048576 | 220.381 (0.286) | unsupported | 166.712 (0.086) | 184.385 (1.038) | 189.616 (0.872) |
| reduce_sum_strided_read_f32_n1048576 | 186.765 (4.885) | unsupported | 21.408 (1.023) | 160.263 (6.446) | 151.124 (5.948) |
| reduce_sum_strided_read_f64_n1048576 | 92.874 (0.443) | unsupported | 37.053 (0.174) | 203.446 (19.455) | 143.862 (1.257) |
| conj_strided_read_c32_n1048576 | 114.299 (0.238) | unsupported | 53.277 (0.191) | 135.776 (4.363) | 92.428 (2.897) |
| conj_strided_read_c64_n1048576 | 213.998 (0.355) | unsupported | 104.178 (0.098) | 142.645 (3.352) | 117.096 (0.546) |

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
| tensordot_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.647 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.299 | 0 |
| tensordot_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.756 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.881 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.961 | 0 |
| vdot_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.521 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.767 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.877 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.534 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.267 | 0 |
| tensordot_f32_n262144 | jax-cuda-jit | 32 | 15 | 2.518 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda-jit | 2 | 15 | 3.213 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.590 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda-jit | 1 | 15 | 5.371 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.662 | 0 |
| vdot_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.650 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.549 | 0 |
| vdot_read_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.779 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.449 | 0.0001786 |
| norm_squared_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.629 | 1.664e-07 |
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
| add_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.469 | 0 |
| add_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.559 | 0 |
| mul_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.192 | 0 |
| mul_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.982 | 0 |
| neg_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.950 | 0 |
| neg_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.881 | 0 |
| exp_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.301 | 0.0001272 |
| exp_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.133 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.089 | 9.325e-05 |
| tanh_strided_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.189 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.430 | 0.0002267 |
| reduce_sum_strided_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.614 | 3.268e-07 |
| conj_strided_read_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.498 | 0 |
| conj_strided_read_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.312 | 0 |
| add_strided_read_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.894 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.586 | 0 |
| mul_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.961 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.754 | 0 |
| neg_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.788 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.733 | 0 |
| exp_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.677 | 0.0001272 |
| exp_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.270 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.827 | 9.508e-05 |
| tanh_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 3.034 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.418 | 0.003094 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.302 | 7.159e-06 |
| conj_strided_read_c32_n1048576 | jax-cuda-jit | 32 | 15 | 2.958 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.874 | 0 |
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
| tensordot_f32_n1024 | jax-cuda | 16 | 15 | 2.887 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda | 16 | 15 | 2.481 | 0 |
| tensordot_c32_n1024 | jax-cuda | 16 | 15 | 2.726 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda | 16 | 15 | 2.744 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda | 32 | 15 | 2.141 | 0 |
| vdot_read_f64_n1024 | jax-cuda | 32 | 15 | 2.194 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda | 32 | 15 | 2.701 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda | 16 | 15 | 2.113 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda | 16 | 15 | 1.840 | 0 |
| norm_squared_read_f64_n1024 | jax-cuda | 32 | 15 | 2.715 | 0 |
| tensordot_f32_n262144 | jax-cuda | 16 | 15 | 2.466 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda | 2 | 15 | 3.422 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda | 8 | 15 | 2.619 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda | 1 | 15 | 5.654 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda | 16 | 15 | 2.546 | 0 |
| vdot_read_f64_n262144 | jax-cuda | 16 | 15 | 2.535 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda | 16 | 15 | 2.382 | 0 |
| vdot_read_c64_n262144 | jax-cuda | 16 | 15 | 3.099 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda | 16 | 15 | 2.200 | 0.0001786 |
| norm_squared_read_f64_n262144 | jax-cuda | 32 | 15 | 3.434 | 1.664e-07 |
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
| add_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.000 | 0 |
| add_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.532 | 0 |
| mul_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.719 | 0 |
| mul_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.669 | 0 |
| neg_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.434 | 0 |
| neg_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.336 | 0 |
| exp_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.200 | 0.0001272 |
| exp_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.359 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 2.398 | 9.325e-05 |
| tanh_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.557 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | jax-cuda | 16 | 15 | 3.169 | 0.0002267 |
| reduce_sum_strided_read_f64_n1024 | jax-cuda | 16 | 15 | 2.973 | 3.268e-07 |
| conj_strided_read_c32_n1024 | jax-cuda | 16 | 15 | 3.024 | 0 |
| conj_strided_read_c64_n1024 | jax-cuda | 16 | 15 | 2.954 | 0 |
| add_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.659 | 0 |
| add_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.172 | 0 |
| mul_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 1.981 | 0 |
| mul_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.028 | 0 |
| neg_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 3.750 | 0 |
| neg_strided_read_f64_n1048576 | jax-cuda | 32 | 15 | 3.782 | 0 |
| exp_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 2.725 | 0.0001272 |
| exp_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.181 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | jax-cuda | 32 | 15 | 2.658 | 9.508e-05 |
| tanh_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 2.950 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | jax-cuda | 16 | 15 | 2.564 | 0.003094 |
| reduce_sum_strided_read_f64_n1048576 | jax-cuda | 16 | 15 | 3.255 | 7.159e-06 |
| conj_strided_read_c32_n1048576 | jax-cuda | 16 | 15 | 2.172 | 0 |
| conj_strided_read_c64_n1048576 | jax-cuda | 16 | 15 | 2.282 | 0 |
| add_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.203 | 0 |
| add_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.403 | 0 |
| add_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.182 | 0 |
| add_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.313 | 0 |
| sub_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.410 | 0 |
| sub_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.412 | 0 |
| sub_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.385 | 0 |
| sub_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.355 | 0 |
| mul_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.364 | 0 |
| mul_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.406 | 0 |
| mul_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.376 | 0 |
| mul_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.571 | 0 |
| div_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.331 | 0 |
| div_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.388 | 0 |
| div_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.389 | 0 |
| div_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.414 | 0 |
| neg_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.155 | 0 |
| neg_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.023 | 0 |
| neg_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.172 | 0 |
| neg_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.015 | 0 |
| conj_into_c32_n1024 | pytorch-cuda | 512 | 15 | 3.575 | 0 |
| conj_into_c64_n1024 | pytorch-cuda | 512 | 15 | 3.449 | 0 |
| conj_read_into_c32_n1024 | pytorch-cuda | 256 | 15 | 1.976 | 0 |
| conj_read_into_c64_n1024 | pytorch-cuda | 256 | 15 | 1.933 | 0 |
| add_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.540 | 0 |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| add_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.539 | 0 |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| sub_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.538 | 0 |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| sub_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.542 | 0 |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.483 | 0 |
| mul_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.537 | 0 |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| mul_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.540 | 0 |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.481 | 0 |
| div_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.002 | 0 |
| div_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.005 | 0 |
| neg_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.741 | 0 |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.692 | 0 |
| neg_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.743 | 0 |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.691 | 0 |
| conj_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.690 | 0 |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.661 | 0 |
| conj_read_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.690 | 0 |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.659 | 0 |
| tensordot_f32_n1024 | pytorch-cuda | 64 | 15 | 3.204 | 0.0008859 |
| tensordot_f64_n1024 | pytorch-cuda | 64 | 15 | 3.434 | 0 |
| tensordot_c32_n1024 | pytorch-cuda | 64 | 15 | 3.314 | 0.0005344 |
| tensordot_c64_n1024 | pytorch-cuda | 64 | 15 | 3.603 | 0 |
| vdot_read_f32_n1024 | pytorch-cuda | 128 | 15 | 1.918 | 0 |
| vdot_read_f64_n1024 | pytorch-cuda | 256 | 15 | 3.829 | 1.722e-07 |
| vdot_read_c32_n1024 | pytorch-cuda | 256 | 15 | 3.885 | 0 |
| vdot_read_c64_n1024 | pytorch-cuda | 256 | 15 | 3.791 | 3.455e-07 |
| norm_squared_read_f32_n1024 | pytorch-cuda | 128 | 15 | 1.902 | 0 |
| norm_squared_read_f64_n1024 | pytorch-cuda | 128 | 15 | 1.890 | 0 |
| tensordot_f32_n262144 | pytorch-cuda | 64 | 15 | 3.674 | 0.006191 |
| tensordot_f64_n262144 | pytorch-cuda | 2 | 15 | 3.257 | 6.714e-06 |
| tensordot_c32_n262144 | pytorch-cuda | 16 | 15 | 2.539 | 0.005717 |
| tensordot_c64_n262144 | pytorch-cuda | 1 | 15 | 5.251 | 4.401e-06 |
| vdot_read_f32_n262144 | pytorch-cuda | 128 | 15 | 2.021 | 0 |
| vdot_read_f64_n262144 | pytorch-cuda | 128 | 15 | 2.596 | 2.995e-06 |
| vdot_read_c32_n262144 | pytorch-cuda | 128 | 15 | 2.228 | 0 |
| vdot_read_c64_n262144 | pytorch-cuda | 64 | 15 | 2.165 | 8.762e-06 |
| norm_squared_read_f32_n262144 | pytorch-cuda | 256 | 15 | 3.912 | 0.0001786 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 256 | 15 | 3.857 | 0 |
| axpby_read_into_accum_f32_n1024 | pytorch-cuda | 512 | 15 | 3.790 | 0 |
| axpby_read_into_accum_f64_n1024 | pytorch-cuda | 512 | 15 | 3.874 | 0 |
| axpby_read_into_accum_c32_n1024 | pytorch-cuda | 512 | 15 | 3.738 | 0 |
| axpby_read_into_accum_c64_n1024 | pytorch-cuda | 256 | 15 | 2.110 | 0 |
| axpby_read_into_accum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.543 | 0 |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| axpby_read_into_accum_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| axpby_read_into_accum_c64_n1048576 | pytorch-cuda | 16 | 15 | 2.458 | 0 |
| dot_general_read_into_f32_s16 | pytorch-cuda | 128 | 15 | 2.432 | 8.645e-05 |
| dot_general_read_into_accum_f32_s16 | pytorch-cuda | 128 | 15 | 2.982 | 7.907e-05 |
| dot_general_read_into_accum_cached_f32_s16 | pytorch-cuda | 128 | 15 | 3.031 | 7.907e-05 |
| dot_general_read_into_f64_s16 | pytorch-cuda | 128 | 15 | 2.788 | 0 |
| dot_general_read_into_accum_f64_s16 | pytorch-cuda | 128 | 15 | 3.277 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | pytorch-cuda | 128 | 15 | 3.443 | 0 |
| dot_general_read_into_c32_s16 | pytorch-cuda | 128 | 15 | 2.839 | 0.0001647 |
| dot_general_read_into_accum_c32_s16 | pytorch-cuda | 128 | 15 | 3.310 | 0.0001125 |
| dot_general_read_into_accum_cached_c32_s16 | pytorch-cuda | 128 | 15 | 3.366 | 0.0001125 |
| dot_general_read_into_c64_s16 | pytorch-cuda | 128 | 15 | 2.998 | 1.373e-07 |
| dot_general_read_into_accum_c64_s16 | pytorch-cuda | 128 | 15 | 3.469 | 7.37e-08 |
| dot_general_read_into_accum_cached_c64_s16 | pytorch-cuda | 128 | 15 | 3.551 | 7.37e-08 |
| dot_general_read_into_f32_s256 | pytorch-cuda | 128 | 15 | 2.532 | 0.0005542 |
| dot_general_read_into_accum_f32_s256 | pytorch-cuda | 128 | 15 | 3.038 | 0.0003617 |
| dot_general_read_into_accum_cached_f32_s256 | pytorch-cuda | 128 | 15 | 3.048 | 0.0003617 |
| dot_general_read_into_f64_s256 | pytorch-cuda | 8 | 15 | 2.375 | 0 |
| dot_general_read_into_accum_f64_s256 | pytorch-cuda | 8 | 15 | 2.631 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | pytorch-cuda | 8 | 15 | 2.553 | 0 |
| dot_general_read_into_c32_s256 | pytorch-cuda | 128 | 15 | 3.434 | 0.0008017 |
| dot_general_read_into_accum_c32_s256 | pytorch-cuda | 64 | 15 | 2.087 | 0.000499 |
| dot_general_read_into_accum_cached_c32_s256 | pytorch-cuda | 64 | 15 | 1.930 | 0.000499 |
| dot_general_read_into_c64_s256 | pytorch-cuda | 4 | 15 | 3.874 | 1.175e-06 |
| dot_general_read_into_accum_c64_s256 | pytorch-cuda | 4 | 15 | 3.897 | 6.523e-07 |
| dot_general_read_into_accum_cached_c64_s256 | pytorch-cuda | 4 | 15 | 3.897 | 6.523e-07 |
| add_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.525 | 0 |
| add_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.546 | 0 |
| mul_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.550 | 0 |
| mul_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.671 | 0 |
| neg_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.429 | 0 |
| neg_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.372 | 0 |
| exp_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.231 | 0.0001272 |
| exp_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.247 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 2.291 | 9.107e-05 |
| tanh_strided_read_f64_n1024 | pytorch-cuda | 256 | 15 | 2.299 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | pytorch-cuda | 256 | 15 | 3.704 | 0.0005233 |
| reduce_sum_strided_read_f64_n1024 | pytorch-cuda | 128 | 15 | 1.863 | 3.504e-07 |
| conj_strided_read_c32_n1024 | pytorch-cuda | 256 | 15 | 2.945 | 0 |
| conj_strided_read_c64_n1024 | pytorch-cuda | 256 | 15 | 3.011 | 0 |
| add_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.547 | 0 |
| add_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.490 | 0 |
| mul_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.546 | 0 |
| mul_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.487 | 0 |
| neg_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.758 | 0 |
| neg_strided_read_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.700 | 0 |
| exp_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 0.0001272 |
| exp_strided_read_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.879 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.757 | 9.135e-05 |
| tanh_strided_read_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.667 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.740 | 0.002693 |
| reduce_sum_strided_read_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.371 | 5.935e-06 |
| conj_strided_read_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.705 | 0 |
| conj_strided_read_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.667 | 0 |
| add_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.447 | 0 |
| add_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.696 | 0 |
| add_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.732 | 0 |
| add_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.728 | 0 |
| sub_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.760 | 0 |
| sub_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.729 | 0 |
| sub_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.836 | 0 |
| sub_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.986 | 0 |
| mul_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.803 | 0 |
| mul_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.788 | 0 |
| mul_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.810 | 0 |
| mul_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.769 | 0 |
| div_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.561 | 0 |
| div_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.553 | 0 |
| div_read_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.500 | 0 |
| div_read_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.584 | 0 |
| neg_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.614 | 0 |
| neg_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.574 | 0 |
| neg_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.506 | 0 |
| neg_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.531 | 0 |
| conj_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.325 | 0 |
| conj_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.792 | 0 |
| conj_read_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.368 | 0 |
| conj_read_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.791 | 0 |
| add_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.765 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.689 | 0 |
| add_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.732 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.627 | 0 |
| sub_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.731 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.625 | 0 |
| sub_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.732 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.632 | 0 |
| mul_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.728 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.636 | 0 |
| mul_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.745 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.648 | 0 |
| div_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.247 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.422 | 0 |
| div_read_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.245 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.422 | 0 |
| neg_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.886 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.820 | 0 |
| neg_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.915 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.822 | 0 |
| conj_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.453 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.348 | 0 |
| conj_read_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.455 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.347 | 0 |
| tensordot_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.361 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.846 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.396 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 1.879 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.257 | 0 |
| vdot_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.367 | 1.722e-07 |
| vdot_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.582 | 0 |
| vdot_read_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.848 | 3.455e-07 |
| norm_squared_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.652 | 0 |
| norm_squared_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.444 | 0 |
| tensordot_f32_n262144 | tenferro-cuda-direct | 32 | 15 | 2.196 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 2 | 15 | 3.279 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.248 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-direct | 1 | 15 | 5.035 | 1.446e-05 |
| vdot_read_f32_n262144 | tenferro-cuda-direct | 8 | 15 | 3.185 | 0 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 8 | 15 | 2.408 | 2.995e-06 |
| vdot_read_c32_n262144 | tenferro-cuda-direct | 8 | 15 | 2.269 | 0 |
| vdot_read_c64_n262144 | tenferro-cuda-direct | 8 | 15 | 2.148 | 8.762e-06 |
| norm_squared_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 3.941 | 0.0001786 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 8 | 15 | 1.990 | 0 |
| axpby_read_into_accum_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.585 | 0 |
| axpby_read_into_accum_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.721 | 0 |
| axpby_read_into_accum_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.884 | 0 |
| axpby_read_into_accum_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.727 | 0 |
| axpby_read_into_accum_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.186 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.137 | 0 |
| axpby_read_into_accum_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.702 | 0 |
| axpby_read_into_accum_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.145 | 0 |
| dot_general_read_into_f32_s16 | tenferro-cuda-direct | 128 | 15 | 2.769 | 8.645e-05 |
| dot_general_read_into_accum_f32_s16 | tenferro-cuda-direct | 128 | 15 | 2.748 | 7.907e-05 |
| dot_general_read_into_accum_cached_f32_s16 | tenferro-cuda-direct | 128 | 15 | 2.784 | 7.907e-05 |
| dot_general_read_into_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.072 | 0 |
| dot_general_read_into_accum_f64_s16 | tenferro-cuda-direct | 128 | 15 | 3.129 | 0 |
| dot_general_read_into_accum_cached_f64_s16 | tenferro-cuda-direct | 128 | 15 | 2.891 | 0 |
| dot_general_read_into_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.728 | 0.0003615 |
| dot_general_read_into_accum_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.762 | 0.0001632 |
| dot_general_read_into_accum_cached_c32_s16 | tenferro-cuda-direct | 128 | 15 | 2.767 | 0.0001632 |
| dot_general_read_into_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.082 | 1.204e-07 |
| dot_general_read_into_accum_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.133 | 6.975e-08 |
| dot_general_read_into_accum_cached_c64_s16 | tenferro-cuda-direct | 128 | 15 | 3.120 | 6.975e-08 |
| dot_general_read_into_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.892 | 0.0005542 |
| dot_general_read_into_accum_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.746 | 0.0003617 |
| dot_general_read_into_accum_cached_f32_s256 | tenferro-cuda-direct | 128 | 15 | 2.761 | 0.0003617 |
| dot_general_read_into_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.102 | 0 |
| dot_general_read_into_accum_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.465 | 0 |
| dot_general_read_into_accum_cached_f64_s256 | tenferro-cuda-direct | 8 | 15 | 2.543 | 0 |
| dot_general_read_into_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.605 | 0.001457 |
| dot_general_read_into_accum_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.827 | 0.0008947 |
| dot_general_read_into_accum_cached_c32_s256 | tenferro-cuda-direct | 128 | 15 | 3.817 | 0.0008947 |
| dot_general_read_into_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.732 | 2.092e-06 |
| dot_general_read_into_accum_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.738 | 1.318e-06 |
| dot_general_read_into_accum_cached_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.738 | 1.318e-06 |
| add_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.259 | 0 |
| add_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.873 | 0 |
| mul_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.687 | 0 |
| mul_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.424 | 0 |
| neg_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.100 | 0 |
| neg_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.834 | 0 |
| exp_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.119 | 0.0001272 |
| exp_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.719 | 1.095e-07 |
| tanh_strided_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.095 | 9.107e-05 |
| tanh_strided_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.699 | 4.434e-08 |
| reduce_sum_strided_read_f32_n1024 | tenferro-cuda-direct | 8 | 15 | 2.701 | 0.0001208 |
| reduce_sum_strided_read_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.356 | 4.202e-07 |
| conj_strided_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.574 | 0 |
| conj_strided_read_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.886 | 0 |
| add_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.144 | 0 |
| add_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.017 | 0 |
| mul_strided_read_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.133 | 0 |
| mul_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.016 | 0 |
| neg_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.606 | 0 |
| neg_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.468 | 0 |
| exp_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.605 | 0.0001272 |
| exp_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.736 | 1.095e-07 |
| tanh_strided_read_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.602 | 9.135e-05 |
| tanh_strided_read_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.526 | 4.437e-08 |
| reduce_sum_strided_read_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.988 | 0.002489 |
| reduce_sum_strided_read_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.972 | 3.227e-06 |
| conj_strided_read_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.658 | 0 |
| conj_strided_read_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.424 | 0 |
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
| tensordot_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.494 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.799 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.569 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.870 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| tensordot_f32_n262144 | tenferro-cuda-eager | 16 | 15 | 0.774 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 2 | 15 | 3.234 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-eager | 16 | 15 | 2.258 | 0.01583 |
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

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: benchmarks/gpu/reuse_contraction.yaml
timestamp: '2026-10-13T17:15:03Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 8f9892841017907c96ac65a7ffa5ce31e2809ac2
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
  expected_cases: 128
  selected_cases: 128
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 2af6e1ff3c0ea6061318abf4ec34116e3398c335bf6dd9e615fc8e2736db9c7f
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/multi.rs: 5efc881b0010ed15d35aff3a42eca76847ab445a117ebd69bc9580d58c116dfa
    src/bin/benchmark_gpu_public_api/read_layout.rs: bd7ce4a42576d1937a7048b7b002cac4838a7cd3c53628bc9aa6aebb0c5c1128
    src/bin/benchmark_gpu_public_api/structural.rs: 8121b12b5fed3b0acbc71e2218856173d53a5acc93092b93c7fd3309d4f45962
    src/bin/benchmark_gpu_public_api/linalg.rs: 84162923fa0d0772c88ec9c4eab169ad3a75e5d16ff76961d1541670dd8565c1
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: a305c36801d5bee0d9843ab6d224918d67b5330006ef997a37d12a36bc95ef7f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: bbf982ad1b55067bf8dabc8676aa8ffe2fe5a1ff769f979b66eca4e2d5eba6e6
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: dce1cbb78e583da5684895eb51ef5914703b7dac73d207c7c0bd76384b936e41
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/reuse_contraction/20261013_171503/fixtures/manifest.json: 35c755b54492476653ea5a4833f4d1bd2465c717f6392dc5add59b431be19851
    target/release/benchmark_gpu_public_api: c31f9c954d3341505d1f24fe317062b2fb75f6f4ed4af7fb0205cad3b3d8f4bd
  reference_numpy_version: 2.4.2
```
