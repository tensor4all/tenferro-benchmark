# gpu/reuse_contraction: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261013_165434/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | 7.432 (0.964) | unsupported | 7.075 (0.422) | unsupported | unsupported |
| add_into_f64_n1024 | 5.109 (0.065) | unsupported | 7.232 (0.307) | unsupported | unsupported |
| add_read_into_f32_n1024 | 5.198 (0.133) | unsupported | 6.911 (0.053) | unsupported | unsupported |
| add_read_into_f64_n1024 | 5.602 (0.186) | unsupported | 7.112 (0.143) | unsupported | unsupported |
| sub_into_f32_n1024 | 5.370 (0.248) | unsupported | 7.270 (0.227) | unsupported | unsupported |
| sub_into_f64_n1024 | 5.207 (0.185) | unsupported | 7.405 (0.073) | unsupported | unsupported |
| sub_read_into_f32_n1024 | 5.379 (0.204) | unsupported | 7.360 (0.422) | unsupported | unsupported |
| sub_read_into_f64_n1024 | 5.284 (0.136) | unsupported | 7.547 (0.184) | unsupported | unsupported |
| mul_into_f32_n1024 | 5.362 (0.159) | unsupported | 7.029 (0.088) | unsupported | unsupported |
| mul_into_f64_n1024 | 5.139 (0.128) | unsupported | 7.251 (0.168) | unsupported | unsupported |
| mul_read_into_f32_n1024 | 5.518 (0.130) | unsupported | 7.066 (0.061) | unsupported | unsupported |
| mul_read_into_f64_n1024 | 5.186 (0.105) | unsupported | 7.229 (0.055) | unsupported | unsupported |
| div_into_f32_n1024 | 63.516 (0.381) | unsupported | 6.927 (0.076) | unsupported | unsupported |
| div_into_f64_n1024 | 43.957 (2.052) | unsupported | 7.277 (0.436) | unsupported | unsupported |
| div_read_into_f32_n1024 | 44.472 (0.690) | unsupported | 6.833 (0.106) | unsupported | unsupported |
| div_read_into_f64_n1024 | 42.799 (1.274) | unsupported | 7.113 (0.102) | unsupported | unsupported |
| neg_into_f32_n1024 | 5.257 (0.318) | unsupported | 6.740 (0.072) | unsupported | unsupported |
| neg_into_f64_n1024 | 4.768 (0.137) | unsupported | 6.903 (0.074) | unsupported | unsupported |
| neg_read_into_f32_n1024 | 4.811 (0.127) | unsupported | 6.725 (0.023) | unsupported | unsupported |
| neg_read_into_f64_n1024 | 4.910 (0.284) | unsupported | 6.719 (0.191) | unsupported | unsupported |
| conj_into_c32_n1024 | 42.044 (2.982) | unsupported | 7.429 (0.118) | unsupported | unsupported |
| conj_into_c64_n1024 | 169.199 (18.222) | unsupported | 7.566 (0.134) | unsupported | unsupported |
| conj_read_into_c32_n1024 | 39.423 (0.489) | unsupported | 7.284 (0.053) | unsupported | unsupported |
| conj_read_into_c64_n1024 | 172.923 (12.687) | unsupported | 7.714 (0.059) | unsupported | unsupported |
| add_into_f32_n1048576 | 43.325 (0.385) | unsupported | 39.880 (0.096) | unsupported | unsupported |
| add_into_f64_n1048576 | 84.229 (1.213) | unsupported | 77.717 (0.071) | unsupported | unsupported |
| add_read_into_f32_n1048576 | 42.647 (0.145) | unsupported | 39.805 (0.103) | unsupported | unsupported |
| add_read_into_f64_n1048576 | 82.185 (0.671) | unsupported | 77.594 (0.054) | unsupported | unsupported |
| sub_into_f32_n1048576 | 42.555 (0.175) | unsupported | 39.700 (0.046) | unsupported | unsupported |
| sub_into_f64_n1048576 | 82.240 (0.490) | unsupported | 77.594 (0.061) | unsupported | unsupported |
| sub_read_into_f32_n1048576 | 42.901 (0.193) | unsupported | 39.698 (0.098) | unsupported | unsupported |
| sub_read_into_f64_n1048576 | 83.051 (0.473) | unsupported | 77.577 (0.047) | unsupported | unsupported |
| mul_into_f32_n1048576 | 42.778 (0.129) | unsupported | 39.686 (0.039) | unsupported | unsupported |
| mul_into_f64_n1048576 | 82.799 (0.667) | unsupported | 77.546 (0.076) | unsupported | unsupported |
| mul_read_into_f32_n1048576 | 42.766 (0.170) | unsupported | 39.683 (0.084) | unsupported | unsupported |
| mul_read_into_f64_n1048576 | 82.844 (0.434) | unsupported | 77.558 (0.057) | unsupported | unsupported |
| div_into_f32_n1048576 | 70.148 (0.183) | unsupported | 39.895 (0.058) | unsupported | unsupported |
| div_into_f64_n1048576 | 151.407 (0.140) | unsupported | 93.787 (0.107) | unsupported | unsupported |
| div_read_into_f32_n1048576 | 70.111 (0.174) | unsupported | 39.887 (0.018) | unsupported | unsupported |
| div_read_into_f64_n1048576 | 151.335 (0.087) | unsupported | 93.846 (0.094) | unsupported | unsupported |
| neg_into_f32_n1048576 | 29.543 (0.231) | unsupported | 27.227 (0.050) | unsupported | unsupported |
| neg_into_f64_n1048576 | 57.034 (0.254) | unsupported | 52.860 (0.021) | unsupported | unsupported |
| neg_read_into_f32_n1048576 | 29.497 (0.073) | unsupported | 27.221 (0.035) | unsupported | unsupported |
| neg_read_into_f64_n1048576 | 56.918 (0.454) | unsupported | 52.884 (0.063) | unsupported | unsupported |
| conj_into_c32_n1048576 | 107.857 (0.076) | unsupported | 52.822 (0.079) | unsupported | unsupported |
| conj_into_c64_n1048576 | 209.196 (0.250) | unsupported | 103.754 (0.142) | unsupported | unsupported |
| conj_read_into_c32_n1048576 | 107.828 (0.074) | unsupported | 52.880 (0.182) | unsupported | unsupported |
| conj_read_into_c64_n1048576 | 209.146 (0.254) | unsupported | 103.761 (0.134) | unsupported | unsupported |
| tensordot_f32_n1024 | 28.291 (0.585) | 36.723 (1.288) | 52.016 (2.240) | 182.124 (5.799) | 82.308 (2.448) |
| tensordot_f64_n1024 | 32.009 (0.873) | 39.908 (0.674) | 52.448 (1.005) | 177.143 (5.983) | 72.152 (1.809) |
| tensordot_c32_n1024 | 30.429 (0.806) | 36.427 (0.705) | 49.185 (0.417) | 176.678 (3.488) | 84.961 (1.911) |
| tensordot_c64_n1024 | 140.503 (3.855) | 40.614 (2.411) | 51.991 (3.210) | 190.508 (4.220) | 90.911 (1.312) |
| vdot_read_f32_n1024 | 169.663 (7.325) | unsupported | 15.479 (0.355) | 94.434 (1.351) | 91.500 (1.969) |
| vdot_read_f64_n1024 | 188.994 (3.395) | unsupported | 16.121 (0.391) | 152.323 (1.282) | 93.608 (1.196) |
| vdot_read_c32_n1024 | 191.767 (1.287) | unsupported | 14.972 (0.164) | 156.250 (8.560) | 112.033 (14.860) |
| vdot_read_c64_n1024 | 233.959 (1.506) | unsupported | 14.920 (0.311) | 177.783 (12.910) | 157.537 (21.818) |
| norm_squared_read_f32_n1024 | 203.335 (2.105) | unsupported | 15.331 (0.162) | 132.826 (27.582) | 106.473 (17.574) |
| norm_squared_read_f64_n1024 | 186.835 (1.612) | unsupported | 15.446 (0.213) | 99.586 (7.059) | 150.594 (6.694) |
| tensordot_f32_n262144 | 71.913 (2.674) | 45.995 (0.608) | 58.887 (0.987) | 189.621 (22.031) | 116.916 (18.093) |
| tensordot_f64_n262144 | 1685.352 (0.387) | 1617.596 (0.831) | 1622.762 (1.462) | 1719.743 (40.679) | 1625.529 (16.775) |
| tensordot_c32_n262144 | 146.233 (0.154) | 141.703 (1.438) | 158.844 (0.921) | 350.987 (29.677) | 160.588 (8.075) |
| tensordot_c64_n262144 | 5041.833 (1.229) | 5025.208 (3.205) | 5251.215 (1.889) | 5639.866 (100.225) | 5374.800 (29.086) |
| vdot_read_f32_n262144 | 204.384 (29.841) | unsupported | 16.204 (0.505) | 173.149 (2.748) | 175.685 (14.224) |
| vdot_read_f64_n262144 | 189.519 (6.847) | unsupported | 20.253 (0.021) | 168.489 (3.571) | 164.167 (9.052) |
| vdot_read_c32_n262144 | 182.773 (8.852) | unsupported | 17.371 (0.082) | 182.266 (17.493) | 176.075 (19.565) |
| vdot_read_c64_n262144 | 184.261 (5.063) | unsupported | 33.798 (0.053) | 171.874 (22.837) | 186.716 (22.985) |
| norm_squared_read_f32_n262144 | 176.876 (3.059) | unsupported | 15.294 (0.511) | 134.326 (23.335) | 166.632 (11.331) |
| norm_squared_read_f64_n262144 | 177.913 (2.743) | unsupported | 16.142 (0.987) | 105.924 (7.597) | 169.517 (6.266) |
| axpby_read_into_accum_f32_n1024 | 15.200 (0.307) | unsupported | 7.792 (0.138) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1024 | 10.255 (0.659) | unsupported | 7.854 (0.055) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1024 | 10.382 (0.249) | unsupported | 7.861 (0.078) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1024 | 10.107 (0.168) | unsupported | 7.849 (0.045) | unsupported | unsupported |
| axpby_read_into_accum_f32_n1048576 | 273.399 (0.139) | unsupported | 39.744 (0.025) | unsupported | unsupported |
| axpby_read_into_accum_f64_n1048576 | 534.720 (0.621) | unsupported | 77.751 (0.216) | unsupported | unsupported |
| axpby_read_into_accum_c32_n1048576 | 168.952 (0.244) | unsupported | 77.690 (0.123) | unsupported | unsupported |
| axpby_read_into_accum_c64_n1048576 | 2087.957 (3.333) | unsupported | 153.538 (0.137) | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.805 | 0 |
| add_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.616 | 0 |
| add_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.662 | 0 |
| add_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.868 | 0 |
| sub_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.750 | 0 |
| sub_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.666 | 0 |
| sub_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.754 | 0 |
| sub_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.705 | 0 |
| mul_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.745 | 0 |
| mul_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.631 | 0 |
| mul_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.825 | 0 |
| mul_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.655 | 0 |
| div_into_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.032 | 0 |
| div_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.813 | 0 |
| div_read_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.846 | 0 |
| div_read_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.739 | 0 |
| neg_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.691 | 0 |
| neg_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.441 | 0 |
| neg_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.463 | 0 |
| neg_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.514 | 0 |
| conj_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.691 | 0 |
| conj_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.707 | 0 |
| conj_read_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.523 | 0 |
| conj_read_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.767 | 0 |
| add_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.773 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.695 | 0 |
| add_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.729 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.630 | 0 |
| sub_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.723 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.632 | 0 |
| sub_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.746 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.658 | 0 |
| mul_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.738 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.650 | 0 |
| mul_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.737 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.651 | 0 |
| div_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.245 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.423 | 0 |
| div_read_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.244 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.421 | 0 |
| neg_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.891 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.825 | 0 |
| neg_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.888 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.821 | 0 |
| conj_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.451 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.347 | 0 |
| conj_read_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.450 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.346 | 0 |
| tensordot_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.621 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.049 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.947 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.248 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.715 | 0 |
| vdot_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.024 | 1.722e-07 |
| vdot_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.068 | 0 |
| vdot_read_c64_n1024 | tenferro-cuda-direct | 8 | 15 | 1.872 | 3.455e-07 |
| norm_squared_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.253 | 0 |
| norm_squared_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.989 | 0 |
| tensordot_f32_n262144 | tenferro-cuda-direct | 32 | 15 | 2.301 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 2 | 15 | 3.371 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.340 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-direct | 1 | 15 | 5.042 | 1.446e-05 |
| vdot_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 3.270 | 0 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.032 | 2.995e-06 |
| vdot_read_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.924 | 0 |
| vdot_read_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.948 | 8.762e-06 |
| norm_squared_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.830 | 0.0001786 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.847 | 0 |
| axpby_read_into_accum_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 1.946 | 0 |
| axpby_read_into_accum_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.625 | 0 |
| axpby_read_into_accum_c32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.658 | 0 |
| axpby_read_into_accum_c64_n1024 | tenferro-cuda-direct | 256 | 15 | 2.587 | 0 |
| axpby_read_into_accum_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.187 | 0 |
| axpby_read_into_accum_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.139 | 0 |
| axpby_read_into_accum_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.703 | 0 |
| axpby_read_into_accum_c64_n1048576 | tenferro-cuda-direct | 1 | 15 | 2.088 | 0 |
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
| tensordot_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.350 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.554 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.331 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.599 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| tensordot_f32_n262144 | tenferro-cuda-eager | 64 | 15 | 2.944 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 2 | 15 | 3.235 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-eager | 16 | 15 | 2.267 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-eager | 1 | 15 | 5.025 | 1.446e-05 |
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
| add_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.623 | 0 |
| add_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.703 | 0 |
| add_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.538 | 0 |
| add_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.641 | 0 |
| sub_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.722 | 0 |
| sub_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.791 | 0 |
| sub_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.768 | 0 |
| sub_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.864 | 0 |
| mul_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.599 | 0 |
| mul_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.712 | 0 |
| mul_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.618 | 0 |
| mul_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.701 | 0 |
| div_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.547 | 0 |
| div_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.726 | 0 |
| div_read_into_f32_n1024 | pytorch-cuda | 256 | 15 | 1.749 | 0 |
| div_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.642 | 0 |
| neg_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.451 | 0 |
| neg_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.534 | 0 |
| neg_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.443 | 0 |
| neg_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.440 | 0 |
| conj_into_c32_n1024 | pytorch-cuda | 512 | 15 | 3.804 | 0 |
| conj_into_c64_n1024 | pytorch-cuda | 512 | 15 | 3.874 | 0 |
| conj_read_into_c32_n1024 | pytorch-cuda | 512 | 15 | 3.729 | 0 |
| conj_read_into_c64_n1024 | pytorch-cuda | 512 | 15 | 3.950 | 0 |
| add_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.552 | 0 |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.487 | 0 |
| add_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.548 | 0 |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.483 | 0 |
| sub_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.541 | 0 |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.483 | 0 |
| sub_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.541 | 0 |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.482 | 0 |
| mul_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.540 | 0 |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.481 | 0 |
| mul_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.540 | 0 |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.482 | 0 |
| div_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.001 | 0 |
| div_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.003 | 0 |
| neg_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.743 | 0 |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.692 | 0 |
| neg_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.742 | 0 |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.692 | 0 |
| conj_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.690 | 0 |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.660 | 0 |
| conj_read_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.692 | 0 |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.660 | 0 |
| tensordot_f32_n1024 | pytorch-cuda | 64 | 15 | 3.329 | 0.0008859 |
| tensordot_f64_n1024 | pytorch-cuda | 64 | 15 | 3.357 | 0 |
| tensordot_c32_n1024 | pytorch-cuda | 64 | 15 | 3.148 | 0.0005344 |
| tensordot_c64_n1024 | pytorch-cuda | 64 | 15 | 3.327 | 0 |
| vdot_read_f32_n1024 | pytorch-cuda | 128 | 15 | 1.981 | 0 |
| vdot_read_f64_n1024 | pytorch-cuda | 128 | 15 | 2.064 | 1.722e-07 |
| vdot_read_c32_n1024 | pytorch-cuda | 256 | 15 | 3.833 | 0 |
| vdot_read_c64_n1024 | pytorch-cuda | 256 | 15 | 3.820 | 3.455e-07 |
| norm_squared_read_f32_n1024 | pytorch-cuda | 256 | 15 | 3.925 | 0 |
| norm_squared_read_f64_n1024 | pytorch-cuda | 256 | 15 | 3.954 | 0 |
| tensordot_f32_n262144 | pytorch-cuda | 64 | 15 | 3.769 | 0.006191 |
| tensordot_f64_n262144 | pytorch-cuda | 2 | 15 | 3.246 | 6.714e-06 |
| tensordot_c32_n262144 | pytorch-cuda | 16 | 15 | 2.542 | 0.005717 |
| tensordot_c64_n262144 | pytorch-cuda | 1 | 15 | 5.251 | 4.401e-06 |
| vdot_read_f32_n262144 | pytorch-cuda | 256 | 15 | 4.148 | 0 |
| vdot_read_f64_n262144 | pytorch-cuda | 128 | 15 | 2.592 | 2.995e-06 |
| vdot_read_c32_n262144 | pytorch-cuda | 128 | 15 | 2.223 | 0 |
| vdot_read_c64_n262144 | pytorch-cuda | 64 | 15 | 2.163 | 8.762e-06 |
| norm_squared_read_f32_n262144 | pytorch-cuda | 256 | 15 | 3.915 | 0.0001786 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 256 | 15 | 4.132 | 0 |
| axpby_read_into_accum_f32_n1024 | pytorch-cuda | 256 | 15 | 1.995 | 0 |
| axpby_read_into_accum_f64_n1024 | pytorch-cuda | 256 | 15 | 2.011 | 0 |
| axpby_read_into_accum_c32_n1024 | pytorch-cuda | 256 | 15 | 2.012 | 0 |
| axpby_read_into_accum_c64_n1024 | pytorch-cuda | 256 | 15 | 2.009 | 0 |
| axpby_read_into_accum_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| axpby_read_into_accum_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| axpby_read_into_accum_c32_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| axpby_read_into_accum_c64_n1048576 | pytorch-cuda | 16 | 15 | 2.457 | 0 |
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
| tensordot_f32_n1024 | jax-cuda | 16 | 15 | 2.914 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda | 16 | 15 | 2.834 | 0 |
| tensordot_c32_n1024 | jax-cuda | 16 | 15 | 2.827 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda | 16 | 15 | 3.048 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda | 32 | 15 | 3.022 | 0 |
| vdot_read_f64_n1024 | jax-cuda | 16 | 15 | 2.437 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda | 16 | 15 | 2.500 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda | 16 | 15 | 2.845 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda | 16 | 15 | 2.125 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda | 32 | 15 | 3.187 | 0 |
| tensordot_f32_n262144 | jax-cuda | 16 | 15 | 3.034 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda | 2 | 15 | 3.439 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda | 8 | 15 | 2.808 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda | 1 | 15 | 5.640 | 1.344e-05 |
| vdot_read_f32_n262144 | jax-cuda | 16 | 15 | 2.770 | 0 |
| vdot_read_f64_n262144 | jax-cuda | 16 | 15 | 2.696 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda | 16 | 15 | 2.916 | 0 |
| vdot_read_c64_n262144 | jax-cuda | 16 | 15 | 2.750 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda | 16 | 15 | 2.149 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda | 32 | 15 | 3.390 | 1.664e-07 |
| axpby_read_into_accum_f32_n1024 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f64_n1024 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c32_n1024 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c64_n1024 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f32_n1048576 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f64_n1048576 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c32_n1048576 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c64_n1048576 | jax-cuda | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
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
| tensordot_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.634 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.309 | 0 |
| tensordot_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.719 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.909 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.928 | 0 |
| vdot_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.498 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.585 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.521 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.407 | 0 |
| norm_squared_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.409 | 0 |
| tensordot_f32_n262144 | jax-cuda-jit | 16 | 15 | 1.871 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda-jit | 2 | 15 | 3.251 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.569 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda-jit | 1 | 15 | 5.375 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.811 | 0 |
| vdot_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.627 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.817 | 0 |
| vdot_read_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.987 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.666 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.712 | 1.664e-07 |
| axpby_read_into_accum_f32_n1024 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f64_n1024 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c32_n1024 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c64_n1024 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f32_n1048576 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_f64_n1048576 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c32_n1048576 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |
| axpby_read_into_accum_c64_n1048576 | jax-cuda-jit | — | — | — | No matching public caller-owned accumulating output API; donation is not output reuse. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: benchmarks/gpu/reuse_contraction.yaml
timestamp: '2026-10-13T16:54:35Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 66ffa13a4226a239dd0cc9c48b42bd52291dfd93
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
  expected_cases: 76
  selected_cases: 76
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: c75141c212523982d38529ee4121fdb03cce8ecf68c03f4727dc5cc87c7fb054
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/multi.rs: 5efc881b0010ed15d35aff3a42eca76847ab445a117ebd69bc9580d58c116dfa
    src/bin/benchmark_gpu_public_api/structural.rs: 8121b12b5fed3b0acbc71e2218856173d53a5acc93092b93c7fd3309d4f45962
    src/bin/benchmark_gpu_public_api/linalg.rs: 84162923fa0d0772c88ec9c4eab169ad3a75e5d16ff76961d1541670dd8565c1
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: 2c4bec1bada7ad58c4cff3e862019b225f8104a4cab71d590708079d9b930af0
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: acf139ff300e63e8184b333f45810eb9096274b7a09e06d37f8f0094f51c82a3
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: de547f615320737c5e886fd44230e2313d4c151d7dcd328c63fb07db91516921
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/reuse_contraction/20261013_165434/fixtures/manifest.json: 3dee5bdddf0f027f00368e5fc9d8cf39cb06a875a1cd7abedf05d880b7df6cba
    target/release/benchmark_gpu_public_api: 46257a3d8f50b9db64925458384d0aec615ed07c7e7c31198638aac3697dd0f6
  reference_numpy_version: 2.4.2
```
