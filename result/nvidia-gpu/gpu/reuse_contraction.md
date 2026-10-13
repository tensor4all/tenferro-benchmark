# gpu/reuse_contraction: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261013_134253/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | 6.638 (0.743) | unsupported | 7.397 (0.266) | unsupported | unsupported |
| add_into_f64_n1024 | 5.457 (0.063) | unsupported | 7.615 (0.278) | unsupported | unsupported |
| add_read_into_f32_n1024 | 5.562 (0.069) | unsupported | 7.690 (0.454) | unsupported | unsupported |
| add_read_into_f64_n1024 | 5.556 (0.071) | unsupported | 7.926 (0.116) | unsupported | unsupported |
| sub_into_f32_n1024 | 5.595 (0.226) | unsupported | 7.949 (0.221) | unsupported | unsupported |
| sub_into_f64_n1024 | 5.618 (0.275) | unsupported | 7.994 (0.170) | unsupported | unsupported |
| sub_read_into_f32_n1024 | 6.019 (0.230) | unsupported | 7.826 (0.186) | unsupported | unsupported |
| sub_read_into_f64_n1024 | 5.630 (0.101) | unsupported | 7.657 (0.137) | unsupported | unsupported |
| mul_into_f32_n1024 | 5.757 (0.052) | unsupported | 7.634 (0.130) | unsupported | unsupported |
| mul_into_f64_n1024 | 5.925 (0.110) | unsupported | 7.830 (0.158) | unsupported | unsupported |
| mul_read_into_f32_n1024 | 6.556 (0.063) | unsupported | 7.637 (0.286) | unsupported | unsupported |
| mul_read_into_f64_n1024 | 6.378 (0.070) | unsupported | 7.867 (0.382) | unsupported | unsupported |
| div_into_f32_n1024 | 58.708 (1.798) | unsupported | 7.519 (0.463) | unsupported | unsupported |
| div_into_f64_n1024 | 47.913 (0.431) | unsupported | 7.664 (0.513) | unsupported | unsupported |
| div_read_into_f32_n1024 | 48.355 (0.138) | unsupported | 7.546 (0.450) | unsupported | unsupported |
| div_read_into_f64_n1024 | 47.511 (0.990) | unsupported | 7.343 (0.608) | unsupported | unsupported |
| neg_into_f32_n1024 | 5.858 (0.132) | unsupported | 7.018 (0.303) | unsupported | unsupported |
| neg_into_f64_n1024 | 5.789 (0.119) | unsupported | 6.898 (0.185) | unsupported | unsupported |
| neg_read_into_f32_n1024 | 5.631 (0.072) | unsupported | 6.832 (0.326) | unsupported | unsupported |
| neg_read_into_f64_n1024 | 5.628 (0.091) | unsupported | 7.188 (0.451) | unsupported | unsupported |
| conj_into_c32_n1024 | 46.543 (0.558) | unsupported | 7.329 (0.352) | unsupported | unsupported |
| conj_into_c64_n1024 | 194.869 (10.834) | unsupported | 7.766 (0.532) | unsupported | unsupported |
| conj_read_into_c32_n1024 | 45.082 (0.252) | unsupported | 7.359 (0.226) | unsupported | unsupported |
| conj_read_into_c64_n1024 | 218.918 (37.283) | unsupported | 7.650 (0.253) | unsupported | unsupported |
| add_into_f32_n1048576 | 46.259 (0.233) | unsupported | 39.796 (0.115) | unsupported | unsupported |
| add_into_f64_n1048576 | 90.232 (0.801) | unsupported | 77.792 (0.256) | unsupported | unsupported |
| add_read_into_f32_n1048576 | 46.777 (0.389) | unsupported | 39.951 (0.114) | unsupported | unsupported |
| add_read_into_f64_n1048576 | 90.040 (0.578) | unsupported | 77.630 (0.116) | unsupported | unsupported |
| sub_into_f32_n1048576 | 46.788 (0.140) | unsupported | 39.740 (0.132) | unsupported | unsupported |
| sub_into_f64_n1048576 | 91.123 (0.529) | unsupported | 77.811 (0.144) | unsupported | unsupported |
| sub_read_into_f32_n1048576 | 47.315 (0.191) | unsupported | 39.801 (0.104) | unsupported | unsupported |
| sub_read_into_f64_n1048576 | 91.442 (0.724) | unsupported | 77.609 (0.124) | unsupported | unsupported |
| mul_into_f32_n1048576 | 46.376 (0.160) | unsupported | 39.691 (0.096) | unsupported | unsupported |
| mul_into_f64_n1048576 | 89.076 (0.791) | unsupported | 77.523 (0.115) | unsupported | unsupported |
| mul_read_into_f32_n1048576 | 43.432 (0.198) | unsupported | 39.676 (0.041) | unsupported | unsupported |
| mul_read_into_f64_n1048576 | 83.413 (0.401) | unsupported | 77.525 (0.044) | unsupported | unsupported |
| div_into_f32_n1048576 | 70.124 (0.091) | unsupported | 39.915 (0.106) | unsupported | unsupported |
| div_into_f64_n1048576 | 152.431 (0.208) | unsupported | 94.238 (0.102) | unsupported | unsupported |
| div_read_into_f32_n1048576 | 70.426 (0.286) | unsupported | 39.886 (0.054) | unsupported | unsupported |
| div_read_into_f64_n1048576 | 152.566 (0.272) | unsupported | 93.345 (0.160) | unsupported | unsupported |
| neg_into_f32_n1048576 | 30.102 (0.165) | unsupported | 27.185 (0.035) | unsupported | unsupported |
| neg_into_f64_n1048576 | 57.955 (0.220) | unsupported | 52.813 (0.044) | unsupported | unsupported |
| neg_read_into_f32_n1048576 | 29.968 (0.492) | unsupported | 27.180 (0.050) | unsupported | unsupported |
| neg_read_into_f64_n1048576 | 57.464 (0.336) | unsupported | 52.840 (0.060) | unsupported | unsupported |
| conj_into_c32_n1048576 | 108.159 (0.350) | unsupported | 52.794 (0.041) | unsupported | unsupported |
| conj_into_c64_n1048576 | 208.996 (0.259) | unsupported | 103.649 (0.138) | unsupported | unsupported |
| conj_read_into_c32_n1048576 | 107.881 (0.105) | unsupported | 52.883 (0.102) | unsupported | unsupported |
| conj_read_into_c64_n1048576 | 208.996 (0.317) | unsupported | 103.706 (0.232) | unsupported | unsupported |
| tensordot_f32_n1024 | 28.558 (1.326) | 67.964 (2.542) | 53.754 (2.854) | 200.509 (0.997) | 76.737 (0.807) |
| tensordot_f64_n1024 | 31.466 (1.735) | 69.197 (2.439) | 58.744 (2.307) | 200.098 (1.280) | 73.926 (0.448) |
| tensordot_c32_n1024 | 31.264 (0.831) | 64.881 (3.416) | 57.983 (0.608) | 216.900 (0.719) | 85.834 (2.776) |
| tensordot_c64_n1024 | 129.600 (6.664) | 46.925 (0.521) | 59.184 (0.836) | 220.978 (2.194) | 90.001 (2.272) |
| vdot_read_f32_n1024 | 161.639 (5.375) | unsupported | 16.991 (0.253) | 125.859 (0.469) | 87.456 (1.351) |
| vdot_read_f64_n1024 | 170.897 (4.991) | unsupported | 17.112 (0.209) | 151.031 (3.027) | 146.799 (14.041) |
| vdot_read_c32_n1024 | 168.142 (3.078) | unsupported | 16.861 (0.216) | 183.355 (18.769) | 147.399 (17.079) |
| vdot_read_c64_n1024 | 199.104 (5.696) | unsupported | 17.367 (0.405) | 169.808 (7.471) | 156.016 (18.041) |
| norm_squared_read_f32_n1024 | 179.386 (5.638) | unsupported | 16.999 (0.264) | 132.454 (29.789) | 140.968 (2.565) |
| norm_squared_read_f64_n1024 | 164.676 (2.839) | unsupported | 17.132 (0.300) | 108.352 (0.565) | 97.657 (22.530) |
| tensordot_f32_n262144 | 70.105 (4.250) | 51.789 (0.750) | 67.824 (0.511) | 261.826 (20.026) | 116.116 (4.631) |
| tensordot_f64_n262144 | 1725.668 (1.790) | 1637.566 (4.903) | 1669.814 (10.524) | 1660.775 (16.160) | 1579.376 (13.842) |
| tensordot_c32_n262144 | 149.225 (2.530) | 142.041 (0.131) | 158.401 (0.310) | 306.849 (24.141) | 161.931 (4.298) |
| tensordot_c64_n262144 | 5054.983 (4.386) | 5130.622 (34.553) | 5342.469 (29.894) | 5496.933 (23.160) | 5331.396 (16.978) |
| vdot_read_f32_n262144 | 183.645 (3.304) | unsupported | 16.679 (0.259) | 165.266 (2.633) | 156.171 (5.970) |
| vdot_read_f64_n262144 | 198.931 (3.332) | unsupported | 20.185 (0.019) | 139.392 (1.613) | 157.648 (14.858) |
| vdot_read_c32_n262144 | 198.099 (4.920) | unsupported | 17.341 (0.024) | 159.984 (7.257) | 157.905 (23.919) |
| vdot_read_c64_n262144 | 197.869 (4.883) | unsupported | 33.745 (0.052) | 161.500 (6.752) | 139.164 (5.176) |
| norm_squared_read_f32_n262144 | 201.221 (6.926) | unsupported | 16.517 (0.289) | 131.338 (37.172) | 92.098 (16.508) |
| norm_squared_read_f64_n262144 | 200.588 (2.925) | unsupported | 16.697 (0.277) | 91.374 (17.316) | 151.883 (2.312) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.399 | 0 |
| add_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.794 | 0 |
| add_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.848 | 0 |
| add_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.845 | 0 |
| sub_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.865 | 0 |
| sub_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.876 | 0 |
| sub_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.082 | 0 |
| sub_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.883 | 0 |
| mul_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.948 | 0 |
| mul_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 3.034 | 0 |
| mul_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.357 | 0 |
| mul_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 3.266 | 0 |
| div_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.757 | 0 |
| div_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.066 | 0 |
| div_read_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.095 | 0 |
| div_read_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.041 | 0 |
| neg_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 3.000 | 0 |
| neg_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.964 | 0 |
| neg_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.883 | 0 |
| neg_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.881 | 0 |
| conj_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.979 | 0 |
| conj_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.118 | 0 |
| conj_read_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.885 | 0 |
| conj_read_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.503 | 0 |
| add_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.961 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.887 | 0 |
| add_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.994 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.881 | 0 |
| sub_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.994 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.916 | 0 |
| sub_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.028 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.926 | 0 |
| mul_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.968 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.850 | 0 |
| mul_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.780 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.669 | 0 |
| div_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.244 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.439 | 0 |
| div_read_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.254 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.441 | 0 |
| neg_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.927 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.855 | 0 |
| neg_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.918 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.839 | 0 |
| conj_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.461 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.344 | 0 |
| conj_read_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.452 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.344 | 0 |
| tensordot_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.655 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.014 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 4.002 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-direct | 32 | 15 | 4.147 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.586 | 0 |
| vdot_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.734 | 1.722e-07 |
| vdot_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.690 | 0 |
| vdot_read_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.186 | 3.455e-07 |
| norm_squared_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.870 | 0 |
| norm_squared_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.635 | 0 |
| tensordot_f32_n262144 | tenferro-cuda-direct | 32 | 15 | 2.243 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 2 | 15 | 3.451 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.388 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-direct | 1 | 15 | 5.055 | 1.446e-05 |
| vdot_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.938 | 0 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.183 | 2.995e-06 |
| vdot_read_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 3.170 | 0 |
| vdot_read_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.166 | 8.762e-06 |
| norm_squared_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 3.220 | 0.0001786 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.209 | 0 |
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
| tensordot_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.175 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.214 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.076 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.003 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| tensordot_f32_n262144 | tenferro-cuda-eager | 32 | 15 | 1.657 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 2 | 15 | 3.275 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-eager | 16 | 15 | 2.273 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-eager | 1 | 15 | 5.131 | 1.446e-05 |
| vdot_read_f32_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_f64_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c32_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c64_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f32_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f64_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| add_into_f32_n1024 | pytorch-cuda | 256 | 15 | 1.894 | 0 |
| add_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.899 | 0 |
| add_read_into_f32_n1024 | pytorch-cuda | 256 | 15 | 1.969 | 0 |
| add_read_into_f64_n1024 | pytorch-cuda | 256 | 15 | 2.029 | 0 |
| sub_into_f32_n1024 | pytorch-cuda | 256 | 15 | 2.035 | 0 |
| sub_into_f64_n1024 | pytorch-cuda | 256 | 15 | 2.046 | 0 |
| sub_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 4.007 | 0 |
| sub_read_into_f64_n1024 | pytorch-cuda | 256 | 15 | 1.960 | 0 |
| mul_into_f32_n1024 | pytorch-cuda | 256 | 15 | 1.954 | 0 |
| mul_into_f64_n1024 | pytorch-cuda | 512 | 15 | 4.009 | 0 |
| mul_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.910 | 0 |
| mul_read_into_f64_n1024 | pytorch-cuda | 256 | 15 | 2.014 | 0 |
| div_into_f32_n1024 | pytorch-cuda | 256 | 15 | 1.925 | 0 |
| div_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.924 | 0 |
| div_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.863 | 0 |
| div_read_into_f64_n1024 | pytorch-cuda | 256 | 15 | 1.880 | 0 |
| neg_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.593 | 0 |
| neg_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.532 | 0 |
| neg_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.498 | 0 |
| neg_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.680 | 0 |
| conj_into_c32_n1024 | pytorch-cuda | 256 | 15 | 1.876 | 0 |
| conj_into_c64_n1024 | pytorch-cuda | 256 | 15 | 1.988 | 0 |
| conj_read_into_c32_n1024 | pytorch-cuda | 256 | 15 | 1.884 | 0 |
| conj_read_into_c64_n1024 | pytorch-cuda | 256 | 15 | 1.958 | 0 |
| add_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.547 | 0 |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.489 | 0 |
| add_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.557 | 0 |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.484 | 0 |
| sub_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.543 | 0 |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.490 | 0 |
| sub_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.547 | 0 |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.483 | 0 |
| mul_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.540 | 0 |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.481 | 0 |
| mul_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.539 | 0 |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.481 | 0 |
| div_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.555 | 0 |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.016 | 0 |
| div_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.987 | 0 |
| neg_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.740 | 0 |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.690 | 0 |
| neg_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.740 | 0 |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.691 | 0 |
| conj_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.689 | 0 |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.658 | 0 |
| conj_read_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.692 | 0 |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.659 | 0 |
| tensordot_f32_n1024 | pytorch-cuda | 64 | 15 | 3.440 | 0.0008859 |
| tensordot_f64_n1024 | pytorch-cuda | 64 | 15 | 3.760 | 0 |
| tensordot_c32_n1024 | pytorch-cuda | 64 | 15 | 3.711 | 0.0005344 |
| tensordot_c64_n1024 | pytorch-cuda | 64 | 15 | 3.788 | 0 |
| vdot_read_f32_n1024 | pytorch-cuda | 128 | 15 | 2.175 | 0 |
| vdot_read_f64_n1024 | pytorch-cuda | 128 | 15 | 2.190 | 1.722e-07 |
| vdot_read_c32_n1024 | pytorch-cuda | 128 | 15 | 2.158 | 0 |
| vdot_read_c64_n1024 | pytorch-cuda | 128 | 15 | 2.223 | 3.455e-07 |
| norm_squared_read_f32_n1024 | pytorch-cuda | 128 | 15 | 2.176 | 0 |
| norm_squared_read_f64_n1024 | pytorch-cuda | 128 | 15 | 2.193 | 0 |
| tensordot_f32_n262144 | pytorch-cuda | 32 | 15 | 2.170 | 0.006191 |
| tensordot_f64_n262144 | pytorch-cuda | 2 | 15 | 3.340 | 6.714e-06 |
| tensordot_c32_n262144 | pytorch-cuda | 16 | 15 | 2.534 | 0.005717 |
| tensordot_c64_n262144 | pytorch-cuda | 1 | 15 | 5.342 | 4.401e-06 |
| vdot_read_f32_n262144 | pytorch-cuda | 128 | 15 | 2.135 | 0 |
| vdot_read_f64_n262144 | pytorch-cuda | 128 | 15 | 2.584 | 2.995e-06 |
| vdot_read_c32_n262144 | pytorch-cuda | 128 | 15 | 2.220 | 0 |
| vdot_read_c64_n262144 | pytorch-cuda | 64 | 15 | 2.160 | 8.762e-06 |
| norm_squared_read_f32_n262144 | pytorch-cuda | 128 | 15 | 2.114 | 0.0001786 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 128 | 15 | 2.137 | 0 |
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
| tensordot_f32_n1024 | jax-cuda | 16 | 15 | 3.208 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda | 16 | 15 | 3.202 | 0 |
| tensordot_c32_n1024 | jax-cuda | 16 | 15 | 3.470 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda | 16 | 15 | 3.536 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda | 16 | 15 | 2.014 | 0 |
| vdot_read_f64_n1024 | jax-cuda | 16 | 15 | 2.416 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda | 16 | 15 | 2.934 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda | 16 | 15 | 2.717 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda | 16 | 15 | 2.119 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda | 32 | 15 | 3.467 | 0 |
| tensordot_f32_n262144 | jax-cuda | 8 | 15 | 2.095 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda | 2 | 15 | 3.322 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda | 8 | 15 | 2.455 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda | 1 | 15 | 5.497 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda | 16 | 15 | 2.644 | 0 |
| vdot_read_f64_n262144 | jax-cuda | 16 | 15 | 2.230 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda | 16 | 15 | 2.560 | 0 |
| vdot_read_c64_n262144 | jax-cuda | 16 | 15 | 2.584 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda | 16 | 15 | 2.101 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda | 32 | 15 | 2.924 | 1.664e-07 |
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
| tensordot_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.456 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.366 | 0 |
| tensordot_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.747 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.880 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.799 | 0 |
| vdot_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.349 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.358 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.496 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.255 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.125 | 0 |
| tensordot_f32_n262144 | jax-cuda-jit | 32 | 15 | 3.716 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda-jit | 2 | 15 | 3.159 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.591 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda-jit | 1 | 15 | 5.331 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.499 | 0 |
| vdot_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.522 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.526 | 0 |
| vdot_read_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.227 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 1.474 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.430 | 1.664e-07 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: benchmarks/gpu/reuse_contraction.yaml
timestamp: '2026-10-13T13:42:54Z'
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
    ./scripts/run_gpu_public_api.sh reuse_contraction
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 68
  selected_cases: 68
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
    data/results/nvidia-gpu/gpu/reuse_contraction/20261013_134253/fixtures/manifest.json: f1578dff7131ed12abdc750eb195b36e62b77f61c1ad6b172883c4161da9d5d8
    target/release/benchmark_gpu_public_api: b505a2bdb3d0c5a7bffd1390af659a154753ea998ec68becd4779e137c9c7150
  reference_numpy_version: 2.4.2
```
