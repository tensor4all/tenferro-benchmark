# gpu/reuse_contraction: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/reuse_contraction/20261013_101849/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | 4.671 (0.155) | unsupported | 7.198 (0.413) | unsupported | unsupported |
| add_into_f64_n1024 | 4.630 (0.227) | unsupported | 7.061 (0.140) | unsupported | unsupported |
| add_read_into_f32_n1024 | 4.648 (0.353) | unsupported | 6.864 (0.135) | unsupported | unsupported |
| add_read_into_f64_n1024 | 4.844 (0.261) | unsupported | 7.033 (0.661) | unsupported | unsupported |
| sub_into_f32_n1024 | 4.995 (0.449) | unsupported | 8.272 (0.676) | unsupported | unsupported |
| sub_into_f64_n1024 | 4.781 (0.316) | unsupported | 7.109 (0.057) | unsupported | unsupported |
| sub_read_into_f32_n1024 | 4.894 (0.635) | unsupported | 7.159 (0.565) | unsupported | unsupported |
| sub_read_into_f64_n1024 | 4.921 (0.315) | unsupported | 7.789 (0.476) | unsupported | unsupported |
| mul_into_f32_n1024 | 4.831 (0.331) | unsupported | 7.050 (0.117) | unsupported | unsupported |
| mul_into_f64_n1024 | 4.735 (0.154) | unsupported | 7.088 (0.333) | unsupported | unsupported |
| mul_read_into_f32_n1024 | 4.987 (0.325) | unsupported | 7.305 (0.404) | unsupported | unsupported |
| mul_read_into_f64_n1024 | 4.941 (0.338) | unsupported | 7.294 (0.142) | unsupported | unsupported |
| div_into_f32_n1024 | 37.068 (2.571) | unsupported | 7.336 (0.453) | unsupported | unsupported |
| div_into_f64_n1024 | 39.950 (2.426) | unsupported | 6.969 (0.253) | unsupported | unsupported |
| div_read_into_f32_n1024 | 38.963 (3.702) | unsupported | 6.938 (0.112) | unsupported | unsupported |
| div_read_into_f64_n1024 | 40.053 (2.346) | unsupported | 6.959 (0.026) | unsupported | unsupported |
| neg_into_f32_n1024 | 4.696 (0.445) | unsupported | 6.863 (0.476) | unsupported | unsupported |
| neg_into_f64_n1024 | 4.803 (0.424) | unsupported | 6.885 (0.128) | unsupported | unsupported |
| neg_read_into_f32_n1024 | 4.287 (0.118) | unsupported | 6.733 (0.093) | unsupported | unsupported |
| neg_read_into_f64_n1024 | 4.344 (0.041) | unsupported | 6.762 (0.070) | unsupported | unsupported |
| conj_into_c32_n1024 | 34.290 (0.909) | unsupported | 7.368 (0.065) | unsupported | unsupported |
| conj_into_c64_n1024 | 142.952 (5.071) | unsupported | 7.337 (0.153) | unsupported | unsupported |
| conj_read_into_c32_n1024 | 34.549 (1.067) | unsupported | 7.306 (0.117) | unsupported | unsupported |
| conj_read_into_c64_n1024 | 128.259 (6.522) | unsupported | 7.204 (0.098) | unsupported | unsupported |
| add_into_f32_n1048576 | 42.598 (0.163) | unsupported | 39.748 (0.037) | unsupported | unsupported |
| add_into_f64_n1048576 | 84.154 (0.438) | unsupported | 77.739 (0.081) | unsupported | unsupported |
| add_read_into_f32_n1048576 | 43.636 (0.257) | unsupported | 39.761 (0.029) | unsupported | unsupported |
| add_read_into_f64_n1048576 | 85.096 (0.521) | unsupported | 77.686 (0.058) | unsupported | unsupported |
| sub_into_f32_n1048576 | 43.561 (0.414) | unsupported | 39.700 (0.100) | unsupported | unsupported |
| sub_into_f64_n1048576 | 85.428 (0.573) | unsupported | 77.591 (0.115) | unsupported | unsupported |
| sub_read_into_f32_n1048576 | 43.555 (0.153) | unsupported | 39.654 (0.053) | unsupported | unsupported |
| sub_read_into_f64_n1048576 | 85.918 (0.754) | unsupported | 77.549 (0.047) | unsupported | unsupported |
| mul_into_f32_n1048576 | 43.672 (0.158) | unsupported | 39.723 (0.098) | unsupported | unsupported |
| mul_into_f64_n1048576 | 84.752 (0.500) | unsupported | 77.541 (0.089) | unsupported | unsupported |
| mul_read_into_f32_n1048576 | 43.593 (0.213) | unsupported | 39.641 (0.043) | unsupported | unsupported |
| mul_read_into_f64_n1048576 | 86.218 (0.516) | unsupported | 77.561 (0.158) | unsupported | unsupported |
| div_into_f32_n1048576 | 70.103 (0.215) | unsupported | 39.894 (0.037) | unsupported | unsupported |
| div_into_f64_n1048576 | 152.901 (0.109) | unsupported | 94.176 (0.093) | unsupported | unsupported |
| div_read_into_f32_n1048576 | 70.027 (0.080) | unsupported | 39.871 (0.073) | unsupported | unsupported |
| div_read_into_f64_n1048576 | 152.889 (0.107) | unsupported | 94.160 (0.128) | unsupported | unsupported |
| neg_into_f32_n1048576 | 29.644 (0.036) | unsupported | 27.204 (0.039) | unsupported | unsupported |
| neg_into_f64_n1048576 | 60.749 (1.133) | unsupported | 52.831 (0.043) | unsupported | unsupported |
| neg_read_into_f32_n1048576 | 30.722 (0.172) | unsupported | 27.221 (0.049) | unsupported | unsupported |
| neg_read_into_f64_n1048576 | 61.523 (0.669) | unsupported | 52.822 (0.030) | unsupported | unsupported |
| conj_into_c32_n1048576 | 108.388 (0.123) | unsupported | 52.809 (0.074) | unsupported | unsupported |
| conj_into_c64_n1048576 | 209.917 (0.170) | unsupported | 103.685 (0.118) | unsupported | unsupported |
| conj_read_into_c32_n1048576 | 108.300 (0.176) | unsupported | 52.844 (0.268) | unsupported | unsupported |
| conj_read_into_c64_n1048576 | 210.019 (0.219) | unsupported | 103.702 (0.143) | unsupported | unsupported |
| tensordot_f32_n1024 | 25.600 (0.615) | 35.955 (0.792) | 51.047 (0.511) | 184.385 (3.822) | 82.087 (2.826) |
| tensordot_f64_n1024 | 29.159 (0.718) | 38.685 (1.786) | 55.223 (2.904) | 158.035 (5.702) | 71.369 (2.166) |
| tensordot_c32_n1024 | 25.273 (0.544) | 34.511 (0.639) | 53.714 (1.861) | 163.916 (3.315) | 85.132 (1.356) |
| tensordot_c64_n1024 | 165.059 (1.606) | 38.338 (1.001) | 55.607 (1.785) | 174.595 (7.845) | 91.337 (0.491) |
| vdot_read_f32_n1024 | 152.436 (5.508) | unsupported | 15.856 (0.323) | 90.935 (1.330) | 73.023 (1.251) |
| vdot_read_f64_n1024 | 162.546 (2.982) | unsupported | 15.890 (0.278) | 96.064 (5.640) | 134.257 (3.961) |
| vdot_read_c32_n1024 | 162.600 (1.174) | unsupported | 15.800 (0.283) | 143.395 (2.974) | 158.576 (23.565) |
| vdot_read_c64_n1024 | 161.838 (0.854) | unsupported | 15.350 (0.186) | 161.575 (15.809) | 175.011 (20.980) |
| norm_squared_read_f32_n1024 | 163.196 (1.537) | unsupported | 15.985 (0.608) | 88.320 (1.101) | 175.584 (11.733) |
| norm_squared_read_f64_n1024 | 159.845 (3.109) | unsupported | 15.736 (0.131) | 83.307 (2.692) | 139.191 (13.563) |
| tensordot_f32_n262144 | 68.370 (1.836) | 45.684 (0.779) | 63.059 (1.977) | 158.824 (4.109) | 88.980 (0.575) |
| tensordot_f64_n262144 | 1667.675 (1.740) | 1631.289 (1.339) | 1632.497 (2.438) | 1597.702 (48.622) | 1613.912 (22.751) |
| tensordot_c32_n262144 | 142.523 (0.199) | 142.327 (0.290) | 161.046 (2.230) | 292.660 (24.512) | 158.652 (1.146) |
| tensordot_c64_n262144 | 5067.433 (2.922) | 5060.457 (3.539) | 5275.542 (5.430) | 5620.855 (20.824) | 5421.368 (24.408) |
| vdot_read_f32_n262144 | 289.986 (30.684) | unsupported | 15.757 (0.359) | 162.092 (23.442) | 160.866 (8.618) |
| vdot_read_f64_n262144 | 312.814 (41.482) | unsupported | 20.245 (0.016) | 157.604 (2.519) | 162.829 (12.234) |
| vdot_read_c32_n262144 | 260.523 (14.473) | unsupported | 17.367 (0.033) | 158.611 (18.490) | 171.748 (19.247) |
| vdot_read_c64_n262144 | 203.681 (16.716) | unsupported | 33.783 (0.031) | 167.398 (6.530) | 172.563 (20.009) |
| norm_squared_read_f32_n262144 | 184.180 (9.132) | unsupported | 15.822 (0.549) | 143.158 (29.474) | 167.909 (6.099) |
| norm_squared_read_f64_n262144 | 169.900 (2.512) | unsupported | 16.661 (0.816) | 120.494 (17.079) | 155.445 (4.345) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| add_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.392 | 0 |
| add_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.371 | 0 |
| add_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.380 | 0 |
| add_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.480 | 0 |
| sub_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.557 | 0 |
| sub_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.448 | 0 |
| sub_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.506 | 0 |
| sub_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.519 | 0 |
| mul_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.473 | 0 |
| mul_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.424 | 0 |
| mul_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.553 | 0 |
| mul_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.530 | 0 |
| div_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.372 | 0 |
| div_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.557 | 0 |
| div_read_into_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.494 | 0 |
| div_read_into_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.563 | 0 |
| neg_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.404 | 0 |
| neg_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.459 | 0 |
| neg_read_into_f32_n1024 | tenferro-cuda-direct | 512 | 15 | 2.195 | 0 |
| neg_read_into_f64_n1024 | tenferro-cuda-direct | 512 | 15 | 2.224 | 0 |
| conj_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.195 | 0 |
| conj_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.287 | 0 |
| conj_read_into_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.211 | 0 |
| conj_read_into_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.052 | 0 |
| add_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.726 | 0 |
| add_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.693 | 0 |
| add_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.793 | 0 |
| add_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.723 | 0 |
| sub_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.788 | 0 |
| sub_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.734 | 0 |
| sub_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.788 | 0 |
| sub_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.749 | 0 |
| mul_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.795 | 0 |
| mul_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.712 | 0 |
| mul_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.790 | 0 |
| mul_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.759 | 0 |
| div_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.243 | 0 |
| div_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.446 | 0 |
| div_read_into_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.241 | 0 |
| div_read_into_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.446 | 0 |
| neg_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.897 | 0 |
| neg_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.944 | 0 |
| neg_read_into_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.966 | 0 |
| neg_read_into_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.969 | 0 |
| conj_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.468 | 0 |
| conj_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.359 | 0 |
| conj_read_into_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.466 | 0 |
| conj_read_into_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.360 | 0 |
| tensordot_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.277 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.732 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.235 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.641 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.439 | 0 |
| vdot_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.601 | 1.722e-07 |
| vdot_read_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.602 | 0 |
| vdot_read_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.589 | 3.455e-07 |
| norm_squared_read_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.611 | 0 |
| norm_squared_read_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.558 | 0 |
| tensordot_f32_n262144 | tenferro-cuda-direct | 32 | 15 | 2.188 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-direct | 2 | 15 | 3.335 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.280 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-direct | 1 | 15 | 5.067 | 1.446e-05 |
| vdot_read_f32_n262144 | tenferro-cuda-direct | 8 | 15 | 2.320 | 0 |
| vdot_read_f64_n262144 | tenferro-cuda-direct | 8 | 15 | 2.503 | 2.995e-06 |
| vdot_read_c32_n262144 | tenferro-cuda-direct | 8 | 15 | 2.084 | 0 |
| vdot_read_c64_n262144 | tenferro-cuda-direct | 16 | 15 | 3.259 | 8.762e-06 |
| norm_squared_read_f32_n262144 | tenferro-cuda-direct | 16 | 15 | 2.947 | 0.0001786 |
| norm_squared_read_f64_n262144 | tenferro-cuda-direct | 16 | 15 | 2.718 | 0 |
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
| tensordot_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.301 | 0.0008859 |
| tensordot_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.476 | 0 |
| tensordot_c32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.209 | 0.0008152 |
| tensordot_c64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.454 | 0 |
| vdot_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f32_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f64_n1024 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| tensordot_f32_n262144 | tenferro-cuda-eager | 64 | 15 | 2.924 | 0.006191 |
| tensordot_f64_n262144 | tenferro-cuda-eager | 2 | 15 | 3.263 | 6.714e-06 |
| tensordot_c32_n262144 | tenferro-cuda-eager | 16 | 15 | 2.277 | 0.01583 |
| tensordot_c64_n262144 | tenferro-cuda-eager | 1 | 15 | 5.060 | 1.446e-05 |
| vdot_read_f32_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_f64_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c32_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| vdot_read_c64_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f32_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| norm_squared_read_f64_n262144 | tenferro-cuda-eager | — | — | — | No matching immediate EagerSession reduction method; the direct BackendSession API is measured separately. |
| add_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.686 | 0 |
| add_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.615 | 0 |
| add_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.514 | 0 |
| add_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.601 | 0 |
| sub_into_f32_n1024 | pytorch-cuda | 256 | 15 | 2.118 | 0 |
| sub_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.640 | 0 |
| sub_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.665 | 0 |
| sub_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.988 | 0 |
| mul_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.610 | 0 |
| mul_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.629 | 0 |
| mul_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.740 | 0 |
| mul_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.735 | 0 |
| div_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.756 | 0 |
| div_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.568 | 0 |
| div_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.552 | 0 |
| div_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.563 | 0 |
| neg_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.514 | 0 |
| neg_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.525 | 0 |
| neg_read_into_f32_n1024 | pytorch-cuda | 512 | 15 | 3.447 | 0 |
| neg_read_into_f64_n1024 | pytorch-cuda | 512 | 15 | 3.462 | 0 |
| conj_into_c32_n1024 | pytorch-cuda | 256 | 15 | 1.886 | 0 |
| conj_into_c64_n1024 | pytorch-cuda | 512 | 15 | 3.757 | 0 |
| conj_read_into_c32_n1024 | pytorch-cuda | 512 | 15 | 3.741 | 0 |
| conj_read_into_c64_n1024 | pytorch-cuda | 512 | 15 | 3.689 | 0 |
| add_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.544 | 0 |
| add_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.488 | 0 |
| add_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.545 | 0 |
| add_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.486 | 0 |
| sub_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.541 | 0 |
| sub_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.483 | 0 |
| sub_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.538 | 0 |
| sub_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.482 | 0 |
| mul_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.542 | 0 |
| mul_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.481 | 0 |
| mul_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.537 | 0 |
| mul_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.482 | 0 |
| div_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.553 | 0 |
| div_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.014 | 0 |
| div_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.552 | 0 |
| div_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 3.013 | 0 |
| neg_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.741 | 0 |
| neg_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.691 | 0 |
| neg_read_into_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.742 | 0 |
| neg_read_into_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.690 | 0 |
| conj_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.690 | 0 |
| conj_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.659 | 0 |
| conj_read_into_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.691 | 0 |
| conj_read_into_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.659 | 0 |
| tensordot_f32_n1024 | pytorch-cuda | 64 | 15 | 3.267 | 0.0008859 |
| tensordot_f64_n1024 | pytorch-cuda | 64 | 15 | 3.534 | 0 |
| tensordot_c32_n1024 | pytorch-cuda | 64 | 15 | 3.438 | 0.0005344 |
| tensordot_c64_n1024 | pytorch-cuda | 64 | 15 | 3.559 | 0 |
| vdot_read_f32_n1024 | pytorch-cuda | 128 | 15 | 2.030 | 0 |
| vdot_read_f64_n1024 | pytorch-cuda | 128 | 15 | 2.034 | 1.722e-07 |
| vdot_read_c32_n1024 | pytorch-cuda | 128 | 15 | 2.022 | 0 |
| vdot_read_c64_n1024 | pytorch-cuda | 128 | 15 | 1.965 | 3.455e-07 |
| norm_squared_read_f32_n1024 | pytorch-cuda | 128 | 15 | 2.046 | 0 |
| norm_squared_read_f64_n1024 | pytorch-cuda | 128 | 15 | 2.014 | 0 |
| tensordot_f32_n262144 | pytorch-cuda | 32 | 15 | 2.018 | 0.006191 |
| tensordot_f64_n262144 | pytorch-cuda | 2 | 15 | 3.265 | 6.714e-06 |
| tensordot_c32_n262144 | pytorch-cuda | 16 | 15 | 2.577 | 0.005717 |
| tensordot_c64_n262144 | pytorch-cuda | 1 | 15 | 5.276 | 4.401e-06 |
| vdot_read_f32_n262144 | pytorch-cuda | 128 | 15 | 2.017 | 0 |
| vdot_read_f64_n262144 | pytorch-cuda | 128 | 15 | 2.591 | 2.995e-06 |
| vdot_read_c32_n262144 | pytorch-cuda | 128 | 15 | 2.223 | 0 |
| vdot_read_c64_n262144 | pytorch-cuda | 64 | 15 | 2.162 | 8.762e-06 |
| norm_squared_read_f32_n262144 | pytorch-cuda | 128 | 15 | 2.025 | 0.0001786 |
| norm_squared_read_f64_n262144 | pytorch-cuda | 128 | 15 | 2.133 | 0 |
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
| tensordot_f32_n1024 | jax-cuda | 16 | 15 | 2.950 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda | 16 | 15 | 2.529 | 0 |
| tensordot_c32_n1024 | jax-cuda | 16 | 15 | 2.623 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda | 16 | 15 | 2.794 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda | 32 | 15 | 2.910 | 0 |
| vdot_read_f64_n1024 | jax-cuda | 32 | 15 | 3.074 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda | 16 | 15 | 2.294 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda | 16 | 15 | 2.585 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda | 16 | 15 | 1.413 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda | 32 | 15 | 2.666 | 0 |
| tensordot_f32_n262144 | jax-cuda | 16 | 15 | 2.541 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda | 2 | 15 | 3.195 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda | 8 | 15 | 2.341 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda | 1 | 15 | 5.621 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda | 16 | 15 | 2.593 | 0 |
| vdot_read_f64_n262144 | jax-cuda | 16 | 15 | 2.522 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda | 16 | 15 | 2.538 | 0 |
| vdot_read_c64_n262144 | jax-cuda | 16 | 15 | 2.678 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda | 16 | 15 | 2.291 | 0.0001786 |
| norm_squared_read_f64_n262144 | jax-cuda | 16 | 15 | 1.928 | 1.664e-07 |
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
| tensordot_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.627 | 0.0008859 |
| tensordot_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.284 | 0 |
| tensordot_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.724 | 0.0007612 |
| tensordot_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.923 | 8.245e-07 |
| vdot_read_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.337 | 0 |
| vdot_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.148 | 1.722e-07 |
| vdot_read_c32_n1024 | jax-cuda-jit | 16 | 15 | 2.537 | 4.601e-05 |
| vdot_read_c64_n1024 | jax-cuda-jit | 16 | 15 | 2.800 | 3.435e-07 |
| norm_squared_read_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.809 | 0.0001766 |
| norm_squared_read_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.227 | 0 |
| tensordot_f32_n262144 | jax-cuda-jit | 32 | 15 | 2.847 | 0.009253 |
| tensordot_f64_n262144 | jax-cuda-jit | 2 | 15 | 3.228 | 6.714e-06 |
| tensordot_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.538 | 0.01992 |
| tensordot_c64_n262144 | jax-cuda-jit | 1 | 15 | 5.421 | 1.119e-05 |
| vdot_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.574 | 0 |
| vdot_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.605 | 2.995e-06 |
| vdot_read_c32_n262144 | jax-cuda-jit | 16 | 15 | 2.748 | 0 |
| vdot_read_c64_n262144 | jax-cuda-jit | 16 | 15 | 2.761 | 8.766e-06 |
| norm_squared_read_f32_n262144 | jax-cuda-jit | 16 | 15 | 2.687 | 0 |
| norm_squared_read_f64_n262144 | jax-cuda-jit | 16 | 15 | 2.487 | 1.664e-07 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/reuse_contraction
suite_file: benchmarks/gpu/reuse_contraction.yaml
timestamp: '2026-10-13T10:18:49Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 8b13220c1e0397a29a27be65557c24105352a35b
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: ed0ed4c79805014f2bba4644383450a6b9bf9361
  dirty: false
environment:
  hostname: 33ed5e46905b
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
    .devcontainer/cuda/devcontainer.json: 6d3b10caecad15c0a513120ce6282ebb13c65be81c40e29f78a329439c50f083
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: f29f1ad0dd99e8e5d24fffc68b6fe30673b0c42d34b5fe4a5557b7f774c2af15
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 86f2856411d902d27dcc5c8e66415eb933e057e04b9fad86be55569004572a36
    src/bin/benchmark_gpu_public_api/structural.rs: b5d780e867a5748be4f29f245d367833a8cd5ee0ebd440926d8b3003b432b0e4
    src/bin/benchmark_gpu_public_api/linalg.rs: 7d4e29355a973f3c60c2af53fd6ceac597d85fe21f9ab5f6bf164d0afe0a6566
    src/bin/benchmark_gpu_public_api/reuse.rs: ad72b565373904275128d7beb8673bada08d922e8f310be6cedb0d2d8187898f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 24cd5cec53eaecc8b6b4a22a1cd69e8cbe430ab6c9945d48df4c03d3b2655999
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 0a9e8d89405fc802e392e047d8b7c587e38ab1850d831735dd53fe13e87dabd3
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/reuse_contraction/20261013_101849/fixtures/manifest.json: f1578dff7131ed12abdc750eb195b36e62b77f61c1ad6b172883c4161da9d5d8
    target/release/benchmark_gpu_public_api: 91380afb8faffb97ca96d131ef98cc8e2656aababe594e82d65a07c9bb1bed38
  reference_numpy_version: 2.4.2
```
