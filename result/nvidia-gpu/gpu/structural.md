# gpu/structural: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/structural/20261013_151831/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| pad_default_f32_n1024 | 11.271 (0.898) | 77.837 (11.791) | 28.552 (0.428) | 404.107 (7.537) | 83.509 (1.983) |
| pad_default_f64_n1024 | 13.384 (0.699) | 122.875 (8.960) | 28.566 (0.213) | 405.905 (18.553) | 85.013 (9.067) |
| slice_default_f32_n1024 | 14.989 (0.997) | unsupported | 23.337 (0.361) | 183.635 (40.786) | 105.615 (4.624) |
| slice_default_f64_n1024 | 16.291 (1.183) | unsupported | 23.868 (0.323) | 165.364 (27.816) | 73.350 (2.739) |
| reverse_default_f32_n1024 | 16.242 (1.379) | 163.844 (8.307) | 13.802 (0.658) | 123.851 (13.928) | 75.830 (5.744) |
| reverse_default_f64_n1024 | 17.459 (0.958) | 134.243 (11.586) | 13.181 (0.134) | 107.286 (11.703) | 77.466 (5.746) |
| concatenate_default_f32_n1024 | 27.570 (4.816) | 266.409 (10.964) | 11.689 (0.549) | 257.455 (14.638) | 82.048 (6.148) |
| concatenate_default_f64_n1024 | 27.537 (2.220) | 292.145 (15.398) | 11.262 (0.236) | 292.566 (12.688) | 82.717 (7.696) |
| extract_diag_default_f32_n1024 | 15.883 (0.934) | 136.255 (3.426) | 15.743 (0.982) | 127.227 (23.447) | 78.445 (9.679) |
| extract_diag_default_f64_n1024 | 17.061 (0.312) | 130.701 (3.446) | 15.226 (0.118) | 128.092 (16.615) | 77.568 (9.431) |
| embed_diag_default_f32_n32 | 26.554 (0.649) | 161.642 (2.545) | 22.267 (0.418) | 133.761 (12.433) | 81.678 (6.119) |
| embed_diag_default_f64_n32 | 25.736 (1.175) | 169.067 (5.206) | 22.880 (0.511) | 138.387 (15.038) | 91.929 (2.829) |
| tril_default_f32_n1024 | 15.847 (1.099) | 153.692 (3.338) | 8.313 (0.270) | 100.330 (15.140) | 97.518 (6.394) |
| tril_default_f64_n1024 | 15.998 (1.005) | 161.146 (4.284) | 7.932 (0.073) | 95.772 (9.884) | 100.831 (10.749) |
| triu_default_f32_n1024 | 15.716 (0.860) | 174.658 (3.065) | 8.199 (0.101) | 98.953 (10.111) | 100.026 (10.390) |
| triu_default_f64_n1024 | 16.194 (1.056) | 131.652 (1.799) | 7.907 (0.033) | 96.402 (8.207) | 100.098 (10.748) |
| take_along_axis_broadcast_indices_f32_n1024 | 86.962 (3.646) | 123.578 (4.500) | 38.041 (0.568) | 99.786 (9.133) | 96.197 (11.185) |
| take_along_axis_batched_indices_f32_n1024 | unsupported | unsupported | 37.252 (0.888) | 103.327 (13.290) | 104.684 (12.341) |
| take_along_axis_broadcast_indices_f64_n1024 | 105.395 (7.575) | 135.442 (2.320) | 38.132 (0.381) | 100.722 (9.315) | 105.319 (13.787) |
| take_along_axis_batched_indices_f64_n1024 | unsupported | unsupported | 38.938 (2.910) | 104.836 (10.568) | 105.218 (12.741) |
| scatter_default_f32_n1024 | 40.436 (0.542) | 460.107 (24.829) | 28.127 (1.306) | 263.663 (11.757) | 129.865 (5.925) |
| scatter_default_f64_n1024 | 40.925 (0.523) | 497.947 (24.261) | 25.898 (1.207) | 265.344 (38.471) | 133.195 (9.693) |
| cast_default_f32_n1024 | 15.215 (0.183) | 203.668 (6.778) | 13.378 (0.394) | 190.095 (32.802) | 99.014 (7.138) |
| cast_default_f64_n1024 | 12.960 (2.024) | 148.424 (29.432) | 13.486 (0.132) | 180.964 (28.585) | 97.823 (5.188) |
| pad_default_f32_n1048576 | 70.820 (0.107) | 94.718 (0.111) | 42.554 (0.201) | 235.784 (8.657) | 98.583 (1.493) |
| pad_default_f64_n1048576 | 70.498 (0.119) | 120.627 (0.086) | 80.530 (0.150) | 222.480 (17.303) | 101.561 (2.685) |
| slice_default_f32_n1048576 | 21.658 (2.211) | unsupported | 24.349 (0.301) | 146.958 (4.273) | 92.865 (2.086) |
| slice_default_f64_n1048576 | 30.980 (0.227) | unsupported | 28.549 (0.102) | 146.300 (4.071) | 88.965 (1.483) |
| reverse_default_f32_n1048576 | 33.208 (0.083) | 59.882 (0.187) | 28.879 (0.034) | 106.726 (0.520) | 93.599 (2.776) |
| reverse_default_f64_n1048576 | 59.309 (0.130) | 105.851 (0.163) | 54.416 (0.051) | 106.681 (1.653) | 98.768 (1.202) |
| concatenate_default_f32_n1048576 | 67.848 (0.471) | 118.715 (0.518) | 54.146 (0.076) | 192.047 (1.250) | 101.323 (0.851) |
| concatenate_default_f64_n1048576 | 121.407 (0.778) | 215.775 (0.614) | 105.496 (0.176) | 153.040 (3.564) | 114.559 (1.631) |
| extract_diag_default_f32_n1048576 | 12.475 (0.225) | 56.402 (1.565) | 15.831 (0.110) | 77.779 (3.821) | 101.737 (11.983) |
| extract_diag_default_f64_n1048576 | 28.986 (2.039) | 56.896 (0.135) | 15.914 (0.627) | 81.267 (1.589) | 100.618 (10.765) |
| embed_diag_default_f32_n1024 | 21.009 (1.471) | 144.163 (5.355) | 23.121 (0.200) | 80.641 (2.505) | 96.410 (2.324) |
| embed_diag_default_f64_n1024 | 33.343 (0.141) | 148.595 (7.072) | 29.741 (0.187) | 83.258 (3.444) | 97.632 (1.381) |
| tril_default_f32_n1048576 | 29.639 (0.046) | 56.892 (1.527) | 22.700 (0.026) | 71.853 (1.352) | 97.772 (2.507) |
| tril_default_f64_n1048576 | 46.779 (0.109) | 95.640 (0.122) | 42.018 (0.058) | 64.965 (2.291) | 98.854 (1.270) |
| triu_default_f32_n1048576 | 29.834 (0.024) | 54.673 (0.259) | 22.539 (0.034) | 70.606 (1.274) | 99.453 (1.782) |
| triu_default_f64_n1048576 | 45.612 (0.324) | 95.632 (0.102) | 41.838 (0.036) | 65.622 (2.368) | 99.145 (0.632) |
| take_along_axis_broadcast_indices_f32_n1048576 | 87.199 (5.472) | 131.160 (2.351) | 50.378 (0.134) | 79.958 (6.182) | 118.545 (8.161) |
| take_along_axis_batched_indices_f32_n1048576 | unsupported | unsupported | 62.995 (0.302) | 75.707 (7.078) | 118.997 (15.486) |
| take_along_axis_broadcast_indices_f64_n1048576 | 403.625 (19.834) | 124.747 (10.122) | 69.818 (0.610) | 83.830 (7.256) | 125.826 (14.776) |
| take_along_axis_batched_indices_f64_n1048576 | unsupported | unsupported | 81.779 (0.229) | 85.360 (11.012) | 122.698 (10.927) |
| scatter_default_f32_n1048576 | 79.769 (0.727) | 148.929 (0.404) | 74.837 (0.104) | 158.764 (6.235) | 122.894 (3.219) |
| scatter_default_f64_n1048576 | 143.190 (1.413) | 244.379 (1.281) | 132.123 (0.128) | 166.078 (9.895) | 142.669 (2.244) |
| cast_default_f32_n1048576 | 44.110 (0.413) | 69.425 (0.270) | 41.277 (0.177) | 127.999 (1.026) | 94.722 (0.453) |
| cast_default_f64_n1048576 | 41.962 (0.075) | 94.687 (0.193) | 40.446 (0.027) | 128.239 (5.031) | 97.839 (2.352) |
| broadcast_in_dim_f32_n1024 | 11.122 (1.445) | unsupported | 16.924 (0.248) | 108.095 (3.492) | 94.378 (0.564) |
| broadcast_in_dim_f64_n1024 | 29.737 (0.128) | unsupported | 16.796 (0.137) | 104.518 (6.181) | 97.219 (6.763) |
| convert_f32_f64_n1024 | 27.005 (0.281) | 160.025 (13.667) | 13.466 (0.102) | 120.912 (1.554) | 97.811 (5.609) |
| broadcast_in_dim_f32_n1048576 | 30.814 (0.254) | unsupported | 18.183 (0.553) | 108.243 (2.435) | 97.110 (0.666) |
| broadcast_in_dim_f64_n1048576 | 34.039 (0.437) | unsupported | 28.052 (0.073) | 109.287 (1.615) | 95.901 (0.368) |
| convert_f32_f64_n1048576 | 44.945 (0.076) | 69.357 (0.104) | 41.269 (0.102) | 126.191 (2.443) | 95.912 (0.467) |
| dynamic_slice_in_bounds_f32_n1024 | 139.645 (3.175) | 282.130 (64.856) | unsupported | 150.663 (13.060) | 102.693 (12.600) |
| dynamic_slice_clamped_f32_n1024 | 130.424 (6.709) | 411.241 (78.096) | unsupported | 126.456 (6.063) | 102.833 (11.128) |
| gather_axis0_f32_n1024 | 29.105 (1.484) | 33.524 (0.492) | 11.882 (0.105) | 122.646 (2.480) | 100.343 (11.963) |
| gather_axis1_f32_n1024 | 31.780 (1.133) | 27.068 (3.536) | 11.863 (0.081) | 126.198 (2.557) | 101.496 (9.389) |
| dynamic_slice_in_bounds_f64_n1024 | 25.974 (1.052) | 483.422 (32.038) | unsupported | 153.512 (11.230) | 103.358 (12.308) |
| dynamic_slice_clamped_f64_n1024 | 21.761 (2.862) | 459.648 (18.034) | unsupported | 132.907 (6.256) | 102.325 (14.806) |
| gather_axis0_f64_n1024 | 27.410 (2.539) | 34.118 (0.513) | 11.978 (0.196) | 124.389 (0.710) | 100.154 (10.857) |
| gather_axis1_f64_n1024 | 21.780 (2.461) | 30.139 (1.444) | 11.840 (0.135) | 139.671 (11.175) | 101.412 (12.584) |
| dynamic_slice_in_bounds_f32_n1048576 | 17.380 (1.082) | 567.992 (255.601) | unsupported | 148.546 (9.851) | 103.681 (7.810) |
| dynamic_slice_clamped_f32_n1048576 | 16.572 (0.395) | 618.804 (45.886) | unsupported | 126.448 (4.902) | 102.472 (8.055) |
| gather_axis0_f32_n1048576 | 23.918 (0.040) | 41.866 (3.531) | 23.387 (0.858) | 124.223 (5.351) | 100.964 (4.329) |
| gather_axis1_f32_n1048576 | 20.626 (0.203) | 31.041 (1.769) | 23.326 (0.111) | 123.921 (3.200) | 101.087 (3.719) |
| dynamic_slice_in_bounds_f64_n1048576 | 17.649 (0.927) | 667.681 (40.175) | unsupported | 140.349 (5.798) | 103.947 (6.146) |
| dynamic_slice_clamped_f64_n1048576 | 17.225 (0.478) | 646.683 (34.298) | unsupported | 134.396 (6.332) | 103.417 (5.275) |
| gather_axis0_f64_n1048576 | 44.901 (0.378) | 54.593 (0.290) | 28.213 (0.107) | 125.589 (1.429) | 97.801 (3.827) |
| gather_axis1_f64_n1048576 | 33.254 (0.174) | 34.699 (0.220) | 41.770 (0.089) | 130.396 (8.749) | 99.921 (0.762) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| pad_default_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.885 | 0 |
| pad_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.713 | 0 |
| slice_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 1.919 | 0 |
| slice_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.085 | 0 |
| reverse_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.079 | 0 |
| reverse_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.235 | 0 |
| concatenate_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.529 | 0 |
| concatenate_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.762 | 0 |
| extract_diag_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.033 | 0 |
| extract_diag_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.092 | 0 |
| embed_diag_default_f32_n32 | tenferro-cuda-direct | 128 | 15 | 3.399 | 0 |
| embed_diag_default_f64_n32 | tenferro-cuda-direct | 128 | 15 | 3.294 | 0 |
| tril_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.028 | 0 |
| tril_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.048 | 0 |
| triu_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.012 | 0 |
| triu_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.073 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.783 | 0 |
| take_along_axis_batched_indices_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 3.373 | 0 |
| take_along_axis_batched_indices_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.588 | 0 |
| scatter_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.619 | 0 |
| cast_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 1.948 | 0 |
| cast_default_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 3.318 | 0 |
| pad_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.266 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-direct | 31 | 15 | 2.185 | 0 |
| slice_default_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.772 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.983 | 0 |
| reverse_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.125 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.898 | 0 |
| concatenate_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.171 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.943 | 0 |
| extract_diag_default_f32_n1048576 | tenferro-cuda-direct | 256 | 15 | 3.194 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 7.421 | 0 |
| embed_diag_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.345 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 1.067 | 0 |
| tril_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.897 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.497 | 0 |
| triu_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.909 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.460 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.790 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 3.229 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.553 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.291 | 0 |
| cast_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.412 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.686 | 0 |
| broadcast_in_dim_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.847 | 0 |
| broadcast_in_dim_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 7.613 | 0 |
| convert_f32_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 6.913 | 0 |
| broadcast_in_dim_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.972 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.089 | 0 |
| convert_f32_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.438 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.234 | 0 |
| dynamic_slice_clamped_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.087 | 0 |
| gather_axis0_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.725 | 0 |
| gather_axis1_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.034 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.325 | 0 |
| dynamic_slice_clamped_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.785 | 0 |
| gather_axis0_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.509 | 0 |
| gather_axis1_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.788 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.225 | 0 |
| dynamic_slice_clamped_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.121 | 0 |
| gather_axis0_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 3.061 | 0 |
| gather_axis1_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.640 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.259 | 0 |
| dynamic_slice_clamped_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.205 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.874 | 0 |
| gather_axis1_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.128 | 0 |
| pad_default_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.491 | 0 |
| pad_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 3.932 | 0 |
| slice_default_f32_n1024 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| slice_default_f64_n1024 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| reverse_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.622 | 0 |
| reverse_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 4.296 | 0 |
| concatenate_default_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.131 | 0 |
| concatenate_default_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.337 | 0 |
| extract_diag_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.180 | 0 |
| extract_diag_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.091 | 0 |
| embed_diag_default_f32_n32 | tenferro-cuda-eager | 16 | 15 | 2.586 | 0 |
| embed_diag_default_f64_n32 | tenferro-cuda-eager | 16 | 15 | 2.705 | 0 |
| tril_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.459 | 0 |
| tril_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.578 | 0 |
| triu_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.795 | 0 |
| triu_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.106 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 1.977 | 0 |
| take_along_axis_batched_indices_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.167 | 0 |
| take_along_axis_batched_indices_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.681 | 0 |
| scatter_default_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.984 | 0 |
| cast_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.259 | 0 |
| cast_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.375 | 0 |
| pad_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.031 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-eager | 31 | 15 | 3.739 | 0 |
| slice_default_f32_n1048576 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| slice_default_f64_n1048576 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| reverse_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.916 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.387 | 0 |
| concatenate_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.799 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.452 | 0 |
| extract_diag_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.610 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.641 | 0 |
| embed_diag_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.307 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.378 | 0 |
| tril_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.641 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.060 | 0 |
| triu_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.499 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.060 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.099 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.992 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.383 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 1.955 | 0 |
| cast_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.222 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.030 | 0 |
| broadcast_in_dim_f32_n1024 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| broadcast_in_dim_f64_n1024 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| convert_f32_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.560 | 0 |
| broadcast_in_dim_f32_n1048576 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| convert_f32_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.219 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.257 | 0 |
| dynamic_slice_clamped_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.290 | 0 |
| gather_axis0_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.146 | 0 |
| gather_axis1_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.465 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.867 | 0 |
| dynamic_slice_clamped_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.677 | 0 |
| gather_axis0_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.184 | 0 |
| gather_axis1_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 1.929 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 4.544 | 0 |
| dynamic_slice_clamped_f32_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.475 | 0 |
| gather_axis0_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.679 | 0 |
| gather_axis1_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.987 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.671 | 0 |
| dynamic_slice_clamped_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.587 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.747 | 0 |
| gather_axis1_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.221 | 0 |
| pad_default_f32_n1024 | pytorch-cuda | 128 | 15 | 3.655 | 0 |
| pad_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.656 | 0 |
| slice_default_f32_n1024 | pytorch-cuda | 128 | 15 | 2.987 | 0 |
| slice_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.055 | 0 |
| reverse_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.533 | 0 |
| reverse_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.374 | 0 |
| concatenate_default_f32_n1024 | pytorch-cuda | 256 | 15 | 2.992 | 0 |
| concatenate_default_f64_n1024 | pytorch-cuda | 256 | 15 | 2.883 | 0 |
| extract_diag_default_f32_n1024 | pytorch-cuda | 256 | 15 | 4.030 | 0 |
| extract_diag_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.898 | 0 |
| embed_diag_default_f32_n32 | pytorch-cuda | 128 | 15 | 2.850 | 0 |
| embed_diag_default_f64_n32 | pytorch-cuda | 128 | 15 | 2.929 | 0 |
| tril_default_f32_n1024 | pytorch-cuda | 256 | 15 | 2.128 | 0 |
| tril_default_f64_n1024 | pytorch-cuda | 256 | 15 | 2.031 | 0 |
| triu_default_f32_n1024 | pytorch-cuda | 256 | 15 | 2.099 | 0 |
| triu_default_f64_n1024 | pytorch-cuda | 256 | 15 | 2.024 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | pytorch-cuda | 64 | 15 | 2.435 | 0 |
| take_along_axis_batched_indices_f32_n1024 | pytorch-cuda | 64 | 15 | 2.384 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | pytorch-cuda | 64 | 15 | 2.440 | 0 |
| take_along_axis_batched_indices_f64_n1024 | pytorch-cuda | 64 | 15 | 2.492 | 0 |
| scatter_default_f32_n1024 | pytorch-cuda | 128 | 15 | 3.600 | 0 |
| scatter_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.315 | 0 |
| cast_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.425 | 0 |
| cast_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.452 | 0 |
| pad_default_f32_n1048576 | pytorch-cuda | 63 | 15 | 2.681 | 0 |
| pad_default_f64_n1048576 | pytorch-cuda | 31 | 15 | 2.496 | 0 |
| slice_default_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.117 | 0 |
| slice_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.827 | 0 |
| reverse_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.848 | 0 |
| reverse_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.741 | 0 |
| concatenate_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 1.733 | 0 |
| concatenate_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 1.688 | 0 |
| extract_diag_default_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.026 | 0 |
| extract_diag_default_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.037 | 0 |
| embed_diag_default_f32_n1024 | pytorch-cuda | 64 | 15 | 1.480 | 0 |
| embed_diag_default_f64_n1024 | pytorch-cuda | 32 | 15 | 0.952 | 0 |
| tril_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.453 | 0 |
| tril_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.345 | 0 |
| triu_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.442 | 0 |
| triu_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.339 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | pytorch-cuda | 64 | 15 | 3.224 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | pytorch-cuda | 32 | 15 | 2.016 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.234 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.617 | 0 |
| scatter_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 2.395 | 0 |
| scatter_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.114 | 0 |
| cast_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 1.321 | 0 |
| cast_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.589 | 0 |
| broadcast_in_dim_f32_n1024 | pytorch-cuda | 128 | 15 | 2.166 | 0 |
| broadcast_in_dim_f64_n1024 | pytorch-cuda | 128 | 15 | 2.150 | 0 |
| convert_f32_f64_n1024 | pytorch-cuda | 256 | 15 | 3.447 | 0 |
| broadcast_in_dim_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.164 | 0 |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.898 | 0 |
| convert_f32_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.321 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f32_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f32_n1024 | pytorch-cuda | 256 | 15 | 3.042 | 0 |
| gather_axis1_f32_n1024 | pytorch-cuda | 256 | 15 | 3.037 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f64_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f64_n1024 | pytorch-cuda | 256 | 15 | 3.066 | 0 |
| gather_axis1_f64_n1024 | pytorch-cuda | 256 | 15 | 3.031 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f32_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.994 | 0 |
| gather_axis1_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.986 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f64_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.806 | 0 |
| gather_axis1_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.673 | 0 |
| pad_default_f32_n1024 | jax-cuda | 8 | 15 | 3.233 | 0 |
| pad_default_f64_n1024 | jax-cuda | 8 | 15 | 3.247 | 0 |
| slice_default_f32_n1024 | jax-cuda | 16 | 15 | 2.938 | 0 |
| slice_default_f64_n1024 | jax-cuda | 16 | 15 | 2.646 | 0 |
| reverse_default_f32_n1024 | jax-cuda | 16 | 15 | 1.982 | 0 |
| reverse_default_f64_n1024 | jax-cuda | 32 | 15 | 3.433 | 0 |
| concatenate_default_f32_n1024 | jax-cuda | 8 | 15 | 2.060 | 0 |
| concatenate_default_f64_n1024 | jax-cuda | 8 | 15 | 2.341 | 0 |
| extract_diag_default_f32_n1024 | jax-cuda | 16 | 15 | 2.036 | 0 |
| extract_diag_default_f64_n1024 | jax-cuda | 16 | 15 | 2.049 | 0 |
| embed_diag_default_f32_n32 | jax-cuda | 16 | 15 | 2.140 | 0 |
| embed_diag_default_f64_n32 | jax-cuda | 16 | 15 | 2.214 | 0 |
| tril_default_f32_n1024 | jax-cuda | 32 | 15 | 3.211 | 0 |
| tril_default_f64_n1024 | jax-cuda | 32 | 15 | 3.065 | 0 |
| triu_default_f32_n1024 | jax-cuda | 32 | 15 | 3.167 | 0 |
| triu_default_f64_n1024 | jax-cuda | 32 | 15 | 3.085 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | jax-cuda | 32 | 15 | 3.193 | 0 |
| take_along_axis_batched_indices_f32_n1024 | jax-cuda | 32 | 15 | 3.306 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | jax-cuda | 32 | 15 | 3.223 | 0 |
| take_along_axis_batched_indices_f64_n1024 | jax-cuda | 32 | 15 | 3.355 | 0 |
| scatter_default_f32_n1024 | jax-cuda | 8 | 15 | 2.109 | 0 |
| scatter_default_f64_n1024 | jax-cuda | 8 | 15 | 2.123 | 0 |
| cast_default_f32_n1024 | jax-cuda | 16 | 15 | 3.042 | 0 |
| cast_default_f64_n1024 | jax-cuda | 16 | 15 | 2.895 | 0 |
| pad_default_f32_n1048576 | jax-cuda | 8 | 15 | 1.886 | 0 |
| pad_default_f64_n1048576 | jax-cuda | 8 | 15 | 1.780 | 0 |
| slice_default_f32_n1048576 | jax-cuda | 16 | 15 | 2.351 | 0 |
| slice_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.341 | 0 |
| reverse_default_f32_n1048576 | jax-cuda | 32 | 15 | 3.415 | 0 |
| reverse_default_f64_n1048576 | jax-cuda | 32 | 15 | 3.414 | 0 |
| concatenate_default_f32_n1048576 | jax-cuda | 16 | 15 | 3.073 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.449 | 0 |
| extract_diag_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.489 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.601 | 0 |
| embed_diag_default_f32_n1024 | jax-cuda | 32 | 15 | 2.581 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 2.664 | 0 |
| tril_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.299 | 0 |
| tril_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.079 | 0 |
| triu_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.259 | 0 |
| triu_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.100 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | jax-cuda | 32 | 15 | 2.559 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | jax-cuda | 32 | 15 | 2.423 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | 32 | 15 | 2.683 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | jax-cuda | 32 | 15 | 2.732 | 0 |
| scatter_default_f32_n1048576 | jax-cuda | 16 | 15 | 2.540 | 0 |
| scatter_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.657 | 0 |
| cast_default_f32_n1048576 | jax-cuda | 16 | 15 | 2.048 | 0 |
| cast_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.052 | 0 |
| broadcast_in_dim_f32_n1024 | jax-cuda | 32 | 15 | 3.459 | 0 |
| broadcast_in_dim_f64_n1024 | jax-cuda | 32 | 15 | 3.345 | 0 |
| convert_f32_f64_n1024 | jax-cuda | 32 | 15 | 3.869 | 0 |
| broadcast_in_dim_f32_n1048576 | jax-cuda | 32 | 15 | 3.464 | 0 |
| broadcast_in_dim_f64_n1048576 | jax-cuda | 32 | 15 | 3.497 | 0 |
| convert_f32_f64_n1048576 | jax-cuda | 16 | 15 | 2.019 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | jax-cuda | 16 | 15 | 2.411 | 0 |
| dynamic_slice_clamped_f32_n1024 | jax-cuda | 16 | 15 | 2.023 | 0 |
| gather_axis0_f32_n1024 | jax-cuda | 16 | 15 | 1.962 | 0 |
| gather_axis1_f32_n1024 | jax-cuda | 16 | 15 | 2.019 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | jax-cuda | 16 | 15 | 2.456 | 0 |
| dynamic_slice_clamped_f64_n1024 | jax-cuda | 16 | 15 | 2.127 | 0 |
| gather_axis0_f64_n1024 | jax-cuda | 16 | 15 | 1.990 | 0 |
| gather_axis1_f64_n1024 | jax-cuda | 16 | 15 | 2.235 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | jax-cuda | 16 | 15 | 2.377 | 0 |
| dynamic_slice_clamped_f32_n1048576 | jax-cuda | 16 | 15 | 2.023 | 0 |
| gather_axis0_f32_n1048576 | jax-cuda | 16 | 15 | 1.988 | 0 |
| gather_axis1_f32_n1048576 | jax-cuda | 16 | 15 | 1.983 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | 16 | 15 | 2.246 | 0 |
| dynamic_slice_clamped_f64_n1048576 | jax-cuda | 16 | 15 | 2.150 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda | 16 | 15 | 2.009 | 0 |
| gather_axis1_f64_n1048576 | jax-cuda | 16 | 15 | 2.086 | 0 |
| pad_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.672 | 0 |
| pad_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.720 | 0 |
| slice_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.380 | 0 |
| slice_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.347 | 0 |
| reverse_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.427 | 0 |
| reverse_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.479 | 0 |
| concatenate_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.626 | 0 |
| concatenate_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.647 | 0 |
| extract_diag_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.510 | 0 |
| extract_diag_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.482 | 0 |
| embed_diag_default_f32_n32 | jax-cuda-jit | 32 | 15 | 2.614 | 0 |
| embed_diag_default_f64_n32 | jax-cuda-jit | 32 | 15 | 2.942 | 0 |
| tril_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.121 | 0 |
| tril_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.227 | 0 |
| triu_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.201 | 0 |
| triu_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.203 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.078 | 0 |
| take_along_axis_batched_indices_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.350 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.370 | 0 |
| take_along_axis_batched_indices_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.367 | 0 |
| scatter_default_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.078 | 0 |
| scatter_default_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.131 | 0 |
| cast_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.168 | 0 |
| cast_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.130 | 0 |
| pad_default_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.577 | 0 |
| pad_default_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.625 | 0 |
| slice_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.972 | 0 |
| slice_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.847 | 0 |
| reverse_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.995 | 0 |
| reverse_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.161 | 0 |
| concatenate_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.242 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.833 | 0 |
| extract_diag_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.256 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.220 | 0 |
| embed_diag_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.085 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.124 | 0 |
| tril_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.129 | 0 |
| tril_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.163 | 0 |
| triu_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.182 | 0 |
| triu_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.173 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.897 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.904 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.013 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.963 | 0 |
| scatter_default_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.966 | 0 |
| scatter_default_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.283 | 0 |
| cast_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.031 | 0 |
| cast_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.131 | 0 |
| broadcast_in_dim_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.020 | 0 |
| broadcast_in_dim_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.111 | 0 |
| convert_f32_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.130 | 0 |
| broadcast_in_dim_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.108 | 0 |
| broadcast_in_dim_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.069 | 0 |
| convert_f32_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.069 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.286 | 0 |
| dynamic_slice_clamped_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.291 | 0 |
| gather_axis0_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.211 | 0 |
| gather_axis1_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.248 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.307 | 0 |
| dynamic_slice_clamped_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.274 | 0 |
| gather_axis0_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.205 | 0 |
| gather_axis1_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.245 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.318 | 0 |
| dynamic_slice_clamped_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.279 | 0 |
| gather_axis0_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.231 | 0 |
| gather_axis1_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.235 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.326 | 0 |
| dynamic_slice_clamped_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.309 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.130 | 0 |
| gather_axis1_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.197 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/structural
suite_file: benchmarks/gpu/structural.yaml
timestamp: '2026-10-13T15:18:31Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: bc70342b20e7ff9257c64cc046c8a90294f012e6
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
    ./scripts/run_gpu_public_api.sh structural
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 70
  selected_cases: 70
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: ef0cec562e6a48dc151cd120b3f916c38ac96663a1228e351018d99aa625d87d
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/structural.rs: 8121b12b5fed3b0acbc71e2218856173d53a5acc93092b93c7fd3309d4f45962
    src/bin/benchmark_gpu_public_api/linalg.rs: 84162923fa0d0772c88ec9c4eab169ad3a75e5d16ff76961d1541670dd8565c1
    src/bin/benchmark_gpu_public_api/reuse.rs: ad72b565373904275128d7beb8673bada08d922e8f310be6cedb0d2d8187898f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: fc27901bbb61c4c56c7a8417090f56f4312928eb502942cebeda7042505eb4c9
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: d9d56faa7d44345e6d3317bc837d84173ad0d7b0bc1e829b776fd41ec68fb1aa
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/structural/20261013_151831/fixtures/manifest.json: 3c9a3818f96b4af9dab18e1915f7ae6c409da58f47ac19eb8e5f4e2c8fd2a6ff
    target/release/benchmark_gpu_public_api: 1f7a23d62012e216687550eb87baaf8cd287c945991f2c1a30df70d669c11c62
  reference_numpy_version: 2.4.2
```
