# gpu/structural: CUDA public API benchmark

Measured cells: **378**. Unsupported cells: **32**, including **8** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/structural/20261013_194942/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| pad_default_f32_n1024 | 12.868 (1.207) | 72.169 (8.325) | 29.404 (2.218) | 111.081 (5.920) | 120.422 (19.247) |
| pad_default_f64_n1024 | 15.892 (1.003) | 108.727 (8.001) | 28.051 (0.186) | 125.653 (9.704) | 137.163 (21.511) |
| slice_default_f32_n1024 | 17.480 (0.279) | unsupported | 23.425 (1.236) | 121.342 (2.129) | 129.370 (16.987) |
| slice_default_f64_n1024 | 17.333 (0.189) | unsupported | 24.433 (0.858) | 107.446 (14.831) | 126.884 (14.601) |
| reverse_default_f32_n1024 | 16.763 (0.134) | 146.417 (3.051) | 12.647 (1.022) | 81.784 (9.138) | 122.698 (16.862) |
| reverse_default_f64_n1024 | 16.760 (0.276) | 120.860 (8.390) | 12.996 (0.511) | 79.827 (4.473) | 119.771 (24.345) |
| concatenate_default_f32_n1024 | 27.447 (3.672) | 259.732 (15.830) | 11.876 (0.113) | 155.475 (19.935) | 129.108 (9.422) |
| concatenate_default_f64_n1024 | 28.892 (2.238) | 265.516 (70.765) | 12.050 (1.091) | 149.339 (19.362) | 126.611 (14.128) |
| extract_diag_default_f32_n1024 | 16.308 (0.582) | 130.447 (5.795) | 16.166 (0.314) | 78.802 (6.316) | 92.918 (14.924) |
| extract_diag_default_f64_n1024 | 17.214 (0.326) | 129.460 (2.292) | 16.387 (0.224) | 81.183 (5.359) | 85.278 (12.368) |
| embed_diag_default_f32_n32 | 25.618 (0.427) | 161.052 (3.168) | 22.502 (0.096) | 82.892 (8.530) | 81.153 (4.897) |
| embed_diag_default_f64_n32 | 25.483 (1.973) | 175.219 (6.061) | 22.418 (0.256) | 83.910 (11.824) | 81.787 (7.486) |
| tril_default_f32_n1024 | 15.965 (0.891) | 157.551 (4.458) | 8.037 (0.216) | 78.074 (7.739) | 81.568 (6.961) |
| tril_default_f64_n1024 | 16.253 (0.210) | 162.894 (4.749) | 8.006 (0.076) | 77.748 (9.786) | 99.916 (2.404) |
| triu_default_f32_n1024 | 15.940 (0.542) | 133.634 (4.382) | 7.900 (0.072) | 80.886 (10.034) | 126.662 (12.905) |
| triu_default_f64_n1024 | 16.230 (0.208) | 135.289 (6.157) | 8.003 (0.144) | 72.939 (1.005) | 133.146 (16.703) |
| take_along_axis_broadcast_indices_f32_n1024 | 81.715 (3.423) | 127.112 (7.602) | 38.218 (1.221) | 77.757 (4.096) | 140.974 (15.628) |
| take_along_axis_batched_indices_f32_n1024 | unsupported | unsupported | 37.395 (0.951) | 80.303 (9.675) | 131.244 (19.193) |
| take_along_axis_broadcast_indices_f64_n1024 | 99.599 (16.069) | 139.626 (5.044) | 38.362 (0.595) | 82.173 (12.817) | 137.657 (11.669) |
| take_along_axis_batched_indices_f64_n1024 | unsupported | unsupported | 37.515 (1.356) | 83.847 (9.767) | 138.693 (21.096) |
| scatter_default_f32_n1024 | 39.347 (0.611) | 474.570 (18.060) | 24.643 (0.935) | 168.825 (22.556) | 152.476 (24.325) |
| scatter_default_f64_n1024 | 40.840 (0.229) | 498.009 (5.256) | 24.700 (0.328) | 175.020 (33.199) | 137.815 (21.550) |
| cast_default_f32_n1024 | 15.850 (0.055) | 199.527 (7.008) | 13.689 (0.181) | 135.554 (5.719) | 99.969 (7.836) |
| cast_default_f64_n1024 | 15.715 (0.101) | 146.214 (36.683) | 14.442 (0.722) | 139.105 (14.529) | 97.663 (4.493) |
| pad_default_f32_n1048576 | 70.585 (0.175) | 94.672 (0.083) | 42.559 (0.058) | 112.567 (11.011) | 99.455 (2.771) |
| pad_default_f64_n1048576 | 70.345 (0.174) | 122.402 (0.086) | 80.588 (0.090) | 109.426 (1.805) | 100.155 (1.398) |
| slice_default_f32_n1048576 | 17.584 (0.037) | unsupported | 23.838 (0.577) | 100.574 (3.065) | 95.070 (2.938) |
| slice_default_f64_n1048576 | 30.506 (0.040) | unsupported | 28.571 (0.143) | 102.725 (1.966) | 98.817 (0.329) |
| reverse_default_f32_n1048576 | 32.825 (0.074) | 66.931 (0.513) | 29.064 (0.172) | 75.546 (3.402) | 99.084 (0.949) |
| reverse_default_f64_n1048576 | 58.094 (0.091) | 105.969 (0.103) | 54.435 (0.058) | 73.422 (1.316) | 100.261 (1.073) |
| concatenate_default_f32_n1048576 | 67.217 (0.290) | 120.422 (2.094) | 54.223 (0.089) | 139.472 (2.387) | 103.291 (1.025) |
| concatenate_default_f64_n1048576 | 114.105 (0.435) | 216.076 (0.354) | 105.652 (0.318) | 144.111 (6.175) | 114.716 (2.177) |
| extract_diag_default_f32_n1048576 | 7.142 (0.273) | 64.556 (1.153) | 18.198 (0.830) | 75.717 (1.107) | 74.075 (1.908) |
| extract_diag_default_f64_n1048576 | 19.695 (1.688) | 65.391 (0.709) | 16.666 (0.104) | 81.234 (4.692) | 79.876 (5.001) |
| embed_diag_default_f32_n1024 | 17.749 (0.267) | 180.522 (13.508) | 23.366 (0.469) | 80.293 (4.360) | 75.882 (1.280) |
| embed_diag_default_f64_n1024 | 31.621 (0.153) | 186.446 (13.183) | 29.900 (0.282) | 76.168 (2.140) | 75.337 (0.868) |
| tril_default_f32_n1048576 | 28.426 (0.174) | 62.452 (0.381) | 22.721 (0.060) | 71.153 (2.701) | 96.320 (1.262) |
| tril_default_f64_n1048576 | 44.577 (0.542) | 95.770 (0.030) | 42.187 (0.153) | 64.379 (1.704) | 94.767 (4.055) |
| triu_default_f32_n1048576 | 28.713 (0.165) | 61.324 (0.446) | 22.548 (0.045) | 67.664 (3.571) | 91.070 (2.627) |
| triu_default_f64_n1048576 | 44.507 (0.456) | 96.030 (0.073) | 41.877 (0.208) | 61.838 (1.177) | 96.318 (3.821) |
| take_along_axis_broadcast_indices_f32_n1048576 | 43.237 (8.777) | 175.194 (4.340) | 50.389 (0.148) | 80.283 (5.685) | 105.119 (4.708) |
| take_along_axis_batched_indices_f32_n1048576 | unsupported | unsupported | 63.292 (0.114) | 70.505 (2.488) | 101.531 (10.358) |
| take_along_axis_broadcast_indices_f64_n1048576 | 212.841 (3.986) | 151.999 (11.396) | 69.812 (0.255) | 81.385 (7.602) | 121.968 (16.831) |
| take_along_axis_batched_indices_f64_n1048576 | unsupported | unsupported | 81.815 (0.128) | 88.133 (11.199) | 104.199 (2.846) |
| scatter_default_f32_n1048576 | 80.016 (0.228) | 149.747 (1.279) | 75.044 (0.209) | 161.645 (6.536) | 112.530 (2.053) |
| scatter_default_f64_n1048576 | 144.624 (2.485) | 242.651 (1.291) | 132.180 (0.091) | 152.828 (6.214) | 142.219 (1.634) |
| cast_default_f32_n1048576 | 44.297 (0.489) | 69.269 (0.268) | 41.258 (0.037) | 124.161 (4.650) | 86.507 (1.592) |
| cast_default_f64_n1048576 | 42.095 (0.131) | 94.410 (0.134) | 40.485 (0.043) | 126.671 (10.154) | 88.024 (2.904) |
| broadcast_in_dim_f32_n1024 | 10.995 (0.796) | unsupported | 16.575 (0.281) | 100.973 (4.481) | 93.019 (6.899) |
| broadcast_in_dim_f64_n1024 | 29.195 (0.460) | unsupported | 16.585 (0.172) | 98.477 (4.315) | 94.603 (9.033) |
| convert_f32_f64_n1024 | 27.059 (0.463) | 151.402 (2.316) | 14.562 (0.455) | 118.860 (3.845) | 97.947 (6.042) |
| broadcast_in_dim_f32_n1048576 | 29.648 (0.087) | unsupported | 17.327 (0.224) | 102.311 (1.403) | 92.867 (0.638) |
| broadcast_in_dim_f64_n1048576 | 32.590 (0.286) | unsupported | 28.192 (0.245) | 102.631 (1.189) | 91.570 (0.409) |
| convert_f32_f64_n1048576 | 44.999 (0.066) | 69.270 (0.114) | 41.306 (0.033) | 120.215 (2.666) | 92.175 (1.322) |
| dynamic_slice_in_bounds_f32_n1024 | 139.529 (2.245) | 484.923 (38.905) | unsupported | 146.503 (8.537) | 104.513 (16.423) |
| dynamic_slice_clamped_f32_n1024 | 134.981 (4.765) | 527.483 (27.214) | unsupported | 122.705 (10.030) | 97.933 (14.822) |
| gather_axis0_f32_n1024 | 27.352 (0.960) | 31.309 (0.459) | 11.851 (0.441) | 111.236 (1.770) | 99.447 (11.119) |
| gather_axis1_f32_n1024 | 23.193 (2.701) | 25.480 (3.085) | 11.719 (0.122) | 110.785 (7.078) | 97.084 (9.500) |
| dynamic_slice_in_bounds_f64_n1024 | 22.103 (0.846) | 539.348 (22.699) | unsupported | 133.234 (2.803) | 98.439 (8.768) |
| dynamic_slice_clamped_f64_n1024 | 19.627 (1.089) | 519.901 (29.705) | unsupported | 128.449 (2.281) | 97.478 (5.216) |
| gather_axis0_f64_n1024 | 26.449 (1.558) | 32.118 (0.178) | 11.518 (0.139) | 115.164 (9.894) | 97.289 (8.794) |
| gather_axis1_f64_n1024 | 21.439 (1.932) | 32.642 (1.803) | 11.672 (0.402) | 113.039 (9.210) | 97.898 (6.834) |
| dynamic_slice_in_bounds_f32_n1048576 | 18.219 (1.338) | 581.718 (255.792) | unsupported | 122.740 (2.436) | 96.217 (4.147) |
| dynamic_slice_clamped_f32_n1048576 | 16.586 (1.117) | 620.100 (44.143) | unsupported | 129.476 (3.869) | 96.437 (4.500) |
| gather_axis0_f32_n1048576 | 23.757 (0.192) | 40.961 (2.318) | 25.017 (1.370) | 112.075 (2.039) | 95.462 (3.280) |
| gather_axis1_f32_n1048576 | 20.207 (0.140) | 25.520 (1.998) | 24.104 (1.061) | 114.447 (5.033) | 93.989 (2.738) |
| dynamic_slice_in_bounds_f64_n1048576 | 17.330 (0.081) | 390.256 (41.272) | unsupported | 130.547 (4.753) | 99.253 (7.108) |
| dynamic_slice_clamped_f64_n1048576 | 16.581 (0.184) | 385.377 (81.692) | unsupported | 123.432 (0.997) | 98.423 (8.762) |
| gather_axis0_f64_n1048576 | 43.159 (0.238) | 46.277 (0.364) | 28.205 (0.196) | 110.848 (6.306) | 78.303 (4.958) |
| gather_axis1_f64_n1048576 | 32.351 (1.080) | 34.371 (0.234) | 41.762 (0.039) | 128.627 (9.257) | 72.669 (4.497) |
| pad_interior_f32_n1024 | 122.911 (4.803) | 356.426 (80.749) | unsupported | 111.750 (14.539) | 66.856 (2.338) |
| pad_crop_f32_n1024 | 143.082 (2.094) | 284.068 (7.590) | 31.943 (0.180) | 108.172 (18.266) | 75.293 (5.675) |
| pad_interior_crop_f32_n1024 | 138.217 (1.584) | 220.226 (6.199) | unsupported | 123.318 (2.053) | 76.007 (1.990) |
| pad_interior_f64_n1024 | 67.667 (0.507) | 301.201 (3.814) | unsupported | 148.217 (9.815) | 76.521 (1.002) |
| pad_crop_f64_n1024 | 150.917 (0.852) | 320.388 (8.311) | 30.988 (0.678) | 158.750 (22.353) | 70.112 (0.494) |
| pad_interior_crop_f64_n1024 | 69.153 (0.465) | 298.465 (2.469) | unsupported | 165.939 (22.378) | 82.184 (11.058) |
| pad_interior_f32_n1048576 | 325.119 (0.767) | 370.277 (0.856) | unsupported | 152.666 (1.505) | 106.880 (2.832) |
| pad_crop_f32_n1048576 | 63.073 (0.346) | 91.269 (0.058) | 42.113 (0.083) | 160.373 (27.002) | 65.437 (1.443) |
| pad_interior_crop_f32_n1048576 | 308.479 (0.257) | 352.476 (0.220) | unsupported | 154.801 (1.657) | 103.487 (3.763) |
| pad_interior_f64_n1048576 | 324.422 (1.600) | 395.073 (0.571) | unsupported | 220.477 (12.582) | 217.667 (12.893) |
| pad_crop_f64_n1048576 | 62.544 (0.367) | 117.082 (0.073) | 79.856 (0.108) | 105.841 (1.518) | 65.891 (2.723) |
| pad_interior_crop_f64_n1048576 | 327.285 (0.482) | 377.873 (0.287) | unsupported | 219.986 (10.837) | 211.508 (17.870) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| pad_default_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.294 | 0 |
| pad_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.034 | 0 |
| slice_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.237 | 0 |
| slice_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.219 | 0 |
| reverse_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.146 | 0 |
| reverse_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.145 | 0 |
| concatenate_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.513 | 0 |
| concatenate_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.849 | 0 |
| extract_diag_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.087 | 0 |
| extract_diag_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.102 | 0 |
| embed_diag_default_f32_n32 | tenferro-cuda-direct | 128 | 15 | 3.279 | 0 |
| embed_diag_default_f64_n32 | tenferro-cuda-direct | 128 | 15 | 3.262 | 0 |
| tril_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.044 | 0 |
| tril_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.080 | 0 |
| triu_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.040 | 0 |
| triu_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.077 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.615 | 0 |
| take_along_axis_batched_indices_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 3.187 | 0 |
| take_along_axis_batched_indices_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.518 | 0 |
| scatter_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.614 | 0 |
| cast_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.029 | 0 |
| cast_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.011 | 0 |
| pad_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.259 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-direct | 31 | 15 | 2.181 | 0 |
| slice_default_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.251 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.952 | 0 |
| reverse_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.101 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.859 | 0 |
| concatenate_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.151 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.826 | 0 |
| extract_diag_default_f32_n1048576 | tenferro-cuda-direct | 256 | 15 | 1.828 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 5.042 | 0 |
| embed_diag_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.136 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 1.012 | 0 |
| tril_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.819 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.426 | 0 |
| triu_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.838 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.424 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.767 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.405 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.561 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.314 | 0 |
| cast_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.417 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.694 | 0 |
| broadcast_in_dim_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 2.815 | 0 |
| broadcast_in_dim_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 7.474 | 0 |
| convert_f32_f64_n1024 | tenferro-cuda-direct | 256 | 15 | 6.927 | 0 |
| broadcast_in_dim_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.897 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.043 | 0 |
| convert_f32_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.440 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.232 | 0 |
| dynamic_slice_clamped_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.160 | 0 |
| gather_axis0_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.501 | 0 |
| gather_axis1_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.969 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.829 | 0 |
| dynamic_slice_clamped_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.512 | 0 |
| gather_axis0_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.385 | 0 |
| gather_axis1_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.744 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.332 | 0 |
| dynamic_slice_clamped_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.123 | 0 |
| gather_axis0_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 3.041 | 0 |
| gather_axis1_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.587 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.218 | 0 |
| dynamic_slice_clamped_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.122 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.762 | 0 |
| gather_axis1_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.070 | 0 |
| pad_interior_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 1.967 | 0 |
| pad_crop_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.289 | 0 |
| pad_interior_crop_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.211 | 0 |
| pad_interior_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.165 | 0 |
| pad_crop_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.415 | 0 |
| pad_interior_crop_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.213 | 0 |
| pad_interior_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.601 | 0 |
| pad_crop_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.018 | 0 |
| pad_interior_crop_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.468 | 0 |
| pad_interior_f64_n1048576 | tenferro-cuda-direct | 5 | 15 | 1.622 | 0 |
| pad_crop_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.001 | 0 |
| pad_interior_crop_f64_n1048576 | tenferro-cuda-direct | 5 | 15 | 1.636 | 0 |
| pad_default_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.309 | 0 |
| pad_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 3.479 | 0 |
| slice_default_f32_n1024 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| slice_default_f64_n1024 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| reverse_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.343 | 0 |
| reverse_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 3.868 | 0 |
| concatenate_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 4.156 | 0 |
| concatenate_default_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.124 | 0 |
| extract_diag_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.087 | 0 |
| extract_diag_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.071 | 0 |
| embed_diag_default_f32_n32 | tenferro-cuda-eager | 16 | 15 | 2.577 | 0 |
| embed_diag_default_f64_n32 | tenferro-cuda-eager | 16 | 15 | 2.804 | 0 |
| tril_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.521 | 0 |
| tril_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.606 | 0 |
| triu_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.138 | 0 |
| triu_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.165 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.034 | 0 |
| take_along_axis_batched_indices_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.234 | 0 |
| take_along_axis_batched_indices_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.797 | 0 |
| scatter_default_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 1.992 | 0 |
| cast_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.192 | 0 |
| cast_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.339 | 0 |
| pad_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.029 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.958 | 0 |
| slice_default_f32_n1048576 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| slice_default_f64_n1048576 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| reverse_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.142 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.391 | 0 |
| concatenate_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.854 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.457 | 0 |
| extract_diag_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.066 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.093 | 0 |
| embed_diag_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.888 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.983 | 0 |
| tril_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.998 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.065 | 0 |
| triu_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.925 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.073 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.803 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.432 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.396 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.882 | 0 |
| cast_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.217 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.021 | 0 |
| broadcast_in_dim_f32_n1024 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| broadcast_in_dim_f64_n1024 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| convert_f32_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.422 | 0 |
| broadcast_in_dim_f32_n1048576 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-eager | — | — | — | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| convert_f32_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.217 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.879 | 0 |
| dynamic_slice_clamped_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 4.220 | 0 |
| gather_axis0_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 2.004 | 0 |
| gather_axis1_f32_n1024 | tenferro-cuda-eager | 128 | 15 | 3.261 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.157 | 0 |
| dynamic_slice_clamped_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 2.080 | 0 |
| gather_axis0_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.056 | 0 |
| gather_axis1_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 2.089 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 4.654 | 0 |
| dynamic_slice_clamped_f32_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.480 | 0 |
| gather_axis0_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.622 | 0 |
| gather_axis1_f32_n1048576 | tenferro-cuda-eager | 128 | 15 | 3.266 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.122 | 0 |
| dynamic_slice_clamped_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.083 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.962 | 0 |
| gather_axis1_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.200 | 0 |
| pad_interior_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.851 | 0 |
| pad_crop_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.273 | 0 |
| pad_interior_crop_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 1.762 | 0 |
| pad_interior_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.410 | 0 |
| pad_crop_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.563 | 0 |
| pad_interior_crop_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.388 | 0 |
| pad_interior_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.962 | 0 |
| pad_crop_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.921 | 0 |
| pad_interior_crop_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.820 | 0 |
| pad_interior_f64_n1048576 | tenferro-cuda-eager | 5 | 15 | 1.975 | 0 |
| pad_crop_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.747 | 0 |
| pad_interior_crop_f64_n1048576 | tenferro-cuda-eager | 5 | 15 | 1.889 | 0 |
| pad_default_f32_n1024 | pytorch-cuda | 128 | 15 | 3.764 | 0 |
| pad_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.591 | 0 |
| slice_default_f32_n1024 | pytorch-cuda | 128 | 15 | 2.998 | 0 |
| slice_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.127 | 0 |
| reverse_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.238 | 0 |
| reverse_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.327 | 0 |
| concatenate_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.040 | 0 |
| concatenate_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.085 | 0 |
| extract_diag_default_f32_n1024 | pytorch-cuda | 128 | 15 | 2.069 | 0 |
| extract_diag_default_f64_n1024 | pytorch-cuda | 128 | 15 | 2.098 | 0 |
| embed_diag_default_f32_n32 | pytorch-cuda | 128 | 15 | 2.880 | 0 |
| embed_diag_default_f64_n32 | pytorch-cuda | 128 | 15 | 2.870 | 0 |
| tril_default_f32_n1024 | pytorch-cuda | 256 | 15 | 2.057 | 0 |
| tril_default_f64_n1024 | pytorch-cuda | 256 | 15 | 2.050 | 0 |
| triu_default_f32_n1024 | pytorch-cuda | 256 | 15 | 2.022 | 0 |
| triu_default_f64_n1024 | pytorch-cuda | 256 | 15 | 2.049 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | pytorch-cuda | 64 | 15 | 2.446 | 0 |
| take_along_axis_batched_indices_f32_n1024 | pytorch-cuda | 64 | 15 | 2.393 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | pytorch-cuda | 64 | 15 | 2.455 | 0 |
| take_along_axis_batched_indices_f64_n1024 | pytorch-cuda | 64 | 15 | 2.401 | 0 |
| scatter_default_f32_n1024 | pytorch-cuda | 128 | 15 | 3.154 | 0 |
| scatter_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.162 | 0 |
| cast_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.504 | 0 |
| cast_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.697 | 0 |
| pad_default_f32_n1048576 | pytorch-cuda | 63 | 15 | 2.681 | 0 |
| pad_default_f64_n1048576 | pytorch-cuda | 31 | 15 | 2.498 | 0 |
| slice_default_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.051 | 0 |
| slice_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.829 | 0 |
| reverse_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.860 | 0 |
| reverse_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.742 | 0 |
| concatenate_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 1.735 | 0 |
| concatenate_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 1.690 | 0 |
| extract_diag_default_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.329 | 0 |
| extract_diag_default_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.133 | 0 |
| embed_diag_default_f32_n1024 | pytorch-cuda | 64 | 15 | 1.495 | 0 |
| embed_diag_default_f64_n1024 | pytorch-cuda | 32 | 15 | 0.957 | 0 |
| tril_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.454 | 0 |
| tril_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.350 | 0 |
| triu_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.443 | 0 |
| triu_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.340 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | pytorch-cuda | 64 | 15 | 3.225 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | pytorch-cuda | 32 | 15 | 2.025 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.234 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.618 | 0 |
| scatter_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 2.401 | 0 |
| scatter_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.115 | 0 |
| cast_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 1.320 | 0 |
| cast_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.591 | 0 |
| broadcast_in_dim_f32_n1024 | pytorch-cuda | 128 | 15 | 2.122 | 0 |
| broadcast_in_dim_f64_n1024 | pytorch-cuda | 128 | 15 | 2.123 | 0 |
| convert_f32_f64_n1024 | pytorch-cuda | 256 | 15 | 3.728 | 0 |
| broadcast_in_dim_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.109 | 0 |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.902 | 0 |
| convert_f32_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.322 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f32_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f32_n1024 | pytorch-cuda | 256 | 15 | 3.034 | 0 |
| gather_axis1_f32_n1024 | pytorch-cuda | 256 | 15 | 3.000 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f64_n1024 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f64_n1024 | pytorch-cuda | 256 | 15 | 2.949 | 0 |
| gather_axis1_f64_n1024 | pytorch-cuda | 256 | 15 | 2.988 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f32_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.202 | 0 |
| gather_axis1_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.085 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_clamped_f64_n1048576 | pytorch-cuda | — | — | — | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| gather_axis0_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.805 | 0 |
| gather_axis1_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.673 | 0 |
| pad_interior_f32_n1024 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_crop_f32_n1024 | pytorch-cuda | 64 | 15 | 2.044 | 0 |
| pad_interior_crop_f32_n1024 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_interior_f64_n1024 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_crop_f64_n1024 | pytorch-cuda | 128 | 15 | 3.966 | 0 |
| pad_interior_crop_f64_n1024 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_interior_f32_n1048576 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_crop_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.695 | 0 |
| pad_interior_crop_f32_n1048576 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_interior_f64_n1048576 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_crop_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.555 | 0 |
| pad_interior_crop_f64_n1048576 | pytorch-cuda | — | — | — | PyTorch pad has no single-call interior padding counterpart |
| pad_default_f32_n1024 | jax-cuda | 32 | 15 | 3.555 | 0 |
| pad_default_f64_n1024 | jax-cuda | 16 | 15 | 2.010 | 0 |
| slice_default_f32_n1024 | jax-cuda | 16 | 15 | 1.941 | 0 |
| slice_default_f64_n1024 | jax-cuda | 32 | 15 | 3.438 | 0 |
| reverse_default_f32_n1024 | jax-cuda | 32 | 15 | 2.617 | 0 |
| reverse_default_f64_n1024 | jax-cuda | 32 | 15 | 2.554 | 0 |
| concatenate_default_f32_n1024 | jax-cuda | 16 | 15 | 2.488 | 0 |
| concatenate_default_f64_n1024 | jax-cuda | 16 | 15 | 2.389 | 0 |
| extract_diag_default_f32_n1024 | jax-cuda | 32 | 15 | 2.522 | 0 |
| extract_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 2.598 | 0 |
| embed_diag_default_f32_n32 | jax-cuda | 32 | 15 | 2.653 | 0 |
| embed_diag_default_f64_n32 | jax-cuda | 32 | 15 | 2.685 | 0 |
| tril_default_f32_n1024 | jax-cuda | 32 | 15 | 2.498 | 0 |
| tril_default_f64_n1024 | jax-cuda | 32 | 15 | 2.488 | 0 |
| triu_default_f32_n1024 | jax-cuda | 32 | 15 | 2.588 | 0 |
| triu_default_f64_n1024 | jax-cuda | 32 | 15 | 2.334 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | jax-cuda | 32 | 15 | 2.488 | 0 |
| take_along_axis_batched_indices_f32_n1024 | jax-cuda | 32 | 15 | 2.570 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | jax-cuda | 32 | 15 | 2.630 | 0 |
| take_along_axis_batched_indices_f64_n1024 | jax-cuda | 32 | 15 | 2.683 | 0 |
| scatter_default_f32_n1024 | jax-cuda | 16 | 15 | 2.701 | 0 |
| scatter_default_f64_n1024 | jax-cuda | 16 | 15 | 2.800 | 0 |
| cast_default_f32_n1024 | jax-cuda | 16 | 15 | 2.169 | 0 |
| cast_default_f64_n1024 | jax-cuda | 16 | 15 | 2.226 | 0 |
| pad_default_f32_n1048576 | jax-cuda | 16 | 15 | 1.801 | 0 |
| pad_default_f64_n1048576 | jax-cuda | 16 | 15 | 1.751 | 0 |
| slice_default_f32_n1048576 | jax-cuda | 32 | 15 | 3.218 | 0 |
| slice_default_f64_n1048576 | jax-cuda | 32 | 15 | 3.287 | 0 |
| reverse_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.417 | 0 |
| reverse_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.350 | 0 |
| concatenate_default_f32_n1048576 | jax-cuda | 16 | 15 | 2.232 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.306 | 0 |
| extract_diag_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.423 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.599 | 0 |
| embed_diag_default_f32_n1024 | jax-cuda | 32 | 15 | 2.569 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 2.437 | 0 |
| tril_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.277 | 0 |
| tril_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.060 | 0 |
| triu_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.165 | 0 |
| triu_default_f64_n1048576 | jax-cuda | 32 | 15 | 1.979 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | jax-cuda | 32 | 15 | 2.569 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | jax-cuda | 32 | 15 | 2.256 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | 32 | 15 | 2.604 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | jax-cuda | 32 | 15 | 2.820 | 0 |
| scatter_default_f32_n1048576 | jax-cuda | 16 | 15 | 2.586 | 0 |
| scatter_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.445 | 0 |
| cast_default_f32_n1048576 | jax-cuda | 16 | 15 | 1.987 | 0 |
| cast_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.027 | 0 |
| broadcast_in_dim_f32_n1024 | jax-cuda | 32 | 15 | 3.231 | 0 |
| broadcast_in_dim_f64_n1024 | jax-cuda | 32 | 15 | 3.151 | 0 |
| convert_f32_f64_n1024 | jax-cuda | 32 | 15 | 3.804 | 0 |
| broadcast_in_dim_f32_n1048576 | jax-cuda | 32 | 15 | 3.274 | 0 |
| broadcast_in_dim_f64_n1048576 | jax-cuda | 32 | 15 | 3.284 | 0 |
| convert_f32_f64_n1048576 | jax-cuda | 32 | 15 | 3.847 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | jax-cuda | 16 | 15 | 2.344 | 0 |
| dynamic_slice_clamped_f32_n1024 | jax-cuda | 16 | 15 | 1.963 | 0 |
| gather_axis0_f32_n1024 | jax-cuda | 32 | 15 | 3.560 | 0 |
| gather_axis1_f32_n1024 | jax-cuda | 32 | 15 | 3.545 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | jax-cuda | 16 | 15 | 2.132 | 0 |
| dynamic_slice_clamped_f64_n1024 | jax-cuda | 16 | 15 | 2.055 | 0 |
| gather_axis0_f64_n1024 | jax-cuda | 32 | 15 | 3.685 | 0 |
| gather_axis1_f64_n1024 | jax-cuda | 32 | 15 | 3.617 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | jax-cuda | 16 | 15 | 1.964 | 0 |
| dynamic_slice_clamped_f32_n1048576 | jax-cuda | 16 | 15 | 2.072 | 0 |
| gather_axis0_f32_n1048576 | jax-cuda | 32 | 15 | 3.586 | 0 |
| gather_axis1_f32_n1048576 | jax-cuda | 32 | 15 | 3.662 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | 16 | 15 | 2.089 | 0 |
| dynamic_slice_clamped_f64_n1048576 | jax-cuda | 32 | 15 | 3.950 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda | 32 | 15 | 3.547 | 0 |
| gather_axis1_f64_n1048576 | jax-cuda | 16 | 15 | 2.058 | 0 |
| pad_interior_f32_n1024 | jax-cuda | 32 | 15 | 3.576 | 0 |
| pad_crop_f32_n1024 | jax-cuda | 32 | 15 | 3.461 | 0 |
| pad_interior_crop_f32_n1024 | jax-cuda | 16 | 15 | 1.973 | 0 |
| pad_interior_f64_n1024 | jax-cuda | 16 | 15 | 2.371 | 0 |
| pad_crop_f64_n1024 | jax-cuda | 16 | 15 | 2.540 | 0 |
| pad_interior_crop_f64_n1024 | jax-cuda | 16 | 15 | 2.655 | 0 |
| pad_interior_f32_n1048576 | jax-cuda | 10 | 15 | 1.527 | 0 |
| pad_crop_f32_n1048576 | jax-cuda | 16 | 15 | 2.566 | 0 |
| pad_interior_crop_f32_n1048576 | jax-cuda | 10 | 15 | 1.548 | 0 |
| pad_interior_f64_n1048576 | jax-cuda | 5 | 15 | 1.102 | 0 |
| pad_crop_f64_n1048576 | jax-cuda | 32 | 15 | 3.387 | 0 |
| pad_interior_crop_f64_n1048576 | jax-cuda | 5 | 15 | 1.100 | 0 |
| pad_default_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.927 | 0 |
| pad_default_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.195 | 0 |
| slice_default_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.070 | 0 |
| slice_default_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.030 | 0 |
| reverse_default_f32_n1024 | jax-cuda-jit | 16 | 15 | 1.963 | 0 |
| reverse_default_f64_n1024 | jax-cuda-jit | 16 | 15 | 1.916 | 0 |
| concatenate_default_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.066 | 0 |
| concatenate_default_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.026 | 0 |
| extract_diag_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.973 | 0 |
| extract_diag_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.729 | 0 |
| embed_diag_default_f32_n32 | jax-cuda-jit | 32 | 15 | 2.597 | 0 |
| embed_diag_default_f64_n32 | jax-cuda-jit | 32 | 15 | 2.617 | 0 |
| tril_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.610 | 0 |
| tril_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.197 | 0 |
| triu_default_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.027 | 0 |
| triu_default_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.130 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.256 | 0 |
| take_along_axis_batched_indices_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.100 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.203 | 0 |
| take_along_axis_batched_indices_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.219 | 0 |
| scatter_default_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.440 | 0 |
| scatter_default_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.205 | 0 |
| cast_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.199 | 0 |
| cast_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.125 | 0 |
| pad_default_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.591 | 0 |
| pad_default_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.602 | 0 |
| slice_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.042 | 0 |
| slice_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.162 | 0 |
| reverse_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.171 | 0 |
| reverse_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.208 | 0 |
| concatenate_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.305 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.835 | 0 |
| extract_diag_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.370 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.556 | 0 |
| embed_diag_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.428 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.411 | 0 |
| tril_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.082 | 0 |
| tril_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.033 | 0 |
| triu_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.914 | 0 |
| triu_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.082 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.364 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.249 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.951 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.334 | 0 |
| scatter_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.601 | 0 |
| scatter_default_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.275 | 0 |
| cast_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.768 | 0 |
| cast_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.817 | 0 |
| broadcast_in_dim_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.977 | 0 |
| broadcast_in_dim_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.027 | 0 |
| convert_f32_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.134 | 0 |
| broadcast_in_dim_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.972 | 0 |
| broadcast_in_dim_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.930 | 0 |
| convert_f32_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.950 | 0 |
| dynamic_slice_in_bounds_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.344 | 0 |
| dynamic_slice_clamped_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.134 | 0 |
| gather_axis0_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.182 | 0 |
| gather_axis1_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.107 | 0 |
| dynamic_slice_in_bounds_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.150 | 0 |
| dynamic_slice_clamped_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.119 | 0 |
| gather_axis0_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.113 | 0 |
| gather_axis1_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.133 | 0 |
| dynamic_slice_in_bounds_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.079 | 0 |
| dynamic_slice_clamped_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.086 | 0 |
| gather_axis0_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.055 | 0 |
| gather_axis1_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.008 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.176 | 0 |
| dynamic_slice_clamped_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.150 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.506 | 0 |
| gather_axis1_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.325 | 0 |
| pad_interior_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.139 | 0 |
| pad_crop_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.409 | 0 |
| pad_interior_crop_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.432 | 0 |
| pad_interior_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.449 | 0 |
| pad_crop_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.244 | 0 |
| pad_interior_crop_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.630 | 0 |
| pad_interior_f32_n1048576 | jax-cuda-jit | 10 | 15 | 1.069 | 0 |
| pad_crop_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.094 | 0 |
| pad_interior_crop_f32_n1048576 | jax-cuda-jit | 10 | 15 | 1.035 | 0 |
| pad_interior_f64_n1048576 | jax-cuda-jit | 5 | 15 | 1.088 | 0 |
| pad_crop_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.108 | 0 |
| pad_interior_crop_f64_n1048576 | jax-cuda-jit | 5 | 15 | 1.058 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/structural
suite_file: benchmarks/gpu/structural.yaml
timestamp: '2026-10-13T19:49:43Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 73195d8120f0b0f77a1a744740a42019394c9f8a
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
  expected_cases: 82
  selected_cases: 82
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 1020d2229d20abf3a9c8e72470e9706a7e8cc2a9e03c308c82a87ac83a556b7c
    src/bin/benchmark_gpu_public_api/plan.rs: 3d0f10f390ebd0e08a40deb5a64e49e086dcd41b9a777d3a208c6846ed6b2fac
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/multi.rs: 1b477fef78260cf28aa31e515d57ab411f6241b91ffb30d625939538fdbc6bab
    src/bin/benchmark_gpu_public_api/lifetime.rs: c687a8b0329f6d3073073671b5d246acbb1f4dc3f983d1430da897a908ef1745
    src/bin/benchmark_gpu_public_api/read_layout.rs: 184d4508d0729f531ee8107e3b70aa6b77c5a9e952c2dc43606a868cd97631a2
    src/bin/benchmark_gpu_public_api/structural.rs: b8fd99ceb5c80fd3474974b0ba9aa57833ad6bd25383af3091c11c0b6ab94f6d
    src/bin/benchmark_gpu_public_api/linalg.rs: 5f5c4afc161f3d6e267861511d0d5871be9192d9968c95b59a73c9dc5a64e22b
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: 1dc14ac94fac700aa696d0ada04315685e3245738f9514de3d7989b8cb59152b
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 3faaaf5ae74a2ef9c91888b8fd8e566d1cf83b24d2b385cc1cd7b87d9d80c6fb
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: b3ddc9ffba1c735b668c49489062c09d17c1a716784d995e1020dd9b193b0724
    scripts/run_gpu_public_api.sh: dd5e4cc47e85fcb5afdc56d460136b2a399a6a7d6a62005de8ba94afb0119ecc
    data/results/nvidia-gpu/gpu/structural/20261013_194942/fixtures/manifest.json: f604e01a5de8c5c10d9e81548d2ee792e942ccab4ef6f79f5a77ab414b554442
    target/release/benchmark_gpu_public_api: ee6ce17b73be0619773ff3e1789ca52b896ce9662d3d85585d83ac1c67875152
  reference_numpy_version: 2.4.2
```
