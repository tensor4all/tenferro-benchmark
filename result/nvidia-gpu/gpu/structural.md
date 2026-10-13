# gpu/structural: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/structural/20261013_133930/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| pad_default_f32_n1024 | 12.079 (1.403) | 69.780 (9.175) | 28.147 (0.766) | 281.811 (11.615) | 96.227 (2.800) |
| pad_default_f64_n1024 | 15.814 (1.155) | 95.792 (27.098) | 28.313 (1.385) | 358.290 (7.498) | 94.322 (19.193) |
| slice_default_f32_n1024 | 17.812 (1.182) | unsupported | 24.028 (0.462) | 162.504 (4.759) | 85.608 (9.885) |
| slice_default_f64_n1024 | 18.928 (0.887) | unsupported | 23.629 (0.202) | 159.800 (1.394) | 79.939 (9.109) |
| reverse_default_f32_n1024 | 17.560 (0.148) | 98.332 (5.148) | 12.277 (0.183) | 125.956 (3.947) | 79.934 (12.967) |
| reverse_default_f64_n1024 | 17.481 (0.396) | 115.381 (10.129) | 12.486 (0.288) | 123.752 (9.833) | 82.395 (9.222) |
| concatenate_default_f32_n1024 | 28.485 (3.657) | 207.913 (5.442) | 11.675 (0.218) | 198.232 (6.870) | 86.918 (14.103) |
| concatenate_default_f64_n1024 | 29.518 (2.111) | 234.912 (9.777) | 11.600 (0.113) | 205.894 (7.249) | 81.189 (9.752) |
| extract_diag_default_f32_n1024 | 16.608 (0.254) | 145.921 (3.688) | 16.011 (0.177) | 129.309 (4.920) | 76.861 (5.368) |
| extract_diag_default_f64_n1024 | 17.919 (0.295) | 148.058 (25.894) | 15.879 (0.134) | 135.725 (11.973) | 74.835 (6.585) |
| embed_diag_default_f32_n32 | 26.191 (0.654) | 152.017 (2.872) | 22.740 (0.193) | 139.232 (20.321) | 76.120 (12.319) |
| embed_diag_default_f64_n32 | 26.059 (1.893) | 162.710 (4.030) | 22.516 (0.178) | 145.105 (22.848) | 76.306 (5.899) |
| tril_default_f32_n1024 | 16.371 (0.790) | 144.983 (2.692) | 7.932 (0.158) | 107.664 (8.438) | 79.636 (12.899) |
| tril_default_f64_n1024 | 16.635 (0.886) | 151.655 (6.149) | 7.996 (0.116) | 108.842 (9.870) | 80.876 (9.120) |
| triu_default_f32_n1024 | 17.199 (0.312) | 165.376 (5.494) | 7.901 (0.636) | 109.666 (10.157) | 80.108 (10.477) |
| triu_default_f64_n1024 | 16.694 (0.171) | 174.491 (11.445) | 8.538 (0.303) | 126.953 (11.592) | 82.045 (9.828) |
| take_along_axis_broadcast_indices_f32_n1024 | 84.087 (9.544) | 170.900 (3.806) | 42.967 (2.052) | 127.185 (2.903) | 80.808 (13.872) |
| take_along_axis_batched_indices_f32_n1024 | unsupported | unsupported | 40.302 (0.461) | 132.550 (8.192) | 82.827 (10.667) |
| take_along_axis_broadcast_indices_f64_n1024 | 108.606 (9.838) | 172.318 (23.486) | 41.781 (1.233) | 124.537 (10.449) | 84.231 (10.390) |
| take_along_axis_batched_indices_f64_n1024 | unsupported | unsupported | 41.518 (2.946) | 107.421 (10.808) | 85.895 (13.026) |
| scatter_default_f32_n1024 | 41.554 (0.635) | 420.235 (16.236) | 24.414 (0.382) | 274.537 (22.602) | 96.778 (16.028) |
| scatter_default_f64_n1024 | 40.750 (0.337) | 446.398 (11.486) | 24.594 (0.623) | 282.225 (25.472) | 89.567 (9.256) |
| cast_default_f32_n1024 | 15.974 (0.221) | 184.102 (4.510) | 13.124 (0.375) | 195.799 (28.045) | 73.305 (4.015) |
| cast_default_f64_n1024 | 15.741 (0.193) | 186.355 (3.200) | 13.051 (0.430) | 189.486 (24.140) | 72.729 (8.196) |
| pad_default_f32_n1048576 | 70.858 (0.390) | 94.670 (0.037) | 42.553 (0.056) | 329.516 (2.816) | 71.640 (2.446) |
| pad_default_f64_n1048576 | 70.436 (0.242) | 120.621 (0.047) | 80.578 (0.067) | 326.060 (6.750) | 65.265 (2.485) |
| slice_default_f32_n1048576 | 17.619 (0.013) | unsupported | 24.263 (0.757) | 159.317 (0.713) | 68.655 (2.749) |
| slice_default_f64_n1048576 | 30.728 (0.112) | unsupported | 28.549 (0.032) | 160.715 (0.481) | 69.280 (2.443) |
| reverse_default_f32_n1048576 | 31.859 (0.072) | 59.240 (0.235) | 28.836 (0.065) | 108.703 (0.583) | 68.451 (2.250) |
| reverse_default_f64_n1048576 | 58.161 (0.175) | 105.464 (0.174) | 54.310 (0.106) | 105.186 (1.877) | 64.736 (1.648) |
| concatenate_default_f32_n1048576 | 64.636 (0.110) | 117.647 (0.193) | 54.037 (0.054) | 192.609 (0.551) | 66.023 (1.077) |
| concatenate_default_f64_n1048576 | 116.878 (0.803) | 215.720 (0.156) | 105.249 (0.109) | 153.366 (3.257) | 114.853 (1.564) |
| extract_diag_default_f32_n1048576 | 9.513 (0.071) | 44.414 (0.844) | 15.974 (0.187) | 85.070 (2.692) | 67.060 (1.250) |
| extract_diag_default_f64_n1048576 | 23.052 (1.427) | 56.322 (0.072) | 16.075 (0.082) | 88.975 (4.625) | 69.865 (6.765) |
| embed_diag_default_f32_n1024 | 18.235 (0.285) | 164.506 (12.463) | 23.807 (0.293) | 90.101 (3.734) | 69.045 (1.992) |
| embed_diag_default_f64_n1024 | 31.545 (0.213) | 137.800 (11.737) | 29.520 (0.081) | 85.055 (0.998) | 69.537 (0.751) |
| tril_default_f32_n1048576 | 27.304 (0.214) | 54.316 (0.166) | 22.528 (0.059) | 76.121 (0.967) | 69.843 (1.676) |
| tril_default_f64_n1048576 | 44.285 (0.423) | 95.246 (0.222) | 41.843 (0.084) | 74.075 (1.978) | 64.755 (1.522) |
| triu_default_f32_n1048576 | 27.403 (0.118) | 54.572 (0.184) | 22.411 (0.090) | 76.105 (0.783) | 70.407 (1.731) |
| triu_default_f64_n1048576 | 44.322 (0.315) | 95.637 (0.158) | 41.692 (0.166) | 74.314 (0.735) | 64.639 (1.181) |
| take_along_axis_broadcast_indices_f32_n1048576 | 42.473 (1.007) | 100.861 (11.176) | 49.960 (0.134) | 91.724 (8.743) | 80.561 (10.430) |
| take_along_axis_batched_indices_f32_n1048576 | unsupported | unsupported | 62.997 (0.362) | 81.149 (1.272) | 78.184 (8.039) |
| take_along_axis_broadcast_indices_f64_n1048576 | 192.518 (8.956) | 119.936 (3.406) | 69.406 (0.209) | 89.322 (8.180) | 81.465 (6.147) |
| take_along_axis_batched_indices_f64_n1048576 | unsupported | unsupported | 81.831 (0.146) | 95.103 (14.362) | 82.223 (10.639) |
| scatter_default_f32_n1048576 | 79.072 (0.616) | 148.510 (0.334) | 74.847 (0.115) | 168.668 (9.650) | 82.865 (2.849) |
| scatter_default_f64_n1048576 | 140.811 (1.032) | 241.621 (0.636) | 132.167 (0.115) | 164.560 (6.264) | 140.024 (3.211) |
| cast_default_f32_n1048576 | 43.285 (0.187) | 69.311 (0.265) | 41.211 (0.048) | 138.523 (3.081) | 65.739 (2.413) |
| cast_default_f64_n1048576 | 41.620 (0.269) | 94.650 (0.186) | 40.451 (0.044) | 137.269 (6.871) | 64.014 (1.081) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| pad_default_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 3.092 | 0 |
| pad_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.024 | 0 |
| slice_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.280 | 0 |
| slice_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.423 | 0 |
| reverse_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.248 | 0 |
| reverse_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.238 | 0 |
| concatenate_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 3.646 | 0 |
| concatenate_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.889 | 0 |
| extract_diag_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.126 | 0 |
| extract_diag_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 1.147 | 0 |
| embed_diag_default_f32_n32 | tenferro-cuda-direct | 128 | 15 | 3.353 | 0 |
| embed_diag_default_f64_n32 | tenferro-cuda-direct | 128 | 15 | 3.336 | 0 |
| tril_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.095 | 0 |
| tril_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.129 | 0 |
| triu_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.201 | 0 |
| triu_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.137 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.691 | 0 |
| take_along_axis_batched_indices_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 3.475 | 0 |
| take_along_axis_batched_indices_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.659 | 0 |
| scatter_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.608 | 0 |
| cast_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.045 | 0 |
| cast_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.015 | 0 |
| pad_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.267 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-direct | 31 | 15 | 2.184 | 0 |
| slice_default_f32_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.255 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.967 | 0 |
| reverse_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.039 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.861 | 0 |
| concatenate_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.068 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.870 | 0 |
| extract_diag_default_f32_n1048576 | tenferro-cuda-direct | 256 | 15 | 2.435 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | 256 | 15 | 5.901 | 0 |
| embed_diag_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 1.167 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 1.009 | 0 |
| tril_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.747 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.417 | 0 |
| triu_default_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.754 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.418 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.718 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.080 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.530 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.253 | 0 |
| cast_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.385 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.664 | 0 |
| pad_default_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 4.466 | 0 |
| pad_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 3.065 | 0 |
| slice_default_f32_n1024 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| slice_default_f64_n1024 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| reverse_default_f32_n1024 | tenferro-cuda-eager | 32 | 15 | 3.147 | 0 |
| reverse_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 3.692 | 0 |
| concatenate_default_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 1.663 | 0 |
| concatenate_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.759 | 0 |
| extract_diag_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.335 | 0 |
| extract_diag_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.369 | 0 |
| embed_diag_default_f32_n32 | tenferro-cuda-eager | 16 | 15 | 2.432 | 0 |
| embed_diag_default_f64_n32 | tenferro-cuda-eager | 16 | 15 | 2.603 | 0 |
| tril_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.320 | 0 |
| tril_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.426 | 0 |
| triu_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.646 | 0 |
| triu_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.792 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.734 | 0 |
| take_along_axis_batched_indices_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.757 | 0 |
| take_along_axis_batched_indices_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.362 | 0 |
| scatter_default_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.571 | 0 |
| cast_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.946 | 0 |
| cast_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.982 | 0 |
| pad_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.029 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-eager | 31 | 15 | 3.739 | 0 |
| slice_default_f32_n1048576 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| slice_default_f64_n1048576 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| reverse_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.791 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.375 | 0 |
| concatenate_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.765 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.452 | 0 |
| extract_diag_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.842 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.605 | 0 |
| embed_diag_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.632 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 4.410 | 0 |
| tril_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.476 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.048 | 0 |
| triu_default_f32_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.493 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.060 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.228 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.838 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.376 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.866 | 0 |
| cast_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.218 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.029 | 0 |
| pad_default_f32_n1024 | pytorch-cuda | 128 | 15 | 3.603 | 0 |
| pad_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.624 | 0 |
| slice_default_f32_n1024 | pytorch-cuda | 128 | 15 | 3.076 | 0 |
| slice_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.025 | 0 |
| reverse_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.143 | 0 |
| reverse_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.197 | 0 |
| concatenate_default_f32_n1024 | pytorch-cuda | 256 | 15 | 2.989 | 0 |
| concatenate_default_f64_n1024 | pytorch-cuda | 256 | 15 | 2.970 | 0 |
| extract_diag_default_f32_n1024 | pytorch-cuda | 128 | 15 | 2.049 | 0 |
| extract_diag_default_f64_n1024 | pytorch-cuda | 128 | 15 | 2.032 | 0 |
| embed_diag_default_f32_n32 | pytorch-cuda | 128 | 15 | 2.911 | 0 |
| embed_diag_default_f64_n32 | pytorch-cuda | 128 | 15 | 2.882 | 0 |
| tril_default_f32_n1024 | pytorch-cuda | 256 | 15 | 2.031 | 0 |
| tril_default_f64_n1024 | pytorch-cuda | 256 | 15 | 2.047 | 0 |
| triu_default_f32_n1024 | pytorch-cuda | 512 | 15 | 4.046 | 0 |
| triu_default_f64_n1024 | pytorch-cuda | 256 | 15 | 2.186 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | pytorch-cuda | 64 | 15 | 2.750 | 0 |
| take_along_axis_batched_indices_f32_n1024 | pytorch-cuda | 64 | 15 | 2.579 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | pytorch-cuda | 64 | 15 | 2.674 | 0 |
| take_along_axis_batched_indices_f64_n1024 | pytorch-cuda | 64 | 15 | 2.657 | 0 |
| scatter_default_f32_n1024 | pytorch-cuda | 128 | 15 | 3.125 | 0 |
| scatter_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.148 | 0 |
| cast_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.360 | 0 |
| cast_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.341 | 0 |
| pad_default_f32_n1048576 | pytorch-cuda | 63 | 15 | 2.681 | 0 |
| pad_default_f64_n1048576 | pytorch-cuda | 31 | 15 | 2.498 | 0 |
| slice_default_f32_n1048576 | pytorch-cuda | 128 | 15 | 3.106 | 0 |
| slice_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.827 | 0 |
| reverse_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.846 | 0 |
| reverse_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.738 | 0 |
| concatenate_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 1.729 | 0 |
| concatenate_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 1.684 | 0 |
| extract_diag_default_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.045 | 0 |
| extract_diag_default_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.058 | 0 |
| embed_diag_default_f32_n1024 | pytorch-cuda | 64 | 15 | 1.524 | 0 |
| embed_diag_default_f64_n1024 | pytorch-cuda | 32 | 15 | 0.945 | 0 |
| tril_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.442 | 0 |
| tril_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.339 | 0 |
| triu_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.434 | 0 |
| triu_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.334 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | pytorch-cuda | 64 | 15 | 3.197 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | pytorch-cuda | 32 | 15 | 2.016 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.221 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.619 | 0 |
| scatter_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 2.395 | 0 |
| scatter_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.115 | 0 |
| cast_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 1.319 | 0 |
| cast_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.589 | 0 |
| pad_default_f32_n1024 | jax-cuda | 8 | 15 | 2.254 | 0 |
| pad_default_f64_n1024 | jax-cuda | 8 | 15 | 2.866 | 0 |
| slice_default_f32_n1024 | jax-cuda | 16 | 15 | 2.600 | 0 |
| slice_default_f64_n1024 | jax-cuda | 16 | 15 | 2.557 | 0 |
| reverse_default_f32_n1024 | jax-cuda | 16 | 15 | 2.015 | 0 |
| reverse_default_f64_n1024 | jax-cuda | 16 | 15 | 1.980 | 0 |
| concatenate_default_f32_n1024 | jax-cuda | 16 | 15 | 3.172 | 0 |
| concatenate_default_f64_n1024 | jax-cuda | 16 | 15 | 3.294 | 0 |
| extract_diag_default_f32_n1024 | jax-cuda | 16 | 15 | 2.069 | 0 |
| extract_diag_default_f64_n1024 | jax-cuda | 16 | 15 | 2.172 | 0 |
| embed_diag_default_f32_n32 | jax-cuda | 16 | 15 | 2.228 | 0 |
| embed_diag_default_f64_n32 | jax-cuda | 16 | 15 | 2.322 | 0 |
| tril_default_f32_n1024 | jax-cuda | 32 | 15 | 3.445 | 0 |
| tril_default_f64_n1024 | jax-cuda | 32 | 15 | 3.483 | 0 |
| triu_default_f32_n1024 | jax-cuda | 32 | 15 | 3.509 | 0 |
| triu_default_f64_n1024 | jax-cuda | 16 | 15 | 2.031 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | jax-cuda | 16 | 15 | 2.035 | 0 |
| take_along_axis_batched_indices_f32_n1024 | jax-cuda | 16 | 15 | 2.121 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | jax-cuda | 16 | 15 | 1.993 | 0 |
| take_along_axis_batched_indices_f64_n1024 | jax-cuda | 32 | 15 | 3.437 | 0 |
| scatter_default_f32_n1024 | jax-cuda | 8 | 15 | 2.196 | 0 |
| scatter_default_f64_n1024 | jax-cuda | 8 | 15 | 2.258 | 0 |
| cast_default_f32_n1024 | jax-cuda | 16 | 15 | 3.133 | 0 |
| cast_default_f64_n1024 | jax-cuda | 16 | 15 | 3.032 | 0 |
| pad_default_f32_n1048576 | jax-cuda | 8 | 15 | 2.636 | 0 |
| pad_default_f64_n1048576 | jax-cuda | 8 | 15 | 2.608 | 0 |
| slice_default_f32_n1048576 | jax-cuda | 16 | 15 | 2.549 | 0 |
| slice_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.571 | 0 |
| reverse_default_f32_n1048576 | jax-cuda | 32 | 15 | 3.478 | 0 |
| reverse_default_f64_n1048576 | jax-cuda | 32 | 15 | 3.366 | 0 |
| concatenate_default_f32_n1048576 | jax-cuda | 16 | 15 | 3.082 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.454 | 0 |
| extract_diag_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.722 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.847 | 0 |
| embed_diag_default_f32_n1024 | jax-cuda | 32 | 15 | 2.883 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 2.722 | 0 |
| tril_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.436 | 0 |
| tril_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.370 | 0 |
| triu_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.435 | 0 |
| triu_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.378 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | jax-cuda | 32 | 15 | 2.935 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | jax-cuda | 32 | 15 | 2.597 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | 32 | 15 | 2.858 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | jax-cuda | 32 | 15 | 3.043 | 0 |
| scatter_default_f32_n1048576 | jax-cuda | 16 | 15 | 2.699 | 0 |
| scatter_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.633 | 0 |
| cast_default_f32_n1048576 | jax-cuda | 16 | 15 | 2.216 | 0 |
| cast_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.196 | 0 |
| pad_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.079 | 0 |
| pad_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.018 | 0 |
| slice_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.739 | 0 |
| slice_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.558 | 0 |
| reverse_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.558 | 0 |
| reverse_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.637 | 0 |
| concatenate_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.781 | 0 |
| concatenate_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.598 | 0 |
| extract_diag_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.460 | 0 |
| extract_diag_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.395 | 0 |
| embed_diag_default_f32_n32 | jax-cuda-jit | 32 | 15 | 2.436 | 0 |
| embed_diag_default_f64_n32 | jax-cuda-jit | 32 | 15 | 2.442 | 0 |
| tril_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.548 | 0 |
| tril_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.588 | 0 |
| triu_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.563 | 0 |
| triu_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.625 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.586 | 0 |
| take_along_axis_batched_indices_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.650 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.695 | 0 |
| take_along_axis_batched_indices_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.749 | 0 |
| scatter_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.097 | 0 |
| scatter_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.866 | 0 |
| cast_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.346 | 0 |
| cast_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.327 | 0 |
| pad_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.292 | 0 |
| pad_default_f64_n1048576 | jax-cuda-jit | 31 | 15 | 2.023 | 0 |
| slice_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.197 | 0 |
| slice_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.217 | 0 |
| reverse_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.190 | 0 |
| reverse_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.072 | 0 |
| concatenate_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.113 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.838 | 0 |
| extract_diag_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.146 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.236 | 0 |
| embed_diag_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.209 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.225 | 0 |
| tril_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.235 | 0 |
| tril_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.072 | 0 |
| triu_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.253 | 0 |
| triu_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.068 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.578 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.502 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.607 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.631 | 0 |
| scatter_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.652 | 0 |
| scatter_default_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.240 | 0 |
| cast_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.104 | 0 |
| cast_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.048 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/structural
suite_file: benchmarks/gpu/structural.yaml
timestamp: '2026-10-13T13:39:31Z'
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
    ./scripts/run_gpu_public_api.sh structural
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 48
  selected_cases: 48
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
    data/results/nvidia-gpu/gpu/structural/20261013_133930/fixtures/manifest.json: b8881ec3ec33df4edad659aa32d13c3cc89286bcc67ec701527f999691786074
    target/release/benchmark_gpu_public_api: b505a2bdb3d0c5a7bffd1390af659a154753ea998ec68becd4779e137c9c7150
  reference_numpy_version: 2.4.2
```
