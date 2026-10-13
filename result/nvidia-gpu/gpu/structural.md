# gpu/structural: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/structural/20261013_101528/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| pad_default_f32_n1024 | 17.939 (0.690) | 80.086 (21.552) | 28.387 (0.239) | 411.945 (23.250) | 83.406 (2.607) |
| pad_default_f64_n1024 | 26.214 (0.567) | 165.984 (10.755) | 29.090 (2.718) | 464.812 (15.376) | 88.498 (11.646) |
| slice_default_f32_n1024 | 33.647 (0.568) | unsupported | 22.543 (0.131) | 208.632 (38.765) | 80.047 (12.019) |
| slice_default_f64_n1024 | 34.938 (1.586) | unsupported | 23.672 (1.595) | 163.754 (27.369) | 74.247 (1.076) |
| reverse_default_f32_n1024 | 34.150 (0.640) | 186.372 (6.625) | 12.647 (0.093) | 121.541 (14.118) | 77.561 (5.494) |
| reverse_default_f64_n1024 | 36.490 (0.539) | 213.465 (25.083) | 12.661 (0.382) | 101.340 (1.623) | 86.092 (6.584) |
| concatenate_default_f32_n1024 | 71.364 (1.389) | 592.615 (60.906) | 11.662 (0.061) | 224.310 (17.417) | 86.570 (4.988) |
| concatenate_default_f64_n1024 | 74.525 (0.703) | 458.639 (43.200) | 11.372 (0.381) | 266.595 (31.777) | 87.612 (8.480) |
| extract_diag_default_f32_n1024 | 40.490 (0.807) | 128.949 (7.353) | 15.749 (0.218) | 124.463 (11.153) | 81.067 (1.677) |
| extract_diag_default_f64_n1024 | 43.660 (0.903) | 143.561 (5.787) | 15.793 (0.149) | 108.309 (13.287) | 82.317 (5.588) |
| embed_diag_default_f32_n32 | 46.591 (1.722) | 179.418 (7.661) | 22.433 (2.152) | 123.305 (9.030) | 83.842 (6.422) |
| embed_diag_default_f64_n32 | 46.066 (1.856) | 199.968 (8.394) | 21.996 (0.094) | 131.929 (11.017) | 81.881 (1.817) |
| tril_default_f32_n1024 | 44.261 (0.940) | 207.007 (12.842) | 7.814 (0.296) | 95.876 (18.368) | 86.847 (6.203) |
| tril_default_f64_n1024 | 44.094 (1.278) | 228.599 (14.010) | 7.706 (0.039) | 92.427 (10.705) | 85.252 (6.909) |
| triu_default_f32_n1024 | 44.035 (1.312) | 252.647 (9.465) | 7.844 (0.043) | 95.684 (13.344) | 86.981 (7.859) |
| triu_default_f64_n1024 | 45.329 (2.985) | 273.472 (4.472) | 7.920 (0.096) | 98.293 (12.558) | 85.825 (8.592) |
| take_along_axis_broadcast_indices_f32_n1024 | 109.256 (26.672) | 312.468 (6.936) | 41.197 (3.420) | 93.710 (0.720) | 89.734 (8.799) |
| take_along_axis_batched_indices_f32_n1024 | unsupported | unsupported | 38.176 (0.217) | 100.010 (9.105) | 89.325 (5.540) |
| take_along_axis_broadcast_indices_f64_n1024 | 183.111 (6.939) | 324.148 (5.251) | 38.812 (0.188) | 100.220 (10.796) | 87.212 (3.339) |
| take_along_axis_batched_indices_f64_n1024 | unsupported | unsupported | 38.447 (0.833) | 101.505 (14.927) | 91.540 (10.445) |
| scatter_default_f32_n1024 | 221.686 (2.998) | 930.949 (49.251) | 25.939 (1.274) | 262.291 (9.906) | 97.331 (14.999) |
| scatter_default_f64_n1024 | 220.910 (3.576) | 774.991 (132.228) | 25.683 (0.188) | 287.372 (28.047) | 97.014 (6.599) |
| cast_default_f32_n1024 | 13.521 (0.221) | 126.029 (5.354) | 12.365 (0.550) | 193.804 (41.770) | 77.297 (3.045) |
| cast_default_f64_n1024 | 13.368 (0.251) | 125.631 (5.135) | 12.633 (0.521) | 177.450 (30.940) | 80.387 (4.935) |
| pad_default_f32_n1048576 | 178.011 (9.408) | 269.776 (10.360) | 42.607 (0.033) | 280.623 (14.727) | 81.009 (5.231) |
| pad_default_f64_n1048576 | 223.638 (5.502) | 295.505 (4.580) | 80.591 (0.109) | 290.513 (8.369) | 81.245 (2.797) |
| slice_default_f32_n1048576 | 270.211 (9.722) | unsupported | 23.168 (0.284) | 140.569 (0.588) | 84.095 (1.485) |
| slice_default_f64_n1048576 | 259.856 (13.747) | unsupported | 28.544 (0.061) | 146.544 (6.420) | 82.741 (1.246) |
| reverse_default_f32_n1048576 | 255.298 (12.825) | 475.171 (47.685) | 28.893 (0.025) | 105.933 (1.058) | 82.522 (1.219) |
| reverse_default_f64_n1048576 | 250.347 (9.679) | 546.964 (35.271) | 54.413 (0.052) | 104.360 (2.645) | 81.166 (3.153) |
| concatenate_default_f32_n1048576 | 486.989 (22.154) | 1009.707 (133.488) | 54.186 (0.039) | 188.259 (0.967) | 74.297 (4.773) |
| concatenate_default_f64_n1048576 | 487.965 (17.702) | 1004.271 (113.154) | 105.445 (0.191) | 191.209 (1.202) | 114.597 (2.001) |
| extract_diag_default_f32_n1048576 | 252.809 (9.933) | 561.774 (34.069) | 16.105 (3.252) | 109.333 (14.048) | 69.384 (4.679) |
| extract_diag_default_f64_n1048576 | 208.664 (21.435) | 550.505 (26.305) | 15.925 (0.087) | 127.143 (8.899) | 72.563 (2.201) |
| embed_diag_default_f32_n1024 | 237.299 (26.516) | 792.679 (36.856) | 22.731 (0.377) | 116.458 (8.893) | 74.775 (1.858) |
| embed_diag_default_f64_n1024 | 290.651 (24.853) | 762.113 (24.315) | 29.742 (0.059) | 102.132 (3.035) | 75.406 (1.443) |
| tril_default_f32_n1048576 | 285.653 (10.790) | 580.696 (34.811) | 22.720 (0.041) | 90.296 (0.975) | 76.159 (2.938) |
| tril_default_f64_n1048576 | 252.314 (16.119) | 577.749 (25.705) | 42.060 (0.042) | 92.438 (2.771) | 71.149 (1.679) |
| triu_default_f32_n1048576 | 272.735 (7.133) | 588.236 (17.409) | 22.563 (0.036) | 89.862 (1.283) | 74.057 (1.876) |
| triu_default_f64_n1048576 | 257.100 (10.202) | 583.216 (16.597) | 41.856 (0.102) | 92.661 (3.285) | 81.321 (3.574) |
| take_along_axis_broadcast_indices_f32_n1048576 | 516.922 (43.878) | 746.461 (34.727) | 50.399 (0.266) | 101.986 (4.421) | 86.157 (6.645) |
| take_along_axis_batched_indices_f32_n1048576 | unsupported | unsupported | 63.191 (0.105) | 96.267 (8.712) | 79.344 (1.464) |
| take_along_axis_broadcast_indices_f64_n1048576 | 547.819 (23.665) | 537.054 (39.626) | 69.683 (0.180) | 123.561 (18.162) | 95.520 (6.649) |
| take_along_axis_batched_indices_f64_n1048576 | unsupported | unsupported | 82.013 (0.121) | 136.087 (8.613) | 90.393 (3.996) |
| scatter_default_f32_n1048576 | 612.608 (22.627) | 767.861 (47.236) | 74.881 (0.131) | 209.332 (13.160) | 96.718 (5.222) |
| scatter_default_f64_n1048576 | 607.042 (13.938) | 772.899 (44.938) | 132.177 (0.188) | 207.471 (9.305) | 137.501 (1.188) |
| cast_default_f32_n1048576 | 46.017 (0.256) | 70.805 (0.627) | 41.216 (0.076) | 162.241 (3.211) | 84.347 (1.666) |
| cast_default_f64_n1048576 | 42.470 (0.046) | 95.916 (0.205) | 40.444 (0.044) | 163.042 (3.538) | 84.267 (1.325) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| pad_default_f32_n1024 | tenferro-cuda-direct | 256 | 15 | 4.592 | 0 |
| pad_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 3.355 | 0 |
| slice_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.153 | 0 |
| slice_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.236 | 0 |
| reverse_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.186 | 0 |
| reverse_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.335 | 0 |
| concatenate_default_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.284 | 0 |
| concatenate_default_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.385 | 0 |
| extract_diag_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.591 | 0 |
| extract_diag_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.794 | 0 |
| embed_diag_default_f32_n32 | tenferro-cuda-direct | 64 | 15 | 2.982 | 0 |
| embed_diag_default_f64_n32 | tenferro-cuda-direct | 64 | 15 | 2.948 | 0 |
| tril_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.833 | 0 |
| tril_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.822 | 0 |
| triu_default_f32_n1024 | tenferro-cuda-direct | 64 | 15 | 2.818 | 0 |
| triu_default_f64_n1024 | tenferro-cuda-direct | 64 | 15 | 2.901 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 3.496 | 0 |
| take_along_axis_batched_indices_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.930 | 0 |
| take_along_axis_batched_indices_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.547 | 0 |
| scatter_default_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.535 | 0 |
| cast_default_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 1.731 | 0 |
| cast_default_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 1.711 | 0 |
| pad_default_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.848 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.578 | 0 |
| slice_default_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.162 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.079 | 0 |
| reverse_default_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.042 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 4.006 | 0 |
| concatenate_default_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 3.896 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 3.904 | 0 |
| extract_diag_default_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 4.045 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.339 | 0 |
| embed_diag_default_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.797 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | 8 | 15 | 2.325 | 0 |
| tril_default_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.285 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 4.037 | 0 |
| triu_default_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.182 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.057 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 4.135 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.191 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.450 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 2.428 | 0 |
| cast_default_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.473 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.718 | 0 |
| pad_default_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 5.126 | 0 |
| pad_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.656 | 0 |
| slice_default_f32_n1024 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| slice_default_f64_n1024 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| reverse_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.982 | 0 |
| reverse_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.415 | 0 |
| concatenate_default_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 4.741 | 0 |
| concatenate_default_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 1.835 | 0 |
| extract_diag_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.063 | 0 |
| extract_diag_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.297 | 0 |
| embed_diag_default_f32_n32 | tenferro-cuda-eager | 16 | 15 | 2.871 | 0 |
| embed_diag_default_f64_n32 | tenferro-cuda-eager | 16 | 15 | 3.199 | 0 |
| tril_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.312 | 0 |
| tril_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.658 | 0 |
| triu_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 4.042 | 0 |
| triu_default_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.188 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.500 | 0 |
| take_along_axis_batched_indices_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.593 | 0 |
| take_along_axis_batched_indices_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 3.724 | 0 |
| scatter_default_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 3.100 | 0 |
| cast_default_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 2.016 | 0 |
| cast_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 4.020 | 0 |
| pad_default_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 4.316 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.364 | 0 |
| slice_default_f32_n1048576 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| slice_default_f64_n1048576 | tenferro-cuda-eager | — | — | — | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| reverse_default_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.801 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 4.376 | 0 |
| concatenate_default_f32_n1048576 | tenferro-cuda-eager | 4 | 15 | 4.039 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 4.017 | 0 |
| extract_diag_default_f32_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.247 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.202 | 0 |
| embed_diag_default_f32_n1024 | tenferro-cuda-eager | 4 | 15 | 3.171 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | 4 | 15 | 3.048 | 0 |
| tril_default_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 4.646 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 4.622 | 0 |
| triu_default_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 4.706 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 4.666 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.986 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 4.296 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: batch-coordinate construction uploads scalar constants inside take_along_axis; unavailable under setup-free timing policy. |
| scatter_default_f32_n1048576 | tenferro-cuda-eager | 4 | 15 | 3.071 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 3.092 | 0 |
| cast_default_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.266 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.069 | 0 |
| pad_default_f32_n1024 | pytorch-cuda | 64 | 15 | 1.817 | 0 |
| pad_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.723 | 0 |
| slice_default_f32_n1024 | pytorch-cuda | 128 | 15 | 2.885 | 0 |
| slice_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.030 | 0 |
| reverse_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.238 | 0 |
| reverse_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.241 | 0 |
| concatenate_default_f32_n1024 | pytorch-cuda | 256 | 15 | 2.985 | 0 |
| concatenate_default_f64_n1024 | pytorch-cuda | 256 | 15 | 2.911 | 0 |
| extract_diag_default_f32_n1024 | pytorch-cuda | 128 | 15 | 2.016 | 0 |
| extract_diag_default_f64_n1024 | pytorch-cuda | 128 | 15 | 2.021 | 0 |
| embed_diag_default_f32_n32 | pytorch-cuda | 128 | 15 | 2.871 | 0 |
| embed_diag_default_f64_n32 | pytorch-cuda | 128 | 15 | 2.815 | 0 |
| tril_default_f32_n1024 | pytorch-cuda | 256 | 15 | 2.000 | 0 |
| tril_default_f64_n1024 | pytorch-cuda | 256 | 15 | 1.973 | 0 |
| triu_default_f32_n1024 | pytorch-cuda | 256 | 15 | 2.008 | 0 |
| triu_default_f64_n1024 | pytorch-cuda | 256 | 15 | 2.028 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | pytorch-cuda | 64 | 15 | 2.637 | 0 |
| take_along_axis_batched_indices_f32_n1024 | pytorch-cuda | 64 | 15 | 2.443 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | pytorch-cuda | 64 | 15 | 2.484 | 0 |
| take_along_axis_batched_indices_f64_n1024 | pytorch-cuda | 64 | 15 | 2.461 | 0 |
| scatter_default_f32_n1024 | pytorch-cuda | 64 | 15 | 1.660 | 0 |
| scatter_default_f64_n1024 | pytorch-cuda | 128 | 15 | 3.287 | 0 |
| cast_default_f32_n1024 | pytorch-cuda | 256 | 15 | 3.166 | 0 |
| cast_default_f64_n1024 | pytorch-cuda | 256 | 15 | 3.234 | 0 |
| pad_default_f32_n1048576 | pytorch-cuda | 63 | 15 | 2.684 | 0 |
| pad_default_f64_n1048576 | pytorch-cuda | 31 | 15 | 2.498 | 0 |
| slice_default_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.965 | 0 |
| slice_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.827 | 0 |
| reverse_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.849 | 0 |
| reverse_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.741 | 0 |
| concatenate_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 1.734 | 0 |
| concatenate_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 1.687 | 0 |
| extract_diag_default_f32_n1048576 | pytorch-cuda | 128 | 15 | 2.061 | 0 |
| extract_diag_default_f64_n1048576 | pytorch-cuda | 128 | 15 | 2.038 | 0 |
| embed_diag_default_f32_n1024 | pytorch-cuda | 64 | 15 | 1.455 | 0 |
| embed_diag_default_f64_n1024 | pytorch-cuda | 32 | 15 | 0.952 | 0 |
| tril_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.454 | 0 |
| tril_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.346 | 0 |
| triu_default_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.444 | 0 |
| triu_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.339 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | pytorch-cuda | 64 | 15 | 3.226 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | pytorch-cuda | 32 | 15 | 2.022 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.230 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | pytorch-cuda | 32 | 15 | 2.624 | 0 |
| scatter_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 2.396 | 0 |
| scatter_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 2.115 | 0 |
| cast_default_f32_n1048576 | pytorch-cuda | 32 | 15 | 1.319 | 0 |
| cast_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 2.588 | 0 |
| pad_default_f32_n1024 | jax-cuda | 8 | 15 | 3.296 | 0 |
| pad_default_f64_n1024 | jax-cuda | 8 | 15 | 3.718 | 0 |
| slice_default_f32_n1024 | jax-cuda | 16 | 15 | 3.338 | 0 |
| slice_default_f64_n1024 | jax-cuda | 16 | 15 | 2.620 | 0 |
| reverse_default_f32_n1024 | jax-cuda | 16 | 15 | 1.945 | 0 |
| reverse_default_f64_n1024 | jax-cuda | 32 | 15 | 3.243 | 0 |
| concatenate_default_f32_n1024 | jax-cuda | 8 | 15 | 1.794 | 0 |
| concatenate_default_f64_n1024 | jax-cuda | 8 | 15 | 2.133 | 0 |
| extract_diag_default_f32_n1024 | jax-cuda | 16 | 15 | 1.991 | 0 |
| extract_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 3.466 | 0 |
| embed_diag_default_f32_n32 | jax-cuda | 16 | 15 | 1.973 | 0 |
| embed_diag_default_f64_n32 | jax-cuda | 16 | 15 | 2.111 | 0 |
| tril_default_f32_n1024 | jax-cuda | 32 | 15 | 3.068 | 0 |
| tril_default_f64_n1024 | jax-cuda | 32 | 15 | 2.958 | 0 |
| triu_default_f32_n1024 | jax-cuda | 32 | 15 | 3.062 | 0 |
| triu_default_f64_n1024 | jax-cuda | 32 | 15 | 3.145 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | jax-cuda | 32 | 15 | 2.999 | 0 |
| take_along_axis_batched_indices_f32_n1024 | jax-cuda | 32 | 15 | 3.200 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | jax-cuda | 32 | 15 | 3.207 | 0 |
| take_along_axis_batched_indices_f64_n1024 | jax-cuda | 32 | 15 | 3.248 | 0 |
| scatter_default_f32_n1024 | jax-cuda | 8 | 15 | 2.098 | 0 |
| scatter_default_f64_n1024 | jax-cuda | 8 | 15 | 2.299 | 0 |
| cast_default_f32_n1024 | jax-cuda | 16 | 15 | 3.101 | 0 |
| cast_default_f64_n1024 | jax-cuda | 16 | 15 | 2.839 | 0 |
| pad_default_f32_n1048576 | jax-cuda | 8 | 15 | 2.245 | 0 |
| pad_default_f64_n1048576 | jax-cuda | 8 | 15 | 2.324 | 0 |
| slice_default_f32_n1048576 | jax-cuda | 16 | 15 | 2.249 | 0 |
| slice_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.345 | 0 |
| reverse_default_f32_n1048576 | jax-cuda | 32 | 15 | 3.390 | 0 |
| reverse_default_f64_n1048576 | jax-cuda | 32 | 15 | 3.340 | 0 |
| concatenate_default_f32_n1048576 | jax-cuda | 16 | 15 | 3.012 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda | 16 | 15 | 3.059 | 0 |
| extract_diag_default_f32_n1048576 | jax-cuda | 32 | 15 | 3.499 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.034 | 0 |
| embed_diag_default_f32_n1024 | jax-cuda | 16 | 15 | 1.863 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 3.268 | 0 |
| tril_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.889 | 0 |
| tril_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.958 | 0 |
| triu_default_f32_n1048576 | jax-cuda | 32 | 15 | 2.876 | 0 |
| triu_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.965 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | jax-cuda | 32 | 15 | 3.264 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | jax-cuda | 32 | 15 | 3.081 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | 16 | 15 | 1.977 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | jax-cuda | 16 | 15 | 2.177 | 0 |
| scatter_default_f32_n1048576 | jax-cuda | 16 | 15 | 3.349 | 0 |
| scatter_default_f64_n1048576 | jax-cuda | 16 | 15 | 3.320 | 0 |
| cast_default_f32_n1048576 | jax-cuda | 16 | 15 | 2.596 | 0 |
| cast_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.609 | 0 |
| pad_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.669 | 0 |
| pad_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.832 | 0 |
| slice_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.561 | 0 |
| slice_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.376 | 0 |
| reverse_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.482 | 0 |
| reverse_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.755 | 0 |
| concatenate_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.770 | 0 |
| concatenate_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.804 | 0 |
| extract_diag_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.594 | 0 |
| extract_diag_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.634 | 0 |
| embed_diag_default_f32_n32 | jax-cuda-jit | 32 | 15 | 2.683 | 0 |
| embed_diag_default_f64_n32 | jax-cuda-jit | 32 | 15 | 2.620 | 0 |
| tril_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.779 | 0 |
| tril_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.728 | 0 |
| triu_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.783 | 0 |
| triu_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.746 | 0 |
| take_along_axis_broadcast_indices_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.871 | 0 |
| take_along_axis_batched_indices_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.858 | 0 |
| take_along_axis_broadcast_indices_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.791 | 0 |
| take_along_axis_batched_indices_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.929 | 0 |
| scatter_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.115 | 0 |
| scatter_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.104 | 0 |
| cast_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.474 | 0 |
| cast_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.572 | 0 |
| pad_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.592 | 0 |
| pad_default_f64_n1048576 | jax-cuda-jit | 31 | 15 | 2.519 | 0 |
| slice_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.691 | 0 |
| slice_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.648 | 0 |
| reverse_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.641 | 0 |
| reverse_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.597 | 0 |
| concatenate_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.377 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda-jit | 16 | 15 | 1.834 | 0 |
| extract_diag_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.220 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.322 | 0 |
| embed_diag_default_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.393 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.413 | 0 |
| tril_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.437 | 0 |
| tril_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.277 | 0 |
| triu_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.370 | 0 |
| triu_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.602 | 0 |
| take_along_axis_broadcast_indices_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.757 | 0 |
| take_along_axis_batched_indices_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.539 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.057 | 0 |
| take_along_axis_batched_indices_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.893 | 0 |
| scatter_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.095 | 0 |
| scatter_default_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.200 | 0 |
| cast_default_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.699 | 0 |
| cast_default_f64_n1048576 | jax-cuda-jit | 32 | 15 | 2.697 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/structural
suite_file: benchmarks/gpu/structural.yaml
timestamp: '2026-10-13T10:15:29Z'
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
    ./scripts/run_gpu_public_api.sh structural
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 48
  selected_cases: 48
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
    data/results/nvidia-gpu/gpu/structural/20261013_101528/fixtures/manifest.json: b8881ec3ec33df4edad659aa32d13c3cc89286bcc67ec701527f999691786074
    target/release/benchmark_gpu_public_api: 91380afb8faffb97ca96d131ef98cc8e2656aababe594e82d65a07c9bb1bed38
  reference_numpy_version: 2.4.2
```
