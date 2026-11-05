# gpu/linalg_public_api_latency: CUDA public API benchmark

Measured cells: **308**. Unsupported cells: **102**, including **28** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

This run metadata records no device copy rate, so the records carry the logical bytes without a copy ratio.

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api_latency/20261105_020224/`.

Small-shape view of the same CUDA public linalg API surface: the 82 cases whose largest matrix dimension is at most 128. `small` is a shape class, not a measured classification: device kernel time is a small fraction of each measurement here, so these rows are per-call launch and synchronization overhead diagnostics rather than GPU throughput. They are never comparable with, or averaged together with, the large-shape rows of `gpu/linalg_public_api_throughput` (tenferro-rs #2087). Issue 155.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | 81.722 (0.797) | 119.866 (1.667) | 64.084 (0.627) | 73.227 (10.697) | 77.821 (0.519) |
| cholesky_f64_32x32 | 83.319 (0.571) | 123.029 (0.495) | 71.799 (0.278) | 75.903 (1.219) | 77.212 (0.418) |
| svdvals_f32_32x32 | 686.831 (3.658) | unsupported | 417.173 (1.747) | 704.899 (7.384) | 714.267 (8.597) |
| svdvals_f64_32x32 | 993.305 (3.254) | unsupported | 738.513 (1.336) | 1034.911 (16.888) | 1018.280 (24.313) |
| eigvalsh_f32_32x32 | 224.517 (0.898) | 261.731 (1.630) | 230.756 (0.648) | 575.725 (5.649) | 566.855 (4.973) |
| eigvalsh_f64_32x32 | 271.722 (1.312) | 304.781 (0.658) | 271.921 (1.153) | 717.107 (8.045) | 707.132 (5.276) |
| pinv_f32_32x32 | unsupported | unsupported | 521.613 (3.827) | 974.464 (13.913) | 705.210 (3.806) |
| pinv_f64_32x32 | unsupported | unsupported | 850.105 (3.982) | 1291.195 (27.949) | 1016.355 (12.066) |
| inv_f32_32x32 | unsupported | unsupported | 199.641 (3.042) | 120.868 (2.904) | 109.307 (0.558) |
| inv_f64_32x32 | unsupported | unsupported | 202.736 (1.358) | 125.813 (3.318) | 115.169 (0.206) |
| det_f32_32x32 | 370.465 (1.333) | 431.541 (4.789) | 113.874 (1.495) | 95.398 (1.880) | 83.998 (0.861) |
| det_f64_32x32 | 373.912 (1.121) | 433.965 (2.279) | 112.872 (1.005) | 96.452 (2.448) | 84.061 (0.335) |
| norm_f32_32x32 | 69.262 (0.339) | 93.096 (0.501) | 7.412 (0.129) | 51.143 (1.547) | 40.032 (0.279) |
| norm_f64_32x32 | 70.179 (0.132) | 96.661 (0.642) | 7.380 (0.021) | 51.623 (2.832) | 40.159 (0.397) |
| triangular_solve_f32_32x32 | 18.175 (0.021) | 73.881 (0.682) | 28.033 (1.125) | 85.025 (3.606) | 71.844 (0.398) |
| triangular_solve_f64_32x32 | 19.339 (0.019) | 74.900 (0.452) | 27.585 (0.235) | 87.412 (2.888) | 72.130 (0.428) |
| svd_full_square_f32_32x32 | 658.805 (3.839) | 713.420 (3.048) | 355.461 (1.573) | 689.146 (9.773) | 657.340 (2.769) |
| svd_full_tall_f32_64x32 | 1047.376 (6.458) | 1101.133 (5.770) | 1049.417 (3.628) | 1065.272 (11.860) | 1037.061 (12.868) |
| svd_full_wide_f32_32x64 | 1123.959 (7.690) | 1177.866 (6.508) | 1132.280 (4.075) | 1166.935 (19.598) | 1107.249 (5.217) |
| slogdet_positive_f32_32x32 | 334.837 (2.089) | 404.513 (2.817) | 135.322 (1.063) | 110.263 (3.747) | 91.298 (1.157) |
| slogdet_negative_f32_32x32 | 342.968 (2.036) | 393.050 (2.635) | 134.369 (0.400) | 112.968 (2.495) | 91.365 (0.346) |
| svd_full_square_f64_32x32 | 967.941 (5.666) | 999.515 (4.828) | 612.174 (1.201) | 953.966 (5.883) | 964.698 (140.150) |
| svd_full_tall_f64_64x32 | 1476.026 (3.990) | 1507.122 (3.940) | 1476.266 (10.380) | 1481.207 (11.410) | 1479.506 (17.863) |
| svd_full_wide_f64_32x64 | 1505.293 (8.601) | 1563.414 (16.780) | 1576.044 (3.472) | 1497.932 (7.143) | 1507.118 (19.881) |
| slogdet_positive_f64_32x32 | 347.272 (2.011) | 366.074 (10.299) | 137.422 (0.643) | 103.072 (1.113) | 91.697 (0.804) |
| slogdet_negative_f64_32x32 | 348.422 (6.605) | 383.975 (8.527) | 133.852 (0.830) | 103.491 (1.156) | 91.977 (0.397) |
| svd_full_square_c32_32x32 | 985.212 (6.709) | 998.385 (18.948) | 596.218 (1.229) | 975.437 (4.694) | 961.281 (4.249) |
| svd_full_tall_c32_64x32 | 1393.998 (4.450) | 1435.965 (5.112) | 1411.379 (5.718) | 1414.815 (18.498) | 1393.113 (14.693) |
| svd_full_wide_c32_32x64 | 1405.559 (2.097) | 1460.201 (3.533) | 1417.729 (4.913) | 1428.735 (3.315) | 1401.744 (19.503) |
| slogdet_positive_c32_32x32 | 344.373 (0.710) | 403.313 (1.129) | 139.594 (0.625) | 102.074 (0.719) | 92.239 (0.539) |
| slogdet_negative_c32_32x32 | 346.581 (1.012) | 402.861 (1.743) | 138.988 (0.607) | 102.650 (3.503) | 88.155 (2.801) |
| svd_full_square_c64_32x32 | 1807.333 (1.980) | 1851.174 (3.380) | 1240.213 (1.605) | 1835.769 (3.213) | 1826.949 (12.686) |
| svd_full_tall_c64_64x32 | 2262.819 (4.830) | 2321.370 (7.750) | 2286.089 (3.990) | 2397.494 (4.636) | 2319.631 (23.627) |
| svd_full_wide_c64_32x64 | 2400.983 (10.335) | 2473.996 (9.075) | 2423.464 (7.520) | 2490.417 (17.976) | 2468.316 (7.650) |
| slogdet_positive_c64_32x32 | 361.203 (10.712) | 421.422 (2.470) | 142.298 (0.607) | 106.585 (1.547) | 92.122 (1.757) |
| slogdet_negative_c64_32x32 | 366.726 (7.310) | 420.085 (2.222) | 144.148 (0.802) | 107.724 (1.568) | 94.433 (0.879) |
| householder_qr_square_f32_32x32 | 127.458 (1.326) | 160.070 (1.299) | 83.013 (0.019) | 85.128 (0.149) | 83.992 (0.108) |
| householder_qr_tall_f32_64x32 | 169.396 (1.995) | 211.240 (1.199) | 124.879 (0.111) | 130.681 (0.464) | 127.729 (0.157) |
| householder_qr_wide_f32_32x64 | 168.750 (2.303) | 207.918 (0.740) | 125.117 (0.098) | 130.735 (0.239) | 128.263 (0.201) |
| householder_qr_square_f64_32x32 | 139.474 (1.360) | 180.100 (0.739) | 99.209 (0.051) | 101.535 (0.193) | 100.583 (0.200) |
| householder_qr_tall_f64_64x32 | 181.815 (1.810) | 225.540 (1.679) | 138.822 (0.085) | 145.026 (0.119) | 142.813 (0.213) |
| householder_qr_wide_f64_32x64 | 186.166 (0.645) | 228.797 (1.738) | 139.921 (0.106) | 145.691 (0.207) | 143.501 (0.285) |
| householder_qr_square_c32_32x32 | 135.139 (0.845) | 182.786 (2.448) | 94.648 (0.042) | 100.732 (0.269) | 95.886 (0.329) |
| householder_qr_tall_c32_64x32 | 186.057 (1.709) | 236.684 (10.812) | 144.662 (0.083) | 149.315 (0.135) | 149.046 (0.279) |
| householder_qr_wide_c32_32x64 | 187.629 (1.355) | 232.247 (0.538) | 145.731 (0.093) | 152.825 (0.953) | 149.503 (0.259) |
| slogdet_phase_c32_32x32 | 345.142 (0.998) | 401.663 (4.587) | 139.332 (0.749) | 110.437 (2.810) | 98.424 (2.217) |
| householder_qr_square_c64_32x32 | 164.975 (0.578) | 205.642 (0.863) | 126.776 (0.046) | 132.048 (0.155) | 130.418 (0.106) |
| householder_qr_tall_c64_64x32 | 218.726 (0.594) | 261.611 (1.491) | 173.045 (0.054) | 178.606 (0.246) | 177.539 (0.216) |
| householder_qr_wide_c64_32x64 | 232.208 (0.841) | 270.267 (1.502) | 182.472 (0.066) | 189.419 (0.417) | 186.353 (0.105) |
| slogdet_phase_c64_32x32 | 368.320 (1.946) | 409.887 (1.667) | 141.505 (0.559) | 108.560 (2.169) | 99.067 (1.722) |
| householder_from_factors_default_f32_64x32 | 367.405 (0.743) | 422.988 (1.462) | unsupported | unsupported | unsupported |
| householder_append_default_f32_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f32_64x32 | 34.786 (0.608) | 45.951 (0.316) | 4.983 (0.008) | 52.756 (2.469) | 43.722 (0.378) |
| householder_q_columns_prefix_f32_64x32 | unsupported | unsupported | 64.011 (0.759) | 163.696 (74.671) | 109.985 (0.914) |
| householder_q_columns_complement_f32_64x32 | unsupported | unsupported | 246.982 (0.132) | unsupported | unsupported |
| householder_from_factors_default_f64_64x32 | 381.669 (1.303) | 450.569 (2.750) | unsupported | unsupported | unsupported |
| householder_append_default_f64_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f64_64x32 | 35.749 (0.117) | 48.961 (0.236) | 5.090 (0.019) | 52.861 (3.513) | 43.953 (0.201) |
| householder_q_columns_prefix_f64_64x32 | unsupported | unsupported | 67.079 (1.216) | 158.991 (17.715) | 117.447 (22.197) |
| householder_q_columns_complement_f64_64x32 | unsupported | unsupported | 276.876 (0.301) | unsupported | unsupported |
| householder_from_factors_default_c32_64x32 | 384.757 (2.132) | 449.112 (1.838) | unsupported | unsupported | unsupported |
| householder_append_default_c32_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c32_64x32 | 35.716 (0.225) | 49.642 (0.522) | 4.983 (0.018) | 59.497 (4.076) | 43.724 (0.252) |
| householder_q_columns_prefix_c32_64x32 | unsupported | unsupported | 60.619 (1.282) | 155.430 (12.143) | 108.719 (0.896) |
| householder_q_columns_complement_c32_64x32 | unsupported | unsupported | 263.639 (0.272) | unsupported | unsupported |
| householder_from_factors_default_c64_64x32 | 414.864 (3.356) | 489.836 (4.424) | unsupported | unsupported | unsupported |
| householder_append_default_c64_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c64_64x32 | 36.102 (0.138) | 51.423 (0.415) | 5.007 (0.012) | 57.983 (3.719) | 44.016 (0.644) |
| householder_q_columns_prefix_c64_64x32 | unsupported | unsupported | 64.788 (0.413) | 173.134 (7.409) | 108.182 (0.793) |
| householder_q_columns_complement_c64_64x32 | unsupported | unsupported | 286.365 (0.217) | unsupported | unsupported |
| lu_square_f32_s32 | 173.149 (0.739) | 211.640 (0.685) | unsupported | unsupported | unsupported |
| lu_square_f64_s32 | 175.054 (0.731) | 213.510 (2.752) | unsupported | unsupported | unsupported |
| lu_square_c32_s32 | 178.702 (1.300) | 217.973 (1.173) | unsupported | unsupported | unsupported |
| lu_square_c64_s32 | 185.241 (0.953) | 226.173 (0.777) | unsupported | unsupported | unsupported |
| lu_tall_f32_s32 | 176.171 (0.217) | 214.993 (1.056) | unsupported | unsupported | unsupported |
| lu_tall_f64_s32 | 181.996 (0.477) | 206.055 (5.425) | unsupported | unsupported | unsupported |
| lu_tall_c32_s32 | 184.753 (1.077) | 221.526 (2.152) | unsupported | unsupported | unsupported |
| lu_tall_c64_s32 | 186.142 (1.299) | 226.034 (0.503) | unsupported | unsupported | unsupported |
| lu_wide_f32_s32 | 180.679 (1.114) | 218.793 (1.964) | unsupported | unsupported | unsupported |
| lu_wide_f64_s32 | 186.320 (0.530) | 226.030 (0.968) | unsupported | unsupported | unsupported |
| lu_wide_c32_s32 | 192.202 (1.903) | 226.663 (1.522) | unsupported | unsupported | unsupported |
| lu_wide_c64_s32 | 198.862 (1.624) | 231.590 (1.531) | unsupported | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | jax-cuda | 32 | 15 | 2.343 | 8192 | unavailable | 0.0002352 |
| cholesky_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.490 | 8192 | unavailable | 0.0002352 |
| cholesky_f32_32x32 | pytorch-cuda | 32 | 15 | 2.051 | 8192 | unavailable | 0.0002352 |
| cholesky_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.615 | 8192 | unavailable | 0.0002352 |
| cholesky_f32_32x32 | tenferro-cuda-eager | 32 | 15 | 3.836 | 8192 | unavailable | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda | 32 | 15 | 2.429 | 16384 | unavailable | 2.188e-07 |
| cholesky_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.471 | 16384 | unavailable | 2.188e-07 |
| cholesky_f64_32x32 | pytorch-cuda | 32 | 15 | 2.298 | 16384 | unavailable | 2.188e-07 |
| cholesky_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.666 | 16384 | unavailable | 2.188e-07 |
| cholesky_f64_32x32 | tenferro-cuda-eager | 32 | 15 | 3.937 | 16384 | unavailable | 2.188e-07 |
| det_f32_32x32 | jax-cuda | 32 | 15 | 3.053 | 4100 | unavailable | 0.002641 |
| det_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.688 | 4100 | unavailable | 0.002641 |
| det_f32_32x32 | pytorch-cuda | 32 | 15 | 3.644 | 4100 | unavailable | 0.002641 |
| det_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.964 | 4100 | unavailable | 0.002641 |
| det_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.452 | 4100 | unavailable | 0.002641 |
| det_f64_32x32 | jax-cuda | 32 | 15 | 3.086 | 8200 | unavailable | 1.135e-06 |
| det_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.690 | 8200 | unavailable | 1.135e-06 |
| det_f64_32x32 | pytorch-cuda | 32 | 15 | 3.612 | 8200 | unavailable | 3.784e-07 |
| det_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.991 | 8200 | unavailable | 1.135e-06 |
| det_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.472 | 8200 | unavailable | 1.135e-06 |
| eigvalsh_f32_32x32 | jax-cuda | 4 | 15 | 2.303 | 4224 | unavailable | 0.004775 |
| eigvalsh_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.267 | 4224 | unavailable | 0.004775 |
| eigvalsh_f32_32x32 | pytorch-cuda | 16 | 15 | 3.692 | 4224 | unavailable | 0.001073 |
| eigvalsh_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.592 | 4224 | unavailable | 0.001073 |
| eigvalsh_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.094 | 4224 | unavailable | 0.001073 |
| eigvalsh_f64_32x32 | jax-cuda | 4 | 15 | 2.868 | 8448 | unavailable | 5.542e-06 |
| eigvalsh_f64_32x32 | jax-cuda-jit | 4 | 15 | 2.829 | 8448 | unavailable | 5.542e-06 |
| eigvalsh_f64_32x32 | pytorch-cuda | 8 | 15 | 2.175 | 8448 | unavailable | 5.433e-07 |
| eigvalsh_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.174 | 8448 | unavailable | 5.433e-07 |
| eigvalsh_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.438 | 8448 | unavailable | 5.433e-07 |
| householder_append_default_c32_64x32 | jax-cuda | — | — | — | 49408 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | jax-cuda-jit | — | — | — | 49408 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | pytorch-cuda | — | — | — | 49408 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | tenferro-cuda-direct | — | — | — | 49408 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_c32_64x32 | tenferro-cuda-eager | — | — | — | 49408 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_c64_64x32 | jax-cuda | — | — | — | 98816 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | jax-cuda-jit | — | — | — | 98816 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | pytorch-cuda | — | — | — | 98816 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | tenferro-cuda-direct | — | — | — | 98816 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_c64_64x32 | tenferro-cuda-eager | — | — | — | 98816 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_f32_64x32 | jax-cuda | — | — | — | 24704 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | jax-cuda-jit | — | — | — | 24704 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | pytorch-cuda | — | — | — | 24704 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | tenferro-cuda-direct | — | — | — | 24704 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_f32_64x32 | tenferro-cuda-eager | — | — | — | 24704 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_f64_64x32 | jax-cuda | — | — | — | 49408 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | jax-cuda-jit | — | — | — | 49408 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | pytorch-cuda | — | — | — | 49408 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | tenferro-cuda-direct | — | — | — | 49408 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_f64_64x32 | tenferro-cuda-eager | — | — | — | 49408 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_64x32 | jax-cuda | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c32_64x32 | jax-cuda-jit | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c32_64x32 | pytorch-cuda | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-direct | 8 | 15 | 3.078 | 57600 | unavailable | 0.0003232 |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-eager | 8 | 15 | 3.593 | 57600 | unavailable | 0.0003232 |
| householder_from_factors_default_c64_64x32 | jax-cuda | — | — | — | 115200 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_64x32 | jax-cuda-jit | — | — | — | 115200 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_64x32 | pytorch-cuda | — | — | — | 115200 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-direct | 8 | 15 | 3.319 | 115200 | unavailable | 4.441e-07 |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-eager | 8 | 15 | 3.919 | 115200 | unavailable | 4.441e-07 |
| householder_from_factors_default_f32_64x32 | jax-cuda | — | — | — | 28800 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_64x32 | jax-cuda-jit | — | — | — | 28800 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_64x32 | pytorch-cuda | — | — | — | 28800 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-direct | 8 | 15 | 2.939 | 28800 | unavailable | 0.0004571 |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-eager | 8 | 15 | 3.384 | 28800 | unavailable | 0.0004571 |
| householder_from_factors_default_f64_64x32 | jax-cuda | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_64x32 | jax-cuda-jit | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_64x32 | pytorch-cuda | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-direct | 8 | 15 | 3.053 | 57600 | unavailable | 3.886e-07 |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-eager | 8 | 15 | 3.605 | 57600 | unavailable | 3.886e-07 |
| householder_q_columns_complement_c32_64x32 | jax-cuda | — | — | — | 32768 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c32_64x32 | jax-cuda-jit | — | — | — | 32768 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c32_64x32 | pytorch-cuda | 8 | 15 | 2.109 | 32768 | unavailable | 0.000299 |
| householder_q_columns_complement_c32_64x32 | tenferro-cuda-direct | — | — | — | 32768 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_64x32 | tenferro-cuda-eager | — | — | — | 32768 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_64x32 | jax-cuda | — | — | — | 65536 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c64_64x32 | jax-cuda-jit | — | — | — | 65536 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c64_64x32 | pytorch-cuda | 8 | 15 | 2.291 | 65536 | unavailable | 3.341e-07 |
| householder_q_columns_complement_c64_64x32 | tenferro-cuda-direct | — | — | — | 65536 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_64x32 | tenferro-cuda-eager | — | — | — | 65536 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_64x32 | jax-cuda | — | — | — | 16384 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f32_64x32 | jax-cuda-jit | — | — | — | 16384 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f32_64x32 | pytorch-cuda | 16 | 15 | 3.952 | 16384 | unavailable | 0.0001791 |
| householder_q_columns_complement_f32_64x32 | tenferro-cuda-direct | — | — | — | 16384 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_64x32 | tenferro-cuda-eager | — | — | — | 16384 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_64x32 | jax-cuda | — | — | — | 32768 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f64_64x32 | jax-cuda-jit | — | — | — | 32768 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f64_64x32 | pytorch-cuda | 8 | 15 | 2.215 | 32768 | unavailable | 2.724e-08 |
| householder_q_columns_complement_f64_64x32 | tenferro-cuda-direct | — | — | — | 32768 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_64x32 | tenferro-cuda-eager | — | — | — | 32768 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c32_64x32 | jax-cuda | 16 | 15 | 2.487 | 20480 | unavailable | 0.00012 |
| householder_q_columns_prefix_c32_64x32 | jax-cuda-jit | 32 | 15 | 3.479 | 20480 | unavailable | 0.00012 |
| householder_q_columns_prefix_c32_64x32 | pytorch-cuda | 64 | 15 | 3.880 | 20480 | unavailable | 0.00012 |
| householder_q_columns_prefix_c32_64x32 | tenferro-cuda-direct | — | — | — | 20480 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c32_64x32 | tenferro-cuda-eager | — | — | — | 20480 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c64_64x32 | jax-cuda | 16 | 15 | 2.770 | 40960 | unavailable | 5.592e-08 |
| householder_q_columns_prefix_c64_64x32 | jax-cuda-jit | 32 | 15 | 3.462 | 40960 | unavailable | 5.592e-08 |
| householder_q_columns_prefix_c64_64x32 | pytorch-cuda | 32 | 15 | 2.073 | 40960 | unavailable | 5.592e-08 |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-direct | — | — | — | 40960 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-eager | — | — | — | 40960 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f32_64x32 | jax-cuda | 8 | 15 | 1.310 | 10240 | unavailable | 0.0001196 |
| householder_q_columns_prefix_f32_64x32 | jax-cuda-jit | 32 | 15 | 3.520 | 10240 | unavailable | 0.0001196 |
| householder_q_columns_prefix_f32_64x32 | pytorch-cuda | 32 | 15 | 2.048 | 10240 | unavailable | 0.0001196 |
| householder_q_columns_prefix_f32_64x32 | tenferro-cuda-direct | — | — | — | 10240 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f32_64x32 | tenferro-cuda-eager | — | — | — | 10240 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f64_64x32 | jax-cuda | 16 | 15 | 2.544 | 20480 | unavailable | 1.114e-07 |
| householder_q_columns_prefix_f64_64x32 | jax-cuda-jit | 16 | 15 | 1.879 | 20480 | unavailable | 1.114e-07 |
| householder_q_columns_prefix_f64_64x32 | pytorch-cuda | 32 | 15 | 2.147 | 20480 | unavailable | 1.114e-07 |
| householder_q_columns_prefix_f64_64x32 | tenferro-cuda-direct | — | — | — | 20480 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f64_64x32 | tenferro-cuda-eager | — | — | — | 20480 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_qr_square_c32_32x32 | jax-cuda | 16 | 15 | 1.612 | 16640 | unavailable | 0.0005833 |
| householder_qr_square_c32_32x32 | jax-cuda-jit | 32 | 15 | 3.068 | 16640 | unavailable | 0.0005833 |
| householder_qr_square_c32_32x32 | pytorch-cuda | 32 | 15 | 3.029 | 16640 | unavailable | 0.0005833 |
| householder_qr_square_c32_32x32 | tenferro-cuda-direct | 16 | 15 | 2.162 | 16640 | unavailable | 0.0005631 |
| householder_qr_square_c32_32x32 | tenferro-cuda-eager | 16 | 15 | 2.925 | 16640 | unavailable | 0.0005631 |
| householder_qr_square_c64_32x32 | jax-cuda | 16 | 15 | 2.113 | 33280 | unavailable | 1.221e-06 |
| householder_qr_square_c64_32x32 | jax-cuda-jit | 16 | 15 | 2.087 | 33280 | unavailable | 1.221e-06 |
| householder_qr_square_c64_32x32 | pytorch-cuda | 16 | 15 | 2.028 | 33280 | unavailable | 1.221e-06 |
| householder_qr_square_c64_32x32 | tenferro-cuda-direct | 16 | 15 | 2.640 | 33280 | unavailable | 4.996e-07 |
| householder_qr_square_c64_32x32 | tenferro-cuda-eager | 16 | 15 | 3.290 | 33280 | unavailable | 4.996e-07 |
| householder_qr_square_f32_32x32 | jax-cuda | 32 | 15 | 2.724 | 8320 | unavailable | 0.000367 |
| householder_qr_square_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.688 | 8320 | unavailable | 0.000367 |
| householder_qr_square_f32_32x32 | pytorch-cuda | 32 | 15 | 2.656 | 8320 | unavailable | 0.000367 |
| householder_qr_square_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 2.039 | 8320 | unavailable | 0.0004212 |
| householder_qr_square_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 2.561 | 8320 | unavailable | 0.0004212 |
| householder_qr_square_f64_32x32 | jax-cuda | 32 | 15 | 3.249 | 16640 | unavailable | 6.106e-07 |
| householder_qr_square_f64_32x32 | jax-cuda-jit | 32 | 15 | 3.219 | 16640 | unavailable | 6.106e-07 |
| householder_qr_square_f64_32x32 | pytorch-cuda | 32 | 15 | 3.175 | 16640 | unavailable | 6.106e-07 |
| householder_qr_square_f64_32x32 | tenferro-cuda-direct | 16 | 15 | 2.232 | 16640 | unavailable | 3.886e-07 |
| householder_qr_square_f64_32x32 | tenferro-cuda-eager | 16 | 15 | 2.882 | 16640 | unavailable | 3.886e-07 |
| householder_qr_tall_c32_64x32 | jax-cuda | 16 | 15 | 2.389 | 33024 | unavailable | 0.0003756 |
| householder_qr_tall_c32_64x32 | jax-cuda-jit | 16 | 15 | 2.385 | 33024 | unavailable | 0.0003756 |
| householder_qr_tall_c32_64x32 | pytorch-cuda | 16 | 15 | 2.315 | 33024 | unavailable | 0.0003756 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-direct | 16 | 15 | 2.977 | 33024 | unavailable | 0.0003724 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-eager | 16 | 15 | 3.787 | 33024 | unavailable | 0.0003724 |
| householder_qr_tall_c64_64x32 | jax-cuda | 16 | 15 | 2.858 | 66048 | unavailable | 9.992e-07 |
| householder_qr_tall_c64_64x32 | jax-cuda-jit | 16 | 15 | 2.841 | 66048 | unavailable | 9.992e-07 |
| householder_qr_tall_c64_64x32 | pytorch-cuda | 16 | 15 | 2.769 | 66048 | unavailable | 9.992e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-direct | 16 | 15 | 3.500 | 66048 | unavailable | 5.551e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-eager | 8 | 15 | 2.093 | 66048 | unavailable | 5.551e-07 |
| householder_qr_tall_f32_64x32 | jax-cuda | 16 | 15 | 2.091 | 16512 | unavailable | 0.000327 |
| householder_qr_tall_f32_64x32 | jax-cuda-jit | 16 | 15 | 2.044 | 16512 | unavailable | 0.000327 |
| householder_qr_tall_f32_64x32 | pytorch-cuda | 16 | 15 | 1.998 | 16512 | unavailable | 0.000327 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-direct | 16 | 15 | 2.710 | 16512 | unavailable | 0.0003159 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-eager | 16 | 15 | 3.380 | 16512 | unavailable | 0.0003159 |
| householder_qr_tall_f64_64x32 | jax-cuda | 16 | 15 | 2.320 | 33024 | unavailable | 6.661e-07 |
| householder_qr_tall_f64_64x32 | jax-cuda-jit | 16 | 15 | 2.285 | 33024 | unavailable | 6.661e-07 |
| householder_qr_tall_f64_64x32 | pytorch-cuda | 16 | 15 | 2.221 | 33024 | unavailable | 6.661e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-direct | 16 | 15 | 2.909 | 33024 | unavailable | 4.441e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-eager | 16 | 15 | 3.609 | 33024 | unavailable | 4.441e-07 |
| householder_qr_wide_c32_32x64 | jax-cuda | 16 | 15 | 2.445 | 33024 | unavailable | 0.0004972 |
| householder_qr_wide_c32_32x64 | jax-cuda-jit | 16 | 15 | 2.392 | 33024 | unavailable | 0.0004972 |
| householder_qr_wide_c32_32x64 | pytorch-cuda | 16 | 15 | 2.332 | 33024 | unavailable | 0.0004972 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-direct | 16 | 15 | 3.002 | 33024 | unavailable | 0.0006002 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-eager | 16 | 15 | 3.716 | 33024 | unavailable | 0.0006002 |
| householder_qr_wide_c64_32x64 | jax-cuda | 16 | 15 | 3.031 | 66048 | unavailable | 1.166e-06 |
| householder_qr_wide_c64_32x64 | jax-cuda-jit | 16 | 15 | 2.982 | 66048 | unavailable | 1.166e-06 |
| householder_qr_wide_c64_32x64 | pytorch-cuda | 16 | 15 | 2.920 | 66048 | unavailable | 1.166e-06 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-direct | 8 | 15 | 1.858 | 66048 | unavailable | 3.886e-07 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-eager | 8 | 15 | 2.162 | 66048 | unavailable | 3.886e-07 |
| householder_qr_wide_f32_32x64 | jax-cuda | 16 | 15 | 2.092 | 16512 | unavailable | 0.0003745 |
| householder_qr_wide_f32_32x64 | jax-cuda-jit | 16 | 15 | 2.052 | 16512 | unavailable | 0.0003745 |
| householder_qr_wide_f32_32x64 | pytorch-cuda | 16 | 15 | 2.002 | 16512 | unavailable | 0.0003745 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-direct | 16 | 15 | 2.700 | 16512 | unavailable | 0.0004731 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-eager | 16 | 15 | 3.327 | 16512 | unavailable | 0.0004731 |
| householder_qr_wide_f64_32x64 | jax-cuda | 16 | 15 | 2.331 | 33024 | unavailable | 5.551e-07 |
| householder_qr_wide_f64_32x64 | jax-cuda-jit | 16 | 15 | 2.296 | 33024 | unavailable | 5.551e-07 |
| householder_qr_wide_f64_32x64 | pytorch-cuda | 16 | 15 | 2.239 | 33024 | unavailable | 5.551e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-direct | 16 | 15 | 2.979 | 33024 | unavailable | 4.441e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-eager | 16 | 15 | 3.661 | 33024 | unavailable | 4.441e-07 |
| householder_r_default_c32_64x32 | jax-cuda | 32 | 15 | 1.904 | 24576 | unavailable | 0.0004713 |
| householder_r_default_c32_64x32 | jax-cuda-jit | 64 | 15 | 2.798 | 24576 | unavailable | 0.0004713 |
| householder_r_default_c32_64x32 | pytorch-cuda | 512 | 15 | 2.551 | 24576 | unavailable | 0.0004713 |
| householder_r_default_c32_64x32 | tenferro-cuda-direct | 64 | 15 | 2.286 | 24576 | unavailable | 0.0004713 |
| householder_r_default_c32_64x32 | tenferro-cuda-eager | 64 | 15 | 3.177 | 24576 | unavailable | 0.0004713 |
| householder_r_default_c64_64x32 | jax-cuda | 64 | 15 | 3.711 | 49152 | unavailable | 3.296e-07 |
| householder_r_default_c64_64x32 | jax-cuda-jit | 64 | 15 | 2.817 | 49152 | unavailable | 3.296e-07 |
| householder_r_default_c64_64x32 | pytorch-cuda | 512 | 15 | 2.564 | 49152 | unavailable | 3.296e-07 |
| householder_r_default_c64_64x32 | tenferro-cuda-direct | 64 | 15 | 2.311 | 49152 | unavailable | 3.296e-07 |
| householder_r_default_c64_64x32 | tenferro-cuda-eager | 64 | 15 | 3.291 | 49152 | unavailable | 3.296e-07 |
| householder_r_default_f32_64x32 | jax-cuda | 64 | 15 | 3.376 | 12288 | unavailable | 0.0002373 |
| householder_r_default_f32_64x32 | jax-cuda-jit | 64 | 15 | 2.798 | 12288 | unavailable | 0.0002373 |
| householder_r_default_f32_64x32 | pytorch-cuda | 512 | 15 | 2.551 | 12288 | unavailable | 0.0002373 |
| householder_r_default_f32_64x32 | tenferro-cuda-direct | 64 | 15 | 2.226 | 12288 | unavailable | 0.0002373 |
| householder_r_default_f32_64x32 | tenferro-cuda-eager | 64 | 15 | 2.941 | 12288 | unavailable | 0.0002373 |
| householder_r_default_f64_64x32 | jax-cuda | 64 | 15 | 3.383 | 24576 | unavailable | 2.2e-07 |
| householder_r_default_f64_64x32 | jax-cuda-jit | 64 | 15 | 2.813 | 24576 | unavailable | 2.2e-07 |
| householder_r_default_f64_64x32 | pytorch-cuda | 512 | 15 | 2.606 | 24576 | unavailable | 2.2e-07 |
| householder_r_default_f64_64x32 | tenferro-cuda-direct | 64 | 15 | 2.288 | 24576 | unavailable | 2.2e-07 |
| householder_r_default_f64_64x32 | tenferro-cuda-eager | 64 | 15 | 3.133 | 24576 | unavailable | 2.2e-07 |
| inv_f32_32x32 | jax-cuda | 32 | 15 | 3.868 | 8192 | unavailable | 0.0004905 |
| inv_f32_32x32 | jax-cuda-jit | 32 | 15 | 3.498 | 8192 | unavailable | 0.0004905 |
| inv_f32_32x32 | pytorch-cuda | 16 | 15 | 3.194 | 8192 | unavailable | 0.0004905 |
| inv_f32_32x32 | tenferro-cuda-direct | — | — | — | 8192 | unavailable | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_32x32 | tenferro-cuda-eager | — | — | — | 8192 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_32x32 | jax-cuda | 16 | 15 | 2.013 | 16384 | unavailable | 2.284e-07 |
| inv_f64_32x32 | jax-cuda-jit | 32 | 15 | 3.685 | 16384 | unavailable | 2.284e-07 |
| inv_f64_32x32 | pytorch-cuda | 16 | 15 | 3.244 | 16384 | unavailable | 2.284e-07 |
| inv_f64_32x32 | tenferro-cuda-direct | — | — | — | 16384 | unavailable | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_32x32 | tenferro-cuda-eager | — | — | — | 16384 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| lu_square_c32_s32 | jax-cuda | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s32 | jax-cuda-jit | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s32 | pytorch-cuda | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.859 | 32776 | unavailable | 0.0002381 |
| lu_square_c32_s32 | tenferro-cuda-eager | 16 | 15 | 3.488 | 32776 | unavailable | 0.0002381 |
| lu_square_c64_s32 | jax-cuda | — | — | — | 65552 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s32 | jax-cuda-jit | — | — | — | 65552 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s32 | pytorch-cuda | — | — | — | 65552 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.964 | 65552 | unavailable | 5.532e-07 |
| lu_square_c64_s32 | tenferro-cuda-eager | 16 | 15 | 3.619 | 65552 | unavailable | 5.532e-07 |
| lu_square_f32_s32 | jax-cuda | — | — | — | 16388 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s32 | jax-cuda-jit | — | — | — | 16388 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s32 | pytorch-cuda | — | — | — | 16388 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.770 | 16388 | unavailable | 0.0003557 |
| lu_square_f32_s32 | tenferro-cuda-eager | 16 | 15 | 3.386 | 16388 | unavailable | 0.0003557 |
| lu_square_f64_s32 | jax-cuda | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s32 | jax-cuda-jit | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s32 | pytorch-cuda | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.801 | 32776 | unavailable | 4.397e-07 |
| lu_square_f64_s32 | tenferro-cuda-eager | 16 | 15 | 3.416 | 32776 | unavailable | 4.397e-07 |
| lu_tall_c32_s32 | jax-cuda | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s32 | jax-cuda-jit | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s32 | pytorch-cuda | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.956 | 73736 | unavailable | 0.0005956 |
| lu_tall_c32_s32 | tenferro-cuda-eager | 16 | 15 | 3.544 | 73736 | unavailable | 0.0005956 |
| lu_tall_c64_s32 | jax-cuda | — | — | — | 147472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s32 | jax-cuda-jit | — | — | — | 147472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s32 | pytorch-cuda | — | — | — | 147472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.978 | 147472 | unavailable | 4.406e-07 |
| lu_tall_c64_s32 | tenferro-cuda-eager | 16 | 15 | 3.617 | 147472 | unavailable | 4.406e-07 |
| lu_tall_f32_s32 | jax-cuda | — | — | — | 36868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s32 | jax-cuda-jit | — | — | — | 36868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s32 | pytorch-cuda | — | — | — | 36868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.819 | 36868 | unavailable | 0.0003557 |
| lu_tall_f32_s32 | tenferro-cuda-eager | 16 | 15 | 3.440 | 36868 | unavailable | 0.0003557 |
| lu_tall_f64_s32 | jax-cuda | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s32 | jax-cuda-jit | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s32 | pytorch-cuda | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.912 | 73736 | unavailable | 4.397e-07 |
| lu_tall_f64_s32 | tenferro-cuda-eager | 16 | 15 | 3.297 | 73736 | unavailable | 4.397e-07 |
| lu_wide_c32_s32 | jax-cuda | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s32 | jax-cuda-jit | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s32 | pytorch-cuda | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s32 | tenferro-cuda-direct | 16 | 15 | 3.075 | 49160 | unavailable | 0.0003567 |
| lu_wide_c32_s32 | tenferro-cuda-eager | 16 | 15 | 3.627 | 49160 | unavailable | 0.0003567 |
| lu_wide_c64_s32 | jax-cuda | — | — | — | 98320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s32 | jax-cuda-jit | — | — | — | 98320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s32 | pytorch-cuda | — | — | — | 98320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s32 | tenferro-cuda-direct | 16 | 15 | 3.182 | 98320 | unavailable | 3.323e-07 |
| lu_wide_c64_s32 | tenferro-cuda-eager | 16 | 15 | 3.705 | 98320 | unavailable | 3.323e-07 |
| lu_wide_f32_s32 | jax-cuda | — | — | — | 24580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s32 | jax-cuda-jit | — | — | — | 24580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s32 | pytorch-cuda | — | — | — | 24580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.891 | 24580 | unavailable | 0.00024 |
| lu_wide_f32_s32 | tenferro-cuda-eager | 16 | 15 | 3.501 | 24580 | unavailable | 0.00024 |
| lu_wide_f64_s32 | jax-cuda | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s32 | jax-cuda-jit | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s32 | pytorch-cuda | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.981 | 49160 | unavailable | 2.214e-07 |
| lu_wide_f64_s32 | tenferro-cuda-eager | 16 | 15 | 3.616 | 49160 | unavailable | 2.214e-07 |
| norm_f32_32x32 | jax-cuda | 64 | 15 | 3.273 | 4100 | unavailable | 0 |
| norm_f32_32x32 | jax-cuda-jit | 64 | 15 | 2.562 | 4100 | unavailable | 0 |
| norm_f32_32x32 | pytorch-cuda | 256 | 15 | 1.897 | 4100 | unavailable | 0 |
| norm_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.216 | 4100 | unavailable | 0 |
| norm_f32_32x32 | tenferro-cuda-eager | 32 | 15 | 2.979 | 4100 | unavailable | 0 |
| norm_f64_32x32 | jax-cuda | 64 | 15 | 3.304 | 8200 | unavailable | 0 |
| norm_f64_32x32 | jax-cuda-jit | 64 | 15 | 2.570 | 8200 | unavailable | 0 |
| norm_f64_32x32 | pytorch-cuda | 512 | 15 | 3.778 | 8200 | unavailable | 0 |
| norm_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.246 | 8200 | unavailable | 0 |
| norm_f64_32x32 | tenferro-cuda-eager | 32 | 15 | 3.093 | 8200 | unavailable | 0 |
| pinv_f32_32x32 | jax-cuda | 4 | 15 | 3.898 | 8200 | unavailable | 0.0006436 |
| pinv_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.821 | 8200 | unavailable | 0.0006436 |
| pinv_f32_32x32 | pytorch-cuda | 4 | 15 | 2.086 | 8200 | unavailable | 0.001025 |
| pinv_f32_32x32 | tenferro-cuda-direct | — | — | — | 8200 | unavailable | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f32_32x32 | tenferro-cuda-eager | — | — | — | 8200 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_32x32 | jax-cuda | 2 | 15 | 2.582 | 16400 | unavailable | 1.367e-06 |
| pinv_f64_32x32 | jax-cuda-jit | 2 | 15 | 2.033 | 16400 | unavailable | 1.367e-06 |
| pinv_f64_32x32 | pytorch-cuda | 4 | 15 | 3.400 | 16400 | unavailable | 9.374e-07 |
| pinv_f64_32x32 | tenferro-cuda-direct | — | — | — | 16400 | unavailable | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_32x32 | tenferro-cuda-eager | — | — | — | 16400 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| slogdet_negative_c32_32x32 | jax-cuda | 32 | 15 | 3.285 | 8204 | unavailable | 0.0004085 |
| slogdet_negative_c32_32x32 | jax-cuda-jit | 32 | 15 | 2.821 | 8204 | unavailable | 0.0004085 |
| slogdet_negative_c32_32x32 | pytorch-cuda | 16 | 15 | 2.224 | 8204 | unavailable | 0.0004085 |
| slogdet_negative_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.773 | 8204 | unavailable | 0.0007148 |
| slogdet_negative_c32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.223 | 8204 | unavailable | 0.0007148 |
| slogdet_negative_c64_32x32 | jax-cuda | 32 | 15 | 3.447 | 16408 | unavailable | 4.755e-07 |
| slogdet_negative_c64_32x32 | jax-cuda-jit | 32 | 15 | 3.022 | 16408 | unavailable | 4.755e-07 |
| slogdet_negative_c64_32x32 | pytorch-cuda | 16 | 15 | 2.306 | 16408 | unavailable | 4.755e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.934 | 16408 | unavailable | 3.804e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.361 | 16408 | unavailable | 3.804e-07 |
| slogdet_negative_f32_32x32 | jax-cuda | 32 | 15 | 3.615 | 4104 | unavailable | 0.000915 |
| slogdet_negative_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.924 | 4104 | unavailable | 0.000915 |
| slogdet_negative_f32_32x32 | pytorch-cuda | 16 | 15 | 2.150 | 4104 | unavailable | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.744 | 4104 | unavailable | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.144 | 4104 | unavailable | 0.000915 |
| slogdet_negative_f64_32x32 | jax-cuda | 32 | 15 | 3.312 | 8208 | unavailable | 4.261e-07 |
| slogdet_negative_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.943 | 8208 | unavailable | 4.261e-07 |
| slogdet_negative_f64_32x32 | pytorch-cuda | 16 | 15 | 2.142 | 8208 | unavailable | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.787 | 8208 | unavailable | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.072 | 8208 | unavailable | 4.261e-07 |
| slogdet_phase_c32_32x32 | jax-cuda | 32 | 15 | 3.534 | 8204 | unavailable | 0.0004085 |
| slogdet_phase_c32_32x32 | jax-cuda-jit | 32 | 15 | 3.150 | 8204 | unavailable | 0.0004085 |
| slogdet_phase_c32_32x32 | pytorch-cuda | 16 | 15 | 2.229 | 8204 | unavailable | 0.0004085 |
| slogdet_phase_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.761 | 8204 | unavailable | 0.0004085 |
| slogdet_phase_c32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.213 | 8204 | unavailable | 0.0004085 |
| slogdet_phase_c64_32x32 | jax-cuda | 32 | 15 | 3.474 | 16408 | unavailable | 5.706e-07 |
| slogdet_phase_c64_32x32 | jax-cuda-jit | 32 | 15 | 3.170 | 16408 | unavailable | 5.706e-07 |
| slogdet_phase_c64_32x32 | pytorch-cuda | 16 | 15 | 2.264 | 16408 | unavailable | 5.706e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.947 | 16408 | unavailable | 4.755e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.279 | 16408 | unavailable | 4.755e-07 |
| slogdet_positive_c32_32x32 | jax-cuda | 32 | 15 | 3.266 | 8204 | unavailable | 0.0004085 |
| slogdet_positive_c32_32x32 | jax-cuda-jit | 32 | 15 | 2.952 | 8204 | unavailable | 0.0004085 |
| slogdet_positive_c32_32x32 | pytorch-cuda | 16 | 15 | 2.233 | 8204 | unavailable | 0.0004085 |
| slogdet_positive_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.755 | 8204 | unavailable | 0.0007148 |
| slogdet_positive_c32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.227 | 8204 | unavailable | 0.0007148 |
| slogdet_positive_c64_32x32 | jax-cuda | 32 | 15 | 3.411 | 16408 | unavailable | 4.755e-07 |
| slogdet_positive_c64_32x32 | jax-cuda-jit | 32 | 15 | 2.948 | 16408 | unavailable | 4.755e-07 |
| slogdet_positive_c64_32x32 | pytorch-cuda | 16 | 15 | 2.277 | 16408 | unavailable | 4.755e-07 |
| slogdet_positive_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.890 | 16408 | unavailable | 3.804e-07 |
| slogdet_positive_c64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.371 | 16408 | unavailable | 3.804e-07 |
| slogdet_positive_f32_32x32 | jax-cuda | 32 | 15 | 3.528 | 4104 | unavailable | 0.000915 |
| slogdet_positive_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.922 | 4104 | unavailable | 0.000915 |
| slogdet_positive_f32_32x32 | pytorch-cuda | 16 | 15 | 2.165 | 4104 | unavailable | 0.000915 |
| slogdet_positive_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.679 | 4104 | unavailable | 0.000915 |
| slogdet_positive_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.236 | 4104 | unavailable | 0.000915 |
| slogdet_positive_f64_32x32 | jax-cuda | 32 | 15 | 3.298 | 8208 | unavailable | 4.261e-07 |
| slogdet_positive_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.934 | 8208 | unavailable | 4.261e-07 |
| slogdet_positive_f64_32x32 | pytorch-cuda | 16 | 15 | 2.199 | 8208 | unavailable | 4.261e-07 |
| slogdet_positive_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.778 | 8208 | unavailable | 4.261e-07 |
| slogdet_positive_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.929 | 8208 | unavailable | 4.261e-07 |
| svd_full_square_c32_32x32 | jax-cuda | 4 | 15 | 3.902 | 24704 | unavailable | 0.009079 |
| svd_full_square_c32_32x32 | jax-cuda-jit | 4 | 15 | 3.845 | 24704 | unavailable | 0.009079 |
| svd_full_square_c32_32x32 | pytorch-cuda | 4 | 15 | 2.385 | 24704 | unavailable | 0.011 |
| svd_full_square_c32_32x32 | tenferro-cuda-direct | 4 | 15 | 3.941 | 24704 | unavailable | 0.009079 |
| svd_full_square_c32_32x32 | tenferro-cuda-eager | 2 | 15 | 1.997 | 24704 | unavailable | 0.009079 |
| svd_full_square_c64_32x32 | jax-cuda | 2 | 15 | 3.672 | 49408 | unavailable | 9.215e-06 |
| svd_full_square_c64_32x32 | jax-cuda-jit | 2 | 15 | 3.654 | 49408 | unavailable | 9.215e-06 |
| svd_full_square_c64_32x32 | pytorch-cuda | 2 | 15 | 2.480 | 49408 | unavailable | 9.992e-06 |
| svd_full_square_c64_32x32 | tenferro-cuda-direct | 2 | 15 | 3.615 | 49408 | unavailable | 9.215e-06 |
| svd_full_square_c64_32x32 | tenferro-cuda-eager | 2 | 15 | 3.702 | 49408 | unavailable | 9.215e-06 |
| svd_full_square_f32_32x32 | jax-cuda | 4 | 15 | 2.757 | 12416 | unavailable | 0.008731 |
| svd_full_square_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.629 | 12416 | unavailable | 0.008731 |
| svd_full_square_f32_32x32 | pytorch-cuda | 8 | 15 | 2.844 | 12416 | unavailable | 0.005571 |
| svd_full_square_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.635 | 12416 | unavailable | 0.008731 |
| svd_full_square_f32_32x32 | tenferro-cuda-eager | 4 | 15 | 2.854 | 12416 | unavailable | 0.008731 |
| svd_full_square_f64_32x32 | jax-cuda | 4 | 15 | 3.816 | 24832 | unavailable | 1.008e-05 |
| svd_full_square_f64_32x32 | jax-cuda-jit | 2 | 15 | 1.929 | 24832 | unavailable | 1.008e-05 |
| svd_full_square_f64_32x32 | pytorch-cuda | 4 | 15 | 2.449 | 24832 | unavailable | 7.216e-06 |
| svd_full_square_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 3.872 | 24832 | unavailable | 1.007e-05 |
| svd_full_square_f64_32x32 | tenferro-cuda-eager | 2 | 15 | 1.999 | 24832 | unavailable | 1.007e-05 |
| svd_full_tall_c32_64x32 | jax-cuda | 2 | 15 | 2.830 | 57472 | unavailable | 0.009498 |
| svd_full_tall_c32_64x32 | jax-cuda-jit | 2 | 15 | 2.786 | 57472 | unavailable | 0.009498 |
| svd_full_tall_c32_64x32 | pytorch-cuda | 2 | 15 | 2.823 | 57472 | unavailable | 0.008819 |
| svd_full_tall_c32_64x32 | tenferro-cuda-direct | 2 | 15 | 2.788 | 57472 | unavailable | 0.009498 |
| svd_full_tall_c32_64x32 | tenferro-cuda-eager | 2 | 15 | 2.872 | 57472 | unavailable | 0.009498 |
| svd_full_tall_c64_64x32 | jax-cuda | 1 | 15 | 2.397 | 114944 | unavailable | 1.243e-05 |
| svd_full_tall_c64_64x32 | jax-cuda-jit | 1 | 15 | 2.320 | 114944 | unavailable | 1.243e-05 |
| svd_full_tall_c64_64x32 | pytorch-cuda | 1 | 15 | 2.286 | 114944 | unavailable | 1.188e-05 |
| svd_full_tall_c64_64x32 | tenferro-cuda-direct | 1 | 15 | 2.263 | 114944 | unavailable | 1.221e-05 |
| svd_full_tall_c64_64x32 | tenferro-cuda-eager | 1 | 15 | 2.321 | 114944 | unavailable | 1.221e-05 |
| svd_full_tall_f32_64x32 | jax-cuda | 2 | 15 | 2.131 | 28800 | unavailable | 0.007661 |
| svd_full_tall_f32_64x32 | jax-cuda-jit | 2 | 15 | 2.074 | 28800 | unavailable | 0.007661 |
| svd_full_tall_f32_64x32 | pytorch-cuda | 2 | 15 | 2.099 | 28800 | unavailable | 0.007658 |
| svd_full_tall_f32_64x32 | tenferro-cuda-direct | 2 | 15 | 2.095 | 28800 | unavailable | 0.007661 |
| svd_full_tall_f32_64x32 | tenferro-cuda-eager | 2 | 15 | 2.202 | 28800 | unavailable | 0.007661 |
| svd_full_tall_f64_64x32 | jax-cuda | 2 | 15 | 2.962 | 57600 | unavailable | 9.548e-06 |
| svd_full_tall_f64_64x32 | jax-cuda-jit | 2 | 15 | 2.959 | 57600 | unavailable | 9.548e-06 |
| svd_full_tall_f64_64x32 | pytorch-cuda | 2 | 15 | 2.953 | 57600 | unavailable | 9.215e-06 |
| svd_full_tall_f64_64x32 | tenferro-cuda-direct | 2 | 15 | 2.952 | 57600 | unavailable | 9.548e-06 |
| svd_full_tall_f64_64x32 | tenferro-cuda-eager | 2 | 15 | 3.014 | 57600 | unavailable | 9.548e-06 |
| svd_full_wide_c32_32x64 | jax-cuda | 2 | 15 | 2.857 | 57472 | unavailable | 0.008495 |
| svd_full_wide_c32_32x64 | jax-cuda-jit | 2 | 15 | 2.803 | 57472 | unavailable | 0.008495 |
| svd_full_wide_c32_32x64 | pytorch-cuda | 2 | 15 | 2.835 | 57472 | unavailable | 0.008905 |
| svd_full_wide_c32_32x64 | tenferro-cuda-direct | 2 | 15 | 2.811 | 57472 | unavailable | 0.008495 |
| svd_full_wide_c32_32x64 | tenferro-cuda-eager | 2 | 15 | 2.920 | 57472 | unavailable | 0.008495 |
| svd_full_wide_c64_32x64 | jax-cuda | 1 | 15 | 2.490 | 114944 | unavailable | 1.288e-05 |
| svd_full_wide_c64_32x64 | jax-cuda-jit | 1 | 15 | 2.468 | 114944 | unavailable | 1.288e-05 |
| svd_full_wide_c64_32x64 | pytorch-cuda | 1 | 15 | 2.423 | 114944 | unavailable | 1.299e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-direct | 1 | 15 | 2.401 | 114944 | unavailable | 1.31e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-eager | 1 | 15 | 2.474 | 114944 | unavailable | 1.31e-05 |
| svd_full_wide_f32_32x64 | jax-cuda | 2 | 15 | 2.334 | 28800 | unavailable | 0.005157 |
| svd_full_wide_f32_32x64 | jax-cuda-jit | 2 | 15 | 2.214 | 28800 | unavailable | 0.005157 |
| svd_full_wide_f32_32x64 | pytorch-cuda | 2 | 15 | 2.265 | 28800 | unavailable | 0.005115 |
| svd_full_wide_f32_32x64 | tenferro-cuda-direct | 2 | 15 | 2.248 | 28800 | unavailable | 0.005157 |
| svd_full_wide_f32_32x64 | tenferro-cuda-eager | 2 | 15 | 2.356 | 28800 | unavailable | 0.005157 |
| svd_full_wide_f64_32x64 | jax-cuda | 2 | 15 | 2.996 | 57600 | unavailable | 9.892e-06 |
| svd_full_wide_f64_32x64 | jax-cuda-jit | 2 | 15 | 3.014 | 57600 | unavailable | 9.892e-06 |
| svd_full_wide_f64_32x64 | pytorch-cuda | 2 | 15 | 3.152 | 57600 | unavailable | 7.327e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-direct | 2 | 15 | 3.011 | 57600 | unavailable | 9.838e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-eager | 2 | 15 | 3.127 | 57600 | unavailable | 9.838e-06 |
| svdvals_f32_32x32 | jax-cuda | 4 | 15 | 2.820 | 4224 | unavailable | 0.003338 |
| svdvals_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.857 | 4224 | unavailable | 0.003338 |
| svdvals_f32_32x32 | pytorch-cuda | 8 | 15 | 3.337 | 4224 | unavailable | 0.007749 |
| svdvals_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.747 | 4224 | unavailable | 0.003338 |
| svdvals_f32_32x32 | tenferro-cuda-eager | — | — | — | 4224 | unavailable | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_32x32 | jax-cuda | 2 | 15 | 2.070 | 8448 | unavailable | 4.546e-06 |
| svdvals_f64_32x32 | jax-cuda-jit | 2 | 15 | 2.037 | 8448 | unavailable | 4.546e-06 |
| svdvals_f64_32x32 | pytorch-cuda | 4 | 15 | 2.954 | 8448 | unavailable | 3.387e-06 |
| svdvals_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 3.973 | 8448 | unavailable | 4.546e-06 |
| svdvals_f64_32x32 | tenferro-cuda-eager | — | — | — | 8448 | unavailable | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| triangular_solve_f32_32x32 | jax-cuda | 32 | 15 | 2.721 | 6144 | unavailable | 0.0001546 |
| triangular_solve_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.299 | 6144 | unavailable | 0.0001546 |
| triangular_solve_f32_32x32 | pytorch-cuda | 128 | 15 | 3.588 | 6144 | unavailable | 9.242e-05 |
| triangular_solve_f32_32x32 | tenferro-cuda-direct | 128 | 15 | 2.326 | 6144 | unavailable | 0.0001546 |
| triangular_solve_f32_32x32 | tenferro-cuda-eager | 32 | 15 | 2.364 | 6144 | unavailable | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda | 32 | 15 | 2.797 | 12288 | unavailable | 1.609e-07 |
| triangular_solve_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.308 | 12288 | unavailable | 1.609e-07 |
| triangular_solve_f64_32x32 | pytorch-cuda | 128 | 15 | 3.531 | 12288 | unavailable | 1.415e-07 |
| triangular_solve_f64_32x32 | tenferro-cuda-direct | 128 | 15 | 2.475 | 12288 | unavailable | 1.609e-07 |
| triangular_solve_f64_32x32 | tenferro-cuda-eager | 32 | 15 | 2.397 | 12288 | unavailable | 1.609e-07 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api_latency
suite_file: benchmarks/gpu/linalg_public_api_latency.yaml
timestamp: '2026-11-05T02:05:47Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: a000e5c4929d5bebb91e783c329b27106baed12e
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: ab5a363734c362746c60ebcbc9cd31e68b2e0768
  dirty: false
resolved_deps:
  strided-perm:
  - 0.8.0
  - 0.10.0
  strided-view:
  - 0.7.0
  threadteam:
  - 0.1.0
  - 0.3.0
host:
  cpu: AMD EPYC 7713P 64-Core Processor
  logical_cpus: 64
  power:
    cpufreq_boost: '1'
    governor: schedutil
    scaling_max_freq_khz: '2000000'
    smt:
      active: '0'
      control: notsupported
  numa:
    nodes: 1
    cpulists:
      '0': 0-63
  smt_siblings:
  - '0'
  - '1'
  - '2'
  - '3'
  - '4'
  - '5'
  - '6'
  - '7'
  - '8'
  - '9'
  - '10'
  - '11'
  - '12'
  - '13'
  - '14'
  - '15'
  - '16'
  - '17'
  - '18'
  - '19'
  - '20'
  - '21'
  - '22'
  - '23'
  - '24'
  - '25'
  - '26'
  - '27'
  - '28'
  - '29'
  - '30'
  - '31'
  - '32'
  - '33'
  - '34'
  - '35'
  - '36'
  - '37'
  - '38'
  - '39'
  - '40'
  - '41'
  - '42'
  - '43'
  - '44'
  - '45'
  - '46'
  - '47'
  - '48'
  - '49'
  - '50'
  - '51'
  - '52'
  - '53'
  - '54'
  - '55'
  - '56'
  - '57'
  - '58'
  - '59'
  - '60'
  - '61'
  - '62'
  - '63'
  l3_domains:
  - 0-7
  - 8-15
  - 16-23
  - 24-31
  - 32-39
  - 40-47
  - 48-55
  - 56-63
build:
  toolchain:
    rustc: 'rustc 1.99.1 (58cf8f9c3 2026-10-13)

      binary: rustc

      commit-hash: 58cf8f9c35b355199727627d07ae34a4e2846d67

      commit-date: 2026-10-13

      host: x86_64-unknown-linux-gnu

      release: 1.99.1

      LLVM version: 23.1.1

      '
    rustc_release: 1.99.1
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: bd7fc8e76a87
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-63
    l3_domains:
    - 0-7
    - 8-15
    - 16-23
    - 24-31
    - 32-39
    - 40-47
    - 48-55
    - 56-63
    cpu_boost: enabled
    scaling_governor: schedutil
    scaling_driver: acpi-cpufreq
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
  trace_execution: 'tenferro trace excluded: this suite compares immediate per-op
    calls, while trace executes a prepared graph in one caller-held session (run_prepared_in);
    JAX JIT compilation outside timing'
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
  device_name: NVIDIA A100 80GB PCIe
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  device_memory_bytes: 85899345920
  driver_version: 580.178.04
  runtime_version: '12.9'
  cuda_version: '13.0'
  toolkit_version: '12.9'
  toolkit_root: /usr/local/cuda
  cutensor_version: 2.2.0
  cusolver_version: 11.7.5
  cudnn_version: '92700'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  cuda_visible_devices: GPU-530977e1-4968-9283-4129-9fbec3e66542
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh linalg_public_api_latency
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 82
  selected_cases: 82
  sha256:
    .devcontainer/cuda/devcontainer.json: 0bf1465bbd1a4c9f896d5320838f2d5a488632a59af7f9b59cc050e95e0b8e09
    .devcontainer/cuda/Dockerfile: 578d9815e5145aa1651ac7e93833468c345ee8bf5e95bfa7717551580ae9feca
    src/bin/benchmark_gpu_public_api.rs: c04fab6297d9d008d5806c829b6c77349cd4665a1bd6cc6835c2fcde79e7a44c
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 291e63a517b6f7108527254a7f83751420aa1d4e73395aa60f0691312b2a3773
    src/bin/benchmark_gpu_public_api/plan.rs: 2d87b9d73bf25df8797133cf0a8dc8b1fa974fc3bee055ad92e42802c9ebf643
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: e11a863aee7a85822c474eb771b45ec402e21dbe92173d178c42119ab8ca307e
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: c888dccb103bcb7a764a1e592eb046ce32e1e55a274fcfd71f8b3052a93a694d
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 31587e7a83440d99417be00110c4dd12c5cc1143e620bf24b1c03fe5f76bfb5e
    data/results/nvidia-gpu/gpu/linalg_public_api_latency/20261105_020224/fixtures/manifest.json: b79e18fe5dcfdee2c281cd060a333b64492775ceb0b8209a39523fa389bb160d
    target/release/benchmark_gpu_public_api: 34546e3ed1766badcc680cf308a6262672e67901a681a3518564249bded54c46
  reference_numpy_version: 2.4.2
```
