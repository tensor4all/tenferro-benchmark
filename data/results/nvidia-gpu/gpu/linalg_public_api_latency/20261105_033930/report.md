# gpu/linalg_public_api_latency: CUDA public API benchmark

Measured cells: **308**. Unsupported cells: **102**, including **28** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1431.2 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api_latency/20261105_033930/`.

Small-shape view of the same CUDA public linalg API surface: the 82 cases whose largest matrix dimension is at most 128. `small` is a shape class, not a measured classification: device kernel time is a small fraction of each measurement here, so these rows are per-call launch and synchronization overhead diagnostics rather than GPU throughput. They are never comparable with, or averaged together with, the large-shape rows of `gpu/linalg_public_api_throughput` (tenferro-rs #2087). Issue 155. Rows here are lower than on pages published before this harness change because the harness used to open the CUDA engine once per case inside the timed region and now opens it once per arm, so the difference is removed harness cost and not a change in the kernels.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | 75.774 (0.346) | 110.362 (7.096) | 61.031 (0.234) | 76.937 (55.052) | 137.331 (1.521) |
| cholesky_f64_32x32 | 77.682 (0.463) | 113.778 (5.436) | 62.561 (0.709) | 75.961 (60.476) | 76.921 (3.042) |
| svdvals_f32_32x32 | 687.119 (7.224) | unsupported | 399.372 (1.365) | 716.760 (62.421) | 723.606 (10.769) |
| svdvals_f64_32x32 | 972.394 (1.139) | unsupported | 721.770 (1.540) | 1118.105 (107.466) | 1102.139 (138.560) |
| eigvalsh_f32_32x32 | 224.074 (0.907) | 259.933 (1.796) | 222.473 (1.006) | 580.148 (2.283) | 583.076 (9.399) |
| eigvalsh_f64_32x32 | 264.947 (1.305) | 303.652 (7.114) | 263.479 (3.041) | 719.953 (3.389) | 714.158 (9.194) |
| pinv_f32_32x32 | unsupported | unsupported | 485.308 (9.314) | 953.531 (8.495) | 721.668 (1.924) |
| pinv_f64_32x32 | unsupported | unsupported | 834.505 (22.490) | 1282.870 (23.798) | 1036.652 (9.990) |
| inv_f32_32x32 | unsupported | unsupported | 182.758 (0.717) | 125.318 (1.743) | 115.587 (0.972) |
| inv_f64_32x32 | unsupported | unsupported | 195.472 (5.946) | 127.525 (98.721) | 116.541 (0.359) |
| det_f32_32x32 | 368.989 (5.317) | 402.036 (0.686) | 99.142 (0.204) | 152.243 (64.281) | 84.933 (2.140) |
| det_f64_32x32 | 376.387 (1.827) | 400.667 (1.189) | 101.159 (0.180) | 96.399 (2.201) | 84.844 (0.425) |
| norm_f32_32x32 | 68.225 (2.648) | 86.635 (0.301) | 6.010 (0.112) | 53.961 (0.502) | 41.828 (0.459) |
| norm_f64_32x32 | 70.476 (4.228) | 91.613 (2.872) | 5.974 (0.930) | 53.268 (0.383) | 42.141 (0.327) |
| triangular_solve_f32_32x32 | 17.812 (0.017) | 68.999 (4.354) | 27.183 (1.351) | 83.730 (0.570) | 76.032 (0.473) |
| triangular_solve_f64_32x32 | 19.004 (0.009) | 64.779 (6.584) | 28.224 (3.447) | 142.024 (54.830) | 78.294 (0.562) |
| svd_full_square_f32_32x32 | 654.233 (5.413) | 686.422 (15.478) | 350.579 (8.562) | 676.976 (109.820) | 670.049 (31.673) |
| svd_full_tall_f32_64x32 | 1040.767 (7.378) | 1081.274 (8.368) | 1040.532 (7.965) | 1058.263 (6.128) | 1041.267 (8.040) |
| svd_full_wide_f32_32x64 | 1108.705 (4.960) | 1165.351 (4.498) | 1108.254 (5.545) | 1140.240 (127.852) | 1128.729 (26.629) |
| slogdet_positive_f32_32x32 | 314.025 (0.668) | 386.028 (26.690) | 122.820 (10.400) | 104.368 (0.604) | 97.081 (0.940) |
| slogdet_negative_f32_32x32 | 314.667 (1.618) | 359.288 (4.443) | 115.052 (0.519) | 104.845 (1.051) | 96.562 (0.512) |
| svd_full_square_f64_32x32 | 934.763 (2.114) | 978.882 (8.209) | 607.421 (0.864) | 962.504 (10.267) | 969.394 (116.332) |
| svd_full_tall_f64_64x32 | 1440.306 (35.949) | 1481.743 (4.548) | 1466.897 (9.438) | 1502.663 (30.688) | 1478.037 (13.033) |
| svd_full_wide_f64_32x64 | 1463.177 (4.803) | 1518.214 (8.568) | 1554.115 (2.048) | 1532.539 (14.238) | 1503.973 (5.583) |
| slogdet_positive_f64_32x32 | 315.974 (1.711) | 365.240 (0.657) | 115.722 (0.361) | 103.267 (0.958) | 95.358 (1.689) |
| slogdet_negative_f64_32x32 | 315.845 (2.058) | 365.897 (1.424) | 115.837 (0.184) | 105.020 (3.411) | 94.748 (0.509) |
| svd_full_square_c32_32x32 | 957.164 (1.444) | 1020.331 (7.788) | 591.078 (1.036) | 990.440 (6.430) | 972.754 (2.440) |
| svd_full_tall_c32_64x32 | 1383.629 (16.021) | 1436.151 (5.477) | 1399.925 (3.678) | 1419.250 (34.209) | 1402.900 (4.728) |
| svd_full_wide_c32_32x64 | 1378.879 (7.873) | 1432.916 (10.580) | 1404.959 (3.272) | 1439.471 (7.728) | 1428.740 (25.188) |
| slogdet_positive_c32_32x32 | 326.291 (5.363) | 392.361 (0.758) | 125.866 (0.175) | 104.634 (3.448) | 97.923 (0.894) |
| slogdet_negative_c32_32x32 | 325.559 (1.568) | 389.595 (1.240) | 126.643 (0.276) | 102.167 (2.806) | 93.247 (1.590) |
| svd_full_square_c64_32x32 | 1786.873 (4.588) | 1848.645 (3.473) | 1232.198 (2.747) | 1838.145 (10.221) | 1816.894 (13.633) |
| svd_full_tall_c64_64x32 | 2273.381 (6.880) | 2328.302 (23.966) | 2280.321 (6.850) | 2845.291 (301.925) | 2354.513 (13.500) |
| svd_full_wide_c64_32x64 | 2386.255 (6.960) | 2475.697 (9.501) | 2416.095 (5.925) | 2550.390 (35.546) | 2534.380 (9.816) |
| slogdet_positive_c64_32x32 | 333.245 (1.133) | 425.614 (1.187) | 130.387 (0.178) | 102.466 (1.685) | 106.223 (0.942) |
| slogdet_negative_c64_32x32 | 331.974 (0.776) | 424.664 (1.994) | 144.118 (9.325) | 106.984 (4.530) | 110.941 (76.972) |
| householder_qr_square_f32_32x32 | 117.649 (0.466) | 160.590 (0.967) | 83.367 (0.167) | 84.900 (0.182) | 88.749 (0.115) |
| householder_qr_tall_f32_64x32 | 156.840 (1.970) | 210.563 (0.553) | 125.929 (0.342) | 129.170 (0.221) | 128.846 (0.137) |
| householder_qr_wide_f32_32x64 | 161.105 (0.635) | 208.424 (1.703) | 126.158 (0.092) | 129.849 (0.303) | 129.945 (0.256) |
| householder_qr_square_f64_32x32 | 134.825 (0.662) | 178.013 (0.407) | 99.422 (0.044) | 101.208 (0.088) | 101.369 (0.223) |
| householder_qr_tall_f64_64x32 | 177.476 (3.416) | 217.465 (14.049) | 139.729 (0.105) | 144.063 (0.185) | 144.697 (0.324) |
| householder_qr_wide_f64_32x64 | 173.161 (0.598) | 225.139 (4.904) | 140.201 (0.059) | 145.359 (0.222) | 145.566 (0.308) |
| householder_qr_square_c32_32x32 | 127.084 (0.820) | 167.487 (1.104) | 94.968 (0.046) | 96.672 (0.146) | 96.663 (0.239) |
| householder_qr_tall_c32_64x32 | 181.581 (2.924) | 217.630 (0.826) | 145.248 (0.079) | 149.822 (0.398) | 149.923 (0.315) |
| householder_qr_wide_c32_32x64 | 180.044 (0.909) | 229.080 (10.743) | 146.311 (0.102) | 150.905 (0.379) | 151.646 (0.426) |
| slogdet_phase_c32_32x32 | 342.628 (1.122) | 390.646 (0.957) | 126.802 (13.600) | 104.102 (0.669) | 100.200 (1.544) |
| householder_qr_square_c64_32x32 | 159.385 (1.887) | 200.817 (1.219) | 127.251 (0.409) | 131.690 (0.192) | 132.567 (0.173) |
| householder_qr_tall_c64_64x32 | 213.144 (3.240) | 247.729 (0.874) | 173.743 (0.134) | 178.516 (1.404) | 179.473 (0.222) |
| householder_qr_wide_c64_32x64 | 223.004 (3.235) | 258.510 (1.964) | 183.260 (0.105) | 187.906 (0.233) | 188.574 (0.458) |
| slogdet_phase_c64_32x32 | 336.256 (15.272) | 392.783 (1.226) | 130.242 (0.181) | 109.732 (1.121) | 109.204 (64.246) |
| householder_from_factors_default_f32_64x32 | 354.484 (20.301) | 421.839 (25.757) | unsupported | unsupported | unsupported |
| householder_append_default_f32_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f32_64x32 | 32.947 (1.124) | 42.662 (0.214) | 4.766 (0.463) | 53.783 (0.811) | 51.245 (18.072) |
| householder_q_columns_prefix_f32_64x32 | unsupported | unsupported | 64.798 (0.500) | 245.906 (81.515) | 188.295 (87.126) |
| householder_q_columns_complement_f32_64x32 | unsupported | unsupported | 253.204 (1.039) | unsupported | unsupported |
| householder_from_factors_default_f64_64x32 | 378.203 (20.286) | 449.026 (6.368) | unsupported | unsupported | unsupported |
| householder_append_default_f64_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f64_64x32 | 36.405 (0.209) | 47.678 (0.220) | 5.096 (0.688) | 54.330 (1.045) | 54.197 (8.012) |
| householder_q_columns_prefix_f64_64x32 | unsupported | unsupported | 68.611 (1.012) | 269.812 (67.430) | 159.260 (101.278) |
| householder_q_columns_complement_f64_64x32 | unsupported | unsupported | 282.620 (0.739) | unsupported | unsupported |
| householder_from_factors_default_c32_64x32 | 373.687 (17.524) | 450.868 (33.384) | unsupported | unsupported | unsupported |
| householder_append_default_c32_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c32_64x32 | 33.664 (0.104) | 45.513 (4.124) | 5.187 (0.029) | 52.582 (11.603) | 54.224 (20.992) |
| householder_q_columns_prefix_c32_64x32 | unsupported | unsupported | 62.008 (2.453) | 143.857 (27.347) | 113.400 (0.648) |
| householder_q_columns_complement_c32_64x32 | unsupported | unsupported | 269.721 (1.016) | unsupported | unsupported |
| householder_from_factors_default_c64_64x32 | 392.434 (1.093) | 485.480 (1.862) | unsupported | unsupported | unsupported |
| householder_append_default_c64_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c64_64x32 | 33.931 (0.136) | 43.442 (0.183) | 4.383 (0.084) | 51.129 (1.941) | 51.582 (2.625) |
| householder_q_columns_prefix_c64_64x32 | unsupported | unsupported | 56.779 (0.108) | 146.820 (6.061) | 111.546 (6.295) |
| householder_q_columns_complement_c64_64x32 | unsupported | unsupported | 290.226 (0.258) | unsupported | unsupported |
| lu_square_f32_s32 | 159.890 (0.553) | 209.956 (9.837) | unsupported | unsupported | unsupported |
| lu_square_f64_s32 | 170.205 (0.493) | 201.967 (0.919) | unsupported | unsupported | unsupported |
| lu_square_c32_s32 | 173.669 (0.641) | 205.446 (2.208) | unsupported | unsupported | unsupported |
| lu_square_c64_s32 | 174.252 (0.192) | 211.534 (0.592) | unsupported | unsupported | unsupported |
| lu_tall_f32_s32 | 170.581 (3.095) | 201.828 (1.555) | unsupported | unsupported | unsupported |
| lu_tall_f64_s32 | 169.305 (0.835) | 206.564 (0.618) | unsupported | unsupported | unsupported |
| lu_tall_c32_s32 | 172.998 (0.220) | 209.987 (7.099) | unsupported | unsupported | unsupported |
| lu_tall_c64_s32 | 173.649 (0.890) | 210.819 (0.684) | unsupported | unsupported | unsupported |
| lu_wide_f32_s32 | 174.213 (0.619) | 204.815 (1.410) | unsupported | unsupported | unsupported |
| lu_wide_f64_s32 | 173.859 (0.797) | 209.807 (0.591) | unsupported | unsupported | unsupported |
| lu_wide_c32_s32 | 176.113 (0.714) | 219.017 (0.732) | unsupported | unsupported | unsupported |
| lu_wide_c64_s32 | 184.288 (0.700) | 221.522 (4.768) | unsupported | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | jax-cuda | 32 | 15 | 2.462 | 8192 | 0.000 | 0.0002352 |
| cholesky_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.197 | 8192 | 0.000 | 0.0002352 |
| cholesky_f32_32x32 | pytorch-cuda | 64 | 15 | 3.906 | 8192 | 0.000 | 0.0002352 |
| cholesky_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.425 | 8192 | 0.000 | 0.0002352 |
| cholesky_f32_32x32 | tenferro-cuda-eager | 32 | 15 | 3.532 | 8192 | 0.000 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda | 16 | 15 | 1.215 | 16384 | 0.000 | 2.188e-07 |
| cholesky_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.461 | 16384 | 0.000 | 2.188e-07 |
| cholesky_f64_32x32 | pytorch-cuda | 32 | 15 | 2.002 | 16384 | 0.000 | 2.188e-07 |
| cholesky_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.486 | 16384 | 0.000 | 2.188e-07 |
| cholesky_f64_32x32 | tenferro-cuda-eager | 32 | 15 | 3.641 | 16384 | 0.000 | 2.188e-07 |
| det_f32_32x32 | jax-cuda | 16 | 15 | 2.436 | 4100 | 0.000 | 0.002641 |
| det_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.718 | 4100 | 0.000 | 0.002641 |
| det_f32_32x32 | pytorch-cuda | 32 | 15 | 3.173 | 4100 | 0.000 | 0.002641 |
| det_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.952 | 4100 | 0.000 | 0.002641 |
| det_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.216 | 4100 | 0.000 | 0.002641 |
| det_f64_32x32 | jax-cuda | 32 | 15 | 3.085 | 8200 | 0.000 | 1.135e-06 |
| det_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.715 | 8200 | 0.000 | 1.135e-06 |
| det_f64_32x32 | pytorch-cuda | 32 | 15 | 3.237 | 8200 | 0.000 | 3.784e-07 |
| det_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 3.011 | 8200 | 0.000 | 1.135e-06 |
| det_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.205 | 8200 | 0.000 | 1.135e-06 |
| eigvalsh_f32_32x32 | jax-cuda | 4 | 15 | 2.321 | 4224 | 0.000 | 0.004775 |
| eigvalsh_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.332 | 4224 | 0.000 | 0.004775 |
| eigvalsh_f32_32x32 | pytorch-cuda | 16 | 15 | 3.560 | 4224 | 0.000 | 0.001073 |
| eigvalsh_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.585 | 4224 | 0.000 | 0.001073 |
| eigvalsh_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.079 | 4224 | 0.000 | 0.001073 |
| eigvalsh_f64_32x32 | jax-cuda | 4 | 15 | 2.880 | 8448 | 0.000 | 5.542e-06 |
| eigvalsh_f64_32x32 | jax-cuda-jit | 4 | 15 | 2.857 | 8448 | 0.000 | 5.542e-06 |
| eigvalsh_f64_32x32 | pytorch-cuda | 8 | 15 | 2.108 | 8448 | 0.000 | 5.433e-07 |
| eigvalsh_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.120 | 8448 | 0.000 | 5.433e-07 |
| eigvalsh_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.429 | 8448 | 0.000 | 5.433e-07 |
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
| householder_from_factors_default_c32_64x32 | tenferro-cuda-direct | 8 | 15 | 2.989 | 57600 | 0.000 | 0.0003232 |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-eager | 8 | 15 | 3.607 | 57600 | 0.000 | 0.0003232 |
| householder_from_factors_default_c64_64x32 | jax-cuda | — | — | — | 115200 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_64x32 | jax-cuda-jit | — | — | — | 115200 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_64x32 | pytorch-cuda | — | — | — | 115200 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-direct | 8 | 15 | 3.139 | 115200 | 0.000 | 4.441e-07 |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-eager | 8 | 15 | 3.884 | 115200 | 0.000 | 4.441e-07 |
| householder_from_factors_default_f32_64x32 | jax-cuda | — | — | — | 28800 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_64x32 | jax-cuda-jit | — | — | — | 28800 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_64x32 | pytorch-cuda | — | — | — | 28800 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-direct | 8 | 15 | 2.836 | 28800 | 0.000 | 0.0004571 |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-eager | 8 | 15 | 3.375 | 28800 | 0.000 | 0.0004571 |
| householder_from_factors_default_f64_64x32 | jax-cuda | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_64x32 | jax-cuda-jit | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_64x32 | pytorch-cuda | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-direct | 8 | 15 | 3.026 | 57600 | 0.000 | 3.886e-07 |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-eager | 8 | 15 | 3.592 | 57600 | 0.000 | 3.886e-07 |
| householder_q_columns_complement_c32_64x32 | jax-cuda | — | — | — | 32768 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c32_64x32 | jax-cuda-jit | — | — | — | 32768 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c32_64x32 | pytorch-cuda | 8 | 15 | 2.158 | 32768 | 0.000 | 0.000299 |
| householder_q_columns_complement_c32_64x32 | tenferro-cuda-direct | — | — | — | 32768 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_64x32 | tenferro-cuda-eager | — | — | — | 32768 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_64x32 | jax-cuda | — | — | — | 65536 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c64_64x32 | jax-cuda-jit | — | — | — | 65536 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c64_64x32 | pytorch-cuda | 8 | 15 | 2.322 | 65536 | 0.000 | 3.341e-07 |
| householder_q_columns_complement_c64_64x32 | tenferro-cuda-direct | — | — | — | 65536 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_64x32 | tenferro-cuda-eager | — | — | — | 65536 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_64x32 | jax-cuda | — | — | — | 16384 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f32_64x32 | jax-cuda-jit | — | — | — | 16384 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f32_64x32 | pytorch-cuda | 8 | 15 | 2.026 | 16384 | 0.000 | 0.0001791 |
| householder_q_columns_complement_f32_64x32 | tenferro-cuda-direct | — | — | — | 16384 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_64x32 | tenferro-cuda-eager | — | — | — | 16384 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_64x32 | jax-cuda | — | — | — | 32768 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f64_64x32 | jax-cuda-jit | — | — | — | 32768 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f64_64x32 | pytorch-cuda | 8 | 15 | 2.261 | 32768 | 0.000 | 2.724e-08 |
| householder_q_columns_complement_f64_64x32 | tenferro-cuda-direct | — | — | — | 32768 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_64x32 | tenferro-cuda-eager | — | — | — | 32768 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c32_64x32 | jax-cuda | 8 | 15 | 1.151 | 20480 | 0.000 | 0.00012 |
| householder_q_columns_prefix_c32_64x32 | jax-cuda-jit | 16 | 15 | 1.814 | 20480 | 0.000 | 0.00012 |
| householder_q_columns_prefix_c32_64x32 | pytorch-cuda | 32 | 15 | 1.984 | 20480 | 0.000 | 0.00012 |
| householder_q_columns_prefix_c32_64x32 | tenferro-cuda-direct | — | — | — | 20480 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c32_64x32 | tenferro-cuda-eager | — | — | — | 20480 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c64_64x32 | jax-cuda | 8 | 15 | 1.175 | 40960 | 0.000 | 5.592e-08 |
| householder_q_columns_prefix_c64_64x32 | jax-cuda-jit | 32 | 15 | 3.569 | 40960 | 0.000 | 5.592e-08 |
| householder_q_columns_prefix_c64_64x32 | pytorch-cuda | 64 | 15 | 3.634 | 40960 | 0.001 | 5.592e-08 |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-direct | — | — | — | 40960 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-eager | — | — | — | 40960 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f32_64x32 | jax-cuda | 8 | 15 | 1.967 | 10240 | 0.000 | 0.0001196 |
| householder_q_columns_prefix_f32_64x32 | jax-cuda-jit | 16 | 15 | 3.013 | 10240 | 0.000 | 0.0001196 |
| householder_q_columns_prefix_f32_64x32 | pytorch-cuda | 32 | 15 | 2.074 | 10240 | 0.000 | 0.0001196 |
| householder_q_columns_prefix_f32_64x32 | tenferro-cuda-direct | — | — | — | 10240 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f32_64x32 | tenferro-cuda-eager | — | — | — | 10240 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f64_64x32 | jax-cuda | 8 | 15 | 2.158 | 20480 | 0.000 | 1.114e-07 |
| householder_q_columns_prefix_f64_64x32 | jax-cuda-jit | 16 | 15 | 2.548 | 20480 | 0.000 | 1.114e-07 |
| householder_q_columns_prefix_f64_64x32 | pytorch-cuda | 32 | 15 | 2.196 | 20480 | 0.000 | 1.114e-07 |
| householder_q_columns_prefix_f64_64x32 | tenferro-cuda-direct | — | — | — | 20480 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f64_64x32 | tenferro-cuda-eager | — | — | — | 20480 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_qr_square_c32_32x32 | jax-cuda | 32 | 15 | 3.093 | 16640 | 0.000 | 0.0005833 |
| householder_qr_square_c32_32x32 | jax-cuda-jit | 32 | 15 | 3.093 | 16640 | 0.000 | 0.0005833 |
| householder_qr_square_c32_32x32 | pytorch-cuda | 32 | 15 | 3.039 | 16640 | 0.000 | 0.0005833 |
| householder_qr_square_c32_32x32 | tenferro-cuda-direct | 16 | 15 | 2.033 | 16640 | 0.000 | 0.0005631 |
| householder_qr_square_c32_32x32 | tenferro-cuda-eager | 16 | 15 | 2.680 | 16640 | 0.000 | 0.0005631 |
| householder_qr_square_c64_32x32 | jax-cuda | 16 | 15 | 2.107 | 33280 | 0.000 | 1.221e-06 |
| householder_qr_square_c64_32x32 | jax-cuda-jit | 16 | 15 | 2.121 | 33280 | 0.000 | 1.221e-06 |
| householder_qr_square_c64_32x32 | pytorch-cuda | 16 | 15 | 2.036 | 33280 | 0.000 | 1.221e-06 |
| householder_qr_square_c64_32x32 | tenferro-cuda-direct | 16 | 15 | 2.550 | 33280 | 0.000 | 4.996e-07 |
| householder_qr_square_c64_32x32 | tenferro-cuda-eager | 16 | 15 | 3.213 | 33280 | 0.000 | 4.996e-07 |
| householder_qr_square_f32_32x32 | jax-cuda | 32 | 15 | 2.717 | 8320 | 0.000 | 0.000367 |
| householder_qr_square_f32_32x32 | jax-cuda-jit | 16 | 15 | 1.420 | 8320 | 0.000 | 0.000367 |
| householder_qr_square_f32_32x32 | pytorch-cuda | 32 | 15 | 2.668 | 8320 | 0.000 | 0.000367 |
| householder_qr_square_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 1.882 | 8320 | 0.000 | 0.0004212 |
| householder_qr_square_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 2.569 | 8320 | 0.000 | 0.0004212 |
| householder_qr_square_f64_32x32 | jax-cuda | 32 | 15 | 3.239 | 16640 | 0.000 | 6.106e-07 |
| householder_qr_square_f64_32x32 | jax-cuda-jit | 32 | 15 | 3.244 | 16640 | 0.000 | 6.106e-07 |
| householder_qr_square_f64_32x32 | pytorch-cuda | 32 | 15 | 3.181 | 16640 | 0.000 | 6.106e-07 |
| householder_qr_square_f64_32x32 | tenferro-cuda-direct | 16 | 15 | 2.157 | 16640 | 0.000 | 3.886e-07 |
| householder_qr_square_f64_32x32 | tenferro-cuda-eager | 16 | 15 | 2.848 | 16640 | 0.000 | 3.886e-07 |
| householder_qr_tall_c32_64x32 | jax-cuda | 16 | 15 | 2.397 | 33024 | 0.000 | 0.0003756 |
| householder_qr_tall_c32_64x32 | jax-cuda-jit | 16 | 15 | 2.399 | 33024 | 0.000 | 0.0003756 |
| householder_qr_tall_c32_64x32 | pytorch-cuda | 16 | 15 | 2.324 | 33024 | 0.000 | 0.0003756 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-direct | 16 | 15 | 2.905 | 33024 | 0.000 | 0.0003724 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-eager | 16 | 15 | 3.482 | 33024 | 0.000 | 0.0003724 |
| householder_qr_tall_c64_64x32 | jax-cuda | 16 | 15 | 2.856 | 66048 | 0.000 | 9.992e-07 |
| householder_qr_tall_c64_64x32 | jax-cuda-jit | 16 | 15 | 2.872 | 66048 | 0.000 | 9.992e-07 |
| householder_qr_tall_c64_64x32 | pytorch-cuda | 16 | 15 | 2.780 | 66048 | 0.000 | 9.992e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-direct | 16 | 15 | 3.410 | 66048 | 0.000 | 5.551e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-eager | 16 | 15 | 3.964 | 66048 | 0.000 | 5.551e-07 |
| householder_qr_tall_f32_64x32 | jax-cuda | 16 | 15 | 2.067 | 16512 | 0.000 | 0.000327 |
| householder_qr_tall_f32_64x32 | jax-cuda-jit | 16 | 15 | 2.062 | 16512 | 0.000 | 0.000327 |
| householder_qr_tall_f32_64x32 | pytorch-cuda | 16 | 15 | 2.015 | 16512 | 0.000 | 0.000327 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-direct | 16 | 15 | 2.509 | 16512 | 0.000 | 0.0003159 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-eager | 16 | 15 | 3.369 | 16512 | 0.000 | 0.0003159 |
| householder_qr_tall_f64_64x32 | jax-cuda | 16 | 15 | 2.305 | 33024 | 0.000 | 6.661e-07 |
| householder_qr_tall_f64_64x32 | jax-cuda-jit | 16 | 15 | 2.315 | 33024 | 0.000 | 6.661e-07 |
| householder_qr_tall_f64_64x32 | pytorch-cuda | 16 | 15 | 2.236 | 33024 | 0.000 | 6.661e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-direct | 16 | 15 | 2.840 | 33024 | 0.000 | 4.441e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-eager | 16 | 15 | 3.479 | 33024 | 0.000 | 4.441e-07 |
| householder_qr_wide_c32_32x64 | jax-cuda | 16 | 15 | 2.414 | 33024 | 0.000 | 0.0004972 |
| householder_qr_wide_c32_32x64 | jax-cuda-jit | 16 | 15 | 2.426 | 33024 | 0.000 | 0.0004972 |
| householder_qr_wide_c32_32x64 | pytorch-cuda | 16 | 15 | 2.341 | 33024 | 0.000 | 0.0004972 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-direct | 16 | 15 | 2.881 | 33024 | 0.000 | 0.0006002 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-eager | 16 | 15 | 3.665 | 33024 | 0.000 | 0.0006002 |
| householder_qr_wide_c64_32x64 | jax-cuda | 16 | 15 | 3.006 | 66048 | 0.000 | 1.166e-06 |
| householder_qr_wide_c64_32x64 | jax-cuda-jit | 16 | 15 | 3.017 | 66048 | 0.000 | 1.166e-06 |
| householder_qr_wide_c64_32x64 | pytorch-cuda | 16 | 15 | 2.932 | 66048 | 0.000 | 1.166e-06 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-direct | 16 | 15 | 3.568 | 66048 | 0.000 | 3.886e-07 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-eager | 8 | 15 | 2.068 | 66048 | 0.000 | 3.886e-07 |
| householder_qr_wide_f32_32x64 | jax-cuda | 16 | 15 | 2.078 | 16512 | 0.000 | 0.0003745 |
| householder_qr_wide_f32_32x64 | jax-cuda-jit | 16 | 15 | 2.079 | 16512 | 0.000 | 0.0003745 |
| householder_qr_wide_f32_32x64 | pytorch-cuda | 16 | 15 | 2.019 | 16512 | 0.000 | 0.0003745 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-direct | 16 | 15 | 2.578 | 16512 | 0.000 | 0.0004731 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-eager | 16 | 15 | 3.335 | 16512 | 0.000 | 0.0004731 |
| householder_qr_wide_f64_32x64 | jax-cuda | 16 | 15 | 2.326 | 33024 | 0.000 | 5.551e-07 |
| householder_qr_wide_f64_32x64 | jax-cuda-jit | 16 | 15 | 2.329 | 33024 | 0.000 | 5.551e-07 |
| householder_qr_wide_f64_32x64 | pytorch-cuda | 16 | 15 | 2.243 | 33024 | 0.000 | 5.551e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-direct | 16 | 15 | 2.771 | 33024 | 0.000 | 4.441e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-eager | 16 | 15 | 3.602 | 33024 | 0.000 | 4.441e-07 |
| householder_r_default_c32_64x32 | jax-cuda | 32 | 15 | 1.683 | 24576 | 0.000 | 0.0004713 |
| householder_r_default_c32_64x32 | jax-cuda-jit | 32 | 15 | 1.735 | 24576 | 0.000 | 0.0004713 |
| householder_r_default_c32_64x32 | pytorch-cuda | 512 | 15 | 2.656 | 24576 | 0.003 | 0.0004713 |
| householder_r_default_c32_64x32 | tenferro-cuda-direct | 64 | 15 | 2.155 | 24576 | 0.001 | 0.0004713 |
| householder_r_default_c32_64x32 | tenferro-cuda-eager | 64 | 15 | 2.913 | 24576 | 0.000 | 0.0004713 |
| householder_r_default_c64_64x32 | jax-cuda | 32 | 15 | 1.636 | 49152 | 0.001 | 3.296e-07 |
| householder_r_default_c64_64x32 | jax-cuda-jit | 32 | 15 | 1.651 | 49152 | 0.001 | 3.296e-07 |
| householder_r_default_c64_64x32 | pytorch-cuda | 512 | 15 | 2.244 | 49152 | 0.008 | 3.296e-07 |
| householder_r_default_c64_64x32 | tenferro-cuda-direct | 64 | 15 | 2.172 | 49152 | 0.001 | 3.296e-07 |
| householder_r_default_c64_64x32 | tenferro-cuda-eager | 64 | 15 | 2.780 | 49152 | 0.001 | 3.296e-07 |
| householder_r_default_f32_64x32 | jax-cuda | 64 | 15 | 3.442 | 12288 | 0.000 | 0.0002373 |
| householder_r_default_f32_64x32 | jax-cuda-jit | 32 | 15 | 1.640 | 12288 | 0.000 | 0.0002373 |
| householder_r_default_f32_64x32 | pytorch-cuda | 512 | 15 | 2.440 | 12288 | 0.002 | 0.0002373 |
| householder_r_default_f32_64x32 | tenferro-cuda-direct | 64 | 15 | 2.109 | 12288 | 0.000 | 0.0002373 |
| householder_r_default_f32_64x32 | tenferro-cuda-eager | 64 | 15 | 2.730 | 12288 | 0.000 | 0.0002373 |
| householder_r_default_f64_64x32 | jax-cuda | 32 | 15 | 1.739 | 24576 | 0.000 | 2.2e-07 |
| householder_r_default_f64_64x32 | jax-cuda-jit | 32 | 15 | 1.734 | 24576 | 0.000 | 2.2e-07 |
| householder_r_default_f64_64x32 | pytorch-cuda | 512 | 15 | 2.609 | 24576 | 0.003 | 2.2e-07 |
| householder_r_default_f64_64x32 | tenferro-cuda-direct | 64 | 15 | 2.330 | 24576 | 0.000 | 2.2e-07 |
| householder_r_default_f64_64x32 | tenferro-cuda-eager | 64 | 15 | 3.051 | 24576 | 0.000 | 2.2e-07 |
| inv_f32_32x32 | jax-cuda | 16 | 15 | 2.005 | 8192 | 0.000 | 0.0004905 |
| inv_f32_32x32 | jax-cuda-jit | 32 | 15 | 3.699 | 8192 | 0.000 | 0.0004905 |
| inv_f32_32x32 | pytorch-cuda | 16 | 15 | 2.924 | 8192 | 0.000 | 0.0004905 |
| inv_f32_32x32 | tenferro-cuda-direct | — | — | — | 8192 | unavailable | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_32x32 | tenferro-cuda-eager | — | — | — | 8192 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_32x32 | jax-cuda | 16 | 15 | 2.040 | 16384 | 0.000 | 2.284e-07 |
| inv_f64_32x32 | jax-cuda-jit | 32 | 15 | 3.729 | 16384 | 0.000 | 2.284e-07 |
| inv_f64_32x32 | pytorch-cuda | 16 | 15 | 3.128 | 16384 | 0.000 | 2.284e-07 |
| inv_f64_32x32 | tenferro-cuda-direct | — | — | — | 16384 | unavailable | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_32x32 | tenferro-cuda-eager | — | — | — | 16384 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| lu_square_c32_s32 | jax-cuda | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s32 | jax-cuda-jit | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s32 | pytorch-cuda | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.779 | 32776 | 0.000 | 0.0002381 |
| lu_square_c32_s32 | tenferro-cuda-eager | 16 | 15 | 3.287 | 32776 | 0.000 | 0.0002381 |
| lu_square_c64_s32 | jax-cuda | — | — | — | 65552 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s32 | jax-cuda-jit | — | — | — | 65552 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s32 | pytorch-cuda | — | — | — | 65552 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.788 | 65552 | 0.000 | 5.532e-07 |
| lu_square_c64_s32 | tenferro-cuda-eager | 16 | 15 | 3.385 | 65552 | 0.000 | 5.532e-07 |
| lu_square_f32_s32 | jax-cuda | — | — | — | 16388 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s32 | jax-cuda-jit | — | — | — | 16388 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s32 | pytorch-cuda | — | — | — | 16388 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.558 | 16388 | 0.000 | 0.0003557 |
| lu_square_f32_s32 | tenferro-cuda-eager | 16 | 15 | 3.359 | 16388 | 0.000 | 0.0003557 |
| lu_square_f64_s32 | jax-cuda | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s32 | jax-cuda-jit | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s32 | pytorch-cuda | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.723 | 32776 | 0.000 | 4.397e-07 |
| lu_square_f64_s32 | tenferro-cuda-eager | 16 | 15 | 3.231 | 32776 | 0.000 | 4.397e-07 |
| lu_tall_c32_s32 | jax-cuda | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s32 | jax-cuda-jit | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s32 | pytorch-cuda | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.768 | 73736 | 0.000 | 0.0005956 |
| lu_tall_c32_s32 | tenferro-cuda-eager | 16 | 15 | 3.360 | 73736 | 0.000 | 0.0005956 |
| lu_tall_c64_s32 | jax-cuda | — | — | — | 147472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s32 | jax-cuda-jit | — | — | — | 147472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s32 | pytorch-cuda | — | — | — | 147472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.778 | 147472 | 0.001 | 4.406e-07 |
| lu_tall_c64_s32 | tenferro-cuda-eager | 16 | 15 | 3.373 | 147472 | 0.000 | 4.406e-07 |
| lu_tall_f32_s32 | jax-cuda | — | — | — | 36868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s32 | jax-cuda-jit | — | — | — | 36868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s32 | pytorch-cuda | — | — | — | 36868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.729 | 36868 | 0.000 | 0.0003557 |
| lu_tall_f32_s32 | tenferro-cuda-eager | 16 | 15 | 3.229 | 36868 | 0.000 | 0.0003557 |
| lu_tall_f64_s32 | jax-cuda | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s32 | jax-cuda-jit | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s32 | pytorch-cuda | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.709 | 73736 | 0.000 | 4.397e-07 |
| lu_tall_f64_s32 | tenferro-cuda-eager | 16 | 15 | 3.305 | 73736 | 0.000 | 4.397e-07 |
| lu_wide_c32_s32 | jax-cuda | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s32 | jax-cuda-jit | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s32 | pytorch-cuda | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.818 | 49160 | 0.000 | 0.0003567 |
| lu_wide_c32_s32 | tenferro-cuda-eager | 16 | 15 | 3.504 | 49160 | 0.000 | 0.0003567 |
| lu_wide_c64_s32 | jax-cuda | — | — | — | 98320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s32 | jax-cuda-jit | — | — | — | 98320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s32 | pytorch-cuda | — | — | — | 98320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.949 | 98320 | 0.000 | 3.323e-07 |
| lu_wide_c64_s32 | tenferro-cuda-eager | 16 | 15 | 3.544 | 98320 | 0.000 | 3.323e-07 |
| lu_wide_f32_s32 | jax-cuda | — | — | — | 24580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s32 | jax-cuda-jit | — | — | — | 24580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s32 | pytorch-cuda | — | — | — | 24580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.787 | 24580 | 0.000 | 0.00024 |
| lu_wide_f32_s32 | tenferro-cuda-eager | 16 | 15 | 3.277 | 24580 | 0.000 | 0.00024 |
| lu_wide_f64_s32 | jax-cuda | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s32 | jax-cuda-jit | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s32 | pytorch-cuda | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.782 | 49160 | 0.000 | 2.214e-07 |
| lu_wide_f64_s32 | tenferro-cuda-eager | 16 | 15 | 3.357 | 49160 | 0.000 | 2.214e-07 |
| norm_f32_32x32 | jax-cuda | 32 | 15 | 1.727 | 4100 | 0.000 | 0 |
| norm_f32_32x32 | jax-cuda-jit | 64 | 15 | 2.677 | 4100 | 0.000 | 0 |
| norm_f32_32x32 | pytorch-cuda | 512 | 15 | 3.077 | 4100 | 0.000 | 0 |
| norm_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.183 | 4100 | 0.000 | 0 |
| norm_f32_32x32 | tenferro-cuda-eager | 32 | 15 | 2.772 | 4100 | 0.000 | 0 |
| norm_f64_32x32 | jax-cuda | 64 | 15 | 3.409 | 8200 | 0.000 | 0 |
| norm_f64_32x32 | jax-cuda-jit | 64 | 15 | 2.697 | 8200 | 0.000 | 0 |
| norm_f64_32x32 | pytorch-cuda | 512 | 15 | 3.059 | 8200 | 0.001 | 0 |
| norm_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.255 | 8200 | 0.000 | 0 |
| norm_f64_32x32 | tenferro-cuda-eager | 32 | 15 | 2.932 | 8200 | 0.000 | 0 |
| pinv_f32_32x32 | jax-cuda | 4 | 15 | 3.814 | 8200 | 0.000 | 0.0006436 |
| pinv_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.887 | 8200 | 0.000 | 0.0006436 |
| pinv_f32_32x32 | pytorch-cuda | 8 | 15 | 3.882 | 8200 | 0.000 | 0.001025 |
| pinv_f32_32x32 | tenferro-cuda-direct | — | — | — | 8200 | unavailable | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f32_32x32 | tenferro-cuda-eager | — | — | — | 8200 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_32x32 | jax-cuda | 2 | 15 | 2.566 | 16400 | 0.000 | 1.367e-06 |
| pinv_f64_32x32 | jax-cuda-jit | 2 | 15 | 2.073 | 16400 | 0.000 | 1.367e-06 |
| pinv_f64_32x32 | pytorch-cuda | 4 | 15 | 3.338 | 16400 | 0.000 | 9.374e-07 |
| pinv_f64_32x32 | tenferro-cuda-direct | — | — | — | 16400 | unavailable | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_32x32 | tenferro-cuda-eager | — | — | — | 16400 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| slogdet_negative_c32_32x32 | jax-cuda | 32 | 15 | 3.269 | 8204 | 0.000 | 0.0004085 |
| slogdet_negative_c32_32x32 | jax-cuda-jit | 32 | 15 | 2.984 | 8204 | 0.000 | 0.0004085 |
| slogdet_negative_c32_32x32 | pytorch-cuda | 16 | 15 | 2.026 | 8204 | 0.000 | 0.0004085 |
| slogdet_negative_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.604 | 8204 | 0.000 | 0.0007148 |
| slogdet_negative_c32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.117 | 8204 | 0.000 | 0.0007148 |
| slogdet_negative_c64_32x32 | jax-cuda | 32 | 15 | 3.423 | 16408 | 0.000 | 4.755e-07 |
| slogdet_negative_c64_32x32 | jax-cuda-jit | 16 | 15 | 1.775 | 16408 | 0.000 | 4.755e-07 |
| slogdet_negative_c64_32x32 | pytorch-cuda | 16 | 15 | 2.306 | 16408 | 0.000 | 4.755e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.656 | 16408 | 0.000 | 3.804e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.397 | 16408 | 0.000 | 3.804e-07 |
| slogdet_negative_f32_32x32 | jax-cuda | 32 | 15 | 3.355 | 4104 | 0.000 | 0.000915 |
| slogdet_negative_f32_32x32 | jax-cuda-jit | 32 | 15 | 3.090 | 4104 | 0.000 | 0.000915 |
| slogdet_negative_f32_32x32 | pytorch-cuda | 32 | 15 | 3.682 | 4104 | 0.000 | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.517 | 4104 | 0.000 | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.874 | 4104 | 0.000 | 0.000915 |
| slogdet_negative_f64_32x32 | jax-cuda | 32 | 15 | 3.361 | 8208 | 0.000 | 4.261e-07 |
| slogdet_negative_f64_32x32 | jax-cuda-jit | 32 | 15 | 3.032 | 8208 | 0.000 | 4.261e-07 |
| slogdet_negative_f64_32x32 | pytorch-cuda | 32 | 15 | 3.707 | 8208 | 0.000 | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.527 | 8208 | 0.000 | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.927 | 8208 | 0.000 | 4.261e-07 |
| slogdet_phase_c32_32x32 | jax-cuda | 32 | 15 | 3.331 | 8204 | 0.000 | 0.0004085 |
| slogdet_phase_c32_32x32 | jax-cuda-jit | 32 | 15 | 3.206 | 8204 | 0.000 | 0.0004085 |
| slogdet_phase_c32_32x32 | pytorch-cuda | 16 | 15 | 2.029 | 8204 | 0.000 | 0.0004085 |
| slogdet_phase_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.741 | 8204 | 0.000 | 0.0004085 |
| slogdet_phase_c32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.125 | 8204 | 0.000 | 0.0004085 |
| slogdet_phase_c64_32x32 | jax-cuda | 32 | 15 | 3.511 | 16408 | 0.000 | 5.706e-07 |
| slogdet_phase_c64_32x32 | jax-cuda-jit | 16 | 15 | 1.747 | 16408 | 0.000 | 5.706e-07 |
| slogdet_phase_c64_32x32 | pytorch-cuda | 16 | 15 | 2.084 | 16408 | 0.000 | 5.706e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.690 | 16408 | 0.000 | 4.755e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.142 | 16408 | 0.000 | 4.755e-07 |
| slogdet_positive_c32_32x32 | jax-cuda | 32 | 15 | 3.348 | 8204 | 0.000 | 0.0004085 |
| slogdet_positive_c32_32x32 | jax-cuda-jit | 16 | 15 | 1.567 | 8204 | 0.000 | 0.0004085 |
| slogdet_positive_c32_32x32 | pytorch-cuda | 16 | 15 | 2.014 | 8204 | 0.000 | 0.0004085 |
| slogdet_positive_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.610 | 8204 | 0.000 | 0.0007148 |
| slogdet_positive_c32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.139 | 8204 | 0.000 | 0.0007148 |
| slogdet_positive_c64_32x32 | jax-cuda | 32 | 15 | 3.279 | 16408 | 0.000 | 4.755e-07 |
| slogdet_positive_c64_32x32 | jax-cuda-jit | 32 | 15 | 3.399 | 16408 | 0.000 | 4.755e-07 |
| slogdet_positive_c64_32x32 | pytorch-cuda | 16 | 15 | 2.086 | 16408 | 0.000 | 4.755e-07 |
| slogdet_positive_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.666 | 16408 | 0.000 | 3.804e-07 |
| slogdet_positive_c64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.405 | 16408 | 0.000 | 3.804e-07 |
| slogdet_positive_f32_32x32 | jax-cuda | 32 | 15 | 3.340 | 4104 | 0.000 | 0.000915 |
| slogdet_positive_f32_32x32 | jax-cuda-jit | 32 | 15 | 3.107 | 4104 | 0.000 | 0.000915 |
| slogdet_positive_f32_32x32 | pytorch-cuda | 32 | 15 | 3.930 | 4104 | 0.000 | 0.000915 |
| slogdet_positive_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.512 | 4104 | 0.000 | 0.000915 |
| slogdet_positive_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.088 | 4104 | 0.000 | 0.000915 |
| slogdet_positive_f64_32x32 | jax-cuda | 16 | 15 | 1.652 | 8208 | 0.000 | 4.261e-07 |
| slogdet_positive_f64_32x32 | jax-cuda-jit | 32 | 15 | 3.051 | 8208 | 0.000 | 4.261e-07 |
| slogdet_positive_f64_32x32 | pytorch-cuda | 32 | 15 | 3.703 | 8208 | 0.000 | 4.261e-07 |
| slogdet_positive_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.528 | 8208 | 0.000 | 4.261e-07 |
| slogdet_positive_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.922 | 8208 | 0.000 | 4.261e-07 |
| svd_full_square_c32_32x32 | jax-cuda | 2 | 15 | 1.981 | 24704 | 0.000 | 0.009079 |
| svd_full_square_c32_32x32 | jax-cuda-jit | 4 | 15 | 3.891 | 24704 | 0.000 | 0.009079 |
| svd_full_square_c32_32x32 | pytorch-cuda | 4 | 15 | 2.364 | 24704 | 0.000 | 0.011 |
| svd_full_square_c32_32x32 | tenferro-cuda-direct | 4 | 15 | 3.829 | 24704 | 0.000 | 0.009079 |
| svd_full_square_c32_32x32 | tenferro-cuda-eager | 2 | 15 | 2.041 | 24704 | 0.000 | 0.009079 |
| svd_full_square_c64_32x32 | jax-cuda | 2 | 15 | 3.676 | 49408 | 0.000 | 9.215e-06 |
| svd_full_square_c64_32x32 | jax-cuda-jit | 2 | 15 | 3.634 | 49408 | 0.000 | 9.215e-06 |
| svd_full_square_c64_32x32 | pytorch-cuda | 2 | 15 | 2.464 | 49408 | 0.000 | 9.992e-06 |
| svd_full_square_c64_32x32 | tenferro-cuda-direct | 2 | 15 | 3.574 | 49408 | 0.000 | 9.215e-06 |
| svd_full_square_c64_32x32 | tenferro-cuda-eager | 2 | 15 | 3.697 | 49408 | 0.000 | 9.215e-06 |
| svd_full_square_f32_32x32 | jax-cuda | 4 | 15 | 2.708 | 12416 | 0.000 | 0.008731 |
| svd_full_square_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.680 | 12416 | 0.000 | 0.008731 |
| svd_full_square_f32_32x32 | pytorch-cuda | 8 | 15 | 2.805 | 12416 | 0.000 | 0.005571 |
| svd_full_square_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.617 | 12416 | 0.000 | 0.008731 |
| svd_full_square_f32_32x32 | tenferro-cuda-eager | 4 | 15 | 2.746 | 12416 | 0.000 | 0.008731 |
| svd_full_square_f64_32x32 | jax-cuda | 4 | 15 | 3.850 | 24832 | 0.000 | 1.008e-05 |
| svd_full_square_f64_32x32 | jax-cuda-jit | 2 | 15 | 1.939 | 24832 | 0.000 | 1.008e-05 |
| svd_full_square_f64_32x32 | pytorch-cuda | 4 | 15 | 2.430 | 24832 | 0.000 | 7.216e-06 |
| svd_full_square_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 3.739 | 24832 | 0.000 | 1.007e-05 |
| svd_full_square_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 3.916 | 24832 | 0.000 | 1.007e-05 |
| svd_full_tall_c32_64x32 | jax-cuda | 2 | 15 | 2.838 | 57472 | 0.000 | 0.009498 |
| svd_full_tall_c32_64x32 | jax-cuda-jit | 2 | 15 | 2.806 | 57472 | 0.000 | 0.009498 |
| svd_full_tall_c32_64x32 | pytorch-cuda | 2 | 15 | 2.800 | 57472 | 0.000 | 0.008819 |
| svd_full_tall_c32_64x32 | tenferro-cuda-direct | 2 | 15 | 2.767 | 57472 | 0.000 | 0.009498 |
| svd_full_tall_c32_64x32 | tenferro-cuda-eager | 2 | 15 | 2.872 | 57472 | 0.000 | 0.009498 |
| svd_full_tall_c64_64x32 | jax-cuda | 1 | 15 | 2.845 | 114944 | 0.000 | 1.243e-05 |
| svd_full_tall_c64_64x32 | jax-cuda-jit | 1 | 15 | 2.355 | 114944 | 0.000 | 1.243e-05 |
| svd_full_tall_c64_64x32 | pytorch-cuda | 1 | 15 | 2.280 | 114944 | 0.000 | 1.188e-05 |
| svd_full_tall_c64_64x32 | tenferro-cuda-direct | 1 | 15 | 2.273 | 114944 | 0.000 | 1.221e-05 |
| svd_full_tall_c64_64x32 | tenferro-cuda-eager | 1 | 15 | 2.328 | 114944 | 0.000 | 1.221e-05 |
| svd_full_tall_f32_64x32 | jax-cuda | 2 | 15 | 2.117 | 28800 | 0.000 | 0.007661 |
| svd_full_tall_f32_64x32 | jax-cuda-jit | 2 | 15 | 2.083 | 28800 | 0.000 | 0.007661 |
| svd_full_tall_f32_64x32 | pytorch-cuda | 2 | 15 | 2.081 | 28800 | 0.000 | 0.007658 |
| svd_full_tall_f32_64x32 | tenferro-cuda-direct | 2 | 15 | 2.082 | 28800 | 0.000 | 0.007661 |
| svd_full_tall_f32_64x32 | tenferro-cuda-eager | 2 | 15 | 2.163 | 28800 | 0.000 | 0.007661 |
| svd_full_tall_f64_64x32 | jax-cuda | 2 | 15 | 3.005 | 57600 | 0.000 | 9.548e-06 |
| svd_full_tall_f64_64x32 | jax-cuda-jit | 2 | 15 | 2.956 | 57600 | 0.000 | 9.548e-06 |
| svd_full_tall_f64_64x32 | pytorch-cuda | 2 | 15 | 2.934 | 57600 | 0.000 | 9.215e-06 |
| svd_full_tall_f64_64x32 | tenferro-cuda-direct | 2 | 15 | 2.881 | 57600 | 0.000 | 9.548e-06 |
| svd_full_tall_f64_64x32 | tenferro-cuda-eager | 2 | 15 | 2.963 | 57600 | 0.000 | 9.548e-06 |
| svd_full_wide_c32_32x64 | jax-cuda | 2 | 15 | 2.879 | 57472 | 0.000 | 0.008495 |
| svd_full_wide_c32_32x64 | jax-cuda-jit | 2 | 15 | 2.857 | 57472 | 0.000 | 0.008495 |
| svd_full_wide_c32_32x64 | pytorch-cuda | 2 | 15 | 2.810 | 57472 | 0.000 | 0.008905 |
| svd_full_wide_c32_32x64 | tenferro-cuda-direct | 2 | 15 | 2.758 | 57472 | 0.000 | 0.008495 |
| svd_full_wide_c32_32x64 | tenferro-cuda-eager | 2 | 15 | 2.866 | 57472 | 0.000 | 0.008495 |
| svd_full_wide_c64_32x64 | jax-cuda | 1 | 15 | 2.550 | 114944 | 0.000 | 1.288e-05 |
| svd_full_wide_c64_32x64 | jax-cuda-jit | 1 | 15 | 2.534 | 114944 | 0.000 | 1.288e-05 |
| svd_full_wide_c64_32x64 | pytorch-cuda | 1 | 15 | 2.416 | 114944 | 0.000 | 1.299e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-direct | 1 | 15 | 2.386 | 114944 | 0.000 | 1.31e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-eager | 1 | 15 | 2.476 | 114944 | 0.000 | 1.31e-05 |
| svd_full_wide_f32_32x64 | jax-cuda | 2 | 15 | 2.280 | 28800 | 0.000 | 0.005157 |
| svd_full_wide_f32_32x64 | jax-cuda-jit | 2 | 15 | 2.257 | 28800 | 0.000 | 0.005157 |
| svd_full_wide_f32_32x64 | pytorch-cuda | 2 | 15 | 2.217 | 28800 | 0.000 | 0.005115 |
| svd_full_wide_f32_32x64 | tenferro-cuda-direct | 2 | 15 | 2.217 | 28800 | 0.000 | 0.005157 |
| svd_full_wide_f32_32x64 | tenferro-cuda-eager | 2 | 15 | 2.331 | 28800 | 0.000 | 0.005157 |
| svd_full_wide_f64_32x64 | jax-cuda | 2 | 15 | 3.065 | 57600 | 0.000 | 9.892e-06 |
| svd_full_wide_f64_32x64 | jax-cuda-jit | 2 | 15 | 3.008 | 57600 | 0.000 | 9.892e-06 |
| svd_full_wide_f64_32x64 | pytorch-cuda | 2 | 15 | 3.108 | 57600 | 0.000 | 7.327e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-direct | 2 | 15 | 2.926 | 57600 | 0.000 | 9.838e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-eager | 2 | 15 | 3.036 | 57600 | 0.000 | 9.838e-06 |
| svdvals_f32_32x32 | jax-cuda | 4 | 15 | 2.867 | 4224 | 0.000 | 0.003338 |
| svdvals_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.894 | 4224 | 0.000 | 0.003338 |
| svdvals_f32_32x32 | pytorch-cuda | 8 | 15 | 3.195 | 4224 | 0.000 | 0.007749 |
| svdvals_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.748 | 4224 | 0.000 | 0.003338 |
| svdvals_f32_32x32 | tenferro-cuda-eager | — | — | — | 4224 | unavailable | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_32x32 | jax-cuda | 2 | 15 | 2.236 | 8448 | 0.000 | 4.546e-06 |
| svdvals_f64_32x32 | jax-cuda-jit | 2 | 15 | 2.204 | 8448 | 0.000 | 4.546e-06 |
| svdvals_f64_32x32 | pytorch-cuda | 4 | 15 | 2.887 | 8448 | 0.000 | 3.387e-06 |
| svdvals_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 3.890 | 8448 | 0.000 | 4.546e-06 |
| svdvals_f64_32x32 | tenferro-cuda-eager | — | — | — | 8448 | unavailable | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| triangular_solve_f32_32x32 | jax-cuda | 32 | 15 | 2.679 | 6144 | 0.000 | 0.0001546 |
| triangular_solve_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.433 | 6144 | 0.000 | 0.0001546 |
| triangular_solve_f32_32x32 | pytorch-cuda | 128 | 15 | 3.479 | 6144 | 0.000 | 9.242e-05 |
| triangular_solve_f32_32x32 | tenferro-cuda-direct | 128 | 15 | 2.280 | 6144 | 0.000 | 0.0001546 |
| triangular_solve_f32_32x32 | tenferro-cuda-eager | 32 | 15 | 2.208 | 6144 | 0.000 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda | 16 | 15 | 2.272 | 12288 | 0.000 | 1.609e-07 |
| triangular_solve_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.505 | 12288 | 0.000 | 1.609e-07 |
| triangular_solve_f64_32x32 | pytorch-cuda | 128 | 15 | 3.613 | 12288 | 0.000 | 1.415e-07 |
| triangular_solve_f64_32x32 | tenferro-cuda-direct | 128 | 15 | 2.432 | 12288 | 0.000 | 1.609e-07 |
| triangular_solve_f64_32x32 | tenferro-cuda-eager | 32 | 15 | 2.073 | 12288 | 0.000 | 1.609e-07 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api_latency
suite_file: benchmarks/gpu/linalg_public_api_latency.yaml
timestamp: '2026-11-05T03:43:18Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: a000e5c4929d5bebb91e783c329b27106baed12e
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: b3641a0261e79a0e6d580c29e1e45784f36483db
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
  artifact_cache:
    path: /home/vscode/.cache/tenferro-benchmark/cuda-artifacts/12.9.86/sm_80
    regime: warm
gpu_memory_calibration:
  kind: device_copy
  method: torch.Tensor.copy_ between two contiguous device buffers, one copy per sample,
    device synchronized before and after the clock
  device_ordinal: 0
  device_name: NVIDIA A100 80GB PCIe
  buffer_bytes: 67108864
  traffic_bytes_per_copy: 134217728
  traffic_model: one read plus one write of buffer_bytes per copy
  warmups: 3
  sample_ms:
  - 0.092423
  - 0.093613
  - 0.097103
  - 0.095083
  - 0.092633
  - 0.094913
  - 0.093994
  - 0.092273
  - 0.095343
  - 0.096063
  - 0.093783
  - 0.095084
  - 0.093243
  - 0.093213
  - 0.093153
  median_ms: 0.093783
  bytes_per_second: 1431151999829.3933
  torch_version: 2.12.0+cu130
  cuda_version: '13.0'
  timestamp: '2026-11-05T03:43:25.215068+00:00'
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
    src/bin/benchmark_gpu_public_api.rs: 4e8f3c519e9210dbc8725dac34da23e543bf462d10c4a202c61e80b59ed108dd
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    src/bin/benchmark_gpu_public_api/multi.rs: 949eadd6272c17ba4e380a228547a03608d328a5ef852eb95294731c288f24d9
    src/bin/benchmark_gpu_public_api/plan.rs: 70fce8a6e9ed7518779a51a9d8692e010fa1e470e95eee0ce5276f595737fde8
    src/bin/benchmark_gpu_public_api/state.rs: a8bbf8985bbc58c435e3c4dda7135161a141a90a10fa50d4f95128397ce949cb
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 1919e33cc5f2d0208901837f80cb7c0600426462f998be6dd933062d9f6bb345
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 74b2e339d743af9c67199edacd504625a5a1f51908c06517d517837507947d76
    data/results/nvidia-gpu/gpu/linalg_public_api_latency/20261105_033930/fixtures/manifest.json: b79e18fe5dcfdee2c281cd060a333b64492775ceb0b8209a39523fa389bb160d
    target/release/benchmark_gpu_public_api: d048341025f978c4b0a4153b8b9e99f23da3db25601102f1c0171ce33bdffa79
  reference_numpy_version: 2.4.2
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| cholesky_f32_32x32 | tenferro-cuda-direct | cholesky_f32_32x32 untimed 111.6 ms is 307% of the 36.4 ms timed region (budget 20%) |
| svdvals_f32_32x32 | tenferro-cuda-direct | svdvals_f32_32x32 untimed 11.8 ms is 29% of the 41.2 ms timed region (budget 20%) |
| svdvals_f64_32x32 | tenferro-cuda-direct | svdvals_f64_32x32 untimed 13.2 ms is 23% of the 58.4 ms timed region (budget 20%) |
| eigvalsh_f32_32x32 | tenferro-cuda-direct | eigvalsh_f32_32x32 untimed 25.4 ms is 47% of the 53.8 ms timed region (budget 20%) |
| det_f32_32x32 | tenferro-cuda-direct | det_f32_32x32 untimed 20.5 ms is 46% of the 44.6 ms timed region (budget 20%) |
| triangular_solve_f32_32x32 | tenferro-cuda-direct | triangular_solve_f32_32x32 untimed 8.8 ms is 26% of the 34.2 ms timed region (budget 20%) |
| triangular_solve_f64_32x32 | tenferro-cuda-direct | triangular_solve_f64_32x32 untimed 9.9 ms is 27% of the 36.5 ms timed region (budget 20%) |
| svd_full_square_f32_32x32 | tenferro-cuda-direct | svd_full_square_f32_32x32 untimed 9.5 ms is 24% of the 39.3 ms timed region (budget 20%) |
| svd_full_tall_f32_64x32 | tenferro-cuda-direct | svd_full_tall_f32_64x32 untimed 71.1 ms is 228% of the 31.2 ms timed region (budget 20%) |
| svd_full_wide_f32_32x64 | tenferro-cuda-direct | svd_full_wide_f32_32x64 untimed 11.6 ms is 35% of the 33.2 ms timed region (budget 20%) |
| svd_full_square_f64_32x32 | tenferro-cuda-direct | svd_full_square_f64_32x32 untimed 13.0 ms is 23% of the 56.2 ms timed region (budget 20%) |
| svd_full_tall_f64_64x32 | tenferro-cuda-direct | svd_full_tall_f64_64x32 untimed 40.8 ms is 94% of the 43.3 ms timed region (budget 20%) |
| svd_full_wide_f64_32x64 | tenferro-cuda-direct | svd_full_wide_f64_32x64 untimed 14.8 ms is 34% of the 43.8 ms timed region (budget 20%) |
| svd_full_square_c32_32x32 | tenferro-cuda-direct | svd_full_square_c32_32x32 untimed 13.5 ms is 23% of the 57.4 ms timed region (budget 20%) |
| svd_full_tall_c32_64x32 | tenferro-cuda-direct | svd_full_tall_c32_64x32 untimed 22.0 ms is 53% of the 41.5 ms timed region (budget 20%) |
| svd_full_wide_c32_32x64 | tenferro-cuda-direct | svd_full_wide_c32_32x64 untimed 14.5 ms is 35% of the 41.4 ms timed region (budget 20%) |
| svd_full_square_c64_32x32 | tenferro-cuda-direct | svd_full_square_c64_32x32 untimed 17.1 ms is 32% of the 53.6 ms timed region (budget 20%) |
| svd_full_tall_c64_64x32 | tenferro-cuda-direct | svd_full_tall_c64_64x32 untimed 58.9 ms is 173% of the 34.1 ms timed region (budget 20%) |
| svd_full_wide_c64_32x64 | tenferro-cuda-direct | svd_full_wide_c64_32x64 untimed 19.2 ms is 54% of the 35.8 ms timed region (budget 20%) |
| householder_qr_square_f32_32x32 | tenferro-cuda-direct | householder_qr_square_f32_32x32 untimed 9.0 ms is 32% of the 28.2 ms timed region (budget 20%) |
| householder_qr_tall_f32_64x32 | tenferro-cuda-direct | householder_qr_tall_f32_64x32 untimed 7.7 ms is 20% of the 37.9 ms timed region (budget 20%) |
| householder_qr_tall_c32_64x32 | tenferro-cuda-direct | householder_qr_tall_c32_64x32 untimed 8.9 ms is 20% of the 43.7 ms timed region (budget 20%) |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-direct | householder_from_factors_default_f32_64x32 untimed 9.7 ms is 23% of the 42.0 ms timed region (budget 20%) |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-direct | householder_from_factors_default_f64_64x32 untimed 9.4 ms is 21% of the 44.7 ms timed region (budget 20%) |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-direct | householder_from_factors_default_c32_64x32 untimed 10.2 ms is 23% of the 45.1 ms timed region (budget 20%) |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-direct | householder_from_factors_default_c64_64x32 untimed 10.4 ms is 22% of the 47.2 ms timed region (budget 20%) |
| lu_wide_c32_s32 | tenferro-cuda-direct | lu_wide_c32_s32 untimed 11.2 ms is 26% of the 42.5 ms timed region (budget 20%) |
| lu_wide_c64_s32 | tenferro-cuda-direct | lu_wide_c64_s32 untimed 12.4 ms is 28% of the 44.3 ms timed region (budget 20%) |
| cholesky_f32_32x32 | tenferro-cuda-eager | cholesky_f32_32x32 untimed 123.8 ms is 229% of the 54.1 ms timed region (budget 20%) |
| eigvalsh_f32_32x32 | tenferro-cuda-eager | eigvalsh_f32_32x32 untimed 22.9 ms is 73% of the 31.2 ms timed region (budget 20%) |
| svd_full_square_f32_32x32 | tenferro-cuda-eager | svd_full_square_f32_32x32 untimed 9.7 ms is 24% of the 41.4 ms timed region (budget 20%) |
| svd_full_tall_f32_64x32 | tenferro-cuda-eager | svd_full_tall_f32_64x32 untimed 19.4 ms is 60% of the 32.5 ms timed region (budget 20%) |
| svd_full_wide_f32_32x64 | tenferro-cuda-eager | svd_full_wide_f32_32x64 untimed 12.2 ms is 35% of the 34.9 ms timed region (budget 20%) |
| svd_full_square_f64_32x32 | tenferro-cuda-eager | svd_full_square_f64_32x32 untimed 13.4 ms is 23% of the 59.0 ms timed region (budget 20%) |
| svd_full_tall_f64_64x32 | tenferro-cuda-eager | svd_full_tall_f64_64x32 untimed 18.2 ms is 41% of the 44.5 ms timed region (budget 20%) |
| svd_full_wide_f64_32x64 | tenferro-cuda-eager | svd_full_wide_f64_32x64 untimed 15.4 ms is 34% of the 45.6 ms timed region (budget 20%) |
| svd_full_square_c32_32x32 | tenferro-cuda-eager | svd_full_square_c32_32x32 untimed 10.1 ms is 33% of the 30.6 ms timed region (budget 20%) |
| svd_full_tall_c32_64x32 | tenferro-cuda-eager | svd_full_tall_c32_64x32 untimed 23.9 ms is 55% of the 43.1 ms timed region (budget 20%) |
| svd_full_wide_c32_32x64 | tenferro-cuda-eager | svd_full_wide_c32_32x64 untimed 15.5 ms is 36% of the 43.1 ms timed region (budget 20%) |
| svd_full_square_c64_32x32 | tenferro-cuda-eager | svd_full_square_c64_32x32 untimed 17.7 ms is 32% of the 55.5 ms timed region (budget 20%) |
| svd_full_tall_c64_64x32 | tenferro-cuda-eager | svd_full_tall_c64_64x32 untimed 60.7 ms is 174% of the 35.0 ms timed region (budget 20%) |
| svd_full_wide_c64_32x64 | tenferro-cuda-eager | svd_full_wide_c64_32x64 untimed 20.0 ms is 54% of the 37.1 ms timed region (budget 20%) |
| householder_qr_square_f32_32x32 | tenferro-cuda-eager | householder_qr_square_f32_32x32 untimed 11.2 ms is 29% of the 38.4 ms timed region (budget 20%) |
| householder_qr_tall_f64_64x32 | tenferro-cuda-eager | householder_qr_tall_f64_64x32 untimed 10.7 ms is 20% of the 52.8 ms timed region (budget 20%) |
| householder_qr_wide_c64_32x64 | tenferro-cuda-eager | householder_qr_wide_c64_32x64 untimed 7.0 ms is 22% of the 31.2 ms timed region (budget 20%) |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-eager | householder_from_factors_default_f32_64x32 untimed 11.0 ms is 22% of the 50.7 ms timed region (budget 20%) |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-eager | householder_from_factors_default_f64_64x32 untimed 11.3 ms is 21% of the 53.4 ms timed region (budget 20%) |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-eager | householder_from_factors_default_c32_64x32 untimed 11.5 ms is 22% of the 52.7 ms timed region (budget 20%) |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-eager | householder_from_factors_default_c64_64x32 untimed 11.8 ms is 20% of the 58.1 ms timed region (budget 20%) |
| lu_wide_c32_s32 | tenferro-cuda-eager | lu_wide_c32_s32 untimed 13.0 ms is 25% of the 52.7 ms timed region (budget 20%) |
| lu_wide_c64_s32 | tenferro-cuda-eager | lu_wide_c64_s32 untimed 14.3 ms is 26% of the 53.8 ms timed region (budget 20%) |
| cholesky_f32_32x32 | pytorch-cuda | cholesky_f32_32x32 untimed 167.2 ms is 285% of the 58.7 ms timed region (budget 20%) |
| cholesky_f64_32x32 | pytorch-cuda | cholesky_f64_32x32 untimed 6.6 ms is 22% of the 29.9 ms timed region (budget 20%) |
| svdvals_f32_32x32 | pytorch-cuda | svdvals_f32_32x32 untimed 14.9 ms is 31% of the 48.1 ms timed region (budget 20%) |
| svdvals_f64_32x32 | pytorch-cuda | svdvals_f64_32x32 untimed 10.3 ms is 24% of the 43.3 ms timed region (budget 20%) |
| eigvalsh_f32_32x32 | pytorch-cuda | eigvalsh_f32_32x32 untimed 32.4 ms is 61% of the 53.5 ms timed region (budget 20%) |
| eigvalsh_f64_32x32 | pytorch-cuda | eigvalsh_f64_32x32 untimed 6.9 ms is 22% of the 31.7 ms timed region (budget 20%) |
| pinv_f32_32x32 | pytorch-cuda | pinv_f32_32x32 untimed 112.4 ms is 192% of the 58.6 ms timed region (budget 20%) |
| pinv_f64_32x32 | pytorch-cuda | pinv_f64_32x32 untimed 35.9 ms is 72% of the 49.7 ms timed region (budget 20%) |
| inv_f32_32x32 | pytorch-cuda | inv_f32_32x32 untimed 47.9 ms is 109% of the 44.0 ms timed region (budget 20%) |
| det_f32_32x32 | pytorch-cuda | det_f32_32x32 untimed 64.2 ms is 135% of the 47.6 ms timed region (budget 20%) |
| norm_f32_32x32 | pytorch-cuda | norm_f32_32x32 untimed 40.5 ms is 88% of the 46.0 ms timed region (budget 20%) |
| norm_f64_32x32 | pytorch-cuda | norm_f64_32x32 untimed 9.9 ms is 20% of the 49.0 ms timed region (budget 20%) |
| svd_full_square_f32_32x32 | pytorch-cuda | svd_full_square_f32_32x32 untimed 11.3 ms is 27% of the 42.0 ms timed region (budget 20%) |
| svd_full_tall_f32_64x32 | pytorch-cuda | svd_full_tall_f32_64x32 untimed 28.5 ms is 91% of the 31.3 ms timed region (budget 20%) |
| svd_full_wide_f32_32x64 | pytorch-cuda | svd_full_wide_f32_32x64 untimed 16.3 ms is 49% of the 33.2 ms timed region (budget 20%) |
| slogdet_positive_f32_32x32 | pytorch-cuda | slogdet_positive_f32_32x32 untimed 31.7 ms is 54% of the 58.8 ms timed region (budget 20%) |
| svd_full_square_f64_32x32 | pytorch-cuda | svd_full_square_f64_32x32 untimed 11.2 ms is 31% of the 36.5 ms timed region (budget 20%) |
| svd_full_tall_f64_64x32 | pytorch-cuda | svd_full_tall_f64_64x32 untimed 23.6 ms is 53% of the 44.1 ms timed region (budget 20%) |
| svd_full_wide_f64_32x64 | pytorch-cuda | svd_full_wide_f64_32x64 untimed 20.3 ms is 43% of the 46.6 ms timed region (budget 20%) |
| svd_full_square_c32_32x32 | pytorch-cuda | svd_full_square_c32_32x32 untimed 12.5 ms is 35% of the 35.5 ms timed region (budget 20%) |
| svd_full_tall_c32_64x32 | pytorch-cuda | svd_full_tall_c32_64x32 untimed 26.8 ms is 64% of the 42.0 ms timed region (budget 20%) |
| svd_full_wide_c32_32x64 | pytorch-cuda | svd_full_wide_c32_32x64 untimed 19.3 ms is 46% of the 42.2 ms timed region (budget 20%) |
| slogdet_positive_c32_32x32 | pytorch-cuda | slogdet_positive_c32_32x32 untimed 9.5 ms is 31% of the 30.2 ms timed region (budget 20%) |
| svd_full_square_c64_32x32 | pytorch-cuda | svd_full_square_c64_32x32 untimed 14.6 ms is 39% of the 37.0 ms timed region (budget 20%) |
| svd_full_tall_c64_64x32 | pytorch-cuda | svd_full_tall_c64_64x32 untimed 65.8 ms is 192% of the 34.2 ms timed region (budget 20%) |
| svd_full_wide_c64_32x64 | pytorch-cuda | svd_full_wide_c64_32x64 untimed 23.7 ms is 65% of the 36.3 ms timed region (budget 20%) |
| slogdet_positive_c64_32x32 | pytorch-cuda | slogdet_positive_c64_32x32 untimed 10.5 ms is 33% of the 31.3 ms timed region (budget 20%) |
| householder_qr_square_f32_32x32 | pytorch-cuda | householder_qr_square_f32_32x32 untimed 9.8 ms is 24% of the 40.0 ms timed region (budget 20%) |
| householder_qr_tall_f32_64x32 | pytorch-cuda | householder_qr_tall_f32_64x32 untimed 9.5 ms is 32% of the 30.2 ms timed region (budget 20%) |
| householder_qr_wide_f32_32x64 | pytorch-cuda | householder_qr_wide_f32_32x64 untimed 9.0 ms is 30% of the 30.3 ms timed region (budget 20%) |
| householder_qr_square_f64_32x32 | pytorch-cuda | householder_qr_square_f64_32x32 untimed 10.4 ms is 22% of the 47.7 ms timed region (budget 20%) |
| householder_qr_tall_f64_64x32 | pytorch-cuda | householder_qr_tall_f64_64x32 untimed 9.6 ms is 29% of the 33.5 ms timed region (budget 20%) |
| householder_qr_wide_f64_32x64 | pytorch-cuda | householder_qr_wide_f64_32x64 untimed 9.2 ms is 27% of the 33.7 ms timed region (budget 20%) |
| householder_qr_square_c32_32x32 | pytorch-cuda | householder_qr_square_c32_32x32 untimed 10.0 ms is 22% of the 45.6 ms timed region (budget 20%) |
| householder_qr_tall_c32_64x32 | pytorch-cuda | householder_qr_tall_c32_64x32 untimed 10.4 ms is 30% of the 34.9 ms timed region (budget 20%) |
| householder_qr_wide_c32_32x64 | pytorch-cuda | householder_qr_wide_c32_32x64 untimed 10.0 ms is 29% of the 35.1 ms timed region (budget 20%) |
| householder_qr_square_c64_32x32 | pytorch-cuda | householder_qr_square_c64_32x32 untimed 8.3 ms is 27% of the 30.5 ms timed region (budget 20%) |
| householder_qr_tall_c64_64x32 | pytorch-cuda | householder_qr_tall_c64_64x32 untimed 11.3 ms is 27% of the 41.7 ms timed region (budget 20%) |
| householder_qr_wide_c64_32x64 | pytorch-cuda | householder_qr_wide_c64_32x64 untimed 11.1 ms is 25% of the 44.0 ms timed region (budget 20%) |
| slogdet_phase_c64_32x32 | pytorch-cuda | slogdet_phase_c64_32x32 untimed 6.4 ms is 20% of the 31.5 ms timed region (budget 20%) |
| householder_r_default_f32_64x32 | pytorch-cuda | householder_r_default_f32_64x32 untimed 10.2 ms is 28% of the 36.2 ms timed region (budget 20%) |
| householder_q_columns_prefix_f32_64x32 | pytorch-cuda | householder_q_columns_prefix_f32_64x32 untimed 6.4 ms is 21% of the 30.7 ms timed region (budget 20%) |
| householder_q_columns_complement_f32_64x32 | pytorch-cuda | householder_q_columns_complement_f32_64x32 untimed 8.7 ms is 29% of the 30.4 ms timed region (budget 20%) |
| householder_r_default_f64_64x32 | pytorch-cuda | householder_r_default_f64_64x32 untimed 10.1 ms is 27% of the 37.2 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_64x32 | pytorch-cuda | householder_q_columns_prefix_f64_64x32 untimed 6.7 ms is 21% of the 32.6 ms timed region (budget 20%) |
| householder_q_columns_complement_f64_64x32 | pytorch-cuda | householder_q_columns_complement_f64_64x32 untimed 9.1 ms is 27% of the 34.2 ms timed region (budget 20%) |
| householder_r_default_c32_64x32 | pytorch-cuda | householder_r_default_c32_64x32 untimed 10.8 ms is 27% of the 39.9 ms timed region (budget 20%) |
| householder_q_columns_prefix_c32_64x32 | pytorch-cuda | householder_q_columns_prefix_c32_64x32 untimed 6.3 ms is 22% of the 28.8 ms timed region (budget 20%) |
| householder_q_columns_complement_c32_64x32 | pytorch-cuda | householder_q_columns_complement_c32_64x32 untimed 9.0 ms is 28% of the 32.4 ms timed region (budget 20%) |
| householder_r_default_c64_64x32 | pytorch-cuda | householder_r_default_c64_64x32 untimed 9.8 ms is 28% of the 34.5 ms timed region (budget 20%) |
| householder_q_columns_complement_c64_64x32 | pytorch-cuda | householder_q_columns_complement_c64_64x32 untimed 9.2 ms is 26% of the 34.8 ms timed region (budget 20%) |
| cholesky_f32_32x32 | jax-cuda | cholesky_f32_32x32 untimed 451.7 ms is 954% of the 47.4 ms timed region (budget 20%) |
| cholesky_f64_32x32 | jax-cuda | cholesky_f64_32x32 untimed 124.6 ms is 526% of the 23.7 ms timed region (budget 20%) |
| svdvals_f32_32x32 | jax-cuda | svdvals_f32_32x32 untimed 120.7 ms is 269% of the 44.8 ms timed region (budget 20%) |
| svdvals_f64_32x32 | jax-cuda | svdvals_f64_32x32 untimed 111.4 ms is 330% of the 33.8 ms timed region (budget 20%) |
| eigvalsh_f32_32x32 | jax-cuda | eigvalsh_f32_32x32 untimed 87.0 ms is 250% of the 34.8 ms timed region (budget 20%) |
| eigvalsh_f64_32x32 | jax-cuda | eigvalsh_f64_32x32 untimed 75.3 ms is 174% of the 43.2 ms timed region (budget 20%) |
| pinv_f32_32x32 | jax-cuda | pinv_f32_32x32 untimed 335.8 ms is 586% of the 57.3 ms timed region (budget 20%) |
| pinv_f64_32x32 | jax-cuda | pinv_f64_32x32 untimed 243.2 ms is 631% of the 38.5 ms timed region (budget 20%) |
| inv_f32_32x32 | jax-cuda | inv_f32_32x32 untimed 144.7 ms is 480% of the 30.2 ms timed region (budget 20%) |
| inv_f64_32x32 | jax-cuda | inv_f64_32x32 untimed 153.2 ms is 393% of the 39.0 ms timed region (budget 20%) |
| det_f32_32x32 | jax-cuda | det_f32_32x32 untimed 165.9 ms is 522% of the 31.8 ms timed region (budget 20%) |
| det_f64_32x32 | jax-cuda | det_f64_32x32 untimed 155.6 ms is 339% of the 45.9 ms timed region (budget 20%) |
| norm_f32_32x32 | jax-cuda | norm_f32_32x32 untimed 74.5 ms is 288% of the 25.9 ms timed region (budget 20%) |
| norm_f64_32x32 | jax-cuda | norm_f64_32x32 untimed 83.6 ms is 163% of the 51.1 ms timed region (budget 20%) |
| triangular_solve_f32_32x32 | jax-cuda | triangular_solve_f32_32x32 untimed 114.9 ms is 285% of the 40.3 ms timed region (budget 20%) |
| triangular_solve_f64_32x32 | jax-cuda | triangular_solve_f64_32x32 untimed 105.5 ms is 358% of the 29.5 ms timed region (budget 20%) |
| svd_full_square_f32_32x32 | jax-cuda | svd_full_square_f32_32x32 untimed 132.3 ms is 307% of the 43.0 ms timed region (budget 20%) |
| svd_full_tall_f32_64x32 | jax-cuda | svd_full_tall_f32_64x32 untimed 161.1 ms is 506% of the 31.9 ms timed region (budget 20%) |
| svd_full_wide_f32_32x64 | jax-cuda | svd_full_wide_f32_32x64 untimed 141.7 ms is 397% of the 35.7 ms timed region (budget 20%) |
| slogdet_positive_f32_32x32 | jax-cuda | slogdet_positive_f32_32x32 untimed 110.6 ms is 221% of the 50.1 ms timed region (budget 20%) |
| slogdet_negative_f32_32x32 | jax-cuda | slogdet_negative_f32_32x32 untimed 11.6 ms is 23% of the 50.4 ms timed region (budget 20%) |
| svd_full_square_f64_32x32 | jax-cuda | svd_full_square_f64_32x32 untimed 148.8 ms is 258% of the 57.6 ms timed region (budget 20%) |
| svd_full_tall_f64_64x32 | jax-cuda | svd_full_tall_f64_64x32 untimed 189.7 ms is 422% of the 45.0 ms timed region (budget 20%) |
| svd_full_wide_f64_32x64 | jax-cuda | svd_full_wide_f64_32x64 untimed 156.5 ms is 341% of the 45.9 ms timed region (budget 20%) |
| slogdet_positive_f64_32x32 | jax-cuda | slogdet_positive_f64_32x32 untimed 112.2 ms is 452% of the 24.8 ms timed region (budget 20%) |
| slogdet_negative_f64_32x32 | jax-cuda | slogdet_negative_f64_32x32 untimed 11.4 ms is 23% of the 50.1 ms timed region (budget 20%) |
| svd_full_square_c32_32x32 | jax-cuda | svd_full_square_c32_32x32 untimed 85.2 ms is 286% of the 29.8 ms timed region (budget 20%) |
| svd_full_tall_c32_64x32 | jax-cuda | svd_full_tall_c32_64x32 untimed 107.8 ms is 251% of the 43.0 ms timed region (budget 20%) |
| svd_full_wide_c32_32x64 | jax-cuda | svd_full_wide_c32_32x64 untimed 91.5 ms is 213% of the 43.1 ms timed region (budget 20%) |
| slogdet_positive_c32_32x32 | jax-cuda | slogdet_positive_c32_32x32 untimed 136.9 ms is 272% of the 50.4 ms timed region (budget 20%) |
| slogdet_negative_c32_32x32 | jax-cuda | slogdet_negative_c32_32x32 untimed 11.3 ms is 23% of the 48.7 ms timed region (budget 20%) |
| svd_full_square_c64_32x32 | jax-cuda | svd_full_square_c64_32x32 untimed 88.8 ms is 157% of the 56.5 ms timed region (budget 20%) |
| svd_full_tall_c64_64x32 | jax-cuda | svd_full_tall_c64_64x32 untimed 153.1 ms is 369% of the 41.5 ms timed region (budget 20%) |
| svd_full_wide_c64_32x64 | jax-cuda | svd_full_wide_c64_32x64 untimed 97.4 ms is 254% of the 38.3 ms timed region (budget 20%) |
| slogdet_positive_c64_32x32 | jax-cuda | slogdet_positive_c64_32x32 untimed 150.1 ms is 305% of the 49.2 ms timed region (budget 20%) |
| slogdet_negative_c64_32x32 | jax-cuda | slogdet_negative_c64_32x32 untimed 12.0 ms is 24% of the 50.8 ms timed region (budget 20%) |
| householder_qr_square_f32_32x32 | jax-cuda | householder_qr_square_f32_32x32 untimed 66.6 ms is 163% of the 40.8 ms timed region (budget 20%) |
| householder_qr_tall_f32_64x32 | jax-cuda | householder_qr_tall_f32_64x32 untimed 72.3 ms is 233% of the 31.0 ms timed region (budget 20%) |
| householder_qr_wide_f32_32x64 | jax-cuda | householder_qr_wide_f32_32x64 untimed 68.5 ms is 220% of the 31.2 ms timed region (budget 20%) |
| householder_qr_square_f64_32x32 | jax-cuda | householder_qr_square_f64_32x32 untimed 69.8 ms is 144% of the 48.6 ms timed region (budget 20%) |
| householder_qr_tall_f64_64x32 | jax-cuda | householder_qr_tall_f64_64x32 untimed 65.5 ms is 189% of the 34.6 ms timed region (budget 20%) |
| householder_qr_wide_f64_32x64 | jax-cuda | householder_qr_wide_f64_32x64 untimed 65.6 ms is 188% of the 34.9 ms timed region (budget 20%) |
| householder_qr_square_c32_32x32 | jax-cuda | householder_qr_square_c32_32x32 untimed 72.9 ms is 157% of the 46.5 ms timed region (budget 20%) |
| householder_qr_tall_c32_64x32 | jax-cuda | householder_qr_tall_c32_64x32 untimed 76.7 ms is 213% of the 36.1 ms timed region (budget 20%) |
| householder_qr_wide_c32_32x64 | jax-cuda | householder_qr_wide_c32_32x64 untimed 76.8 ms is 205% of the 37.4 ms timed region (budget 20%) |
| slogdet_phase_c32_32x32 | jax-cuda | slogdet_phase_c32_32x32 untimed 11.9 ms is 24% of the 49.9 ms timed region (budget 20%) |
| householder_qr_square_c64_32x32 | jax-cuda | householder_qr_square_c64_32x32 untimed 68.5 ms is 216% of the 31.6 ms timed region (budget 20%) |
| householder_qr_tall_c64_64x32 | jax-cuda | householder_qr_tall_c64_64x32 untimed 79.3 ms is 179% of the 44.4 ms timed region (budget 20%) |
| householder_qr_wide_c64_32x64 | jax-cuda | householder_qr_wide_c64_32x64 untimed 72.2 ms is 160% of the 45.1 ms timed region (budget 20%) |
| slogdet_phase_c64_32x32 | jax-cuda | slogdet_phase_c64_32x32 untimed 12.5 ms is 24% of the 52.6 ms timed region (budget 20%) |
| householder_r_default_f32_64x32 | jax-cuda | householder_r_default_f32_64x32 untimed 141.9 ms is 275% of the 51.7 ms timed region (budget 20%) |
| householder_q_columns_prefix_f32_64x32 | jax-cuda | householder_q_columns_prefix_f32_64x32 untimed 172.4 ms is 657% of the 26.2 ms timed region (budget 20%) |
| householder_r_default_f64_64x32 | jax-cuda | householder_r_default_f64_64x32 untimed 146.8 ms is 504% of the 29.1 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_64x32 | jax-cuda | householder_q_columns_prefix_f64_64x32 untimed 170.4 ms is 603% of the 28.3 ms timed region (budget 20%) |
| householder_r_default_c32_64x32 | jax-cuda | householder_r_default_c32_64x32 untimed 157.5 ms is 531% of the 29.7 ms timed region (budget 20%) |
| householder_q_columns_prefix_c32_64x32 | jax-cuda | householder_q_columns_prefix_c32_64x32 untimed 121.6 ms is 613% of the 19.8 ms timed region (budget 20%) |
| householder_r_default_c64_64x32 | jax-cuda | householder_r_default_c64_64x32 untimed 151.9 ms is 580% of the 26.2 ms timed region (budget 20%) |
| householder_q_columns_prefix_c64_64x32 | jax-cuda | householder_q_columns_prefix_c64_64x32 untimed 116.3 ms is 589% of the 19.7 ms timed region (budget 20%) |
| cholesky_f32_32x32 | jax-cuda-jit | cholesky_f32_32x32 untimed 425.5 ms is 1279% of the 33.3 ms timed region (budget 20%) |
| cholesky_f64_32x32 | jax-cuda-jit | cholesky_f64_32x32 untimed 152.1 ms is 385% of the 39.5 ms timed region (budget 20%) |
| svdvals_f32_32x32 | jax-cuda-jit | svdvals_f32_32x32 untimed 122.9 ms is 276% of the 44.6 ms timed region (budget 20%) |
| svdvals_f64_32x32 | jax-cuda-jit | svdvals_f64_32x32 untimed 121.7 ms is 368% of the 33.1 ms timed region (budget 20%) |
| eigvalsh_f32_32x32 | jax-cuda-jit | eigvalsh_f32_32x32 untimed 93.5 ms is 267% of the 35.0 ms timed region (budget 20%) |
| eigvalsh_f64_32x32 | jax-cuda-jit | eigvalsh_f64_32x32 untimed 75.5 ms is 175% of the 43.0 ms timed region (budget 20%) |
| pinv_f32_32x32 | jax-cuda-jit | pinv_f32_32x32 untimed 341.7 ms is 789% of the 43.3 ms timed region (budget 20%) |
| pinv_f64_32x32 | jax-cuda-jit | pinv_f64_32x32 untimed 231.9 ms is 743% of the 31.2 ms timed region (budget 20%) |
| inv_f32_32x32 | jax-cuda-jit | inv_f32_32x32 untimed 160.5 ms is 289% of the 55.5 ms timed region (budget 20%) |
| inv_f64_32x32 | jax-cuda-jit | inv_f64_32x32 untimed 151.1 ms is 270% of the 56.0 ms timed region (budget 20%) |
| det_f32_32x32 | jax-cuda-jit | det_f32_32x32 untimed 181.1 ms is 396% of the 45.7 ms timed region (budget 20%) |
| det_f64_32x32 | jax-cuda-jit | det_f64_32x32 untimed 150.6 ms is 370% of the 40.8 ms timed region (budget 20%) |
| norm_f32_32x32 | jax-cuda-jit | norm_f32_32x32 untimed 79.8 ms is 198% of the 40.3 ms timed region (budget 20%) |
| norm_f64_32x32 | jax-cuda-jit | norm_f64_32x32 untimed 89.6 ms is 221% of the 40.4 ms timed region (budget 20%) |
| triangular_solve_f32_32x32 | jax-cuda-jit | triangular_solve_f32_32x32 untimed 119.4 ms is 327% of the 36.5 ms timed region (budget 20%) |
| triangular_solve_f64_32x32 | jax-cuda-jit | triangular_solve_f64_32x32 untimed 126.9 ms is 338% of the 37.6 ms timed region (budget 20%) |
| svd_full_square_f32_32x32 | jax-cuda-jit | svd_full_square_f32_32x32 untimed 140.3 ms is 334% of the 42.0 ms timed region (budget 20%) |
| svd_full_tall_f32_64x32 | jax-cuda-jit | svd_full_tall_f32_64x32 untimed 158.0 ms is 503% of the 31.4 ms timed region (budget 20%) |
| svd_full_wide_f32_32x64 | jax-cuda-jit | svd_full_wide_f32_32x64 untimed 141.7 ms is 406% of the 34.9 ms timed region (budget 20%) |
| slogdet_positive_f32_32x32 | jax-cuda-jit | slogdet_positive_f32_32x32 untimed 114.9 ms is 247% of the 46.5 ms timed region (budget 20%) |
| slogdet_negative_f32_32x32 | jax-cuda-jit | slogdet_negative_f32_32x32 untimed 110.1 ms is 237% of the 46.4 ms timed region (budget 20%) |
| svd_full_square_f64_32x32 | jax-cuda-jit | svd_full_square_f64_32x32 untimed 133.7 ms is 434% of the 30.8 ms timed region (budget 20%) |
| svd_full_tall_f64_64x32 | jax-cuda-jit | svd_full_tall_f64_64x32 untimed 169.3 ms is 381% of the 44.4 ms timed region (budget 20%) |
| svd_full_wide_f64_32x64 | jax-cuda-jit | svd_full_wide_f64_32x64 untimed 138.2 ms is 306% of the 45.2 ms timed region (budget 20%) |
| slogdet_positive_f64_32x32 | jax-cuda-jit | slogdet_positive_f64_32x32 untimed 113.1 ms is 242% of the 46.7 ms timed region (budget 20%) |
| slogdet_negative_f64_32x32 | jax-cuda-jit | slogdet_negative_f64_32x32 untimed 103.5 ms is 227% of the 45.6 ms timed region (budget 20%) |
| svd_full_square_c32_32x32 | jax-cuda-jit | svd_full_square_c32_32x32 untimed 75.9 ms is 130% of the 58.4 ms timed region (budget 20%) |
| svd_full_tall_c32_64x32 | jax-cuda-jit | svd_full_tall_c32_64x32 untimed 110.7 ms is 263% of the 42.1 ms timed region (budget 20%) |
| svd_full_wide_c32_32x64 | jax-cuda-jit | svd_full_wide_c32_32x64 untimed 96.4 ms is 226% of the 42.6 ms timed region (budget 20%) |
| slogdet_positive_c32_32x32 | jax-cuda-jit | slogdet_positive_c32_32x32 untimed 126.0 ms is 534% of the 23.6 ms timed region (budget 20%) |
| slogdet_negative_c32_32x32 | jax-cuda-jit | slogdet_negative_c32_32x32 untimed 130.9 ms is 291% of the 45.0 ms timed region (budget 20%) |
| svd_full_square_c64_32x32 | jax-cuda-jit | svd_full_square_c64_32x32 untimed 90.7 ms is 166% of the 54.6 ms timed region (budget 20%) |
| svd_full_tall_c64_64x32 | jax-cuda-jit | svd_full_tall_c64_64x32 untimed 152.7 ms is 428% of the 35.7 ms timed region (budget 20%) |
| svd_full_wide_c64_32x64 | jax-cuda-jit | svd_full_wide_c64_32x64 untimed 105.6 ms is 278% of the 38.0 ms timed region (budget 20%) |
| slogdet_positive_c64_32x32 | jax-cuda-jit | slogdet_positive_c64_32x32 untimed 174.8 ms is 343% of the 51.0 ms timed region (budget 20%) |
| slogdet_negative_c64_32x32 | jax-cuda-jit | slogdet_negative_c64_32x32 untimed 160.6 ms is 489% of the 32.9 ms timed region (budget 20%) |
| householder_qr_square_f32_32x32 | jax-cuda-jit | householder_qr_square_f32_32x32 untimed 71.3 ms is 334% of the 21.3 ms timed region (budget 20%) |
| householder_qr_tall_f32_64x32 | jax-cuda-jit | householder_qr_tall_f32_64x32 untimed 72.6 ms is 235% of the 30.9 ms timed region (budget 20%) |
| householder_qr_wide_f32_32x64 | jax-cuda-jit | householder_qr_wide_f32_32x64 untimed 80.0 ms is 256% of the 31.2 ms timed region (budget 20%) |
| householder_qr_square_f64_32x32 | jax-cuda-jit | householder_qr_square_f64_32x32 untimed 63.5 ms is 131% of the 48.7 ms timed region (budget 20%) |
| householder_qr_tall_f64_64x32 | jax-cuda-jit | householder_qr_tall_f64_64x32 untimed 77.9 ms is 224% of the 34.7 ms timed region (budget 20%) |
| householder_qr_wide_f64_32x64 | jax-cuda-jit | householder_qr_wide_f64_32x64 untimed 67.8 ms is 191% of the 35.6 ms timed region (budget 20%) |
| householder_qr_square_c32_32x32 | jax-cuda-jit | householder_qr_square_c32_32x32 untimed 79.6 ms is 172% of the 46.4 ms timed region (budget 20%) |
| householder_qr_tall_c32_64x32 | jax-cuda-jit | householder_qr_tall_c32_64x32 untimed 76.1 ms is 211% of the 36.0 ms timed region (budget 20%) |
| householder_qr_wide_c32_32x64 | jax-cuda-jit | householder_qr_wide_c32_32x64 untimed 81.5 ms is 220% of the 37.0 ms timed region (budget 20%) |
| slogdet_phase_c32_32x32 | jax-cuda-jit | slogdet_phase_c32_32x32 untimed 131.4 ms is 272% of the 48.3 ms timed region (budget 20%) |
| householder_qr_square_c64_32x32 | jax-cuda-jit | householder_qr_square_c64_32x32 untimed 65.9 ms is 207% of the 31.8 ms timed region (budget 20%) |
| householder_qr_tall_c64_64x32 | jax-cuda-jit | householder_qr_tall_c64_64x32 untimed 82.7 ms is 190% of the 43.5 ms timed region (budget 20%) |
| householder_qr_wide_c64_32x64 | jax-cuda-jit | householder_qr_wide_c64_32x64 untimed 69.8 ms is 154% of the 45.3 ms timed region (budget 20%) |
| slogdet_phase_c64_32x32 | jax-cuda-jit | slogdet_phase_c64_32x32 untimed 155.7 ms is 489% of the 31.8 ms timed region (budget 20%) |
| householder_r_default_f32_64x32 | jax-cuda-jit | householder_r_default_f32_64x32 untimed 216.8 ms is 742% of the 29.2 ms timed region (budget 20%) |
| householder_q_columns_prefix_f32_64x32 | jax-cuda-jit | householder_q_columns_prefix_f32_64x32 untimed 168.7 ms is 429% of the 39.3 ms timed region (budget 20%) |
| householder_r_default_f64_64x32 | jax-cuda-jit | householder_r_default_f64_64x32 untimed 213.2 ms is 760% of the 28.0 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_64x32 | jax-cuda-jit | householder_q_columns_prefix_f64_64x32 untimed 177.2 ms is 451% of the 39.3 ms timed region (budget 20%) |
| householder_r_default_c32_64x32 | jax-cuda-jit | householder_r_default_c32_64x32 untimed 227.0 ms is 737% of the 30.8 ms timed region (budget 20%) |
| householder_q_columns_prefix_c32_64x32 | jax-cuda-jit | householder_q_columns_prefix_c32_64x32 untimed 124.0 ms is 441% of the 28.1 ms timed region (budget 20%) |
| householder_r_default_c64_64x32 | jax-cuda-jit | householder_r_default_c64_64x32 untimed 199.7 ms is 783% of the 25.5 ms timed region (budget 20%) |
| householder_q_columns_prefix_c64_64x32 | jax-cuda-jit | householder_q_columns_prefix_c64_64x32 untimed 128.0 ms is 246% of the 52.1 ms timed region (budget 20%) |