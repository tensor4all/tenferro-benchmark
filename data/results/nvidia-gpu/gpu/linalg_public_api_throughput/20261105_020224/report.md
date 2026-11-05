# gpu/linalg_public_api_throughput: CUDA public API benchmark

Measured cells: **308**. Unsupported cells: **102**, including **28** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

This run metadata records no device copy rate, so the records carry the logical bytes without a copy ratio.

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261105_020224/`.

Large-shape view of the CUDA public linalg API surface: the 82 cases whose largest matrix dimension is at least 256. `large` is a shape class, not a measured classification: no row is asserted to be compute- or bandwidth-bound, and a memory-bound row takes the same time here as at a small shape (`householder_r_default_f32` measured 34.698 µs at 512x256 and 35.223 µs at 64x32). These rows are never comparable with, or averaged together with, the per-call overhead rows of `gpu/linalg_public_api_latency` (tenferro-rs #2087). The same problems, protocol and backends are split by shape class between the two suites. Issue 155.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_512x512 | 284.479 (0.610) | 328.898 (3.186) | 274.490 (1.679) | 241.716 (0.828) | 238.594 (0.619) |
| cholesky_f64_512x512 | 2097.823 (10.976) | 2287.360 (15.640) | 327.850 (2.498) | 302.759 (3.541) | 300.060 (3.066) |
| svdvals_f32_512x512 | 31163.050 (10.995) | unsupported | 34100.574 (21.660) | 31235.694 (13.580) | 31285.436 (20.745) |
| svdvals_f64_512x512 | 77755.224 (27.146) | unsupported | 84429.970 (32.657) | 77663.226 (24.367) | 77836.693 (26.886) |
| eigvalsh_f32_512x512 | 5208.810 (8.011) | 5270.462 (10.545) | 5208.401 (7.391) | 5926.066 (39.691) | 5911.445 (10.806) |
| eigvalsh_f64_512x512 | 5588.264 (11.325) | 6196.684 (14.741) | 5556.683 (15.345) | 6119.482 (64.957) | 6032.290 (36.956) |
| pinv_f32_512x512 | unsupported | unsupported | 34277.170 (14.020) | 31528.005 (183.101) | 31398.450 (19.114) |
| pinv_f64_512x512 | unsupported | unsupported | 84627.347 (17.756) | 78119.022 (135.544) | 78078.801 (60.173) |
| inv_f32_512x512 | unsupported | unsupported | 1505.572 (4.653) | 1505.072 (10.765) | 1544.423 (15.103) |
| inv_f64_512x512 | unsupported | unsupported | 1800.402 (6.003) | 1839.634 (10.553) | 1796.327 (5.763) |
| det_f32_512x512 | 1647.517 (4.903) | 1707.774 (9.230) | 1172.921 (3.545) | 1210.917 (26.141) | 1229.978 (25.133) |
| det_f64_512x512 | 2566.869 (11.345) | 2660.553 (17.105) | 1316.880 (7.805) | 1411.659 (10.530) | 1394.068 (29.303) |
| norm_f32_512x512 | 95.461 (0.421) | 118.643 (0.279) | 10.817 (0.080) | 49.319 (15.120) | 52.308 (0.605) |
| norm_f64_512x512 | 95.593 (0.369) | 120.359 (0.629) | 10.804 (0.100) | 48.049 (0.513) | 56.079 (0.792) |
| triangular_solve_f32_512x512 | 169.935 (0.618) | 223.462 (0.647) | 120.503 (0.041) | 184.583 (4.664) | 182.656 (1.726) |
| triangular_solve_f64_512x512 | 194.502 (0.693) | 246.548 (1.899) | 307.061 (0.152) | 215.783 (3.834) | 215.922 (2.865) |
| svd_full_square_f32_256x256 | 10817.655 (10.470) | 10851.636 (14.765) | 11370.034 (12.145) | 10845.286 (27.780) | 10877.878 (32.336) |
| svd_full_tall_f32_512x256 | 11248.300 (14.831) | 11260.910 (16.841) | 11883.463 (22.236) | 11264.961 (13.912) | 11300.352 (24.276) |
| svd_full_wide_f32_256x512 | 10971.330 (11.285) | 10932.639 (11.025) | 11070.395 (14.815) | 10998.372 (17.716) | 11022.263 (21.817) |
| slogdet_positive_f32_256x256 | 781.452 (2.120) | 833.979 (5.289) | 530.831 (1.319) | 518.341 (11.690) | 528.498 (11.055) |
| slogdet_negative_f32_256x256 | 778.632 (3.465) | 834.952 (2.616) | 530.673 (0.747) | 536.036 (2.462) | 541.086 (4.900) |
| svd_full_square_f64_256x256 | 25818.925 (7.244) | 25789.924 (17.886) | 25711.213 (10.196) | 25808.126 (13.211) | 25837.267 (30.962) |
| svd_full_tall_f64_512x256 | 19611.180 (15.126) | 19650.262 (12.770) | 19816.747 (24.006) | 19640.351 (20.361) | 19614.581 (20.401) |
| svd_full_wide_f64_256x512 | 17730.395 (13.095) | 17792.567 (14.456) | 18191.361 (32.702) | 17815.079 (18.136) | 17818.119 (450.865) |
| slogdet_positive_f64_256x256 | 824.379 (2.245) | 875.178 (2.929) | 569.160 (0.642) | 559.779 (4.085) | 573.565 (13.804) |
| slogdet_negative_f64_256x256 | 823.094 (2.718) | 874.856 (2.731) | 567.250 (1.058) | 562.370 (3.786) | 576.265 (2.698) |
| svd_full_square_c32_256x256 | 20311.824 (3.854) | 20386.337 (15.346) | 22071.956 (9.236) | 20428.949 (14.571) | 20383.247 (14.496) |
| svd_full_tall_c32_512x256 | 19621.650 (9.726) | 19687.573 (15.556) | 20017.675 (22.736) | 19670.832 (23.281) | 19704.695 (41.281) |
| svd_full_wide_c32_256x512 | 19804.536 (6.961) | 19831.408 (9.880) | 20218.692 (17.265) | 19822.478 (30.386) | 19868.891 (24.335) |
| slogdet_positive_c32_256x256 | 832.021 (3.534) | 941.750 (13.704) | 576.582 (0.916) | 558.639 (2.841) | 559.937 (2.471) |
| slogdet_negative_c32_256x256 | 831.271 (3.723) | 941.040 (22.127) | 574.347 (1.239) | 579.533 (3.006) | 576.185 (2.946) |
| svd_full_square_c64_256x256 | 60065.572 (19.995) | 60133.815 (15.575) | 63621.528 (27.087) | 60193.050 (262.579) | 60253.483 (24.021) |
| svd_full_tall_c64_512x256 | 42704.470 (40.392) | 42704.741 (20.306) | 40996.053 (14.625) | 40692.483 (17.741) | 40769.446 (21.176) |
| svd_full_wide_c64_256x512 | 42831.345 (29.091) | 42886.087 (17.015) | 41572.943 (20.291) | 40803.157 (21.135) | 40837.998 (48.212) |
| slogdet_positive_c64_256x256 | 889.844 (2.934) | 951.533 (4.373) | 631.645 (1.351) | 651.893 (19.929) | 641.330 (5.917) |
| slogdet_negative_c64_256x256 | 889.023 (3.286) | 951.993 (15.190) | 628.232 (0.984) | 650.470 (4.095) | 638.360 (1.272) |
| householder_qr_square_f32_256x256 | 931.307 (2.334) | 983.922 (3.449) | 911.264 (1.676) | 994.789 (26.524) | 945.148 (2.404) |
| householder_qr_tall_f32_512x256 | 1217.628 (4.196) | 1276.889 (5.965) | 1202.087 (5.816) | 1268.444 (33.269) | 1288.564 (47.114) |
| householder_qr_wide_f32_256x512 | 1107.678 (10.555) | 1208.121 (13.755) | 1112.099 (6.683) | 1207.452 (35.559) | 1249.983 (50.535) |
| householder_qr_square_f64_256x256 | 1038.331 (6.676) | 1081.268 (3.883) | 1019.600 (2.762) | 1073.322 (31.091) | 1091.883 (33.383) |
| householder_qr_tall_f64_512x256 | 1346.307 (5.006) | 1392.424 (5.998) | 1325.671 (8.180) | 1397.819 (11.903) | 1417.249 (8.700) |
| householder_qr_wide_f64_256x512 | 1267.934 (7.160) | 1321.561 (11.503) | 1235.318 (17.303) | 1314.251 (30.394) | 1327.596 (30.208) |
| householder_qr_square_c32_256x256 | 1002.155 (1.718) | 1046.691 (4.738) | 975.822 (1.779) | 1019.181 (24.011) | 1033.281 (17.046) |
| householder_qr_tall_c32_512x256 | 1312.896 (6.740) | 1361.307 (4.908) | 1272.544 (5.280) | 1345.032 (19.323) | 1379.298 (18.240) |
| householder_qr_wide_c32_256x512 | 1237.348 (10.361) | 1289.100 (14.188) | 1206.712 (12.493) | 1274.364 (14.488) | 1304.050 (36.366) |
| slogdet_phase_c32_256x256 | 833.059 (3.841) | 892.459 (2.338) | 576.897 (1.460) | 553.997 (1.400) | 577.102 (0.974) |
| householder_qr_square_c64_256x256 | 1204.952 (4.325) | 1246.133 (3.231) | 1174.356 (4.318) | 1228.038 (11.666) | 1286.604 (28.034) |
| householder_qr_tall_c64_512x256 | 2279.769 (12.400) | 2311.760 (10.310) | 1645.987 (6.855) | 1658.193 (7.678) | 1677.039 (7.080) |
| householder_qr_wide_c64_256x512 | 2135.254 (18.045) | 2154.405 (15.145) | 1485.386 (7.853) | 1541.119 (15.181) | 1586.990 (18.156) |
| slogdet_phase_c64_256x256 | 889.028 (2.140) | 945.063 (4.275) | 629.154 (1.136) | 628.797 (4.575) | 655.466 (4.796) |
| householder_from_factors_default_f32_512x256 | 1451.780 (8.506) | 1519.668 (7.611) | unsupported | unsupported | unsupported |
| householder_append_default_f32_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f32_512x256 | 41.421 (1.651) | 48.637 (0.469) | 5.193 (0.130) | 39.895 (0.668) | 50.347 (1.188) |
| householder_q_columns_prefix_f32_512x256 | unsupported | unsupported | 66.001 (0.999) | 129.183 (1.067) | 108.624 (3.131) |
| householder_q_columns_complement_f32_512x256 | unsupported | unsupported | 164.518 (0.150) | unsupported | unsupported |
| householder_from_factors_default_f64_512x256 | 1584.465 (7.335) | 1664.222 (9.043) | unsupported | unsupported | unsupported |
| householder_append_default_f64_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f64_512x256 | 38.098 (0.097) | 50.827 (0.615) | 5.259 (0.070) | 39.221 (0.276) | 50.496 (1.335) |
| householder_q_columns_prefix_f64_512x256 | unsupported | unsupported | 78.849 (1.367) | 135.972 (15.661) | 108.073 (0.838) |
| householder_q_columns_complement_f64_512x256 | unsupported | unsupported | 164.362 (0.395) | unsupported | unsupported |
| householder_from_factors_default_c32_512x256 | 1560.814 (11.590) | 1600.265 (9.743) | unsupported | unsupported | unsupported |
| householder_append_default_c32_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c32_512x256 | 37.879 (0.216) | 48.398 (3.114) | 5.274 (0.103) | 39.462 (0.234) | 42.097 (1.238) |
| householder_q_columns_prefix_c32_512x256 | unsupported | unsupported | 90.910 (0.080) | 127.919 (0.806) | 105.583 (5.209) |
| householder_q_columns_complement_c32_512x256 | unsupported | unsupported | 249.885 (0.121) | unsupported | unsupported |
| householder_from_factors_default_c64_512x256 | 1883.876 (8.243) | 1958.708 (12.676) | unsupported | unsupported | unsupported |
| householder_append_default_c64_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c64_512x256 | 38.689 (0.144) | 49.653 (0.366) | 5.347 (0.082) | 39.224 (0.292) | 43.226 (2.679) |
| householder_q_columns_prefix_c64_512x256 | unsupported | unsupported | 77.534 (1.184) | 131.160 (0.744) | 112.811 (2.160) |
| householder_q_columns_complement_c64_512x256 | unsupported | unsupported | 251.978 (0.254) | unsupported | unsupported |
| lu_square_f32_s256 | 621.109 (1.900) | 657.070 (2.541) | unsupported | unsupported | unsupported |
| lu_square_f64_s256 | 657.418 (0.989) | 704.647 (2.908) | unsupported | unsupported | unsupported |
| lu_square_c32_s256 | 661.143 (3.575) | 707.539 (5.345) | unsupported | unsupported | unsupported |
| lu_square_c64_s256 | 715.190 (1.404) | 758.094 (7.554) | unsupported | unsupported | unsupported |
| lu_tall_f32_s256 | 770.274 (2.538) | 810.206 (3.406) | unsupported | unsupported | unsupported |
| lu_tall_f64_s256 | 837.734 (1.125) | 872.828 (2.009) | unsupported | unsupported | unsupported |
| lu_tall_c32_s256 | 817.473 (2.760) | 853.117 (2.438) | unsupported | unsupported | unsupported |
| lu_tall_c64_s256 | 2258.428 (15.120) | 2314.180 (14.701) | unsupported | unsupported | unsupported |
| lu_wide_f32_s256 | 668.358 (3.434) | 707.132 (2.864) | unsupported | unsupported | unsupported |
| lu_wide_f64_s256 | 749.149 (3.024) | 785.002 (2.785) | unsupported | unsupported | unsupported |
| lu_wide_c32_s256 | 727.908 (1.925) | 765.262 (2.359) | unsupported | unsupported | unsupported |
| lu_wide_c64_s256 | 2337.171 (107.648) | 2394.363 (18.286) | unsupported | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_512x512 | jax-cuda | 16 | 15 | 3.867 | 2097152 | unavailable | 0.0003558 |
| cholesky_f32_512x512 | jax-cuda-jit | 16 | 15 | 3.818 | 2097152 | unavailable | 0.0003558 |
| cholesky_f32_512x512 | pytorch-cuda | 8 | 15 | 2.196 | 2097152 | unavailable | 0.0003558 |
| cholesky_f32_512x512 | tenferro-cuda-direct | 8 | 15 | 2.276 | 2097152 | unavailable | 0.0003558 |
| cholesky_f32_512x512 | tenferro-cuda-eager | 8 | 15 | 2.631 | 2097152 | unavailable | 0.0003558 |
| cholesky_f64_512x512 | jax-cuda | 8 | 15 | 2.422 | 4194304 | unavailable | 4.418e-07 |
| cholesky_f64_512x512 | jax-cuda-jit | 8 | 15 | 2.400 | 4194304 | unavailable | 4.418e-07 |
| cholesky_f64_512x512 | pytorch-cuda | 8 | 15 | 2.623 | 4194304 | unavailable | 4.418e-07 |
| cholesky_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.098 | 4194304 | unavailable | 4.418e-07 |
| cholesky_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 2.287 | 4194304 | unavailable | 4.418e-07 |
| det_f32_512x512 | jax-cuda | 2 | 15 | 2.422 | 1048580 | unavailable | 0.004748 |
| det_f32_512x512 | jax-cuda-jit | 2 | 15 | 2.460 | 1048580 | unavailable | 0.004748 |
| det_f32_512x512 | pytorch-cuda | 2 | 15 | 2.346 | 1048580 | unavailable | 0.007965 |
| det_f32_512x512 | tenferro-cuda-direct | 2 | 15 | 3.295 | 1048580 | unavailable | 0.004748 |
| det_f32_512x512 | tenferro-cuda-eager | 2 | 15 | 3.416 | 1048580 | unavailable | 0.004748 |
| det_f64_512x512 | jax-cuda | 2 | 15 | 2.823 | 2097160 | unavailable | 1.426e-05 |
| det_f64_512x512 | jax-cuda-jit | 2 | 15 | 2.788 | 2097160 | unavailable | 1.426e-05 |
| det_f64_512x512 | pytorch-cuda | 2 | 15 | 2.634 | 2097160 | unavailable | 1.369e-05 |
| det_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.567 | 2097160 | unavailable | 1.426e-05 |
| det_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 2.661 | 2097160 | unavailable | 1.426e-05 |
| eigvalsh_f32_512x512 | jax-cuda | 1 | 15 | 5.926 | 1050624 | unavailable | 0.008342 |
| eigvalsh_f32_512x512 | jax-cuda-jit | 1 | 15 | 5.911 | 1050624 | unavailable | 0.008342 |
| eigvalsh_f32_512x512 | pytorch-cuda | 1 | 15 | 5.208 | 1050624 | unavailable | 0.008342 |
| eigvalsh_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 5.209 | 1050624 | unavailable | 0.008342 |
| eigvalsh_f32_512x512 | tenferro-cuda-eager | 1 | 15 | 5.270 | 1050624 | unavailable | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda | 1 | 15 | 6.119 | 2101248 | unavailable | 1.55e-06 |
| eigvalsh_f64_512x512 | jax-cuda-jit | 1 | 15 | 6.032 | 2101248 | unavailable | 1.55e-06 |
| eigvalsh_f64_512x512 | pytorch-cuda | 1 | 15 | 5.557 | 2101248 | unavailable | 1.55e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 5.588 | 2101248 | unavailable | 1.55e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 6.197 | 2101248 | unavailable | 1.55e-06 |
| householder_append_default_c32_512x256 | jax-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_512x256 | jax-cuda-jit | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_512x256 | pytorch-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_512x256 | tenferro-cuda-direct | — | — | — | 3147776 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_c32_512x256 | tenferro-cuda-eager | — | — | — | 3147776 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_c64_512x256 | jax-cuda | — | — | — | 6295552 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_512x256 | jax-cuda-jit | — | — | — | 6295552 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_512x256 | pytorch-cuda | — | — | — | 6295552 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_512x256 | tenferro-cuda-direct | — | — | — | 6295552 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_c64_512x256 | tenferro-cuda-eager | — | — | — | 6295552 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_f32_512x256 | jax-cuda | — | — | — | 1573888 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_512x256 | jax-cuda-jit | — | — | — | 1573888 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_512x256 | pytorch-cuda | — | — | — | 1573888 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_512x256 | tenferro-cuda-direct | — | — | — | 1573888 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_f32_512x256 | tenferro-cuda-eager | — | — | — | 1573888 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_f64_512x256 | jax-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | jax-cuda-jit | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | pytorch-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | tenferro-cuda-direct | — | — | — | 3147776 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_f64_512x256 | tenferro-cuda-eager | — | — | — | 3147776 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_512x256 | jax-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c32_512x256 | jax-cuda-jit | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c32_512x256 | pytorch-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c32_512x256 | tenferro-cuda-direct | 2 | 15 | 3.122 | 3672064 | unavailable | 0.001103 |
| householder_from_factors_default_c32_512x256 | tenferro-cuda-eager | 2 | 15 | 3.201 | 3672064 | unavailable | 0.001103 |
| householder_from_factors_default_c64_512x256 | jax-cuda | — | — | — | 7344128 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_512x256 | jax-cuda-jit | — | — | — | 7344128 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_512x256 | pytorch-cuda | — | — | — | 7344128 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_512x256 | tenferro-cuda-direct | 2 | 15 | 3.768 | 7344128 | unavailable | 9.992e-07 |
| householder_from_factors_default_c64_512x256 | tenferro-cuda-eager | 2 | 15 | 3.917 | 7344128 | unavailable | 9.992e-07 |
| householder_from_factors_default_f32_512x256 | jax-cuda | — | — | — | 1836032 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_512x256 | jax-cuda-jit | — | — | — | 1836032 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_512x256 | pytorch-cuda | — | — | — | 1836032 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.904 | 1836032 | unavailable | 0.0008522 |
| householder_from_factors_default_f32_512x256 | tenferro-cuda-eager | 2 | 15 | 3.039 | 1836032 | unavailable | 0.0008522 |
| householder_from_factors_default_f64_512x256 | jax-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | jax-cuda-jit | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | pytorch-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 3.169 | 3672064 | unavailable | 8.882e-07 |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 3.328 | 3672064 | unavailable | 8.882e-07 |
| householder_q_columns_complement_c32_512x256 | jax-cuda | — | — | — | 2097152 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c32_512x256 | jax-cuda-jit | — | — | — | 2097152 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c32_512x256 | pytorch-cuda | 8 | 15 | 1.999 | 2097152 | unavailable | 0.0001141 |
| householder_q_columns_complement_c32_512x256 | tenferro-cuda-direct | — | — | — | 2097152 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_512x256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_512x256 | jax-cuda | — | — | — | 4194304 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c64_512x256 | jax-cuda-jit | — | — | — | 4194304 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c64_512x256 | pytorch-cuda | 8 | 15 | 2.016 | 4194304 | unavailable | 7.282e-07 |
| householder_q_columns_complement_c64_512x256 | tenferro-cuda-direct | — | — | — | 4194304 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_512x256 | tenferro-cuda-eager | — | — | — | 4194304 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_512x256 | jax-cuda | — | — | — | 1048576 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f32_512x256 | jax-cuda-jit | — | — | — | 1048576 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f32_512x256 | pytorch-cuda | 16 | 15 | 2.632 | 1048576 | unavailable | 8.761e-05 |
| householder_q_columns_complement_f32_512x256 | tenferro-cuda-direct | — | — | — | 1048576 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_512x256 | tenferro-cuda-eager | — | — | — | 1048576 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_512x256 | jax-cuda | — | — | — | 2097152 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f64_512x256 | jax-cuda-jit | — | — | — | 2097152 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f64_512x256 | pytorch-cuda | 16 | 15 | 2.630 | 2097152 | unavailable | 7.26e-07 |
| householder_q_columns_complement_f64_512x256 | tenferro-cuda-direct | — | — | — | 2097152 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_512x256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c32_512x256 | jax-cuda | 16 | 15 | 2.047 | 1081344 | unavailable | 0.0001246 |
| householder_q_columns_prefix_c32_512x256 | jax-cuda-jit | 32 | 15 | 3.379 | 1081344 | unavailable | 0.0001246 |
| householder_q_columns_prefix_c32_512x256 | pytorch-cuda | 32 | 15 | 2.909 | 1081344 | unavailable | 0.0001246 |
| householder_q_columns_prefix_c32_512x256 | tenferro-cuda-direct | — | — | — | 1081344 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c32_512x256 | tenferro-cuda-eager | — | — | — | 1081344 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c64_512x256 | jax-cuda | 16 | 15 | 2.099 | 2162688 | unavailable | 5.968e-08 |
| householder_q_columns_prefix_c64_512x256 | jax-cuda-jit | 32 | 15 | 3.610 | 2162688 | unavailable | 5.968e-08 |
| householder_q_columns_prefix_c64_512x256 | pytorch-cuda | 32 | 15 | 2.481 | 2162688 | unavailable | 5.968e-08 |
| householder_q_columns_prefix_c64_512x256 | tenferro-cuda-direct | — | — | — | 2162688 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c64_512x256 | tenferro-cuda-eager | — | — | — | 2162688 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f32_512x256 | jax-cuda | 16 | 15 | 2.067 | 540672 | unavailable | 0.0001829 |
| householder_q_columns_prefix_f32_512x256 | jax-cuda-jit | 16 | 15 | 1.738 | 540672 | unavailable | 0.0001829 |
| householder_q_columns_prefix_f32_512x256 | pytorch-cuda | 32 | 15 | 2.112 | 540672 | unavailable | 0.0001829 |
| householder_q_columns_prefix_f32_512x256 | tenferro-cuda-direct | — | — | — | 540672 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f32_512x256 | tenferro-cuda-eager | — | — | — | 540672 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | 16 | 15 | 2.176 | 1081344 | unavailable | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | jax-cuda-jit | 32 | 15 | 3.458 | 1081344 | unavailable | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | 32 | 15 | 2.523 | 1081344 | unavailable | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-direct | — | — | — | 1081344 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-eager | — | — | — | 1081344 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_qr_square_c32_256x256 | jax-cuda | 2 | 15 | 2.038 | 1050624 | unavailable | 0.001258 |
| householder_qr_square_c32_256x256 | jax-cuda-jit | 2 | 15 | 2.067 | 1050624 | unavailable | 0.001258 |
| householder_qr_square_c32_256x256 | pytorch-cuda | 4 | 15 | 3.903 | 1050624 | unavailable | 0.001258 |
| householder_qr_square_c32_256x256 | tenferro-cuda-direct | 2 | 15 | 2.004 | 1050624 | unavailable | 0.001198 |
| householder_qr_square_c32_256x256 | tenferro-cuda-eager | 2 | 15 | 2.093 | 1050624 | unavailable | 0.001198 |
| householder_qr_square_c64_256x256 | jax-cuda | 2 | 15 | 2.456 | 2101248 | unavailable | 2.109e-06 |
| householder_qr_square_c64_256x256 | jax-cuda-jit | 2 | 15 | 2.573 | 2101248 | unavailable | 2.109e-06 |
| householder_qr_square_c64_256x256 | pytorch-cuda | 2 | 15 | 2.349 | 2101248 | unavailable | 2.109e-06 |
| householder_qr_square_c64_256x256 | tenferro-cuda-direct | 2 | 15 | 2.410 | 2101248 | unavailable | 1.208e-06 |
| householder_qr_square_c64_256x256 | tenferro-cuda-eager | 2 | 15 | 2.492 | 2101248 | unavailable | 1.208e-06 |
| householder_qr_square_f32_256x256 | jax-cuda | 2 | 15 | 1.990 | 525312 | unavailable | 0.001043 |
| householder_qr_square_f32_256x256 | jax-cuda-jit | 4 | 15 | 3.781 | 525312 | unavailable | 0.001043 |
| householder_qr_square_f32_256x256 | pytorch-cuda | 4 | 15 | 3.645 | 525312 | unavailable | 0.001043 |
| householder_qr_square_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.725 | 525312 | unavailable | 0.001075 |
| householder_qr_square_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.936 | 525312 | unavailable | 0.001075 |
| householder_qr_square_f64_256x256 | jax-cuda | 2 | 15 | 2.147 | 1050624 | unavailable | 1.887e-06 |
| householder_qr_square_f64_256x256 | jax-cuda-jit | 2 | 15 | 2.184 | 1050624 | unavailable | 1.887e-06 |
| householder_qr_square_f64_256x256 | pytorch-cuda | 2 | 15 | 2.039 | 1050624 | unavailable | 1.887e-06 |
| householder_qr_square_f64_256x256 | tenferro-cuda-direct | 2 | 15 | 2.077 | 1050624 | unavailable | 1.098e-06 |
| householder_qr_square_f64_256x256 | tenferro-cuda-eager | 2 | 15 | 2.163 | 1050624 | unavailable | 1.098e-06 |
| householder_qr_tall_c32_512x256 | jax-cuda | 2 | 15 | 2.690 | 2099200 | unavailable | 0.001161 |
| householder_qr_tall_c32_512x256 | jax-cuda-jit | 2 | 15 | 2.759 | 2099200 | unavailable | 0.001161 |
| householder_qr_tall_c32_512x256 | pytorch-cuda | 2 | 15 | 2.545 | 2099200 | unavailable | 0.001161 |
| householder_qr_tall_c32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.626 | 2099200 | unavailable | 0.001185 |
| householder_qr_tall_c32_512x256 | tenferro-cuda-eager | 2 | 15 | 2.723 | 2099200 | unavailable | 0.001185 |
| householder_qr_tall_c64_512x256 | jax-cuda | 2 | 15 | 3.316 | 4198400 | unavailable | 2.942e-06 |
| householder_qr_tall_c64_512x256 | jax-cuda-jit | 2 | 15 | 3.354 | 4198400 | unavailable | 2.942e-06 |
| householder_qr_tall_c64_512x256 | pytorch-cuda | 2 | 15 | 3.292 | 4198400 | unavailable | 2.942e-06 |
| householder_qr_tall_c64_512x256 | tenferro-cuda-direct | 1 | 15 | 2.280 | 4198400 | unavailable | 1.11e-06 |
| householder_qr_tall_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 2.312 | 4198400 | unavailable | 1.11e-06 |
| householder_qr_tall_f32_512x256 | jax-cuda | 2 | 15 | 2.537 | 1049600 | unavailable | 0.001475 |
| householder_qr_tall_f32_512x256 | jax-cuda-jit | 2 | 15 | 2.577 | 1049600 | unavailable | 0.001475 |
| householder_qr_tall_f32_512x256 | pytorch-cuda | 2 | 15 | 2.404 | 1049600 | unavailable | 0.001475 |
| householder_qr_tall_f32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.435 | 1049600 | unavailable | 0.001435 |
| householder_qr_tall_f32_512x256 | tenferro-cuda-eager | 2 | 15 | 2.554 | 1049600 | unavailable | 0.001435 |
| householder_qr_tall_f64_512x256 | jax-cuda | 2 | 15 | 2.796 | 2099200 | unavailable | 1.887e-06 |
| householder_qr_tall_f64_512x256 | jax-cuda-jit | 2 | 15 | 2.834 | 2099200 | unavailable | 1.887e-06 |
| householder_qr_tall_f64_512x256 | pytorch-cuda | 2 | 15 | 2.651 | 2099200 | unavailable | 1.887e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 2.693 | 2099200 | unavailable | 9.992e-07 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 2.785 | 2099200 | unavailable | 9.992e-07 |
| householder_qr_wide_c32_256x512 | jax-cuda | 2 | 15 | 2.549 | 2099200 | unavailable | 0.001226 |
| householder_qr_wide_c32_256x512 | jax-cuda-jit | 2 | 15 | 2.608 | 2099200 | unavailable | 0.001226 |
| householder_qr_wide_c32_256x512 | pytorch-cuda | 2 | 15 | 2.413 | 2099200 | unavailable | 0.001226 |
| householder_qr_wide_c32_256x512 | tenferro-cuda-direct | 2 | 15 | 2.475 | 2099200 | unavailable | 0.001329 |
| householder_qr_wide_c32_256x512 | tenferro-cuda-eager | 2 | 15 | 2.578 | 2099200 | unavailable | 0.001329 |
| householder_qr_wide_c64_256x512 | jax-cuda | 2 | 15 | 3.082 | 4198400 | unavailable | 2.276e-06 |
| householder_qr_wide_c64_256x512 | jax-cuda-jit | 2 | 15 | 3.174 | 4198400 | unavailable | 2.276e-06 |
| householder_qr_wide_c64_256x512 | pytorch-cuda | 2 | 15 | 2.971 | 4198400 | unavailable | 2.276e-06 |
| householder_qr_wide_c64_256x512 | tenferro-cuda-direct | 1 | 15 | 2.135 | 4198400 | unavailable | 1.206e-06 |
| householder_qr_wide_c64_256x512 | tenferro-cuda-eager | 1 | 15 | 2.154 | 4198400 | unavailable | 1.206e-06 |
| householder_qr_wide_f32_256x512 | jax-cuda | 2 | 15 | 2.415 | 1049600 | unavailable | 0.0009736 |
| householder_qr_wide_f32_256x512 | jax-cuda-jit | 2 | 15 | 2.500 | 1049600 | unavailable | 0.0009736 |
| householder_qr_wide_f32_256x512 | pytorch-cuda | 2 | 15 | 2.224 | 1049600 | unavailable | 0.0009736 |
| householder_qr_wide_f32_256x512 | tenferro-cuda-direct | 2 | 15 | 2.215 | 1049600 | unavailable | 0.0009125 |
| householder_qr_wide_f32_256x512 | tenferro-cuda-eager | 2 | 15 | 2.416 | 1049600 | unavailable | 0.0009125 |
| householder_qr_wide_f64_256x512 | jax-cuda | 2 | 15 | 2.629 | 2099200 | unavailable | 1.759e-06 |
| householder_qr_wide_f64_256x512 | jax-cuda-jit | 2 | 15 | 2.655 | 2099200 | unavailable | 1.759e-06 |
| householder_qr_wide_f64_256x512 | pytorch-cuda | 2 | 15 | 2.471 | 2099200 | unavailable | 1.759e-06 |
| householder_qr_wide_f64_256x512 | tenferro-cuda-direct | 2 | 15 | 2.536 | 2099200 | unavailable | 1.649e-06 |
| householder_qr_wide_f64_256x512 | tenferro-cuda-eager | 2 | 15 | 2.643 | 2099200 | unavailable | 1.649e-06 |
| householder_r_default_c32_512x256 | jax-cuda | 64 | 15 | 2.526 | 1572864 | unavailable | 0.001069 |
| householder_r_default_c32_512x256 | jax-cuda-jit | 64 | 15 | 2.694 | 1572864 | unavailable | 0.001069 |
| householder_r_default_c32_512x256 | pytorch-cuda | 512 | 15 | 2.700 | 1572864 | unavailable | 0.001069 |
| householder_r_default_c32_512x256 | tenferro-cuda-direct | 64 | 15 | 2.424 | 1572864 | unavailable | 0.001069 |
| householder_r_default_c32_512x256 | tenferro-cuda-eager | 64 | 15 | 3.097 | 1572864 | unavailable | 0.001069 |
| householder_r_default_c64_512x256 | jax-cuda | 64 | 15 | 2.510 | 3145728 | unavailable | 9.985e-07 |
| householder_r_default_c64_512x256 | jax-cuda-jit | 64 | 15 | 2.766 | 3145728 | unavailable | 9.985e-07 |
| householder_r_default_c64_512x256 | pytorch-cuda | 256 | 15 | 1.369 | 3145728 | unavailable | 9.985e-07 |
| householder_r_default_c64_512x256 | tenferro-cuda-direct | 64 | 15 | 2.476 | 3145728 | unavailable | 9.985e-07 |
| householder_r_default_c64_512x256 | tenferro-cuda-eager | 64 | 15 | 3.178 | 3145728 | unavailable | 9.985e-07 |
| householder_r_default_f32_512x256 | jax-cuda | 32 | 15 | 1.277 | 786432 | unavailable | 0.001415 |
| householder_r_default_f32_512x256 | jax-cuda-jit | 32 | 15 | 1.611 | 786432 | unavailable | 0.001415 |
| householder_r_default_f32_512x256 | pytorch-cuda | 256 | 15 | 1.329 | 786432 | unavailable | 0.001415 |
| householder_r_default_f32_512x256 | tenferro-cuda-direct | 64 | 15 | 2.651 | 786432 | unavailable | 0.001415 |
| householder_r_default_f32_512x256 | tenferro-cuda-eager | 64 | 15 | 3.113 | 786432 | unavailable | 0.001415 |
| householder_r_default_f64_512x256 | jax-cuda | 64 | 15 | 2.510 | 1572864 | unavailable | 9.976e-07 |
| householder_r_default_f64_512x256 | jax-cuda-jit | 64 | 15 | 3.232 | 1572864 | unavailable | 9.976e-07 |
| householder_r_default_f64_512x256 | pytorch-cuda | 256 | 15 | 1.346 | 1572864 | unavailable | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | 64 | 15 | 2.438 | 1572864 | unavailable | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | 64 | 15 | 3.253 | 1572864 | unavailable | 9.976e-07 |
| inv_f32_512x512 | jax-cuda | 2 | 15 | 3.010 | 2097152 | unavailable | 0.002107 |
| inv_f32_512x512 | jax-cuda-jit | 2 | 15 | 3.089 | 2097152 | unavailable | 0.002107 |
| inv_f32_512x512 | pytorch-cuda | 2 | 15 | 3.011 | 2097152 | unavailable | 0.002107 |
| inv_f32_512x512 | tenferro-cuda-direct | — | — | — | 2097152 | unavailable | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_512x512 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_512x512 | jax-cuda | 2 | 15 | 3.679 | 4194304 | unavailable | 1.85e-06 |
| inv_f64_512x512 | jax-cuda-jit | 2 | 15 | 3.593 | 4194304 | unavailable | 1.85e-06 |
| inv_f64_512x512 | pytorch-cuda | 2 | 15 | 3.601 | 4194304 | unavailable | 1.85e-06 |
| inv_f64_512x512 | tenferro-cuda-direct | — | — | — | 4194304 | unavailable | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_512x512 | tenferro-cuda-eager | — | — | — | 4194304 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| lu_square_c32_s256 | jax-cuda | — | — | — | 2097160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s256 | jax-cuda-jit | — | — | — | 2097160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s256 | pytorch-cuda | — | — | — | 2097160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s256 | tenferro-cuda-direct | 4 | 15 | 2.645 | 2097160 | unavailable | 0.001541 |
| lu_square_c32_s256 | tenferro-cuda-eager | 4 | 15 | 2.830 | 2097160 | unavailable | 0.001541 |
| lu_square_c64_s256 | jax-cuda | — | — | — | 4194320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s256 | jax-cuda-jit | — | — | — | 4194320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s256 | pytorch-cuda | — | — | — | 4194320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.861 | 4194320 | unavailable | 1.431e-06 |
| lu_square_c64_s256 | tenferro-cuda-eager | 4 | 15 | 3.032 | 4194320 | unavailable | 1.431e-06 |
| lu_square_f32_s256 | jax-cuda | — | — | — | 1048580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s256 | jax-cuda-jit | — | — | — | 1048580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s256 | pytorch-cuda | — | — | — | 1048580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s256 | tenferro-cuda-direct | 4 | 15 | 2.484 | 1048580 | unavailable | 0.001428 |
| lu_square_f32_s256 | tenferro-cuda-eager | 4 | 15 | 2.628 | 1048580 | unavailable | 0.001428 |
| lu_square_f64_s256 | jax-cuda | — | — | — | 2097160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s256 | jax-cuda-jit | — | — | — | 2097160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s256 | pytorch-cuda | — | — | — | 2097160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s256 | tenferro-cuda-direct | 4 | 15 | 2.630 | 2097160 | unavailable | 1.102e-06 |
| lu_square_f64_s256 | tenferro-cuda-eager | 4 | 15 | 2.819 | 2097160 | unavailable | 1.102e-06 |
| lu_tall_c32_s256 | jax-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s256 | jax-cuda-jit | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s256 | pytorch-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s256 | tenferro-cuda-direct | 4 | 15 | 3.270 | 4718600 | unavailable | 0.002239 |
| lu_tall_c32_s256 | tenferro-cuda-eager | 4 | 15 | 3.412 | 4718600 | unavailable | 0.002239 |
| lu_tall_c64_s256 | jax-cuda | — | — | — | 9437200 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s256 | jax-cuda-jit | — | — | — | 9437200 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s256 | pytorch-cuda | — | — | — | 9437200 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s256 | tenferro-cuda-direct | 1 | 15 | 2.258 | 9437200 | unavailable | 1.542e-06 |
| lu_tall_c64_s256 | tenferro-cuda-eager | 1 | 15 | 2.314 | 9437200 | unavailable | 1.542e-06 |
| lu_tall_f32_s256 | jax-cuda | — | — | — | 2359300 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s256 | jax-cuda-jit | — | — | — | 2359300 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s256 | pytorch-cuda | — | — | — | 2359300 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s256 | tenferro-cuda-direct | 4 | 15 | 3.081 | 2359300 | unavailable | 0.001428 |
| lu_tall_f32_s256 | tenferro-cuda-eager | 4 | 15 | 3.241 | 2359300 | unavailable | 0.001428 |
| lu_tall_f64_s256 | jax-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | jax-cuda-jit | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | pytorch-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | tenferro-cuda-direct | 4 | 15 | 3.351 | 4718600 | unavailable | 1.102e-06 |
| lu_tall_f64_s256 | tenferro-cuda-eager | 4 | 15 | 3.491 | 4718600 | unavailable | 1.102e-06 |
| lu_wide_c32_s256 | jax-cuda | — | — | — | 3145736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s256 | jax-cuda-jit | — | — | — | 3145736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s256 | pytorch-cuda | — | — | — | 3145736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s256 | tenferro-cuda-direct | 4 | 15 | 2.912 | 3145736 | unavailable | 0.002021 |
| lu_wide_c32_s256 | tenferro-cuda-eager | 4 | 15 | 3.061 | 3145736 | unavailable | 0.002021 |
| lu_wide_c64_s256 | jax-cuda | — | — | — | 6291472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s256 | jax-cuda-jit | — | — | — | 6291472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s256 | pytorch-cuda | — | — | — | 6291472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s256 | tenferro-cuda-direct | 1 | 15 | 2.337 | 6291472 | unavailable | 1.433e-06 |
| lu_wide_c64_s256 | tenferro-cuda-eager | 1 | 15 | 2.394 | 6291472 | unavailable | 1.433e-06 |
| lu_wide_f32_s256 | jax-cuda | — | — | — | 1572868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s256 | jax-cuda-jit | — | — | — | 1572868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s256 | pytorch-cuda | — | — | — | 1572868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s256 | tenferro-cuda-direct | 4 | 15 | 2.673 | 1572868 | unavailable | 0.001297 |
| lu_wide_f32_s256 | tenferro-cuda-eager | 4 | 15 | 2.829 | 1572868 | unavailable | 0.001297 |
| lu_wide_f64_s256 | jax-cuda | — | — | — | 3145736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s256 | jax-cuda-jit | — | — | — | 3145736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s256 | pytorch-cuda | — | — | — | 3145736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s256 | tenferro-cuda-direct | 4 | 15 | 2.997 | 3145736 | unavailable | 8.806e-07 |
| lu_wide_f64_s256 | tenferro-cuda-eager | 4 | 15 | 3.140 | 3145736 | unavailable | 8.806e-07 |
| norm_f32_512x512 | jax-cuda | 32 | 15 | 1.578 | 1048580 | unavailable | 0 |
| norm_f32_512x512 | jax-cuda-jit | 64 | 15 | 3.348 | 1048580 | unavailable | 0 |
| norm_f32_512x512 | pytorch-cuda | 256 | 15 | 2.769 | 1048580 | unavailable | 0 |
| norm_f32_512x512 | tenferro-cuda-direct | 32 | 15 | 3.055 | 1048580 | unavailable | 0 |
| norm_f32_512x512 | tenferro-cuda-eager | 32 | 15 | 3.797 | 1048580 | unavailable | 0 |
| norm_f64_512x512 | jax-cuda | 64 | 15 | 3.075 | 2097160 | unavailable | 1.909e-07 |
| norm_f64_512x512 | jax-cuda-jit | 64 | 15 | 3.589 | 2097160 | unavailable | 1.909e-07 |
| norm_f64_512x512 | pytorch-cuda | 256 | 15 | 2.766 | 2097160 | unavailable | 1.909e-07 |
| norm_f64_512x512 | tenferro-cuda-direct | 32 | 15 | 3.059 | 2097160 | unavailable | 1.909e-07 |
| norm_f64_512x512 | tenferro-cuda-eager | 32 | 15 | 3.851 | 2097160 | unavailable | 1.909e-07 |
| pinv_f32_512x512 | jax-cuda | 1 | 15 | 31.528 | 2097160 | unavailable | 0.009697 |
| pinv_f32_512x512 | jax-cuda-jit | 1 | 15 | 31.398 | 2097160 | unavailable | 0.009581 |
| pinv_f32_512x512 | pytorch-cuda | 1 | 15 | 34.277 | 2097160 | unavailable | 0.01174 |
| pinv_f32_512x512 | tenferro-cuda-direct | — | — | — | 2097160 | unavailable | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f32_512x512 | tenferro-cuda-eager | — | — | — | 2097160 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_512x512 | jax-cuda | 1 | 15 | 78.119 | 4194320 | unavailable | 8.975e-06 |
| pinv_f64_512x512 | jax-cuda-jit | 1 | 15 | 78.079 | 4194320 | unavailable | 8.975e-06 |
| pinv_f64_512x512 | pytorch-cuda | 1 | 15 | 84.627 | 4194320 | unavailable | 8.747e-06 |
| pinv_f64_512x512 | tenferro-cuda-direct | — | — | — | 4194320 | unavailable | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_512x512 | tenferro-cuda-eager | — | — | — | 4194320 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| slogdet_negative_c32_256x256 | jax-cuda | 4 | 15 | 2.318 | 524300 | unavailable | 0.01211 |
| slogdet_negative_c32_256x256 | jax-cuda-jit | 4 | 15 | 2.305 | 524300 | unavailable | 0.01211 |
| slogdet_negative_c32_256x256 | pytorch-cuda | 4 | 15 | 2.297 | 524300 | unavailable | 0.01211 |
| slogdet_negative_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.325 | 524300 | unavailable | 0.01211 |
| slogdet_negative_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.764 | 524300 | unavailable | 0.01211 |
| slogdet_negative_c64_256x256 | jax-cuda | 4 | 15 | 2.602 | 1048600 | unavailable | 1.025e-06 |
| slogdet_negative_c64_256x256 | jax-cuda-jit | 4 | 15 | 2.553 | 1048600 | unavailable | 1.025e-06 |
| slogdet_negative_c64_256x256 | pytorch-cuda | 4 | 15 | 2.513 | 1048600 | unavailable | 1.23e-06 |
| slogdet_negative_c64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.556 | 1048600 | unavailable | 1.025e-06 |
| slogdet_negative_c64_256x256 | tenferro-cuda-eager | 4 | 15 | 3.808 | 1048600 | unavailable | 1.025e-06 |
| slogdet_negative_f32_256x256 | jax-cuda | 4 | 15 | 2.144 | 262152 | unavailable | 0.0005003 |
| slogdet_negative_f32_256x256 | jax-cuda-jit | 4 | 15 | 2.164 | 262152 | unavailable | 0.0005003 |
| slogdet_negative_f32_256x256 | pytorch-cuda | 4 | 15 | 2.123 | 262152 | unavailable | 0.0005003 |
| slogdet_negative_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.115 | 262152 | unavailable | 0.0005003 |
| slogdet_negative_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.340 | 262152 | unavailable | 0.0005003 |
| slogdet_negative_f64_256x256 | jax-cuda | 4 | 15 | 2.249 | 524304 | unavailable | 1.864e-06 |
| slogdet_negative_f64_256x256 | jax-cuda-jit | 4 | 15 | 2.305 | 524304 | unavailable | 1.864e-06 |
| slogdet_negative_f64_256x256 | pytorch-cuda | 4 | 15 | 2.269 | 524304 | unavailable | 1.708e-06 |
| slogdet_negative_f64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.292 | 524304 | unavailable | 1.864e-06 |
| slogdet_negative_f64_256x256 | tenferro-cuda-eager | 4 | 15 | 3.499 | 524304 | unavailable | 1.864e-06 |
| slogdet_phase_c32_256x256 | jax-cuda | 4 | 15 | 2.216 | 524300 | unavailable | 0.01233 |
| slogdet_phase_c32_256x256 | jax-cuda-jit | 4 | 15 | 2.308 | 524300 | unavailable | 0.01233 |
| slogdet_phase_c32_256x256 | pytorch-cuda | 4 | 15 | 2.308 | 524300 | unavailable | 0.01233 |
| slogdet_phase_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.332 | 524300 | unavailable | 0.01211 |
| slogdet_phase_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.570 | 524300 | unavailable | 0.01211 |
| slogdet_phase_c64_256x256 | jax-cuda | 4 | 15 | 2.515 | 1048600 | unavailable | 1.025e-06 |
| slogdet_phase_c64_256x256 | jax-cuda-jit | 4 | 15 | 2.622 | 1048600 | unavailable | 1.025e-06 |
| slogdet_phase_c64_256x256 | pytorch-cuda | 4 | 15 | 2.517 | 1048600 | unavailable | 1.025e-06 |
| slogdet_phase_c64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.556 | 1048600 | unavailable | 1.025e-06 |
| slogdet_phase_c64_256x256 | tenferro-cuda-eager | 4 | 15 | 3.780 | 1048600 | unavailable | 1.025e-06 |
| slogdet_positive_c32_256x256 | jax-cuda | 4 | 15 | 2.235 | 524300 | unavailable | 0.01211 |
| slogdet_positive_c32_256x256 | jax-cuda-jit | 4 | 15 | 2.240 | 524300 | unavailable | 0.01211 |
| slogdet_positive_c32_256x256 | pytorch-cuda | 4 | 15 | 2.306 | 524300 | unavailable | 0.01211 |
| slogdet_positive_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.328 | 524300 | unavailable | 0.01211 |
| slogdet_positive_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.767 | 524300 | unavailable | 0.01211 |
| slogdet_positive_c64_256x256 | jax-cuda | 4 | 15 | 2.608 | 1048600 | unavailable | 1.025e-06 |
| slogdet_positive_c64_256x256 | jax-cuda-jit | 4 | 15 | 2.565 | 1048600 | unavailable | 1.025e-06 |
| slogdet_positive_c64_256x256 | pytorch-cuda | 4 | 15 | 2.527 | 1048600 | unavailable | 1.23e-06 |
| slogdet_positive_c64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.559 | 1048600 | unavailable | 1.025e-06 |
| slogdet_positive_c64_256x256 | tenferro-cuda-eager | 4 | 15 | 3.806 | 1048600 | unavailable | 1.025e-06 |
| slogdet_positive_f32_256x256 | jax-cuda | 4 | 15 | 2.073 | 262152 | unavailable | 0.0005003 |
| slogdet_positive_f32_256x256 | jax-cuda-jit | 4 | 15 | 2.114 | 262152 | unavailable | 0.0005003 |
| slogdet_positive_f32_256x256 | pytorch-cuda | 4 | 15 | 2.123 | 262152 | unavailable | 0.0005003 |
| slogdet_positive_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.126 | 262152 | unavailable | 0.0005003 |
| slogdet_positive_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.336 | 262152 | unavailable | 0.0005003 |
| slogdet_positive_f64_256x256 | jax-cuda | 4 | 15 | 2.239 | 524304 | unavailable | 1.864e-06 |
| slogdet_positive_f64_256x256 | jax-cuda-jit | 4 | 15 | 2.294 | 524304 | unavailable | 1.864e-06 |
| slogdet_positive_f64_256x256 | pytorch-cuda | 4 | 15 | 2.277 | 524304 | unavailable | 1.708e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.298 | 524304 | unavailable | 1.864e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | 4 | 15 | 3.501 | 524304 | unavailable | 1.864e-06 |
| svd_full_square_c32_256x256 | jax-cuda | 1 | 15 | 20.429 | 1573888 | unavailable | 0.133 |
| svd_full_square_c32_256x256 | jax-cuda-jit | 1 | 15 | 20.383 | 1573888 | unavailable | 0.133 |
| svd_full_square_c32_256x256 | pytorch-cuda | 1 | 15 | 22.072 | 1573888 | unavailable | 0.1305 |
| svd_full_square_c32_256x256 | tenferro-cuda-direct | 1 | 15 | 20.312 | 1573888 | unavailable | 0.133 |
| svd_full_square_c32_256x256 | tenferro-cuda-eager | 1 | 15 | 20.386 | 1573888 | unavailable | 0.133 |
| svd_full_square_c64_256x256 | jax-cuda | 1 | 15 | 60.193 | 3147776 | unavailable | 0.000204 |
| svd_full_square_c64_256x256 | jax-cuda-jit | 1 | 15 | 60.253 | 3147776 | unavailable | 0.000204 |
| svd_full_square_c64_256x256 | pytorch-cuda | 1 | 15 | 63.622 | 3147776 | unavailable | 0.0002064 |
| svd_full_square_c64_256x256 | tenferro-cuda-direct | 1 | 15 | 60.066 | 3147776 | unavailable | 0.000204 |
| svd_full_square_c64_256x256 | tenferro-cuda-eager | 1 | 15 | 60.134 | 3147776 | unavailable | 0.000204 |
| svd_full_square_f32_256x256 | jax-cuda | 1 | 15 | 10.845 | 787456 | unavailable | 0.1152 |
| svd_full_square_f32_256x256 | jax-cuda-jit | 1 | 15 | 10.878 | 787456 | unavailable | 0.1152 |
| svd_full_square_f32_256x256 | pytorch-cuda | 1 | 15 | 11.370 | 787456 | unavailable | 0.1135 |
| svd_full_square_f32_256x256 | tenferro-cuda-direct | 1 | 15 | 10.818 | 787456 | unavailable | 0.1152 |
| svd_full_square_f32_256x256 | tenferro-cuda-eager | 1 | 15 | 10.852 | 787456 | unavailable | 0.1152 |
| svd_full_square_f64_256x256 | jax-cuda | 1 | 15 | 25.808 | 1574912 | unavailable | 8.171e-05 |
| svd_full_square_f64_256x256 | jax-cuda-jit | 1 | 15 | 25.837 | 1574912 | unavailable | 8.171e-05 |
| svd_full_square_f64_256x256 | pytorch-cuda | 1 | 15 | 25.711 | 1574912 | unavailable | 8.105e-05 |
| svd_full_square_f64_256x256 | tenferro-cuda-direct | 1 | 15 | 25.819 | 1574912 | unavailable | 8.171e-05 |
| svd_full_square_f64_256x256 | tenferro-cuda-eager | 1 | 15 | 25.790 | 1574912 | unavailable | 8.171e-05 |
| svd_full_tall_c32_512x256 | jax-cuda | 1 | 15 | 19.671 | 3671040 | unavailable | 0.1126 |
| svd_full_tall_c32_512x256 | jax-cuda-jit | 1 | 15 | 19.705 | 3671040 | unavailable | 0.1126 |
| svd_full_tall_c32_512x256 | pytorch-cuda | 1 | 15 | 20.018 | 3671040 | unavailable | 0.1089 |
| svd_full_tall_c32_512x256 | tenferro-cuda-direct | 1 | 15 | 19.622 | 3671040 | unavailable | 0.1126 |
| svd_full_tall_c32_512x256 | tenferro-cuda-eager | 1 | 15 | 19.688 | 3671040 | unavailable | 0.1126 |
| svd_full_tall_c64_512x256 | jax-cuda | 1 | 15 | 40.692 | 7342080 | unavailable | 0.0002021 |
| svd_full_tall_c64_512x256 | jax-cuda-jit | 1 | 15 | 40.769 | 7342080 | unavailable | 0.0002021 |
| svd_full_tall_c64_512x256 | pytorch-cuda | 1 | 15 | 40.996 | 7342080 | unavailable | 0.0001992 |
| svd_full_tall_c64_512x256 | tenferro-cuda-direct | 1 | 15 | 42.704 | 7342080 | unavailable | 0.0002022 |
| svd_full_tall_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 42.705 | 7342080 | unavailable | 0.0002022 |
| svd_full_tall_f32_512x256 | jax-cuda | 1 | 15 | 11.265 | 1836032 | unavailable | 0.1022 |
| svd_full_tall_f32_512x256 | jax-cuda-jit | 1 | 15 | 11.300 | 1836032 | unavailable | 0.1022 |
| svd_full_tall_f32_512x256 | pytorch-cuda | 1 | 15 | 11.883 | 1836032 | unavailable | 0.1033 |
| svd_full_tall_f32_512x256 | tenferro-cuda-direct | 1 | 15 | 11.248 | 1836032 | unavailable | 0.1022 |
| svd_full_tall_f32_512x256 | tenferro-cuda-eager | 1 | 15 | 11.261 | 1836032 | unavailable | 0.1022 |
| svd_full_tall_f64_512x256 | jax-cuda | 1 | 15 | 19.640 | 3672064 | unavailable | 7.76e-05 |
| svd_full_tall_f64_512x256 | jax-cuda-jit | 1 | 15 | 19.615 | 3672064 | unavailable | 7.76e-05 |
| svd_full_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 19.817 | 3672064 | unavailable | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 19.611 | 3672064 | unavailable | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 19.650 | 3672064 | unavailable | 7.76e-05 |
| svd_full_wide_c32_256x512 | jax-cuda | 1 | 15 | 19.822 | 3671040 | unavailable | 0.1103 |
| svd_full_wide_c32_256x512 | jax-cuda-jit | 1 | 15 | 19.869 | 3671040 | unavailable | 0.1103 |
| svd_full_wide_c32_256x512 | pytorch-cuda | 1 | 15 | 20.219 | 3671040 | unavailable | 0.108 |
| svd_full_wide_c32_256x512 | tenferro-cuda-direct | 1 | 15 | 19.805 | 3671040 | unavailable | 0.1103 |
| svd_full_wide_c32_256x512 | tenferro-cuda-eager | 1 | 15 | 19.831 | 3671040 | unavailable | 0.1103 |
| svd_full_wide_c64_256x512 | jax-cuda | 1 | 15 | 40.803 | 7342080 | unavailable | 0.0001946 |
| svd_full_wide_c64_256x512 | jax-cuda-jit | 1 | 15 | 40.838 | 7342080 | unavailable | 0.0001946 |
| svd_full_wide_c64_256x512 | pytorch-cuda | 1 | 15 | 41.573 | 7342080 | unavailable | 0.0001941 |
| svd_full_wide_c64_256x512 | tenferro-cuda-direct | 1 | 15 | 42.831 | 7342080 | unavailable | 0.0001943 |
| svd_full_wide_c64_256x512 | tenferro-cuda-eager | 1 | 15 | 42.886 | 7342080 | unavailable | 0.0001943 |
| svd_full_wide_f32_256x512 | jax-cuda | 1 | 15 | 10.998 | 1836032 | unavailable | 0.1057 |
| svd_full_wide_f32_256x512 | jax-cuda-jit | 1 | 15 | 11.022 | 1836032 | unavailable | 0.1057 |
| svd_full_wide_f32_256x512 | pytorch-cuda | 1 | 15 | 11.070 | 1836032 | unavailable | 0.106 |
| svd_full_wide_f32_256x512 | tenferro-cuda-direct | 1 | 15 | 10.971 | 1836032 | unavailable | 0.1057 |
| svd_full_wide_f32_256x512 | tenferro-cuda-eager | 1 | 15 | 10.933 | 1836032 | unavailable | 0.1057 |
| svd_full_wide_f64_256x512 | jax-cuda | 1 | 15 | 17.815 | 3672064 | unavailable | 7.938e-05 |
| svd_full_wide_f64_256x512 | jax-cuda-jit | 1 | 15 | 17.818 | 3672064 | unavailable | 7.938e-05 |
| svd_full_wide_f64_256x512 | pytorch-cuda | 1 | 15 | 18.191 | 3672064 | unavailable | 7.938e-05 |
| svd_full_wide_f64_256x512 | tenferro-cuda-direct | 1 | 15 | 17.730 | 3672064 | unavailable | 7.938e-05 |
| svd_full_wide_f64_256x512 | tenferro-cuda-eager | 1 | 15 | 17.793 | 3672064 | unavailable | 7.938e-05 |
| svdvals_f32_512x512 | jax-cuda | 1 | 15 | 31.236 | 1050624 | unavailable | 0.1547 |
| svdvals_f32_512x512 | jax-cuda-jit | 1 | 15 | 31.285 | 1050624 | unavailable | 0.1547 |
| svdvals_f32_512x512 | pytorch-cuda | 1 | 15 | 34.101 | 1050624 | unavailable | 0.1733 |
| svdvals_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 31.163 | 1050624 | unavailable | 0.1547 |
| svdvals_f32_512x512 | tenferro-cuda-eager | — | — | — | 1050624 | unavailable | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_512x512 | jax-cuda | 1 | 15 | 77.663 | 2101248 | unavailable | 0.0002063 |
| svdvals_f64_512x512 | jax-cuda-jit | 1 | 15 | 77.837 | 2101248 | unavailable | 0.0002063 |
| svdvals_f64_512x512 | pytorch-cuda | 1 | 15 | 84.430 | 2101248 | unavailable | 0.0002066 |
| svdvals_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 77.755 | 2101248 | unavailable | 0.0002063 |
| svdvals_f64_512x512 | tenferro-cuda-eager | — | — | — | 2101248 | unavailable | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| triangular_solve_f32_512x512 | jax-cuda | 16 | 15 | 2.953 | 1081344 | unavailable | 0.0007877 |
| triangular_solve_f32_512x512 | jax-cuda-jit | 16 | 15 | 2.923 | 1081344 | unavailable | 0.0007877 |
| triangular_solve_f32_512x512 | pytorch-cuda | 32 | 15 | 3.856 | 1081344 | unavailable | 0.0006008 |
| triangular_solve_f32_512x512 | tenferro-cuda-direct | 16 | 15 | 2.719 | 1081344 | unavailable | 0.0007877 |
| triangular_solve_f32_512x512 | tenferro-cuda-eager | 16 | 15 | 3.575 | 1081344 | unavailable | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda | 16 | 15 | 3.453 | 2162688 | unavailable | 1.157e-06 |
| triangular_solve_f64_512x512 | jax-cuda-jit | 16 | 15 | 3.455 | 2162688 | unavailable | 1.157e-06 |
| triangular_solve_f64_512x512 | pytorch-cuda | 8 | 15 | 2.456 | 2162688 | unavailable | 6.241e-07 |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | 16 | 15 | 3.112 | 2162688 | unavailable | 1.157e-06 |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | 16 | 15 | 3.945 | 2162688 | unavailable | 1.157e-06 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api_throughput
suite_file: benchmarks/gpu/linalg_public_api_throughput.yaml
timestamp: '2026-11-05T02:02:29Z'
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
    ./scripts/run_gpu_public_api.sh linalg_public_api_throughput
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
    data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261105_020224/fixtures/manifest.json: 9bc6d3164be9e92417459d741f7875f4416224b26ecb3d4bd0381358bee92930
    target/release/benchmark_gpu_public_api: 34546e3ed1766badcc680cf308a6262672e67901a681a3518564249bded54c46
  reference_numpy_version: 2.4.2
```
