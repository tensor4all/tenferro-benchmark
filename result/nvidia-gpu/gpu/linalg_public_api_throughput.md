# gpu/linalg_public_api_throughput: CUDA public API benchmark

Measured cells: **308**. Unsupported cells: **102**, including **28** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api/20261102_210935/`.

This page is the throughput view: it holds the 82 cases whose largest matrix dimension is at least 256. Rows are the published measurements of that class from the run above, unchanged.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_512x512 | 285.479 (0.618) | 2148.871 (16.305) | 275.240 (1.548) | 240.111 (1.220) | 239.082 (0.687) |
| cholesky_f64_512x512 | 2141.610 (20.606) | 4032.273 (16.049) | 325.708 (1.898) | 300.422 (7.476) | 298.691 (1.383) |
| svdvals_f32_512x512 | 31065.968 (14.275) | unsupported | 34148.647 (10.954) | 31626.475 (335.176) | 31260.894 (206.732) |
| svdvals_f64_512x512 | 77516.391 (26.072) | unsupported | 84564.751 (16.875) | 77713.747 (404.304) | 77738.951 (124.500) |
| eigvalsh_f32_512x512 | 5197.481 (23.680) | 7081.793 (11.256) | 5223.322 (10.165) | 5817.993 (12.315) | 5714.039 (76.492) |
| eigvalsh_f64_512x512 | 5571.232 (15.345) | 8025.464 (17.150) | 5584.105 (7.361) | 6102.222 (476.261) | 6032.760 (24.885) |
| pinv_f32_512x512 | unsupported | unsupported | 34282.021 (9.910) | 31690.857 (448.609) | 31800.522 (377.688) |
| pinv_f64_512x512 | unsupported | unsupported | 84753.837 (37.377) | 78076.270 (20.005) | 78052.922 (386.638) |
| inv_f32_512x512 | unsupported | unsupported | 1494.589 (5.152) | 1524.021 (9.958) | 1509.435 (14.158) |
| inv_f64_512x512 | unsupported | unsupported | 1792.580 (7.458) | 1820.195 (8.585) | 1813.065 (18.398) |
| det_f32_512x512 | 1620.268 (11.556) | 3286.248 (16.560) | 1174.319 (3.483) | 1207.300 (2.880) | 1204.290 (6.100) |
| det_f64_512x512 | 2267.824 (11.186) | 4319.472 (48.627) | 1311.829 (5.797) | 1371.321 (14.861) | 1371.771 (13.805) |
| norm_f32_512x512 | 87.644 (1.999) | 122.144 (0.377) | 9.006 (0.041) | 45.216 (5.315) | 46.020 (1.568) |
| norm_f64_512x512 | 89.225 (0.481) | 119.660 (1.881) | 9.403 (0.038) | 44.601 (0.411) | 46.162 (0.178) |
| triangular_solve_f32_512x512 | 169.273 (0.638) | 2013.036 (7.181) | 120.670 (0.073) | 181.569 (1.243) | 184.475 (2.211) |
| triangular_solve_f64_512x512 | 194.000 (0.960) | 2059.668 (23.115) | 308.119 (0.156) | 215.491 (2.255) | 215.957 (1.596) |
| svd_full_square_f32_256x256 | 10774.513 (10.610) | 12624.685 (39.757) | 11445.348 (13.700) | 10816.248 (54.188) | 10805.627 (538.422) |
| svd_full_tall_f32_512x256 | 11152.205 (15.825) | 12980.847 (5.380) | 11937.364 (9.761) | 11238.751 (24.926) | 11671.876 (103.784) |
| svd_full_wide_f32_256x512 | 10903.527 (17.796) | 12719.808 (15.340) | 11131.837 (12.886) | 11060.045 (454.974) | 10960.223 (492.266) |
| slogdet_positive_f32_256x256 | 782.108 (2.250) | 2484.592 (17.791) | 532.070 (0.536) | 517.220 (6.920) | 523.360 (1.418) |
| slogdet_negative_f32_256x256 | 783.738 (2.292) | 2467.601 (101.468) | 533.173 (0.869) | 535.030 (1.140) | 523.372 (3.574) |
| svd_full_square_f64_256x256 | 25702.093 (9.391) | 27464.404 (38.712) | 25803.312 (14.076) | 26210.837 (132.114) | 25811.664 (40.426) |
| svd_full_tall_f64_512x256 | 19520.250 (11.975) | 21266.260 (26.951) | 19906.527 (18.391) | 19934.689 (403.723) | 19624.859 (37.457) |
| svd_full_wide_f64_256x512 | 17637.728 (19.201) | 19520.162 (87.323) | 18262.073 (16.431) | 18035.856 (448.009) | 17729.216 (36.697) |
| slogdet_positive_f64_256x256 | 798.269 (2.509) | 2515.623 (65.171) | 567.279 (0.879) | 554.496 (5.918) | 556.151 (6.945) |
| slogdet_negative_f64_256x256 | 798.951 (2.609) | 2516.083 (15.345) | 572.414 (0.980) | 577.139 (1.104) | 566.816 (2.135) |
| svd_full_square_c32_256x256 | 20293.225 (7.695) | 22012.865 (87.513) | 22152.822 (11.935) | 20360.073 (30.016) | 20369.064 (16.331) |
| svd_full_tall_c32_512x256 | 19617.984 (20.351) | 21397.684 (18.946) | 20058.012 (22.441) | 19648.689 (15.625) | 19648.030 (10.910) |
| svd_full_wide_c32_256x512 | 19815.470 (13.436) | 21625.161 (17.885) | 20272.799 (8.500) | 20041.763 (344.737) | 19805.956 (24.060) |
| slogdet_positive_c32_256x256 | 830.755 (3.879) | 2526.093 (13.615) | 574.501 (1.759) | 562.396 (3.626) | 562.324 (1.150) |
| slogdet_negative_c32_256x256 | 832.505 (3.226) | 2528.724 (21.892) | 579.424 (1.399) | 574.822 (1.654) | 564.291 (1.306) |
| svd_full_square_c64_256x256 | 59992.978 (17.196) | 61775.064 (121.340) | 63674.882 (43.051) | 60286.202 (240.483) | 60156.351 (389.153) |
| svd_full_tall_c64_512x256 | 42626.228 (15.861) | 44579.718 (242.028) | 41016.005 (21.130) | 41064.927 (332.441) | 41001.967 (498.257) |
| svd_full_wide_c64_256x512 | 42802.934 (41.627) | 44344.930 (201.712) | 41597.143 (23.541) | 40884.692 (164.397) | 41053.159 (398.318) |
| slogdet_positive_c64_256x256 | 896.250 (2.216) | 2527.063 (90.173) | 627.506 (0.914) | 629.678 (2.627) | 640.861 (1.235) |
| slogdet_negative_c64_256x256 | 891.174 (2.544) | 2469.141 (16.581) | 631.663 (1.240) | 626.033 (1.675) | 625.098 (5.676) |
| householder_qr_square_f32_256x256 | 924.743 (3.715) | 2739.690 (90.759) | 909.695 (1.935) | 928.128 (9.397) | 937.909 (3.105) |
| householder_qr_tall_f32_512x256 | 1213.255 (4.418) | 3101.042 (48.902) | 1194.629 (3.390) | 1275.287 (46.919) | 1289.673 (40.184) |
| householder_qr_wide_f32_256x512 | 1105.201 (10.640) | 3028.289 (51.091) | 1099.646 (12.313) | 1201.645 (31.886) | 1216.370 (13.433) |
| householder_qr_square_f64_256x256 | 1037.624 (7.568) | 2887.975 (10.865) | 1008.568 (4.083) | 1061.000 (37.039) | 1077.466 (23.161) |
| householder_qr_tall_f64_512x256 | 1347.324 (5.650) | 3179.234 (30.296) | 1320.843 (6.975) | 1417.037 (29.951) | 1401.476 (24.658) |
| householder_qr_wide_f64_256x512 | 1274.887 (6.368) | 3059.011 (18.346) | 1231.961 (11.880) | 1322.369 (33.649) | 1331.044 (32.931) |
| householder_qr_square_c32_256x256 | 1003.933 (3.115) | 2766.021 (12.036) | 969.985 (2.374) | 1012.339 (21.005) | 1025.539 (12.303) |
| householder_qr_tall_c32_512x256 | 1312.433 (10.063) | 3075.651 (79.392) | 1269.507 (5.623) | 1371.446 (19.478) | 1351.279 (13.883) |
| householder_qr_wide_c32_256x512 | 1227.715 (7.515) | 3093.052 (59.927) | 1199.604 (10.848) | 1299.958 (37.071) | 1275.037 (11.653) |
| slogdet_phase_c32_256x256 | 831.655 (2.707) | 2525.733 (29.831) | 574.019 (0.911) | 554.468 (4.019) | 581.514 (4.197) |
| householder_qr_square_c64_256x256 | 1206.415 (3.293) | 3046.021 (16.255) | 1173.659 (4.318) | 1238.471 (42.056) | 1236.261 (7.915) |
| householder_qr_tall_c64_512x256 | 2249.254 (7.680) | 4116.255 (92.833) | 1639.889 (4.770) | 1656.195 (7.280) | 1676.260 (4.232) |
| householder_qr_wide_c64_256x512 | 2116.240 (15.381) | 4031.573 (59.077) | 1484.489 (11.748) | 1576.282 (12.395) | 1556.857 (12.683) |
| slogdet_phase_c64_256x256 | 896.479 (3.155) | 2580.635 (11.116) | 628.348 (0.681) | 629.966 (4.474) | 631.218 (11.198) |
| householder_from_factors_default_f32_512x256 | 1430.822 (6.923) | 3194.356 (117.199) | unsupported | unsupported | unsupported |
| householder_append_default_f32_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f32_512x256 | 34.698 (0.201) | 48.570 (0.301) | 5.033 (0.026) | 40.069 (0.283) | 41.461 (0.204) |
| householder_q_columns_prefix_f32_512x256 | unsupported | unsupported | 64.817 (2.031) | 151.903 (12.971) | 103.556 (19.007) |
| householder_q_columns_complement_f32_512x256 | unsupported | unsupported | 166.399 (0.212) | unsupported | unsupported |
| householder_from_factors_default_f64_512x256 | 1551.716 (10.860) | 3304.149 (76.438) | unsupported | unsupported | unsupported |
| householder_append_default_f64_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f64_512x256 | 34.574 (0.171) | 48.938 (0.237) | 5.069 (0.023) | 43.905 (3.066) | 41.426 (0.362) |
| householder_q_columns_prefix_f64_512x256 | unsupported | unsupported | 75.228 (0.298) | 150.168 (14.547) | 105.807 (1.945) |
| householder_q_columns_complement_f64_512x256 | unsupported | unsupported | 165.741 (0.220) | unsupported | unsupported |
| householder_from_factors_default_c32_512x256 | 1540.480 (10.388) | 3277.198 (25.331) | unsupported | unsupported | unsupported |
| householder_append_default_c32_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c32_512x256 | 37.896 (0.147) | 49.325 (0.142) | 5.253 (0.011) | 44.365 (6.446) | 41.352 (0.240) |
| householder_q_columns_prefix_c32_512x256 | unsupported | unsupported | 91.455 (0.033) | 151.072 (10.639) | 97.739 (1.146) |
| householder_q_columns_complement_c32_512x256 | unsupported | unsupported | 251.514 (0.178) | unsupported | unsupported |
| householder_from_factors_default_c64_512x256 | 1893.182 (7.155) | 3650.221 (23.245) | unsupported | unsupported | unsupported |
| householder_append_default_c64_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c64_512x256 | 37.194 (0.428) | 45.174 (2.803) | 5.230 (0.023) | 48.592 (3.018) | 41.892 (0.161) |
| householder_q_columns_prefix_c64_512x256 | unsupported | unsupported | 73.864 (0.229) | 156.098 (6.817) | 106.284 (0.492) |
| householder_q_columns_complement_c64_512x256 | unsupported | unsupported | 253.176 (0.367) | unsupported | unsupported |
| lu_square_f32_s256 | 623.948 (2.609) | 2403.559 (101.628) | unsupported | unsupported | unsupported |
| lu_square_f64_s256 | 659.114 (2.861) | 2427.000 (120.064) | unsupported | unsupported | unsupported |
| lu_square_c32_s256 | 662.224 (1.536) | 2422.110 (86.122) | unsupported | unsupported | unsupported |
| lu_square_c64_s256 | 718.776 (1.480) | 2551.664 (108.704) | unsupported | unsupported | unsupported |
| lu_tall_f32_s256 | 769.320 (2.955) | 2616.876 (34.637) | unsupported | unsupported | unsupported |
| lu_tall_f64_s256 | 838.435 (2.591) | 2689.689 (13.680) | unsupported | unsupported | unsupported |
| lu_tall_c32_s256 | 816.227 (1.624) | 2663.018 (13.620) | unsupported | unsupported | unsupported |
| lu_tall_c64_s256 | 2290.386 (8.555) | 4187.028 (19.986) | unsupported | unsupported | unsupported |
| lu_wide_f32_s256 | 672.882 (1.264) | 2515.403 (16.256) | unsupported | unsupported | unsupported |
| lu_wide_f64_s256 | 749.830 (1.150) | 2600.426 (23.381) | unsupported | unsupported | unsupported |
| lu_wide_c32_s256 | 730.339 (1.778) | 2586.626 (21.670) | unsupported | unsupported | unsupported |
| lu_wide_c64_s256 | 2285.445 (11.745) | 4167.597 (14.336) | unsupported | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| cholesky_f32_512x512 | tenferro-cuda-direct | 8 | 15 | 2.284 | 0.0003558 |
| cholesky_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.142 | 4.418e-07 |
| svdvals_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 31.066 | 0.1547 |
| svdvals_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 77.516 | 0.0002063 |
| eigvalsh_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 5.197 | 0.008342 |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 5.571 | 1.55e-06 |
| pinv_f32_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| det_f32_512x512 | tenferro-cuda-direct | 2 | 15 | 3.241 | 0.004748 |
| det_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.268 | 1.426e-05 |
| norm_f32_512x512 | tenferro-cuda-direct | 32 | 15 | 2.805 | 0 |
| norm_f64_512x512 | tenferro-cuda-direct | 32 | 15 | 2.855 | 1.909e-07 |
| triangular_solve_f32_512x512 | tenferro-cuda-direct | 16 | 15 | 2.708 | 0.0007877 |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | 16 | 15 | 3.104 | 1.157e-06 |
| svd_full_square_f32_256x256 | tenferro-cuda-direct | 1 | 15 | 10.775 | 0.1152 |
| svd_full_tall_f32_512x256 | tenferro-cuda-direct | 1 | 15 | 11.152 | 0.1022 |
| svd_full_wide_f32_256x512 | tenferro-cuda-direct | 1 | 15 | 10.904 | 0.1057 |
| slogdet_positive_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.128 | 0.0005003 |
| slogdet_negative_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.135 | 0.0005003 |
| svd_full_square_f64_256x256 | tenferro-cuda-direct | 1 | 15 | 25.702 | 8.171e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 19.520 | 7.76e-05 |
| svd_full_wide_f64_256x512 | tenferro-cuda-direct | 1 | 15 | 17.638 | 7.938e-05 |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.193 | 1.864e-06 |
| slogdet_negative_f64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.196 | 1.864e-06 |
| svd_full_square_c32_256x256 | tenferro-cuda-direct | 1 | 15 | 20.293 | 0.133 |
| svd_full_tall_c32_512x256 | tenferro-cuda-direct | 1 | 15 | 19.618 | 0.1126 |
| svd_full_wide_c32_256x512 | tenferro-cuda-direct | 1 | 15 | 19.815 | 0.1103 |
| slogdet_positive_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.323 | 0.01211 |
| slogdet_negative_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.330 | 0.01211 |
| svd_full_square_c64_256x256 | tenferro-cuda-direct | 1 | 15 | 59.993 | 0.000204 |
| svd_full_tall_c64_512x256 | tenferro-cuda-direct | 1 | 15 | 42.626 | 0.0002022 |
| svd_full_wide_c64_256x512 | tenferro-cuda-direct | 1 | 15 | 42.803 | 0.0001943 |
| slogdet_positive_c64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.585 | 1.025e-06 |
| slogdet_negative_c64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.565 | 1.025e-06 |
| householder_qr_square_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.699 | 0.001075 |
| householder_qr_tall_f32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.427 | 0.001435 |
| householder_qr_wide_f32_256x512 | tenferro-cuda-direct | 2 | 15 | 2.210 | 0.0009125 |
| householder_qr_square_f64_256x256 | tenferro-cuda-direct | 2 | 15 | 2.075 | 1.098e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 2.695 | 9.992e-07 |
| householder_qr_wide_f64_256x512 | tenferro-cuda-direct | 2 | 15 | 2.550 | 1.649e-06 |
| householder_qr_square_c32_256x256 | tenferro-cuda-direct | 2 | 15 | 2.008 | 0.001198 |
| householder_qr_tall_c32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.625 | 0.001185 |
| householder_qr_wide_c32_256x512 | tenferro-cuda-direct | 2 | 15 | 2.455 | 0.001329 |
| slogdet_phase_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.327 | 0.01211 |
| householder_qr_square_c64_256x256 | tenferro-cuda-direct | 2 | 15 | 2.413 | 1.208e-06 |
| householder_qr_tall_c64_512x256 | tenferro-cuda-direct | 1 | 15 | 2.249 | 1.11e-06 |
| householder_qr_wide_c64_256x512 | tenferro-cuda-direct | 1 | 15 | 2.116 | 1.206e-06 |
| slogdet_phase_c64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.586 | 1.025e-06 |
| householder_from_factors_default_f32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.862 | 0.0008522 |
| householder_append_default_f32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f32_512x256 | tenferro-cuda-direct | 64 | 15 | 2.221 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 3.103 | 8.882e-07 |
| householder_append_default_f64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | 64 | 15 | 2.213 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_512x256 | tenferro-cuda-direct | 2 | 15 | 3.081 | 0.001103 |
| householder_append_default_c32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c32_512x256 | tenferro-cuda-direct | 64 | 15 | 2.425 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c64_512x256 | tenferro-cuda-direct | 2 | 15 | 3.786 | 9.992e-07 |
| householder_append_default_c64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c64_512x256 | tenferro-cuda-direct | 64 | 15 | 2.380 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| lu_square_f32_s256 | tenferro-cuda-direct | 4 | 15 | 2.496 | 0.001428 |
| lu_square_f64_s256 | tenferro-cuda-direct | 4 | 15 | 2.636 | 1.102e-06 |
| lu_square_c32_s256 | tenferro-cuda-direct | 4 | 15 | 2.649 | 0.001541 |
| lu_square_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.875 | 1.431e-06 |
| lu_tall_f32_s256 | tenferro-cuda-direct | 4 | 15 | 3.077 | 0.001428 |
| lu_tall_f64_s256 | tenferro-cuda-direct | 4 | 15 | 3.354 | 1.102e-06 |
| lu_tall_c32_s256 | tenferro-cuda-direct | 4 | 15 | 3.265 | 0.002239 |
| lu_tall_c64_s256 | tenferro-cuda-direct | 1 | 15 | 2.290 | 1.542e-06 |
| lu_wide_f32_s256 | tenferro-cuda-direct | 4 | 15 | 2.692 | 0.001297 |
| lu_wide_f64_s256 | tenferro-cuda-direct | 4 | 15 | 2.999 | 8.806e-07 |
| lu_wide_c32_s256 | tenferro-cuda-direct | 4 | 15 | 2.921 | 0.002021 |
| lu_wide_c64_s256 | tenferro-cuda-direct | 1 | 15 | 2.285 | 1.433e-06 |
| cholesky_f32_512x512 | tenferro-cuda-eager | 1 | 15 | 2.149 | 0.0003558 |
| cholesky_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 4.032 | 4.418e-07 |
| svdvals_f32_512x512 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_512x512 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| eigvalsh_f32_512x512 | tenferro-cuda-eager | 1 | 15 | 7.082 | 0.008342 |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 8.025 | 1.55e-06 |
| pinv_f32_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f32_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| det_f32_512x512 | tenferro-cuda-eager | 1 | 15 | 3.286 | 0.004748 |
| det_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 4.319 | 1.426e-05 |
| norm_f32_512x512 | tenferro-cuda-eager | 32 | 15 | 3.909 | 0 |
| norm_f64_512x512 | tenferro-cuda-eager | 32 | 15 | 3.829 | 1.909e-07 |
| triangular_solve_f32_512x512 | tenferro-cuda-eager | 1 | 15 | 2.013 | 0.0007877 |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 2.060 | 1.157e-06 |
| svd_full_square_f32_256x256 | tenferro-cuda-eager | 1 | 15 | 12.625 | 0.1152 |
| svd_full_tall_f32_512x256 | tenferro-cuda-eager | 1 | 15 | 12.981 | 0.1022 |
| svd_full_wide_f32_256x512 | tenferro-cuda-eager | 1 | 15 | 12.720 | 0.1057 |
| slogdet_positive_f32_256x256 | tenferro-cuda-eager | 1 | 15 | 2.485 | 0.0005003 |
| slogdet_negative_f32_256x256 | tenferro-cuda-eager | 1 | 15 | 2.468 | 0.0005003 |
| svd_full_square_f64_256x256 | tenferro-cuda-eager | 1 | 15 | 27.464 | 8.171e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 21.266 | 7.76e-05 |
| svd_full_wide_f64_256x512 | tenferro-cuda-eager | 1 | 15 | 19.520 | 7.938e-05 |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | 1 | 15 | 2.516 | 1.864e-06 |
| slogdet_negative_f64_256x256 | tenferro-cuda-eager | 1 | 15 | 2.516 | 1.864e-06 |
| svd_full_square_c32_256x256 | tenferro-cuda-eager | 1 | 15 | 22.013 | 0.133 |
| svd_full_tall_c32_512x256 | tenferro-cuda-eager | 1 | 15 | 21.398 | 0.1126 |
| svd_full_wide_c32_256x512 | tenferro-cuda-eager | 1 | 15 | 21.625 | 0.1103 |
| slogdet_positive_c32_256x256 | tenferro-cuda-eager | 1 | 15 | 2.526 | 0.01211 |
| slogdet_negative_c32_256x256 | tenferro-cuda-eager | 1 | 15 | 2.529 | 0.01211 |
| svd_full_square_c64_256x256 | tenferro-cuda-eager | 1 | 15 | 61.775 | 0.000204 |
| svd_full_tall_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 44.580 | 0.0002022 |
| svd_full_wide_c64_256x512 | tenferro-cuda-eager | 1 | 15 | 44.345 | 0.0001943 |
| slogdet_positive_c64_256x256 | tenferro-cuda-eager | 1 | 15 | 2.527 | 1.025e-06 |
| slogdet_negative_c64_256x256 | tenferro-cuda-eager | 1 | 15 | 2.469 | 1.025e-06 |
| householder_qr_square_f32_256x256 | tenferro-cuda-eager | 1 | 15 | 2.740 | 0.001075 |
| householder_qr_tall_f32_512x256 | tenferro-cuda-eager | 1 | 15 | 3.101 | 0.001435 |
| householder_qr_wide_f32_256x512 | tenferro-cuda-eager | 1 | 15 | 3.028 | 0.0009125 |
| householder_qr_square_f64_256x256 | tenferro-cuda-eager | 1 | 15 | 2.888 | 1.098e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 3.179 | 9.992e-07 |
| householder_qr_wide_f64_256x512 | tenferro-cuda-eager | 1 | 15 | 3.059 | 1.649e-06 |
| householder_qr_square_c32_256x256 | tenferro-cuda-eager | 1 | 15 | 2.766 | 0.001198 |
| householder_qr_tall_c32_512x256 | tenferro-cuda-eager | 1 | 15 | 3.076 | 0.001185 |
| householder_qr_wide_c32_256x512 | tenferro-cuda-eager | 1 | 15 | 3.093 | 0.001329 |
| slogdet_phase_c32_256x256 | tenferro-cuda-eager | 1 | 15 | 2.526 | 0.01211 |
| householder_qr_square_c64_256x256 | tenferro-cuda-eager | 1 | 15 | 3.046 | 1.208e-06 |
| householder_qr_tall_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 4.116 | 1.11e-06 |
| householder_qr_wide_c64_256x512 | tenferro-cuda-eager | 1 | 15 | 4.032 | 1.206e-06 |
| slogdet_phase_c64_256x256 | tenferro-cuda-eager | 1 | 15 | 2.581 | 1.025e-06 |
| householder_from_factors_default_f32_512x256 | tenferro-cuda-eager | 1 | 15 | 3.194 | 0.0008522 |
| householder_append_default_f32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f32_512x256 | tenferro-cuda-eager | 64 | 15 | 3.109 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 3.304 | 8.882e-07 |
| householder_append_default_f64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | 64 | 15 | 3.132 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_512x256 | tenferro-cuda-eager | 1 | 15 | 3.277 | 0.001103 |
| householder_append_default_c32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c32_512x256 | tenferro-cuda-eager | 64 | 15 | 3.157 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 3.650 | 9.992e-07 |
| householder_append_default_c64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c64_512x256 | tenferro-cuda-eager | 64 | 15 | 2.891 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| lu_square_f32_s256 | tenferro-cuda-eager | 1 | 15 | 2.404 | 0.001428 |
| lu_square_f64_s256 | tenferro-cuda-eager | 1 | 15 | 2.427 | 1.102e-06 |
| lu_square_c32_s256 | tenferro-cuda-eager | 1 | 15 | 2.422 | 0.001541 |
| lu_square_c64_s256 | tenferro-cuda-eager | 1 | 15 | 2.552 | 1.431e-06 |
| lu_tall_f32_s256 | tenferro-cuda-eager | 1 | 15 | 2.617 | 0.001428 |
| lu_tall_f64_s256 | tenferro-cuda-eager | 1 | 15 | 2.690 | 1.102e-06 |
| lu_tall_c32_s256 | tenferro-cuda-eager | 1 | 15 | 2.663 | 0.002239 |
| lu_tall_c64_s256 | tenferro-cuda-eager | 1 | 15 | 4.187 | 1.542e-06 |
| lu_wide_f32_s256 | tenferro-cuda-eager | 1 | 15 | 2.515 | 0.001297 |
| lu_wide_f64_s256 | tenferro-cuda-eager | 1 | 15 | 2.600 | 8.806e-07 |
| lu_wide_c32_s256 | tenferro-cuda-eager | 1 | 15 | 2.587 | 0.002021 |
| lu_wide_c64_s256 | tenferro-cuda-eager | 1 | 15 | 4.168 | 1.433e-06 |
| cholesky_f32_512x512 | pytorch-cuda | 8 | 15 | 2.202 | 0.0003558 |
| cholesky_f64_512x512 | pytorch-cuda | 8 | 15 | 2.606 | 4.418e-07 |
| svdvals_f32_512x512 | pytorch-cuda | 1 | 15 | 34.149 | 0.1733 |
| svdvals_f64_512x512 | pytorch-cuda | 1 | 15 | 84.565 | 0.0002066 |
| eigvalsh_f32_512x512 | pytorch-cuda | 1 | 15 | 5.223 | 0.008342 |
| eigvalsh_f64_512x512 | pytorch-cuda | 1 | 15 | 5.584 | 1.55e-06 |
| pinv_f32_512x512 | pytorch-cuda | 1 | 15 | 34.282 | 0.01174 |
| pinv_f64_512x512 | pytorch-cuda | 1 | 15 | 84.754 | 8.747e-06 |
| inv_f32_512x512 | pytorch-cuda | 2 | 15 | 2.989 | 0.002107 |
| inv_f64_512x512 | pytorch-cuda | 2 | 15 | 3.585 | 1.85e-06 |
| det_f32_512x512 | pytorch-cuda | 2 | 15 | 2.349 | 0.007965 |
| det_f64_512x512 | pytorch-cuda | 2 | 15 | 2.624 | 1.369e-05 |
| norm_f32_512x512 | pytorch-cuda | 256 | 15 | 2.305 | 0 |
| norm_f64_512x512 | pytorch-cuda | 256 | 15 | 2.407 | 1.909e-07 |
| triangular_solve_f32_512x512 | pytorch-cuda | 32 | 15 | 3.861 | 0.0006008 |
| triangular_solve_f64_512x512 | pytorch-cuda | 8 | 15 | 2.465 | 6.241e-07 |
| svd_full_square_f32_256x256 | pytorch-cuda | 1 | 15 | 11.445 | 0.1135 |
| svd_full_tall_f32_512x256 | pytorch-cuda | 1 | 15 | 11.937 | 0.1033 |
| svd_full_wide_f32_256x512 | pytorch-cuda | 1 | 15 | 11.132 | 0.106 |
| slogdet_positive_f32_256x256 | pytorch-cuda | 4 | 15 | 2.128 | 0.0005003 |
| slogdet_negative_f32_256x256 | pytorch-cuda | 4 | 15 | 2.133 | 0.0005003 |
| svd_full_square_f64_256x256 | pytorch-cuda | 1 | 15 | 25.803 | 8.105e-05 |
| svd_full_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 19.907 | 7.76e-05 |
| svd_full_wide_f64_256x512 | pytorch-cuda | 1 | 15 | 18.262 | 7.938e-05 |
| slogdet_positive_f64_256x256 | pytorch-cuda | 4 | 15 | 2.269 | 1.708e-06 |
| slogdet_negative_f64_256x256 | pytorch-cuda | 4 | 15 | 2.290 | 1.708e-06 |
| svd_full_square_c32_256x256 | pytorch-cuda | 1 | 15 | 22.153 | 0.1305 |
| svd_full_tall_c32_512x256 | pytorch-cuda | 1 | 15 | 20.058 | 0.1089 |
| svd_full_wide_c32_256x512 | pytorch-cuda | 1 | 15 | 20.273 | 0.108 |
| slogdet_positive_c32_256x256 | pytorch-cuda | 4 | 15 | 2.298 | 0.01211 |
| slogdet_negative_c32_256x256 | pytorch-cuda | 4 | 15 | 2.318 | 0.01211 |
| svd_full_square_c64_256x256 | pytorch-cuda | 1 | 15 | 63.675 | 0.0002064 |
| svd_full_tall_c64_512x256 | pytorch-cuda | 1 | 15 | 41.016 | 0.0001992 |
| svd_full_wide_c64_256x512 | pytorch-cuda | 1 | 15 | 41.597 | 0.0001941 |
| slogdet_positive_c64_256x256 | pytorch-cuda | 4 | 15 | 2.510 | 1.23e-06 |
| slogdet_negative_c64_256x256 | pytorch-cuda | 4 | 15 | 2.527 | 1.23e-06 |
| householder_qr_square_f32_256x256 | pytorch-cuda | 4 | 15 | 3.639 | 0.001043 |
| householder_qr_tall_f32_512x256 | pytorch-cuda | 2 | 15 | 2.389 | 0.001475 |
| householder_qr_wide_f32_256x512 | pytorch-cuda | 2 | 15 | 2.199 | 0.0009736 |
| householder_qr_square_f64_256x256 | pytorch-cuda | 2 | 15 | 2.017 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | pytorch-cuda | 2 | 15 | 2.642 | 1.887e-06 |
| householder_qr_wide_f64_256x512 | pytorch-cuda | 2 | 15 | 2.464 | 1.759e-06 |
| householder_qr_square_c32_256x256 | pytorch-cuda | 4 | 15 | 3.880 | 0.001258 |
| householder_qr_tall_c32_512x256 | pytorch-cuda | 2 | 15 | 2.539 | 0.001161 |
| householder_qr_wide_c32_256x512 | pytorch-cuda | 2 | 15 | 2.399 | 0.001226 |
| slogdet_phase_c32_256x256 | pytorch-cuda | 4 | 15 | 2.296 | 0.01233 |
| householder_qr_square_c64_256x256 | pytorch-cuda | 2 | 15 | 2.347 | 2.109e-06 |
| householder_qr_tall_c64_512x256 | pytorch-cuda | 2 | 15 | 3.280 | 2.942e-06 |
| householder_qr_wide_c64_256x512 | pytorch-cuda | 2 | 15 | 2.969 | 2.276e-06 |
| slogdet_phase_c64_256x256 | pytorch-cuda | 4 | 15 | 2.513 | 1.025e-06 |
| householder_from_factors_default_f32_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_512x256 | pytorch-cuda | 256 | 15 | 1.288 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | pytorch-cuda | 32 | 15 | 2.074 | 0.0001829 |
| householder_q_columns_complement_f32_512x256 | pytorch-cuda | 16 | 15 | 2.662 | 8.761e-05 |
| householder_from_factors_default_f64_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_512x256 | pytorch-cuda | 256 | 15 | 1.298 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | 32 | 15 | 2.407 | 1.136e-07 |
| householder_q_columns_complement_f64_512x256 | pytorch-cuda | 16 | 15 | 2.652 | 7.26e-07 |
| householder_from_factors_default_c32_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_512x256 | pytorch-cuda | 512 | 15 | 2.690 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | pytorch-cuda | 32 | 15 | 2.927 | 0.0001246 |
| householder_q_columns_complement_c32_512x256 | pytorch-cuda | 8 | 15 | 2.012 | 0.0001141 |
| householder_from_factors_default_c64_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_512x256 | pytorch-cuda | 256 | 15 | 1.339 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | pytorch-cuda | 32 | 15 | 2.364 | 5.968e-08 |
| householder_q_columns_complement_c64_512x256 | pytorch-cuda | 8 | 15 | 2.025 | 7.282e-07 |
| lu_square_f32_s256 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s256 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s256 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s256 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s256 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s256 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s256 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s256 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s256 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s256 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s256 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| cholesky_f32_512x512 | jax-cuda | 16 | 15 | 3.842 | 0.0003558 |
| cholesky_f64_512x512 | jax-cuda | 8 | 15 | 2.403 | 4.418e-07 |
| svdvals_f32_512x512 | jax-cuda | 1 | 15 | 31.626 | 0.1547 |
| svdvals_f64_512x512 | jax-cuda | 1 | 15 | 77.714 | 0.0002063 |
| eigvalsh_f32_512x512 | jax-cuda | 1 | 15 | 5.818 | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda | 1 | 15 | 6.102 | 1.55e-06 |
| pinv_f32_512x512 | jax-cuda | 1 | 15 | 31.691 | 0.009697 |
| pinv_f64_512x512 | jax-cuda | 1 | 15 | 78.076 | 8.975e-06 |
| inv_f32_512x512 | jax-cuda | 2 | 15 | 3.048 | 0.002107 |
| inv_f64_512x512 | jax-cuda | 2 | 15 | 3.640 | 1.85e-06 |
| det_f32_512x512 | jax-cuda | 2 | 15 | 2.415 | 0.004748 |
| det_f64_512x512 | jax-cuda | 2 | 15 | 2.743 | 1.426e-05 |
| norm_f32_512x512 | jax-cuda | 64 | 15 | 2.894 | 0 |
| norm_f64_512x512 | jax-cuda | 64 | 15 | 2.854 | 1.909e-07 |
| triangular_solve_f32_512x512 | jax-cuda | 16 | 15 | 2.905 | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda | 16 | 15 | 3.448 | 1.157e-06 |
| svd_full_square_f32_256x256 | jax-cuda | 1 | 15 | 10.816 | 0.1152 |
| svd_full_tall_f32_512x256 | jax-cuda | 1 | 15 | 11.239 | 0.1022 |
| svd_full_wide_f32_256x512 | jax-cuda | 1 | 15 | 11.060 | 0.1057 |
| slogdet_positive_f32_256x256 | jax-cuda | 4 | 15 | 2.069 | 0.0005003 |
| slogdet_negative_f32_256x256 | jax-cuda | 4 | 15 | 2.140 | 0.0005003 |
| svd_full_square_f64_256x256 | jax-cuda | 1 | 15 | 26.211 | 8.171e-05 |
| svd_full_tall_f64_512x256 | jax-cuda | 1 | 15 | 19.935 | 7.76e-05 |
| svd_full_wide_f64_256x512 | jax-cuda | 1 | 15 | 18.036 | 7.938e-05 |
| slogdet_positive_f64_256x256 | jax-cuda | 4 | 15 | 2.218 | 1.864e-06 |
| slogdet_negative_f64_256x256 | jax-cuda | 4 | 15 | 2.309 | 1.864e-06 |
| svd_full_square_c32_256x256 | jax-cuda | 1 | 15 | 20.360 | 0.133 |
| svd_full_tall_c32_512x256 | jax-cuda | 1 | 15 | 19.649 | 0.1126 |
| svd_full_wide_c32_256x512 | jax-cuda | 1 | 15 | 20.042 | 0.1103 |
| slogdet_positive_c32_256x256 | jax-cuda | 4 | 15 | 2.250 | 0.01211 |
| slogdet_negative_c32_256x256 | jax-cuda | 4 | 15 | 2.299 | 0.01211 |
| svd_full_square_c64_256x256 | jax-cuda | 1 | 15 | 60.286 | 0.000204 |
| svd_full_tall_c64_512x256 | jax-cuda | 1 | 15 | 41.065 | 0.0002021 |
| svd_full_wide_c64_256x512 | jax-cuda | 1 | 15 | 40.885 | 0.0001946 |
| slogdet_positive_c64_256x256 | jax-cuda | 4 | 15 | 2.519 | 1.025e-06 |
| slogdet_negative_c64_256x256 | jax-cuda | 4 | 15 | 2.504 | 1.025e-06 |
| householder_qr_square_f32_256x256 | jax-cuda | 4 | 15 | 3.713 | 0.001043 |
| householder_qr_tall_f32_512x256 | jax-cuda | 2 | 15 | 2.551 | 0.001475 |
| householder_qr_wide_f32_256x512 | jax-cuda | 2 | 15 | 2.403 | 0.0009736 |
| householder_qr_square_f64_256x256 | jax-cuda | 2 | 15 | 2.122 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | jax-cuda | 2 | 15 | 2.834 | 1.887e-06 |
| householder_qr_wide_f64_256x512 | jax-cuda | 2 | 15 | 2.645 | 1.759e-06 |
| householder_qr_square_c32_256x256 | jax-cuda | 2 | 15 | 2.025 | 0.001258 |
| householder_qr_tall_c32_512x256 | jax-cuda | 2 | 15 | 2.743 | 0.001161 |
| householder_qr_wide_c32_256x512 | jax-cuda | 2 | 15 | 2.600 | 0.001226 |
| slogdet_phase_c32_256x256 | jax-cuda | 4 | 15 | 2.218 | 0.01233 |
| householder_qr_square_c64_256x256 | jax-cuda | 2 | 15 | 2.477 | 2.109e-06 |
| householder_qr_tall_c64_512x256 | jax-cuda | 2 | 15 | 3.312 | 2.942e-06 |
| householder_qr_wide_c64_256x512 | jax-cuda | 2 | 15 | 3.153 | 2.276e-06 |
| slogdet_phase_c64_256x256 | jax-cuda | 4 | 15 | 2.520 | 1.025e-06 |
| householder_from_factors_default_f32_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_512x256 | jax-cuda | 64 | 15 | 2.564 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | jax-cuda | 16 | 15 | 2.430 | 0.0001829 |
| householder_q_columns_complement_f32_512x256 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f64_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_512x256 | jax-cuda | 64 | 15 | 2.810 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | 16 | 15 | 2.403 | 1.136e-07 |
| householder_q_columns_complement_f64_512x256 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c32_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_512x256 | jax-cuda | 64 | 15 | 2.839 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | jax-cuda | 16 | 15 | 2.417 | 0.0001246 |
| householder_q_columns_complement_c32_512x256 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c64_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_512x256 | jax-cuda | 64 | 15 | 3.110 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | jax-cuda | 16 | 15 | 2.498 | 5.968e-08 |
| householder_q_columns_complement_c64_512x256 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| lu_square_f32_s256 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s256 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s256 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s256 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s256 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s256 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s256 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s256 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s256 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s256 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s256 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| cholesky_f32_512x512 | jax-cuda-jit | 16 | 15 | 3.825 | 0.0003558 |
| cholesky_f64_512x512 | jax-cuda-jit | 8 | 15 | 2.390 | 4.418e-07 |
| svdvals_f32_512x512 | jax-cuda-jit | 1 | 15 | 31.261 | 0.1547 |
| svdvals_f64_512x512 | jax-cuda-jit | 1 | 15 | 77.739 | 0.0002063 |
| eigvalsh_f32_512x512 | jax-cuda-jit | 1 | 15 | 5.714 | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda-jit | 1 | 15 | 6.033 | 1.55e-06 |
| pinv_f32_512x512 | jax-cuda-jit | 1 | 15 | 31.801 | 0.009581 |
| pinv_f64_512x512 | jax-cuda-jit | 1 | 15 | 78.053 | 8.975e-06 |
| inv_f32_512x512 | jax-cuda-jit | 2 | 15 | 3.019 | 0.002107 |
| inv_f64_512x512 | jax-cuda-jit | 2 | 15 | 3.626 | 1.85e-06 |
| det_f32_512x512 | jax-cuda-jit | 2 | 15 | 2.409 | 0.004748 |
| det_f64_512x512 | jax-cuda-jit | 2 | 15 | 2.744 | 1.426e-05 |
| norm_f32_512x512 | jax-cuda-jit | 64 | 15 | 2.945 | 0 |
| norm_f64_512x512 | jax-cuda-jit | 64 | 15 | 2.954 | 1.909e-07 |
| triangular_solve_f32_512x512 | jax-cuda-jit | 16 | 15 | 2.952 | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda-jit | 16 | 15 | 3.455 | 1.157e-06 |
| svd_full_square_f32_256x256 | jax-cuda-jit | 1 | 15 | 10.806 | 0.1152 |
| svd_full_tall_f32_512x256 | jax-cuda-jit | 1 | 15 | 11.672 | 0.1022 |
| svd_full_wide_f32_256x512 | jax-cuda-jit | 1 | 15 | 10.960 | 0.1057 |
| slogdet_positive_f32_256x256 | jax-cuda-jit | 4 | 15 | 2.093 | 0.0005003 |
| slogdet_negative_f32_256x256 | jax-cuda-jit | 4 | 15 | 2.093 | 0.0005003 |
| svd_full_square_f64_256x256 | jax-cuda-jit | 1 | 15 | 25.812 | 8.171e-05 |
| svd_full_tall_f64_512x256 | jax-cuda-jit | 1 | 15 | 19.625 | 7.76e-05 |
| svd_full_wide_f64_256x512 | jax-cuda-jit | 1 | 15 | 17.729 | 7.938e-05 |
| slogdet_positive_f64_256x256 | jax-cuda-jit | 4 | 15 | 2.225 | 1.864e-06 |
| slogdet_negative_f64_256x256 | jax-cuda-jit | 4 | 15 | 2.267 | 1.864e-06 |
| svd_full_square_c32_256x256 | jax-cuda-jit | 1 | 15 | 20.369 | 0.133 |
| svd_full_tall_c32_512x256 | jax-cuda-jit | 1 | 15 | 19.648 | 0.1126 |
| svd_full_wide_c32_256x512 | jax-cuda-jit | 1 | 15 | 19.806 | 0.1103 |
| slogdet_positive_c32_256x256 | jax-cuda-jit | 4 | 15 | 2.249 | 0.01211 |
| slogdet_negative_c32_256x256 | jax-cuda-jit | 4 | 15 | 2.257 | 0.01211 |
| svd_full_square_c64_256x256 | jax-cuda-jit | 1 | 15 | 60.156 | 0.000204 |
| svd_full_tall_c64_512x256 | jax-cuda-jit | 1 | 15 | 41.002 | 0.0002021 |
| svd_full_wide_c64_256x512 | jax-cuda-jit | 1 | 15 | 41.053 | 0.0001946 |
| slogdet_positive_c64_256x256 | jax-cuda-jit | 4 | 15 | 2.563 | 1.025e-06 |
| slogdet_negative_c64_256x256 | jax-cuda-jit | 4 | 15 | 2.500 | 1.025e-06 |
| householder_qr_square_f32_256x256 | jax-cuda-jit | 4 | 15 | 3.752 | 0.001043 |
| householder_qr_tall_f32_512x256 | jax-cuda-jit | 2 | 15 | 2.579 | 0.001475 |
| householder_qr_wide_f32_256x512 | jax-cuda-jit | 2 | 15 | 2.433 | 0.0009736 |
| householder_qr_square_f64_256x256 | jax-cuda-jit | 2 | 15 | 2.155 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | jax-cuda-jit | 2 | 15 | 2.803 | 1.887e-06 |
| householder_qr_wide_f64_256x512 | jax-cuda-jit | 2 | 15 | 2.662 | 1.759e-06 |
| householder_qr_square_c32_256x256 | jax-cuda-jit | 2 | 15 | 2.051 | 0.001258 |
| householder_qr_tall_c32_512x256 | jax-cuda-jit | 2 | 15 | 2.703 | 0.001161 |
| householder_qr_wide_c32_256x512 | jax-cuda-jit | 2 | 15 | 2.550 | 0.001226 |
| slogdet_phase_c32_256x256 | jax-cuda-jit | 4 | 15 | 2.326 | 0.01233 |
| householder_qr_square_c64_256x256 | jax-cuda-jit | 2 | 15 | 2.473 | 2.109e-06 |
| householder_qr_tall_c64_512x256 | jax-cuda-jit | 2 | 15 | 3.353 | 2.942e-06 |
| householder_qr_wide_c64_256x512 | jax-cuda-jit | 2 | 15 | 3.114 | 2.276e-06 |
| slogdet_phase_c64_256x256 | jax-cuda-jit | 4 | 15 | 2.525 | 1.025e-06 |
| householder_from_factors_default_f32_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_512x256 | jax-cuda-jit | 64 | 15 | 2.654 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | jax-cuda-jit | 16 | 15 | 1.657 | 0.0001829 |
| householder_q_columns_complement_f32_512x256 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f64_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_512x256 | jax-cuda-jit | 64 | 15 | 2.651 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | jax-cuda-jit | 32 | 15 | 3.386 | 1.136e-07 |
| householder_q_columns_complement_f64_512x256 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c32_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_512x256 | jax-cuda-jit | 64 | 15 | 2.647 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | jax-cuda-jit | 32 | 15 | 3.128 | 0.0001246 |
| householder_q_columns_complement_c32_512x256 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c64_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_512x256 | jax-cuda-jit | 64 | 15 | 2.681 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | jax-cuda-jit | 32 | 15 | 3.401 | 5.968e-08 |
| householder_q_columns_complement_c64_512x256 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| lu_square_f32_s256 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s256 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s256 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s256 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s256 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s256 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s256 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s256 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s256 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s256 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s256 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api
suite_file: benchmarks/gpu/linalg_public_api.yaml
timestamp: '2026-11-02T21:09:35Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 433121a7756fdb4f109b02f0422d3714a04514c6
  dirty: false
  features:
  - cuda
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-batched-gemm
harness:
  path: /workspaces/tenferro-benchmark
  commit: 8987033651c18a03e47e16cdd365df807b77b6be
  dirty: true
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
  hostname: 0fe295fad79b
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-7
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
    ./scripts/run_gpu_public_api.sh linalg_public_api
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 164
  selected_cases: 164
  sha256:
    .devcontainer/cuda/devcontainer.json: 0bf1465bbd1a4c9f896d5320838f2d5a488632a59af7f9b59cc050e95e0b8e09
    .devcontainer/cuda/Dockerfile: 578d9815e5145aa1651ac7e93833468c345ee8bf5e95bfa7717551580ae9feca
    src/bin/benchmark_gpu_public_api.rs: 1a5f3469fc9ffde64a79e05eb9853632366855ece3280ba83e2ba21fe3fb2cdc
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 291e63a517b6f7108527254a7f83751420aa1d4e73395aa60f0691312b2a3773
    src/bin/benchmark_gpu_public_api/plan.rs: 2d87b9d73bf25df8797133cf0a8dc8b1fa974fc3bee055ad92e42802c9ebf643
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: e11a863aee7a85822c474eb771b45ec402e21dbe92173d178c42119ab8ca307e
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: ddc112ea5d568a0399378f2be8b31c91b3144fe7d286e8094d27ca5ccda959bf
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 9cbcb473cba55b1605bab07fd3ff14247859fd7fe9e73d5378dd70108313256b
    data/results/nvidia-gpu/gpu/linalg_public_api/20261102_210935/fixtures/manifest.json: e8153a1102e04a6f09305645386390e52b03aa74d9b41adcaa496512183a88c9
    target/release/benchmark_gpu_public_api: c0a0be40dd246cd4cb7ca59cac19bacd1549a4f3a593aea91e887cf41f6e4aa6
  reference_numpy_version: 2.4.2
```
