# gpu/linalg_public_api_throughput: CUDA public API benchmark

Measured cells: **308**. Unsupported cells: **102**, including **28** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1416.6 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261105_033930/`.

Large-shape view of the CUDA public linalg API surface: the 82 cases whose largest matrix dimension is at least 256. `large` is a shape class, not a measured classification: no row is asserted to be compute- or bandwidth-bound, and a memory-bound row takes the same time here as at a small shape (`householder_r_default_f32` measured 34.698 µs at 512x256 and 35.223 µs at 64x32). These rows are never comparable with, or averaged together with, the per-call overhead rows of `gpu/linalg_public_api_latency` (tenferro-rs #2087). Rows here are lower than on pages published before this harness change because the harness used to open the CUDA engine once per case inside the timed region and now opens it once per arm, so the difference is removed harness cost and not a change in the kernels. The same problems, protocol and backends are split by shape class between the two suites. Issue 155.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_512x512 | 276.613 (0.529) | 325.179 (7.070) | 268.776 (5.183) | 241.345 (0.700) | 240.684 (0.697) |
| cholesky_f64_512x512 | 333.986 (0.951) | 368.800 (1.570) | 326.692 (1.298) | 303.084 (2.469) | 301.306 (1.255) |
| svdvals_f32_512x512 | 31118.960 (17.215) | unsupported | 34143.406 (9.585) | 31426.100 (79.272) | 31233.144 (420.754) |
| svdvals_f64_512x512 | 77432.006 (13.611) | unsupported | 84564.548 (12.140) | 78018.817 (34.751) | 77747.218 (36.657) |
| eigvalsh_f32_512x512 | 5192.744 (7.360) | 5260.875 (7.730) | 5209.554 (9.296) | 5951.720 (43.776) | 6391.326 (502.638) |
| eigvalsh_f64_512x512 | 5570.467 (9.431) | 5611.108 (11.495) | 5560.977 (18.051) | 6179.488 (96.908) | 6211.210 (41.712) |
| pinv_f32_512x512 | unsupported | unsupported | 34312.152 (29.466) | 31673.879 (76.813) | 31388.259 (101.113) |
| pinv_f64_512x512 | unsupported | unsupported | 84684.382 (27.812) | 78382.079 (57.957) | 77777.709 (45.902) |
| inv_f32_512x512 | unsupported | unsupported | 1517.079 (4.320) | 1551.420 (18.341) | 1574.111 (11.591) |
| inv_f64_512x512 | unsupported | unsupported | 1790.414 (4.130) | 1951.389 (90.934) | 1868.141 (15.315) |
| det_f32_512x512 | 1629.978 (21.264) | 1661.109 (39.146) | 1174.062 (6.321) | 1239.769 (28.986) | 1243.574 (34.066) |
| det_f64_512x512 | 1779.718 (24.131) | 1816.079 (21.728) | 1314.216 (6.806) | 1391.534 (20.521) | 1389.534 (32.051) |
| norm_f32_512x512 | 89.357 (2.916) | 112.010 (0.159) | 9.942 (0.975) | 98.201 (42.874) | 71.696 (18.840) |
| norm_f64_512x512 | 95.839 (0.492) | 113.825 (0.402) | 10.598 (1.078) | 97.339 (41.627) | 59.132 (50.422) |
| triangular_solve_f32_512x512 | 169.388 (0.690) | 218.080 (0.309) | 120.827 (0.108) | 185.140 (2.094) | 181.263 (1.925) |
| triangular_solve_f64_512x512 | 193.510 (0.307) | 241.195 (1.275) | 308.371 (0.092) | 218.085 (0.645) | 215.718 (0.353) |
| svd_full_square_f32_256x256 | 10707.439 (14.056) | 10816.072 (10.750) | 11422.574 (8.315) | 10981.278 (6.750) | 10856.134 (93.669) |
| svd_full_tall_f32_512x256 | 11142.823 (10.200) | 11237.107 (7.845) | 11891.710 (5.406) | 11381.283 (20.281) | 11340.851 (429.761) |
| svd_full_wide_f32_256x512 | 10879.714 (26.356) | 10965.797 (8.721) | 11097.052 (8.520) | 11193.125 (77.133) | 11009.459 (81.368) |
| slogdet_positive_f32_256x256 | 755.947 (1.120) | 805.596 (2.144) | 530.841 (1.375) | 528.691 (3.262) | 529.136 (3.036) |
| slogdet_negative_f32_256x256 | 755.744 (2.426) | 803.118 (2.338) | 529.919 (0.491) | 531.629 (3.684) | 540.752 (2.920) |
| svd_full_square_f64_256x256 | 25683.777 (8.955) | 25806.822 (13.266) | 25809.402 (9.061) | 26297.599 (320.621) | 25819.022 (66.173) |
| svd_full_tall_f64_512x256 | 19497.769 (14.385) | 19678.885 (35.571) | 19860.732 (27.256) | 19810.171 (36.842) | 19642.964 (113.404) |
| svd_full_wide_f64_256x512 | 17586.082 (32.372) | 17713.186 (15.921) | 18234.465 (21.745) | 17952.194 (50.087) | 17823.620 (69.343) |
| slogdet_positive_f64_256x256 | 800.261 (2.283) | 875.256 (30.997) | 567.857 (0.483) | 573.768 (1.595) | 574.610 (1.075) |
| slogdet_negative_f64_256x256 | 800.961 (4.745) | 842.025 (6.486) | 565.113 (1.085) | 576.958 (1.116) | 575.995 (7.973) |
| svd_full_square_c32_256x256 | 20262.846 (11.330) | 20373.730 (6.145) | 22109.271 (12.361) | 20594.118 (54.292) | 20527.546 (324.032) |
| svd_full_tall_c32_512x256 | 19537.850 (17.056) | 19663.295 (14.586) | 20031.928 (33.467) | 19852.431 (35.036) | 19755.748 (34.732) |
| svd_full_wide_c32_256x512 | 19699.916 (15.356) | 19794.549 (24.956) | 20229.395 (12.645) | 19976.606 (59.472) | 19846.772 (109.128) |
| slogdet_positive_c32_256x256 | 808.431 (1.764) | 859.403 (1.393) | 576.086 (0.864) | 575.565 (1.366) | 574.875 (12.240) |
| slogdet_negative_c32_256x256 | 809.968 (12.499) | 858.328 (1.458) | 573.025 (0.649) | 578.321 (0.890) | 569.300 (4.161) |
| svd_full_square_c64_256x256 | 60025.221 (23.336) | 60144.556 (11.086) | 63684.860 (20.640) | 60332.232 (96.518) | 60210.598 (37.751) |
| svd_full_tall_c64_512x256 | 40615.645 (11.085) | 40624.835 (22.795) | 41017.430 (22.506) | 40861.194 (40.276) | 40738.469 (59.537) |
| svd_full_wide_c64_256x512 | 40786.591 (26.302) | 40784.311 (30.776) | 41587.549 (21.391) | 40871.404 (32.991) | 40829.853 (28.866) |
| slogdet_positive_c64_256x256 | 862.578 (24.567) | 945.128 (2.370) | 631.602 (0.885) | 632.000 (4.685) | 655.001 (2.205) |
| slogdet_negative_c64_256x256 | 866.025 (2.536) | 940.921 (5.799) | 627.945 (0.993) | 633.670 (8.795) | 648.655 (2.370) |
| householder_qr_square_f32_256x256 | 926.018 (3.323) | 971.587 (7.915) | 907.322 (1.204) | 932.365 (11.503) | 966.870 (25.661) |
| householder_qr_tall_f32_512x256 | 1211.432 (4.185) | 1260.700 (7.176) | 1194.308 (6.693) | 1294.276 (51.759) | 1300.701 (29.119) |
| householder_qr_wide_f32_256x512 | 1126.655 (11.278) | 1152.021 (9.895) | 1102.659 (9.588) | 1231.864 (48.807) | 1237.904 (42.329) |
| householder_qr_square_f64_256x256 | 1033.306 (5.015) | 1077.993 (2.738) | 1015.516 (3.888) | 1057.988 (48.709) | 1093.424 (29.572) |
| householder_qr_tall_f64_512x256 | 1351.088 (5.960) | 1406.090 (9.728) | 1323.807 (11.048) | 1423.946 (30.321) | 1424.730 (43.619) |
| householder_qr_wide_f64_256x512 | 1245.269 (9.525) | 1294.521 (17.668) | 1238.589 (13.753) | 1353.958 (21.771) | 1356.183 (38.669) |
| householder_qr_square_c32_256x256 | 995.673 (4.450) | 1035.411 (8.820) | 973.314 (3.058) | 1014.686 (40.929) | 1069.327 (23.573) |
| householder_qr_tall_c32_512x256 | 1305.686 (4.365) | 1347.077 (15.268) | 1267.460 (4.800) | 1403.059 (34.448) | 1371.264 (25.748) |
| householder_qr_wide_c32_256x512 | 1220.658 (5.763) | 1266.985 (12.238) | 1201.918 (10.123) | 1308.851 (25.146) | 1298.321 (33.494) |
| slogdet_phase_c32_256x256 | 803.974 (16.794) | 856.808 (2.363) | 575.420 (0.872) | 554.979 (5.244) | 578.258 (1.056) |
| householder_qr_square_c64_256x256 | 1200.358 (2.493) | 1234.863 (5.220) | 1170.777 (5.263) | 1269.419 (50.079) | 1276.780 (33.069) |
| householder_qr_tall_c64_512x256 | 1653.728 (2.900) | 1658.149 (4.350) | 1642.478 (5.063) | 1690.400 (33.564) | 1721.021 (4.375) |
| householder_qr_wide_c64_256x512 | 1513.434 (12.683) | 1544.314 (13.461) | 1490.317 (6.925) | 1614.197 (16.520) | 1670.649 (34.157) |
| slogdet_phase_c64_256x256 | 894.657 (4.606) | 943.996 (29.557) | 628.630 (1.120) | 631.462 (5.074) | 643.780 (11.000) |
| householder_from_factors_default_f32_512x256 | 1433.010 (6.071) | 1491.628 (7.738) | unsupported | unsupported | unsupported |
| householder_append_default_f32_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f32_512x256 | 38.373 (2.702) | 43.888 (1.478) | 5.086 (0.364) | 39.530 (0.169) | 46.647 (6.759) |
| householder_q_columns_prefix_f32_512x256 | unsupported | unsupported | 61.228 (0.052) | 128.393 (0.918) | 105.450 (1.713) |
| householder_q_columns_complement_f32_512x256 | unsupported | unsupported | 166.436 (0.195) | unsupported | unsupported |
| householder_from_factors_default_f64_512x256 | 1585.891 (20.943) | 1617.987 (6.230) | unsupported | unsupported | unsupported |
| householder_append_default_f64_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f64_512x256 | 36.445 (1.840) | 45.573 (2.516) | 4.720 (0.478) | 39.646 (0.203) | 45.183 (4.120) |
| householder_q_columns_prefix_f64_512x256 | unsupported | unsupported | 65.354 (0.903) | 135.158 (0.581) | 114.267 (29.080) |
| householder_q_columns_complement_f64_512x256 | unsupported | unsupported | 165.201 (0.260) | unsupported | unsupported |
| householder_from_factors_default_c32_512x256 | 1529.364 (12.672) | 1589.872 (15.081) | unsupported | unsupported | unsupported |
| householder_append_default_c32_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c32_512x256 | 38.525 (1.436) | 44.100 (0.155) | 4.659 (0.007) | 39.714 (0.253) | 49.403 (0.353) |
| householder_q_columns_prefix_c32_512x256 | unsupported | unsupported | 91.220 (0.047) | 127.838 (0.839) | 109.189 (0.773) |
| householder_q_columns_complement_c32_512x256 | unsupported | unsupported | 251.123 (0.279) | unsupported | unsupported |
| householder_from_factors_default_c64_512x256 | 1865.476 (15.166) | 1836.600 (6.550) | unsupported | unsupported | unsupported |
| householder_append_default_c64_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c64_512x256 | 36.529 (1.533) | 48.545 (0.124) | 4.796 (0.011) | 39.998 (0.355) | 47.874 (4.500) |
| householder_q_columns_prefix_c64_512x256 | unsupported | unsupported | 63.581 (0.267) | 130.793 (0.956) | 111.381 (0.559) |
| householder_q_columns_complement_c64_512x256 | unsupported | unsupported | 252.314 (1.265) | unsupported | unsupported |
| lu_square_f32_s256 | 615.947 (4.070) | 657.646 (1.879) | unsupported | unsupported | unsupported |
| lu_square_f64_s256 | 646.953 (2.449) | 681.506 (3.233) | unsupported | unsupported | unsupported |
| lu_square_c32_s256 | 659.191 (7.699) | 682.074 (1.768) | unsupported | unsupported | unsupported |
| lu_square_c64_s256 | 710.925 (1.629) | 744.976 (0.911) | unsupported | unsupported | unsupported |
| lu_tall_f32_s256 | 760.522 (1.240) | 787.980 (2.913) | unsupported | unsupported | unsupported |
| lu_tall_f64_s256 | 839.932 (0.681) | 858.028 (3.289) | unsupported | unsupported | unsupported |
| lu_tall_c32_s256 | 821.854 (2.766) | 837.562 (1.365) | unsupported | unsupported | unsupported |
| lu_tall_c64_s256 | 961.406 (8.192) | 990.162 (4.476) | unsupported | unsupported | unsupported |
| lu_wide_f32_s256 | 655.626 (13.454) | 693.649 (2.895) | unsupported | unsupported | unsupported |
| lu_wide_f64_s256 | 730.798 (1.611) | 782.433 (16.851) | unsupported | unsupported | unsupported |
| lu_wide_c32_s256 | 711.270 (1.772) | 766.117 (3.983) | unsupported | unsupported | unsupported |
| lu_wide_c64_s256 | 834.100 (1.663) | 876.053 (11.759) | unsupported | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_512x512 | jax-cuda | 16 | 15 | 3.862 | 2097152 | 0.006 | 0.0003558 |
| cholesky_f32_512x512 | jax-cuda-jit | 16 | 15 | 3.851 | 2097152 | 0.006 | 0.0003558 |
| cholesky_f32_512x512 | pytorch-cuda | 8 | 15 | 2.150 | 2097152 | 0.006 | 0.0003558 |
| cholesky_f32_512x512 | tenferro-cuda-direct | 8 | 15 | 2.213 | 2097152 | 0.005 | 0.0003558 |
| cholesky_f32_512x512 | tenferro-cuda-eager | 8 | 15 | 2.601 | 2097152 | 0.005 | 0.0003558 |
| cholesky_f64_512x512 | jax-cuda | 8 | 15 | 2.425 | 4194304 | 0.010 | 4.418e-07 |
| cholesky_f64_512x512 | jax-cuda-jit | 8 | 15 | 2.410 | 4194304 | 0.010 | 4.418e-07 |
| cholesky_f64_512x512 | pytorch-cuda | 8 | 15 | 2.614 | 4194304 | 0.009 | 4.418e-07 |
| cholesky_f64_512x512 | tenferro-cuda-direct | 8 | 15 | 2.672 | 4194304 | 0.009 | 4.418e-07 |
| cholesky_f64_512x512 | tenferro-cuda-eager | 8 | 15 | 2.950 | 4194304 | 0.008 | 4.418e-07 |
| det_f32_512x512 | jax-cuda | 2 | 15 | 2.480 | 1048580 | 0.001 | 0.004748 |
| det_f32_512x512 | jax-cuda-jit | 2 | 15 | 2.487 | 1048580 | 0.001 | 0.004748 |
| det_f32_512x512 | pytorch-cuda | 2 | 15 | 2.348 | 1048580 | 0.001 | 0.007965 |
| det_f32_512x512 | tenferro-cuda-direct | 2 | 15 | 3.260 | 1048580 | 0.000 | 0.004748 |
| det_f32_512x512 | tenferro-cuda-eager | 2 | 15 | 3.322 | 1048580 | 0.000 | 0.004748 |
| det_f64_512x512 | jax-cuda | 2 | 15 | 2.783 | 2097160 | 0.001 | 1.426e-05 |
| det_f64_512x512 | jax-cuda-jit | 2 | 15 | 2.779 | 2097160 | 0.001 | 1.426e-05 |
| det_f64_512x512 | pytorch-cuda | 2 | 15 | 2.628 | 2097160 | 0.001 | 1.369e-05 |
| det_f64_512x512 | tenferro-cuda-direct | 2 | 15 | 3.559 | 2097160 | 0.001 | 1.426e-05 |
| det_f64_512x512 | tenferro-cuda-eager | 2 | 15 | 3.632 | 2097160 | 0.001 | 1.426e-05 |
| eigvalsh_f32_512x512 | jax-cuda | 1 | 15 | 5.952 | 1050624 | 0.000 | 0.008342 |
| eigvalsh_f32_512x512 | jax-cuda-jit | 1 | 15 | 6.391 | 1050624 | 0.000 | 0.008342 |
| eigvalsh_f32_512x512 | pytorch-cuda | 1 | 15 | 5.210 | 1050624 | 0.000 | 0.008342 |
| eigvalsh_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 5.193 | 1050624 | 0.000 | 0.008342 |
| eigvalsh_f32_512x512 | tenferro-cuda-eager | 1 | 15 | 5.261 | 1050624 | 0.000 | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda | 1 | 15 | 6.179 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | jax-cuda-jit | 1 | 15 | 6.211 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | pytorch-cuda | 1 | 15 | 5.561 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 5.570 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 5.611 | 2101248 | 0.000 | 1.55e-06 |
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
| householder_from_factors_default_c32_512x256 | tenferro-cuda-direct | 2 | 15 | 3.059 | 3672064 | 0.002 | 0.001103 |
| householder_from_factors_default_c32_512x256 | tenferro-cuda-eager | 2 | 15 | 3.180 | 3672064 | 0.002 | 0.001103 |
| householder_from_factors_default_c64_512x256 | jax-cuda | — | — | — | 7344128 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_512x256 | jax-cuda-jit | — | — | — | 7344128 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_512x256 | pytorch-cuda | — | — | — | 7344128 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_512x256 | tenferro-cuda-direct | 2 | 15 | 3.731 | 7344128 | 0.003 | 9.992e-07 |
| householder_from_factors_default_c64_512x256 | tenferro-cuda-eager | 2 | 15 | 3.673 | 7344128 | 0.003 | 9.992e-07 |
| householder_from_factors_default_f32_512x256 | jax-cuda | — | — | — | 1836032 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_512x256 | jax-cuda-jit | — | — | — | 1836032 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_512x256 | pytorch-cuda | — | — | — | 1836032 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.866 | 1836032 | 0.001 | 0.0008522 |
| householder_from_factors_default_f32_512x256 | tenferro-cuda-eager | 2 | 15 | 2.983 | 1836032 | 0.001 | 0.0008522 |
| householder_from_factors_default_f64_512x256 | jax-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | jax-cuda-jit | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | pytorch-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 3.172 | 3672064 | 0.002 | 8.882e-07 |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 3.236 | 3672064 | 0.002 | 8.882e-07 |
| householder_q_columns_complement_c32_512x256 | jax-cuda | — | — | — | 2097152 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c32_512x256 | jax-cuda-jit | — | — | — | 2097152 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c32_512x256 | pytorch-cuda | 8 | 15 | 2.009 | 2097152 | 0.006 | 0.0001141 |
| householder_q_columns_complement_c32_512x256 | tenferro-cuda-direct | — | — | — | 2097152 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_512x256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_512x256 | jax-cuda | — | — | — | 4194304 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c64_512x256 | jax-cuda-jit | — | — | — | 4194304 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c64_512x256 | pytorch-cuda | 8 | 15 | 2.019 | 4194304 | 0.012 | 7.282e-07 |
| householder_q_columns_complement_c64_512x256 | tenferro-cuda-direct | — | — | — | 4194304 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_512x256 | tenferro-cuda-eager | — | — | — | 4194304 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_512x256 | jax-cuda | — | — | — | 1048576 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f32_512x256 | jax-cuda-jit | — | — | — | 1048576 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f32_512x256 | pytorch-cuda | 16 | 15 | 2.663 | 1048576 | 0.004 | 8.761e-05 |
| householder_q_columns_complement_f32_512x256 | tenferro-cuda-direct | — | — | — | 1048576 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_512x256 | tenferro-cuda-eager | — | — | — | 1048576 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_512x256 | jax-cuda | — | — | — | 2097152 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f64_512x256 | jax-cuda-jit | — | — | — | 2097152 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f64_512x256 | pytorch-cuda | 16 | 15 | 2.643 | 2097152 | 0.009 | 7.26e-07 |
| householder_q_columns_complement_f64_512x256 | tenferro-cuda-direct | — | — | — | 2097152 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_512x256 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c32_512x256 | jax-cuda | 16 | 15 | 2.045 | 1081344 | 0.006 | 0.0001246 |
| householder_q_columns_prefix_c32_512x256 | jax-cuda-jit | 32 | 15 | 3.494 | 1081344 | 0.007 | 0.0001246 |
| householder_q_columns_prefix_c32_512x256 | pytorch-cuda | 32 | 15 | 2.919 | 1081344 | 0.008 | 0.0001246 |
| householder_q_columns_prefix_c32_512x256 | tenferro-cuda-direct | — | — | — | 1081344 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c32_512x256 | tenferro-cuda-eager | — | — | — | 1081344 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c64_512x256 | jax-cuda | 16 | 15 | 2.093 | 2162688 | 0.012 | 5.968e-08 |
| householder_q_columns_prefix_c64_512x256 | jax-cuda-jit | 32 | 15 | 3.564 | 2162688 | 0.014 | 5.968e-08 |
| householder_q_columns_prefix_c64_512x256 | pytorch-cuda | 32 | 15 | 2.035 | 2162688 | 0.024 | 5.968e-08 |
| householder_q_columns_prefix_c64_512x256 | tenferro-cuda-direct | — | — | — | 2162688 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_c64_512x256 | tenferro-cuda-eager | — | — | — | 2162688 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f32_512x256 | jax-cuda | 16 | 15 | 2.054 | 540672 | 0.003 | 0.0001829 |
| householder_q_columns_prefix_f32_512x256 | jax-cuda-jit | 32 | 15 | 3.374 | 540672 | 0.004 | 0.0001829 |
| householder_q_columns_prefix_f32_512x256 | pytorch-cuda | 64 | 15 | 3.919 | 540672 | 0.006 | 0.0001829 |
| householder_q_columns_prefix_f32_512x256 | tenferro-cuda-direct | — | — | — | 540672 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f32_512x256 | tenferro-cuda-eager | — | — | — | 540672 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | 16 | 15 | 2.163 | 1081344 | 0.006 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | jax-cuda-jit | 16 | 15 | 1.828 | 1081344 | 0.007 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | 32 | 15 | 2.091 | 1081344 | 0.012 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-direct | — | — | — | 1081344 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-eager | — | — | — | 1081344 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_qr_square_c32_256x256 | jax-cuda | 2 | 15 | 2.029 | 1050624 | 0.001 | 0.001258 |
| householder_qr_square_c32_256x256 | jax-cuda-jit | 2 | 15 | 2.139 | 1050624 | 0.001 | 0.001258 |
| householder_qr_square_c32_256x256 | pytorch-cuda | 4 | 15 | 3.893 | 1050624 | 0.001 | 0.001258 |
| householder_qr_square_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.983 | 1050624 | 0.001 | 0.001198 |
| householder_qr_square_c32_256x256 | tenferro-cuda-eager | 2 | 15 | 2.071 | 1050624 | 0.001 | 0.001198 |
| householder_qr_square_c64_256x256 | jax-cuda | 2 | 15 | 2.539 | 2101248 | 0.001 | 2.109e-06 |
| householder_qr_square_c64_256x256 | jax-cuda-jit | 2 | 15 | 2.554 | 2101248 | 0.001 | 2.109e-06 |
| householder_qr_square_c64_256x256 | pytorch-cuda | 2 | 15 | 2.342 | 2101248 | 0.001 | 2.109e-06 |
| householder_qr_square_c64_256x256 | tenferro-cuda-direct | 2 | 15 | 2.401 | 2101248 | 0.001 | 1.208e-06 |
| householder_qr_square_c64_256x256 | tenferro-cuda-eager | 2 | 15 | 2.470 | 2101248 | 0.001 | 1.208e-06 |
| householder_qr_square_f32_256x256 | jax-cuda | 4 | 15 | 3.729 | 525312 | 0.000 | 0.001043 |
| householder_qr_square_f32_256x256 | jax-cuda-jit | 2 | 15 | 1.934 | 525312 | 0.000 | 0.001043 |
| householder_qr_square_f32_256x256 | pytorch-cuda | 4 | 15 | 3.629 | 525312 | 0.000 | 0.001043 |
| householder_qr_square_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.704 | 525312 | 0.000 | 0.001075 |
| householder_qr_square_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.886 | 525312 | 0.000 | 0.001075 |
| householder_qr_square_f64_256x256 | jax-cuda | 2 | 15 | 2.116 | 1050624 | 0.001 | 1.887e-06 |
| householder_qr_square_f64_256x256 | jax-cuda-jit | 2 | 15 | 2.187 | 1050624 | 0.001 | 1.887e-06 |
| householder_qr_square_f64_256x256 | pytorch-cuda | 2 | 15 | 2.031 | 1050624 | 0.001 | 1.887e-06 |
| householder_qr_square_f64_256x256 | tenferro-cuda-direct | 2 | 15 | 2.067 | 1050624 | 0.001 | 1.098e-06 |
| householder_qr_square_f64_256x256 | tenferro-cuda-eager | 2 | 15 | 2.156 | 1050624 | 0.001 | 1.098e-06 |
| householder_qr_tall_c32_512x256 | jax-cuda | 2 | 15 | 2.806 | 2099200 | 0.001 | 0.001161 |
| householder_qr_tall_c32_512x256 | jax-cuda-jit | 2 | 15 | 2.743 | 2099200 | 0.001 | 0.001161 |
| householder_qr_tall_c32_512x256 | pytorch-cuda | 2 | 15 | 2.535 | 2099200 | 0.001 | 0.001161 |
| householder_qr_tall_c32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.611 | 2099200 | 0.001 | 0.001185 |
| householder_qr_tall_c32_512x256 | tenferro-cuda-eager | 2 | 15 | 2.694 | 2099200 | 0.001 | 0.001185 |
| householder_qr_tall_c64_512x256 | jax-cuda | 2 | 15 | 3.381 | 4198400 | 0.002 | 2.942e-06 |
| householder_qr_tall_c64_512x256 | jax-cuda-jit | 2 | 15 | 3.442 | 4198400 | 0.002 | 2.942e-06 |
| householder_qr_tall_c64_512x256 | pytorch-cuda | 2 | 15 | 3.285 | 4198400 | 0.002 | 2.942e-06 |
| householder_qr_tall_c64_512x256 | tenferro-cuda-direct | 2 | 15 | 3.307 | 4198400 | 0.002 | 1.11e-06 |
| householder_qr_tall_c64_512x256 | tenferro-cuda-eager | 2 | 15 | 3.316 | 4198400 | 0.002 | 1.11e-06 |
| householder_qr_tall_f32_512x256 | jax-cuda | 2 | 15 | 2.589 | 1049600 | 0.001 | 0.001475 |
| householder_qr_tall_f32_512x256 | jax-cuda-jit | 2 | 15 | 2.601 | 1049600 | 0.001 | 0.001475 |
| householder_qr_tall_f32_512x256 | pytorch-cuda | 2 | 15 | 2.389 | 1049600 | 0.001 | 0.001475 |
| householder_qr_tall_f32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.423 | 1049600 | 0.001 | 0.001435 |
| householder_qr_tall_f32_512x256 | tenferro-cuda-eager | 2 | 15 | 2.521 | 1049600 | 0.001 | 0.001435 |
| householder_qr_tall_f64_512x256 | jax-cuda | 2 | 15 | 2.848 | 2099200 | 0.001 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | jax-cuda-jit | 2 | 15 | 2.849 | 2099200 | 0.001 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | pytorch-cuda | 2 | 15 | 2.648 | 2099200 | 0.001 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 2.702 | 2099200 | 0.001 | 9.992e-07 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 2.812 | 2099200 | 0.001 | 9.992e-07 |
| householder_qr_wide_c32_256x512 | jax-cuda | 2 | 15 | 2.618 | 2099200 | 0.001 | 0.001226 |
| householder_qr_wide_c32_256x512 | jax-cuda-jit | 2 | 15 | 2.597 | 2099200 | 0.001 | 0.001226 |
| householder_qr_wide_c32_256x512 | pytorch-cuda | 2 | 15 | 2.404 | 2099200 | 0.001 | 0.001226 |
| householder_qr_wide_c32_256x512 | tenferro-cuda-direct | 2 | 15 | 2.441 | 2099200 | 0.001 | 0.001329 |
| householder_qr_wide_c32_256x512 | tenferro-cuda-eager | 2 | 15 | 2.534 | 2099200 | 0.001 | 0.001329 |
| householder_qr_wide_c64_256x512 | jax-cuda | 2 | 15 | 3.228 | 4198400 | 0.002 | 2.276e-06 |
| householder_qr_wide_c64_256x512 | jax-cuda-jit | 1 | 15 | 1.671 | 4198400 | 0.002 | 2.276e-06 |
| householder_qr_wide_c64_256x512 | pytorch-cuda | 2 | 15 | 2.981 | 4198400 | 0.002 | 2.276e-06 |
| householder_qr_wide_c64_256x512 | tenferro-cuda-direct | 2 | 15 | 3.027 | 4198400 | 0.002 | 1.206e-06 |
| householder_qr_wide_c64_256x512 | tenferro-cuda-eager | 2 | 15 | 3.089 | 4198400 | 0.002 | 1.206e-06 |
| householder_qr_wide_f32_256x512 | jax-cuda | 2 | 15 | 2.464 | 1049600 | 0.001 | 0.0009736 |
| householder_qr_wide_f32_256x512 | jax-cuda-jit | 2 | 15 | 2.476 | 1049600 | 0.001 | 0.0009736 |
| householder_qr_wide_f32_256x512 | pytorch-cuda | 2 | 15 | 2.205 | 1049600 | 0.001 | 0.0009736 |
| householder_qr_wide_f32_256x512 | tenferro-cuda-direct | 2 | 15 | 2.253 | 1049600 | 0.001 | 0.0009125 |
| householder_qr_wide_f32_256x512 | tenferro-cuda-eager | 2 | 15 | 2.304 | 1049600 | 0.001 | 0.0009125 |
| householder_qr_wide_f64_256x512 | jax-cuda | 2 | 15 | 2.708 | 2099200 | 0.001 | 1.759e-06 |
| householder_qr_wide_f64_256x512 | jax-cuda-jit | 2 | 15 | 2.712 | 2099200 | 0.001 | 1.759e-06 |
| householder_qr_wide_f64_256x512 | pytorch-cuda | 2 | 15 | 2.477 | 2099200 | 0.001 | 1.759e-06 |
| householder_qr_wide_f64_256x512 | tenferro-cuda-direct | 2 | 15 | 2.491 | 2099200 | 0.001 | 1.649e-06 |
| householder_qr_wide_f64_256x512 | tenferro-cuda-eager | 2 | 15 | 2.589 | 2099200 | 0.001 | 1.649e-06 |
| householder_r_default_c32_512x256 | jax-cuda | 64 | 15 | 2.542 | 1572864 | 0.028 | 0.001069 |
| householder_r_default_c32_512x256 | jax-cuda-jit | 64 | 15 | 3.162 | 1572864 | 0.022 | 0.001069 |
| householder_r_default_c32_512x256 | pytorch-cuda | 512 | 15 | 2.386 | 1572864 | 0.238 | 0.001069 |
| householder_r_default_c32_512x256 | tenferro-cuda-direct | 64 | 15 | 2.466 | 1572864 | 0.029 | 0.001069 |
| householder_r_default_c32_512x256 | tenferro-cuda-eager | 64 | 15 | 2.822 | 1572864 | 0.025 | 0.001069 |
| householder_r_default_c64_512x256 | jax-cuda | 64 | 15 | 2.560 | 3145728 | 0.056 | 9.985e-07 |
| householder_r_default_c64_512x256 | jax-cuda-jit | 64 | 15 | 3.064 | 3145728 | 0.046 | 9.985e-07 |
| householder_r_default_c64_512x256 | pytorch-cuda | 256 | 15 | 1.228 | 3145728 | 0.463 | 9.985e-07 |
| householder_r_default_c64_512x256 | tenferro-cuda-direct | 64 | 15 | 2.338 | 3145728 | 0.061 | 9.985e-07 |
| householder_r_default_c64_512x256 | tenferro-cuda-eager | 64 | 15 | 3.107 | 3145728 | 0.046 | 9.985e-07 |
| householder_r_default_f32_512x256 | jax-cuda | 64 | 15 | 2.530 | 786432 | 0.014 | 0.001415 |
| householder_r_default_f32_512x256 | jax-cuda-jit | 64 | 15 | 2.985 | 786432 | 0.012 | 0.001415 |
| householder_r_default_f32_512x256 | pytorch-cuda | 256 | 15 | 1.302 | 786432 | 0.109 | 0.001415 |
| householder_r_default_f32_512x256 | tenferro-cuda-direct | 64 | 15 | 2.456 | 786432 | 0.014 | 0.001415 |
| householder_r_default_f32_512x256 | tenferro-cuda-eager | 64 | 15 | 2.809 | 786432 | 0.013 | 0.001415 |
| householder_r_default_f64_512x256 | jax-cuda | 64 | 15 | 2.537 | 1572864 | 0.028 | 9.976e-07 |
| householder_r_default_f64_512x256 | jax-cuda-jit | 64 | 15 | 2.892 | 1572864 | 0.025 | 9.976e-07 |
| householder_r_default_f64_512x256 | pytorch-cuda | 256 | 15 | 1.208 | 1572864 | 0.235 | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | 64 | 15 | 2.332 | 1572864 | 0.030 | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | 64 | 15 | 2.917 | 1572864 | 0.024 | 9.976e-07 |
| inv_f32_512x512 | jax-cuda | 2 | 15 | 3.103 | 2097152 | 0.001 | 0.002107 |
| inv_f32_512x512 | jax-cuda-jit | 2 | 15 | 3.148 | 2097152 | 0.001 | 0.002107 |
| inv_f32_512x512 | pytorch-cuda | 2 | 15 | 3.034 | 2097152 | 0.001 | 0.002107 |
| inv_f32_512x512 | tenferro-cuda-direct | — | — | — | 2097152 | unavailable | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_512x512 | tenferro-cuda-eager | — | — | — | 2097152 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_512x512 | jax-cuda | 1 | 15 | 1.951 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | jax-cuda-jit | 2 | 15 | 3.736 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | pytorch-cuda | 2 | 15 | 3.581 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | tenferro-cuda-direct | — | — | — | 4194304 | unavailable | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_512x512 | tenferro-cuda-eager | — | — | — | 4194304 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| lu_square_c32_s256 | jax-cuda | — | — | — | 2097160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s256 | jax-cuda-jit | — | — | — | 2097160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s256 | pytorch-cuda | — | — | — | 2097160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s256 | tenferro-cuda-direct | 4 | 15 | 2.637 | 2097160 | 0.002 | 0.001541 |
| lu_square_c32_s256 | tenferro-cuda-eager | 4 | 15 | 2.728 | 2097160 | 0.002 | 0.001541 |
| lu_square_c64_s256 | jax-cuda | — | — | — | 4194320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s256 | jax-cuda-jit | — | — | — | 4194320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s256 | pytorch-cuda | — | — | — | 4194320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s256 | tenferro-cuda-direct | 4 | 15 | 2.844 | 4194320 | 0.004 | 1.431e-06 |
| lu_square_c64_s256 | tenferro-cuda-eager | 4 | 15 | 2.980 | 4194320 | 0.004 | 1.431e-06 |
| lu_square_f32_s256 | jax-cuda | — | — | — | 1048580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s256 | jax-cuda-jit | — | — | — | 1048580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s256 | pytorch-cuda | — | — | — | 1048580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s256 | tenferro-cuda-direct | 4 | 15 | 2.464 | 1048580 | 0.001 | 0.001428 |
| lu_square_f32_s256 | tenferro-cuda-eager | 4 | 15 | 2.631 | 1048580 | 0.001 | 0.001428 |
| lu_square_f64_s256 | jax-cuda | — | — | — | 2097160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s256 | jax-cuda-jit | — | — | — | 2097160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s256 | pytorch-cuda | — | — | — | 2097160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s256 | tenferro-cuda-direct | 4 | 15 | 2.588 | 2097160 | 0.002 | 1.102e-06 |
| lu_square_f64_s256 | tenferro-cuda-eager | 4 | 15 | 2.726 | 2097160 | 0.002 | 1.102e-06 |
| lu_tall_c32_s256 | jax-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s256 | jax-cuda-jit | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s256 | pytorch-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s256 | tenferro-cuda-direct | 4 | 15 | 3.287 | 4718600 | 0.004 | 0.002239 |
| lu_tall_c32_s256 | tenferro-cuda-eager | 4 | 15 | 3.350 | 4718600 | 0.004 | 0.002239 |
| lu_tall_c64_s256 | jax-cuda | — | — | — | 9437200 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s256 | jax-cuda-jit | — | — | — | 9437200 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s256 | pytorch-cuda | — | — | — | 9437200 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s256 | tenferro-cuda-direct | 4 | 15 | 3.846 | 9437200 | 0.007 | 1.542e-06 |
| lu_tall_c64_s256 | tenferro-cuda-eager | 4 | 15 | 3.961 | 9437200 | 0.007 | 1.542e-06 |
| lu_tall_f32_s256 | jax-cuda | — | — | — | 2359300 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s256 | jax-cuda-jit | — | — | — | 2359300 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s256 | pytorch-cuda | — | — | — | 2359300 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s256 | tenferro-cuda-direct | 4 | 15 | 3.042 | 2359300 | 0.002 | 0.001428 |
| lu_tall_f32_s256 | tenferro-cuda-eager | 4 | 15 | 3.152 | 2359300 | 0.002 | 0.001428 |
| lu_tall_f64_s256 | jax-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | jax-cuda-jit | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | pytorch-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | tenferro-cuda-direct | 4 | 15 | 3.360 | 4718600 | 0.004 | 1.102e-06 |
| lu_tall_f64_s256 | tenferro-cuda-eager | 4 | 15 | 3.432 | 4718600 | 0.004 | 1.102e-06 |
| lu_wide_c32_s256 | jax-cuda | — | — | — | 3145736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s256 | jax-cuda-jit | — | — | — | 3145736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s256 | pytorch-cuda | — | — | — | 3145736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s256 | tenferro-cuda-direct | 4 | 15 | 2.845 | 3145736 | 0.003 | 0.002021 |
| lu_wide_c32_s256 | tenferro-cuda-eager | 4 | 15 | 3.064 | 3145736 | 0.003 | 0.002021 |
| lu_wide_c64_s256 | jax-cuda | — | — | — | 6291472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s256 | jax-cuda-jit | — | — | — | 6291472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s256 | pytorch-cuda | — | — | — | 6291472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s256 | tenferro-cuda-direct | 4 | 15 | 3.336 | 6291472 | 0.005 | 1.433e-06 |
| lu_wide_c64_s256 | tenferro-cuda-eager | 4 | 15 | 3.504 | 6291472 | 0.005 | 1.433e-06 |
| lu_wide_f32_s256 | jax-cuda | — | — | — | 1572868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s256 | jax-cuda-jit | — | — | — | 1572868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s256 | pytorch-cuda | — | — | — | 1572868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s256 | tenferro-cuda-direct | 4 | 15 | 2.623 | 1572868 | 0.002 | 0.001297 |
| lu_wide_f32_s256 | tenferro-cuda-eager | 4 | 15 | 2.775 | 1572868 | 0.002 | 0.001297 |
| lu_wide_f64_s256 | jax-cuda | — | — | — | 3145736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s256 | jax-cuda-jit | — | — | — | 3145736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s256 | pytorch-cuda | — | — | — | 3145736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s256 | tenferro-cuda-direct | 4 | 15 | 2.923 | 3145736 | 0.003 | 8.806e-07 |
| lu_wide_f64_s256 | tenferro-cuda-eager | 4 | 15 | 3.130 | 3145736 | 0.003 | 8.806e-07 |
| norm_f32_512x512 | jax-cuda | 32 | 15 | 3.142 | 1048580 | 0.008 | 0 |
| norm_f32_512x512 | jax-cuda-jit | 32 | 15 | 2.294 | 1048580 | 0.010 | 0 |
| norm_f32_512x512 | pytorch-cuda | 256 | 15 | 2.545 | 1048580 | 0.074 | 0 |
| norm_f32_512x512 | tenferro-cuda-direct | 32 | 15 | 2.859 | 1048580 | 0.008 | 0 |
| norm_f32_512x512 | tenferro-cuda-eager | 32 | 15 | 3.584 | 1048580 | 0.007 | 0 |
| norm_f64_512x512 | jax-cuda | 32 | 15 | 3.115 | 2097160 | 0.015 | 1.909e-07 |
| norm_f64_512x512 | jax-cuda-jit | 32 | 15 | 1.892 | 2097160 | 0.025 | 1.909e-07 |
| norm_f64_512x512 | pytorch-cuda | 256 | 15 | 2.713 | 2097160 | 0.140 | 1.909e-07 |
| norm_f64_512x512 | tenferro-cuda-direct | 32 | 15 | 3.067 | 2097160 | 0.015 | 1.909e-07 |
| norm_f64_512x512 | tenferro-cuda-eager | 32 | 15 | 3.642 | 2097160 | 0.013 | 1.909e-07 |
| pinv_f32_512x512 | jax-cuda | 1 | 15 | 31.674 | 2097160 | 0.000 | 0.009697 |
| pinv_f32_512x512 | jax-cuda-jit | 1 | 15 | 31.388 | 2097160 | 0.000 | 0.009697 |
| pinv_f32_512x512 | pytorch-cuda | 1 | 15 | 34.312 | 2097160 | 0.000 | 0.01174 |
| pinv_f32_512x512 | tenferro-cuda-direct | — | — | — | 2097160 | unavailable | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f32_512x512 | tenferro-cuda-eager | — | — | — | 2097160 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_512x512 | jax-cuda | 1 | 15 | 78.382 | 4194320 | 0.000 | 8.975e-06 |
| pinv_f64_512x512 | jax-cuda-jit | 1 | 15 | 77.778 | 4194320 | 0.000 | 8.975e-06 |
| pinv_f64_512x512 | pytorch-cuda | 1 | 15 | 84.684 | 4194320 | 0.000 | 8.747e-06 |
| pinv_f64_512x512 | tenferro-cuda-direct | — | — | — | 4194320 | unavailable | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_512x512 | tenferro-cuda-eager | — | — | — | 4194320 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| slogdet_negative_c32_256x256 | jax-cuda | 4 | 15 | 2.313 | 524300 | 0.001 | 0.01211 |
| slogdet_negative_c32_256x256 | jax-cuda-jit | 4 | 15 | 2.277 | 524300 | 0.001 | 0.01211 |
| slogdet_negative_c32_256x256 | pytorch-cuda | 4 | 15 | 2.292 | 524300 | 0.001 | 0.01211 |
| slogdet_negative_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.240 | 524300 | 0.000 | 0.01211 |
| slogdet_negative_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.433 | 524300 | 0.000 | 0.01211 |
| slogdet_negative_c64_256x256 | jax-cuda | 4 | 15 | 2.535 | 1048600 | 0.001 | 1.025e-06 |
| slogdet_negative_c64_256x256 | jax-cuda-jit | 4 | 15 | 2.595 | 1048600 | 0.001 | 1.025e-06 |
| slogdet_negative_c64_256x256 | pytorch-cuda | 4 | 15 | 2.512 | 1048600 | 0.001 | 1.23e-06 |
| slogdet_negative_c64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.464 | 1048600 | 0.001 | 1.025e-06 |
| slogdet_negative_c64_256x256 | tenferro-cuda-eager | 4 | 15 | 3.764 | 1048600 | 0.001 | 1.025e-06 |
| slogdet_negative_f32_256x256 | jax-cuda | 4 | 15 | 2.127 | 262152 | 0.000 | 0.0005003 |
| slogdet_negative_f32_256x256 | jax-cuda-jit | 4 | 15 | 2.163 | 262152 | 0.000 | 0.0005003 |
| slogdet_negative_f32_256x256 | pytorch-cuda | 4 | 15 | 2.120 | 262152 | 0.000 | 0.0005003 |
| slogdet_negative_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.023 | 262152 | 0.000 | 0.0005003 |
| slogdet_negative_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.212 | 262152 | 0.000 | 0.0005003 |
| slogdet_negative_f64_256x256 | jax-cuda | 4 | 15 | 2.308 | 524304 | 0.001 | 1.864e-06 |
| slogdet_negative_f64_256x256 | jax-cuda-jit | 4 | 15 | 2.304 | 524304 | 0.001 | 1.864e-06 |
| slogdet_negative_f64_256x256 | pytorch-cuda | 4 | 15 | 2.260 | 524304 | 0.001 | 1.708e-06 |
| slogdet_negative_f64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.204 | 524304 | 0.000 | 1.864e-06 |
| slogdet_negative_f64_256x256 | tenferro-cuda-eager | 4 | 15 | 3.368 | 524304 | 0.000 | 1.864e-06 |
| slogdet_phase_c32_256x256 | jax-cuda | 4 | 15 | 2.220 | 524300 | 0.001 | 0.01233 |
| slogdet_phase_c32_256x256 | jax-cuda-jit | 4 | 15 | 2.313 | 524300 | 0.001 | 0.01233 |
| slogdet_phase_c32_256x256 | pytorch-cuda | 4 | 15 | 2.302 | 524300 | 0.001 | 0.01233 |
| slogdet_phase_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.216 | 524300 | 0.000 | 0.01211 |
| slogdet_phase_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.427 | 524300 | 0.000 | 0.01211 |
| slogdet_phase_c64_256x256 | jax-cuda | 4 | 15 | 2.526 | 1048600 | 0.001 | 1.025e-06 |
| slogdet_phase_c64_256x256 | jax-cuda-jit | 4 | 15 | 2.575 | 1048600 | 0.001 | 1.025e-06 |
| slogdet_phase_c64_256x256 | pytorch-cuda | 4 | 15 | 2.515 | 1048600 | 0.001 | 1.025e-06 |
| slogdet_phase_c64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.579 | 1048600 | 0.001 | 1.025e-06 |
| slogdet_phase_c64_256x256 | tenferro-cuda-eager | 4 | 15 | 3.776 | 1048600 | 0.001 | 1.025e-06 |
| slogdet_positive_c32_256x256 | jax-cuda | 4 | 15 | 2.302 | 524300 | 0.001 | 0.01211 |
| slogdet_positive_c32_256x256 | jax-cuda-jit | 4 | 15 | 2.300 | 524300 | 0.001 | 0.01211 |
| slogdet_positive_c32_256x256 | pytorch-cuda | 4 | 15 | 2.304 | 524300 | 0.001 | 0.01211 |
| slogdet_positive_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.234 | 524300 | 0.000 | 0.01211 |
| slogdet_positive_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.438 | 524300 | 0.000 | 0.01211 |
| slogdet_positive_c64_256x256 | jax-cuda | 4 | 15 | 2.528 | 1048600 | 0.001 | 1.025e-06 |
| slogdet_positive_c64_256x256 | jax-cuda-jit | 4 | 15 | 2.620 | 1048600 | 0.001 | 1.025e-06 |
| slogdet_positive_c64_256x256 | pytorch-cuda | 4 | 15 | 2.526 | 1048600 | 0.001 | 1.23e-06 |
| slogdet_positive_c64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.450 | 1048600 | 0.001 | 1.025e-06 |
| slogdet_positive_c64_256x256 | tenferro-cuda-eager | 4 | 15 | 3.781 | 1048600 | 0.001 | 1.025e-06 |
| slogdet_positive_f32_256x256 | jax-cuda | 4 | 15 | 2.115 | 262152 | 0.000 | 0.0005003 |
| slogdet_positive_f32_256x256 | jax-cuda-jit | 4 | 15 | 2.117 | 262152 | 0.000 | 0.0005003 |
| slogdet_positive_f32_256x256 | pytorch-cuda | 4 | 15 | 2.123 | 262152 | 0.000 | 0.0005003 |
| slogdet_positive_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.024 | 262152 | 0.000 | 0.0005003 |
| slogdet_positive_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.222 | 262152 | 0.000 | 0.0005003 |
| slogdet_positive_f64_256x256 | jax-cuda | 4 | 15 | 2.295 | 524304 | 0.001 | 1.864e-06 |
| slogdet_positive_f64_256x256 | jax-cuda-jit | 4 | 15 | 2.298 | 524304 | 0.001 | 1.864e-06 |
| slogdet_positive_f64_256x256 | pytorch-cuda | 4 | 15 | 2.271 | 524304 | 0.001 | 1.708e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.201 | 524304 | 0.000 | 1.864e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | 4 | 15 | 3.501 | 524304 | 0.000 | 1.864e-06 |
| svd_full_square_c32_256x256 | jax-cuda | 1 | 15 | 20.594 | 1573888 | 0.000 | 0.133 |
| svd_full_square_c32_256x256 | jax-cuda-jit | 1 | 15 | 20.528 | 1573888 | 0.000 | 0.133 |
| svd_full_square_c32_256x256 | pytorch-cuda | 1 | 15 | 22.109 | 1573888 | 0.000 | 0.1305 |
| svd_full_square_c32_256x256 | tenferro-cuda-direct | 1 | 15 | 20.263 | 1573888 | 0.000 | 0.133 |
| svd_full_square_c32_256x256 | tenferro-cuda-eager | 1 | 15 | 20.374 | 1573888 | 0.000 | 0.133 |
| svd_full_square_c64_256x256 | jax-cuda | 1 | 15 | 60.332 | 3147776 | 0.000 | 0.000204 |
| svd_full_square_c64_256x256 | jax-cuda-jit | 1 | 15 | 60.211 | 3147776 | 0.000 | 0.000204 |
| svd_full_square_c64_256x256 | pytorch-cuda | 1 | 15 | 63.685 | 3147776 | 0.000 | 0.0002064 |
| svd_full_square_c64_256x256 | tenferro-cuda-direct | 1 | 15 | 60.025 | 3147776 | 0.000 | 0.000204 |
| svd_full_square_c64_256x256 | tenferro-cuda-eager | 1 | 15 | 60.145 | 3147776 | 0.000 | 0.000204 |
| svd_full_square_f32_256x256 | jax-cuda | 1 | 15 | 10.981 | 787456 | 0.000 | 0.1152 |
| svd_full_square_f32_256x256 | jax-cuda-jit | 1 | 15 | 10.856 | 787456 | 0.000 | 0.1152 |
| svd_full_square_f32_256x256 | pytorch-cuda | 1 | 15 | 11.423 | 787456 | 0.000 | 0.1135 |
| svd_full_square_f32_256x256 | tenferro-cuda-direct | 1 | 15 | 10.707 | 787456 | 0.000 | 0.1152 |
| svd_full_square_f32_256x256 | tenferro-cuda-eager | 1 | 15 | 10.816 | 787456 | 0.000 | 0.1152 |
| svd_full_square_f64_256x256 | jax-cuda | 1 | 15 | 26.298 | 1574912 | 0.000 | 8.171e-05 |
| svd_full_square_f64_256x256 | jax-cuda-jit | 1 | 15 | 25.819 | 1574912 | 0.000 | 8.171e-05 |
| svd_full_square_f64_256x256 | pytorch-cuda | 1 | 15 | 25.809 | 1574912 | 0.000 | 8.105e-05 |
| svd_full_square_f64_256x256 | tenferro-cuda-direct | 1 | 15 | 25.684 | 1574912 | 0.000 | 8.171e-05 |
| svd_full_square_f64_256x256 | tenferro-cuda-eager | 1 | 15 | 25.807 | 1574912 | 0.000 | 8.171e-05 |
| svd_full_tall_c32_512x256 | jax-cuda | 1 | 15 | 19.852 | 3671040 | 0.000 | 0.1126 |
| svd_full_tall_c32_512x256 | jax-cuda-jit | 1 | 15 | 19.756 | 3671040 | 0.000 | 0.1126 |
| svd_full_tall_c32_512x256 | pytorch-cuda | 1 | 15 | 20.032 | 3671040 | 0.000 | 0.1089 |
| svd_full_tall_c32_512x256 | tenferro-cuda-direct | 1 | 15 | 19.538 | 3671040 | 0.000 | 0.1126 |
| svd_full_tall_c32_512x256 | tenferro-cuda-eager | 1 | 15 | 19.663 | 3671040 | 0.000 | 0.1126 |
| svd_full_tall_c64_512x256 | jax-cuda | 1 | 15 | 40.861 | 7342080 | 0.000 | 0.0002021 |
| svd_full_tall_c64_512x256 | jax-cuda-jit | 1 | 15 | 40.738 | 7342080 | 0.000 | 0.0002021 |
| svd_full_tall_c64_512x256 | pytorch-cuda | 1 | 15 | 41.017 | 7342080 | 0.000 | 0.0001992 |
| svd_full_tall_c64_512x256 | tenferro-cuda-direct | 1 | 15 | 40.616 | 7342080 | 0.000 | 0.0002022 |
| svd_full_tall_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 40.625 | 7342080 | 0.000 | 0.0002022 |
| svd_full_tall_f32_512x256 | jax-cuda | 1 | 15 | 11.381 | 1836032 | 0.000 | 0.1022 |
| svd_full_tall_f32_512x256 | jax-cuda-jit | 1 | 15 | 11.341 | 1836032 | 0.000 | 0.1022 |
| svd_full_tall_f32_512x256 | pytorch-cuda | 1 | 15 | 11.892 | 1836032 | 0.000 | 0.1033 |
| svd_full_tall_f32_512x256 | tenferro-cuda-direct | 1 | 15 | 11.143 | 1836032 | 0.000 | 0.1022 |
| svd_full_tall_f32_512x256 | tenferro-cuda-eager | 1 | 15 | 11.237 | 1836032 | 0.000 | 0.1022 |
| svd_full_tall_f64_512x256 | jax-cuda | 1 | 15 | 19.810 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | jax-cuda-jit | 1 | 15 | 19.643 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 19.861 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 19.498 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 19.679 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_wide_c32_256x512 | jax-cuda | 1 | 15 | 19.977 | 3671040 | 0.000 | 0.1103 |
| svd_full_wide_c32_256x512 | jax-cuda-jit | 1 | 15 | 19.847 | 3671040 | 0.000 | 0.1103 |
| svd_full_wide_c32_256x512 | pytorch-cuda | 1 | 15 | 20.229 | 3671040 | 0.000 | 0.108 |
| svd_full_wide_c32_256x512 | tenferro-cuda-direct | 1 | 15 | 19.700 | 3671040 | 0.000 | 0.1103 |
| svd_full_wide_c32_256x512 | tenferro-cuda-eager | 1 | 15 | 19.795 | 3671040 | 0.000 | 0.1103 |
| svd_full_wide_c64_256x512 | jax-cuda | 1 | 15 | 40.871 | 7342080 | 0.000 | 0.0001946 |
| svd_full_wide_c64_256x512 | jax-cuda-jit | 1 | 15 | 40.830 | 7342080 | 0.000 | 0.0001946 |
| svd_full_wide_c64_256x512 | pytorch-cuda | 1 | 15 | 41.588 | 7342080 | 0.000 | 0.0001941 |
| svd_full_wide_c64_256x512 | tenferro-cuda-direct | 1 | 15 | 40.787 | 7342080 | 0.000 | 0.0001943 |
| svd_full_wide_c64_256x512 | tenferro-cuda-eager | 1 | 15 | 40.784 | 7342080 | 0.000 | 0.0001943 |
| svd_full_wide_f32_256x512 | jax-cuda | 1 | 15 | 11.193 | 1836032 | 0.000 | 0.1057 |
| svd_full_wide_f32_256x512 | jax-cuda-jit | 1 | 15 | 11.009 | 1836032 | 0.000 | 0.1057 |
| svd_full_wide_f32_256x512 | pytorch-cuda | 1 | 15 | 11.097 | 1836032 | 0.000 | 0.106 |
| svd_full_wide_f32_256x512 | tenferro-cuda-direct | 1 | 15 | 10.880 | 1836032 | 0.000 | 0.1057 |
| svd_full_wide_f32_256x512 | tenferro-cuda-eager | 1 | 15 | 10.966 | 1836032 | 0.000 | 0.1057 |
| svd_full_wide_f64_256x512 | jax-cuda | 1 | 15 | 17.952 | 3672064 | 0.000 | 7.938e-05 |
| svd_full_wide_f64_256x512 | jax-cuda-jit | 1 | 15 | 17.824 | 3672064 | 0.000 | 7.938e-05 |
| svd_full_wide_f64_256x512 | pytorch-cuda | 1 | 15 | 18.234 | 3672064 | 0.000 | 7.938e-05 |
| svd_full_wide_f64_256x512 | tenferro-cuda-direct | 1 | 15 | 17.586 | 3672064 | 0.000 | 7.938e-05 |
| svd_full_wide_f64_256x512 | tenferro-cuda-eager | 1 | 15 | 17.713 | 3672064 | 0.000 | 7.938e-05 |
| svdvals_f32_512x512 | jax-cuda | 1 | 15 | 31.426 | 1050624 | 0.000 | 0.1547 |
| svdvals_f32_512x512 | jax-cuda-jit | 1 | 15 | 31.233 | 1050624 | 0.000 | 0.1547 |
| svdvals_f32_512x512 | pytorch-cuda | 1 | 15 | 34.143 | 1050624 | 0.000 | 0.1733 |
| svdvals_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 31.119 | 1050624 | 0.000 | 0.1547 |
| svdvals_f32_512x512 | tenferro-cuda-eager | — | — | — | 1050624 | unavailable | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_512x512 | jax-cuda | 1 | 15 | 78.019 | 2101248 | 0.000 | 0.0002063 |
| svdvals_f64_512x512 | jax-cuda-jit | 1 | 15 | 77.747 | 2101248 | 0.000 | 0.0002063 |
| svdvals_f64_512x512 | pytorch-cuda | 1 | 15 | 84.565 | 2101248 | 0.000 | 0.0002066 |
| svdvals_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 77.432 | 2101248 | 0.000 | 0.0002063 |
| svdvals_f64_512x512 | tenferro-cuda-eager | — | — | — | 2101248 | unavailable | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| triangular_solve_f32_512x512 | jax-cuda | 16 | 15 | 2.962 | 1081344 | 0.004 | 0.0007877 |
| triangular_solve_f32_512x512 | jax-cuda-jit | 16 | 15 | 2.900 | 1081344 | 0.004 | 0.0007877 |
| triangular_solve_f32_512x512 | pytorch-cuda | 32 | 15 | 3.866 | 1081344 | 0.006 | 0.0006008 |
| triangular_solve_f32_512x512 | tenferro-cuda-direct | 16 | 15 | 2.710 | 1081344 | 0.005 | 0.0007877 |
| triangular_solve_f32_512x512 | tenferro-cuda-eager | 16 | 15 | 3.489 | 1081344 | 0.004 | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda | 16 | 15 | 3.489 | 2162688 | 0.007 | 1.157e-06 |
| triangular_solve_f64_512x512 | jax-cuda-jit | 16 | 15 | 3.451 | 2162688 | 0.007 | 1.157e-06 |
| triangular_solve_f64_512x512 | pytorch-cuda | 8 | 15 | 2.467 | 2162688 | 0.005 | 6.241e-07 |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | 16 | 15 | 3.096 | 2162688 | 0.008 | 1.157e-06 |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | 16 | 15 | 3.859 | 2162688 | 0.006 | 1.157e-06 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api_throughput
suite_file: benchmarks/gpu/linalg_public_api_throughput.yaml
timestamp: '2026-11-05T03:40:34Z'
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
    regime: cold
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
  - 0.092893
  - 0.093224
  - 0.097923
  - 0.094584
  - 0.092673
  - 0.096254
  - 0.095863
  - 0.091653
  - 0.095584
  - 0.093643
  - 0.094744
  - 0.095683
  - 0.094844
  - 0.094753
  - 0.094443
  median_ms: 0.094744
  bytes_per_second: 1416635649750.9077
  torch_version: 2.12.0+cu130
  cuda_version: '13.0'
  timestamp: '2026-11-05T03:40:41.380883+00:00'
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
    data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261105_033930/fixtures/manifest.json: 9bc6d3164be9e92417459d741f7875f4416224b26ecb3d4bd0381358bee92930
    target/release/benchmark_gpu_public_api: d048341025f978c4b0a4153b8b9e99f23da3db25601102f1c0171ce33bdffa79
  reference_numpy_version: 2.4.2
```
