# gpu/linalg_public_api: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api/20261013_160222/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | 203.207 (14.963) | 277.198 (7.322) | 90.948 (5.713) | 149.386 (1.222) | 141.512 (0.938) |
| cholesky_f64_32x32 | 301.489 (3.279) | 413.308 (7.647) | 164.955 (4.660) | 116.980 (5.974) | 194.338 (6.917) |
| svdvals_f32_32x32 | 731.863 (31.073) | unsupported | 432.364 (13.915) | 851.042 (38.584) | 874.391 (19.406) |
| svdvals_f64_32x32 | 5314.712 (135.655) | unsupported | 5084.771 (10.579) | 5732.628 (63.678) | 5860.261 (96.005) |
| eigvalsh_f32_32x32 | 235.665 (6.236) | 371.793 (11.606) | 223.541 (2.669) | 658.143 (19.377) | 660.672 (26.874) |
| eigvalsh_f64_32x32 | 615.124 (4.024) | 740.044 (3.560) | 615.255 (8.773) | 2994.101 (32.250) | 2948.942 (78.102) |
| pinv_f32_32x32 | unsupported | unsupported | 534.678 (20.713) | 1239.010 (114.635) | 714.940 (14.427) |
| pinv_f64_32x32 | unsupported | unsupported | 4932.240 (19.962) | 5830.939 (32.798) | 5320.784 (22.131) |
| inv_f32_32x32 | unsupported | unsupported | 287.484 (3.765) | 304.519 (9.265) | 197.806 (1.598) |
| inv_f64_32x32 | unsupported | unsupported | 287.447 (2.129) | 200.071 (2.390) | 259.033 (13.089) |
| det_f32_32x32 | 635.168 (12.394) | 834.209 (38.881) | 164.438 (5.213) | 255.175 (1.781) | 206.980 (32.793) |
| det_f64_32x32 | 642.611 (11.799) | 843.413 (11.307) | 165.076 (4.937) | 171.588 (3.874) | 206.048 (49.312) |
| norm_f32_32x32 | 78.382 (0.897) | 186.846 (3.919) | 14.611 (0.829) | 134.536 (2.698) | 138.651 (4.335) |
| norm_f64_32x32 | 78.614 (2.231) | 197.858 (4.182) | 14.423 (0.189) | 162.646 (12.900) | 164.829 (15.797) |
| triangular_solve_f32_32x32 | 87.975 (5.398) | 267.882 (54.897) | 52.257 (0.743) | 213.407 (46.967) | 194.979 (24.326) |
| triangular_solve_f64_32x32 | 80.860 (4.227) | 260.542 (11.753) | 51.803 (0.343) | 135.562 (2.324) | 207.690 (48.078) |
| cholesky_f32_512x512 | 350.272 (10.425) | 479.189 (18.745) | 279.659 (2.117) | 266.542 (4.647) | 272.681 (3.593) |
| cholesky_f64_512x512 | 2226.039 (13.222) | 2515.643 (10.348) | 2198.718 (2.149) | 2771.568 (16.553) | 2755.042 (15.791) |
| svdvals_f32_512x512 | 25666.431 (33.206) | unsupported | 28126.184 (70.198) | 26030.868 (14.262) | 25872.765 (11.641) |
| svdvals_f64_512x512 | 540005.619 (59.976) | unsupported | 594555.044 (82.201) | 540414.485 (46.796) | 537917.960 (58.344) |
| eigvalsh_f32_512x512 | 4246.726 (11.353) | 4395.217 (14.608) | 4371.663 (13.493) | 5047.555 (67.265) | 4965.551 (94.270) |
| eigvalsh_f64_512x512 | 14645.592 (26.965) | 14845.171 (100.379) | 14689.954 (49.299) | 19592.432 (77.803) | 19455.952 (79.351) |
| pinv_f32_512x512 | unsupported | unsupported | 31675.013 (31.511) | 29315.286 (151.423) | 28797.082 (20.352) |
| pinv_f64_512x512 | unsupported | unsupported | 680023.102 (75.516) | 615891.915 (119.284) | 615487.668 (64.971) |
| inv_f32_512x512 | unsupported | unsupported | 1945.069 (4.494) | 2288.475 (69.282) | 2320.777 (23.586) |
| inv_f64_512x512 | unsupported | unsupported | 5988.213 (9.079) | 6418.192 (40.451) | 6429.427 (30.274) |
| det_f32_512x512 | 1774.614 (5.047) | 1959.307 (46.787) | 1443.287 (1.665) | 1643.314 (14.731) | 1624.581 (21.053) |
| det_f64_512x512 | 2857.377 (21.692) | 2985.805 (8.697) | 2499.150 (9.298) | 2909.158 (18.593) | 2926.940 (31.381) |
| norm_f32_512x512 | 36.370 (0.581) | 81.152 (0.681) | 20.123 (0.743) | 155.340 (25.240) | 160.050 (29.340) |
| norm_f64_512x512 | 37.802 (2.644) | 85.618 (1.765) | 19.767 (0.107) | 165.711 (5.229) | 167.625 (12.498) |
| triangular_solve_f32_512x512 | 71.370 (0.126) | 169.266 (5.189) | 103.393 (0.056) | 224.423 (39.068) | 228.080 (22.295) |
| triangular_solve_f64_512x512 | 141.156 (0.503) | 200.419 (0.614) | 842.514 (0.397) | 261.624 (4.463) | 222.913 (23.043) |
| svd_full_square_f32_32x32 | 605.843 (6.894) | 739.802 (16.876) | 341.977 (2.301) | 852.484 (52.579) | 703.993 (22.365) |
| svd_full_tall_f32_64x32 | 848.685 (7.931) | 949.483 (11.823) | 840.411 (16.043) | 1384.278 (17.194) | 1215.082 (40.565) |
| svd_full_wide_f32_32x64 | 901.779 (5.362) | 1044.413 (34.004) | 896.521 (7.953) | 1492.503 (25.284) | 1163.234 (34.047) |
| slogdet_positive_f32_32x32 | 302.112 (4.290) | 445.020 (5.325) | 209.700 (4.391) | 279.959 (35.045) | 201.890 (42.452) |
| slogdet_negative_f32_32x32 | 303.468 (2.583) | 473.296 (32.000) | 207.270 (2.696) | 268.726 (15.186) | 176.581 (23.555) |
| svd_full_square_f64_32x32 | 5077.229 (21.864) | 5239.539 (17.254) | 4260.230 (8.647) | 5641.724 (89.285) | 5751.169 (64.609) |
| svd_full_tall_f64_64x32 | 5862.139 (22.425) | 6021.905 (24.910) | 5932.611 (6.333) | 6140.388 (75.159) | 6592.139 (128.025) |
| svd_full_wide_f64_32x64 | 5996.205 (18.108) | 6200.886 (21.488) | 6282.156 (16.145) | 6197.990 (32.354) | 6299.968 (139.986) |
| slogdet_positive_f64_32x32 | 319.723 (3.895) | 511.649 (31.735) | 207.486 (7.866) | 174.985 (16.702) | 203.066 (40.120) |
| slogdet_negative_f64_32x32 | 322.508 (14.212) | 509.533 (14.722) | 207.782 (1.722) | 144.588 (13.524) | 277.051 (23.234) |
| svd_full_square_c32_32x32 | 807.045 (11.470) | 933.523 (13.403) | 473.500 (0.604) | 867.170 (20.958) | 1056.948 (17.158) |
| svd_full_tall_c32_64x32 | 1053.232 (13.239) | 1179.417 (27.798) | 1047.386 (13.665) | 1306.743 (75.993) | 1481.364 (69.649) |
| svd_full_wide_c32_32x64 | 1062.476 (32.592) | 1215.771 (26.360) | 1055.424 (8.233) | 1487.887 (107.793) | 1568.892 (57.122) |
| slogdet_positive_c32_32x32 | 559.919 (6.734) | 825.758 (16.804) | 223.995 (6.893) | 300.547 (13.305) | 276.494 (24.418) |
| slogdet_negative_c32_32x32 | 507.912 (9.365) | 748.295 (20.494) | 238.797 (10.110) | 253.277 (25.203) | 297.678 (20.088) |
| svd_full_square_c64_32x32 | 14354.869 (10.223) | 14486.850 (22.163) | 11362.935 (2.640) | 14425.388 (21.785) | 14418.910 (11.948) |
| svd_full_tall_c64_64x32 | 15226.056 (18.655) | 15494.108 (21.150) | 15403.867 (15.469) | 15287.473 (28.141) | 15293.432 (16.596) |
| svd_full_wide_c64_32x64 | 15884.977 (15.963) | 16220.013 (43.092) | 16144.973 (5.342) | 15971.714 (31.213) | 16035.441 (25.408) |
| slogdet_positive_c64_32x32 | 336.660 (12.666) | 421.337 (10.539) | 223.203 (0.790) | 221.278 (34.249) | 188.867 (7.019) |
| slogdet_negative_c64_32x32 | 343.359 (3.836) | 597.878 (28.718) | 222.108 (0.920) | 180.452 (30.440) | 265.762 (10.555) |
| svd_full_square_f32_256x256 | 8574.780 (31.429) | 8654.144 (11.740) | 9051.508 (21.894) | 8750.922 (47.078) | 8762.495 (83.322) |
| svd_full_tall_f32_512x256 | 9024.013 (19.534) | 9132.883 (34.190) | 9535.730 (17.236) | 9200.001 (17.390) | 9178.302 (27.383) |
| svd_full_wide_f32_256x512 | 8802.546 (35.298) | 8889.513 (30.679) | 8942.611 (16.673) | 8965.199 (19.167) | 8958.678 (24.738) |
| slogdet_positive_f32_256x256 | 615.488 (4.551) | 746.014 (12.945) | 368.495 (0.339) | 393.562 (12.307) | 388.700 (14.768) |
| slogdet_negative_f32_256x256 | 621.954 (12.026) | 747.488 (13.598) | 372.247 (0.936) | 428.406 (26.981) | 381.220 (15.775) |
| svd_full_square_f64_256x256 | 160887.777 (48.231) | 162066.611 (42.462) | 165133.596 (38.181) | 160420.455 (41.940) | 160342.944 (48.500) |
| svd_full_tall_f64_512x256 | 128078.872 (153.264) | 129102.567 (114.620) | 130718.727 (50.832) | 127824.667 (158.349) | 127902.814 (138.772) |
| svd_full_wide_f64_256x512 | 118554.558 (40.481) | 119435.044 (79.973) | 121256.231 (36.574) | 118428.923 (85.415) | 118439.801 (112.779) |
| slogdet_positive_f64_256x256 | 1081.887 (19.300) | 1076.048 (13.380) | 719.316 (1.940) | 804.493 (2.249) | 798.422 (16.159) |
| slogdet_negative_f64_256x256 | 1075.972 (20.663) | 1074.103 (31.751) | 721.247 (0.573) | 813.001 (4.175) | 809.200 (6.157) |
| svd_full_square_c32_256x256 | 13130.745 (25.590) | 13232.237 (32.019) | 14259.603 (18.484) | 13251.970 (17.375) | 13147.706 (21.955) |
| svd_full_tall_c32_512x256 | 13190.384 (28.457) | 13195.531 (13.821) | 13300.448 (12.356) | 13266.987 (19.368) | 13151.575 (13.566) |
| svd_full_wide_c32_256x512 | 13255.577 (21.869) | 13275.352 (20.135) | 13499.002 (16.171) | 13220.257 (45.797) | 13206.469 (14.191) |
| slogdet_positive_c32_256x256 | 726.246 (8.974) | 784.456 (19.568) | 407.846 (0.470) | 430.047 (15.935) | 432.164 (5.311) |
| slogdet_negative_c32_256x256 | 717.054 (6.004) | 779.478 (37.924) | 407.916 (0.305) | 434.340 (13.767) | 485.197 (18.748) |
| svd_full_square_c64_256x256 | 517022.750 (30.356) | 517104.423 (66.071) | 543462.026 (85.728) | 511557.216 (72.459) | 511526.764 (46.458) |
| svd_full_tall_c64_512x256 | 369732.834 (181.982) | 367207.442 (87.207) | 371563.419 (93.590) | 366084.185 (122.617) | 366045.267 (192.761) |
| svd_full_wide_c64_256x512 | 370938.657 (183.882) | 368381.669 (79.880) | 374795.794 (60.532) | 367174.810 (101.652) | 367224.339 (104.398) |
| slogdet_positive_c64_256x256 | 1706.783 (27.424) | 1807.173 (191.024) | 1395.918 (0.867) | 1585.863 (6.500) | 1593.658 (12.168) |
| slogdet_negative_c64_256x256 | 1696.054 (8.030) | 1754.852 (25.921) | 1396.949 (6.347) | 1595.729 (6.855) | 1577.430 (7.454) |
| householder_qr_square_f32_32x32 | 175.991 (1.035) | 271.512 (2.536) | 61.652 (2.402) | 196.940 (60.820) | 195.802 (48.803) |
| householder_qr_tall_f32_64x32 | 215.095 (10.986) | 321.991 (9.315) | 110.337 (4.250) | 323.635 (21.311) | 355.184 (51.828) |
| householder_qr_wide_f32_32x64 | 196.671 (1.877) | 287.517 (6.331) | 110.986 (0.811) | 309.100 (5.236) | 190.064 (2.561) |
| householder_qr_square_f64_32x32 | 466.834 (6.644) | 578.647 (9.413) | 286.407 (0.081) | 313.891 (8.726) | 318.402 (7.014) |
| householder_qr_tall_f64_64x32 | 538.529 (12.279) | 592.206 (16.008) | 355.491 (0.459) | 393.684 (12.590) | 430.187 (7.265) |
| householder_qr_wide_f64_32x64 | 522.264 (8.302) | 560.207 (14.045) | 360.031 (0.196) | 389.091 (20.519) | 428.187 (8.334) |
| householder_qr_square_c32_32x32 | 227.206 (2.921) | 358.920 (4.690) | 67.722 (0.054) | 172.084 (40.180) | 260.757 (19.557) |
| householder_qr_tall_c32_64x32 | 276.591 (18.882) | 386.679 (24.324) | 114.458 (1.314) | 280.743 (4.775) | 391.456 (18.052) |
| householder_qr_wide_c32_32x64 | 219.608 (5.766) | 313.328 (17.236) | 108.898 (2.370) | 319.869 (18.332) | 310.802 (60.905) |
| slogdet_phase_c32_32x32 | 407.221 (12.554) | 563.374 (8.793) | 222.777 (2.932) | 224.442 (33.152) | 287.216 (31.283) |
| householder_qr_square_c64_32x32 | 656.568 (8.527) | 757.162 (11.374) | 482.045 (1.889) | 630.078 (14.949) | 623.397 (15.321) |
| householder_qr_tall_c64_64x32 | 739.771 (12.295) | 841.127 (12.835) | 576.972 (0.358) | 734.852 (19.038) | 726.980 (4.464) |
| householder_qr_wide_c64_32x64 | 737.534 (40.617) | 851.277 (4.480) | 621.132 (0.222) | 782.306 (8.920) | 777.792 (16.511) |
| slogdet_phase_c64_32x32 | 425.560 (3.315) | 526.476 (14.037) | 222.636 (7.764) | 348.892 (50.448) | 309.376 (10.067) |
| householder_qr_square_f32_256x256 | 855.218 (4.127) | 916.463 (7.061) | 759.323 (4.478) | 1032.236 (24.890) | 914.238 (11.128) |
| householder_qr_tall_f32_512x256 | 1117.422 (11.379) | 1192.179 (14.282) | 1047.950 (2.785) | 1337.813 (9.159) | 1338.646 (19.193) |
| householder_qr_wide_f32_256x512 | 1300.382 (8.614) | 1370.006 (10.276) | 1244.818 (4.215) | 1540.976 (31.346) | 1533.976 (22.292) |
| householder_qr_square_f64_256x256 | 3136.869 (144.505) | 3059.374 (184.245) | 2928.433 (8.186) | 3565.406 (34.538) | 3578.805 (40.782) |
| householder_qr_tall_f64_512x256 | 3549.978 (17.317) | 3622.769 (18.449) | 3322.298 (15.539) | 3879.211 (53.293) | 3986.467 (32.473) |
| householder_qr_wide_f64_256x512 | 5138.597 (46.995) | 5211.027 (12.727) | 4802.989 (32.933) | 5592.913 (45.213) | 5552.316 (30.945) |
| householder_qr_square_c32_256x256 | 905.492 (6.534) | 986.180 (12.194) | 792.694 (2.457) | 1025.969 (16.291) | 1012.268 (15.319) |
| householder_qr_tall_c32_512x256 | 1255.212 (6.113) | 1330.615 (11.645) | 1136.411 (5.951) | 1296.114 (34.717) | 1303.549 (47.733) |
| householder_qr_wide_c32_256x512 | 1439.172 (10.433) | 1489.424 (9.437) | 1304.601 (5.257) | 1538.337 (20.454) | 1578.272 (28.907) |
| slogdet_phase_c32_256x256 | 680.731 (6.928) | 964.513 (18.313) | 407.830 (0.483) | 443.420 (6.580) | 450.583 (12.698) |
| householder_qr_square_c64_256x256 | 5429.803 (29.271) | 5443.730 (166.935) | 5068.680 (9.770) | 5721.248 (78.760) | 5825.564 (40.176) |
| householder_qr_tall_c64_512x256 | 5768.163 (24.631) | 5759.793 (61.914) | 5692.416 (15.469) | 6012.062 (80.914) | 6071.515 (64.716) |
| householder_qr_wide_c64_256x512 | 9796.516 (28.610) | 9800.407 (39.379) | 9694.381 (13.438) | 10065.661 (59.219) | 10087.382 (81.270) |
| slogdet_phase_c64_256x256 | 1627.418 (6.913) | 1770.614 (13.394) | 1395.459 (0.998) | 1587.006 (7.907) | 1566.491 (7.374) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.251 | 0.0002352 |
| cholesky_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.412 | 2.188e-07 |
| svdvals_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.927 | 0.003338 |
| svdvals_f64_32x32 | tenferro-cuda-direct | 1 | 15 | 5.315 | 4.546e-06 |
| eigvalsh_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.771 | 0.001073 |
| eigvalsh_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.460 | 5.433e-07 |
| pinv_f32_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| det_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.541 | 0.002641 |
| det_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.570 | 1.135e-06 |
| norm_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.508 | 0 |
| norm_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.516 | 0 |
| triangular_solve_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.815 | 0.0001546 |
| triangular_solve_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.588 | 1.609e-07 |
| cholesky_f32_512x512 | tenferro-cuda-direct | 8 | 15 | 2.802 | 0.0003554 |
| cholesky_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.226 | 4.418e-07 |
| svdvals_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 25.666 | 0.1547 |
| svdvals_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 540.006 | 0.0002063 |
| eigvalsh_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 4.247 | 0.008342 |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 14.646 | 1.55e-06 |
| pinv_f32_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| det_f32_512x512 | tenferro-cuda-direct | 2 | 15 | 3.549 | 0.008577 |
| det_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.857 | 1.426e-05 |
| norm_f32_512x512 | tenferro-cuda-direct | 64 | 15 | 2.328 | 0 |
| norm_f64_512x512 | tenferro-cuda-direct | 64 | 15 | 2.419 | 1.909e-07 |
| triangular_solve_f32_512x512 | tenferro-cuda-direct | 32 | 15 | 2.284 | 0.0007877 |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | 16 | 15 | 2.258 | 1.157e-06 |
| svd_full_square_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.423 | 0.008731 |
| svd_full_tall_f32_64x32 | tenferro-cuda-direct | 4 | 15 | 3.395 | 0.004865 |
| svd_full_wide_f32_32x64 | tenferro-cuda-direct | 4 | 15 | 3.607 | 0.005495 |
| slogdet_positive_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.417 | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.428 | 0.000915 |
| svd_full_square_f64_32x32 | tenferro-cuda-direct | 1 | 15 | 5.077 | 1.007e-05 |
| svd_full_tall_f64_64x32 | tenferro-cuda-direct | 1 | 15 | 5.862 | 8.327e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-direct | 1 | 15 | 5.996 | 1.197e-05 |
| slogdet_positive_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.558 | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.580 | 4.261e-07 |
| svd_full_square_c32_32x32 | tenferro-cuda-direct | 4 | 15 | 3.228 | 0.009079 |
| svd_full_tall_c32_64x32 | tenferro-cuda-direct | 2 | 15 | 2.106 | 0.009498 |
| svd_full_wide_c32_32x64 | tenferro-cuda-direct | 2 | 15 | 2.125 | 0.008495 |
| slogdet_positive_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 4.479 | 0.0007148 |
| slogdet_negative_c32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.032 | 0.0007148 |
| svd_full_square_c64_32x32 | tenferro-cuda-direct | 1 | 15 | 14.355 | 9.215e-06 |
| svd_full_tall_c64_64x32 | tenferro-cuda-direct | 1 | 15 | 15.226 | 1.221e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-direct | 1 | 15 | 15.885 | 1.31e-05 |
| slogdet_positive_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.693 | 3.804e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.747 | 3.804e-07 |
| svd_full_square_f32_256x256 | tenferro-cuda-direct | 1 | 15 | 8.575 | 0.1152 |
| svd_full_tall_f32_512x256 | tenferro-cuda-direct | 1 | 15 | 9.024 | 0.1022 |
| svd_full_wide_f32_256x512 | tenferro-cuda-direct | 1 | 15 | 8.803 | 0.1056 |
| slogdet_positive_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.462 | 0.0005003 |
| slogdet_negative_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.488 | 0.0005003 |
| svd_full_square_f64_256x256 | tenferro-cuda-direct | 1 | 15 | 160.888 | 8.171e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 128.079 | 7.76e-05 |
| svd_full_wide_f64_256x512 | tenferro-cuda-direct | 1 | 15 | 118.555 | 7.938e-05 |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | 2 | 15 | 2.164 | 1.864e-06 |
| slogdet_negative_f64_256x256 | tenferro-cuda-direct | 2 | 15 | 2.152 | 1.864e-06 |
| svd_full_square_c32_256x256 | tenferro-cuda-direct | 1 | 15 | 13.131 | 0.133 |
| svd_full_tall_c32_512x256 | tenferro-cuda-direct | 1 | 15 | 13.190 | 0.1126 |
| svd_full_wide_c32_256x512 | tenferro-cuda-direct | 1 | 15 | 13.256 | 0.1103 |
| slogdet_positive_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.905 | 0.01211 |
| slogdet_negative_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.868 | 0.01211 |
| svd_full_square_c64_256x256 | tenferro-cuda-direct | 1 | 15 | 517.023 | 0.000204 |
| svd_full_tall_c64_512x256 | tenferro-cuda-direct | 1 | 15 | 369.733 | 0.0002024 |
| svd_full_wide_c64_256x512 | tenferro-cuda-direct | 1 | 15 | 370.939 | 0.0001941 |
| slogdet_positive_c64_256x256 | tenferro-cuda-direct | 2 | 15 | 3.414 | 1.025e-06 |
| slogdet_negative_c64_256x256 | tenferro-cuda-direct | 2 | 15 | 3.392 | 1.025e-06 |
| householder_qr_square_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 2.816 | 0.0004213 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-direct | 16 | 15 | 3.442 | 0.0003161 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-direct | 16 | 15 | 3.147 | 0.0004732 |
| householder_qr_square_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 1.867 | 3.886e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-direct | 4 | 15 | 2.154 | 4.441e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-direct | 4 | 15 | 2.089 | 4.441e-07 |
| householder_qr_square_c32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.635 | 0.0005631 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-direct | 8 | 15 | 2.213 | 0.0003724 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-direct | 16 | 15 | 3.514 | 0.0006002 |
| slogdet_phase_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 3.258 | 0.0004085 |
| householder_qr_square_c64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.626 | 4.996e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-direct | 4 | 15 | 2.959 | 5.551e-07 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-direct | 4 | 15 | 2.950 | 3.886e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 3.404 | 4.755e-07 |
| householder_qr_square_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.421 | 0.0009546 |
| householder_qr_tall_f32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.235 | 0.001435 |
| householder_qr_wide_f32_256x512 | tenferro-cuda-direct | 2 | 15 | 2.601 | 0.001037 |
| householder_qr_square_f64_256x256 | tenferro-cuda-direct | 1 | 15 | 3.137 | 1.098e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 3.550 | 9.992e-07 |
| householder_qr_wide_f64_256x512 | tenferro-cuda-direct | 1 | 15 | 5.139 | 1.649e-06 |
| householder_qr_square_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.622 | 0.001198 |
| householder_qr_tall_c32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.510 | 0.001185 |
| householder_qr_wide_c32_256x512 | tenferro-cuda-direct | 2 | 15 | 2.878 | 0.001329 |
| slogdet_phase_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.723 | 0.01211 |
| householder_qr_square_c64_256x256 | tenferro-cuda-direct | 1 | 15 | 5.430 | 1.208e-06 |
| householder_qr_tall_c64_512x256 | tenferro-cuda-direct | 1 | 15 | 5.768 | 1.11e-06 |
| householder_qr_wide_c64_256x512 | tenferro-cuda-direct | 1 | 15 | 9.797 | 1.206e-06 |
| slogdet_phase_c64_256x256 | tenferro-cuda-direct | 2 | 15 | 3.255 | 1.025e-06 |
| cholesky_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 4.435 | 0.0002352 |
| cholesky_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.306 | 2.188e-07 |
| svdvals_f32_32x32 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_32x32 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| eigvalsh_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.974 | 0.001073 |
| eigvalsh_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 2.960 | 5.433e-07 |
| pinv_f32_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f32_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| det_f32_32x32 | tenferro-cuda-eager | 4 | 15 | 3.337 | 0.002641 |
| det_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 3.374 | 1.135e-06 |
| norm_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 2.990 | 0 |
| norm_f64_32x32 | tenferro-cuda-eager | 16 | 15 | 3.166 | 0 |
| triangular_solve_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.143 | 0.0001546 |
| triangular_solve_f64_32x32 | tenferro-cuda-eager | 16 | 15 | 4.169 | 1.609e-07 |
| cholesky_f32_512x512 | tenferro-cuda-eager | 8 | 15 | 3.834 | 0.0003554 |
| cholesky_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 2.516 | 4.418e-07 |
| svdvals_f32_512x512 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_512x512 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| eigvalsh_f32_512x512 | tenferro-cuda-eager | 1 | 15 | 4.395 | 0.008342 |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 14.845 | 1.55e-06 |
| pinv_f32_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f32_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| det_f32_512x512 | tenferro-cuda-eager | 2 | 15 | 3.919 | 0.008577 |
| det_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 2.986 | 1.426e-05 |
| norm_f32_512x512 | tenferro-cuda-eager | 32 | 15 | 2.597 | 0 |
| norm_f64_512x512 | tenferro-cuda-eager | 32 | 15 | 2.740 | 1.909e-07 |
| triangular_solve_f32_512x512 | tenferro-cuda-eager | 16 | 15 | 2.708 | 0.0007877 |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | 16 | 15 | 3.207 | 1.157e-06 |
| svd_full_square_f32_32x32 | tenferro-cuda-eager | 4 | 15 | 2.959 | 0.008731 |
| svd_full_tall_f32_64x32 | tenferro-cuda-eager | 4 | 15 | 3.798 | 0.004865 |
| svd_full_wide_f32_32x64 | tenferro-cuda-eager | 4 | 15 | 4.178 | 0.005495 |
| slogdet_positive_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.560 | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.786 | 0.000915 |
| svd_full_square_f64_32x32 | tenferro-cuda-eager | 1 | 15 | 5.240 | 1.007e-05 |
| svd_full_tall_f64_64x32 | tenferro-cuda-eager | 1 | 15 | 6.022 | 8.327e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-eager | 1 | 15 | 6.201 | 1.197e-05 |
| slogdet_positive_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 4.093 | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 2.038 | 4.261e-07 |
| svd_full_square_c32_32x32 | tenferro-cuda-eager | 4 | 15 | 3.734 | 0.009079 |
| svd_full_tall_c32_64x32 | tenferro-cuda-eager | 2 | 15 | 2.359 | 0.009498 |
| svd_full_wide_c32_32x64 | tenferro-cuda-eager | 2 | 15 | 2.432 | 0.008495 |
| slogdet_positive_c32_32x32 | tenferro-cuda-eager | 8 | 15 | 6.606 | 0.0007148 |
| slogdet_negative_c32_32x32 | tenferro-cuda-eager | 4 | 15 | 2.993 | 0.0007148 |
| svd_full_square_c64_32x32 | tenferro-cuda-eager | 1 | 15 | 14.487 | 9.215e-06 |
| svd_full_tall_c64_64x32 | tenferro-cuda-eager | 1 | 15 | 15.494 | 1.221e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-eager | 1 | 15 | 16.220 | 1.31e-05 |
| slogdet_positive_c64_32x32 | tenferro-cuda-eager | 4 | 15 | 1.685 | 3.804e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-eager | 8 | 15 | 4.783 | 3.804e-07 |
| svd_full_square_f32_256x256 | tenferro-cuda-eager | 1 | 15 | 8.654 | 0.1152 |
| svd_full_tall_f32_512x256 | tenferro-cuda-eager | 1 | 15 | 9.133 | 0.1022 |
| svd_full_wide_f32_256x512 | tenferro-cuda-eager | 1 | 15 | 8.890 | 0.1056 |
| slogdet_positive_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 2.984 | 0.0005003 |
| slogdet_negative_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 2.990 | 0.0005003 |
| svd_full_square_f64_256x256 | tenferro-cuda-eager | 1 | 15 | 162.067 | 8.171e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 129.103 | 7.76e-05 |
| svd_full_wide_f64_256x512 | tenferro-cuda-eager | 1 | 15 | 119.435 | 7.938e-05 |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | 2 | 15 | 2.152 | 1.864e-06 |
| slogdet_negative_f64_256x256 | tenferro-cuda-eager | 2 | 15 | 2.148 | 1.864e-06 |
| svd_full_square_c32_256x256 | tenferro-cuda-eager | 1 | 15 | 13.232 | 0.133 |
| svd_full_tall_c32_512x256 | tenferro-cuda-eager | 1 | 15 | 13.196 | 0.1126 |
| svd_full_wide_c32_256x512 | tenferro-cuda-eager | 1 | 15 | 13.275 | 0.1103 |
| slogdet_positive_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.138 | 0.01211 |
| slogdet_negative_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.118 | 0.01211 |
| svd_full_square_c64_256x256 | tenferro-cuda-eager | 1 | 15 | 517.104 | 0.000204 |
| svd_full_tall_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 367.207 | 0.0002024 |
| svd_full_wide_c64_256x512 | tenferro-cuda-eager | 1 | 15 | 368.382 | 0.0001941 |
| slogdet_positive_c64_256x256 | tenferro-cuda-eager | 2 | 15 | 3.614 | 1.025e-06 |
| slogdet_negative_c64_256x256 | tenferro-cuda-eager | 2 | 15 | 3.510 | 1.025e-06 |
| householder_qr_square_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.172 | 0.0004213 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-eager | 8 | 15 | 2.576 | 0.0003161 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-eager | 8 | 15 | 2.300 | 0.0004732 |
| householder_qr_square_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 2.315 | 3.886e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-eager | 4 | 15 | 2.369 | 4.441e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-eager | 4 | 15 | 2.241 | 4.441e-07 |
| householder_qr_square_c32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.871 | 0.0005631 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-eager | 8 | 15 | 3.093 | 0.0003724 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-eager | 8 | 15 | 2.507 | 0.0006002 |
| slogdet_phase_c32_32x32 | tenferro-cuda-eager | 4 | 15 | 2.253 | 0.0004085 |
| householder_qr_square_c64_32x32 | tenferro-cuda-eager | 4 | 15 | 3.029 | 4.996e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-eager | 4 | 15 | 3.365 | 5.551e-07 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-eager | 4 | 15 | 3.405 | 3.886e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-eager | 4 | 15 | 2.106 | 4.755e-07 |
| householder_qr_square_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.666 | 0.0009546 |
| householder_qr_tall_f32_512x256 | tenferro-cuda-eager | 2 | 15 | 2.384 | 0.001435 |
| householder_qr_wide_f32_256x512 | tenferro-cuda-eager | 2 | 15 | 2.740 | 0.001037 |
| householder_qr_square_f64_256x256 | tenferro-cuda-eager | 1 | 15 | 3.059 | 1.098e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 3.623 | 9.992e-07 |
| householder_qr_wide_f64_256x512 | tenferro-cuda-eager | 1 | 15 | 5.211 | 1.649e-06 |
| householder_qr_square_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.945 | 0.001198 |
| householder_qr_tall_c32_512x256 | tenferro-cuda-eager | 2 | 15 | 2.661 | 0.001185 |
| householder_qr_wide_c32_256x512 | tenferro-cuda-eager | 2 | 15 | 2.979 | 0.001329 |
| slogdet_phase_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.858 | 0.01211 |
| householder_qr_square_c64_256x256 | tenferro-cuda-eager | 1 | 15 | 5.444 | 1.208e-06 |
| householder_qr_tall_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 5.760 | 1.11e-06 |
| householder_qr_wide_c64_256x512 | tenferro-cuda-eager | 1 | 15 | 9.800 | 1.206e-06 |
| slogdet_phase_c64_256x256 | tenferro-cuda-eager | 2 | 15 | 3.541 | 1.025e-06 |
| cholesky_f32_32x32 | pytorch-cuda | 32 | 15 | 2.910 | 0.0002352 |
| cholesky_f64_32x32 | pytorch-cuda | 16 | 15 | 2.639 | 2.188e-07 |
| svdvals_f32_32x32 | pytorch-cuda | 8 | 15 | 3.459 | 0.007749 |
| svdvals_f64_32x32 | pytorch-cuda | 1 | 15 | 5.085 | 3.387e-06 |
| eigvalsh_f32_32x32 | pytorch-cuda | 16 | 15 | 3.577 | 0.001073 |
| eigvalsh_f64_32x32 | pytorch-cuda | 4 | 15 | 2.461 | 5.433e-07 |
| pinv_f32_32x32 | pytorch-cuda | 4 | 15 | 2.139 | 0.001023 |
| pinv_f64_32x32 | pytorch-cuda | 1 | 15 | 4.932 | 9.374e-07 |
| inv_f32_32x32 | pytorch-cuda | 8 | 15 | 2.300 | 0.0004905 |
| inv_f64_32x32 | pytorch-cuda | 8 | 15 | 2.300 | 2.284e-07 |
| det_f32_32x32 | pytorch-cuda | 16 | 15 | 2.631 | 0.002641 |
| det_f64_32x32 | pytorch-cuda | 16 | 15 | 2.641 | 3.784e-07 |
| norm_f32_32x32 | pytorch-cuda | 128 | 15 | 1.870 | 0 |
| norm_f64_32x32 | pytorch-cuda | 128 | 15 | 1.846 | 0 |
| triangular_solve_f32_32x32 | pytorch-cuda | 32 | 15 | 1.672 | 9.242e-05 |
| triangular_solve_f64_32x32 | pytorch-cuda | 64 | 15 | 3.315 | 1.415e-07 |
| cholesky_f32_512x512 | pytorch-cuda | 8 | 15 | 2.237 | 0.0003554 |
| cholesky_f64_512x512 | pytorch-cuda | 1 | 15 | 2.199 | 4.418e-07 |
| svdvals_f32_512x512 | pytorch-cuda | 1 | 15 | 28.126 | 0.1733 |
| svdvals_f64_512x512 | pytorch-cuda | 1 | 15 | 594.555 | 0.0002066 |
| eigvalsh_f32_512x512 | pytorch-cuda | 1 | 15 | 4.372 | 0.008342 |
| eigvalsh_f64_512x512 | pytorch-cuda | 1 | 15 | 14.690 | 1.55e-06 |
| pinv_f32_512x512 | pytorch-cuda | 1 | 15 | 31.675 | 0.01126 |
| pinv_f64_512x512 | pytorch-cuda | 1 | 15 | 680.023 | 8.747e-06 |
| inv_f32_512x512 | pytorch-cuda | 2 | 15 | 3.890 | 0.002107 |
| inv_f64_512x512 | pytorch-cuda | 1 | 15 | 5.988 | 1.85e-06 |
| det_f32_512x512 | pytorch-cuda | 2 | 15 | 2.887 | 0.007965 |
| det_f64_512x512 | pytorch-cuda | 1 | 15 | 2.499 | 1.369e-05 |
| norm_f32_512x512 | pytorch-cuda | 128 | 15 | 2.576 | 0 |
| norm_f64_512x512 | pytorch-cuda | 128 | 15 | 2.530 | 1.909e-07 |
| triangular_solve_f32_512x512 | pytorch-cuda | 32 | 15 | 3.309 | 0.0006008 |
| triangular_solve_f64_512x512 | pytorch-cuda | 4 | 15 | 3.370 | 6.887e-07 |
| svd_full_square_f32_32x32 | pytorch-cuda | 8 | 15 | 2.736 | 0.005571 |
| svd_full_tall_f32_64x32 | pytorch-cuda | 4 | 15 | 3.362 | 0.005287 |
| svd_full_wide_f32_32x64 | pytorch-cuda | 4 | 15 | 3.586 | 0.005456 |
| slogdet_positive_f32_32x32 | pytorch-cuda | 16 | 15 | 3.355 | 0.000915 |
| slogdet_negative_f32_32x32 | pytorch-cuda | 16 | 15 | 3.316 | 0.000915 |
| svd_full_square_f64_32x32 | pytorch-cuda | 1 | 15 | 4.260 | 7.216e-06 |
| svd_full_tall_f64_64x32 | pytorch-cuda | 1 | 15 | 5.933 | 8.826e-06 |
| svd_full_wide_f64_32x64 | pytorch-cuda | 1 | 15 | 6.282 | 8.438e-06 |
| slogdet_positive_f64_32x32 | pytorch-cuda | 16 | 15 | 3.320 | 4.261e-07 |
| slogdet_negative_f64_32x32 | pytorch-cuda | 16 | 15 | 3.325 | 4.261e-07 |
| svd_full_square_c32_32x32 | pytorch-cuda | 8 | 15 | 3.788 | 0.011 |
| svd_full_tall_c32_64x32 | pytorch-cuda | 2 | 15 | 2.095 | 0.008819 |
| svd_full_wide_c32_32x64 | pytorch-cuda | 2 | 15 | 2.111 | 0.008905 |
| slogdet_positive_c32_32x32 | pytorch-cuda | 16 | 15 | 3.584 | 0.0004085 |
| slogdet_negative_c32_32x32 | pytorch-cuda | 16 | 15 | 3.821 | 0.0004085 |
| svd_full_square_c64_32x32 | pytorch-cuda | 1 | 15 | 11.363 | 9.992e-06 |
| svd_full_tall_c64_64x32 | pytorch-cuda | 1 | 15 | 15.404 | 1.188e-05 |
| svd_full_wide_c64_32x64 | pytorch-cuda | 1 | 15 | 16.145 | 1.299e-05 |
| slogdet_positive_c64_32x32 | pytorch-cuda | 16 | 15 | 3.571 | 4.755e-07 |
| slogdet_negative_c64_32x32 | pytorch-cuda | 16 | 15 | 3.554 | 4.755e-07 |
| svd_full_square_f32_256x256 | pytorch-cuda | 1 | 15 | 9.052 | 0.1135 |
| svd_full_tall_f32_512x256 | pytorch-cuda | 1 | 15 | 9.536 | 0.1033 |
| svd_full_wide_f32_256x512 | pytorch-cuda | 1 | 15 | 8.943 | 0.106 |
| slogdet_positive_f32_256x256 | pytorch-cuda | 8 | 15 | 2.948 | 0.0005003 |
| slogdet_negative_f32_256x256 | pytorch-cuda | 8 | 15 | 2.978 | 0.0005003 |
| svd_full_square_f64_256x256 | pytorch-cuda | 1 | 15 | 165.134 | 8.105e-05 |
| svd_full_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 130.719 | 7.76e-05 |
| svd_full_wide_f64_256x512 | pytorch-cuda | 1 | 15 | 121.256 | 7.938e-05 |
| slogdet_positive_f64_256x256 | pytorch-cuda | 4 | 15 | 2.877 | 1.708e-06 |
| slogdet_negative_f64_256x256 | pytorch-cuda | 4 | 15 | 2.885 | 1.708e-06 |
| svd_full_square_c32_256x256 | pytorch-cuda | 1 | 15 | 14.260 | 0.1305 |
| svd_full_tall_c32_512x256 | pytorch-cuda | 1 | 15 | 13.300 | 0.1089 |
| svd_full_wide_c32_256x512 | pytorch-cuda | 1 | 15 | 13.499 | 0.108 |
| slogdet_positive_c32_256x256 | pytorch-cuda | 8 | 15 | 3.263 | 0.01211 |
| slogdet_negative_c32_256x256 | pytorch-cuda | 8 | 15 | 3.263 | 0.01211 |
| svd_full_square_c64_256x256 | pytorch-cuda | 1 | 15 | 543.462 | 0.0002064 |
| svd_full_tall_c64_512x256 | pytorch-cuda | 1 | 15 | 371.563 | 0.0001994 |
| svd_full_wide_c64_256x512 | pytorch-cuda | 1 | 15 | 374.796 | 0.0001943 |
| slogdet_positive_c64_256x256 | pytorch-cuda | 2 | 15 | 2.792 | 1.23e-06 |
| slogdet_negative_c64_256x256 | pytorch-cuda | 2 | 15 | 2.794 | 1.23e-06 |
| householder_qr_square_f32_32x32 | pytorch-cuda | 64 | 15 | 3.946 | 0.000367 |
| householder_qr_tall_f32_64x32 | pytorch-cuda | 32 | 15 | 3.531 | 0.0003269 |
| householder_qr_wide_f32_32x64 | pytorch-cuda | 32 | 15 | 3.552 | 0.0003745 |
| householder_qr_square_f64_32x32 | pytorch-cuda | 8 | 15 | 2.291 | 6.106e-07 |
| householder_qr_tall_f64_64x32 | pytorch-cuda | 8 | 15 | 2.844 | 6.661e-07 |
| householder_qr_wide_f64_32x64 | pytorch-cuda | 8 | 15 | 2.880 | 5.551e-07 |
| householder_qr_square_c32_32x32 | pytorch-cuda | 32 | 15 | 2.167 | 0.0005833 |
| householder_qr_tall_c32_64x32 | pytorch-cuda | 32 | 15 | 3.663 | 0.0003756 |
| householder_qr_wide_c32_32x64 | pytorch-cuda | 32 | 15 | 3.485 | 0.0004972 |
| slogdet_phase_c32_32x32 | pytorch-cuda | 16 | 15 | 3.564 | 0.0004085 |
| householder_qr_square_c64_32x32 | pytorch-cuda | 8 | 15 | 3.856 | 1.221e-06 |
| householder_qr_tall_c64_64x32 | pytorch-cuda | 4 | 15 | 2.308 | 9.992e-07 |
| householder_qr_wide_c64_32x64 | pytorch-cuda | 4 | 15 | 2.485 | 1.166e-06 |
| slogdet_phase_c64_32x32 | pytorch-cuda | 16 | 15 | 3.562 | 5.706e-07 |
| householder_qr_square_f32_256x256 | pytorch-cuda | 4 | 15 | 3.037 | 0.001043 |
| householder_qr_tall_f32_512x256 | pytorch-cuda | 2 | 15 | 2.096 | 0.001475 |
| householder_qr_wide_f32_256x512 | pytorch-cuda | 2 | 15 | 2.490 | 0.0009736 |
| householder_qr_square_f64_256x256 | pytorch-cuda | 1 | 15 | 2.928 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 3.322 | 1.887e-06 |
| householder_qr_wide_f64_256x512 | pytorch-cuda | 1 | 15 | 4.803 | 1.759e-06 |
| householder_qr_square_c32_256x256 | pytorch-cuda | 4 | 15 | 3.171 | 0.001258 |
| householder_qr_tall_c32_512x256 | pytorch-cuda | 2 | 15 | 2.273 | 0.001161 |
| householder_qr_wide_c32_256x512 | pytorch-cuda | 2 | 15 | 2.609 | 0.001226 |
| slogdet_phase_c32_256x256 | pytorch-cuda | 8 | 15 | 3.263 | 0.01233 |
| householder_qr_square_c64_256x256 | pytorch-cuda | 1 | 15 | 5.069 | 2.109e-06 |
| householder_qr_tall_c64_512x256 | pytorch-cuda | 1 | 15 | 5.692 | 2.942e-06 |
| householder_qr_wide_c64_256x512 | pytorch-cuda | 1 | 15 | 9.694 | 2.276e-06 |
| slogdet_phase_c64_256x256 | pytorch-cuda | 2 | 15 | 2.791 | 1.025e-06 |
| cholesky_f32_32x32 | jax-cuda | 16 | 15 | 2.390 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda | 32 | 15 | 3.743 | 2.188e-07 |
| svdvals_f32_32x32 | jax-cuda | 4 | 15 | 3.404 | 0.003338 |
| svdvals_f64_32x32 | jax-cuda | 1 | 15 | 5.733 | 4.546e-06 |
| eigvalsh_f32_32x32 | jax-cuda | 4 | 15 | 2.633 | 0.004775 |
| eigvalsh_f64_32x32 | jax-cuda | 1 | 15 | 2.994 | 5.542e-06 |
| pinv_f32_32x32 | jax-cuda | 2 | 15 | 2.478 | 0.0006436 |
| pinv_f64_32x32 | jax-cuda | 1 | 15 | 5.831 | 1.367e-06 |
| inv_f32_32x32 | jax-cuda | 8 | 15 | 2.436 | 0.0004905 |
| inv_f64_32x32 | jax-cuda | 16 | 15 | 3.201 | 2.284e-07 |
| det_f32_32x32 | jax-cuda | 8 | 15 | 2.041 | 0.002641 |
| det_f64_32x32 | jax-cuda | 16 | 15 | 2.745 | 1.135e-06 |
| norm_f32_32x32 | jax-cuda | 16 | 15 | 2.153 | 0 |
| norm_f64_32x32 | jax-cuda | 16 | 15 | 2.602 | 0 |
| triangular_solve_f32_32x32 | jax-cuda | 16 | 15 | 3.415 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda | 16 | 15 | 2.169 | 1.609e-07 |
| cholesky_f32_512x512 | jax-cuda | 8 | 15 | 2.132 | 0.0003554 |
| cholesky_f64_512x512 | jax-cuda | 1 | 15 | 2.772 | 4.418e-07 |
| svdvals_f32_512x512 | jax-cuda | 1 | 15 | 26.031 | 0.1547 |
| svdvals_f64_512x512 | jax-cuda | 1 | 15 | 540.414 | 0.0002063 |
| eigvalsh_f32_512x512 | jax-cuda | 1 | 15 | 5.048 | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda | 1 | 15 | 19.592 | 1.55e-06 |
| pinv_f32_512x512 | jax-cuda | 1 | 15 | 29.315 | 0.009647 |
| pinv_f64_512x512 | jax-cuda | 1 | 15 | 615.892 | 8.975e-06 |
| inv_f32_512x512 | jax-cuda | 1 | 15 | 2.288 | 0.002107 |
| inv_f64_512x512 | jax-cuda | 1 | 15 | 6.418 | 1.85e-06 |
| det_f32_512x512 | jax-cuda | 2 | 15 | 3.287 | 0.004748 |
| det_f64_512x512 | jax-cuda | 1 | 15 | 2.909 | 1.426e-05 |
| norm_f32_512x512 | jax-cuda | 16 | 15 | 2.485 | 0.000205 |
| norm_f64_512x512 | jax-cuda | 16 | 15 | 2.651 | 1.909e-07 |
| triangular_solve_f32_512x512 | jax-cuda | 16 | 15 | 3.591 | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda | 8 | 15 | 2.093 | 1.157e-06 |
| svd_full_square_f32_32x32 | jax-cuda | 4 | 15 | 3.410 | 0.008731 |
| svd_full_tall_f32_64x32 | jax-cuda | 2 | 15 | 2.769 | 0.004865 |
| svd_full_wide_f32_32x64 | jax-cuda | 2 | 15 | 2.985 | 0.005495 |
| slogdet_positive_f32_32x32 | jax-cuda | 8 | 15 | 2.240 | 0.000915 |
| slogdet_negative_f32_32x32 | jax-cuda | 8 | 15 | 2.150 | 0.000915 |
| svd_full_square_f64_32x32 | jax-cuda | 1 | 15 | 5.642 | 1.008e-05 |
| svd_full_tall_f64_64x32 | jax-cuda | 1 | 15 | 6.140 | 8.327e-06 |
| svd_full_wide_f64_32x64 | jax-cuda | 1 | 15 | 6.198 | 1.197e-05 |
| slogdet_positive_f64_32x32 | jax-cuda | 16 | 15 | 2.800 | 4.261e-07 |
| slogdet_negative_f64_32x32 | jax-cuda | 16 | 15 | 2.313 | 4.261e-07 |
| svd_full_square_c32_32x32 | jax-cuda | 4 | 15 | 3.469 | 0.009079 |
| svd_full_tall_c32_64x32 | jax-cuda | 2 | 15 | 2.613 | 0.009498 |
| svd_full_wide_c32_32x64 | jax-cuda | 2 | 15 | 2.976 | 0.008495 |
| slogdet_positive_c32_32x32 | jax-cuda | 8 | 15 | 2.404 | 0.0004085 |
| slogdet_negative_c32_32x32 | jax-cuda | 8 | 15 | 2.026 | 0.0004085 |
| svd_full_square_c64_32x32 | jax-cuda | 1 | 15 | 14.425 | 9.215e-06 |
| svd_full_tall_c64_64x32 | jax-cuda | 1 | 15 | 15.287 | 1.243e-05 |
| svd_full_wide_c64_32x64 | jax-cuda | 1 | 15 | 15.972 | 1.288e-05 |
| slogdet_positive_c64_32x32 | jax-cuda | 16 | 15 | 3.540 | 4.755e-07 |
| slogdet_negative_c64_32x32 | jax-cuda | 16 | 15 | 2.887 | 4.755e-07 |
| svd_full_square_f32_256x256 | jax-cuda | 1 | 15 | 8.751 | 0.1152 |
| svd_full_tall_f32_512x256 | jax-cuda | 1 | 15 | 9.200 | 0.1022 |
| svd_full_wide_f32_256x512 | jax-cuda | 1 | 15 | 8.965 | 0.1056 |
| slogdet_positive_f32_256x256 | jax-cuda | 8 | 15 | 3.148 | 0.0005003 |
| slogdet_negative_f32_256x256 | jax-cuda | 4 | 15 | 1.714 | 0.0005003 |
| svd_full_square_f64_256x256 | jax-cuda | 1 | 15 | 160.420 | 8.171e-05 |
| svd_full_tall_f64_512x256 | jax-cuda | 1 | 15 | 127.825 | 7.76e-05 |
| svd_full_wide_f64_256x512 | jax-cuda | 1 | 15 | 118.429 | 7.938e-05 |
| slogdet_positive_f64_256x256 | jax-cuda | 4 | 15 | 3.218 | 1.864e-06 |
| slogdet_negative_f64_256x256 | jax-cuda | 4 | 15 | 3.252 | 1.864e-06 |
| svd_full_square_c32_256x256 | jax-cuda | 1 | 15 | 13.252 | 0.133 |
| svd_full_tall_c32_512x256 | jax-cuda | 1 | 15 | 13.267 | 0.1126 |
| svd_full_wide_c32_256x512 | jax-cuda | 1 | 15 | 13.220 | 0.1103 |
| slogdet_positive_c32_256x256 | jax-cuda | 8 | 15 | 3.440 | 0.01211 |
| slogdet_negative_c32_256x256 | jax-cuda | 8 | 15 | 3.475 | 0.01211 |
| svd_full_square_c64_256x256 | jax-cuda | 1 | 15 | 511.557 | 0.000204 |
| svd_full_tall_c64_512x256 | jax-cuda | 1 | 15 | 366.084 | 0.0002021 |
| svd_full_wide_c64_256x512 | jax-cuda | 1 | 15 | 367.175 | 0.0001946 |
| slogdet_positive_c64_256x256 | jax-cuda | 2 | 15 | 3.172 | 1.025e-06 |
| slogdet_negative_c64_256x256 | jax-cuda | 2 | 15 | 3.191 | 1.025e-06 |
| householder_qr_square_f32_32x32 | jax-cuda | 16 | 15 | 3.151 | 0.000367 |
| householder_qr_tall_f32_64x32 | jax-cuda | 8 | 15 | 2.589 | 0.0003269 |
| householder_qr_wide_f32_32x64 | jax-cuda | 8 | 15 | 2.473 | 0.0003745 |
| householder_qr_square_f64_32x32 | jax-cuda | 8 | 15 | 2.511 | 6.106e-07 |
| householder_qr_tall_f64_64x32 | jax-cuda | 8 | 15 | 3.149 | 6.661e-07 |
| householder_qr_wide_f64_32x64 | jax-cuda | 8 | 15 | 3.113 | 5.551e-07 |
| householder_qr_square_c32_32x32 | jax-cuda | 16 | 15 | 2.753 | 0.0005833 |
| householder_qr_tall_c32_64x32 | jax-cuda | 8 | 15 | 2.246 | 0.0003756 |
| householder_qr_wide_c32_32x64 | jax-cuda | 8 | 15 | 2.559 | 0.0004972 |
| slogdet_phase_c32_32x32 | jax-cuda | 8 | 15 | 1.796 | 0.0004085 |
| householder_qr_square_c64_32x32 | jax-cuda | 4 | 15 | 2.520 | 1.221e-06 |
| householder_qr_tall_c64_64x32 | jax-cuda | 4 | 15 | 2.939 | 9.992e-07 |
| householder_qr_wide_c64_32x64 | jax-cuda | 4 | 15 | 3.129 | 1.166e-06 |
| slogdet_phase_c64_32x32 | jax-cuda | 8 | 15 | 2.791 | 5.706e-07 |
| householder_qr_square_f32_256x256 | jax-cuda | 2 | 15 | 2.064 | 0.001043 |
| householder_qr_tall_f32_512x256 | jax-cuda | 2 | 15 | 2.676 | 0.001475 |
| householder_qr_wide_f32_256x512 | jax-cuda | 2 | 15 | 3.082 | 0.0009736 |
| householder_qr_square_f64_256x256 | jax-cuda | 1 | 15 | 3.565 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | jax-cuda | 1 | 15 | 3.879 | 1.887e-06 |
| householder_qr_wide_f64_256x512 | jax-cuda | 1 | 15 | 5.593 | 1.759e-06 |
| householder_qr_square_c32_256x256 | jax-cuda | 2 | 15 | 2.052 | 0.001258 |
| householder_qr_tall_c32_512x256 | jax-cuda | 2 | 15 | 2.592 | 0.001161 |
| householder_qr_wide_c32_256x512 | jax-cuda | 2 | 15 | 3.077 | 0.001226 |
| slogdet_phase_c32_256x256 | jax-cuda | 8 | 15 | 3.547 | 0.01233 |
| householder_qr_square_c64_256x256 | jax-cuda | 1 | 15 | 5.721 | 2.109e-06 |
| householder_qr_tall_c64_512x256 | jax-cuda | 1 | 15 | 6.012 | 2.942e-06 |
| householder_qr_wide_c64_256x512 | jax-cuda | 1 | 15 | 10.066 | 2.276e-06 |
| slogdet_phase_c64_256x256 | jax-cuda | 2 | 15 | 3.174 | 1.025e-06 |
| cholesky_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.264 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda-jit | 16 | 15 | 3.109 | 2.188e-07 |
| svdvals_f32_32x32 | jax-cuda-jit | 4 | 15 | 3.498 | 0.003338 |
| svdvals_f64_32x32 | jax-cuda-jit | 1 | 15 | 5.860 | 4.546e-06 |
| eigvalsh_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.643 | 0.004775 |
| eigvalsh_f64_32x32 | jax-cuda-jit | 1 | 15 | 2.949 | 5.542e-06 |
| pinv_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.860 | 0.0006436 |
| pinv_f64_32x32 | jax-cuda-jit | 1 | 15 | 5.321 | 1.367e-06 |
| inv_f32_32x32 | jax-cuda-jit | 16 | 15 | 3.165 | 0.0004905 |
| inv_f64_32x32 | jax-cuda-jit | 8 | 15 | 2.072 | 2.284e-07 |
| det_f32_32x32 | jax-cuda-jit | 16 | 15 | 3.312 | 0.002641 |
| det_f64_32x32 | jax-cuda-jit | 16 | 15 | 3.297 | 1.135e-06 |
| norm_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.218 | 0 |
| norm_f64_32x32 | jax-cuda-jit | 16 | 15 | 2.637 | 0 |
| triangular_solve_f32_32x32 | jax-cuda-jit | 16 | 15 | 3.120 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda-jit | 16 | 15 | 3.323 | 1.609e-07 |
| cholesky_f32_512x512 | jax-cuda-jit | 8 | 15 | 2.181 | 0.0003554 |
| cholesky_f64_512x512 | jax-cuda-jit | 1 | 15 | 2.755 | 4.418e-07 |
| svdvals_f32_512x512 | jax-cuda-jit | 1 | 15 | 25.873 | 0.1547 |
| svdvals_f64_512x512 | jax-cuda-jit | 1 | 15 | 537.918 | 0.0002063 |
| eigvalsh_f32_512x512 | jax-cuda-jit | 1 | 15 | 4.966 | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda-jit | 1 | 15 | 19.456 | 1.55e-06 |
| pinv_f32_512x512 | jax-cuda-jit | 1 | 15 | 28.797 | 0.009647 |
| pinv_f64_512x512 | jax-cuda-jit | 1 | 15 | 615.488 | 8.975e-06 |
| inv_f32_512x512 | jax-cuda-jit | 1 | 15 | 2.321 | 0.002107 |
| inv_f64_512x512 | jax-cuda-jit | 1 | 15 | 6.429 | 1.85e-06 |
| det_f32_512x512 | jax-cuda-jit | 2 | 15 | 3.249 | 0.004748 |
| det_f64_512x512 | jax-cuda-jit | 1 | 15 | 2.927 | 1.426e-05 |
| norm_f32_512x512 | jax-cuda-jit | 16 | 15 | 2.561 | 0.000205 |
| norm_f64_512x512 | jax-cuda-jit | 16 | 15 | 2.682 | 1.909e-07 |
| triangular_solve_f32_512x512 | jax-cuda-jit | 16 | 15 | 3.649 | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda-jit | 16 | 15 | 3.567 | 1.157e-06 |
| svd_full_square_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.816 | 0.008731 |
| svd_full_tall_f32_64x32 | jax-cuda-jit | 2 | 15 | 2.430 | 0.004865 |
| svd_full_wide_f32_32x64 | jax-cuda-jit | 2 | 15 | 2.326 | 0.005495 |
| slogdet_positive_f32_32x32 | jax-cuda-jit | 16 | 15 | 3.230 | 0.000915 |
| slogdet_negative_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.825 | 0.000915 |
| svd_full_square_f64_32x32 | jax-cuda-jit | 1 | 15 | 5.751 | 1.008e-05 |
| svd_full_tall_f64_64x32 | jax-cuda-jit | 1 | 15 | 6.592 | 8.327e-06 |
| svd_full_wide_f64_32x64 | jax-cuda-jit | 1 | 15 | 6.300 | 1.197e-05 |
| slogdet_positive_f64_32x32 | jax-cuda-jit | 16 | 15 | 3.249 | 4.261e-07 |
| slogdet_negative_f64_32x32 | jax-cuda-jit | 8 | 15 | 2.216 | 4.261e-07 |
| svd_full_square_c32_32x32 | jax-cuda-jit | 2 | 15 | 2.114 | 0.009079 |
| svd_full_tall_c32_64x32 | jax-cuda-jit | 2 | 15 | 2.963 | 0.009498 |
| svd_full_wide_c32_32x64 | jax-cuda-jit | 2 | 15 | 3.138 | 0.008495 |
| slogdet_positive_c32_32x32 | jax-cuda-jit | 8 | 15 | 2.212 | 0.0004085 |
| slogdet_negative_c32_32x32 | jax-cuda-jit | 8 | 15 | 2.381 | 0.0004085 |
| svd_full_square_c64_32x32 | jax-cuda-jit | 1 | 15 | 14.419 | 9.215e-06 |
| svd_full_tall_c64_64x32 | jax-cuda-jit | 1 | 15 | 15.293 | 1.243e-05 |
| svd_full_wide_c64_32x64 | jax-cuda-jit | 1 | 15 | 16.035 | 1.288e-05 |
| slogdet_positive_c64_32x32 | jax-cuda-jit | 16 | 15 | 3.022 | 4.755e-07 |
| slogdet_negative_c64_32x32 | jax-cuda-jit | 8 | 15 | 2.126 | 4.755e-07 |
| svd_full_square_f32_256x256 | jax-cuda-jit | 1 | 15 | 8.762 | 0.1152 |
| svd_full_tall_f32_512x256 | jax-cuda-jit | 1 | 15 | 9.178 | 0.1022 |
| svd_full_wide_f32_256x512 | jax-cuda-jit | 1 | 15 | 8.959 | 0.1056 |
| slogdet_positive_f32_256x256 | jax-cuda-jit | 8 | 15 | 3.110 | 0.0005003 |
| slogdet_negative_f32_256x256 | jax-cuda-jit | 8 | 15 | 3.050 | 0.0005003 |
| svd_full_square_f64_256x256 | jax-cuda-jit | 1 | 15 | 160.343 | 8.171e-05 |
| svd_full_tall_f64_512x256 | jax-cuda-jit | 1 | 15 | 127.903 | 7.76e-05 |
| svd_full_wide_f64_256x512 | jax-cuda-jit | 1 | 15 | 118.440 | 7.938e-05 |
| slogdet_positive_f64_256x256 | jax-cuda-jit | 4 | 15 | 3.194 | 1.864e-06 |
| slogdet_negative_f64_256x256 | jax-cuda-jit | 4 | 15 | 3.237 | 1.864e-06 |
| svd_full_square_c32_256x256 | jax-cuda-jit | 1 | 15 | 13.148 | 0.133 |
| svd_full_tall_c32_512x256 | jax-cuda-jit | 1 | 15 | 13.152 | 0.1126 |
| svd_full_wide_c32_256x512 | jax-cuda-jit | 1 | 15 | 13.206 | 0.1103 |
| slogdet_positive_c32_256x256 | jax-cuda-jit | 8 | 15 | 3.457 | 0.01211 |
| slogdet_negative_c32_256x256 | jax-cuda-jit | 4 | 15 | 1.941 | 0.01211 |
| svd_full_square_c64_256x256 | jax-cuda-jit | 1 | 15 | 511.527 | 0.000204 |
| svd_full_tall_c64_512x256 | jax-cuda-jit | 1 | 15 | 366.045 | 0.0002021 |
| svd_full_wide_c64_256x512 | jax-cuda-jit | 1 | 15 | 367.224 | 0.0001946 |
| slogdet_positive_c64_256x256 | jax-cuda-jit | 2 | 15 | 3.187 | 1.025e-06 |
| slogdet_negative_c64_256x256 | jax-cuda-jit | 2 | 15 | 3.155 | 1.025e-06 |
| householder_qr_square_f32_32x32 | jax-cuda-jit | 16 | 15 | 3.133 | 0.000367 |
| householder_qr_tall_f32_64x32 | jax-cuda-jit | 8 | 15 | 2.841 | 0.0003269 |
| householder_qr_wide_f32_32x64 | jax-cuda-jit | 16 | 15 | 3.041 | 0.0003745 |
| householder_qr_square_f64_32x32 | jax-cuda-jit | 8 | 15 | 2.547 | 6.106e-07 |
| householder_qr_tall_f64_64x32 | jax-cuda-jit | 8 | 15 | 3.441 | 6.661e-07 |
| householder_qr_wide_f64_32x64 | jax-cuda-jit | 8 | 15 | 3.425 | 5.551e-07 |
| householder_qr_square_c32_32x32 | jax-cuda-jit | 8 | 15 | 2.086 | 0.0005833 |
| householder_qr_tall_c32_64x32 | jax-cuda-jit | 8 | 15 | 3.132 | 0.0003756 |
| householder_qr_wide_c32_32x64 | jax-cuda-jit | 8 | 15 | 2.486 | 0.0004972 |
| slogdet_phase_c32_32x32 | jax-cuda-jit | 8 | 15 | 2.298 | 0.0004085 |
| householder_qr_square_c64_32x32 | jax-cuda-jit | 4 | 15 | 2.494 | 1.221e-06 |
| householder_qr_tall_c64_64x32 | jax-cuda-jit | 4 | 15 | 2.908 | 9.992e-07 |
| householder_qr_wide_c64_32x64 | jax-cuda-jit | 4 | 15 | 3.111 | 1.166e-06 |
| slogdet_phase_c64_32x32 | jax-cuda-jit | 8 | 15 | 2.475 | 5.706e-07 |
| householder_qr_square_f32_256x256 | jax-cuda-jit | 4 | 15 | 3.657 | 0.001043 |
| householder_qr_tall_f32_512x256 | jax-cuda-jit | 2 | 15 | 2.677 | 0.001475 |
| householder_qr_wide_f32_256x512 | jax-cuda-jit | 2 | 15 | 3.068 | 0.0009736 |
| householder_qr_square_f64_256x256 | jax-cuda-jit | 1 | 15 | 3.579 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | jax-cuda-jit | 1 | 15 | 3.986 | 1.887e-06 |
| householder_qr_wide_f64_256x512 | jax-cuda-jit | 1 | 15 | 5.552 | 1.759e-06 |
| householder_qr_square_c32_256x256 | jax-cuda-jit | 2 | 15 | 2.025 | 0.001258 |
| householder_qr_tall_c32_512x256 | jax-cuda-jit | 2 | 15 | 2.607 | 0.001161 |
| householder_qr_wide_c32_256x512 | jax-cuda-jit | 2 | 15 | 3.157 | 0.001226 |
| slogdet_phase_c32_256x256 | jax-cuda-jit | 8 | 15 | 3.605 | 0.01233 |
| householder_qr_square_c64_256x256 | jax-cuda-jit | 1 | 15 | 5.826 | 2.109e-06 |
| householder_qr_tall_c64_512x256 | jax-cuda-jit | 1 | 15 | 6.072 | 2.942e-06 |
| householder_qr_wide_c64_256x512 | jax-cuda-jit | 1 | 15 | 10.087 | 2.276e-06 |
| slogdet_phase_c64_256x256 | jax-cuda-jit | 2 | 15 | 3.133 | 1.025e-06 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api
suite_file: benchmarks/gpu/linalg_public_api.yaml
timestamp: '2026-10-13T16:02:23Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: f3e921d714c2a7daacedacd61c7b6edf7650c6f7
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
    ./scripts/run_gpu_public_api.sh linalg_public_api
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 100
  selected_cases: 100
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: cf552be18d7b121a7c203aa91799ffa2fb0642b99b46b45210fcab7f29eb0c6c
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/multi.rs: 03a6f94ba97b0dfe533469c3e97f5ac20e7bed2e2bed402dea047d82bcf57b19
    src/bin/benchmark_gpu_public_api/structural.rs: 8121b12b5fed3b0acbc71e2218856173d53a5acc93092b93c7fd3309d4f45962
    src/bin/benchmark_gpu_public_api/linalg.rs: 84162923fa0d0772c88ec9c4eab169ad3a75e5d16ff76961d1541670dd8565c1
    src/bin/benchmark_gpu_public_api/reuse.rs: ad72b565373904275128d7beb8673bada08d922e8f310be6cedb0d2d8187898f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 850ab2ead404652e542a2f13085d7ee070edbda379ccaeec4658219973d242d2
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 8c3a56d805475c8d250b0bffd499b97b351b51d70061f1af645e75fe572e6bff
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/linalg_public_api/20261013_160222/fixtures/manifest.json: 729ab872a820044a508cf73ef9b3e3ecfd05b44d7d31bdf96be535132b2bd6e0
    target/release/benchmark_gpu_public_api: 587a654c59546a650636c06e850e109e52dde2395dd6b89a9fc4c687a685b2e7
  reference_numpy_version: 2.4.2
```
