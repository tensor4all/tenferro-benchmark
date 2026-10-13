# gpu/linalg_public_api: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api/20261013_163327/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | 127.373 (1.561) | 206.196 (6.742) | 90.207 (0.887) | 151.277 (2.900) | 176.547 (13.552) |
| cholesky_f64_32x32 | 265.956 (5.051) | 376.038 (3.600) | 160.966 (0.804) | 168.987 (12.182) | 201.485 (7.078) |
| svdvals_f32_32x32 | 660.928 (25.413) | unsupported | 406.038 (2.702) | 768.373 (16.280) | 887.415 (20.453) |
| svdvals_f64_32x32 | 5492.768 (17.756) | unsupported | 5089.551 (10.512) | 5824.538 (51.878) | 5837.514 (60.182) |
| eigvalsh_f32_32x32 | 253.615 (7.563) | 314.927 (4.163) | 221.374 (3.612) | 673.550 (8.802) | 699.469 (36.384) |
| eigvalsh_f64_32x32 | 634.334 (6.283) | 693.561 (7.479) | 611.951 (2.062) | 3141.436 (24.272) | 3105.290 (10.238) |
| pinv_f32_32x32 | unsupported | unsupported | 537.803 (4.992) | 1267.701 (72.130) | 733.552 (27.483) |
| pinv_f64_32x32 | unsupported | unsupported | 4937.386 (5.288) | 5826.198 (23.192) | 5708.272 (87.025) |
| inv_f32_32x32 | unsupported | unsupported | 293.730 (9.639) | 195.147 (5.734) | 293.803 (38.807) |
| inv_f64_32x32 | unsupported | unsupported | 301.468 (22.447) | 215.237 (25.666) | 213.227 (23.089) |
| det_f32_32x32 | 653.739 (7.319) | 828.975 (30.895) | 181.540 (8.562) | 160.379 (0.877) | 261.119 (8.323) |
| det_f64_32x32 | 652.762 (17.828) | 837.685 (28.394) | 182.698 (5.241) | 206.739 (52.721) | 232.194 (32.978) |
| norm_f32_32x32 | 83.991 (0.811) | 204.931 (8.502) | 14.860 (0.086) | 132.068 (5.759) | 165.148 (26.710) |
| norm_f64_32x32 | 83.370 (2.135) | 213.692 (17.168) | 14.965 (0.100) | 160.042 (4.686) | 168.754 (13.514) |
| triangular_solve_f32_32x32 | 88.262 (2.028) | 307.035 (55.762) | 53.719 (0.779) | 190.112 (30.425) | 182.177 (29.068) |
| triangular_solve_f64_32x32 | 84.087 (2.091) | 272.925 (17.673) | 54.182 (1.719) | 233.550 (26.380) | 210.669 (46.281) |
| cholesky_f32_512x512 | 354.369 (9.311) | 485.218 (19.115) | 280.109 (1.540) | 270.782 (8.257) | 272.114 (5.954) |
| cholesky_f64_512x512 | 2233.843 (13.116) | 2502.304 (5.574) | 2199.983 (2.227) | 2767.638 (22.468) | 2750.609 (27.976) |
| svdvals_f32_512x512 | 25671.958 (20.978) | unsupported | 28149.651 (75.084) | 26037.148 (20.962) | 25870.686 (14.772) |
| svdvals_f64_512x512 | 541163.734 (45.568) | unsupported | 595682.266 (66.796) | 541501.663 (30.245) | 537881.216 (81.051) |
| eigvalsh_f32_512x512 | 4265.220 (16.403) | 4395.797 (23.846) | 4374.443 (8.800) | 5040.858 (65.294) | 4998.035 (71.114) |
| eigvalsh_f64_512x512 | 14708.456 (50.245) | 14851.415 (57.935) | 14727.924 (65.004) | 19561.523 (106.800) | 19538.870 (118.311) |
| pinv_f32_512x512 | unsupported | unsupported | 31670.520 (21.176) | 29342.502 (44.665) | 28915.938 (33.408) |
| pinv_f64_512x512 | unsupported | unsupported | 681258.754 (45.768) | 615665.021 (112.221) | 615494.770 (76.362) |
| inv_f32_512x512 | unsupported | unsupported | 1951.293 (6.597) | 2299.548 (23.014) | 2279.238 (33.000) |
| inv_f64_512x512 | unsupported | unsupported | 5996.484 (9.169) | 6395.455 (35.586) | 6412.025 (19.644) |
| det_f32_512x512 | 1774.960 (7.910) | 1960.530 (35.829) | 1446.482 (1.662) | 1635.849 (10.040) | 1634.670 (12.901) |
| det_f64_512x512 | 2856.176 (8.991) | 3165.777 (61.089) | 2506.560 (11.814) | 2906.442 (18.222) | 2882.604 (24.845) |
| norm_f32_512x512 | 35.666 (1.162) | 141.588 (0.714) | 20.241 (0.178) | 156.992 (4.230) | 156.998 (32.172) |
| norm_f64_512x512 | 37.128 (1.064) | 138.315 (3.725) | 20.836 (0.251) | 154.508 (3.763) | 165.029 (3.209) |
| triangular_solve_f32_512x512 | 71.448 (0.266) | 226.225 (13.459) | 103.670 (0.034) | 235.822 (14.159) | 171.003 (4.241) |
| triangular_solve_f64_512x512 | 141.515 (0.163) | 225.989 (2.768) | 844.819 (0.291) | 223.295 (2.235) | 241.419 (8.083) |
| svd_full_square_f32_32x32 | 616.639 (15.449) | 787.795 (6.554) | 345.816 (3.409) | 756.170 (26.041) | 720.648 (21.365) |
| svd_full_tall_f32_64x32 | 847.793 (18.857) | 1036.067 (35.338) | 847.359 (5.483) | 1380.339 (15.926) | 1417.609 (70.981) |
| svd_full_wide_f32_32x64 | 906.000 (8.000) | 1148.344 (16.187) | 931.485 (26.088) | 1488.823 (23.256) | 1209.333 (47.946) |
| slogdet_positive_f32_32x32 | 305.542 (8.084) | 563.771 (11.824) | 213.481 (5.984) | 167.123 (1.106) | 271.727 (23.781) |
| slogdet_negative_f32_32x32 | 304.248 (6.066) | 562.186 (6.033) | 212.812 (0.604) | 168.073 (3.306) | 208.255 (48.543) |
| svd_full_square_f64_32x32 | 5106.126 (26.122) | 5244.610 (15.633) | 4282.316 (5.632) | 5650.697 (114.126) | 5739.423 (68.899) |
| svd_full_tall_f64_64x32 | 5868.547 (23.972) | 6100.683 (23.913) | 5972.092 (7.358) | 6051.421 (21.140) | 6049.545 (14.578) |
| svd_full_wide_f64_32x64 | 6005.916 (13.765) | 6192.042 (20.826) | 6318.005 (21.118) | 6214.999 (69.082) | 6265.119 (85.338) |
| slogdet_positive_f64_32x32 | 316.149 (7.567) | 493.844 (9.054) | 212.514 (9.496) | 281.269 (28.420) | 256.631 (4.411) |
| slogdet_negative_f64_32x32 | 447.303 (11.976) | 497.937 (3.523) | 211.566 (2.912) | 248.162 (27.461) | 297.181 (4.503) |
| svd_full_square_c32_32x32 | 784.138 (8.396) | 912.919 (13.855) | 476.985 (1.684) | 1049.500 (15.832) | 1057.022 (20.878) |
| svd_full_tall_c32_64x32 | 1047.274 (16.291) | 1152.470 (17.373) | 1051.616 (12.338) | 1444.384 (116.333) | 1548.866 (107.768) |
| svd_full_wide_c32_32x64 | 1053.537 (16.948) | 1188.160 (33.726) | 1056.785 (8.864) | 1559.006 (68.404) | 1557.305 (68.354) |
| slogdet_positive_c32_32x32 | 312.347 (8.789) | 824.828 (12.455) | 224.570 (11.147) | 282.961 (21.996) | 274.299 (20.333) |
| slogdet_negative_c32_32x32 | 327.998 (13.552) | 785.709 (8.744) | 224.667 (1.865) | 235.101 (38.857) | 295.002 (14.925) |
| svd_full_square_c64_32x32 | 14325.204 (14.222) | 14496.126 (23.358) | 11400.582 (6.192) | 14395.271 (21.641) | 15402.520 (38.785) |
| svd_full_tall_c64_64x32 | 15197.666 (20.203) | 15460.664 (19.487) | 15451.811 (10.056) | 15286.640 (16.242) | 15296.888 (14.673) |
| svd_full_wide_c64_32x64 | 15926.919 (38.506) | 16170.599 (39.450) | 16192.873 (14.383) | 15973.081 (76.771) | 16026.556 (25.172) |
| slogdet_positive_c64_32x32 | 583.495 (26.029) | 860.332 (27.972) | 229.256 (1.211) | 136.642 (4.790) | 196.618 (17.741) |
| slogdet_negative_c64_32x32 | 531.479 (23.560) | 818.697 (11.993) | 229.204 (0.830) | 134.515 (8.548) | 298.081 (9.540) |
| svd_full_square_f32_256x256 | 8560.251 (24.088) | 8657.061 (22.616) | 9086.058 (12.597) | 8630.624 (67.724) | 8761.831 (102.219) |
| svd_full_tall_f32_512x256 | 9020.659 (23.529) | 9163.869 (49.276) | 9569.159 (18.559) | 9115.155 (24.230) | 9209.484 (13.334) |
| svd_full_wide_f32_256x512 | 8803.422 (30.636) | 8923.464 (41.772) | 8959.066 (24.019) | 8875.693 (15.424) | 8936.508 (19.150) |
| slogdet_positive_f32_256x256 | 618.190 (7.167) | 872.405 (7.655) | 368.993 (0.392) | 377.621 (2.946) | 380.875 (10.787) |
| slogdet_negative_f32_256x256 | 613.551 (7.646) | 872.930 (35.198) | 372.453 (0.208) | 383.879 (4.263) | 391.031 (14.117) |
| svd_full_square_f64_256x256 | 161282.428 (15.456) | 162068.582 (23.792) | 165543.619 (24.686) | 160310.985 (43.034) | 160332.343 (27.281) |
| svd_full_tall_f64_512x256 | 128465.139 (115.224) | 129122.081 (84.885) | 131050.837 (50.367) | 127848.142 (121.760) | 127869.808 (104.827) |
| svd_full_wide_f64_256x512 | 118831.398 (144.147) | 119478.656 (112.383) | 121550.642 (19.405) | 118360.229 (142.229) | 118424.767 (78.980) |
| slogdet_positive_f64_256x256 | 963.122 (18.434) | 1204.028 (19.122) | 720.159 (0.671) | 803.588 (4.771) | 805.176 (5.103) |
| slogdet_negative_f64_256x256 | 960.311 (14.388) | 1209.205 (26.868) | 722.677 (0.470) | 810.586 (3.982) | 815.529 (8.934) |
| svd_full_square_c32_256x256 | 13107.528 (17.367) | 13247.174 (18.889) | 14280.424 (25.218) | 13233.770 (19.029) | 13224.767 (19.482) |
| svd_full_tall_c32_512x256 | 13097.207 (21.316) | 13256.494 (41.380) | 13333.079 (20.653) | 13247.991 (15.639) | 13229.031 (24.259) |
| svd_full_wide_c32_256x512 | 13166.118 (28.870) | 13328.463 (20.203) | 13512.640 (17.259) | 13284.535 (58.293) | 13263.674 (61.574) |
| slogdet_positive_c32_256x256 | 660.377 (10.277) | 989.094 (9.189) | 408.804 (0.732) | 424.569 (7.758) | 428.591 (15.695) |
| slogdet_negative_c32_256x256 | 648.976 (7.747) | 925.727 (20.466) | 408.827 (0.448) | 430.883 (5.394) | 433.231 (9.713) |
| svd_full_square_c64_256x256 | 514644.631 (58.237) | 517129.548 (54.286) | 544849.621 (50.563) | 511322.306 (87.208) | 511535.724 (166.742) |
| svd_full_tall_c64_512x256 | 368106.432 (1727.204) | 369854.967 (222.187) | 372368.756 (35.413) | 365875.928 (77.431) | 365882.037 (138.001) |
| svd_full_wide_c64_256x512 | 371072.825 (63.616) | 371234.449 (53.540) | 375665.441 (67.512) | 366935.555 (63.724) | 367084.022 (123.180) |
| slogdet_positive_c64_256x256 | 1691.618 (17.731) | 1907.782 (32.426) | 1397.858 (1.544) | 1576.142 (8.043) | 1572.790 (6.065) |
| slogdet_negative_c64_256x256 | 1680.835 (9.087) | 1795.728 (11.479) | 1397.196 (1.332) | 1588.433 (6.139) | 1570.794 (8.658) |
| householder_qr_square_f32_32x32 | 206.274 (4.160) | 353.631 (5.879) | 60.038 (0.106) | 127.164 (4.315) | 283.902 (28.407) |
| householder_qr_tall_f32_64x32 | 248.391 (13.794) | 365.603 (20.150) | 106.120 (0.807) | 273.846 (23.326) | 378.433 (86.583) |
| householder_qr_wide_f32_32x64 | 201.528 (16.074) | 301.214 (13.757) | 107.316 (3.116) | 362.893 (12.505) | 374.458 (33.603) |
| householder_qr_square_f64_32x32 | 442.241 (4.162) | 594.377 (4.203) | 287.565 (0.308) | 316.389 (4.634) | 323.553 (7.178) |
| householder_qr_tall_f64_64x32 | 506.545 (6.195) | 666.542 (19.151) | 356.766 (0.102) | 391.795 (9.765) | 418.292 (9.368) |
| householder_qr_wide_f64_32x64 | 494.959 (5.301) | 627.717 (20.716) | 361.214 (0.564) | 389.642 (18.563) | 541.711 (16.021) |
| householder_qr_square_c32_32x32 | 230.764 (1.475) | 364.730 (2.127) | 68.096 (0.059) | 200.045 (55.631) | 301.520 (36.509) |
| householder_qr_tall_c32_64x32 | 294.446 (22.753) | 408.467 (29.849) | 113.557 (2.323) | 365.611 (23.917) | 401.194 (39.498) |
| householder_qr_wide_c32_32x64 | 229.308 (10.052) | 322.496 (13.086) | 127.248 (8.173) | 364.458 (7.240) | 360.048 (73.762) |
| slogdet_phase_c32_32x32 | 436.131 (6.562) | 553.227 (11.642) | 259.823 (20.644) | 235.273 (47.965) | 259.656 (6.643) |
| householder_qr_square_c64_32x32 | 620.519 (8.325) | 764.522 (8.483) | 482.322 (1.950) | 587.264 (14.978) | 632.783 (7.555) |
| householder_qr_tall_c64_64x32 | 697.595 (13.331) | 839.847 (16.480) | 578.706 (0.266) | 690.906 (13.873) | 737.083 (14.378) |
| householder_qr_wide_c64_32x64 | 718.721 (6.806) | 835.028 (11.366) | 623.008 (0.370) | 776.593 (4.971) | 783.615 (4.971) |
| slogdet_phase_c64_32x32 | 422.015 (5.751) | 558.198 (10.511) | 230.231 (15.915) | 286.671 (56.183) | 298.644 (19.709) |
| householder_qr_square_f32_256x256 | 855.219 (5.554) | 920.804 (7.721) | 759.980 (4.286) | 978.928 (31.509) | 909.404 (10.837) |
| householder_qr_tall_f32_512x256 | 1178.913 (9.513) | 1160.216 (11.973) | 1051.334 (2.535) | 1306.239 (28.612) | 1336.803 (10.746) |
| householder_qr_wide_f32_256x512 | 1374.902 (6.591) | 1354.075 (13.752) | 1243.531 (8.206) | 1503.373 (43.113) | 1542.076 (31.016) |
| householder_qr_square_f64_256x256 | 3135.767 (12.037) | 2999.207 (12.877) | 2931.214 (8.458) | 3521.144 (52.646) | 3546.502 (54.695) |
| householder_qr_tall_f64_512x256 | 3544.450 (10.711) | 3609.358 (20.277) | 3325.656 (16.527) | 3884.487 (61.367) | 4003.881 (34.500) |
| householder_qr_wide_f64_256x512 | 5138.500 (52.337) | 5204.252 (39.097) | 4821.833 (18.912) | 5565.347 (76.309) | 5589.587 (25.533) |
| householder_qr_square_c32_256x256 | 863.203 (6.071) | 957.470 (6.040) | 793.371 (2.537) | 1070.959 (20.953) | 1073.309 (17.312) |
| householder_qr_tall_c32_512x256 | 1192.778 (8.611) | 1314.737 (9.348) | 1138.759 (6.052) | 1378.206 (28.603) | 1407.490 (37.696) |
| householder_qr_wide_c32_256x512 | 1362.878 (7.050) | 1511.870 (7.198) | 1306.420 (5.506) | 1566.774 (24.134) | 1580.596 (35.151) |
| slogdet_phase_c32_256x256 | 666.928 (11.130) | 930.042 (13.481) | 408.696 (0.587) | 447.377 (6.416) | 451.579 (10.313) |
| householder_qr_square_c64_256x256 | 5338.664 (200.136) | 5473.247 (184.309) | 5075.086 (9.713) | 5834.964 (30.666) | 5847.134 (14.368) |
| householder_qr_tall_c64_512x256 | 5855.271 (8.050) | 5792.467 (22.213) | 5701.185 (19.690) | 6431.001 (97.026) | 6049.421 (83.574) |
| householder_qr_wide_c64_256x512 | 9792.421 (52.267) | 9807.314 (39.520) | 9707.827 (8.700) | 10065.737 (28.468) | 10078.262 (41.759) |
| slogdet_phase_c64_256x256 | 1655.997 (26.591) | 1876.502 (18.590) | 1396.716 (1.150) | 1578.477 (23.102) | 1585.879 (9.310) |
| householder_from_factors_default_f32_64x32 | 494.792 (22.482) | 916.120 (14.784) | unsupported | unsupported | unsupported |
| householder_append_default_f32_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f32_64x32 | 21.203 (1.224) | 39.024 (2.559) | 8.826 (0.077) | 163.938 (22.665) | 156.244 (20.634) |
| householder_q_columns_prefix_f32_64x32 | unsupported | unsupported | 100.021 (1.095) | 408.012 (27.183) | 311.279 (7.808) |
| householder_q_columns_complement_f32_64x32 | unsupported | unsupported | 183.328 (0.264) | unsupported | unsupported |
| householder_from_factors_default_f64_64x32 | 764.092 (14.821) | 1148.488 (64.263) | unsupported | unsupported | unsupported |
| householder_append_default_f64_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f64_64x32 | 31.664 (1.857) | 41.348 (2.172) | 8.623 (0.093) | 123.789 (11.059) | 160.840 (34.846) |
| householder_q_columns_prefix_f64_64x32 | unsupported | unsupported | 107.823 (8.161) | 409.252 (8.392) | 342.918 (14.640) |
| householder_q_columns_complement_f64_64x32 | unsupported | unsupported | 231.262 (0.129) | unsupported | unsupported |
| householder_from_factors_default_c32_64x32 | 581.963 (27.835) | 949.751 (7.413) | unsupported | unsupported | unsupported |
| householder_append_default_c32_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c32_64x32 | 31.898 (0.732) | 42.957 (3.179) | 9.780 (1.405) | 164.206 (22.233) | 157.592 (36.690) |
| householder_q_columns_prefix_c32_64x32 | unsupported | unsupported | 128.310 (3.793) | 418.431 (64.226) | 397.995 (53.819) |
| householder_q_columns_complement_c32_64x32 | unsupported | unsupported | 182.084 (5.383) | unsupported | unsupported |
| householder_from_factors_default_c64_64x32 | 1077.203 (30.622) | 1628.183 (11.319) | unsupported | unsupported | unsupported |
| householder_append_default_c64_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c64_64x32 | 30.462 (1.624) | 64.903 (2.644) | 8.757 (0.164) | 160.146 (29.211) | 170.014 (30.310) |
| householder_q_columns_prefix_c64_64x32 | unsupported | unsupported | 108.904 (2.821) | 455.323 (30.682) | 379.399 (18.745) |
| householder_q_columns_complement_c64_64x32 | unsupported | unsupported | 437.775 (0.365) | unsupported | unsupported |
| householder_from_factors_default_f32_512x256 | 1431.529 (10.453) | 1619.332 (10.392) | unsupported | unsupported | unsupported |
| householder_append_default_f32_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f32_512x256 | 33.950 (1.114) | 25.457 (0.434) | 8.935 (0.162) | 117.025 (15.550) | 145.220 (26.313) |
| householder_q_columns_prefix_f32_512x256 | unsupported | unsupported | 101.102 (1.988) | 380.009 (18.928) | 310.026 (8.393) |
| householder_q_columns_complement_f32_512x256 | unsupported | unsupported | 201.815 (0.685) | unsupported | unsupported |
| householder_from_factors_default_f64_512x256 | 4164.554 (18.806) | 4191.279 (23.109) | unsupported | unsupported | unsupported |
| householder_append_default_f64_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f64_512x256 | 30.632 (3.245) | 28.608 (2.612) | 9.403 (0.515) | 155.957 (28.893) | 162.951 (32.319) |
| householder_q_columns_prefix_f64_512x256 | unsupported | unsupported | 193.575 (0.326) | 473.899 (19.109) | 328.834 (18.401) |
| householder_q_columns_complement_f64_512x256 | unsupported | unsupported | 1549.858 (3.580) | unsupported | unsupported |
| householder_from_factors_default_c32_512x256 | 1558.124 (11.736) | 1660.083 (19.214) | unsupported | unsupported | unsupported |
| householder_append_default_c32_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c32_512x256 | 29.024 (0.821) | 30.612 (1.294) | 8.911 (1.015) | 161.279 (22.222) | 161.630 (32.467) |
| householder_q_columns_prefix_c32_512x256 | unsupported | unsupported | 123.719 (0.454) | 464.698 (50.671) | 358.985 (60.119) |
| householder_q_columns_complement_c32_512x256 | unsupported | unsupported | 257.598 (0.292) | unsupported | unsupported |
| householder_from_factors_default_c64_512x256 | 7314.806 (280.826) | 7339.798 (79.811) | unsupported | unsupported | unsupported |
| householder_append_default_c64_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c64_512x256 | 31.561 (2.032) | 33.219 (0.950) | 9.069 (0.132) | 164.975 (22.158) | 158.255 (31.557) |
| householder_q_columns_prefix_c64_512x256 | unsupported | unsupported | 376.743 (0.077) | 584.217 (48.074) | 505.059 (30.655) |
| householder_q_columns_complement_c64_512x256 | unsupported | unsupported | 4960.285 (4.752) | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 2.038 | 0.0002352 |
| cholesky_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.128 | 2.188e-07 |
| svdvals_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.644 | 0.003338 |
| svdvals_f64_32x32 | tenferro-cuda-direct | 1 | 15 | 5.493 | 4.546e-06 |
| eigvalsh_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.029 | 0.001073 |
| eigvalsh_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.537 | 5.433e-07 |
| pinv_f32_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| det_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.615 | 0.002641 |
| det_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.611 | 1.135e-06 |
| norm_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.688 | 0 |
| norm_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.668 | 0 |
| triangular_solve_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.824 | 0.0001546 |
| triangular_solve_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.691 | 1.609e-07 |
| cholesky_f32_512x512 | tenferro-cuda-direct | 8 | 15 | 2.835 | 0.0003554 |
| cholesky_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.234 | 4.418e-07 |
| svdvals_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 25.672 | 0.1547 |
| svdvals_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 541.164 | 0.0002063 |
| eigvalsh_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 4.265 | 0.008342 |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 14.708 | 1.55e-06 |
| pinv_f32_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| det_f32_512x512 | tenferro-cuda-direct | 2 | 15 | 3.550 | 0.008577 |
| det_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.856 | 1.426e-05 |
| norm_f32_512x512 | tenferro-cuda-direct | 64 | 15 | 2.283 | 0 |
| norm_f64_512x512 | tenferro-cuda-direct | 64 | 15 | 2.376 | 1.909e-07 |
| triangular_solve_f32_512x512 | tenferro-cuda-direct | 32 | 15 | 2.286 | 0.0007877 |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | 16 | 15 | 2.264 | 1.157e-06 |
| svd_full_square_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.467 | 0.008731 |
| svd_full_tall_f32_64x32 | tenferro-cuda-direct | 4 | 15 | 3.391 | 0.004865 |
| svd_full_wide_f32_32x64 | tenferro-cuda-direct | 4 | 15 | 3.624 | 0.005495 |
| slogdet_positive_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.444 | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.434 | 0.000915 |
| svd_full_square_f64_32x32 | tenferro-cuda-direct | 1 | 15 | 5.106 | 1.007e-05 |
| svd_full_tall_f64_64x32 | tenferro-cuda-direct | 1 | 15 | 5.869 | 8.327e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-direct | 1 | 15 | 6.006 | 1.197e-05 |
| slogdet_positive_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.529 | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 3.578 | 4.261e-07 |
| svd_full_square_c32_32x32 | tenferro-cuda-direct | 4 | 15 | 3.137 | 0.009079 |
| svd_full_tall_c32_64x32 | tenferro-cuda-direct | 2 | 15 | 2.095 | 0.009498 |
| svd_full_wide_c32_32x64 | tenferro-cuda-direct | 2 | 15 | 2.107 | 0.008495 |
| slogdet_positive_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.499 | 0.0007148 |
| slogdet_negative_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.624 | 0.0007148 |
| svd_full_square_c64_32x32 | tenferro-cuda-direct | 1 | 15 | 14.325 | 9.215e-06 |
| svd_full_tall_c64_64x32 | tenferro-cuda-direct | 1 | 15 | 15.198 | 1.221e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-direct | 1 | 15 | 15.927 | 1.31e-05 |
| slogdet_positive_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 4.668 | 3.804e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.126 | 3.804e-07 |
| svd_full_square_f32_256x256 | tenferro-cuda-direct | 1 | 15 | 8.560 | 0.1152 |
| svd_full_tall_f32_512x256 | tenferro-cuda-direct | 1 | 15 | 9.021 | 0.1022 |
| svd_full_wide_f32_256x512 | tenferro-cuda-direct | 1 | 15 | 8.803 | 0.1056 |
| slogdet_positive_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.473 | 0.0005003 |
| slogdet_negative_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.454 | 0.0005003 |
| svd_full_square_f64_256x256 | tenferro-cuda-direct | 1 | 15 | 161.282 | 8.171e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 128.465 | 7.76e-05 |
| svd_full_wide_f64_256x512 | tenferro-cuda-direct | 1 | 15 | 118.831 | 7.938e-05 |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.852 | 1.864e-06 |
| slogdet_negative_f64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.841 | 1.864e-06 |
| svd_full_square_c32_256x256 | tenferro-cuda-direct | 1 | 15 | 13.108 | 0.133 |
| svd_full_tall_c32_512x256 | tenferro-cuda-direct | 1 | 15 | 13.097 | 0.1126 |
| svd_full_wide_c32_256x512 | tenferro-cuda-direct | 1 | 15 | 13.166 | 0.1103 |
| slogdet_positive_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.642 | 0.01211 |
| slogdet_negative_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.596 | 0.01211 |
| svd_full_square_c64_256x256 | tenferro-cuda-direct | 1 | 15 | 514.645 | 0.000204 |
| svd_full_tall_c64_512x256 | tenferro-cuda-direct | 1 | 15 | 368.106 | 0.0002024 |
| svd_full_wide_c64_256x512 | tenferro-cuda-direct | 1 | 15 | 371.073 | 0.0001941 |
| slogdet_positive_c64_256x256 | tenferro-cuda-direct | 2 | 15 | 3.383 | 1.025e-06 |
| slogdet_negative_c64_256x256 | tenferro-cuda-direct | 2 | 15 | 3.362 | 1.025e-06 |
| householder_qr_square_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.300 | 0.0004213 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-direct | 8 | 15 | 1.987 | 0.0003161 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-direct | 16 | 15 | 3.224 | 0.0004732 |
| householder_qr_square_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 3.538 | 3.886e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-direct | 4 | 15 | 2.026 | 4.441e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-direct | 4 | 15 | 1.980 | 4.441e-07 |
| householder_qr_square_c32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.692 | 0.0005631 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-direct | 8 | 15 | 2.356 | 0.0003724 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-direct | 8 | 15 | 1.834 | 0.0006002 |
| slogdet_phase_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 3.489 | 0.0004085 |
| householder_qr_square_c64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.482 | 4.996e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-direct | 4 | 15 | 2.790 | 5.551e-07 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-direct | 4 | 15 | 2.875 | 3.886e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 3.376 | 4.755e-07 |
| householder_qr_square_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.421 | 0.0009546 |
| householder_qr_tall_f32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.358 | 0.001435 |
| householder_qr_wide_f32_256x512 | tenferro-cuda-direct | 2 | 15 | 2.750 | 0.001037 |
| householder_qr_square_f64_256x256 | tenferro-cuda-direct | 1 | 15 | 3.136 | 1.098e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 3.544 | 9.992e-07 |
| householder_qr_wide_f64_256x512 | tenferro-cuda-direct | 1 | 15 | 5.138 | 1.649e-06 |
| householder_qr_square_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.453 | 0.001198 |
| householder_qr_tall_c32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.386 | 0.001185 |
| householder_qr_wide_c32_256x512 | tenferro-cuda-direct | 2 | 15 | 2.726 | 0.001329 |
| slogdet_phase_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.668 | 0.01211 |
| householder_qr_square_c64_256x256 | tenferro-cuda-direct | 1 | 15 | 5.339 | 1.208e-06 |
| householder_qr_tall_c64_512x256 | tenferro-cuda-direct | 1 | 15 | 5.855 | 1.11e-06 |
| householder_qr_wide_c64_256x512 | tenferro-cuda-direct | 1 | 15 | 9.792 | 1.206e-06 |
| slogdet_phase_c64_256x256 | tenferro-cuda-direct | 2 | 15 | 3.312 | 1.025e-06 |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-direct | 8 | 15 | 3.958 | 0.0002423 |
| householder_append_default_f32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f32_64x32 | tenferro-cuda-direct | 128 | 15 | 2.714 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-direct | 4 | 15 | 3.056 | 3.886e-07 |
| householder_append_default_f64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f64_64x32 | tenferro-cuda-direct | 128 | 15 | 4.053 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-direct | 4 | 15 | 2.328 | 0.0003232 |
| householder_append_default_c32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c32_64x32 | tenferro-cuda-direct | 64 | 15 | 2.041 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-direct | 2 | 15 | 2.154 | 4.441e-07 |
| householder_append_default_c64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c64_64x32 | tenferro-cuda-direct | 128 | 15 | 3.899 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.863 | 0.0008522 |
| householder_append_default_f32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f32_512x256 | tenferro-cuda-direct | 128 | 15 | 4.346 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 4.165 | 8.882e-07 |
| householder_append_default_f64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | 128 | 15 | 3.921 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_512x256 | tenferro-cuda-direct | 2 | 15 | 3.116 | 0.001103 |
| householder_append_default_c32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c32_512x256 | tenferro-cuda-direct | 128 | 15 | 3.715 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c64_512x256 | tenferro-cuda-direct | 1 | 15 | 7.315 | 9.992e-07 |
| householder_append_default_c64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c64_512x256 | tenferro-cuda-direct | 128 | 15 | 4.040 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| cholesky_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 3.299 | 0.0002352 |
| cholesky_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.008 | 2.188e-07 |
| svdvals_f32_32x32 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_32x32 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| eigvalsh_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.519 | 0.001073 |
| eigvalsh_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 2.774 | 5.433e-07 |
| pinv_f32_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f32_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| det_f32_32x32 | tenferro-cuda-eager | 4 | 15 | 3.316 | 0.002641 |
| det_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 3.351 | 1.135e-06 |
| norm_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 3.279 | 0 |
| norm_f64_32x32 | tenferro-cuda-eager | 16 | 15 | 3.419 | 0 |
| triangular_solve_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.456 | 0.0001546 |
| triangular_solve_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.183 | 1.609e-07 |
| cholesky_f32_512x512 | tenferro-cuda-eager | 8 | 15 | 3.882 | 0.0003554 |
| cholesky_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 2.502 | 4.418e-07 |
| svdvals_f32_512x512 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_512x512 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| eigvalsh_f32_512x512 | tenferro-cuda-eager | 1 | 15 | 4.396 | 0.008342 |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 14.851 | 1.55e-06 |
| pinv_f32_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f32_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| det_f32_512x512 | tenferro-cuda-eager | 2 | 15 | 3.921 | 0.008577 |
| det_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 3.166 | 1.426e-05 |
| norm_f32_512x512 | tenferro-cuda-eager | 16 | 15 | 2.265 | 0 |
| norm_f64_512x512 | tenferro-cuda-eager | 16 | 15 | 2.213 | 1.909e-07 |
| triangular_solve_f32_512x512 | tenferro-cuda-eager | 16 | 15 | 3.620 | 0.0007877 |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | 16 | 15 | 3.616 | 1.157e-06 |
| svd_full_square_f32_32x32 | tenferro-cuda-eager | 4 | 15 | 3.151 | 0.008731 |
| svd_full_tall_f32_64x32 | tenferro-cuda-eager | 2 | 15 | 2.072 | 0.004865 |
| svd_full_wide_f32_32x64 | tenferro-cuda-eager | 2 | 15 | 2.297 | 0.005495 |
| slogdet_positive_f32_32x32 | tenferro-cuda-eager | 4 | 15 | 2.255 | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-eager | 4 | 15 | 2.249 | 0.000915 |
| svd_full_square_f64_32x32 | tenferro-cuda-eager | 1 | 15 | 5.245 | 1.007e-05 |
| svd_full_tall_f64_64x32 | tenferro-cuda-eager | 1 | 15 | 6.101 | 8.327e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-eager | 1 | 15 | 6.192 | 1.197e-05 |
| slogdet_positive_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 1.975 | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 1.992 | 4.261e-07 |
| svd_full_square_c32_32x32 | tenferro-cuda-eager | 4 | 15 | 3.652 | 0.009079 |
| svd_full_tall_c32_64x32 | tenferro-cuda-eager | 2 | 15 | 2.305 | 0.009498 |
| svd_full_wide_c32_32x64 | tenferro-cuda-eager | 2 | 15 | 2.376 | 0.008495 |
| slogdet_positive_c32_32x32 | tenferro-cuda-eager | 4 | 15 | 3.299 | 0.0007148 |
| slogdet_negative_c32_32x32 | tenferro-cuda-eager | 4 | 15 | 3.143 | 0.0007148 |
| svd_full_square_c64_32x32 | tenferro-cuda-eager | 1 | 15 | 14.496 | 9.215e-06 |
| svd_full_tall_c64_64x32 | tenferro-cuda-eager | 1 | 15 | 15.461 | 1.221e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-eager | 1 | 15 | 16.171 | 1.31e-05 |
| slogdet_positive_c64_32x32 | tenferro-cuda-eager | 4 | 15 | 3.441 | 3.804e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-eager | 4 | 15 | 3.275 | 3.804e-07 |
| svd_full_square_f32_256x256 | tenferro-cuda-eager | 1 | 15 | 8.657 | 0.1152 |
| svd_full_tall_f32_512x256 | tenferro-cuda-eager | 1 | 15 | 9.164 | 0.1022 |
| svd_full_wide_f32_256x512 | tenferro-cuda-eager | 1 | 15 | 8.923 | 0.1056 |
| slogdet_positive_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.490 | 0.0005003 |
| slogdet_negative_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.492 | 0.0005003 |
| svd_full_square_f64_256x256 | tenferro-cuda-eager | 1 | 15 | 162.069 | 8.171e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 129.122 | 7.76e-05 |
| svd_full_wide_f64_256x512 | tenferro-cuda-eager | 1 | 15 | 119.479 | 7.938e-05 |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | 2 | 15 | 2.408 | 1.864e-06 |
| slogdet_negative_f64_256x256 | tenferro-cuda-eager | 2 | 15 | 2.418 | 1.864e-06 |
| svd_full_square_c32_256x256 | tenferro-cuda-eager | 1 | 15 | 13.247 | 0.133 |
| svd_full_tall_c32_512x256 | tenferro-cuda-eager | 1 | 15 | 13.256 | 0.1126 |
| svd_full_wide_c32_256x512 | tenferro-cuda-eager | 1 | 15 | 13.328 | 0.1103 |
| slogdet_positive_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.956 | 0.01211 |
| slogdet_negative_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.703 | 0.01211 |
| svd_full_square_c64_256x256 | tenferro-cuda-eager | 1 | 15 | 517.130 | 0.000204 |
| svd_full_tall_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 369.855 | 0.0002024 |
| svd_full_wide_c64_256x512 | tenferro-cuda-eager | 1 | 15 | 371.234 | 0.0001941 |
| slogdet_positive_c64_256x256 | tenferro-cuda-eager | 2 | 15 | 3.816 | 1.025e-06 |
| slogdet_negative_c64_256x256 | tenferro-cuda-eager | 2 | 15 | 3.591 | 1.025e-06 |
| householder_qr_square_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.829 | 0.0004213 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-eager | 8 | 15 | 2.925 | 0.0003161 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-eager | 8 | 15 | 2.410 | 0.0004732 |
| householder_qr_square_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 2.378 | 3.886e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-eager | 4 | 15 | 2.666 | 4.441e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-eager | 4 | 15 | 2.511 | 4.441e-07 |
| householder_qr_square_c32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.918 | 0.0005631 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-eager | 8 | 15 | 3.268 | 0.0003724 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-eager | 8 | 15 | 2.580 | 0.0006002 |
| slogdet_phase_c32_32x32 | tenferro-cuda-eager | 4 | 15 | 2.213 | 0.0004085 |
| householder_qr_square_c64_32x32 | tenferro-cuda-eager | 4 | 15 | 3.058 | 4.996e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-eager | 4 | 15 | 3.359 | 5.551e-07 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-eager | 4 | 15 | 3.340 | 3.886e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-eager | 4 | 15 | 2.233 | 4.755e-07 |
| householder_qr_square_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.683 | 0.0009546 |
| householder_qr_tall_f32_512x256 | tenferro-cuda-eager | 2 | 15 | 2.320 | 0.001435 |
| householder_qr_wide_f32_256x512 | tenferro-cuda-eager | 2 | 15 | 2.708 | 0.001037 |
| householder_qr_square_f64_256x256 | tenferro-cuda-eager | 1 | 15 | 2.999 | 1.098e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 3.609 | 9.992e-07 |
| householder_qr_wide_f64_256x512 | tenferro-cuda-eager | 1 | 15 | 5.204 | 1.649e-06 |
| householder_qr_square_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.830 | 0.001198 |
| householder_qr_tall_c32_512x256 | tenferro-cuda-eager | 2 | 15 | 2.629 | 0.001185 |
| householder_qr_wide_c32_256x512 | tenferro-cuda-eager | 2 | 15 | 3.024 | 0.001329 |
| slogdet_phase_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.720 | 0.01211 |
| householder_qr_square_c64_256x256 | tenferro-cuda-eager | 1 | 15 | 5.473 | 1.208e-06 |
| householder_qr_tall_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 5.792 | 1.11e-06 |
| householder_qr_wide_c64_256x512 | tenferro-cuda-eager | 1 | 15 | 9.807 | 1.206e-06 |
| slogdet_phase_c64_256x256 | tenferro-cuda-eager | 2 | 15 | 3.753 | 1.025e-06 |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-eager | 4 | 15 | 3.664 | 0.0002423 |
| householder_append_default_f32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f32_64x32 | tenferro-cuda-eager | 64 | 15 | 2.498 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-eager | 2 | 15 | 2.297 | 3.886e-07 |
| householder_append_default_f64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f64_64x32 | tenferro-cuda-eager | 64 | 15 | 2.646 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-eager | 4 | 15 | 3.799 | 0.0003232 |
| householder_append_default_c32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c32_64x32 | tenferro-cuda-eager | 64 | 15 | 2.749 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-eager | 2 | 15 | 3.256 | 4.441e-07 |
| householder_append_default_c64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c64_64x32 | tenferro-cuda-eager | 32 | 15 | 2.077 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f32_512x256 | tenferro-cuda-eager | 2 | 15 | 3.239 | 0.0008522 |
| householder_append_default_f32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f32_512x256 | tenferro-cuda-eager | 128 | 15 | 3.258 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 4.191 | 8.882e-07 |
| householder_append_default_f64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | 128 | 15 | 3.662 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_512x256 | tenferro-cuda-eager | 2 | 15 | 3.320 | 0.001103 |
| householder_append_default_c32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c32_512x256 | tenferro-cuda-eager | 128 | 15 | 3.918 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 7.340 | 9.992e-07 |
| householder_append_default_c64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c64_512x256 | tenferro-cuda-eager | 64 | 15 | 2.126 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| cholesky_f32_32x32 | pytorch-cuda | 32 | 15 | 2.887 | 0.0002352 |
| cholesky_f64_32x32 | pytorch-cuda | 16 | 15 | 2.575 | 2.188e-07 |
| svdvals_f32_32x32 | pytorch-cuda | 8 | 15 | 3.248 | 0.007749 |
| svdvals_f64_32x32 | pytorch-cuda | 1 | 15 | 5.090 | 3.387e-06 |
| eigvalsh_f32_32x32 | pytorch-cuda | 16 | 15 | 3.542 | 0.001073 |
| eigvalsh_f64_32x32 | pytorch-cuda | 4 | 15 | 2.448 | 5.433e-07 |
| pinv_f32_32x32 | pytorch-cuda | 4 | 15 | 2.151 | 0.001023 |
| pinv_f64_32x32 | pytorch-cuda | 1 | 15 | 4.937 | 9.374e-07 |
| inv_f32_32x32 | pytorch-cuda | 8 | 15 | 2.350 | 0.0004905 |
| inv_f64_32x32 | pytorch-cuda | 8 | 15 | 2.412 | 2.284e-07 |
| det_f32_32x32 | pytorch-cuda | 16 | 15 | 2.905 | 0.002641 |
| det_f64_32x32 | pytorch-cuda | 16 | 15 | 2.923 | 3.784e-07 |
| norm_f32_32x32 | pytorch-cuda | 128 | 15 | 1.902 | 0 |
| norm_f64_32x32 | pytorch-cuda | 128 | 15 | 1.916 | 0 |
| triangular_solve_f32_32x32 | pytorch-cuda | 64 | 15 | 3.438 | 9.242e-05 |
| triangular_solve_f64_32x32 | pytorch-cuda | 64 | 15 | 3.468 | 1.415e-07 |
| cholesky_f32_512x512 | pytorch-cuda | 8 | 15 | 2.241 | 0.0003554 |
| cholesky_f64_512x512 | pytorch-cuda | 1 | 15 | 2.200 | 4.418e-07 |
| svdvals_f32_512x512 | pytorch-cuda | 1 | 15 | 28.150 | 0.1733 |
| svdvals_f64_512x512 | pytorch-cuda | 1 | 15 | 595.682 | 0.0002066 |
| eigvalsh_f32_512x512 | pytorch-cuda | 1 | 15 | 4.374 | 0.008342 |
| eigvalsh_f64_512x512 | pytorch-cuda | 1 | 15 | 14.728 | 1.55e-06 |
| pinv_f32_512x512 | pytorch-cuda | 1 | 15 | 31.671 | 0.01126 |
| pinv_f64_512x512 | pytorch-cuda | 1 | 15 | 681.259 | 8.747e-06 |
| inv_f32_512x512 | pytorch-cuda | 2 | 15 | 3.903 | 0.002107 |
| inv_f64_512x512 | pytorch-cuda | 1 | 15 | 5.996 | 1.85e-06 |
| det_f32_512x512 | pytorch-cuda | 2 | 15 | 2.893 | 0.007965 |
| det_f64_512x512 | pytorch-cuda | 1 | 15 | 2.507 | 1.369e-05 |
| norm_f32_512x512 | pytorch-cuda | 128 | 15 | 2.591 | 0 |
| norm_f64_512x512 | pytorch-cuda | 128 | 15 | 2.667 | 1.909e-07 |
| triangular_solve_f32_512x512 | pytorch-cuda | 32 | 15 | 3.317 | 0.0006008 |
| triangular_solve_f64_512x512 | pytorch-cuda | 4 | 15 | 3.379 | 6.887e-07 |
| svd_full_square_f32_32x32 | pytorch-cuda | 8 | 15 | 2.767 | 0.005571 |
| svd_full_tall_f32_64x32 | pytorch-cuda | 4 | 15 | 3.389 | 0.005287 |
| svd_full_wide_f32_32x64 | pytorch-cuda | 4 | 15 | 3.726 | 0.005456 |
| slogdet_positive_f32_32x32 | pytorch-cuda | 16 | 15 | 3.416 | 0.000915 |
| slogdet_negative_f32_32x32 | pytorch-cuda | 16 | 15 | 3.405 | 0.000915 |
| svd_full_square_f64_32x32 | pytorch-cuda | 1 | 15 | 4.282 | 7.216e-06 |
| svd_full_tall_f64_64x32 | pytorch-cuda | 1 | 15 | 5.972 | 8.826e-06 |
| svd_full_wide_f64_32x64 | pytorch-cuda | 1 | 15 | 6.318 | 8.438e-06 |
| slogdet_positive_f64_32x32 | pytorch-cuda | 16 | 15 | 3.400 | 4.261e-07 |
| slogdet_negative_f64_32x32 | pytorch-cuda | 16 | 15 | 3.385 | 4.261e-07 |
| svd_full_square_c32_32x32 | pytorch-cuda | 8 | 15 | 3.816 | 0.011 |
| svd_full_tall_c32_64x32 | pytorch-cuda | 2 | 15 | 2.103 | 0.008819 |
| svd_full_wide_c32_32x64 | pytorch-cuda | 2 | 15 | 2.114 | 0.008905 |
| slogdet_positive_c32_32x32 | pytorch-cuda | 16 | 15 | 3.593 | 0.0004085 |
| slogdet_negative_c32_32x32 | pytorch-cuda | 16 | 15 | 3.595 | 0.0004085 |
| svd_full_square_c64_32x32 | pytorch-cuda | 1 | 15 | 11.401 | 9.992e-06 |
| svd_full_tall_c64_64x32 | pytorch-cuda | 1 | 15 | 15.452 | 1.188e-05 |
| svd_full_wide_c64_32x64 | pytorch-cuda | 1 | 15 | 16.193 | 1.299e-05 |
| slogdet_positive_c64_32x32 | pytorch-cuda | 8 | 15 | 1.834 | 4.755e-07 |
| slogdet_negative_c64_32x32 | pytorch-cuda | 8 | 15 | 1.834 | 4.755e-07 |
| svd_full_square_f32_256x256 | pytorch-cuda | 1 | 15 | 9.086 | 0.1135 |
| svd_full_tall_f32_512x256 | pytorch-cuda | 1 | 15 | 9.569 | 0.1033 |
| svd_full_wide_f32_256x512 | pytorch-cuda | 1 | 15 | 8.959 | 0.106 |
| slogdet_positive_f32_256x256 | pytorch-cuda | 8 | 15 | 2.952 | 0.0005003 |
| slogdet_negative_f32_256x256 | pytorch-cuda | 8 | 15 | 2.980 | 0.0005003 |
| svd_full_square_f64_256x256 | pytorch-cuda | 1 | 15 | 165.544 | 8.105e-05 |
| svd_full_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 131.051 | 7.76e-05 |
| svd_full_wide_f64_256x512 | pytorch-cuda | 1 | 15 | 121.551 | 7.938e-05 |
| slogdet_positive_f64_256x256 | pytorch-cuda | 4 | 15 | 2.881 | 1.708e-06 |
| slogdet_negative_f64_256x256 | pytorch-cuda | 4 | 15 | 2.891 | 1.708e-06 |
| svd_full_square_c32_256x256 | pytorch-cuda | 1 | 15 | 14.280 | 0.1305 |
| svd_full_tall_c32_512x256 | pytorch-cuda | 1 | 15 | 13.333 | 0.1089 |
| svd_full_wide_c32_256x512 | pytorch-cuda | 1 | 15 | 13.513 | 0.108 |
| slogdet_positive_c32_256x256 | pytorch-cuda | 8 | 15 | 3.270 | 0.01211 |
| slogdet_negative_c32_256x256 | pytorch-cuda | 8 | 15 | 3.271 | 0.01211 |
| svd_full_square_c64_256x256 | pytorch-cuda | 1 | 15 | 544.850 | 0.0002064 |
| svd_full_tall_c64_512x256 | pytorch-cuda | 1 | 15 | 372.369 | 0.0001994 |
| svd_full_wide_c64_256x512 | pytorch-cuda | 1 | 15 | 375.665 | 0.0001943 |
| slogdet_positive_c64_256x256 | pytorch-cuda | 2 | 15 | 2.796 | 1.23e-06 |
| slogdet_negative_c64_256x256 | pytorch-cuda | 2 | 15 | 2.794 | 1.23e-06 |
| householder_qr_square_f32_32x32 | pytorch-cuda | 64 | 15 | 3.842 | 0.000367 |
| householder_qr_tall_f32_64x32 | pytorch-cuda | 32 | 15 | 3.396 | 0.0003269 |
| householder_qr_wide_f32_32x64 | pytorch-cuda | 32 | 15 | 3.434 | 0.0003745 |
| householder_qr_square_f64_32x32 | pytorch-cuda | 8 | 15 | 2.301 | 6.106e-07 |
| householder_qr_tall_f64_64x32 | pytorch-cuda | 8 | 15 | 2.854 | 6.661e-07 |
| householder_qr_wide_f64_32x64 | pytorch-cuda | 8 | 15 | 2.890 | 5.551e-07 |
| householder_qr_square_c32_32x32 | pytorch-cuda | 32 | 15 | 2.179 | 0.0005833 |
| householder_qr_tall_c32_64x32 | pytorch-cuda | 32 | 15 | 3.634 | 0.0003756 |
| householder_qr_wide_c32_32x64 | pytorch-cuda | 32 | 15 | 4.072 | 0.0004972 |
| slogdet_phase_c32_32x32 | pytorch-cuda | 8 | 15 | 2.079 | 0.0004085 |
| householder_qr_square_c64_32x32 | pytorch-cuda | 8 | 15 | 3.859 | 1.221e-06 |
| householder_qr_tall_c64_64x32 | pytorch-cuda | 4 | 15 | 2.315 | 9.992e-07 |
| householder_qr_wide_c64_32x64 | pytorch-cuda | 4 | 15 | 2.492 | 1.166e-06 |
| slogdet_phase_c64_32x32 | pytorch-cuda | 16 | 15 | 3.684 | 5.706e-07 |
| householder_qr_square_f32_256x256 | pytorch-cuda | 4 | 15 | 3.040 | 0.001043 |
| householder_qr_tall_f32_512x256 | pytorch-cuda | 2 | 15 | 2.103 | 0.001475 |
| householder_qr_wide_f32_256x512 | pytorch-cuda | 2 | 15 | 2.487 | 0.0009736 |
| householder_qr_square_f64_256x256 | pytorch-cuda | 1 | 15 | 2.931 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 3.326 | 1.887e-06 |
| householder_qr_wide_f64_256x512 | pytorch-cuda | 1 | 15 | 4.822 | 1.759e-06 |
| householder_qr_square_c32_256x256 | pytorch-cuda | 4 | 15 | 3.173 | 0.001258 |
| householder_qr_tall_c32_512x256 | pytorch-cuda | 2 | 15 | 2.278 | 0.001161 |
| householder_qr_wide_c32_256x512 | pytorch-cuda | 2 | 15 | 2.613 | 0.001226 |
| slogdet_phase_c32_256x256 | pytorch-cuda | 8 | 15 | 3.270 | 0.01233 |
| householder_qr_square_c64_256x256 | pytorch-cuda | 1 | 15 | 5.075 | 2.109e-06 |
| householder_qr_tall_c64_512x256 | pytorch-cuda | 1 | 15 | 5.701 | 2.942e-06 |
| householder_qr_wide_c64_256x512 | pytorch-cuda | 1 | 15 | 9.708 | 2.276e-06 |
| slogdet_phase_c64_256x256 | pytorch-cuda | 2 | 15 | 2.793 | 1.025e-06 |
| householder_from_factors_default_f32_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_64x32 | pytorch-cuda | 256 | 15 | 2.259 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | pytorch-cuda | 32 | 15 | 3.201 | 0.0001196 |
| householder_q_columns_complement_f32_64x32 | pytorch-cuda | 16 | 15 | 2.933 | 0.0001791 |
| householder_from_factors_default_f64_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_64x32 | pytorch-cuda | 256 | 15 | 2.208 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | pytorch-cuda | 32 | 15 | 3.450 | 1.114e-07 |
| householder_q_columns_complement_f64_64x32 | pytorch-cuda | 16 | 15 | 3.700 | 2.724e-08 |
| householder_from_factors_default_c32_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_64x32 | pytorch-cuda | 256 | 15 | 2.504 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | pytorch-cuda | 16 | 15 | 2.053 | 0.00012 |
| householder_q_columns_complement_c32_64x32 | pytorch-cuda | 16 | 15 | 2.913 | 0.000299 |
| householder_from_factors_default_c64_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_64x32 | pytorch-cuda | 256 | 15 | 2.242 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | pytorch-cuda | 32 | 15 | 3.485 | 1.676e-07 |
| householder_q_columns_complement_c64_64x32 | pytorch-cuda | 8 | 15 | 3.502 | 3.341e-07 |
| householder_from_factors_default_f32_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_512x256 | pytorch-cuda | 128 | 15 | 1.144 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | pytorch-cuda | 32 | 15 | 3.235 | 0.0001829 |
| householder_q_columns_complement_f32_512x256 | pytorch-cuda | 16 | 15 | 3.229 | 8.761e-05 |
| householder_from_factors_default_f64_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_512x256 | pytorch-cuda | 128 | 15 | 1.204 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | 16 | 15 | 3.097 | 1.136e-07 |
| householder_q_columns_complement_f64_512x256 | pytorch-cuda | 2 | 15 | 3.100 | 7.26e-07 |
| householder_from_factors_default_c32_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_512x256 | pytorch-cuda | 256 | 15 | 2.281 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | pytorch-cuda | 16 | 15 | 1.980 | 0.0001246 |
| householder_q_columns_complement_c32_512x256 | pytorch-cuda | 8 | 15 | 2.061 | 0.0001149 |
| householder_from_factors_default_c64_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_512x256 | pytorch-cuda | 256 | 15 | 2.322 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | pytorch-cuda | 8 | 15 | 3.014 | 1.154e-07 |
| householder_q_columns_complement_c64_512x256 | pytorch-cuda | 1 | 15 | 4.960 | 7.282e-07 |
| cholesky_f32_32x32 | jax-cuda | 16 | 15 | 2.420 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda | 16 | 15 | 2.704 | 2.188e-07 |
| svdvals_f32_32x32 | jax-cuda | 4 | 15 | 3.073 | 0.003338 |
| svdvals_f64_32x32 | jax-cuda | 1 | 15 | 5.825 | 4.546e-06 |
| eigvalsh_f32_32x32 | jax-cuda | 4 | 15 | 2.694 | 0.004775 |
| eigvalsh_f64_32x32 | jax-cuda | 1 | 15 | 3.141 | 5.542e-06 |
| pinv_f32_32x32 | jax-cuda | 2 | 15 | 2.535 | 0.0006436 |
| pinv_f64_32x32 | jax-cuda | 1 | 15 | 5.826 | 1.367e-06 |
| inv_f32_32x32 | jax-cuda | 16 | 15 | 3.122 | 0.0004905 |
| inv_f64_32x32 | jax-cuda | 16 | 15 | 3.444 | 2.284e-07 |
| det_f32_32x32 | jax-cuda | 16 | 15 | 2.566 | 0.002641 |
| det_f64_32x32 | jax-cuda | 16 | 15 | 3.308 | 1.135e-06 |
| norm_f32_32x32 | jax-cuda | 16 | 15 | 2.113 | 0 |
| norm_f64_32x32 | jax-cuda | 16 | 15 | 2.561 | 0 |
| triangular_solve_f32_32x32 | jax-cuda | 16 | 15 | 3.042 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda | 16 | 15 | 3.737 | 1.609e-07 |
| cholesky_f32_512x512 | jax-cuda | 8 | 15 | 2.166 | 0.0003554 |
| cholesky_f64_512x512 | jax-cuda | 1 | 15 | 2.768 | 4.418e-07 |
| svdvals_f32_512x512 | jax-cuda | 1 | 15 | 26.037 | 0.1547 |
| svdvals_f64_512x512 | jax-cuda | 1 | 15 | 541.502 | 0.0002063 |
| eigvalsh_f32_512x512 | jax-cuda | 1 | 15 | 5.041 | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda | 1 | 15 | 19.562 | 1.55e-06 |
| pinv_f32_512x512 | jax-cuda | 1 | 15 | 29.343 | 0.009647 |
| pinv_f64_512x512 | jax-cuda | 1 | 15 | 615.665 | 8.975e-06 |
| inv_f32_512x512 | jax-cuda | 1 | 15 | 2.300 | 0.002107 |
| inv_f64_512x512 | jax-cuda | 1 | 15 | 6.395 | 1.85e-06 |
| det_f32_512x512 | jax-cuda | 2 | 15 | 3.272 | 0.004748 |
| det_f64_512x512 | jax-cuda | 1 | 15 | 2.906 | 1.426e-05 |
| norm_f32_512x512 | jax-cuda | 16 | 15 | 2.512 | 0 |
| norm_f64_512x512 | jax-cuda | 16 | 15 | 2.472 | 1.909e-07 |
| triangular_solve_f32_512x512 | jax-cuda | 16 | 15 | 3.773 | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda | 16 | 15 | 3.573 | 1.157e-06 |
| svd_full_square_f32_32x32 | jax-cuda | 4 | 15 | 3.025 | 0.008731 |
| svd_full_tall_f32_64x32 | jax-cuda | 2 | 15 | 2.761 | 0.004865 |
| svd_full_wide_f32_32x64 | jax-cuda | 2 | 15 | 2.978 | 0.005495 |
| slogdet_positive_f32_32x32 | jax-cuda | 16 | 15 | 2.674 | 0.000915 |
| slogdet_negative_f32_32x32 | jax-cuda | 16 | 15 | 2.689 | 0.000915 |
| svd_full_square_f64_32x32 | jax-cuda | 1 | 15 | 5.651 | 1.008e-05 |
| svd_full_tall_f64_64x32 | jax-cuda | 1 | 15 | 6.051 | 8.327e-06 |
| svd_full_wide_f64_32x64 | jax-cuda | 1 | 15 | 6.215 | 1.197e-05 |
| slogdet_positive_f64_32x32 | jax-cuda | 8 | 15 | 2.250 | 4.261e-07 |
| slogdet_negative_f64_32x32 | jax-cuda | 8 | 15 | 1.985 | 4.261e-07 |
| svd_full_square_c32_32x32 | jax-cuda | 2 | 15 | 2.099 | 0.009079 |
| svd_full_tall_c32_64x32 | jax-cuda | 2 | 15 | 2.889 | 0.009498 |
| svd_full_wide_c32_32x64 | jax-cuda | 2 | 15 | 3.118 | 0.008495 |
| slogdet_positive_c32_32x32 | jax-cuda | 8 | 15 | 2.264 | 0.0004085 |
| slogdet_negative_c32_32x32 | jax-cuda | 8 | 15 | 1.881 | 0.0004085 |
| svd_full_square_c64_32x32 | jax-cuda | 1 | 15 | 14.395 | 9.215e-06 |
| svd_full_tall_c64_64x32 | jax-cuda | 1 | 15 | 15.287 | 1.243e-05 |
| svd_full_wide_c64_32x64 | jax-cuda | 1 | 15 | 15.973 | 1.288e-05 |
| slogdet_positive_c64_32x32 | jax-cuda | 16 | 15 | 2.186 | 4.755e-07 |
| slogdet_negative_c64_32x32 | jax-cuda | 16 | 15 | 2.152 | 4.755e-07 |
| svd_full_square_f32_256x256 | jax-cuda | 1 | 15 | 8.631 | 0.1152 |
| svd_full_tall_f32_512x256 | jax-cuda | 1 | 15 | 9.115 | 0.1022 |
| svd_full_wide_f32_256x512 | jax-cuda | 1 | 15 | 8.876 | 0.1056 |
| slogdet_positive_f32_256x256 | jax-cuda | 8 | 15 | 3.021 | 0.0005003 |
| slogdet_negative_f32_256x256 | jax-cuda | 8 | 15 | 3.071 | 0.0005003 |
| svd_full_square_f64_256x256 | jax-cuda | 1 | 15 | 160.311 | 8.171e-05 |
| svd_full_tall_f64_512x256 | jax-cuda | 1 | 15 | 127.848 | 7.76e-05 |
| svd_full_wide_f64_256x512 | jax-cuda | 1 | 15 | 118.360 | 7.938e-05 |
| slogdet_positive_f64_256x256 | jax-cuda | 4 | 15 | 3.214 | 1.864e-06 |
| slogdet_negative_f64_256x256 | jax-cuda | 4 | 15 | 3.242 | 1.864e-06 |
| svd_full_square_c32_256x256 | jax-cuda | 1 | 15 | 13.234 | 0.133 |
| svd_full_tall_c32_512x256 | jax-cuda | 1 | 15 | 13.248 | 0.1126 |
| svd_full_wide_c32_256x512 | jax-cuda | 1 | 15 | 13.285 | 0.1103 |
| slogdet_positive_c32_256x256 | jax-cuda | 8 | 15 | 3.397 | 0.01211 |
| slogdet_negative_c32_256x256 | jax-cuda | 8 | 15 | 3.447 | 0.01211 |
| svd_full_square_c64_256x256 | jax-cuda | 1 | 15 | 511.322 | 0.000204 |
| svd_full_tall_c64_512x256 | jax-cuda | 1 | 15 | 365.876 | 0.0002021 |
| svd_full_wide_c64_256x512 | jax-cuda | 1 | 15 | 366.936 | 0.0001946 |
| slogdet_positive_c64_256x256 | jax-cuda | 2 | 15 | 3.152 | 1.025e-06 |
| slogdet_negative_c64_256x256 | jax-cuda | 2 | 15 | 3.177 | 1.025e-06 |
| householder_qr_square_f32_32x32 | jax-cuda | 16 | 15 | 2.035 | 0.000367 |
| householder_qr_tall_f32_64x32 | jax-cuda | 8 | 15 | 2.191 | 0.0003269 |
| householder_qr_wide_f32_32x64 | jax-cuda | 8 | 15 | 2.903 | 0.0003745 |
| householder_qr_square_f64_32x32 | jax-cuda | 8 | 15 | 2.531 | 6.106e-07 |
| householder_qr_tall_f64_64x32 | jax-cuda | 8 | 15 | 3.134 | 6.661e-07 |
| householder_qr_wide_f64_32x64 | jax-cuda | 8 | 15 | 3.117 | 5.551e-07 |
| householder_qr_square_c32_32x32 | jax-cuda | 16 | 15 | 3.201 | 0.0005833 |
| householder_qr_tall_c32_64x32 | jax-cuda | 8 | 15 | 2.925 | 0.0003756 |
| householder_qr_wide_c32_32x64 | jax-cuda | 8 | 15 | 2.916 | 0.0004972 |
| slogdet_phase_c32_32x32 | jax-cuda | 8 | 15 | 1.882 | 0.0004085 |
| householder_qr_square_c64_32x32 | jax-cuda | 4 | 15 | 2.349 | 1.221e-06 |
| householder_qr_tall_c64_64x32 | jax-cuda | 4 | 15 | 2.764 | 9.992e-07 |
| householder_qr_wide_c64_32x64 | jax-cuda | 4 | 15 | 3.106 | 1.166e-06 |
| slogdet_phase_c64_32x32 | jax-cuda | 8 | 15 | 2.293 | 5.706e-07 |
| householder_qr_square_f32_256x256 | jax-cuda | 2 | 15 | 1.958 | 0.001043 |
| householder_qr_tall_f32_512x256 | jax-cuda | 2 | 15 | 2.612 | 0.001475 |
| householder_qr_wide_f32_256x512 | jax-cuda | 2 | 15 | 3.007 | 0.0009736 |
| householder_qr_square_f64_256x256 | jax-cuda | 1 | 15 | 3.521 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | jax-cuda | 1 | 15 | 3.884 | 1.887e-06 |
| householder_qr_wide_f64_256x512 | jax-cuda | 1 | 15 | 5.565 | 1.759e-06 |
| householder_qr_square_c32_256x256 | jax-cuda | 2 | 15 | 2.142 | 0.001258 |
| householder_qr_tall_c32_512x256 | jax-cuda | 2 | 15 | 2.756 | 0.001161 |
| householder_qr_wide_c32_256x512 | jax-cuda | 2 | 15 | 3.134 | 0.001226 |
| slogdet_phase_c32_256x256 | jax-cuda | 8 | 15 | 3.579 | 0.01233 |
| householder_qr_square_c64_256x256 | jax-cuda | 1 | 15 | 5.835 | 2.109e-06 |
| householder_qr_tall_c64_512x256 | jax-cuda | 1 | 15 | 6.431 | 2.942e-06 |
| householder_qr_wide_c64_256x512 | jax-cuda | 1 | 15 | 10.066 | 2.276e-06 |
| slogdet_phase_c64_256x256 | jax-cuda | 2 | 15 | 3.157 | 1.025e-06 |
| householder_from_factors_default_f32_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_64x32 | jax-cuda | 16 | 15 | 2.623 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | jax-cuda | 8 | 15 | 3.264 | 0.0001196 |
| householder_q_columns_complement_f32_64x32 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f64_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_64x32 | jax-cuda | 16 | 15 | 1.981 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | jax-cuda | 8 | 15 | 3.274 | 1.114e-07 |
| householder_q_columns_complement_f64_64x32 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c32_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_64x32 | jax-cuda | 16 | 15 | 2.627 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | jax-cuda | 8 | 15 | 3.347 | 0.00012 |
| householder_q_columns_complement_c32_64x32 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c64_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_64x32 | jax-cuda | 16 | 15 | 2.562 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | jax-cuda | 8 | 15 | 3.643 | 1.676e-07 |
| householder_q_columns_complement_c64_64x32 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f32_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_512x256 | jax-cuda | 16 | 15 | 1.872 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | jax-cuda | 8 | 15 | 3.040 | 0.0001829 |
| householder_q_columns_complement_f32_512x256 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f64_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_512x256 | jax-cuda | 16 | 15 | 2.495 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | 8 | 15 | 3.791 | 1.136e-07 |
| householder_q_columns_complement_f64_512x256 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c32_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_512x256 | jax-cuda | 16 | 15 | 2.580 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | jax-cuda | 8 | 15 | 3.718 | 0.0001246 |
| householder_q_columns_complement_c32_512x256 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c64_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_512x256 | jax-cuda | 16 | 15 | 2.640 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | jax-cuda | 4 | 15 | 2.337 | 1.154e-07 |
| householder_q_columns_complement_c64_512x256 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| cholesky_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.825 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda-jit | 16 | 15 | 3.224 | 2.188e-07 |
| svdvals_f32_32x32 | jax-cuda-jit | 4 | 15 | 3.550 | 0.003338 |
| svdvals_f64_32x32 | jax-cuda-jit | 1 | 15 | 5.838 | 4.546e-06 |
| eigvalsh_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.798 | 0.004775 |
| eigvalsh_f64_32x32 | jax-cuda-jit | 1 | 15 | 3.105 | 5.542e-06 |
| pinv_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.934 | 0.0006436 |
| pinv_f64_32x32 | jax-cuda-jit | 1 | 15 | 5.708 | 1.367e-06 |
| inv_f32_32x32 | jax-cuda-jit | 8 | 15 | 2.350 | 0.0004905 |
| inv_f64_32x32 | jax-cuda-jit | 16 | 15 | 3.412 | 2.284e-07 |
| det_f32_32x32 | jax-cuda-jit | 8 | 15 | 2.089 | 0.002641 |
| det_f64_32x32 | jax-cuda-jit | 16 | 15 | 3.715 | 1.135e-06 |
| norm_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.642 | 0 |
| norm_f64_32x32 | jax-cuda-jit | 16 | 15 | 2.700 | 0 |
| triangular_solve_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.915 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda-jit | 16 | 15 | 3.371 | 1.609e-07 |
| cholesky_f32_512x512 | jax-cuda-jit | 8 | 15 | 2.177 | 0.0003554 |
| cholesky_f64_512x512 | jax-cuda-jit | 1 | 15 | 2.751 | 4.418e-07 |
| svdvals_f32_512x512 | jax-cuda-jit | 1 | 15 | 25.871 | 0.1547 |
| svdvals_f64_512x512 | jax-cuda-jit | 1 | 15 | 537.881 | 0.0002063 |
| eigvalsh_f32_512x512 | jax-cuda-jit | 1 | 15 | 4.998 | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda-jit | 1 | 15 | 19.539 | 1.55e-06 |
| pinv_f32_512x512 | jax-cuda-jit | 1 | 15 | 28.916 | 0.009647 |
| pinv_f64_512x512 | jax-cuda-jit | 1 | 15 | 615.495 | 8.975e-06 |
| inv_f32_512x512 | jax-cuda-jit | 1 | 15 | 2.279 | 0.002107 |
| inv_f64_512x512 | jax-cuda-jit | 1 | 15 | 6.412 | 1.85e-06 |
| det_f32_512x512 | jax-cuda-jit | 2 | 15 | 3.269 | 0.004748 |
| det_f64_512x512 | jax-cuda-jit | 1 | 15 | 2.883 | 1.426e-05 |
| norm_f32_512x512 | jax-cuda-jit | 16 | 15 | 2.512 | 0.000205 |
| norm_f64_512x512 | jax-cuda-jit | 16 | 15 | 2.640 | 1.909e-07 |
| triangular_solve_f32_512x512 | jax-cuda-jit | 16 | 15 | 2.736 | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda-jit | 16 | 15 | 3.863 | 1.157e-06 |
| svd_full_square_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.883 | 0.008731 |
| svd_full_tall_f32_64x32 | jax-cuda-jit | 2 | 15 | 2.835 | 0.004865 |
| svd_full_wide_f32_32x64 | jax-cuda-jit | 2 | 15 | 2.419 | 0.005495 |
| slogdet_positive_f32_32x32 | jax-cuda-jit | 8 | 15 | 2.174 | 0.000915 |
| slogdet_negative_f32_32x32 | jax-cuda-jit | 16 | 15 | 3.332 | 0.000915 |
| svd_full_square_f64_32x32 | jax-cuda-jit | 1 | 15 | 5.739 | 1.008e-05 |
| svd_full_tall_f64_64x32 | jax-cuda-jit | 1 | 15 | 6.050 | 8.327e-06 |
| svd_full_wide_f64_32x64 | jax-cuda-jit | 1 | 15 | 6.265 | 1.197e-05 |
| slogdet_positive_f64_32x32 | jax-cuda-jit | 8 | 15 | 2.053 | 4.261e-07 |
| slogdet_negative_f64_32x32 | jax-cuda-jit | 8 | 15 | 2.377 | 4.261e-07 |
| svd_full_square_c32_32x32 | jax-cuda-jit | 2 | 15 | 2.114 | 0.009079 |
| svd_full_tall_c32_64x32 | jax-cuda-jit | 2 | 15 | 3.098 | 0.009498 |
| svd_full_wide_c32_32x64 | jax-cuda-jit | 2 | 15 | 3.115 | 0.008495 |
| slogdet_positive_c32_32x32 | jax-cuda-jit | 8 | 15 | 2.194 | 0.0004085 |
| slogdet_negative_c32_32x32 | jax-cuda-jit | 8 | 15 | 2.360 | 0.0004085 |
| svd_full_square_c64_32x32 | jax-cuda-jit | 1 | 15 | 15.403 | 9.215e-06 |
| svd_full_tall_c64_64x32 | jax-cuda-jit | 1 | 15 | 15.297 | 1.243e-05 |
| svd_full_wide_c64_32x64 | jax-cuda-jit | 1 | 15 | 16.027 | 1.288e-05 |
| slogdet_positive_c64_32x32 | jax-cuda-jit | 16 | 15 | 3.146 | 4.755e-07 |
| slogdet_negative_c64_32x32 | jax-cuda-jit | 8 | 15 | 2.385 | 4.755e-07 |
| svd_full_square_f32_256x256 | jax-cuda-jit | 1 | 15 | 8.762 | 0.1152 |
| svd_full_tall_f32_512x256 | jax-cuda-jit | 1 | 15 | 9.209 | 0.1022 |
| svd_full_wide_f32_256x512 | jax-cuda-jit | 1 | 15 | 8.937 | 0.1056 |
| slogdet_positive_f32_256x256 | jax-cuda-jit | 8 | 15 | 3.047 | 0.0005003 |
| slogdet_negative_f32_256x256 | jax-cuda-jit | 8 | 15 | 3.128 | 0.0005003 |
| svd_full_square_f64_256x256 | jax-cuda-jit | 1 | 15 | 160.332 | 8.171e-05 |
| svd_full_tall_f64_512x256 | jax-cuda-jit | 1 | 15 | 127.870 | 7.76e-05 |
| svd_full_wide_f64_256x512 | jax-cuda-jit | 1 | 15 | 118.425 | 7.938e-05 |
| slogdet_positive_f64_256x256 | jax-cuda-jit | 4 | 15 | 3.221 | 1.864e-06 |
| slogdet_negative_f64_256x256 | jax-cuda-jit | 4 | 15 | 3.262 | 1.864e-06 |
| svd_full_square_c32_256x256 | jax-cuda-jit | 1 | 15 | 13.225 | 0.133 |
| svd_full_tall_c32_512x256 | jax-cuda-jit | 1 | 15 | 13.229 | 0.1126 |
| svd_full_wide_c32_256x512 | jax-cuda-jit | 1 | 15 | 13.264 | 0.1103 |
| slogdet_positive_c32_256x256 | jax-cuda-jit | 8 | 15 | 3.429 | 0.01211 |
| slogdet_negative_c32_256x256 | jax-cuda-jit | 8 | 15 | 3.466 | 0.01211 |
| svd_full_square_c64_256x256 | jax-cuda-jit | 1 | 15 | 511.536 | 0.000204 |
| svd_full_tall_c64_512x256 | jax-cuda-jit | 1 | 15 | 365.882 | 0.0002021 |
| svd_full_wide_c64_256x512 | jax-cuda-jit | 1 | 15 | 367.084 | 0.0001946 |
| slogdet_positive_c64_256x256 | jax-cuda-jit | 2 | 15 | 3.146 | 1.025e-06 |
| slogdet_negative_c64_256x256 | jax-cuda-jit | 2 | 15 | 3.142 | 1.025e-06 |
| householder_qr_square_f32_32x32 | jax-cuda-jit | 8 | 15 | 2.271 | 0.000367 |
| householder_qr_tall_f32_64x32 | jax-cuda-jit | 8 | 15 | 3.027 | 0.0003269 |
| householder_qr_wide_f32_32x64 | jax-cuda-jit | 8 | 15 | 2.996 | 0.0003745 |
| householder_qr_square_f64_32x32 | jax-cuda-jit | 8 | 15 | 2.588 | 6.106e-07 |
| householder_qr_tall_f64_64x32 | jax-cuda-jit | 8 | 15 | 3.346 | 6.661e-07 |
| householder_qr_wide_f64_32x64 | jax-cuda-jit | 4 | 15 | 2.167 | 5.551e-07 |
| householder_qr_square_c32_32x32 | jax-cuda-jit | 8 | 15 | 2.412 | 0.0005833 |
| householder_qr_tall_c32_64x32 | jax-cuda-jit | 8 | 15 | 3.210 | 0.0003756 |
| householder_qr_wide_c32_32x64 | jax-cuda-jit | 8 | 15 | 2.880 | 0.0004972 |
| slogdet_phase_c32_32x32 | jax-cuda-jit | 8 | 15 | 2.077 | 0.0004085 |
| householder_qr_square_c64_32x32 | jax-cuda-jit | 4 | 15 | 2.531 | 1.221e-06 |
| householder_qr_tall_c64_64x32 | jax-cuda-jit | 4 | 15 | 2.948 | 9.992e-07 |
| householder_qr_wide_c64_32x64 | jax-cuda-jit | 4 | 15 | 3.134 | 1.166e-06 |
| slogdet_phase_c64_32x32 | jax-cuda-jit | 8 | 15 | 2.389 | 5.706e-07 |
| householder_qr_square_f32_256x256 | jax-cuda-jit | 4 | 15 | 3.638 | 0.001043 |
| householder_qr_tall_f32_512x256 | jax-cuda-jit | 2 | 15 | 2.674 | 0.001475 |
| householder_qr_wide_f32_256x512 | jax-cuda-jit | 2 | 15 | 3.084 | 0.0009736 |
| householder_qr_square_f64_256x256 | jax-cuda-jit | 1 | 15 | 3.547 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | jax-cuda-jit | 1 | 15 | 4.004 | 1.887e-06 |
| householder_qr_wide_f64_256x512 | jax-cuda-jit | 1 | 15 | 5.590 | 1.759e-06 |
| householder_qr_square_c32_256x256 | jax-cuda-jit | 2 | 15 | 2.147 | 0.001258 |
| householder_qr_tall_c32_512x256 | jax-cuda-jit | 2 | 15 | 2.815 | 0.001161 |
| householder_qr_wide_c32_256x512 | jax-cuda-jit | 2 | 15 | 3.161 | 0.001226 |
| slogdet_phase_c32_256x256 | jax-cuda-jit | 8 | 15 | 3.613 | 0.01233 |
| householder_qr_square_c64_256x256 | jax-cuda-jit | 1 | 15 | 5.847 | 2.109e-06 |
| householder_qr_tall_c64_512x256 | jax-cuda-jit | 1 | 15 | 6.049 | 2.942e-06 |
| householder_qr_wide_c64_256x512 | jax-cuda-jit | 1 | 15 | 10.078 | 2.276e-06 |
| slogdet_phase_c64_256x256 | jax-cuda-jit | 2 | 15 | 3.172 | 1.025e-06 |
| householder_from_factors_default_f32_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_64x32 | jax-cuda-jit | 16 | 15 | 2.500 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | jax-cuda-jit | 8 | 15 | 2.490 | 0.0001196 |
| householder_q_columns_complement_f32_64x32 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f64_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_64x32 | jax-cuda-jit | 16 | 15 | 2.573 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | jax-cuda-jit | 8 | 15 | 2.743 | 1.114e-07 |
| householder_q_columns_complement_f64_64x32 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c32_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_64x32 | jax-cuda-jit | 16 | 15 | 2.521 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | jax-cuda-jit | 8 | 15 | 3.184 | 0.00012 |
| householder_q_columns_complement_c32_64x32 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c64_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_64x32 | jax-cuda-jit | 16 | 15 | 2.720 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | jax-cuda-jit | 8 | 15 | 3.035 | 1.676e-07 |
| householder_q_columns_complement_c64_64x32 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f32_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_512x256 | jax-cuda-jit | 16 | 15 | 2.324 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | jax-cuda-jit | 8 | 15 | 2.480 | 0.0001829 |
| householder_q_columns_complement_f32_512x256 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f64_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_512x256 | jax-cuda-jit | 16 | 15 | 2.607 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | jax-cuda-jit | 8 | 15 | 2.631 | 1.136e-07 |
| householder_q_columns_complement_f64_512x256 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c32_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_512x256 | jax-cuda-jit | 16 | 15 | 2.586 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | jax-cuda-jit | 8 | 15 | 2.872 | 0.0001246 |
| householder_q_columns_complement_c32_512x256 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c64_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_512x256 | jax-cuda-jit | 16 | 15 | 2.532 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | jax-cuda-jit | 4 | 15 | 2.020 | 1.154e-07 |
| householder_q_columns_complement_c64_512x256 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api
suite_file: benchmarks/gpu/linalg_public_api.yaml
timestamp: '2026-10-13T16:33:27Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: eb683d5ccde2070d7d0fd541303dcd0596afe415
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
  expected_cases: 140
  selected_cases: 140
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 5f4d30185e0a01686a1ca18a8b113772eb7698aacbe08f3ae0406f5a00b10357
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/multi.rs: 5efc881b0010ed15d35aff3a42eca76847ab445a117ebd69bc9580d58c116dfa
    src/bin/benchmark_gpu_public_api/structural.rs: 8121b12b5fed3b0acbc71e2218856173d53a5acc93092b93c7fd3309d4f45962
    src/bin/benchmark_gpu_public_api/linalg.rs: 84162923fa0d0772c88ec9c4eab169ad3a75e5d16ff76961d1541670dd8565c1
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: ad72b565373904275128d7beb8673bada08d922e8f310be6cedb0d2d8187898f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 850ab2ead404652e542a2f13085d7ee070edbda379ccaeec4658219973d242d2
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 386e07c26784dad6630b00435133a78af798d2455fdc382754a89f66faceb1a3
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/linalg_public_api/20261013_163327/fixtures/manifest.json: d5de07c13c7fb634ec9403788d1ec4a0de87a81d97e4c72947384369fa0664d9
    target/release/benchmark_gpu_public_api: 7098bc335ed2fefea246f9596e45182481de4e54039090f96f59e8adc542115c
  reference_numpy_version: 2.4.2
```
