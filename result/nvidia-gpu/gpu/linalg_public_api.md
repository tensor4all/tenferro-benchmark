# gpu/linalg_public_api: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api/20261013_172820/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | 139.988 (10.646) | 216.309 (3.663) | 87.329 (0.987) | 181.665 (22.401) | 126.641 (3.190) |
| cholesky_f64_32x32 | 255.985 (7.680) | 305.965 (13.420) | 161.779 (1.631) | 199.819 (11.329) | 211.930 (11.268) |
| svdvals_f32_32x32 | 677.922 (44.375) | unsupported | 406.349 (5.804) | 861.571 (51.005) | 775.705 (16.373) |
| svdvals_f64_32x32 | 5457.634 (10.611) | unsupported | 5085.741 (6.806) | 5870.276 (112.226) | 5811.319 (44.400) |
| eigvalsh_f32_32x32 | 232.659 (4.428) | 297.917 (5.729) | 218.620 (2.581) | 585.806 (19.675) | 623.888 (12.375) |
| eigvalsh_f64_32x32 | 614.881 (7.787) | 678.374 (5.909) | 608.468 (1.817) | 2949.592 (19.856) | 2952.808 (7.314) |
| pinv_f32_32x32 | unsupported | unsupported | 532.632 (6.681) | 1033.751 (41.633) | 727.558 (5.631) |
| pinv_f64_32x32 | unsupported | unsupported | 4917.851 (19.193) | 5607.210 (47.292) | 5435.463 (26.045) |
| inv_f32_32x32 | unsupported | unsupported | 280.317 (0.954) | 157.838 (4.839) | 248.295 (27.330) |
| inv_f64_32x32 | unsupported | unsupported | 283.794 (10.770) | 212.195 (27.128) | 328.856 (26.399) |
| det_f32_32x32 | 536.396 (43.869) | 801.617 (9.814) | 162.775 (2.396) | 156.083 (3.182) | 167.421 (1.047) |
| det_f64_32x32 | 590.523 (11.917) | 832.361 (25.089) | 162.977 (1.530) | 159.139 (8.141) | 259.205 (35.972) |
| norm_f32_32x32 | 78.284 (1.293) | 193.964 (9.642) | 13.897 (0.836) | 100.741 (18.275) | 173.464 (22.922) |
| norm_f64_32x32 | 75.175 (0.584) | 211.407 (6.146) | 13.885 (0.752) | 132.203 (3.589) | 173.218 (14.508) |
| triangular_solve_f32_32x32 | 87.285 (3.255) | 269.855 (51.695) | 56.014 (2.264) | 184.748 (31.691) | 169.375 (10.296) |
| triangular_solve_f64_32x32 | 77.176 (1.850) | 262.026 (12.524) | 54.798 (3.106) | 219.282 (38.874) | 160.750 (3.875) |
| cholesky_f32_512x512 | 349.833 (6.956) | 474.189 (11.825) | 279.936 (3.570) | 273.149 (7.372) | 269.086 (5.692) |
| cholesky_f64_512x512 | 2271.014 (10.506) | 2517.891 (12.871) | 2189.536 (2.017) | 2761.647 (17.505) | 2748.704 (18.151) |
| svdvals_f32_512x512 | 25797.517 (22.039) | unsupported | 28091.013 (16.754) | 25956.080 (24.487) | 25966.316 (761.064) |
| svdvals_f64_512x512 | 541448.902 (72.939) | unsupported | 593344.102 (21.549) | 539421.794 (31.948) | 539571.955 (19.226) |
| eigvalsh_f32_512x512 | 4281.002 (12.815) | 4629.390 (8.107) | 4358.437 (10.206) | 5019.781 (58.466) | 5001.827 (59.369) |
| eigvalsh_f64_512x512 | 14699.938 (33.079) | 14781.635 (32.589) | 14676.821 (56.767) | 19608.080 (159.690) | 19571.301 (127.513) |
| pinv_f32_512x512 | unsupported | unsupported | 31576.210 (33.618) | 29187.533 (25.520) | 28736.590 (16.028) |
| pinv_f64_512x512 | unsupported | unsupported | 681760.782 (99.113) | 617696.596 (62.923) | 617321.952 (34.446) |
| inv_f32_512x512 | unsupported | unsupported | 1947.600 (2.995) | 2234.152 (26.123) | 2321.394 (41.349) |
| inv_f64_512x512 | unsupported | unsupported | 5996.381 (4.698) | 6395.804 (34.474) | 6419.423 (39.669) |
| det_f32_512x512 | 1818.793 (23.004) | 1956.802 (61.672) | 1437.682 (4.334) | 1613.519 (9.573) | 1626.516 (12.062) |
| det_f64_512x512 | 2918.315 (22.854) | 2971.039 (15.573) | 2503.787 (8.318) | 2888.702 (31.290) | 2793.689 (21.754) |
| norm_f32_512x512 | 41.915 (1.566) | 79.938 (1.204) | 20.493 (0.413) | 159.086 (1.559) | 165.980 (7.364) |
| norm_f64_512x512 | 39.632 (0.513) | 78.890 (1.053) | 19.668 (0.188) | 161.529 (2.231) | 168.205 (10.972) |
| triangular_solve_f32_512x512 | 72.191 (0.740) | 171.077 (4.073) | 103.649 (0.022) | 219.346 (48.036) | 201.773 (11.279) |
| triangular_solve_f64_512x512 | 142.056 (0.541) | 217.411 (8.625) | 845.039 (0.494) | 205.469 (18.909) | 217.847 (17.790) |
| svd_full_square_f32_32x32 | 661.085 (21.227) | 755.844 (16.601) | 340.343 (2.156) | 798.429 (38.903) | 727.709 (5.400) |
| svd_full_tall_f32_64x32 | 871.132 (10.448) | 938.699 (13.364) | 833.331 (2.153) | 1426.260 (28.712) | 1103.709 (28.848) |
| svd_full_wide_f32_32x64 | 922.530 (17.270) | 1021.235 (44.318) | 888.445 (5.176) | 1398.864 (84.156) | 1479.114 (37.579) |
| slogdet_positive_f32_32x32 | 331.876 (5.773) | 443.507 (18.288) | 206.840 (2.376) | 165.053 (4.221) | 184.461 (1.343) |
| slogdet_negative_f32_32x32 | 330.779 (3.755) | 448.291 (15.699) | 207.562 (10.109) | 158.787 (2.076) | 175.358 (4.636) |
| svd_full_square_f64_32x32 | 5107.905 (22.986) | 5219.948 (26.306) | 4281.866 (31.391) | 5577.999 (15.508) | 5732.233 (67.577) |
| svd_full_tall_f64_64x32 | 5877.777 (16.581) | 6016.575 (26.086) | 5970.553 (5.981) | 6186.186 (102.135) | 6096.761 (15.748) |
| svd_full_wide_f64_32x64 | 6038.479 (38.680) | 6221.996 (31.555) | 6311.521 (17.252) | 6228.152 (24.356) | 6268.722 (24.435) |
| slogdet_positive_f64_32x32 | 419.259 (37.989) | 498.725 (9.983) | 206.755 (1.926) | 272.232 (4.088) | 263.864 (14.429) |
| slogdet_negative_f64_32x32 | 363.489 (4.286) | 507.855 (16.119) | 215.628 (10.421) | 224.401 (35.802) | 250.409 (24.185) |
| svd_full_square_c32_32x32 | 834.823 (20.047) | 945.195 (19.391) | 472.516 (1.758) | 1043.753 (19.767) | 1074.558 (23.365) |
| svd_full_tall_c32_64x32 | 1078.828 (13.295) | 1186.974 (25.656) | 1046.177 (6.288) | 1514.364 (83.909) | 1484.303 (79.209) |
| svd_full_wide_c32_32x64 | 1056.314 (15.572) | 1218.733 (24.969) | 1054.072 (7.178) | 1568.915 (68.446) | 1527.938 (70.330) |
| slogdet_positive_c32_32x32 | 611.081 (12.537) | 496.543 (82.679) | 220.204 (4.341) | 289.293 (5.698) | 290.079 (39.399) |
| slogdet_negative_c32_32x32 | 568.081 (6.089) | 683.553 (11.077) | 220.686 (2.540) | 246.340 (36.382) | 280.210 (30.568) |
| svd_full_square_c64_32x32 | 14347.848 (12.282) | 14410.743 (51.600) | 11397.091 (9.395) | 14455.544 (17.131) | 14480.677 (7.948) |
| svd_full_tall_c64_64x32 | 15242.137 (21.184) | 15391.417 (27.911) | 15448.905 (16.959) | 15355.208 (18.394) | 15360.904 (16.107) |
| svd_full_wide_c64_32x64 | 15916.611 (19.567) | 16159.875 (29.851) | 16183.069 (21.376) | 16045.407 (68.117) | 16091.396 (17.606) |
| slogdet_positive_c64_32x32 | 574.754 (33.990) | 750.009 (14.470) | 224.193 (2.937) | 191.104 (13.976) | 225.797 (3.780) |
| slogdet_negative_c64_32x32 | 518.413 (29.955) | 667.931 (26.649) | 223.809 (0.709) | 169.351 (7.762) | 291.747 (15.463) |
| svd_full_square_f32_256x256 | 8576.467 (24.886) | 8609.780 (18.644) | 9066.244 (22.553) | 8764.434 (94.842) | 8761.784 (8.556) |
| svd_full_tall_f32_512x256 | 9078.558 (17.562) | 9093.966 (33.700) | 9585.433 (12.231) | 9192.599 (28.147) | 9226.586 (15.566) |
| svd_full_wide_f32_256x512 | 8850.308 (47.361) | 8859.950 (38.034) | 8969.166 (26.117) | 8956.006 (16.055) | 8977.183 (15.173) |
| slogdet_positive_f32_256x256 | 728.302 (7.445) | 742.553 (14.421) | 371.997 (0.179) | 385.003 (12.994) | 397.205 (14.646) |
| slogdet_negative_f32_256x256 | 662.528 (10.184) | 747.245 (17.902) | 372.317 (0.393) | 395.108 (9.857) | 397.097 (8.348) |
| svd_full_square_f64_256x256 | 161356.723 (16.474) | 161481.826 (57.848) | 165646.696 (46.785) | 160793.823 (44.786) | 160808.379 (29.307) |
| svd_full_tall_f64_512x256 | 128538.997 (89.404) | 128642.559 (114.619) | 131148.298 (53.625) | 128177.700 (101.828) | 128251.232 (106.470) |
| svd_full_wide_f64_256x512 | 118927.568 (159.292) | 119063.294 (52.805) | 121651.503 (47.522) | 118656.736 (128.909) | 118742.820 (64.589) |
| slogdet_positive_f64_256x256 | 1072.258 (15.954) | 1257.216 (24.227) | 722.716 (0.878) | 801.660 (7.832) | 808.267 (6.686) |
| slogdet_negative_f64_256x256 | 1076.652 (30.102) | 1255.975 (15.617) | 722.560 (0.438) | 811.696 (2.936) | 813.370 (4.184) |
| svd_full_square_c32_256x256 | 13167.865 (88.278) | 13216.337 (40.443) | 14282.212 (11.722) | 13256.206 (19.201) | 13279.087 (16.792) |
| svd_full_tall_c32_512x256 | 13218.321 (37.284) | 13187.385 (21.507) | 13373.694 (11.181) | 13245.451 (15.444) | 13263.139 (21.637) |
| svd_full_wide_c32_256x512 | 13269.362 (27.084) | 13259.378 (19.532) | 13515.658 (15.767) | 13185.851 (29.514) | 13329.088 (27.170) |
| slogdet_positive_c32_256x256 | 711.245 (17.986) | 1012.338 (26.487) | 409.365 (0.381) | 421.212 (14.900) | 481.774 (14.039) |
| slogdet_negative_c32_256x256 | 709.438 (6.332) | 965.308 (24.613) | 410.036 (0.392) | 425.570 (18.407) | 432.822 (13.787) |
| svd_full_square_c64_256x256 | 517205.149 (41.800) | 515067.181 (37.030) | 545355.103 (98.143) | 512835.722 (33.673) | 512815.025 (39.094) |
| svd_full_tall_c64_512x256 | 369824.683 (132.110) | 368442.017 (130.998) | 372768.794 (88.133) | 366828.326 (89.581) | 366855.308 (56.973) |
| svd_full_wide_c64_256x512 | 371159.009 (118.379) | 369688.546 (114.093) | 375984.699 (76.595) | 368158.599 (110.815) | 368158.441 (63.734) |
| slogdet_positive_c64_256x256 | 1705.702 (21.084) | 1951.417 (36.716) | 1398.665 (2.120) | 1587.599 (8.797) | 1594.436 (10.415) |
| slogdet_negative_c64_256x256 | 1684.414 (4.563) | 1833.985 (20.340) | 1398.621 (0.816) | 1598.915 (7.000) | 1597.733 (10.616) |
| householder_qr_square_f32_32x32 | 170.640 (2.719) | 350.741 (6.256) | 59.922 (0.145) | 131.517 (2.714) | 129.372 (1.503) |
| householder_qr_tall_f32_64x32 | 194.198 (9.602) | 397.158 (10.636) | 105.090 (0.839) | 273.502 (51.309) | 196.834 (4.567) |
| householder_qr_wide_f32_32x64 | 177.392 (4.082) | 343.122 (13.750) | 105.461 (1.633) | 182.460 (4.046) | 305.578 (29.365) |
| householder_qr_square_f64_32x32 | 410.177 (4.783) | 599.087 (8.905) | 287.179 (0.206) | 314.244 (2.156) | 322.444 (10.657) |
| householder_qr_tall_f64_64x32 | 463.636 (5.867) | 660.858 (17.651) | 356.611 (0.068) | 406.496 (12.146) | 424.620 (7.824) |
| householder_qr_wide_f64_32x64 | 470.957 (3.509) | 649.707 (9.386) | 360.805 (0.281) | 398.284 (7.753) | 423.544 (6.531) |
| householder_qr_square_c32_32x32 | 181.145 (0.800) | 396.190 (4.946) | 68.001 (0.039) | 186.670 (50.478) | 271.846 (33.860) |
| householder_qr_tall_c32_64x32 | 212.308 (13.332) | 414.950 (32.685) | 111.786 (0.369) | 376.862 (16.548) | 388.182 (99.155) |
| householder_qr_wide_c32_32x64 | 199.877 (2.266) | 327.056 (4.871) | 106.264 (0.505) | 361.406 (10.812) | 321.512 (78.849) |
| slogdet_phase_c32_32x32 | 392.136 (9.185) | 603.933 (15.389) | 224.426 (8.510) | 266.726 (46.713) | 304.324 (39.937) |
| householder_qr_square_c64_32x32 | 589.958 (4.264) | 778.427 (12.591) | 481.938 (0.247) | 604.274 (6.554) | 623.941 (20.267) |
| householder_qr_tall_c64_64x32 | 676.141 (8.273) | 878.845 (13.590) | 578.500 (0.185) | 694.178 (17.534) | 733.961 (4.446) |
| householder_qr_wide_c64_32x64 | 715.403 (4.925) | 886.065 (11.200) | 622.856 (0.209) | 765.320 (9.535) | 763.499 (12.467) |
| slogdet_phase_c64_32x32 | 423.985 (9.893) | 578.300 (11.385) | 227.662 (8.177) | 307.647 (11.796) | 343.186 (41.684) |
| householder_qr_square_f32_256x256 | 833.306 (2.917) | 919.658 (6.965) | 756.783 (5.467) | 1057.144 (19.140) | 1016.568 (19.878) |
| householder_qr_tall_f32_512x256 | 1118.885 (10.638) | 1163.850 (8.978) | 1044.385 (5.347) | 1375.746 (7.031) | 1352.710 (33.111) |
| householder_qr_wide_f32_256x512 | 1305.609 (5.944) | 1357.236 (10.409) | 1240.201 (4.025) | 1521.931 (38.410) | 1506.515 (21.995) |
| householder_qr_square_f64_256x256 | 2974.139 (13.347) | 3206.022 (22.950) | 2911.675 (15.666) | 3552.898 (49.117) | 3480.895 (16.523) |
| householder_qr_tall_f64_512x256 | 3551.358 (22.890) | 3603.438 (8.896) | 3297.446 (10.262) | 3941.020 (67.268) | 3909.817 (80.576) |
| householder_qr_wide_f64_256x512 | 5124.883 (26.969) | 5192.488 (30.106) | 4796.509 (21.491) | 5556.317 (50.689) | 5549.328 (52.791) |
| householder_qr_square_c32_256x256 | 867.117 (5.676) | 957.174 (9.054) | 789.602 (3.260) | 1015.716 (11.572) | 880.361 (5.461) |
| householder_qr_tall_c32_512x256 | 1196.762 (9.565) | 1305.403 (7.453) | 1131.352 (9.847) | 1274.614 (31.210) | 1271.021 (31.781) |
| householder_qr_wide_c32_256x512 | 1362.624 (8.311) | 1482.852 (4.106) | 1299.999 (6.745) | 1565.995 (30.347) | 1577.403 (29.233) |
| slogdet_phase_c32_256x256 | 661.852 (6.047) | 803.177 (15.438) | 406.832 (0.504) | 445.850 (7.868) | 506.693 (16.062) |
| householder_qr_square_c64_256x256 | 5426.768 (20.408) | 5458.204 (40.480) | 5398.782 (9.104) | 5840.887 (78.770) | 5846.618 (44.466) |
| householder_qr_tall_c64_512x256 | 5772.370 (9.437) | 5771.317 (42.547) | 5677.920 (15.887) | 6101.618 (96.524) | 6061.120 (72.041) |
| householder_qr_wide_c64_256x512 | 9821.330 (36.659) | 9782.251 (49.132) | 9662.313 (10.665) | 10114.687 (56.960) | 10123.069 (67.769) |
| slogdet_phase_c64_256x256 | 1750.272 (16.381) | 1739.053 (12.759) | 1392.257 (1.122) | 1576.853 (5.954) | 1572.611 (5.128) |
| householder_from_factors_default_f32_64x32 | 747.521 (9.958) | 667.267 (20.074) | unsupported | unsupported | unsupported |
| householder_append_default_f32_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f32_64x32 | 29.179 (1.929) | 32.976 (0.307) | 7.986 (0.106) | 132.756 (27.732) | 147.253 (21.633) |
| householder_q_columns_prefix_f32_64x32 | unsupported | unsupported | 98.618 (3.293) | 369.166 (71.038) | 192.918 (1.480) |
| householder_q_columns_complement_f32_64x32 | unsupported | unsupported | 181.234 (0.723) | unsupported | unsupported |
| householder_from_factors_default_f64_64x32 | 952.422 (30.370) | 974.510 (30.056) | unsupported | unsupported | unsupported |
| householder_append_default_f64_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f64_64x32 | 32.126 (0.433) | 33.174 (0.417) | 7.949 (0.102) | 136.326 (15.823) | 157.619 (26.308) |
| householder_q_columns_prefix_f64_64x32 | unsupported | unsupported | 103.797 (0.549) | 445.195 (21.045) | 359.504 (9.587) |
| householder_q_columns_complement_f64_64x32 | unsupported | unsupported | 229.485 (0.102) | unsupported | unsupported |
| householder_from_factors_default_c32_64x32 | 799.488 (5.849) | 619.301 (14.505) | unsupported | unsupported | unsupported |
| householder_append_default_c32_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c32_64x32 | 40.761 (1.867) | 35.165 (0.538) | 8.740 (0.082) | 155.962 (25.609) | 156.257 (31.496) |
| householder_q_columns_prefix_c32_64x32 | unsupported | unsupported | 122.216 (2.181) | 503.225 (13.222) | 415.904 (26.322) |
| householder_q_columns_complement_c32_64x32 | unsupported | unsupported | 179.857 (1.896) | unsupported | unsupported |
| householder_from_factors_default_c64_64x32 | 1253.750 (15.295) | 1511.777 (48.233) | unsupported | unsupported | unsupported |
| householder_append_default_c64_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c64_64x32 | 42.624 (3.017) | 64.750 (3.470) | 7.934 (0.199) | 145.394 (27.497) | 170.952 (34.105) |
| householder_q_columns_prefix_c64_64x32 | unsupported | unsupported | 110.468 (2.727) | 459.893 (35.801) | 392.512 (20.884) |
| householder_q_columns_complement_c64_64x32 | unsupported | unsupported | 434.672 (0.202) | unsupported | unsupported |
| householder_from_factors_default_f32_512x256 | 1479.176 (8.335) | 1523.805 (8.022) | unsupported | unsupported | unsupported |
| householder_append_default_f32_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f32_512x256 | 17.604 (0.390) | 30.768 (2.460) | 8.068 (0.113) | 104.745 (21.404) | 168.855 (35.858) |
| householder_q_columns_prefix_f32_512x256 | unsupported | unsupported | 99.362 (2.832) | 440.182 (25.650) | 192.824 (0.745) |
| householder_q_columns_complement_f32_512x256 | unsupported | unsupported | 204.741 (0.813) | unsupported | unsupported |
| householder_from_factors_default_f64_512x256 | 4165.370 (31.635) | 4221.898 (23.159) | unsupported | unsupported | unsupported |
| householder_append_default_f64_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f64_512x256 | 30.712 (0.674) | 26.468 (0.561) | 8.095 (0.127) | 156.703 (32.393) | 168.303 (44.168) |
| householder_q_columns_prefix_f64_512x256 | unsupported | unsupported | 192.723 (0.186) | 460.030 (36.887) | 357.486 (33.202) |
| householder_q_columns_complement_f64_512x256 | unsupported | unsupported | 1541.619 (5.124) | unsupported | unsupported |
| householder_from_factors_default_c32_512x256 | 1566.844 (6.740) | 1570.574 (14.049) | unsupported | unsupported | unsupported |
| householder_append_default_c32_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c32_512x256 | 30.683 (0.414) | 29.370 (0.623) | 8.717 (0.094) | 169.378 (18.940) | 170.577 (35.483) |
| householder_q_columns_prefix_c32_512x256 | unsupported | unsupported | 122.277 (1.368) | 496.501 (7.185) | 413.665 (29.426) |
| householder_q_columns_complement_c32_512x256 | unsupported | unsupported | 257.148 (0.205) | unsupported | unsupported |
| householder_from_factors_default_c64_512x256 | 7351.974 (31.898) | 7331.847 (60.722) | unsupported | unsupported | unsupported |
| householder_append_default_c64_512x256 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c64_512x256 | 29.393 (0.639) | 26.442 (0.903) | 8.805 (0.367) | 161.856 (30.412) | 168.539 (37.212) |
| householder_q_columns_prefix_c64_512x256 | unsupported | unsupported | 375.835 (1.248) | 588.094 (39.597) | 500.335 (5.815) |
| householder_q_columns_complement_c64_512x256 | unsupported | unsupported | 4940.339 (3.823) | unsupported | unsupported |
| lu_square_f32_s32 | 285.686 (19.934) | 373.852 (11.700) | unsupported | unsupported | unsupported |
| lu_square_f64_s32 | 269.144 (11.015) | 354.582 (9.295) | unsupported | unsupported | unsupported |
| lu_square_c32_s32 | 249.007 (2.876) | 340.163 (7.082) | unsupported | unsupported | unsupported |
| lu_square_c64_s32 | 277.480 (1.873) | 382.798 (9.336) | unsupported | unsupported | unsupported |
| lu_tall_f32_s32 | 252.380 (0.897) | 356.683 (27.601) | unsupported | unsupported | unsupported |
| lu_tall_f64_s32 | 267.732 (2.131) | 418.996 (24.508) | unsupported | unsupported | unsupported |
| lu_tall_c32_s32 | 252.352 (3.656) | 406.313 (9.756) | unsupported | unsupported | unsupported |
| lu_tall_c64_s32 | 292.659 (2.908) | 451.246 (9.446) | unsupported | unsupported | unsupported |
| lu_wide_f32_s32 | 257.097 (2.958) | 410.010 (19.729) | unsupported | unsupported | unsupported |
| lu_wide_f64_s32 | 273.571 (1.442) | 447.585 (11.252) | unsupported | unsupported | unsupported |
| lu_wide_c32_s32 | 259.194 (2.360) | 447.291 (12.205) | unsupported | unsupported | unsupported |
| lu_wide_c64_s32 | 305.181 (1.126) | 481.512 (13.225) | unsupported | unsupported | unsupported |
| lu_square_f32_s256 | 566.827 (4.131) | 705.094 (6.449) | unsupported | unsupported | unsupported |
| lu_square_f64_s256 | 892.840 (7.600) | 1043.465 (33.621) | unsupported | unsupported | unsupported |
| lu_square_c32_s256 | 588.526 (5.215) | 746.094 (7.205) | unsupported | unsupported | unsupported |
| lu_square_c64_s256 | 1552.562 (5.824) | 1684.987 (12.222) | unsupported | unsupported | unsupported |
| lu_tall_f32_s256 | 784.612 (5.402) | 922.876 (6.406) | unsupported | unsupported | unsupported |
| lu_tall_f64_s256 | 1467.219 (6.556) | 1618.254 (11.764) | unsupported | unsupported | unsupported |
| lu_tall_c32_s256 | 818.396 (6.511) | 970.276 (6.213) | unsupported | unsupported | unsupported |
| lu_tall_c64_s256 | 2607.138 (15.312) | 2759.150 (25.700) | unsupported | unsupported | unsupported |
| lu_wide_f32_s256 | 599.369 (6.939) | 744.590 (10.150) | unsupported | unsupported | unsupported |
| lu_wide_f64_s256 | 1088.138 (14.143) | 1240.324 (14.760) | unsupported | unsupported | unsupported |
| lu_wide_c32_s256 | 643.956 (10.471) | 803.614 (14.824) | unsupported | unsupported | unsupported |
| lu_wide_c64_s256 | 2290.877 (85.738) | 2374.952 (76.349) | unsupported | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 2.240 | 0.0002352 |
| cholesky_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.048 | 2.188e-07 |
| svdvals_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.712 | 0.003338 |
| svdvals_f64_32x32 | tenferro-cuda-direct | 1 | 15 | 5.458 | 4.546e-06 |
| eigvalsh_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.723 | 0.001073 |
| eigvalsh_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.460 | 5.433e-07 |
| pinv_f32_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| det_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 4.291 | 0.002641 |
| det_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.362 | 1.135e-06 |
| norm_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.505 | 0 |
| norm_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.406 | 0 |
| triangular_solve_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.793 | 0.0001546 |
| triangular_solve_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.470 | 1.609e-07 |
| cholesky_f32_512x512 | tenferro-cuda-direct | 8 | 15 | 2.799 | 0.0003554 |
| cholesky_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.271 | 4.418e-07 |
| svdvals_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 25.798 | 0.1547 |
| svdvals_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 541.449 | 0.0002063 |
| eigvalsh_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 4.281 | 0.008342 |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 14.700 | 1.55e-06 |
| pinv_f32_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| det_f32_512x512 | tenferro-cuda-direct | 2 | 15 | 3.638 | 0.008577 |
| det_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.918 | 1.426e-05 |
| norm_f32_512x512 | tenferro-cuda-direct | 64 | 15 | 2.683 | 0 |
| norm_f64_512x512 | tenferro-cuda-direct | 64 | 15 | 2.536 | 1.909e-07 |
| triangular_solve_f32_512x512 | tenferro-cuda-direct | 32 | 15 | 2.310 | 0.0007877 |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | 16 | 15 | 2.273 | 1.157e-06 |
| svd_full_square_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.644 | 0.008731 |
| svd_full_tall_f32_64x32 | tenferro-cuda-direct | 4 | 15 | 3.485 | 0.004865 |
| svd_full_wide_f32_32x64 | tenferro-cuda-direct | 4 | 15 | 3.690 | 0.005495 |
| slogdet_positive_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.655 | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.646 | 0.000915 |
| svd_full_square_f64_32x32 | tenferro-cuda-direct | 1 | 15 | 5.108 | 1.007e-05 |
| svd_full_tall_f64_64x32 | tenferro-cuda-direct | 1 | 15 | 5.878 | 8.327e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-direct | 1 | 15 | 6.038 | 1.197e-05 |
| slogdet_positive_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 3.354 | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.908 | 4.261e-07 |
| svd_full_square_c32_32x32 | tenferro-cuda-direct | 4 | 15 | 3.339 | 0.009079 |
| svd_full_tall_c32_64x32 | tenferro-cuda-direct | 2 | 15 | 2.158 | 0.009498 |
| svd_full_wide_c32_32x64 | tenferro-cuda-direct | 2 | 15 | 2.113 | 0.008495 |
| slogdet_positive_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 4.889 | 0.0007148 |
| slogdet_negative_c32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.272 | 0.0007148 |
| svd_full_square_c64_32x32 | tenferro-cuda-direct | 1 | 15 | 14.348 | 9.215e-06 |
| svd_full_tall_c64_64x32 | tenferro-cuda-direct | 1 | 15 | 15.242 | 1.221e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-direct | 1 | 15 | 15.917 | 1.31e-05 |
| slogdet_positive_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 4.598 | 3.804e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.074 | 3.804e-07 |
| svd_full_square_f32_256x256 | tenferro-cuda-direct | 1 | 15 | 8.576 | 0.1152 |
| svd_full_tall_f32_512x256 | tenferro-cuda-direct | 1 | 15 | 9.079 | 0.1022 |
| svd_full_wide_f32_256x512 | tenferro-cuda-direct | 1 | 15 | 8.850 | 0.1056 |
| slogdet_positive_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.913 | 0.0005003 |
| slogdet_negative_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.650 | 0.0005003 |
| svd_full_square_f64_256x256 | tenferro-cuda-direct | 1 | 15 | 161.357 | 8.171e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 128.539 | 7.76e-05 |
| svd_full_wide_f64_256x512 | tenferro-cuda-direct | 1 | 15 | 118.928 | 7.938e-05 |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | 2 | 15 | 2.145 | 1.864e-06 |
| slogdet_negative_f64_256x256 | tenferro-cuda-direct | 2 | 15 | 2.153 | 1.864e-06 |
| svd_full_square_c32_256x256 | tenferro-cuda-direct | 1 | 15 | 13.168 | 0.133 |
| svd_full_tall_c32_512x256 | tenferro-cuda-direct | 1 | 15 | 13.218 | 0.1126 |
| svd_full_wide_c32_256x512 | tenferro-cuda-direct | 1 | 15 | 13.269 | 0.1103 |
| slogdet_positive_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.845 | 0.01211 |
| slogdet_negative_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.838 | 0.01211 |
| svd_full_square_c64_256x256 | tenferro-cuda-direct | 1 | 15 | 517.205 | 0.000204 |
| svd_full_tall_c64_512x256 | tenferro-cuda-direct | 1 | 15 | 369.825 | 0.0002024 |
| svd_full_wide_c64_256x512 | tenferro-cuda-direct | 1 | 15 | 371.159 | 0.0001941 |
| slogdet_positive_c64_256x256 | tenferro-cuda-direct | 2 | 15 | 3.411 | 1.025e-06 |
| slogdet_negative_c64_256x256 | tenferro-cuda-direct | 2 | 15 | 3.369 | 1.025e-06 |
| householder_qr_square_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 2.730 | 0.0004213 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-direct | 16 | 15 | 3.107 | 0.0003161 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-direct | 16 | 15 | 2.838 | 0.0004732 |
| householder_qr_square_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 3.281 | 3.886e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-direct | 8 | 15 | 3.709 | 4.441e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-direct | 8 | 15 | 3.768 | 4.441e-07 |
| householder_qr_square_c32_32x32 | tenferro-cuda-direct | 16 | 15 | 2.898 | 0.0005631 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-direct | 16 | 15 | 3.397 | 0.0003724 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-direct | 16 | 15 | 3.198 | 0.0006002 |
| slogdet_phase_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 3.137 | 0.0004085 |
| householder_qr_square_c64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.360 | 4.996e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-direct | 4 | 15 | 2.705 | 5.551e-07 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-direct | 4 | 15 | 2.862 | 3.886e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 3.392 | 4.755e-07 |
| householder_qr_square_f32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.333 | 0.0009546 |
| householder_qr_tall_f32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.238 | 0.001435 |
| householder_qr_wide_f32_256x512 | tenferro-cuda-direct | 2 | 15 | 2.611 | 0.001037 |
| householder_qr_square_f64_256x256 | tenferro-cuda-direct | 1 | 15 | 2.974 | 1.098e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 3.551 | 9.992e-07 |
| householder_qr_wide_f64_256x512 | tenferro-cuda-direct | 1 | 15 | 5.125 | 1.649e-06 |
| householder_qr_square_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 3.468 | 0.001198 |
| householder_qr_tall_c32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.394 | 0.001185 |
| householder_qr_wide_c32_256x512 | tenferro-cuda-direct | 2 | 15 | 2.725 | 0.001329 |
| slogdet_phase_c32_256x256 | tenferro-cuda-direct | 4 | 15 | 2.647 | 0.01211 |
| householder_qr_square_c64_256x256 | tenferro-cuda-direct | 1 | 15 | 5.427 | 1.208e-06 |
| householder_qr_tall_c64_512x256 | tenferro-cuda-direct | 1 | 15 | 5.772 | 1.11e-06 |
| householder_qr_wide_c64_256x512 | tenferro-cuda-direct | 1 | 15 | 9.821 | 1.206e-06 |
| slogdet_phase_c64_256x256 | tenferro-cuda-direct | 2 | 15 | 3.501 | 1.025e-06 |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-direct | 4 | 15 | 2.990 | 0.0002423 |
| householder_append_default_f32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f32_64x32 | tenferro-cuda-direct | 64 | 15 | 1.867 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-direct | 4 | 15 | 3.810 | 3.886e-07 |
| householder_append_default_f64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f64_64x32 | tenferro-cuda-direct | 64 | 15 | 2.056 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-direct | 4 | 15 | 3.198 | 0.0003232 |
| householder_append_default_c32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c32_64x32 | tenferro-cuda-direct | 64 | 15 | 2.609 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-direct | 2 | 15 | 2.507 | 4.441e-07 |
| householder_append_default_c64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c64_64x32 | tenferro-cuda-direct | 64 | 15 | 2.728 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f32_512x256 | tenferro-cuda-direct | 2 | 15 | 2.958 | 0.0008522 |
| householder_append_default_f32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f32_512x256 | tenferro-cuda-direct | 128 | 15 | 2.253 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 4.165 | 8.882e-07 |
| householder_append_default_f64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | 128 | 15 | 3.931 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_512x256 | tenferro-cuda-direct | 2 | 15 | 3.134 | 0.001103 |
| householder_append_default_c32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c32_512x256 | tenferro-cuda-direct | 128 | 15 | 3.927 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c64_512x256 | tenferro-cuda-direct | 1 | 15 | 7.352 | 9.992e-07 |
| householder_append_default_c64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c64_512x256 | tenferro-cuda-direct | 128 | 15 | 3.762 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_512x256 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| lu_square_f32_s32 | tenferro-cuda-direct | 8 | 15 | 2.285 | 0.0003557 |
| lu_square_f64_s32 | tenferro-cuda-direct | 8 | 15 | 2.153 | 4.397e-07 |
| lu_square_c32_s32 | tenferro-cuda-direct | 16 | 15 | 3.984 | 0.0002381 |
| lu_square_c64_s32 | tenferro-cuda-direct | 8 | 15 | 2.220 | 5.532e-07 |
| lu_tall_f32_s32 | tenferro-cuda-direct | 8 | 15 | 2.019 | 0.0003557 |
| lu_tall_f64_s32 | tenferro-cuda-direct | 8 | 15 | 2.142 | 4.397e-07 |
| lu_tall_c32_s32 | tenferro-cuda-direct | 8 | 15 | 2.019 | 0.0005956 |
| lu_tall_c64_s32 | tenferro-cuda-direct | 8 | 15 | 2.341 | 4.406e-07 |
| lu_wide_f32_s32 | tenferro-cuda-direct | 8 | 15 | 2.057 | 0.00024 |
| lu_wide_f64_s32 | tenferro-cuda-direct | 8 | 15 | 2.189 | 2.214e-07 |
| lu_wide_c32_s32 | tenferro-cuda-direct | 8 | 15 | 2.074 | 0.0003567 |
| lu_wide_c64_s32 | tenferro-cuda-direct | 8 | 15 | 2.441 | 3.323e-07 |
| lu_square_f32_s256 | tenferro-cuda-direct | 4 | 15 | 2.267 | 0.001428 |
| lu_square_f64_s256 | tenferro-cuda-direct | 4 | 15 | 3.571 | 1.102e-06 |
| lu_square_c32_s256 | tenferro-cuda-direct | 4 | 15 | 2.354 | 0.001541 |
| lu_square_c64_s256 | tenferro-cuda-direct | 2 | 15 | 3.105 | 1.431e-06 |
| lu_tall_f32_s256 | tenferro-cuda-direct | 4 | 15 | 3.138 | 0.001428 |
| lu_tall_f64_s256 | tenferro-cuda-direct | 2 | 15 | 2.934 | 1.102e-06 |
| lu_tall_c32_s256 | tenferro-cuda-direct | 4 | 15 | 3.274 | 0.002239 |
| lu_tall_c64_s256 | tenferro-cuda-direct | 1 | 15 | 2.607 | 1.542e-06 |
| lu_wide_f32_s256 | tenferro-cuda-direct | 4 | 15 | 2.397 | 0.001297 |
| lu_wide_f64_s256 | tenferro-cuda-direct | 2 | 15 | 2.176 | 8.806e-07 |
| lu_wide_c32_s256 | tenferro-cuda-direct | 4 | 15 | 2.576 | 0.002021 |
| lu_wide_c64_s256 | tenferro-cuda-direct | 1 | 15 | 2.291 | 1.433e-06 |
| cholesky_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 3.461 | 0.0002352 |
| cholesky_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.448 | 2.188e-07 |
| svdvals_f32_32x32 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_32x32 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| eigvalsh_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.383 | 0.001073 |
| eigvalsh_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 2.713 | 5.433e-07 |
| pinv_f32_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f32_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| det_f32_32x32 | tenferro-cuda-eager | 4 | 15 | 3.206 | 0.002641 |
| det_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 3.329 | 1.135e-06 |
| norm_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 3.103 | 0 |
| norm_f64_32x32 | tenferro-cuda-eager | 16 | 15 | 3.383 | 0 |
| triangular_solve_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.159 | 0.0001546 |
| triangular_solve_f64_32x32 | tenferro-cuda-eager | 16 | 15 | 4.192 | 1.609e-07 |
| cholesky_f32_512x512 | tenferro-cuda-eager | 8 | 15 | 3.794 | 0.0003554 |
| cholesky_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 2.518 | 4.418e-07 |
| svdvals_f32_512x512 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_512x512 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| eigvalsh_f32_512x512 | tenferro-cuda-eager | 1 | 15 | 4.629 | 0.008342 |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 14.782 | 1.55e-06 |
| pinv_f32_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f32_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| det_f32_512x512 | tenferro-cuda-eager | 2 | 15 | 3.914 | 0.008577 |
| det_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 2.971 | 1.426e-05 |
| norm_f32_512x512 | tenferro-cuda-eager | 32 | 15 | 2.558 | 0 |
| norm_f64_512x512 | tenferro-cuda-eager | 32 | 15 | 2.524 | 1.909e-07 |
| triangular_solve_f32_512x512 | tenferro-cuda-eager | 16 | 15 | 2.737 | 0.0007877 |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | 16 | 15 | 3.479 | 1.157e-06 |
| svd_full_square_f32_32x32 | tenferro-cuda-eager | 4 | 15 | 3.023 | 0.008731 |
| svd_full_tall_f32_64x32 | tenferro-cuda-eager | 4 | 15 | 3.755 | 0.004865 |
| svd_full_wide_f32_32x64 | tenferro-cuda-eager | 4 | 15 | 4.085 | 0.005495 |
| slogdet_positive_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.548 | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.586 | 0.000915 |
| svd_full_square_f64_32x32 | tenferro-cuda-eager | 1 | 15 | 5.220 | 1.007e-05 |
| svd_full_tall_f64_64x32 | tenferro-cuda-eager | 1 | 15 | 6.017 | 8.327e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-eager | 1 | 15 | 6.222 | 1.197e-05 |
| slogdet_positive_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.990 | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 2.031 | 4.261e-07 |
| svd_full_square_c32_32x32 | tenferro-cuda-eager | 4 | 15 | 3.781 | 0.009079 |
| svd_full_tall_c32_64x32 | tenferro-cuda-eager | 2 | 15 | 2.374 | 0.009498 |
| svd_full_wide_c32_32x64 | tenferro-cuda-eager | 2 | 15 | 2.437 | 0.008495 |
| slogdet_positive_c32_32x32 | tenferro-cuda-eager | 4 | 15 | 1.986 | 0.0007148 |
| slogdet_negative_c32_32x32 | tenferro-cuda-eager | 4 | 15 | 2.734 | 0.0007148 |
| svd_full_square_c64_32x32 | tenferro-cuda-eager | 1 | 15 | 14.411 | 9.215e-06 |
| svd_full_tall_c64_64x32 | tenferro-cuda-eager | 1 | 15 | 15.391 | 1.221e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-eager | 1 | 15 | 16.160 | 1.31e-05 |
| slogdet_positive_c64_32x32 | tenferro-cuda-eager | 4 | 15 | 3.000 | 3.804e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-eager | 4 | 15 | 2.672 | 3.804e-07 |
| svd_full_square_f32_256x256 | tenferro-cuda-eager | 1 | 15 | 8.610 | 0.1152 |
| svd_full_tall_f32_512x256 | tenferro-cuda-eager | 1 | 15 | 9.094 | 0.1022 |
| svd_full_wide_f32_256x512 | tenferro-cuda-eager | 1 | 15 | 8.860 | 0.1056 |
| slogdet_positive_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 2.970 | 0.0005003 |
| slogdet_negative_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 2.989 | 0.0005003 |
| svd_full_square_f64_256x256 | tenferro-cuda-eager | 1 | 15 | 161.482 | 8.171e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 128.643 | 7.76e-05 |
| svd_full_wide_f64_256x512 | tenferro-cuda-eager | 1 | 15 | 119.063 | 7.938e-05 |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | 2 | 15 | 2.514 | 1.864e-06 |
| slogdet_negative_f64_256x256 | tenferro-cuda-eager | 2 | 15 | 2.512 | 1.864e-06 |
| svd_full_square_c32_256x256 | tenferro-cuda-eager | 1 | 15 | 13.216 | 0.133 |
| svd_full_tall_c32_512x256 | tenferro-cuda-eager | 1 | 15 | 13.187 | 0.1126 |
| svd_full_wide_c32_256x512 | tenferro-cuda-eager | 1 | 15 | 13.259 | 0.1103 |
| slogdet_positive_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 4.049 | 0.01211 |
| slogdet_negative_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.861 | 0.01211 |
| svd_full_square_c64_256x256 | tenferro-cuda-eager | 1 | 15 | 515.067 | 0.000204 |
| svd_full_tall_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 368.442 | 0.0002024 |
| svd_full_wide_c64_256x512 | tenferro-cuda-eager | 1 | 15 | 369.689 | 0.0001941 |
| slogdet_positive_c64_256x256 | tenferro-cuda-eager | 2 | 15 | 3.903 | 1.025e-06 |
| slogdet_negative_c64_256x256 | tenferro-cuda-eager | 2 | 15 | 3.668 | 1.025e-06 |
| householder_qr_square_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.806 | 0.0004213 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-eager | 8 | 15 | 3.177 | 0.0003161 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-eager | 8 | 15 | 2.745 | 0.0004732 |
| householder_qr_square_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 2.396 | 3.886e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-eager | 4 | 15 | 2.643 | 4.441e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-eager | 4 | 15 | 2.599 | 4.441e-07 |
| householder_qr_square_c32_32x32 | tenferro-cuda-eager | 8 | 15 | 3.170 | 0.0005631 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-eager | 8 | 15 | 3.320 | 0.0003724 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-eager | 8 | 15 | 2.616 | 0.0006002 |
| slogdet_phase_c32_32x32 | tenferro-cuda-eager | 4 | 15 | 2.416 | 0.0004085 |
| householder_qr_square_c64_32x32 | tenferro-cuda-eager | 4 | 15 | 3.114 | 4.996e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-eager | 4 | 15 | 3.515 | 5.551e-07 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-eager | 4 | 15 | 3.544 | 3.886e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-eager | 4 | 15 | 2.313 | 4.755e-07 |
| householder_qr_square_f32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.679 | 0.0009546 |
| householder_qr_tall_f32_512x256 | tenferro-cuda-eager | 2 | 15 | 2.328 | 0.001435 |
| householder_qr_wide_f32_256x512 | tenferro-cuda-eager | 2 | 15 | 2.714 | 0.001037 |
| householder_qr_square_f64_256x256 | tenferro-cuda-eager | 1 | 15 | 3.206 | 1.098e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 3.603 | 9.992e-07 |
| householder_qr_wide_f64_256x512 | tenferro-cuda-eager | 1 | 15 | 5.192 | 1.649e-06 |
| householder_qr_square_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.829 | 0.001198 |
| householder_qr_tall_c32_512x256 | tenferro-cuda-eager | 2 | 15 | 2.611 | 0.001185 |
| householder_qr_wide_c32_256x512 | tenferro-cuda-eager | 2 | 15 | 2.966 | 0.001329 |
| slogdet_phase_c32_256x256 | tenferro-cuda-eager | 4 | 15 | 3.213 | 0.01211 |
| householder_qr_square_c64_256x256 | tenferro-cuda-eager | 1 | 15 | 5.458 | 1.208e-06 |
| householder_qr_tall_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 5.771 | 1.11e-06 |
| householder_qr_wide_c64_256x512 | tenferro-cuda-eager | 1 | 15 | 9.782 | 1.206e-06 |
| slogdet_phase_c64_256x256 | tenferro-cuda-eager | 2 | 15 | 3.478 | 1.025e-06 |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-eager | 4 | 15 | 2.669 | 0.0002423 |
| householder_append_default_f32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f32_64x32 | tenferro-cuda-eager | 64 | 15 | 2.110 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-eager | 4 | 15 | 3.898 | 3.886e-07 |
| householder_append_default_f64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f64_64x32 | tenferro-cuda-eager | 64 | 15 | 2.123 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-eager | 4 | 15 | 2.477 | 0.0003232 |
| householder_append_default_c32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c32_64x32 | tenferro-cuda-eager | 64 | 15 | 2.251 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-eager | 2 | 15 | 3.024 | 4.441e-07 |
| householder_append_default_c64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c64_64x32 | tenferro-cuda-eager | 64 | 15 | 4.144 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f32_512x256 | tenferro-cuda-eager | 2 | 15 | 3.048 | 0.0008522 |
| householder_append_default_f32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f32_512x256 | tenferro-cuda-eager | 128 | 15 | 3.938 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 4.222 | 8.882e-07 |
| householder_append_default_f64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | 128 | 15 | 3.388 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_512x256 | tenferro-cuda-eager | 2 | 15 | 3.141 | 0.001103 |
| householder_append_default_c32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c32_512x256 | tenferro-cuda-eager | 128 | 15 | 3.759 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c64_512x256 | tenferro-cuda-eager | 1 | 15 | 7.332 | 9.992e-07 |
| householder_append_default_c64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c64_512x256 | tenferro-cuda-eager | 128 | 15 | 3.385 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_512x256 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| lu_square_f32_s32 | tenferro-cuda-eager | 8 | 15 | 2.991 | 0.0003557 |
| lu_square_f64_s32 | tenferro-cuda-eager | 8 | 15 | 2.837 | 4.397e-07 |
| lu_square_c32_s32 | tenferro-cuda-eager | 8 | 15 | 2.721 | 0.0002381 |
| lu_square_c64_s32 | tenferro-cuda-eager | 8 | 15 | 3.062 | 5.532e-07 |
| lu_tall_f32_s32 | tenferro-cuda-eager | 8 | 15 | 2.853 | 0.0003557 |
| lu_tall_f64_s32 | tenferro-cuda-eager | 8 | 15 | 3.352 | 4.397e-07 |
| lu_tall_c32_s32 | tenferro-cuda-eager | 8 | 15 | 3.251 | 0.0005956 |
| lu_tall_c64_s32 | tenferro-cuda-eager | 8 | 15 | 3.610 | 4.406e-07 |
| lu_wide_f32_s32 | tenferro-cuda-eager | 8 | 15 | 3.280 | 0.00024 |
| lu_wide_f64_s32 | tenferro-cuda-eager | 8 | 15 | 3.581 | 2.214e-07 |
| lu_wide_c32_s32 | tenferro-cuda-eager | 8 | 15 | 3.578 | 0.0003567 |
| lu_wide_c64_s32 | tenferro-cuda-eager | 8 | 15 | 3.852 | 3.323e-07 |
| lu_square_f32_s256 | tenferro-cuda-eager | 4 | 15 | 2.820 | 0.001428 |
| lu_square_f64_s256 | tenferro-cuda-eager | 2 | 15 | 2.087 | 1.102e-06 |
| lu_square_c32_s256 | tenferro-cuda-eager | 4 | 15 | 2.984 | 0.001541 |
| lu_square_c64_s256 | tenferro-cuda-eager | 2 | 15 | 3.370 | 1.431e-06 |
| lu_tall_f32_s256 | tenferro-cuda-eager | 4 | 15 | 3.692 | 0.001428 |
| lu_tall_f64_s256 | tenferro-cuda-eager | 2 | 15 | 3.237 | 1.102e-06 |
| lu_tall_c32_s256 | tenferro-cuda-eager | 4 | 15 | 3.881 | 0.002239 |
| lu_tall_c64_s256 | tenferro-cuda-eager | 1 | 15 | 2.759 | 1.542e-06 |
| lu_wide_f32_s256 | tenferro-cuda-eager | 4 | 15 | 2.978 | 0.001297 |
| lu_wide_f64_s256 | tenferro-cuda-eager | 2 | 15 | 2.481 | 8.806e-07 |
| lu_wide_c32_s256 | tenferro-cuda-eager | 4 | 15 | 3.214 | 0.002021 |
| lu_wide_c64_s256 | tenferro-cuda-eager | 1 | 15 | 2.375 | 1.433e-06 |
| cholesky_f32_32x32 | pytorch-cuda | 32 | 15 | 2.795 | 0.0002352 |
| cholesky_f64_32x32 | pytorch-cuda | 16 | 15 | 2.588 | 2.188e-07 |
| svdvals_f32_32x32 | pytorch-cuda | 8 | 15 | 3.251 | 0.007749 |
| svdvals_f64_32x32 | pytorch-cuda | 1 | 15 | 5.086 | 3.387e-06 |
| eigvalsh_f32_32x32 | pytorch-cuda | 16 | 15 | 3.498 | 0.001073 |
| eigvalsh_f64_32x32 | pytorch-cuda | 4 | 15 | 2.434 | 5.433e-07 |
| pinv_f32_32x32 | pytorch-cuda | 4 | 15 | 2.131 | 0.001023 |
| pinv_f64_32x32 | pytorch-cuda | 1 | 15 | 4.918 | 9.374e-07 |
| inv_f32_32x32 | pytorch-cuda | 8 | 15 | 2.243 | 0.0004905 |
| inv_f64_32x32 | pytorch-cuda | 8 | 15 | 2.270 | 2.284e-07 |
| det_f32_32x32 | pytorch-cuda | 16 | 15 | 2.604 | 0.002641 |
| det_f64_32x32 | pytorch-cuda | 16 | 15 | 2.608 | 3.784e-07 |
| norm_f32_32x32 | pytorch-cuda | 256 | 15 | 3.558 | 0 |
| norm_f64_32x32 | pytorch-cuda | 256 | 15 | 3.555 | 0 |
| triangular_solve_f32_32x32 | pytorch-cuda | 64 | 15 | 3.585 | 9.242e-05 |
| triangular_solve_f64_32x32 | pytorch-cuda | 64 | 15 | 3.507 | 1.415e-07 |
| cholesky_f32_512x512 | pytorch-cuda | 8 | 15 | 2.239 | 0.0003554 |
| cholesky_f64_512x512 | pytorch-cuda | 1 | 15 | 2.190 | 4.418e-07 |
| svdvals_f32_512x512 | pytorch-cuda | 1 | 15 | 28.091 | 0.1733 |
| svdvals_f64_512x512 | pytorch-cuda | 1 | 15 | 593.344 | 0.0002066 |
| eigvalsh_f32_512x512 | pytorch-cuda | 1 | 15 | 4.358 | 0.008342 |
| eigvalsh_f64_512x512 | pytorch-cuda | 1 | 15 | 14.677 | 1.55e-06 |
| pinv_f32_512x512 | pytorch-cuda | 1 | 15 | 31.576 | 0.01126 |
| pinv_f64_512x512 | pytorch-cuda | 1 | 15 | 681.761 | 8.747e-06 |
| inv_f32_512x512 | pytorch-cuda | 2 | 15 | 3.895 | 0.002107 |
| inv_f64_512x512 | pytorch-cuda | 1 | 15 | 5.996 | 1.85e-06 |
| det_f32_512x512 | pytorch-cuda | 2 | 15 | 2.875 | 0.007965 |
| det_f64_512x512 | pytorch-cuda | 1 | 15 | 2.504 | 1.369e-05 |
| norm_f32_512x512 | pytorch-cuda | 128 | 15 | 2.623 | 0 |
| norm_f64_512x512 | pytorch-cuda | 128 | 15 | 2.518 | 1.909e-07 |
| triangular_solve_f32_512x512 | pytorch-cuda | 32 | 15 | 3.317 | 0.0006008 |
| triangular_solve_f64_512x512 | pytorch-cuda | 4 | 15 | 3.380 | 6.887e-07 |
| svd_full_square_f32_32x32 | pytorch-cuda | 8 | 15 | 2.723 | 0.005571 |
| svd_full_tall_f32_64x32 | pytorch-cuda | 4 | 15 | 3.333 | 0.005287 |
| svd_full_wide_f32_32x64 | pytorch-cuda | 4 | 15 | 3.554 | 0.005456 |
| slogdet_positive_f32_32x32 | pytorch-cuda | 16 | 15 | 3.309 | 0.000915 |
| slogdet_negative_f32_32x32 | pytorch-cuda | 16 | 15 | 3.321 | 0.000915 |
| svd_full_square_f64_32x32 | pytorch-cuda | 1 | 15 | 4.282 | 7.216e-06 |
| svd_full_tall_f64_64x32 | pytorch-cuda | 1 | 15 | 5.971 | 8.826e-06 |
| svd_full_wide_f64_32x64 | pytorch-cuda | 1 | 15 | 6.312 | 8.438e-06 |
| slogdet_positive_f64_32x32 | pytorch-cuda | 16 | 15 | 3.308 | 4.261e-07 |
| slogdet_negative_f64_32x32 | pytorch-cuda | 16 | 15 | 3.450 | 4.261e-07 |
| svd_full_square_c32_32x32 | pytorch-cuda | 8 | 15 | 3.780 | 0.011 |
| svd_full_tall_c32_64x32 | pytorch-cuda | 2 | 15 | 2.092 | 0.008819 |
| svd_full_wide_c32_32x64 | pytorch-cuda | 2 | 15 | 2.108 | 0.008905 |
| slogdet_positive_c32_32x32 | pytorch-cuda | 16 | 15 | 3.523 | 0.0004085 |
| slogdet_negative_c32_32x32 | pytorch-cuda | 16 | 15 | 3.531 | 0.0004085 |
| svd_full_square_c64_32x32 | pytorch-cuda | 1 | 15 | 11.397 | 9.992e-06 |
| svd_full_tall_c64_64x32 | pytorch-cuda | 1 | 15 | 15.449 | 1.188e-05 |
| svd_full_wide_c64_32x64 | pytorch-cuda | 1 | 15 | 16.183 | 1.299e-05 |
| slogdet_positive_c64_32x32 | pytorch-cuda | 16 | 15 | 3.587 | 4.755e-07 |
| slogdet_negative_c64_32x32 | pytorch-cuda | 16 | 15 | 3.581 | 4.755e-07 |
| svd_full_square_f32_256x256 | pytorch-cuda | 1 | 15 | 9.066 | 0.1135 |
| svd_full_tall_f32_512x256 | pytorch-cuda | 1 | 15 | 9.585 | 0.1033 |
| svd_full_wide_f32_256x512 | pytorch-cuda | 1 | 15 | 8.969 | 0.106 |
| slogdet_positive_f32_256x256 | pytorch-cuda | 8 | 15 | 2.976 | 0.0005003 |
| slogdet_negative_f32_256x256 | pytorch-cuda | 8 | 15 | 2.979 | 0.0005003 |
| svd_full_square_f64_256x256 | pytorch-cuda | 1 | 15 | 165.647 | 8.105e-05 |
| svd_full_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 131.148 | 7.76e-05 |
| svd_full_wide_f64_256x512 | pytorch-cuda | 1 | 15 | 121.652 | 7.938e-05 |
| slogdet_positive_f64_256x256 | pytorch-cuda | 4 | 15 | 2.891 | 1.708e-06 |
| slogdet_negative_f64_256x256 | pytorch-cuda | 4 | 15 | 2.890 | 1.708e-06 |
| svd_full_square_c32_256x256 | pytorch-cuda | 1 | 15 | 14.282 | 0.1305 |
| svd_full_tall_c32_512x256 | pytorch-cuda | 1 | 15 | 13.374 | 0.1089 |
| svd_full_wide_c32_256x512 | pytorch-cuda | 1 | 15 | 13.516 | 0.108 |
| slogdet_positive_c32_256x256 | pytorch-cuda | 8 | 15 | 3.275 | 0.01211 |
| slogdet_negative_c32_256x256 | pytorch-cuda | 8 | 15 | 3.280 | 0.01211 |
| svd_full_square_c64_256x256 | pytorch-cuda | 1 | 15 | 545.355 | 0.0002064 |
| svd_full_tall_c64_512x256 | pytorch-cuda | 1 | 15 | 372.769 | 0.0001994 |
| svd_full_wide_c64_256x512 | pytorch-cuda | 1 | 15 | 375.985 | 0.0001943 |
| slogdet_positive_c64_256x256 | pytorch-cuda | 2 | 15 | 2.797 | 1.23e-06 |
| slogdet_negative_c64_256x256 | pytorch-cuda | 2 | 15 | 2.797 | 1.23e-06 |
| householder_qr_square_f32_32x32 | pytorch-cuda | 64 | 15 | 3.835 | 0.000367 |
| householder_qr_tall_f32_64x32 | pytorch-cuda | 32 | 15 | 3.363 | 0.0003269 |
| householder_qr_wide_f32_32x64 | pytorch-cuda | 32 | 15 | 3.375 | 0.0003745 |
| householder_qr_square_f64_32x32 | pytorch-cuda | 8 | 15 | 2.297 | 6.106e-07 |
| householder_qr_tall_f64_64x32 | pytorch-cuda | 8 | 15 | 2.853 | 6.661e-07 |
| householder_qr_wide_f64_32x64 | pytorch-cuda | 8 | 15 | 2.886 | 5.551e-07 |
| householder_qr_square_c32_32x32 | pytorch-cuda | 32 | 15 | 2.176 | 0.0005833 |
| householder_qr_tall_c32_64x32 | pytorch-cuda | 32 | 15 | 3.577 | 0.0003756 |
| householder_qr_wide_c32_32x64 | pytorch-cuda | 32 | 15 | 3.400 | 0.0004972 |
| slogdet_phase_c32_32x32 | pytorch-cuda | 16 | 15 | 3.591 | 0.0004085 |
| householder_qr_square_c64_32x32 | pytorch-cuda | 8 | 15 | 3.856 | 1.221e-06 |
| householder_qr_tall_c64_64x32 | pytorch-cuda | 4 | 15 | 2.314 | 9.992e-07 |
| householder_qr_wide_c64_32x64 | pytorch-cuda | 4 | 15 | 2.491 | 1.166e-06 |
| slogdet_phase_c64_32x32 | pytorch-cuda | 16 | 15 | 3.643 | 5.706e-07 |
| householder_qr_square_f32_256x256 | pytorch-cuda | 4 | 15 | 3.027 | 0.001043 |
| householder_qr_tall_f32_512x256 | pytorch-cuda | 2 | 15 | 2.089 | 0.001475 |
| householder_qr_wide_f32_256x512 | pytorch-cuda | 2 | 15 | 2.480 | 0.0009736 |
| householder_qr_square_f64_256x256 | pytorch-cuda | 1 | 15 | 2.912 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 3.297 | 1.887e-06 |
| householder_qr_wide_f64_256x512 | pytorch-cuda | 1 | 15 | 4.797 | 1.759e-06 |
| householder_qr_square_c32_256x256 | pytorch-cuda | 4 | 15 | 3.158 | 0.001258 |
| householder_qr_tall_c32_512x256 | pytorch-cuda | 2 | 15 | 2.263 | 0.001161 |
| householder_qr_wide_c32_256x512 | pytorch-cuda | 2 | 15 | 2.600 | 0.001226 |
| slogdet_phase_c32_256x256 | pytorch-cuda | 8 | 15 | 3.255 | 0.01233 |
| householder_qr_square_c64_256x256 | pytorch-cuda | 1 | 15 | 5.399 | 2.109e-06 |
| householder_qr_tall_c64_512x256 | pytorch-cuda | 1 | 15 | 5.678 | 2.942e-06 |
| householder_qr_wide_c64_256x512 | pytorch-cuda | 1 | 15 | 9.662 | 2.276e-06 |
| slogdet_phase_c64_256x256 | pytorch-cuda | 2 | 15 | 2.785 | 1.025e-06 |
| householder_from_factors_default_f32_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_64x32 | pytorch-cuda | 256 | 15 | 2.045 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | pytorch-cuda | 32 | 15 | 3.156 | 0.0001196 |
| householder_q_columns_complement_f32_64x32 | pytorch-cuda | 16 | 15 | 2.900 | 0.0001791 |
| householder_from_factors_default_f64_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_64x32 | pytorch-cuda | 256 | 15 | 2.035 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | pytorch-cuda | 32 | 15 | 3.322 | 1.114e-07 |
| householder_q_columns_complement_f64_64x32 | pytorch-cuda | 16 | 15 | 3.672 | 2.724e-08 |
| householder_from_factors_default_c32_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_64x32 | pytorch-cuda | 256 | 15 | 2.237 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | pytorch-cuda | 32 | 15 | 3.911 | 0.00012 |
| householder_q_columns_complement_c32_64x32 | pytorch-cuda | 16 | 15 | 2.878 | 0.000299 |
| householder_from_factors_default_c64_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_64x32 | pytorch-cuda | 256 | 15 | 2.031 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | pytorch-cuda | 32 | 15 | 3.535 | 1.676e-07 |
| householder_q_columns_complement_c64_64x32 | pytorch-cuda | 8 | 15 | 3.477 | 3.341e-07 |
| householder_from_factors_default_f32_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_512x256 | pytorch-cuda | 256 | 15 | 2.066 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | pytorch-cuda | 32 | 15 | 3.180 | 0.0001829 |
| householder_q_columns_complement_f32_512x256 | pytorch-cuda | 16 | 15 | 3.276 | 8.761e-05 |
| householder_from_factors_default_f64_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_512x256 | pytorch-cuda | 256 | 15 | 2.072 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | 16 | 15 | 3.084 | 1.136e-07 |
| householder_q_columns_complement_f64_512x256 | pytorch-cuda | 2 | 15 | 3.083 | 7.26e-07 |
| householder_from_factors_default_c32_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_512x256 | pytorch-cuda | 256 | 15 | 2.232 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | pytorch-cuda | 32 | 15 | 3.913 | 0.0001246 |
| householder_q_columns_complement_c32_512x256 | pytorch-cuda | 8 | 15 | 2.057 | 0.0001149 |
| householder_from_factors_default_c64_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_512x256 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_512x256 | pytorch-cuda | 256 | 15 | 2.254 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | pytorch-cuda | 8 | 15 | 3.007 | 1.154e-07 |
| householder_q_columns_complement_c64_512x256 | pytorch-cuda | 1 | 15 | 4.940 | 7.282e-07 |
| lu_square_f32_s32 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s32 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s32 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s32 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s32 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s32 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s32 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s32 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s32 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s32 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s32 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s32 | pytorch-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
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
| cholesky_f32_32x32 | jax-cuda | 16 | 15 | 2.907 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda | 16 | 15 | 3.197 | 2.188e-07 |
| svdvals_f32_32x32 | jax-cuda | 4 | 15 | 3.446 | 0.003338 |
| svdvals_f64_32x32 | jax-cuda | 1 | 15 | 5.870 | 4.546e-06 |
| eigvalsh_f32_32x32 | jax-cuda | 4 | 15 | 2.343 | 0.004775 |
| eigvalsh_f64_32x32 | jax-cuda | 1 | 15 | 2.950 | 5.542e-06 |
| pinv_f32_32x32 | jax-cuda | 2 | 15 | 2.068 | 0.0006436 |
| pinv_f64_32x32 | jax-cuda | 1 | 15 | 5.607 | 1.367e-06 |
| inv_f32_32x32 | jax-cuda | 16 | 15 | 2.525 | 0.0004905 |
| inv_f64_32x32 | jax-cuda | 16 | 15 | 3.395 | 2.284e-07 |
| det_f32_32x32 | jax-cuda | 16 | 15 | 2.497 | 0.002641 |
| det_f64_32x32 | jax-cuda | 16 | 15 | 2.546 | 1.135e-06 |
| norm_f32_32x32 | jax-cuda | 32 | 15 | 3.224 | 0 |
| norm_f64_32x32 | jax-cuda | 16 | 15 | 2.115 | 0 |
| triangular_solve_f32_32x32 | jax-cuda | 16 | 15 | 2.956 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda | 16 | 15 | 3.509 | 1.609e-07 |
| cholesky_f32_512x512 | jax-cuda | 8 | 15 | 2.185 | 0.0003554 |
| cholesky_f64_512x512 | jax-cuda | 1 | 15 | 2.762 | 4.418e-07 |
| svdvals_f32_512x512 | jax-cuda | 1 | 15 | 25.956 | 0.1547 |
| svdvals_f64_512x512 | jax-cuda | 1 | 15 | 539.422 | 0.0002063 |
| eigvalsh_f32_512x512 | jax-cuda | 1 | 15 | 5.020 | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda | 1 | 15 | 19.608 | 1.55e-06 |
| pinv_f32_512x512 | jax-cuda | 1 | 15 | 29.188 | 0.009647 |
| pinv_f64_512x512 | jax-cuda | 1 | 15 | 617.697 | 8.975e-06 |
| inv_f32_512x512 | jax-cuda | 1 | 15 | 2.234 | 0.002107 |
| inv_f64_512x512 | jax-cuda | 1 | 15 | 6.396 | 1.85e-06 |
| det_f32_512x512 | jax-cuda | 2 | 15 | 3.227 | 0.004748 |
| det_f64_512x512 | jax-cuda | 1 | 15 | 2.889 | 1.426e-05 |
| norm_f32_512x512 | jax-cuda | 16 | 15 | 2.545 | 0 |
| norm_f64_512x512 | jax-cuda | 16 | 15 | 2.584 | 1.909e-07 |
| triangular_solve_f32_512x512 | jax-cuda | 16 | 15 | 3.510 | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda | 16 | 15 | 3.288 | 1.157e-06 |
| svd_full_square_f32_32x32 | jax-cuda | 4 | 15 | 3.194 | 0.008731 |
| svd_full_tall_f32_64x32 | jax-cuda | 2 | 15 | 2.853 | 0.004865 |
| svd_full_wide_f32_32x64 | jax-cuda | 2 | 15 | 2.798 | 0.005495 |
| slogdet_positive_f32_32x32 | jax-cuda | 16 | 15 | 2.641 | 0.000915 |
| slogdet_negative_f32_32x32 | jax-cuda | 16 | 15 | 2.541 | 0.000915 |
| svd_full_square_f64_32x32 | jax-cuda | 1 | 15 | 5.578 | 1.008e-05 |
| svd_full_tall_f64_64x32 | jax-cuda | 1 | 15 | 6.186 | 8.327e-06 |
| svd_full_wide_f64_32x64 | jax-cuda | 1 | 15 | 6.228 | 1.197e-05 |
| slogdet_positive_f64_32x32 | jax-cuda | 8 | 15 | 2.178 | 4.261e-07 |
| slogdet_negative_f64_32x32 | jax-cuda | 8 | 15 | 1.795 | 4.261e-07 |
| svd_full_square_c32_32x32 | jax-cuda | 2 | 15 | 2.088 | 0.009079 |
| svd_full_tall_c32_64x32 | jax-cuda | 2 | 15 | 3.029 | 0.009498 |
| svd_full_wide_c32_32x64 | jax-cuda | 2 | 15 | 3.138 | 0.008495 |
| slogdet_positive_c32_32x32 | jax-cuda | 8 | 15 | 2.314 | 0.0004085 |
| slogdet_negative_c32_32x32 | jax-cuda | 8 | 15 | 1.971 | 0.0004085 |
| svd_full_square_c64_32x32 | jax-cuda | 1 | 15 | 14.456 | 9.215e-06 |
| svd_full_tall_c64_64x32 | jax-cuda | 1 | 15 | 15.355 | 1.243e-05 |
| svd_full_wide_c64_32x64 | jax-cuda | 1 | 15 | 16.045 | 1.288e-05 |
| slogdet_positive_c64_32x32 | jax-cuda | 16 | 15 | 3.058 | 4.755e-07 |
| slogdet_negative_c64_32x32 | jax-cuda | 16 | 15 | 2.710 | 4.755e-07 |
| svd_full_square_f32_256x256 | jax-cuda | 1 | 15 | 8.764 | 0.1152 |
| svd_full_tall_f32_512x256 | jax-cuda | 1 | 15 | 9.193 | 0.1022 |
| svd_full_wide_f32_256x512 | jax-cuda | 1 | 15 | 8.956 | 0.1056 |
| slogdet_positive_f32_256x256 | jax-cuda | 8 | 15 | 3.080 | 0.0005003 |
| slogdet_negative_f32_256x256 | jax-cuda | 8 | 15 | 3.161 | 0.0005003 |
| svd_full_square_f64_256x256 | jax-cuda | 1 | 15 | 160.794 | 8.171e-05 |
| svd_full_tall_f64_512x256 | jax-cuda | 1 | 15 | 128.178 | 7.76e-05 |
| svd_full_wide_f64_256x512 | jax-cuda | 1 | 15 | 118.657 | 7.938e-05 |
| slogdet_positive_f64_256x256 | jax-cuda | 4 | 15 | 3.207 | 1.864e-06 |
| slogdet_negative_f64_256x256 | jax-cuda | 4 | 15 | 3.247 | 1.864e-06 |
| svd_full_square_c32_256x256 | jax-cuda | 1 | 15 | 13.256 | 0.133 |
| svd_full_tall_c32_512x256 | jax-cuda | 1 | 15 | 13.245 | 0.1126 |
| svd_full_wide_c32_256x512 | jax-cuda | 1 | 15 | 13.186 | 0.1103 |
| slogdet_positive_c32_256x256 | jax-cuda | 8 | 15 | 3.370 | 0.01211 |
| slogdet_negative_c32_256x256 | jax-cuda | 8 | 15 | 3.405 | 0.01211 |
| svd_full_square_c64_256x256 | jax-cuda | 1 | 15 | 512.836 | 0.000204 |
| svd_full_tall_c64_512x256 | jax-cuda | 1 | 15 | 366.828 | 0.0002021 |
| svd_full_wide_c64_256x512 | jax-cuda | 1 | 15 | 368.159 | 0.0001946 |
| slogdet_positive_c64_256x256 | jax-cuda | 2 | 15 | 3.175 | 1.025e-06 |
| slogdet_negative_c64_256x256 | jax-cuda | 2 | 15 | 3.198 | 1.025e-06 |
| householder_qr_square_f32_32x32 | jax-cuda | 16 | 15 | 2.104 | 0.000367 |
| householder_qr_tall_f32_64x32 | jax-cuda | 8 | 15 | 2.188 | 0.0003269 |
| householder_qr_wide_f32_32x64 | jax-cuda | 16 | 15 | 2.919 | 0.0003745 |
| householder_qr_square_f64_32x32 | jax-cuda | 8 | 15 | 2.514 | 6.106e-07 |
| householder_qr_tall_f64_64x32 | jax-cuda | 8 | 15 | 3.252 | 6.661e-07 |
| householder_qr_wide_f64_32x64 | jax-cuda | 8 | 15 | 3.186 | 5.551e-07 |
| householder_qr_square_c32_32x32 | jax-cuda | 16 | 15 | 2.987 | 0.0005833 |
| householder_qr_tall_c32_64x32 | jax-cuda | 8 | 15 | 3.015 | 0.0003756 |
| householder_qr_wide_c32_32x64 | jax-cuda | 8 | 15 | 2.891 | 0.0004972 |
| slogdet_phase_c32_32x32 | jax-cuda | 8 | 15 | 2.134 | 0.0004085 |
| householder_qr_square_c64_32x32 | jax-cuda | 4 | 15 | 2.417 | 1.221e-06 |
| householder_qr_tall_c64_64x32 | jax-cuda | 4 | 15 | 2.777 | 9.992e-07 |
| householder_qr_wide_c64_32x64 | jax-cuda | 4 | 15 | 3.061 | 1.166e-06 |
| slogdet_phase_c64_32x32 | jax-cuda | 8 | 15 | 2.461 | 5.706e-07 |
| householder_qr_square_f32_256x256 | jax-cuda | 2 | 15 | 2.114 | 0.001043 |
| householder_qr_tall_f32_512x256 | jax-cuda | 2 | 15 | 2.751 | 0.001475 |
| householder_qr_wide_f32_256x512 | jax-cuda | 2 | 15 | 3.044 | 0.0009736 |
| householder_qr_square_f64_256x256 | jax-cuda | 1 | 15 | 3.553 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | jax-cuda | 1 | 15 | 3.941 | 1.887e-06 |
| householder_qr_wide_f64_256x512 | jax-cuda | 1 | 15 | 5.556 | 1.759e-06 |
| householder_qr_square_c32_256x256 | jax-cuda | 2 | 15 | 2.031 | 0.001258 |
| householder_qr_tall_c32_512x256 | jax-cuda | 2 | 15 | 2.549 | 0.001161 |
| householder_qr_wide_c32_256x512 | jax-cuda | 2 | 15 | 3.132 | 0.001226 |
| slogdet_phase_c32_256x256 | jax-cuda | 8 | 15 | 3.567 | 0.01233 |
| householder_qr_square_c64_256x256 | jax-cuda | 1 | 15 | 5.841 | 2.109e-06 |
| householder_qr_tall_c64_512x256 | jax-cuda | 1 | 15 | 6.102 | 2.942e-06 |
| householder_qr_wide_c64_256x512 | jax-cuda | 1 | 15 | 10.115 | 2.276e-06 |
| slogdet_phase_c64_256x256 | jax-cuda | 2 | 15 | 3.154 | 1.025e-06 |
| householder_from_factors_default_f32_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_64x32 | jax-cuda | 16 | 15 | 2.124 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | jax-cuda | 8 | 15 | 2.953 | 0.0001196 |
| householder_q_columns_complement_f32_64x32 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f64_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_64x32 | jax-cuda | 16 | 15 | 2.181 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | jax-cuda | 8 | 15 | 3.562 | 1.114e-07 |
| householder_q_columns_complement_f64_64x32 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c32_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_64x32 | jax-cuda | 16 | 15 | 2.495 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | jax-cuda | 4 | 15 | 2.013 | 0.00012 |
| householder_q_columns_complement_c32_64x32 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c64_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_64x32 | jax-cuda | 16 | 15 | 2.326 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | jax-cuda | 8 | 15 | 3.679 | 1.676e-07 |
| householder_q_columns_complement_c64_64x32 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f32_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_512x256 | jax-cuda | 32 | 15 | 3.352 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | jax-cuda | 8 | 15 | 3.521 | 0.0001829 |
| householder_q_columns_complement_f32_512x256 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f64_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_512x256 | jax-cuda | 16 | 15 | 2.507 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | 8 | 15 | 3.680 | 1.136e-07 |
| householder_q_columns_complement_f64_512x256 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c32_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_512x256 | jax-cuda | 16 | 15 | 2.710 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | jax-cuda | 4 | 15 | 1.986 | 0.0001246 |
| householder_q_columns_complement_c32_512x256 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c64_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_512x256 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_512x256 | jax-cuda | 16 | 15 | 2.590 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | jax-cuda | 4 | 15 | 2.352 | 1.154e-07 |
| householder_q_columns_complement_c64_512x256 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| lu_square_f32_s32 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s32 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s32 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s32 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s32 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s32 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s32 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s32 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s32 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s32 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s32 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s32 | jax-cuda | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
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
| cholesky_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.026 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda-jit | 16 | 15 | 3.391 | 2.188e-07 |
| svdvals_f32_32x32 | jax-cuda-jit | 4 | 15 | 3.103 | 0.003338 |
| svdvals_f64_32x32 | jax-cuda-jit | 1 | 15 | 5.811 | 4.546e-06 |
| eigvalsh_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.496 | 0.004775 |
| eigvalsh_f64_32x32 | jax-cuda-jit | 1 | 15 | 2.953 | 5.542e-06 |
| pinv_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.910 | 0.0006436 |
| pinv_f64_32x32 | jax-cuda-jit | 1 | 15 | 5.435 | 1.367e-06 |
| inv_f32_32x32 | jax-cuda-jit | 16 | 15 | 3.973 | 0.0004905 |
| inv_f64_32x32 | jax-cuda-jit | 8 | 15 | 2.631 | 2.284e-07 |
| det_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.679 | 0.002641 |
| det_f64_32x32 | jax-cuda-jit | 8 | 15 | 2.074 | 1.135e-06 |
| norm_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.775 | 0 |
| norm_f64_32x32 | jax-cuda-jit | 16 | 15 | 2.771 | 0 |
| triangular_solve_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.710 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda-jit | 16 | 15 | 2.572 | 1.609e-07 |
| cholesky_f32_512x512 | jax-cuda-jit | 8 | 15 | 2.153 | 0.0003554 |
| cholesky_f64_512x512 | jax-cuda-jit | 1 | 15 | 2.749 | 4.418e-07 |
| svdvals_f32_512x512 | jax-cuda-jit | 1 | 15 | 25.966 | 0.1547 |
| svdvals_f64_512x512 | jax-cuda-jit | 1 | 15 | 539.572 | 0.0002063 |
| eigvalsh_f32_512x512 | jax-cuda-jit | 1 | 15 | 5.002 | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda-jit | 1 | 15 | 19.571 | 1.55e-06 |
| pinv_f32_512x512 | jax-cuda-jit | 1 | 15 | 28.737 | 0.009647 |
| pinv_f64_512x512 | jax-cuda-jit | 1 | 15 | 617.322 | 8.975e-06 |
| inv_f32_512x512 | jax-cuda-jit | 1 | 15 | 2.321 | 0.002107 |
| inv_f64_512x512 | jax-cuda-jit | 1 | 15 | 6.419 | 1.85e-06 |
| det_f32_512x512 | jax-cuda-jit | 2 | 15 | 3.253 | 0.004748 |
| det_f64_512x512 | jax-cuda-jit | 1 | 15 | 2.794 | 1.426e-05 |
| norm_f32_512x512 | jax-cuda-jit | 16 | 15 | 2.656 | 0 |
| norm_f64_512x512 | jax-cuda-jit | 16 | 15 | 2.691 | 1.909e-07 |
| triangular_solve_f32_512x512 | jax-cuda-jit | 16 | 15 | 3.228 | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda-jit | 16 | 15 | 3.486 | 1.157e-06 |
| svd_full_square_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.911 | 0.008731 |
| svd_full_tall_f32_64x32 | jax-cuda-jit | 2 | 15 | 2.207 | 0.004865 |
| svd_full_wide_f32_32x64 | jax-cuda-jit | 2 | 15 | 2.958 | 0.005495 |
| slogdet_positive_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.951 | 0.000915 |
| slogdet_negative_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.806 | 0.000915 |
| svd_full_square_f64_32x32 | jax-cuda-jit | 1 | 15 | 5.732 | 1.008e-05 |
| svd_full_tall_f64_64x32 | jax-cuda-jit | 1 | 15 | 6.097 | 8.327e-06 |
| svd_full_wide_f64_32x64 | jax-cuda-jit | 1 | 15 | 6.269 | 1.197e-05 |
| slogdet_positive_f64_32x32 | jax-cuda-jit | 8 | 15 | 2.111 | 4.261e-07 |
| slogdet_negative_f64_32x32 | jax-cuda-jit | 8 | 15 | 2.003 | 4.261e-07 |
| svd_full_square_c32_32x32 | jax-cuda-jit | 2 | 15 | 2.149 | 0.009079 |
| svd_full_tall_c32_64x32 | jax-cuda-jit | 2 | 15 | 2.969 | 0.009498 |
| svd_full_wide_c32_32x64 | jax-cuda-jit | 2 | 15 | 3.056 | 0.008495 |
| slogdet_positive_c32_32x32 | jax-cuda-jit | 8 | 15 | 2.321 | 0.0004085 |
| slogdet_negative_c32_32x32 | jax-cuda-jit | 8 | 15 | 2.242 | 0.0004085 |
| svd_full_square_c64_32x32 | jax-cuda-jit | 1 | 15 | 14.481 | 9.215e-06 |
| svd_full_tall_c64_64x32 | jax-cuda-jit | 1 | 15 | 15.361 | 1.243e-05 |
| svd_full_wide_c64_32x64 | jax-cuda-jit | 1 | 15 | 16.091 | 1.288e-05 |
| slogdet_positive_c64_32x32 | jax-cuda-jit | 8 | 15 | 1.806 | 4.755e-07 |
| slogdet_negative_c64_32x32 | jax-cuda-jit | 8 | 15 | 2.334 | 4.755e-07 |
| svd_full_square_f32_256x256 | jax-cuda-jit | 1 | 15 | 8.762 | 0.1152 |
| svd_full_tall_f32_512x256 | jax-cuda-jit | 1 | 15 | 9.227 | 0.1022 |
| svd_full_wide_f32_256x512 | jax-cuda-jit | 1 | 15 | 8.977 | 0.1056 |
| slogdet_positive_f32_256x256 | jax-cuda-jit | 8 | 15 | 3.178 | 0.0005003 |
| slogdet_negative_f32_256x256 | jax-cuda-jit | 8 | 15 | 3.177 | 0.0005003 |
| svd_full_square_f64_256x256 | jax-cuda-jit | 1 | 15 | 160.808 | 8.171e-05 |
| svd_full_tall_f64_512x256 | jax-cuda-jit | 1 | 15 | 128.251 | 7.76e-05 |
| svd_full_wide_f64_256x512 | jax-cuda-jit | 1 | 15 | 118.743 | 7.938e-05 |
| slogdet_positive_f64_256x256 | jax-cuda-jit | 4 | 15 | 3.233 | 1.864e-06 |
| slogdet_negative_f64_256x256 | jax-cuda-jit | 4 | 15 | 3.253 | 1.864e-06 |
| svd_full_square_c32_256x256 | jax-cuda-jit | 1 | 15 | 13.279 | 0.133 |
| svd_full_tall_c32_512x256 | jax-cuda-jit | 1 | 15 | 13.263 | 0.1126 |
| svd_full_wide_c32_256x512 | jax-cuda-jit | 1 | 15 | 13.329 | 0.1103 |
| slogdet_positive_c32_256x256 | jax-cuda-jit | 4 | 15 | 1.927 | 0.01211 |
| slogdet_negative_c32_256x256 | jax-cuda-jit | 8 | 15 | 3.463 | 0.01211 |
| svd_full_square_c64_256x256 | jax-cuda-jit | 1 | 15 | 512.815 | 0.000204 |
| svd_full_tall_c64_512x256 | jax-cuda-jit | 1 | 15 | 366.855 | 0.0002021 |
| svd_full_wide_c64_256x512 | jax-cuda-jit | 1 | 15 | 368.158 | 0.0001946 |
| slogdet_positive_c64_256x256 | jax-cuda-jit | 2 | 15 | 3.189 | 1.025e-06 |
| slogdet_negative_c64_256x256 | jax-cuda-jit | 2 | 15 | 3.195 | 1.025e-06 |
| householder_qr_square_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.070 | 0.000367 |
| householder_qr_tall_f32_64x32 | jax-cuda-jit | 16 | 15 | 3.149 | 0.0003269 |
| householder_qr_wide_f32_32x64 | jax-cuda-jit | 8 | 15 | 2.445 | 0.0003745 |
| householder_qr_square_f64_32x32 | jax-cuda-jit | 8 | 15 | 2.580 | 6.106e-07 |
| householder_qr_tall_f64_64x32 | jax-cuda-jit | 8 | 15 | 3.397 | 6.661e-07 |
| householder_qr_wide_f64_32x64 | jax-cuda-jit | 8 | 15 | 3.388 | 5.551e-07 |
| householder_qr_square_c32_32x32 | jax-cuda-jit | 8 | 15 | 2.175 | 0.0005833 |
| householder_qr_tall_c32_64x32 | jax-cuda-jit | 8 | 15 | 3.105 | 0.0003756 |
| householder_qr_wide_c32_32x64 | jax-cuda-jit | 8 | 15 | 2.572 | 0.0004972 |
| slogdet_phase_c32_32x32 | jax-cuda-jit | 8 | 15 | 2.435 | 0.0004085 |
| householder_qr_square_c64_32x32 | jax-cuda-jit | 4 | 15 | 2.496 | 1.221e-06 |
| householder_qr_tall_c64_64x32 | jax-cuda-jit | 4 | 15 | 2.936 | 9.992e-07 |
| householder_qr_wide_c64_32x64 | jax-cuda-jit | 4 | 15 | 3.054 | 1.166e-06 |
| slogdet_phase_c64_32x32 | jax-cuda-jit | 8 | 15 | 2.745 | 5.706e-07 |
| householder_qr_square_f32_256x256 | jax-cuda-jit | 2 | 15 | 2.033 | 0.001043 |
| householder_qr_tall_f32_512x256 | jax-cuda-jit | 2 | 15 | 2.705 | 0.001475 |
| householder_qr_wide_f32_256x512 | jax-cuda-jit | 2 | 15 | 3.013 | 0.0009736 |
| householder_qr_square_f64_256x256 | jax-cuda-jit | 1 | 15 | 3.481 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | jax-cuda-jit | 1 | 15 | 3.910 | 1.887e-06 |
| householder_qr_wide_f64_256x512 | jax-cuda-jit | 1 | 15 | 5.549 | 1.759e-06 |
| householder_qr_square_c32_256x256 | jax-cuda-jit | 4 | 15 | 3.521 | 0.001258 |
| householder_qr_tall_c32_512x256 | jax-cuda-jit | 2 | 15 | 2.542 | 0.001161 |
| householder_qr_wide_c32_256x512 | jax-cuda-jit | 2 | 15 | 3.155 | 0.001226 |
| slogdet_phase_c32_256x256 | jax-cuda-jit | 4 | 15 | 2.027 | 0.01233 |
| householder_qr_square_c64_256x256 | jax-cuda-jit | 1 | 15 | 5.847 | 2.109e-06 |
| householder_qr_tall_c64_512x256 | jax-cuda-jit | 1 | 15 | 6.061 | 2.942e-06 |
| householder_qr_wide_c64_256x512 | jax-cuda-jit | 1 | 15 | 10.123 | 2.276e-06 |
| slogdet_phase_c64_256x256 | jax-cuda-jit | 2 | 15 | 3.145 | 1.025e-06 |
| householder_from_factors_default_f32_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_64x32 | jax-cuda-jit | 16 | 15 | 2.356 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | jax-cuda-jit | 16 | 15 | 3.087 | 0.0001196 |
| householder_q_columns_complement_f32_64x32 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f64_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_64x32 | jax-cuda-jit | 16 | 15 | 2.522 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | jax-cuda-jit | 8 | 15 | 2.876 | 1.114e-07 |
| householder_q_columns_complement_f64_64x32 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c32_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_64x32 | jax-cuda-jit | 16 | 15 | 2.500 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | jax-cuda-jit | 8 | 15 | 3.327 | 0.00012 |
| householder_q_columns_complement_c32_64x32 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c64_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_64x32 | jax-cuda-jit | 16 | 15 | 2.735 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | jax-cuda-jit | 8 | 15 | 3.140 | 1.676e-07 |
| householder_q_columns_complement_c64_64x32 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f32_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_512x256 | jax-cuda-jit | 16 | 15 | 2.702 | 0.001415 |
| householder_q_columns_prefix_f32_512x256 | jax-cuda-jit | 16 | 15 | 3.085 | 0.0001829 |
| householder_q_columns_complement_f32_512x256 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f64_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_512x256 | jax-cuda-jit | 16 | 15 | 2.693 | 9.976e-07 |
| householder_q_columns_prefix_f64_512x256 | jax-cuda-jit | 8 | 15 | 2.860 | 1.136e-07 |
| householder_q_columns_complement_f64_512x256 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c32_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_512x256 | jax-cuda-jit | 16 | 15 | 2.729 | 0.001069 |
| householder_q_columns_prefix_c32_512x256 | jax-cuda-jit | 8 | 15 | 3.309 | 0.0001246 |
| householder_q_columns_complement_c32_512x256 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c64_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_512x256 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_512x256 | jax-cuda-jit | 16 | 15 | 2.697 | 9.985e-07 |
| householder_q_columns_prefix_c64_512x256 | jax-cuda-jit | 4 | 15 | 2.001 | 1.154e-07 |
| householder_q_columns_complement_c64_512x256 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| lu_square_f32_s32 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s32 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s32 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s32 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s32 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s32 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s32 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s32 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s32 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s32 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s32 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s32 | jax-cuda-jit | — | — | — | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
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
timestamp: '2026-10-13T17:28:20Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 0d36b7255b91d1340119a0f2fe300bf29e364dba
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
  expected_cases: 164
  selected_cases: 164
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 897fe5518ea92f3922c9bd33f306f9e6441de45204bcb5e4338aa5ed0ab01df6
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/multi.rs: 1b477fef78260cf28aa31e515d57ab411f6241b91ffb30d625939538fdbc6bab
    src/bin/benchmark_gpu_public_api/read_layout.rs: bd7ce4a42576d1937a7048b7b002cac4838a7cd3c53628bc9aa6aebb0c5c1128
    src/bin/benchmark_gpu_public_api/structural.rs: 8121b12b5fed3b0acbc71e2218856173d53a5acc93092b93c7fd3309d4f45962
    src/bin/benchmark_gpu_public_api/linalg.rs: 84162923fa0d0772c88ec9c4eab169ad3a75e5d16ff76961d1541670dd8565c1
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: a305c36801d5bee0d9843ab6d224918d67b5330006ef997a37d12a36bc95ef7f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: bbf982ad1b55067bf8dabc8676aa8ffe2fe5a1ff769f979b66eca4e2d5eba6e6
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 0d9cd2ce4b5125d16f35427075939e3d8f55954e8f8040032cfeed914de0d518
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/linalg_public_api/20261013_172820/fixtures/manifest.json: e8153a1102e04a6f09305645386390e52b03aa74d9b41adcaa496512183a88c9
    target/release/benchmark_gpu_public_api: 846cd01d6727188e1fb29521776e9a2783de8e8255933de7145e57d7c98fd18b
  reference_numpy_version: 2.4.2
```
