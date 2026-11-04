# gpu/linalg_public_api_latency: CUDA public API benchmark

Measured cells: **308**. Unsupported cells: **102**, including **28** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api/20261102_210935/`.

This page is the per-call overhead view: it holds the 82 cases whose largest matrix dimension is at most 128. Rows are the published measurements of that class from the run above, unchanged; at these sizes launch and synchronization cost dominates and the numbers are not GPU throughput.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | 77.166 (1.596) | 1978.690 (6.582) | 65.961 (0.523) | 72.109 (10.738) | 64.674 (0.310) |
| cholesky_f64_32x32 | 79.861 (0.317) | 1994.211 (6.560) | 69.449 (0.259) | 67.881 (2.499) | 67.038 (0.813) |
| svdvals_f32_32x32 | 693.678 (4.922) | unsupported | 409.947 (1.066) | 710.798 (15.682) | 686.590 (7.848) |
| svdvals_f64_32x32 | 976.350 (5.673) | unsupported | 733.631 (1.096) | 1034.569 (29.311) | 1001.303 (5.310) |
| eigvalsh_f32_32x32 | 227.262 (0.423) | 2115.140 (18.286) | 227.057 (0.880) | 581.822 (18.857) | 562.421 (5.734) |
| eigvalsh_f64_32x32 | 266.959 (0.894) | 2159.351 (17.145) | 270.950 (1.247) | 706.768 (6.689) | 700.096 (4.770) |
| pinv_f32_32x32 | unsupported | unsupported | 509.074 (3.229) | 948.111 (10.462) | 709.124 (3.058) |
| pinv_f64_32x32 | unsupported | unsupported | 839.618 (2.091) | 1267.337 (6.447) | 1022.129 (7.513) |
| inv_f32_32x32 | unsupported | unsupported | 189.402 (0.582) | 109.064 (0.848) | 111.049 (0.861) |
| inv_f64_32x32 | unsupported | unsupported | 196.875 (0.910) | 115.251 (0.267) | 115.590 (0.275) |
| det_f32_32x32 | 379.001 (7.613) | 2043.432 (46.702) | 106.000 (0.476) | 81.060 (0.258) | 82.244 (2.322) |
| det_f64_32x32 | 381.734 (1.114) | 1951.679 (10.520) | 108.085 (0.297) | 84.182 (0.434) | 84.594 (0.204) |
| norm_f32_32x32 | 69.939 (0.291) | 87.640 (2.227) | 6.876 (0.530) | 40.581 (0.244) | 41.276 (0.253) |
| norm_f64_32x32 | 69.689 (0.288) | 95.617 (0.329) | 6.044 (0.594) | 40.132 (0.217) | 41.304 (0.313) |
| triangular_solve_f32_32x32 | 17.820 (0.017) | 1881.052 (5.162) | 23.461 (3.736) | 74.917 (0.274) | 74.006 (0.748) |
| triangular_solve_f64_32x32 | 19.058 (0.016) | 1883.397 (7.198) | 23.631 (2.379) | 75.275 (0.698) | 72.503 (0.623) |
| svd_full_square_f32_32x32 | 656.129 (4.673) | 2521.503 (78.253) | 344.548 (1.644) | 640.884 (8.693) | 638.319 (2.234) |
| svd_full_tall_f32_64x32 | 1048.844 (8.515) | 2897.946 (26.876) | 1053.135 (3.365) | 1035.980 (7.602) | 1033.924 (4.740) |
| svd_full_wide_f32_32x64 | 1121.396 (9.968) | 2919.196 (92.789) | 1130.368 (3.998) | 1115.007 (8.778) | 1109.171 (3.325) |
| slogdet_positive_f32_32x32 | 337.020 (23.751) | 2025.651 (96.456) | 115.329 (9.695) | 91.704 (0.788) | 93.135 (0.965) |
| slogdet_negative_f32_32x32 | 313.583 (1.129) | 1940.639 (12.268) | 116.248 (2.404) | 92.104 (0.360) | 94.097 (1.182) |
| svd_full_square_f64_32x32 | 932.018 (2.805) | 2717.740 (61.948) | 613.365 (2.491) | 942.156 (4.986) | 942.501 (3.644) |
| svd_full_tall_f64_64x32 | 1448.007 (10.626) | 3207.506 (12.301) | 1483.424 (8.171) | 1448.718 (4.105) | 1452.129 (3.847) |
| svd_full_wide_f64_32x64 | 1485.494 (3.726) | 3274.458 (20.246) | 1577.417 (10.405) | 1476.814 (3.938) | 1477.874 (2.520) |
| slogdet_positive_f64_32x32 | 316.890 (1.718) | 1944.809 (51.492) | 134.811 (18.442) | 92.640 (0.852) | 94.488 (0.269) |
| slogdet_negative_f64_32x32 | 318.010 (1.837) | 1936.639 (57.040) | 135.766 (0.828) | 92.544 (1.278) | 94.667 (0.743) |
| svd_full_square_c32_32x32 | 954.324 (5.006) | 2818.342 (17.200) | 596.070 (2.316) | 959.914 (6.815) | 965.409 (5.772) |
| svd_full_tall_c32_64x32 | 1388.505 (19.838) | 3131.743 (19.351) | 1388.846 (3.748) | 1387.010 (4.102) | 1375.520 (2.948) |
| svd_full_wide_c32_32x64 | 1411.771 (15.695) | 3146.134 (17.486) | 1428.337 (7.945) | 1394.211 (6.395) | 1389.111 (3.327) |
| slogdet_positive_c32_32x32 | 347.945 (1.419) | 1940.679 (4.345) | 142.307 (1.152) | 92.076 (0.594) | 92.921 (0.902) |
| slogdet_negative_c32_32x32 | 353.924 (1.425) | 2044.993 (6.060) | 140.675 (0.459) | 92.466 (1.063) | 92.357 (0.591) |
| svd_full_square_c64_32x32 | 1815.959 (6.805) | 3658.020 (26.001) | 1240.171 (3.083) | 1787.564 (4.117) | 1783.769 (3.098) |
| svd_full_tall_c64_64x32 | 2288.085 (4.015) | 4154.366 (43.557) | 2303.796 (5.800) | 2302.977 (5.805) | 2319.376 (18.076) |
| svd_full_wide_c64_32x64 | 2421.269 (5.236) | 4407.675 (83.503) | 2439.131 (10.625) | 2396.000 (2.685) | 2406.279 (7.939) |
| slogdet_positive_c64_32x32 | 357.500 (1.092) | 2041.388 (13.461) | 145.442 (0.379) | 91.645 (0.188) | 92.539 (0.637) |
| slogdet_negative_c64_32x32 | 362.561 (0.772) | 2047.327 (16.515) | 146.061 (0.627) | 91.721 (0.309) | 92.100 (0.177) |
| householder_qr_square_f32_32x32 | 121.186 (1.055) | 1879.527 (30.101) | 83.416 (0.034) | 84.211 (23.117) | 84.082 (1.248) |
| householder_qr_tall_f32_64x32 | 167.897 (5.704) | 1930.409 (24.826) | 125.588 (0.249) | 127.941 (0.242) | 128.263 (0.188) |
| householder_qr_wide_f32_32x64 | 161.362 (2.119) | 1946.120 (13.478) | 125.664 (0.482) | 128.334 (0.203) | 128.221 (0.169) |
| householder_qr_square_f64_32x32 | 137.554 (0.843) | 1927.113 (11.588) | 99.422 (0.038) | 100.590 (0.084) | 100.568 (0.077) |
| householder_qr_tall_f64_64x32 | 180.964 (0.683) | 2061.118 (6.975) | 139.678 (0.108) | 142.565 (0.227) | 143.003 (0.106) |
| householder_qr_wide_f64_32x64 | 184.868 (2.238) | 2050.838 (13.450) | 140.026 (0.116) | 143.581 (0.091) | 143.442 (0.137) |
| householder_qr_square_c32_32x32 | 131.682 (0.254) | 2009.526 (6.356) | 94.877 (0.171) | 95.919 (0.071) | 95.838 (0.075) |
| householder_qr_tall_c32_64x32 | 184.472 (1.105) | 2069.778 (11.020) | 145.143 (0.130) | 148.159 (0.142) | 148.640 (0.188) |
| householder_qr_wide_c32_32x64 | 185.565 (1.768) | 2054.788 (12.371) | 146.326 (0.114) | 149.398 (0.186) | 149.462 (0.113) |
| slogdet_phase_c32_32x32 | 348.090 (15.361) | 2048.938 (9.116) | 126.195 (8.028) | 94.082 (0.427) | 92.122 (0.341) |
| householder_qr_square_c64_32x32 | 160.237 (0.690) | 2014.467 (3.840) | 127.072 (0.102) | 130.774 (0.316) | 130.044 (0.304) |
| householder_qr_tall_c64_64x32 | 213.844 (0.959) | 2093.849 (9.560) | 173.731 (0.164) | 177.908 (0.200) | 177.425 (0.113) |
| householder_qr_wide_c64_32x64 | 222.945 (0.658) | 2098.229 (10.695) | 183.500 (0.231) | 186.712 (0.237) | 186.614 (0.290) |
| slogdet_phase_c64_32x32 | 331.546 (1.008) | 2042.657 (11.030) | 134.466 (13.624) | 92.352 (2.864) | 93.016 (1.384) |
| householder_from_factors_default_f32_64x32 | 360.657 (2.523) | 2124.021 (13.940) | unsupported | unsupported | unsupported |
| householder_append_default_f32_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f32_64x32 | 35.223 (0.215) | 46.532 (0.202) | 4.445 (0.679) | 40.400 (0.368) | 41.069 (0.159) |
| householder_q_columns_prefix_f32_64x32 | unsupported | unsupported | 55.478 (5.400) | 127.008 (0.802) | 99.542 (0.438) |
| householder_q_columns_complement_f32_64x32 | unsupported | unsupported | 250.520 (0.459) | unsupported | unsupported |
| householder_from_factors_default_f64_64x32 | 388.585 (4.728) | 2132.260 (9.394) | unsupported | unsupported | unsupported |
| householder_append_default_f64_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f64_64x32 | 35.981 (0.126) | 48.509 (0.945) | 5.033 (0.614) | 39.905 (0.267) | 41.224 (0.281) |
| householder_q_columns_prefix_f64_64x32 | unsupported | unsupported | 66.708 (0.226) | 130.514 (0.651) | 104.614 (0.444) |
| householder_q_columns_complement_f64_64x32 | unsupported | unsupported | 288.306 (0.411) | unsupported | unsupported |
| householder_from_factors_default_c32_64x32 | 390.408 (4.021) | 2046.777 (28.041) | unsupported | unsupported | unsupported |
| householder_append_default_c32_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c32_64x32 | 36.568 (0.281) | 46.757 (0.095) | 5.183 (0.026) | 40.267 (0.230) | 41.471 (0.323) |
| householder_q_columns_prefix_c32_64x32 | unsupported | unsupported | 62.087 (1.136) | 129.461 (0.859) | 101.273 (0.699) |
| householder_q_columns_complement_c32_64x32 | unsupported | unsupported | 268.524 (0.291) | unsupported | unsupported |
| householder_from_factors_default_c64_64x32 | 399.138 (8.840) | 2085.029 (61.359) | unsupported | unsupported | unsupported |
| householder_append_default_c64_64x32 | unsupported | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_c64_64x32 | 33.437 (0.377) | 42.776 (0.215) | 5.098 (0.062) | 40.297 (0.267) | 41.265 (0.195) |
| householder_q_columns_prefix_c64_64x32 | unsupported | unsupported | 68.008 (0.560) | 132.805 (0.994) | 103.149 (0.656) |
| householder_q_columns_complement_c64_64x32 | unsupported | unsupported | 291.278 (0.149) | unsupported | unsupported |
| lu_square_f32_s32 | 178.243 (0.971) | 1923.284 (16.786) | unsupported | unsupported | unsupported |
| lu_square_f64_s32 | 175.489 (0.871) | 1964.999 (61.205) | unsupported | unsupported | unsupported |
| lu_square_c32_s32 | 182.222 (1.207) | 2020.046 (79.380) | unsupported | unsupported | unsupported |
| lu_square_c64_s32 | 190.700 (0.961) | 2027.277 (86.668) | unsupported | unsupported | unsupported |
| lu_tall_f32_s32 | 177.468 (2.199) | 1927.209 (5.707) | unsupported | unsupported | unsupported |
| lu_tall_f64_s32 | 183.570 (0.707) | 1924.424 (5.770) | unsupported | unsupported | unsupported |
| lu_tall_c32_s32 | 186.599 (0.560) | 1936.309 (8.143) | unsupported | unsupported | unsupported |
| lu_tall_c64_s32 | 188.437 (1.399) | 1937.574 (6.698) | unsupported | unsupported | unsupported |
| lu_wide_f32_s32 | 181.027 (0.971) | 2040.767 (13.926) | unsupported | unsupported | unsupported |
| lu_wide_f64_s32 | 188.882 (0.864) | 1941.529 (81.111) | unsupported | unsupported | unsupported |
| lu_wide_c32_s32 | 190.948 (1.066) | 2023.557 (37.200) | unsupported | unsupported | unsupported |
| lu_wide_c64_s32 | 194.869 (0.666) | 1999.061 (84.598) | unsupported | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.469 | 0.0002352 |
| cholesky_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.556 | 2.188e-07 |
| svdvals_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.775 | 0.003338 |
| svdvals_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 3.905 | 4.546e-06 |
| eigvalsh_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.636 | 0.001073 |
| eigvalsh_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.136 | 5.433e-07 |
| pinv_f32_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| det_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 3.032 | 0.002641 |
| det_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 3.054 | 1.135e-06 |
| norm_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.238 | 0 |
| norm_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.230 | 0 |
| triangular_solve_f32_32x32 | tenferro-cuda-direct | 128 | 15 | 2.281 | 0.0001546 |
| triangular_solve_f64_32x32 | tenferro-cuda-direct | 128 | 15 | 2.439 | 1.609e-07 |
| svd_full_square_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.625 | 0.008731 |
| svd_full_tall_f32_64x32 | tenferro-cuda-direct | 2 | 15 | 2.098 | 0.007661 |
| svd_full_wide_f32_32x64 | tenferro-cuda-direct | 2 | 15 | 2.243 | 0.005157 |
| slogdet_positive_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.696 | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.509 | 0.000915 |
| svd_full_square_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 3.728 | 1.007e-05 |
| svd_full_tall_f64_64x32 | tenferro-cuda-direct | 2 | 15 | 2.896 | 9.548e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-direct | 2 | 15 | 2.971 | 9.838e-06 |
| slogdet_positive_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.535 | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.544 | 4.261e-07 |
| svd_full_square_c32_32x32 | tenferro-cuda-direct | 4 | 15 | 3.817 | 0.009079 |
| svd_full_tall_c32_64x32 | tenferro-cuda-direct | 2 | 15 | 2.777 | 0.009498 |
| svd_full_wide_c32_32x64 | tenferro-cuda-direct | 2 | 15 | 2.824 | 0.008495 |
| slogdet_positive_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.784 | 0.0007148 |
| slogdet_negative_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.831 | 0.0007148 |
| svd_full_square_c64_32x32 | tenferro-cuda-direct | 2 | 15 | 3.632 | 9.215e-06 |
| svd_full_tall_c64_64x32 | tenferro-cuda-direct | 1 | 15 | 2.288 | 1.221e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-direct | 1 | 15 | 2.421 | 1.31e-05 |
| slogdet_positive_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.860 | 3.804e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.900 | 3.804e-07 |
| householder_qr_square_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 1.939 | 0.0004212 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-direct | 16 | 15 | 2.686 | 0.0003159 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-direct | 16 | 15 | 2.582 | 0.0004731 |
| householder_qr_square_f64_32x32 | tenferro-cuda-direct | 16 | 15 | 2.201 | 3.886e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-direct | 16 | 15 | 2.895 | 4.441e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-direct | 16 | 15 | 2.958 | 4.441e-07 |
| householder_qr_square_c32_32x32 | tenferro-cuda-direct | 16 | 15 | 2.107 | 0.0005631 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-direct | 16 | 15 | 2.952 | 0.0003724 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-direct | 16 | 15 | 2.969 | 0.0006002 |
| slogdet_phase_c32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.785 | 0.0004085 |
| householder_qr_square_c64_32x32 | tenferro-cuda-direct | 16 | 15 | 2.564 | 4.996e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-direct | 16 | 15 | 3.422 | 5.551e-07 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-direct | 16 | 15 | 3.567 | 3.886e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.652 | 4.755e-07 |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-direct | 8 | 15 | 2.885 | 0.0004571 |
| householder_append_default_f32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f32_64x32 | tenferro-cuda-direct | 64 | 15 | 2.254 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-direct | 8 | 15 | 3.109 | 3.886e-07 |
| householder_append_default_f64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f64_64x32 | tenferro-cuda-direct | 64 | 15 | 2.303 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-direct | 8 | 15 | 3.123 | 0.0003232 |
| householder_append_default_c32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c32_64x32 | tenferro-cuda-direct | 64 | 15 | 2.340 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-direct | 8 | 15 | 3.193 | 4.441e-07 |
| householder_append_default_c64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c64_64x32 | tenferro-cuda-direct | 64 | 15 | 2.140 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_64x32 | tenferro-cuda-direct | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| lu_square_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.852 | 0.0003557 |
| lu_square_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.808 | 4.397e-07 |
| lu_square_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.916 | 0.0002381 |
| lu_square_c64_s32 | tenferro-cuda-direct | 16 | 15 | 3.051 | 5.532e-07 |
| lu_tall_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.839 | 0.0003557 |
| lu_tall_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.937 | 4.397e-07 |
| lu_tall_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.986 | 0.0005956 |
| lu_tall_c64_s32 | tenferro-cuda-direct | 16 | 15 | 3.015 | 4.406e-07 |
| lu_wide_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.896 | 0.00024 |
| lu_wide_f64_s32 | tenferro-cuda-direct | 16 | 15 | 3.022 | 2.214e-07 |
| lu_wide_c32_s32 | tenferro-cuda-direct | 16 | 15 | 3.055 | 0.0003567 |
| lu_wide_c64_s32 | tenferro-cuda-direct | 16 | 15 | 3.118 | 3.323e-07 |
| cholesky_f32_32x32 | tenferro-cuda-eager | 2 | 15 | 3.957 | 0.0002352 |
| cholesky_f64_32x32 | tenferro-cuda-eager | 2 | 15 | 3.988 | 2.188e-07 |
| svdvals_f32_32x32 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_32x32 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| eigvalsh_f32_32x32 | tenferro-cuda-eager | 1 | 15 | 2.115 | 0.001073 |
| eigvalsh_f64_32x32 | tenferro-cuda-eager | 1 | 15 | 2.159 | 5.433e-07 |
| pinv_f32_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f32_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| det_f32_32x32 | tenferro-cuda-eager | 2 | 15 | 4.087 | 0.002641 |
| det_f64_32x32 | tenferro-cuda-eager | 2 | 15 | 3.903 | 1.135e-06 |
| norm_f32_32x32 | tenferro-cuda-eager | 32 | 15 | 2.804 | 0 |
| norm_f64_32x32 | tenferro-cuda-eager | 32 | 15 | 3.060 | 0 |
| triangular_solve_f32_32x32 | tenferro-cuda-eager | 2 | 15 | 3.762 | 0.0001546 |
| triangular_solve_f64_32x32 | tenferro-cuda-eager | 2 | 15 | 3.767 | 1.609e-07 |
| svd_full_square_f32_32x32 | tenferro-cuda-eager | 1 | 15 | 2.522 | 0.008731 |
| svd_full_tall_f32_64x32 | tenferro-cuda-eager | 1 | 15 | 2.898 | 0.007661 |
| svd_full_wide_f32_32x64 | tenferro-cuda-eager | 1 | 15 | 2.919 | 0.005157 |
| slogdet_positive_f32_32x32 | tenferro-cuda-eager | 2 | 15 | 4.051 | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-eager | 2 | 15 | 3.881 | 0.000915 |
| svd_full_square_f64_32x32 | tenferro-cuda-eager | 1 | 15 | 2.718 | 1.007e-05 |
| svd_full_tall_f64_64x32 | tenferro-cuda-eager | 1 | 15 | 3.208 | 9.548e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-eager | 1 | 15 | 3.274 | 9.838e-06 |
| slogdet_positive_f64_32x32 | tenferro-cuda-eager | 2 | 15 | 3.890 | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-eager | 2 | 15 | 3.873 | 4.261e-07 |
| svd_full_square_c32_32x32 | tenferro-cuda-eager | 1 | 15 | 2.818 | 0.009079 |
| svd_full_tall_c32_64x32 | tenferro-cuda-eager | 1 | 15 | 3.132 | 0.009498 |
| svd_full_wide_c32_32x64 | tenferro-cuda-eager | 1 | 15 | 3.146 | 0.008495 |
| slogdet_positive_c32_32x32 | tenferro-cuda-eager | 2 | 15 | 3.881 | 0.0007148 |
| slogdet_negative_c32_32x32 | tenferro-cuda-eager | 2 | 15 | 4.090 | 0.0007148 |
| svd_full_square_c64_32x32 | tenferro-cuda-eager | 1 | 15 | 3.658 | 9.215e-06 |
| svd_full_tall_c64_64x32 | tenferro-cuda-eager | 1 | 15 | 4.154 | 1.221e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-eager | 1 | 15 | 4.408 | 1.31e-05 |
| slogdet_positive_c64_32x32 | tenferro-cuda-eager | 1 | 15 | 2.041 | 3.804e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-eager | 1 | 15 | 2.047 | 3.804e-07 |
| householder_qr_square_f32_32x32 | tenferro-cuda-eager | 2 | 15 | 3.759 | 0.0004212 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-eager | 2 | 15 | 3.861 | 0.0003159 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-eager | 2 | 15 | 3.892 | 0.0004731 |
| householder_qr_square_f64_32x32 | tenferro-cuda-eager | 2 | 15 | 3.854 | 3.886e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-eager | 1 | 15 | 2.061 | 4.441e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-eager | 1 | 15 | 2.051 | 4.441e-07 |
| householder_qr_square_c32_32x32 | tenferro-cuda-eager | 1 | 15 | 2.010 | 0.0005631 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-eager | 1 | 15 | 2.070 | 0.0003724 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-eager | 1 | 15 | 2.055 | 0.0006002 |
| slogdet_phase_c32_32x32 | tenferro-cuda-eager | 1 | 15 | 2.049 | 0.0004085 |
| householder_qr_square_c64_32x32 | tenferro-cuda-eager | 1 | 15 | 2.014 | 4.996e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-eager | 1 | 15 | 2.094 | 5.551e-07 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-eager | 1 | 15 | 2.098 | 3.886e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-eager | 1 | 15 | 2.043 | 4.755e-07 |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-eager | 1 | 15 | 2.124 | 0.0004571 |
| householder_append_default_f32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f32_64x32 | tenferro-cuda-eager | 64 | 15 | 2.978 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-eager | 1 | 15 | 2.132 | 3.886e-07 |
| householder_append_default_f64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_f64_64x32 | tenferro-cuda-eager | 64 | 15 | 3.105 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_f64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-eager | 1 | 15 | 2.047 | 0.0003232 |
| householder_append_default_c32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c32_64x32 | tenferro-cuda-eager | 64 | 15 | 2.992 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c32_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-eager | 2 | 15 | 4.170 | 4.441e-07 |
| householder_append_default_c64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_r_default_c64_64x32 | tenferro-cuda-eager | 64 | 15 | 2.738 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_complement_c64_64x32 | tenferro-cuda-eager | — | — | — | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| lu_square_f32_s32 | tenferro-cuda-eager | 1 | 15 | 1.923 | 0.0003557 |
| lu_square_f64_s32 | tenferro-cuda-eager | 2 | 15 | 3.930 | 4.397e-07 |
| lu_square_c32_s32 | tenferro-cuda-eager | 2 | 15 | 4.040 | 0.0002381 |
| lu_square_c64_s32 | tenferro-cuda-eager | 1 | 15 | 2.027 | 5.532e-07 |
| lu_tall_f32_s32 | tenferro-cuda-eager | 2 | 15 | 3.854 | 0.0003557 |
| lu_tall_f64_s32 | tenferro-cuda-eager | 2 | 15 | 3.849 | 4.397e-07 |
| lu_tall_c32_s32 | tenferro-cuda-eager | 2 | 15 | 3.873 | 0.0005956 |
| lu_tall_c64_s32 | tenferro-cuda-eager | 2 | 15 | 3.875 | 4.406e-07 |
| lu_wide_f32_s32 | tenferro-cuda-eager | 1 | 15 | 2.041 | 0.00024 |
| lu_wide_f64_s32 | tenferro-cuda-eager | 2 | 15 | 3.883 | 2.214e-07 |
| lu_wide_c32_s32 | tenferro-cuda-eager | 1 | 15 | 2.024 | 0.0003567 |
| lu_wide_c64_s32 | tenferro-cuda-eager | 2 | 15 | 3.998 | 3.323e-07 |
| cholesky_f32_32x32 | pytorch-cuda | 32 | 15 | 2.111 | 0.0002352 |
| cholesky_f64_32x32 | pytorch-cuda | 32 | 15 | 2.222 | 2.188e-07 |
| svdvals_f32_32x32 | pytorch-cuda | 8 | 15 | 3.280 | 0.007749 |
| svdvals_f64_32x32 | pytorch-cuda | 4 | 15 | 2.935 | 3.387e-06 |
| eigvalsh_f32_32x32 | pytorch-cuda | 16 | 15 | 3.633 | 0.001073 |
| eigvalsh_f64_32x32 | pytorch-cuda | 8 | 15 | 2.168 | 5.433e-07 |
| pinv_f32_32x32 | pytorch-cuda | 4 | 15 | 2.036 | 0.001025 |
| pinv_f64_32x32 | pytorch-cuda | 4 | 15 | 3.358 | 9.374e-07 |
| inv_f32_32x32 | pytorch-cuda | 16 | 15 | 3.030 | 0.0004905 |
| inv_f64_32x32 | pytorch-cuda | 16 | 15 | 3.150 | 2.284e-07 |
| det_f32_32x32 | pytorch-cuda | 32 | 15 | 3.392 | 0.002641 |
| det_f64_32x32 | pytorch-cuda | 32 | 15 | 3.459 | 3.784e-07 |
| norm_f32_32x32 | pytorch-cuda | 512 | 15 | 3.521 | 0 |
| norm_f64_32x32 | pytorch-cuda | 512 | 15 | 3.094 | 0 |
| triangular_solve_f32_32x32 | pytorch-cuda | 128 | 15 | 3.003 | 9.242e-05 |
| triangular_solve_f64_32x32 | pytorch-cuda | 128 | 15 | 3.025 | 1.415e-07 |
| svd_full_square_f32_32x32 | pytorch-cuda | 8 | 15 | 2.756 | 0.005571 |
| svd_full_tall_f32_64x32 | pytorch-cuda | 2 | 15 | 2.106 | 0.007658 |
| svd_full_wide_f32_32x64 | pytorch-cuda | 2 | 15 | 2.261 | 0.005115 |
| slogdet_positive_f32_32x32 | pytorch-cuda | 32 | 15 | 3.691 | 0.000915 |
| slogdet_negative_f32_32x32 | pytorch-cuda | 16 | 15 | 1.860 | 0.000915 |
| svd_full_square_f64_32x32 | pytorch-cuda | 4 | 15 | 2.453 | 7.216e-06 |
| svd_full_tall_f64_64x32 | pytorch-cuda | 2 | 15 | 2.967 | 9.215e-06 |
| svd_full_wide_f64_32x64 | pytorch-cuda | 2 | 15 | 3.155 | 7.327e-06 |
| slogdet_positive_f64_32x32 | pytorch-cuda | 16 | 15 | 2.157 | 4.261e-07 |
| slogdet_negative_f64_32x32 | pytorch-cuda | 32 | 15 | 4.345 | 4.261e-07 |
| svd_full_square_c32_32x32 | pytorch-cuda | 4 | 15 | 2.384 | 0.011 |
| svd_full_tall_c32_64x32 | pytorch-cuda | 2 | 15 | 2.778 | 0.008819 |
| svd_full_wide_c32_32x64 | pytorch-cuda | 2 | 15 | 2.857 | 0.008905 |
| slogdet_positive_c32_32x32 | pytorch-cuda | 16 | 15 | 2.277 | 0.0004085 |
| slogdet_negative_c32_32x32 | pytorch-cuda | 16 | 15 | 2.251 | 0.0004085 |
| svd_full_square_c64_32x32 | pytorch-cuda | 2 | 15 | 2.480 | 9.992e-06 |
| svd_full_tall_c64_64x32 | pytorch-cuda | 1 | 15 | 2.304 | 1.188e-05 |
| svd_full_wide_c64_32x64 | pytorch-cuda | 1 | 15 | 2.439 | 1.299e-05 |
| slogdet_positive_c64_32x32 | pytorch-cuda | 16 | 15 | 2.327 | 4.755e-07 |
| slogdet_negative_c64_32x32 | pytorch-cuda | 16 | 15 | 2.337 | 4.755e-07 |
| householder_qr_square_f32_32x32 | pytorch-cuda | 32 | 15 | 2.669 | 0.000367 |
| householder_qr_tall_f32_64x32 | pytorch-cuda | 16 | 15 | 2.009 | 0.000327 |
| householder_qr_wide_f32_32x64 | pytorch-cuda | 16 | 15 | 2.011 | 0.0003745 |
| householder_qr_square_f64_32x32 | pytorch-cuda | 32 | 15 | 3.181 | 6.106e-07 |
| householder_qr_tall_f64_64x32 | pytorch-cuda | 16 | 15 | 2.235 | 6.661e-07 |
| householder_qr_wide_f64_32x64 | pytorch-cuda | 16 | 15 | 2.240 | 5.551e-07 |
| householder_qr_square_c32_32x32 | pytorch-cuda | 32 | 15 | 3.036 | 0.0005833 |
| householder_qr_tall_c32_64x32 | pytorch-cuda | 16 | 15 | 2.322 | 0.0003756 |
| householder_qr_wide_c32_32x64 | pytorch-cuda | 16 | 15 | 2.341 | 0.0004972 |
| slogdet_phase_c32_32x32 | pytorch-cuda | 16 | 15 | 2.019 | 0.0004085 |
| householder_qr_square_c64_32x32 | pytorch-cuda | 16 | 15 | 2.033 | 1.221e-06 |
| householder_qr_tall_c64_64x32 | pytorch-cuda | 16 | 15 | 2.780 | 9.992e-07 |
| householder_qr_wide_c64_32x64 | pytorch-cuda | 16 | 15 | 2.936 | 1.166e-06 |
| slogdet_phase_c64_32x32 | pytorch-cuda | 16 | 15 | 2.151 | 5.706e-07 |
| householder_from_factors_default_f32_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_64x32 | pytorch-cuda | 512 | 15 | 2.276 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | pytorch-cuda | 32 | 15 | 1.775 | 0.0001196 |
| householder_q_columns_complement_f32_64x32 | pytorch-cuda | 8 | 15 | 2.004 | 0.0001791 |
| householder_from_factors_default_f64_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_64x32 | pytorch-cuda | 512 | 15 | 2.577 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | pytorch-cuda | 32 | 15 | 2.135 | 1.114e-07 |
| householder_q_columns_complement_f64_64x32 | pytorch-cuda | 8 | 15 | 2.306 | 2.724e-08 |
| householder_from_factors_default_c32_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_64x32 | pytorch-cuda | 512 | 15 | 2.654 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | pytorch-cuda | 32 | 15 | 1.987 | 0.00012 |
| householder_q_columns_complement_c32_64x32 | pytorch-cuda | 8 | 15 | 2.148 | 0.000299 |
| householder_from_factors_default_c64_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | pytorch-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_64x32 | pytorch-cuda | 512 | 15 | 2.610 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | pytorch-cuda | 32 | 15 | 2.176 | 5.592e-08 |
| householder_q_columns_complement_c64_64x32 | pytorch-cuda | 8 | 15 | 2.330 | 3.341e-07 |
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
| cholesky_f32_32x32 | jax-cuda | 32 | 15 | 2.307 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda | 32 | 15 | 2.172 | 2.188e-07 |
| svdvals_f32_32x32 | jax-cuda | 4 | 15 | 2.843 | 0.003338 |
| svdvals_f64_32x32 | jax-cuda | 2 | 15 | 2.069 | 4.546e-06 |
| eigvalsh_f32_32x32 | jax-cuda | 4 | 15 | 2.327 | 0.004775 |
| eigvalsh_f64_32x32 | jax-cuda | 4 | 15 | 2.827 | 5.542e-06 |
| pinv_f32_32x32 | jax-cuda | 4 | 15 | 3.792 | 0.0006436 |
| pinv_f64_32x32 | jax-cuda | 2 | 15 | 2.535 | 1.367e-06 |
| inv_f32_32x32 | jax-cuda | 32 | 15 | 3.490 | 0.0004905 |
| inv_f64_32x32 | jax-cuda | 32 | 15 | 3.688 | 2.284e-07 |
| det_f32_32x32 | jax-cuda | 32 | 15 | 2.594 | 0.002641 |
| det_f64_32x32 | jax-cuda | 32 | 15 | 2.694 | 1.135e-06 |
| norm_f32_32x32 | jax-cuda | 64 | 15 | 2.597 | 0 |
| norm_f64_32x32 | jax-cuda | 64 | 15 | 2.568 | 0 |
| triangular_solve_f32_32x32 | jax-cuda | 32 | 15 | 2.397 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda | 32 | 15 | 2.409 | 1.609e-07 |
| svd_full_square_f32_32x32 | jax-cuda | 4 | 15 | 2.564 | 0.008731 |
| svd_full_tall_f32_64x32 | jax-cuda | 2 | 15 | 2.072 | 0.007661 |
| svd_full_wide_f32_32x64 | jax-cuda | 2 | 15 | 2.230 | 0.005157 |
| slogdet_positive_f32_32x32 | jax-cuda | 32 | 15 | 2.935 | 0.000915 |
| slogdet_negative_f32_32x32 | jax-cuda | 32 | 15 | 2.947 | 0.000915 |
| svd_full_square_f64_32x32 | jax-cuda | 4 | 15 | 3.769 | 1.008e-05 |
| svd_full_tall_f64_64x32 | jax-cuda | 2 | 15 | 2.897 | 9.548e-06 |
| svd_full_wide_f64_32x64 | jax-cuda | 2 | 15 | 2.954 | 9.892e-06 |
| slogdet_positive_f64_32x32 | jax-cuda | 32 | 15 | 2.964 | 4.261e-07 |
| slogdet_negative_f64_32x32 | jax-cuda | 32 | 15 | 2.961 | 4.261e-07 |
| svd_full_square_c32_32x32 | jax-cuda | 4 | 15 | 3.840 | 0.009079 |
| svd_full_tall_c32_64x32 | jax-cuda | 2 | 15 | 2.774 | 0.009498 |
| svd_full_wide_c32_32x64 | jax-cuda | 2 | 15 | 2.788 | 0.008495 |
| slogdet_positive_c32_32x32 | jax-cuda | 32 | 15 | 2.946 | 0.0004085 |
| slogdet_negative_c32_32x32 | jax-cuda | 32 | 15 | 2.959 | 0.0004085 |
| svd_full_square_c64_32x32 | jax-cuda | 2 | 15 | 3.575 | 9.215e-06 |
| svd_full_tall_c64_64x32 | jax-cuda | 1 | 15 | 2.303 | 1.243e-05 |
| svd_full_wide_c64_32x64 | jax-cuda | 1 | 15 | 2.396 | 1.288e-05 |
| slogdet_positive_c64_32x32 | jax-cuda | 32 | 15 | 2.933 | 4.755e-07 |
| slogdet_negative_c64_32x32 | jax-cuda | 32 | 15 | 2.935 | 4.755e-07 |
| householder_qr_square_f32_32x32 | jax-cuda | 32 | 15 | 2.695 | 0.000367 |
| householder_qr_tall_f32_64x32 | jax-cuda | 16 | 15 | 2.047 | 0.000327 |
| householder_qr_wide_f32_32x64 | jax-cuda | 16 | 15 | 2.053 | 0.0003745 |
| householder_qr_square_f64_32x32 | jax-cuda | 32 | 15 | 3.219 | 6.106e-07 |
| householder_qr_tall_f64_64x32 | jax-cuda | 16 | 15 | 2.281 | 6.661e-07 |
| householder_qr_wide_f64_32x64 | jax-cuda | 16 | 15 | 2.297 | 5.551e-07 |
| householder_qr_square_c32_32x32 | jax-cuda | 32 | 15 | 3.069 | 0.0005833 |
| householder_qr_tall_c32_64x32 | jax-cuda | 16 | 15 | 2.371 | 0.0003756 |
| householder_qr_wide_c32_32x64 | jax-cuda | 16 | 15 | 2.390 | 0.0004972 |
| slogdet_phase_c32_32x32 | jax-cuda | 32 | 15 | 3.011 | 0.0004085 |
| householder_qr_square_c64_32x32 | jax-cuda | 16 | 15 | 2.092 | 1.221e-06 |
| householder_qr_tall_c64_64x32 | jax-cuda | 16 | 15 | 2.847 | 9.992e-07 |
| householder_qr_wide_c64_32x64 | jax-cuda | 16 | 15 | 2.987 | 1.166e-06 |
| slogdet_phase_c64_32x32 | jax-cuda | 32 | 15 | 2.955 | 5.706e-07 |
| householder_from_factors_default_f32_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_64x32 | jax-cuda | 64 | 15 | 2.586 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | jax-cuda | 16 | 15 | 2.032 | 0.0001196 |
| householder_q_columns_complement_f32_64x32 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f64_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_64x32 | jax-cuda | 64 | 15 | 2.554 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | jax-cuda | 16 | 15 | 2.088 | 1.114e-07 |
| householder_q_columns_complement_f64_64x32 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c32_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_64x32 | jax-cuda | 64 | 15 | 2.577 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | jax-cuda | 16 | 15 | 2.071 | 0.00012 |
| householder_q_columns_complement_c32_64x32 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c64_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | jax-cuda | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_64x32 | jax-cuda | 64 | 15 | 2.579 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | jax-cuda | 16 | 15 | 2.125 | 5.592e-08 |
| householder_q_columns_complement_c64_64x32 | jax-cuda | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
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
| cholesky_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.070 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.145 | 2.188e-07 |
| svdvals_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.746 | 0.003338 |
| svdvals_f64_32x32 | jax-cuda-jit | 2 | 15 | 2.003 | 4.546e-06 |
| eigvalsh_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.250 | 0.004775 |
| eigvalsh_f64_32x32 | jax-cuda-jit | 4 | 15 | 2.800 | 5.542e-06 |
| pinv_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.836 | 0.0006436 |
| pinv_f64_32x32 | jax-cuda-jit | 2 | 15 | 2.044 | 1.367e-06 |
| inv_f32_32x32 | jax-cuda-jit | 32 | 15 | 3.554 | 0.0004905 |
| inv_f64_32x32 | jax-cuda-jit | 32 | 15 | 3.699 | 2.284e-07 |
| det_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.632 | 0.002641 |
| det_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.707 | 1.135e-06 |
| norm_f32_32x32 | jax-cuda-jit | 64 | 15 | 2.642 | 0 |
| norm_f64_32x32 | jax-cuda-jit | 64 | 15 | 2.643 | 0 |
| triangular_solve_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.368 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.320 | 1.609e-07 |
| svd_full_square_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.553 | 0.008731 |
| svd_full_tall_f32_64x32 | jax-cuda-jit | 2 | 15 | 2.068 | 0.007661 |
| svd_full_wide_f32_32x64 | jax-cuda-jit | 2 | 15 | 2.218 | 0.005157 |
| slogdet_positive_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.980 | 0.000915 |
| slogdet_negative_f32_32x32 | jax-cuda-jit | 32 | 15 | 3.011 | 0.000915 |
| svd_full_square_f64_32x32 | jax-cuda-jit | 4 | 15 | 3.770 | 1.008e-05 |
| svd_full_tall_f64_64x32 | jax-cuda-jit | 2 | 15 | 2.904 | 9.548e-06 |
| svd_full_wide_f64_32x64 | jax-cuda-jit | 2 | 15 | 2.956 | 9.892e-06 |
| slogdet_positive_f64_32x32 | jax-cuda-jit | 32 | 15 | 3.024 | 4.261e-07 |
| slogdet_negative_f64_32x32 | jax-cuda-jit | 32 | 15 | 3.029 | 4.261e-07 |
| svd_full_square_c32_32x32 | jax-cuda-jit | 4 | 15 | 3.862 | 0.009079 |
| svd_full_tall_c32_64x32 | jax-cuda-jit | 2 | 15 | 2.751 | 0.009498 |
| svd_full_wide_c32_32x64 | jax-cuda-jit | 2 | 15 | 2.778 | 0.008495 |
| slogdet_positive_c32_32x32 | jax-cuda-jit | 32 | 15 | 2.973 | 0.0004085 |
| slogdet_negative_c32_32x32 | jax-cuda-jit | 32 | 15 | 2.955 | 0.0004085 |
| svd_full_square_c64_32x32 | jax-cuda-jit | 2 | 15 | 3.568 | 9.215e-06 |
| svd_full_tall_c64_64x32 | jax-cuda-jit | 1 | 15 | 2.319 | 1.243e-05 |
| svd_full_wide_c64_32x64 | jax-cuda-jit | 1 | 15 | 2.406 | 1.288e-05 |
| slogdet_positive_c64_32x32 | jax-cuda-jit | 32 | 15 | 2.961 | 4.755e-07 |
| slogdet_negative_c64_32x32 | jax-cuda-jit | 32 | 15 | 2.947 | 4.755e-07 |
| householder_qr_square_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.691 | 0.000367 |
| householder_qr_tall_f32_64x32 | jax-cuda-jit | 16 | 15 | 2.052 | 0.000327 |
| householder_qr_wide_f32_32x64 | jax-cuda-jit | 16 | 15 | 2.052 | 0.0003745 |
| householder_qr_square_f64_32x32 | jax-cuda-jit | 32 | 15 | 3.218 | 6.106e-07 |
| householder_qr_tall_f64_64x32 | jax-cuda-jit | 16 | 15 | 2.288 | 6.661e-07 |
| householder_qr_wide_f64_32x64 | jax-cuda-jit | 16 | 15 | 2.295 | 5.551e-07 |
| householder_qr_square_c32_32x32 | jax-cuda-jit | 32 | 15 | 3.067 | 0.0005833 |
| householder_qr_tall_c32_64x32 | jax-cuda-jit | 16 | 15 | 2.378 | 0.0003756 |
| householder_qr_wide_c32_32x64 | jax-cuda-jit | 16 | 15 | 2.391 | 0.0004972 |
| slogdet_phase_c32_32x32 | jax-cuda-jit | 32 | 15 | 2.948 | 0.0004085 |
| householder_qr_square_c64_32x32 | jax-cuda-jit | 16 | 15 | 2.081 | 1.221e-06 |
| householder_qr_tall_c64_64x32 | jax-cuda-jit | 16 | 15 | 2.839 | 9.992e-07 |
| householder_qr_wide_c64_32x64 | jax-cuda-jit | 16 | 15 | 2.986 | 1.166e-06 |
| slogdet_phase_c64_32x32 | jax-cuda-jit | 32 | 15 | 2.977 | 5.706e-07 |
| householder_from_factors_default_f32_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f32_64x32 | jax-cuda-jit | 64 | 15 | 2.628 | 0.0002373 |
| householder_q_columns_prefix_f32_64x32 | jax-cuda-jit | 32 | 15 | 3.185 | 0.0001196 |
| householder_q_columns_complement_f32_64x32 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_f64_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_f64_64x32 | jax-cuda-jit | 64 | 15 | 2.638 | 2.2e-07 |
| householder_q_columns_prefix_f64_64x32 | jax-cuda-jit | 32 | 15 | 3.348 | 1.114e-07 |
| householder_q_columns_complement_f64_64x32 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c32_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c32_64x32 | jax-cuda-jit | 64 | 15 | 2.654 | 0.0004713 |
| householder_q_columns_prefix_c32_64x32 | jax-cuda-jit | 32 | 15 | 3.241 | 0.00012 |
| householder_q_columns_complement_c32_64x32 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_from_factors_default_c64_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | jax-cuda-jit | — | — | — | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_r_default_c64_64x32 | jax-cuda-jit | 64 | 15 | 2.641 | 3.296e-07 |
| householder_q_columns_prefix_c64_64x32 | jax-cuda-jit | 32 | 15 | 3.301 | 5.592e-08 |
| householder_q_columns_complement_c64_64x32 | jax-cuda-jit | — | — | — | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
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
