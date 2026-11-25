# gpu/linalg_public_api_latency: CUDA public API benchmark

Measured cells: **332**. Unsupported cells: **78**, including **4** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1388.5 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api_latency/20261125_071457/`.

Small-shape view of the same CUDA public linalg API surface: the 82 cases whose largest matrix dimension is at most 128. `small` is a shape class, not a measured classification: device kernel time is a small fraction of each measurement here, so these rows are per-call launch and synchronization overhead diagnostics rather than GPU throughput. They are never comparable with, or averaged together with, the large-shape rows of `gpu/linalg_public_api_throughput` (tenferro-rs #2087). Issue 155. Repeated **untimed setup** work was removed here as well: the CUDA engine is opened once per arm and held for the whole run instead of once per case. Engine construction was never inside a case median, because the sample clock starts inside `measure`, so this change shortens the untimed wall clock around the cases and moves no operation median on this page.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**47** of 82) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | 70.674 (1.584) | 99.296 (2.484) | 67.605 (2.207) | 68.695 (2.945) | 67.072 (2.556) |
| cholesky_f64_32x32 | 79.573 (1.298) | 109.069 (2.586) | 71.546 (1.067) | 73.526 (1.546) | 68.038 (0.414) |
| svdvals_f32_32x32 † | 707.828 (8.074) | unsupported | 419.852 (4.178) | 741.487 (6.800) | 720.871 (9.820) |
| svdvals_f64_32x32 † | 1000.488 (12.005) | unsupported | 732.634 (3.141) | 1046.615 (6.658) | 1029.530 (7.727) |
| eigvalsh_f32_32x32 † | 224.192 (2.312) | 258.456 (3.891) | 574.911 (10.719) | 582.191 (9.114) | 589.437 (8.004) |
| eigvalsh_f64_32x32 † | 267.282 (5.122) | 304.293 (3.755) | 281.805 (2.417) | 745.892 (10.394) | 731.484 (8.185) |
| pinv_f32_32x32 † | unsupported | unsupported | 523.707 (5.354) | 997.649 (10.918) | 740.177 (3.776) |
| pinv_f64_32x32 † | unsupported | unsupported | 868.221 (8.274) | 1309.100 (16.840) | 1064.306 (12.045) |
| inv_f32_32x32 † | unsupported | unsupported | 209.165 (3.512) | 125.171 (2.738) | 135.306 (6.658) |
| inv_f64_32x32 † | unsupported | unsupported | 208.342 (5.766) | 126.706 (3.416) | 138.427 (4.678) |
| det_f32_32x32 † | 210.526 (7.596) | 249.007 (10.520) | 108.699 (6.501) | 110.404 (15.067) | 83.987 (2.929) |
| det_f64_32x32 † | 222.618 (3.495) | 259.936 (10.493) | 104.977 (1.583) | 85.783 (8.054) | 86.737 (1.234) |
| norm_f32_32x32 † | 17.935 (0.537) | 18.471 (0.490) | 7.208 (0.188) | 41.785 (0.507) | 42.957 (1.028) |
| norm_f64_32x32 † | 17.738 (0.437) | 18.635 (0.376) | 7.554 (0.072) | 42.580 (0.568) | 42.218 (1.033) |
| triangular_solve_f32_32x32 † | 17.901 (0.057) | 49.962 (2.018) | 28.073 (0.400) | 81.531 (5.798) | 74.958 (1.648) |
| triangular_solve_f64_32x32 † | 19.196 (0.043) | 48.713 (0.781) | 26.300 (0.189) | 79.995 (3.748) | 73.434 (2.075) |
| svd_full_square_f32_32x32 † | 662.475 (6.690) | 677.535 (6.254) | 357.909 (7.378) | 684.598 (4.320) | 687.589 (15.079) |
| svd_full_tall_f32_64x32 | 1047.165 (9.170) | 1071.780 (9.382) | 1064.910 (13.598) | 1071.686 (13.275) | 1090.307 (15.148) |
| svd_full_wide_f32_32x64 | 1128.806 (10.105) | 1145.696 (3.840) | 1142.687 (9.618) | 1147.502 (10.258) | 1160.098 (13.790) |
| slogdet_positive_f32_32x32 † | 195.315 (3.325) | 233.882 (6.854) | 146.168 (2.105) | 107.462 (2.167) | 123.853 (3.266) |
| slogdet_negative_f32_32x32 † | 193.618 (1.905) | 232.204 (3.465) | 140.470 (5.405) | 106.941 (3.026) | 108.914 (2.381) |
| svd_full_square_f64_32x32 † | 974.085 (4.941) | 978.943 (10.407) | 613.519 (0.415) | 972.074 (4.524) | 1008.715 (5.830) |
| svd_full_tall_f64_64x32 | 1471.807 (18.048) | 1504.973 (17.018) | 1523.499 (13.163) | 1483.459 (8.800) | 1514.425 (12.705) |
| svd_full_wide_f64_32x64 | 1511.518 (16.453) | 1522.948 (8.560) | 1612.626 (12.735) | 1565.775 (28.660) | 1565.171 (45.101) |
| slogdet_positive_f64_32x32 † | 202.490 (11.145) | 243.564 (6.933) | 140.773 (4.842) | 104.451 (2.342) | 102.775 (1.550) |
| slogdet_negative_f64_32x32 † | 207.630 (5.454) | 235.521 (5.672) | 146.289 (1.951) | 102.668 (1.669) | 110.160 (4.122) |
| svd_full_square_c32_32x32 † | 989.891 (11.497) | 1006.363 (6.968) | 606.979 (4.584) | 993.474 (10.312) | 1020.301 (14.780) |
| svd_full_tall_c32_64x32 | 1414.911 (26.541) | 1435.612 (9.612) | 1415.382 (13.540) | 1411.012 (7.400) | 1456.769 (6.560) |
| svd_full_wide_c32_32x64 | 1420.476 (18.723) | 1453.993 (11.162) | 1446.053 (11.735) | 1426.598 (9.078) | 1419.608 (5.953) |
| slogdet_positive_c32_32x32 † | 205.734 (4.206) | 255.624 (4.251) | 136.456 (2.399) | 109.338 (6.926) | 100.622 (1.521) |
| slogdet_negative_c32_32x32 † | 205.299 (6.042) | 255.381 (2.631) | 135.859 (2.736) | 100.456 (3.391) | 105.028 (2.292) |
| svd_full_square_c64_32x32 | 1821.463 (7.718) | 1874.655 (10.953) | 1242.433 (4.810) | 1943.703 (61.221) | 1835.161 (15.896) |
| svd_full_tall_c64_64x32 | 2304.162 (41.086) | 2350.414 (39.855) | 2302.433 (24.096) | 2408.096 (95.552) | 2409.168 (36.015) |
| svd_full_wide_c64_32x64 | 2445.406 (20.960) | 2470.056 (23.831) | 2434.007 (36.769) | 2537.529 (96.287) | 2475.329 (21.766) |
| slogdet_positive_c64_32x32 † | 219.091 (4.553) | 255.217 (5.944) | 143.255 (0.412) | 110.553 (5.754) | 104.452 (1.301) |
| slogdet_negative_c64_32x32 † | 212.665 (7.184) | 257.080 (3.029) | 144.031 (2.234) | 108.729 (8.377) | 105.096 (40.232) |
| householder_qr_square_f32_32x32 † | 122.055 (3.827) | 153.305 (2.366) | 83.078 (0.092) | 84.916 (0.307) | 85.586 (0.794) |
| householder_qr_tall_f32_64x32 | 164.407 (2.254) | 200.686 (2.891) | 124.922 (0.058) | 131.060 (1.445) | 130.508 (0.722) |
| householder_qr_wide_f32_32x64 † | 168.581 (3.034) | 200.229 (3.092) | 124.039 (0.148) | 132.256 (2.858) | 130.076 (1.368) |
| householder_qr_square_f64_32x32 † | 141.402 (2.738) | 172.855 (1.988) | 99.332 (0.090) | 101.782 (0.728) | 101.362 (0.125) |
| householder_qr_tall_f64_64x32 | 183.341 (2.713) | 220.096 (1.438) | 138.872 (0.067) | 145.778 (1.354) | 143.761 (0.434) |
| householder_qr_wide_f64_32x64 | 183.662 (3.418) | 213.186 (3.118) | 140.129 (0.335) | 146.037 (1.075) | 145.073 (0.374) |
| householder_qr_square_c32_32x32 † | 131.801 (1.529) | 156.419 (2.704) | 94.653 (0.096) | 97.388 (0.699) | 96.483 (0.171) |
| householder_qr_tall_c32_64x32 | 184.500 (1.324) | 212.209 (7.108) | 144.740 (0.175) | 150.777 (0.668) | 149.856 (0.317) |
| householder_qr_wide_c32_32x64 | 187.392 (1.322) | 217.363 (3.092) | 145.865 (0.203) | 150.995 (0.916) | 150.837 (0.208) |
| slogdet_phase_c32_32x32 † | 209.848 (6.073) | 248.249 (3.160) | 139.795 (1.732) | 102.511 (2.517) | 114.937 (2.297) |
| householder_qr_square_c64_32x32 | 165.530 (2.043) | 192.131 (1.789) | 126.710 (0.345) | 131.983 (0.554) | 131.120 (0.626) |
| householder_qr_tall_c64_64x32 | 217.079 (1.975) | 246.431 (1.303) | 173.392 (0.214) | 180.118 (0.542) | 178.798 (0.598) |
| householder_qr_wide_c64_32x64 † | 228.292 (1.110) | 255.617 (1.969) | 182.652 (0.094) | 188.483 (0.678) | 189.807 (0.920) |
| slogdet_phase_c64_32x32 † | 215.624 (3.419) | 253.142 (2.334) | 138.926 (1.294) | 114.622 (3.056) | 108.602 (3.849) |
| householder_from_factors_default_f32_64x32 | 294.755 (5.633) | 339.589 (4.424) | unsupported | unsupported | unsupported |
| householder_append_default_f32_64x32 † | 187.536 (5.549) | 238.088 (11.298) | unsupported | unsupported | unsupported |
| householder_r_default_f32_64x32 † | 16.115 (0.670) | 24.664 (0.382) | 5.262 (0.158) | 44.144 (0.940) | 40.061 (0.705) |
| householder_q_columns_prefix_f32_64x32 † | 67.165 (2.566) | 94.531 (5.058) | 63.404 (1.499) | 145.676 (5.139) | 105.067 (6.737) |
| householder_q_columns_complement_f32_64x32 † | 67.931 (2.332) | 93.737 (1.580) | 247.378 (0.363) | unsupported | unsupported |
| householder_from_factors_default_f64_64x32 | 311.125 (7.728) | 352.502 (5.560) | unsupported | unsupported | unsupported |
| householder_append_default_f64_64x32 † | 205.146 (4.855) | 251.275 (7.878) | unsupported | unsupported | unsupported |
| householder_r_default_f64_64x32 † | 17.224 (0.534) | 24.919 (0.251) | 5.163 (0.077) | 72.060 (15.546) | 45.771 (2.067) |
| householder_q_columns_prefix_f64_64x32 † | 76.861 (3.592) | 98.094 (2.121) | 65.976 (1.729) | 144.011 (11.332) | 120.757 (5.122) |
| householder_q_columns_complement_f64_64x32 † | 78.829 (3.881) | 95.092 (1.919) | 276.790 (0.255) | unsupported | unsupported |
| householder_from_factors_default_c32_64x32 | 311.573 (5.180) | 346.008 (3.138) | unsupported | unsupported | unsupported |
| householder_append_default_c32_64x32 | 187.635 (4.708) | 232.662 (4.823) | unsupported | unsupported | unsupported |
| householder_r_default_c32_64x32 † | 16.066 (0.474) | 25.412 (0.556) | 5.048 (0.088) | 45.638 (1.887) | 47.841 (1.136) |
| householder_q_columns_prefix_c32_64x32 † | 77.194 (0.270) | 90.313 (7.341) | 59.711 (2.046) | 139.578 (6.137) | 108.657 (10.446) |
| householder_q_columns_complement_c32_64x32 † | 80.378 (0.140) | 87.619 (3.622) | 263.692 (0.717) | unsupported | unsupported |
| householder_from_factors_default_c64_64x32 | 343.969 (5.639) | 388.785 (11.174) | unsupported | unsupported | unsupported |
| householder_append_default_c64_64x32 † | 220.766 (5.902) | 261.820 (8.230) | unsupported | unsupported | unsupported |
| householder_r_default_c64_64x32 † | 17.000 (0.736) | 26.409 (0.852) | 5.274 (0.120) | 43.797 (1.326) | 44.413 (1.876) |
| householder_q_columns_prefix_c64_64x32 † | 80.611 (0.302) | 92.505 (4.596) | 64.990 (1.079) | 141.026 (4.043) | 107.328 (3.969) |
| householder_q_columns_complement_c64_64x32 † | 84.679 (0.341) | 94.647 (2.622) | 286.297 (0.496) | unsupported | unsupported |
| lu_square_f32_s32 | 147.151 (2.995) | 163.722 (1.904) | unsupported | unsupported | unsupported |
| lu_square_f64_s32 | 153.271 (2.189) | 165.876 (3.255) | unsupported | unsupported | unsupported |
| lu_square_c32_s32 | 151.971 (2.886) | 169.151 (3.551) | unsupported | unsupported | unsupported |
| lu_square_c64_s32 | 157.115 (2.708) | 180.663 (4.223) | unsupported | unsupported | unsupported |
| lu_tall_f32_s32 | 152.650 (1.480) | 166.440 (3.283) | unsupported | unsupported | unsupported |
| lu_tall_f64_s32 | 153.105 (2.012) | 171.116 (3.553) | unsupported | unsupported | unsupported |
| lu_tall_c32_s32 | 153.061 (1.702) | 171.788 (5.637) | unsupported | unsupported | unsupported |
| lu_tall_c64_s32 | 154.494 (1.717) | 174.433 (4.503) | unsupported | unsupported | unsupported |
| lu_wide_f32_s32 | 157.169 (2.242) | 175.392 (1.519) | unsupported | unsupported | unsupported |
| lu_wide_f64_s32 | 161.289 (1.820) | 179.447 (3.623) | unsupported | unsupported | unsupported |
| lu_wide_c32_s32 | 158.555 (3.753) | 178.151 (5.652) | unsupported | unsupported | unsupported |
| lu_wide_c64_s32 | 168.290 (2.985) | 186.817 (4.886) | unsupported | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | jax-cuda | 32 | 15 | 2.198 | 8192 | 0.000 | 0.0002352 |
| cholesky_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.146 | 8192 | 0.000 | 0.0002352 |
| cholesky_f32_32x32 | pytorch-cuda | 32 | 15 | 2.163 | 8192 | 0.000 | 0.0002352 |
| cholesky_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.262 | 8192 | 0.000 | 0.0002352 |
| cholesky_f32_32x32 | tenferro-cuda-eager | 32 | 15 | 3.177 | 8192 | 0.000 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda | 32 | 15 | 2.353 | 16384 | 0.000 | 2.188e-07 |
| cholesky_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.177 | 16384 | 0.000 | 2.188e-07 |
| cholesky_f64_32x32 | pytorch-cuda | 32 | 15 | 2.289 | 16384 | 0.000 | 2.188e-07 |
| cholesky_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.546 | 16384 | 0.000 | 2.188e-07 |
| cholesky_f64_32x32 | tenferro-cuda-eager | 32 | 15 | 3.490 | 16384 | 0.000 | 2.188e-07 |
| det_f32_32x32 | jax-cuda | 32 | 15 | 3.533 | 4100 | 0.000 | 0.002641 |
| det_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.688 | 4100 | 0.000 | 0.002641 |
| det_f32_32x32 | pytorch-cuda | 32 | 15 | 3.478 | 4100 | 0.000 | 0.002641 |
| det_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.368 | 4100 | 0.000 | 0.002641 |
| det_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 1.992 | 4100 | 0.000 | 0.002641 |
| det_f64_32x32 | jax-cuda | 32 | 15 | 2.745 | 8200 | 0.000 | 1.135e-06 |
| det_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.776 | 8200 | 0.000 | 1.135e-06 |
| det_f64_32x32 | pytorch-cuda | 32 | 15 | 3.359 | 8200 | 0.000 | 3.784e-07 |
| det_f64_32x32 | tenferro-cuda-direct | 16 | 15 | 3.562 | 8200 | 0.000 | 1.135e-06 |
| det_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.079 | 8200 | 0.000 | 1.135e-06 |
| eigvalsh_f32_32x32 | jax-cuda | 4 | 15 | 2.329 | 4224 | 0.000 | 0.004775 |
| eigvalsh_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.358 | 4224 | 0.000 | 0.004775 |
| eigvalsh_f32_32x32 | pytorch-cuda | 4 | 15 | 2.300 | 4224 | 0.000 | 0.004775 |
| eigvalsh_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.587 | 4224 | 0.000 | 0.001073 |
| eigvalsh_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.068 | 4224 | 0.000 | 0.001073 |
| eigvalsh_f64_32x32 | jax-cuda | 4 | 15 | 2.984 | 8448 | 0.000 | 5.542e-06 |
| eigvalsh_f64_32x32 | jax-cuda-jit | 4 | 15 | 2.926 | 8448 | 0.000 | 5.542e-06 |
| eigvalsh_f64_32x32 | pytorch-cuda | 8 | 15 | 2.254 | 8448 | 0.000 | 5.433e-07 |
| eigvalsh_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.138 | 8448 | 0.000 | 5.433e-07 |
| eigvalsh_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.434 | 8448 | 0.000 | 5.433e-07 |
| householder_append_default_c32_64x32 | jax-cuda | — | — | — | 49408 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | jax-cuda-jit | — | — | — | 49408 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | pytorch-cuda | — | — | — | 49408 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c32_64x32 | tenferro-cuda-direct | 16 | 15 | 3.002 | 49408 | 0.000 | 0.0003724 |
| householder_append_default_c32_64x32 | tenferro-cuda-eager | 16 | 15 | 3.723 | 49408 | 0.000 | 0.0003724 |
| householder_append_default_c64_64x32 | jax-cuda | — | — | — | 98816 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | jax-cuda-jit | — | — | — | 98816 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | pytorch-cuda | — | — | — | 98816 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_c64_64x32 | tenferro-cuda-direct | 16 | 15 | 3.532 | 98816 | 0.000 | 5.551e-07 |
| householder_append_default_c64_64x32 | tenferro-cuda-eager | 8 | 15 | 2.095 | 98816 | 0.000 | 5.551e-07 |
| householder_append_default_f32_64x32 | jax-cuda | — | — | — | 24704 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | jax-cuda-jit | — | — | — | 24704 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | pytorch-cuda | — | — | — | 24704 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f32_64x32 | tenferro-cuda-direct | 16 | 15 | 3.001 | 24704 | 0.000 | 0.0003159 |
| householder_append_default_f32_64x32 | tenferro-cuda-eager | 8 | 15 | 1.905 | 24704 | 0.000 | 0.0003159 |
| householder_append_default_f64_64x32 | jax-cuda | — | — | — | 49408 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | jax-cuda-jit | — | — | — | 49408 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | pytorch-cuda | — | — | — | 49408 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_64x32 | tenferro-cuda-direct | 16 | 15 | 3.282 | 49408 | 0.000 | 4.441e-07 |
| householder_append_default_f64_64x32 | tenferro-cuda-eager | 8 | 15 | 2.010 | 49408 | 0.000 | 4.441e-07 |
| householder_from_factors_default_c32_64x32 | jax-cuda | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c32_64x32 | jax-cuda-jit | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c32_64x32 | pytorch-cuda | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-direct | 8 | 15 | 2.493 | 57600 | 0.000 | 0.0003232 |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-eager | 8 | 15 | 2.768 | 57600 | 0.000 | 0.0003232 |
| householder_from_factors_default_c64_64x32 | jax-cuda | — | — | — | 115200 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_64x32 | jax-cuda-jit | — | — | — | 115200 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_64x32 | pytorch-cuda | — | — | — | 115200 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-direct | 8 | 15 | 2.752 | 115200 | 0.000 | 4.441e-07 |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-eager | 8 | 15 | 3.110 | 115200 | 0.000 | 4.441e-07 |
| householder_from_factors_default_f32_64x32 | jax-cuda | — | — | — | 28800 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_64x32 | jax-cuda-jit | — | — | — | 28800 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_64x32 | pytorch-cuda | — | — | — | 28800 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-direct | 8 | 15 | 2.358 | 28800 | 0.000 | 0.0004571 |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-eager | 8 | 15 | 2.717 | 28800 | 0.000 | 0.0004571 |
| householder_from_factors_default_f64_64x32 | jax-cuda | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_64x32 | jax-cuda-jit | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_64x32 | pytorch-cuda | — | — | — | 57600 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-direct | 8 | 15 | 2.489 | 57600 | 0.000 | 3.886e-07 |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-eager | 8 | 15 | 2.820 | 57600 | 0.000 | 3.886e-07 |
| householder_q_columns_complement_c32_64x32 | jax-cuda | — | — | — | 32768 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c32_64x32 | jax-cuda-jit | — | — | — | 32768 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c32_64x32 | pytorch-cuda | 8 | 15 | 2.110 | 32768 | 0.000 | 0.000299 |
| householder_q_columns_complement_c32_64x32 | tenferro-cuda-direct | 32 | 15 | 2.572 | 32768 | 0.000 | 4.597e-05 |
| householder_q_columns_complement_c32_64x32 | tenferro-cuda-eager | 32 | 15 | 2.804 | 32768 | 0.000 | 4.597e-05 |
| householder_q_columns_complement_c64_64x32 | jax-cuda | — | — | — | 65536 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c64_64x32 | jax-cuda-jit | — | — | — | 65536 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_c64_64x32 | pytorch-cuda | 8 | 15 | 2.290 | 65536 | 0.000 | 3.341e-07 |
| householder_q_columns_complement_c64_64x32 | tenferro-cuda-direct | 32 | 15 | 2.710 | 65536 | 0.001 | 3.34e-07 |
| householder_q_columns_complement_c64_64x32 | tenferro-cuda-eager | 32 | 15 | 3.029 | 65536 | 0.000 | 3.34e-07 |
| householder_q_columns_complement_f32_64x32 | jax-cuda | — | — | — | 16384 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f32_64x32 | jax-cuda-jit | — | — | — | 16384 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f32_64x32 | pytorch-cuda | 16 | 15 | 3.958 | 16384 | 0.000 | 0.0001791 |
| householder_q_columns_complement_f32_64x32 | tenferro-cuda-direct | 32 | 15 | 2.174 | 16384 | 0.000 | 5.97e-05 |
| householder_q_columns_complement_f32_64x32 | tenferro-cuda-eager | 32 | 15 | 3.000 | 16384 | 0.000 | 5.97e-05 |
| householder_q_columns_complement_f64_64x32 | jax-cuda | — | — | — | 32768 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f64_64x32 | jax-cuda-jit | — | — | — | 32768 | unavailable | No verified single JAX operation for a selected non-prefix Q range; full-Q construction plus slicing changes requested outputs. |
| householder_q_columns_complement_f64_64x32 | pytorch-cuda | 8 | 15 | 2.214 | 32768 | 0.000 | 2.724e-08 |
| householder_q_columns_complement_f64_64x32 | tenferro-cuda-direct | 32 | 15 | 2.523 | 32768 | 0.000 | 1.668e-07 |
| householder_q_columns_complement_f64_64x32 | tenferro-cuda-eager | 32 | 15 | 3.043 | 32768 | 0.000 | 1.668e-07 |
| householder_q_columns_prefix_c32_64x32 | jax-cuda | 16 | 15 | 2.233 | 20480 | 0.000 | 0.00012 |
| householder_q_columns_prefix_c32_64x32 | jax-cuda-jit | 32 | 15 | 3.477 | 20480 | 0.000 | 0.00012 |
| householder_q_columns_prefix_c32_64x32 | pytorch-cuda | 64 | 15 | 3.822 | 20480 | 0.000 | 0.00012 |
| householder_q_columns_prefix_c32_64x32 | tenferro-cuda-direct | 32 | 15 | 2.470 | 20480 | 0.000 | 0.00012 |
| householder_q_columns_prefix_c32_64x32 | tenferro-cuda-eager | 32 | 15 | 2.890 | 20480 | 0.000 | 0.00012 |
| householder_q_columns_prefix_c64_64x32 | jax-cuda | 16 | 15 | 2.256 | 40960 | 0.000 | 5.592e-08 |
| householder_q_columns_prefix_c64_64x32 | jax-cuda-jit | 32 | 15 | 3.434 | 40960 | 0.000 | 5.592e-08 |
| householder_q_columns_prefix_c64_64x32 | pytorch-cuda | 32 | 15 | 2.080 | 40960 | 0.000 | 5.592e-08 |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-direct | 32 | 15 | 2.580 | 40960 | 0.000 | 5.592e-08 |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-eager | 32 | 15 | 2.960 | 40960 | 0.000 | 5.592e-08 |
| householder_q_columns_prefix_f32_64x32 | jax-cuda | 16 | 15 | 2.331 | 10240 | 0.000 | 0.0001196 |
| householder_q_columns_prefix_f32_64x32 | jax-cuda-jit | 32 | 15 | 3.362 | 10240 | 0.000 | 0.0001196 |
| householder_q_columns_prefix_f32_64x32 | pytorch-cuda | 32 | 15 | 2.029 | 10240 | 0.000 | 0.0001196 |
| householder_q_columns_prefix_f32_64x32 | tenferro-cuda-direct | 32 | 15 | 2.149 | 10240 | 0.000 | 0.0001196 |
| householder_q_columns_prefix_f32_64x32 | tenferro-cuda-eager | 32 | 15 | 3.025 | 10240 | 0.000 | 0.0001196 |
| householder_q_columns_prefix_f64_64x32 | jax-cuda | 16 | 15 | 2.304 | 20480 | 0.000 | 1.114e-07 |
| householder_q_columns_prefix_f64_64x32 | jax-cuda-jit | 32 | 15 | 3.864 | 20480 | 0.000 | 1.114e-07 |
| householder_q_columns_prefix_f64_64x32 | pytorch-cuda | 32 | 15 | 2.111 | 20480 | 0.000 | 1.114e-07 |
| householder_q_columns_prefix_f64_64x32 | tenferro-cuda-direct | 32 | 15 | 2.460 | 20480 | 0.000 | 1.114e-07 |
| householder_q_columns_prefix_f64_64x32 | tenferro-cuda-eager | 32 | 15 | 3.139 | 20480 | 0.000 | 1.114e-07 |
| householder_qr_square_c32_32x32 | jax-cuda | 32 | 15 | 3.116 | 16640 | 0.000 | 0.0005833 |
| householder_qr_square_c32_32x32 | jax-cuda-jit | 32 | 15 | 3.087 | 16640 | 0.000 | 0.0005833 |
| householder_qr_square_c32_32x32 | pytorch-cuda | 32 | 15 | 3.029 | 16640 | 0.000 | 0.0005833 |
| householder_qr_square_c32_32x32 | tenferro-cuda-direct | 16 | 15 | 2.109 | 16640 | 0.000 | 0.0005631 |
| householder_qr_square_c32_32x32 | tenferro-cuda-eager | 16 | 15 | 2.503 | 16640 | 0.000 | 0.0005631 |
| householder_qr_square_c64_32x32 | jax-cuda | 16 | 15 | 2.112 | 33280 | 0.000 | 1.221e-06 |
| householder_qr_square_c64_32x32 | jax-cuda-jit | 16 | 15 | 2.098 | 33280 | 0.000 | 1.221e-06 |
| householder_qr_square_c64_32x32 | pytorch-cuda | 16 | 15 | 2.027 | 33280 | 0.000 | 1.221e-06 |
| householder_qr_square_c64_32x32 | tenferro-cuda-direct | 16 | 15 | 2.648 | 33280 | 0.000 | 4.996e-07 |
| householder_qr_square_c64_32x32 | tenferro-cuda-eager | 16 | 15 | 3.074 | 33280 | 0.000 | 4.996e-07 |
| householder_qr_square_f32_32x32 | jax-cuda | 32 | 15 | 2.717 | 8320 | 0.000 | 0.000367 |
| householder_qr_square_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.739 | 8320 | 0.000 | 0.000367 |
| householder_qr_square_f32_32x32 | pytorch-cuda | 32 | 15 | 2.658 | 8320 | 0.000 | 0.000367 |
| householder_qr_square_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 1.953 | 8320 | 0.000 | 0.0004212 |
| householder_qr_square_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 2.453 | 8320 | 0.000 | 0.0004212 |
| householder_qr_square_f64_32x32 | jax-cuda | 32 | 15 | 3.257 | 16640 | 0.000 | 6.106e-07 |
| householder_qr_square_f64_32x32 | jax-cuda-jit | 32 | 15 | 3.244 | 16640 | 0.000 | 6.106e-07 |
| householder_qr_square_f64_32x32 | pytorch-cuda | 32 | 15 | 3.179 | 16640 | 0.000 | 6.106e-07 |
| householder_qr_square_f64_32x32 | tenferro-cuda-direct | 16 | 15 | 2.262 | 16640 | 0.000 | 3.886e-07 |
| householder_qr_square_f64_32x32 | tenferro-cuda-eager | 16 | 15 | 2.766 | 16640 | 0.000 | 3.886e-07 |
| householder_qr_tall_c32_64x32 | jax-cuda | 16 | 15 | 2.412 | 33024 | 0.000 | 0.0003756 |
| householder_qr_tall_c32_64x32 | jax-cuda-jit | 16 | 15 | 2.398 | 33024 | 0.000 | 0.0003756 |
| householder_qr_tall_c32_64x32 | pytorch-cuda | 16 | 15 | 2.316 | 33024 | 0.000 | 0.0003756 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-direct | 16 | 15 | 2.952 | 33024 | 0.000 | 0.0003724 |
| householder_qr_tall_c32_64x32 | tenferro-cuda-eager | 16 | 15 | 3.395 | 33024 | 0.000 | 0.0003724 |
| householder_qr_tall_c64_64x32 | jax-cuda | 16 | 15 | 2.882 | 66048 | 0.000 | 9.992e-07 |
| householder_qr_tall_c64_64x32 | jax-cuda-jit | 16 | 15 | 2.861 | 66048 | 0.000 | 9.992e-07 |
| householder_qr_tall_c64_64x32 | pytorch-cuda | 16 | 15 | 2.774 | 66048 | 0.000 | 9.992e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-direct | 16 | 15 | 3.473 | 66048 | 0.000 | 5.551e-07 |
| householder_qr_tall_c64_64x32 | tenferro-cuda-eager | 16 | 15 | 3.943 | 66048 | 0.000 | 5.551e-07 |
| householder_qr_tall_f32_64x32 | jax-cuda | 16 | 15 | 2.097 | 16512 | 0.000 | 0.000327 |
| householder_qr_tall_f32_64x32 | jax-cuda-jit | 16 | 15 | 2.088 | 16512 | 0.000 | 0.000327 |
| householder_qr_tall_f32_64x32 | pytorch-cuda | 16 | 15 | 1.999 | 16512 | 0.000 | 0.000327 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-direct | 16 | 15 | 2.631 | 16512 | 0.000 | 0.0003159 |
| householder_qr_tall_f32_64x32 | tenferro-cuda-eager | 16 | 15 | 3.211 | 16512 | 0.000 | 0.0003159 |
| householder_qr_tall_f64_64x32 | jax-cuda | 16 | 15 | 2.332 | 33024 | 0.000 | 6.661e-07 |
| householder_qr_tall_f64_64x32 | jax-cuda-jit | 16 | 15 | 2.300 | 33024 | 0.000 | 6.661e-07 |
| householder_qr_tall_f64_64x32 | pytorch-cuda | 16 | 15 | 2.222 | 33024 | 0.000 | 6.661e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-direct | 16 | 15 | 2.933 | 33024 | 0.000 | 4.441e-07 |
| householder_qr_tall_f64_64x32 | tenferro-cuda-eager | 16 | 15 | 3.522 | 33024 | 0.000 | 4.441e-07 |
| householder_qr_wide_c32_32x64 | jax-cuda | 16 | 15 | 2.416 | 33024 | 0.000 | 0.0004972 |
| householder_qr_wide_c32_32x64 | jax-cuda-jit | 16 | 15 | 2.413 | 33024 | 0.000 | 0.0004972 |
| householder_qr_wide_c32_32x64 | pytorch-cuda | 16 | 15 | 2.334 | 33024 | 0.000 | 0.0004972 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-direct | 16 | 15 | 2.998 | 33024 | 0.000 | 0.0006002 |
| householder_qr_wide_c32_32x64 | tenferro-cuda-eager | 16 | 15 | 3.478 | 33024 | 0.000 | 0.0006002 |
| householder_qr_wide_c64_32x64 | jax-cuda | 16 | 15 | 3.016 | 66048 | 0.000 | 1.166e-06 |
| householder_qr_wide_c64_32x64 | jax-cuda-jit | 16 | 15 | 3.037 | 66048 | 0.000 | 1.166e-06 |
| householder_qr_wide_c64_32x64 | pytorch-cuda | 16 | 15 | 2.922 | 66048 | 0.000 | 1.166e-06 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-direct | 16 | 15 | 3.653 | 66048 | 0.000 | 3.886e-07 |
| householder_qr_wide_c64_32x64 | tenferro-cuda-eager | 8 | 15 | 2.045 | 66048 | 0.000 | 3.886e-07 |
| householder_qr_wide_f32_32x64 | jax-cuda | 16 | 15 | 2.116 | 16512 | 0.000 | 0.0003745 |
| householder_qr_wide_f32_32x64 | jax-cuda-jit | 16 | 15 | 2.081 | 16512 | 0.000 | 0.0003745 |
| householder_qr_wide_f32_32x64 | pytorch-cuda | 32 | 15 | 3.969 | 16512 | 0.000 | 0.0003745 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-direct | 16 | 15 | 2.697 | 16512 | 0.000 | 0.0004731 |
| householder_qr_wide_f32_32x64 | tenferro-cuda-eager | 16 | 15 | 3.204 | 16512 | 0.000 | 0.0004731 |
| householder_qr_wide_f64_32x64 | jax-cuda | 16 | 15 | 2.337 | 33024 | 0.000 | 5.551e-07 |
| householder_qr_wide_f64_32x64 | jax-cuda-jit | 16 | 15 | 2.321 | 33024 | 0.000 | 5.551e-07 |
| householder_qr_wide_f64_32x64 | pytorch-cuda | 16 | 15 | 2.242 | 33024 | 0.000 | 5.551e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-direct | 16 | 15 | 2.939 | 33024 | 0.000 | 4.441e-07 |
| householder_qr_wide_f64_32x64 | tenferro-cuda-eager | 16 | 15 | 3.411 | 33024 | 0.000 | 4.441e-07 |
| householder_r_default_c32_64x32 | jax-cuda | 64 | 15 | 2.921 | 24576 | 0.000 | 0.0004713 |
| householder_r_default_c32_64x32 | jax-cuda-jit | 64 | 15 | 3.062 | 24576 | 0.000 | 0.0004713 |
| householder_r_default_c32_64x32 | pytorch-cuda | 512 | 15 | 2.585 | 24576 | 0.004 | 0.0004713 |
| householder_r_default_c32_64x32 | tenferro-cuda-direct | 128 | 15 | 2.056 | 24576 | 0.001 | 0.0004713 |
| householder_r_default_c32_64x32 | tenferro-cuda-eager | 128 | 15 | 3.253 | 24576 | 0.001 | 0.0004713 |
| householder_r_default_c64_64x32 | jax-cuda | 64 | 15 | 2.803 | 49152 | 0.001 | 3.296e-07 |
| householder_r_default_c64_64x32 | jax-cuda-jit | 64 | 15 | 2.842 | 49152 | 0.001 | 3.296e-07 |
| householder_r_default_c64_64x32 | pytorch-cuda | 512 | 15 | 2.700 | 49152 | 0.007 | 3.296e-07 |
| householder_r_default_c64_64x32 | tenferro-cuda-direct | 128 | 15 | 2.176 | 49152 | 0.002 | 3.296e-07 |
| householder_r_default_c64_64x32 | tenferro-cuda-eager | 128 | 15 | 3.380 | 49152 | 0.001 | 3.296e-07 |
| householder_r_default_f32_64x32 | jax-cuda | 64 | 15 | 2.825 | 12288 | 0.000 | 0.0002373 |
| householder_r_default_f32_64x32 | jax-cuda-jit | 64 | 15 | 2.564 | 12288 | 0.000 | 0.0002373 |
| householder_r_default_f32_64x32 | pytorch-cuda | 512 | 15 | 2.694 | 12288 | 0.002 | 0.0002373 |
| householder_r_default_f32_64x32 | tenferro-cuda-direct | 128 | 15 | 2.063 | 12288 | 0.001 | 0.0002373 |
| householder_r_default_f32_64x32 | tenferro-cuda-eager | 128 | 15 | 3.157 | 12288 | 0.000 | 0.0002373 |
| householder_r_default_f64_64x32 | jax-cuda | 32 | 15 | 2.306 | 24576 | 0.000 | 2.2e-07 |
| householder_r_default_f64_64x32 | jax-cuda-jit | 64 | 15 | 2.929 | 24576 | 0.000 | 2.2e-07 |
| householder_r_default_f64_64x32 | pytorch-cuda | 512 | 15 | 2.644 | 24576 | 0.003 | 2.2e-07 |
| householder_r_default_f64_64x32 | tenferro-cuda-direct | 128 | 15 | 2.205 | 24576 | 0.001 | 2.2e-07 |
| householder_r_default_f64_64x32 | tenferro-cuda-eager | 128 | 15 | 3.190 | 24576 | 0.001 | 2.2e-07 |
| inv_f32_32x32 | jax-cuda | 16 | 15 | 2.003 | 8192 | 0.000 | 0.0004905 |
| inv_f32_32x32 | jax-cuda-jit | 32 | 15 | 4.330 | 8192 | 0.000 | 0.0004905 |
| inv_f32_32x32 | pytorch-cuda | 16 | 15 | 3.347 | 8192 | 0.000 | 0.0004905 |
| inv_f32_32x32 | tenferro-cuda-direct | — | — | — | 8192 | unavailable | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_32x32 | tenferro-cuda-eager | — | — | — | 8192 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_32x32 | jax-cuda | 16 | 15 | 2.027 | 16384 | 0.000 | 2.284e-07 |
| inv_f64_32x32 | jax-cuda-jit | 32 | 15 | 4.430 | 16384 | 0.000 | 2.284e-07 |
| inv_f64_32x32 | pytorch-cuda | 16 | 15 | 3.333 | 16384 | 0.000 | 2.284e-07 |
| inv_f64_32x32 | tenferro-cuda-direct | — | — | — | 16384 | unavailable | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_32x32 | tenferro-cuda-eager | — | — | — | 16384 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| lu_square_c32_s32 | jax-cuda | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s32 | jax-cuda-jit | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s32 | pytorch-cuda | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.432 | 32776 | 0.000 | 0.0002381 |
| lu_square_c32_s32 | tenferro-cuda-eager | 16 | 15 | 2.706 | 32776 | 0.000 | 0.0002381 |
| lu_square_c64_s32 | jax-cuda | — | — | — | 65552 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s32 | jax-cuda-jit | — | — | — | 65552 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s32 | pytorch-cuda | — | — | — | 65552 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.514 | 65552 | 0.000 | 5.532e-07 |
| lu_square_c64_s32 | tenferro-cuda-eager | 16 | 15 | 2.891 | 65552 | 0.000 | 5.532e-07 |
| lu_square_f32_s32 | jax-cuda | — | — | — | 16388 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s32 | jax-cuda-jit | — | — | — | 16388 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s32 | pytorch-cuda | — | — | — | 16388 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.354 | 16388 | 0.000 | 0.0003557 |
| lu_square_f32_s32 | tenferro-cuda-eager | 16 | 15 | 2.620 | 16388 | 0.000 | 0.0003557 |
| lu_square_f64_s32 | jax-cuda | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s32 | jax-cuda-jit | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s32 | pytorch-cuda | — | — | — | 32776 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_square_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.452 | 32776 | 0.000 | 4.397e-07 |
| lu_square_f64_s32 | tenferro-cuda-eager | 16 | 15 | 2.654 | 32776 | 0.000 | 4.397e-07 |
| lu_tall_c32_s32 | jax-cuda | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s32 | jax-cuda-jit | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s32 | pytorch-cuda | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.449 | 73736 | 0.000 | 0.0005956 |
| lu_tall_c32_s32 | tenferro-cuda-eager | 16 | 15 | 2.749 | 73736 | 0.000 | 0.0005956 |
| lu_tall_c64_s32 | jax-cuda | — | — | — | 147472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s32 | jax-cuda-jit | — | — | — | 147472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s32 | pytorch-cuda | — | — | — | 147472 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.472 | 147472 | 0.001 | 4.406e-07 |
| lu_tall_c64_s32 | tenferro-cuda-eager | 16 | 15 | 2.791 | 147472 | 0.001 | 4.406e-07 |
| lu_tall_f32_s32 | jax-cuda | — | — | — | 36868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s32 | jax-cuda-jit | — | — | — | 36868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s32 | pytorch-cuda | — | — | — | 36868 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.442 | 36868 | 0.000 | 0.0003557 |
| lu_tall_f32_s32 | tenferro-cuda-eager | 16 | 15 | 2.663 | 36868 | 0.000 | 0.0003557 |
| lu_tall_f64_s32 | jax-cuda | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s32 | jax-cuda-jit | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s32 | pytorch-cuda | — | — | — | 73736 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.450 | 73736 | 0.000 | 4.397e-07 |
| lu_tall_f64_s32 | tenferro-cuda-eager | 16 | 15 | 2.738 | 73736 | 0.000 | 4.397e-07 |
| lu_wide_c32_s32 | jax-cuda | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s32 | jax-cuda-jit | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s32 | pytorch-cuda | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c32_s32 | tenferro-cuda-direct | 16 | 15 | 2.537 | 49160 | 0.000 | 0.0003567 |
| lu_wide_c32_s32 | tenferro-cuda-eager | 16 | 15 | 2.850 | 49160 | 0.000 | 0.0003567 |
| lu_wide_c64_s32 | jax-cuda | — | — | — | 98320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s32 | jax-cuda-jit | — | — | — | 98320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s32 | pytorch-cuda | — | — | — | 98320 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_c64_s32 | tenferro-cuda-direct | 16 | 15 | 2.693 | 98320 | 0.000 | 3.323e-07 |
| lu_wide_c64_s32 | tenferro-cuda-eager | 16 | 15 | 2.989 | 98320 | 0.000 | 3.323e-07 |
| lu_wide_f32_s32 | jax-cuda | — | — | — | 24580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s32 | jax-cuda-jit | — | — | — | 24580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s32 | pytorch-cuda | — | — | — | 24580 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f32_s32 | tenferro-cuda-direct | 16 | 15 | 2.515 | 24580 | 0.000 | 0.00024 |
| lu_wide_f32_s32 | tenferro-cuda-eager | 16 | 15 | 2.806 | 24580 | 0.000 | 0.00024 |
| lu_wide_f64_s32 | jax-cuda | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s32 | jax-cuda-jit | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s32 | pytorch-cuda | — | — | — | 49160 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_wide_f64_s32 | tenferro-cuda-direct | 16 | 15 | 2.581 | 49160 | 0.000 | 2.214e-07 |
| lu_wide_f64_s32 | tenferro-cuda-eager | 16 | 15 | 2.871 | 49160 | 0.000 | 2.214e-07 |
| norm_f32_32x32 | jax-cuda | 64 | 15 | 2.674 | 4100 | 0.000 | 0 |
| norm_f32_32x32 | jax-cuda-jit | 64 | 15 | 2.749 | 4100 | 0.000 | 0 |
| norm_f32_32x32 | pytorch-cuda | 512 | 15 | 3.690 | 4100 | 0.000 | 0 |
| norm_f32_32x32 | tenferro-cuda-direct | 128 | 15 | 2.296 | 4100 | 0.000 | 0 |
| norm_f32_32x32 | tenferro-cuda-eager | 128 | 15 | 2.364 | 4100 | 0.000 | 0 |
| norm_f64_32x32 | jax-cuda | 64 | 15 | 2.725 | 8200 | 0.000 | 0 |
| norm_f64_32x32 | jax-cuda-jit | 64 | 15 | 2.702 | 8200 | 0.000 | 0 |
| norm_f64_32x32 | pytorch-cuda | 512 | 15 | 3.867 | 8200 | 0.001 | 0 |
| norm_f64_32x32 | tenferro-cuda-direct | 128 | 15 | 2.270 | 8200 | 0.000 | 0 |
| norm_f64_32x32 | tenferro-cuda-eager | 128 | 15 | 2.385 | 8200 | 0.000 | 0 |
| pinv_f32_32x32 | jax-cuda | 2 | 15 | 1.995 | 8200 | 0.000 | 0.0006436 |
| pinv_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.961 | 8200 | 0.000 | 0.0006436 |
| pinv_f32_32x32 | pytorch-cuda | 4 | 15 | 2.095 | 8200 | 0.000 | 0.001025 |
| pinv_f32_32x32 | tenferro-cuda-direct | — | — | — | 8200 | unavailable | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f32_32x32 | tenferro-cuda-eager | — | — | — | 8200 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_32x32 | jax-cuda | 2 | 15 | 2.618 | 16400 | 0.000 | 1.367e-06 |
| pinv_f64_32x32 | jax-cuda-jit | 2 | 15 | 2.129 | 16400 | 0.000 | 1.367e-06 |
| pinv_f64_32x32 | pytorch-cuda | 4 | 15 | 3.473 | 16400 | 0.000 | 9.374e-07 |
| pinv_f64_32x32 | tenferro-cuda-direct | — | — | — | 16400 | unavailable | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_32x32 | tenferro-cuda-eager | — | — | — | 16400 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| slogdet_negative_c32_32x32 | jax-cuda | 32 | 15 | 3.215 | 8204 | 0.000 | 0.0004085 |
| slogdet_negative_c32_32x32 | jax-cuda-jit | 32 | 15 | 3.361 | 8204 | 0.000 | 0.0004085 |
| slogdet_negative_c32_32x32 | pytorch-cuda | 16 | 15 | 2.174 | 8204 | 0.000 | 0.0004085 |
| slogdet_negative_c32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.285 | 8204 | 0.000 | 0.0007148 |
| slogdet_negative_c32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.043 | 8204 | 0.000 | 0.0007148 |
| slogdet_negative_c64_32x32 | jax-cuda | 32 | 15 | 3.479 | 16408 | 0.000 | 4.755e-07 |
| slogdet_negative_c64_32x32 | jax-cuda-jit | 16 | 15 | 1.682 | 16408 | 0.000 | 4.755e-07 |
| slogdet_negative_c64_32x32 | pytorch-cuda | 16 | 15 | 2.304 | 16408 | 0.000 | 4.755e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-direct | 16 | 15 | 3.403 | 16408 | 0.000 | 3.804e-07 |
| slogdet_negative_c64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.057 | 16408 | 0.000 | 3.804e-07 |
| slogdet_negative_f32_32x32 | jax-cuda | 32 | 15 | 3.422 | 4104 | 0.000 | 0.000915 |
| slogdet_negative_f32_32x32 | jax-cuda-jit | 16 | 15 | 1.743 | 4104 | 0.000 | 0.000915 |
| slogdet_negative_f32_32x32 | pytorch-cuda | 16 | 15 | 2.248 | 4104 | 0.000 | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.098 | 4104 | 0.000 | 0.000915 |
| slogdet_negative_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 3.715 | 4104 | 0.000 | 0.000915 |
| slogdet_negative_f64_32x32 | jax-cuda | 32 | 15 | 3.285 | 8208 | 0.000 | 4.261e-07 |
| slogdet_negative_f64_32x32 | jax-cuda-jit | 16 | 15 | 1.763 | 8208 | 0.000 | 4.261e-07 |
| slogdet_negative_f64_32x32 | pytorch-cuda | 16 | 15 | 2.341 | 8208 | 0.000 | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-direct | 16 | 15 | 3.322 | 8208 | 0.000 | 4.261e-07 |
| slogdet_negative_f64_32x32 | tenferro-cuda-eager | 16 | 15 | 3.768 | 8208 | 0.000 | 4.261e-07 |
| slogdet_phase_c32_32x32 | jax-cuda | 32 | 15 | 3.280 | 8204 | 0.000 | 0.0004085 |
| slogdet_phase_c32_32x32 | jax-cuda-jit | 16 | 15 | 1.839 | 8204 | 0.000 | 0.0004085 |
| slogdet_phase_c32_32x32 | pytorch-cuda | 16 | 15 | 2.237 | 8204 | 0.000 | 0.0004085 |
| slogdet_phase_c32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.358 | 8204 | 0.000 | 0.0004085 |
| slogdet_phase_c32_32x32 | tenferro-cuda-eager | 8 | 15 | 1.986 | 8204 | 0.000 | 0.0004085 |
| slogdet_phase_c64_32x32 | jax-cuda | 32 | 15 | 3.668 | 16408 | 0.000 | 5.706e-07 |
| slogdet_phase_c64_32x32 | jax-cuda-jit | 16 | 15 | 1.738 | 16408 | 0.000 | 5.706e-07 |
| slogdet_phase_c64_32x32 | pytorch-cuda | 16 | 15 | 2.223 | 16408 | 0.000 | 5.706e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-direct | 16 | 15 | 3.450 | 16408 | 0.000 | 4.755e-07 |
| slogdet_phase_c64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.025 | 16408 | 0.000 | 4.755e-07 |
| slogdet_positive_c32_32x32 | jax-cuda | 32 | 15 | 3.499 | 8204 | 0.000 | 0.0004085 |
| slogdet_positive_c32_32x32 | jax-cuda-jit | 32 | 15 | 3.220 | 8204 | 0.000 | 0.0004085 |
| slogdet_positive_c32_32x32 | pytorch-cuda | 16 | 15 | 2.183 | 8204 | 0.000 | 0.0004085 |
| slogdet_positive_c32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.292 | 8204 | 0.000 | 0.0007148 |
| slogdet_positive_c32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.045 | 8204 | 0.000 | 0.0007148 |
| slogdet_positive_c64_32x32 | jax-cuda | 32 | 15 | 3.538 | 16408 | 0.000 | 4.755e-07 |
| slogdet_positive_c64_32x32 | jax-cuda-jit | 32 | 15 | 3.342 | 16408 | 0.000 | 4.755e-07 |
| slogdet_positive_c64_32x32 | pytorch-cuda | 16 | 15 | 2.292 | 16408 | 0.000 | 4.755e-07 |
| slogdet_positive_c64_32x32 | tenferro-cuda-direct | 16 | 15 | 3.505 | 16408 | 0.000 | 3.804e-07 |
| slogdet_positive_c64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.042 | 16408 | 0.000 | 3.804e-07 |
| slogdet_positive_f32_32x32 | jax-cuda | 32 | 15 | 3.439 | 4104 | 0.000 | 0.000915 |
| slogdet_positive_f32_32x32 | jax-cuda-jit | 32 | 15 | 3.963 | 4104 | 0.000 | 0.000915 |
| slogdet_positive_f32_32x32 | pytorch-cuda | 16 | 15 | 2.339 | 4104 | 0.000 | 0.000915 |
| slogdet_positive_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.125 | 4104 | 0.000 | 0.000915 |
| slogdet_positive_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 3.742 | 4104 | 0.000 | 0.000915 |
| slogdet_positive_f64_32x32 | jax-cuda | 32 | 15 | 3.342 | 8208 | 0.000 | 4.261e-07 |
| slogdet_positive_f64_32x32 | jax-cuda-jit | 32 | 15 | 3.289 | 8208 | 0.000 | 4.261e-07 |
| slogdet_positive_f64_32x32 | pytorch-cuda | 16 | 15 | 2.252 | 8208 | 0.000 | 4.261e-07 |
| slogdet_positive_f64_32x32 | tenferro-cuda-direct | 16 | 15 | 3.240 | 8208 | 0.000 | 4.261e-07 |
| slogdet_positive_f64_32x32 | tenferro-cuda-eager | 16 | 15 | 3.897 | 8208 | 0.000 | 4.261e-07 |
| svd_full_square_c32_32x32 | jax-cuda | 4 | 15 | 3.974 | 24704 | 0.000 | 0.009079 |
| svd_full_square_c32_32x32 | jax-cuda-jit | 2 | 15 | 2.041 | 24704 | 0.000 | 0.009079 |
| svd_full_square_c32_32x32 | pytorch-cuda | 4 | 15 | 2.428 | 24704 | 0.000 | 0.011 |
| svd_full_square_c32_32x32 | tenferro-cuda-direct | 4 | 15 | 3.960 | 24704 | 0.000 | 0.009079 |
| svd_full_square_c32_32x32 | tenferro-cuda-eager | 2 | 15 | 2.013 | 24704 | 0.000 | 0.009079 |
| svd_full_square_c64_32x32 | jax-cuda | 2 | 15 | 3.887 | 49408 | 0.000 | 9.215e-06 |
| svd_full_square_c64_32x32 | jax-cuda-jit | 2 | 15 | 3.670 | 49408 | 0.000 | 9.215e-06 |
| svd_full_square_c64_32x32 | pytorch-cuda | 2 | 15 | 2.485 | 49408 | 0.000 | 9.992e-06 |
| svd_full_square_c64_32x32 | tenferro-cuda-direct | 2 | 15 | 3.643 | 49408 | 0.000 | 9.215e-06 |
| svd_full_square_c64_32x32 | tenferro-cuda-eager | 2 | 15 | 3.749 | 49408 | 0.000 | 9.215e-06 |
| svd_full_square_f32_32x32 | jax-cuda | 4 | 15 | 2.738 | 12416 | 0.000 | 0.008731 |
| svd_full_square_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.750 | 12416 | 0.000 | 0.008731 |
| svd_full_square_f32_32x32 | pytorch-cuda | 8 | 15 | 2.863 | 12416 | 0.000 | 0.005571 |
| svd_full_square_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.650 | 12416 | 0.000 | 0.008731 |
| svd_full_square_f32_32x32 | tenferro-cuda-eager | 4 | 15 | 2.710 | 12416 | 0.000 | 0.008731 |
| svd_full_square_f64_32x32 | jax-cuda | 4 | 15 | 3.888 | 24832 | 0.000 | 1.008e-05 |
| svd_full_square_f64_32x32 | jax-cuda-jit | 2 | 15 | 2.017 | 24832 | 0.000 | 1.008e-05 |
| svd_full_square_f64_32x32 | pytorch-cuda | 4 | 15 | 2.454 | 24832 | 0.000 | 7.216e-06 |
| svd_full_square_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 3.896 | 24832 | 0.000 | 1.007e-05 |
| svd_full_square_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 3.916 | 24832 | 0.000 | 1.007e-05 |
| svd_full_tall_c32_64x32 | jax-cuda | 2 | 15 | 2.822 | 57472 | 0.000 | 0.009498 |
| svd_full_tall_c32_64x32 | jax-cuda-jit | 2 | 15 | 2.914 | 57472 | 0.000 | 0.009498 |
| svd_full_tall_c32_64x32 | pytorch-cuda | 2 | 15 | 2.831 | 57472 | 0.000 | 0.008819 |
| svd_full_tall_c32_64x32 | tenferro-cuda-direct | 2 | 15 | 2.830 | 57472 | 0.000 | 0.009498 |
| svd_full_tall_c32_64x32 | tenferro-cuda-eager | 2 | 15 | 2.871 | 57472 | 0.000 | 0.009498 |
| svd_full_tall_c64_64x32 | jax-cuda | 1 | 15 | 2.408 | 114944 | 0.000 | 1.243e-05 |
| svd_full_tall_c64_64x32 | jax-cuda-jit | 1 | 15 | 2.409 | 114944 | 0.000 | 1.243e-05 |
| svd_full_tall_c64_64x32 | pytorch-cuda | 1 | 15 | 2.302 | 114944 | 0.000 | 1.188e-05 |
| svd_full_tall_c64_64x32 | tenferro-cuda-direct | 1 | 15 | 2.304 | 114944 | 0.000 | 1.221e-05 |
| svd_full_tall_c64_64x32 | tenferro-cuda-eager | 1 | 15 | 2.350 | 114944 | 0.000 | 1.221e-05 |
| svd_full_tall_f32_64x32 | jax-cuda | 2 | 15 | 2.143 | 28800 | 0.000 | 0.007661 |
| svd_full_tall_f32_64x32 | jax-cuda-jit | 2 | 15 | 2.181 | 28800 | 0.000 | 0.007661 |
| svd_full_tall_f32_64x32 | pytorch-cuda | 2 | 15 | 2.130 | 28800 | 0.000 | 0.007658 |
| svd_full_tall_f32_64x32 | tenferro-cuda-direct | 2 | 15 | 2.094 | 28800 | 0.000 | 0.007661 |
| svd_full_tall_f32_64x32 | tenferro-cuda-eager | 2 | 15 | 2.144 | 28800 | 0.000 | 0.007661 |
| svd_full_tall_f64_64x32 | jax-cuda | 2 | 15 | 2.967 | 57600 | 0.000 | 9.548e-06 |
| svd_full_tall_f64_64x32 | jax-cuda-jit | 2 | 15 | 3.029 | 57600 | 0.000 | 9.548e-06 |
| svd_full_tall_f64_64x32 | pytorch-cuda | 2 | 15 | 3.047 | 57600 | 0.000 | 9.215e-06 |
| svd_full_tall_f64_64x32 | tenferro-cuda-direct | 2 | 15 | 2.944 | 57600 | 0.000 | 9.548e-06 |
| svd_full_tall_f64_64x32 | tenferro-cuda-eager | 2 | 15 | 3.010 | 57600 | 0.000 | 9.548e-06 |
| svd_full_wide_c32_32x64 | jax-cuda | 2 | 15 | 2.853 | 57472 | 0.000 | 0.008495 |
| svd_full_wide_c32_32x64 | jax-cuda-jit | 2 | 15 | 2.839 | 57472 | 0.000 | 0.008495 |
| svd_full_wide_c32_32x64 | pytorch-cuda | 2 | 15 | 2.892 | 57472 | 0.000 | 0.008905 |
| svd_full_wide_c32_32x64 | tenferro-cuda-direct | 2 | 15 | 2.841 | 57472 | 0.000 | 0.008495 |
| svd_full_wide_c32_32x64 | tenferro-cuda-eager | 2 | 15 | 2.908 | 57472 | 0.000 | 0.008495 |
| svd_full_wide_c64_32x64 | jax-cuda | 1 | 15 | 2.538 | 114944 | 0.000 | 1.288e-05 |
| svd_full_wide_c64_32x64 | jax-cuda-jit | 1 | 15 | 2.475 | 114944 | 0.000 | 1.288e-05 |
| svd_full_wide_c64_32x64 | pytorch-cuda | 1 | 15 | 2.434 | 114944 | 0.000 | 1.299e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-direct | 1 | 15 | 2.445 | 114944 | 0.000 | 1.31e-05 |
| svd_full_wide_c64_32x64 | tenferro-cuda-eager | 1 | 15 | 2.470 | 114944 | 0.000 | 1.31e-05 |
| svd_full_wide_f32_32x64 | jax-cuda | 2 | 15 | 2.295 | 28800 | 0.000 | 0.005157 |
| svd_full_wide_f32_32x64 | jax-cuda-jit | 2 | 15 | 2.320 | 28800 | 0.000 | 0.005157 |
| svd_full_wide_f32_32x64 | pytorch-cuda | 2 | 15 | 2.285 | 28800 | 0.000 | 0.005115 |
| svd_full_wide_f32_32x64 | tenferro-cuda-direct | 2 | 15 | 2.258 | 28800 | 0.000 | 0.005157 |
| svd_full_wide_f32_32x64 | tenferro-cuda-eager | 2 | 15 | 2.291 | 28800 | 0.000 | 0.005157 |
| svd_full_wide_f64_32x64 | jax-cuda | 2 | 15 | 3.132 | 57600 | 0.000 | 9.892e-06 |
| svd_full_wide_f64_32x64 | jax-cuda-jit | 2 | 15 | 3.130 | 57600 | 0.000 | 9.892e-06 |
| svd_full_wide_f64_32x64 | pytorch-cuda | 2 | 15 | 3.225 | 57600 | 0.000 | 7.327e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-direct | 2 | 15 | 3.023 | 57600 | 0.000 | 9.838e-06 |
| svd_full_wide_f64_32x64 | tenferro-cuda-eager | 2 | 15 | 3.046 | 57600 | 0.000 | 9.838e-06 |
| svdvals_f32_32x32 | jax-cuda | 4 | 15 | 2.966 | 4224 | 0.000 | 0.003338 |
| svdvals_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.883 | 4224 | 0.000 | 0.003338 |
| svdvals_f32_32x32 | pytorch-cuda | 8 | 15 | 3.359 | 4224 | 0.000 | 0.007749 |
| svdvals_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.831 | 4224 | 0.000 | 0.003338 |
| svdvals_f32_32x32 | tenferro-cuda-eager | — | — | — | 4224 | unavailable | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_32x32 | jax-cuda | 2 | 15 | 2.093 | 8448 | 0.000 | 4.546e-06 |
| svdvals_f64_32x32 | jax-cuda-jit | 2 | 15 | 2.059 | 8448 | 0.000 | 4.546e-06 |
| svdvals_f64_32x32 | pytorch-cuda | 4 | 15 | 2.931 | 8448 | 0.000 | 3.387e-06 |
| svdvals_f64_32x32 | tenferro-cuda-direct | 2 | 15 | 2.001 | 8448 | 0.000 | 4.546e-06 |
| svdvals_f64_32x32 | tenferro-cuda-eager | — | — | — | 8448 | unavailable | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| triangular_solve_f32_32x32 | jax-cuda | 32 | 15 | 2.609 | 6144 | 0.000 | 0.0001546 |
| triangular_solve_f32_32x32 | jax-cuda-jit | 32 | 15 | 2.399 | 6144 | 0.000 | 0.0001546 |
| triangular_solve_f32_32x32 | pytorch-cuda | 128 | 15 | 3.593 | 6144 | 0.000 | 9.242e-05 |
| triangular_solve_f32_32x32 | tenferro-cuda-direct | 128 | 15 | 2.291 | 6144 | 0.000 | 0.0001546 |
| triangular_solve_f32_32x32 | tenferro-cuda-eager | 64 | 15 | 3.198 | 6144 | 0.000 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda | 32 | 15 | 2.560 | 12288 | 0.000 | 1.609e-07 |
| triangular_solve_f64_32x32 | jax-cuda-jit | 32 | 15 | 2.350 | 12288 | 0.000 | 1.609e-07 |
| triangular_solve_f64_32x32 | pytorch-cuda | 128 | 15 | 3.366 | 12288 | 0.000 | 1.415e-07 |
| triangular_solve_f64_32x32 | tenferro-cuda-direct | 128 | 15 | 2.457 | 12288 | 0.000 | 1.609e-07 |
| triangular_solve_f64_32x32 | tenferro-cuda-eager | 64 | 15 | 3.118 | 12288 | 0.000 | 1.609e-07 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api_latency
suite_file: benchmarks/gpu/linalg_public_api_latency.yaml
timestamp: '2026-11-25T07:14:58Z'
tenferro_rs:
  path: /home/shinaoka/tensor4all/tenferro-benchmark-2701/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - cuda
  resolved_path: /home/shinaoka/tensor4all/tenferro-benchmark-2701/extern/tenferro-rs
harness:
  path: /home/shinaoka/tensor4all/tenferro-benchmark-2701
  commit: acdeaf6e4f6845a465352d7b7c3b67c7dbcf005a
  dirty: false
resolved_deps:
  strided-perm:
  - 0.8.0
  - 0.15.1
  strided-view:
  - 0.7.1
  - 0.8.1
  threadteam:
  - 0.1.0
  - 0.6.1
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
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: primerose
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 27,29
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
    OMP_THREAD_LIMIT: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_NUM_THREADS: '1'
    MATMUL_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    CUDA_HOME: /opt/tenferro-benchmark-ci/cuda
    CUDA_PATH: /opt/tenferro-benchmark-ci/cuda
    TENFERRO_CUTENSOR_PATH: /opt/tenferro-benchmark-ci/cutensor/lib/libcutensor.so.2
    UV_PROJECT_ENVIRONMENT: /opt/tenferro-benchmark-ci/venv
    BENCH_RUNS: '15'
    BENCH_WARMUPS: '3'
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
python_backends:
  pytorch:
    available: true
    version: 2.11.0+cu128
    file: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/torch/__init__.py
    provider: mkl
    blas_info: mkl
    lapack_info: mkl
    config: "PyTorch built with:\n  - GCC 13.3\n  - C++ Version: 201703\n  - Intel(R)\
      \ oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R)\
      \ 64 architecture applications\n  - Intel(R) MKL-DNN v3.10.2 (Git Hash f1d471933dc852f956fd05389f9313c7148783d5)\n\
      \  - OpenMP 201511 (a.k.a. OpenMP 4.5)\n  - LAPACK is enabled (usually provided\
      \ by MKL)\n  - NNPACK is enabled\n  - CPU capability usage: AVX2\n  - CUDA Runtime\
      \ 12.8\n  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120\n\
      \  - CuDNN 91.9  (built against CUDA 12.9)\n  - Magma 2.6.1\n  - Build settings:\
      \ BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=70d99e998b4955e0049d13a98d77ae1b14db1f45,\
      \ CUDA_VERSION=12.8, CUDNN_VERSION=9.19.0, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++,\
      \ CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO\
      \ -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_FBGEMM -DUSE_MSLK -DUSE_PYTORCH_QNNPACK\
      \ -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED\
      \ -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct\
      \ -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas\
      \ -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow\
      \ -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized\
      \ -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference\
      \ -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1,\
      \ PERF_WITH_AVX2=1, TORCH_VERSION=2.11.0, USE_CUDA=ON, USE_CUDNN=ON, USE_CUSPARSELT=1,\
      \ USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF,\
      \ USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF,\
      \ USE_XCCL=OFF, USE_XPU=OFF, \n"
    library: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
    linked_libraries:
    - /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1
  jax:
    available: true
    version: 0.10.1
    jaxlib_version: 0.10.1
    backend: gpu
    file: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/jaxlib/__init__.py
    provider: xla_gpu
    dot_backend: xla_gpu
    lapack_provider: none_detected
    library: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/jaxlib/_ifrt_proxy.so
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
  runtime_version: '12.8'
  cuda_version: '13.0'
  toolkit_version: null
  toolkit_root: /opt/tenferro-benchmark-ci/cuda
  cutensor_version: 2.2.0
  cusolver_version: 11.7.3
  cudnn_version: '91900'
  artifact_cache:
    path: /home/shinaoka/tensor4all/.t4a-509-gpu-runtime/home/.cache/tenferro-benchmark/cuda-artifacts/12.8.93/sm_80
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
  - 0.098481
  - 0.093451
  - 0.097982
  - 0.098202
  - 0.092732
  - 0.109452
  - 0.096662
  - 0.095601
  - 0.095702
  - 0.097352
  - 0.097802
  - 0.094632
  - 0.095832
  - 0.096141
  - 0.099882
  median_ms: 0.096662
  bytes_per_second: 1388526287475.947
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-25T07:15:10.230054+00:00'
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
    .devcontainer/cuda/devcontainer.json: 402f3763533ea1531d0e400193395302213fd1b151a42baaeaebeafca3acabcd
    .devcontainer/cuda/Dockerfile: 6eff822595fa44f53a6fd9eebe0f4e3cc9e28362d9cfa8cac977a21b10a823f2
    src/bin/benchmark_gpu_public_api.rs: b85ad2743b1594fe0a647253a17fa3bb59800202fa8ed4016d0255f0170bf5b3
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 3d5bedcc3620f75cab2687fdf49210df324021d043b23aec6e169642ceff15ba
    src/bin/benchmark_gpu_public_api/multi.rs: 950cb1b3c1a7cca37665b58e3de074bd906dafb8c84b17a51c27ce797a73afa8
    src/bin/benchmark_gpu_public_api/plan.rs: 016e752c59edefc3ded1405c36325e8680736860cb65954919ea4cf0f3eed061
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 88038508963fbc13303fb54b3898808e4303131321cabedb9f38255941ead438
    src/bin/benchmark_gpu_public_api/state.rs: e1cc2781ce7392154f32cc19372c9fc41ae29765aedc544c7ae7bf02f873c12b
    src/bin/benchmark_gpu_public_api/structural.rs: 10850458a640d11e79262c215072482ee463bc4ca0604f40d82ae56f345f6564
    src/bin/benchmark_gpu_public_api/reuse.rs: 8934cd12e336764b5ed831f0c92d55f2827142de939a1f33daa5d9ee57b9f482
    /home/shinaoka/tensor4all/tenferro-benchmark-2701/scripts/benchmark_gpu_public_api.py: d1ff68f86496d991bcbc730138fe6e3076001b3f23cc1a94c44b96f6857cb8ed
    /home/shinaoka/tensor4all/tenferro-benchmark-2701/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 7995bcf409258185e07210969a738896f64860b67df251fcd1100d11128a5431
    data/results/nvidia-gpu/gpu/linalg_public_api_latency/20261125_071457/fixtures/manifest.json: cfed1a8c0cf2329dcb1b8e98b0b088ab80a95266196d0c49cdec73475a9b3af1
    target/release/benchmark_gpu_public_api: 250c27c741c6e6364e4ff58525a6341f7e6a3c3b81e6f429f4c5662a4d8f0f86
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda-jit:
        arm: jax-cuda-jit
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-25T07:16:19Z'
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-25T07:16:01Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-25T07:15:43Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-25T07:15:18Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-25T07:15:26Z'
  compile:
    case_wall_seconds: 55.658
    compiling_processes: 0
    modules_compiled: 0
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 0.0
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| cholesky_f32_32x32 | tenferro-cuda-direct | cholesky_f32_32x32 untimed 183.7 ms is 541% of the 34.0 ms timed region (budget 20%) |
| svdvals_f32_32x32 | tenferro-cuda-direct | svdvals_f32_32x32 untimed 14.3 ms is 34% of the 42.5 ms timed region (budget 20%) |
| svdvals_f64_32x32 | tenferro-cuda-direct | svdvals_f64_32x32 untimed 10.4 ms is 35% of the 30.1 ms timed region (budget 20%) |
| eigvalsh_f32_32x32 | tenferro-cuda-direct | eigvalsh_f32_32x32 untimed 38.3 ms is 71% of the 53.9 ms timed region (budget 20%) |
| eigvalsh_f64_32x32 | tenferro-cuda-direct | eigvalsh_f64_32x32 untimed 7.1 ms is 22% of the 32.2 ms timed region (budget 20%) |
| det_f32_32x32 | tenferro-cuda-direct | det_f32_32x32 untimed 27.0 ms is 54% of the 50.3 ms timed region (budget 20%) |
| norm_f32_32x32 | tenferro-cuda-direct | norm_f32_32x32 untimed 8.5 ms is 24% of the 34.6 ms timed region (budget 20%) |
| norm_f64_32x32 | tenferro-cuda-direct | norm_f64_32x32 untimed 7.2 ms is 21% of the 34.3 ms timed region (budget 20%) |
| triangular_solve_f32_32x32 | tenferro-cuda-direct | triangular_solve_f32_32x32 untimed 17.5 ms is 51% of the 34.4 ms timed region (budget 20%) |
| triangular_solve_f64_32x32 | tenferro-cuda-direct | triangular_solve_f64_32x32 untimed 13.3 ms is 36% of the 36.9 ms timed region (budget 20%) |
| svd_full_square_f32_32x32 | tenferro-cuda-direct | svd_full_square_f32_32x32 untimed 11.0 ms is 28% of the 39.7 ms timed region (budget 20%) |
| svd_full_tall_f32_64x32 | tenferro-cuda-direct | svd_full_tall_f32_64x32 untimed 121.8 ms is 388% of the 31.4 ms timed region (budget 20%) |
| svd_full_wide_f32_32x64 | tenferro-cuda-direct | svd_full_wide_f32_32x64 untimed 14.1 ms is 42% of the 33.9 ms timed region (budget 20%) |
| svd_full_square_f64_32x32 | tenferro-cuda-direct | svd_full_square_f64_32x32 untimed 14.4 ms is 25% of the 58.5 ms timed region (budget 20%) |
| svd_full_tall_f64_64x32 | tenferro-cuda-direct | svd_full_tall_f64_64x32 untimed 202.4 ms is 459% of the 44.1 ms timed region (budget 20%) |
| svd_full_wide_f64_32x64 | tenferro-cuda-direct | svd_full_wide_f64_32x64 untimed 32.0 ms is 70% of the 45.4 ms timed region (budget 20%) |
| svd_full_square_c32_32x32 | tenferro-cuda-direct | svd_full_square_c32_32x32 untimed 14.9 ms is 25% of the 59.3 ms timed region (budget 20%) |
| svd_full_tall_c32_64x32 | tenferro-cuda-direct | svd_full_tall_c32_64x32 untimed 103.8 ms is 244% of the 42.5 ms timed region (budget 20%) |
| svd_full_wide_c32_32x64 | tenferro-cuda-direct | svd_full_wide_c32_32x64 untimed 16.6 ms is 39% of the 42.6 ms timed region (budget 20%) |
| svd_full_square_c64_32x32 | tenferro-cuda-direct | svd_full_square_c64_32x32 untimed 18.1 ms is 33% of the 54.6 ms timed region (budget 20%) |
| svd_full_tall_c64_64x32 | tenferro-cuda-direct | svd_full_tall_c64_64x32 untimed 430.0 ms is 1236% of the 34.8 ms timed region (budget 20%) |
| svd_full_wide_c64_32x64 | tenferro-cuda-direct | svd_full_wide_c64_32x64 untimed 20.4 ms is 55% of the 36.8 ms timed region (budget 20%) |
| householder_qr_square_f32_32x32 | tenferro-cuda-direct | householder_qr_square_f32_32x32 untimed 12.7 ms is 43% of the 29.3 ms timed region (budget 20%) |
| householder_qr_tall_f32_64x32 | tenferro-cuda-direct | householder_qr_tall_f32_64x32 untimed 9.0 ms is 23% of the 39.6 ms timed region (budget 20%) |
| householder_qr_square_f64_32x32 | tenferro-cuda-direct | householder_qr_square_f64_32x32 untimed 8.3 ms is 24% of the 33.9 ms timed region (budget 20%) |
| householder_qr_tall_f64_64x32 | tenferro-cuda-direct | householder_qr_tall_f64_64x32 untimed 10.0 ms is 23% of the 44.0 ms timed region (budget 20%) |
| householder_qr_square_c32_32x32 | tenferro-cuda-direct | householder_qr_square_c32_32x32 untimed 7.1 ms is 23% of the 31.6 ms timed region (budget 20%) |
| householder_qr_tall_c32_64x32 | tenferro-cuda-direct | householder_qr_tall_c32_64x32 untimed 9.9 ms is 22% of the 44.4 ms timed region (budget 20%) |
| householder_qr_square_c64_32x32 | tenferro-cuda-direct | householder_qr_square_c64_32x32 untimed 9.1 ms is 23% of the 39.7 ms timed region (budget 20%) |
| householder_qr_tall_c64_64x32 | tenferro-cuda-direct | householder_qr_tall_c64_64x32 untimed 11.0 ms is 21% of the 52.1 ms timed region (budget 20%) |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-direct | householder_from_factors_default_f32_64x32 untimed 10.2 ms is 29% of the 35.4 ms timed region (budget 20%) |
| householder_append_default_f32_64x32 | tenferro-cuda-direct | householder_append_default_f32_64x32 untimed 10.2 ms is 22% of the 45.3 ms timed region (budget 20%) |
| householder_r_default_f32_64x32 | tenferro-cuda-direct | householder_r_default_f32_64x32 untimed 6.8 ms is 22% of the 30.6 ms timed region (budget 20%) |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-direct | householder_from_factors_default_f64_64x32 untimed 9.1 ms is 25% of the 37.2 ms timed region (budget 20%) |
| householder_append_default_f64_64x32 | tenferro-cuda-direct | householder_append_default_f64_64x32 untimed 10.6 ms is 22% of the 49.0 ms timed region (budget 20%) |
| householder_r_default_f64_64x32 | tenferro-cuda-direct | householder_r_default_f64_64x32 untimed 7.4 ms is 22% of the 33.2 ms timed region (budget 20%) |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-direct | householder_from_factors_default_c32_64x32 untimed 9.5 ms is 25% of the 37.4 ms timed region (budget 20%) |
| householder_append_default_c32_64x32 | tenferro-cuda-direct | householder_append_default_c32_64x32 untimed 10.2 ms is 23% of the 44.9 ms timed region (budget 20%) |
| householder_r_default_c32_64x32 | tenferro-cuda-direct | householder_r_default_c32_64x32 untimed 6.5 ms is 21% of the 31.3 ms timed region (budget 20%) |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-direct | householder_from_factors_default_c64_64x32 untimed 10.2 ms is 25% of the 41.3 ms timed region (budget 20%) |
| householder_append_default_c64_64x32 | tenferro-cuda-direct | householder_append_default_c64_64x32 untimed 11.3 ms is 21% of the 53.2 ms timed region (budget 20%) |
| householder_r_default_c64_64x32 | tenferro-cuda-direct | householder_r_default_c64_64x32 untimed 7.0 ms is 21% of the 32.7 ms timed region (budget 20%) |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-direct | householder_q_columns_prefix_c64_64x32 untimed 9.8 ms is 25% of the 38.7 ms timed region (budget 20%) |
| lu_square_f32_s32 | tenferro-cuda-direct | lu_square_f32_s32 untimed 7.4 ms is 21% of the 35.4 ms timed region (budget 20%) |
| lu_square_f64_s32 | tenferro-cuda-direct | lu_square_f64_s32 untimed 7.5 ms is 20% of the 36.8 ms timed region (budget 20%) |
| lu_square_c32_s32 | tenferro-cuda-direct | lu_square_c32_s32 untimed 7.7 ms is 21% of the 36.7 ms timed region (budget 20%) |
| lu_square_c64_s32 | tenferro-cuda-direct | lu_square_c64_s32 untimed 7.8 ms is 21% of the 37.7 ms timed region (budget 20%) |
| lu_tall_f32_s32 | tenferro-cuda-direct | lu_tall_f32_s32 untimed 7.9 ms is 21% of the 36.7 ms timed region (budget 20%) |
| lu_tall_f64_s32 | tenferro-cuda-direct | lu_tall_f64_s32 untimed 7.9 ms is 21% of the 36.7 ms timed region (budget 20%) |
| lu_tall_c32_s32 | tenferro-cuda-direct | lu_tall_c32_s32 untimed 7.7 ms is 21% of the 36.8 ms timed region (budget 20%) |
| lu_tall_c64_s32 | tenferro-cuda-direct | lu_tall_c64_s32 untimed 8.1 ms is 22% of the 37.0 ms timed region (budget 20%) |
| lu_wide_f32_s32 | tenferro-cuda-direct | lu_wide_f32_s32 untimed 7.7 ms is 20% of the 37.7 ms timed region (budget 20%) |
| lu_wide_f64_s32 | tenferro-cuda-direct | lu_wide_f64_s32 untimed 7.9 ms is 20% of the 38.7 ms timed region (budget 20%) |
| lu_wide_c32_s32 | tenferro-cuda-direct | lu_wide_c32_s32 untimed 22.9 ms is 60% of the 38.3 ms timed region (budget 20%) |
| lu_wide_c64_s32 | tenferro-cuda-direct | lu_wide_c64_s32 untimed 21.8 ms is 54% of the 40.3 ms timed region (budget 20%) |
| cholesky_f32_32x32 | tenferro-cuda-eager | cholesky_f32_32x32 untimed 282.5 ms is 592% of the 47.7 ms timed region (budget 20%) |
| eigvalsh_f32_32x32 | tenferro-cuda-eager | eigvalsh_f32_32x32 untimed 50.4 ms is 162% of the 31.2 ms timed region (budget 20%) |
| eigvalsh_f64_32x32 | tenferro-cuda-eager | eigvalsh_f64_32x32 untimed 8.6 ms is 24% of the 36.5 ms timed region (budget 20%) |
| det_f32_32x32 | tenferro-cuda-eager | det_f32_32x32 untimed 7.5 ms is 25% of the 30.0 ms timed region (budget 20%) |
| norm_f32_32x32 | tenferro-cuda-eager | norm_f32_32x32 untimed 10.6 ms is 30% of the 35.5 ms timed region (budget 20%) |
| norm_f64_32x32 | tenferro-cuda-eager | norm_f64_32x32 untimed 8.8 ms is 25% of the 35.9 ms timed region (budget 20%) |
| triangular_solve_f64_32x32 | tenferro-cuda-eager | triangular_solve_f64_32x32 untimed 9.4 ms is 20% of the 46.9 ms timed region (budget 20%) |
| svd_full_square_f32_32x32 | tenferro-cuda-eager | svd_full_square_f32_32x32 untimed 10.2 ms is 25% of the 40.7 ms timed region (budget 20%) |
| svd_full_tall_f32_64x32 | tenferro-cuda-eager | svd_full_tall_f32_64x32 untimed 27.8 ms is 86% of the 32.2 ms timed region (budget 20%) |
| svd_full_wide_f32_32x64 | tenferro-cuda-eager | svd_full_wide_f32_32x64 untimed 13.6 ms is 40% of the 34.4 ms timed region (budget 20%) |
| svd_full_square_f64_32x32 | tenferro-cuda-eager | svd_full_square_f64_32x32 untimed 14.2 ms is 24% of the 58.7 ms timed region (budget 20%) |
| svd_full_tall_f64_64x32 | tenferro-cuda-eager | svd_full_tall_f64_64x32 untimed 21.7 ms is 48% of the 45.2 ms timed region (budget 20%) |
| svd_full_wide_f64_32x64 | tenferro-cuda-eager | svd_full_wide_f64_32x64 untimed 17.2 ms is 38% of the 45.7 ms timed region (budget 20%) |
| svd_full_square_c32_32x32 | tenferro-cuda-eager | svd_full_square_c32_32x32 untimed 10.8 ms is 36% of the 30.3 ms timed region (budget 20%) |
| svd_full_tall_c32_64x32 | tenferro-cuda-eager | svd_full_tall_c32_64x32 untimed 28.5 ms is 66% of the 43.2 ms timed region (budget 20%) |
| svd_full_wide_c32_32x64 | tenferro-cuda-eager | svd_full_wide_c32_32x64 untimed 16.7 ms is 38% of the 43.7 ms timed region (budget 20%) |
| slogdet_positive_c32_32x32 | tenferro-cuda-eager | slogdet_positive_c32_32x32 untimed 7.1 ms is 23% of the 30.8 ms timed region (budget 20%) |
| slogdet_negative_c32_32x32 | tenferro-cuda-eager | slogdet_negative_c32_32x32 untimed 6.5 ms is 21% of the 30.6 ms timed region (budget 20%) |
| svd_full_square_c64_32x32 | tenferro-cuda-eager | svd_full_square_c64_32x32 untimed 19.1 ms is 34% of the 56.3 ms timed region (budget 20%) |
| svd_full_tall_c64_64x32 | tenferro-cuda-eager | svd_full_tall_c64_64x32 untimed 81.6 ms is 232% of the 35.2 ms timed region (budget 20%) |
| svd_full_wide_c64_32x64 | tenferro-cuda-eager | svd_full_wide_c64_32x64 untimed 21.4 ms is 58% of the 37.1 ms timed region (budget 20%) |
| slogdet_positive_c64_32x32 | tenferro-cuda-eager | slogdet_positive_c64_32x32 untimed 7.0 ms is 23% of the 30.6 ms timed region (budget 20%) |
| slogdet_negative_c64_32x32 | tenferro-cuda-eager | slogdet_negative_c64_32x32 untimed 6.5 ms is 21% of the 30.9 ms timed region (budget 20%) |
| householder_qr_square_f32_32x32 | tenferro-cuda-eager | householder_qr_square_f32_32x32 untimed 15.1 ms is 41% of the 36.9 ms timed region (budget 20%) |
| householder_qr_tall_f32_64x32 | tenferro-cuda-eager | householder_qr_tall_f32_64x32 untimed 10.6 ms is 22% of the 48.1 ms timed region (budget 20%) |
| householder_qr_wide_f32_32x64 | tenferro-cuda-eager | householder_qr_wide_f32_32x64 untimed 9.8 ms is 20% of the 48.2 ms timed region (budget 20%) |
| householder_qr_square_f64_32x32 | tenferro-cuda-eager | householder_qr_square_f64_32x32 untimed 9.7 ms is 23% of the 41.5 ms timed region (budget 20%) |
| householder_qr_tall_f64_64x32 | tenferro-cuda-eager | householder_qr_tall_f64_64x32 untimed 12.2 ms is 23% of the 52.9 ms timed region (budget 20%) |
| householder_qr_wide_f64_32x64 | tenferro-cuda-eager | householder_qr_wide_f64_32x64 untimed 10.5 ms is 20% of the 51.3 ms timed region (budget 20%) |
| householder_qr_square_c32_32x32 | tenferro-cuda-eager | householder_qr_square_c32_32x32 untimed 8.0 ms is 21% of the 37.6 ms timed region (budget 20%) |
| householder_qr_tall_c32_64x32 | tenferro-cuda-eager | householder_qr_tall_c32_64x32 untimed 11.0 ms is 21% of the 51.6 ms timed region (budget 20%) |
| householder_qr_wide_c32_32x64 | tenferro-cuda-eager | householder_qr_wide_c32_32x64 untimed 10.6 ms is 20% of the 52.3 ms timed region (budget 20%) |
| slogdet_phase_c32_32x32 | tenferro-cuda-eager | slogdet_phase_c32_32x32 untimed 6.5 ms is 22% of the 29.8 ms timed region (budget 20%) |
| householder_qr_square_c64_32x32 | tenferro-cuda-eager | householder_qr_square_c64_32x32 untimed 10.4 ms is 23% of the 46.1 ms timed region (budget 20%) |
| householder_qr_tall_c64_64x32 | tenferro-cuda-eager | householder_qr_tall_c64_64x32 untimed 12.4 ms is 21% of the 59.0 ms timed region (budget 20%) |
| householder_qr_wide_c64_32x64 | tenferro-cuda-eager | householder_qr_wide_c64_32x64 untimed 7.4 ms is 24% of the 30.8 ms timed region (budget 20%) |
| slogdet_phase_c64_32x32 | tenferro-cuda-eager | slogdet_phase_c64_32x32 untimed 6.9 ms is 23% of the 30.4 ms timed region (budget 20%) |
| householder_from_factors_default_f32_64x32 | tenferro-cuda-eager | householder_from_factors_default_f32_64x32 untimed 11.2 ms is 27% of the 40.9 ms timed region (budget 20%) |
| householder_append_default_f32_64x32 | tenferro-cuda-eager | householder_append_default_f32_64x32 untimed 8.1 ms is 28% of the 28.7 ms timed region (budget 20%) |
| householder_r_default_f32_64x32 | tenferro-cuda-eager | householder_r_default_f32_64x32 untimed 10.1 ms is 21% of the 47.5 ms timed region (budget 20%) |
| householder_from_factors_default_f64_64x32 | tenferro-cuda-eager | householder_from_factors_default_f64_64x32 untimed 11.0 ms is 26% of the 42.3 ms timed region (budget 20%) |
| householder_append_default_f64_64x32 | tenferro-cuda-eager | householder_append_default_f64_64x32 untimed 8.3 ms is 28% of the 30.1 ms timed region (budget 20%) |
| householder_r_default_f64_64x32 | tenferro-cuda-eager | householder_r_default_f64_64x32 untimed 10.5 ms is 22% of the 48.0 ms timed region (budget 20%) |
| householder_from_factors_default_c32_64x32 | tenferro-cuda-eager | householder_from_factors_default_c32_64x32 untimed 10.4 ms is 25% of the 41.6 ms timed region (budget 20%) |
| householder_append_default_c32_64x32 | tenferro-cuda-eager | householder_append_default_c32_64x32 untimed 11.9 ms is 21% of the 55.9 ms timed region (budget 20%) |
| householder_from_factors_default_c64_64x32 | tenferro-cuda-eager | householder_from_factors_default_c64_64x32 untimed 11.6 ms is 25% of the 46.7 ms timed region (budget 20%) |
| householder_append_default_c64_64x32 | tenferro-cuda-eager | householder_append_default_c64_64x32 untimed 9.1 ms is 29% of the 31.7 ms timed region (budget 20%) |
| householder_r_default_c64_64x32 | tenferro-cuda-eager | householder_r_default_c64_64x32 untimed 10.7 ms is 21% of the 50.4 ms timed region (budget 20%) |
| householder_q_columns_prefix_c64_64x32 | tenferro-cuda-eager | householder_q_columns_prefix_c64_64x32 untimed 9.5 ms is 21% of the 45.1 ms timed region (budget 20%) |
| lu_square_f32_s32 | tenferro-cuda-eager | lu_square_f32_s32 untimed 8.2 ms is 21% of the 39.4 ms timed region (budget 20%) |
| lu_square_f64_s32 | tenferro-cuda-eager | lu_square_f64_s32 untimed 8.5 ms is 21% of the 40.0 ms timed region (budget 20%) |
| lu_square_c32_s32 | tenferro-cuda-eager | lu_square_c32_s32 untimed 8.7 ms is 21% of the 40.8 ms timed region (budget 20%) |
| lu_square_c64_s32 | tenferro-cuda-eager | lu_square_c64_s32 untimed 9.0 ms is 21% of the 43.4 ms timed region (budget 20%) |
| lu_tall_f32_s32 | tenferro-cuda-eager | lu_tall_f32_s32 untimed 8.6 ms is 22% of the 39.9 ms timed region (budget 20%) |
| lu_tall_f64_s32 | tenferro-cuda-eager | lu_tall_f64_s32 untimed 9.0 ms is 22% of the 41.2 ms timed region (budget 20%) |
| lu_tall_c32_s32 | tenferro-cuda-eager | lu_tall_c32_s32 untimed 8.7 ms is 21% of the 41.6 ms timed region (budget 20%) |
| lu_tall_c64_s32 | tenferro-cuda-eager | lu_tall_c64_s32 untimed 9.1 ms is 22% of the 42.2 ms timed region (budget 20%) |
| lu_wide_f32_s32 | tenferro-cuda-eager | lu_wide_f32_s32 untimed 8.6 ms is 20% of the 42.2 ms timed region (budget 20%) |
| lu_wide_f64_s32 | tenferro-cuda-eager | lu_wide_f64_s32 untimed 8.8 ms is 20% of the 43.4 ms timed region (budget 20%) |
| lu_wide_c32_s32 | tenferro-cuda-eager | lu_wide_c32_s32 untimed 16.7 ms is 39% of the 43.3 ms timed region (budget 20%) |
| lu_wide_c64_s32 | tenferro-cuda-eager | lu_wide_c64_s32 untimed 17.5 ms is 39% of the 44.9 ms timed region (budget 20%) |
| cholesky_f32_32x32 | pytorch-cuda | cholesky_f32_32x32 untimed 605.3 ms is 1861% of the 32.5 ms timed region (budget 20%) |
| cholesky_f64_32x32 | pytorch-cuda | cholesky_f64_32x32 untimed 9.8 ms is 28% of the 34.7 ms timed region (budget 20%) |
| svdvals_f32_32x32 | pytorch-cuda | svdvals_f32_32x32 untimed 32.8 ms is 65% of the 50.4 ms timed region (budget 20%) |
| svdvals_f64_32x32 | pytorch-cuda | svdvals_f64_32x32 untimed 12.2 ms is 28% of the 44.0 ms timed region (budget 20%) |
| eigvalsh_f32_32x32 | pytorch-cuda | eigvalsh_f32_32x32 untimed 26.5 ms is 77% of the 34.4 ms timed region (budget 20%) |
| eigvalsh_f64_32x32 | pytorch-cuda | eigvalsh_f64_32x32 untimed 58.2 ms is 172% of the 33.8 ms timed region (budget 20%) |
| pinv_f32_32x32 | pytorch-cuda | pinv_f32_32x32 untimed 266.1 ms is 848% of the 31.4 ms timed region (budget 20%) |
| pinv_f64_32x32 | pytorch-cuda | pinv_f64_32x32 untimed 57.3 ms is 110% of the 52.1 ms timed region (budget 20%) |
| inv_f32_32x32 | pytorch-cuda | inv_f32_32x32 untimed 90.7 ms is 181% of the 50.2 ms timed region (budget 20%) |
| inv_f64_32x32 | pytorch-cuda | inv_f64_32x32 untimed 11.7 ms is 23% of the 50.1 ms timed region (budget 20%) |
| det_f32_32x32 | pytorch-cuda | det_f32_32x32 untimed 926.7 ms is 1809% of the 51.2 ms timed region (budget 20%) |
| det_f64_32x32 | pytorch-cuda | det_f64_32x32 untimed 11.4 ms is 23% of the 50.6 ms timed region (budget 20%) |
| norm_f32_32x32 | pytorch-cuda | norm_f32_32x32 untimed 56.6 ms is 102% of the 55.4 ms timed region (budget 20%) |
| norm_f64_32x32 | pytorch-cuda | norm_f64_32x32 untimed 13.5 ms is 23% of the 58.1 ms timed region (budget 20%) |
| triangular_solve_f32_32x32 | pytorch-cuda | triangular_solve_f32_32x32 untimed 10.8 ms is 20% of the 53.7 ms timed region (budget 20%) |
| svd_full_square_f32_32x32 | pytorch-cuda | svd_full_square_f32_32x32 untimed 10.8 ms is 25% of the 43.2 ms timed region (budget 20%) |
| svd_full_tall_f32_64x32 | pytorch-cuda | svd_full_tall_f32_64x32 untimed 35.7 ms is 112% of the 32.0 ms timed region (budget 20%) |
| svd_full_wide_f32_32x64 | pytorch-cuda | svd_full_wide_f32_32x64 untimed 15.3 ms is 45% of the 34.3 ms timed region (budget 20%) |
| slogdet_positive_f32_32x32 | pytorch-cuda | slogdet_positive_f32_32x32 untimed 38.4 ms is 109% of the 35.2 ms timed region (budget 20%) |
| slogdet_negative_f32_32x32 | pytorch-cuda | slogdet_negative_f32_32x32 untimed 8.1 ms is 24% of the 33.8 ms timed region (budget 20%) |
| svd_full_square_f64_32x32 | pytorch-cuda | svd_full_square_f64_32x32 untimed 11.1 ms is 30% of the 36.8 ms timed region (budget 20%) |
| svd_full_tall_f64_64x32 | pytorch-cuda | svd_full_tall_f64_64x32 untimed 23.3 ms is 51% of the 45.7 ms timed region (budget 20%) |
| svd_full_wide_f64_32x64 | pytorch-cuda | svd_full_wide_f64_32x64 untimed 19.7 ms is 41% of the 48.4 ms timed region (budget 20%) |
| slogdet_positive_f64_32x32 | pytorch-cuda | slogdet_positive_f64_32x32 untimed 7.8 ms is 23% of the 33.5 ms timed region (budget 20%) |
| slogdet_negative_f64_32x32 | pytorch-cuda | slogdet_negative_f64_32x32 untimed 8.2 ms is 23% of the 35.1 ms timed region (budget 20%) |
| svd_full_square_c32_32x32 | pytorch-cuda | svd_full_square_c32_32x32 untimed 14.4 ms is 40% of the 36.4 ms timed region (budget 20%) |
| svd_full_tall_c32_64x32 | pytorch-cuda | svd_full_tall_c32_64x32 untimed 32.8 ms is 77% of the 42.4 ms timed region (budget 20%) |
| svd_full_wide_c32_32x64 | pytorch-cuda | svd_full_wide_c32_32x64 untimed 17.3 ms is 40% of the 43.4 ms timed region (budget 20%) |
| slogdet_positive_c32_32x32 | pytorch-cuda | slogdet_positive_c32_32x32 untimed 2779.4 ms is 8434% of the 33.0 ms timed region (budget 20%) |
| slogdet_negative_c32_32x32 | pytorch-cuda | slogdet_negative_c32_32x32 untimed 7.7 ms is 23% of the 32.8 ms timed region (budget 20%) |
| svd_full_square_c64_32x32 | pytorch-cuda | svd_full_square_c64_32x32 untimed 14.7 ms is 39% of the 37.3 ms timed region (budget 20%) |
| svd_full_tall_c64_64x32 | pytorch-cuda | svd_full_tall_c64_64x32 untimed 495.9 ms is 1426% of the 34.8 ms timed region (budget 20%) |
| svd_full_wide_c64_32x64 | pytorch-cuda | svd_full_wide_c64_32x64 untimed 21.8 ms is 59% of the 36.6 ms timed region (budget 20%) |
| slogdet_positive_c64_32x32 | pytorch-cuda | slogdet_positive_c64_32x32 untimed 4913.7 ms is 14289% of the 34.4 ms timed region (budget 20%) |
| slogdet_negative_c64_32x32 | pytorch-cuda | slogdet_negative_c64_32x32 untimed 7.6 ms is 22% of the 34.7 ms timed region (budget 20%) |
| householder_qr_square_f32_32x32 | pytorch-cuda | householder_qr_square_f32_32x32 untimed 10.1 ms is 25% of the 40.2 ms timed region (budget 20%) |
| householder_qr_tall_f32_64x32 | pytorch-cuda | householder_qr_tall_f32_64x32 untimed 9.4 ms is 31% of the 30.0 ms timed region (budget 20%) |
| householder_qr_wide_f32_32x64 | pytorch-cuda | householder_qr_wide_f32_32x64 untimed 13.2 ms is 22% of the 59.5 ms timed region (budget 20%) |
| householder_qr_square_f64_32x32 | pytorch-cuda | householder_qr_square_f64_32x32 untimed 11.7 ms is 25% of the 47.7 ms timed region (budget 20%) |
| householder_qr_tall_f64_64x32 | pytorch-cuda | householder_qr_tall_f64_64x32 untimed 9.8 ms is 29% of the 33.3 ms timed region (budget 20%) |
| householder_qr_wide_f64_32x64 | pytorch-cuda | householder_qr_wide_f64_32x64 untimed 9.6 ms is 28% of the 33.7 ms timed region (budget 20%) |
| householder_qr_square_c32_32x32 | pytorch-cuda | householder_qr_square_c32_32x32 untimed 11.1 ms is 25% of the 45.5 ms timed region (budget 20%) |
| householder_qr_tall_c32_64x32 | pytorch-cuda | householder_qr_tall_c32_64x32 untimed 11.0 ms is 32% of the 34.8 ms timed region (budget 20%) |
| householder_qr_wide_c32_32x64 | pytorch-cuda | householder_qr_wide_c32_32x64 untimed 10.1 ms is 29% of the 35.0 ms timed region (budget 20%) |
| slogdet_phase_c32_32x32 | pytorch-cuda | slogdet_phase_c32_32x32 untimed 8.2 ms is 24% of the 33.5 ms timed region (budget 20%) |
| householder_qr_square_c64_32x32 | pytorch-cuda | householder_qr_square_c64_32x32 untimed 9.6 ms is 32% of the 30.4 ms timed region (budget 20%) |
| householder_qr_tall_c64_64x32 | pytorch-cuda | householder_qr_tall_c64_64x32 untimed 12.4 ms is 30% of the 41.6 ms timed region (budget 20%) |
| householder_qr_wide_c64_32x64 | pytorch-cuda | householder_qr_wide_c64_32x64 untimed 11.6 ms is 27% of the 43.8 ms timed region (budget 20%) |
| slogdet_phase_c64_32x32 | pytorch-cuda | slogdet_phase_c64_32x32 untimed 8.3 ms is 25% of the 33.4 ms timed region (budget 20%) |
| householder_r_default_f32_64x32 | pytorch-cuda | householder_r_default_f32_64x32 untimed 12.7 ms is 31% of the 40.6 ms timed region (budget 20%) |
| householder_q_columns_prefix_f32_64x32 | pytorch-cuda | householder_q_columns_prefix_f32_64x32 untimed 8.4 ms is 28% of the 30.7 ms timed region (budget 20%) |
| householder_q_columns_complement_f32_64x32 | pytorch-cuda | householder_q_columns_complement_f32_64x32 untimed 12.1 ms is 20% of the 59.4 ms timed region (budget 20%) |
| householder_r_default_f64_64x32 | pytorch-cuda | householder_r_default_f64_64x32 untimed 12.3 ms is 31% of the 39.9 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_64x32 | pytorch-cuda | householder_q_columns_prefix_f64_64x32 untimed 7.6 ms is 24% of the 31.7 ms timed region (budget 20%) |
| householder_q_columns_complement_f64_64x32 | pytorch-cuda | householder_q_columns_complement_f64_64x32 untimed 8.3 ms is 25% of the 33.2 ms timed region (budget 20%) |
| householder_r_default_c32_64x32 | pytorch-cuda | householder_r_default_c32_64x32 untimed 11.8 ms is 31% of the 38.7 ms timed region (budget 20%) |
| householder_q_columns_complement_c32_64x32 | pytorch-cuda | householder_q_columns_complement_c32_64x32 untimed 8.2 ms is 26% of the 31.7 ms timed region (budget 20%) |
| householder_r_default_c64_64x32 | pytorch-cuda | householder_r_default_c64_64x32 untimed 12.5 ms is 31% of the 40.5 ms timed region (budget 20%) |
| householder_q_columns_prefix_c64_64x32 | pytorch-cuda | householder_q_columns_prefix_c64_64x32 untimed 9.9 ms is 32% of the 31.2 ms timed region (budget 20%) |
| householder_q_columns_complement_c64_64x32 | pytorch-cuda | householder_q_columns_complement_c64_64x32 untimed 8.3 ms is 24% of the 34.4 ms timed region (budget 20%) |
| cholesky_f32_32x32 | jax-cuda | cholesky_f32_32x32 untimed 962.6 ms is 2927% of the 32.9 ms timed region (budget 20%) |
| cholesky_f64_32x32 | jax-cuda | cholesky_f64_32x32 untimed 256.7 ms is 719% of the 35.7 ms timed region (budget 20%) |
| svdvals_f32_32x32 | jax-cuda | svdvals_f32_32x32 untimed 220.0 ms is 494% of the 44.6 ms timed region (budget 20%) |
| svdvals_f64_32x32 | jax-cuda | svdvals_f64_32x32 untimed 165.4 ms is 528% of the 31.4 ms timed region (budget 20%) |
| eigvalsh_f32_32x32 | jax-cuda | eigvalsh_f32_32x32 untimed 141.0 ms is 404% of the 34.9 ms timed region (budget 20%) |
| eigvalsh_f64_32x32 | jax-cuda | eigvalsh_f64_32x32 untimed 91.9 ms is 204% of the 45.0 ms timed region (budget 20%) |
| pinv_f32_32x32 | jax-cuda | pinv_f32_32x32 untimed 813.6 ms is 2718% of the 29.9 ms timed region (budget 20%) |
| pinv_f64_32x32 | jax-cuda | pinv_f64_32x32 untimed 663.4 ms is 1687% of the 39.3 ms timed region (budget 20%) |
| inv_f32_32x32 | jax-cuda | inv_f32_32x32 untimed 362.2 ms is 1210% of the 29.9 ms timed region (budget 20%) |
| inv_f64_32x32 | jax-cuda | inv_f64_32x32 untimed 340.2 ms is 1102% of the 30.9 ms timed region (budget 20%) |
| det_f32_32x32 | jax-cuda | det_f32_32x32 untimed 395.4 ms is 754% of the 52.4 ms timed region (budget 20%) |
| det_f64_32x32 | jax-cuda | det_f64_32x32 untimed 251.0 ms is 583% of the 43.0 ms timed region (budget 20%) |
| norm_f32_32x32 | jax-cuda | norm_f32_32x32 untimed 73.7 ms is 183% of the 40.2 ms timed region (budget 20%) |
| norm_f64_32x32 | jax-cuda | norm_f64_32x32 untimed 88.9 ms is 213% of the 41.7 ms timed region (budget 20%) |
| triangular_solve_f32_32x32 | jax-cuda | triangular_solve_f32_32x32 untimed 242.9 ms is 620% of the 39.2 ms timed region (budget 20%) |
| triangular_solve_f64_32x32 | jax-cuda | triangular_solve_f64_32x32 untimed 192.3 ms is 495% of the 38.9 ms timed region (budget 20%) |
| svd_full_square_f32_32x32 | jax-cuda | svd_full_square_f32_32x32 untimed 233.5 ms is 568% of the 41.1 ms timed region (budget 20%) |
| svd_full_tall_f32_64x32 | jax-cuda | svd_full_tall_f32_64x32 untimed 337.3 ms is 1049% of the 32.2 ms timed region (budget 20%) |
| svd_full_wide_f32_32x64 | jax-cuda | svd_full_wide_f32_32x64 untimed 179.9 ms is 523% of the 34.4 ms timed region (budget 20%) |
| slogdet_positive_f32_32x32 | jax-cuda | slogdet_positive_f32_32x32 untimed 214.7 ms is 416% of the 51.7 ms timed region (budget 20%) |
| slogdet_negative_f32_32x32 | jax-cuda | slogdet_negative_f32_32x32 untimed 13.1 ms is 26% of the 51.1 ms timed region (budget 20%) |
| svd_full_square_f64_32x32 | jax-cuda | svd_full_square_f64_32x32 untimed 199.0 ms is 341% of the 58.3 ms timed region (budget 20%) |
| svd_full_tall_f64_64x32 | jax-cuda | svd_full_tall_f64_64x32 untimed 254.9 ms is 573% of the 44.5 ms timed region (budget 20%) |
| svd_full_wide_f64_32x64 | jax-cuda | svd_full_wide_f64_32x64 untimed 193.9 ms is 412% of the 47.1 ms timed region (budget 20%) |
| slogdet_positive_f64_32x32 | jax-cuda | slogdet_positive_f64_32x32 untimed 206.4 ms is 411% of the 50.3 ms timed region (budget 20%) |
| slogdet_negative_f64_32x32 | jax-cuda | slogdet_negative_f64_32x32 untimed 13.5 ms is 27% of the 49.3 ms timed region (budget 20%) |
| svd_full_square_c32_32x32 | jax-cuda | svd_full_square_c32_32x32 untimed 135.8 ms is 227% of the 59.8 ms timed region (budget 20%) |
| svd_full_tall_c32_64x32 | jax-cuda | svd_full_tall_c32_64x32 untimed 307.1 ms is 725% of the 42.3 ms timed region (budget 20%) |
| svd_full_wide_c32_32x64 | jax-cuda | svd_full_wide_c32_32x64 untimed 160.0 ms is 373% of the 42.9 ms timed region (budget 20%) |
| slogdet_positive_c32_32x32 | jax-cuda | slogdet_positive_c32_32x32 untimed 202.6 ms is 375% of the 54.0 ms timed region (budget 20%) |
| slogdet_negative_c32_32x32 | jax-cuda | slogdet_negative_c32_32x32 untimed 13.0 ms is 27% of the 48.8 ms timed region (budget 20%) |
| svd_full_square_c64_32x32 | jax-cuda | svd_full_square_c64_32x32 untimed 158.1 ms is 269% of the 58.7 ms timed region (budget 20%) |
| svd_full_tall_c64_64x32 | jax-cuda | svd_full_tall_c64_64x32 untimed 252.2 ms is 699% of the 36.1 ms timed region (budget 20%) |
| svd_full_wide_c64_32x64 | jax-cuda | svd_full_wide_c64_32x64 untimed 179.1 ms is 464% of the 38.6 ms timed region (budget 20%) |
| slogdet_positive_c64_32x32 | jax-cuda | slogdet_positive_c64_32x32 untimed 299.3 ms is 564% of the 53.1 ms timed region (budget 20%) |
| slogdet_negative_c64_32x32 | jax-cuda | slogdet_negative_c64_32x32 untimed 13.9 ms is 26% of the 52.5 ms timed region (budget 20%) |
| householder_qr_square_f32_32x32 | jax-cuda | householder_qr_square_f32_32x32 untimed 100.7 ms is 247% of the 40.8 ms timed region (budget 20%) |
| householder_qr_tall_f32_64x32 | jax-cuda | householder_qr_tall_f32_64x32 untimed 108.4 ms is 343% of the 31.6 ms timed region (budget 20%) |
| householder_qr_wide_f32_32x64 | jax-cuda | householder_qr_wide_f32_32x64 untimed 98.2 ms is 309% of the 31.8 ms timed region (budget 20%) |
| householder_qr_square_f64_32x32 | jax-cuda | householder_qr_square_f64_32x32 untimed 88.3 ms is 180% of the 48.9 ms timed region (budget 20%) |
| householder_qr_tall_f64_64x32 | jax-cuda | householder_qr_tall_f64_64x32 untimed 94.0 ms is 268% of the 35.1 ms timed region (budget 20%) |
| householder_qr_wide_f64_32x64 | jax-cuda | householder_qr_wide_f64_32x64 untimed 97.8 ms is 279% of the 35.1 ms timed region (budget 20%) |
| householder_qr_square_c32_32x32 | jax-cuda | householder_qr_square_c32_32x32 untimed 95.6 ms is 204% of the 46.8 ms timed region (budget 20%) |
| householder_qr_tall_c32_64x32 | jax-cuda | householder_qr_tall_c32_64x32 untimed 101.4 ms is 280% of the 36.2 ms timed region (budget 20%) |
| householder_qr_wide_c32_32x64 | jax-cuda | householder_qr_wide_c32_32x64 untimed 100.0 ms is 276% of the 36.3 ms timed region (budget 20%) |
| slogdet_phase_c32_32x32 | jax-cuda | slogdet_phase_c32_32x32 untimed 13.1 ms is 27% of the 49.2 ms timed region (budget 20%) |
| householder_qr_square_c64_32x32 | jax-cuda | householder_qr_square_c64_32x32 untimed 95.7 ms is 302% of the 31.7 ms timed region (budget 20%) |
| householder_qr_tall_c64_64x32 | jax-cuda | householder_qr_tall_c64_64x32 untimed 106.6 ms is 246% of the 43.2 ms timed region (budget 20%) |
| householder_qr_wide_c64_32x64 | jax-cuda | householder_qr_wide_c64_32x64 untimed 101.5 ms is 224% of the 45.3 ms timed region (budget 20%) |
| slogdet_phase_c64_32x32 | jax-cuda | slogdet_phase_c64_32x32 untimed 17.6 ms is 32% of the 55.2 ms timed region (budget 20%) |
| householder_r_default_f32_64x32 | jax-cuda | householder_r_default_f32_64x32 untimed 232.2 ms is 548% of the 42.4 ms timed region (budget 20%) |
| householder_q_columns_prefix_f32_64x32 | jax-cuda | householder_q_columns_prefix_f32_64x32 untimed 349.6 ms is 999% of the 35.0 ms timed region (budget 20%) |
| householder_r_default_f64_64x32 | jax-cuda | householder_r_default_f64_64x32 untimed 243.0 ms is 721% of the 33.7 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_64x32 | jax-cuda | householder_q_columns_prefix_f64_64x32 untimed 343.7 ms is 945% of the 36.4 ms timed region (budget 20%) |
| householder_r_default_c32_64x32 | jax-cuda | householder_r_default_c32_64x32 untimed 278.7 ms is 635% of the 43.9 ms timed region (budget 20%) |
| householder_q_columns_prefix_c32_64x32 | jax-cuda | householder_q_columns_prefix_c32_64x32 untimed 272.0 ms is 803% of the 33.9 ms timed region (budget 20%) |
| householder_r_default_c64_64x32 | jax-cuda | householder_r_default_c64_64x32 untimed 266.6 ms is 632% of the 42.2 ms timed region (budget 20%) |
| householder_q_columns_prefix_c64_64x32 | jax-cuda | householder_q_columns_prefix_c64_64x32 untimed 247.3 ms is 728% of the 34.0 ms timed region (budget 20%) |
| cholesky_f32_32x32 | jax-cuda-jit | cholesky_f32_32x32 untimed 821.0 ms is 2505% of the 32.8 ms timed region (budget 20%) |
| cholesky_f64_32x32 | jax-cuda-jit | cholesky_f64_32x32 untimed 210.6 ms is 646% of the 32.6 ms timed region (budget 20%) |
| svdvals_f32_32x32 | jax-cuda-jit | svdvals_f32_32x32 untimed 126.9 ms is 294% of the 43.1 ms timed region (budget 20%) |
| svdvals_f64_32x32 | jax-cuda-jit | svdvals_f64_32x32 untimed 132.9 ms is 431% of the 30.8 ms timed region (budget 20%) |
| eigvalsh_f32_32x32 | jax-cuda-jit | eigvalsh_f32_32x32 untimed 117.5 ms is 332% of the 35.4 ms timed region (budget 20%) |
| eigvalsh_f64_32x32 | jax-cuda-jit | eigvalsh_f64_32x32 untimed 110.9 ms is 253% of the 43.8 ms timed region (budget 20%) |
| pinv_f32_32x32 | jax-cuda-jit | pinv_f32_32x32 untimed 721.9 ms is 1625% of the 44.4 ms timed region (budget 20%) |
| pinv_f64_32x32 | jax-cuda-jit | pinv_f64_32x32 untimed 528.9 ms is 1660% of the 31.9 ms timed region (budget 20%) |
| inv_f32_32x32 | jax-cuda-jit | inv_f32_32x32 untimed 236.8 ms is 374% of the 63.4 ms timed region (budget 20%) |
| inv_f64_32x32 | jax-cuda-jit | inv_f64_32x32 untimed 256.4 ms is 393% of the 65.2 ms timed region (budget 20%) |
| det_f32_32x32 | jax-cuda-jit | det_f32_32x32 untimed 268.0 ms is 653% of the 41.0 ms timed region (budget 20%) |
| det_f64_32x32 | jax-cuda-jit | det_f64_32x32 untimed 197.5 ms is 471% of the 41.9 ms timed region (budget 20%) |
| norm_f32_32x32 | jax-cuda-jit | norm_f32_32x32 untimed 99.3 ms is 240% of the 41.4 ms timed region (budget 20%) |
| norm_f64_32x32 | jax-cuda-jit | norm_f64_32x32 untimed 110.6 ms is 272% of the 40.7 ms timed region (budget 20%) |
| triangular_solve_f32_32x32 | jax-cuda-jit | triangular_solve_f32_32x32 untimed 257.0 ms is 703% of the 36.5 ms timed region (budget 20%) |
| triangular_solve_f64_32x32 | jax-cuda-jit | triangular_solve_f64_32x32 untimed 228.3 ms is 640% of the 35.7 ms timed region (budget 20%) |
| svd_full_square_f32_32x32 | jax-cuda-jit | svd_full_square_f32_32x32 untimed 185.0 ms is 448% of the 41.3 ms timed region (budget 20%) |
| svd_full_tall_f32_64x32 | jax-cuda-jit | svd_full_tall_f32_64x32 untimed 262.1 ms is 802% of the 32.7 ms timed region (budget 20%) |
| svd_full_wide_f32_32x64 | jax-cuda-jit | svd_full_wide_f32_32x64 untimed 246.6 ms is 707% of the 34.9 ms timed region (budget 20%) |
| slogdet_positive_f32_32x32 | jax-cuda-jit | slogdet_positive_f32_32x32 untimed 205.8 ms is 350% of the 58.9 ms timed region (budget 20%) |
| slogdet_negative_f32_32x32 | jax-cuda-jit | slogdet_negative_f32_32x32 untimed 509.5 ms is 1905% of the 26.7 ms timed region (budget 20%) |
| svd_full_square_f64_32x32 | jax-cuda-jit | svd_full_square_f64_32x32 untimed 279.0 ms is 921% of the 30.3 ms timed region (budget 20%) |
| svd_full_tall_f64_64x32 | jax-cuda-jit | svd_full_tall_f64_64x32 untimed 385.4 ms is 849% of the 45.4 ms timed region (budget 20%) |
| svd_full_wide_f64_32x64 | jax-cuda-jit | svd_full_wide_f64_32x64 untimed 282.1 ms is 599% of the 47.1 ms timed region (budget 20%) |
| slogdet_positive_f64_32x32 | jax-cuda-jit | slogdet_positive_f64_32x32 untimed 265.1 ms is 531% of the 49.9 ms timed region (budget 20%) |
| slogdet_negative_f64_32x32 | jax-cuda-jit | slogdet_negative_f64_32x32 untimed 231.0 ms is 870% of the 26.5 ms timed region (budget 20%) |
| svd_full_square_c32_32x32 | jax-cuda-jit | svd_full_square_c32_32x32 untimed 184.1 ms is 600% of the 30.7 ms timed region (budget 20%) |
| svd_full_tall_c32_64x32 | jax-cuda-jit | svd_full_tall_c32_64x32 untimed 204.0 ms is 467% of the 43.6 ms timed region (budget 20%) |
| svd_full_wide_c32_32x64 | jax-cuda-jit | svd_full_wide_c32_32x64 untimed 190.9 ms is 448% of the 42.6 ms timed region (budget 20%) |
| slogdet_positive_c32_32x32 | jax-cuda-jit | slogdet_positive_c32_32x32 untimed 249.1 ms is 517% of the 48.2 ms timed region (budget 20%) |
| slogdet_negative_c32_32x32 | jax-cuda-jit | slogdet_negative_c32_32x32 untimed 221.6 ms is 438% of the 50.6 ms timed region (budget 20%) |
| svd_full_square_c64_32x32 | jax-cuda-jit | svd_full_square_c64_32x32 untimed 147.3 ms is 265% of the 55.7 ms timed region (budget 20%) |
| svd_full_tall_c64_64x32 | jax-cuda-jit | svd_full_tall_c64_64x32 untimed 231.5 ms is 640% of the 36.2 ms timed region (budget 20%) |
| svd_full_wide_c64_32x64 | jax-cuda-jit | svd_full_wide_c64_32x64 untimed 173.8 ms is 467% of the 37.2 ms timed region (budget 20%) |
| slogdet_positive_c64_32x32 | jax-cuda-jit | slogdet_positive_c64_32x32 untimed 279.6 ms is 558% of the 50.1 ms timed region (budget 20%) |
| slogdet_negative_c64_32x32 | jax-cuda-jit | slogdet_negative_c64_32x32 untimed 251.5 ms is 847% of the 29.7 ms timed region (budget 20%) |
| householder_qr_square_f32_32x32 | jax-cuda-jit | householder_qr_square_f32_32x32 untimed 86.7 ms is 206% of the 42.0 ms timed region (budget 20%) |
| householder_qr_tall_f32_64x32 | jax-cuda-jit | householder_qr_tall_f32_64x32 untimed 89.2 ms is 284% of the 31.4 ms timed region (budget 20%) |
| householder_qr_wide_f32_32x64 | jax-cuda-jit | householder_qr_wide_f32_32x64 untimed 101.1 ms is 323% of the 31.3 ms timed region (budget 20%) |
| householder_qr_square_f64_32x32 | jax-cuda-jit | householder_qr_square_f64_32x32 untimed 94.2 ms is 193% of the 48.7 ms timed region (budget 20%) |
| householder_qr_tall_f64_64x32 | jax-cuda-jit | householder_qr_tall_f64_64x32 untimed 119.6 ms is 346% of the 34.5 ms timed region (budget 20%) |
| householder_qr_wide_f64_32x64 | jax-cuda-jit | householder_qr_wide_f64_32x64 untimed 94.1 ms is 270% of the 34.8 ms timed region (budget 20%) |
| householder_qr_square_c32_32x32 | jax-cuda-jit | householder_qr_square_c32_32x32 untimed 95.7 ms is 206% of the 46.3 ms timed region (budget 20%) |
| householder_qr_tall_c32_64x32 | jax-cuda-jit | householder_qr_tall_c32_64x32 untimed 100.9 ms is 280% of the 36.1 ms timed region (budget 20%) |
| householder_qr_wide_c32_32x64 | jax-cuda-jit | householder_qr_wide_c32_32x64 untimed 100.4 ms is 277% of the 36.2 ms timed region (budget 20%) |
| slogdet_phase_c32_32x32 | jax-cuda-jit | slogdet_phase_c32_32x32 untimed 235.4 ms is 840% of the 28.0 ms timed region (budget 20%) |
| householder_qr_square_c64_32x32 | jax-cuda-jit | householder_qr_square_c64_32x32 untimed 98.8 ms is 314% of the 31.5 ms timed region (budget 20%) |
| householder_qr_tall_c64_64x32 | jax-cuda-jit | householder_qr_tall_c64_64x32 untimed 97.4 ms is 226% of the 43.0 ms timed region (budget 20%) |
| householder_qr_wide_c64_32x64 | jax-cuda-jit | householder_qr_wide_c64_32x64 untimed 111.3 ms is 244% of the 45.6 ms timed region (budget 20%) |
| slogdet_phase_c64_32x32 | jax-cuda-jit | slogdet_phase_c64_32x32 untimed 259.5 ms is 976% of the 26.6 ms timed region (budget 20%) |
| householder_r_default_f32_64x32 | jax-cuda-jit | householder_r_default_f32_64x32 untimed 315.6 ms is 816% of the 38.7 ms timed region (budget 20%) |
| householder_q_columns_prefix_f32_64x32 | jax-cuda-jit | householder_q_columns_prefix_f32_64x32 untimed 258.5 ms is 507% of the 51.0 ms timed region (budget 20%) |
| householder_r_default_f64_64x32 | jax-cuda-jit | householder_r_default_f64_64x32 untimed 256.6 ms is 580% of the 44.3 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_64x32 | jax-cuda-jit | householder_q_columns_prefix_f64_64x32 untimed 281.0 ms is 487% of the 57.7 ms timed region (budget 20%) |
| householder_r_default_c32_64x32 | jax-cuda-jit | householder_r_default_c32_64x32 untimed 282.5 ms is 613% of the 46.1 ms timed region (budget 20%) |
| householder_q_columns_prefix_c32_64x32 | jax-cuda-jit | householder_q_columns_prefix_c32_64x32 untimed 199.8 ms is 382% of the 52.3 ms timed region (budget 20%) |
| householder_r_default_c64_64x32 | jax-cuda-jit | householder_r_default_c64_64x32 untimed 294.4 ms is 684% of the 43.0 ms timed region (budget 20%) |
| householder_q_columns_prefix_c64_64x32 | jax-cuda-jit | householder_q_columns_prefix_c64_64x32 untimed 209.5 ms is 403% of the 51.9 ms timed region (budget 20%) |