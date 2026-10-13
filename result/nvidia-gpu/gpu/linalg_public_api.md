# gpu/linalg_public_api: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api/20261013_101613/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | 130.040 (2.091) | 249.887 (7.255) | 90.156 (2.117) | 169.094 (3.257) | 146.020 (3.059) |
| cholesky_f64_32x32 | 238.886 (1.421) | 403.185 (6.012) | 161.516 (1.372) | 203.744 (6.686) | 205.496 (7.071) |
| svdvals_f32_32x32 | 715.295 (30.071) | unsupported | 406.341 (5.212) | 885.450 (11.759) | 885.699 (8.912) |
| svdvals_f64_32x32 | 5298.836 (19.012) | unsupported | 4820.312 (171.167) | 5857.478 (92.565) | 5768.628 (95.057) |
| eigvalsh_f32_32x32 | 241.573 (3.425) | 373.004 (3.480) | 224.273 (6.263) | 565.720 (21.982) | 596.632 (16.673) |
| eigvalsh_f64_32x32 | 648.880 (3.495) | 855.691 (17.164) | 613.120 (2.918) | 2930.452 (13.233) | 2941.297 (34.206) |
| pinv_f32_32x32 | unsupported | unsupported | 563.202 (4.410) | 1324.928 (9.897) | 730.941 (27.177) |
| pinv_f64_32x32 | unsupported | unsupported | 4997.068 (27.326) | 5908.659 (14.055) | 5375.175 (14.025) |
| inv_f32_32x32 | unsupported | unsupported | 293.569 (5.374) | 329.557 (13.677) | 332.802 (37.696) |
| inv_f64_32x32 | unsupported | unsupported | 292.296 (3.553) | 353.310 (17.971) | 198.803 (0.950) |
| det_f32_32x32 | 651.188 (110.254) | 1238.098 (28.011) | 167.981 (1.041) | 272.887 (8.319) | 260.131 (10.154) |
| det_f64_32x32 | 722.213 (8.964) | 1256.517 (20.758) | 169.362 (6.754) | 270.488 (6.588) | 156.995 (1.843) |
| norm_f32_32x32 | 75.580 (0.458) | 318.381 (7.205) | 15.447 (0.327) | 176.414 (21.460) | 134.295 (10.234) |
| norm_f64_32x32 | 79.621 (1.305) | 338.313 (5.742) | 15.440 (0.159) | 177.780 (5.788) | 158.144 (14.946) |
| triangular_solve_f32_32x32 | 105.744 (4.122) | 352.045 (5.500) | 52.880 (0.602) | 152.035 (3.196) | 183.609 (25.868) |
| triangular_solve_f64_32x32 | 86.407 (2.684) | 371.721 (12.223) | 52.856 (0.731) | 239.003 (32.598) | 152.917 (3.972) |
| cholesky_f32_512x512 | 391.623 (2.808) | 823.328 (21.928) | 281.743 (5.925) | 274.596 (7.101) | 269.842 (7.627) |
| cholesky_f64_512x512 | 2464.848 (11.277) | 3262.570 (65.521) | 2214.669 (5.137) | 2765.570 (29.936) | 2756.016 (17.089) |
| svdvals_f32_512x512 | 25827.753 (300.535) | unsupported | 28073.332 (235.671) | 25955.011 (26.504) | 25930.276 (24.575) |
| svdvals_f64_512x512 | 537796.838 (132.987) | unsupported | 592194.332 (42.269) | 538401.628 (70.318) | 538417.532 (49.927) |
| eigvalsh_f32_512x512 | 4328.039 (25.415) | 4772.507 (14.361) | 4355.966 (11.323) | 5033.768 (36.534) | 4997.400 (54.307) |
| eigvalsh_f64_512x512 | 14756.073 (53.790) | 14833.286 (66.987) | 14624.744 (65.884) | 19595.045 (95.832) | 19554.677 (172.996) |
| pinv_f32_512x512 | unsupported | unsupported | 31582.627 (83.548) | 29357.140 (28.941) | 28691.940 (15.153) |
| pinv_f64_512x512 | unsupported | unsupported | 677383.900 (34.948) | 616627.675 (51.543) | 616152.925 (38.989) |
| inv_f32_512x512 | unsupported | unsupported | 1942.568 (7.961) | 2318.617 (16.454) | 2216.962 (32.482) |
| inv_f64_512x512 | unsupported | unsupported | 5964.873 (5.483) | 6413.906 (28.791) | 6381.314 (33.022) |
| det_f32_512x512 | 2296.159 (38.500) | 3960.321 (118.417) | 1434.401 (2.383) | 1631.807 (11.520) | 1615.737 (16.091) |
| det_f64_512x512 | 3481.581 (31.386) | 5282.868 (341.729) | 2497.102 (11.403) | 2907.185 (21.198) | 2903.040 (20.102) |
| norm_f32_512x512 | 83.909 (6.704) | 705.021 (45.486) | 19.940 (0.287) | 167.950 (3.566) | 163.613 (3.559) |
| norm_f64_512x512 | 69.107 (4.794) | 583.040 (31.949) | 20.527 (0.286) | 158.032 (23.992) | 99.309 (0.958) |
| triangular_solve_f32_512x512 | 82.099 (4.490) | 346.761 (10.424) | 102.869 (0.081) | 149.424 (4.927) | 162.322 (9.178) |
| triangular_solve_f64_512x512 | 142.860 (0.467) | 377.832 (25.444) | 839.245 (0.693) | 191.771 (13.258) | 252.209 (5.499) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 2.081 | 0.0002352 |
| cholesky_f64_32x32 | tenferro-cuda-direct | 16 | 15 | 3.822 | 2.188e-07 |
| svdvals_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.861 | 0.003338 |
| svdvals_f64_32x32 | tenferro-cuda-direct | 1 | 15 | 5.299 | 4.546e-06 |
| eigvalsh_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 3.865 | 0.001073 |
| eigvalsh_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.596 | 5.433e-07 |
| pinv_f32_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| det_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 5.210 | 0.002641 |
| det_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.889 | 1.135e-06 |
| norm_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.419 | 0 |
| norm_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.548 | 0 |
| triangular_solve_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 3.384 | 0.0001546 |
| triangular_solve_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.765 | 1.609e-07 |
| cholesky_f32_512x512 | tenferro-cuda-direct | 8 | 15 | 3.133 | 0.0003554 |
| cholesky_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.465 | 4.418e-07 |
| svdvals_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 25.828 | 0.1547 |
| svdvals_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 537.797 | 0.0002063 |
| eigvalsh_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 4.328 | 0.008342 |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 14.756 | 1.55e-06 |
| pinv_f32_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| det_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 2.296 | 0.008577 |
| det_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 3.482 | 1.426e-05 |
| norm_f32_512x512 | tenferro-cuda-direct | 32 | 15 | 2.685 | 0 |
| norm_f64_512x512 | tenferro-cuda-direct | 32 | 15 | 2.211 | 1.909e-07 |
| triangular_solve_f32_512x512 | tenferro-cuda-direct | 32 | 15 | 2.627 | 0.0007877 |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | 16 | 15 | 2.286 | 1.157e-06 |
| cholesky_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 3.998 | 0.0002352 |
| cholesky_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.225 | 2.188e-07 |
| svdvals_f32_32x32 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_32x32 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| eigvalsh_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.984 | 0.001073 |
| eigvalsh_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 3.423 | 5.433e-07 |
| pinv_f32_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f32_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| det_f32_32x32 | tenferro-cuda-eager | 2 | 15 | 2.476 | 0.002641 |
| det_f64_32x32 | tenferro-cuda-eager | 2 | 15 | 2.513 | 1.135e-06 |
| norm_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.547 | 0 |
| norm_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.707 | 0 |
| triangular_solve_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.816 | 0.0001546 |
| triangular_solve_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 2.974 | 1.609e-07 |
| cholesky_f32_512x512 | tenferro-cuda-eager | 4 | 15 | 3.293 | 0.0003554 |
| cholesky_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 3.263 | 4.418e-07 |
| svdvals_f32_512x512 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_512x512 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| eigvalsh_f32_512x512 | tenferro-cuda-eager | 1 | 15 | 4.773 | 0.008342 |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 14.833 | 1.55e-06 |
| pinv_f32_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f32_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| det_f32_512x512 | tenferro-cuda-eager | 1 | 15 | 3.960 | 0.008577 |
| det_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 5.283 | 1.426e-05 |
| norm_f32_512x512 | tenferro-cuda-eager | 4 | 15 | 2.820 | 0 |
| norm_f64_512x512 | tenferro-cuda-eager | 4 | 15 | 2.332 | 1.909e-07 |
| triangular_solve_f32_512x512 | tenferro-cuda-eager | 8 | 15 | 2.774 | 0.0007877 |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | 8 | 15 | 3.023 | 1.157e-06 |
| cholesky_f32_32x32 | pytorch-cuda | 32 | 15 | 2.885 | 0.0002352 |
| cholesky_f64_32x32 | pytorch-cuda | 16 | 15 | 2.584 | 2.188e-07 |
| svdvals_f32_32x32 | pytorch-cuda | 8 | 15 | 3.251 | 0.007749 |
| svdvals_f64_32x32 | pytorch-cuda | 1 | 15 | 4.820 | 3.387e-06 |
| eigvalsh_f32_32x32 | pytorch-cuda | 16 | 15 | 3.588 | 0.001073 |
| eigvalsh_f64_32x32 | pytorch-cuda | 4 | 15 | 2.452 | 5.433e-07 |
| pinv_f32_32x32 | pytorch-cuda | 4 | 15 | 2.253 | 0.001023 |
| pinv_f64_32x32 | pytorch-cuda | 1 | 15 | 4.997 | 9.374e-07 |
| inv_f32_32x32 | pytorch-cuda | 8 | 15 | 2.349 | 0.0004905 |
| inv_f64_32x32 | pytorch-cuda | 8 | 15 | 2.338 | 2.284e-07 |
| det_f32_32x32 | pytorch-cuda | 16 | 15 | 2.688 | 0.002641 |
| det_f64_32x32 | pytorch-cuda | 16 | 15 | 2.710 | 3.784e-07 |
| norm_f32_32x32 | pytorch-cuda | 128 | 15 | 1.977 | 0 |
| norm_f64_32x32 | pytorch-cuda | 128 | 15 | 1.976 | 0 |
| triangular_solve_f32_32x32 | pytorch-cuda | 64 | 15 | 3.384 | 9.242e-05 |
| triangular_solve_f64_32x32 | pytorch-cuda | 64 | 15 | 3.383 | 1.415e-07 |
| cholesky_f32_512x512 | pytorch-cuda | 8 | 15 | 2.254 | 0.0003554 |
| cholesky_f64_512x512 | pytorch-cuda | 1 | 15 | 2.215 | 4.418e-07 |
| svdvals_f32_512x512 | pytorch-cuda | 1 | 15 | 28.073 | 0.1733 |
| svdvals_f64_512x512 | pytorch-cuda | 1 | 15 | 592.194 | 0.0002066 |
| eigvalsh_f32_512x512 | pytorch-cuda | 1 | 15 | 4.356 | 0.008342 |
| eigvalsh_f64_512x512 | pytorch-cuda | 1 | 15 | 14.625 | 1.55e-06 |
| pinv_f32_512x512 | pytorch-cuda | 1 | 15 | 31.583 | 0.01126 |
| pinv_f64_512x512 | pytorch-cuda | 1 | 15 | 677.384 | 8.747e-06 |
| inv_f32_512x512 | pytorch-cuda | 2 | 15 | 3.885 | 0.002107 |
| inv_f64_512x512 | pytorch-cuda | 1 | 15 | 5.965 | 1.85e-06 |
| det_f32_512x512 | pytorch-cuda | 2 | 15 | 2.869 | 0.007965 |
| det_f64_512x512 | pytorch-cuda | 1 | 15 | 2.497 | 1.369e-05 |
| norm_f32_512x512 | pytorch-cuda | 128 | 15 | 2.552 | 0 |
| norm_f64_512x512 | pytorch-cuda | 128 | 15 | 2.627 | 1.909e-07 |
| triangular_solve_f32_512x512 | pytorch-cuda | 32 | 15 | 3.292 | 0.0006008 |
| triangular_solve_f64_512x512 | pytorch-cuda | 4 | 15 | 3.357 | 6.887e-07 |
| cholesky_f32_32x32 | jax-cuda | 16 | 15 | 2.706 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda | 16 | 15 | 3.260 | 2.188e-07 |
| svdvals_f32_32x32 | jax-cuda | 4 | 15 | 3.542 | 0.003338 |
| svdvals_f64_32x32 | jax-cuda | 1 | 15 | 5.857 | 4.546e-06 |
| eigvalsh_f32_32x32 | jax-cuda | 4 | 15 | 2.263 | 0.004775 |
| eigvalsh_f64_32x32 | jax-cuda | 1 | 15 | 2.930 | 5.542e-06 |
| pinv_f32_32x32 | jax-cuda | 2 | 15 | 2.650 | 0.0006436 |
| pinv_f64_32x32 | jax-cuda | 1 | 15 | 5.909 | 1.367e-06 |
| inv_f32_32x32 | jax-cuda | 8 | 15 | 2.636 | 0.0004905 |
| inv_f64_32x32 | jax-cuda | 8 | 15 | 2.826 | 2.284e-07 |
| det_f32_32x32 | jax-cuda | 8 | 15 | 2.183 | 0.002641 |
| det_f64_32x32 | jax-cuda | 8 | 15 | 2.164 | 1.135e-06 |
| norm_f32_32x32 | jax-cuda | 16 | 15 | 2.823 | 0 |
| norm_f64_32x32 | jax-cuda | 16 | 15 | 2.844 | 0 |
| triangular_solve_f32_32x32 | jax-cuda | 16 | 15 | 2.433 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda | 16 | 15 | 3.824 | 1.609e-07 |
| cholesky_f32_512x512 | jax-cuda | 8 | 15 | 2.197 | 0.0003554 |
| cholesky_f64_512x512 | jax-cuda | 1 | 15 | 2.766 | 4.418e-07 |
| svdvals_f32_512x512 | jax-cuda | 1 | 15 | 25.955 | 0.1547 |
| svdvals_f64_512x512 | jax-cuda | 1 | 15 | 538.402 | 0.0002063 |
| eigvalsh_f32_512x512 | jax-cuda | 1 | 15 | 5.034 | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda | 1 | 15 | 19.595 | 1.55e-06 |
| pinv_f32_512x512 | jax-cuda | 1 | 15 | 29.357 | 0.009647 |
| pinv_f64_512x512 | jax-cuda | 1 | 15 | 616.628 | 8.975e-06 |
| inv_f32_512x512 | jax-cuda | 1 | 15 | 2.319 | 0.002107 |
| inv_f64_512x512 | jax-cuda | 1 | 15 | 6.414 | 1.85e-06 |
| det_f32_512x512 | jax-cuda | 2 | 15 | 3.264 | 0.004748 |
| det_f64_512x512 | jax-cuda | 1 | 15 | 2.907 | 1.426e-05 |
| norm_f32_512x512 | jax-cuda | 16 | 15 | 2.687 | 0 |
| norm_f64_512x512 | jax-cuda | 16 | 15 | 2.529 | 1.909e-07 |
| triangular_solve_f32_512x512 | jax-cuda | 16 | 15 | 2.391 | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda | 16 | 15 | 3.068 | 1.157e-06 |
| cholesky_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.336 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda-jit | 16 | 15 | 3.288 | 2.188e-07 |
| svdvals_f32_32x32 | jax-cuda-jit | 4 | 15 | 3.543 | 0.003338 |
| svdvals_f64_32x32 | jax-cuda-jit | 1 | 15 | 5.769 | 4.546e-06 |
| eigvalsh_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.387 | 0.004775 |
| eigvalsh_f64_32x32 | jax-cuda-jit | 1 | 15 | 2.941 | 5.542e-06 |
| pinv_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.924 | 0.0006436 |
| pinv_f64_32x32 | jax-cuda-jit | 1 | 15 | 5.375 | 1.367e-06 |
| inv_f32_32x32 | jax-cuda-jit | 8 | 15 | 2.662 | 0.0004905 |
| inv_f64_32x32 | jax-cuda-jit | 16 | 15 | 3.181 | 2.284e-07 |
| det_f32_32x32 | jax-cuda-jit | 8 | 15 | 2.081 | 0.002641 |
| det_f64_32x32 | jax-cuda-jit | 16 | 15 | 2.512 | 1.135e-06 |
| norm_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.149 | 0 |
| norm_f64_32x32 | jax-cuda-jit | 16 | 15 | 2.530 | 0 |
| triangular_solve_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.938 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda-jit | 16 | 15 | 2.447 | 1.609e-07 |
| cholesky_f32_512x512 | jax-cuda-jit | 8 | 15 | 2.159 | 0.0003554 |
| cholesky_f64_512x512 | jax-cuda-jit | 1 | 15 | 2.756 | 4.418e-07 |
| svdvals_f32_512x512 | jax-cuda-jit | 1 | 15 | 25.930 | 0.1547 |
| svdvals_f64_512x512 | jax-cuda-jit | 1 | 15 | 538.418 | 0.0002063 |
| eigvalsh_f32_512x512 | jax-cuda-jit | 1 | 15 | 4.997 | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda-jit | 1 | 15 | 19.555 | 1.55e-06 |
| pinv_f32_512x512 | jax-cuda-jit | 1 | 15 | 28.692 | 0.009647 |
| pinv_f64_512x512 | jax-cuda-jit | 1 | 15 | 616.153 | 8.975e-06 |
| inv_f32_512x512 | jax-cuda-jit | 1 | 15 | 2.217 | 0.002107 |
| inv_f64_512x512 | jax-cuda-jit | 1 | 15 | 6.381 | 1.85e-06 |
| det_f32_512x512 | jax-cuda-jit | 2 | 15 | 3.231 | 0.004748 |
| det_f64_512x512 | jax-cuda-jit | 1 | 15 | 2.903 | 1.426e-05 |
| norm_f32_512x512 | jax-cuda-jit | 16 | 15 | 2.618 | 0 |
| norm_f64_512x512 | jax-cuda-jit | 32 | 15 | 3.178 | 1.909e-07 |
| triangular_solve_f32_512x512 | jax-cuda-jit | 16 | 15 | 2.597 | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda-jit | 8 | 15 | 2.018 | 1.157e-06 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api
suite_file: benchmarks/gpu/linalg_public_api.yaml
timestamp: '2026-10-13T10:16:13Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 8b13220c1e0397a29a27be65557c24105352a35b
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: ed0ed4c79805014f2bba4644383450a6b9bf9361
  dirty: false
environment:
  hostname: 33ed5e46905b
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
  expected_cases: 32
  selected_cases: 32
  sha256:
    .devcontainer/cuda/devcontainer.json: 6d3b10caecad15c0a513120ce6282ebb13c65be81c40e29f78a329439c50f083
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: f29f1ad0dd99e8e5d24fffc68b6fe30673b0c42d34b5fe4a5557b7f774c2af15
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 86f2856411d902d27dcc5c8e66415eb933e057e04b9fad86be55569004572a36
    src/bin/benchmark_gpu_public_api/structural.rs: b5d780e867a5748be4f29f245d367833a8cd5ee0ebd440926d8b3003b432b0e4
    src/bin/benchmark_gpu_public_api/linalg.rs: 7d4e29355a973f3c60c2af53fd6ceac597d85fe21f9ab5f6bf164d0afe0a6566
    src/bin/benchmark_gpu_public_api/reuse.rs: ad72b565373904275128d7beb8673bada08d922e8f310be6cedb0d2d8187898f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 24cd5cec53eaecc8b6b4a22a1cd69e8cbe430ab6c9945d48df4c03d3b2655999
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 0a9e8d89405fc802e392e047d8b7c587e38ab1850d831735dd53fe13e87dabd3
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/linalg_public_api/20261013_101613/fixtures/manifest.json: 370eb17f94269bef6bc63e658c5697f52b2acf97be1c43a97e1e93b5ea728991
    target/release/benchmark_gpu_public_api: 91380afb8faffb97ca96d131ef98cc8e2656aababe594e82d65a07c9bb1bed38
  reference_numpy_version: 2.4.2
```
