# gpu/linalg_public_api: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api/20261013_140259/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | 186.547 (18.894) | 241.600 (70.172) | 92.509 (5.279) | 120.365 (3.674) | 126.999 (4.898) |
| cholesky_f64_32x32 | 303.051 (8.605) | 391.302 (5.920) | 160.714 (1.814) | 157.469 (2.739) | 186.383 (6.741) |
| svdvals_f32_32x32 | 725.537 (28.893) | unsupported | 402.653 (3.339) | 881.068 (21.458) | 792.253 (28.625) |
| svdvals_f64_32x32 | 5529.583 (9.140) | unsupported | 5087.390 (3.995) | 5861.534 (74.657) | 5695.905 (23.352) |
| eigvalsh_f32_32x32 | 272.062 (6.740) | 321.488 (5.311) | 218.222 (9.062) | 678.622 (6.943) | 578.570 (20.915) |
| eigvalsh_f64_32x32 | 655.767 (8.750) | 693.189 (4.650) | 610.523 (8.213) | 2923.483 (25.909) | 2945.922 (21.972) |
| pinv_f32_32x32 | unsupported | unsupported | 551.987 (16.052) | 1201.884 (15.014) | 752.948 (28.676) |
| pinv_f64_32x32 | unsupported | unsupported | 4913.612 (12.096) | 5862.271 (23.788) | 5421.338 (19.537) |
| inv_f32_32x32 | unsupported | unsupported | 281.667 (2.557) | 207.357 (16.979) | 210.352 (1.289) |
| inv_f64_32x32 | unsupported | unsupported | 280.308 (3.358) | 345.790 (18.312) | 208.123 (8.889) |
| det_f32_32x32 | 645.041 (9.834) | 832.793 (56.635) | 166.823 (4.846) | 224.261 (26.041) | 182.467 (9.373) |
| det_f64_32x32 | 496.647 (4.875) | 826.206 (18.462) | 160.927 (3.287) | 182.958 (5.258) | 154.161 (2.660) |
| norm_f32_32x32 | 65.258 (4.247) | 194.028 (5.673) | 15.148 (1.360) | 130.442 (3.477) | 138.506 (11.876) |
| norm_f64_32x32 | 60.235 (1.881) | 216.385 (4.274) | 15.391 (0.787) | 166.396 (7.371) | 159.234 (14.593) |
| triangular_solve_f32_32x32 | 75.898 (2.549) | 281.788 (50.996) | 53.248 (2.667) | 225.951 (50.798) | 188.975 (33.717) |
| triangular_solve_f64_32x32 | 68.127 (2.806) | 271.465 (15.141) | 57.194 (4.486) | 226.679 (40.810) | 184.485 (19.416) |
| cholesky_f32_512x512 | 346.813 (4.992) | 500.045 (35.053) | 278.923 (2.841) | 267.847 (2.738) | 267.845 (8.676) |
| cholesky_f64_512x512 | 2258.461 (4.608) | 2513.682 (12.445) | 2194.067 (13.448) | 2750.091 (25.986) | 2763.194 (20.290) |
| svdvals_f32_512x512 | 25716.713 (29.275) | unsupported | 28276.004 (10.507) | 26020.736 (499.938) | 25983.728 (33.459) |
| svdvals_f64_512x512 | 539504.415 (56.632) | unsupported | 597865.760 (68.452) | 543407.286 (35.600) | 539932.945 (66.899) |
| eigvalsh_f32_512x512 | 4270.918 (19.878) | 4604.263 (262.386) | 4388.359 (5.848) | 5074.599 (78.923) | 4996.120 (49.936) |
| eigvalsh_f64_512x512 | 14637.413 (61.296) | 14764.809 (75.300) | 14752.492 (43.385) | 19701.325 (122.978) | 19552.279 (277.571) |
| pinv_f32_512x512 | unsupported | unsupported | 31720.141 (60.248) | 29316.131 (42.217) | 28726.192 (19.217) |
| pinv_f64_512x512 | unsupported | unsupported | 683694.991 (72.039) | 622432.157 (123.508) | 617797.992 (63.716) |
| inv_f32_512x512 | unsupported | unsupported | 1953.397 (5.083) | 2297.197 (27.490) | 2311.348 (49.080) |
| inv_f64_512x512 | unsupported | unsupported | 6015.930 (11.542) | 6437.421 (27.432) | 6381.455 (20.660) |
| det_f32_512x512 | 1781.625 (5.618) | 1975.797 (34.755) | 1442.305 (1.732) | 1632.793 (16.643) | 1596.318 (29.846) |
| det_f64_512x512 | 2865.967 (11.324) | 3017.181 (17.663) | 2512.110 (10.721) | 2900.718 (12.605) | 2810.352 (45.737) |
| norm_f32_512x512 | 37.968 (0.941) | 97.224 (1.394) | 19.756 (0.987) | 163.698 (2.727) | 167.684 (6.064) |
| norm_f64_512x512 | 36.962 (0.654) | 101.405 (1.420) | 20.335 (0.681) | 164.728 (2.576) | 164.272 (6.172) |
| triangular_solve_f32_512x512 | 71.295 (0.139) | 181.570 (2.083) | 103.812 (0.044) | 148.458 (1.430) | 179.491 (8.692) |
| triangular_solve_f64_512x512 | 141.214 (0.181) | 213.157 (5.790) | 847.054 (0.295) | 217.894 (38.170) | 191.346 (22.371) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| cholesky_f32_32x32 | tenferro-cuda-direct | 16 | 15 | 2.985 | 0.0002352 |
| cholesky_f64_32x32 | tenferro-cuda-direct | 8 | 15 | 2.424 | 2.188e-07 |
| svdvals_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.902 | 0.003338 |
| svdvals_f64_32x32 | tenferro-cuda-direct | 1 | 15 | 5.530 | 4.546e-06 |
| eigvalsh_f32_32x32 | tenferro-cuda-direct | 8 | 15 | 2.176 | 0.001073 |
| eigvalsh_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 2.623 | 5.433e-07 |
| pinv_f32_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_32x32 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| det_f32_32x32 | tenferro-cuda-direct | 4 | 15 | 2.580 | 0.002641 |
| det_f64_32x32 | tenferro-cuda-direct | 4 | 15 | 1.987 | 1.135e-06 |
| norm_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.088 | 0 |
| norm_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 1.928 | 0 |
| triangular_solve_f32_32x32 | tenferro-cuda-direct | 32 | 15 | 2.429 | 0.0001546 |
| triangular_solve_f64_32x32 | tenferro-cuda-direct | 32 | 15 | 2.180 | 1.609e-07 |
| cholesky_f32_512x512 | tenferro-cuda-direct | 8 | 15 | 2.775 | 0.0003554 |
| cholesky_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.258 | 4.418e-07 |
| svdvals_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 25.717 | 0.1547 |
| svdvals_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 539.504 | 0.0002063 |
| eigvalsh_f32_512x512 | tenferro-cuda-direct | 1 | 15 | 4.271 | 0.008342 |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 14.637 | 1.55e-06 |
| pinv_f32_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f32_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_512x512 | tenferro-cuda-direct | — | — | — | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| det_f32_512x512 | tenferro-cuda-direct | 2 | 15 | 3.563 | 0.008577 |
| det_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.866 | 1.426e-05 |
| norm_f32_512x512 | tenferro-cuda-direct | 64 | 15 | 2.430 | 0 |
| norm_f64_512x512 | tenferro-cuda-direct | 64 | 15 | 2.366 | 1.909e-07 |
| triangular_solve_f32_512x512 | tenferro-cuda-direct | 32 | 15 | 2.281 | 0.0007877 |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | 16 | 15 | 2.259 | 1.157e-06 |
| cholesky_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 3.866 | 0.0002352 |
| cholesky_f64_32x32 | tenferro-cuda-eager | 8 | 15 | 3.130 | 2.188e-07 |
| svdvals_f32_32x32 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_32x32 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| eigvalsh_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.572 | 0.001073 |
| eigvalsh_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 2.773 | 5.433e-07 |
| pinv_f32_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f32_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_32x32 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| det_f32_32x32 | tenferro-cuda-eager | 4 | 15 | 3.331 | 0.002641 |
| det_f64_32x32 | tenferro-cuda-eager | 4 | 15 | 3.305 | 1.135e-06 |
| norm_f32_32x32 | tenferro-cuda-eager | 16 | 15 | 3.104 | 0 |
| norm_f64_32x32 | tenferro-cuda-eager | 16 | 15 | 3.462 | 0 |
| triangular_solve_f32_32x32 | tenferro-cuda-eager | 8 | 15 | 2.254 | 0.0001546 |
| triangular_solve_f64_32x32 | tenferro-cuda-eager | 16 | 15 | 4.343 | 1.609e-07 |
| cholesky_f32_512x512 | tenferro-cuda-eager | 4 | 15 | 2.000 | 0.0003554 |
| cholesky_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 2.514 | 4.418e-07 |
| svdvals_f32_512x512 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| svdvals_f64_512x512 | tenferro-cuda-eager | — | — | — | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| eigvalsh_f32_512x512 | tenferro-cuda-eager | 1 | 15 | 4.604 | 0.008342 |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 14.765 | 1.55e-06 |
| pinv_f32_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| pinv_f64_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f32_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| inv_f64_512x512 | tenferro-cuda-eager | — | — | — | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| det_f32_512x512 | tenferro-cuda-eager | 2 | 15 | 3.952 | 0.008577 |
| det_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 3.017 | 1.426e-05 |
| norm_f32_512x512 | tenferro-cuda-eager | 32 | 15 | 3.111 | 0 |
| norm_f64_512x512 | tenferro-cuda-eager | 32 | 15 | 3.245 | 1.909e-07 |
| triangular_solve_f32_512x512 | tenferro-cuda-eager | 16 | 15 | 2.905 | 0.0007877 |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | 16 | 15 | 3.411 | 1.157e-06 |
| cholesky_f32_32x32 | pytorch-cuda | 32 | 15 | 2.960 | 0.0002352 |
| cholesky_f64_32x32 | pytorch-cuda | 16 | 15 | 2.571 | 2.188e-07 |
| svdvals_f32_32x32 | pytorch-cuda | 8 | 15 | 3.221 | 0.007749 |
| svdvals_f64_32x32 | pytorch-cuda | 1 | 15 | 5.087 | 3.387e-06 |
| eigvalsh_f32_32x32 | pytorch-cuda | 16 | 15 | 3.492 | 0.001073 |
| eigvalsh_f64_32x32 | pytorch-cuda | 4 | 15 | 2.442 | 5.433e-07 |
| pinv_f32_32x32 | pytorch-cuda | 4 | 15 | 2.208 | 0.001023 |
| pinv_f64_32x32 | pytorch-cuda | 1 | 15 | 4.914 | 9.374e-07 |
| inv_f32_32x32 | pytorch-cuda | 8 | 15 | 2.253 | 0.0004905 |
| inv_f64_32x32 | pytorch-cuda | 8 | 15 | 2.242 | 2.284e-07 |
| det_f32_32x32 | pytorch-cuda | 16 | 15 | 2.669 | 0.002641 |
| det_f64_32x32 | pytorch-cuda | 16 | 15 | 2.575 | 3.784e-07 |
| norm_f32_32x32 | pytorch-cuda | 128 | 15 | 1.939 | 0 |
| norm_f64_32x32 | pytorch-cuda | 128 | 15 | 1.970 | 0 |
| triangular_solve_f32_32x32 | pytorch-cuda | 64 | 15 | 3.408 | 9.242e-05 |
| triangular_solve_f64_32x32 | pytorch-cuda | 64 | 15 | 3.660 | 1.415e-07 |
| cholesky_f32_512x512 | pytorch-cuda | 8 | 15 | 2.231 | 0.0003554 |
| cholesky_f64_512x512 | pytorch-cuda | 1 | 15 | 2.194 | 4.418e-07 |
| svdvals_f32_512x512 | pytorch-cuda | 1 | 15 | 28.276 | 0.1733 |
| svdvals_f64_512x512 | pytorch-cuda | 1 | 15 | 597.866 | 0.0002066 |
| eigvalsh_f32_512x512 | pytorch-cuda | 1 | 15 | 4.388 | 0.008342 |
| eigvalsh_f64_512x512 | pytorch-cuda | 1 | 15 | 14.752 | 1.55e-06 |
| pinv_f32_512x512 | pytorch-cuda | 1 | 15 | 31.720 | 0.01126 |
| pinv_f64_512x512 | pytorch-cuda | 1 | 15 | 683.695 | 8.747e-06 |
| inv_f32_512x512 | pytorch-cuda | 2 | 15 | 3.907 | 0.002107 |
| inv_f64_512x512 | pytorch-cuda | 1 | 15 | 6.016 | 1.85e-06 |
| det_f32_512x512 | pytorch-cuda | 2 | 15 | 2.885 | 0.007965 |
| det_f64_512x512 | pytorch-cuda | 1 | 15 | 2.512 | 1.369e-05 |
| norm_f32_512x512 | pytorch-cuda | 128 | 15 | 2.529 | 0 |
| norm_f64_512x512 | pytorch-cuda | 128 | 15 | 2.603 | 1.909e-07 |
| triangular_solve_f32_512x512 | pytorch-cuda | 32 | 15 | 3.322 | 0.0006008 |
| triangular_solve_f64_512x512 | pytorch-cuda | 4 | 15 | 3.388 | 6.887e-07 |
| cholesky_f32_32x32 | jax-cuda | 16 | 15 | 1.926 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda | 16 | 15 | 2.519 | 2.188e-07 |
| svdvals_f32_32x32 | jax-cuda | 4 | 15 | 3.524 | 0.003338 |
| svdvals_f64_32x32 | jax-cuda | 1 | 15 | 5.862 | 4.546e-06 |
| eigvalsh_f32_32x32 | jax-cuda | 4 | 15 | 2.714 | 0.004775 |
| eigvalsh_f64_32x32 | jax-cuda | 1 | 15 | 2.923 | 5.542e-06 |
| pinv_f32_32x32 | jax-cuda | 2 | 15 | 2.404 | 0.0006436 |
| pinv_f64_32x32 | jax-cuda | 1 | 15 | 5.862 | 1.367e-06 |
| inv_f32_32x32 | jax-cuda | 16 | 15 | 3.318 | 0.0004905 |
| inv_f64_32x32 | jax-cuda | 8 | 15 | 2.766 | 2.284e-07 |
| det_f32_32x32 | jax-cuda | 16 | 15 | 3.588 | 0.002641 |
| det_f64_32x32 | jax-cuda | 16 | 15 | 2.927 | 1.135e-06 |
| norm_f32_32x32 | jax-cuda | 16 | 15 | 2.087 | 0 |
| norm_f64_32x32 | jax-cuda | 16 | 15 | 2.662 | 0 |
| triangular_solve_f32_32x32 | jax-cuda | 16 | 15 | 3.615 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda | 16 | 15 | 3.627 | 1.609e-07 |
| cholesky_f32_512x512 | jax-cuda | 8 | 15 | 2.143 | 0.0003554 |
| cholesky_f64_512x512 | jax-cuda | 1 | 15 | 2.750 | 4.418e-07 |
| svdvals_f32_512x512 | jax-cuda | 1 | 15 | 26.021 | 0.1547 |
| svdvals_f64_512x512 | jax-cuda | 1 | 15 | 543.407 | 0.0002063 |
| eigvalsh_f32_512x512 | jax-cuda | 1 | 15 | 5.075 | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda | 1 | 15 | 19.701 | 1.55e-06 |
| pinv_f32_512x512 | jax-cuda | 1 | 15 | 29.316 | 0.009647 |
| pinv_f64_512x512 | jax-cuda | 1 | 15 | 622.432 | 8.975e-06 |
| inv_f32_512x512 | jax-cuda | 1 | 15 | 2.297 | 0.002107 |
| inv_f64_512x512 | jax-cuda | 1 | 15 | 6.437 | 1.85e-06 |
| det_f32_512x512 | jax-cuda | 2 | 15 | 3.266 | 0.004748 |
| det_f64_512x512 | jax-cuda | 1 | 15 | 2.901 | 1.426e-05 |
| norm_f32_512x512 | jax-cuda | 16 | 15 | 2.619 | 0 |
| norm_f64_512x512 | jax-cuda | 16 | 15 | 2.636 | 1.909e-07 |
| triangular_solve_f32_512x512 | jax-cuda | 16 | 15 | 2.375 | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda | 16 | 15 | 3.486 | 1.157e-06 |
| cholesky_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.032 | 0.0002352 |
| cholesky_f64_32x32 | jax-cuda-jit | 16 | 15 | 2.982 | 2.188e-07 |
| svdvals_f32_32x32 | jax-cuda-jit | 4 | 15 | 3.169 | 0.003338 |
| svdvals_f64_32x32 | jax-cuda-jit | 1 | 15 | 5.696 | 4.546e-06 |
| eigvalsh_f32_32x32 | jax-cuda-jit | 4 | 15 | 2.314 | 0.004775 |
| eigvalsh_f64_32x32 | jax-cuda-jit | 1 | 15 | 2.946 | 5.542e-06 |
| pinv_f32_32x32 | jax-cuda-jit | 4 | 15 | 3.012 | 0.0006436 |
| pinv_f64_32x32 | jax-cuda-jit | 1 | 15 | 5.421 | 1.367e-06 |
| inv_f32_32x32 | jax-cuda-jit | 16 | 15 | 3.366 | 0.0004905 |
| inv_f64_32x32 | jax-cuda-jit | 16 | 15 | 3.330 | 2.284e-07 |
| det_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.919 | 0.002641 |
| det_f64_32x32 | jax-cuda-jit | 16 | 15 | 2.467 | 1.135e-06 |
| norm_f32_32x32 | jax-cuda-jit | 16 | 15 | 2.216 | 0 |
| norm_f64_32x32 | jax-cuda-jit | 16 | 15 | 2.548 | 0 |
| triangular_solve_f32_32x32 | jax-cuda-jit | 16 | 15 | 3.024 | 0.0001546 |
| triangular_solve_f64_32x32 | jax-cuda-jit | 16 | 15 | 2.952 | 1.609e-07 |
| cholesky_f32_512x512 | jax-cuda-jit | 8 | 15 | 2.143 | 0.0003554 |
| cholesky_f64_512x512 | jax-cuda-jit | 1 | 15 | 2.763 | 4.418e-07 |
| svdvals_f32_512x512 | jax-cuda-jit | 1 | 15 | 25.984 | 0.1547 |
| svdvals_f64_512x512 | jax-cuda-jit | 1 | 15 | 539.933 | 0.0002063 |
| eigvalsh_f32_512x512 | jax-cuda-jit | 1 | 15 | 4.996 | 0.008342 |
| eigvalsh_f64_512x512 | jax-cuda-jit | 1 | 15 | 19.552 | 1.55e-06 |
| pinv_f32_512x512 | jax-cuda-jit | 1 | 15 | 28.726 | 0.009647 |
| pinv_f64_512x512 | jax-cuda-jit | 1 | 15 | 617.798 | 8.975e-06 |
| inv_f32_512x512 | jax-cuda-jit | 1 | 15 | 2.311 | 0.002107 |
| inv_f64_512x512 | jax-cuda-jit | 1 | 15 | 6.381 | 1.85e-06 |
| det_f32_512x512 | jax-cuda-jit | 2 | 15 | 3.193 | 0.004748 |
| det_f64_512x512 | jax-cuda-jit | 1 | 15 | 2.810 | 1.426e-05 |
| norm_f32_512x512 | jax-cuda-jit | 16 | 15 | 2.683 | 0 |
| norm_f64_512x512 | jax-cuda-jit | 16 | 15 | 2.628 | 1.909e-07 |
| triangular_solve_f32_512x512 | jax-cuda-jit | 16 | 15 | 2.872 | 0.0007877 |
| triangular_solve_f64_512x512 | jax-cuda-jit | 16 | 15 | 3.062 | 1.157e-06 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api
suite_file: benchmarks/gpu/linalg_public_api.yaml
timestamp: '2026-10-13T14:03:00Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 71cf328826a92cc9628a8c15e8f45430a8a6d98c
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
  expected_cases: 32
  selected_cases: 32
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 249184323d09433d1fbf401284e39fa797c0252dceb20351bbcc151551d1b32c
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/structural.rs: 382fb5ffb6d86ac1a4a66723376808e9c95c8a8731570a5617f2ffed2a6e40a7
    src/bin/benchmark_gpu_public_api/linalg.rs: 84162923fa0d0772c88ec9c4eab169ad3a75e5d16ff76961d1541670dd8565c1
    src/bin/benchmark_gpu_public_api/reuse.rs: ad72b565373904275128d7beb8673bada08d922e8f310be6cedb0d2d8187898f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 4d05b9facc0b7e5a3ff31146fc42f98d680010264c6606a3154e9da2dfa6a0cc
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: c5ae66679294bcb1bb27982832ef2733d76544f94a2d321c99dab3b370062b0c
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/linalg_public_api/20261013_140259/fixtures/manifest.json: db62318dc4ca5c2f46c47b35d0e9804251416a8c0fa52f5c1e022b5e2de1e38e
    target/release/benchmark_gpu_public_api: b505a2bdb3d0c5a7bffd1390af659a154753ea998ec68becd4779e137c9c7150
  reference_numpy_version: 2.4.2
```
