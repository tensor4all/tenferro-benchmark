# CPU eager investigation after PR #1820

Baseline: `a081f618`; candidate CPU code: `26abf5ec`.
AMD EPYC 7713P, release, oneMKL 2026.1; Torch 2.12.0+cpu / MKL 2024.2.
1T: CPU1. 4T: CPU1–4 with explicit Intel OpenMP placement; 3 warmups / 15 bounded-batch samples.
All input preparation is untimed. Raw samples and protocol: `data/results/amd-cpu/cpu/eager-1820/`.

| T | suite | operation | shape | baseline µs | candidate µs | Torch µs | speedup | candidate/Torch |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | small | matmul | 2x2 | 6.443 | 6.020 | 1.938 | 1.07 | 3.11 |
| 1 | small | einsum_ij_jk_ik | 2x2 | 6.509 | 6.703 | 11.153 | 0.97 | 0.60 |
| 1 | small | svd | 2x2 | 12.913 | 12.968 | 9.244 | 1.00 | 1.40 |
| 1 | small | qr | 2x2 | 10.362 | 9.885 | 4.538 | 1.05 | 2.18 |
| 1 | small | eigh | 2x2 | 12.036 | 10.976 | 6.859 | 1.10 | 1.60 |
| 1 | small | solve | 2x2,rhs=1 | 15.877 | 16.687 | 13.804 | 0.95 | 1.21 |
| 1 | small | solve | 2x2,rhs=4 | 15.756 | 15.418 | 13.476 | 1.02 | 1.14 |
| 1 | small | grad_sum_matmul_backward | 2x2 | 276.415 | 279.541 | 60.088 | 0.99 | 4.65 |
| 1 | small | grad_sum_svd_s_backward | 2x2 | 252.235 | 254.209 | 100.363 | 0.99 | 2.53 |
| 1 | small | grad_sum_solve_backward | 2x2,rhs=1 | 371.565 | 377.326 | 117.832 | 0.98 | 3.20 |
| 1 | small | matmul | 4x4 | 6.568 | 6.562 | 1.764 | 1.00 | 3.72 |
| 1 | small | einsum_ij_jk_ik | 4x4 | 7.608 | 7.419 | 11.086 | 1.03 | 0.67 |
| 1 | small | svd | 4x4 | 18.024 | 17.224 | 12.892 | 1.05 | 1.34 |
| 1 | small | qr | 4x4 | 10.446 | 10.499 | 5.087 | 0.99 | 2.06 |
| 1 | small | eigh | 4x4 | 13.712 | 13.444 | 8.761 | 1.02 | 1.53 |
| 1 | small | solve | 4x4,rhs=1 | 15.995 | 15.845 | 14.035 | 1.01 | 1.13 |
| 1 | small | solve | 4x4,rhs=4 | 16.406 | 16.273 | 14.416 | 1.01 | 1.13 |
| 1 | small | grad_sum_matmul_backward | 4x4 | 292.458 | 316.323 | 60.790 | 0.92 | 5.20 |
| 1 | small | grad_sum_svd_s_backward | 4x4 | 269.618 | 278.884 | 95.985 | 0.97 | 2.91 |
| 1 | small | grad_sum_solve_backward | 4x4,rhs=1 | 384.557 | 389.788 | 101.311 | 0.99 | 3.85 |
| 1 | small | matmul | 8x8 | 6.741 | 6.451 | 2.101 | 1.05 | 3.07 |
| 1 | small | einsum_ij_jk_ik | 8x8 | 7.249 | 7.148 | 11.477 | 1.01 | 0.62 |
| 1 | small | svd | 8x8 | 25.062 | 24.881 | 19.880 | 1.01 | 1.25 |
| 1 | small | qr | 8x8 | 11.651 | 11.636 | 5.622 | 1.00 | 2.07 |
| 1 | small | eigh | 8x8 | 20.030 | 16.568 | 11.674 | 1.21 | 1.42 |
| 1 | small | solve | 8x8,rhs=1 | 17.336 | 16.118 | 13.871 | 1.08 | 1.16 |
| 1 | small | solve | 8x8,rhs=4 | 17.129 | 18.243 | 15.062 | 0.94 | 1.21 |
| 1 | small | grad_sum_matmul_backward | 8x8 | 294.482 | 309.243 | 62.746 | 0.95 | 4.93 |
| 1 | small | grad_sum_svd_s_backward | 8x8 | 292.484 | 281.072 | 109.835 | 1.04 | 2.56 |
| 1 | small | grad_sum_solve_backward | 8x8,rhs=1 | 397.249 | 404.039 | 101.941 | 0.98 | 3.96 |
| 1 | small | matmul | 16x16 | 7.923 | 7.289 | 2.599 | 1.09 | 2.80 |
| 1 | small | einsum_ij_jk_ik | 16x16 | 8.059 | 8.809 | 12.328 | 0.91 | 0.71 |
| 1 | small | svd | 16x16 | 47.977 | 54.510 | 46.944 | 0.88 | 1.16 |
| 1 | small | qr | 16x16 | 15.469 | 16.045 | 8.876 | 0.96 | 1.81 |
| 1 | small | eigh | 16x16 | 28.327 | 29.375 | 26.640 | 0.96 | 1.10 |
| 1 | small | solve | 16x16,rhs=1 | 18.116 | 17.670 | 15.415 | 1.03 | 1.15 |
| 1 | small | solve | 16x16,rhs=4 | 19.981 | 18.932 | 17.849 | 1.06 | 1.06 |
| 1 | small | grad_sum_matmul_backward | 16x16 | 307.947 | 317.412 | 70.064 | 0.97 | 4.53 |
| 1 | small | grad_sum_svd_s_backward | 16x16 | 332.458 | 336.778 | 135.558 | 0.99 | 2.48 |
| 1 | small | grad_sum_solve_backward | 16x16,rhs=1 | 414.462 | 443.424 | 108.113 | 0.93 | 4.10 |
| 1 | small | matmul | 32x32 | 10.755 | 9.841 | 5.348 | 1.09 | 1.84 |
| 1 | small | einsum_ij_jk_ik | 32x32 | 11.648 | 11.554 | 17.065 | 1.01 | 0.68 |
| 1 | small | svd | 32x32 | 137.370 | 137.570 | 154.493 | 1.00 | 0.89 |
| 1 | small | qr | 32x32 | 26.044 | 26.400 | 23.311 | 0.99 | 1.13 |
| 1 | small | eigh | 32x32 | 75.417 | 75.788 | 84.274 | 1.00 | 0.90 |
| 1 | small | solve | 32x32,rhs=1 | 29.648 | 27.010 | 23.087 | 1.10 | 1.17 |
| 1 | small | solve | 32x32,rhs=4 | 31.246 | 26.469 | 24.775 | 1.18 | 1.07 |
| 1 | small | grad_sum_matmul_backward | 32x32 | 314.916 | 318.817 | 76.450 | 0.99 | 4.17 |
| 1 | small | grad_sum_svd_s_backward | 32x32 | 523.471 | 526.336 | 239.179 | 0.99 | 2.20 |
| 1 | small | grad_sum_solve_backward | 32x32,rhs=1 | 483.280 | 474.357 | 119.001 | 1.02 | 3.99 |
| 1 | large | matmul | 128x128 | 138.059 | 116.113 | 161.502 | 1.19 | 0.72 |
| 1 | large | matmul | 256x256 | 714.439 | 833.549 | 900.010 | 0.86 | 0.93 |
| 1 | large | matmul | 512x512 | 5500.479 | 5392.446 | 6731.234 | 1.02 | 0.80 |
| 1 | large | matmul | 1024x1024 | 46818.864 | 43654.652 | 49928.353 | 1.07 | 0.87 |
| 1 | large | matmul_rect | 1024x256 * 256x1024 | 10551.253 | 12168.886 | 11585.466 | 0.87 | 1.05 |
| 1 | large | matmul_rect | 256x1024 * 1024x256 | 2829.992 | 2846.382 | 3034.521 | 0.99 | 0.94 |
| 1 | large | svd | 64x64 | 524.180 | 539.694 | 575.192 | 0.97 | 0.94 |
| 1 | large | qr | 64x64 | 93.215 | 94.117 | 96.014 | 0.99 | 0.98 |
| 1 | large | eigh | 64x64 | 284.971 | 271.330 | 276.831 | 1.05 | 0.98 |
| 1 | large | solve | 64x64,rhs=1 | 59.993 | 50.507 | 41.964 | 1.19 | 1.20 |
| 1 | large | solve | 64x64,rhs=16 | 76.210 | 67.951 | 57.042 | 1.12 | 1.19 |
| 1 | large | solve | 64x64,rhs=64 | 123.094 | 140.114 | 102.124 | 0.88 | 1.37 |
| 1 | large | svd | 128x128 | 2200.072 | 2198.957 | 2340.643 | 1.00 | 0.94 |
| 1 | large | qr | 128x128 | 458.140 | 460.941 | 501.090 | 0.99 | 0.92 |
| 1 | large | eigh | 128x128 | 1253.821 | 1165.778 | 1152.096 | 1.08 | 1.01 |
| 1 | large | solve | 128x128,rhs=1 | 182.242 | 146.305 | 125.742 | 1.25 | 1.16 |
| 1 | large | solve | 128x128,rhs=16 | 231.077 | 236.968 | 183.196 | 0.98 | 1.29 |
| 1 | large | solve | 128x128,rhs=64 | 418.453 | 341.501 | 284.489 | 1.23 | 1.20 |
| 1 | large | svd | 256x256 | 11674.030 | 12993.903 | 13338.754 | 0.90 | 0.97 |
| 1 | large | qr | 256x256 | 2634.390 | 3184.484 | 2821.109 | 0.83 | 1.13 |
| 1 | large | eigh | 256x256 | 6904.385 | 6396.428 | 6479.426 | 1.08 | 0.99 |
| 1 | large | solve | 256x256,rhs=1 | 851.032 | 575.524 | 667.611 | 1.48 | 0.86 |
| 1 | large | solve | 256x256,rhs=16 | 975.612 | 720.454 | 872.199 | 1.35 | 0.83 |
| 1 | large | solve | 256x256,rhs=64 | 1327.433 | 1059.275 | 1071.956 | 1.25 | 0.99 |
| 1 | large | grad_sum_matmul | 64x64 | 44.288 | 44.383 | 23.568 | 1.00 | 1.88 |
| 1 | large | grad_sum_matmul_backward | 64x64 | 379.874 | 421.808 | 121.813 | 0.90 | 3.46 |
| 1 | large | grad_sum_svd_s_backward | 64x64 | 1442.165 | 1379.487 | 660.014 | 1.05 | 2.09 |
| 1 | large | grad_sum_solve_backward | 64x64,rhs=1 | 715.412 | 613.501 | 141.601 | 1.17 | 4.33 |
| 1 | large | grad_sum_matmul | 128x128 | 147.592 | 148.530 | 130.289 | 0.99 | 1.14 |
| 1 | large | grad_sum_matmul_backward | 128x128 | 702.722 | 836.025 | 404.959 | 0.84 | 2.06 |
| 1 | large | grad_sum_svd_s_backward | 128x128 | 5063.085 | 4881.629 | 2584.001 | 1.04 | 1.89 |
| 1 | large | grad_sum_solve_backward | 128x128,rhs=1 | 1329.800 | 1131.943 | 240.253 | 1.17 | 4.71 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 2x2xbatch16 (native batch layout) | 8.583 | 8.530 | 13.485 | 1.01 | 0.63 |
| 1 | batched | batched_svd | 2x2xbatch16 (native batch layout) | 84.769 | 97.656 | 23.696 | 0.87 | 4.12 |
| 1 | batched | batched_qr | 2x2xbatch16 (native batch layout) | 50.747 | 14.851 | 7.819 | 3.42 | 1.90 |
| 1 | batched | batched_eigh | 2x2xbatch16 (native batch layout) | 57.629 | 55.985 | 15.909 | 1.03 | 3.52 |
| 1 | batched | batched_solve | 2x2xbatch16 (native batch layout),rhs=1 | 57.757 | 16.114 | 17.856 | 3.58 | 0.90 |
| 1 | batched | grad_sum_batched_matmul_backward | 2x2xbatch16 (native batch layout) | 327.496 | 340.199 | 122.800 | 0.96 | 2.77 |
| 1 | batched | grad_sum_batched_solve_backward | 2x2xbatch16 (native batch layout),rhs=1 | 552.898 | 425.459 | 112.577 | 1.30 | 3.78 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 4x4xbatch16 (native batch layout) | 9.728 | 9.286 | 13.670 | 1.05 | 0.68 |
| 1 | batched | batched_svd | 4x4xbatch16 (native batch layout) | 139.828 | 130.135 | 64.244 | 1.07 | 2.03 |
| 1 | batched | batched_qr | 4x4xbatch16 (native batch layout) | 62.998 | 20.287 | 14.214 | 3.11 | 1.43 |
| 1 | batched | batched_eigh | 4x4xbatch16 (native batch layout) | 78.220 | 76.407 | 33.328 | 1.02 | 2.29 |
| 1 | batched | batched_solve | 4x4xbatch16 (native batch layout),rhs=1 | 51.157 | 17.885 | 20.363 | 2.86 | 0.88 |
| 1 | batched | grad_sum_batched_matmul_backward | 4x4xbatch16 (native batch layout) | 328.032 | 332.341 | 147.440 | 0.99 | 2.25 |
| 1 | batched | grad_sum_batched_solve_backward | 4x4xbatch16 (native batch layout),rhs=1 | 572.596 | 437.096 | 126.554 | 1.31 | 3.45 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 8x8xbatch16 (native batch layout) | 10.440 | 10.576 | 15.525 | 0.99 | 0.68 |
| 1 | batched | batched_svd | 8x8xbatch16 (native batch layout) | 266.716 | 268.567 | 189.903 | 0.99 | 1.41 |
| 1 | batched | batched_qr | 8x8xbatch16 (native batch layout) | 70.729 | 34.143 | 23.202 | 2.07 | 1.47 |
| 1 | batched | batched_eigh | 8x8xbatch16 (native batch layout) | 137.726 | 127.698 | 84.998 | 1.08 | 1.50 |
| 1 | batched | batched_solve | 8x8xbatch16 (native batch layout),rhs=1 | 58.790 | 19.939 | 20.790 | 2.95 | 0.96 |
| 1 | batched | grad_sum_batched_matmul_backward | 8x8xbatch16 (native batch layout) | 333.851 | 334.703 | 210.433 | 1.00 | 1.59 |
| 1 | batched | grad_sum_batched_solve_backward | 8x8xbatch16 (native batch layout),rhs=1 | 601.118 | 450.506 | 123.477 | 1.33 | 3.65 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 16x16xbatch16 (native batch layout) | 19.061 | 20.765 | 25.283 | 0.92 | 0.82 |
| 1 | batched | batched_svd | 16x16xbatch16 (native batch layout) | 607.358 | 607.027 | 593.270 | 1.00 | 1.02 |
| 1 | batched | batched_qr | 16x16xbatch16 (native batch layout) | 119.580 | 71.913 | 73.325 | 1.66 | 0.98 |
| 1 | batched | batched_eigh | 16x16xbatch16 (native batch layout) | 304.097 | 313.764 | 328.066 | 0.97 | 0.96 |
| 1 | batched | batched_solve | 16x16xbatch16 (native batch layout),rhs=1 | 86.162 | 31.363 | 40.715 | 2.75 | 0.77 |
| 1 | batched | grad_sum_batched_matmul_backward | 16x16xbatch16 (native batch layout) | 380.464 | 388.934 | 244.129 | 0.98 | 1.59 |
| 1 | batched | grad_sum_batched_solve_backward | 16x16xbatch16 (native batch layout),rhs=1 | 780.390 | 550.692 | 175.612 | 1.42 | 3.14 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 2x2xbatch64 (native batch layout) | 14.168 | 14.996 | 13.243 | 0.94 | 1.13 |
| 1 | batched | batched_svd | 2x2xbatch64 (native batch layout) | 279.058 | 272.059 | 68.049 | 1.03 | 4.00 |
| 1 | batched | batched_qr | 2x2xbatch64 (native batch layout) | 165.703 | 26.080 | 17.102 | 6.35 | 1.52 |
| 1 | batched | batched_eigh | 2x2xbatch64 (native batch layout) | 190.068 | 180.474 | 34.437 | 1.05 | 5.24 |
| 1 | batched | batched_solve | 2x2xbatch64 (native batch layout),rhs=1 | 143.159 | 18.323 | 20.354 | 7.81 | 0.90 |
| 1 | batched | grad_sum_batched_matmul_backward | 2x2xbatch64 (native batch layout) | 343.927 | 354.832 | 121.404 | 0.97 | 2.92 |
| 1 | batched | grad_sum_batched_solve_backward | 2x2xbatch64 (native batch layout),rhs=1 | 937.304 | 501.788 | 118.412 | 1.87 | 4.24 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 4x4xbatch64 (native batch layout) | 15.712 | 16.105 | 24.297 | 0.98 | 0.66 |
| 1 | batched | batched_svd | 4x4xbatch64 (native batch layout) | 456.369 | 459.252 | 231.360 | 0.99 | 1.99 |
| 1 | batched | batched_qr | 4x4xbatch64 (native batch layout) | 191.674 | 55.907 | 39.495 | 3.43 | 1.42 |
| 1 | batched | batched_eigh | 4x4xbatch64 (native batch layout) | 262.290 | 275.331 | 124.705 | 0.95 | 2.21 |
| 1 | batched | batched_solve | 4x4xbatch64 (native batch layout),rhs=1 | 149.182 | 22.641 | 23.912 | 6.59 | 0.95 |
| 1 | batched | grad_sum_batched_matmul_backward | 4x4xbatch64 (native batch layout) | 376.951 | 357.624 | 142.843 | 1.05 | 2.50 |
| 1 | batched | grad_sum_batched_solve_backward | 4x4xbatch64 (native batch layout),rhs=1 | 1052.154 | 462.734 | 144.118 | 2.27 | 3.21 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 8x8xbatch64 (native batch layout) | 20.743 | 21.599 | 24.465 | 0.96 | 0.88 |
| 1 | batched | batched_svd | 8x8xbatch64 (native batch layout) | 966.874 | 961.481 | 797.658 | 1.01 | 1.21 |
| 1 | batched | batched_qr | 8x8xbatch64 (native batch layout) | 233.218 | 94.883 | 80.223 | 2.46 | 1.18 |
| 1 | batched | batched_eigh | 8x8xbatch64 (native batch layout) | 466.442 | 468.447 | 318.686 | 1.00 | 1.47 |
| 1 | batched | batched_solve | 8x8xbatch64 (native batch layout),rhs=1 | 185.092 | 36.911 | 40.009 | 5.01 | 0.92 |
| 1 | batched | grad_sum_batched_matmul_backward | 8x8xbatch64 (native batch layout) | 396.867 | 397.584 | 452.539 | 1.00 | 0.88 |
| 1 | batched | grad_sum_batched_solve_backward | 8x8xbatch64 (native batch layout),rhs=1 | 1124.111 | 558.334 | 164.520 | 2.01 | 3.39 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 16x16xbatch64 (native batch layout) | 55.093 | 56.032 | 60.189 | 0.98 | 0.93 |
| 1 | batched | batched_svd | 16x16xbatch64 (native batch layout) | 2397.388 | 2382.613 | 2382.907 | 1.01 | 1.00 |
| 1 | batched | batched_qr | 16x16xbatch64 (native batch layout) | 429.121 | 251.927 | 277.385 | 1.70 | 0.91 |
| 1 | batched | batched_eigh | 16x16xbatch64 (native batch layout) | 1203.605 | 1370.151 | 1116.446 | 0.88 | 1.23 |
| 1 | batched | batched_solve | 16x16xbatch64 (native batch layout),rhs=1 | 307.749 | 81.563 | 89.400 | 3.77 | 0.91 |
| 1 | batched | grad_sum_batched_matmul_backward | 16x16xbatch64 (native batch layout) | 556.252 | 554.024 | 666.010 | 1.00 | 0.83 |
| 1 | batched | grad_sum_batched_solve_backward | 16x16xbatch64 (native batch layout),rhs=1 | 1729.464 | 834.302 | 300.047 | 2.07 | 2.78 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 2x2xbatch256 (native batch layout) | 38.942 | 43.284 | 20.150 | 0.90 | 2.15 |
| 1 | batched | batched_svd | 2x2xbatch256 (native batch layout) | 1060.570 | 1028.185 | 258.755 | 1.03 | 3.97 |
| 1 | batched | batched_qr | 2x2xbatch256 (native batch layout) | 615.273 | 68.502 | 52.463 | 8.98 | 1.31 |
| 1 | batched | batched_eigh | 2x2xbatch256 (native batch layout) | 698.366 | 684.569 | 116.782 | 1.02 | 5.86 |
| 1 | batched | batched_solve | 2x2xbatch256 (native batch layout),rhs=1 | 528.806 | 31.598 | 34.450 | 16.74 | 0.92 |
| 1 | batched | grad_sum_batched_matmul_backward | 2x2xbatch256 (native batch layout) | 426.491 | 422.962 | 135.599 | 1.01 | 3.12 |
| 1 | batched | grad_sum_batched_solve_backward | 2x2xbatch256 (native batch layout),rhs=1 | 2456.130 | 522.819 | 169.847 | 4.70 | 3.08 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 4x4xbatch256 (native batch layout) | 43.143 | 43.302 | 46.823 | 1.00 | 0.92 |
| 1 | batched | batched_svd | 4x4xbatch256 (native batch layout) | 1761.790 | 1728.769 | 895.377 | 1.02 | 1.93 |
| 1 | batched | batched_qr | 4x4xbatch256 (native batch layout) | 713.360 | 152.312 | 128.249 | 4.68 | 1.19 |
| 1 | batched | batched_eigh | 4x4xbatch256 (native batch layout) | 993.640 | 1147.321 | 491.537 | 0.87 | 2.33 |
| 1 | batched | batched_solve | 4x4xbatch256 (native batch layout),rhs=1 | 543.979 | 39.527 | 50.094 | 13.76 | 0.79 |
| 1 | batched | grad_sum_batched_matmul_backward | 4x4xbatch256 (native batch layout) | 461.566 | 467.373 | 222.473 | 0.99 | 2.10 |
| 1 | batched | grad_sum_batched_solve_backward | 4x4xbatch256 (native batch layout),rhs=1 | 2603.280 | 597.389 | 180.759 | 4.36 | 3.30 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 8x8xbatch256 (native batch layout) | 62.529 | 67.276 | 55.415 | 0.93 | 1.21 |
| 1 | batched | batched_svd | 8x8xbatch256 (native batch layout) | 3798.019 | 3761.057 | 2900.568 | 1.01 | 1.30 |
| 1 | batched | batched_qr | 8x8xbatch256 (native batch layout) | 883.908 | 325.167 | 298.861 | 2.72 | 1.09 |
| 1 | batched | batched_eigh | 8x8xbatch256 (native batch layout) | 1821.377 | 1797.536 | 1254.983 | 1.01 | 1.43 |
| 1 | batched | batched_solve | 8x8xbatch256 (native batch layout),rhs=1 | 702.213 | 106.642 | 108.957 | 6.58 | 0.98 |
| 1 | batched | grad_sum_batched_matmul_backward | 8x8xbatch256 (native batch layout) | 619.653 | 604.592 | 1420.829 | 1.02 | 0.43 |
| 1 | batched | grad_sum_batched_solve_backward | 8x8xbatch256 (native batch layout),rhs=1 | 3266.021 | 957.153 | 321.559 | 3.41 | 2.98 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 16x16xbatch256 (native batch layout) | 236.881 | 195.736 | 205.589 | 1.21 | 0.95 |
| 1 | batched | batched_svd | 16x16xbatch256 (native batch layout) | 9530.110 | 9516.500 | 9523.221 | 1.00 | 1.00 |
| 1 | batched | batched_qr | 16x16xbatch256 (native batch layout) | 1659.246 | 948.269 | 1043.004 | 1.75 | 0.91 |
| 1 | batched | batched_eigh | 16x16xbatch256 (native batch layout) | 4749.499 | 4787.646 | 4440.386 | 0.99 | 1.08 |
| 1 | batched | batched_solve | 16x16xbatch256 (native batch layout),rhs=1 | 1230.392 | 290.569 | 302.078 | 4.23 | 0.96 |
| 1 | batched | grad_sum_batched_matmul_backward | 16x16xbatch256 (native batch layout) | 1197.611 | 1127.137 | 1927.944 | 1.06 | 0.58 |
| 1 | batched | grad_sum_batched_solve_backward | 16x16xbatch256 (native batch layout),rhs=1 | 5199.270 | 1748.857 | 778.371 | 2.97 | 2.25 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 2x2xbatch1024 (native batch layout) | 138.050 | 141.952 | 37.995 | 0.97 | 3.74 |
| 1 | batched | batched_svd | 2x2xbatch1024 (native batch layout) | 4141.585 | 4022.911 | 942.268 | 1.03 | 4.27 |
| 1 | batched | batched_qr | 2x2xbatch1024 (native batch layout) | 2441.567 | 236.519 | 196.843 | 10.32 | 1.20 |
| 1 | batched | batched_eigh | 2x2xbatch1024 (native batch layout) | 2723.264 | 3118.062 | 447.796 | 0.87 | 6.96 |
| 1 | batched | batched_solve | 2x2xbatch1024 (native batch layout),rhs=1 | 2139.615 | 78.933 | 92.609 | 27.11 | 0.85 |
| 1 | batched | grad_sum_batched_matmul_backward | 2x2xbatch1024 (native batch layout) | 800.904 | 723.765 | 195.359 | 1.11 | 3.70 |
| 1 | batched | grad_sum_batched_solve_backward | 2x2xbatch1024 (native batch layout),rhs=1 | 10133.480 | 796.715 | 268.294 | 12.72 | 2.97 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 4x4xbatch1024 (native batch layout) | 164.904 | 152.629 | 148.733 | 1.08 | 1.03 |
| 1 | batched | batched_svd | 4x4xbatch1024 (native batch layout) | 6976.777 | 7026.808 | 3980.227 | 0.99 | 1.77 |
| 1 | batched | batched_qr | 4x4xbatch1024 (native batch layout) | 2812.122 | 576.642 | 459.304 | 4.88 | 1.26 |
| 1 | batched | batched_eigh | 4x4xbatch1024 (native batch layout) | 4246.408 | 3831.265 | 1688.850 | 1.11 | 2.27 |
| 1 | batched | batched_solve | 4x4xbatch1024 (native batch layout),rhs=1 | 2140.167 | 111.973 | 124.524 | 19.11 | 0.90 |
| 1 | batched | grad_sum_batched_matmul_backward | 4x4xbatch1024 (native batch layout) | 884.288 | 830.638 | 610.229 | 1.06 | 1.36 |
| 1 | batched | grad_sum_batched_solve_backward | 4x4xbatch1024 (native batch layout),rhs=1 | 9350.764 | 1061.513 | 340.800 | 8.81 | 3.11 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 8x8xbatch1024 (native batch layout) | 280.806 | 237.043 | 210.656 | 1.18 | 1.13 |
| 1 | batched | batched_svd | 8x8xbatch1024 (native batch layout) | 15113.662 | 15294.308 | 11503.176 | 0.99 | 1.33 |
| 1 | batched | batched_qr | 8x8xbatch1024 (native batch layout) | 3473.499 | 1257.426 | 1191.570 | 2.76 | 1.06 |
| 1 | batched | batched_eigh | 8x8xbatch1024 (native batch layout) | 8531.488 | 7525.565 | 4970.909 | 1.13 | 1.51 |
| 1 | batched | batched_solve | 8x8xbatch1024 (native batch layout),rhs=1 | 2829.227 | 375.151 | 358.834 | 7.54 | 1.05 |
| 1 | batched | grad_sum_batched_matmul_backward | 8x8xbatch1024 (native batch layout) | 1292.990 | 1281.407 | 5124.839 | 1.01 | 0.25 |
| 1 | batched | grad_sum_batched_solve_backward | 8x8xbatch1024 (native batch layout),rhs=1 | 11331.019 | 2189.244 | 943.171 | 5.18 | 2.32 |
| 1 | batched | batched_matmul_ikb_kjb_ijb | 16x16xbatch1024 (native batch layout) | 786.276 | 768.925 | 799.524 | 1.02 | 0.96 |
| 1 | batched | batched_svd | 16x16xbatch1024 (native batch layout) | 39306.370 | 39646.741 | 40473.884 | 0.99 | 0.98 |
| 1 | batched | batched_qr | 16x16xbatch1024 (native batch layout) | 7194.964 | 3747.262 | 4130.968 | 1.92 | 0.91 |
| 1 | batched | batched_eigh | 16x16xbatch1024 (native batch layout) | 19426.812 | 19593.818 | 17769.301 | 0.99 | 1.10 |
| 1 | batched | batched_solve | 16x16xbatch1024 (native batch layout),rhs=1 | 4606.420 | 1092.395 | 1158.656 | 4.22 | 0.94 |
| 1 | batched | grad_sum_batched_matmul_backward | 16x16xbatch1024 (native batch layout) | 4464.645 | 3449.273 | 7656.998 | 1.29 | 0.45 |
| 1 | batched | grad_sum_batched_solve_backward | 16x16xbatch1024 (native batch layout),rhs=1 | 20952.982 | 5555.021 | 2698.533 | 3.77 | 2.06 |
| 4 | small | matmul | 2x2 | 6.936 | 6.001 | 1.896 | 1.16 | 3.16 |
| 4 | small | einsum_ij_jk_ik | 2x2 | 6.574 | 6.738 | 11.536 | 0.98 | 0.58 |
| 4 | small | svd | 2x2 | 12.819 | 12.843 | 9.029 | 1.00 | 1.42 |
| 4 | small | qr | 2x2 | 9.675 | 10.018 | 6.882 | 0.97 | 1.46 |
| 4 | small | eigh | 2x2 | 10.718 | 10.645 | 7.326 | 1.01 | 1.45 |
| 4 | small | solve | 2x2,rhs=1 | 15.631 | 16.362 | 14.586 | 0.96 | 1.12 |
| 4 | small | solve | 2x2,rhs=4 | 15.628 | 15.269 | 15.265 | 1.02 | 1.00 |
| 4 | small | grad_sum_matmul_backward | 2x2 | 296.309 | 285.466 | 68.211 | 1.04 | 4.19 |
| 4 | small | grad_sum_svd_s_backward | 2x2 | 253.071 | 256.353 | 95.346 | 0.99 | 2.69 |
| 4 | small | grad_sum_solve_backward | 2x2,rhs=1 | 377.926 | 383.367 | 99.954 | 0.99 | 3.84 |
| 4 | small | matmul | 4x4 | 6.642 | 6.687 | 1.805 | 0.99 | 3.70 |
| 4 | small | einsum_ij_jk_ik | 4x4 | 7.666 | 7.314 | 11.132 | 1.05 | 0.66 |
| 4 | small | svd | 4x4 | 17.432 | 17.152 | 12.434 | 1.02 | 1.38 |
| 4 | small | qr | 4x4 | 10.478 | 10.908 | 7.386 | 0.96 | 1.48 |
| 4 | small | eigh | 4x4 | 13.578 | 13.421 | 8.348 | 1.01 | 1.61 |
| 4 | small | solve | 4x4,rhs=1 | 16.881 | 15.647 | 14.179 | 1.08 | 1.10 |
| 4 | small | solve | 4x4,rhs=4 | 16.232 | 15.922 | 14.073 | 1.02 | 1.13 |
| 4 | small | grad_sum_matmul_backward | 4x4 | 290.526 | 300.707 | 61.427 | 0.97 | 4.90 |
| 4 | small | grad_sum_svd_s_backward | 4x4 | 270.825 | 283.896 | 100.739 | 0.95 | 2.82 |
| 4 | small | grad_sum_solve_backward | 4x4,rhs=1 | 405.576 | 429.397 | 106.801 | 0.94 | 4.02 |
| 4 | small | matmul | 8x8 | 7.402 | 6.922 | 2.110 | 1.07 | 3.28 |
| 4 | small | einsum_ij_jk_ik | 8x8 | 8.212 | 7.310 | 11.447 | 1.12 | 0.64 |
| 4 | small | svd | 8x8 | 27.438 | 25.117 | 19.609 | 1.09 | 1.28 |
| 4 | small | qr | 8x8 | 12.325 | 11.633 | 8.249 | 1.06 | 1.41 |
| 4 | small | eigh | 8x8 | 16.407 | 17.041 | 11.658 | 0.96 | 1.46 |
| 4 | small | solve | 8x8,rhs=1 | 16.771 | 16.836 | 14.664 | 1.00 | 1.15 |
| 4 | small | solve | 8x8,rhs=4 | 18.109 | 16.579 | 15.834 | 1.09 | 1.05 |
| 4 | small | grad_sum_matmul_backward | 8x8 | 295.714 | 303.732 | 60.540 | 0.97 | 5.02 |
| 4 | small | grad_sum_svd_s_backward | 8x8 | 295.280 | 288.085 | 105.042 | 1.02 | 2.74 |
| 4 | small | grad_sum_solve_backward | 8x8,rhs=1 | 404.792 | 412.986 | 108.074 | 0.98 | 3.82 |
| 4 | small | matmul | 16x16 | 7.403 | 7.624 | 2.853 | 0.97 | 2.67 |
| 4 | small | einsum_ij_jk_ik | 16x16 | 8.781 | 7.863 | 13.507 | 1.12 | 0.58 |
| 4 | small | svd | 16x16 | 50.907 | 51.792 | 46.714 | 0.98 | 1.11 |
| 4 | small | qr | 16x16 | 15.454 | 15.107 | 12.519 | 1.02 | 1.21 |
| 4 | small | eigh | 16x16 | 28.325 | 28.395 | 24.055 | 1.00 | 1.18 |
| 4 | small | solve | 16x16,rhs=1 | 19.080 | 18.354 | 15.783 | 1.04 | 1.16 |
| 4 | small | solve | 16x16,rhs=4 | 20.183 | 18.018 | 16.400 | 1.12 | 1.10 |
| 4 | small | grad_sum_matmul_backward | 16x16 | 317.649 | 314.823 | 65.032 | 1.01 | 4.84 |
| 4 | small | grad_sum_svd_s_backward | 16x16 | 336.857 | 338.389 | 137.349 | 1.00 | 2.46 |
| 4 | small | grad_sum_solve_backward | 16x16,rhs=1 | 423.654 | 427.105 | 113.860 | 0.99 | 3.75 |
| 4 | small | matmul | 32x32 | 13.203 | 13.192 | 8.284 | 1.00 | 1.59 |
| 4 | small | einsum_ij_jk_ik | 32x32 | 14.215 | 14.668 | 20.833 | 0.97 | 0.70 |
| 4 | small | svd | 32x32 | 177.310 | 178.170 | 186.056 | 1.00 | 0.96 |
| 4 | small | qr | 32x32 | 30.495 | 30.525 | 28.623 | 1.00 | 1.07 |
| 4 | small | eigh | 32x32 | 99.717 | 99.047 | 96.395 | 1.01 | 1.03 |
| 4 | small | solve | 32x32,rhs=1 | 35.114 | 29.494 | 24.887 | 1.19 | 1.19 |
| 4 | small | solve | 32x32,rhs=4 | 42.205 | 36.481 | 33.094 | 1.16 | 1.10 |
| 4 | small | grad_sum_matmul_backward | 32x32 | 386.034 | 391.086 | 94.854 | 0.99 | 4.12 |
| 4 | small | grad_sum_svd_s_backward | 32x32 | 650.434 | 661.979 | 300.290 | 0.98 | 2.20 |
| 4 | small | grad_sum_solve_backward | 32x32,rhs=1 | 579.852 | 564.398 | 122.973 | 1.03 | 4.59 |
| 4 | large | matmul | 128x128 | 65.175 | 69.643 | 108.844 | 0.94 | 0.64 |
| 4 | large | matmul | 256x256 | 301.497 | 301.732 | 441.169 | 1.00 | 0.68 |
| 4 | large | matmul | 512x512 | 2008.103 | 2046.044 | 2987.367 | 0.98 | 0.68 |
| 4 | large | matmul | 1024x1024 | 12953.259 | 13085.551 | 20934.004 | 0.99 | 0.63 |
| 4 | large | matmul_rect | 1024x256 * 256x1024 | 3226.547 | 3284.128 | 5656.101 | 0.98 | 0.58 |
| 4 | large | matmul_rect | 256x1024 * 1024x256 | 987.536 | 1007.866 | 1550.065 | 0.98 | 0.65 |
| 4 | large | svd | 64x64 | 584.511 | 610.561 | 623.320 | 0.96 | 0.98 |
| 4 | large | qr | 64x64 | 132.679 | 139.030 | 122.076 | 0.95 | 1.14 |
| 4 | large | eigh | 64x64 | 251.010 | 252.148 | 239.424 | 1.00 | 1.05 |
| 4 | large | solve | 64x64,rhs=1 | 70.784 | 60.613 | 41.336 | 1.17 | 1.47 |
| 4 | large | solve | 64x64,rhs=16 | 92.666 | 81.206 | 66.660 | 1.14 | 1.22 |
| 4 | large | solve | 64x64,rhs=64 | 134.915 | 114.564 | 94.263 | 1.18 | 1.22 |
| 4 | large | svd | 128x128 | 2203.765 | 2202.114 | 2209.096 | 1.00 | 1.00 |
| 4 | large | qr | 128x128 | 428.289 | 384.364 | 384.355 | 1.11 | 1.00 |
| 4 | large | eigh | 128x128 | 898.759 | 896.303 | 3938.316 | 1.00 | 0.23 |
| 4 | large | solve | 128x128,rhs=1 | 196.042 | 175.982 | 151.271 | 1.11 | 1.16 |
| 4 | large | solve | 128x128,rhs=16 | 277.390 | 229.159 | 958.314 | 1.21 | 0.24 |
| 4 | large | solve | 128x128,rhs=64 | 358.983 | 250.669 | 249.988 | 1.43 | 1.00 |
| 4 | large | svd | 256x256 | 10157.958 | 10092.583 | 10797.643 | 1.01 | 0.93 |
| 4 | large | qr | 256x256 | 1630.089 | 1642.451 | 1645.088 | 0.99 | 1.00 |
| 4 | large | eigh | 256x256 | 3921.002 | 3909.541 | 6671.606 | 1.00 | 0.59 |
| 4 | large | solve | 256x256,rhs=1 | 679.067 | 340.302 | 361.918 | 2.00 | 0.94 |
| 4 | large | solve | 256x256,rhs=16 | 806.299 | 473.537 | 11192.527 | 1.70 | 0.04 |
| 4 | large | solve | 256x256,rhs=64 | 995.995 | 663.624 | 41627.286 | 1.50 | 0.02 |
| 4 | large | grad_sum_matmul | 64x64 | 54.594 | 53.913 | 35.011 | 1.01 | 1.54 |
| 4 | large | grad_sum_matmul_backward | 64x64 | 445.915 | 447.399 | 130.959 | 1.00 | 3.42 |
| 4 | large | grad_sum_svd_s_backward | 64x64 | 1601.011 | 1598.345 | 762.965 | 1.00 | 2.09 |
| 4 | large | grad_sum_solve_backward | 64x64,rhs=1 | 703.010 | 715.728 | 151.535 | 0.98 | 4.72 |
| 4 | large | grad_sum_matmul | 128x128 | 123.290 | 124.150 | 87.708 | 0.99 | 1.42 |
| 4 | large | grad_sum_matmul_backward | 128x128 | 632.319 | 654.929 | 307.143 | 0.97 | 2.13 |
| 4 | large | grad_sum_svd_s_backward | 128x128 | 4838.685 | 4377.057 | 2616.225 | 1.11 | 1.67 |
| 4 | large | grad_sum_solve_backward | 128x128,rhs=1 | 1482.458 | 1275.381 | 284.326 | 1.16 | 4.49 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 2x2xbatch16 (native batch layout) | 9.013 | 8.504 | 13.757 | 1.06 | 0.62 |
| 4 | batched | batched_svd | 2x2xbatch16 (native batch layout) | 85.987 | 79.982 | 26.779 | 1.08 | 2.99 |
| 4 | batched | batched_qr | 2x2xbatch16 (native batch layout) | 53.211 | 14.719 | 10.489 | 3.62 | 1.40 |
| 4 | batched | batched_eigh | 2x2xbatch16 (native batch layout) | 55.860 | 55.472 | 14.098 | 1.01 | 3.93 |
| 4 | batched | batched_solve | 2x2xbatch16 (native batch layout),rhs=1 | 48.619 | 15.502 | 20.490 | 3.14 | 0.76 |
| 4 | batched | grad_sum_batched_matmul_backward | 2x2xbatch16 (native batch layout) | 317.733 | 329.073 | 141.771 | 0.97 | 2.32 |
| 4 | batched | grad_sum_batched_solve_backward | 2x2xbatch16 (native batch layout),rhs=1 | 557.649 | 433.385 | 136.068 | 1.29 | 3.19 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 4x4xbatch16 (native batch layout) | 8.807 | 9.507 | 16.171 | 0.93 | 0.59 |
| 4 | batched | batched_svd | 4x4xbatch16 (native batch layout) | 129.628 | 128.018 | 74.852 | 1.01 | 1.71 |
| 4 | batched | batched_qr | 4x4xbatch16 (native batch layout) | 61.267 | 20.605 | 15.608 | 2.97 | 1.32 |
| 4 | batched | batched_eigh | 4x4xbatch16 (native batch layout) | 78.199 | 77.080 | 39.506 | 1.01 | 1.95 |
| 4 | batched | batched_solve | 4x4xbatch16 (native batch layout),rhs=1 | 50.284 | 19.458 | 22.247 | 2.58 | 0.87 |
| 4 | batched | grad_sum_batched_matmul_backward | 4x4xbatch16 (native batch layout) | 327.519 | 329.846 | 146.931 | 0.99 | 2.24 |
| 4 | batched | grad_sum_batched_solve_backward | 4x4xbatch16 (native batch layout),rhs=1 | 595.162 | 454.487 | 139.266 | 1.31 | 3.26 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 8x8xbatch16 (native batch layout) | 11.014 | 9.871 | 17.678 | 1.12 | 0.56 |
| 4 | batched | batched_svd | 8x8xbatch16 (native batch layout) | 278.108 | 302.731 | 224.874 | 0.92 | 1.35 |
| 4 | batched | batched_qr | 8x8xbatch16 (native batch layout) | 69.997 | 32.179 | 29.159 | 2.18 | 1.10 |
| 4 | batched | batched_eigh | 8x8xbatch16 (native batch layout) | 128.024 | 142.732 | 101.013 | 0.90 | 1.41 |
| 4 | batched | batched_solve | 8x8xbatch16 (native batch layout),rhs=1 | 61.559 | 22.438 | 24.718 | 2.74 | 0.91 |
| 4 | batched | grad_sum_batched_matmul_backward | 8x8xbatch16 (native batch layout) | 340.628 | 347.172 | 253.800 | 0.98 | 1.37 |
| 4 | batched | grad_sum_batched_solve_backward | 8x8xbatch16 (native batch layout),rhs=1 | 632.593 | 502.780 | 150.151 | 1.26 | 3.35 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 16x16xbatch16 (native batch layout) | 19.385 | 22.107 | 21.652 | 0.88 | 1.02 |
| 4 | batched | batched_svd | 16x16xbatch16 (native batch layout) | 621.566 | 609.479 | 711.188 | 1.02 | 0.86 |
| 4 | batched | batched_qr | 16x16xbatch16 (native batch layout) | 119.313 | 82.005 | 85.159 | 1.45 | 0.96 |
| 4 | batched | batched_eigh | 16x16xbatch16 (native batch layout) | 309.099 | 348.523 | 330.313 | 0.89 | 1.06 |
| 4 | batched | batched_solve | 16x16xbatch16 (native batch layout),rhs=1 | 87.085 | 35.070 | 34.956 | 2.48 | 1.00 |
| 4 | batched | grad_sum_batched_matmul_backward | 16x16xbatch16 (native batch layout) | 396.752 | 455.672 | 288.663 | 0.87 | 1.58 |
| 4 | batched | grad_sum_batched_solve_backward | 16x16xbatch16 (native batch layout),rhs=1 | 824.222 | 533.418 | 181.273 | 1.55 | 2.94 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 2x2xbatch64 (native batch layout) | 14.270 | 16.630 | 15.746 | 0.86 | 1.06 |
| 4 | batched | batched_svd | 2x2xbatch64 (native batch layout) | 276.707 | 279.769 | 75.957 | 0.99 | 3.68 |
| 4 | batched | batched_qr | 2x2xbatch64 (native batch layout) | 166.206 | 27.197 | 19.737 | 6.11 | 1.38 |
| 4 | batched | batched_eigh | 2x2xbatch64 (native batch layout) | 185.149 | 180.785 | 39.657 | 1.02 | 4.56 |
| 4 | batched | batched_solve | 2x2xbatch64 (native batch layout),rhs=1 | 144.922 | 18.532 | 24.927 | 7.82 | 0.74 |
| 4 | batched | grad_sum_batched_matmul_backward | 2x2xbatch64 (native batch layout) | 357.409 | 353.376 | 129.352 | 1.01 | 2.73 |
| 4 | batched | grad_sum_batched_solve_backward | 2x2xbatch64 (native batch layout),rhs=1 | 947.836 | 495.568 | 144.118 | 1.91 | 3.44 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 4x4xbatch64 (native batch layout) | 16.536 | 17.313 | 21.453 | 0.96 | 0.81 |
| 4 | batched | batched_svd | 4x4xbatch64 (native batch layout) | 456.131 | 452.316 | 239.073 | 1.01 | 1.89 |
| 4 | batched | batched_qr | 4x4xbatch64 (native batch layout) | 188.476 | 50.719 | 40.023 | 3.72 | 1.27 |
| 4 | batched | batched_eigh | 4x4xbatch64 (native batch layout) | 262.457 | 286.498 | 111.274 | 0.92 | 2.57 |
| 4 | batched | batched_solve | 4x4xbatch64 (native batch layout),rhs=1 | 150.579 | 21.593 | 27.041 | 6.97 | 0.80 |
| 4 | batched | grad_sum_batched_matmul_backward | 4x4xbatch64 (native batch layout) | 360.301 | 403.183 | 172.187 | 0.89 | 2.34 |
| 4 | batched | grad_sum_batched_solve_backward | 4x4xbatch64 (native batch layout),rhs=1 | 982.375 | 525.514 | 152.194 | 1.87 | 3.45 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 8x8xbatch64 (native batch layout) | 20.654 | 20.944 | 21.000 | 0.99 | 1.00 |
| 4 | batched | batched_svd | 8x8xbatch64 (native batch layout) | 963.220 | 960.816 | 870.540 | 1.00 | 1.10 |
| 4 | batched | batched_qr | 8x8xbatch64 (native batch layout) | 233.407 | 89.917 | 96.250 | 2.60 | 0.93 |
| 4 | batched | batched_eigh | 8x8xbatch64 (native batch layout) | 465.340 | 492.858 | 380.019 | 0.94 | 1.30 |
| 4 | batched | batched_solve | 8x8xbatch64 (native batch layout),rhs=1 | 204.543 | 43.362 | 38.543 | 4.72 | 1.13 |
| 4 | batched | grad_sum_batched_matmul_backward | 8x8xbatch64 (native batch layout) | 407.060 | 404.809 | 533.129 | 1.01 | 0.76 |
| 4 | batched | grad_sum_batched_solve_backward | 8x8xbatch64 (native batch layout),rhs=1 | 1158.708 | 569.543 | 191.699 | 2.03 | 2.97 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 16x16xbatch64 (native batch layout) | 56.054 | 54.242 | 36.591 | 1.03 | 1.48 |
| 4 | batched | batched_svd | 16x16xbatch64 (native batch layout) | 2393.591 | 2853.595 | 2466.561 | 0.84 | 1.16 |
| 4 | batched | batched_qr | 16x16xbatch64 (native batch layout) | 442.585 | 247.021 | 318.665 | 1.79 | 0.78 |
| 4 | batched | batched_eigh | 16x16xbatch64 (native batch layout) | 1205.651 | 1206.802 | 1333.164 | 1.00 | 0.91 |
| 4 | batched | batched_solve | 16x16xbatch64 (native batch layout),rhs=1 | 302.463 | 82.518 | 77.286 | 3.67 | 1.07 |
| 4 | batched | grad_sum_batched_matmul_backward | 16x16xbatch64 (native batch layout) | 540.116 | 575.814 | 661.767 | 0.94 | 0.87 |
| 4 | batched | grad_sum_batched_solve_backward | 16x16xbatch64 (native batch layout),rhs=1 | 1696.744 | 875.331 | 296.237 | 1.94 | 2.95 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 2x2xbatch256 (native batch layout) | 39.188 | 44.105 | 18.228 | 0.89 | 2.42 |
| 4 | batched | batched_svd | 2x2xbatch256 (native batch layout) | 1077.255 | 1157.688 | 251.363 | 0.93 | 4.61 |
| 4 | batched | batched_qr | 2x2xbatch256 (native batch layout) | 623.974 | 77.287 | 57.725 | 8.07 | 1.34 |
| 4 | batched | batched_eigh | 2x2xbatch256 (native batch layout) | 692.705 | 712.543 | 134.569 | 0.97 | 5.29 |
| 4 | batched | batched_solve | 2x2xbatch256 (native batch layout),rhs=1 | 521.083 | 30.358 | 37.874 | 17.16 | 0.80 |
| 4 | batched | grad_sum_batched_matmul_backward | 2x2xbatch256 (native batch layout) | 444.625 | 434.635 | 147.755 | 1.02 | 2.94 |
| 4 | batched | grad_sum_batched_solve_backward | 2x2xbatch256 (native batch layout),rhs=1 | 2489.938 | 530.463 | 170.379 | 4.69 | 3.11 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 4x4xbatch256 (native batch layout) | 42.299 | 43.404 | 54.531 | 0.97 | 0.80 |
| 4 | batched | batched_svd | 4x4xbatch256 (native batch layout) | 1759.411 | 1918.877 | 945.862 | 0.92 | 2.03 |
| 4 | batched | batched_qr | 4x4xbatch256 (native batch layout) | 759.194 | 156.700 | 137.428 | 4.84 | 1.14 |
| 4 | batched | batched_eigh | 4x4xbatch256 (native batch layout) | 1002.736 | 1044.020 | 442.010 | 0.96 | 2.36 |
| 4 | batched | batched_solve | 4x4xbatch256 (native batch layout),rhs=1 | 569.548 | 42.407 | 45.333 | 13.43 | 0.94 |
| 4 | batched | grad_sum_batched_matmul_backward | 4x4xbatch256 (native batch layout) | 513.285 | 545.949 | 243.696 | 0.94 | 2.24 |
| 4 | batched | grad_sum_batched_solve_backward | 4x4xbatch256 (native batch layout),rhs=1 | 2783.691 | 675.964 | 198.989 | 4.12 | 3.40 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 8x8xbatch256 (native batch layout) | 63.123 | 70.340 | 37.091 | 0.90 | 1.90 |
| 4 | batched | batched_svd | 8x8xbatch256 (native batch layout) | 3956.868 | 4022.914 | 3015.450 | 0.98 | 1.33 |
| 4 | batched | batched_qr | 8x8xbatch256 (native batch layout) | 932.526 | 323.217 | 356.970 | 2.89 | 0.91 |
| 4 | batched | batched_eigh | 8x8xbatch256 (native batch layout) | 1830.431 | 1805.667 | 1248.696 | 1.01 | 1.45 |
| 4 | batched | batched_solve | 8x8xbatch256 (native batch layout),rhs=1 | 681.655 | 115.005 | 92.476 | 5.93 | 1.24 |
| 4 | batched | grad_sum_batched_matmul_backward | 8x8xbatch256 (native batch layout) | 697.492 | 641.494 | 1629.917 | 1.09 | 0.39 |
| 4 | batched | grad_sum_batched_solve_backward | 8x8xbatch256 (native batch layout),rhs=1 | 3821.683 | 942.414 | 354.616 | 4.06 | 2.66 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 16x16xbatch256 (native batch layout) | 241.252 | 234.204 | 89.210 | 1.03 | 2.63 |
| 4 | batched | batched_svd | 16x16xbatch256 (native batch layout) | 10041.563 | 10614.451 | 11360.943 | 0.95 | 0.93 |
| 4 | batched | batched_qr | 16x16xbatch256 (native batch layout) | 1697.999 | 1136.794 | 1218.328 | 1.49 | 0.93 |
| 4 | batched | batched_eigh | 16x16xbatch256 (native batch layout) | 5144.116 | 5456.356 | 5292.943 | 0.94 | 1.03 |
| 4 | batched | batched_solve | 16x16xbatch256 (native batch layout),rhs=1 | 1167.301 | 348.885 | 212.703 | 3.35 | 1.64 |
| 4 | batched | grad_sum_batched_matmul_backward | 16x16xbatch256 (native batch layout) | 1400.351 | 1366.914 | 2147.337 | 1.02 | 0.64 |
| 4 | batched | grad_sum_batched_solve_backward | 16x16xbatch256 (native batch layout),rhs=1 | 6088.610 | 2063.524 | 568.377 | 2.95 | 3.63 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 2x2xbatch1024 (native batch layout) | 138.495 | 169.374 | 41.524 | 0.82 | 4.08 |
| 4 | batched | batched_svd | 2x2xbatch1024 (native batch layout) | 4377.618 | 4245.198 | 874.560 | 1.03 | 4.85 |
| 4 | batched | batched_qr | 2x2xbatch1024 (native batch layout) | 2639.785 | 260.769 | 210.976 | 10.12 | 1.24 |
| 4 | batched | batched_eigh | 2x2xbatch1024 (native batch layout) | 2717.878 | 2679.411 | 473.014 | 1.01 | 5.66 |
| 4 | batched | batched_solve | 2x2xbatch1024 (native batch layout),rhs=1 | 2128.287 | 74.987 | 80.645 | 28.38 | 0.93 |
| 4 | batched | grad_sum_batched_matmul_backward | 2x2xbatch1024 (native batch layout) | 809.424 | 789.478 | 204.959 | 1.03 | 3.85 |
| 4 | batched | grad_sum_batched_solve_backward | 2x2xbatch1024 (native batch layout),rhs=1 | 8844.470 | 867.427 | 260.327 | 10.20 | 3.33 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 4x4xbatch1024 (native batch layout) | 150.304 | 170.875 | 71.571 | 0.88 | 2.39 |
| 4 | batched | batched_svd | 4x4xbatch1024 (native batch layout) | 6960.892 | 7429.427 | 3758.649 | 0.94 | 1.98 |
| 4 | batched | batched_qr | 4x4xbatch1024 (native batch layout) | 2787.391 | 622.399 | 523.384 | 4.48 | 1.19 |
| 4 | batched | batched_eigh | 4x4xbatch1024 (native batch layout) | 3932.526 | 4322.380 | 1667.543 | 0.91 | 2.59 |
| 4 | batched | batched_solve | 4x4xbatch1024 (native batch layout),rhs=1 | 2128.316 | 130.560 | 117.476 | 16.30 | 1.11 |
| 4 | batched | grad_sum_batched_matmul_backward | 4x4xbatch1024 (native batch layout) | 849.198 | 868.776 | 308.301 | 0.98 | 2.82 |
| 4 | batched | grad_sum_batched_solve_backward | 4x4xbatch1024 (native batch layout),rhs=1 | 9414.390 | 1242.775 | 376.534 | 7.58 | 3.30 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 8x8xbatch1024 (native batch layout) | 236.693 | 282.589 | 82.464 | 0.84 | 3.43 |
| 4 | batched | batched_svd | 8x8xbatch1024 (native batch layout) | 15339.864 | 17531.069 | 13635.289 | 0.88 | 1.29 |
| 4 | batched | batched_qr | 8x8xbatch1024 (native batch layout) | 4116.194 | 1419.988 | 1378.360 | 2.90 | 1.03 |
| 4 | batched | batched_eigh | 8x8xbatch1024 (native batch layout) | 7584.823 | 7462.238 | 5897.523 | 1.02 | 1.27 |
| 4 | batched | batched_solve | 8x8xbatch1024 (native batch layout),rhs=1 | 3077.121 | 443.819 | 281.515 | 6.93 | 1.58 |
| 4 | batched | grad_sum_batched_matmul_backward | 8x8xbatch1024 (native batch layout) | 1569.039 | 1551.131 | 6046.258 | 1.01 | 0.26 |
| 4 | batched | grad_sum_batched_solve_backward | 8x8xbatch1024 (native batch layout),rhs=1 | 12130.288 | 2547.671 | 779.763 | 4.76 | 3.27 |
| 4 | batched | batched_matmul_ikb_kjb_ijb | 16x16xbatch1024 (native batch layout) | 937.153 | 921.353 | 319.971 | 1.02 | 2.88 |
| 4 | batched | batched_svd | 16x16xbatch1024 (native batch layout) | 41086.173 | 40910.207 | 42527.433 | 1.00 | 0.96 |
| 4 | batched | batched_qr | 16x16xbatch1024 (native batch layout) | 6650.550 | 3795.626 | 4774.984 | 1.75 | 0.79 |
| 4 | batched | batched_eigh | 16x16xbatch1024 (native batch layout) | 19717.872 | 19504.020 | 20322.239 | 1.01 | 0.96 |
| 4 | batched | batched_solve | 16x16xbatch1024 (native batch layout),rhs=1 | 4594.235 | 1330.078 | 774.956 | 3.45 | 1.72 |
| 4 | batched | grad_sum_batched_matmul_backward | 16x16xbatch1024 (native batch layout) | 6343.459 | 3964.262 | 8565.114 | 1.60 | 0.46 |
| 4 | batched | grad_sum_batched_solve_backward | 16x16xbatch1024 (native batch layout),rhs=1 | 19648.758 | 6394.609 | 1605.815 | 3.07 | 3.98 |

## Einsum: execution-scope correction

Same 19 manifest instances and both paths, 3 warmups / 7 samples. Old timings include per-contraction executor entry; shared timings enter once outside sampling. These are distinct boundaries, not a LAPACK library speedup.

| T | path | instance | old ms | shared ms | Torch ms | shared/Torch |
|---:|---|---|---:|---:|---:|---:|
| 1 | opt_flops | bin_batched_matmul_b32_m128_n128_k128 | 3.426 | 4.072 | 3.522 | 1.16 |
| 1 | opt_flops | bin_batched_matmul_b32_m64_n64_k64 | 0.548 | 0.613 | 0.575 | 1.07 |
| 1 | opt_flops | bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.256 | 0.219 | 0.207 | 1.06 |
| 1 | opt_flops | bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.246 | 0.218 | 0.174 | 1.25 |
| 1 | opt_flops | bin_elementwise_mul_2048x2048 | 15.275 | 15.701 | 18.657 | 0.84 |
| 1 | opt_flops | bin_matmul_1024 | 46.072 | 45.792 | 54.231 | 0.84 |
| 1 | opt_flops | bin_matmul_256 | 0.755 | 0.733 | 1.229 | 0.60 |
| 1 | opt_flops | bin_outer_product_4096 | 63.246 | 59.740 | 63.020 | 0.95 |
| 1 | opt_flops | gm_queen5_5_3.wcsp | 6846.891 | 7024.456 | 6389.978 | 1.10 |
| 1 | opt_flops | lm_batch_likelihood_brackets_4_4d | 32.997 | 31.554 | 33.217 | 0.95 |
| 1 | opt_flops | lm_batch_likelihood_sentence_3_12d | 132.514 | 99.681 | 92.139 | 1.08 |
| 1 | opt_flops | lm_batch_likelihood_sentence_4_4d | 37.466 | 34.977 | 33.210 | 1.05 |
| 1 | opt_flops | nary_matmul_chain_64 | 0.085 | 0.057 | 0.066 | 0.86 |
| 1 | opt_flops | str_matrix_chain_multiplication_100 | 13.684 | 11.357 | 15.635 | 0.73 |
| 1 | opt_flops | str_mps_varying_inner_product_200 | 21.491 | 18.421 | 23.006 | 0.80 |
| 1 | opt_flops | str_nw_mera_closed_120 | 2078.003 | 2127.761 | 1931.621 | 1.10 |
| 1 | opt_flops | str_nw_mera_open_26 | 1666.504 | 1626.777 | 1227.079 | 1.33 |
| 1 | opt_flops | tensornetwork_permutation_focus_step409_316 | 665.438 | 610.533 | 899.313 | 0.68 |
| 1 | opt_flops | tensornetwork_permutation_light_415 | 590.870 | 537.882 | 855.359 | 0.63 |
| 1 | opt_size | bin_batched_matmul_b32_m128_n128_k128 | 3.426 | 4.072 | 3.522 | 1.16 |
| 1 | opt_size | bin_batched_matmul_b32_m64_n64_k64 | 0.548 | 0.613 | 0.575 | 1.07 |
| 1 | opt_size | bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.256 | 0.219 | 0.207 | 1.06 |
| 1 | opt_size | bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.246 | 0.218 | 0.174 | 1.25 |
| 1 | opt_size | bin_elementwise_mul_2048x2048 | 15.275 | 15.701 | 18.657 | 0.84 |
| 1 | opt_size | bin_matmul_1024 | 46.072 | 45.792 | 54.231 | 0.84 |
| 1 | opt_size | bin_matmul_256 | 0.755 | 0.733 | 1.229 | 0.60 |
| 1 | opt_size | bin_outer_product_4096 | 63.246 | 59.740 | 63.020 | 0.95 |
| 1 | opt_size | gm_queen5_5_3.wcsp | 2534.629 | 2510.839 | 2460.176 | 1.02 |
| 1 | opt_size | lm_batch_likelihood_brackets_4_4d | 34.627 | 32.122 | 30.990 | 1.04 |
| 1 | opt_size | lm_batch_likelihood_sentence_3_12d | 149.741 | 87.612 | 60.871 | 1.44 |
| 1 | opt_size | lm_batch_likelihood_sentence_4_4d | 38.020 | 36.330 | 33.651 | 1.08 |
| 1 | opt_size | nary_matmul_chain_64 | 0.085 | 0.057 | 0.066 | 0.86 |
| 1 | opt_size | str_matrix_chain_multiplication_100 | 16.240 | 14.367 | 14.809 | 0.97 |
| 1 | opt_size | str_mps_varying_inner_product_200 | 24.362 | 20.886 | 23.205 | 0.90 |
| 1 | opt_size | str_nw_mera_closed_120 | 1560.555 | 1615.148 | 1511.712 | 1.07 |
| 1 | opt_size | str_nw_mera_open_26 | 1647.241 | 1671.251 | 1233.652 | 1.35 |
| 1 | opt_size | tensornetwork_permutation_focus_step409_316 | 665.438 | 610.533 | 899.313 | 0.68 |
| 1 | opt_size | tensornetwork_permutation_light_415 | 590.870 | 537.882 | 855.359 | 0.63 |
| 4 | opt_flops | bin_batched_matmul_b32_m128_n128_k128 | 4.563 | 1.669 | 1.346 | 1.24 |
| 4 | opt_flops | bin_batched_matmul_b32_m64_n64_k64 | 0.694 | 0.466 | 0.222 | 2.10 |
| 4 | opt_flops | bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.612 | 0.293 | 0.202 | 1.45 |
| 4 | opt_flops | bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.727 | 0.316 | 0.190 | 1.66 |
| 4 | opt_flops | bin_elementwise_mul_2048x2048 | 5.223 | 5.676 | 5.586 | 1.02 |
| 4 | opt_flops | bin_matmul_1024 | 13.701 | 13.960 | 26.307 | 0.53 |
| 4 | opt_flops | bin_matmul_256 | 0.468 | 0.276 | 0.799 | 0.35 |
| 4 | opt_flops | bin_outer_product_4096 | 19.196 | 19.125 | 19.067 | 1.00 |
| 4 | opt_flops | gm_queen5_5_3.wcsp | 6208.222 | 3641.569 | 2451.108 | 1.49 |
| 4 | opt_flops | lm_batch_likelihood_brackets_4_4d | 75.094 | 28.842 | 10.405 | 2.77 |
| 4 | opt_flops | lm_batch_likelihood_sentence_3_12d | 146.923 | 91.332 | 37.279 | 2.45 |
| 4 | opt_flops | lm_batch_likelihood_sentence_4_4d | 83.563 | 35.835 | 10.689 | 3.35 |
| 4 | opt_flops | nary_matmul_chain_64 | 0.495 | 0.045 | 0.067 | 0.67 |
| 4 | opt_flops | str_matrix_chain_multiplication_100 | 23.118 | 6.322 | 6.721 | 0.94 |
| 4 | opt_flops | str_mps_varying_inner_product_200 | 62.917 | 19.313 | 18.023 | 1.07 |
| 4 | opt_flops | str_nw_mera_closed_120 | 1129.078 | 731.145 | 700.956 | 1.04 |
| 4 | opt_flops | str_nw_mera_open_26 | 1191.062 | 750.114 | 419.182 | 1.79 |
| 4 | opt_flops | tensornetwork_permutation_focus_step409_316 | 897.882 | 383.605 | 296.604 | 1.29 |
| 4 | opt_flops | tensornetwork_permutation_light_415 | 854.345 | 330.765 | 286.997 | 1.15 |
| 4 | opt_size | bin_batched_matmul_b32_m128_n128_k128 | 4.563 | 1.669 | 1.346 | 1.24 |
| 4 | opt_size | bin_batched_matmul_b32_m64_n64_k64 | 0.694 | 0.466 | 0.222 | 2.10 |
| 4 | opt_size | bin_batched_outer_product_compact_j16_k16_o64_t64 | 0.612 | 0.293 | 0.202 | 1.45 |
| 4 | opt_size | bin_batched_outer_product_noncompact_j16_k16_o64_t64 | 0.727 | 0.316 | 0.190 | 1.66 |
| 4 | opt_size | bin_elementwise_mul_2048x2048 | 5.223 | 5.676 | 5.586 | 1.02 |
| 4 | opt_size | bin_matmul_1024 | 13.701 | 13.960 | 26.307 | 0.53 |
| 4 | opt_size | bin_matmul_256 | 0.468 | 0.276 | 0.799 | 0.35 |
| 4 | opt_size | bin_outer_product_4096 | 19.196 | 19.125 | 19.067 | 1.00 |
| 4 | opt_size | gm_queen5_5_3.wcsp | 2492.950 | 1264.832 | 940.420 | 1.34 |
| 4 | opt_size | lm_batch_likelihood_brackets_4_4d | 81.188 | 29.984 | 11.033 | 2.72 |
| 4 | opt_size | lm_batch_likelihood_sentence_3_12d | 179.733 | 50.065 | 19.910 | 2.51 |
| 4 | opt_size | lm_batch_likelihood_sentence_4_4d | 82.676 | 32.290 | 11.438 | 2.82 |
| 4 | opt_size | nary_matmul_chain_64 | 0.495 | 0.045 | 0.067 | 0.67 |
| 4 | opt_size | str_matrix_chain_multiplication_100 | 26.113 | 6.296 | 6.699 | 0.94 |
| 4 | opt_size | str_mps_varying_inner_product_200 | 68.627 | 22.422 | 19.462 | 1.15 |
| 4 | opt_size | str_nw_mera_closed_120 | 864.757 | 553.250 | 495.063 | 1.12 |
| 4 | opt_size | str_nw_mera_open_26 | 1189.818 | 771.380 | 396.985 | 1.94 |
| 4 | opt_size | tensornetwork_permutation_focus_step409_316 | 897.882 | 383.605 | 296.604 | 1.29 |
| 4 | opt_size | tensornetwork_permutation_light_415 | 854.345 | 330.765 | 286.997 | 1.15 |

## Interpretation / provenance

- Library fixes remove repeated LU/QR scratch allocation, redundant copies and workspace queries. All 146 linalg unit tests and 13 release batched tests passed.
- Residual small/batched eigh and SVD, eager backward graph setup, 4T language-model contraction paths, and single-transform RustFFT threading are not solved here; no parity target or speculative replacement kernels.
- Full FFT CSVs are in the archive. Short eager FFT rows are skipped by existing policy. At n=65536, no 1T eager deficit; 4T C32 complex FFT/IFFT is about 2.6x slower than Torch, while the F64/C64 variants are faster.
- Initial 4T CPU ops (`paired-baseline-t4`, `candidate-t4`) are **misplacement diagnostics only**. Correctly placed comparison uses `placed-*`. Count=4 alone did not prove use of four cores; `/proc` task masks were inspected. Explicit placement gave 1024 GEMM 116→14.6 ms in a separate diagnostic.
- QR-family repeat addresses the suspicious initial large-QR difference; full tables use the original complete paired runs, not per-row best-of selection. Host noise is accepted.
- [Raw records, scripts and protocol](eager-1820/raw-data.tar.gz) (binaries excluded); [SHA256](eager-1820/SHA256SUMS). Extract beside this report for inspection. The archive retains original directory names and source/runtime metadata.
- Source PR: https://github.com/tensor4all/tenferro-rs/pull/1821. Report generated by archived `summarize.py`; no published historical tables were overwritten.
