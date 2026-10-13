# gpu/nn: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/nn/20261013_101308/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| sigmoid_f32_n1024 | unsupported | unsupported | 8.973 (0.113) | 77.895 (2.301) | 78.901 (2.485) |
| sigmoid_f64_n1024 | unsupported | unsupported | 9.821 (0.421) | 92.013 (8.645) | 84.943 (6.892) |
| silu_f32_n1024 | unsupported | unsupported | 10.299 (0.149) | 90.238 (9.924) | 86.604 (9.994) |
| silu_f64_n1024 | unsupported | unsupported | 10.487 (0.342) | 94.684 (22.797) | 93.498 (10.870) |
| softplus_f32_n1024 | 110.959 (0.792) | 368.698 (12.203) | 9.564 (0.111) | 98.584 (18.946) | 94.463 (14.895) |
| softplus_f64_n1024 | 112.909 (8.758) | 396.181 (8.955) | 26.038 (0.061) | 110.582 (23.065) | 127.039 (8.948) |
| gelu_f32_n1024 | unsupported | unsupported | 10.604 (0.072) | 1719.042 (192.023) | 155.602 (26.059) |
| gelu_f64_n1024 | unsupported | unsupported | 17.013 (0.011) | 1675.726 (206.678) | 149.170 (11.218) |
| gelu_tanh_f32_n1024 | unsupported | unsupported | 10.554 (0.229) | 2673.288 (34.633) | 143.540 (16.187) |
| gelu_tanh_f64_n1024 | unsupported | unsupported | 10.823 (0.149) | 2678.185 (57.063) | 131.943 (5.853) |
| softmax_f32_n1024 | unsupported | unsupported | 10.477 (0.145) | 1263.574 (26.984) | 93.352 (0.596) |
| softmax_f64_n1024 | unsupported | unsupported | 10.035 (0.212) | 1292.280 (19.724) | 93.005 (0.538) |
| log_softmax_f32_n1024 | unsupported | unsupported | 10.757 (0.172) | 91.236 (2.351) | 92.902 (0.766) |
| log_softmax_f64_n1024 | unsupported | unsupported | 10.063 (0.218) | 90.469 (1.364) | 93.137 (0.762) |
| masked_softmax_f32_n1024 | unsupported | unsupported | unsupported | 2123.412 (44.851) | 95.807 (0.618) |
| masked_softmax_f64_n1024 | unsupported | unsupported | unsupported | 2107.985 (66.890) | 95.794 (0.745) |
| masked_log_softmax_f32_n1024 | unsupported | unsupported | unsupported | 92.569 (3.423) | 96.239 (0.920) |
| masked_log_softmax_f64_n1024 | unsupported | unsupported | unsupported | 92.683 (0.768) | 96.201 (0.317) |
| layer_norm_f32_n1024 | unsupported | unsupported | 21.474 (1.175) | 285.086 (20.745) | 104.303 (9.286) |
| layer_norm_f64_n1024 | unsupported | unsupported | 23.232 (0.647) | 360.683 (15.062) | 142.730 (11.510) |
| rms_norm_f32_n1024 | unsupported | unsupported | 21.445 (0.208) | unsupported | unsupported |
| rms_norm_f64_n1024 | unsupported | unsupported | 24.000 (0.513) | unsupported | unsupported |
| sigmoid_f32_n1048576 | unsupported | unsupported | 27.590 (0.071) | 94.606 (4.473) | 96.161 (0.561) |
| sigmoid_f64_n1048576 | unsupported | unsupported | 258.690 (0.193) | 305.395 (1.420) | 303.457 (4.027) |
| silu_f32_n1048576 | unsupported | unsupported | 27.529 (0.028) | 93.584 (8.168) | 95.979 (0.779) |
| silu_f64_n1048576 | unsupported | unsupported | 293.503 (0.227) | 317.447 (1.361) | 316.105 (2.452) |
| softplus_f32_n1048576 | 273.976 (1.503) | 310.936 (2.930) | 28.352 (0.193) | 93.054 (9.143) | 88.784 (6.877) |
| softplus_f64_n1048576 | 1261.139 (11.163) | 1260.996 (0.861) | 898.837 (0.449) | 1100.847 (16.202) | 1094.080 (7.473) |
| gelu_f32_n1048576 | unsupported | unsupported | 27.450 (0.039) | 1114.370 (17.534) | 99.497 (5.126) |
| gelu_f64_n1048576 | unsupported | unsupported | 575.457 (0.229) | 2964.210 (118.052) | 1751.634 (13.042) |
| gelu_tanh_f32_n1048576 | unsupported | unsupported | 27.443 (0.042) | 1696.221 (42.337) | 97.157 (5.685) |
| gelu_tanh_f64_n1048576 | unsupported | unsupported | 250.526 (0.118) | 1678.639 (20.637) | 307.973 (4.212) |
| softmax_f32_n1048576 | unsupported | unsupported | 27.679 (0.129) | 835.732 (101.598) | 93.713 (0.786) |
| softmax_f64_n1048576 | unsupported | unsupported | 338.941 (0.485) | 767.594 (82.080) | 441.945 (5.734) |
| log_softmax_f32_n1048576 | unsupported | unsupported | 28.227 (0.048) | 82.364 (0.548) | 93.493 (0.480) |
| log_softmax_f64_n1048576 | unsupported | unsupported | 292.342 (1.743) | 389.531 (2.032) | 401.893 (8.772) |
| masked_softmax_f32_n1048576 | unsupported | unsupported | unsupported | 1073.433 (19.180) | 96.255 (2.326) |
| masked_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 1259.480 (28.536) | 672.631 (10.388) |
| masked_log_softmax_f32_n1048576 | unsupported | unsupported | unsupported | 91.003 (4.013) | 96.540 (0.972) |
| masked_log_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 385.088 (2.344) | 392.838 (11.580) |
| layer_norm_f32_n1048576 | unsupported | unsupported | 34.355 (0.077) | 189.365 (7.691) | 119.164 (11.141) |
| layer_norm_f64_n1048576 | unsupported | unsupported | 709.192 (0.213) | 200.854 (5.651) | 119.307 (1.503) |
| rms_norm_f32_n1048576 | unsupported | unsupported | 33.681 (0.086) | unsupported | unsupported |
| rms_norm_f64_n1048576 | unsupported | unsupported | 709.420 (0.208) | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| sigmoid_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softplus_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 3.551 | 8.073e-05 |
| softplus_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 3.613 | 1.428e-07 |
| gelu_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softplus_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.192 | 0.0001273 |
| softplus_f64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.522 | 1.467e-07 |
| gelu_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softplus_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.950 | 8.073e-05 |
| softplus_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.169 | 1.428e-07 |
| gelu_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softplus_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.487 | 0.0001273 |
| softplus_f64_n1048576 | tenferro-cuda-eager | 2 | 15 | 2.522 | 1.467e-07 |
| gelu_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f32_n1024 | pytorch-cuda | 256 | 15 | 2.297 | 8.389e-05 |
| sigmoid_f64_n1024 | pytorch-cuda | 256 | 15 | 2.514 | 7.964e-08 |
| silu_f32_n1024 | pytorch-cuda | 256 | 15 | 2.636 | 4.766e-05 |
| silu_f64_n1024 | pytorch-cuda | 256 | 15 | 2.685 | 4.848e-08 |
| softplus_f32_n1024 | pytorch-cuda | 256 | 15 | 2.448 | 8.073e-05 |
| softplus_f64_n1024 | pytorch-cuda | 128 | 15 | 3.333 | 1.428e-07 |
| gelu_f32_n1024 | pytorch-cuda | 256 | 15 | 2.715 | 4.754e-05 |
| gelu_f64_n1024 | pytorch-cuda | 128 | 15 | 2.178 | 4.899e-08 |
| gelu_tanh_f32_n1024 | pytorch-cuda | 256 | 15 | 2.702 | 4.766e-05 |
| gelu_tanh_f64_n1024 | pytorch-cuda | 256 | 15 | 2.771 | 4.349e-08 |
| softmax_f32_n1024 | pytorch-cuda | 256 | 15 | 2.682 | 1.435e-05 |
| softmax_f64_n1024 | pytorch-cuda | 256 | 15 | 2.569 | 2.012e-08 |
| log_softmax_f32_n1024 | pytorch-cuda | 256 | 15 | 2.754 | 0.0001185 |
| log_softmax_f64_n1024 | pytorch-cuda | 256 | 15 | 2.576 | 2.068e-07 |
| masked_softmax_f32_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f32_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| layer_norm_f32_n1024 | pytorch-cuda | 128 | 15 | 2.749 | 0.000193 |
| layer_norm_f64_n1024 | pytorch-cuda | 128 | 15 | 2.974 | 1.757e-07 |
| rms_norm_f32_n1024 | pytorch-cuda | 128 | 15 | 2.745 | 0.0001828 |
| rms_norm_f64_n1024 | pytorch-cuda | 128 | 15 | 3.072 | 2.032e-07 |
| sigmoid_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.766 | 8.57e-05 |
| sigmoid_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.070 | 7.964e-08 |
| silu_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.762 | 4.781e-05 |
| silu_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.348 | 4.848e-08 |
| softplus_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.815 | 0.0001261 |
| softplus_f64_n1048576 | pytorch-cuda | 4 | 15 | 3.595 | 1.467e-07 |
| gelu_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.757 | 4.754e-05 |
| gelu_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.302 | 4.899e-08 |
| gelu_tanh_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.756 | 5.198e-05 |
| gelu_tanh_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.004 | 4.349e-08 |
| softmax_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.771 | 6.976e-07 |
| softmax_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.712 | 8.661e-10 |
| log_softmax_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.807 | 0.0001277 |
| log_softmax_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.339 | 2.377e-07 |
| masked_softmax_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| layer_norm_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.199 | 0.0002616 |
| layer_norm_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.837 | 3.138e-07 |
| rms_norm_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.156 | 0.000212 |
| rms_norm_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.838 | 2.654e-07 |
| sigmoid_f32_n1024 | jax-cuda | 32 | 15 | 2.493 | 8.389e-05 |
| sigmoid_f64_n1024 | jax-cuda | 32 | 15 | 2.944 | 7.964e-08 |
| silu_f32_n1024 | jax-cuda | 32 | 15 | 2.888 | 9.241e-05 |
| silu_f64_n1024 | jax-cuda | 32 | 15 | 3.030 | 4.848e-08 |
| softplus_f32_n1024 | jax-cuda | 32 | 15 | 3.155 | 8.073e-05 |
| softplus_f64_n1024 | jax-cuda | 32 | 15 | 3.539 | 1.428e-07 |
| gelu_f32_n1024 | jax-cuda | 2 | 15 | 3.438 | 4.763e-05 |
| gelu_f64_n1024 | jax-cuda | 2 | 15 | 3.351 | 4.625e-08 |
| gelu_tanh_f32_n1024 | jax-cuda | 1 | 15 | 2.673 | 4.766e-05 |
| gelu_tanh_f64_n1024 | jax-cuda | 1 | 15 | 2.678 | 4.349e-08 |
| softmax_f32_n1024 | jax-cuda | 2 | 15 | 2.527 | 1.445e-05 |
| softmax_f64_n1024 | jax-cuda | 2 | 15 | 2.585 | 2.012e-08 |
| log_softmax_f32_n1024 | jax-cuda | 32 | 15 | 2.920 | 0.0001185 |
| log_softmax_f64_n1024 | jax-cuda | 32 | 15 | 2.895 | 2.068e-07 |
| masked_softmax_f32_n1024 | jax-cuda | 1 | 15 | 2.123 | 2.137e-05 |
| masked_softmax_f64_n1024 | jax-cuda | 1 | 15 | 2.108 | 1.335e-08 |
| masked_log_softmax_f32_n1024 | jax-cuda | 32 | 15 | 2.962 | 0.0001303 |
| masked_log_softmax_f64_n1024 | jax-cuda | 32 | 15 | 2.966 | 2.225e-07 |
| layer_norm_f32_n1024 | jax-cuda | 8 | 15 | 2.281 | 0.000223 |
| layer_norm_f64_n1024 | jax-cuda | 8 | 15 | 2.885 | 1.939e-07 |
| rms_norm_f32_n1024 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1024 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| sigmoid_f32_n1048576 | jax-cuda | 16 | 15 | 1.514 | 8.57e-05 |
| sigmoid_f64_n1048576 | jax-cuda | 8 | 15 | 2.443 | 7.964e-08 |
| silu_f32_n1048576 | jax-cuda | 32 | 15 | 2.995 | 9.241e-05 |
| silu_f64_n1048576 | jax-cuda | 8 | 15 | 2.540 | 4.848e-08 |
| softplus_f32_n1048576 | jax-cuda | 32 | 15 | 2.978 | 0.0001273 |
| softplus_f64_n1048576 | jax-cuda | 2 | 15 | 2.202 | 1.467e-07 |
| gelu_f32_n1048576 | jax-cuda | 2 | 15 | 2.229 | 4.939e-05 |
| gelu_f64_n1048576 | jax-cuda | 1 | 15 | 2.964 | 4.81e-08 |
| gelu_tanh_f32_n1048576 | jax-cuda | 2 | 15 | 3.392 | 5.166e-05 |
| gelu_tanh_f64_n1048576 | jax-cuda | 2 | 15 | 3.357 | 4.349e-08 |
| softmax_f32_n1048576 | jax-cuda | 4 | 15 | 3.343 | 6.977e-07 |
| softmax_f64_n1048576 | jax-cuda | 4 | 15 | 3.070 | 6.499e-10 |
| log_softmax_f32_n1048576 | jax-cuda | 32 | 15 | 2.636 | 0.0001277 |
| log_softmax_f64_n1048576 | jax-cuda | 8 | 15 | 3.116 | 2.377e-07 |
| masked_softmax_f32_n1048576 | jax-cuda | 2 | 15 | 2.147 | 9.297e-07 |
| masked_softmax_f64_n1048576 | jax-cuda | 2 | 15 | 2.519 | 8.659e-10 |
| masked_log_softmax_f32_n1048576 | jax-cuda | 32 | 15 | 2.912 | 0.0001337 |
| masked_log_softmax_f64_n1048576 | jax-cuda | 8 | 15 | 3.081 | 2.489e-07 |
| layer_norm_f32_n1048576 | jax-cuda | 16 | 15 | 3.030 | 0.0002619 |
| layer_norm_f64_n1048576 | jax-cuda | 16 | 15 | 3.214 | 3.398e-07 |
| rms_norm_f32_n1048576 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1048576 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| sigmoid_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.525 | 8.389e-05 |
| sigmoid_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.718 | 7.964e-08 |
| silu_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.771 | 9.241e-05 |
| silu_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.992 | 4.848e-08 |
| softplus_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.023 | 8.073e-05 |
| softplus_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.033 | 1.428e-07 |
| gelu_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.490 | 4.763e-05 |
| gelu_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.387 | 4.625e-08 |
| gelu_tanh_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.297 | 4.766e-05 |
| gelu_tanh_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.111 | 4.349e-08 |
| softmax_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.987 | 1.441e-05 |
| softmax_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.976 | 2.012e-08 |
| log_softmax_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.973 | 0.0001185 |
| log_softmax_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.980 | 2.068e-07 |
| masked_softmax_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.066 | 2.137e-05 |
| masked_softmax_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.065 | 1.335e-08 |
| masked_log_softmax_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.080 | 0.0001303 |
| masked_log_softmax_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.078 | 2.225e-07 |
| layer_norm_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.338 | 0.000223 |
| layer_norm_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.284 | 1.939e-07 |
| rms_norm_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| sigmoid_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.077 | 8.57e-05 |
| sigmoid_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.428 | 7.964e-08 |
| silu_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.071 | 9.241e-05 |
| silu_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.529 | 4.848e-08 |
| softplus_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.841 | 0.0001273 |
| softplus_f64_n1048576 | jax-cuda-jit | 2 | 15 | 2.188 | 1.467e-07 |
| gelu_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.184 | 4.939e-05 |
| gelu_f64_n1048576 | jax-cuda-jit | 2 | 15 | 3.503 | 4.81e-08 |
| gelu_tanh_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.109 | 5.166e-05 |
| gelu_tanh_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.464 | 4.349e-08 |
| softmax_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.999 | 6.977e-07 |
| softmax_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.536 | 6.499e-10 |
| log_softmax_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.992 | 0.0001277 |
| log_softmax_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.215 | 2.377e-07 |
| masked_softmax_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.080 | 9.297e-07 |
| masked_softmax_f64_n1048576 | jax-cuda-jit | 4 | 15 | 2.691 | 8.659e-10 |
| masked_log_softmax_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.089 | 0.0001337 |
| masked_log_softmax_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.143 | 2.489e-07 |
| layer_norm_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.907 | 0.0002619 |
| layer_norm_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.818 | 3.398e-07 |
| rms_norm_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/nn
suite_file: benchmarks/gpu/nn.yaml
timestamp: '2026-10-13T10:13:08Z'
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
    ./scripts/run_gpu_public_api.sh nn
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 44
  selected_cases: 44
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
    data/results/nvidia-gpu/gpu/nn/20261013_101308/fixtures/manifest.json: 8b70cf1b7b8b1ff4c32c7efa328c0213d6b03eb4b8973c1de68138f01651075c
    target/release/benchmark_gpu_public_api: 91380afb8faffb97ca96d131ef98cc8e2656aababe594e82d65a07c9bb1bed38
  reference_numpy_version: 2.4.2
```
