# gpu/nn: CUDA public API benchmark

Measured cells: **140**. Unsupported cells: **200**, including **128** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/nn/20261013_195913/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| sigmoid_f32_n1024 | unsupported | unsupported | 8.161 (0.133) | 97.171 (6.711) | 83.176 (1.971) |
| sigmoid_f64_n1024 | unsupported | unsupported | 9.008 (0.189) | 96.714 (18.318) | 86.608 (4.832) |
| silu_f32_n1024 | unsupported | unsupported | 9.298 (0.116) | 92.513 (9.831) | 82.750 (12.064) |
| silu_f64_n1024 | unsupported | unsupported | 10.325 (0.005) | 94.901 (14.045) | 92.965 (9.313) |
| softplus_f32_n1024 | 139.401 (1.070) | 426.851 (11.426) | 8.766 (0.104) | 94.103 (14.759) | 90.131 (13.796) |
| softplus_f64_n1024 | 142.408 (1.080) | 456.453 (14.745) | 27.768 (0.011) | 107.954 (14.086) | 105.071 (13.731) |
| gelu_f32_n1024 | unsupported | unsupported | 9.692 (0.280) | 1973.115 (131.451) | 101.829 (22.828) |
| gelu_f64_n1024 | unsupported | unsupported | 16.930 (0.012) | 1725.259 (245.590) | 101.251 (9.779) |
| gelu_tanh_f32_n1024 | unsupported | unsupported | 9.214 (0.137) | 2793.260 (36.061) | 100.409 (7.646) |
| gelu_tanh_f64_n1024 | unsupported | unsupported | 9.472 (0.187) | 2711.393 (125.588) | 111.923 (15.216) |
| softmax_f32_n1024 | unsupported | unsupported | 9.241 (0.405) | 1236.437 (36.722) | 88.788 (2.897) |
| softmax_f64_n1024 | unsupported | unsupported | 9.841 (0.297) | 1324.941 (36.415) | 90.925 (3.193) |
| log_softmax_f32_n1024 | unsupported | unsupported | 9.708 (0.441) | 92.338 (3.545) | 86.491 (1.671) |
| log_softmax_f64_n1024 | unsupported | unsupported | 9.103 (0.091) | 90.165 (1.461) | 94.165 (0.667) |
| masked_softmax_f32_n1024 | unsupported | unsupported | unsupported | 1985.904 (194.724) | 77.376 (2.581) |
| masked_softmax_f64_n1024 | unsupported | unsupported | unsupported | 2079.537 (61.463) | 75.414 (2.866) |
| masked_log_softmax_f32_n1024 | unsupported | unsupported | unsupported | 97.104 (0.536) | 73.271 (3.439) |
| masked_log_softmax_f64_n1024 | unsupported | unsupported | unsupported | 96.200 (0.374) | 77.692 (2.174) |
| layer_norm_f32_n1024 | unsupported | unsupported | 19.764 (0.112) | 310.805 (6.311) | 87.244 (12.044) |
| layer_norm_f64_n1024 | unsupported | unsupported | 22.254 (0.149) | 325.776 (10.657) | 87.001 (5.418) |
| rms_norm_f32_n1024 | unsupported | unsupported | 21.102 (0.298) | unsupported | unsupported |
| rms_norm_f64_n1024 | unsupported | unsupported | 23.765 (0.120) | unsupported | unsupported |
| sigmoid_f32_n1048576 | unsupported | unsupported | 27.552 (0.025) | 98.467 (2.958) | 97.653 (2.156) |
| sigmoid_f64_n1048576 | unsupported | unsupported | 258.554 (0.234) | 304.445 (2.818) | 294.273 (8.424) |
| silu_f32_n1048576 | unsupported | unsupported | 27.591 (0.123) | 94.713 (3.514) | 102.162 (11.405) |
| silu_f64_n1048576 | unsupported | unsupported | 293.441 (0.196) | 319.023 (2.523) | 307.624 (17.082) |
| softplus_f32_n1048576 | 292.379 (10.910) | 311.972 (1.164) | 28.233 (0.044) | 94.895 (0.720) | 131.022 (8.119) |
| softplus_f64_n1048576 | 1243.135 (11.969) | 1254.017 (1.591) | 898.240 (0.565) | 1113.581 (7.088) | 1094.401 (17.741) |
| gelu_f32_n1048576 | unsupported | unsupported | 27.495 (0.077) | 1115.829 (31.707) | 146.526 (12.300) |
| gelu_f64_n1048576 | unsupported | unsupported | 575.376 (0.264) | 3044.404 (84.247) | 1751.241 (14.057) |
| gelu_tanh_f32_n1048576 | unsupported | unsupported | 27.423 (0.038) | 1798.457 (182.567) | 139.627 (5.041) |
| gelu_tanh_f64_n1048576 | unsupported | unsupported | 250.459 (0.151) | 1621.246 (38.491) | 312.214 (13.854) |
| softmax_f32_n1048576 | unsupported | unsupported | 27.594 (0.097) | 931.600 (74.505) | 128.996 (3.477) |
| softmax_f64_n1048576 | unsupported | unsupported | 338.554 (0.324) | 776.893 (19.572) | 439.136 (11.251) |
| log_softmax_f32_n1048576 | unsupported | unsupported | 28.250 (0.121) | 92.114 (3.262) | 125.618 (2.064) |
| log_softmax_f64_n1048576 | unsupported | unsupported | 296.895 (3.057) | 396.421 (15.127) | 389.289 (5.692) |
| masked_softmax_f32_n1048576 | unsupported | unsupported | unsupported | 1214.910 (38.140) | 125.935 (1.141) |
| masked_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 1334.785 (35.747) | 674.893 (11.938) |
| masked_log_softmax_f32_n1048576 | unsupported | unsupported | unsupported | 93.151 (3.587) | 129.423 (2.410) |
| masked_log_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 388.475 (16.253) | 386.563 (13.285) |
| layer_norm_f32_n1048576 | unsupported | unsupported | 34.496 (0.125) | 220.572 (4.985) | 144.623 (3.682) |
| layer_norm_f64_n1048576 | unsupported | unsupported | 750.495 (0.306) | 239.872 (16.110) | 147.673 (6.156) |
| rms_norm_f32_n1048576 | unsupported | unsupported | 33.633 (0.054) | unsupported | unsupported |
| rms_norm_f64_n1048576 | unsupported | unsupported | 709.330 (0.423) | unsupported | unsupported |
| layer_norm_weight_f32_n1024 | unsupported | unsupported | 20.139 (0.459) | unsupported | unsupported |
| layer_norm_bias_f32_n1024 | unsupported | unsupported | 20.843 (0.694) | unsupported | unsupported |
| layer_norm_weight_bias_f32_n1024 | unsupported | unsupported | 19.655 (0.588) | unsupported | unsupported |
| layer_norm_weight_f64_n1024 | unsupported | unsupported | 23.385 (1.688) | unsupported | unsupported |
| layer_norm_bias_f64_n1024 | unsupported | unsupported | 22.362 (0.180) | unsupported | unsupported |
| layer_norm_weight_bias_f64_n1024 | unsupported | unsupported | 23.605 (0.735) | unsupported | unsupported |
| rms_norm_weight_f32_n1024 | unsupported | unsupported | 22.540 (1.297) | unsupported | unsupported |
| rms_norm_bias_f32_n1024 | unsupported | unsupported | unsupported | unsupported | unsupported |
| rms_norm_weight_bias_f32_n1024 | unsupported | unsupported | unsupported | unsupported | unsupported |
| rms_norm_weight_f64_n1024 | unsupported | unsupported | 23.900 (0.573) | unsupported | unsupported |
| rms_norm_bias_f64_n1024 | unsupported | unsupported | unsupported | unsupported | unsupported |
| rms_norm_weight_bias_f64_n1024 | unsupported | unsupported | unsupported | unsupported | unsupported |
| layer_norm_weight_f32_n1048576 | unsupported | unsupported | 34.233 (0.091) | unsupported | unsupported |
| layer_norm_bias_f32_n1048576 | unsupported | unsupported | 34.323 (0.143) | unsupported | unsupported |
| layer_norm_weight_bias_f32_n1048576 | unsupported | unsupported | 34.405 (0.084) | unsupported | unsupported |
| layer_norm_weight_f64_n1048576 | unsupported | unsupported | 710.102 (0.457) | unsupported | unsupported |
| layer_norm_bias_f64_n1048576 | unsupported | unsupported | 708.912 (0.342) | unsupported | unsupported |
| layer_norm_weight_bias_f64_n1048576 | unsupported | unsupported | 709.946 (0.360) | unsupported | unsupported |
| rms_norm_weight_f32_n1048576 | unsupported | unsupported | 33.077 (0.088) | unsupported | unsupported |
| rms_norm_bias_f32_n1048576 | unsupported | unsupported | unsupported | unsupported | unsupported |
| rms_norm_weight_bias_f32_n1048576 | unsupported | unsupported | unsupported | unsupported | unsupported |
| rms_norm_weight_f64_n1048576 | unsupported | unsupported | 709.098 (0.338) | unsupported | unsupported |
| rms_norm_bias_f64_n1048576 | unsupported | unsupported | unsupported | unsupported | unsupported |
| rms_norm_weight_bias_f64_n1048576 | unsupported | unsupported | unsupported | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| sigmoid_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softplus_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.230 | 8.073e-05 |
| softplus_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.279 | 1.428e-07 |
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
| softplus_f32_n1048576 | tenferro-cuda-direct | 4 | 15 | 1.170 | 0.0001273 |
| softplus_f64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.486 | 1.467e-07 |
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
| layer_norm_weight_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_bias_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_bias_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_bias_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_bias_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_bias_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_bias_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_bias_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_bias_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_bias_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_bias_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_bias_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_bias_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_bias_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_bias_f32_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_bias_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_bias_f64_n1048576 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softplus_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.415 | 8.073e-05 |
| softplus_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.652 | 1.428e-07 |
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
| softplus_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.496 | 0.0001273 |
| softplus_f64_n1048576 | tenferro-cuda-eager | 2 | 15 | 2.508 | 1.467e-07 |
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
| layer_norm_weight_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_bias_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_bias_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_bias_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_bias_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_bias_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_bias_f32_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_bias_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_bias_f64_n1024 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_bias_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_bias_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_bias_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_weight_bias_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_bias_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_bias_f32_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_bias_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_weight_bias_f64_n1048576 | tenferro-cuda-eager | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f32_n1024 | pytorch-cuda | 256 | 15 | 2.089 | 8.389e-05 |
| sigmoid_f64_n1024 | pytorch-cuda | 256 | 15 | 2.306 | 7.964e-08 |
| silu_f32_n1024 | pytorch-cuda | 256 | 15 | 2.380 | 4.766e-05 |
| silu_f64_n1024 | pytorch-cuda | 256 | 15 | 2.643 | 4.848e-08 |
| softplus_f32_n1024 | pytorch-cuda | 256 | 15 | 2.244 | 8.073e-05 |
| softplus_f64_n1024 | pytorch-cuda | 128 | 15 | 3.554 | 1.428e-07 |
| gelu_f32_n1024 | pytorch-cuda | 256 | 15 | 2.481 | 4.754e-05 |
| gelu_f64_n1024 | pytorch-cuda | 128 | 15 | 2.167 | 4.899e-08 |
| gelu_tanh_f32_n1024 | pytorch-cuda | 256 | 15 | 2.359 | 4.766e-05 |
| gelu_tanh_f64_n1024 | pytorch-cuda | 256 | 15 | 2.425 | 4.349e-08 |
| softmax_f32_n1024 | pytorch-cuda | 256 | 15 | 2.366 | 1.435e-05 |
| softmax_f64_n1024 | pytorch-cuda | 256 | 15 | 2.519 | 2.012e-08 |
| log_softmax_f32_n1024 | pytorch-cuda | 256 | 15 | 2.485 | 0.0001185 |
| log_softmax_f64_n1024 | pytorch-cuda | 256 | 15 | 2.330 | 2.068e-07 |
| masked_softmax_f32_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f32_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| layer_norm_f32_n1024 | pytorch-cuda | 128 | 15 | 2.530 | 0.000193 |
| layer_norm_f64_n1024 | pytorch-cuda | 128 | 15 | 2.848 | 1.757e-07 |
| rms_norm_f32_n1024 | pytorch-cuda | 128 | 15 | 2.701 | 0.0001828 |
| rms_norm_f64_n1024 | pytorch-cuda | 128 | 15 | 3.042 | 2.032e-07 |
| sigmoid_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.763 | 8.57e-05 |
| sigmoid_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.068 | 7.964e-08 |
| silu_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.766 | 4.781e-05 |
| silu_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.348 | 4.848e-08 |
| softplus_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.807 | 0.0001261 |
| softplus_f64_n1048576 | pytorch-cuda | 4 | 15 | 3.593 | 1.467e-07 |
| gelu_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.760 | 4.754e-05 |
| gelu_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.302 | 4.899e-08 |
| gelu_tanh_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.755 | 5.198e-05 |
| gelu_tanh_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.004 | 4.349e-08 |
| softmax_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.766 | 6.976e-07 |
| softmax_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.708 | 8.661e-10 |
| log_softmax_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.808 | 0.0001277 |
| log_softmax_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.375 | 2.377e-07 |
| masked_softmax_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| layer_norm_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.208 | 0.0002616 |
| layer_norm_f64_n1048576 | pytorch-cuda | 4 | 15 | 3.002 | 3.138e-07 |
| rms_norm_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.153 | 0.000212 |
| rms_norm_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.837 | 2.654e-07 |
| layer_norm_weight_f32_n1024 | pytorch-cuda | 128 | 15 | 2.578 | 0.0001499 |
| layer_norm_bias_f32_n1024 | pytorch-cuda | 128 | 15 | 2.668 | 0.0001796 |
| layer_norm_weight_bias_f32_n1024 | pytorch-cuda | 128 | 15 | 2.516 | 0.0001944 |
| layer_norm_weight_f64_n1024 | pytorch-cuda | 128 | 15 | 2.993 | 1.782e-07 |
| layer_norm_bias_f64_n1024 | pytorch-cuda | 128 | 15 | 2.862 | 1.949e-07 |
| layer_norm_weight_bias_f64_n1024 | pytorch-cuda | 128 | 15 | 3.021 | 2.091e-07 |
| rms_norm_weight_f32_n1024 | pytorch-cuda | 128 | 15 | 2.885 | 0.0001965 |
| rms_norm_bias_f32_n1024 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| rms_norm_weight_bias_f32_n1024 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| rms_norm_weight_f64_n1024 | pytorch-cuda | 128 | 15 | 3.059 | 2.218e-07 |
| rms_norm_bias_f64_n1024 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| rms_norm_weight_bias_f64_n1024 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| layer_norm_weight_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.191 | 0.0002792 |
| layer_norm_bias_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.197 | 0.0003084 |
| layer_norm_weight_bias_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.202 | 0.000289 |
| layer_norm_weight_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.840 | 3.219e-07 |
| layer_norm_bias_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.836 | 3.501e-07 |
| layer_norm_weight_bias_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.840 | 3.7e-07 |
| rms_norm_weight_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.117 | 0.000238 |
| rms_norm_bias_f32_n1048576 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| rms_norm_weight_bias_f32_n1048576 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| rms_norm_weight_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.836 | 2.979e-07 |
| rms_norm_bias_f64_n1048576 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| rms_norm_weight_bias_f64_n1048576 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| sigmoid_f32_n1024 | jax-cuda | 32 | 15 | 3.109 | 8.389e-05 |
| sigmoid_f64_n1024 | jax-cuda | 32 | 15 | 3.095 | 7.964e-08 |
| silu_f32_n1024 | jax-cuda | 32 | 15 | 2.960 | 9.241e-05 |
| silu_f64_n1024 | jax-cuda | 32 | 15 | 3.037 | 4.848e-08 |
| softplus_f32_n1024 | jax-cuda | 32 | 15 | 3.011 | 8.073e-05 |
| softplus_f64_n1024 | jax-cuda | 32 | 15 | 3.455 | 1.428e-07 |
| gelu_f32_n1024 | jax-cuda | 1 | 15 | 1.973 | 4.763e-05 |
| gelu_f64_n1024 | jax-cuda | 2 | 15 | 3.451 | 4.625e-08 |
| gelu_tanh_f32_n1024 | jax-cuda | 1 | 15 | 2.793 | 4.766e-05 |
| gelu_tanh_f64_n1024 | jax-cuda | 1 | 15 | 2.711 | 4.349e-08 |
| softmax_f32_n1024 | jax-cuda | 2 | 15 | 2.473 | 1.441e-05 |
| softmax_f64_n1024 | jax-cuda | 2 | 15 | 2.650 | 2.012e-08 |
| log_softmax_f32_n1024 | jax-cuda | 32 | 15 | 2.955 | 0.0001185 |
| log_softmax_f64_n1024 | jax-cuda | 32 | 15 | 2.885 | 2.068e-07 |
| masked_softmax_f32_n1024 | jax-cuda | 2 | 15 | 3.972 | 2.132e-05 |
| masked_softmax_f64_n1024 | jax-cuda | 1 | 15 | 2.080 | 1.335e-08 |
| masked_log_softmax_f32_n1024 | jax-cuda | 32 | 15 | 3.107 | 0.0001303 |
| masked_log_softmax_f64_n1024 | jax-cuda | 32 | 15 | 3.078 | 2.225e-07 |
| layer_norm_f32_n1024 | jax-cuda | 8 | 15 | 2.486 | 0.000223 |
| layer_norm_f64_n1024 | jax-cuda | 8 | 15 | 2.606 | 1.939e-07 |
| rms_norm_f32_n1024 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1024 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| sigmoid_f32_n1048576 | jax-cuda | 16 | 15 | 1.575 | 8.57e-05 |
| sigmoid_f64_n1048576 | jax-cuda | 8 | 15 | 2.436 | 7.964e-08 |
| silu_f32_n1048576 | jax-cuda | 32 | 15 | 3.031 | 9.241e-05 |
| silu_f64_n1048576 | jax-cuda | 8 | 15 | 2.552 | 4.848e-08 |
| softplus_f32_n1048576 | jax-cuda | 32 | 15 | 3.037 | 0.0001273 |
| softplus_f64_n1048576 | jax-cuda | 2 | 15 | 2.227 | 1.467e-07 |
| gelu_f32_n1048576 | jax-cuda | 2 | 15 | 2.232 | 4.939e-05 |
| gelu_f64_n1048576 | jax-cuda | 1 | 15 | 3.044 | 4.81e-08 |
| gelu_tanh_f32_n1048576 | jax-cuda | 2 | 15 | 3.597 | 5.166e-05 |
| gelu_tanh_f64_n1048576 | jax-cuda | 2 | 15 | 3.242 | 4.349e-08 |
| softmax_f32_n1048576 | jax-cuda | 4 | 15 | 3.726 | 6.977e-07 |
| softmax_f64_n1048576 | jax-cuda | 4 | 15 | 3.108 | 6.499e-10 |
| log_softmax_f32_n1048576 | jax-cuda | 32 | 15 | 2.948 | 0.0001277 |
| log_softmax_f64_n1048576 | jax-cuda | 8 | 15 | 3.171 | 2.377e-07 |
| masked_softmax_f32_n1048576 | jax-cuda | 2 | 15 | 2.430 | 9.297e-07 |
| masked_softmax_f64_n1048576 | jax-cuda | 2 | 15 | 2.670 | 8.659e-10 |
| masked_log_softmax_f32_n1048576 | jax-cuda | 32 | 15 | 2.981 | 0.0001337 |
| masked_log_softmax_f64_n1048576 | jax-cuda | 8 | 15 | 3.108 | 2.489e-07 |
| layer_norm_f32_n1048576 | jax-cuda | 16 | 15 | 3.529 | 0.0002619 |
| layer_norm_f64_n1048576 | jax-cuda | 8 | 15 | 1.919 | 3.398e-07 |
| rms_norm_f32_n1048576 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1048576 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| layer_norm_weight_f32_n1024 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_bias_f32_n1024 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_bias_f32_n1024 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_f64_n1024 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_bias_f64_n1024 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_bias_f64_n1024 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_f32_n1024 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_bias_f32_n1024 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_bias_f32_n1024 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_f64_n1024 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_bias_f64_n1024 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_bias_f64_n1024 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_f32_n1048576 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_bias_f32_n1048576 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_bias_f32_n1048576 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_f64_n1048576 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_bias_f64_n1048576 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_bias_f64_n1048576 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_f32_n1048576 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_bias_f32_n1048576 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_bias_f32_n1048576 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_f64_n1048576 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_bias_f64_n1048576 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_bias_f64_n1048576 | jax-cuda | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| sigmoid_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.662 | 8.389e-05 |
| sigmoid_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.771 | 7.964e-08 |
| silu_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.648 | 9.241e-05 |
| silu_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.975 | 4.848e-08 |
| softplus_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.884 | 8.073e-05 |
| softplus_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.362 | 1.428e-07 |
| gelu_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.259 | 4.763e-05 |
| gelu_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.240 | 4.625e-08 |
| gelu_tanh_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.213 | 4.766e-05 |
| gelu_tanh_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.582 | 4.349e-08 |
| softmax_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.841 | 1.441e-05 |
| softmax_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.910 | 2.012e-08 |
| log_softmax_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.768 | 0.0001185 |
| log_softmax_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.013 | 2.068e-07 |
| masked_softmax_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.476 | 2.137e-05 |
| masked_softmax_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.413 | 1.335e-08 |
| masked_log_softmax_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.345 | 0.0001303 |
| masked_log_softmax_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.486 | 2.225e-07 |
| layer_norm_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.792 | 0.000223 |
| layer_norm_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.784 | 1.939e-07 |
| rms_norm_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| sigmoid_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.125 | 8.57e-05 |
| sigmoid_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.354 | 7.964e-08 |
| silu_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.269 | 9.241e-05 |
| silu_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.461 | 4.848e-08 |
| softplus_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.096 | 0.0001273 |
| softplus_f64_n1048576 | jax-cuda-jit | 2 | 15 | 2.189 | 1.467e-07 |
| gelu_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.344 | 4.939e-05 |
| gelu_f64_n1048576 | jax-cuda-jit | 2 | 15 | 3.502 | 4.81e-08 |
| gelu_tanh_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.234 | 5.166e-05 |
| gelu_tanh_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.498 | 4.349e-08 |
| softmax_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.064 | 6.977e-07 |
| softmax_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.513 | 6.499e-10 |
| log_softmax_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.010 | 0.0001277 |
| log_softmax_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.114 | 2.377e-07 |
| masked_softmax_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.015 | 9.297e-07 |
| masked_softmax_f64_n1048576 | jax-cuda-jit | 4 | 15 | 2.700 | 8.659e-10 |
| masked_log_softmax_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.071 | 0.0001337 |
| masked_log_softmax_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.093 | 2.489e-07 |
| layer_norm_f32_n1048576 | jax-cuda-jit | 16 | 15 | 2.314 | 0.0002619 |
| layer_norm_f64_n1048576 | jax-cuda-jit | 16 | 15 | 2.363 | 3.398e-07 |
| rms_norm_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| layer_norm_weight_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_bias_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_bias_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_bias_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_bias_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_bias_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_bias_f32_n1024 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_bias_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_bias_f64_n1024 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_bias_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_bias_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_bias_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| layer_norm_weight_bias_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_bias_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_bias_f32_n1048576 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_bias_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |
| rms_norm_weight_bias_f64_n1048576 | jax-cuda-jit | — | — | — | No verified single-call affine normalization counterpart; composition is not an equivalent isolated API |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/nn
suite_file: benchmarks/gpu/nn.yaml
timestamp: '2026-10-13T19:59:13Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 8016925918e04de86dc9ceee64dca1717868e185
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
    ./scripts/run_gpu_public_api.sh nn
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 68
  selected_cases: 68
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: e04b88277edb1bb3202f50da23a7b0d359a81675b9fe8a8a3692b1595961aebe
    src/bin/benchmark_gpu_public_api/plan.rs: 3d0f10f390ebd0e08a40deb5a64e49e086dcd41b9a777d3a208c6846ed6b2fac
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: ffdfb477b4896756bb69d1d545f0a3ec7df694b6dac17427fe641873108d119c
    src/bin/benchmark_gpu_public_api/multi.rs: 1b477fef78260cf28aa31e515d57ab411f6241b91ffb30d625939538fdbc6bab
    src/bin/benchmark_gpu_public_api/lifetime.rs: c687a8b0329f6d3073073671b5d246acbb1f4dc3f983d1430da897a908ef1745
    src/bin/benchmark_gpu_public_api/read_layout.rs: 184d4508d0729f531ee8107e3b70aa6b77c5a9e952c2dc43606a868cd97631a2
    src/bin/benchmark_gpu_public_api/structural.rs: b8fd99ceb5c80fd3474974b0ba9aa57833ad6bd25383af3091c11c0b6ab94f6d
    src/bin/benchmark_gpu_public_api/linalg.rs: 5f5c4afc161f3d6e267861511d0d5871be9192d9968c95b59a73c9dc5a64e22b
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: 1dc14ac94fac700aa696d0ada04315685e3245738f9514de3d7989b8cb59152b
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 3faaaf5ae74a2ef9c91888b8fd8e566d1cf83b24d2b385cc1cd7b87d9d80c6fb
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: b4c98110fe1fbb5639d85dd5c376ed17bb629b14a987f176adc891ccf4bc1c69
    scripts/run_gpu_public_api.sh: dd5e4cc47e85fcb5afdc56d460136b2a399a6a7d6a62005de8ba94afb0119ecc
    data/results/nvidia-gpu/gpu/nn/20261013_195913/fixtures/manifest.json: a6ccc785611d00470335a51f3b441e026b0ca1b6ec38cc79a176f27702706cd5
    target/release/benchmark_gpu_public_api: ae31b904a3df5df3e63b845bfdababaa343caa2b31fe9ba535523c0b94b6e5df
  reference_numpy_version: 2.4.2
```
