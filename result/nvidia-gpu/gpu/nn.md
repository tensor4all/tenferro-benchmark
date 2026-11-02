# gpu/nn: CUDA public API benchmark

Measured cells: **140**. Unsupported cells: **200**, including **128** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/nn/20261102_205922/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| sigmoid_f32_n1024 | unsupported | unsupported | 5.259 (0.505) | 39.659 (0.222) | 40.457 (0.351) |
| sigmoid_f64_n1024 | unsupported | unsupported | 4.621 (0.016) | 40.174 (0.254) | 40.925 (0.193) |
| silu_f32_n1024 | unsupported | unsupported | 5.580 (0.093) | 39.857 (0.236) | 40.418 (0.319) |
| silu_f64_n1024 | unsupported | unsupported | 5.130 (0.739) | 39.701 (0.161) | 39.609 (0.331) |
| softplus_f32_n1024 | 118.166 (0.330) | 193.139 (6.105) | 4.610 (0.539) | 39.412 (0.245) | 39.842 (0.256) |
| softplus_f64_n1024 | 124.444 (0.454) | 211.588 (0.869) | 7.445 (0.029) | 39.551 (0.181) | 40.405 (1.049) |
| gelu_f32_n1024 | unsupported | unsupported | 4.871 (0.028) | 478.175 (4.878) | 39.657 (0.146) |
| gelu_f64_n1024 | unsupported | unsupported | 4.882 (0.023) | 484.353 (3.118) | 41.976 (0.329) |
| gelu_tanh_f32_n1024 | unsupported | unsupported | 5.109 (0.571) | 628.880 (4.945) | 40.534 (0.158) |
| gelu_tanh_f64_n1024 | unsupported | unsupported | 5.684 (0.611) | 622.692 (4.255) | 40.525 (0.412) |
| softmax_f32_n1024 | unsupported | unsupported | 4.836 (0.792) | 315.246 (0.948) | 39.927 (0.305) |
| softmax_f64_n1024 | unsupported | unsupported | 5.261 (0.604) | 316.657 (1.804) | 40.660 (0.438) |
| log_softmax_f32_n1024 | unsupported | unsupported | 4.896 (0.547) | 41.303 (0.427) | 39.790 (0.287) |
| log_softmax_f64_n1024 | unsupported | unsupported | 4.953 (0.225) | 41.703 (0.445) | 40.724 (0.392) |
| masked_softmax_f32_n1024 | unsupported | unsupported | unsupported | 488.073 (3.418) | 42.003 (0.222) |
| masked_softmax_f64_n1024 | unsupported | unsupported | unsupported | 487.153 (1.713) | 43.670 (0.248) |
| masked_log_softmax_f32_n1024 | unsupported | unsupported | unsupported | 42.155 (0.284) | 42.389 (0.230) |
| masked_log_softmax_f64_n1024 | unsupported | unsupported | unsupported | 43.706 (0.162) | 42.956 (0.453) |
| layer_norm_f32_n1024 | unsupported | unsupported | 9.846 (0.027) | 77.926 (0.894) | 45.451 (0.138) |
| layer_norm_f64_n1024 | unsupported | unsupported | 12.528 (0.248) | 78.365 (0.697) | 45.266 (0.421) |
| rms_norm_f32_n1024 | unsupported | unsupported | 9.643 (0.036) | unsupported | unsupported |
| rms_norm_f64_n1024 | unsupported | unsupported | 12.262 (0.227) | unsupported | unsupported |
| sigmoid_f32_n1048576 | unsupported | unsupported | 5.481 (0.049) | 41.185 (0.166) | 41.726 (0.483) |
| sigmoid_f64_n1048576 | unsupported | unsupported | 10.126 (0.064) | 41.517 (0.372) | 41.830 (0.408) |
| silu_f32_n1048576 | unsupported | unsupported | 5.658 (0.080) | 40.840 (0.238) | 41.618 (0.657) |
| silu_f64_n1048576 | unsupported | unsupported | 11.259 (0.057) | 41.120 (0.397) | 42.204 (0.533) |
| softplus_f32_n1048576 | 158.922 (3.183) | 267.508 (3.370) | 8.252 (0.027) | 41.030 (0.137) | 41.575 (0.350) |
| softplus_f64_n1048576 | 199.716 (4.258) | 330.924 (1.531) | 25.909 (0.036) | 40.235 (1.113) | 40.443 (0.306) |
| gelu_f32_n1048576 | unsupported | unsupported | 5.877 (0.104) | 488.215 (1.580) | 41.457 (0.202) |
| gelu_f64_n1048576 | unsupported | unsupported | 15.310 (0.070) | 493.586 (2.369) | 39.503 (0.708) |
| gelu_tanh_f32_n1048576 | unsupported | unsupported | 5.836 (0.150) | 640.985 (3.623) | 41.586 (0.201) |
| gelu_tanh_f64_n1048576 | unsupported | unsupported | 9.753 (0.056) | 636.667 (4.690) | 41.578 (0.401) |
| softmax_f32_n1048576 | unsupported | unsupported | 8.639 (0.028) | 318.689 (1.310) | 40.593 (0.109) |
| softmax_f64_n1048576 | unsupported | unsupported | 16.505 (0.041) | 322.409 (12.215) | 39.008 (0.555) |
| log_softmax_f32_n1048576 | unsupported | unsupported | 6.688 (0.039) | 46.671 (4.430) | 40.869 (0.199) |
| log_softmax_f64_n1048576 | unsupported | unsupported | 13.046 (0.028) | 43.373 (1.242) | 41.089 (0.424) |
| masked_softmax_f32_n1048576 | unsupported | unsupported | unsupported | 496.590 (65.867) | 43.802 (0.256) |
| masked_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 501.768 (8.419) | 41.832 (0.347) |
| masked_log_softmax_f32_n1048576 | unsupported | unsupported | unsupported | 45.233 (7.024) | 43.752 (0.462) |
| masked_log_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 42.075 (0.486) | 41.745 (0.407) |
| layer_norm_f32_n1048576 | unsupported | unsupported | 10.667 (0.122) | 81.044 (2.235) | 47.433 (0.434) |
| layer_norm_f64_n1048576 | unsupported | unsupported | 31.069 (0.065) | 82.016 (0.736) | 50.464 (3.334) |
| rms_norm_f32_n1048576 | unsupported | unsupported | 10.342 (0.124) | unsupported | unsupported |
| rms_norm_f64_n1048576 | unsupported | unsupported | 30.985 (0.038) | unsupported | unsupported |
| layer_norm_weight_f32_n1024 | unsupported | unsupported | 10.040 (0.037) | unsupported | unsupported |
| layer_norm_bias_f32_n1024 | unsupported | unsupported | 10.072 (0.096) | unsupported | unsupported |
| layer_norm_weight_bias_f32_n1024 | unsupported | unsupported | 10.025 (0.049) | unsupported | unsupported |
| layer_norm_weight_f64_n1024 | unsupported | unsupported | 12.821 (0.171) | unsupported | unsupported |
| layer_norm_bias_f64_n1024 | unsupported | unsupported | 12.794 (0.072) | unsupported | unsupported |
| layer_norm_weight_bias_f64_n1024 | unsupported | unsupported | 12.667 (0.074) | unsupported | unsupported |
| rms_norm_weight_f32_n1024 | unsupported | unsupported | 9.732 (0.045) | unsupported | unsupported |
| rms_norm_bias_f32_n1024 | unsupported | unsupported | unsupported | unsupported | unsupported |
| rms_norm_weight_bias_f32_n1024 | unsupported | unsupported | unsupported | unsupported | unsupported |
| rms_norm_weight_f64_n1024 | unsupported | unsupported | 12.266 (0.059) | unsupported | unsupported |
| rms_norm_bias_f64_n1024 | unsupported | unsupported | unsupported | unsupported | unsupported |
| rms_norm_weight_bias_f64_n1024 | unsupported | unsupported | unsupported | unsupported | unsupported |
| layer_norm_weight_f32_n1048576 | unsupported | unsupported | 10.593 (0.063) | unsupported | unsupported |
| layer_norm_bias_f32_n1048576 | unsupported | unsupported | 10.524 (0.118) | unsupported | unsupported |
| layer_norm_weight_bias_f32_n1048576 | unsupported | unsupported | 10.477 (0.125) | unsupported | unsupported |
| layer_norm_weight_f64_n1048576 | unsupported | unsupported | 31.324 (0.073) | unsupported | unsupported |
| layer_norm_bias_f64_n1048576 | unsupported | unsupported | 31.225 (0.083) | unsupported | unsupported |
| layer_norm_weight_bias_f64_n1048576 | unsupported | unsupported | 31.334 (0.067) | unsupported | unsupported |
| rms_norm_weight_f32_n1048576 | unsupported | unsupported | 10.237 (0.089) | unsupported | unsupported |
| rms_norm_bias_f32_n1048576 | unsupported | unsupported | unsupported | unsupported | unsupported |
| rms_norm_weight_bias_f32_n1048576 | unsupported | unsupported | unsupported | unsupported | unsupported |
| rms_norm_weight_f64_n1048576 | unsupported | unsupported | 31.164 (0.042) | unsupported | unsupported |
| rms_norm_bias_f64_n1048576 | unsupported | unsupported | unsupported | unsupported | unsupported |
| rms_norm_weight_bias_f64_n1048576 | unsupported | unsupported | unsupported | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| sigmoid_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softplus_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 3.781 | 8.073e-05 |
| softplus_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 1.991 | 1.428e-07 |
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
| softplus_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.543 | 0.0001273 |
| softplus_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.195 | 1.467e-07 |
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
| softplus_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.090 | 8.073e-05 |
| softplus_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.385 | 1.428e-07 |
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
| softplus_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.140 | 0.0001273 |
| softplus_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.647 | 1.467e-07 |
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
| sigmoid_f32_n1024 | pytorch-cuda | 512 | 15 | 2.692 | 8.389e-05 |
| sigmoid_f64_n1024 | pytorch-cuda | 512 | 15 | 2.366 | 7.964e-08 |
| silu_f32_n1024 | pytorch-cuda | 512 | 15 | 2.857 | 4.766e-05 |
| silu_f64_n1024 | pytorch-cuda | 512 | 15 | 2.627 | 4.848e-08 |
| softplus_f32_n1024 | pytorch-cuda | 512 | 15 | 2.360 | 8.073e-05 |
| softplus_f64_n1024 | pytorch-cuda | 512 | 15 | 3.812 | 1.428e-07 |
| gelu_f32_n1024 | pytorch-cuda | 512 | 15 | 2.494 | 4.754e-05 |
| gelu_f64_n1024 | pytorch-cuda | 512 | 15 | 2.500 | 4.899e-08 |
| gelu_tanh_f32_n1024 | pytorch-cuda | 512 | 15 | 2.616 | 4.766e-05 |
| gelu_tanh_f64_n1024 | pytorch-cuda | 512 | 15 | 2.910 | 4.349e-08 |
| softmax_f32_n1024 | pytorch-cuda | 512 | 15 | 2.476 | 1.435e-05 |
| softmax_f64_n1024 | pytorch-cuda | 512 | 15 | 2.694 | 2.012e-08 |
| log_softmax_f32_n1024 | pytorch-cuda | 512 | 15 | 2.507 | 0.0001185 |
| log_softmax_f64_n1024 | pytorch-cuda | 512 | 15 | 2.536 | 2.068e-07 |
| masked_softmax_f32_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f32_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| layer_norm_f32_n1024 | pytorch-cuda | 256 | 15 | 2.521 | 0.000193 |
| layer_norm_f64_n1024 | pytorch-cuda | 256 | 15 | 3.207 | 1.757e-07 |
| rms_norm_f32_n1024 | pytorch-cuda | 256 | 15 | 2.469 | 0.0001828 |
| rms_norm_f64_n1024 | pytorch-cuda | 256 | 15 | 3.139 | 2.032e-07 |
| sigmoid_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.351 | 8.57e-05 |
| sigmoid_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.324 | 7.964e-08 |
| silu_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.362 | 4.781e-05 |
| silu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.360 | 4.848e-08 |
| softplus_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.528 | 0.0001261 |
| softplus_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.829 | 1.467e-07 |
| gelu_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.376 | 4.754e-05 |
| gelu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.490 | 4.899e-08 |
| gelu_tanh_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.374 | 5.198e-05 |
| gelu_tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.312 | 4.349e-08 |
| softmax_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.553 | 6.976e-07 |
| softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.528 | 8.661e-10 |
| log_softmax_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.428 | 0.0001277 |
| log_softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.417 | 2.377e-07 |
| masked_softmax_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| layer_norm_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.683 | 0.0002616 |
| layer_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.994 | 3.138e-07 |
| rms_norm_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.662 | 0.000212 |
| rms_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.992 | 2.654e-07 |
| layer_norm_weight_f32_n1024 | pytorch-cuda | 256 | 15 | 2.570 | 0.0001499 |
| layer_norm_bias_f32_n1024 | pytorch-cuda | 256 | 15 | 2.579 | 0.0001796 |
| layer_norm_weight_bias_f32_n1024 | pytorch-cuda | 256 | 15 | 2.566 | 0.0001944 |
| layer_norm_weight_f64_n1024 | pytorch-cuda | 256 | 15 | 3.282 | 1.782e-07 |
| layer_norm_bias_f64_n1024 | pytorch-cuda | 256 | 15 | 3.275 | 1.949e-07 |
| layer_norm_weight_bias_f64_n1024 | pytorch-cuda | 256 | 15 | 3.243 | 2.091e-07 |
| rms_norm_weight_f32_n1024 | pytorch-cuda | 256 | 15 | 2.491 | 0.0001965 |
| rms_norm_bias_f32_n1024 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| rms_norm_weight_bias_f32_n1024 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| rms_norm_weight_f64_n1024 | pytorch-cuda | 256 | 15 | 3.140 | 2.218e-07 |
| rms_norm_bias_f64_n1024 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| rms_norm_weight_bias_f64_n1024 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| layer_norm_weight_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.678 | 0.0002792 |
| layer_norm_bias_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.674 | 0.0003084 |
| layer_norm_weight_bias_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.671 | 0.000289 |
| layer_norm_weight_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.002 | 3.219e-07 |
| layer_norm_bias_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.999 | 3.501e-07 |
| layer_norm_weight_bias_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.003 | 3.7e-07 |
| rms_norm_weight_f32_n1048576 | pytorch-cuda | 64 | 15 | 0.655 | 0.000238 |
| rms_norm_bias_f32_n1048576 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| rms_norm_weight_bias_f32_n1048576 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| rms_norm_weight_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.997 | 2.979e-07 |
| rms_norm_bias_f64_n1048576 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| rms_norm_weight_bias_f64_n1048576 | pytorch-cuda | — | — | — | PyTorch rms_norm has no bias argument |
| sigmoid_f32_n1024 | jax-cuda | 64 | 15 | 2.538 | 8.389e-05 |
| sigmoid_f64_n1024 | jax-cuda | 64 | 15 | 2.571 | 7.964e-08 |
| silu_f32_n1024 | jax-cuda | 64 | 15 | 2.551 | 9.241e-05 |
| silu_f64_n1024 | jax-cuda | 64 | 15 | 2.541 | 4.848e-08 |
| softplus_f32_n1024 | jax-cuda | 64 | 15 | 2.522 | 8.073e-05 |
| softplus_f64_n1024 | jax-cuda | 64 | 15 | 2.531 | 1.428e-07 |
| gelu_f32_n1024 | jax-cuda | 8 | 15 | 3.825 | 4.763e-05 |
| gelu_f64_n1024 | jax-cuda | 4 | 15 | 1.937 | 4.625e-08 |
| gelu_tanh_f32_n1024 | jax-cuda | 4 | 15 | 2.516 | 4.766e-05 |
| gelu_tanh_f64_n1024 | jax-cuda | 4 | 15 | 2.491 | 4.349e-08 |
| softmax_f32_n1024 | jax-cuda | 8 | 15 | 2.522 | 1.441e-05 |
| softmax_f64_n1024 | jax-cuda | 8 | 15 | 2.533 | 2.012e-08 |
| log_softmax_f32_n1024 | jax-cuda | 64 | 15 | 2.643 | 0.0001185 |
| log_softmax_f64_n1024 | jax-cuda | 64 | 15 | 2.669 | 2.068e-07 |
| masked_softmax_f32_n1024 | jax-cuda | 4 | 15 | 1.952 | 2.137e-05 |
| masked_softmax_f64_n1024 | jax-cuda | 4 | 15 | 1.949 | 1.335e-08 |
| masked_log_softmax_f32_n1024 | jax-cuda | 64 | 15 | 2.698 | 0.0001303 |
| masked_log_softmax_f64_n1024 | jax-cuda | 64 | 15 | 2.797 | 2.225e-07 |
| layer_norm_f32_n1024 | jax-cuda | 32 | 15 | 2.494 | 0.000223 |
| layer_norm_f64_n1024 | jax-cuda | 32 | 15 | 2.508 | 1.939e-07 |
| rms_norm_f32_n1024 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1024 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| sigmoid_f32_n1048576 | jax-cuda | 64 | 15 | 2.636 | 8.57e-05 |
| sigmoid_f64_n1048576 | jax-cuda | 32 | 15 | 1.329 | 7.964e-08 |
| silu_f32_n1048576 | jax-cuda | 64 | 15 | 2.614 | 9.241e-05 |
| silu_f64_n1048576 | jax-cuda | 32 | 15 | 1.316 | 4.848e-08 |
| softplus_f32_n1048576 | jax-cuda | 64 | 15 | 2.626 | 0.0001273 |
| softplus_f64_n1048576 | jax-cuda | 32 | 15 | 1.288 | 1.467e-07 |
| gelu_f32_n1048576 | jax-cuda | 4 | 15 | 1.953 | 4.939e-05 |
| gelu_f64_n1048576 | jax-cuda | 4 | 15 | 1.974 | 4.81e-08 |
| gelu_tanh_f32_n1048576 | jax-cuda | 4 | 15 | 2.564 | 5.166e-05 |
| gelu_tanh_f64_n1048576 | jax-cuda | 4 | 15 | 2.547 | 4.349e-08 |
| softmax_f32_n1048576 | jax-cuda | 8 | 15 | 2.550 | 6.977e-07 |
| softmax_f64_n1048576 | jax-cuda | 8 | 15 | 2.579 | 6.499e-10 |
| log_softmax_f32_n1048576 | jax-cuda | 64 | 15 | 2.987 | 0.0001276 |
| log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 1.388 | 2.377e-07 |
| masked_softmax_f32_n1048576 | jax-cuda | 4 | 15 | 1.986 | 9.296e-07 |
| masked_softmax_f64_n1048576 | jax-cuda | 4 | 15 | 2.007 | 8.659e-10 |
| masked_log_softmax_f32_n1048576 | jax-cuda | 64 | 15 | 2.895 | 0.0001337 |
| masked_log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 1.346 | 2.489e-07 |
| layer_norm_f32_n1048576 | jax-cuda | 32 | 15 | 2.593 | 0.0002619 |
| layer_norm_f64_n1048576 | jax-cuda | 32 | 15 | 2.625 | 3.398e-07 |
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
| sigmoid_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.589 | 8.389e-05 |
| sigmoid_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.619 | 7.964e-08 |
| silu_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.587 | 9.241e-05 |
| silu_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.535 | 4.848e-08 |
| softplus_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.550 | 8.073e-05 |
| softplus_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.586 | 1.428e-07 |
| gelu_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.538 | 4.763e-05 |
| gelu_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.686 | 4.625e-08 |
| gelu_tanh_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.594 | 4.766e-05 |
| gelu_tanh_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.594 | 4.349e-08 |
| softmax_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.555 | 1.441e-05 |
| softmax_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.602 | 2.012e-08 |
| log_softmax_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.547 | 0.0001185 |
| log_softmax_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.606 | 2.068e-07 |
| masked_softmax_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.688 | 2.137e-05 |
| masked_softmax_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.795 | 1.335e-08 |
| masked_log_softmax_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.713 | 0.0001303 |
| masked_log_softmax_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.749 | 2.225e-07 |
| layer_norm_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.909 | 0.000223 |
| layer_norm_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.897 | 1.939e-07 |
| rms_norm_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| sigmoid_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.670 | 8.57e-05 |
| sigmoid_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.339 | 7.964e-08 |
| silu_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.664 | 9.241e-05 |
| silu_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.351 | 4.848e-08 |
| softplus_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.661 | 0.0001273 |
| softplus_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.294 | 1.467e-07 |
| gelu_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.653 | 4.939e-05 |
| gelu_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.264 | 4.81e-08 |
| gelu_tanh_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.661 | 5.166e-05 |
| gelu_tanh_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.330 | 4.349e-08 |
| softmax_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.598 | 9.299e-07 |
| softmax_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.248 | 8.661e-10 |
| log_softmax_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.616 | 0.0001276 |
| log_softmax_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.315 | 2.377e-07 |
| masked_softmax_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.803 | 9.296e-07 |
| masked_softmax_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.339 | 8.658e-10 |
| masked_log_softmax_f32_n1048576 | jax-cuda-jit | 64 | 15 | 2.800 | 0.0001337 |
| masked_log_softmax_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.336 | 2.489e-07 |
| layer_norm_f32_n1048576 | jax-cuda-jit | 64 | 15 | 3.036 | 0.0002619 |
| layer_norm_f64_n1048576 | jax-cuda-jit | 32 | 15 | 1.615 | 3.398e-07 |
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
timestamp: '2026-11-02T20:59:22Z'
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
    ./scripts/run_gpu_public_api.sh nn
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 68
  selected_cases: 68
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
    data/results/nvidia-gpu/gpu/nn/20261102_205922/fixtures/manifest.json: a6ccc785611d00470335a51f3b441e026b0ca1b6ec38cc79a176f27702706cd5
    target/release/benchmark_gpu_public_api: c0a0be40dd246cd4cb7ca59cac19bacd1549a4f3a593aea91e887cf41f6e4aa6
  reference_numpy_version: 2.4.2
```
