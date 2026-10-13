# gpu/nn: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/nn/20261013_133644/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| sigmoid_f32_n1024 | unsupported | unsupported | 8.876 (0.242) | 100.999 (3.097) | 88.209 (2.508) |
| sigmoid_f64_n1024 | unsupported | unsupported | 9.061 (0.138) | 123.104 (3.842) | 90.726 (6.248) |
| silu_f32_n1024 | unsupported | unsupported | 9.766 (0.179) | 140.147 (24.381) | 89.390 (13.051) |
| silu_f64_n1024 | unsupported | unsupported | 10.439 (0.357) | 147.836 (28.606) | 93.945 (14.384) |
| softplus_f32_n1024 | 129.662 (1.935) | 416.957 (11.236) | 9.272 (0.129) | 149.108 (11.424) | 98.562 (15.860) |
| softplus_f64_n1024 | 134.003 (1.533) | 454.481 (22.615) | 27.764 (0.007) | 146.336 (10.196) | 104.147 (16.739) |
| gelu_f32_n1024 | unsupported | unsupported | 10.489 (0.172) | 2046.881 (87.806) | 130.023 (21.445) |
| gelu_f64_n1024 | unsupported | unsupported | 16.933 (0.005) | 1728.828 (220.394) | 128.259 (3.207) |
| gelu_tanh_f32_n1024 | unsupported | unsupported | 10.960 (0.175) | 2855.015 (40.744) | 130.291 (23.718) |
| gelu_tanh_f64_n1024 | unsupported | unsupported | 10.707 (0.202) | 2728.146 (124.363) | 133.786 (30.159) |
| softmax_f32_n1024 | unsupported | unsupported | 9.893 (0.182) | 1376.865 (7.280) | 89.579 (2.246) |
| softmax_f64_n1024 | unsupported | unsupported | 10.239 (0.492) | 1380.749 (23.740) | 85.359 (1.386) |
| log_softmax_f32_n1024 | unsupported | unsupported | 9.797 (0.114) | 96.356 (1.013) | 86.838 (1.862) |
| log_softmax_f64_n1024 | unsupported | unsupported | 9.864 (0.135) | 90.664 (1.067) | 88.807 (1.406) |
| masked_softmax_f32_n1024 | unsupported | unsupported | unsupported | 2216.621 (23.454) | 92.014 (2.415) |
| masked_softmax_f64_n1024 | unsupported | unsupported | unsupported | 2123.874 (61.059) | 94.745 (0.406) |
| masked_log_softmax_f32_n1024 | unsupported | unsupported | unsupported | 97.595 (2.496) | 94.423 (0.782) |
| masked_log_softmax_f64_n1024 | unsupported | unsupported | unsupported | 97.800 (0.257) | 95.582 (0.664) |
| layer_norm_f32_n1024 | unsupported | unsupported | 22.347 (0.127) | 309.026 (13.946) | 112.942 (7.656) |
| layer_norm_f64_n1024 | unsupported | unsupported | 24.213 (0.239) | 313.301 (20.376) | 137.452 (2.259) |
| rms_norm_f32_n1024 | unsupported | unsupported | 21.190 (0.182) | unsupported | unsupported |
| rms_norm_f64_n1024 | unsupported | unsupported | 24.396 (0.727) | unsupported | unsupported |
| sigmoid_f32_n1048576 | unsupported | unsupported | 27.553 (0.026) | 104.495 (3.536) | 107.287 (1.579) |
| sigmoid_f64_n1048576 | unsupported | unsupported | 258.496 (0.249) | 307.015 (11.451) | 300.646 (14.114) |
| silu_f32_n1048576 | unsupported | unsupported | 27.525 (0.056) | 97.101 (1.410) | 106.275 (2.765) |
| silu_f64_n1048576 | unsupported | unsupported | 293.354 (0.051) | 317.157 (2.508) | 315.619 (2.739) |
| softplus_f32_n1048576 | 269.739 (1.908) | 311.635 (1.516) | 28.245 (0.050) | 99.770 (4.705) | 94.974 (5.053) |
| softplus_f64_n1048576 | 1249.864 (2.969) | 1253.303 (0.948) | 898.631 (0.652) | 1102.660 (10.949) | 1111.924 (15.809) |
| gelu_f32_n1048576 | unsupported | unsupported | 27.447 (0.022) | 1173.154 (25.159) | 97.329 (9.115) |
| gelu_f64_n1048576 | unsupported | unsupported | 575.268 (0.430) | 2984.561 (88.470) | 1755.353 (12.376) |
| gelu_tanh_f32_n1048576 | unsupported | unsupported | 27.425 (0.036) | 1783.378 (50.961) | 96.335 (4.263) |
| gelu_tanh_f64_n1048576 | unsupported | unsupported | 250.416 (0.133) | 1765.875 (23.997) | 308.568 (3.551) |
| softmax_f32_n1048576 | unsupported | unsupported | 27.643 (0.068) | 1032.611 (29.270) | 90.772 (0.555) |
| softmax_f64_n1048576 | unsupported | unsupported | 338.552 (0.345) | 835.277 (18.342) | 438.894 (11.209) |
| log_softmax_f32_n1048576 | unsupported | unsupported | 28.220 (0.036) | 92.289 (2.360) | 98.949 (5.393) |
| log_softmax_f64_n1048576 | unsupported | unsupported | 292.138 (4.869) | 397.839 (10.803) | 397.288 (10.773) |
| masked_softmax_f32_n1048576 | unsupported | unsupported | unsupported | 1356.867 (17.173) | 86.932 (0.946) |
| masked_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 1364.810 (32.561) | 680.962 (10.967) |
| masked_log_softmax_f32_n1048576 | unsupported | unsupported | unsupported | 98.327 (4.461) | 92.965 (2.622) |
| masked_log_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 386.068 (3.678) | 387.578 (9.824) |
| layer_norm_f32_n1048576 | unsupported | unsupported | 34.379 (0.164) | 220.613 (13.199) | 110.069 (5.018) |
| layer_norm_f64_n1048576 | unsupported | unsupported | 709.133 (0.756) | 251.006 (10.896) | 118.834 (1.295) |
| rms_norm_f32_n1048576 | unsupported | unsupported | 33.626 (0.049) | unsupported | unsupported |
| rms_norm_f64_n1048576 | unsupported | unsupported | 709.370 (0.212) | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| sigmoid_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f32_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1024 | tenferro-cuda-direct | — | — | — | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softplus_f32_n1024 | tenferro-cuda-direct | 16 | 15 | 2.075 | 8.073e-05 |
| softplus_f64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.144 | 1.428e-07 |
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
| softplus_f32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.158 | 0.0001273 |
| softplus_f64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.500 | 1.467e-07 |
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
| softplus_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.336 | 8.073e-05 |
| softplus_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.636 | 1.428e-07 |
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
| softplus_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.493 | 0.0001273 |
| softplus_f64_n1048576 | tenferro-cuda-eager | 2 | 15 | 2.507 | 1.467e-07 |
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
| sigmoid_f32_n1024 | pytorch-cuda | 256 | 15 | 2.272 | 8.389e-05 |
| sigmoid_f64_n1024 | pytorch-cuda | 256 | 15 | 2.320 | 7.964e-08 |
| silu_f32_n1024 | pytorch-cuda | 256 | 15 | 2.500 | 4.766e-05 |
| silu_f64_n1024 | pytorch-cuda | 256 | 15 | 2.672 | 4.848e-08 |
| softplus_f32_n1024 | pytorch-cuda | 256 | 15 | 2.374 | 8.073e-05 |
| softplus_f64_n1024 | pytorch-cuda | 128 | 15 | 3.554 | 1.428e-07 |
| gelu_f32_n1024 | pytorch-cuda | 256 | 15 | 2.685 | 4.754e-05 |
| gelu_f64_n1024 | pytorch-cuda | 128 | 15 | 2.167 | 4.899e-08 |
| gelu_tanh_f32_n1024 | pytorch-cuda | 256 | 15 | 2.806 | 4.766e-05 |
| gelu_tanh_f64_n1024 | pytorch-cuda | 256 | 15 | 2.741 | 4.349e-08 |
| softmax_f32_n1024 | pytorch-cuda | 256 | 15 | 2.533 | 1.435e-05 |
| softmax_f64_n1024 | pytorch-cuda | 256 | 15 | 2.621 | 2.012e-08 |
| log_softmax_f32_n1024 | pytorch-cuda | 256 | 15 | 2.508 | 0.0001185 |
| log_softmax_f64_n1024 | pytorch-cuda | 256 | 15 | 2.525 | 2.068e-07 |
| masked_softmax_f32_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f32_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1024 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| layer_norm_f32_n1024 | pytorch-cuda | 128 | 15 | 2.860 | 0.000193 |
| layer_norm_f64_n1024 | pytorch-cuda | 128 | 15 | 3.099 | 1.757e-07 |
| rms_norm_f32_n1024 | pytorch-cuda | 128 | 15 | 2.712 | 0.0001828 |
| rms_norm_f64_n1024 | pytorch-cuda | 128 | 15 | 3.123 | 2.032e-07 |
| sigmoid_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.763 | 8.57e-05 |
| sigmoid_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.068 | 7.964e-08 |
| silu_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.762 | 4.781e-05 |
| silu_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.347 | 4.848e-08 |
| softplus_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.808 | 0.0001261 |
| softplus_f64_n1048576 | pytorch-cuda | 4 | 15 | 3.595 | 1.467e-07 |
| gelu_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.757 | 4.754e-05 |
| gelu_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.301 | 4.899e-08 |
| gelu_tanh_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.755 | 5.198e-05 |
| gelu_tanh_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.003 | 4.349e-08 |
| softmax_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.769 | 6.976e-07 |
| softmax_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.708 | 8.661e-10 |
| log_softmax_f32_n1048576 | pytorch-cuda | 64 | 15 | 1.806 | 0.0001277 |
| log_softmax_f64_n1048576 | pytorch-cuda | 8 | 15 | 2.337 | 2.377e-07 |
| masked_softmax_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f32_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1048576 | pytorch-cuda | — | — | — | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| layer_norm_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.200 | 0.0002616 |
| layer_norm_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.837 | 3.138e-07 |
| rms_norm_f32_n1048576 | pytorch-cuda | 64 | 15 | 2.152 | 0.000212 |
| rms_norm_f64_n1048576 | pytorch-cuda | 4 | 15 | 2.837 | 2.654e-07 |
| sigmoid_f32_n1024 | jax-cuda | 32 | 15 | 3.232 | 8.389e-05 |
| sigmoid_f64_n1024 | jax-cuda | 16 | 15 | 1.970 | 7.964e-08 |
| silu_f32_n1024 | jax-cuda | 16 | 15 | 2.242 | 9.241e-05 |
| silu_f64_n1024 | jax-cuda | 16 | 15 | 2.365 | 4.848e-08 |
| softplus_f32_n1024 | jax-cuda | 16 | 15 | 2.386 | 8.073e-05 |
| softplus_f64_n1024 | jax-cuda | 16 | 15 | 2.341 | 1.428e-07 |
| gelu_f32_n1024 | jax-cuda | 1 | 15 | 2.047 | 4.763e-05 |
| gelu_f64_n1024 | jax-cuda | 2 | 15 | 3.458 | 4.625e-08 |
| gelu_tanh_f32_n1024 | jax-cuda | 1 | 15 | 2.855 | 4.766e-05 |
| gelu_tanh_f64_n1024 | jax-cuda | 1 | 15 | 2.728 | 4.349e-08 |
| softmax_f32_n1024 | jax-cuda | 2 | 15 | 2.754 | 1.441e-05 |
| softmax_f64_n1024 | jax-cuda | 2 | 15 | 2.761 | 2.012e-08 |
| log_softmax_f32_n1024 | jax-cuda | 32 | 15 | 3.083 | 0.0001185 |
| log_softmax_f64_n1024 | jax-cuda | 32 | 15 | 2.901 | 2.068e-07 |
| masked_softmax_f32_n1024 | jax-cuda | 1 | 15 | 2.217 | 2.132e-05 |
| masked_softmax_f64_n1024 | jax-cuda | 1 | 15 | 2.124 | 1.335e-08 |
| masked_log_softmax_f32_n1024 | jax-cuda | 32 | 15 | 3.123 | 0.0001303 |
| masked_log_softmax_f64_n1024 | jax-cuda | 32 | 15 | 3.130 | 2.225e-07 |
| layer_norm_f32_n1024 | jax-cuda | 8 | 15 | 2.472 | 0.000223 |
| layer_norm_f64_n1024 | jax-cuda | 8 | 15 | 2.506 | 1.939e-07 |
| rms_norm_f32_n1024 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1024 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| sigmoid_f32_n1048576 | jax-cuda | 16 | 15 | 1.672 | 8.57e-05 |
| sigmoid_f64_n1048576 | jax-cuda | 8 | 15 | 2.456 | 7.964e-08 |
| silu_f32_n1048576 | jax-cuda | 32 | 15 | 3.107 | 9.241e-05 |
| silu_f64_n1048576 | jax-cuda | 8 | 15 | 2.537 | 4.848e-08 |
| softplus_f32_n1048576 | jax-cuda | 32 | 15 | 3.193 | 0.0001273 |
| softplus_f64_n1048576 | jax-cuda | 2 | 15 | 2.205 | 1.467e-07 |
| gelu_f32_n1048576 | jax-cuda | 2 | 15 | 2.346 | 4.939e-05 |
| gelu_f64_n1048576 | jax-cuda | 1 | 15 | 2.985 | 4.81e-08 |
| gelu_tanh_f32_n1048576 | jax-cuda | 2 | 15 | 3.567 | 5.166e-05 |
| gelu_tanh_f64_n1048576 | jax-cuda | 2 | 15 | 3.532 | 4.349e-08 |
| softmax_f32_n1048576 | jax-cuda | 2 | 15 | 2.065 | 6.977e-07 |
| softmax_f64_n1048576 | jax-cuda | 4 | 15 | 3.341 | 6.499e-10 |
| log_softmax_f32_n1048576 | jax-cuda | 32 | 15 | 2.953 | 0.0001277 |
| log_softmax_f64_n1048576 | jax-cuda | 8 | 15 | 3.183 | 2.377e-07 |
| masked_softmax_f32_n1048576 | jax-cuda | 2 | 15 | 2.714 | 9.296e-07 |
| masked_softmax_f64_n1048576 | jax-cuda | 2 | 15 | 2.730 | 8.659e-10 |
| masked_log_softmax_f32_n1048576 | jax-cuda | 32 | 15 | 3.146 | 0.0001337 |
| masked_log_softmax_f64_n1048576 | jax-cuda | 8 | 15 | 3.089 | 2.489e-07 |
| layer_norm_f32_n1048576 | jax-cuda | 16 | 15 | 3.530 | 0.0002619 |
| layer_norm_f64_n1048576 | jax-cuda | 8 | 15 | 2.008 | 3.398e-07 |
| rms_norm_f32_n1048576 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1048576 | jax-cuda | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| sigmoid_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.823 | 8.389e-05 |
| sigmoid_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.903 | 7.964e-08 |
| silu_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.860 | 9.241e-05 |
| silu_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.006 | 4.848e-08 |
| softplus_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.154 | 8.073e-05 |
| softplus_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.333 | 1.428e-07 |
| gelu_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.080 | 4.763e-05 |
| gelu_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.052 | 4.625e-08 |
| gelu_tanh_f32_n1024 | jax-cuda-jit | 16 | 15 | 2.085 | 4.766e-05 |
| gelu_tanh_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.141 | 4.349e-08 |
| softmax_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.867 | 1.441e-05 |
| softmax_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.732 | 2.012e-08 |
| log_softmax_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.779 | 0.0001185 |
| log_softmax_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.842 | 2.068e-07 |
| masked_softmax_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.944 | 2.137e-05 |
| masked_softmax_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.032 | 1.335e-08 |
| masked_log_softmax_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.022 | 0.0001303 |
| masked_log_softmax_f64_n1024 | jax-cuda-jit | 32 | 15 | 3.059 | 2.225e-07 |
| layer_norm_f32_n1024 | jax-cuda-jit | 32 | 15 | 3.614 | 0.000223 |
| layer_norm_f64_n1024 | jax-cuda-jit | 16 | 15 | 2.199 | 1.939e-07 |
| rms_norm_f32_n1024 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1024 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| sigmoid_f32_n1048576 | jax-cuda-jit | 16 | 15 | 1.717 | 8.57e-05 |
| sigmoid_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.405 | 7.964e-08 |
| silu_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.401 | 9.241e-05 |
| silu_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.525 | 4.848e-08 |
| softplus_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.039 | 0.0001273 |
| softplus_f64_n1048576 | jax-cuda-jit | 2 | 15 | 2.224 | 1.467e-07 |
| gelu_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.115 | 4.939e-05 |
| gelu_f64_n1048576 | jax-cuda-jit | 2 | 15 | 3.511 | 4.81e-08 |
| gelu_tanh_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.083 | 5.166e-05 |
| gelu_tanh_f64_n1048576 | jax-cuda-jit | 8 | 15 | 2.469 | 4.349e-08 |
| softmax_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.905 | 6.977e-07 |
| softmax_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.511 | 6.499e-10 |
| log_softmax_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.166 | 0.0001277 |
| log_softmax_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.178 | 2.377e-07 |
| masked_softmax_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.782 | 9.297e-07 |
| masked_softmax_f64_n1048576 | jax-cuda-jit | 4 | 15 | 2.724 | 8.659e-10 |
| masked_log_softmax_f32_n1048576 | jax-cuda-jit | 32 | 15 | 2.975 | 0.0001337 |
| masked_log_softmax_f64_n1048576 | jax-cuda-jit | 8 | 15 | 3.101 | 2.489e-07 |
| layer_norm_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.522 | 0.0002619 |
| layer_norm_f64_n1048576 | jax-cuda-jit | 32 | 15 | 3.803 | 3.398e-07 |
| rms_norm_f32_n1048576 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1048576 | jax-cuda-jit | — | — | — | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/nn
suite_file: benchmarks/gpu/nn.yaml
timestamp: '2026-10-13T13:36:45Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 2858dcda46699788bc1df84d923a95ed93ef6469
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
  expected_cases: 44
  selected_cases: 44
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 249184323d09433d1fbf401284e39fa797c0252dceb20351bbcc151551d1b32c
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/structural.rs: 382fb5ffb6d86ac1a4a66723376808e9c95c8a8731570a5617f2ffed2a6e40a7
    src/bin/benchmark_gpu_public_api/linalg.rs: 84162923fa0d0772c88ec9c4eab169ad3a75e5d16ff76961d1541670dd8565c1
    src/bin/benchmark_gpu_public_api/reuse.rs: ad72b565373904275128d7beb8673bada08d922e8f310be6cedb0d2d8187898f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 4d05b9facc0b7e5a3ff31146fc42f98d680010264c6606a3154e9da2dfa6a0cc
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 71cb26510a35647e7aadc46827ca00d06a06e283eacfcba8ebe93d2e8e8add86
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/nn/20261013_133644/fixtures/manifest.json: 8b70cf1b7b8b1ff4c32c7efa328c0213d6b03eb4b8973c1de68138f01651075c
    target/release/benchmark_gpu_public_api: b505a2bdb3d0c5a7bffd1390af659a154753ea998ec68becd4779e137c9c7150
  reference_numpy_version: 2.4.2
```
