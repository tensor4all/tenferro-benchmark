# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense`
- Suite file: `benchmarks/gpu/dense.yaml`
- Timestamp: `2026-10-13T13:44:20.569187+00:00`
- tenferro-rs commit: `dfbdf6b66d40162f4731da116ec42ed034da0dce`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA GeForce RTX 3060`
- UUID: `GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`
- Memory: `12 GiB`
- Driver version: `580.178.04`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92700`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-146-generic-x86_64-with-glibc2.39`

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.
Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.
tenferro-rs uses native column-major GPU tensors; PyTorch and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.

## gpu/dense / batched_matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| dense_batched_matmul_f64_b1024_256 | unsupported | 187.869 | 185.636 | 161.360 |

## gpu/dense / eigh / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| dense_eigh_f64_1024 | unsupported | 74.636 | 75.254 | 75.624 |

## gpu/dense / matmul / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| dense_matmul_f64_3072 | unsupported | 316.076 | 312.244 | 270.648 |

## gpu/dense / qr / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| dense_qr_f64_1536 | unsupported | 96.367 | 94.877 | 96.905 |

## gpu/dense / solve / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| dense_solve_f64_1024_rhs16 | unsupported | 12.471 | 10.795 | 11.078 |
| dense_solve_f64_2048_rhs128 | unsupported | 55.809 | 53.889 | 54.440 |
| dense_solve_f64_2048_rhs16 | unsupported | 52.251 | 48.344 | 48.941 |
| dense_solve_f64_4096_rhs16 | unsupported | 290.669 | 288.439 | 290.013 |
| dense_solve_f64_512_rhs16 | unsupported | 4.863 | 3.162 | 3.392 |

## gpu/dense / svd / allocating output

> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; PyTorch's default row may use a different SVD driver and row-major framework layout.

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| dense_svd_f64_256 | unsupported | 83.546 | 85.219 | 82.570 |

## Python sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| dense_matmul_f64_3072 | tenferro-cuda-eager | 1 | 316.102, 316.094, 316.061, 316.076, 316.075, 316.076, 316.081 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | 1 | 187.873, 187.883, 187.903, 187.869, 187.83, 187.86, 187.844 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_qr_f64_1536 | tenferro-cuda-eager | 1 | 96.3479, 96.3471, 96.5014, 96.3668, 96.5378 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_512_rhs16 | tenferro-cuda-eager | 1 | 5.59121, 6.19363, 4.8632, 4.59639, 4.54806 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_1024_rhs16 | tenferro-cuda-eager | 1 | 12.6539, 12.4633, 12.4706, 12.4252, 15.5831 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-eager | 1 | 49.44, 55.7445, 49.4819, 52.2506, 52.5053 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_4096_rhs16 | tenferro-cuda-eager | 1 | 290.785, 290.59, 290.769, 290.669, 290.615 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_solve_f64_2048_rhs128 | tenferro-cuda-eager | 1 | 55.805, 59.0202, 55.8529, 55.8094, 55.7961 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_svd_f64_256 | tenferro-cuda-eager | 1 | 83.5335, 83.5055, 83.5464, 83.5641, 83.5888 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_eigh_f64_1024 | tenferro-cuda-eager | 1 | 74.7723, 74.5624, 74.6054, 74.6679, 74.6362 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| dense_matmul_f64_3072 | pytorch-cuda | 1 | 312.265, 312.261, 312.245, 312.244, 312.225, 312.229, 312.171 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | 1 | 185.642, 185.626, 185.638, 185.635, 185.632, 185.638, 185.636 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | pytorch-cuda | 1 | 94.904, 94.8721, 94.8845, 94.8771, 94.8494 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_512_rhs16 | pytorch-cuda | 1 | 3.16826, 3.15391, 3.1322, 3.16236, 3.18404 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_1024_rhs16 | pytorch-cuda | 1 | 10.8016, 10.7952, 10.7875, 10.7825, 10.7947 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | pytorch-cuda | 1 | 47.9807, 48.0559, 48.3437, 48.3466, 48.3529 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_4096_rhs16 | pytorch-cuda | 1 | 288.375, 288.416, 288.439, 288.478, 288.443 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs128 | pytorch-cuda | 1 | 53.8827, 54.0161, 53.8892, 53.8921, 53.8869 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | pytorch-cuda | 1 | 85.2188, 85.1169, 85.5817, 85.2431, 85.2038 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | pytorch-cuda | 1 | 75.2544, 75.2779, 75.333, 75.1971, 75.1852 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_matmul_f64_3072 | jax-cuda | 1 | 270.648, 270.532, 270.947, 270.726, 270.666, 270.607, 270.645 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_batched_matmul_f64_b1024_256 | jax-cuda | 1 | 161.341, 161.336, 161.364, 161.36, 161.38, 161.374, 161.357 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_qr_f64_1536 | jax-cuda | 1 | 96.8839, 96.8084, 96.9108, 96.9147, 96.9048 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_512_rhs16 | jax-cuda | 1 | 3.43392, 3.40223, 3.37024, 3.36135, 3.39161 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_1024_rhs16 | jax-cuda | 1 | 11.0559, 11.0779, 11.1011, 11.12, 11.0342 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs16 | jax-cuda | 1 | 48.9407, 48.9637, 48.9404, 48.9703, 48.933 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_4096_rhs16 | jax-cuda | 1 | 289.981, 289.927, 290.235, 290.159, 290.013 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_solve_f64_2048_rhs128 | jax-cuda | 1 | 54.5016, 54.2936, 54.318, 54.489, 54.4401 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_svd_f64_256 | jax-cuda | 1 | 82.575, 82.5701, 82.5857, 82.5608, 82.52 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| dense_eigh_f64_1024 | jax-cuda | 1 | 75.6981, 75.6245, 75.4961, 75.7162, 75.5337 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
