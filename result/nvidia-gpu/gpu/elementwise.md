# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/elementwise`
- Suite file: `benchmarks/gpu/elementwise.yaml`
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

## gpu/elementwise / elementwise_chain / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA JIT |
|---|---:|---:|---:|---:|
| elementwise_chain_f64_1k | unsupported | 0.458 | 0.251 | 0.155 |
| elementwise_chain_f64_1m | unsupported | 4.834 | 4.880 | 4.297 |

## Python sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| elementwise_chain_f64_1k | tenferro-cuda-eager | 8 | 2.64143, 3.33895, 3.46367, 3.69089, 3.66386, 3.67829, 3.70677 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1m | tenferro-cuda-eager | 1 | 4.84696, 4.83284, 4.85691, 4.8334, 4.83428, 4.82187, 4.83501 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1k | pytorch-cuda | 8 | 2.00918, 2.0062, 2.01059, 2.0165, 2.0074, 2.01771, 2.19509 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1m | pytorch-cuda | 1 | 4.87907, 4.87701, 4.87801, 4.87982, 4.87998, 4.88635, 4.8864 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1k | jax-cuda | 16 | 2.48195, 2.53381, 2.55659, 2.44954, 2.53417, 2.47662, 2.47437 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| elementwise_chain_f64_1m | jax-cuda | 1 | 4.28601, 4.30957, 4.29723, 4.26971, 4.29731, 4.29164, 4.29843 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
