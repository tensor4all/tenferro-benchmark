# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/einsum`
- Suite file: `benchmarks/gpu/einsum.yaml`
- Timestamp: `2026-06-07T01:20:58.621560+00:00`
- tenferro-rs commit: `d30e592289abf59a7d91f0ec56eaadbe864220a6`

## CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.
Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.
tenferro-rs uses native column-major GPU tensors; Torch, JAX, and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently JAX-compatible gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.

## gpu/einsum / einsum

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| einsum_bin_matmul_3072_f64 | 3.338 | 3.224 | 3.155 | 3.552 | 3.157 | 3.481 | unsupported | unsupported | unsupported |
