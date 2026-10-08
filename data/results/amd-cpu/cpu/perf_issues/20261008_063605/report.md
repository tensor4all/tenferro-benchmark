# CPU Performance-Issue Workloads

- Suite: `cpu/perf_issues`
- Raw run: `data/results/amd-cpu/cpu/perf_issues/20261008_063605`

## CPU Information

- Model: `AMD Ryzen 9 9955HX 16-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `32`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`

## run_t1

Full metadata: `data/results/amd-cpu/cpu/perf_issues/20261008_063605/run_t1.yaml`

```yaml
timestamp: '2026-10-08T06:36:05Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
  dirty: false
  features:
  - system-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.1
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: 14099c68ff8e
  os: Linux-7.0.0-38-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    TENFERRO_CPU_BACKEND_KIND: blas
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: 2cd0fd6445dcd9aad98990f76c17e33e773c4a7d
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
    OPENBLAS_NUM_THREADS: '1'
    GOTO_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    VECLIB_MAXIMUM_THREADS: '1'
    VECLIB_NUM_THREADS: '1'
    NUMEXPR_NUM_THREADS: '1'
    BLIS_NUM_THREADS: '1'
    PJRT_NPROC: '1'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type=
      intra_op_parallelism_threads=1
    JULIA_NUM_THREADS: '1'
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    MATMUL_NUM_THREADS: '1'
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/amd-cpu/cpu/perf_issues/20261008_063605/run_t4.yaml`

```yaml
timestamp: '2026-10-08T06:36:57Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
  dirty: false
  features:
  - system-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.1
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: 14099c68ff8e
  os: Linux-7.0.0-38-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    TENFERRO_CPU_BACKEND_KIND: blas
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: 2cd0fd6445dcd9aad98990f76c17e33e773c4a7d
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
    OPENBLAS_NUM_THREADS: '4'
    GOTO_NUM_THREADS: '4'
    MKL_NUM_THREADS: '4'
    VECLIB_MAXIMUM_THREADS: '4'
    VECLIB_NUM_THREADS: '4'
    NUMEXPR_NUM_THREADS: '4'
    BLIS_NUM_THREADS: '4'
    PJRT_NPROC: '4'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type=
      intra_op_parallelism_threads=4
    JULIA_NUM_THREADS: '4'
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    MATMUL_NUM_THREADS: '4'
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/perf_issues` version 2, coverage `full`, effort `standard`
- Expected 306, selected 306, executed 306, unsupported 0, failed 0, missing 0, noisy 32.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `eager_backward_matmul2x2_f64_leaves0@t1`, `eager_backward_matmul2x2_f64_leaves2@t1`, `eager_backward_matmul2x2_f64_leaves32@t1`, `eager_backward_matmul2x2_f64_leaves128@t1`, `layer_norm_f32_eager-single-call_d1024_len8_b8@t1`, `rms_norm_f32_eager-composed_d1024_len64_b8@t1`, `small_contraction_abcd-dbef-acef_f64_per-call-default-pool_d4@t1`, `masked_softmax_f32_eager-single-call_len64_b8@t1`, `decode_proj_f32_eager-shared_in1024_out4096_len8@t4`, `decode_proj_f32_faer-direct_in1024_out4096_len64@t4`, `decode_proj_f32_host-sgemm_in1024_out4096_len512@t4`, `conj_dot_f64_pytorch-vdot_len10000@t4`, `eager_backward_matmul2x2_f64_leaves2@t4`, `eager_backward_matmul2x2_f64_leaves128@t4`, `eager_backward_matmul2x2_f64_leaves512@t4`, `tanh_chain_f32_compiled-prepared_k10_1024x64@t4`, `tanh_chain_f32_eager-shared_k10_1024x64@t4`, `layer_norm_f32_pytorch-fused_d1024_len64_b1@t4`, `layer_norm_f32_eager-single-call_d1024_len64_b1@t4`, `layer_norm_f32_eager-composed_d1024_len64_b8@t4`, `layer_norm_f32_pytorch-fused_d1024_len64_b8@t4`, `layer_norm_f32_eager-single-call_d1024_len64_b8@t4`, `rms_norm_f32_eager-composed_d1024_len64_b8@t4`, `rms_norm_f32_pytorch-fused_d1024_len64_b8@t4`, `rms_norm_f32_eager-single-call_d1024_len64_b8@t4`, `activation_sigmoid_f32_eager-single-call_1024x64@t4`, `activation_sigmoid_f32_eager-composed_1024x64@t4`, `activation_silu_f32_eager-single-call_1024x64@t4`, `softmax_f32_eager-single-call_len512_b8@t4`, `softmax_f32_pytorch_len512_b8@t4`, `log_softmax_f32_eager-composed_len512_b8@t4`, `masked_softmax_f32_eager-single-call_len512_b8@t4`

Numerical checks: **306/306 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; faer-direct and host-sgemm write into a preallocated output, the tenferro and PyTorch arms allocate their output. `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 4 | 846817.2 | 6205.4 | 0.5% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 1 | 8 | 428932.8 | 7447.2 | 1.2% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 1 | 4 | 967835.0 | 23848.6 | 2.2% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 8 | 435865.8 | 3314.5 | 0.5% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 1 | 1 | 2107138.0 | 17683.0 | 0.8% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 1 | 4 | 882992.5 | 1966.1 | 0.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1704820.5 | 25803.2 | 0.9% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 1 | 1 | 2167912.0 | 8832.5 | 1.0% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 1 | 1 | 15829257.0 | 52173.0 | 0.3% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 1 | 1 | 7028776.0 | 20614.0 | 1.4% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 1 | 1 | 8623969.0 | 25338.0 | 1.0% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 1 | 1 | 15815111.0 | 16887.0 | 0.1% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 1 | 1 | 3340741.0 | 89122.5 | 4.4% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1718255.5 | 10432.2 | 5.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 1 | 1 | 3812138.0 | 16341.5 | 0.4% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1648614.5 | 213114.2 | 6.3% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 1 | 1 | 8418793.0 | 75717.0 | 3.5% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 1 | 1 | 3522804.0 | 22963.5 | 2.8% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 1 | 1 | 6710386.0 | 121804.0 | 1.2% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 1 | 1 | 8509363.0 | 27817.5 | 0.4% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 1 | 1 | 63454859.0 | 140114.0 | 0.2% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 1 | 1 | 28081890.0 | 117501.0 | 0.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 1 | 1 | 34289008.0 | 106685.5 | 0.2% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 1 | 1 | 63740538.0 | 107242.0 | 0.2% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 1 | 1 | 4063451.0 | 39008.5 | 2.9% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 1 | 2 | 953570.5 | 2099.2 | 7.0% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 1 | 2048 | 1318.4 | 9.8 | 4.9% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 1 | 65536 | 45.5 | 0.1 | 6.0% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 1 | 2048 | 1323.2 | 7.7 | 6.3% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 1 | 512 | 5247.8 | 8.8 | 0.2% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 1 | 256 | 7953.8 | 14.4 | 4.2% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 1 | 1024 | 3453.6 | 25.4 | 1.3% | ok |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 1 | 8 | 428458.1 | 2138.4 | 0.4% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 1 | 4 | 820703.0 | 2847.9 | 2.1% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 1 | 8 | 280018.6 | 5518.6 | 2.9% | ok |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 1 | 2048 | 1397.9 | 34.8 | 3.4% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 1 | 32768 | 102.8 | 0.1 | 0.6% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 1 | 2048 | 1253.6 | 5.4 | 5.5% | ok |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 1 | 256 | 9351.5 | 37.8 | 5.9% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 1 | 256 | 10230.6 | 12.6 | 0.2% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 1 | 512 | 7484.1 | 16.8 | 0.3% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 1 | 4 | 868958.8 | 32593.8 | 2.3% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 1 | 2 | 1040304.0 | 14960.5 | 2.8% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 1 | 4 | 723489.8 | 8787.8 | 3.0% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 8 | 301045.6 | 2589.3 | 4.2% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 139261.6 | 456.8 | 8.3% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 204644.8 | 500.7 | 3.3% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 16 | 184918.9 | 1030.7 | 6.1% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 32 | 73751.7 | 165.6 | 0.3% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 4884568.0 | 17864.0 | 0.3% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 2 | 1718376.0 | 9485.5 | 4.7% | ok |
| small_solve_f64_solve-batched_k64_b64 | #2007 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 4483042.0 | 35717.0 | 3.8% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 1 | 4729175.0 | 15403.5 | 0.2% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 2 | 1454864.5 | 5094.2 | 0.3% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 1 | 29532721.0 | 31951.0 | 0.5% | ok |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 1 | 7135256.0 | 14984.0 | 0.2% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 312445793.0 | 1094867.0 | 0.3% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 57821161.0 | 197297.0 | 0.7% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 610082670.0 | 658740.5 | 0.1% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 109331688.0 | 99332.0 | 0.1% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 32 | 73324.3 | 3938.4 | 4.9% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 512 | 6357.5 | 85.4 | 3.4% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 32 | 76122.1 | 3932.7 | 7.9% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 16 | 148006.2 | 1423.0 | 8.5% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 128 | 26924.1 | 343.8 | 1.8% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 16 | 157040.1 | 2817.2 | 11.4% | NOISY |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 148134.6 | 877.9 | 0.5% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 128 | 27134.9 | 270.8 | 1.9% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 156733.9 | 2495.9 | 8.2% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 442806.2 | 3912.1 | 0.8% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 16 | 194851.9 | 1044.5 | 0.5% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 841119.0 | 39240.1 | 8.2% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 64 | 37090.5 | 284.2 | 5.7% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 256 | 14378.8 | 59.9 | 1.1% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 64 | 44063.8 | 408.1 | 3.4% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 32 | 93675.2 | 906.7 | 0.8% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 64 | 32519.8 | 139.6 | 0.5% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 32 | 102870.6 | 1458.2 | 9.5% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 32 | 95864.0 | 1063.4 | 3.7% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 64 | 32561.6 | 301.9 | 6.7% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 16 | 107714.7 | 890.4 | 0.9% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 8 | 430592.0 | 6223.0 | 21.1% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 16 | 249851.1 | 7304.4 | 2.2% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 8 | 524586.6 | 7842.9 | 1.8% | ok |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 1 | 256 | 7241.4 | 30.3 | 0.4% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | 2 | 1069659.5 | 4548.2 | 0.3% | ok |
| activation_erf_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | 2 | 1066152.5 | 2537.8 | 2.4% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 1 | 32 | 110847.5 | 258.8 | 0.3% | ok |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 8 | 312594.9 | 2759.4 | 3.3% | ok |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 8 | 292717.5 | 7371.4 | 2.2% | ok |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 16 | 219479.5 | 2285.2 | 5.8% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 128 | 24340.4 | 121.2 | 0.4% | ok |
| activation_silu_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 8 | 314767.8 | 1120.9 | 3.4% | ok |
| activation_silu_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 8 | 306131.5 | 6918.7 | 2.2% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 16 | 239102.1 | 39047.0 | 9.3% | ok |
| activation_silu_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 1 | 128 | 25007.7 | 156.7 | 0.7% | ok |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 4 | 606281.5 | 3067.1 | 5.8% | ok |
| activation_softplus_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 4 | 656669.0 | 3020.6 | 0.6% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 4 | 591406.2 | 1322.4 | 0.2% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 1 | 32 | 115225.4 | 288.3 | 3.4% | ok |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 2 | 1034538.0 | 5217.0 | 8.4% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 2 | 1086486.0 | 5640.8 | 2.3% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 1 | 128 | 27255.4 | 271.2 | 0.9% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 2 | 1054125.0 | 6797.8 | 0.5% | ok |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 2 | 1138884.5 | 15236.2 | 3.6% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 2 | 1076692.5 | 1197.5 | 0.6% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 16 | 129421.2 | 892.7 | 4.7% | ok |
| softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 16 | 203191.4 | 1318.7 | 6.0% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 16 | 180652.0 | 947.7 | 0.8% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 16 | 178291.4 | 1676.0 | 8.1% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 128 | 30060.0 | 179.1 | 5.3% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 6448233.0 | 201855.5 | 1.6% | ok |
| softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 8177338.0 | 38192.5 | 0.4% | ok |
| softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 6462268.0 | 67211.5 | 6.7% | ok |
| softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 2 | 1485983.0 | 10281.8 | 0.5% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 199419.9 | 1152.5 | 0.5% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 184216.9 | 7345.4 | 6.0% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 179901.3 | 1104.6 | 4.5% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 128 | 21463.2 | 133.8 | 0.8% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 6268073.0 | 187122.0 | 1.6% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 8149576.0 | 217349.5 | 1.5% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 6525899.0 | 53214.5 | 0.9% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 2 | 1223098.0 | 7902.5 | 1.4% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 8 | 253677.8 | 4842.2 | 12.3% | NOISY |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 225845.2 | 654.7 | 1.7% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 209401.2 | 2043.2 | 4.6% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 32 | 69504.7 | 915.0 | 0.8% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 7777114.0 | 66245.0 | 1.5% | ok |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 9180858.0 | 104196.5 | 0.8% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 7765002.0 | 190874.5 | 5.2% | ok |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3470645.0 | 103169.0 | 2.9% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 128 | 15358.5 | 73.5 | 0.3% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 512 | 7095.8 | 30.5 | 4.3% | ok |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 15040.4 | 62.6 | 0.3% | ok |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 1 | 512 | 6756.3 | 75.1 | 7.0% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 7082637.0 | 181156.0 | 3.1% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 6738499.0 | 65664.0 | 1.2% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 6823069.0 | 88782.5 | 0.8% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 1 | 1 | 3583067.0 | 13365.5 | 0.4% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 4 | 843586.0 | 3390.0 | 1.9% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 4 | 16 | 213299.1 | 9986.6 | 2.9% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 4 | 4 | 828435.0 | 4271.6 | 1.4% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 16 | 159555.4 | 1625.6 | 1.2% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1144134.5 | 29312.8 | 5.5% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 4 | 4 | 549950.8 | 32611.5 | 4.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1239519.0 | 65019.8 | 4.1% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 4 | 4 | 750403.0 | 8836.6 | 7.9% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 4 | 1 | 5617698.0 | 26324.5 | 0.9% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 4 | 1 | 2016226.0 | 18895.0 | 1.9% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 4 | 1 | 3038311.0 | 24185.5 | 4.4% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 4 | 1 | 5501720.0 | 23484.0 | 0.3% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 4 | 1 | 3351942.0 | 86959.0 | 11.6% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 4 | 4 | 550338.8 | 14253.2 | 2.0% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 4 | 1 | 3549915.0 | 22397.0 | 0.4% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 4 | 4 | 610684.8 | 4601.1 | 1.2% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 4 | 1 | 4857737.0 | 390154.0 | 4.7% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 4 | 2 | 1248596.0 | 68096.0 | 20.1% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 4 | 1 | 4876182.0 | 32435.5 | 1.1% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 4 | 1 | 2905962.0 | 25022.0 | 0.8% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 4 | 1 | 22567937.0 | 50940.5 | 0.3% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 4 | 1 | 7508909.0 | 231926.5 | 3.9% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 4 | 1 | 11909446.0 | 765300.5 | 11.4% | NOISY |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 4 | 1 | 21511748.0 | 325889.0 | 2.3% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 4 | 2 | 1458842.0 | 14126.5 | 3.9% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 4 | 4 | 347123.8 | 4339.4 | 8.8% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 4 | 2048 | 1293.8 | 4.8 | 5.9% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 4 | 65536 | 45.6 | 0.1 | 2.4% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 4 | 2048 | 1349.5 | 4.7 | 0.6% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 4 | 512 | 5274.6 | 14.2 | 0.3% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 4 | 256 | 7973.2 | 43.9 | 6.4% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 4 | 1024 | 2735.1 | 1026.9 | 16.3% | NOISY |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 4 | 8 | 407325.8 | 24996.4 | 5.9% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 4 | 4 | 820417.2 | 4442.0 | 0.4% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 4 | 32 | 69691.5 | 536.3 | 0.6% | ok |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 4 | 2048 | 1396.6 | 7.3 | 3.8% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 4 | 32768 | 102.8 | 0.5 | 5.3% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 4 | 2048 | 1264.3 | 9.4 | 0.8% | ok |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 4 | 256 | 9342.4 | 46.8 | 5.8% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 4 | 256 | 10264.4 | 61.9 | 0.4% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 4 | 1024 | 3644.0 | 13.3 | 1.0% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 4 | 4 | 876941.2 | 30860.6 | 2.0% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 4 | 2 | 1121441.5 | 14472.2 | 5.0% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 4 | 16 | 186661.5 | 1939.3 | 2.6% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 8 | 301567.8 | 1888.6 | 4.4% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 16 | 141776.4 | 1211.4 | 1.9% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 4 | 16 | 216907.8 | 1660.7 | 1.1% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 16 | 143132.6 | 967.1 | 0.4% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 32 | 74093.9 | 462.9 | 1.0% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 3525018.0 | 22598.5 | 1.2% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 2 | 1258995.5 | 13057.5 | 2.4% | ok |
| small_solve_f64_solve-batched_k64_b64 | #2007 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 3627561.0 | 70253.0 | 7.4% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 2 | 1779245.5 | 32140.2 | 2.0% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 8 | 439387.4 | 1439.6 | 2.6% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 1 | 15370163.0 | 2049034.0 | 11.6% | NOISY |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 1 | 3736486.0 | 339113.5 | 13.9% | NOISY |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 157529427.0 | 234872.0 | 0.2% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 30132101.0 | 653866.0 | 1.5% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 308630118.0 | 736667.0 | 0.2% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 55972219.0 | 1134160.5 | 1.5% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 32 | 71238.8 | 467.1 | 2.6% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 512 | 5137.2 | 40.1 | 0.9% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 32 | 80415.8 | 3927.7 | 4.0% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 16 | 177707.8 | 3891.1 | 7.1% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 256 | 12024.3 | 218.6 | 6.4% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 16 | 180367.8 | 2558.9 | 8.7% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 16 | 162407.1 | 2329.1 | 7.3% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 256 | 12217.5 | 755.8 | 12.9% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 16 | 163243.6 | 1416.7 | 10.3% | NOISY |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 4 | 784084.0 | 352883.2 | 28.0% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 32 | 68028.4 | 2434.9 | 14.7% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 4 | 584630.8 | 4828.9 | 17.6% | NOISY |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 64 | 37657.0 | 1248.2 | 2.6% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 128 | 14421.5 | 165.0 | 7.1% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 64 | 43862.8 | 368.7 | 2.2% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 32 | 98556.8 | 948.7 | 1.7% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 64 | 35688.9 | 534.9 | 6.6% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 114240.8 | 1272.4 | 1.4% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 32 | 103370.0 | 2701.5 | 3.6% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 64 | 34995.9 | 720.7 | 1.7% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 32 | 108916.3 | 443.7 | 0.5% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 8 | 463055.8 | 10741.5 | 11.6% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 16 | 81385.1 | 46251.8 | 31.0% | NOISY |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 8 | 564138.6 | 61398.8 | 19.6% | NOISY |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 4 | 512 | 7350.0 | 18.0 | 1.0% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | 4 | 542466.5 | 4457.1 | 4.6% | ok |
| activation_erf_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | 4 | 542226.0 | 3341.2 | 3.0% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 4 | 128 | 30035.5 | 164.8 | 4.0% | ok |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 8 | 254841.1 | 9780.3 | 15.0% | NOISY |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 8 | 239139.0 | 27821.8 | 8.0% | ok |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 16 | 170280.6 | 25718.8 | 15.1% | NOISY |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 128 | 13788.9 | 112.0 | 2.8% | ok |
| activation_silu_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 8 | 242372.6 | 13983.2 | 10.1% | NOISY |
| activation_silu_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 16 | 219623.6 | 3981.8 | 2.4% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 16 | 187432.3 | 12490.7 | 6.0% | ok |
| activation_silu_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 4 | 128 | 14466.7 | 270.8 | 1.5% | ok |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 8 | 439681.8 | 41113.0 | 8.5% | ok |
| activation_softplus_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 8 | 385763.9 | 20998.2 | 3.6% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 8 | 374564.0 | 30642.1 | 8.6% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 4 | 64 | 58869.1 | 197.3 | 0.6% | ok |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 4 | 652070.2 | 33524.5 | 4.4% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 4 | 621405.2 | 6600.0 | 1.6% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 4 | 128 | 14654.5 | 120.8 | 0.9% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 4 | 626008.8 | 50501.2 | 4.3% | ok |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 4 | 697794.0 | 22448.4 | 1.8% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 4 | 600528.5 | 117349.8 | 9.2% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 64 | 35769.4 | 64.6 | 5.8% | ok |
| softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 16 | 202646.0 | 702.9 | 9.7% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 16 | 182313.3 | 6451.5 | 7.4% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 16 | 178526.8 | 343.8 | 1.0% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 256 | 9545.9 | 1149.4 | 6.8% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 1 | 1992552.0 | 366475.0 | 26.2% | NOISY |
| softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 1 | 2501611.0 | 238910.0 | 8.9% | ok |
| softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 2052420.0 | 80589.0 | 3.5% | ok |
| softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 4 | 375096.2 | 58887.2 | 18.7% | NOISY |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 16 | 201096.1 | 3569.8 | 3.1% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 16 | 182464.2 | 1482.5 | 0.9% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 16 | 178699.0 | 1982.8 | 3.0% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 256 | 8303.0 | 216.4 | 2.4% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 1967345.0 | 282772.0 | 7.7% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2192809.0 | 132579.5 | 7.5% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 1970160.0 | 1215740.0 | 28.5% | NOISY |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 8 | 582975.4 | 47012.9 | 4.5% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 8 | 255474.8 | 1538.6 | 2.7% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 16 | 228595.4 | 2220.7 | 2.0% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 16 | 209202.1 | 1278.0 | 0.4% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 61507.4 | 1014.1 | 2.3% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2194562.0 | 326765.0 | 10.7% | NOISY |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2688072.0 | 113368.5 | 3.8% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2551104.0 | 80812.5 | 4.9% | ok |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 2 | 992083.0 | 57060.0 | 5.9% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 128 | 17510.2 | 76.2 | 1.4% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 8365.1 | 145.1 | 1.0% | ok |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 128 | 21012.5 | 3323.2 | 9.8% | ok |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 4 | 512 | 6185.3 | 73.5 | 1.7% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 2326882.0 | 35631.5 | 6.0% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 2130532.0 | 102107.5 | 7.8% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 2165648.0 | 157812.0 | 9.4% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 4 | 2 | 993751.5 | 11947.2 | 4.1% | ok |

## Per-call session entry diagnostics (not operation comparisons)

A session (or eager session) is entered and left for every operation, inside the timer.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| decode_proj_f32_eager-per-call_in1024_out1024_len8 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 4 | 850060.8 | 9560.5 | 6.2% | ok |
| decode_proj_f32_eager-per-call_in1024_out1024_len64 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=1024, len=64 | 1 | 1 | 2134509.0 | 24536.5 | 7.0% | ok |
| decode_proj_f32_eager-per-call_in1024_out1024_len512 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=1024, len=512 | 1 | 1 | 15868972.0 | 42125.0 | 0.2% | ok |
| decode_proj_f32_eager-per-call_in1024_out4096_len8 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=4096, len=8 | 1 | 1 | 3389493.0 | 14543.5 | 0.5% | ok |
| decode_proj_f32_eager-per-call_in1024_out4096_len64 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=4096, len=64 | 1 | 1 | 8519002.0 | 16701.0 | 0.2% | ok |
| decode_proj_f32_eager-per-call_in1024_out4096_len512 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=4096, len=512 | 1 | 1 | 63669634.0 | 170681.5 | 0.2% | ok |
| small_contraction_abcd-dbef-acef_f64_per-call-default-pool_d4 | #1885 | per-call-default-pool | tenferro-rs | f64 | extent=4 | 1 | 256 | 13215.2 | 2387.6 | 10.4% | NOISY |
| small_contraction_abcd-dbef-acef_f64_per-call-threads1_d4 | #1885 | per-call-threads1 | tenferro-rs | f64 | extent=4 | 1 | 128 | 10806.0 | 29.1 | 10.0% | ok |
| decode_proj_f32_eager-per-call_in1024_out1024_len8 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 4 | 855215.5 | 10736.5 | 1.2% | ok |
| decode_proj_f32_eager-per-call_in1024_out1024_len64 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1260969.5 | 8801.2 | 1.2% | ok |
| decode_proj_f32_eager-per-call_in1024_out1024_len512 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=1024, len=512 | 4 | 1 | 5917473.0 | 93771.0 | 1.6% | ok |
| decode_proj_f32_eager-per-call_in1024_out4096_len8 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=4096, len=8 | 4 | 1 | 3388240.0 | 160242.0 | 3.2% | ok |
| decode_proj_f32_eager-per-call_in1024_out4096_len64 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=4096, len=64 | 4 | 1 | 4616072.0 | 37611.0 | 2.4% | ok |
| decode_proj_f32_eager-per-call_in1024_out4096_len512 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=4096, len=512 | 4 | 1 | 22975725.0 | 150763.5 | 0.8% | ok |
| small_contraction_abcd-dbef-acef_f64_per-call-default-pool_d4 | #1885 | per-call-default-pool | tenferro-rs | f64 | extent=4 | 4 | 128 | 15799.1 | 830.9 | 4.3% | ok |
| small_contraction_abcd-dbef-acef_f64_per-call-threads1_d4 | #1885 | per-call-threads1 | tenferro-rs | f64 | extent=4 | 4 | 256 | 10813.8 | 64.3 | 3.8% | ok |

## Eager AD workflow diagnostics

Forward + reduction + backward through the eager runtime, including its internal session entries; gradients accumulate across a batch and are reset outside timing.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| eager_backward_matmul2x2_f64_leaves0 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=0 | 1 | 16 | 245274.4 | 73342.5 | 18.6% | NOISY |
| eager_backward_matmul2x2_f64_leaves2 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=2 | 1 | 16 | 163698.2 | 39817.2 | 24.2% | NOISY |
| eager_backward_matmul2x2_f64_leaves32 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=32 | 1 | 16 | 280857.6 | 103550.6 | 22.1% | NOISY |
| eager_backward_matmul2x2_f64_leaves128 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=128 | 1 | 8 | 303226.0 | 126528.9 | 24.2% | NOISY |
| eager_backward_matmul2x2_f64_leaves512 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=512 | 1 | 4 | 668571.5 | 4867.9 | 4.2% | ok |
| eager_backward_matmul2x2_f64_leaves0 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=0 | 4 | 16 | 181590.8 | 10095.9 | 6.6% | ok |
| eager_backward_matmul2x2_f64_leaves2 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=2 | 4 | 8 | 183668.4 | 17924.9 | 20.6% | NOISY |
| eager_backward_matmul2x2_f64_leaves32 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=32 | 4 | 16 | 205064.2 | 7610.6 | 5.3% | ok |
| eager_backward_matmul2x2_f64_leaves128 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=128 | 4 | 8 | 264941.4 | 119329.6 | 22.1% | NOISY |
| eager_backward_matmul2x2_f64_leaves512 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=512 | 4 | 4 | 454683.5 | 32836.8 | 15.3% | NOISY |

## Counter diagnostics (no timing)

### decode_block_copy_volume_f32_d1024_len8 (#1995, 1 threads)

Source: counting global allocator in this process, per eager op including its own session entry. Unavailable: per-op layout-copy count, copied bytes (pooled buffer reuse is invisible to the host allocator).

| Op | Output bytes | Host allocations/op | Allocated bytes/op | Allocated/output |
|---|---:|---:|---:|---:|
| broadcast_in_dim (d)->(d,len) | 32768 | 15.06 | 8698 | 0.27 |
| reshape (d,len)->(d,len,1) | 32768 | 13.06 | 37208 | 1.14 |
| reshape (d,len)->(d*len) | 32768 | 13.06 | 37096 | 1.13 |
| transpose (d,len)->(len,d) | 32768 | 13.12 | 37431 | 1.14 |
| add (d,len)+(d,len) | 32768 | 19.06 | 42216 | 1.29 |
| mul (d,len)*(d,len) | 32768 | 19.06 | 42216 | 1.29 |
| reduce_sum_squares axis 0 | 32 | 29.06 | 5240 | 163.75 |
| rsqrt (len) | 32 | 14.12 | 4903 | 153.22 |
| dot_general projection (d,len)x(d,d) | 32768 | 20.06 | 42200 | 1.29 |

### decode_block_copy_volume_f32_d1024_len8 (#1995, 4 threads)

Source: counting global allocator in this process, per eager op including its own session entry. Unavailable: per-op layout-copy count, copied bytes (pooled buffer reuse is invisible to the host allocator).

| Op | Output bytes | Host allocations/op | Allocated bytes/op | Allocated/output |
|---|---:|---:|---:|---:|
| broadcast_in_dim (d)->(d,len) | 32768 | 15.19 | 8702 | 0.27 |
| reshape (d,len)->(d,len,1) | 32768 | 13.06 | 37208 | 1.14 |
| reshape (d,len)->(d*len) | 32768 | 13.06 | 37096 | 1.13 |
| transpose (d,len)->(len,d) | 32768 | 13.19 | 37433 | 1.14 |
| add (d,len)+(d,len) | 32768 | 19.06 | 42216 | 1.29 |
| mul (d,len)*(d,len) | 32768 | 19.06 | 42216 | 1.29 |
| reduce_sum_squares axis 0 | 32 | 29.06 | 5240 | 163.75 |
| rsqrt (len) | 32 | 14.12 | 4903 | 153.22 |
| dot_general projection (d,len)x(d,d) | 32768 | 20.06 | 42200 | 1.29 |


## Timing boundaries

- **activation / eager-composed** — inside: hand-written formula from pre-B1 eager primitives, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **activation / eager-single-call** — inside: one activation call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **activation / pytorch** — inside: torch gelu_tanh, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **activation / session-single-call** — inside: one activation call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, with_backend_session entry (CpuBackend::new()), correctness_check.
- **backward_live_leaves / eager-ad** — inside: eager matmul forward, reduce_sum, backward (gradients accumulate across a batch), internal eager session entries; outside: eager_runtime_construction, leaf_construction (including the unrelated leaves), correctness_check, gradient reset between cases.
- **composed_norm / eager-composed** — inside: eager composition (reduce_sum, scale_real, broadcast_in_dim, sub/add/mul, reduce_sum_squares, rsqrt), intermediate and output allocation; outside: eager_runtime_construction, input/weight/bias/eps construction, eager session entry, correctness_check.
- **composed_norm / eager-single-call** — inside: one EagerSession layer_norm / rms_norm call (composite), intermediate and output allocation; outside: eager_runtime_construction, input/weight/bias/eps construction, eager session entry, correctness_check.
- **composed_norm / pytorch-fused** — inside: torch.nn.functional.rms_norm (fused), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **copy_volume / eager-counters** — inside: —; outside: everything (counter diagnostic, no timing).
- **decode_projection / eager-per-call** — inside: eager_session_entry_exit, eager_dot_general, output_allocation; outside: backend_construction, input_construction, correctness_check.
- **decode_projection / eager-shared** — inside: eager_dot_general, output_allocation; outside: backend_construction, input_construction, correctness_check, eager_session_entry_exit.
- **decode_projection / faer-direct** — inside: faer_matmul_into_preallocated_output; outside: backend_construction, input_construction, correctness_check, output_allocation.
- **decode_projection / host-sgemm** — inside: matrixmultiply_sgemm_into_preallocated_output; outside: backend_construction, input_construction, correctness_check, output_allocation.
- **decode_projection / pytorch** — inside: torch.nn.functional.linear, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **gemm_mm256 / dot-general-shared** — inside: dot_general_read, output_allocation; outside: backend_construction (CpuBackend::with_threads(1)), input_construction, session_entry_exit, correctness_check.
- **rank1_dot / dot-general-rank1** — inside: dot_general_read (rank-1, rank-0 tensor output), output_allocation; outside: backend_construction, input_construction, operand_conjugation, session_entry_exit, correctness_check.
- **rank1_dot / portable-loop** — inside: sequential Rust loop over borrowed slices, scalar result; outside: input_construction, correctness_check.
- **rank1_dot / pytorch-vdot** — inside: torch.vdot (BLAS ddot/zdotc), scalar tensor allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **reduce_mean / eager-composed** — inside: reduce_sum + scale_real, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **reduce_mean / eager-single-call** — inside: one reduce_mean call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **reduce_mean / pytorch** — inside: torch.mean, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **reduce_mean / session-single-call** — inside: one reduce_mean call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, with_backend_session entry (CpuBackend::new()), correctness_check.
- **small_contraction_pool_hop / per-call-default-pool** — inside: with_backend_session entry/exit per call, dot_general_read, output_allocation; outside: backend_construction, input_construction, correctness_check.
- **small_contraction_pool_hop / per-call-threads1** — inside: with_backend_session entry/exit per call, dot_general_read, output_allocation; outside: backend_construction, input_construction, correctness_check.
- **small_contraction_pool_hop / shared-session-default-pool** — inside: dot_general_read, output_allocation; outside: backend_construction, input_construction, correctness_check.
- **small_solves / pytorch-solve-batched** — inside: torch.linalg.solve on (batch, k, k), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_solves / pytorch-solve-triangular-batched** — inside: torch.linalg.solve_triangular on (batch, k, k), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_solves / solve-batched** — inside: one batched solve call ([k, k, batch]), output_allocation; outside: backend_construction, input_construction (items pre-sliced), session_entry_exit, correctness_check.
- **small_solves / solve-loop** — inside: batch x rank-2 solve calls, output_allocation per item; outside: backend_construction, input_construction (items pre-sliced), session_entry_exit, correctness_check.
- **small_solves / triangular-solve-loop** — inside: batch x rank-2 solve calls, output_allocation per item; outside: backend_construction, input_construction (items pre-sliced), session_entry_exit, correctness_check.
- **softmax / eager-composed** — inside: max-subtracted composition (reduce_max, broadcast_in_dim, sub, exp, reduce_sum, div/log; select when masked), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **softmax / eager-single-call** — inside: one softmax-family call (composite, all-masked guard included), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **softmax / pytorch** — inside: torch masked_softmax, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **softmax / session-single-call** — inside: one softmax-family call (composite, all-masked guard included), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, with_backend_session entry (CpuBackend::new()), correctness_check.
- **take_along_axis / eager-gather-prebuilt** — inside: gather with prebuilt (row, batch) index tuples, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **take_along_axis / eager-single-call** — inside: one take_along_axis call (index tuples built inside, then gather), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **take_along_axis / pytorch** — inside: torch.take_along_dim, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **take_along_axis / session-single-call** — inside: one take_along_axis call (index tuples built inside, then gather), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, with_backend_session entry (CpuBackend::new()), correctness_check.
- **tanh_chain / compiled-prepared** — inside: run_prepared (runtime-internal session entry), fused region execution, output_allocation; outside: backend/runtime construction, trace and compile, prepare_compiled, input_construction, eager session entry (eager arm), correctness_check.
- **tanh_chain / eager-shared** — inside: chain of eager tanh ops, intermediate and output allocation; outside: backend/runtime construction, trace and compile, prepare_compiled, input_construction, eager session entry (eager arm), correctness_check.
