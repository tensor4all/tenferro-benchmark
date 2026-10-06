# CPU Performance-Issue Workloads

- Suite: `cpu/perf_issues`
- Raw run: `data/results/amd-cpu/cpu/perf_issues/20261006_185348`

## CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

## run_t1

Full metadata: `data/results/amd-cpu/cpu/perf_issues/20261006_185348/run_t1.yaml`

```yaml
timestamp: '2026-10-06T18:53:48Z'
tenferro_rs:
  path: /workspaces/t4a-2010-bench-migrate/extern/tenferro-rs
  commit: 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
  dirty: false
  features:
  - system-mkl
  resolved_path: /workspaces/t4a-2010-bench-migrate/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.1
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: 6b639a2fbdf8
  os: Linux-6.8.0-101-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: 320964928fdd481aaaf81ceae9606c0ea8df559f
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
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1
    JULIA_NUM_THREADS: '1'
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    MATMUL_NUM_THREADS: '1'
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/amd-cpu/cpu/perf_issues/20261006_185348/run_t4.yaml`

```yaml
timestamp: '2026-10-06T18:54:37Z'
tenferro_rs:
  path: /workspaces/t4a-2010-bench-migrate/extern/tenferro-rs
  commit: 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
  dirty: false
  features:
  - system-mkl
  resolved_path: /workspaces/t4a-2010-bench-migrate/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.1
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: 6b639a2fbdf8
  os: Linux-6.8.0-101-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: 320964928fdd481aaaf81ceae9606c0ea8df559f
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
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4
    JULIA_NUM_THREADS: '4'
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    MATMUL_NUM_THREADS: '4'
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/perf_issues` version 2, coverage `full`, effort `standard`
- Expected 306, selected 306, executed 306, unsupported 0, failed 0, missing 0, noisy 29.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `conj_dot_f64_pytorch-vdot_len10000@t1`, `layer_norm_f32_eager-composed_d1024_len64_b8@t1`, `layer_norm_f32_eager-single-call_d1024_len64_b8@t1`, `rms_norm_f32_eager-composed_d1024_len8_b1@t1`, `rms_norm_f32_eager-composed_d1024_len64_b8@t1`, `rms_norm_f32_eager-single-call_d1024_len64_b8@t1`, `activation_silu_f32_eager-composed_1024x64@t1`, `activation_silu_f32_pytorch_1024x64@t1`, `activation_softplus_f32_pytorch_1024x64@t1`, `softmax_f32_eager-composed_len512_b8@t1`, `log_softmax_f32_pytorch_len64_b8@t1`, `masked_softmax_f32_eager-composed_len512_b8@t1`, `decode_proj_f32_pytorch_in1024_out4096_len8@t4`, `conj_dot_f64_pytorch-vdot_len100@t4`, `conj_dot_c64_pytorch-vdot_len100@t4`, `small_solve_f64_pytorch-solve-batched_k16_b64@t4`, `tanh_chain_f32_compiled-prepared_k10_1024x64@t4`, `rms_norm_f32_eager-composed_d1024_len8_b1@t4`, `rms_norm_f32_eager-composed_d1024_len64_b8@t4`, `rms_norm_f32_eager-single-call_d1024_len64_b8@t4`, `activation_erf_f32_pytorch_1024x64@t4`, `softmax_f32_eager-single-call_len512_b8@t4`, `softmax_f32_eager-composed_len512_b8@t4`, `log_softmax_f32_session-single-call_len64_b8@t4`, `log_softmax_f32_eager-single-call_len512_b8@t4`, `log_softmax_f32_eager-composed_len512_b8@t4`, `masked_softmax_f32_eager-single-call_len512_b8@t4`, `masked_softmax_f32_eager-composed_len512_b8@t4`, `take_along_axis_rows_f64_eager-single-call_n64_b256@t4`

Numerical checks: **306/306 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; faer-direct and host-sgemm write into a preallocated output, the tenferro and PyTorch arms allocate their output. `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 2 | 1038513.5 | 1717.5 | 0.2% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 1 | 8 | 306933.8 | 15576.8 | 4.8% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 1 | 8 | 459340.0 | 52663.0 | 7.0% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 8 | 382662.5 | 10680.4 | 4.5% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1688985.0 | 33568.5 | 3.4% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1225595.0 | 23540.8 | 7.1% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1841895.5 | 32243.5 | 5.2% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1743572.0 | 5195.0 | 0.6% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 1 | 1 | 11993412.0 | 1288392.0 | 7.7% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 1 | 1 | 10989369.0 | 1965279.0 | 7.8% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 1 | 1 | 10669379.0 | 48526.5 | 1.0% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 1 | 1 | 9807041.0 | 309024.5 | 2.4% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 1 | 1 | 4098114.0 | 449430.0 | 7.5% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1481708.0 | 243443.0 | 8.6% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1672429.5 | 4207.5 | 0.3% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1627038.0 | 12240.2 | 2.5% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 1 | 1 | 5690366.0 | 393502.5 | 3.8% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 1 | 1 | 4837808.0 | 5870.0 | 2.7% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 1 | 1 | 6189842.0 | 77683.0 | 2.5% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 1 | 1 | 5939174.0 | 300590.0 | 6.1% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 1 | 1 | 43179882.0 | 2797136.0 | 4.1% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 1 | 1 | 42919673.0 | 1992519.5 | 3.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 1 | 1 | 43628896.0 | 3309328.0 | 5.5% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 1 | 1 | 39515991.0 | 428224.0 | 1.5% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 1 | 1 | 2490521.0 | 297970.5 | 7.2% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 1 | 4 | 1071090.0 | 54033.0 | 3.9% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 1 | 2048 | 1252.7 | 4.0 | 3.0% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 1 | 32768 | 78.0 | 0.3 | 2.4% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 1 | 2048 | 1367.8 | 54.9 | 4.9% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 1 | 512 | 6148.7 | 188.9 | 5.9% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 1 | 256 | 9717.9 | 15.8 | 0.1% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 1 | 1024 | 3273.2 | 16.1 | 166.0% | NOISY |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 1 | 8 | 494324.9 | 28120.9 | 3.8% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 1 | 4 | 953801.2 | 73633.6 | 5.4% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 1 | 16 | 190240.6 | 21371.0 | 6.8% | ok |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 1 | 2048 | 1277.3 | 117.6 | 7.7% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 1 | 32768 | 108.6 | 1.1 | 5.8% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 1 | 2048 | 1295.5 | 5.2 | 0.3% | ok |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 1 | 256 | 11483.3 | 658.7 | 5.5% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 1 | 256 | 9948.2 | 157.5 | 0.8% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 1 | 256 | 8084.2 | 39.2 | 0.5% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 1 | 2 | 1123952.0 | 23125.8 | 4.5% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 1 | 2 | 1095960.5 | 6667.5 | 0.8% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 1 | 4 | 777870.5 | 54263.0 | 4.3% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 8 | 348407.6 | 538.1 | 0.2% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 161234.0 | 18414.7 | 5.7% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 1 | 8 | 248526.8 | 483.2 | 4.6% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 16 | 195882.7 | 685.3 | 1.5% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 32 | 72466.7 | 1424.6 | 3.6% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 6059078.0 | 68423.5 | 3.4% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 2037577.0 | 4181.0 | 4.6% | ok |
| small_solve_f64_solve-batched_k64_b64 | #2007 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 4383903.0 | 357152.0 | 4.1% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 1 | 4429525.0 | 332651.0 | 5.4% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 2 | 1284667.0 | 2267.5 | 0.1% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 1 | 22541867.0 | 1313678.0 | 5.5% | ok |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 1 | 8054403.0 | 871668.5 | 5.7% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 220052840.0 | 7348534.0 | 1.9% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 62438660.0 | 2948881.0 | 3.7% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 435197732.0 | 12664113.0 | 2.4% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 114164248.0 | 1442002.0 | 2.8% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 32 | 69992.3 | 4272.8 | 5.0% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 512 | 5750.1 | 221.9 | 5.3% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 32 | 73433.3 | 4384.2 | 6.6% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 16 | 238540.3 | 28594.1 | 6.9% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 64 | 22309.5 | 856.6 | 6.3% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 16 | 226633.0 | 1731.6 | 3.8% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 218344.0 | 36212.1 | 8.6% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 64 | 22468.5 | 1211.1 | 8.2% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 223976.1 | 641.9 | 4.0% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 2 | 438749.0 | 5814.8 | 30.7% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 16 | 181229.6 | 1426.6 | 6.6% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 416898.5 | 11061.6 | 13.4% | NOISY |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 64 | 38162.2 | 5807.7 | 13.8% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 128 | 20164.5 | 196.5 | 6.5% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 32 | 51547.3 | 2115.4 | 7.3% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 16 | 157247.0 | 29822.2 | 8.9% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 64 | 41185.9 | 629.8 | 2.6% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 16 | 171284.3 | 1006.9 | 5.8% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 16 | 180774.0 | 2637.9 | 2.0% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 64 | 50239.3 | 1930.0 | 3.2% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 16 | 204581.1 | 15697.1 | 7.2% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 4 | 246555.5 | 131885.6 | 83.5% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 8 | 313097.8 | 32326.8 | 6.2% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 4 | 249740.8 | 29543.5 | 32.4% | NOISY |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 1 | 256 | 7432.8 | 64.8 | 5.3% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | 2 | 1170778.5 | 23948.8 | 5.0% | ok |
| activation_erf_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | 2 | 1171083.5 | 2322.5 | 0.3% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 1 | 32 | 102959.3 | 499.4 | 3.7% | ok |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 8 | 437613.1 | 67100.3 | 7.3% | ok |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 8 | 433491.6 | 45652.1 | 6.8% | ok |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 8 | 361111.8 | 20447.4 | 6.0% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 32 | 58140.3 | 614.6 | 6.7% | ok |
| activation_silu_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 8 | 495849.9 | 1068.1 | 0.6% | ok |
| activation_silu_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 8 | 466996.5 | 15960.5 | 4.0% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 8 | 382005.0 | 113325.6 | 20.2% | NOISY |
| activation_silu_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 1 | 16 | 66651.6 | 8357.5 | 37.8% | NOISY |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 4 | 756067.2 | 1152.6 | 0.2% | ok |
| activation_softplus_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 4 | 745296.8 | 554.9 | 0.1% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 4 | 728804.0 | 1401.4 | 0.1% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 1 | 8 | 218418.4 | 1253.2 | 31.5% | NOISY |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 2 | 1138467.0 | 17736.0 | 4.4% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 2 | 1185353.5 | 5027.8 | 2.3% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 64064.6 | 7165.4 | 6.8% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 2 | 1285827.0 | 119969.0 | 6.5% | ok |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 2 | 1129922.0 | 110583.5 | 6.7% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 2 | 1132357.0 | 138112.0 | 7.3% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 16 | 207842.4 | 29650.9 | 7.4% | ok |
| softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 8 | 276031.5 | 4922.0 | 6.0% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 16 | 252479.5 | 1066.9 | 1.4% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 16 | 251340.7 | 19324.7 | 6.5% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 34900.7 | 5657.1 | 8.1% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 6951887.0 | 298919.5 | 3.0% | ok |
| softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 8755396.0 | 1047539.0 | 6.2% | ok |
| softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 7766263.0 | 385827.5 | 12.2% | NOISY |
| softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 2 | 1917972.5 | 4652.8 | 3.2% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 252735.8 | 20995.3 | 5.5% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 240404.8 | 40838.2 | 8.5% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 207804.9 | 15793.2 | 5.3% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 64 | 32724.4 | 1259.4 | 16.2% | NOISY |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 7180545.0 | 833662.5 | 6.9% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 8928841.0 | 654621.5 | 4.5% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 7522245.0 | 841778.0 | 6.7% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 2 | 1813414.5 | 81755.2 | 3.6% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 8 | 284273.0 | 21215.8 | 7.4% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 8 | 293963.4 | 47808.4 | 8.3% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 8 | 266167.4 | 6368.4 | 5.2% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 32 | 90695.8 | 7712.3 | 6.7% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 8833868.0 | 717842.5 | 5.5% | ok |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 9545731.0 | 562693.5 | 8.5% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 9380906.0 | 272504.5 | 11.8% | NOISY |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 4181787.0 | 12881.0 | 2.1% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 128 | 16143.3 | 121.4 | 2.9% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 512 | 7716.2 | 1203.9 | 7.6% | ok |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 128 | 18317.8 | 128.2 | 0.5% | ok |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 9060.3 | 1310.2 | 7.4% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 7314639.0 | 94278.0 | 2.6% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 6637966.0 | 43542.0 | 2.2% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 7082742.0 | 330462.0 | 5.5% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 1 | 1 | 2822553.0 | 190601.0 | 8.5% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 2 | 1042324.0 | 1122.8 | 0.2% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 4 | 16 | 156041.4 | 3380.5 | 1.9% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 4 | 4 | 949898.5 | 2884.1 | 0.3% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 16 | 177150.2 | 13404.8 | 5.6% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1277501.5 | 40138.8 | 2.9% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 4 | 4 | 641018.2 | 24640.9 | 3.8% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1839060.0 | 5310.0 | 2.5% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 4 | 4 | 642358.5 | 11679.1 | 1.8% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 4 | 1 | 4971773.0 | 27305.5 | 0.9% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 4 | 1 | 3422661.0 | 8004.5 | 0.2% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 4 | 1 | 5114527.0 | 177916.0 | 3.2% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 4 | 1 | 3487914.0 | 100733.5 | 8.0% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 4 | 1 | 4172247.0 | 4805.5 | 0.6% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 4 | 8 | 484649.6 | 9687.8 | 2.5% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 4 | 1 | 4325801.0 | 558658.5 | 7.0% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 4 | 4 | 658296.5 | 4849.0 | 13.2% | NOISY |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 4 | 1 | 5263582.0 | 311345.5 | 3.3% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 4 | 2 | 1891236.5 | 65522.0 | 2.1% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 4 | 1 | 6693848.0 | 320345.0 | 4.3% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 4 | 1 | 2462240.0 | 10285.0 | 3.8% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 4 | 1 | 19783305.0 | 882819.0 | 3.9% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 4 | 1 | 13283204.0 | 283939.0 | 5.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 4 | 1 | 19957791.0 | 442964.0 | 5.5% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 4 | 1 | 15275048.0 | 1540350.0 | 6.0% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 4 | 2 | 1090625.5 | 5805.2 | 0.3% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 4 | 8 | 393782.9 | 22515.1 | 3.1% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 4 | 2048 | 1259.7 | 1.8 | 3.3% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 4 | 32768 | 78.1 | 0.2 | 1.1% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 4 | 1024 | 2022.5 | 781.3 | 19.1% | NOISY |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 4 | 512 | 6117.1 | 388.0 | 5.1% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 4 | 256 | 9705.6 | 11.0 | 0.2% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 4 | 512 | 4182.2 | 181.1 | 7.1% | ok |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 4 | 8 | 495382.5 | 1392.5 | 0.3% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 4 | 4 | 855850.5 | 87495.4 | 5.6% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 4 | 32 | 60674.2 | 4695.8 | 7.5% | ok |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 4 | 2048 | 1296.1 | 87.2 | 6.8% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 4 | 32768 | 108.1 | 0.2 | 3.0% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 4 | 2048 | 1217.3 | 189.0 | 190.4% | NOISY |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 4 | 256 | 11191.0 | 53.3 | 1.0% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 4 | 256 | 9790.8 | 9.7 | 0.4% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 4 | 512 | 5524.4 | 19.7 | 0.4% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 4 | 2 | 1168323.0 | 44463.5 | 2.2% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 4 | 4 | 1060369.5 | 116400.0 | 6.4% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 4 | 16 | 174417.6 | 12329.8 | 5.1% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 8 | 352884.1 | 520.6 | 0.1% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 16 | 188904.3 | 250.7 | 0.1% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 4 | 8 | 253557.0 | 1251.9 | 5.9% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 16 | 167915.4 | 1772.3 | 21.8% | NOISY |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 32 | 72534.9 | 5761.0 | 7.0% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 5698816.0 | 374366.5 | 3.2% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 2217812.0 | 9235.5 | 0.4% | ok |
| small_solve_f64_solve-batched_k64_b64 | #2007 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 4453666.0 | 19560.5 | 1.7% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 1 | 2992758.0 | 21601.0 | 2.8% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 4 | 705878.0 | 6057.8 | 2.8% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 1 | 12839518.0 | 1142777.0 | 29.4% | NOISY |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 1 | 4285890.0 | 10680.0 | 9.7% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 130218567.0 | 2338236.5 | 4.6% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 33916936.0 | 2318815.5 | 6.1% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 244728270.0 | 11967925.0 | 4.4% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 72470663.0 | 7168783.5 | 6.6% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 32 | 100358.6 | 16759.5 | 9.4% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 256 | 8598.0 | 937.8 | 5.8% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 32 | 106216.6 | 1042.5 | 1.2% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 8 | 252263.2 | 1985.1 | 8.3% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 256 | 14109.3 | 108.1 | 2.0% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 8 | 308606.4 | 2765.8 | 0.7% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 8 | 270960.1 | 5498.9 | 4.8% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 256 | 13381.6 | 71.2 | 1.5% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 8 | 305902.5 | 3299.4 | 5.9% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 2 | 647181.0 | 85652.5 | 8.4% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 32 | 54718.7 | 5644.2 | 7.9% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 2 | 659376.5 | 12932.8 | 3.8% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 32 | 42398.6 | 541.8 | 13.5% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 128 | 20987.6 | 668.7 | 3.8% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 32 | 51268.9 | 276.4 | 0.9% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 199885.9 | 1895.4 | 1.4% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 32 | 68684.4 | 226.6 | 0.3% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 217982.1 | 1260.9 | 1.1% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 16 | 182332.2 | 1334.7 | 1.1% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 32 | 67984.7 | 760.4 | 0.8% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 16 | 215783.2 | 5855.8 | 1.4% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 4 | 247540.5 | 7851.5 | 27.3% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 16 | 101260.2 | 6599.0 | 6.0% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 4 | 248498.0 | 20424.4 | 101.6% | NOISY |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 4 | 256 | 7344.4 | 69.1 | 6.1% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | 4 | 581096.2 | 1033.8 | 0.2% | ok |
| activation_erf_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | 4 | 586444.2 | 38217.6 | 9.8% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 4 | 64 | 34871.4 | 179.2 | 15.1% | NOISY |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 8 | 465686.5 | 28534.0 | 4.5% | ok |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 8 | 471311.6 | 30844.1 | 5.2% | ok |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 8 | 317105.2 | 30065.4 | 8.2% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 64 | 35919.0 | 162.1 | 6.2% | ok |
| activation_silu_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 8 | 422450.0 | 47740.9 | 7.1% | ok |
| activation_silu_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 8 | 503964.0 | 27655.2 | 4.3% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 8 | 321326.8 | 24245.1 | 4.6% | ok |
| activation_silu_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 4 | 64 | 37344.5 | 156.9 | 1.1% | ok |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 4 | 524717.2 | 21097.0 | 3.5% | ok |
| activation_softplus_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 4 | 602899.5 | 73454.8 | 8.3% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 8 | 535228.8 | 19723.8 | 2.6% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 4 | 32 | 113427.5 | 214.6 | 0.1% | ok |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 4 | 758592.2 | 74242.4 | 6.1% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 4 | 862470.5 | 29093.6 | 2.6% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 4 | 128 | 26329.1 | 136.9 | 0.9% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 4 | 827192.2 | 86464.1 | 9.1% | ok |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 2 | 870128.5 | 37149.0 | 5.7% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 4 | 803123.8 | 43797.5 | 5.4% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 32 | 63188.3 | 175.0 | 0.3% | ok |
| softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 8 | 275267.6 | 1272.6 | 0.4% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 8 | 258710.9 | 8584.6 | 3.1% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 16 | 250692.6 | 812.8 | 3.8% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 256 | 15899.1 | 61.8 | 1.6% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 1 | 3772503.0 | 166616.0 | 14.1% | NOISY |
| softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 1 | 4563628.0 | 316296.0 | 6.4% | ok |
| softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 1 | 4259209.0 | 194866.5 | 21.1% | NOISY |
| softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 4 | 590921.8 | 5917.9 | 0.8% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 8 | 275430.2 | 2122.6 | 0.8% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 8 | 253949.6 | 1003.1 | 10.8% | NOISY |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 16 | 248664.9 | 1510.1 | 0.4% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 256 | 14847.9 | 347.6 | 1.8% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2855153.0 | 710284.0 | 12.4% | NOISY |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 4621141.0 | 591114.0 | 9.1% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3694061.0 | 151560.5 | 22.8% | NOISY |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 4 | 603812.2 | 15270.5 | 1.8% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 8 | 338902.4 | 2230.1 | 1.3% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 8 | 295078.4 | 866.2 | 0.3% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 8 | 249864.4 | 773.2 | 0.3% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 77718.5 | 195.5 | 0.3% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 4196916.0 | 206301.5 | 10.6% | NOISY |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 5246061.0 | 427579.0 | 7.9% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3759422.0 | 228187.5 | 14.3% | NOISY |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1368170.0 | 15115.5 | 3.0% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 128 | 22231.4 | 228.3 | 2.2% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 11116.3 | 96.7 | 0.7% | ok |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 128 | 22394.0 | 281.2 | 1.4% | ok |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 11229.2 | 1017.2 | 5.4% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 2938226.0 | 92203.0 | 24.6% | NOISY |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 2738249.0 | 49947.0 | 1.9% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 2627236.0 | 157076.0 | 5.9% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 4 | 4 | 853057.8 | 3263.9 | 9.7% | ok |

## Per-call session entry diagnostics (not operation comparisons)

A session (or eager session) is entered and left for every operation, inside the timer.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| decode_proj_f32_eager-per-call_in1024_out1024_len8 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 2 | 1056869.5 | 1910.2 | 0.8% | ok |
| decode_proj_f32_eager-per-call_in1024_out1024_len64 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1713651.5 | 18503.2 | 0.9% | ok |
| decode_proj_f32_eager-per-call_in1024_out1024_len512 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=1024, len=512 | 1 | 1 | 10714510.0 | 1576636.0 | 8.0% | ok |
| decode_proj_f32_eager-per-call_in1024_out4096_len8 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=4096, len=8 | 1 | 1 | 3709741.0 | 422284.5 | 7.2% | ok |
| decode_proj_f32_eager-per-call_in1024_out4096_len64 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=4096, len=64 | 1 | 1 | 5628524.0 | 61527.0 | 2.7% | ok |
| decode_proj_f32_eager-per-call_in1024_out4096_len512 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=4096, len=512 | 1 | 1 | 42635864.0 | 2049687.5 | 3.3% | ok |
| small_contraction_abcd-dbef-acef_f64_per-call-default-pool_d4 | #1885 | per-call-default-pool | tenferro-rs | f64 | extent=4 | 1 | 128 | 15021.9 | 2188.7 | 7.0% | ok |
| small_contraction_abcd-dbef-acef_f64_per-call-threads1_d4 | #1885 | per-call-threads1 | tenferro-rs | f64 | extent=4 | 1 | 128 | 16638.4 | 74.6 | 0.3% | ok |
| decode_proj_f32_eager-per-call_in1024_out1024_len8 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 2 | 1074510.0 | 4675.2 | 0.3% | ok |
| decode_proj_f32_eager-per-call_in1024_out1024_len64 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1192229.0 | 20490.8 | 2.4% | ok |
| decode_proj_f32_eager-per-call_in1024_out1024_len512 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=1024, len=512 | 4 | 1 | 4911381.0 | 116738.5 | 1.3% | ok |
| decode_proj_f32_eager-per-call_in1024_out4096_len8 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=4096, len=8 | 4 | 1 | 4203557.0 | 20345.5 | 0.5% | ok |
| decode_proj_f32_eager-per-call_in1024_out4096_len64 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=4096, len=64 | 4 | 1 | 5145108.0 | 115923.5 | 3.4% | ok |
| decode_proj_f32_eager-per-call_in1024_out4096_len512 | #1992 #2003 #1995 | eager-per-call | tenferro-rs | f32 | in=1024, out=4096, len=512 | 4 | 1 | 19142375.0 | 1210065.0 | 4.5% | ok |
| small_contraction_abcd-dbef-acef_f64_per-call-default-pool_d4 | #1885 | per-call-default-pool | tenferro-rs | f64 | extent=4 | 4 | 128 | 23898.1 | 4038.3 | 8.7% | ok |
| small_contraction_abcd-dbef-acef_f64_per-call-threads1_d4 | #1885 | per-call-threads1 | tenferro-rs | f64 | extent=4 | 4 | 128 | 17548.5 | 47.7 | 0.3% | ok |

## Eager AD workflow diagnostics

Forward + reduction + backward through the eager runtime, including its internal session entries; gradients accumulate across a batch and are reset outside timing.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| eager_backward_matmul2x2_f64_leaves0 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=0 | 1 | 8 | 385148.8 | 14036.7 | 4.5% | ok |
| eager_backward_matmul2x2_f64_leaves2 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=2 | 1 | 8 | 406297.0 | 6219.6 | 0.9% | ok |
| eager_backward_matmul2x2_f64_leaves32 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=32 | 1 | 8 | 435813.0 | 8559.6 | 1.2% | ok |
| eager_backward_matmul2x2_f64_leaves128 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=128 | 1 | 4 | 519069.2 | 6531.5 | 0.9% | ok |
| eager_backward_matmul2x2_f64_leaves512 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=512 | 1 | 4 | 867803.2 | 13579.0 | 1.2% | ok |
| eager_backward_matmul2x2_f64_leaves0 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=0 | 4 | 8 | 400080.6 | 3943.2 | 0.6% | ok |
| eager_backward_matmul2x2_f64_leaves2 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=2 | 4 | 8 | 405759.5 | 10111.6 | 2.0% | ok |
| eager_backward_matmul2x2_f64_leaves32 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=32 | 4 | 8 | 423312.6 | 7404.7 | 1.3% | ok |
| eager_backward_matmul2x2_f64_leaves128 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=128 | 4 | 4 | 577738.8 | 7769.1 | 1.4% | ok |
| eager_backward_matmul2x2_f64_leaves512 | #1803 | eager-ad | tenferro-rs | f64 | unrelated_leaves=512 | 4 | 4 | 837572.2 | 26163.6 | 2.1% | ok |

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
| transpose (d,len)->(len,d) | 32768 | 13.12 | 37431 | 1.14 |
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
