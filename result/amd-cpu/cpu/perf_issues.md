# CPU Performance-Issue Workloads

- Suite: `cpu/perf_issues`
- Raw run: `data/results/amd-cpu/cpu/perf_issues/20261105_194552`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

## run_t1

Full metadata: `data/results/amd-cpu/cpu/perf_issues/20261105_194552/run_t1.yaml`

```yaml
timestamp: '2026-11-05T19:45:52Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: a000e5c4929d5bebb91e783c329b27106baed12e
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.2
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  container: devcontainer
  hostname: ee95e5e52160
  os: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-3
    l3_domains:
    - 0-3
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: ce2efd37d9e132703dfd2ebefbf4353249cf53ea
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
    OPENBLAS_NUM_THREADS: '1'
    MATMUL_NUM_THREADS: '1'
    GOTO_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    MKL_DYNAMIC: 'FALSE'
    VECLIB_MAXIMUM_THREADS: '1'
    VECLIB_NUM_THREADS: '1'
    NUMEXPR_NUM_THREADS: '1'
    BLIS_NUM_THREADS: '1'
    PJRT_NPROC: '1'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type=
      intra_op_parallelism_threads=1
    JULIA_NUM_THREADS: '1'
    PUBLICATION_GATE_PROFILE: quick
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/amd-cpu/cpu/perf_issues/20261105_194552/run_t4.yaml`

```yaml
timestamp: '2026-11-05T19:53:54Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: a000e5c4929d5bebb91e783c329b27106baed12e
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.2
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  container: devcontainer
  hostname: ee95e5e52160
  os: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '4'
    measured_cpus: 0-3
    l3_domains:
    - 0-3
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: ce2efd37d9e132703dfd2ebefbf4353249cf53ea
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
    OPENBLAS_NUM_THREADS: '4'
    MATMUL_NUM_THREADS: '4'
    GOTO_NUM_THREADS: '4'
    MKL_NUM_THREADS: '4'
    MKL_DYNAMIC: 'FALSE'
    VECLIB_MAXIMUM_THREADS: '4'
    VECLIB_NUM_THREADS: '4'
    NUMEXPR_NUM_THREADS: '4'
    BLIS_NUM_THREADS: '4'
    PJRT_NPROC: '4'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type=
      intra_op_parallelism_threads=4
    JULIA_NUM_THREADS: '4'
    PUBLICATION_GATE_PROFILE: quick
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/perf_issues` version 21, coverage `quick`, effort `standard`
- Expected 622, selected 622, executed 622, unsupported 0, failed 0, skipped 0, missing 0, noisy 79.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 622 | 622 | 622 | 0 | 0 | 0 | 0 | 79 |

- Noisy: `decode_proj_f32_pytorch_in1024_out4096_len8@t1`, `conj_dot_c64_pytorch-vdot_len100@t1`, `conj_dot_c64_dot-general-rank1_len10000@t1`, `small_solve_f64_triangular-solve-loop_k64_b64@t1`, `layer_norm_f32_pytorch-fused_d1024_len8_b1@t1`, `activation_sigmoid_f32_session-single-call_1024x64@t1`, `metadata_transpose_view_f64_rank8_t1@t1`, `metadata_transpose_view_f64_rank8_t1_julia-base@t1`, `followup_metadata_transpose_view_metadata@t1`, `followup_metadata_slice_view_julia-base@t1`, `degenerate_dot_outer_f64_dot-general-shared_n1024@t1`, `degenerate_dot_rank0_f64_pytorch_n1024@t1`, `erased_construct_f64_col_n8@t1`, `erased_construct_f64_row_n8@t1`, `decode_proj_f32_faer-direct_in1024_out1024_len8@t4`, `decode_proj_f32_pytorch_in1024_out4096_len8@t4`, `decode_proj_f32_eager-shared_in1024_out4096_len64@t4`, `decode_proj_f32_eager-shared_in1024_out4096_len512@t4`, `decode_proj_f32_faer-direct_in1024_out4096_len512@t4`, `decode_proj_f32_host-sgemm_in1024_out4096_len512@t4`, `gemm_c64_mm256_t1@t4`, `conj_dot_f64_pytorch-vdot_len1000000@t4`, `conj_dot_c64_pytorch-vdot_len100@t4`, `conj_dot_c64_pytorch-vdot_len10000@t4`, `conj_dot_c64_pytorch-vdot_len1000000@t4`, `small_solve_f64_solve-loop_k64_b64@t4`, `small_solve_f64_triangular-solve-loop_k64_b64@t4`, `small_solve_f64_solve-batched_k64_b64@t4`, `tanh_chain_f32_eager-shared_k10_1024x64@t4`, `tanh_chain_f32_compiled-prepared_k100_1024x64@t4`, `layer_norm_f32_eager-composed_d1024_len8_b1@t4`, `layer_norm_f32_eager-single-call_d1024_len8_b1@t4`, `rms_norm_f32_eager-composed_d1024_len8_b1@t4`, `rms_norm_f32_eager-single-call_d1024_len8_b1@t4`, `rms_norm_f32_pytorch-fused_d1024_len64_b1@t4`, `rms_norm_f32_eager-composed_d1024_len64_b8@t4`, `activation_erf_f32_eager-single-call_1024x64@t4`, `activation_erf_f32_session-single-call_1024x64@t4`, `activation_erf_f32_pytorch_1024x64@t4`, `activation_sigmoid_f32_eager-composed_1024x64@t4` (+39 more)

Numerical checks: **622/622 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; faer-direct and host-sgemm write into a preallocated output, the tenferro and PyTorch arms allocate their output. `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

### Compute- and bandwidth-bound operations

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| softmax_f64_pool_serial_1024x4096 | #2273 #2274 #2276 | serial | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 17571173.0 | 29996.5 | 0.2% | ok |
| softmax_f64_pool_pytorch_t1_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 25686849.0 | 396675.5 | 1.1% | ok |
| softmax_f64_pool_pool4_1024x4096 | #2273 #2274 #2276 | pool4 | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 6738395.0 | 44363.5 | 1.7% | ok |
| softmax_f64_pool_pytorch_t4_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 9102133.0 | 333460.5 | 2.3% | ok |

### Latency and per-call overhead

These rows measure dispatch/operation latency; they do not measure compute throughput.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 2 | 1000335.5 | 6820.2 | 0.6% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 1 | 8 | 350543.4 | 4137.8 | 0.7% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 1 | 8 | 485632.9 | 6390.7 | 0.9% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 8 | 454390.5 | 12437.1 | 2.2% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1632380.0 | 13285.0 | 0.8% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1405459.5 | 7870.0 | 0.5% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1786964.5 | 14930.8 | 0.6% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1689637.5 | 12884.0 | 1.0% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 1 | 1 | 11799769.0 | 56240.0 | 0.6% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 1 | 1 | 11956714.0 | 44864.5 | 0.4% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 1 | 1 | 12349690.0 | 29435.5 | 0.3% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 1 | 1 | 11283792.0 | 50665.5 | 0.4% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 1 | 1 | 4220064.0 | 144916.5 | 2.3% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1445360.5 | 15864.5 | 0.8% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 1 | 2 | 2005942.0 | 72881.5 | 2.7% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1929794.0 | 313903.8 | 22.7% | NOISY |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 1 | 1 | 6372191.0 | 45300.5 | 0.9% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 1 | 1 | 5609599.0 | 39859.5 | 0.6% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 1 | 1 | 7351776.0 | 146510.0 | 1.9% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 1 | 1 | 6878790.0 | 73021.5 | 1.1% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 1 | 1 | 46472116.0 | 222957.5 | 0.4% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 1 | 1 | 48323218.0 | 352095.5 | 0.5% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 1 | 1 | 50008989.0 | 348067.5 | 2.7% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 1 | 1 | 45898080.0 | 418024.0 | 0.6% | ok |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 1 | 3470468.0 | 12623.5 | 0.4% | ok |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 2 | 997660.5 | 3181.0 | 0.7% | ok |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t1 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 8 | 455191.1 | 18221.7 | 2.3% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 1 | 1 | 2887575.0 | 137527.0 | 2.5% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 1 | 4 | 841012.2 | 17183.5 | 1.1% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 1 | 2048 | 1367.8 | 20.1 | 3.9% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 1 | 32768 | 75.8 | 0.9 | 1.5% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 1 | 2048 | 1284.6 | 6.6 | 1.7% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 1 | 512 | 6107.9 | 43.0 | 1.2% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 1 | 256 | 9350.6 | 61.2 | 0.5% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 1 | 1024 | 3266.1 | 40.9 | 0.9% | ok |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 1 | 8 | 480012.6 | 4721.3 | 0.9% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 1 | 4 | 949320.2 | 17419.0 | 1.1% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 1 | 8 | 305560.5 | 14232.2 | 6.6% | ok |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 1 | 2048 | 1470.0 | 2.4 | 0.4% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 1 | 32768 | 110.1 | 2.8 | 5.8% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 1 | 2048 | 1274.0 | 238.0 | 184.7% | NOISY |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 1 | 256 | 11127.1 | 280.2 | 12.5% | NOISY |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 1 | 256 | 9671.8 | 75.4 | 0.6% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 1 | 512 | 7942.3 | 116.0 | 1.4% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 1 | 2 | 1081247.5 | 16789.2 | 1.5% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 1 | 2 | 1178846.0 | 41353.0 | 3.6% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 1 | 2 | 901165.0 | 49072.2 | 4.0% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 8 | 268251.8 | 3809.8 | 1.2% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 176217.4 | 1220.1 | 1.1% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 249000.1 | 1826.9 | 0.7% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 16 | 189578.6 | 1463.7 | 1.0% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 32 | 70917.7 | 1546.7 | 3.2% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 4992052.0 | 16852.0 | 0.3% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 1543304.0 | 116002.5 | 12.9% | NOISY |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 6478058.0 | 44238.0 | 0.8% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 1 | 4842090.0 | 66169.0 | 1.1% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 2 | 1490650.5 | 9225.0 | 0.5% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 2 | 1713593.5 | 4445.8 | 0.5% | ok |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 2 | 1712411.5 | 5706.0 | 1.4% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 16997247.0 | 13214.5 | 0.1% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 17092325.0 | 33537.5 | 0.2% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 34013929.0 | 30847.5 | 0.2% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 34164140.0 | 77931.5 | 0.2% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 32 | 70936.5 | 2419.2 | 2.2% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 256 | 6864.6 | 471.5 | 11.1% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 32 | 74104.0 | 762.7 | 1.2% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 16 | 122680.1 | 3034.8 | 2.8% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 64 | 27169.1 | 1140.9 | 3.5% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 16 | 126574.2 | 3133.3 | 2.5% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 114756.4 | 2989.1 | 7.2% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 128 | 31575.5 | 1438.8 | 3.8% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 122286.2 | 4484.4 | 2.4% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 477408.0 | 19229.9 | 2.7% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 16 | 232551.1 | 6759.9 | 2.6% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 423379.0 | 14274.2 | 7.2% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 64 | 37127.0 | 792.8 | 1.4% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 128 | 19691.5 | 405.9 | 2.0% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 64 | 44716.8 | 710.0 | 1.7% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 16 | 61693.2 | 2382.9 | 4.1% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 64 | 45944.5 | 1992.0 | 3.8% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 16 | 68183.5 | 1129.9 | 1.3% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 32 | 60561.7 | 2759.7 | 2.6% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 64 | 47728.6 | 656.0 | 1.4% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 16 | 68515.4 | 2336.6 | 2.7% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 4 | 216147.8 | 8895.4 | 5.1% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 8 | 332601.2 | 16284.9 | 3.0% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 4 | 234091.5 | 7177.2 | 4.7% | ok |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 1 | 1024 | 3219.6 | 21.6 | 5.7% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | 8 | 314441.0 | 1968.6 | 0.9% | ok |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | 8 | 317758.4 | 1513.4 | 0.3% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 1 | 16 | 153010.8 | 2268.3 | 1.2% | ok |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 32 | 54909.8 | 257.2 | 0.6% | ok |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 32 | 55214.2 | 13944.1 | 11.3% | NOISY |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 16 | 87879.2 | 1445.6 | 1.5% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 16 | 110871.7 | 2888.5 | 2.3% | ok |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 32 | 57069.8 | 347.1 | 0.5% | ok |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 32 | 56096.8 | 322.3 | 0.5% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 16 | 104969.4 | 2548.8 | 2.3% | ok |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 1 | 16 | 114750.2 | 699.8 | 1.0% | ok |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 16 | 183933.1 | 864.7 | 0.7% | ok |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 16 | 182741.5 | 508.2 | 0.7% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 8 | 187022.0 | 4119.0 | 1.4% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 1 | 8 | 219735.9 | 536.6 | 0.4% | ok |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 16 | 79392.0 | 1029.4 | 0.9% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 16 | 78322.5 | 654.1 | 1.2% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 65707.9 | 488.1 | 3.0% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 8 | 199621.9 | 2478.4 | 0.6% | ok |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 8 | 196701.4 | 2147.7 | 1.0% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 8 | 330256.9 | 9131.4 | 1.9% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 8 | 226260.6 | 1312.6 | 0.4% | ok |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 32 | 31605.2 | 282.5 | 0.6% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 31422.7 | 305.4 | 0.8% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 16 | 108421.5 | 2322.8 | 1.6% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 38717.0 | 207.7 | 0.4% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 2006673.0 | 11942.5 | 0.5% | ok |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 2 | 2007385.0 | 21289.5 | 0.7% | ok |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 3377454.0 | 91661.5 | 3.6% | ok |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 2220434.0 | 14331.5 | 1.1% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 131960.6 | 3261.0 | 1.5% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 101906.1 | 2626.4 | 1.8% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 107379.5 | 3021.6 | 1.7% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 64 | 38071.4 | 404.0 | 1.7% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3222884.0 | 183924.5 | 3.5% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3491939.0 | 245755.5 | 5.6% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3252459.0 | 68753.5 | 1.8% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 2183996.0 | 89577.5 | 2.6% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 8 | 238491.0 | 2445.1 | 1.0% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 196052.7 | 1208.8 | 0.6% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 125584.9 | 2004.7 | 1.5% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 32 | 89620.6 | 1073.6 | 0.8% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 8237128.0 | 181025.0 | 1.4% | ok |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 8288485.0 | 130670.0 | 1.0% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 4444385.0 | 120490.5 | 1.8% | ok |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 4861313.0 | 57277.5 | 1.4% | ok |
| softmax_f32_eager-single-call_len64_b8_t1 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 32 | 31802.1 | 169.8 | 0.6% | ok |
| softmax_f32_eager-composed_len64_b8_t1 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 16 | 108458.3 | 2074.8 | 1.3% | ok |
| softmax_f32_pytorch_len64_b8_t1 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 40745.6 | 231.6 | 1.0% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 128 | 17535.9 | 405.3 | 2.0% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 8123.3 | 127.3 | 1.5% | ok |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 128 | 17715.3 | 818.6 | 3.5% | ok |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 9633.4 | 280.0 | 1.7% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 8680443.0 | 60463.0 | 0.8% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 8466291.0 | 60228.5 | 2.8% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 8378546.0 | 56560.5 | 1.5% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 1 | 1 | 3324292.0 | 45179.5 | 1.3% | ok |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 1 | 1 | 20320933.0 | 416431.0 | 1.3% | ok |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 1 | 1 | 9134364.0 | 88436.5 | 0.6% | ok |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 1 | 1 | 34338531.0 | 312490.5 | 0.8% | ok |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 1 | 1 | 30496068.0 | 295975.0 | 0.7% | ok |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 1 | 1 | 138487841.0 | 9008962.0 | 3.3% | ok |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 1 | 1 | 132949030.0 | 4393070.0 | 1.8% | ok |
| metadata_transpose_view_f64_rank8_t1 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 8192 | 60.0 | 1.2 | 86.4% | NOISY |
| metadata_transpose_view_f64_rank8_t1_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 1048576 | 0.3 | 0.0 | 17.6% | NOISY |
| followup_structural_broadcast_in_dim_direct-shared | #2021 | direct-shared | tenferro-rs | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 119985131.0 | 2366979.0 | 1.9% | ok |
| followup_structural_broadcast_in_dim_pytorch | #2021 | pytorch | pytorch-cpu | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 116013762.0 | 1394744.5 | 0.9% | ok |
| followup_real_tanh_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 51305478.0 | 66720.0 | 0.1% | ok |
| followup_real_tanh_jax | #2033 | jax | jax-cpu | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 41021742.0 | 1293460.0 | 1.7% | ok |
| followup_real_log1p_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 13226155.0 | 59602.5 | 0.3% | ok |
| followup_real_log1p_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 33849131.0 | 249783.5 | 0.7% | ok |
| followup_real_cos_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 28286489.0 | 176251.5 | 0.4% | ok |
| followup_real_cos_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 59450408.0 | 1523756.5 | 1.5% | ok |
| followup_real_sin_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 24513402.0 | 47007.5 | 0.2% | ok |
| followup_real_sin_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 60386475.0 | 249037.5 | 0.3% | ok |
| followup_real_exp_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 20616583.0 | 30036.5 | 0.1% | ok |
| followup_real_exp_jax | #2033 | jax | jax-cpu | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 43554158.0 | 618905.5 | 0.9% | ok |
| followup_real_expm1_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 20401470.0 | 47113.0 | 0.2% | ok |
| followup_real_expm1_jax | #2033 | jax | jax-cpu | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 28870568.0 | 371015.5 | 0.8% | ok |
| followup_real_pow_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 58295216.0 | 98861.0 | 0.2% | ok |
| followup_real_pow_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 64560945.0 | 144892.0 | 0.2% | ok |
| followup_complex_exp_direct-shared | #2033 | direct-shared | tenferro-rs | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 70541263.0 | 274109.0 | 0.3% | ok |
| followup_complex_exp_julia-base | #2033 | julia-base | julia-base | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 81537721.0 | 460376.5 | 0.7% | ok |
| followup_real_log_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 25376213.0 | 47573.5 | 0.2% | ok |
| followup_real_log_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 52779528.0 | 1420197.5 | 1.5% | ok |
| followup_real_reduce_max_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 2524685.0 | 34604.0 | 1.6% | ok |
| followup_real_reduce_max_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1450860.5 | 100819.0 | 9.4% | ok |
| followup_real_reduce_prod_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1337473.0 | 47371.5 | 3.2% | ok |
| followup_real_reduce_prod_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 964644.0 | 57793.5 | 3.8% | ok |
| followup_real_reduce_sum_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1345032.0 | 52180.5 | 3.0% | ok |
| followup_real_reduce_sum_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 4 | 894991.8 | 36531.0 | 4.9% | ok |
| followup_real_reduce_max_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 1976880.0 | 46427.5 | 3.5% | ok |
| followup_real_reduce_max_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1533455.5 | 109500.0 | 7.4% | ok |
| followup_real_reduce_min_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 11037268.0 | 228082.5 | 2.3% | ok |
| followup_real_reduce_min_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 7561599.0 | 269064.0 | 3.1% | ok |
| followup_real_reduce_min_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1845636.0 | 24348.2 | 1.4% | ok |
| followup_real_reduce_min_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1278873.0 | 56498.2 | 9.5% | ok |
| followup_real_reduce_max_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 16518506.0 | 133355.5 | 0.6% | ok |
| followup_real_reduce_max_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 13211786.0 | 157906.0 | 1.0% | ok |
| followup_real_reduce_min_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 17388769.0 | 177342.5 | 1.0% | ok |
| followup_real_reduce_min_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 13875240.0 | 259562.0 | 1.2% | ok |
| followup_complex_norm_fro_direct-shared | #2035 | direct-shared | tenferro-rs | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 2355197.0 | 78140.5 | 2.0% | ok |
| followup_complex_norm_fro_jax | #2035 | jax | jax-cpu | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 3975485.0 | 170505.0 | 2.8% | ok |
| followup_index_gather_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 4 | 406795.2 | 7152.1 | 3.8% | ok |
| followup_index_gather_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 4 | 1135724.8 | 19815.8 | 1.3% | ok |
| followup_index_scatter_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 819933.5 | 13089.8 | 1.5% | ok |
| followup_index_scatter_pytorch | #2036 #1719 | pytorch | pytorch-cpu | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 4 | 1269366.2 | 24951.8 | 1.9% | ok |
| followup_index_dynamic_update_slice_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1532439.5 | 87574.2 | 4.9% | ok |
| followup_index_dynamic_update_slice_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 8200307.5 | 151506.5 | 1.8% | ok |
| followup_perm_reverse_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 111257285.0 | 1694934.0 | 1.6% | ok |
| followup_perm_reverse_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 100401243.0 | 909135.0 | 0.6% | ok |
| followup_structural_transpose_direct-shared | #2037 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 90017512.0 | 2789986.5 | 2.1% | ok |
| followup_structural_transpose_pytorch | #2037 #2228 | pytorch | pytorch-cpu | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 80451111.0 | 717579.0 | 0.8% | ok |
| followup_perm_reverse_23d_2_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 31747329.0 | 243020.5 | 0.6% | ok |
| followup_perm_reverse_23d_2_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 60835211.0 | 435722.0 | 0.5% | ok |
| followup_perm_transpose_3d_256_201_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 87797476.0 | 1486751.5 | 2.0% | ok |
| followup_perm_transpose_3d_256_201_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 91418838.0 | 1023830.0 | 1.3% | ok |
| followup_perm_cyclic_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 60791016.0 | 265288.0 | 0.4% | ok |
| followup_perm_cyclic_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 61112709.0 | 571157.5 | 1.1% | ok |
| followup_perm_transpose_3d_256_102_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 90966560.0 | 1209659.5 | 1.3% | ok |
| followup_perm_transpose_3d_256_102_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 87241786.0 | 4495447.5 | 3.1% | ok |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 77019505.0 | 2444789.0 | 1.9% | ok |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 78327655.0 | 5294012.5 | 3.8% | ok |
| followup_perm_tn_light_415_24d_contiguous_same_perm_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 71313161.0 | 2171790.5 | 2.8% | ok |
| followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 72059679.0 | 485397.5 | 0.7% | ok |
| followup_perm_rotation_6d_32_32_32_32_16_16_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 1380649380.0 | 35891039.0 | 1.9% | ok |
| followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 1393005490.0 | 29166371.0 | 1.8% | ok |
| followup_structural_extract_diagonal_direct-shared | #2038 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 72133268.0 | 2123359.5 | 1.6% | ok |
| followup_structural_extract_diagonal_jax | #2038 #2228 | jax | jax-cpu | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 75560233.0 | 598421.5 | 0.8% | ok |
| followup_fft_irfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 3891709.0 | 186441.5 | 3.5% | ok |
| followup_fft_irfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 2502092.0 | 62703.5 | 2.5% | ok |
| followup_fft_rfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 3846034.0 | 177794.5 | 3.2% | ok |
| followup_fft_rfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 2529283.0 | 134794.0 | 5.0% | ok |
| followup_fft_fft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 12749128.0 | 869522.5 | 5.8% | ok |
| followup_fft_fft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 12312436.0 | 665052.5 | 4.5% | ok |
| followup_fft_ifft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 6252431.0 | 353901.5 | 3.3% | ok |
| followup_fft_ifft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 5377453.0 | 315453.0 | 4.5% | ok |
| followup_metadata_transpose_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 336.9 | 8.4 | 12.8% | NOISY |
| followup_metadata_transpose_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 1864135 | 0.8 | 0.0 | 3.6% | ok |
| followup_metadata_transpose_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 395.5 | 7.2 | 1.4% | ok |
| followup_metadata_transpose_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 328.7 | 4.4 | 1.1% | ok |
| followup_metadata_slice_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 384.7 | 7.2 | 1.2% | ok |
| followup_metadata_slice_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 16256 | 3.7 | 0.7 | 12.4% | NOISY |
| followup_metadata_slice_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 4096 | 476.0 | 7.8 | 1.8% | ok |
| followup_metadata_slice_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 379.6 | 10.5 | 1.9% | ok |
| followup_metadata_reshape_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 465.8 | 4.5 | 0.7% | ok |
| followup_metadata_reshape_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 63550 | 8.0 | 0.8 | 7.9% | ok |
| followup_metadata_reshape_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 4096 | 479.6 | 9.8 | 1.8% | ok |
| followup_metadata_reshape_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 4096 | 466.7 | 5.3 | 1.0% | ok |
| typed_elemwise_f64_add_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=held | 1 | 2048 | 1071.5 | 19.0 | 3.8% | ok |
| typed_elemwise_f64_add_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=held | 1 | 2048 | 1445.7 | 16.0 | 1.9% | ok |
| typed_elemwise_f64_add_into_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=into_held | 1 | 4096 | 846.0 | 8.5 | 1.2% | ok |
| typed_elemwise_f64_add_into_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=into_held | 1 | 2048 | 1380.7 | 18.7 | 1.2% | ok |
| typed_elemwise_f64_add_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=held | 1 | 2048 | 1168.6 | 6.3 | 3.2% | ok |
| typed_elemwise_f64_add_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=held | 1 | 2048 | 1536.3 | 30.3 | 1.7% | ok |
| typed_elemwise_f64_add_into_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=into_held | 1 | 4096 | 883.1 | 9.6 | 1.1% | ok |
| typed_elemwise_f64_add_into_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=into_held | 1 | 2048 | 1414.2 | 9.6 | 0.7% | ok |
| typed_elemwise_f64_add_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=held | 1 | 256 | 2245.9 | 13.4 | 1.8% | ok |
| typed_elemwise_f64_add_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=held | 1 | 1024 | 3221.6 | 167.9 | 3.8% | ok |
| typed_elemwise_f64_add_into_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=into_held | 1 | 2048 | 1872.3 | 1.7 | 0.1% | ok |
| typed_elemwise_f64_add_into_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=into_held | 1 | 1024 | 2414.6 | 18.6 | 0.7% | ok |
| tiny_matmul_2x2_typed_held | #2139 | tenferro | tenferro-rs | f64 | n=2 | 1 | 2048 | 1431.3 | 8.6 | 0.9% | ok |
| tiny_matmul_2x2_typed_held_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=2 | 1 | 2048 | 1587.5 | 11.5 | 1.7% | ok |
| analytic_neg_f64_sweep_8MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=1048576 | 1 | 2 | 315157.5 | 32734.2 | 7.9% | ok |
| analytic_neg_f64_sweep_8MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 4 | 460569.2 | 43936.5 | 6.2% | ok |
| analytic_neg_f64_sweep_32MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=4194304 | 1 | 1 | 2289278.0 | 159154.0 | 5.1% | ok |
| analytic_neg_f64_sweep_32MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4194304 | 1 | 1 | 15739387.0 | 133121.5 | 0.6% | ok |
| gemm_bound_f64_t1_t8_scaling | #2141 | tenferro | tenferro-rs | f64 | n=1536 | 1 | 1 | 156605421.0 | 322141.5 | 0.2% | ok |
| gemm_bound_f64_t1_t8_scaling_pytorch | #2141 | pytorch | pytorch-cpu | f64 | n=1536 | 1 | 1 | 156334802.0 | 1210180.5 | 0.7% | ok |
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 1 | 8 | 324168.6 | 2048.1 | 0.7% | ok |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 1 | 4 | 507219.5 | 2376.8 | 0.9% | ok |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 1 | 16 | 204632.5 | 1490.3 | 1.3% | ok |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 1 | 16 | 169245.3 | 1700.7 | 1.1% | ok |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 1 | 2 | 1048397.0 | 10212.0 | 1.2% | ok |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 1 | 2 | 2157579.0 | 27256.0 | 1.0% | ok |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 1 | 8 | 307819.1 | 4595.0 | 0.9% | ok |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 1 | 8 | 373395.0 | 3042.0 | 0.6% | ok |
| axpby_accum_strided_dest_f64_block | #2143 | tenferro | tenferro-rs | f64 | n=64 | 1 | 512 | 6105.4 | 39.7 | 0.7% | ok |
| axpby_accum_strided_dest_f64_block_host | #2143 | host | rust-loop | f64 | n=64 | 1 | 512 | 6315.5 | 53.8 | 1.2% | ok |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 1 | 32 | 63153.8 | 848.6 | 1.3% | ok |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 1 | 16 | 133321.4 | 1900.8 | 1.6% | ok |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 1 | 32 | 74538.0 | 2474.2 | 3.5% | ok |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 1 | 16 | 145364.8 | 1925.4 | 2.5% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 1 | 32 | 72656.7 | 774.9 | 0.8% | ok |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 1 | 16 | 139026.6 | 3709.5 | 1.8% | ok |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 1 | 32 | 103007.6 | 1398.5 | 1.0% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 1 | 16 | 166750.6 | 2602.7 | 1.6% | ok |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 1 | 64 | 31517.3 | 487.5 | 1.5% | ok |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 1 | 16 | 147324.6 | 1949.0 | 1.2% | ok |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 1 | 64 | 39788.9 | 1199.8 | 2.4% | ok |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 1 | 16 | 157968.4 | 1829.4 | 1.8% | ok |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 1 | 32 | 107587.3 | 2027.0 | 1.7% | ok |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 1 | 16 | 181702.4 | 5428.0 | 1.7% | ok |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 1 | 16 | 157982.8 | 2623.7 | 1.4% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 1 | 16 | 231422.6 | 3246.7 | 2.0% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 1 | 128 | 22406.0 | 354.5 | 2.5% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 1 | 32 | 117582.3 | 1982.3 | 1.2% | ok |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 1 | 64 | 31097.0 | 582.9 | 1.0% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 1 | 16 | 130831.2 | 3411.4 | 1.7% | ok |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 1 | 1 | 7330852.0 | 64110.0 | 1.2% | ok |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 1 | 1 | 27382101.0 | 1025971.0 | 2.3% | ok |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 1 | 2 | 565301.0 | 28443.5 | 4.5% | ok |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 1 | 2 | 547006.5 | 49896.2 | 5.9% | ok |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 1 | 2 | 219016.0 | 16831.5 | 12.0% | NOISY |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 1 | 8 | 2404761.6 | 16942.9 | 1.0% | ok |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 1 | 2 | 1167225.5 | 55244.0 | 3.5% | ok |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 1 | 4 | 435264.5 | 79468.0 | 10.1% | NOISY |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 1 | 2048 | 903.4 | 19.9 | 4.8% | ok |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 1 | 2048 | 1519.3 | 22.5 | 1.4% | ok |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 1 | 256 | 2052.0 | 66.4 | 9.1% | ok |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 1 | 1024 | 3194.1 | 138.6 | 2.8% | ok |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 1 | 2 | 565814.5 | 12428.0 | 4.6% | ok |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 4 | 647122.5 | 29291.4 | 3.9% | ok |
| axpby_strided_source_f64_1024x1024_t1 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 1 | 2 | 2590128.0 | 244973.2 | 7.4% | ok |
| axpby_contiguous_source_f64_1024x1024_t1 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 1 | 4 | 568968.0 | 19414.0 | 2.5% | ok |
| axpby_strided_source_f64_1024x1024_host_t1 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 1 | 2 | 1681969.5 | 46267.0 | 2.1% | ok |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 1 | 16384 | 101.7 | 0.9 | 28.7% | NOISY |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 1 | 8192 | 206.5 | 3.5 | 15.3% | NOISY |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 1 | 8192 | 236.4 | 4.6 | 1.5% | ok |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 1 | 128 | 19785.7 | 91.5 | 5.1% | ok |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 1 | 1024 | 103.3 | 0.6 | 3.5% | ok |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 1 | 8 | 340543.0 | 2646.2 | 0.5% | ok |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 1 | 512 | 3526.4 | 83.4 | 1.8% | ok |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 1 | 1024 | 1884.5 | 22.8 | 1.8% | ok |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 1 | 2048 | 1446.5 | 14.1 | 1.4% | ok |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 1 | 1024 | 2940.0 | 11.2 | 1.0% | ok |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 1 | 2048 | 1669.2 | 12.0 | 0.6% | ok |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 5208948.0 | 32646.0 | 0.4% | ok |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 2251235.0 | 18991.5 | 1.0% | ok |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 5839504.0 | 21325.0 | 0.5% | ok |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 2556099.0 | 13165.0 | 2.6% | ok |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 1 | 1024 | 3142.3 | 13.2 | 0.6% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 2 | 1518784.0 | 6655.2 | 0.4% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 4 | 16 | 159497.9 | 14231.4 | 37.7% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 4 | 8 | 498047.1 | 15824.7 | 2.2% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 8 | 367955.2 | 10930.5 | 3.7% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1138289.0 | 20461.2 | 1.5% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 4 | 4 | 762515.5 | 6221.8 | 0.5% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1139411.5 | 11386.2 | 0.8% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1291717.5 | 17475.2 | 1.3% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 4 | 1 | 7836748.0 | 141325.5 | 1.2% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 4 | 1 | 5723116.0 | 41278.0 | 0.6% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 4 | 1 | 6599533.0 | 78397.5 | 0.7% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 4 | 1 | 7684071.0 | 26279.5 | 0.3% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 4 | 1 | 6065829.0 | 29521.0 | 0.9% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 4 | 4 | 563417.2 | 7122.1 | 1.7% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 4 | 1 | 2011834.0 | 279192.5 | 6.9% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 4 | 2 | 1618934.5 | 170019.8 | 27.4% | NOISY |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 4 | 1 | 4495354.0 | 144962.5 | 25.5% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 4 | 1 | 2851513.0 | 13465.5 | 0.7% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 4 | 1 | 4648121.0 | 103645.0 | 2.8% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 4 | 1 | 4937466.0 | 98405.0 | 8.4% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 4 | 1 | 32677194.0 | 2020039.5 | 33.5% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 4 | 1 | 29983355.0 | 3834113.5 | 11.0% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 4 | 1 | 33183085.0 | 4090172.0 | 10.3% | NOISY |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 4 | 1 | 30677242.0 | 972814.5 | 2.4% | ok |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 1 | 4037944.0 | 9853.0 | 0.6% | ok |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 2 | 1513792.0 | 8676.5 | 0.4% | ok |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t4 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 8 | 437436.5 | 13291.2 | 7.0% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 4 | 1 | 1971458.0 | 55084.0 | 13.1% | NOISY |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 4 | 4 | 477119.8 | 7682.0 | 1.1% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 4 | 1024 | 2357.6 | 9.6 | 1.8% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 4 | 32768 | 75.2 | 0.4 | 1.2% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 4 | 2048 | 1341.3 | 17.5 | 2.0% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 4 | 256 | 7379.8 | 131.1 | 4.9% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 4 | 256 | 9341.6 | 59.6 | 0.5% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 4 | 512 | 4347.8 | 27.5 | 0.6% | ok |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 4 | 8 | 477250.0 | 2055.8 | 0.5% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 4 | 4 | 942765.8 | 2237.9 | 0.4% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 4 | 16 | 131015.8 | 85616.0 | 42.7% | NOISY |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 4 | 1024 | 2457.1 | 10.3 | 4.9% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 4 | 32768 | 108.4 | 2.3 | 2.2% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 4 | 2048 | 1236.2 | 21.0 | 200.8% | NOISY |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 4 | 256 | 12403.4 | 95.5 | 0.5% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 4 | 256 | 9632.7 | 116.3 | 2.1% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 4 | 512 | 6950.4 | 393.3 | 23.0% | NOISY |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 4 | 2 | 1137284.0 | 24275.8 | 1.7% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 4 | 2 | 1124500.0 | 27514.5 | 1.9% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 4 | 4 | 405748.2 | 101261.9 | 15.6% | NOISY |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 8 | 342528.0 | 2583.6 | 0.9% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 16 | 243622.6 | 1838.2 | 6.7% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 4 | 16 | 255508.0 | 2781.8 | 0.8% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 8 | 267251.4 | 2499.1 | 0.8% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 32 | 70975.0 | 980.8 | 4.5% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 5684521.0 | 35927.5 | 23.9% | NOISY |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 1268996.0 | 36113.0 | 54.7% | NOISY |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 7202716.0 | 109796.5 | 17.2% | NOISY |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 1 | 3429901.0 | 44513.5 | 1.9% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 4 | 924794.5 | 8806.5 | 0.9% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 4 | 677712.0 | 2692.6 | 2.5% | ok |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 2 | 946711.0 | 274749.0 | 18.6% | NOISY |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 6652091.0 | 66836.0 | 15.7% | NOISY |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 8999997.0 | 62132.0 | 0.6% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 13262624.0 | 50485.5 | 0.6% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 18027415.0 | 59748.0 | 0.4% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 16 | 88576.6 | 2015.1 | 13.6% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 256 | 8809.2 | 93.5 | 1.8% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 16 | 93011.2 | 2163.7 | 13.9% | NOISY |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 8 | 167922.1 | 3909.2 | 8.4% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 128 | 18278.3 | 219.1 | 1.1% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 8 | 197494.0 | 4412.1 | 2.1% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 8 | 184954.1 | 4107.1 | 1.5% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 128 | 18287.7 | 260.9 | 1.2% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 8 | 207303.8 | 4043.8 | 2.2% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 2 | 553561.0 | 12731.5 | 2.3% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 16 | 93903.5 | 5908.0 | 5.5% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 4 | 558811.0 | 20388.2 | 3.1% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 32 | 46189.7 | 756.7 | 14.0% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 128 | 18998.3 | 285.9 | 1.8% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 32 | 52660.0 | 1083.9 | 10.7% | NOISY |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 94877.8 | 3002.8 | 2.2% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 32 | 53814.3 | 903.4 | 6.5% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 111708.2 | 1743.6 | 1.6% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 16 | 92383.1 | 1719.7 | 1.4% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 64 | 54155.8 | 1342.2 | 38.4% | NOISY |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 16 | 88823.9 | 3252.0 | 7.8% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 4 | 256705.2 | 92535.2 | 109.6% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 8 | 180110.0 | 14682.7 | 9.5% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 4 | 242117.8 | 7730.9 | 3.6% | ok |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 4 | 512 | 4261.9 | 21.5 | 0.4% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | 8 | 244463.5 | 77478.9 | 18.6% | NOISY |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | 8 | 244038.9 | 81634.1 | 19.4% | NOISY |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 4 | 8 | 56552.6 | 621067.4 | 96.7% | NOISY |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 64 | 28023.4 | 161.6 | 1.9% | ok |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 64 | 27232.9 | 311.9 | 0.8% | ok |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 16 | 75785.6 | 5987.5 | 10.4% | NOISY |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 16 | 94204.6 | 183655.2 | 127.8% | NOISY |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 64 | 29039.1 | 189.6 | 0.5% | ok |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 64 | 27943.5 | 200.8 | 0.7% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 16 | 89135.1 | 2920.5 | 8.0% | ok |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 4 | 16 | 99009.7 | 242826.2 | 59.7% | NOISY |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 32 | 73954.9 | 717.0 | 1.1% | ok |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 32 | 73135.5 | 598.8 | 4.0% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 8 | 209635.0 | 65439.8 | 18.5% | NOISY |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 4 | 1 | 163196.0 | 9031.5 | 214.9% | NOISY |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 64 | 38718.4 | 312.3 | 5.0% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 64 | 37835.8 | 314.1 | 2.1% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 4 | 32 | 42714.1 | 917.0 | 6.7% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 32 | 77951.6 | 9063.5 | 14.9% | NOISY |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 32 | 77131.3 | 839.7 | 10.4% | NOISY |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 8 | 274957.0 | 52056.6 | 19.5% | NOISY |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 1 | 82545.0 | 2630.5 | 198.0% | NOISY |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 22247.0 | 263.4 | 1.2% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 21036.8 | 283.5 | 0.8% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 16 | 121194.1 | 2138.1 | 4.7% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 128 | 19856.9 | 418.8 | 5.9% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 860763.5 | 12768.8 | 4.5% | ok |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 839944.0 | 10227.0 | 3.7% | ok |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 2101256.0 | 59800.0 | 4.9% | ok |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 955221.0 | 4483.2 | 1.0% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 4 | 156892.8 | 8081.4 | 12.5% | NOISY |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 16 | 113579.2 | 3447.8 | 4.8% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 16 | 117356.9 | 2360.7 | 4.8% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 128 | 20268.7 | 116.5 | 2.5% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 1935810.0 | 37981.5 | 2.8% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1956679.0 | 36067.8 | 4.1% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2049093.0 | 42054.5 | 2.2% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 996629.0 | 25951.5 | 1.8% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 8 | 258367.0 | 5983.8 | 6.1% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 8 | 209401.2 | 4147.8 | 4.5% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 8 | 137412.1 | 5915.5 | 17.3% | NOISY |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 94642.3 | 1142.6 | 3.7% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3992567.0 | 200451.5 | 7.7% | ok |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3882770.0 | 54637.5 | 1.5% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2965596.0 | 746959.0 | 17.3% | NOISY |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2319943.0 | 48501.0 | 3.2% | ok |
| softmax_f32_eager-single-call_len64_b8_t4 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 21931.5 | 402.6 | 5.3% | ok |
| softmax_f32_eager-composed_len64_b8_t4 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 16 | 119672.5 | 2328.2 | 6.6% | ok |
| softmax_f32_pytorch_len64_b8_t4 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 128 | 19558.6 | 167.3 | 1.4% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 128 | 20124.5 | 668.6 | 5.3% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 8862.8 | 163.7 | 10.5% | NOISY |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 64 | 30296.0 | 11208.0 | 21.9% | NOISY |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 12236.1 | 61.0 | 0.4% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 5252206.0 | 39239.0 | 3.9% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 4931513.0 | 31409.5 | 2.4% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 4770821.0 | 26094.5 | 0.5% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 4 | 2 | 1734417.0 | 75683.2 | 4.2% | ok |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 4 | 1 | 27438067.0 | 7091518.0 | 14.6% | NOISY |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 4 | 1 | 6207560.0 | 121567.0 | 2.0% | ok |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 4 | 1 | 24900156.0 | 103137.0 | 1.7% | ok |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 4 | 1 | 42950947.0 | 28064586.0 | 35.0% | NOISY |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 4 | 1 | 43016601.0 | 553560.5 | 1.1% | ok |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 4 | 1 | 44502804.0 | 1033042.5 | 5.9% | ok |
| metadata_transpose_view_f64_rank8_t4 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 8192 | 66.1 | 2.0 | 82.7% | NOISY |
| metadata_transpose_view_f64_rank8_t4_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 1048576 | 0.3 | 0.0 | 9.5% | ok |
| followup_structural_broadcast_in_dim_direct-shared | #2021 | direct-shared | tenferro-rs | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 46638592.0 | 192297.0 | 0.4% | ok |
| followup_structural_broadcast_in_dim_pytorch | #2021 | pytorch | pytorch-cpu | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 38612112.0 | 1010837.0 | 3.4% | ok |
| followup_real_tanh_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 20518362.0 | 55394.0 | 0.2% | ok |
| followup_real_tanh_jax | #2033 | jax | jax-cpu | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 21448477.0 | 304757.5 | 1.4% | ok |
| followup_real_log1p_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 5652172.0 | 20885.0 | 1.0% | ok |
| followup_real_log1p_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 12462612.0 | 217038.5 | 1.5% | ok |
| followup_real_cos_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 11873825.0 | 87424.5 | 2.5% | ok |
| followup_real_cos_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 23187056.0 | 198629.0 | 0.7% | ok |
| followup_real_sin_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 10252985.0 | 35201.5 | 4.1% | ok |
| followup_real_sin_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 23608256.0 | 134689.0 | 0.6% | ok |
| followup_real_exp_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 8434802.0 | 20593.5 | 2.1% | ok |
| followup_real_exp_jax | #2033 | jax | jax-cpu | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 22823272.0 | 321502.5 | 1.0% | ok |
| followup_real_expm1_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 8135972.0 | 68584.5 | 2.1% | ok |
| followup_real_expm1_jax | #2033 | jax | jax-cpu | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 14921641.0 | 280096.0 | 3.3% | ok |
| followup_real_pow_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 23668638.0 | 261446.0 | 1.2% | ok |
| followup_real_pow_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 23446451.0 | 195016.5 | 0.7% | ok |
| followup_complex_exp_direct-shared | #2033 | direct-shared | tenferro-rs | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 31158938.0 | 309470.5 | 8.9% | ok |
| followup_complex_exp_julia-base | #2033 | julia-base | julia-base | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 82799068.0 | 1564081.5 | 1.7% | ok |
| followup_real_log_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 10501683.0 | 17076.5 | 0.2% | ok |
| followup_real_log_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 20072373.0 | 338996.5 | 0.9% | ok |
| followup_real_reduce_max_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 2 | 1272520.0 | 33746.2 | 3.4% | ok |
| followup_real_reduce_max_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 2 | 1539686.5 | 183442.2 | 7.7% | ok |
| followup_real_reduce_prod_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 655303.0 | 72878.0 | 7.1% | ok |
| followup_real_reduce_prod_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 400915.5 | 151092.4 | 20.3% | NOISY |
| followup_real_reduce_sum_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 625737.8 | 36200.5 | 4.7% | ok |
| followup_real_reduce_sum_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 364637.5 | 79847.6 | 17.1% | NOISY |
| followup_real_reduce_max_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 816406.0 | 10337.1 | 1.1% | ok |
| followup_real_reduce_max_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 2 | 1378484.0 | 128689.2 | 10.4% | NOISY |
| followup_real_reduce_min_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 5257815.0 | 225839.0 | 3.2% | ok |
| followup_real_reduce_min_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 7101791.0 | 105027.0 | 1.6% | ok |
| followup_real_reduce_min_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 783887.5 | 11873.6 | 1.6% | ok |
| followup_real_reduce_min_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 2 | 1333780.5 | 105270.2 | 8.6% | ok |
| followup_real_reduce_max_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 6970554.0 | 93169.0 | 1.6% | ok |
| followup_real_reduce_max_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 12962387.0 | 168020.0 | 1.2% | ok |
| followup_real_reduce_min_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 6832322.0 | 235091.5 | 2.0% | ok |
| followup_real_reduce_min_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 13108028.0 | 264776.5 | 1.5% | ok |
| followup_complex_norm_fro_direct-shared | #2035 | direct-shared | tenferro-rs | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 903051.2 | 30246.8 | 5.4% | ok |
| followup_complex_norm_fro_jax | #2035 | jax | jax-cpu | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 2777928.0 | 546023.0 | 12.9% | NOISY |
| followup_index_gather_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 347377.2 | 15266.1 | 7.4% | ok |
| followup_index_gather_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 311783.8 | 42553.6 | 12.5% | NOISY |
| followup_index_scatter_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 2 | 1066913.5 | 19506.0 | 2.6% | ok |
| followup_index_scatter_pytorch | #2036 #1719 | pytorch | pytorch-cpu | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 703521.0 | 543721.0 | 71.0% | NOISY |
| followup_index_dynamic_update_slice_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 2 | 873054.5 | 29943.8 | 4.0% | ok |
| followup_index_dynamic_update_slice_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 2 | 3510803.5 | 266592.0 | 9.0% | ok |
| followup_perm_reverse_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 60376489.0 | 706675.5 | 0.9% | ok |
| followup_perm_reverse_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 58019909.0 | 491511.5 | 0.9% | ok |
| followup_structural_transpose_direct-shared | #2037 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 32492043.0 | 2436500.5 | 5.2% | ok |
| followup_structural_transpose_pytorch | #2037 #2228 | pytorch | pytorch-cpu | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 81824955.0 | 285465.0 | 0.4% | ok |
| followup_perm_reverse_23d_2_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 16372960.0 | 72035.0 | 0.6% | ok |
| followup_perm_reverse_23d_2_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 45177772.0 | 205731.5 | 0.4% | ok |
| followup_perm_transpose_3d_256_201_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 28841831.0 | 203177.0 | 1.1% | ok |
| followup_perm_transpose_3d_256_201_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 28458982.0 | 593699.5 | 1.4% | ok |
| followup_perm_cyclic_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 20392627.0 | 500444.5 | 1.8% | ok |
| followup_perm_cyclic_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 20241964.0 | 264446.0 | 1.4% | ok |
| followup_perm_transpose_3d_256_102_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 30235493.0 | 513755.0 | 2.2% | ok |
| followup_perm_transpose_3d_256_102_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 30436471.0 | 1049130.0 | 3.5% | ok |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 26815072.0 | 135379.0 | 0.8% | ok |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 26113722.0 | 560232.0 | 6.4% | ok |
| followup_perm_tn_light_415_24d_contiguous_same_perm_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 23456503.0 | 433849.5 | 2.1% | ok |
| followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 24098238.0 | 496451.5 | 1.7% | ok |
| followup_perm_rotation_6d_32_32_32_32_16_16_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 515002525.0 | 5779043.5 | 0.7% | ok |
| followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 495840576.0 | 5291505.5 | 3.6% | ok |
| followup_structural_extract_diagonal_direct-shared | #2038 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 38777131.0 | 304295.5 | 0.8% | ok |
| followup_structural_extract_diagonal_jax | #2038 #2228 | jax | jax-cpu | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 39601909.0 | 1133707.5 | 2.9% | ok |
| followup_fft_irfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 3731149.0 | 53840.0 | 1.2% | ok |
| followup_fft_irfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 2484770.0 | 23624.0 | 0.8% | ok |
| followup_fft_rfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 3710080.0 | 73468.0 | 1.7% | ok |
| followup_fft_rfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 2540536.0 | 19987.0 | 1.2% | ok |
| followup_fft_fft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 11704648.0 | 260493.5 | 10.0% | NOISY |
| followup_fft_fft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 11522207.0 | 357501.0 | 2.4% | ok |
| followup_fft_ifft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 5883398.0 | 60494.0 | 0.8% | ok |
| followup_fft_ifft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 3410479.0 | 83842.5 | 2.0% | ok |
| followup_metadata_transpose_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 319.3 | 1.7 | 1.0% | ok |
| followup_metadata_transpose_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 1864135 | 0.8 | 0.1 | 5.9% | ok |
| followup_metadata_transpose_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 400.4 | 8.1 | 1.5% | ok |
| followup_metadata_transpose_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 321.0 | 7.2 | 1.4% | ok |
| followup_metadata_slice_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 404.0 | 8.4 | 2.2% | ok |
| followup_metadata_slice_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 16256 | 3.3 | 0.5 | 17.7% | NOISY |
| followup_metadata_slice_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 461.6 | 11.5 | 1.8% | ok |
| followup_metadata_slice_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 366.8 | 11.5 | 1.7% | ok |
| followup_metadata_reshape_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 4096 | 459.5 | 7.2 | 1.0% | ok |
| followup_metadata_reshape_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 63550 | 8.7 | 1.8 | 15.1% | NOISY |
| followup_metadata_reshape_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 4096 | 488.2 | 7.3 | 1.3% | ok |
| followup_metadata_reshape_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 4096 | 464.6 | 4.5 | 1.1% | ok |
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 4 | 8 | 323793.9 | 1867.4 | 0.8% | ok |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 4 | 4 | 506693.2 | 3284.9 | 1.0% | ok |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 4 | 16 | 206657.1 | 835.0 | 0.4% | ok |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 4 | 16 | 169330.2 | 1716.0 | 1.3% | ok |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 4 | 2 | 1047866.5 | 8503.0 | 7.4% | ok |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 4 | 2 | 2152899.5 | 20503.2 | 1.1% | ok |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 4 | 8 | 309761.2 | 2580.5 | 1.2% | ok |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 4 | 4 | 373122.5 | 5899.8 | 12.0% | NOISY |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 4 | 32 | 88732.1 | 3373.7 | 2.5% | ok |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 4 | 16 | 178577.6 | 2813.4 | 1.6% | ok |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 4 | 32 | 101595.3 | 1423.3 | 2.0% | ok |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 4 | 16 | 189156.9 | 2152.5 | 0.7% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 4 | 32 | 114927.5 | 4678.8 | 3.1% | ok |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 4 | 16 | 138027.8 | 3318.8 | 1.7% | ok |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 4 | 16 | 137806.8 | 2125.3 | 1.4% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 4 | 16 | 164467.9 | 2408.0 | 1.4% | ok |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 4 | 64 | 49437.6 | 297.5 | 0.5% | ok |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 4 | 16 | 193296.4 | 1251.1 | 1.8% | ok |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 4 | 64 | 56960.0 | 1440.1 | 2.1% | ok |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 4 | 16 | 212642.8 | 2768.0 | 4.3% | ok |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 4 | 16 | 164974.6 | 3119.6 | 1.4% | ok |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 4 | 16 | 187051.0 | 6285.9 | 4.5% | ok |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 4 | 8 | 216209.4 | 1349.4 | 1.0% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 4 | 16 | 234773.1 | 2240.2 | 0.8% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 4 | 64 | 34893.0 | 423.8 | 1.0% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 4 | 32 | 117805.4 | 1743.0 | 1.3% | ok |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 4 | 64 | 43822.5 | 594.4 | 1.1% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 4 | 16 | 132002.7 | 1730.1 | 1.4% | ok |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 4 | 1 | 9816449.0 | 73964.0 | 0.4% | ok |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 4 | 1 | 9411139.0 | 269926.0 | 2.9% | ok |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 4 | 4 | 294610.2 | 42467.4 | 14.1% | NOISY |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 4 | 4 | 342197.2 | 23427.8 | 14.0% | NOISY |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 4 | 2 | 254101.5 | 32017.8 | 8.3% | ok |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 4 | 4 | 148431.0 | 68959.2 | 26.7% | NOISY |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 4 | 2 | 1159125.0 | 20293.0 | 1.7% | ok |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 4 | 4 | 217330.2 | 60481.2 | 18.9% | NOISY |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 4 | 1024 | 1993.5 | 55.8 | 10.0% | NOISY |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 4 | 2048 | 1578.7 | 10.6 | 2.6% | ok |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 4 | 256 | 3187.5 | 39.4 | 11.3% | NOISY |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 4 | 512 | 2696.3 | 130.4 | 4.4% | ok |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 4 | 2 | 305393.0 | 90670.0 | 25.5% | NOISY |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 4 | 4 | 282530.0 | 74008.6 | 19.4% | NOISY |
| axpby_strided_source_f64_1024x1024_t4 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 4 | 4 | 1142954.8 | 137739.8 | 7.3% | ok |
| axpby_contiguous_source_f64_1024x1024_t4 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 4 | 8 | 401802.8 | 25812.8 | 3.7% | ok |
| axpby_strided_source_f64_1024x1024_host_t4 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 4 | 2 | 1613006.5 | 40207.8 | 1.9% | ok |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 4 | 16384 | 102.5 | 1.7 | 29.5% | NOISY |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 4 | 8192 | 211.5 | 4.0 | 15.4% | NOISY |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 4 | 16384 | 238.2 | 4.5 | 1.3% | ok |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 4 | 128 | 19816.7 | 122.2 | 0.6% | ok |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 4 | 1024 | 99.2 | 1.7 | 5.4% | ok |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 4 | 8 | 337180.4 | 2552.9 | 0.8% | ok |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 4 | 512 | 3163.5 | 128.9 | 10.7% | NOISY |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 4 | 512 | 3122.1 | 32.0 | 14.8% | NOISY |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 4 | 2048 | 1495.4 | 8.4 | 0.9% | ok |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 4 | 512 | 4192.9 | 23.7 | 0.7% | ok |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 4 | 2048 | 1703.7 | 22.2 | 0.8% | ok |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1954612.5 | 18259.2 | 1.2% | ok |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1966299.0 | 41712.5 | 10.1% | NOISY |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 1 | 2861712.0 | 27937.0 | 0.7% | ok |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1672839.5 | 13312.2 | 0.9% | ok |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 4 | 512 | 4362.5 | 49.7 | 4.7% | ok |


## Timing boundaries

- **activation / eager-composed** — inside: hand-written formula from pre-B1 eager primitives, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **activation / eager-single-call** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
- **activation / eager-single-call** — inside: one activation call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **activation / pytorch** — inside: API execution, intrinsic output allocation, native completion; outside: fixtures, conversion, JIT compilation, initialization, validation, retention allocation, output destruction.
- **activation / pytorch** — inside: torch gelu, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **activation / session-single-call** — inside: one activation call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, with_backend_session entry (CpuBackend::new()), correctness_check.
- **axpby_strided_source / contiguous-source** — inside: Engine::axpby_read_into_accum (contiguous owned source, compact owned destination); outside: backend/session construction, input_construction, strided/offset view construction (metadata only), per-sample destination allocation and seed restoration, correctness_check.
- **axpby_strided_source / host** — inside: sequential Rust strided axpby loop into a compact owned destination; outside: backend/session construction, input_construction, strided/offset view construction (metadata only), per-sample destination allocation and seed restoration, correctness_check.
- **axpby_strided_source / strided-source** — inside: Engine::axpby_read_into_accum (strided/offset source view, compact owned destination); outside: backend/session construction, input_construction, strided/offset view construction (metadata only), per-sample destination allocation and seed restoration, correctness_check.
- **composed_norm / eager-composed** — inside: eager composition (reduce_sum, scale_real, broadcast_in_dim, sub/add/mul, reduce_sum_squares, rsqrt), intermediate and output allocation; outside: eager_runtime_construction, input/weight/bias/eps construction, eager session entry, correctness_check.
- **composed_norm / eager-single-call** — inside: one EagerSession layer_norm / rms_norm call (composite), intermediate and output allocation; outside: eager_runtime_construction, input/weight/bias/eps construction, eager session entry, correctness_check.
- **composed_norm / pytorch-fused** — inside: torch.nn.functional.layer_norm (fused), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **composed_norm / pytorch-fused** — inside: torch.nn.functional.rms_norm (fused), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **cpu_followup_mwe / direct-shared** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
- **cpu_followup_mwe / jax** — inside: API execution, intrinsic output allocation, native completion; outside: fixtures, conversion, JIT compilation, initialization, validation, retention allocation, output destruction.
- **cpu_followup_mwe / materialize** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
- **cpu_followup_mwe / metadata** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
- **cpu_followup_mwe / metadata-host-dyn** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
- **cpu_followup_mwe / metadata-host-static** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
- **cpu_followup_mwe / mkl-dfti** — inside: API execution, intrinsic output allocation, native completion; outside: fixtures, conversion, JIT compilation, initialization, validation, retention allocation, output destruction.
- **cpu_followup_mwe / pytorch** — inside: API execution, intrinsic output allocation, native completion; outside: fixtures, conversion, JIT compilation, initialization, validation, retention allocation, output destruction.
- **cpu_followup_mwe / strided-rs** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
- **cpu_gap_mwe / concrete-shared** — inside: reshape, intrinsic output allocation; outside: fixture construction, runtime/backend construction, session entry, FFT plans and scratch priming, correctness validation, retention vector allocation, output destruction.
- **cpu_gap_mwe / concrete-shared** — inside: triangular, intrinsic output allocation; outside: fixture construction, runtime/backend construction, session entry, FFT plans and scratch priming, correctness validation, retention vector allocation, output destruction.
- **cpu_gap_mwe / eager-shared** — inside: lstsq, intrinsic output allocation; outside: fixture construction, runtime/backend construction, session entry, FFT plans and scratch priming, correctness validation, retention vector allocation, output destruction.
- **cpu_gap_mwe / pytorch** — inside: lstsq, intrinsic output allocation, Python dispatch; outside: input_construction, torch.set_num_threads, correctness_check.
- **cpu_gap_mwe / pytorch** — inside: reshape, intrinsic output allocation, Python dispatch; outside: input_construction, torch.set_num_threads, correctness_check.
- **cpu_gap_mwe / pytorch** — inside: triangular, intrinsic output allocation, Python dispatch; outside: input_construction, torch.set_num_threads, correctness_check.
- **decode_projection / eager-shared** — inside: eager_dot_general, output_allocation; outside: backend_construction, input_construction, correctness_check, eager_session_entry_exit.
- **decode_projection / faer-direct** — inside: faer_matmul_into_preallocated_output; outside: backend_construction, input_construction, correctness_check, output_allocation.
- **decode_projection / host-sgemm** — inside: matrixmultiply_sgemm_into_preallocated_output; outside: backend_construction, input_construction, correctness_check, output_allocation.
- **decode_projection / pytorch** — inside: torch.nn.functional.linear, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **decode_projection_view_operand / prepared-contiguous** — inside: Session::dot_general_read_cached (prepared cache slot) on the borrowed caller-held session, output_allocation; outside: backend/session construction, input_construction, weight view construction (metadata only), correctness_check.
- **decode_projection_view_operand / prepared-view** — inside: Session::dot_general_read_cached (prepared cache slot) on the borrowed caller-held session, output_allocation; outside: backend/session construction, input_construction, weight view construction (metadata only), correctness_check.
- **decode_projection_view_operand / pytorch** — inside: torch.matmul(x, W.T), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **degenerate_dot_general / dot-general-shared** — inside: Session::dot_general_read on the borrowed caller-held session, output_allocation; outside: backend/session construction, input_construction, correctness_check.
- **degenerate_dot_general / pytorch** — inside: torch hadamard (degenerate dot), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **degenerate_dot_general / pytorch** — inside: torch outer (degenerate dot), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **degenerate_dot_general / pytorch** — inside: torch rank0 (degenerate dot), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **eager_small_call / eager-held-session** — inside: EagerSession::add on two held 8-element leaves; outside: backend/runtime construction, eager session entry (held, outside the clock), leaf input construction, correctness_check.
- **eager_small_call / eager-held-session** — inside: EagerSession::constant_from on a prepared 8-element payload; outside: backend/runtime construction, eager session entry (held, outside the clock), leaf input construction, correctness_check.
- **eager_small_call / eager-held-session** — inside: EagerSession::dot_general on two held 4x4 leaves; outside: backend/runtime construction, eager session entry (held, outside the clock), leaf input construction, correctness_check.
- **eager_small_call / pytorch** — inside: torch.add (8 elements), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **eager_small_call / pytorch** — inside: torch.matmul (4x4), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **erased_binary_add / pytorch** — inside: torch.add, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **erased_binary_add / session-add** — inside: erased Tensor::add on the borrowed caller-held session, output_allocation; outside: backend/session construction, input_construction, correctness_check.
- **erased_construct / col-major** — inside: erased Tensor::from_vec_* on a prepared payload Vec; outside: payload clone and restore, shape_check, allocation counting pass (separate).
- **erased_construct / row-major** — inside: erased Tensor::from_vec_* on a prepared payload Vec; outside: payload clone and restore, shape_check, allocation counting pass (separate).
- **fft_axis_c2c / fft-executor** — inside: FftExecutor::fft over axis 0 on the borrowed caller-held session, output_allocation; outside: backend/session construction, input_construction, correctness_check.
- **fft_axis_c2c / pytorch** — inside: torch.fft.fft(dim=0), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **gemm_mm256 / dot-general-shared** — inside: dot_general_read, output_allocation; outside: backend_construction (CpuBackend::with_threads(1)), input_construction, session_entry_exit, correctness_check.
- **metadata_clone / julia-base** — inside: PermutedDimsArray metadata descriptor; outside: owner and transposed view creation, alias/layout checks, JIT compilation, output retention allocation and cleanup.
- **metadata_clone / metadata-clone** — inside: clone of prepared borrowed rank-8 transpose-view metadata; outside: owner allocation and values, transpose-view construction, alias and layout validation, output destruction.
- **rank1_dot / dot-general-rank1** — inside: dot_general_read (rank-1, rank-0 tensor output), output_allocation; outside: backend_construction, input_construction, operand_conjugation, session_entry_exit, correctness_check.
- **rank1_dot / portable-loop** — inside: sequential Rust loop over borrowed slices, scalar result; outside: input_construction, correctness_check.
- **rank1_dot / pytorch-vdot** — inside: torch.vdot (BLAS ddot/zdotc), scalar tensor allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **reduce_mean / eager-composed** — inside: reduce_sum + scale_real, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **reduce_mean / eager-single-call** — inside: one reduce_mean call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **reduce_mean / pytorch** — inside: torch.mean, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **reduce_mean / session-single-call** — inside: one reduce_mean call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, with_backend_session entry (CpuBackend::new()), correctness_check.
- **small_contraction_pool_hop / shared-session-default-pool** — inside: dot_general_read, output_allocation; outside: backend_construction, input_construction, correctness_check.
- **small_grad_trace / pytorch** — inside: torch.linalg.eigh + autograd.grad, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_grad_trace / pytorch** — inside: torch.linalg.lu + autograd.grad, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_grad_trace / pytorch** — inside: torch.linalg.qr + autograd.grad, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_grad_trace / pytorch** — inside: torch.linalg.solve + autograd.grad, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_grad_trace / pytorch** — inside: torch.linalg.svd + autograd.grad, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_grad_trace / trace-prepared** — inside: Runtime::run of the prepared gradient program on the caller-held engine, output_allocation; outside: backend/runtime construction, trace, AD transform and compile, prepare, input_construction, finite-difference correctness check.
- **small_solves / pytorch-solve-batched** — inside: torch.linalg.solve on (batch, k, k), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_solves / pytorch-solve-triangular-batched** — inside: torch.linalg.solve_triangular on (batch, k, k), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_solves / solve-batched** — inside: one batched solve call ([k, k, batch]), output_allocation; outside: backend_construction, input_construction (items pre-sliced), session_entry_exit, correctness_check.
- **small_solves / solve-loop** — inside: batch x rank-2 solve calls, output_allocation per item; outside: backend_construction, input_construction (items pre-sliced), session_entry_exit, correctness_check.
- **small_solves / triangular-solve-loop** — inside: batch x rank-2 solve calls, output_allocation per item; outside: backend_construction, input_construction (items pre-sliced), session_entry_exit, correctness_check.
- **softmax / eager-composed** — inside: max-subtracted composition (reduce_max, broadcast_in_dim, sub, exp, reduce_sum, div/log; select when masked), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **softmax / eager-single-call** — inside: one softmax-family call (composite, all-masked guard included), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **softmax / pytorch** — inside: torch log_softmax, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **softmax / pytorch** — inside: torch masked_softmax, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **softmax / pytorch** — inside: torch softmax, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **softmax / session-single-call** — inside: one softmax-family call (composite, all-masked guard included), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, with_backend_session entry (CpuBackend::new()), correctness_check.
- **softmax_pool / pool4** — inside: axis-0 softmax, output allocation; outside: input and engine construction, session entry, correctness check.
- **softmax_pool / pytorch** — inside: torch softmax, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **softmax_pool / serial** — inside: axis-0 softmax, output allocation; outside: input and engine construction, session entry, correctness check.
- **take_along_axis / eager-gather-prebuilt** — inside: gather with prebuilt (row, batch) index tuples, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **take_along_axis / eager-single-call** — inside: one take_along_axis call (index tuples built inside, then gather), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **take_along_axis / pytorch** — inside: torch.take_along_dim, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **take_along_axis / session-single-call** — inside: one take_along_axis call (index tuples built inside, then gather), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, with_backend_session entry (CpuBackend::new()), correctness_check.
- **tanh_chain / compiled-prepared** — inside: Runtime::run on the caller-held engine, fused region execution, output_allocation; outside: backend/runtime construction, trace and compile, prepare, input_construction, eager session entry (eager arm), correctness_check.
- **tanh_chain / eager-shared** — inside: chain of eager tanh ops, intermediate and output allocation; outside: backend/runtime construction, trace and compile, prepare, input_construction, eager session entry (eager arm), correctness_check.
- **trace_metadata / eager** — inside: triu transpose chain and neg, output allocation; outside: fixture, engine, session, tracing, compilation and preparation, correctness check.
- **trace_metadata / pytorch** — inside: triu and transpose chain, neg, output allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **trace_metadata / trace-fold** — inside: triu transpose chain and neg, output allocation; outside: fixture, engine, session, tracing, compilation and preparation, correctness check.
- **trace_metadata / trace-outputs** — inside: triu transpose chain and neg, output allocation; outside: fixture, engine, session, tracing, compilation and preparation, correctness check.
- **umbrella_add / pytorch** — inside: torch.add allocating; no session concept; outside: input_construction, torch.set_num_threads, correctness_check.
- **umbrella_add / pytorch** — inside: torch.add out= caller output; no session concept; outside: input_construction, torch.set_num_threads, correctness_check.
- **umbrella_add / tenferro** — inside: add_read_into with typed reads and caller output; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_add / tenferro** — inside: typed allocating add in held session; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_axpby / host** — inside: strided alpha*x+beta*y, no staging; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_axpby / tenferro** — inside: strided alpha*x+beta*y, no staging; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_gemm / pytorch** — inside: torch.matmul allocating; equivalent prepared binary contraction; outside: input_construction, torch.set_num_threads, correctness_check.
- **umbrella_gemm / tenferro** — inside: prepared allocating einsum execution; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_lu / host** — inside: independent scalar partial-pivot packed LU, pivots and parity; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_lu / pytorch** — inside: torch.linalg.lu_solve only; factorization prepared outside timing; outside: input_construction, torch.set_num_threads, correctness_check.
- **umbrella_lu / tenferro** — inside: public LinalgBackend packed LU, pivots and parity; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_lu / tenferro** — inside: public LinalgBackend prepared LU solve only; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_matmul / pytorch** — inside: torch.matmul allocating; outside: input_construction, torch.set_num_threads, correctness_check.
- **umbrella_matmul / tenferro** — inside: typed allocating matmul in held session; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_neg / pytorch** — inside: torch.neg allocating; outside: input_construction, torch.set_num_threads, correctness_check.
- **umbrella_neg / tenferro** — inside: typed allocating neg; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
