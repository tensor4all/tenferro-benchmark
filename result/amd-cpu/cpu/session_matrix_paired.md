- Raw runs: `data/results/amd-cpu/cpu/session_matrix_paired/20260929_135012`

- Command: `BENCH_CONFIRM_CONFIG=data/results/amd-cpu/cpu/confirmation-1946-blas.yaml BENCH_AA_DIR=data/results/amd-cpu/cpu/session_matrix_aa/20260929_133603 BENCHMARK_TARGET_PROFILE=amd-cpu TENFERRO_CPU_FEATURES=system-openblas scripts/run_paired_timing.sh paired /home/shinaoka/tensor4all/tenferro-rs/.worktrees/bench-main /home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1938-redesign `

# Regression detection (confirm)

- Overall: **OK**

## Deterministic findings

None.

## Timing

| Case | Verdict | Statistic | Rounds | A/A spread | Reason |
|---|---|---:|---:|---:|---|
| `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t1` | NO_CHANGE | 1.016 | 4 | 0.106 | within the declared thresholds |
| `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t4` | IMPROVEMENT | 0.340 | 4 | 0.000 | ratio 0.340, -151367.8 ns/op |
| `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t1` | NO_CHANGE | 1.009 | 4 | 0.004 | within the declared thresholds |
| `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t4` | NO_CHANGE | 1.000 | 4 | 0.013 | within the declared thresholds |
| `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1` | NO_CHANGE | 0.991 | 4 | 0.001 | within the declared thresholds |
| `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4` | IMPROVEMENT | 0.382 | 4 | 0.002 | ratio 0.382, -77803.1 ns/op |
| `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t1` | NO_CHANGE | 1.037 | 4 | 0.003 | within the declared thresholds |
| `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t4` | NO_CHANGE | 0.994 | 4 | 0.060 | within the declared thresholds |
| `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t1` | NO_CHANGE | 1.000 | 4 | 0.000 | within the declared thresholds |
| `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t4` | IMPROVEMENT | 0.408 | 4 | 0.001 | ratio 0.408, -128527.1 ns/op |
| `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t1` | NO_CHANGE | 1.034 | 4 | 0.006 | within the declared thresholds |
| `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t4` | INCONCLUSIVE | 1.119 | 4 | 0.200 | A/A spread 0.200 > declared 0.15 |
| `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t1` | NO_CHANGE | 1.003 | 4 | 0.023 | within the declared thresholds |
| `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t4` | NO_CHANGE | 1.007 | 4 | 0.008 | within the declared thresholds |
| `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t1` | NO_CHANGE | 1.009 | 4 | 0.001 | within the declared thresholds |
| `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t4` | NO_CHANGE | 0.911 | 4 | 0.003 | within the declared thresholds |
| `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t1` | NO_CHANGE | 0.981 | 4 | 0.026 | within the declared thresholds |
| `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t4` | IMPROVEMENT | 0.735 | 4 | 0.111 | ratio 0.735, -12029.7 ns/op |
| `bdot_f64_b64_m64n1k64_direct_into_auto[faer]@t1` | NO_CHANGE | 1.016 | 4 | 0.007 | within the declared thresholds |
| `bdot_f64_b64_m64n1k64_direct_into_auto[faer]@t4` | NO_CHANGE | 0.992 | 4 | 0.066 | within the declared thresholds |
| `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1` | NO_CHANGE | 1.022 | 4 | 0.003 | within the declared thresholds |
| `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4` | IMPROVEMENT | 0.447 | 4 | 0.005 | ratio 0.447, -75948.3 ns/op |
| `beinsum_f64_b1024_m4n4k4_direct_into_auto[faer]@t1` | NO_CHANGE | 1.035 | 4 | 0.006 | within the declared thresholds |
| `beinsum_f64_b1024_m4n4k4_direct_into_auto[faer]@t4` | NO_CHANGE | 1.007 | 4 | 0.005 | within the declared thresholds |
| `chain3_f64_n4_einsum_alloc[faer]@t1` | NO_CHANGE | 1.023 | 4 | 0.004 | within the declared thresholds |
| `chain3_f64_n4_einsum_alloc[faer]@t4` | NO_CHANGE | 1.028 | 4 | 0.017 | within the declared thresholds |
| `hadamard_f64_m64n64_dot_alloc[faer]@t1` | NO_CHANGE | 1.001 | 4 | 0.059 | within the declared thresholds |
| `hadamard_f64_m64n64_dot_alloc[faer]@t4` | NO_CHANGE | 1.005 | 4 | 0.003 | within the declared thresholds |
| `hadamard_f64_m64n64_einsum_alloc[faer]@t1` | NO_CHANGE | 0.998 | 4 | 0.045 | within the declared thresholds |
| `hadamard_f64_m64n64_einsum_alloc[faer]@t4` | NO_CHANGE | 0.997 | 4 | 0.006 | within the declared thresholds |
| `matmul_n16_count1024[blas]@t1` | NO_CHANGE | 1.022 | 4 | 0.003 | within the declared thresholds |
| `matmul_n16_count1024[blas]@t4` | NO_CHANGE | 1.002 | 4 | 0.002 | within the declared thresholds |
| `matmul_n16_count1024[faer]@t1` | NO_CHANGE | 0.935 | 4 | 0.030 | within the declared thresholds |
| `matmul_n16_count1024[faer]@t4` | NO_CHANGE | 1.011 | 4 | 0.008 | within the declared thresholds |
| `matmul_n16_count1024[pytorch]@t1` | NO_CHANGE | 0.967 | 4 | 0.007 | within the declared thresholds |
| `matmul_n16_count1024[pytorch]@t4` | NO_CHANGE | 0.977 | 4 | 0.068 | within the declared thresholds |
| `matmul_n2_count1024[blas]@t1` | NO_CHANGE | 1.093 | 4 | 0.001 | within the declared thresholds |
| `matmul_n2_count1024[blas]@t4` | NO_CHANGE | 1.033 | 4 | 0.002 | within the declared thresholds |
| `matmul_n2_count1024[faer]@t1` | NO_CHANGE | 1.074 | 4 | 0.000 | within the declared thresholds |
| `matmul_n2_count1024[faer]@t4` | NO_CHANGE | 1.063 | 4 | 0.095 | within the declared thresholds |
| `matmul_n2_count1024[pytorch]@t1` | NO_CHANGE | 0.996 | 4 | 0.005 | within the declared thresholds |
| `matmul_n2_count1024[pytorch]@t4` | NO_CHANGE | 0.974 | 4 | 0.017 | within the declared thresholds |
| `matmul_n32_count1024[blas]@t1` | NO_CHANGE | 0.940 | 4 | 0.043 | within the declared thresholds |
| `matmul_n32_count1024[blas]@t4` | NO_CHANGE | 0.953 | 4 | 0.010 | within the declared thresholds |
| `matmul_n32_count1024[faer]@t1` | NO_CHANGE | 0.949 | 4 | 0.012 | within the declared thresholds |
| `matmul_n32_count1024[faer]@t4` | NO_CHANGE | 1.021 | 4 | 0.036 | within the declared thresholds |
| `matmul_n32_count1024[pytorch]@t1` | NO_CHANGE | 0.972 | 4 | 0.018 | within the declared thresholds |
| `matmul_n32_count1024[pytorch]@t4` | NO_CHANGE | 0.992 | 4 | 0.019 | within the declared thresholds |
| `matmul_n4_count1024[blas]@t1` | NO_CHANGE | 1.015 | 4 | 0.077 | within the declared thresholds |
| `matmul_n4_count1024[blas]@t4` | NO_CHANGE | 1.025 | 4 | 0.013 | within the declared thresholds |
| `matmul_n4_count1024[faer]@t1` | NO_CHANGE | 1.063 | 4 | 0.005 | within the declared thresholds |
| `matmul_n4_count1024[faer]@t4` | NO_CHANGE | 1.076 | 4 | 0.002 | within the declared thresholds |
| `matmul_n4_count1024[pytorch]@t1` | NO_CHANGE | 1.004 | 4 | 0.027 | within the declared thresholds |
| `matmul_n4_count1024[pytorch]@t4` | NO_CHANGE | 0.959 | 4 | 0.002 | within the declared thresholds |
| `matmul_n8_count1024[blas]@t1` | INCONCLUSIVE | 1.027 | 4 | 0.158 | A/A spread 0.158 > declared 0.15 |
| `matmul_n8_count1024[blas]@t4` | NO_CHANGE | 1.020 | 4 | 0.005 | within the declared thresholds |
| `matmul_n8_count1024[faer]@t1` | NO_CHANGE | 1.035 | 4 | 0.038 | within the declared thresholds |
| `matmul_n8_count1024[faer]@t4` | NO_CHANGE | 1.035 | 4 | 0.000 | within the declared thresholds |
| `matmul_n8_count1024[pytorch]@t1` | NO_CHANGE | 0.971 | 4 | 0.001 | within the declared thresholds |
| `matmul_n8_count1024[pytorch]@t4` | NO_CHANGE | 1.015 | 4 | 0.016 | within the declared thresholds |
| `solve_n16_count1024[blas]@t1` | NO_CHANGE | 1.049 | 4 | 0.002 | within the declared thresholds |
| `solve_n16_count1024[blas]@t4` | NO_CHANGE | 1.020 | 4 | 0.007 | within the declared thresholds |
| `solve_n16_count1024[faer]@t1` | NO_CHANGE | 1.032 | 4 | 0.105 | within the declared thresholds |
| `solve_n16_count1024[faer]@t4` | NO_CHANGE | 0.994 | 4 | 0.004 | within the declared thresholds |
| `solve_n16_count1024[pytorch]@t1` | NO_CHANGE | 1.014 | 4 | 0.036 | within the declared thresholds |
| `solve_n16_count1024[pytorch]@t4` | NO_CHANGE | 1.042 | 4 | 0.003 | within the declared thresholds |
| `solve_n2_count1024[blas]@t1` | NO_CHANGE | 0.956 | 4 | 0.011 | within the declared thresholds |
| `solve_n2_count1024[blas]@t4` | NO_CHANGE | 0.974 | 4 | 0.055 | within the declared thresholds |
| `solve_n2_count1024[faer]@t1` | NO_CHANGE | 0.954 | 4 | 0.073 | within the declared thresholds |
| `solve_n2_count1024[faer]@t4` | NO_CHANGE | 0.954 | 4 | 0.009 | within the declared thresholds |
| `solve_n2_count1024[pytorch]@t1` | NO_CHANGE | 0.964 | 4 | 0.021 | within the declared thresholds |
| `solve_n2_count1024[pytorch]@t4` | NO_CHANGE | 0.988 | 4 | 0.010 | within the declared thresholds |
| `solve_n32_count1024[blas]@t1` | NO_CHANGE | 0.993 | 4 | 0.012 | within the declared thresholds |
| `solve_n32_count1024[blas]@t4` | NO_CHANGE | 0.986 | 4 | 0.002 | within the declared thresholds |
| `solve_n32_count1024[faer]@t1` | NO_CHANGE | 0.963 | 4 | 0.006 | within the declared thresholds |
| `solve_n32_count1024[faer]@t4` | NO_CHANGE | 1.050 | 4 | 0.001 | within the declared thresholds |
| `solve_n32_count1024[pytorch]@t1` | NO_CHANGE | 1.045 | 4 | 0.000 | within the declared thresholds |
| `solve_n32_count1024[pytorch]@t4` | NO_CHANGE | 0.993 | 4 | 0.005 | within the declared thresholds |
| `solve_n4_count1024[blas]@t1` | NO_CHANGE | 0.899 | 4 | 0.121 | within the declared thresholds |
| `solve_n4_count1024[blas]@t4` | NO_CHANGE | 1.017 | 4 | 0.018 | within the declared thresholds |
| `solve_n4_count1024[faer]@t1` | NO_CHANGE | 0.945 | 4 | 0.002 | within the declared thresholds |
| `solve_n4_count1024[faer]@t4` | NO_CHANGE | 0.981 | 4 | 0.003 | within the declared thresholds |
| `solve_n4_count1024[pytorch]@t1` | NO_CHANGE | 0.957 | 4 | 0.001 | within the declared thresholds |
| `solve_n4_count1024[pytorch]@t4` | NO_CHANGE | 1.122 | 4 | 0.004 | within the declared thresholds |
| `solve_n8_count1024[blas]@t1` | NO_CHANGE | 0.855 | 4 | 0.008 | within the declared thresholds |
| `solve_n8_count1024[blas]@t4` | NO_CHANGE | 1.022 | 4 | 0.044 | within the declared thresholds |
| `solve_n8_count1024[faer]@t1` | NO_CHANGE | 0.935 | 4 | 0.065 | within the declared thresholds |
| `solve_n8_count1024[faer]@t4` | NO_CHANGE | 0.983 | 4 | 0.015 | within the declared thresholds |
| `solve_n8_count1024[pytorch]@t1` | NO_CHANGE | 1.010 | 4 | 0.034 | within the declared thresholds |
| `solve_n8_count1024[pytorch]@t4` | NO_CHANGE | 1.036 | 4 | 0.020 | within the declared thresholds |
| `stream_f64_fixed_len32_einsum_alloc[faer]@t1` | NO_CHANGE | 0.983 | 4 | 0.016 | within the declared thresholds |
| `stream_f64_fixed_len32_einsum_alloc[faer]@t4` | NO_CHANGE | 1.017 | 4 | 0.013 | within the declared thresholds |
| `stream_f64_fresh_len32_einsum_alloc[faer]@t1` | NO_CHANGE | 0.994 | 4 | 0.009 | within the declared thresholds |
| `stream_f64_fresh_len32_einsum_alloc[faer]@t4` | NO_CHANGE | 1.000 | 4 | 0.005 | within the declared thresholds |
| `stream_f64_mixed_len32_einsum_alloc[faer]@t1` | NO_CHANGE | 1.003 | 4 | 0.001 | within the declared thresholds |
| `stream_f64_mixed_len32_einsum_alloc[faer]@t4` | NO_CHANGE | 0.982 | 4 | 0.008 | within the declared thresholds |
| `stream_f64_strides_len32_einsum_alloc[faer]@t1` | NO_CHANGE | 1.006 | 4 | 0.006 | within the declared thresholds |
| `stream_f64_strides_len32_einsum_alloc[faer]@t4` | NO_CHANGE | 0.982 | 4 | 0.022 | within the declared thresholds |
