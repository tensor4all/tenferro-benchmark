- Raw runs: `data/results/amd-cpu/cpu/session_matrix_paired/20260929_131411`

- Command: `BENCH_CONFIRM_CONFIG=data/results/amd-cpu/cpu/confirmation-1946.yaml BENCH_AA_DIR=data/results/amd-cpu/cpu/session_matrix_aa/20260929_124504 BENCHMARK_TARGET_PROFILE=amd-cpu TENFERRO_CPU_FEATURES=cpu-faer scripts/run_paired_timing.sh paired /home/shinaoka/tensor4all/tenferro-rs/.worktrees/bench-main /home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1938-redesign `

# Regression detection (confirm)

- Overall: **OK**

## Deterministic findings

None.

## Timing

| Case | Verdict | Statistic | Rounds | A/A spread | Reason |
|---|---|---:|---:|---:|---|
| `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t1` | NO_CHANGE | 0.967 | 4 | 0.003 | within the declared thresholds |
| `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t4` | IMPROVEMENT | 0.344 | 4 | 0.001 | ratio 0.344, -145872.1 ns/op |
| `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t1` | NO_CHANGE | 1.010 | 4 | 0.004 | within the declared thresholds |
| `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t4` | NO_CHANGE | 1.032 | 4 | 0.003 | within the declared thresholds |
| `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1` | NO_CHANGE | 1.007 | 4 | 0.011 | within the declared thresholds |
| `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4` | IMPROVEMENT | 0.423 | 4 | 0.018 | ratio 0.423, -67967.0 ns/op |
| `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t1` | NO_CHANGE | 1.007 | 4 | 0.016 | within the declared thresholds |
| `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t4` | NO_CHANGE | 1.000 | 4 | 0.007 | within the declared thresholds |
| `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t1` | NO_CHANGE | 0.934 | 4 | 0.081 | within the declared thresholds |
| `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t4` | IMPROVEMENT | 0.588 | 4 | 0.003 | ratio 0.588, -87730.3 ns/op |
| `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t1` | NO_CHANGE | 1.104 | 4 | 0.002 | within the declared thresholds |
| `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t4` | NO_CHANGE | 0.986 | 4 | 0.010 | within the declared thresholds |
| `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t1` | NO_CHANGE | 1.013 | 4 | 0.016 | within the declared thresholds |
| `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t4` | NO_CHANGE | 1.011 | 4 | 0.001 | within the declared thresholds |
| `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t1` | NO_CHANGE | 0.974 | 4 | 0.014 | within the declared thresholds |
| `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t4` | NO_CHANGE | 0.951 | 4 | 0.118 | within the declared thresholds |
| `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t1` | NO_CHANGE | 1.011 | 4 | 0.019 | within the declared thresholds |
| `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t4` | IMPROVEMENT | 0.671 | 4 | 0.001 | ratio 0.671, -17728.6 ns/op |
| `bdot_f64_b64_m64n1k64_direct_into_auto[faer]@t1` | NO_CHANGE | 0.992 | 4 | 0.002 | within the declared thresholds |
| `bdot_f64_b64_m64n1k64_direct_into_auto[faer]@t4` | NO_CHANGE | 0.991 | 4 | 0.051 | within the declared thresholds |
| `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1` | NO_CHANGE | 1.010 | 4 | 0.002 | within the declared thresholds |
| `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4` | IMPROVEMENT | 0.481 | 4 | 0.005 | ratio 0.481, -66314.9 ns/op |
| `beinsum_f64_b1024_m4n4k4_direct_into_auto[faer]@t1` | NO_CHANGE | 1.017 | 4 | 0.003 | within the declared thresholds |
| `beinsum_f64_b1024_m4n4k4_direct_into_auto[faer]@t4` | NO_CHANGE | 1.031 | 4 | 0.016 | within the declared thresholds |
| `chain3_f64_n4_einsum_alloc[faer]@t1` | NO_CHANGE | 1.007 | 4 | 0.011 | within the declared thresholds |
| `chain3_f64_n4_einsum_alloc[faer]@t4` | NO_CHANGE | 1.037 | 4 | 0.145 | within the declared thresholds |
| `hadamard_f64_m64n64_dot_alloc[faer]@t1` | NO_CHANGE | 1.011 | 4 | 0.002 | within the declared thresholds |
| `hadamard_f64_m64n64_dot_alloc[faer]@t4` | NO_CHANGE | 1.005 | 4 | 0.005 | within the declared thresholds |
| `hadamard_f64_m64n64_einsum_alloc[faer]@t1` | NO_CHANGE | 0.992 | 4 | 0.014 | within the declared thresholds |
| `hadamard_f64_m64n64_einsum_alloc[faer]@t4` | NO_CHANGE | 0.985 | 4 | 0.014 | within the declared thresholds |
| `matmul_n16_count1024[blas]@t1` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `matmul_n16_count1024[blas]@t4` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `matmul_n16_count1024[faer]@t1` | NO_CHANGE | 1.022 | 4 | 0.012 | within the declared thresholds |
| `matmul_n16_count1024[faer]@t4` | NO_CHANGE | 0.869 | 4 | 0.072 | within the declared thresholds |
| `matmul_n16_count1024[pytorch]@t1` | NO_CHANGE | 1.071 | 4 | 0.009 | within the declared thresholds |
| `matmul_n16_count1024[pytorch]@t4` | NO_CHANGE | 1.121 | 4 | 0.073 | within the declared thresholds |
| `matmul_n2_count1024[blas]@t1` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `matmul_n2_count1024[blas]@t4` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `matmul_n2_count1024[faer]@t1` | NO_CHANGE | 1.017 | 4 | 0.001 | within the declared thresholds |
| `matmul_n2_count1024[faer]@t4` | NO_CHANGE | 1.014 | 4 | 0.006 | within the declared thresholds |
| `matmul_n2_count1024[pytorch]@t1` | NO_CHANGE | 1.131 | 4 | 0.001 | within the declared thresholds |
| `matmul_n2_count1024[pytorch]@t4` | NO_CHANGE | 0.946 | 4 | 0.006 | within the declared thresholds |
| `matmul_n32_count1024[blas]@t1` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `matmul_n32_count1024[blas]@t4` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `matmul_n32_count1024[faer]@t1` | NO_CHANGE | 1.005 | 4 | 0.017 | within the declared thresholds |
| `matmul_n32_count1024[faer]@t4` | NO_CHANGE | 0.984 | 4 | 0.008 | within the declared thresholds |
| `matmul_n32_count1024[pytorch]@t1` | NO_CHANGE | 1.022 | 4 | 0.025 | within the declared thresholds |
| `matmul_n32_count1024[pytorch]@t4` | NO_CHANGE | 1.017 | 4 | 0.017 | within the declared thresholds |
| `matmul_n4_count1024[blas]@t1` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `matmul_n4_count1024[blas]@t4` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `matmul_n4_count1024[faer]@t1` | NO_CHANGE | 0.991 | 4 | 0.005 | within the declared thresholds |
| `matmul_n4_count1024[faer]@t4` | NO_CHANGE | 0.999 | 4 | 0.002 | within the declared thresholds |
| `matmul_n4_count1024[pytorch]@t1` | NO_CHANGE | 1.062 | 4 | 0.019 | within the declared thresholds |
| `matmul_n4_count1024[pytorch]@t4` | NO_CHANGE | 1.018 | 4 | 0.018 | within the declared thresholds |
| `matmul_n8_count1024[blas]@t1` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `matmul_n8_count1024[blas]@t4` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `matmul_n8_count1024[faer]@t1` | NO_CHANGE | 1.003 | 4 | 0.146 | within the declared thresholds |
| `matmul_n8_count1024[faer]@t4` | NO_CHANGE | 0.999 | 4 | 0.002 | within the declared thresholds |
| `matmul_n8_count1024[pytorch]@t1` | NO_CHANGE | 1.059 | 4 | 0.015 | within the declared thresholds |
| `matmul_n8_count1024[pytorch]@t4` | NO_CHANGE | 1.041 | 4 | 0.007 | within the declared thresholds |
| `solve_n16_count1024[blas]@t1` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `solve_n16_count1024[blas]@t4` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `solve_n16_count1024[faer]@t1` | NO_CHANGE | 1.057 | 4 | 0.008 | within the declared thresholds |
| `solve_n16_count1024[faer]@t4` | NO_CHANGE | 0.961 | 4 | 0.006 | within the declared thresholds |
| `solve_n16_count1024[pytorch]@t1` | NO_CHANGE | 1.041 | 4 | 0.002 | within the declared thresholds |
| `solve_n16_count1024[pytorch]@t4` | NO_CHANGE | 0.980 | 4 | 0.010 | within the declared thresholds |
| `solve_n2_count1024[blas]@t1` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `solve_n2_count1024[blas]@t4` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `solve_n2_count1024[faer]@t1` | NO_CHANGE | 0.979 | 4 | 0.028 | within the declared thresholds |
| `solve_n2_count1024[faer]@t4` | NO_CHANGE | 0.981 | 4 | 0.009 | within the declared thresholds |
| `solve_n2_count1024[pytorch]@t1` | NO_CHANGE | 0.974 | 4 | 0.029 | within the declared thresholds |
| `solve_n2_count1024[pytorch]@t4` | NO_CHANGE | 0.976 | 4 | 0.000 | within the declared thresholds |
| `solve_n32_count1024[blas]@t1` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `solve_n32_count1024[blas]@t4` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `solve_n32_count1024[faer]@t1` | NO_CHANGE | 1.005 | 4 | 0.004 | within the declared thresholds |
| `solve_n32_count1024[faer]@t4` | NO_CHANGE | 1.043 | 4 | 0.010 | within the declared thresholds |
| `solve_n32_count1024[pytorch]@t1` | NO_CHANGE | 1.023 | 4 | 0.006 | within the declared thresholds |
| `solve_n32_count1024[pytorch]@t4` | NO_CHANGE | 1.030 | 4 | 0.022 | within the declared thresholds |
| `solve_n4_count1024[blas]@t1` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `solve_n4_count1024[blas]@t4` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `solve_n4_count1024[faer]@t1` | NO_CHANGE | 0.991 | 4 | 0.008 | within the declared thresholds |
| `solve_n4_count1024[faer]@t4` | NO_CHANGE | 0.974 | 4 | 0.001 | within the declared thresholds |
| `solve_n4_count1024[pytorch]@t1` | NO_CHANGE | 1.001 | 4 | 0.015 | within the declared thresholds |
| `solve_n4_count1024[pytorch]@t4` | NO_CHANGE | 0.999 | 4 | 0.005 | within the declared thresholds |
| `solve_n8_count1024[blas]@t1` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `solve_n8_count1024[blas]@t4` | INCONCLUSIVE | — | — | — | a round is missing or did not pass |
| `solve_n8_count1024[faer]@t1` | NO_CHANGE | 0.976 | 4 | 0.002 | within the declared thresholds |
| `solve_n8_count1024[faer]@t4` | NO_CHANGE | 0.987 | 4 | 0.076 | within the declared thresholds |
| `solve_n8_count1024[pytorch]@t1` | NO_CHANGE | 0.938 | 4 | 0.028 | within the declared thresholds |
| `solve_n8_count1024[pytorch]@t4` | NO_CHANGE | 1.010 | 4 | 0.017 | within the declared thresholds |
| `stream_f64_fixed_len32_einsum_alloc[faer]@t1` | NO_CHANGE | 0.990 | 4 | 0.004 | within the declared thresholds |
| `stream_f64_fixed_len32_einsum_alloc[faer]@t4` | NO_CHANGE | 0.978 | 4 | 0.016 | within the declared thresholds |
| `stream_f64_fresh_len32_einsum_alloc[faer]@t1` | NO_CHANGE | 0.983 | 4 | 0.007 | within the declared thresholds |
| `stream_f64_fresh_len32_einsum_alloc[faer]@t4` | NO_CHANGE | 0.988 | 4 | 0.002 | within the declared thresholds |
| `stream_f64_mixed_len32_einsum_alloc[faer]@t1` | NO_CHANGE | 1.005 | 4 | 0.004 | within the declared thresholds |
| `stream_f64_mixed_len32_einsum_alloc[faer]@t4` | NO_CHANGE | 0.994 | 4 | 0.007 | within the declared thresholds |
| `stream_f64_strides_len32_einsum_alloc[faer]@t1` | NO_CHANGE | 0.986 | 4 | 0.008 | within the declared thresholds |
| `stream_f64_strides_len32_einsum_alloc[faer]@t4` | NO_CHANGE | 0.991 | 4 | 0.006 | within the declared thresholds |
