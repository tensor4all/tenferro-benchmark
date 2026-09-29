- Raw runs: `data/results/amd-cpu/cpu/session_matrix_aa/20260929_124504`

- Command: `BENCH_AA_ROUNDS=4 BENCH_AA_WARMUPS=3 BENCH_AA_RUNS=15 BENCH_AA_THREADS='1 4' BENCH_COVERAGE=quick BENCHMARK_TARGET_PROFILE=amd-cpu TENFERRO_CPU_FEATURES=cpu-faer scripts/run_paired_timing.sh aa /home/shinaoka/tensor4all/tenferro-rs/.worktrees/bench-main `

# A/A noise characterization

Same tenferro-rs build in both arms, balanced order. Use these spreads to declare `noise.max_aa_relative_spread`, `noise.max_cov` and the relative threshold in the confirmation config **before** any candidate run. No verdicts are produced here.

- Balanced order: yes

| Case | A/A statistic | Spread | Max CoV | Rounds | Round ratios |
|---|---:|---:|---:|---:|---|
| `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t1` | 1.0034 | 0.0034 | 0.0078 | 4 | 1.0084, 0.9954, 0.9983, 1.2873 |
| `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t4` | 1.0007 | 0.0007 | 0.0098 | 4 | 1.0005, 1.0009, 0.7746, 1.0027 |
| `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t1` | 1.0036 | 0.0036 | 0.0043 | 4 | 1.0023, 1.0072, 0.9992, 1.0049 |
| `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t4` | 0.9969 | 0.0031 | 0.1036 | 4 | 0.9896, 1.1648, 1.0043, 0.8448 |
| `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1` | 1.0107 | 0.0107 | 0.0866 | 4 | 1.0131, 1.0083, 1.0372, 0.8005 |
| `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4` | 0.9824 | 0.0176 | 0.1005 | 4 | 1.0077, 1.0020, 0.6640, 0.9627 |
| `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t1` | 0.9835 | 0.0165 | 0.0561 | 4 | 0.9907, 0.9917, 0.9764, 0.9631 |
| `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t4` | 1.0070 | 0.0070 | 0.1295 | 4 | 0.9745, 0.9606, 1.0396, 1.0682 |
| `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t1` | 0.9187 | 0.0813 | 0.0960 | 4 | 0.8285, 0.8215, 1.0090, 1.1746 |
| `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t4` | 0.9970 | 0.0030 | 0.0551 | 4 | 0.9969, 0.9971, 1.0040, 0.9926 |
| `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t1` | 1.0019 | 0.0019 | 0.0680 | 4 | 0.8407, 1.2040, 1.0046, 0.9993 |
| `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t4` | 0.9898 | 0.0102 | 0.0693 | 4 | 0.9799, 0.9752, 0.9996, 1.0261 |
| `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t1` | 1.0159 | 0.0159 | 0.0067 | 4 | 1.0323, 1.0419, 0.9893, 0.9995 |
| `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t4` | 0.9988 | 0.0012 | 0.0100 | 4 | 1.0016, 1.0036, 0.9876, 0.9960 |
| `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t1` | 1.0138 | 0.0138 | 0.0843 | 4 | 0.9957, 1.0364, 1.0294, 0.9982 |
| `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t4` | 0.8819 | 0.1181 | 0.1492 | 4 | 0.9996, 0.7642, 0.7570, 1.2554 |
| `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t1` | 0.9805 | 0.0195 | 0.0644 | 4 | 1.0103, 0.9902, 0.9708, 0.8785 |
| `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t4` | 0.9991 | 0.0009 | 0.0476 | 4 | 1.0238, 1.0053, 0.9752, 0.9928 |
| `bdot_f64_b64_m64n1k64_direct_into_auto[faer]@t1` | 1.0015 | 0.0015 | 0.0833 | 4 | 1.0206, 1.0062, 0.9968, 0.9589 |
| `bdot_f64_b64_m64n1k64_direct_into_auto[faer]@t4` | 0.9493 | 0.0507 | 0.0632 | 4 | 0.9197, 0.9154, 1.1443, 0.9789 |
| `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1` | 1.0017 | 0.0017 | 0.0827 | 4 | 1.0006, 1.0028, 0.9907, 1.0058 |
| `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4` | 1.0048 | 0.0048 | 0.0200 | 4 | 1.0036, 1.0117, 1.0059, 1.0007 |
| `beinsum_f64_b1024_m4n4k4_direct_into_auto[faer]@t1` | 1.0026 | 0.0026 | 0.0708 | 4 | 0.9897, 0.9992, 1.0060, 1.0092 |
| `beinsum_f64_b1024_m4n4k4_direct_into_auto[faer]@t4` | 0.9843 | 0.0157 | 0.0892 | 4 | 0.9843, 0.9843, 1.0093, 0.9398 |
| `chain3_f64_n4_einsum_alloc[faer]@t1` | 1.0106 | 0.0106 | 0.3050 | 4 | 0.9650, 1.0230, 1.0076, 1.0136 |
| `chain3_f64_n4_einsum_alloc[faer]@t4` | 1.1448 | 0.1448 | 0.1692 | 4 | 0.9532, 1.0075, 1.2821, 1.3638 |
| `hadamard_f64_m64n64_dot_alloc[faer]@t1` | 0.9982 | 0.0018 | 0.0879 | 4 | 0.9912, 1.0089, 0.8105, 1.0051 |
| `hadamard_f64_m64n64_dot_alloc[faer]@t4` | 1.0054 | 0.0054 | 0.0897 | 4 | 1.0267, 1.0129, 0.8347, 0.9979 |
| `hadamard_f64_m64n64_einsum_alloc[faer]@t1` | 0.9861 | 0.0139 | 0.0165 | 4 | 0.9687, 0.9595, 1.0035, 1.0225 |
| `hadamard_f64_m64n64_einsum_alloc[faer]@t4` | 0.9855 | 0.0145 | 0.1535 | 4 | 0.9776, 0.8227, 1.0235, 0.9934 |
| `matmul_n16_count1024[blas]@t1` | — | — | — | — | incomplete rounds |
| `matmul_n16_count1024[blas]@t4` | — | — | — | — | incomplete rounds |
| `matmul_n16_count1024[faer]@t1` | 1.0125 | 0.0125 | 0.0861 | 4 | 0.9084, 1.0114, 1.0136, 1.0257 |
| `matmul_n16_count1024[faer]@t4` | 0.9276 | 0.0724 | 0.0904 | 4 | 0.8389, 0.8451, 1.0101, 1.0138 |
| `matmul_n16_count1024[pytorch]@t1` | 1.0093 | 0.0093 | 0.1391 | 4 | 1.0148, 1.0653, 0.9848, 1.0039 |
| `matmul_n16_count1024[pytorch]@t4` | 0.9270 | 0.0730 | 0.1323 | 4 | 0.9355, 0.9135, 0.9184, 0.9985 |
| `matmul_n2_count1024[blas]@t1` | — | — | — | — | incomplete rounds |
| `matmul_n2_count1024[blas]@t4` | — | — | — | — | incomplete rounds |
| `matmul_n2_count1024[faer]@t1` | 1.0006 | 0.0006 | 0.0882 | 4 | 0.8342, 1.0090, 1.0330, 0.9922 |
| `matmul_n2_count1024[faer]@t4` | 0.9941 | 0.0059 | 0.1826 | 4 | 0.9964, 0.9771, 1.0604, 0.9919 |
| `matmul_n2_count1024[pytorch]@t1` | 0.9986 | 0.0014 | 0.0865 | 4 | 1.0637, 0.9679, 1.0168, 0.9805 |
| `matmul_n2_count1024[pytorch]@t4` | 0.9940 | 0.0060 | 0.0898 | 4 | 0.9877, 0.9882, 0.9998, 1.0037 |
| `matmul_n32_count1024[blas]@t1` | — | — | — | — | incomplete rounds |
| `matmul_n32_count1024[blas]@t4` | — | — | — | — | incomplete rounds |
| `matmul_n32_count1024[faer]@t1` | 0.9835 | 0.0165 | 0.0820 | 4 | 0.9728, 0.9900, 0.9793, 0.9877 |
| `matmul_n32_count1024[faer]@t4` | 0.9919 | 0.0081 | 0.0680 | 4 | 1.0363, 0.9778, 0.9829, 1.0009 |
| `matmul_n32_count1024[pytorch]@t1` | 0.9750 | 0.0250 | 0.0678 | 4 | 0.9455, 1.0067, 0.9503, 0.9997 |
| `matmul_n32_count1024[pytorch]@t4` | 0.9831 | 0.0169 | 0.0610 | 4 | 1.0411, 0.9826, 0.9602, 0.9836 |
| `matmul_n4_count1024[blas]@t1` | — | — | — | — | incomplete rounds |
| `matmul_n4_count1024[blas]@t4` | — | — | — | — | incomplete rounds |
| `matmul_n4_count1024[faer]@t1` | 1.0052 | 0.0052 | 0.0660 | 4 | 0.9975, 1.0130, 0.9927, 1.1961 |
| `matmul_n4_count1024[faer]@t4` | 1.0018 | 0.0018 | 0.0119 | 4 | 0.9870, 1.0004, 1.0147, 1.0031 |
| `matmul_n4_count1024[pytorch]@t1` | 1.0194 | 0.0194 | 0.0981 | 4 | 1.0059, 1.0456, 0.9616, 1.0330 |
| `matmul_n4_count1024[pytorch]@t4` | 1.0185 | 0.0185 | 0.1771 | 4 | 0.9861, 1.0497, 1.0300, 1.0069 |
| `matmul_n8_count1024[blas]@t1` | — | — | — | — | incomplete rounds |
| `matmul_n8_count1024[blas]@t4` | — | — | — | — | incomplete rounds |
| `matmul_n8_count1024[faer]@t1` | 0.8542 | 0.1458 | 0.0916 | 4 | 0.8528, 0.8330, 0.9937, 0.8556 |
| `matmul_n8_count1024[faer]@t4` | 0.9980 | 0.0020 | 0.0448 | 4 | 0.9986, 0.9963, 0.9974, 1.0030 |
| `matmul_n8_count1024[pytorch]@t1` | 1.0153 | 0.0153 | 0.0819 | 4 | 1.0085, 0.9960, 1.0220, 1.0522 |
| `matmul_n8_count1024[pytorch]@t4` | 0.9926 | 0.0074 | 0.2149 | 4 | 0.9601, 1.0037, 1.0295, 0.9816 |
| `solve_n16_count1024[blas]@t1` | — | — | — | — | incomplete rounds |
| `solve_n16_count1024[blas]@t4` | — | — | — | — | incomplete rounds |
| `solve_n16_count1024[faer]@t1` | 1.0084 | 0.0084 | 0.0784 | 4 | 0.8677, 1.0672, 1.0351, 0.9817 |
| `solve_n16_count1024[faer]@t4` | 1.0060 | 0.0060 | 0.0703 | 4 | 0.9956, 1.0355, 1.0164, 0.9954 |
| `solve_n16_count1024[pytorch]@t1` | 0.9978 | 0.0022 | 0.0817 | 4 | 0.9617, 1.0424, 0.9899, 1.0058 |
| `solve_n16_count1024[pytorch]@t4` | 1.0104 | 0.0104 | 0.0651 | 4 | 1.0336, 0.9785, 1.0078, 1.0130 |
| `solve_n2_count1024[blas]@t1` | — | — | — | — | incomplete rounds |
| `solve_n2_count1024[blas]@t4` | — | — | — | — | incomplete rounds |
| `solve_n2_count1024[faer]@t1` | 0.9719 | 0.0281 | 0.0858 | 4 | 0.9436, 1.0001, 1.1696, 0.9084 |
| `solve_n2_count1024[faer]@t4` | 0.9908 | 0.0092 | 0.0217 | 4 | 1.0279, 0.9774, 1.0042, 0.9631 |
| `solve_n2_count1024[pytorch]@t1` | 0.9707 | 0.0293 | 0.0714 | 4 | 0.9317, 0.9886, 0.9553, 0.9861 |
| `solve_n2_count1024[pytorch]@t4` | 1.0003 | 0.0003 | 0.0893 | 4 | 0.9825, 1.0181, 0.9787, 1.0345 |
| `solve_n32_count1024[blas]@t1` | — | — | — | — | incomplete rounds |
| `solve_n32_count1024[blas]@t4` | — | — | — | — | incomplete rounds |
| `solve_n32_count1024[faer]@t1` | 0.9959 | 0.0041 | 0.0528 | 4 | 0.9557, 1.0237, 0.9892, 1.0025 |
| `solve_n32_count1024[faer]@t4` | 1.0095 | 0.0095 | 0.0515 | 4 | 1.0168, 1.0051, 1.0107, 1.0083 |
| `solve_n32_count1024[pytorch]@t1` | 1.0059 | 0.0059 | 0.0714 | 4 | 0.9242, 1.0247, 0.9978, 1.0140 |
| `solve_n32_count1024[pytorch]@t4` | 1.0215 | 0.0215 | 0.0525 | 4 | 1.0183, 1.0248, 0.9670, 1.0331 |
| `solve_n4_count1024[blas]@t1` | — | — | — | — | incomplete rounds |
| `solve_n4_count1024[blas]@t4` | — | — | — | — | incomplete rounds |
| `solve_n4_count1024[faer]@t1` | 1.0081 | 0.0081 | 0.0884 | 4 | 0.8419, 0.9954, 1.0207, 1.1786 |
| `solve_n4_count1024[faer]@t4` | 0.9994 | 0.0006 | 0.0789 | 4 | 0.9899, 0.9995, 1.0045, 0.9993 |
| `solve_n4_count1024[pytorch]@t1` | 0.9846 | 0.0154 | 0.0815 | 4 | 1.0217, 1.0014, 0.9542, 0.9677 |
| `solve_n4_count1024[pytorch]@t4` | 1.0047 | 0.0047 | 0.0711 | 4 | 0.9939, 0.9610, 1.0154, 1.0197 |
| `solve_n8_count1024[blas]@t1` | — | — | — | — | incomplete rounds |
| `solve_n8_count1024[blas]@t4` | — | — | — | — | incomplete rounds |
| `solve_n8_count1024[faer]@t1` | 0.9980 | 0.0020 | 0.0885 | 4 | 0.9526, 1.1957, 1.0089, 0.9871 |
| `solve_n8_count1024[faer]@t4` | 0.9236 | 0.0764 | 0.0917 | 4 | 0.8416, 1.1601, 1.0055, 0.8270 |
| `solve_n8_count1024[pytorch]@t1` | 0.9718 | 0.0282 | 0.0652 | 4 | 0.9542, 0.9738, 0.9971, 0.9698 |
| `solve_n8_count1024[pytorch]@t4` | 1.0174 | 0.0174 | 0.0883 | 4 | 1.0189, 1.0207, 1.0160, 0.9653 |
| `stream_f64_fixed_len32_einsum_alloc[faer]@t1` | 1.0038 | 0.0038 | 0.0323 | 4 | 0.9897, 1.0062, 1.0121, 1.0014 |
| `stream_f64_fixed_len32_einsum_alloc[faer]@t4` | 1.0162 | 0.0162 | 0.0380 | 4 | 1.0178, 1.0147, 1.0249, 0.9858 |
| `stream_f64_fresh_len32_einsum_alloc[faer]@t1` | 1.0071 | 0.0071 | 0.0199 | 4 | 1.0144, 0.9944, 0.9998, 1.0244 |
| `stream_f64_fresh_len32_einsum_alloc[faer]@t4` | 0.9983 | 0.0017 | 0.0342 | 4 | 0.9930, 1.0036, 0.9628, 1.0156 |
| `stream_f64_mixed_len32_einsum_alloc[faer]@t1` | 0.9957 | 0.0043 | 0.0909 | 4 | 1.0095, 0.9859, 0.9860, 1.0054 |
| `stream_f64_mixed_len32_einsum_alloc[faer]@t4` | 0.9933 | 0.0067 | 0.0205 | 4 | 0.9860, 1.0032, 1.0006, 0.9722 |
| `stream_f64_strides_len32_einsum_alloc[faer]@t1` | 0.9924 | 0.0076 | 0.0277 | 4 | 1.0008, 0.9836, 0.9878, 0.9970 |
| `stream_f64_strides_len32_einsum_alloc[faer]@t4` | 0.9941 | 0.0059 | 0.0610 | 4 | 0.9969, 0.9856, 0.9912, 1.0400 |
