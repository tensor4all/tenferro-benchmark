# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20260923_091801`.

Each sample executes 1024 independent, distinct f64 matrix pairs. Three warmups and 15 samples; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The execution mode and worker count are recorded in each Rust row. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs.

EagerTensor and compiled trace are not labeled shared-session: their current public interfaces do not accept this borrowed session and create internal sessions during execution. The old single-call measurements remain diagnostic evidence only.

| Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | Check |
|---|---|---:|---:|---|---|---|---:|---:|---:|---|
| matmul | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.251583 | 0.188459 | 1222.25 | passed |
| matmul | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 0.938334 | 0.051791 | 916.34 | passed |
| matmul | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 0.321584 | 0.008354 | 314.05 | passed |
| matmul | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 0.763792 | 0.021354 | 745.89 | passed |
| matmul | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 0.759083 | 0.026770 | 741.29 | passed |
| matmul | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 0.313459 | 0.007916 | 306.11 | passed |
| matmul | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 0.783375 | 0.029791 | 765.01 | passed |
| matmul | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 0.764541 | 0.022542 | 746.62 | passed |
| matmul | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 0.336125 | 0.019958 | 328.25 | passed |
| matmul | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 0.775792 | 0.015688 | 757.61 | passed |
| matmul | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 0.762875 | 0.013688 | 745.00 | passed |
| matmul | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 0.325292 | 0.020500 | 317.67 | passed |
| matmul | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 0.831542 | 0.029729 | 812.05 | passed |
| matmul | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 0.787791 | 0.019855 | 769.33 | passed |
| matmul | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 0.390584 | 0.017854 | 381.43 | passed |
| matmul | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 0.832667 | 0.045291 | 813.15 | passed |
| matmul | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 0.796167 | 0.030813 | 777.51 | passed |
| matmul | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 0.366708 | 0.010771 | 358.11 | passed |
| matmul | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.334458 | 0.065708 | 1303.18 | passed |
| matmul | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 1.061916 | 0.033147 | 1037.03 | passed |
| matmul | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 0.775542 | 0.023041 | 757.37 | passed |
| matmul | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.332250 | 0.038833 | 1301.03 | passed |
| matmul | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 1.049792 | 0.029292 | 1025.19 | passed |
| matmul | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 0.792791 | 0.023417 | 774.21 | passed |
| matmul | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.628667 | 0.097979 | 1590.50 | passed |
| matmul | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 2.363000 | 0.098687 | 2307.62 | passed |
| matmul | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 0.981167 | 0.013709 | 958.17 | passed |
| matmul | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.601042 | 0.055479 | 1563.52 | passed |
| matmul | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 2.417417 | 0.046417 | 2360.76 | passed |
| matmul | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 1.013958 | 0.009605 | 990.19 | passed |
| solve | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.558334 | 0.013520 | 1521.81 | passed |
| solve | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 1.616459 | 0.060625 | 1578.57 | passed |
| solve | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 4.169750 | 0.174792 | 4072.02 | passed |
| solve | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.555625 | 0.023105 | 1519.17 | passed |
| solve | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 1.590583 | 0.030062 | 1553.30 | passed |
| solve | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 4.107792 | 0.120438 | 4011.52 | passed |
| solve | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.635166 | 0.032146 | 1596.84 | passed |
| solve | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 1.727375 | 0.039499 | 1686.89 | passed |
| solve | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 4.329000 | 0.129709 | 4227.54 | passed |
| solve | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.634375 | 0.014730 | 1596.07 | passed |
| solve | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 1.725125 | 0.028979 | 1684.69 | passed |
| solve | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 4.245292 | 0.076750 | 4145.79 | passed |
| solve | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.901167 | 0.022291 | 1856.61 | passed |
| solve | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 2.136542 | 0.038624 | 2086.47 | passed |
| solve | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 4.391417 | 0.106646 | 4288.49 | passed |
| solve | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.868000 | 0.029562 | 1824.22 | passed |
| solve | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 2.097375 | 0.031105 | 2048.22 | passed |
| solve | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 4.262209 | 0.082667 | 4162.31 | passed |
| solve | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.922334 | 0.055479 | 2853.84 | passed |
| solve | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 3.630292 | 0.035312 | 3545.21 | passed |
| solve | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 5.077083 | 0.104291 | 4958.09 | passed |
| solve | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 2.835958 | 0.069334 | 2769.49 | passed |
| solve | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 3.562459 | 0.041438 | 3478.96 | passed |
| solve | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 5.012959 | 0.067354 | 4895.47 | passed |
| solve | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 8.952875 | 0.134959 | 8743.04 | passed |
| solve | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 10.632708 | 0.135459 | 10383.50 | passed |
| solve | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 10.921750 | 0.280938 | 10665.77 | passed |
| solve | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 9.109750 | 0.061312 | 8896.24 | passed |
| solve | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 10.445875 | 0.048834 | 10201.05 | passed |
| solve | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 10.909084 | 0.233959 | 10653.40 | passed |
