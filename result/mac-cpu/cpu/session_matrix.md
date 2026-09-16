# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20260916_014946`.

Each sample executes 1024 independent, distinct f64 matrix pairs. Three warmups and 15 samples; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The execution mode and worker count are recorded in each Rust row. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs.

EagerTensor and compiled trace are not labeled shared-session: their current public interfaces do not accept this borrowed session and create internal sessions during execution. The old single-call measurements remain diagnostic evidence only.

| Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | Check |
|---|---|---:|---:|---|---|---|---:|---:|---:|---|
| matmul | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.518542 | 0.187187 | 1482.95 | passed |
| matmul | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 0.950125 | 0.054625 | 927.86 | passed |
| matmul | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 0.335250 | 0.009562 | 327.39 | passed |
| matmul | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.026917 | 0.018270 | 1002.85 | passed |
| matmul | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 0.796000 | 0.012479 | 777.34 | passed |
| matmul | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 0.327250 | 0.014457 | 319.58 | passed |
| matmul | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.076833 | 0.016042 | 1051.59 | passed |
| matmul | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 0.814333 | 0.024375 | 795.25 | passed |
| matmul | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 0.337833 | 0.011666 | 329.92 | passed |
| matmul | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.033750 | 0.023208 | 1009.52 | passed |
| matmul | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 0.814417 | 0.033896 | 795.33 | passed |
| matmul | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 0.347833 | 0.015459 | 339.68 | passed |
| matmul | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.117333 | 0.048688 | 1091.15 | passed |
| matmul | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 0.824167 | 0.022874 | 804.85 | passed |
| matmul | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 0.433917 | 0.049187 | 423.75 | passed |
| matmul | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.093666 | 0.044895 | 1068.03 | passed |
| matmul | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 0.844541 | 0.028146 | 824.75 | passed |
| matmul | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 0.397417 | 0.018271 | 388.10 | passed |
| matmul | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.811042 | 0.133771 | 1768.60 | passed |
| matmul | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 1.121667 | 0.059354 | 1095.38 | passed |
| matmul | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 0.807750 | 0.042084 | 788.82 | passed |
| matmul | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.774875 | 0.029021 | 1733.28 | passed |
| matmul | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 1.103417 | 0.036062 | 1077.56 | passed |
| matmul | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 0.790375 | 0.020001 | 771.85 | passed |
| matmul | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.204708 | 0.038833 | 2153.04 | passed |
| matmul | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 2.428917 | 0.037771 | 2371.99 | passed |
| matmul | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 1.013625 | 0.021042 | 989.87 | passed |
| matmul | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 2.259000 | 0.031417 | 2206.05 | passed |
| matmul | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 2.473667 | 0.052146 | 2415.69 | passed |
| matmul | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 1.058083 | 0.022791 | 1033.28 | passed |
| solve | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.958958 | 0.025854 | 1913.04 | passed |
| solve | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 2.033500 | 0.014521 | 1985.84 | passed |
| solve | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 4.223708 | 0.045124 | 4124.71 | passed |
| solve | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.970583 | 0.019333 | 1924.40 | passed |
| solve | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 2.021917 | 0.038458 | 1974.53 | passed |
| solve | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 4.233084 | 0.058125 | 4133.87 | passed |
| solve | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.061541 | 0.039374 | 2013.22 | passed |
| solve | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 2.130125 | 0.029646 | 2080.20 | passed |
| solve | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 4.430625 | 0.059416 | 4326.78 | passed |
| solve | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 2.048834 | 0.025604 | 2000.81 | passed |
| solve | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 2.117833 | 0.031897 | 2068.20 | passed |
| solve | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 4.279917 | 0.088229 | 4179.61 | passed |
| solve | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.425292 | 0.017229 | 2368.45 | passed |
| solve | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 2.575583 | 0.038667 | 2515.22 | passed |
| solve | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 4.492334 | 0.082438 | 4387.04 | passed |
| solve | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 2.428541 | 0.025062 | 2371.62 | passed |
| solve | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 2.557708 | 0.021771 | 2497.76 | passed |
| solve | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 4.518542 | 0.202771 | 4412.64 | passed |
| solve | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 3.837125 | 0.059834 | 3747.19 | passed |
| solve | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 4.071708 | 0.034979 | 3976.28 | passed |
| solve | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 5.319000 | 0.177730 | 5194.34 | passed |
| solve | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 3.783750 | 0.058833 | 3695.07 | passed |
| solve | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 4.106542 | 0.072083 | 4010.29 | passed |
| solve | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 5.259292 | 0.149645 | 5136.03 | passed |
| solve | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 12.331959 | 0.119562 | 12042.93 | passed |
| solve | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 11.272459 | 0.079228 | 11008.26 | passed |
| solve | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 11.003875 | 0.232270 | 10745.97 | passed |
| solve | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 12.214917 | 0.090708 | 11928.63 | passed |
| solve | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 11.203917 | 0.076021 | 10941.33 | passed |
| solve | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 10.843458 | 0.101396 | 10589.31 | passed |
