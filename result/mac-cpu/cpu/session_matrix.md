# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20260916_080229`.

Each sample executes 1024 independent, distinct f64 matrix pairs. Three warmups and 15 samples; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The execution mode and worker count are recorded in each Rust row. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs.

EagerTensor and compiled trace are not labeled shared-session: their current public interfaces do not accept this borrowed session and create internal sessions during execution. The old single-call measurements remain diagnostic evidence only.

| Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | Check |
|---|---|---:|---:|---|---|---|---:|---:|---:|---|
| matmul | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.676292 | 0.229750 | 1637.00 | passed |
| matmul | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 0.982083 | 0.041146 | 959.07 | passed |
| matmul | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 0.335708 | 0.008000 | 327.84 | passed |
| matmul | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.090000 | 0.022229 | 1064.45 | passed |
| matmul | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 0.782834 | 0.026583 | 764.49 | passed |
| matmul | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 0.331292 | 0.014313 | 323.53 | passed |
| matmul | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.096000 | 0.014021 | 1070.31 | passed |
| matmul | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 0.817875 | 0.042792 | 798.71 | passed |
| matmul | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 0.363791 | 0.074063 | 355.26 | passed |
| matmul | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.083375 | 0.025687 | 1057.98 | passed |
| matmul | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 0.795041 | 0.018541 | 776.41 | passed |
| matmul | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 0.331125 | 0.012771 | 323.36 | passed |
| matmul | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.142166 | 0.039729 | 1115.40 | passed |
| matmul | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 0.827666 | 0.021438 | 808.27 | passed |
| matmul | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 0.404125 | 0.014230 | 394.65 | passed |
| matmul | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.141583 | 0.016353 | 1114.83 | passed |
| matmul | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 0.829125 | 0.014812 | 809.69 | passed |
| matmul | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 0.400250 | 0.017583 | 390.87 | passed |
| matmul | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.921541 | 0.076688 | 1876.50 | passed |
| matmul | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 1.079625 | 0.035125 | 1054.32 | passed |
| matmul | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 0.780791 | 0.020751 | 762.49 | passed |
| matmul | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.797416 | 0.046646 | 1755.29 | passed |
| matmul | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 1.105542 | 0.041980 | 1079.63 | passed |
| matmul | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 0.795875 | 0.028792 | 777.22 | passed |
| matmul | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.369333 | 0.053354 | 2313.80 | passed |
| matmul | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 2.341417 | 0.134625 | 2286.54 | passed |
| matmul | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 1.094042 | 0.038416 | 1068.40 | passed |
| matmul | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 2.314375 | 0.041292 | 2260.13 | passed |
| matmul | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 2.461750 | 0.041772 | 2404.05 | passed |
| matmul | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 1.064625 | 0.028312 | 1039.67 | passed |
| solve | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.996792 | 0.038271 | 1949.99 | passed |
| solve | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 2.038625 | 0.049125 | 1990.84 | passed |
| solve | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 4.254250 | 0.088708 | 4154.54 | passed |
| solve | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.982750 | 0.019083 | 1936.28 | passed |
| solve | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 2.047625 | 0.020730 | 1999.63 | passed |
| solve | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 4.278792 | 0.125438 | 4178.51 | passed |
| solve | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.059667 | 0.041938 | 2011.39 | passed |
| solve | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 2.143583 | 0.032459 | 2093.34 | passed |
| solve | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 4.506875 | 0.211895 | 4401.25 | passed |
| solve | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 2.060625 | 0.046021 | 2012.33 | passed |
| solve | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 2.126625 | 0.037750 | 2076.78 | passed |
| solve | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 4.290209 | 0.127167 | 4189.66 | passed |
| solve | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.441791 | 0.032313 | 2384.56 | passed |
| solve | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 2.540042 | 0.043250 | 2480.51 | passed |
| solve | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 4.592250 | 0.138313 | 4484.62 | passed |
| solve | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 2.448292 | 0.054208 | 2390.91 | passed |
| solve | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 2.592584 | 0.045855 | 2531.82 | passed |
| solve | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 4.518167 | 0.122333 | 4412.27 | passed |
| solve | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 3.808916 | 0.063937 | 3719.64 | passed |
| solve | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 4.041458 | 0.035771 | 3946.74 | passed |
| solve | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 5.387042 | 0.095395 | 5260.78 | passed |
| solve | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 3.788542 | 0.050875 | 3699.75 | passed |
| solve | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 4.045542 | 0.021917 | 3950.72 | passed |
| solve | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 5.270042 | 0.097292 | 5146.53 | passed |
| solve | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 12.308459 | 0.551646 | 12019.98 | passed |
| solve | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 11.168500 | 0.073104 | 10906.74 | passed |
| solve | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 10.909667 | 0.050563 | 10653.97 | passed |
| solve | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 12.238458 | 0.065603 | 11951.62 | passed |
| solve | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 11.164500 | 0.090896 | 10902.83 | passed |
| solve | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 10.969750 | 0.129729 | 10712.65 | passed |
