# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20260913_094534`.

Each sample executes 1024 independent, distinct f64 matrix pairs. Three warmups and 15 samples; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The execution mode and worker count are recorded in each Rust row. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs.

EagerTensor and compiled trace are not labeled shared-session: their current public interfaces do not accept this borrowed session and create internal sessions during execution. The old single-call measurements remain diagnostic evidence only.

| Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | Check |
|---|---|---:|---:|---|---|---|---:|---:|---:|---|
| matmul | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.547250 | 0.206417 | 1510.99 | passed |
| matmul | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 0.979584 | 0.061625 | 956.62 | passed |
| matmul | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 0.335666 | 0.013270 | 327.80 | passed |
| matmul | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.064083 | 0.011729 | 1039.14 | passed |
| matmul | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 0.807500 | 0.013854 | 788.57 | passed |
| matmul | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 0.307625 | 0.005041 | 300.42 | passed |
| matmul | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.117791 | 0.059020 | 1091.59 | passed |
| matmul | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 0.865458 | 0.042500 | 845.17 | passed |
| matmul | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 0.320125 | 0.005771 | 312.62 | passed |
| matmul | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.084125 | 0.033084 | 1058.72 | passed |
| matmul | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 0.813959 | 0.015125 | 794.88 | passed |
| matmul | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 0.323458 | 0.006605 | 315.88 | passed |
| matmul | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.093375 | 0.012833 | 1067.75 | passed |
| matmul | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 0.829791 | 0.013812 | 810.34 | passed |
| matmul | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 0.392208 | 0.023291 | 383.02 | passed |
| matmul | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.122167 | 0.020854 | 1095.87 | passed |
| matmul | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 0.836042 | 0.031125 | 816.45 | passed |
| matmul | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 0.367500 | 0.010479 | 358.89 | passed |
| matmul | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.754000 | 0.028938 | 1712.89 | passed |
| matmul | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 1.089666 | 0.022916 | 1064.13 | passed |
| matmul | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 0.794208 | 0.011626 | 775.59 | passed |
| matmul | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.753125 | 0.028938 | 1712.04 | passed |
| matmul | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 1.095500 | 0.018958 | 1069.82 | passed |
| matmul | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 0.792292 | 0.030208 | 773.72 | passed |
| matmul | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.173083 | 0.029000 | 2122.15 | passed |
| matmul | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 2.368833 | 0.032979 | 2313.31 | passed |
| matmul | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 0.990583 | 0.009416 | 967.37 | passed |
| matmul | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 2.181625 | 0.048667 | 2130.49 | passed |
| matmul | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 2.324041 | 0.045937 | 2269.57 | passed |
| matmul | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 0.994750 | 0.029729 | 971.44 | passed |
| solve | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.975917 | 0.019478 | 1929.61 | passed |
| solve | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 2.025917 | 0.035229 | 1978.43 | passed |
| solve | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 4.141834 | 0.116833 | 4044.76 | passed |
| solve | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.969833 | 0.021396 | 1923.67 | passed |
| solve | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 2.033625 | 0.051666 | 1985.96 | passed |
| solve | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 4.110875 | 0.090647 | 4014.53 | passed |
| solve | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.039292 | 0.033521 | 1991.50 | passed |
| solve | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 2.125500 | 0.042042 | 2075.68 | passed |
| solve | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 4.188125 | 0.083396 | 4089.97 | passed |
| solve | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 2.073375 | 0.036729 | 2024.78 | passed |
| solve | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 2.123250 | 0.035917 | 2073.49 | passed |
| solve | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 4.225416 | 0.092791 | 4126.38 | passed |
| solve | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.399083 | 0.042041 | 2342.85 | passed |
| solve | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 2.542917 | 0.025229 | 2483.32 | passed |
| solve | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 4.284625 | 0.065146 | 4184.20 | passed |
| solve | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 2.417834 | 0.049730 | 2361.17 | passed |
| solve | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 2.575333 | 0.048376 | 2514.97 | passed |
| solve | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 4.350833 | 0.103897 | 4248.86 | passed |
| solve | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 3.750125 | 0.030895 | 3662.23 | passed |
| solve | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 3.974084 | 0.023292 | 3880.94 | passed |
| solve | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 5.102000 | 0.092062 | 4982.42 | passed |
| solve | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 3.739500 | 0.049625 | 3651.86 | passed |
| solve | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 4.037667 | 0.045271 | 3943.03 | passed |
| solve | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 5.098750 | 0.049521 | 4979.25 | passed |
| solve | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 12.094292 | 0.078083 | 11810.83 | passed |
| solve | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 11.058333 | 0.177188 | 10799.15 | passed |
| solve | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 10.680792 | 0.103000 | 10430.46 | passed |
| solve | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 12.047416 | 0.149938 | 11765.05 | passed |
| solve | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 11.038000 | 0.119188 | 10779.30 | passed |
| solve | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 10.808167 | 0.083062 | 10554.85 | passed |
