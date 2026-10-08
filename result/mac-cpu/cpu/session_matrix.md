# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20260923_120503`.

Each sample executes 1024 independent, distinct f64 matrix pairs. Three warmups and 15 samples; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The execution mode and worker count are recorded in each Rust row. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs.

EagerTensor and compiled trace are not labeled shared-session: their current public interfaces do not accept this borrowed session and create internal sessions during execution. The old single-call measurements remain diagnostic evidence only.

| Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | Check |
|---|---|---:|---:|---|---|---|---:|---:|---:|---|
| matmul | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.233042 | 0.221521 | 1204.14 | passed |
| matmul | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 0.953500 | 0.053979 | 931.15 | passed |
| matmul | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 0.309917 | 0.023770 | 302.65 | passed |
| matmul | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 0.756208 | 0.020063 | 738.48 | passed |
| matmul | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 0.738084 | 0.014771 | 720.79 | passed |
| matmul | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 0.318750 | 0.009500 | 311.28 | passed |
| matmul | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 0.760125 | 0.022250 | 742.31 | passed |
| matmul | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 0.752458 | 0.019396 | 734.82 | passed |
| matmul | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 0.330084 | 0.010167 | 322.35 | passed |
| matmul | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 0.746791 | 0.024812 | 729.29 | passed |
| matmul | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 0.732375 | 0.021562 | 715.21 | passed |
| matmul | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 0.320000 | 0.016625 | 312.50 | passed |
| matmul | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 0.816792 | 0.007792 | 797.65 | passed |
| matmul | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 0.790875 | 0.019666 | 772.34 | passed |
| matmul | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 0.377750 | 0.008480 | 368.90 | passed |
| matmul | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 0.793125 | 0.008833 | 774.54 | passed |
| matmul | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 0.787709 | 0.011145 | 769.25 | passed |
| matmul | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 0.374167 | 0.014625 | 365.40 | passed |
| matmul | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.321458 | 0.059166 | 1290.49 | passed |
| matmul | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 1.062042 | 0.020813 | 1037.15 | passed |
| matmul | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 0.831250 | 0.066417 | 811.77 | passed |
| matmul | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.316291 | 0.042958 | 1285.44 | passed |
| matmul | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 1.013167 | 0.023063 | 989.42 | passed |
| matmul | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 0.759458 | 0.017729 | 741.66 | passed |
| matmul | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.739166 | 0.035500 | 1698.40 | passed |
| matmul | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 2.267167 | 0.131125 | 2214.03 | passed |
| matmul | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 1.001208 | 0.022875 | 977.74 | passed |
| matmul | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.584292 | 0.051666 | 1547.16 | passed |
| matmul | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 2.266792 | 0.018812 | 2213.66 | passed |
| matmul | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 0.991875 | 0.010521 | 968.63 | passed |
| solve | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.576125 | 0.028854 | 1539.18 | passed |
| solve | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 1.632000 | 0.022208 | 1593.75 | passed |
| solve | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 4.141625 | 0.055563 | 4044.56 | passed |
| solve | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.563667 | 0.037749 | 1527.02 | passed |
| solve | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 1.605708 | 0.030625 | 1568.07 | passed |
| solve | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 4.100333 | 0.087980 | 4004.23 | passed |
| solve | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.645750 | 0.033584 | 1607.18 | passed |
| solve | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 1.717959 | 0.021895 | 1677.69 | passed |
| solve | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 4.194917 | 0.062187 | 4096.60 | passed |
| solve | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.624625 | 0.024103 | 1586.55 | passed |
| solve | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 1.715458 | 0.022729 | 1675.25 | passed |
| solve | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 4.125416 | 0.099271 | 4028.73 | passed |
| solve | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.877875 | 0.020042 | 1833.86 | passed |
| solve | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 2.133750 | 0.029770 | 2083.74 | passed |
| solve | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 4.199542 | 0.110125 | 4101.12 | passed |
| solve | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.870833 | 0.030104 | 1826.99 | passed |
| solve | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 2.102375 | 0.033209 | 2053.10 | passed |
| solve | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 4.291292 | 0.097978 | 4190.71 | passed |
| solve | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.859875 | 0.035542 | 2792.85 | passed |
| solve | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 3.569667 | 0.030312 | 3486.00 | passed |
| solve | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 4.978208 | 0.086353 | 4861.53 | passed |
| solve | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 2.849792 | 0.064917 | 2783.00 | passed |
| solve | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 3.553875 | 0.050084 | 3470.58 | passed |
| solve | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 4.993708 | 0.059249 | 4876.67 | passed |
| solve | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 9.196458 | 0.087313 | 8980.92 | passed |
| solve | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 10.556291 | 0.041875 | 10308.88 | passed |
| solve | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 10.784667 | 0.109812 | 10531.90 | passed |
| solve | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 9.192625 | 0.150375 | 8977.17 | passed |
| solve | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 10.550084 | 0.060187 | 10302.82 | passed |
| solve | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 10.695709 | 0.167666 | 10445.03 | passed |
