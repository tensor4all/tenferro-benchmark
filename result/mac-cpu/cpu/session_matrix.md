# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20260923_021923`.

Each sample executes 1024 independent, distinct f64 matrix pairs. Three warmups and 15 samples; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The execution mode and worker count are recorded in each Rust row. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs.

EagerTensor and compiled trace are not labeled shared-session: their current public interfaces do not accept this borrowed session and create internal sessions during execution. The old single-call measurements remain diagnostic evidence only.

| Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | Check |
|---|---|---:|---:|---|---|---|---:|---:|---:|---|
| matmul | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.238667 | 0.152417 | 1209.64 | passed |
| matmul | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 0.946667 | 0.038895 | 924.48 | passed |
| matmul | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 0.300750 | 0.012688 | 293.70 | passed |
| matmul | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 0.778542 | 0.011416 | 760.29 | passed |
| matmul | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 0.755625 | 0.006208 | 737.92 | passed |
| matmul | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 0.293583 | 0.008083 | 286.70 | passed |
| matmul | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 0.760959 | 0.009437 | 743.12 | passed |
| matmul | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 0.755625 | 0.012833 | 737.92 | passed |
| matmul | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 0.325917 | 0.006188 | 318.28 | passed |
| matmul | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 0.762709 | 0.015104 | 744.83 | passed |
| matmul | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 0.739125 | 0.015521 | 721.80 | passed |
| matmul | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 0.310667 | 0.014667 | 303.39 | passed |
| matmul | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 0.824791 | 0.009041 | 805.46 | passed |
| matmul | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 0.822500 | 0.027751 | 803.22 | passed |
| matmul | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 0.371625 | 0.012708 | 362.92 | passed |
| matmul | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 0.835333 | 0.022541 | 815.75 | passed |
| matmul | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 0.766375 | 0.011188 | 748.41 | passed |
| matmul | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 0.367959 | 0.008042 | 359.33 | passed |
| matmul | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.385750 | 0.040167 | 1353.27 | passed |
| matmul | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 1.067917 | 0.021854 | 1042.89 | passed |
| matmul | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 0.784125 | 0.015459 | 765.75 | passed |
| matmul | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.363500 | 0.031312 | 1331.54 | passed |
| matmul | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 1.062208 | 0.012021 | 1037.31 | passed |
| matmul | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 0.752417 | 0.022708 | 734.78 | passed |
| matmul | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.573959 | 0.045687 | 1537.07 | passed |
| matmul | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 2.297750 | 0.050604 | 2243.90 | passed |
| matmul | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 1.003000 | 0.014021 | 979.49 | passed |
| matmul | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.641500 | 0.028313 | 1603.03 | passed |
| matmul | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 2.313958 | 0.043604 | 2259.72 | passed |
| matmul | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 1.002458 | 0.033397 | 978.96 | passed |
| solve | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.508416 | 0.041021 | 1473.06 | passed |
| solve | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 1.562125 | 0.024250 | 1525.51 | passed |
| solve | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 4.047167 | 0.059812 | 3952.31 | passed |
| solve | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.482917 | 0.031209 | 1448.16 | passed |
| solve | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 1.562541 | 0.029396 | 1525.92 | passed |
| solve | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 4.571000 | 0.247854 | 4463.87 | passed |
| solve | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.578500 | 0.023062 | 1541.50 | passed |
| solve | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 1.671500 | 0.024417 | 1632.32 | passed |
| solve | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 4.168708 | 0.077521 | 4071.00 | passed |
| solve | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.517000 | 0.023146 | 1481.45 | passed |
| solve | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 1.636208 | 0.028813 | 1597.86 | passed |
| solve | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 4.198916 | 0.096333 | 4100.50 | passed |
| solve | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.840375 | 0.041126 | 1797.24 | passed |
| solve | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 2.090334 | 0.013855 | 2041.34 | passed |
| solve | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 4.296583 | 0.052625 | 4195.88 | passed |
| solve | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 1.831667 | 0.039563 | 1788.74 | passed |
| solve | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 2.072292 | 0.021542 | 2023.72 | passed |
| solve | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 4.264042 | 0.137687 | 4164.10 | passed |
| solve | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.826000 | 0.066209 | 2759.77 | passed |
| solve | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 3.525125 | 0.073729 | 3442.50 | passed |
| solve | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 5.035958 | 0.116146 | 4917.93 | passed |
| solve | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 2.808584 | 0.060708 | 2742.76 | passed |
| solve | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 3.518250 | 0.027751 | 3435.79 | passed |
| solve | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 5.008166 | 0.158126 | 4890.79 | passed |
| solve | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 9.194166 | 0.153250 | 8978.68 | passed |
| solve | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 10.485541 | 0.051667 | 10239.79 | passed |
| solve | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 10.797666 | 0.648021 | 10544.60 | passed |
| solve | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 9.071000 | 0.108875 | 8858.40 | passed |
| solve | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 10.476042 | 0.077021 | 10230.51 | passed |
| solve | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 10.710833 | 0.111126 | 10459.80 | passed |
