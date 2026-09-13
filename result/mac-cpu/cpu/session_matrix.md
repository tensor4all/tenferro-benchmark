# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20260913_065008`.

Each sample executes 1024 independent, distinct f64 matrix pairs. Three warmups and 15 samples; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. The BLAS ProviderDefaultExclusive path does not enter the executor around the session body: individual operations still enter it, and that internal cost remains timed. Faer reuses the entered executor context. The execution mode and worker count are recorded in each Rust row. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs.

EagerTensor and compiled trace are not labeled shared-session: their current public interfaces do not accept this borrowed session and create internal sessions during execution. The old single-call measurements remain diagnostic evidence only.

| Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | Check |
|---|---|---:|---:|---|---|---|---:|---:|---:|---|
| matmul | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.328458 | 0.096083 | 1297.32 | passed |
| matmul | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 0.831834 | 0.030958 | 812.34 | passed |
| matmul | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 0.331750 | 0.021583 | 323.97 | passed |
| matmul | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 8.689458 | 0.209188 | 8485.80 | passed |
| matmul | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 0.802375 | 0.010667 | 783.57 | passed |
| matmul | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 0.296667 | 0.010438 | 289.71 | passed |
| matmul | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.237792 | 0.038501 | 1208.78 | passed |
| matmul | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 0.819666 | 0.028979 | 800.46 | passed |
| matmul | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 0.345042 | 0.008521 | 336.96 | passed |
| matmul | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 8.626542 | 0.153271 | 8424.36 | passed |
| matmul | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 0.795459 | 0.012792 | 776.82 | passed |
| matmul | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 0.323750 | 0.010105 | 316.16 | passed |
| matmul | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 1.264791 | 0.027270 | 1235.15 | passed |
| matmul | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 0.834750 | 0.025770 | 815.19 | passed |
| matmul | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 0.401833 | 0.016396 | 392.42 | passed |
| matmul | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 8.622041 | 0.160521 | 8419.96 | passed |
| matmul | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 0.826792 | 0.009562 | 807.41 | passed |
| matmul | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 0.353750 | 0.018270 | 345.46 | passed |
| matmul | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.030125 | 0.055730 | 1982.54 | passed |
| matmul | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 1.115334 | 0.068250 | 1089.19 | passed |
| matmul | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 0.820208 | 0.024251 | 800.98 | passed |
| matmul | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 10.016708 | 0.374125 | 9781.94 | passed |
| matmul | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 1.129375 | 0.022563 | 1102.91 | passed |
| matmul | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 0.790083 | 0.014354 | 771.57 | passed |
| matmul | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.334958 | 0.088979 | 2280.23 | passed |
| matmul | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 2.391125 | 0.078104 | 2335.08 | passed |
| matmul | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 1.135750 | 0.095250 | 1109.13 | passed |
| matmul | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 10.481708 | 0.065000 | 10236.04 | passed |
| matmul | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 2.291917 | 0.093749 | 2238.20 | passed |
| matmul | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 0.980792 | 0.024896 | 957.80 | passed |
| solve | 2×2 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.103500 | 0.052041 | 2054.20 | passed |
| solve | 2×2 | 1024 | 1 | faer | shared-session | Compatibility | 2.080750 | 0.057834 | 2031.98 | passed |
| solve | 2×2 | 1024 | 1 | pytorch | python-loop | not applicable | 4.229917 | 0.167979 | 4130.78 | passed |
| solve | 2×2 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 8.843292 | 0.058250 | 8636.03 | passed |
| solve | 2×2 | 1024 | 4 | faer | shared-session | Compatibility | 2.049417 | 0.071791 | 2001.38 | passed |
| solve | 2×2 | 1024 | 4 | pytorch | python-loop | not applicable | 4.131167 | 0.219959 | 4034.34 | passed |
| solve | 4×4 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.310125 | 0.223395 | 2255.98 | passed |
| solve | 4×4 | 1024 | 1 | faer | shared-session | Compatibility | 2.259125 | 0.250354 | 2206.18 | passed |
| solve | 4×4 | 1024 | 1 | pytorch | python-loop | not applicable | 4.283042 | 0.190083 | 4182.66 | passed |
| solve | 4×4 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 8.986416 | 0.081562 | 8775.80 | passed |
| solve | 4×4 | 1024 | 4 | faer | shared-session | Compatibility | 2.157833 | 0.074458 | 2107.26 | passed |
| solve | 4×4 | 1024 | 4 | pytorch | python-loop | not applicable | 4.144625 | 0.086625 | 4047.49 | passed |
| solve | 8×8 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 2.448375 | 0.052687 | 2390.99 | passed |
| solve | 8×8 | 1024 | 1 | faer | shared-session | Compatibility | 2.569458 | 0.043292 | 2509.24 | passed |
| solve | 8×8 | 1024 | 1 | pytorch | python-loop | not applicable | 4.111333 | 0.176792 | 4014.97 | passed |
| solve | 8×8 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 9.460292 | 0.066146 | 9238.57 | passed |
| solve | 8×8 | 1024 | 4 | faer | shared-session | Compatibility | 2.536291 | 0.027167 | 2476.85 | passed |
| solve | 8×8 | 1024 | 4 | pytorch | python-loop | not applicable | 4.186125 | 0.185335 | 4088.01 | passed |
| solve | 16×16 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 3.725917 | 0.133188 | 3638.59 | passed |
| solve | 16×16 | 1024 | 1 | faer | shared-session | Compatibility | 3.986375 | 0.078833 | 3892.94 | passed |
| solve | 16×16 | 1024 | 1 | pytorch | python-loop | not applicable | 4.888875 | 0.088458 | 4774.29 | passed |
| solve | 16×16 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 11.211417 | 0.033563 | 10948.65 | passed |
| solve | 16×16 | 1024 | 4 | faer | shared-session | Compatibility | 3.962500 | 0.040896 | 3869.63 | passed |
| solve | 16×16 | 1024 | 4 | pytorch | python-loop | not applicable | 4.934500 | 0.093688 | 4818.85 | passed |
| solve | 32×32 | 1024 | 1 | accelerate | shared-session | ProviderDefaultExclusive | 11.771125 | 0.076209 | 11495.24 | passed |
| solve | 32×32 | 1024 | 1 | faer | shared-session | Compatibility | 10.722417 | 0.226521 | 10471.11 | passed |
| solve | 32×32 | 1024 | 1 | pytorch | python-loop | not applicable | 10.644625 | 0.163167 | 10395.14 | passed |
| solve | 32×32 | 1024 | 4 | accelerate | shared-session | ProviderDefaultExclusive | 18.807084 | 0.273833 | 18366.29 | passed |
| solve | 32×32 | 1024 | 4 | faer | shared-session | Compatibility | 10.662875 | 0.048291 | 10412.96 | passed |
| solve | 32×32 | 1024 | 4 | pytorch | python-loop | not applicable | 10.541208 | 0.212707 | 10294.15 | passed |
