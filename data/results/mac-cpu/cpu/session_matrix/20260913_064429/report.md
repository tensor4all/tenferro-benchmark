# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20260913_064429`.

Each sample executes 1024 independent, distinct f64 matrix pairs. Three warmups and 15 samples; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope; 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs.

EagerTensor and compiled trace are not labeled shared-session: their current public interfaces do not accept this borrowed session and create internal sessions during execution. The old single-call measurements remain diagnostic evidence only.

| Operation | Matrix | Operations | Threads | Provider | Route | Median total ms | IQR total ms | Median ns/op | Check |
|---|---|---:|---:|---|---|---:|---:|---:|---|
| matmul | 2×2 | 1024 | 1 | accelerate | shared-session | 1.719625 | 0.128708 | 1679.32 | passed |
| matmul | 2×2 | 1024 | 1 | faer | shared-session | 0.933875 | 0.040209 | 911.99 | passed |
| matmul | 2×2 | 1024 | 1 | pytorch | python-loop | 0.301667 | 0.017417 | 294.60 | passed |
| matmul | 2×2 | 1024 | 4 | accelerate | shared-session | 8.635417 | 0.130709 | 8433.02 | passed |
| matmul | 2×2 | 1024 | 4 | faer | shared-session | 0.797625 | 0.039876 | 778.93 | passed |
| matmul | 2×2 | 1024 | 4 | pytorch | python-loop | 0.325917 | 0.024542 | 318.28 | passed |
| matmul | 4×4 | 1024 | 1 | accelerate | shared-session | 1.223833 | 0.033396 | 1195.15 | passed |
| matmul | 4×4 | 1024 | 1 | faer | shared-session | 0.788583 | 0.005730 | 770.10 | passed |
| matmul | 4×4 | 1024 | 1 | pytorch | python-loop | 0.333125 | 0.015749 | 325.32 | passed |
| matmul | 4×4 | 1024 | 4 | accelerate | shared-session | 8.710083 | 0.141230 | 8505.94 | passed |
| matmul | 4×4 | 1024 | 4 | faer | shared-session | 0.812709 | 0.014958 | 793.66 | passed |
| matmul | 4×4 | 1024 | 4 | pytorch | python-loop | 0.318417 | 0.001770 | 310.95 | passed |
| matmul | 8×8 | 1024 | 1 | accelerate | shared-session | 1.262250 | 0.022541 | 1232.67 | passed |
| matmul | 8×8 | 1024 | 1 | faer | shared-session | 0.851042 | 0.028563 | 831.10 | passed |
| matmul | 8×8 | 1024 | 1 | pytorch | python-loop | 0.364458 | 0.008042 | 355.92 | passed |
| matmul | 8×8 | 1024 | 4 | accelerate | shared-session | 8.774708 | 0.148270 | 8569.05 | passed |
| matmul | 8×8 | 1024 | 4 | faer | shared-session | 0.829000 | 0.009166 | 809.57 | passed |
| matmul | 8×8 | 1024 | 4 | pytorch | python-loop | 0.363209 | 0.004938 | 354.70 | passed |
| matmul | 16×16 | 1024 | 1 | accelerate | shared-session | 1.940209 | 0.022937 | 1894.74 | passed |
| matmul | 16×16 | 1024 | 1 | faer | shared-session | 1.117875 | 0.038479 | 1091.67 | passed |
| matmul | 16×16 | 1024 | 1 | pytorch | python-loop | 0.760250 | 0.018167 | 742.43 | passed |
| matmul | 16×16 | 1024 | 4 | accelerate | shared-session | 9.956541 | 0.040459 | 9723.18 | passed |
| matmul | 16×16 | 1024 | 4 | faer | shared-session | 1.080917 | 0.012021 | 1055.58 | passed |
| matmul | 16×16 | 1024 | 4 | pytorch | python-loop | 0.799875 | 0.022958 | 781.13 | passed |
| matmul | 32×32 | 1024 | 1 | accelerate | shared-session | 2.306583 | 0.033542 | 2252.52 | passed |
| matmul | 32×32 | 1024 | 1 | faer | shared-session | 2.386791 | 0.068625 | 2330.85 | passed |
| matmul | 32×32 | 1024 | 1 | pytorch | python-loop | 0.997417 | 0.042979 | 974.04 | passed |
| matmul | 32×32 | 1024 | 4 | accelerate | shared-session | 10.612959 | 0.081562 | 10364.22 | passed |
| matmul | 32×32 | 1024 | 4 | faer | shared-session | 2.342750 | 0.064334 | 2287.84 | passed |
| matmul | 32×32 | 1024 | 4 | pytorch | python-loop | 0.986917 | 0.022333 | 963.79 | passed |
| solve | 2×2 | 1024 | 1 | accelerate | shared-session | 2.007750 | 0.030936 | 1960.69 | passed |
| solve | 2×2 | 1024 | 1 | faer | shared-session | 2.022583 | 0.021583 | 1975.18 | passed |
| solve | 2×2 | 1024 | 1 | pytorch | python-loop | 4.059000 | 0.152417 | 3963.87 | passed |
| solve | 2×2 | 1024 | 4 | accelerate | shared-session | 8.805875 | 0.094021 | 8599.49 | passed |
| solve | 2×2 | 1024 | 4 | faer | shared-session | 2.027667 | 0.018583 | 1980.14 | passed |
| solve | 2×2 | 1024 | 4 | pytorch | python-loop | 4.126458 | 0.042375 | 4029.74 | passed |
| solve | 4×4 | 1024 | 1 | accelerate | shared-session | 2.092500 | 0.019271 | 2043.46 | passed |
| solve | 4×4 | 1024 | 1 | faer | shared-session | 2.125375 | 0.033751 | 2075.56 | passed |
| solve | 4×4 | 1024 | 1 | pytorch | python-loop | 4.193625 | 0.124729 | 4095.34 | passed |
| solve | 4×4 | 1024 | 4 | accelerate | shared-session | 8.968666 | 0.072354 | 8758.46 | passed |
| solve | 4×4 | 1024 | 4 | faer | shared-session | 2.142666 | 0.016500 | 2092.45 | passed |
| solve | 4×4 | 1024 | 4 | pytorch | python-loop | 4.134083 | 0.044250 | 4037.19 | passed |
| solve | 8×8 | 1024 | 1 | accelerate | shared-session | 2.450208 | 0.032813 | 2392.78 | passed |
| solve | 8×8 | 1024 | 1 | faer | shared-session | 2.509625 | 0.017458 | 2450.81 | passed |
| solve | 8×8 | 1024 | 1 | pytorch | python-loop | 4.404375 | 0.224416 | 4301.15 | passed |
| solve | 8×8 | 1024 | 4 | accelerate | shared-session | 9.532042 | 0.128271 | 9308.63 | passed |
| solve | 8×8 | 1024 | 4 | faer | shared-session | 2.516208 | 0.032355 | 2457.23 | passed |
| solve | 8×8 | 1024 | 4 | pytorch | python-loop | 4.297250 | 0.081438 | 4196.53 | passed |
| solve | 16×16 | 1024 | 1 | accelerate | shared-session | 3.965208 | 1.071083 | 3872.27 | passed |
| solve | 16×16 | 1024 | 1 | faer | shared-session | 4.048292 | 0.029812 | 3953.41 | passed |
| solve | 16×16 | 1024 | 1 | pytorch | python-loop | 5.029375 | 0.100876 | 4911.50 | passed |
| solve | 16×16 | 1024 | 4 | accelerate | shared-session | 11.114333 | 0.080834 | 10853.84 | passed |
| solve | 16×16 | 1024 | 4 | faer | shared-session | 4.010291 | 0.013270 | 3916.30 | passed |
| solve | 16×16 | 1024 | 4 | pytorch | python-loop | 5.126042 | 0.133500 | 5005.90 | passed |
| solve | 32×32 | 1024 | 1 | accelerate | shared-session | 12.200250 | 0.075563 | 11914.31 | passed |
| solve | 32×32 | 1024 | 1 | faer | shared-session | 10.822000 | 0.047937 | 10568.36 | passed |
| solve | 32×32 | 1024 | 1 | pytorch | python-loop | 10.666584 | 0.103583 | 10416.59 | passed |
| solve | 32×32 | 1024 | 4 | accelerate | shared-session | 18.579250 | 0.219292 | 18143.80 | passed |
| solve | 32×32 | 1024 | 4 | faer | shared-session | 10.820208 | 0.076438 | 10566.61 | passed |
| solve | 32×32 | 1024 | 4 | pytorch | python-loop | 10.658917 | 0.162166 | 10409.10 | passed |
