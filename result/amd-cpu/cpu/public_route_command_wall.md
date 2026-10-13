# CPU public-route command wall observations

Sequential repository CPU devcontainer commands. Observed CPU models: `Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz`. These are command costs, not operation speedups. Times include build/checkout, metadata, prepared input generation, correctness, warmup, collection, separate diagnostics and report generation; final report file writes are excluded. Build cache state differs between campaigns. Initial reference runs preceded budget configuration. An over-budget command is diagnostic and never a functional failure.

| Suite | Coverage | Effort | Features | Threads | Command s | Build s | Collection/metadata s | Diagnostic build s | Diagnostic collection s | Report s | Configured budget s | Budget status | Raw run |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| cpu/small_work | quick | scan | blas-mkl | 1,4 | 26.447 | 0.501 | 19.937 | 4.205 | 1.640 | 0.163 | unconfigured | unconfigured | `data/results/amd-cpu/cpu/small_work/20261013_101533` |
| cpu/small_work | quick | standard | blas-mkl | 1,4 | 41.598 | 0.492 | 39.059 | 0.226 | 1.648 | 0.173 | unconfigured | unconfigured | `data/results/amd-cpu/cpu/small_work/20261013_101925` |
| cpu/small_work | full | standard | blas-mkl | 1,4 | 181.019 | 0.484 | 175.111 | 0.244 | 4.932 | 0.248 | unconfigured | unconfigured | `data/results/amd-cpu/cpu/small_work/20261013_101616` |
| cpu/session_matrix | quick | scan | blas-mkl | 1,4 | 89.161 | 2.579 | 86.440 | 0.000 | 0.000 | 0.142 | unconfigured | unconfigured | `data/results/amd-cpu/cpu/session_matrix/20261013_102025` |
| cpu/session_matrix | full | standard | blas-mkl | 1,4 | 115.774 | 0.817 | 114.787 | 0.000 | 0.000 | 0.170 | unconfigured | unconfigured | `data/results/amd-cpu/cpu/session_matrix/20261013_102201` |
| cpu/session_matrix | quick | standard | native | 1,4 | 204.840 | 81.660 | 123.020 | 0.000 | 0.000 | 0.160 | 300.0 | within_budget | `data/results/amd-cpu/cpu/session_matrix/20261013_102639` |
| cpu/session_matrix | full | standard | native | 1,4 | 126.371 | 0.795 | 125.403 | 0.000 | 0.000 | 0.173 | 600.0 | within_budget | `data/results/amd-cpu/cpu/session_matrix/20261013_102936` |
| cpu/session_matrix | quick | standard | blas-mkl | 1,4 | 113.029 | 1.120 | 111.752 | 0.000 | 0.000 | 0.157 | 300.0 | within_budget | `data/results/amd-cpu/cpu/session_matrix/20261013_103151` |
