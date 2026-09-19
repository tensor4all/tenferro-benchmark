# Read-path repair paired diagnostic protocol

Declared before collecting candidate measurements.

- Source baseline: tenferro origin/main 40e24634f9aa814db82efd9faae5e99a8fcee8c4. Candidate: uncommitted fix/read-path-materialization worktree; archive its exact final diff alongside logs before collection. Cargo dependencies unchanged. Benchmark source: 9a5b104, benchmark_gpu_rust.
- Environment: existing bench-gpu-20260919 CUDA container, A100 80GB PCIe; release build, cargo -j16. Explicit CPU/provider 1T via scripts/thread_env.sh, no affinity binding. Record thread environment. CubeCL internal service threads remain present.
- Case list: elementwise_chain_f64_1k and elementwise_chain_f64_1m, eager and trace; dense_batched_matmul_f64_b1024_256, eager and trace. Use earlier focused YAMLs (5 warmups, 30 samples). Retain outputs and completion barrier as in the existing benchmark; setup excluded. Small-chain samples remain public-API single-call latency diagnostics, not shared-session throughput.
- Repetition: three fresh-process baseline/candidate pairs per backend; baseline first in rounds1/3, candidate first in round2. All cases in each process. Sequential runs, no overlapping builds/tests/benchmarks.
- Primary evidence: 1m eager expression must no longer contain the previous 40 materialization kernels; collect separate original/candidate Nsight profiles. Correctness must pass for every measured case. Profiled time is not the performance metric.
- Wall-time metric: per-round median, retain IQR. Claim eager1m speedup only if all three candidate/baseline ratios <=0.90. Nonregression diagnostic: trace cases and BMM median ratios <=1.10 in every round; any failure remains visible and requires investigation rather than selective retry.
- Noise gate: no other CUDA compute process and idle GPU utilization <=5% before the paired collection; record nvidia-smi. No concurrent owned builds/tests/benchmark jobs. Record host load and processes; if gate fails, classify entire paired comparison inconclusive. Never selectively omit/retry cases.
- CPU common-linalg dispatch change is validated with existing Faer read hooks, focused regression and integration tests. No CPU speedup claim without a separate measured protocol.
- These are diagnostic results, not replacements for published benchmark reports. Production sleep policy is unchanged.
