# Small eager diagnostic (not acceptance reruns)

Investigate the previous wake-candidate nonregression failures without replacing their evidence or promoting the candidate.

Hypotheses: (1) persistent notification overhead; (2) process/scheduler placement; (3) GPU startup/clock state. Previous batch=1 medians form ~48/~67 us groups in BOTH baseline and candidate, with low within-process IQR. This suggests process-level effects but does not establish their cause.

Unchanged baseline/wake executables from ../wake-results, identical inputs and setup/synchronization boundaries. Two cases: dense_batched_matmul_f64_b1_256 and elementwise_chain_f64_1k, eager only. Six fresh-process pairs alternating baseline-first/candidate-first, both cases per process, existing 5 warmups/30 samples. Explicit provider 1T, unbound CPU affinity. Sequential; no competing GPU process; idle utilization <=5% and load1/available CPUs <=0.5 before collection. Record GPU clocks/pstate, host state and per-process CPU. Compare all medians/IQRs and within-variant variation; no adoption gate is redefined.

Then collect two independent CUDA+OSRT profiles for each variant, both cases in one process. Extract individual kernel durations, kernel-launch API durations, device-service OS waits and inter-kernel gaps. Profiles perturb timing, so use for attribution only. If GPU-kernel durations track the ~20 us batch=1 shift, investigate GPU-clock state; if not, investigate host timing/placement. Findings may remain inconclusive.
