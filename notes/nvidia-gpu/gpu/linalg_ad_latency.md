# Small GPU linalg AD latency notes

These n=2/4/8 rows are single-call latency diagnostics, not shared-session or
GPU-throughput measurements. CPU placement and service-thread scheduling can
matter. See the [refresh protocol](evidence/integration-refresh-protocol.md).

## Verification limitation and correction

The initial post-merge `20260919_142600` run, like its historical predecessor,
marked AD execution as `verification: passed` without numerical comparison.
This affects both AD suites (100 rows). Those raw records remain archived but
must not be cited as numerical evidence. Corrected runners check one deterministic
directional derivative against the backend's primal central difference, outside
timing, with fixed h=1e-5 and unchanged YAML tolerances. They do not compare every
gradient component or enforce cross-backend decomposition gauges.

The corrected `20260919_151433` runs passed all 100 directional checks (60 small,
40 GPU-sized), without changing h or the YAML tolerances. The fixed runner also
rejects deliberately wrong and non-finite derivatives in focused tests.

<a id="linalg_ad_small_grad_sum_svd_s_jvp_f64_2"></a>
## SVD-values JVP, n=2 — corrected run

The numerically verified final tenferro trace median is 1.011777 ms, 45.2% above
the historical median. This is retained as a nonblocking small-case regression;
CPU placement was unrestricted, and the change is not causally attributed to a
single implementation detail. No latency rerun was selected to make it disappear.

The three small-case increases below were observed before that verification
correction and are retained, not erased by the subsequent AD remeasurement.
The user accepted small-case latency regressions as nonblocking. These are
historical/current ratios, not randomized causal attribution to one change.

<a id="linalg_ad_small_grad_sum_qr_jvp_f64_4"></a>
## QR JVP, n=4

The first post-merge tenferro trace median was 12.9% above `20260919_084000`.

<a id="linalg_ad_small_grad_sum_qr_jvp_f64_8"></a>
## QR JVP, n=8

The first post-merge tenferro trace median was 31.6% above `20260919_084000`
(1.22433 ms to 1.61144 ms).

<a id="linalg_ad_small_grad_sum_solve_jvp_f64_8"></a>
## Solve JVP, n=8

The first post-merge tenferro trace median was 10.6% above `20260919_084000`.
