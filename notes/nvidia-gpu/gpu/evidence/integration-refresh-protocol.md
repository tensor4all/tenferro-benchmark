# Post-merge notification/materialization benchmark refresh

Declared 2026-09-19 before the integration measurements.

## Preconditions and scope

1. Merge CubeCL PR #19 after correctness/safety validation and applicable CI.
2. Align Cubek's existing 0.2 release line on the same merged CubeCL commit
   (needed for shared runtime type identity), then update tenferro-rs to both
   merged commits, integrate the existing materialization repair, validate and
   merge that PR.
3. Resolve the merged tenferro-rs commit and record it, the CubeCL pin, benchmark
   revision and dirty-diff status in the measurement metadata before running.
   Do not measure an experimental local dependency override as the final result.

Refresh all eight existing NVIDIA GPU suites: `dense`, `einsum`, `elementwise`,
`linalg_ad_latency`, `linalg_jvp_vjp`, `permutation`, `sparse`, `tensornetwork`.
Use their existing checked-in YAML definitions, backend selections, warmups,
samples, verification tolerances and standard runners. Record any necessary
change rather than silently comparing different contracts.

## Measurement contract

- Use the existing A100 CUDA benchmark container and record current software,
  device, CPU and dependency metadata; do not assume they remained unchanged.
- Run suites/backends sequentially. Configure the provider/backend to explicit
  1T with `scripts/thread_env.sh` and verify effective thread counts. Record
  process affinity; do not impose a library affinity policy.
- Keep preparation outside timing, retain outputs until timing stops, include
  backend completion synchronization, and preserve numerical verification.
- Keep every run and failure. A setup or runtime failure must be diagnosed;
  label retries and their reason rather than overwriting failed evidence.
- Keep existing published records and all diagnostic experiments. Save fresh raw
  records and run metadata under a new timestamp before regenerating reports.

## Pre-collection configuration correction

The post-merge preflight found PyTorch intra-op=1 but inter-op=64 with the existing
thread environment. Before collecting any refresh measurements, the Python
runners used here were updated to set intra-op from `OMP_NUM_THREADS` and inter-op
to 1 at CLI startup, printing both effective values. The refresh uses 1 for both;
OpenBLAS also reports 1. This is an additional configuration difference from the
historical run, not an isolated CubeCL/materialization comparison. Tensor setup,
operation timing, backend selections and numerical tolerances are unchanged.

## AD verification correction (declared before the corrected AD runs)

The initial `20260919_142600` collection completed all eight suites, but source
inspection found that both AD runners marked successful execution as
`verification: passed` without comparing outputs. Those 100 AD rows are retained
as execution-only evidence, not numerical verification. Their first timings are
not substituted into the corrected AD reports.

Add an untimed directional derivative check: compare the JVP scalar, or the dot
product of the VJP gradient with the existing deterministic tangent, to a central
finite difference of the same backend's primal loss, with fixed step `h=1e-5`.
Use the existing YAML rtol/atol unchanged, reject non-finite values, and retain
verification failures. This checks one declared direction, not every gradient
component or cross-backend gauge equality. Downloads and reference evaluations
remain outside timing. Test the check, then rerun only the two affected AD
suites under a new timestamp; the other six suites need no remeasurement.

## Interpretation and acceptance

The historical published comparison is benchmark `9a5b104`, tenferro `40e24634`,
with records from `20260919_084000` (`tensornetwork`: `20260919_083200`). This
refresh is not a randomized paired causal experiment. Historical/current timing
ratios must not be presented as controlled proof that one isolated change caused
all differences.

All supported numerical checks must pass; report unsupported/not-configured
backends explicitly. Diagnose unexpected large-case regressions or runtime
failures rather than silently declaring success. The user explicitly accepts
small GPU-case latency regressions as a documented limitation: they do not block
adoption. Preserve the original failed notification nonregression gate and the
CPU/L3-placement diagnostics; do not relabel that experiment as successful.

Update generated reports and per-item notes with the actual merged revisions,
observations and remaining limitations. Verify report/record agreement, link
integrity and that regeneration does not overwrite hand-maintained notes.
Commit and push the coherent benchmark refresh only after verification.
