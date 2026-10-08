# Full GPU CI cost campaign (tenferro-rs #2002)

## Decisions

- Measure CI end-to-end paid cost, separately from steady-state tensor-operation
  performance. Fix the three original archives from tenferro-rs run
  37783818267, artifact 11553472760, and checkout tenferro-rs
  3f10f960af05efa4fc5705aac3dafbac1ae2269e for both arms. Expected workload:
  all 285 CUDA GPU cases, all three PJRT GPU cases and the CUDA tutorial. Keep
  nextest serial execution and the existing 200-second per-case timeout.
- Run complete fresh-pod jobs on A40 (baseline, price ceiling $0.60/hour) and
  RTX A5000 (candidate, ceiling $0.35/hour). Both use the same pinned CUDA 12.8
  image, workflow, test payload, toolchain and runtime preparation. No fallback
  GPU is an eligible sample; unavailable capacity fails explicitly without
  changing the selected arm. One provision attempt per trial bounds exploratory
  spend. A seeded CUDA image follows the unresolved direction recorded in
  tenferro-rs #1907; do not infer its production effect from the GPU comparison.
- Diagnostic pilot precedes confirmation; pilot data never enters the primary
  comparison. Confirmation order is baseline/candidate, candidate/baseline,
  baseline/candidate (three complete pairs, fresh pods, no selective retries).
  Primary statistic: median total estimated paid cost per complete workload;
  accept at least 20% reduction with every pair non-regressing. Require every
  test and tutorial to pass, correct GPU assignment, successful cleanup, complete
  cost timestamps, and a within-arm max/min cost ratio no larger than 1.5.
  An incomplete or noisy campaign is INCONCLUSIVE; repeat the complete campaign
  before claiming success. Record host CPU placement, cgroup quota and hardware;
  this is provider-assigned cloud CI cost, without CPU affinity changes or a
  tensor-operation speedup claim.
- Estimate paid cost using the pod's actual adjusted hourly price (list price
  fallback) and its lastStartedAt through confirmed deletion. Record rejected
  attempts and pilot expense separately; do not conceal failures as free runs.
  Successful-run savings do not recover the initial experiment spend immediately.

## Verification conclusions and constraints

- Current-production observation, before the five-part transport merged:
  tenferro-rs run 37783818267 passed the complete workload on A40 at $0.49/hour,
  estimated $0.079 before deletion. CUDA cases took 370.991 seconds; archive
  transfer 35 seconds, runtime restores 54 seconds, PJRT setup plus cases
  23 seconds. This existing observation motivates the campaign but is not a
  paired or statistically validated baseline.
- #2010 and #2011 mark deduplication, timestamp telemetry and the daily runner
  pin check as implemented; the never-picked-up watchdog and pre-seeding remain.
  #1933 shows why startup failures cannot simply be diagnosed as bad hardware.
- The current pod volume disk is deleted with each pod; it is not a shared
  network volume, so cross-pod cache reuse cannot be assumed. No persistent
  network storage is created by this campaign.
- The benchmark experiment evaluates the frozen complete CI workload. It does
  not publish operation benchmark tables and does not run PR-controlled
  tenferro workflows with repository secrets. Production deployment and an
  exact main-workflow cost comparison remain separate verification steps.

- A4500 diagnostic pilot 37787740830 stopped on no in-stock offer before pod
  creation; no GPU cost. A5000 replaces it before any paired measurements.
