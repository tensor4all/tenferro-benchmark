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

- A5000 pilot 37788211207 was rejected before pod creation by RunPod HTTP 500;
  no GPU cost. A40 diagnostic 37788536318 passed CUDA 285/285, PJRT 3/3 and
  tutorial, with confirmed deletion. It cost an estimated $0.08658 for 636.124s
  at $0.49/hour. Startup/queue was 152.783s, CUDA cases 378.776s, archive
  transfer/verification 2s, cuTENSOR setup 41s. The larger pre-seeded PyTorch
  image is not promoted: compared with the production observation it was
  about 10% more expensive, without a statistically validated causal claim.
- Next diagnostic candidate: NVIDIA CUDA runtime image only (2,056.6 MiB
  compressed layers, digest 4a801ef9232d2b05e69df4eb8aa054dbbe2824e5499e1e6e857320bb01ac41a9),
  with CI execution tools, cuTENSOR and PJRT wheels staged on the hosted runner.
  No primary paired measurements have started; the previous diagnostic data
  remain retained and excluded by the original pilot policy.

- Slim-image diagnostic uses A40 (ceiling $0.60/hour), retaining the frozen
  workload and serial test execution. It stages real Cargo and nextest binaries,
  cuTENSOR, CUDA headers and the three pinned PJRT wheels before provisioning.
  A local archived host test passed with no Rust compiler/toolchain on PATH;
  this establishes archive execution feasibility, not GPU correctness. Both
  workflow arms currently select this diagnostic configuration; no primary
  paired campaign is dispatched until distinct baseline/candidate preparation
  is restored and its protocol is recorded.

- Diagnostic 37791565589 failed during hosted wheel preparation: the explicitly requested manylinux2014
  platform excluded the pinned PJRT 0.10.2 manylinux_2_27 wheel. No pod was created. Switch the runtime image
  to Ubuntu 24.04 (digest ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867)
  and stage Python 3.12 wheels; retain pinned PJRT/NVIDIA versions.

- Retry 37791879934 exposed the same platform-filter error before provisioning.
  PyPI metadata confirms PJRT is a py3-none-manylinux_2_27 wheel, not a
  Python-ABI-specific wheel. Correct the download platform list to include
  manylinux_2_27 and retain manylinux2014 for the NVIDIA wheels. The earlier
  Python-version diagnosis was incorrect; no paid pods were created.

- Slim runtime preparation succeeded in 37792189451. A40 allocation was
  rejected before creation (no available instance, HTTP 500); no GPU charge.
  A6000 is also unavailable in the current price feed. Next diagnostic uses
  RTX 4090, ceiling $0.85/hour, to measure complete workload cost rather than
  selecting solely by hourly rate. It remains excluded from confirmation data.

- RTX 4090 diagnostic 37792733020 stopped on no available Secure Cloud offer
  before pod creation. The next pilot selects RTX A4000 (16 GB, observed
  $0.25/hour; ceiling $0.35/hour), with no GPU fallback or workload changes.
- The comparison helper evaluates the predeclared six-run order, complete
  frozen workload, confirmed cleanup, actual duration/price, within-arm 1.5
  noise gate, median >=20% reduction and pairwise nonregression. Pilots remain
  excluded. It supports reviewable arithmetic; measured inputs still require
  underlying workflow logs and provider timestamps.

- A4000 pilot 37793172714 stopped on unavailable stock before creation; no
  charge. Reuse its already prepared immutable five-part artifact for subsequent
  diagnostic allocations, removing the hosted preparation delay during which
  provider stock changes. The exact requested GPU and ceiling remain explicit;
  one allocation attempt, no fallback, and frozen payload checksums remain.
  This transport reuse is not a relaxation of primary confirmation validity.

- Slim runtime diagnostic 37793677143 passed all CUDA 285/285, PJRT 3/3 and
  tutorial on A40 ($0.49/hour), pod 3iabx5zw42ym9r deleted at
  2026-10-08T14:43:49.059609Z. Paid start 14:35:28.943Z gives 500.116609s
  and estimated $0.06807143. CUDA took 379.303s; preparation/transfer improved
  but this diagnostic does not meet 20% against the original single production
  observation ($0.07859461), and no paired campaign has confirmed savings.
- Baseline per-case logs reveal the full SVD tall/wide-above-1024 cases took
  83.584s/83.900s. Preserve threshold shapes and every acceptance identity while
  investigating the CPU Gram oracle, which currently recomputes both ordered
  column pairs. No diagnostic is retroactively promoted to confirmation.

- The next candidate changes only the full-SVD host oracle plus CI preparation
  and telemetry. Baseline source remains 3f10f960af05efa4fc5705aac3dafbac1ae2269e;
  candidate source must be one exact committed ref, recorded before any primary
  measurement. Numerical library code, threshold sizes, tolerances, all 285
  CUDA cases, three PJRT cases and tutorial remain. The comparison helper now
  requires one frozen ref per arm instead of the same ref across both arms,
  allowing the explicitly recorded test-oracle change. No primary runs have
  started, and no acceptance thresholds or prior pilot classifications change.

- Freeze source candidate 829a4b1aa2f89521730216251fee978d60fba081 (tenferro-rs
  fix/runpod-ci-cost-breakdown) before GPU diagnostics. It contains unchanged
  library kernels plus the host Gram checker and post-deletion telemetry.
  The first oracle diagnostic runs trusted root main workflow 3f10f960 with
  its original production image/preparation and serial workload. This isolates
  the host checker before combining it with staged runtime preparation.
  Numerical/debug assertions and the ci build profile remain unchanged.
