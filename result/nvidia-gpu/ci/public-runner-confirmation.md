# Public official runner bootstrap confirmation

**Verdict: INCONCLUSIVE.** All 18 allocations completed their numerical workload and Pod deletion, but the final run lacks required paid-lifetime and placement evidence. The full eight-pair primary comparison cannot be evaluated. No replacement, exclusion or repeat was performed.

Final run [38064818085](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38064818085): the pre-delete Pod metadata GET failed during DNS resolution after 2001 ms. The following DELETE returned HTTP 204 at 2026-10-10T15:49:42Z. Without the Pod record, the cost helper emitted no cost artifact and placement extraction had no input. GitHub marked the workflow successful because reporting is nonblocking; the experiment observer correctly rejected missing evidence.

## Verified outcome

- Two A/A runs and all eight A/B pairs were attempted, in the predeclared order. All 18 ran the exact 292 CUDA and three PJRT cases plus the tutorial successfully; all 18 Pod deletions were confirmed.
- A/A: 344.524 and 352.152 seconds, a 2.214% spread; the <=10% validity gate passed.
- Estimated GPU cost for the 17 complete records: $0.950932. The final Pod was deleted, but its cost record is unavailable. This subtotal is not the campaign total.
- Missing evidence is a terminal INCONCLUSIVE condition under the accepted one-attempt protocol. No primary p-value or pass is assigned to the incomplete experiment.

## Every allocated sample

| Position | Role | Run | Paid seconds | Driver | Cost estimate USD |
|---|---|---|---:|---|---:|
| 1 | A/A A | [38057206340](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38057206340) | 344.524 | 570.211.01 | 0.056464 |
| 2 | A/A A | [38057652140](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38057652140) | 352.152 | 570.211.01 | 0.057714 |
| 3 | pair 1 B | [38058097288](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38058097288) | 326.085 | 570.195.03 | 0.053442 |
| 4 | pair 1 A | [38058512806](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38058512806) | 347.288 | 570.195.03 | 0.056917 |
| 5 | pair 2 B | [38058969198](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38058969198) | 311.278 | 570.195.03 | 0.051015 |
| 6 | pair 2 A | [38059387668](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38059387668) | 352.527 | 570.195.03 | 0.057775 |
| 7 | pair 3 B | [38059835294](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38059835294) | 322.135 | 570.195.03 | 0.052794 |
| 8 | pair 3 A | [38060277401](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38060277401) | 353.386 | 570.195.03 | 0.057916 |
| 9 | pair 4 A | [38060740735](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38060740735) | 373.435 | 570.211.01 | 0.061202 |
| 10 | pair 4 B | [38061231994](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38061231994) | 314.424 | 570.211.01 | 0.051531 |
| 11 | pair 5 A | [38061655992](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38061655992) | 344.523 | 570.211.01 | 0.056463 |
| 12 | pair 5 B | [38062120486](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38062120486) | 285.534 | 570.211.01 | 0.046796 |
| 13 | pair 6 B | [38062510150](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38062510150) | 335.406 | 570.211.01 | 0.054969 |
| 14 | pair 6 A | [38062942301](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38062942301) | 370.745 | 570.195.03 | 0.060761 |
| 15 | pair 7 A | [38063409750](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38063409750) | 399.084 | 570.211.01 | 0.065405 |
| 16 | pair 7 B | [38063924492](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38063924492) | 312.547 | 570.211.01 | 0.051223 |
| 17 | pair 8 A | [38064354774](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38064354774) | 357.224 | 570.195.03 | 0.058545 |
| 18 | pair 8 B | [38064818085](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38064818085) | **missing** | 570.211.01 | **missing** |

## Partial observations (descriptive only)

The first seven complete pairs are shown only to describe the retained observations. They are not a substitute seven-pair experiment, not a successful primary result, and not evidence that the full predeclared 10% gate passed. Pair eight remains in the table above with its missing value.

| Statistic, first seven pairs only | Historical bootstrap | Official runner | Observed reduction |
|---|---:|---:|---:|
| Mean | 362.998 s | 315.344 s | 13.128% |
| Median | 353.386 s | 314.424 s | 11.025% |

| Stage mean, first seven pairs only | Baseline seconds | Candidate seconds |
|---|---:|---:|
| runner_startup_and_queue | 70.855 | 34.916 |
| cleanup_and_queue | 6.429 | 6.000 |
| other_setup | 11.000 | 12.286 |
| runtime_setup | 25.143 | 24.714 |
| archive_transfer | 3.571 | 3.286 |
| cuda_tests | 238.286 | 226.714 |
| tutorial | 1.000 | 0.714 |
| pjrt_tests_and_setup | 4.143 | 4.286 |
| unassigned_overhead | 2.571 | 2.429 |

## Scope and provenance

- Harness: `d0751b61c33fab56b55d5cd2844664b14c7d5a15` in tensor4all/tenferro-benchmark.
- Baseline source: `ff94aeded9cc98d6c49889c8ef6fca365763d93a` in tensor4all/tenferro-rs.
- Candidate and common tested source: `8dafe4202c33078f3be5c4821fc748e79b13c5ce` in tensor4all/tenferro-rs.
- Frozen artifact source run: 38051889375; all artifact identities and test identities are retained in the protocol.
- Both arms use the same current Ubuntu 24.04-built test archives and pre-expanded CUDA 12.8 payload, profile `ci`, and one nextest thread. This compares the historical bootstrap with the main implementation, not another 10% beyond main and not an isolated Ubuntu migration effect.
- The declared treatment changes the runner image/bootstrap, execution-package preparation, NVRTC installation method and readiness polling. The measured scope includes the entire paid Pod lifetime, not just setup.
- The approved design uses two A/A runs and eight randomized balanced pairs, a $0.60/hour price ceiling, and a $3 aggregate ceiling. Mean and median reductions must both reach 10%, with the predeclared randomization, tail and driver-distribution gates. Missing evidence prevents that full evaluation.
- The old strict-driver result remains INCONCLUSIVE. No old samples were reused.
- [Full logs, observations, bound protocol and original observer failure](public-runner-confirmation.json.gz)
- [Decision and preflight record](../../../docs/worklogs/2026-10-10-public-runner-confirmation-plan.md)

## Follow-up

Capture the accepted Pod metadata during provisioning and persist it before execution, then use it as the cost/placement fallback when the bounded pre-delete read fails. Keep deletion independent of telemetry and avoid extending paid lifetime just to retry a diagnostic read. This follow-up is not implemented by this result record and does not authorize a new paid campaign.

## Reproduction

Collection used the accepted harness above and the retained external protocol:

```bash
RUNPOD_COMPARISON_DIR=/tmp/runpod-public-confirmation-accepted-20261010 \
  python3 -u scripts/runpod-public-confirmation/run-comparison.py
```

The original observer state, final diagnosis and all samples are retained. Do not overwrite the state or allocate a replacement for the missing observation.
