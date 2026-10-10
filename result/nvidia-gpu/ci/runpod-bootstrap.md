# RunPod bootstrap CI measurements

**Decision: INCONCLUSIVE. The 10% goal is not confirmed; no production promotion.**

Metric: paid Pod lifetime, including setup and the complete CUDA/PJRT/tutorial workload. These are CI workflow measurements, not operation latency benchmarks.

The final NVRTC candidate measured medians of 381.896 → 328.569 seconds (13.96% observed reduction). Its last run used driver 570.211.01 instead of 570.195.03, violating the predeclared identity gate. The final sample is retained; it is not replaced or omitted.

[Complete protocols, observations, controller source and individual test results](runpod-bootstrap.json.gz). All 23 paid Pods were deleted; each passed the identical 285 CUDA / 3 PJRT / tutorial workload. One additional attempt failed before Pod allocation.

Total estimated GPU spending across all paid diagnostics and confirmations: $1.3662463. Provider image-cache state is uncontrolled. The current candidate's two-attempt limit is exhausted; further paid sampling is not scheduled.

## Final observed pairs

| Pair | Baseline seconds | Candidate seconds | Reduction | Driver identity |
| --- | ---: | ---: | ---: | --- |
| 1 | 381.896 | 328.569 | 13.96% | matched |
| 2 | 392.523 | 309.942 | 21.04% | matched |
| 3 | 356.320 | 355.454 | 0.24% | mismatch |

## All attempts

| Campaign | Arm | Run | Paid seconds | Driver |
| --- | --- | --- | ---: | --- |
| runpod-bootstrap-campaign | baseline | [38021085175](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38021085175) | 348.271 | 580.159.04 |
| runpod-bootstrap-campaign | candidate | [38021459568](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38021459568) | 355.202 | 580.173.02 |
| runpod-runner-image-campaign | baseline | [38022475657](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38022475657) | 376.778 | 580.178.04 |
| runpod-runner-image-campaign | candidate | [38022904055](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38022904055) | 327.120 | 580.173.02 |
| runpod-runner-confirmation | baseline | [38023892878](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38023892878) | 348.515 | 570.211.01 |
| runpod-runner-confirmation-v2 | baseline | [38024401197](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38024401197) | 355.077 | 580.159.04 |
| runpod-runner-confirmation-v3 | baseline | [38025076627](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38025076627) | 400.098 | 570.211.01 |
| runpod-runner-confirmation-v3 | baseline | [38025500323](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38025500323) | 384.674 | 580.178.04 |
| runpod-runner-confirmation-v4 | baseline | [38025993901](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38025993901) | 383.227 | 570.211.01 |
| runpod-runner-confirmation-v4 | baseline | [38026434121](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38026434121) | 366.365 | 570.211.01 |
| runpod-runner-confirmation-v4 | baseline | [38026823607](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38026823607) | 360.010 | 570.211.01 |
| runpod-runner-confirmation-v4 | candidate | [38027220726](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38027220726) | 346.326 | 570.211.01 |
| runpod-runner-confirmation-v4 | candidate | [38027577012](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38027577012) | 330.362 | 570.211.01 |
| runpod-runner-confirmation-v4 | baseline | [38027941308](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38027941308) | 357.142 | 570.211.01 |
| runpod-runner-confirmation-v4 | baseline | [38028333331](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38028333331) | 356.411 | 570.195.03 |
| runpod-runner-confirmation-v5 | baseline | [38029114443](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38029114443) | not allocated | — |
| runpod-runner-confirmation-v6 | baseline | [38029311950](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38029311950) | 400.192 | 570.195.03 |
| runpod-runner-confirmation-v6 | baseline | [38029746890](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38029746890) | 415.944 | 570.195.03 |
| runpod-runner-confirmation-v6 | baseline | [38030200863](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38030200863) | 381.896 | 570.195.03 |
| runpod-runner-confirmation-v6 | candidate | [38030631866](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38030631866) | 328.569 | 570.195.03 |
| runpod-runner-confirmation-v6 | candidate | [38030998512](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38030998512) | 309.942 | 570.195.03 |
| runpod-runner-confirmation-v6 | baseline | [38031349035](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38031349035) | 392.523 | 570.195.03 |
| runpod-runner-confirmation-v6 | baseline | [38031773497](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38031773497) | 356.320 | 570.195.03 |
| runpod-runner-confirmation-v6 | candidate | [38032171073](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38032171073) | 355.454 | 570.211.01 |

The compressed record retains each campaign’s frozen source and artifact identities, acceptance thresholds, validity failures, phase costs, host observations and all 288 CUDA/PJRT test results per paid run. Failed placement-recording and preallocation attempts are included. No missing cost is counted as zero.
