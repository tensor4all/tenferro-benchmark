# RunPod bootstrap CI measurements

Metric: paid Pod lifetime, including setup and the complete CUDA/PJRT/tutorial workload. These are CI workflow measurements, not operation latency benchmarks.

[Complete protocols, observations and individual test results](runpod-bootstrap.json.gz). All recorded Pods were deleted. No campaign here passed the acceptance gate.

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

The compressed record retains each campaign’s frozen source and artifact identities, acceptance thresholds, validity failures and all 288 CUDA/PJRT test results per run. The failed placement-recording attempts are included. Provider image-cache state is uncontrolled.
