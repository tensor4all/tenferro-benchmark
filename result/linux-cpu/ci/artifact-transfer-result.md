# Raw artifact transfer screening: FAIL

[CPU workflow](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38068119138) at harness `6ef11b0a42772798ab8f5f51964cfd5a795fa833`.

The complete one-attempt trial passed byte equivalence and A/A validity but failed the predeclared 20% transfer-reduction gate. Do not change production transport or allocate GPUs for this candidate.

| Statistic, common + CUDA 12.8 | ZIP level 0 | Raw |
|---|---:|---:|
| Mean, four paired samples per arm | 23.996 s | 25.320 s |
| Median | 24.431 s | 24.648 s |

Raw was 5.52% slower by observed mean and 0.88% slower by median. Bundle median ratios (raw/ZIP) were 0.963 common and 1.075 CUDA SDK. A/A max/min ratios were 1.128 common and 1.061 SDK, within the declared 1.15 limit. These four pairs do not establish a general slowdown; they provide no evidence to promote the candidate.

| Sample | Arm | Common seconds | SDK seconds |
|---|---|---:|---:|
| 1 | ZIP | 13.071 | 11.925 |
| 2 | ZIP | 11.591 | 11.238 |
| 3 | raw | 10.689 | 14.207 |
| 4 | ZIP | 11.461 | 12.455 |
| 5 | raw | 11.585 | 12.780 |
| 6 | ZIP | 13.026 | 12.094 |
| 7 | ZIP | 12.165 | 9.837 |
| 8 | raw | 14.001 | 13.618 |
| 9 | ZIP | 12.112 | 12.835 |
| 10 | raw | 11.782 | 12.618 |

All 20 bundle downloads passed exact part sizes, part hashes, and concatenated archive hashes. No sample was omitted or replaced. Official raw upload naming and download placement worked; removing ZIP is functionally feasible. The raw expected archive hashes were transported in a separate preparation manifest before timing. Production adoption would require wiring that metadata across its reusable workflows; it is unnecessary given this negative result.

The workload reused the exact ten compressed parts from source run 38056837195. Preparation uploaded both arms with pinned official Actions; all measurements ran sequentially on one GitHub-hosted Ubuntu 24.04 AMD EPYC 7763 VM. Default affinity and five concurrent downloads per bundle were retained. Timing includes action setup, network transfer, extraction and hashing through the next timer step, but excludes the independent byte validation. Host/image/load metadata are retained. No GPU was allocated for this screening. GitHub storage/network caches were not controlled, so this is a CPU-runner transport observation, not a RunPod paid-lifetime estimate.

## Prior RunPod transfer diagnosis

In run 38066318486, the common parts finished at 6.80, 6.64, 7.27, 118.80 and 126.10 seconds from their respective requests. SDK parts took 111.61, 6.37, 115.88, 7.25 and 122.23 seconds. The next run used the same immutable artifacts and all parts finished in about 7–8 seconds. No explicit retry messages appeared. Existing logs do not isolate first-byte latency, network throughput and ZIP processing, so the slow-tail cause remains unproven. The CPU raw comparison does not reproduce the RunPod slow tail.

[Per-artifact diagnosis, first run](prior-transfer-38066318486.json); [second run](prior-transfer-38067070576.json). The reusable parser joins interleaved completion events by digest, not log order, and leaves missing or ambiguous completions unassigned.

[Predeclared protocol](artifact-transfer-protocol.json); [full evidence](artifact-transfer-result.json.gz), including all observations, logs, jobs, host metadata and exact payload identities.
