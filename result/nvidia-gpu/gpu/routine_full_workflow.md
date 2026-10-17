# GPU routine command wall

Raw selection: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/data/results/nvidia-gpu/gpu/routine_selection/20261017_104441`.

tenferro-rs `88daa885a5bb09e07f3031814bbb1a21f7a74e62`; harness `2f15f3d00b8565b86873285e6c39d65d15941b3f`. GPU: NVIDIA A100 80GB PCIe (`GPU-530977e1-4968-9283-4129-9fbec3e66542`).

Coverage `full`, effort `standard`; core measured 75/94. Routine controls selected: True.

Controls measured 30/30; state observations 12/12. Untimed states are not operation timings.

| Command checkpoint/phase | Seconds |
|---|---:|
| Core through final suite report | 458.837 |
| Remaining controls/accounting/reporting | 75.272 |
| Script through final wall checkpoint | 534.109 |

Scope: script clock read through core and control collection, validation, accounting and report writes; excludes devcontainer CLI startup and final wall metadata/report formatting.

Reference budget: **300 seconds** for NVIDIA GeForce RTX 3060, quick/standard. Conditions: existing CUDA devcontainer and dependency cache; mandatory Rust binary rebuilds included.

Budget scope: core and routine controls through accounting/reporting; devcontainer CLI startup and final wall metadata/report formatting excluded. Reference characterization: `data/results/nvidia-gpu/gpu/routine_selection/20261013_234113`.

Observation: budget not evaluated for this hardware, selection or incomplete/unmeasured run.

Budget status is descriptive and never changes functional success. A cold dependency build or busy host can exceed it; retain the observation rather than treating it as a confirmed regression. Detailed core case phases and control observations are published separately for the same coverage profile and raw-run identity. These observations establish no implementation speedup.
