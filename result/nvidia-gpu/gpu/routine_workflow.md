# GPU routine command wall

Raw selection: `data/results/nvidia-gpu/gpu/routine_selection/20261013_234113`.

tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `9e89674e200526e294a01595152425befafd53f5`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Coverage `quick`, effort `standard`; core measured 27/27. Routine controls selected: True.

Controls measured 10/10; state observations 8/8. Untimed states are not operation timings.

| Command checkpoint/phase | Seconds |
|---|---:|
| Core through final suite report | 155.349 |
| Remaining controls/accounting/reporting | 57.386 |
| Script through final wall checkpoint | 212.735 |

Scope: script clock read through core and control collection, validation, accounting and report writes; excludes devcontainer CLI startup and final wall-record write.

External devcontainer command including CLI startup and final writes: **213.47 seconds**; tool `/usr/bin/time elapsed real`. This is an independently measured enclosing interval, not an additive phase.

Reference budget: **300 seconds** for NVIDIA GeForce RTX 3060, quick/standard. Conditions: existing CUDA devcontainer and dependency cache; mandatory Rust binary rebuilds included.

Budget scope: core and routine controls through accounting/reporting; devcontainer CLI startup and final wall metadata/report formatting excluded. Reference characterization: `data/results/nvidia-gpu/gpu/routine_selection/20261013_234113`.

Observation: **within budget**.

Budget status is descriptive and never changes functional success. A cold dependency build or busy host can exceed it; retain the observation rather than treating it as a confirmed regression. Detailed core case phases are in `routine_command_wall.md`; complete sequences and public state observations are in `routine_controls.md`. These observations establish no implementation speedup.
