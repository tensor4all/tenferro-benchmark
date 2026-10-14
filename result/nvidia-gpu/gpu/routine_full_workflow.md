# GPU routine command wall

Raw selection: `data/results/nvidia-gpu/gpu/routine_selection/20261014_005047`.

tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `25f190048045c01cb363384ba45b2a0b0b640db8`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Coverage `full`, effort `scan`; core measured 63/95. Routine controls selected: True.

Controls measured 30/30; state observations 12/12. Untimed states are not operation timings.

| Command checkpoint/phase | Seconds |
|---|---:|
| Core through final suite report | 865.537 |
| Remaining controls/accounting/reporting | 86.360 |
| Script through final wall checkpoint | 951.897 |

Scope: script clock read through core and control collection, validation, accounting and report writes; excludes devcontainer CLI startup and final wall metadata/report formatting.

External devcontainer command including CLI startup and final writes: **952.71 seconds**; tool `GNU /usr/bin/time`. This is an independently measured enclosing interval, not an additive phase.

Reference budget: **300 seconds** for NVIDIA GeForce RTX 3060, quick/standard. Conditions: existing CUDA devcontainer and dependency cache; mandatory Rust binary rebuilds included.

Budget scope: core and routine controls through accounting/reporting; devcontainer CLI startup and final wall metadata/report formatting excluded. Reference characterization: `data/results/nvidia-gpu/gpu/routine_selection/20261013_234113`.

Observation: budget not evaluated for this hardware, selection or incomplete/unmeasured run.

Budget status is descriptive and never changes functional success. A cold dependency build or busy host can exceed it; retain the observation rather than treating it as a confirmed regression. Detailed core case phases and control observations are published separately for the same coverage profile and raw-run identity. These observations establish no implementation speedup.
