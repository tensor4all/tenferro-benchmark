# RunPod GPU benchmark workflow

`.github/workflows/benchmark-runpod-gpu.yml` provisions an ephemeral RunPod
GPU as a just-in-time GitHub Actions runner, runs the existing GPU benchmark
entry point, uploads the raw/report artifacts, and deletes the pod afterward.

Configure these repository or organization secrets before dispatching it:

- `RUNPOD_API_KEY`: RunPod API key, used only by the GitHub-hosted provisioning
  and cleanup jobs.
- `GH_APP_ID` and `GH_APP_PRIVATE_KEY`: a GitHub App allowed to create
  organization JIT runners.

The workflow registers runners in organization runner group `4`
(`runpod-tenferro`). Allow this repository in that group's repository
access settings, including public repositories. To use a different group,
set the repository Actions variable `RUNPOD_RUNNER_GROUP_ID` to its numeric ID.

The pod receives only the single-use JIT runner configuration. The workflow is
manual-dispatch only for now; it does not add RunPod credentials to pull
request jobs. It uses the NVIDIA CUDA 12.8.1 development image and its default
entrypoint, installs the benchmark toolchain and Python environment, and checks
CUDA execution before collection.
The unique per-run label sends the benchmark job to the newly created RunPod
runner, rather than the existing `ubuntu-gpu` runner in the same group.

The configured GPU candidates mirror tenferro-rs's reviewed GPU tiers, including
RTX A-series, RTX 30/40/50-series, L4/L40/L40S, V100, and A100. RunPod selects by
availability; this is not price-ordered provisioning. The optional `gpu_type`
dispatch input restricts a run to one exact RunPod GPU type ID for diagnostics.
Record the actual GPU from the run metadata when comparing results.

Run a first GPU benchmark from GitHub Actions, or with:

```bash
gh workflow run benchmark-runpod-gpu.yml --repo tensor4all/tenferro-benchmark \
  -f suite=benchmarks/gpu/dense.yaml \
  -f backends=tenferro-cuda-trace,tenferro-cuda-eager,pytorch-cuda \
  -f keep_failed_pod=false
```

`suite=all` runs the standard `run_gpu_suite.sh` suites sequentially: dense,
einsum, sparse, and tensornetwork. A specific suite YAML or comma-separated list
can be supplied instead. Standalone GPU runners such as permutation and linalg
AD latency are not included in `all`.

Download the `runpod-gpu-benchmark-<run_id>` artifact for raw measurements under
`data/results/nvidia-gpu/` and reports under `result/nvidia-gpu/`. The workflow
does not commit generated reports. The benchmark entry point rebuilds the Rust
GPU binary before measurement.

By default the cleanup job deletes the pod, including after benchmark failure
or cancellation. `keep_failed_pod=true` retains the pod only when the benchmark
job fails; it is for debugging and leaves the pod billable until manually
deleted.
