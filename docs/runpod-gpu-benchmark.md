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
request jobs. It uses the same RunPod image and provisioning modules as
`tenferro-rs`, adapted for this repository. Imported modules and tests are
covered by `scripts/ci/LICENSE-tenferro-MIT`.

Provisioning runs on a GitHub-hosted runner:

1. Validate the reviewed GPU/CUDA allowlist against RunPod's live OpenAPI.
2. Order available candidates by live Secure Cloud price, falling back to the
   reviewed tier order when pricing is unavailable. Try one GPU type at a time.
3. Mint a fresh JIT configuration and unique runner label for every attempt.
4. Before registering the runner, prove NVRTC compilation, PTX loading, kernel
   launch, device synchronization, readback, and at least 8 GB VRAM.
5. Observe both the GitHub runner registry and RunPod's GraphQL runtime state.
   Delete rejected pods before trying another candidate. Stop if deletion
   cannot be confirmed.

The limits are six provisioning attempts, 420 seconds per startup, and early
termination after two consecutive created pods fail to register. Capacity
failures do not count as paid startup failures. Every attempt records GPU,
price, startup duration, and rejection reason in the Actions logs. Successful
provisioning publishes the accepted per-attempt label, preventing stale runners
from receiving the benchmark job. CUDA 12.8 or newer is required by this harness.

The optional `gpu_type` input restricts diagnostics to one reviewed GPU ID.
The accepted GPU is recorded in benchmark metadata; comparisons must use the
actual device, rather than treating all RunPod runs as the same hardware.

Before renting a GPU, a hosted Ubuntu 22.04 job builds a fresh Rust CUDA
benchmark and archives it with both source commits and its SHA-256 digest.
The accepted pod checks out the exact tenferro-rs commit used for that build
and verifies the artifact before sampling. Missing, stale, or modified artifacts
fail rather than silently rebuilding on a paid GPU.

The benchmark job installs the full CUDA toolkit and Python environment, then
runs the selected suites sequentially using the verified binary. Local GPU runs
continue to rebuild Rust before measurement.

Provisioning regression tests use injected transports and clocks and never
create paid resources:

```bash
uv run python -m unittest discover -s scripts/ci/tests -p 'test_*.py'
```

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
