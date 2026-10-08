# RunPod GPU benchmark workflow

`.github/workflows/benchmark-runpod-gpu.yml` provisions an ephemeral RunPod
GPU as a just-in-time GitHub Actions runner, runs the existing GPU benchmark
entry point, uploads the raw/report artifacts, and deletes the pod afterward.

Configure these repository or organization secrets before dispatching it:

- `RUNPOD_API_KEY`: RunPod API key, used only by the GitHub-hosted provisioning
  and cleanup jobs.
- `GH_APP_ID` and `GH_APP_PRIVATE_KEY`: a GitHub App allowed to create
  organization JIT runners.

The pod receives only the single-use JIT runner configuration. The workflow is
manual-dispatch only for now; it does not add RunPod credentials to pull
request jobs. `keep_failed_pod=true` is for debugging only and leaves the pod
billable until it is deleted manually.
