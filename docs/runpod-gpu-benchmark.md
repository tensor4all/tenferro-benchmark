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

The pod receives only the single-use JIT runner configuration. The workflow
runs weekly and on manual dispatch; it does not add RunPod credentials to pull
request jobs.

## Weekly schedule, GPU type, and rental cap

The schedule fires four retry slots across the weekend (UTC): Saturday 08:00,
14:00, 20:00, and Sunday 08:00. `scripts/ci/weekly_gate.py` lets a scheduled
slot proceed only when no scheduled run has succeeded since Saturday 00:00 UTC,
so the first slot that obtains a GPU and completes provides that week's
results, and a slot that cannot obtain a card is retried by the next one
without renting anything in between. Manual dispatches always proceed.
Scheduled runs use `suite=all` and the default backends.

Both scheduled and dispatched runs request only `NVIDIA A100 80GB PCIe`, the
standard `nvidia-gpu` hardware. Passing a different reviewed `gpu_type` is
for diagnostics only; such results must not replace maintained A100 reports.

The watchdog starts deleting after 3300 seconds (`POD_MAX_LIFETIME_SECONDS`).
Its 180-second cleanup budget stays below one hour when deletion succeeds;
if the provider cannot confirm deletion, manual cleanup is required:

- The pod startup script arms a watchdog that deletes the pod after the
  budget, independently of GitHub, using the pod-scoped `RUNPOD_POD_ID` and
  `RUNPOD_API_KEY` that RunPod injects. `runpod_watchdog.py` embeds the same
  standard-library-only `runpod_http.py` client used by the hosted cleanup
  steps, with the API URL from `runpod_config.json`. It starts before package
  installation and does not depend on runner checkout or `runpodctl`.
  The benchmark job fails unless the watchdog is running and proves its
  pod-scoped credentials with a bounded read-only API call.
- On the GitHub side, startup is bounded by `startup_timeout_seconds` (420 s),
  the benchmark job by `timeout-minutes: 45`, and the cleanup job deletes the
  pod after success, failure, timeout, or cancellation.

If the standard suites outgrow the 45-minute job budget, split them across
dispatches rather than raising the cap. It uses the same RunPod image and provisioning modules as
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
from receiving the benchmark job. Rust uses CUDA 12.8, while the locked PyTorch
wheel uses CUDA 13.0. Provisioning requests CUDA 13.0-capable hosts and startup
rejects older drivers before transferring the runtime. Hosted preparation checks
the wheel's CUDA version so dependency upgrades cannot bypass this requirement.

## HTTP retries, polling, and failed cleanup

All four deletion paths (rejected candidates, startup-failure cleanup,
end-of-workflow cleanup, and the pod watchdog) use `runpod_http.py`:

- Up to five DELETE attempts within a shared 180-second budget, with each
  network request limited to 20 seconds or the remaining budget.
- Network errors, HTTP 408/429/500/502/503/504 use exponential backoff starting
  at 5 seconds, with positive jitter and a 60-second backoff cap. Retry-After
  seconds and HTTP-date values are minimum waits, even above that cap.
  A wait that cannot fit the budget ends retries without an early resend.
- HTTP 2xx and 404 count as success. After an exhausted transport failure,
  a verification GET can confirm 404, within the same overall budget.
  Other permanent errors stop immediately. No verification request bypasses
  a provider's Retry-After pause.
- Unconfirmed deletion prints a failure with the pod ID and dashboard cleanup
  instructions, and writes the same information to the Actions job summary
  when available. The command exits nonzero. Provisioning retains the pod ID
  and stops before creating another pod. The detached watchdog writes its
  failure to `/tmp/pod-watchdog.log`; it cannot deliver an Actions notification
  if the runner/workflow is already gone. An operator must check that log or
  the dashboard when hosted cleanup fails.

Startup observes GraphQL at most once per 10 seconds normally (6 requests/minute
per active provisioner, excluding request duration). Failed observations use
10/20/40/60-second exponential backoff with positive jitter, respect both
Retry-After formats, and reset after success. Cooldown survives candidate changes.
A failed observation remains unknown, not a container-stop signal. Missing
runtime fields and GraphQL errors also remain unknown. Authentication and other
permanent HTTP failures reject the candidate and enter cleanup. Pod and GitHub
runner requests use at most 30 seconds or the remaining 420-second startup
budget; no new poll starts after that budget, including a long Retry-After.

Pricing is one unauthenticated GraphQL query per provisioning invocation,
with a 20-second request timeout and immediate static fallback on failure;
there is no pricing retry loop to amplify 429s. Pricing and pod observations
share an endpoint cooldown within the provisioner, so a pricing Retry-After
also delays the first pod-state query. The workflow concurrency group
allows one active benchmark workflow in this repository. It does not coordinate
with tenferro-rs or other clients sharing an account: normal GraphQL demand is
approximately 6 requests/minute per active provisioner plus each invocation's
one pricing query. Budget shared-account usage using the sum across clients;
there is no cross-repository rate limiter. These client settings do not assert
that a particular provider quota is sufficient.

Run the mock/fixture coverage without renting a pod:

```bash
python3 -m unittest discover -s scripts/ci/tests -p 'test_runpod*.py'
```

The `gpu_type` input (default `NVIDIA A100 80GB PCIe`) restricts provisioning
to one reviewed GPU ID. The accepted GPU is recorded in benchmark metadata;
comparisons must use the actual device, rather than treating all RunPod runs
as the same hardware.

Before renting a GPU, a hosted Ubuntu 24.04 job builds a fresh Rust CUDA
benchmark and archives it with both source commits and its SHA-256 digest.
The hosted job caches Cargo registry/git dependencies and compiled artifacts,
including workspace crates, after resolving the lockfile. The cache separates
Ubuntu 24.04/CUDA 12.8 builds and includes Rust/toolchain and dependency inputs.
Only `main` saves it. Every invocation still runs `cargo build --locked` against
the current sources; a cache hit does not skip freshness or artifact checks.
The accepted pod checks out the exact tenferro-rs commit used for that build
and verifies the artifact before sampling. Missing, stale, or modified artifacts
fail rather than silently rebuilding on a paid GPU.

Another hosted Ubuntu 24.04 job prepares a managed Python 3.12.12 interpreter,
the frozen `uv.lock` environment, CUDA 12.8 shared runtime libraries and JIT headers, and
cuTENSOR 2.2.0.0. JAX CUDA wheels are included only when `jax-cuda` is requested;
their versions are also locked (`gpu` extra). The runtime archive is cached by
the lockfile, preparation scripts, platform, and tool versions. Only `main`
writes the shared cache. The hosted job compiles a CubeCL-style source with
the bundled NVRTC and CUDA/CCCL headers without requiring a GPU. Both
preparation jobs must finish before renting a GPU.

The runtime is created and restored at `/opt/tenferro-benchmark-ci` on both
hosts. This preserves Python virtualenv interpreter links and script paths;
it is not an arbitrarily relocatable virtualenv. The Ubuntu 24.04 RunPod image
(`runpod/pytorch:1.1.0-cu1281-torch291-ubuntu2404`) is pinned by digest; its
preinstalled framework is not used for timing. The accepted pod downloads
the immutable archive, checks its SHA-256 against the hosted job output and
its dependency identity, checks archive paths, and extracts it. Transfer uses
up to eight 1GiB parts as separate immutable artifacts; the download action
fetches up to five concurrently. Restoration checks every part's digest,
reconstructs the archive, and verifies the hosted archive digest before extraction.
An archive over 8GiB fails on the hosted job before renting a GPU. Transfer and
restore each have a ten-minute limit. A missing or invalid bundle fails;
there is no compiler or package-install fallback on the paid benchmark job.
`uv` runs offline without resynchronizing dependencies. The startup probe
uses the image's CUDA 12.8 NVRTC before registering a runner, then benchmark
preflight checks the restored libraries on the GPU.

Actions records hosted preparation, transfer, extraction, GPU preflight, and
measurement as separate steps; scripts also log preparation/compression and
verification/extraction durations. Local GPU runs continue to rebuild Rust
before measurement. See issue #128 for the 10m48s previous Pod setup baseline.

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
  -f gpu_type="NVIDIA A100 80GB PCIe" \
  -f keep_failed_pod=false
```

For hosted preparation/cache diagnostics without renting a GPU, dispatch with
`-f prepare_only=true`. This still prepares and verifies both artifacts, but
skips provisioning and measurements. It produces no new benchmark receipt.

`suite=all` runs the standard `run_gpu_suite.sh` suites sequentially: dense,
einsum, sparse, and tensornetwork. A specific suite YAML or comma-separated list
can be supplied instead. Standalone GPU runners such as permutation and linalg
AD latency are not included in `all`.

Download the `runpod-gpu-benchmark-<run_id>` artifact for raw measurements under
`data/results/nvidia-gpu/` and reports under `result/nvidia-gpu/`. The workflow
does not commit generated reports. The hosted build produces a fresh Rust binary
for every invocation; local benchmark entry points rebuild before measurement.

After successful GPU or CPU nightly runs on `main`, `publish-results.yml`
automatically deploys the reports to https://tensor4all.org/tenferro-benchmark/.
It starts with maintained reports in `result/`, overlays successful CPU
artifacts, then takes the latest validated GPU artifact separately for each
suite. Running one GPU suite therefore preserves other suites' latest reports.
Manual publication and the daily schedule remain available.

GPU publication is a per-suite choice between the repository's committed A100
report/summary and the latest successful weekly RunPod bundle. The RunPod bundle
wins only when it is newer than the summary already copied into the site's
`result/nvidia-gpu` tree and is eligible; a bundle is eligible only when its
recorded GPU is an NVIDIA A100 and it contains the dashboard `*.summary.json`
sidecar produced by the current harness. Non-A100 bundles and older-harness
bundles without a summary are skipped with one log line, leaving the repository
A100 results in place. The chosen report `.md`, summary `.json`, and raw run are
copied into the Pages tree before the dashboard generator runs.

GPU publication requires every requested problem/backend row to be present,
with finite measurements and passing verification. Explicit unsupported rows
remain labeled unsupported; runtime failures, unconfigured backends, missing
rows, and failed verification block publication. The bundle includes only this
invocation's raw data and reports, plus a hashed publication manifest. Failed
runs preserve available diagnostic raw data without a publication manifest.
Pages rechecks the manifest and measurements without executing artifact code.
It searches the latest 100 successful main runs; artifacts are retained for
30 days. When artifacts expire, the maintained repository reports provide the
fallback. Raw data are published under `raw/nvidia-gpu/gpu/<suite>/<timestamp>/`.

CPU nightlies resolve `tenferro_ref` once per invocation and keep that detached
revision for all suites. Linux collection runs in the maintained MKL CPU
devcontainer; macOS uses native execution. Failed CPU artifacts are diagnostic
only and are not overlaid by Pages. Shared hosted CPU runners remain reference
indicators rather than canonical regression measurements.

By default the cleanup job deletes the pod, including after benchmark failure
or cancellation. `keep_failed_pod=true` retains the pod only when the benchmark
job fails; it is for debugging and leaves the pod billable until manually
deleted.
