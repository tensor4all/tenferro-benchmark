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

## Debugging A100 availability

Run **debug-runpod-a100** from the Actions tab using **Run workflow**. It needs
only `RUNPOD_API_KEY`, runs on a GitHub-hosted runner, and performs read-only
requests without building binaries, renting pods, or registering runners.

The job compares the production Secure Cloud catalog query (including the
configured CUDA versions and GPU count) with the same query without the CUDA
filter. For each of the three reviewed A100 models it records catalog presence,
stock, Secure Cloud eligibility, VRAM, price, and the existing provisioning
code's candidate selection. It also checks the A100 IDs and CUDA syntax against
the live OpenAPI schema. Production settings are unchanged.

Read the job summary or download `runpod-a100-diagnostics-<run>-<attempt>`:
`summary.md` and `diagnostics.json` include timestamps and HTTP statuses even
when an API check fails. API failures fail the job; an empty stock snapshot is
a successful diagnostic observation. Response bodies from failed requests,
headers, and credentials are not archived.

If stock appears only without the CUDA filter, investigate host driver
availability. If both queries show no usable A100 offers, capacity may be the
limiting factor. Catalog stock is a snapshot and does not prove a create request
will succeed; account balance, placement errors, and CUDA startup/runner failures
still require the provisioning logs from `benchmark-runpod-gpu`.

## Weekly schedule, GPU type, and rental cap

The schedule fires four retry slots across the weekend (UTC): Saturday 08:00,
14:00, 20:00, and Sunday 08:00. `scripts/ci/weekly_gate.py` lets a scheduled
slot proceed only when no scheduled run has succeeded since Saturday 00:00 UTC,
so the first slot that obtains a GPU and completes provides that week's
results, and a slot that cannot obtain a card is retried by the next one
without renting anything in between. Manual dispatches always proceed.
Scheduled runs use `suite=all` and the default backends.

Both scheduled and dispatched runs request the `a100` tier: any in-stock
A100 (`NVIDIA A100 80GB PCIe`, `NVIDIA A100-SXM4-80GB` or
`NVIDIA A100-SXM4-40GB`), the standard `nvidia-gpu` hardware. The exact model
is recorded in the run metadata and summaries. Passing a non-A100 reviewed
`gpu_type` is for diagnostics only; such results must not replace maintained
A100 reports.

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
- On the GitHub side, startup is bounded by `startup_timeout_seconds` (600 s),
  the benchmark job by `timeout-minutes: 42`, and the cleanup job deletes the
  pod after success, failure, timeout, or cancellation.

If the standard suites outgrow the 42-minute job budget, split them across
dispatches rather than raising the cap. It uses the same RunPod image and provisioning modules as
`tenferro-rs`, adapted for this repository. Imported modules and tests are
covered by `scripts/ci/LICENSE-tenferro-MIT`.

Provisioning runs on a GitHub-hosted runner:

1. Check the v2 create schema, resolve reviewed GPU IDs against the catalog,
   and validate CUDA version syntax. Live capacity remains a separate check.
2. Order available candidates by live Secure Cloud price, falling back to the
   reviewed tier order when pricing is unavailable. Try one GPU type at a time.
3. Mint a fresh JIT configuration and unique runner label for every attempt.
4. Before registering the runner, prove NVRTC compilation, PTX loading, kernel
   launch, device synchronization, readback, and at least 8 GB VRAM.
5. Observe both the GitHub runner registry and REST v2 Pod status/runtime.
   Delete rejected pods before trying another candidate. Stop if deletion
   cannot be confirmed.

The limits are six provisioning attempts, 600 seconds per startup, and early
termination after two consecutive created pods fail to register. Capacity
failures do not count as paid startup failures. Every attempt records GPU,
price, startup duration, and rejection reason in the Actions logs. Successful
provisioning publishes the accepted per-attempt label, preventing stale runners
from receiving the benchmark job. Rust uses CUDA 12.8, while the locked PyTorch
wheel is pinned to PyTorch 2.11.0+cu128 in the separate
`scripts/ci/runpod_runtime/` project. Catalog queries and Pod creation both use
`minCudaVersion: "12.8"`, accepting CUDA 12.8 and newer hosts without an exact
version allowlist; startup requires at least CUDA 12.8 driver support before
transferring the runtime. Both hosted preparation and restoration check the
wheel's version and CUDA version. The root CPU dependency environment is separate.
RunPod's `cudaVersions`/`allowedCudaVersions` filters match exact versions: the A100
catalog observed on 2026-10-22 advertised CUDA 13.2, so allowing only 13.0
excluded compatible newer hosts as well as CUDA 12 hosts. The minimum-version
filter avoids having to add future driver versions by hand. Diagnostics retain
the catalog's advertised CUDA versions to make future version gaps visible.

## REST v2 integration

All RunPod operational calls use `https://api.runpod.io/v2`. The migration follows
[the official guide](https://docs.runpod.io/api-reference-v2/migrate-from-v1)
and [the published OpenAPI](https://api.runpod.io/v2/openapi.json).

| Purpose | REST v2 API and fields |
| --- | --- |
| Create | `POST /v2/pods`: `image`, `cloud`, `gpu.id/count/minCudaVersion`, `disk`, `mounts.persistent`, `entrypoint`, `cmd`, `env` |
| Assigned hardware and cost | Created Pod's `gpu.id` and `cost` |
| Pricing and stock | `GET /v2/catalog/gpus`: wrapped `gpus`, `price.secure`, `memory`, `availability` |
| Startup and container liveness | `GET /v2/pods/{id}`: `status` and `runtime` |
| Diagnostic listing | `GET /v2/pods`: wrapped `pods` with cursor pagination |
| Deletion | `DELETE /v2/pods/{id}` |

A v2 create selects exactly one GPU type; both priced candidates and static
tiers are expanded into individual requests, under the existing attempt cap.
The deployment remains Secure Cloud with no spot configuration. Invalid
requests and insufficient balance stop creation; known capacity failures and
inaccessible pools move to the next candidate. RFC 9457 problem diagnostics
are redacted before logging.

The previous GraphQL pricing and container-state queries have been removed.
The replacement uses the authenticated
[GPU catalog](https://docs.runpod.io/api-reference-v2/catalog/list-gpu-types)
and [Pod retrieval](https://docs.runpod.io/api-reference-v2/pods/get-a-pod).
Only `RUNNING` with a runtime plus the fresh runner registration is accepted.
`EXITED`, `ERROR`, and `TERMINATED` reject immediately; loss of a previously
observed runtime is also checked, rather than trusting status alone.

The contract check handles v2 `allOf`/local refs. GPU IDs come from the catalog,
not a schema enum. CUDA strings are checked against the schema's format;
this does not prove a version is offered on a particular machine. The catalog
availability filter and the on-Pod CUDA smoke proof keep that distinction.
Fixtures in `scripts/ci/tests/fixtures/runpod_v2_*` record the public schema
source/digest and response examples. Tests validate payloads against its actual
constraints, including rejection of old flat fields.

To inspect possible remaining Pods without exposing environment variables or
startup commands (requires a key with Pod read access):

```bash
python3 -m scripts.ci.runpod_http list --config scripts/ci/runpod_config.json
```

This reads all cursor pages within a 120-second/100-page bound and prints only
Pod ID, status, and cost. It does not automatically delete anything; no new
reaper or retention policy is introduced.

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

Startup observes `GET /v2/pods/{id}` at most once per 10 seconds normally (6 requests/minute
per active provisioner, excluding request duration). Failed observations use
10/20/40/60-second exponential backoff with positive jitter, respect both
Retry-After formats, and reset after success. Cooldown survives candidate changes.
A failed observation remains unknown, not a container-stop signal. Missing
runtime fields, malformed responses, and unknown lifecycle states also remain unknown. Authentication and other
permanent HTTP failures reject the candidate and enter cleanup. Pod and GitHub
runner requests use at most 30 seconds or the remaining 600-second startup
budget; no new poll starts after that budget, including a long Retry-After.

Pricing is one authenticated catalog GET per provisioning invocation, with a
20-second request timeout and immediate static fallback on failure. The query
requests `include=AVAILABILITY&product=POD&cloud=SECURE`, the GPU count, and the
configured exact CUDA versions. It reads `price.secure` (the list price per GPU),
`memory` (VRAM), and `availability`; actual billed hourly Pod cost comes from
`cost` on the created Pod. Missing/unknown stock is not treated as available.
Catalog prices do not imply negotiated discounts or reserved capacity.

Catalog reads, creation, Pod observations, and rejected-Pod deletion share a
process-local cooldown, so switching operations or candidates cannot bypass
Retry-After. Creation also respects HTTP-date Retry-After values and does not
shorten a provider wait to fit its deadline. The standalone workflow cleanup
steps and detached watchdog each have their own retry budget.

The workflow concurrency group allows one active benchmark workflow in this
repository. It does not coordinate with tenferro-rs or other clients sharing an
account. Startup demand is at most approximately 6 Pod GETs/minute per active
provisioner, plus one pricing catalog GET per invocation and bounded create and
cleanup requests. Contract preflight separately reads the OpenAPI document and
the GPU catalog before renting. Budget shared-account usage using the sum
across clients; there is no cross-repository rate limiter. These settings do
not assert that a particular provider quota is sufficient.

Run the mock/fixture coverage without renting a pod:

```bash
python3 -m unittest discover -s scripts/ci/tests -p 'test_runpod*.py'
```

The `gpu_type` input (default `a100`) restricts provisioning to one tier of
`scripts/ci/runpod_config.json` or to one reviewed GPU ID. The accepted GPU is recorded in benchmark metadata;
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
the frozen `scripts/ci/runpod_runtime/uv.lock` environment, CUDA 12.8 shared runtime libraries and JIT headers, and
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
  -f gpu_type=a100 \
  -f keep_failed_pod=false
```

For hosted preparation/cache diagnostics without renting a GPU, dispatch with
`-f prepare_only=true`. This still prepares and verifies both artifacts, but
skips provisioning and measurements. It produces no new benchmark receipt.

`suite=all` runs the standard `run_gpu_suite.sh` suites sequentially: dense,
einsum, sparse, and tensornetwork. A specific suite YAML or comma-separated list
can be supplied instead. Standalone GPU runners such as permutation and linalg
AD latency are not included in `all`.

Download the `runpod-gpu-benchmark-<run_id>-attempt<attempt>` artifact
(legacy bundles use the run-ID-only name) for raw measurements under
`data/results/nvidia-gpu/` and reports under `result/nvidia-gpu/`. The workflow
does not commit generated reports. The hosted build produces a fresh Rust binary
for every invocation; local benchmark entry points rebuild before measurement.

After successful GPU or CPU nightly runs on `main`, `ingest-results.yml`
downloads that exact run's result bundle, validates it, and commits accepted
report/summary/provenance files. The GPU schema-2 publication manifest is
revalidated against raw measurements, digests and NVIDIA A100 hardware (any
A100 model). Missing/expired artifacts, incorrect hardware,
incomplete suites, and failed verification are visible errors; there is no
latest-run search or silent artifact fallback. Older valid results and duplicate
submissions are explicit no-ops. Complete selected suites update independently
without removing other committed suites.

`publish-results.yml` generates https://tensor4all.org/tenferro-benchmark/ only
from committed `result/` and git history. It does not download benchmark data.
Raw bundles remain in the exact source run's artifact store, linked from each
accepted suite's provenance. New result bundles retain raw data for 90 days;
legacy GPU bundles may expire after the earlier 30-day policy. The API-recorded
expiry is authoritative. Expiry does not change published report data.

CPU nightlies resolve `tenferro_ref` once per invocation and keep that detached
revision for all suites. Linux collection uses the maintained MKL CPU
container; macOS uses native execution. CPU bundles explicitly identify the
selected 1T/4T raw sources, and ingestion regenerates summaries with trusted
formatters. Failed runs upload separate diagnostics and cannot publish. Shared
hosted CPU runners remain reference indicators, not controlled performance
measurements. See [result ingestion](result-ingestion.md) for run-ID recovery.

By default the cleanup job deletes the pod, including after benchmark failure
or cancellation. `keep_failed_pod=true` retains the pod only when the benchmark
job fails; it is for debugging and leaves the pod billable until manually
deleted.
