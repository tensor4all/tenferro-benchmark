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

## tenferro-rs revision the GPU job builds

The RunPod workflow builds `pins/tenferro-rs.rev` — the same revision CI checks
out — and the pod is pinned to the commit that build produced. The pin is owned
by `campaign-sync-pins` and shared with the sibling repositories, so it is not
edited here: a bump arrives through that sync, and any source change the new API
needs lands in the same pull request.

Building a fresh `main` checkout here instead is what hid a break once: the GPU
job tracked `main` while CI compiled the pin, so a renamed driver enum failed
every weekly collection while CI stayed green.

## Weekly schedule, GPU type, and rental cap

The schedule fires four retry slots across the weekend (UTC): Saturday 08:00,
14:00, 20:00, and Sunday 08:00. `scripts/ci/weekly_gate.py` lets a scheduled
slot proceed only when no scheduled run has succeeded since Saturday 00:00 UTC,
so the first slot that obtains a GPU and completes provides that week's
results, and a slot that cannot obtain a card is retried by the next one
without renting anything in between. Scheduled runs use `suite=all`, the default
backends, and count as `weekly` attempts.

### Dispatch purposes and their budgets

Every dispatch carries a `purpose` input, and the run name records it:
`RunPod GPU benchmark [purpose=verify target=tensor4all/tenferro-rs#419]`. A run
with no marker counts as `weekly`, the strict direction.

- `weekly` and `backfill` are attempts to satisfy the schedule and share a cap
  of **3 per weekly window** (Saturday 00:00 UTC onward). Restarting a failed
  weekly collection counts here, which is what the cap exists for.
- `verify` checks a fix and has its own cap of **6 per weekly window**, and it
  must name what it verifies through `target: <repo>#<n>`.

The budget counts manual dispatches. A scheduled slot is not a dispatch, and a
slot that skipped never rented a GPU, so the weekend retries do not spend the
budget. A week whose dispatches already spent it refuses the slots too. Counters
are per branch: a dispatch from a feature branch spends that branch's budget and
never refuses main's scheduled slot.

`scripts/ci/weekly_gate.py` runs before anything is rented and again immediately
before the pod is created, prints both counters (`weekly 1/3, verify 4/6`) into the step summary, and fails
the run with the reason when the budget is spent. The answer to a 7th
verification is to batch the fixes into one run in the next weekly window.
Manual dispatches proceed only within those budgets, and no re-run rents: what
the attempt it repeats actually did cannot be told from the record (a failure can
follow a successful collection, and a failed provisioning can leave a pod
billing), so a scheduled slot retries at its next slot and a retry that needs a
GPU is a dispatch carrying its purpose. A slot that collected keeps counting after
a re-run of it was refused: that refusal leaves the run failed, so the attempt it
repeats is read from its own history and the next slot still sees the week.
A readback dispatch is counted under the
purpose it was given. The counters are read back from the run names in the current
window, counting the runs that started before this one plus any successor that
already finished (the concurrency group does not order runs by id), so there is no
second place to update. A window of `WINDOW_LIMIT` runs fails
closed instead of counting a truncated history.

Both scheduled and dispatched runs request the `a100` tier: any in-stock
A100 (`NVIDIA A100 80GB PCIe`, `NVIDIA A100-SXM4-80GB` or
`NVIDIA A100-SXM4-40GB`), the standard `nvidia-gpu` hardware. The exact model
is recorded in the run metadata and summaries. Passing a non-A100 reviewed
`gpu_type` is for diagnostics only; such results must not replace maintained
A100 reports.

The watchdog starts deleting after 7200 seconds (`POD_MAX_LIFETIME_SECONDS`). At the A100 PCIe
rate of $1.79/h a full two-hour pod costs $3.58 of compute (extending 55 -> 120 min adds $1.94);
a run that finishes early pays only for the time it used. That is the arithmetic, not a hard
billing ceiling: deletion retries can add up to 180 s (~$0.09) and time before the watchdog is
armed is billed too. The collection itself is not bounded inside the two hours yet: cutting it
short can strand a suite after its run directory exists but before its reports are written, and
packaging then refuses the whole bundle (see the follow-up issue). A run that does not finish
before the watchdog deletes the pod is billed for the time it used.
If the provider cannot confirm deletion, manual cleanup is required:

- The pod startup script arms a watchdog that deletes the pod after the
  budget, independently of GitHub, using the pod-scoped `RUNPOD_POD_ID` and
  `RUNPOD_API_KEY` that RunPod injects. `runpod_watchdog.py` embeds the same
  standard-library-only `runpod_http.py` client used by the hosted cleanup
  steps, with the API URL from `runpod_config.json`. It starts before package
  installation and does not depend on runner checkout or `runpodctl`.
  The benchmark job fails unless the watchdog is running and proves its
  pod-scoped credentials with a bounded read-only API call.
- On the GitHub side, startup is bounded by `startup_timeout_seconds` (600 s),
  the benchmark job by `timeout-minutes: 130` (it used to be 42, which cut the
  collection short of the new lifetime), and the cleanup job deletes the pod after success,
  failure, timeout, or cancellation. Both hosted timeouts are deliberately above
  the watchdog: the pod stops itself first.

If the standard suites outgrow the two-hour pod, split them across
dispatches rather than raising the pod lifetime again.
It uses the same RunPod image and provisioning modules as
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
from receiving the benchmark job. Readiness queries GitHub by that runner's
unique name and verifies its label and online status, so registrations beyond
the first 100 organization runners remain discoverable.
Rust uses CUDA 12.8, while the locked PyTorch
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

## JAX compilation cache on a network volume

The pod is ephemeral, so XLA's persistent compilation cache only survives the
night if it is written to storage the next pod can mount. RunPod **network
volumes** provide that; GitHub Actions caches do not, because an entry evicted
after seven idle days would silently change the measurement conditions of a
weekly job.

### What is cached, and why it does not expire weekly

XLA compiles per case, and the result depends only on the JAX version, the
CUDA runtime/driver version and the program itself (shapes, ops, compiler
flags). It does **not** depend on which tenferro revision produced those shapes:
the same `(4096, 4096)` SVD is the same CUDA kernel every week. The cache key is
therefore

```
sha256( jax.__version__ | cuda_version | driver_version | sorted(case matrix) )
```

and the tenferro pin is deliberately excluded — including it would miss the
cache on every weekly run, which is the only situation this exists for. The
case matrix is one entry per `(suite, problem, JAX arm)` built from the routine
manifest, with `run`/`verify`/`notes` stripped: sampling counts and tolerances
cannot change a compiled program, so editing them must not invalidate a
generation. A JAX upgrade, a CUDA/driver change or a shape/op/arm change does
start a new generation, which is the intent: a stale cubin is the one failure
mode worth paying a recompile to avoid.

`scripts/ci/jax_cache.py` owns the key, the generation directories and the
pod-side setup; `scripts/ci/tests/test_jax_cache.py` pins the key's
determinism, the pruning rule and the recording.

### Setup

Create the volume once, in a data center where the benchmark GPU types are
offered, and put its ID in the repository variable
`RUNPOD_JAX_CACHE_VOLUME_ID` (Settings → Variables → Actions). Without that
variable nothing changes: no volume is attached and the run collects with an
empty cache, logging `jax compile cache: disabled (no network volume
configured)`.

Creating the volume needs `RUNPOD_API_KEY`, which only the workflows hold, so it
runs there: Actions → "Create the JAX compile-cache network volume", with
`data_center` set to a data center that has the A100 and `dry_run=true` first.
Before anything is created the script checks the data center catalog — the data
center must exist, support network volumes and offer the configured `a100` tier
— because the binding cannot be changed afterwards. The run is idempotent by
name, refuses a same-named volume whose data center or size differ, and refuses
to proceed when the name appears twice: a name is not a unique key in the RunPod
schema. A POST whose response was lost is never re-sent — it is reconciled by
reading the listing back, and the run stops if the volume is not visible yet —
because re-sending it can pay for a second volume.

The volume is mounted at `network_volume_mount_path`
(`scripts/ci/runpod_config.json`, default `/opt/bench-cache`). REST v2 accepts at
most one mount kind per pod, so a configured volume replaces the host-local
`mounts.persistent` entry; `runpod_provision.py` reads the ID from
`RUNPOD_NETWORK_VOLUME_ID`, and `runpod_contract.py` validates its shape before
a paid request is made.

A stale volume keeps billing, and the data center it was created in may stop
offering the A100. "Delete a network volume" removes one: it requires the id and
the name (a name is not a unique key, so the name is confirmed against the id
before anything is deleted), runs a dry run by default, and refuses the volume
named by `RUNPOD_JAX_CACHE_VOLUME_ID`. `debug-runpod-a100` reports the data
centers that offer the A100 and support a network volume, which is where a
replacement belongs.

A pod can only attach a volume in the pod's own data center, and the scheduler
otherwise picks one freely. `runpod_provision.py` therefore reads the binding
back from `GET /v2/network-volumes/{id}` and sends `dataCenterIds: [<that data
center>]`, so there is no second variable to keep in sync and no way to point
the pod at the wrong one.

### Pod-side behavior

The `Configure the JAX compilation cache` step derives the key from the
installed runtime and exports three variables for the collection:

| Variable | Value |
| --- | --- |
| `JAX_COMPILATION_CACHE_DIR` | `<mount>/jax/<key>` |
| `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` | `1` — only expensive compiles are stored |
| `JAX_PERSISTENT_CACHE_MAX_SIZE_BYTES` | `4294967296` (4 GiB) |

Under `<mount>/jax/`, each key is one directory. At startup the step keeps the
current generation and the newest previous one and deletes older directories
whole; entries inside a generation are never edited individually, because JAX
owns that layout. The previous generation is kept so a driver bump can be
rolled back without losing a week of compiles.

Every failure mode degrades to "collect cold": an unset variable, an unwritable
or missing mount, or an unreadable volume prints one diagnostic line and the
run continues. The cache is infrastructure, never a reason to lose a night.

### Why no volume is configured today

Measured on run 42464908517: across the 168 `jax-*` rows the recorded
`compile_time_ms` totals 11.0 s, and **only two rows reach the 1 s floor** that
`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` enforces (a 1,989 ms tensor-network
tree compile and a 1,284 ms QR JVP). Every other row is either an eager path with
no XLA compile or a sub-second single-op compile, so the persistence window can
save at most ~3.3 s of a ~1,400 s pod (about $0.002 per run). The volume cost
$1.4/month and pinned every pod to one data center, which already cost two runs
when A100 capacity left that region. The volume and the repository variable were
therefore removed on 2026-11-08; the code path above stays, opt-in and cold by
default, so re-enabling it is one variable away if the floor is ever lowered.

### What the recorded numbers mean

`timing.first_call_ms` is the first warmup — the explicit compilation plus the
first executed call — measured outside every timed sample. The timed region is
unchanged: `warmups`, `timed_runs`, the sampling loop and `rtol`/`atol` do not
move, so **a cache hit cannot show up in `median_ms`**, and it should not: the
steady state never contained the compile.

`cache_bytes_delta` and `cache_entries_delta` in `execution.notes` are the
cache directory's growth around the case. They are evidence, not proof:
positive means this case added entries, zero is *consistent with* a hit and is
not a claim of one. The only unambiguous hit signal is JAX's own
`Persistent compilation cache hit` log line, which is not part of the result
bundle.

`timing.verification_time_ms` is the untimed work that proves the numbers:
building the CPU reference and comparing the output against it, summed over the
case's checks. Every collector that verifies numerically records it: the routine
framework arms, the public API families (the Python arms also keep the wider
`overhead` split, and `normalize_gpu_records` carries both the field and that
split into the published record), the linalg AD arms, and the separate Rust AD
verification pass. A collector that runs no CPU reference leaves it empty rather
than recording a zero. It is recorded because it dominates the pod's wall clock: on
`gpu/dense` with `jax-cuda`, 7 cases spent 8.8 s building references, 5.7 s
comparing, 2.1 s compiling and 0.6 s measuring — so the measurement was 3% of
the work the pod paid for. The field never moves the operation clocks.

### Measured behavior (CPU JAX)

The mechanism was verified end to end on CPU JAX 0.10.1 with separate processes
sharing one cache directory, because the pod budget does not allow a GPU probe:

| Case | `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` | First process | Second process | Directory |
| --- | --- | --- | --- | --- |
| 64² f64 SVD | `0` | 1963 ms | 83 ms | 3 → 4 files, 18.9 kB → 23.9 kB |
| 512² f64 SVD | `0` | 249 ms | 158 ms | 3 → 4 files, 22.4 kB → 27.6 kB |
| 512² f64 SVD | `1` (production) | 315 ms | — | unchanged: no entry |

The second process logged `Persistent compilation cache hit for 'jit_svd_of'`,
so these are hits, not inference from a stopwatch. Two honest limits:

* **The 1 s floor is not exercised by CPU SVD.** XLA's CPU backend compiles the
  SVD program in ~0.2 s even at 2048², so with the production floor those cases
  are never stored. A synthetic 700-step chained program (6.5 s compile) does
  cross the floor on CPU: it produced one 236 kB entry, and later processes
  logged the cache hit. Whether the CUDA backend's 4096-AD compiles clear 1 s
  the way the issue expects is **not verified here** — that needs one GPU night
  and the `first_call_ms` columns of two consecutive runs.
* A cache hit does not make `lower().compile()` instant: tracing and HLO
  construction are not cached. On the 700-step program the hit took ~4.7 s
  against ~6.7 s cold, which is why `first_call_ms` — not `compile_time_ms`
  alone — is the number to compare across weeks.

## REST v2 integration

All RunPod operational calls use `https://api.runpod.io/v2`. The migration follows
[the official guide](https://docs.runpod.io/api-reference-v2/migrate-from-v1)
and [the published OpenAPI](https://api.runpod.io/v2/openapi.json).

| Purpose | REST v2 API and fields |
| --- | --- |
| Create | `POST /v2/pods`: `image`, `cloud`, `gpu.id/count/minCudaVersion`, `disk`, `mounts.persistent` (or `mounts.network` when a cache volume is configured), `entrypoint`, `cmd`, `env` |
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
and verifies the artifact before sampling. The binaries the pod requires come
from the `driver:` of each selected suite (`scripts/ci/gpu_binaries.py`), plus
the routine control lane's `benchmark_gpu_perf_issues` runner for the unfiltered
weekly command, which collects that lane after the routine set. A suite or lane
whose binary did not arrive fails the transfer step by name instead of becoming
`missing` rows later. Missing, stale, or modified artifacts fail rather than
silently rebuilding on a paid GPU.

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

Leave `backends` empty to collect the routine manifest's own arm set, which
includes the `jax-cuda` control. Naming backends is a probe: it narrows the
arms, and a run that drops the JAX control is not the weekly matrix even when
every suite completes (#42473121688 dispatched the old three-arm default and
lost `jax-cuda` on all fourteen suites).

That run also shows why the bundle cannot detect it alone: the routine selection
writes the expected rows from the arms it was given, so a narrowed run validates
against itself. `prepare_result_bundle` therefore compares each suite's plan with
`benchmarks/gpu/manifests/routine.yaml` and records `narrowed_routine_arms`, and
`ingest_benchmark_results` refuses to publish routine pages when that list is
non-empty. The raw artifact is still uploaded and archived; only the published
report is left on the last complete measurement.

For hosted preparation/cache diagnostics without renting a GPU, dispatch with
`-f prepare_only=true`. This still prepares and verifies both artifacts, but
skips provisioning and measurements. It produces no new benchmark receipt.

`suite=all` runs the standard `run_gpu_suite.sh` suites sequentially: dense,
einsum, sparse, and tensornetwork. A specific suite YAML or comma-separated list
can be supplied instead. Standalone GPU runners such as permutation and linalg
AD latency are not included in `all`.

Download the `runpod-gpu-benchmark-<run_id>-attempt<attempt>` artifact
(legacy bundles use the run-ID-only name) for raw measurements under
`data/results/nvidia-gpu/` and reports under `result/nvidia-gpu/`.

The bundle leaves out `fixtures/` and `cache/`: they are inputs and caches
regenerated from the declared seeds, and a weekly run produces about 4.4 GB of
them against about 20 MB of results. The pod would otherwise spend roughly
150 seconds of its billed lifetime uploading them and GitHub would store 1.76 GB
per run. Each suite's `excluded_derived` entry in `publication.json` records how
many files were left out, their total size and a digest of their names and
sizes, so a reproduction can tell whether the assets changed. The workflow
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

## Solve/QR readback collection (#247)

Use the maintained RunPod execution environment for the standalone readback
cases (an alternative to local CUDA-devcontainer collection):

```bash
gh workflow run benchmark-runpod-gpu.yml --repo tensor4all/tenferro-benchmark \
  -f suite=perf-issues-readback -f gpu_type=a100 -f keep_failed_pod=false
```

The hosted job rebuilds `benchmark_gpu_perf_issues` and records its source
commits, digest and compiler version. The A100 verifies that build before use;
no compiler or Python installation is needed on the paid runner. The runner
uses the prepared CUDA runtime and locked PyTorch environment, records
`BENCHMARK_EXECUTION_ENVIRONMENT=runpod`, and leaves the 300 ms clock ramp enabled.

`scripts/ci/run_gpu_readback.sh` selects all 16 `linalg_readback_latency` rows:
solve/QR at n=2,4,8,32, with tenferro and PyTorch references. It fixes physical GPU
0 and CPU thread environment to 1, verifies every row without sampling, then
collects standard-effort timings sequentially on the same pod. A missing,
unsupported or failed correctness row stops before timing. Setup and downloads
remain outside the timed operation, with device synchronization at batch edges.
`backends` does not filter this selection: both reference arms are required.

Download `runpod-gpu-readback-<run_id>-attempt<attempt>` for the verification
and measurement raw runs, metadata, commands, and generated report at
`result/nvidia-gpu/gpu/perf_issues.md`. This filtered standalone report is reviewed
and committed through a PR; it does not carry the standard-suite publication
receipt. `ingest-results.yml` skips runs titled `RunPod readback (manual report)`
(and the purpose marker that follows that title) so they cannot automatically
replace the complete performance-issues report.
The existing cleanup job and pod lifetime watchdog apply to this selection too.
