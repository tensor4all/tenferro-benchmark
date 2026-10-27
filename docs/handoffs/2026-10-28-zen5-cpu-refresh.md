# Handoff: zen5-cpu refresh at the Team-engine tenferro-rs revision

Written 2026-10-28. Nothing here has been timed: the scaffolding was prepared while
another build was running on the host, and no benchmark was run. The measurement is a
separate step on a quiet machine.

## What is ready

- New CPU profile `zen5-cpu` (AMD Ryzen AI 9 HX 470, 12 cores, SMT2, two L3 domains:
  cores 0-3 with siblings 12-15, cores 4-11 with siblings 16-23). It replaces `amd-cpu`
  (an EPYC 7713P) as the latest CPU generation; the break in the series is accepted.
  `amd-cpu`, `linux-cpu`, `mac-cpu` and `mac-gpu` data stays in `result/` and git history
  but is no longer generated into, or linked from, the Pages site and README.
- Rows 1, 4 and 8 threads, **pinned by the runner**. `scripts/cpu_pinning.sh` holds the
  table; `configure_cpu_thread_env` (`scripts/thread_env.sh`), which every suite script
  already calls once per thread row, runs `taskset -pc <mask> $$` so the script and every
  child (cargo, the Rust binaries, Python, Julia) inherit the mask.

  | threads | measured `taskset -c` | observer `taskset -c` |
  |---|---|---|
  | 1 | `0` | `4-11,16-23` |
  | 4 | `0-3` | `4-11,16-23` |
  | 8 | `4-11` | `0-3,12-15` |

  Any other thread count is an error on this profile. `BENCHMARK_CPU_AFFINITY=<mask>`
  overrides the measured mask for diagnostics only (recorded as `override_of`).
- `run.yaml` `environment.affinity` records: `pinning` (`taskset`/`none`), the requested and
  the actual measured CPUs (`os.sched_getaffinity`), the observer CPUs, the thread budget,
  the L3 domains (from sysfs), the CPU boost state (`/sys/devices/system/cpu/cpufreq/boost`,
  `enabled`/`disabled`), the scaling governor and driver. The report's "CPU Information"
  section also lists the L3 domains and boost state. Read them after the run; do not
  publish a row whose `measured_cpus` differs from `requested_measured_cpus`.
- `scripts/benchmark_host_idle.sh` (see "Idle guard" below).
- Landing page, per-suite pages and a per-commit `history.html`, generated in CI
  (`publish-results.yml`, `fetch-depth: 0`), latest generation only (`zen5-cpu`,
  `nvidia-gpu`). Until the first `result/zen5-cpu/` report lands, the CPU card says
  "no results published yet" and links nothing.
- The harness already compiles against the Team-engine branch: `cargo check` of all
  binaries with `--features native` and with `--features blas-mkl` passed against
  `tenferro-rs` branch `feat/engine-threadteam-team` (head `0442b87c`, plus the local
  uncommitted work in that worktree). No harness source change was needed. Re-check at the
  final rev (step 2).

## 0. Before anything: a quiet host

1. One measurement at a time; nothing else compiling, testing or benchmarking, on the host
   or in a container. On the host: `ps -axo pid,etime,args | grep -E 'rustc|cargo|nextest|benchmark_|python.*benchmark'`
   must be empty and `uptime` load near 0. (A `cargo nextest` run was active when this was
   prepared.)
2. Move the observer off the measured cores **first, in its own command**, and verify it.
   For this agent session: `taskset -apc 4-11,16-23 "$(pgrep -x claude)"` for the 1T and
   4T rows, then check with `taskset -pc "$(pgrep -x claude)"`. Start the run from a *new*
   shell so it inherits the mask. A process already forked keeps its old mask.
3. The 8T row needs the other observer set (`0-3,12-15`), so a single multi-row invocation
   cannot satisfy the table by itself:
   - On the host (no container), set `BENCHMARK_OBSERVER_FOLLOW=1`: at every row the
     runner moves its own ancestors (shell, multiplexer, the agent) to the row's observer
     set. It reaches only processes in its PID namespace.
   - In the devcontainer it cannot reach a host agent. Leave the agent idle (waiting, not
     polling) and move it by hand when the log shows `BENCHMARK_MEASURED_CPUS=4-11`:
     `taskset -apc 0-3,12-15 "$(pgrep -x claude)"`. An idle waiting agent costs about
     nothing, but the move is what the protocol prescribes.

## 1. Check the checkout

```bash
cd /home/shinaoka/projects/tensor4all/tenferro-benchmark
git switch main && git pull --ff-only          # the merged scaffolding PR
git submodule update --init
git status --short                              # must be clean
export RUSTC_WRAPPER=                           # the kache wrapper breaks builds; keep it for build AND run
```

Use the same `RUSTC_WRAPPER` value for the pre-build and the run so cargo reuses the
artifacts.

## 2. Move `extern/tenferro-rs` to the rev

`extern/` is git-ignored: the tenferro-rs checkout is not a submodule, and a PR cannot
record its rev. The rev that was measured is recorded in each `run.yaml`
(`tenferro_rs.commit`) and in the summaries. Ask for the exact merged commit sha `<REV>`.

```bash
TENFERRO_UPDATE=1 TENFERRO_REF=<REV> \
STRIDED_RS_UPDATE=1 STRIDED_RS_REF=<STRIDED_REV> \
  ./scripts/setup_extern_deps.sh
git -C extern/tenferro-rs rev-parse HEAD        # must equal <REV>
git -C extern/tenferro-rs status --short        # must be clean (tenferro_rs.dirty=false)
```

- If `extern/tenferro-rs` does not exist, the script clones it, then checks out `<REV>`
  detached. If it exists it must be clean for the move; `TENFERRO_UPDATE=1` fetches and
  `git checkout --detach`es the ref.
- `STRIDED_REV`: the strided-rs commit the tenferro rev builds against (`main`, 0.8.0, is
  `c3e1958` at the time of writing; CI pins `d24474d`, which is older).
- Do not set `SETUP_EXTERN_MIGRATE_SIBLINGS=1` here: it would `mv` `../tenferro-rs`.
- Do not use `git pull` in `extern/tenferro-rs` for this measurement: `AGENTS.md` says to
  keep a pinned/detached checkout.
- Check it builds before timing (not a timing run):
  `taskset -c 4-11,16-23 cargo check --no-default-features --features blas-mkl --bins`
  (and `--features native`).
- Optional, in the results PR: bump `TENFERRO_REF` (twice) and `STRIDED_RS_REF` in
  `.github/workflows/ci.yml` to the same revs so CI validates what was measured.

## 3. Build everything before timing

A build started inside a pinned row would compile on the measured cores (1T: a single
core). Build all binaries on the observer set first:

```bash
F=blas-mkl                                     # see "Feature set" below
export MKLROOT=/opt/intel/oneapi/mkl/latest     # devcontainer: already exported
taskset -c 4-11,16-23 cargo build --release --no-default-features --features "$F" --bins
taskset -c 4-11,16-23 cargo build --release --no-default-features --features "$F,strided-rs,hptt" --bin benchmark_permutation
uv sync --frozen || uv sync                     # Python venv (uv.lock)
```

## 4. Measure: one invocation, rows pinned by the runner

Per `AGENTS.md`, Linux CPU collection runs inside the devcontainer, with the default
(MKL) image so tenferro `blas-mkl` matches the PyTorch wheel's MKL provider. This is the
same recipe as the nightly (`benchmark-nightly-linux-cpu.yml`) with the profile changed,
three thread rows, and the idle guard left **on** (the nightly sets
`BENCHMARK_ALLOW_BUSY_HOST=1`; do not):

```bash
npx --yes @devcontainers/cli up --workspace-folder . --remove-existing-container
npx --yes @devcontainers/cli exec --workspace-folder . bash -lc '
  set -euo pipefail
  test -d "$MKLROOT"
  export BENCHMARK_TARGET_PROFILE=zen5-cpu
  export TENFERRO_CPU_FEATURES=blas-mkl PUBLICATION_GATE_FEATURES=blas-mkl
  export PUBLICATION_GATE_PROFILE=full          # full publication profile (3 warmups, 15 runs)
  export PERMUTATION_EXTRA_FEATURES=hptt
  uv sync --frozen || uv sync
  ./scripts/run_all.sh 1 4 8'
```

`run_all.sh 1 4 8` runs the main einsum / cpu_ops / linalg JVP/VJP path once per thread
count, then fft, public_api, small_work, session_matrix, overhead_smoke (1 thread) and
permutation over the same list, all sequentially. It must be one invocation: the per-row
reports are merged from the runs of one invocation, so separate `run_all.sh 1`,
`run_all.sh 4`, `run_all.sh 8` calls would each overwrite the previous reports.

Host-native alternative (no container; allowed only if the maintainer agrees, because
`AGENTS.md` says Linux CPU runs in the devcontainer): the same body with `MKLROOT`
set to `/opt/intel/oneapi/mkl/latest` and `BENCHMARK_OBSERVER_FOLLOW=1`, run from a shell
already moved to `4-11,16-23`.

What the runner executes for each row, for reference (never run these by hand for the
published numbers; the runner does it and records it):

```bash
taskset -c 0     <1T row>      # observer 4-11,16-23
taskset -c 0-3   <4T row>      # observer 4-11,16-23
taskset -c 4-11  <8T row>      # observer 0-3,12-15
```

`cpu/perf_issues` and the single-case probes are separate entry points and take an
explicit thread list, for example
`BENCHMARK_TARGET_PROFILE=zen5-cpu TENFERRO_CPU_FEATURES=blas-mkl ./scripts/run_perf_issues.sh 1 4 8`.
Run them after `run_all.sh`, never concurrently.

### Feature set (a decision to confirm)

The Team-engine change is to tenferro's own (`native`) engine; the `blas-mkl` path uses
cpueinsum/tlinalg 0.7 over MKL. The standing repository rule (MKL, matching PyTorch's
wheel) is used above. If the maintainer wants the `native` engine measured, use
`TENFERRO_CPU_FEATURES=native` (no MKL needed), but it writes the same
`result/zen5-cpu/...` paths, so run only one feature set per result generation and say
which in the PR. Both builds were checked to compile.

### Expected duration

Not measured here. The nightly (`benchmark-nightly-linux-cpu.yml`, `run_all.sh 1 4` on a
hosted 4-core runner, including devcontainer image build) took 72-103 minutes in the last
three successful runs. Budget about 3-4 hours on this host: a cold `cargo` build (10-30
minutes), a third thread row, the `full` public-API profile, and a build of the
devcontainer image. Treat this as an estimate; read the elapsed times from the first rows
and adjust.

## Idle guard (`scripts/benchmark_host_idle.sh`)

- It refuses to start timing while a `rustc`, `clippy`, `cargo test|nextest|bench|clippy`,
  `premerge_gate.sh`, `metamorphic_equivalence.sh`, `target/{release,debug}/benchmark_*`
  or `python ... benchmark_*.py` process is listed. It matches process names only.
- Inside the devcontainer `ps` sees only the container's PID namespace, so a build or test
  on the host is invisible to that scan. For pinned rows the guard therefore also reads
  `/proc/stat` (host-wide) for the measured CPUs of the row and refuses when they were more
  than 5% busy over a 1 s window (`BENCHMARK_IDLE_MAX_BUSY`, `BENCHMARK_IDLE_WINDOW_SECONDS`).
  It only knows the mask once the row's `configure_cpu_thread_env` ran, so the very first
  call of a script (before its first row) still uses the process scan alone: do the host
  check in step 0 yourself.
- `BENCHMARK_ALLOW_BUSY_HOST=1` disables both checks: diagnostics only, never for
  publication. If the guard rejects the run, wait and rerun.
- When this was prepared, cores 0-3 were 64% busy (another build) and 4-11 idle.

## 5. After the run

1. Inspect `data/results/zen5-cpu/cpu/einsum/<timestamp>/run.yaml` (and the `run_t1/4/8`
   files): `environment.affinity.pinning: taskset`, `measured_cpus` equals the table for
   the row, `observer_cpus`, `l3_domains`, `cpu_boost`, `tenferro_rs.commit` = `<REV>`,
   `tenferro_rs.dirty: false`, BLAS provider `mkl`. Check each report names the new
   timestamp, includes all three thread counts, and has no duplicate rows.
2. Do not hand-edit timing values.

## 6. Publish (PR flow)

`main` requires status checks, so results land through a PR (the ingestion bot does the
same for hosted runs, squash-merging after the checks pass).

```bash
git switch -c results/zen5-cpu-<YYYYMMDD>
git add result/zen5-cpu                    # reports + .summary.json; data/results is git-ignored
git commit -m "Publish zen5-cpu results at tenferro-rs <REV7>"
git push -u origin HEAD
gh pr create --title "Publish zen5-cpu results at tenferro-rs <REV7>" --body-file <body>
gh pr comment <N> --body "<exact commands: the env assignments, thread rows, taskset masks, observer masks, REV>"
```

`AGENTS.md` requires that PR comment (exact commands, environment variables, thread
counts). Wait for `ci` (`validate`) to finish, then merge. On merge, `publish-results.yml`
(trigger: push to `main` touching `result/**`, plus daily 21:00 UTC and manual dispatch)
rebuilds Pages: the CPU card appears and `history.html` gets a new entry listing the
tenferro-rs revision. Verify <https://tensor4all.org/tenferro-benchmark/> and
`/history.html` after the deploy. Smoke or partial runs overwrite `result/`: restore with
`git checkout -- result/` before committing anything you do not intend to publish.

## GPU: `benchmark-runpod-gpu.yml`

Triggers: weekly schedule (Saturday 08:00, 14:00, 20:00 and Sunday 08:00 UTC; the first
success of the week wins and later slots skip) and `workflow_dispatch`. A manual dispatch
always proceeds past the weekly gate.

Inputs (all optional in the CLI; defaults in parentheses):

| input | default | meaning |
|---|---|---|
| `suite` | `all` | a GPU suite YAML, `all`, or `perf-issues-readback` (manual report; not ingested) |
| `backends` | `tenferro-cuda-trace,tenferro-cuda-eager,pytorch-cuda` | comma-separated GPU backends |
| `prepare_only` | `false` | build binaries and runtime only, rent no GPU |
| `keep_failed_pod` | `false` | keep a failed pod for debugging (billing continues) |
| `gpu_type` | `a100` | RunPod GPU id or tier; `a100` = any in-stock A100, the standard `nvidia-gpu` hardware |

```bash
# rehearsal: builds against current tenferro-rs main, rents nothing
gh workflow run benchmark-runpod-gpu.yml -R tensor4all/tenferro-benchmark --ref main -f prepare_only=true
# the real run, after the tenferro-rs Team/0.7 PR has merged
gh workflow run benchmark-runpod-gpu.yml -R tensor4all/tenferro-benchmark --ref main \
  -f suite=all -f gpu_type=a100
gh run list -R tensor4all/tenferro-benchmark --workflow benchmark-runpod-gpu.yml --limit 3
gh run watch <run-id> -R tensor4all/tenferro-benchmark
```

Secrets and approval: it needs the repository secrets `RUNPOD_API_KEY`, `GH_APP_ID` and
`GH_APP_PRIVATE_KEY` (all three are set), and the org runner group `4`
(`runpod-tenferro`). No GitHub environment protection and no manual approval gate is
configured on this workflow (the only environment in the repository is `github-pages`).
Running it spends money (a rented A100 for up to about 55 minutes, hard-capped by the pod
itself; the job timeouts are 60 min build, 42 min benchmark), so confirm with the
maintainer before dispatching.

Revision: there is no `tenferro_ref` input. The job builds the **current `main` head of
tenferro-rs** (`setup_extern_deps.sh` clones `main`, then `git pull --ff-only`), records the
commit, and checks the runner has exactly that commit. So dispatch it only after the
tenferro-rs PR is merged, and confirm `tenferro_rs.commit` in the published summaries.

Publication: no manual step. When the run succeeds, `ingest-results.yml` (`workflow_run`)
validates the bundle, opens a result PR as the `tensor4all-ai-bot` GitHub App, waits for the
required checks and squash-merges it; that push triggers the Pages deploy. To re-ingest or
validate by hand: `gh workflow run ingest-results.yml -f run_id=<id> [-f validate_only=true]`.
Runs titled "RunPod readback (manual report)" are not ingested.
