# The zen5-cpu self-hosted runner

`benchmark-weekly-zen5-cpu.yml` is the only scheduled job that can refresh the CPU page
(<https://tensor4all.org/tenferro-benchmark/cpu/>): the page is rendered from the
`maintained` profiles in `benchmarks/profiles.yaml`, which is `zen5-cpu` alone, and
`zen5-cpu` is one machine. Before this job the page refreshed only when someone collected
by hand (#468 on 2026-11-12), and it went stale for 11 days while every scheduled job was
green (#515).

## Why a self-hosted runner, and not the nightlies

The other choice in #515 was to add `amd-cpu` / `mac-cpu` to `maintained` so the existing
nightlies appear on the page. It was not taken:

- The nightlies run on GitHub-hosted runners with `BENCHMARK_ALLOW_BUSY_HOST=1`, no pinned
  rows, `PUBLICATION_GATE_PROFILE=quick`, and a VM whose CPU model is not fixed; their own
  workflow calls the results "reference-quality indicators only". `amd-cpu` no longer names
  the EPYC 7713P it was created for. Putting them on the page changes what the page claims.
- `AGENTS.md` links only the latest generation because people mistake old data for current;
  the 2026-10-28 decision replaced `amd-cpu` with `zen5-cpu` on the page.
- `regenerate_report_summaries.py --check` re-derives each maintained page from tracked run
  directories; `amd-cpu` / `mac-cpu` runs are local-only (`.gitignore`), so their pages
  could not be verified without changing the ingestion path as well.

## How a runner reaches the machine

A GitHub runner is pull-based: the runner process on the host polls GitHub over outbound
HTTPS, so the host needs no inbound port or public address.

1. An organisation or repository admin creates a registration token
   (*Settings → Actions → Runners → New self-hosted runner*, or
   `gh api -X POST repos/tensor4all/tenferro-benchmark/actions/runners/registration-token`).
2. On the host, as an unprivileged user, in a directory outside every checkout:

   ```sh
   ./config.sh --url https://github.com/tensor4all/tenferro-benchmark \
     --token <token> --name evo-zen5 --labels zen5-cpu --unattended
   sudo ./svc.sh install && sudo ./svc.sh start
   ```

   The job selects `[self-hosted, linux, x64, zen5-cpu]`; no other runner carries that label.
3. The runner service should start on the observer set:
   `systemctl edit actions.runner.*` with `CPUAffinity=4-11 16-23`. The collection step moves
   the runner processes to each row's observer set through the handshake helper.
4. The operator keeps `cpufreq/boost` at 0 and the governor at `performance`. The job checks
   both and fails if they are not set; it never changes them.

## Safety on a public repository

- The workflow triggers on `schedule` and `workflow_dispatch` only. No `pull_request` job
  targets the `zen5-cpu` label, so code from a fork never runs on the host. Keep it that
  way: a review must reject any workflow that adds `zen5-cpu` to a PR-triggered job.
- *Settings → Actions → General → Fork pull request workflows* should require approval for
  all outside collaborators.
- The result PR is opened with the ingestion App token and is never merged by the job.

## When the host is busy

The harness idle gate refuses a row whose measured CPUs were more than 5% busy. The job
then fails red rather than skipping, so the nightly monitor sees it. Re-dispatch it once
the host is quiet; do not set `BENCHMARK_ALLOW_BUSY_HOST`.
