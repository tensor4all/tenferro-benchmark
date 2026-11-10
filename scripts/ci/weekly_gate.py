"""Decide whether a RunPod GPU run may collect, and report this week's dispatch budget.

The weekly schedule fires several retry slots across the weekend. The first
slot that completes successfully collects and commits that week's results;
later slots see the successful run and skip before renting a GPU. Manual
dispatches proceed within the dispatch budget below.

Every dispatch carries a purpose: `weekly` and `backfill` are attempts to
satisfy the schedule and share the 3-per-week cap, `verify` checks a fix and has
its own 6-per-week cap plus a required `target`. The purpose is recorded in the
run name (`[purpose=... target=...]`), so the counters are read back from the
runs themselves instead of a second store. Only manual dispatches are counted:
a scheduled slot is not a dispatch, and a slot that skipped never rented a GPU.
A week that already spent the budget refuses the scheduled slot too.

Every count is scoped to the branch the run belongs to. A dispatch from a
feature branch is a record of the same workflow in the same window, and it must
not spend main's weekly budget; it still spends its own branch's cap.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone

# Saturday 00:00 UTC opens the weekly window; the retry slots fall inside it.
WEEK_START_WEEKDAY = 5

# The branch a run is dispatched from (the schedule runs on the default branch).
MAIN_BRANCH = 'main'

# A100 hours are the scarce resource. The shared cap stops us retrying a weekly
# collection; `verify` runs are counted separately because batching two fixes
# into one verification is cheap, while a missed week is not.
WEEKLY_CAP = 3
VERIFY_CAP = 6
# Runs the history query returns; reaching it fails closed instead of under-counting.
WINDOW_LIMIT = 100
PURPOSES = ('weekly', 'verify', 'backfill')
TARGET_RE = re.compile(r'[A-Za-z0-9._-]+/[A-Za-z0-9._-]+#[0-9]+')
PURPOSE_RE = re.compile(r'purpose=([\w-]+)')


def week_start(now: datetime) -> datetime:
    midnight = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight - timedelta(days=(midnight.weekday() - WEEK_START_WEEKDAY) % 7)


def normalize_purpose(purpose: str | None) -> str:
    """Unknown or missing purposes count as weekly: the strict direction."""
    return purpose if purpose in PURPOSES else 'weekly'


def purpose_of(run: dict) -> str:
    """The purpose recorded in a run's name, weekly when it carries none."""
    found = PURPOSE_RE.search(run.get('displayTitle') or '')
    return normalize_purpose(found.group(1) if found else '')


def completed(run: dict) -> bool:
    """A run that has a conclusion, or a second attempt, has already run."""
    # A re-run clears the conclusion while it runs, but its first attempt had one, so the
    # attempt count keeps a requeued successor in the count.
    return bool(run.get('conclusion')) or int(run.get('attempt') or 1) > 1


def week_runs(runs: list[dict], now: datetime, current_run_id: str,
              branch: str = MAIN_BRANCH, predecessors_only: bool = True) -> list[dict]:
    """This week's other runs on `branch`.

    `predecessors_only` ignores a successor that is still on its first attempt: it may
    not have rented yet, and counting it would make a dispatch refuse itself because
    someone queued another. A successor that already finished, or that is re-running an
    attempt that finished, may have rented, and the concurrency group does not order runs
    by id, so it is counted. A re-run executes after the runs it has to be counted
    against, so it looks at the whole window instead.
    """
    start = week_start(now)
    return [run for run in runs
            if str(run['databaseId']) != current_run_id
            and run.get('headBranch') == branch
            and (not predecessors_only or int(run['databaseId']) < int(current_run_id)
                 or completed(run))
            and datetime.fromisoformat(run['createdAt'].replace('Z', '+00:00')) >= start]


def count_purposes(runs: list[dict], now: datetime, current_run_id: str,
                   branch: str = MAIN_BRANCH, predecessors_only: bool = True) -> dict[str, int]:
    counts = dict.fromkeys(PURPOSES, 0)
    # Scheduled slots have their own gate: a slot that skipped never rented a GPU,
    # and counting the weekend's four slots would spend the cap on a normal week.
    for run in week_runs(runs, now, current_run_id, branch, predecessors_only):
        if run.get('event') != 'schedule':
            counts[purpose_of(run)] += 1
    return counts


def budget_report(counts: dict[str, int]) -> str:
    """Both counters, in the form the weekly report prints them."""
    return f'weekly {counts["weekly"] + counts["backfill"]}/{WEEKLY_CAP}, verify {counts["verify"]}/{VERIFY_CAP}'


def refusal(purpose: str, target: str, counts: dict[str, int]) -> str | None:
    """Why this run may not rent a GPU, or None when it may."""
    if purpose == 'verify':
        if not TARGET_RE.fullmatch(target):
            return (f'purpose=verify needs target=<repo>#<n>, got {target!r}: '
                    'name the issue or pull request this run verifies.')
        if counts['verify'] >= VERIFY_CAP:
            return (f'verify budget spent this week ({budget_report(counts)}): '
                    'batch the remaining fixes into one run in the next weekly window.')
    elif counts['weekly'] + counts['backfill'] >= WEEKLY_CAP:
        return (f'weekly budget spent this week ({budget_report(counts)}): '
                'restarting a weekly collection counts against it.')
    return None


def truncation_refusal(returned: int) -> str | None:
    """The window has to be complete; a truncated history under-counts the caps."""
    if returned < WINDOW_LIMIT:
        return None
    return (f'gh returned {WINDOW_LIMIT} runs for this week, so the counters may be truncated: '
            'fail closed rather than rent against an unknown budget.')


def rerun_refusal(run_attempt: int) -> str | None:
    """A re-run may not rent: what the attempt it repeats actually did cannot be told.

    A `failure` can follow a successful collection (the pod deletion or an upload
    failing afterwards), and a failed provisioning attempt can leave a pod behind, so no
    conclusion proves that a re-run would not rent and collect a second time. A
    scheduled slot has four chances per weekend and the next slot reads this week's
    result; a retry that needs a GPU now is a dispatch carrying its purpose, which the
    budget counts.
    """
    if run_attempt <= 1:
        return None
    return ('a re-run repeats a run that is already recorded, and a retry that may rent has to be '
            'counted: let the next scheduled slot try, or dispatch a new run with its purpose.')


def resolve_attempts(repository: str, runs: list[dict]) -> list[dict]:
    """Mark the scheduled runs whose earlier attempt succeeded.

    `gh run list` reports only the latest attempt of a run, and refusing a re-run turns
    a slot that collected into a failed run. Without the attempt it repeats, the next
    slot would read an uncollected week and rent again, so ask for the attempts
    themselves. Only a scheduled run past its first attempt that does not already show
    a success needs the query, which is rare.
    """
    for run in runs:
        attempts = int(run.get('attempt') or 1)
        if run.get('event') != 'schedule' or attempts < 2 or run.get('conclusion') == 'success':
            continue
        # `any` stops at the first success: a later attempt that cannot be read must not
        # take an answer that is already decided down with it.
        earlier = (subprocess.check_output([
            'gh', 'api', f'repos/{repository}/actions/runs/{run["databaseId"]}/attempts/{attempt}',
            '--jq', '.conclusion',
        ], text=True).strip() for attempt in range(1, attempts))
        run['prior_success'] = any(conclusion == 'success' for conclusion in earlier)
    return runs


def already_collected(runs: list[dict], now: datetime, current_run_id: str,
                      branch: str = MAIN_BRANCH) -> bool:
    """Whether a scheduled run already provided this week's results.

    Only a *scheduled* success satisfies the week. A `verify` run checks a fix
    and never counts as a collected week, because that reading is the one that
    hides a missed collection.

    The whole window is asked, not the predecessors: the concurrency group
    serializes the weekend slots but does not order them by run id, so a later
    slot can finish the collection first, and skipping only the run ids below
    this one would rent a second pod for a week that is already collected.

    `prior_success` is set by `resolve_attempts()`: a slot that collected keeps
    counting after a re-run of it was refused, because that refusal leaves the run
    with conclusion=failure while its first attempt still provided the results.
    """
    for run in week_runs(runs, now, current_run_id, branch, predecessors_only=False):
        if run.get('event') == 'schedule' and purpose_of(run) != 'verify' \
                and (run.get('conclusion') == 'success' or run.get('prior_success')):
            return True
    return False


def note(text: str) -> None:
    """Append to the step summary, so the counters land in the run record."""
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if path:
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(f'{text}\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--event', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--run-attempt', type=int, default=1, help='github.run_attempt; a re-run is refused')
    parser.add_argument('--branch', default=MAIN_BRANCH,
                        help='the branch this run belongs to; each branch has its own budget')
    parser.add_argument('--purpose', default='', help='weekly|backfill|verify; anything else counts as weekly')
    parser.add_argument('--target', default='', help='<repo>#<n> this run verifies, required for verify')
    args = parser.parse_args()
    purpose = normalize_purpose(args.purpose)
    now = datetime.now(timezone.utc)
    # The created filter bounds the window on the server, and --branch keeps another
    # branch's runs from eating the 100-run limit; WINDOW_LIMIT is the point where
    # the counters stop being trustworthy.
    runs = json.loads(subprocess.check_output([
        'gh', 'run', 'list', '--repo', args.repository, '--workflow', 'benchmark-runpod-gpu.yml',
        '--branch', args.branch,
        '--created', f'>={week_start(now).strftime("%Y-%m-%d")}', '--limit', str(WINDOW_LIMIT),
        '--json', 'databaseId,attempt,createdAt,event,conclusion,displayTitle,headBranch',
    ], text=True))
    runs = resolve_attempts(args.repository, runs)
    predecessors_only = args.run_attempt == 1
    counts = count_purposes(runs, now, args.run_id, args.branch, predecessors_only)
    report = budget_report(counts)
    print(f'Dispatch budget this week: {report}')
    note(f'purpose={purpose} target={args.target or "-"}: {report}')
    refused = (truncation_refusal(len(runs))
               or rerun_refusal(args.run_attempt)
               or refusal(purpose, args.target, counts))
    if refused:
        print(refused)
        note(f'refused: {refused}')
        raise SystemExit(1)
    proceed = True
    if args.event == 'schedule':
        proceed = not already_collected(runs, now, args.run_id, args.branch)
        print('This week already has successful GPU results; skipping.' if not proceed
              else 'No successful GPU run this week yet; collecting.')
    output = os.environ.get('GITHUB_OUTPUT')
    if output:
        with open(output, 'a', encoding='utf-8') as handle:
            handle.write(f'proceed={str(proceed).lower()}\n')


if __name__ == '__main__':
    main()
