"""Decide whether a scheduled weekly RunPod GPU run still needs to collect.

The weekly schedule fires several retry slots across the weekend. The first
slot that completes successfully collects and commits that week's results;
later slots see the successful run and skip before renting a GPU. Manual
dispatches always proceed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone

# Saturday 00:00 UTC opens the weekly window; the retry slots fall inside it.
WEEK_START_WEEKDAY = 5


def week_start(now: datetime) -> datetime:
    midnight = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight - timedelta(days=(midnight.weekday() - WEEK_START_WEEKDAY) % 7)


def already_collected(runs: list[dict], now: datetime, current_run_id: str) -> bool:
    start = week_start(now)
    for run in runs:
        if str(run['databaseId']) == current_run_id:
            continue
        created = datetime.fromisoformat(run['createdAt'].replace('Z', '+00:00'))
        if created >= start:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--event', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    proceed = True
    if args.event == 'schedule':
        runs = json.loads(subprocess.check_output([
            'gh', 'run', 'list', '--repo', args.repository, '--workflow', 'benchmark-runpod-gpu.yml',
            '--branch', 'main', '--event', 'schedule', '--status', 'success', '--limit', '20',
            '--json', 'databaseId,createdAt',
        ], text=True))
        proceed = not already_collected(runs, datetime.now(timezone.utc), args.run_id)
        print('This week already has successful GPU results; skipping.' if not proceed
              else 'No successful GPU run this week yet; collecting.')
    output = os.environ.get('GITHUB_OUTPUT')
    if output:
        with open(output, 'a', encoding='utf-8') as handle:
            handle.write(f'proceed={str(proceed).lower()}\n')


if __name__ == '__main__':
    main()
