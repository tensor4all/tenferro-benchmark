#!/usr/bin/env python3
"""Compare one accepted GPU run with the previous ISO week's accepted run (#420).

    python scripts/ci/gpu_bench_results.py --repo . --profile nvidia-gpu --suite gpu/dense

The baseline is selected by week number: the newest accepted run of the same
suite in the ISO week before the candidate's.  It is never "the last two
accepted runs", and a missing previous week says ``no baseline`` instead of
borrowing a neighbouring run.  There is no override for the baseline: a pair
outside that policy is not what this tool reports.

Two runs are only comparable when the ``tenferro_commit`` changed, both were
collected on the same ``hardware.host``, they fall in different ISO weeks, and
they record the same unit.  That gate is applied to the pair before any row is
computed; when it fires, the report names the mechanisms, prints no table, and
exits 1.  When the table is printed, the two runs' commit, host, week and arm
set are stated above it, so the reader never reconstructs which measurements
were compared.

A run is accepted when the data says so: the ingestion driver commits the
sidecar together with ``<suite>.provenance.json``, whose ``files`` map holds the
sidecar's sha256 (``result_ingestion.plan_updates``).  A receipt that disagrees
belongs to an edit, not to an accepted measurement; a revision without a receipt
is history from before the receipts exist, and otherwise only a commit that
publishes the report with the sidecar.  A committed receipt is a trusted marker
of the driver's decision, not authentication.

Exit status: 0 a comparison table was printed, 1 refused or no baseline, 2 the
history could not be read (git's message on stderr) or the threshold is invalid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


class GitError(RuntimeError):
    """A git command failed; the message is git's own diagnostic."""


MAX_THRESHOLD = 1_000_000


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise GitError(done.stderr.strip() or f'git {args[0]} exited {done.returncode}')
    return done.stdout


def git_file(repo: Path, sha: str, relative: str) -> bytes | None:
    """The file's committed content at ``sha``, or None when that commit lacks it.

    Absence is ``rev-parse --verify --quiet`` exiting 1, so an unreadable
    repository is an error and never an absent record.
    """
    name = f'{sha}:{relative}'
    present = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--verify', '--quiet', name],
                             capture_output=True)
    if present.returncode == 1:
        return None
    if present.returncode != 0:
        raise GitError(present.stderr.decode(errors='replace').strip() or f'cannot read {name}')
    done = subprocess.run(['git', '-C', str(repo), 'cat-file', 'blob', name], capture_output=True)
    if done.returncode != 0:
        raise GitError(done.stderr.decode(errors='replace').strip() or f'cannot read {name}')
    return done.stdout


def is_ancestor(repo: Path, older: str, newer: str) -> bool:
    done = subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', older, newer],
                          capture_output=True, text=True)
    if done.returncode in (0, 1):
        return done.returncode == 0
    raise GitError(done.stderr.strip() or 'git merge-base --is-ancestor failed')


def require_full_history(repo: Path) -> None:
    """A shallow clone hides the week a baseline has to come from."""
    if git(repo, 'rev-parse', '--is-shallow-repository').strip() == 'true':
        raise GitError('the repository is a shallow clone, so the previous week may be missing; '
                       'fetch the full history first (git fetch --unshallow)')


def first_receipt(repo: Path, *paths: str) -> str | None:
    """The oldest commit that added one of ``paths``, if this history has one."""
    shas = git(repo, 'log', '--reverse', '--format=%H', '--diff-filter=A', '--', *paths).split()
    return shas[0] if shas else None


def accepted(repo: Path, sha: str, relative: str, receipt_path: str, content: bytes,
             cutoff: str | None) -> bool:
    """Whether this sidecar revision is a published run rather than an edit."""
    receipt = git_file(repo, sha, receipt_path)
    if receipt is not None:
        return json.loads(receipt).get('files', {}).get(relative) == hashlib.sha256(content).hexdigest()
    # No receipt in that commit: acceptance was not recorded there.  Up to the
    # fixed boundary that is the published history, and afterwards only a commit
    # that publishes the run's report with the sidecar.
    if cutoff is not None and is_ancestor(repo, sha, cutoff):
        return True
    # shortcut: a commit that rewrites the report as well is still taken as a
    # publish, so an edit of both files passes; the fix is for every publisher of
    # this sidecar to write the ingestion receipt (#431).
    return bool(git(repo, 'diff-tree', '--root', '--no-commit-id', '--name-only', '-r', '-m', sha, '--',
                    relative.replace('.summary.json', '.md')).strip())


def collected(record: dict) -> datetime:
    """The collection instant in UTC, or ValueError without a recorded timezone.

    Same contract as ``result_ingestion.instant``, which is not imported here
    because that module requires jsonschema for a timestamp: upgrade this to the
    shared helper once the contract lives where a reporting tool can reach it.
    """
    parsed = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('collection timestamp must include a timezone')
    return parsed.astimezone(timezone.utc)


def host(record: dict) -> str | None:
    return (record.get('hardware') or {}).get('host')


def accepted_runs(repo: Path, profile: str, suite: str) -> list[dict]:
    """Every accepted run of one suite, newest measurement first.

    Acceptance is the driver's receipt for that revision, not the commit
    message.  A revision that records no commit, host or collection time cannot
    be placed in a week or gated, so it is not a run this tool can compare.
    """
    relative = f'result/{profile}/{suite}.summary.json'
    receipt_path = f'result/{profile}/{suite}.provenance.json'
    require_full_history(repo)
    # The revisions from before acceptance receipts exist are the published
    # history; the boundary is the first receipt in the history and never moves.
    cutoff = first_receipt(repo, 'result/*/*/*.provenance.json')
    records, seen = [], set()
    # A receipt-only acceptance changes just the provenance path, so both are walked.
    for sha in git(repo, 'log', '--format=%H', '--', relative, receipt_path).split():
        content = git_file(repo, sha, relative)
        if content is None or not accepted(repo, sha, relative, receipt_path, content, cutoff):
            continue
        document = json.loads(content)
        if not (document.get('tenferro_commit') and host(document) and document.get('timestamp')):
            continue
        try:
            collected(document)
        except ValueError:
            continue
        key = (document['tenferro_commit'], host(document), document['timestamp'])
        if key in seen:
            continue
        seen.add(key)
        records.append(document)
    # Newest measurement first, not newest acceptance: re-accepting an older run
    # must not make it the candidate.
    return sorted(records, key=collected, reverse=True)


def iso_week(record: dict) -> str:
    year, week, _ = collected(record).isocalendar()
    return f'{year}-W{week:02d}'


def previous_iso_week(record: dict) -> str:
    year, week, _ = (collected(record) - timedelta(days=7)).isocalendar()
    return f'{year}-W{week:02d}'


def baseline_for(records: list[dict], candidate: dict) -> dict | None:
    wanted = previous_iso_week(candidate)
    return next((item for item in records if iso_week(item) == wanted), None)


def unit(record: dict) -> str:
    """The suite's recorded unit: ``ms`` for most GPU suites, ``ns/op`` for perf_issues."""
    return record.get('unit') or 'ms'


def arm_set(record: dict) -> list[str]:
    return sorted({row['backend'] for row in record['rows']})


def refusals(candidate: dict, baseline: dict) -> list[str]:
    """The comparability mechanisms that fired, each named, in a fixed order."""
    out = []
    if candidate['tenferro_commit'] == baseline['tenferro_commit']:
        out.append(f"same tenferro_commit {candidate['tenferro_commit'][:7]}: no code change to compare")
    if host(candidate) != host(baseline):
        out.append(f'different host: {host(baseline)} vs {host(candidate)}: the comparison is across machines')
    if iso_week(candidate) == iso_week(baseline):
        out.append(f'same ISO week {iso_week(candidate)}: not a week-over-week comparison')
    if unit(candidate) != unit(baseline):
        out.append(f'different unit: {unit(baseline)} vs {unit(candidate)}: '
                   f'two units do not make one ratio')
    return out


def slower_rows(baseline: dict, candidate: dict, threshold: float) -> list[dict]:
    def measured(record: dict) -> dict:
        return {(row['case'], row['backend']): row for row in record['rows']
                if row.get('status') == 'ok' and isinstance(row.get('median'), (int, float))
                and row['median'] > 0}

    before, after = measured(baseline), measured(candidate)
    rows = []
    for key in sorted(before.keys() & after.keys()):
        ratio = after[key]['median'] / before[key]['median']
        if ratio >= threshold:
            rows.append({'case': key[0], 'backend': key[1], 'baseline_ms': before[key]['median'],
                         'candidate_ms': after[key]['median'], 'ratio': ratio})
    return sorted(rows, key=lambda row: row['ratio'], reverse=True)


def compare(candidate: dict, baseline: dict | None, threshold: float) -> dict:
    """Gate the pair first: a refused pair never reaches the table."""
    fired = refusals(candidate, baseline) if baseline is not None else []
    rows = slower_rows(baseline, candidate, threshold) if baseline is not None and not fired else []
    return {'candidate': candidate, 'baseline': baseline, 'refusals': fired, 'rows': rows,
            'threshold': threshold}


def identity(record: dict) -> str:
    return (f"tenferro-rs `{record['tenferro_commit'][:7]}`, host `{host(record)}`, "
            f"week `{iso_week(record)}`, arms: {', '.join(arm_set(record))}")


def render(result: dict) -> str:
    candidate, baseline = result['candidate'], result['baseline']
    lines = [f"# Accepted run comparison: {candidate['suite_id']}", '',
             f'- candidate: {identity(candidate)}']
    if baseline is not None:
        lines.append(f'- baseline: {identity(baseline)}')
    if baseline is None:
        lines += ['', f"no baseline: {candidate['suite_id']} has no accepted run in the previous ISO week "
                      f"{previous_iso_week(candidate)}; no neighbouring run is substituted."]
        return '\n'.join(lines) + '\n'
    if result['refusals']:
        lines += ['', 'refused: ' + '; '.join(result['refusals']) + '.']
        return '\n'.join(lines) + '\n'
    measure = unit(candidate)
    lines += ['', f'Rows slower than the baseline by at least {result["threshold"]}x:', '',
              f'| case | backend | baseline {measure} | candidate {measure} | ratio |',
              '|---|---|---|---|---|']
    for row in result['rows']:
        lines.append(f"| {row['case']} | {row['backend']} | {row['baseline_ms']:.4g} "
                     f"| {row['candidate_ms']:.4g} | {row['ratio']:.2f}x |")
    if not result['rows']:
        lines.append('| (none) | | | | |')
    return '\n'.join(lines) + '\n'


def status(result: dict) -> int:
    return 0 if result['baseline'] is not None and not result['refusals'] else 1


def threshold(text: str) -> float:
    """A finite regression factor between 1 and 1000000, the permitted range.

    NaN compares false with every ratio, which would drop every regression and
    still exit 0; a factor below 1x would list rows that got faster, and one
    above the range reports nothing by construction.
    """
    value = float(text)
    if not math.isfinite(value) or not 1 <= value <= MAX_THRESHOLD:
        raise argparse.ArgumentTypeError(
            f'must be a finite factor between 1 and {MAX_THRESHOLD}, not {text}')
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('.'))
    parser.add_argument('--profile', required=True)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--threshold', type=threshold, default=1.2,
                        help=f'report rows at least this factor slower (1 to {MAX_THRESHOLD}; default 1.2)')
    args = parser.parse_args()
    try:
        records = accepted_runs(args.repo, args.profile, args.suite)
    except (GitError, ValueError) as error:
        print(f'gpu_bench_results: {error}', file=sys.stderr)
        return 2
    if not records:
        print(f"no accepted run of {args.profile} {args.suite} records a commit, a host and a "
              f'collection time; nothing to compare')
        return 1
    candidate = records[0]
    result = compare(candidate, baseline_for(records, candidate), args.threshold)
    print(render(result), end='')
    return status(result)


if __name__ == '__main__':
    sys.exit(main())
