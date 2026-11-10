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
exits 1.  A row enters a table only when the ingestion driver would accept it
(status ``ok``, a finite non-negative median and IQR, ``verified: true``).  An
unverified row is named under the table instead, and so is a row the other run
did not record as a measurement.  A row that only one of the two runs recorded
at all is named as well, so a case that vanished from a week is not read as no
change.  A record the driver would reject as a whole (a
repeated identity) refuses the comparison by name rather than overwriting one of
the rows: keeping the last one hides a regression.  When the table is printed,
the two runs' commit, host, week and arm set are stated above it, so the reader
never reconstructs which measurements were compared.

A run is accepted when the data says so: the ingestion driver commits the
sidecar together with ``<suite>.provenance.json``, whose ``files`` map holds the
sidecar's sha256 (``result_ingestion.plan_updates``).  A receipt that disagrees
belongs to an edit, not to an accepted measurement; a revision without a receipt
is history when no receipt exists at or before it, and otherwise only a commit
that publishes the report with the sidecar.  A committed receipt is a trusted
marker of the driver's decision, not authentication.

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
RECEIPTS = 'result/*/*/*.provenance.json'


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


def require_full_history(repo: Path) -> None:
    """A shallow clone hides the week a baseline has to come from."""
    if git(repo, 'rev-parse', '--is-shallow-repository').strip() == 'true':
        raise GitError('the repository is a shallow clone, so the previous week may be missing; '
                       'fetch the full history first (git fetch --unshallow)')


def receipt_history(repo: Path, sha: str) -> bool:
    """Whether an acceptance receipt exists at or before ``sha`` in its own history.

    That is a property of the revision's ancestry, which never changes, so no
    later merge can reclassify a revision that was already read.  A single
    boundary commit picked from the history does move: merging an unrelated
    branch that carries an earlier-dated receipt makes it the boundary and drops
    the published runs that are not its ancestors.  ``--full-history`` because
    the default path simplification can hide a receipt a merge did not keep.
    """
    return bool(git(repo, 'log', '-1', '--full-history', '--format=%H', sha, '--',
                    RECEIPTS).strip())


def accepted(repo: Path, sha: str, relative: str, receipt_path: str, content: bytes) -> bool:
    """Whether this sidecar revision is a published run rather than an edit."""
    receipt = git_file(repo, sha, receipt_path)
    if receipt is not None:
        return json.loads(receipt).get('files', {}).get(relative) == hashlib.sha256(content).hexdigest()
    # No receipt in that commit.  A revision whose own history records no receipt
    # at all predates the driver's bookkeeping and is the published history;
    # afterwards only a commit that publishes the report with the sidecar.
    if not receipt_history(repo, sha):
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
    records, seen = [], set()
    # --full-history: Git's default path simplification drops a commit whose
    # sidecar a merge did not keep, which would hide a real run and report
    # "no baseline".  A receipt-only acceptance changes just the receipt path, so
    # both paths are walked.
    for sha in git(repo, 'log', '--full-history', '--format=%H', '--',
                   relative, receipt_path).split():
        content = git_file(repo, sha, relative)
        if content is None or not accepted(repo, sha, relative, receipt_path, content):
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


def row_key(row: dict) -> tuple[str, str, str]:
    """The identity the ingestion driver enforces (``validate_summary``).

    (case, group, backend): (case, backend) collides across groups and one row
    would hide the other.
    """
    return (row['case'], row.get('group', ''), row['backend'])


def name(key: tuple[str, str, str]) -> str:
    return '/'.join(key)


def invalid_rows(record: dict) -> list[str]:
    """Why the ingestion driver would reject this record's rows, if it would.

    ``result_ingestion.validate_summary`` refuses a repeated identity and a
    measurement whose median or IQR is not a finite non-negative number, and it
    refuses the whole summary when it does.  A receipt-free revision never
    passed that gate, so its rows are checked here before any ratio is taken.
    """
    def valid(value) -> bool:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    out, seen = [], set()
    for row in record['rows']:
        key = row_key(row)
        if key in seen:
            out.append(f'{name(key)}: duplicate identity')
            continue
        seen.add(key)
        if row.get('status') != 'ok':
            continue
        if not valid(row.get('median')):
            out.append(f'{name(key)}: median is not a finite non-negative number')
        if row.get('iqr') is not None and not valid(row['iqr']):
            out.append(f'{name(key)}: IQR is not finite and non-negative')
    return out


def slower_rows(baseline: dict, candidate: dict, threshold: float) -> dict:
    """Slower rows, the rows whose ratio cannot be taken, and the ones not compared."""
    def measured(row: dict) -> bool:
        return (row.get('status') == 'ok' and type(row.get('median')) in (int, float)
                and math.isfinite(row['median']) and row['median'] >= 0)

    before = {row_key(row): row for row in baseline['rows']}
    after = {row_key(row): row for row in candidate['rows']}
    rows, undefined, unverified, not_ok, absent = [], [], [], [], []
    for key in sorted(before.keys() | after.keys()):
        left, right = before.get(key), after.get(key)
        # Rows are classified over both records' identities, not their
        # intersection: intersecting first is what let an unverified row with no
        # counterpart vanish without the diagnostic the table owes the reader.
        sides = [side for side, row in (('baseline', left), ('candidate', right))
                 if row is not None and measured(row) and row.get('verified') is not True]
        if sides:
            # ``result_ingestion`` refuses a measurement whose ``verified`` is not
            # true, so a receipt-free record can carry rows the driver rejected:
            # they are named instead of being compared or dropped silently.
            unverified.append({'case': key[0], 'group': key[1], 'backend': key[2],
                               'baseline': (left or {}).get('median'),
                               'candidate': (right or {}).get('median'), 'sides': sides})
            continue
        dropped = [f'{side} status {(row or {}).get("status")}'
                   for side, row, other in (('baseline', left, right), ('candidate', right, left))
                   if row is not None and not measured(row) and other is not None and measured(other)]
        if dropped:
            # The identity is a measurement on one side and a non-`ok` row on the
            # other, so there is no ratio; naming it is the diagnostic.
            not_ok.append({'case': key[0], 'group': key[1], 'backend': key[2], 'notes': dropped})
            continue
        if left is None or right is None:
            # One run has no entry for this identity at all.  A case that an
            # earlier week measured and this week's run did not record is lost
            # coverage: it used to leave the table empty with no diagnostic.
            side, row = ('baseline', left) if right is None else ('candidate', right)
            absent.append({'case': key[0], 'group': key[1], 'backend': key[2],
                           'side': side, 'status': row.get('status'),
                           'median': row.get('median')})
            continue
        if not (measured(left) and measured(right)):
            continue
        row = {'case': key[0], 'group': key[1], 'backend': key[2],
               'baseline': left['median'], 'candidate': right['median']}
        if row['baseline'] == 0:
            # A zero baseline is a legal measurement, and 0 -> positive is not
            # "no change": report the pair instead of dropping it.
            undefined.append(row)
        elif row['candidate'] / row['baseline'] >= threshold:
            rows.append(dict(row, ratio=row['candidate'] / row['baseline']))
    return {'rows': sorted(rows, key=lambda row: row['ratio'], reverse=True),
            'undefined': undefined, 'unverified': unverified, 'not_ok': not_ok, 'absent': absent}


def compare(candidate: dict, baseline: dict | None, threshold: float) -> dict:
    """Gate the pair first: a refused or invalid pair never reaches the table."""
    fired = refusals(candidate, baseline) if baseline is not None else []
    invalid = [f'{side} {why}' for side, record in (('baseline', baseline), ('candidate', candidate))
               if record is not None for why in invalid_rows(record)]
    computed = (slower_rows(baseline, candidate, threshold)
                if baseline is not None and not fired and not invalid
                else {'rows': [], 'undefined': [], 'unverified': [], 'not_ok': [], 'absent': []})
    return {'candidate': candidate, 'baseline': baseline, 'refusals': fired, 'invalid': invalid,
            **computed, 'threshold': threshold}


def identity(record: dict) -> str:
    return (f"tenferro-rs `{record['tenferro_commit'][:7]}`, host `{host(record)}`, "
            f"week `{iso_week(record)}`, arms: {', '.join(arm_set(record))}")


def render(result: dict) -> str:
    candidate, baseline = result['candidate'], result['baseline']
    lines = [f"# Accepted run comparison: {candidate['suite_id']}", '',
             f'- candidate: {identity(candidate)}']
    if baseline is not None:
        lines.append(f'- baseline: {identity(baseline)}')
    # A refused record is named before the missing-baseline message: an invalid
    # candidate is not "no baseline", and the old order let that message swallow
    # the diagnostic when the week had no accepted run.
    if baseline is None and not result['invalid']:
        lines += ['', f"no baseline: {candidate['suite_id']} has no accepted run in the previous ISO week "
                      f"{previous_iso_week(candidate)}; no neighbouring run is substituted."]
        return '\n'.join(lines) + '\n'
    if result['invalid']:
        lines += ['', 'invalid: ' + '; '.join(result['invalid']) + '.']
        if baseline is None:
            lines.append(f"no baseline either: no accepted run in the previous ISO week "
                         f"{previous_iso_week(candidate)}.")
        return '\n'.join(lines) + '\n'
    if result['refusals']:
        lines += ['', 'refused: ' + '; '.join(result['refusals']) + '.']
        return '\n'.join(lines) + '\n'
    measure = unit(candidate)
    lines += ['', f'Rows slower than the baseline by at least {result["threshold"]}x:', '',
              f'| case | group | backend | baseline {measure} | candidate {measure} | ratio |',
              '|---|---|---|---|---|---|']
    for row in result['rows']:
        lines.append(f"| {row['case']} | {row['group']} | {row['backend']} | {row['baseline']:.4g} "
                     f"| {row['candidate']:.4g} | {row['ratio']:.2f}x |")
    if not result['rows']:
        lines.append('| (none) | | | | | |')
    if result['undefined']:
        lines += ['', f'Ratio undefined (the baseline median is 0) for '
                      f'{len(result["undefined"])} row(s):', '',
                  f'| case | group | backend | baseline {measure} | candidate {measure} |',
                  '|---|---|---|---|---|']
        for row in result['undefined']:
            lines.append(f"| {row['case']} | {row['group']} | {row['backend']} "
                         f"| {row['baseline']:.4g} | {row['candidate']:.4g} |")
    if result['unverified']:
        named = '; '.join(f"{row['case']}/{row['group']}/{row['backend']} "
                          f"({' and '.join(row['sides'])} unverified)"
                          for row in result['unverified'])
        lines += ['', f'Skipped {len(result["unverified"])} unverified row(s) (the ingestion driver '
                      f'records only measurements whose `verified` is true): {named}.']
    if result['not_ok']:
        named = '; '.join(f"{row['case']}/{row['group']}/{row['backend']} "
                          f"({' and '.join(row['notes'])})"
                          for row in result['not_ok'])
        lines += ['', f'Not compared: {len(result["not_ok"])} row(s) the other run did not record as '
                      f'a measurement: {named}.']
    if result['absent']:
        named = '; '.join(f"{row['case']}/{row['group']}/{row['backend']} ({row['side']} only, "
                          f"status {row['status']})" for row in result['absent'])
        lines += ['', f'Only one run recorded {len(result["absent"])} row(s), so there is nothing to '
                      f'compare: {named}.']
    return '\n'.join(lines) + '\n'


def status(result: dict) -> int:
    return 0 if (result['baseline'] is not None and not result['refusals']
                 and not result['invalid']) else 1


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
