"""Validate complete GPU measurements and bundle only this invocation's results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ROUTINE_MANIFEST = ROOT / 'benchmarks/gpu/manifests/routine.yaml'


def routine_suite_files(manifest_path: Path = ROUTINE_MANIFEST) -> list[str]:
    """The suite files the single routine manifest declares, in manifest order."""
    manifest = yaml.safe_load(Path(manifest_path).read_text())
    return [spec['file'] for spec in manifest['suites'].values()]


DEFAULT_SUITES = ','.join(routine_suite_files())


def planned_rows(run_dir: Path, suite_id: str) -> set[tuple[str, str]] | None:
    """The routine selection's expected (problem, backend) rows for one suite.

    ``None`` when the run metadata records no plan at all, so a caller can fall
    back to the suite declaration. Metadata written before this suite's plan was
    per-suite carries the whole campaign's rows; those entries are selected by
    suite id rather than rejected.
    """
    metadata_path = Path(run_dir) / 'run.yaml'
    if not metadata_path.is_file():
        return None
    metadata = yaml.safe_load(metadata_path.read_text()) or {}
    keys = (metadata.get('collection') or {}).get('expected_case_ids')
    if not keys:
        return None
    rows = set()
    for key in keys:
        suite, problem, backend = key.split(':')
        if suite == suite_id:
            rows.add((problem, backend))
    return rows or None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_records(path: Path, suite_id: str, expected: set[tuple[str, str]]) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    measured = set()
    for row in rows:
        key = (row['problem_id'], row['backend'])
        if key in seen or row['suite_id'] != suite_id:
            raise ValueError(f'{suite_id}: duplicate row or incorrect suite: {key}')
        seen.add(key)
        status = row['status']
        if status == 'unsupported':
            if not row.get('execution', {}).get('unsupported_reason'):
                raise ValueError(f'{suite_id}: unexplained unsupported row: {key}')
            continue
        if status != 'ok':
            reason = row.get('execution', {}).get('unsupported_reason')
            raise ValueError(f'{suite_id}: {key}: {status}: {reason}')
        timing = row['timing']
        median = timing.get('median_ms')
        if not isinstance(median, (int, float)) or not math.isfinite(median) or median < 0:
            raise ValueError(f'{suite_id}: invalid timing: {key}')
        if timing.get('timed_runs', 0) < 1:
            raise ValueError(f'{suite_id}: no measured samples: {key}')
        if row.get('verification', {}).get('status') != 'passed':
            raise ValueError(f'{suite_id}: verification did not pass: {key}')
        measured.add(row['backend'])
    if seen != expected:
        raise ValueError(f'{suite_id}: missing rows {expected - seen}; unexpected rows {seen - expected}')
    if not measured:
        raise ValueError(f'{suite_id}: no successful measurements')


def bundle_run(root: Path, timestamp: str, suites: list[Path], backends: list[str], bundle: Path) -> None:
    if not re.fullmatch(r'\d{8}_\d{6}', timestamp):
        raise ValueError('Invalid run timestamp')
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / 'publication.json').unlink(missing_ok=True)
    entries = []
    for suite_file in suites:
        suite = yaml.safe_load((root / suite_file).read_text())
        suite_id = suite['suite_id']
        if not re.fullmatch(r'gpu/[a-z0-9][a-z0-9_.-]*', suite_id):
            raise ValueError('Invalid GPU suite ID')
        run_relative = Path('data/results/nvidia-gpu') / suite_id / timestamp
        run = root / run_relative
        target = bundle / run_relative
        # Preserve diagnostics even if validation below rejects this invocation.
        shutil.copytree(run, target, dirs_exist_ok=True)
        # The routine selection owns which rows a suite must produce. A run that
        # predates it falls back to the suite declaration, where a suite only
        # runs the requested backends it declares (run_gpu_suite.sh).
        expected = planned_rows(run, suite_id)
        if expected is None:
            declared = suite.get('backends', backends)
            expected = {(p['id'], b) for p in suite['problems'] for b in backends if b in declared}
        validate_records(run / 'records.jsonl', suite_id, expected)
        report_relative = Path('result/nvidia-gpu') / f'{suite_id}.md'
        report = bundle / report_relative
        report.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / 'report.md', report)
        # Dashboard summary sidecar.  The Pages collector only lets a bundle
        # carrying it override the repository report, so older-harness runs
        # without a summary never replace maintained results.
        run_summary = run / 'report.summary.json'
        if not run_summary.exists():
            raise ValueError(f'{suite_id}: report.summary.json is missing from the run')
        summary_relative = Path('result/nvidia-gpu') / f'{suite_id}.summary.json'
        summary = bundle / summary_relative
        shutil.copy2(run_summary, summary)
        files = [report, summary, *sorted(p for p in target.rglob('*') if p.is_file())]
        entries.append({'suite_id': suite_id, 'expected_rows': sorted(expected),
                        'files': {str(p.relative_to(bundle)): digest(p) for p in files}})
    if not entries:
        raise ValueError('No GPU suites selected')
    # Schema 2 adds the required dashboard summary sidecar to each entry; the
    # Pages collector accepts both 1 (pre-summary harness) and 2.
    (bundle / 'publication.json').write_text(json.dumps({
        'schema_version': 2, 'timestamp': timestamp, 'suites': entries,
    }, indent=2) + '\n')
    print(f'Validated {len(entries)} GPU suites for publication')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timestamp', required=True)
    parser.add_argument('--suites', default=DEFAULT_SUITES)
    parser.add_argument('--backends', required=True)
    parser.add_argument('--bundle-dir', type=Path, default=Path('_benchmark_artifact'))
    args = parser.parse_args()
    suites = DEFAULT_SUITES if args.suites == 'all' else args.suites
    bundle_run(Path('.'), args.timestamp, [Path(s.strip()) for s in suites.split(',')],
               [b.strip() for b in args.backends.split(',')], args.bundle_dir)


if __name__ == '__main__':
    main()
