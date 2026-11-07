#!/usr/bin/env python3
"""Package produced outputs after collection, including failed collections.

Also used for recovery of an exact run's diagnostic artifact. Never execute
artifact code or choose stale reports as substitutes for unproduced suites.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import yaml

if __package__:
    from .bundle_cpu_results import package_cpu
    from .cpu_ingestion_adapter import CPU_SUITES
    from .check_gpu_run import DEFAULT_SUITES, planned_rows, routine_arm_gaps
    from .result_ingestion import digest, instant, safe_file, validate_bundle
else:
    from bundle_cpu_results import package_cpu
    from cpu_ingestion_adapter import CPU_SUITES
    from check_gpu_run import DEFAULT_SUITES, planned_rows, routine_arm_gaps
    from result_ingestion import digest, instant, safe_file, validate_bundle

ROOT = Path(__file__).resolve().parents[2]
GPU_BACKENDS = 'tenferro-cuda-trace,tenferro-cuda-eager,pytorch-cuda'
#: Directories inside a run that hold regenerable inputs and caches. A weekly run
#: produces about 4.4 GB of them and about 20 MB of actual results, so shipping
#: them pays twice: the pod uploads them inside its billed lifetime and GitHub
#: stores them for the artifact's retention. Their inventory (names and sizes) is
#: recorded instead, which is what a reproduction needs.
DERIVED_DIRECTORIES = frozenset({'fixtures', 'cache'})


def derived_asset(relative: str) -> bool:
    return any(part in DERIVED_DIRECTORIES for part in Path(relative).parts)


def _inventory(entries) -> tuple[int, int, str]:
    ordered = sorted(entries)
    payload = '\n'.join(f'{name}\t{size}' for name, size in ordered)
    return len(ordered), sum(size for _, size in ordered), hashlib.sha256(payload.encode()).hexdigest()


def current_runs(root, profile, suite, source):
    runs = []
    base = root / f'data/results/{profile}/{suite}'
    for directory in sorted(base.glob('*')):
        try:
            stamp = datetime.strptime(directory.name, '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if (instant(source['run_started_at']).replace(microsecond=0) <= stamp <= instant(source['created_at'])
                and not list(directory.glob('run*.yaml'))):
            raise ValueError(f'{suite}: current collection outputs are missing run metadata')
    for path in sorted(base.glob('*/run*.yaml')):
        metadata = yaml.safe_load(safe_file(root, str(path.relative_to(root))).read_text())
        if instant(source['run_started_at']) <= instant(metadata['timestamp']) <= instant(source['created_at']):
            runs.append(path.parent)
    return sorted(set(runs))


class NoSuiteOutputs(Exception):
    """This suite produced no run directory in the selected attempt."""


def package_gpu_suite(root: Path, profile: str, output: Path, suite_file: str, source: dict,
                      timestamp: str | None, backends: str) -> tuple[str, dict, str, dict]:
    """Copy and declare one GPU suite, raising when this suite cannot package.

    A suite is packaged whole or not at all: its files are copied, its row
    inventory is derived from the plan, and only then is it declared.  A suite
    that raises leaves nothing behind, so the caller can record it as
    unpublished and keep the suites that did produce their outputs.
    """
    declaration = yaml.safe_load(safe_file(ROOT, suite_file.strip()).read_text())
    suite = declaration['suite_id']
    if not suite.startswith('gpu/'):
        raise ValueError('Not a GPU publication suite')
    runs = current_runs(root, profile, suite, source)
    if timestamp and any(run.name != timestamp for run in runs):
        raise ValueError(f'{suite}: inconsistent or ambiguous collection timestamp')
    if not runs:
        raise NoSuiteOutputs(suite)
    if len(runs) != 1:
        raise ValueError(f'{suite}: ambiguous collection outputs in selected run attempt')
    run = runs[0]
    stamp = run.name
    if timestamp and stamp != timestamp:
        raise ValueError('GPU suites have inconsistent collection timestamps')
    run_relative = str(run.relative_to(root))
    target = output / run_relative
    excluded = []
    for path in sorted(run.rglob('*')):
        if path.is_file() or path.is_symlink():
            relative = str(path.relative_to(root))
            safe_file(root, relative)
            if derived_asset(relative):
                excluded.append((str(path.relative_to(run)), path.stat().st_size))
                continue
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    for filename, suffix in (('report.md', '.md'), ('report.summary.json', '.summary.json')):
        original = safe_file(root, run_relative + '/' + filename)
        destination = output / f'result/{profile}/{suite}{suffix}'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    # The routine selection owns which rows a suite must produce; a run that
    # predates it falls back to the suite declaration.
    expected = planned_rows(run, suite)
    if expected is None:
        declared = declaration.get('backends', backends.split(','))
        expected = {(problem['id'], backend) for problem in declaration['problems']
                    for backend in backends.split(',') if backend in declared}
    expected = sorted(expected)
    # Recorded, and refused at ingestion: a run that dropped a routine arm is
    # not the weekly matrix, however complete its own rows look.
    gaps = routine_arm_gaps(run, suite)
    files = [output / f'result/{profile}/{suite}.summary.json',
             output / f'result/{profile}/{suite}.md']
    files += sorted(path for path in target.rglob('*') if path.is_file())
    count, total_bytes, inventory_digest = _inventory(excluded)
    entry = {'suite_id': suite, 'expected_rows': expected,
             'files': {str(path.relative_to(output)): digest(path) for path in files},
             'excluded_derived': {'files': count, 'bytes': total_bytes,
                                  'sha256': inventory_digest}}
    return suite, entry, stamp, gaps


def discard_gpu_suite(output: Path, suite: str, profile: str = 'nvidia-gpu') -> None:
    """Remove a suite's partial copies so the bundle declares only packaged suites."""
    shutil.rmtree(output / 'data/results' / profile / suite, ignore_errors=True)
    for suffix in ('.md', '.summary.json'):
        (output / f'result/{profile}/{suite}{suffix}').unlink(missing_ok=True)


def prepare_bundle(root: Path, profile: str, source: dict, output: Path,
                   *, suites='all', backends=GPU_BACKENDS, timestamp=None):
    """Validate all current outputs; missing suites are recorded, not replaced."""
    if profile != 'nvidia-gpu':
        produced, missing = [], []
        for name in CPU_SUITES:
            raw_suite = 'einsum' if name in ('cpu_ops', 'linalg_jvp_vjp') else name
            if current_runs(root, profile, 'cpu/' + raw_suite, source):
                produced.append(name)
            else:
                missing.append('cpu/' + name)
        if not produced:
            raise ValueError('No CPU collection outputs in the selected run attempt')
        manifest = package_cpu(root, profile, source['head_sha'], output, suites=produced)
        manifest['missing_suites'] = missing
        (output / 'ingestion.json').write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        if suites == 'perf-issues-readback':
            raise ValueError('Manual readback is not a publication collection')
        suite_files = DEFAULT_SUITES if suites == 'all' else suites
        entries, missing, narrowed = [], [], {}
        output.mkdir(parents=True, exist_ok=False)
        for suite_file in suite_files.split(','):
            # One suite that cannot be packaged (or whose own accounting is
            # incomplete) is recorded as unpublished. It never costs the other
            # suites the results they did produce.
            suite_decl = yaml.safe_load(safe_file(ROOT, suite_file.strip()).read_text())
            suite = suite_decl.get('suite_id', suite_file.strip())
            try:
                suite, entry, stamp, gaps = package_gpu_suite(root, profile, output, suite_file, source,
                                                              timestamp, backends)
            except NoSuiteOutputs:
                missing.append(suite)
                continue
            except (ValueError, OSError, KeyError) as error:
                discard_gpu_suite(output, suite)
                missing.append(suite)
                print(f'{suite}: not published: {error}', file=sys.stderr)
                continue
            if gaps:
                narrowed[suite] = gaps
            timestamp = timestamp or stamp
            entries.append(entry)
        if not entries:
            raise ValueError('No GPU collection outputs in the selected run attempt')
        manifest = {'schema_version': 2, 'timestamp': timestamp, 'suites': entries, 'missing_suites': missing}
        if narrowed:
            manifest['narrowed_routine_arms'] = narrowed
        (output / 'publication.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # One suite whose artifacts fail the publication contract must not cost
        # the week the suites that pass it. The failing suite is named at the
        # front of the message, so it can be removed while the rest publishes;
        # a bundle that has nothing left still fails with the original error.
        while len(manifest['suites']) > 1:
            try:
                return validate_bundle(output, profile, source)
            except Exception as error:
                dropped = next((entry['suite_id'] for entry in manifest['suites']
                                if str(error).startswith(entry['suite_id'] + ':')), None)
                if dropped is None:
                    raise
                print(f'{dropped}: not published: {error}', file=sys.stderr)
                manifest['suites'] = [entry for entry in manifest['suites']
                                      if entry['suite_id'] != dropped]
                manifest['missing_suites'] = sorted(set(manifest['missing_suites']) | {dropped})
                discard_gpu_suite(output, dropped)
                (output / 'publication.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # No manifest from a failed validation may authorize a results upload.
    try:
        return validate_bundle(output, profile, source)
    except Exception:
        (output / ('publication.json' if profile == 'nvidia-gpu' else 'ingestion.json')).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['amd-cpu', 'mac-cpu', 'nvidia-gpu'], required=True)
    parser.add_argument('--harness-commit', required=True)
    parser.add_argument('--started-at', required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--suites', default='all')
    parser.add_argument('--backends', default=GPU_BACKENDS)
    parser.add_argument('--timestamp')
    args = parser.parse_args()
    source = {'head_sha': args.harness_commit, 'run_started_at': args.started_at,
              'created_at': datetime.now(timezone.utc).isoformat()}
    validated = prepare_bundle(args.root.resolve(), args.profile, source, args.bundle_dir.resolve(),
                               suites=args.suites, backends=args.backends, timestamp=args.timestamp)
    print(json.dumps({receipt['suite_id']: {'status': receipt['status'],
                      'case_outcomes': receipt['case_outcomes'], 'missing_suites': receipt['missing_suites']}
                      for receipt, _ in validated}, indent=2))


if __name__ == '__main__':
    main()
