#!/usr/bin/env python3
"""Package produced outputs after collection, including failed collections.

Also used for recovery of an exact run's diagnostic artifact. Never execute
artifact code or choose stale reports as substitutes for unproduced suites.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import yaml

if __package__:
    from .bundle_cpu_results import package_cpu
    from .cpu_ingestion_adapter import CPU_SUITES
    from .check_gpu_run import DEFAULT_SUITES
    from .result_ingestion import digest, instant, safe_file, validate_bundle
else:
    from bundle_cpu_results import package_cpu
    from cpu_ingestion_adapter import CPU_SUITES
    from check_gpu_run import DEFAULT_SUITES
    from result_ingestion import digest, instant, safe_file, validate_bundle

ROOT = Path(__file__).resolve().parents[2]
GPU_BACKENDS = 'tenferro-cuda-trace,tenferro-cuda-eager,pytorch-cuda'


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
        entries, missing = [], []
        output.mkdir(parents=True, exist_ok=False)
        for suite_file in suite_files.split(','):
            declaration = yaml.safe_load(safe_file(ROOT, suite_file.strip()).read_text())
            suite = declaration['suite_id']
            if not suite.startswith('gpu/'):
                raise ValueError('Not a GPU publication suite')
            runs = current_runs(root, profile, suite, source)
            if timestamp and any(run.name != timestamp for run in runs):
                raise ValueError(f'{suite}: inconsistent or ambiguous collection timestamp')
            if not runs:
                missing.append(suite)
                continue
            if len(runs) != 1:
                raise ValueError(f'{suite}: ambiguous collection outputs in selected run attempt')
            run = runs[0]
            stamp = run.name
            if timestamp is None:
                timestamp = stamp
            if stamp != timestamp:
                raise ValueError('GPU suites have inconsistent collection timestamps')
            run_relative = str(run.relative_to(root))
            target = output / run_relative
            for path in sorted(run.rglob('*')):
                if path.is_file() or path.is_symlink():
                    relative = str(path.relative_to(root))
                    safe_file(root, relative)
                    destination = output / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
            for filename, suffix in (('report.md', '.md'), ('report.summary.json', '.summary.json')):
                original = safe_file(root, run_relative + '/' + filename)
                destination = output / f'result/{profile}/{suite}{suffix}'
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, destination)
            declared = declaration.get('backends', backends.split(','))
            expected = sorted((problem['id'], backend) for problem in declaration['problems']
                              for backend in backends.split(',') if backend in declared)
            files = [output / f'result/{profile}/{suite}{suffix}' for suffix in ('.md', '.summary.json')]
            files += sorted(path for path in target.rglob('*') if path.is_file())
            entries.append({'suite_id': suite, 'expected_rows': expected,
                            'files': {str(path.relative_to(output)): digest(path) for path in files}})
        if not entries:
            raise ValueError('No GPU collection outputs in the selected run attempt')
        manifest = {'schema_version': 2, 'timestamp': timestamp, 'suites': entries, 'missing_suites': missing}
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
