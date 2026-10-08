"""Overlay each suite's latest validated RunPod artifact onto the Pages tree."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

if __package__:
    from .check_gpu_run import digest, validate_records
else:
    from check_gpu_run import digest, validate_records


def publish_bundle(bundle: Path, site: Path, seen: dict[str, str]) -> int:
    receipt = bundle / 'publication.json'
    if not receipt.exists():
        return 0  # Older artifacts were not checked for failed/missing measurements.
    manifest = json.loads(receipt.read_text())
    timestamp = manifest['timestamp']
    if manifest['schema_version'] != 1 or not re.fullmatch(r'\d{8}_\d{6}', timestamp):
        raise ValueError('Invalid publication manifest')
    count = 0
    for entry in manifest['suites']:
        suite = entry['suite_id']
        if not re.fullmatch(r'gpu/[a-z0-9][a-z0-9_.-]*', suite):
            raise ValueError('Invalid publication suite')
        if seen.get(suite, '') >= timestamp:
            continue
        run_relative = Path('data/results/nvidia-gpu') / suite / timestamp
        report_relative = Path('result/nvidia-gpu') / f'{suite}.md'
        files = entry['files']
        required = {str(report_relative), str(run_relative / 'records.jsonl'),
                    str(run_relative / 'run.yaml'), str(run_relative / 'report.md')}
        if not required.issubset(files):
            raise ValueError(f'{suite}: publication is missing required files')
        for relative, expected_digest in files.items():
            path = (bundle / relative).resolve()
            if not path.is_relative_to(bundle.resolve()) or digest(path) != expected_digest:
                raise ValueError(f'{suite}: invalid artifact file {relative}')
        expected = {tuple(row) for row in entry['expected_rows']}
        validate_records(bundle / run_relative / 'records.jsonl', suite, expected)
        destination = site / report_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / report_relative, destination)
        shutil.copytree(bundle / run_relative, site / 'raw/nvidia-gpu' / suite / timestamp,
                        dirs_exist_ok=True)
        seen[suite] = timestamp
        count += 1
        print(f'Published {suite}: {timestamp}')
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--site-dir', type=Path, required=True)
    args = parser.parse_args()
    runs = json.loads(subprocess.check_output([
        'gh', 'run', 'list', '--repo', args.repository, '--workflow', 'benchmark-runpod-gpu.yml',
        '--branch', 'main', '--status', 'success', '--limit', '100',
        '--json', 'databaseId,createdAt',
    ], text=True))
    seen = {}
    # Explicit sorting avoids relying on the CLI's presentation order.
    for run in sorted(runs, key=lambda run: (run['createdAt'], run['databaseId']), reverse=True):
        run_id = str(run['databaseId'])
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            download = subprocess.run([
                'gh', 'run', 'download', run_id, '--repo', args.repository,
                '--name', f'runpod-gpu-benchmark-{run_id}', '--dir', str(bundle),
            ], text=True, capture_output=True)
            if download.returncode:
                if 'no artifact' in download.stderr.lower() or 'expired' in download.stderr.lower():
                    print(f'Artifact unavailable for {run_id}; preserving maintained reports')
                    continue
                raise RuntimeError(download.stderr)
            publish_bundle(bundle, args.site_dir, seen)
    print(f'Updated {len(seen)} GPU suites; other reports remain from the repository')


if __name__ == '__main__':
    main()
