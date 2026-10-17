"""Overlay each suite's latest validated RunPod artifact onto the Pages tree."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .check_gpu_run import digest, validate_records
else:
    from check_gpu_run import digest, validate_records

# Only the standard nvidia-gpu hardware (NVIDIA A100 80GB PCIe) may override the
# committed A100 reports; non-A100 RunPod bundles are diagnostics only.
A100_RE = re.compile(r'A100', re.IGNORECASE)


def _run_timestamp(value: str) -> str | None:
    """Normalize an ISO summary timestamp to the run-directory form."""
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime('%Y%m%d_%H%M%S')


def _bundle_gpu_name(bundle: Path, run_relative: Path, summary: dict) -> str | None:
    """GPU name from the dashboard summary, falling back to the run metadata."""
    hardware = summary.get('hardware') or {}
    gpu = hardware.get('gpu')
    if gpu:
        return str(gpu)
    run_yaml = bundle / run_relative / 'run.yaml'
    if not run_yaml.exists():
        return None
    import yaml

    metadata = yaml.safe_load(run_yaml.read_text()) or {}
    cuda = metadata.get('cuda') or {}
    device = cuda.get('device_name')
    return str(device) if device else None


# A repository summary that is not valid JSON only means we cannot compare
# timestamps against it; the bundle is still allowed to publish.
def _site_summary_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return _run_timestamp(document.get('timestamp'))


def publish_bundle(bundle: Path, site: Path, seen: dict[str, str]) -> int:
    receipt = bundle / 'publication.json'
    if not receipt.exists():
        return 0  # Older artifacts were not checked for failed/missing measurements.
    manifest = json.loads(receipt.read_text())
    timestamp = manifest['timestamp']
    # Accept both the pre-summary schema (1) and the current one (2).
    if manifest['schema_version'] not in (1, 2) or not re.fullmatch(r'\d{8}_\d{6}', timestamp):
        raise ValueError('Invalid publication manifest')
    count = 0
    for entry in manifest['suites']:
        suite = entry['suite_id']
        if not re.fullmatch(r'gpu/[a-z0-9][a-z0-9_.-]*', suite):
            raise ValueError('Invalid publication suite')
        run_relative = Path('data/results/nvidia-gpu') / suite / timestamp
        report_relative = Path('result/nvidia-gpu') / f'{suite}.md'
        summary_relative = Path('result/nvidia-gpu') / f'{suite}.summary.json'
        files = entry['files']
        required = {str(report_relative), str(run_relative / 'records.jsonl'),
                    str(run_relative / 'run.yaml'), str(run_relative / 'report.md')}
        if not required.issubset(files):
            raise ValueError(f'{suite}: publication is missing required files')
        # Eligibility: only bundles produced by this harness carry the dashboard
        # summary sidecar.  Pre-summary bundles may not override the repository.
        if str(summary_relative) not in files:
            print(f'Skipping {suite}: bundle has no dashboard summary (older harness)')
            continue
        for relative, expected_digest in files.items():
            path = (bundle / relative).resolve()
            if not path.is_relative_to(bundle.resolve()) or digest(path) != expected_digest:
                raise ValueError(f'{suite}: invalid artifact file {relative}')
        summary_document = json.loads((bundle / summary_relative).read_text())
        gpu = _bundle_gpu_name(bundle, run_relative, summary_document)
        if not gpu or not A100_RE.search(gpu):
            print(f'Skipping {suite}: bundle GPU is not an A100 ({gpu!r})')
            continue
        # The dashboard reads whichever summary wins, so a RunPod bundle must be
        # newer than both the repository summary already copied into the site
        # and any RunPod bundle seen earlier in this invocation.
        baseline = seen.get(suite, '')
        repository_timestamp = _site_summary_timestamp(site / summary_relative)
        if repository_timestamp and repository_timestamp > baseline:
            baseline = repository_timestamp
        if baseline and baseline >= timestamp:
            print(f'Skipping {suite}: {baseline} is not older than bundle {timestamp}')
            continue
        expected = {tuple(row) for row in entry['expected_rows']}
        validate_records(bundle / run_relative / 'records.jsonl', suite, expected)
        destination = site / report_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / report_relative, destination)
        shutil.copy2(bundle / summary_relative, site / summary_relative)
        shutil.copytree(bundle / run_relative, site / 'raw/nvidia-gpu' / suite / timestamp,
                        dirs_exist_ok=True)
        seen[suite] = timestamp
        count += 1
        print(f'Published {suite}: {timestamp}')
    return count


def artifact_unavailable(stderr: str) -> bool:
    # Expired artifacts, and successful runs that never produced one (weekly-gate
    # skips, prepare_only), are expected; any other download error is not.
    message = stderr.lower()
    return any(text in message for text in ('no artifact', 'no valid artifacts', 'expired'))


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
                if artifact_unavailable(download.stderr):
                    print(f'Artifact unavailable for {run_id}; preserving maintained reports')
                    continue
                raise RuntimeError(download.stderr)
            publish_bundle(bundle, args.site_dir, seen)
    print(f'Updated {len(seen)} GPU suites; other reports remain from the repository')


if __name__ == '__main__':
    main()
