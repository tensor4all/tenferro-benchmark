#!/usr/bin/env python3
"""Ingest exactly one benchmark run; all executable code comes from this checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import zipfile

if __package__:
    from .result_ingestion import REPOSITORY, WORKFLOWS, plan_updates, source_from_api, validate_bundle
else:
    from result_ingestion import REPOSITORY, WORKFLOWS, plan_updates, source_from_api, validate_bundle


def gh_json(endpoint: str):
    return json.loads(subprocess.check_output(['gh', 'api', endpoint], text=True))


def selected_source(run_id: int):
    run = gh_json(f'repos/{REPOSITORY}/actions/runs/{run_id}')
    workflow = run.get('path')
    if workflow not in WORKFLOWS:
        raise ValueError('Unsupported source workflow')
    # Verify the run before downloading anything, including fork and branch.
    prefix = WORKFLOWS[workflow][1]
    artifacts = []
    page = 1
    while True:
        response = gh_json(f'repos/{REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100&page={page}')
        artifacts.extend(response['artifacts'])
        if len(artifacts) >= response['total_count']:
            break
        page += 1
        if page > 100:
            raise ValueError('Artifact pagination exceeded limit')
    selected = [item for item in artifacts if item['name'] == f'{prefix}-{run_id}']
    if len(selected) != 1:
        raise ValueError('Selected run must contain exactly one named result artifact; no fallback is allowed')
    artifact = selected[0]
    profile, source = source_from_api(run, artifact)
    return profile, source, artifact


def extract_archive(archive: Path, destination: Path, max_bytes=20 * 1024**3):
    """Reject unsafe archives before extraction, including duplicate file names."""
    with zipfile.ZipFile(archive) as zipped:
        seen = set()
        total = 0
        for info in zipped.infolist():
            name = info.filename.rstrip('/')
            parts = name.split('/')
            if (not name or name.startswith('/') or '\\' in name
                    or any(part in ('', '.', '..') for part in parts) or name in seen):
                raise ValueError(f'Unsafe or duplicate archive path: {info.filename}')
            seen.add(name)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError(f'Non-regular archive entry: {name}')
            total += info.file_size
            if total > max_bytes:
                raise ValueError('Uncompressed artifact exceeds size limit')
        zipped.extractall(destination)


def download_artifact(artifact: dict, destination: Path):
    archive = destination.parent / 'artifact.zip'
    with archive.open('wb') as output:
        subprocess.run(['gh', 'api', f"repos/{REPOSITORY}/actions/artifacts/{artifact['id']}/zip"],
                       stdout=output, check=True)
    expected = artifact.get('digest')
    if expected:
        with archive.open('rb') as data:
            actual = 'sha256:' + hashlib.file_digest(data, 'sha256').hexdigest()
        if expected != actual:
            raise ValueError('Artifact archive digest mismatch')
    extract_archive(archive, destination)


def git(root: Path, *args: str, check=True):
    return subprocess.run(['git', '-C', str(root), *args], text=True, capture_output=True, check=check)


def commit_results(root: Path, validated, source: dict, attempts=3):
    """Reevaluate freshness on every fetched main; never force-push."""
    if git(root, 'status', '--porcelain').stdout.strip():
        raise ValueError('Ingestion checkout must be clean')
    decisions = {}
    for _ in range(attempts):
        git(root, 'fetch', 'origin', 'main')
        git(root, 'checkout', '--detach', 'origin/main')
        updates, decisions = plan_updates(root, validated)
        if not updates:
            return decisions, None
        for relative, value in updates.items():
            path = root / relative
            # These paths are constructed by validation, never copied from
            # arbitrary manifest file names. Existing symlinks are forbidden.
            current = root
            for part in Path(relative).parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError(f'Unsafe destination: {relative}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        git(root, 'add', '--', *sorted(updates))
        staged = set(git(root, 'diff', '--cached', '--name-only').stdout.splitlines())
        if staged != set(updates):
            # All accepted suites must change at least provenance; individual
            # report/summary files may be identical to a legacy baseline.
            if not staged.issubset(updates) or not staged:
                raise ValueError('Unexpected staged paths')
        git(root, '-c', 'user.name=github-actions[bot]',
            '-c', 'user.email=41898282+github-actions[bot]@users.noreply.github.com',
            'commit', '-m', f"Accept benchmark run {source['run_id']} attempt {source['run_attempt']}")
        commit = git(root, 'rev-parse', 'HEAD').stdout.strip()
        pushed = git(root, 'push', 'origin', 'HEAD:main', check=False)
        if pushed.returncode == 0:
            return decisions, commit
        # Only discard our own generated tracked result changes in this
        # dedicated ingestion checkout before replaying against current main.
        git(root, 'checkout', '--detach', 'origin/main')
    raise RuntimeError('Could not push accepted results after bounded retries')


def request_publication():
    subprocess.run(['gh', 'api', '--method', 'POST', f'repos/{REPOSITORY}/dispatches',
                    '--input', '-'], input=json.dumps({'event_type': 'benchmark-results-ingested'}),
                   text=True, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', type=int, required=True)
    parser.add_argument('--repository-dir', type=Path, default=Path('.'))
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.run_id < 1:
        parser.error('run-id must be positive')
    profile, source, artifact = selected_source(args.run_id)
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / 'bundle'
        download_artifact(artifact, bundle)
        validated = validate_bundle(bundle, profile, source)
        if args.validate_only:
            decisions = {receipt['suite_id']: 'validated' for receipt, _ in validated}
            commit = None
        else:
            decisions, commit = commit_results(args.repository_dir.resolve(), validated, source)
            # Dispatch even after duplicate/stale ingestion to recover a
            # previous successful commit followed by a failed dispatch.
            request_publication()
        summary = json.dumps({'source_run': args.run_id, 'commit': commit, 'suites': decisions}, indent=2)
        print(summary)
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as output:
                output.write('## Benchmark ingestion\n\n```json\n' + summary + '\n```\n')


if __name__ == '__main__':
    main()
