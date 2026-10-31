#!/usr/bin/env python3
"""Ingest exactly one benchmark run; all executable code comes from this checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import subprocess
import tempfile
import time
import zipfile

if __package__:
    from .result_ingestion import REPOSITORY, WORKFLOWS, plan_updates, source_from_api, validate_bundle, validate_source_run
else:
    from result_ingestion import REPOSITORY, WORKFLOWS, plan_updates, source_from_api, validate_bundle, validate_source_run


def gh_json(endpoint: str):
    return json.loads(subprocess.check_output(['gh', 'api', endpoint], text=True))


# Workflows whose collection job can be skipped by design in a successful run
# (the weekly GPU gate skips later slots once that week's results exist).
GATED_COLLECTION_JOBS = {
    '.github/workflows/benchmark-runpod-gpu.yml': 'Run GPU benchmark on RunPod',
}


def collection_skipped(run: dict) -> bool:
    job_name = GATED_COLLECTION_JOBS.get(run.get('path'))
    if job_name is None:
        return False
    jobs = gh_json(f"repos/{REPOSITORY}/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs?per_page=100")['jobs']
    matching = [job for job in jobs if job['name'] == job_name]
    return bool(matching) and all(job.get('conclusion') == 'skipped' for job in matching)


def selected_source(run_id: int):
    """Return (profile, source, artifact), or None when the run collected nothing."""
    run = gh_json(f'repos/{REPOSITORY}/actions/runs/{run_id}')
    validate_source_run(run)
    workflow = run['path']
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
    from datetime import datetime
    names = {f'{prefix}-{run_id}', f"{prefix}-{run_id}-attempt{run['run_attempt']}"}
    selected = [item for item in artifacts if item['name'] in names
                and datetime.fromisoformat(item['created_at'].replace('Z', '+00:00')) >=
                datetime.fromisoformat(run['run_started_at'].replace('Z', '+00:00'))]
    if not selected and run['conclusion'] == 'success' and collection_skipped(run):
        return None
    if not selected and run['conclusion'] == 'failure':
        diagnostic_prefix = {
            'amd-cpu': 'linux-cpu-benchmark-diagnostics',
            'mac-cpu': 'mac-cpu-benchmark-diagnostics',
            'nvidia-gpu': 'runpod-gpu-diagnostics',
        }[WORKFLOWS[workflow][0]]
        selected = [item for item in artifacts if item['name'] == f'{diagnostic_prefix}-{run_id}'
                    and datetime.fromisoformat(item['created_at'].replace('Z', '+00:00')) >=
                    datetime.fromisoformat(run['run_started_at'].replace('Z', '+00:00'))]
    if len(selected) != 1:
        raise ValueError(f'Run {run_id} must contain exactly one named result artifact ({sorted(names)}) '
                         'or its same-attempt failure diagnostics with complete collection outputs; '
                         'required artifact missing or ambiguous; no other run/attempt fallback is allowed')
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


def gh_pull_request(*args: str, check=True):
    """Pull-request API calls run as the ingestion App so the PR triggers CI.

    Pull requests opened with GITHUB_TOKEN never start workflows, so the
    required status checks on main would never report.
    """
    env = dict(os.environ, GH_TOKEN=os.environ.get('INGEST_PR_TOKEN') or os.environ.get('GH_TOKEN', ''))
    return subprocess.run(['gh', 'api', *args], text=True, capture_output=True, check=check, env=env)


def required_checks() -> set[str]:
    rules = gh_json(f'repos/{REPOSITORY}/rules/branches/main')
    return {check['context'] for rule in rules if rule['type'] == 'required_status_checks'
            for check in rule['parameters']['required_status_checks']}


def wait_for_checks(commit: str, names: set[str], timeout=40 * 60, interval=30):
    """Wait for every required check on the exact result commit to succeed."""
    deadline = time.monotonic() + timeout
    while True:
        runs = gh_json(f'repos/{REPOSITORY}/commits/{commit}/check-runs?per_page=100')['check_runs']
        # A rerun leaves the older run in the list; the newest one counts.
        latest = {}
        for run in sorted(runs, key=lambda item: item.get('started_at') or ''):
            if run['name'] in names:
                latest[run['name']] = run
        failed = sorted(name for name, run in latest.items()
                        if run['status'] == 'completed' and run['conclusion'] != 'success')
        if failed:
            raise RuntimeError(f'Required checks failed on {commit}: {", ".join(failed)}')
        if set(latest) == names and all(run['status'] == 'completed' for run in latest.values()):
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f'Required checks did not finish on {commit}')
        time.sleep(interval)


def merge_through_pull_request(branch: str, commit: str, title: str):
    """Merge the exact result commit through a PR; None means main moved and a replay is needed."""
    owner = REPOSITORY.split('/')[0]
    found = json.loads(gh_pull_request(f'repos/{REPOSITORY}/pulls?head={owner}:{branch}&state=open').stdout)
    if found:
        number = found[0]['number']
    else:
        created = gh_pull_request('--method', 'POST', f'repos/{REPOSITORY}/pulls', '-f', f'title={title}',
                                  '-f', f'head={branch}', '-f', 'base=main',
                                  '-f', 'body=Validated benchmark results opened by ingest-results. '
                                        'It merges itself once the required checks pass.')
        number = json.loads(created.stdout)['number']
    wait_for_checks(commit, required_checks())
    # Pinning the head SHA merges exactly the validated commit or nothing.
    merged = gh_pull_request('--method', 'PUT', f'repos/{REPOSITORY}/pulls/{number}/merge',
                             '-f', 'merge_method=squash', '-f', f'sha={commit}',
                             '-f', f'commit_title={title}', '-f', 'commit_message=', check=False)
    if merged.returncode == 0:
        return json.loads(merged.stdout)['sha']
    state = json.loads(gh_pull_request(f'repos/{REPOSITORY}/pulls/{number}').stdout)
    if state.get('merged'):
        return state['merge_commit_sha']
    if not state.get('mergeable'):
        return None
    raise RuntimeError(f'Could not merge result pull request #{number}: {merged.stderr.strip()}')


def commit_results(root: Path, validated, source: dict, attempts=3):
    """Reevaluate freshness on every fetched main; main itself is never pushed.

    Results land through a pull request because main requires status checks.
    The ingest/ branch is owned by this driver and is overwritten on replay.
    """
    if git(root, 'status', '--porcelain').stdout.strip():
        raise ValueError('Ingestion checkout must be clean')
    decisions = {}
    title = f"Accept benchmark run {source['run_id']} attempt {source['run_attempt']}"
    branch = f"ingest/run-{source['run_id']}-attempt-{source['run_attempt']}"
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
            'commit', '-m', title)
        commit = git(root, 'rev-parse', 'HEAD').stdout.strip()
        git(root, 'push', '--force', 'origin', f'HEAD:refs/heads/{branch}')
        merged = merge_through_pull_request(branch, commit, title)
        if merged:
            return decisions, merged
        # Only discard our own generated tracked result changes in this
        # dedicated ingestion checkout before replaying against current main.
        git(root, 'checkout', '--detach', 'origin/main')
    raise RuntimeError('Could not merge accepted results after bounded retries')


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
    selection = selected_source(args.run_id)
    if selection is None:
        message = f'Run {args.run_id} skipped its collection job; there are no results to ingest.'
        print(message)
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as output:
                output.write('## Benchmark ingestion\n\n' + message + '\n')
        return
    profile, source, artifact = selection
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / 'bundle'
        download_artifact(artifact, bundle)
        if source['artifact_kind'] == 'diagnostics':
            if profile == 'nvidia-gpu' and source['event'] != 'schedule':
                raise ValueError('GPU diagnostic recovery requires a scheduled default-suite run; custom selection is unknown')
            if __package__:
                from .prepare_result_bundle import prepare_bundle
            else:
                from prepare_result_bundle import prepare_bundle
            validated = prepare_bundle(bundle, profile, source, Path(directory) / 'recovered')
        else:
            validated = validate_bundle(bundle, profile, source)
        if args.validate_only:
            decisions = {receipt['suite_id']: 'validated' for receipt, _ in validated}
            commit = None
        else:
            decisions, commit = commit_results(args.repository_dir.resolve(), validated, source)
            # Dispatch even after duplicate/stale ingestion to recover a
            # previous successful commit followed by a failed dispatch.
            request_publication()
        summary = json.dumps({'source_run': args.run_id, 'commit': commit, 'suites': decisions,
                              'records': {receipt['suite_id']: {key: receipt[key] for key in
                                          ('status', 'case_outcomes', 'missing_suites')}
                                          for receipt, _ in validated}}, indent=2)
        print(summary)
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as output:
                output.write('## Benchmark ingestion\n\n```json\n' + summary + '\n```\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        message = f'Benchmark ingestion rejected or failed: {error}'
        print(message, file=sys.stderr)
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as output:
                output.write('## Benchmark ingestion failed\n\n' + message + '\n')
        raise SystemExit(1) from error
