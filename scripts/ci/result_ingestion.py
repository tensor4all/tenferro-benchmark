"""Deterministic, offline acceptance of benchmark result bundles.

Network/run identity acquisition belongs to the driver. This module never
executes artifact contents and validates the complete bundle before writes.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

if __package__:
    from .check_gpu_run import digest
else:
    from check_gpu_run import digest

REPOSITORY = 'tensor4all/tenferro-benchmark'
WORKFLOWS = {
    '.github/workflows/benchmark-nightly-mac-cpu.yml': ('mac-cpu', 'mac-cpu-benchmark-results'),
    '.github/workflows/benchmark-nightly-linux-cpu.yml': ('amd-cpu', 'linux-cpu-benchmark-results'),
    '.github/workflows/benchmark-runpod-gpu.yml': ('nvidia-gpu', 'runpod-gpu-benchmark'),
}


def instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Collection timestamp must include a timezone')
    return parsed.astimezone(timezone.utc)


def safe_file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or '\\' in relative
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise ValueError(f'Unsafe bundle path: {relative}')
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'Symlink in bundle: {relative}')
    if not current.resolve().is_relative_to(root.resolve()) or not current.is_file():
        raise ValueError(f'Missing or unsafe file: {relative}')
    return current


def source_from_api(run: dict, artifact: dict) -> tuple[str, dict]:
    workflow = run['path']
    if (run.get('repository', {}).get('full_name') != REPOSITORY
            or run.get('head_repository', {}).get('full_name') != REPOSITORY
            or run.get('head_branch') != 'main' or run.get('conclusion') != 'success'
            or run.get('status') != 'completed' or workflow not in WORKFLOWS):
        raise ValueError('Run is not a successful canonical main workflow')
    profile, prefix = WORKFLOWS[workflow]
    if artifact.get('expired') or artifact['name'] != f"{prefix}-{run['id']}":
        raise ValueError('Artifact is expired or has an unexpected name')
    if instant(artifact['created_at']) < instant(run['run_started_at']):
        raise ValueError('Artifact predates the selected run attempt')
    artifact_run = artifact.get('workflow_run') or {}
    if artifact_run.get('id') != run['id'] or artifact_run.get('head_sha') != run['head_sha']:
        raise ValueError('Artifact does not belong to the selected source run')
    source = {
        'repository': REPOSITORY, 'workflow_path': workflow, 'head_branch': 'main',
        'head_sha': run['head_sha'], 'run_id': run['id'], 'run_attempt': run['run_attempt'],
        'artifact_id': artifact['id'], 'run_started_at': run['run_started_at'], 'run_url': f"https://github.com/{REPOSITORY}/actions/runs/{run['id']}",
        'artifact_url': f"https://github.com/{REPOSITORY}/actions/runs/{run['id']}/artifacts/{artifact['id']}",
        'created_at': artifact['created_at'], 'expires_at': artifact['expires_at'],
        'digest': artifact.get('digest'),
    }
    for key in ('run_id', 'run_attempt', 'artifact_id'):
        if type(source[key]) is not int or source[key] < 1:
            raise ValueError(f'Invalid source identity: {key}')
    if not re.fullmatch('[0-9a-f]{40}', source['head_sha']):
        raise ValueError('Invalid source revision')
    return profile, source


def validate_summary(document: dict, profile: str, entry: dict) -> None:
    if document.get('schema_version') != 1:
        raise ValueError('Unsupported summary schema')
    for key, expected in [('target_profile', profile), ('suite_id', entry['suite_id']),
                          ('tenferro_commit', entry['tenferro_commit']), ('hardware', entry['hardware'])]:
        if document.get(key) != expected:
            raise ValueError(f'Summary does not match manifest: {key}')
    if instant(document['timestamp']) != instant(entry['timestamp']):
        raise ValueError('Summary timestamp mismatch')
    if not re.fullmatch('[0-9a-f]{40}', entry['tenferro_commit']):
        raise ValueError('Invalid measured revision')
    hardware = entry['hardware']
    if profile == 'nvidia-gpu':
        if ' '.join(str(hardware.get('gpu', '')).lower().split()) != 'nvidia a100 80gb pcie':
            raise ValueError('Only standard NVIDIA A100 80GB PCIe results may publish')
    elif not (hardware.get('host') or hardware.get('cpu')):
        raise ValueError('CPU hardware identity is missing')
    expected = [tuple(row) for row in entry['expected_rows']]
    if not expected or len(set(expected)) != len(expected) or any(len(row) != 3 for row in expected):
        raise ValueError('Invalid expected summary identities')
    seen = set()
    measured = 0
    for row in document['rows']:
        key = (row['case'], row.get('group', ''), row['backend'])
        if key in seen:
            raise ValueError(f'Duplicate summary row: {key}')
        seen.add(key)
        if row['status'] == 'ok':
            value = row.get('median')
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or row.get('verified') is not True:
                raise ValueError(f'Invalid or unverified measurement: {key}')
            measured += 1
        elif (row['status'] not in ('unsupported', 'skipped') and not str(row['status']).startswith('unsupported:')) or not row.get('reason') or row.get('median') is not None:
            raise ValueError(f'Incomplete or unexplained row: {key}')
    if seen != set(expected) or measured == 0:
        raise ValueError('Missing/unexpected rows or no measured rows')


def validate_bundle(bundle: Path, profile: str, source: dict) -> list[tuple[dict, dict[str, bytes]]]:
    if profile == 'nvidia-gpu':
        if __package__:
            from .gpu_ingestion_adapter import adapt_publication
        else:
            from gpu_ingestion_adapter import adapt_publication
        manifest = adapt_publication(bundle, source)
    else:
        manifest = json.loads(safe_file(bundle, 'ingestion.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('target_profile') != profile:
        raise ValueError('Invalid ingestion manifest')
    if manifest.get('harness_commit') != source['head_sha']:
        raise ValueError('Bundle harness commit does not match source run')
    entries = manifest['suites']
    if not entries:
        raise ValueError('Empty bundle')
    validated = []
    seen = set()
    for entry in entries:
        suite = entry['suite_id']
        family = 'gpu' if profile == 'nvidia-gpu' else 'cpu'
        if not re.fullmatch(f'{family}/[a-z0-9][a-z0-9_.-]*', suite) or suite.split('/')[1] in ('.', '..') or suite in seen:
            raise ValueError('Invalid or duplicate suite')
        seen.add(suite)
        run_directory = entry['run_directory']
        prefix = f'data/results/{profile}/{suite}/'
        if not run_directory.startswith(prefix) or not re.fullmatch(r'\d{8}_\d{6}', run_directory[len(prefix):]):
            raise ValueError('Invalid run directory')
        report = f'result/{profile}/{suite}.md'
        summary = f'result/{profile}/{suite}.summary.json'
        files = entry['files']
        required = {report, summary, f'{run_directory}/report.md', f'{run_directory}/report.summary.json', f'{run_directory}/run.yaml'}
        if not required.issubset(files):
            raise ValueError('Required bundle files are missing')
        for relative, expected in files.items():
            if relative not in (report, summary) and not relative.startswith(run_directory + '/'):
                raise ValueError('Undeclared file scope')
            path = safe_file(bundle, relative)
            if not re.fullmatch('[0-9a-f]{64}', expected) or digest(path) != expected:
                raise ValueError(f'Digest mismatch: {relative}')
        payload = {relative: safe_file(bundle, relative).read_bytes() for relative in (report, summary)}
        if payload[report] != safe_file(bundle, f'{run_directory}/report.md').read_bytes() or payload[summary] != safe_file(bundle, f'{run_directory}/report.summary.json').read_bytes():
            raise ValueError('Published report/summary differ from run copies')
        document = json.loads(payload[summary])
        if profile != 'nvidia-gpu':
            if __package__:
                from .cpu_ingestion_adapter import rebuild_cpu
            else:
                from cpu_ingestion_adapter import rebuild_cpu
            if rebuild_cpu(bundle, entry, profile, source) != document:
                raise ValueError('CPU summary differs from raw inputs')
        if 'run_started_at' in source and not (instant(source['run_started_at']) <= instant(entry['timestamp']) <= instant(source['created_at'])):
            raise ValueError('Collection timestamp is outside the source run attempt')
        validate_summary(document, profile, entry)
        receipt = {'schema_version': 1, 'target_profile': profile, 'suite_id': suite,
                   'collected_at': entry['timestamp'], 'harness_commit': manifest['harness_commit'],
                   'tenferro_commit': entry['tenferro_commit'], 'hardware': entry['hardware'],
                   'source': source, 'files': files}
        validated.append((receipt, payload))
    return validated


def order_key(receipt: dict) -> tuple:
    source = receipt['source']
    return (instant(receipt['collected_at']), source['run_id'], source['run_attempt'], source['artifact_id'])


def plan_updates(root: Path, validated: list[tuple[dict, dict[str, bytes]]]) -> tuple[dict[str, bytes], dict[str, str]]:
    updates, decisions = {}, {}
    for receipt, payload in validated:
        suite = receipt['suite_id']
        base = f"result/{receipt['target_profile']}/{suite}"
        relative = base + '.provenance.json'
        path = root / relative
        decision = 'accepted'
        if path.exists():
            previous = json.loads(path.read_text())
            old, new = order_key(previous), order_key(receipt)
            if old == new:
                if previous != receipt:
                    raise ValueError('Identical source key has conflicting content')
                # Detect manual divergence instead of claiming a false duplicate.
                if any(not (root / name).exists() or (root / name).read_bytes() != value for name, value in payload.items()):
                    raise ValueError('Committed result diverges from accepted provenance')
                decision = 'duplicate'
            elif old > new:
                decision = 'stale'
        elif (root / (base + '.summary.json')).exists():
            legacy = json.loads((root / (base + '.summary.json')).read_text())
            if instant(legacy['timestamp']) >= instant(receipt['collected_at']):
                decision = 'stale'
        decisions[suite] = decision
        if decision == 'accepted':
            updates.update(payload)
            updates[relative] = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
    return updates, decisions
