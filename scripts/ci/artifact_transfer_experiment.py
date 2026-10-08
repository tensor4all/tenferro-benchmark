#!/usr/bin/env python3
"""Measure Actions artifact transfer plus exact reconstruction on a paid pod.

This explicitly measures CI setup, not tensor-operation performance. The
partitioned transport follows gpu_environment.py's separate-artifact approach;
it partitions the concatenated original files into exactly five equal parts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import time
from pathlib import Path

FILES = ('cuda-tests.tar.zst', 'pjrt-tests.tar.zst', 'cuda-tutorial.tar.zst')
PARTS = 5
BUFFER = 8 * 1024 * 1024
PROTOCOL = {
    'scope': 'Actions download plus reconstruction and whole-file SHA-256 verification',
    'order': ['single', 'split', 'split', 'single', 'single', 'split'],
    'pairs': 3,
    'minimum_median_reduction': 0.20,
    'maximum_within_arm_ratio': 1.5,
    'maximum_idle_cpu_busy_fraction': 0.20,
    'maximum_idle_receive_bytes_per_second': 5 * 1024 * 1024,
    'retries': 0,
    'download_step_timeout_minutes': 3,
    'paid_job_timeout_minutes': 20,
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(BUFFER), b''):
            result.update(chunk)
    return result.hexdigest()


def prepare(source: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    sizes = [source.joinpath(name).stat().st_size for name in FILES]
    total = sum(sizes)
    if total < PARTS or total > 2 * 1024 ** 3:
        raise ValueError('Experiment requires 5 bytes to 2 GiB of original archives')
    manifest = {'files': [{'name': name, 'bytes': size, 'sha256': digest(source / name)}
                          for name, size in zip(FILES, sizes)], 'parts': [],
                'protocol': PROTOCOL}
    index = 0
    original = source.joinpath(FILES[index]).open('rb')
    try:
        for part in range(PARTS):
            folder = destination / f'{part:02d}'
            folder.mkdir()
            path = folder / f'archives.part{part:02d}'
            remaining = total * (part + 1) // PARTS - total * part // PARTS
            with path.open('wb') as output:
                while remaining:
                    chunk = original.read(min(BUFFER, remaining))
                    if not chunk:
                        original.close()
                        index += 1
                        if index >= len(FILES):
                            raise ValueError('Original archive changed while partitioning')
                        original = source.joinpath(FILES[index]).open('rb')
                        continue
                    output.write(chunk)
                    remaining -= len(chunk)
            manifest['parts'].append({'name': path.name, 'bytes': path.stat().st_size})
    finally:
        original.close()
    text = json.dumps(manifest, indent=2) + '\n'
    (source / 'transfer.json').write_text(text)
    (destination / '00' / 'transfer.json').write_text(text)
    return manifest


def verify(folder: Path, arm: str, manifest_sha256: str) -> None:
    manifest_path = folder / 'transfer.json'
    if digest(manifest_path) != manifest_sha256:
        raise ValueError('Transfer manifest does not match hosted preparation')
    manifest = json.loads(manifest_path.read_text())
    if [entry['name'] for entry in manifest['files']] != list(FILES):
        raise ValueError('Unexpected original archive list')
    if arm == 'split':
        parts = manifest['parts']
        names = [f'archives.part{part:02d}' for part in range(PARTS)]
        if [entry['name'] for entry in parts] != names:
            raise ValueError('Unexpected part list')
        if sorted(path.name for path in folder.glob('archives.part*')) != names:
            raise ValueError('Missing or extra transfer parts')
        for entry in parts:
            if (folder / entry['name']).stat().st_size != entry['bytes']:
                raise ValueError('Transfer part size mismatch')
        part_index = 0
        stream = folder.joinpath(names[part_index]).open('rb')
        try:
            for entry in manifest['files']:
                remaining = entry['bytes']
                with folder.joinpath(entry['name']).open('wb') as output:
                    while remaining:
                        chunk = stream.read(min(BUFFER, remaining))
                        if not chunk:
                            stream.close()
                            part_index += 1
                            if part_index >= PARTS:
                                raise ValueError('Incomplete partitioned archive')
                            stream = folder.joinpath(names[part_index]).open('rb')
                            continue
                        output.write(chunk)
                        remaining -= len(chunk)
        finally:
            stream.close()
        if sum(entry['bytes'] for entry in parts) != sum(entry['bytes'] for entry in manifest['files']):
            raise ValueError('Unexpected total transfer length')
    for entry in manifest['files']:
        path = folder / entry['name']
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f"Archive content mismatch: {entry['name']}")


def idle_observation(core: int) -> dict:
    def counters():
        cpu = next(line for line in Path('/proc/stat').read_text().splitlines()
                   if line.startswith(f'cpu{core} '))
        fields = list(map(int, cpu.split()[1:9]))
        receive = sum(int(line.split(':', 1)[1].split()[0])
                      for line in Path('/proc/net/dev').read_text().splitlines()
                      if ':' in line and line.split(':', 1)[0].strip() != 'lo')
        return sum(fields), fields[3] + fields[4], receive
    before = counters()
    begin = time.monotonic()
    time.sleep(1)
    after = counters()
    total = after[0] - before[0]
    return {'core': core, 'cpu_busy_fraction': 1 - (after[1] - before[1]) / max(total, 1),
            'receive_bytes_per_second': (after[2] - before[2]) / (time.monotonic() - begin),
            'load1': os.getloadavg()[0], 'cpu_count': os.cpu_count()}


def start(root: Path, label: str) -> None:
    root.mkdir(exist_ok=True)
    folder = root / label
    shutil.rmtree(folder, ignore_errors=True)
    core = min(os.sched_getaffinity(0))
    before = idle_observation(core)
    state = {'label': label, 'before': before, 'start': time.monotonic(),
             'machine': platform.platform(), 'affinity': sorted(os.sched_getaffinity(0))}
    (root / 'active.json').write_text(json.dumps(state))


def finish(root: Path, label: str, arm: str, manifest_sha256: str) -> dict:
    state = json.loads((root / 'active.json').read_text())
    if state['label'] != label:
        raise ValueError('Measurement label mismatch')
    verify(root / label, arm, manifest_sha256)
    state['seconds'] = time.monotonic() - state.pop('start')
    state['arm'] = arm
    state['after'] = idle_observation(state['before']['core'])
    with (root / 'samples.jsonl').open('a') as output:
        output.write(json.dumps(state) + '\n')
    shutil.rmtree(root / label)
    return state


def report(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'samples.jsonl'
    samples = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    complete = [row['arm'] for row in samples] == PROTOCOL['order']
    medians = {arm: statistics.median(row['seconds'] for row in samples if row['arm'] == arm)
               for arm in ('single', 'split') if any(row['arm'] == arm for row in samples)}
    reasons = []
    if not complete:
        reasons.append('Incomplete predeclared six-sample suite')
    for row in samples:
        for phase in ('before', 'after'):
            observation = row[phase]
            if observation['cpu_busy_fraction'] > PROTOCOL['maximum_idle_cpu_busy_fraction']:
                reasons.append(f"{row['label']} {phase}: monitored core busy")
            if observation['receive_bytes_per_second'] > PROTOCOL['maximum_idle_receive_bytes_per_second']:
                reasons.append(f"{row['label']} {phase}: background receive traffic")
            if observation['load1'] > observation['cpu_count']:
                reasons.append(f"{row['label']} {phase}: host load exceeds CPU count")
    for arm in ('single', 'split'):
        values = [row['seconds'] for row in samples if row['arm'] == arm]
        if values and max(values) / min(values) > PROTOCOL['maximum_within_arm_ratio']:
            reasons.append(f'{arm}: repeated-arm noise exceeds 1.5x')
    reduction = 1 - medians['split'] / medians['single'] if len(medians) == 2 else None
    result = {'protocol': PROTOCOL, 'samples': samples, 'median_seconds': medians,
              'median_reduction': reduction, 'validity_failures': reasons,
              'verdict': 'INCONCLUSIVE' if reasons else
                         ('PASS' if reduction >= PROTOCOL['minimum_median_reduction'] else 'FAIL')}
    (root / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a') as output:
            output.write(f"## Artifact transfer experiment\n\n{result['verdict']}\n\n")
            output.write(f"Median seconds: {medians}\n\nMedian reduction: {reduction}\n\n")
            output.write('\n'.join(reasons) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'start', 'finish', 'report'))
    parser.add_argument('--source', type=Path, default=Path('_transfer_source'))
    parser.add_argument('--parts', type=Path, default=Path('_transfer_parts'))
    parser.add_argument('--root', type=Path, default=Path('_transfer_results'))
    parser.add_argument('--label', default='')
    parser.add_argument('--arm', choices=('single', 'split'), default='single')
    parser.add_argument('--manifest-sha256', default='')
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.source, args.parts)
    elif args.action == 'start':
        result = start(args.root, args.label)
    elif args.action == 'finish':
        result = finish(args.root, args.label, args.arm, args.manifest_sha256)
    else:
        result = report(args.root)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
