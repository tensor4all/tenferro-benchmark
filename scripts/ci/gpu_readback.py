"""Selection and completeness check for the solve/QR RunPod collection."""
import argparse
import json
import math
from pathlib import Path


def case_ids():
    cases = json.loads(Path('data/instances/gpu_perf_issues.json').read_text())
    ids = [row['id'] for row in cases if row['kind'] == 'linalg_readback_latency']
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate readback selection')
    return ids


def validate(path, expected, correctness_only):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row['case_id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Missing, duplicate, or unexpected readback rows')
    for row in rows:
        if row['correctness_status'] != 'passed' or row['regime'] != 'latency':
            raise ValueError(f"Readback validation failed: {row['case_id']}")
        samples = row['samples']
        if correctness_only:
            if samples:
                raise ValueError('Correctness-only run collected timings')
        elif not samples:
            raise ValueError('Readback row has no measured samples')
        for sample in samples:
            if (type(sample['iterations']) is not int or sample['iterations'] < 1
                    or not math.isfinite(sample['elapsed_ns']) or sample['elapsed_ns'] <= 0):
                raise ValueError('Invalid readback timing sample')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    if args.verify:
        validate(args.verify, case_ids(), args.correctness_only)
    else:
        print(','.join(case_ids()))
