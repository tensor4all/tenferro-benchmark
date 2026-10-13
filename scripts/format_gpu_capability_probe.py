#!/usr/bin/env python3
"""Render untimed public CUDA capability observations, including actual errors."""
import argparse
import json
import math
from pathlib import Path


def render(root):
    suite = json.loads((root / 'suite.json').read_text())
    provenance = json.loads((root / 'provenance.json').read_text())
    rows = [json.loads(line) for line in (root / 'direct.jsonl').read_text().splitlines()]
    if [row['case_id'] for row in rows] != [case['id'] for case in suite['cases']]:
        raise ValueError('missing, duplicate or reordered capability cases')
    if provenance['timings_collected'] is not False or len(rows) != provenance['cases']:
        raise ValueError('invalid capability provenance')
    lines = ['# CUDA public API capability recheck', '',
             f'Raw probe: `{root}/`.', '',
             f"tenferro-rs: `{provenance['tenferro_commit']}`; harness binary source: `{provenance['harness_binary_source']}`.", '',
             'GPU1 diagnostic only. Previous unsupported declarations were removed before calling '
             'the current public direct APIs. No timings were collected. Runtime failures remain '
             'explicit failures and do not count as numerical or measured operation coverage.', '',
             '| Case | Actual status | Observation |', '|---|---|---|']
    for row in rows:
        if row['gpu_role'] != 'debug' or row['physical_device_index'] != 1 or row.get('timing') is not None:
            raise ValueError('expected untimed debug GPU1 rows')
        if row['status'] == 'runtime_failed':
            observation = row.get('reason')
            if not observation:
                raise ValueError('missing runtime error')
        elif row['status'] == 'verification_only':
            error = row['verification']['max_error_over_tolerance']
            if not math.isfinite(error) or not 0 <= error <= 1:
                raise ValueError('invalid numerical verification')
            observation = f'Normalized numerical error: {error:.6g}'
        else:
            raise ValueError('unexpected capability status')
        lines.append(f"| {row['case_id']} | {row['status']} | {observation.replace('|', '/')} |")
    lines += ['', 'Current source explains these direct-path failures: inv_owned calls eye_like, '
              'which passes scalar_real host storage to broadcast; pinv_from_svd passes '
              'a scalar_real host threshold to binary_mul. Neither public call accepts a '
              'replacement device scalar. Eager linalg routes have separate implementations '
              'and are not classified by this direct-only probe.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.root))
