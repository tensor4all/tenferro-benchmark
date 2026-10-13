#!/usr/bin/env python3
"""Render untimed public CUDA counter observations, never timing result rows."""
import argparse
import json
from pathlib import Path


def render(path: Path) -> str:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    before = [row for row in rows if row['phase'] == 'before']
    ids = [row['case_id'] for row in before]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('missing or duplicate initial observations')
    if any(row['status'] != 'instrumentation_only' or row['timing'] is not None
           or row['measurement_scope'] != 'separate_untimed_public_counter_pass' for row in rows):
        raise ValueError('not an untimed instrumentation pass')
    lines = ['# Public CUDA lifetime counters', '',
             f'Raw observations: `{path}`.', '',
             'These observations were collected separately from timing on debug GPU1. '
             'They are not performance measurements or standard operation coverage. '
             'Each case uses one entered backend session and retains outputs until native completion.', '',
             'Driver free memory covers the whole device. Its change cannot isolate an output pool, '
             'prove a leak, or explain the separately observed latency growth. '
             'No public output-pool statistics or clear API is available.', '',
             '| Case | Batches | Operations/batch | Maximum normalized error | Driver free-byte change after final drop | Workspace retained bytes | Retirement in flight / leaked |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for start in before:
        if start['gpu_role'] != 'debug' or start['physical_device_index'] != 1:
            raise ValueError('expected debug GPU1')
        batches = [row for row in rows if row['case_id'] == start['case_id'] and row['phase'] == 'batch']
        if [row['batch'] for row in batches] != list(range(start['batches'])):
            raise ValueError('missing, duplicated, or reordered batches')
        if any(row['operations_per_batch'] != start['operations_per_batch'] or
               not 0 <= row['max_error_over_tolerance'] <= 1 for row in batches):
            raise ValueError('invalid counts or numerical verification')
        end = batches[-1]['after_drop_and_barrier']
        retirement = end['workspace_retirement']
        lines.append(f"| {start['case_id']} | {len(batches)} | {start['operations_per_batch']} | "
                     f"{max(row['max_error_over_tolerance'] for row in batches):.6g} | "
                     f"{end['driver_free_bytes'] - start['snapshot']['driver_free_bytes']} | "
                     f"{end['workspace']['retained_bytes']} | {retirement['in_flight']} / {retirement['leaked']} |")
    if len(rows) != sum(start['batches'] + 1 for start in before):
        raise ValueError('unexpected observations')
    lines += ['', 'Raw JSONL preserves the cache entries, bytes, hits, misses, evictions, clears, '
              'workspace retirement counters, and pending transfer retirements at enqueue, completion, '
              'and output drop. Queries can perturb execution; they are not synchronization primitives.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(render(args.raw))
