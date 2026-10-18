"""List rows where a tenferro-rs backend is clearly slower than a reference.

Reads ``tenferro-bench-row/1`` JSONL (scripts/ci/bench_rows.py) and compares
each measured tenferro row with the fastest measured reference row of the same
profile, suite, problem and thread count. A row is reported when the median
ratio reaches the threshold and the IQR-widened intervals do not overlap, as
in gpu_slow_rows.py. A row whose IQR exceeds its median is below the timer's
useful resolution and takes no part in a comparison. Bound rows (memcpy) are
never references. Suites without any reference row are listed as such.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from scripts.ci.bench_rows import load
from scripts.ci.gpu_slow_rows import ms


def comparable(row: dict) -> bool:
    return row['status'] == 'ok' and row['iqr_ms'] is not None and row['iqr_ms'] <= row['median_ms']


def slow_rows(rows: list[dict], threshold: float) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in filter(comparable, rows):
        key = (row['target_profile'], row['suite_id'], row['problem_id'], row['threads'])
        groups.setdefault(key, []).append(row)
    findings = []
    for group in groups.values():
        references = [r for r in group if r['role'] == 'reference']
        if not references:
            continue
        ref = min(references, key=lambda r: r['median_ms'])
        ref_high = ref['median_ms'] + ref['iqr_ms']
        for row in group:
            ratio = row['median_ms'] / ref['median_ms']
            if row['role'] == 'tenferro' and ratio >= threshold and row['median_ms'] - row['iqr_ms'] > ref_high:
                findings.append(dict(row, reference_backend=ref['backend'], reference_median_ms=ref['median_ms'],
                                     reference_iqr_ms=ref['iqr_ms'], ratio=round(ratio, 3)))
    return sorted(findings, key=lambda f: f['ratio'], reverse=True)


def suite_summary(rows: list[dict]) -> list[dict]:
    suites: dict[tuple, dict] = {}
    for row in rows:
        entry = suites.setdefault((row['suite_id'], row['raw_run']), {
            'suite_id': row['suite_id'], 'raw_run': row['raw_run'], 'tenferro_commit': row['tenferro_commit'],
            'blas': row.get('blas'), 'device': row.get('device'), 'status': Counter(), 'has_reference': False})
        entry['status'][row['status']] += 1
        entry['has_reference'] |= row['role'] == 'reference' and row['status'] == 'ok'
    return [dict(e, status=dict(e['status'])) for e in suites.values()]


def op_summary(findings: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for f in findings:
        groups.setdefault((f['suite_id'], f['op'], f['backend']), []).append(f['ratio'])
    summary = [{'suite_id': s, 'op': op, 'backend': b, 'rows': len(r), 'max_ratio': max(r),
                'median_ratio': statistics.median(r)} for (s, op, b), r in groups.items()]
    return sorted(summary, key=lambda g: g['max_ratio'], reverse=True)


def collect(rows: list[dict], threshold: float) -> dict:
    findings = slow_rows(rows, threshold)
    return {'threshold': threshold, 'runs': suite_summary(rows), 'groups': op_summary(findings),
            'findings': findings}


def to_markdown(result: dict, limit: int) -> str:
    lines = [f"# tenferro-rs rows slower than the fastest reference (ratio >= {result['threshold']})", '']
    for run in result['runs']:
        where = ', '.join(x for x in (run['device'], run['blas'] and f"BLAS {run['blas']}") if x)
        counts = ', '.join(f'{n} {s}' for s, n in sorted(run['status'].items()))
        note = '' if run['has_reference'] else '; no reference backend, not compared'
        lines.append(f"- `{run['suite_id']}`: `{run['raw_run']}`, tenferro-rs `{run['tenferro_commit']}`"
                     f"{', ' + where if where else ''}; {counts}{note}")
    findings = result['findings']
    lines += ['', '## Slow rows grouped by operation', '',
              '| suite | op | tenferro backend | rows | max ratio | median ratio |', '|---|---|---|---|---|---|']
    lines += [f"| {g['suite_id']} | {g['op']} | {g['backend']} | {g['rows']} | {g['max_ratio']:.2f}x | "
              f"{g['median_ratio']:.2f}x |" for g in result['groups']] or ['| (none) | | | | | |']
    shown = f'; the {limit} highest ratios are listed, the rest are in the JSON file' if len(findings) > limit else ''
    lines += ['', '## Slow rows', '', f'{len(findings)} slow rows{shown}', '',
              '| suite | problem | threads | tenferro backend | median ms (IQR) | reference | median ms (IQR) | ratio |',
              '|---|---|---|---|---|---|---|---|']
    lines += [f"| {f['suite_id']} | {f['problem_id']} | {ms(f['threads'])} | {f['backend']} | "
              f"{f['median_ms']:.4g} ({ms(f['iqr_ms'])}) | {f['reference_backend']} | "
              f"{f['reference_median_ms']:.4g} ({ms(f['reference_iqr_ms'])}) | {f['ratio']:.2f}x |"
              for f in findings[:limit]] or ['| (none) | | | | | | | |']
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rows', type=Path, help='tenferro-bench-row/1 JSONL from scripts/ci/bench_rows.py')
    parser.add_argument('--threshold', type=float, default=1.5)
    parser.add_argument('--markdown-limit', type=int, default=60)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    result = collect(load(args.rows), args.threshold)
    args.json.write_text(json.dumps(result, indent=2) + '\n')
    args.markdown.write_text(to_markdown(result, args.markdown_limit))
    print(f"{len(result['findings'])} slow rows across {len(result['runs'])} runs")


if __name__ == '__main__':
    main()
