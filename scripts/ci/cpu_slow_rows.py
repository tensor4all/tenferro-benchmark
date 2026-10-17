"""List CPU rows where a tenferro-rs backend is clearly slower than a reference.

Reads a nightly Linux CPU benchmark artifact (or a repository checkout) and
compares each measured tenferro backend with the fastest measured reference
backend of the same suite, problem and thread count. A row is reported when
the median ratio reaches the threshold and the IQR-widened intervals do not
overlap, as in gpu_slow_rows.py. A row whose IQR exceeds its median is below
the timer's useful resolution and takes no part in a comparison. Only raw runs at or after --since are read,
because the artifact also carries the repository's older raw runs.

The suites store results in three formats:

- cpu_ops/einsum, cpu/fft and cpu/public_api: ``*_t<N>_*.csv`` tables;
- cpu/permutation: ``*_output_t<N>.jsonl`` with p25/p75 quantiles;
- cpu/session_matrix: ``samples_t<N>.jsonl`` with per-sample nanoseconds,
  where the ``blas``/``faer`` providers are tenferro-rs and ``pytorch`` is the
  reference.

cpu/small_work and cpu/overhead_smoke measure tenferro-rs alone, so they have
no reference to compare against and are listed as unanalysed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import yaml

from scripts.ci.gpu_slow_rows import ms

TIMESTAMP = re.compile(r'^\d{8}_\d{6}$')
CSV_TABLES = ('cpu_ops_t*.csv', 'cpu_fft_t*.csv', 'cpu_public_api_t*.csv')
# memcpy is a bandwidth bound, not an implementation of the operation.
NOT_REFERENCES = {'memcpy'}
NO_REFERENCE_SUITES = {
    'cpu/small_work': 'tenferro-rs only; no reference backend',
    'cpu/overhead_smoke': 'ns/op and allocs/op trend view; no reference backend',
}


def number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def csv_rows(run: Path, suite_id: str) -> list[dict]:
    rows = []
    for pattern in CSV_TABLES:
        for table in sorted(run.glob(pattern)):
            with table.open(newline='') as handle:
                for row in csv.DictReader(handle):
                    median = number(row.get('median_ms'))
                    if row.get('status') != 'ok' or median is None:
                        continue
                    rows.append({
                        'suite_id': suite_id, 'threads': int(row['threads']), 'backend': row['backend'],
                        'op': row['benchmark'],
                        'problem': '/'.join(row[k] for k in ('suite', 'benchmark', 'dtype', 'shape')),
                        'median_ms': median, 'iqr_ms': number(row.get('iqr_ms')),
                    })
    return rows


def permutation_rows(run: Path, suite_id: str) -> list[dict]:
    rows = []
    for output in sorted(run.glob('*_output_t*.jsonl')):
        for line in output.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            median = number(row.get('median_ms'))
            if row.get('status') != 'ok' or row.get('correctness') != 'passed' or median is None:
                continue
            p25, p75 = number(row.get('p25_ms')), number(row.get('p75_ms'))
            rows.append({
                'suite_id': suite_id, 'threads': int(row['threads']), 'backend': row['backend'],
                'op': row['pattern_id'], 'problem': f"{row['pattern_id']}/{row['dtype']}", 'median_ms': median,
                'iqr_ms': p75 - p25 if p25 is not None and p75 is not None else None,
            })
    return rows


def session_rows(run: Path, suite_id: str) -> list[dict]:
    rows = []
    for samples in sorted(run.glob('samples_t*.jsonl')):
        for line in samples.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            values = [v / 1e6 for v in row.get('samples_ns') or [] if isinstance(v, (int, float))]
            if row.get('status') != 'passed' or row.get('correctness') != 'passed' or not values:
                continue
            q1, _, q3 = statistics.quantiles(values, n=4) if len(values) > 1 else (values[0],) * 3
            provider = row['provider']
            rows.append({
                'suite_id': suite_id, 'threads': int(row['threads']), 'op': row['operation'],
                'problem': row['case_id'],
                'backend': 'pytorch-cpu' if provider == 'pytorch' else f'tenferro-{provider}',
                'median_ms': statistics.median(values), 'iqr_ms': q3 - q1,
            })
    return rows


def suite_rows(run: Path, suite_id: str) -> list[dict]:
    if suite_id == 'cpu/permutation':
        return permutation_rows(run, suite_id)
    if suite_id == 'cpu/session_matrix':
        return session_rows(run, suite_id)
    return csv_rows(run, suite_id)


def stable(row: dict) -> bool:
    return row['iqr_ms'] is None or row['iqr_ms'] <= row['median_ms']


def slow_rows(rows: list[dict], threshold: float) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in filter(stable, rows):
        groups.setdefault((row['suite_id'], row['problem'], row['threads']), []).append(row)
    findings = []
    for group in groups.values():
        references = [r for r in group if not r['backend'].startswith('tenferro')
                      and r['backend'] not in NOT_REFERENCES]
        if not references:
            continue
        ref = min(references, key=lambda r: r['median_ms'])
        ref_high = ref['median_ms'] + (ref['iqr_ms'] or 0)
        for row in group:
            if not row['backend'].startswith('tenferro'):
                continue
            ratio = row['median_ms'] / ref['median_ms']
            if ratio >= threshold and row['median_ms'] - (row['iqr_ms'] or 0) > ref_high:
                findings.append(dict(row, reference_backend=ref['backend'],
                                     reference_median_ms=ref['median_ms'],
                                     reference_iqr_ms=ref['iqr_ms'], ratio=round(ratio, 3)))
    return findings


def run_metadata(run: Path) -> dict:
    for name in ('run.yaml', 'run_t1.yaml', 'run_t4.yaml'):
        if (run / name).is_file():
            return yaml.safe_load((run / name).read_text()) or {}
    return {}


def collect(root: Path, threshold: float, since: str) -> dict:
    findings, runs, unanalysed = [], [], []
    for run in sorted(root.glob('data/results/amd-cpu/cpu/*/*')):
        if not run.is_dir() or not TIMESTAMP.match(run.name) or run.name < since:
            continue
        meta = run_metadata(run)
        suite_id = meta.get('suite_id') or f'cpu/{run.parent.name}'
        raw_run = str(run.relative_to(root))
        if suite_id in NO_REFERENCE_SUITES:
            unanalysed.append({'suite_id': suite_id, 'raw_run': raw_run,
                               'reason': NO_REFERENCE_SUITES[suite_id]})
            continue
        rows = suite_rows(run, suite_id)
        runs.append({
            'suite_id': suite_id, 'raw_run': raw_run, 'measured_rows': len(rows),
            'tenferro_commit': (meta.get('tenferro_rs') or {}).get('commit'),
            'harness_commit': (meta.get('harness') or {}).get('commit'),
            'blas': (meta.get('blas') or {}).get('implementation'),
        })
        findings += [dict(f, raw_run=raw_run) for f in slow_rows(rows, threshold)]
    findings.sort(key=lambda f: f['ratio'], reverse=True)
    return {'threshold': threshold, 'since': since, 'runs': runs,
            'unanalysed': unanalysed, 'findings': findings}


def op_summary(findings: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for f in findings:
        groups.setdefault((f['suite_id'], f['op'], f['backend']), []).append(f['ratio'])
    summary = [{'suite_id': s, 'op': op, 'backend': b, 'rows': len(r), 'max_ratio': max(r),
                'median_ratio': statistics.median(r)} for (s, op, b), r in groups.items()]
    return sorted(summary, key=lambda g: g['max_ratio'], reverse=True)


def to_markdown(result: dict, limit: int) -> str:
    lines = [f"# tenferro-rs CPU rows slower than the fastest reference (ratio >= {result['threshold']})", '']
    for run in result['runs']:
        lines.append(f"- `{run['suite_id']}`: `{run['raw_run']}`, {run['measured_rows']} measured rows, "
                     f"tenferro-rs `{run['tenferro_commit']}`, BLAS {run['blas']}")
    for run in result['unanalysed']:
        lines.append(f"- `{run['suite_id']}`: `{run['raw_run']}`, not analysed ({run['reason']})")
    findings = result['findings']
    lines += ['', '## Slow rows grouped by operation', '',
              '| suite | op | tenferro backend | rows | max ratio | median ratio |', '|---|---|---|---|---|---|']
    lines += [f"| {g['suite_id']} | {g['op']} | {g['backend']} | {g['rows']} | {g['max_ratio']:.2f}x | "
              f"{g['median_ratio']:.2f}x |" for g in op_summary(findings)]
    if not findings:
        lines.append('| (none) | | | | | |')
    lines += ['', '## Slow rows', '', f'{len(findings)} slow rows' + (f'; the {limit} highest ratios are listed below, the rest '
                                                 'are in the JSON file' if len(findings) > limit else ''),
              '', '| suite | problem | threads | tenferro backend | median ms (IQR) | reference | median ms (IQR) | ratio |',
              '|---|---|---|---|---|---|---|---|']
    for f in findings[:limit]:
        lines.append(f"| {f['suite_id']} | {f['problem']} | {f['threads']} | {f['backend']} | "
                     f"{f['median_ms']:.4g} ({ms(f['iqr_ms'])}) | {f['reference_backend']} | "
                     f"{f['reference_median_ms']:.4g} ({ms(f['reference_iqr_ms'])}) | {f['ratio']:.2f}x |")
    if not findings:
        lines.append('| (none) | | | | | | | |')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='artifact or checkout containing data/results/amd-cpu')
    parser.add_argument('--since', required=True, help='first raw-run timestamp to read, YYYYMMDD_HHMMSS (UTC)')
    parser.add_argument('--threshold', type=float, default=1.5)
    parser.add_argument('--markdown-limit', type=int, default=60)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    if not TIMESTAMP.match(args.since):
        parser.error('--since must be YYYYMMDD_HHMMSS')
    result = collect(args.root, args.threshold, args.since)
    args.json.write_text(json.dumps(result, indent=2) + '\n')
    args.markdown.write_text(to_markdown(result, args.markdown_limit))
    print(f"{len(result['findings'])} slow rows across {len(result['runs'])} runs")


if __name__ == '__main__':
    main()
