"""List GPU rows where a tenferro-rs backend is clearly slower than a reference.

Reads a RunPod GPU benchmark artifact (or a repository checkout) and compares
each measured tenferro backend with the fastest measured non-tenferro backend
of the same problem. A row is reported when the median ratio reaches the
threshold and the IQR-widened intervals do not overlap, so noise alone does
not produce a finding. Unsupported and failed rows are not timing evidence and
are never reported here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_rows(records: Path) -> list[dict]:
    return [json.loads(line) for line in records.read_text().splitlines() if line.strip()]


def measured(row: dict) -> bool:
    median = row.get('timing', {}).get('median_ms')
    return (row.get('status') == 'ok' and isinstance(median, (int, float)) and median > 0
            and row.get('verification', {}).get('status') == 'passed')


def slow_rows(records: Path, threshold: float) -> list[dict]:
    by_problem: dict[str, list[dict]] = {}
    for row in load_rows(records):
        if measured(row):
            by_problem.setdefault(row['problem_id'], []).append(row)
    findings = []
    for problem, rows in by_problem.items():
        references = [r for r in rows if not r['backend'].startswith('tenferro-')]
        if not references:
            continue
        ref = min(references, key=lambda r: r['timing']['median_ms'])
        ref_median = ref['timing']['median_ms']
        ref_high = ref_median + (ref['timing'].get('iqr_ms') or 0)
        for row in rows:
            if not row['backend'].startswith('tenferro-'):
                continue
            median = row['timing']['median_ms']
            low = median - (row['timing'].get('iqr_ms') or 0)
            ratio = median / ref_median
            if ratio >= threshold and low > ref_high:
                findings.append({
                    'suite_id': row['suite_id'], 'problem_id': problem, 'op': row.get('op'),
                    'backend': row['backend'], 'median_ms': median,
                    'iqr_ms': row['timing'].get('iqr_ms'),
                    'execution_path': row.get('execution', {}).get('execution_path'),
                    'reference_backend': ref['backend'], 'reference_median_ms': ref_median,
                    'reference_iqr_ms': ref['timing'].get('iqr_ms'), 'ratio': round(ratio, 3),
                })
    return findings


def collect(root: Path, threshold: float) -> dict:
    findings, runs = [], []
    for records in sorted(root.glob('data/results/nvidia-gpu/gpu/*/*/records.jsonl')):
        run = yaml.safe_load((records.parent / 'run.yaml').read_text()) or {}
        runs.append({
            'suite_id': run.get('suite_id'), 'raw_run': str(records.parent.relative_to(root)),
            'tenferro_commit': (run.get('tenferro_rs') or {}).get('commit'),
            'harness_commit': (run.get('harness') or {}).get('commit'),
            'device_name': (run.get('cuda') or {}).get('device_name'),
        })
        findings += [dict(f, raw_run=runs[-1]['raw_run']) for f in slow_rows(records, threshold)]
    findings.sort(key=lambda f: f['ratio'], reverse=True)
    return {'threshold': threshold, 'runs': runs, 'findings': findings}


def ms(value) -> str:
    return 'n/a' if value is None else f'{value:.4g}'


def to_markdown(result: dict) -> str:
    lines = [f"# tenferro-rs GPU rows slower than the fastest reference (ratio >= {result['threshold']})", '']
    for run in result['runs']:
        lines.append(f"- `{run['suite_id']}`: `{run['raw_run']}`, tenferro-rs `{run['tenferro_commit']}`, "
                     f"{run['device_name']}")
    lines += ['', '| suite | problem | tenferro backend | median ms (IQR) | reference | median ms (IQR) | ratio |',
              '|---|---|---|---|---|---|---|']
    for f in result['findings']:
        lines.append(f"| {f['suite_id']} | {f['problem_id']} | {f['backend']} | {f['median_ms']:.4g} "
                     f"({ms(f['iqr_ms'])}) | {f['reference_backend']} | {f['reference_median_ms']:.4g} "
                     f"({ms(f['reference_iqr_ms'])}) | {f['ratio']:.2f}x |")
    if not result['findings']:
        lines.append('| (none) | | | | | | |')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='artifact or checkout containing data/results/nvidia-gpu')
    parser.add_argument('--threshold', type=float, default=1.5)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root, args.threshold)
    args.json.write_text(json.dumps(result, indent=2) + '\n')
    args.markdown.write_text(to_markdown(result))
    print(f"{len(result['findings'])} slow rows across {len(result['runs'])} runs")


if __name__ == '__main__':
    main()
