"""Export raw benchmark runs as flat ``tenferro-bench-row/1`` JSONL rows.

Each suite stores results in its own format (CSV tables, per-sample JSONL,
nested GPU records). This module converts them into one row per
(suite, problem, threads, backend) measurement so reports, slow-row audits
and ad hoc analysis (jq, pandas) read a single format. The format is
documented in docs/benchmark-rows-format.md.

A collection bundle with an ``ingestion.json`` manifest (docs/result-ingestion.md)
is read through the manifest's raw files; other roots are scanned as raw-run
directories.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import statistics
import sys
from pathlib import Path

import yaml

SCHEMA = 'tenferro-bench-row/1'
TIMESTAMP = re.compile(r'^\d{8}_\d{6}$')
STATUSES = {'ok', 'unsupported', 'failed', 'skipped'}
# Lower bounds such as memcpy are not implementations of the operation.
BOUNDS = {'memcpy'}
CSV_TABLES = ('cpu_ops_t*.csv', 'cpu_fft_t*.csv', 'cpu_public_api_t*.csv')
PERMUTATION_OUTPUTS = '*_output_t*.jsonl'
SAMPLES = 'samples_t*.jsonl'
GPU_RECORDS = 'records.jsonl'
REQUIRED = ('schema', 'target_profile', 'suite_id', 'problem_id', 'op', 'threads', 'backend', 'role',
            'status', 'median_ms', 'iqr_ms', 'raw_run', 'tenferro_commit')


def role(backend: str) -> str:
    if backend.startswith('tenferro'):
        return 'tenferro'
    return 'bound' if backend in BOUNDS else 'reference'


def positive(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def spread(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    q1, median, q3 = statistics.quantiles(values, n=4)
    return statistics.median(values), q3 - q1


def csv_status(status: str) -> tuple[str, str | None]:
    head, _, reason = status.partition(':')
    head = head.strip()
    return (head if head in STATUSES else 'failed'), (reason.strip() or None)


def matching(files: list[Path], *patterns: str) -> list[Path]:
    return sorted(f for f in files if any(fnmatch.fnmatch(f.name, p) for p in patterns))


def csv_rows(files: list[Path]):
    for pattern in CSV_TABLES:
        for table in matching(files, pattern):
            with table.open(newline='') as handle:
                for row in csv.DictReader(handle):
                    status, reason = csv_status(row['status'])
                    median = positive(row.get('median_ms'))
                    yield {
                        'problem_id': '/'.join(row[k] for k in ('suite', 'benchmark', 'dtype', 'shape')),
                        'op': row['benchmark'], 'dtype': row['dtype'], 'shape': row['shape'],
                        'threads': int(row['threads']), 'backend': row['backend'],
                        'status': status if median or status != 'ok' else 'failed',
                        'median_ms': median, 'iqr_ms': positive(row.get('iqr_ms')) or (0.0 if median else None),
                        'note': reason or row.get('notes') or None,
                    }


def permutation_rows(files: list[Path]):
    for output in matching(files, PERMUTATION_OUTPUTS):
        for line in filter(str.strip, output.read_text().splitlines()):
            row = json.loads(line)
            median = positive(row.get('median_ms'))
            p25, p75 = positive(row.get('p25_ms')), positive(row.get('p75_ms'))
            status = row.get('status') if row.get('status') in STATUSES else 'failed'
            if status == 'ok' and (row.get('correctness') != 'passed' or median is None):
                status = 'failed'
            yield {
                'problem_id': f"{row['pattern_id']}/{row['dtype']}", 'op': row['pattern_id'],
                'dtype': row['dtype'], 'shape': 'x'.join(map(str, row.get('shape') or [])) or None,
                'threads': int(row['threads']), 'backend': row['backend'], 'status': status,
                'median_ms': median if status == 'ok' else None,
                'iqr_ms': (p75 - p25 if p25 and p75 else 0.0) if status == 'ok' else None,
                'note': row.get('notes'),
            }


def session_rows(files: list[Path]):
    for samples in matching(files, SAMPLES):
        for line in filter(str.strip, samples.read_text().splitlines()):
            row = json.loads(line)
            values = [v / 1e6 for v in row.get('samples_ns') or [] if isinstance(v, (int, float)) and v > 0]
            if row.get('status') == 'unsupported':
                status = 'unsupported'
            elif row.get('status') == 'passed' and row.get('correctness') == 'passed' and values:
                status = 'ok'
            else:
                status = 'failed'
            median, iqr = spread(values) if status == 'ok' else (None, None)
            provider = row['provider']
            yield {
                'problem_id': row['case_id'], 'op': row.get('operation') or row.get('workload'),
                'dtype': row.get('dtype'),
                'shape': f"n{row['n']}" if row.get('n') else None, 'threads': int(row['threads']),
                'backend': 'pytorch-cpu' if provider == 'pytorch' else f'tenferro-{provider}',
                'execution_path': (row.get('route') or {}).get('execution_path'),
                'status': status, 'median_ms': median, 'iqr_ms': iqr,
                'note': row.get('unsupported_reason'),
            }


def small_work_rows(files: list[Path]):
    for samples in matching(files, SAMPLES):
        for line in filter(str.strip, samples.read_text().splitlines()):
            row = json.loads(line)
            values = [s['elapsed_ns'] / s['iterations'] / 1e6 for s in row.get('samples') or []
                      if s.get('iterations') and s.get('elapsed_ns')]
            ok = row.get('status') == 'passed' and row.get('correctness_status') == 'passed' and values
            median, iqr = spread(values) if ok else (None, None)
            yield {
                'problem_id': row['id'], 'op': row.get('operation'), 'dtype': row.get('dtype'),
                'shape': json.dumps(row.get('shape')) if row.get('shape') is not None else None,
                'threads': int(row['threads']), 'backend': row.get('backend') or 'tenferro-rs',
                'execution_path': row.get('api_tier'), 'status': 'ok' if ok else 'failed',
                'median_ms': median, 'iqr_ms': iqr, 'note': None,
            }


def gpu_rows(files: list[Path]):
    for line in (line for records in matching(files, GPU_RECORDS)
                 for line in filter(str.strip, records.read_text().splitlines())):
        row = json.loads(line)
        timing, execution = row.get('timing') or {}, row.get('execution') or {}
        status = row.get('status') if row.get('status') in STATUSES else 'failed'
        if status == 'ok' and ((row.get('verification') or {}).get('status') != 'passed'
                               or not positive(timing.get('median_ms'))):
            status = 'failed'
        yield {
            'problem_id': row['problem_id'], 'op': row.get('op'), 'dtype': execution.get('dtype'),
            'shape': None, 'threads': None, 'backend': row['backend'],
            'execution_path': execution.get('execution_path'), 'status': status,
            'median_ms': timing.get('median_ms') if status == 'ok' else None,
            'iqr_ms': (timing.get('iqr_ms') or 0.0) if status == 'ok' else None,
            'note': execution.get('unsupported_reason') or (row.get('verification') or {}).get('reason'),
        }


ADAPTERS = {
    'cpu/permutation': permutation_rows,
    'cpu/session_matrix': session_rows,
    'cpu/small_work': small_work_rows,
}


def adapter(files: list[Path], suite_id: str):
    if suite_id in ADAPTERS:
        return ADAPTERS[suite_id]
    if matching(files, GPU_RECORDS):
        return gpu_rows
    if matching(files, *CSV_TABLES):
        return csv_rows
    return None


def run_metadata(run: Path) -> dict:
    for name in ('run.yaml', 'run_t1.yaml', 'run_t4.yaml'):
        if (run / name).is_file():
            return yaml.safe_load((run / name).read_text()) or {}
    return {}


def context(profile: str, suite_id: str, raw_run: str, timestamp: str, meta: dict, **overrides) -> dict:
    return {
        'schema': SCHEMA, 'target_profile': profile, 'suite_id': suite_id, 'raw_run': raw_run,
        'timestamp': timestamp,
        'tenferro_commit': (meta.get('tenferro_rs') or {}).get('commit'),
        'harness_commit': (meta.get('harness') or {}).get('commit'),
        'blas': (meta.get('blas') or {}).get('implementation'),
        'device': (meta.get('cuda') or {}).get('device_name'),
        **overrides,
    }


def bundle_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError(f'unsafe path in ingestion.json: {relative}')
    return root / relative


def export_bundle(root: Path, manifest: dict, since: str):
    """Yield (context, adapter, files) per manifest source, or (skip, None, None).

    Several suites of one collection can list the same raw file (cpu/cpu_ops and
    cpu/linalg_jvp_vjp share the cpu_ops CSV tables); a file is exported once,
    under the first suite that lists it.
    """
    seen = set()
    for suite in manifest['suites']:
        raw_run = suite['run_directory']
        timestamp = Path(raw_run).name
        if timestamp < since:
            continue
        digests = suite.get('files') or {}
        for source in suite['sources']:
            fresh = [f for f in source['raw_files'] if digests.get(f, f) not in seen]
            seen.update(digests.get(f, f) for f in fresh)
            files = [bundle_path(root, f) for f in fresh]
            if not files:
                continue
            meta = yaml.safe_load(bundle_path(root, source['metadata']).read_text()) or {}
            yield (context(manifest['target_profile'], suite['suite_id'], raw_run, timestamp, meta,
                           tenferro_commit=suite.get('tenferro_commit'),
                           harness_commit=manifest.get('harness_commit')),
                   adapter(files, suite['suite_id']), files)


def export_runs(root: Path, profile: str, since: str):
    for run in sorted(root.glob(f'data/results/{profile}/*/*/*')):
        if not run.is_dir() or not TIMESTAMP.match(run.name) or run.name < since:
            continue
        meta = run_metadata(run)
        suite_id = meta.get('suite_id') or f'{run.parent.parent.name}/{run.parent.name}'
        files = sorted(p for p in run.iterdir() if p.is_file())
        yield (context(profile, suite_id, str(run.relative_to(root)), run.name, meta),
               adapter(files, suite_id), files)


def export(root: Path, profile: str, since: str) -> tuple[list[dict], list[dict]]:
    """Return (rows, skipped runs) for raw runs of ``profile`` at or after ``since``."""
    manifest_path = root / 'ingestion.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if manifest and manifest.get('target_profile') == profile:
        sources = export_bundle(root, manifest, since)
    else:
        sources = export_runs(root, profile, since)
    rows, skipped = [], []
    for ctx, read, files in sources:
        if read is None:
            skip = {'suite_id': ctx['suite_id'], 'raw_run': ctx['raw_run'], 'reason': 'no adapter for this format'}
            if skip not in skipped:
                skipped.append(skip)
            continue
        for row in read(files):
            rows.append({**ctx, 'execution_path': None, **row, 'role': role(row['backend'])})
    return rows, skipped


def load(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [k for k in REQUIRED if k not in row]
        if row.get('schema') != SCHEMA or missing or row['status'] not in STATUSES:
            raise ValueError(f'{path}:{number}: not a {SCHEMA} row (missing {missing})')
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='artifact or checkout containing data/results/<profile>')
    parser.add_argument('--profile', required=True, help='target profile, for example amd-cpu or nvidia-gpu')
    parser.add_argument('--since', default='00000000_000000',
                        help='first raw-run timestamp to export, YYYYMMDD_HHMMSS (UTC)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not TIMESTAMP.match(args.since):
        parser.error('--since must be YYYYMMDD_HHMMSS')
    rows, skipped = export(args.root, args.profile, args.since)
    args.output.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    for run in skipped:
        print(f"warning: skipped {run['raw_run']} ({run['suite_id']}): {run['reason']}", file=sys.stderr)
    print(f'{len(rows)} rows from {len({r["raw_run"] for r in rows})} raw runs')


if __name__ == '__main__':
    main()
