"""Trusted CPU raw-data adapters shared by collection and ingestion.

No benchmark subprocess, torch/JAX import, or artifact code execution occurs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

if __package__:
    from .result_ingestion import instant, safe_file
else:
    from result_ingestion import instant, safe_file

SCRIPTS = str(Path(__file__).resolve().parents[1])
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
import report_summary

SELECTION_PATH = Path(__file__).resolve().parents[2] / 'benchmarks/cpu/ingestion-selection.json'


def validate_selection(document):
    plan = json.loads(SELECTION_PATH.read_text())
    if plan['schema_version'] != 1:
        raise ValueError('Unsupported CPU workload inventory')
    rows = sorted(identities(document))
    fingerprint = hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
    expected = plan['suites'].get(document['suite_id'])
    if expected != {'row_count': len(rows), 'identities_sha256': fingerprint}:
        raise ValueError(f"CPU workload selection does not match declared nightly inventory: {document['suite_id']}")
    if document['hardware'].get('threads') != plan['threads']:
        raise ValueError('CPU nightly thread selection mismatch')


CPU_SUITES = ('einsum', 'cpu_ops', 'linalg_jvp_vjp', 'fft', 'public_api',
              'small_work', 'session_matrix', 'permutation')


def identities(document):
    return [[row['case'], row.get('group', ''), row['backend']] for row in document['rows']]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_source(bundle: Path, descriptor: dict, suite: str, profile: str, source: dict) -> dict:
    metadata = yaml.safe_load(safe_file(bundle, descriptor['metadata']).read_text())
    raw = [safe_file(bundle, path) for path in descriptor['raw_files']]
    if not raw or len(raw) != len(set(raw)):
        raise ValueError('Missing or duplicate CPU raw inputs')
    if metadata.get('target_profile') != profile:
        raise ValueError('CPU metadata profile mismatch')
    allowed_suite = 'cpu/einsum' if suite in ('cpu/cpu_ops', 'cpu/linalg_jvp_vjp') else suite
    if metadata.get('suite_id') != allowed_suite:
        raise ValueError('CPU metadata suite mismatch')
    recorded = metadata.get('environment', {}).get('env', {}).get('BENCHMARK_COMMIT')
    recorded = recorded or metadata.get('harness', {}).get('commit')
    if recorded != source['head_sha']:
        raise ValueError('CPU metadata harness commit mismatch')
    if not metadata.get('tenferro_rs', {}).get('commit'):
        raise ValueError('CPU metadata measured revision is missing')
    features = metadata.get('tenferro_rs', {}).get('features', [])
    if profile == 'amd-cpu' and ('blas-mkl' not in features or metadata.get('blas', {}).get('implementation') != 'mkl'):
        raise ValueError('Standard Linux CPU publication requires recorded MKL collection')
    if profile == 'mac-cpu' and 'blas-accelerate' not in features:
        raise ValueError('Standard macOS CPU publication requires recorded Accelerate collection')
    metadata = dict(metadata, suite_id=suite)
    if suite == 'cpu/einsum':
        import format_results
        results, environment = {}, {}
        for path in raw:
            parsed, env = format_results.parse_log(str(path))
            if results.keys() & parsed.keys():
                raise ValueError('Duplicate raw einsum measurements')
            results.update(parsed)
            environment.update(env)
        document = format_results.build_summary(results, environment, metadata)
    elif suite in ('cpu/cpu_ops', 'cpu/public_api', 'cpu/linalg_jvp_vjp', 'cpu/fft'):
        if suite == 'cpu/linalg_jvp_vjp':
            from format_linalg_ad_results import build_summary
            if len(raw) != 1:
                raise ValueError('Each linalg source must identify one CSV')
            document = build_summary(raw[0], metadata)
        else:
            from format_cpu_ops_results import build_summary
            document = build_summary(raw, metadata)
        # Legacy formatters discarded CSV notes for unsupported rows. Restore
        # explanations from raw inputs without changing any measured values.
        notes = {}
        from format_cpu_ops_results import normalize_row
        for path in raw:
            with path.open(newline='') as stream:
                for row in csv.DictReader(stream):
                    (raw_suite, case, dtype, threads, shape), backend, _ = normalize_row(row)
                    group = f'dtype={dtype} / threads={threads} / shape={shape}'
                    if suite != 'cpu/linalg_jvp_vjp':
                        group = f'suite={raw_suite} / ' + group
                    status = row.get('status', '').strip()
                    reason = row.get('notes', '').strip() or row.get('reason', '').strip()
                    if status.startswith('unsupported:'):
                        reason = status.partition(':')[2].strip()
                    if reason:
                        notes[(case, group, backend)] = reason
        for row in document['rows']:
            if row['status'] != 'ok':
                reason = notes.get((row['case'], row.get('group', ''), row['backend']))
                if reason:
                    row['reason'] = reason
    elif suite == 'cpu/permutation':
        from format_permutation_results import build_summary
        records = [row for path in raw for row in read_jsonl(path)]
        # Avoid the formatter's current-host CPU probe during offline replay.
        # The CPU identity is read from the collection's cpu_info.md instead.
        document = build_summary(records, metadata, cpu_model=descriptor['cpu_model'])
        by_key = {(row['pattern_id'], int(row['threads']), row['backend']): row for row in records}
        for row in document['rows']:
            if row['status'] != 'ok':
                threads = int(row['group'].split(' / ')[0].split(': ')[1])
                record = by_key[(row['case'], threads, row['backend'])]
                if record.get('notes'):
                    row['reason'] = record['notes']
    elif suite in ('cpu/small_work', 'cpu/session_matrix'):
        records = [row for path in raw for row in read_jsonl(path)]
        rows = []
        for record in records:
            case = record['case_id']
            if suite == 'cpu/small_work':
                samples = record.get('samples', [])
                values = [sample['elapsed_ns'] / sample['iterations'] for sample in samples]
                verified = record.get('correctness_status') == 'passed'
                backend = 'tenferro-' + record['api_tier']
            else:
                count = record.get('operations_per_sample', 1)
                values = [value / count for value in record.get('samples_ns', [])]
                verified = record.get('correctness') == 'passed'
                backend = 'pytorch-cpu' if record.get('provider') == 'pytorch' else 'tenferro-' + str(record.get('provider', 'unknown'))
            status = 'ok' if verified and values else 'failed'
            reason = record.get('unsupported_reason') or record.get('reason') or record.get('error')
            if record.get('status') == 'unsupported' and reason:
                status = 'unsupported'
            if values and any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
                raise ValueError('Invalid CPU batch sample')
            quartiles = statistics.quantiles(values, n=4, method='inclusive') if len(values) > 1 else [values[0]]*3 if values else []
            row = report_summary.make_row(case, backend, statistics.median(values) if status == 'ok' else None,
                                          status=status, verified=verified,
                                          iqr=quartiles[2]-quartiles[0] if quartiles and status == 'ok' else None,
                                          group=f"threads={record['threads']}", reason=reason)
            row['timing_scope'] = record.get('timing_scope') or record.get('scope') or 'declared_per_case_in_raw_metadata'
            rows.append(row)
        header = report_summary.metadata_from_run_metadata(metadata)
        header['hardware']['threads'] = sorted({int(row['threads']) for row in records})
        document = report_summary.build_document(rows, **header, unit='ns')
        collection = metadata.get('collection', {})
        declared = collection.get('selected_cases')
        if not isinstance(declared, int) or len({record['case_id'] for record in records}) != declared:
            raise ValueError('CPU selected-case count does not match raw rows')
    else:
        raise ValueError(f'Unsupported CPU adapter: {suite}')
    recorded_threads = set()
    for relative in descriptor.get('thread_metadata', [descriptor['metadata']]):
        thread_metadata = yaml.safe_load(safe_file(bundle, relative).read_text())
        if (thread_metadata.get('tenferro_rs', {}).get('commit') != document['tenferro_commit']
                or thread_metadata.get('target_profile') != profile):
            raise ValueError('CPU per-thread metadata identity mismatch')
        value = thread_metadata.get('environment', {}).get('env', {}).get('RAYON_NUM_THREADS')
        if value is None:
            raise ValueError('CPU per-thread metadata lacks RAYON_NUM_THREADS')
        recorded_threads.add(int(value))
    if recorded_threads != set(document['hardware'].get('threads', [])):
        raise ValueError('CPU per-thread metadata does not match raw threads')
    if len(identities(document)) != len({tuple(row) for row in identities(document)}):
        raise ValueError('Duplicate CPU raw summary identities')
    if not document['rows']:
        raise ValueError('No CPU raw measurements')
    return document


def rebuild_cpu(bundle: Path, entry: dict, profile: str, source: dict) -> dict:
    descriptors = entry.get('sources')
    if not descriptors:
        raise ValueError('CPU manifest must declare raw sources')
    listed = entry['files']
    documents = []
    thread_sets = []
    for descriptor in descriptors:
        for relative in [descriptor['metadata'], *descriptor['raw_files'], *descriptor.get('thread_metadata', [])]:
            if relative not in listed:
                raise ValueError('CPU source file is not digest-covered')
        document = build_source(bundle, descriptor, entry['suite_id'], profile, source)
        documents.append(document)
        thread_sets.append(set(document['hardware'].get('threads', [])))
    if any(thread_sets[i] & thread_sets[j] for i in range(len(thread_sets)) for j in range(i)):
        raise ValueError('Duplicate thread counts across CPU sources')
    if len({doc['tenferro_commit'] for doc in documents}) != 1:
        raise ValueError('CPU sources mix measured revisions')
    if len({json.dumps({k:v for k,v in doc['hardware'].items() if k != 'threads'}, sort_keys=True) for doc in documents}) != 1:
        raise ValueError('CPU sources mix hardware')
    document = report_summary.merge_thread_documents(documents) if len(documents) > 1 else documents[0]
    if set(document['hardware'].get('threads', [])) != set(entry['threads']):
        raise ValueError('CPU thread coverage differs from declared selection')
    document['suite_id'] = entry['suite_id']
    validate_selection(document)
    return document
