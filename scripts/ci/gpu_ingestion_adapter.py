"""Validate the existing GPU publication contract without publishing overlays."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

if __package__:
    from .check_gpu_run import validate_records
    from .result_ingestion import digest, instant, safe_file
else:
    from check_gpu_run import validate_records
    from result_ingestion import digest, instant, safe_file


def adapt_publication(bundle: Path, source: dict) -> dict:
    """Return ingestion entries for an existing schema-2 GPU artifact.

    The source bundle is never rewritten; schema adaptation is in memory.
    Every original file digest and raw record is checked first.
    """
    publication = json.loads(safe_file(bundle, 'publication.json').read_text())
    if publication.get('schema_version') != 2:
        raise ValueError('Only GPU publication schema 2 with summaries is supported')
    entries = []
    for original in publication['suites']:
        suite = original['suite_id']
        stamp = publication['timestamp']
        run = f'data/results/nvidia-gpu/{suite}/{stamp}'
        report = f'result/nvidia-gpu/{suite}.md'
        summary_path = f'result/nvidia-gpu/{suite}.summary.json'
        files = original['files']
        required = {report, summary_path, run+'/records.jsonl', run+'/run.yaml',
                    run+'/report.md', run+'/report.summary.json'}
        if not required.issubset(files):
            raise ValueError('GPU publication is missing report, summary, or raw data')
        for relative, expected in files.items():
            if digest(safe_file(bundle, relative)) != expected:
                raise ValueError(f'GPU publication digest mismatch: {relative}')
        summary = json.loads(safe_file(bundle, summary_path).read_text())
        expected_rows = original['expected_rows']
        if len(expected_rows) != len({tuple(row) for row in expected_rows}):
            raise ValueError('Duplicate expected GPU rows')
        validate_records(safe_file(bundle, run+'/records.jsonl'), suite,
                         {tuple(row) for row in expected_rows})
        metadata = yaml.safe_load(safe_file(bundle, run+'/run.yaml').read_text())
        if (metadata.get('target_profile') != 'nvidia-gpu' or metadata.get('suite_id') != suite
                or metadata.get('tenferro_rs', {}).get('commit') != summary.get('tenferro_commit')
                or instant(metadata['timestamp']) != instant(summary['timestamp'])):
            raise ValueError('GPU summary does not match run metadata')
        if instant(summary['timestamp']).strftime('%Y%m%d_%H%M%S') != stamp:
            raise ValueError('GPU collection timestamp does not match run directory')
        if metadata.get('cuda', {}).get('device_name') != summary.get('hardware', {}).get('gpu'):
            raise ValueError('GPU hardware does not match run metadata')
        # Import only trusted checkout formatter code, never bundle code.
        scripts = str(Path(__file__).resolve().parents[1])
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        if suite in ('gpu/linalg_jvp_vjp', 'gpu/linalg_ad_latency'):
            from format_gpu_linalg_ad_results import build_summary
        else:
            from format_gpu_results import build_summary
        records = [json.loads(line) for line in safe_file(bundle, run+'/records.jsonl').read_text().splitlines() if line.strip()]
        rebuilt = build_summary(records, metadata)
        if rebuilt != summary:
            raise ValueError('GPU summary differs from raw records')
        # Attribution is established by the API-selected run. Some legacy GPU
        # metadata did not record a harness SHA; a conflicting recorded SHA fails.
        recorded = metadata.get('environment', {}).get('env', {}).get('BENCHMARK_COMMIT')
        if recorded and recorded != source['head_sha']:
            raise ValueError('GPU recorded harness differs from source run')
        entries.append({'suite_id': suite, 'timestamp': summary['timestamp'],
                        'tenferro_commit': summary['tenferro_commit'], 'hardware': summary['hardware'],
                        'run_directory': run, 'adapter': 'gpu_records',
                        'expected_rows': [[row['case'], row.get('group', ''), row['backend']] for row in summary['rows']],
                        'files': files})
    if not entries:
        raise ValueError('No GPU suites in publication')
    return {'schema_version': 1, 'target_profile': 'nvidia-gpu',
            'harness_commit': source['head_sha'], 'suites': entries}
