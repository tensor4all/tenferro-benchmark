#!/usr/bin/env python3
"""Package only explicit CPU report sources; never scan for the latest raw run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import yaml

if __package__:
    from .cpu_ingestion_adapter import CPU_SUITES, build_source, identities, rebuild_cpu
    from .result_ingestion import digest, instant, safe_file
else:
    from cpu_ingestion_adapter import CPU_SUITES, build_source, identities, rebuild_cpu
    from result_ingestion import digest, instant, safe_file


def source_directories(report: str, profile: str) -> list[str]:
    pattern = rf'data/results/{re.escape(profile)}/cpu/[a-z0-9_.-]+/\d{{8}}_\d{{6}}'
    return sorted(set(re.findall(pattern, report)))


def package_cpu(root: Path, profile: str, head_sha: str, bundle: Path, suites=CPU_SUITES, threads=(1, 4)) -> dict:
    if profile not in ('amd-cpu', 'mac-cpu') or not re.fullmatch('[a-f0-9]{40}', head_sha):
        raise ValueError('Invalid CPU collection identity')
    if bundle.exists():
        raise ValueError('CPU bundle destination must not already exist')
    if not suites or len(suites) != len(set(suites)) or set(suites) - set(CPU_SUITES):
        raise ValueError('Invalid selected CPU suites')
    if not threads or any(type(thread) is not int or thread < 1 for thread in threads) or len(set(threads)) != len(threads):
        raise ValueError('Invalid selected thread counts')
    bundle.mkdir(parents=True)
    entries = []
    source = {'head_sha': head_sha}
    for name in suites:
        suite = 'cpu/' + name
        report_relative = f'result/{profile}/{suite}.md'
        report = safe_file(root, report_relative)
        directories = source_directories(report.read_text(), profile)
        if not directories and name == 'permutation':
            stamp_match = re.search(r'^- Timestamp: `([^`]+)`', report.read_text(), re.MULTILINE)
            if stamp_match:
                for candidate in sorted((root/f'data/results/{profile}/{suite}').glob('*/run.yaml')):
                    safe_file(root, str(candidate.relative_to(root)))
                    metadata = yaml.safe_load(candidate.read_text())
                    if instant(metadata['timestamp']) == instant(stamp_match.group(1)):
                        directories.append(str(candidate.parent.relative_to(root)))
            if len(directories) != 1:
                raise ValueError('Permutation report timestamp does not identify exactly one raw run')
        if not directories:
            raise ValueError(f'{suite}: report does not explicitly identify raw sources')
        expected_source_suite = 'einsum' if name in ('cpu_ops', 'linalg_jvp_vjp') else name
        if any(Path(directory).parent.name != expected_source_suite for directory in directories):
            raise ValueError(f'{suite}: report contains an unrelated raw source')
        descriptors = []
        for directory in directories:
            run = root / directory
            if name == 'einsum':
                paths = sorted(run.glob('*.log'))
                paths = [path for path in paths if path.name.startswith(('tenferro_trace_', 'tenferro_eager_', 'pytorch_cpu_', 'jax_cpu_', 'julia_omeinsum_'))]
                metadata_paths = [run/'run.yaml']
            elif name in ('cpu_ops', 'linalg_jvp_vjp'):
                paths = sorted(run.glob('cpu_ops_t*.csv'))
                metadata_paths = [run/'run.yaml']
            elif name == 'public_api':
                paths = sorted(run.glob('cpu_public_api_t*.csv'))
                metadata_paths = [run/'run.yaml']
            elif name == 'fft':
                paths = sorted(run.glob('cpu_fft_t*.csv'))
                metadata_paths = [run/'run.yaml']
            elif name == 'permutation':
                paths = sorted(run.glob('*_output_t*.jsonl'))
                metadata_paths = [run/'run.yaml']
            else:
                metadata_paths = sorted(run.glob('run_t*.yaml'))
                paths = sorted(run.glob('samples_t*.jsonl'))
            if not paths or not metadata_paths:
                raise ValueError(f'{suite}: missing raw data or metadata')
            for metadata in metadata_paths:
                selected_paths = paths
                if name in ('small_work', 'session_matrix', 'overhead_smoke'):
                    thread = re.fullmatch(r'run_t(\d+)', metadata.stem).group(1)
                    selected_paths = [run/f'samples_t{thread}.jsonl']
                descriptor = {'metadata': str(metadata.relative_to(root)),
                              'raw_files': [str(path.relative_to(root)) for path in selected_paths]}
                thread_metadata = sorted(run.glob('run_t*.yaml')) if name not in ('small_work', 'session_matrix', 'overhead_smoke') else [metadata]
                descriptor['thread_metadata'] = [str(path.relative_to(root)) for path in thread_metadata] or [str(metadata.relative_to(root))]
                if name == 'permutation':
                    match = re.search(r'^- Model: `([^`]+)`', report.read_text(), re.MULTILINE)
                    if not match:
                        raise ValueError('Permutation report lacks collected CPU model')
                    descriptor['cpu_model'] = match.group(1)
                document = build_source(root, descriptor, suite, profile, source)
                descriptors.append((descriptor, document))
        documents = [document for _, document in descriptors]
        stamp = max(document['timestamp'] for document in documents)
        from datetime import datetime
        run_directory = f'data/results/{profile}/{suite}/' + datetime.fromisoformat(stamp).strftime('%Y%m%d_%H%M%S')
        destination = bundle / run_directory
        destination.mkdir(parents=True)
        packed_sources = []
        for index, (descriptor, document) in enumerate(descriptors):
            original_run = root / Path(descriptor['metadata']).parent
            normalized = dict(descriptor)
            prefix = destination/'inputs'/str(index)
            # Copy regular files only; safe_file checks the source and every
            # ancestor before copying. Preserve diagnostics, never execute them.
            for path in sorted(original_run.rglob('*')):
                if path.is_file() or path.is_symlink():
                    relative = str(path.relative_to(root))
                    safe_file(root, relative)
                    target = prefix/path.relative_to(original_run)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            normalized['thread_metadata'] = [str((prefix/Path(path).name).relative_to(bundle)) for path in descriptor['thread_metadata']]
            normalized['metadata'] = str((prefix/Path(descriptor['metadata']).name).relative_to(bundle))
            normalized['raw_files'] = [str((prefix/Path(path).name).relative_to(bundle)) for path in descriptor['raw_files']]
            packed_sources.append(normalized)
        shutil.copyfile(report, destination/'report.md')
        target_report = bundle/report_relative
        target_report.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report, target_report)
        entry = {'suite_id': suite, 'timestamp': stamp, 'run_directory': run_directory,
                 'threads': [1] if name == 'overhead_smoke' else list(threads), 'sources': packed_sources,
                 'files': {str(path.relative_to(bundle)): digest(path) for path in destination.rglob('*') if path.is_file()}}
        document = rebuild_cpu(bundle, entry, profile, source)
        entry.update(tenferro_commit=document['tenferro_commit'], hardware=document['hardware'], expected_rows=identities(document))
        encoded = json.dumps(document, indent=2)+'\n'
        (destination/'report.summary.json').write_text(encoded)
        (bundle/f'result/{profile}/{suite}.summary.json').write_text(encoded)
        # Aggregate metadata is packaging metadata; original collection metadata
        # remains digest-covered under inputs/ and is always revalidated.
        (destination/'run.yaml').write_text(yaml.safe_dump({'target_profile': profile, 'suite_id': suite,
            'timestamp': stamp, 'tenferro_rs': {'commit': document['tenferro_commit']},
            'harness': {'commit': head_sha}, 'source_metadata': [item['metadata'] for item in packed_sources]}))
        files = [target_report, bundle/f'result/{profile}/{suite}.summary.json',
                 *sorted(path for path in destination.rglob('*') if path.is_file())]
        entry['files'] = {str(path.relative_to(bundle)): digest(path) for path in files}
        entries.append(entry)
    manifest = {'schema_version': 1, 'target_profile': profile, 'harness_commit': head_sha, 'suites': entries}
    (bundle/'ingestion.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--profile', choices=['amd-cpu', 'mac-cpu'], required=True)
    parser.add_argument('--harness-commit', required=True)
    parser.add_argument('--bundle-dir', type=Path, default=Path('_cpu_benchmark_artifact'))
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 4])
    args = parser.parse_args()
    package_cpu(args.root.resolve(), args.profile, args.harness_commit, args.bundle_dir.resolve(), threads=args.threads)


if __name__ == '__main__':
    main()
