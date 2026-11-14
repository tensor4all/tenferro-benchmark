#!/usr/bin/env python3
"""Package produced outputs after collection, including failed collections.

Also used for recovery of an exact run's diagnostic artifact. Never execute
artifact code or choose stale reports as substitutes for unproduced suites.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import yaml

if __package__:
    from .bundle_cpu_results import package_cpu
    from .cpu_ingestion_adapter import CPU_SUITES
    from .check_gpu_run import DEFAULT_SUITES, planned_rows, routine_arm_gaps
    from .result_ingestion import digest, instant, safe_file, validate_bundle
else:
    from bundle_cpu_results import package_cpu
    from cpu_ingestion_adapter import CPU_SUITES
    from check_gpu_run import DEFAULT_SUITES, planned_rows, routine_arm_gaps
    from result_ingestion import digest, instant, safe_file, validate_bundle

ROOT = Path(__file__).resolve().parents[2]
GPU_BACKENDS = 'tenferro-cuda-trace,tenferro-cuda-eager,pytorch-cuda'
#: Directories inside a run that hold regenerable inputs and caches. A weekly run
#: produces about 4.4 GB of them and about 20 MB of actual results, so shipping
#: them pays twice: the pod uploads them inside its billed lifetime and GitHub
#: stores them for the artifact's retention. Their inventory (names and sizes) is
#: recorded instead, which is what a reproduction needs.
DERIVED_DIRECTORIES = frozenset({'fixtures', 'cache'})


def derived_asset(relative: str) -> bool:
    return any(part in DERIVED_DIRECTORIES for part in Path(relative).parts)


def _inventory(entries) -> tuple[int, int, str]:
    ordered = sorted(entries)
    payload = '\n'.join(f'{name}\t{size}' for name, size in ordered)
    return len(ordered), sum(size for _, size in ordered), hashlib.sha256(payload.encode()).hexdigest()


def current_runs(root, profile, suite, source):
    runs = []
    base = root / f'data/results/{profile}/{suite}'
    for directory in sorted(base.glob('*')):
        try:
            stamp = datetime.strptime(directory.name, '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if (instant(source['run_started_at']).replace(microsecond=0) <= stamp <= instant(source['created_at'])
                and not list(directory.glob('run*.yaml'))):
            raise ValueError(f'{suite}: current collection outputs are missing run metadata')
    for path in sorted(base.glob('*/run*.yaml')):
        metadata = yaml.safe_load(safe_file(root, str(path.relative_to(root))).read_text())
        if instant(source['run_started_at']) <= instant(metadata['timestamp']) <= instant(source['created_at']):
            runs.append(path.parent)
    return sorted(set(runs))


class NoSuiteOutputs(Exception):
    """This suite produced no run directory in the selected attempt."""


def package_gpu_suite(root: Path, profile: str, output: Path, suite_file: str, source: dict,
                      timestamp: str | None, backends: str) -> tuple[str, dict, str, list[str]]:
    """Copy and declare one GPU suite, raising when this suite cannot package.

    A suite is packaged whole or not at all: its files are copied, its row
    inventory is derived from the plan, and only then is it declared.  A suite
    that raises leaves nothing behind, so the caller can record it as
    unpublished and keep the suites that did produce their outputs.
    """
    declaration = yaml.safe_load(safe_file(ROOT, suite_file.strip()).read_text())
    suite = declaration['suite_id']
    if not suite.startswith('gpu/'):
        raise ValueError('Not a GPU publication suite')
    runs = current_runs(root, profile, suite, source)
    if timestamp and any(run.name != timestamp for run in runs):
        raise ValueError(f'{suite}: inconsistent or ambiguous collection timestamp')
    if not runs:
        raise NoSuiteOutputs(suite)
    if len(runs) != 1:
        raise ValueError(f'{suite}: ambiguous collection outputs in selected run attempt')
    run = runs[0]
    stamp = run.name
    if timestamp and stamp != timestamp:
        raise ValueError('GPU suites have inconsistent collection timestamps')
    unbacked = unbacked_medians(run)
    if unbacked:
        raise ValueError(f'{suite}: no usable sample backs {", ".join(unbacked)}')
    run_relative = str(run.relative_to(root))
    target = output / run_relative
    excluded = []
    for path in sorted(run.rglob('*')):
        if path.is_file() or path.is_symlink():
            relative = str(path.relative_to(root))
            safe_file(root, relative)
            if derived_asset(relative):
                excluded.append((str(path.relative_to(run)), path.stat().st_size))
                continue
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    for filename, suffix in (('report.md', '.md'), ('report.summary.json', '.summary.json')):
        original = safe_file(root, run_relative + '/' + filename)
        destination = output / f'result/{profile}/{suite}{suffix}'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    # The routine selection owns which rows a suite must produce; a run that
    # predates it falls back to the suite declaration.
    expected = planned_rows(run, suite)
    if expected is None:
        declared = declaration.get('backends', backends.split(','))
        # A suite that declares its own arms and a run that recorded no plan is
        # expected to carry exactly those arms. The workflow's runtime arm set
        # is a different contract: the throughput lane declares only
        # `tenferro-cuda-trace-deferred-batch`, which the runtime set never
        # contains, so intersecting first planned zero rows and refused every
        # throughput bundle (run 43553702617).
        selected = [backend for backend in backends.split(',') if backend in declared]
        expected = {(problem['id'], backend) for problem in declaration['problems']
                    for backend in (selected or list(declared))}
    expected = sorted(expected)
    # Recorded, and refused at ingestion: a run that dropped a routine arm
    # is not the weekly matrix, however complete its own rows look.
    gaps = routine_arm_gaps(run, suite)
    files = [output / f'result/{profile}/{suite}.summary.json',
             output / f'result/{profile}/{suite}.md']
    files += sorted(path for path in target.rglob('*') if path.is_file())
    count, total_bytes, inventory_digest = _inventory(excluded)
    entry = {'suite_id': suite, 'expected_rows': expected,
             'files': {str(path.relative_to(output)): digest(path) for path in files},
             'excluded_derived': {'files': count, 'bytes': total_bytes,
                                  'sha256': inventory_digest}}
    return suite, entry, stamp, gaps


def unbacked_medians(run: Path) -> list[str]:
    """Cases whose recorded samples contradict the median they publish.

    `timed_runs >= 1` says a row was measured; it does not say the samples were
    usable. A row that lists samples but whose `sampling.batch_ms` holds a
    non-finite, non-positive or malformed value still carries a median, and
    publishing it would put a number on a page that no measurement supports.

    A row with no sampling inventory (or an empty one) is not rejected here: the
    accounting files that as `raw_samples_unavailable`, not as a failure, and
    several collectors legitimately record only the summary statistics. What
    matters is that no *stated* sample contradicts the number.
    """
    records = run / 'records.jsonl'
    if not records.is_file():
        return []
    unbacked = []
    for line in records.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('status') != 'ok':
            continue
        try:
            notes = json.loads((row.get('execution') or {}).get('notes') or '{}')
            sampling = notes['sampling']
        except (KeyError, TypeError, ValueError):
            # No sampling inventory at all is the accounting's
            # `raw_samples_unavailable` category, not a bad measurement.
            continue
        if not isinstance(sampling, dict) or 'batch_ms' not in sampling:
            continue
        samples = sampling['batch_ms']
        if not isinstance(samples, list) or not samples:
            # An explicit `null` or empty batch says the row has no samples;
            # that is not the same statement as "this collector records none".
            unbacked.append(str(row.get('problem_id') or row.get('case') or '?'))
            continue
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) or value <= 0 for value in samples):
            unbacked.append(str(row.get('problem_id') or row.get('case') or '?'))
    return sorted(set(unbacked))


def discard_gpu_suite(output: Path, suite: str, profile: str = 'nvidia-gpu') -> None:
    """Remove a suite's partial copies so the bundle declares only packaged suites."""
    shutil.rmtree(output / 'data/results' / profile / suite, ignore_errors=True)
    for suffix in ('.md', '.summary.json'):
        (output / f'result/{profile}/{suite}{suffix}').unlink(missing_ok=True)


def drop_unpublishable_suites(manifest: dict, output: Path, profile: str, source: dict) -> None:
    """Drop the suites the bundle contract rejects, one at a time.

    A failure is attributed by retrying each suite alone, not by reading the
    error text: a message that does not name its suite would otherwise withhold
    every suite in the bundle. One suite is always left in place; the final
    bundle validation decides whether it can publish at all.
    """
    while len(manifest['suites']) > 1:
        before = len(manifest['suites'])
        for entry in list(manifest['suites']):
            if len(manifest['suites']) == 1:
                return
            suite = entry['suite_id']
            try:
                validate_bundle(output, profile, source, only_suite=suite)
                continue
            except Exception as error:
                reason = str(error)
                print(f'{suite}: not published: {reason}', file=sys.stderr)
            manifest['suites'] = [item for item in manifest['suites']
                                  if item['suite_id'] != suite]
            manifest['missing_suites'] = sorted(set(manifest['missing_suites']) | {suite})
            manifest.setdefault('missing_reasons', {})[suite] = reason
            # The bundle-wide veto for a narrowed routine arm belongs to the
            # suite that lost the arm: leaving it behind rejects the suites that
            # are still published.
            narrowed = manifest.get('narrowed_routine_arms')
            if narrowed:
                narrowed.pop(suite, None)
                if not narrowed:
                    del manifest['narrowed_routine_arms']
            discard_gpu_suite(output, suite)
            (output / 'publication.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if len(manifest['suites']) == before:
            return


def prepare_bundle(root: Path, profile: str, source: dict, output: Path,
                   *, suites='all', backends=GPU_BACKENDS, timestamp=None):
    """Validate all current outputs; missing suites are recorded, not replaced."""
    if profile != 'nvidia-gpu':
        produced, missing = [], []
        for name in CPU_SUITES:
            raw_suite = 'einsum' if name in ('cpu_ops', 'linalg_jvp_vjp') else name
            if current_runs(root, profile, 'cpu/' + raw_suite, source):
                produced.append(name)
            else:
                missing.append('cpu/' + name)
        if not produced:
            raise ValueError('No CPU collection outputs in the selected run attempt')
        manifest = package_cpu(root, profile, source['head_sha'], output, suites=produced)
        manifest['missing_suites'] = missing
        (output / 'ingestion.json').write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        if suites == 'perf-issues-readback':
            raise ValueError('Manual readback is not a publication collection')
        suite_files = DEFAULT_SUITES if suites == 'all' else suites
        entries, missing, narrowed, unpublished = [], [], {}, {}
        output.mkdir(parents=True, exist_ok=False)
        for suite_file in suite_files.split(','):
            # One suite that cannot be packaged (or whose own accounting is
            # incomplete) is recorded as unpublished. It never costs the other
            # suites the results they did produce.
            suite_decl = yaml.safe_load(safe_file(ROOT, suite_file.strip()).read_text())
            suite = suite_decl.get('suite_id', suite_file.strip())
            try:
                suite, entry, stamp, gaps = package_gpu_suite(root, profile, output, suite_file, source,
                                                             timestamp, backends)
            except NoSuiteOutputs:
                missing.append(suite)
                unpublished[suite] = 'no collection output in the selected run attempt'
                continue
            except (ValueError, OSError, KeyError) as error:
                discard_gpu_suite(output, suite)
                missing.append(suite)
                unpublished[suite] = str(error)
                print(f'{suite}: not published: {error}', file=sys.stderr)
                continue
            timestamp = timestamp or stamp
            entries.append(entry)
            if gaps:
                narrowed[suite] = gaps
        if not entries:
            raise ValueError('No GPU collection outputs in the selected run attempt')
        manifest = {'schema_version': 2, 'timestamp': timestamp, 'suites': entries, 'missing_suites': missing}
        if narrowed:
            manifest['narrowed_routine_arms'] = narrowed
        if unpublished:
            # The stderr line is not the record: the bundle names why a suite it
            # left out was left out, beside the list of suites it left out.
            manifest['missing_reasons'] = unpublished
        (output / 'publication.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # One suite whose artifacts fail the publication contract must not cost
        # the week the suites that pass it. A failure is attributed by retrying
        # each suite alone, not by reading the error text: a message that does not
        # name its suite would otherwise withhold every suite in the bundle.
        try:
            drop_unpublishable_suites(manifest, output, profile, source)
        except BaseException:
            # Anything that escapes after the manifest exists -- a failed
            # rewrite included -- leaves no file the upload step could accept.
            (output / 'publication.json').unlink(missing_ok=True)
            raise
    # No manifest from a failed validation may authorize a results upload.
    try:
        return validate_bundle(output, profile, source)
    except Exception:
        (output / ('publication.json' if profile == 'nvidia-gpu' else 'ingestion.json')).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['amd-cpu', 'mac-cpu', 'nvidia-gpu'], required=True)
    parser.add_argument('--harness-commit', required=True)
    parser.add_argument('--started-at', required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--suites', default='all')
    parser.add_argument('--backends', default=GPU_BACKENDS)
    parser.add_argument('--timestamp')
    args = parser.parse_args()
    source = {'head_sha': args.harness_commit, 'run_started_at': args.started_at,
              'created_at': datetime.now(timezone.utc).isoformat()}
    validated = prepare_bundle(args.root.resolve(), args.profile, source, args.bundle_dir.resolve(),
                               suites=args.suites, backends=args.backends, timestamp=args.timestamp)
    print(json.dumps({receipt['suite_id']: {'status': receipt['status'],
                      'case_outcomes': receipt['case_outcomes'], 'missing_suites': receipt['missing_suites']}
                      for receipt, _ in validated}, indent=2))


if __name__ == '__main__':
    main()
