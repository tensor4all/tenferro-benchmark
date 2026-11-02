"""The collected binaries the selected GPU suites need on the pod.

A suite's `driver:` in the routine manifest decides which collector binary runs
it, so this is derived from the selection rather than from a list of suite names
beside the transfer step: a driver that ships no binary has to fail here, before
a paid suite is collected as `missing` rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

MANIFEST = Path('benchmarks/gpu/manifests/routine.yaml')
READBACK = 'perf-issues-readback'
READBACK_BINARY = 'benchmark_gpu_perf_issues'
DEFAULT_DRIVER = 'gpu_suite'
# The driver -> binary coupling of scripts/run_gpu_suite.sh. A driver missing
# here is refused instead of guessed, so a new collector cannot be silently
# under-delivered.
DRIVER_BINARIES = {
    'gpu_suite': 'benchmark_gpu_rust',
    'gpu_linalg_ad': 'benchmark_gpu_linalg_ad',
    'gpu_permutation': 'benchmark_gpu_permutation',
    'gpu_public_api': 'benchmark_gpu_public_api',
}


def required_binaries(selection, manifest=MANIFEST):
    """Binary -> the selected suites that require it.

    `selection` is the workflow's suite input: `all`, a comma-separated list of
    routine manifest suite files, or `perf-issues-readback` (which ships its own
    readback binary instead of the routine collectors). A suite that is not in
    the manifest, or a driver without a known binary, raises instead of being
    skipped.
    """
    selection = (selection or 'all').strip()
    if selection == READBACK:
        return {READBACK_BINARY: [READBACK]}
    suites = yaml.safe_load(Path(manifest).read_text())['suites']
    drivers = {spec['file']: spec.get('driver') or DEFAULT_DRIVER for spec in suites.values()}
    selected = (list(drivers) if selection == 'all'
                else [path.strip() for path in selection.split(',') if path.strip()])
    unknown = [path for path in selected if path not in drivers]
    if unknown:
        raise ValueError(f'GPU suites absent from {manifest}: {unknown}')
    required = {}
    for path in selected:
        driver = drivers[path]
        if driver not in DRIVER_BINARIES:
            raise ValueError(f'unknown GPU suite driver {driver!r} for {path}')
        required.setdefault(DRIVER_BINARIES[driver], []).append(path)
    return required


def check(required, directory):
    """Every required binary, or a message naming it and the suites behind it."""
    missing = []
    for binary, suites in required.items():
        path = Path(directory) / binary
        if path.is_file():
            print(f'{path} present for {", ".join(suites)}')
        else:
            missing.append(f'{path} (required by {", ".join(suites)})')
    if missing:
        raise SystemExit('ERROR: the hosted build did not deliver every binary the '
                         'selected suites run, so this pod cannot collect them: '
                         + '; '.join(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='all', help="the workflow's suite input")
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--directory', type=Path,
                        help='verify the required binaries exist here instead of listing them')
    args = parser.parse_args()
    try:
        required = required_binaries(args.suite, args.manifest)
    except ValueError as error:
        raise SystemExit(f'ERROR: {error}') from error
    if args.directory is None:
        for binary, suites in required.items():
            print(f'{binary}\t{", ".join(suites)}')
        return
    check(required, args.directory)


if __name__ == '__main__':
    main()
