#!/usr/bin/env python3
"""Regenerate benchmarks/cpu/ingestion-selection.json from a collection.

The inventory is the ledger of 1T/4T workload identities the standard quick
CPU nightly must produce. It is derived, not hand-written: this script packages
one collection's produced suites with the same code path that gates ingestion
and records the identity fingerprints it finds. `--check` reports a stale
ledger with a nonzero exit instead of the opaque ingestion rejection, and
`--write` updates it in place.

Collection and ingestion still gate against the checked-in file, so an
accidental suite-row change stays a failure; this only makes the intended
update one reviewed command instead of transcribed SHA-256 values.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

if __package__:
    from .bundle_cpu_results import package_cpu
    from .cpu_ingestion_adapter import CPU_SUITES, SELECTION_PATH, identity_fingerprint
    from .prepare_result_bundle import current_runs
else:
    from bundle_cpu_results import package_cpu
    from cpu_ingestion_adapter import CPU_SUITES, SELECTION_PATH, identity_fingerprint
    from prepare_result_bundle import current_runs


def collected(root: Path, profile: str, source: dict, bundle: Path) -> dict:
    """Identity entries for the suites this collection produced, keyed by suite."""
    # perf_issues is declared structurally from its case manifest, not here.
    produced = []
    for name in (name for name in CPU_SUITES if name != 'perf_issues'):
        raw_suite = 'einsum' if name in ('cpu_ops', 'linalg_jvp_vjp') else name
        if current_runs(root, profile, 'cpu/' + raw_suite, source):
            produced.append(name)
    if not produced:
        raise ValueError('No CPU collection outputs in the selected run attempt')
    manifest = package_cpu(root, profile, source['head_sha'], bundle, suites=produced, enforce_inventory=False,
                           window={'run_started_at': source['run_started_at'], 'created_at': source['created_at']})
    return {entry['suite_id']: {'row_count': len(entry['expected_rows']),
                                'identities_sha256': identity_fingerprint(entry['expected_rows']),
                                'threads': entry['threads']} for entry in manifest['suites']}


def updated_plan(plan: dict, computed: dict) -> dict:
    """Merge collected suites into the ledger; suites this run did not produce stay as declared."""
    threads = sorted({*plan['threads'], *(thread for entry in computed.values() for thread in entry['threads'])})
    suite_threads = dict(plan.get('suite_threads', {}))
    for suite, entry in computed.items():
        if entry['threads'] == threads:
            suite_threads.pop(suite, None)
        else:
            suite_threads[suite] = entry['threads']
    suites = dict(plan['suites'])
    for suite, entry in computed.items():
        suites[suite] = {'row_count': entry['row_count'], 'identities_sha256': entry['identities_sha256']}
    return {**plan, 'threads': threads, 'suites': suites, 'suite_threads': suite_threads}


def differences(plan: dict, computed: dict) -> list[str]:
    lines = []
    for suite, entry in sorted(computed.items()):
        declared = plan['suites'].get(suite)
        found = f"{entry['row_count']} identities, {entry['identities_sha256']}"
        if declared is None:
            lines.append(f'{suite}: undeclared, collected {found}')
        elif (declared['row_count'], declared['identities_sha256']) != (entry['row_count'], entry['identities_sha256']):
            lines.append(f"{suite}: declared {declared['row_count']} identities, {declared['identities_sha256']}; "
                         f'collected {found}')
        threads = plan.get('suite_threads', {}).get(suite, plan['threads'])
        if threads != entry['threads']:
            lines.append(f'{suite}: declared threads {threads}, collected {entry["threads"]}')
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', choices=['amd-cpu', 'mac-cpu'], required=True)
    parser.add_argument('--run-root', type=Path, required=True,
                        help='collection tree holding data/results and result for one run')
    parser.add_argument('--harness-commit', required=True)
    parser.add_argument('--started-at', required=True)
    parser.add_argument('--created-at', default=datetime.now(timezone.utc).isoformat(),
                        help='upper bound of the collection window; defaults to now')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true', help='compare and exit nonzero when they differ')
    action.add_argument('--write', action='store_true', help='write the checked-in inventory')
    args = parser.parse_args()

    source = {'head_sha': args.harness_commit, 'run_started_at': args.started_at, 'created_at': args.created_at}
    with tempfile.TemporaryDirectory(prefix='cpu-inventory-') as scratch:
        computed = collected(args.run_root.resolve(), args.profile, source, Path(scratch)/'bundle')

    plan = json.loads(SELECTION_PATH.read_text())
    updated = updated_plan(plan, computed)
    if args.write:
        SELECTION_PATH.write_text(json.dumps(updated, indent=2) + '\n')
        print(f'wrote {SELECTION_PATH.relative_to(SELECTION_PATH.parents[2])}')
    elif args.check:
        stale = differences(plan, computed)
        if stale:
            print('CPU ingestion inventory is stale:', file=sys.stderr)
            for line in stale:
                print('  ' + line, file=sys.stderr)
            raise SystemExit(1)
        print(f'{SELECTION_PATH.name} matches all {len(computed)} collected suites')
    else:
        print(json.dumps({'suites': computed, 'threads': updated['threads'],
                          'suite_threads': updated['suite_threads']}, indent=2))


if __name__ == '__main__':
    main()
