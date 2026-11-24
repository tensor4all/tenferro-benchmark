import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from scripts.ci import cpu_ingestion_adapter, generate_cpu_ingestion_selection as generator
from scripts.ci.bundle_cpu_results import package_cpu
from scripts.ci.cpu_ingestion_adapter import identity_fingerprint
from scripts.ci.tests import test_cpu_ingestion


class GenerateCPUInventoryTest(unittest.TestCase):
    """The declaration is derived from a collection, never transcribed by hand."""

    def setUp(self):
        self.fixture = test_cpu_ingestion.CPUIngestionTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.source = {'head_sha': 'a'*40, 'run_started_at': '2026-10-17T23:00:00Z',
                       'created_at': '2026-10-18T01:00:00Z'}
        self.plan = {'schema_version': 1, 'profile': 'quick', 'threads': [1, 4],
                     'description': 'fixture', 'suites': {}, 'suite_threads': {'cpu/overhead_smoke': [1]}}

    def collect(self):
        bundle = self.fixture.root/'inventory-bundle'
        return generator.collected(self.fixture.root, 'amd-cpu', self.source, bundle)

    def test_collection_supplies_the_identity_fingerprint(self):
        computed = self.collect()
        self.assertEqual(list(computed), ['cpu/fft'])
        entry = computed['cpu/fft']
        self.assertEqual(entry['row_count'], 2)
        self.assertEqual(entry['threads'], [1, 4])
        self.assertEqual(entry['identities_sha256'],
                         identity_fingerprint([['fft', 'suite=cpu/fft / dtype=c64 / threads=1 / shape=1d_n16',
                                                 'tenferro-fft-executor-cached'],
                                                ['fft', 'suite=cpu/fft / dtype=c64 / threads=4 / shape=1d_n16',
                                                 'tenferro-fft-executor-cached']]))

    def test_regeneration_reads_the_same_ledger_the_ingestion_gate_rejects(self):
        # The generator must skip only the comparison, not the collection path:
        # a stale ledger still fails packaging, and the generator still reports it.
        stale = self.fixture.root/'stale-ledger.json'
        stale.write_text(json.dumps(dict(self.plan, suites={'cpu/fft': {'row_count': 1,
                                                                      'identities_sha256': '0'*64}})))
        with patch.object(cpu_ingestion_adapter, 'SELECTION_PATH', stale), \
                patch.object(generator, 'SELECTION_PATH', stale), \
                patch('scripts.ci.bundle_cpu_results.rebuild_cpu', cpu_ingestion_adapter.rebuild_cpu):
            with self.assertRaises(ValueError):
                package_cpu(self.fixture.root, 'amd-cpu', 'a'*40, self.fixture.root/'rejected', suites=('fft',))
            computed = generator.collected(self.fixture.root, 'amd-cpu', self.source,
                                           self.fixture.root/'inventory-bundle')
        self.assertEqual(computed['cpu/fft']['row_count'], 2)
        self.assertEqual(len(generator.differences(json.loads(stale.read_text()), computed)), 1)

    def test_a_source_outside_the_collection_window_is_rejected(self):
        # Regression: packaging used to rebuild its own source without the
        # window, so a report naming an older run was measured silently.
        with self.assertRaisesRegex(ValueError, 'outside the selected run attempt'):
            package_cpu(self.fixture.root, 'amd-cpu', 'a'*40, self.fixture.root/'late-bundle', suites=('fft',),
                        enforce_inventory=False,
                        window={'run_started_at': '2026-10-18T00:00:01Z', 'created_at': '2026-10-18T01:00:00Z'})

    def test_collected_forwards_the_window_so_a_stale_report_is_rejected(self):
        # A current run exists (fft is produced) but its report names an older
        # run; the generator must not fingerprint that older collection.
        import shutil
        import yaml
        fixture = self.fixture
        old = fixture.root/'data/results/amd-cpu/cpu/fft/20261017_000000'
        shutil.copytree(fixture.run, old)
        for path in old.glob('run*.yaml'):
            metadata = yaml.safe_load(path.read_text())
            metadata['timestamp'] = '2026-10-17T00:00:00+00:00'
            path.write_text(yaml.safe_dump(metadata))
        report = fixture.root/'result/amd-cpu/cpu/fft.md'
        report.write_text(report.read_text().replace('20261018_000000', '20261017_000000'))
        with self.assertRaisesRegex(ValueError, 'outside the selected run attempt'):
            generator.collected(fixture.root, 'amd-cpu', self.source, fixture.root/'stale-bundle')

    def test_write_saves_the_recomputed_ledger(self):
        computed = self.collect()
        ledger = self.fixture.root/'written.json'
        ledger.write_text(json.dumps(self.plan))
        argv = ['generate', '--profile', 'amd-cpu', '--run-root', str(self.fixture.root),
                '--harness-commit', 'a'*40, '--started-at', self.source['run_started_at'],
                '--created-at', self.source['created_at'], '--write']
        with patch.object(sys, 'argv', argv), patch.object(generator, 'SELECTION_PATH', ledger), \
                patch.object(generator, 'collected', return_value=computed):
            generator.main()
        written = json.loads(ledger.read_text())
        self.assertEqual(written['suites']['cpu/fft'], {'row_count': 2,
                                                       'identities_sha256': computed['cpu/fft']['identities_sha256']})
        self.assertEqual(generator.differences(written, computed), [])

    def test_declaration_matching_the_collection_reports_no_difference(self):
        computed = self.collect()
        self.assertEqual(generator.differences(generator.updated_plan(self.plan, computed), computed), [])

    def test_stale_declaration_names_the_suite_and_both_counts(self):
        computed = self.collect()
        stale = json.loads(json.dumps(self.plan, default=str))
        stale['suites']['cpu/fft'] = {'row_count': 1, 'identities_sha256': '0'*64}
        lines = generator.differences(stale, computed)
        self.assertEqual(len(lines), 1)
        self.assertIn('cpu/fft: declared 1 identities', lines[0])
        self.assertIn('collected 2 identities', lines[0])

    def test_undeclared_suite_and_thread_selection_are_reported(self):
        computed = self.collect()
        self.assertIn('undeclared', generator.differences(self.plan, computed)[0])
        self.assertTrue(any('declared threads [4]' in line
                            for line in generator.differences(dict(self.plan, threads=[4]), computed)))

    def test_merge_keeps_undeclared_outputs_of_a_partial_collection(self):
        plan = json.loads(json.dumps(self.plan))
        plan['suites']['cpu/small_work'] = {'row_count': 144, 'identities_sha256': '1'*64}
        updated = generator.updated_plan(plan, self.collect())
        self.assertEqual(updated['suites']['cpu/small_work'], plan['suites']['cpu/small_work'])
        self.assertEqual(updated['suites']['cpu/fft']['row_count'], 2)
        self.assertEqual(updated['threads'], [1, 4])
        self.assertEqual(updated['suite_threads'], {'cpu/overhead_smoke': [1]})

    def test_check_exits_nonzero_on_a_stale_ledger_and_zero_when_current(self):
        computed = self.collect()
        ledger = self.fixture.root/'ledger.json'
        argv = ['generate', '--profile', 'amd-cpu', '--run-root', str(self.fixture.root),
                '--harness-commit', 'a'*40, '--started-at', self.source['run_started_at'],
                '--created-at', self.source['created_at'], '--check']
        ledger.write_text(json.dumps(generator.updated_plan(self.plan, computed)))
        with patch.object(sys, 'argv', argv), patch.object(generator, 'SELECTION_PATH', ledger), \
                patch.object(generator, 'collected', return_value=computed):
            generator.main()
        ledger.write_text(json.dumps(self.plan))
        with patch.object(sys, 'argv', argv), patch.object(generator, 'SELECTION_PATH', ledger), \
                patch.object(generator, 'collected', return_value=computed):
            with self.assertRaises(SystemExit) as raised:
                generator.main()
        self.assertEqual(raised.exception.code, 1)


ROOT = Path(__file__).resolve().parents[3]
NIGHTLY_WORKFLOWS = ('benchmark-nightly-linux-cpu.yml', 'benchmark-nightly-mac-cpu.yml')
SESSION_MATRIX_MANIFEST = ROOT/'benchmarks/cpu/manifests/session_matrix.yaml'


def nightly_selection() -> tuple[list[int], str]:
    """The thread list and session_matrix coverage the CPU nightlies actually run.

    Read from the nightly workflows, never from the ledger under test: threads
    are the arguments of their `scripts/run_all.sh` call, and coverage is their
    BENCH_COVERAGE if they set one, else the harness default (bench_selection).
    """
    import re
    import bench_selection
    selections = set()
    for name in NIGHTLY_WORKFLOWS:
        text = (ROOT/'.github/workflows'/name).read_text()
        calls = re.findall(r'^\s*\./scripts/run_all\.sh((?:\s+\d+)+)\s*$', text, re.MULTILINE)
        if len(calls) != 1:
            raise AssertionError(f'{name}: expected one ./scripts/run_all.sh <threads> call, found {len(calls)}')
        coverage = re.findall(r'BENCH_COVERAGE[=:]\s*["\']?(\w+)', text)
        if len(set(coverage)) > 1:
            raise AssertionError(f'{name}: conflicting BENCH_COVERAGE values {coverage}')
        with patch.dict('os.environ', {}, clear=False) as env:
            env.pop('BENCH_COVERAGE', None)
            resolved = bench_selection.resolve_coverage(coverage[0] if coverage else None)
        selections.add((tuple(int(t) for t in calls[0].split()), resolved))
    if len(selections) != 1:
        raise AssertionError(f'the CPU nightlies disagree on threads/coverage: {sorted(selections)}')
    threads, coverage = selections.pop()
    return sorted(threads), coverage


def declared_session_matrix_identities(threads, coverage, manifest_path: Path = SESSION_MATRIX_MANIFEST) -> list:
    """The [case, group, backend] rows a cpu/session_matrix collection must produce.

    Derived statically from the case manifest and instance file, with the same
    provider canonicalisation the ingestion adapter applies to collected rows.
    """
    import bench_selection
    manifest = bench_selection.load_manifest(manifest_path)
    cases = {case['id']: case for case in json.loads((ROOT/manifest['source']).read_text())}

    def backend(provider):
        provider = 'blas' if provider in cpu_ingestion_adapter.BLAS_IMPLEMENTATIONS else provider
        return 'pytorch-cpu' if provider == 'pytorch' else 'tenferro-' + provider

    return sorted([case_id, f'threads={thread}', backend(provider)]
                  for thread in threads for case_id in manifest['coverage'][coverage]
                  for provider in cases[case_id]['providers'])


def session_matrix_ledger_problems(plan: dict) -> list[str]:
    """Every way `plan` disagrees with what the nightlies run and the cases declare."""
    threads, coverage = nightly_selection()
    problems = []
    if plan.get('profile') != coverage:
        problems.append(f"ledger profile {plan.get('profile')!r}, nightlies run coverage {coverage!r}")
    ledger_threads = plan.get('suite_threads', {}).get('cpu/session_matrix', plan.get('threads'))
    if ledger_threads != threads:
        problems.append(f'ledger cpu/session_matrix threads {ledger_threads}, nightlies run {threads}')
    rows = declared_session_matrix_identities(threads, coverage)
    expected = {'row_count': len(rows), 'identities_sha256': identity_fingerprint(rows)}
    if plan.get('suites', {}).get('cpu/session_matrix') != expected:
        problems.append(f"ledger cpu/session_matrix {plan.get('suites', {}).get('cpu/session_matrix')}, "
                        f'declared cases give {expected}')
    return problems


class CheckedInSessionMatrixLedgerTest(unittest.TestCase):
    """A change to the session_matrix cases or nightly threads must regenerate the ledger in the same PR.

    Regression: #491 dropped the vendor-BLAS provider slots (114 -> 86 quick
    identities) without regenerating benchmarks/cpu/ingestion-selection.json, and
    nothing failed until both CPU nightlies rejected their collections after
    ~2h40m, every night from 2026-11-17 to 2026-11-23.
    """

    def setUp(self):
        self.plan = json.loads(cpu_ingestion_adapter.SELECTION_PATH.read_text())

    def test_ledger_matches_the_nightly_selection_and_declared_cases(self):
        self.assertEqual(session_matrix_ledger_problems(self.plan), [],
                         'regenerate benchmarks/cpu/ingestion-selection.json with '
                         'scripts/ci/generate_cpu_ingestion_selection.py from a nightly collection')

    def test_selection_comes_from_the_nightly_workflows(self):
        self.assertEqual(nightly_selection(), ([1, 4], 'quick'))

    def test_wrong_ledger_threads_fail_even_when_the_rows_agree_with_them(self):
        # Review of #514: deriving threads from the ledger made this pass.
        # regenerate's updated_plan unions collected threads, so a 1/4/8T tree
        # moves the ledger to three threads; the nightly runs 1 4 and rejects it.
        rows = declared_session_matrix_identities([1, 4, 8], 'quick')
        plan = dict(self.plan, threads=[1, 4, 8], suites=dict(
            self.plan['suites'], **{'cpu/session_matrix': {'row_count': len(rows),
                                                           'identities_sha256': identity_fingerprint(rows)}}))
        problems = session_matrix_ledger_problems(plan)
        self.assertEqual(len(rows), 129)
        self.assertTrue(any('threads [1, 4, 8], nightlies run [1, 4]' in p for p in problems), problems)
        self.assertTrue(any('declared cases give' in p for p in problems), problems)

    def test_wrong_ledger_coverage_fails_even_when_the_rows_agree_with_it(self):
        rows = declared_session_matrix_identities([1, 4], 'full')
        plan = dict(self.plan, profile='full', suites=dict(
            self.plan['suites'], **{'cpu/session_matrix': {'row_count': len(rows),
                                                           'identities_sha256': identity_fingerprint(rows)}}))
        problems = session_matrix_ledger_problems(plan)
        self.assertTrue(any("profile 'full', nightlies run coverage 'quick'" in p for p in problems), problems)
        self.assertTrue(any('declared cases give' in p for p in problems), problems)

    def test_a_suite_threads_override_is_checked_too(self):
        plan = dict(self.plan, suite_threads=dict(self.plan.get('suite_threads', {}),
                                                  **{'cpu/session_matrix': [1]}))
        self.assertTrue(any('threads [1], nightlies run [1, 4]' in p
                            for p in session_matrix_ledger_problems(plan)))

    def test_nightlies_that_disagree_on_threads_are_reported(self):
        real = Path.read_text

        def fake(path, *args, **kwargs):
            text = real(path, *args, **kwargs)
            return text.replace('./scripts/run_all.sh 1 4', './scripts/run_all.sh 1 4 8') \
                if path.name == 'benchmark-nightly-mac-cpu.yml' else text
        with patch.object(Path, 'read_text', fake):
            with self.assertRaisesRegex(AssertionError, 'disagree on threads/coverage'):
                nightly_selection()

if __name__ == '__main__':
    unittest.main()
