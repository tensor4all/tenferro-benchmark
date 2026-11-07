"""A run publishes what it measured, and names every case it did not.

One case that fails (or produces no row at all) must never cost the run the rows
that did succeed, and it must never leave a hole: the summary sidecar and the
report carry the case with its status and a human-readable reason.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import report_summary  # noqa: E402
from format_gpu_results import build_summary, format_markdown  # noqa: E402
from scripts.ci import prepare_result_bundle as packaging  # noqa: E402

SUITE = 'gpu/dense'
BACKEND = 'tenferro-cuda-trace'
STAMP = '20261018_000000'


def record(problem, *, status='ok', median=1.0, reason=None, verification='passed'):
    """One raw GPU record in the shared contract."""
    return {
        'schema_version': 1, 'suite_id': SUITE, 'problem_id': problem, 'op': 'matmul',
        'backend': BACKEND, 'status': status,
        'timing': {'warmup_runs': 1, 'timed_runs': 3, 'compile_time_ms': None,
                   'first_run_ms': None,
                   'median_ms': median if status == 'ok' else None,
                   'min_ms': median if status == 'ok' else None,
                   'p95_ms': median if status == 'ok' else None,
                   'iqr_ms': 0.05 if status == 'ok' else None,
                   'timing_scope': 'steady_state_host_api_plus_device_sync'},
        'verification': {'status': verification, 'reference_backend': 'pytorch-cuda',
                         'rtol': 1e-5, 'atol': 1e-6, 'reason': reason},
        'execution': {'device': 'cuda', 'device_ordinal': 0, 'execution_path': 'fixture',
                      'synchronization': 'fixture', 'layout': 'col_major', 'dtype': 'f32',
                      'notes': json.dumps({'sampling': {'batch_ms': [1.0, 1.0, 1.0],
                                                        'operations_per_sample': 1},
                                           'setup': 'fixture inputs on device',
                                           'precision': 'f32'}),
                      'unsupported_reason': None},
    }


def metadata(planned):
    return {
        'target_profile': 'nvidia-gpu', 'suite_id': SUITE,
        'timestamp': '2026-10-18T00:00:00+00:00',
        'tenferro_rs': {'commit': 'b' * 40},
        'cuda': {'device_name': 'NVIDIA A100 80GB PCIe'},
        'collection': {'effort': 'standard', 'gpu_routine_manifest_version': 2,
                       'gpu_routine_selection_file': 'data/results/nvidia-gpu/gpu/'
                                                     'routine_selection/20261018_000000/selection.json',
                       'gpu_routine_vendor_arms': False,
                       'selected_case_ids': planned, 'expected_case_ids': planned},
    }


class PartialSummaryTest(unittest.TestCase):
    """The sidecar the dashboard reads: every planned case, with its reason."""

    def setUp(self):
        self.planned = [f'{SUITE}:{name}:{BACKEND}'
                        for name in ('matmul', 'matmul_failed', 'matmul_missing')]
        self.records = [
            record('matmul'),
            record('matmul_failed', status='runtime_failed', median=None, verification='skipped',
                   reason='runtime state failure: backend storage cannot be borrowed as host data'),
        ]
        self.summary = build_summary(self.records, metadata(self.planned))

    def test_every_planned_case_keeps_a_row_with_status_and_reason(self):
        rows = {(row['case'], row['backend']): row for row in self.summary['rows']}
        self.assertEqual(set(rows), {('matmul', BACKEND), ('matmul_failed', BACKEND),
                                     ('matmul_missing', BACKEND)})
        measured = rows[('matmul', BACKEND)]
        self.assertEqual((measured['status'], measured['median'], measured['verified']),
                         ('ok', 1.0, True))
        failed = rows[('matmul_failed', BACKEND)]
        self.assertEqual(failed['status'], 'runtime_failed')
        self.assertIsNone(failed['median'])
        self.assertIs(failed['verified'], False)
        self.assertEqual(failed['reason'],
                         'runtime state failure: backend storage cannot be borrowed as host data')
        missing = rows[('matmul_missing', BACKEND)]
        self.assertEqual(missing['status'], report_summary.MISSING_STATUS)
        self.assertIsNone(missing['median'])
        self.assertIn('no measurement row was written', missing['reason'])

    def test_no_non_ok_row_ever_carries_a_blank_reason(self):
        for row in self.summary['rows']:
            if row['status'] != 'ok':
                self.assertTrue(row.get('reason', '').strip(), row)
        # A failure whose runner recorded nothing still explains itself, in the
        # wording `rows_from_gpu_records` owns (shared with #376 on main).
        bare = build_summary([record('matmul_failed', status='runtime_failed', median=None,
                                     verification='skipped')],
                             metadata([self.planned[1]]))
        failed = next(row for row in bare['rows'] if row['case'] == 'matmul_failed')
        self.assertEqual(failed['status'], 'runtime_failed')
        self.assertEqual(failed['reason'], 'runtime_failed: no reason recorded')

    def test_the_report_lists_the_unmeasured_cases(self):
        markdown = format_markdown(self.records, metadata(self.planned))
        self.assertIn('## Not measured', markdown)
        self.assertIn('| matmul_failed | tenferro-cuda-trace | runtime_failed |', markdown)
        self.assertIn('| matmul_missing | tenferro-cuda-trace | missing |', markdown)

    def test_an_all_failed_suite_still_renders_its_failures(self):
        markdown = format_markdown([], metadata(self.planned))
        self.assertIn('No case in this run produced a measurement', markdown)
        self.assertIn('| matmul_missing |', markdown)


class PartialPublicationTest(unittest.TestCase):
    """A failed or unrecorded case does not stop the suite from publishing."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(packaging, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.source = {'head_sha': 'a' * 40, 'run_id': 1, 'run_attempt': 1, 'artifact_id': 2,
                       'run_started_at': '2026-10-17T23:00:00Z',
                       'created_at': '2026-10-18T01:00:00Z', 'conclusion': 'failure'}

    def declaration(self, name, suite, problems):
        (self.root / name).write_text(yaml.safe_dump({
            'suite_id': suite, 'backends': [BACKEND],
            'problems': [{'id': problem} for problem in problems]}))

    def run_directory(self, *, problems, records):
        run = self.root / f'data/results/nvidia-gpu/{SUITE}/{STAMP}'
        run.mkdir(parents=True, exist_ok=True)
        meta = metadata([f'{SUITE}:{problem}:{BACKEND}' for problem in problems])
        (run / 'run.yaml').write_text(yaml.safe_dump(meta))
        (run / 'records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
        (run / 'report.md').write_text('# Measured report\n')
        (run / 'report.summary.json').write_text(
            json.dumps(build_summary(records, meta), indent=2) + '\n')
        return run

    def test_failed_and_unrecorded_cases_publish_beside_the_measured_row(self):
        problems = ['matmul', 'matmul_failed', 'matmul_missing']
        self.declaration('dense.yaml', SUITE, problems)
        self.run_directory(problems=problems, records=[
            record('matmul'),
            record('matmul_failed', status='runtime_failed', median=None,
                   verification='skipped', reason='launch failed: out of memory'),
        ])
        validated = packaging.prepare_bundle(self.root, 'nvidia-gpu', self.source,
                                             self.root / 'bundle', suites='dense.yaml',
                                             backends=BACKEND)
        receipt, payload = validated[0]
        # The measured row publishes; the two failures are named in the receipt.
        self.assertEqual(receipt['status'], 'partial')
        self.assertEqual(receipt['case_outcomes']['failed'],
                         {'count': 2, 'cases': ['matmul_failed', 'matmul_missing']})
        document = json.loads(payload['result/nvidia-gpu/gpu/dense.summary.json'])
        rows = {row['case']: row for row in document['rows']}
        self.assertEqual(set(rows), set(problems))
        self.assertEqual(rows['matmul']['median'], 1.0)
        self.assertEqual(rows['matmul_failed']['status'], 'runtime_failed')
        self.assertEqual(rows['matmul_failed']['reason'], 'launch failed: out of memory')
        self.assertEqual(rows['matmul_missing']['status'], 'missing')
        self.assertTrue(rows['matmul_missing']['reason'].strip())
        self.assertTrue((self.root / 'bundle/publication.json').is_file())

    def test_a_suite_that_cannot_publish_does_not_take_the_others_with_it(self):
        self.declaration('dense.yaml', SUITE, ['matmul'])
        self.declaration('einsum.yaml', 'gpu/einsum', ['matmul'])
        run = self.run_directory(problems=['matmul'], records=[record('matmul')])
        # A second suite whose artifacts disagree with each other: its summary
        # and metadata name gpu/einsum while its raw records name gpu/dense.
        broken = self.root / f'data/results/nvidia-gpu/gpu/einsum/{STAMP}'
        shutil.copytree(run, broken)
        meta = yaml.safe_load((broken / 'run.yaml').read_text())
        meta['suite_id'] = 'gpu/einsum'
        (broken / 'run.yaml').write_text(yaml.safe_dump(meta))
        document = json.loads((broken / 'report.summary.json').read_text())
        document['suite_id'] = 'gpu/einsum'
        (broken / 'report.summary.json').write_text(json.dumps(document) + '\n')

        validated = packaging.prepare_bundle(self.root, 'nvidia-gpu', self.source,
                                             self.root / 'bundle',
                                             suites='dense.yaml,einsum.yaml', backends=BACKEND)
        self.assertEqual([receipt['suite_id'] for receipt, _ in validated], [SUITE])
        publication = json.loads((self.root / 'bundle/publication.json').read_text())
        self.assertEqual([entry['suite_id'] for entry in publication['suites']], [SUITE])
        self.assertIn('gpu/einsum', publication['missing_suites'])
        # The dropped suite leaves no half-copied file behind.
        self.assertFalse((self.root / 'bundle/result/nvidia-gpu/gpu/einsum.summary.json').exists())


if __name__ == '__main__':
    unittest.main()
