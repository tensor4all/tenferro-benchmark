import json
import tempfile
import unittest
from pathlib import Path

from scripts.ci.check_gpu_run import bundle_run, validate_records


class GpuRunBundleTests(unittest.TestCase):
    """The RunPod workflow still builds a validated artifact bundle.

    The bundle is no longer overlaid on the GitHub Pages dashboard (the site
    publishes the repository's A100 results under result/nvidia-gpu), but the
    build-time validation is still exercised here.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.suite = self.root / 'dense.yaml'
        self.suite.write_text('suite_id: gpu/dense\nproblems: [{id: matmul}]\n')
        self.timestamp = '20261008_120000'
        self.run = self.root / 'data/results/nvidia-gpu/gpu/dense' / self.timestamp
        self.run.mkdir(parents=True)
        self.row = {'suite_id': 'gpu/dense', 'problem_id': 'matmul', 'backend': 'tenferro-cuda-trace',
                    'status': 'ok', 'timing': {'median_ms': 1, 'timed_runs': 3},
                    'verification': {'status': 'passed'}}
        (self.run / 'run.yaml').write_text('suite_id: gpu/dense\n')
        (self.run / 'report.md').write_text('new measured report')
        self.records = self.run / 'records.jsonl'
        self.write_rows([self.row])
        self.bundle = self.root / 'bundle'

    def write_rows(self, rows):
        self.records.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def build(self):
        bundle_run(self.root, self.timestamp, [Path('dense.yaml')],
                   ['tenferro-cuda-trace'], self.bundle)

    def test_complete_run_bundles_report_and_raw_data(self):
        self.build()
        self.assertTrue((self.bundle / 'publication.json').exists())
        self.assertEqual((self.bundle / 'result/nvidia-gpu/gpu/dense.md').read_text(), 'new measured report')
        self.assertTrue((self.bundle / 'data/results/nvidia-gpu/gpu/dense' / self.timestamp / 'records.jsonl').exists())

    def test_runtime_failed_run_keeps_diagnostics_without_publication_receipt(self):
        self.row['status'] = 'runtime_failed'
        self.write_rows([self.row])
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse((self.bundle / 'publication.json').exists())
        self.assertTrue((self.bundle / self.records.relative_to(self.root)).exists())

    def test_missing_duplicate_unverified_and_invalid_timing_are_rejected(self):
        for rows in [[], [self.row, self.row],
                     [dict(self.row, verification={'status': 'failed'})],
                     [dict(self.row, timing={'median_ms': float('nan'), 'timed_runs': 3})],
                     [dict(self.row, timing={'median_ms': 1, 'timed_runs': 0})]]:
            with self.subTest(rows=rows):
                self.write_rows(rows)
                with self.assertRaises(ValueError):
                    validate_records(self.records, 'gpu/dense', {('matmul', 'tenferro-cuda-trace')})

    def test_missing_requested_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            bundle_run(self.root, self.timestamp, [Path('dense.yaml')],
                       ['tenferro-cuda-trace', 'pytorch-cuda'], self.bundle)


if __name__ == '__main__':
    unittest.main()
