import json
import tempfile
import unittest
from pathlib import Path

from scripts.ci.check_gpu_run import bundle_run, validate_records
from scripts.ci.collect_gpu_pages_results import publish_bundle


class PublicationTests(unittest.TestCase):
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

    def test_complete_run_publishes_report_and_raw_data(self):
        self.build()
        site = self.root / 'site'
        self.assertEqual(publish_bundle(self.bundle, site, {}), 1)
        self.assertEqual((site / 'result/nvidia-gpu/gpu/dense.md').read_text(), 'new measured report')
        self.assertTrue((site / 'raw/nvidia-gpu/gpu/dense' / self.timestamp / 'records.jsonl').exists())

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

    def test_modified_report_is_rejected(self):
        self.build()
        (self.bundle / 'result/nvidia-gpu/gpu/dense.md').write_text('unverified report')
        with self.assertRaises(ValueError):
            publish_bundle(self.bundle, self.root / 'site', {})

    def test_older_bundle_cannot_overwrite_newer_suite(self):
        self.build()
        site = self.root / 'site'
        seen = {}
        publish_bundle(self.bundle, site, seen)
        self.assertEqual(publish_bundle(self.bundle, site, seen), 0)

    def test_old_unchecked_snapshot_is_not_published(self):
        self.assertEqual(publish_bundle(self.root, self.root / 'site', {}), 0)

    def test_rerun_with_newer_measurements_replaces_older_suite(self):
        self.build()
        site = self.root / 'site'
        seen = {}
        publish_bundle(self.bundle, site, seen)
        receipt = self.bundle / 'publication.json'
        # A workflow rerun retains its old creation date but samples at a new timestamp.
        import shutil
        newer = '20261009_120000'
        old_run = self.bundle / 'data/results/nvidia-gpu/gpu/dense' / self.timestamp
        new_run = old_run.with_name(newer)
        shutil.copytree(old_run, new_run)
        manifest = json.loads(receipt.read_text())
        manifest['timestamp'] = newer
        entry = manifest['suites'][0]
        entry['files'] = {path.replace(self.timestamp, newer): digest
                          for path, digest in entry['files'].items()}
        receipt.write_text(json.dumps(manifest))
        self.assertEqual(publish_bundle(self.bundle, site, seen), 1)
        self.assertEqual(seen['gpu/dense'], newer)
