import json
import tempfile
import unittest
from pathlib import Path

from scripts.ci.check_gpu_run import bundle_run, validate_records
from scripts.ci.collect_gpu_pages_results import artifact_unavailable, publish_bundle


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
        self.summary = self.run / 'report.summary.json'
        self.write_summary()
        self.records = self.run / 'records.jsonl'
        self.write_rows([self.row])
        self.bundle = self.root / 'bundle'

    def write_rows(self, rows):
        self.records.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def write_summary(self, gpu='NVIDIA A100 80GB PCIe',
                      timestamp='2026-10-08T12:00:00+00:00'):
        self.summary.write_text(json.dumps({
            'schema_version': 1, 'target_profile': 'nvidia-gpu',
            'suite_id': 'gpu/dense', 'timestamp': timestamp,
            'tenferro_commit': 'deadbeef', 'hardware': {'gpu': gpu},
            'unit': 'ms', 'rows': [],
        }) + '\n')

    def build(self):
        bundle_run(self.root, self.timestamp, [Path('dense.yaml')],
                   ['tenferro-cuda-trace'], self.bundle)

    def site_summary(self, site, timestamp, gpu='NVIDIA A100 80GB PCIe'):
        path = site / 'result/nvidia-gpu/gpu/dense.summary.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            'schema_version': 1, 'target_profile': 'nvidia-gpu',
            'suite_id': 'gpu/dense', 'timestamp': timestamp,
            'hardware': {'gpu': gpu}, 'unit': 'ms', 'rows': [],
        }) + '\n')
        return path

    def test_complete_run_publishes_report_and_raw_data(self):
        self.build()
        site = self.root / 'site'
        self.assertEqual(publish_bundle(self.bundle, site, {}), 1)
        self.assertEqual((site / 'result/nvidia-gpu/gpu/dense.md').read_text(), 'new measured report')
        self.assertTrue((site / 'result/nvidia-gpu/gpu/dense.summary.json').exists())
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

    def test_requested_backend_not_declared_by_suite_is_not_expected(self):
        self.suite.write_text('suite_id: gpu/dense\nbackends: [tenferro-cuda-trace]\n'
                              'problems: [{id: matmul}]\n')
        bundle_run(self.root, self.timestamp, [Path('dense.yaml')],
                   ['tenferro-cuda-trace', 'pytorch-cuda'], self.bundle)
        self.assertTrue((self.bundle / 'publication.json').exists())

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

    def test_newer_a100_bundle_replaces_repository_suite(self):
        self.build()
        site = self.root / 'site'
        repository = self.site_summary(site, '2026-10-01T00:00:00+00:00')
        self.assertEqual(publish_bundle(self.bundle, site, {}), 1)
        self.assertEqual((site / 'result/nvidia-gpu/gpu/dense.md').read_text(), 'new measured report')
        published = json.loads(repository.read_text())
        self.assertEqual(published['timestamp'], '2026-10-08T12:00:00+00:00')

    def test_bundle_older_than_repository_summary_is_skipped(self):
        self.build()
        site = self.root / 'site'
        repository = self.site_summary(site, '2026-10-09T12:00:00+00:00')
        self.assertEqual(publish_bundle(self.bundle, site, {}), 0)
        self.assertFalse((site / 'result/nvidia-gpu/gpu/dense.md').exists())
        # The repository summary is untouched, not replaced by the older bundle.
        self.assertEqual(json.loads(repository.read_text())['timestamp'],
                         '2026-10-09T12:00:00+00:00')

    def test_non_a100_bundle_is_skipped(self):
        self.write_summary(gpu='NVIDIA H100 80GB HBM3')
        self.build()
        site = self.root / 'site'
        self.assertEqual(publish_bundle(self.bundle, site, {}), 0)
        self.assertFalse((site / 'result/nvidia-gpu/gpu/dense.md').exists())

    def test_bundle_without_summary_is_skipped(self):
        self.build()
        summary_relative = 'result/nvidia-gpu/gpu/dense.summary.json'
        (self.bundle / summary_relative).unlink()
        receipt = self.bundle / 'publication.json'
        manifest = json.loads(receipt.read_text())
        del manifest['suites'][0]['files'][summary_relative]
        receipt.write_text(json.dumps(manifest))
        site = self.root / 'site'
        self.assertEqual(publish_bundle(self.bundle, site, {}), 0)
        self.assertFalse((site / 'result/nvidia-gpu/gpu/dense.md').exists())

class ArtifactAvailabilityTests(unittest.TestCase):
    def test_missing_or_expired_artifacts_are_skipped_but_other_errors_are_not(self):
        self.assertTrue(artifact_unavailable('no valid artifacts found to download'))
        self.assertTrue(artifact_unavailable('no artifact matches any of the names or patterns provided'))
        self.assertTrue(artifact_unavailable('Artifact has expired'))
        self.assertFalse(artifact_unavailable('HTTP 502: Bad Gateway'))
