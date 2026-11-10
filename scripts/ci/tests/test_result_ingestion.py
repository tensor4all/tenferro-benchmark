import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ci.result_ingestion import (digest, instant, plan_updates, publish_receipt,
                                         safe_file, source_from_api, validate_bundle,
                                         validate_summary)


class IngestionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'bundle'
        self.bundle.mkdir()
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.run = {'run_started_at': '2026-10-17T23:00:00Z', 'id': 123, 'run_attempt': 1, 'head_sha': 'a'*40,
                    'head_branch': 'main', 'status': 'completed', 'conclusion': 'success',
                    'path': '.github/workflows/benchmark-nightly-linux-cpu.yml',
                    'repository': {'full_name': 'tensor4all/tenferro-benchmark'},
                    'head_repository': {'full_name': 'tensor4all/tenferro-benchmark'}}
        self.artifact = {'id': 456, 'name': 'linux-cpu-benchmark-results-123', 'expired': False,
                         'created_at': '2026-10-18T01:00:00Z', 'expires_at': '2027-01-16T01:00:00Z',
                         'workflow_run': {'id': 123, 'head_sha': 'a'*40}}
        self.profile, self.source = source_from_api(self.run, self.artifact)
        self.summary = {'schema_version': 1, 'target_profile': 'amd-cpu', 'suite_id': 'cpu/einsum',
                        'timestamp': '2026-10-18T00:00:00+00:00', 'tenferro_commit': 'b'*40,
                        'hardware': {'host': 'fixture'}, 'unit': 'ms',
                        'rows': [{'case': 'matmul', 'group': '1T', 'backend': 'tenferro-trace',
                                  'status': 'ok', 'verified': True, 'median': 1.5}]}
        self.entry = {'suite_id': 'cpu/einsum', 'timestamp': self.summary['timestamp'],
                      'tenferro_commit': 'b'*40, 'hardware': self.summary['hardware'],
                      'expected_rows': [['matmul', '1T', 'tenferro-trace']],
                      'run_directory': 'data/results/amd-cpu/cpu/einsum/20261018_000000', 'files': {}}
        base = self.entry['run_directory']
        for relative, content in {
            'result/amd-cpu/cpu/einsum.md': b'report',
            'result/amd-cpu/cpu/einsum.summary.json': json.dumps(self.summary).encode(),
            base+'/report.md': b'report', base+'/report.summary.json': json.dumps(self.summary).encode(),
            base+'/run.yaml': b'fixture: true',
        }.items():
            path = self.bundle / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            self.entry['files'][relative] = digest(path)
        self.manifest = {'schema_version': 1, 'target_profile': self.profile,
                         'harness_commit': 'a'*40, 'suites': [self.entry]}
        self.save()
        # These tests isolate acceptance/ordering from the independently tested
        # CPU raw adapter, using a fixed rebuilt summary for the fixture.
        self.raw_patch = patch('scripts.ci.cpu_ingestion_adapter.rebuild_cpu', return_value=self.summary)
        self.raw_patch.start()
        self.addCleanup(self.raw_patch.stop)

    def save(self):
        (self.bundle / 'ingestion.json').write_text(json.dumps(self.manifest))

    def validate(self):
        return validate_bundle(self.bundle, self.profile, self.source)

    def apply(self, updates):
        for relative, value in updates.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)

    def test_accept_duplicate_stale_and_conflicting_source(self):
        validated = self.validate()
        updates, decisions = plan_updates(self.repo, validated)
        self.assertEqual(decisions, {'cpu/einsum': 'accepted'})
        self.assertEqual(len(updates), 3)
        self.apply(updates)
        self.assertEqual(plan_updates(self.repo, validated), ({}, {'cpu/einsum': 'duplicate'}))
        extended = copy.deepcopy(validated)
        extended[0][0]['source']['expires_at'] = '2027-02-01T00:00:00Z'
        self.assertEqual(plan_updates(self.repo, extended), ({}, {'cpu/einsum': 'duplicate'}))
        older = copy.deepcopy(validated)
        older[0][0]['collected_at'] = '2026-10-17T23:59:59Z'
        older[0][0]['source']['run_attempt'] = 999
        self.assertEqual(plan_updates(self.repo, older), ({}, {'cpu/einsum': 'stale'}))
        conflicting = copy.deepcopy(validated)
        conflicting[0][0]['files']['result/amd-cpu/cpu/einsum.md'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            plan_updates(self.repo, conflicting)

    def test_a_publisher_receipt_records_the_published_files(self):
        # Every path that writes a sidecar writes this receipt (#431), so the
        # acceptance check reads the same digest map whether the driver or a
        # runner published the page.
        report = self.repo / 'result/amd-cpu/cpu/cpu_ops.md'
        summary = report.with_suffix('.summary.json')
        report.parent.mkdir(parents=True)
        report.write_bytes(b'report')
        # The sidecar still says cpu/einsum: an einsum run publishes the cpu_ops
        # page, and the receipt has to name the page, not the run's suite.
        summary.write_text(json.dumps(self.summary))
        receipt = json.loads(publish_receipt(report, self.repo).read_text())
        self.assertEqual(receipt['schema_version'], 1)
        self.assertEqual((receipt['target_profile'], receipt['suite_id']),
                         ('amd-cpu', 'cpu/cpu_ops'))
        self.assertEqual(receipt['collected_at'], self.summary['timestamp'])
        self.assertEqual(receipt['files'], {
            'result/amd-cpu/cpu/cpu_ops.md': digest(report),
            'result/amd-cpu/cpu/cpu_ops.summary.json': digest(summary)})
        self.assertNotIn('source', receipt)

    def test_a_publish_over_a_campaign_receipt_keeps_its_coverage(self):
        # A campaign that collected only cpu/einsum left its receipt here, naming
        # cpu/cpu_ops as missing. A runner that republishes the page replaces that
        # file, and with it the only record of what the campaign did not collect.
        path = self.repo / 'result/amd-cpu/cpu/cpu_ops.provenance.json'
        path.parent.mkdir(parents=True)
        campaign = {'schema_version': 1, 'target_profile': 'amd-cpu', 'suite_id': 'cpu/cpu_ops',
                    'collected_at': '2026-10-17T00:00:00Z', 'status': 'partial',
                    'source': {'run_id': 9, 'run_attempt': 1, 'artifact_id': 10},
                    'case_outcomes': {'ok': 1}, 'missing_suites': ['cpu/einsum'], 'files': {}}
        path.write_text(json.dumps(campaign))
        report = self.repo / 'result/amd-cpu/cpu/cpu_ops.md'
        report.write_bytes(b'republished')
        report.with_suffix('.summary.json').write_text(json.dumps(self.summary))
        receipt = json.loads(publish_receipt(report, self.repo).read_text())
        self.assertNotIn('source', receipt)
        # The carried receipt is the campaign's, minus its digest map: those
        # hashes describe the page this publish just replaced.
        self.assertEqual(receipt['campaign'], {k: v for k, v in campaign.items() if k != 'files'})
        self.assertEqual(receipt['files']['result/amd-cpu/cpu/cpu_ops.md'], digest(report))
        # A second publish sees the carried campaign, not the campaign's receipt.
        # Dropping it here would lose the inventory as surely as the first overwrite.
        report.write_bytes(b'republished twice')
        again = json.loads(publish_receipt(report, self.repo).read_text())
        self.assertEqual(again['campaign'], receipt['campaign'])
        self.assertEqual(again['files']['result/amd-cpu/cpu/cpu_ops.md'], digest(report))

    def test_republishing_the_same_bytes_keeps_the_run_that_produced_them(self):
        # Writing the same page again is not a new measurement. Demoting the receipt
        # to a carried campaign would drop the recorded workflow failure, and the
        # dashboard would stop calling this a partial run.
        path = self.repo / 'result/amd-cpu/cpu/cpu_ops.provenance.json'
        path.parent.mkdir(parents=True)
        report = self.repo / 'result/amd-cpu/cpu/cpu_ops.md'
        report.write_bytes(b'report')
        summary = report.with_suffix('.summary.json')
        summary.write_text(json.dumps(self.summary))
        files = {'result/amd-cpu/cpu/cpu_ops.md': digest(report),
                 'result/amd-cpu/cpu/cpu_ops.summary.json': digest(summary)}
        campaign = {'schema_version': 1, 'target_profile': 'amd-cpu', 'suite_id': 'cpu/cpu_ops',
                    'collected_at': '2026-10-17T00:00:00Z', 'status': 'partial',
                    'source': {'run_id': 9, 'run_attempt': 1, 'artifact_id': 10,
                               'conclusion': 'failure'},
                    'case_outcomes': {'ok': 1}, 'missing_suites': ['cpu/einsum'], 'files': files}
        path.write_text(json.dumps(campaign))
        receipt = json.loads(publish_receipt(report, self.repo).read_text())
        self.assertEqual(receipt['source'], campaign['source'])
        self.assertEqual(receipt['status'], 'partial')
        self.assertEqual(receipt['missing_suites'], ['cpu/einsum'])
        self.assertNotIn('campaign', receipt)

    def test_a_publisher_receipt_orders_by_collection_time(self):
        # A receipt without a source run records no ingestion order, so the
        # candidate is newer only when its collection time is.
        path = self.repo / 'result/amd-cpu/cpu/einsum.provenance.json'
        path.parent.mkdir(parents=True)
        published = {'schema_version': 1, 'target_profile': 'amd-cpu', 'suite_id': 'cpu/einsum',
                     'collected_at': self.summary['timestamp'], 'files': {}}
        path.write_text(json.dumps(published))
        self.assertEqual(plan_updates(self.repo, self.validate())[1]['cpu/einsum'], 'stale')
        published['collected_at'] = '2026-10-17T00:00:00+00:00'
        path.write_text(json.dumps(published))
        self.assertEqual(plan_updates(self.repo, self.validate())[1]['cpu/einsum'], 'accepted')

    def test_retired_profile_rejected(self):
        self.manifest['target_profile'] = 'linux-cpu'
        self.save()
        with self.assertRaisesRegex(ValueError, 'Retired target profile'):
            self.validate()

    def test_legacy_timestamp_floor_and_timezone(self):
        path = self.repo / 'result/amd-cpu/cpu/einsum.summary.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.summary))
        self.assertEqual(plan_updates(self.repo, self.validate())[1]['cpu/einsum'], 'stale')
        self.assertEqual(instant('2026-10-18T09:00:00+09:00'), instant('2026-10-18T00:00:00Z'))
        with self.assertRaises(ValueError):
            instant('2026-10-18T00:00:00')
        with self.assertRaisesRegex(ValueError, 'not a string'):
            instant(1762600000)

    def test_digest_and_incomplete_second_suite_reject_without_writes(self):
        self.entry['files']['result/amd-cpu/cpu/einsum.md'] = 'f'*64
        self.save()
        with self.assertRaisesRegex(ValueError, 'Digest mismatch'):
            self.validate()
        self.assertEqual(list(self.repo.iterdir()), [])
        self.entry['files']['result/amd-cpu/cpu/einsum.md'] = digest(self.bundle/'result/amd-cpu/cpu/einsum.md')
        self.manifest['suites'].append(dict(self.entry, suite_id='cpu/missing'))
        self.save()
        with self.assertRaises(ValueError):
            self.validate()
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_invalid_uncertainty_is_rejected(self):
        for iqr in (-0.1, float('nan'), float('inf'), True):
            self.summary['rows'][0]['iqr']=iqr
            with self.subTest(iqr=iqr), self.assertRaisesRegex(ValueError,'IQR'):
                validate_summary(self.summary,self.profile,self.entry)

    def test_paths_and_symlinks_rejected(self):
        for relative in ('../outside', '/etc/passwd', 'a/../outside', 'a\\b', 'a//b'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                safe_file(self.bundle, relative)
        (self.bundle/'link').symlink_to('/etc/passwd')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            safe_file(self.bundle, 'link')

    def test_untrusted_sources_expired_artifact_and_wrong_run_rejected(self):
        for field, value in [('head_branch', 'feature'), ('conclusion', 'cancelled'), ('head_repository', {'full_name': 'fork/repo'}), ('path', '.github/workflows/ci.yml')]:
            run = dict(self.run, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                source_from_api(run, self.artifact)
        for change in ({'created_at': '2026-10-17T22:00:00Z'}, {'expired': True}, {'workflow_run': {'id': 999, 'head_sha': 'a'*40}}):
            with self.assertRaises(ValueError):
                source_from_api(self.run, dict(self.artifact, **change))


if __name__ == '__main__':
    unittest.main()
