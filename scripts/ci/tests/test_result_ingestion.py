import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.ci.result_ingestion import (digest, instant, plan_updates, safe_file,
                                         source_from_api, validate_bundle)


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
        older = copy.deepcopy(validated)
        older[0][0]['collected_at'] = '2026-10-17T23:59:59Z'
        older[0][0]['source']['run_attempt'] = 999
        self.assertEqual(plan_updates(self.repo, older), ({}, {'cpu/einsum': 'stale'}))
        conflicting = copy.deepcopy(validated)
        conflicting[0][0]['files']['result/amd-cpu/cpu/einsum.md'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            plan_updates(self.repo, conflicting)

    def test_legacy_timestamp_floor_and_timezone(self):
        path = self.repo / 'result/amd-cpu/cpu/einsum.summary.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.summary))
        self.assertEqual(plan_updates(self.repo, self.validate())[1]['cpu/einsum'], 'stale')
        self.assertEqual(instant('2026-10-18T09:00:00+09:00'), instant('2026-10-18T00:00:00Z'))
        with self.assertRaises(ValueError):
            instant('2026-10-18T00:00:00')

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

    def test_paths_and_symlinks_rejected(self):
        for relative in ('../outside', '/etc/passwd', 'a/../outside', 'a\\b', 'a//b'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                safe_file(self.bundle, relative)
        (self.bundle/'link').symlink_to('/etc/passwd')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            safe_file(self.bundle, 'link')

    def test_untrusted_sources_expired_artifact_and_wrong_run_rejected(self):
        for field, value in [('head_branch', 'feature'), ('conclusion', 'failure'), ('head_repository', {'full_name': 'fork/repo'}), ('path', '.github/workflows/ci.yml')]:
            run = dict(self.run, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                source_from_api(run, self.artifact)
        for change in ({'created_at': '2026-10-17T22:00:00Z'}, {'expired': True}, {'workflow_run': {'id': 999, 'head_sha': 'a'*40}}):
            with self.assertRaises(ValueError):
                source_from_api(self.run, dict(self.artifact, **change))


if __name__ == '__main__':
    unittest.main()
