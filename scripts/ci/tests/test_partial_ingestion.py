"""Policy tests at the production driver's API/download/commit boundary."""
import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from scripts.ci import ingest_benchmark_results as driver
from scripts.ci.result_ingestion import digest, plan_updates, validate_summary
from scripts.ci.tests import test_result_ingestion as fixtures


class PartialDriverTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.IngestionTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.run = self.fixture.run
        self.artifact = self.fixture.artifact

    def rewrite_rows(self, rows):
        fixture = self.fixture
        fixture.summary['rows'] = rows
        fixture.entry['expected_rows'] = [[row['case'], row.get('group', ''), row['backend']] for row in rows]
        for relative in ('result/amd-cpu/cpu/einsum.summary.json', fixture.entry['run_directory']+'/report.summary.json'):
            path = fixture.bundle / relative
            path.write_text(json.dumps(fixture.summary))
            fixture.entry['files'][relative] = digest(path)
        fixture.save()

    def invoke(self, artifacts=None):
        artifacts = [self.artifact] if artifacts is None else artifacts
        api = patch.object(driver, 'gh_json', side_effect=[self.run, {'total_count': len(artifacts), 'artifacts': artifacts}])
        def download(artifact, path):
            shutil.copytree(self.fixture.bundle, path)
        self.commit = self.enterContext(patch.object(driver, 'commit_results'))
        self.publish = self.enterContext(patch.object(driver, 'request_publication'))
        self.commit.return_value = ({'cpu/einsum': 'accepted'}, 'fixture-commit')
        with api, patch.object(driver, 'download_artifact', side_effect=download), \
                patch('sys.argv', ['ingest', '--run-id', '123']), patch.dict('os.environ', {'GITHUB_STEP_SUMMARY': ''}):
            driver.main()
        return self.commit.call_args.args[1][0]

    def test_success_complete_artifact_is_ingested_as_complete(self):
        receipt, payload = self.invoke()
        self.assertEqual(receipt['status'], 'complete')
        self.assertEqual(receipt['case_outcomes']['failed'], {'count': 0, 'cases': []})
        self.assertEqual(receipt['source']['conclusion'], 'success')
        self.publish.assert_called_once()
        self.assertEqual(json.loads(payload['result/amd-cpu/cpu/einsum.summary.json'])['rows'][0]['median'], 1.5)

    def test_failure_complete_artifact_preserves_three_failed_cases_as_partial(self):
        self.run['conclusion'] = 'failure'
        rows = copy.deepcopy(self.fixture.summary['rows'])
        rows += [{'case': f'failed_{index}', 'backend': 'tenferro-trace', 'group': '1T',
                  'status': 'failed', 'verified': False, 'median': None, 'reason': 'collection error'} for index in range(3)]
        # Another backend/thread failure is not a fourth distinct case.
        rows.append(dict(rows[-1], backend='pytorch', group='4T'))
        self.rewrite_rows(rows)
        receipt, payload = self.invoke()
        self.assertEqual(receipt['status'], 'partial')
        self.assertEqual(receipt['case_outcomes']['failed'], {'count': 3, 'cases': ['failed_0', 'failed_1', 'failed_2']})
        self.assertEqual(json.loads(payload['result/amd-cpu/cpu/einsum.summary.json'])['rows'], rows)
        updates, _ = plan_updates(self.fixture.repo, [(receipt, payload)])
        self.assertFalse(any(name.startswith('data/') for name in updates))
        self.publish.assert_called_once()

    def test_failure_missing_artifact_is_rejected_with_run_and_reason(self):
        self.run['conclusion'] = 'failure'
        with self.assertRaisesRegex(ValueError, 'Run 123.*required artifact missing'):
            self.invoke([])
        self.commit.assert_not_called()
        self.publish.assert_not_called()

    def test_failure_missing_required_bundle_file_is_rejected(self):
        self.run['conclusion'] = 'failure'
        self.fixture.entry['files'].pop(self.fixture.entry['run_directory']+'/run.yaml')
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, 'Required bundle files are missing'):
            self.invoke()
        self.commit.assert_not_called()
        self.publish.assert_not_called()

    def test_untrusted_or_ineligible_run_rejects_before_artifact_lookup(self):
        for change in ({'head_repository': {'full_name': 'fork/repo'}},
                       {'repository': {'full_name': 'fork/repo'}},
                       {'display_title': 'RunPod readback (manual report)'},
                       *({'conclusion': value} for value in ('cancelled', 'timed_out', 'skipped', 'neutral'))):
            run = dict(self.run, **change)
            with self.subTest(change=change), patch.object(driver, 'gh_json', return_value=run) as api:
                with self.assertRaisesRegex(ValueError, 'canonical|conclusion'):
                    driver.selected_source(123)
                self.assertEqual(api.call_count, 1)

    def test_only_same_attempt_named_diagnostics_are_recoverable(self):
        self.run['conclusion'] = 'failure'
        diagnostic = dict(self.artifact, name='linux-cpu-benchmark-diagnostics-123')
        runtime = dict(self.artifact, name='gpu-runtime-123-00')
        older = dict(diagnostic, created_at='2026-10-17T22:00:00Z')
        for artifacts, accepted in (([diagnostic, runtime], True), ([runtime], False), ([older], False)):
            with self.subTest(artifacts=artifacts), patch.object(driver, 'gh_json', side_effect=[self.run, {'total_count':len(artifacts),'artifacts':artifacts}]):
                if accepted:
                    _, source, selected = driver.selected_source(123)
                    self.assertEqual(source['artifact_kind'], 'diagnostics')
                    self.assertEqual(selected, diagnostic)
                else:
                    with self.assertRaisesRegex(ValueError, 'required artifact missing'):
                        driver.selected_source(123)

    def test_unknown_failure_status_or_fake_failed_value_is_not_accepted(self):
        original = copy.deepcopy(self.fixture.summary['rows'])
        for status, median in (('mystery', None), ('failed', 0), ('verification_failed', 12.5)):
            rows = copy.deepcopy(original)
            rows.append(dict(rows[0], case='bad', status=status, median=median))
            self.rewrite_rows(rows)
            with self.subTest(status=status, median=median), self.assertRaisesRegex(ValueError, 'Incomplete'):
                validate_summary(self.fixture.summary, self.fixture.profile, self.fixture.entry)

    def test_skipped_and_unsupported_are_recorded_without_invented_numbers(self):
        rows = copy.deepcopy(self.fixture.summary['rows'])
        rows += [dict(rows[0], case=status, status=status, median=None, verified=False, reason='not supported')
                 for status in ('skipped', 'unsupported')]
        self.rewrite_rows(rows)
        receipt, payload = self.invoke()
        self.assertEqual(receipt['status'], 'partial')
        for kind in ('skipped', 'unsupported'):
            self.assertEqual(receipt['case_outcomes'][kind], {'count': 1, 'cases': [kind]})
        self.assertEqual(json.loads(payload['result/amd-cpu/cpu/einsum.summary.json'])['rows'], rows)


if __name__ == '__main__':
    unittest.main()
