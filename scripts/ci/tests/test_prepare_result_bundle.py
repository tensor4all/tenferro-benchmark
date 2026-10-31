import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import yaml

from scripts.ci import prepare_result_bundle as packaging
from scripts.ci.result_ingestion import digest, validate_bundle
from scripts.ci.tests import test_cpu_ingestion, test_gpu_ingestion


class PackagePartialCPUTest(unittest.TestCase):
    def setUp(self):
        self.fixture = test_cpu_ingestion.CPUIngestionTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.source = dict(self.fixture.source, conclusion='failure',
                           run_started_at='2026-10-17T23:00:00Z', created_at='2026-10-18T01:00:00Z')

    def add_perf(self):
        fixture = self.fixture
        run = fixture.root/'data/results/amd-cpu/cpu/perf_issues/20261018_000000'
        run.mkdir(parents=True)
        metadata = dict(fixture.metadata, suite_id='cpu/perf_issues')
        (run/'run.yaml').write_text(yaml.safe_dump(metadata))
        (run/'cpu_info.md').write_text('- Model: `Fixture CPU`\n')
        import benchmark_perf_issues
        suite_path, manifest_path, _ = benchmark_perf_issues.SUITES['cpu']
        suite = yaml.safe_load(suite_path.read_text())
        cases = {case['id']: case for case in json.loads((packaging.ROOT/suite['problems']['source']).read_text())}
        selected = yaml.safe_load(manifest_path.read_text())['coverage']['quick']
        failed = sorted(selected)[:3]
        for thread in (1, 4):
            meta = copy.deepcopy(metadata)
            meta['environment']['env']['RAYON_NUM_THREADS'] = str(thread)
            (run/f'run_t{thread}.yaml').write_text(yaml.safe_dump(meta))
            rows = [dict(case_id=case_id, backend=cases[case_id]['backend'], threads=thread,
                         measurement_scope=cases[case_id]['measurement_scope'],
                         correctness_status='failed' if case_id in failed else 'passed',
                         error='fixture failure' if case_id in failed else None,
                         samples=[] if case_id in failed else [{'elapsed_ns': 1000, 'iterations': 2}])
                    for case_id in selected if thread in cases[case_id].get('thread_counts', [thread])]
            (run/f'samples_t{thread}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        report = fixture.root/'result/amd-cpu/cpu/perf_issues.md'
        report.write_text('# Perf issues\nRaw run: `data/results/amd-cpu/cpu/perf_issues/20261018_000000`\n')
        return run, failed

    def test_failure_outputs_package_three_failed_cases_and_record_unproduced_suites(self):
        fixture = self.fixture
        run, failed = self.add_perf()
        original = (run/'samples_t1.jsonl').read_bytes()
        validated = packaging.prepare_bundle(fixture.root, 'amd-cpu', self.source, fixture.bundle)
        receipt, payload = next(item for item in validated if item[0]['suite_id'] == 'cpu/perf_issues')
        self.assertEqual(receipt['status'], 'partial')
        self.assertEqual(receipt['case_outcomes']['failed'], {'count':3, 'cases':failed})
        self.assertIn('cpu/permutation', receipt['missing_suites'])
        self.assertEqual((run/'samples_t1.jsonl').read_bytes(), original)
        doc = json.loads(payload['result/amd-cpu/cpu/perf_issues.summary.json'])
        self.assertEqual({row['median'] for row in doc['rows'] if row['status']=='ok'}, {500})
        self.assertTrue(all(row['median'] is None for row in doc['rows'] if row['status']=='failed'))
        # Moving the archive to another machine/extraction root changes no summary.
        moved = fixture.root/'moved'
        shutil.copytree(fixture.bundle, moved)
        self.assertEqual(validate_bundle(moved, 'amd-cpu', self.source), validated)

    def test_manual_driver_recovers_exact_failed_run_diagnostics(self):
        from scripts.ci import ingest_benchmark_results as driver
        self.add_perf()
        api_run = {'id':123, 'run_attempt':1, 'head_sha':'a'*40, 'head_branch':'main',
                   'status':'completed', 'conclusion':'failure', 'event':'schedule',
                   'path':'.github/workflows/benchmark-nightly-linux-cpu.yml',
                   'repository':{'full_name':'tensor4all/tenferro-benchmark'},
                   'head_repository':{'full_name':'tensor4all/tenferro-benchmark'},
                   'run_started_at':self.source['run_started_at']}
        artifact={'id':456, 'name':'linux-cpu-benchmark-diagnostics-123', 'expired':False,
                  'created_at':self.source['created_at'], 'expires_at':'2027-01-01T00:00:00Z',
                  'workflow_run':{'id':123, 'head_sha':'a'*40}}
        def download(artifact, path):
            shutil.copytree(self.fixture.root, path)
        with patch.object(driver,'gh_json',side_effect=[api_run,{'total_count':1,'artifacts':[artifact]}]), \
                patch.object(driver,'download_artifact',side_effect=download), \
                patch.object(driver,'commit_results',return_value=({'cpu/perf_issues':'accepted'},'commit')) as commit, \
                patch.object(driver,'request_publication') as publish, \
                patch('sys.argv',['ingest','--run-id','123']), patch.dict('os.environ',{'GITHUB_STEP_SUMMARY':''}):
            driver.main()
        receipts=[receipt for receipt,_ in commit.call_args.args[1]]
        perf=next(receipt for receipt in receipts if receipt['suite_id']=='cpu/perf_issues')
        self.assertEqual(perf['status'],'partial')
        self.assertEqual(perf['case_outcomes']['failed']['count'],3)
        self.assertEqual(perf['source']['artifact_id'],456)
        self.assertEqual(perf['source']['artifact_kind'],'diagnostics')
        publish.assert_called_once()

    def test_every_perf_thread_metadata_must_match_harness_suite_and_pytorch_provider(self):
        run, _ = self.add_perf()
        path = run/'run_t4.yaml'
        original = yaml.safe_load(path.read_text())
        for keys, value in ((('environment', 'env', 'BENCHMARK_COMMIT'), 'c'*40),
                            (('suite_id',), 'cpu/fft'),
                            (('python_backends', 'pytorch', 'provider'), 'openblas')):
            shutil.rmtree(self.fixture.bundle, ignore_errors=True)
            metadata = copy.deepcopy(original)
            target = metadata
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            path.write_text(yaml.safe_dump(metadata))
            with self.subTest(keys=keys), self.assertRaisesRegex(ValueError, 'metadata|provider'):
                packaging.prepare_bundle(self.fixture.root, 'amd-cpu', self.source, self.fixture.bundle)

    def test_incomplete_attempted_suite_is_rejected_not_reported_as_unproduced(self):
        fixture = self.fixture
        run, _ = self.add_perf()
        (run/'samples_t4.jsonl').unlink()
        with self.assertRaisesRegex(ValueError, 'inventory|thread|selection'):
            packaging.prepare_bundle(fixture.root, 'amd-cpu', self.source, fixture.bundle)
        self.assertFalse((fixture.bundle/'ingestion.json').exists())

    def test_current_raw_without_metadata_is_rejected_not_silently_omitted(self):
        run = self.fixture.root/'data/results/amd-cpu/cpu/perf_issues/20261018_000000'
        run.mkdir(parents=True)
        (run/'samples_t1.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'current collection outputs are missing run metadata'):
            packaging.prepare_bundle(self.fixture.root, 'amd-cpu', self.source, self.fixture.bundle)

    def test_no_current_outputs_is_explicit_rejection(self):
        self.source['run_started_at'] = '2026-10-19T00:00:00Z'
        self.source['created_at'] = '2026-10-19T01:00:00Z'
        with self.assertRaisesRegex(ValueError, 'No CPU collection outputs'):
            packaging.prepare_bundle(self.fixture.root, 'amd-cpu', self.source, self.fixture.bundle)


class PackagePartialGPUTest(unittest.TestCase):
    def setUp(self):
        self.fixture = test_gpu_ingestion.GPUIngestionTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.source = dict(self.fixture.source, conclusion='failure',
                           run_started_at='2026-10-17T23:00:00Z', created_at='2026-10-18T01:00:00Z')
        # Preparation reads only trusted declarations, not declarations in the artifact.
        self.root_patch = self.enterContext(patch.object(packaging, 'ROOT', self.fixture.root))

    def package(self):
        return packaging.prepare_bundle(self.fixture.root, 'nvidia-gpu', self.source,
                                        self.fixture.root/'partial-bundle', suites='dense.yaml',
                                        backends='tenferro-cuda-trace')

    def add_failure(self):
        fixture = self.fixture
        (fixture.root/'dense.yaml').write_text('suite_id: gpu/dense\nproblems: [{id: matmul}, {id: failed_case}]\n')
        failed = dict(fixture.row, problem_id='failed_case', status='verification_failed',
                      timing={'median_ms':None, 'iqr_ms':None, 'timed_runs':0}, verification={'status':'failed'})
        rows = [fixture.row, failed]
        (fixture.run/'records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        (fixture.run/'report.summary.json').write_text(json.dumps(test_gpu_ingestion.build_summary(rows, fixture.metadata)))

    def test_failed_gpu_rows_are_packaged_without_changing_collector_or_numbers(self):
        self.add_failure()
        receipt, payload = self.package()[0]
        self.assertEqual(receipt['case_outcomes']['failed'], {'count':1,'cases':['failed_case']})
        self.assertEqual(receipt['status'],'partial')
        self.assertEqual([row['median'] for row in json.loads(payload['result/nvidia-gpu/gpu/dense.summary.json'])['rows']], [None, 1.0])

    def test_complete_failed_only_suite_is_preserved_without_invented_measurements(self):
        self.add_failure()
        path = self.fixture.run/'records.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0].update(status='runtime_failed', timing={'median_ms':None, 'iqr_ms':None, 'timed_runs':0},
                       verification={'status':'skipped'})
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        (self.fixture.run/'report.summary.json').write_text(json.dumps(test_gpu_ingestion.build_summary(rows, self.fixture.metadata)))
        receipt, payload = self.package()[0]
        self.assertEqual(receipt['status'], 'partial')
        self.assertEqual(receipt['case_outcomes']['failed'], {'count':2,'cases':['failed_case','matmul']})
        self.assertTrue(all(row['median'] is None for row in json.loads(payload['result/nvidia-gpu/gpu/dense.summary.json'])['rows']))

    def test_failed_gpu_raw_timing_cannot_be_hidden_by_summary_normalization(self):
        for metric in ('first_run_ms', 'median_ms', 'min_ms', 'p95_ms', 'iqr_ms'):
            shutil.rmtree(self.fixture.root/'partial-bundle', ignore_errors=True)
            self.add_failure()
            path = self.fixture.run/'records.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]['timing'][metric] = 0.0
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.subTest(metric=metric), self.assertRaisesRegex(ValueError, 'failed row has timing'):
                self.package()

    def test_missing_gpu_row_or_missing_required_report_rejects(self):
        self.add_failure()
        fixture = self.fixture
        (fixture.run/'records.jsonl').write_text(json.dumps(fixture.row)+'\n')
        with self.assertRaisesRegex(ValueError, 'missing/unexpected raw rows'):
            self.package()
        self.assertFalse((fixture.root/'partial-bundle/publication.json').exists())

    def test_unverified_or_no_sample_ok_row_rejects(self):
        for field, value, reason in (('verification', {'status':'failed'}, 'unverified'),
                                     ('timing', {'median_ms':1,'iqr_ms':0,'timed_runs':0}, 'no measured samples')):
            fixture=self.fixture
            row=dict(fixture.row, **{field:value})
            (fixture.run/'records.jsonl').write_text(json.dumps(row)+'\n')
            (fixture.run/'report.summary.json').write_text(json.dumps(test_gpu_ingestion.build_summary([row], fixture.metadata)))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, reason):
                self.package()
            shutil.rmtree(fixture.root/'partial-bundle')


if __name__ == '__main__':
    unittest.main()
