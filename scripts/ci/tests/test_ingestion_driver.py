import io
import json
import stat
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.ci.ingest_benchmark_results import commit_results, extract_archive


class ArchiveTest(unittest.TestCase):
    def test_unsafe_archives_are_rejected_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for bad in ('../escape', '/absolute', 'a/../escape', 'a\\b'):
                archive = root/'artifact.zip'
                with zipfile.ZipFile(archive, 'w') as zipped:
                    zipped.writestr('valid.txt', 'valid')
                    zipped.writestr(bad, 'bad')
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    extract_archive(archive, root/'unpacked')
                self.assertFalse((root/'unpacked').exists())
            with zipfile.ZipFile(root/'artifact.zip', 'w') as zipped:
                info = zipfile.ZipInfo('link')
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                zipped.writestr(info, '/etc/passwd')
            with self.assertRaises(ValueError):
                extract_archive(root/'artifact.zip', root/'unpacked')


class SourceSelectionTest(unittest.TestCase):
    def test_rerun_selects_only_its_exact_attempt_and_dispatch_has_no_payload_code(self):
        from scripts.ci.tests import test_result_ingestion as fixtures
        from scripts.ci.ingest_benchmark_results import selected_source, request_publication
        fixture=fixtures.IngestionTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        run=dict(fixture.run,run_attempt=2,run_started_at='2026-10-18T00:30:00Z')
        older=dict(fixture.artifact,id=455,created_at='2026-10-18T00:00:00Z')
        current=dict(fixture.artifact,name='linux-cpu-benchmark-results-123-attempt2')
        with patch('scripts.ci.ingest_benchmark_results.gh_json',side_effect=[run,{'total_count':2,'artifacts':[older,current]}]):
            profile,source,artifact=selected_source(123)
        self.assertEqual((profile,source['run_attempt'],artifact['id']),('amd-cpu',2,456))
        with patch('scripts.ci.ingest_benchmark_results.gh_json',side_effect=[run,{'total_count':1,'artifacts':[older]}]):
            with self.assertRaisesRegex(ValueError,'exactly one'):
                selected_source(123)
        with patch('scripts.ci.ingest_benchmark_results.subprocess.run') as post:
            request_publication()
        self.assertEqual(json.loads(post.call_args.kwargs['input']),{'event_type':'benchmark-results-ingested'})
        self.assertIn('repos/tensor4all/tenferro-benchmark/dispatches',post.call_args.args[0])

    def test_gated_gpu_slot_without_collection_is_a_no_op_but_a_lost_artifact_is_not(self):
        from scripts.ci.ingest_benchmark_results import selected_source
        run={'id':789,'run_attempt':1,'path':'.github/workflows/benchmark-runpod-gpu.yml',
             'run_started_at':'2026-10-18T08:00:00Z', 'head_branch':'main',
             'repository':{'full_name':'tensor4all/tenferro-benchmark'},
             'head_repository':{'full_name':'tensor4all/tenferro-benchmark'},
             'status':'completed', 'conclusion':'success'}
        no_artifacts={'total_count':0,'artifacts':[]}
        def jobs(conclusion):
            return {'jobs':[{'name':'Skip if this week\'s GPU results already exist','conclusion':'success'},
                            {'name':'Run GPU benchmark on RunPod','conclusion':conclusion}]}
        with patch('scripts.ci.ingest_benchmark_results.gh_json',side_effect=[run,no_artifacts,jobs('skipped')]) as api:
            self.assertIsNone(selected_source(789))
        self.assertIn('runs/789/attempts/1/jobs',api.call_args.args[0])
        with patch('scripts.ci.ingest_benchmark_results.gh_json',side_effect=[run,no_artifacts,jobs('success')]):
            with self.assertRaisesRegex(ValueError,'exactly one'):
                selected_source(789)
        cpu=dict(run,path='.github/workflows/benchmark-nightly-linux-cpu.yml')
        with patch('scripts.ci.ingest_benchmark_results.gh_json',side_effect=[cpu,no_artifacts]):
            with self.assertRaisesRegex(ValueError,'exactly one'):
                selected_source(789)


class CommitRaceTest(unittest.TestCase):
    def test_race_replays_only_results_and_preserves_competing_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, ingest, competitor = root/'remote.git', root/'ingest', root/'competitor'
            def run(*args):
                return subprocess.run(['git', *map(str, args)], capture_output=True, text=True, check=True)
            run('init', '--bare', '--initial-branch=main', remote)
            run('clone', remote, competitor)
            run('-C', competitor, 'config', 'user.email', 'fixture@example.invalid')
            run('-C', competitor, 'config', 'user.name', 'Fixture')
            (competitor/'README.md').write_text('baseline')
            run('-C', competitor, 'add', 'README.md')
            run('-C', competitor, 'commit', '-m', 'baseline')
            run('-C', competitor, 'push', 'origin', 'main')
            run('clone', remote, ingest)
            source = {'run_id': 123, 'run_attempt': 1, 'artifact_id': 456}
            receipt = {'schema_version': 1, 'target_profile': 'amd-cpu', 'suite_id': 'cpu/einsum',
                       'collected_at': '2026-10-18T00:00:00Z', 'source': source, 'files': {}}
            payload = {'result/amd-cpu/cpu/einsum.md': b'validated report',
                       'result/amd-cpu/cpu/einsum.summary.json': b'{"timestamp":"2026-10-18T00:00:00Z"}'}
            original = subprocess.run
            raced = False
            merged_branches = []
            def merge(branch, commit, title):
                # Stands in for the PR merge: lands the exact commit or reports a moved main.
                merged_branches.append(branch)
                landed = original(['git', '-C', str(ingest), 'push', 'origin', f'{commit}:refs/heads/main'],
                                  capture_output=True, text=True)
                return commit if landed.returncode == 0 else None
            def racing_run(args, **kwargs):
                nonlocal raced
                if args[:3] == ['git', '-C', str(ingest)] and 'push' in args and not raced:
                    raced = True
                    (competitor/'README.md').write_text('unrelated concurrent change')
                    run('-C', competitor, 'add', 'README.md')
                    run('-C', competitor, 'commit', '-m', 'competing change')
                    run('-C', competitor, 'push', 'origin', 'main')
                return original(args, **kwargs)
            with patch('scripts.ci.ingest_benchmark_results.subprocess.run', side_effect=racing_run), \
                 patch('scripts.ci.ingest_benchmark_results.merge_through_pull_request', side_effect=merge):
                decisions, commit = commit_results(ingest, [(receipt, payload)], source)
            self.assertTrue(raced)
            self.assertEqual(merged_branches, ['ingest/run-123-attempt-1'] * 2)
            self.assertIn('ingest/run-123-attempt-1', run('--git-dir', remote, 'branch', '--list').stdout)
            self.assertEqual(decisions, {'cpu/einsum': 'accepted'})
            self.assertEqual((ingest/'README.md').read_text(), 'unrelated concurrent change')
            changed = set(run('-C', ingest, 'show', '--format=', '--name-only', commit).stdout.splitlines())
            self.assertEqual(changed, set(payload) | {'result/amd-cpu/cpu/einsum.provenance.json'})
            self.assertEqual(run('--git-dir', remote, 'rev-parse', 'main').stdout.strip(), commit)
            self.assertEqual(commit_results(ingest, [(receipt, payload)], source), ({'cpu/einsum': 'duplicate'}, None))


class PullRequestMergeTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.responses = {}
        def fake(*args, check=True):
            self.calls.append(args)
            for key, (code, body) in self.responses.items():
                if key in args:
                    return subprocess.CompletedProcess(args, code, json.dumps(body), 'error' if code else '')
            raise AssertionError(f'unexpected call {args}')
        patcher = patch('scripts.ci.ingest_benchmark_results.gh_pull_request', side_effect=fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def checks(self, *runs):
        return {'check_runs': [{'name': name, 'status': status, 'conclusion': conclusion,
                                'started_at': started} for name, status, conclusion, started in runs]}

    def test_opens_pr_waits_for_required_checks_and_merges_the_pinned_commit(self):
        from scripts.ci.ingest_benchmark_results import merge_through_pull_request
        listing = 'repos/tensor4all/tenferro-benchmark/pulls?head=tensor4all:ingest/run-1-attempt-1&state=open'
        self.responses = {listing: (0, []), 'repos/tensor4all/tenferro-benchmark/pulls': (0, {'number': 7}),
                          'repos/tensor4all/tenferro-benchmark/pulls/7/merge': (0, {'sha': 'squashed'})}
        rules = [{'type': 'deletion'}, {'type': 'required_status_checks', 'parameters': {
            'required_status_checks': [{'context': 'validate'}, {'context': 'followup-mwe'}]}}]
        pending = self.checks(('validate', 'in_progress', None, '2'), ('deploy', 'completed', 'failure', '1'))
        done = self.checks(('validate', 'completed', 'failure', '1'), ('validate', 'completed', 'success', '3'),
                           ('followup-mwe', 'completed', 'success', '2'))
        with patch('scripts.ci.ingest_benchmark_results.gh_json', side_effect=[rules, pending, done]), \
             patch('scripts.ci.ingest_benchmark_results.time.sleep'):
            self.assertEqual(merge_through_pull_request('ingest/run-1-attempt-1', 'abc', 'Accept'), 'squashed')
        merge = self.calls[-1]
        self.assertIn('sha=abc', merge)
        self.assertIn('merge_method=squash', merge)

    def test_failed_required_check_stops_without_merging(self):
        from scripts.ci.ingest_benchmark_results import wait_for_checks
        failed = self.checks(('validate', 'completed', 'failure', '1'), ('followup-mwe', 'in_progress', None, '1'))
        with patch('scripts.ci.ingest_benchmark_results.gh_json', return_value=failed):
            with self.assertRaisesRegex(RuntimeError, 'validate'):
                wait_for_checks('abc', {'validate', 'followup-mwe'})

    def test_unmergeable_pr_asks_for_a_replay(self):
        from scripts.ci.ingest_benchmark_results import merge_through_pull_request
        listing = 'repos/tensor4all/tenferro-benchmark/pulls?head=tensor4all:b&state=open'
        self.responses = {listing: (0, [{'number': 7}]),
                          'repos/tensor4all/tenferro-benchmark/pulls/7/merge': (1, {}),
                          'repos/tensor4all/tenferro-benchmark/pulls/7': (0, {'merged': False, 'mergeable': False})}
        with patch('scripts.ci.ingest_benchmark_results.required_checks', return_value=set()), \
             patch('scripts.ci.ingest_benchmark_results.gh_json', return_value={'check_runs': []}):
            self.assertIsNone(merge_through_pull_request('b', 'abc', 'Accept'))


if __name__ == '__main__':
    unittest.main()
