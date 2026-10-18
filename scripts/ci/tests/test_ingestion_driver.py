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
            def racing_run(args, **kwargs):
                nonlocal raced
                if args[:3] == ['git', '-C', str(ingest)] and 'push' in args and not raced:
                    raced = True
                    (competitor/'README.md').write_text('unrelated concurrent change')
                    run('-C', competitor, 'add', 'README.md')
                    run('-C', competitor, 'commit', '-m', 'competing change')
                    run('-C', competitor, 'push', 'origin', 'main')
                return original(args, **kwargs)
            with patch('scripts.ci.ingest_benchmark_results.subprocess.run', side_effect=racing_run):
                decisions, commit = commit_results(ingest, [(receipt, payload)], source)
            self.assertTrue(raced)
            self.assertEqual(decisions, {'cpu/einsum': 'accepted'})
            self.assertEqual((ingest/'README.md').read_text(), 'unrelated concurrent change')
            changed = set(run('-C', ingest, 'show', '--format=', '--name-only', commit).stdout.splitlines())
            self.assertEqual(changed, set(payload) | {'result/amd-cpu/cpu/einsum.provenance.json'})
            self.assertEqual(run('--git-dir', remote, 'rev-parse', 'main').stdout.strip(), commit)
            self.assertEqual(commit_results(ingest, [(receipt, payload)], source), ({'cpu/einsum': 'duplicate'}, None))


if __name__ == '__main__':
    unittest.main()
