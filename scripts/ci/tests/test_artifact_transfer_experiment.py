import json
import tempfile
import unittest
from pathlib import Path

from scripts.ci.artifact_transfer_experiment import FILES, PROTOCOL, digest, prepare, report, verify


class TransferExperimentTests(unittest.TestCase):
    def prepare_fixture(self, root):
        source = root / 'source'
        source.mkdir()
        contents = [bytes(range(256)) * 7, b'pjrt' * 211, b'tutorial']
        for name, content in zip(FILES, contents):
            (source / name).write_bytes(content)
        parts = root / 'parts'
        manifest = prepare(source, parts)
        incoming = root / 'incoming'
        incoming.mkdir()
        for path in parts.glob('*/*'):
            incoming.joinpath(path.name).write_bytes(path.read_bytes())
        return source, incoming, manifest, digest(source / 'transfer.json')

    def test_both_transports_preserve_three_files_across_partition_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            source, incoming, manifest, sha = self.prepare_fixture(Path(directory))
            verify(source, 'single', sha)
            verify(incoming, 'split', sha)
            self.assertEqual(len(manifest['parts']), 5)
            self.assertLessEqual(max(row['bytes'] for row in manifest['parts']) -
                                 min(row['bytes'] for row in manifest['parts']), 1)
            for name in FILES:
                self.assertEqual(source.joinpath(name).read_bytes(), incoming.joinpath(name).read_bytes())

    def test_missing_or_corrupted_part_and_wrong_manifest_fail(self):
        for failure in ('missing', 'corrupt', 'manifest'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                _, incoming, _, sha = self.prepare_fixture(Path(directory))
                path = incoming / 'archives.part02'
                if failure == 'missing':
                    path.unlink()
                elif failure == 'corrupt':
                    path.write_bytes(b'x' * path.stat().st_size)
                else:
                    incoming.joinpath('transfer.json').write_text('{}')
                with self.assertRaises(ValueError):
                    verify(incoming, 'split', sha)

    def test_changed_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, _, sha = self.prepare_fixture(Path(directory))
            source.joinpath(FILES[0]).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify(source, 'single', sha)

    def test_predeclared_complete_suite_and_noise_gate(self):
        idle = {'cpu_busy_fraction': 0.0, 'receive_bytes_per_second': 0,
                'load1': 0, 'cpu_count': 4}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(report(root)['verdict'], 'INCONCLUSIVE')
            rows = [{'label': str(i), 'arm': arm, 'seconds': 100 if arm == 'single' else 50,
                     'before': dict(idle), 'after': dict(idle)}
                    for i, arm in enumerate(PROTOCOL['order'])]
            path = root / 'samples.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(report(root)['verdict'], 'PASS')
            rows[0]['seconds'] = 200
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(report(root)['verdict'], 'INCONCLUSIVE')
            rows[0]['seconds'] = 100
            rows[0]['after']['cpu_busy_fraction'] = 0.8
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(report(root)['verdict'], 'INCONCLUSIVE')


if __name__ == '__main__':
    unittest.main()
