import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ci.gpu_artifact import main


class ArtifactTests(unittest.TestCase):
    def invoke(self, binary, commits=('harness', 'tenferro'), write=False):
        argv = ['gpu_artifact.py', str(binary)] + (['--write'] if write else [])
        with patch('sys.argv', argv), patch(
            'scripts.ci.gpu_artifact.subprocess.check_output', side_effect=commits
        ), contextlib.redirect_stdout(io.StringIO()):
            main()

    def test_matching_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'benchmark_gpu_rust'
            binary.write_bytes(b'built binary')
            self.invoke(binary, write=True)
            self.invoke(binary)

    def test_modified_binary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'benchmark_gpu_rust'
            binary.write_bytes(b'built binary')
            self.invoke(binary, write=True)
            binary.write_bytes(b'old or modified binary')
            with self.assertRaises(SystemExit):
                self.invoke(binary)

    def test_different_source_commits_are_rejected(self):
        for commits in [('other-harness', 'tenferro'), ('harness', 'other-tenferro')]:
            with self.subTest(commits=commits), tempfile.TemporaryDirectory() as directory:
                binary = Path(directory) / 'benchmark_gpu_rust'
                binary.write_bytes(b'built binary')
                self.invoke(binary, write=True)
                with self.assertRaises(SystemExit):
                    self.invoke(binary, commits=commits)
