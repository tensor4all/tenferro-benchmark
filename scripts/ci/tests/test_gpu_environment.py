import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ci.gpu_environment import INPUTS, PREFIX, cache_key, check_tar, digest, identity, verify


class EnvironmentTests(unittest.TestCase):
    def test_key_binds_inputs_and_only_needed_python_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in INPUTS:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            original = identity('tenferro-cuda-trace,pytorch-cuda', root)
            self.assertEqual(original, identity('pytorch-cuda', root))
            self.assertNotEqual(cache_key(original), cache_key(identity('jax-cuda', root)))
            (root / 'uv.lock').write_text('changed dependencies')
            self.assertNotEqual(cache_key(original), cache_key(identity('pytorch-cuda', root)))

    @patch('scripts.ci.gpu_environment.platform.system', return_value='Linux')
    @patch('scripts.ci.gpu_environment.platform.machine', return_value='x86_64')
    def test_archive_digest_and_dependency_identity(self, *_):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'runtime.tar.zst'
            archive.write_bytes(b'runtime')
            sha = digest(archive)
            expected = {'platform': 'ubuntu-22.04-x86_64'}
            manifest = {'identity': expected, 'archive_sha256': sha}
            verify(archive, manifest, expected, sha)
            with self.assertRaises(ValueError):
                verify(archive, manifest, {'platform': 'other'}, sha)
            with self.assertRaises(ValueError):
                verify(archive, manifest, expected, 'untrusted output')
            archive.write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                verify(archive, manifest, expected, sha)

    def test_archive_paths_and_links(self):
        def entry(name, link=None):
            member = tarfile.TarInfo(name)
            if link is not None:
                member.type = tarfile.SYMTYPE
                member.linkname = link
            return member
        check_tar([entry('tenferro-benchmark-ci/venv/bin/python', PREFIX + '/python/bin/python'),
                   entry('tenferro-benchmark-ci/cuda/lib64/libnvrtc.so', 'libnvrtc.so.12')])
        for item in [entry('/etc/passwd'), entry('tenferro-benchmark-ci/../escape'),
                     entry('other/file'), entry('tenferro-benchmark-ci/lib/link', '/etc/passwd'),
                     entry('tenferro-benchmark-ci/lib/link', '../../escape')]:
            with self.subTest(name=item.name, link=item.linkname), self.assertRaises(ValueError):
                check_tar([item])


if __name__ == '__main__':
    unittest.main()
