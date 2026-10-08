import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ci.gpu_environment import INPUTS, PREFIX, assemble_bundle, cache_key, check_tar, digest, identity, split_bundle, verify
from scripts.collect_gpu_info import _cuda_runtime_version


class EnvironmentTests(unittest.TestCase):
    def test_transfer_reconstructs_exact_archive_and_rejects_missing_or_corrupt_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'bundle'
            transfer = Path(directory) / 'transfer'
            bundle.mkdir()
            (bundle / 'runtime.tar.zst').write_bytes(b'123456789')
            (bundle / 'runtime.json').write_text('{}')
            self.assertEqual(split_bundle(bundle, transfer, part_bytes=4), 3)
            assemble_bundle(transfer)
            self.assertEqual((transfer / 'runtime.tar.zst').read_bytes(), b'123456789')
            (transfer / 'runtime.part01').write_bytes(b'xxxx')
            with self.assertRaises(ValueError):
                assemble_bundle(transfer)
            (transfer / 'runtime.part01').unlink()
            with self.assertRaises(ValueError):
                assemble_bundle(transfer)

    @patch('scripts.collect_gpu_info._nvcc_runtime_version', return_value='11.8')
    @patch('scripts.collect_gpu_info.ctypes.CDLL')
    def test_runtime_metadata_uses_library_instead_of_base_image_compiler(self, load, nvcc):
        def version(argument):
            argument._obj.value = 12080
            return 0
        load.return_value.cudaRuntimeGetVersion.side_effect = version
        self.assertEqual(_cuda_runtime_version(), '12.8')
        nvcc.assert_not_called()

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
            expected = {'platform': 'ubuntu-24.04-x86_64'}
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
