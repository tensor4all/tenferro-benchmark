"""Preloaded libraries may be reused only when their content is identical."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.ci.reuse_preloaded_sdk import reuse

class ReusePreloadedSdkTests(unittest.TestCase):
    def fixture(self, root, name='libfixture.so.12'):
        sdk, image = root / 'sdk', root / 'image'
        for tree in (sdk, image):
            (tree / 'targets/x86_64-linux/lib').mkdir(parents=True)
            (tree / 'targets/x86_64-linux/lib' / name).write_bytes(b'library')
        manifest = {name: {'bytes': 7, 'sha256': hashlib.sha256(b'library').hexdigest()}}
        return sdk, image, manifest

    def test_prepare_and_restore_preserve_exact_library_and_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            sdk, image, manifest = self.fixture(Path(temp))
            lib = sdk / 'targets/x86_64-linux/lib'
            (lib / 'libfixture.so').symlink_to('libfixture.so.12')
            self.assertEqual(reuse(sdk, image, manifest, prepare=True), 7)
            self.assertFalse((lib / 'libfixture.so.12').exists())
            self.assertEqual(reuse(sdk, image, manifest, prepare=False), 7)
            self.assertEqual((lib / 'libfixture.so').read_bytes(), b'library')
            self.assertTrue((lib / 'libfixture.so.12').is_symlink())

    def test_corrupt_preload_fails_without_installing_any_links(self):
        with tempfile.TemporaryDirectory() as temp:
            sdk, image, manifest = self.fixture(Path(temp))
            reuse(sdk, image, manifest, prepare=True)
            (image / 'targets/x86_64-linux/lib/libfixture.so.12').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'content differs'):
                reuse(sdk, image, manifest, prepare=False)
            self.assertFalse((sdk / 'targets/x86_64-linux/lib/libfixture.so.12').is_symlink())

    def test_invalid_staged_source_keeps_original_before_transfer(self):
        with tempfile.TemporaryDirectory() as temp:
            sdk, image, manifest = self.fixture(Path(temp))
            manifest['libfixture.so.12']['bytes'] = 8
            with self.assertRaisesRegex(ValueError, 'size differs'):
                reuse(sdk, image, manifest, prepare=True)
            self.assertEqual((sdk / 'targets/x86_64-linux/lib/libfixture.so.12').read_bytes(), b'library')
