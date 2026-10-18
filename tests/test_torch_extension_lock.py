"""Unit tests for the stale torch-extension build-lock recovery.

No GPU and no ``torch`` import: ``scripts/torch_extension_lock.py`` is
stdlib-only, and the build directory is monkeypatched to a temporary dir.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch_extension_lock as lock


class BuildLockRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.build = Path(self.tmp.name) / "cutlass_gemm_f64"
        self.build.mkdir()
        self.lock_path = self.build / "lock"

    def _call(self, *, compiler_pids=None, **kwargs):
        kwargs.setdefault("build_directory", self.build)
        kwargs.setdefault("stream", io.StringIO())
        kwargs["compiler_pids"] = (compiler_pids if compiler_pids is not None
                                   else (lambda _build: []))
        return lock.ensure_build_lock_clear("cutlass_gemm_f64", **kwargs)

    def test_stale_lock_with_no_live_compiler_is_removed(self):
        self.lock_path.write_text("")
        old = time.time() - 10_000
        os.utime(self.lock_path, (old, old))
        stream = io.StringIO()
        self._call(stream=stream)
        self.assertFalse(self.lock_path.exists())
        message = stream.getvalue()
        self.assertIn(str(self.lock_path), message)
        self.assertIn("stale", message)

    def test_fresh_lock_with_live_compiler_is_not_removed(self):
        self.lock_path.write_text("")
        with self.assertRaises(lock.ExtensionBuildLockError):
            self._call(compiler_pids=lambda _build: [4242],
                       timeout=0.05, poll_seconds=0.01)
        self.assertTrue(self.lock_path.exists())

    def test_stale_lock_with_live_compiler_is_not_removed(self):
        self.lock_path.write_text("")
        old = time.time() - 10_000
        os.utime(self.lock_path, (old, old))
        with self.assertRaises(lock.ExtensionBuildLockError):
            self._call(compiler_pids=lambda _build: [4242],
                       timeout=0.05, poll_seconds=0.01)
        self.assertTrue(self.lock_path.exists())

    def test_missing_lock_is_untouched(self):
        self.assertFalse(self.lock_path.exists())
        self._call()
        self.assertFalse(self.lock_path.exists())

    def test_compiler_filter_requires_build_dir_in_cmdline(self):
        matching = self.build / "build.ninja"
        cmdlines = [
            (1, f"ninja -C {self.build}"),
            (2, f"/usr/bin/nvcc -c {matching}"),
            (3, "ninja -C /some/other/build"),
            (4, "/usr/bin/python3 scripts/benchmark_gpu_python.py"),
        ]
        self.assertEqual(lock.live_compiler_pids(self.build, cmdlines=cmdlines), [1, 2])

    def test_build_directory_falls_back_to_torch_extensions_dir(self):
        with patch.dict(os.environ, {"TORCH_EXTENSIONS_DIR": "/tmp/torch-ext"}):
            resolved = lock.resolve_build_directory("cutlass_gemm_f64")
        self.assertEqual(resolved, Path("/tmp/torch-ext/cutlass_gemm_f64"))


if __name__ == "__main__":
    unittest.main()
