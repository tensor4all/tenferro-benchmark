"""The per-commit history page lists only the latest-generation profiles."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_history as history  # noqa: E402


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@example.invalid")
        self.git("config", "user.name", "t")
        self.git("remote", "add", "origin", "git@github.com:tensor4all/tenferro-benchmark.git")

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True)

    def commit(self, subject, files):
        for rel, text in files.items():
            path = self.repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", subject)

    def test_only_latest_profiles_are_listed_with_their_tenferro_rev(self):
        self.commit("old amd", {"result/amd-cpu/cpu/einsum.md": "old\n"})
        self.commit(
            "first zen5",
            {
                "result/zen5-cpu/cpu/einsum.md": "a\n",
                "result/zen5-cpu/cpu/einsum.summary.json": '{"tenferro_commit": "0123456789abcdef"}',
            },
        )
        self.commit("unrelated", {"README.md": "x\n"})
        self.commit("zen5 cpu_ops", {"result/zen5-cpu/cpu/cpu_ops.md": "b\n"})
        out = self.repo / "_site"
        written = history.write_history(self.repo, out, ("zen5-cpu", "nvidia-gpu"))
        self.assertEqual(written, out / "history.html")
        page = written.read_text()
        self.assertIn("first zen5", page)
        self.assertIn("zen5 cpu_ops", page)
        self.assertIn("0123456789ab", page)
        self.assertIn("github.com/tensor4all/tenferro-benchmark/commit/", page)
        self.assertNotIn("old amd", page)
        self.assertNotIn("amd-cpu", page)
        self.assertNotIn("unrelated", page)
        # The newer commit carries einsum over and lists only cpu_ops as changed.
        self.assertLess(page.index("zen5 cpu_ops"), page.index("first zen5"))
        self.assertIn("Carried over (1)", page)

    def test_no_matching_history_writes_nothing(self):
        self.commit("old amd", {"result/amd-cpu/cpu/einsum.md": "old\n"})
        out = self.repo / "_site"
        self.assertIsNone(history.write_history(self.repo, out, ("zen5-cpu",)))
        self.assertFalse((out / "history.html").exists())


if __name__ == "__main__":
    unittest.main()
