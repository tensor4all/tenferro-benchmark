#!/usr/bin/env python3
"""Harness dirty state ignores untracked generated files but not tracked edits."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_run_metadata as meta  # noqa: E402


class HarnessDirtyTest(unittest.TestCase):
    def test_untracked_reports_do_not_dirty_the_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True)
            git("init", "-q")
            git("config", "user.email", "t@example.invalid")
            git("config", "user.name", "t")
            (repo / "tracked.txt").write_text("x\n")
            git("add", "tracked.txt")
            git("commit", "-q", "-m", "init")
            (repo / "result.md").write_text("generated\n")
            self.assertFalse(meta.collect_harness(repo)["dirty"])
            self.assertTrue(meta.git_dirty(repo))  # tenferro-rs keeps the strict check
            (repo / "tracked.txt").write_text("y\n")
            self.assertTrue(meta.collect_harness(repo)["dirty"])


if __name__ == "__main__":
    unittest.main()
