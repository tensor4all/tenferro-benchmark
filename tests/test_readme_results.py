"""The README results table is generated from the published pages and checked for drift."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_readme_results as readme  # noqa: E402

PROFILES = """maintained:
  - zen5-cpu
optional:
  - zen2-cpu
"""

README = """# tenferro-benchmark

## Latest Results

Prose that is written by hand.

<!-- results:begin -->
<!-- results:end -->

More prose.
"""


class ReadmeResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "benchmarks").mkdir()
        (self.root / "benchmarks/profiles.yaml").write_text(PROFILES)
        (self.root / "README.md").write_text(README)
        self.page("zen5-cpu/cpu/einsum")
        self.page("zen5-cpu/cpu/cpu_ops")
        self.page("nvidia-gpu/gpu/dense")

    def page(self, relative: str, text: str = "# Report\n") -> Path:
        path = self.root / "result" / f"{relative}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def check(self) -> tuple[int, str]:
        stream = io.StringIO()
        code = readme.check(self.root, stream)
        return code, stream.getvalue()

    def test_table_lists_the_maintained_profiles_then_the_gpu_profile(self):
        self.page("amd-cpu/cpu/einsum")
        self.assertEqual(readme.update(self.root), 0)
        text = (self.root / "README.md").read_text()
        section = text.split(readme.BEGIN)[1].split(readme.END)[0]
        self.assertIn("| Target profile | Suite | Report |", section)
        self.assertIn("[result/zen5-cpu/cpu/cpu_ops.md](result/zen5-cpu/cpu/cpu_ops.md)", section)
        self.assertIn("| `nvidia-gpu` | `gpu/dense` |", section)
        self.assertNotIn("amd-cpu", section)
        # Profiles keep the declared order; suites sort within a profile.
        self.assertLess(section.index("zen5-cpu/cpu/cpu_ops.md"), section.index("zen5-cpu/cpu/einsum.md"))
        self.assertLess(section.index("zen5-cpu/cpu/"), section.index("nvidia-gpu/gpu/"))
        self.assertTrue(text.startswith("# tenferro-benchmark"))
        self.assertIn("More prose.", text)

    def test_check_is_clean_after_update(self):
        self.assertEqual(readme.update(self.root), 0)
        code, output = self.check()
        self.assertEqual(code, 0)
        self.assertIn("up to date (3 pages)", output)

    def test_check_reports_a_stale_section(self):
        self.assertEqual(readme.update(self.root), 0)
        page = self.page("zen5-cpu/cpu/public_api")
        code, output = self.check()
        self.assertEqual(code, 1)
        self.assertIn(f"+| `zen5-cpu` | `cpu/public_api` |", output)
        page.unlink()
        self.assertEqual(self.check()[0], 0)

    def test_check_detects_a_hand_edited_row(self):
        self.assertEqual(readme.update(self.root), 0)
        path = self.root / "README.md"
        path.write_text(path.read_text().replace("`gpu/dense`", "`gpu/dense_old`"))
        code, output = self.check()
        self.assertEqual(code, 1)
        self.assertIn("-| `nvidia-gpu` | `gpu/dense_old` |", output)
        self.assertIn("+| `nvidia-gpu` | `gpu/dense` |", output)

    def test_missing_markers_are_an_error(self):
        (self.root / "README.md").write_text("# tenferro-benchmark\n")
        self.assertEqual(readme.main(["--root", str(self.root), "--check"]), 1)
        self.assertEqual(readme.main(["--root", str(self.root)]), 1)

    def test_the_committed_readme_is_up_to_date(self):
        self.assertEqual(readme.check(ROOT), 0)


if __name__ == "__main__":
    unittest.main()
