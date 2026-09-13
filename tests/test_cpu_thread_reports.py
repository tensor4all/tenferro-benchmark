import sys
from pathlib import Path
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from format_cpu_thread_reports import combine, REPORTS


class ThreadReports(unittest.TestCase):
    def test_combines_both_threads_and_keeps_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [self.fixture(root, 4), self.fixture(root, 1)]
            combine(runs, root)
            for name in REPORTS:
                text = (root / "result/mac-cpu/cpu" / f"{name}.md").read_text()
                self.assertLess(text.index("## Threads: 1"), text.index("## Threads: 4"))
                self.assertIn("| case | 1.0 |", text)
                self.assertIn("| case | 4.0 |", text)
                self.assertIn("### Thread Environment", text)
                self.assertIn("run1", text)
                self.assertIn("run4", text)

    def test_rejects_mixed_commits_or_duplicate_threads(self):
        for change in ("tenferro", "benchmark", "threads"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                first = self.fixture(root, 1)
                second = self.fixture(root, 4)
                meta = yaml.safe_load((second / "run.yaml").read_text())
                if change == "tenferro":
                    meta["tenferro_rs"]["commit"] = "different"
                elif change == "benchmark":
                    meta["environment"]["env"]["BENCHMARK_COMMIT"] = "different"
                else:
                    meta["environment"]["env"]["RAYON_NUM_THREADS"] = "1"
                (second / "run.yaml").write_text(yaml.safe_dump(meta))
                with self.assertRaises(ValueError):
                    combine([first, second], root)

    @staticmethod
    def fixture(root, threads):
        run = root / f"run{threads}"
        run.mkdir()
        (run / "run.yaml").write_text(yaml.safe_dump({
            "target_profile": "mac-cpu", "tenferro_rs": {"commit": "core"},
            "environment": {"env": {"RAYON_NUM_THREADS": str(threads), "BENCHMARK_COMMIT": "bench"}},
        }))
        for filename, title in REPORTS.values():
            (run / filename).write_text(f"# {title}\n\n## Thread Environment\n\nthreads={threads}\n\n## Threads: {threads}\n\n| case | {threads}.0 |\n")
        return run


if __name__ == "__main__":
    unittest.main()
