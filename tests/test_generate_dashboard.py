"""Tests for the static results dashboard and summary sidecars."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_dashboard as dashboard  # noqa: E402
import report_summary  # noqa: E402


def summary_document(rows, *, hardware=None, timestamp="2026-10-17T00:00:00+00:00", commit="abc123"):
    return {
        "schema_version": report_summary.SCHEMA_VERSION,
        "target_profile": "nvidia-gpu",
        "suite_id": "gpu/dense",
        "timestamp": timestamp,
        "tenferro_commit": commit,
        "hardware": hardware or {"gpu": "NVIDIA A100 80GB PCIe"},
        "unit": "ms",
        "rows": rows,
    }


def row(case, backend, median, status="ok", group="matmul"):
    return {
        "case": case,
        "group": group,
        "backend": backend,
        "median": median,
        "status": status,
        "verified": status == "ok",
    }


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = self.root / "result"
        self.baseline = self.root / "baseline"
        (self.result / "nvidia-gpu" / "gpu").mkdir(parents=True)
        (self.baseline / "nvidia-gpu" / "gpu").mkdir(parents=True)
        (self.result / "linux-cpu" / "cpu").mkdir(parents=True)

        # Two cases: one with ratio/status/regression flags, one status-only.
        current_rows = [
            row("case_a", "tenferro-cuda-eager", 20.0),
            row("case_a", "tenferro-cuda-trace", 30.0),
            row("case_a", "pytorch-cuda", 10.0),
            row("case_a", "jax-cuda", 5.0),
            row("case_b", "tenferro-cuda-eager", 5.0),
            row("case_b", "tenferro-cuda-trace", None, status="unsupported"),
            row("case_b", "pytorch-cuda", 10.0),
            row("case_b", "jax-cuda", 25.0),
        ]
        previous_rows = [
            row("case_a", "tenferro-cuda-eager", 10.0),
            row("case_a", "tenferro-cuda-trace", 29.0),
            row("case_a", "pytorch-cuda", 10.0),
            row("case_a", "jax-cuda", 5.0),
        ]
        (self.result / "nvidia-gpu/gpu/dense.md").write_text("# GPU Dense\n\nreport body\n")
        (self.result / "nvidia-gpu/gpu/dense.summary.json").write_text(
            json.dumps(summary_document(current_rows))
        )
        (self.baseline / "nvidia-gpu/gpu/dense.summary.json").write_text(
            json.dumps(summary_document(previous_rows))
        )
        # A maintained report with no summary must still render.
        (self.result / "linux-cpu/cpu/cpu_ops.md").write_text("# CPU Ops\n\nno summary here\n")

    def generate(self, output):
        return dashboard.generate(
            self.result,
            output,
            baseline_dir=self.baseline,
            git_repo=None,
        )

    def test_two_summaries_produce_expected_ratios_and_flags(self):
        current = json.loads((self.result / "nvidia-gpu/gpu/dense.summary.json").read_text())
        ratios = dashboard.suite_ratios(current)
        self.assertAlmostEqual(ratios["pytorch"]["eager"], 1.0, places=6)
        self.assertAlmostEqual(ratios["jax"]["eager"], (4.0 * 0.2) ** 0.5, places=6)

        grouped = dashboard.case_rows(current)
        flags_a = dashboard.compute_case_flags(grouped["case_a"], [row("case_a", "tenferro-cuda-eager", 10.0)])
        self.assertEqual(
            set(flags_a),
            {"eager/torch", "eager/jax", "trace/eager", "regression"},
        )
        flags_b = dashboard.compute_case_flags(grouped["case_b"])
        self.assertEqual(flags_b, ["status"])

    def test_landing_and_suite_pages_render(self):
        output = self.root / "site"
        self.assertEqual(self.generate(output), 0)
        index = (output / "index.html").read_text()
        # Ratios and the flag count appear on the landing table.
        self.assertIn("eager 1.00x", index)
        self.assertIn("nvidia-gpu", index)
        self.assertIn("flag", index)
        # Flagged cases come first on the suite page.
        suite = (output / "nvidia-gpu/gpu/dense.html").read_text()
        self.assertIn("Flagged cases (2)", suite)
        self.assertIn("case_a", suite)
        self.assertIn("<svg", suite)
        self.assertIn("Original report", suite)
        # A profile without summaries still renders (report link, no ratio).
        self.assertTrue((output / "linux-cpu/cpu/cpu_ops.html").exists())
        self.assertIn('href="linux-cpu/cpu/cpu_ops.html"', index)
        self.assertIn("linux-cpu", index)
        # The reference-quality notice only applies to GitHub-hosted profiles.
        self.assertNotIn("Reference-quality indicators only", index.split("linux-cpu")[0])
        self.assertIn("Reference-quality indicators only", index)

    def test_generator_is_deterministic(self):
        first = self.root / "site-a"
        second = self.root / "site-b"
        self.generate(first)
        self.generate(second)
        self.assertEqual(
            (first / "index.html").read_bytes(),
            (second / "index.html").read_bytes(),
        )
        self.assertEqual(
            (first / "nvidia-gpu/gpu/dense.html").read_bytes(),
            (second / "nvidia-gpu/gpu/dense.html").read_bytes(),
        )

    def test_discover_reports_and_summary_loading(self):
        reports = dashboard.discover_reports(self.result)
        self.assertEqual(
            {(profile, suite) for profile, suite, _path in reports},
            {("nvidia-gpu", "gpu/dense"), ("linux-cpu", "cpu/cpu_ops")},
        )
        summaries = dashboard.load_summaries(self.result)
        self.assertIn(("nvidia-gpu", "gpu/dense"), summaries)


class RegenerateSummariesTests(unittest.TestCase):
    """The local regeneration script matches a report to its raw run."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = self.root / "result"
        self.data = self.root / "data/results"

    def write_run(self, profile, suite, stamp, records):
        run = self.data / profile / suite / stamp
        run.mkdir(parents=True)
        (run / "records.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        (run / "run.yaml").write_text(
            "\n".join(
                [
                    "schema_version: 1",
                    f"target_profile: {profile}",
                    f"suite_id: {suite}",
                    "timestamp: '2026-10-16T23:51:16+00:00'",
                    "tenferro_rs: {commit: abc123, dirty: false}",
                    "cuda: {device_name: NVIDIA A100 80GB PCIe}",
                    "",
                ]
            )
        )
        return run

    def test_regenerates_matching_run_and_skips_missing_raw(self):
        import regenerate_report_summaries as regenerate

        record = {
            "problem_id": "dense_matmul_f64_16",
            "backend": "tenferro-cuda-eager",
            "status": "ok",
            "timing": {"median_ms": 1.25, "iqr_ms": 0.1},
            "verification": {"status": "passed"},
        }
        self.write_run("nvidia-gpu", "gpu/dense", "20261016_235116", [record])
        (self.result / "nvidia-gpu/gpu").mkdir(parents=True)
        (self.result / "nvidia-gpu/gpu/dense.md").write_text(
            "# GPU Dense\n\n- Timestamp: `2026-10-16T23:51:16+00:00`\n"
        )
        # Declares a run that is not present; must not fall back to a stale run.
        self.write_run("mac-cpu", "cpu/permutation", "20260913_143137", [])
        (self.result / "mac-cpu/cpu").mkdir(parents=True)
        (self.result / "mac-cpu/cpu/permutation.md").write_text(
            "# CPU Permutation\n\n- Timestamp: `2026-09-23T12:05:30+00:00`\n"
        )

        written = regenerate.regenerate(self.result, self.data)
        self.assertEqual(written, [self.result / "nvidia-gpu/gpu/dense.summary.json"])
        document = json.loads(written[0].read_text())
        self.assertEqual(document["hardware"]["gpu"], "NVIDIA A100 80GB PCIe")
        self.assertEqual(document["rows"][0]["case"], "dense_matmul_f64_16")
        self.assertFalse((self.result / "mac-cpu/cpu/permutation.summary.json").exists())


class SummaryHelperTests(unittest.TestCase):
    def test_cpu_thread_report_merge_writes_summary(self):
        import format_cpu_thread_reports as thread_reports

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        run_dirs = []
        for threads in (1, 4):
            directory = root / "data/results/amd-cpu/cpu/einsum" / f"2026101{threads}_000000"
            directory.mkdir(parents=True)
            run_dirs.append(str(directory))
            (directory / "report.md").write_text(f"# CPU Einsum\n\n## Threads: {threads}\n\nbody\n")
            (directory / "run.yaml").write_text(
                "schema_version: 1\n"
                "target_profile: amd-cpu\n"
                "suite_id: cpu/einsum\n"
                f"timestamp: '2026-10-17T00:00:0{threads}+00:00'\n"
                "tenferro_rs: {commit: abc, dirty: false}\n"
                "environment:\n  env:\n"
                f"    RAYON_NUM_THREADS: '{threads}'\n"
                "    BENCHMARK_COMMIT: deadbeef\n"
            )
            document = summary_document([row("case", "tenferro-trace", float(threads))])
            document["hardware"] = {"cpu": "m5", "threads": [threads]}
            document["rows"][0]["group"] = f"Threads: {threads} / opt"
            (directory / "einsum.summary.json").write_text(json.dumps(document))
        thread_reports.combine(run_dirs, root=root)
        merged = json.loads((root / "result/amd-cpu/cpu/einsum.summary.json").read_text())
        self.assertEqual(merged["hardware"]["threads"], [1, 4])
        self.assertEqual(len(merged["rows"]), 2)

    def test_summary_path_and_round_trip(self):
        path = report_summary.summary_path_for_report(Path("result/nvidia-gpu/gpu/dense.md"))
        self.assertEqual(path, Path("result/nvidia-gpu/gpu/dense.summary.json"))
        rows = [
            report_summary.make_row("b", "backend", 2.0, iqr=0.1, group="g", verified=True),
            report_summary.make_row("a", "backend", 1.0, status="unsupported"),
        ]
        document = report_summary.build_document(
            rows,
            target_profile="nvidia-gpu",
            suite_id="gpu/dense",
            timestamp="t",
            tenferro_commit="c",
            hardware={"gpu": "A100"},
            unit="ms",
        )
        self.assertEqual([r["case"] for r in document["rows"]], ["a", "b"])
        with tempfile.TemporaryDirectory() as directory:
            out = report_summary.write_document(Path(directory) / "x.summary.json", document)
            self.assertEqual(json.loads(out.read_text()), document)

    def test_merge_thread_documents_unions_threads(self):
        first = summary_document([row("case", "tenferro-trace", 1.0)])
        first["hardware"] = {"cpu": "m5", "threads": [1]}
        first["rows"][0]["group"] = "Threads: 1 / opt"
        second = summary_document([row("case", "tenferro-trace", 2.0)])
        second["hardware"] = {"cpu": "m5", "threads": [4]}
        second["rows"][0]["group"] = "Threads: 4 / opt"
        merged = report_summary.merge_thread_documents([first, second])
        self.assertEqual(merged["hardware"]["threads"], [1, 4])
        self.assertEqual(len(merged["rows"]), 2)


if __name__ == "__main__":
    unittest.main()
