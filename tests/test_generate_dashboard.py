"""Tests for the static results dashboard and summary sidecars."""

from __future__ import annotations

import json
import re
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


def row(case, backend, median, status="ok", group="matmul", reason=None):
    result = {
        "case": case,
        "group": group,
        "backend": backend,
        "median": median,
        "status": status,
        "verified": status == "ok",
    }
    if reason is not None:
        result["reason"] = reason
    return result


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
            instances_dir=self.root / "instances",
        )

    def test_two_summaries_produce_expected_ratios_and_flags(self):
        current = json.loads((self.result / "nvidia-gpu/gpu/dense.summary.json").read_text())
        ratios = dashboard.suite_ratios(current, "gpu/dense")
        self.assertAlmostEqual(ratios["vs PyTorch"]["arms"]["eager"], 1.0, places=6)
        self.assertAlmostEqual(ratios["vs JAX"]["arms"]["eager"], (4.0 * 0.2) ** 0.5, places=6)

        grouped = dashboard.case_rows(current)
        flags_a = dashboard.compute_case_flags(grouped["case_a"], [row("case_a", "tenferro-cuda-eager", 10.0)])
        self.assertEqual(
            set(flags_a),
            {"eager/torch", "eager/jax", "trace/eager", "regression"},
        )
        flags_b = dashboard.compute_case_flags(grouped["case_b"])
        self.assertEqual(flags_b, ["status"])

    def test_slower_reference_backend_is_not_a_regression(self):
        previous = [row("case", "tenferro-cuda-trace", 0.55), row("case", "pytorch-cuda", 0.36)]
        current = [row("case", "tenferro-cuda-trace", 0.54), row("case", "pytorch-cuda", 0.47)]
        self.assertNotIn("regression", dashboard.compute_case_flags(current, previous))
        slower = [row("case", "tenferro-cuda-trace", 0.70), row("case", "pytorch-cuda", 0.36)]
        self.assertIn("regression", dashboard.compute_case_flags(slower, previous))

    def test_status_policy_is_not_an_anomaly(self):
        neutral = row("case_n", "tenferro-cuda-trace", None, status="unsupported",
                      reason="trace-session-setup-unavailable")
        unexpected = row("case_u", "tenferro-cuda-eager", None, status="unsupported")
        failed = row("case_f", "tenferro-cuda", None, status="failed", reason="bad launch")
        self.assertTrue(dashboard.is_neutral_unsupported(neutral))
        self.assertFalse(dashboard.is_tenferro_anomaly(neutral))
        self.assertTrue(dashboard.is_tenferro_anomaly(unexpected))
        self.assertTrue(dashboard.is_tenferro_anomaly(failed))
        self.assertEqual(dashboard.compute_case_flags([neutral]), [])
        self.assertEqual(dashboard.compute_case_flags([unexpected]), ["status"])
        document = summary_document([neutral, unexpected, failed])
        self.assertEqual(
            {key[0] for key in dashboard.not_measured_cases(document)}, {"case_n"}
        )
        self.assertEqual(
            {key[0] for key in dashboard.suite_flags(document, None)}, {"case_u", "case_f"}
        )

    def test_thread_counts_do_not_produce_false_regression(self):
        previous = summary_document([
            row("case", "tenferro-trace", 1.0, group="Threads: 1 / opt"),
            row("case", "pytorch-cpu", 1.0, group="Threads: 1 / opt"),
        ])
        current = summary_document([
            row("case", "tenferro-trace", 1.0, group="Threads: 1 / opt"),
            row("case", "tenferro-trace", 4.0, group="Threads: 4 / opt"),
            row("case", "pytorch-cpu", 1.0, group="Threads: 1 / opt"),
            row("case", "pytorch-cpu", 4.0, group="Threads: 4 / opt"),
        ])
        # The 4T workload has no previous 4T row; it must not be compared to 1T.
        self.assertEqual(dashboard.suite_flags(current, previous["rows"]), {})
        # The same workload still flags a real regression.
        regressed = summary_document([
            row("case", "tenferro-trace", 4.0, group="Threads: 1 / opt"),
        ])
        self.assertEqual(
            dashboard.suite_flags(regressed, previous["rows"]),
            {("case", "Threads: 1 / opt"): ["regression"]},
        )

    def test_linear_bars_use_a_zero_baseline(self):
        rows = [row("c", "backend-a", 1.0), row("c", "backend-b", 1.01)]
        svg = dashboard.svg_bar_chart(rows, "ms")
        widths = [float(width) for width in re.findall(r'<rect[^>]*width="([0-9.]+)"', svg)]
        self.assertEqual(len(widths), 2)
        # 1.00 and 1.01 are nearly equal with a zero baseline, not 2 px vs 440 px.
        self.assertLess(max(widths) / min(widths), 1.1)
        self.assertNotIn("log scale", svg)
        # A >10x spread switches to a labelled log scale.
        log_svg = dashboard.svg_bar_chart([row("c", "a", 1.0), row("c", "b", 100.0)], "ms")
        self.assertIn("log scale", log_svg)

    def test_permutation_and_perf_issue_ratio_mapping(self):
        permutation = summary_document([
            row("pat", "tenferro-cuda-transpose", 2.0, group="pat"),
            row("pat", "tenferro-cuda-to-contiguous", 1.0, group="pat"),
            row("pat", "pytorch-cuda", 4.0, group="pat"),
            row("pat", "cutensor", 2.0, group="pat"),
        ])
        ratios = dashboard.suite_ratios(permutation, "gpu/permutation")
        self.assertEqual(set(ratios), {"vs PyTorch", "vs cuTENSOR"})
        self.assertAlmostEqual(ratios["vs PyTorch"]["arms"]["transpose"], 0.5)
        self.assertAlmostEqual(ratios["vs PyTorch"]["arms"]["to_contiguous"], 0.25)
        self.assertAlmostEqual(ratios["vs cuTENSOR"]["arms"]["transpose"], 1.0)
        self.assertAlmostEqual(ratios["vs cuTENSOR"]["arms"]["to_contiguous"], 0.5)

        perf = summary_document([
            row("transfer_tenferro_8B", "tenferro-cuda", 10.0, group="steady_state"),
            row("transfer_cudarc_8B", "cudarc", 5.0, group="steady_state"),
        ])
        reference_map = {"transfer_tenferro_8B": ["transfer_cudarc_8B"]}
        ratios = dashboard.suite_ratios(perf, "gpu/perf_issues", reference_map)
        self.assertAlmostEqual(ratios["vs reference"]["arms"]["tenferro-cuda"], 2.0)
        details = dashboard.perf_reference_details(perf, reference_map)
        self.assertEqual(len(details), 1)
        self.assertAlmostEqual(details[0]["ratio"], 2.0)
        self.assertEqual(details[0]["reference_case"], "transfer_cudarc_8B")

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

    def test_missing_declared_raw_run_never_falls_back_to_a_newer_run(self):
        import regenerate_report_summaries as regenerate

        record = {
            "problem_id": "dense_matmul_f64_16",
            "backend": "tenferro-cuda-eager",
            "status": "ok",
            "timing": {"median_ms": 1.25, "iqr_ms": 0.1},
            "verification": {"status": "passed"},
        }
        # The tracked report names R1, which is gone; a newer unpublished R2 exists.
        self.write_run("nvidia-gpu", "gpu/dense", "20261017_000000", [record])
        (self.result / "nvidia-gpu/gpu").mkdir(parents=True)
        (self.result / "nvidia-gpu/gpu/dense.md").write_text(
            "# GPU Dense\n\n- Raw run: `data/results/nvidia-gpu/gpu/dense/20261016_235116`\n"
        )
        written = regenerate.regenerate(self.result, self.data)
        self.assertEqual(written, [])
        self.assertFalse((self.result / "nvidia-gpu/gpu/dense.summary.json").exists())


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
