"""Trend points: one point per recorded measurement, with its recorded conditions."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import trend_points  # noqa: E402
from campaign_toolkit import trend  # noqa: E402

EINSUM_CASE = "bin_batched_matmul_b32_m128_n128_k128"
EINSUM_GROUP = "Threads: {threads} / opt_flops"
OVERHEAD_CASE = "typed_elemwise_f64_add_bcast_n16"
OVERHEAD_GROUP = "threads={threads} / scope={scope}"
COMMIT_V1 = "1111111111111111111111111111111111111111"
COMMIT_V2 = "2222222222222222222222222222222222222222"


def row(case: str, group: str, backend: str, median: float, **extra) -> dict:
    record = {"case": case, "group": group, "backend": backend, "median": median,
              "iqr": 0.01, "status": "ok", "verified": True}
    record.update(extra)
    return record


def einsum_rows(latency: float = 2.0) -> list[dict]:
    rows = []
    for threads in (1, 8):
        group = EINSUM_GROUP.format(threads=threads)
        rows += [
            row(EINSUM_CASE, group, "tenferro-eager", latency),
            row(EINSUM_CASE, group, "pytorch-cpu", latency * 1.2),
            row(EINSUM_CASE, group, "jax-cpu", latency * 0.9),
        ]
    # Not selected: another strategy, another case, and a failed row of the selected case.
    rows.append(row(EINSUM_CASE, EINSUM_GROUP.format(threads=1).replace("opt_flops", "opt_size"),
                    "tenferro-eager", 9.0))
    rows.append(row("bin_matmul_1024", EINSUM_GROUP.format(threads=1), "tenferro-eager", 35.0))
    rows.append({"case": EINSUM_CASE, "group": EINSUM_GROUP.format(threads=1),
                 "backend": "tenferro-trace", "median": None, "status": "failed", "verified": False})
    return rows


def overhead_rows() -> list[dict]:
    return [
        row(OVERHEAD_CASE, OVERHEAD_GROUP.format(threads=1, scope="steady_state_host_api"),
            "tenferro-rs", 762.6, timing_scope="steady_state_host_api", backend_thread_budget=1),
        # A steady-state row recorded at another run thread: the suite is 1T-only, so it is not selected.
        row(OVERHEAD_CASE, OVERHEAD_GROUP.format(threads=2, scope="steady_state_host_api"),
            "tenferro-rs", 800.0),
        # A run_prepared session-entry diagnostic, not the caller-held-engine steady state.
        row("prepared_mul_add_f64_budget1_32x32x32_session_entry_diagnostic",
            OVERHEAD_GROUP.format(threads=1, scope="session_entry_diagnostic"),
            "tenferro-rs", 20883.2, backend_thread_budget=1),
    ]


def summary(schema_version: int, rows: list[dict], *, suite: str, timestamp: str, commit: str,
            unit: str = "ms", source_runs: list[dict] | None = None,
            timing_policy: dict | None = None) -> str:
    document = {
        "schema_version": schema_version,
        "target_profile": "zen5-cpu",
        "suite_id": suite,
        "timestamp": timestamp,
        "tenferro_commit": commit,
        "hardware": {"cpu": "AMD Ryzen AI 9 HX 470", "threads": [1, 8]},
        "unit": unit,
        "rows": rows,
    }
    if schema_version == 2:
        document["source_runs"] = source_runs or []
        document["timing_policy"] = timing_policy
    return json.dumps(document, indent=2) + "\n"


def report(runs: dict[int, str]) -> str:
    text = ["# CPU Benchmark Results", ""]
    for threads, run in sorted(runs.items()):
        text += [f"## Threads: {threads}", "", f"Source report: `{run}/report.md`.", ""]
    return "\n".join(text)


def run_yaml(policy: int = 2, blas: str = "mkl") -> str:
    return (f"timing_policy: {{version: {policy}}}\n"
            f"blas: {{implementation: {blas}, version: '2026.0.2'}}\n")


class Fixture(unittest.TestCase):
    """A throwaway git repository holding two publication commits of zen5-cpu summaries."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@example.invalid")
        self.git("config", "user.name", "t")

    def git(self, *args: str) -> None:
        subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True)

    def commit(self, subject: str, files: dict[str, str]) -> None:
        for relative, text in files.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", subject)

    def v1_commit(self) -> None:
        run_1t = "data/results/zen5-cpu/cpu/einsum/20260101_000000"
        run_8t = "data/results/zen5-cpu/cpu/einsum/20260102_000000"
        overhead_run = "data/results/zen5-cpu/cpu/overhead_smoke/20260103_000000"
        self.commit(
            "zen5-cpu results, summary schema v1",
            {
                "result/zen5-cpu/cpu/einsum.md": report({1: run_1t, 8: run_8t}),
                "result/zen5-cpu/cpu/einsum.summary.json": summary(
                    1, einsum_rows(), suite="cpu/einsum",
                    timestamp="2026-01-04T00:00:00+00:00", commit=COMMIT_V1),
                "result/zen5-cpu/cpu/overhead_smoke.md":
                    f"# CPU Overhead Smoke\n\n- Raw run: `{overhead_run}`\n",
                "result/zen5-cpu/cpu/overhead_smoke.summary.json": summary(
                    1, overhead_rows(), suite="cpu/overhead_smoke",
                    timestamp="2026-01-04T01:00:00Z", commit=COMMIT_V1, unit="ns"),
                "result/zen5-cpu/cpu/cpu_ops.summary.json": summary(
                    1, [row("matmul", "suite=large / dtype=f64 / threads=1 / shape=1024x1024",
                            "tenferro-eager", 29.3)],
                    suite="cpu/einsum", timestamp="2026-01-04T02:00:00+00:00", commit=COMMIT_V1),
                # Only the 1T source run is tracked, so only that row's record is readable.
                f"{run_1t}/run.yaml": run_yaml(),
            },
        )

    def v2_commit(self) -> None:
        run = "data/results/zen5-cpu/cpu/einsum/20260201_000000"
        self.commit(
            "zen5-cpu results, summary schema v2",
            {
                "result/zen5-cpu/cpu/einsum.md": report({1: run, 8: run}),
                "result/zen5-cpu/cpu/einsum.summary.json": summary(
                    2, einsum_rows(1.5), suite="cpu/einsum",
                    timestamp="2026-02-02T00:00:00+00:00", commit=COMMIT_V2,
                    source_runs=[{
                        "run": run,
                        "threads": [1, 8],
                        "host": {"cpu": "AMD Ryzen AI 9 HX 470"},
                        "toolchain": {"rustc": "rustc 1.90.0"},
                        "power": "boost on, performance",
                    }],
                    timing_policy={"version": 2, "scope": "operation execution"}),
            },
        )

    def build(self) -> list[dict]:
        return trend_points.build_points(self.repo)

    def find(self, points: list[dict], *, suite: str, engine: str, threads: int,
             timestamp_prefix: str) -> dict:
        matches = [
            point for point in points
            if point["suite"] == suite and point["engine"] == engine
            and point["threads"] == threads and point["timestamp"].startswith(timestamp_prefix)
        ]
        self.assertEqual(len(matches), 1, matches)
        return matches[0]


class TrendPointsTest(Fixture):
    def test_points_validate_and_record_their_conditions(self):
        self.v1_commit()
        self.v2_commit()
        points = self.build()

        out = Path(self.temp.name) / "points.json"
        chart_dir = Path(self.temp.name) / "trend"
        self.assertEqual(
            trend_points.main(["--repo", str(self.repo), "--out", str(out), "--render", str(chart_dir)]), 0)
        # `read_points` validates the document against the toolkit schema.
        written = trend.read_points(out)
        self.assertEqual(written, points)
        # One embedded chart and one plotted-data table per selection.
        charts = sorted(chart_dir.glob("trend-*.svg"))
        self.assertEqual(len(charts), len(trend_points.SELECTIONS))
        self.assertEqual(len(sorted(chart_dir.glob("trend-*.md"))), len(charts))

        # Schema v1 records no source_runs: no power state, and the timing policy
        # comes from the source run.yaml only where that record is tracked.
        v1_1t = self.find(points, suite="cpu/einsum", engine="tenferro-eager", threads=1,
                          timestamp_prefix="2026-01-04")
        self.assertEqual(v1_1t["power"], "unrecorded")
        self.assertEqual(v1_1t["timing_policy"], "v2")
        self.assertEqual(v1_1t["vendor"], "mkl")
        self.assertEqual(v1_1t["run_id"], "data/results/zen5-cpu/cpu/einsum/20260101_000000")
        self.assertEqual(v1_1t["spread"], 0)
        self.assertIsNone(v1_1t["library_version"])
        v1_8t = self.find(points, suite="cpu/einsum", engine="tenferro-eager", threads=8,
                          timestamp_prefix="2026-01-04")
        self.assertEqual(v1_8t["run_id"], "data/results/zen5-cpu/cpu/einsum/20260102_000000")
        self.assertEqual(v1_8t["timing_policy"], "unrecorded")
        self.assertIsNone(v1_8t["vendor"])

        # A v2 summary records the source run, its power state and the policy.
        v2 = self.find(points, suite="cpu/einsum", engine="tenferro-eager", threads=1,
                       timestamp_prefix="2026-02-02")
        self.assertEqual(v2["power"], "boost on, performance")
        self.assertEqual(v2["timing_policy"], "v2")
        self.assertEqual(v2["median"], 1.5)
        self.assertEqual(v2["library_commit"], COMMIT_V2)
        self.assertEqual(v2["run_id"], "data/results/zen5-cpu/cpu/einsum/20260201_000000")

        # Only the selected case and thread rows are exported, for every recorded engine.
        engines = {point["engine"] for point in points if point["suite"] == "cpu/einsum"}
        self.assertEqual(engines, {"tenferro-eager", "pytorch-cpu", "jax-cpu"})
        self.assertEqual(
            {point["case_key"] for point in points},
            {selection.case_key for selection in trend_points.SELECTIONS},
        )
        self.assertEqual(sorted({point["threads"] for point in points}), [1, 8])

    def test_carried_over_summary_and_unchanged_page_add_no_point(self):
        self.v1_commit()
        before = {(point["run_id"], point["engine"], point["threads"]) for point in self.build()}
        self.v2_commit()
        after = {(point["run_id"], point["engine"], point["threads"]) for point in self.build()}
        # The v2 commit rewrites the einsum summary and carries cpu_ops over unchanged.
        self.assertEqual(len(after - before), 6)
        self.assertEqual(len(before - after), 0)

        # A commit that only adds a report carries every summary over: nothing is measured twice.
        self.commit("zen5-cpu cpu_ops page", {"result/zen5-cpu/cpu/cpu_ops.md": "# CPU Ops\n"})
        self.assertEqual(
            {(point["run_id"], point["engine"], point["threads"]) for point in self.build()},
            after,
        )

    def test_overhead_smoke_exports_its_one_run_thread_steady_state_row_only(self):
        self.v1_commit()
        points = [point for point in self.build() if point["suite"] == "cpu/overhead_smoke"]
        self.assertEqual(len(points), 1, points)
        point = points[0]
        self.assertEqual(point["threads"], 1)
        self.assertEqual(point["case_key"],
                         "typed_elemwise_f64_add_bcast_n16 (scope=steady_state_host_api)")
        self.assertEqual(point["unit"], "ns")
        self.assertEqual(point["power"], "unrecorded")
        self.assertEqual(point["run_id"], "data/results/zen5-cpu/cpu/overhead_smoke/20260103_000000")


class TrendPageTest(Fixture):
    """The CPU pages embed the rendered charts and link them, split by regime."""

    def setUp(self):
        super().setUp()
        self.v1_commit()
        (self.repo / "benchmarks/cpu").mkdir(parents=True)
        (self.repo / "benchmarks/cpu/einsum.yaml").write_text(
            "schema_version: 1\nsuite_id: cpu/einsum\ndefaults:\n  regime: throughput\n"
            f"problems:\n  - id: {EINSUM_CASE}\n")
        (self.repo / "benchmarks/cpu/overhead_smoke.yaml").write_text(
            "schema_version: 1\nsuite_id: cpu/overhead_smoke\ndefaults:\n  regime: latency\n"
            f"problems:\n  - id: {OVERHEAD_CASE}\n")

    def site(self, trend_dir: Path | None) -> Path:
        import generate_dashboard as dashboard

        site = Path(self.temp.name) / "site"
        self.assertEqual(dashboard.generate(
            self.repo / "result", site, git_repo=None,
            instances_dir=Path(self.temp.name) / "instances",
            benchmarks_dir=self.repo / "benchmarks",
            latest_profiles=("zen5-cpu",), trend_dir=trend_dir), 0)
        return site

    def test_cpu_pages_embed_and_link_the_rendered_charts(self):
        trend_dir = Path(self.temp.name) / "site/trend"
        self.assertEqual(trend_points.main([
            "--repo", str(self.repo), "--out", str(trend_dir / "points.json"),
            "--render", str(trend_dir)]), 0)
        site = self.site(trend_dir)

        throughput = (site / "cpu/index.html").read_text()
        latency = (site / "cpu/latency/index.html").read_text()
        self.assertIn("<h2>Trend</h2>", throughput)
        self.assertIn("cpu/einsum: bin_batched_matmul_b32_m128_n128_k128", throughput)
        self.assertIn("<svg", throughput)
        self.assertIn('href="../trend/points.json"', throughput)
        self.assertIn('href="#trend"', throughput)
        # The per-call overhead case is a latency case: its chart stays on the latency page.
        self.assertNotIn("cpu/overhead_smoke:", throughput)
        self.assertIn("<h2>Trend</h2>", latency)
        self.assertIn("cpu/overhead_smoke: typed_elemwise_f64_add_bcast_n16", latency)

    def test_trend_links_resolve_from_each_family_page(self):
        trend_dir = Path(self.temp.name) / "site/trend"
        self.assertEqual(trend_points.main([
            "--repo", str(self.repo), "--out", str(trend_dir / "points.json"),
            "--render", str(trend_dir)]), 0)
        site = self.site(trend_dir)
        for page in (site / "cpu/index.html", site / "cpu/latency/index.html"):
            html = page.read_text()
            self.assertIn("trend/points.json", html)
            for href in re.findall(r'href="([^"#]+)"', html):
                if "://" in href or href.startswith("mailto:"):
                    continue
                self.assertTrue((page.parent / href).exists(), f"{page}: {href}")

    def test_no_trend_directory_renders_no_trend_section(self):
        site = self.site(None)
        self.assertNotIn('<h2>Trend</h2>', (site / "cpu/index.html").read_text())
        self.assertNotIn('<h2>Trend</h2>', (site / "cpu/latency/index.html").read_text())


if __name__ == "__main__":
    unittest.main()
