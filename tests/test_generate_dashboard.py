"""Tests for the static results dashboard and summary sidecars."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
import unittest.mock
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_dashboard as dashboard  # noqa: E402
import report_summary  # noqa: E402
import report_summary  # noqa: E402
import result_status  # noqa: E402

ZERO_OUTCOMES = {
    kind: {"count": 0, "cases": []} for kind in ("failed", "skipped", "unsupported")
}


def summary_document(
    rows,
    *,
    hardware=None,
    timestamp="2026-10-17T00:00:00+00:00",
    commit="abc123",
    profile="nvidia-gpu",
    suite_id="gpu/dense",
):
    return {
        "schema_version": report_summary.SCHEMA_VERSION,
        "target_profile": profile,
        "suite_id": suite_id,
        "timestamp": timestamp,
        "tenferro_commit": commit,
        "hardware": hardware or {"gpu": "NVIDIA A100 80GB PCIe"},
        "unit": "ms",
        "rows": rows,
    }


def receipt_document(
    report,
    summary,
    *,
    profile="nvidia-gpu",
    suite_id="gpu/dense",
    status="complete",
    case_outcomes=None,
    missing_suites=None,
    conclusion="success",
    collected_at="2026-10-17T00:00:00+00:00",
    run_id=1,
    run_attempt=1,
    artifact_id=1,
):
    """An accepted-result receipt; ``None`` leaves an additive field out."""
    files = {}
    for candidate in (report, summary):
        relative = f"result/{profile}/{candidate.parent.name}/{candidate.name}"
        files[relative] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    source = {
        "repository": "tensor4all/tenferro-benchmark",
        "workflow_path": ".github/workflows/benchmark-runpod-gpu.yml",
        "head_branch": "main",
        "head_sha": "a" * 40,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "artifact_id": artifact_id,
        "run_started_at": "2026-10-17T00:00:00Z",
        "run_url": f"https://github.com/tensor4all/tenferro-benchmark/actions/runs/{run_id}",
        "artifact_url": f"https://github.com/tensor4all/tenferro-benchmark/actions/runs/{run_id}/artifacts/{artifact_id}",
        "created_at": "2026-10-17T01:00:00Z",
        "expires_at": "2027-01-17T00:00:00Z",
        "digest": "sha256:" + "b" * 64,
        "conclusion": conclusion,
    }
    document = {
        "schema_version": 1,
        "target_profile": profile,
        "suite_id": suite_id,
        "collected_at": collected_at,
        "harness_commit": "c" * 40,
        "tenferro_commit": "abc123",
        "hardware": {"gpu": "NVIDIA A100 80GB PCIe"},
        "source": source,
        "files": files,
    }
    for key, value in (
        ("status", status),
        ("case_outcomes", case_outcomes),
        ("missing_suites", missing_suites),
    ):
        if value is not None:
            document[key] = value
    return document


def row(case, backend, median, status="ok", group="matmul", reason=None, iqr=None):
    result = {
        "case": case,
        "group": group,
        "backend": backend,
        "median": median,
        "status": status,
        "verified": status == "ok",
    }
    if iqr is not None:
        result["iqr"] = iqr
    if reason is not None:
        result["reason"] = reason
    return result


class DashboardTests(unittest.TestCase):
    def test_reasoned_legacy_cpu_policy_skips_are_neutral(self):
        for status in ("skipped", "unsupported: no borrowed-session API"):
            entry=row("policy", "tenferro-trace", None, status=status, reason="setup-inclusive API excluded")
            with self.subTest(status=status):
                self.assertTrue(dashboard.is_neutral_unsupported(entry))
                self.assertFalse(dashboard.is_tenferro_anomaly(entry))
                self.assertEqual(dashboard.compute_case_flags([entry]),[])
                entry.pop("reason")
                self.assertTrue(dashboard.is_tenferro_anomaly(entry))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = self.root / "result"
        self.baseline = self.root / "baseline"
        (self.result / "nvidia-gpu" / "gpu").mkdir(parents=True)
        (self.baseline / "nvidia-gpu" / "gpu").mkdir(parents=True)
        (self.result / "mac-cpu" / "cpu").mkdir(parents=True)

        # Two cases: one with ratio/status/regression flags, one status-only.
        current_rows = [
            row("case_a", "tenferro-cuda-eager", 20.0, iqr=6.0),
            row("case_a", "tenferro-cuda-trace", 30.0),
            row("case_a", "pytorch-cuda", 10.0),
            row("case_a", "jax-cuda", 5.0),
            row("case_b", "tenferro-cuda-eager", 5.0),
            row("case_b", "tenferro-cuda-trace", None, status="unsupported"),
            row("case_b", "pytorch-cuda", 10.0),
            row("case_b", "jax-cuda", 25.0),
            row("case_lat", "tenferro-cuda-eager", 50.0, group="lat"),
            row("case_lat", "pytorch-cuda", 40.0, group="lat"),
            row("case_lat", "jax-cuda", 45.0, group="lat"),
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
        (self.result / "mac-cpu/cpu/cpu_ops.md").write_text("# CPU Ops\n\nno summary here\n")
        # Regimes and run counts come from the suite YAML.
        self.benchmarks = self.root / "benchmarks" / "gpu"
        self.benchmarks.mkdir(parents=True)
        (self.benchmarks / "dense.yaml").write_text(
            "schema_version: 1\nsuite_id: gpu/dense\ndefaults:\n"
            "  regime: throughput\n  run: {runs: 7}\n"
            "problems:\n  - {id: case_a, size_elements: 1048576}\n  - {id: case_b, size_elements: 1048576}\n"
            "  - {id: case_lat, regime: latency}\n"
        )

    def generate(self, output, latest_profiles=None):
        # The fixtures use a legacy profile (mac-cpu); None keeps every profile.
        return dashboard.generate(
            self.result,
            output,
            baseline_dir=self.baseline,
            git_repo=None,
            instances_dir=self.root / "instances",
            benchmarks_dir=self.root / "benchmarks",
            latest_profiles=latest_profiles,
        )

    def test_latest_generation_only_is_published_and_linked(self):
        # The default publication covers zen5-cpu and nvidia-gpu; the legacy
        # mac-cpu/amd-cpu reports are neither generated nor linked.
        for legacy in ("amd-cpu", "mac-cpu"):
            (self.result / legacy / "cpu").mkdir(parents=True, exist_ok=True)
            (self.result / legacy / "cpu" / "cpu_ops.md").write_text("# old\n")
        output = self.root / "site"
        self.assertEqual(self.generate(output, dashboard.LATEST_PROFILES), 0)
        pages = "".join(path.read_text() for path in output.rglob("*.html"))
        for legacy in ("amd-cpu", "mac-cpu", "mac-gpu"):
            self.assertNotIn(legacy, pages)
            self.assertFalse((output / legacy).exists(), legacy)
        self.assertTrue((output / "nvidia-gpu/gpu/dense.html").exists())
        index = (output / "index.html").read_text()
        self.assertIn('href="gpu/index.html"', index)
        # zen5-cpu has no committed report yet: an unlinked placeholder, not an old CPU page.
        self.assertNotIn('href="cpu/index.html"', index)
        self.assertIn("no results published yet", index)
        self.assertIn("zen5-cpu", index)

    def test_zen5_cpu_report_gets_the_cpu_family_page(self):
        (self.result / "zen5-cpu" / "cpu").mkdir(parents=True)
        (self.result / "zen5-cpu" / "cpu" / "einsum.md").write_text("# einsum\n")
        output = self.root / "site"
        self.assertEqual(self.generate(output, dashboard.LATEST_PROFILES), 0)
        index = (output / "index.html").read_text()
        self.assertIn('href="cpu/index.html"', index)
        self.assertIn("AMD Ryzen AI 9 HX 470", (output / "cpu/index.html").read_text())
        self.assertTrue((output / "zen5-cpu/cpu/einsum.html").exists())

    def test_zen5_cpu_latency_suites_count_as_latency_cases(self):
        # The per-call suites declare `regime: latency`; their maintained pages
        # must show up as latency cases on the CPU family card, never as compute.
        suites = {
            "cpu/overhead_smoke": ["typed_elemwise_f64_add_bcast_n16"],
            "cpu/small_work": ["add_f64_concrete_fresh"],
            "cpu/session_matrix": ["matmul_n2_count1024"],
        }
        cpu_benchmarks = self.root / "benchmarks" / "cpu"
        cpu_benchmarks.mkdir(parents=True)
        zen5 = self.result / "zen5-cpu" / "cpu"
        zen5.mkdir(parents=True)
        for suite_id, cases in suites.items():
            stem = suite_id.split("/", 1)[1]
            (zen5 / f"{stem}.md").write_text(f"# {suite_id}\n")
            (zen5 / f"{stem}.summary.json").write_text(
                json.dumps(
                    summary_document(
                        [row(case, "tenferro-rs", 100.0, group="ns/op") for case in cases]
                    )
                )
            )
            (cpu_benchmarks / f"{stem}.yaml").write_text(
                "schema_version: 1\n"
                f"suite_id: {suite_id}\n"
                "defaults:\n  regime: latency\n"
                "problems:\n" + "".join(f"  - {{id: {case}}}\n" for case in cases)
            )
        output = self.root / "site"
        self.assertEqual(self.generate(output, dashboard.LATEST_PROFILES), 0)
        index = (output / "index.html").read_text()
        match = re.search(
            r'<a class="family-card" href="cpu/index\.html">(.*?)</a>', index, re.S
        )
        self.assertIsNotNone(match)
        card = match.group(1)
        self.assertIn("0 compute cases", card)
        self.assertIn("3 latency cases", card)
        for suite in suites:
            self.assertTrue((output / "zen5-cpu" / f"{suite}.html").exists(), suite)

    def test_library_profile_supplies_labels_and_latest_profiles(self):
        self.assertEqual(dashboard.LATEST_PROFILES, ("zen5-cpu", "nvidia-gpu"))
        self.assertEqual(dashboard.profile_label("zen5-cpu"), "AMD Ryzen AI 9 HX 470")
        self.assertEqual(dashboard.profile_label("zen2-cpu"), "AMD EPYC 7702P")
        # GPU and legacy labels stay local.
        self.assertEqual(dashboard.profile_label("nvidia-gpu"), "NVIDIA A100")

    def test_two_summaries_produce_expected_ratios_and_flags(self):
        current = json.loads((self.result / "nvidia-gpu/gpu/dense.summary.json").read_text())
        ratios = dashboard.suite_ratios(current, "gpu/dense")
        self.assertAlmostEqual(
            ratios["vs PyTorch"]["arms"]["eager"], (2.0 * 0.5 * 1.25) ** (1 / 3), places=6
        )
        self.assertAlmostEqual(
            ratios["vs JAX"]["arms"]["eager"], (4.0 * 0.2 * (50.0 / 45.0)) ** (1 / 3), places=6
        )

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

    def test_case_chart_uses_log_axis_baseline_and_ci_whisker(self):
        rows = [
            row("c", "pytorch-cuda", 10.0, iqr=0.2),
            row("c", "tenferro-cuda-eager", 12.0, iqr=0.3),
        ]
        svg = dashboard.svg_case_chart(rows, "ms", runs=7, baseline=rows[0], baseline_label="PyTorch")
        self.assertIn("log scale", svg)
        self.assertIn("1.20x", svg)
        self.assertIn("PyTorch (10.000 ms)", svg)
        # The estimated CI is drawn as a black line over a white halo.
        self.assertIn('stroke="#1b1f24"', svg)
        # Without a sample count there is no interval.
        no_ci = dashboard.svg_case_chart(rows, "ms", runs=None, baseline=rows[0], baseline_label="PyTorch")
        self.assertNotIn('stroke="#1b1f24"', no_ci)

    def test_ratio_chart_outlines_competitor_bars(self):
        cases = [
            {
                "case": "case_one",
                "baseline_median": 10.0,
                "baseline_iqr": 0.4,
                "contenders": [
                    {
                        "label": "eager",
                        "kind": "tenferro",
                        "ratio": 1.2,
                        "median": 12.0,
                        "iqr": 0.3,
                        "interval": (1.1, 1.3),
                    },
                    {"label": "JAX", "kind": "reference", "ratio": 0.9, "median": 9.0, "interval": None},
                ],
            }
        ]
        svg = dashboard.svg_ratio_chart(cases, "PyTorch")
        # The bar label carries the ratio and both absolute medians.
        self.assertIn("eager 1.20x · 12 / 10 ms", svg)
        self.assertIn("JAX 0.90x · 9 / 10 ms", svg)
        # The legend says which side of "x / y" is which.
        self.assertIn("label = contender / PyTorch, ms", svg)
        # Competitor bars are white with a black outline.
        self.assertIn('fill="#ffffff" stroke="#1b1f24"', svg)
        # The tooltip keeps both medians and IQRs.
        self.assertIn("median 12.000 ms (IQR 0.3 ms) vs PyTorch median 10.000 ms (IQR 0.4 ms)", svg)

    def test_ratio_chart_labels_units_and_three_significant_digits(self):
        self.assertEqual(dashboard.fmt_sig(12.3456), "12.3")
        self.assertEqual(dashboard.fmt_sig(6.024), "6.02")
        self.assertEqual(dashboard.fmt_sig(0.0012345), "0.00123")
        cases = [
            {
                "case": "c",
                "baseline_median": 112034.0,
                "contenders": [
                    {"label": "tenferro-cuda", "kind": "tenferro", "ratio": 3.87, "median": 434054.0},
                ],
            }
        ]
        # A non-ms unit is printed on both values, so no row mixes units.
        svg = dashboard.svg_ratio_chart(cases, "declared reference", "ns/op")
        self.assertIn("tenferro-cuda 3.87x · 434000 ns/op / 112000 ns/op", svg)
        self.assertIn("label = contender / declared reference, ns/op", svg)

    def test_missing_median_is_not_invented(self):
        cases = [{"case": "c", "contenders": [{"label": "eager", "kind": "tenferro", "ratio": 1.2}]}]
        svg = dashboard.svg_ratio_chart(cases, "PyTorch")
        # No median on either side means no absolute numbers, not a made-up value.
        self.assertIn('class="chart-value">eager 1.20x</text>', svg)
        self.assertIn("eager 1.20x of PyTorch", svg)

    def test_noisy_cell_is_badged_and_the_ratio_is_not_asserted(self):
        # IQR/median = 50%: the run-to-run spread is half the median itself.
        cases = [
            {
                "case": "dense_solve_f64_512_rhs16",
                "baseline_median": 2.367,
                "baseline_iqr": 0.011,
                "contenders": [
                    {
                        "label": "eager",
                        "kind": "tenferro",
                        "ratio": 8.65,
                        "median": 20.476,
                        "iqr": 10.164,
                        "iqr_share": 0.496,
                        "noisy": True,
                    },
                    {"label": "trace", "kind": "tenferro", "ratio": 1.05, "median": 2.49, "iqr": 0.02, "noisy": False},
                ],
            }
        ]
        svg = dashboard.svg_ratio_chart(cases, "PyTorch")
        self.assertIn("eager ~8.65x · 20.5 / 2.37 ms", svg)
        self.assertIn('<tspan class="chart-noisy"> noisy</tspan>', svg)
        # A quiet cell keeps the plain ratio and gets no badge.
        self.assertIn("trace 1.05x · 2.49 / 2.37 ms</text>", svg)
        self.assertEqual(svg.count('<tspan class="chart-noisy">'), 1)
        # The legend states the rule and that the ratio is indicative.
        self.assertIn("noisy = IQR/median &gt; 0.2 (ratio is indicative)", svg)

    def test_noisy_threshold_defaults_and_reads_the_environment(self):
        self.assertEqual(dashboard.noisy_iqr_ratio(), 0.2)
        for raw, expected in (("0.35", 0.35), ("", 0.2), ("nonsense", 0.2), ("0", 0.2), ("-1", 0.2)):
            with self.subTest(raw=raw):
                with unittest.mock.patch.dict("os.environ", {"DASHBOARD_NOISY_IQR_RATIO": raw}):
                    self.assertEqual(dashboard.noisy_iqr_ratio(), expected)
        self.assertEqual(dashboard.iqr_share({"median": 20.0, "iqr": 10.0}), 0.5)
        self.assertIsNone(dashboard.iqr_share({"median": 20.0}))

    def test_baseline_noise_alone_makes_the_ratio_indicative(self):
        rows = [
            row("c", "tenferro-cuda-eager", 12.0, iqr=0.1),
            row("c", "pytorch-cuda", 10.0, iqr=6.0),
        ]
        case = dashboard.CaseView(case="c", group="", regime=dashboard.THROUGHPUT, runs=7, rows=rows)
        item = dashboard.case_ratio_row(case, "gpu/dense", {}, dashboard.case_rows(summary_document(rows)))
        self.assertAlmostEqual(item["baseline_iqr_share"], 0.6)
        self.assertTrue(item["contenders"][0]["noisy"])

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

    def test_top_family_and_suite_pages_render(self):
        output = self.root / "site"
        self.assertEqual(self.generate(output), 0)
        index = (output / "index.html").read_text()
        self.assertIn('href="gpu/index.html"', index)
        self.assertIn('href="cpu/index.html"', index)
        self.assertIn("NVIDIA A100 80GB PCIe", index)

        gpu = (output / "gpu/index.html").read_text()
        self.assertIn("How to read", gpu)
        self.assertIn("gpu/dense", gpu)
        self.assertIn("case_a", gpu)
        # The landing bars carry the absolute times behind each ratio.
        self.assertIn("trace 3.00x · 30 / 10 ms", gpu)
        self.assertIn("JAX 0.50x · 5 / 10 ms", gpu)
        self.assertIn("worst case 10 ms", gpu)
        # A spread-out cell is badged and its ratio is not asserted.
        self.assertIn("eager ~2.00x · 20 / 10 ms", gpu)
        self.assertIn('<tspan class="chart-noisy"> noisy</tspan>', gpu)
        self.assertIn("noisy = IQR/median", gpu)
        # A latency case never appears on the compute-bound page.
        self.assertNotIn("case_lat", gpu)

        latency = (output / "gpu/latency/index.html").read_text()
        self.assertIn("case_lat", latency)
        self.assertNotIn("case_a", latency)
        self.assertIn("not by device compute", latency)

        suite = (output / "nvidia-gpu/gpu/dense.html").read_text()
        self.assertIn("Throughput-regime cases", suite)
        self.assertIn("not a measured compute- or bandwidth-bound classification", suite)
        self.assertIn("Latency and per-call overhead", suite)
        self.assertIn("<svg", suite)
        self.assertIn("Original report", suite)

        cpu = (output / "cpu/index.html").read_text()
        self.assertIn("mac-cpu", cpu)
        self.assertIn("Reference-quality indicators only", cpu)
        self.assertTrue((output / "mac-cpu/cpu/cpu_ops.html").exists())

    def test_retired_profile_is_never_generated(self):
        (self.result / "linux-cpu" / "cpu").mkdir(parents=True)
        (self.result / "linux-cpu" / "cpu" / "linalg_jvp_jvp.md").write_text("# retired\n")
        self.assertEqual(
            {profile for profile, _suite, _path in dashboard.discover_reports(self.result)},
            {"nvidia-gpu", "mac-cpu"},
        )
        output = self.root / "site"
        self.assertEqual(self.generate(output), 0)
        self.assertNotIn("linux-cpu", "".join(path.read_text() for path in output.rglob("*.html")))
        self.assertFalse((output / "linux-cpu").exists())
        # An explicit retired profile fails instead of generating an empty page.
        with self.assertRaisesRegex(ValueError, "linux-cpu"):
            self.generate(self.root / "site-retired", ["linux-cpu"])

    def test_generator_is_deterministic(self):
        first = self.root / "site-a"
        second = self.root / "site-b"
        self.generate(first)
        self.generate(second)
        for relative in ("index.html", "gpu/index.html", "gpu/latency/index.html", "nvidia-gpu/gpu/dense.html"):
            self.assertEqual(
                (first / relative).read_bytes(),
                (second / relative).read_bytes(),
                relative,
            )

    def test_discover_reports_and_summary_loading(self):
        reports = dashboard.discover_reports(self.result)
        self.assertEqual(
            {(profile, suite) for profile, suite, _path in reports},
            {("nvidia-gpu", "gpu/dense"), ("mac-cpu", "cpu/cpu_ops")},
        )
        summaries = dashboard.load_summaries(self.result)
        self.assertIn(("nvidia-gpu", "gpu/dense"), summaries)


class BatchingComparabilityTests(unittest.TestCase):
    """#336: no ratio or flag between arms that batched different operation counts."""

    @staticmethod
    def batched(case, backend, median, count):
        result = row(case, backend, median)
        result["operations_per_sample"] = count
        return result

    def test_a_reference_with_different_batching_is_not_a_denominator(self):
        document = {"rows": [
            self.batched("a", "tenferro-cuda-eager", 4.0, 16),
            self.batched("a", "pytorch-cuda", 1.0, 32),
            self.batched("b", "tenferro-cuda-eager", 3.0, 32),
            self.batched("b", "pytorch-cuda", 1.0, 32),
        ]}
        ratios = dashboard.suite_ratios(document, "gpu/dense")
        # Only case b compares like with like.
        self.assertEqual(ratios["vs PyTorch"]["count"], 1)
        self.assertAlmostEqual(ratios["vs PyTorch"]["arms"]["eager"], 3.0)
        rows_a = dashboard.case_rows(document)["a"]
        self.assertNotIn("eager/torch", dashboard.compute_case_flags(rows_a))
        self.assertIn("eager/torch", dashboard.compute_case_flags(dashboard.case_rows(document)["b"]))

    def test_rows_without_a_recorded_count_are_still_compared(self):
        document = {"rows": [row("a", "tenferro-cuda-eager", 4.0), row("a", "pytorch-cuda", 1.0)]}
        self.assertAlmostEqual(
            dashboard.suite_ratios(document, "gpu/dense")["vs PyTorch"]["arms"]["eager"], 4.0)

    def test_the_gpu_summary_row_carries_the_recorded_count(self):
        record = {"problem_id": "a", "backend": "pytorch-cuda", "status": "ok", "op": "add",
                  "timing": {"median_ms": 0.1, "iqr_ms": 0.0},
                  "verification": {"status": "passed"},
                  "execution": {"notes": json.dumps({"sampling": {"operations_per_sample": 32}})}}
        [summary_row] = report_summary.rows_from_gpu_records([record])
        self.assertEqual(summary_row["operations_per_sample"], 32)
        record["execution"]["notes"] = "free text"
        [summary_row] = report_summary.rows_from_gpu_records([record])
        self.assertNotIn("operations_per_sample", summary_row)


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

    def record(self):
        return {
            "problem_id": "dense_matmul_f64_16",
            "backend": "tenferro-cuda-eager",
            "status": "ok",
            "timing": {"median_ms": 1.25, "iqr_ms": 0.1},
            "verification": {"status": "passed"},
        }

    def test_regenerates_matching_run(self):
        import regenerate_report_summaries as regenerate

        self.write_run("nvidia-gpu", "gpu/dense", "20261016_235116", [self.record()])
        (self.result / "nvidia-gpu/gpu").mkdir(parents=True)
        (self.result / "nvidia-gpu/gpu/dense.md").write_text(
            "# GPU Dense\n\n- Timestamp: `2026-10-16T23:51:16+00:00`\n"
        )

        written = regenerate.regenerate(self.result, self.data, suites={"gpu/dense"})
        self.assertEqual(written, [self.result / "nvidia-gpu/gpu/dense.summary.json"])
        document = json.loads(written[0].read_text())
        self.assertEqual(document["hardware"]["gpu"], "NVIDIA A100 80GB PCIe")
        self.assertEqual(document["rows"][0]["case"], "dense_matmul_f64_16")

    def test_missing_declared_raw_run_fails_instead_of_skipping_the_page(self):
        import regenerate_report_summaries as regenerate

        # The page declares a run that is not present; the older run recorded
        # under the page's suite directory must not be substituted for it.
        self.write_run("nvidia-gpu", "gpu/dense", "20260913_143137", [self.record()])
        (self.result / "nvidia-gpu/gpu").mkdir(parents=True)
        (self.result / "nvidia-gpu/gpu/dense.md").write_text(
            "# GPU Dense\n\n- Timestamp: `2026-09-23T12:05:30+00:00`\n"
        )

        with self.assertRaises(regenerate.RegenerateError) as raised:
            regenerate.regenerate(self.result, self.data, suites={"gpu/dense"})
        self.assertIn("20260923_120530", str(raised.exception))
        self.assertFalse((self.result / "nvidia-gpu/gpu/dense.summary.json").exists())

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
        with self.assertRaises(regenerate.RegenerateError) as raised:
            regenerate.regenerate(self.result, self.data, suites={"gpu/dense"})
        self.assertIn("20261016_235116", str(raised.exception))
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


class PartialReceiptDashboardTests(unittest.TestCase):
    """Accepted receipts add run status, non-ok cases and campaign coverage.

    These fields annotate the pages; they never change a number, the original
    report or the ``updated`` line.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = self.root / "result"
        self.benchmarks = self.root / "benchmarks" / "gpu"
        self.benchmarks.mkdir(parents=True)
        (self.benchmarks / "dense.yaml").write_text(
            "schema_version: 1\nsuite_id: gpu/dense\ndefaults:\n  regime: throughput\n"
            "problems:\n  - {id: case_ok, size_elements: 1048576}\n  - {id: case_bad, size_elements: 1048576}\n"
        )

    def test_a_publisher_receipt_has_no_source_run_to_link(self):
        # A runner publishes a page without a source run (#431): the receipt still
        # carries the digests, and the page says so instead of inventing a run.
        report = self.write_suite()
        summary = report.with_suffix(".summary.json")
        receipt = receipt_document(report, summary)
        del receipt["source"]
        report.with_suffix(".provenance.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True)
        )
        page = (self.generate() / "nvidia-gpu/gpu/dense.html").read_text()
        self.assertIn("Published result", page)
        self.assertNotIn("Source run", page)
        # The digest map is what the receipt is read for, so it is still enforced.
        summary.write_text(summary.read_text().replace("10.0", "11.0"))
        with self.assertRaisesRegex(ValueError, "differs from provenance"):
            self.generate("site-b")

    def test_a_publisher_receipt_does_not_hide_the_campaign_gap(self):
        # A runner publish is newer than the last campaign but is not one: it has
        # no missing-suite inventory, so counting it would drop the coverage gap.
        campaign = {'collected_at': '2026-10-17T00:00:00Z', 'status': 'partial',
                    'source': {'run_id': 1, 'run_attempt': 1, 'artifact_id': 1},
                    'missing_suites': ['gpu/sparse']}
        published = {'collected_at': '2026-10-18T00:00:00Z', 'files': {}}
        views = [dashboard.SuiteView('nvidia-gpu', 'gpu/dense', Path('dense.md'), None, [], campaign),
                 dashboard.SuiteView('nvidia-gpu', 'gpu/sparse', Path('sparse.md'), None, [], published)]
        self.assertEqual(dashboard.latest_campaigns(views)['nvidia-gpu'], campaign)

    def test_a_published_page_that_replaced_a_campaign_receipt_still_reports_the_gap(self):
        # result_ingestion.publish_receipt copies the campaign it overwrote into the
        # new receipt under "campaign". The coverage inventory must survive a
        # republish, or the warning disappears with the page it described.
        campaign = {'collected_at': '2026-10-17T00:00:00Z', 'status': 'partial',
                    'source': {'run_id': 1, 'run_attempt': 1, 'artifact_id': 1},
                    'missing_suites': ['gpu/sparse']}
        published = {'collected_at': '2026-10-18T00:00:00Z', 'files': {}, 'campaign': campaign}
        views = [dashboard.SuiteView('nvidia-gpu', 'gpu/dense', Path('dense.md'), None, [], published)]
        self.assertEqual(dashboard.latest_campaigns(views)['nvidia-gpu'], campaign)
        self.assertEqual(dashboard.campaign_missing_suites(views), {'nvidia-gpu': ['gpu/sparse']})

    def test_campaign_order_uses_utc_and_counts_one_run_across_suites(self):
        first = {'collected_at':'2026-10-18T09:00:00+09:00', 'status':'partial',
                 'source':{'run_id':123,'run_attempt':1,'artifact_id':456}, 'missing_suites':['gpu/sparse']}
        newer = dict(first, collected_at='2026-10-18T00:01:00Z', missing_suites=[])
        views = [dashboard.SuiteView('nvidia-gpu', 'gpu/dense', Path('dense.md'), None, [], first),
                 dashboard.SuiteView('nvidia-gpu', 'gpu/einsum', Path('einsum.md'), None, [], newer)]
        self.assertEqual(dashboard.latest_campaigns(views)['nvidia-gpu'], newer)
        self.assertEqual(dashboard.partial_run_count(views), 1)

    def write_suite(self, suite_id="gpu/dense", rows=None, receipt=None, body="body\n"):
        suite_dir = self.result / "nvidia-gpu" / suite_id.split("/", 1)[0]
        suite_dir.mkdir(parents=True, exist_ok=True)
        stem = suite_id.split("/", 1)[1]
        report = suite_dir / f"{stem}.md"
        summary = suite_dir / f"{stem}.summary.json"
        rows = rows if rows is not None else [row("case_ok", "tenferro-cuda-eager", 10.0)]
        report.write_text(f"# {suite_id}\n\n{body}")
        summary.write_text(json.dumps(summary_document(rows, suite_id=suite_id)))
        # The manifest declares the regime of every measured case: the loader has
        # no default, and a throughput case needs a working set above its family
        # floor.  The fixture cases are all large enough to stay throughput.
        ids = sorted({str(item["case"]) for item in rows})
        manifest = self.root / "benchmarks" / suite_dir.name / f"{stem}.yaml"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            "schema_version: 1\n"
            f"suite_id: {suite_id}\n"
            "defaults:\n  regime: throughput\n  run: {runs: 7}\n"
            "problems:\n"
            + "".join(
                f"  - {{id: {case}, size_elements: 1048576}}\n" for case in ids
            )
        )
        if receipt is not None:
            document = receipt_document(report, summary, suite_id=suite_id, **receipt)
            report.with_suffix(".provenance.json").write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n"
            )
        return report

    def generate(self, name="site"):
        output = self.root / name
        self.assertEqual(
            dashboard.generate(
                self.result,
                output,
                baseline_dir=self.root / "baseline",
                git_repo=None,
                instances_dir=self.root / "instances",
                benchmarks_dir=self.root / "benchmarks",
                latest_profiles=None,
            ),
            0,
        )
        return output

    def test_complete_receipt_adds_no_annotation(self):
        self.write_suite(
            receipt={"status": "complete", "case_outcomes": ZERO_OUTCOMES, "missing_suites": []}
        )
        output = self.generate()
        suite = (output / "nvidia-gpu/gpu/dense.html").read_text()
        self.assertIn("Accepted result provenance", suite)
        self.assertNotIn("Partial run", suite)
        self.assertNotIn("Non-ok cases", suite)
        self.assertNotIn("Not in the latest campaign", suite)
        for relative in ("index.html", "gpu/index.html"):
            page = (output / relative).read_text()
            self.assertNotIn("Coverage.", page)
            self.assertNotIn("non-ok cases", page)
            self.assertNotIn("partial run", page)

    def test_partial_receipt_shows_distinct_escaped_failed_and_skipped_cases(self):
        rows = [
            row("case_ok", "tenferro-cuda-eager", 10.0),
            row("case_ok", "tenferro-cuda-trace", 12.0),
            # The same case on two backends and in two groups is one case.
            row("case_<bad>", "tenferro-cuda-eager", None, status="failed", reason="bad launch"),
            row("case_<bad>", "tenferro-cuda-trace", None, status="failed", reason="bad launch"),
            row("case_beta", "tenferro-cuda-eager", None, status="failed", reason="bad launch"),
            row("case_beta", "tenferro-cuda-eager", None, status="failed", reason="bad launch", group="lat"),
            row("case_gamma", "tenferro-cuda-trace", None, status="runtime_failed", reason="crash"),
            row("case_skip", "tenferro-cuda-trace", None, status="skipped", reason="policy"),
            row("case_unsup", "tenferro-cuda-eager", None, status="unsupported", reason="policy"),
        ]
        self.write_suite(
            rows=rows,
            receipt={"status": "partial", "case_outcomes": result_status.case_outcomes(rows)},
        )
        output = self.generate()
        suite = (output / "nvidia-gpu/gpu/dense.html").read_text()
        self.assertIn("Partial run", suite)
        self.assertIn("3 cases failed", suite)
        self.assertIn("1 case skipped", suite)
        self.assertIn("1 case unsupported", suite)
        self.assertIn("case_&lt;bad&gt;", suite)
        self.assertNotIn("case_<bad>", suite)
        details = re.search(
            r'<div class="notice"><strong>Non-ok cases\.</strong>.*?</div>', suite, re.S
        )
        self.assertIsNotNone(details)
        self.assertEqual(details.group(0).count("<code>case_beta</code>"), 1)

        family = (output / "gpu/index.html").read_text()
        self.assertIn("3 cases failed · 1 case skipped · 1 case unsupported", family)
        self.assertIn("case_&lt;bad&gt;</code> <span class=\"muted\">gpu/dense</span>", family)

        index = (output / "index.html").read_text()
        card = re.search(r'<a class="family-card" href="gpu/index\.html">(.*?)</a>', index, re.S)
        self.assertIsNotNone(card)
        self.assertIn("non-ok cases: 3 cases failed · 1 case skipped · 1 case unsupported", card.group(1))
        # Compact counts only: interactive details never nest inside the card link.
        self.assertNotIn("<details", card.group(1))

    def test_family_counts_qualify_suites_so_a_shared_case_name_is_not_deduped(self):
        outcomes = {
            "failed": {"count": 1, "cases": ["shared_case"]},
            "skipped": {"count": 0, "cases": []},
            "unsupported": {"count": 0, "cases": []},
        }
        failing = [row("shared_case", "tenferro-cuda-eager", None, status="failed", reason="x")]
        for suite_id in ("gpu/dense", "gpu/sparse"):
            self.write_suite(
                suite_id=suite_id,
                rows=failing,
                receipt={"status": "partial", "case_outcomes": outcomes},
            )
        output = self.generate()
        family = (output / "gpu/index.html").read_text()
        self.assertIn("2 cases failed", family)
        self.assertIn("<code>shared_case</code> <span class=\"muted\">gpu/dense</span>", family)
        self.assertIn("<code>shared_case</code> <span class=\"muted\">gpu/sparse</span>", family)
        for suite_id in ("gpu/dense", "gpu/sparse"):
            suite = (output / f"nvidia-gpu/{suite_id}.html").read_text()
            self.assertIn("1 case failed", suite)
            self.assertNotIn("2 cases failed", suite)
        self.assertIn("non-ok cases: 2 cases failed", (output / "index.html").read_text())

    def test_workflow_failure_with_all_ok_rows_is_a_partial_run(self):
        self.write_suite(
            receipt={
                "status": "partial",
                "conclusion": "failure",
                "case_outcomes": ZERO_OUTCOMES,
                "missing_suites": [],
            }
        )
        output = self.generate()
        suite = (output / "nvidia-gpu/gpu/dense.html").read_text()
        self.assertIn("Partial run", suite)
        self.assertIn("source workflow failed", suite)
        # A failed workflow never invents a failed case.
        self.assertNotIn("Non-ok cases", suite)
        self.assertIn("1 partial run", (output / "gpu/index.html").read_text())
        index = (output / "index.html").read_text()
        self.assertIn("1 partial run", index)
        self.assertNotIn("non-ok cases", index)

    def test_suite_missing_from_latest_campaign_keeps_its_old_report(self):
        self.write_suite(
            suite_id="gpu/dense",
            receipt={
                "status": "partial",
                "case_outcomes": ZERO_OUTCOMES,
                "missing_suites": ["gpu/sparse"],
                "run_id": 2,
                "collected_at": "2026-10-18T00:00:00+00:00",
            },
        )
        self.write_suite(
            suite_id="gpu/sparse",
            body="sparse original report body\n",
            receipt={"status": "complete", "case_outcomes": ZERO_OUTCOMES, "missing_suites": []},
        )
        output = self.generate()
        sparse = (output / "nvidia-gpu/gpu/sparse.html").read_text()
        self.assertIn("Not in the latest campaign", sparse)
        self.assertIn("run 2", sparse)
        self.assertIn("sparse original report body", sparse)
        dense = (output / "nvidia-gpu/gpu/dense.html").read_text()
        self.assertNotIn("Not in the latest campaign", dense)

        family = (output / "gpu/index.html").read_text()
        self.assertIn("Coverage.", family)
        self.assertIn("nvidia-gpu: 1 suite not in the latest campaign", family)
        coverage = re.search(r'<details class="coverage">(.*?)</details>', family, re.S)
        self.assertIsNotNone(coverage)
        self.assertIn("<code>gpu/sparse</code>", coverage.group(1))
        self.assertNotIn("<code>gpu/dense</code>", coverage.group(1))

        index = (output / "index.html").read_text()
        self.assertIn("Coverage.", index)
        self.assertIn("1 suite missing from the latest campaign", index)

    def test_older_campaign_does_not_drive_coverage(self):
        self.write_suite(
            suite_id="gpu/dense",
            receipt={
                "status": "complete",
                "case_outcomes": ZERO_OUTCOMES,
                "missing_suites": [],
                "run_id": 2,
                "collected_at": "2026-10-18T00:00:00+00:00",
            },
        )
        self.write_suite(
            suite_id="gpu/sparse",
            receipt={
                "status": "partial",
                "case_outcomes": ZERO_OUTCOMES,
                "missing_suites": ["gpu/dense"],
                "run_id": 1,
                "collected_at": "2026-10-17T00:00:00+00:00",
            },
        )
        output = self.generate()
        for relative in ("index.html", "gpu/index.html", "nvidia-gpu/gpu/dense.html",
                         "nvidia-gpu/gpu/sparse.html"):
            page = (output / relative).read_text()
            self.assertNotIn("Coverage.", page)
            self.assertNotIn("Not in the latest campaign", page)

    def test_legacy_receipt_and_plain_report_derive_row_outcomes(self):
        legacy_rows = [
            row("case_ok", "tenferro-cuda-eager", 10.0),
            row("legacy_skip", "tenferro-cuda-trace", None, status="skipped", reason="policy"),
            row("legacy_fail", "tenferro-cuda-eager", None, status="compile_failed", reason="boom"),
        ]
        self.write_suite(
            suite_id="gpu/dense",
            rows=legacy_rows,
            # A historical receipt: none of the additive fields exist.
            receipt={"status": None, "case_outcomes": None, "missing_suites": None},
        )
        self.write_suite(
            suite_id="gpu/sparse",
            rows=[row("plain_fail", "tenferro-cuda-eager", None, status="failed", reason="x")],
        )
        output = self.generate()
        legacy = (output / "nvidia-gpu/gpu/dense.html").read_text()
        self.assertNotIn("Partial run", legacy)
        self.assertIn("1 case failed", legacy)
        self.assertIn("1 case skipped", legacy)
        self.assertIn("<code>legacy_fail</code>", legacy)
        self.assertIn("1 case failed", (output / "nvidia-gpu/gpu/sparse.html").read_text())
        self.assertIn("2 cases failed · 1 case skipped", (output / "gpu/index.html").read_text())

    def test_receipt_status_keeps_numbers_report_and_updated_line(self):
        rows = [
            row("case_ok", "tenferro-cuda-eager", 20.0),
            row("case_ok", "tenferro-cuda-trace", 30.0),
            row("case_ok", "pytorch-cuda", 10.0),
        ]
        self.write_suite(
            rows=rows,
            receipt={
                "status": "partial",
                "collected_at": "2026-10-20T00:00:00+00:00",
                "missing_suites": ["gpu/sparse"],
                "case_outcomes": {
                    "failed": {"count": 3, "cases": ["case_x", "case_y", "case_z"]},
                    "skipped": {"count": 0, "cases": []},
                    "unsupported": {"count": 0, "cases": []},
                },
            },
        )
        annotated = self.generate("site-a")
        (self.result / "nvidia-gpu/gpu/dense.provenance.json").unlink()
        plain = self.generate("site-b")

        def updated(output):
            return re.search(r"dates: [^<]*", (output / "index.html").read_text()).group(0)

        self.assertEqual(updated(annotated), updated(plain))
        self.assertIn("2026-10-17", updated(annotated))

        def pages(output):
            return (output / "nvidia-gpu/gpu/dense.html").read_text()

        for page in (pages(annotated), pages(plain)):
            self.assertIn('<span class="ratio ratio-much-slower">eager 2.00x</span>', page)
        self.assertEqual(
            pages(annotated).split("<h2>Original report</h2>")[1],
            pages(plain).split("<h2>Original report</h2>")[1],
        )
        self.assertIn("3 cases failed", pages(annotated))
        self.assertNotIn("Non-ok cases", pages(plain))


class DeclaredArmsTest(unittest.TestCase):
    """The dashboard draws the arms a CPU suite declares, not an assumed eager+trace pair.

    An empty eager slot reads as a lost measurement; for `cpu/linalg_jvp_vjp` and the suites
    that measure other arms it is a declaration (`docs/cpu-suite-arms.md`).
    """

    def arms(self, suite_id):
        return [
            (rule.column, tuple(arm.label for arm in rule.arms))
            for rule in dashboard.ratio_rules_for(suite_id)
        ]

    def test_trace_only_page_has_no_eager_arm(self):
        self.assertEqual(self.arms("cpu/linalg_jvp_vjp"), [("vs PyTorch", ("trace",)), ("vs JAX", ("trace",))])
        self.assertEqual(dashboard.undeclared_arm_labels("cpu/linalg_jvp_vjp"), ["eager"])

    def test_suite_that_declares_both_keeps_both(self):
        for suite_id in ("cpu/cpu_ops", "cpu/einsum", "cpu/public_api"):
            self.assertEqual(
                [arms for _, arms in self.arms(suite_id)],
                [("eager", "trace"), ("eager", "trace")],
                suite_id,
            )
            self.assertEqual(dashboard.undeclared_arm_labels(suite_id), [], suite_id)

    def test_suite_without_the_eager_trace_axis_gets_no_ratio_column(self):
        for suite_id in ("cpu/permutation", "cpu/fft", "cpu/small_work", "cpu/overhead_smoke"):
            self.assertEqual(self.arms(suite_id), [], suite_id)
            self.assertEqual(
                dashboard.undeclared_arm_labels(suite_id), ["eager", "trace"], suite_id
            )

    def test_a_gpu_suite_keeps_the_default_arms(self):
        self.assertEqual(
            [arms for _, arms in self.arms("gpu/dense")], [("eager", "trace"), ("eager", "trace")]
        )
        self.assertEqual(dashboard.undeclared_arm_labels("gpu/dense"), [])


class CollectionDatesTest(unittest.TestCase):
    """One page aggregates suites measured weeks apart; it says so per block.

    The header used to print one ``date:`` (the newest suite) for the whole page,
    which read as "everything here was measured then" while `gpu/permutation` and
    friends were three weeks older.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = self.root / "result"
        self.benchmarks = self.root / "benchmarks" / "gpu"
        self.benchmarks.mkdir(parents=True)
        for suite_id in ("gpu/dense", "gpu/einsum"):
            (self.benchmarks / f"{suite_id.split('/', 1)[1]}.yaml").write_text(
                f"schema_version: 1\nsuite_id: {suite_id}\ndefaults:\n  regime: throughput\n"
                "problems:\n  - {id: case_ok, size_elements: 1048576}\n"
            )

    def write_suite(self, suite_id, timestamp):
        suite_dir = self.result / "nvidia-gpu" / suite_id.split("/", 1)[0]
        suite_dir.mkdir(parents=True, exist_ok=True)
        stem = suite_id.split("/", 1)[1]
        report = suite_dir / f"{stem}.md"
        report.write_text(f"# {suite_id}\n\nbody\n")
        document = summary_document(
            [
                row("case_ok", "tenferro-cuda-eager", 20.0),
                row("case_ok", "tenferro-cuda-trace", 30.0),
                row("case_ok", "pytorch-cuda", 10.0),
            ],
            suite_id=suite_id,
            timestamp=timestamp,
        )
        report.with_suffix(".summary.json").write_text(json.dumps(document))

    def generate(self, name="site"):
        output = self.root / name
        self.assertEqual(
            dashboard.generate(
                self.result,
                output,
                baseline_dir=self.root / "baseline",
                git_repo=None,
                instances_dir=self.root / "instances",
                benchmarks_dir=self.root / "benchmarks",
                latest_profiles=None,
            ),
            0,
        )
        return output

    def test_header_shows_the_range_and_the_old_suite_is_badged_stale(self):
        self.write_suite("gpu/dense", "2026-10-17T10:53:37+00:00")
        self.write_suite("gpu/einsum", "2026-11-05T03:43:18Z")
        page = (self.generate() / "nvidia-gpu/gpu/dense.html").read_text()
        family = (self.root / "site/gpu/index.html").read_text()

        # The header is a range, not the newest suite standing for all of them.
        self.assertIn("dates: 2026-10-17 – 2026-11-05 · 2 suites", family)
        self.assertIn("1 suite older than 14 days (ages from 2026-11-05)", family)
        self.assertNotIn("<span>date: ", family)
        self.assertIn(
            "This page mixes data collected on different dates; each block carries its own date.",
            family,
        )
        self.assertIn("1 of 2 suites were collected more than 14 days before the newest", family)

        # Each block carries its own date and age; only the old one is stale.
        self.assertIn('collected 2026-10-17 · 19 days old</span> <span class="stale"', family)
        self.assertIn('<span class="collected">collected 2026-11-05</span>', family)
        self.assertEqual(family.count('<span class="stale" title='), 1)

        # The suite page still prints its own date, and the numbers are untouched.
        self.assertIn("2026-10-17", page)
        self.assertIn('<span class="ratio ratio-much-slower">eager 2.00x</span>', page)

    def test_two_generations_of_the_same_tree_are_byte_identical(self):
        self.write_suite("gpu/dense", "2026-10-17T10:53:37+00:00")
        self.write_suite("gpu/einsum", "2026-11-05T03:43:18Z")
        first = self.generate("site-a")
        second = self.generate("site-b")
        for relative in ("index.html", "gpu/index.html", "gpu/latency/index.html"):
            self.assertEqual(
                (first / relative).read_bytes(),
                (second / relative).read_bytes(),
                relative,
            )


class FailedBackendTest(unittest.TestCase):
    """A failed backend never removes the case, and never hides the arms that ran."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = self.root / "result"
        self.benchmarks = self.root / "benchmarks" / "gpu"
        self.benchmarks.mkdir(parents=True)
        (self.benchmarks / "dense.yaml").write_text(
            "schema_version: 1\nsuite_id: gpu/dense\ndefaults:\n  regime: throughput\n"
            "problems:\n  - {id: case_ok, size_elements: 1048576}\n"
            # A GPU suite is policed: every measured case is declared, so this
            # fixture declares the case it measures (#383).
            "  - {id: case_mixed, size_elements: 1048576}\n"
        )
        suite_dir = self.result / "nvidia-gpu" / "gpu"
        suite_dir.mkdir(parents=True)
        report = suite_dir / "dense.md"
        report.write_text("# gpu/dense\n\nbody\n")
        document = summary_document(
            [
                row("case_mixed", "tenferro-cuda-eager", 20.0),
                row("case_mixed", "tenferro-cuda-trace", 0.0, status="failed",
                    reason="CUDA error: misaligned address"),
                row("case_mixed", "pytorch-cuda", 10.0),
            ],
            suite_id="gpu/dense",
            timestamp="2026-10-17T00:00:00+00:00",
        )
        report.with_suffix(".summary.json").write_text(json.dumps(document))
        output = self.root / "site"
        dashboard.generate(
            self.result,
            output,
            baseline_dir=self.root / "baseline",
            git_repo=None,
            instances_dir=self.root / "instances",
            benchmarks_dir=self.root / "benchmarks",
            latest_profiles=None,
        )
        self.suite_page = (output / "nvidia-gpu/gpu/dense.html").read_text()
        self.family_page = (output / "gpu/index.html").read_text()

    def test_the_case_keeps_its_row_and_the_successful_arm_keeps_its_number(self):
        self.assertIn("<th>not ok</th>", self.suite_page)
        self.assertIn("<code>case_mixed</code>", self.suite_page)
        # The arm that ran is still a number, in the table and in the chart.
        self.assertIn('<span class="ratio ratio-much-slower">eager 2.00x</span>', self.suite_page)
        self.assertIn('class="bar-tenferro"', self.family_page)
        # The arm that failed is named, with its reason, in the same row.
        self.assertIn(
            '<span class="fail">fail</span> <code>tenferro-cuda-trace</code> '
            '<span class="muted">CUDA error: misaligned address</span>',
            self.suite_page,
        )
        # And it is a fail line in the chart, not a silently missing bar.
        self.assertIn(
            'class="chart-fail">fail · tenferro-cuda-trace · CUDA error: misaligned address',
            self.family_page,
        )


def chart_no_ratio_note(html):
    """The paragraph ``svg_ratio_chart`` prints when no case has a ratio."""
    start = html.index("No ratio could be computed")
    return html[start : html.index("</p>", start)]


# SVG text does not wrap and a chart's viewBox is a fixed 1060 units wide, so a row's
# text has to fit between 300 (the row's label column) and 1060.  `text_units` prices a
# line from Chrome's measurement of this font at 11px ('W' 0.97 em, '%' 0.93, 'm' 0.88,
# other capitals 0.66-0.78, digits 0.47-0.65, lowercase 0.25-0.62): the narrow set is
# 0.62 em and everything else a full em, so a wide glyph cannot be undercounted.  The
# 0.60 em/char this file used before was not conservative: it priced each of 95 'W's at
# 6.6 units where Chrome renders them 10.7 units wide.
CHART_TEXT_START = dashboard.BAR_LABEL_WIDTH
CHART_TEXT_END = dashboard.RATIO_CHART_WIDTH


def chart_layout(chart):
    """The chart's height and, per case label, the texts drawn under it.

    Each row is ``(name, label_y, [(x, y, class, text)])`` in document order, so a
    test can hold a row's lines to the row the renderer gave them.
    """
    height = float(re.search(r'viewBox="0 0 \d+ (\d+)"', chart).group(1))
    blocks = re.split(r'<text x="0" y="(\d+)" class="chart-label">([^<]*)</text>', chart)
    rows = []
    for index in range(1, len(blocks), 3):
        texts = [
            (float(x), float(y), css, text)
            for x, y, css, text in re.findall(
                r'<text x="([\d.]+)" y="(\d+)" class="(chart-[a-z]+)">([^<]*)</text>',
                blocks[index + 2],
            )
        ]
        rows.append((blocks[index + 1], float(blocks[index]), texts))
    return height, rows


def chart_svg(html):
    """The rendered ratio chart of a page (a chart with ratios, so an SVG)."""
    return re.findall(r'<svg class="bars".*?</svg>', html, re.S)[0]


def profile_header(html, index=0):
    """One ``profile-head`` div of a page (the spans are inline, so it is one line)."""
    return re.findall(r'<div class="profile-head">.*?</div>', html, re.S)[index]


class RatioChartWithoutBaselineTest(unittest.TestCase):
    """A case whose baseline arm failed keeps the number its other arm measured.

    ``svg_ratio_chart`` printed "no measurement was recorded as ok" for a case whose
    *baseline* failed, even though the contender had a median, and the paragraph had
    no place for the failure reason either.  "Measured but not comparable" and
    "nothing was measured" are different claims and must not share a sentence.
    """

    def cases(self, measured, not_ok):
        return [
            {
                "case": "dense_solve_f64_512_rhs16",
                "contenders": [],
                "measured": measured,
                "not_ok": not_ok,
            }
        ]

    def test_a_measured_case_with_a_failed_baseline_is_not_called_unmeasured(self):
        html = dashboard.svg_ratio_chart(
            self.cases(
                [{"backend": "tenferro-cuda-eager", "median": 20.0}],
                [{"backend": "pytorch-cuda", "status": "failed", "label": "fail", "reason": "CUDA OOM"}],
            ),
            "PyTorch",
        )
        note = chart_no_ratio_note(html)
        # The measurement that exists is printed, with its unit.
        self.assertIn("<code>tenferro-cuda-eager</code>", note)
        self.assertIn("20.000 ms", note)
        # The failure reason survives even though no bar could be drawn.
        self.assertIn("CUDA OOM", note)
        # And the case is not described as having recorded nothing.
        self.assertNotIn("no measurement was recorded as ok", note)

    def test_a_case_without_any_ok_row_still_says_that(self):
        html = dashboard.svg_ratio_chart(
            self.cases(
                [],
                [{"backend": "pytorch-cuda", "status": "failed", "label": "fail", "reason": "CUDA OOM"}],
            ),
            "PyTorch",
        )
        note = chart_no_ratio_note(html)
        self.assertIn("no measurement was recorded as ok", note)
        self.assertIn("CUDA OOM", note)

    def test_the_page_keeps_the_measured_median_and_the_failure_reason(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        result = root / "result"
        benchmarks = root / "benchmarks" / "gpu"
        benchmarks.mkdir(parents=True)
        (benchmarks / "dense.yaml").write_text(
            "schema_version: 1\nsuite_id: gpu/dense\ndefaults:\n  regime: throughput\n"
            "problems:\n  - {id: case_ok, size_elements: 1048576}\n"
        )
        suite_dir = result / "nvidia-gpu" / "gpu"
        suite_dir.mkdir(parents=True)
        report = suite_dir / "dense.md"
        report.write_text("# gpu/dense\n\nbody\n")
        document = summary_document(
            [
                row("case_ok", "tenferro-cuda-eager", 20.0),
                row("case_ok", "pytorch-cuda", 0.0, status="failed", reason="CUDA OOM"),
            ],
            suite_id="gpu/dense",
        )
        report.with_suffix(".summary.json").write_text(json.dumps(document))
        output = root / "site"
        self.assertEqual(
            dashboard.generate(
                result,
                output,
                baseline_dir=root / "baseline",
                git_repo=None,
                instances_dir=root / "instances",
                benchmarks_dir=root / "benchmarks",
                latest_profiles=None,
            ),
            0,
        )
        family = (output / "gpu/index.html").read_text()
        note = chart_no_ratio_note(family)
        self.assertIn("20.000 ms", note)
        self.assertIn("CUDA OOM", note)
        self.assertNotIn("no measurement was recorded as ok", note)


class ChartNotePerCaseTest(unittest.TestCase):
    """The no-ratio note belongs to a case, not to a suite.

    ``svg_ratio_chart`` printed that note only when *no* case of the suite had a
    ratio.  One comparable neighbour was enough to switch the chart to bars, and
    then a case whose own arm measured 20 ms printed nothing but
    ``fail · pytorch-cuda · CUDA OOM``: the number the run recorded disappeared
    because a different case could be compared.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = self.root / "result"
        self.benchmarks = self.root / "benchmarks" / "gpu"
        self.instances = self.root / "instances"
        self.instances.mkdir(parents=True)

    def write_suite(self, suite_id, regime, problems, rows, profile="nvidia-gpu"):
        self.benchmarks.mkdir(parents=True, exist_ok=True)
        stem = suite_id.split("/", 1)[1]
        (self.benchmarks / f"{stem}.yaml").write_text(
            f"schema_version: 1\nsuite_id: {suite_id}\ndefaults:\n  regime: {regime}\n"
            "problems:\n"
            + "".join(
                f"  - {{id: {problem}{', size_elements: 1048576' if regime == 'throughput' else ''}}}\n"
                for problem in problems
            )
        )
        suite_dir = self.result / profile / suite_id.split("/", 1)[0]
        suite_dir.mkdir(parents=True)
        report = suite_dir / f"{stem}.md"
        report.write_text(f"# {suite_id}\n\nbody\n")
        report.with_suffix(".summary.json").write_text(
            json.dumps(summary_document(rows, suite_id=suite_id, profile=profile))
        )

    def generate(self):
        output = self.root / "site"
        self.assertEqual(
            dashboard.generate(
                self.result,
                output,
                baseline_dir=self.root / "baseline",
                git_repo=None,
                instances_dir=self.instances,
                benchmarks_dir=self.root / "benchmarks",
                latest_profiles=None,
            ),
            0,
        )
        return output

    def test_a_case_without_a_ratio_keeps_its_median_beside_a_comparable_case(self):
        self.write_suite(
            "gpu/dense",
            "throughput",
            ("case_ok", "case_comparable"),
            [
                row("case_ok", "tenferro-cuda-eager", 20.0),
                row("case_ok", "pytorch-cuda", 0.0, status="failed", reason="CUDA OOM"),
                row("case_comparable", "tenferro-cuda-eager", 30.0),
                row("case_comparable", "pytorch-cuda", 10.0),
            ],
        )
        chart = chart_svg((self.generate() / "gpu/index.html").read_text())
        # The comparable case keeps the comparison it earned: exactly one bar, filled
        # as a tenferro bar, carrying both medians, and the ratio it is drawn at.
        bars = re.findall(r'<rect\b[^>]*class="bar-[^"]*"[^>]*>', chart)
        self.assertEqual(len(bars), 1, chart)
        self.assertIn('class="bar-tenferro"', bars[0])
        self.assertGreater(float(re.search(r'width="([\d.]+)"', bars[0]).group(1)), 0.0)
        self.assertIn("<title>eager: median 30.000 ms vs PyTorch median 10.000 ms</title>", chart)
        self.assertIn('class="chart-value">eager 3.00x · 30 / 10 ms</text>', chart)
        self.assertIn('class="chart-label">case_comparable', chart)
        # ...and the neighbour beside it still says what it measured, with no bar:
        # nothing was measured on the other side of it.
        self.assertIn('class="chart-label">case_ok', chart)
        self.assertIn("20.000 ms", chart)
        self.assertIn("CUDA OOM", chart)
        # Its own line, below its own fail line and above the next case.
        fail_y = int(
            re.search(r'<text x="300" y="(\d+)" class="chart-fail">', chart).group(1)
        )
        measured_y = int(
            re.search(r'<text x="300" y="(\d+)" class="chart-value">measured ', chart).group(1)
        )
        next_case_y = int(
            re.search(r'<text x="0" y="(\d+)" class="chart-label">case_comparable', chart).group(1)
        )
        height = int(re.search(r'viewBox="0 0 1060 (\d+)"', chart).group(1))
        self.assertLess(fail_y, measured_y)
        # The row is tall enough for its extra line: it stays inside the chart.
        self.assertLessEqual(measured_y + 9, height)
        # And the case above it keeps its own single line.
        self.assertLess(next_case_y, fail_y)

    def test_a_note_with_several_medians_stays_inside_the_chart(self):
        # cpu/public_api declares the eager and trace arms while this row set is measured
        # on five other backends: no arm matches, so the case has no bar and a note that
        # has to name every median.  1e150 is the number bound: `%.1f` would print 152
        # integer digits into a `<text>` that cannot wrap.  95 'W's is the free-text
        # bound: a backend name is not bounded by anything, and one character of it
        # measures 0.97 em, so this line is wider than the row's whole column.
        self.write_suite(
            "cpu/public_api",
            "throughput",
            ("add_into_f64_32x32", "case_long_name", "case_comparable"),
            [
                row("add_into_f64_32x32", "julia-base", 21.347),
                row("add_into_f64_32x32", "pytorch-cpu", 21.431),
                row("add_into_f64_32x32", "strided-jl", 67.0),
                row("add_into_f64_32x32", "tenferro-direct", 21.511),
                row("add_into_f64_32x32", "tenferro-rs", 1e150),
                row("case_long_name", "W" * 95, 20.0),
                row("case_long_name", "pytorch-cuda", 0.0, status="failed", reason="CUDA OOM"),
                # A neighbour with a ratio, so this suite draws a chart (not a paragraph).
                row("case_comparable", "tenferro-eager", 30.0),
                row("case_comparable", "pytorch-cpu", 10.0),
            ],
            profile="mac-cpu",
        )
        chart = chart_svg((self.generate() / "cpu/index.html").read_text())
        height, rows = chart_layout(chart)
        # A row's text cannot wrap on its own, so every line drawn in the label column
        # has to fit between it and the chart's right edge.
        for name, _, texts in rows:
            for x, _, _, text in texts:
                self.assertGreaterEqual(x, CHART_TEXT_START)
                self.assertLessEqual(x + dashboard.text_units(text), CHART_TEXT_END, text)
        # The 95-character name is wider than the column on its own, so the note took
        # more lines instead of running off the chart, and nothing was dropped: the
        # name it was drawn for and the median it recorded are both still there.
        long_row = next(texts for name, _, texts in rows if name == "case_long_name")
        note = [text for _, _, css, text in long_row if css == "chart-value"]
        self.assertGreater(len(note), 2, note)
        self.assertIn("W" * 95, "".join(note).replace(" ", ""))
        self.assertTrue(any("20.000 ms" in text for text in note), note)
        # And the note names each arm with the median it was drawn from, one per line,
        # then why there is no ratio (anything with `x · ` is a bar's ratio label).
        multi_arm = [
            text
            for name, _, texts in rows if name == "add_into_f64_32x32"
            for _, _, css, text in texts if css == "chart-value"
        ]
        self.assertEqual(
            multi_arm,
            [
                "measured julia-base 21.347 ms",
                "pytorch-cpu 21.431 ms",
                "strided-jl 67.000 ms",
                "tenferro-direct 21.511 ms",
                "tenferro-rs 1e+150 ms",
                "but no arm of it had a ratio against that baseline",
            ],
        )
        # Vertical containment: a row's lines have to fit the height the chart reserved
        # for them and the label the renderer put them under.  `note_line_count` used to
        # be an independent guess at how many lines the note would draw, so a wrong guess
        # clipped the extra lines off the bottom (or let them run into the next row) and
        # the element-count assertions above still passed.
        self.assertEqual(len(rows), 3)
        for index, (name, label_y, texts) in enumerate(rows):
            self.assertTrue(texts, name)
            for x, text_y, _, text in texts:
                self.assertGreaterEqual(text_y, label_y, text)
                self.assertLessEqual(text_y + 4, height, text)
                if index + 1 < len(rows):
                    self.assertLess(text_y, rows[index + 1][1], text)

    def test_a_case_whose_declared_references_both_failed_keeps_its_row(self):
        (self.instances / "gpu_perf_issues.json").write_text(
            json.dumps(
                [
                    {
                        "id": "transfer_up_f64_cudarc-pageable_8B",
                        "reference_for": "transfer_up_f64_tenferro_8B",
                    },
                    {
                        "id": "transfer_up_f64_cudarc-pinned_8B",
                        "reference_for": "transfer_up_f64_tenferro_8B",
                    },
                ]
            )
        )
        self.write_suite(
            "gpu/perf_issues",
            "latency",
            (
                "transfer_up_f64_tenferro_8B",
                "transfer_up_f64_cudarc-pageable_8B",
                "transfer_up_f64_cudarc-pinned_8B",
            ),
            [
                row("transfer_up_f64_tenferro_8B", "tenferro-cuda-eager", 20.0),
                row(
                    "transfer_up_f64_cudarc-pageable_8B",
                    "cudarc",
                    0.0,
                    status="failed",
                    reason="probe crashed",
                ),
                row(
                    "transfer_up_f64_cudarc-pinned_8B",
                    "cudarc",
                    0.0,
                    status="failed",
                    reason="probe crashed",
                ),
            ],
        )
        note = chart_no_ratio_note((self.generate() / "gpu/latency/index.html").read_text())
        # The case whose own arm ran keeps the median it recorded, next to the
        # reference failures that are the reason it has no ratio.
        self.assertIn("20.000 ms", note)
        self.assertIn(
            "<code>transfer_up_f64_tenferro_8B</code> measured "
            "<code>tenferro-cuda-eager</code> 20.000 ms, but its baseline arm produced no number",
            note,
        )
        # The failures are still named, with their reason, beside it.
        for reference in ("cudarc-pageable", "cudarc-pinned"):
            self.assertIn(f"<code>transfer_up_f64_{reference}_8B</code>", note)
        self.assertIn('<span class="muted">probe crashed</span>', note)


# Every chart text, priced without a browser.  A chart is an SVG with a fixed viewBox
# (1060 units for the ratio chart, 880 for the case chart) and SVG text does not wrap,
# so a text whose right edge passes the viewBox is cut off.  The width bound is
#
#     width(text, px) = px * sum(0.62 if char in TEXT_NARROW else 1.0 for char in text)
#
# with px = 10 for .chart-note and 11 for every other chart class.  Chrome measures
# this face at 'W' 0.97 em, '%' 0.93, 'm' 0.88, other capitals 0.66-0.78, digits
# 0.47-0.65 and lowercase 0.25-0.62 (0.47 em per character on average over the
# committed result), so pricing the narrow set at its widest glyph and everything
# else at a full em cannot undercount a line.  The formula is written out here, not
# imported, so a change to the generator's own estimate cannot loosen this check.
TEXT_NARROW = set("abcdefghijklnopqrstuvxyz !\"'()*,-./:;?[]_`{|}")
TEXT_PX = {"chart-note": 10}


def text_width_bound(text, px):
    return px * sum(0.62 if char in TEXT_NARROW else 1.0 for char in text)


def chart_texts(page_html):
    """Every ``<text>`` of every chart as ``(viewbox_width, x, class, text)``.

    The text is what the reader sees: a ``<tspan>`` is part of its line and a
    ``<title>`` (hover only) is not.
    """
    texts = []
    for width, body in re.findall(
        r'<svg class="bars" viewBox="0 0 (\d+) \d+"(.*?)</svg>', page_html, re.S
    ):
        for attrs, inner in re.findall(r"<text ([^>]*)>(.*?)</text>", body, re.S):
            x = float(re.search(r'\bx="(-?[\d.]+)"', attrs).group(1))
            css = re.search(r'\bclass="([\w-]+)"', attrs).group(1)
            inner = re.sub(r"<title>.*?</title>", "", inner, flags=re.S)
            texts.append((float(width), x, css, unescape(re.sub(r"<[^>]+>", "", inner))))
    return texts


def overflowing_texts(page_html):
    """The chart texts whose bound passes the left or right edge of their viewBox."""
    return [
        (width, x, css, text)
        for width, x, css, text in chart_texts(page_html)
        if x < 0 or x + text_width_bound(text, TEXT_PX.get(css, 11)) > width
    ]


class ChartTextInsideViewBoxTest(unittest.TestCase):
    """No chart text is drawn past its viewBox, and none is shortened to get there.

    Issue #436: on the committed result, 354 chart texts ran past the right edge in
    Chrome and were cut off: a ``fail`` line whose recorded reason ended
    mid-word, a ratio label pushed out of the chart by a wide whisker, and the
    ratio chart's one-line legend.  A cut line is a claim the reader cannot finish,
    so a long line takes more lines instead, and a label that does not fit to the
    right of its whisker sits on the whisker instead of past the edge.
    """

    REASON = (
        "CUDA implementation checked numerically only: API creates and uploads scalar "
        "constants through context.upload, so the timed path is not the one measured; "
        "host-placement bug fixed by upstream 917d5341"
    )

    def test_case_chart_keeps_a_long_reason_and_a_wide_whisker_inside(self):
        rows = [
            row("c", "pytorch-cuda", 10.0, iqr=0.2),
            # IQR five times the median: the CI runs far past the axis.
            row("c", "tenferro-cuda-eager", 1.0, iqr=5.0),
            row("c", "tenferro-cuda-trace", None, status="unsupported", reason=self.REASON),
        ]
        svg = dashboard.svg_case_chart(rows, "ms", runs=7, baseline=rows[0], baseline_label="PyTorch")
        self.assertEqual(overflowing_texts(svg), [])
        texts = chart_texts(svg)
        # Nothing was dropped to fit: the whole reason is still on the page, in order...
        fail = " ".join(text for _, _, css, text in texts if css == "chart-fail")
        self.assertIn(self.REASON, fail)
        # ...and the label keeps every character of the number it was drawn from.
        self.assertIn("1.000 ms · 0.10x", [text for _, _, css, text in texts if css == "chart-value"])

    def test_ratio_chart_keeps_its_legend_and_a_pushed_label_inside(self):
        cases = [
            {
                "case": "c",
                "baseline_median": 112034.0,
                "contenders": [
                    {
                        "label": "tenferro-cuda",
                        "kind": "tenferro",
                        "ratio": 3.87,
                        "median": 434054.0,
                        "interval": (0.5, 40.0),
                        "noisy": True,
                    },
                    {"label": "JAX", "kind": "reference", "ratio": 0.9, "median": 100000.0},
                ],
            }
        ]
        svg = dashboard.svg_ratio_chart(cases, "declared reference", "ns/op")
        self.assertEqual(overflowing_texts(svg), [])
        texts = chart_texts(svg)
        values = [text for _, _, css, text in texts if css == "chart-value"]
        self.assertIn("tenferro-cuda ~3.87x · 434000 ns/op / 112000 ns/op noisy", values)
        # The legend is broken between its clauses, never inside one.
        legend = [text for _, _, css, text in texts if css == "chart-note"]
        self.assertGreater(len(legend), 1, legend)
        for clause in (
            "label = contender / declared reference, ns/op",
            "dashed line = baseline (1.00x)",
            "fail = that arm produced no number, so it has no bar",
        ):
            self.assertTrue(any(clause in line for line in legend), (clause, legend))

    def test_the_committed_result_draws_no_text_past_a_chart_edge(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        output = Path(temp.name) / "site"
        # Every profile, not only the deployed set: issue #436 counted 1001 texts there.
        self.assertEqual(
            dashboard.generate(
                ROOT / "result",
                output,
                baseline_dir=Path(temp.name) / "baseline",
                git_repo=None,
                latest_profiles=None,
            ),
            0,
        )
        pages = sorted(output.rglob("*.html"))
        charted = [path for path in pages if '<svg class="bars"' in path.read_text()]
        self.assertGreater(len(charted), 0)
        overflow = {
            str(path.relative_to(output)): found
            for path in charted
            if (found := overflowing_texts(path.read_text()))
        }
        total = sum(len(found) for found in overflow.values())
        sample = [item for found in overflow.values() for item in found][:5]
        self.assertEqual(total, 0, f"{total} chart texts on {len(overflow)} pages, e.g. {sample}")


class ProfileHeaderScopeTest(unittest.TestCase):
    """A page's header dates only the suites that page renders.

    ``gpu/latency/index.html`` printed the throughput suites' range
    (``dates: 2026-10-17 – 2026-11-05 · 2 suites``) although it draws no throughput
    block: with nothing of its own to describe, the header fell back to every suite
    of the profile, diagnostics included.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = self.root / "result"
        self.benchmarks = self.root / "benchmarks" / "gpu"
        self.benchmarks.mkdir(parents=True)
        for suite_id, regime, problem in (
            ("gpu/dense", "throughput", "case_throughput"),
            ("gpu/linalg_ad_latency", "latency", "case_latency"),
        ):
            (self.benchmarks / f"{suite_id.split('/', 1)[1]}.yaml").write_text(
                f"schema_version: 1\nsuite_id: {suite_id}\ndefaults:\n  regime: {regime}\n"
                # A throughput case declares the working set its floor is applied to.
                f"problems:\n  - {{id: {problem}{', size_elements: 1048576' if regime == 'throughput' else ''}}}\n"
            )

    def write_suite(self, suite_id, timestamp, case):
        suite_dir = self.result / "nvidia-gpu" / suite_id.split("/", 1)[0]
        suite_dir.mkdir(parents=True, exist_ok=True)
        stem = suite_id.split("/", 1)[1]
        report = suite_dir / f"{stem}.md"
        report.write_text(f"# {suite_id}\n\nbody\n")
        document = summary_document(
            [
                row(case, "tenferro-cuda-eager", 20.0),
                row(case, "tenferro-cuda-trace", 30.0),
                row(case, "pytorch-cuda", 10.0),
            ],
            suite_id=suite_id,
            timestamp=timestamp,
        )
        report.with_suffix(".summary.json").write_text(json.dumps(document))

    def generate(self, name="site"):
        output = self.root / name
        self.assertEqual(
            dashboard.generate(
                self.result,
                output,
                baseline_dir=self.root / "baseline",
                git_repo=None,
                instances_dir=self.root / "instances",
                benchmarks_dir=self.root / "benchmarks",
                latest_profiles=None,
            ),
            0,
        )
        return output

    def test_a_regime_page_dates_only_the_suites_it_renders(self):
        self.write_suite("gpu/dense", "2026-11-05T03:43:18Z", "case_throughput")
        self.write_suite("gpu/linalg_ad_latency", "2026-10-17T10:53:37+00:00", "case_latency")
        output = self.generate()

        latency = profile_header((output / "gpu/latency/index.html").read_text())
        self.assertIn("dates: 2026-10-17 · 1 suite", latency)
        # The throughput suite is not drawn here, so it cannot date this page.
        self.assertNotIn("2026-11-05", latency)
        self.assertNotIn("2 suites", latency)
        self.assertNotIn("older than", latency)

        throughput = profile_header((output / "gpu/index.html").read_text())
        self.assertIn("dates: 2026-11-05 · 1 suite", throughput)
        self.assertNotIn("2026-10-17", throughput)

    def test_a_profile_with_nothing_rendered_has_no_date_line(self):
        self.write_suite("gpu/dense", "2026-11-05T03:43:18Z", "case_throughput")
        page = (self.generate() / "gpu/latency/index.html").read_text()
        latency = profile_header(page)
        self.assertNotIn("dates:", latency)
        self.assertIn("no suite from this profile is rendered on this page", latency)
        # A summary exists for this profile; it is just not this page's subject.
        self.assertNotIn("no machine-readable summary", latency)


    def test_a_profile_with_no_summary_at_all_still_says_so(self):
        # A report with no sidecar, and no maintained suite in either regime.
        legacy = self.result / "nvidia-gpu" / "gpu" / "legacy.md"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("# gpu/legacy\n\nbody\n")
        page = (self.generate() / "gpu/latency/index.html").read_text()
        latency = profile_header(page)
        self.assertNotIn("dates:", latency)
        self.assertIn("no machine-readable summary", latency)
        self.assertNotIn("no suite from this profile is rendered on this page", latency)


class NonOkStatusTest(unittest.TestCase):
    """A non-ok row that carries no ``reason`` still says why it has no number.

    The rows are the issue's own artefact, copied verbatim from
    ``result/nvidia-gpu/gpu/tensornetwork.summary.json`` at commit ``a0792c4``
    (run 41237278524, tenferro ``c15dfcf0ea52``).  The trace arm recorded
    ``status: runtime_failed``, ``median: null`` and no ``reason`` at all, so the
    status name is the only explanation in the record: a page that prints ``fail``
    and stops throws it away, and a reader cannot tell a crash from a wrong
    number.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = self.root / "result"
        benchmark = self.root / "benchmarks" / "gpu" / "tensornetwork.yaml"
        benchmark.parent.mkdir(parents=True)
        benchmark.write_text(
            "schema_version: 1\nsuite_id: gpu/tensornetwork\n"
            "defaults:\n  regime: throughput\n"
            "problems:\n  - {id: tensornetwork_permutation_optimized_f32, size_elements: 1048576}\n"
        )
        suite_dir = self.result / "nvidia-gpu" / "gpu"
        suite_dir.mkdir(parents=True)
        report = suite_dir / "tensornetwork.md"
        report.write_text("# gpu/tensornetwork\n\nbody\n")
        document = summary_document(
            [
                {"case": "tensornetwork_permutation_optimized_f32",
                 "group": "tensor_network_contract", "backend": "pytorch-cuda",
                 "median": 107.9337315, "iqr": 0.5981489999999923,
                 "status": "ok", "verified": True},
                {"case": "tensornetwork_permutation_optimized_f32",
                 "group": "tensor_network_contract", "backend": "tenferro-cuda-eager",
                 "median": 305.9101845, "iqr": 64.00893299999996,
                 "status": "ok", "verified": True},
                {"case": "tensornetwork_permutation_optimized_f32",
                 "group": "tensor_network_contract", "backend": "tenferro-cuda-trace",
                 "median": None, "status": "runtime_failed", "verified": False},
            ],
            suite_id="gpu/tensornetwork",
            timestamp="2026-10-31T20:18:08.642468+00:00",
        )
        report.with_suffix(".summary.json").write_text(json.dumps(document))
        output = self.root / "site"
        dashboard.generate(
            self.result,
            output,
            baseline_dir=self.root / "baseline",
            git_repo=None,
            instances_dir=self.root / "instances",
            benchmarks_dir=self.root / "benchmarks",
            latest_profiles=None,
        )
        self.suite_page = (output / "nvidia-gpu/gpu/tensornetwork.html").read_text()
        self.family_page = (output / "gpu/index.html").read_text()

    def test_the_failed_arm_names_its_recorded_status_and_the_other_arms_keep_their_bars(self):
        # The arm that ran keeps its number: 305.9101845 / 107.9337315 = 2.83x.
        self.assertIn('<span class="ratio ratio-much-slower">eager 2.83x</span>', self.suite_page)
        self.assertIn('class="bar-tenferro"', self.family_page)
        # The failed arm is a `fail` row that still says why: runtime_failed.
        self.assertIn(
            '<span class="fail">fail</span> <code>tenferro-cuda-trace</code> '
            '<span class="muted">runtime_failed</span>',
            self.suite_page,
        )
        # And the chart carries the same words under the case, not a bare `fail`.
        self.assertIn(
            'class="chart-fail">fail · tenferro-cuda-trace · runtime_failed',
            self.family_page,
        )


if __name__ == "__main__":
    unittest.main()
