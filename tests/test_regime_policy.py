"""Tests for the regime policy: no implicit regime, and floors for throughput.

The dashboard used to default an undeclared case to `throughput`, so a suite
that declared nothing published its per-call overhead probes on the throughput
page.  These tests pin the three behaviours that replace that fallback: a
missing declaration fails closed, a `throughput` case below its family floor
fails closed, and the page banner/report is generated from the cases it actually
holds instead of a fixed sentence.
"""

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
import regime_policy  # noqa: E402


def suite_yaml(suite_id: str, defaults: str, problems: str) -> str:
    return (
        "schema_version: 1\n"
        f"suite_id: {suite_id}\n"
        "title: fixture\n"
        "description: fixture\n"
        "defaults:\n"
        f"{defaults}"
        "backends:\n  - tenferro-cuda-trace\n"
        "problems:\n"
        f"{problems}"
    )


def summary(profile: str, suite_id: str, cases: list[tuple[str, float]]) -> dict:
    return {
        "schema_version": 2,
        "target_profile": profile,
        "suite_id": suite_id,
        "timestamp": "2026-11-08T00:00:00+00:00",
        "tenferro_commit": "abc123",
        "hardware": {"gpu": "NVIDIA A100 80GB PCIe"},
        "unit": "ms",
        "rows": [
            {
                "case": case,
                "group": case.split("_")[0],
                "backend": backend,
                "median": median,
                "iqr": median / 10,
                "status": "ok",
                "verified": True,
            }
            for case, median in cases
            for backend in ("tenferro-cuda-trace", "pytorch-cuda")
        ],
    }


class RegimePolicyUnitTests(unittest.TestCase):
    def test_undeclared_regime_is_rejected_for_a_gpu_case(self):
        document = {
            "suite_id": "gpu/dense",
            "defaults": {"run": {"runs": 7}},
            "problems": [{"id": "case_a", "size_elements": 1_048_576}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            errors = regime_policy.check_suite(document, Path(tmp))
        self.assertEqual(len(errors), 1)
        self.assertIn("declares no regime", errors[0])
        self.assertIn("no implicit regime", errors[0])

    def test_throughput_below_the_family_floor_is_rejected(self):
        document = {
            "suite_id": "gpu/dense",
            "defaults": {"regime": "throughput"},
            "problems": [{"id": "tiny", "family": "elementwise", "elementwise_chain": {"n": 1024}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            errors = regime_policy.check_suite(document, Path(tmp))
        self.assertEqual(len(errors), 1)
        self.assertIn("below the elementwise floor", errors[0])
        self.assertIn("1,024 elements", errors[0])
        self.assertIn("declare", errors[0])

    def test_latency_case_needs_no_size(self):
        document = {
            "suite_id": "gpu/dense",
            "defaults": {"regime": "latency"},
            "problems": [{"id": "tiny"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(regime_policy.check_suite(document, Path(tmp)), [])

    def test_cpu_suites_are_not_policed(self):
        # The CPU catalogues still mix size classes; their floors are a separate
        # change, so a below-floor CPU case must not fail the load.
        document = {
            "suite_id": "cpu/cpu_ops",
            "defaults": {"regime": "throughput"},
            "problems": [{"id": "tiny", "family": "elementwise", "elementwise_chain": {"n": 16}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(regime_policy.check_suite(document, Path(tmp)), [])

    def test_unknown_measure_is_rejected(self):
        document = {
            "suite_id": "gpu/dense",
            "defaults": {"regime": "latency", "measure": "sometimes"},
            "problems": [{"id": "case_a"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            errors = regime_policy.check_suite(document, Path(tmp))
        self.assertTrue(any("unknown measure" in error for error in errors), errors)

    def test_working_set_is_derived_from_each_shape_block(self):
        cases = [
            ({"matmul": {"m": 3072, "n": 3072, "k": 3072}}, 9_437_184),
            ({"batched_matmul": {"batch": 16, "m": 32, "n": 32, "k": 32}}, 16_384),
            ({"linalg": {"n": 2048, "rhs_cols": 16}}, 4_194_304),
            ({"linalg_ad": {"n": 256}}, 65_536),
            ({"elementwise_chain": {"n": 1024}}, 1024),
            ({"sparse": {"nnz": 67_108_864}}, 67_108_864),
            ({"einsum": {"shapes_rowmajor": [[3072, 3072], [3072, 3072]]}}, 9_437_184),
            ({"n": 1024, "shape": [32, 32]}, 1024),
            ({"shapes": [[32, 32], [1024]]}, 1024),
            ({"size_elements": 4096}, 4096),
        ]
        for problem, expected in cases:
            self.assertEqual(regime_policy.case_elements(problem), expected, problem)

    def test_floor_table_covers_every_gpu_family_with_a_basis(self):
        families = set(regime_policy.SUITE_FAMILIES.values())
        self.assertTrue(families)
        for family in families:
            floor = regime_policy.FAMILY_FLOORS.get(family)
            self.assertIsNotNone(floor, family)
            self.assertGreater(floor.elements, 0, family)
            self.assertGreater(len(floor.basis), 40, family)

    def test_repository_manifests_satisfy_the_policy(self):
        import yaml

        for path in sorted((ROOT / "benchmarks").rglob("*.yaml")):
            if "manifests" in path.parts:
                continue
            document = yaml.safe_load(path.read_text())
            if not isinstance(document, dict) or not document.get("suite_id"):
                continue
            self.assertEqual(regime_policy.check_suite(document, ROOT), [], path.name)


class GeneratedSuiteTests(unittest.TestCase):
    """The suites the collector regenerates must carry the same declarations.

    `run_gpu_linalg_jvp_vjp.sh` rewrites both suites from their seeds before every
    collection, so a policy that only holds for the committed file would stop the
    run at load time on the next collection.
    """

    def test_generated_linalg_suites_match_the_committed_manifests(self):
        import yaml

        import generate_gpu_linalg_jvp_vjp_yaml as generator

        for name, suite in generator.SUITES.items():
            path = ROOT / "benchmarks" / "gpu" / f"{name}.yaml"
            self.assertEqual(yaml.safe_load(path.read_text()), suite, path.name)
            self.assertEqual(regime_policy.check_suite(suite, ROOT), [], path.name)

    def test_regime_follows_the_size_across_the_floor(self):
        def regime(n: int) -> str:
            case = {"family": "linalg_ad", "linalg_ad": {"n": n}}
            return regime_policy.regime_for_case("gpu/linalg_jvp_vjp", case)

        self.assertEqual(regime(256), "latency")
        self.assertEqual(regime(512), "latency")
        self.assertEqual(regime(1024), "throughput")
        self.assertEqual(regime(4096), "throughput")


class RegimePagesTests(unittest.TestCase):
    """Each page is generated from the cases it holds, not from a fixed banner."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.result = self.root / "result"
        self.benchmarks = self.root / "benchmarks" / "gpu"
        self.benchmarks.mkdir(parents=True)
        (self.result / "nvidia-gpu/gpu").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def add_suite(self, stem: str, suite: str, cases: list[tuple[str, float]], body="# report\n"):
        (self.benchmarks / f"{stem}.yaml").write_text(suite)
        (self.result / f"nvidia-gpu/gpu/{stem}.md").write_text(body)
        (self.result / f"nvidia-gpu/gpu/{stem}.summary.json").write_text(
            json.dumps(summary("nvidia-gpu", f"gpu/{stem}", cases))
        )

    def generate(self) -> Path:
        output = self.root / "site"
        self.assertEqual(
            dashboard.generate(
                self.result,
                output,
                baseline_dir=None,
                git_repo=None,
                instances_dir=self.root / "instances",
                benchmarks_dir=self.root / "benchmarks",
                latest_profiles=None,
            ),
            0,
        )
        return output

    def test_below_floor_throughput_case_fails_the_load(self):
        self.add_suite(
            "ad_throughput",
            suite_yaml(
                "gpu/linalg_ad_throughput",
                "  regime: throughput\n  run: {runs: 7}\n",
                "  - {id: qr_jvp_f64_256, family: linalg_ad, linalg_ad: {n: 256}}\n",
            ),
            [("qr_jvp_f64_256", 1.426)],
        )
        with self.assertRaises(SystemExit) as raised:
            self.generate()
        message = str(raised.exception)
        self.assertIn("below the linalg_ad floor", message)
        self.assertIn("65,536 elements", message)
        self.assertFalse((self.root / "site").exists())

    def test_measured_case_without_a_declaration_fails_the_load(self):
        # The suite has a manifest, so every measured case must resolve: a case
        # the manifest does not declare cannot inherit anything.
        self.add_suite(
            "dense",
            suite_yaml(
                "gpu/dense",
                "  run: {runs: 7}\n",
                "  - {id: declared_case, regime: throughput, size_elements: 1048576}\n",
            ),
            [("declared_case", 1.0), ("undeclared_case", 2.0)],
        )
        with self.assertRaises(SystemExit) as raised:
            self.generate()
        self.assertIn("declares no regime", str(raised.exception))

    def test_all_latency_suite_page_says_it_has_no_throughput_case(self):
        self.add_suite(
            "linalg_ad_throughput",
            suite_yaml(
                "gpu/linalg_ad_throughput",
                "  regime: throughput\n  measure: deferred_batch\n  run: {runs: 7}\n",
                "  - {id: qr_jvp_f64_256, family: linalg_ad, regime: latency, "
                "linalg_ad: {n: 256}}\n",
            ),
            [("qr_jvp_f64_256", 1.426)],
        )
        output = self.generate()
        page = (output / "nvidia-gpu/gpu/linalg_ad_throughput.html").read_text()
        self.assertIn("This suite has no throughput-regime case.", page)
        self.assertIn("latency page", page)
        self.assertNotIn("Throughput-regime cases", page)
        self.assertIn("Latency and per-call overhead (1)", page)
        # The deferred batch is stated where the numbers are, not hidden.
        self.assertIn("timed in deferred-batch mode", page)

    def test_moved_case_leaves_the_throughput_page(self):
        self.add_suite(
            "dense",
            suite_yaml(
                "gpu/dense",
                "  regime: latency\n  run: {runs: 7}\n",
                "  - {id: qr_jvp_f64_256, family: linalg_ad, linalg_ad: {n: 256}}\n",
            ),
            [("qr_jvp_f64_256", 1.426)],
        )
        output = self.generate()
        self.assertNotIn("qr_jvp_f64_256", (output / "gpu/index.html").read_text())
        self.assertIn("qr_jvp_f64_256", (output / "gpu/latency/index.html").read_text())

    def test_throughput_banner_states_the_rendered_size_range(self):
        self.add_suite(
            "dense",
            suite_yaml(
                "gpu/dense",
                "  regime: throughput\n  run: {runs: 7}\n",
                "  - {id: big_matmul, family: dense, matmul: {m: 3072, n: 3072, k: 3072}}\n"
                "  - {id: small_svd, family: dense, linalg: {m: 256, n: 256}}\n",
            ),
            [("big_matmul", 3.16), ("small_svd", 12.0)],
        )
        page = (self.generate() / "gpu/index.html").read_text()
        notice = re.search(r'<p class="notice"><strong>Throughput.*?</p>', page, re.S)
        self.assertIsNotNone(notice, page)
        text = re.sub(r"\s+", " ", notice.group(0))
        self.assertIn("from 65,536 to 9,437,184 elements", text)
        self.assertIn("at or above its family floor", text)
        self.assertNotIn("cases only. Per-call latency", text)

    def test_page_with_no_throughput_case_says_so(self):
        self.add_suite(
            "perf_issues",
            suite_yaml(
                "gpu/perf_issues",
                "  regime: latency\n  run: {runs: 7}\n",
                "  - {id: readback, params: {n: 4}}\n",
            ),
            [("readback", 278.0)],
        )
        page = (self.generate() / "gpu/index.html").read_text()
        notice = re.search(r'<p class="notice"><strong>No throughput.*?</p>', page, re.S)
        self.assertIsNotNone(notice, page)
        self.assertIn("No throughput-regime GPU case", notice.group(0))


class PublishedNumbersTests(unittest.TestCase):
    """The fix is a classification change: no number, median or bar may move.

    Renders the repository's own `result/` tree and checks every published
    median against the summary sidecar it came from, and that each case's bars
    are exactly its numeric arms wherever the case is published.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name) / "site"
        code = dashboard.generate(
            ROOT / "result",
            cls.output,
            baseline_dir=None,
            git_repo=None,
            instances_dir=ROOT / "data" / "instances",
            benchmarks_dir=ROOT / "benchmarks",
            latest_profiles=None,
        )
        if code != 0:
            raise AssertionError(f"dashboard generation failed with {code}")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def case_blocks(html: str) -> dict[str, list[str]]:
        """Case id → the chart fragments of its <h3> blocks (one per group)."""
        blocks: dict[str, list[str]] = {}
        for chunk in html.split("<h3>")[1:]:
            match = re.search(r"<code>([^<]+)</code>", chunk)
            if not match:
                continue
            blocks.setdefault(match.group(1), []).append(chunk.split("</svg>", 1)[0])
        return blocks

    def suite_pages(self):
        """(page, suite document) for every suite page the dashboard wrote."""
        for path in sorted((ROOT / "result").rglob("*.summary.json")):
            document = json.loads(path.read_text())
            profile = path.parent.parent.name
            stem = path.name.removesuffix(".summary.json")
            page = self.output / profile / path.parent.name / f"{stem}.html"
            if page.exists():
                yield page, document

    def numeric_rows(self, document) -> dict[str, list[float]]:
        rows: dict[str, list[float]] = {}
        for row in document.get("rows", []):
            if row.get("status") != "ok" or row.get("median") is None:
                continue
            rows.setdefault(row["case"], []).append(float(row["median"]))
        return rows

    def test_every_bar_and_label_is_one_median_from_the_summary(self):
        """No number moved: each case keeps exactly its measured arms."""
        checked = 0
        for page, document in self.suite_pages():
            expected = self.numeric_rows(document)
            if not expected:
                continue
            # The embedded original report is markdown-rendered HTML with its own
            # headings and tables: only the generated part carries the charts.
            html = page.read_text().split("<h2>Original report</h2>")[0]
            unit = str(document.get("unit") or "ms")
            for case, charts in self.case_blocks(html).items():
                values = expected.get(case)
                if not values:
                    continue
                bars = sum(
                    chart.count("<rect ") for chart in charts if 'class="bars"' in chart
                )
                self.assertEqual(bars, len(values), f"{page.name}: {case} bars")
                labels = [
                    match.split("·")[0].strip()
                    for chart in charts
                    for match in re.findall(r'class="chart-value">([^<]*)', chart)
                ]
                shown = []
                for label in labels:
                    number = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", label)
                    self.assertIsNotNone(number, f"{page.name}: {case} {label!r}")
                    shown.append(float(number.group(0)))
                self.assertEqual(len(shown), len(values), f"{page.name}: {case} labels")
                # Every rendered value is one of the case's measured medians.
                remaining = list(values)
                for value in shown:
                    match = min(remaining, key=lambda item: abs(item - value) / item)
                    self.assertLess(
                        abs(match - value) / match, 0.01,
                        f"{page.name}: {case} rendered {value} is not a median",
                    )
                    remaining.remove(match)
                checked += 1
        self.assertGreater(checked, 50)

    def test_no_published_case_claims_throughput_below_its_floor(self):
        metadata = dashboard.load_suite_metadata(ROOT / "benchmarks")
        checked = 0
        for suite_id, suite in metadata.items():
            if not regime_policy.is_policed(suite_id):
                continue
            for case, problem in (suite.get("problems") or {}).items():
                if problem.get("regime") != regime_policy.THROUGHPUT:
                    continue
                floor = regime_policy.FAMILY_FLOORS.get(problem.get("family") or "")
                if floor is None or problem.get("elements") is None:
                    continue
                self.assertGreaterEqual(
                    problem["elements"], floor.elements, f"{suite_id}: {case}"
                )
                checked += 1
        self.assertGreater(checked, 50)


if __name__ == "__main__":
    unittest.main()
