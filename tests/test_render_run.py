"""Offline rendering from a run directory: determinism, provenance, schema v2."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import format_cpu_thread_reports  # noqa: E402
import render_run  # noqa: E402
import report_summary  # noqa: E402
from scripts.ci.result_ingestion import validate_summary  # noqa: E402

FIXTURES = ROOT / "tests/fixtures/render_run"
SUMMARY_SCHEMA = json.loads((ROOT / "schemas/report-summary.schema.json").read_text())

RUNS = {
    "einsum": ("20261018_000001", "einsum", "cpu/einsum"),
    "permutation": ("20261018_010000", "permutation", "cpu/permutation"),
    "overhead_smoke": ("20261018_020000", "overhead_smoke", "cpu/overhead_smoke"),
    "perf_issues": ("20261018_030000", "perf_issues", "cpu/perf_issues"),
    "session_matrix": ("20261018_040000", "session_matrix", "cpu/session_matrix"),
    "fft": ("20261018_050000", "fft", "cpu/fft"),
}

# The power condition a summary records is the toolkit's serialization of the
# run's `host.power` block.
BOOST_OFF = "boost off, performance"
BOOST_OFF_WITH_PROFILE = (
    "boost off, performance, platform balanced, smt {'active': '1', 'control': 'auto'}"
)
BOOST_ON = "boost on, powersave"
POWER_BY_RUN = {
    "einsum": BOOST_OFF_WITH_PROFILE,
    "fft": BOOST_OFF_WITH_PROFILE,
    "permutation": BOOST_OFF,
    "overhead_smoke": BOOST_OFF,
    "perf_issues": BOOST_OFF,
    "session_matrix": BOOST_OFF,
}


def render_to(directory: Path, run_dir: Path) -> dict[str, bytes]:
    render_run.render(run_dir, directory)
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class RenderRunTest(unittest.TestCase):
    def fixture(self, name: str) -> Path:
        stamp, suite, _ = RUNS[name]
        return FIXTURES / suite / stamp

    def test_render_is_deterministic_and_ignores_live_host_state(self):
        for name, (_, suite, suite_id) in RUNS.items():
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second, tempfile.TemporaryDirectory() as third:
                run_dir = self.fixture(name)
                before = {
                    str(path.relative_to(run_dir)): path.read_bytes()
                    for path in sorted(run_dir.rglob("*"))
                    if path.is_file()
                }
                base = render_to(Path(first), run_dir)
                after = {
                    str(path.relative_to(run_dir)): path.read_bytes()
                    for path in sorted(run_dir.rglob("*"))
                    if path.is_file()
                }
                self.assertEqual(before, after, f"{suite}: rendering wrote into the run directory")
                self.assertTrue(base, f"{suite}: rendered nothing")
                self.assertTrue(any(key.endswith(".summary.json") for key in base))

                # Anything that reads the current shell, CPU or clock would
                # change this output.
                with patch.dict(
                    os.environ,
                    {
                        "RAYON_NUM_THREADS": "999",
                        "OMP_NUM_THREADS": "999",
                        "BENCHMARK_TARGET_PROFILE": "bogus-cpu",
                        "BENCHMARK_TIMESTAMP": "19990101_000000",
                    },
                ), patch(
                    "collect_cpu_info.collect_cpu_info",
                    side_effect=AssertionError("renderer probed the live CPU"),
                ), patch(
                    "collect_cpu_info.markdown",
                    side_effect=AssertionError("renderer probed the live CPU"),
                ), patch(
                    "platform.processor", side_effect=AssertionError("renderer probed the live CPU")
                ):
                    live = render_to(Path(second), run_dir)
                self.assertEqual(base, live)

                # Rendering twice into the same directory is idempotent.
                render_run.render(run_dir, Path(third))
                self.assertEqual(base, render_to(Path(third), run_dir))

    def test_every_summary_is_v2_and_validates(self):
        for name, (_, suite, suite_id) in RUNS.items():
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as directory:
                outputs = render_to(Path(directory), self.fixture(name))
                summaries = [key for key in outputs if key.endswith(".summary.json")]
                self.assertTrue(summaries)
                for key in summaries:
                    document = json.loads(outputs[key])
                    jsonschema.validate(document, SUMMARY_SCHEMA)
                    if key == "report.summary.json":
                        continue
                    self.assertEqual(document["schema_version"], 2)
                    self.assertEqual(document["suite_id"], suite_id)
                    self.assertEqual(
                        document["hardware"]["cpu"], "Fixture Zen 5 CPU"
                    )
                    self.assertTrue(document["source_runs"])
                    for source in document["source_runs"]:
                        self.assertFalse(Path(source["run"]).is_absolute())
                        self.assertEqual(source["power"], POWER_BY_RUN[name])
                        self.assertEqual(source["toolchain"]["target_cpu"], "native")

    def test_perf_issue_rows_keep_their_thread_count(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = render_to(Path(directory), self.fixture("perf_issues"))
            document = json.loads(outputs["perf_issues.summary.json"])
            groups = {row["group"] for row in document["rows"]}
            self.assertIn("threads=1 / steady_state", groups)
            self.assertIn("threads=8 / steady_state", groups)
            identities = [(row["case"], row["group"], row["backend"]) for row in document["rows"]]
            self.assertEqual(len(identities), len(set(identities)))
            self.assertEqual(document["hardware"]["threads"], [1, 8])
            self.assertEqual([entry["threads"] for entry in document["source_runs"]], [[1], [8]])

    def test_einsum_thread_runs_combine_with_union_source_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for threads in (1, 4):
                target = root / f"run{threads}"
                shutil.copytree(
                    FIXTURES / "einsum" / f"20261018_00000{threads}", target
                )
                render_run.render(target)
                runs.append(target)
            format_cpu_thread_reports.combine(runs, root)
            document = json.loads(
                (root / "result/mac-cpu/cpu/einsum.summary.json").read_text()
            )
            self.assertEqual(document["schema_version"], 2)
            self.assertEqual(document["hardware"]["threads"], [1, 4])
            self.assertEqual(len(document["source_runs"]), 2)
            self.assertEqual(
                [source["threads"] for source in document["source_runs"]], [[1], [4]]
            )
            # Each source keeps its own recorded conditions.
            self.assertEqual(
                [source["power"] for source in document["source_runs"]],
                [BOOST_OFF_WITH_PROFILE, BOOST_ON],
            )
            groups = {row["group"] for row in document["rows"]}
            self.assertEqual(groups, {"Threads: 1 / opt_flops", "Threads: 4 / opt_flops"})

    def test_merge_refuses_mixed_commit_or_timing_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = render_to(Path(directory), self.fixture("overhead_smoke"))
            first = json.loads(outputs["report.summary.json"])
            second = json.loads(json.dumps(first))
            report_summary.merge_thread_documents([first, second])
            for field, mutate in (
                ("tenferro_commit", lambda doc: doc.update(tenferro_commit="c" * 40)),
                ("timing_policy", lambda doc: doc["timing_policy"].update(version=3)),
            ):
                with self.subTest(field=field):
                    changed = json.loads(json.dumps(first))
                    mutate(changed)
                    with self.assertRaises(ValueError):
                        report_summary.merge_thread_documents([first, changed])

    def test_ingestion_validates_v1_and_v2_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = render_to(Path(directory), self.fixture("overhead_smoke"))
            document = json.loads(outputs["overhead_smoke.summary.json"])
            entry = {
                "suite_id": document["suite_id"],
                "tenferro_commit": document["tenferro_commit"],
                "hardware": document["hardware"],
                "timestamp": document["timestamp"],
                "expected_rows": [
                    [row["case"], row.get("group", ""), row["backend"]]
                    for row in document["rows"]
                ],
            }
            validate_summary(document, "zen5-cpu", entry)
            legacy = json.loads(json.dumps(document))
            legacy["schema_version"] = 1
            legacy.pop("source_runs")
            legacy.pop("timing_policy")
            validate_summary(legacy, "zen5-cpu", entry)
            legacy["schema_version"] = 3
            with self.assertRaises(ValueError):
                validate_summary(legacy, "zen5-cpu", entry)

    def test_permutation_reads_recorded_cpu_and_pinning(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = render_to(Path(directory), self.fixture("permutation"))
            report = outputs["report.md"].decode()
            self.assertIn("- Model: `Fixture Zen 5 CPU`", report)
            self.assertIn("pinned with `taskset`", report)
            self.assertIn("measured CPUs `0`", report)
            self.assertNotIn("no CPU-affinity pinning", report)
            self.assertNotIn("Darwin", report)

    def test_cpu_model_is_read_from_the_recorded_snapshot(self):
        self.assertEqual(
            report_summary.cpu_model_from_run_dir(self.fixture("einsum")),
            "Fixture Zen 5 CPU",
        )
        self.assertIsNone(report_summary.cpu_model_from_run_dir(None))

    def test_render_rejects_an_unidentifiable_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(render_run.RenderError):
                render_run.render(Path(directory))

    def test_cpu_ops_statistics_are_recomputed_from_the_raw_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            shutil.copytree(FIXTURES / "einsum" / "20261018_000002", run_dir)
            outputs = render_to(Path(directory) / "out", run_dir)
            self.assertIn("cpu_ops_report.md", outputs)
            self.assertIn("linalg_jvp_vjp_report.md", outputs)

            samples = run_dir / "cpu_ops_samples_t2_20261018_000002.jsonl"
            original = samples.read_text()
            records = [json.loads(line) for line in original.splitlines()]
            edited = [dict(record) for record in records]
            edited[0]["per_operation_ns"] = [
                value * 1.5 for value in edited[0]["per_operation_ns"]
            ]
            samples.write_text("".join(json.dumps(record) + "\n" for record in edited))
            with self.assertRaises(render_run.RenderError):
                render_run.render(run_dir, Path(directory) / "edited")

            # A consistent batch-time edit still moves the published statistics.
            samples.write_text(original)
            doubled = [dict(record) for record in records]
            doubled[0]["batch_elapsed_ns"] = [2 * doubled[0]["batch_elapsed_ns"][0]] + doubled[0]["batch_elapsed_ns"][1:]
            doubled[0]["per_operation_ns"] = [2 * doubled[0]["per_operation_ns"][0]] + doubled[0]["per_operation_ns"][1:]
            samples.write_text("".join(json.dumps(record) + "\n" for record in doubled))
            with self.assertRaises(render_run.RenderError):
                render_run.render(run_dir, Path(directory) / "doubled")

            # Editing the CSV statistic the page publishes fails the same way.
            samples.write_text(original)
            csv_path = run_dir / "cpu_ops_t2_20261018_000002.csv"
            csv_path.write_text(csv_path.read_text().replace("0.027213", "0.031000"))
            with self.assertRaises(render_run.RenderError):
                render_run.render(run_dir, Path(directory) / "csv")

    def test_fft_summary_is_the_shared_cpu_ops_document(self):
        import format_cpu_ops_results

        run_dir = self.fixture("fft")
        metadata, _ = render_run.load_metadata(run_dir)
        document = format_cpu_ops_results.build_summary(
            [run_dir / "cpu_fft_t1_20261018_050000.csv"],
            metadata,
            suite_id="cpu/fft",
            run_dir=run_dir,
            cpu_model="Fixture Zen 5 CPU",
        )
        expected = {
            "schema_version": 2,
            "target_profile": "mac-cpu",
            "suite_id": "cpu/fft",
            "timestamp": "2026-10-18T00:00:05+00:00",
            "tenferro_commit": "b" * 40,
            "hardware": {"cpu": "Fixture Zen 5 CPU", "host": "fixture-host", "threads": [1]},
            "unit": "ms",
            "rows": [
                {
                    "case": "fft_1d_f64_1024",
                    "group": "suite=cpu/fft / dtype=f64 / threads=1 / shape=1024",
                    "backend": "pytorch-cpu",
                    "median": 0.014,
                    "iqr": 0.0004,
                    "status": "ok",
                    "verified": True,
                },
                {
                    "case": "fft_1d_f64_1024",
                    "group": "suite=cpu/fft / dtype=f64 / threads=1 / shape=1024",
                    "backend": "tenferro-fft-executor-cached",
                    "median": 0.012345,
                    "iqr": 0.000321,
                    "status": "ok",
                    "verified": True,
                },
                {
                    "case": "fft_1d_f64_immediate",
                    "group": "suite=cpu/fft_setup / dtype=f64 / threads=1 / shape=1024",
                    "backend": "tenferro-fft-immediate",
                    "median": None,
                    "status": "unsupported: BENCH_INCLUDE_SETUP_DIAGNOSTICS=1",
                    "verified": False,
                },
            ],
            "source_runs": [
                {
                    "run": report_summary.repo_relative(run_dir),
                    "threads": [1],
                    "host": {
                        "cpu": "Fixture Zen 5 CPU",
                        "logical_cpus": 24,
                        "power": {
                            "cpufreq_boost": "0",
                            "governor": "performance",
                            "platform_profile": "balanced",
                            "smt": {"active": "1", "control": "auto"},
                        },
                    },
                    "toolchain": {
                        "rustc": "rustc 1.90.0",
                        "rustc_release": "1.90.0",
                        "rustflags": "-C target-cpu=native",
                        "target_cpu": "native",
                    },
                    "power": BOOST_OFF_WITH_PROFILE,
                }
            ],
            "timing_policy": {
                "version": 2,
                "scope": "operation execution",
                "setup": "inputs outside timer",
                "cleanup": "outputs retained",
                "minimum_untimed_priming_runs": 1,
                "setup_diagnostics": False,
            },
        }
        self.assertEqual(
            json.dumps(document, indent=2) + "\n", json.dumps(expected, indent=2) + "\n"
        )

    def test_rust_only_permutation_run_renders(self):
        # A supported collection with no Julia participant records only its Rust
        # output; an empty optional JSONL placeholder is not a record.
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            shutil.copytree(FIXTURES / "permutation" / "20261018_010000", run_dir)
            self.assertEqual(list(run_dir.glob("julia_output*.jsonl")), [])
            outputs = render_to(Path(directory) / "out", run_dir)
            self.assertIn("permutation.summary.json", outputs)
            report = outputs["report.md"].decode()
            self.assertIn("## Threads: 1", report)
            self.assertIn("| `transpose_2d_2048` |", report)

        runner = (ROOT / "scripts/run_permutation.sh").read_text()
        guard = runner.index('if [[ "$HAVE_JULIA" == "1" ]]; then')
        self.assertLess(guard, runner.index(': > "$JULIA_JSONL"'))

    def test_session_matrix_render_ignores_the_collection_timeout(self):
        with patch.dict(os.environ, {"BENCH_CASE_TIMEOUT_S": "not-a-number"}), \
                tempfile.TemporaryDirectory() as directory:
            outputs = render_to(Path(directory), self.fixture("session_matrix"))
            self.assertIn("report.md", outputs)
            self.assertIn("session_matrix.summary.json", outputs)

    def test_cli_writes_into_an_explicit_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            code = render_run.main([str(self.fixture("perf_issues")), "--out", directory])
            self.assertEqual(code, 0)
            self.assertTrue((Path(directory) / "report.md").is_file())
            self.assertTrue((Path(directory) / "perf_issues.summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
