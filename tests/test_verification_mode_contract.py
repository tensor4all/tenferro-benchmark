"""The manifest's verification mode survives load, record, summary and page.

Three states have to stay distinguishable end to end: a row checked against a
CPU reference, a row checked by a cheaper reference, and a row that was measured
with no check at all.  The third is the dangerous one -- it must never read as
verified -- so it is refused without a reason at load time and carries its own
badge on the page.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import generate_dashboard as dashboard  # noqa: E402
import report_summary  # noqa: E402
from normalize_gpu_records import fold  # noqa: E402
from validate_benchmark_suite import validate_suite  # noqa: E402

SUITE_SCHEMA = json.loads((ROOT / "schemas/benchmark-suite.schema.json").read_text())
RESULT_SCHEMA = json.loads((ROOT / "schemas/benchmark-result.schema.json").read_text())
VERIFICATION_SCHEMA = RESULT_SCHEMA["properties"]["verification"]

BASE_SUITE = {
    "schema_version": 1,
    "suite_id": "gpu/linalg_jvp_vjp",
    "title": "GPU linalg AD",
    "description": "fixture",
    "defaults": {"run": {"warmups": 3, "runs": 7}},
    "backends": ["tenferro-cuda-trace"],
    "problems": [
        {
            "id": "linalg_ad_large_grad_sum_qr_jvp_f64_64",
            "family": "linalg_ad",
            "regime": "latency",
            "op": "jvp",
            "dtype": {"a": "f64"},
            "data": {"generator": "well_conditioned", "seed": 34},
            "linalg_ad": {"suite": "large", "loss": "grad_sum_qr", "n": 64, "matrix_seed": 34},
        }
    ],
}


def suite_with(verify):
    suite = json.loads(json.dumps(BASE_SUITE))
    suite["problems"][0]["verify"] = verify
    return suite


class SuiteSchema(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "suite.yaml"

    def validate(self, suite):
        self.path.write_text(yaml.safe_dump(suite))
        return validate_suite(self.path)

    def test_a_manifest_without_a_mode_is_unchanged_and_loads(self):
        # The tolerances are the whole of the block an existing manifest states.
        self.assertTrue(self.validate(suite_with({"reference": "cpu_fp64", "rtol": 1e-5, "atol": 1e-8})))

    def test_every_named_mode_loads(self):
        for mode in ("cpu_reference", "analytic", "finite_difference"):
            with self.subTest(mode=mode):
                self.assertTrue(
                    self.validate(suite_with({"mode": mode, "reference": "cpu_fp64",
                                              "rtol": 1e-5, "atol": 1e-8}))
                )

    def test_measure_only_with_a_reason_loads(self):
        self.assertTrue(self.validate(suite_with({"mode": "measure_only", "reason": "no reference fits"})))

    def test_measure_only_without_a_reason_is_refused_at_load(self):
        for block in ({"mode": "measure_only"},
                      {"mode": "measure_only", "reason": ""},
                      {"mode": "measure_only", "reason": "   "},
                      {"mode": "measure_only", "reference": "cpu_fp64", "rtol": 1e-5, "atol": 1e-8}):
            with self.subTest(block=block):
                self.assertFalse(self.validate(suite_with(block)))

    def test_a_verify_block_that_states_tolerances_still_needs_them(self):
        self.assertFalse(self.validate(suite_with({"mode": "analytic"})))


class ResultSchema(unittest.TestCase):
    def record(self, **verification):
        block = {
            "status": "not_run",
            "mode": "measure_only",
            "reference_backend": None,
            "max_abs_error": None,
            "max_rel_error": None,
            "residual": None,
            "rtol": None,
            "atol": None,
            "reason": "n=4096 has no affordable reference",
        }
        block.update(verification)
        return {
            "schema_version": 1,
            "suite_id": "gpu/linalg_jvp_vjp",
            "problem_id": "case_a",
            "op": "jvp",
            "backend": "tenferro-cuda-trace",
            "status": "ok",
            "timing": {
                "warmup_runs": 3, "timed_runs": 7, "compile_time_ms": None,
                "first_run_ms": 1.0, "median_ms": 1.0, "min_ms": 1.0, "p95_ms": 1.0,
                "iqr_ms": 0.0, "timing_scope": "steady_state_host_api_plus_device_sync",
            },
            "performance": {"tflops": None, "effective_bandwidth_gbps": None, "peak_memory_bytes": None},
            "verification": block,
            "execution": {
                "device": "cuda", "device_ordinal": 0, "execution_path": "test",
                "synchronization": "sync", "layout": "{}", "dtype": "{}",
                "notes": None, "unsupported_reason": None,
            },
        }

    def test_a_measured_unchecked_row_is_valid(self):
        jsonschema.validate(self.record(), RESULT_SCHEMA)

    def test_not_run_without_a_reason_is_refused(self):
        row = self.record(reason=None)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(row, RESULT_SCHEMA)

    def test_not_run_must_name_measure_only(self):
        row = self.record(mode="analytic")
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(row, RESULT_SCHEMA)

    def test_not_run_must_name_a_mode(self):
        # Without a mode, a row that says no check ran could still carry
        # `reference_backend: cpu_fp64` and `residual: 0.0`, which reads as
        # evidence. The status alone is not the claim; the pair is.
        #
        # Delete the mode from the FINAL record: `record(**block)` reinstates its
        # default, which made this test pass for the wrong reason (the non-null
        # evidence) even with the requirement removed from the schema.
        row = self.record()
        del row["verification"]["mode"]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(row, RESULT_SCHEMA)
        # And the record this defect was reported with: no mode plus the
        # invented evidence. Rejected as well, but it is the first case above
        # that keeps `required: ["mode"]` load-bearing.
        row = self.record(reference_backend="cpu_fp64", residual=0.0)
        del row["verification"]["mode"]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(row, RESULT_SCHEMA)

    def test_not_run_with_measure_only_is_a_record(self):
        # The pair the Rust writer emits for an unchecked measure_only case.
        jsonschema.validate(self.record(status="not_run", mode="measure_only"), RESULT_SCHEMA)

    def test_the_row_a_runner_writes_for_an_unchecked_case_is_valid(self):
        # `verification_mode::unchecked_verification` writes `not_run` for a
        # `measure_only` problem that never ran and `skipped` for every other
        # mode. Both have to be records, and a `measure_only` row that still
        # claims a check is not one.
        unchecked = self.record(status="not_run", reference_backend=None)
        unchecked["status"] = "unsupported"
        jsonschema.validate(unchecked, RESULT_SCHEMA)
        unchecked["status"] = "runtime_failed"
        jsonschema.validate(unchecked, RESULT_SCHEMA)
        checked = self.record(status="skipped", mode="cpu_reference", reference_backend=None)
        checked["status"] = "unsupported"
        jsonschema.validate(checked, RESULT_SCHEMA)
        claimed = self.record(status="skipped")
        claimed["status"] = "unsupported"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(claimed, RESULT_SCHEMA)

    def test_an_unknown_mode_is_refused(self):
        row = self.record(status="passed", mode="cpu_fp64")
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(row, RESULT_SCHEMA)

    def test_measure_only_cannot_carry_a_verdict(self):
        # The earlier constraint only ran when `status` was `not_run`, so a row
        # could name measure_only and still claim `passed`.
        row = self.record(status="passed")
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(row, RESULT_SCHEMA)

    def test_measure_only_cannot_carry_evidence(self):
        row = self.record(residual=0.0)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(row, RESULT_SCHEMA)


class RecordAndSummary(unittest.TestCase):
    def raw_row(self, **verification):
        return {
            "suite_id": "gpu/linalg_jvp_vjp",
            "problem_id": "case_a",
            "op": "jvp",
            "backend": "tenferro-cuda-trace",
            "status": "ok",
            "median_ms": 1.5,
            "verification": verification,
        }

    def test_the_fold_keeps_the_mode_and_the_missing_check(self):
        record = fold(self.raw_row(status="not_run", mode="measure_only",
                                   reason="no reference fits at n=4096"))
        self.assertEqual(record["verification"]["status"], "not_run")
        self.assertEqual(record["verification"]["mode"], "measure_only")
        self.assertEqual(record["verification"]["reason"], "no reference fits at n=4096")
        # A measurement with no check is still a measurement.
        self.assertEqual(record["timing"]["median_ms"], 1.5)

    def test_a_legacy_record_gains_no_mode(self):
        record = fold(self.raw_row(status="passed"))
        self.assertNotIn("mode", record["verification"])

    def test_the_fold_invents_no_reference(self):
        # A family runner that recorded no reference did not check against the
        # CPU; stamping `cpu_fp64` on it changed what a published number meant.
        record = fold(self.raw_row(status="passed"))
        self.assertIsNone(record["verification"]["reference_backend"])

    def test_the_fold_keeps_an_unchecked_row_self_consistent(self):
        # `measure_only` requires a null reference: inventing one produced the
        # one combination the result schema has to reject.
        record = fold(self.raw_row(status="not_run", mode="measure_only",
                                   reason="no reference fits at n=4096"))
        jsonschema.validate(record["verification"], VERIFICATION_SCHEMA)

    def test_a_legacy_record_makes_a_byte_identical_summary_row(self):
        legacy = {
            "suite_id": "gpu/linalg_jvp_vjp", "problem_id": "case_a", "op": "jvp",
            "backend": "tenferro-cuda-trace", "status": "ok",
            "timing": {"median_ms": 1.5, "iqr_ms": 0.1},
            "verification": {"status": "passed", "reference_backend": "cpu_fp64",
                             "max_abs_error": 0.0, "max_rel_error": 0.0, "residual": 0.0},
        }
        row = report_summary.rows_from_gpu_records([legacy])[0]
        self.assertEqual(
            row,
            {"case": "case_a", "group": "jvp", "backend": "tenferro-cuda-trace",
             "median": 1.5, "iqr": 0.1, "status": "ok", "verified": True},
        )

    def test_the_summary_carries_the_mode_and_the_reason(self):
        unchecked = {
            "suite_id": "gpu/linalg_jvp_vjp", "problem_id": "case_a", "op": "jvp",
            "backend": "tenferro-cuda-trace", "status": "ok",
            "timing": {"median_ms": 1.5, "iqr_ms": 0.1},
            "verification": {"status": "not_run", "mode": "measure_only",
                             "reason": "n=4096 has no affordable reference"},
        }
        row = report_summary.rows_from_gpu_records([unchecked])[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["median"], 1.5)
        self.assertFalse(row["verified"])
        self.assertEqual(row["verification"], "measure_only")
        self.assertEqual(row["reason"], "n=4096 has no affordable reference")


class SummarySchema(unittest.TestCase):
    """A measure_only row is never published as verified."""

    SCHEMA = json.loads((ROOT / "schemas/report-summary.schema.json").read_text())

    def document(self, **row):
        base = {
            "case": "case_a", "group": "jvp", "backend": "tenferro-cuda-trace",
            "median": 1.5, "status": "ok", "verified": False,
            "verification": "measure_only", "reason": "n=4096 has no affordable reference",
        }
        base.update(row)
        return {
            "schema_version": report_summary.SCHEMA_VERSION,
            "target_profile": "nvidia-gpu",
            "suite_id": "gpu/linalg_jvp_vjp",
            "timestamp": "2026-11-08T00:00:00+00:00",
            "tenferro_commit": "abc123",
            "hardware": {"gpu": "NVIDIA A100 80GB PCIe"},
            "unit": "ms",
            "rows": [base],
        }

    def test_an_unchecked_row_is_valid(self):
        jsonschema.validate(self.document(), self.SCHEMA)

    def test_an_unchecked_row_cannot_be_verified(self):
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(self.document(verified=True), self.SCHEMA)

    def test_an_unchecked_row_states_why(self):
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(self.document(reason=""), self.SCHEMA)


class DashboardBadges(unittest.TestCase):
    """One table holds a number, an `unverified` badge and a `fail`."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = self.root / "result"
        self.benchmarks = self.root / "benchmarks" / "gpu"
        self.benchmarks.mkdir(parents=True)
        (self.benchmarks / "jvp.yaml").write_text(
            "schema_version: 1\nsuite_id: gpu/jvp\ndefaults:\n  regime: latency\n"
            "problems:\n  - {id: case_mixed}\n  - {id: case_unchecked}\n"
            "  - {id: case_plain}\n"
        )

    def generate(self, rows, name="site"):
        suite_dir = self.result / "nvidia-gpu" / "gpu"
        suite_dir.mkdir(parents=True, exist_ok=True)
        report = suite_dir / "jvp.md"
        report.write_text("# gpu/linalg_jvp_vjp\n\nbody\n")
        document = {
            "schema_version": report_summary.SCHEMA_VERSION,
            "target_profile": "nvidia-gpu",
            "suite_id": "gpu/linalg_jvp_vjp",
            "timestamp": "2026-11-07T00:00:00+00:00",
            "tenferro_commit": "abc123",
            "hardware": {"gpu": "NVIDIA A100 80GB PCIe"},
            "unit": "ms",
            "rows": rows,
        }
        report.with_suffix(".summary.json").write_text(json.dumps(document))
        output = self.root / name
        dashboard.generate(
            self.result,
            output,
            baseline_dir=self.root / "baseline",
            git_repo=None,
            instances_dir=self.root / "instances",
            benchmarks_dir=self.root / "benchmarks",
            latest_profiles=None,
        )
        return (output / "nvidia-gpu/gpu/jvp.html").read_text()

    @staticmethod
    def row(case, backend, median, **extra):
        return {
            "case": case, "group": "grad_sum_qr_jvp", "backend": backend,
            "median": median, "status": "ok", "verified": True, **extra,
        }

    def test_half_verified_half_unchecked_and_one_failed_arm(self):
        page = self.generate([
            self.row("case_mixed", "tenferro-cuda-trace", 20.0,
                     verification="analytic"),
            self.row("case_mixed", "tenferro-cuda-eager", 15.0,
                     verification="cpu_reference"),
            {"case": "case_mixed", "group": "grad_sum_qr_jvp",
             "backend": "jax-cuda", "median": 10.0, "status": "ok",
             "verified": True, "verification": "finite_difference"},
            {"case": "case_mixed", "group": "grad_sum_qr_jvp",
             "backend": "pytorch-cuda", "median": None, "status": "failed",
             "verified": False, "reason": "CUDA error: misaligned address"},
            {"case": "case_unchecked", "group": "grad_sum_qr_jvp",
             "backend": "tenferro-cuda-trace", "median": 30.0, "status": "ok",
             "verified": False, "verification": "measure_only",
             "reason": "n=4096 has no affordable reference"},
        ])
        # The column and its one-line legend appear together.
        self.assertIn("<th>verification</th>", page)
        self.assertIn("Verification: ", page)
        # Three checks are named, and only the analytic/finite-difference ones
        # need a tooltip to be read.
        self.assertIn('class="verified-mode" title="every element compared against', page)
        self.assertIn('>analytic</span>', page)
        self.assertIn('>finite-difference</span>', page)
        self.assertIn('class="verified-mode" title="the whole case re-executed', page)
        # The unchecked row keeps its number and states why it has no evidence.
        self.assertIn('class="unverified" title="measured without a numerical check', page)
        self.assertIn("n=4096 has no affordable reference", page)
        self.assertIn(">unverified</span>", page)
        # A failed arm is still a fail, not an unverified one.
        self.assertIn('<span class="fail">fail</span>', page)
        self.assertIn("CUDA error: misaligned address", page)
        # The unchecked row is not an anomaly: no flags entry for it.
        unchecked_row = page.split("<code>case_unchecked</code>")[1].split("</tr>")[0]
        self.assertNotIn('class="flag"', unchecked_row)
        self.assertIn(">unverified</span>", unchecked_row)

    def test_a_page_of_default_checks_gains_no_column(self):
        page = self.generate([
            self.row("case_plain", "tenferro-cuda-trace", 20.0),
            self.row("case_plain", "pytorch-cuda", 10.0),
        ])
        self.assertNotIn("<th>verification</th>", page)
        self.assertNotIn("Verification: ", page)
        self.assertNotIn(".unverified {", page)


class IngestionGates(unittest.TestCase):
    """The pod-side gates accept a stated measure_only row, and nothing less."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "records.jsonl"

    def write(self, verification):
        row = {
            "schema_version": 1, "suite_id": "gpu/linalg_jvp_vjp",
            "problem_id": "case_a", "op": "jvp", "backend": "tenferro-cuda-trace",
            "status": "ok",
            "timing": {"warmup_runs": 3, "timed_runs": 7, "median_ms": 1.5,
                       "iqr_ms": 0.1, "timing_scope": "steady_state_host_api_plus_device_sync"},
            "performance": {"tflops": None, "effective_bandwidth_gbps": None, "peak_memory_bytes": None},
            "verification": verification,
            "execution": {"device": "cuda", "device_ordinal": 0, "execution_path": "test",
                          "synchronization": "sync", "layout": "{}", "dtype": "{}",
                          "notes": None, "unsupported_reason": None},
        }
        self.path.write_text(json.dumps(row) + "\n")
        return row

    def stated(self):
        return self.write({"status": "not_run", "mode": "measure_only", "reference_backend": None,
                           "max_abs_error": None, "max_rel_error": None, "residual": None,
                           "rtol": None, "atol": None,
                           "reason": "n=4096 has no affordable reference"})

    def test_the_run_gate_accepts_a_stated_measure_only_row(self):
        from scripts.ci.check_gpu_run import validate_records

        self.stated()
        validate_records(self.path, "gpu/linalg_jvp_vjp",
                         {("case_a", "tenferro-cuda-trace")})

    def test_the_run_gate_still_refuses_a_row_that_lost_its_check(self):
        from scripts.ci.check_gpu_run import validate_records

        self.write({"status": "not_run", "mode": "measure_only", "reference_backend": None,
                    "max_abs_error": None, "max_rel_error": None, "residual": None,
                    "rtol": None, "atol": None, "reason": "  "})
        with self.assertRaisesRegex(ValueError, "verification did not pass"):
            validate_records(self.path, "gpu/linalg_jvp_vjp",
                             {("case_a", "tenferro-cuda-trace")})

    def test_the_rows_view_keeps_a_stated_measure_only_row_a_measurement(self):
        from scripts.ci.bench_rows import gpu_rows

        self.stated()
        [row] = list(gpu_rows([self.path]))
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["median_ms"], 1.5)
        self.assertEqual(row["note"], "n=4096 has no affordable reference")

    def test_the_rows_view_still_fails_an_unexplained_unverified_row(self):
        from scripts.ci.bench_rows import gpu_rows

        self.write({"status": "skipped", "reference_backend": None,
                    "max_abs_error": None, "max_rel_error": None, "residual": None,
                    "rtol": None, "atol": None, "reason": None})
        [row] = list(gpu_rows([self.path]))
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["median_ms"])


if __name__ == "__main__":
    unittest.main()
