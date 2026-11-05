"""The bandwidth diagnostic records the logical-byte rate and never rejects a record."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_gpu_public_api as bench  # noqa: E402
import format_gpu_public_api_results as formatter  # noqa: E402
import gpu_bandwidth_guard as guard  # noqa: E402
import gpu_case_traffic as traffic  # noqa: E402
import gpu_copy_calibration as calibration  # noqa: E402
from gpu_multi_output_verification import leaves  # noqa: E402
from gpu_public_api_ops import fixtures  # noqa: E402

# stack_axis0_f32_n1048576: two declared 1024x1024 f32 inputs and one 2x1024x1024
# f32 output, so 16 MiB of logical traffic per operation.
CASE = "stack_axis0_f32_n1048576"
BYTES = 16 * 1024 * 1024
# A copy of 64 MiB moves 128 MiB; 96 us per copy is about 1.4 TB/s.
COPY_SAMPLES_NS = [95_800, 96_000, 96_200]
BUFFER_BYTES = 64 << 20


def calibration_metadata():
    document = calibration.copy_document(
        COPY_SAMPLES_NS, BUFFER_BYTES, device_ordinal=0, device_name="test device",
        warmups=3, torch_version="test", cuda_version="test",
        timestamp="2026-11-04T00:00:00+00:00")
    return {"gpu_memory_calibration": document}, document


def raw_record(*, count=1, batch_ms, case_id=CASE, bytes_per_operation=BYTES,
               status="ok"):
    traffic_block = {
        "kind": traffic.TRAFFIC_KIND,
        "bytes_per_operation": bytes_per_operation,
        "reads_bytes": None if bytes_per_operation is None else bytes_per_operation // 2,
        "writes_bytes": None if bytes_per_operation is None else bytes_per_operation // 2,
        "destination_read_included": False,
        "basis": traffic.BASIS,
        "unknown_reason": None if bytes_per_operation is not None
        else "case declares no output array, so the bytes it must write are not derivable",
    }
    return {"schema_version": 1, "case_id": case_id, "op": "stack", "backend": "pytorch-cuda",
            "status": status, "traffic": traffic_block,
            "timing": {"operations_per_sample": count, "batch_ms": batch_ms,
                       "warmups": 3, "scope": "steady_state_host_api_plus_device_sync"},
            "verification": {"max_error_over_tolerance": 0.0, "reference": "numpy"}}


def run_guard(rows, metadata):
    """Run the diagnostic exactly as the family runner does, one arm file at a time."""
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "run.yaml").write_text(yaml.safe_dump(metadata))
        path = root / "arm.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        process = subprocess.run(
            [sys.executable, str(ROOT / "scripts/gpu_bandwidth_guard.py"),
             "--run-metadata", str(root / "run.yaml"), str(path)],
            capture_output=True, text=True, check=False)
        written = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return process, written


class BandwidthDiagnosticRun(unittest.TestCase):
    def test_a_record_above_the_copy_rate_keeps_its_number_and_records_the_ratio(self):
        metadata, copy = calibration_metadata()
        # 0.283 us per operation for 16 MiB is about 59 TB/s, the number the
        # 2026-11-04 run reported for this case on an A100 80 GB. A cache-resident
        # case can legitimately do this, so it is recorded and kept.
        process, written = run_guard([raw_record(batch_ms=[0.000283])], metadata)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn(CASE, process.stderr)
        row = written[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["timing"]["batch_ms"], [0.000283])
        self.assertNotIn("reason", row)
        block = row["bandwidth_guard"]
        self.assertEqual(block["decision"], "above_copy_rate")
        self.assertGreater(block["implied_bytes_per_second"], 5e13)
        self.assertGreater(block["copy_rate_ratio"], 40)
        self.assertEqual(block["copy_bytes_per_second"], copy["bytes_per_second"])
        self.assertIn("not judged", block["reason"])

    def test_a_record_within_the_copy_rate_is_recorded_the_same_way(self):
        metadata, copy = calibration_metadata()
        # 16 MiB in 21 us per operation is 799 GB/s, below the 1.4 TB/s copy rate.
        process, written = run_guard([raw_record(batch_ms=[0.021])], metadata)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(process.stderr, "")
        row = written[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["timing"]["batch_ms"], [0.021])
        self.assertEqual(row["bandwidth_guard"]["decision"], "within_copy_rate")
        self.assertLess(row["bandwidth_guard"]["copy_rate_ratio"], 1.0)
        self.assertEqual(row["bandwidth_guard"]["copy_bytes_per_second"],
                         copy["bytes_per_second"])

    def test_the_ratio_uses_one_operation_not_one_batch(self):
        metadata, _ = calibration_metadata()
        # A batch of 64 operations in 0.2 ms is 3.125 us per operation, so the ratio
        # is above one; the raw batch time alone would look slow.
        process, written = run_guard([raw_record(count=64, batch_ms=[0.2])], metadata)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertAlmostEqual(written[0]["bandwidth_guard"]["median_ms_per_operation"],
                               0.2 / 64)
        self.assertEqual(written[0]["bandwidth_guard"]["decision"], "above_copy_rate")
        self.assertEqual(written[0]["timing"]["batch_ms"], [0.2])
        # A long batch of many operations is not a long operation.
        process, written = run_guard([raw_record(count=2048, batch_ms=[40.0])], metadata)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(written[0]["bandwidth_guard"]["decision"], "within_copy_rate")
        self.assertEqual(written[0]["status"], "ok")

    def test_unknown_traffic_is_marked_not_guessed(self):
        metadata, _ = calibration_metadata()
        process, written = run_guard(
            [raw_record(batch_ms=[0.021], bytes_per_operation=None)], metadata)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(written[0]["status"], "ok")
        self.assertIsNone(written[0]["bandwidth_guard"]["copy_rate_ratio"])
        self.assertEqual(written[0]["bandwidth_guard"]["decision"], "not_checked")
        self.assertIn("no output array", written[0]["bandwidth_guard"]["reason"])

    def test_a_missing_calibration_leaves_the_ratio_unavailable(self):
        process, written = run_guard([raw_record(batch_ms=[0.021])], {})
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("gpu_memory_calibration", process.stderr)
        self.assertEqual(written[0]["status"], "ok")
        self.assertEqual(written[0]["timing"]["batch_ms"], [0.021])
        self.assertEqual(written[0]["bandwidth_guard"]["decision"], "not_checked")
        self.assertIsNone(written[0]["bandwidth_guard"]["copy_rate_ratio"])
        self.assertIn("no usable device copy rate", written[0]["bandwidth_guard"]["reason"])

    def test_an_unusable_copy_rate_does_not_fail_the_run(self):
        for rate in (0, -1.0, "not a number"):
            metadata = {"gpu_memory_calibration": {
                "kind": "device_copy", "method": "test", "device_ordinal": 0,
                "buffer_bytes": BUFFER_BYTES, "traffic_bytes_per_copy": 2 * BUFFER_BYTES,
                "median_ms": 0.096, "bytes_per_second": rate}}
            process, written = run_guard([raw_record(batch_ms=[0.021])], metadata)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(written[0]["status"], "ok")
            self.assertEqual(written[0]["timing"]["batch_ms"], [0.021])
            self.assertIsNone(written[0]["bandwidth_guard"]["copy_rate_ratio"])

    def test_a_verification_failure_is_left_exactly_as_the_arm_recorded_it(self):
        metadata, _ = calibration_metadata()
        row = raw_record(batch_ms=[0.021], status="verification_failed")
        row["reason"] = "verification error / tolerance = 3"
        process, written = run_guard([row], metadata)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(written[0]["status"], "verification_failed")
        self.assertEqual(written[0]["reason"], "verification error / tolerance = 3")
        self.assertNotIn("bandwidth_guard", written[0])


class CaseTraffic(unittest.TestCase):
    def test_declared_arrays_size_the_logical_traffic(self):
        case = dict(op="stack", dtype="f32", n=1048576, shape=[1024, 1024], axis=0)
        inputs, expected = fixtures(case, bench.DTYPES["f32"])
        block = traffic.case_traffic(inputs, leaves(expected), op="stack")
        self.assertEqual(block["bytes_per_operation"], BYTES)
        self.assertEqual(block["reads_bytes"], 2 * 4 * 1024 * 1024)
        self.assertEqual(block["writes_bytes"], 8 * 1024 * 1024)
        self.assertEqual(block["kind"], "logical_traffic")
        self.assertIsNone(block["unknown_reason"])

    def test_an_accumulating_case_counts_its_declared_destination(self):
        block = traffic.case_traffic([np.zeros(8, dtype=np.float32)] * 3,
                                     [np.zeros(8, dtype=np.float32)],
                                     op="dot_general_read_into_accum")
        self.assertTrue(block["destination_read_included"])
        self.assertEqual(block["reads_bytes"], 3 * 32)

    def test_arrays_that_do_not_determine_the_traffic_are_marked_unknown(self):
        block = traffic.case_traffic([np.zeros(8, dtype=np.float32)], [],
                                     op="reduce_sum")
        self.assertIsNone(block["bytes_per_operation"])
        self.assertIn("no output array", block["unknown_reason"])

    def test_the_fixture_step_writes_traffic_into_every_manifest_case(self):
        suite = {"suite_id": "gpu/structural", "backends": ["pytorch-cuda"],
                 "cases": [dict(id="stack_axis0_f32_n64", op="stack", dtype="f32",
                                n=64, shape=[8, 8], axis=0)]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixtures"
            bench.make_fixtures(root, suite)
            case = json.loads((root / "manifest.json").read_text())[0]
            self.assertEqual(case["traffic"]["bytes_per_operation"], 4 * 64 * 4)
            self.assertIsNone(case["traffic"]["unknown_reason"])

    def test_the_copy_document_matches_the_run_metadata_schema(self):
        from jsonschema import Draft202012Validator
        schema = json.loads((ROOT / "schemas/benchmark-run.schema.json").read_text())
        _, document = calibration_metadata()
        validator = Draft202012Validator(
            {"$defs": schema["$defs"], "$ref": "#/$defs/gpu_memory_calibration"})
        self.assertEqual(list(validator.iter_errors(document)), [])
        self.assertEqual(document["traffic_bytes_per_copy"], 2 * BUFFER_BYTES)
        self.assertNotIn("limit_bytes_per_second", document)


class Publication(unittest.TestCase):
    def suite(self):
        return {"suite_id": "gpu/structural", "cases": [{"id": CASE}],
                "backends": ["pytorch-cuda"],
                "description": "what this shape class means"}

    def render(self, row, metadata):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "arm.jsonl"
            path.write_text(json.dumps(row) + "\n")
            return formatter.render(self.suite(), [path], metadata, "test")

    def test_an_above_copy_rate_row_is_published_with_its_ratio(self):
        metadata, _ = calibration_metadata()
        row = raw_record(batch_ms=[0.000283])
        decision = guard.apply(row, guard.calibration_of(metadata))
        report = self.render(row, metadata)
        self.assertIn("| 0.283 (0.000) |", report)
        self.assertIn(f"| {BYTES} | {decision['copy_rate_ratio']:.3f} |", report)
        self.assertIn("Measured cells: **1**", report)
        self.assertNotIn("unmeasurable", report)

    def test_the_formatter_publishes_a_row_the_runner_never_recorded(self):
        metadata, _ = calibration_metadata()
        report = self.render(raw_record(batch_ms=[0.021]), metadata)
        self.assertIn("| 21.000 (0.000) |", report)
        self.assertIn("| unavailable |", report)
        self.assertIn("Measured cells: **1**", report)

    def test_a_page_without_a_copy_rate_says_so(self):
        report = self.render(raw_record(batch_ms=[0.021]), {})
        self.assertIn("no device copy rate", report)
        self.assertIn("| unavailable |", report)

    def test_an_unsupported_row_keeps_the_sampling_columns(self):
        row = raw_record(batch_ms=[0.021], status="unsupported")
        row["timing"] = None
        row["reason"] = "no public API"
        report = self.render(row, {})
        header = next(line for line in report.splitlines()
                      if line.startswith("| Case | Backend |"))
        body = next(line for line in report.splitlines()
                    if line.startswith(f"| {CASE} | pytorch-cuda | —"))
        self.assertEqual(body.count(" | "), header.count(" | "))
        self.assertIn("| unavailable | no public API |", body)


if __name__ == "__main__":
    unittest.main()
