"""A non-ok summary row must publish why the case did not produce a number.

Pins tenferro-benchmark#291.  The scheduled 2026-10-31 run recorded
``gpu/tensornetwork:tensornetwork_permutation_optimized_f32:tenferro-cuda-trace``
as ``runtime_failed``, but the published ``tensornetwork.summary.json`` carried
no ``reason``: ``rows_from_gpu_records`` read the cause only for ``unsupported``.
The exception
(``run: TypedTensor::host_data: runtime state failure: backend storage cannot be
borrowed as host data; download explicitly first``) then lived only in the run
artifact, which expired a week later.

These tests hold the summary to the rule ``scripts/gpu_routine_selection.py``
already applies: read the cause from ``execution.unsupported_reason`` or
``verification.reason``, and publish something for every non-ok row.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import report_summary  # noqa: E402

# The exact message the tenferro-cuda-trace row produced on 2026-10-31.
HOST_DATA_FAILURE = (
    "run: TypedTensor::host_data: runtime state failure: backend storage cannot "
    "be borrowed as host data; download explicitly first"
)


def record(status, *, reason=None, verification_status="skipped", path="trace-run",
           median_ms=None):
    """A GPU result record shaped like the runners' JSONL."""
    return {
        "schema_version": 1,
        "suite_id": "gpu/tensornetwork",
        "problem_id": "tensornetwork_permutation_optimized_f32",
        "op": "tensor_network_contract",
        "backend": "tenferro-cuda-trace",
        "status": status,
        "timing": {"median_ms": median_ms, "iqr_ms": None, "timed_runs": 0},
        "verification": {"status": verification_status, "reason": reason},
        "execution": {
            "device": "cuda",
            "device_ordinal": 0,
            "execution_path": path,
            "synchronization": "not executed",
            "layout": "col_major",
            "dtype": "f32",
            "notes": None,
            "unsupported_reason": reason,
        },
    }


class NonOkReasonTests(unittest.TestCase):
    def test_the_runtime_failure_that_had_no_reason_now_publishes_its_exception(self):
        [row] = report_summary.rows_from_gpu_records([record("runtime_failed", reason=HOST_DATA_FAILURE)])
        self.assertEqual(row["status"], "runtime_failed")
        self.assertIsNone(row["median"])
        self.assertEqual(row["reason"], HOST_DATA_FAILURE)
        self.assertTrue(row["reason"].strip())

    def test_a_verification_only_reason_in_the_verification_block_is_published(self):
        failed = record("verification_failed", verification_status="failed")
        failed["verification"]["reason"] = "max_rel_error exceeds rtol"
        failed["execution"]["unsupported_reason"] = None
        [row] = report_summary.rows_from_gpu_records([failed])
        self.assertEqual(row["reason"], "max_rel_error exceeds rtol")

    def test_unsupported_and_skipped_rows_keep_their_reason(self):
        rows = report_summary.rows_from_gpu_records([
            record("unsupported", reason="compiled traced execution owns its output tensors"),
            record("skipped", reason="hptt has no prepared-plan API"),
        ])
        self.assertEqual([row["reason"] for row in rows],
                         ["compiled traced execution owns its output tensors",
                          "hptt has no prepared-plan API"])

    def test_a_non_ok_row_without_any_reason_still_says_something(self):
        bare = record("runtime_failed")
        bare["verification"].pop("reason")
        bare["execution"].pop("unsupported_reason")
        [row] = report_summary.rows_from_gpu_records([bare])
        self.assertEqual(row["reason"], "runtime_failed: no reason recorded")

    def test_a_blank_reason_counts_as_no_reason(self):
        blank = record("runtime_failed", reason="   ")
        [row] = report_summary.rows_from_gpu_records([blank])
        self.assertEqual(row["reason"], "runtime_failed: no reason recorded")

    def test_an_unavailable_execution_path_still_explains_an_unsupported_row(self):
        unavailable = record("unsupported", path="backend unavailable: no CUDA device")
        unavailable["verification"].pop("reason")
        unavailable["execution"].pop("unsupported_reason")
        [row] = report_summary.rows_from_gpu_records([unavailable])
        self.assertEqual(row["reason"], "backend unavailable: no CUDA device")

    def test_an_ok_row_gains_no_reason_key(self):
        [row] = report_summary.rows_from_gpu_records([
            record("ok", verification_status="passed", median_ms=53.669)
        ])
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["median"], 53.669)
        self.assertNotIn("reason", row)


if __name__ == "__main__":
    unittest.main()
