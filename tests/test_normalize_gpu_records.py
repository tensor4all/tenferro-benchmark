"""Folding a family's raw rows into the routine record contract."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from normalize_gpu_records import fold
from scripts.ci.check_gpu_run import validate_records


def family_row(**overrides):
    row = {"schema_version": 1, "suite_id": "gpu/lstsq", "case_id": "lstsq_tall_f64_s256_rhs8",
           "op": "lstsq", "backend": "pytorch-cuda", "status": "ok",
           "timing": {"operations_per_sample": 4, "batch_ms": [0.4, 0.5, 0.45], "warmups": 1},
           "verification": {"max_error_over_tolerance": 1e-3, "reference": "numpy/scipy"}}
    row.update(overrides)
    return row


def permutation_row(**overrides):
    row = {"schema_version": 1, "suite_id": "gpu/permutation", "runner": "python",
           "pattern_id": "reverse_18d_3", "label": "reverse 2d", "backend": "pytorch-cuda",
           "dtype": "f64", "status": "ok", "correctness": "passed", "warmup": 3, "iters": 7,
           "median_ms": 0.0123, "p25_ms": 0.011, "p75_ms": 0.013}
    row.update(overrides)
    return row


class NormalizeTest(unittest.TestCase):
    def fold_and_validate(self, row):
        folded = fold(row)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text(json.dumps(folded) + "\n")
            validate_records(path, folded["suite_id"],
                             {(folded["problem_id"], folded["backend"])})
        return folded

    def test_batched_samples_become_a_per_operation_median(self):
        folded = self.fold_and_validate(family_row())
        # 0.4/4, 0.5/4, 0.45/4 ms per operation.
        self.assertAlmostEqual(folded["timing"]["median_ms"], 0.1125)
        self.assertEqual(folded["timing"]["timed_runs"], 3)
        self.assertEqual(folded["verification"]["status"], "passed")

    def test_single_sample_rows_keep_their_median_and_recorded_iqr(self):
        folded = self.fold_and_validate(permutation_row())
        self.assertAlmostEqual(folded["timing"]["median_ms"], 0.0123)
        self.assertAlmostEqual(folded["timing"]["iqr_ms"], 0.002)
        self.assertEqual(folded["verification"]["status"], "passed")

    def test_unverified_and_failed_rows_never_look_measured(self):
        failed = fold(permutation_row(correctness="mismatch"))
        self.assertEqual(failed["verification"]["status"], "failed")
        # The publication gate refuses such a row rather than publishing it.
        with self.assertRaisesRegex(ValueError, "verification did not pass"):
            self.fold_and_validate(permutation_row(correctness="mismatch"))
        unsupported = fold(family_row(status="unsupported", reason="output reuse unsupported",
                                      verification={"max_error_over_tolerance": 0.5}))
        self.assertEqual(unsupported["status"], "unsupported")
        self.assertEqual(unsupported["execution"]["unsupported_reason"], "output reuse unsupported")

    def test_a_row_without_a_case_identity_is_rejected(self):
        row = family_row()
        del row["case_id"]
        with self.assertRaisesRegex(ValueError, "without a case identity"):
            fold(row)


if __name__ == "__main__":
    unittest.main()
