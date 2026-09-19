import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_gpu_linalg_ad import directional_verification


class DirectionalVerificationTests(unittest.TestCase):
    def test_accepts_matching_derivative_and_records_error(self):
        result = directional_verification(2.0 + 1e-7, 2.0, 1e-5, 1e-8)
        self.assertEqual(result["status"], "passed")
        self.assertIsNotNone(result["reference_backend"])
        self.assertGreater(result["max_abs_error"], 0)

    def test_rejects_wrong_derivative_without_relaxing_tolerance(self):
        result = directional_verification(3.0, 2.0, 1e-5, 1e-8)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["max_abs_error"], 1.0)

    def test_zero_and_nonfinite(self):
        self.assertEqual(directional_verification(0.0, 0.0, 1e-5, 1e-8)["status"], "passed")
        for value in (math.nan, math.inf, -math.inf):
            self.assertEqual(directional_verification(value, 1.0, 1e-5, 1e-8)["status"], "failed")
            self.assertEqual(directional_verification(1.0, value, 1e-5, 1e-8)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
