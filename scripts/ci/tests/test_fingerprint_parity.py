"""The Rust harness and the fixture writer must compute one fingerprint, not two.

`src/fingerprint.rs` and `scripts/benchmark_gpu_public_api.py` each implement the five
invariants and the seeded weight vector, because the Rust arms consume what the Python
preparation step stores. A silent divergence would turn every diagnostic into a
disagreement, and the record would say the invariants disagree with fixtures that are in
fact correct. Both sides therefore pin the same reference array and the same expected
numbers, and this test is that pin.

The invariants are diagnostics: `verify_outputs` compares every element whatever they say.
"""
import unittest

import numpy as np

from scripts.benchmark_gpu_public_api import (
    FINGERPRINT_SCHEMA_VERSION,
    fingerprint,
    fingerprint_matches,
    seeded_weights,
)

# The Rust side asserts exactly these numbers in `src/fingerprint.rs`.
REFERENCE_REAL = [1.0, -2.5, 0.0, 4.0, -0.5, 3.0]
REFERENCE_IMAGINARY = [0.5, 0.0, -1.0, 2.0, 0.0, -0.25]
REFERENCE = {
    "sum_re": 5.0,
    "sum_im": 1.25,
    "sum_abs": 12.600568588447548,
    "sum_squares": 37.8125,
    "max_abs": 4.47213595499958,
    "dot_seeded_re": 4.233645350156905,
    "dot_seeded_im": 3.3003082955346983,
    "elements": 6,
}
REFERENCE_WEIGHTS = [
    0.7666216164272852,
    -0.13694400590298006,
    -0.9471324568148045,
    0.941763956307657,
    -0.7873066168655751,
    -0.3453484715637485,
    -0.6522642680806343,
    0.543093112663134,
]


class FingerprintParityTests(unittest.TestCase):
    def test_schema_version_matches_the_rust_module(self):
        self.assertEqual(FINGERPRINT_SCHEMA_VERSION, 1)

    def test_seeded_weights_match_the_rust_sequence(self):
        weights = list(seeded_weights(len(REFERENCE_WEIGHTS)))
        self.assertEqual(weights, REFERENCE_WEIGHTS)
        self.assertTrue(all(abs(weight) <= 1.0 for weight in weights))

    def test_reference_invariants_match_the_rust_constants(self):
        values = np.array(REFERENCE_REAL) + 1j * np.array(REFERENCE_IMAGINARY)
        live = fingerprint(values)
        for key, expected in REFERENCE.items():
            self.assertAlmostEqual(live[key], expected, places=12, msg=key)

    def test_a_real_dtype_reports_a_zero_imaginary_sum(self):
        self.assertEqual(fingerprint(np.array([1.0, 2.0]))["sum_im"], 0.0)

    def test_a_permutation_is_caught(self):
        case = {"atol": 1e-9, "rtol": 1e-9}
        expected = fingerprint(np.array([1.0, 2.0, 3.0, 4.0]))
        permuted = fingerprint(np.array([1.0, 4.0, 3.0, 2.0]))
        self.assertFalse(fingerprint_matches(np.array([1.0, 4.0, 3.0, 2.0]), expected, case))
        self.assertNotEqual(permuted["dot_seeded_re"], expected["dot_seeded_re"])

    def test_adjacent_permutations_differ(self):
        # `±1` weights made the first four weights [-1, +1, +1, -1], so both arrays had
        # dot product 0 and equal magnitude invariants: the swap was invisible.
        case = {"atol": 1e-9, "rtol": 1e-9}
        expected = fingerprint(np.array([1.0, 2.0, 3.0, 4.0]))
        actual_values = np.array([1.0, 3.0, 2.0, 4.0])
        actual = fingerprint(actual_values)
        self.assertAlmostEqual(actual["sum_abs"], expected["sum_abs"], places=12)
        self.assertAlmostEqual(actual["sum_squares"], expected["sum_squares"], places=12)
        self.assertAlmostEqual(actual["max_abs"], expected["max_abs"], places=12)
        self.assertGreater(abs(actual["dot_seeded_re"] - expected["dot_seeded_re"]), 1e-3)
        self.assertFalse(fingerprint_matches(actual_values, expected, case))

    def test_roundoff_still_matches(self):
        case = {"atol": 1e-9, "rtol": 1e-9}
        expected = fingerprint(np.array([1.0, -2.0, 3.5]))
        self.assertTrue(fingerprint_matches(np.array([1.0 + 1e-12, -2.0 - 1e-12, 3.5]), expected, case))


if __name__ == "__main__":
    unittest.main()
