"""Unit tests for the gpu/permutation sampled correctness gate.

Guards `scripts/benchmark_gpu_permutation_python.py`'s cheap exact checker:
a correct output passes, while a corrupted entry and a deliberately wrong
permutation (identity instead of the declared `[1, 0]`) must fail.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_gpu_permutation_python",
    ROOT / "scripts" / "benchmark_gpu_permutation_python.py",
)
perm = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = perm
SPEC.loader.exec_module(perm)


class SampledVerifyTest(unittest.TestCase):
    def setUp(self):
        # shape [3, 4], perm [1, 0] -> out_shape [4, 3], all col-major.
        self.pattern = {
            "shape": [3, 4],
            "perm": [1, 0],
            "src_layout": {"kind": "col_major"},
        }
        self.out_shape = [4, 3]
        self.src_data = perm.deterministic_data(12)
        self.src_strides = perm.layout_strides([3, 4], {"kind": "col_major"})
        self.positions = perm.sample_coordinates(self.out_shape)

    def check(self, actual):
        return perm.verify_sampled(
            actual,
            self.src_data,
            self.out_shape,
            self.src_strides,
            self.pattern["perm"],
            self.positions,
        )

    def test_correct_output_passes(self):
        reference = perm.compute_reference(self.pattern, self.src_data)
        self.assertIsNone(self.check(reference))

    def test_corrupted_corner_fails(self):
        reference = perm.compute_reference(self.pattern, self.src_data)
        corrupted = reference.copy()
        corrupted[0] = -12345.0
        error = self.check(corrupted)
        self.assertIsNotNone(error)

    def test_wrong_permutation_fails(self):
        # Identity source mapping over the output shape: same out_shape, but a
        # different value at many coordinates than the declared [1, 0].
        wrong = np.lib.stride_tricks.as_strided(
            self.src_data, shape=tuple(self.out_shape), strides=(8, 24)
        ).copy()
        self.assertFalse(np.array_equal(wrong, perm.compute_reference(self.pattern, self.src_data)))
        self.assertIsNotNone(self.check(wrong))

    def test_sampled_positions_are_deterministic_and_include_corners(self):
        self.assertTrue(np.array_equal(perm.sample_coordinates(self.out_shape), self.positions))
        corners = {tuple(int(v) for v in row) for row in self.positions}
        self.assertIn((0, 0), corners)
        self.assertIn((3, 2), corners)


if __name__ == "__main__":
    unittest.main()
