"""Full SVD must validate nullspace outputs without imposing a vector gauge."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gpu_public_api_ops import fixtures
from gpu_multi_output_verification import verify_outputs
from benchmark_gpu_public_api import verify


class MultiOutputVerification(unittest.TestCase):
    def test_lu_all_outputs_follow_pa_equals_lu_and_pivot_parity(self):
        for dtype in (np.float64, np.complex128):
            for shape in ([4, 4], [6, 4], [4, 6]):
                case = dict(op="lu", n=np.prod(shape), shape=shape, rtol=1e-9, atol=1e-9)
                inputs, expected = fixtures(case, dtype)
                p, l, u, parity = expected
                np.testing.assert_allclose(p @ inputs[0], l @ u, rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(p.sum(axis=0), np.ones(shape[0]))
                np.testing.assert_array_equal(p.sum(axis=1), np.ones(shape[0]))
                self.assertEqual(parity.shape, ())
                self.assertEqual(parity.dtype, np.dtype(dtype))
                np.testing.assert_allclose(parity, np.linalg.det(p), atol=1e-12)
                if shape == [4, 4]:
                    self.assertFalse(np.array_equal(p, p.T))
                for index in range(4):
                    bad = [value.copy() for value in expected]
                    bad[index].flat[0] += 1
                    with self.assertRaises(ValueError):
                        verify_outputs(bad, expected, case, inputs, verify)

    def test_state_inputs_and_complement_reference_preserve_contracts(self):
        for dtype in (np.float64, np.complex128):
            case = dict(op="householder_from_factors", n=60, shape=[12, 5])
            inputs, expected = fixtures(case, dtype)
            np.testing.assert_allclose(inputs[1] @ inputs[2], inputs[0], rtol=1e-12, atol=1e-12)
            self.assertEqual(expected[0].shape, (12, 5))
            case["op"] = "householder_append"
            inputs, _ = fixtures(case, dtype)
            np.testing.assert_array_equal(np.concatenate(inputs[1:], axis=1), inputs[0])
            case.update(op="householder_q_columns", column_range=[5, 12])
            inputs, complement = fixtures(case, dtype)
            np.testing.assert_allclose(complement.conj().T @ inputs[0], 0, atol=1e-12)
            np.testing.assert_allclose(complement.conj().T @ complement, np.eye(7), atol=1e-12)

    def test_raw_householder_coefficients_and_jax_layout_are_checked(self):
        from gpu_multi_output_verification import normalized_outputs
        for dtype in (np.float64, np.complex128):
            case = dict(op="householder_qr", n=60, shape=[12, 5], rtol=1e-9, atol=1e-9)
            inputs, expected = fixtures(case, dtype)
            native_jax = expected[0].T, expected[1]
            logical = normalized_outputs(native_jax, case, "jax-cuda")
            self.assertLessEqual(verify_outputs(logical, expected, case, inputs, verify), 1)
            corrupted = [a.copy() for a in expected]
            corrupted[1][0] *= .5
            with self.assertRaises(ValueError):
                verify_outputs(corrupted, expected, case, inputs, verify)

    def test_svd_gauge_changes_pass_but_corrupted_nullspaces_fail(self):
        for dtype in (np.float64, np.complex128):
            for shape in ([12, 5], [5, 12]):
                case = dict(op="svd_full", n=np.prod(shape), shape=shape, rtol=1e-9, atol=1e-9)
                inputs, expected = fixtures(case, dtype)
                u, s, vh = [a.copy() for a in expected]
                phase = np.exp(.3j) if dtype == np.complex128 else -1.
                u[:, 0] *= phase
                vh[0, :] *= np.conj(phase)
                self.assertLessEqual(verify_outputs([u, s, vh], expected, case, inputs, verify), 1)
                if shape[0] > shape[1]:
                    u[:, -1] = 0  # does not affect reconstruction, but violates full U
                else:
                    vh[-1, :] = 0  # likewise corrupt only the full right nullspace
                with self.assertRaises(ValueError):
                    verify_outputs([u, s, vh], expected, case, inputs, verify)

    def test_slogdet_requires_both_sign_and_log_outputs(self):
        case = dict(op="slogdet", n=16, shape=[4, 4], variant="negative", rtol=1e-9, atol=1e-9)
        inputs, expected = fixtures(case, np.float64)
        self.assertEqual(expected[0], -1)
        self.assertEqual(verify_outputs(expected, expected, case, inputs, verify), 0)
        for bad in ([expected[0]], [np.asarray(1.), expected[1]], [expected[0], expected[1]+1]):
            with self.assertRaises(ValueError):
                verify_outputs(bad, expected, case, inputs, verify)


if __name__ == "__main__":
    unittest.main()
