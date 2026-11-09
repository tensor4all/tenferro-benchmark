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
    def test_prepared_plan_fixtures_preserve_storage_and_accumulation(self):
        for dtype in (np.float64,np.complex128):
            for op in ("einsum_plan_owned","einsum_plan_read","einsum_plan_strided_read",
                       "einsum_plan_into","einsum_plan_read_into","einsum_plan_read_into_accum",
                       "einsum_plan_strided_read_into","einsum_plan_strided_read_into_accum"):
                inputs,expected = fixtures(dict(op=op,n=8,shape=[4,2],rhs_shape=[2,7]),dtype)
                a,b = inputs[:2]
                if "strided" in op:
                    self.assertEqual(a.shape,(2,4))
                    self.assertEqual(b.shape,(7,2))
                    a,b = a.T,b.T
                oracle = a @ b
                if "_accum" in op:
                    self.assertEqual(inputs[2].shape,(4,7))
                    oracle = .5*oracle+inputs[2]
                    if dtype==np.complex128:
                        self.assertTrue(np.any(inputs[2].imag != 0))
                self.assertEqual(expected.shape,(4,7))
                np.testing.assert_allclose(expected,oracle,rtol=1e-12,atol=1e-12)

    def test_general_dot_batch_order_and_multiple_axes(self):
        from gpu_multi_output_verification import normalized_outputs
        case = dict(op="dot_general_axes", variant="batch", n=60,
                    shape=[4,3,5], rhs_shape=[5,6,3], equation="mbk,knb->mnb")
        inputs, expected = fixtures(case, np.complex128)
        self.assertEqual(expected.shape, (4,6,3))
        for batch in range(3):
            np.testing.assert_allclose(inputs[0][:,batch,:] @ inputs[1][:,:,batch], expected[:,:,batch])
        native = expected.transpose(2,0,1)
        for backend in ("pytorch-cuda","jax-cuda","jax-cuda-jit"):
            logical = normalized_outputs([native], case, backend)[0]
            self.assertTrue(np.shares_memory(logical,native))
            np.testing.assert_array_equal(logical,expected)
        case.update(variant="multiple", n=24, shape=[4,2,3], rhs_shape=[3,6,2], equation="mkl,lnk->mn")
        inputs, expected = fixtures(case,np.float64)
        for i in range(4):
            for j in range(6):
                self.assertAlmostEqual(expected[i,j],sum(inputs[0][i,k,l]*inputs[1][l,j,k] for k in range(2) for l in range(3)))

    def test_conjugation_flags_have_distinct_complex_results(self):
        case = dict(op="dot_general_conj", n=1024, shape=[1024], rhs_shape=[1024], equation="i,i->")
        values = {}
        for left,right in ((False,False),(True,False),(False,True),(True,True)):
            inputs, expected = fixtures(dict(case,lhs_conj=left,rhs_conj=right),np.complex128)
            a,b = inputs
            oracle = sum((x.conjugate() if left else x)*(y.conjugate() if right else y) for x,y in zip(a,b))
            np.testing.assert_allclose(expected,oracle,rtol=1e-12,atol=1e-12)
            values[left,right] = expected
        self.assertGreater(abs(values[True,False].imag),1e-3)
        self.assertGreater(abs(values[False,False].imag),1e-3)
        np.testing.assert_allclose(values[True,False].conj(),values[False,True])
        np.testing.assert_allclose(values[False,False].conj(),values[True,True])

    def test_lstsq_fixture_is_overdetermined_with_nonzero_residual(self):
        for dtype in (np.float64, np.complex128):
            inputs, solution = fixtures(dict(op="lstsq", n=60, shape=[12, 5], rhs_cols=3), dtype)
            a, b = inputs
            self.assertEqual(solution.shape, (5, 3))
            self.assertEqual(solution.dtype, np.dtype(dtype))
            self.assertEqual(np.linalg.matrix_rank(a), 5)
            residual = a @ solution - b
            self.assertGreater(np.linalg.norm(residual), .1)
            np.testing.assert_allclose(a.conj().T @ residual, 0, atol=1e-12)

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

    def test_a_budget_that_overflows_from_finite_components_is_not_a_pass(self):
        # #417 review: both components are finite, but `abs(1.7e308+1.7e308j)` and the
        # scale built from it overflow to inf, so the miss scored 0.0 and the output
        # counted as verified. `verify_outputs` is the collector's path.
        case = {"op": "exp", "rtol": 1e-9, "atol": 1e-9}
        actual = np.array([complex(1.7e308, 0)])
        expected = np.array([complex(1.7e308, 1.7e308)])
        for run in (lambda: verify_outputs([actual], [expected], case, [], verify),
                    lambda: verify(actual, expected, case)):
            with self.assertRaisesRegex(ValueError, "^verification.*not finite"):
                run()
        self.assertTrue(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)))

    def test_the_factorization_branches_use_the_same_budget(self):
        # Second #417 reproduction: `householder_qr` and `svd_full` scored their own
        # budget inline, so the same overflow passed a 0-miss QR (rtol=1.5 on 1.7e308)
        # and accepted an 8.5e307 reconstruction miss at 0.5 of the budget.
        qr = {"op": "householder_qr", "rtol": 1.5, "atol": 0.0}
        for tolerance in (qr, dict(qr, rtol=float("inf"))):
            with self.assertRaisesRegex(ValueError, "^verification tolerance"):
                verify_outputs([np.array([[0.0]]), np.array([0.0])],
                               [np.array([[1.7e308]]), np.array([0.0])],
                               tolerance, [np.array([[1.7e308]])], verify)
        svd = {"op": "svd_full", "rtol": 1.5, "atol": 0.0}
        u, s, vh = np.array([[0.5]]), np.array([1.7e308]), np.array([[1.0]])
        for tolerance in (svd, dict(svd, rtol=float("inf"))):
            with self.assertRaisesRegex(ValueError, "^verification tolerance"):
                verify_outputs([u, s, vh], [u, s, vh], tolerance, [np.array([[1.7e308]])], verify)

    def test_a_tolerance_the_verifier_cannot_express_is_refused(self):
        values = np.array([1.0])
        for tolerance in ({"rtol": float("inf"), "atol": 0.0},
                          {"rtol": 1e-9, "atol": float("nan")},
                          {"rtol": -1e-9, "atol": 0.0}):
            with self.assertRaisesRegex(ValueError, "^verification tolerance must be finite and non-negative"):
                verify(values, np.array([2.0]), tolerance)


    def test_a_nonunitary_factor_is_not_scored_away_by_a_nan(self):
        # #417 review follow-up: with atol = 0 every zero of the identity
        # reference carries an allowance of 0, so the off-diagonal misses were
        # 0/0 = NaN, np.max returned NaN, and max(worst, NaN) dropped the whole
        # orthogonality comparison. A u with u.T @ u = 8 I then scored only its
        # reconstruction (2.2e-07) and was accepted.
        a = np.array([[1.0, 1.0], [1.0, -1.0]])
        u, s, vh = 2 * np.sqrt(2) * np.eye(2), np.full(2, np.sqrt(2)), .25 * a
        expected = list(np.linalg.svd(a, full_matrices=True))
        case = {'op': 'svd_full', 'rtol': 1e-9, 'atol': 0.0}
        np.testing.assert_allclose(u.T @ u, 8 * np.eye(2))
        with self.assertRaisesRegex(ValueError, '^verification full-SVD'):
            verify_outputs([u, s, vh], expected, case, [a], verify)

    def test_the_orthogonality_guards_refuse_a_budget_they_cannot_express(self):
        # A regression test that only exercises reconstruction lets the inline
        # denominator back in: rtol = atol = 1e308 overflows on the identity
        # reference, and with no guard every orthogonality miss scores 0.0.
        tolerance = {'rtol': 1e308, 'atol': 1e308}
        qr = dict(tolerance, op='householder_qr')
        with self.assertRaisesRegex(ValueError, 'budget is not finite'):
            verify_outputs([np.array([[0.0]]), np.array([-1.0])],
                           [np.array([[0.0]]), np.array([-1.0])],
                           qr, [np.array([[0.0]])], verify)
        svd = dict(tolerance, op='svd_full')
        with self.assertRaisesRegex(ValueError, 'budget is not finite'):
            verify_outputs([np.array([[2.0]]), np.array([0.0]), np.array([[1.0]])],
                           [np.array([[1.0]]), np.array([0.0]), np.array([[1.0]])],
                           svd, [np.array([[0.0]])], verify)


    def test_an_overflowing_gram_matrix_is_not_scored_away_by_a_nan(self):
        # An overflowing product inside u.conj().T @ u is NaN, and NaN survives
        # both np.max and max(): the orthogonality comparison scored nothing and
        # the reconstruction score alone accepted the output.
        a = np.eye(4, 2, dtype=np.complex128)
        u = np.eye(4, dtype=np.complex128)
        u[2:, 2:] = 1e308 * np.array([[1 + 1j, 1j], [-1 + 1j, 1]])
        s, vh = np.ones(2), np.eye(2, dtype=np.complex128)
        case = {'op': 'svd_full', 'rtol': 1e-9, 'atol': 1e-9}
        self.assertTrue(all(np.isfinite(x).all() for x in (a, u, s, vh)))
        expected = list(np.linalg.svd(a, full_matrices=True))
        with self.assertRaisesRegex(ValueError, '^verification full-SVD'):
            verify_outputs([u, s, vh], expected, case, [a], verify)


if __name__ == "__main__":
    unittest.main()
