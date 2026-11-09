"""The cheaper checks must accept a correct row and reject a wrong one.

The point of ``analytic`` and ``finite_difference`` is to reach a GPU-sized case
without a second CPU AD pass.  A cheaper check that only ever passes would be
worse than no check, so every mode here is held to the CPU reference at a size
the CPU can afford: the analytic and finite-difference verdicts must agree with
the ``cpu_reference`` verdict on the same outputs, and both must reject an
output that is wrong by more than the manifest tolerance.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import linalg_ad_verify_modes as modes  # noqa: E402
from benchmark_cpu_ops_python import (  # noqa: E402
    data,
    matrix_for_linalg_ad,
    tangent_seed,
    torch_loss_eigh,
    torch_loss_lu,
    torch_loss_qr,
    torch_loss_solve_wrt_a,
    torch_loss_svd_s,
)
from gpu_linalg_ad_contracts import cpu_reference, verify_arrays  # noqa: E402

LOSSES = {
    "grad_sum_svd_s": torch_loss_svd_s,
    "grad_sum_qr": torch_loss_qr,
    "grad_sum_eigh": torch_loss_eigh,
    "grad_sum_lu": torch_loss_lu,
    "grad_sum_solve": torch_loss_solve_wrt_a,
}
RTOL, ATOL = 1.0e-5, 1.0e-8
SEED = 34


def case(loss, n=64):
    matrix = matrix_for_linalg_ad(loss, n, SEED)
    tangent = data((n, n), tangent_seed(SEED))
    rhs = data((n, 1), 7) if loss == "grad_sum_solve" else None
    return matrix, tangent, rhs


def cpu_outputs(loss, phase, n=64):
    """The CPU AD row this harness has always published, as the truth."""
    matrix, tangent, rhs = case(loss, n)
    expected, _ = cpu_reference(LOSSES[loss], matrix, tangent, rhs, phase)
    return expected, matrix, tangent, rhs


def corrupt(phase, expected, relative=2.0e-2):
    """A derivative that is right in shape and wrong in value."""
    primal, derivative = expected
    if phase == "jvp":
        return [primal, np.asarray(float(derivative) * (1.0 + relative))]
    return [primal, np.asarray(derivative) * (1.0 + relative)]


class ModeContract(unittest.TestCase):
    """Which check a manifest names, and what it may not leave out."""

    def test_default_is_the_cpu_reference(self):
        self.assertEqual(modes.mode_for({"id": "p"}), "cpu_reference")
        self.assertEqual(modes.mode_for({"id": "p", "verify": {}}), "cpu_reference")

    def test_suite_default_fills_in_for_a_problem_that_declares_none(self):
        problem = {"id": "p", "verify": {"rtol": 1.0e-6}}
        defaults = {"verify": {"mode": "analytic", "rtol": RTOL, "atol": ATOL}}
        self.assertEqual(modes.mode_for(problem, defaults), "analytic")

    def test_a_problem_overrides_the_suite_default(self):
        problem = {"id": "p", "verify": {"mode": "finite_difference"}}
        defaults = {"verify": {"mode": "analytic", "rtol": RTOL, "atol": ATOL}}
        self.assertEqual(modes.mode_for(problem, defaults), "finite_difference")

    def test_measure_only_without_a_reason_is_refused(self):
        for block in ({}, {"reason": ""}, {"reason": "   "}):
            problem = {"id": "p", "verify": dict(block, mode="measure_only")}
            with self.assertRaisesRegex(ValueError, "non-empty verify.reason"):
                modes.mode_for(problem)

    def test_measure_only_keeps_its_reason(self):
        problem = {"id": "p", "verify": {"mode": "measure_only", "reason": " no reference fits "}}
        self.assertEqual(modes.mode_for(problem), "measure_only")
        self.assertEqual(modes.measure_only_reason(problem), "no reference fits")
        self.assertEqual(modes.not_run(problem)["status"], "not_run")
        self.assertEqual(modes.not_run(problem)["reason"], "no reference fits")

    def test_unknown_mode_is_refused(self):
        problem = {"id": "p", "verify": {"mode": "cpu_fp64", "rtol": RTOL, "atol": ATOL}}
        with self.assertRaisesRegex(ValueError, "unknown verify.mode"):
            modes.mode_for(problem)


class CheckMustFailClosed(unittest.TestCase):
    """A check that cannot fail would publish a wrong number as verified.

    Two blind spots the review found: `zip` silently dropped outputs the row did
    not produce, and a reference element that is exactly 0 under `atol=0` gave a
    0/0 term that the maximum over the array could swallow.
    """

    def test_a_missing_output_is_rejected(self):
        expected, matrix, tangent, rhs = cpu_outputs("grad_sum_qr", "jvp")
        with self.assertRaisesRegex(ValueError, "arity mismatch"):
            modes.check(
                "cpu_reference", "grad_sum_qr", "jvp", expected[:1],
                matrix, tangent, rhs, RTOL, ATOL, device="cpu",
            )
        expected, matrix, tangent, rhs = cpu_outputs("grad_sum_eigh", "vjp")
        with self.assertRaisesRegex(ValueError, "arity mismatch"):
            modes.check(
                "analytic", "grad_sum_eigh", "vjp", expected[:1],
                matrix, tangent, rhs, RTOL, ATOL, device="cpu",
            )

    def test_an_extra_output_is_rejected(self):
        expected, matrix, tangent, rhs = cpu_outputs("grad_sum_qr", "jvp")
        with self.assertRaisesRegex(ValueError, "arity mismatch"):
            modes.check(
                "cpu_reference", "grad_sum_qr", "jvp", [*expected, expected[-1]],
                matrix, tangent, rhs, RTOL, ATOL, device="cpu",
            )

    def test_a_zero_element_with_no_tolerance_cannot_hide_a_big_error(self):
        correct = np.array([1.0, 0.0])
        self.assertLessEqual(modes._worst(correct, correct, RTOL, 0.0)[2], 1.0)
        wrong = np.array([1.0, 1000.0])
        max_abs, max_rel, worst = modes._worst(wrong, correct, RTOL, 0.0)
        self.assertGreater(worst, 1.0)
        self.assertGreater(max_rel, 1.0)
        self.assertEqual(max_abs, 1000.0)

    def test_an_ordinary_failure_stays_a_finite_number(self):
        # The residual used to be `inf` for every out-of-tolerance element, which
        # `json.dumps` writes as the token `Infinity` -- not JSON. A finite
        # failure has to report its finite ratio.
        max_abs, _, worst = modes._worst(np.array([1.02]), np.array([1.0]), RTOL, ATOL)
        self.assertAlmostEqual(max_abs, 0.02, places=12)
        self.assertAlmostEqual(worst, 0.02 / (RTOL * 1.0 + ATOL), places=3)
        verdict_row = {"max_abs_error": max_abs, "residual": worst}
        self.assertNotIn("Infinity", json.dumps(verdict_row, allow_nan=False))

    def test_a_subnormal_error_fails_a_zero_tolerance(self):
        # The ratio denominator guard used to shrink a subnormal error below 1,
        # so a nonzero error passed when both tolerances were zero.
        _, _, worst = modes._worst(np.array([0.0]), np.array([1e-310]), 0.0, 0.0)
        self.assertGreater(worst, 1.0)
        self.assertLessEqual(modes._worst(np.array([0.0]), np.array([0.0]), 0.0, 0.0)[2], 1.0)

    def test_an_overflowing_difference_stays_a_json_number(self):
        # Two finite inputs of opposite sign overflow the subtraction; the
        # reported max_abs_error must stay a finite JSON number all the same.
        max_abs, _, worst = modes._worst(np.array([-1e308]), np.array([1e308]), 1e-5, 1e-8)
        self.assertTrue(np.isfinite(max_abs) and np.isfinite(worst))
        json.dumps({"max_abs_error": max_abs, "residual": worst}, allow_nan=False)
        self.assertGreater(worst, 1.0)

    def test_an_overflowing_difference_is_not_clamped_into_a_pass(self):
        # Clamping the error before the tolerance comparison turned a 2x miss
        # (tolerance 1.0, true error 2x max) into `passed` (#417 review).
        largest = np.finfo(np.float64).max
        self.assertGreater(
            modes._worst(np.asarray(-largest), np.asarray(largest), 1.0, 0.0)[2], 1.0
        )

    def test_two_overflowing_values_cannot_score_a_pass(self):
        # Both sides overflow: error is inf and `rtol * |reference|` is inf, so
        # `inf/inf` is nan, `max(0.0, nan)` is 0.0, and compare() published a 2x
        # miss as `passed 0.0` (#417 review repro).
        prepared = ([np.asarray(1.7e308)], "ref", {})
        with self.assertRaisesRegex(ValueError, "tolerance overflows"):
            modes.compare(
                "cpu_reference", "jvp", [np.asarray(-1.7e308)], prepared, None, 1.5, 0.0
            )

    def test_a_non_finite_score_is_refused_instead_of_scored(self):
        # The same class, whatever produces it: a nan term must never reach
        # `max(0.0, nan)`, which is 0.0 and reads as a pass.
        import unittest.mock as mock

        with mock.patch.object(modes, "_worst", return_value=(0.0, 0.0, float("nan"))):
            with self.assertRaisesRegex(ValueError, "non-finite score"):
                modes.compare(
                    "cpu_reference", "jvp", [np.asarray(1.0)],
                    ([np.asarray(1.0)], "ref", {}), None, 1e-5, 1e-8,
                )

    def test_a_failed_check_keeps_its_residual_serializable(self):
        expected, matrix, tangent, rhs = cpu_outputs("grad_sum_qr", "jvp")
        with self.assertRaises(modes.VerificationFailure) as caught:
            modes.check(
                "cpu_reference", "grad_sum_qr", "jvp", corrupt("jvp", expected),
                matrix, tangent, rhs, RTOL, ATOL, device="cpu",
            )
        verdict = caught.exception.verdict
        self.assertGreater(verdict["residual"], 1.0)
        json.dumps(verdict, allow_nan=False)  # must not raise


class OneReferencePerRow(unittest.TestCase):
    """The check a row declares is built once, not once per download."""

    def test_one_finite_difference_reference_evaluates_the_loss_three_times(self):
        # `docs/verification-modes.md` states three (the point and both ends of
        # the central difference); #381's body said two, and a Rust row runs the
        # check once per row after #416. Counted, not estimated.
        import unittest.mock as mock

        matrix, tangent, rhs = case("grad_sum_qr", n=8)
        original = modes._loss_fn
        calls = []

        def counting(name):
            fn = original(name)

            def counted(*args, **kwargs):
                calls.append(1)
                return fn(*args, **kwargs)

            return counted

        with mock.patch.object(modes, "_loss_fn", counting):
            modes.reference("finite_difference", "grad_sum_qr", "vjp",
                            matrix, tangent, rhs, device="cpu")
        self.assertEqual(len(calls), 3)

    def test_the_rust_verifier_builds_the_reference_once(self):
        import unittest.mock as mock

        import gpu_linalg_ad_contracts
        from verify_gpu_linalg_ad_rust import validate

        seen = []

        def fake_cpu_reference(loss_fn, matrix, tangent, rhs, phase):
            seen.append(phase)
            reference = [np.asarray(1.0), np.zeros(np.shape(matrix))]
            return reference, {"step": 1.0e-5, "max_error_over_budget": 0.0}

        outputs = {
            phase: [
                {"shape": [], "dtype": "f64", "values_col_major": [1.0]},
                {"shape": [4, 4], "dtype": "f64",
                 "values_col_major": [0.0] * 16},
            ]
            for phase in ("initial", "final")
        }
        row = {
            "status": "ok",
            "problem_id": "linalg_ad_grad_sum_qr_vjp_f64_4",
            "op": "vjp",
            "verification": {"status": "skipped"},
            "execution": {"notes": json.dumps({"verification_outputs": outputs})},
        }
        problem = {
            "id": "linalg_ad_grad_sum_qr_vjp_f64_4",
            "op": "vjp",
            "linalg_ad": {"loss": "grad_sum_qr", "n": 4, "matrix_seed": 34},
            "verify": {"mode": "cpu_reference", "rtol": RTOL, "atol": ATOL},
        }
        with mock.patch.object(gpu_linalg_ad_contracts, "cpu_reference", fake_cpu_reference):
            row = validate(row, problem)
        self.assertEqual(row["verification"]["status"], "passed")
        self.assertEqual(seen, ["vjp"])


class TolerancesComeFromTheManifest(unittest.TestCase):
    """A suite-level tolerance must not be replaced by the runner's fallback."""

    def test_the_problem_wins(self):
        problem = {"verify": {"rtol": 1e-7, "atol": 1e-9}}
        self.assertEqual(
            modes.tolerances(problem, {"verify": {"rtol": 1e-3, "atol": 1e-3}}),
            (1e-7, 1e-9),
        )

    def test_the_suite_fills_in_what_the_problem_omits(self):
        self.assertEqual(
            modes.tolerances({"verify": {}}, {"verify": {"rtol": 1e-7, "atol": 1e-9}}),
            (1e-7, 1e-9),
        )

    def test_the_fallback_is_only_for_a_manifest_that_states_nothing(self):
        self.assertEqual(modes.tolerances({}), (modes.DEFAULT_RTOL, modes.DEFAULT_ATOL))

    def test_a_tolerance_that_is_not_a_finite_non_negative_number_is_refused(self):
        # `validate_suite()` cannot express "not NaN" in its schema, so a
        # `verify.rtol: .nan` reaches the check. It must not reach the score:
        # `nan > 0` is False, an exact match scored 0, and the row published
        # `passed` with `rtol: NaN` in a record no JSON reader accepts.
        for block in ({"rtol": float("nan"), "atol": 1e-8},
                      {"rtol": 1e-5, "atol": float("nan")},
                      {"rtol": float("inf"), "atol": 1e-8},
                      {"rtol": 1e-5, "atol": float("inf")},
                      {"rtol": -1e-5, "atol": 1e-8},
                      {"rtol": 1e-5, "atol": -1e-8}):
            with self.subTest(block=block):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    modes.tolerances({"verify": block})
        # The suite default is checked the same way, and a blank block still
        # takes the fallback.
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            modes.tolerances({}, {"verify": {"rtol": float("nan")}})
        self.assertEqual(modes.tolerances({}), (modes.DEFAULT_RTOL, modes.DEFAULT_ATOL))


class FiniteDifferenceShape(unittest.TestCase):
    """A derivative that is not the row's shape cannot be reduced to one number."""

    def test_a_scalar_derivative_is_refused_instead_of_broadcast(self):
        matrix, tangent, rhs = case("grad_sum_qr", n=8)
        primal, directional, _ = modes.finite_difference_directional(
            "grad_sum_qr", "vjp", matrix, tangent, rhs, device="cpu"
        )
        with self.assertRaisesRegex(ValueError, "does not match the tangent"):
            modes.check(
                "finite_difference", "grad_sum_qr", "vjp",
                [np.asarray(primal), np.asarray(float(directional))],
                matrix, tangent, rhs, RTOL, ATOL, device="cpu",
            )

    def test_an_extra_output_is_refused(self):
        matrix, tangent, rhs = case("grad_sum_qr", n=8)
        primal, directional, _ = modes.finite_difference_directional(
            "grad_sum_qr", "vjp", matrix, tangent, rhs, device="cpu"
        )
        gradient = np.asarray(tangent) * (directional / np.sum(np.asarray(tangent) ** 2))
        with self.assertRaisesRegex(ValueError, "exactly a primal and a derivative"):
            modes.check(
                "finite_difference", "grad_sum_qr", "vjp",
                [np.asarray(primal), gradient, gradient],
                matrix, tangent, rhs, RTOL, ATOL, device="cpu",
            )


class CpuReferenceTolerance(unittest.TestCase):
    """The Python cpu_reference path must not pass what the shared rule rejects."""

    def test_a_wrong_row_with_atol_zero_is_rejected(self):
        reference = np.eye(2)
        wrong = 2.0 * np.eye(2)
        with self.assertRaisesRegex(ValueError, "error/tolerance"):
            verify_arrays([wrong], [reference], RTOL, 0.0)

    def test_a_correct_row_with_atol_zero_passes(self):
        reference = np.eye(2)
        max_abs, max_rel, worst = verify_arrays(
            [reference.copy()], [reference], RTOL, 0.0
        )
        self.assertEqual(max_abs, 0.0)
        self.assertEqual(worst, 0.0)


class AnalyticReference(unittest.TestCase):
    """The closed form must be the same answer as the CPU AD, element by element."""

    def test_closed_form_matches_the_cpu_ad_reference(self):
        for loss in modes.ANALYTIC_LOSSES:
            for phase in ("jvp", "vjp"):
                with self.subTest(loss=loss, phase=phase):
                    expected, matrix, tangent, rhs = cpu_outputs(loss, phase)
                    reference, diagnostic = modes.analytic_reference(
                        loss, phase, matrix, tangent, rhs, device="cpu"
                    )
                    self.assertIn("closed form", diagnostic["reference"])
                    # Same tolerance as the manifest: the cheap reference is not
                    # allowed to be a looser one.
                    verify_arrays(reference, expected, RTOL, ATOL)

    def test_a_correct_row_passes(self):
        for loss in modes.ANALYTIC_LOSSES:
            for phase in ("jvp", "vjp"):
                with self.subTest(loss=loss, phase=phase):
                    expected, matrix, tangent, rhs = cpu_outputs(loss, phase)
                    verdict = modes.check(
                        "analytic", loss, phase, expected, matrix, tangent, rhs,
                        RTOL, ATOL, device="cpu",
                    )
                    self.assertEqual(verdict["status"], "passed")
                    self.assertEqual(verdict["mode"], "analytic")
                    self.assertLessEqual(verdict["residual"], 1.0)
                    self.assertIsNone(verdict["reason"])

    def test_a_wrong_derivative_is_rejected(self):
        for loss in modes.ANALYTIC_LOSSES:
            for phase in ("jvp", "vjp"):
                with self.subTest(loss=loss, phase=phase):
                    expected, matrix, tangent, rhs = cpu_outputs(loss, phase)
                    with self.assertRaises(modes.VerificationFailure) as caught:
                        modes.check(
                            "analytic", loss, phase, corrupt(phase, expected),
                            matrix, tangent, rhs, RTOL, ATOL, device="cpu",
                        )
                    self.assertEqual(caught.exception.verdict["status"], "failed")

    def test_a_loss_without_a_closed_form_is_refused(self):
        matrix, tangent, rhs = case("grad_sum_qr")
        with self.assertRaisesRegex(ValueError, "finite_difference"):
            modes.analytic_reference("grad_sum_qr", "jvp", matrix, tangent, rhs)


class FiniteDifferenceReference(unittest.TestCase):
    """The central difference must agree with the CPU AD derivative, at eps^(1/3)."""

    def _truth_and_step(self, loss, phase, n=64):
        expected, matrix, tangent, rhs = cpu_outputs(loss, phase, n)
        truth = modes.directional_of_actual(phase, expected[1], tangent)
        return expected, matrix, tangent, rhs, truth

    def test_a_correct_row_passes_for_every_loss(self):
        for loss in modes.FINITE_DIFFERENCE_LOSSES:
            for phase in ("jvp", "vjp"):
                with self.subTest(loss=loss, phase=phase):
                    expected, matrix, tangent, rhs = cpu_outputs(loss, phase)
                    verdict = modes.check(
                        "finite_difference", loss, phase, expected, matrix, tangent, rhs,
                        RTOL, ATOL, device="cpu",
                    )
                    self.assertEqual(verdict["status"], "passed")
                    self.assertIsNotNone(verdict["reference_backend"])
                    self.assertIn("finite_difference:", verdict["reason"])
                    # A 100x margin, not a marginal pass: the step is derived,
                    # so the check stays a check at every size rather than
                    # landing just inside the tolerance.
                    self.assertLessEqual(verdict["residual"], 1.0e-2)

    def test_a_wrong_directional_derivative_is_rejected(self):
        for loss in modes.FINITE_DIFFERENCE_LOSSES:
            for phase in ("jvp", "vjp"):
                with self.subTest(loss=loss, phase=phase):
                    expected, matrix, tangent, rhs = cpu_outputs(loss, phase)
                    with self.assertRaises(modes.VerificationFailure):
                        modes.check(
                            "finite_difference", loss, phase, corrupt(phase, expected),
                            matrix, tangent, rhs, RTOL, ATOL, device="cpu",
                        )

    def test_the_derived_step_is_the_best_of_the_swept_ones(self):
        """Why eps^(1/3): smaller is rounding-limited, larger is truncation-limited."""
        swept = (0.1, 1.0, 10.0)
        errors = {}
        for loss in modes.ANALYTIC_LOSSES + modes.FINITE_DIFFERENCE_LOSSES:
            expected, matrix, tangent, rhs, truth = self._truth_and_step(loss, "vjp")
            step = modes.fd_step(loss, matrix, tangent, rhs, device="cpu")
            row = {}
            for factor in swept:
                _, directional, _ = modes.finite_difference_directional(
                    loss, "vjp", matrix, tangent, rhs, device="cpu", step=factor * step
                )
                row[factor] = abs(directional - truth) / (ATOL + RTOL * abs(truth))
            errors[loss] = row
        for loss, row in errors.items():
            # The derived step is inside the budget with two orders of margin.
            self.assertLess(row[1.0], 1.0e-2, f"{loss}: {row}")
        # For the losses whose second derivative is non-zero, both neighbours of
        # the derived step are worse: one by cancellation, one by truncation.
        for loss in ("grad_sum_svd_s", "grad_sum_qr", "grad_sum_lu"):
            row = errors[loss]
            self.assertLess(row[1.0], row[0.1], f"{loss}: {row}")
            self.assertLess(row[1.0], row[10.0], f"{loss}: {row}")

    def test_the_step_scales_with_the_perturbation_not_the_size(self):
        for loss in modes.FINITE_DIFFERENCE_LOSSES:
            steps = []
            for n in (32, 128):
                matrix, tangent, rhs = case(loss, n)
                steps.append(modes.fd_step(loss, matrix, tangent, rhs, device="cpu"))
            with self.subTest(loss=loss):
                # t = eps^(1/3) * ||A|| / ||V||: both norms grow like n, so the
                # step stays the same relative perturbation at every size.
                self.assertLess(max(steps) / min(steps), 4.0, steps)


class RecordShape(unittest.TestCase):
    """One record shape whichever check produced the verdict."""

    def test_every_verifying_mode_writes_the_same_fields(self):
        fields = {"status", "mode", "reference_backend", "max_abs_error",
                  "max_rel_error", "residual", "rtol", "atol", "reason"}
        verdicts = []
        for mode, loss in (
            ("cpu_reference", "grad_sum_qr"),
            ("analytic", "grad_sum_eigh"),
            ("finite_difference", "grad_sum_lu"),
        ):
            expected, matrix, tangent, rhs = cpu_outputs(loss, "vjp")
            verdict = modes.check(mode, loss, "vjp", expected, matrix, tangent, rhs,
                                  RTOL, ATOL, device="cpu")
            self.assertEqual(verdict["mode"], mode)
            self.assertEqual(verdict["status"], "passed")
            verdicts.append(verdict)
        for verdict in verdicts:
            self.assertEqual(fields - set(verdict), set())

    def test_the_failure_carries_the_measured_error(self):
        expected, matrix, tangent, rhs = cpu_outputs("grad_sum_qr", "jvp")
        with self.assertRaises(modes.VerificationFailure) as caught:
            modes.check("cpu_reference", "grad_sum_qr", "jvp", corrupt("jvp", expected),
                        matrix, tangent, rhs, RTOL, ATOL)
        self.assertEqual(caught.exception.verdict["status"], "failed")
        self.assertGreater(caught.exception.verdict["residual"], 1.0)


class VerifierRecord(unittest.TestCase):
    """The Rust-row verifier writes the mode and honours the opt-out."""

    def row(self, expected, problem, mode_verification, status="ok"):
        outputs = {
            phase: [
                {"shape": list(np.shape(value)), "dtype": "f64",
                 "values_col_major": np.asarray(value).ravel(order="F").tolist()}
                for value in expected
            ]
            for phase in ("initial", "final")
        }
        return {
            "status": status,
            "problem_id": problem["id"],
            "op": problem["op"],
            "verification": dict(mode_verification),
            "execution": {"notes": json.dumps({"verification_outputs": outputs})},
        }

    @staticmethod
    def problem(mode):
        return {
            "id": "linalg_ad_large_grad_sum_qr_vjp_f64_64",
            "op": "vjp",
            "linalg_ad": {"suite": "large", "loss": "grad_sum_qr", "n": 64,
                          "matrix_seed": 34},
            "verify": {"mode": mode, "rtol": RTOL, "atol": ATOL},
        }

    def test_an_analytic_row_records_the_mode_and_passes(self):
        from verify_gpu_linalg_ad_rust import validate

        expected, matrix, tangent, rhs = cpu_outputs("grad_sum_eigh", "vjp")
        problem = self.problem("analytic")
        problem["linalg_ad"]["loss"] = "grad_sum_eigh"
        row = validate(self.row(expected, problem, {"status": "skipped", "reason": "pending"}),
                       problem)
        self.assertEqual(row["verification"]["status"], "passed")
        self.assertEqual(row["verification"]["mode"], "analytic")
        self.assertIn("closed form", row["verification"]["reference_backend"])
        self.assertLessEqual(row["verification"]["residual"], 1.0)

    def test_a_finite_difference_row_records_the_step_it_used(self):
        from verify_gpu_linalg_ad_rust import validate

        expected, matrix, tangent, rhs = cpu_outputs("grad_sum_qr", "vjp")
        problem = self.problem("finite_difference")
        row = validate(self.row(expected, problem, {"status": "skipped", "reason": "pending"}),
                       problem)
        self.assertEqual(row["verification"]["status"], "passed")
        self.assertEqual(row["verification"]["mode"], "finite_difference")
        self.assertIn("finite_difference: t=", row["verification"]["reason"])

    def test_a_wrong_row_fails_and_keeps_the_measured_error(self):
        from verify_gpu_linalg_ad_rust import validate

        expected, _, _, _ = cpu_outputs("grad_sum_qr", "vjp")
        wrong = [expected[0], np.asarray(expected[1]) * 1.02]
        problem = self.problem("finite_difference")
        with self.assertRaises(modes.VerificationFailure):
            validate(self.row(wrong, problem, {"status": "skipped", "reason": "pending"}),
                     problem)

    def test_a_measure_only_row_is_left_unchecked_but_must_say_so(self):
        from verify_gpu_linalg_ad_rust import validate

        problem = self.problem("measure_only")
        problem["verify"]["reason"] = "no reference fits"
        row = validate({"status": "ok", "problem_id": problem["id"], "op": "vjp",
                        "verification": {"status": "not_run", "mode": "measure_only",
                                         "reason": "no reference fits"},
                        "execution": {"notes": json.dumps({"verification_outputs": None})}},
                       problem)
        self.assertEqual(row["verification"]["status"], "not_run")

        stale = {"status": "ok", "problem_id": problem["id"], "op": "vjp",
                 "verification": {"status": "skipped"},
                 "execution": {"notes": json.dumps({"verification_outputs": None})}}
        with self.assertRaisesRegex(ValueError, "expected 'not_run'"):
            validate(stale, problem)


if __name__ == "__main__":
    unittest.main()
