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
