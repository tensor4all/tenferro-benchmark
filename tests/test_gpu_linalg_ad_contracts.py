"""The GPU AD gate must reject wrong gradients even with correct primals."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gpu_linalg_ad_contracts import cpu_reference, verify_arrays


class AdVerification(unittest.TestCase):
    def test_verified_timing_unsupported_rows_cannot_carry_operation_timings(self):
        import json
        import jsonschema
        from benchmark_gpu_linalg_ad import stub
        problem = dict(id="control", op="jvp")
        row = stub("gpu/linalg_jvp_vjp", problem, "tenferro-cuda-trace", 0,
                   status="unsupported", reason="cannot borrow entered session", path="control", ts=None, bc=None, tc=None)
        row["verification"].update(status="passed", reference_backend="independent CPU control",
                                   max_abs_error=0., max_rel_error=0., residual=0.)
        schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/benchmark-result.schema.json").read_text())
        jsonschema.validate(row, schema)
        row["timing"]["median_ms"] = 1.
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(row, schema)

    def test_rust_unsupported_prose_does_not_count_as_verified_outputs(self):
        from verify_gpu_linalg_ad_rust import validate
        row = dict(status="unsupported", execution=dict(notes="unsupported scalar broadcast"),
                   verification=dict(status="skipped"))
        self.assertIs(validate(row, {}), row)
        self.assertEqual(row["verification"]["status"], "skipped")

    def test_rust_success_without_numerical_artifact_is_rejected(self):
        from verify_gpu_linalg_ad_rust import validate
        for status in ("ok", "verification_only"):
            with self.assertRaises(ValueError):
                validate(dict(status=status, execution=dict(notes="{}")), {})

    def test_quadratic_jvp_and_vjp_match_independent_derivatives(self):
        x = np.array([[1., 2.], [-3., 4.]])
        t = np.array([[.2, -.4], [.1, .3]])
        loss = lambda a: (a*a).sum()
        for phase in ("jvp", "vjp"):
            expected, diagnostic = cpu_reference(loss, x, t, None, phase)
            actual = [np.asarray(np.sum(x*x)), np.asarray(np.sum(2*x*t)) if phase == "jvp" else 2*x]
            self.assertEqual(verify_arrays(actual, expected, 1e-6, 1e-8), (0., 0., 0.))
            self.assertLessEqual(diagnostic["max_error_over_budget"], 1)
            with self.assertRaises(ValueError):
                verify_arrays([actual[0], np.zeros_like(actual[1])], expected, 1e-6, 1e-8)

    def test_shape_dtype_nonfinite_and_missing_derivative_rejected(self):
        expected = [np.asarray(1.), np.array([1., -1.])]
        for values in ([expected[0]], [expected[0], np.array([[1., -1.]])],
                       [expected[0], np.array([1., -1.], dtype=np.float32)],
                       [expected[0], np.array([np.nan, -1.])]):
            with self.assertRaises(ValueError):
                verify_arrays(values, expected, 1e-6, 1e-8)


if __name__ == "__main__":
    unittest.main()
