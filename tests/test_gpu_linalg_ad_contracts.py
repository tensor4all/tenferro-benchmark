"""The GPU AD gate must reject wrong gradients even with correct primals."""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gpu_linalg_ad_contracts import cpu_reference, verify_arrays


class NoDownloadWithoutACheck(unittest.TestCase):
    """`measure_only` skips the host copies it exists to skip."""

    def test_measure_only_touches_neither_the_outputs_nor_the_device(self):
        from benchmark_gpu_linalg_ad import verified_arrays

        def boom(*args, **kwargs):
            raise AssertionError("measure_only must not read the outputs")

        # A 4096² VJP gradient is 134,217,728 bytes per probe; the mode compares
        # nothing, so it must not transfer it or even check where it lives.
        self.assertIsNone(verified_arrays([object()], "measure_only", boom, boom))

    def test_a_checked_row_still_places_and_downloads_every_leaf(self):
        from benchmark_gpu_linalg_ad import verified_arrays

        placed = []
        arrays = verified_arrays([1, 2], "cpu_reference", placed.append,
                                 lambda value: value * 10)
        self.assertEqual(arrays, [10, 20])
        self.assertEqual(placed, [1, 2])


class CollectorErrorBoundary(unittest.TestCase):
    """A case the check refuses costs one row, not the whole collector run."""

    SCHEMA = json.loads(
        (Path(__file__).resolve().parents[1] / "schemas/benchmark-result.schema.json").read_text()
    )

    @staticmethod
    def problem(verify):
        return {"id": "case_a", "op": "vjp",
                "linalg_ad": {"loss": "grad_sum_qr", "n": 4, "matrix_seed": 34},
                "verify": dict(verify)}

    def test_a_refused_manifest_becomes_one_stated_row(self):
        # `run_one` is the collector's per-case entry point and nothing upstream
        # catches what it raises, so resolution defects have to be contained
        # here: at 560edec a `nan` tolerance raised straight out of it and ended
        # the run. Both a refused tolerance and a refused mode go through the
        # same boundary, and the row is a valid record either way.
        import jsonschema

        from benchmark_gpu_linalg_ad import run_one

        cases = (
            # A tolerance the check cannot use; the mode itself resolved.
            ({"mode": "cpu_reference", "rtol": float("nan"), "atol": 1e-8}, "cpu_reference"),
            ({"mode": "cpu_reference", "rtol": 1e-5, "atol": float("inf")}, "cpu_reference"),
            # A mode the check cannot resolve; the row may not name one.
            ({"mode": "measure_only"}, None),
            ({"mode": "cpu_fp64", "rtol": 1e-5, "atol": 1e-8}, None),
        )
        for verify, resolved_mode in cases:
            with self.subTest(verify=verify):
                row = run_one("gpu/linalg_jvp_vjp", self.problem(verify), "pytorch-cuda", 0,
                              ts="2026-01-01T00:00:00+00:00", bc=None, tc=None,
                              verify_defaults={"verify": dict(verify)})
                self.assertEqual(row["status"], "runtime_failed")
                reason = row["execution"]["unsupported_reason"]
                self.assertTrue(reason and reason.strip(), reason)
                # No check ran: no verdict, no evidence, and no mode when the
                # mode is itself what could not be resolved.
                self.assertEqual(row["verification"]["status"], "skipped")
                self.assertEqual(row["verification"].get("mode"), resolved_mode)
                self.assertIsNone(row["verification"]["reference_backend"])
                self.assertIsNone(row["verification"]["residual"])
                jsonschema.validate(row, self.SCHEMA)


    def test_a_silent_exception_still_states_a_reason(self):
        # Measured on 78454d8: a MemoryError from the fixtures left
        # verification.reason="" on a measure_only row, and a `not_run` row
        # requires a non-blank reason (minLength 1, pattern \S). The class name
        # is the reason when the exception has no message.
        import jsonschema
        import benchmark_gpu_linalg_ad as collector

        problem = self.problem({"mode": "measure_only", "reason": "no independent reference"})
        with mock.patch.object(collector, "matrix_for_linalg_ad", side_effect=MemoryError()):
            row = collector.run_one("gpu/linalg_jvp_vjp", problem, "pytorch-cuda", 0,
                                    ts="2026-01-01T00:00:00+00:00", bc=None, tc=None,
                                    verify_defaults={"verify": dict(problem["verify"])})
        jsonschema.validate(row, self.SCHEMA)
        self.assertEqual(row["status"], "runtime_failed")
        self.assertEqual(row["verification"]["status"], "not_run")
        self.assertEqual(row["verification"]["mode"], "measure_only")
        self.assertEqual(row["verification"]["reason"], "MemoryError")


class PartialPageTest(unittest.TestCase):
    """A failed arm still leaves a page for the arms that ran."""

    def rows(self):
        ok = dict(suite_id="gpu/linalg_jvp_vjp",
                  problem_id="linalg_ad_large_grad_sum_qr_jvp_f64_256",
                  backend="pytorch-cuda", status="ok",
                  verification=dict(status="passed", reference_backend="fixture",
                                    max_abs_error=0., max_rel_error=0., residual=0.),
                  timing=dict(timing_scope="steady_state_host_api_plus_device_sync",
                              timed_runs=1, median_ms=1., iqr_ms=0.),
                  execution=dict(notes=json.dumps(dict(sampling=dict(
                      operations_per_sample=1, batch_ms=[1.])))))
        failed = dict(suite_id="gpu/linalg_jvp_vjp",
                      problem_id="linalg_ad_large_grad_sum_qr_jvp_f64_256",
                      backend="tenferro-cuda-trace", status="runtime_failed",
                      reason="arm exited 3",
                      execution=dict(unsupported_reason=None))
        return ok, failed

    def test_without_the_flag_a_failed_row_still_refuses_the_page(self):
        import json
        from format_gpu_linalg_ad_results import format_markdown
        ok, failed = self.rows()
        with self.assertRaisesRegex(ValueError, "cannot publish AD status"):
            format_markdown([ok, failed])

    def test_the_flag_publishes_the_measured_cell_and_names_the_failure(self):
        import json
        from format_gpu_linalg_ad_results import format_markdown
        ok, failed = self.rows()
        page = format_markdown([ok, failed], allow_partial=True)
        self.assertIn("Rows with a failure status", page)
        self.assertIn("arm exited 3", page)
        self.assertIn("linalg_ad_large_grad_sum_qr_jvp_f64_256", page)


class AdVerification(unittest.TestCase):
    def test_latency_publication_requires_matching_scope_and_per_call_sync(self):
        import copy
        import json
        from format_gpu_linalg_ad_results import format_markdown
        sampling = dict(operations_per_sample=2, batch_ms=[2.],
                        scope="steady_state_host_api_plus_device_sync",
                        synchronization_policy="after_each_operation")
        row = dict(suite_id="gpu/linalg_ad_latency",
                   problem_id="linalg_ad_small_grad_sum_qr_jvp_f64_4",
                   backend="tenferro-cuda-trace", status="ok",
                   verification=dict(status="passed", reference_backend="fixture",
                                     max_abs_error=0., max_rel_error=0., residual=0.),
                   timing=dict(timing_scope=sampling["scope"], timed_runs=1,
                               median_ms=1., iqr_ms=0.),
                   execution=dict(notes=json.dumps(dict(sampling=sampling))))
        self.assertIn("grad_sum_qr_jvp", format_markdown([row]))
        for change in ("scope", "sampling_scope", "barrier"):
            bad = copy.deepcopy(row)
            if change == "scope":
                bad["timing"]["timing_scope"] = "public_path_latency_including_internal_session_entry"
            else:
                altered = dict(sampling)
                altered["scope" if change == "sampling_scope" else "synchronization_policy"] = "after_batch"
                bad["execution"]["notes"] = json.dumps(dict(sampling=altered))
            with self.assertRaisesRegex(ValueError, "timing boundary mismatch"):
                format_markdown([bad])

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

    def test_a_measure_only_stub_is_the_not_run_pair(self):
        # A runtime failure on a measure_only problem used to write `skipped`
        # without a mode: neither the Rust builder nor the schema contract
        # (#417 review).
        import json
        import jsonschema
        from benchmark_gpu_linalg_ad import stub
        problem = dict(id="control", op="jvp")
        row = stub("gpu/linalg_jvp_vjp", problem, "tenferro-cuda-trace", 0,
                   status="runtime_failed", reason="no GPU", path="control",
                   ts=None, bc=None, tc=None, mode="measure_only")
        self.assertEqual(row["verification"]["status"], "not_run")
        self.assertEqual(row["verification"]["mode"], "measure_only")
        schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/benchmark-result.schema.json").read_text())
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
            self.assertLessEqual(verify_arrays(actual, expected, 1e-6, 1e-8)[2], 1e-6)  # ~1.8e-10 of the budget: summation order, not a formula (numpy 2.0.2)
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


class CollectorJudgesTheFinalDownload(unittest.TestCase):
    """The collector compares the final download, not only the probe.

    #417 review: replacing `after = check(final)` with `after = verification` in
    the collector left the whole suite green, so a corrupt final download still
    published as verified. The device half is substituted (`_torch_framework`),
    everything else — mode and tolerance resolution, the CPU reference, the
    probe comparison, the merge — is the collector's own code.
    """

    PROBLEM = {"id": "case_a", "op": "vjp",
               "linalg_ad": {"loss": "grad_sum_qr", "n": 4, "matrix_seed": 34},
               "verify": {"mode": "cpu_reference", "rtol": 1e-5, "atol": 1e-8}}

    def collect(self, corrupt_final):
        import benchmark_gpu_linalg_ad as collector
        import benchmark_gpu_public_api
        import gpu_linalg_ad_contracts
        from benchmark_cpu_ops_python import (data, matrix_for_linalg_ad,
                                              tangent_seed, torch_loss_qr)

        ad = self.PROBLEM["linalg_ad"]
        matrix = matrix_for_linalg_ad(ad["loss"], ad["n"], ad["matrix_seed"])
        tangent = data((ad["n"], ad["n"]), tangent_seed(ad["matrix_seed"]))
        expected, _ = gpu_linalg_ad_contracts.cpu_reference(
            torch_loss_qr, matrix, tangent, None, "vjp")
        calls = {"reference": 0, "runner": 0}
        reference = gpu_linalg_ad_contracts.cpu_reference

        def counted_reference(*args, **kwargs):
            calls["reference"] += 1
            return reference(*args, **kwargs)

        def framework(matrix, tangent, rhs, loss_name, phase, device_ordinal):
            def runner():
                calls["runner"] += 1
                # Correct for the probe and for the sampled runs; only the
                # download taken after sampling carries the wrong derivative.
                derivative = expected[1]
                if corrupt_final and calls["runner"] > 2:
                    derivative = derivative * 1.02
                return [expected[0], derivative]
            return {"runner": runner, "ready": lambda values: None,
                    "sync": lambda: None, "check_placement": lambda value: None,
                    "download": lambda value: value, "version": "cpu-fixture",
                    "compile_ms": None}

        def fake_measure(runner, sync, ready, output_bytes, warmups, runs, **_):
            runner()
            sync()
            return {"operations_per_sample": 1, "batch_ms": [1.0]}

        with mock.patch.object(gpu_linalg_ad_contracts, "cpu_reference", counted_reference), \
                mock.patch.object(collector, "_torch_framework", framework), \
                mock.patch.object(benchmark_gpu_public_api, "measure", fake_measure):
            row = collector.run_one("gpu/linalg_jvp_vjp", self.PROBLEM, "pytorch-cuda", 0,
                                    ts="2026-01-01T00:00:00+00:00", bc=None, tc=None)
        return row, calls

    def test_a_correct_row_passes_and_builds_its_reference_once(self):
        row, calls = self.collect(corrupt_final=False)
        self.assertEqual(row["status"], "ok", row.get("execution"))
        self.assertEqual(row["verification"]["status"], "passed")
        self.assertLessEqual(row["verification"]["residual"], 1.0)
        self.assertEqual(calls["reference"], 1)
        self.assertEqual(calls["runner"], 3)  # probe, one sampled batch, final

    def test_a_corrupt_final_download_is_refused(self):
        row, calls = self.collect(corrupt_final=True)
        self.assertEqual(row["status"], "runtime_failed", row["verification"])
        self.assertIn("error/tolerance", row["execution"]["unsupported_reason"])
        self.assertEqual(row["verification"]["status"], "skipped")
        self.assertEqual(calls["reference"], 1)
        self.assertEqual(calls["runner"], 3)  # probe, one sampled batch, final


if __name__ == "__main__":
    unittest.main()
