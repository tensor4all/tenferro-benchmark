"""Numerical fixtures and publication/timing contracts; no GPU required."""
import gc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_gpu_public_api as bench
import format_gpu_public_api_results as formatter
from gpu_public_api_ops import fixtures


class PublicGpuContracts(unittest.TestCase):
    def test_the_cold_path_is_compilation_plus_the_first_call(self):
        # PyTorch arms do not compile: compile_ms is 0 and the cold path is the
        # first call alone. This is what an arm-specific local used to break.
        self.assertAlmostEqual(bench.cold_path_ms(0.0, 100.0, 100.5), 500.0)
        self.assertAlmostEqual(bench.cold_path_ms(None, 100.0, 100.5), 500.0)
        self.assertAlmostEqual(bench.cold_path_ms(12.0, 100.0, 100.5), 512.0)

    def test_reshape_and_stack_preserve_logical_axes_and_dtype(self):
        for dtype in ("f64", "c64", "i32", "i64"):
            case = dict(op="reshape", dtype=dtype, n=12, shape=[3, 4], output_shape=[1, 3, 4])
            inputs, expected = fixtures(case, bench.DTYPES[dtype])
            np.testing.assert_array_equal(expected[0], inputs[0])
            self.assertEqual(expected.dtype, inputs[0].dtype)
            for axis in (0, 2):
                inputs, expected = fixtures(dict(case, op="stack", axis=axis), bench.DTYPES[dtype])
                np.testing.assert_array_equal(np.take(expected, 0, axis=axis), inputs[0])
                np.testing.assert_array_equal(np.take(expected, 1, axis=axis), inputs[1])
                self.assertFalse(np.array_equal(inputs[0], inputs[1]))

    def test_i64_verification_rejects_single_bit_error_above_float_precision(self):
        expected = np.array([2**54 + 1, -(2**54 + 3)], dtype=np.int64)
        case = {"rtol": 5e-4, "atol": 5e-4}
        self.assertEqual(bench.verify(expected.copy(), expected, case), 0)
        wrong = expected.copy()
        wrong[0] += 1
        with self.assertRaisesRegex(ValueError, "exact integer mismatch"):
            bench.verify(wrong, expected, case)

    def test_integer_division_and_remainder_use_truncation(self):
        for dtype, spelling in (("<i4", "i32"), ("<i8", "i64")):
            case = dict(op="div", dtype=spelling, n=128, shape=[8, 16])
            inputs, quotient = fixtures(case, dtype)
            np.testing.assert_array_equal(quotient, np.trunc(inputs[0] / inputs[1]).astype(dtype))
            _, remainder = fixtures(dict(case, op="rem"), dtype)
            np.testing.assert_array_equal(remainder, np.fmod(*inputs))
            self.assertTrue(np.any(remainder != np.mod(*inputs)))
            _, total = fixtures(dict(case, op="reduce_sum"), dtype)
            self.assertEqual(total.dtype, np.dtype(dtype))

    def test_complex_scale_checks_real_and_imaginary_cross_terms(self):
        case = {"op": "scale_complex", "dtype": "c64", "n": 16, "shape": [16],
                "alpha": .5, "beta": .75}
        inputs, output = fixtures(case, "<c16")
        real, imaginary = inputs[0].real, inputs[0].imag
        np.testing.assert_allclose(output.real, .5*real - .75*imaginary)
        np.testing.assert_allclose(output.imag, .75*real + .5*imaginary)
        self.assertEqual(inputs[1].shape, ())

    def test_affine_normalization_parameters_have_independent_effects(self):
        for op in ("layer_norm", "rms_norm"):
            case = {"op": op, "dtype": "f64", "n": 32, "shape": [4, 8]}
            _, baseline = fixtures(case, "<f8")
            for weight, bias in ((True, False), (False, True), (True, True)):
                inputs, output = fixtures(dict(case, normalization_weight=weight,
                                                normalization_bias=bias), "<f8")
                expected = baseline.copy()
                if weight:
                    expected *= inputs[1]
                if bias:
                    expected += inputs[1 + int(weight)]
                np.testing.assert_allclose(output, expected, rtol=1e-14, atol=1e-14)
                self.assertFalse(np.array_equal(output, baseline))

    def test_pad_holes_and_crop_keep_correct_source_coordinates(self):
        case = {"op": "pad", "dtype": "f64", "n": 6, "shape": [2, 3],
                "padding": [(-1, 1, 1), (1, -1, 2)]}
        inputs, output = fixtures(case, "<f8")
        expected = np.zeros((3, 7))
        expected[1, 1] = inputs[0][1, 0]
        expected[1, 4] = inputs[0][1, 1]
        np.testing.assert_array_equal(output, expected)

    def test_completion_tokens_survive_until_clock_stop(self):
        stopped, violations = [True], []
        class Token:
            def __del__(self):
                if not stopped[0]:
                    violations.append("completion token released inside clock")
        tick = [0]
        def clock():
            stopped[0] = not stopped[0]
            tick[0] += 1_000_000
            return tick[0]
        with patch.object(bench.time, "perf_counter_ns", clock):
            bench.measure(lambda: object(), lambda: None,
                          lambda outputs: [Token() for _ in outputs],
                          256 << 20, 0, 3)
        gc.collect()
        self.assertEqual(violations, [])

    def test_per_operation_completion_precedes_next_dispatch(self):
        events, live, tick = [], None, [0]

        class Output:
            def __del__(self):
                events.append("drop")

        def operation():
            events.append("run")
            return Output()

        def ready(output):
            self.assertIsInstance(output, Output)
            events.append("ready")
            return Output()  # completion tokens must survive the stopped clock too

        def clock():
            tick[0] += 1
            events.append("start" if tick[0] % 2 else "stop")
            # Keep calibration below target; the memory budget caps it at two.
            return tick[0] * 1_000_000

        with patch.object(bench.time, "perf_counter_ns", clock):
            sample = bench.measure(operation, lambda: events.append("sync"), ready,
                                   128 << 20, 0, 2, synchronize_each_operation=True)
        self.assertEqual(sample["operations_per_sample"], 2)
        self.assertEqual(sample["synchronization_policy"], "after_each_operation")
        for event in events:
            if event == "start":
                live = []
            elif event == "stop":
                self.assertTrue(live)
                self.assertEqual(live, ["run", "ready"] * (len(live) // 2))
                live = None
            elif live is not None:
                live.append(event)

    def test_unqualified_ref_adapter_is_rejected_before_any_timing(self):
        def operation():
            self.fail("numerical-only Ref operation executed by sampler")
        operation.verification_only = True
        with patch.object(bench.time, "perf_counter_ns", side_effect=AssertionError("timer entered")):
            with self.assertRaisesRegex(ValueError, "synchronization"):
                bench.measure(operation, lambda: None, lambda outputs: None, 16, 0, 1)

    def test_excluded_verified_rows_cannot_hide_failed_validation_or_timings(self):
        suite = {"suite_id": "gpu/lstsq", "cases": [{"id": "one"}], "backends": ["a"],
                 "description": "Small-shape view of the public linalg surface."}
        row = dict(case_id="one", backend="a", status="unsupported", reason="internal setup",
                   timing=None, cuda_implementation_verified=True,
                   verification={"max_error_over_tolerance": 0.})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n")
            report = formatter.render(suite, [path], {}, "test")
            self.assertIn("Measured cells: **0**", report)
            self.assertIn("**1** numerically verified but timing-excluded", report)
            for bad in (dict(row, timing={"batch_ms": [1.]}), dict(row, reason=""),
                        dict(row, verification={"max_error_over_tolerance": 1.1})):
                path.write_text(json.dumps(bad) + "\n")
                with self.assertRaises(ValueError):
                    formatter.load_rows(suite, [path])

    def test_strided_rectangular_fixture_preserves_logical_axes(self):
        case = dict(op="reduce_sum_strided_read", read_op="reduce_sum",
                    n=1024, shape=[16, 64])
        inputs, expected = fixtures(case, "<f8")
        self.assertEqual(inputs[0].shape, (64, 16))
        self.assertEqual(expected.shape, (16,))
        np.testing.assert_allclose(inputs[0].T.sum(axis=1), expected,
                                   rtol=1e-14, atol=1e-14)
        case.update(op="conj_strided_read", read_op="conj")
        inputs, expected = fixtures(case, "<c16")
        self.assertTrue(np.any(inputs[0].imag != 0))
        np.testing.assert_array_equal(inputs[0].T.conj(), expected)

    def test_device_validation_rejects_fallback_in_any_output_leaf(self):
        def torch_value(kind, index):
            return SimpleNamespace(device=SimpleNamespace(type=kind, index=index))
        good = torch_value("cuda", 0)
        bench.verify_device((good, good), "pytorch-cuda", 0)
        for bad in (torch_value("cpu", None), torch_value("cuda", 1)):
            with self.assertRaisesRegex(RuntimeError, "fallback or wrong"):
                bench.verify_device((good, bad), "pytorch-cuda", 0)
        def jax_value(devices):
            return SimpleNamespace(devices=lambda: devices)
        bench.verify_device((jax_value({"gpu0"}),), "jax-cuda-jit", "gpu0")
        for devices in ({"cpu"}, {"gpu1"}, {"gpu0", "gpu1"}):
            with self.assertRaisesRegex(RuntimeError, "fallback or wrong"):
                bench.verify_device((jax_value({"gpu0"}), jax_value(devices)), "jax-cuda-jit", "gpu0")

    def test_comparison_fixtures_include_equality_and_both_orderings(self):
        import gpu_public_api_ops as operations
        for direction in ("eq", "lt", "le", "gt", "ge"):
            case = dict(op="compare", n=1024, shape=[1024], compare_direction=direction)
            inputs, expected = operations.fixtures(case, "<f4")
            self.assertTrue(np.any(inputs[0] == inputs[1]))
            self.assertTrue(np.any(inputs[0] < inputs[1]))
            self.assertTrue(np.any(inputs[0] > inputs[1]))
            self.assertTrue(np.any(expected))
            self.assertTrue(np.any(~expected))

    def test_full_reduction_references_are_scalars_and_preserve_dtype(self):
        for op in ("sum", "prod", "max", "min"):
            case = dict(op="reduce_" + op, n=1024, shape=[1024], full_reduction=True)
            inputs, expected = fixtures(case, "<f8")
            self.assertEqual(expected.shape, ())
            self.assertEqual(expected.dtype, np.dtype("float64"))
            self.assertEqual(expected, getattr(np, op)(inputs[0]))

    def test_large_reduction_policy_still_rejects_zero_or_percent_product_error(self):
        suite = {"cases": [dict(id=op, op=op, dtype="f32", n=1048576,
                                shape=[1048576], full_reduction=True)
                           for op in ("reduce_sum", "reduce_prod", "reduce_mean")]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixtures"
            bench.make_fixtures(root, suite)
            for case in json.loads((root / "manifest.json").read_text()):
                expected = bench.load_array(root, case["outputs"][0])
                self.assertIn("tolerance_policy", case)
                with self.assertRaises(ValueError):
                    bench.verify(np.zeros_like(expected), expected, case)
                if case["op"] == "reduce_prod":
                    with self.assertRaises(ValueError):
                        bench.verify(expected * np.float32(1.01), expected, case)

    def test_all_fixtures_have_finite_data_and_matching_binary_layout(self):
        for path in (ROOT / "data/instances").glob("gpu_*.json"):
            if path.stem not in {"gpu_nn", "gpu_isolated_ops", "gpu_structural", "gpu_linalg_public_api", "gpu_reuse_contraction"}:
                continue
            for case in json.loads(path.read_text())["cases"]:
                if case["n"] > 1024:
                    continue
                arrays, expected = fixtures(case, bench.DTYPES[case["dtype"]])
                self.assertTrue(np.all(np.isfinite(arrays[0])), case["id"])
                from gpu_multi_output_verification import leaves
                self.assertTrue(all(not np.any(np.isnan(v)) for v in leaves(expected)), case["id"])
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    for i, a in enumerate([*arrays, *leaves(expected)]):
                        spec = bench.save_array(root, str(i), a)
                        restored = bench.load_array(root, spec)
                        np.testing.assert_array_equal(restored, a)
                        self.assertEqual(restored.shape, a.shape)

    def test_the_linalg_shape_class_split_keeps_every_case_exactly_once(self):
        """One case source, two pages: the include lists must partition it."""
        import yaml

        source = json.loads((ROOT / "data/instances/gpu_linalg_public_api.json").read_text())
        shapes = {case["id"]: case["shape"] for case in source["cases"]}
        pages = {}
        for regime in ("throughput", "latency"):
            declaration = yaml.safe_load(
                (ROOT / f"benchmarks/gpu/linalg_public_api_{regime}.yaml").read_text())
            self.assertEqual(declaration["suite_id"], f"gpu/linalg_public_api_{regime}")
            self.assertEqual(declaration["defaults"]["regime"], regime)
            self.assertEqual(declaration["problems"]["source"],
                             "data/instances/gpu_linalg_public_api.json")
            pages[regime] = set(declaration["problems"]["include"])
        self.assertFalse(pages["throughput"] & pages["latency"], "a case is on both pages")
        self.assertEqual(pages["throughput"] | pages["latency"], set(shapes), "a case is on neither page")
        self.assertTrue(all(max(shapes[case]) >= 256 for case in pages["throughput"]))
        self.assertTrue(all(max(shapes[case]) <= 128 for case in pages["latency"]))

    def test_fft_reference_precision_shape_and_roundtrip(self):
        suite = json.loads((ROOT / "data/instances/gpu_fft.json").read_text())
        suite["cases"] = [c for c in suite["cases"] if c["n"] == 1024]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixtures"
            bench.make_fixtures(root, suite)
            for c in json.loads((root / "manifest.json").read_text()):
                x, y = bench.load_array(root, c["inputs"][0]), bench.load_array(root, c["outputs"][0])
                self.assertEqual(y.shape, (513 if c["op"] == "rfft" else 1024,))
                inverse = {"fft": np.fft.ifft, "ifft": np.fft.fft, "rfft": np.fft.irfft, "irfft": np.fft.rfft}[c["op"]]
                np.testing.assert_allclose(inverse(y, n=1024), x, rtol=1e-5, atol=1e-5)

    def test_verification_rejects_nan_wrong_shape_dtype_and_values(self):
        case = {"rtol": 1e-5, "atol": 1e-6}
        expected = np.array([1., -np.inf], dtype=np.float64)
        self.assertEqual(bench.verify(expected.copy(), expected, case), 0)
        for bad in [np.array([np.nan, -np.inf]), np.zeros(2), expected.astype(np.float32), expected.reshape(1, 2)]:
            with self.assertRaises(ValueError):
                bench.verify(bad, expected, case)

    def test_batch_retains_outputs_through_sync_and_clock_stop(self):
        alive, stopped, ready_counts, violations = [], [True], [], []
        class Output:
            def __init__(self):
                alive.append(1)
            def __del__(self):
                if not stopped[0]:
                    violations.append("output destroyed inside timer")
                alive.pop()
        tick = [0]
        def clock():
            stopped[0] = not stopped[0]
            tick[0] += 1000
            return tick[0]
        def ready(outputs):
            self.assertEqual(len(alive), len(outputs))
            ready_counts.append(len(outputs))
        with patch.object(bench.time, "perf_counter_ns", clock):
            result = bench.measure(Output, lambda: None, ready, 128 << 20, 0, 3)
        gc.collect()
        self.assertEqual(result["operations_per_sample"], 2)
        self.assertEqual(ready_counts[:3], [1, 1, 2])
        self.assertEqual(ready_counts[-3:], [2, 2, 2])
        self.assertEqual(alive, [])
        self.assertEqual(violations, [])

    def test_accumulation_restores_destination_before_every_batch(self):
        destination, prepared, stopped = [0.], [], [True]
        tick = [0]
        def prepare(count):
            self.assertTrue(stopped[0])
            destination[:] = [2.] * count
            prepared.append(count)
        def operation(i):
            destination[i] += .5
            self.assertEqual(destination[i], 2.5)
            return destination[i]
        operation.prepare = prepare
        def clock():
            stopped[0] = not stopped[0]
            tick[0] += 1_000_000
            return tick[0]
        with patch.object(bench.time, "perf_counter_ns", clock):
            result = bench.measure(operation, lambda: None, lambda outputs: None,
                                   256 << 20, 0, 3, reuse=True)
        self.assertEqual(prepared, [1] * 5)
        self.assertEqual(len(result["batch_ms"]), 3)

    def test_publication_rejects_missing_duplicate_and_failed_rows(self):
        suite = {"suite_id": "gpu/fft", "cases": [{"id": "one"}], "backends": ["a"]}
        good = {"case_id": "one", "backend": "a", "status": "ok",
                "timing": {"operations_per_sample": 2, "batch_ms": [1., 2., 3.]},
                "verification": {"max_error_over_tolerance": 0.}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            for rows in [[], [good, good], [dict(good, status="runtime_failed", reason="failure")]]:
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                with self.assertRaises(ValueError):
                    formatter.render(suite, [path], {}, "test")

    def test_verification_only_rows_cannot_be_published_as_timings(self):
        suite = {"suite_id": "gpu/fft", "cases": [{"id": "one"}], "backends": ["a"]}
        row = {"case_id": "one", "backend": "a", "status": "verification_only",
               "timing": None, "verification": {"max_error_over_tolerance": 0.}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(len(formatter.load_rows(suite, [path], verify_only=True)), 1)
            with self.assertRaises(ValueError):
                formatter.render(suite, [path], {}, "test")
            row["timing"] = {"operations_per_sample": 1, "batch_ms": [1.]}
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                formatter.load_rows(suite, [path], verify_only=True)


class OverheadBudgetDiagnostic(unittest.TestCase):
    """The budget labels the untimed split; it never decides anything."""

    def test_an_over_budget_case_records_a_warning(self):
        # The 20 % default against the measured 21.6 s wall clock of a 50 ms case:
        # the case is far outside it, and that must be visible in its row.
        warning = bench.overhead_warning("cholesky_f64_32x32", 50.0, 21_600.0, 0.2)
        self.assertTrue(warning.startswith("cholesky_f64_32x32 untimed"), warning)
        self.assertIn("43100%", warning)
        self.assertIn("budget 20%", warning)

    def test_a_case_inside_the_budget_records_no_warning(self):
        self.assertIsNone(bench.overhead_warning("case", 50.0, 55.0, 0.2))

    def test_a_case_that_sampled_nothing_records_no_warning(self):
        self.assertIsNone(bench.overhead_warning("case", None, 55.0, 0.2))
        self.assertIsNone(bench.overhead_warning("case", 0.0, 55.0, 0.2))

    def test_the_published_report_carries_the_recorded_warning(self):
        suite = {"suite_id": "gpu/structural", "cases": [{"id": "one"}], "backends": ["a"],
                 "description": "what this shape class means"}
        overhead = {"timed_ms": 50.0, "case_wall_ms": 21_600.0, "untimed_ms": 21_550.0,
                    "budget_ratio": 0.2, "over_budget": True,
                    "warning": "one untimed 21550.0 ms is 43100% of the 50.0 ms timed region (budget 20%)"}
        row = {"case_id": "one", "backend": "a", "status": "ok", "overhead": overhead,
               "timing": {"operations_per_sample": 1, "batch_ms": [0.05]},
               "verification": {"max_error_over_tolerance": 0.0}}
        quiet = {"case_id": "one", "backend": "a", "status": "ok",
                 "overhead": {**overhead, "over_budget": False, "warning": None},
                 "timing": {"operations_per_sample": 1, "batch_ms": [0.05]},
                 "verification": {"max_error_over_tolerance": 0.0}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            for candidate, expected in ((row, True), (quiet, False)):
                path.write_text(json.dumps(candidate) + "\n")
                # The section is appended to the page the CLI writes, not by render(),
                # so this mirrors main(): render, then the recorded warnings.
                rows = formatter.load_rows(suite, [path])
                report = formatter.render(suite, [path], {}, "test")
                report += formatter.overhead_warning_section(rows)
                self.assertEqual("## Untimed work over the recorded budget" in report, expected)
                self.assertIn("| one | 50.000 (0.000) |", report)
                if expected:
                    self.assertIn("43100%", report)
                    self.assertIn("does not change a cell", report)


class CompileNoteTest(unittest.TestCase):
    """The page says what this run compiled, and which regime produced it."""

    def test_a_cold_run_quotes_its_own_recorded_totals(self):
        note = formatter.compile_note({"case_wall_seconds": 228.4, "compiling_processes": 2,
                                       "modules_compiled": 8, "compile_time_ms": None,
                                       "untimed_wall_seconds": 69.0142})
        # Sourced from this run's own accounting, never from another run's figures.
        self.assertIn("**8 NVRTC modules**", note)
        self.assertIn("2 of its arms", note)
        self.assertIn("started cold", note)
        # The wall is named as a wall, because it is not a compile interval.
        self.assertIn("The untimed wall of the cases that compiled is **69.0 s**", note)
        self.assertIn("228.4 s", note)
        self.assertIn("not a compile time", note)

    def test_a_run_that_did_not_compile_says_so_without_claiming_a_warm_cache(self):
        note = formatter.compile_note({"case_wall_seconds": 45.8, "compiling_processes": 0,
                                       "modules_compiled": 0, "compile_time_ms": None,
                                       "untimed_wall_seconds": 0.0})
        self.assertIn("did not compile", note)
        self.assertIn("no NVRTC module was built", note)
        # 0 modules is a statement about this run, not proof of a warm cache.
        self.assertNotIn("warm**", note)
        self.assertNotIn("because the CUDA artifact cache was", note)
        self.assertIn("does not show that the CUDA artifact cache was warm", note)

    def test_no_page_quotes_another_runs_compile(self):
        # The hardcoded cold reference is gone: a page of any family describes only the run
        # that produced it, so the figures recorded for a different run never appear.
        self.assertFalse(hasattr(formatter, "COLD_REFERENCE"))
        for suite_id in ("gpu/linalg_public_api_throughput", "gpu/linalg_public_api_latency"):
            note = formatter.compile_note({"case_wall_seconds": 5.0, "compiling_processes": 0,
                                           "modules_compiled": 0, "compile_time_ms": None,
                                           "untimed_wall_seconds": 0.0})
            self.assertNotIn("94.1", note)
            self.assertNotIn("66.9", note)
            self.assertNotIn(suite_id, note)

    def test_a_run_without_accounting_adds_no_sentence(self):
        suite = {"suite_id": "gpu/one", "backends": ["b"], "description": "d",
                 "cases": [{"id": "one", "op": "add", "dtype": "f32", "shape": [4]}]}
        row = {"case_id": "one", "op": "add", "backend": "b", "status": "ok",
               "timing": {"operations_per_sample": 1, "batch_ms": [0.05]},
               "verification": {"max_error_over_tolerance": 0.0}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertNotIn("NVRTC", formatter.render(suite, [path], {}, "test"))

class BatchingComparabilityTest(unittest.TestCase):
    """#336: a case whose arms batched different counts is marked, not presented as like for like."""

    def render(self, counts):
        suite = {"suite_id": "gpu/one", "backends": list(counts), "description": "d",
                 "cases": [{"id": "one", "op": "add", "dtype": "f32", "shape": [4]}]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_text("".join(json.dumps({
                "case_id": "one", "op": "add", "backend": backend, "status": "ok",
                "timing": {"operations_per_sample": count, "batch_ms": [0.05 * count]},
                "verification": {"max_error_over_tolerance": 0.0}}) + "\n"
                for backend, count in counts.items()))
            return formatter.render(suite, [path], {}, "test")

    def test_mixed_batching_marks_the_case(self):
        report = self.render({"tenferro-cuda-direct": 16, "pytorch-cuda": 32})
        self.assertIn(f"| one {formatter.BATCHING_MARK} |", report)
        self.assertIn(f"Cases marked {formatter.BATCHING_MARK} (**1** of 1)", report)
        # The value stays per operation: 0.8 ms / 16 = 1.6 ms / 32 = 50 µs.
        self.assertIn("50.000 (0.000) | 50.000 (0.000)", report)

    def test_equal_batching_is_not_marked(self):
        report = self.render({"tenferro-cuda-direct": 32, "pytorch-cuda": 32})
        self.assertIn("| one |", report)
        self.assertIn(f"Cases marked {formatter.BATCHING_MARK} (**0** of 1)", report)


if __name__ == "__main__":
    unittest.main()
