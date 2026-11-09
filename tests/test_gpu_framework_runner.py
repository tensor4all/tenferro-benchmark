"""Reference selection, skipped verification, and debug publication contracts."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_gpu_python as runner
import format_gpu_results as formatter
from gpu_framework_runner import compiler_options_for, require_verification


class LegacyGpuContracts(unittest.TestCase):
    def test_compiler_policy_records_only_options_used_by_the_execution_path(self):
        problem = {"tensor_network": {"jax_compiler_options": {"xla_gpu_autotune_level": 0}}}
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(compiler_options_for(problem, "jax-cuda-tree-jit"), {"xla_gpu_autotune_level": 0})
            self.assertEqual(compiler_options_for(problem, "jax-cuda-eager"), {})
            self.assertEqual(compiler_options_for(problem, "pytorch-cuda"), {})
        with patch.dict("os.environ", {"GPU_BENCH_JAX_COMPILER_OPTIONS": '{"xla_gpu_autotune_level":1}'}):
            self.assertEqual(compiler_options_for(problem, "jax-cuda-tree-jit"), {"xla_gpu_autotune_level": 1})
            self.assertEqual(compiler_options_for(problem, "jax-cuda-eager"), {})

    def test_binary_jax_lowering_matches_batch_contraction_and_output_order(self):
        import jax
        import numpy as np
        import tensornetwork_contract as tnc
        cpu = jax.devices("cpu")[0]
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 20
        b = np.arange(40, dtype=np.float32).reshape(2, 4, 5) / 30
        operation = tnc._jax_binary_contract([[0, 1, 2], [0, 2, 3]], [3, 0, 1], [a.shape, b.shape])
        actual = operation(jax.device_put(a, cpu), jax.device_put(b, cpu))
        np.testing.assert_allclose(np.asarray(actual), np.einsum("bik,bkj->jbi", a, b), rtol=1e-6)
        self.assertIsNone(tnc._jax_binary_contract([[0, 1], [1, 2]], [2], [(2, 3), (3, 4)]))

    def test_native_jax_tree_propagates_permuted_intermediate_axes(self):
        import jax
        import numpy as np
        import tensornetwork_contract as tnc
        tree = {"args": [
            {"args": [{"isleaf": True, "tensorindex": 1}, {"isleaf": True, "tensorindex": 2}],
             "eins": {"ixs": [[0, 1], [1, 2]], "iy": [2, 0]}},
            {"isleaf": True, "tensorindex": 3}],
            "eins": {"ixs": [[2, 0], [0, 3]], "iy": [3, 2]}}
        hosts = [np.arange(6, dtype=np.float32).reshape(2, 3) / 10,
                 np.arange(12, dtype=np.float32).reshape(3, 4) / 20,
                 np.arange(10, dtype=np.float32).reshape(2, 5) / 30]
        expected = tnc.prepare_tree(tree, hosts, "numpy")(hosts)
        inputs = [jax.device_put(value, jax.devices("cpu")[0]) for value in hosts]
        np.testing.assert_allclose(np.asarray(tnc.prepare_tree(tree, inputs, "jax")(inputs)), expected, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(tnc.prepare_tree_jax_kernels(tree, inputs)(inputs)), expected, rtol=1e-6)

    def test_constant_tree_oracle_matches_dense_reference_with_sums_and_free_axes(self):
        import numpy as np
        import tensornetwork_contract as tnc
        for output in [[], [0], [0, 2]]:
            network = {"inputs": [[0, 1], [1, 2]], "tree": {
                "args": [{"isleaf": True, "tensorindex": 1}, {"isleaf": True, "tensorindex": 2}],
                "eins": {"ixs": [[0, 1], [1, 2]], "iy": output}}}
            c = float(np.float32(.8408964152537145))
            arrays = [np.full((3, 3), c, dtype=np.float64) for _ in range(2)]
            expected = tnc.prepare_tree(network["tree"], arrays, "numpy")(arrays)
            np.testing.assert_allclose(tnc.constant_tree_reference(network, 3, c), expected, rtol=1e-14)

    def test_prepared_tree_kernels_preserve_nonconstant_inputs_and_full_tree(self):
        import jax
        import numpy as np
        import tensornetwork_contract as tnc
        tree = {"args": [
            {"args": [{"isleaf": True, "tensorindex": 1}, {"isleaf": True, "tensorindex": 2}],
             "eins": {"ixs": [[0, 1], [1, 2]], "iy": [0, 2]}},
            {"isleaf": True, "tensorindex": 3}],
            "eins": {"ixs": [[0, 2], [2, 3]], "iy": [0, 3]}}
        hosts = [np.arange(9, dtype=np.float32).reshape(3, 3) / 10 + i for i in range(3)]
        cpu = jax.devices("cpu")[0]
        inputs = [jax.device_put(value, cpu) for value in hosts]
        execute = tnc.prepare_tree_jax_kernels(tree, inputs)
        np.testing.assert_allclose(np.asarray(execute(inputs)), hosts[0] @ hosts[1] @ hosts[2], rtol=1e-6)

    def test_supported_standard_cases_select_jax(self):
        import yaml
        for family in ["dense", "einsum", "elementwise", "tensornetwork"]:
            suite = yaml.safe_load((ROOT / "benchmarks/gpu" / (family + ".yaml")).read_text())
            self.assertIn("jax-cuda", suite["backends"])
            for case in suite["problems"]:
                self.assertIn(case["op"], runner._JAX_OPS)
                expected = "jax-cuda"
                self.assertIn(expected, runner._backend_candidates(case, suite["backends"]), case["id"])

    def test_skipped_numerical_check_cannot_count_as_success(self):
        for status in ["skipped", "failed", "not_applicable"]:
            with self.assertRaises(ValueError):
                require_verification((status, None, None))
        self.assertEqual(require_verification(("passed", 0., 0.)), ("passed", 0., 0.))

    def test_a_tolerance_the_legacy_check_cannot_express_is_not_a_pass(self):
        # Measured on c2dca56: both helpers returned ("passed", ...) for both of
        # these — an infinite rtol is an infinite budget, and `atol + rtol *
        # 1.7e308` overflows to one, so `inf <= inf` passed a 1.7e308 miss. Same
        # fail-open the Rust `compare_vectors` carries the guarded version of.
        import numpy as np

        for name in ("_verify_torch", "_verify_jax"):
            check = getattr(runner, name)
            for gpu, cpu, rtol, atol in (
                # Finite errors, infinite tolerance: 999 vs 1 has to fail, so a
                # fix that only rejected non-finite *errors* would not do.
                ([999.], [1.], float("inf"), 0.),
                # Finite errors, finite tolerance, threshold overflows.
                ([1.], [1.7e308], 1.5, 0.),
                # The reviewer's second reproduction verbatim: the error and
                # the threshold both overflow.
                ([-1.7e308], [1.7e308], 1.5, 0.),
                # The overflowing error under a tolerance whose threshold stays
                # finite: only the error check rejects this one.
                ([-1.7e308], [1.7e308], 1e-5, 0.),
                ([999.], [1.], float("nan"), 0.),
                ([1.], [1.], -1e-5, 0.),
            ):
                with self.subTest(check=name, rtol=rtol, gpu=gpu, cpu=cpu):
                    self.assertEqual(
                        check("matmul", np.array(gpu), np.array(cpu), {}, rtol, atol),
                        ("failed", None, None))
            # The ordinary verdicts and their metrics are unchanged.
            self.assertEqual(check("matmul", np.array([1.]), np.array([1.]), {}, 1e-5, 1e-8),
                             ("passed", 0.0, 0.0))
            status, max_abs, _ = check("matmul", np.array([999.]), np.array([1.]), {}, 1e-5, 1e-8)
            self.assertEqual(status, "failed")
            self.assertEqual(max_abs, 998.0)

    def test_a_failed_row_cannot_record_the_tolerance_it_refused(self):
        # Measured on c2dca56: the failure stub copied `rtol=inf` into the record,
        # which `json.dumps` writes as the invalid token `Infinity` and the result
        # schema rejects (a negative one violates its minimum). The reason keeps
        # the value the row refused.
        row = runner._stub(
            "gpu/dense", {"id": "p", "op": "matmul", "layout": {}, "dtype": {}},
            "pytorch-cuda", 0, status="verification_failed",
            reason="verification failed", path="legacy", ts=None, bc=None, tc=None,
            ver_status="failed", rtol=float("inf"), atol=-1e-8)
        json.dumps(row, allow_nan=False)
        import jsonschema
        schema = json.loads((ROOT / "schemas/benchmark-result.schema.json").read_text())
        jsonschema.validate(row, schema)
        self.assertIsNone(row["verification"]["rtol"])
        self.assertIsNone(row["verification"]["atol"])
        self.assertIn("refused tolerance rtol=inf atol=-1e-08", row["verification"]["reason"])
        # A refused tolerance does not take the usable one with it.
        mixed = runner._stub(
            "gpu/dense", {"id": "p", "op": "matmul", "layout": {}, "dtype": {}},
            "pytorch-cuda", 0, status="verification_failed",
            reason="verification failed", path="legacy", ts=None, bc=None, tc=None,
            ver_status="failed", rtol=float("inf"), atol=1e-8)
        self.assertIsNone(mixed["verification"]["rtol"])
        self.assertEqual(mixed["verification"]["atol"], 1e-8)
        self.assertEqual(mixed["verification"]["reason"],
                         "verification failed; refused tolerance rtol=inf")
        # A usable tolerance is recorded untouched.
        kept = runner._stub(
            "gpu/dense", {"id": "p", "op": "matmul", "layout": {}, "dtype": {}},
            "pytorch-cuda", 0, status="verification_failed",
            reason="verification failed", path="legacy", ts=None, bc=None, tc=None,
            ver_status="failed", rtol=1e-5, atol=1e-8)
        self.assertEqual((kept["verification"]["rtol"], kept["verification"]["atol"]),
                         (1e-5, 1e-8))
        self.assertEqual(kept["verification"]["reason"], "verification failed")

    def test_verification_only_records_cannot_be_published(self):
        with self.assertRaises(ValueError):
            formatter.format_markdown([{"status": "verification_only"}])

    def test_duplicate_backend_measurements_are_not_silently_replaced(self):
        row = {"suite_id": "gpu/dense", "problem_id": "one", "backend": "jax-cuda", "status": "ok"}
        with self.assertRaises(ValueError):
            formatter.format_markdown([row, row])

    def test_jax_chain_matches_declared_eight_steps(self):
        import jax
        import numpy as np
        jax.config.update("jax_enable_x64", True)
        cpu = jax.devices("cpu")[0]
        a, b = np.linspace(-.5, .5, 17), np.linspace(.3, -.2, 17)
        fn, args = runner._jax_fn("elementwise_chain", {"A": jax.device_put(a, cpu), "B": jax.device_put(b, cpu)})
        expected = a
        for _ in range(8):
            expected = np.tanh(expected * a + b)
        np.testing.assert_allclose(np.asarray(fn(*args)), expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
