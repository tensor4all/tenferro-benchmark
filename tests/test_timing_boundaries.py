#!/usr/bin/env python3
"""Check actual setup/synchronization/output lifetime and comparison semantics."""

import ast
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_cpu_ops_python as ops
import format_gpu_permutation_results as permutation
import format_cpu_ops_results as cpu_formatter
import tensornetwork_contract as tnc


class TimingBoundaries(unittest.TestCase):
    def test_setup_and_destruction_outside_clock_even_with_zero_warmups(self):
        state = {
            "timed": False,
            "clock": 0,
            "setup": 0,
            "execute": 0,
            "drop": 0,
            "sync": 0,
        }

        class Output:
            def __del__(self):
                self_check.assertFalse(state["timed"])
                state["drop"] += 1

        self_check = self

        def setup():
            self.assertFalse(state["timed"])
            state["setup"] += 1
            return (state["setup"],)

        def execute(inputs):
            self.assertTrue(state["timed"])
            state["execute"] += 1
            return Output()

        def sync(value):
            if isinstance(value, Output):
                self.assertTrue(state["timed"])
                state["sync"] += 1

        def clock():
            state["timed"] = not state["timed"]
            state["clock"] += 1
            return state["clock"] * 0.001

        with patch.object(ops.time, "perf_counter", clock):
            median, _ = ops.bench(
                ops.PreparedCase(setup, execute), sync, runs=2, warmups=0
            )
        self.assertAlmostEqual(median, 1.0)
        self.assertEqual(
            [state[k] for k in ("setup", "execute", "drop", "sync")], [3, 3, 3, 3]
        )

    def test_all_cpu_ops_fixture_callbacks_are_prepared(self):
        tree = ast.parse((ROOT / "scripts/benchmark_cpu_ops_python.py").read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "emit_row"
            ):
                for arg in node.args:
                    if isinstance(arg, ast.Lambda):
                        self.fail(f"unprepared timed lambda at line {arg.lineno}")
                    if (
                        isinstance(arg, ast.Call)
                        and isinstance(arg.func, ast.Name)
                        and arg.func.id == "PreparedCase"
                    ):
                        operation = ast.unparse(arg.args[1])
                        for fixture in (
                            "data(",
                            "spd(",
                            "well_conditioned(",
                            "tensor(",
                            "array(",
                            "jax.grad(",
                        ):
                            self.assertNotIn(fixture, operation)

    def test_direct_and_eager_are_distinct_even_with_legacy_override(self):
        self.assertIn("tenferro-direct", cpu_formatter.BACKEND_ORDER)
        self.assertNotEqual(
            cpu_formatter.BACKEND_LABELS["tenferro-direct"],
            cpu_formatter.BACKEND_LABELS["tenferro-eager"],
        )

    def test_output_reuse_has_separate_fastest_comparison(self):
        records = [
            dict(
                pattern_id="p",
                label="test",
                dtype="f64",
                status="ok",
                median_ms=ms,
                backend=backend,
                allocates_output=allocate,
            )
            for backend, ms, allocate in [
                ("tenferro-cuda-transpose", 3.0, True),
                ("tenferro-cuda-destination-reuse", 2.0, False),
                ("pytorch-cuda", 1.0, False),
            ]
        ]
        report = "\n".join(permutation.format_table(records))
        allocating, reuse = report.split("## Reusing output")
        self.assertIn("**3.000**", allocating)
        self.assertNotIn("PyTorch", allocating)
        self.assertIn("**1.000**", reuse)
        self.assertNotIn("transpose (ms)", reuse)

    def test_prepared_tensor_network_matches_tree_execution(self):
        import numpy as np

        inputs = [np.arange(6.0).reshape(2, 3), np.arange(12.0).reshape(3, 4)]
        tree = {
            "isleaf": False,
            "args": [
                {"isleaf": True, "tensorindex": 1},
                {"isleaf": True, "tensorindex": 2},
            ],
            "eins": {"ixs": [[1, 2], [2, 3]], "iy": [1, 3]},
        }
        run = tnc.prepare_tree(tree, inputs, "numpy")
        np.testing.assert_array_equal(run(inputs), inputs[0] @ inputs[1])
        np.testing.assert_array_equal(
            run([inputs[0] + 1, inputs[1]]), (inputs[0] + 1) @ inputs[1]
        )


if __name__ == "__main__":
    unittest.main()
