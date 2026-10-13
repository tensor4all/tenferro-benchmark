"""Numerical and setup-contract checks for the six candidate operations."""
import csv
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_api_candidates as candidates
import benchmark_cpu_public_api_python as pytorch_runner
import benchmark_cpu_public_api_jax as jax_runner


class CandidateTests(unittest.TestCase):
    def setUp(self):
        import jax
        jax.config.update("jax_enable_x64", True)

    def test_compact_qr_reconstruction_and_solve_output_reuse(self):
        import numpy as np
        for factory in (candidates.pytorch_linalg_cases, candidates.jax_linalg_cases):
            rows = factory(shapes=((12, 8),), solve_sizes=(8,))
            self.assertEqual(len(rows), 4)
            for row in rows:
                op = row[-1]
                if op is None:
                    continue
                with self.assertRaisesRegex(RuntimeError, "prepared"):
                    op()
                op.prepare()
                op.validate()
                op.validate()  # repeated solve must overwrite the same storage
        a = candidates.qr_fixture(12, 8, deficient=True)
        self.assertEqual(np.linalg.matrix_rank(a), 6)
        np.testing.assert_array_equal(a[:, 6:], 0)
        # Independent Rust LCG formula, logical column-major index.
        for i, j in ((0, 0), (3, 1), (8, 4)):
            idx = i + 12*j
            value = (((idx*6364136223846793005+1442695040888963407) % (1<<64)) % 2048-1024)/1024
            expected = value*(0.125/12)+(2+j/8 if i == j else 0)
            self.assertEqual(a[i, j], expected)

    def test_selection_broadcast_and_masked_log_softmax(self):
        import numpy as np
        import torch
        import jax.numpy as jnp
        for framework, lazy, array in (("pytorch", pytorch_runner.LazyTensor, torch.tensor),
                                      ("jax", jax_runner.LazyArray, jnp.asarray)):
            condition = lazy(lambda: array([[True], [False]]))
            x = lazy(lambda: array([[1.0, 2.0, 3.0]]))
            y = lazy(lambda: array([[-1.0, -2.0, -3.0]]))
            op = candidates.selection_operation(framework, condition, x, y)
            op.prepare()
            np.testing.assert_array_equal(np.asarray(op()), [[1, 2, 3], [-1, -2, -3]])
        x = jax_runner.LazyArray(lambda: jnp.array([[1., 2., 3.], [-4., 7., 8.]]))
        mask = jax_runner.LazyArray(lambda: jnp.array([[True, False, True], [False, True, True]]))
        op = candidates.masked_log_softmax_operation(x, mask)
        op.prepare()
        out = np.asarray(op())
        np.testing.assert_array_equal(np.isneginf(out), ~np.asarray(mask.get()))
        np.testing.assert_allclose(np.exp(out).sum(axis=1), 1, rtol=1e-12)

    def test_unsupported_contracts_are_rows_without_timing(self):
        args = SimpleNamespace(num_threads=1, runs=1, warmups=0)
        for runner, rows in ((pytorch_runner, candidates.pytorch_linalg_cases()),
                             (jax_runner, candidates.jax_linalg_cases())):
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=runner.FIELDNAMES)
            writer.writeheader()
            with patch.object(runner, "bench", side_effect=AssertionError("must not time unsupported arm")):
                for row in rows:
                    if row[-1] is not None:
                        continue
                    if runner is pytorch_runner:
                        runner.emit_case(writer, args, *row)
                    else:
                        runner.emit_case(writer, args, row)
            emitted = list(csv.DictReader(io.StringIO(stream.getvalue())))
            self.assertTrue(emitted)
            self.assertTrue(all(row["status"] == "unsupported" and not row["median_ms"] for row in emitted))


if __name__ == "__main__":
    unittest.main()
