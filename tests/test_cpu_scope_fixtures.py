#!/usr/bin/env python3
"""Logical CPU fixture parity and independent batched-matmul AD references."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_cpu_ops_python as ops


def value(index, seed):
    word = (index * 6364136223846793005 + seed * 1442695040888963407) % (1 << 64)
    return ((word % 1024) - 512) / 512.0


class CpuScopeFixtures(unittest.TestCase):
    def test_native_layout_preserves_rust_logical_indices(self):
        matrix = ops.data((2, 3), 48)
        batched = ops.batched_data((16, 2, 3), 49)
        self.assertTrue(matrix.flags.c_contiguous)
        self.assertTrue(batched.flags.c_contiguous)
        for i in range(2):
            for j in range(3):
                self.assertEqual(matrix[i, j], value(i + 2 * j, 48))
                for z in range(16):
                    self.assertEqual(batched[z, i, j], value(i + 2 * (j + 3 * z), 49))
        self.assertEqual(ops.retention_estimate("2x2xbatch16 (native batch layout)"), 16_384)
        self.assertTrue(ops.suite_enabled("small"))

    def test_both_backends_match_independent_primal_and_two_gradients(self):
        import torch
        import jax
        import jax.numpy as jnp

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        jax.config.update("jax_enable_x64", True)
        for n, batch in [(2, 16), (4, 3), (16, 1)]:
            with self.subTest(n=n, batch=batch):
                a = ops.batched_data((batch, n, n), 48)
                b = ops.batched_data((batch, n, n), 49)
                expected = np.zeros_like(a)
                da, db = np.zeros_like(a), np.zeros_like(b)
                for z in range(batch):
                    for i in range(n):
                        for j in range(n):
                            for k in range(n):
                                expected[z, i, j] += a[z, i, k] * b[z, k, j]
                                da[z, i, k] += b[z, k, j]
                                db[z, k, j] += a[z, i, k]
                x = torch.tensor(a, requires_grad=True)
                y = torch.tensor(b, requires_grad=True)
                np.testing.assert_allclose((x @ y).detach().numpy(), expected, atol=1e-11, rtol=0)
                gx, gy, loss = ops.grad_torch_batched_matmul(x, y)
                for actual, reference in [(gx.detach().numpy(), da), (gy.detach().numpy(), db)]:
                    np.testing.assert_allclose(actual, reference, atol=1e-11, rtol=0)
                self.assertAlmostEqual(loss.item(), expected.sum())
                x, y = jnp.asarray(a), jnp.asarray(b)
                np.testing.assert_allclose(np.asarray(x @ y), expected, atol=1e-11, rtol=0)
                loss, (gx, gy) = jax.jit(jax.value_and_grad(
                    lambda x, y: jnp.einsum("bik,bkj->bij", x, y).sum(), argnums=(0, 1)
                ))(x, y)
                for actual, reference in [(gx, da), (gy, db)]:
                    np.testing.assert_allclose(np.asarray(actual), reference, atol=1e-11, rtol=0)
                self.assertAlmostEqual(float(loss), expected.sum())


if __name__ == "__main__":
    unittest.main()
