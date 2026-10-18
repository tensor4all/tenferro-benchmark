"""Unit tests for the GPU synthetic-sparse cache in benchmark_gpu_python.py.

Guards `scripts/benchmark_gpu_python.py`'s cross-process fixture cache: the
same problem id + seed + shape must reuse the stored matrix, a stale/incorrect
identity must be rejected, and generated vs cached values must be identical.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_gpu_python", ROOT / "scripts" / "benchmark_gpu_python.py"
)
gpu = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gpu
SPEC.loader.exec_module(gpu)


def sparse_problem(problem_id="sparse_cache_test", rows=8, cols=6, nnz=10, seed=0):
    return {
        "id": problem_id,
        "op": "spmv",
        "data": {"generator": "suitesparse", "seed": seed},
        "sparse": {"source": "suitesparse", "storage": "csr",
                   "rows": rows, "cols": cols, "nnz": nnz},
    }


class SparseCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["GPU_BENCH_SPARSE_CACHE_DIR"] = self.tmp.name
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.environ.pop, "GPU_BENCH_SPARSE_CACHE_DIR", None)

    def test_generation_then_reuse_is_identical(self):
        problem = sparse_problem()
        fresh = gpu._make_data_np(problem)
        cache_path = gpu._sparse_cache_path(problem)
        self.assertTrue(cache_path.exists())

        cached = gpu._load_sparse_cache(problem)
        self.assertIsNotNone(cached)
        for key in ("row_idx", "col_idx", "values", "x"):
            self.assertTrue(np.array_equal(fresh[key], cached[key]), key)

    def test_identity_mismatch_is_rejected(self):
        problem = sparse_problem()
        gpu._make_data_np(problem)
        cache_path = gpu._sparse_cache_path(problem)

        with np.load(cache_path) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["rows"] = np.array(999)  # same file name, wrong stored identity
        with open(cache_path, "wb") as handle:
            np.savez(handle, **payload)

        self.assertIsNone(gpu._load_sparse_cache(problem))

    def test_missing_cache_returns_none(self):
        self.assertIsNone(gpu._load_sparse_cache(sparse_problem(nnz=999999999)))


if __name__ == "__main__":
    unittest.main()
