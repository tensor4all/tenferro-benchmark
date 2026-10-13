"""Numerical fixtures and publication/timing contracts; no GPU required."""
import gc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_gpu_public_api as bench
import format_gpu_public_api_results as formatter
from gpu_public_api_ops import fixtures


class PublicGpuContracts(unittest.TestCase):
    def test_all_fixtures_have_finite_data_and_matching_binary_layout(self):
        for path in (ROOT / "data/instances").glob("gpu_*.json"):
            if path.stem not in {"gpu_nn", "gpu_isolated_ops", "gpu_structural", "gpu_linalg_public_api", "gpu_reuse_contraction"}:
                continue
            for case in json.loads(path.read_text())["cases"]:
                if case["n"] > 1024:
                    continue
                arrays, expected = fixtures(case, bench.DTYPES[case["dtype"]])
                self.assertTrue(np.all(np.isfinite(arrays[0])), case["id"])
                self.assertFalse(np.any(np.isnan(expected)), case["id"])
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    for i, a in enumerate([*arrays, expected]):
                        spec = bench.save_array(root, str(i), a)
                        restored = bench.load_array(root, spec)
                        np.testing.assert_array_equal(restored, a)
                        self.assertEqual(restored.shape, a.shape)

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


if __name__ == "__main__":
    unittest.main()
