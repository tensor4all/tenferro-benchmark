#!/usr/bin/env python3
"""Regression tests for public-API benchmark fixture and report validity."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPD_3_SEED_7 = [
    [1.9408365885416667, -0.010172526041666666, 0.038818359375],
    [-0.010172526041666666, 2.3721516927083335, 0.004475911458333333],
    [0.038818359375, 0.004475911458333333, 2.636800130208333],
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DenseSpdFixtureTests(unittest.TestCase):
    def test_pytorch_spd_fixture_is_dense_symmetric_positive_definite(self) -> None:
        import torch

        module = load_module(
            "benchmark_cpu_public_api_python",
            ROOT / "scripts" / "benchmark_cpu_public_api_python.py",
        )
        matrix = module.spd(3, 7).get()

        off_diagonal = matrix - torch.diag(torch.diagonal(matrix))
        self.assertGreater(torch.count_nonzero(off_diagonal).item(), 0)
        self.assertTrue(torch.equal(matrix, matrix.T))
        self.assertGreater(torch.linalg.eigvalsh(matrix).min().item(), 0.0)
        torch.testing.assert_close(
            matrix,
            torch.tensor(SPD_3_SEED_7, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_jax_spd_fixture_is_dense_symmetric_positive_definite(self) -> None:
        import jax
        import jax.numpy as jnp
        import numpy as np

        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_platform_name", "cpu")
        module = load_module(
            "benchmark_cpu_public_api_jax",
            ROOT / "scripts" / "benchmark_cpu_public_api_jax.py",
        )
        matrix = np.asarray(module.spd(3, 7).get())

        off_diagonal = matrix - np.diag(np.diag(matrix))
        self.assertGreater(np.count_nonzero(off_diagonal), 0)
        np.testing.assert_array_equal(matrix, matrix.T)
        self.assertGreater(float(jnp.linalg.eigvalsh(matrix).min()), 0.0)
        np.testing.assert_array_equal(matrix, np.asarray(SPD_3_SEED_7))


class PlausibilityReportTests(unittest.TestCase):
    def test_report_flags_cells_more_than_ten_times_faster_than_row_reference(
        self,
    ) -> None:
        formatter = load_module(
            "format_cpu_ops_results",
            ROOT / "scripts" / "format_cpu_ops_results.py",
        )
        rows = [
            {
                "suite": "cpu/linalg_uncovered",
                "benchmark": "eigvalsh",
                "dtype": "f64",
                "threads": "1",
                "shape": "512x512",
                "backend": backend,
                "median_ms": median,
                "iqr_ms": "0.01",
                "status": "ok",
            }
            for backend, median in (
                ("tenferro-eager", "7.824"),
                ("tenferro-trace", "0.228"),
                ("pytorch-cpu", "0.282"),
                ("jax-cpu", "8.050"),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            report = formatter.format_table([path])

        self.assertIn("## Cross-Backend Spread Audit", report)
        self.assertIn("## Physical Bound Audit", report)
        self.assertIn("tenferro-trace", report)
        self.assertIn("pytorch-cpu", report)
        self.assertIn("35.3x faster", report)
        self.assertNotIn("tenferro-eager` is", report)

    def test_physical_bound_flags_implausible_row_without_backend_spread(self) -> None:
        formatter = load_module(
            "format_cpu_ops_results_physical",
            ROOT / "scripts" / "format_cpu_ops_results.py",
        )
        rows = [
            {
                "suite": "cpu/linalg_uncovered",
                "benchmark": "eigvalsh",
                "dtype": "f64",
                "threads": "1",
                "shape": "512x512",
                "backend": backend,
                "median_ms": "0.100",
                "iqr_ms": "0.01",
                "status": "ok",
            }
            for backend in (
                "tenferro-eager",
                "tenferro-trace",
                "pytorch-cpu",
                "jax-cpu",
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            report = formatter.format_table([path])

        self.assertIn("## Physical Bound Audit", report)
        self.assertIn("134217728 estimated FLOPs", report)
        self.assertIn("1342.2 GFLOP/s", report)
        self.assertIn("No cross-backend spread above 10x was detected.", report)

    def test_byte_bound_only_applies_to_known_full_touch_operations(self) -> None:
        formatter = load_module(
            "format_cpu_ops_results_bytes",
            ROOT / "scripts" / "format_cpu_ops_results.py",
        )
        rows = [
            {
                "suite": suite,
                "benchmark": benchmark,
                "dtype": "f64",
                "threads": "1",
                "shape": shape,
                "backend": "tenferro-eager",
                "median_ms": "0.001",
                "iqr_ms": "0.0",
                "status": "ok",
            }
            for suite, benchmark, shape in (
                ("cpu/structural_shape", "transpose", "100000000"),
                ("cpu/structural_shape", "extract_diagonal", "8192x8192"),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            report = formatter.format_table([path])

        physical = report.split("## Physical Bound Audit", 1)[1]
        self.assertIn("cpu/structural_shape/transpose", physical)
        self.assertIn("800000000 minimum logical bytes", physical)
        self.assertIn("800000.0 GB/s", physical)
        self.assertNotIn("extract_diagonal", physical)

    def test_physical_bound_covers_public_api_compute_classes(self) -> None:
        formatter = load_module(
            "format_cpu_ops_results_compute_classes",
            ROOT / "scripts" / "format_cpu_ops_results.py",
        )
        workloads = (
            ("cpu/einsum_concrete", "einsum_ij_jk_ik", "f64", "1024x1024"),
            ("cpu/complex", "dot_general", "c64", "640x640"),
            (
                "cpu/linalg_uncovered",
                "triangular_solve",
                "f64",
                "4096x4096,rhs=64",
            ),
            ("cpu/linalg_uncovered", "norm_fro", "f64", "2048x2048"),
        )
        timings = {
            (suite, benchmark, dtype, "1", shape): {
                backend: 0.001
                for backend in (
                    "tenferro-eager",
                    "tenferro-trace",
                    "pytorch-cpu",
                    "jax-cpu",
                )
            }
            for suite, benchmark, dtype, shape in workloads
        }

        findings = "\n".join(formatter.physical_bound_findings(timings))

        for _suite, benchmark, _dtype, _shape in workloads:
            self.assertIn(f"/{benchmark}`", findings)
        self.assertIn("1073741824 estimated FLOPs", findings)
        self.assertIn("262144000 estimated FLOPs", findings)
        self.assertIn("536870912 estimated FLOPs", findings)
        self.assertIn("4194304 estimated FLOPs", findings)

    def test_shape_parser_handles_outputs_rhs_and_concatenation(self) -> None:
        formatter = load_module(
            "format_cpu_ops_results_shape_specs",
            ROOT / "scripts" / "format_cpu_ops_results.py",
        )

        solve = formatter.parse_workload_shape("4096x4096,rhs=64")
        broadcast = formatter.parse_workload_shape("8192x1 -> 8192x4096")
        concatenate = formatter.parse_workload_shape("1048576+1048576")

        self.assertEqual(solve.primary, (4096, 4096))
        self.assertEqual(solve.output, (4096, 4096))
        self.assertEqual(solve.rhs, 64)
        self.assertEqual(broadcast.primary, (8192, 1))
        self.assertEqual(broadcast.output, (8192, 4096))
        self.assertEqual(concatenate.primary, (2_097_152,))
        self.assertEqual(concatenate.output, (2_097_152,))

    def test_public_api_materialization_uses_output_byte_lower_bound(self) -> None:
        formatter = load_module(
            "format_cpu_ops_results_materialization",
            ROOT / "scripts" / "format_cpu_ops_results.py",
        )
        key = (
            "cpu/structural_shape",
            "broadcast_in_dim",
            "f64",
            "1",
            "8192x1 -> 8192x4096",
        )
        timings = {key: {"tenferro-eager": 0.001}}

        findings = "\n".join(formatter.physical_bound_findings(timings))

        self.assertIn("268435456 minimum logical bytes", findings)

    def test_slice_bound_uses_declared_output_extent(self) -> None:
        formatter = load_module(
            "format_cpu_ops_results_slice_output",
            ROOT / "scripts" / "format_cpu_ops_results.py",
        )
        workload = formatter.parse_workload_shape("4194304 -> 2096128")

        bytes_moved = formatter.estimated_minimum_bytes(
            "cpu/indexing_layout",
            "slice",
            workload,
            8,
            0,
        )

        self.assertEqual(bytes_moved, 2_096_128 * 8)

    def test_full_public_api_report_has_models_except_metadata_only_views(self) -> None:
        formatter = load_module(
            "format_cpu_ops_results_full_suite",
            ROOT / "scripts" / "format_cpu_ops_results.py",
        )
        report = ROOT / "result" / "mac-cpu" / "cpu" / "public_api.md"
        workloads: set[tuple[str, str, str, str]] = set()
        for line in report.read_text().splitlines():
            if not line.startswith("| cpu/"):
                continue
            fields = [field.strip() for field in line.strip("|").split("|")]
            suite, benchmark, dtype, _threads, shape = fields[:5]
            workloads.add((suite, benchmark.strip("`"), dtype, shape.strip("`")))

        unmodeled = []
        for suite, benchmark, dtype, shape in workloads:
            workload = formatter.parse_workload_shape(shape)
            dtype_bytes = formatter.DTYPE_BYTES.get(dtype)
            self.assertIsNotNone(workload, (suite, benchmark, shape))
            self.assertIsNotNone(dtype_bytes, (suite, benchmark, dtype))
            flops = formatter.estimated_flops(benchmark, workload)
            bytes_moved = formatter.estimated_minimum_bytes(
                suite,
                benchmark,
                workload,
                dtype_bytes,
                flops,
            )
            if not flops and not bytes_moved:
                unmodeled.append((suite, benchmark))

        self.assertEqual(
            sorted(unmodeled),
            [
                ("cpu/view_metadata", "broadcast_in_dim_view"),
                ("cpu/view_metadata", "reshape_view"),
                ("cpu/view_metadata", "slice_view"),
                ("cpu/view_metadata", "transpose_view"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
