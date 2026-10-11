#!/usr/bin/env python3
"""Checks for the cpu/perf_issues and gpu/perf_issues case files and runner."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_perf_issues as suite  # noqa: E402
import generate_perf_issue_cases as gen  # noqa: E402


class PerfIssueSuiteTest(unittest.TestCase):
    def test_generated_files_are_fresh(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/generate_perf_issue_cases.py"), "--check"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_case_names_its_issue_and_reference(self):
        for key in ("cpu", "gpu"):
            with patch.dict(os.environ, {"BENCH_INSTANCE": "", "BENCH_COVERAGE": "full"}):
                _, cases, _, _, expected, selected, _ = suite.plan(key)
            self.assertEqual(expected, selected)
            for case in cases.values():
                self.assertTrue(case["issues"], case["id"])
                self.assertTrue(all(issue in case["intent"] for issue in case["issues"]), case["id"])
                if "reference_for" in case:
                    self.assertIn(case["reference_for"], cases, case["id"])

    def test_quick_excludes_diagnostics(self):
        for key in ("cpu", "gpu"):
            with patch.dict(os.environ, {"BENCH_INSTANCE": "", "BENCH_COVERAGE": "quick"}):
                _, cases, _, _, _, selected, _ = suite.plan(key)
            self.assertTrue(selected)
            self.assertEqual({cases[i]["measurement_scope"] for i in selected}, {"steady_state"})

    def test_issue_table_rows_are_present(self):
        cpu = {i for c in gen.generate_cases() for i in c["issues"]}
        gpu = {i for c in gen.generate_gpu_cases() for i in c["issues"]}
        self.assertTrue({"#1992", "#2003", "#1995", "#1900", "#1615", "#2007", "#1803", "#1990",
                         "#2006", "#1885", "#1975", "#1976", "#2008"} <= cpu)
        self.assertEqual(gpu, {"#2009", "#1887", "#1885", "#2078", "#2079", "#2080", "#2081",
                               "#2082", "#2083", "#1924", "#1839", "#1849", "#1846", "#1852",
                               "#1923", "#1898", "#1873", "#1805", "#1949"})
        einsum = (ROOT / "benchmarks/cpu/einsum.yaml").read_text()
        for name in gen.einsum_instances():
            self.assertIn(name, einsum)

    def test_every_gpu_case_kind_has_a_runner_dispatch_arm(self):
        source = (ROOT / "src/bin/benchmark_gpu_perf_issues.rs").read_text()
        kinds = {c["kind"] for c in gen.generate_gpu_cases()}
        for kind in kinds:
            self.assertIn(f'"{kind}" =>', source, kind)

    def test_state_diagnostic_renders_its_own_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "data/results/nvidia-gpu/gpu/perf_issues/20260101_000000"
            run_dir.mkdir(parents=True)
            (run_dir / "run_t1.yaml").write_text(
                "timestamp: '2026-01-01T00:00:00Z'\ntenferro_rs:\n  commit: deadbeef\n")
            issue_row = {
                "case_id": "contraction_plan_churn_f64_distinct-plans", "issues": ["#1873", "#1805"],
                "kind": "contraction_plan_churn", "arm": "distinct-plans",
                "backend": "tenferro-cuda", "dtype": "f64",
                "measurement_scope": "issue_workflow_diagnostic", "threads": 1,
                "correctness_status": "passed", "max_rel_error": 0.0,
                "params": {"distinct": 72, "n": 64, "kind": "contraction_plan_churn",
                           "arm": "distinct-plans", "dtype": "f64"},
                "samples": [{"sample_index": 0, "elapsed_ns": 88917, "iterations": 1}],
            }
            (run_dir / "samples_t1.jsonl").write_text(json.dumps({
                "case_id": "pool_retention_f64_alloc-free", "issues": ["#2081"],
                "kind": "pool_retention", "arm": "alloc-free", "backend": "tenferro-cuda",
                "dtype": "f64", "measurement_scope": "state_diagnostic", "threads": 1,
                "correctness_status": "passed", "samples": [], "max_rel_error": 0.0,
                "free_bytes_start": 1 << 30, "retained_bytes_after_drop": 1 << 29,
                "probe_status": "failed: out of memory",
                "per_cycle": [{"free_bytes": 1 << 30, "delta_bytes": 0}],
            }) + "\n" + json.dumps(issue_row) + "\n")
            with patch.object(suite, "ROOT", Path(tmp)):
                suite.report(run_dir, "nvidia-gpu", "gpu")
            report = Path(tmp) / "result/nvidia-gpu/gpu/perf_issues.md"
            text = report.read_text()
            self.assertIn("## State diagnostics (no timing)", text)
            self.assertIn("pool_retention_f64_alloc-free", text)
            self.assertIn("retained_bytes_after_drop", text)
            self.assertIn("| Cycle | Free bytes | Delta bytes |", text)
            # The state row never appears as a steady-state operation row.
            self.assertEqual(text.count("pool_retention_f64_alloc-free"), 1)
            self.assertIn("## Issue-workload diagnostics (not steady-state operation rows)", text)
            self.assertEqual(text.count("contraction_plan_churn_f64_distinct-plans"), 1)

    def test_gpu_run_script_refuses_a_busy_device(self):
        script = (ROOT / "scripts/run_gpu_perf_issues.sh").read_text()
        self.assertIn("query-compute-apps", script)
        self.assertIn("BENCHMARK_ALLOW_BUSY_GPU", script)
        # The guard runs again right before measuring, and the post-run list is
        # recorded and checked, not just printed.
        self.assertIn('require_idle_gpu "at measurement time"', script)
        self.assertIn("gpu_compute_apps_at_start", script)
        self.assertIn("gpu_compute_apps_at_end", script)
        self.assertIn("not a clean measurement", script)
        # The recorded snapshots are the enforced ones, and both are enforced.
        self.assertIn("BENCH_GPU_APPS_START", script)
        self.assertIn("BENCH_GPU_APPS_END", script)
        self.assertIn('if [[ -n "$BENCH_GPU_APPS_START" || -n "$GPU_APPS_AT_END" ]]', script)

    def test_failed_binary_is_a_failed_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "fail"
            binary.write_text("#!/bin/sh\necho numerical-failure >&2\nexit 1\n")
            binary.chmod(0o755)
            output = Path(tmp) / "samples_t1.jsonl"
            with patch.dict(os.environ, {"BENCH_INSTANCE": "gemm_f64_mm256_t1", "BENCH_COVERAGE": "quick"}):
                self.assertTrue(suite.collect(binary, output, 1))
            row = json.loads(output.read_text())
            self.assertEqual(row["samples"], [])
            self.assertEqual(row["correctness_status"], "failed")
            status = json.loads((Path(tmp) / "case_status_t1.json").read_text())
            self.assertFalse(status["complete"])


if __name__ == "__main__":
    unittest.main()
