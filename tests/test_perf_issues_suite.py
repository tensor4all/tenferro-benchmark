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
                         "#2006", "#1885"} <= cpu)
        self.assertEqual(gpu, {"#2009", "#1887", "#1885"})
        einsum = (ROOT / "benchmarks/cpu/einsum.yaml").read_text()
        for name in gen.einsum_instances():
            self.assertIn(name, einsum)

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
