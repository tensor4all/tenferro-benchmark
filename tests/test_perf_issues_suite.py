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
    def test_metadata_fold_controls_and_references(self):
        cases = {row['id']: row for row in gen.trace_metadata_cases()}
        target = 'trace_metadata_trace-fold_f32_1024x64_k50'
        self.assertEqual(cases[target]['thread_counts'], [1, 4])
        self.assertEqual(cases['trace_metadata_trace-outputs_f32_1024x64_k50']['params']['steps'], 50)
        for arm in ('eager', 'pytorch'):
            ref = cases[f'trace_metadata_{arm}_f32_1024x64_k50']
            self.assertEqual(ref['reference_for'], target)
            self.assertEqual(ref['thread_counts'], [1, 4])
        control = cases['trace_metadata_fixed_neg_f32_8']
        self.assertEqual((control['params']['rows'], control['params']['cols'], control['params']['steps']), (8, 1, 0))

    def test_umbrella_thread_domains_and_reference_arms(self):
        rows = gen.cpu_umbrella_cases()
        cases = {row['id']: row for row in rows}
        self.assertEqual(len(rows), len(cases))
        self.assertTrue({'#2139', '#2141', '#2142', '#2143'} <=
                        {issue for row in rows for issue in row['issues']})
        for row in rows:
            if row['arm'] == 'tenferro':
                references = [c for c in rows if c.get('reference_for') == row['id']]
                self.assertEqual(len(references), 1, row['id'])
                self.assertEqual(references[0]['thread_counts'], row['thread_counts'])
                self.assertEqual(references[0]['measurement_scope'], row['measurement_scope'])
        with patch.dict(os.environ, {'BENCH_COVERAGE':'full', 'BENCH_INSTANCE':''}):
            _, _, _, _, expected1, _, _ = suite.plan(threads=1)
            _, _, _, _, expected4, _, _ = suite.plan(threads=4)
            _, _, _, _, expected8, _, _ = suite.plan(threads=8)
        self.assertIn('typed_elemwise_f64_add_held_len16', expected1)
        self.assertNotIn('typed_elemwise_f64_add_held_len16', expected4)
        self.assertIn('lu_batched_f64_factor_n8_b1024', expected4)
        self.assertNotIn('lu_batched_f64_factor_n8_b1024', expected8)
        self.assertIn('gemm_bound_f64_t1_t8_scaling', expected8)

    def test_unavailable_public_counter_is_not_successful_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'counter'
            binary.write_text('#!/bin/sh\ncat <<\'JSON\'\n' + json.dumps(dict(
                correctness_status='unsupported', samples=[],
                unsupported_reason='no public lane observer', actual_lanes=None)) + '\nJSON\n')
            binary.chmod(0o755)
            output = Path(tmp) / 'samples_t4.jsonl'
            with patch.dict(os.environ, {'BENCH_COVERAGE':'full',
                                          'BENCH_INSTANCE':'linalg_lane_plan_counter'}):
                self.assertFalse(suite.collect(binary, output, 4))
            status = json.loads((Path(tmp)/'case_status_t4.json').read_text())
            self.assertFalse(status['complete'])
            self.assertEqual(status['executed'], [])
            self.assertEqual(status['failed'], [])
            self.assertEqual(status['unsupported'], ['linalg_lane_plan_counter@t4'])

    def test_counter_report_preserves_unavailable_observations_and_metadata_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / 'data/verification/amd-cpu/cpu/perf_issues/probe'
            run_dir.mkdir(parents=True)
            row = dict(gen.cpu_umbrella_cases()[0], case_id='counter-probe',
                       measurement_scope='counter_diagnostic', threads=1,
                       correctness_status='unsupported', samples=[],
                       unsupported_reason='no public plan-build observer',
                       binary_plan_builds_per_call=None,
                       counters=[dict(op='metadata', output_bytes=0,
                                      host_allocations_per_op=0., host_allocated_bytes_per_op=0.,
                                      allocated_bytes_over_output_bytes=None)])
            (run_dir/'samples_t1.jsonl').write_text(json.dumps(row)+'\n')
            with patch.object(suite, 'ROOT', root):
                suite.report(run_dir, 'amd-cpu')
            text = (run_dir/'report.md').read_text()
            self.assertIn('UNSUPPORTED: no public plan-build observer', text)
            self.assertIn('binary_plan_builds_per_call: `null`', text)
            self.assertIn('| metadata | 0 | 0.00 | 0 | — |', text)

    def test_filtered_gpu_diagnostic_can_preserve_latest_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "data/results/nvidia-gpu/gpu/perf_issues/probe"
            run_dir.mkdir(parents=True)
            latest = root / "result/nvidia-gpu/gpu/perf_issues.md"
            latest.parent.mkdir(parents=True)
            latest.write_text("previous publication\n")
            with patch.object(suite, "ROOT", root), patch.dict(os.environ, {"GPU_BENCH_SKIP_LATEST": "1"}):
                suite.report(run_dir, "nvidia-gpu", "gpu")
            self.assertTrue((run_dir / "report.md").is_file())
            self.assertEqual(latest.read_text(), "previous publication\n")

    def test_generated_files_are_fresh(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/generate_perf_issue_cases.py"), "--check"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_steady_report_separates_latency_and_throughput(self):
        cases = json.loads((ROOT / 'data/instances/gpu_perf_issues.json').read_text())
        base = next(c for c in cases if c['kind'] == 'linalg_readback_latency')
        base = dict(base, threads=1, correctness_status='passed',
                    samples=[dict(elapsed_ns=2000, iterations=8)])
        rows = [dict(base, case_id='latency-case', regime='latency'),
                dict(base, case_id='compute-case', regime='throughput')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'run'
            run.mkdir()
            (run / 'samples_t1.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with patch.object(suite, 'ROOT', root), patch.dict(os.environ, {'GPU_BENCH_SKIP_LATEST': '1'}):
                suite.report(run, 'nvidia-gpu', 'gpu')
            text = (run / 'report.md').read_text()
            compute, latency = text.split('### Latency and per-call overhead')
            self.assertIn('| compute-case |', compute)
            self.assertNotIn('| latency-case |', compute)
            self.assertIn('| latency-case |', latency)
            self.assertNotIn('| compute-case |', latency)
            self.assertIn('do not measure compute throughput', latency)

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

    def test_quick_keeps_declared_gpu_latency_controls(self):
        for key in ("cpu", "gpu"):
            with patch.dict(os.environ, {"BENCH_INSTANCE": "", "BENCH_COVERAGE": "quick"}):
                _, cases, _, _, _, selected, _ = suite.plan(key)
            self.assertTrue(selected)
            scopes = {cases[i]["measurement_scope"] for i in selected}
            self.assertEqual(scopes, {"steady_state"} if key == "cpu" else
                             {"steady_state", "shape_stream_diagnostic", "session_entry_diagnostic"})
            if key == "gpu":
                streams = [cases[i] for i in selected if cases[i]["kind"] == "shape_stream"]
                self.assertEqual(len(streams), 6)
                self.assertEqual({c["arm"] for c in streams}, {"fixed-warm", "mixed-warm", "fresh"})
                self.assertTrue(all(not c["params"]["instrument"] for c in streams))
                allocation_controls = [cases[i] for i in selected
                                       if cases[i]["measurement_scope"] == "session_entry_diagnostic"]
                self.assertEqual(len(allocation_controls), 10)
                self.assertTrue(all(c["kind"] == "alloc_zero" for c in allocation_controls))

    def test_dispatch_regressions_have_matched_latency_references(self):
        rows = gen.gpu_dispatch_regression_cases(("nvidia-gpu",))
        cases = {c["id"]: c for c in rows}
        for c in rows:
            self.assertEqual(c["regime"], "latency")
            self.assertEqual(c["measurement_scope"], "steady_state")
            for issue in c["issues"]:
                self.assertIn(issue, c["intent"])
            if "reference_for" in c:
                target = cases[c["reference_for"]]
                self.assertEqual(c["dtype"], target["dtype"])
                self.assertEqual({k:v for k,v in c["params"].items() if k != "arm"},
                                 {k:v for k,v in target["params"].items() if k != "arm"})
        batched = [c for c in rows if c["kind"] == "standard_batched_matmul"]
        self.assertEqual({c["arm"] for c in batched}, {"batch-trailing", "batch-outer", "pytorch"})
        readback = [c for c in rows if c["kind"] == "linalg_readback_latency"]
        self.assertEqual({c["params"]["op"] for c in readback}, {"solve", "qr"})
        self.assertEqual(len(readback), 16)

    def test_rank8_clone_has_metadata_reference_and_thread_controls(self):
        cases = gen.metadata_clone_cases()
        self.assertEqual(len(cases), 4)
        for threads in (1, 4):
            pair = [c for c in cases if c["thread_counts"] == [threads]]
            self.assertEqual({c["backend"] for c in pair}, {"tenferro-rs", "julia-base"})
            for c in pair:
                self.assertEqual(c["params"]["shape"], [2]*8)
                self.assertEqual(c["regime"], "latency")
                self.assertEqual(c["issues"], ["#2180"])

    def test_issue_table_rows_are_present(self):
        cpu = {i for c in gen.generate_cases() for i in c["issues"]}
        gpu = {i for c in gen.generate_gpu_cases() for i in c["issues"]}
        self.assertTrue({"#1992", "#2003", "#1995", "#1900", "#1615", "#2007", "#1803", "#1990",
                         "#2006", "#1885", "#1975", "#1976", "#2008"} <= cpu)
        self.assertEqual(gpu, {"#2009", "#1887", "#1885", "#2078", "#2079", "#2080", "#2081",
                               "#2082", "#2083", "#1924", "#1839", "#1849", "#1846", "#1852",
                               "#1923", "#1898", "#1873", "#1805", "#1949", "#1925", "#1720", "#2192", "#2204", "#2195", "#2193", "#2197"})
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
                "per_cycle": [{"rss_bytes": 1 << 20, "free_bytes": 1 << 30, "delta_bytes": 0}],
            }) + "\n" + json.dumps(issue_row) + "\n")
            with patch.object(suite, "ROOT", Path(tmp)):
                suite.report(run_dir, "nvidia-gpu", "gpu")
            report = Path(tmp) / "result/nvidia-gpu/gpu/perf_issues.md"
            text = report.read_text()
            self.assertIn("## State diagnostics (no timing)", text)
            self.assertIn("pool_retention_f64_alloc-free", text)
            self.assertIn("retained_bytes_after_drop", text)
            self.assertIn("| Cycle | rss_bytes | free_bytes | delta_bytes |", text)
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
        # The snapshot queries are scoped to the device the suite measures on.
        self.assertIn('nvidia-smi -i "$PHYSICAL_DEVICE_ORDINAL"', script)
        self.assertIn("gpu_compute_apps_at_start", script)
        self.assertIn("gpu_compute_apps_at_end", script)
        self.assertIn("not a clean measurement", script)
        # The recorded snapshots are the enforced ones, and both are enforced.
        self.assertIn("BENCH_GPU_APPS_START", script)
        self.assertIn("BENCH_GPU_APPS_END", script)
        self.assertIn('if [[ -n "$BENCH_GPU_APPS_START" || -n "$GPU_APPS_AT_END" ]]', script)

    def test_shape_stream_units_and_counter_values_are_rendered_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "data/results/nvidia-gpu/gpu/perf_issues/probe"
            run_dir.mkdir(parents=True)
            base = dict(case_id="stream", issues=["#1873"], kind="shape_stream", arm="fresh",
                        backend="tenferro-cuda", dtype="f64", threads=1, params={"n": 16},
                        correctness_status="passed", sequence=[{"key": 0, "lhs_strides": [1, 16, 32]}])
            latency = dict(base, measurement_scope="shape_stream_diagnostic",
                           samples=[dict(sample_index=0, elapsed_ns=123456, iterations=1)])
            counter = dict(base, case_id="stream-counters", measurement_scope="counter_diagnostic", samples=[],
                           counter_observations=[dict(phase="completion", stats=dict(
                               plans=dict(entries=8, hits=0, misses=8, evictions=0, retained_bytes=4096),
                               workspace=dict(bytes=2048), retirement=dict(in_flight=0)))])
            (run_dir / "samples_t1.jsonl").write_text(json.dumps(latency) + "\n" + json.dumps(counter) + "\n")
            with patch.object(suite, "ROOT", Path(tmp)):
                suite.report(run_dir, "nvidia-gpu", "gpu")
            text = (Path(tmp) / "result/nvidia-gpu/gpu/perf_issues.md").read_text()
            self.assertIn("Median ns/sequence", text)
            self.assertIn('"lhs_strides":[1,16,32]', text)
            self.assertIn("| completion | — | — | 8 | 0 | 8 | 0 | 4096 | 2048 | 0 |", text)
            self.assertNotIn("stream-counters |", text)

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
