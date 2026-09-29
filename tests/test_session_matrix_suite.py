#!/usr/bin/env python3
"""cpu/session_matrix manifest, selection, case-status and failure accounting."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_selection  # noqa: E402
import benchmark_cpu_session as suite  # noqa: E402
import case_status as cs  # noqa: E402
import generate_session_matrix_cases as gen  # noqa: E402

F2 = ["bdot_f64_b1024_m4n4k4_direct_alloc_auto", "bdot_f64_b1024_m4n4k4_direct_into_auto"]


class ManifestTest(unittest.TestCase):
    def test_generated_files_are_fresh(self):
        instances, manifest, suite_yaml = gen.generate()
        self.assertEqual(instances, gen.INSTANCES.read_text())
        self.assertEqual(manifest, gen.MANIFEST.read_text())
        self.assertEqual(suite_yaml, gen.SUITE.read_text())

    def test_quick_holds_the_f2_reproducer_distinct_loop_and_controls(self):
        manifest = bench_selection.load_manifest(suite.MANIFEST)
        quick = manifest["coverage"]["quick"]
        for case_id in F2 + ["matmul_n4_count1024", "beinsum_f64_b1024_m4n4k4_direct_alloc_auto",
                             "bdot_f64_b64_m4n4k4_direct_alloc_auto",
                             "bdot_f64_b16_m64n64k64_direct_alloc_auto",
                             "bdot_f64_b1024_m4n4k4_canonical_alloc_auto",
                             "hadamard_f64_m64n64_einsum_alloc", "chain3_f64_n4_einsum_alloc"]:
            self.assertIn(case_id, quick)
        for stream in ("fixed", "mixed", "fresh", "strides"):
            self.assertIn(f"stream_f64_{stream}_len32_einsum_alloc", quick)
        cases = suite.load_cases()
        # The batched op and the 1024-call loop are different workloads.
        self.assertEqual(cases[F2[0]]["route"]["workload_shape"], "one-batched-op")
        self.assertEqual(cases["matmul_n4_count1024"]["route"]["workload_shape"],
                         "loop-of-1024-independent-ops")
        self.assertEqual({cases[i]["pair"] for i in F2}, {"bdot_f64_b1024_m4n4k4_direct_auto"})
        full = manifest["coverage"]["full"]
        self.assertTrue(any(cases[i].get("dtype") == "c64" for i in full))
        self.assertTrue(any(cases[i].get("layout") == "strided" for i in full))
        self.assertTrue(any(cases[i].get("policy_id", "").startswith("forced-") for i in full))
        # Every pair has both twins in the same coverage tier.
        for pair in manifest["route_contract_pairs"]:
            members = [c for c in cases.values() if c.get("pair") == pair]
            self.assertEqual(sorted(c["output"] for c in members), ["alloc", "into"])
            self.assertEqual(len({c["coverage"] for c in members}), 1)

    def test_selection_is_explicit(self):
        manifest = bench_selection.load_manifest(suite.MANIFEST)
        expected, selected, raw = bench_selection.select(manifest, "quick", "")
        self.assertEqual(expected, selected)
        self.assertIsNone(raw)
        _, selected, raw = bench_selection.select(manifest, "quick", ",".join(F2))
        self.assertEqual(selected, F2)
        with self.assertRaises(bench_selection.SelectionError):
            bench_selection.select(manifest, "quick", "misspelled")
        with self.assertRaises(bench_selection.SelectionError):
            bench_selection.select(manifest, "quick", "bdot_c64_b1024_m4n4k4_direct_alloc_auto")

    def test_effort_is_independent_and_confirm_needs_a_declared_config(self):
        defaults = {"warmups": 3, "runs": 15}
        self.assertEqual(bench_selection.resolve_effort(defaults, None, "standard")["runs"], 15)
        self.assertEqual(bench_selection.resolve_effort(defaults, {"scan": {"runs": 3}}, "scan")["runs"], 3)
        with self.assertRaises(bench_selection.SelectionError) as raised:
            bench_selection.resolve_effort(defaults, None, "confirm")
        self.assertIn("thresholds.relative", str(raised.exception))
        placeholder = yaml.safe_load(bench_selection.CONFIRMATION_CONFIG.read_text())
        self.assertEqual(placeholder["status"], "placeholder")
        self.assertIsNone(placeholder["thresholds"]["relative"])


class CaseStatusTest(unittest.TestCase):
    def test_only_passing_rows_count_as_executed(self):
        rows = [{"case_key": "a@t1", "status": "passed", "cov": 0.5},
                {"case_key": "b@t1", "status": "unsupported"},
                {"case_key": "c@t1", "status": "failed"},
                {"case_key": "e@t1", "status": "timeout"}]
        status = cs.build_case_status(suite_id="cpu/x", manifest_version=1, coverage="quick",
                                      effort="scan", expected=["a@t1", "b@t1", "c@t1", "d@t1", "e@t1", "f@t1"],
                                      selected=["a@t1", "b@t1", "c@t1", "d@t1", "e@t1"], rows=rows,
                                      selection_filter="a,b,c,d,e")
        self.assertEqual(status["executed"], ["a@t1"])
        self.assertEqual(status["noisy"], ["a@t1"])
        self.assertEqual(status["unsupported"], ["b@t1"])
        self.assertEqual(status["failed"], ["c@t1", "e@t1"])
        self.assertEqual(status["missing"], ["d@t1"])
        self.assertEqual(status["not_selected"], ["f@t1"])
        self.assertFalse(status["complete"])


class CollectTest(unittest.TestCase):
    def run_collect(self, script, ids):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        binary = Path(tmp.name) / "case"
        binary.write_text(script)
        binary.chmod(0o755)
        with patch.dict(os.environ, {"BENCH_INSTANCE": ",".join(ids), "BENCH_EFFORT": "scan"}):
            failed = suite.collect(binary, Path(tmp.name), 4)
        rows = [json.loads(l) for l in (Path(tmp.name) / "samples_t4.jsonl").read_text().splitlines()]
        status = json.loads((Path(tmp.name) / "case_status_t4.json").read_text())
        return failed, rows, status

    def test_crash_and_unsupported_rows_never_become_latencies(self):
        failed, rows, status = self.run_collect("#!/bin/sh\necho boom >&2\nexit 1\n", F2)
        self.assertTrue(failed)
        self.assertTrue(all(r["status"] == "failed" and r["samples_ns"] == [] for r in rows))
        self.assertEqual(status["executed"], [])
        unsupported = ('#!/bin/sh\necho \'{"correctness":"unsupported","unsupported_reason":"x",'
                       '"samples_ns":[]}\'\n')
        failed, rows, status = self.run_collect(unsupported, F2)
        self.assertFalse(failed)
        self.assertEqual(len(status["unsupported"]), 2)
        self.assertEqual(status["executed"], [])

    def test_timeout_is_a_liveness_failure(self):
        with patch.object(suite, "CASE_TIMEOUT_S", 1):
            row = suite.run_one(["sleep", "5"], timeout=1)
        self.assertEqual(row["correctness"], "timeout")


if __name__ == "__main__":
    unittest.main()
