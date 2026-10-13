#!/usr/bin/env python3
"""Focused checks for ordinary small-work selection, failures and reporting."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_small_work as suite


class SmallWorkSuiteTest(unittest.TestCase):
    def test_saved_cases_and_selection(self):
        with patch.dict(os.environ, {"BENCH_INSTANCE": ""}):
            cases, config = suite.selected_cases()
        self.assertGreater(len(cases), 0)
        self.assertLess(len(cases), 292)
        self.assertFalse(any(c["api_tier"].endswith(("-fresh", "-setup")) for c in cases if not c["id"].startswith("nonad_")))
        with patch.dict(os.environ, {"BENCH_INSTANCE": "", "BENCH_INCLUDE_SETUP_DIAGNOSTICS": "1"}):
            self.assertEqual(len(suite.selected_cases()[0]), 292)
        self.assertEqual(len({c['id'] for c in cases}), len(cases))
        self.assertEqual(config['runs'], 15)
        with patch.dict(os.environ, {"BENCH_INSTANCE": cases[0]['id']}):
            self.assertEqual(suite.selected_cases()[0], cases[:1])
        with patch.dict(os.environ, {"BENCH_INSTANCE": "misspelled"}):
            with self.assertRaises(ValueError):
                suite.selected_cases()

    def test_manifest_keeps_previous_steady_cases_and_adds_bounded_latency_lane(self):
        def old_diagnostic(c):
            return (c["api_tier"] in {"eager-no-ad", "eager-ad", "compiled-repeat"}
                    or c["api_tier"].endswith(("-fresh", "-setup"))
                    or (c["operation"] == "einsum"
                        and c["api_tier"] in {"concrete-shared", "borrowed-shared"}))
        with patch.dict(os.environ, {"BENCH_INSTANCE": "", "BENCH_INCLUDE_SETUP_DIAGNOSTICS": "1"}):
            every = suite.selected_cases()[0]
        with patch.dict(os.environ, {"BENCH_INSTANCE": "", "BENCH_COVERAGE": "quick"}):
            quick = suite.selected_cases()[0]
        self.assertEqual([c["id"] for c in quick if not c["id"].startswith("nonad_")], [c["id"] for c in every if not c["id"].startswith("nonad_") and not old_diagnostic(c)])
        latency = [c for c in quick if c["id"].startswith("nonad_")]
        self.assertEqual(len(latency), 38)
        self.assertEqual({c["api_tier"] for c in latency}, {"concrete-fresh", "concrete-shared", "concrete-scoped", "eager-no-ad", "eager-shared"})
        self.assertEqual({tuple(c["shape"]) for c in latency}, {(2, 2), (16, 16)})
        for case in every:
            route = suite.route_for(case)
            self.assertIn(route["measurement_scope"], {"steady_state", "session_entry_diagnostic",
                                                       "preparation_diagnostic"})
        self.assertEqual({suite.route_for(c)["measurement_scope"] for c in quick}, {"steady_state", "session_entry_diagnostic"})
        self.assertNotIn("all 166", (Path(suite.ROOT) / "scripts/run_small_work.sh").read_text())

    def test_collect_writes_case_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'ok'
            binary.write_text('#!/bin/sh\necho \'{"correctness_status":"passed","samples":'
                              '[{"elapsed_ns":10,"iterations":1}]}\'\n')
            binary.chmod(0o755)
            output = Path(tmp) / 'samples_t1.jsonl'
            with patch.dict(os.environ, {'BENCH_INSTANCE': 'add_f64_concrete_shared',
                                         'BENCH_COVERAGE': 'quick', 'BENCH_EFFORT': 'scan'}):
                self.assertFalse(suite.collect(binary, output, 1))
            status = json.loads((Path(tmp) / 'case_status_t1.json').read_text())
            self.assertEqual(status['executed'], ['add_f64_concrete_shared@t1'])
            self.assertEqual(status['selection_filter'], 'add_f64_concrete_shared')
            self.assertFalse(status['complete'])  # a filtered run never looks complete
            self.assertEqual(json.loads(output.read_text())['effort'], 'scan')

    def test_generic_einsum_cases_carry_subscripts_and_issue_intent(self):
        with patch.dict(os.environ, {"BENCH_INSTANCE": "", "BENCH_INCLUDE_SETUP_DIAGNOSTICS": "1"}):
            cases, _ = suite.selected_cases()
        generic = [c for c in cases if "operand_shapes" in c]
        self.assertEqual(len(generic), 12)
        for issue, subscripts in [("#1897", "abcd,dbef->acef"), ("#1899", "ax,asb->xsb"),
                                  ("#1904", "ij,jk->ik")]:
            family = [c for c in generic if c["subscripts"] == subscripts]
            self.assertEqual({c["api_tier"] for c in family},
                             {"concrete-shared", "prepared-repeat", "prepared-into-repeat",
                              "dot-general-into-shared"})
            self.assertTrue(all(issue in c["intent"] for c in family))
        with patch.dict(os.environ, {"BENCH_INSTANCE": ""}):
            default_ids = {c["id"] for c in suite.selected_cases()[0]}
        # Steady-state generic routes are in the default run; ordinary einsum is a diagnostic.
        self.assertIn("einsum_f64_prepared-into-repeat_ax-asb-xsb_d4s2_single", default_ids)
        self.assertIn("einsum_f64_dot-general-into-shared_abcd-dbef-acef_d4_single", default_ids)
        self.assertIn("einsum_f64_prepared-repeat_ij-jk-ik_m95k95n1_single", default_ids)
        self.assertNotIn("einsum_f64_concrete-shared_ax-asb-xsb_d4s2_single", default_ids)
        row = dict(generic[1], samples=[])
        self.assertEqual(suite.shape_label(row), "`abcd,dbef->acef` 4×4×4×4, 4×4×4×4 → 4×4×4×4")
        self.assertEqual(suite.shape_label({"shape": [2, 2]}), "2×2")

    def test_noisy_chain_and_failure_are_retained(self):
        with patch.dict(os.environ, {"BENCH_INSTANCE": ""}):
            cases, _ = suite.selected_cases()
        row = dict(cases[1], case_id=cases[1]['id'], threads=1, correctness_status='passed',
                   samples=[{'elapsed_ns': n, 'iterations': 2} for n in [100, 200, 300]],
                   scope={'timer': ['add'], 'outside_timer': ['input_construction']})
        self.assertEqual(suite.summary(row), ('100.00', '50.00', '50.0%', 'NOISY'))
        failed = dict(row, case_id='failed-case', correctness_status='failed', samples=[], error='numerical mismatch')
        self.assertEqual(suite.summary(failed), ('—', '—', '—', 'FAILED'))
        self.assertEqual(suite.summary(dict(row, samples=[]))[-1], 'MISSING')
        with tempfile.TemporaryDirectory() as tmp, patch.object(suite, 'ROOT', Path(tmp).resolve()):
            run = Path(tmp) / 'data/results/amd-cpu/cpu/small_work/test'
            run.mkdir(parents=True)
            (run/'samples_t1.jsonl').write_text('\n'.join(map(json.dumps, [row, failed])))
            suite.report(Path(os.path.relpath(run)), 'amd-cpu')
            report = (run/'report.md').read_text()
            self.assertIn('100.00 | 50.00 | 50.0% | 10.00 | NOISY', report)
            self.assertIn('FAILED', report)
            self.assertIn('numerical mismatch', report)
            self.assertEqual(report, (Path(tmp)/'result/amd-cpu/cpu/small_work.md').read_text())

    def test_execution_failure_is_not_a_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp)/'fail'
            binary.write_text('#!/bin/sh\necho numerical-failure >&2\nexit 1\n')
            binary.chmod(0o755)
            output = Path(tmp)/'samples.jsonl'
            with patch.dict(os.environ, {'BENCH_INSTANCE': 'add_f64_concrete_shared'}):
                self.assertTrue(suite.collect(binary, output, 1))
            row = json.loads(output.read_text())
            self.assertEqual(row['samples'], [])
            self.assertEqual(row['correctness_status'], 'failed')

    def test_nonad_matched_pairs_and_separate_allocation_rows(self):
        case = dict(operation='qr', dtype='f64', shape=[2, 2], layout='col_major_contiguous',
                    workflow='single', calls_per_workflow=1, threads=1, correctness_status='passed')
        rows = [dict(case, case_id='nonad_qr_' + tier, api_tier=tier,
                     samples=[dict(iterations=1000, elapsed_ns=duration)]*3)
                for tier, duration in [('concrete-shared', 2_000_000), ('eager-shared', 2_200_000)]]
        allocation = dict(rows[1], samples=[], allocation_samples=[dict(allocations=7, bytes=123)]*7)
        with tempfile.TemporaryDirectory() as tmp, patch.object(suite, 'ROOT', Path(tmp).resolve()):
            run = Path(tmp)/'data/results/amd-cpu/cpu/small_work/test'
            run.mkdir(parents=True)
            (run/'samples_t1.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            (run/'allocations_t1.jsonl').write_text(json.dumps(allocation))
            suite.report(run, 'amd-cpu')
            text = (run/'report.md').read_text()
            self.assertIn('2000.00 | 2200.00 | 200.00 | 1.100', text)
            self.assertIn('nonad_qr_eager-shared | 1 | 7 | 123 | passed', text)
        clipped = dict(rows[0], samples=[dict(iterations=10, elapsed_ns=100)]*3)
        self.assertIn('INCONCLUSIVE', suite.summary(clipped)[3])


if __name__ == '__main__':
    unittest.main()
