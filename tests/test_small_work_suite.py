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
        self.assertLess(len(cases), 154)
        self.assertFalse(any(c["api_tier"].endswith(("-fresh", "-setup")) for c in cases))
        with patch.dict(os.environ, {"BENCH_INSTANCE": "", "BENCH_INCLUDE_SETUP_DIAGNOSTICS": "1"}):
            self.assertEqual(len(suite.selected_cases()[0]), 154)
        self.assertEqual(len({c['id'] for c in cases}), len(cases))
        self.assertEqual(config['runs'], 15)
        with patch.dict(os.environ, {"BENCH_INSTANCE": cases[0]['id']}):
            self.assertEqual(suite.selected_cases()[0], cases[:1])
        with patch.dict(os.environ, {"BENCH_INSTANCE": "misspelled"}):
            with self.assertRaises(ValueError):
                suite.selected_cases()

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


if __name__ == '__main__':
    unittest.main()
