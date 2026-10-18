"""Timing-scope regressions for the overhead smoke trend and legacy raw runs."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_overhead_smoke as report
import generate_overhead_smoke_cases as generator


class OverheadSmokeTests(unittest.TestCase):
    def test_runtime_rows_are_named_diagnostics_with_explicit_scope(self):
        cases = generator.generate_cases()
        steady = [
            case for case in cases if case['timing_scope'] == generator.STEADY_SCOPE
        ]
        diagnostic = [
            case for case in cases if case['timing_scope'] == generator.ENTRY_SCOPE
        ]
        self.assertEqual(len(steady), 8)
        self.assertEqual(len(diagnostic), 2)
        for case in diagnostic:
            self.assertTrue(case['id'].endswith('_session_entry_diagnostic'))
            self.assertEqual(case['params']['timing_scope'], generator.ENTRY_SCOPE)
        policy = generator.collection_policy(cases)
        self.assertIn('mixed', policy['timing_scope'])
        self.assertEqual(
            set(policy['timing_scopes_by_case']), {case['id'] for case in cases}
        )
        self.assertIn(
            'entry/exit inside timing', policy['setup_policy'][generator.ENTRY_SCOPE]
        )
        _, suite = generator.render(cases)
        self.assertNotIn('timing_scope', yaml.safe_load(suite)['defaults']['run'])

    def test_legacy_report_separates_runtime_rows_without_modifying_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'data/results/amd-cpu/cpu/overhead_smoke/fixture'
            run.mkdir(parents=True)
            base = {
                'refs': ['tenferro-rs#2178'],
                'threads': 1,
                'note': 'fixture',
                'backend_thread_budget': 1,
                'correctness_status': 'passed',
                'samples': [{'elapsed_ns': 200, 'iterations': 10}],
                'counters': [
                    {'host_allocations_per_op': 3, 'host_allocated_bytes_per_op': 24}
                ],
            }
            rows = [
                {**base, 'case_id': 'add', 'fixture': 'add_rank0_scalar_n16'},
                {
                    **base,
                    'case_id': 'legacy_fused',
                    'fixture': 'run_prepared_mul_add_32x32x32',
                    'timing_scope': generator.STEADY_SCOPE,
                },
            ]
            samples = run / 'samples_t1.jsonl'
            samples.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
            metadata = run / 'run.yaml'
            metadata.write_text(
                yaml.safe_dump(
                    {
                        'collection': {
                            'timing_scope': 'steady_state_host_api; session entry outside timing'
                        }
                    }
                )
            )
            original_samples = samples.read_bytes()
            original_metadata = metadata.read_bytes()
            with patch.object(report, 'ROOT', root):
                report.report(run, 'amd-cpu')
            text = (run / 'report.md').read_text()
            steady, diagnostics = text.split('## Runtime session-entry diagnostics')
            self.assertIn('| add |', steady)
            self.assertNotIn('legacy_fused', steady)
            self.assertIn(
                'legacy_fused (session-entry diagnostic; legacy raw ID)', diagnostics
            )
            self.assertIn('old blanket claim', diagnostics)
            self.assertIn('20.0', diagnostics)
            self.assertIn('2/2 rows passed', text)
            self.assertEqual(samples.read_bytes(), original_samples)
            self.assertEqual(metadata.read_bytes(), original_metadata)
            self.assertEqual(
                text, (root / 'result/amd-cpu/cpu/overhead_smoke.md').read_text()
            )


if __name__ == '__main__':
    unittest.main()
