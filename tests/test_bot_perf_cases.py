"""Regression controls and matched numerical fixtures for #338–#340."""
import unittest
from scripts import generate_perf_issue_cases as gen
from scripts.gpu_perf_issue_references import fixture
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_perf_issues import torch_case


class BotPerfCasesTest(unittest.TestCase):
    def test_cpu_controls_and_thread_domains(self):
        binary = gen.binary_shortcut_cases()
        self.assertEqual(len(binary), 25)
        for row in binary:
            self.assertEqual(row['thread_counts'], [1, 4])
            self.assertEqual(row['regime'], 'latency')
            siblings = [r for r in binary if r['params'] | {'arm': ''} == row['params'] | {'arm': ''}]
            self.assertEqual({r['arm'] for r in siblings},
                             {'string', 'subscripts', 'prepared', 'dot-general', 'pytorch'})
        thin = gen.thin_complex_cases()
        self.assertEqual({r['params']['m'] for r in thin}, {1,2,8})
        for row in thin:
            self.assertEqual(row['profiles'], ['mac-cpu'])
            self.assertEqual(row['thread_counts'], [1])

    def test_dashboard_reads_source_regimes(self):
        from scripts.generate_dashboard import load_suite_metadata, case_regime
        root = Path(__file__).resolve().parents[1]
        metadata = load_suite_metadata(root / 'benchmarks')
        self.assertEqual(case_regime(metadata, 'gpu/perf_issues',
                                    'cholesky_readback_latency_f64_n512'), 'throughput')
        self.assertEqual(case_regime(metadata, 'gpu/perf_issues',
                                    'solve_readback_latency_f64_n4'), 'latency')

    def test_cpu_reference_fixtures(self):
        for row in gen.binary_shortcut_cases() + gen.thin_complex_cases():
            if row['backend'] != 'pytorch-cpu':
                continue
            with self.subTest(case=row['id']):
                op, check, retained, _ = torch_case(row)
                self.assertLess(check(), 1e-10)
                out = op()
                self.assertEqual(retained, out.numel()*out.element_size())

    def test_cholesky_controls_and_numerical_oracle(self):
        import torch
        rows = gen.gpu_cholesky_staging_cases(('nvidia-gpu',))
        self.assertEqual(len(rows), 6)
        self.assertEqual({r['execution_path'] for r in rows}, {'direct','eager','pytorch'})
        for row in rows:
            self.assertEqual(row['regime'], 'throughput')
            self.assertEqual(row['params']['n'], 512)
            self.assertEqual(set(row['issues']), {'#2436','#2440','#2445'})
        for dtype in ('f32','f64'):
            row = next(r for r in rows if r['dtype']==dtype)
            # Run the real 512x512 fixture and reject a corrupted result.
            op, check, retained = fixture(row, 'cpu')
            out = op()
            check(out)
            self.assertEqual(retained, out.numel()*out.element_size())
            with self.assertRaises(AssertionError):
                check(torch.zeros_like(out))
