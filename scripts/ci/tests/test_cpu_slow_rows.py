import json
import tempfile
import unittest
from pathlib import Path

from scripts.ci.cpu_slow_rows import collect, slow_rows, to_markdown

CSV_HEADER = 'suite,benchmark,dtype,threads,shape,backend,median_ms,iqr_ms,sampling_policy,status\n'
RUN_YAML = 'suite_id: {suite}\ntenferro_rs:\n  commit: abc\nblas:\n  implementation: mkl\n'


def row(backend, median, iqr=0.0, problem='matmul', threads=1):
    return {'suite_id': 'cpu/einsum', 'op': problem, 'problem': problem, 'threads': threads,
            'backend': backend, 'median_ms': median, 'iqr_ms': iqr}


class SlowRowsTests(unittest.TestCase):
    def test_clearly_slower_tenferro_row_is_reported_against_fastest_reference(self):
        [finding] = slow_rows([row('tenferro-trace', 3.0, 0.1), row('pytorch-cpu', 1.0, 0.1),
                               row('jax-cpu', 1.5, 0.1), row('tenferro-eager', 1.1, 0.1)], 1.5)
        self.assertEqual((finding['backend'], finding['reference_backend'], finding['ratio']),
                         ('tenferro-trace', 'pytorch-cpu', 3.0))

    def test_noise_unstable_unreferenced_and_bound_rows_are_not_reported(self):
        rows = [row('tenferro-trace', 2.0, 0.6), row('pytorch-cpu', 1.0, 0.5),
                row('tenferro-direct', 0.5, problem='view'), row('julia-base', 0.01, 0.02, problem='view'),
                row('tenferro-eager', 9.0, problem='qr'),
                row('tenferro-rs', 9.0, problem='transpose'), row('memcpy', 1.0, problem='transpose')]
        self.assertEqual(slow_rows(rows, 1.5), [])

    def test_thread_counts_are_compared_separately(self):
        rows = [row('tenferro-trace', 3.0, threads=1), row('pytorch-cpu', 1.0, threads=4)]
        self.assertEqual(slow_rows(rows, 1.5), [])


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cpu = self.root / 'data/results/amd-cpu/cpu'

    def run_dir(self, suite, stamp, meta='run.yaml'):
        run = self.cpu / suite / stamp
        run.mkdir(parents=True)
        (run / meta).write_text(RUN_YAML.format(suite=f'cpu/{suite}'))
        return run

    def test_reads_every_suite_format_since_the_nightly_run(self):
        old = self.run_dir('einsum', '20261009_180000')
        (old / 'cpu_ops_t1_20261009_180000.csv').write_text(
            CSV_HEADER + 'small,matmul,f64,1,2x2,tenferro-trace,9.0,0.1,b,ok\n'
            'small,matmul,f64,1,2x2,pytorch-cpu,1.0,0.1,b,ok\n')
        einsum = self.run_dir('einsum', '20261016_180000')
        (einsum / 'cpu_ops_t1_20261016_180000.csv').write_text(
            CSV_HEADER + 'small,matmul,f64,1,2x2,tenferro-trace,3.0,0.1,b,ok\n'
            'small,matmul,f64,1,2x2,pytorch-cpu,1.0,0.1,b,ok\n'
            'small,qr,f64,1,2x2,tenferro-eager,,,b,unsupported: no session API\n'
            'small,qr,f64,1,2x2,pytorch-cpu,1.0,0.1,b,ok\n')
        perm = self.run_dir('permutation', '20261016_190000')
        (perm / 'rust_output_t1.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in [
            {'backend': 'tenferro-rs', 'pattern_id': 'transpose_2d', 'dtype': 'f64', 'threads': 1,
             'median_ms': 4.0, 'p25_ms': 3.9, 'p75_ms': 4.1, 'status': 'ok', 'correctness': 'passed'},
            {'backend': 'hptt', 'pattern_id': 'transpose_2d', 'dtype': 'f64', 'threads': 1,
             'median_ms': 1.0, 'p25_ms': 0.9, 'p75_ms': 1.1, 'status': 'ok', 'correctness': 'passed'}]))
        session = self.run_dir('session_matrix', '20261016_191000', meta='run_t1.yaml')
        (session / 'samples_t1.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in [
            {'case_id': 'matmul_n2', 'operation': 'matmul', 'provider': 'blas', 'threads': 1,
             'status': 'passed', 'correctness': 'passed', 'samples_ns': [5e6, 5e6, 5e6, 5e6]},
            {'case_id': 'matmul_n2', 'operation': 'matmul', 'provider': 'pytorch', 'threads': 1,
             'status': 'passed', 'correctness': 'passed', 'samples_ns': [1e6, 1e6, 1e6, 1e6]},
            {'case_id': 'matmul_n2', 'operation': 'matmul', 'provider': 'faer', 'threads': 1,
             'status': 'unsupported', 'samples_ns': []}]))
        self.run_dir('small_work', '20261016_190500')

        result = collect(self.root, 1.5, '20261016_174851')

        found = {(f['suite_id'], f['backend'], f['reference_backend'], f['ratio']) for f in result['findings']}
        self.assertEqual(found, {('cpu/einsum', 'tenferro-trace', 'pytorch-cpu', 3.0),
                                 ('cpu/permutation', 'tenferro-rs', 'hptt', 4.0),
                                 ('cpu/session_matrix', 'tenferro-blas', 'pytorch-cpu', 5.0)})
        self.assertEqual({r['raw_run'] for r in result['runs']} & {str(old.relative_to(self.root))}, set())
        self.assertEqual({r['tenferro_commit'] for r in result['runs']}, {'abc'})
        self.assertEqual([u['suite_id'] for u in result['unanalysed']], ['cpu/small_work'])
        markdown = to_markdown(result, limit=1)
        self.assertIn('| cpu/session_matrix | matmul | tenferro-blas | 1 | 5.00x | 5.00x |', markdown)
        self.assertIn('the 1 highest ratios are listed below', markdown)


if __name__ == '__main__':
    unittest.main()
