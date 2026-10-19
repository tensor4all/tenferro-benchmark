import json
import tempfile
import unittest
from pathlib import Path

from scripts.ci.bench_rows import SCHEMA, export, load
from scripts.ci.slow_rows import collect, slow_rows, to_markdown

CSV_HEADER = 'suite,benchmark,dtype,threads,shape,backend,median_ms,iqr_ms,sampling_policy,status\n'
RUN_YAML = 'suite_id: {suite}\ntenferro_rs:\n  commit: abc\nblas:\n  implementation: mkl\n'


def jsonl(rows):
    return ''.join(json.dumps(r) + '\n' for r in rows)


def row(backend, median, iqr=0.0, problem='matmul', threads=1, role=None, status='ok'):
    return {'schema': SCHEMA, 'target_profile': 'amd-cpu', 'suite_id': 'cpu/einsum', 'problem_id': problem,
            'op': problem, 'threads': threads, 'backend': backend, 'status': status,
            'role': role or ('tenferro' if backend.startswith('tenferro') else 'reference'),
            'median_ms': median, 'iqr_ms': iqr, 'raw_run': 'r', 'tenferro_commit': 'abc'}


class SlowRowsTests(unittest.TestCase):
    def test_clearly_slower_tenferro_row_is_reported_against_fastest_reference(self):
        [finding] = slow_rows([row('tenferro-trace', 3.0, 0.1), row('pytorch-cpu', 1.0, 0.1),
                               row('jax-cpu', 1.5, 0.1), row('tenferro-eager', 1.1, 0.1)], 1.5)
        self.assertEqual((finding['backend'], finding['reference_backend'], finding['ratio']),
                         ('tenferro-trace', 'pytorch-cpu', 3.0))

    def test_noise_unstable_unsupported_unreferenced_and_bound_rows_are_not_reported(self):
        rows = [row('tenferro-trace', 2.0, 0.6), row('pytorch-cpu', 1.0, 0.5),
                row('tenferro-direct', 0.5, problem='view'), row('julia-base', 0.01, 0.02, problem='view'),
                row('tenferro-eager', None, None, problem='svd', status='unsupported'),
                row('pytorch-cpu', 1.0, problem='svd'),
                row('tenferro-eager', 9.0, problem='qr'),
                row('tenferro-rs', 9.0, problem='transpose'), row('memcpy', 1.0, problem='transpose', role='bound')]
        self.assertEqual(slow_rows(rows, 1.5), [])

    def test_thread_counts_are_compared_separately(self):
        self.assertEqual(slow_rows([row('tenferro-trace', 3.0, threads=1), row('pytorch-cpu', 1.0, threads=4)], 1.5), [])


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_dir(self, suite, stamp, meta='run.yaml', profile='amd-cpu'):
        run = self.root / 'data/results' / profile / suite / stamp
        run.mkdir(parents=True)
        (run / meta).write_text(RUN_YAML.format(suite=suite))
        return run

    def test_every_suite_format_since_the_nightly_run_round_trips(self):
        old = self.run_dir('cpu/einsum', '20261009_180000')
        (old / 'cpu_ops_t1_x.csv').write_text(CSV_HEADER + 'small,matmul,f64,1,2x2,tenferro-trace,9.0,0.1,b,ok\n')
        einsum = self.run_dir('cpu/einsum', '20261016_180000')
        (einsum / 'cpu_ops_t1_x.csv').write_text(
            CSV_HEADER + 'small,matmul,f64,1,2x2,tenferro-trace,3.0,0.1,b,ok\n'
            'small,matmul,f64,1,2x2,pytorch-cpu,1.0,0.1,b,ok\n'
            'small,qr,f64,1,2x2,tenferro-eager,,,b,unsupported: no session API\n')
        perm = self.run_dir('cpu/permutation', '20261016_190000')
        (perm / 'rust_output_t1.jsonl').write_text(jsonl([
            {'backend': b, 'pattern_id': 'transpose_2d', 'dtype': 'f64', 'threads': 1, 'shape': [4, 4],
             'median_ms': m, 'p25_ms': m - 0.1, 'p75_ms': m + 0.1, 'status': 'ok', 'correctness': 'passed'}
            for b, m in (('tenferro-rs', 4.0), ('hptt', 1.0), ('memcpy', 0.5))]))
        session = self.run_dir('cpu/session_matrix', '20261016_191000', meta='run_t1.yaml')
        (session / 'samples_t1.jsonl').write_text(jsonl([
            {'case_id': 'matmul_n2', 'operation': 'matmul', 'provider': p, 'threads': 1, 'status': 'passed',
             'correctness': 'passed', 'samples_ns': [ns] * 4} for p, ns in (('blas', 5e6), ('pytorch', 1e6))] + [
            {'case_id': 'matmul_n2', 'provider': 'faer', 'threads': 1, 'status': 'unsupported',
             'unsupported_reason': 'blas build', 'samples_ns': []}]))
        small = self.run_dir('cpu/small_work', '20261016_190500')
        (small / 'samples_t1.jsonl').write_text(jsonl([
            {'id': 'add_size4', 'operation': 'add', 'backend': 'tenferro-rs', 'threads': 1, 'status': 'passed',
             'correctness_status': 'passed', 'samples': [{'elapsed_ns': 4000, 'iterations': 4}] * 3}]))
        self.run_dir('cpu/overhead_smoke', '20261016_192000')

        rows, skipped = export(self.root, 'amd-cpu', '20261016_174851')
        path = self.root / 'rows.jsonl'
        path.write_text(jsonl(rows))
        rows = load(path)

        self.assertNotIn(str(old.relative_to(self.root)), {r['raw_run'] for r in rows})
        self.assertEqual([s['suite_id'] for s in skipped], ['cpu/overhead_smoke'])
        by_key = {(r['suite_id'], r['backend'], r['problem_id']): r for r in rows}
        self.assertEqual(by_key['cpu/einsum', 'tenferro-eager', 'small/qr/f64/2x2']['status'], 'unsupported')
        self.assertEqual(by_key['cpu/permutation', 'memcpy', 'transpose_2d/f64']['role'], 'bound')
        self.assertEqual(by_key['cpu/session_matrix', 'tenferro-faer', 'matmul_n2']['note'], 'blas build')
        self.assertEqual(by_key['cpu/small_work', 'tenferro-rs', 'add_size4']['median_ms'], 0.001)

        result = collect(rows, 1.5)
        found = {(f['suite_id'], f['backend'], f['reference_backend'], f['ratio']) for f in result['findings']}
        self.assertEqual(found, {('cpu/einsum', 'tenferro-trace', 'pytorch-cpu', 3.0),
                                 ('cpu/permutation', 'tenferro-rs', 'hptt', 4.0),
                                 ('cpu/session_matrix', 'tenferro-blas', 'pytorch-cpu', 5.0)})
        markdown = to_markdown(result, limit=1)
        self.assertIn('| cpu/session_matrix | matmul | tenferro-blas | 1 | 5.00x | 5.00x |', markdown)
        self.assertIn('the 1 highest ratios are listed', markdown)
        self.assertIn('`cpu/small_work`', markdown)
        self.assertIn('no reference backend, not compared', markdown)

    def test_gpu_records_export_with_device_and_verification(self):
        run = self.run_dir('gpu/dense', '20261017_080000', profile='nvidia-gpu')
        (run / 'run.yaml').write_text('suite_id: gpu/dense\ntenferro_rs:\n  commit: abc\n'
                                      'cuda:\n  device_name: NVIDIA A100 80GB PCIe\n')
        (run / 'records.jsonl').write_text(jsonl([
            {'suite_id': 'gpu/dense', 'problem_id': 'mm', 'op': 'matmul', 'backend': b, 'status': 'ok',
             'timing': {'median_ms': m, 'iqr_ms': 0.0}, 'verification': {'status': v}, 'execution': {}}
            for b, m, v in (('tenferro-cuda-eager', 3.0, 'passed'), ('pytorch-cuda', 1.0, 'passed'),
                            ('tenferro-cuda-trace', 9.0, 'failed'))]))
        rows, _ = export(self.root, 'nvidia-gpu', '00000000_000000')
        self.assertEqual({r['device'] for r in rows}, {'NVIDIA A100 80GB PCIe'})
        self.assertEqual({r['backend']: r['status'] for r in rows}['tenferro-cuda-trace'], 'failed')
        [finding] = slow_rows(rows, 1.5)
        self.assertEqual((finding['backend'], finding['threads']), ('tenferro-cuda-eager', None))

    def test_ingestion_bundle_is_read_through_its_manifest(self):
        def source(suite, stamp, index, files, meta='run.yaml'):
            inputs = self.root / 'data/results/amd-cpu' / suite / stamp / 'inputs' / str(index)
            inputs.mkdir(parents=True, exist_ok=True)
            (inputs / meta).write_text(RUN_YAML.format(suite=suite))
            for name, text in files.items():
                (inputs / name).write_text(text)
            return {'metadata': str((inputs / meta).relative_to(self.root)),
                    'raw_files': [str((inputs / name).relative_to(self.root)) for name in files]}

        table = CSV_HEADER + ('small,qr,f64,1,2x2,tenferro-trace,3.0,0.1,b,ok\n'
                              'small,qr,f64,1,2x2,jax-cpu,1.0,0.1,b,ok\n')
        ops = source('cpu/cpu_ops', '20261018_183622', 0, {'cpu_ops_t1_x.csv': table})
        linalg = source('cpu/linalg_jvp_vjp', '20261018_183622', 0, {'cpu_ops_t1_x.csv': table})
        smoke = source('cpu/overhead_smoke', '20261018_191746', 0, {'samples_t1.jsonl': ''})
        # A raw run outside the manifest is not exported.
        (self.run_dir('cpu/cpu_ops', '20261018_200000') / 'cpu_ops_t1_x.csv').write_text(table)
        suites = []
        for suite, src in (('cpu/cpu_ops', ops), ('cpu/linalg_jvp_vjp', linalg), ('cpu/overhead_smoke', smoke)):
            run_directory = str(Path(src['metadata']).parent.parent.parent)
            suites.append({'suite_id': suite, 'run_directory': run_directory, 'tenferro_commit': 'def',
                           'sources': [src], 'files': {f: 'sha-' + Path(f).name for f in src['raw_files']}})
        (self.root / 'ingestion.json').write_text(json.dumps(
            {'schema_version': 1, 'target_profile': 'amd-cpu', 'harness_commit': 'h1', 'suites': suites}))

        rows, skipped = export(self.root, 'amd-cpu', '20261018_174834')

        self.assertEqual({(r['suite_id'], r['backend']) for r in rows},
                         {('cpu/cpu_ops', 'tenferro-trace'), ('cpu/cpu_ops', 'jax-cpu')})
        self.assertEqual({(r['tenferro_commit'], r['harness_commit'], r['blas'], r['raw_run']) for r in rows},
                         {('def', 'h1', 'mkl', 'data/results/amd-cpu/cpu/cpu_ops/20261018_183622')})
        self.assertEqual([s['suite_id'] for s in skipped], ['cpu/overhead_smoke'])
        [finding] = slow_rows(rows, 1.5)
        self.assertEqual(finding['ratio'], 3.0)

    def test_load_rejects_rows_from_another_format(self):
        path = self.root / 'rows.jsonl'
        path.write_text(jsonl([{'backend': 'x'}]))
        with self.assertRaises(ValueError):
            load(path)


if __name__ == '__main__':
    unittest.main()
