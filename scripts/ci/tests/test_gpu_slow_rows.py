import json
import tempfile
import unittest
from pathlib import Path

from scripts.ci.gpu_slow_rows import collect, slow_rows


def row(backend, median, iqr=0.0, status='ok', verified='passed', problem='matmul'):
    return {'suite_id': 'gpu/dense', 'problem_id': problem, 'backend': backend, 'status': status,
            'timing': {'median_ms': median, 'iqr_ms': iqr}, 'verification': {'status': verified}}


class SlowRowsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'data/results/nvidia-gpu/gpu/dense/20261017_080000'
        self.run.mkdir(parents=True)
        (self.run / 'run.yaml').write_text(
            'suite_id: gpu/dense\ntenferro_rs:\n  commit: abc\ncuda:\n  device_name: NVIDIA A100 80GB PCIe\n')
        self.records = self.run / 'records.jsonl'

    def write(self, rows):
        self.records.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def test_clearly_slower_tenferro_row_is_reported_against_fastest_reference(self):
        self.write([row('tenferro-cuda-eager', 3.0, 0.1), row('pytorch-cuda', 1.0, 0.1),
                    row('jax-cuda', 1.5, 0.1), row('tenferro-cuda-trace', 1.1, 0.1)])
        [finding] = slow_rows(self.records, 1.5)
        self.assertEqual((finding['backend'], finding['reference_backend'], finding['ratio']),
                         ('tenferro-cuda-eager', 'pytorch-cuda', 3.0))

    def test_noise_unsupported_failed_and_unreferenced_rows_are_not_reported(self):
        self.write([row('tenferro-cuda-eager', 2.0, 0.6), row('pytorch-cuda', 1.0, 0.5),
                    row('tenferro-cuda-trace', 9.0, status='unsupported'),
                    row('tenferro-cuda-trace', 9.0, verified='failed', problem='svd'),
                    row('pytorch-cuda', 1.0, problem='svd'),
                    row('tenferro-cuda-eager', 9.0, problem='qr')])
        self.assertEqual(slow_rows(self.records, 1.5), [])

    def test_collect_records_run_identity(self):
        self.write([row('tenferro-cuda-eager', 3.0), row('pytorch-cuda', 1.0)])
        result = collect(self.root, 1.5)
        self.assertEqual(result['runs'][0]['tenferro_commit'], 'abc')
        self.assertEqual(result['findings'][0]['raw_run'],
                         'data/results/nvidia-gpu/gpu/dense/20261017_080000')


if __name__ == '__main__':
    unittest.main()
