import json
from pathlib import Path
import unittest
from compare import evaluate

PROTOCOL = json.loads(Path('result/nvidia-gpu/ci/runpod-prepared-runtime-protocol.json').read_text())


def rows(a=120, b=100):
    return [{'arm': arm, 'paid_seconds': a if arm == 'A' else b, 'driver': '570.195.03'} for arm in PROTOCOL['order']]


class ComparisonTests(unittest.TestCase):
    def test_consistent_reduction_is_significant_in_all_70_assignments(self):
        result = evaluate(rows(), PROTOCOL)
        self.assertEqual(result['verdict'], 'PASS')
        self.assertAlmostEqual(result['mean_reduction'], 1 / 6)
        self.assertAlmostEqual(result['randomization_p'], 1 / 70)

    def test_identical_times_prove_no_improvement(self):
        result = evaluate(rows(b=120), PROTOCOL)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['randomization_p'], 1)

    def test_cold_pull_cannot_be_hidden_by_median(self):
        data = rows()
        next(r for r in data if r['arm'] == 'B')['paid_seconds'] = 240
        result = evaluate(data, PROTOCOL)
        self.assertGreater(result['median_reduction'], .10)
        self.assertLess(result['mean_reduction'], .10)
        self.assertEqual(result['verdict'], 'FAIL')

    def test_driver_imbalance_invalidates_favorable_times(self):
        data = rows()
        for r in data:
            if r['arm'] == 'B': r['driver'] = '570.211.01'
        self.assertEqual(evaluate(data, PROTOCOL)['verdict'], 'INCONCLUSIVE')

    def test_missing_sample_is_not_replaced_or_omitted(self):
        with self.assertRaises(ValueError): evaluate(rows()[:-1], PROTOCOL)

    def test_aa_noise_stops_acceptance(self):
        data = rows(); data[0]['paid_seconds'] = 150
        self.assertEqual(evaluate(data, PROTOCOL)['verdict'], 'INCONCLUSIVE')


if __name__ == '__main__': unittest.main()
