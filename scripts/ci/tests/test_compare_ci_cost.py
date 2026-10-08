import unittest
from scripts.ci.compare_ci_cost import compare


class CompareCiCostTests(unittest.TestCase):
    def samples(self):
        return [dict(arm=arm, run_id=i, tested_ref='frozen-source', phase='confirmation',
            paid_seconds=600 if arm == 'baseline' else 450, price_per_hour=0.5,
            cuda_passed=285, pjrt_passed=3, tutorial_passed=True,
            cleanup_confirmed=True, gpu_job_conclusion='success')
            for i, arm in enumerate(['baseline', 'candidate', 'candidate', 'baseline', 'baseline', 'candidate'])]

    def test_complete_cost_reduction(self):
        result = compare(self.samples())
        self.assertEqual(result['verdict'], 'PASS')
        self.assertAlmostEqual(result['reduction_fraction'], 0.25)

    def test_faster_but_more_expensive_is_not_savings(self):
        samples = self.samples()
        for sample in samples:
            if sample['arm'] == 'candidate':
                sample['price_per_hour'] = 1
        self.assertEqual(compare(samples)['verdict'], 'FAIL')

    def test_incomplete_or_pilot_or_mixed_ref_is_inconclusive(self):
        for key, value in [('cuda_passed', 284), ('pjrt_passed', 2), ('cleanup_confirmed', False),
                           ('phase', 'diagnostic'), ('tested_ref', 'other-source')]:
            with self.subTest(key=key):
                samples = self.samples()
                samples[1][key] = value
                self.assertEqual(compare(samples)['verdict'], 'INCONCLUSIVE')

    def test_noisy_sample_cannot_be_dropped(self):
        samples = self.samples()
        samples[1]['paid_seconds'] = 900
        self.assertEqual(compare(samples)['verdict'], 'INCONCLUSIVE')
        self.assertEqual(compare(samples[1:])['verdict'], 'INCONCLUSIVE')
