"""Stateful QR replicate completeness and report quality contracts."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import benchmark_cpu_stateful_linalg as suite


class StatefulLinalgTest(unittest.TestCase):
    def rows(self):
        return [dict(correctness='passed', samples_ns=[2_000_000]*3,
                     workflows_per_sample=64, threads=1, worker_count=1,
                     provider='blas', affinity='none', max_reconstruction_error=1e-15,
                     max_orthogonality_error=2e-15, process_index=i) for i in range(7)]

    def test_incomplete_failed_short_noisy_or_unobserved_replicates_are_inconclusive(self):
        rows = self.rows()
        self.assertEqual(suite.statistics_for(rows)['status'], 'observed')
        self.assertEqual(suite.statistics_for(rows, environment_observed=False)['status'], 'inconclusive')
        self.assertIsNone(suite.statistics_for(rows[:-1]))
        duplicated = self.rows(); duplicated[0]['process_index'] = 1
        self.assertIsNone(suite.statistics_for(duplicated))
        rows[0]['correctness'] = 'failed'
        self.assertIsNone(suite.statistics_for(rows))
        rows = self.rows(); rows[0]['samples_ns'] = [999_999]*3
        self.assertEqual(suite.statistics_for(rows)['status'], 'inconclusive')
        rows = self.rows(); rows[0]['samples_ns'] = [10_000_000]*3
        self.assertEqual(suite.statistics_for(rows)['status'], 'inconclusive')
        rows = self.rows(); rows[0]['worker_count'] = None
        self.assertEqual(suite.statistics_for(rows)['status'], 'inconclusive')

    def test_required_paths_and_workload_sizes_are_declared(self):
        cases = json.loads((ROOT/'data/instances/stateful_linalg.json').read_text())
        self.assertEqual(len(cases), 21)
        self.assertEqual(len({c['id'] for c in cases}), 21)
        self.assertEqual({c['rows'] for c in cases}, {64,128,256})
        for rows in (64,128,256):
            self.assertEqual({c['path'] for c in cases if c['rows']==rows}, set(suite.PATHS))


if __name__ == '__main__':
    unittest.main()
