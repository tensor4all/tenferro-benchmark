"""The CPU page refreshes only from zen5-cpu (#515): a scheduled job must collect it."""
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / '.github/workflows'


def load(path):
    doc = yaml.safe_load(path.read_text())
    # PyYAML reads the bare `on:` key as the boolean True.
    doc['on'] = doc.pop(True, doc.get('on'))
    return doc


class Zen5WeeklyWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.doc = load(WORKFLOWS / 'benchmark-weekly-zen5-cpu.yml')
        self.job = self.doc['jobs']['zen5-cpu']
        self.run = '\n'.join(step.get('run', '') for step in self.job['steps'])

    def test_every_maintained_profile_has_a_scheduled_collection(self):
        maintained = yaml.safe_load((ROOT / 'benchmarks/profiles.yaml').read_text())['maintained']
        self.assertEqual(maintained, ['zen5-cpu'])
        self.assertIn('schedule', self.doc['on'])
        self.assertEqual(self.job['env']['BENCHMARK_TARGET_PROFILE'], 'zen5-cpu')

    def test_only_trusted_triggers_reach_the_self_hosted_runner(self):
        self.assertEqual(set(self.doc['on']), {'schedule', 'workflow_dispatch'})
        self.assertEqual(self.job['runs-on'], ['self-hosted', 'linux', 'x64', 'zen5-cpu'])
        for path in WORKFLOWS.glob('*.yml'):
            doc = load(path)
            triggers = doc['on'] if isinstance(doc['on'], (dict, list)) else [doc['on']]
            if not {'pull_request', 'pull_request_target'} & set(triggers):
                continue
            for name, job in doc.get('jobs', {}).items():
                with self.subTest(workflow=path.name, job=name):
                    self.assertNotIn('zen5-cpu', str(job.get('runs-on', '')))

    def test_the_recorded_series_and_the_idle_gate_are_kept(self):
        env = self.job['env']
        self.assertEqual(env['TENFERRO_CPU_FEATURES'], 'native')
        self.assertEqual(env['PUBLICATION_GATE_PROFILE'], 'full')
        self.assertNotIn('BENCHMARK_ALLOW_BUSY_HOST', self.run)
        self.assertNotIn('BENCHMARK_OBSERVER_FOLLOW=1', self.run)
        self.assertIn('BENCHMARK_OBSERVER_HANDSHAKE', self.run)
        self.assertIn('pins/tenferro-rs.rev', self.run)
        self.assertIn('run_all.sh 1 4 8', self.run)

    def test_pages_are_generated_and_checked_and_nothing_is_merged(self):
        self.assertIn('regenerate_report_summaries.py --publish', self.run)
        self.assertIn('regenerate_report_summaries.py --check', self.run)
        self.assertIn('generate_readme_results.py --check', self.run)
        self.assertIn('gh pr create', self.run)
        self.assertNotIn('gh pr merge', self.run)


if __name__ == '__main__':
    unittest.main()
