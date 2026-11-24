"""The CPU page refreshes only from zen5-cpu (#515): a scheduled job must collect it."""
from pathlib import Path
import re
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / '.github/workflows'


HOSTED = re.compile(r'(ubuntu|macos|windows)-[a-z0-9.-]+')


def github_hosted(job) -> bool:
    """A literal GitHub-hosted image: never a self-hosted label, expression,
    runner group or reusable workflow (the zen5 host also carries
    self-hosted/linux/x64)."""
    runs_on = job.get('runs-on')
    return 'uses' not in job and isinstance(runs_on, str) and bool(HOSTED.fullmatch(runs_on))


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
                    self.assertTrue(github_hosted(job), job.get('runs-on'))

    def test_the_host_runner_check_rejects_a_bare_self_hosted_pr_job(self):
        for job in ({'runs-on': ['self-hosted', 'linux', 'x64']}, {'runs-on': 'self-hosted'},
                    {'runs-on': '${{ inputs.runner }}'}, {'runs-on': {'group': 'runpod-tenferro'}},
                    {'uses': './.github/workflows/benchmark-weekly-zen5-cpu.yml'}):
            self.assertFalse(github_hosted(job), job)
        self.assertTrue(github_hosted({'runs-on': 'ubuntu-latest'}))

    def test_the_recorded_series_and_the_idle_gate_are_kept(self):
        env = self.job['env']
        self.assertEqual(env['TENFERRO_CPU_FEATURES'], 'native')
        self.assertEqual(env['PUBLICATION_GATE_PROFILE'], 'full')
        self.assertNotIn('BENCHMARK_ALLOW_BUSY_HOST', self.run)
        self.assertNotIn('BENCHMARK_OBSERVER_FOLLOW=1', self.run)
        self.assertIn('BENCHMARK_OBSERVER_HANDSHAKE', self.run)
        self.assertIn('pins/tenferro-rs.rev', self.run)
        self.assertIn('run_all.sh 1 4 8', self.run)

    def step(self, prefix):
        return next(step['run'] for step in self.job['steps'] if step.get('name', '').startswith(prefix))

    def test_publish_is_per_suite_and_check_sees_staged_runs(self):
        render = self.step('Render the pages per suite')
        # A direct --publish of every new run dir mixes suites and is refused.
        self.assertNotIn('regenerate_report_summaries.py --publish', self.run)
        publish = render.index('scripts/publish_suite_runs.py')
        stage = render.index('git add data/results/zen5-cpu')
        check = render.index('regenerate_report_summaries.py --check')
        self.assertLess(publish, stage)
        self.assertLess(stage, check)
        self.assertIn('gh pr create', self.run)
        self.assertNotIn('gh pr merge', self.run)

    def test_failed_perf_issues_cases_do_not_abort_the_collection(self):
        collect = self.step('Collect the recorded')
        main = next(line for line in collect.splitlines() if 'RUN_PUBLIC_API_SUITE=0' in line)
        self.assertIn('RUN_PERF_ISSUES_SUITE=0', main)
        self.assertIn('run_perf_issues.sh 1 4 8 || rc=$?', collect)
        # only a finalized record (case_status.json) excuses the nonzero exit
        self.assertIn('case_status.json', collect)
        self.assertIn('exit "$rc"', collect)

    def test_a_redispatch_gets_its_own_branch(self):
        publish = self.step('Open the results pull request')
        self.assertIn('${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}', publish)

    def test_a_broken_julia_project_fails_instead_of_dropping_the_column(self):
        self.assertIn("Pkg.instantiate(); using OMEinsum", self.step('Check out tenferro-rs'))
        self.assertIn('julia_omeinsum_t*.log', self.step('Render the pages per suite'))

    def test_the_runner_toolchain_is_required(self):
        guard = self.step('Refuse a host')
        for tool in ('cargo', 'uv', 'gh', 'julia'):
            self.assertIn(tool, guard)


class PublishBySuiteTest(unittest.TestCase):
    """What the workflow's `--publish` call got wrong: one call per suite."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        import publish_suite_runs
        self.module = publish_suite_runs

    def test_a_mixed_collection_is_published_one_suite_at_a_time(self):
        runs = [ROOT / f'data/results/zen5-cpu/cpu/{suite}/{stamp}' for suite, stamp in (
            ('einsum', '1'), ('fft', '2'), ('einsum', '3'), ('public_api', '4'), ('einsum', '5'))]
        calls = []

        def publish(selection):
            calls.append(list(selection))
            return []

        self.module.publish_by_suite(runs, publish=publish, suite_of=lambda run: 'cpu/' + run.parent.name)
        for selection in calls:
            self.assertEqual(len({run.parent.name for run in selection}), 1, selection)
        self.assertEqual([[run.name for run in selection] for selection in calls],
                         [['1', '3', '5'], ['2'], ['4']])

    def test_the_real_publish_refuses_what_the_grouping_avoids(self):
        import regenerate_report_summaries as regen
        einsum = sorted((ROOT / 'data/results/zen5-cpu/cpu/einsum').glob('*'))
        fft = sorted((ROOT / 'data/results/zen5-cpu/cpu/fft').glob('*'))
        if not einsum or not fft:
            self.skipTest('no tracked zen5-cpu runs')
        with tempfile.TemporaryDirectory() as work:
            with self.assertRaisesRegex(regen.PublishError, 'select runs of one suite'):
                regen.render_pages([einsum[-1], fft[-1]], Path(work), ROOT)

    def test_a_run_selected_twice_is_an_error(self):
        run = ROOT / 'data/results/zen5-cpu/cpu/fft/x'
        with self.assertRaises(self.module.regen.PublishError):
            self.module.group_by_suite([run, run], suite_of=lambda _: 'cpu/fft')

if __name__ == '__main__':
    unittest.main()
