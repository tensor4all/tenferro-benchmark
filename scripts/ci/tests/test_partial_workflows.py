"""Production wiring: failed collection must still reach packaging/upload."""
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[3]


class PartialWorkflowTest(unittest.TestCase):
    def workflow(self, name):
        return yaml.safe_load((ROOT/f'.github/workflows/{name}.yml').read_text())

    def test_cpu_packaging_and_upload_run_after_a_failed_collection(self):
        for name, job in (('benchmark-nightly-linux-cpu','linux-cpu'), ('benchmark-nightly-mac-cpu','mac-cpu')):
            steps=self.workflow(name)['jobs'][job]['steps']
            collect=next(step for step in steps if step.get('id')=='collect')
            package=next(step for step in steps if step.get('name')=="Package this collection's CPU results")
            upload=next(step for step in steps if step.get('name')=='Archive validated raw bundle')
            with self.subTest(workflow=name):
                self.assertIn('started_at=', collect['run'])
                self.assertIn('always()',package['if'])
                self.assertIn('prepare_result_bundle.py',package['run'])
                self.assertIn('dashboard-requirements.txt',package['run'])
                self.assertIn('always()',upload['if'])
                self.assertIn('ingestion.json',upload['if'])
                self.assertEqual(upload['with']['if-no-files-found'],'error')
                self.assertNotIn('continue-on-error',collect)
                self.assertTrue(any(step.get('if')=='failure()' and 'diagnostics' in step.get('with',{}).get('name','') for step in steps))

    def test_gpu_packaging_is_a_separate_always_step_and_diagnostics_stay(self):
        workflow=self.workflow('benchmark-runpod-gpu')
        steps=workflow['jobs']['run-gpu-benchmark']['steps']
        collect=next(step for step in steps if step.get('id')=='collect')
        package=next(step for step in steps if step.get('name')=="Package this collection's GPU results")
        upload=next(step for step in steps if step.get('name')=='Archive benchmark results')
        self.assertNotIn('check_gpu_run.py',collect['run'])
        self.assertIn('timestamp=$GPU_BENCH_TIMESTAMP',collect['run'])
        self.assertIn('always()',package['if'])
        self.assertIn('perf-issues-readback',package['if'])
        self.assertIn('prepare_result_bundle.py',package['run'])
        self.assertIn('always()',upload['if'])
        self.assertIn('publication.json',upload['if'])
        self.assertEqual(upload['with']['if-no-files-found'],'error')
        diagnostics=next(step for step in steps if step.get('name')=='Archive failed-run diagnostics')
        self.assertEqual(diagnostics['if'],'failure()')
        self.assertEqual(diagnostics['with']['name'],'runpod-gpu-diagnostics-${{ github.run_id }}')
        self.assertEqual(diagnostics['with']['retention-days'],7)
        self.assertEqual(diagnostics['with']['if-no-files-found'],'warn')
        runtime=[step for job in workflow['jobs'].values() for step in job.get('steps',[])
                 if step.get('uses')=='actions/upload-artifact@v4' and step['with']['name'].startswith('gpu-runtime-')]
        self.assertEqual(len(runtime),9)
        self.assertTrue(all(step['with']['retention-days']==7 for step in runtime))

    def test_dispatching_with_defaults_does_not_narrow_the_routine_arms(self):
        """A manual dispatch with defaults must collect the routine set, not a probe.

        The input used to default to the three arms without `jax-cuda`, so pressing
        "Run workflow" silently dropped the JAX control from all of a week's suites
        while every suite still completed and the bundle still validated.
        """
        workflow = self.workflow('benchmark-runpod-gpu')
        backends = workflow[True]['workflow_dispatch']['inputs']['backends']
        self.assertEqual(backends['default'], '')
        job = workflow['jobs']['run-gpu-benchmark']
        self.assertIn('jax-cuda', job['env']['GPU_BENCH_RUNTIME_BACKENDS'])
        # The routine manifest's own set is the fallback when the input is empty.
        self.assertIn('gpu_routine_selection.py', (ROOT/'scripts/run_gpu_suite.sh').read_text())

    def test_scheduled_gpu_collection_is_not_a_backend_probe(self):
        workflow = self.workflow('benchmark-runpod-gpu')
        job = workflow['jobs']['run-gpu-benchmark']
        # A scheduled run must not look like a filtered probe: it would skip the
        # routine controls and stop refreshing the routine reports.
        self.assertNotIn('GPU_BENCH_BACKENDS', job['env'])
        self.assertIn('jax-cuda', job['env']['GPU_BENCH_RUNTIME_BACKENDS'])
        collect = next(step for step in job['steps'] if step.get('id') == 'collect')
        self.assertEqual(collect['env']['GPU_BENCH_BACKENDS'], '${{ inputs.backends }}')
        package = next(step for step in job['steps']
                       if step.get('name') == "Package this collection's GPU results")
        self.assertIn('--backends "$GPU_BENCH_RUNTIME_BACKENDS"', package['run'])
        # The pod has no Rust toolchain, so both routine runners ship from the
        # hosted build job.
        build = next(step for step in workflow['jobs']['prepare-benchmark']['steps']
                     if step.get('id') == 'build')
        for binary in ('benchmark_gpu_linalg_ad', 'benchmark_gpu_public_api', 'benchmark_gpu_permutation'):
            self.assertIn(binary, build['run'])
        self.assertIn('artifact_paths', build['run'])
        verify = next(step for step in job['steps']
                      if step.get('name') == 'Verify hosted benchmark build')
        # The required set comes from the selected suites' drivers
        # (scripts/ci/gpu_binaries.py), not from a list beside the transfer step.
        self.assertIn('uv run python scripts/ci/gpu_binaries.py --suite "$REQUESTED_SUITE" '
                      '--directory target/release', verify['run'])
        self.assertIn('gpu_artifact.py', verify['run'])
        self.assertNotIn('|| continue', verify['run'])

    def test_every_gpu_dispatch_carries_its_purpose_into_the_run_record(self):
        workflow = self.workflow('benchmark-runpod-gpu')
        inputs = workflow[True]['workflow_dispatch']['inputs']
        # An unmarked dispatch counts as weekly (the strict direction), so the
        # default is the weekly attempt rather than an exemption.
        self.assertEqual(inputs['purpose']['default'], 'weekly')
        self.assertEqual(inputs['purpose']['options'], ['weekly', 'backfill', 'verify'])
        self.assertEqual(inputs['target']['default'], '')
        # The purpose has to reach the run record: weekly_gate.py reads it back
        # from the run name to count this week's dispatches. The readback family
        # keeps its title prefix, which ingestion excludes on, so a readback is
        # counted under its own purpose too.
        run_name = workflow['run-name']
        self.assertIn("inputs.purpose || 'weekly'", run_name)
        self.assertIn('inputs.target', run_name)
        self.assertIn("'RunPod readback (manual report)'", run_name)
        self.assertIn('[purpose={1} target={2}]', run_name)
        gate = next(step for step in workflow['jobs']['weekly-gate']['steps']
                    if step.get('id') == 'gate')
        self.assertEqual(gate['env']['PURPOSE'], '${{ inputs.purpose }}')
        self.assertEqual(gate['env']['TARGET'], '${{ inputs.target }}')
        self.assertEqual(gate['env']['RUN_ATTEMPT'], '${{ github.run_attempt }}')
        self.assertIn('--run-attempt "$RUN_ATTEMPT" --purpose "$PURPOSE" --target "$TARGET"',
                      gate['run'])
        # A dispatch re-run repeats a dispatch that is already counted, so the
        # renting job refuses it before it can rent; a scheduled slot retries itself.
        self.assertIn("github.event_name == 'schedule' || github.run_attempt == 1",
                      workflow['jobs']['start-runpod']['if'])
        # A re-run of a scheduled slot skips the gate job, so the budget is checked
        # again where the money is spent.
        recheck = next(step for step in workflow['jobs']['start-runpod']['steps']
                       if step.get('name') == 'Re-check the dispatch budget before renting')
        self.assertIn('--purpose "$PURPOSE" --target "$TARGET"', recheck['run'])
        self.assertIn("grep -qx 'proceed=true' \"$GITHUB_OUTPUT\"", recheck['run'])

    def test_ingest_if_admits_only_success_failure_and_preserves_trust_conditions(self):
        condition=self.workflow('ingest-results')['jobs']['ingest']['if']
        self.assertIn('fromJSON(\'["success", "failure"]\')',condition)
        for trust in ('github.repository ==', 'head_repository.full_name == github.repository',
                      "!startsWith(github.event.workflow_run.display_title, 'RunPod readback (manual report)')"):
            self.assertIn(trust,condition)


if __name__ == '__main__':
    unittest.main()
