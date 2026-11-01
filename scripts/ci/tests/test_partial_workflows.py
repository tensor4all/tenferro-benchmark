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
        self.assertIn('benchmark_gpu_linalg_ad', build['run'])
        self.assertIn('artifact_paths', build['run'])
        verify = next(step for step in job['steps']
                      if step.get('name') == 'Verify hosted benchmark build')
        self.assertIn('benchmark_gpu_linalg_ad', verify['run'])

    def test_ingest_if_admits_only_success_failure_and_preserves_trust_conditions(self):
        condition=self.workflow('ingest-results')['jobs']['ingest']['if']
        self.assertIn('fromJSON(\'["success", "failure"]\')',condition)
        for trust in ('github.repository ==', 'head_repository.full_name == github.repository',
                      "display_title != 'RunPod readback (manual report)'"):
            self.assertIn(trust,condition)


if __name__ == '__main__':
    unittest.main()
