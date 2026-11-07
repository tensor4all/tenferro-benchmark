import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.ci.gpu_binaries import (
    MANIFEST, READBACK, check, main, required_binaries,
)

ROOT = Path(__file__).resolve().parents[3]

# A minimal routine manifest: one generic suite, one delegated AD driver.
ROUTINE = '''
suites:
  gpu/dense:
    file: benchmarks/gpu/dense.yaml
    problems: [dense_matmul_f64_3072]
    backends: [pytorch-cuda]
  gpu/linalg_jvp_vjp:
    file: benchmarks/gpu/linalg_jvp_vjp.yaml
    driver: gpu_linalg_ad
    problems: [linalg_ad_large_grad_sum_svd_s_jvp_f64_256]
    backends: [pytorch-cuda]
'''


def built_binaries():
    """The binaries the hosted build compiles before any pod is rented.

    The first `binaries=(...)` of the build step is the unfiltered weekly list;
    the readback branch replaces it below.
    """
    workflow = yaml.safe_load(
        (ROOT / '.github/workflows/benchmark-runpod-gpu.yml').read_text())
    build = next(step for step in workflow['jobs']['prepare-benchmark']['steps']
                 if step.get('id') == 'build')
    match = re.search(r'binaries=\(([^)]*)\)', build['run'])
    if match is None:
        raise AssertionError('the build step declares no binaries=(...) array')
    return match.group(1).split()


class RequiredBinariesTest(unittest.TestCase):
    def manifest(self, directory, text=ROUTINE):
        path = Path(directory) / 'routine.yaml'
        path.write_text(text)
        return path

    def test_whole_routine_set_needs_the_generic_the_ad_runner_and_the_controls_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            required = required_binaries('all', self.manifest(directory))
        self.assertEqual(required, {
            'benchmark_gpu_rust': ['benchmarks/gpu/dense.yaml'],
            'benchmark_gpu_linalg_ad': ['benchmarks/gpu/linalg_jvp_vjp.yaml'],
            # The routine command collects the control lane after the routine
            # set, and that lane is not a routine manifest entry.
            'benchmark_gpu_perf_issues': ['routine_controls '
                                          '(benchmarks/gpu/manifests/routine_controls.yaml)'],
        })

    def test_a_filtered_selection_does_not_need_the_controls_runner(self):
        # A suite, backend or problem filter turns the control lane off
        # (scripts/run_gpu_suite.sh), so a named suite must not demand its
        # binary.
        with tempfile.TemporaryDirectory() as directory:
            required = required_binaries(
                'benchmarks/gpu/dense.yaml', self.manifest(directory))
        self.assertNotIn('benchmark_gpu_perf_issues', required)

    def test_selection_without_the_ad_driver_ships_one_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            required = required_binaries(
                'benchmarks/gpu/dense.yaml', self.manifest(directory))
        self.assertEqual(list(required), ['benchmark_gpu_rust'])

    def test_selection_names_the_requesting_suite_in_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.manifest(directory)
            required = required_binaries('benchmarks/gpu/linalg_jvp_vjp.yaml', manifest)
            with self.assertRaisesRegex(SystemExit, 'benchmark_gpu_linalg_ad.*linalg_jvp_vjp'):
                check(required, directory)
            # A transferred binary passes the same check.
            (Path(directory) / 'benchmark_gpu_linalg_ad').write_bytes(b'binary')
            with contextlib.redirect_stdout(io.StringIO()):
                check(required, directory)

    def test_readback_ships_its_own_binary(self):
        self.assertEqual(required_binaries(READBACK), {
            'benchmark_gpu_perf_issues': [READBACK]})

    def test_unknown_suite_and_unknown_driver_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.manifest(directory)
            with self.assertRaisesRegex(ValueError, 'absent from'):
                required_binaries('benchmarks/gpu/not_a_routine_suite.yaml', manifest)
            unshipped = self.manifest(directory, ROUTINE.replace(
                'driver: gpu_linalg_ad', 'driver: gpu_new_collector'))
            with self.assertRaisesRegex(ValueError, 'gpu_new_collector'):
                required_binaries('benchmarks/gpu/linalg_jvp_vjp.yaml', unshipped)

    def test_routine_manifest_requires_the_ad_runner(self):
        required = required_binaries('all', MANIFEST)
        self.assertIn('benchmark_gpu_linalg_ad', required)
        self.assertIn('benchmark_gpu_rust', required)
        every = {path for suites in required.values() for path in suites}
        self.assertIn('benchmarks/gpu/linalg_jvp_vjp.yaml', every)
        self.assertIn('benchmarks/gpu/linalg_ad_latency.yaml', every)

    def test_the_weekly_requirement_is_exactly_what_the_workflow_builds(self):
        # Two declarations cannot drift: the build step's `binaries` array and
        # this module. A missed binary is only noticed on the pod after the
        # routine set is collected, so the drift has to fail here instead.
        required = required_binaries('all', MANIFEST)
        self.assertEqual(set(required), set(built_binaries()))

    def test_cli_lists_binaries_or_checks_the_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.manifest(directory)
            argv = ['gpu_binaries.py', '--suite', 'all', '--manifest', str(manifest)]
            out = io.StringIO()
            with patch('sys.argv', argv), contextlib.redirect_stdout(out):
                main()
            self.assertEqual(out.getvalue().splitlines(), [
                'benchmark_gpu_rust\tbenchmarks/gpu/dense.yaml',
                'benchmark_gpu_linalg_ad\tbenchmarks/gpu/linalg_jvp_vjp.yaml',
                'benchmark_gpu_perf_issues\troutine_controls '
                '(benchmarks/gpu/manifests/routine_controls.yaml)',
            ])
            with patch('sys.argv', argv + ['--directory', directory]):
                with self.assertRaisesRegex(SystemExit, 'benchmark_gpu_linalg_ad'):
                    main()


if __name__ == '__main__':
    unittest.main()
