"""Contracts for the full-CI cost experiment (no timing in these tests)."""
import unittest
from pathlib import Path


class CiCostWorkflowTests(unittest.TestCase):
    def test_frozen_workload_and_serial_cuda_execution(self):
        source = (Path(__file__).resolve().parents[3] / '.github/workflows/benchmark-runpod-gpu.yml').read_text()
        self.assertIn('3f10f960af05efa4fc5705aac3dafbac1ae2269e', source)
        self.assertIn("run-id: ${{ inputs.arm == 'candidate' && '37796517303' || '37783818267' }}", source)
        self.assertIn('2605c46f45016a2472ebdf7ba75bba533bc6a5fd', source)
        self.assertEqual(source.count('ref: ${{ env.TENFERRO_REF }}'), 2)
        self.assertIn('Run CUDA tests from archive', source)
        self.assertIn('Run CUDA tutorial artifact', source)
        self.assertIn('Run OpenXLA PJRT E2E tests from archive', source)
        self.assertIn("NEXTEST_TEST_THREADS: '1'", source)
        self.assertNotIn('cargo build', source)
        self.assertIn('CI_COST_DELETE_CONFIRMED', source)
        self.assertIn('!inputs.prepare_only', source)
        self.assertIn("config['max_provision_attempts']=1", source)
        self.assertIn("config['same_tier_retries']=0", source)

    def test_driver_selected_sdk_is_prepared_before_gpu_allocation(self):
        root = Path(__file__).resolve().parents[3]
        source = (root / '.github/workflows/benchmark-runpod-gpu.yml').read_text()
        preparation = (root / 'scripts/ci/prepare_ci_cost_runtime.sh').read_text()
        self.assertIn('for runtime in 12.6 12.8; do', preparation)
        self.assertIn('check_cuda_headers.py --cuda-root "$sdk_root"', preparation)
        self.assertIn('nvidia/cuda:12.6.3-runtime-ubuntu24.04@sha256:', source)
        self.assertIn('--min-runtime-version 12.6 --full-runtime-version 12.8', source)
        self.assertNotIn('--skip-nvrtc-install', source)
        for runtime in ('12.6', '12.8'):
            for part in ('00', '01', '02', '03', '04'):
                self.assertIn(f'path: runtime-sdk-{runtime}/{part}/', source)
        selection = source.index('- name: Select CUDA runtime for driver')
        transfer = source.index('- name: Transfer selected CUDA SDK')
        installation = source.index('- name: Install selected CUDA SDK')
        configuration = source.index('- name: Configure CUDA runtime libraries')
        verification = source.index('- name: Verify loaded NVRTC version')
        execution = source.index('- name: Run CUDA tests from archive')
        self.assertLess(selection, transfer)
        self.assertLess(transfer, installation)
        self.assertLess(installation, configuration)
        self.assertLess(configuration, verification)
        self.assertLess(verification, execution)
        self.assertIn('sha256sum --check sdk.sha256', source)
        self.assertIn('steps.select_cuda_runtime.outputs.runtime_version', source)

    def test_capacity_retry_never_replaces_a_paid_or_other_failure(self):
        import os
        import subprocess
        import tempfile
        import textwrap
        source = (Path(__file__).resolve().parents[3] / '.github/workflows/benchmark-runpod-gpu.yml').read_text()
        start = source.index('        # Capacity retry boundary:')
        end = source.index('\n    - name:', start)
        script = textwrap.dedent(source[start:end])
        stub = '''
python3() {
  local number
  number=$(( $(cat "$COUNT_FILE") + 1 ))
  echo "$number" > "$COUNT_FILE"
  echo "$PROVISION_RUNNER_LABEL" >> "$LABEL_FILE"
  case "$TEST_MODE:$number" in
    capacity:3|success:1) echo 'Created pod accepted: GPU NVIDIA A40 at $0.59/hr'; return 0 ;;
    paid:*) echo 'Created pod rejected: GPU NVIDIA A40 at $0.59/hr'; echo 'RunPod HTTP 500: There are no instances currently available'; return 1 ;;
    other:*) echo 'RunPod HTTP 503: upstream unavailable'; return 1 ;;
    *) echo 'RunPod HTTP 500: There are no instances currently available'; return 1 ;;
  esac
}
sleep() { return 0; }
'''
        for mode, calls, success in [('capacity', 3, True), ('success', 1, True),
                                     ('paid', 1, False), ('other', 1, False),
                                     ('exhausted', 3, False)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                count, labels = root / 'count', root / 'labels'
                count.write_text('0')
                result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + stub +
                                         script.replace('/tmp/runpod-create-', f'{directory}/create-')],
                                        env=dict(os.environ, COUNT_FILE=str(count), LABEL_FILE=str(labels),
                                                 TEST_MODE=mode, PROVISION_RUNNER_LABEL='frozen-run',
                                                 RUNPOD_IMAGE='frozen-image'), capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
                self.assertEqual(int(count.read_text()), calls)
                self.assertEqual(len(set(labels.read_text().splitlines())), calls)
