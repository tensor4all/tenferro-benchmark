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
