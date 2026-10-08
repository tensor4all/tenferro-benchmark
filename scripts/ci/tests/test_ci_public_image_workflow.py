"""The public-image comparison changes setup, preserving the full workload."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

class PublicImageWorkflowTests(unittest.TestCase):
    def test_image_campaign_keeps_same_source_and_complete_serial_workload(self):
        text = (ROOT / '.github/workflows/benchmark-runpod-gpu.yml').read_text()
        self.assertIn('TENFERRO_REF: 2605c46f45016a2472ebdf7ba75bba533bc6a5fd', text)
        self.assertNotIn('3f10f960af05efa4fc5705aac3dafbac1ae2269e', text)
        self.assertIn("NEXTEST_TEST_THREADS: '1'", text)
        for name in ('Run CUDA tests from archive', 'Run CUDA tutorial artifact',
                     'Run OpenXLA PJRT E2E tests from archive', 'CI_COST_DELETE_CONFIRMED'):
            self.assertIn(name, text)
        self.assertIn("config['allowed_cuda_versions']=['13.0','12.9','12.8']", text)

    def test_preloaded_runtime_keeps_gpu_smoke_proof_and_content_verification(self):
        text = (ROOT / '.github/workflows/benchmark-runpod-gpu.yml').read_text()
        self.assertIn('cuda_smoke_test.py', text)
        self.assertIn('--skip-nvrtc-install --min-runtime-version 12.8', text)
        self.assertIn('reuse_preloaded_sdk.py --sdk "$sdk_root"', text)
        self.assertIn('sha256sum --check sdk.sha256', text)
        self.assertIn('12.8.1-runtime-ubuntu24.04@sha256:ebef3c171', text)
        self.assertIn('12.6.3-runtime-ubuntu24.04@sha256:92906d875', text)
