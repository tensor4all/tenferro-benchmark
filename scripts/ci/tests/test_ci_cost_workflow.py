"""Contracts for the full-CI cost experiment (no timing in these tests)."""
import unittest
from pathlib import Path


class CiCostWorkflowTests(unittest.TestCase):
    def test_frozen_workload_and_serial_cuda_execution(self):
        source = (Path(__file__).resolve().parents[3] / '.github/workflows/benchmark-runpod-gpu.yml').read_text()
        self.assertIn('3f10f960af05efa4fc5705aac3dafbac1ae2269e', source)
        self.assertIn('run-id: 37783818267', source)
        self.assertIn('Run CUDA tests from archive', source)
        self.assertIn('Run CUDA tutorial artifact', source)
        self.assertIn('Run OpenXLA PJRT E2E tests from archive', source)
        self.assertIn("NEXTEST_TEST_THREADS: '1'", source)
        self.assertNotIn('cargo build', source)
        self.assertIn('CI_COST_DELETE_CONFIRMED', source)
        self.assertIn("config['max_provision_attempts']=1", source)
        self.assertIn("config['same_tier_retries']=0", source)
