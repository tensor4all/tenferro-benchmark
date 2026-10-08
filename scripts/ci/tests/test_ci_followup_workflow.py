"""Frozen follow-up experiments must not repeat the SVD source comparison."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

class FollowupWorkflowTests(unittest.TestCase):
    def test_both_arms_use_the_same_frozen_test_source_and_sdk(self):
        text = (ROOT / '.github/workflows/benchmark-runpod-gpu.yml').read_text()
        self.assertIn('TENFERRO_REF: 2605c46f45016a2472ebdf7ba75bba533bc6a5fd', text)
        self.assertNotIn('3f10f960af05efa4fc5705aac3dafbac1ae2269e', text)
        self.assertIn('run-id: 37814278025', text)
        self.assertIn('sha256sum --check ../frozen-transfer/archives.sha256', text)
        self.assertIn('sha256sum --check /tmp/cutensor-before.sha256', text)

    def test_guard_matches_real_job_and_queued_drill_is_explicit(self):
        text = (ROOT / '.github/workflows/benchmark-runpod-gpu.yml').read_text()
        self.assertIn('name: CUDA GPU tests on RunPod', text)
        self.assertIn('unassigned-setup-drill-', text)
        self.assertIn('Run CUDA tests from archive', text)
        self.assertIn("NEXTEST_TEST_THREADS: '1'", text)
        self.assertIn('CI_COST_DELETE_CONFIRMED', text)
        self.assertIn('scripts.ci.runpod_setup_watchdog --budget-seconds', text)

    def test_paid_job_has_no_provider_secret(self):
        text = (ROOT / '.github/workflows/benchmark-runpod-gpu.yml').read_text()
        paid = text.split('  run-ci:', 1)[1].split('  cleanup-runpod:', 1)[0]
        self.assertNotIn('secrets.', paid)
