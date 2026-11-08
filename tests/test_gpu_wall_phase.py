"""Wall records written by the suites that own their own arm loop."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gpu_wall_phase


class WallPhaseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_an_arm_row_keeps_the_generic_driver_shape(self):
        output = self.root / "wall_backends.jsonl"
        gpu_wall_phase.arm(output, "pytorch-cuda", 1_000_000_000, 3_500_000_000, 12.5)
        row = json.loads(output.read_text().strip())
        # The command-wall report and the run contract both read these keys.
        self.assertEqual(row["backend"], "pytorch-cuda")
        self.assertEqual(row["build_seconds"], 12.5)
        self.assertAlmostEqual(row["process_seconds"], 2.5)
        self.assertIn("scope", row)

    def test_arms_append_rather_than_replace(self):
        output = self.root / "wall_backends.jsonl"
        gpu_wall_phase.arm(output, "jax-cuda", 0, 1_000_000_000, 0.0)
        gpu_wall_phase.arm(output, "pytorch-cuda", 0, 2_000_000_000, 0.0)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([row["backend"] for row in rows], ["jax-cuda", "pytorch-cuda"])

    def test_a_repeated_phase_is_summed_and_never_looks_like_an_arm(self):
        output = self.root / "wall_phases.json"
        gpu_wall_phase.phase(output, "arm_prepare", 0, 500_000_000)
        gpu_wall_phase.phase(output, "arm_prepare", 0, 250_000_000)
        gpu_wall_phase.phase(output, "rust_build", 0, 2_000_000_000)
        recorded = json.loads(output.read_text())
        self.assertAlmostEqual(recorded["phases_seconds"]["arm_prepare"], 0.75)
        self.assertAlmostEqual(recorded["phases_seconds"]["rust_build"], 2.0)
        self.assertIn("scope", recorded)

    def test_an_interval_that_ends_before_it_starts_is_rejected(self):
        with self.assertRaises(ValueError):
            gpu_wall_phase.phase(self.root / "wall_phases.json", "fixtures", 2, 1)

    def test_a_negative_build_is_rejected(self):
        with self.assertRaises(ValueError):
            gpu_wall_phase.arm(self.root / "wall_backends.jsonl", "jax-cuda", 0, 1, -1.0)


if __name__ == "__main__":
    unittest.main()
