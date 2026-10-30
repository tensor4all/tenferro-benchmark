import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_wall import record_collection_wall


class CommandWallTests(unittest.TestCase):
    def test_over_budget_is_a_descriptive_observation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "BENCH_COMMAND_START_NS": "1000000000",
            "BENCH_COMMAND_BUDGET_SECONDS": "1",
        }), patch("benchmark_wall.time.monotonic_ns", return_value=3000000000):
            record_collection_wall(Path(directory))
            row = json.loads((Path(directory) / "wall_command.json").read_text())
            self.assertEqual(row["command_seconds_through_collection"], 2)
            self.assertIn("through collection", row["scope"])
            self.assertEqual(row["budget_status"], "over_budget; diagnostic only")

    def test_unconfigured_budget_and_no_start_observation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "BENCH_COMMAND_START_NS": "1000000000",
        }, clear=True), patch("benchmark_wall.time.monotonic_ns", return_value=2000000000):
            record_collection_wall(Path(directory))
            row = json.loads((Path(directory) / "wall_command.json").read_text())
            self.assertEqual(row["budget_seconds"], None)
            self.assertEqual(row["budget_status"], "unconfigured")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            record_collection_wall(Path(directory))
            self.assertFalse((Path(directory) / "wall_command.json").exists())


if __name__ == "__main__":
    unittest.main()
