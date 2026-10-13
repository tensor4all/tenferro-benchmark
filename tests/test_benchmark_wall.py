import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_wall import record_report_wall


class CommandWallTests(unittest.TestCase):
    def test_over_budget_is_a_descriptive_observation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "BENCH_COMMAND_START_NS": "1000000000",
            "BENCH_REPORT_START_NS": "2500000000",
            "BENCH_COMMAND_BUDGET_SECONDS": "1",
        }), patch("benchmark_wall.time.monotonic_ns", return_value=3000000000):
            record_report_wall(Path(directory))
            row = json.loads((Path(directory) / "wall_command.json").read_text())
            self.assertEqual(row["command_seconds_before_final_report_write"], 2)
            self.assertEqual(row["report_generation_seconds"], .5)
            self.assertEqual(row["budget_status"], "over_budget; diagnostic only")


if __name__ == "__main__":
    unittest.main()
