import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import format_gpu_command_wall as formatter


class CommandWallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        (self.path / "run.yaml").write_text(yaml.safe_dump(dict(
            suite_id="gpu/dense", tenferro_rs=dict(commit="library"), harness=dict(commit="harness"),
            cuda=dict(device_name="RTX 3060", device_uuid="GPU-0"))))
        (self.path / "wall_command.json").write_text(json.dumps(dict(
            command_seconds_through_this_suite=12,
            collection_seconds_including_build_and_backend_release_delays=10,
            report_generation_and_copy_seconds=.5, scope="test scope")))
        (self.path / "wall_backends.jsonl").write_text(json.dumps(dict(
            backend="unsupported", build_seconds=0, process_seconds=1))+"\n")
        self.row = dict(problem_id="case", backend="unsupported", status="unsupported",
                        execution=dict(notes="No backend configured"))
        (self.path / "records.jsonl").write_text(json.dumps(self.row)+"\n")

    def test_unavailable_phases_and_descriptive_budget(self):
        text = formatter.render([self.path], 5)
        self.assertIn("over budget", text)
        self.assertIn("| case | unsupported | unsupported | — | —", text)
        self.assertIn("does not change functional success", text)

    def test_duplicate_rows_are_rejected(self):
        (self.path / "records.jsonl").write_text((json.dumps(self.row)+"\n")*2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            formatter.render([self.path])

    def test_nan_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            formatter.render([self.path], float("nan"))

    def test_python_sampling_and_validation_phases(self):
        self.row["status"] = "ok"
        self.row["execution"]["notes"] = json.dumps(dict(wall_phases_ns=dict(
            case_body_before_json_write=3_000_000_000,
            sampler_sampling=1_000_000_000,
            initial_validation=100_000_000,
            post_sampling_execution_and_validation=200_000_000)))
        (self.path / "records.jsonl").write_text(json.dumps(self.row)+"\n")
        text = formatter.render([self.path])
        self.assertIn("| 1.000 | 0.300 |", text)


if __name__ == "__main__":
    unittest.main()
