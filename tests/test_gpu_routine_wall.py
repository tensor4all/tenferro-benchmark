import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from format_gpu_routine_wall import render


class RoutineWallTest(unittest.TestCase):
    def fixture(self, root, suite_metadata=None):
        selection = root / "routine_selection/run"
        core = root / "dense/run"
        controls = root / "routine_controls/run"
        for folder in (selection, core, controls):
            folder.mkdir(parents=True)
        def write(folder, name, data):
            (folder / name).write_text(json.dumps(data))
        write(selection, "selection.json", dict(effort="standard", selected=["gpu/dense:p:b"]))
        write(selection, "case_status.json", dict(executed=["p"], expected=["p"], complete=True))
        write(controls, "case_status.json", dict(measured=["op"], timing=["op"], observed=["state"], state=["state"], complete=True))
        metadata = dict(target_profile="nvidia-gpu", tenferro_rs=dict(commit="provider"),
                        harness=dict(commit="harness"), cuda=dict(device_name="test GPU", device_uuid="GPU-test"))
        if suite_metadata is not None:
            metadata["collection"] = suite_metadata
        write(core, "run.yaml", metadata)
        write(core, "wall_command.json", dict(command_seconds_through_this_suite=100))
        write(selection, "wall_command_complete.json",
              dict(script_seconds_before_final_wall_record_write=400, routine_controls_selected=True, scope="test scope"))
        return selection

    def budget(self):
        return dict(script_seconds=300, effort="standard", target_profile="nvidia-gpu",
                    device_name="test GPU", cache_conditions="test cache", scope="test scope", reference_run="baseline")

    def test_budget_is_descriptive_and_incomplete_selection_is_not_evaluated(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = self.fixture(Path(tmp) / "gpu")
            self.assertIn("**over budget**", render(selection, self.budget()))
            (selection / "wall_devcontainer_command.json").write_text(json.dumps(dict(seconds=401, tool="test clock")))
            self.assertIn("**401.00 seconds**", render(selection, self.budget()))
            (selection / "wall_command_complete.json").write_text(json.dumps(
                dict(script_seconds_before_final_wall_record_write=400,
                     routine_controls_selected=False, scope="test scope")))
            report = render(selection, self.budget())
            self.assertIn("budget not evaluated", report)
            self.assertNotIn("**within budget**", report)
            self.assertIn("Per-suite clocks are unavailable for this run.", report)

    def test_per_suite_measured_time_comes_from_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = self.fixture(Path(tmp) / "gpu", suite_metadata=dict(
                suite_started_at="2026-11-02T00:00:00Z", suite_finished_at="2026-11-02T00:01:40Z",
                suite_seconds=100.5))
            report = render(selection, self.budget())
            self.assertIn("| Suite | Started (UTC) | Finished (UTC) | Seconds |", report)
            self.assertIn("| dense | 2026-11-02T00:00:00Z | 2026-11-02T00:01:40Z | 100.500 |", report)
            self.assertIn("Effort `standard`", report)
            self.assertNotIn("Coverage `", report)


if __name__ == "__main__":
    unittest.main()
