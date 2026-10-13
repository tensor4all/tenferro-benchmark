import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from format_gpu_routine_wall import render


class RoutineWallTest(unittest.TestCase):
    def test_budget_is_descriptive_and_incomplete_selection_is_not_evaluated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "gpu"
            selection = root / "routine_selection/run"
            core = root / "dense/run"
            controls = root / "routine_controls/run"
            for folder in (selection, core, controls):
                folder.mkdir(parents=True)
            def write(folder, name, data):
                (folder / name).write_text(json.dumps(data))
            write(selection, "selection.json", dict(coverage="quick", effort="standard", selected=["gpu/dense:p:b"]))
            write(selection, "case_status.json", dict(executed=["p"], expected=["p"], complete=True))
            write(controls, "case_status.json", dict(measured=["op"], timing=["op"], observed=["state"], state=["state"], complete=True))
            write(core, "run.yaml", dict(target_profile="nvidia-gpu", tenferro_rs=dict(commit="provider"),
                harness=dict(commit="harness"), cuda=dict(device_name="test GPU", device_uuid="GPU-test")))
            write(core, "wall_command.json", dict(command_seconds_through_this_suite=100))
            wall = dict(script_seconds_before_final_wall_record_write=400, routine_controls_selected=True, scope="test scope")
            write(selection, "wall_command_complete.json", wall)
            budget = dict(script_seconds=300, coverage="quick", effort="standard", target_profile="nvidia-gpu",
                          device_name="test GPU", cache_conditions="test cache", scope="test scope", reference_run="baseline")
            self.assertIn("**over budget**", render(selection, budget))
            write(selection, "wall_devcontainer_command.json", dict(seconds=401, tool="test clock"))
            self.assertIn("**401.00 seconds**", render(selection, budget))
            wall["routine_controls_selected"] = False
            write(selection, "wall_command_complete.json", wall)
            report = render(selection, budget)
            self.assertIn("budget not evaluated", report)
            self.assertNotIn("**within budget**", report)


if __name__ == "__main__":
    unittest.main()
