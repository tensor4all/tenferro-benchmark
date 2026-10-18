import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gpu_routine_controls as controls


class RoutineControlsTest(unittest.TestCase):
    def test_quick_keeps_each_mechanism_and_full_retains_diagnostics(self):
        quick = controls.plan("quick", "standard")
        full = controls.plan("full", "scan")
        self.assertEqual((len(quick["timing"]), len(quick["state"])), (10, 8))
        self.assertEqual((len(full["timing"]), len(full["state"])), (30, 12))
        self.assertEqual(controls.plan("quick", "scan")["timing"], quick["timing"])
        for n in (16, 64):
            for mode in ("fixed-warm", "mixed-warm", "fresh"):
                self.assertIn(f"shape_stream_f64_n{n}_{mode}_latency", quick["timing"])
                self.assertIn(f"shape_stream_f64_n{n}_{mode}_counters", quick["state"])
        for n in (16, 32768):
            for arm in ("alloc-zero-within-callback", "alloc-zero-per-callback"):
                self.assertIn(f"alloc_zero_f64_{arm}_{n}el", quick["timing"])
        self.assertIn("vendor_churn_f64_cycles64", quick["state"])
        self.assertTrue(any("capacity-thrash" in key for key in full["timing"]))
        self.assertFalse(any("capacity-thrash" in key for key in quick["timing"]))

    def test_observation_verification_and_measurement_are_distinct(self):
        plan = dict(manifest_version=1, coverage="quick", effort="scan",
                    timing=["operation"], state=["counter"], unselected=["large"])
        timing = dict(case_id="operation", correctness_status="passed",
                      samples=[dict(elapsed_ns=1000, iterations=10)])
        state = dict(case_id="counter", correctness_status="passed", samples=[])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            (path.parent / "run.yaml").write_text(json.dumps(dict(
                tenferro_rs=dict(commit="provider", features=["cuda"]),
                harness=dict(commit="harness"), cuda=dict(device_uuid="GPU-test"))))
            def account(rows, **kwargs):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                return controls.account(plan, [path], **kwargs)
            result = account([timing, state])
            self.assertTrue(result["complete"])
            self.assertEqual(result["measured"], ["operation"])
            self.assertEqual(result["observed"], ["counter"])
            verify = copy.deepcopy(timing)
            verify["samples"] = []
            result = account([verify, state], verify_only=True)
            self.assertFalse(result["complete"])
            self.assertEqual(result["measured"], [])
            self.assertEqual(result["verification_only"], ["operation"])
            self.assertEqual(account([verify, state])["failed"], ["operation"])
            self.assertEqual(account([timing])["missing"], ["counter"])
            self.assertEqual(account([timing, timing, state])["duplicates"], ["operation"])

    def test_different_provider_revisions_cannot_complete_one_control_run(self):
        selection = dict(manifest_version=1, coverage="quick", effort="scan",
                         timing=["operation"], state=["counter"], unselected=[])
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for lane, revision in (("timing", "old"), ("state", "new")):
                folder = Path(tmp) / lane
                folder.mkdir()
                (folder / "run.yaml").write_text(json.dumps(dict(
                    tenferro_rs=dict(commit=revision, features=["cuda"]),
                    harness=dict(commit="same"), cuda=dict(device_uuid=lane))))
                path = folder / "samples.jsonl"
                path.write_text(json.dumps(dict(
                    case_id="operation" if lane == "timing" else "counter", correctness_status="passed",
                    samples=[dict(elapsed_ns=100, iterations=1)] if lane == "timing" else [])) + "\n")
                paths.append(path)
            result = controls.account(selection, paths)
            self.assertTrue(result["provenance_mismatch"])
            self.assertFalse(result["complete"])

    def test_restricted_accounting_ignores_full_only_controls(self):
        selection = dict(manifest_version=1, coverage="quick", effort="scan",
                         timing=["operation"], state=["counter"], unselected=["extra"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            (path.parent / "run.yaml").write_text(json.dumps(dict(
                tenferro_rs=dict(commit="provider", features=["cuda"]),
                harness=dict(commit="harness"), cuda=dict(device_uuid="GPU-test"))))
            rows = [
                dict(case_id="operation", correctness_status="passed",
                     samples=[dict(elapsed_ns=1000, iterations=10)]),
                dict(case_id="counter", correctness_status="passed", samples=[]),
                dict(case_id="extra", correctness_status="passed",
                     samples=[dict(elapsed_ns=1000, iterations=10)]),
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(controls.account(selection, [path])["unexpected"], ["extra"])
            restricted = controls.account(selection, [path], restrict=True)
            self.assertEqual(restricted["unexpected"], [])
            self.assertTrue(restricted["complete"])


if __name__ == "__main__":
    unittest.main()
