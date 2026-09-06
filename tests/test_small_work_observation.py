"""Passive diagnostic observation, not performance acceptance tests."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from small_work import (ContractError, _communicate_observed, _terminate_process_group,
                        read_machine_observation, run_sequential)


class MachineObservationTests(unittest.TestCase):
    def test_sensor_units_and_missing_values_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in {
                "devices/system/cpu/cpu2/cpufreq/scaling_cur_freq": "1700000\n",
                "class/thermal/thermal_zone0/temp": "42000\n",
                "class/hwmon/hwmon0/temp1_input": "not a number\n",
            }.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            result = read_machine_observation({2, 3}, sysfs=root)
            self.assertEqual(result["frequency_khz"], {"2": 1700000, "3": None})
            self.assertEqual(result["sensor_temperature_millicelsius"], {
                str(root / "class/thermal/thermal_zone0/temp"): 42000,
                str(root / "class/hwmon/hwmon0/temp1_input"): None,
            })
            self.assertEqual(set(result["read_errors"].values()), {"FileNotFoundError", "ValueError"})

    def simulate(self, duration, timeout):
        clock = [0.0]
        class Process:
            def communicate(self, timeout=None):
                if timeout is None or clock[0] + timeout >= duration:
                    clock[0] = duration
                    return "complete stdout", "complete stderr"
                clock[0] += timeout
                raise subprocess.TimeoutExpired("fake", timeout)
        record = {}
        with patch("small_work.time.monotonic", side_effect=lambda: clock[0]), \
             patch("small_work.read_machine_observation", return_value={"frequency_khz": {"0": None}}):
            try:
                output = _communicate_observed(Process(), {0}, timeout, record)
            except subprocess.TimeoutExpired:
                output = "timeout"
        return output, record["machine_observations"], clock[0]

    def test_observes_during_wait_and_keeps_complete_output(self):
        output, trace, elapsed = self.simulate(0.25, 1.0)
        self.assertEqual(output, ("complete stdout", "complete stderr"))
        self.assertEqual([s["phase"] for s in trace["samples"]], ["start", "waiting", "waiting", "end"])
        self.assertEqual(trace["coverage"], "interior")
        self.assertFalse(trace["truncated"])
        self.assertEqual(elapsed, 0.25)

    def test_short_child_is_not_claimed_as_interior_observation(self):
        _, trace, _ = self.simulate(0.01, 1.0)
        self.assertEqual(trace["coverage"], "boundary_only")
        self.assertEqual([s["phase"] for s in trace["samples"]], ["start", "end"])

    def test_timeout_is_not_reset_by_polling(self):
        output, trace, elapsed = self.simulate(10.0, 0.25)
        self.assertEqual(output, "timeout")
        self.assertAlmostEqual(elapsed, 0.25)
        self.assertNotIn("end", [s["phase"] for s in trace["samples"]])

    def test_history_is_bounded_without_a_command_timeout(self):
        output, trace, _ = self.simulate(200.0, None)
        self.assertEqual(output[0], "complete stdout")
        self.assertTrue(trace["truncated"])
        self.assertEqual(len(trace["samples"]), trace["sample_limit"])
        self.assertEqual(trace["samples"][-1]["phase"], "end")

    def test_requires_explicit_selected_cpus(self):
        with self.assertRaises(ContractError):
            run_sequential([], observe_machine=True)

    @unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Linux affinity required")
    def test_real_child_and_observer_failure_cleanup(self):
        command = [sys.executable, "-c", "import sys,time; print('preserved', flush=True); print('stderr', file=sys.stderr, flush=True); time.sleep(.25); print('final')"]
        with patch("small_work.read_machine_observation", return_value={}):
            row = run_sequential([command], expected_affinity=os.sched_getaffinity(0),
                                 observe_machine=True, timeout_s=5, return_records=True)[0]
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["stdout"], "preserved\nfinal\n")
        self.assertEqual(row["stderr"], "stderr\n")
        self.assertEqual(row["machine_observations"]["coverage"], "interior")
        with patch("small_work.read_machine_observation", side_effect=RuntimeError("broken observer")), \
             patch("small_work._terminate_process_group", wraps=_terminate_process_group) as terminate:
            rows = run_sequential([command, command], expected_affinity=os.sched_getaffinity(0),
                                  observe_machine=True, timeout_s=5, return_records=True)
        terminate.assert_called_once()
        self.assertIsNotNone(terminate.call_args.args[0].poll())
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(rows[0]["error"], "child communication/observation failed")
        self.assertEqual(rows[1]["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
