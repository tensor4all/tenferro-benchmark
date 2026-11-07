"""#335: the measurement CPU is the quietest of the allocation, and a busy one fails loudly."""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gpu_measurement_cpu as selector  # noqa: E402


def stat_line(cpu, busy, idle):
    # user nice system idle iowait irq softirq steal guest guest_nice
    return f"cpu{cpu} {busy} 0 0 {idle} 0 0 0 0 0 0"


class MeasurementCpuTest(unittest.TestCase):
    def run_select(self, before, after, *argv, env=None):
        """Run ``select`` with /proc/stat moving from ``before`` to ``after``."""
        with tempfile.TemporaryDirectory() as temp:
            stat = Path(temp) / "stat"
            record = Path(temp) / "arm.json"
            stat.write_text("cpu 0 0 0 0\n" + "\n".join(before) + "\n")

            def advance(_seconds):
                stat.write_text("cpu 0 0 0 0\n" + "\n".join(after) + "\n")

            out, err = io.StringIO(), io.StringIO()
            with patch.object(selector.time, "sleep", advance), \
                    patch.dict(selector.os.environ, env or {}, clear=False), \
                    redirect_stdout(out), redirect_stderr(err):
                code = selector.main(["select", "--stat", str(stat), "--window", "0",
                                      "--arm", "pytorch-cuda", "--record", str(record), *argv])
            return code, out.getvalue(), err.getvalue(), json.loads(record.read_text())

    def test_parse_and_format_round_trip(self):
        self.assertEqual(selector.parse_cpuset("0-3,8,10-11"), [0, 1, 2, 3, 8, 10, 11])
        self.assertEqual(selector.format_cpuset([0, 1, 2, 3, 8, 10, 11]), "0-3,8,10-11")

    def test_the_quietest_cpu_of_a_larger_allocation_is_taken(self):
        before = [stat_line(c, 0, 0) for c in range(4)]
        # cpu0 70% busy, cpu1 2%, cpu2 1%, cpu3 40%.
        after = [stat_line(0, 70, 30), stat_line(1, 2, 98), stat_line(2, 1, 99), stat_line(3, 40, 60)]
        code, out, _, record = self.run_select(before, after, "--cpuset", "0-3")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "2")
        self.assertEqual(record["cpu"], 2)
        self.assertEqual(record["busy_fraction"], 0.01)
        self.assertEqual(record["cpuset"], "0-3")
        self.assertEqual(record["busy_fraction_by_cpu"], {"0": 0.7, "1": 0.02, "2": 0.01, "3": 0.4})
        self.assertEqual(record["max_busy"], selector.DEFAULT_MAX_BUSY)
        self.assertEqual(record["max_busy_source"], "default")

    def test_no_usable_cpu_exits_76_and_names_every_cpu(self):
        before = [stat_line(c, 0, 0) for c in range(2)]
        after = [stat_line(0, 30, 70), stat_line(1, 20, 80)]
        code, out, err, record = self.run_select(before, after, "--cpuset", "0-1")
        self.assertEqual(code, selector.EXIT_NO_USABLE_CPU)
        self.assertEqual(out, "")
        self.assertIn("cpuset 0-1", err)
        self.assertIn("cpu0=0.300", err)
        self.assertIn("cpu1=0.200", err)
        self.assertEqual(record["status"], "no_usable_cpu")
        self.assertIsNone(record["cpu"])
        self.assertEqual(record["quietest_cpu"], 1)

    def test_a_relaxed_threshold_is_recorded_as_such(self):
        before = [stat_line(0, 0, 0)]
        after = [stat_line(0, 30, 70)]
        code, out, _, record = self.run_select(before, after, "--cpuset", "0",
                                               env={"GPU_BENCH_CPU_MAX_BUSY": "0.5"})
        self.assertEqual((code, out.strip()), (0, "0"))
        self.assertEqual(record["max_busy"], 0.5)
        self.assertEqual(record["max_busy_source"], "GPU_BENCH_CPU_MAX_BUSY")

    def test_the_allocation_defaults_to_the_process_affinity(self):
        with patch.object(selector.os, "sched_getaffinity", return_value={5, 6}), \
                patch.dict(selector.os.environ, {}, clear=True):
            self.assertEqual(selector.allocation(None), ([5, 6], "process affinity"))

    def test_record_run_puts_every_arm_into_run_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            run = temp / "run.yaml"
            run.write_text("suite_id: gpu/fft\ncollection:\n  effort: standard\n")
            (temp / "a.json").write_text(json.dumps({"arm": "pytorch-cuda", "status": "selected",
                                                     "cpu": 3, "busy_fraction": 0.01}))
            (temp / "b.json").write_text(json.dumps({"arm": "jax-cuda", "status": "skipped",
                                                     "reason": "already holds rows"}))
            selector.main(["record-run", "--run-metadata", str(run),
                           str(temp / "a.json"), str(temp / "b.json")])
            block = yaml.safe_load(run.read_text())["collection"]
            self.assertEqual(block["effort"], "standard")
            self.assertEqual(block["measurement_cpu"]["arms"]["pytorch-cuda"]["cpu"], 3)
            self.assertEqual(block["measurement_cpu"]["arms"]["jax-cuda"]["status"], "skipped")
            self.assertIn("quietest CPU", block["measurement_cpu"]["policy"])


if __name__ == "__main__":
    unittest.main()
