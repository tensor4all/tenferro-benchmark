import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import small_work_components as components
import run_small_work


class ComponentTests(unittest.TestCase):
    def child(self, records, status="completed"):
        return [{"status": status, "returncode": 0 if status == "completed" else 1,
                 "stdout": "running 1 test\ntest component ... " + "\n".join(
                     components.PREFIX + json.dumps(record) for record in records) + "\nok\n",
                 "stderr": ""}]

    def test_tagged_records_accept_libtest_prefix_and_reject_malformed_output(self):
        self.assertEqual(components.probe_records(self.child([{"ok": True}])[0]["stdout"]),
                         [{"ok": True}])
        for text in ("running 0 tests\ntest result: ok", components.PREFIX + "oops",
                     components.PREFIX + "[]"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                components.probe_records(text)

    def test_cli_rejects_timing_and_archives_invalid_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = Path(__file__).resolve().parents[1] / "scripts/run_small_work.py"
            output, receipt = root / "output.json", root / "receipt.json"
            receipt.write_text("[]")
            command = [sys.executable, str(script), "--component-probe", "--artifact-kind", "lib-test",
                       "--case-binary", str(root / "must-not-execute"), "--output", str(output),
                       "--results-root", str(root / "data")]
            timing = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(timing.returncode, 0)
            self.assertIn("--correctness-only", timing.stderr)
            invalid = subprocess.run(command + ["--correctness-only", "--preparation-receipt", str(receipt)],
                                     capture_output=True, text=True, timeout=10)
            self.assertNotEqual(invalid.returncode, 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "FAILED")
            self.assertIn("preparation receipt is not an object", result["errors"])

    def test_timing_cli_gates_receipt_resources_and_restores_affinity(self):
        for variant in ("valid", "no_receipt", "dirty", "busy", "busy_after", "changed_source", "restore_failed"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt = root / "receipt.json"
                receipt.write_text("{}")
                args = SimpleNamespace(case_binary=root / "probe", artifact_kind="lib-test",
                    correctness_only=False, allocation_iterations=None,
                    suite=Path(__file__).resolve().parents[1] / "benchmarks/cpu/small_work.yaml",
                    results_root=root / "data", target_profile="amd-cpu", output=root / "run.json",
                    preparation_receipt=None if variant == "no_receipt" else receipt,
                    tenferro_dir=root / "library", cargo="cargo", build_project=root / "library",
                    cargo_package="tenferro-einsum", cargo_target_name="tenferro_einsum",
                    observation_window=0, cpuset=None, command_timeout=1)
                valid = {"status": "valid", "cpus": [0], "effective_threads": 1, "reasons": []}
                busy = {"status": "inconclusive", "cpus": [], "reasons": ["busy"]}
                selections = [busy if variant == "busy" else valid, busy if variant == "busy_after" else valid]
                errors = [["dirty"] if variant == "dirty" else [], ["changed"] if variant == "changed_source" else []]
                with mock.patch.object(run_small_work, "verify_preparation_receipt", side_effect=errors) as verify, \
                     mock.patch.object(run_small_work, "_select_live_resources", side_effect=selections), \
                     mock.patch.object(run_small_work.os, "sched_getaffinity", side_effect=[{0,1}, {0}, {0,1}]), \
                     mock.patch.object(run_small_work.os, "sched_setaffinity", side_effect=[None, OSError("restore") if variant == "restore_failed" else None]) as pin, \
                     mock.patch.object(components, "check_components", return_value={"status": "TIMING_DIAGNOSTIC", "errors": []}) as run:
                    self.assertEqual(run_small_work._component_run(args), 0)
                result = json.loads(args.output.read_text())
                self.assertEqual(result["status"], "TIMING_DIAGNOSTIC" if variant == "valid" else "INCONCLUSIVE")
                for call in verify.call_args_list:
                    self.assertTrue(call.kwargs["require_timing"])
                if variant in ("no_receipt", "dirty", "busy"):
                    run.assert_not_called()
                    pin.assert_not_called()
                else:
                    self.assertEqual(pin.call_args_list[-1].args, (0, {0, 1}))
                    self.assertEqual(run.call_args.kwargs["expected_affinity"], {0})

    def test_timing_calibrates_then_validates_independent_process_samples(self):
        protocol = {"warmups": 1, "samples_per_process": 2, "independent_processes": 2,
                    "calibration": {"target_ns": 100}, "noise_policy": {"process_median_cov_max": .10}}
        for variant in ("valid", "truncated", "under_duration", "wrong_stage", "high_cov", "failed_child"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                measured = []
                def child(commands, **kwargs):
                    self.assertEqual(kwargs["expected_affinity"], {0})
                    env = kwargs["env"]
                    mode = env["TENFERRO_PROBE_MODE"]
                    if mode == "contract":
                        return self.child([{"kind": "contract", "schema": "tenferro.einsum.component-probe.v1",
                                            "case_id": "case-a", "stage": "parse", "calls_per_workflow": 1}])
                    if mode == "correctness":
                        return self.child([{"kind": "correctness", "case_id": "case-a", "stage": "parse", "ok": True}])
                    iterations = int(env["TENFERRO_PROBE_ITERATIONS"])
                    count = int(env["TENFERRO_PROBE_SAMPLES"])
                    is_measurement = int(env["TENFERRO_PROBE_MIN_AGGREGATE_NS"]) == 100
                    records = [{"kind": "timing", "case_id": "case-a", "stage": "parse", "sample": index,
                                "iterations": iterations, "completed_iterations": iterations,
                                "elapsed_ns": iterations * 10, "valid": True, "invalid_reason": None}
                               for index in range(count)]
                    if is_measurement:
                        measured.append((iterations, count))
                        if variant == "truncated":
                            records.pop()
                        elif variant == "under_duration":
                            records[-1].update(elapsed_ns=99, valid=False, invalid_reason="under_duration")
                        elif variant == "wrong_stage":
                            records[0]["stage"] = "input_metadata"
                        elif variant == "high_cov" and len(measured) == 2:
                            for record in records:
                                record["elapsed_ns"] *= 10
                    return self.child(records, "failed" if variant in ("failed_child", "under_duration") and is_measurement else "completed")
                with mock.patch.object(components, "run_sequential", side_effect=child):
                    result = components.check_components(Path("probe"), Path(directory), 5,
                                                         protocol=protocol, expected_affinity={0})
                expected_status = "TIMING_DIAGNOSTIC" if variant == "valid" else ("INCONCLUSIVE" if variant in ("high_cov", "under_duration") else "FAILED")
                self.assertEqual(result["status"], expected_status)
                if variant == "valid":
                    self.assertEqual(measured, [(32, 3), (32, 3)])
                    timing = result["timings"][0]
                    self.assertEqual(len(timing["samples"]), 4)
                    self.assertEqual(timing["statistics"]["normalized_ns_per_call"], 10)
                    self.assertEqual(timing["statistics"]["process_count"], 2)
                    self.assertNotIn("allocations", result)
                else:
                    self.assertTrue(result["errors"])
        with self.assertRaises(ValueError):
            components.check_components(Path("unused"), Path("unused"), 1, protocol=protocol)
        with self.assertRaises(ValueError):
            components.check_components(Path("unused"), Path("unused"), 1, 2,
                                         protocol=protocol, expected_affinity={0})

    def test_allocation_diagnostics_are_separate_and_require_complete_counts(self):
        for variant in ("valid", "zero_counts", "wrong_stage", "incomplete", "negative", "bool", "invalid", "duplicate", "child_failed"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                contract = {"kind": "contract", "schema": "tenferro.einsum.component-probe.v1",
                            "case_id": "case-a", "stage": "parse"}
                correct = {"kind": "correctness", "case_id": "case-a", "stage": "parse", "ok": True}
                allocation = {"kind": "allocation", "case_id": "case-a", "stage": "parse",
                              "valid": True, "invalid_reason": None, "iterations": 2,
                              "completed_iterations": 2, "allocation_calls": 4, "requested_bytes": 128}
                if variant == "zero_counts":
                    allocation.update(allocation_calls=0, requested_bytes=0)
                elif variant == "wrong_stage":
                    allocation["stage"] = "input_metadata"
                elif variant == "incomplete":
                    allocation["completed_iterations"] = 1
                elif variant == "negative":
                    allocation["requested_bytes"] = -1
                elif variant == "bool":
                    allocation["allocation_calls"] = True
                elif variant == "invalid":
                    allocation.update(valid=False, invalid_reason="wrong_result")
                records = [allocation, allocation] if variant == "duplicate" else [allocation]
                children = [self.child([contract]), self.child([correct]),
                            self.child(records, "failed" if variant == "child_failed" else "completed")]
                with mock.patch.object(components, "run_sequential", side_effect=children) as run:
                    result = components.check_components(Path("probe"), Path(directory), 5, 2)
                self.assertEqual(result["status"], "ALLOCATION_DIAGNOSTIC" if variant in ("valid", "zero_counts") else "FAILED")
                self.assertEqual(run.call_args_list[2].kwargs["env"]["TENFERRO_PROBE_MODE"], "alloc")
                self.assertEqual(run.call_args_list[2].kwargs["env"]["TENFERRO_PROBE_ITERATIONS"], "2")
                self.assertEqual(run.call_args_list[2].kwargs["env"]["TENFERRO_PROBE_STAGE"], "parse")
                self.assertIn("not retained memory or latency", result["allocation_scope"])
                self.assertEqual(result["records"], [correct])
        for iterations in (0, -1, True):
            with self.subTest(iterations=iterations), self.assertRaises(ValueError):
                components.check_components(Path("unused"), Path("unused"), 1, iterations)

    def test_correctness_requires_all_exported_stages_and_successful_children(self):
        for variant in ("valid", "missing", "duplicate", "wrong_case", "false_ok", "failed_child", "duplicate_contract"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                contracts = [{"kind": "contract", "schema": "tenferro.einsum.component-probe.v1",
                              "case_id": "case-a", "stage": stage} for stage in ("parse", "input_metadata")]
                records = [{"kind": "correctness", "case_id": "case-a", "stage": stage, "ok": True}
                           for stage in ("parse", "input_metadata")]
                if variant == "missing":
                    records.pop()
                elif variant == "duplicate":
                    records[1] = dict(records[0])
                elif variant == "wrong_case":
                    records[0]["case_id"] = "case-b"
                elif variant == "false_ok":
                    records[0]["ok"] = False
                elif variant == "duplicate_contract":
                    contracts.append(dict(contracts[0]))
                children = [self.child(contracts), self.child(records, "failed" if variant == "failed_child" else "completed")]
                with mock.patch.object(components, "run_sequential", side_effect=children) as run:
                    result = components.check_components(Path("probe"), Path(directory), 5)
                self.assertEqual(result["status"], "CORRECTNESS_ONLY" if variant == "valid" else "FAILED")
                self.assertEqual(bool(result["errors"]), variant != "valid")
                self.assertEqual(run.call_args_list[0].kwargs["env"]["TENFERRO_PROBE_MODE"], "contract")
                self.assertIn("--exact", run.call_args_list[0].args[0][0])
                self.assertIn(components.ENTRYPOINT, run.call_args_list[0].args[0][0])
                if variant != "duplicate_contract":
                    self.assertEqual(run.call_args_list[1].kwargs["env"]["TENFERRO_PROBE_CASE"], "case-a")
                self.assertTrue(result["commands"])


if __name__ == "__main__":
    unittest.main()
