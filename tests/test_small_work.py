#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_small_work  # noqa: E402
from small_work import (  # noqa: E402
    CaseContract,
    ContractError,
    cpu_busy_fraction,
    parse_cpu_set,
    parse_proc_stat,
    render_report,
    run_sequential,
    select_idle_cpus,
    timing_statistics,
    validate_record,
    read_cgroup_limits,
    _display_ns,
    validate_suite_contract,
    make_record,
    resolve_case_filter,
)


class SmallWorkTests(unittest.TestCase):
    def setUp(self) -> None:
        self._binding_patch = mock.patch.object(run_small_work, "verify_canonical_binding", return_value={})
        self._binding_patch.start()
        snapshot_patch = mock.patch.object(run_small_work, "verify_canonical_snapshot")
        snapshot_patch.start()
        self.addCleanup(snapshot_patch.stop)
        self.case = CaseContract.from_mapping({
            "id": "add", "contract_id": "core.add.ordinary.concrete", "family": "core", "surface": "concrete",
            "operation": "add", "phase": "execution",
            "api_tier": "concrete-fresh", "backend": "fixture", "dtype": "f64",
            "layout": "col_major", "shape": [4], "calls_per_workflow": 10,
            "workflow": "ten_calls", "setup": {"includes": ["output"], "excludes": ["input"]},
            "scope": {"timer": ["backend_call", "output_lifetime"], "outside_timer": ["input_construction"]},
        })

    def tearDown(self) -> None:
        self._binding_patch.stop()

    def test_case_filter_resolves_exact_ids_and_rejects_bad_lists(self) -> None:
        cases = [self.case, CaseContract.from_mapping({
            "id": "other", "contract_id": "core.add.ordinary.concrete", "family": "core", "surface": "concrete",
            "operation": "add", "phase": "execution", "api_tier": "concrete-fresh", "backend": "fixture", "dtype": "f64",
            "layout": "col_major", "shape": [4], "calls_per_workflow": 1, "workflow": "one",
            "setup": {"includes": [], "excludes": []}, "scope": {"timer": ["op"], "outside_timer": []},
        })]
        selected, raw = resolve_case_filter(cases, "other,add")
        self.assertEqual([case.case_id for case in selected], ["other", "add"])
        self.assertEqual(raw, "other,add")
        for value in ("", "add,", "missing", "add,add"):
            with self.subTest(value=value), self.assertRaises(ContractError):
                resolve_case_filter(cases, value)

    def test_contract_rejects_setup_overlap_and_unknown_tier(self) -> None:
        value = {"id": "x", "operation": "add", "phase": "execution", "api_tier": "bad",
                 "backend": "fixture", "dtype": "f64", "layout": "col_major", "shape": [1],
                 "calls_per_workflow": 1, "workflow": "one", "setup": {"includes": [], "excludes": []},
                 "scope": {"timer": [], "outside_timer": []}}
        with self.assertRaises(ContractError):
            CaseContract.from_mapping(value)
        value["api_tier"] = "concrete-fresh"
        value["setup"] = {"includes": ["output"], "excludes": ["output"]}
        with self.assertRaises(ContractError):
            CaseContract.from_mapping(value)

    def test_ns_statistics_preserve_raw_and_normalize_ten_call_workflow(self) -> None:
        samples = [{"process_index": process, "sample_index": i, "elapsed_ns": raw, "iterations": 2}
                   for process in (0, 1) for i, raw in enumerate((2000, 2200, 2400))]
        stats = timing_statistics(samples, 10)
        self.assertEqual(stats["raw_elapsed_ns"], [2000, 2200, 2400, 2000, 2200, 2400])
        self.assertEqual(stats["median_ns"], 2200)
        self.assertEqual(stats["normalized_ns_per_workflow"], 1100)
        self.assertEqual(stats["normalized_ns_per_call"], 110)

    def test_correctness_and_timing_validity_are_independent(self) -> None:
        record = make_record(self.case, [], correctness_status="failed", timing_status="inconclusive",
                             timing_reasons=["resource unavailable"], errors=["mismatch"])
        validate_record(record)
        self.assertEqual(record["correctness"]["status"], "failed")
        self.assertEqual(record["timing_validity"]["status"], "inconclusive")

    def test_valid_record_report_has_adaptive_units_and_chain_units(self) -> None:
        record = make_record(self.case,
                             [{"process_index": process, "sample_index": i, "elapsed_ns": 20_000 + (process + i) * 1000, "iterations": 1}
                              for process in (0, 1) for i in range(2)], correctness_status="passed", provider="fixture")
        report = render_report([record])
        self.assertIn("2.100 us", report)
        self.assertIn("2.100 us", report)

    def test_failed_correctness_cannot_claim_valid_timing(self) -> None:
        record = make_record(self.case,
                             [{"process_index": 0, "sample_index": 0, "elapsed_ns": 1000, "iterations": 1}],
                             correctness_status="failed")
        validate_record(record)
        self.assertEqual(record["timing_validity"]["status"], "invalid")

    def test_malformed_or_empty_valid_samples_fail(self) -> None:
        with self.assertRaises(ValueError):
            timing_statistics([], 1)
        record = make_record(self.case, [], correctness_status="not_run", timing_status="valid")
        validate_record(record)
        self.assertEqual(record["timing_validity"]["status"], "invalid")

    def test_proc_stat_fixture_produces_per_cpu_load(self) -> None:
        before = parse_proc_stat("cpu0 10 0 10 80 0 0 0 0 0 0\n")
        after = parse_proc_stat("cpu0 20 0 20 80 0 0 0 0 0 0\n")
        self.assertEqual(cpu_busy_fraction(before, after), {0: 1.0})

    def test_resource_fixture_honors_smt_and_quota(self) -> None:
        before = {0: (100, 10), 1: (100, 10), 2: (100, 10), 3: (100, 10)}
        after = {0: (200, 20), 1: (200, 20), 2: (200, 20), 3: (200, 90)}
        loads = cpu_busy_fraction(before, after)
        selected = select_idle_cpus(loads, parse_cpu_set("0-3"), 2,
                                    smt_siblings={0: 0, 1: 0, 2: 2, 3: 2}, quota_cpus=2)
        self.assertEqual(selected["status"], "inconclusive")
        self.assertEqual(selected["cpus"], [0])
        unavailable = select_idle_cpus({0: 0.9, 1: 0.9}, [0, 1], 1)
        self.assertEqual(unavailable["status"], "inconclusive")

    def test_guest_ticks_are_not_double_counted(self) -> None:
        parsed = parse_proc_stat("cpu0 10 0 20 30 4 5 6 7 100 200\n")
        self.assertEqual(parsed[0], (82, 48))

    def test_normalize_before_process_statistics_and_reject_identity_bools(self) -> None:
        samples = [
            {"process_index": process, "sample_index": index, "elapsed_ns": raw, "iterations": iterations}
            for index, (process, raw, iterations) in enumerate(((0, 100, 1), (0, 100, 1), (1, 1000, 10), (1, 1000, 10)))
        ]
        stats = timing_statistics(samples, 1)
        self.assertEqual(stats["process_median_ns"], 100)
        self.assertEqual(stats["process_median_cov"], 0)
        with self.assertRaises(ValueError):
            timing_statistics([{"process_index": False, "sample_index": 0,
                                "elapsed_ns": 1, "iterations": 1}], 1)
        with self.assertRaises(ValueError):
            timing_statistics(samples + [samples[0]], 1)

    def test_non_core_canonical_contract_id_matches_case_exactly(self) -> None:
        case = CaseContract.from_mapping({
            "id": "einsum", "contract_id": "einsum.einsum.ordinary.concrete", "family": "einsum",
            "surface": "concrete", "operation": "einsum", "phase": "execution",
            "api_tier": "concrete-fresh", "backend": "fixture", "dtype": "f64",
            "layout": "col_major", "shape": [4], "calls_per_workflow": 1,
            "workflow": "one", "setup": {"includes": [], "excludes": []},
            "scope": {"timer": ["backend_call"], "outside_timer": ["input_construction"]},
        })
        record = make_record(case, [], correctness_status="passed", timing_status="inconclusive")
        validate_record(record, case)
        mismatched = dict(record)
        mismatched["contract_id"] = "Einsum.einsum.ordinary.concrete"
        with self.assertRaises(ContractError):
            validate_record(mismatched, case)
        with self.assertRaises(ContractError):
            validate_record(record, self.case)

    def test_validate_record_compares_supplied_statistics_and_operation(self) -> None:
        record = make_record(self.case,
                             [{"process_index": process, "sample_index": i, "elapsed_ns": 1000 + process * 100 + i, "iterations": 1}
                              for process in (0, 1) for i in range(2)],
                             correctness_status="passed", provider="fixture")
        record["timing"]["normalized_ns_per_call"] = 999
        with self.assertRaises(ContractError):
            validate_record(record)
        record = make_record(self.case, [], correctness_status="passed", timing_status="inconclusive")
        validate_record(record)
        record["operation"] = "fabricated"
        self.assertEqual(record["operation"], "fabricated")
        # An operation is required and cannot be blank even without samples.
        record["operation"] = ""
        with self.assertRaises(ContractError):
            validate_record(record)
        unknown = make_record(self.case,
                              [{"process_index": process, "sample_index": index, "elapsed_ns": 1000 + process + index, "iterations": 1}
                               for process in (0, 1) for index in (0, 1)],
                              correctness_status="passed")
        with self.assertRaises(ContractError):
            validate_record(unknown)

    def test_display_preserves_nonzero_subnanosecond_values(self) -> None:
        self.assertNotEqual(_display_ns(0.125), "0 ns")
        self.assertEqual(_display_ns(0), "0 ns")

    def test_cgroup_limits_walk_ancestors_and_allow_missing_intermediate_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mount = root / "cgroup"
            leaf = mount / "user.slice" / "session.scope"
            leaf.mkdir(parents=True)
            (mount / "cpuset.cpus.effective").write_text("0-3")
            (mount / "cpu.max").write_text("200000 100000")
            (leaf / "cpu.max").write_text("max 100000")
            result = read_cgroup_limits(cgroup_text="0::/user.slice/session.scope\n",
                                        mountinfo_text="1 1 0:1 / /cgroup rw - cgroup2 cgroup2 rw\n",
                                        cgroup_mount=mount)
            self.assertEqual(result["cpuset"], (0, 1, 2, 3))
            self.assertEqual(result["quota_cpus"], 2.0)

    def test_fixture_cli_cannot_execute_commands_or_claim_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after, output = root / "before", root / "after", root / "result.json"
            before.write_text("cpu0 10 0 10 80 0 0 0 0\n")
            after.write_text("cpu0 20 0 20 80 0 0 0 0\n")
            completed = subprocess.run([
                sys.executable, "scripts/run_small_work.py", "--synthetic-fixture",
                "--before", str(before), "--after", str(after), "--output", str(output),
                "--command", json.dumps(["/bin/true"])], check=False,
                capture_output=True, text=True, timeout=10)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())

    def test_campaign_requires_every_child_and_exact_sample_identities(self) -> None:
        payloads = [self._payload(0), self._payload(1)]
        self.assertEqual(run_small_work._payload_contract_errors(self.case, payloads[0], 0, 2, 1, 2, False), [])
        missing = dict(payloads[1]); missing["samples"] = missing["samples"][:1]
        self.assertTrue(run_small_work._payload_contract_errors(self.case, missing, 1, 2, 1, 2, False))
        duplicate = dict(payloads[1]); duplicate["samples"] = [dict(payloads[1]["samples"][0]), dict(payloads[1]["samples"][0])]
        self.assertTrue(run_small_work._payload_contract_errors(self.case, duplicate, 1, 2, 1, 2, False))
        reordered = dict(payloads[1]); reordered["samples"] = list(reversed(payloads[1]["samples"]))
        self.assertTrue(run_small_work._payload_contract_errors(self.case, reordered, 1, 2, 1, 2, False))

    def test_campaign_distinguishes_noise_from_invalid_duration(self) -> None:
        samples = [{"process_index": p, "sample_index": i, "elapsed_ns": (100 if p == 0 else 1000), "iterations": 1}
                    for p in (0, 1) for i in range(2)]
        stats = timing_statistics(samples, 1)
        self.assertGreater(stats["process_median_cov"], 0.1)
        reasons = run_small_work._validate_campaign_timing(stats, target_ns=100, cov_max=0.1)
        self.assertEqual(len(reasons), 1)
        self.assertIn("CoV exceeds noise policy", reasons[0])
        self.assertEqual(run_small_work._validate_campaign_timing(stats, target_ns=100, cov_max=stats["process_median_cov"]), [])
        for cov in (None, True, float("nan"), float("inf")):
            with self.subTest(cov=cov), self.assertRaisesRegex(ContractError, "CoV is unavailable"):
                run_small_work._validate_campaign_timing({**stats, "process_median_cov": cov}, target_ns=100, cov_max=0.1)
        with self.assertRaises(ContractError):
            run_small_work._validate_campaign_timing({**stats, "raw_elapsed_ns": [100, 100, 99, 100]}, target_ns=100, cov_max=1.0)

    def test_suite_protocol_requires_two_processes_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            suite = yaml.safe_load(self._one_case_suite(Path(directory)).read_text())
            for key in ("independent_processes", "samples_per_process"):
                suite["protocol"][key] = 1
                with self.subTest(key=key), self.assertRaises(ContractError):
                    validate_suite_contract(suite)
                suite["protocol"][key] = 2

    def test_suite_noise_policy_is_explicit(self) -> None:
        suite = {"suite_id": "cpu/small_work", "contract_version": 1,
                 "protocol": {"warmups": 0, "runs": 1, "independent_processes": 2,
                               "samples_per_process": 2, "calibration": {"enabled": True, "target_ns": 1},
                               "noise_policy": {"process_median_cov_max": 0.1},
                               "resource_policy": {"requested_threads": 1, "busy_threshold": 1.0}},
                 "cases": []}
        with self.assertRaises(ContractError):
            validate_suite_contract(suite)

    def test_live_cli_inherits_exact_affinity_without_timing(self) -> None:
        if not hasattr(os, "sched_getaffinity") or sys.platform != "linux":
            self.skipTest("Linux affinity smoke only")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            child_output = Path(directory) / "child.txt"
            command = [sys.executable, "-c",
                       ("from pathlib import Path; import os, threading; "
                        "m=','.join(map(str, sorted(os.sched_getaffinity(0)))); "
                        "out=[]; t=threading.Thread(target=lambda: out.append(','.join(map(str, sorted(os.sched_getaffinity(0)))))); "
                        "t.start(); t.join(); Path(%r).write_text(m+'|'+out[0])") % str(child_output)]
            result = None
            completed = None
            for _ in range(3):
                completed = subprocess.run([
                    sys.executable, "scripts/run_small_work.py", "--output", str(output),
                    "--observation-window", "0", "--busy-threshold", "1", "--requested-threads", "1",
                    "--command-timeout", "10", "--command", json.dumps(command)],
                    check=False, capture_output=True, text=True, timeout=20)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads(output.read_text())
                if result["status"] == "READY":
                    break
            assert result is not None
            if result["status"] == "INCONCLUSIVE":
                self.skipTest("live resource telemetry unavailable: " + str(result["resource"]["reasons"]))
            self.assertEqual(result["status"], "READY", result)
            self.assertTrue(result["commands"][0]["affinity_verified"])
            expected = ",".join(map(str, result["resource"]["cpus"]))
            self.assertEqual(child_output.read_text(), expected + "|" + expected)

    def test_sequential_driver_does_not_overlap(self) -> None:
        calls = []
        class Result:
            returncode = 0
            stdout = "out"
            stderr = "err"
        def runner(command, **kwargs):
            calls.append(command)
            return Result()
        with tempfile.TemporaryDirectory() as directory:
            records = run_sequential([["one"], ["two"]], runner=runner,
                                     output_dir=Path(directory), return_records=True)
            self.assertEqual([record["status"] for record in records], ["completed", "completed"])
            self.assertEqual((Path(directory) / "child_000.stdout").read_text(), "out")
        self.assertEqual(calls, [["one"], ["two"]])

    def test_sequential_timeout_archives_output_and_stops_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = run_sequential(
                [[sys.executable, "-c", "print('before', flush=True); import time; time.sleep(10)"]],
                timeout_s=0.1, output_dir=Path(directory), return_records=True)
            self.assertEqual(records[0]["status"], "failed")
            self.assertEqual(records[0]["error"], "timeout")
            self.assertIn("before", (Path(directory) / "child_000.stdout").read_text())

    def test_sequential_driver_retains_failed_and_unrun_records(self) -> None:
        class Failed:
            returncode = 7
        records = run_sequential([["bad"], ["not-run"]], runner=lambda *_args, **_kwargs: Failed(),
                                 return_records=True)
        self.assertEqual([record["status"] for record in records], ["failed", "not_run"])
        self.assertEqual(records[0]["returncode"], 7)

    def _suite_args(self, root: Path, suite: Path, output: Path, *, correctness_only: bool = False):
        return run_small_work.argparse.Namespace(
            suite=suite, case_binary=None, target_profile="amd-cpu",
            results_root=root / "data", result_root=root / "result", tenferro_dir=None,
            tenferro_commit=None, features=[], correctness_only=correctness_only,
            before=None, after=None, synthetic_fixture=False, cpuset=None,
            requested_threads=1, busy_threshold=1.0, observation_window=0,
            command_timeout=1.0, output=output, command=[])

    def _one_case_suite(self, root: Path) -> Path:
        suite = {
            "suite_id": "cpu/small_work", "contract_version": 2,
            "protocol": {"warmups": 0, "runs": 1, "independent_processes": 2,
                          "samples_per_process": 2,
                          "calibration": {"enabled": True, "target_ns": 1},
                          "noise_policy": {"process_median_cov_max": 0.1},
                          "resource_policy": {"requested_threads": 1, "busy_threshold": 1.0}},
            "cases": [{"id": "add", "contract_id": "core.add.ordinary.concrete", "family": "core", "surface": "concrete",
                       "operation": "add", "phase": "execution",
                       "api_tier": "concrete-fresh", "backend": "fixture", "dtype": "f64",
                       "layout": "col_major", "shape": [1], "calls_per_workflow": 1,
                       "workflow": "one", "setup": {"includes": [], "excludes": []},
                       "scope": {"timer": ["backend_call", "output_lifetime"], "outside_timer": ["input_construction"]}}],
        }
        path = root / "suite.yaml"
        path.write_text(yaml.safe_dump(suite), encoding="utf-8")
        return path

    def _valid_live_snapshot(self):
        return {"before": {0: (0, 0)}, "topology": {0: {"package": 0, "core": 0,
                "siblings": [0], "numa": 0}}, "cgroup": {"cpuset": (0,),
                "quota_cpus": 1.0, "paths": ["/cg"], "reasons": []},
                "sched_affinity": (0,), "allowed": (0,)}

    def _payload(self, process: int, *, correctness: str = "passed", case: CaseContract | None = None):
        case = case or self.case
        return {"schema_version": 2, "case_id": case.case_id, "contract_id": case.contract_id,
                "family": case.family, "surface": case.surface, "operation": case.operation, "phase": case.phase,
                "api_tier": case.api_tier, "backend": case.backend, "dtype": case.dtype,
                "layout": case.layout, "shape": list(case.shape), "workflow": case.workflow,
                "calls_per_workflow": case.calls_per_workflow,
                "setup": {"includes": list(case.setup_includes), "excludes": list(case.setup_excludes)},
                "scope": {"timer": list(case.scope_timer), "outside_timer": list(case.scope_outside_timer)},
                "process_index": process, "descriptor_source": "rust_case_selection",
                "correctness_status": correctness,
                "timing_validity": "valid", "provider": "fixture",
                "samples": ([{"process_index": process, "sample_index": i,
                              "elapsed_ns": 100 + i, "iterations": 1} for i in range(2)]
                            if correctness == "passed" else []),
                "timing_reasons": [], "errors": [], "affinity": [0],
                "thread_affinity_observations": {"after_correctness": [{"cpus": [0]}],
                    "after_warmup": [{"cpus": [0]}], "after_timing": [{"cpus": [0]}]},
                "declared_thread_budget": 1, "observed_thread_count": 1,
                "calibration": {"target_ns": 1, "elapsed_ns": 101}}

    def test_suite_gate_honors_cpuset_and_cgroup_reasons(self) -> None:
        snapshot = self._valid_live_snapshot()
        with mock.patch.object(run_small_work, "read_live_resource_snapshot", return_value=snapshot), \
             mock.patch.object(run_small_work, "parse_proc_stat", return_value={0: (1, 0)}), \
             mock.patch.object(run_small_work, "select_idle_cpus", return_value={"status": "valid", "cpus": [0], "effective_threads": 1, "reasons": []}) as select:
            selected = run_small_work._select_live_resources(1, 1.0, 0, "0")
        self.assertEqual(selected["status"], "valid")
        self.assertEqual(select.call_args.args[1], {0})
        snapshot["cgroup"]["reasons"] = ["quota unavailable"]
        with mock.patch.object(run_small_work, "read_live_resource_snapshot", return_value=snapshot):
            selected = run_small_work._select_live_resources(1, 1.0, 0, None)
        self.assertEqual(selected["status"], "inconclusive")
        self.assertIn("quota unavailable", selected["reasons"])
        snapshot["cgroup"] = {"cpuset": (0,), "quota_cpus": None, "quota_known": False,
                                "paths": ["/cg"], "reasons": []}
        with mock.patch.object(run_small_work, "read_live_resource_snapshot", return_value=snapshot):
            selected = run_small_work._select_live_resources(1, 1.0, 0, None)
        self.assertEqual(selected["status"], "inconclusive")

    def test_latest_is_unchanged_when_affinity_restoration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); suite = self._one_case_suite(root); output = root / "run.json"
            latest = root / "result/amd-cpu/cpu/small_work.md"; latest.parent.mkdir(parents=True)
            latest.write_text("previous\n", encoding="utf-8")
            commands = [{"index": i, "command": [], "status": "completed",
                         "returncode": 0, "payload": self._payload(i, case=run_small_work.CaseContract.from_mapping(yaml.safe_load(suite.read_text())["cases"][0]))} for i in range(2)]
            affinity_reads = iter([{0}, {0}, OSError("restore")])
            with mock.patch.object(run_small_work, "_select_live_resources", return_value={"status": "valid", "cpus": [0], "effective_threads": 1, "reasons": []}), \
                 mock.patch.object(run_small_work, "run_sequential", return_value=commands), \
                 mock.patch.object(run_small_work.os, "sched_getaffinity", side_effect=lambda _pid: (lambda value: (_ for _ in ()).throw(value) if isinstance(value, Exception) else value)(next(affinity_reads))), \
                 mock.patch.object(run_small_work.os, "sched_setaffinity"):
                code = run_small_work._suite_run(self._suite_args(root, suite, output))
            self.assertEqual(code, 0)
            self.assertEqual(latest.read_text(encoding="utf-8"), "previous\n")
            self.assertEqual(json.loads(output.read_text())['status'], "INCONCLUSIVE")

    def test_filtered_ready_run_archives_selected_report_without_replacing_latest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); suite = self._one_case_suite(root); output = root / "run.json"
            latest = root / "result/amd-cpu/cpu/small_work.md"; latest.parent.mkdir(parents=True)
            previous = "previous full suite\n"
            latest.write_text(previous, encoding="utf-8")
            case = run_small_work.CaseContract.from_mapping(yaml.safe_load(suite.read_text())["cases"][0])
            commands = [{"index": i, "command": [], "status": "completed", "returncode": 0,
                         "payload": self._payload(i, case=case)} for i in range(2)]
            args = self._suite_args(root, suite, output)
            args.case_binary = root / "fixture-child"
            args.preparation_receipt = root / "receipt.json"
            args.preparation_receipt.write_text("{}", encoding="utf-8")
            with mock.patch.dict(os.environ, {"SMALL_WORK_CASE_FILTER": "add"}), \
                 mock.patch.object(run_small_work, "_select_live_resources", return_value={"status": "valid", "cpus": [0], "effective_threads": 1, "reasons": []}), \
                 mock.patch.object(run_small_work, "run_sequential", return_value=commands), \
                 mock.patch.object(run_small_work, "verify_preparation_receipt", return_value=[]), \
                 mock.patch.object(run_small_work, "verify_canonical_snapshot"), \
                 mock.patch.object(run_small_work.os, "sched_getaffinity", return_value={0}), \
                 mock.patch.object(run_small_work.os, "sched_setaffinity"):
                self.assertEqual(run_small_work._suite_run(args), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "READY")
            self.assertEqual(result["selection"], {"case_ids": ["add"], "filter": "add"})
            self.assertEqual(latest.read_text(encoding="utf-8"), previous)
            archive = next((root / "data/amd-cpu/cpu/small_work").iterdir())
            report = (archive / "report.md").read_text(encoding="utf-8")
            self.assertIn("Selected-case archive", report)
            self.assertIn("`add`", report)
            archived = json.loads((archive / "run.json").read_text())
            self.assertEqual(archived["selection"]["case_ids"], ["add"])
            metadata = yaml.safe_load((archive / "run.yaml").read_text())
            self.assertEqual(metadata["small_work"]["selection"]["case_ids"], ["add"])

    def test_unfiltered_ready_run_publishes_latest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); suite = self._one_case_suite(root); output = root / "run.json"
            latest = root / "result/amd-cpu/cpu/small_work.md"; latest.parent.mkdir(parents=True)
            latest.write_text("previous\n", encoding="utf-8")
            case = run_small_work.CaseContract.from_mapping(yaml.safe_load(suite.read_text())["cases"][0])
            commands = [{"index": i, "command": [], "status": "completed", "returncode": 0,
                         "payload": self._payload(i, case=case)} for i in range(2)]
            args = self._suite_args(root, suite, output)
            args.case_binary = root / "fixture-child"
            args.preparation_receipt = root / "receipt.json"
            args.preparation_receipt.write_text("{}", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(run_small_work, "_select_live_resources", return_value={"status": "valid", "cpus": [0], "effective_threads": 1, "reasons": []}), \
                 mock.patch.object(run_small_work, "run_sequential", return_value=commands), \
                 mock.patch.object(run_small_work, "verify_preparation_receipt", return_value=[]), \
                 mock.patch.object(run_small_work, "verify_canonical_snapshot"), \
                 mock.patch.object(run_small_work.os, "sched_getaffinity", return_value={0}), \
                 mock.patch.object(run_small_work.os, "sched_setaffinity"):
                self.assertEqual(run_small_work._suite_run(args), 0)
            self.assertEqual(json.loads(output.read_text())["status"], "READY")
            self.assertNotEqual(latest.read_text(encoding="utf-8"), "previous\n")

    def test_campaign_failures_keep_previous_latest(self) -> None:
        variants = ("valid", "missing", "duplicate", "truncated", "reordered", "mismatched", "high_cov", "noise_and_correctness", "noise_and_missing", "under_duration")
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); suite = self._one_case_suite(root); output = root / "run.json"
                case = run_small_work.CaseContract.from_mapping(yaml.safe_load(suite.read_text())["cases"][0])
                first, second = self._payload(0, case=case), self._payload(1, case=case)
                if variant == "missing":
                    commands = [{"index": 0, "command": [], "status": "completed", "returncode": 0, "payload": first}]
                else:
                    if variant == "duplicate":
                        second["samples"] = [dict(second["samples"][0]), dict(second["samples"][0])]
                    elif variant == "truncated":
                        second["samples"] = second["samples"][:1]
                    elif variant == "reordered":
                        second["samples"] = list(reversed(second["samples"]))
                    elif variant == "mismatched":
                        second["operation"] = "mul"
                    elif variant in ("high_cov", "noise_and_correctness", "noise_and_missing"):
                        for sample in second["samples"]:
                            sample["elapsed_ns"] = 1000
                        if variant == "noise_and_correctness":
                            second["correctness_status"] = "failed"
                        elif variant == "noise_and_missing":
                            second["samples"].pop()
                    elif variant == "under_duration":
                        second["samples"][0]["elapsed_ns"] = 0
                    commands = [{"index": 0, "command": [], "status": "completed", "returncode": 0, "payload": first},
                                {"index": 1, "command": [], "status": "completed", "returncode": 0, "payload": second}]
                latest = root / "result/amd-cpu/cpu/small_work.md"; latest.parent.mkdir(parents=True)
                latest.write_text("previous\n", encoding="utf-8")
                args = self._suite_args(root, suite, output)
                args.case_binary = root / "fixture-child"
                args.preparation_receipt = root / "receipt.json"
                args.preparation_receipt.write_text("{}", encoding="utf-8")
                with mock.patch.object(run_small_work, "_select_live_resources", return_value={"status": "valid", "cpus": [0], "effective_threads": 1, "reasons": []}), \
                     mock.patch.object(run_small_work, "verify_preparation_receipt", return_value=[]), \
                     mock.patch.object(run_small_work, "run_sequential", return_value=commands) as run_children, \
                     mock.patch.object(run_small_work.os, "sched_getaffinity", return_value={0}), \
                     mock.patch.object(run_small_work.os, "sched_setaffinity"):
                    code = run_small_work._suite_run(args)
                run_children.assert_called_once()
                if variant == "valid":
                    self.assertEqual(json.loads(output.read_text())["status"], "READY")
                    self.assertNotEqual(latest.read_text(encoding="utf-8"), "previous\n")
                    self.assertEqual(code, 0)
                elif variant == "high_cov":
                    result = json.loads(output.read_text())
                    self.assertEqual(result["status"], "INCONCLUSIVE")
                    self.assertEqual(code, 0)
                    self.assertEqual(latest.read_text(encoding="utf-8"), "previous\n")
                    record = result["records"][0]
                    self.assertEqual(record["samples"], first["samples"] + second["samples"])
                    self.assertEqual(record["timing_validity"]["status"], "inconclusive")
                    self.assertGreater(record["timing"]["process_median_cov"], 0.1)
                    self.assertEqual(record["errors"], [])
                    validate_record(record, case)
                    archive = next((root / "data/amd-cpu/cpu/small_work").iterdir())
                    self.assertEqual(json.loads((archive / "results.jsonl").read_text()), record)
                    self.assertIn("| inconclusive | — | — |", (archive / "report.md").read_text())
                else:
                    self.assertEqual(json.loads(output.read_text())["status"], "FAILED")
                    self.assertEqual(latest.read_text(encoding="utf-8"), "previous\n")
                    self.assertNotEqual(code, 0)

    def test_correctness_only_child_failure_is_nonzero_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); suite = self._one_case_suite(root); output = root / "run.json"
            commands = [{"index": 0, "command": [], "status": "completed", "returncode": 0,
                         "payload": self._payload(0, correctness="failed")}]
            with mock.patch.object(run_small_work, "run_sequential", return_value=commands):
                code = run_small_work._suite_run(self._suite_args(root, suite, output, correctness_only=True))
            self.assertNotEqual(code, 0)
            self.assertEqual(json.loads(output.read_text())['status'], "FAILED")

    def test_correctness_only_rejects_invalid_success_payload(self) -> None:
        for variant in ("valid", "wrong_operation", "child_failure"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                suite = self._one_case_suite(root)
                output = root / "run.json"
                case = CaseContract.from_mapping(yaml.safe_load(suite.read_text())["cases"][0])
                payload = self._payload(0, case=case)
                payload.update(samples=[], timing_validity="inconclusive")
                command = {"status": "completed", "returncode": 0, "payload": payload}
                if variant == "wrong_operation":
                    payload["operation"] = "mul"
                elif variant == "child_failure":
                    command.update(status="failed", returncode=1, error="child failed")
                args = self._suite_args(root, suite, output, correctness_only=True)
                args.case_binary = root / "fixture-child"
                with mock.patch.object(run_small_work, "run_sequential", return_value=[command]), \
                     mock.patch.object(run_small_work, "verify_canonical_snapshot"):
                    code = run_small_work._suite_run(args)
                result = json.loads(output.read_text())
                if variant == "valid":
                    self.assertEqual(code, 0)
                    self.assertEqual(result["status"], "CORRECTNESS_ONLY")
                else:
                    self.assertNotEqual(code, 0)
                    self.assertEqual(result["status"], "FAILED")
                    self.assertTrue(result["records"][0]["errors"])

    def test_suite_exception_still_writes_run_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); suite = self._one_case_suite(root); output = root / "run.json"
            with mock.patch.object(run_small_work, "_select_live_resources", return_value={"status": "valid", "cpus": [0], "effective_threads": 1, "reasons": []}), \
                 mock.patch.object(run_small_work, "run_sequential", side_effect=RuntimeError("launch")), \
                 mock.patch.object(run_small_work.os, "sched_getaffinity", return_value={0}), \
                 mock.patch.object(run_small_work.os, "sched_setaffinity"):
                code = run_small_work._suite_run(self._suite_args(root, suite, output))
            self.assertNotEqual(code, 0)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["status"], "FAILED")
            self.assertTrue(payload["errors"])

    def test_pin_and_child_parse_failures_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); suite = self._one_case_suite(root); output = root / "run.json"
            with mock.patch.object(run_small_work, "_select_live_resources", return_value={"status": "valid", "cpus": [0], "effective_threads": 1, "reasons": []}), \
                 mock.patch.object(run_small_work.os, "sched_getaffinity", return_value={0}), \
                 mock.patch.object(run_small_work.os, "sched_setaffinity", side_effect=[OSError("pin"), None]):
                self.assertNotEqual(run_small_work._suite_run(self._suite_args(root, suite, output)), 0)
            self.assertEqual(json.loads(output.read_text())["status"], "FAILED")
            malformed = [{"index": 0, "command": [], "status": "completed", "returncode": 0,
                          "payload": {"case_id": "add", "operation": "add", "samples": "bad"}}]
            output.unlink()
            with mock.patch.object(run_small_work, "_select_live_resources", return_value={"status": "valid", "cpus": [0], "effective_threads": 1, "reasons": []}), \
                 mock.patch.object(run_small_work, "run_sequential", return_value=malformed), \
                 mock.patch.object(run_small_work.os, "sched_getaffinity", return_value={0}), \
                 mock.patch.object(run_small_work.os, "sched_setaffinity"):
                self.assertNotEqual(run_small_work._suite_run(self._suite_args(root, suite, output)), 0)
            self.assertEqual(json.loads(output.read_text())["status"], "FAILED")

    def test_correctness_only_suite_cli_ingests_stdout_without_timing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child.py"
            child.write_text(
                "#!/usr/bin/env python3\n"
                "import json,sys\n"
                "case=sys.argv[sys.argv.index('--case')+1]\n"
                "print(json.dumps({'schema_version':2,'case_id':case,'contract_id':'core.add.ordinary.concrete','family':'core','surface':'concrete','operation':'add','phase':'execution','api_tier':'concrete-fresh','backend':'tenferro-rs','dtype':'f64','layout':'col_major_contiguous','shape':[4],'workflow':'single','calls_per_workflow':1,'setup':{'includes':['session_entry_exit','add','output_lifetime'],'excludes':['backend_construction','input_construction','correctness_check']},'scope':{'timer':['session_entry_exit','add','output_lifetime'],'outside_timer':['backend_construction','input_construction','correctness_check']},'descriptor_source':'rust_case_selection','process_index':0,'correctness_status':'passed','timing_validity':'inconclusive','provider':'fixture','samples':[],'affinity':[],'thread_affinity_observations':{},'declared_thread_budget':1,'observed_thread_count':0,'timing_reasons':['correctness-only'],'errors':[]}))\n",
                encoding="utf-8")
            child.chmod(0o755)
            output = root / "run.json"
            environment = {**os.environ, "SMALL_WORK_CASE_FILTER": "add_f64_concrete_fresh"}
            completed = subprocess.run([
                sys.executable, "scripts/run_small_work.py", "--suite", "benchmarks/cpu/small_work.yaml",
                "--case-binary", str(child), "--correctness-only", "--results-root", str(root / "data"),
                "--result-root", str(root / "result"), "--output", str(output), "--observation-window", "0",
            ], env=environment, check=False, capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["status"], "CORRECTNESS_ONLY")
            self.assertTrue(all(record["timing"] is None for record in payload["records"]))
            self.assertFalse((root / "result" / "amd-cpu" / "cpu" / "small_work.md").exists())


if __name__ == "__main__":
    unittest.main()
