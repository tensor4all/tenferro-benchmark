#!/usr/bin/env python3
"""Run cpu/small_work cases serially with frozen provenance and raw outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import yaml

from small_work import (
    ContractError,
    CaseContract,
    cpu_busy_fraction,
    load_json,
    make_record,
    parse_cpu_set,
    parse_proc_stat,
    read_live_resource_snapshot,
    render_report,
    run_sequential,
    select_idle_cpus,
    timing_statistics,
    validate_suite_contract,
    resolve_case_filter,
    select_changed_cases,
    verify_canonical_snapshot,
    verify_canonical_binding,
    verify_affinity,
)
from small_work_provenance import prepare_cargo, verify_preparation_receipt


def _commands(values: list[str]) -> list[list[str]]:
    commands = []
    for value in values:
        command = json.loads(value)
        if (not isinstance(command, list) or not command or
                not all(isinstance(item, str) and item for item in command)):
            raise SystemExit("--command must be a JSON argv list of strings")
        commands.append(command)
    return commands


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resource_for_metadata(selection: dict[str, Any]) -> dict[str, Any]:
    keys = ("status", "requested_threads", "effective_threads", "cpus", "reasons",
            "busy_threshold", "quota_cpus", "affinity_verified", "smt_policy")
    return {key: selection[key] for key in keys if key in selection}


def _select_live_resources(requested_threads: int, busy_threshold: float,
                           observation_window: float, cpuset: str | None) -> dict[str, Any]:
    """Apply one live resource gate for both command and suite runners."""
    live = read_live_resource_snapshot()
    if observation_window:
        time.sleep(observation_window)
    after = parse_proc_stat(Path("/proc/stat").read_text(encoding="utf-8"))
    allowed = set(live["allowed"])
    requested = parse_cpu_set(cpuset) if cpuset is not None else None
    outside = (set(requested) - allowed) if requested is not None else set()
    if requested is not None:
        allowed &= set(requested)
    if outside:
        selection = {"status": "inconclusive", "cpus": [], "effective_threads": 0,
                     "reasons": ["explicit cpuset includes CPUs outside live allowed mask"]}
    elif live["cgroup"].get("reasons"):
        selection = {"status": "inconclusive", "cpus": [], "effective_threads": 0,
                     "reasons": list(live["cgroup"]["reasons"])}
    elif (live["cgroup"].get("quota_cpus") is None and
          not live["cgroup"].get("quota_known", False)):
        selection = {"status": "inconclusive", "cpus": [], "effective_threads": 0,
                     "reasons": ["CPU quota telemetry is unavailable"]}
    else:
        selection = select_idle_cpus(
            cpu_busy_fraction(live["before"], after), allowed, requested_threads,
            busy_threshold=busy_threshold, quota_cpus=live["cgroup"]["quota_cpus"],
            topology=live["topology"], require_topology=True)
    selection.update({"allowed_cpus": sorted(allowed), "sched_affinity": list(live["sched_affinity"]),
                      "cgroup_paths": live["cgroup"]["paths"], "source": "live_proc_sysfs_cgroup",
                      "requested_threads": requested_threads, "busy_threshold": busy_threshold,
                      "quota_cpus": live["cgroup"]["quota_cpus"], "smt_policy": "whole-core"})
    return selection


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _archive_result(root: Path, output: Path | None, result: Mapping[str, Any]) -> None:
    text = json.dumps(result, indent=2, default=str) + "\n"
    _atomic_write(root / "run.json", text)
    if output is not None and output != root / "run.json":
        _atomic_write(output, text)


def _write_frozen_metadata(path: Path, *, args: argparse.Namespace,
                           suite_path: Path, timestamp: str,
                           selection: dict[str, Any], protocol: dict[str, Any],
                           selected_case_ids: list[str] | None = None,
                           case_filter: str | None = None,
                           provenance: Mapping[str, Any] | None = None,
                           binding: Mapping[str, Any] | None = None) -> None:
    checkout = binding.get("checkout", {}) if isinstance(binding, Mapping) else {}
    features = []
    if isinstance(provenance, Mapping):
        receipt = provenance.get("receipt_data")
        build = receipt.get("build") if isinstance(receipt, Mapping) else None
        if isinstance(build, Mapping) and isinstance(build.get("requested_features"), list):
            features = list(build["requested_features"])
    metadata = {
        "schema_version": 1,
        "target_profile": args.target_profile,
        "suite_id": "cpu/small_work",
        "suite_file": str(suite_path),
        "timestamp": timestamp,
        "tenferro_rs": {"path": checkout.get("path", "unknown"),
                        "commit": checkout.get("head"),
                        "dirty": checkout.get("dirty"), "features": features},
        "environment": {"hostname": os.uname().nodename if hasattr(os, "uname") else None,
                        "os": sys.platform, "arch": os.uname().machine if hasattr(os, "uname") else None},
        "small_work": {"contract_version": 2, "protocol": protocol,
                       "resource": _resource_for_metadata(selection),
                       "selection": {"case_ids": selected_case_ids or [], "filter": case_filter},
                       "canonical_binding": dict(binding) if binding is not None else None},
    }
    if provenance is not None:
        metadata["small_work"]["provenance"] = {
            key: value for key, value in provenance.items() if key != "receipt_data"
        }
    _atomic_write(path, yaml.safe_dump(metadata, sort_keys=False))


def _payload_contract_errors(case: CaseContract, payload: Mapping[str, Any], process_index: int,
                             samples_per_process: int, target_ns: int,
                             process_count: int, correctness_only: bool) -> list[str]:
    """Validate one child receipt against the declared case, never its echoes."""
    errors: list[str] = []
    required = ("schema_version", "case_id", "contract_id", "family", "surface", "operation", "phase", "api_tier", "backend", "provider",
                "dtype", "layout", "shape", "workflow", "calls_per_workflow", "setup",
                "scope", "descriptor_source", "process_index", "correctness_status", "timing_validity",
                "timing_reasons", "errors", "samples", "affinity",
                "thread_affinity_observations", "declared_thread_budget", "observed_thread_count")
    errors.extend(f"missing child field {key}" for key in required if key not in payload)
    if errors:
        return errors
    if payload.get("schema_version") != 2:
        errors.append("child schema_version is not 2")
    expected = {"case_id": case.case_id, "contract_id": case.contract_id, "family": case.family,
                "surface": case.surface, "operation": case.operation, "phase": case.phase,
                "api_tier": case.api_tier, "backend": case.backend, "dtype": case.dtype,
                "layout": case.layout, "shape": list(case.shape), "workflow": case.workflow,
                "calls_per_workflow": case.calls_per_workflow,
                "setup": {"includes": list(case.setup_includes), "excludes": list(case.setup_excludes)},
                "scope": {"timer": list(case.scope_timer), "outside_timer": list(case.scope_outside_timer)}}
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"child {key} does not match declared case")
    if payload.get("descriptor_source") != "rust_case_selection":
        errors.append("child descriptors are not from Rust case selection")
    if type(payload["provider"]) is not str or not payload["provider"] or payload["provider"] == "unknown":
        errors.append("child provider identity is missing")
    if payload.get("process_index") != process_index:
        errors.append("child process identity does not match command")
    if payload.get("correctness_status") != "passed":
        errors.append("child correctness did not pass")
    if payload.get("errors") != [] and not correctness_only:
        errors.append("child reported errors")
    samples = payload["samples"]
    if not isinstance(samples, list):
        return errors + ["child samples are not a list"]
    if correctness_only:
        if samples:
            errors.append("correctness-only child must not report timing samples")
        if payload.get("timing_validity") != "inconclusive":
            errors.append("correctness-only timing must be inconclusive")
        return errors
    if payload.get("timing_validity") != "valid":
        errors.append("child timing is not valid")
    if len(samples) != samples_per_process:
        errors.append("child sample count does not match protocol")
    expected_ids = {(process_index, index) for index in range(samples_per_process)}
    actual_ids: set[tuple[int, int]] = set()
    ordered_ids: list[tuple[int, int]] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            errors.append("child sample is not an object")
            continue
        process = sample.get("process_index")
        index = sample.get("sample_index")
        elapsed = sample.get("elapsed_ns")
        iterations = sample.get("iterations")
        if type(process) is not int or type(index) is not int:
            errors.append("child sample identity is invalid")
            continue
        identity = (process, index)
        if identity in actual_ids:
            errors.append("duplicate child sample identity")
        actual_ids.add(identity)
        ordered_ids.append(identity)
        if type(elapsed) is not int or elapsed <= 0 or elapsed < target_ns:
            errors.append("child sample duration is below calibration target")
        if type(iterations) is not int or iterations < 1:
            errors.append("child sample iterations are invalid")
    if actual_ids != expected_ids or ordered_ids != sorted(expected_ids):
        errors.append("child sample identities are incomplete or reordered")
    calibration = payload.get("calibration")
    if (not isinstance(calibration, Mapping) or calibration.get("target_ns") != target_ns or
            type(calibration.get("elapsed_ns")) is not int or calibration["elapsed_ns"] < target_ns):
        errors.append("child calibration receipt is missing or below target")
    observations = payload.get("thread_affinity_observations")
    if not isinstance(observations, Mapping) or any(
            not isinstance(observations.get(stage), list) or not observations[stage]
            for stage in ("after_correctness", "after_warmup", "after_timing")):
        errors.append("child thread-affinity receipts are incomplete")
    return errors


def _validate_campaign_timing(timing: Mapping[str, Any], *, target_ns: int,
                              cov_max: float) -> list[str]:
    """Reject malformed evidence; return noise reasons without discarding samples."""
    raw = timing.get("raw_elapsed_ns")
    if not isinstance(raw, list) or not raw or any(type(value) is not int or value < target_ns for value in raw):
        raise ContractError("campaign contains an under-duration sample")
    cov = timing.get("process_median_cov")
    if type(cov) not in (int, float) or isinstance(cov, bool) or not math.isfinite(float(cov)):
        raise ContractError("campaign process-median CoV is unavailable")
    if float(cov) > cov_max:
        return [f"campaign process-median CoV exceeds noise policy ({cov} > {cov_max})"]
    return []


def _archive_receipt(root: Path, receipt: Mapping[str, Any]) -> dict[str, str]:
    path = root / "preparation_receipt.json"
    text = json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n"
    _atomic_write(path, text)
    import hashlib
    return {"file": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def _observe_suite_resources(args: argparse.Namespace, protocol: Mapping[str, Any]) -> dict[str, Any]:
    policy = protocol["resource_policy"]
    try:
        return _select_live_resources(policy["requested_threads"], policy["busy_threshold"],
                                      args.observation_window, args.cpuset)
    except Exception as exc:
        return {"status": "inconclusive", "cpus": [], "effective_threads": 0,
                "requested_threads": policy["requested_threads"],
                "reasons": [f"live resource observation failed: {type(exc).__name__}"]}


def _suite_run(args: argparse.Namespace) -> int:
    suite_path = args.suite
    timestamp = _timestamp()
    root = args.results_root / args.target_profile / "cpu" / "small_work" / timestamp
    root.mkdir(parents=True, exist_ok=True)
    output = args.output
    library = getattr(args, "tenferro_dir", None) or Path(__file__).resolve().parents[1] / "extern" / "tenferro-rs"
    change_selection = None
    try:
        suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
        all_cases = validate_suite_contract(suite)
        protocol = suite["protocol"]
        cases, case_filter = resolve_case_filter(all_cases, os.environ.get("SMALL_WORK_CASE_FILTER"))
        if getattr(args, "changed_path", []):
            if case_filter is not None:
                raise ContractError("--changed-path cannot be combined with SMALL_WORK_CASE_FILTER")
            cases, change_selection = select_changed_cases(library, all_cases, args.changed_path)
            case_filter = ",".join(case.case_id for case in cases)
            _atomic_write(root / "change_selection.json", json.dumps(change_selection, indent=2) + "\n")
    except Exception as exc:
        result = {"schema_version": 2, "suite_id": "cpu/small_work", "status": "FAILED",
                  "synthetic_fixture": False, "commands": [], "records": [],
                  "errors": [f"suite validation failed: {type(exc).__name__}: {exc}"]}
        _archive_result(root, output, result)
        if getattr(args, "dry_run", False):
            print(json.dumps(result, indent=2))
        return 1
    selection_incomplete = change_selection is not None and (bool(change_selection["missing_contract_ids"]) or not cases)
    if selection_incomplete and not getattr(args, "dry_run", False):
        result = {"schema_version": 2, "suite_id": suite["suite_id"], "status": "INCONCLUSIVE",
                  "commands": [], "records": [], "change_selection": change_selection,
                  "errors": ["changed-path selection has missing suite contracts or no executable cases"]}
        _archive_result(root, output, result)
        return 1
    if getattr(args, "dry_run", False):
        selection = _observe_suite_resources(args, protocol)
        result = {"schema_version": 2, "suite_id": suite["suite_id"], "mode": "dry-run",
                  "status": "DRY_RUN" if selection["status"] == "valid" and not selection_incomplete else "INCONCLUSIVE",
                  "timestamp": timestamp, "suite_file": str(suite_path),
                  "selection_source": "canonical changed-path selector" if change_selection is not None else "manual suite and SMALL_WORK_CASE_FILTER",
                  "change_selection": change_selection,
                  "case_filter": case_filter,
                  "cases": [{"id": case.case_id, "contract_id": case.contract_id} for case in cases],
                  "protocol": protocol, "resource": selection, "commands": [],
                  "not_checked": ["canonical_binding", "build_provenance", "correctness", "allocation", "timing"]}
        _archive_result(root, output, result)
        print(json.dumps(result, indent=2))
        return 0
    try:
        binding = verify_canonical_binding(library, cases, check_inventory=change_selection is None)
        if change_selection is not None:
            verify_canonical_snapshot(change_selection["source"], library, all_cases)
    except (OSError, ValueError, ContractError) as exc:
        details = getattr(exc, "details", {})
        records = [make_record(case, [], correctness_status="not_run",
                                timing_status="inconclusive",
                                errors=[f"canonical binding failed: {type(exc).__name__}: {exc}"])
                   for case in cases]
        result = {"schema_version": 2, "suite_id": "cpu/small_work",
                  "status": "FAILED" if args.correctness_only else "INCONCLUSIVE",
                  "synthetic_fixture": False, "commands": [], "records": records,
                  "errors": [f"canonical binding failed: {type(exc).__name__}: {exc}"],
                  "binding_error": details, "timestamp": timestamp}
        _archive_result(root, output, result)
        return 1
    requested = protocol["resource_policy"]["requested_threads"]
    selection: dict[str, Any]
    if args.correctness_only:
        selection = {"status": "inconclusive", "cpus": [], "effective_threads": 0,
                     "requested_threads": requested,
                     "reasons": ["correctness-only mode does not require idle CPUs"]}
    else:
        selection = _observe_suite_resources(args, protocol)
    receipt_path = getattr(args, "preparation_receipt", None)
    provenance: dict[str, Any] = {"status": "unverified", "reasons": ["no Cargo preparation receipt"]}
    provenance_errors: list[str] = list(provenance["reasons"])
    receipt: Mapping[str, Any] | None = None
    if receipt_path is not None:
        try:
            receipt_value = load_json(Path(receipt_path))
            if not isinstance(receipt_value, Mapping):
                raise ValueError("preparation receipt is not an object")
            receipt = receipt_value
            provenance_errors = verify_preparation_receipt(
                receipt, benchmark=Path(__file__).resolve().parents[1],
                library=args.tenferro_dir or Path(__file__).resolve().parents[1] / "extern" / "tenferro-rs",
                binary=args.case_binary, require_timing=not args.correctness_only, cargo=getattr(args, "cargo", "cargo"),
                build_project=getattr(args, "build_project", None), package=getattr(args, "cargo_package", None),
                target_name=getattr(args, "cargo_target_name", None), artifact_kind=getattr(args, "artifact_kind", "bin"))
            provenance = {"status": "verified" if not provenance_errors else "unverified",
                          "reasons": provenance_errors, "receipt_data": receipt}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            provenance_errors = [f"preparation receipt unavailable: {type(exc).__name__}"]
            provenance = {"status": "unverified", "reasons": provenance_errors}
    receipt_ref = None
    if receipt is not None:
        receipt_ref = _archive_receipt(root, receipt)
    if receipt_ref is not None:
        provenance["receipt"] = receipt_ref
    result: dict[str, Any] = {"schema_version": 2, "suite_id": "cpu/small_work",
                              "status": "INCONCLUSIVE", "resource": selection,
                              "selection": {"case_ids": [case.case_id for case in cases], "filter": case_filter},
                              "provenance": {k: v for k, v in provenance.items() if k != "receipt_data"},
                              "canonical_binding": binding,
                              "synthetic_fixture": False,
                              "commands": [], "records": [], "errors": [], "timestamp": timestamp}
    if args.tenferro_commit is not None and args.tenferro_commit != binding.get("checkout_head"):
        message = "claimed tenferro commit does not match observed checkout HEAD"
        result["errors"].append(message)
        result["status"] = "FAILED" if args.correctness_only else "INCONCLUSIVE"
        _archive_result(root, output, result)
        return 1
    try:
        _write_frozen_metadata(root / "run.yaml", args=args, suite_path=suite_path,
                               timestamp=timestamp, selection=selection, protocol=protocol,
                               selected_case_ids=[case.case_id for case in cases],
                               case_filter=case_filter, provenance=provenance, binding=binding)
    except Exception as exc:
        result["errors"].append(f"metadata freeze failed: {type(exc).__name__}")
        _archive_result(root, output, result)
        return 1
    if selection["status"] != "valid" and not args.correctness_only:
        _archive_result(root, output, result)
        return 0
    # A production timing child may only be launched from a verified Cargo receipt.
    if args.case_binary is not None and not args.correctness_only and provenance_errors:
        result["errors"].extend(provenance_errors)
        result["status"] = "INCONCLUSIVE"
        _archive_result(root, output, result)
        return 0

    report_text: str | None = None
    publish = False
    original: set[int] | None = None
    expected = set(selection.get("cpus", [])) if selection["status"] == "valid" else None
    try:
        binary = args.case_binary
        process_count = 1 if args.correctness_only else protocol["independent_processes"]
        samples_per_process = 1 if args.correctness_only else protocol["samples_per_process"]
        commands: list[list[str]] = []
        command_cases: list[int] = []
        if binary:
            for case_index, case in enumerate(cases):
                for process_index in range(process_count):
                    commands.append([str(binary), "--case", case.case_id,
                                     "--operation", case.operation, "--dtype", case.dtype,
                                     "--api-tier", case.api_tier, "--workflow", case.workflow,
                                     "--layout", case.layout,
                                     "--mode", "correctness-only" if args.correctness_only else "measure",
                                     "--size", str(max(1, int(math.prod(case.shape)))),
                                     "--warmups", str(protocol["warmups"]), "--samples", str(samples_per_process),
                                     "--target-ns", str(protocol["calibration"]["target_ns"]),
                                     "--calls", str(case.calls_per_workflow), "--process-index", str(process_index),
                                     "--sample-start", "0",
                                     "--declared-threads", str(requested),
                                     "--expected-cpus", ",".join(str(cpu) for cpu in sorted(selection.get("cpus", [])))])
                    command_cases.append(case_index)
        else:
            commands = _commands(args.command)
            command_cases = list(range(len(commands)))
        if expected is not None:
            original = set(os.sched_getaffinity(0))
            os.sched_setaffinity(0, expected)
            verify_affinity(expected, os.sched_getaffinity(0))
        verify_canonical_snapshot(binding, library, cases)
        command_records = run_sequential(commands, expected_affinity=expected,
                                         timeout_s=args.command_timeout, output_dir=root / "children",
                                         require_json=bool(binary), return_records=True,
                                         observe_machine=not args.correctness_only)
        result["commands"] = command_records
        records = []
        cov_max = float(protocol["noise_policy"]["process_median_cov_max"])
        target_ns = protocol["calibration"]["target_ns"]
        for case_index, case in enumerate(cases):
            child_indexes = [index for index, mapped in enumerate(command_cases) if mapped == case_index]
            errors: list[str] = []
            if len(child_indexes) != process_count:
                errors.append(f"expected {process_count} children, received {len(child_indexes)}")
            payloads: list[Mapping[str, Any]] = []
            for expected_process, index in enumerate(child_indexes[:process_count]):
                command_record = command_records[index] if index < len(command_records) else {}
                if command_record.get("status") != "completed":
                    errors.append(str(command_record.get("error", "child failed")))
                payload = command_record.get("payload")
                if isinstance(payload, Mapping):
                    payloads.append(payload)
                    errors.extend(_payload_contract_errors(case, payload, expected_process,
                                                           samples_per_process, target_ns,
                                                           process_count, args.correctness_only))
                else:
                    errors.append("child produced no result payload")
            if not payloads:
                record = make_record(case, [], correctness_status="not_run",
                                     timing_status="inconclusive", errors=errors or ["child produced no result"])
                records.append(record)
                continue
            providers = {payload.get("provider") for payload in payloads}
            if len(providers) != 1:
                errors.append("child provider identities are inconsistent")
            correctness_values = {payload.get("correctness_status") for payload in payloads}
            correctness = "passed" if correctness_values == {"passed"} else (
                "failed" if "failed" in correctness_values else "not_run")
            samples = [sample for payload in payloads for sample in payload.get("samples", [])]
            reasons = [reason for payload in payloads for reason in payload.get("timing_reasons", [])
                       if isinstance(reason, str)]
            provider = next(iter(providers), "unknown")
            if errors:
                timing_status = "inconclusive" if args.correctness_only else "invalid"
                reasons.extend(errors)
                samples_for_record = []
            else:
                samples_for_record = [] if args.correctness_only else samples
                timing_status = "inconclusive" if args.correctness_only else "valid"
            try:
                record = make_record(case, samples_for_record, correctness_status=correctness,
                                     timing_status=timing_status, timing_reasons=reasons,
                                     provider=provider, errors=errors)
                if not args.correctness_only and not errors:
                    noise = _validate_campaign_timing(record["timing"], target_ns=target_ns, cov_max=cov_max)
                    if noise:
                        record["timing_validity"] = {"status": "inconclusive", "reasons": reasons + noise}
            except (TypeError, ValueError, ContractError) as exc:
                record = make_record(case, [] if not args.correctness_only else samples_for_record,
                                     correctness_status=correctness,
                                     timing_status="inconclusive" if args.correctness_only else "invalid",
                                     timing_reasons=reasons + [str(exc)], provider=provider,
                                     errors=errors + [f"invalid campaign receipt: {type(exc).__name__}"])
            records.append(record)
        result["records"] = records
        (root / "results.jsonl").write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")
        if args.correctness_only:
            complete = all(record["correctness"]["status"] == "passed" and not record["errors"]
                           for record in records)
            result["status"] = "CORRECTNESS_ONLY" if complete else "FAILED"
        else:
            report_text = render_report(records)
            if case_filter is not None:
                report_text = "# Selected-case archive\n\n" + report_text
            (root / "report.md").write_text(report_text, encoding="utf-8")
            failed = any(record["correctness"]["status"] != "passed" or record["errors"] or
                         record["timing_validity"]["status"] == "invalid" for record in records)
            publish = not failed and all(record["timing_validity"]["status"] == "valid" for record in records)
            result["status"] = "FAILED" if failed else ("READY" if publish else "INCONCLUSIVE")
    except Exception as exc:
        result["status"] = "FAILED"
        result["errors"].append(f"suite execution failed: {type(exc).__name__}")
    finally:
        if original is not None:
            try:
                os.sched_setaffinity(0, original)
                verify_affinity(original, os.sched_getaffinity(0))
            except Exception as exc:
                result["status"] = "INCONCLUSIVE"
                publish = False
                result["errors"].append(f"affinity restoration failed: {type(exc).__name__}")
    if publish and result["status"] == "READY" and report_text is not None:
        try:
            verify_canonical_snapshot(binding, library, cases)
        except (OSError, ValueError, ContractError) as exc:
            publish = False
            result["status"] = "INCONCLUSIVE"
            result["errors"].append(f"canonical binding changed: {type(exc).__name__}")
    if publish and result["status"] == "READY" and report_text is not None and args.case_binary is not None:
        final_errors = verify_preparation_receipt(
            receipt or {}, benchmark=Path(__file__).resolve().parents[1],
            library=args.tenferro_dir or Path(__file__).resolve().parents[1] / "extern" / "tenferro-rs",
            binary=args.case_binary, require_timing=True, cargo=getattr(args, "cargo", "cargo"),
            build_project=getattr(args, "build_project", None), package=getattr(args, "cargo_package", None),
            target_name=getattr(args, "cargo_target_name", None), artifact_kind=getattr(args, "artifact_kind", "bin"))
        if final_errors:
            publish = False
            result["status"] = "FAILED"
            result["errors"].extend(f"promotion provenance: {error}" for error in final_errors)
    if publish and result["status"] == "READY" and report_text is not None and case_filter is None:
        try:
            _atomic_write(args.result_root / args.target_profile / "cpu" / "small_work.md", report_text)
        except Exception as exc:
            result["status"] = "FAILED"
            result["errors"].append(f"latest report publication failed: {type(exc).__name__}")
    _archive_result(root, output, result)
    return 0 if result["status"] in {"READY", "INCONCLUSIVE", "CORRECTNESS_ONLY"} else 1


def _component_run(args: argparse.Namespace) -> int:
    from small_work_components import check_components

    if args.case_binary is None or args.artifact_kind != "lib-test":
        raise SystemExit("--component-probe requires --case-binary and --artifact-kind lib-test")
    if not args.correctness_only and args.suite is None:
        raise SystemExit("--component-probe requires --correctness-only or --suite for timing protocol")
    if args.allocation_iterations is not None and not args.correctness_only:
        raise SystemExit("allocation diagnostics require --correctness-only; do not mix allocation and timing")
    timestamp = _timestamp()
    root = args.results_root / args.target_profile / "cpu" / "small_work_components" / timestamp
    root.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "suite_id": "cpu/small_work_components",
              "timestamp": timestamp, "status": "FAILED", "errors": [],
              "provenance": {"status": "unverified", "reasons": ["no Cargo preparation receipt"]}}
    timing = not args.correctness_only
    original = None
    benchmark = Path(__file__).resolve().parents[1]
    library = args.tenferro_dir or benchmark / "extern/tenferro-rs"
    def receipt_errors(receipt):
        return verify_preparation_receipt(
            receipt, benchmark=benchmark, library=library,
            binary=args.case_binary, cargo=args.cargo, build_project=args.build_project,
            package=args.cargo_package, target_name=args.cargo_target_name,
            artifact_kind="lib-test", require_timing=timing)
    try:
        protocol = None
        selection = None
        if timing:
            suite = yaml.safe_load(args.suite.read_text())
            validate_suite_contract(suite)
            protocol = suite["protocol"]
            result["protocol"] = protocol
            if args.preparation_receipt is None:
                raise ValueError("component timing requires a verified release preparation receipt")
        if args.preparation_receipt is not None:
            receipt = load_json(args.preparation_receipt)
            if not isinstance(receipt, Mapping):
                raise ValueError("preparation receipt is not an object")
            errors = receipt_errors(receipt)
            result["provenance"] = {"status": "unverified" if errors else "verified", "reasons": errors,
                                    "receipt": _archive_receipt(root, receipt)}
            if errors:
                raise ValueError("; ".join(errors))
        if timing:
            policy = protocol["resource_policy"]
            selection = _select_live_resources(policy["requested_threads"], policy["busy_threshold"],
                                               args.observation_window, args.cpuset)
            result["resource"] = selection
            if selection["status"] != "valid":
                raise ValueError("no eligible resources for component timing")
            original = set(os.sched_getaffinity(0))
            os.sched_setaffinity(0, set(selection["cpus"]))
            verify_affinity(selection["cpus"], os.sched_getaffinity(0))
        _archive_result(root, args.output, result)
        result.update(check_components(args.case_binary, root / "children", args.command_timeout,
                                       args.allocation_iterations, protocol=protocol,
                                       expected_affinity=set(selection["cpus"]) if selection else None))
        if timing and result["status"] in {"TIMING_DIAGNOSTIC", "INCONCLUSIVE"}:
            final_errors = receipt_errors(receipt)
            if final_errors:
                raise ValueError("source/artifact changed during timing: " + "; ".join(final_errors))
            after = _select_live_resources(policy["requested_threads"], policy["busy_threshold"],
                                           args.observation_window, args.cpuset)
            result["resource_after"] = after
            if after["status"] != "valid":
                raise ValueError("resources no longer eligible after component timing")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        result["status"] = "INCONCLUSIVE" if timing else "FAILED"
        result["errors"].append(str(exc))
    finally:
        if original is not None:
            try:
                os.sched_setaffinity(0, original)
                verify_affinity(original, os.sched_getaffinity(0))
            except (OSError, ValueError) as exc:
                result["status"] = "INCONCLUSIVE"
                result["errors"].append(f"affinity restoration failed: {exc}")
    _archive_result(root, args.output, result)
    return 0 if result["status"] in {"CORRECTNESS_ONLY", "ALLOCATION_DIAGNOSTIC", "TIMING_DIAGNOSTIC", "INCONCLUSIVE"} else 1


def _prepare_run(args: argparse.Namespace) -> int:
    if args.output is None or (args.artifact_kind == "bin" and args.case_binary is None):
        raise SystemExit("--prepare requires --output and --case-binary for bin artifacts")
    benchmark = Path(__file__).resolve().parents[1]
    library = args.tenferro_dir or benchmark / "extern" / "tenferro-rs"
    receipt = prepare_cargo(benchmark=benchmark, library=library, binary=args.case_binary,
                           receipt=args.output, features=args.features,
                           profile=args.profile, cargo=args.cargo,
                           previous_receipt=args.preparation_receipt,
                           build_project=args.build_project, package=args.cargo_package,
                           target_name=args.cargo_target_name, artifact_kind=args.artifact_kind)
    print(json.dumps({"status": receipt.get("status"), "receipt": str(args.output),
                      "errors": receipt.get("errors", [])}))
    return 0 if receipt.get("status") == "verified" else 1


def _fixture_run(args: argparse.Namespace) -> int:
    if not args.before or not args.after:
        raise SystemExit("--synthetic-fixture requires --before and --after")
    if args.command:
        raise SystemExit("synthetic fixture mode cannot execute production commands")
    before = parse_proc_stat(args.before.read_text(encoding="utf-8"))
    after = parse_proc_stat(args.after.read_text(encoding="utf-8"))
    allowed = parse_cpu_set(args.cpuset) if args.cpuset else tuple(sorted(after))
    selection = select_idle_cpus(cpu_busy_fraction(before, after), allowed, args.requested_threads,
                                 busy_threshold=args.busy_threshold)
    selection["status"] = "inconclusive"
    selection["reasons"].append("synthetic fixture mode is not production resource evidence")
    result = {"schema_version": 1, "suite_id": "cpu/small_work", "status": "INCONCLUSIVE",
              "synthetic_fixture": True, "resource": selection, "commands": [], "errors": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def _legacy_run(args: argparse.Namespace) -> int:
    if args.before or args.after:
        raise SystemExit("fixture snapshots require explicit --synthetic-fixture")
    try:
        selection = _select_live_resources(args.requested_threads, args.busy_threshold,
                                           args.observation_window, args.cpuset)
    except (OSError, ValueError, ContractError) as exc:
        selection = {"status": "inconclusive", "cpus": [], "effective_threads": 0,
                     "reasons": [f"live resource observation failed: {type(exc).__name__}"]}
    result: dict[str, Any] = {"schema_version": 1, "suite_id": "cpu/small_work",
                              "commands": [], "errors": [], "resource": selection,
                              "timestamp_monotonic_ns": time.monotonic_ns(), "synthetic_fixture": False}
    if selection["status"] != "valid":
        result["status"] = "INCONCLUSIVE"
    else:
        original = set(os.sched_getaffinity(0))
        try:
            chosen = set(selection["cpus"])
            os.sched_setaffinity(0, chosen)
            verify_affinity(chosen, os.sched_getaffinity(0))
            records = run_sequential(_commands(args.command), expected_affinity=chosen,
                                     timeout_s=args.command_timeout, return_records=True)
            result["commands"] = records
            result["status"] = "READY" if not any(r.get("status") in {"failed", "error"} for r in records) else "FAILED"
        except (OSError, AttributeError, ContractError) as exc:
            result["status"] = "INCONCLUSIVE"
            result["errors"] = [f"affinity policy failed: {type(exc).__name__}"]
        finally:
            try:
                os.sched_setaffinity(0, original)
                verify_affinity(original, os.sched_getaffinity(0))
            except Exception as exc:
                result["status"] = "INCONCLUSIVE"
                result["errors"].append(f"affinity restoration failed: {type(exc).__name__}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] in {"READY", "INCONCLUSIVE"} else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview suite cases and live resources without building or executing benchmarks")
    parser.add_argument("--changed-path", action="append", default=[],
                        help="Library-relative changed path; select every suite variant via the canonical selector")
    parser.add_argument("--case-binary", type=Path)
    parser.add_argument("--artifact-kind", choices=("bin", "lib-test"), default="bin")
    parser.add_argument("--build-project", type=Path)
    parser.add_argument("--cargo-package")
    parser.add_argument("--cargo-target-name")
    parser.add_argument("--component-probe", action="store_true",
                        help="Run crate-owned einsum component correctness, allocation or timing diagnostics")
    parser.add_argument("--allocation-iterations", type=int,
                        help="After component correctness checks, collect separate caller allocation diagnostics")
    parser.add_argument("--prepare", action="store_true",
                        help="Cargo-build the artifact; --preparation-receipt is prior proof for cached output")
    parser.add_argument("--preparation-receipt", type=Path,
                        help="receipt to verify, or prior proof when used with --prepare")
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--profile", default="release")
    parser.add_argument("--target-profile", default="amd-cpu")
    parser.add_argument("--results-root", type=Path, default=Path("data/results"))
    parser.add_argument("--result-root", type=Path, default=Path("result"))
    parser.add_argument("--tenferro-dir", type=Path)
    parser.add_argument("--tenferro-commit")
    parser.add_argument("--features", action="append", default=[])
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--synthetic-fixture", action="store_true")
    parser.add_argument("--cpuset")
    parser.add_argument("--requested-threads", type=int, default=1)
    parser.add_argument("--busy-threshold", type=float, default=0.20)
    parser.add_argument("--observation-window", type=float, default=1.0)
    parser.add_argument("--command-timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--command", action="append", default=[])
    args = parser.parse_args()
    if args.changed_path and (not args.suite or args.prepare or args.component_probe or args.synthetic_fixture or args.command):
        parser.error("--changed-path requires ordinary --suite mode")
    if args.dry_run and (not args.suite or args.prepare or args.component_probe or
                         args.synthetic_fixture or args.correctness_only or args.command):
        parser.error("--dry-run requires --suite and cannot be combined with execution modes")
    if args.allocation_iterations is not None and (not args.component_probe or args.allocation_iterations < 1):
        parser.error("--allocation-iterations requires --component-probe and a positive count")
    if args.component_probe:
        if args.prepare or args.synthetic_fixture:
            parser.error("--component-probe cannot be combined with --prepare or --synthetic-fixture")
        return _component_run(args)
    if args.prepare:
        return _prepare_run(args)
    if args.synthetic_fixture:
        if not args.output:
            raise SystemExit("--output is required for fixture mode")
        return _fixture_run(args)
    if args.suite:
        if not args.dry_run:
            args.output = args.output or Path("small_work_run.json")
        return _suite_run(args)
    if args.output is None:
        raise SystemExit("--output is required")
    # Keep the original command-only resource gate for focused affinity tests.
    return _legacy_run(args)


if __name__ == "__main__":
    sys.exit(main())
