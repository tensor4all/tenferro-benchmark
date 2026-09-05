#!/usr/bin/env python3
"""Contracts, statistics, resource selection, and reporting for cpu/small_work.

This module deliberately keeps collection policy separate from execution.  It can
be exercised with synthetic samples without claiming those samples are benchmark
 evidence.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence

API_TIERS = frozenset({
    "concrete-fresh", "concrete-shared", "eager-no-ad", "eager-ad",
    "prepared-setup", "prepared-repeat", "compiled-repeat",
})
PHASES = frozenset({"setup", "execution", "validation"})
CORRECTNESS = frozenset({"passed", "failed", "not_run"})
TIMING_VALIDITY = frozenset({"valid", "inconclusive", "invalid"})
CANONICAL_EXPORT = "docs/internals/public-boundary-benchmarks.json"
CANONICAL_INVENTORY = "docs/internals/public-boundary-overhead-inventory.json"


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_canonical_catalog(library: Path) -> dict[str, dict[str, Any]]:
    data = load_json(library / CANONICAL_EXPORT)
    if not isinstance(data, Mapping) or data.get("schema") != "tenferro.public-boundary-benchmarks.v1":
        raise ContractError("canonical export schema is invalid")
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ContractError("canonical export cases are missing")
    catalog: dict[str, dict[str, Any]] = {}
    for item in cases:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise ContractError("canonical export contains malformed case")
        ident = item["id"]
        if ident in catalog:
            raise ContractError(f"duplicate canonical contract ID: {ident}")
        catalog[ident] = dict(item)
    return catalog


def run_inventory_checker(library: Path, timeout_s: float = 30.0) -> dict[str, Any]:
    checker = library / "scripts/check-public-boundary-inventory.py"
    if not checker.is_file():
        raise ContractError("canonical inventory checker is missing")
    try:
        completed = subprocess.run(["python3", "scripts/check-public-boundary-inventory.py"], cwd=library, capture_output=True,
                                   text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ContractError("canonical inventory checker timed out", details={"stdout": exc.stdout, "stderr": exc.stderr}) from exc
    if completed.returncode != 0:
        raise ContractError("canonical inventory checker failed", details={"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def canonical_snapshot(library: Path, cases: Sequence["CaseContract"]) -> dict[str, Any]:
    export = (library / CANONICAL_EXPORT).resolve()
    inventory = (library / CANONICAL_INVENTORY).resolve()
    from small_work_provenance import _git_state
    state = dict(_git_state(library))
    state.setdefault("error", None)
    if (state.get("error") or state.get("head") is None or state.get("dirty") is not False
            or state.get("untracked") is not False):
        raise ContractError("canonical checkout identity is not clean", details=state)
    try:
        export_hash, inventory_hash = sha256_file(export), sha256_file(inventory)
    except OSError as exc:
        raise ContractError("canonical binding files unavailable") from exc
    return {"export": str(export), "inventory": str(inventory), "checkout": state,
            "checkout_path": state.get("path"), "checkout_head": state.get("head"),
            "checkout_dirty": state.get("dirty"), "checkout_untracked": state.get("untracked"),
            "checkout_error": state.get("error"), "checkout_revision": state["head"], "export_sha256": export_hash,
            "inventory_sha256": inventory_hash,
            "contract_ids": sorted({case.contract_id for case in cases})}


def verify_canonical_snapshot(snapshot: Mapping[str, Any], library: Path, cases: Sequence["CaseContract"]) -> None:
    current = canonical_snapshot(library, cases)
    for key in ("export", "inventory", "checkout", "checkout_path", "checkout_head",
                "checkout_dirty", "checkout_untracked", "checkout_error",
                "export_sha256", "inventory_sha256", "contract_ids"):
        if current.get(key) != snapshot.get(key):
            raise ContractError(f"canonical binding changed before publication: {key}")


def verify_canonical_binding(library: Path, cases: Sequence["CaseContract"], *, check_inventory: bool = True) -> dict[str, Any]:
    export_path, inventory_path = library / CANONICAL_EXPORT, library / CANONICAL_INVENTORY
    before = canonical_snapshot(library, cases)
    checker = run_inventory_checker(library) if check_inventory else None
    catalog = load_canonical_catalog(library)
    inventory = load_json(inventory_path)
    if not isinstance(inventory, Mapping) or not isinstance(inventory.get("families"), Mapping):
        raise ContractError("canonical inventory schema is invalid")
    for case in cases:
        route = catalog.get(case.contract_id)
        if route is None:
            raise ContractError(f"unknown canonical contract ID: {case.contract_id}")
        for key, expected in (("family", case.family), ("operation", case.operation),
                              ("surface", case.surface), ("phase", case.phase)):
            if route.get(key) != expected:
                raise ContractError(f"canonical {key} mismatch for {case.contract_id}")
        family = inventory["families"].get(case.family)
        selectors = family.get("selectors", []) if isinstance(family, Mapping) else []
        matches = []
        for selector in selectors:
            if not isinstance(selector, Mapping):
                continue
            if case.operation not in selector.get("operations", []):
                continue
            for item in selector.get("cases", []):
                if isinstance(item, Mapping) and item.get("id", "").replace("{family}", case.family).replace("{operation}", case.operation) == case.contract_id and item.get("surface") == case.surface:
                    matches.append((selector, item))
        if len(matches) != 1:
            raise ContractError(f"canonical inventory selector is ambiguous for {case.contract_id}")
        selector, item = matches[0]
        surfaces = selector.get("surfaces", {})
        disposition = surfaces.get(case.surface, {}).get("disposition") if isinstance(surfaces, Mapping) else None
        if disposition not in {"supported", "follow-up"}:
            raise ContractError(f"canonical route disposition is invalid: {case.contract_id}")
    verify_canonical_snapshot(before, library, cases)
    before["checker"] = checker
    return before


class ContractError(ValueError):
    """A suite or case contract is incomplete or internally inconsistent."""
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.details = dict(details or {})


@dataclass(frozen=True)
class CaseContract:
    case_id: str
    contract_id: str
    family: str
    surface: str
    operation: str
    phase: str
    api_tier: str
    backend: str
    dtype: str
    layout: str
    shape: tuple[int, ...]
    calls_per_workflow: int
    workflow: str
    setup_includes: tuple[str, ...]
    setup_excludes: tuple[str, ...]
    scope_timer: tuple[str, ...]
    scope_outside_timer: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CaseContract":
        required = ("id", "contract_id", "family", "surface", "operation", "phase", "api_tier", "backend", "dtype",
                    "layout", "shape", "calls_per_workflow", "workflow", "setup", "scope")
        missing = [key for key in required if key not in value]
        if missing:
            raise ContractError(f"case contract missing: {', '.join(missing)}")
        case_id = value["id"]
        if not isinstance(case_id, str) or not case_id:
            raise ContractError("case id must be a non-empty string")
        canonical = {"contract_id": value["contract_id"], "family": value["family"], "surface": value["surface"]}
        for field in ("operation", "backend", "dtype", "layout", "workflow"):
            if not isinstance(value[field], str) or not value[field]:
                raise ContractError(f"{case_id}: {field} must be a non-empty string")
        if any(not isinstance(item, str) or not item for item in canonical.values()):
            raise ContractError(f"{case_id}: canonical descriptors must be non-empty strings")
        phase, tier = value["phase"], value["api_tier"]
        if phase not in PHASES:
            raise ContractError(f"{case_id}: unsupported phase {phase!r}")
        if tier not in API_TIERS:
            raise ContractError(f"{case_id}: unsupported api_tier {tier!r}")
        if tier == "prepared-setup" and phase != "setup":
            raise ContractError(f"{case_id}: prepared-setup requires setup phase")
        if tier == "prepared-repeat" and phase != "execution":
            raise ContractError(f"{case_id}: prepared-repeat requires execution phase")
        shape = value["shape"]
        if (not isinstance(shape, list) or not shape or
                any(type(dim) is not int or dim < 1 for dim in shape)):
            raise ContractError(f"{case_id}: shape must contain positive integers")
        calls = value["calls_per_workflow"]
        if type(calls) is not int or calls < 1:
            raise ContractError(f"{case_id}: calls_per_workflow must be positive")
        setup = value["setup"]
        if not isinstance(setup, Mapping):
            raise ContractError(f"{case_id}: setup must be an object")
        includes = setup.get("includes", [])
        excludes = setup.get("excludes", [])
        if (not isinstance(includes, list) or not isinstance(excludes, list) or
                any(not isinstance(item, str) or not item for item in includes + excludes)):
            raise ContractError(f"{case_id}: setup includes/excludes must be string lists")
        overlap = sorted(set(includes) & set(excludes))
        if overlap:
            raise ContractError(f"{case_id}: setup both includes and excludes {overlap}")
        scope = value["scope"]
        if (not isinstance(scope, Mapping) or
                not isinstance(scope.get("timer"), list) or
                not isinstance(scope.get("outside_timer"), list) or
                any(type(item) is not str or not item
                    for item in scope["timer"] + scope["outside_timer"]) or
                set(scope["timer"]) & set(scope["outside_timer"])):
            raise ContractError(f"{case_id}: scope must contain disjoint timer/outside_timer string lists")
        return cls(case_id, canonical["contract_id"], canonical["family"], canonical["surface"], value["operation"], phase, tier, value["backend"],
                   value["dtype"], value["layout"], tuple(shape), calls,
                   value["workflow"], tuple(includes), tuple(excludes),
                   tuple(scope["timer"]), tuple(scope["outside_timer"]))


def resolve_case_filter(cases: Sequence[CaseContract], raw: str | None) -> tuple[tuple[CaseContract, ...], str | None]:
    """Resolve exact case IDs before canonical binding or child launch."""
    if raw is None:
        return tuple(cases), None
    if not raw:
        raise ContractError("SMALL_WORK_CASE_FILTER must not be empty")
    selected_ids = raw.split(",")
    if any(not case_id for case_id in selected_ids):
        raise ContractError("SMALL_WORK_CASE_FILTER contains an empty case ID")
    available = {case.case_id: case for case in cases}
    unknown = [case_id for case_id in selected_ids if case_id not in available]
    if unknown:
        raise ContractError(f"SMALL_WORK_CASE_FILTER contains unknown case ID: {unknown[0]}")
    if len(selected_ids) != len(set(selected_ids)):
        raise ContractError("SMALL_WORK_CASE_FILTER contains duplicate case IDs")
    return tuple(available[case_id] for case_id in selected_ids), raw


def validate_suite_contract(suite: Mapping[str, Any]) -> tuple[CaseContract, ...]:
    if suite.get("suite_id") != "cpu/small_work":
        raise ContractError("small-work suite_id must be cpu/small_work")
    if suite.get("contract_version") != 2:
        raise ContractError("unsupported small-work contract_version")
    protocol = suite.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ContractError("small-work protocol is required")
    for key in ("warmups", "runs", "independent_processes", "samples_per_process"):
        minimum = 0 if key == "warmups" else (2 if key in ("independent_processes", "samples_per_process") else 1)
        if type(protocol.get(key)) is not int or protocol[key] < minimum:
            raise ContractError(f"protocol {key} must be at least {minimum}")
    noise = protocol.get("noise_policy")
    if (not isinstance(noise, Mapping) or
            type(noise.get("process_median_cov_max")) not in (int, float) or
            isinstance(noise.get("process_median_cov_max"), bool) or
            not math.isfinite(float(noise["process_median_cov_max"])) or
            float(noise["process_median_cov_max"]) < 0):
        raise ContractError("protocol noise_policy.process_median_cov_max must be finite and non-negative")
    calibration = protocol.get("calibration")
    if (not isinstance(calibration, Mapping) or calibration.get("enabled") is not True or
            type(calibration.get("target_ns")) is not int or calibration["target_ns"] <= 0):
        raise ContractError("protocol calibration must be enabled with target_ns > 0")
    resource = protocol.get("resource_policy")
    if (not isinstance(resource, Mapping) or
            type(resource.get("requested_threads")) is not int or resource["requested_threads"] < 1 or
            type(resource.get("busy_threshold")) not in (int, float) or
            isinstance(resource["busy_threshold"], bool) or
            not math.isfinite(float(resource["busy_threshold"])) or
            not 0 <= float(resource["busy_threshold"]) <= 1):
        raise ContractError("protocol resource_policy has invalid thread/busy settings")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ContractError("small-work suite requires non-empty cases")
    for case in cases:
        if any(key not in case for key in ("contract_id", "family", "surface")):
            raise ContractError("small-work cases require canonical contract_id, family, and surface")
    parsed = tuple(CaseContract.from_mapping(case) for case in cases)
    ids = [case.case_id for case in parsed]
    if len(ids) != len(set(ids)):
        raise ContractError("small-work case IDs must be unique")
    return parsed


def _quartile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("at least one sample is required")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def timing_statistics(samples: Sequence[Mapping[str, Any]], calls_per_workflow: int,
                      *, min_processes: int = 2,
                      min_samples_per_process: int = 2) -> dict[str, Any]:
    """Normalize aggregates before process-based statistics are computed."""
    if (type(calls_per_workflow) is not int or calls_per_workflow < 1 or not samples or
            type(min_processes) is not int or min_processes < 1 or
            type(min_samples_per_process) is not int or min_samples_per_process < 1):
        raise ValueError("calls, minimum processes, samples, and samples per process must be positive integers")
    by_process: dict[int, list[float]] = defaultdict(list)
    elapsed: list[int] = []
    identities: set[tuple[int, int]] = set()
    for sample in samples:
        process = sample.get("process_index")
        index = sample.get("sample_index")
        raw = sample.get("elapsed_ns")
        iterations = sample.get("iterations", 1)
        if (type(process) is not int or process < 0 or type(index) is not int or index < 0 or
                type(raw) is not int or raw <= 0 or type(iterations) is not int or iterations < 1):
            raise ValueError("samples require positive integer elapsed_ns/iterations and indices")
        identity = (process, index)
        if identity in identities:
            raise ValueError("duplicate process/sample identity")
        identities.add(identity)
        workflow = raw / iterations
        if not math.isfinite(workflow) or workflow <= 0:
            raise ValueError("normalized sample must be finite and positive")
        elapsed.append(raw)
        by_process[process].append(workflow)
    if len(by_process) < min_processes or any(
            len(values) < min_samples_per_process for values in by_process.values()):
        raise ValueError("independent process/sample minimum was not met")
    process_medians = [statistics.median(by_process[key]) for key in sorted(by_process)]
    process_iqr = _quartile(process_medians, 0.75) - _quartile(process_medians, 0.25)
    mean = statistics.fmean(process_medians)
    cov = (statistics.pstdev(process_medians) / mean) if len(process_medians) > 1 else 0.0
    return {
        "sample_count": len(elapsed),
        "process_count": len(process_medians),
        "raw_elapsed_ns": elapsed,
        "median_ns": int(statistics.median(elapsed)),
        "iqr_ns": int(round(_quartile(elapsed, 0.75) - _quartile(elapsed, 0.25))),
        "raw_aggregate_median_ns": int(statistics.median(elapsed)),
        "raw_aggregate_iqr_ns": int(round(_quartile(elapsed, 0.75) - _quartile(elapsed, 0.25))),
        "process_median_ns": _number(statistics.median(process_medians)),
        "process_median_iqr_ns": _number(process_iqr),
        "process_median_dispersion_ns": _number(max(process_medians) - min(process_medians)),
        "process_median_cov": cov,
        "normalized_ns_per_call": _number(statistics.median(process_medians) / calls_per_workflow),
        "normalized_ns_per_workflow": _number(statistics.median(process_medians)),
    }


def make_record(case: CaseContract, samples: Sequence[Mapping[str, Any]], *,
                correctness_status: str, timing_status: str = "valid",
                timing_reasons: Sequence[str] = (), provider: str = "unknown",
                errors: Sequence[str] = ()) -> dict[str, Any]:
    if correctness_status not in CORRECTNESS or timing_status not in TIMING_VALIDITY:
        raise ValueError("invalid correctness or timing status")
    if correctness_status != "passed" and timing_status == "valid":
        timing_status = "invalid"
        timing_reasons = tuple(timing_reasons) + ("correctness did not pass",)
    if correctness_status == "failed" or correctness_status == "not_run":
        timing = None
    elif samples:
        timing = timing_statistics(samples, case.calls_per_workflow)
    else:
        timing = None
    return {
        "schema_version": 2,
        "suite_id": "cpu/small_work",
        "case_id": case.case_id,
        "contract_id": case.contract_id,
        "family": case.family,
        "surface": case.surface,
        "operation": case.operation,
        "phase": case.phase,
        "api_tier": case.api_tier,
        "backend": case.backend,
        "provider": provider,
        "dtype": case.dtype,
        "layout": case.layout,
        "shape": list(case.shape),
        "workflow": case.workflow,
        "calls_per_workflow": case.calls_per_workflow,
        "setup": {"includes": list(case.setup_includes), "excludes": list(case.setup_excludes)},
        "scope": {"timer": list(case.scope_timer), "outside_timer": list(case.scope_outside_timer)},
        "samples": list(samples),
        "timing": timing,
        "correctness": {"status": correctness_status},
        "timing_validity": {"status": timing_status, "reasons": list(timing_reasons)},
        "errors": list(errors),
    }


def validate_record(record: Mapping[str, Any], case: CaseContract | None = None) -> None:
    required = ("schema_version", "suite_id", "case_id", "contract_id", "family", "surface", "operation", "phase",
                "api_tier", "backend", "provider", "dtype", "layout", "shape",
                "workflow", "calls_per_workflow", "setup", "scope", "samples", "timing",
                "correctness", "timing_validity", "errors")
    missing = [key for key in required if key not in record]
    if missing:
        raise ContractError(f"record missing: {', '.join(missing)}")
    if record["suite_id"] != "cpu/small_work" or type(record["schema_version"]) is not int or record["schema_version"] != 2:
        raise ContractError("record has unsupported suite/schema")
    for key in ("case_id", "contract_id", "family", "surface", "operation", "backend", "provider", "dtype", "layout", "workflow"):
        if not isinstance(record[key], str) or not record[key]:
            raise ContractError(f"record {key} must be a non-empty string")
    if case is not None:
        expected = {"case_id": case.case_id, "contract_id": case.contract_id, "family": case.family,
                    "surface": case.surface, "operation": case.operation, "phase": case.phase,
                    "api_tier": case.api_tier, "backend": case.backend, "dtype": case.dtype,
                    "layout": case.layout, "shape": list(case.shape), "workflow": case.workflow,
                    "calls_per_workflow": case.calls_per_workflow,
                    "setup": {"includes": list(case.setup_includes), "excludes": list(case.setup_excludes)},
                    "scope": {"timer": list(case.scope_timer), "outside_timer": list(case.scope_outside_timer)}}
        for key, value in expected.items():
            if record.get(key) != value:
                raise ContractError(f"record {key} does not match case contract")
    if record["phase"] not in PHASES or record["api_tier"] not in API_TIERS:
        raise ContractError("record has unsupported phase/api tier")
    if (not isinstance(record["shape"], list) or not record["shape"] or
            any(type(dim) is not int or dim < 1 for dim in record["shape"])):
        raise ContractError("record shape is invalid")
    scope = record["scope"]
    if (not isinstance(scope, Mapping) or
            not isinstance(scope.get("timer"), list) or
            not isinstance(scope.get("outside_timer"), list) or
            any(type(item) is not str or not item
                for item in scope["timer"] + scope["outside_timer"]) or
            set(scope["timer"]) & set(scope["outside_timer"])):
        raise ContractError("record scope is invalid")
    calls = record["calls_per_workflow"]
    if type(calls) is not int or calls < 1:
        raise ContractError("record calls_per_workflow is invalid")
    setup = record["setup"]
    if (not isinstance(setup, Mapping) or not isinstance(setup.get("includes"), list) or
            not isinstance(setup.get("excludes"), list) or
            any(not isinstance(item, str) or not item
                for item in setup["includes"] + setup["excludes"]) or
            set(setup["includes"]) & set(setup["excludes"])):
        raise ContractError("record setup is invalid")
    correctness_obj, validity_obj = record["correctness"], record["timing_validity"]
    if not isinstance(correctness_obj, Mapping) or not isinstance(validity_obj, Mapping):
        raise ContractError("record status objects are invalid")
    reasons = validity_obj.get("reasons")
    errors = record["errors"]
    if (not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons) or
            not isinstance(errors, list) or any(not isinstance(item, str) for item in errors)):
        raise ContractError("record reasons/errors are invalid")
    correctness = correctness_obj.get("status")
    validity = validity_obj.get("status")
    if correctness not in CORRECTNESS or validity not in TIMING_VALIDITY:
        raise ContractError("record has invalid independent statuses")
    samples = record["samples"]
    if not isinstance(samples, list):
        raise ContractError("samples must be a list")
    expected_timing = None
    if samples and correctness == "passed":
        expected_timing = timing_statistics(samples, calls)
    if validity == "valid" and (not samples or correctness != "passed"):
        raise ContractError("valid timing requires samples and passed correctness")
    if validity == "valid" and record["provider"] == "unknown":
        raise ContractError("valid timing requires provider provenance")
    if correctness == "passed" and validity == "valid" and record["timing"] is None:
        raise ContractError("valid timing requires statistics")
    if record["timing"] is not None:
        if expected_timing is None or record["timing"] != expected_timing:
            raise ContractError("record timing does not match samples")
    elif validity == "valid":
        raise ContractError("valid timing cannot omit statistics")


def _display_ns(value: float | int) -> str:
    value = float(value)
    if value < 0 or not math.isfinite(value):
        raise ValueError("display value must be finite and non-negative")
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f} ms"
    if value >= 1_000:
        return f"{value / 1_000:.3f} us"
    if 0 < value < 1:
        return f"{value:.3g} ns"
    return f"{value:.3f} ns" if value % 1 else f"{value:.0f} ns"


def render_report(records: Iterable[Mapping[str, Any]]) -> str:
    checked = list(records)
    if not checked:
        raise ValueError("cannot render an empty small-work report")
    for record in checked:
        validate_record(record)
    lines = ["# CPU Small-work Results", "", "Synthetic fixture records are contract checks, not performance evidence.", "",
             "| Case | Tier | Backend | Correctness | Timing validity | Raw aggregate median | Workflow ns/call |", "|---|---|---|---|---|---:|---:|"]
    for record in checked:
        timing = record["timing"] or {}
        usable = (record["correctness"]["status"] == "passed" and
                  record["timing_validity"]["status"] == "valid" and timing)
        median = _display_ns(timing.get("raw_aggregate_median_ns", timing.get("median_ns", 0))) if usable else "—"
        per_call = _display_ns(timing.get("normalized_ns_per_call", 0)) if usable else "—"
        lines.append(f"| `{record['case_id']}` | `{record['api_tier']}` | `{record['backend']}` | "
                     f"{record['correctness']['status']} | {record['timing_validity']['status']} | {median} | {per_call} |")
    return "\n".join(lines) + "\n"


def parse_cpu_set(value: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    for part in value.split(","):
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            if end < start:
                raise ValueError("invalid descending CPU range")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return tuple(sorted(cpus))


def parse_proc_stat(text: str) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or not fields[0][3:].isdigit():
            continue
        values = [int(value) for value in fields[1:9]]  # user..steal; exclude guest/guest_nice
        if len(values) < 4:
            continue
        total = sum(values)
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        result[int(fields[0][3:])] = (total, total - idle)
    return result


def cpu_busy_fraction(before: Mapping[int, tuple[int, int]], after: Mapping[int, tuple[int, int]]) -> dict[int, float]:
    fractions: dict[int, float] = {}
    for cpu in before.keys() & after.keys():
        total_delta = after[cpu][0] - before[cpu][0]
        busy_delta = after[cpu][1] - before[cpu][1]
        if total_delta > 0 and 0 <= busy_delta <= total_delta:
            fractions[cpu] = busy_delta / total_delta
    return fractions


def _topology_value(info: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in info:
            return info[name]
    return None


def select_idle_cpus(loads: Mapping[int, float], allowed: Iterable[int], requested_threads: int,
                     *, busy_threshold: float = 0.20,
                     smt_siblings: Mapping[int, int | Sequence[int]] | None = None,
                     quota_cpus: float | None = None,
                     topology: Mapping[int, Mapping[str, Any]] | None = None,
                     require_topology: bool = False,
                     preferred_numa: str | None = None) -> dict[str, Any]:
    if type(requested_threads) is not int or requested_threads < 1:
        raise ValueError("requested_threads must be a positive integer")
    if require_topology and not topology:
        return {"status": "inconclusive", "cpus": [], "effective_threads": 0,
                "reasons": ["CPU topology telemetry is unavailable"]}
    if quota_cpus is not None and (not math.isfinite(quota_cpus) or quota_cpus <= 0):
        return {"status": "inconclusive", "cpus": [], "effective_threads": 0,
                "reasons": ["CPU quota telemetry is unavailable or insufficient"]}
    allowed_set = set(allowed)
    groups: dict[tuple[Any, Any], dict[str, Any]] = {}
    missing: set[int] = set()
    for cpu in sorted(allowed_set):
        info = topology.get(cpu, {}) if topology else {}
        package = _topology_value(info, "package", "physical_package_id")
        core = _topology_value(info, "core", "core_id")
        numa = _topology_value(info, "numa", "numa_node", "node")
        if topology and (package is None or core is None):
            return {"status": "inconclusive", "cpus": [], "effective_threads": 0,
                    "reasons": ["CPU package/core topology is incomplete"]}
        if topology:
            sibling_value = _topology_value(info, "siblings", "thread_siblings")
            if sibling_value is None:
                siblings = [cpu]
            elif isinstance(sibling_value, int):
                siblings = [sibling_value]
            else:
                siblings = list(sibling_value)
            key = (package, core)
        else:
            descriptor = smt_siblings.get(cpu, cpu) if smt_siblings else cpu
            if isinstance(descriptor, int):
                core = descriptor
                siblings = [other for other in allowed_set
                            if (smt_siblings.get(other, other) if smt_siblings else other) == core]
            else:
                siblings = list(descriptor)
                core = min(siblings) if siblings else cpu
            key, numa = (core, core), None
        if any(sibling not in loads for sibling in siblings):
            missing.update(sibling for sibling in siblings if sibling not in loads)
            continue
        if any(not math.isfinite(loads[sibling]) for sibling in siblings):
            missing.update(sibling for sibling in siblings if not math.isfinite(loads[sibling]))
            continue
        entry = groups.setdefault(key, {"cpus": [], "siblings": set(siblings), "numa": numa, "package": package})
        entry["cpus"].append(cpu)
    if missing:
        return {"status": "inconclusive", "cpus": [], "effective_threads": 0,
                "reasons": ["missing per-CPU telemetry for SMT siblings"]}
    viable = []
    for entry in groups.values():
        siblings = entry["siblings"]
        if all(loads[sibling] <= busy_threshold for sibling in siblings):
            viable.append(entry)
    if not viable:
        return {"status": "inconclusive", "cpus": [], "effective_threads": 0,
                "reasons": ["no physical core has idle telemetry for every SMT sibling"]}
    preferred = preferred_numa
    if preferred is None:
        counts: dict[Any, int] = defaultdict(int)
        for entry in viable:
            counts[entry["numa"]] += 1
        preferred = max(counts, key=counts.get)
    viable.sort(key=lambda entry: (entry["numa"] != preferred, entry["package"], min(entry["cpus"])))
    quota_limit = math.floor(quota_cpus) if quota_cpus is not None else requested_threads
    if quota_limit < requested_threads:
        return {"status": "inconclusive", "cpus": [], "effective_threads": 0,
                "reasons": ["effective CPU quota is below requested threads"]}
    selected = [min(entry["cpus"]) for entry in viable[:min(requested_threads, quota_limit)]]
    if len(selected) < requested_threads:
        return {"status": "inconclusive", "cpus": selected, "effective_threads": len(selected),
                "reasons": ["insufficient idle physical cores under cpuset/quota/SMT policy"]}
    return {"status": "valid", "cpus": selected, "effective_threads": len(selected), "reasons": []}


def verify_affinity(expected: Iterable[int], actual: Iterable[int]) -> None:
    expected_set, actual_set = set(expected), set(actual)
    if expected_set != actual_set:
        raise ContractError("effective affinity differs from selected CPUs")


def _archive_child(output_dir: Path | None, index: int, stdout: str | bytes, stderr: str | bytes) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    def text(value: str | bytes | None) -> str:
        return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
    (output_dir / f"child_{index:03d}.stdout").write_text(text(stdout), encoding="utf-8")
    (output_dir / f"child_{index:03d}.stderr").write_text(text(stderr), encoding="utf-8")


def run_sequential(commands: Sequence[Sequence[str]], *, cwd: Path | None = None,
                   env: Mapping[str, str] | None = None,
                   runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
                   expected_affinity: Iterable[int] | None = None,
                   timeout_s: float | None = None,
                   output_dir: Path | None = None,
                   require_json: bool = False,
                   return_records: bool = False) -> list[Any]:
    """Run children serially, preserving output, failures, and not-run entries."""
    records: list[dict[str, Any]] = []
    statuses: list[int] = []
    expected = set(expected_affinity) if expected_affinity is not None else None
    for index, command in enumerate(commands):
        record: dict[str, Any] = {"index": index, "command": list(command)}
        try:
            if expected is None:
                kwargs: dict[str, Any] = {"cwd": cwd, "env": dict(env) if env else None,
                                          "check": False, "text": True,
                                          "capture_output": True}
                if timeout_s is not None:
                    kwargs["timeout"] = timeout_s
                result = runner(list(command), **kwargs)
                code = result.returncode
                stdout, stderr = getattr(result, "stdout", "") or "", getattr(result, "stderr", "") or ""
            else:
                process = subprocess.Popen(
                    list(command), cwd=cwd, env=dict(env) if env else None,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    start_new_session=(os.name == "posix"),
                )
                try:
                    affinity = os.sched_getaffinity(process.pid)
                    verify_affinity(expected, affinity)
                except (ContractError, OSError, AttributeError) as exc:
                    _terminate_process_group(process)
                    stdout, stderr = process.communicate()
                    _archive_child(output_dir, index, stdout, stderr)
                    record.update({"status": "error", "returncode": -1,
                                   "error": "child affinity verification failed",
                                   "detail": type(exc).__name__, "stdout": stdout, "stderr": stderr})
                    statuses.append(-1)
                    records.append(record)
                    break
                try:
                    stdout, stderr = process.communicate(timeout=timeout_s)
                    code = process.returncode
                except subprocess.TimeoutExpired as exc:
                    _terminate_process_group(process)
                    stdout, stderr = process.communicate()
                    stdout, stderr = stdout or exc.stdout or "", stderr or exc.stderr or ""
                    _archive_child(output_dir, index, stdout, stderr)
                    record.update({"status": "failed", "returncode": -9, "error": "timeout",
                                   "stdout": stdout, "stderr": stderr})
                    statuses.append(-9)
                    records.append(record)
                    break
                record["affinity_verified"] = True
            record.update({"returncode": code, "stdout": stdout, "stderr": stderr,
                           "status": "completed" if code == 0 else "failed"})
            if require_json and code == 0:
                try:
                    payload = json.loads(stdout.strip().splitlines()[-1])
                    if not isinstance(payload, Mapping):
                        raise ValueError("child JSON is not an object")
                    payload_affinity = payload.get("affinity")
                    if not isinstance(payload_affinity, list):
                        raise ValueError("child affinity evidence is missing")
                    if expected is not None:
                        verify_affinity(expected, payload_affinity)
                        observations = payload.get("thread_affinity_observations")
                        budget = payload.get("declared_thread_budget")
                        observed_count = payload.get("observed_thread_count")
                        if (not isinstance(observations, Mapping) or
                                not isinstance(budget, int) or isinstance(budget, bool) or budget < 1):
                            raise ValueError("child thread-budget/task evidence is missing")
                        final_tasks = observations.get("after_timing")
                        if not isinstance(final_tasks, list) or not final_tasks:
                            raise ValueError("child post-timing task affinity evidence is missing")
                        for stage in ("after_correctness", "after_warmup", "after_timing"):
                            tasks = observations.get(stage)
                            if not isinstance(tasks, list) or not tasks:
                                raise ValueError(f"child {stage} task affinity evidence is missing")
                            for task in tasks:
                                if (not isinstance(task, Mapping) or
                                        not isinstance(task.get("cpus"), list)):
                                    raise ValueError("child task affinity observation is malformed")
                                verify_affinity(expected, task["cpus"])
                        if type(observed_count) is not int or observed_count != len(final_tasks):
                            raise ValueError("child observed task count is inconsistent")
                    record["payload"] = payload
                except (json.JSONDecodeError, IndexError, TypeError, ValueError, ContractError) as exc:
                    record.update({"status": "error", "returncode": -1,
                                   "error": "invalid child JSON/affinity", "detail": str(exc)})
                    code = -1
            _archive_child(output_dir, index, stdout, stderr)
            statuses.append(code)
            records.append(record)
            if code:
                break
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            _archive_child(output_dir, index, stdout, stderr)
            record.update({"status": "failed", "returncode": -9, "error": "timeout",
                           "stdout": stdout, "stderr": stderr})
            statuses.append(-9)
            records.append(record)
            break
        except OSError as exc:
            record.update({"status": "error", "returncode": -1, "error": type(exc).__name__})
            statuses.append(-1)
            records.append(record)
            break
    for index in range(len(records), len(commands)):
        records.append({"index": index, "command": list(commands[index]), "status": "not_run"})
    return records if return_records else statuses


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate a child process group so compiler/worker descendants do not leak."""
    try:
        if os.name == "posix" and process.pid > 0:
            os.killpg(process.pid, 15)
        else:
            process.terminate()
        process.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix" and process.pid > 0:
                os.killpg(process.pid, 9)
            else:
                process.kill()
            process.wait()
        except OSError:
            pass


def read_cpu_topology(sys_cpu_root: Path = Path("/sys/devices/system/cpu")) -> dict[int, dict[str, Any]]:
    topology: dict[int, dict[str, Any]] = {}
    for cpu_path in sorted(sys_cpu_root.glob("cpu[0-9]*")):
        try:
            cpu = int(cpu_path.name[3:])
            core = (cpu_path / "topology/core_id").read_text().strip()
            package = (cpu_path / "topology/physical_package_id").read_text().strip()
            siblings = parse_cpu_set((cpu_path / "topology/thread_siblings_list").read_text())
            nodes = sorted(path.name for path in cpu_path.glob("node[0-9]*"))
            if not core or not package or not siblings:
                continue
            topology[cpu] = {"core_id": core, "physical_package_id": package,
                             "siblings": siblings, "numa_node": nodes[0] if nodes else None}
        except (OSError, ValueError):
            continue
    return topology


def _cgroup2_mount(mountinfo: str) -> Path | None:
    for line in mountinfo.splitlines():
        if " - cgroup2 " not in line:
            continue
        fields = line.split(" - ", 1)[0].split()
        if len(fields) >= 5:
            return Path(fields[4])
    return None


def read_cgroup_limits(*, cgroup_text: str | None = None, mountinfo_text: str | None = None,
                       cgroup_mount: Path | None = None) -> dict[str, Any]:
    """Resolve process cgroup and all ancestors; absent root limits mean unlimited."""
    cgroup_text = cgroup_text if cgroup_text is not None else Path("/proc/self/cgroup").read_text()
    mountinfo_text = mountinfo_text if mountinfo_text is not None else Path("/proc/self/mountinfo").read_text()
    relative = None
    for line in cgroup_text.splitlines():
        hierarchy, _, path = line.partition("::")
        if hierarchy == "0" and path:
            relative = Path(path)
            break
    mount = cgroup_mount or _cgroup2_mount(mountinfo_text)
    if relative is None or mount is None:
        return {"cpuset": None, "quota_cpus": None, "paths": [],
                "reasons": ["cgroup v2 process path is unavailable"]}
    target = mount / relative.relative_to("/")
    paths: list[str] = []
    cpus: set[int] | None = None
    quotas: list[float] = []
    quota_known = False
    current = target
    while True:
        paths.append(str(current))
        for name in ("cpuset.cpus.effective", "cpuset.cpus"):
            candidate = current / name
            if candidate.exists():
                try:
                    value = parse_cpu_set(candidate.read_text().strip())
                except (OSError, ValueError) as exc:
                    return {"cpuset": None, "quota_cpus": None, "paths": paths,
                            "reasons": [f"unreadable cgroup cpuset: {type(exc).__name__}"]}
                if cpus is None:
                    cpus = set(value)
                else:
                    cpus &= set(value)
                break
        quota_file = current / "cpu.max"
        if quota_file.exists():
            quota_known = True
            try:
                fields = quota_file.read_text().split()
                if len(fields) != 2:
                    raise ValueError("cpu.max requires quota and period")
                if fields[0] != "max":
                    quota, period = int(fields[0]), int(fields[1])
                    if quota <= 0 or period <= 0:
                        raise ValueError("invalid cpu.max")
                    quotas.append(quota / period)
            except (OSError, ValueError) as exc:
                return {"cpuset": tuple(sorted(cpus)) if cpus is not None else None,
                        "quota_cpus": None, "paths": paths,
                        "reasons": [f"unreadable cgroup quota: {type(exc).__name__}"]}
        if current == mount:
            break
        if current.parent == current:
            break
        current = current.parent
    return {"cpuset": tuple(sorted(cpus)) if cpus is not None else None,
            "quota_cpus": min(quotas) if quotas else None, "quota_known": quota_known,
            "paths": paths, "reasons": []}


def read_live_resource_snapshot() -> dict[str, Any]:
    """Read production resource evidence from this process and live kernel paths."""
    before = parse_proc_stat(Path("/proc/stat").read_text(encoding="utf-8"))
    topology = read_cpu_topology()
    cgroup = read_cgroup_limits()
    sched = tuple(sorted(os.sched_getaffinity(0)))
    allowed = set(sched)
    if cgroup["cpuset"] is not None:
        allowed &= set(cgroup["cpuset"])
    return {"before": before, "topology": topology, "cgroup": cgroup,
            "sched_affinity": sched, "allowed": tuple(sorted(allowed))}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit("use scripts/format_small_work_results.py or import small_work")
