"""Correctness adapter for the crate-owned einsum lib-test probe."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from small_work import run_sequential, timing_statistics

PREFIX = "TENFERRO_EINSUM_PROBE_JSON "
ENTRYPOINT = "concrete::probes::component_probe_entrypoint"


def probe_records(stdout: str) -> list[dict[str, Any]]:
    records = []
    for line in stdout.splitlines():
        # libtest may print its test name before the first tagged record.
        if PREFIX in line:
            value = json.loads(line.split(PREFIX, 1)[1])
            if not isinstance(value, dict):
                raise ValueError("component record is not an object")
            records.append(value)
    if not records:
        raise ValueError("component probe emitted no tagged records")
    return records


def check_components(binary: Path, archive: Path, timeout: float,
                     allocation_iterations: int | None = None, *,
                     protocol: dict[str, Any] | None = None,
                     expected_affinity: set[int] | None = None) -> dict[str, Any]:
    """Check exported cases and optionally collect separate caller allocation diagnostics."""
    if allocation_iterations is not None and (type(allocation_iterations) is not int or allocation_iterations < 1):
        raise ValueError("allocation iterations must be a positive integer")
    if protocol is not None and (allocation_iterations is not None or not expected_affinity):
        raise ValueError("component timing requires affinity and excludes allocation diagnostics")
    result: dict[str, Any] = {"status": "FAILED", "contracts": [], "records": [],
                              "commands": [], "errors": []}
    command = [str(binary.resolve()), "--exact", ENTRYPOINT, "--ignored",
               "--nocapture", "--test-threads=1"]
    environment = {k: v for k, v in os.environ.items() if not k.startswith("TENFERRO_PROBE_")}

    def invoke(mode: str, case: str, stage: str | None = None, *,
               iterations: int | None = None, samples: int = 1, minimum_ns: int = 1) -> list[dict[str, Any]]:
        env = {**environment, "TENFERRO_PROBE_MODE": mode, "TENFERRO_PROBE_CASE": case}
        if stage is not None:
            env.update(TENFERRO_PROBE_STAGE=stage, TENFERRO_PROBE_ITERATIONS=str(iterations if iterations is not None else allocation_iterations))
        if mode == "timed":
            env.update(TENFERRO_PROBE_SAMPLES=str(samples), TENFERRO_PROBE_MIN_AGGREGATE_NS=str(minimum_ns))
        child = run_sequential([command], env=env,
                               timeout_s=timeout, expected_affinity=expected_affinity,
                               output_dir=archive / str(len(result["commands"])),
                               return_records=True)[0]
        child.update(mode=mode, case=case, stage=stage, iterations=iterations,
                     requested_samples=samples, minimum_ns=minimum_ns)
        result["commands"].append(child)
        records = probe_records(child["stdout"])
        if child["status"] != "completed":
            if mode == "timed" and any(record.get("invalid_reason") == "under_duration" for record in records):
                result["status"] = "INCONCLUSIVE"
            raise ValueError(f"component {mode}/{case} child failed: {child.get('error', child.get('returncode'))}")
        return records

    try:
        contracts = invoke("contract", "all")
        result["contracts"] = contracts
        expected: dict[str, set[str]] = {}
        for contract in contracts:
            case, stage = contract.get("case_id"), contract.get("stage")
            if (contract.get("kind") != "contract" or
                    contract.get("schema") != "tenferro.einsum.component-probe.v1" or
                    not isinstance(case, str) or not case or
                    not isinstance(stage, str) or not stage):
                raise ValueError("invalid component contract identity")
            stages = expected.setdefault(case, set())
            if stage in stages:
                raise ValueError(f"duplicate component contract: {case}/{stage}")
            stages.add(stage)
        for case, stages in expected.items():
            records = invoke("correctness", case)
            result["records"].extend(records)
            observed = []
            for record in records:
                if (record.get("kind") != "correctness" or record.get("case_id") != case or
                        record.get("ok") is not True or not isinstance(record.get("stage"), str)):
                    raise ValueError(f"invalid correctness record for {case}")
                observed.append(record["stage"])
            if len(observed) != len(stages) or set(observed) != stages:
                raise ValueError(f"missing, duplicate or unexpected correctness stages for {case}")
        result["status"] = "CORRECTNESS_ONLY"
        if protocol is not None:
            result["status"] = "FAILED"
            result["timings"] = []
            target = protocol["calibration"]["target_ns"]
            warmups, sample_count = protocol["warmups"], protocol["samples_per_process"]
            process_count = protocol["independent_processes"]
            for contract in contracts:
                case, stage = contract["case_id"], contract["stage"]
                def checked_timing(iterations: int, count: int, minimum: int) -> list[dict[str, Any]]:
                    records = invoke("timed", case, stage, iterations=iterations, samples=count, minimum_ns=minimum)
                    if len(records) != count:
                        raise ValueError(f"incomplete timing samples for {case}/{stage}")
                    for index, record in enumerate(records):
                        if (record.get("kind") != "timing" or record.get("case_id") != case or
                                record.get("stage") != stage or record.get("valid") is not True or
                                record.get("invalid_reason") is not None):
                            raise ValueError(f"invalid timing record for {case}/{stage}")
                        for key in ("sample", "iterations", "completed_iterations", "elapsed_ns"):
                            if type(record.get(key)) is not int:
                                raise ValueError(f"invalid timing integer {key}")
                        if (record["sample"] != index or record["iterations"] != iterations or
                                record["completed_iterations"] != iterations or record["elapsed_ns"] < minimum):
                            raise ValueError(f"timing identity, count or duration mismatch for {case}/{stage}")
                    return records
                iterations = 1
                # Leave headroom without weakening the measured-duration threshold.
                while checked_timing(iterations, 1, 1)[0]["elapsed_ns"] < 2 * target:
                    if iterations >= 1 << 30:
                        raise ValueError(f"calibration limit reached for {case}/{stage}")
                    iterations *= 2
                normalized = []
                for process in range(process_count):
                    records = checked_timing(iterations, warmups + sample_count, target)
                    normalized.extend({"process_index": process, "sample_index": index,
                                       "iterations": iterations, "elapsed_ns": record["elapsed_ns"]}
                                      for index, record in enumerate(records[warmups:]))
                if type(contract.get("calls_per_workflow")) is not int or contract["calls_per_workflow"] < 1:
                    raise ValueError(f"invalid calls_per_workflow for {case}/{stage}")
                statistics = timing_statistics(normalized, contract["calls_per_workflow"],
                                               min_processes=process_count,
                                               min_samples_per_process=sample_count)
                result["timings"].append({"case_id": case, "stage": stage, "samples": normalized,
                                          "statistics": statistics})
                if statistics["process_median_cov"] > protocol["noise_policy"]["process_median_cov_max"]:
                    result["status"] = "INCONCLUSIVE"
                    raise ValueError(f"process-median variability exceeds protocol for {case}/{stage}")
            result["status"] = "TIMING_DIAGNOSTIC"
        if allocation_iterations is not None:
            result["status"] = "FAILED"
            result["allocations"] = []
            result["allocation_scope"] = "caller-thread allocation calls and requested-byte traffic; excludes fixture setup, worker/provider/native allocations; not retained memory or latency"
            for contract in contracts:
                case, stage = contract["case_id"], contract["stage"]
                records = invoke("alloc", case, stage)
                result["allocations"].extend(records)
                if len(records) != 1:
                    raise ValueError(f"expected one allocation record for {case}/{stage}")
                record = records[0]
                if (record.get("kind") != "allocation" or record.get("case_id") != case or
                        record.get("stage") != stage or record.get("valid") is not True or
                        record.get("invalid_reason") is not None):
                    raise ValueError(f"invalid allocation record for {case}/{stage}")
                for key in ("iterations", "completed_iterations", "allocation_calls", "requested_bytes"):
                    if type(record.get(key)) is not int or record[key] < 0:
                        raise ValueError(f"invalid allocation counter {key}")
                if record["iterations"] != allocation_iterations or record["completed_iterations"] != allocation_iterations:
                    raise ValueError(f"incomplete allocation iterations for {case}/{stage}")
            result["status"] = "ALLOCATION_DIAGNOSTIC"
    except (OSError, ValueError) as exc:
        result["errors"].append(str(exc))
    return result
