#!/usr/bin/env python3
"""Record, per GPU record, the logical traffic rate it implies and its ratio to a copy.

Two numbers are recorded next to every timing, and neither of them decides anything:

* the **logical bytes** a case must move per operation, sized by
  ``scripts/gpu_case_traffic.py`` from the arrays the fixture step persists (one read
  of every declared input, one write of every declared output);
* the **copy ratio**: those bytes divided by the case's per operation median time,
  relative to the traffic rate one plain device copy reached on the same device in the
  same run (``scripts/gpu_copy_calibration.py``).

The ratio is a diagnostic, not a bandwidth measurement. Logical bytes are not device
DRAM traffic: a contraction whose operands stay resident in cache moves far less DRAM
than its array sizes suggest, so a correct case can legitimately show a ratio of many
times the copy rate. An intermediate-materialising kernel moves more than the logical
minimum for the same reason in the other direction. Nothing here is a plausible-time
check, and nothing here changes a record's status, its timing or the run's exit code.

Usage, one arm at a time, from the run script::

    python3 scripts/gpu_bandwidth_guard.py --run-metadata run.yaml arm.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

CALIBRATION_KEY = "gpu_memory_calibration"
GIGABYTE = 1e9
ABOVE_COPY_RATE = "above_copy_rate"
WITHIN_COPY_RATE = "within_copy_rate"
NOT_CHECKED = "not_checked"


def median_ms(values):
    finite = [float(value) for value in values
              if isinstance(value, (int, float)) and math.isfinite(value)]
    return statistics.median(finite) if finite else None


def per_operation_ms(record):
    """Median time of one operation, from the batch samples the arm recorded.

    The samples time a batch of ``operations_per_sample`` operations, so a raw
    sample is not the quantity a reader compares.
    """
    timing = record.get("timing") or {}
    samples = timing.get("batch_ms") or []
    count = timing.get("operations_per_sample") or 1
    if not isinstance(count, int) or count < 1:
        return None
    median = median_ms(samples)
    return None if median is None else median / count


def traffic_bytes_per_operation(record):
    traffic = record.get("traffic") or {}
    value = traffic.get("bytes_per_operation")
    if not isinstance(value, int) or value <= 0:
        return None
    return value


def implied_bytes_per_second(traffic_bytes, operation_ms):
    if not isinstance(traffic_bytes, int) or traffic_bytes <= 0:
        raise ValueError("an implied rate needs a positive traffic size")
    if not isinstance(operation_ms, (int, float)) or not math.isfinite(operation_ms):
        raise ValueError("an implied rate needs a finite per operation time")
    if operation_ms <= 0:
        raise ValueError("an implied rate needs a positive per operation time")
    return traffic_bytes / (operation_ms / 1000.0)


def calibration_of(metadata):
    """The measured copy rate of this run, or ``None`` when it is absent or unusable.

    An unusable calibration is reported by returning ``None``: the ratio is then
    recorded as unavailable instead of failing the run that produced the records.
    """
    calibration = (metadata or {}).get(CALIBRATION_KEY)
    if not isinstance(calibration, dict):
        return None
    rate = calibration.get("bytes_per_second")
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        return None
    return {**calibration, "bytes_per_second": float(rate)}


def evaluate(record, calibration):
    """One diagnostic decision for a record, or ``None`` when the record is not timed."""
    status = record.get("status")
    if status != "ok":
        return None
    case = record.get("case_id") or record.get("problem_id") or "unknown case"
    backend = record.get("backend") or "unknown backend"
    traffic = traffic_bytes_per_operation(record)
    operation_ms = per_operation_ms(record)
    decision = {
        "traffic_bytes_per_operation": traffic,
        "median_ms_per_operation": operation_ms,
        "copy_bytes_per_second": (calibration or {}).get("bytes_per_second"),
        "copy_rate_ratio": None,
        "reason": None,
    }
    if calibration is None:
        decision["reason"] = ("this run records no usable device copy rate, "
                              "so the logical bytes have no reference to be compared with")
        return {**decision, "decision": NOT_CHECKED}
    if traffic is None:
        decision["reason"] = (record.get("traffic") or {}).get("unknown_reason") \
            or "record carries no logical traffic size"
        return {**decision, "decision": NOT_CHECKED}
    if operation_ms is None or operation_ms <= 0:
        decision["reason"] = "record carries no usable per operation time"
        return {**decision, "decision": NOT_CHECKED}
    implied = implied_bytes_per_second(traffic, operation_ms)
    ratio = implied / calibration["bytes_per_second"]
    decision["implied_bytes_per_second"] = implied
    decision["copy_rate_ratio"] = ratio
    if ratio > 1.0:
        decision["reason"] = (
            f"case {case} on {backend} moves {traffic} logical bytes per operation in "
            f"{operation_ms * 1000.0:.3f} us, {ratio:.1f} times the measured device copy "
            f"rate of {calibration['bytes_per_second'] / GIGABYTE:.1f} GB/s; logical bytes "
            "are not device DRAM traffic, so this is recorded, not judged"
        )
        return {**decision, "decision": ABOVE_COPY_RATE}
    return {**decision, "decision": WITHIN_COPY_RATE}


def apply(record, calibration):
    """Record the diagnostic block on one record; return the decision.

    The record's ``status``, ``timing`` and ``reason`` are never touched: the block is
    appended to what the arm measured.
    """
    decision = evaluate(record, calibration)
    if decision is None:
        return None
    block = {
        "decision": decision["decision"],
        "traffic_bytes_per_operation": decision["traffic_bytes_per_operation"],
        "median_ms_per_operation": decision["median_ms_per_operation"],
        "copy_bytes_per_second": decision["copy_bytes_per_second"],
        "copy_rate_ratio": decision["copy_rate_ratio"],
    }
    if "implied_bytes_per_second" in decision:
        block["implied_bytes_per_second"] = decision["implied_bytes_per_second"]
    if decision["reason"]:
        block["reason"] = decision["reason"]
    record["bandwidth_guard"] = block
    return decision


def apply_to_rows(rows, calibration):
    """Record the diagnostic block on every timed row; return the decisions."""
    decisions = []
    for row in rows:
        decision = apply(row, calibration)
        if decision is not None:
            decisions.append(decision)
    return decisions


def load_calibration(path):
    metadata = yaml.safe_load(Path(path).read_text()) or {}
    return calibration_of(metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("records", nargs="+", type=Path)
    args = parser.parse_args()
    calibration = load_calibration(args.run_metadata)
    if calibration is None:
        print(f"{args.run_metadata}: no usable {CALIBRATION_KEY}; the copy ratio is "
              "recorded as unavailable for this run", file=sys.stderr)
    for path in args.records:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        decisions = apply_to_rows(rows, calibration)
        above = [decision for decision in decisions if decision["decision"] == ABOVE_COPY_RATE]
        for decision in above:
            print(decision["reason"], file=sys.stderr)
        path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    # Recording never fails a run: the exit code is not a judgement about bandwidth.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
