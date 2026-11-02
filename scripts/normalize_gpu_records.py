#!/usr/bin/env python3
"""Fold a family's raw GPU rows into the routine record contract.

The routine accounting, the publication bundle and the dashboard summary read one
shared GPU record (``problem_id``, ``backend``, ``status``, ``timing.median_ms``,
``timing.timed_runs``, ``verification.status``). Some families keep a richer
shape of their own — ``case_id`` with batched per-operation samples, or
``pattern_id`` with one median per row — and their reports stay in that shape.
This script writes the routine-contract view of the same measurements next to
them, so the routine path never needs a second accounting or bundle rule.

    python3 scripts/normalize_gpu_records.py --output records.jsonl \
        --summary report.summary.json --run-metadata run.yaml raw1.jsonl raw2.jsonl

Medians stay in milliseconds; a batched family's per-operation duration is
already milliseconds after dividing by the declared operations per sample.
Unsupported and failed rows keep their status and reason so the accounting can
classify them; a row that cannot be folded fails here instead of silently
becoming a measurement.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import report_summary  # noqa: E402
from format_gpu_results import build_summary  # noqa: E402

TIMING_SCOPE = "steady_state_host_api_plus_device_sync"


def _spread(values):
    if len(values) < 2:
        return 0.0 if values else None
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


CORRECTNESS_TO_VERIFICATION = {"passed": "passed", "skipped": "skipped",
                               "not_applicable": "not_applicable"}


def _verification(row):
    recorded = row.get("verification") or {}
    if recorded.get("status"):
        return recorded["status"]
    if row.get("correctness"):
        # A row whose correctness was never evaluated is not a wrong row. The
        # three states have to survive the fold: collapsing `skipped` into
        # `failed` makes the accounting print a numerical failure for a cell
        # that was never measured.
        return CORRECTNESS_TO_VERIFICATION.get(row["correctness"], "failed")
    error = recorded.get("max_error_over_tolerance")
    if isinstance(error, (int, float)) and math.isfinite(error):
        return "passed" if error <= 1 else "failed"
    return "not_applicable"


def _reason(row):
    """The row's own explanation, whichever field its family runner used.

    Families disagree: some write ``reason``, the permutation runners write the
    explanation into ``notes``, and a few only have ``error``. The accounting
    prefers ``execution.unsupported_reason`` and falls back to
    ``verification.reason``, so a dropped reason turns a benign skip into a
    failure whose cause is lost.
    """
    recorded = row.get("verification") or {}
    for value in (row.get("reason"), row.get("notes"), row.get("error"),
                  recorded.get("reason")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def fold(row, suite_id=None):
    """One routine-contract row from any family record that carries a measurement.

    The row's own ``suite_id`` wins. The public API families' raw rows carry a
    ``case_id`` and no ``suite_id``, so the run's identity comes from ``run.yaml``
    through ``suite_id`` instead of a second copy in every row.
    """
    suite_id = row.get("suite_id") or suite_id
    if not suite_id:
        raise ValueError(f"record without a suite identity: {sorted(row)}")
    problem = row.get("problem_id") or row.get("case_id") or row.get("pattern_id")
    if not problem:
        raise ValueError(f"record without a case identity: {sorted(row)}")
    timing = row.get("timing") or {}
    samples = timing.get("batch_ms")
    if samples:
        count = timing.get("operations_per_sample") or 1
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"{problem}: invalid operations_per_sample")
        values = [value / count for value in samples]
    elif isinstance(row.get("median_ms"), (int, float)):
        values = [row["median_ms"]]
        p25, p75 = row.get("p25_ms"), row.get("p75_ms")
        if isinstance(p25, (int, float)) and isinstance(p75, (int, float)):
            timing["iqr_ms_recorded"] = p75 - p25
    else:
        values = []
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    status = row.get("status")
    reason = _reason(row)
    median = statistics.median(values) if values else None
    iqr = timing.get("iqr_ms_recorded", _spread(values))
    if status not in {"ok", "unsupported", "skipped", "not_configured", "verification_only"}:
        median = None
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "problem_id": problem,
        "op": row.get("op") or "permutation",
        "backend": row["backend"],
        "status": status,
        "timing": {
            "warmup_runs": row.get("warmup") or timing.get("warmups") or 0,
            "timed_runs": len(values),
            "compile_time_ms": None, "first_run_ms": None,
            "median_ms": median, "min_ms": min(values) if values else None,
            "p95_ms": None, "iqr_ms": iqr,
            "timing_scope": TIMING_SCOPE,
        },
        "verification": {
            "status": _verification(row),
            "reference_backend": (row.get("verification") or {}).get("reference") or "cpu_fp64",
            "rtol": None, "atol": None,
            "reason": reason,
        },
        "execution": {
            "device": "cuda", "device_ordinal": int(row.get("device_ordinal") or 0),
            "execution_path": row.get("runner") or "family-runner",
            "synchronization": row.get("synchronization") or "recorded by the family runner",
            "layout": "", "dtype": str(row.get("dtype") or ""),
            "unsupported_reason": reason if status != "ok" else row.get("reason"),
            "notes": json.dumps({"sampling": {"operations_per_sample": timing.get("operations_per_sample") or 1,
                                              "batch_ms": values}}),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    metadata = yaml.safe_load(args.run_metadata.read_text()) if args.run_metadata else None
    suite_id = (metadata or {}).get("suite_id")
    rows = [fold(json.loads(line), suite_id) for path in args.inputs
            for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("no raw rows to normalize")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    if args.summary:
        report_summary.write_document(args.summary, build_summary(rows, metadata))


if __name__ == "__main__":
    main()
