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

# What a record's `compile_time_ms` actually holds. A Rust public-API arm cannot time
# NVRTC on its own: it records the module count and the untimed case wall around a
# compile, which also holds that case's fixture read, transfers and verification. The
# arms publish `compile_time_ms` as null for that reason, so a consumer must be told
# which of the two it is holding rather than inferring "compile time" from the name.
COMPILE_TIME_SCOPE = ("the untimed case wall of the case that compiled, including fixture read, "
                      "transfers and verification; not a compile interval")
COMPILE_TIME_ABSENT = ("not measured: no compile interval was recorded; `compile_modules` counts "
                       "the NVRTC modules and the untimed wall is the case wall")
# A record without `compile_modules` is not an NVRTC arm. The Python arms time their own
# compile call and nothing else (jax-cuda-jit: `jax.jit(fn).lower(*inputs).compile()`), and
# record 0.0 when the arm has no compile step, so their value is a real compile interval
# and must not be labelled as the legacy untimed wall (#363).
FRAMEWORK_COMPILE_SCOPE = ("the arm's own lowering/compile interval, timed around its compile call "
                           "only (jax-cuda-jit: jax.jit(fn).lower(*inputs).compile()); 0.0 when the "
                           "arm has no compile step")


def compile_time_scope(row, recorded_compile_ms):
    """What one record's `compile_time_ms` holds, decided by the kind of record it is.

    Only an NVRTC record (one that carries `compile_modules`) can hold the legacy untimed
    case wall; any other recorded value is the arm's own compile interval.
    """
    if recorded_compile_ms is None:
        return COMPILE_TIME_ABSENT
    if "compile_modules" in row:
        return COMPILE_TIME_SCOPE
    return FRAMEWORK_COMPILE_SCOPE


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
    recorded_compile_ms = timing.get("compile_time_ms")
    record = {
        "schema_version": 1,
        "suite_id": suite_id,
        "problem_id": problem,
        "op": row.get("op") or "permutation",
        "backend": row["backend"],
        "status": status,
        "timing": {
            "warmup_runs": row.get("warmup") or timing.get("warmups") or 0,
            "timed_runs": len(values),
            "compile_time_ms": recorded_compile_ms,
            "compile_time_scope": compile_time_scope(row, recorded_compile_ms),
            "first_call_ms": timing.get("first_call_ms"),
            "first_run_ms": None,
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
    # The compile accounting only exists on the arms that compile NVRTC modules, so a
    # family that does not carry it keeps byte-identical records.
    if "compile_modules" in row:
        record["timing"]["compile_cache"] = row.get("compile_cache")
        record["timing"]["compile_modules"] = row.get("compile_modules")
    return record


def nvrtc_compile_time_ms(row):
    """The compile-time field one row recorded, or None when it recorded none.

    A case that compiled but recorded no sampling keeps its value at the top level, because
    a null timing block has nowhere else to put it. Only rows that carry
    ``compile_modules`` are NVRTC rows; a Python arm's compile is a different measurement
    and is not summed into this total. None rather than 0.0 for an absent value, so an arm
    that measured no compile interval cannot be summed into a compile total.
    """
    recorded = (row.get("timing") or {}).get("compile_time_ms")
    return recorded if recorded is not None else row.get("compile_time_ms")


def nvrtc_untimed_wall_ms(row):
    """The untimed case wall of one NVRTC row, whatever the record era calls the field.

    An arm that publishes no compile interval reports the same wall as the case's
    ``overhead.untimed_ms``; an older arm file reported it as ``compile_time_ms``. The
    total is the same either way, and it stays a wall: it holds fixture read, transfers
    and verification, so it is not a compile interval.
    """
    recorded = nvrtc_compile_time_ms(row)
    if recorded is not None:
        return recorded
    if (row.get("compile_modules") or 0) > 0:
        return (row.get("overhead") or {}).get("untimed_ms")
    return None


def compile_accounting(raw):
    """The run's own wall clock and compile totals, from what its arm files recorded.

    ``case_wall_seconds`` sums every record's own case wall clock, so the compile it
    carries is a share of that same denominator. ``compiling_processes`` counts the arm
    processes that compiled at least one module, not the modules. The compile numbers are
    read from the NVRTC arms only, and ``compile_time_scope`` says what
    ``compile_time_ms`` is so a consumer cannot read it as a compile interval.
    """
    nvrtc = [row for row in raw if "compile_modules" in row]
    if not nvrtc:
        return None
    recorded = [value for value in (nvrtc_compile_time_ms(row) for row in nvrtc)
                if value is not None]
    walls = [value for value in (nvrtc_untimed_wall_ms(row) for row in nvrtc)
             if value is not None]
    return {
        "case_wall_seconds": round(sum((row.get("overhead") or {}).get("case_wall_ms") or 0.0
                                      for row in raw) / 1e3, 3),
        "compiling_processes": len({row.get("backend") for row in nvrtc
                                    if (row.get("compile_modules") or 0) > 0}),
        "modules_compiled": sum(row.get("compile_modules") or 0 for row in nvrtc),
        "compile_time_ms": round(sum(recorded), 3) if recorded else None,
        "compile_time_scope": (COMPILE_TIME_SCOPE if recorded else COMPILE_TIME_ABSENT),
        "untimed_wall_seconds": round(sum(walls) / 1e3, 3),
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
    raw = [json.loads(line) for path in args.inputs
           for line in path.read_text().splitlines() if line.strip()]
    rows = [fold(row, suite_id) for row in raw]
    if not rows:
        raise SystemExit("no raw rows to normalize")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    if args.summary:
        report_summary.write_document(args.summary, build_summary(rows, metadata))
    # The arm files are the only place the compile totals exist, and they are written
    # after the run metadata is, so the totals are folded back into the run metadata
    # here. A run whose cache was warm records zeros rather than omitting the block.
    accounting = compile_accounting(raw)
    if accounting is not None and args.run_metadata is not None:
        metadata.setdefault("collection", {})["compile"] = accounting
        # Replace rather than truncate: an interrupted write would otherwise destroy
        # collected provenance that nothing can regenerate.
        staged = args.run_metadata.with_name(args.run_metadata.name + ".tmp")
        staged.write_text(yaml.safe_dump(metadata, sort_keys=False))
        staged.replace(args.run_metadata)


if __name__ == "__main__":
    main()
