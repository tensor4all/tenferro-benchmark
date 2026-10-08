#!/usr/bin/env python3
"""Versioned expected/selected/executed/unsupported/failed/noisy case tracking.

Every runner that uses this writes ``case_status.json`` next to its raw rows.
The accounting is deliberately one-directional: a case counts as *executed*
(and therefore as covered) only when a row for it exists and passed its
numerical check. Missing, failed (including compile/launch failures that
produce no row) and unsupported cases are listed separately and can never be
promoted to success. ``noisy`` is a descriptive subset of executed.

Case keys are ``<case_id>`` or ``<case_id>[<provider>]`` when one case runs on
several providers, suffixed with ``@t<threads>``.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Iterable

CASE_STATUS_VERSION = 1
# Descriptive label only; never a gate and never a reason to drop a row.
NOISY_COV = 0.10

PASSED = {"passed", "observed"}
UNSUPPORTED = {"unsupported"}


def case_key(case_id: str, threads: int, provider: str | None = None) -> str:
    base = f"{case_id}[{provider}]" if provider else case_id
    return f"{base}@t{threads}"


def coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if mean else None


def build_case_status(*, suite_id: str, manifest_version: int, coverage: str,
                      effort: str, expected: Iterable[str], selected: Iterable[str],
                      rows: Iterable[dict], selection_filter: str | None) -> dict:
    """Classify rows (each with ``case_key`` and ``status``) against the plan.

    ``status`` is ``passed``/``observed`` (success), ``unsupported``, or anything
    else (failure). ``cov`` may be given for noisy labelling.
    """
    expected = list(dict.fromkeys(expected))
    selected = list(dict.fromkeys(selected))
    by_key: dict[str, dict] = {}
    duplicates = []
    for row in rows:
        key = row["case_key"]
        if key in by_key:
            duplicates.append(key)
        by_key[key] = row
    executed, unsupported, failed, noisy = [], [], [], []
    for key in selected:
        row = by_key.get(key)
        if row is None:
            continue
        status = row.get("status")
        if status in PASSED:
            executed.append(key)
            cov = row.get("cov")
            if cov is not None and cov > NOISY_COV:
                noisy.append(key)
        elif status in UNSUPPORTED:
            unsupported.append(key)
        else:
            failed.append(key)
    accounted = set(executed) | set(unsupported) | set(failed)
    missing = [k for k in selected if k not in accounted]
    unexpected = sorted(set(by_key) - set(selected))
    not_selected = [k for k in expected if k not in set(selected)]
    return {
        "case_status_version": CASE_STATUS_VERSION,
        "suite_id": suite_id,
        "manifest_version": manifest_version,
        "coverage": coverage,
        "effort": effort,
        "selection_filter": selection_filter,
        "counts": {
            "expected": len(expected), "selected": len(selected),
            "executed": len(executed), "unsupported": len(unsupported),
            "failed": len(failed), "missing": len(missing), "noisy": len(noisy),
        },
        # A run is complete only when everything expected was selected and
        # every selected case was accounted for without failure.
        "complete": not not_selected and not missing and not failed,
        "expected": expected,
        "selected": selected,
        "not_selected": not_selected,
        "executed": executed,
        "unsupported": unsupported,
        "failed": failed,
        "missing": missing,
        "noisy": noisy,
        "unexpected_rows": unexpected,
        "duplicate_rows": duplicates,
    }


def merge_case_status(parts: list[dict]) -> dict:
    """Merge per-thread-count status files of one run into one."""
    if not parts:
        raise ValueError("no case status parts")
    first = parts[0]
    for part in parts[1:]:
        for field in ("suite_id", "manifest_version", "coverage", "effort"):
            if part[field] != first[field]:
                raise ValueError(f"cannot merge case status with different {field}")
    merged = {k: first[k] for k in ("case_status_version", "suite_id", "manifest_version",
                                    "coverage", "effort", "selection_filter")}
    lists = ("expected", "selected", "not_selected", "executed", "unsupported", "failed",
             "missing", "noisy", "unexpected_rows", "duplicate_rows")
    for field in lists:
        merged[field] = [k for part in parts for k in part[field]]
    merged["counts"] = {f: len(merged[f]) for f in
                        ("expected", "selected", "executed", "unsupported", "failed",
                         "missing", "noisy")}
    merged["complete"] = all(part["complete"] for part in parts)
    return merged


def write_case_status(path: Path, status: dict) -> None:
    path.write_text(json.dumps(status, indent=2) + "\n")


def markdown_section(status: dict) -> list[str]:
    counts = status["counts"]
    lines = [
        "## Case status",
        "",
        f"- Manifest: `{status['suite_id']}` version {status['manifest_version']}, "
        f"coverage `{status['coverage']}`, effort `{status['effort']}`"
        + (f", filter `{status['selection_filter']}`" if status["selection_filter"] else ""),
        f"- Expected {counts['expected']}, selected {counts['selected']}, executed "
        f"{counts['executed']}, unsupported {counts['unsupported']}, failed "
        f"{counts['failed']}, missing {counts['missing']}, noisy {counts['noisy']}.",
        f"- Complete: **{'yes' if status['complete'] else 'NO'}** — only executed "
        "cases count as covered; unsupported, failed, missing and unselected cases never do.",
    ]
    for field in ("not_selected", "missing", "failed", "unsupported", "noisy"):
        if status[field]:
            shown = ", ".join(f"`{k}`" for k in status[field][:40])
            more = f" (+{len(status[field]) - 40} more)" if len(status[field]) > 40 else ""
            lines.append(f"- {field.replace('_', ' ').capitalize()}: {shown}{more}")
    return lines + [""]
