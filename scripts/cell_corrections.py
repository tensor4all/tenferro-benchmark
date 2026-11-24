#!/usr/bin/env python3
"""Cells corrected or withdrawn after collection (``benchmarks/cell-corrections.yaml``).

The run record stays as collected. The page generators read a run's CSV through
:func:`read_rows`, which applies the registry to the listed rows:

- ``withdraw``: no median or IQR, status ``withdrawn``, plus the recorded reason;
- ``correct``: the re-measured median and IQR from the entry, status ``ok``, plus
  the reason and a ``corrected`` marker on the cell.

:func:`page_note` names every applied entry under the table, so a reader sees
why a cell differs from its run record. ``regenerate_report_summaries.py --check``
re-derives the same page from the same run plus this registry.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import yaml

from result_status import WITHDRAWN_STATUS

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmarks" / "cell-corrections.yaml"
KEY_FIELDS = ("benchmark", "dtype", "threads", "shape", "backend")
ACTIONS = ("withdraw", "correct")
CORRECTED_MARK = "corrected"


@lru_cache(maxsize=None)
def _entries(path: Path = REGISTRY) -> dict[tuple[str, ...], dict[str, str]]:
    if not path.exists():
        return {}
    document = yaml.safe_load(path.read_text()) or {}
    entries: dict[tuple[str, ...], dict[str, str]] = {}
    for cell in document.get("cells") or []:
        required = ["run", "action", "reason", "ref", "evidence", *KEY_FIELDS]
        if cell.get("action") == "correct":
            required += ["median_ms", "iqr_ms"]
        missing = [field for field in required if cell.get(field) in (None, "")]
        if missing:
            raise ValueError(f"{path}: corrected cell {cell!r} lacks {', '.join(missing)}")
        if cell["action"] not in ACTIONS:
            raise ValueError(f"{path}: action {cell['action']!r} is not one of {ACTIONS}")
        if not (path.resolve().parents[1] / str(cell["evidence"])).is_dir():
            raise ValueError(f"{path}: evidence {cell['evidence']} is not a directory")
        key = (str(cell["run"]).strip("/"), *(str(cell[field]) for field in KEY_FIELDS))
        if key in entries:
            raise ValueError(f"{path}: cell listed twice: {key}")
        entry = {"action": str(cell["action"]), "ref": str(cell["ref"]),
                 "reason": " ".join(str(cell["reason"]).split())}
        if cell["action"] == "correct":
            entry["median_ms"] = f"{float(cell['median_ms']):.6f}"
            entry["iqr_ms"] = f"{float(cell['iqr_ms']):.6f}"
        entries[key] = entry
    return entries


def run_key(csv_path: Path) -> str:
    """``<profile>/<family>/<suite>/<timestamp>`` of the run directory holding ``csv_path``."""
    return "/".join(Path(csv_path).resolve().parent.parts[-4:])


def correction(csv_path: Path, row: dict[str, str]) -> dict[str, str] | None:
    """The registry entry for one CSV row, if that cell is corrected or withdrawn."""
    from format_cpu_ops_results import normalize_backend

    values = {field: row.get(field, "") for field in KEY_FIELDS}
    values["backend"] = normalize_backend(row)
    return _entries().get((run_key(csv_path), *(values[field] for field in KEY_FIELDS)))


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """The CSV rows of a run with the registry applied."""
    rows = []
    with Path(csv_path).open(newline="") as f:
        for row in csv.DictReader(f):
            entry = correction(csv_path, row)
            if entry is not None:
                common = {"reason": entry["reason"], "correction_ref": entry["ref"],
                          "correction_action": entry["action"], "mean_us": ""}
                if entry["action"] == "withdraw":
                    row = {**row, **common, "median_ms": "", "iqr_ms": "", "status": WITHDRAWN_STATUS}
                else:
                    row = {**row, **common, "median_ms": entry["median_ms"],
                           "iqr_ms": entry["iqr_ms"], "status": "ok"}
            rows.append(row)
    return rows


def mark(row: dict[str, str], value: str) -> str:
    """A table cell's text, marked when the registry corrected its number."""
    return f"{value} ({CORRECTED_MARK})" if row.get("correction_action") == "correct" else value


def page_note(rows_by_path: list[tuple[Path, list[dict[str, str]]]]) -> list[str]:
    """Markdown lines naming every corrected or withdrawn cell of the rendered rows."""
    from format_cpu_ops_results import normalize_backend

    lines = []
    for _path, rows in rows_by_path:
        for row in rows:
            action = row.get("correction_action")
            if not action:
                continue
            label = WITHDRAWN_STATUS if action == "withdraw" else CORRECTED_MARK
            lines.append(
                f"- {label}: `{row.get('benchmark')}` {row.get('threads')}T `{row.get('shape')}` "
                f"`{normalize_backend(row)}` ({row['correction_ref']}): {row['reason']}."
            )
    if not lines:
        return []
    return [
        "",
        "## Corrected and withdrawn cells",
        "",
        "These cells differ from their run record, which is unchanged (`benchmarks/cell-corrections.yaml`).",
        "A withdrawn cell prints no number. A corrected cell prints its re-measurement.",
        "",
        *lines,
    ]
