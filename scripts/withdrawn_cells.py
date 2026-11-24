#!/usr/bin/env python3
"""Cells withdrawn from a recorded run (``benchmarks/withdrawn-cells.yaml``).

A withdrawn cell keeps its run record untouched; the page generators read the
CSV through :func:`read_rows`, which replaces a listed row's median and IQR
with the ``withdrawn`` status and the recorded reason. A page therefore says
the cell was measured and withdrawn, instead of printing a number that does not
measure what its column says, and ``regenerate_report_summaries.py --check``
re-derives the same page from the same run and this registry.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import yaml

from result_status import WITHDRAWN_STATUS

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmarks" / "withdrawn-cells.yaml"
KEY_FIELDS = ("benchmark", "dtype", "threads", "shape", "backend")


@lru_cache(maxsize=None)
def _entries(path: Path = REGISTRY) -> dict[tuple[str, ...], dict[str, str]]:
    if not path.exists():
        return {}
    document = yaml.safe_load(path.read_text()) or {}
    entries: dict[tuple[str, ...], dict[str, str]] = {}
    for cell in document.get("cells") or []:
        missing = [field for field in ("run", "reason", "ref", "evidence", *KEY_FIELDS)
                   if cell.get(field) in (None, "")]
        if missing:
            raise ValueError(f"{path}: withdrawn cell {cell!r} lacks {', '.join(missing)}")
        if not (path.resolve().parents[1] / str(cell["evidence"])).is_dir():
            raise ValueError(f"{path}: evidence {cell['evidence']} of withdrawn cell is not a directory")
        key = (str(cell["run"]).strip("/"), *(str(cell[field]) for field in KEY_FIELDS))
        if key in entries:
            raise ValueError(f"{path}: withdrawn cell listed twice: {key}")
        entries[key] = {"reason": " ".join(str(cell["reason"]).split()), "ref": str(cell["ref"])}
    return entries


def run_key(csv_path: Path) -> str:
    """``<profile>/<family>/<suite>/<timestamp>`` of the run directory holding ``csv_path``."""
    return "/".join(Path(csv_path).resolve().parent.parts[-4:])


def withdrawal(csv_path: Path, row: dict[str, str]) -> dict[str, str] | None:
    """The registry entry for one CSV row, if that cell is withdrawn."""
    from format_cpu_ops_results import normalize_backend

    values = {field: row.get(field, "") for field in KEY_FIELDS}
    values["backend"] = normalize_backend(row)
    if values["benchmark"] == "" and row.get("op"):
        phase = row.get("phase", "")
        values["benchmark"] = row["op"] + (f"_{phase}" if phase and phase != "primal" else "")
    return _entries().get((run_key(csv_path), *(values[field] for field in KEY_FIELDS)))


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """The CSV rows of a run, with withdrawn cells carrying no number."""
    rows = []
    with Path(csv_path).open(newline="") as f:
        for row in csv.DictReader(f):
            entry = withdrawal(csv_path, row)
            if entry is not None:
                row = {**row, "median_ms": "", "iqr_ms": "", "mean_us": "",
                       "status": WITHDRAWN_STATUS, "reason": entry["reason"], "withdrawn_ref": entry["ref"]}
            rows.append(row)
    return rows


def page_note(rows_by_path: list[tuple[Path, list[dict[str, str]]]]) -> list[str]:
    """Markdown lines naming every withdrawn cell of the rendered rows, or nothing."""
    from format_cpu_ops_results import normalize_backend

    lines = []
    for _path, rows in rows_by_path:
        for row in rows:
            if row.get("status") != WITHDRAWN_STATUS:
                continue
            lines.append(
                f"- `{row.get('benchmark')}` {row.get('threads')}T `{row.get('shape')}` "
                f"`{normalize_backend(row)}` ({row['withdrawn_ref']}): {row['reason']}."
            )
    if not lines:
        return []
    return [
        "",
        "## Withdrawn cells",
        "",
        "These cells were measured, but their number does not measure what the column says,",
        "so it is not printed (`benchmarks/withdrawn-cells.yaml`; the run record is unchanged).",
        "",
        *lines,
    ]
