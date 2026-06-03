#!/usr/bin/env python3
"""Helpers for benchmark suite result paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

SUITE_ID_RE = re.compile(r"^(cpu|gpu)/([a-z0-9][a-z0-9_.-]*)$")


@dataclass(frozen=True)
class SuiteRunPaths:
    run_dir: Path
    run_yaml: Path
    records_jsonl: Path
    report_md: Path
    latest_report: Path


def safe_suite_id_parts(suite_id: str) -> tuple[str, str]:
    match = SUITE_ID_RE.fullmatch(suite_id)
    if match is None:
        raise ValueError(f"invalid suite_id: {suite_id!r}")
    return match.group(1), match.group(2)


def paths_for_suite_run(project_dir: Path, suite_id: str, timestamp: str) -> SuiteRunPaths:
    device, suite_name = safe_suite_id_parts(suite_id)
    root = Path(project_dir)
    run_dir = root / "data" / "results" / device / suite_name / timestamp
    return SuiteRunPaths(
        run_dir=run_dir,
        run_yaml=run_dir / "run.yaml",
        records_jsonl=run_dir / "records.jsonl",
        report_md=run_dir / "report.md",
        latest_report=root / "result" / device / f"{suite_name}.md",
    )
