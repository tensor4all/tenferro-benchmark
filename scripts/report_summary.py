#!/usr/bin/env python3
"""Machine-readable summary sidecars for maintained benchmark reports.

Every maintained `result/<profile>/<suite>.md` report gets a sibling
`result/<profile>/<suite>.summary.json` written by the formatter that produces
the Markdown, from the same records.  The dashboard consumes these summaries
instead of parsing Markdown.

Schema (``summary-schema-v1``)::

    {
      "schema_version": 1,
      "target_profile": "nvidia-gpu",
      "suite_id": "gpu/dense",
      "timestamp": "2026-10-16T23:51:16.689453+00:00",
      "tenferro_commit": "2754c8c8...",
      "hardware": {"gpu": "NVIDIA A100 80GB PCIe"},
      "unit": "ms",
      "rows": [
        {"case": "dense_matmul_f64_3072", "group": "matmul",
         "backend": "tenferro-cuda-eager", "median": 1.234,
         "iqr": 0.01, "status": "ok", "verified": true}
      ]
    }

Rows are sorted by ``(case, group, backend)`` so a summary is deterministic for
the same input records.  ``median``/``iqr`` are ``null`` when a record is
missing or not ``ok``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA_VERSION = 1
SUMMARY_SUFFIX = ".summary.json"

# Environment variable the CPU run scripts use to ask a formatter to write its
# per-run summary.  Using an environment variable keeps the existing
# stdout-piped formatter CLI (and its fixture tests) unchanged.
SUMMARY_ENV = "BENCHMARK_REPORT_SUMMARY"
RUN_METADATA_ENV = "BENCHMARK_RUN_METADATA"


def summary_path_for_report(report_path: Path) -> Path:
    """Return the summary path next to a report path (`x.md` -> `x.summary.json`)."""
    report_path = Path(report_path)
    stem = report_path.stem if report_path.name.endswith(".md") else report_path.name
    return report_path.parent / (stem + SUMMARY_SUFFIX)


def make_row(
    case: str,
    backend: str,
    median: float | None,
    *,
    status: str = "ok",
    iqr: float | None = None,
    group: str | None = None,
    verified: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"case": str(case)}
    if group not in (None, ""):
        row["group"] = str(group)
    row["backend"] = str(backend)
    row["median"] = float(median) if median is not None else None
    if iqr is not None:
        row["iqr"] = float(iqr)
    row["status"] = str(status)
    row["verified"] = bool(verified)
    if reason:
        row["reason"] = str(reason)
    return row


def build_document(
    rows: Iterable[dict[str, Any]],
    *,
    target_profile: str | None,
    suite_id: str | None,
    timestamp: str | None,
    tenferro_commit: str | None,
    hardware: dict[str, Any] | None,
    unit: str,
    schema_version: int = SCHEMA_VERSION,
) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (str(row.get("case", "")), str(row.get("group", "")), str(row.get("backend", ""))),
    )
    return {
        "schema_version": schema_version,
        "target_profile": target_profile,
        "suite_id": suite_id,
        "timestamp": timestamp,
        "tenferro_commit": tenferro_commit,
        "hardware": hardware or {},
        "unit": unit,
        "rows": ordered,
    }


def write_document(path: Path, document: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def metadata_from_run_metadata(run_metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Extract the summary header fields shared by all formatters."""
    run_metadata = run_metadata or {}
    tenferro = run_metadata.get("tenferro_rs")
    commit = tenferro.get("commit") if isinstance(tenferro, dict) else None
    hardware: dict[str, Any] = {}
    cuda = run_metadata.get("cuda")
    if isinstance(cuda, dict) and cuda.get("device_name"):
        hardware["gpu"] = str(cuda["device_name"])
    environment = run_metadata.get("environment")
    if isinstance(environment, dict):
        if environment.get("hostname"):
            hardware.setdefault("host", str(environment["hostname"]))
    return {
        "target_profile": run_metadata.get("target_profile"),
        "suite_id": run_metadata.get("suite_id"),
        "timestamp": run_metadata.get("timestamp"),
        "tenferro_commit": commit,
        "hardware": hardware,
    }


def load_run_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    import yaml

    with Path(path).open() as fh:
        metadata = yaml.safe_load(fh)
    return metadata if isinstance(metadata, dict) else None


def rows_from_gpu_records(
    records: Iterable[dict[str, Any]],
    *,
    group: Callable[[dict[str, Any]], str | None] | None = None,
) -> list[dict[str, Any]]:
    """Build summary rows from the shared GPU result-record schema.

    ``records`` are the JSONL records used by ``format_gpu_results.py`` and
    ``format_gpu_linalg_ad_results.py``: ``problem_id``, ``backend``,
    ``status``, ``timing.median_ms``/``timing.iqr_ms`` and
    ``verification.status``.
    """
    rows = []
    for record in records:
        status = str(record.get("status", "unknown"))
        timing = record.get("timing") or {}
        verification = record.get("verification") or {}
        execution = record.get("execution") or {}
        median = timing.get("median_ms")
        iqr = timing.get("iqr_ms")
        if status != "ok":
            median = None
            iqr = None
        reason = None
        if status == "unsupported":
            reason = execution.get("unsupported_reason")
            path = execution.get("execution_path")
            if not reason and path and "unavailable" in str(path):
                reason = str(path)
        case = record.get("problem_id") or record.get("case") or record.get("case_id")
        grp = group(record) if group is not None else record.get("op")
        rows.append(
            make_row(
                case,
                record["backend"],
                median,
                status=status,
                iqr=iqr,
                group=grp,
                verified=verification.get("status") == "passed",
                reason=reason,
            )
        )
    return rows


def maybe_write(
    document: dict[str, Any] | None,
    *,
    summary_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> Path | None:
    """Write a summary document when an explicit path or env var is available."""
    if document is None:
        return None
    env = env if env is not None else os.environ
    path = summary_path
    if path is None and env.get(SUMMARY_ENV):
        path = Path(env[SUMMARY_ENV])
    if path is None:
        return None
    return write_document(Path(path), document)


def env_run_metadata(env: dict[str, str] | None = None) -> dict[str, Any] | None:
    env = env if env is not None else os.environ
    path = env.get(RUN_METADATA_ENV)
    return load_run_metadata(Path(path)) if path else None


def merge_thread_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-thread summaries of one suite into a single document.

    The per-thread formatters already record their thread count in each row's
    ``group``, so merging is a concatenation plus a sorted, de-duplicated row
    list.  ``hardware.threads`` collects every thread count seen.
    """
    if not documents:
        raise ValueError("no documents to merge")
    base = dict(documents[0])
    rows: list[dict[str, Any]] = []
    threads: list[int] = []
    timestamps = [doc.get("timestamp") for doc in documents if doc.get("timestamp")]
    base["timestamp"] = max(timestamps) if timestamps else base.get("timestamp")
    for document in documents:
        hardware = document.get("hardware") or {}
        for thread in hardware.get("threads") or []:
            if thread not in threads:
                threads.append(thread)
        rows.extend(document.get("rows", []))
    hardware = dict(base.get("hardware") or {})
    if threads:
        hardware["threads"] = sorted(threads)
    base["hardware"] = hardware
    seen: set[tuple[str, str, str]] = set()
    unique_rows = []
    for row in sorted(
        rows,
        key=lambda row: (str(row.get("case", "")), str(row.get("group", "")), str(row.get("backend", ""))),
    ):
        key = (str(row.get("case")), str(row.get("group", "")), str(row.get("backend")))
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    base["rows"] = unique_rows
    return base
