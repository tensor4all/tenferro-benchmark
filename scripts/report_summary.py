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

Schema (``summary-schema-v2``, written by offline run rendering) adds the
run-level provenance v1 cannot express::

    "schema_version": 2,
    "source_runs": [
      {"run": "data/results/<profile>/<suite>/<timestamp>",
       "threads": [1],
       "host": {...} | null,
       "toolchain": {...} | null,
       "power": "..." | "unrecorded"}
    ],
    "timing_policy": {...} | null

Rows are sorted by ``(case, group, backend)`` so a summary is deterministic for
the same input records.  ``median``/``iqr`` are ``null`` when a record is
missing or not ``ok``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA_VERSION = 1
SCHEMA_VERSION_V2 = 2
SUMMARY_SUFFIX = ".summary.json"
REPO_ROOT = Path(__file__).resolve().parents[1]

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
    source_runs: list[dict[str, Any]] | None = None,
    timing_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (str(row.get("case", "")), str(row.get("group", "")), str(row.get("backend", ""))),
    )
    document = {
        "schema_version": schema_version,
        "target_profile": target_profile,
        "suite_id": suite_id,
        "timestamp": timestamp,
        "tenferro_commit": tenferro_commit,
        "hardware": hardware or {},
        "unit": unit,
        "rows": ordered,
    }
    if source_runs is not None or timing_policy is not None:
        document["schema_version"] = SCHEMA_VERSION_V2
        document["source_runs"] = ordered_source_runs(source_runs or [])
        document["timing_policy"] = timing_policy
    return document


def recorded_threads(metadata: dict[str, Any] | None) -> list[int]:
    """Thread count a run record was collected with, when it recorded one."""
    environment = (metadata or {}).get("environment") or {}
    value = (environment.get("env") or {}).get("RAYON_NUM_THREADS")
    try:
        return [int(value)] if value is not None and int(value) > 0 else []
    except (TypeError, ValueError):
        return []


def power_label(metadata: dict[str, Any] | None) -> str:
    """Recorded power/frequency controls, never the current host state.

    The string is the toolkit's power-condition serialization, so a summary and
    the trend points drawn from it group measurements under the same condition:
    boost, turbo, governor, energy preference, frequency cap, amd-pstate mode,
    platform profile and SMT.
    """
    from campaign_toolkit import trend

    return trend.power_of(metadata or {})


def cpu_model_from_run_dir(run_dir: Path | None) -> str | None:
    """CPU model recorded in a run's own cpu_info.md snapshot, if present."""
    if run_dir is None:
        return None
    path = Path(run_dir) / "cpu_info.md"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^- Model:\s*`?([^`]+?)`?\s*$", line)
        if match:
            return match.group(1).strip()
    return None


def repo_relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def source_run(
    run_dir: Path,
    metadata: dict[str, Any] | None,
    *,
    threads: Iterable[int] = (),
) -> dict[str, Any]:
    """One recorded source run and the conditions it was collected under."""
    metadata = metadata or {}
    recorded = sorted({int(value) for value in threads} | set(recorded_threads(metadata)))
    return {
        "run": repo_relative(run_dir),
        "threads": recorded,
        "host": metadata.get("host"),
        "toolchain": ((metadata.get("build") or {}).get("toolchain")),
        "power": power_label(metadata),
    }


def ordered_source_runs(source_runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for entry in source_runs:
        key = (str(entry.get("run")), tuple(int(value) for value in entry.get("threads") or []))
        unique.setdefault(key, entry)
    return [unique[key] for key in sorted(unique)]


def source_runs_from_run_dir(run_dir: Path, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Source-run records from a run directory's per-thread metadata files."""
    run_dir = Path(run_dir)
    if metadata is None:
        run_yaml = run_dir / "run.yaml"
        metadata = load_run_metadata(run_yaml) if run_yaml.is_file() else {}
    import yaml

    thread_files = sorted(run_dir.glob("run_t*.yaml"))
    if not thread_files:
        return [source_run(run_dir, metadata)]
    entries = []
    for path in thread_files:
        with path.open() as fh:
            thread_metadata = yaml.safe_load(fh)
        if not isinstance(thread_metadata, dict):
            thread_metadata = metadata
        entries.append(source_run(run_dir, thread_metadata, threads=recorded_threads(thread_metadata)))
    return ordered_source_runs(entries)


def v2_provenance(
    run_dir: Path | None,
    run_metadata: dict[str, Any] | None,
    *,
    source_runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Keyword arguments that turn a summary document into schema version 2."""
    if run_dir is not None:
        runs = source_runs if source_runs is not None else source_runs_from_run_dir(run_dir, run_metadata)
    else:
        runs = source_runs or []
    return {"source_runs": runs, "timing_policy": (run_metadata or {}).get("timing_policy")}


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
    host = run_metadata.get("host")
    if isinstance(host, dict) and host.get("cpu"):
        hardware["cpu"] = str(host["cpu"])
    environment = run_metadata.get("environment")
    if isinstance(environment, dict):
        if environment.get("hostname"):
            hardware.setdefault("host", str(environment["hostname"]))
        if environment.get("cpu"):
            hardware.setdefault("cpu", str(environment["cpu"]))
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
    list.  ``hardware.threads`` collects every thread count seen, and
    ``source_runs`` unions every source with its own recorded conditions.
    Documents that mix measured revisions or timing policies are rejected:
    their rows would otherwise be read under one source's conditions.
    """
    if not documents:
        raise ValueError("no documents to merge")
    commits = {document.get("tenferro_commit") for document in documents}
    if len(commits) != 1:
        raise ValueError("cannot merge summaries with different tenferro_commit")
    policies = {json.dumps(document.get("timing_policy"), sort_keys=True) for document in documents}
    if len(policies) != 1:
        raise ValueError("cannot merge summaries with different timing_policy")
    base = dict(documents[0])
    rows: list[dict[str, Any]] = []
    threads: list[int] = []
    source_runs: list[dict[str, Any]] = []
    timestamps = [doc.get("timestamp") for doc in documents if doc.get("timestamp")]
    base["timestamp"] = max(timestamps) if timestamps else base.get("timestamp")
    for document in documents:
        hardware = document.get("hardware") or {}
        for thread in hardware.get("threads") or []:
            if thread not in threads:
                threads.append(thread)
        source_runs.extend(document.get("source_runs") or [])
        rows.extend(document.get("rows", []))
    hardware = dict(base.get("hardware") or {})
    if threads:
        hardware["threads"] = sorted(threads)
    base["hardware"] = hardware
    if any("source_runs" in document for document in documents):
        base["source_runs"] = ordered_source_runs(source_runs)
        base["schema_version"] = SCHEMA_VERSION_V2
        base["timing_policy"] = documents[0].get("timing_policy")
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
