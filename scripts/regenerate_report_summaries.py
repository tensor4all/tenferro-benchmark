#!/usr/bin/env python3
"""Regenerate machine-readable report summaries from local raw runs.

Maintained reports are tracked as ``result/<profile>/<suite>.md``; their
``.summary.json`` sidecars are normally written by the formatter that produced
the Markdown.  Raw runs live under ``data/results/...`` (gitignored), so this
script can rebuild the summaries for reports whose raw data is available
locally without re-running any benchmark.

Usage::

    uv run python scripts/regenerate_report_summaries.py
    uv run python scripts/regenerate_report_summaries.py --profile nvidia-gpu
    uv run python scripts/regenerate_report_summaries.py --suite gpu/dense --dry-run

Reports without local raw data are left untouched; the dashboard lists them
with a report link and no ratio/flag columns.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import report_summary  # noqa: E402

RUN_DIR_RE = re.compile(r"\d{8}_\d{6}")

# Maintained reports that can be regenerated from a plain timestamped run dir.
MAINTAINED: dict[tuple[str, str], str] = {
    ("nvidia-gpu", "gpu/dense"): "gpu_records",
    ("nvidia-gpu", "gpu/einsum"): "gpu_records",
    ("nvidia-gpu", "gpu/sparse"): "gpu_records",
    ("nvidia-gpu", "gpu/elementwise"): "gpu_records",
    ("nvidia-gpu", "gpu/linalg_jvp_vjp"): "gpu_records",
    ("nvidia-gpu", "gpu/linalg_ad_latency"): "gpu_records",
    ("nvidia-gpu", "gpu/permutation"): "gpu_permutation",
    ("nvidia-gpu", "gpu/perf_issues"): "perf_issues",
    ("mac-cpu", "cpu/permutation"): "cpu_permutation",
    ("amd-cpu", "cpu/permutation"): "cpu_permutation",
}


def latest_run_dir(suite_dir: Path) -> Path | None:
    if not suite_dir.is_dir():
        return None
    candidates = [
        child
        for child in suite_dir.iterdir()
        if child.is_dir() and RUN_DIR_RE.fullmatch(child.name)
    ]
    return max(candidates, key=lambda child: child.name) if candidates else None


def run_dir_from_report(report: Path, suite_dir: Path) -> Path | None:
    """Return the raw run directory a report names, when it names one.

    Some maintained reports (for example ``gpu/perf_issues``) embed the exact
    ``data/results/...`` path they were generated from; others carry the run
    timestamp in their header.  Prefer that over the newest timestamp so a
    regenerated summary always matches the tracked report even when a newer run
    was not published.  ``None`` means the report does not name a run (the
    caller may fall back to the newest local run); a missing declared run
    raises ``FileNotFoundError``.
    """
    if not report.exists():
        return None
    text = report.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"data/results/[^\s`)]+")
    for match in pattern.findall(text):
        candidate = Path(match.rstrip(".,;"))
        if candidate.is_dir() and candidate.parent.resolve() == suite_dir.resolve():
            return candidate
    stamp = re.search(r"^-\s*Timestamp:\s*`([^`]+)`", text, re.MULTILINE)
    if stamp:
        try:
            parsed = datetime.fromisoformat(stamp.group(1))
        except ValueError:
            return None
        name = parsed.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
        candidate = suite_dir / name
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"{report} names run {candidate}, which is not available locally")
    return None


def _load_gpu_records(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    import json

    records: list[dict[str, Any]] = []
    records_path = run_dir / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(records_path)
    for line in records_path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    metadata = report_summary.load_run_metadata(run_dir / "run.yaml")
    return records, metadata


def _summary_gpu_records(run_dir: Path) -> dict[str, Any]:
    from format_gpu_results import build_summary

    records, metadata = _load_gpu_records(run_dir)
    suite_id = (metadata or {}).get("suite_id", "")
    if suite_id == "gpu/linalg_jvp_vjp" or suite_id == "gpu/linalg_ad_latency":
        from format_gpu_linalg_ad_results import build_summary as build_linalg

        return build_linalg(records, metadata)
    return build_summary(records, metadata)


def _summary_gpu_permutation(run_dir: Path) -> dict[str, Any]:
    import json

    from format_gpu_permutation_results import build_summary, load_records

    paths = [run_dir / "rust_output.jsonl", *sorted(run_dir.glob("python_output_*.jsonl"))]
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            records.extend(load_records(path))
    if not records:
        raise FileNotFoundError(run_dir)
    return build_summary(records, report_summary.load_run_metadata(run_dir / "run.yaml"))


def _summary_cpu_permutation(run_dir: Path) -> dict[str, Any]:
    from format_permutation_results import build_summary

    paths = sorted(run_dir.glob("*_output_t*.jsonl"))
    if not paths:
        raise FileNotFoundError(run_dir)
    records: list[dict[str, Any]] = []
    import json

    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    metadata = report_summary.load_run_metadata(run_dir / "run.yaml")
    if metadata is None:
        metadata = report_summary.load_run_metadata(run_dir / "run_t1.yaml")
    return build_summary(records, metadata)


def _summary_perf_issues(run_dir: Path) -> dict[str, Any]:
    import benchmark_perf_issues

    metadata = None
    for candidate in (run_dir / "run.yaml", run_dir / "run_t1.yaml"):
        metadata = report_summary.load_run_metadata(candidate)
        if metadata:
            break
    target = (metadata or {}).get("target_profile", "nvidia-gpu")
    suite_key = "gpu" if str((metadata or {}).get("suite_id", "")).startswith("gpu/") else "cpu"
    return benchmark_perf_issues.build_summary(run_dir, target, suite_key)


BUILDERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "gpu_records": _summary_gpu_records,
    "gpu_permutation": _summary_gpu_permutation,
    "cpu_permutation": _summary_cpu_permutation,
    "perf_issues": _summary_perf_issues,
}


def regenerate(
    result_dir: Path,
    data_dir: Path,
    profiles: set[str] | None = None,
    suites: set[str] | None = None,
    dry_run: bool = False,
) -> list[Path]:
    written: list[Path] = []
    for (profile, suite_id), kind in sorted(MAINTAINED.items()):
        if profiles and profile not in profiles:
            continue
        if suites and suite_id not in suites:
            continue
        report = result_dir / profile / f"{suite_id}.md"
        suite_dir = data_dir / profile / suite_id
        try:
            run_dir = run_dir_from_report(report, suite_dir)
        except FileNotFoundError as error:
            print(f"skip {profile} {suite_id}: {error}", file=sys.stderr)
            continue
        if run_dir is None:
            run_dir = latest_run_dir(suite_dir)
        if run_dir is None:
            print(f"skip {profile} {suite_id}: no local raw run under {suite_dir}", file=sys.stderr)
            continue
        try:
            document = BUILDERS[kind](run_dir)
        except (FileNotFoundError, ValueError) as error:
            print(f"skip {profile} {suite_id}: {error}", file=sys.stderr)
            continue
        out = report_summary.summary_path_for_report(report)
        if dry_run:
            print(f"would write {out} from {run_dir}")
            continue
        report_summary.write_document(out, document)
        print(f"wrote {out} from {run_dir} ({len(document['rows'])} rows)")
        written.append(out)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--profile", action="append", dest="profiles", default=None)
    parser.add_argument("--suite", action="append", dest="suites", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    written = regenerate(
        args.result_dir,
        args.data_dir,
        set(args.profiles) if args.profiles else None,
        set(args.suites) if args.suites else None,
        args.dry_run,
    )
    print(f"{'would write' if args.dry_run else 'wrote'} {len(written)} summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
