#!/usr/bin/env python3
"""Render report summaries and published pages from recorded run directories.

A run directory under ``data/results/<profile>/<suite>/<timestamp>`` is the
record: raw measurement files, ``run.yaml``/``run_t<N>.yaml``, and the host and
thread-environment snapshots collection wrote.  ``result/<profile>/cpu/<suite>.md``
and its ``.summary.json`` sidecar are derived from that record and are never
edited by hand.

Usage::

    uv run python scripts/regenerate_report_summaries.py --publish RUN_DIR...
    uv run python scripts/regenerate_report_summaries.py --check
    uv run python scripts/regenerate_report_summaries.py
    uv run python scripts/regenerate_report_summaries.py --profile nvidia-gpu

``--publish`` renders the selected runs with ``scripts/render_run.py`` into a
temporary directory and writes the pages from that rendering.  An einsum run
directory holds three reports, and its per-thread runs are combined with
``format_cpu_thread_reports.combine``; every other CPU suite publishes the one
run directory it was collected as.  Only explicitly selected runs publish: no
run directory is ever discovered from the clock or from ``result/``.

``--check`` re-renders every page of the ``maintained`` profiles in
``benchmarks/profiles.yaml`` from the source runs its summary records, and
compares page and summary bytes.  A maintained page without a summary, or whose
source run is not tracked by git, is an error.  Pages listed in
``benchmarks/unverified-pages.txt`` are printed as UNVERIFIED instead: their raw
records are not in git yet.

Without those flags the maintained GPU reports are regenerated from the newest
local raw run their Markdown names.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import report_summary  # noqa: E402

ROOT = SCRIPT_DIR.parent
RUN_DIR_RE = re.compile(r"\d{8}_\d{6}")
PROFILES_FILE = Path("benchmarks/profiles.yaml")
UNVERIFIED_FILE = Path("benchmarks/unverified-pages.txt")

# Pages an einsum run directory publishes. It holds the einsum report plus the
# CPU-ops and linalg JVP/VJP reports; ``format_cpu_thread_reports.combine``
# writes each of them as ``<page>.md`` for the selected thread runs.
EINSUM_PAGES = ("einsum", "cpu_ops", "linalg_jvp_vjp")

# Maintained GPU reports that can be regenerated from a plain timestamped run dir.
MAINTAINED: dict[tuple[str, str], str] = {
    ("nvidia-gpu", "gpu/dense"): "gpu_records",
    ("nvidia-gpu", "gpu/einsum"): "gpu_records",
    ("nvidia-gpu", "gpu/sparse"): "gpu_records",
    ("nvidia-gpu", "gpu/elementwise"): "gpu_records",
    ("nvidia-gpu", "gpu/linalg_jvp_vjp"): "gpu_records",
    ("nvidia-gpu", "gpu/linalg_ad_latency"): "gpu_records",
    ("nvidia-gpu", "gpu/permutation"): "gpu_permutation",
    ("nvidia-gpu", "gpu/perf_issues"): "perf_issues",
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
        if not RUN_DIR_RE.fullmatch(candidate.name):
            continue
        same_suite = (
            candidate.parent.name == suite_dir.name
            and candidate.parent.parent.name == suite_dir.parent.name
        )
        if not same_suite:
            continue
        declared = suite_dir / candidate.name
        if declared.is_dir():
            return declared
        raise FileNotFoundError(
            f"{report} names run {declared}, which is not available locally"
        )
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


# --------------------------------------------------------------------------
# Publication of a recorded run directory as a page
# --------------------------------------------------------------------------


class PublishError(RuntimeError):
    """The selected run directories cannot be rendered as a published page."""


@dataclass(frozen=True)
class Page:
    """One published page and the summary sidecar next to it."""

    profile: str
    name: str
    markdown: bytes
    summary: bytes


def maintained_profiles(root: Path = ROOT) -> list[str]:
    import yaml

    path = Path(root) / PROFILES_FILE
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [str(name) for name in document.get("maintained") or []]


def unverified_pages(root: Path = ROOT) -> dict[str, str]:
    """Pages whose raw records are not in git yet, with the stated reason."""
    path = Path(root) / UNVERIFIED_FILE
    if not path.is_file():
        return {}
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        page, _, reason = line.partition("#")
        entries[page.strip()] = reason.strip() or "unverified"
    return entries


def _repo_relative(path: Path, root: Path) -> Path:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        raise PublishError(f"{path}: is not inside the repository root {root}") from None


def _render_runs(run_dirs: list[Path], work: Path, root: Path) -> list[Path]:
    """Render every run into ``work``, under its repository-relative path.

    The mirrored layout keeps the repository-relative source links of the
    combined CPU reports pointing at the recorded run directories.
    """
    import render_run

    rendered: list[Path] = []
    for run_dir in run_dirs:
        out = work / _repo_relative(run_dir, root)
        try:
            render_run.render(run_dir, out)
        except render_run.RenderError as error:
            raise PublishError(str(error)) from None
        metadata = run_dir / "run.yaml"
        if metadata.is_file():
            shutil.copyfile(metadata, out / "run.yaml")
        rendered.append(out)
    return rendered


def _suite_of(run_dir: Path) -> str:
    import render_run

    try:
        return render_run.detect_suite(run_dir, render_run.load_metadata(run_dir)[0])
    except render_run.RenderError as error:
        raise PublishError(str(error)) from None


def _page(directory: Path, name: str, report: str, profile: str) -> Page | None:
    markdown = directory / report
    if not markdown.is_file():
        return None
    summary = directory / f"{name}.summary.json"
    if not summary.is_file():
        raise PublishError(f"{markdown}: rendered without {summary.name}")
    return Page(profile, name, markdown.read_bytes(), summary.read_bytes())


def render_pages(run_dirs: Iterable[Path], work: Path, root: Path = ROOT) -> dict[str, Page]:
    """Render the selected runs in ``work`` and return every page they publish.

    Publication always renders the record; it never copies a report out of a
    run directory, so a stale or hand-edited run report cannot reach a page.
    Pages a run produces no input for (CPU ops skipped, for example) are
    omitted instead of failing.
    """
    import render_run

    selected: list[Path] = []
    for candidate in run_dirs:
        run_dir = Path(candidate).resolve()
        if run_dir in selected:
            raise PublishError(f"{run_dir}: selected twice")
        if not run_dir.is_dir():
            raise PublishError(f"{run_dir}: not a run directory")
        selected.append(run_dir)
    if not selected:
        raise PublishError("no run directories selected")

    suites = {_suite_of(run_dir) for run_dir in selected}
    if len(suites) != 1:
        raise PublishError(
            "select runs of one suite; got " + ", ".join(sorted(suites))
        )
    suite = suites.pop()
    metadata, _ = render_run.load_metadata(selected[0])
    profile = str(metadata.get("target_profile") or "")
    if not profile:
        raise PublishError(f"{selected[0]}: run metadata records no target_profile")

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    rendered = _render_runs(selected, work, root)
    pages: dict[str, Page] = {}
    if suite == "cpu/einsum":
        import format_cpu_thread_reports

        format_cpu_thread_reports.combine(rendered, work)
        destination = work / "result" / profile / "cpu"
        for name in EINSUM_PAGES:
            page = _page(destination, name, f"{name}.md", profile)
            if page is not None:
                pages[name] = page
        return pages

    if len(rendered) != 1:
        raise PublishError(f"{suite}: publishes exactly one run directory")
    name = str(metadata.get("suite_id") or suite).split("/")[-1]
    page = _page(rendered[0], name, "report.md", profile)
    if page is None:
        raise PublishError(f"{selected[0]}: rendered no {name} page")
    pages[name] = page
    return pages


def publish(
    run_dirs: Iterable[Path], result_dir: Path = ROOT / "result", root: Path = ROOT
) -> list[Path]:
    """Write the pages derived from ``run_dirs`` and return the written paths."""
    written: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="publish-run-") as work:
        pages = render_pages(run_dirs, Path(work), root)
        for page in pages.values():
            destination = Path(result_dir) / page.profile / "cpu"
            destination.mkdir(parents=True, exist_ok=True)
            markdown = destination / f"{page.name}.md"
            markdown.write_bytes(page.markdown)
            summary = report_summary.summary_path_for_report(markdown)
            summary.write_bytes(page.summary)
            written += [markdown, summary]
    return written


def _tracked_sources(run: str, root: Path) -> bool:
    """Whether a run directory has at least one file in the git index."""
    if not (Path(root) / run).is_dir():
        return False
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", run],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PublishError(
            f"{root}: git ls-files failed: {completed.stderr.decode().strip()}"
        )
    return bool(completed.stdout)


def check(result_dir: Path = ROOT / "result", root: Path = ROOT) -> int:
    """Re-render every maintained page from its recorded sources and diff bytes."""
    result_dir = Path(result_dir)
    root = Path(root)
    unverified = unverified_pages(root)
    problems: list[str] = []
    checked = 0
    for profile in maintained_profiles(root):
        cpu_dir = result_dir / profile / "cpu"
        if not cpu_dir.is_dir():
            continue
        for markdown in sorted(cpu_dir.glob("*.md")):
            relative = _display(markdown, root, result_dir)
            if relative in unverified:
                print(f"UNVERIFIED {relative}: {unverified[relative]}")
                continue
            checked += 1
            summary = report_summary.summary_path_for_report(markdown)
            if not summary.is_file():
                problems.append(f"{relative}: no {summary.name}")
                continue
            try:
                document = json.loads(summary.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                problems.append(f"{relative}: {summary.name} is not JSON ({error})")
                continue
            sources = list(
                dict.fromkeys(
                    str(entry["run"])
                    for entry in document.get("source_runs") or []
                    if isinstance(entry, dict) and entry.get("run")
                )
            )
            if not sources:
                problems.append(f"{relative}: {summary.name} records no source_runs")
                continue
            untracked = [run for run in sources if not _tracked_sources(run, root)]
            if untracked:
                problems += [
                    f"{relative}: source run `{run}` is not tracked by git"
                    for run in untracked
                ]
                continue
            with tempfile.TemporaryDirectory(prefix="check-run-") as work:
                try:
                    pages = render_pages([root / run for run in sources], Path(work), root)
                except PublishError as error:
                    problems.append(f"{relative}: {error}")
                    continue
                page = pages.get(markdown.stem)
                if page is None:
                    problems.append(
                        f"{relative}: its source runs render no `{markdown.stem}` page"
                    )
                    continue
            if page.markdown != markdown.read_bytes():
                problems.append(
                    f"{relative}: differs from the runs its summary records"
                )
            if page.summary != summary.read_bytes():
                problems.append(
                    f"{_display(summary, root, result_dir)}: differs from the runs its summary records"
                )
    for problem in problems:
        print(problem, file=sys.stderr)
    print(
        f"checked {checked} maintained pages from their recorded runs: "
        f"{'FAILED' if problems else 'ok'}"
    )
    return 1 if problems else 0


def _display(path: Path, root: Path, result_dir: Path) -> str:
    """Repository-relative spelling of a page, ``result/...`` by default."""
    for base in (root, result_dir.parent):
        try:
            return str(Path(path).resolve().relative_to(Path(base).resolve()))
        except ValueError:
            continue
    return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/results")
    parser.add_argument("--profile", action="append", dest="profiles", default=None)
    parser.add_argument("--suite", action="append", dest="suites", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--publish",
        nargs="+",
        type=Path,
        metavar="RUN_DIR",
        help="publish the pages of these recorded run directories",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-render every maintained page from its sources and diff bytes",
    )
    args = parser.parse_args(argv)
    if args.publish:
        try:
            written = publish(args.publish, args.result_dir)
        except PublishError as error:
            print(f"publish: {error}", file=sys.stderr)
            return 1
        for path in written:
            print(f"wrote {path}")
        return 0
    if args.check:
        return check(args.result_dir)
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
