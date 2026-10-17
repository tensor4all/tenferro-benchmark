#!/usr/bin/env python3
"""Generate the static benchmark results dashboard for GitHub Pages.

The dashboard is plain HTML + CSS + inline SVG; there is no JavaScript, no CDN,
and the only rendering dependency is the existing ``markdown`` package used for
the original report pages.  It reads the machine-readable
``result/<profile>/<suite>.summary.json`` sidecars written by the formatters and
falls back to a report-only row for maintained reports without a summary.

Landing page::

    profile section -> header (hardware / date / tenferro commit)
                     -> suite table (tenferro vs PyTorch / JAX, flags, report)

Suite page::

    metadata -> flagged cases -> per-case ratio table -> SVG bar charts
             -> full original Markdown report

Diagnostics (every non-maintained report) are collapsed under one ``<details>``.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import markdown

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import report_summary  # noqa: E402

# Maintained reports from AGENTS.md "Expected latest report paths" plus the
# three GPU reports it lists in the design.  Everything else is a diagnostic.
MAINTAINED: set[tuple[str, str]] = {
    ("mac-cpu", "cpu/einsum"),
    ("mac-cpu", "cpu/cpu_ops"),
    ("mac-cpu", "cpu/public_api"),
    ("mac-cpu", "cpu/linalg_jvp_vjp"),
    ("mac-cpu", "cpu/permutation"),
    ("amd-cpu", "cpu/einsum"),
    ("amd-cpu", "cpu/cpu_ops"),
    ("amd-cpu", "cpu/linalg_jvp_vjp"),
    ("amd-cpu", "cpu/permutation"),
    ("amd-cpu", "cpu/perf_issues"),
    ("linux-cpu", "cpu/linalg_jvp_jvp"),
    ("nvidia-gpu", "gpu/dense"),
    ("nvidia-gpu", "gpu/einsum"),
    ("nvidia-gpu", "gpu/elementwise"),
    ("nvidia-gpu", "gpu/sparse"),
    ("nvidia-gpu", "gpu/linalg_jvp_vjp"),
    ("nvidia-gpu", "gpu/linalg_ad_latency"),
    ("nvidia-gpu", "gpu/permutation"),
    ("nvidia-gpu", "gpu/perf_issues"),
}

PROFILE_ORDER = ["nvidia-gpu", "linux-cpu", "mac-cpu", "amd-cpu"]

# Profiles whose numbers come from GitHub-hosted runners and therefore carry
# the reference-quality caveat.  Dedicated A100 / devcontainer hosts do not.
GITHUB_HOSTED_PROFILES = {"mac-cpu", "linux-cpu"}

PYTORCH_BACKENDS = {"pytorch-cuda", "pytorch-cpu", "pytorch-mps"}
JAX_BACKENDS = {"jax-cuda", "jax-cpu", "jax-cuda-tree-jit", "jax-cuda-eager", "jax-metal"}
TENFERRO_EAGER = "tenferro-cuda-eager"
TENFERRO_TRACE = "tenferro-cuda-trace"

EAGER_TORCH_FLAG = 1.5
EAGER_JAX_FLAG = 3.0
TRACE_EAGER_FLAG = 1.3
REGRESSION_FLAG = 1.25

REPORT_QUALITY_NOTICE = (
    "<strong>Reference-quality indicators only.</strong> These numbers were "
    "collected on GitHub-hosted runners (shared, variable-load infrastructure), "
    "so they are useful for spot-checks but not for strict regression tracking. "
    "Canonical regression series require stable dedicated runners."
)


# ---------------------------------------------------------------------------
# Discovery and data loading
# ---------------------------------------------------------------------------

def discover_reports(result_dir: Path) -> list[tuple[str, str, Path]]:
    """Return (target_profile, suite_id, report_path) for every report."""
    reports: list[tuple[str, str, Path]] = []
    if not result_dir.exists():
        return reports
    for profile_dir in sorted(result_dir.iterdir()):
        if not profile_dir.is_dir():
            continue
        for suite_dir in sorted(profile_dir.iterdir()):
            if not suite_dir.is_dir():
                continue
            for report in sorted(suite_dir.glob("*.md")):
                reports.append((profile_dir.name, f"{suite_dir.name}/{report.stem}", report))
    return reports


def load_summaries(result_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    if not result_dir.exists():
        return summaries
    for path in sorted(result_dir.glob("*/*/*.summary.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        profile = path.parent.parent.name
        suite_id = f"{path.parent.name}/{path.name[: -len(report_summary.SUMMARY_SUFFIX)]}"
        summaries[(profile, suite_id)] = document
    return summaries


def case_rows(document: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group a summary document's rows by case, preserving backend order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in document.get("rows", []):
        grouped.setdefault(str(row.get("case")), []).append(row)
    return grouped


def row_median(rows: list[dict[str, Any]], backend: str) -> float | None:
    for row in rows:
        if row.get("backend") == backend and row.get("status") == "ok" and row.get("median") is not None:
            return float(row["median"])
    return None


def best_reference(rows: list[dict[str, Any]], backends: set[str]) -> float | None:
    values = [row_median(rows, backend) for backend in backends]
    values = [value for value in values if value is not None and value > 0]
    return min(values) if values else None


def tenferro_median(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """Return (eager, trace) tenferro medians for a case."""
    return row_median(rows, TENFERRO_EAGER), row_median(rows, TENFERRO_TRACE)


def primary_tenferro(rows: list[dict[str, Any]]) -> float | None:
    eager, trace = tenferro_median(rows)
    return eager if eager is not None else trace


# ---------------------------------------------------------------------------
# Ratios and flags
# ---------------------------------------------------------------------------

def geometric_mean(values: Iterable[float]) -> float | None:
    values = [value for value in values if value > 0]
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def suite_ratios(document: dict[str, Any]) -> dict[str, Any]:
    """Geometric-mean tenferro/reference ratios for a summary.

    Returns ``{"pytorch": {"eager": r, "trace": r, "count": n}, ...}``; a
    ratio key is missing when no comparable case exists.
    """
    result: dict[str, Any] = {}
    for label, references in (("pytorch", PYTORCH_BACKENDS), ("jax", JAX_BACKENDS)):
        entry: dict[str, Any] = {}
        for key, backend in (("eager", TENFERRO_EAGER), ("trace", TENFERRO_TRACE)):
            ratios = []
            for rows in case_rows(document).values():
                numerator = row_median(rows, backend)
                denominator = best_reference(rows, references)
                if numerator is not None and denominator is not None and numerator > 0:
                    ratios.append(numerator / denominator)
            if ratios:
                entry[key] = geometric_mean(ratios)
        if entry:
            entry["count"] = max(
                sum(
                    1
                    for rows in case_rows(document).values()
                    if any(
                        row_median(rows, backend) is not None
                        and best_reference(rows, references) is not None
                        for backend in (TENFERRO_EAGER, TENFERRO_TRACE)
                    )
                ),
                0,
            )
            result[label] = entry
    return result


def compute_case_flags(
    rows: list[dict[str, Any]],
    previous_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return the anomaly flags for one case.

    Flags: ``eager/torch``, ``eager/jax``, ``trace/eager``, ``status`` and
    ``regression``.  The regression check matches the same backend and needs a
    previous summary for the same hardware (passed as ``previous_rows``).
    """
    flags: list[str] = []
    eager, trace = tenferro_median(rows)
    pytorch = best_reference(rows, PYTORCH_BACKENDS)
    jax = best_reference(rows, JAX_BACKENDS)
    if eager is not None and pytorch is not None and eager / pytorch >= EAGER_TORCH_FLAG:
        flags.append("eager/torch")
    if eager is not None and jax is not None and eager / jax >= EAGER_JAX_FLAG:
        flags.append("eager/jax")
    if eager is not None and trace is not None and trace / eager >= TRACE_EAGER_FLAG:
        flags.append("trace/eager")
    if any(
        str(row.get("backend", "")).startswith("tenferro") and row.get("status") != "ok"
        for row in rows
    ):
        flags.append("status")
    if previous_rows:
        for row in rows:
            if row.get("status") != "ok" or row.get("median") is None:
                continue
            backend = row.get("backend")
            previous = next(
                (
                    prev
                    for prev in previous_rows
                    if prev.get("backend") == backend
                    and prev.get("status") == "ok"
                    and prev.get("median")
                ),
                None,
            )
            if previous and float(row["median"]) / float(previous["median"]) >= REGRESSION_FLAG:
                flags.append("regression")
                break
    return flags


def hardware_name(document: dict[str, Any] | None) -> str | None:
    if not document:
        return None
    hardware = document.get("hardware") or {}
    return hardware.get("gpu") or hardware.get("cpu") or hardware.get("host")


def previous_rows_for(
    profile: str,
    suite_id: str,
    document: dict[str, Any],
    baseline_dir: Path | None,
    git_repo: Path | None,
    git_result_dir: str,
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """Load the previous summary for regression comparison.

    With ``baseline_dir`` set (tests, local inspection) it reads the file from
    that tree.  Otherwise it walks ``git log`` for the summary path and returns
    the most recent earlier committed version whose hardware name matches.
    """
    suite_name = suite_id.split("/", 1)[1] if "/" in suite_id else suite_id
    hardware = hardware_name(document)
    if baseline_dir is not None:
        candidate = baseline_dir / profile / f"{suite_id}.summary.json"
        if candidate.exists():
            previous = json.loads(candidate.read_text(encoding="utf-8"))
            if hardware_name(previous) == hardware:
                return hardware, previous.get("rows")
        return None, None
    if git_repo is None:
        return None, None
    relative = f"{git_result_dir}/{profile}/{suite_id}.summary.json"
    try:
        commits = subprocess.run(
            ["git", "-C", str(git_repo), "log", "--format=%H", "--", relative],
            check=True, capture_output=True, text=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    current_text = json.dumps(document, indent=2) + "\n"
    for commit in commits:
        show = subprocess.run(
            ["git", "-C", str(git_repo), "show", f"{commit}:{relative}"],
            capture_output=True, text=True,
        )
        if show.returncode != 0:
            continue
        if show.stdout == current_text:
            continue
        try:
            previous = json.loads(show.stdout)
        except ValueError:
            continue
        if hardware_name(previous) == hardware:
            return hardware, previous.get("rows")
    return hardware, None


def suite_flags(
    document: dict[str, Any] | None,
    previous_rows: list[dict[str, Any]] | None,
) -> dict[str, list[str]]:
    if not document:
        return {}
    grouped = case_rows(document)
    previous_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in previous_rows or []:
        previous_by_case.setdefault(str(row.get("case")), []).append(row)
    flags: dict[str, list[str]] = {}
    for case, rows in grouped.items():
        case_flags = compute_case_flags(rows, previous_by_case.get(case))
        if case_flags:
            flags[case] = case_flags
    return flags


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def ratio_class(value: float | None) -> str:
    if value is None:
        return "ratio-none"
    if value <= 0.90:
        return "ratio-faster"
    if value <= 1.10:
        return "ratio-parity"
    if value <= 1.50:
        return "ratio-slower"
    return "ratio-much-slower"


def fmt_median(value: float | None, unit: str) -> str:
    if value is None:
        return "-"
    if value >= 1000:
        return f"{value:.1f} {unit}"
    if value >= 1:
        return f"{value:.3f} {unit}"
    return f"{value:.4g} {unit}"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_ratio_cell(entry: dict[str, Any] | None) -> str:
    if not entry:
        return '<td class="ratio-none">-</td>'
    parts = []
    for key in ("eager", "trace"):
        value = entry.get(key)
        if value is None:
            continue
        label = "eager" if key == "eager" else "trace"
        parts.append(
            f'<span class="ratio {ratio_class(value)}">{label} {fmt_ratio(value)}</span>'
        )
    if not parts:
        return '<td class="ratio-none">-</td>'
    count = entry.get("count")
    count_text = f'<span class="ratio-count">n={count}</span>' if count else ""
    return "<td>" + " ".join(parts) + count_text + "</td>"


def flag_badges(flags: list[str]) -> str:
    return " ".join(f'<span class="flag">{esc(flag)}</span>' for flag in flags)


# ---------------------------------------------------------------------------
# SVG bar chart
# ---------------------------------------------------------------------------

BAR_CHART_WIDTH = 720
BAR_LABEL_WIDTH = 190
BAR_AREA_WIDTH = 440
BAR_HEIGHT = 20
BAR_GAP = 6


def svg_bar_chart(rows: list[dict[str, Any]], unit: str) -> str:
    points = [
        (str(row.get("backend")), float(row["median"]))
        for row in rows
        if row.get("status") == "ok" and row.get("median") is not None and float(row["median"]) > 0
    ]
    if not points:
        return '<p class="muted">No successful measurements.</p>'
    points.sort(key=lambda item: item[1])
    values = [value for _, value in points]
    use_log = max(values) / min(values) > 10
    transformed = [math.log10(value) if use_log else value for value in values]
    low, high = min(transformed), max(transformed)
    span = high - low or 1.0

    height = len(points) * (BAR_HEIGHT + BAR_GAP) + 8
    parts = [
        f'<svg class="bars" viewBox="0 0 {BAR_CHART_WIDTH} {height}" '
        f'width="100%" role="img" aria-label="median by backend">'
    ]
    if use_log:
        parts.append('<text x="0" y="10" class="chart-note">log scale</text>')
    for index, ((backend, value), transformed_value) in enumerate(zip(points, transformed)):
        y = index * (BAR_HEIGHT + BAR_GAP) + (16 if use_log else 4)
        fraction = (transformed_value - low) / span if not use_log else (transformed_value - low) / span
        width = max(2.0, fraction * BAR_AREA_WIDTH)
        cls = "bar-tenferro" if backend.startswith("tenferro") else "bar-reference"
        parts.append(
            f'<text x="0" y="{y + 14}" class="chart-label">{esc(backend)}</text>'
        )
        parts.append(
            f'<rect x="{BAR_LABEL_WIDTH}" y="{y}" width="{width:.1f}" '
            f'height="{BAR_HEIGHT}" class="{cls}"></rect>'
        )
        parts.append(
            f'<text x="{BAR_LABEL_WIDTH + width + 6:.1f}" y="{y + 14}" '
            f'class="chart-value">{esc(fmt_median(value, unit))}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

STYLESHEET = """
:root {
  --bg: #ffffff; --fg: #1b1f24; --muted: #59636e; --line: #d0d7de;
  --card: #f6f8fa; --link: #0969da; --accent: #0969da;
  --faster: #1a7f37; --parity: #57606a; --slower: #9a6700; --much-slower: #cf222e;
  --bar-tenferro: #0969da; --bar-reference: #8c959f; --flag-bg: #fff1e5; --flag-fg: #953800;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #30363d;
    --card: #161b22; --link: #4493f8; --accent: #4493f8;
    --faster: #3fb950; --parity: #9198a1; --slower: #d29922; --much-slower: #f85149;
    --bar-tenferro: #4493f8; --bar-reference: #6e7681; --flag-bg: #3d1d00; --flag-fg: #ffa657;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0 auto; padding: 1.5rem 1rem 4rem; max-width: 1080px;
  background: var(--bg); color: var(--fg);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.5;
}
h1 { border-bottom: 2px solid var(--line); padding-bottom: .3rem; font-size: 1.6rem; }
h2 { margin-top: 2.2rem; font-size: 1.25rem; }
h3 { margin-top: 1.6rem; font-size: 1.05rem; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
code { background: var(--card); padding: .1rem .3rem; border-radius: 4px; font-size: .9em; }
.muted { color: var(--muted); }
.notice {
  background: var(--card); border-left: 4px solid var(--accent);
  padding: .7rem 1rem; margin: 1rem 0; border-radius: 0 6px 6px 0;
}
.profile-card { border: 1px solid var(--line); border-radius: 8px; padding: 1rem; margin: 1.2rem 0; }
.profile-head { display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; color: var(--muted); font-size: .9rem; }
.profile-head strong { color: var(--fg); }
table { border-collapse: collapse; width: 100%; margin: .8rem 0; font-size: .92rem; }
th, td { border: 1px solid var(--line); padding: .4rem .55rem; text-align: left; vertical-align: top; }
th { background: var(--card); position: sticky; top: 0; z-index: 1; }
thead th { position: sticky; top: 0; }
.table-wrap { overflow-x: auto; }
.ratio { display: inline-block; font-variant-numeric: tabular-nums; font-weight: 600; }
.ratio-faster { color: var(--faster); }
.ratio-parity { color: var(--parity); }
.ratio-slower { color: var(--slower); }
.ratio-much-slower { color: var(--much-slower); }
.ratio-count { color: var(--muted); font-size: .8rem; margin-left: .3rem; }
.flag {
  display: inline-block; background: var(--flag-bg); color: var(--flag-fg);
  border-radius: 999px; padding: 0 .5rem; font-size: .78rem; margin: 0 .15rem .15rem 0;
}
.bars { display: block; max-width: 100%; height: auto; margin: .4rem 0 1.2rem; }
.chart-label { font-size: 11px; fill: var(--fg); font-family: inherit; }
.chart-value { font-size: 11px; fill: var(--muted); font-family: inherit; }
.chart-note { font-size: 10px; fill: var(--muted); font-family: inherit; }
.bar-tenferro { fill: var(--bar-tenferro); }
.bar-reference { fill: var(--bar-reference); }
details { border: 1px solid var(--line); border-radius: 8px; padding: .6rem .9rem; margin: 1rem 0; }
summary { cursor: pointer; font-weight: 600; }
details.diag { background: var(--card); }
footer { margin-top: 3rem; font-size: .85rem; color: var(--muted); border-top: 1px solid var(--line); padding-top: 1rem; }
@media (max-width: 640px) {
  body { padding: 1rem .6rem 3rem; }
  th, td { padding: .35rem .4rem; font-size: .85rem; }
}
"""


def page(title: str, body: str, *, description: str = "") -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"  <title>{esc(title)}</title>\n"
        + (f'  <meta name="description" content="{esc(description)}">\n' if description else "")
        + f"  <style>{STYLESHEET}</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>\n"
    )


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

def ordered_profiles(profiles: Iterable[str]) -> list[str]:
    known = [profile for profile in PROFILE_ORDER if profile in profiles]
    known.extend(sorted(profile for profile in profiles if profile not in PROFILE_ORDER))
    return known


def profile_header(profile: str, summaries: dict[tuple[str, str], dict[str, Any]], maintained: list[tuple[str, str, Path]]) -> str:
    documents = [summaries[key] for key in summaries if key[0] == profile]
    hardware = None
    for document in documents:
        hardware = hardware or hardware_name(document)
    timestamp = max((str(doc.get("timestamp")) for doc in documents if doc.get("timestamp")), default=None)
    commit = None
    for document in documents:
        if document.get("tenferro_commit"):
            commit = document["tenferro_commit"]
            break
    if not documents:
        for _profile, _suite, report in maintained:
            if _profile == profile:
                mtime = datetime.fromtimestamp(report.stat().st_mtime, tz=timezone.utc)
                timestamp = mtime.isoformat(timespec="seconds")
                break
    pieces = [f"<strong>{esc(profile)}</strong>"]
    if hardware:
        pieces.append(f"hardware: {esc(hardware)}")
    if timestamp:
        pieces.append(f"date: {esc(timestamp[:19].replace('T', ' '))}Z")
    if commit:
        pieces.append(f"tenferro: <code>{esc(commit[:12])}</code>")
    if not documents:
        pieces.append('<span class="muted">no machine-readable summary</span>')
    return '<div class="profile-head">' + "".join(f"<span>{piece}</span>" for piece in pieces) + "</div>"


def landing_ratios(
    document: dict[str, Any] | None,
    previous_rows: list[dict[str, Any]] | None,
) -> tuple[str, str, str]:
    if document is None:
        return '<td class="ratio-none">-</td>', '<td class="ratio-none">-</td>', '<td class="muted">-</td>'
    ratios = suite_ratios(document)
    flags = suite_flags(document, previous_rows)
    flag_cells = '<span class="muted">none</span>'
    if flags:
        counts: dict[str, int] = {}
        for case_flags in flags.values():
            for flag in case_flags:
                counts[flag] = counts.get(flag, 0) + 1
        flag_cells = f'<strong>{len(flags)}</strong> case(s): ' + " ".join(
            f'<span class="flag">{esc(flag)}×{count}</span>' for flag, count in sorted(counts.items())
        )
    return render_ratio_cell(ratios.get("pytorch")), render_ratio_cell(ratios.get("jax")), f"<td>{flag_cells}</td>"


def render_landing(
    reports: list[tuple[str, str, Path]],
    summaries: dict[tuple[str, str], dict[str, Any]],
    previous_by_key: dict[tuple[str, str], list[dict[str, Any]] | None],
) -> str:
    profiles = ordered_profiles({profile for profile, _suite, _path in reports})
    all_diagnostics: dict[str, list[tuple[str, str, Path]]] = {}
    body = [
        "<h1>tenferro-benchmark results</h1>",
        '<p class="muted">Latest benchmark reports by target profile and suite. '
        "Ratios are geometric means of tenferro/reference medians over comparable "
        "cases (&lt;1 is faster). Flags mark per-case anomalies; open a suite for "
        "the flagged cases, ratios and charts.</p>",
    ]
    for profile in profiles:
        maintained = [
            (p, suite, report)
            for p, suite, report in reports
            if p == profile and (p, suite) in MAINTAINED
        ]
        diagnostics = [
            (p, suite, report)
            for p, suite, report in reports
            if p == profile and (p, suite) not in MAINTAINED
        ]
        if diagnostics:
            all_diagnostics[profile] = diagnostics
        if not maintained and not diagnostics:
            continue
        body.append('<section class="profile-card">')
        body.append(f"<h2>{esc(profile)}</h2>")
        body.append(profile_header(profile, summaries, maintained or diagnostics))
        if profile in GITHUB_HOSTED_PROFILES:
            body.append(f'<p class="notice">{REPORT_QUALITY_NOTICE}</p>')
        if maintained:
            body.append('<div class="table-wrap"><table>')
            body.append(
                "<thead><tr><th>suite</th><th>tenferro vs PyTorch</th>"
                "<th>tenferro vs JAX</th><th>flags</th><th>report</th></tr></thead>"
            )
            body.append("<tbody>")
            for _profile, suite_id, report in sorted(maintained):
                document = summaries.get((profile, suite_id))
                pytorch_cell, jax_cell, flags_cell = landing_ratios(
                    document, previous_by_key.get((profile, suite_id))
                )
                suite_dir = suite_id.split("/", 1)[0]
                href = f"{profile}/{suite_dir}/{Path(report).stem}.html"
                body.append(
                    f'<tr><td><code>{esc(suite_id)}</code></td>'
                    f"{pytorch_cell}{jax_cell}{flags_cell}"
                    f'<td><a href="{esc(href)}">report</a></td></tr>'
                )
            body.append("</tbody></table></div>")
        else:
            body.append('<p class="muted">No maintained reports.</p>')
        body.append("</section>")
    if all_diagnostics:
        body.append(render_diagnostics(all_diagnostics))
    body.append(
        '<footer>Generated by <code>scripts/generate_dashboard.py</code> from the '
        '<a href="https://github.com/tensor4all/tenferro-benchmark">tensor4all/tenferro-benchmark</a> '
        "repository. Summary schema: <code>summary-schema-v1</code>.</footer>"
    )
    return page("tenferro-benchmark results", "\n".join(body))


def render_diagnostics(diagnostics: dict[str, list[tuple[str, str, Path]]]) -> str:
    total = sum(len(entries) for entries in diagnostics.values())
    groups = []
    for profile in ordered_profiles(diagnostics):
        rows = []
        for _profile, suite_id, report in sorted(diagnostics[profile]):
            mtime = datetime.fromtimestamp(report.stat().st_mtime, tz=timezone.utc)
            suite_dir = suite_id.split("/", 1)[0]
            href = f"{profile}/{suite_dir}/{Path(report).stem}.html"
            rows.append(
                f'<li><a href="{esc(href)}"><code>{esc(suite_id)}</code></a> '
                f'<span class="muted">({esc(mtime.date().isoformat())})</span></li>'
            )
        groups.append(f"<h3>{esc(profile)}</h3><ul>" + "".join(rows) + "</ul>")
    return (
        f'<details class="diag"><summary>Diagnostic reports ({total})</summary>'
        + "".join(groups)
        + "</details>"
    )


# ---------------------------------------------------------------------------
# Suite page
# ---------------------------------------------------------------------------

def render_case_table(document: dict[str, Any], previous_rows: list[dict[str, Any]] | None) -> str:
    unit = document.get("unit", "ms")
    flags = suite_flags(document, previous_rows)
    grouped = case_rows(document)
    rows = []
    for case in sorted(grouped):
        rows_for_case = grouped[case]
        eager, trace = tenferro_median(rows_for_case)
        pytorch = best_reference(rows_for_case, PYTORCH_BACKENDS)
        jax = best_reference(rows_for_case, JAX_BACKENDS)
        eager_torch = eager / pytorch if eager is not None and pytorch else None
        eager_jax = eager / jax if eager is not None and jax else None
        trace_eager = trace / eager if trace is not None and eager else None
        statuses = sorted(
            {str(row.get("status")) for row in rows_for_case if str(row.get("backend", "")).startswith("tenferro")}
        )
        status_text = ", ".join(statuses) if statuses else "-"
        flags_text = flag_badges(flags.get(case, [])) or '<span class="muted">-</span>'
        rows.append(
            f"<tr><td><code>{esc(case)}</code></td>"
            f"<td>{esc(rows_for_case[0].get('group', '') or '-')}</td>"
            f'<td class="ratio {ratio_class(eager_torch)}">{fmt_ratio(eager_torch)}</td>'
            f'<td class="ratio {ratio_class(eager_jax)}">{fmt_ratio(eager_jax)}</td>'
            f'<td class="ratio {ratio_class(trace_eager)}">{fmt_ratio(trace_eager)}</td>'
            f"<td>{status_text}</td><td>{flags_text}</td></tr>"
        )
    flagged = [(case, flags[case]) for case in sorted(flags)]
    flagged_html = ""
    if flagged:
        items = "".join(
            f"<li><code>{esc(case)}</code> {flag_badges(case_flags)}</li>" for case, case_flags in flagged
        )
        flagged_html = f"<h2>Flagged cases ({len(flagged)})</h2><ul>{items}</ul>"
    table = (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>case</th><th>group</th><th>eager/PyTorch</th><th>eager/JAX</th>"
        "<th>trace/eager</th><th>tenferro status</th><th>flags</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    return flagged_html + "<h2>Case ratios</h2>" + table


def render_charts(document: dict[str, Any]) -> str:
    unit = document.get("unit", "ms")
    grouped = case_rows(document)
    parts = ["<h2>Per-case medians</h2>"]
    for case in sorted(grouped):
        parts.append(f"<h3><code>{esc(case)}</code></h3>")
        parts.append(svg_bar_chart(grouped[case], unit))
    return "".join(parts)


def render_suite_page(
    profile: str,
    suite_id: str,
    report: Path,
    document: dict[str, Any] | None,
    previous_rows: list[dict[str, Any]] | None,
    markdown_html: str,
) -> str:
    body = [f"<p><a href=\"../../index.html\">&larr; all results</a></p>", f"<h1>{esc(suite_id)}</h1>"]
    meta = [f"<span>profile: <strong>{esc(profile)}</strong></span>"]
    if document:
        hardware = hardware_name(document)
        if hardware:
            meta.append(f"<span>hardware: {esc(hardware)}</span>")
        if document.get("timestamp"):
            meta.append(f"<span>date: {esc(str(document['timestamp'])[:19].replace('T', ' '))}Z</span>")
        if document.get("tenferro_commit"):
            meta.append(f"<span>tenferro: <code>{esc(str(document['tenferro_commit'])[:12])}</code></span>")
        if document.get("unit"):
            meta.append(f"<span>unit: {esc(document['unit'])}</span>")
        if (document.get("hardware") or {}).get("threads"):
            meta.append(f"<span>threads: {esc(document['hardware']['threads'])}</span>")
    else:
        meta.append('<span class="muted">no machine-readable summary</span>')
    body.append('<div class="profile-head">' + "".join(meta) + "</div>")
    if profile in GITHUB_HOSTED_PROFILES:
        body.append(f'<p class="notice">{REPORT_QUALITY_NOTICE}</p>')
    if document:
        body.append(render_case_table(document, previous_rows))
        body.append(render_charts(document))
    body.append("<h2>Original report</h2>")
    body.append(markdown_html)
    return page(f"{suite_id} - {profile}", "\n".join(body))


# ---------------------------------------------------------------------------
# Rendering / generation
# ---------------------------------------------------------------------------

def render_markdown(text: str) -> str:
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


def standalone_report_page(report: Path) -> str:
    return page(
        report.stem,
        f'<p><a href="../../index.html">&larr; all results</a></p>'
        + render_markdown(report.read_text(encoding="utf-8")),
    )


def generate(
    result_dir: Path,
    output_dir: Path,
    baseline_dir: Path | None = None,
    git_repo: Path | None = None,
    git_result_dir: str = "result",
) -> int:
    reports = discover_reports(result_dir)
    if not reports:
        print(f"error: no reports found under {result_dir}", file=sys.stderr)
        return 1
    summaries = load_summaries(result_dir)
    previous_by_key: dict[tuple[str, str], list[dict[str, Any]] | None] = {}
    for profile, suite_id, _report in reports:
        if (profile, suite_id) in MAINTAINED:
            _hardware, previous = previous_rows_for(
                profile, suite_id, summaries.get((profile, suite_id)) or {}, baseline_dir, git_repo, git_result_dir
            )
            previous_by_key[(profile, suite_id)] = previous

    output_dir.mkdir(parents=True, exist_ok=True)
    for profile, suite_id, report in reports:
        suite_dir = suite_id.split("/", 1)[0]
        target = output_dir / profile / suite_dir
        target.mkdir(parents=True, exist_ok=True)
        document = summaries.get((profile, suite_id))
        page_html = render_suite_page(
            profile,
            suite_id,
            report,
            document,
            previous_by_key.get((profile, suite_id)),
            render_markdown(report.read_text(encoding="utf-8")),
        )
        (target / f"{report.stem}.html").write_text(page_html, encoding="utf-8")
        shutil.copy2(report, target / report.name)

    landing = render_landing(reports, summaries, previous_by_key)
    (output_dir / "index.html").write_text(landing, encoding="utf-8")
    print(f"generated dashboard in {output_dir} with {len(reports)} reports")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=None,
                        help="read previous summaries from this tree instead of git history")
    parser.add_argument("--git-repo", type=Path, default=Path("."))
    parser.add_argument("--git-result-dir", default="result")
    args = parser.parse_args()
    return generate(
        args.result_dir,
        args.output_dir,
        args.baseline_dir,
        args.git_repo if args.baseline_dir is None else None,
        args.git_result_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
