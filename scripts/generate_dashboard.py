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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Arm:
    """A tenferro column arm, resolved to the first backend id present."""

    label: str
    backends: tuple[str, ...]


@dataclass(frozen=True)
class RatioRule:
    """One landing-page ratio column for a suite.

    ``reference_backends`` is the set of competitor ids; the best (smallest)
    successful median of that set is the denominator.  ``per_case_reference``
    resolves the denominator per case from the perf-issue reference map
    instead of a fixed backend set.
    """

    column: str
    reference_backends: tuple[str, ...]
    arms: tuple[Arm, ...]
    per_case_reference: bool = False


EAGER_TRACE_ARMS = (
    Arm("eager", (TENFERRO_EAGER, "tenferro-eager")),
    Arm("trace", (TENFERRO_TRACE, "tenferro-trace")),
)
PERMUTATION_ARMS = (
    Arm("transpose", ("tenferro-cuda-transpose",)),
    Arm("to_contiguous", ("tenferro-cuda-to-contiguous",)),
    Arm("copy_into", ("tenferro-cuda-destination-reuse",)),
)
DEFAULT_RATIO_RULES = (
    RatioRule("vs PyTorch", tuple(sorted(PYTORCH_BACKENDS)), EAGER_TRACE_ARMS),
    RatioRule("vs JAX", tuple(sorted(JAX_BACKENDS)), EAGER_TRACE_ARMS),
)
# suite_id -> ratio columns.  Anything not listed uses DEFAULT_RATIO_RULES;
# an explicit empty tuple means the suite has no tenferro comparison column.
RATIO_RULES: dict[str, tuple[RatioRule, ...]] = {
    "gpu/permutation": (
        RatioRule("vs PyTorch", ("pytorch-cuda",), PERMUTATION_ARMS),
        RatioRule("vs cuTENSOR", ("cutensor",), PERMUTATION_ARMS),
    ),
    "gpu/perf_issues": (RatioRule("vs reference", (), (), per_case_reference=True),),
    "gpu/sparse": (),
}
RATIO_COLUMN_ORDER = ("vs PyTorch", "vs JAX", "vs cuTENSOR", "vs reference")
# The perf-issue cases declare their reference arms in this instance file.
GPU_PERF_ISSUES_INSTANCES = Path("data/instances/gpu_perf_issues.json")


def ratio_rules_for(suite_id: str) -> tuple[RatioRule, ...]:
    return RATIO_RULES.get(suite_id, DEFAULT_RATIO_RULES)


def ratio_columns(suite_ids: Iterable[str]) -> list[str]:
    present = {rule.column for suite_id in suite_ids for rule in ratio_rules_for(suite_id)}
    return [column for column in RATIO_COLUMN_ORDER if column in present] + sorted(
        present - set(RATIO_COLUMN_ORDER)
    )

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


def workload_key(row: dict[str, Any]) -> tuple[str, str]:
    """Full workload identity: benchmark name plus dtype/shape/threads group."""
    return (str(row.get("case")), str(row.get("group", "")))


def grouped_workloads(document: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group rows by ``(case, group)`` so 1T/4T or dtype variants stay distinct."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in document.get("rows", []):
        grouped.setdefault(workload_key(row), []).append(row)
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


# ---------------------------------------------------------------------------
# Ratios and flags
# ---------------------------------------------------------------------------

def geometric_mean(values: Iterable[float]) -> float | None:
    values = [value for value in values if value > 0]
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def is_tenferro(row: dict[str, Any]) -> bool:
    return str(row.get("backend", "")).startswith("tenferro")


def is_neutral_unsupported(row: dict[str, Any]) -> bool:
    """A tenferro row the harness explicitly did not measure by policy."""
    return row.get("status") == "unsupported" and bool(row.get("reason"))


def is_tenferro_anomaly(row: dict[str, Any]) -> bool:
    """True when a tenferro row is a genuine anomaly, not a policy skip."""
    if not is_tenferro(row):
        return False
    status = row.get("status")
    if status == "ok":
        return row.get("verified") is False
    if status == "unsupported":
        return not bool(row.get("reason"))
    return True


def not_measured_cases(
    document: dict[str, Any],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Workloads with tenferro rows that carry a harness-policy unsupported reason."""
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, rows in grouped_workloads(document).items():
        entries = [row for row in rows if is_tenferro(row) and is_neutral_unsupported(row)]
        if entries:
            result[key] = entries
    return result


def arm_median(rows: list[dict[str, Any]], arm: Arm) -> float | None:
    for backend in arm.backends:
        value = row_median(rows, backend)
        if value is not None:
            return value
    return None


def load_perf_reference_map(path: Path) -> dict[str, list[str]]:
    """Map a perf-issue tenferro case id to its reference case ids."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    references: dict[str, list[str]] = {}
    for entry in data:
        target = entry.get("reference_for")
        if target:
            references.setdefault(target, []).append(entry["id"])
    return references


def case_ratio_entries(
    rows: list[dict[str, Any]],
    case: str,
    rule: RatioRule,
    reference_map: dict[str, list[str]],
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    """Return ``{arm_label: tenferro/reference ratio}`` for one case and rule."""
    entries: dict[str, float] = {}
    if rule.per_case_reference:
        reference_values: list[float] = []
        for reference_case in reference_map.get(case, []):
            for row in grouped.get(reference_case, []):
                if row.get("status") == "ok" and row.get("median"):
                    reference_values.append(float(row["median"]))
        if not reference_values:
            return entries
        denominator = min(reference_values)
        for row in rows:
            if is_tenferro(row) and row.get("status") == "ok" and row.get("median"):
                entries[str(row.get("backend"))] = float(row["median"]) / denominator
        return entries
    references = set(rule.reference_backends)
    for arm in rule.arms:
        numerator = arm_median(rows, arm)
        denominator = best_reference(rows, references)
        if numerator is not None and denominator:
            entries[arm.label] = numerator / denominator
    return entries


def suite_ratios(
    document: dict[str, Any],
    suite_id: str,
    reference_map: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Geometric-mean tenferro/reference ratios per landing column.

    Returns ``{column: {"arms": {arm_label: ratio}, "count": n}}``.
    """
    grouped = grouped_workloads(document)
    by_case = case_rows(document)
    result: dict[str, Any] = {}
    for rule in ratio_rules_for(suite_id):
        arm_values: dict[str, list[float]] = {}
        count = 0
        for (case, _group), rows in grouped.items():
            entries = case_ratio_entries(rows, case, rule, reference_map or {}, by_case)
            if entries:
                count += 1
            for label, value in entries.items():
                arm_values.setdefault(label, []).append(value)
        if arm_values:
            result[rule.column] = {
                "arms": {
                    label: geometric_mean(values) for label, values in arm_values.items() if values
                },
                "count": count,
            }
    return result


def perf_reference_details(
    document: dict[str, Any],
    reference_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Per-case tenferro/reference ratios for perf-issue suites, worst first."""
    grouped = grouped_workloads(document)
    by_case = case_rows(document)
    details: list[dict[str, Any]] = []
    for (case, _group), rows in grouped.items():
        for row in rows:
            if not is_tenferro(row) or row.get("status") != "ok" or row.get("median") is None:
                continue
            reference_values: list[tuple[float, str]] = []
            for reference_case in reference_map.get(case, []):
                for reference_row in by_case.get(reference_case, []):
                    if reference_row.get("status") == "ok" and reference_row.get("median"):
                        reference_values.append((float(reference_row["median"]), reference_case))
            if not reference_values:
                continue
            denominator, reference_case = min(reference_values)
            details.append({
                "case": case,
                "backend": str(row.get("backend")),
                "median": float(row["median"]),
                "reference_case": reference_case,
                "reference": denominator,
                "ratio": float(row["median"]) / denominator,
            })
    return sorted(details, key=lambda item: item["ratio"], reverse=True)


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
    if any(is_tenferro_anomaly(row) for row in rows):
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
) -> dict[tuple[str, str], list[str]]:
    if not document:
        return {}
    grouped = grouped_workloads(document)
    previous_by_workload: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in previous_rows or []:
        previous_by_workload.setdefault(workload_key(row), []).append(row)
    flags: dict[tuple[str, str], list[str]] = {}
    for key, rows in grouped.items():
        case_flags = compute_case_flags(rows, previous_by_workload.get(key))
        if case_flags:
            flags[key] = case_flags
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
    if not entry or not entry.get("arms"):
        return '<td class="ratio-none">-</td>'
    parts = []
    for label, value in entry["arms"].items():
        if value is None:
            continue
        parts.append(
            f'<span class="ratio {ratio_class(value)}">{esc(label)} {fmt_ratio(value)}</span>'
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
    if use_log:
        low = math.log10(min(values))
        high = math.log10(max(values))
        span = high - low or 1.0
        fractions = [(math.log10(value) - low) / span for value in values]
    else:
        # Linear bars must start at zero, not at the smallest median.
        span = max(values) or 1.0
        fractions = [value / span for value in values]

    height = len(points) * (BAR_HEIGHT + BAR_GAP) + 8
    parts = [
        f'<svg class="bars" viewBox="0 0 {BAR_CHART_WIDTH} {height}" '
        f'width="100%" role="img" aria-label="median by backend">'
    ]
    if use_log:
        parts.append('<text x="0" y="10" class="chart-note">log scale (baseline = smallest median)</text>')
    for index, ((backend, value), fraction) in enumerate(zip(points, fractions)):
        y = index * (BAR_HEIGHT + BAR_GAP) + (16 if use_log else 4)
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
.not-measured { color: var(--muted); font-weight: 600; }
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


def render_flags_cell(flags: dict[str, list[str]]) -> str:
    if not flags:
        return '<td><span class="muted">none</span></td>'
    counts: dict[str, int] = {}
    for case_flags in flags.values():
        for flag in case_flags:
            counts[flag] = counts.get(flag, 0) + 1
    return (
        f'<td><strong>{len(flags)}</strong> case(s): '
        + " ".join(f'<span class="flag">{esc(flag)}×{count}</span>' for flag, count in sorted(counts.items()))
        + "</td>"
    )


def render_not_measured_cell(neutral: dict[str, list[dict[str, Any]]]) -> str:
    if not neutral:
        return '<td><span class="muted">0</span></td>'
    return f'<td><span class="not-measured">{len(neutral)}</span></td>'


def landing_ratio_cells(
    document: dict[str, Any] | None,
    suite_id: str,
    reference_map: dict[str, list[str]],
) -> dict[str, str]:
    ratios = suite_ratios(document, suite_id, reference_map) if document else {}
    return {
        rule.column: render_ratio_cell(ratios.get(rule.column))
        for rule in ratio_rules_for(suite_id)
    }


def render_landing(
    reports: list[tuple[str, str, Path]],
    summaries: dict[tuple[str, str], dict[str, Any]],
    previous_by_key: dict[tuple[str, str], list[dict[str, Any]] | None],
    reference_map: dict[str, list[str]],
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
            columns = ratio_columns(suite_id for _profile, suite_id, _report in maintained)
            body.append('<div class="table-wrap"><table>')
            body.append(
                "<thead><tr><th>suite</th>"
                + "".join(f"<th>{esc(column)}</th>" for column in columns)
                + "<th>flags</th><th>not measured</th><th>report</th></tr></thead>"
            )
            body.append("<tbody>")
            for _profile, suite_id, report in sorted(maintained):
                document = summaries.get((profile, suite_id))
                cells = landing_ratio_cells(document, suite_id, reference_map)
                ratio_html = "".join(
                    cells.get(column, '<td class="ratio-none">-</td>') for column in columns
                )
                flags = suite_flags(document, previous_by_key.get((profile, suite_id))) if document else {}
                neutral = not_measured_cases(document) if document else {}
                if document is None:
                    flags_cell = '<td class="ratio-none">-</td>'
                    neutral_cell = '<td class="ratio-none">-</td>'
                else:
                    flags_cell = render_flags_cell(flags)
                    neutral_cell = render_not_measured_cell(neutral)
                suite_dir = suite_id.split("/", 1)[0]
                href = f"{profile}/{suite_dir}/{Path(report).stem}.html"
                body.append(
                    f'<tr><td><code>{esc(suite_id)}</code></td>'
                    f"{ratio_html}{flags_cell}{neutral_cell}"
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

def render_case_table(
    document: dict[str, Any],
    suite_id: str,
    previous_rows: list[dict[str, Any]] | None,
    reference_map: dict[str, list[str]],
) -> str:
    flags = suite_flags(document, previous_rows)
    neutral = not_measured_cases(document)
    grouped = grouped_workloads(document)
    by_case = case_rows(document)
    rules = ratio_rules_for(suite_id)
    rows = []
    for case, group in sorted(grouped):
        rows_for_case = grouped[(case, group)]
        cells = []
        for rule in rules:
            entries = case_ratio_entries(rows_for_case, case, rule, reference_map, by_case)
            cells.append(render_ratio_cell({"arms": entries} if entries else None))
        if (case, group) in neutral:
            reasons = "; ".join(
                f"{esc(row.get('backend'))}: {esc(row.get('reason') or '')}" for row in neutral[(case, group)]
            )
            note = f'<span class="not-measured">not measured</span> <span class="muted">{reasons}</span>'
        else:
            note = '<span class="muted">-</span>'
        flags_text = flag_badges(flags.get((case, group), [])) or '<span class="muted">-</span>'
        rows.append(
            f"<tr><td><code>{esc(case)}</code></td>"
            f"<td>{esc(group or '-')}</td>"
            + "".join(cells)
            + f"<td>{note}</td><td>{flags_text}</td></tr>"
        )
    flagged = [(key, flags[key]) for key in sorted(flags)]
    sections = []
    if flagged:
        items = "".join(
            f"<li><code>{esc(case)}</code> <span class=\"muted\">{esc(group)}</span> {flag_badges(case_flags)}</li>"
            for (case, group), case_flags in flagged
        )
        sections.append(f"<h2>Flagged cases ({len(flagged)})</h2><ul>{items}</ul>")
    if neutral:
        items = "".join(
            f"<li><code>{esc(case)}</code> <span class=\"muted\">{esc(group)}</span> "
            + '<span class="not-measured">not measured</span>: '
            + "; ".join(f"{esc(row.get('backend'))} — {esc(row.get('reason') or '')}" for row in rows_)
            + "</li>"
            for (case, group), rows_ in sorted(neutral.items())
        )
        sections.append(f"<h2>Not measured (harness policy) ({len(neutral)})</h2><ul>{items}</ul>")
    table = (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>case</th><th>group</th>"
        + "".join(f"<th>{esc(rule.column)}</th>" for rule in rules)
        + "<th>not measured</th><th>flags</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    return "".join(sections) + "<h2>Case ratios</h2>" + table


def render_charts(document: dict[str, Any]) -> str:
    unit = document.get("unit", "ms")
    grouped = grouped_workloads(document)
    parts = ["<h2>Per-case medians</h2>"]
    for case, group in sorted(grouped):
        title = f"<code>{esc(case)}</code>"
        if group:
            title += f' <span class="muted">{esc(group)}</span>'
        parts.append(f"<h3>{title}</h3>")
        parts.append(svg_bar_chart(grouped[(case, group)], unit))
    return "".join(parts)


def render_perf_reference_table(
    document: dict[str, Any],
    reference_map: dict[str, list[str]],
    limit: int = 10,
) -> str:
    details = perf_reference_details(document, reference_map)
    if not details:
        return ""
    unit = document.get("unit", "ns/op")
    rows = "".join(
        f"<tr><td><code>{esc(item['case'])}</code></td><td>{esc(item['backend'])}</td>"
        f"<td>{fmt_median(item['median'], unit)}</td>"
        f"<td><code>{esc(item['reference_case'])}</code></td>"
        f"<td>{fmt_median(item['reference'], unit)}</td>"
        f'<td class="ratio {ratio_class(item["ratio"])}">{fmt_ratio(item["ratio"])}</td></tr>'
        for item in details[:limit]
    )
    return (
        f"<h2>Worst ratios vs reference (top {min(limit, len(details))} of {len(details)})</h2>"
        '<div class="table-wrap"><table><thead><tr>'
        "<th>tenferro case</th><th>backend</th><th>tenferro</th>"
        "<th>reference case</th><th>reference</th><th>ratio</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table></div>"
    )


def render_suite_page(
    profile: str,
    suite_id: str,
    report: Path,
    document: dict[str, Any] | None,
    previous_rows: list[dict[str, Any]] | None,
    reference_map: dict[str, list[str]],
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
        body.append(render_case_table(document, suite_id, previous_rows, reference_map))
        if any(rule.per_case_reference for rule in ratio_rules_for(suite_id)):
            body.append(render_perf_reference_table(document, reference_map))
        body.append(render_charts(document))
    body.append("<h2>Original report</h2>")
    body.append(markdown_html)
    return page(f"{suite_id} - {profile}", "\n".join(body))


# ---------------------------------------------------------------------------
# Rendering / generation
# ---------------------------------------------------------------------------

def render_markdown(text: str) -> str:
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


def generate(
    result_dir: Path,
    output_dir: Path,
    baseline_dir: Path | None = None,
    git_repo: Path | None = None,
    git_result_dir: str = "result",
    instances_dir: Path = GPU_PERF_ISSUES_INSTANCES.parent,
) -> int:
    reports = discover_reports(result_dir)
    if not reports:
        print(f"error: no reports found under {result_dir}", file=sys.stderr)
        return 1
    summaries = load_summaries(result_dir)
    reference_map = load_perf_reference_map(instances_dir / GPU_PERF_ISSUES_INSTANCES.name)
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
            reference_map,
            render_markdown(report.read_text(encoding="utf-8")),
        )
        (target / f"{report.stem}.html").write_text(page_html, encoding="utf-8")
        shutil.copy2(report, target / report.name)

    landing = render_landing(reports, summaries, previous_by_key, reference_map)
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
    parser.add_argument("--instances-dir", type=Path, default=GPU_PERF_ISSUES_INSTANCES.parent)
    args = parser.parse_args()
    return generate(
        args.result_dir,
        args.output_dir,
        args.baseline_dir,
        args.git_repo if args.baseline_dir is None else None,
        args.git_result_dir,
        args.instances_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
