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
import yaml

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
    ("nvidia-gpu", "gpu/tensornetwork"),
    ("nvidia-gpu", "gpu/elementwise"),
    ("nvidia-gpu", "gpu/sparse"),
    ("nvidia-gpu", "gpu/linalg_jvp_vjp"),
    ("nvidia-gpu", "gpu/linalg_ad_latency"),
    ("nvidia-gpu", "gpu/permutation"),
    ("nvidia-gpu", "gpu/perf_issues"),
}

PROFILE_ORDER = ["nvidia-gpu", "linux-cpu", "mac-cpu", "amd-cpu"]

# Hardware families are the top-level navigation entries.  A profile belongs to
# exactly one family; unknown ids fall back to their "-gpu"/"-cpu" suffix.
FAMILY_ORDER = ["gpu", "cpu"]
FAMILY_LABEL = {"gpu": "GPU", "cpu": "CPU"}
FAMILY_BLURB = {
    "gpu": "Compute- and bandwidth-bound GPU workloads.",
    "cpu": "Compute- and bandwidth-bound CPU workloads by hardware profile.",
}
PROFILE_FAMILY = {
    "nvidia-gpu": "gpu",
    "mac-gpu": "gpu",
    "linux-cpu": "cpu",
    "mac-cpu": "cpu",
    "amd-cpu": "cpu",
}
PROFILE_LABEL = {
    "nvidia-gpu": "NVIDIA A100 80GB PCIe",
    "mac-gpu": "Apple GPU (GitHub runner)",
    "linux-cpu": "AMD EPYC 7713P (GitHub runner)",
    "mac-cpu": "Apple silicon (GitHub runner)",
    "amd-cpu": "AMD EPYC 7713P (dedicated)",
}

OVERVIEW_CASE_LIMIT = 12

THROUGHPUT = "throughput"
LATENCY = "latency"
REGIMES = (THROUGHPUT, LATENCY)
REGIME_LABEL = {
    THROUGHPUT: "Compute / bandwidth bound",
    LATENCY: "Latency and per-call overhead",
}

# Two-sided 95% Student-t quantiles for the small sample counts the suites use.
T_QUANTILE = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
              8: 2.365, 9: 2.306, 10: 2.262, 15: 2.145, 20: 2.093, 30: 2.045}


def family_of(profile: str) -> str:
    if profile in PROFILE_FAMILY:
        return PROFILE_FAMILY[profile]
    return "gpu" if profile.endswith("-gpu") else "cpu"


def profile_label(profile: str) -> str:
    return PROFILE_LABEL.get(profile, profile)


def load_suite_metadata(benchmarks_dir: Path) -> dict[str, dict[str, Any]]:
    """Map ``suite_id`` to its regime and per-case run counts.

    The dashboard renders from ``result/`` summaries, but regimes and sample
    counts are declared in the suite YAML that produced those summaries.  A
    suite or case missing from the YAML keeps the safe default: throughput, no
    sample count (so no confidence whisker).
    """
    metadata: dict[str, dict[str, Any]] = {}
    if not benchmarks_dir.exists():
        return metadata
    for path in sorted(benchmarks_dir.rglob("*.yaml")):
        if "manifests" in path.parts:
            continue
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(document, dict) or not document.get("suite_id"):
            continue
        defaults = document.get("defaults") or {}
        default_run = defaults.get("run") or {}
        default_regime = defaults.get("regime") or THROUGHPUT
        problems: dict[str, dict[str, Any]] = {}
        for problem in document.get("problems") or []:
            if not isinstance(problem, dict) or not problem.get("id"):
                continue
            run = problem.get("run") or {}
            problems[str(problem["id"])] = {
                "regime": problem.get("regime") or default_regime,
                "runs": run.get("runs") or default_run.get("runs"),
            }
        suite_id = str(document["suite_id"])
        entry = {"regime": default_regime, "runs": default_run.get("runs"), "problems": problems}
        previous = metadata.get(suite_id)
        if previous is None or len(problems) >= len(previous.get("problems") or {}):
            metadata[suite_id] = entry
    return metadata


def case_regime(metadata: dict[str, dict[str, Any]], suite_id: str, case: str) -> str:
    suite = metadata.get(suite_id) or {}
    problem = (suite.get("problems") or {}).get(case) or {}
    return str(problem.get("regime") or suite.get("regime") or THROUGHPUT)


def case_runs(metadata: dict[str, dict[str, Any]], suite_id: str, case: str) -> int | None:
    suite = metadata.get(suite_id) or {}
    problem = (suite.get("problems") or {}).get(case) or {}
    value = problem.get("runs") or suite.get("runs")
    return int(value) if isinstance(value, (int, float)) and value > 0 else None

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
    previous summary for the same hardware (passed as ``previous_rows``); only
    tenferro rows count - a slower reference backend is not a tenferro regression.
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
            if not is_tenferro(row) or row.get("status") != "ok" or row.get("median") is None:
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

BAR_CHART_WIDTH = 880
BAR_LABEL_WIDTH = 300
BAR_AREA_WIDTH = 420
BAR_HEIGHT = 12
BAR_GAP = 4

RATIO_COLORS = {"eager": "bar-tenferro", "trace": "bar-tenferro-alt"}


def relative_interval(median: float, iqr: float | None, runs: int | None) -> tuple[float, float] | None:
    """Estimated 95% CI of a median, as multiplicative bounds around it."""
    if not iqr or not runs or median <= 0:
        return None
    tq = T_QUANTILE.get(runs)
    if tq is None:
        tq = 2.0 if runs >= 30 else 2.5
    se = tq * 1.253 * (iqr / 1.349) / (math.sqrt(runs) * median)
    return math.exp(-se), math.exp(se)


def case_points(rows: list[dict[str, Any]]) -> list[tuple[str, float, float | None]]:
    """Successful medians as (backend, median, iqr), references first."""
    points = []
    for row in rows:
        if row.get("status") != "ok" or row.get("median") is None:
            continue
        value = float(row["median"])
        if value <= 0:
            continue
        iqr = float(row["iqr"]) if row.get("iqr") else None
        points.append((str(row.get("backend")), value, iqr))
    references = sorted((p for p in points if not p[0].startswith("tenferro")), key=lambda p: p[1])
    tenferro = [p for p in points if p[0].startswith("tenferro")]
    return references + tenferro


def whisker(parts: list[str], x1: float, x2: float, y: float, height: float) -> None:
    """Thin black interval with caps and a white halo, quiet against the bars."""
    centre = y + height / 2
    parts.append(f'<line x1="{x1:.1f}" y1="{centre:.1f}" x2="{x2:.1f}" y2="{centre:.1f}" stroke="#ffffff" stroke-width="4"></line>')
    parts.append(f'<line x1="{x1:.1f}" y1="{centre:.1f}" x2="{x2:.1f}" y2="{centre:.1f}" stroke="#1b1f24" stroke-width="1"></line>')
    for x in (x1, x2):
        parts.append(f'<line x1="{x:.1f}" y1="{centre - 4:.1f}" x2="{x:.1f}" y2="{centre + 4:.1f}" stroke="#ffffff" stroke-width="3"></line>')
        parts.append(f'<line x1="{x:.1f}" y1="{centre - 4:.1f}" x2="{x:.1f}" y2="{centre + 4:.1f}" stroke="#1b1f24" stroke-width="1"></line>')


def svg_case_chart(
    rows: list[dict[str, Any]],
    unit: str,
    runs: int | None = None,
    baseline: dict[str, Any] | None = None,
    baseline_label: str = "PyTorch",
) -> str:
    """Absolute medians for one case, on a log time axis.

    Bar length is the median time; the dashed line is the best reference.  A
    thin outlined whisker marks the estimated 95% CI from the case's ``runs``
    samples.  Tenferro arms are filled, reference arms are outlined, so a slow
    reference never dominates the picture.
    """
    points = case_points(rows)
    if not points:
        return '<p class="muted">No successful measurements.</p>'
    reference = float(baseline["median"]) if baseline is not None else next(
        (value for backend, value, _ in points if not backend.startswith("tenferro")), None
    )
    values = [value for _, value, _ in points]
    low = math.log10(min(values) / 1.25)
    high = math.log10(max(values) * 1.25)
    span = high - low or 1.0

    def x_of(value: float) -> float:
        return BAR_LABEL_WIDTH + (math.log10(value) - low) / span * BAR_AREA_WIDTH

    top = 24
    height = len(points) * (BAR_HEIGHT + BAR_GAP) + top + 8
    parts = [
        f'<svg class="bars" viewBox="0 0 {BAR_CHART_WIDTH} {height}" '
        f'width="100%" role="img" aria-label="median by backend">'
    ]
    note = "bar = median time (log scale) · shorter (left) is faster"
    if reference is not None:
        note += f" · dashed line = {baseline_label} ({fmt_median(reference, unit)})"
    parts.append(f'<text x="0" y="10" class="chart-note">{esc(note)}</text>')
    if reference is not None:
        parts.append(
            f'<line class="chart-reference" x1="{x_of(reference):.1f}" y1="{top - 6}" '
            f'x2="{x_of(reference):.1f}" y2="{height - 4}"></line>'
        )
    for index, (backend, value, iqr) in enumerate(points):
        y = index * (BAR_HEIGHT + BAR_GAP) + top
        end = x_of(value)
        width = max(2.0, end - BAR_LABEL_WIDTH)
        if backend.startswith("tenferro"):
            cls = "bar-tenferro" if "eager" in backend or "transpose" in backend else "bar-tenferro-alt"
        else:
            cls = "bar-reference"
        parts.append(f'<text x="0" y="{y + 9}" class="chart-label">{esc(backend)}</text>')
        parts.append(f'<rect x="{BAR_LABEL_WIDTH}" y="{y}" width="{width:.1f}" height="{BAR_HEIGHT}" class="{cls}"></rect>')
        label = fmt_median(value, unit)
        right = end
        interval = relative_interval(value, iqr, runs)
        if interval:
            x1, x2 = x_of(value * interval[0]), x_of(value * interval[1])
            whisker(parts, x1, x2, y, BAR_HEIGHT)
            right = max(right, x2)
        if reference is not None and backend.startswith("tenferro"):
            label += f" · {value / reference:.2f}x"
        parts.append(f'<text x="{right + 6:.1f}" y="{y + 9}" class="chart-value">{esc(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_ratio_chart(cases: list[dict[str, Any]], baseline_label: str) -> str:
    """Per-case ratios for one suite, one bar per contender.

    Every bar is ``case median / baseline median``; the dashed line is the
    baseline (1.00x).  Tenferro arms are filled and the competitor arms are
    outlined white, so the reference arms recede.  Whiskers are the estimated
    ratio CI from the case's sample count.
    """
    rows = [case for case in cases if case["contenders"]]
    if not rows:
        return ""
    ratios = [item["ratio"] for case in rows for item in case["contenders"]]
    low = math.log10(min(ratios) / 1.15)
    high = math.log10(max(ratios) * 1.15)
    span = high - low or 1.0

    def x_of(value: float) -> float:
        return BAR_LABEL_WIDTH + (math.log10(value) - low) / span * BAR_AREA_WIDTH

    row_height = max(1, max(len(case["contenders"]) for case in rows)) * (BAR_HEIGHT + BAR_GAP) + 10
    height = len(rows) * row_height + 34
    parts = [
        f'<svg class="bars" viewBox="0 0 {BAR_CHART_WIDTH} {height}" '
        f'width="100%" role="img" aria-label="per-case ratio against {esc(baseline_label)}">'
    ]
    parts.append(
        f'<text x="0" y="10" class="chart-note">bar = case median / {esc(baseline_label)} median · '
        "dashed line = baseline (1.00x) · log scale · shorter (left) is faster · "
        "whisker = estimated 95% CI</text>"
    )
    parity = x_of(1.0)
    parts.append(
        f'<line class="chart-reference" x1="{parity:.1f}" y1="16" x2="{parity:.1f}" y2="{height - 4}"></line>'
    )
    y = 24
    for case in rows:
        parts.append(f'<text x="0" y="{y + 9}" class="chart-label">{esc(case["case"])}</text>')
        for index, item in enumerate(case["contenders"]):
            bar_y = y + index * (BAR_HEIGHT + BAR_GAP)
            end = x_of(item["ratio"])
            width = max(2.0, end - BAR_LABEL_WIDTH)
            if item.get("kind") == "reference":
                parts.append(
                    f'<rect x="{BAR_LABEL_WIDTH}" y="{bar_y}" width="{width:.1f}" height="{BAR_HEIGHT}" '
                    'fill="#ffffff" stroke="#1b1f24" stroke-width="1"></rect>'
                )
            else:
                parts.append(
                    f'<rect x="{BAR_LABEL_WIDTH}" y="{bar_y}" width="{width:.1f}" height="{BAR_HEIGHT}" '
                    f'class="{RATIO_COLORS.get(item["label"], "bar-tenferro")}"></rect>'
                )
            interval = item.get("interval")
            right = end
            if interval:
                x1, x2 = x_of(item["ratio"] * interval[0]), x_of(item["ratio"] * interval[1])
                whisker(parts, x1, x2, bar_y, BAR_HEIGHT)
                right = max(right, x2)
            parts.append(
                f'<text x="{right + 6:.1f}" y="{bar_y + 9}" class="chart-value">'
                f'{esc(item["label"])} {item["ratio"]:.2f}x</text>'
            )
        y += row_height
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
  --bar-tenferro: #0969da; --bar-tenferro-alt: #8250df; --bar-reference: #8c959f; --flag-bg: #fff1e5; --flag-fg: #953800;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #30363d;
    --card: #161b22; --link: #4493f8; --accent: #4493f8;
    --faster: #3fb950; --parity: #9198a1; --slower: #d29922; --much-slower: #f85149;
    --bar-tenferro: #4493f8; --bar-tenferro-alt: #a371f7; --bar-reference: #6e7681; --flag-bg: #3d1d00; --flag-fg: #ffa657;
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
.chart-reference { stroke: var(--fg); stroke-width: 1; stroke-dasharray: 4 3; opacity: .5; }
.chart-legend { color: var(--muted); font-size: .9rem; margin: .2rem 0 .8rem; }
.bar-tenferro { fill: var(--bar-tenferro); }
.bar-tenferro-alt { fill: var(--bar-tenferro-alt); }
.bar-reference { fill: #ffffff; stroke: var(--fg); stroke-width: 1; }
.bar-faster { fill: var(--faster); }
.bar-parity { fill: var(--parity); }
.bar-slower { fill: var(--slower); }
.bar-much-slower { fill: var(--much-slower); }
.legend { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: .8rem 1rem; margin: 1rem 0; }
.legend h2 { margin: .1rem 0 .5rem; font-size: 1rem; }
.legend dl { display: grid; grid-template-columns: max-content 1fr; gap: .25rem .9rem; margin: 0; font-size: .9rem; }
.legend dt { font-weight: 600; }
.legend dd { margin: 0; color: var(--muted); }
details { border: 1px solid var(--line); border-radius: 8px; padding: .6rem .9rem; margin: 1rem 0; }
summary { cursor: pointer; font-weight: 600; }
details.diag { background: var(--card); }
footer { margin-top: 3rem; font-size: .85rem; color: var(--muted); border-top: 1px solid var(--line); padding-top: 1rem; }
.family-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin: 2rem 0; }
.family-card { display: block; border: 1px solid var(--line); border-radius: 10px; padding: 1.2rem; background: var(--card); color: var(--fg); }
.family-card:hover { border-color: var(--accent); text-decoration: none; }
.family-card h2 { margin: 0 0 .4rem; font-size: 1.4rem; }
.family-card p { margin: .2rem 0; }
.family-card .facts { color: var(--muted); font-size: .9rem; }
.suite-block { margin: 1rem 0; }
.suite-block > summary { font-size: 1rem; }
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


@dataclass
class CaseView:
    """One measured workload: the unit that is never averaged with another."""

    case: str
    group: str
    regime: str
    runs: int | None
    rows: list[dict[str, Any]]


@dataclass
class SuiteView:
    profile: str
    suite_id: str
    report: Path
    document: dict[str, Any] | None
    cases: list[CaseView]

    def regime_cases(self, regime: str) -> list[CaseView]:
        return [case for case in self.cases if case.regime == regime]


def build_suite_views(
    reports: list[tuple[str, str, Path]],
    summaries: dict[tuple[str, str], dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> list[SuiteView]:
    views: list[SuiteView] = []
    for profile, suite_id, report in reports:
        document = summaries.get((profile, suite_id))
        cases: list[CaseView] = []
        if document:
            for (case, group), rows in sorted(grouped_workloads(document).items()):
                cases.append(
                    CaseView(
                        case=case,
                        group=group,
                        regime=case_regime(metadata, suite_id, case),
                        runs=case_runs(metadata, suite_id, case),
                        rows=rows,
                    )
                )
        views.append(SuiteView(profile, suite_id, report, document, cases))
    return views


def ok_row(rows: list[dict[str, Any]], backends: Iterable[str]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("backend") in backends and row.get("status") == "ok" and row.get("median") is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["median"]))


def case_baseline(
    case: CaseView,
    suite_id: str,
    reference_map: dict[str, list[str]],
    grouped: dict[str, list[dict[str, Any]]],
) -> tuple[str, dict[str, Any] | None]:
    """Return (label, baseline row) for one case.

    PyTorch is the baseline wherever it was measured; otherwise the case's
    declared reference arm (perf issues); otherwise the best measured competitor.
    """
    row = ok_row(case.rows, PYTORCH_BACKENDS)
    if row is not None:
        return "PyTorch", row
    reference_rows: list[dict[str, Any]] = []
    for reference_case in reference_map.get(case.case, []):
        reference_rows.extend(grouped.get(reference_case, []))
    row = ok_row(reference_rows, {str(item.get("backend")) for item in reference_rows})
    if row is not None:
        return "declared reference", row
    others = {str(item.get("backend")) for item in case.rows if not is_tenferro(item)}
    row = ok_row(case.rows, others)
    if row is not None:
        return "best reference", row
    return "", None


def ratio_interval(
    contender: dict[str, Any],
    baseline: dict[str, Any],
    runs: int | None,
) -> tuple[float, float] | None:
    numerator = relative_interval(float(contender["median"]), contender.get("iqr"), runs)
    denominator = relative_interval(float(baseline["median"]), baseline.get("iqr"), runs)
    if not numerator or not denominator:
        return None
    return numerator[0] / denominator[1], numerator[1] / denominator[0]


def case_ratio_row(
    case: CaseView,
    suite_id: str,
    reference_map: dict[str, list[str]],
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Per-case contender ratios against the case baseline."""
    label, baseline = case_baseline(case, suite_id, reference_map, grouped)
    item: dict[str, Any] = {"case": case.case, "baseline": label, "contenders": []}
    if baseline is None:
        return item
    baseline_median = float(baseline["median"])
    rule = next(
        (rule for rule in ratio_rules_for(suite_id) if not rule.per_case_reference),
        None,
    )
    seen: set[str] = set()
    contenders: list[tuple[str, dict[str, Any], str]] = []
    if rule is not None:
        for arm in rule.arms:
            row = ok_row(case.rows, arm.backends)
            if row is not None:
                contenders.append((arm.label, row, "tenferro"))
    else:
        for row in case.rows:
            if is_tenferro(row) and row.get("status") == "ok" and row.get("median") is not None:
                contenders.append((str(row.get("backend")), row, "tenferro"))
    for vendor, backends in (("JAX", JAX_BACKENDS), ("cuTENSOR", {"cutensor"})):
        row = ok_row(case.rows, backends)
        if row is not None and row is not baseline:
            contenders.append((vendor, row, "reference"))
    for name, row, kind in contenders:
        if name in seen:
            continue
        seen.add(name)
        ratio = float(row["median"]) / baseline_median
        item["contenders"].append(
            {
                "label": name,
                "kind": kind,
                "ratio": ratio,
                "interval": ratio_interval(row, baseline, case.runs),
            }
        )
    return item


def suite_baseline_label(cases: list[CaseView], suite_id: str, reference_map: dict[str, list[str]], grouped: dict[str, list[dict[str, Any]]]) -> str:
    counts: dict[str, int] = {}
    for case in cases:
        label = case_baseline(case, suite_id, reference_map, grouped)[0]
        if label:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "baseline"
    return max(counts, key=lambda label: counts[label])


def how_to_read() -> str:
    """Plain-language legend shared by the CPU and GPU pages."""
    return (
        '<section class="legend"><h2>How to read</h2><dl>'
        "<dt>1.02x</dt><dd>this case's median divided by the baseline median. "
        "Below 1.00x is faster, above is slower. <strong>We never average cases</strong>: "
        "a 32x32 batched matmul and a 3072 matmul are different workloads.</dd>"
        "<dt>baseline</dt><dd>PyTorch wherever the suite measured it; otherwise the case's "
        "declared reference arm. The dashed line is that baseline at 1.00x.</dd>"
        "<dt>bars</dt><dd>tenferro eager (blue) and trace (purple) are filled; PyTorch, JAX and "
        "cuTENSOR bars are outlined white so the reference arms stay quiet.</dd>"
        "<dt>whisker</dt><dd>estimated 95% CI of the median from the case's measured runs "
        "(IQR and sample count). A difference smaller than the whisker is not established "
        "by this run.</dd>"
        "<dt>regimes</dt><dd>compute/bandwidth-bound results and latency/overhead results are "
        "on separate pages and are never mixed. A small workload is not evidence of slow compute.</dd>"
        "</dl></section>"
    )


def family_notice(family: str) -> str:
    return (
        f'<p class="notice">Compute- and bandwidth-bound {FAMILY_LABEL[family]} cases only. '
        'Per-call latency results are on the <a href="latency/index.html">latency page</a>.</p>'
    )


def render_diagnostics(views: list[SuiteView], prefix: str) -> str:
    diagnostics = [view for view in views if (view.profile, view.suite_id) not in MAINTAINED]
    if not diagnostics:
        return ""
    groups = []
    for profile in ordered_profiles({view.profile for view in diagnostics}):
        rows = []
        for view in sorted((v for v in diagnostics if v.profile == profile), key=lambda v: v.suite_id):
            mtime = datetime.fromtimestamp(view.report.stat().st_mtime, tz=timezone.utc)
            suite_dir = view.suite_id.split("/", 1)[0]
            href = f"{prefix}{profile}/{suite_dir}/{view.report.stem}.html"
            rows.append(
                f'<li><a href="{esc(href)}"><code>{esc(view.suite_id)}</code></a> '
                f'<span class="muted">({esc(mtime.date().isoformat())})</span></li>'
            )
        groups.append(f"<h3>{esc(profile)}</h3><ul>" + "".join(rows) + "</ul>")
    return (
        f'<details class="diag"><summary>Diagnostic reports ({len(diagnostics)})</summary>'
        + "".join(groups)
        + "</details>"
    )


def family_facts(views: list[SuiteView]) -> dict[str, Any]:
    maintained = [v for v in views if (v.profile, v.suite_id) in MAINTAINED]
    docs = [v.document for v in maintained if v.document]
    hardware = next((hardware_name(doc) for doc in docs if hardware_name(doc)), None)
    timestamps = [str(doc.get("timestamp")) for doc in docs if doc.get("timestamp")]
    commit = next((str(doc["tenferro_commit"]) for doc in docs if doc.get("tenferro_commit")), None)
    return {
        "profiles": ordered_profiles({view.profile for view in views}),
        "hardware": hardware,
        "timestamp": max(timestamps) if timestamps else None,
        "commit": commit,
        "throughput": sum(len(view.regime_cases(THROUGHPUT)) for view in maintained),
        "latency": sum(len(view.regime_cases(LATENCY)) for view in maintained),
    }


def render_top(views: list[SuiteView]) -> str:
    body = [
        "<h1>tenferro-benchmark results</h1>",
        '<p class="muted">Benchmark results by hardware family. Each family page shows one '
        "bar per case against the baseline; suites, absolute times and raw reports are one "
        "click further.</p>",
        '<div class="family-grid">',
    ]
    for family in FAMILY_ORDER:
        family_views = [view for view in views if family_of(view.profile) == family]
        if not family_views:
            continue
        facts = family_facts(family_views)
        pieces = [f"<h2>{esc(FAMILY_LABEL[family])}</h2>", f'<p>{esc(FAMILY_BLURB[family])}</p>']
        if facts["hardware"]:
            pieces.append(f'<p class="facts">{esc(facts["hardware"])}</p>')
        counts = f'{facts["throughput"]} compute cases · {facts["latency"]} latency cases'
        pieces.append(f'<p class="facts">{counts}</p>')
        if facts["timestamp"]:
            pieces.append(
                f'<p class="facts">updated {esc(facts["timestamp"][:19].replace("T", " "))}Z'
                + (f' · tenferro <code>{esc(facts["commit"][:12])}</code>' if facts["commit"] else "")
                + "</p>"
            )
        body.append(
            f'<a class="family-card" href="{family}/index.html">' + "".join(pieces) + "</a>"
        )
    body.append("</div>")
    body.append(
        '<footer>Generated by <code>scripts/generate_dashboard.py</code> from the '
        '<a href="https://github.com/tensor4all/tenferro-benchmark">tensor4all/tenferro-benchmark</a> '
        "repository. Summary schema: <code>summary-schema-v1</code>.</footer>"
    )
    return page("tenferro-benchmark results", "\n".join(body))


def profile_header_from_views(profile: str, views: list[SuiteView]) -> str:
    documents = [view.document for view in views if view.document]
    hardware = next((hardware_name(doc) for doc in documents if hardware_name(doc)), None)
    timestamps = [str(doc.get("timestamp")) for doc in documents if doc.get("timestamp")]
    commit = next((str(doc["tenferro_commit"]) for doc in documents if doc.get("tenferro_commit")), None)
    pieces = [f"<strong>{esc(profile_label(profile))}</strong>", f"<span>{esc(profile)}</span>"]
    if hardware:
        pieces.append(f"<span>hardware: {esc(hardware)}</span>")
    if timestamps:
        pieces.append(f"<span>date: {esc(max(timestamps)[:19].replace('T', ' '))}Z</span>")
    if commit:
        pieces.append(f"<span>tenferro: <code>{esc(commit[:12])}</code></span>")
    if not documents:
        pieces.append('<span class="muted">no machine-readable summary</span>')
    return '<div class="profile-head">' + "".join(pieces) + "</div>"


def render_family(
    family: str,
    views: list[SuiteView],
    previous_by_key: dict[tuple[str, str], list[dict[str, Any]] | None],
    reference_map: dict[str, list[str]],
    regime: str,
) -> str:
    label = FAMILY_LABEL[family]
    body = [
        '<p><a href="../index.html">&larr; all results</a></p>',
        f"<h1>{esc(label)} {'latency results' if regime == LATENCY else 'benchmark results'}</h1>",
    ]
    if regime == LATENCY:
        body.append(
            '<p class="notice"><strong>Latency and per-call overhead.</strong> These cases are '
            "dominated by launch, dispatch and synchronization, not by device compute. "
            "Read them as API overhead, never as GPU/CPU throughput.</p>"
        )
    else:
        body.append(how_to_read())
        body.append(family_notice(family))
    for profile in ordered_profiles({view.profile for view in views}):
        profile_views = [view for view in views if view.profile == profile]
        body.append('<section class="profile-card">')
        body.append(f'<h2>{esc(profile_label(profile))}</h2>')
        body.append(profile_header_from_views(profile, profile_views))
        if profile in GITHUB_HOSTED_PROFILES:
            body.append(f'<p class="notice">{REPORT_QUALITY_NOTICE}</p>')
        any_suite = False
        for view in sorted(profile_views, key=lambda item: item.suite_id):
            if (view.profile, view.suite_id) not in MAINTAINED:
                continue
            cases = view.regime_cases(regime)
            if not cases:
                continue
            any_suite = True
            document = view.document or {}
            grouped = case_rows(document)
            all_items = [case_ratio_row(case, view.suite_id, reference_map, grouped) for case in cases]
            all_items.sort(key=lambda item: max((c["ratio"] for c in item["contenders"]), default=0.0), reverse=True)
            shown = all_items[:OVERVIEW_CASE_LIMIT]
            items = shown
            hidden = len(all_items) - len(shown)
            baseline = suite_baseline_label(cases, view.suite_id, reference_map, grouped)
            flagged = suite_flags(document, previous_by_key.get((view.profile, view.suite_id))) if document else {}
            flagged_count = sum(1 for (case_id, group), _ in flagged.items() if (case_id, group) in {(c.case, c.group) for c in cases})
            suffix = f" · {flagged_count} flagged" if flagged_count else ""
            suite_dir = view.suite_id.split("/", 1)[0]
            suite_href = f"../{view.profile}/{suite_dir}/{view.report.stem}.html"
            body.append('<details class="suite-block" open>')
            body.append(
                f'<summary><code>{esc(view.suite_id)}</code> '
                f'<span class="muted">{len(cases)} case{"s" if len(cases) != 1 else ""}{suffix}</span></summary>'
            )
            note = f"baseline {esc(baseline)} · worst first"
            if hidden:
                note += f" (showing {len(shown)} of {len(cases)})"
            note += f' · <a href="{esc(suite_href)}">case table, absolute times and raw report &rarr;</a>'
            body.append(f'<p class="chart-legend">{note}</p>')
            body.append(svg_ratio_chart(items, baseline))
            body.append("</details>")
        if not any_suite:
            body.append(f'<p class="muted">No {esc(REGIME_LABEL[regime].lower())} cases.</p>')
        if regime == THROUGHPUT:
            report_only = [
                view
                for view in profile_views
                if (view.profile, view.suite_id) in MAINTAINED and view.document is None
            ]
            if report_only:
                links = " · ".join(
                    f'<a href="../{esc(view.profile)}/{esc(view.suite_id.split("/", 1)[0])}/{esc(view.report.stem)}.html">'
                    f"<code>{esc(view.suite_id)}</code></a>"
                    for view in sorted(report_only, key=lambda item: item.suite_id)
                )
                body.append(f'<p class="chart-legend">Report only (no machine-readable summary): {links}</p>')
        body.append("</section>")
    prefix = "../../" if regime == LATENCY else "../"
    body.append(render_diagnostics(views, prefix))
    body.append(
        '<footer>Generated by <code>scripts/generate_dashboard.py</code>. '
        "Suites and raw reports are linked from each block.</footer>"
    )
    title = f"{label} benchmark results"
    if regime == LATENCY:
        title = f"{label} latency results"
    return page(title, "\n".join(body))


def render_case_table(
    document: dict[str, Any],
    suite_id: str,
    previous_rows: list[dict[str, Any]] | None,
    reference_map: dict[str, list[str]],
    metadata: dict[str, dict[str, Any]],
    regime: str,
) -> str:
    flags = suite_flags(document, previous_rows)
    neutral = not_measured_cases(document)
    grouped = {
        key: rows
        for key, rows in grouped_workloads(document).items()
        if case_regime(metadata, suite_id, key[0]) == regime
    }
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
    if not rows:
        return ""
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>case</th><th>group</th>"
        + "".join(f"<th>{esc(rule.column)}</th>" for rule in rules)
        + "<th>not measured</th><th>flags</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


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
    view: SuiteView,
    previous_rows: list[dict[str, Any]] | None,
    reference_map: dict[str, list[str]],
    metadata: dict[str, dict[str, Any]],
    markdown_html: str,
) -> str:
    profile = view.profile
    suite_id = view.suite_id
    document = view.document
    family = family_of(profile)
    body = [f'<p><a href="../../{family}/index.html">&larr; {esc(FAMILY_LABEL[family])} results</a></p>', f"<h1>{esc(suite_id)}</h1>"]
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
        unit = document.get("unit", "ms")
        grouped = case_rows(document)
        for regime in REGIMES:
            cases = view.regime_cases(regime)
            if not cases:
                continue
            title = "Compute-bound cases" if regime == THROUGHPUT else "Latency and per-call overhead"
            body.append(f"<h2>{esc(title)} ({len(cases)})</h2>")
            if regime == LATENCY:
                body.append(
                    '<p class="chart-legend">Dominated by launch, dispatch and synchronization; '
                    "these numbers are overhead, not device throughput.</p>"
                )
            else:
                body.append(
                    '<p class="chart-legend">Absolute medians on a log time axis. Bars are shorter '
                    "for faster calls; the dashed line is the best reference for each case.</p>"
                )
            body.append(render_case_table(document, suite_id, previous_rows, reference_map, metadata, regime))
            for case in cases:
                title_text = f"<code>{esc(case.case)}</code>"
                if case.group:
                    title_text += f' <span class="muted">{esc(case.group)}</span>'
                runs = f" · n={case.runs}" if case.runs else ""
                body.append(f"<h3>{title_text}<span class=\"muted\">{runs}</span></h3>")
                baseline_label, baseline_row = case_baseline(case, suite_id, reference_map, grouped)
                body.append(
                    svg_case_chart(case.rows, unit, case.runs, baseline_row, baseline_label or "best reference")
                )
        if any(rule.per_case_reference for rule in ratio_rules_for(suite_id)):
            body.append(render_perf_reference_table(document, reference_map))
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
    benchmarks_dir: Path = Path("benchmarks"),
) -> int:
    reports = discover_reports(result_dir)
    if not reports:
        print(f"error: no reports found under {result_dir}", file=sys.stderr)
        return 1
    summaries = load_summaries(result_dir)
    metadata = load_suite_metadata(benchmarks_dir)
    reference_map = load_perf_reference_map(instances_dir / GPU_PERF_ISSUES_INSTANCES.name)
    previous_by_key: dict[tuple[str, str], list[dict[str, Any]] | None] = {}
    for profile, suite_id, _report in reports:
        if (profile, suite_id) in MAINTAINED:
            _hardware, previous = previous_rows_for(
                profile, suite_id, summaries.get((profile, suite_id)) or {}, baseline_dir, git_repo, git_result_dir
            )
            previous_by_key[(profile, suite_id)] = previous
    views = build_suite_views(reports, summaries, metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    for view in views:
        target = output_dir / view.profile / view.suite_id.split("/", 1)[0]
        target.mkdir(parents=True, exist_ok=True)
        page_html = render_suite_page(
            view,
            previous_by_key.get((view.profile, view.suite_id)),
            reference_map,
            metadata,
            render_markdown(view.report.read_text(encoding="utf-8")),
        )
        (target / f"{view.report.stem}.html").write_text(page_html, encoding="utf-8")
        shutil.copy2(view.report, target / view.report.name)

    for family in FAMILY_ORDER:
        family_views = [view for view in views if family_of(view.profile) == family]
        if not family_views:
            continue
        family_dir = output_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        (family_dir / "index.html").write_text(
            render_family(family, family_views, previous_by_key, reference_map, THROUGHPUT),
            encoding="utf-8",
        )
        latency_dir = family_dir / "latency"
        latency_dir.mkdir(parents=True, exist_ok=True)
        (latency_dir / "index.html").write_text(
            render_family(family, family_views, previous_by_key, reference_map, LATENCY),
            encoding="utf-8",
        )
    (output_dir / "index.html").write_text(render_top(views), encoding="utf-8")
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
    parser.add_argument("--benchmarks-dir", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()
    return generate(
        args.result_dir,
        args.output_dir,
        args.baseline_dir,
        args.git_repo if args.baseline_dir is None else None,
        args.git_result_dir,
        args.instances_dir,
        args.benchmarks_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
