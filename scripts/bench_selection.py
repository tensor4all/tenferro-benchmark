#!/usr/bin/env python3
"""Coverage choice (quick/full) and measurement effort (scan/standard/confirm).

The two are independent (tenferro-rs #1946 B5, tenferro-benchmark #107):

* coverage selects *which* case IDs run, from a versioned manifest;
* effort selects *how hard* each selected case is measured.

``scan`` is a broad low-repetition screen: it flags suspects and never makes a
performance claim. ``standard`` is the suite's ordinary repetition count used
for routine reports. ``confirm`` takes its repetitions from a predeclared
confirmation config (``benchmarks/cpu/confirmation.yaml``) and refuses to run
while any required value is still a placeholder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
COVERAGES = ("quick", "full")
EFFORTS = ("scan", "standard", "confirm")
SCAN_DEFAULT = {"warmups": 1, "runs": 3}
CONFIRMATION_CONFIG = ROOT / "benchmarks/cpu/confirmation.yaml"

# Every value here must be declared (non-null) before a confirmation run and
# before any candidate result is looked at.
REQUIRED_CONFIRMATION_FIELDS = (
    "declared_before_candidate_results",
    "library.baseline_commit",
    "library.candidate_commit",
    "harness_commit",
    "build.profile",
    "build.features",
    "host.target_profile",
    "host.hostname",
    "host.affinity",
    "host.provider",
    "threads",
    "cases.suite_id",
    "cases.manifest_version",
    "cases.coverage",
    "timing.scope",
    "timing.cache_pool_state",
    "repetitions.warmups",
    "repetitions.runs",
    "repetitions.rounds",
    "statistic",
    "thresholds.relative",
    "thresholds.absolute_ns",
    "noise.max_cov",
    "noise.max_aa_relative_spread",
    "noise.host_idle_guard",
)
STATISTICS = ("median_of_round_ratios",)


class SelectionError(ValueError):
    pass


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(path.read_text())
    for field in ("manifest_version", "suite_id", "coverage"):
        if field not in manifest:
            raise SelectionError(f"{path}: manifest lacks {field}")
    quick, full = manifest["coverage"]["quick"], manifest["coverage"]["full"]
    if not set(quick) <= set(full):
        raise SelectionError(f"{path}: quick must be a subset of full")
    return manifest


def resolve_coverage(value: str | None = None) -> str:
    coverage = value or os.environ.get("BENCH_COVERAGE") or "quick"
    if coverage not in COVERAGES:
        raise SelectionError(f"BENCH_COVERAGE must be one of {COVERAGES}, got {coverage!r}")
    return coverage


def select(manifest: dict[str, Any], coverage: str,
           instance_filter: str | None = None) -> tuple[list[str], list[str], str | None]:
    """Return (expected, selected, filter) case IDs for a coverage choice."""
    expected = list(manifest["coverage"][coverage])
    raw = instance_filter if instance_filter is not None else os.environ.get("BENCH_INSTANCE", "")
    if not raw:
        return expected, expected, None
    requested = [x for x in raw.split(",") if x]
    unknown = sorted(set(requested) - set(manifest["coverage"]["full"]))
    if unknown:
        raise SelectionError(f"unknown case IDs: {unknown}")
    outside = sorted(set(requested) - set(expected))
    if outside:
        raise SelectionError(f"case IDs outside the {coverage} coverage: {outside}; "
                             "use BENCH_COVERAGE=full")
    return expected, [i for i in expected if i in set(requested)], raw


def _lookup(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def missing_confirmation_fields(config: dict[str, Any]) -> list[str]:
    missing = [f for f in REQUIRED_CONFIRMATION_FIELDS if _lookup(config, f) in (None, "", [])]
    if config.get("status") != "declared":
        missing.insert(0, "status (must be 'declared')")
    statistic = config.get("statistic")
    if statistic is not None and statistic not in STATISTICS:
        missing.append(f"statistic (unknown {statistic!r}; one of {STATISTICS})")
    return missing


def load_confirmation_config(path: Path | None = None) -> dict[str, Any]:
    path = path or Path(os.environ.get("BENCH_CONFIRM_CONFIG") or CONFIRMATION_CONFIG)
    config = yaml.safe_load(Path(path).read_text())
    missing = missing_confirmation_fields(config)
    if missing:
        raise SelectionError(
            f"{path}: confirmation config is not declared; fill these before any candidate "
            f"run: {', '.join(missing)}")
    return config


def resolve_effort(run_defaults: dict[str, Any], effort_defaults: dict[str, Any] | None = None,
                   value: str | None = None) -> dict[str, Any]:
    """Return {'effort', 'warmups', 'runs', ...} for the requested effort."""
    effort = value or os.environ.get("BENCH_EFFORT") or "standard"
    if effort not in EFFORTS:
        raise SelectionError(f"BENCH_EFFORT must be one of {EFFORTS}, got {effort!r}")
    if effort == "standard":
        return {"effort": effort, "warmups": run_defaults["warmups"], "runs": run_defaults["runs"]}
    if effort == "scan":
        scan = dict(SCAN_DEFAULT, **((effort_defaults or {}).get("scan") or {}))
        return {"effort": effort, "warmups": scan["warmups"], "runs": scan["runs"]}
    config = load_confirmation_config()
    reps = config["repetitions"]
    return {"effort": effort, "warmups": reps["warmups"], "runs": reps["runs"],
            "confirmation_config": str(os.environ.get("BENCH_CONFIRM_CONFIG")
                                       or CONFIRMATION_CONFIG.relative_to(ROOT))}
