#!/usr/bin/env python3
"""Trend points for the latest-generation CPU results, in the toolkit's exchange format.

    uv run python scripts/trend_points.py --out /tmp/points.json
    uv run python scripts/trend_points.py --out _site/trend/points.json --render _site/trend

A point is one median of one case, engine and thread budget in one recorded run,
with the conditions it was measured under (``campaign_toolkit``'s
``schemas/trend-points.schema.json``).  The points come from the
``result/zen5-cpu/cpu/*.summary.json`` sidecars in git history: every
publication commit that changed a summary sidecar is read, and a point is kept
once per measurement identity ``(source run, case, engine, threads)``, so a
commit that only carries a summary over adds nothing.  The newest point wins
when the same measurement is described twice (a v1 summary rewritten as v2 over
the same source run).

A summary schema v1 records no provenance beyond the revision, so a v1 point's
power state is ``unrecorded`` and its timing policy is read from the source
``run.yaml`` at that commit when that file is tracked, else ``unrecorded``.  A
v1 point's ``spread`` is 0: one process start has no between-start spread, and
the within-process IQR that the page also prints is a different quantity that
must not be relabelled as one.  The vendor arm is the BLAS implementation the
source run's ``run.yaml`` records, or null when no such record is tracked.

``--render`` draws the chart with the toolkit's trend renderer
(``campaign_toolkit.trend.write``); the toolkit CLI is not used because it takes
its selections from a campaign descriptor, which this repository does not have.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Iterator

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_history  # noqa: E402

from campaign_toolkit import jsonschema_lite, trend  # noqa: E402
from campaign_toolkit.descriptor import schemas_dir  # noqa: E402

CAMPAIGN = "tenferro-benchmark"
PROFILE = "zen5-cpu"
SCHEMA = "trend-points.schema.json"
SUMMARY_RE = re.compile(r"^result/(?P<profile>[^/]+)/cpu/(?P<suite>[^/]+)\.summary\.json$")
THREADS_IN_GROUP = re.compile(r"^(?:Threads|threads)[:=]\s*\d+$")
THREADS_VALUE = re.compile(r"(?:Threads|threads)[:=]\s*(\d+)")
RUN_DIR = re.compile(r"\d{8}_\d{6}$")


@dataclass(frozen=True)
class Selection:
    """One representative case: the case key the page names, its engine rows, its thread rows."""

    suite: str
    case_key: str
    engines: tuple[str, ...]
    threads: tuple[int, ...]


# The representative cases (D4), one per suite.  The case key is the case as the
# page names it: the case id plus every declared axis of its report row
# (strategy for `cpu/einsum`, dtype and shape for `cpu/cpu_ops`, the timing scope
# for `cpu/overhead_smoke`), with the thread count stripped because `threads` is
# its own field.  `cpu/overhead_smoke` runs one run thread
# (`scripts/run_overhead_smoke.sh` sets a single thread count), so it is selected
# at 1T only; its `run_prepared` session-entry rows are diagnostics, not the
# caller-held-engine steady state, and are not selected.
SELECTIONS: tuple[Selection, ...] = (
    Selection(
        suite="cpu/einsum",
        case_key="bin_batched_matmul_b32_m128_n128_k128 (opt_flops)",
        engines=("tenferro-eager", "tenferro-trace", "pytorch-cpu"),
        threads=(1, 8),
    ),
    Selection(
        suite="cpu/cpu_ops",
        case_key="matmul (suite=large / dtype=f64 / shape=1024x1024)",
        engines=("tenferro-eager", "tenferro-trace", "pytorch-cpu"),
        threads=(1, 8),
    ),
    Selection(
        suite="cpu/overhead_smoke",
        case_key="typed_elemwise_f64_add_bcast_n16 (scope=steady_state_host_api)",
        engines=("tenferro-rs",),
        threads=(1,),
    ),
)
SELECTIONS_BY_SUITE = {selection.suite: selection for selection in SELECTIONS}


def schema() -> dict[str, Any]:
    return json.loads((schemas_dir() / SCHEMA).read_text(encoding="utf-8"))


def case_key(row: dict[str, Any]) -> str:
    """The case as its report names it: the case id and its declared axes, without the thread count."""
    case = str(row.get("case", ""))
    parts = [part.strip() for part in str(row.get("group") or "").split("/")]
    kept = [part for part in parts if part and not THREADS_IN_GROUP.match(part)]
    return f"{case} ({' / '.join(kept)})" if kept else case


def row_threads(row: dict[str, Any], document: dict[str, Any]) -> int | None:
    """The run thread count of a summary row, from its group or a single recorded thread row."""
    match = THREADS_VALUE.search(str(row.get("group") or ""))
    if match:
        return int(match.group(1))
    threads = (document.get("hardware") or {}).get("threads") or []
    return int(threads[0]) if len(threads) == 1 else None


def timing_policy_of(document: dict[str, Any]) -> str | None:
    policy = document.get("timing_policy")
    if isinstance(policy, dict) and policy.get("version"):
        return f"v{policy['version']}"
    return None


def source_for_thread(sources: list[dict[str, Any]], threads: int) -> dict[str, Any] | None:
    """The recorded source run a row belongs to, when the summary attributes it to one."""
    for source in sources:
        if threads in [int(value) for value in source.get("threads") or []]:
            return source
    return sources[0] if len(sources) == 1 else None


def summaries_at(repo: pathlib.Path, sha: str, profile: str) -> dict[str, str]:
    """{repo-relative summary path: blob id} of one profile's summary sidecars at a commit."""
    records: dict[str, str] = {}
    out = generate_history.git(repo, "ls-tree", "-r", "-z", sha, "--", f"result/{profile}/cpu/")
    for entry in out.split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        if SUMMARY_RE.match(path):
            records[path] = meta.split()[2]
    return records


def summary_commits(repo: pathlib.Path, profile: str) -> list[tuple[str, list[str]]]:
    """[(sha, changed summary paths)] newest first: only sidecars a commit added or rewrote."""
    commits = generate_history.publication_commits(repo, (profile,))
    states = [summaries_at(repo, sha, profile) for sha, _, _ in commits]
    out: list[tuple[str, list[str]]] = []
    for index, (sha, _epoch, _subject) in enumerate(commits):
        previous = states[index + 1] if index + 1 < len(commits) else {}
        changed = sorted(path for path, blob in states[index].items() if previous.get(path) != blob)
        if changed:
            out.append((sha, changed))
    return out


@lru_cache(maxsize=None)
def run_metadata(repo: str, sha: str, run: str) -> dict[str, Any] | None:
    """The source run's recorded ``run.yaml`` at a commit, when the record is tracked there."""
    import yaml

    text = generate_history.git(pathlib.Path(repo), "show", f"{sha}:{run}/run.yaml")
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return document if isinstance(document, dict) else None


@lru_cache(maxsize=None)
def source_runs_of_summary(repo: str, sha: str, path: str) -> dict[int | None, str]:
    """The run directories the sibling report names at a commit, by thread row.

    A combined CPU report has one ``## Threads: N`` section per thread row and names
    the run behind each of them; a single-run report names one run, which is the
    default entry (``None``).  Summary schema v1 records no ``source_runs``, so this
    is where its points get a run identity.
    """
    report = path[: -len(".summary.json")] + ".md"
    text = generate_history.git(pathlib.Path(repo), "show", f"{sha}:{report}")
    found: dict[int | None, str] = {}
    for section in re.split(r"^## Threads:\s*", text, flags=re.MULTILINE)[1:]:
        heading, _, body = section.partition("\n")
        try:
            threads = int(heading.strip())
        except ValueError:
            continue
        run = run_dir_in(body)
        if run is not None:
            found[threads] = run
    if not found:
        run = run_dir_in(text)
        if run is not None:
            found[None] = run
    return found


def run_dir_in(text: str) -> str | None:
    """The first recorded run directory a report fragment names."""
    for candidate in re.findall(r"data/results/[^\s`)]+", text):
        parts = candidate.rstrip(".,;").split("/")
        for index, part in enumerate(parts):
            if index >= 2 and RUN_DIR.fullmatch(part):
                return "/".join(parts[: index + 1])
    return None


def points_of_document(
    document: dict[str, Any], path: str, sha: str, repo: pathlib.Path
) -> Iterator[dict[str, Any]]:
    """Every selected trend point of one summary sidecar."""
    match = SUMMARY_RE.match(path)
    if match is None:
        return
    # The sidecar's path is the published suite: one run directory holds several
    # reports (einsum, cpu_ops, linalg), each with its own suite header.
    suite = f"cpu/{match['suite']}"
    profile = str(document.get("target_profile") or "")
    selection = SELECTIONS_BY_SUITE.get(suite)
    commit = str(document.get("tenferro_commit") or "")
    if selection is None or not profile or not commit:
        return
    sources = [source for source in document.get("source_runs") or [] if isinstance(source, dict)]
    report_runs = source_runs_of_summary(str(repo), sha, path)
    unit = str(document.get("unit") or "ms")
    for row in document.get("rows") or []:
        if row.get("status") != "ok" or row.get("median") is None:
            continue
        median = float(row["median"])
        if median <= 0:
            continue
        if case_key(row) != selection.case_key:
            continue
        threads = row_threads(row, document)
        if threads not in selection.threads:
            continue
        source = source_for_thread(sources, threads) if sources else None
        if source is not None and source.get("run"):
            run = str(source["run"])
        else:
            run = report_runs.get(threads) or report_runs.get(None) or f"{path}@{sha[:12]}"
        recorded = run_metadata(str(repo), sha, run) or {}
        policy = timing_policy_of(document)
        if policy is None:
            version = (recorded.get("timing_policy") or {}).get("version")
            policy = f"v{version}" if version else "unrecorded"
        blas = recorded.get("blas") or {}
        yield {
            "campaign": CAMPAIGN,
            "suite": suite,
            "case_key": selection.case_key,
            "engine": str(row.get("backend")),
            "threads": threads,
            "profile": profile,
            "library_version": document.get("library_version"),
            "library_commit": commit,
            "timing_policy": policy,
            # Summary schema v1 has no source_runs, so it records no power state.
            "power": str(source.get("power") or "unrecorded") if source else "unrecorded",
            "vendor": str(blas["implementation"]) if blas.get("implementation") else None,
            "median": median,
            "spread": 0.0,
            "unit": unit,
            "run_id": run,
            "timestamp": str(document.get("timestamp") or ""),
        }


def build_points(repo: pathlib.Path, profile: str = PROFILE) -> list[dict[str, Any]]:
    """The trend points of every publication commit that changed a summary sidecar."""
    points: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for sha, changed in summary_commits(repo, profile):
        for path in changed:
            text = generate_history.git(repo, "show", f"{sha}:{path}")
            try:
                document = json.loads(text)
            except ValueError:
                continue
            if not isinstance(document, dict):
                continue
            for point in points_of_document(document, path, sha, repo):
                identity = (
                    point["run_id"],
                    point["suite"],
                    point["case_key"],
                    point["engine"],
                    point["threads"],
                )
                points.setdefault(identity, point)
    return [points[identity] for identity in sorted(points)]


def trend_entries() -> list[dict[str, Any]]:
    """The chart selections, in the shape `campaign_toolkit.trend` reads."""
    return [
        {
            "suite": selection.suite,
            "case": selection.case_key,
            "engines": list(selection.engines),
            "threads": list(selection.threads),
        }
        for selection in SELECTIONS
    ]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="where to write the trend-points document")
    parser.add_argument("--render", type=pathlib.Path, default=None,
                        help="also draw the charts and their tables into this directory")
    args = parser.parse_args(argv)
    points = build_points(args.repo, args.profile)
    document = {"schema_version": 1, "points": points}
    jsonschema_lite.validate(document, schema())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(points)} points)")
    if args.render is not None:
        # `read_points` is the toolkit's own reader of the document the chart is drawn from.
        for entry, svg, _table in trend.write(trend_entries(), trend.read_points(args.out), args.render):
            drawn = len(trend.select(points, entry))
            print(f"wrote {svg} ({drawn} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
