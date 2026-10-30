#!/usr/bin/env python3
"""Run and report the cpu/overhead_smoke suite (tenferro-benchmark#197).

Each row is one ``perf_issue_case`` process on one run thread: a pooled
allocator pass (allocs/op, bytes/op) followed by the standard sampling scheme
(warmups, calibration toward ``min_runtime_ms`` per batch, measured batches of
many operations whose outputs are retained until the clock stops). Rows are a
trend view only and never gate; ``--report`` writes the run report and the
latest ``result/<target_profile>/cpu/overhead_smoke.md``.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

import yaml
import report_summary

from benchmark_layout import safe_target_profile
from generate_overhead_smoke_cases import case_timing_scope, STEADY_SCOPE, ENTRY_SCOPE

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/cpu/overhead_smoke.yaml"
SUITE_ID = "cpu/overhead_smoke"


def load():
    suite = yaml.safe_load(SUITE.read_text())
    cases = {
        c["id"]: c for c in json.loads((ROOT / suite["problems"]["source"]).read_text())
    }
    selected = suite["problems"]["include"]
    raw = os.environ.get("BENCH_INSTANCE", "")
    if raw:
        wanted = {cid.strip() for cid in raw.split(",") if cid.strip()}
        unknown = wanted - set(selected)
        if unknown:
            raise SystemExit(f"unknown BENCH_INSTANCE ids: {sorted(unknown)}")
        selected = [cid for cid in selected if cid in wanted]
    return suite, cases, selected


def collect(binary, output, threads):
    suite, cases, selected = load()
    run = suite["defaults"]["run"]
    failed = 0
    with output.open("w") as out:
        for cid in selected:
            case = cases[cid]
            row = {
                "suite_id": SUITE_ID,
                "case_id": cid,
                "threads": threads,
                **{
                    k: case[k]
                    for k in ("refs", "fixture", "backend_thread_budget", "note")
                },
                "counter_case": case.get("counter_case"),
                "timing_scope": case_timing_scope(case),
            }
            command = [
                str(binary),
                "--case",
                cid,
                "--params",
                json.dumps(case["params"]),
                "--threads",
                str(threads),
                "--device",
                "0",
                "--warmups",
                str(run["warmups"]),
                "--samples",
                str(run["runs"]),
                "--target-ns",
                str(int(run["min_runtime_ms"] * 1_000_000)),
            ]
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                failed = 1
                row.update(correctness_status="failed", error=result.stderr.strip())
            else:
                measured = json.loads(result.stdout.strip().splitlines()[-1])
                measured.pop("suite_id", None)
                measured.pop("case_id", None)
                row.update(measured)
            out.write(json.dumps(row) + "\n")
            print(f"{cid}: {row['correctness_status']}")
    return failed


def build_summary(records, metadata, *, run_dir=None, cpu_model=None):
    """Rebuild ns/op and allocator trends from original retained-output samples."""
    rows = []
    for record in records:
        values = []
        for sample in record.get("samples", []):
            elapsed, count = sample["elapsed_ns"], sample["iterations"]
            if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                    or elapsed < 0 or type(count) is not int or count <= 0):
                raise ValueError("Invalid overhead batch sample")
            values.append(elapsed / count)
        verified = record.get("correctness_status") == "passed"
        ok = verified and bool(values)
        quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [0]*3
        scope = case_timing_scope(record)
        row = report_summary.make_row(
            record["case_id"], "tenferro-rs", statistics.median(values) if ok else None,
            iqr=quartiles[2]-quartiles[0] if ok else None,
            status="ok" if ok else "failed", verified=verified,
            group=f"threads={record['threads']} / scope={scope}",
            reason=record.get("error"))
        row.update(timing_scope=scope, backend_thread_budget=record["backend_thread_budget"])
        counters = record.get("counters") or [{}]
        for key in ("host_allocations_per_op", "host_allocated_bytes_per_op"):
            value = counters[0].get(key)
            if ok and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError("Invalid overhead allocation counter")
            row[key] = value
        rows.append(row)
    header = report_summary.metadata_from_run_metadata(metadata)
    if cpu_model:
        header["hardware"]["cpu"] = cpu_model
    header["hardware"]["threads"] = sorted({record["threads"] for record in records})
    return report_summary.build_document(
        rows,
        **header,
        unit="ns",
        **(report_summary.v2_provenance(run_dir, metadata) if run_dir is not None else {}),
    )


def summary(row):
    values = [s["elapsed_ns"] / s["iterations"] for s in row.get("samples", [])]
    if row.get("correctness_status") != "passed" or not values:
        return "—", "—", "—"
    median = statistics.median(values)
    q = (
        statistics.quantiles(values, n=4, method="inclusive")
        if len(values) > 1
        else [median] * 3
    )
    cov = statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0
    return (
        f"{median:.1f}",
        f"{q[2] - q[0]:.1f}",
        f"{cov * 100:.1f}%" + (" NOISY" if cov > 0.10 else ""),
    )


def report_table(rows):
    lines = [
        "| Case | Workload | Budget | Median ns/op | IQR ns | CoV | allocs/op | bytes/op | Refs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        counters = row.get("counters") or [{}]
        median, iqr, cov = summary(row)
        ok = row.get("correctness_status") == "passed"
        allocs = counters[0].get("host_allocations_per_op")
        allocated = counters[0].get("host_allocated_bytes_per_op")
        label = row["case_id"] if ok else f"{row['case_id']} (FAILED)"
        if case_timing_scope(row) == ENTRY_SCOPE and not row["case_id"].endswith(
            "_session_entry_diagnostic"
        ):
            label += " (session-entry diagnostic; legacy raw ID)"
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    row["note"],
                    str(row["backend_thread_budget"]),
                    median,
                    iqr,
                    cov,
                    "—" if allocs is None else f"{allocs:.2f}",
                    "—" if allocated is None else f"{allocated:.0f}",
                    " ".join(row["refs"]),
                ]
            )
            + " |"
        )
    return lines


def render_report(run_dir, target):
    """Report text and summary built only from files recorded in the run directory."""
    run_dir = run_dir.resolve()
    rows = [
        json.loads(line)
        for path in sorted(run_dir.glob("samples_t*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    tenferro = {}
    lines = [
        "# CPU Overhead Smoke (ns/op and allocs/op)",
        "",
        f"- Suite: `{SUITE_ID}`",
        f"- Target profile: `{target}`",
        f"- Raw run: `{report_summary.repo_relative(run_dir)}`",
    ]
    metadata = {}
    run_yaml = run_dir / "run.yaml"
    if run_yaml.exists():
        metadata = yaml.safe_load(run_yaml.read_text())
        tenferro = metadata.get("tenferro_rs", {})
        lines += [
            f"- tenferro-rs commit: `{tenferro.get('commit')}`",
            f"- Collection command: `{metadata.get('collection', {}).get('command', '')}`",
        ]
    lines += [
        "",
        "Trend view for tenferro-rs #2178 (tenferro-benchmark #197); nothing here gates. "
        "Every row runs on one run thread; the prepared-runtime diagnostics set the CPU "
        "backend thread budget explicitly. **allocs/op** counts process-wide Rust allocator "
        "events over 64 calls after pool warmup, dropping outputs between calls. These "
        "observed counts include Rust workers and exclude foreign BLAS/C allocations; "
        "compare matching fixtures, thread budgets and execution scopes. **ns/op** is the "
        "median of retained-output batches, with destruction after the clock stops. "
        "Shared nightly runner timings are reference indicators only.",
        "",
    ]
    steady = [row for row in rows if case_timing_scope(row) == STEADY_SCOPE]
    diagnostics = [row for row in rows if case_timing_scope(row) == ENTRY_SCOPE]
    lines += [
        "## Steady-state operations",
        "",
        "Inputs, entered reusable sessions, initialization, tracing and preparation are "
        "outside timing. Only the declared operation and intrinsic output allocation "
        "or pooled checkout are timed.",
        "",
    ]
    lines += report_table(steady)
    if diagnostics:
        lines += [
            "",
            "## Runtime session-entry diagnostics",
            "",
            "These rows time public `Runtime::run_prepared`, including its internal session "
            "entry and exit. Runtime construction, inputs, tracing and preparation remain "
            "outside timing. They are **diagnostic public-path latency**, not shared-session "
            "operation-performance evidence. Their allocation counters also include internal "
            "session entry. They are intentionally separate from the steady-state table.",
            "",
        ]
        if any(
            not row['case_id'].endswith('_session_entry_diagnostic')
            for row in diagnostics
        ):
            lines += [
                "This raw run predates explicit diagnostic case IDs and per-case timing metadata. "
                "Its `run_prepared` rows are reclassified here from their recorded fixtures; "
                "the original raw IDs, samples and metadata are unchanged. The old blanket "
                "claim that session entry was outside timing does not apply to these rows.",
                "",
            ]
        lines += report_table(diagnostics)
    passed = sum(row.get("correctness_status") == "passed" for row in rows)
    lines += [
        "",
        f"Numerical checks: **{passed}/{len(rows)} rows passed** against independent host "
        "loops before counting and sampling.",
        "",
    ]
    fused = [r for r in rows if r.get("fused_regions")]
    if fused:
        lines += [
            "## Prepared-runtime region execution (diagnostics)",
            "",
            "Whether the runtime executed the traced region as one fused command or declined "
            "to per-op dispatch (cumulative over validation, the allocator pass and timed "
            "batches). `run_prepared` enters a runtime-internal session per call, and its "
            "entry is included in both diagnostic ns/op and allocs/op.",
            "",
            "| Case | Planned regions | Fused executions | Fallbacks |",
            "|---|---:|---:|---:|",
        ]
        for row in fused:
            for info in row["fused_regions"].values():
                counts = info["elementwise_region_execution_counts"]
                lines.append(
                    f"| {row['case_id']} | {info['elementwise_region_summary']['regions']} | "
                    f"{counts['fused']} | {counts['fallbacks']} |"
                )
        lines.append("")
    failures = [r for r in rows if r.get("correctness_status") != "passed"]
    if failures:
        lines += ["## Failures", ""]
        lines += [
            f"- `{r['case_id']}`: {r.get('error', 'unknown error')}" for r in failures
        ]
        lines.append("")
    if (run_dir / "cpu_info.md").exists():
        lines += [(run_dir / "cpu_info.md").read_text().rstrip(), ""]
    if run_yaml.exists():
        conditions = {
            k: metadata[k]
            for k in ("timestamp", "tenferro_rs", "blas", "environment", "collection")
            if k in metadata
        }
        lines += [
            "## Run metadata",
            "",
            f"Full metadata: `{report_summary.repo_relative(run_yaml)}`",
            "",
            "```yaml",
            yaml.safe_dump(conditions, sort_keys=False).rstrip(),
            "```",
            "",
        ]
    document = build_summary(
        rows,
        metadata,
        run_dir=run_dir,
        cpu_model=report_summary.cpu_model_from_run_dir(run_dir),
    )
    return "\n".join(lines), document


def report(run_dir, target):
    safe_target_profile(target)
    run_dir = Path(run_dir).resolve()
    text, document = render_report(run_dir, target)
    (run_dir / "report.md").write_text(text)
    report_summary.write_document(run_dir / "report.summary.json", document)
    report_summary.write_document(run_dir / f"{Path(SUITE_ID).name}.summary.json", document)
    latest = ROOT / "result" / safe_target_profile(target) / "cpu" / "overhead_smoke.md"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(text)
    report_summary.write_document(latest.with_suffix(".summary.json"), document)
    print(latest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--target-profile", default="amd-cpu")
    args = parser.parse_args()
    if args.report:
        report(args.report, args.target_profile)
    else:
        raise SystemExit(collect(args.binary, args.output, args.threads))
