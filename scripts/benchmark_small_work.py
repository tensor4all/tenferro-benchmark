#!/usr/bin/env python3
"""Run the saved small-work cases and format their ordinary suite results."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

import yaml

from benchmark_layout import safe_target_profile
import bench_selection
import case_status as cs
from suite_instances import resolve_suite_instance_ids

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/cpu/small_work.yaml"
MANIFEST = ROOT / "benchmarks/cpu/manifests/small_work.yaml"
SUITE_ID = "cpu/small_work"

# Route dimensions of each API tier (tenferro-rs #1946 B1). `scope` separates
# steady-state operation rows from explicitly labelled diagnostics; only the
# former are operation comparisons.
TIER_ROUTES = {
    "concrete-shared": ("concrete", "allocating", "owned", "shared-session", "steady_state"),
    "concrete-fresh": ("concrete", "allocating", "owned", "session-per-call", "session_entry_diagnostic"),
    "borrowed-shared": ("concrete", "allocating", "view", "shared-session", "steady_state"),
    "borrowed-fresh": ("concrete", "allocating", "view", "session-per-call", "session_entry_diagnostic"),
    "prepared-setup": ("concrete-prepared", "none", "owned", "shared-session", "preparation_diagnostic"),
    "prepared-repeat": ("concrete-prepared", "allocating", "owned", "shared-session", "steady_state"),
    "prepared-into-repeat": ("concrete-prepared", "reused", "owned", "shared-session", "steady_state"),
    "dot-general-into-shared": ("concrete", "reused", "owned", "shared-session", "steady_state"),
    "eager-no-ad": ("eager", "allocating", "owned", "session-per-call", "session_entry_diagnostic"),
    "eager-ad": ("eager", "allocating", "owned", "session-per-call", "session_entry_diagnostic"),
    "compiled-setup": ("compiled-trace", "none", "owned", "runtime", "preparation_diagnostic"),
    "compiled-repeat": ("compiled-trace", "allocating", "owned", "runtime-internal-session", "session_entry_diagnostic"),
}


def route_for(case):
    path, output, representation, boundary, scope = TIER_ROUTES[case["api_tier"]]
    layout = case.get("layout", "col_major_contiguous")
    return {
        "execution_path": path,
        "output": output,
        "representation": representation,
        "layout": {"col_major_contiguous": "direct", "row_major_contiguous": "direct-transposed",
                   "strided": "direct-strided", "broadcast": "broadcast"}.get(layout, layout),
        "session_boundary": boundary,
        "measurement_scope": scope,
    }


def coverage_choice():
    # BENCH_INCLUDE_SETUP_DIAGNOSTICS=1 is the older spelling of full coverage.
    if not os.environ.get("BENCH_COVERAGE") and os.environ.get("BENCH_INCLUDE_SETUP_DIAGNOSTICS") == "1":
        return "full"
    return bench_selection.resolve_coverage()


def plan():
    suite = yaml.safe_load(SUITE.read_text())
    cases = {c["id"]: c for c in json.loads((ROOT / suite["problems"]["source"]).read_text())}
    manifest = bench_selection.load_manifest(MANIFEST)
    if set(manifest["coverage"]["full"]) != set(resolve_suite_instance_ids(SUITE)):
        raise bench_selection.SelectionError("small_work manifest full list != suite include list")
    coverage = coverage_choice()
    expected, selected, raw = bench_selection.select(manifest, coverage)
    return suite, cases, manifest, coverage, expected, selected, raw


def selected_cases():
    suite, cases, _, _, _, selected, _ = plan()
    return [cases[i] for i in selected], suite["defaults"]["run"]


def collect(binary, output, threads):
    suite, all_cases, manifest, coverage, expected, selected, raw = plan()
    cases = [all_cases[i] for i in selected]
    config = dict(suite["defaults"]["run"])
    effort = bench_selection.resolve_effort(config, suite["defaults"].get("effort"))
    config.update(warmups=effort["warmups"], runs=effort["runs"])
    env = dict(os.environ)
    if coverage == "full":
        env["BENCH_INCLUDE_SETUP_DIAGNOSTICS"] = "1"
    failed = False
    rows = []
    with output.open("w") as fh:
        for case in cases:
            command = [str(binary), "--case", case["id"]]
            for key in ("operation", "dtype", "api_tier", "workflow", "layout"):
                command += ["--" + key.replace("_", "-"), str(case[key])]
            if "operand_shapes" in case:
                # Generic einsum cases carry explicit subscripts and operand shapes.
                command += ["--subscripts", case["subscripts"], "--operand-shapes",
                            ",".join("x".join(map(str, shape)) for shape in case["operand_shapes"])]
            command += ["--size", str(math.prod(case["shape"])),
                        "--calls", str(case["calls_per_workflow"]),
                        "--warmups", str(config["warmups"]), "--samples", str(config["runs"]),
                        "--target-ns", str(int(config["min_runtime_ms"] * 1_000_000))]
            result = subprocess.run(command, text=True, capture_output=True, env=env)
            row = dict(case, threads=threads, case_id=case["id"], samples=[],
                       route=route_for(case), case_key=cs.case_key(case["id"], threads),
                       effort=effort["effort"])
            if result.returncode:
                row.update(correctness_status="failed", error=result.stderr.strip())
            else:
                try:
                    row.update(json.loads(result.stdout))
                except (ValueError, TypeError) as error:
                    row.update(correctness_status="failed", error=f"invalid case output: {error}")
            failed |= row["correctness_status"] != "passed"
            row["status"] = row["correctness_status"]
            values = [s["elapsed_ns"] / s["iterations"] for s in row["samples"]]
            row["cov"] = cs.coefficient_of_variation(values) if row["status"] == "passed" else None
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
            print(f"{case['id']}: {row['correctness_status']}", flush=True)
    status = cs.build_case_status(
        suite_id=SUITE_ID, manifest_version=manifest["manifest_version"], coverage=coverage,
        effort=effort["effort"], expected=[cs.case_key(i, threads) for i in expected],
        selected=[cs.case_key(i, threads) for i in selected], rows=rows, selection_filter=raw)
    cs.write_case_status(output.parent / f"case_status_t{threads}.json", status)
    return failed or bool(status["missing"])


def summary(row):
    if row["correctness_status"] != "passed":
        return "—", "—", "—", "FAILED"
    values = [s["elapsed_ns"] / s["iterations"] for s in row["samples"]]
    if not values:
        return "—", "—", "—", "MISSING"
    median = statistics.median(values)
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [median]*3
    cov = statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0
    # A descriptive label, never a performance gate or reason to drop a row.
    status = "NOISY" if cov > 0.10 else "ok"
    return f"{median:.2f}", f"{q[2]-q[0]:.2f}", f"{cov*100:.1f}%", status


def shape_label(row):
    """Output shape; generic einsum rows also name subscripts and operand shapes."""
    shape = "×".join(map(str, row["shape"]))
    if "operand_shapes" not in row:
        return shape
    operands = ", ".join("×".join(map(str, s)) for s in row["operand_shapes"])
    return f"`{row['subscripts']}` {operands} → {shape}"


def report(run_dir, target):
    run_dir = run_dir.resolve()
    rows = [json.loads(line) for path in sorted(run_dir.glob("samples_t*.jsonl"))
            for line in path.read_text().splitlines()]
    lines = ["# CPU Small-Work Benchmark Results", "", "- Suite: `cpu/small_work`",
             f"- Raw run: `{run_dir.relative_to(ROOT)}`", ""]
    cpu_info = run_dir / "cpu_info.md"
    if cpu_info.exists():
        lines += [cpu_info.read_text().rstrip(), ""]
    notes = run_dir / "measurement_notes.md"
    if notes.exists():
        lines += [notes.read_text().rstrip(), ""]
    for path in sorted(run_dir.glob("run_t*.yaml")):
        metadata = yaml.safe_load(path.read_text())
        conditions = {k: metadata[k] for k in ("timestamp", "tenferro_rs", "blas", "environment") if k in metadata}
        lines += [f"## {path.stem}", "", f"Full metadata: `{path.relative_to(ROOT)}`", "",
                  "```yaml", yaml.safe_dump(conditions, sort_keys=False).rstrip(), "```", ""]
    parts = [json.loads(p.read_text()) for p in sorted(run_dir.glob("case_status_t*.json"))]
    if parts:
        status = cs.merge_case_status(parts)
        cs.write_case_status(run_dir / "case_status.json", status)
        lines += cs.markdown_section(status)
    passed = sum(r["correctness_status"] == "passed" for r in rows)
    lines += [f"Numerical checks: **{passed}/{len(rows)} recorded rows passed** (case × thread count).", ""]
    lines += ["## Timing", "",
              "Sequential release runs; warmups, then calibration from at least 1024 operations toward a 1 ms batch, then the measured batches of the recorded effort (standard: 3 warmups and 15 batches; scan: a low-repetition screen that only marks suspects). "
              "Each batch uses one wall-clock interval; all outputs are retained until it stops. Statistics divide batch time by iterations. "
              "Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. "
              "NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.", "",
              "Workflow time is the total for one call or the complete dependent chain. "
              "The separate ns/op column divides the chain median by its operation count, not an independently measured call. "
              "Setup rows measure preparation only, not execution.", "",
              "Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. "
              "Returned outputs/plans are retained until after each batch interval; "
              "value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.", "",
              "| Case ID | Operation | API route | Route (path / output / repr / session) | Dtype | Shape | Layout | Workflow | Threads | Provider | Operations/batch | Median batch ms | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |",
              "|---|---|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        if row.get("measurement_kind") == "setup_diagnostic" or row["api_tier"].endswith(("-fresh", "-setup")):
            continue
        median, iqr, cov, status = summary(row)
        per_op = f"{float(median)/row['calls_per_workflow']:.2f}" if median != "—" and row['calls_per_workflow'] > 1 else "—"
        route = row.get("route") or route_for(row)
        route_label = (f"{route['execution_path']} / {route['output']} / {route['representation']} / "
                       f"{route['session_boundary']}")
        lines.append("| " + " | ".join(map(str, [row["case_id"], row["operation"], row["api_tier"], route_label, row["dtype"],
            shape_label(row), row["layout"], row["workflow"], row["threads"], row.get("provider", "unknown"),
            row["samples"][0]["iterations"] * row["calls_per_workflow"] if row["samples"] else "—",
            f"{statistics.median(s['elapsed_ns'] for s in row['samples'])/1e6:.6f}" if row["samples"] else "—",
            median, iqr, cov, per_op, status])) + " |")
    diagnostics = [r for r in rows if r.get("measurement_kind") == "setup_diagnostic"]
    if diagnostics:
        lines += ["", "## Explicit setup diagnostics (not operation comparisons)", "",
                  "| Case | Median ns | IQR ns | Status |", "|---|---:|---:|---|"]
        for row in diagnostics:
            median, iqr, _, status = summary(row)
            lines.append(f"| {row['case_id']} | {median} | {iqr} | {status} |")
    lines += ["", "## Per-route timing boundaries", "",
              "Shared-session routes enter/exit the session once around warmup/calibration/all samples, outside timing. "
              "With tenferro-rs PR #1796 or later, managed BLAS and faer sessions reuse the entered executor context. Earlier BLAS revisions included per-operation executor entry; consult the recorded tenferro commit. "
              "Fresh routes enter/exit for each operation. Prepared-repeat excludes plan preparation; compiled-repeat excludes tracing/compilation and runtime construction. Prepared-into-repeat and dot-general-into-shared write into one destination preallocated outside timing (no per-call output allocation) and re-check it after the last batch.", ""]
    scopes = {}
    for row in rows:
        if "scope" in row:
            scopes[(row["operation"], row["api_tier"])] = row["scope"]
    for (operation, tier), scope in sorted(scopes.items()):
        lines += [f"- **{operation} / {tier}** — inside: {', '.join(scope['timer'])}; outside: {', '.join(scope['outside_timer'])}."]
    for row in rows:
        if row.get("error"):
            lines += ["", f"### FAILED: {row['case_id']} ({row['threads']} threads)", "```", row["error"], "```"]
    text = "\n".join(lines) + "\n"
    (run_dir / "report.md").write_text(text)
    latest = ROOT / "result" / safe_target_profile(target) / "cpu/small_work.md"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(text)
    print(latest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--target-profile", default="amd-cpu")
    args = parser.parse_args()
    if args.report:
        report(args.report, args.target_profile)
    else:
        raise SystemExit(collect(args.binary, args.output, args.threads))
