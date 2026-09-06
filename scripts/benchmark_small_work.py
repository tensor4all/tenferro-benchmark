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
from suite_instances import resolve_suite_instance_ids

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/cpu/small_work.yaml"


def selected_cases():
    suite = yaml.safe_load(SUITE.read_text())
    cases = {c["id"]: c for c in json.loads((ROOT / suite["problems"]["source"]).read_text())}
    ids = resolve_suite_instance_ids(SUITE)
    selected = os.environ.get("BENCH_INSTANCE", "")
    if selected:
        requested = selected.split(",")
        unknown = set(requested) - set(ids)
        if unknown:
            raise ValueError(f"unknown case IDs: {sorted(unknown)}")
        ids = [i for i in ids if i in requested]
    return [cases[i] for i in ids], suite["defaults"]["run"]


def collect(binary, output, threads):
    cases, config = selected_cases()
    failed = False
    with output.open("w") as fh:
        for case in cases:
            command = [str(binary), "--case", case["id"]]
            for key in ("operation", "dtype", "api_tier", "workflow", "layout"):
                command += ["--" + key.replace("_", "-"), str(case[key])]
            command += ["--size", str(math.prod(case["shape"])),
                        "--calls", str(case["calls_per_workflow"]),
                        "--warmups", str(config["warmups"]), "--samples", str(config["runs"]),
                        "--target-ns", str(int(config["min_runtime_ms"] * 1_000_000))]
            result = subprocess.run(command, text=True, capture_output=True)
            row = dict(case, threads=threads, case_id=case["id"], samples=[])
            if result.returncode:
                row.update(correctness_status="failed", error=result.stderr.strip())
            else:
                try:
                    row.update(json.loads(result.stdout))
                except (ValueError, TypeError) as error:
                    row.update(correctness_status="failed", error=f"invalid case output: {error}")
            failed |= row["correctness_status"] != "passed"
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{case['id']}: {row['correctness_status']}", flush=True)
    return failed


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
    passed = sum(r["correctness_status"] == "passed" for r in rows)
    lines += [f"Numerical checks: **{passed}/{len(rows)} recorded rows passed** (case × thread count).", ""]
    lines += ["## Timing", "",
              "Sequential release runs; 3 warmups, then calibration to a 1 ms batch and 15 measured batches per case. "
              "Statistics describe batch elapsed time / iterations, not individually timed calls. "
              "Median and inclusive IQR are in nanoseconds (ns); CoV is sample standard deviation / mean. "
              "NOISY means CoV > 10%, a descriptive label only. No noisy rows are excluded.", "",
              "Workflow time is the total for one call or the complete dependent chain. "
              "The separate ns/op column divides the chain median by its operation count, not an independently measured call. "
              "Setup rows measure preparation only, not execution.", "",
              "Inputs and backend construction, independent numerical checks and AD backward checks are outside timing. "
              "Returned outputs/plans are consumed with black_box and dropped inside timing; "
              "value downloads/checks are outside. Eager AD rows measure forward execution/recording, not backward.", "",
              "| Case ID | Operation | API route | Dtype | Shape | Layout | Workflow | Threads | Median workflow ns | IQR ns | CoV | ns/op (chain only) | Status |",
              "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        median, iqr, cov, status = summary(row)
        per_op = f"{float(median)/row['calls_per_workflow']:.2f}" if median != "—" and row['calls_per_workflow'] > 1 else "—"
        lines.append("| " + " | ".join(map(str, [row["case_id"], row["operation"], row["api_tier"], row["dtype"],
            "×".join(map(str, row["shape"])), row["layout"], row["workflow"], row["threads"], median, iqr, cov, per_op, status])) + " |")
    lines += ["", "## Per-route timing boundaries", "",
              "Shared-session routes enter/exit the session once around warmup/calibration/all samples, outside timing. "
              "Fresh routes enter/exit for each operation. Prepared-repeat excludes plan preparation; compiled-repeat excludes tracing/compilation and runtime construction.", ""]
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
