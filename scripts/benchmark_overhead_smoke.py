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
import os
from pathlib import Path
import statistics
import subprocess

import yaml

from benchmark_layout import safe_target_profile

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/cpu/overhead_smoke.yaml"
SUITE_ID = "cpu/overhead_smoke"


def load():
    suite = yaml.safe_load(SUITE.read_text())
    cases = {c["id"]: c for c in json.loads((ROOT / suite["problems"]["source"]).read_text())}
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
            row = {"suite_id": SUITE_ID, "case_id": cid, "threads": threads,
                   **{k: case[k] for k in ("refs", "fixture", "backend_thread_budget", "note")},
                   "counter_case": case.get("counter_case")}
            command = [str(binary), "--case", cid, "--params", json.dumps(case["params"]),
                       "--threads", str(threads), "--device", "0",
                       "--warmups", str(run["warmups"]), "--samples", str(run["runs"]),
                       "--target-ns", str(int(run["min_runtime_ms"] * 1_000_000))]
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


def summary(row):
    values = [s["elapsed_ns"] / s["iterations"] for s in row.get("samples", [])]
    if row.get("correctness_status") != "passed" or not values:
        return "—", "—", "—"
    median = statistics.median(values)
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [median] * 3
    cov = statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0
    return f"{median:.1f}", f"{q[2] - q[0]:.1f}", f"{cov * 100:.1f}%" + (" NOISY" if cov > 0.10 else "")


def report(run_dir, target):
    safe_target_profile(target)
    run_dir = run_dir.resolve()
    rows = [json.loads(line) for path in sorted(run_dir.glob("samples_t*.jsonl"))
            for line in path.read_text().splitlines()]
    tenferro = {}
    lines = ["# CPU Overhead Smoke (ns/op and allocs/op)", "", f"- Suite: `{SUITE_ID}`",
             f"- Target profile: `{target}`", f"- Raw run: `{run_dir.relative_to(ROOT)}`"]
    run_yaml = run_dir / "run.yaml"
    if run_yaml.exists():
        metadata = yaml.safe_load(run_yaml.read_text())
        tenferro = metadata.get("tenferro_rs", {})
        lines += [f"- tenferro-rs commit: `{tenferro.get('commit')}`",
                  f"- Collection command: `{metadata.get('collection', {}).get('command', '')}`"]
    lines += ["", "Trend view for tenferro-rs #2178 (tenferro-benchmark #197); nothing here gates. "
              "Every row runs on one run thread; the fused rows set the CPU backend thread budget "
              "explicitly. **allocs/op** counts Rust global-allocator calls of the process (calling "
              "thread and Rust workers) over 64 calls after pool warmup, outputs dropped between "
              "calls, so it is the pooled steady state of the matching `cpu/perf_issues` counter "
              "case: exact and hardware independent, comparable across target profiles and nights. "
              "Foreign BLAS/C allocations are not counted. **ns/op** is the median per-operation "
              "time of batches whose outputs are retained until the clock stops (setup, session "
              "entry where the API takes a session, tracing, compilation and preparation are "
              "outside timing); on shared nightly runners it is a reference indicator only.", "",
              "| Case | Workload | Budget | Median ns/op | IQR ns | CoV | allocs/op | bytes/op | Refs |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    passed = 0
    for row in rows:
        counters = row.get("counters") or [{}]
        median, iqr, cov = summary(row)
        ok = row.get("correctness_status") == "passed"
        passed += ok
        allocs = counters[0].get("host_allocations_per_op")
        allocated = counters[0].get("host_allocated_bytes_per_op")
        lines.append("| " + " | ".join([
            row["case_id"] if ok else f"{row['case_id']} (FAILED)", row["note"],
            str(row["backend_thread_budget"]), median, iqr, cov,
            "—" if allocs is None else f"{allocs:.2f}",
            "—" if allocated is None else f"{allocated:.0f}", " ".join(row["refs"])]) + " |")
    lines += ["", f"Numerical checks: **{passed}/{len(rows)} rows passed** against independent host "
              "loops before counting and sampling.", ""]
    fused = [r for r in rows if r.get("fused_regions")]
    if fused:
        lines += ["## Fused region execution", "",
                  "Whether the runtime executed the traced region as one fused command or declined "
                  "to per-op dispatch (cumulative over validation, the allocator pass and timed "
                  "batches). `run_prepared` enters a runtime-internal session per call, and its "
                  "entry allocations are part of allocs/op.", "",
                  "| Case | Planned regions | Fused executions | Fallbacks |", "|---|---:|---:|---:|"]
        for row in fused:
            for info in row["fused_regions"].values():
                counts = info["elementwise_region_execution_counts"]
                lines.append(f"| {row['case_id']} | {info['elementwise_region_summary']['regions']} | "
                             f"{counts['fused']} | {counts['fallbacks']} |")
        lines.append("")
    failures = [r for r in rows if r.get("correctness_status") != "passed"]
    if failures:
        lines += ["## Failures", ""]
        lines += [f"- `{r['case_id']}`: {r.get('error', 'unknown error')}" for r in failures]
        lines.append("")
    if (run_dir / "cpu_info.md").exists():
        lines += [(run_dir / "cpu_info.md").read_text().rstrip(), ""]
    if run_yaml.exists():
        conditions = {k: metadata[k] for k in ("timestamp", "tenferro_rs", "blas", "environment")
                      if k in metadata}
        lines += ["## Run metadata", "", f"Full metadata: `{run_yaml.relative_to(ROOT)}`", "",
                  "```yaml", yaml.safe_dump(conditions, sort_keys=False).rstrip(), "```", ""]
    text = "\n".join(lines)
    (run_dir / "report.md").write_text(text)
    latest = ROOT / "result" / target / "cpu" / "overhead_smoke.md"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(text)
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
