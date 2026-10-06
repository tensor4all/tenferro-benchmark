#!/usr/bin/env python3
"""Run the cpu/perf_issues cases (tenferro-rs performance-issue workloads).

Rust cases run one process each through ``perf_issue_case``; ``pytorch-cpu``
reference cases run in this process. Both use the same sampling scheme:
warmups, then calibration toward ``min_runtime_ms`` per batch (bounded by the
retained output memory), then the measured batches; each batch is one
wall-clock interval over many operations whose outputs are retained until the
clock stops. Coverage (quick/full) comes from the versioned manifest
``benchmarks/cpu/manifests/perf_issues.yaml``; effort (scan/standard) is
independent. Counter diagnostics have no samples; their counters are reported
in their own table.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import yaml

from benchmark_layout import safe_target_profile
import bench_selection
import case_status as cs
from suite_instances import resolve_suite_instance_ids

ROOT = Path(__file__).resolve().parents[1]
# suite key -> (suite yaml, manifest, suite id)
SUITES = {
    "cpu": (ROOT / "benchmarks/cpu/perf_issues.yaml", ROOT / "benchmarks/cpu/manifests/perf_issues.yaml",
            "cpu/perf_issues"),
    "gpu": (ROOT / "benchmarks/gpu/perf_issues.yaml", ROOT / "benchmarks/gpu/manifests/perf_issues.yaml",
            "gpu/perf_issues"),
}
RETENTION_BUDGET_BYTES = 512 << 20


def plan(suite_key="cpu"):
    suite_path, manifest_path, _ = SUITES[suite_key]
    suite = yaml.safe_load(suite_path.read_text())
    cases = {c["id"]: c for c in json.loads((ROOT / suite["problems"]["source"]).read_text())}
    manifest = bench_selection.load_manifest(manifest_path)
    if set(manifest["coverage"]["full"]) != set(resolve_suite_instance_ids(suite_path)):
        raise bench_selection.SelectionError("perf_issues manifest full list != suite include list")
    coverage = bench_selection.resolve_coverage()
    expected, selected, raw = bench_selection.select(manifest, coverage)
    return suite, cases, manifest, coverage, expected, selected, raw


# ---------------------------------------------------------------------------
# PyTorch reference arms
# ---------------------------------------------------------------------------

def _values(n, seed):
    return [((((i * 2654435761) + seed * 97) % 2001) / 1000.0 - 1.0) * 0.5 for i in range(n)]


def torch_case(case):
    """Return (setup -> (op, check, retained_bytes), timer, outside) for a PyTorch arm."""
    import torch

    p = case["params"]
    kind = case["kind"]
    if kind == "decode_projection":
        din, dout, length = p["in"], p["out"], p["len"]
        # Same logical x (in, len) and W (in, out) as the Rust arms; PyTorch's
        # native row-major Linear layout is x (len, in), W (out, in).
        x = torch.tensor(_values(din * length, 1), dtype=torch.float32).reshape(length, din)
        w = torch.tensor(_values(din * dout, 2), dtype=torch.float32).reshape(dout, din)

        def check():
            y = torch.nn.functional.linear(x, w)
            ref = x.double() @ w.double().T
            return float(((y.double() - ref).abs() / ref.abs().clamp(min=1.0)).max())
        return (lambda: torch.nn.functional.linear(x, w)), check, length * dout * 4, ["torch.nn.functional.linear", "output_allocation"]
    if kind == "rank1_dot":
        n = p["len"]
        re_x, im_x, re_y, im_y = _values(n, 9), _values(n, 10), _values(n, 11), _values(n, 12)
        if case["dtype"] == "c64":
            x = torch.complex(torch.tensor(re_x, dtype=torch.float64), torch.tensor(im_x, dtype=torch.float64))
            y = torch.complex(torch.tensor(re_y, dtype=torch.float64), torch.tensor(im_y, dtype=torch.float64))
        else:
            x, y = torch.tensor(re_x, dtype=torch.float64), torch.tensor(re_y, dtype=torch.float64)

        def check():
            got = complex(torch.vdot(x, y))
            ref = complex((x.conj() * y).sum())
            return abs(got - ref) / max(abs(ref), 1.0)
        return (lambda: torch.vdot(x, y)), check, 64, ["torch.vdot (BLAS ddot/zdotc)", "scalar tensor allocation"]
    if kind == "small_solves":
        k, batch = p["k"], p["batch"]
        a = torch.zeros(batch, k, k, dtype=torch.float64)
        for b in range(batch):
            raw = torch.tensor(_values(k * k, 20 + b), dtype=torch.float64).reshape(k, k).T  # column-major item
            a[b] = torch.tril(raw / k, -1) + torch.eye(k, dtype=torch.float64)
        rhs = torch.stack([torch.tensor(_values(k * k, 500 + b), dtype=torch.float64).reshape(k, k).T for b in range(batch)])
        rhs = rhs.contiguous()
        if case["arm"] == "pytorch-solve-batched":
            op = lambda: torch.linalg.solve(a, rhs)
            label = "torch.linalg.solve on (batch, k, k)"
        else:
            op = lambda: torch.linalg.solve_triangular(a, rhs, upper=False, unitriangular=True)
            label = "torch.linalg.solve_triangular on (batch, k, k)"

        def check():
            x = op()
            return float(((a @ x - rhs).abs() / rhs.abs().clamp(min=1.0)).max())
        return op, check, batch * k * k * 8, [label, "output_allocation"]
    if kind == "composed_norm":
        d, length, batch = p["d"], p["len"], p["batch"]
        # Feature-first (d, len, batch) column-major == row-major (batch, len, d).
        x = torch.tensor(_values(d * length * batch, 30), dtype=torch.float32).reshape(batch, length, d)
        w = 1.0 + torch.tensor(_values(d, 31), dtype=torch.float32)
        bias = torch.tensor(_values(d, 32), dtype=torch.float32)
        if p["norm"] == "layer_norm":
            op = lambda: torch.nn.functional.layer_norm(x, (d,), w, bias, 1e-5)

            def reference():
                xd = x.double()
                mean = xd.mean(-1, keepdim=True)
                var = ((xd - mean) ** 2).mean(-1, keepdim=True)
                return (xd - mean) / torch.sqrt(var + 1e-5) * w.double() + bias.double()
        else:
            op = lambda: torch.nn.functional.rms_norm(x, (d,), w, 1e-5)

            def reference():
                xd = x.double()
                return xd / torch.sqrt((xd ** 2).mean(-1, keepdim=True) + 1e-5) * w.double()

        def check():
            ref = reference()
            return float(((op().double() - ref).abs() / ref.abs().clamp(min=1.0)).max())
        return op, check, d * length * batch * 4, [f"torch.nn.functional.{p['norm']} (fused)", "output_allocation"]
    raise ValueError(f"no PyTorch arm for {kind}")


def run_torch(case, threads, config, correctness_only):
    import torch

    torch.set_num_threads(threads)
    op, check, retained, timer = torch_case(case)
    error = check()
    tol = 1e-4 if case["dtype"] == "f32" else 1e-10
    row = {"max_rel_error": error, "samples": [],
           "scope": {"timer": timer, "outside_timer": ["input_construction", "torch.set_num_threads", "correctness_check"]},
           "provider": torch.__config__.show().split("\n")[0].strip()}
    if not (error <= tol):
        row.update(correctness_status="failed", error=f"relative error {error:e} exceeds {tol:e}")
        return row
    row["correctness_status"] = "passed"
    if correctness_only:
        return row
    for _ in range(max(config["warmups"], 1)):
        op()
    cap = max(1, min(1 << 20, RETENTION_BUDGET_BYTES // max(retained, 1)))
    target_ns = int(config["min_runtime_ms"] * 1_000_000)

    def batch(iterations):
        outputs = []
        start = time.perf_counter_ns()
        for _ in range(iterations):
            outputs.append(op())
        elapsed = time.perf_counter_ns() - start
        del outputs
        return elapsed

    iterations = 1
    while True:
        elapsed = batch(iterations)
        if elapsed >= target_ns or iterations >= cap:
            break
        iterations = min(iterations * 2, cap)
    row["calibration"] = {"target_ns": target_ns, "iterations": iterations, "elapsed_ns": elapsed}
    row["samples"] = [{"sample_index": i, "elapsed_ns": batch(iterations), "iterations": iterations}
                      for i in range(config["runs"])]
    return row


# ---------------------------------------------------------------------------
# Collection and report
# ---------------------------------------------------------------------------

def collect(binary, output, threads, correctness_only=False, suite_key="cpu", device=0):
    suite, all_cases, manifest, coverage, expected, selected, raw = plan(suite_key)
    suite_id = SUITES[suite_key][2]
    config = dict(suite["defaults"]["run"])
    effort = bench_selection.resolve_effort(config, suite["defaults"].get("effort"))
    config.update(warmups=effort["warmups"], runs=effort["runs"])
    failed = False
    rows = []
    with output.open("w") as fh:
        for case in (all_cases[i] for i in selected):
            row = dict(case, threads=threads, case_id=case["id"], samples=[],
                       case_key=cs.case_key(case["id"], threads), effort=effort["effort"])
            if case["backend"] == "pytorch-cpu":
                try:
                    row.update(run_torch(case, threads, config, correctness_only))
                except Exception as error:  # a crash is a failed row, never a latency
                    row.update(correctness_status="failed", error=repr(error))
            else:
                command = [str(binary), "--case", case["id"], "--params", json.dumps(case["params"]),
                           "--threads", str(threads), "--device", str(device),
                           "--warmups", str(config["warmups"]),
                           "--samples", str(config["runs"]),
                           "--target-ns", str(int(config["min_runtime_ms"] * 1_000_000))]
                if correctness_only:
                    command += ["--mode", "correctness-only"]
                result = subprocess.run(command, text=True, capture_output=True)
                if result.returncode:
                    row.update(correctness_status="failed", error=result.stderr.strip())
                else:
                    try:
                        row.update(json.loads(result.stdout))
                    except (ValueError, TypeError) as error:
                        row.update(correctness_status="failed", error=f"invalid case output: {error}")
            row["status"] = row["correctness_status"]
            failed |= row["status"] != "passed"
            values = [s["elapsed_ns"] / s["iterations"] for s in row["samples"]]
            row["cov"] = cs.coefficient_of_variation(values) if row["status"] == "passed" else None
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
            print(f"{case['id']}: {row['correctness_status']}", flush=True)
    status = cs.build_case_status(
        suite_id=suite_id, manifest_version=manifest["manifest_version"], coverage=coverage,
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
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [median] * 3
    cov = statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0
    return f"{median:.1f}", f"{q[2] - q[0]:.1f}", f"{cov * 100:.1f}%", "NOISY" if cov > 0.10 else "ok"


def shape_label(row):
    return ", ".join(f"{k}={v}" for k, v in row["params"].items() if k not in {"kind", "arm", "dtype"})


def report(run_dir, target, suite_key="cpu"):
    suite_id = SUITES[suite_key][2]
    run_dir = run_dir.resolve()
    rows = [json.loads(line) for path in sorted(run_dir.glob("samples_t*.jsonl"))
            for line in path.read_text().splitlines()]
    lines = [f"# {suite_key.upper()} Performance-Issue Workloads", "", f"- Suite: `{suite_id}`",
             f"- Raw run: `{run_dir.relative_to(ROOT)}`", ""]
    for info in ("cpu_info.md", "gpu_info.md"):
        if (run_dir / info).exists():
            lines += [(run_dir / info).read_text().rstrip(), ""]
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
    header = ["| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |",
              "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|"]

    def table(selected):
        out = list(header)
        for row in selected:
            median, iqr, cov, status = summary(row)
            out.append("| " + " | ".join(map(str, [
                row["case_id"], " ".join(row["issues"]), row["arm"], row["backend"], row["dtype"],
                shape_label(row), row["threads"],
                row["samples"][0]["iterations"] if row["samples"] else "—", median, iqr, cov, status])) + " |")
        return out

    steady = [r for r in rows if r["measurement_scope"] == "steady_state"]
    lines += ["## Steady-state operation rows", "",
              "Each batch is one wall-clock interval over many operations; every output is retained until the "
              "clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard "
              "deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, "
              "runtimes, session entry, tracing/compilation, preparation) is outside timing; faer-direct and "
              "host-sgemm write into a preallocated output, the tenferro and PyTorch arms allocate their output. "
              "`reference_for` in the case file names the case each reference arm is compared with. "
              "GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); "
              "the Threads column is the host thread setting.", ""]
    lines += table(steady)
    for scope, title, note in (
            ("session_entry_diagnostic", "Per-call session entry diagnostics (not operation comparisons)",
             "A session (or eager session) is entered and left for every operation, inside the timer."),
            ("eager_ad_workflow", "Eager AD workflow diagnostics",
             "Forward + reduction + backward through the eager runtime, including its internal session entries; "
             "gradients accumulate across a batch and are reset outside timing.")):
        selected = [r for r in rows if r["measurement_scope"] == scope]
        if selected:
            lines += ["", f"## {title}", "", note, ""] + table(selected)
    counters = [r for r in rows if r["measurement_scope"] == "counter_diagnostic"]
    if counters:
        lines += ["", "## Counter diagnostics (no timing)", ""]
        for row in counters:
            lines += [f"### {row['case_id']} ({' '.join(row['issues'])}, {row['threads']} threads)", ""]
            if row["correctness_status"] != "passed":
                lines += ["FAILED", ""]
                continue
            lines += [f"Source: {row.get('counter_source', '')}. Unavailable: {', '.join(row.get('unavailable', []))}.", "",
                      "| Op | Output bytes | Host allocations/op | Allocated bytes/op | Allocated/output |",
                      "|---|---:|---:|---:|---:|"]
            for c in row.get("counters", []):
                lines.append(f"| {c['op']} | {c['output_bytes']} | {c['host_allocations_per_op']:.2f} | "
                             f"{c['host_allocated_bytes_per_op']:.0f} | {c['allocated_bytes_over_output_bytes']:.2f} |")
            lines.append("")
    first_calls = [r for r in rows if r["measurement_scope"] == "first_call_diagnostic"]
    if first_calls:
        lines += ["", "## First-call diagnostics (not steady-state rows)", "",
                  "One cold process; each distinct layout is uploaded outside timing, then its first and "
                  "second call are timed separately (each includes a device synchronize).", ""]
        for row in first_calls:
            lines += [f"### {row['case_id']} ({' '.join(row['issues'])})", ""]
            if row["correctness_status"] != "passed":
                lines += ["FAILED", ""]
                continue
            lines += [f"Median over {row['distinct_layouts']} distinct layouts: first call "
                      f"{row['median_first_call_ms']:.3f} ms, second call {row['median_second_call_ms']:.3f} ms.", "",
                      "| Layout | Warm-up layout | First call ms | Second call ms |", "|---|---|---:|---:|"]
            for c in row["first_call_layouts"]:
                lines.append(f"| {c['layout']} | {'yes' if c['warm_up_layout'] else ''} | "
                             f"{c['first_call_ms']:.3f} | {c['second_call_ms']:.3f} |")
            lines.append("")
    lines += ["", "## Timing boundaries", ""]
    scopes = {}
    for row in rows:
        if row.get("scope"):
            scopes[(row["kind"], row["arm"])] = row["scope"]
    for (kind, arm), scope in sorted(scopes.items()):
        lines.append(f"- **{kind} / {arm}** — inside: {', '.join(scope['timer']) or '—'}; "
                     f"outside: {', '.join(scope['outside_timer'])}.")
    for row in rows:
        if row.get("error"):
            lines += ["", f"### FAILED: {row['case_id']} ({row['threads']} threads)", "```", row["error"], "```"]
    text = "\n".join(lines) + "\n"
    (run_dir / "report.md").write_text(text)
    latest = ROOT / "result" / safe_target_profile(target) / f"{suite_id}.md"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(text)
    print(latest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--target-profile", default="amd-cpu")
    parser.add_argument("--suite", choices=sorted(SUITES), default="cpu")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.report:
        report(args.report, args.target_profile, args.suite)
    else:
        raise SystemExit(collect(args.binary, args.output, args.threads, args.correctness_only,
                                 args.suite, args.device))
