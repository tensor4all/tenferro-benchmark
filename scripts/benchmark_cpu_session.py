#!/usr/bin/env python3
"""cpu/session_matrix: many small matrices per sample, and one-operation batched routes.

Two workload families share this suite and keep distinct case IDs:

* ``loop_*``: 1024 independent small matrix calls per sample (each Rust
  process enters one session; PyTorch runs a Python loop);
* one-operation batched routes (tenferro-rs #1946 B2): one batched
  ``dot_general_read`` / ``_into`` / concrete einsum call per operation, many
  operations per wall-clock interval, one entered session.

Coverage (``BENCH_COVERAGE=quick|full``) picks case IDs from the versioned
manifest ``benchmarks/cpu/manifests/session_matrix.yaml``; effort
(``BENCH_EFFORT=scan|standard|confirm``) picks repetitions; see
scripts/bench_selection.py. Every run writes ``case_status.json``.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_selection  # noqa: E402
import case_status as cs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/cpu/session_matrix.yaml"
MANIFEST = ROOT / "benchmarks/cpu/manifests/session_matrix.yaml"
INSTANCES = ROOT / "data/instances/session_matrix.json"
SUITE_ID = "cpu/session_matrix"
BATCHED = {"batched_dot", "batched_einsum", "hadamard", "chain3", "stream"}
CASE_TIMEOUT_S = int(os.environ.get("BENCH_CASE_TIMEOUT_S", "900"))


def torch_case(n, count, samples, threads, op, warmups=3):
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    fixtures = []
    for k in range(count):
        # Same logical values as Rust; native contiguous layout per library.
        a = torch.tensor([[n+1+(k%17)*0.01 if r == c else ((r+c+k)%7)*0.01
                           for c in range(n)] for r in range(n)], dtype=torch.float64)
        b = torch.tensor([[((r+n*c+k)%13)*0.02-0.1 for c in range(n)] for r in range(n)], dtype=torch.float64)
        fixtures.append((a,b))
    expected = [a @ b for a,b in fixtures] if op == 'matmul' else None
    execute = torch.matmul if op == 'matmul' else torch.linalg.solve
    outputs = [None]*count
    durations = []
    for sample in range(samples+warmups):
        for i in range(count):
            outputs[i] = None
        start = time.perf_counter_ns()
        for i,(a,b) in enumerate(fixtures):
            outputs[i] = execute(a,b)
        elapsed = time.perf_counter_ns()-start
        if sample >= warmups:
            durations.append(elapsed)
        for i,((a,b),out) in enumerate(zip(fixtures,outputs)):
            actual, ref = (out,expected[i]) if expected is not None else (a @ out,b)
            torch.testing.assert_close(actual,ref,rtol=1e-11,atol=1e-11)
    return dict(operation=op,n=n,operations_per_sample=count,provider='pytorch',
                route='python-loop',session_count=None,warmups=warmups,samples_ns=durations,
                correctness='passed',torch_version=torch.__version__,torch_config=torch.__config__.show())


def load_cases():
    return {c["id"]: c for c in json.loads(INSTANCES.read_text())}


def plan(coverage=None, instance_filter=None):
    """Resolve the manifest, coverage and selection for this run."""
    manifest = bench_selection.load_manifest(MANIFEST)
    coverage = bench_selection.resolve_coverage(coverage)
    expected, selected, raw = bench_selection.select(manifest, coverage, instance_filter)
    cases = load_cases()
    missing = [i for i in manifest["coverage"]["full"] if i not in cases]
    if missing:
        raise bench_selection.SelectionError(f"manifest names unknown cases: {missing}")
    return manifest, coverage, expected, selected, raw, cases


def expand(ids, cases, threads):
    return [cs.case_key(i, threads, p) for i in ids for p in cases[i]["providers"]]


def per_op(row):
    ops = row.get("operations_per_sample") or 1
    return [v / ops for v in row.get("samples_ns", [])]


def run_one(command, timeout=CASE_TIMEOUT_S):
    """Run one case process; failures and timeouts become rows, never latencies."""
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"correctness": "timeout", "error": f"no result within {timeout} s (liveness)"}
    if result.returncode:
        return {"correctness": "failed", "error": result.stderr.strip()[-4000:]}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        return {"correctness": "failed", "error": f"invalid case output: {error}"}


def collect(binary, run_dir, threads, coverage=None):
    manifest, coverage, expected, selected, raw, cases = plan(coverage)
    suite = yaml.safe_load(SUITE.read_text())
    effort = bench_selection.resolve_effort(suite["defaults"]["run"],
                                            suite["defaults"].get("effort"))
    target_ns = int(suite["defaults"]["run"].get("min_runtime_ms", 1) * 1_000_000)
    rows = []
    with (run_dir / f"samples_t{threads}.jsonl").open("w") as fh:
        for case_id in selected:
            case = cases[case_id]
            for provider in case["providers"]:
                reps = ["--warmups", str(effort["warmups"]), "--samples", str(effort["runs"])]
                if case["workload"] in BATCHED:
                    command = [str(binary), "--case", case_id, "--threads", str(threads),
                               "--target-ns", str(target_ns), "--instances", str(INSTANCES),
                               "--provider", provider] + reps
                elif provider == "pytorch":
                    command = [sys.executable, __file__, "--torch-worker", "--threads", str(threads),
                               "--n", str(case["shape"][0]), "--count", str(case["operations_per_sample"]),
                               "--op", case["operation"]] + reps
                else:
                    command = [str(binary), "--provider", provider, "--n", str(case["shape"][0]),
                               "--count", str(case["operations_per_sample"]),
                               "--op", case["operation"]] + reps
                row = run_one(command)
                row.update(case_id=case_id, workload=case["workload"], threads=threads,
                           provider=row.get("provider", provider), route=case["route"],
                           coverage_tier=case["coverage"], effort=effort["effort"],
                           case_key=cs.case_key(case_id, threads, provider),
                           cpu_features=os.environ.get("TENFERRO_CPU_FEATURES"))
                if case["workload"] not in BATCHED:
                    row.update(timing_scope="many_operations_single_interval",
                               input_policy="distinct_prepared_pairs")
                    if provider == "blas":
                        row["provider"] = "accelerate" if sys.platform == "darwin" else "blas"
                row.setdefault("samples_ns", [])
                status = row.get("correctness", "failed")
                row["status"] = status
                row["cov"] = cs.coefficient_of_variation(per_op(row)) if status == "passed" else None
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                shown = (f"{statistics.median(per_op(row)):.1f} ns/op" if row["samples_ns"]
                         else row.get("unsupported_reason") or row.get("error", "")[:120])
                print(f"{case_id} [{provider}] t={threads}: {status} {shown}", flush=True)
    status = cs.build_case_status(
        suite_id=SUITE_ID, manifest_version=manifest["manifest_version"], coverage=coverage,
        effort=effort["effort"], expected=expand(expected, cases, threads),
        selected=expand(selected, cases, threads), rows=rows, selection_filter=raw)
    cs.write_case_status(run_dir / f"case_status_t{threads}.json", status)
    return bool(status["failed"] or status["missing"])


def fmt(values, scale=1.0):
    if not values:
        return "—", "—", "—"
    med = statistics.median(values) / scale
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [med * scale] * 3
    cov = cs.coefficient_of_variation(values)
    return f"{med:.2f}", f"{(q[2]-q[0])/scale:.2f}", "—" if cov is None else f"{cov*100:.1f}%"


def report(run_dir, target, latest=True):
    run_dir = run_dir.resolve()
    rows = [json.loads(line) for p in sorted(run_dir.glob('samples_t*.jsonl'))
            for line in p.read_text().splitlines() if line.strip()]
    parts = [json.loads(p.read_text()) for p in sorted(run_dir.glob('case_status_t*.json'))]
    status = cs.merge_case_status(parts) if parts else None
    if status:
        cs.write_case_status(run_dir / "case_status.json", status)
    rel = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
    lines = ['# CPU shared-session matrix results', '',
             f'Raw data and provenance: `{rel}`.', '']
    for path in sorted(run_dir.glob("run_t*.yaml")):
        meta = yaml.safe_load(path.read_text())
        keep = {k: meta[k] for k in ("timestamp", "tenferro_rs", "harness", "collection") if k in meta}
        lines += [f"### {path.stem}", "", "```yaml", yaml.safe_dump(keep, sort_keys=False).rstrip(), "```", ""]
    if status:
        lines += cs.markdown_section(status)
        if status["effort"] == "scan":
            lines += ["**Scan effort**: a low-repetition screen. Rows mark suspects only; "
                      "no performance conclusion may be drawn from them.", ""]
    loop = [r for r in rows if r["workload"] not in BATCHED]
    batched = [r for r in rows if r["workload"] in BATCHED]
    lines += ['## Loop of independent small calls', '',
             'Each sample executes 1024 independent, distinct f64 matrix pairs; '
             'one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, '
             'and initialization are outside timing. Outputs remain alive until timer stop. '
             'Every output is checked after timing (solve uses the residual).', '',
             'tenferro enters exactly one backend session around all warmups and samples. '
             'Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s '
             'Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). '
             'PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. '
             'Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; '
             'the one-operation batched cases below are a different workload.', '',
             'EagerTensor and compiled trace are not labeled shared-session: their current public interfaces '
             'do not accept this borrowed session and create internal sessions during execution. '
             'The old single-call measurements remain diagnostic evidence only.', '',
             '| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |',
             '|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|']
    for r in sorted(loop, key=lambda r: (r['case_id'], r['threads'], r['provider'])):
        med, iqr, cov = fmt(r['samples_ns'], 1e6)
        ops = r.get('operations_per_sample') if r['samples_ns'] else None
        per = f"{float(med) * 1e6 / ops:.2f}" if ops else "—"
        ops = ops or "—"
        n = r['case_id'].split('_')[1]
        lines.append(f"| {r['case_id']} | {r['case_id'].split('_')[0]} | {n} | {ops} | {r['threads']} | {r['provider']} | "
                     f"{r.get('route', {}).get('session_boundary', 'shared-session') if isinstance(r.get('route'), dict) else r.get('route')} | "
                     f"{r.get('execution_mode', 'not applicable')} | {med} | {iqr} | {per} | {cov} | {r['status']} |")
    lines += ['', '## One-operation batched routes (tenferro-rs #1946 B2)', '',
              'One public call per operation; many operations per wall-clock interval (calibrated toward '
              'the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session '
              'around warmup, calibration and all samples. tenferro-rs #2004 removed the batch-policy API, so '
              'cases whose manifest policy is an override report `unsupported` instead of being measured with '
              'the compiled backend default, and the batched and stream cases are declared for the faer '
              'provider, so a build without it reports them `unsupported` too. Allocating routes include their intrinsic output '
              'allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the '
              'first validated call and all numerical checks are outside timing. Route columns come from the '
              'case manifest, never from timing. NOISY (CoV > 10%) is descriptive only.', '',
              '| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |',
              '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in sorted(batched, key=lambda r: (r['case_id'], r['threads'])):
        route = r.get('route', {})
        med, iqr, cov = fmt(per_op(r))
        state = r['status']
        if state == 'passed' and r.get('cov') and r['cov'] > cs.NOISY_COV:
            state = 'passed (NOISY)'
        if state == 'unsupported':
            state = f"unsupported: {r.get('unsupported_reason', '')[:100]}"
        lines.append(f"| {r['case_id']} | {route.get('execution_path')} / {route.get('output')} / "
                     f"{route.get('representation')} / {route.get('layout')} | {route.get('policy')} | {r.get('dtype')} | "
                     f"{r.get('batch')}×{r.get('m')}×{r.get('n')}×{r.get('k')} | {r['threads']} | {r.get('worker_count', '—')} | "
                     f"{r.get('operations_per_sample', '—')} | {med} | {iqr} | {cov} | {state} |")
    for r in rows:
        if r['status'] not in ('passed', 'unsupported'):
            lines += ['', f"### {r['status'].upper()}: {r['case_key']}", '```', str(r.get('error', '')), '```']
    text = '\n'.join(lines) + '\n'
    (run_dir / 'report.md').write_text(text)
    if latest:
        path = ROOT / 'result' / target / 'cpu/session_matrix.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--torch-worker', action='store_true')
    p.add_argument('--n', type=int, default=2); p.add_argument('--count', type=int, default=1024)
    p.add_argument('--samples', type=int, default=15); p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--op', choices=['matmul', 'solve'], default='matmul')
    p.add_argument('--binary', type=Path); p.add_argument('--run-dir', type=Path)
    p.add_argument('--report', action='store_true'); p.add_argument('--target-profile', default='mac-cpu')
    p.add_argument('--coverage', choices=bench_selection.COVERAGES)
    p.add_argument('--no-latest', action='store_true', help='write only the raw-run report')
    p.add_argument('--describe-selection', action='store_true',
                   help='print the resolved coverage/effort/selection as JSON (run metadata)')
    args = p.parse_args()
    if args.torch_worker:
        print(json.dumps(torch_case(args.n, args.count, args.samples, args.threads, args.op, args.warmups)))
        return
    if args.describe_selection:
        manifest, coverage, expected, selected, raw, _ = plan(args.coverage)
        suite = yaml.safe_load(SUITE.read_text())
        effort = bench_selection.resolve_effort(suite["defaults"]["run"], suite["defaults"].get("effort"))
        print(json.dumps(dict(effort, coverage=coverage, manifest_version=manifest["manifest_version"],
                              selection_filter=raw, expected_cases=len(expected),
                              selected_cases=len(selected))))
        return
    if args.report:
        report(args.run_dir, args.target_profile, latest=not args.no_latest)
        return
    raise SystemExit(1 if collect(args.binary, args.run_dir, args.threads, args.coverage) else 0)


if __name__ == '__main__':
    main()
