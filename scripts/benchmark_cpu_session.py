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
import report_summary  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/cpu/session_matrix.yaml"
MANIFEST = ROOT / "benchmarks/cpu/manifests/session_matrix.yaml"
INSTANCES = ROOT / "data/instances/session_matrix.json"
SUITE_ID = "cpu/session_matrix"
BATCHED = {"batched_dot", "batched_einsum", "hadamard", "chain3", "stream", "eager_stream"}
BLAS_IMPLEMENTATIONS = ("accelerate", "openblas", "mkl")
DEFAULT_CASE_TIMEOUT_S = 900


def case_timeout_seconds(env=None):
    """Collection-time liveness timeout; read where a case process is spawned."""
    env = os.environ if env is None else env
    return int(env.get("BENCH_CASE_TIMEOUT_S", str(DEFAULT_CASE_TIMEOUT_S)))


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
    # Initialize this exact public operation even with zero requested warmups.
    initialized = execute(*fixtures[0])
    del initialized
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


def run_one(command, timeout=None):
    """Run one case process; failures and timeouts become rows, never latencies."""
    if timeout is None:
        timeout = case_timeout_seconds()
    started = time.monotonic()
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"correctness": "timeout", "error": f"no result within {timeout} s (liveness)", "command_wall_seconds": time.monotonic()-started}
    if result.returncode:
        return {"correctness": "failed", "error": result.stderr.strip()[-4000:], "command_wall_seconds": time.monotonic()-started}
    try:
        row = json.loads(result.stdout.strip().splitlines()[-1])
        row["command_wall_seconds"] = time.monotonic()-started
        return row
    except (ValueError, IndexError) as error:
        return {"correctness": "failed", "error": f"invalid case output: {error}", "command_wall_seconds": time.monotonic()-started}


def collect(binary, run_dir, threads, coverage=None):
    started = time.monotonic()
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
    latency_finished = time.monotonic()
    diagnostic_failed = False
    # All latency collection for this thread setting has stopped. Counters
    # execute separately, without latency samples or per-call clocks.
    with (run_dir / f"cache_diagnostics_t{threads}.jsonl").open("w") as fh:
        for row in rows:
            if row.get("workload") != "eager_stream" or row.get("status") != "passed":
                continue
            command = [str(binary), "--case", row["case_id"], "--threads", str(threads),
                       "--instances", str(INSTANCES), "--provider", row["provider"],
                       "--mode", "cache-diagnostic", "--warmups", str(effort["warmups"]),
                       "--samples", str(effort["runs"])]
            diagnostic = run_one(command)
            diagnostic.update(case_id=row["case_id"], threads=threads, command=command)
            diagnostic_failed |= diagnostic.get("correctness") != "passed" or len(diagnostic.get("cache_diagnostics", [])) != effort["runs"]
            fh.write(json.dumps(diagnostic) + "\n")
    (run_dir / f"wall_collection_t{threads}.json").write_text(json.dumps({
        "latency_case_processes_seconds": latency_finished-started,
        "separate_cache_diagnostics_seconds": time.monotonic()-latency_finished,
        "collection_total_seconds": time.monotonic()-started}, indent=2)+"\n")
    return bool(status["failed"] or status["missing"] or diagnostic_failed)


def fmt(values, scale=1.0):
    if not values:
        return "—", "—", "—"
    med = statistics.median(values) / scale
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [med * scale] * 3
    cov = cs.coefficient_of_variation(values)
    return f"{med:.2f}", f"{(q[2]-q[0])/scale:.2f}", "—" if cov is None else f"{cov*100:.1f}%"


def merged_case_status(run_dir):
    parts = [json.loads(path.read_text()) for path in sorted(Path(run_dir).glob('case_status_t*.json'))]
    return cs.merge_case_status(parts) if parts else None


def build_summary(run_dir, *, cpu_model=None):
    """Machine-readable sidecar for the same records the report is built from."""
    run_dir = Path(run_dir).resolve()
    metadata = {}
    run_yaml = run_dir / "run.yaml"
    if run_yaml.is_file():
        metadata = yaml.safe_load(run_yaml.read_text()) or {}
    rows = []
    threads = set()
    for path in sorted(run_dir.glob('samples_t*.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            count = record.get('operations_per_sample', 1)
            values = [value / count for value in record.get('samples_ns', [])]
            verified = record.get('correctness') == 'passed'
            provider = str(record.get('provider', 'unknown'))
            provider = 'blas' if provider in BLAS_IMPLEMENTATIONS else provider
            backend = 'pytorch-cpu' if provider == 'pytorch' else 'tenferro-' + provider
            status = 'ok' if verified and values else 'failed'
            reason = record.get('unsupported_reason') or record.get('reason') or record.get('error')
            if record.get('status') == 'unsupported' and reason:
                status = 'unsupported'
            quartiles = (
                statistics.quantiles(values, n=4, method='inclusive')
                if len(values) > 1
                else [values[0]] * 3 if values else []
            )
            threads.add(int(record['threads']))
            rows.append(
                report_summary.make_row(
                    record['case_id'],
                    backend,
                    statistics.median(values) if status == 'ok' else None,
                    status=status,
                    verified=verified,
                    iqr=quartiles[2] - quartiles[0] if quartiles and status == 'ok' else None,
                    group=f"threads={record['threads']}",
                    reason=reason,
                )
            )
    header = report_summary.metadata_from_run_metadata(metadata)
    hardware = dict(header['hardware'])
    if cpu_model:
        hardware['cpu'] = cpu_model
    hardware.setdefault('cpu', 'unknown')
    if threads:
        hardware['threads'] = sorted(threads)
    return report_summary.build_document(
        rows,
        target_profile=header['target_profile'],
        suite_id='cpu/session_matrix',
        timestamp=header['timestamp'],
        tenferro_commit=header['tenferro_commit'],
        hardware=hardware,
        unit='ns',
        **report_summary.v2_provenance(run_dir, metadata),
    )


def render_report(run_dir, target):
    """Report text built only from files recorded in the run directory."""
    run_dir = Path(run_dir).resolve()
    rows = [json.loads(line) for p in sorted(run_dir.glob('samples_t*.jsonl'))
            for line in p.read_text().splitlines() if line.strip()]
    status = merged_case_status(run_dir)
    rel = report_summary.repo_relative(run_dir)
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
    batched = [r for r in rows if r["workload"] in BATCHED and r["workload"] not in {"stream", "eager_stream"}]
    streams = [r for r in rows if r["workload"] in {"stream", "eager_stream"}]
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
             'This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session '
             'shape streams are reported separately below. Compiled trace still uses runtime-internal '
             'session entry and is not a borrowed-session route.', '',
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
    if streams:
        lines += ['', '## Whole shape-sequence public-call diagnostics', '',
                  'One sample is the entire ordered sequence, with required planning/cache lookup inside the clock. Inputs, views, wrapping, bindings and retention containers are prepared first; outputs survive clock stop. The initialization pass and warmups are separate. Exact keys and ordering for every pass are in the raw row. Cold mode clears only the declared eager extension caches outside each sequence; it does not bypass lookup or retention. Capacity thrash bounds that store to one entry with four revisited keys. Concrete and eager three-operand streams are distinct workloads.', '',
                  '| Case | Provider | Threads | Calls/sequence | Median sequence ms | IQR ms | p90 ms | Maximum ms | CoV | Status |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for row in streams:
            values = row['samples_ns']; med, iqr, cov = fmt(values, 1e6)
            p90 = f"{statistics.quantiles(values, n=10, method='inclusive')[8]/1e6:.3f}" if len(values)>1 else '—'
            maximum = f'{max(values)/1e6:.3f}' if values else '—'
            state = row['status'] + (' (NOISY; inconclusive)' if row.get('cov') and row['cov']>.1 else '')
            lines.append(f"| {row['case_id']} | {row['provider']} | {row['threads']} | {row.get('operations_per_sample', '—')} | {med} | {iqr} | {p90} | {maximum} | {cov} | {state} |")
        diagnostic_rows = [json.loads(line) for p in sorted(run_dir.glob('cache_diagnostics_t*.jsonl')) for line in p.read_text().splitlines()]
        if diagnostic_rows:
            lines += ['', '### Separate-process cache observations', '',
                      'Public EagerRuntime extension-store counters only. Provider GEMM-analysis/indexed-plan caches and runtime prepared caches are outside this scope. No public plan-build counter exists; lookup misses are not treated as builds. These processes collect no latency samples. Hits/misses/evictions are measured-pass deltas; retained bytes are a logical estimate, not process RSS.', '',
                      '| Case | Threads | Hits | Misses | Evictions | Final entries | Final retained bytes | Builds | Status |',
                      '|---|---:|---:|---:|---:|---:|---:|---|---|']
            for row in diagnostic_rows:
                samples = row.get('cache_diagnostics', [])
                deltas = [sum(s['after'][key]-s['before'][key] for s in samples) for key in ('hits','misses','evictions')]
                final = samples[-1]['after'] if samples else {}
                lines.append(f"| {row['case_id']} | {row['threads']} | {deltas[0]} | {deltas[1]} | {deltas[2]} | {final.get('entries', '—')} | {final.get('retained_bytes', '—')} | unavailable | {row.get('correctness')} |")
    for path in sorted(run_dir.glob('wall*.json')):
        lines += ['', f'Wall observations `{path.name}`:', '```json', path.read_text().rstrip(), '```']
    return '\n'.join(lines) + '\n'


def report(run_dir, target, latest=True):
    run_dir = Path(run_dir).resolve()
    text = render_report(run_dir, target)
    document = build_summary(run_dir, cpu_model=report_summary.cpu_model_from_run_dir(run_dir))
    status = merged_case_status(run_dir)
    if status:
        cs.write_case_status(run_dir / 'case_status.json', status)
    (run_dir / 'report.md').write_text(text)
    report_summary.write_document(run_dir / 'session_matrix.summary.json', document)
    if latest:
        path = ROOT / 'result' / target / 'cpu/session_matrix.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        report_summary.write_document(report_summary.summary_path_for_report(path), document)
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
