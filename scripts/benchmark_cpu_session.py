#!/usr/bin/env python3
"""Collect many independent matrices per sample; each Rust process enters one session."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def torch_case(n, count, samples, threads, op):
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
    for sample in range(samples+3):
        for i in range(count):
            outputs[i] = None
        start = time.perf_counter_ns()
        for i,(a,b) in enumerate(fixtures):
            outputs[i] = execute(a,b)
        elapsed = time.perf_counter_ns()-start
        if sample >= 3:
            durations.append(elapsed)
        for i,((a,b),out) in enumerate(zip(fixtures,outputs)):
            actual, ref = (out,expected[i]) if expected is not None else (a @ out,b)
            torch.testing.assert_close(actual,ref,rtol=1e-11,atol=1e-11)
    return dict(operation=op,n=n,operations_per_sample=count,provider='pytorch',
                route='python-loop',session_count=None,warmups=3,samples_ns=durations,
                correctness='passed',torch_version=torch.__version__,torch_config=torch.__config__.show())


def report(run_dir, target):
    rows = [json.loads(line) for p in sorted(run_dir.glob('samples_t*.jsonl')) for line in p.read_text().splitlines()]
    lines = ['# CPU shared-session matrix results', '',
             f'Raw data and provenance: `{run_dir.as_posix()}`.', '',
             'Each sample executes 1024 independent, distinct f64 matrix pairs. Three warmups and 15 samples; '
             'one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, '
             'and initialization are outside timing. Outputs remain alive until timer stop. '
             'Every output is checked after timing (solve uses the residual).', '',
             'tenferro enters exactly one backend session around all warmups and samples. '
             'Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The execution mode and worker count are recorded in each Rust row. 4-thread faer uses tenferro’s '
             'Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). '
             'PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. '
             'Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs.', '',
             'EagerTensor and compiled trace are not labeled shared-session: their current public interfaces '
             'do not accept this borrowed session and create internal sessions during execution. '
             'The old single-call measurements remain diagnostic evidence only.', '',
             '| Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | Check |',
             '|---|---|---:|---:|---|---|---|---:|---:|---:|---|']
    for r in sorted(rows,key=lambda r:(r['operation'],r['n'],r['threads'],r['provider'])):
        values = r['samples_ns']; med = statistics.median(values)
        q = statistics.quantiles(values,n=4,method='inclusive')
        lines.append(f"| {r['operation']} | {r['n']}×{r['n']} | {r['operations_per_sample']} | {r['threads']} | {r['provider']} | {r['route']} | {r.get('execution_mode', 'not applicable')} | {med/1e6:.6f} | {(q[2]-q[0])/1e6:.6f} | {med/r['operations_per_sample']:.2f} | {r['correctness']} |")
    text = '\n'.join(lines)+'\n'
    (run_dir/'report.md').write_text(text)
    latest = Path('result')/target/'cpu/session_matrix.md'
    latest.parent.mkdir(parents=True,exist_ok=True); latest.write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--torch-worker',action='store_true')
    p.add_argument('--n',type=int,default=2); p.add_argument('--count',type=int,default=1024)
    p.add_argument('--samples',type=int,default=15); p.add_argument('--threads',type=int,default=1)
    p.add_argument('--op',choices=['matmul','solve'],default='matmul')
    p.add_argument('--binary',type=Path); p.add_argument('--run-dir',type=Path)
    p.add_argument('--report',action='store_true'); p.add_argument('--target-profile',default='mac-cpu')
    args = p.parse_args()
    if args.torch_worker:
        print(json.dumps(torch_case(args.n,args.count,args.samples,args.threads,args.op)))
        return
    if args.report:
        report(args.run_dir,args.target_profile); return
    with (args.run_dir/f'samples_t{args.threads}.jsonl').open('w') as fh:
        cases = json.loads(Path('data/instances/session_matrix.json').read_text())
        for case in cases:
            op, n = case['operation'], case['shape'][0]
            for provider in ['blas','faer','pytorch']:
                common = ['--n',str(n),'--count',str(args.count),'--samples',str(args.samples),'--op',op]
                command = ([sys.executable,__file__,'--torch-worker','--threads',str(args.threads)] if provider == 'pytorch'
                           else [str(args.binary),'--provider',provider]) + common
                row = json.loads(subprocess.check_output(command,text=True))
                row.update(threads=args.threads,timing_scope='many_operations_single_interval',
                           input_policy='distinct_prepared_pairs',cpu_features=os.environ.get('TENFERRO_CPU_FEATURES'))
                if provider == 'blas': row['provider'] = 'accelerate' if sys.platform == 'darwin' else 'blas'
                fh.write(json.dumps(row)+'\n'); fh.flush()
                print(f"{op} n={n} {row['provider']} t={args.threads}: {statistics.median(row['samples_ns'])/1e6:.3f} ms",flush=True)
if __name__ == '__main__': main()
