#!/usr/bin/env python3
"""Sequential process replicates for the stateful QR suite (#94)."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import yaml
from benchmark_layout import safe_target_profile
import case_status as cs

ROOT = Path(__file__).resolve().parents[1]
PATHS = ['shared', 'fresh', 'eager-fresh', 'eager-shared', 'attributed', 'bcgs2', 'bcgs2-attributed']


def collect(binary, output, threads):
    start = time.monotonic()
    cases = json.loads((ROOT / 'data/instances/stateful_linalg.json').read_text())
    rows = []
    with output.open('w') as fh:
        # Every process gets three warmups and three measured intervals.
        # Seven fresh process medians, with alternating order of paths.
        for process_index in range(7):
            ordered = cases if process_index % 2 == 0 else list(reversed(cases))
            for case in ordered:
                clock = time.monotonic()
                command = [str(binary), '--rows', str(case['rows']), '--path', case['path'],
                           '--threads', str(threads), '--warmups', '3', '--samples', '3']
                run = subprocess.run(command, text=True, capture_output=True)
                row = dict(case, case_id=case['id'], process_index=process_index,
                           threads=threads, command=command, wall_seconds=time.monotonic()-clock,
                           samples_ns=[], correctness='failed')
                if run.returncode:
                    row['error'] = run.stderr.strip()
                else:
                    try:
                        row.update(json.loads(run.stdout))
                    except (ValueError, TypeError) as error:
                        row['error'] = str(error)
                rows.append(row)
                fh.write(json.dumps(row) + '\n'); fh.flush()
                print(f"process {process_index}: {case['id']}: {row['correctness']}", flush=True)
    (output.parent / f'wall_t{threads}.json').write_text(json.dumps({
        'collection_wall_seconds': time.monotonic()-start,
        'phase_policy': 'one process per case: input/setup, warmup, timed intervals, correctness and teardown; per-case wall plus binary initialization_ms recorded',
        'process_order': 'ascending case order on even rounds; reverse on odd rounds'}, indent=2)+'\n')
    return any(row['correctness'] != 'passed' for row in rows)


def statistics_for(rows, environment_observed=True):
    if len(rows) != 7 or any(r['correctness'] != 'passed' or len(r['samples_ns']) != 3 for r in rows):
        return None
    medians = [statistics.median(r['samples_ns']) for r in rows]
    per_sequence = [n/r['workflows_per_sample'] for n, r in zip(medians, rows)]
    mean = statistics.mean(per_sequence)
    cov = statistics.stdev(per_sequence)/mean
    observed = environment_observed and all(r.get('worker_count') == r['threads'] and r.get('provider') and r.get('affinity') == 'none' for r in rows)
    status = 'observed' if cov <= .1 and min(medians) >= 1_000_000 and observed else 'inconclusive'
    return dict(median=statistics.median(per_sequence), cov=cov, min_batch=min(medians), status=status,
                reconstruction=max(r['max_reconstruction_error'] for r in rows),
                orthogonality=max(r['max_orthogonality_error'] for r in rows))


def report(run_dir, target):
    groups = {}
    for path in sorted(run_dir.glob('samples_t*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            groups.setdefault((row['threads'], row['rows'], row['path']), []).append(row)
    lines = ['# CPU Stateful Incremental QR', '', '- Suite: `cpu/stateful_linalg`',
             f'- Raw run: `{run_dir.relative_to(ROOT)}`', '',
             (run_dir/'cpu_info.md').read_text().rstrip(), '',
             'F64, compact column-major inputs, initial rank 5, nine width-3 appends, final rank 32. Each workflow interleaves append, R snapshot, and the three new Q columns. Release execution; seven fresh processes per case, alternating path order; three warmups and three measured 64-workflow batches per process. One entered session surrounds all shared-route samples. Fresh routes include per-operation session entry and are explicitly workflow diagnostics. Both eager paths reuse one runtime; capture is inactive and values untracked.', '',
             'Initial factors, input blocks, restoration, output-retention containers, runtime/backend/session setup, validation and destruction are untimed. All compact states, R/Q snapshots and explicit-Q BCGS2 intermediates survive clock stop. Numerical checks cover every interleaved prefix, final reconstruction and orthogonality. Attribution is a separate diagnostic with individual call clocks; ordinary workflow rows contain no component clocks.', '',
             'Medians are microseconds per complete sequence; CoV uses the seven process medians. CoV >10%, missing provider/thread observations, missing replicates or a process-median batch shorter than 1 ms is inconclusive. No result here claims a speedup against historical hardware.', '',
             '| Rows | Path | Threads | Median sequence µs | Process CoV | Minimum process-median batch ms | Reconstruction error | Orthogonality error | Status |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---|']
    observations = {}
    for threads, _, _ in groups:
        path = run_dir / f'run_t{threads}.yaml'
        metadata = yaml.safe_load(path.read_text()) if path.exists() else {}
        env = metadata.get('environment', {}).get('env', {})
        observations[threads] = bool(metadata.get('tenferro_rs', {}).get('commit') and
            metadata.get('collection', {}).get('harness_commit') and
            all(str(env.get(name)) == str(threads) for name in ('RAYON_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')))
    summaries = {key: statistics_for(rows, observations[key[0]]) for key, rows in groups.items()}
    for (threads, m, path), summary in sorted(summaries.items()):
        if summary:
            s = summary
            lines.append(f"| {m} | {path} | {threads} | {s['median']/1000:.3f} | {s['cov']:.1%} | {s['min_batch']/1e6:.3f} | {s['reconstruction']:.2e} | {s['orthogonality']:.2e} | {s['status']} |")
        else:
            lines.append(f'| {m} | {path} | {threads} | — | — | — | — | — | incomplete/failed |')
    lines += ['', '| Rows | Threads | Append µs | R µs | New Q µs | Complete/(append+R+Q) | Fresh/shared | Eager fresh/shared concrete | Eager shared/shared concrete | Compact append/BCGS2 append | Quality |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for threads, m in sorted({(t,m) for t,m,_ in groups}):
        paths = {p: summaries.get((threads,m,p)) for p in PATHS}
        if any(s is None for s in paths.values()):
            lines.append(f'| {m} | {threads} | — | — | — | — | — | — | — | — | incomplete |'); continue
        parts = [statistics.median(statistics.median(sample[i] for sample in row['component_samples_ns'])/row['workflows_per_sample'] for row in groups[(threads,m,'attributed')]) for i in range(3)]
        bcgs_append = statistics.median(statistics.median(sample[0] for sample in row['component_samples_ns'])/row['workflows_per_sample'] for row in groups[(threads,m,'bcgs2-attributed')])
        shared = paths['shared']['median']
        quality = 'observed diagnostic ratios' if all(s['status']=='observed' for s in paths.values()) else 'inconclusive'
        numbers = [*(p/1000 for p in parts), shared/sum(parts), paths['fresh']['median']/shared,
                   paths['eager-fresh']['median']/shared, paths['eager-shared']['median']/shared, parts[0]/bcgs_append]
        lines.append(f'| {m} | {threads} | '+' | '.join(f'{n:.3f}' for n in numbers)+f' | {quality} |')
    lines += ['', 'BCGS2 uses two QᵀB projections, two residual corrections, QR of the new residual block, and public concatenation to update Q/R. It is an explicit-Q diagnostic using the same provider, not accumulated full QR. Compact/BCGS2 compares the separately attributed append work; extraction/retention clocks are excluded from that component. Ratios between fresh and shared paths include session policy differences, so they do not isolate eager wrapper cost.', '']
    for path in sorted(run_dir.glob('run_t*.yaml')):
        metadata = yaml.safe_load(path.read_text())
        lines += [f'## {path.stem}', '', '```yaml', yaml.safe_dump({k:metadata[k] for k in ('timestamp','tenferro_rs','blas','environment','collection') if k in metadata},sort_keys=False).rstrip(), '```', '']
    expected = [cs.case_key(c['id'],t) for t in sorted({t for t,_,_ in groups}) for c in json.loads((ROOT/'data/instances/stateful_linalg.json').read_text())]
    rows = []
    for (t,m,p), summary in summaries.items():
        rows.append(dict(case_key=cs.case_key(f'incremental_qr_m{m}_{p}',t), status='passed' if summary else 'failed', cov=summary['cov'] if summary else None))
    status = cs.build_case_status(suite_id='cpu/stateful_linalg',manifest_version=1,coverage='full',effort='standard',expected=expected,selected=expected,rows=rows,selection_filter=None)
    cs.write_case_status(run_dir/'case_status.json',status); lines += cs.markdown_section(status)
    for path in sorted(run_dir.glob('wall*.json')):
        lines += ['', f'Wall time `{path.name}`:', '```json',path.read_text().rstrip(),'```']
    for group in groups.values():
        for row in group:
            if 'error' in row: lines += ['',f"Failed: {row['case_id']} process {row['process_index']}",'```',row['error'],'```']
    text = '\n'.join(lines)+'\n';(run_dir/'report.md').write_text(text)
    latest = ROOT/'result'/safe_target_profile(target)/'cpu/stateful_linalg.md';latest.parent.mkdir(parents=True,exist_ok=True);latest.write_text(text);print(latest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--threads',type=int)
    parser.add_argument('--report',type=Path);parser.add_argument('--target-profile',default='amd-cpu')
    args = parser.parse_args()
    if args.report: report(args.report.resolve(),args.target_profile)
    else: raise SystemExit(collect(args.binary,args.output,args.threads))
