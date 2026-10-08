"""Sequential #2040 paired comparison at the maintainer-requested pinned revision."""
import argparse, datetime, gzip, json, shlex, statistics, subprocess, time
from pathlib import Path
from validation import verify
ROOT = Path(__file__).resolve().parents[2]
ARMS = ['julia-base', 'metadata', 'metadata-host-dyn', 'metadata-host-static']
CASES = ['metadata.' + x for x in ('reshape_view', 'slice_view', 'transpose_view')]

def timing(row):
    xs = [s['elapsed_ns']/s['iterations'] for s in row['samples']]
    return dict(median_ns=statistics.median(xs), cov=statistics.pstdev(xs)/statistics.mean(xs))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--container', default='14099c68ff8e')
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    lib = subprocess.check_output(['git','-C',str(ROOT/'extern/tenferro-rs'),'rev-parse','HEAD'],text=True).strip()
    assert lib == 'b3f47296244ff7b7c55ac0a75f782cb0835418c1'
    harness = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    declaration = dict(suite_id='cpu/metadata_host',library_commit=lib,harness_commit=harness,
        declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), threads=[1], cases=CASES, arms=ARMS,
        runs=15,warmups=3,target_ns=50000000,memory_cap_bytes=536870912,max_operations=2000000,
        order=[ARMS,list(reversed(ARMS)),list(reversed(ARMS)),ARMS], aa_order=['AB','BA','BA','AB'],
        thresholds=dict(relative_ratio=1.2,max_cov=.2,max_aa_pair_deviation=.1),
        scope='pure metadata API calls plus intrinsic output descriptor allocation; no session; setup, input views, owner allocation and destruction untimed; all outputs retained',
        lifetime_contract=dict(metadata='consumes owned dtype-erased DynRank TensorValue',
            host='borrows typed Host view; each separately retained owner outlives input/output views',
            static='Rank<1> slice; Rank<2> transpose; Rank<1> reshape input returns DynRank'),
        affinity='no external pinning',aa_control='each Rust arm independently, same representation on both sides',
        statistic='median of four paired round-median ratios; every AA pair must pass the predeclared bound')
    dp=out/'declaration.json'
    if dp.exists():
        old=json.loads(dp.read_text()); assert old['harness_commit']==harness and old['library_commit']==lib
        declaration=old
    else: dp.write_text(json.dumps(declaration,indent=2)+'\n')
    def arm(case,path,name):
        dest=out/(name+'.json')
        if dest.exists(): return json.loads(dest.read_text())
        cmd=(['/home/vscode/.juliaup/bin/julia','--project=.','mwe/cpu_followup/reference.jl',case,'1','15'] if path=='julia-base' else
            ['target/followup-b3f47296/release/cpu-followup-mwe',case,path,'1','15'])
        shell=('set -e; source scripts/thread_env.sh; configure_cpu_thread_env 1; '
            'source scripts/benchmark_host_idle.sh; assert_benchmark_host_idle; '
            'export CPU_FOLLOWUP_TARGET_NS=50000000; '
            'export LD_LIBRARY_PATH="$MKLROOT/lib:/opt/intel/oneapi/compiler/2026.1/lib:${LD_LIBRARY_PATH:-}"; '+shlex.join(cmd))
        command=['docker','exec','-u','vscode','-w','/workspaces/tenferro-benchmark',a.container,'bash','-lc',shell]
        subprocess.run(['bash','-lc','source scripts/benchmark_host_idle.sh; assert_benchmark_host_idle'],cwd=ROOT,check=True)
        start=time.time(); r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        (out/(name+'.stderr')).write_text(r.stderr)
        if r.returncode: raise RuntimeError(r.stderr)
        row=json.loads(r.stdout.strip().splitlines()[-1]);row.update(status='ok',command=command,library_commit=lib,harness_commit=harness,wall_seconds=time.time()-start)
        dest.write_text(json.dumps(row,indent=2)+'\n'); print(name,round(row['wall_seconds'],1),flush=True)
        return row
    results=[]
    for case in CASES:
        controls={}
        for path in ARMS[1:]:
            pairs=[]
            for rnd,order in enumerate(declaration['aa_order']):
                rows={label:arm(case,path,f'{case}-{path}-aa{rnd}-{label}') for label in order}
                pairs.append({label:timing(row) for label,row in rows.items()})
            controls[path]=dict(ratios=[p['A']['median_ns']/p['B']['median_ns'] for p in pairs],
                max_cov=max(x['cov'] for p in pairs for x in p.values()))
        rounds=[]; correct=True
        for rnd,order in enumerate(declaration['order']):
            rows={path:arm(case,path,f'{case}-pair{rnd}-{path}') for path in order}
            for path in ARMS[1:]:
                ok,why=verify(rows[path],rows['julia-base']);assert ok,why
                correct &= ok
            rounds.append({path:timing(row) for path,row in rows.items()})
        for path in ARMS[1:]:
            control=controls[path]
            aa_spread=max(abs(x-1) for x in control['ratios'])
            cov=max(control['max_cov'],max(r[p]['cov'] for r in rounds for p in [path,'julia-base']))
            ratios=[r[path]['median_ns']/r['julia-base']['median_ns'] for r in rounds]
            ratio=statistics.median(ratios)
            status=('confirmed-slower' if ratio>=1.2 else 'below-threshold') if correct and aa_spread<=.1 and cov<=.2 else 'inconclusive'
            results.append(dict(case_id=case,path=path,median_ns=statistics.median(r[path]['median_ns'] for r in rounds),
                julia_ns=statistics.median(r['julia-base']['median_ns'] for r in rounds),ratio=ratio,round_ratios=ratios,
                owned_over_arm_round_ratios=[r['metadata']['median_ns']/r[path]['median_ns'] for r in rounds],
                aa_ratios=control['ratios'],max_aa_deviation=aa_spread,max_cov=cov,correct=correct,status=status))
        (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results[-3:],indent=2),flush=True)
    # Lossless original process JSON, with a small tracked footprint.
    with gzip.open(out/'processes.jsonl.gz','wt',encoding='utf-8') as packed:
        for f in sorted(out.glob('*.json')):
            if f.name not in ('declaration.json','summary.json'):
                packed.write(json.dumps(dict(file=f.name,content=f.read_text()))+'\n')

if __name__ == '__main__': main()
