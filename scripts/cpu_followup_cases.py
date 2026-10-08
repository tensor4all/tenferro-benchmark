"""Execute registered latest-main public CPU MWEs with a matched reference."""
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mwe/cpu_followup'))
from validation import verify


def command(case_id, path, threads, runs):
    if path=='julia-base':
        return [os.environ.get('JULIA', '/home/vscode/.juliaup/bin/julia'), '--project=.',
                str(ROOT/'mwe/cpu_followup/reference.jl'),case_id,str(threads),str(runs)]
    if path in ('pytorch','jax'):
        return [sys.executable,str(ROOT/'mwe/cpu_followup/reference.py'),case_id,path,str(threads),str(runs)]
    return [os.environ.get('CPU_FOLLOWUP_BIN',str(ROOT/'target/followup-b3f47296/release/cpu-followup-mwe')),
            case_id,path,str(threads),str(runs)]


def execute(case_id,path,threads,runs,target_ns):
    env=dict(os.environ,CPU_FOLLOWUP_TARGET_NS=str(target_ns))
    result=subprocess.run(command(case_id,path,threads,runs),cwd=ROOT,env=env,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(f'{path}: {result.stderr.strip()}')
    row=json.loads(result.stdout.strip().splitlines()[-1]);row['status']='ok'
    return row


def run(case,threads,config,correctness_only=False):
    p=case['params']; cid=p['case_id'];path=case['arm'];target=int(config['min_runtime_ms']*1_000_000)
    # Two independent completed operations establish correctness before this
    # arm's collection. Fixture creation, validation and processes are untimed.
    rust=execute(cid,p['rust_path'],threads,0,target)
    reference=execute(cid,p['reference_path'],threads,0,target)
    valid,reason=verify(rust,reference)
    if not valid:
        return dict(correctness_status='failed',samples=[],error=reason)
    row=execute(cid,path,threads,0 if correctness_only else config['runs'],target)
    row.update(correctness_status='passed',validation=reason,warmups=3,
               correctness_method='matched output shape, aggregates and deterministic probes')
    # Preserve the suite identity. The standalone binary emits its own case ID.
    row.pop('case_id',None);row.pop('status',None)
    return row


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,required=True)
    parser.add_argument('--runs',type=int,default=15)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',action='append')
    args=parser.parse_args()
    cases=json.loads((ROOT/'data/instances/perf_issues.json').read_text())
    selected=[c for c in cases if c['kind']=='cpu_followup_mwe' and (not args.case or c['params']['case_id'] in args.case)]
    if not selected:raise SystemExit('No registered follow-up cases selected')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    failed=False
    with args.output.open('w') as stream:
        for c in selected:
            try:
                row=run(c,args.threads,dict(runs=args.runs,min_runtime_ms=10),args.runs==0)
            except Exception as error:
                row=dict(correctness_status='failed',samples=[],error=str(error))
            failed |= row['correctness_status']!='passed'
            stream.write(json.dumps(dict(c,**row,case_id=c['id'],threads=args.threads))+'\n');stream.flush()
            print(c['id'],row['correctness_status'],flush=True)
    raise SystemExit(int(failed))

if __name__=='__main__':main()
