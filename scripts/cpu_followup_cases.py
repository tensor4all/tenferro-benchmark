"""Execute registered latest-main public CPU MWEs with a matched reference."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mwe/cpu_followup'))
from validation import verify

# The collector and the builder must agree on one path: scripts/run_cpu_followup.sh
# builds into $PWD/target/followup-current with this directory as its target dir.
DEFAULT_MWE_BINARY=ROOT/'target/followup-current/release/cpu-followup-mwe'


def mwe_binary():
    """The Rust MWE binary, honoring the CPU_FOLLOWUP_BIN override."""
    return Path(os.environ.get('CPU_FOLLOWUP_BIN',str(DEFAULT_MWE_BINARY)))


def mwe_build_hint(binary):
    return (f'CPU follow-up MWE binary is missing: {binary}. Build it with '
            f'`./scripts/run_cpu_followup.sh --prepare-mwe` (or '
            f'`CARGO_TARGET_DIR="$PWD/target/followup-current" cargo build --release '
            f'--manifest-path mwe/cpu_followup/Cargo.toml`); the nightly path builds it '
            f'through `./scripts/run_cpu_followup.sh --prepare-mwe`. '
            f'`CPU_FOLLOWUP_BIN` overrides the location.')


def followup_mwe_supported():
    """mwe/cpu_followup's default feature is system-mkl, so it builds on Linux only."""
    return sys.platform=='linux'


def resolve_julia():
    """Honor an explicit JULIA; otherwise search PATH, then juliaup's default."""
    override = os.environ.get('JULIA')
    if override:
        executable = shutil.which(override)
        if executable:
            return executable
        raise RuntimeError(f'julia not found: JULIA={override!r} is not executable; set JULIA or install juliaup')
    executable = shutil.which('julia')
    if executable:
        return executable
    juliaup = Path.home() / '.juliaup/bin/julia'
    if juliaup.is_file() and os.access(juliaup, os.X_OK):
        return str(juliaup)
    raise RuntimeError('julia not found: set JULIA or install juliaup')


def command(case_id, path, threads, runs):
    if path=='julia-base':
        return [resolve_julia(), '--project=.',
                str(ROOT/'mwe/cpu_followup/reference.jl'),case_id,str(threads),str(runs)]
    if path in ('pytorch','jax','mkl-dfti'):
        return [sys.executable,str(ROOT/'mwe/cpu_followup/reference.py'),case_id,path,str(threads),str(runs)]
    return [str(mwe_binary()),case_id,path,str(threads),str(runs)]


def execute(case_id,path,threads,runs,target_ns):
    invocation=command(case_id,path,threads,runs)
    env=dict(os.environ,CPU_FOLLOWUP_TARGET_NS=str(target_ns))
    try:
        result=subprocess.run(invocation,cwd=ROOT,env=env,capture_output=True,text=True)
    except FileNotFoundError as error:
        # A bare FileNotFoundError once cost 94 rows x 2 thread counts of triage:
        # name the missing artifact and the command that produces it.
        if invocation[0]==str(mwe_binary()):
            raise RuntimeError(mwe_build_hint(Path(invocation[0]))) from error
        raise
    if result.returncode:
        raise RuntimeError(f'{path}: {result.stderr.strip()}')
    row=json.loads(result.stdout.strip().splitlines()[-1])
    row['status']='skipped' if row.get('correctness_status')=='skipped' else 'ok'
    return row


def run(case,threads,config,correctness_only=False):
    if not followup_mwe_supported():
        # Both nightly profiles run this suite; only the Linux one can build the
        # MWE, so the other records a skip instead of a failure.
        return dict(correctness_status='skipped',samples=[],
                    skip_reason='cpu_followup MWE needs the Linux system-MKL build; '
                                f'skipped on {sys.platform}')
    p=case.get('followup_mwe',case['params']); cid=p['case_id'];path=p.get('path',case['arm']);target=int(config['min_runtime_ms']*1_000_000)
    # Two independent completed operations establish correctness before this
    # arm's collection. Fixture creation, validation and processes are untimed.
    reference=execute(cid,p['reference_path'],threads,0,target)
    if reference['status']=='skipped':
        return dict(correctness_status='skipped',samples=[],skip_reason=reference['skip_reason'])
    rust=execute(cid,p['rust_path'],threads,0,target)
    valid,reason=verify(rust,reference)
    if not valid:
        return dict(correctness_status='failed',samples=[],error=reason)
    row=execute(cid,path,threads,0 if correctness_only else config['runs'],target)
    if row['status']=='skipped':
        return dict(correctness_status='skipped',samples=[],skip_reason=row['skip_reason'])
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
    selected=[c for c in cases if (c['kind']=='cpu_followup_mwe' or 'followup_mwe' in c) and (not args.case or c.get('followup_mwe',c['params'])['case_id'] in args.case)]
    if not selected:raise SystemExit('No registered follow-up cases selected')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    failed=False
    with args.output.open('w') as stream:
        for c in selected:
            try:
                row=run(c,args.threads,dict(runs=args.runs,min_runtime_ms=10),args.runs==0)
            except Exception as error:
                row=dict(correctness_status='failed',samples=[],error=str(error))
            failed |= row['correctness_status'] not in ('passed','skipped')
            record=dict(c);record.update(row);record.update(case_id=c['id'],threads=args.threads)
            stream.write(json.dumps(record)+'\n');stream.flush()
            print(c['id'],row['correctness_status'],flush=True)
    raise SystemExit(int(failed))


if __name__=='__main__':main()
