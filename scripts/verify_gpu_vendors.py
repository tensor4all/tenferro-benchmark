#!/usr/bin/env python3
"""Untimed CUDA checks for native/preferred vendor reference paths."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml
import benchmark_gpu_python as runner

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('GPU_BENCH_VERIFY_ONLY') != '1':
        parser.error('GPU_BENCH_VERIFY_ONLY=1 is required; this program collects no timings')
    os.chdir(ROOT)
    args.output.mkdir(parents=True,exist_ok=False)
    dense=yaml.safe_load((ROOT/'benchmarks/gpu/dense.yaml').read_text())
    sparse=yaml.safe_load((ROOT/'benchmarks/gpu/sparse.yaml').read_text())
    probes=[]
    for op,backends in [('matmul',['cutlass','cublaslt']),('batched_matmul',['cutlass']),('qr',['cusolver'])]:
        case=copy.deepcopy(next(p for p in dense['problems'] if p['op']==op))
        case['id']='vendor_verify_'+op
        dimensions=case['linalg'] if op=='qr' else case[op]
        dimensions.update(m=32,n=32)
        if op in ('matmul','batched_matmul'):dimensions['k']=32
        if op=='batched_matmul':dimensions['batch']=16
        case['run'].update(warmups=0,runs=3)
        for backend in backends:probes.append((dense,case,backend))
    for op in ('spmv','spmm'):
        case=copy.deepcopy(next(p for p in sparse['problems'] if p['op']==op))
        case['id']='vendor_verify_'+op
        case['sparse'].update(rows=64,cols=64,nnz=256,name='debug')
        if op=='spmm':case['sparse']['rhs_cols']=8
        case['run'].update(warmups=0,runs=3)
        for backend in ('pytorch-cuda','cusparse','ginkgo'):probes.append((sparse,case,backend))
    (args.output/'manifest.json').write_text(json.dumps([dict(problem=p,backend=b) for _,p,b in probes],indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/collect_run_metadata.py'),'--suite-id','gpu/vendor_verification',
        '--suite-file',str(args.output/'manifest.json'),'--target-profile','nvidia-gpu',
        '--timestamp',datetime.now(timezone.utc).isoformat(),'--tenferro-dir',str(ROOT/'extern/tenferro-rs'),
        '--features','cuda','--blas','none','--cuda-device-ordinal',os.environ.get('GPU_BENCH_PHYSICAL_DEVICE','0'),
        '--output',str(args.output/'run.yaml')],check=True)
    failed=[]
    with (args.output/'records.jsonl').open('w') as output:
        for suite,problem,backend in probes:
            row=runner._run_one(suite['suite_id'],problem,backend,0,suite_backends=suite['backends'],
                ts=datetime.now(timezone.utc).isoformat(),bc=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                tc=subprocess.check_output(['git','-C','extern/tenferro-rs','rev-parse','HEAD'],text=True).strip())
            output.write(json.dumps(row)+'\n');output.flush()
            passed=(row['status']=='verification_only' and row['verification']['status']=='passed'
                    and row['timing']['timed_runs']==0 and row['timing']['median_ms'] is None)
            print(problem['id'],backend,'passed' if passed else row,flush=True)
            if not passed:failed.append((problem['id'],backend))
    if failed:raise SystemExit(f'failed vendor checks: {failed}')


if __name__=='__main__':main()
