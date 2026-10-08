"""Sequential devcontainer collection; immutable declaration precedes all timings."""
import argparse, datetime, json, os, shlex, statistics, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CASES=json.loads(Path(__file__).with_name('cases.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--container',default='14099c68ff8e');p.add_argument('--output',required=True);p.add_argument('--phase',choices=['scan','confirm'],default='scan');p.add_argument('--case',action='append');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
lib=subprocess.check_output(['git','-C',str(ROOT/'extern/tenferro-rs'),'rev-parse','HEAD'],text=True).strip()
harness=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
declaration=out/'declaration.json'
if not declaration.exists():
 declaration.write_text(json.dumps(dict(status='declared',declared_before_candidate_results=datetime.datetime.now(datetime.timezone.utc).isoformat(),library_commit=lib,harness_commit=harness,threads=[1,4],suite_id='cpu/followup_12x',cases=[c['id'] for c in CASES],scope='setup untimed; borrowed direct/eager session; pure metadata; prepared trace internal session explicitly diagnostic',provider='tenferro MKL / PyTorch wheel MKL; JAX XLA; Julia own BLAS; strided-rs Rayon',affinity='none',scan_runs=3,confirmation_runs=15,warmups=3,confirmation_rounds=4,order=['AB','BA','BA','AB'],statistic='median of paired round median ratios',thresholds={'relative':.2,'absolute_ns':0},noise={'max_cov':.2,'max_aa_relative_spread':.1,'host_idle_guard':'enabled'},batch_target_ns=10000000,retained_memory_limit_bytes=536870912),indent=2)+'\n')
config=json.loads(declaration.read_text());assert config['library_commit']==lib and config['harness_commit']==harness, 'Source changed during campaign'

def arm(case,path,threads,runs,name):
 dest=out/(name+'.json')
 if dest.exists():return json.loads(dest.read_text())
 if path=='julia-base':cmd=['/home/vscode/.juliaup/bin/julia','--project=.','mwe/cpu_followup/reference.jl',case['id'],str(threads),str(runs)]
 elif path in ('pytorch','jax'):cmd=['.venv/bin/python','mwe/cpu_followup/reference.py',case['id'],path,str(threads),str(runs)]
 else:cmd=['target/followup-b3f47296/release/cpu-followup-mwe',case['id'],path,str(threads),str(runs)]
 shell='set -e; source scripts/thread_env.sh; configure_cpu_thread_env '+str(threads)+'; source scripts/benchmark_host_idle.sh; assert_benchmark_host_idle; export LD_LIBRARY_PATH="$MKLROOT/lib:/opt/intel/oneapi/compiler/2026.1/lib:${LD_LIBRARY_PATH:-}"; '+shlex.join(cmd)
 command=['docker','exec','-u','vscode','-w','/workspaces/tenferro-benchmark',a.container,'bash','-lc',shell]
 # The external host guard runs before Docker and cannot see the newly started own arm.
 guard=subprocess.run(['bash','-lc','source scripts/benchmark_host_idle.sh; assert_benchmark_host_idle'],cwd=ROOT,capture_output=True,text=True)
 if guard.returncode:raise RuntimeError(guard.stderr)
 started=datetime.datetime.now(datetime.timezone.utc).isoformat();start=time.time();r=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
 (out/(name+'.stderr')).write_text(r.stderr)
 if r.returncode:
  row=dict(status='failed',command=command,returncode=r.returncode,stderr=r.stderr)
 else:
  try:row=json.loads(r.stdout.strip().splitlines()[-1]);row['status']='ok'
  except Exception:row=dict(status='failed',stdout=r.stdout,stderr=r.stderr)
 row['command']=command;row['started_utc']=started;row['library_commit']=lib;row['harness_commit']=harness;row['wall_seconds']=time.time()-start
 dest.write_text(json.dumps(row,indent=2)+'\n');print(name,row['status'],round(row['wall_seconds'],1),flush=True)
 return row

def timing(r):
 xs=[s['elapsed_ns']/s['iterations'] for s in r.get('samples',[])]
 return dict(median_ns=statistics.median(xs),cov=statistics.pstdev(xs)/statistics.mean(xs)) if xs else None

def verify(x,y):
 if x.get('status')!='ok' or y.get('status')!='ok':return False,'arm failed'
 left,right=x['outputs'],y['outputs']
 if len(left)!=len(right):return False,'output count differs'
 for u,v in zip(left,right):
  if u.get('kind')=='metadata':
   if any(u[k]!=v[k] for k in ('shape','strides','offset','metadata_only')):return False,'metadata differs'
   continue
  if u['shape']!=v['shape'] or u['count']!=v['count']:return False,'output shape differs'
  # Probe tolerance follows dtype, with scaled aggregate checks for cancellation.
  tol=3e-4 if '.fft' in x.get('case_id','') or x.get('case_id','').startswith(('activation.','fft.')) else 1e-9
  scale=max(u['sum_abs'],v['sum_abs'],1)
  for aa,bb in zip(u['sum'],v['sum']):
   if abs(aa-bb)>tol*scale:return False,'aggregate sum differs'
  for k in ('sum_abs','sum_sq'):
   if abs(u[k]-v[k])>tol*max(abs(u[k]),abs(v[k]),1):return False,k+' differs'
  for aa,bb in zip(u['probes'],v['probes']):
   if aa[0]!=bb[0] or any(abs(c-d)>tol*max(1,abs(c),abs(d)) for c,d in zip(aa[1:],bb[1:])):return False,'probe differs'
 return True,'matched aggregates and 130 deterministic probes; metadata descriptors exact'

def reference(c):
 if c['category']=='linalg':return 'pytorch'  # MKL-matched primary linalg comparison
 if c['category']=='perm':return 'strided-rs'
 if c['category']=='metadata' or c['source_reference']=='julia-base':return 'julia-base'
 if c['id']=='index.dynamic_update_slice':return 'jax'
 return 'jax' if c['source_reference']=='jax-cpu' else 'pytorch'

if a.phase=='scan':
 results=[]
 for c in CASES:
  if a.case and c['id'] not in a.case:continue
  for t in (1,4):
   ref=arm(c,reference(c),t,3,f"scan-{c['id']}-t{t}-ref")
   for path in c['paths']:
    # Trace sessions are intrinsic API-call diagnostics and are not eligible for
    # standard operation evidence under AGENTS.md. Retain an explicit disposition.
    if path=='prepared-trace-call':
     results.append(dict(case_id=c['id'],threads=t,path=path,status='scope-ineligible',reason='no public borrowed trace execution-session API; do not mix internal session setup into operation timing'));continue
    rust=arm(c,path,t,3,f"scan-{c['id']}-t{t}-{path}")
    ok,reason=verify(rust,ref);rt,ft=timing(rust),timing(ref)
    row=dict(case_id=c['id'],threads=t,path=path,reference=reference(c),correct=ok,reason=reason,status='screened' if ok else 'invalid',rust=rt,ref=ft)
    if ok and rt and ft:row['ratio']=rt['median_ns']/ft['median_ns']
    results.append(row)
  (out/'scan.json').write_text(json.dumps(results,indent=2)+'\n')
else:
 scan=json.loads((out/'scan.json').read_text());results=[]
 for c in CASES:
  if a.case and c['id'] not in a.case:continue
  candidates=[r for r in scan if r['case_id']==c['id'] and r.get('correct') and r.get('ratio',0)>=1.2]
  if not candidates:continue
  selected=max(candidates,key=lambda r:r['ratio']);t=selected['threads'];path=selected['path'];refpath=reference(c)
  prefix=f"confirm-{c['id']}-t{t}"
  aa=[]
  for rnd in range(4):
   first=arm(c,path,t,15,f'{prefix}-aa{rnd}-A');second=arm(c,path,t,15,f'{prefix}-aa{rnd}-B')
   aa.append((timing(first),timing(second)))
  paired=[];correct=True;why=[]
  for rnd,order in enumerate(config['order']):
   arms={}
   for label in order:arms[label]=arm(c,path if label=='A' else refpath,t,15,f'{prefix}-pair{rnd}-{label}')
   ok,reason=verify(arms['A'],arms['B']);correct &=ok;why.append(reason)
   paired.append((timing(arms['A']),timing(arms['B'])))
  valid=all(x and y for x,y in aa+paired)
  if valid:
   ratios=[x['median_ns']/y['median_ns'] for x,y in paired];aa_ratio=statistics.median(x['median_ns']/y['median_ns'] for x,y in aa)
   cov=max(r['cov'] for pair in aa+paired for r in pair);ratio=statistics.median(ratios)
   status='confirmed' if correct and ratio>=1.2 and abs(aa_ratio-1)<=.1 and cov<=.2 else 'below-threshold' if correct and abs(aa_ratio-1)<=.1 and cov<=.2 else 'inconclusive'
   row=dict(case_id=c['id'],threads=t,path=path,reference=refpath,status=status,ratio=ratio,round_ratios=ratios,aa_ratio=aa_ratio,max_cov=cov,correct=correct,validation=why,tenferro_ns=statistics.median(x['median_ns'] for x,y in paired),reference_ns=statistics.median(y['median_ns'] for x,y in paired))
  else:row=dict(case_id=c['id'],status='failed',threads=t,path=path)
  results.append(row);(out/'confirm.json').write_text(json.dumps(results,indent=2)+'\n');print('VERDICT',json.dumps(row),flush=True)
