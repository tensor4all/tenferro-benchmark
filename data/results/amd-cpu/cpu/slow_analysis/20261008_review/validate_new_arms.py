import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,'scripts')
from benchmark_perf_issues import run_torch
cases=[r for r in json.loads(Path('data/instances/perf_issues.json').read_text()) if r['kind']=='cpu_gap_mwe']
assert len(cases)==6
for c in cases:
 if c['backend']=='pytorch-cpu':
  r=run_torch(c,1,{'warmups':0},True);assert r['correctness_status']=='passed' and r['max_rel_error'] is None and not r['samples']
 else:
  args=['target/compatibility-mkl/release/perf_issue_case','--case',c['id'],'--params',json.dumps(c['params']),'--threads','1','--mode','correctness-only','--warmups','0']
  r=json.loads(subprocess.check_output(args,text=True));assert r['correctness_status']=='passed' and not r['samples'] and r['calibration']['iterations']==0
 print(c['id'],'validated')
