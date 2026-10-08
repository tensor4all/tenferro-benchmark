"""Untimed runtime/compiled-route observation; not a performance result."""
import collections,json,os,runpy,sys
from pathlib import Path
case_id,backend,threads=sys.argv[1:4]
sys.argv=['mwe/cpu_followup/reference.py',case_id,backend,threads,'0']
ns=runpy.run_path(sys.argv[0],run_name='__main__')
names=collections.Counter()
for path in Path('/proc/self/task').glob('*/comm'):
 try:names[path.read_text().strip()]+=1
 except FileNotFoundError:pass
row=dict(record_type='untimed runtime observation',case_id=case_id,backend=backend,requested_threads=int(threads),thread_names=dict(names),affinity=sorted(os.sched_getaffinity(0)),environment={k:os.environ.get(k) for k in ('PJRT_NPROC','XLA_FLAGS','RAYON_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','OMP_THREAD_LIMIT')})
if backend=='jax':
 row['eigen_worker_count']=names.get('tf_XLAEigen');row['devices']=[str(x) for x in ns['jax'].devices()]
 hlo=ns['compiled'].lower(*ns['inputs']).compile().as_text()
 row['optimized_hlo']=hlo
else:
 row['intraop_threads']=ns['torch'].get_num_threads();row['interop_threads']=ns['torch'].get_num_interop_threads();row['device']=str(ns['x'].device)
print(json.dumps(row))
