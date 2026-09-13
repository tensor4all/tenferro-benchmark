import csv, pathlib, subprocess, os, collections, math
root=pathlib.Path('.')
run=pathlib.Path('data/results/mac-cpu/cpu/cpu_ops/20260913_130307')
keys=('suite','benchmark','dtype','threads','shape','backend')
def read(p): return list(csv.DictReader(p.open()))
def key(r): return tuple(r[k] for k in keys)
def merge(base, patch, dest):
 old=read(base); new=read(patch) if isinstance(patch,pathlib.Path) else patch
 mapping={key(r):r for r in new}
 assert len(mapping)==len(new)
 assert set(mapping)<=set(map(key,old))
 assert all(r['status']=='ok' and math.isfinite(float(r['median_ms'])) for r in new)
 rows=[{**r,**mapping[key(r)]} if key(r) in mapping else r for r in old]
 assert len(rows)==len({key(r) for r in rows})
 with dest.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(old[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
 print(dest,len(rows),'replaced',len(new))
 return dest
ops=[];views=[]
for t,stamp in [(1,'20260913_123040'),(4,'20260913_124035')]:
 normalized=[]
 for r in read(run/f'eager_t{t}.csv'):
  normalized.append(dict(suite=r['suite'],benchmark=r['op']+('_'+r['phase'] if r['phase']!='primal' else ''),dtype=r['dtype'],threads=str(t),shape=r['shape'],backend='tenferro-eager',median_ms=r['median_ms'],iqr_ms=r['iqr_ms'],status=r['status']))
 assert len(normalized)==194
 ops.append(merge(pathlib.Path(f'data/results/mac-cpu/cpu/einsum/{stamp}/cpu_ops_t{t}_{stamp}.csv'),normalized,run/f'cpu_ops_merged_t{t}.csv'))
 views.append(merge(pathlib.Path(f'data/results/mac-cpu/cpu/public_api/20260913_124711/cpu_public_api_t{t}_20260913_124711.csv'),run/f'views_confirmed_t{t}.csv',run/f'public_api_merged_t{t}.csv'))
def old_report(name):return subprocess.check_output(['git','show',f'19a4fc1:result/mac-cpu/cpu/{name}.md'],text=True)
def table(paths,direct=False):
 env=dict(os.environ)
 if direct:env['CPU_FORMAT_TENFERRO_EAGER_LABEL']='tenferro-rs direct API (ms)'
 return subprocess.check_output(['uv','run','python','scripts/format_cpu_ops_results.py',*map(str,paths)],env=env,text=True)
provenance=f'''## Targeted timing correction

- Refresh: `20260913_130307`, Apple M5 Max, 1 and 4 threads; tenferro `a48866b1a0bb52e6f9712c485925105b14ea30b9` (unchanged).
- Collection commands: `{run}/collection.sh` and `{run}/view_recheck.sh`.
- Sampling: 3 warmups and 15 measured runs; sequential collection with the idle-host guard enabled.
- Raw corrected rows and per-thread metadata: `{run}/`.
- All unselected cells are retained verbatim from the original runs named below; the tables combine those original measurements with this targeted correction.

'''
prefix=old_report('cpu_ops').split('## Threads: 4')[0]
prefix=prefix.replace('Latest run:','Original baseline run:').replace('This file is generated from one CPU ops run under','Original 4-thread baseline is under')
note=provenance+'''- Replaced only tenferro-eager CPU ops rows (194 per thread). Runtime initialization occurs before sampling and the runtime is shared. Input construction, eager wrapping, operation execution and output consumption remain timed, as in the original suite.
- Baselines: `data/results/mac-cpu/cpu/einsum/20260913_123040/` (1T) and `data/results/mac-cpu/cpu/einsum/20260913_124035/` (4T). Trace and Python cells were not rerun.

'''
(root/'result/mac-cpu/cpu/cpu_ops.md').write_text(prefix+note+table(ops))
prefix=old_report('public_api').split('## CPU Benchmark Items')[0]
prefix=prefix.replace('Latest run:','Original baseline run:').replace('This file is generated from sequential CPU public API runs under','Original baseline runs are under')
note=provenance+'''- Replaced only four tenferro direct `cpu/view_metadata` cells per thread. Input duplication is outside timing; outputs retain their allocation until after the clock stops. These cells measure concrete TensorValue view operations, not EagerTensor dispatch.
- Baseline for all other cells: `data/results/mac-cpu/cpu/public_api/20260913_124711/`.
- View-only confirmation was collected after the first run; both raw runs are retained. Sub-microsecond single-call samples have substantial relative timer/scheduling variability; use the reported IQR and avoid precise speedup claims from these cells.

'''
(root/'result/mac-cpu/cpu/public_api.md').write_text(prefix+note+table(views,True))
