import csv,json,re,statistics
from collections import defaultdict
from pathlib import Path
R=Path('data/results/amd-cpu/cpu'); rows=json.load(open('data/results/amd-cpu/cpu/slow_analysis/20261008_review/initial-csv-observations.json'))
issues=json.load(open('data/results/amd-cpu/cpu/slow_analysis/20261008_review/issue-index-before.json'))
issues.extend({'number':n,'title':t,'state':'OPEN','url':f'https://github.com/tensor4all/tenferro-rs/issues/{n}'} for n,t in [(2027,'Borrowed-session CPU lstsq gap'),(2028,'Rank-2 triangular_solve gap'),(2029,'Serial materializing reshape_read'),(2030,'Borrowed-session exact GELU CPU throughput gap'),(2031,'Borrowed-session len64 CPU softmax gap')])
index={r['number']:r for r in issues}
def owner(r):
 op=r['benchmark'];suite=r['suite']
 if op=='lstsq': return [2027],'focused'
 if op=='triangular_solve':return [2028],'focused'
 if op=='reshape' and suite!='cpu/view_metadata':return [2029],'focused'
 if op=='extract_diagonal' or op=='cast_f64_f32':return [1001,674],'historical cluster'
 if 'view_metadata' in suite:return [1719,1917],'historical cluster; #1917 is related host metadata'
 if op=='norm_fro':return [1505],'focused historical'
 if op=='chain_log1p_exp_mul':return [1990],'related fused-kernel issue (different chain)'
 if any(w in op for w in ('tanh','exp','log','sin','cos','pow','reduce_')):return [1719,1001],'historical cluster'
 if 'grad' in op:return [1328,1803],'AD cluster; #1803 primarily eager'
 if 'batched' in op or suite in ('batched','cpu/linalg_batch_families','cpu/linalg_batched'):
  return ([1884,1878,2000] if any(w in op for w in ('lu','solve')) else [1878,2000]),'batch-linalg cluster'
 if 'indexing' in suite:return [1482,1719,1512],'historical indexing cluster'
 if 'fft' in suite:return [1765,1484],'related historical; no focused large single-lane issue'
 if suite=='cpu/permutation':return [1393],'focused historical'
 if suite=='cpu/einsum':return [1865,1899,1904,975],'related contraction/planning cluster'
 if suite=='cpu/session_matrix':return [1904,1899,1946],'related public-route/session cluster'
 if suite=='cpu/structural_shape':return [1001,1512,1480],'related structural cluster'
 if 'output_reuse' in suite:return [1899,1904],'related public contraction path'
 if suite=='cpu/complex':return [1900,1660],'related provider issues; #1900 is Apple-specific'
 if suite=='cpu/linalg_uncovered':return [1956,1660],'related provider/extraction issues; no focused gap issue verified'
 if suite=='small':return [1904,1762],'small-call cluster'
 return [1927],'umbrella only; no focused gap issue verified'

def add(suite,op,dtype,t,shape,backend,ms,ref,refms,source,flag='unconfirmed observation',known=None):
 if ms>refms and refms>0:
  rows.append(dict(suite=suite,benchmark=op,dtype=dtype,threads=str(t),shape=str(shape),backend=backend,median_ms=ms,iqr_ms=None,reference=ref,reference_ms=refms,reference_iqr_ms=None,ratio=ms/refms,source=str(source),evidence=flag,known_issues=known))
# Permutation: already allocation-equivalent, validation occurs before timing.
for t in (1,4):
 ps=list((R/'permutation/20261008_062910').glob(f'*output_t{t}.jsonl'))
 rs=[json.loads(l) for p in ps for l in p.read_text().splitlines()];g=defaultdict(list)
 for r in rs:
  if r['status']=='ok':g[r['pattern_id']].append(r)
 for op,rs in g.items():
  refs=[r for r in rs if r['backend'] not in ('tenferro-rs','memcpy')]
  if not refs:continue
  ref=min(refs,key=lambda r:r['median_ms'])
  for r in rs:
   if r['backend']=='tenferro-rs':add('cpu/permutation',op,r['dtype'],t,r['shape'],r['backend'],r['median_ms'],ref['backend'],ref['median_ms'],ps[0])
# Perf cases have declared reference arms. Exclude explicitly named diagnostics.
for t in (1,4):
 p=R/f'perf_issues/20261008_063605/samples_t{t}.jsonl';rs=[json.loads(l) for l in p.read_text().splitlines()];by={r['case_id']:r for r in rs}
 def med(r):return statistics.median(s['elapsed_ns']/s['iterations'] for s in r['samples'])/1e6
 for ref in rs:
  cid=ref.get('reference_for');r=by.get(cid)
  if not r or r['backend']!='tenferro-rs' or ref['backend']=='tenferro-rs':continue
  if any(x['measurement_scope']!='steady_state' or x['correctness_status']!='passed' or not x['samples'] for x in (r,ref)):continue
  flag='unconfirmed observation'
  if ref['backend']=='pytorch-cpu':flag='screening only: original PyTorch retention-list growth timed'
  if ref['backend'] in ('faer-direct','matrixmultiply-sgemm'):flag='diagnostic: preallocated reference vs allocating tenferro'
  add('cpu/perf_issues',cid,r['dtype'],t,r['params'],r['arm'],med(r),ref['backend'],med(ref),p,flag,r['issues'])
# Session matrix compares identical workloads; different providers remain explicit.
for t in (1,4):
 p=R/f'session_matrix/20261008_062752/samples_t{t}.jsonl';rs=[json.loads(l) for l in p.read_text().splitlines()];g=defaultdict(list)
 for r in rs:
  if r.get('status')=='passed' and r.get('samples_ns'):g[r['case_id']].append(r)
 for op,rs in g.items():
  refs=[r for r in rs if r['provider']=='pytorch']
  if not refs:continue
  ref=refs[0];refms=statistics.median(ref['samples_ns'])/ref['operations_per_sample']/1e6
  for r in rs:
   if r['provider']=='pytorch':continue
   ms=statistics.median(r['samples_ns'])/r['operations_per_sample']/1e6
   add('cpu/session_matrix',op,r.get('dtype','f64; c64 named in case ID'),t,r.get('route',{}).get('workload_shape',r.get('n')),f"tenferro-{r['provider']}",ms,'pytorch',refms,p)
# Einsum table preserves strategy and all comparator backends.
for t,stamp in ((1,'20261008_044146'),(4,'20261008_053424')):
 p=R/f'einsum/{stamp}/einsum_table_t{t}_{stamp}.md';strategy='';headers=[]
 for line in p.read_text().splitlines():
  if line.startswith('#### Strategy:'):strategy=line.split(':',1)[1].strip()
  if not line.startswith('|'):continue
  cells=[c.strip() for c in line.strip('|').split('|')]
  if cells[0]=='Instance':headers=cells;continue
  if not headers or cells[0].startswith('---'):continue
  def val(s):
   s=s.replace('**','').replace('`','').strip()
   return None if s=='-' else float(s.split('±')[0].strip())
  pairs=[(h,val(c)) for h,c in zip(headers[1:],cells[1:])]
  refs=[(h,v) for h,v in pairs if not h.startswith('tenferro') and v is not None and v>0]
  if not refs:continue
  ref,minms=min(refs,key=lambda kv:kv[1])
  for h,v in pairs:
   if h.startswith('tenferro') and v is not None:add('cpu/einsum',cells[0]+' / '+strategy,'f64 (see instance)',t,'see instance JSON',h,v,ref,minms,p,'unconfirmed observation' if v>=1 else 'short isolated-clock diagnostic')
# Scope findings are read from the exact collected harness, not inferred from ratios.
valid_session={'norm_fro','cholesky','solve','eig','qr','svd','lu','pinv_with_rtol','pinv','inv','slogdet','det','triangular_solve','eigvalsh','eigvals'}
for r in rows:
 if 'evidence' not in r:
  r['evidence']='unconfirmed observation'
  if '/public_api/' in r['source']:
   if r['backend']=='tenferro-eager' or (r['backend']=='tenferro-direct' and r['suite'] not in ('cpu/view_metadata','cpu/linalg_batched','cpu/linalg_batch_families') and r['benchmark'] not in valid_session):r['evidence']='excluded: timed session entry'
   elif r['suite']!='cpu/view_metadata' and r['median_ms']<1:r['evidence']='short isolated-clock diagnostic'
  if r['backend']=='tenferro-fft-eager':r['evidence']='excluded: timed eager session entry'
  if r['suite']=='cpu/view_metadata' and (r['median_ms']-r['reference_ms'])*1e6<500:r['evidence']='sub-500ns absolute difference; metadata-only'
 nums,relation=owner(r)
 if r.get('known_issues'):nums=[int(n[1:]) for n in r['known_issues']];relation='explicit workload issue IDs'
 if r['suite']=='cpu/perf_issues' and '#1975' in r.get('known_issues',[]):
  relation='API feature request; no focused performance owner verified'
  if r['benchmark'].startswith('activation_gelu_f32_'):nums.append(2030);relation='API feature request plus confirmed focused performance issue #2030'
 if r['suite']=='cpu/perf_issues' and '#1976' in r.get('known_issues',[]):
  relation='API feature request; #2021 is related concrete-session materialization'
  if r['benchmark'].startswith('softmax_f32_') and '_len64_b8' in r['benchmark']:nums.append(2031);relation='API feature request plus confirmed focused performance issue #2031'
 r['issues']=' '.join(f'#{n}' for n in nums);r['issue_states']=' '.join(f'#{n}:{index[n]["state"]}' for n in nums if n in index);r['issue_match']=relation
 r['issue_urls']=' '.join(index[n]['url'] for n in nums if n in index)
rows.sort(key=lambda r:r['ratio'],reverse=True)
folder=R/'slow_analysis/20261008_review';folder.mkdir(parents=True,exist_ok=True)
fields=['suite','benchmark','dtype','threads','shape','backend','median_ms','iqr_ms','reference','reference_ms','reference_iqr_ms','ratio','issues','issue_states','issue_match','evidence','source','issue_urls']
with (folder/'slow-observations.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator="\n");w.writeheader();w.writerows(rows)
(folder/'issue-index.json').write_text(json.dumps(issues,ensure_ascii=False,indent=2)+'\n')
groups=defaultdict(list)
for r in rows:groups[(r['suite'],r['benchmark'],r['backend'])].append(r)
confirm=[]
for stamp in ('20261008_issues','20261008_extended','20261008_reshape_matched','20261008_activation_softmax','20261008_gelu_10ms','20261008_gelu_shape_matched'):
 for r in json.loads((R/f'slow_analysis/{stamp}/confirmation.json').read_text())['results']:
  if stamp=='20261008_extended' and r['operation']=='reshape':continue # superseded by logical-output-matched repro
  if stamp=='20261008_activation_softmax' and r['operation']=='gelu':continue # failed noise gate; superseded by 10 ms intervals
  if stamp=='20261008_gelu_10ms':continue # flattened reference superseded by matching logical rank/layout
  confirm.append(dict(r,source=str(R/f'slow_analysis/{stamp}')))
lines=['# Ryzen CPU slowdown observations and issue triage','', '- Source tenferro-rs: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`; Ryzen 9 9955HX; Linux MKL devcontainer; unpinned 1T/4T.', '- Issue search: all 1,173 pre-existing open/closed issues, plus focused operation searches, 2026-10-08.', f'- Inventory: {len(rows)} rows with tenferro median above the fastest recorded external comparator; {len(groups)} operation/path groups. This includes flagged diagnostics, not {len(rows)} confirmed defects.', '- Full inventory and per-row source/provenance: `data/results/amd-cpu/cpu/slow_analysis/20261008_review/slow-observations.csv`.', '', '## Confirmed MWE and negative controls','', 'These are current-host cross-implementation comparisons, not revision regressions. Inputs/setup/session entry/priming/cleanup are untimed; intrinsic output allocation is timed and outputs retained. Four balanced paired rounds follow same-build A/A. Immutable declarations precede the PyTorch samples. Rust/PyTorch use MKL family, with independently recorded vendor versions. Logical fixtures and requested outputs match; native layouts are preserved. The matched reshape control returns the same compact column-major [8192,4096] values.', '', '| Operation | Threads | Rust ms | PyTorch ms | Paired ratio | Verdict / disposition | Issue |','|---|---:|---:|---:|---:|---|---|']
for r in confirm:
 op=r['operation'];nums={'lstsq':2027,'triangular':2028,'reshape':2029,'diagonal':1001,'gelu':2030,'softmax':2031}.get(op)
 issue=f'[#{nums}]({index[nums]["url"]}) ({index[nums]["state"]})' if nums else 'No focused issue filed'
 disposition=r['verdict']
 if op=='ifft' and r['threads']==4:disposition='Borderline constant-input result; mixed round direction, not used for issue filing'
 lines.append(f"| {op} | {r['threads']} | {r['rust_ns']/1e6:.3f} | {r['pytorch_ns']/1e6:.3f} | {r['ratio']:.3f}x | {disposition} | {issue} |")
lines+=['', 'Raw declarations, A/A and paired sample JSON live in `data/results/amd-cpu/cpu/slow_analysis/20261008_{issues,extended,reshape_matched,activation_softmax,gelu_10ms,gelu_shape_matched}/`. The earlier extended reshape measurement had a different logical output shape; the matched-output run supersedes it. The original eager FFT/direct session-inclusive data are not used to establish operation gaps. Initial 2 ms GELU comparisons failed the declared noise gate; the final 10 ms logical-shape-matched rerun passed the same gate and supersedes those ratios and the earlier flattened-reference control. Feature/API-request IDs alone are not focused performance owners. Large IFFT and cast did not reproduce the original large ratios; no issue is filed from those observations.', '', '## Timing and issue-match limitations','', '- `excluded: timed session entry`: source inspection found public-API generic direct wrappers or eager entry inside the measured closure. Do not interpret these as operation-performance evidence; the standalone borrowed-session MWE is the replacement where available.', '- `short isolated-clock diagnostic`: the source samples one call per interval. Standard short-operation claims require batched intervals; these rows are candidates for reproduction, not confirmed throughput gaps.', '- Prepared trace rows have graph compilation/plan preparation outside timing, but runtime dispatch is part of that execution path; they are not immediate direct calls or borrowed eager calls. Candidate observations have not individually passed paired confirmation.', '- Existing closed issues remain closed in this inventory; the matching symptom does not prove the old cause/regression. Related/umbrella matches are explicitly weaker than focused ownership. Apple-specific #1900 is only a related lead for Ryzen complex GEMM, not a confirmed match.', '- References using preallocated destinations are flagged and are not allocation-equivalent comparisons. Julia uses OpenBLAS while tenferro uses MKL; provider-dependent comparisons do not isolate a tenferro implementation bottleneck.', '- Original `cpu/perf_issues` PyTorch sampling grew the retention list inside timing. The runner now preallocates it; historical ratios are screening observations, and only the separately confirmed MWEs establish the reported gaps.', '- `cpu/small_work` currently contains no external comparator in its recorded raw rows, so it cannot establish a cross-implementation slowdown here. Unsupported/failed/missing rows are never assigned performance ratios.', '', '## Complete operation/path inventory','', 'Each row below is the largest observed ratio in that operation/path group. CSV retains every dtype, shape, thread count and external comparator; ratios are descriptive observations until the evidence column says otherwise.', '', '| Suite | Operation / path | Worst observed ratio | Case | Reference | Issue matches | Evidence |','|---|---|---:|---|---|---|---|']
for key,rs in sorted(groups.items(),key=lambda kv:max(r['ratio'] for r in kv[1]),reverse=True):
 r=max(rs,key=lambda r:r['ratio']);links=' '.join(f'[{n}]({url})' for n,url in zip(r['issues'].split(),r['issue_urls'].split()))
 case=f"{r['dtype']}, t{r['threads']}, {r['shape']}".replace('|','/')
 lines.append(f"| {r['suite']} | `{r['benchmark']}` / {r['backend']} | {r['ratio']:.3f}x | {case} | {r['reference']} | {links}; {r['issue_states']}; {r['issue_match']} | {r['evidence']} |")
Path('result/amd-cpu/cpu/slow_cases.md').write_text('\n'.join(lines).rstrip()+'\n')
print(len(rows),'rows',len(groups),'groups')
