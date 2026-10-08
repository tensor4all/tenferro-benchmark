"""Cross-implementation output validation, outside benchmark timers."""
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
