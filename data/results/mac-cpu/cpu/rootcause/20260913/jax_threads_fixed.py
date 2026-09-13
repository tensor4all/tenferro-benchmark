import os,time,json,statistics
import jax,jax.numpy as jnp,numpy as np
jax.config.update('jax_enable_x64',True)
for case,shape in [('tanh',(8388608,)),('max',(2048,2048))]:
 x=jnp.asarray(((np.arange(np.prod(shape))%1009)+1).reshape(shape).astype(np.float64)/1009)
 f=jax.jit(jnp.tanh if case=='tanh' else lambda x:jnp.max(x,axis=0))
 c=f.lower(x).compile(); txt=c.as_text(); open('/tmp/tenferro-rootcause/jax_'+case+'_nproc'+os.environ['PJRT_NPROC']+'.hlo','w').write(txt)
 for _ in range(10): c(x).block_until_ready()
 ns=[]; cpu=time.process_time(); wall=time.perf_counter()
 for _ in range(100):
  s=time.perf_counter_ns(); out=c(x); out.block_until_ready(); ns.append(time.perf_counter_ns()-s); del out
 cpu=time.process_time()-cpu;wall=time.perf_counter()-wall
 print(json.dumps(dict(case=case,threads=os.environ['OMP_NUM_THREADS'],median_ms=statistics.median(ns)/1e6,cpu_per_wall=cpu/wall,cpu_s=cpu,wall_s=wall)),flush=True)
