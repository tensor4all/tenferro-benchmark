import os,time,json,statistics
import numpy as np, torch, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
for n in (2,32,512):
 a=torch.ones((n,n),dtype=torch.float64); b=a.clone()
 for _ in range(20): out=a@b
 times=[]
 cpu=time.process_time(); main=time.thread_time()
 for _ in range(20000 if n<100 else 100):
  t=time.perf_counter_ns(); out=a@b; times.append(time.perf_counter_ns()-t); del out
 proc=time.process_time()-cpu; caller=time.thread_time()-main
 print(json.dumps(dict(kind='torch_mm',n=n,threads=torch.get_num_threads(),median_us=statistics.median(times)/1000,process_cpu_s=proc,caller_cpu_s=caller,worker_cpu_s=proc-caller)),flush=True)
# Compile once and compare exact same f64 inputs; inspect optimized HLO separately.
x_np=((np.arange(8388608,dtype=np.float64)%1009)+1)/1009
x=jnp.asarray(x_np); fn=jax.jit(jnp.tanh); compiled=fn.lower(x).compile()
open('/tmp/tenferro-rootcause/jax_tanh_hlo.txt','w').write(compiled.as_text())
y=np.asarray(compiled(x)); ref=np.tanh(x_np)
print(json.dumps(dict(kind='jax_tanh_accuracy',input_dtype=str(x.dtype),output_dtype=str(y.dtype),max_abs_error=float(np.max(np.abs(y-ref))))),flush=True)
