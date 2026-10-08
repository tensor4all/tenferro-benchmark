"""Matched Python CPU public-API references. All setup precedes every clock."""
import json, os, sys, time
from pathlib import Path
import numpy as np

case_id, backend, threads = sys.argv[1:4]
threads=int(threads); runs=int(sys.argv[4]) if len(sys.argv)>4 else 15
case=next(c for c in json.loads(Path(__file__).with_name('cases.json').read_text()) if c['id']==case_id)
cat, op=case['category'],case['op']
if backend in ('pytorch','mkl-dfti'):
 import torch
 torch.set_num_threads(threads);torch.set_num_interop_threads(1)
 api=torch
 def wrap(x): return torch.from_numpy(np.array(x,order='C',copy=True))
 def ready(x): return x
 def numpy(x): return x.detach().numpy()
else:
 import jax
 jax.config.update('jax_enable_x64', True)
 import jax.numpy as jnp
 api=jnp
 def wrap(x): return jnp.asarray(x)
 def ready(x): return jax.block_until_ready(x)
 def numpy(x): return np.asarray(x)

def values(n,seed):
 return ((np.arange(n,dtype=np.int64)*1837+seed*335)%2048-1024).astype(np.float64)/1024

def fixture(shape,seed=1,positive=False):
 a=values(int(np.prod(shape)),seed)
 if positive: a=.25+np.abs(a)
 return a.reshape(shape,order='F')

if cat=='real':
 reduction=op.startswith('reduce_')
 shape=([8192,4096] if op.endswith('_all') else [4096,4096] if op=='reduce_min_axis1' else [2048,2048]) if reduction else [4194304 if op in ('pow','expm1','log1p') else 8388608]
 a=fixture(shape,positive=op in ('pow','log','log1p'))
 if 'reduce_prod' in op:a=np.full(shape,1.000001)
 x=wrap(a); del a
 if reduction:
  name=op.split('_')[1];axis=None if op.endswith('_all') else 0 if op.endswith('axis0') else 1
  if backend=='pytorch':
   f={'max':torch.amax,'min':torch.amin,'sum':torch.sum,'prod':torch.prod}[name]
   call=(lambda:f(x)) if axis is None else (lambda:f(x,dim=axis))
  else:
   f=getattr(jnp,name);call=lambda:f(x,axis=axis)
 elif op=='pow':
  exponent=wrap(np.full(shape,1.5));call=lambda:api.pow(x,exponent) if backend=='pytorch' else jnp.power(x,exponent)
 else:call=lambda:getattr(api,op)(x)
elif cat=='activation':
 a=(((np.arange(65536,dtype=np.int64)*2654435761+50*97)%2001).astype(np.float64)/1000-1).astype(np.float32)*4
 x=wrap(a.reshape((1024,64),order='F')); del a
 if backend!='pytorch': raise ValueError('activation reference is PyTorch')
 if op in ('erf','sigmoid'):call=lambda:getattr(torch,op)(x)
 else:
  f=getattr(torch.nn.functional,'gelu' if op=='gelu_tanh' else op)
  call=(lambda:f(x,approximate='tanh')) if op=='gelu_tanh' else lambda:f(x)
elif cat=='index':
 n=262144 if op in ('gather','scatter') else 4194304 if op in ('slice','dynamic_slice') else 1048576 if op=='concatenate' else 2097152
 x=wrap(fixture([n]));y=wrap(fixture([n//2 if op=='dynamic_update_slice' else n],2))
 idx=wrap((np.arange(n,dtype=np.int64)*37+11)%n)
 if backend=='pytorch':
  if op=='gather':call=lambda:torch.index_select(x,0,idx)
  elif op=='scatter':
   zeros=torch.zeros_like(x);call=lambda:torch.scatter_add(zeros,0,idx,y)
  elif op=='slice':call=lambda:x[1024:n-1024:2].clone()
  elif op=='dynamic_slice':call=lambda:x[1024:1024+n//2].clone()
  elif op=='dynamic_update_slice':
   # PyTorch has no one-call allocation-returning equivalent. Do not time a composition.
   raise ValueError('unsupported: PyTorch has no allocation-returning dynamic_update_slice API')
  elif op=='pad':call=lambda:torch.nn.functional.pad(x,(128,128))
  elif op=='concatenate':call=lambda:torch.cat((x,y),dim=0)
  elif op=='reverse':call=lambda:torch.flip(x,(0,))
 else:
  if op=='gather':call=lambda:jnp.take(x,idx,axis=0)
  elif op=='scatter':
   zeros=jnp.zeros_like(x);call=lambda:zeros.at[idx].add(y)
  elif op=='slice':call=lambda:jax.lax.slice(x,(1024,),(n-1024,),(2,))
  elif op=='dynamic_slice':call=lambda:jax.lax.dynamic_slice(x,(1024,),(n//2,))
  elif op=='dynamic_update_slice':call=lambda:jax.lax.dynamic_update_slice(x,y,(1024,))
  elif op=='pad':call=lambda:jnp.pad(x,(128,128))
  elif op=='concatenate':call=lambda:jnp.concatenate((x,y))
  elif op=='reverse':call=lambda:jnp.flip(x)
elif cat=='structural':
 shape=[33554432] if op=='cast_f64_f32' else [8388608,2,2] if op=='extract_diagonal' else [8192,1] if op=='broadcast_in_dim' else [4096,4096]
 x=wrap(fixture(shape))
 if backend=='pytorch':
  if op=='cast_f64_f32':call=lambda:x.to(torch.float32)
  elif op=='extract_diagonal':call=lambda:torch.diagonal(x,dim1=1,dim2=2).clone()
  elif op=='transpose':call=lambda:x.permute(1,0).contiguous()
  elif op=='broadcast_in_dim':call=lambda:x.expand(8192,4096).clone()
  else:call=lambda:getattr(torch,op)(x)
 else:
  if op=='cast_f64_f32':call=lambda:x.astype(jnp.float32)
  elif op=='extract_diagonal':call=lambda:jnp.diagonal(x,axis1=1,axis2=2)
  elif op=='transpose':call=lambda:jnp.transpose(x)
  elif op=='broadcast_in_dim':call=lambda:jnp.broadcast_to(x,(8192,4096))
  else:call=lambda:getattr(jnp,op)(x)
elif cat in ('complex','linalg'):
 shape=[2048,1536] if op=='norm_fro' else [4194304] if cat=='complex' else [1024,1024]
 a=fixture(shape)
 if cat=='complex':a=a+1j*fixture(shape,2)
 else:a[np.arange(1024),np.arange(1024)]+=2+np.arange(1024)/1024
 x=wrap(a);del a
 if op=='norm_fro':call=lambda:api.linalg.norm(x)
 elif op=='slogdet':call=lambda:api.linalg.slogdet(x)
 else:call=lambda:getattr(api,op)(x)
elif cat=='fft':
 n=1048576;m=n//2+1 if op=='irfft' else n
 a=values(m,17)
 if op!='rfft':a=a+1j*values(m,18)
 a=a.astype(np.complex128 if case['dtype']=='c64' else np.float32 if op=='rfft' else np.complex64)
 x=wrap(a);del a
 call=(lambda:getattr(api.fft,op)(x,n=n,norm='backward')) if op=='irfft' else lambda:getattr(api.fft,op)(x,norm='backward')
else:raise ValueError('unsupported reference category')

if backend=='mkl-dfti':
 if cat!='fft':raise ValueError('oneMKL DFTI reference only supports FFT cases')
 from mkl_fft import prepared_fft
 call,cleanup,provider_metadata=prepared_fft(torch,x,op,n,threads)
 original_input=np.array(numpy(x),copy=True)

# Compiled JAX graphs are separately labelled; compilation and first execution untimed.
if backend=='jax':
 # Every tensor is a dynamic JIT argument. Capturing fixture arrays as constants
 # lets XLA fold the declared operation and invalidates performance comparisons.
 inputs=tuple(globals().get(k) for k in ('x','y','idx','exponent','zeros'))
 def dynamic(x,y,idx,exponent,zeros):
  if cat=='real':
   if reduction:return getattr(jnp,name)(x,axis=axis)
   return jnp.power(x,exponent) if op=='pow' else getattr(jnp,op)(x)
  if cat=='index':
   if op=='gather':return jnp.take(x,idx,axis=0)
   if op=='scatter':return zeros.at[idx].add(y)
   if op=='slice':return jax.lax.slice(x,(1024,),(n-1024,),(2,))
   if op=='dynamic_slice':return jax.lax.dynamic_slice(x,(1024,),(n//2,))
   if op=='dynamic_update_slice':return jax.lax.dynamic_update_slice(x,y,(1024,))
   if op=='pad':return jnp.pad(x,(128,128))
   if op=='concatenate':return jnp.concatenate((x,y))
   if op=='reverse':return jnp.flip(x)
  if cat=='structural':
   if op=='cast_f64_f32':return x.astype(jnp.float32)
   if op=='extract_diagonal':return jnp.diagonal(x,axis1=1,axis2=2)
   if op=='transpose':return jnp.transpose(x)
   if op=='broadcast_in_dim':return jnp.broadcast_to(x,(8192,4096))
   return getattr(jnp,op)(x)
  if cat in ('complex','linalg'):
   return jnp.linalg.norm(x) if op=='norm_fro' else jnp.linalg.slogdet(x) if op=='slogdet' else getattr(jnp,op)(x)
  if cat=='fft':return getattr(jnp.fft,op)(x,n=n,norm='backward') if op=='irfft' else getattr(jnp.fft,op)(x,norm='backward')
  raise ValueError('unsupported dynamic operation')
 compiled=jax.jit(dynamic)
 call=lambda:compiled(*inputs)
first=ready(call())
if backend=='mkl-dfti':assert np.array_equal(numpy(x),original_input), 'DFTI changed input during priming'
outs=first if isinstance(first,tuple) else (first,)

def signature(output):
 a=np.asarray(numpy(output));flat=a.ravel(order='F');n=flat.size
 ids=sorted(set([0,n-1,n//2]+[(i*1597334677)%n for i in range(1,128)])) if n else []
 assert np.all(np.isfinite(flat))
 z=flat.astype(np.complex128)
 return dict(shape=list(a.shape),count=n,sum=[float(z.real.sum()),float(z.imag.sum())],sum_abs=float(np.abs(z).sum()),sum_sq=float((np.abs(z)**2).sum()),probes=[[i,float(z[i].real),float(z[i].imag)] for i in ids])
sigs=[signature(o) for o in outs];output_bytes=sum(np.asarray(numpy(o)).nbytes for o in outs)
del first,outs
cap=max(1,min(65536,512*1024*1024//max(1,output_bytes)));target=int(os.environ.get('CPU_FOLLOWUP_TARGET_NS','10000000'))
def batch(count):
 retained=[None]*count
 start=time.perf_counter_ns()
 for i in range(count):retained[i]=ready(call())
 elapsed=time.perf_counter_ns()-start
 del retained
 return elapsed
samples=[];count=0;elapsed=0
if runs:
 for _ in range(3):ready(call())
 count=1
 while True:
  elapsed=batch(count)
  if elapsed>=target or count==cap:break
  count=min(count*2,cap)
 samples=[dict(sample_index=i,iterations=count,elapsed_ns=batch(count)) for i in range(runs)]
row=dict(case_id=case_id,path=backend+'-compiled' if backend=='jax' else backend+'-eager',threads=threads,outputs=sigs,samples=samples,calibration=dict(iterations=count,elapsed_ns=elapsed,target_ns=target,memory_cap_bytes=512*1024*1024),scope=dict(outside_timer=['fixtures','conversion','JIT compilation','initialization','validation','retention allocation','output destruction'],inside_timer=['API execution','intrinsic output allocation','native completion']))

if backend=='mkl-dfti':
 assert np.array_equal(numpy(x),original_input), 'DFTI changed input during sampling'
 row['path']='mkl-dfti-cached';row['provider']=provider_metadata;row['input_preserved']=True;cleanup()
print(json.dumps(row))
