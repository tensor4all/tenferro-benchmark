"""Prepared native framework arms for the existing perf_issues API follow-ups."""
from __future__ import annotations
from public_api_candidates import PreparedOperation

SUITE="cpu/neural_public_api"
NAMES=("erf","sigmoid","silu","softplus","gelu","gelu_tanh","reduce_mean","softmax","log_softmax","masked_softmax","layer_norm","rms_norm","take_along_axis")


def fixture(m,n):
    import numpy as np
    p=np.arange(m*n,dtype=np.int64)
    x=(.75*((p*1837+335)%2048-1024)/1024+((p%m*13+p//m*7)%31)/32-.5).astype(np.float32)
    return (x.reshape(m,n,order="F").copy(),
            (p%3==0).reshape(m,n,order="F").copy(),
            ((p*37+11)%n).reshape(m,n,order="F").copy())


def reference(name,x,mask,indices):
    import numpy as np
    from scipy.special import erf,expit
    x=x.astype(np.float64)
    if name=="erf":return erf(x)
    if name=="sigmoid":return expit(x)
    if name=="silu":return x*expit(x)
    if name=="softplus":return np.logaddexp(0,x)
    if name=="gelu":return .5*x*(1+erf(x/np.sqrt(2)))
    if name=="gelu_tanh":return .5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x**3)))
    if name=="take_along_axis":return np.take_along_axis(x,indices,axis=1)
    if name=="reduce_mean":return np.mean(x,axis=1)
    if name in ("layer_norm","rms_norm"):
        center=np.mean(x,axis=1,keepdims=True) if name=="layer_norm" else 0
        return (x-center)/np.sqrt(np.mean((x-center)**2,axis=1,keepdims=True)+1e-5)
    if name=="masked_softmax":x=np.where(mask,x,-np.inf)
    shifted=x-np.max(x,axis=1,keepdims=True)
    if name=="log_softmax":return shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
    return np.exp(shifted)/np.exp(shifted).sum(axis=1,keepdims=True)


def cases(framework,shapes=((1024,64),(4096,256))):
    rows=[]
    for m,n in shapes:
        for name in NAMES:
            note="public operation; axis=1; eps=1e-5; affine=none; setup/validation=untimed"
            if framework=="pytorch" and name=="masked_softmax":
                rows.append((SUITE,name,"f32",f"{m}x{n}","no matching direct boolean-masked softmax with specified zero outputs; composition excluded",None))
                continue
            if framework=="jax" and name=="rms_norm":
                note+="; direct jax.nn.standardize(mean=0, algorithm=stable) matches non-affine RMS norm"
            def setup(name=name,m=m,n=n):
                import numpy as np
                source,mask_data,index_data=fixture(m,n)
                expected=reference(name,source,mask_data,index_data)
                if framework=="pytorch":
                    import torch
                    from torch.nn import functional as f
                    x=torch.from_numpy(source); mask=torch.from_numpy(mask_data); indices=torch.from_numpy(index_data)
                    funcs={"erf":lambda:torch.erf(x),"sigmoid":lambda:torch.sigmoid(x),
                           "silu":lambda:f.silu(x),"softplus":lambda:f.softplus(x),
                           "gelu":lambda:f.gelu(x,approximate="none"),"gelu_tanh":lambda:f.gelu(x,approximate="tanh"),
                           "reduce_mean":lambda:torch.mean(x,dim=1),"softmax":lambda:f.softmax(x,dim=1),
                           "log_softmax":lambda:f.log_softmax(x,dim=1),"layer_norm":lambda:f.layer_norm(x,(n,),eps=1e-5),
                           "rms_norm":lambda:f.rms_norm(x,(n,),eps=1e-5),
                           "take_along_axis":lambda:torch.take_along_dim(x,indices,dim=1)}
                    def validate(out):np.testing.assert_allclose(out.numpy(),expected,rtol=2e-5,atol=2e-5)
                    return funcs[name],validate
                import jax
                import jax.numpy as jnp
                from jax.scipy.special import erf
                x=jnp.asarray(source); mask=jnp.asarray(mask_data); indices=jnp.asarray(index_data)
                funcs={"erf":lambda x,m,i:erf(x),"sigmoid":lambda x,m,i:jax.nn.sigmoid(x),
                       "silu":lambda x,m,i:jax.nn.silu(x),"softplus":lambda x,m,i:jax.nn.softplus(x),
                       "gelu":lambda x,m,i:jax.nn.gelu(x,approximate=False),"gelu_tanh":lambda x,m,i:jax.nn.gelu(x,approximate=True),
                       "reduce_mean":lambda x,m,i:jnp.mean(x,axis=1),"softmax":lambda x,m,i:jax.nn.softmax(x,axis=1),
                       "log_softmax":lambda x,m,i:jax.nn.log_softmax(x,axis=1),"masked_softmax":lambda x,m,i:jax.nn.softmax(x,axis=1,where=m),
                       "layer_norm":lambda x,m,i:jax.nn.standardize(x,axis=1,epsilon=1e-5,algorithm="stable"),
                       "rms_norm":lambda x,m,i:jax.nn.standardize(x,axis=1,mean=0.,epsilon=1e-5,algorithm="stable"),
                       "take_along_axis":lambda x,m,i:jnp.take_along_axis(x,i,axis=1)}
                func=funcs[name]
                if name=="masked_softmax":
                    fn=jax.jit(lambda a,b:func(a,b,None))
                    operation=lambda:fn(x,mask)
                elif name=="take_along_axis":
                    fn=jax.jit(lambda a,b:func(a,None,b))
                    operation=lambda:fn(x,indices)
                else:
                    fn=jax.jit(lambda a:func(a,None,None))
                    operation=lambda:fn(x)
                def validate(out):np.testing.assert_allclose(np.asarray(jax.block_until_ready(out)),expected,rtol=2e-5,atol=2e-5)
                return operation,validate
            rows.append((SUITE,name,"f32",f"{m}x{n}",note,PreparedOperation(setup,64 if m==1024 else 8)))
    return rows
