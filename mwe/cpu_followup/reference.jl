using JSON, LinearAlgebra
id=ARGS[1]; threads=parse(Int,ARGS[2]); runs=length(ARGS)>2 ? parse(Int,ARGS[3]) : 15
BLAS.set_num_threads(threads)
case=only(filter(c->c["id"]==id,JSON.parsefile(joinpath(@__DIR__,"cases.json"))))
category=case["category"]; op=case["op"]
value(i,seed)=Float64(mod(i*1837+seed*335,2048)-1024)/1024
fixture(shape,seed=1)=reshape([value(i,seed) for i in 0:prod(shape)-1],shape...)
metadata=category=="metadata"
if metadata
 shape=op=="slice_view" ? (4096,) : op=="transpose_view" ? (2,2) : (1024,)
 setup(n)=[fixture(shape) for _ in 1:n]
 call=op=="reshape_view" ? x->reshape(x,32,32) : op=="transpose_view" ? x->PermutedDimsArray(x,(2,1)) : x->view(x,129:2:3968)
 first=call(only(setup(1)))
 sigs=[Dict("kind"=>"metadata","shape"=>collect(size(first)),"strides"=>collect(strides(first)),"offset"=>op=="slice_view" ? 128 : 0,"metadata_only"=>true)]
 bytes=prod(shape)*8+256
elseif category=="real"
 shape=endswith(op,"_all") ? (8192,4096) : op=="reduce_min_axis1" ? (4096,4096) : (2048,2048)
 x=fixture(shape); axis=endswith(op,"axis0") ? 1 : 2
 f=occursin("_max",op) ? maximum : minimum
 call=endswith(op,"_all") ? (_->f(x)) : (_->vec(f(x,dims=axis)))
 setup(n)=fill(nothing,n); first=call(nothing);bytes=first isa Number ? 8 : length(first)*8
elseif category=="complex"
 shape=op=="norm_fro" ? (2048,1536) : (4194304,)
 x=fixture(shape).+im.*fixture(shape,2)
 call=op=="norm_fro" ? (_->norm(x)) : op=="exp" ? (_->exp.(x)) : (_->log.(x))
 setup(n)=fill(nothing,n);first=call(nothing);bytes=first isa Number ? 8 : length(first)*16
elseif category=="linalg"
 x=fixture((1024,1024));for i in 1:1024;x[i,i]+=2+(i-1)/1024;end
 # Julia returns (logabs, sign); tenferro and Python return (sign, logabs).
 call=_->begin l,s=logabsdet(x);(s,l) end
 setup(n)=fill(nothing,n);first=call(nothing);bytes=16
else
 error("unsupported Julia category")
end
function signature(o)
 a=o isa Number ? fill(o) : Array(o); xs=vec(a);n=length(xs)
 ids=sort(unique(vcat([0,n-1,n÷2],[mod(i*1597334677,n) for i in 1:127])))
 @assert all(isfinite,xs)
 Dict("shape"=>Int[size(a)...],"count"=>n,"sum"=>[sum(real,xs),sum(imag,xs)],"sum_abs"=>sum(abs,xs),"sum_sq"=>sum(abs2,xs),"probes"=>[[i,real(xs[i+1]),imag(xs[i+1])] for i in ids])
end
if !metadata;sigs=map(signature,first isa Tuple ? collect(first) : [first]);end
first=nothing
cap=clamp((512*1024*1024)÷max(bytes,1),1,metadata ? 2000000 : 65536);target=parse(Int,get(ENV,"CPU_FOLLOWUP_TARGET_NS","10000000"))
# Typed retention storage and all owned inputs allocated before each clock.
const LAST_OUTPUTS=Ref{Any}(nothing)
function execute_batch!(outputs::Vector{O},inputs::Vector{I},operation::F) where {O,I,F}
 for i in eachindex(inputs);outputs[i]=operation(inputs[i]);end
 nothing
end
function batch(n,operation::F) where {F}
 # All types used by the executed loop are explicit; prime its exact
 # specialization, including output assignment, before the clock.
 LAST_OUTPUTS[]=nothing
 raw=setup(n); inputs=convert(Vector{typeof(raw[1])},raw)
 first=operation(inputs[1]);outputs=Vector{typeof(first)}(undef,n)
 execute_batch!(Vector{typeof(first)}(undef,1),inputs[1:1],operation)
 GC.gc(false)
 start=time_ns()
 execute_batch!(outputs,inputs,operation)
 elapsed=time_ns()-start
 # An observable escape after the clock prevents result/operation elimination
 # and retains every output until the next sample's untimed setup.
 LAST_OUTPUTS[]=outputs
 elapsed
end

samples=[];count=0;elapsed=0
if runs>0
 for _ in 1:3;call(only(setup(1)));end
 count=1
 while true
  global elapsed=batch(count,call)
  (elapsed>=target || count==cap) && break
  global count=min(count*2,cap)
 end
 samples=[Dict("sample_index"=>i-1,"iterations"=>count,"elapsed_ns"=>batch(count,call)) for i in 1:runs]
end
println(JSON.json(Dict("case_id"=>id,"path"=>"julia-base","threads"=>threads,"outputs"=>sigs,"samples"=>samples,"calibration"=>Dict("iterations"=>count,"elapsed_ns"=>elapsed,"target_ns"=>target,"memory_cap_bytes"=>512*1024*1024))))
