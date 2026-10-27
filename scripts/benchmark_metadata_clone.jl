# Rank-8 metadata-only reference for tenferro-rs #2180. No payload copies.
using JSON
runs = parse(Int, ARGS[1])
target_ns = parse(Int, ARGS[2])
const owner = reshape(collect(1.0:256.0), ntuple(_ -> 2, 8))
const prepared = PermutedDimsArray(owner, (8,7,6,5,4,3,2,1))
# Julia descriptors are immutable; reconstruct the same borrowing descriptor.
clone_metadata(v) = PermutedDimsArray(parent(v), (8,7,6,5,4,3,2,1))
probe = clone_metadata(prepared)
@assert parent(probe) === owner
@assert size(probe) == ntuple(_ -> 2, 8)
@assert strides(probe) == (128,64,32,16,8,4,2,1)
@assert probe == prepared
const RETAINED = Ref{Any}(nothing)
function execute!(outputs, view)
    for i in eachindex(outputs)
        outputs[i] = clone_metadata(view)
    end
end
function batch(n)
    outputs = Vector{typeof(prepared)}(undef, n)
    execute!(Vector{typeof(prepared)}(undef, 1), prepared)
    start = time_ns()
    execute!(outputs, prepared)
    elapsed = time_ns() - start
    RETAINED[] = outputs
    elapsed
end
samples = Any[]
iterations = 0
if runs > 0
    for _ in 1:3; batch(1); end
    iterations = 1
    # Bound retained descriptor memory as well as operation count.
    cap = min(1 << 20, (512 << 20) ÷ max(sizeof(probe), 1))
    while batch(iterations) < target_ns && iterations < cap
        global iterations = min(2iterations, cap)
    end
    for i in 0:runs-1
        push!(samples, Dict("sample_index"=>i,"elapsed_ns"=>batch(iterations),"iterations"=>iterations))
    end
end
println(JSON.json(Dict("correctness_status"=>"passed", "samples"=>samples,
    "calibration"=>Dict("iterations"=>iterations,"target_ns"=>target_ns),
    "operation_requires_session"=>false,"payload_copied"=>false,
    "scope"=>Dict("timer"=>["PermutedDimsArray metadata descriptor"],
        "outside_timer"=>["owner and transposed view creation", "alias/layout checks", "JIT compilation", "output retention allocation and cleanup"])) ))
