#!/usr/bin/env julia
# OMEinsum.jl reference column for the CPU einsum suite (cpu/einsum).
#
# Loads instance JSON files from data/instances/ and benchmarks pairwise
# contraction using OMEinsum.jl's `DynamicEinCode`, forced through each
# instance's PRECOMPUTED contraction path (opt_flops / opt_size) rather than
# OMEinsum's own optimizer, so the comparison stays apples-to-apples with the
# other einsum backends in this suite (tenferro trace/eager, PyTorch, JAX).
#
# Output format mirrors scripts/benchmark_python.py so
# scripts/format_results.py can parse Rust, Python, and Julia results in one
# unified table. The Julia-specific header line
# "Mode: omeinsum_path / Strategy: <strategy>" tells the formatter to render
# this column as "OMEinsum.jl OpenBLAS (ms)" and to treat it as a fair,
# path-forced comparison (mode `omeinsum_opt`, OMEinsum's own optimizer, is
# excluded by the formatter as unfair and intentionally never emitted here).
#
# Usage:
#     julia --project=. scripts/benchmark_einsum_omeinsum.jl
#
# Environment:
#     BENCH_INSTANCE        Run only the named instance (default: all in suite)
#     BENCH_SUITE_INCLUDE   Comma-separated instance IDs from the suite YAML
#     BENCH_RUNS            Timed runs per instance (default: 15)
#     BENCH_WARMUPS         Warmup runs per instance (default: 3)
#     JULIA_NUM_THREADS     Also used to pin LinearAlgebra.BLAS thread count
#                            (OMEinsum's pairwise contractions call into BLAS
#                            for matrix-shaped contractions)

using JSON
using LinearAlgebra
using OMEinsum
using Printf

const SCRIPT_DIR = @__DIR__
const PROJECT_DIR = normpath(joinpath(SCRIPT_DIR, ".."))
const DATA_DIR = joinpath(PROJECT_DIR, "data", "instances")

const DEFAULT_WARMUPS = 3
const DEFAULT_RUNS = 15

bench_warmups() = parse(Int, get(ENV, "BENCH_WARMUPS", string(DEFAULT_WARMUPS)))
bench_runs() = parse(Int, get(ENV, "BENCH_RUNS", string(DEFAULT_RUNS)))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

"""
    load_instances(instance_filter, suite_include) -> Vector{Dict{String,Any}}

Mirror scripts/benchmark_python.py's `load_instances`: read every
data/instances/*.json file, skip anything that is not a single-instance
record (missing "name", e.g. permutation_patterns.json), then filter by
BENCH_INSTANCE (exact name match) or BENCH_SUITE_INCLUDE (comma-separated
allow-list), same semantics as src/main.rs / benchmark_python.py.
"""
function load_instances(instance_filter::AbstractString, suite_include::AbstractString)
    paths = sort(filter(p -> endswith(p, ".json"), readdir(DATA_DIR; join=true)))

    allowed = nothing
    if !isempty(suite_include)
        allowed = Set(s for s in strip.(split(suite_include, ",")) if !isempty(s))
    end

    instances = Dict{String,Any}[]
    for path in paths
        local d
        try
            d = JSON.parsefile(path)
        catch e
            println(stderr, "Warning: skip $(basename(path)) ($e)")
            continue
        end
        if !(d isa AbstractDict) || !haskey(d, "name")
            println(stderr, "Warning: skip $(basename(path)) (missing field 'name')")
            continue
        end
        name = d["name"]
        if !isempty(instance_filter)
            name == instance_filter || continue
        elseif allowed !== nothing
            (name in allowed) || continue
        end
        push!(instances, d)
    end
    return instances
end

# ---------------------------------------------------------------------------
# Format string / contraction plan
# ---------------------------------------------------------------------------

"""
    parse_format(fmt) -> (input_labels, output_labels)

Split a column-major format string ("ji,kj->ki") into per-operand label
vectors and the output label vector. Index labels are arbitrary Unicode
characters (see str_nw_mera_open_26), so this operates on `Vector{Char}`
rather than bytes.
"""
function parse_format(fmt::AbstractString)
    parts = split(fmt, "->"; limit=2)
    lhs = parts[1]
    rhs = length(parts) > 1 ? parts[2] : ""
    ins = [collect(Char, s) for s in split(lhs, ",")]
    out = collect(Char, rhs)
    return ins, out
end

struct Step
    abs_i::Int
    abs_j::Int
    code::DynamicEinCode{Char}
end

"""
    build_plan(n_inputs, path, operand_labels, output_labels) -> Vector{Step}

Convert an opt_einsum-style relative pair path into absolute operand ids
(mirroring src/main.rs::path_to_pairs exactly: relative indices refer to a
shrinking list of "available" tensors, with each contraction step replacing
its two operands with one new intermediate appended to the end of that
list), and derive each intermediate's output labels using the standard
einsum pairwise-contraction rule: a label survives the pairwise step iff it
is still needed afterward, i.e. it appears in some *other* remaining operand
or in the final output. The very last step is forced to the instance's
actual output label order so the returned tensor matches the reference
layout for correctness comparisons.

The full nested plan (all DynamicEinCode objects) is built once, outside any
timed region, and reused by every warmup/timed run.
"""
function build_plan(
    n_inputs::Int,
    path::Vector{Vector{Int}},
    operand_labels::Vector{Vector{Char}},
    output_labels::Vector{Char},
)
    available = collect(0:(n_inputs - 1))
    label_sets = Dict{Int,Vector{Char}}()
    for i in 0:(n_inputs - 1)
        label_sets[i] = operand_labels[i + 1]
    end

    nsteps = length(path)
    steps = Vector{Step}(undef, max(nsteps, 0))

    if nsteps == 0
        # No suite instance currently hits this (every instance has
        # num_tensors >= 2), but handle it defensively: a single operand is
        # contracted directly to the output labels (e.g. summed over any
        # label absent from the output).
        return steps
    end

    for (step_idx, pair) in enumerate(path)
        a, b = pair[1], pair[2]
        i, j = a < b ? (a, b) : (b, a)
        i1, j1 = i + 1, j + 1

        abs_j = available[j1]
        abs_i = available[i1]
        deleteat!(available, j1)
        deleteat!(available, i1)

        ix_i = label_sets[abs_i]
        ix_j = label_sets[abs_j]

        is_final = step_idx == nsteps
        local iy::Vector{Char}
        if is_final
            iy = output_labels
        else
            remaining = Set{Char}()
            for id in available
                union!(remaining, label_sets[id])
            end
            union!(remaining, output_labels)

            iy = Char[]
            for c in Iterators.flatten((ix_i, ix_j))
                if (c in remaining) && !(c in iy)
                    push!(iy, c)
                end
            end
        end

        code = DynamicEinCode(Vector{Char}[copy(ix_i), copy(ix_j)], iy)
        steps[step_idx] = Step(abs_i, abs_j, code)

        intermediate_idx = n_inputs + step_idx - 1
        label_sets[intermediate_idx] = iy
        push!(available, intermediate_idx)
    end

    return steps
end

"""
    run_plan(steps, n_inputs, operands) -> result tensor

Execute the precomputed pairwise contraction plan. Only contractions run
here; DynamicEinCode construction happened once in `build_plan`.
"""
function run_plan(steps::Vector{Step}, n_inputs::Int, operands::Vector{Array{Float64}})
    if isempty(steps)
        return operands[1]
    end
    tensors = Vector{Any}(undef, n_inputs + length(steps))
    for i in 1:n_inputs
        tensors[i] = operands[i]
    end
    local result
    for (k, step) in enumerate(steps)
        x = tensors[step.abs_i + 1]
        y = tensors[step.abs_j + 1]
        result = step.code(x, y)
        tensors[n_inputs + k] = result
    end
    return result
end

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

"""
    compute_stats(times_ms) -> (median, iqr)

Same sample statistics as scripts/benchmark_python.py's `compute_stats`:
0-based indexing into the sorted sample (`times[n÷4]`, `times[3n÷4]`), so a
1-based Julia array uses `sorted[n÷4 + 1]` / `sorted[3n÷4 + 1]`.
"""
function compute_stats(times_ms::Vector{Float64})
    sorted = sort(times_ms)
    n = length(sorted)
    median = sorted[n ÷ 2 + 1]
    q1 = sorted[n ÷ 4 + 1]
    q3 = sorted[(3 * n) ÷ 4 + 1]
    iqr = q3 - q1
    return median, iqr
end

function path_to_vec(path_json)::Vector{Vector{Int}}
    return [Int.(pair) for pair in path_json]
end

"""
    strategy_cache_key(instance, path_meta) -> key

Reuse the measurement for identical physical contractions within one run,
matching scripts/benchmark_python.py's `strategy_cache_key` (some
instances/strategies share the same path).
"""
function strategy_cache_key(instance, path_meta)
    return (instance["name"], Tuple(Tuple(p) for p in path_meta["path"]))
end

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

function benchmark_instance(instance, strategy::String)
    dtype_str = get(instance, "dtype", "float64")
    if occursin("complex", dtype_str)
        return nothing, "complex dtype ($dtype_str) not supported"
    end

    fmt = instance["format_string_colmajor"]
    shapes = [Int.(s) for s in instance["shapes_colmajor"]]
    n_inputs = instance["num_tensors"]
    path_meta = instance["paths"][strategy]
    path = path_to_vec(path_meta["path"])

    operand_labels, output_labels = parse_format(fmt)

    local steps
    try
        steps = build_plan(n_inputs, path, operand_labels, output_labels)
    catch e
        return nothing, "plan construction failed: $e"
    end

    operands = Array{Float64}[zeros(Float64, shape...) for shape in shapes]

    try
        # Warmup (also triggers Julia JIT compilation of the contraction
        # kernels for this instance/strategy).
        for _ in 1:max(bench_warmups(), 1)
            r = run_plan(steps, n_inputs, operands)
            r === nothing && error("run_plan returned nothing")
        end

        times = Float64[]
        sizehint!(times, bench_runs())
        for _ in 1:bench_runs()
            r = nothing
            t0 = time_ns()
            r = run_plan(steps, n_inputs, operands)
            elapsed_ms = (time_ns() - t0) / 1.0e6
            r === nothing && error("run_plan returned nothing")
            push!(times, elapsed_ms)
        end

        return compute_stats(times), nothing
    catch e
        return nothing, sprint(showerror, e)
    end
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    num_threads = parse(Int, get(ENV, "JULIA_NUM_THREADS", "1"))
    LinearAlgebra.BLAS.set_num_threads(num_threads)

    instance_filter = get(ENV, "BENCH_INSTANCE", "")
    suite_include = get(ENV, "BENCH_SUITE_INCLUDE", "")

    instances = load_instances(instance_filter, suite_include)
    if !isempty(instance_filter) && isempty(instances)
        println(stderr, "BENCH_INSTANCE=$(repr(instance_filter)): no matching instance found")
        exit(1)
    end
    if isempty(instances)
        println(stderr, "No benchmark instances matched the suite selection")
        exit(1)
    end

    backend_name = "omeinsum-jl"
    println("$backend_name einsum benchmark suite")
    println("==================================")
    println("Loaded $(length(instances)) instances from $DATA_DIR")
    println("Backend: $backend_name")
    println("JULIA_NUM_THREADS=$num_threads")
    println("OMP_NUM_THREADS=", get(ENV, "OMP_NUM_THREADS", ""))
    println("OPENBLAS_NUM_THREADS=", get(ENV, "OPENBLAS_NUM_THREADS", ""))
    println("BLAS.get_num_threads()=", LinearAlgebra.BLAS.get_num_threads())
    println(
        "Timing: median ± IQR of $(bench_runs()) runs ($(bench_warmups()) warmup), " *
        "path precomputed (OMEinsum's own optimizer is not used)",
    )

    strategies = ["opt_flops", "opt_size"]
    col_w = 106

    measured_by_path = Dict{Tuple{String,Tuple},Tuple{Any,Union{String,Nothing}}}()

    for strategy in strategies
        println()
        println("Mode: omeinsum_path / Strategy: $strategy")
        @printf(
            "%-50s %8s %10s %12s %12s %10s\n",
            "Instance", "Tensors", "log10FLOPS", "log2SIZE", "Median (ms)", "IQR (ms)",
        )
        println("-"^col_w)

        for (idx, instance) in enumerate(instances)
            name = instance["name"]
            path_meta = instance["paths"][strategy]
            num_tensors = instance["num_tensors"]
            log10_flops = path_meta["log10_flops"]
            log2_size = path_meta["log2_size"]

            println(stderr, "  [$idx/$(length(instances))] $name...")

            cache_key = strategy_cache_key(instance, path_meta)
            cached = get(measured_by_path, cache_key, nothing)
            local result, err
            if cached !== nothing
                println(
                    stderr,
                    "  -> $name strategy=$strategy: reusing previous measurement for identical path",
                )
                result, err = cached
            else
                result, err = benchmark_instance(instance, strategy)
                measured_by_path[cache_key] = (result, err)
            end

            if result === nothing
                println(stderr, "  -> $name (error: $err)")
                @printf(
                    "%-50s %8d %10.2f %12.2f %12s %10s\n",
                    name, num_tensors, log10_flops, log2_size, "SKIP", "-",
                )
            else
                median, iqr = result
                @printf(
                    "%-50s %8d %10.2f %12.2f %12.3f %10.3f\n",
                    name, num_tensors, log10_flops, log2_size, median, iqr,
                )
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
