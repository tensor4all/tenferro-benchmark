#!/usr/bin/env julia
# CPU permutation / materialize-kernel benchmark suite (Julia runners).
#
# Ported from extern/strided-rs-benchmark-suite's
# benchmarks/strided_benchmarks/permute/permute.jl per
# docs/permutation-suite.md. Measures the `julia-base` (permutedims! /
# copyto!) and `strided-jl` (Strided.jl @strided) participants for every
# pattern in data/instances/permutation_patterns.json; other participants are
# measured by src/bin/benchmark_permutation.rs.
#
# Every participant's output is compared elementwise against a straightforward
# reference (Julia's own permutedims of the full source array) before any
# timing.
#
# Environment variables:
# - PATTERN_ID: run a single pattern id instead of the full suite.
# - BENCH_RUNS / BENCH_WARMUPS: override the size-scaled iteration and warmup
#   counts for every participant.
# - BENCH_OUTPUT: path to write machine-readable JSON Lines results. When
#   unset, only the human-readable table is printed to stdout.

using JSON
using Strided

const PROJECT_DIR = normpath(joinpath(@__DIR__, ".."))
const PATTERN_PATH = joinpath(PROJECT_DIR, "data", "instances", "permutation_patterns.json")
const SUITE_ID = "cpu/permutation"

function col_major_strides(shape::Vector{Int})
    strides = ones(Int, length(shape))
    for i in 2:length(shape)
        strides[i] = strides[i - 1] * shape[i - 1]
    end
    return strides
end

function deterministic_data(total::Int)
    return Float64.(1:total)
end

function median_duration(samples::Vector{Float64})
    sorted = sort(samples)
    n = length(sorted)
    if isodd(n)
        return sorted[(n + 1) ÷ 2]
    else
        return (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
    end
end

struct Timing
    warmup::Int
    iters::Int
    median_ms::Float64
    p25_ms::Float64
    p75_ms::Float64
    gbps::Float64
end

function bench(f, warmup::Int, iters::Int, bytes::Int)
    for _ in 1:warmup
        f()
    end
    samples = Float64[]
    sizehint!(samples, iters)
    for _ in 1:iters
        t0 = time_ns()
        f()
        push!(samples, (time_ns() - t0) / 1e9)
    end
    med = median_duration(samples)
    sorted = sort(samples)
    p25 = sorted[max(1, length(sorted) ÷ 4)]
    p75 = sorted[max(1, (length(sorted) * 3) ÷ 4)]
    return Timing(warmup, iters, med * 1e3, p25 * 1e3, p75 * 1e3, bytes / med / 1e9)
end

function timing_counts(total::Int)
    return total >= (1 << 23) ? (3, 15) : (5, 40)
end

function timing_counts_env(total::Int)
    default_warmup, default_iters = timing_counts(total)
    warmup = haskey(ENV, "BENCH_WARMUPS") ? parse(Int, ENV["BENCH_WARMUPS"]) : default_warmup
    iters = haskey(ENV, "BENCH_RUNS") ? parse(Int, ENV["BENCH_RUNS"]) : default_iters
    return warmup, max(iters, 1)
end

function load_patterns()
    suite = JSON.parsefile(PATTERN_PATH)
    suite["version"] == 1 || error("unsupported pattern schema version")
    suite["index_base"] == 0 || error("patterns must use index_base = 0")
    suite["semantics"] == "out[i0,...,ik] = src[i_perm0,...,i_permk]" ||
        error("unsupported semantics")
    suite["data"] == "deterministic_index_value" || error("unsupported data mode")
    return suite["patterns"]
end

function source_view(pattern)
    shape = Int.(pattern["shape"])
    total = prod(shape)
    parent = deterministic_data(total)
    layout = pattern["src_layout"]
    strides = if layout["kind"] == "col_major"
        col_major_strides(shape)
    elseif layout["kind"] == "explicit_strides"
        Int.(layout["strides"])
    else
        kind = layout["kind"]
        error("unsupported src_layout kind $kind")
    end
    return parent, StridedView(parent, Tuple(shape), Tuple(strides), 0)
end

function output_shape(pattern)
    shape = Int.(pattern["shape"])
    perm = Int.(pattern["perm"]) .+ 1
    return collect(shape[perm])
end

function verify_output(actual, expected)
    va = vec(actual)
    ve = vec(expected)
    length(va) == length(ve) || return "length mismatch $(length(va)) != $(length(ve))"
    for i in eachindex(va, ve)
        if va[i] != ve[i]
            return "mismatch at element $(i - 1): $(va[i]) != $(ve[i])"
        end
    end
    return nothing
end

function base_record(pattern, backend::String, total::Int, bytes::Int, threads::Int;
                      status::String = "ok", correctness::String = "passed",
                      per_call_allocation::Bool = false)
    return Dict{String,Any}(
        "schema_version" => 2,
        "suite_id" => SUITE_ID,
        "runner" => "julia",
        "pattern_id" => pattern["id"],
        "label" => pattern["label"],
        "backend" => backend,
        "shape" => Int.(pattern["shape"]),
        "perm" => Int.(pattern["perm"]),
        "dtype" => "f64",
        "elems" => total,
        "bytes_rw" => bytes,
        "threads" => threads,
        "status" => status,
        "correctness" => correctness,
        "per_call_allocation" => per_call_allocation,
        "warmup" => nothing,
        "iters" => nothing,
        "median_ms" => nothing,
        "p25_ms" => nothing,
        "p75_ms" => nothing,
        "gbps" => nothing,
        "notes" => nothing,
    )
end

function attach_timing!(record::Dict{String,Any}, timing::Timing)
    record["warmup"] = timing.warmup
    record["iters"] = timing.iters
    record["median_ms"] = timing.median_ms
    record["p25_ms"] = timing.p25_ms
    record["p75_ms"] = timing.p75_ms
    record["gbps"] = timing.gbps
    return record
end

function emit(io::Union{IO,Nothing}, record::Dict{String,Any})
    if io !== nothing
        println(io, JSON.json(record))
    end
    return record["status"] == "verification_failed"
end

function print_human_row(backend::String, timing::Union{Timing,Nothing}, note::Union{String,Nothing})
    if timing === nothing
        println("  ", rpad(backend, 30), " skipped: ", something(note, "no timing"))
    else
        println(
            "  ", rpad(backend, 30), " ", lpad(round(timing.median_ms, digits = 3), 8),
            " ms  (", round(timing.p25_ms, digits = 3), " / ",
            round(timing.p75_ms, digits = 3), ")  ", round(timing.gbps, digits = 2), " GB/s",
        )
    end
end

const THREADS = Threads.nthreads()

function run_pattern(pattern, io::Union{IO,Nothing})
    participants = Set(String.(pattern["participants"]))
    relevant = ("julia-base" in participants) || ("strided-jl" in participants)
    relevant || return (false, false)

    id = pattern["id"]
    label = pattern["label"]
    out_shape = output_shape(pattern)
    total = prod(out_shape)
    bytes = total * sizeof(Float64) * 2
    warmup, iters = timing_counts_env(total)

    println("=== $label ===")
    println("  id=$id elems=$total bytes(r+w)=$bytes")

    parent_data, src = source_view(pattern)
    perm = Int.(pattern["perm"]) .+ 1
    perm_tuple = Tuple(perm)
    src_perm = permutedims(src, perm_tuple)
    reference = Array(src_perm)

    any_failed = false

    if "julia-base" in participants
        if pattern["src_layout"]["kind"] == "col_major"
            src_array = reshape(parent_data, Tuple(Int.(pattern["shape"])))
            f = () -> begin
                dst = Array{Float64}(undef, Tuple(out_shape))
                permutedims!(dst, src_array, perm_tuple)
                dst
            end
        else
            src_base_perm = PermutedDimsArray(src, perm_tuple)
            f = () -> begin
                dst = Array{Float64}(undef, Tuple(out_shape))
                copyto!(dst, src_base_perm)
                dst
            end
        end
        actual = f()
        err = verify_output(actual, reference)
        record = base_record(
            pattern,
            "julia-base",
            total,
            bytes,
            THREADS;
            per_call_allocation = true,
        )
        if err !== nothing
            record["status"] = "verification_failed"
            record["correctness"] = "failed"
            record["notes"] = err
            print_human_row("julia-base", nothing, err)
        else
            timing = bench(f, warmup, iters, bytes)
            attach_timing!(record, timing)
            print_human_row("julia-base", timing, nothing)
        end
        any_failed |= emit(io, record)
    end

    if "strided-jl" in participants
        f = () -> begin
            dst_parent = Vector{Float64}(undef, total)
            dst = StridedView(
                dst_parent,
                Tuple(out_shape),
                Tuple(col_major_strides(out_shape)),
                0,
            )
            @strided dst .= src_perm
            dst
        end
        actual = f()
        err = verify_output(Array(actual), reference)
        record = base_record(
            pattern,
            "strided-jl",
            total,
            bytes,
            THREADS;
            per_call_allocation = true,
        )
        if err !== nothing
            record["status"] = "verification_failed"
            record["correctness"] = "failed"
            record["notes"] = err
            print_human_row("strided-jl", nothing, err)
        else
            timing = bench(f, warmup, iters, bytes)
            attach_timing!(record, timing)
            print_human_row("strided-jl", timing, nothing)
        end
        any_failed |= emit(io, record)
    end

    println()
    return (true, any_failed)
end

function main()
    println("tenferro-benchmark permutation suite (cpu/permutation) -- Julia runners")
    println("========================================================================")
    println("Patterns: $PATTERN_PATH")
    if haskey(ENV, "PATTERN_ID")
        println("Pattern filter: ", ENV["PATTERN_ID"])
    end
    output_path = get(ENV, "BENCH_OUTPUT", nothing)
    if output_path !== nothing
        println("JSON output: ", output_path)
    end
    println("Julia threads: ", THREADS, " Strided.jl threads: ", Strided.get_num_threads())
    println("Format: label  median_ms  (p25 / p75)  bandwidth_GB/s")
    println()

    patterns = load_patterns()
    if haskey(ENV, "PATTERN_ID")
        patterns = filter(p -> p["id"] == ENV["PATTERN_ID"], patterns)
    end
    isempty(patterns) && error("PATTERN_ID did not match any pattern")

    io = output_path === nothing ? nothing : open(output_path, "w")
    any_failed = false
    ran = false
    try
        for pattern in patterns
            pattern_ran, pattern_failed = run_pattern(pattern, io)
            ran |= pattern_ran
            any_failed |= pattern_failed
        end
    finally
        io !== nothing && close(io)
    end

    if !ran
        println("No patterns matched the selected filter.")
    end
    if any_failed
        error("one or more participants failed the correctness gate")
    end
    println("Done.")
end

main()
