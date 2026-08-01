#!/usr/bin/env julia
# Julia baseline columns for the CPU public API coverage benchmark suite.
#
# Ported from scripts/benchmark_cpu_public_api_python.py so that the row key
# (suite, benchmark, dtype, threads, shape) merges with the tenferro-rs /
# PyTorch / JAX rows already produced by scripts/run_cpu_public_api.sh. Emits
# two backends into the same CSV:
#
# - `julia-base`: the natural Base / LinearAlgebra spelling.
# - `strided-jl`: the Strided.jl (`@strided`) spelling, only for the
#   elementwise / chain / transpose rows where it naturally applies.
#
# Julia is column-major like tenferro-rs, so (unlike the PyTorch/JAX runners)
# no layout reconstruction is needed to keep the same logical fixture values:
# the linear index formula below is applied directly, column-major, via
# `reshape`.
#
# Environment variables (same semantics as benchmark_cpu_public_api_python.py):
# - PUBLICATION_GATE_PROFILE / BENCH_RUNS / BENCH_WARMUPS: timing profile.
# - PUBLIC_API_SUITE_FILTER / PUBLIC_API_BENCHMARK_FILTER: row filters used to
#   split large fixture families across separate julia processes.

using LinearAlgebra
using Strided

const PROJECT_DIR = normpath(joinpath(@__DIR__, ".."))

const FIELDNAMES = ("suite", "benchmark", "dtype", "threads", "shape", "backend", "median_ms", "iqr_ms", "status", "notes")

# ---------------------------------------------------------------------------
# Fixtures: reproduce the Rust LCG exactly. For 0-based linear index i (in
# column-major order, which is Julia's native order), the value is
# ((i*1837 + seed*335) mod 2048 - 1024) / 1024, as Float64.
# ---------------------------------------------------------------------------

function raw_values(n::Int, seed::Int)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 0:(n - 1)
        raw = mod(i * 1837 + seed * 335, 2048)
        out[i + 1] = (raw - 1024) / 1024
    end
    return out
end

tensor_f64(shape::Tuple, seed::Int) = reshape(raw_values(prod(shape), seed), shape)

tensor_f64_positive(shape::Tuple, seed::Int) = 0.25 .+ abs.(tensor_f64(shape, seed))

function well_conditioned(n::Int, seed::Int)
    x = tensor_f64((n, n), seed)
    for i in 1:n
        x[i, i] += 2.0 + (i - 1) / n
    end
    return x
end

function lower_triangular_fixture(n::Int, seed::Int)
    x = LinearAlgebra.tril(0.05 .* tensor_f64((n, n), seed))
    for i in 1:n
        x[i, i] = 2.0 + (i - 1) / n
    end
    return x
end

function spd_fixture(n::Int, seed::Int)
    source = tensor_f64((n, n), seed)
    matrix = (0.125 / n) .* (source .+ transpose(source))
    for i in 1:n
        matrix[i, i] += 2.0 + (i - 1) / n
    end
    return matrix
end

# ---------------------------------------------------------------------------
# Timing.
# ---------------------------------------------------------------------------

function runs_from_env()
    profile = lowercase(get(ENV, "PUBLICATION_GATE_PROFILE", "quick"))
    default_runs = profile == "full" ? 15 : 7
    runs = parse(Int, get(ENV, "BENCH_RUNS", string(default_runs)))
    warmups = parse(Int, get(ENV, "BENCH_WARMUPS", "3"))
    return runs, warmups
end

function median_iqr(times::Vector{Float64})
    sorted_times = sort(times)
    n = length(sorted_times)
    med = isodd(n) ? sorted_times[(n + 1) ÷ 2] : (sorted_times[n ÷ 2] + sorted_times[n ÷ 2 + 1]) / 2
    iqr = sorted_times[(3 * n) ÷ 4 + 1] - sorted_times[n ÷ 4 + 1]
    return med, iqr
end

function bench(f, runs::Int, warmups::Int)
    for _ in 1:warmups
        f()
    end
    times = Vector{Float64}(undef, runs)
    for i in 1:runs
        t0 = time_ns()
        f()
        times[i] = (time_ns() - t0) / 1e6
    end
    return median_iqr(times)
end

# ---------------------------------------------------------------------------
# CSV output.
# ---------------------------------------------------------------------------

csv_escape(value::AbstractString) = occursin(r"[,\"\n]", value) ? "\"" * replace(value, "\"" => "\"\"") * "\"" : value

function write_row(io::IO, suite, benchmark, dtype, threads, shape, backend, median_ms, iqr_ms, status, notes)
    fields = (
        suite, benchmark, dtype, string(threads), shape, backend,
        median_ms === nothing ? "" : string(round(median_ms, digits = 6)),
        iqr_ms === nothing ? "" : string(round(iqr_ms, digits = 6)),
        status, notes,
    )
    println(io, join(csv_escape.(fields), ","))
    flush(io)
end

function emit_case(io::IO, suite, benchmark, dtype, threads, shape, backend, notes, runs, warmups, f)
    try
        median_ms, iqr_ms = bench(f, runs, warmups)
        write_row(io, suite, benchmark, dtype, threads, shape, backend, median_ms, iqr_ms, "ok", notes)
    catch e
        write_row(io, suite, benchmark, dtype, threads, shape, backend, nothing, nothing, "failed", sprint(showerror, e))
    end
end

function selected(suite::String, benchmark::String)
    suite_filter = get(ENV, "PUBLIC_API_SUITE_FILTER", "")
    benchmark_filter = get(ENV, "PUBLIC_API_BENCHMARK_FILTER", "")
    suite_ok = isempty(suite_filter) || suite == suite_filter
    benchmark_ok = isempty(benchmark_filter) || (benchmark in split(benchmark_filter, ","))
    return suite_ok && benchmark_ok
end

# ---------------------------------------------------------------------------
# Elementwise fixtures (matches benchmark_cpu_public_api_python.py's make_cases).
# ---------------------------------------------------------------------------

struct Fixtures
    fast_n::Int
    ew_n::Int
    slow_n::Int
    x_fast::Vector{Float64}
    y_fast::Vector{Float64}
    yp_fast::Vector{Float64}
    xp_fast::Vector{Float64}
    cond_fast::Vector{Bool}
    x::Vector{Float64}
    yp::Vector{Float64}
    xp::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    x_slow::Vector{Float64}
    y_slow::Vector{Float64}
    xp_slow::Vector{Float64}
    exponent_slow::Vector{Float64}
    matrix_sum::Matrix{Float64}
    prod_matrix::Matrix{Float64}
    matrix_max::Matrix{Float64}
    matrix_min::Matrix{Float64}
end

function build_elementwise_fixtures()
    fast_n = 33_554_432
    ew_n = 8_388_608
    slow_n = 4_194_304
    return Fixtures(
        fast_n, ew_n, slow_n,
        tensor_f64((fast_n,), 1),
        tensor_f64((fast_n,), 2),
        tensor_f64_positive((fast_n,), 2),
        tensor_f64_positive((fast_n,), 1),
        [mod(i, 3) == 0 for i in 0:(fast_n - 1)],
        tensor_f64((ew_n,), 1),
        tensor_f64_positive((ew_n,), 2),
        tensor_f64_positive((ew_n,), 1),
        fill(-0.5, ew_n),
        fill(0.5, ew_n),
        tensor_f64((slow_n,), 1),
        tensor_f64((slow_n,), 2),
        tensor_f64_positive((slow_n,), 1),
        fill(1.5, slow_n),
        tensor_f64((8192, 4096), 1),
        fill(1.000001, 8192, 4096),
        tensor_f64((2048, 2048), 1),
        tensor_f64((4096, 4096), 1),
    )
end

function elementwise_cases(fx::Fixtures)
    cases = Vector{Tuple{String,String,String,String,String,Function,Union{Function,Nothing}}}()
    # (suite, benchmark, dtype, shape, notes, julia_base_fn, strided_fn_or_nothing)
    # Each strided_fn is a zero-arg closure that allocates its own `dst`
    # inside the timed call, then fills it with a fused `@strided @.` write
    # (see scripts/benchmark_permutation.jl for the analogous StridedView
    # pattern used by the cpu/permutation suite).
    push!(cases, ("cpu/elementwise_reduction", "add", "f64", "33554432", "binary elementwise",
        () -> fx.x_fast .+ fx.y_fast,
        () -> (dst = similar(fx.x_fast); @strided @. dst = fx.x_fast + fx.y_fast; dst)))
    push!(cases, ("cpu/elementwise_reduction", "sub", "f64", "33554432", "binary elementwise",
        () -> fx.x_fast .- fx.y_fast,
        () -> (dst = similar(fx.x_fast); @strided @. dst = fx.x_fast - fx.y_fast; dst)))
    push!(cases, ("cpu/elementwise_reduction", "mul", "f64", "33554432", "binary elementwise",
        () -> fx.x_fast .* fx.y_fast,
        () -> (dst = similar(fx.x_fast); @strided @. dst = fx.x_fast * fx.y_fast; dst)))
    push!(cases, ("cpu/elementwise_reduction", "div", "f64", "33554432", "binary elementwise",
        () -> fx.x_fast ./ fx.yp_fast,
        () -> (dst = similar(fx.x_fast); @strided @. dst = fx.x_fast / fx.yp_fast; dst)))
    push!(cases, ("cpu/elementwise_reduction", "rem", "f64", "8388608", "binary elementwise",
        () -> rem.(fx.x, fx.yp),
        () -> (dst = similar(fx.x); @strided @. dst = rem(fx.x, fx.yp); dst)))
    push!(cases, ("cpu/elementwise_reduction", "neg", "f64", "33554432", "unary elementwise",
        () -> .-fx.x_fast,
        () -> (dst = similar(fx.x_fast); @strided @. dst = -fx.x_fast; dst)))
    push!(cases, ("cpu/elementwise_reduction", "abs", "f64", "33554432", "unary elementwise",
        () -> abs.(fx.x_fast),
        () -> (dst = similar(fx.x_fast); @strided @. dst = abs(fx.x_fast); dst)))
    push!(cases, ("cpu/elementwise_reduction", "sign", "f64", "33554432", "unary elementwise",
        () -> sign.(fx.x_fast),
        () -> (dst = similar(fx.x_fast); @strided @. dst = sign(fx.x_fast); dst)))
    push!(cases, ("cpu/elementwise_reduction", "maximum", "f64", "33554432", "binary elementwise",
        () -> max.(fx.x_fast, fx.y_fast),
        () -> (dst = similar(fx.x_fast); @strided @. dst = max(fx.x_fast, fx.y_fast); dst)))
    push!(cases, ("cpu/elementwise_reduction", "minimum", "f64", "33554432", "binary elementwise",
        () -> min.(fx.x_fast, fx.y_fast),
        () -> (dst = similar(fx.x_fast); @strided @. dst = min(fx.x_fast, fx.y_fast); dst)))
    # compare_lt / select produce Bool output; @strided's fused broadcast
    # does not cleanly cover Bool-typed destinations the way it does the
    # f64 elementwise family, so no strided-jl row is emitted for these two.
    push!(cases, ("cpu/elementwise_reduction", "compare_lt", "f64", "33554432", "ordered compare",
        () -> fx.x_fast .< fx.y_fast,
        nothing))
    push!(cases, ("cpu/elementwise_reduction", "select", "f64", "33554432", "ternary select",
        () -> ifelse.(fx.cond_fast, fx.x_fast, fx.y_fast),
        nothing))
    push!(cases, ("cpu/elementwise_reduction", "clamp", "f64", "8388608", "clamp with tensor bounds",
        () -> clamp.(fx.x, fx.lower, fx.upper),
        () -> (dst = similar(fx.x); @strided @. dst = clamp(fx.x, fx.lower, fx.upper); dst)))
    push!(cases, ("cpu/elementwise_reduction", "exp", "f64", "8388608", "analytic unary",
        () -> exp.(fx.x),
        () -> (dst = similar(fx.x); @strided @. dst = exp(fx.x); dst)))
    push!(cases, ("cpu/elementwise_reduction", "log", "f64", "8388608", "analytic unary",
        () -> log.(fx.xp),
        () -> (dst = similar(fx.xp); @strided @. dst = log(fx.xp); dst)))
    push!(cases, ("cpu/elementwise_reduction", "sin", "f64", "8388608", "analytic unary",
        () -> sin.(fx.x),
        () -> (dst = similar(fx.x); @strided @. dst = sin(fx.x); dst)))
    push!(cases, ("cpu/elementwise_reduction", "cos", "f64", "8388608", "analytic unary",
        () -> cos.(fx.x),
        () -> (dst = similar(fx.x); @strided @. dst = cos(fx.x); dst)))
    push!(cases, ("cpu/elementwise_reduction", "tanh", "f64", "8388608", "analytic unary",
        () -> tanh.(fx.x),
        () -> (dst = similar(fx.x); @strided @. dst = tanh(fx.x); dst)))
    push!(cases, ("cpu/elementwise_reduction", "sqrt", "f64", "33554432", "analytic unary",
        () -> sqrt.(fx.xp_fast),
        () -> (dst = similar(fx.xp_fast); @strided @. dst = sqrt(fx.xp_fast); dst)))
    push!(cases, ("cpu/elementwise_reduction", "rsqrt", "f64", "33554432", "analytic unary",
        () -> @.(1 / sqrt(fx.xp_fast)),
        () -> (dst = similar(fx.xp_fast); @strided @. dst = 1 / sqrt(fx.xp_fast); dst)))
    push!(cases, ("cpu/elementwise_reduction", "pow", "f64", "4194304", "binary analytic with tensor exponent",
        () -> fx.xp_slow .^ fx.exponent_slow,
        () -> (dst = similar(fx.xp_slow); @strided @. dst = fx.xp_slow^fx.exponent_slow; dst)))
    push!(cases, ("cpu/elementwise_reduction", "expm1", "f64", "4194304", "analytic unary",
        () -> expm1.(fx.x_slow),
        () -> (dst = similar(fx.x_slow); @strided @. dst = expm1(fx.x_slow); dst)))
    push!(cases, ("cpu/elementwise_reduction", "log1p", "f64", "4194304", "analytic unary",
        () -> log1p.(fx.xp_slow),
        () -> (dst = similar(fx.xp_slow); @strided @. dst = log1p(fx.xp_slow); dst)))
    push!(cases, ("cpu/elementwise_reduction", "chain_log1p_exp_mul", "f64", "4194304", "short elementwise chain",
        () -> @.(exp(log1p(fx.xp_slow)) * fx.y_slow),
        () -> (dst = similar(fx.xp_slow); @strided @. dst = exp(log1p(fx.xp_slow)) * fx.y_slow; dst)))
    push!(cases, ("cpu/elementwise_reduction", "reduce_sum_all", "f64", "8192x4096", "full reduction",
        () -> sum(fx.matrix_sum), nothing))
    push!(cases, ("cpu/elementwise_reduction", "reduce_prod_all", "f64", "8192x4096", "full reduction",
        () -> prod(fx.prod_matrix), nothing))
    push!(cases, ("cpu/elementwise_reduction", "reduce_max_axis0", "f64", "2048x2048", "axis reduction",
        () -> maximum(fx.matrix_max; dims = 1), nothing))
    push!(cases, ("cpu/elementwise_reduction", "reduce_min_axis1", "f64", "4096x4096", "axis reduction",
        () -> minimum(fx.matrix_min; dims = 2), nothing))
    return cases
end

# ---------------------------------------------------------------------------
# Structural shape cases.
# ---------------------------------------------------------------------------

function structural_shape_cases()
    structural_matrix = tensor_f64((4096, 4096), 1)
    reshape_input = tensor_f64((33_554_432,), 1)
    broadcast_input = tensor_f64((8192, 1), 1)
    diagonal_input = tensor_f64((8192,), 1)

    cases = Vector{Tuple{String,String,String,String,String,Function,Union{Function,Nothing}}}()
    push!(cases, ("cpu/structural_shape", "transpose", "f64", "4096x4096", "materialized matrix transpose",
        () -> begin
            dst = Matrix{Float64}(undef, 4096, 4096)
            permutedims!(dst, structural_matrix, (2, 1))
            dst
        end,
        () -> begin
            dst = Matrix{Float64}(undef, 4096, 4096)
            src_view = StridedView(structural_matrix)
            @strided dst .= permutedims(src_view, (2, 1))
            dst
        end))
    push!(cases, ("cpu/structural_shape", "reshape", "f64", "33554432 -> 8192x4096", "materialized reshape; copy makes Julia perform the same output-sized write",
        () -> copy(reshape(reshape_input, 8192, 4096)), nothing))
    push!(cases, ("cpu/structural_shape", "broadcast_in_dim", "f64", "8192x1 -> 8192x4096", "materialized broadcast",
        () -> repeat(broadcast_input, 1, 4096), nothing))
    push!(cases, ("cpu/structural_shape", "cast_f64_f32", "f64->f32", "33554432", "dtype cast",
        () -> Float32.(reshape_input), nothing))
    push!(cases, ("cpu/structural_shape", "embed_diagonal", "f64", "8192 -> 8192x8192", "embed vector as matrix diagonal",
        () -> diagm(diagonal_input), nothing))
    push!(cases, ("cpu/structural_shape", "tril", "f64", "4096x4096", "lower triangle",
        () -> LinearAlgebra.tril(structural_matrix), nothing))
    push!(cases, ("cpu/structural_shape", "triu", "f64", "4096x4096", "upper triangle",
        () -> LinearAlgebra.triu(structural_matrix), nothing))
    return cases
end

# ---------------------------------------------------------------------------
# linalg_uncovered cases.
# ---------------------------------------------------------------------------

function linalg_uncovered_cases()
    spd1536 = spd_fixture(1536, 1)
    a160 = well_conditioned(160, 1)
    a192 = well_conditioned(192, 1)
    spd512 = spd_fixture(512, 1)
    l4096 = lower_triangular_fixture(4096, 1)
    rhs4096x64 = tensor_f64((4096, 64), 2)
    a1024 = well_conditioned(1024, 1)
    a768 = well_conditioned(768, 1)
    rect512x256 = tensor_f64((512, 256), 1)
    lu1024 = well_conditioned(1024, 1)
    lstsq_a = tensor_f64((768, 384), 1)
    lstsq_rhs = tensor_f64((768, 16), 2)
    svd_full_a = tensor_f64((768, 384), 1)
    norm2048 = tensor_f64((2048, 2048), 1)

    cases = Vector{Tuple{String,String,String,String,String,Function}}()
    push!(cases, ("cpu/linalg_uncovered", "cholesky", "f64", "1536x1536", "SPD input",
        () -> cholesky(Symmetric(spd1536))))
    push!(cases, ("cpu/linalg_uncovered", "eig", "f64", "160x160", "general input",
        () -> eigen(a160)))
    push!(cases, ("cpu/linalg_uncovered", "eigvals", "f64", "192x192", "general input values only",
        () -> eigvals(a192)))
    push!(cases, ("cpu/linalg_uncovered", "eigvalsh", "f64", "512x512", "SPD input values only",
        () -> eigvals(Symmetric(spd512))))
    push!(cases, ("cpu/linalg_uncovered", "triangular_solve", "f64", "4096x4096,rhs=64", "lower-triangular solve",
        () -> LowerTriangular(l4096) \ rhs4096x64))
    push!(cases, ("cpu/linalg_uncovered", "det", "f64", "1024x1024", "well-conditioned input",
        () -> det(a1024)))
    push!(cases, ("cpu/linalg_uncovered", "slogdet", "f64", "1024x1024", "well-conditioned input",
        () -> logabsdet(a1024)))
    push!(cases, ("cpu/linalg_uncovered", "inv", "f64", "768x768", "well-conditioned input",
        () -> inv(a768)))
    push!(cases, ("cpu/linalg_uncovered", "pinv", "f64", "512x256", "rectangular input",
        () -> pinv(rect512x256)))
    push!(cases, ("cpu/linalg_uncovered", "pinv_with_rtol", "f64", "512x256", "rectangular input; rtol=1e-12",
        () -> pinv(rect512x256; rtol = 1e-12)))
    push!(cases, ("cpu/linalg_uncovered", "lu", "f64", "1024x1024", "partial-pivot LU",
        () -> lu(lu1024)))
    push!(cases, ("cpu/linalg_uncovered", "lstsq", "f64", "768x384,rhs=16", "tall full-column-rank QR least-squares solve",
        () -> lstsq_a \ lstsq_rhs))
    push!(cases, ("cpu/linalg_uncovered", "svd_full", "f64", "768x384", "full-matrices SVD",
        () -> svd(svd_full_a; full = true)))
    push!(cases, ("cpu/linalg_uncovered", "norm_fro", "f64", "2048x2048", "Frobenius norm",
        () -> norm(norm2048)))
    return cases
end

# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

function parse_args()
    output = nothing
    num_threads = nothing
    args = ARGS
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--output"
            output = args[i + 1]
            i += 2
        elseif arg == "--num-threads"
            num_threads = parse(Int, args[i + 1])
            i += 2
        else
            error("unrecognized argument: $arg")
        end
    end
    output === nothing && error("--output is required")
    num_threads === nothing && error("--num-threads is required")
    return output, num_threads
end

function main()
    output_path, num_threads = parse_args()
    runs, warmups = runs_from_env()

    LinearAlgebra.BLAS.set_num_threads(num_threads)

    append = isfile(output_path)
    io = open(output_path, append ? "a" : "w")
    try
        if !append
            println(io, join(FIELDNAMES, ","))
        end

        elementwise_names = (
            "add", "sub", "mul", "div", "rem", "neg", "abs", "sign", "maximum", "minimum",
            "compare_lt", "select", "clamp", "exp", "log", "sin", "cos", "tanh", "sqrt", "rsqrt",
            "pow", "expm1", "log1p", "chain_log1p_exp_mul", "reduce_sum_all", "reduce_prod_all",
            "reduce_max_axis0", "reduce_min_axis1",
        )
        if any(name -> selected("cpu/elementwise_reduction", name), elementwise_names)
            fx = build_elementwise_fixtures()
            for (suite, benchmark, dtype, shape, notes, base_fn, strided_fn) in elementwise_cases(fx)
                selected(suite, benchmark) || continue
                emit_case(io, suite, benchmark, dtype, num_threads, shape, "julia-base",
                    "$notes; input allocation outside timed region", runs, warmups, base_fn)
                if strided_fn !== nothing
                    emit_case(io, suite, benchmark, dtype, num_threads, shape, "strided-jl",
                        "@strided fused broadcast; input allocation outside timed region", runs, warmups,
                        strided_fn)
                end
            end
        end

        if any(name -> selected("cpu/structural_shape", name),
               ("transpose", "reshape", "broadcast_in_dim", "cast_f64_f32", "embed_diagonal", "tril", "triu"))
            for (suite, benchmark, dtype, shape, notes, base_fn, strided_fn) in structural_shape_cases()
                selected(suite, benchmark) || continue
                emit_case(io, suite, benchmark, dtype, num_threads, shape, "julia-base",
                    "$notes; input allocation outside timed region", runs, warmups, base_fn)
                if strided_fn !== nothing
                    emit_case(io, suite, benchmark, dtype, num_threads, shape, "strided-jl",
                        "@strided permutedims; input allocation outside timed region", runs, warmups, strided_fn)
                end
            end
        end

        if any(name -> selected("cpu/linalg_uncovered", name),
               ("cholesky", "eig", "eigvals", "eigvalsh", "triangular_solve", "det", "slogdet", "inv",
                "pinv", "pinv_with_rtol", "lu", "lstsq", "svd_full", "norm_fro"))
            for (suite, benchmark, dtype, shape, notes, fn) in linalg_uncovered_cases()
                selected(suite, benchmark) || continue
                emit_case(io, suite, benchmark, dtype, num_threads, shape, "julia-base",
                    "$notes; input allocation outside timed region", runs, warmups, fn)
            end
        end
    finally
        close(io)
    end
end

main()
