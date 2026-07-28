//! CPU public API benchmark runner for tenferro-rs APIs not covered by the
//! focused FFT/einsum/permutation suites.

use std::collections::HashMap;
use std::env;
use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use num_complex::Complex64;
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_linalg::{TensorLinalgExt, TracedTensorLinalgExt};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor,
    TensorAnalytic, TensorDot, TensorElementwise, TensorIndexing, TensorReduction,
};

type BenchResult<T> = Result<T, Box<dyn std::error::Error>>;

const EW_FAST_N: usize = 33_554_432;
const EW_N: usize = 8_388_608;
const EW_SLOW_N: usize = 4_194_304;

struct Args {
    output: PathBuf,
    num_threads: usize,
    runs: usize,
    warmups: usize,
}

struct Case {
    suite: &'static str,
    benchmark: &'static str,
    dtype: &'static str,
    shape: &'static str,
    notes: &'static str,
    run: fn(&mut CpuBackend) -> tenferro_tensor::Result<()>,
}

fn main() -> BenchResult<()> {
    let args = parse_args()?;
    let append = args.output.exists();
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.output)?;
    let mut writer = BufWriter::new(file);
    if !append {
        writeln!(
            writer,
            "suite,benchmark,dtype,threads,shape,backend,median_ms,iqr_ms,status,notes"
        )?;
    }

    let mut backend = cpu_backend_from_env()?;
    let suite_filter = env::var("PUBLIC_API_SUITE_FILTER").ok();
    let benchmark_filter = env::var("PUBLIC_API_BENCHMARK_FILTER").ok();
    for case in cases().into_iter().filter(|case| {
        let suite_matches = suite_filter
            .as_deref()
            .map_or(true, |suite| suite == case.suite);
        let benchmark_matches = benchmark_filter.as_deref().map_or(true, |benchmarks| {
            benchmarks.is_empty()
                || benchmarks
                    .split(',')
                    .any(|benchmark| benchmark == case.benchmark)
        });
        suite_matches && benchmark_matches
    }) {
        emit_case(&mut writer, &args, &mut backend, &case)?;
    }
    writer.flush()?;
    Ok(())
}

fn parse_args() -> BenchResult<Args> {
    let mut output = None;
    let mut num_threads = None;
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--output" => output = iter.next().map(PathBuf::from),
            "--num-threads" => {
                num_threads = Some(
                    iter.next()
                        .ok_or("--num-threads requires a value")?
                        .parse::<usize>()?,
                );
            }
            "-h" | "--help" => {
                println!("Usage: benchmark_cpu_public_api --output <csv> [--num-threads N]");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    let profile = env::var("PUBLICATION_GATE_PROFILE")
        .unwrap_or_else(|_| "quick".to_string())
        .to_ascii_lowercase();
    Ok(Args {
        output: output.ok_or("--output is required")?,
        num_threads: num_threads
            .or_else(|| env::var("RAYON_NUM_THREADS").ok()?.parse::<usize>().ok())
            .unwrap_or(1),
        runs: env::var("BENCH_RUNS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(if profile == "full" { 15 } else { 7 }),
        warmups: env::var("BENCH_WARMUPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3),
    })
}

fn cases() -> Vec<Case> {
    vec![
        // Elementwise and reductions (#73).
        elem("add", "f64", "33554432", "binary elementwise", add_f64),
        elem("sub", "f64", "33554432", "binary elementwise", sub_f64),
        elem("mul", "f64", "33554432", "binary elementwise", mul_f64),
        elem("div", "f64", "33554432", "binary elementwise", div_f64),
        elem("rem", "f64", "8388608", "binary elementwise", rem_f64),
        elem("neg", "f64", "33554432", "unary elementwise", neg_f64),
        elem("abs", "f64", "33554432", "unary elementwise", abs_f64),
        elem("sign", "f64", "33554432", "unary elementwise", sign_f64),
        elem(
            "maximum",
            "f64",
            "33554432",
            "binary elementwise",
            maximum_f64,
        ),
        elem(
            "minimum",
            "f64",
            "33554432",
            "binary elementwise",
            minimum_f64,
        ),
        elem(
            "compare_lt",
            "f64",
            "33554432",
            "ordered compare",
            compare_lt_f64,
        ),
        elem("select", "f64", "33554432", "ternary select", select_f64),
        elem(
            "clamp",
            "f64",
            "8388608",
            "clamp with tensor bounds",
            clamp_f64,
        ),
        elem("exp", "f64", "8388608", "analytic unary", exp_f64),
        elem("log", "f64", "8388608", "analytic unary", log_f64),
        elem("sin", "f64", "8388608", "analytic unary", sin_f64),
        elem("cos", "f64", "8388608", "analytic unary", cos_f64),
        elem("tanh", "f64", "8388608", "analytic unary", tanh_f64),
        elem("sqrt", "f64", "33554432", "analytic unary", sqrt_f64),
        elem("rsqrt", "f64", "33554432", "analytic unary", rsqrt_f64),
        elem("pow", "f64", "4194304", "binary analytic", pow_f64),
        elem("expm1", "f64", "4194304", "analytic unary", expm1_f64),
        elem("log1p", "f64", "4194304", "analytic unary", log1p_f64),
        elem(
            "chain_log1p_exp_mul",
            "f64",
            "4194304",
            "short elementwise chain",
            chain_f64,
        ),
        elem(
            "reduce_sum_all",
            "f64",
            "8192x4096",
            "full reduction",
            reduce_sum_all_f64,
        ),
        elem(
            "reduce_prod_all",
            "f64",
            "8192x4096",
            "full reduction",
            reduce_prod_all_f64,
        ),
        elem(
            "reduce_max_axis0",
            "f64",
            "2048x2048",
            "axis reduction",
            reduce_max_axis0_f64,
        ),
        elem(
            "reduce_min_axis1",
            "f64",
            "4096x4096",
            "axis reduction",
            reduce_min_axis1_f64,
        ),
        // Indexing/layout (#72).
        idx(
            "gather",
            "f64",
            "262144",
            "1D StableHLO-style gather",
            gather_f64,
        ),
        idx(
            "scatter",
            "f64",
            "262144",
            "1D StableHLO-style scatter",
            scatter_f64,
        ),
        idx("slice", "f64", "4194304", "static slice", slice_f64),
        idx(
            "dynamic_slice",
            "f64",
            "4194304",
            "runtime-start slice",
            dynamic_slice_f64,
        ),
        idx(
            "dynamic_update_slice",
            "f64",
            "2097152",
            "runtime-start update",
            dynamic_update_slice_f64,
        ),
        idx("pad", "f64", "2097152", "edge padding", pad_f64),
        idx(
            "concatenate",
            "f64",
            "1048576+1048576",
            "concatenate along axis 0",
            concatenate_f64,
        ),
        idx("reverse", "f64", "2097152", "reverse axis 0", reverse_f64),
        // Uncovered linalg (#71).
        lin("cholesky", "f64", "1536x1536", "SPD input", cholesky_f64),
        lin("eig", "f64", "160x160", "general input", eig_f64),
        lin(
            "eigvals",
            "f64",
            "192x192",
            "general input values only",
            eigvals_f64,
        ),
        lin(
            "eigvalsh",
            "f64",
            "512x512",
            "SPD input values only",
            eigvalsh_f64,
        ),
        lin(
            "triangular_solve",
            "f64",
            "4096x4096,rhs=64",
            "lower-triangular solve",
            triangular_solve_f64,
        ),
        lin("det", "f64", "1024x1024", "well-conditioned input", det_f64),
        lin(
            "slogdet",
            "f64",
            "1024x1024",
            "well-conditioned input",
            slogdet_f64,
        ),
        lin("inv", "f64", "768x768", "well-conditioned input", inv_f64),
        lin("pinv", "f64", "512x256", "rectangular input", pinv_f64),
        lin("norm_fro", "f64", "2048x2048", "Frobenius norm", norm_f64),
        // Complex coverage (#74).
        cplx("conj", "c64", "16777216", "complex elementwise", conj_c64),
        cplx("mul", "c64", "8388608", "complex elementwise", mul_c64),
        cplx("div", "c64", "8388608", "complex elementwise", div_c64),
        cplx("exp", "c64", "4194304", "complex analytic", exp_c64),
        cplx("log", "c64", "4194304", "complex analytic", log_c64),
        cplx(
            "dot_general",
            "c64",
            "640x640",
            "complex matrix multiply",
            dot_c64,
        ),
        cplx("svd", "c64", "160x160", "complex SVD", svd_c64),
        cplx("qr", "c64", "256x256", "complex QR", qr_c64),
        cplx("eig", "c64", "112x112", "complex eig", eig_c64),
        cplx("solve", "c64", "384x384,rhs=8", "complex solve", solve_c64),
        cplx(
            "cholesky",
            "c64",
            "448x448",
            "Hermitian positive definite",
            cholesky_c64,
        ),
        cplx(
            "norm_fro",
            "c64",
            "2048x1536",
            "complex Frobenius norm",
            norm_c64,
        ),
    ]
}

fn elem(
    benchmark: &'static str,
    dtype: &'static str,
    shape: &'static str,
    notes: &'static str,
    run: fn(&mut CpuBackend) -> tenferro_tensor::Result<()>,
) -> Case {
    Case {
        suite: "cpu/elementwise_reduction",
        benchmark,
        dtype,
        shape,
        notes,
        run,
    }
}

fn idx(
    benchmark: &'static str,
    dtype: &'static str,
    shape: &'static str,
    notes: &'static str,
    run: fn(&mut CpuBackend) -> tenferro_tensor::Result<()>,
) -> Case {
    Case {
        suite: "cpu/indexing_layout",
        benchmark,
        dtype,
        shape,
        notes,
        run,
    }
}

fn lin(
    benchmark: &'static str,
    dtype: &'static str,
    shape: &'static str,
    notes: &'static str,
    run: fn(&mut CpuBackend) -> tenferro_tensor::Result<()>,
) -> Case {
    Case {
        suite: "cpu/linalg_uncovered",
        benchmark,
        dtype,
        shape,
        notes,
        run,
    }
}

fn cplx(
    benchmark: &'static str,
    dtype: &'static str,
    shape: &'static str,
    notes: &'static str,
    run: fn(&mut CpuBackend) -> tenferro_tensor::Result<()>,
) -> Case {
    Case {
        suite: "cpu/complex",
        benchmark,
        dtype,
        shape,
        notes,
        run,
    }
}

fn emit_case(
    writer: &mut impl Write,
    args: &Args,
    backend: &mut CpuBackend,
    case: &Case,
) -> BenchResult<()> {
    match time_case(args, backend, case.run) {
        Ok((median_ms, iqr_ms)) => {
            writeln!(
                writer,
                "{},{},{},{},\"{}\",tenferro-eager,{median_ms:.6},{iqr_ms:.6},ok,\"{}\"",
                case.suite,
                case.benchmark,
                case.dtype,
                args.num_threads,
                csv_escape(case.shape),
                csv_escape(case.notes),
            )?;
        }
        Err(err) => {
            writeln!(
                writer,
                "{},{},{},{},\"{}\",tenferro-eager,,,failed,\"{}\"",
                case.suite,
                case.benchmark,
                case.dtype,
                args.num_threads,
                csv_escape(case.shape),
                csv_escape(&err.to_string()),
            )?;
        }
    }
    emit_trace_case(writer, args, case)?;
    Ok(())
}

fn emit_trace_case(writer: &mut impl Write, args: &Args, case: &Case) -> BenchResult<()> {
    if case.suite == "cpu/indexing_layout" && case.benchmark == "dynamic_update_slice" {
        writeln!(
            writer,
            "{},{},{},{},\"{}\",tenferro-trace,,,unsupported,\"{}\"",
            case.suite,
            case.benchmark,
            case.dtype,
            args.num_threads,
            csv_escape(case.shape),
            "tenferro-rs has no TracedTensor dynamic_update_slice API",
        )?;
        return Ok(());
    }

    match time_trace_case(args, case) {
        Ok((median_ms, iqr_ms)) => {
            writeln!(
                writer,
                "{},{},{},{},\"{}\",tenferro-trace,{median_ms:.6},{iqr_ms:.6},ok,\"{}\"",
                case.suite,
                case.benchmark,
                case.dtype,
                args.num_threads,
                csv_escape(case.shape),
                csv_escape(case.notes),
            )?;
        }
        Err(err) => {
            writeln!(
                writer,
                "{},{},{},{},\"{}\",tenferro-trace,,,failed,\"{}\"",
                case.suite,
                case.benchmark,
                case.dtype,
                args.num_threads,
                csv_escape(case.shape),
                csv_escape(&err.to_string()),
            )?;
        }
    }
    Ok(())
}

fn time_case(
    args: &Args,
    backend: &mut CpuBackend,
    run: fn(&mut CpuBackend) -> tenferro_tensor::Result<()>,
) -> BenchResult<(f64, f64)> {
    for _ in 0..args.warmups {
        run(backend)?;
    }
    let mut times = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let start = Instant::now();
        run(backend)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(median_iqr(&times))
}

fn time_trace_case(args: &Args, case: &Case) -> BenchResult<(f64, f64)> {
    // Fixture creation, graph construction, and compilation are deliberately
    // outside the measured region. Timings cover execution of the reused
    // compiled graph, including creation of its owned output tensors.
    let outputs = build_trace_case(case)?;
    let output_refs: Vec<&TracedTensor> = outputs.iter().collect();
    let program = GraphCompiler::new().compile_many(&output_refs)?;
    let runtime = cpu_trace_runtime()?;

    for _ in 0..args.warmups {
        consume_many(runtime.run_compiled(&program, &[])?);
    }
    let mut times = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let start = Instant::now();
        consume_many(runtime.run_compiled(&program, &[])?);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(median_iqr(&times))
}

fn cpu_trace_runtime() -> BenchResult<Runtime> {
    let backend = cpu_backend_from_env()?;
    let engine_id = tenferro_cpu::runtime_engine_id()?;
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend)?)?;
    builder
        .install_extension_module(tenferro_linalg::extension_module::<CpuBackend>(engine_id)?)?;
    Ok(builder.build()?)
}

fn median_iqr(times: &[f64]) -> (f64, f64) {
    let mut values = times.to_vec();
    values.sort_by(|a, b| a.total_cmp(b));
    let median = if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    };
    let iqr = values[(3 * values.len()) / 4] - values[values.len() / 4];
    (median, iqr)
}

fn cpu_backend_from_env() -> BenchResult<CpuBackend> {
    match env::var("TENFERRO_CPU_BACKEND_KIND")
        .unwrap_or_else(|_| "default".to_string())
        .as_str()
    {
        "" | "default" => Ok(CpuBackend::new()),
        "blas" => Ok(CpuBackend::with_kind(CpuBackendKind::Blas)?),
        "faer" => Ok(CpuBackend::with_kind(CpuBackendKind::Faer)?),
        other => Err(format!("unsupported TENFERRO_CPU_BACKEND_KIND={other}").into()),
    }
}

fn csv_escape(value: &str) -> String {
    value.replace('"', "\"\"")
}

fn consume(tensor: Tensor) {
    black_box(tensor.shape().len());
    black_box(tensor.dtype());
}

fn consume_many(values: impl IntoIterator<Item = Tensor>) {
    for value in values {
        consume(value);
    }
}

fn data_f64(n: usize, seed: u64) -> Vec<f64> {
    (0..n).map(|i| pseudo_value(i, seed)).collect()
}

fn positive_data_f64(n: usize, seed: u64) -> Vec<f64> {
    (0..n).map(|i| 0.25 + pseudo_value(i, seed).abs()).collect()
}

fn data_i64(n: usize, modulus: usize) -> Vec<i64> {
    (0..n).map(|i| ((i * 37 + 11) % modulus) as i64).collect()
}

fn data_bool(n: usize) -> Vec<bool> {
    (0..n).map(|i| i % 3 == 0).collect()
}

fn data_c64(n: usize, seed: u64) -> Vec<Complex64> {
    (0..n)
        .map(|i| Complex64::new(pseudo_value(i, seed), pseudo_value(i, seed + 1)))
        .collect()
}

fn pseudo_value(i: usize, seed: u64) -> f64 {
    let x = (i as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(seed.wrapping_mul(1_442_695_040_888_963_407));
    ((x % 2048) as f64 - 1024.0) / 1024.0
}

type TensorCache = OnceLock<Mutex<HashMap<String, &'static Tensor>>>;

fn cached_tensor(
    cache: &'static TensorCache,
    key: String,
    build: impl FnOnce() -> Tensor,
) -> &'static Tensor {
    let mut entries = cache
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    if let Some(tensor) = entries.get(&key) {
        return tensor;
    }
    // Benchmark fixtures live for the short lifetime of this runner process.
    // Leaking them gives timed operations direct immutable references while
    // keeping all fixture allocation in the warmup phase.
    let tensor: &'static Tensor = Box::leak(Box::new(build()));
    entries.insert(key, tensor);
    tensor
}

fn tensor_f64(shape: &[usize], seed: u64) -> &'static Tensor {
    static CACHE: TensorCache = OnceLock::new();
    let key = format!("{shape:?}:{seed}");
    cached_tensor(&CACHE, key, || {
        Tensor::from_vec_col_major(shape.to_vec(), data_f64(shape.iter().product(), seed)).unwrap()
    })
}

fn tensor_f64_positive(shape: &[usize], seed: u64) -> &'static Tensor {
    static CACHE: TensorCache = OnceLock::new();
    let key = format!("{shape:?}:{seed}");
    cached_tensor(&CACHE, key, || {
        Tensor::from_vec_col_major(
            shape.to_vec(),
            positive_data_f64(shape.iter().product(), seed),
        )
        .unwrap()
    })
}

fn tensor_f64_constant(shape: &[usize], value: f64) -> &'static Tensor {
    static CACHE: TensorCache = OnceLock::new();
    let key = format!("{shape:?}:{}", value.to_bits());
    cached_tensor(&CACHE, key, || {
        Tensor::from_vec_col_major(shape.to_vec(), vec![value; shape.iter().product()]).unwrap()
    })
}

fn tensor_i64_indices(shape: &[usize], modulus: usize) -> &'static Tensor {
    static CACHE: TensorCache = OnceLock::new();
    let key = format!("{shape:?}:{modulus}");
    cached_tensor(&CACHE, key, || {
        Tensor::from_vec_col_major(shape.to_vec(), data_i64(shape.iter().product(), modulus))
            .unwrap()
    })
}

fn tensor_i64_constant(shape: &[usize], value: i64) -> &'static Tensor {
    static CACHE: TensorCache = OnceLock::new();
    let key = format!("{shape:?}:{value}");
    cached_tensor(&CACHE, key, || {
        Tensor::from_vec_col_major(shape.to_vec(), vec![value; shape.iter().product()]).unwrap()
    })
}

fn tensor_bool(shape: &[usize]) -> &'static Tensor {
    static CACHE: TensorCache = OnceLock::new();
    let key = format!("{shape:?}");
    cached_tensor(&CACHE, key, || {
        Tensor::from_vec_col_major(shape.to_vec(), data_bool(shape.iter().product())).unwrap()
    })
}

fn tensor_c64(shape: &[usize], seed: u64) -> &'static Tensor {
    static CACHE: TensorCache = OnceLock::new();
    let key = format!("{shape:?}:{seed}");
    cached_tensor(&CACHE, key, || {
        Tensor::from_vec_col_major(shape.to_vec(), data_c64(shape.iter().product(), seed)).unwrap()
    })
}

fn tensor_c64_constant(shape: &[usize], value: Complex64) -> &'static Tensor {
    static CACHE: TensorCache = OnceLock::new();
    let key = format!("{shape:?}:{}:{}", value.re.to_bits(), value.im.to_bits());
    cached_tensor(&CACHE, key, || {
        Tensor::from_vec_col_major(shape.to_vec(), vec![value; shape.iter().product()]).unwrap()
    })
}

fn well_conditioned(n: usize, seed: u64) -> &'static Tensor {
    static CACHE: TensorCache = OnceLock::new();
    cached_tensor(&CACHE, format!("{n}:{seed}"), || {
        let mut values = data_f64(n * n, seed);
        for j in 0..n {
            values[j + j * n] += 2.0 + j as f64 / n as f64;
        }
        Tensor::from_vec_col_major(vec![n, n], values).unwrap()
    })
}

fn lower_triangular(n: usize, seed: u64) -> &'static Tensor {
    static CACHE: TensorCache = OnceLock::new();
    cached_tensor(&CACHE, format!("{n}:{seed}"), || {
        let mut values = vec![0.0; n * n];
        for col in 0..n {
            for row in col..n {
                values[row + col * n] = if row == col {
                    2.0 + row as f64 / n as f64
                } else {
                    0.05 * pseudo_value(row + col * n, seed)
                };
            }
        }
        Tensor::from_vec_col_major(vec![n, n], values).unwrap()
    })
}

fn spd(n: usize, seed: u64) -> &'static Tensor {
    static CACHE: TensorCache = OnceLock::new();
    cached_tensor(&CACHE, format!("{n}:{seed}"), || {
        let _ = seed;
        let mut values = vec![0.0; n * n];
        for col in 0..n {
            values[col + col * n] = 2.0 + col as f64 / n as f64;
        }
        Tensor::from_vec_col_major(vec![n, n], values).unwrap()
    })
}

fn hpd_c64(n: usize, seed: u64) -> &'static Tensor {
    static CACHE: TensorCache = OnceLock::new();
    cached_tensor(&CACHE, format!("{n}:{seed}"), || {
        let _ = seed;
        let mut values = vec![Complex64::new(0.0, 0.0); n * n];
        for col in 0..n {
            values[col + col * n] = Complex64::new(2.0 + col as f64 / n as f64, 0.0);
        }
        Tensor::from_vec_col_major(vec![n, n], values).unwrap()
    })
}

fn well_conditioned_c64(n: usize, seed: u64) -> &'static Tensor {
    static CACHE: TensorCache = OnceLock::new();
    cached_tensor(&CACHE, format!("{n}:{seed}"), || {
        let mut values = data_c64(n * n, seed);
        for j in 0..n {
            values[j + j * n] += Complex64::new(3.0 + j as f64 / n as f64, 0.0);
        }
        Tensor::from_vec_col_major(vec![n, n], values).unwrap()
    })
}

fn traced(tensor: &Tensor) -> BenchResult<TracedTensor> {
    Ok(TracedTensor::from_tensor_concrete_shape(tensor.clone())?)
}

fn build_trace_case(case: &Case) -> BenchResult<Vec<TracedTensor>> {
    let one = |value| Ok(vec![value]);
    match (case.suite, case.benchmark) {
        ("cpu/elementwise_reduction", "add") => {
            one(traced(tensor_f64(&[EW_FAST_N], 1))?.add(&traced(tensor_f64(&[EW_FAST_N], 2))?)?)
        }
        ("cpu/elementwise_reduction", "sub") => {
            one(traced(tensor_f64(&[EW_FAST_N], 1))?.sub(&traced(tensor_f64(&[EW_FAST_N], 2))?)?)
        }
        ("cpu/elementwise_reduction", "mul") => {
            one(traced(tensor_f64(&[EW_FAST_N], 1))?.mul(&traced(tensor_f64(&[EW_FAST_N], 2))?)?)
        }
        ("cpu/elementwise_reduction", "div") => one(traced(tensor_f64(&[EW_FAST_N], 1))?
            .div(&traced(tensor_f64_positive(&[EW_FAST_N], 2))?)?),
        ("cpu/elementwise_reduction", "rem") => {
            one(traced(tensor_f64(&[EW_N], 1))?.rem(&traced(tensor_f64_positive(&[EW_N], 2))?)?)
        }
        ("cpu/elementwise_reduction", "neg") => one(traced(tensor_f64(&[EW_FAST_N], 1))?.neg()?),
        ("cpu/elementwise_reduction", "abs") => one(traced(tensor_f64(&[EW_FAST_N], 1))?.abs()?),
        ("cpu/elementwise_reduction", "sign") => one(traced(tensor_f64(&[EW_FAST_N], 1))?.sign()?),
        ("cpu/elementwise_reduction", "maximum") => {
            one(traced(tensor_f64(&[EW_FAST_N], 1))?
                .maximum(&traced(tensor_f64(&[EW_FAST_N], 2))?)?)
        }
        ("cpu/elementwise_reduction", "minimum") => {
            one(traced(tensor_f64(&[EW_FAST_N], 1))?
                .minimum(&traced(tensor_f64(&[EW_FAST_N], 2))?)?)
        }
        ("cpu/elementwise_reduction", "compare_lt") => one(traced(tensor_f64(&[EW_FAST_N], 1))?
            .compare(&traced(tensor_f64(&[EW_FAST_N], 2))?, CompareDir::Lt)?),
        ("cpu/elementwise_reduction", "select") => one(TracedTensor::select(
            &traced(tensor_bool(&[EW_FAST_N]))?,
            &traced(tensor_f64(&[EW_FAST_N], 1))?,
            &traced(tensor_f64(&[EW_FAST_N], 2))?,
        )?),
        ("cpu/elementwise_reduction", "clamp") => one(traced(tensor_f64(&[EW_N], 1))?.clamp(
            &traced(tensor_f64_constant(&[EW_N], -0.5))?,
            &traced(tensor_f64_constant(&[EW_N], 0.5))?,
        )?),
        ("cpu/elementwise_reduction", "exp") => one(traced(tensor_f64(&[EW_N], 1))?.exp()?),
        ("cpu/elementwise_reduction", "log") => {
            one(traced(tensor_f64_positive(&[EW_N], 1))?.log()?)
        }
        ("cpu/elementwise_reduction", "sin") => one(traced(tensor_f64(&[EW_N], 1))?.sin()?),
        ("cpu/elementwise_reduction", "cos") => one(traced(tensor_f64(&[EW_N], 1))?.cos()?),
        ("cpu/elementwise_reduction", "tanh") => one(traced(tensor_f64(&[EW_N], 1))?.tanh()?),
        ("cpu/elementwise_reduction", "sqrt") => {
            one(traced(tensor_f64_positive(&[EW_FAST_N], 1))?.sqrt()?)
        }
        ("cpu/elementwise_reduction", "rsqrt") => {
            one(traced(tensor_f64_positive(&[EW_FAST_N], 1))?.rsqrt()?)
        }
        ("cpu/elementwise_reduction", "pow") => one(traced(tensor_f64_positive(&[EW_SLOW_N], 1))?
            .pow(&traced(tensor_f64_constant(&[EW_SLOW_N], 1.5))?)?),
        ("cpu/elementwise_reduction", "expm1") => {
            one(traced(tensor_f64(&[EW_SLOW_N], 1))?.expm1()?)
        }
        ("cpu/elementwise_reduction", "log1p") => {
            one(traced(tensor_f64_positive(&[EW_SLOW_N], 1))?.log1p()?)
        }
        ("cpu/elementwise_reduction", "chain_log1p_exp_mul") => {
            let x = traced(tensor_f64_positive(&[EW_SLOW_N], 1))?.log1p()?;
            let y = x.exp()?;
            one(y.mul(&traced(tensor_f64(&[EW_SLOW_N], 2))?)?)
        }
        ("cpu/elementwise_reduction", "reduce_sum_all") => {
            one(traced(tensor_f64(&[8192, 4096], 1))?.reduce_sum(Some(&[0, 1]))?)
        }
        ("cpu/elementwise_reduction", "reduce_prod_all") => {
            one(traced(tensor_f64_constant(&[8192, 4096], 1.000001))?.reduce_prod(Some(&[0, 1]))?)
        }
        ("cpu/elementwise_reduction", "reduce_max_axis0") => {
            one(traced(tensor_f64(&[2048, 2048], 1))?.reduce_max(Some(&[0]))?)
        }
        ("cpu/elementwise_reduction", "reduce_min_axis1") => {
            one(traced(tensor_f64(&[4096, 4096], 1))?.reduce_min(Some(&[1]))?)
        }
        ("cpu/indexing_layout", "gather") => {
            const N: usize = 262_144;
            one(traced(tensor_f64(&[N], 1))?.gather(
                &traced(tensor_i64_indices(&[N], N))?,
                GatherConfig {
                    offset_dims: vec![],
                    collapsed_slice_dims: vec![0],
                    start_index_map: vec![0],
                    index_vector_dim: 1,
                    slice_sizes: vec![1],
                },
            )?)
        }
        ("cpu/indexing_layout", "scatter") => {
            const N: usize = 262_144;
            one(traced(tensor_f64_constant(&[N], 0.0))?.scatter(
                &traced(tensor_i64_indices(&[N, 1], N))?,
                &traced(tensor_f64(&[N], 2))?,
                ScatterConfig {
                    update_window_dims: vec![],
                    inserted_window_dims: vec![0],
                    scatter_dims_to_operand_dims: vec![0],
                    index_vector_dim: 1,
                },
            )?)
        }
        ("cpu/indexing_layout", "slice") => {
            const N: usize = 4_194_304;
            one(traced(tensor_f64(&[N], 1))?.slice(SliceConfig {
                starts: vec![1024],
                limits: vec![N - 1024],
                strides: vec![2],
            })?)
        }
        ("cpu/indexing_layout", "dynamic_slice") => {
            const N: usize = 4_194_304;
            one(traced(tensor_f64(&[N], 1))?
                .dynamic_slice(&traced(tensor_i64_constant(&[1], 1024))?, &[N / 2])?)
        }
        ("cpu/indexing_layout", "pad") => {
            const N: usize = 2_097_152;
            one(traced(tensor_f64(&[N], 1))?.pad(PadConfig {
                edge_padding_low: vec![128],
                edge_padding_high: vec![128],
                interior_padding: vec![0],
            })?)
        }
        ("cpu/indexing_layout", "concatenate") => {
            let a = traced(tensor_f64(&[1_048_576], 1))?;
            let b = traced(tensor_f64(&[1_048_576], 2))?;
            one(TracedTensor::concatenate(&[&a, &b], 0)?)
        }
        ("cpu/indexing_layout", "reverse") => {
            one(traced(tensor_f64(&[2_097_152], 1))?.reverse(&[0])?)
        }
        ("cpu/linalg_uncovered", "cholesky") => one(traced(spd(1536, 1))?.cholesky()?),
        ("cpu/linalg_uncovered", "eig") => {
            let (w, v) = traced(well_conditioned(160, 1))?.eig()?;
            Ok(vec![w, v])
        }
        ("cpu/linalg_uncovered", "eigvals") => one(traced(well_conditioned(192, 1))?.eigvals()?),
        ("cpu/linalg_uncovered", "eigvalsh") => one(traced(spd(512, 1))?.eigvalsh()?),
        ("cpu/linalg_uncovered", "triangular_solve") => one(traced(lower_triangular(4096, 1))?
            .triangular_solve(
                &traced(tensor_f64(&[4096, 64], 2))?,
                true,
                true,
                false,
                false,
            )?),
        ("cpu/linalg_uncovered", "det") => one(traced(well_conditioned(1024, 1))?.det()?),
        ("cpu/linalg_uncovered", "slogdet") => {
            let (sign, logabsdet) = traced(well_conditioned(1024, 1))?.slogdet()?;
            Ok(vec![sign, logabsdet])
        }
        ("cpu/linalg_uncovered", "inv") => one(traced(well_conditioned(768, 1))?.inv()?),
        ("cpu/linalg_uncovered", "pinv") => one(traced(tensor_f64(&[512, 256], 1))?.pinv()?),
        ("cpu/linalg_uncovered", "norm_fro") => {
            one(traced(tensor_f64(&[2048, 2048], 1))?.norm(None, Some(&[0, 1]), false)?)
        }
        ("cpu/complex", "conj") => one(traced(tensor_c64(&[16_777_216], 1))?.conj()?),
        ("cpu/complex", "mul") => {
            one(traced(tensor_c64(&[8_388_608], 1))?.mul(&traced(tensor_c64(&[8_388_608], 2))?)?)
        }
        ("cpu/complex", "div") => one(traced(tensor_c64(&[8_388_608], 1))?.div(&traced(
            tensor_c64_constant(&[8_388_608], Complex64::new(1.5, 0.25)),
        )?)?),
        ("cpu/complex", "exp") => one(traced(tensor_c64(&[4_194_304], 1))?.exp()?),
        ("cpu/complex", "log") => {
            one(traced(tensor_c64_constant(&[4_194_304], Complex64::new(1.5, 0.25)))?.log()?)
        }
        ("cpu/complex", "dot_general") => one(traced(tensor_c64(&[640, 640], 1))?.dot_general(
            &traced(tensor_c64(&[640, 640], 2))?,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )?),
        ("cpu/complex", "svd") => {
            let (u, s, vt) = traced(tensor_c64(&[160, 160], 1))?.svd()?;
            Ok(vec![u, s, vt])
        }
        ("cpu/complex", "qr") => {
            let (q, r) = traced(tensor_c64(&[256, 256], 1))?.qr()?;
            Ok(vec![q, r])
        }
        ("cpu/complex", "eig") => {
            let (w, v) = traced(tensor_c64(&[112, 112], 1))?.eig()?;
            Ok(vec![w, v])
        }
        ("cpu/complex", "solve") => {
            one(traced(well_conditioned_c64(384, 1))?.solve(&traced(tensor_c64(&[384, 8], 2))?)?)
        }
        ("cpu/complex", "cholesky") => one(traced(hpd_c64(448, 1))?.cholesky()?),
        ("cpu/complex", "norm_fro") => {
            one(traced(tensor_c64(&[2048, 1536], 1))?.norm(None, Some(&[0, 1]), false)?)
        }
        _ => Err(format!(
            "no trace benchmark implementation for {}/{}",
            case.suite, case.benchmark
        )
        .into()),
    }
}

// Elementwise/reduction.
fn add_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.add(tensor_f64(&[EW_FAST_N], 1), tensor_f64(&[EW_FAST_N], 2))?);
    Ok(())
}
fn sub_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.sub(tensor_f64(&[EW_FAST_N], 1), tensor_f64(&[EW_FAST_N], 2))?);
    Ok(())
}
fn mul_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.mul(tensor_f64(&[EW_FAST_N], 1), tensor_f64(&[EW_FAST_N], 2))?);
    Ok(())
}
fn div_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.div(
        tensor_f64(&[EW_FAST_N], 1),
        tensor_f64_positive(&[EW_FAST_N], 2),
    )?);
    Ok(())
}
fn rem_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.rem(tensor_f64(&[EW_N], 1), tensor_f64_positive(&[EW_N], 2))?);
    Ok(())
}
fn neg_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.neg(tensor_f64(&[EW_FAST_N], 1))?);
    Ok(())
}
fn abs_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.abs(tensor_f64(&[EW_FAST_N], 1))?);
    Ok(())
}
fn sign_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.sign(tensor_f64(&[EW_FAST_N], 1))?);
    Ok(())
}
fn maximum_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.maximum(tensor_f64(&[EW_FAST_N], 1), tensor_f64(&[EW_FAST_N], 2))?);
    Ok(())
}
fn minimum_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.minimum(tensor_f64(&[EW_FAST_N], 1), tensor_f64(&[EW_FAST_N], 2))?);
    Ok(())
}
fn compare_lt_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.compare(
        tensor_f64(&[EW_FAST_N], 1),
        tensor_f64(&[EW_FAST_N], 2),
        &CompareDir::Lt,
    )?);
    Ok(())
}
fn select_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.select(
        tensor_bool(&[EW_FAST_N]),
        tensor_f64(&[EW_FAST_N], 1),
        tensor_f64(&[EW_FAST_N], 2),
    )?);
    Ok(())
}
fn clamp_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.clamp(
        tensor_f64(&[EW_N], 1),
        tensor_f64_constant(&[EW_N], -0.5),
        tensor_f64_constant(&[EW_N], 0.5),
    )?);
    Ok(())
}
fn exp_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.exp(tensor_f64(&[EW_N], 1))?);
    Ok(())
}
fn log_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.log(tensor_f64_positive(&[EW_N], 1))?);
    Ok(())
}
fn sin_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.sin(tensor_f64(&[EW_N], 1))?);
    Ok(())
}
fn cos_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.cos(tensor_f64(&[EW_N], 1))?);
    Ok(())
}
fn tanh_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.tanh(tensor_f64(&[EW_N], 1))?);
    Ok(())
}
fn sqrt_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.sqrt(tensor_f64_positive(&[EW_FAST_N], 1))?);
    Ok(())
}
fn rsqrt_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.rsqrt(tensor_f64_positive(&[EW_FAST_N], 1))?);
    Ok(())
}
fn pow_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.pow(
        tensor_f64_positive(&[EW_SLOW_N], 1),
        tensor_f64_constant(&[EW_SLOW_N], 1.5),
    )?);
    Ok(())
}
fn expm1_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.expm1(tensor_f64(&[EW_SLOW_N], 1))?);
    Ok(())
}
fn log1p_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.log1p(tensor_f64_positive(&[EW_SLOW_N], 1))?);
    Ok(())
}
fn chain_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let x = b.log1p(tensor_f64_positive(&[EW_SLOW_N], 1))?;
    let y = b.exp(&x)?;
    consume(b.mul(&y, tensor_f64(&[EW_SLOW_N], 2))?);
    Ok(())
}
fn reduce_sum_all_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.reduce_sum(tensor_f64(&[8192, 4096], 1), &[0, 1])?);
    Ok(())
}
fn reduce_prod_all_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.reduce_prod(tensor_f64_constant(&[8192, 4096], 1.000001), &[0, 1])?);
    Ok(())
}
fn reduce_max_axis0_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.reduce_max(tensor_f64(&[2048, 2048], 1), &[0])?);
    Ok(())
}
fn reduce_min_axis1_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.reduce_min(tensor_f64(&[4096, 4096], 1), &[1])?);
    Ok(())
}

// Indexing/layout.
fn gather_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    const N: usize = 262_144;
    consume(b.gather(
        tensor_f64(&[N], 1),
        tensor_i64_indices(&[N], N),
        &GatherConfig {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        },
    )?);
    Ok(())
}
fn scatter_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    const N: usize = 262_144;
    consume(b.scatter(
        tensor_f64_constant(&[N], 0.0),
        tensor_i64_indices(&[N, 1], N),
        tensor_f64(&[N], 2),
        &ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        },
    )?);
    Ok(())
}
fn slice_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    const N: usize = 4_194_304;
    consume(b.slice(
        tensor_f64(&[N], 1),
        &SliceConfig {
            starts: vec![1024],
            limits: vec![N - 1024],
            strides: vec![2],
        },
    )?);
    Ok(())
}
fn dynamic_slice_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    const N: usize = 4_194_304;
    consume(b.dynamic_slice(
        tensor_f64(&[N], 1),
        tensor_i64_constant(&[1], 1024),
        &[N / 2],
    )?);
    Ok(())
}
fn dynamic_update_slice_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    const N: usize = 2_097_152;
    consume(b.dynamic_update_slice(
        tensor_f64(&[N], 1),
        tensor_f64(&[N / 2], 2),
        tensor_i64_constant(&[1], 1024),
    )?);
    Ok(())
}
fn pad_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    const N: usize = 2_097_152;
    consume(b.pad(
        tensor_f64(&[N], 1),
        &PadConfig {
            edge_padding_low: vec![128],
            edge_padding_high: vec![128],
            interior_padding: vec![0],
        },
    )?);
    Ok(())
}
fn concatenate_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let a = tensor_f64(&[1_048_576], 1);
    let c = tensor_f64(&[1_048_576], 2);
    consume(b.concatenate(&[a, c], 0)?);
    Ok(())
}
fn reverse_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.reverse(tensor_f64(&[2_097_152], 1), &[0])?);
    Ok(())
}

// Linalg.
fn cholesky_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(spd(1536, 1).cholesky(b)?);
    Ok(())
}
fn eig_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let (w, v) = well_conditioned(160, 1).eig(b)?;
    consume_many([w, v]);
    Ok(())
}
fn eigvals_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(well_conditioned(192, 1).eigvals(b)?);
    Ok(())
}
fn eigvalsh_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(spd(512, 1).eigvalsh(b)?);
    Ok(())
}
fn triangular_solve_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(lower_triangular(4096, 1).triangular_solve(
        tensor_f64(&[4096, 64], 2),
        true,
        true,
        false,
        false,
        b,
    )?);
    Ok(())
}
fn det_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(well_conditioned(1024, 1).det(b)?);
    Ok(())
}
fn slogdet_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let (s, l) = well_conditioned(1024, 1).slogdet(b)?;
    consume_many([s, l]);
    Ok(())
}
fn inv_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(well_conditioned(768, 1).inv(b)?);
    Ok(())
}
fn pinv_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(tensor_f64(&[512, 256], 1).pinv(b)?);
    Ok(())
}
fn norm_f64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(tensor_f64(&[2048, 2048], 1).norm(None, Some(&[0, 1]), false, b)?);
    Ok(())
}
// Complex.
fn conj_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.conj(tensor_c64(&[16_777_216], 1))?);
    Ok(())
}
fn mul_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.mul(tensor_c64(&[8_388_608], 1), tensor_c64(&[8_388_608], 2))?);
    Ok(())
}
fn div_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.div(
        tensor_c64(&[8_388_608], 1),
        tensor_c64_constant(&[8_388_608], Complex64::new(1.5, 0.25)),
    )?);
    Ok(())
}
fn exp_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.exp(tensor_c64(&[4_194_304], 1))?);
    Ok(())
}
fn log_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.log(tensor_c64_constant(&[4_194_304], Complex64::new(1.5, 0.25)))?);
    Ok(())
}
fn dot_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(b.dot_general(
        tensor_c64(&[640, 640], 1),
        tensor_c64(&[640, 640], 2),
        &DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    )?);
    Ok(())
}
fn svd_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let (u, s, vt) = tensor_c64(&[160, 160], 1).svd(b)?;
    consume_many([u, s, vt]);
    Ok(())
}
fn qr_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let (q, r) = tensor_c64(&[256, 256], 1).qr(b)?;
    consume_many([q, r]);
    Ok(())
}
fn eig_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    let (w, v) = tensor_c64(&[112, 112], 1).eig(b)?;
    consume_many([w, v]);
    Ok(())
}
fn solve_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(well_conditioned_c64(384, 1).solve(tensor_c64(&[384, 8], 2), b)?);
    Ok(())
}
fn cholesky_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(hpd_c64(448, 1).cholesky(b)?);
    Ok(())
}
fn norm_c64(b: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    consume(tensor_c64(&[2048, 1536], 1).norm(None, Some(&[0, 1]), false, b)?);
    Ok(())
}
