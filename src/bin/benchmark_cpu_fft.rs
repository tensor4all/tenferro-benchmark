//! CPU FFT benchmark runner for tenferro-rs public FFT APIs.

use std::env;
use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use num_complex::{Complex32, Complex64};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_fft::{
    EagerTensorFftExt, FftExecutor, FftNorm, TensorFftExt, TensorReadFftExt, TracedTensorFftExt,
};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{Tensor, TensorRead};

type BenchResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy)]
enum Op {
    Fft,
    Ifft,
    Rfft,
    Irfft,
}

impl Op {
    fn as_str(self) -> &'static str {
        match self {
            Self::Fft => "fft",
            Self::Ifft => "ifft",
            Self::Rfft => "rfft",
            Self::Irfft => "irfft",
        }
    }
}

#[derive(Clone, Copy)]
enum DTypeCase {
    F32,
    F64,
    C32,
    C64,
}

impl DTypeCase {
    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::C32 => "c32",
            Self::C64 => "c64",
        }
    }
}

#[derive(Clone, Copy)]
enum TenferroMode {
    Immediate,
    Read,
    ExecutorCached,
    Eager,
    Trace,
}

impl TenferroMode {
    fn backend_name(self) -> &'static str {
        match self {
            Self::Immediate => "tenferro-fft-immediate",
            Self::Read => "tenferro-fft-read",
            Self::ExecutorCached => "tenferro-fft-executor-cached",
            Self::Eager => "tenferro-fft-eager",
            Self::Trace => "tenferro-fft-trace",
        }
    }
}

struct Args {
    output: PathBuf,
    num_threads: usize,
    runs: usize,
    warmups: usize,
    lengths: Vec<usize>,
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

    for &n in &args.lengths {
        for (op, dtype) in [
            (Op::Fft, DTypeCase::C32),
            (Op::Fft, DTypeCase::C64),
            (Op::Ifft, DTypeCase::C32),
            (Op::Ifft, DTypeCase::C64),
            (Op::Rfft, DTypeCase::F32),
            (Op::Rfft, DTypeCase::F64),
            (Op::Irfft, DTypeCase::C32),
            (Op::Irfft, DTypeCase::C64),
        ] {
            for mode in [
                TenferroMode::Immediate,
                TenferroMode::Read,
                TenferroMode::ExecutorCached,
                TenferroMode::Eager,
                TenferroMode::Trace,
            ] {
                emit_case(&mut writer, &args, op, dtype, n, mode)?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

fn parse_args() -> BenchResult<Args> {
    let mut output = None;
    let mut num_threads = None;
    let mut lengths = lengths_from_env();
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
            "--lengths" => {
                lengths = parse_lengths(&iter.next().ok_or("--lengths requires a value")?)?;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    let output = output.ok_or("--output is required")?;
    let num_threads = num_threads
        .or_else(|| env::var("RAYON_NUM_THREADS").ok()?.parse::<usize>().ok())
        .unwrap_or(1);
    let profile = env::var("PUBLICATION_GATE_PROFILE")
        .unwrap_or_else(|_| "quick".to_string())
        .to_ascii_lowercase();
    let runs = env::var("BENCH_RUNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(if profile == "full" { 15 } else { 7 });
    let warmups = env::var("BENCH_WARMUPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3);
    Ok(Args {
        output,
        num_threads,
        runs,
        warmups,
        lengths,
    })
}

fn print_help() {
    println!("Usage: benchmark_cpu_fft --output <csv> [--num-threads N] [--lengths 1024,65536]");
}

fn lengths_from_env() -> Vec<usize> {
    env::var("FFT_BENCH_LENGTHS")
        .ok()
        .and_then(|value| parse_lengths(&value).ok())
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![1_048_576])
}

fn parse_lengths(value: &str) -> BenchResult<Vec<usize>> {
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| Ok(part.trim().parse::<usize>()?))
        .collect()
}

fn emit_case(
    writer: &mut impl Write,
    args: &Args,
    op: Op,
    dtype: DTypeCase,
    n: usize,
    mode: TenferroMode,
) -> BenchResult<()> {
    let input = input_tensor(op, dtype, n)?;
    let benchmark = op.as_str();
    let shape = format!("1d_n{n}");
    let backend = mode.backend_name();
    let notes = match mode {
        TenferroMode::Immediate => "one-shot TensorFftExt call; no caller-owned plan cache",
        TenferroMode::Read => {
            "TensorReadFftExt on an owned contiguous TensorRead; materialization and one-shot planning included"
        }
        TenferroMode::ExecutorCached => "FftExecutor reused across warmups and measured runs",
        TenferroMode::Eager => "EagerTensorFftExt with one reused eager runtime and input",
        TenferroMode::Trace => {
            "TracedTensorFftExt graph compiled once; compiled program reused across runs"
        }
    };

    let measured = match match mode {
        TenferroMode::Trace => time_trace_case(args, op, &input, n),
        TenferroMode::Eager => time_eager_case(args, op, &input, n),
        _ => time_case(args, mode, op, &input, n),
    } {
        Ok((median_ms, iqr_ms)) => {
            writeln!(
                writer,
                "cpu/fft,{benchmark},{},{},{shape},{backend},{median_ms:.6},{iqr_ms:.6},ok,\"{notes}\"",
                dtype.as_str(),
                args.num_threads,
            )?;
            Ok(())
        }
        Err(err) => {
            writeln!(
                writer,
                "cpu/fft,{benchmark},{},{},{shape},{backend},,,failed,\"{}\"",
                dtype.as_str(),
                args.num_threads,
                csv_escape(&err.to_string()),
            )?;
            Ok(())
        }
    };
    measured
}

fn time_case(
    args: &Args,
    mode: TenferroMode,
    op: Op,
    input: &Tensor,
    n: usize,
) -> BenchResult<(f64, f64)> {
    let mut backend = cpu_backend_from_env()?;
    let mut executor = FftExecutor::default();

    for _ in 0..args.warmups {
        consume(run_fft(mode, op, input, n, &mut backend, &mut executor)?);
    }

    let mut times = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let start = Instant::now();
        let out = run_fft(mode, op, input, n, &mut backend, &mut executor)?;
        consume(out);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(median_iqr(&times))
}

fn time_trace_case(args: &Args, op: Op, input: &Tensor, n: usize) -> BenchResult<(f64, f64)> {
    // Constant construction, graph construction, compilation, runtime setup,
    // and FFT planning warmup are all outside the measured region.
    let traced = TracedTensor::from_tensor_concrete_shape(input.clone())?;
    let output = match op {
        Op::Fft => traced.fft(None, -1, FftNorm::Backward)?,
        Op::Ifft => traced.ifft(None, -1, FftNorm::Backward)?,
        Op::Rfft => traced.rfft(None, -1, FftNorm::Backward)?,
        Op::Irfft => traced.irfft(Some(n), -1, FftNorm::Backward)?,
    };
    let program = GraphCompiler::new().compile(&output)?;
    let runtime = cpu_trace_runtime()?;

    for _ in 0..args.warmups {
        consume(runtime.run_compiled(&program, &[])?.remove(0));
    }
    let mut times = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let start = Instant::now();
        consume(runtime.run_compiled(&program, &[])?.remove(0));
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(median_iqr(&times))
}

fn time_eager_case(args: &Args, op: Op, input: &Tensor, n: usize) -> BenchResult<(f64, f64)> {
    let runtime = EagerRuntime::with_cpu_backend(cpu_backend_from_env()?);
    let input = EagerTensor::from_tensor_in(input.clone(), runtime)?;
    let run = || -> Result<EagerTensor, tenferro_ad::Error> {
        match op {
            Op::Fft => input.fft(None, -1, FftNorm::Backward),
            Op::Ifft => input.ifft(None, -1, FftNorm::Backward),
            Op::Rfft => input.rfft(None, -1, FftNorm::Backward),
            Op::Irfft => input.irfft(Some(n), -1, FftNorm::Backward),
        }
    };

    for _ in 0..args.warmups {
        consume_eager(run()?);
    }
    let mut times = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let start = Instant::now();
        consume_eager(run()?);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(median_iqr(&times))
}

fn cpu_trace_runtime() -> BenchResult<Runtime> {
    let backend = cpu_backend_from_env()?;
    let engine_id = tenferro_cpu::runtime_engine_id()?;
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend)?)?;
    builder.install_extension_module(tenferro_fft::extension_module::<CpuBackend>(engine_id)?)?;
    Ok(builder.build()?)
}

fn run_fft(
    mode: TenferroMode,
    op: Op,
    input: &Tensor,
    n: usize,
    backend: &mut CpuBackend,
    executor: &mut FftExecutor,
) -> tenferro_tensor::Result<Tensor> {
    match (mode, op) {
        (TenferroMode::Immediate, Op::Fft) => input.fft(None, -1, FftNorm::Backward, backend),
        (TenferroMode::Immediate, Op::Ifft) => input.ifft(None, -1, FftNorm::Backward, backend),
        (TenferroMode::Immediate, Op::Rfft) => input.rfft(None, -1, FftNorm::Backward, backend),
        (TenferroMode::Immediate, Op::Irfft) => {
            input.irfft(Some(n), -1, FftNorm::Backward, backend)
        }
        (TenferroMode::Read, Op::Fft) => {
            TensorRead::from_tensor(input).fft_read(None, -1, FftNorm::Backward, backend)
        }
        (TenferroMode::Read, Op::Ifft) => {
            TensorRead::from_tensor(input).ifft_read(None, -1, FftNorm::Backward, backend)
        }
        (TenferroMode::Read, Op::Rfft) => {
            TensorRead::from_tensor(input).rfft_read(None, -1, FftNorm::Backward, backend)
        }
        (TenferroMode::Read, Op::Irfft) => {
            TensorRead::from_tensor(input).irfft_read(Some(n), -1, FftNorm::Backward, backend)
        }
        (TenferroMode::ExecutorCached, Op::Fft) => {
            executor.fft(input, None, -1, FftNorm::Backward, backend)
        }
        (TenferroMode::ExecutorCached, Op::Ifft) => {
            executor.ifft(input, None, -1, FftNorm::Backward, backend)
        }
        (TenferroMode::ExecutorCached, Op::Rfft) => {
            executor.rfft(input, None, -1, FftNorm::Backward, backend)
        }
        (TenferroMode::ExecutorCached, Op::Irfft) => {
            executor.irfft(input, Some(n), -1, FftNorm::Backward, backend)
        }
        (TenferroMode::Trace, _) => unreachable!("trace FFT uses time_trace_case"),
        (TenferroMode::Eager, _) => unreachable!("eager FFT uses time_eager_case"),
    }
}

fn consume(tensor: Tensor) {
    black_box(tensor.shape().len());
    black_box(tensor.dtype());
}

fn consume_eager(tensor: EagerTensor) {
    black_box(tensor.shape().len());
    black_box(tensor.dtype());
}

fn input_tensor(op: Op, dtype: DTypeCase, n: usize) -> tenferro_tensor::Result<Tensor> {
    match (op, dtype) {
        (Op::Fft | Op::Ifft, DTypeCase::C32) => {
            Tensor::from_vec_col_major(vec![n], complex32_data(n, 17))
        }
        (Op::Fft | Op::Ifft, DTypeCase::C64) => {
            Tensor::from_vec_col_major(vec![n], complex64_data(n, 17))
        }
        (Op::Rfft, DTypeCase::F32) => Tensor::from_vec_col_major(vec![n], real32_data(n, 29)),
        (Op::Rfft, DTypeCase::F64) => Tensor::from_vec_col_major(vec![n], real64_data(n, 29)),
        (Op::Irfft, DTypeCase::C32) => {
            let spectrum_len = n / 2 + 1;
            Tensor::from_vec_col_major(vec![spectrum_len], complex32_data(spectrum_len, 31))
        }
        (Op::Irfft, DTypeCase::C64) => {
            let spectrum_len = n / 2 + 1;
            Tensor::from_vec_col_major(vec![spectrum_len], complex64_data(spectrum_len, 31))
        }
        _ => unreachable!("invalid FFT dtype case"),
    }
}

fn real32_data(n: usize, seed: u64) -> Vec<f32> {
    (0..n).map(|i| pseudo_value(i, seed) as f32).collect()
}

fn real64_data(n: usize, seed: u64) -> Vec<f64> {
    (0..n).map(|i| pseudo_value(i, seed)).collect()
}

fn complex32_data(n: usize, seed: u64) -> Vec<Complex32> {
    (0..n)
        .map(|i| {
            Complex32::new(
                pseudo_value(i, seed) as f32,
                pseudo_value(i, seed + 1) as f32,
            )
        })
        .collect()
}

fn complex64_data(n: usize, seed: u64) -> Vec<Complex64> {
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

#[allow(dead_code)]
fn ensure_parent(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
