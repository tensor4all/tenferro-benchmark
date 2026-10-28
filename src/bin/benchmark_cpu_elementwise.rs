//! CPU elementwise unary-kernel benchmark suite (`cpu/elementwise`).
//!
//! Measures f64 `sqrt` / `rsqrt` / `exp` on `N = 33_554_432` elements (512 MiB
//! of traffic per call: read + write) through tenferro-rs's eager CPU API,
//! once per CPU buffer-pool configuration:
//!
//! - `pool=default`: `DEFAULT_MAX_RETAINED_CAPACITY_BYTES` = 100 MiB, set
//!   explicitly (the pool also honors `TENFERRO_BUFFER_POOL_MAX_RETAINED_BYTES`,
//!   which would otherwise move this arm).
//! - `pool=raised`: `CpuEngine::set_buffer_pool_limit_bytes(1 << 32)` (4 GiB).
//!
//! Every iteration drops the output tensor and only black-boxes its shape and
//! dtype, so nothing retains it. With the default limit a 256 MiB output is
//! never retained and each call first-touches a fresh allocation; the raised
//! limit lets the next call reuse the previous output buffer. The
//! default/raised ratio therefore measures allocation/first-touch overhead, not
//! the kernel — see `docs/elementwise-suite.md` and tensor4all/tenferro-rs#2284.
//!
//! This suite deliberately has no external reference implementation: JAX and
//! PyTorch differ in allocator and caching strategy, which is exactly the
//! difference the pool configurations isolate.
//!
//! Environment variables:
//! - `BENCH_OUTPUT`: CSV output path. Unset prints the rows to stdout.
//!
//! Threads are given by `--num-threads N` (default: rayon's global pool width,
//! which `scripts/run_elementwise.sh` sets from its argument). The engine owns
//! the workers: a one-thread run uses `CpuEngine::serial()`, a wider run gets a
//! private pool, and the runner checks the engine's own width and records it.

use std::env;
use std::fmt;
use std::fs::File;
use std::hint::black_box;
use std::io::{self, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor};
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_tensor::Tensor;

const SUITE_ID: &str = "cpu/elementwise";
const OPS: [&str; 3] = ["sqrt", "rsqrt", "exp"];
const N: usize = 33_554_432;
const DTYPE: &str = "f64";
const ELEMENT_BYTES: usize = 8;
const WARMUPS: usize = 3;
const RUNS: usize = 7;
/// The shipped retention limit (`DEFAULT_MAX_RETAINED_CAPACITY_BYTES`), pinned
/// explicitly: `BufferPool::new()` takes the limit from
/// `TENFERRO_BUFFER_POOL_MAX_RETAINED_BYTES` when that variable is set, so an
/// inherited value would silently turn the baseline arm into a second raised arm.
const DEFAULT_POOL_LIMIT_BYTES: usize = 100 * 1024 * 1024;
/// 4 GiB, well above the 256 MiB output, so the pool keeps the output buffer.
const RAISED_POOL_LIMIT_BYTES: usize = 1 << 32;

/// `pool=default` (None) and `pool=raised` (Some(4 GiB)) in report order.
const POOL_CONFIGS: [usize; 2] = [DEFAULT_POOL_LIMIT_BYTES, RAISED_POOL_LIMIT_BYTES];

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

struct Args {
    num_threads: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut num_threads: Option<usize> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--num-threads" => {
                let value = args.next().ok_or("--num-threads requires a value")?;
                let parsed = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --num-threads {value:?}: {err}"))?;
                num_threads = Some(parsed);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let num_threads = match num_threads {
        Some(0) => return Err("--num-threads must be >= 1".to_string()),
        Some(n) => n,
        None => rayon::current_num_threads(),
    };
    Ok(Args { num_threads })
}

/// An engine that computes on `num_threads` workers, with its own pool.
fn build_engine(num_threads: usize) -> Result<CpuEngine, String> {
    if num_threads <= 1 {
        return Ok(CpuEngine::serial());
    }
    let pool = Pool::builder()
        .threads(num_threads)
        .build()
        .map_err(|err| format!("failed to build a {num_threads}-thread pool: {err}"))?;
    Ok(CpuEngine::new(pool))
}

// ---------------------------------------------------------------------------
// Op dispatch
// ---------------------------------------------------------------------------

fn apply(
    session: &mut EagerSession<'_>,
    op: &str,
    x: &EagerTensor,
) -> tenferro_ad::Result<EagerTensor> {
    match op {
        "sqrt" => session.sqrt(x),
        "rsqrt" => session.rsqrt(x),
        "exp" => session.exp(x),
        other => unreachable!("unknown elementwise op {other}"),
    }
}

fn reference(op: &str, v: f64) -> f64 {
    match op {
        "sqrt" => v.sqrt(),
        "rsqrt" => 1.0 / v.sqrt(),
        "exp" => v.exp(),
        other => unreachable!("unknown elementwise op {other}"),
    }
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

fn median(samples: &[Duration]) -> Duration {
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        (samples[n / 2 - 1] + samples[n / 2]) / 2
    }
}

struct Timing {
    median_ms: f64,
    iqr_ms: f64,
    gbps: f64,
}

/// Consumes a measured output the way the harnesses do (shape and dtype only)
/// and returns its storage to the engine's pool.
fn release(y: EagerTensor) -> Result<(), String> {
    black_box(y.shape().len());
    let _ = black_box(y.dtype());
    drop(y);
    Ok(())
}

fn bench(
    ctx: &Arc<EagerRuntime>,
    engine: &mut CpuEngine,
    op: &str,
    x: &EagerTensor,
) -> Result<Timing, String> {
    // The timed region is the call itself: the returned output is consumed and
    // released after the clock stops, so the run metadata's
    // "returned outputs retained until after the timer stops" holds. The buffer
    // is back in the pool before the next sample starts, which is what lets the
    // raised configuration reuse it.
    let mut one = || -> Result<EagerTensor, String> {
        ctx.with_eager_session(engine, |session| apply(session, op, x))
            .map_err(|err| err.to_string())
    };

    for _ in 0..WARMUPS {
        release(one()?)?;
    }
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let started = Instant::now();
        let y = one()?;
        let elapsed = started.elapsed();
        samples.push(elapsed);
        release(y)?;
    }
    samples.sort();

    let med = median(&samples);
    let p25 = samples[samples.len() / 4];
    let p75 = samples[samples.len() * 3 / 4];
    let bytes = 2 * N * ELEMENT_BYTES;
    Ok(Timing {
        median_ms: med.as_secs_f64() * 1e3,
        iqr_ms: p75.saturating_sub(p25).as_secs_f64() * 1e3,
        gbps: bytes as f64 / med.as_secs_f64() / 1e9,
    })
}

/// One untimed call per op, checked against `std` at three sample points.
/// A wrong dispatch would otherwise show up only as a suspiciously fast cell.
fn verify(
    ctx: &Arc<EagerRuntime>,
    engine: &mut CpuEngine,
    op: &str,
    x: &EagerTensor,
    sample: impl Fn(usize) -> f64 + Copy,
) -> Result<(), String> {
    let y = ctx
        .with_eager_session(engine, |session| apply(session, op, x))
        .map_err(|err| format!("{op}: {err}"))?;
    let value = y.value().map_err(|err| format!("{op}: {err}"))?;
    let out = value
        .as_slice::<f64>()
        .map_err(|err| format!("{op}: {err}"))?;
    if out.len() != N {
        return Err(format!("{op}: output length {} != {N}", out.len()));
    }
    for index in [0, N / 2, N - 1] {
        let expected = reference(op, sample(index));
        let actual = out[index];
        let tolerance = 1e-12 * expected.abs().max(1.0);
        // A NaN output must fail here; `NaN > tolerance` alone would pass it.
        if actual.is_nan() || (actual - expected).abs() > tolerance {
            return Err(format!(
                "{op}: output[{index}] = {actual} but expected {expected}"
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

struct Row<'a> {
    benchmark: &'a str,
    threads: usize,
    pool_limit_bytes: usize,
    timing: Timing,
}

impl fmt::Display for Row<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{SUITE_ID},{},{DTYPE},{N},{},{},{:.3},{:.3},{:.3}",
            self.benchmark,
            self.threads,
            self.pool_limit_bytes,
            self.timing.median_ms,
            self.timing.iqr_ms,
            self.timing.gbps,
        )
    }
}

fn writer() -> Result<Box<dyn Write>, String> {
    match env::var("BENCH_OUTPUT") {
        Ok(path) => {
            let file =
                File::create(&path).map_err(|err| format!("cannot write {path}: {err}"))?;
            Ok(Box::new(io::BufWriter::new(file)))
        }
        Err(_) => Ok(Box::new(io::stdout())),
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn run() -> Result<(), String> {
    let args = parse_args()?;

    let ctx = EagerRuntime::new().map_err(|err| format!("eager runtime: {err}"))?;
    // 1.0 + [0, 0.5] repeats every 17 elements, so the verification samples can
    // recompute their expected value from the index instead of keeping a second
    // 256 MiB copy alive.
    let sample = |index: usize| 1.0 + (index % 17) as f64 / 32.0;
    let input: Vec<f64> = (0..N).map(sample).collect();
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![N], input)
            .map_err(|err| format!("input tensor: {err}"))?,
        ctx.clone(),
    )
    .map_err(|err| format!("input tensor: {err}"))?;

    let mut out = writer()?;
    writeln!(
        out,
        "suite,benchmark,dtype,shape,threads,pool_limit_bytes,median_ms,iqr_ms,gb_per_s"
    )
    .map_err(|err| format!("write header: {err}"))?;

    for pool_config in POOL_CONFIGS {
        // A fresh engine per configuration: warmup-retained buffers from one
        // configuration must not leak into the other.
        let mut engine = build_engine(args.num_threads)?;
        engine.set_buffer_pool_limit_bytes(pool_config);
        let pool_limit_bytes = engine.buffer_pool_limit_bytes();
        if engine.num_threads() != args.num_threads {
            return Err(format!(
                "engine reports {} threads, requested {}",
                engine.num_threads(),
                args.num_threads
            ));
        }
        eprintln!(
            "cpu/elementwise: threads={} pool_limit_bytes={pool_limit_bytes}",
            args.num_threads
        );

        for op in OPS {
            verify(&ctx, &mut engine, op, &x, sample)?;
            let timing = bench(&ctx, &mut engine, op, &x)?;
            let row = Row {
                benchmark: op,
                threads: args.num_threads,
                pool_limit_bytes,
                timing,
            };
            writeln!(out, "{row}").map_err(|err| format!("write row: {err}"))?;
            out.flush().map_err(|err| format!("flush: {err}"))?;
        }
    }
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("benchmark_cpu_elementwise: {err}");
        std::process::exit(1);
    }
}
