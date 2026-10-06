//! Executable case producer for cpu/perf_issues.
//!
//! Each case reproduces the workload of an open tenferro-rs performance issue
//! (the issue number is recorded next to the case in
//! `data/instances/perf_issues.json`). Backends are built only through the
//! public constructors that survive tenferro-rs #2004: `CpuBackend::new()`,
//! `CpuBackend::with_threads(n)`, `with_backend_session` and the public
//! concrete, eager and traced operations. No provider kind, provider bundle or
//! thread proof is used here.
//!
//! Numerical checks precede timing. Setup (inputs, backends, runtimes,
//! sessions, compilation) is outside the timer; each sample times many
//! declared operations in one wall-clock interval and retains every output
//! until the clock stops. Emits one JSON object on stdout.

use std::alloc::{GlobalAlloc, Layout, System};
use std::error::Error;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;
use serde::Serialize;
use serde_json::{json, Value};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::{runtime_engine_registration, CpuBackend};
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::{DType, GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{BackendSession, BackendSessionHost, DotGeneralConfig, Tensor, TensorRead};

type BoxError = Box<dyn Error + Send + Sync>;
type Result<T> = std::result::Result<T, BoxError>;

const CALIBRATION_MAX_ITERATIONS: usize = 1 << 20;
const CALIBRATION_DEADLINE_NS: u128 = 20_000_000_000;
/// Upper bound on the outputs retained inside one timed batch.
const RETENTION_BUDGET_BYTES: usize = 512 << 20;

// ---------------------------------------------------------------------------
// Allocation counters (#1995 ask 2). Counting is switched on only around the
// counter diagnostic; elsewhere the allocator forwards with one relaxed load.
// ---------------------------------------------------------------------------

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Count host allocations made by `f` (the calling thread and any worker).
fn count_allocations<R>(f: impl FnOnce() -> Result<R>) -> Result<(R, u64, u64)> {
    ALLOCATIONS.store(0, Ordering::SeqCst);
    ALLOCATED_BYTES.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    let result = f();
    COUNTING.store(false, Ordering::SeqCst);
    let value = result?;
    Ok((
        value,
        ALLOCATIONS.load(Ordering::SeqCst),
        ALLOCATED_BYTES.load(Ordering::SeqCst),
    ))
}

// ---------------------------------------------------------------------------
// Arguments and timing
// ---------------------------------------------------------------------------

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn arg_usize(name: &str, default: usize) -> Result<usize> {
    Ok(arg(name).map(|v| v.parse()).transpose()?.unwrap_or(default))
}

#[derive(Serialize)]
struct Sample {
    sample_index: usize,
    elapsed_ns: u128,
    iterations: usize,
}

struct Timing {
    warmups: usize,
    samples: usize,
    target_ns: u128,
}

struct Measurement {
    iterations: usize,
    calibrated_ns: u128,
    samples: Vec<Sample>,
}

/// Time many operations per wall-clock interval. `retained_bytes` is the
/// output size kept alive per operation; it bounds the batch size.
fn measure<O>(
    timing: &Timing,
    retained_bytes: usize,
    mut op: impl FnMut() -> Result<O>,
) -> Result<Measurement> {
    for _ in 0..timing.warmups.max(1) {
        black_box(op()?);
    }
    let cap = (RETENTION_BUDGET_BYTES / retained_bytes.max(1)).clamp(1, CALIBRATION_MAX_ITERATIONS);
    let mut batch = |iterations: usize| -> Result<u128> {
        let mut outputs = Vec::with_capacity(iterations);
        let start = Instant::now();
        for _ in 0..iterations {
            outputs.push(op()?);
        }
        let elapsed = start.elapsed().as_nanos();
        black_box(&outputs);
        drop(outputs);
        Ok(elapsed)
    };
    let calibration_start = Instant::now();
    let mut iterations = 1usize;
    let calibrated_ns = loop {
        let elapsed = batch(iterations)?;
        if elapsed >= timing.target_ns || iterations >= cap {
            break elapsed;
        }
        if calibration_start.elapsed().as_nanos() >= CALIBRATION_DEADLINE_NS {
            return Err("calibration deadline exceeded".into());
        }
        iterations = (iterations * 2).min(cap);
    };
    let mut samples = Vec::with_capacity(timing.samples);
    for sample_index in 0..timing.samples {
        let elapsed_ns = batch(iterations)?;
        if elapsed_ns == 0 {
            return Err("timed sample elapsed zero nanoseconds".into());
        }
        samples.push(Sample {
            sample_index,
            elapsed_ns,
            iterations,
        });
    }
    Ok(Measurement {
        iterations,
        calibrated_ns,
        samples,
    })
}

/// What a case reports besides its samples.
struct CaseOutcome {
    max_rel_error: f64,
    measurement: Option<Measurement>,
    timer: Vec<&'static str>,
    outside_timer: Vec<&'static str>,
    extra: Value,
}

// ---------------------------------------------------------------------------
// Deterministic data and checks
// ---------------------------------------------------------------------------

fn values(n: usize, seed: u64) -> Vec<f64> {
    (0..n as u64)
        .map(|i| {
            let x = i
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(97))
                % 2001;
            (x as f64 / 1000.0 - 1.0) * 0.5
        })
        .collect()
}

fn values_f32(n: usize, seed: u64) -> Vec<f32> {
    values(n, seed).into_iter().map(|v| v as f32).collect()
}

fn complex_values(n: usize, seed: u64) -> Vec<Complex64> {
    let re = values(n, seed);
    let im = values(n, seed + 1);
    re.into_iter()
        .zip(im)
        .map(|(r, i)| Complex64::new(r, i))
        .collect()
}

fn rel_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1.0)
}

fn check_tol(error: f64, tol: f64, what: &str) -> Result<f64> {
    if error.is_finite() && error <= tol {
        Ok(error)
    } else {
        Err(format!("{what}: relative error {error:e} exceeds {tol:e}").into())
    }
}

fn matmul_config(lc: &[usize], rc: &[usize]) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: lc.into(),
        rhs_contracting_dims: rc.into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    }
}

fn param_usize(params: &Value, key: &str) -> Result<usize> {
    params[key]
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("case parameter {key} missing").into())
}

fn param_str<'a>(params: &'a Value, key: &str) -> Result<&'a str> {
    params[key]
        .as_str()
        .ok_or_else(|| format!("case parameter {key} missing").into())
}

fn eager_runtime(backend: CpuBackend) -> Result<Arc<EagerRuntime>> {
    Ok(EagerRuntime::with_cpu_backend(backend)?)
}

fn eager_const(runtime: &Arc<EagerRuntime>, tensor: Tensor) -> Result<EagerTensor> {
    Ok(runtime.with_eager_session(|s| s.constant_from(tensor))??)
}

// ---------------------------------------------------------------------------
// #1992 / #2003 / #1995: decode projections y = W^T x, f32
// ---------------------------------------------------------------------------

/// `x` is column-major `(in, len)`, `w` column-major `(in, out)`; the output is
/// column-major `(len, out)`, the MWE's `dot_general(x, w)` contracting axis 0.
fn decode_projection(params: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    let (din, dout, len) = (
        param_usize(params, "in")?,
        param_usize(params, "out")?,
        param_usize(params, "len")?,
    );
    let arm = param_str(params, "arm")?;
    let x = values_f32(din * len, 1);
    let w = values_f32(din * dout, 2);
    let out_bytes = len * dout * 4;
    // Independent f64 reference on a fixed sample of output entries.
    let probes: Vec<(usize, usize)> = (0..256)
        .map(|i| ((i * 7) % len, (i * 131) % dout))
        .collect();
    let check = |y: &[f32]| -> Result<f64> {
        let mut worst = 0.0f64;
        for &(i, j) in &probes {
            let expected: f64 = (0..din)
                .map(|p| x[p + i * din] as f64 * w[p + j * din] as f64)
                .sum();
            worst = worst.max(rel_error(y[i + j * len] as f64, expected));
        }
        check_tol(worst, 1e-4, "decode projection")
    };
    let cfg = matmul_config(&[0], &[0]);
    let (error, measurement, timer, outside): (f64, Measurement, Vec<&str>, Vec<&str>) = match arm {
        "eager-shared" | "eager-per-call" => {
            let runtime = eager_runtime(CpuBackend::new())?;
            let xt = eager_const(
                &runtime,
                Tensor::from_vec_col_major(vec![din, len], x.clone())?,
            )?;
            let wt = eager_const(
                &runtime,
                Tensor::from_vec_col_major(vec![din, dout], w.clone())?,
            )?;
            let y = runtime.with_eager_session(|s| s.dot_general(&xt, &wt, cfg.clone()))??;
            let error = check(y.to_tensor()?.as_slice::<f32>()?)?;
            if arm == "eager-shared" {
                let measurement = runtime.with_eager_session(|s| {
                    measure(timing, out_bytes, || {
                        Ok(s.dot_general(&xt, &wt, cfg.clone())?)
                    })
                })??;
                (
                    error,
                    measurement,
                    vec!["eager_dot_general", "output_allocation"],
                    vec!["eager_session_entry_exit"],
                )
            } else {
                let measurement = measure(timing, out_bytes, || {
                    Ok(runtime.with_eager_session(|s| s.dot_general(&xt, &wt, cfg.clone()))??)
                })?;
                (
                    error,
                    measurement,
                    vec![
                        "eager_session_entry_exit",
                        "eager_dot_general",
                        "output_allocation",
                    ],
                    vec![],
                )
            }
        }
        "faer-direct" => {
            use faer::{Accum, MatMut, MatRef, Par};
            let par = if threads <= 1 {
                Par::Seq
            } else {
                Par::rayon(threads)
            };
            let a = unsafe { MatRef::<f32>::from_raw_parts(x.as_ptr(), din, len, 1, din as isize) }
                .transpose();
            let b =
                unsafe { MatRef::<f32>::from_raw_parts(w.as_ptr(), din, dout, 1, din as isize) };
            let mut y = vec![f32::NAN; len * dout];
            let run = |y: &mut [f32]| {
                let mut c = unsafe {
                    MatMut::<f32>::from_raw_parts_mut(y.as_mut_ptr(), len, dout, 1, len as isize)
                };
                faer::linalg::matmul::matmul(&mut c, Accum::Replace, &a, &b, 1.0f32, par);
            };
            run(&mut y);
            let error = check(&y)?;
            let measurement = measure(timing, 0, || {
                run(&mut y);
                Ok(())
            })?;
            black_box(&y);
            (
                error,
                measurement,
                vec!["faer_matmul_into_preallocated_output"],
                vec!["output_allocation"],
            )
        }
        "host-sgemm" => {
            let mut y = vec![f32::NAN; len * dout];
            let run = |y: &mut [f32]| unsafe {
                matrixmultiply::sgemm(
                    len,
                    din,
                    dout,
                    1.0,
                    x.as_ptr(),
                    din as isize,
                    1,
                    w.as_ptr(),
                    1,
                    din as isize,
                    0.0,
                    y.as_mut_ptr(),
                    1,
                    len as isize,
                );
            };
            run(&mut y);
            let error = check(&y)?;
            let measurement = measure(timing, 0, || {
                run(&mut y);
                Ok(())
            })?;
            black_box(&y);
            (
                error,
                measurement,
                vec!["matrixmultiply_sgemm_into_preallocated_output"],
                vec!["output_allocation"],
            )
        }
        _ => return Err(format!("unknown decode projection arm {arm}").into()),
    };
    let mut outside_timer = vec![
        "backend_construction",
        "input_construction",
        "correctness_check",
    ];
    outside_timer.extend(outside);
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer,
        outside_timer,
        extra: json!({}),
    })
}

// ---------------------------------------------------------------------------
// #1995 ask 2: copy / allocation volume of a decode block (counters, untimed)
// ---------------------------------------------------------------------------

fn copy_volume(params: &Value) -> Result<CaseOutcome> {
    let (d, len) = (param_usize(params, "d")?, param_usize(params, "len")?);
    let repeats = 16usize;
    let runtime = eager_runtime(CpuBackend::new())?;
    let x = eager_const(
        &runtime,
        Tensor::from_vec_col_major(vec![d, len], values_f32(d * len, 3))?,
    )?;
    let v = eager_const(
        &runtime,
        Tensor::from_vec_col_major(vec![d], values_f32(d, 4))?,
    )?;
    let w = eager_const(
        &runtime,
        Tensor::from_vec_col_major(vec![d, d], values_f32(d * d, 5))?,
    )?;
    let eps = eager_const(
        &runtime,
        Tensor::from_vec_col_major(vec![len], vec![1e-5f32; len])?,
    )?;
    let full = d * len * 4;
    let mut rows = Vec::new();
    // Each op runs once untimed, then `repeats` times under the counters.
    type Op<'a> = Box<
        dyn Fn(&mut tenferro_ad::EagerSession<'_>) -> tenferro_ad::Result<EagerTensor>
            + Send
            + Sync
            + 'a,
    >;
    let ops: Vec<(&str, usize, Op)> = vec![
        (
            "broadcast_in_dim (d)->(d,len)",
            full,
            Box::new(|s| s.broadcast_in_dim(&v, &[d, len], &[0])),
        ),
        (
            "reshape (d,len)->(d,len,1)",
            full,
            Box::new(|s| s.reshape(&x, vec![d, len, 1])),
        ),
        (
            "reshape (d,len)->(d*len)",
            full,
            Box::new(|s| s.reshape(&x, vec![d * len])),
        ),
        (
            "transpose (d,len)->(len,d)",
            full,
            Box::new(|s| s.transpose(&x, &[1, 0])),
        ),
        ("add (d,len)+(d,len)", full, Box::new(|s| s.add(&x, &x))),
        ("mul (d,len)*(d,len)", full, Box::new(|s| s.mul(&x, &x))),
        (
            "reduce_sum_squares axis 0",
            len * 4,
            Box::new(|s| s.reduce_sum_squares(&x, &[0])),
        ),
        ("rsqrt (len)", len * 4, Box::new(|s| s.rsqrt(&eps))),
        (
            "dot_general projection (d,len)x(d,d)",
            len * d * 4,
            Box::new(|s| s.dot_general(&x, &w, matmul_config(&[0], &[0]))),
        ),
    ];
    for (name, output_bytes, op) in &ops {
        runtime.with_eager_session(|s| op(s))??;
        let (_, allocations, bytes) = count_allocations(|| {
            let mut kept = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                kept.push(runtime.with_eager_session(|s| op(s))??);
            }
            black_box(&kept);
            Ok(())
        })?;
        rows.push(json!({
            "op": name,
            "output_bytes": output_bytes,
            "host_allocations_per_op": allocations as f64 / repeats as f64,
            "host_allocated_bytes_per_op": bytes as f64 / repeats as f64,
            "allocated_bytes_over_output_bytes": bytes as f64 / repeats as f64 / *output_bytes as f64,
        }));
    }
    Ok(CaseOutcome {
        max_rel_error: 0.0,
        measurement: None,
        timer: vec![],
        outside_timer: vec!["everything (counter diagnostic, no timing)"],
        extra: json!({
            "counters": rows,
            "counter_source": "counting global allocator in this process, per eager op including its own session entry",
            "unavailable": ["per-op layout-copy count", "copied bytes (pooled buffer reuse is invisible to the host allocator)"],
        }),
    })
}

// ---------------------------------------------------------------------------
// #1900: 256^3 GEMM, c64 vs f64, one thread
// ---------------------------------------------------------------------------

fn gemm_mm256(params: &Value, timing: &Timing) -> Result<CaseOutcome> {
    let n = param_usize(params, "n")?;
    let dtype = param_str(params, "dtype")?;
    let mut backend = CpuBackend::with_threads(1)?;
    let cfg = matmul_config(&[1], &[0]);
    let (a, b, expected) = match dtype {
        "f64" => {
            let (a, b) = (values(n * n, 6), values(n * n, 7));
            let probes: Vec<f64> = (0..64)
                .map(|t| {
                    let (i, j) = ((t * 37) % n, (t * 91) % n);
                    (0..n).map(|k| a[i + k * n] * b[k + j * n]).sum()
                })
                .collect();
            (
                Tensor::from_vec_col_major(vec![n, n], a)?,
                Tensor::from_vec_col_major(vec![n, n], b)?,
                probes
                    .into_iter()
                    .map(|v| Complex64::new(v, 0.0))
                    .collect::<Vec<_>>(),
            )
        }
        "c64" => {
            let (a, b) = (complex_values(n * n, 6), complex_values(n * n, 8));
            let probes: Vec<Complex64> = (0..64)
                .map(|t| {
                    let (i, j) = ((t * 37) % n, (t * 91) % n);
                    (0..n).map(|k| a[i + k * n] * b[k + j * n]).sum()
                })
                .collect();
            (
                Tensor::from_vec_col_major(vec![n, n], a)?,
                Tensor::from_vec_col_major(vec![n, n], b)?,
                probes,
            )
        }
        _ => return Err(format!("unsupported dtype {dtype}").into()),
    };
    let elem = if dtype == "c64" { 16 } else { 8 };
    let (error, measurement) =
        backend.with_backend_session(|s| -> Result<(f64, Measurement)> {
            let c = s.dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
                &cfg,
            )?;
            let mut worst = 0.0f64;
            for (t, e) in expected.iter().enumerate() {
                let (i, j) = ((t * 37) % n, (t * 91) % n);
                let got = if dtype == "c64" {
                    c.as_slice::<Complex64>()?[i + j * n]
                } else {
                    Complex64::new(c.as_slice::<f64>()?[i + j * n], 0.0)
                };
                worst = worst.max((got - e).norm() / e.norm().max(1.0));
            }
            let error = check_tol(worst, 1e-10, "mm256")?;
            let measurement = measure(timing, n * n * elem, || {
                Ok(s.dot_general_read(
                    TensorRead::from_tensor(&a),
                    TensorRead::from_tensor(&b),
                    &cfg,
                )?)
            })?;
            Ok((error, measurement))
        })??;
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer: vec!["dot_general_read", "output_allocation"],
        outside_timer: vec![
            "backend_construction (CpuBackend::with_threads(1))",
            "input_construction",
            "session_entry_exit",
            "correctness_check",
        ],
        extra: json!({"flops_per_op": if dtype == "c64" { 8 } else { 2 } * n * n * n}),
    })
}

// ---------------------------------------------------------------------------
// #1615: conjugated rank-1 dot
// ---------------------------------------------------------------------------

fn rank1_dot(params: &Value, timing: &Timing) -> Result<CaseOutcome> {
    let len = param_usize(params, "len")?;
    let dtype = param_str(params, "dtype")?;
    let arm = param_str(params, "arm")?;
    let complex = dtype == "c64";
    let (xr, yr) = (complex_values(len, 9), complex_values(len, 11));
    let (x, y): (Vec<Complex64>, Vec<Complex64>) = if complex {
        (xr, yr)
    } else {
        (
            xr.iter().map(|v| Complex64::new(v.re, 0.0)).collect(),
            yr.iter().map(|v| Complex64::new(v.re, 0.0)).collect(),
        )
    };
    let expected: Complex64 = x.iter().zip(&y).map(|(a, b)| a.conj() * b).sum();
    let check = |got: Complex64| {
        check_tol(
            (got - expected).norm() / expected.norm().max(1.0),
            1e-10,
            "conjugated dot",
        )
    };
    match arm {
        "dot-general-rank1" => {
            // dot_general has no conjugation; the conjugated operand is
            // prepared once outside timing, so the timed route is the plain
            // rank-1 contraction the issue measured.
            let (lhs, rhs) = if complex {
                (
                    Tensor::from_vec_col_major(vec![len], x.iter().map(|v| v.conj()).collect())?,
                    Tensor::from_vec_col_major(vec![len], y.clone())?,
                )
            } else {
                (
                    Tensor::from_vec_col_major(vec![len], x.iter().map(|v| v.re).collect())?,
                    Tensor::from_vec_col_major(vec![len], y.iter().map(|v| v.re).collect())?,
                )
            };
            let cfg = matmul_config(&[0], &[0]);
            let mut backend = CpuBackend::new();
            let (error, measurement) =
                backend.with_backend_session(|s| -> Result<(f64, Measurement)> {
                    let out = s.dot_general_read(
                        TensorRead::from_tensor(&lhs),
                        TensorRead::from_tensor(&rhs),
                        &cfg,
                    )?;
                    let got = if complex {
                        out.as_slice::<Complex64>()?[0]
                    } else {
                        Complex64::new(out.as_slice::<f64>()?[0], 0.0)
                    };
                    let error = check(got)?;
                    let measurement = measure(timing, 64, || {
                        Ok(s.dot_general_read(
                            TensorRead::from_tensor(&lhs),
                            TensorRead::from_tensor(&rhs),
                            &cfg,
                        )?)
                    })?;
                    Ok((error, measurement))
                })??;
            Ok(CaseOutcome {
                max_rel_error: error,
                measurement: Some(measurement),
                timer: vec![
                    "dot_general_read (rank-1, rank-0 tensor output)",
                    "output_allocation",
                ],
                outside_timer: vec![
                    "backend_construction",
                    "input_construction",
                    "operand_conjugation",
                    "session_entry_exit",
                    "correctness_check",
                ],
                extra: json!({}),
            })
        }
        "portable-loop" => {
            // Allocation-free scalar reference over borrowed slices.
            let measurement;
            let error;
            if complex {
                let dot = || -> Complex64 { x.iter().zip(&y).map(|(a, b)| a.conj() * b).sum() };
                error = check(dot())?;
                measurement = measure(timing, 0, || Ok(black_box(dot())))?;
            } else {
                let (xs, ys): (Vec<f64>, Vec<f64>) = (
                    x.iter().map(|v| v.re).collect(),
                    y.iter().map(|v| v.re).collect(),
                );
                let dot = || -> f64 { xs.iter().zip(&ys).map(|(a, b)| a * b).sum() };
                error = check(Complex64::new(dot(), 0.0))?;
                measurement = measure(timing, 0, || Ok(black_box(dot())))?;
            }
            Ok(CaseOutcome {
                max_rel_error: error,
                measurement: Some(measurement),
                timer: vec!["sequential Rust loop over borrowed slices, scalar result"],
                outside_timer: vec!["input_construction", "correctness_check"],
                extra: json!({}),
            })
        }
        _ => Err(format!("unknown rank-1 dot arm {arm}").into()),
    }
}

// ---------------------------------------------------------------------------
// #2007: per-batch loop of solve / triangular_solve vs one batched solve
// ---------------------------------------------------------------------------

fn small_solves(params: &Value, timing: &Timing) -> Result<CaseOutcome> {
    let (k, batch) = (param_usize(params, "k")?, param_usize(params, "batch")?);
    let arm = param_str(params, "arm")?;
    // Diagonally dominant lower-triangular items with a unit diagonal, so the
    // same matrix is valid for solve and for a unit-diagonal triangular solve.
    let item = |b: usize| -> Vec<f64> {
        let raw = values(k * k, 20 + b as u64);
        let mut a = vec![0.0; k * k];
        for j in 0..k {
            for i in j..k {
                a[i + j * k] = if i == j {
                    1.0
                } else {
                    raw[i + j * k] / k as f64
                };
            }
        }
        a
    };
    let rhs_item = |b: usize| values(k * k, 500 + b as u64);
    let items: Vec<(Vec<f64>, Vec<f64>)> = (0..batch).map(|b| (item(b), rhs_item(b))).collect();
    let residual = |a: &[f64], x: &[f64], b: &[f64]| -> f64 {
        let mut worst = 0.0f64;
        for c in 0..k {
            for i in 0..k {
                let ax: f64 = (0..k).map(|p| a[i + p * k] * x[p + c * k]).sum();
                worst = worst.max(rel_error(ax, b[i + c * k]));
            }
        }
        worst
    };
    let mut backend = CpuBackend::new();
    let bytes = batch * k * k * 8;
    let (error, measurement, timer) = match arm {
        "solve-loop" | "triangular-solve-loop" => {
            let tensors: Vec<(Tensor, Tensor)> = items
                .iter()
                .map(|(a, b)| {
                    Ok((
                        Tensor::from_vec_col_major(vec![k, k], a.clone())?,
                        Tensor::from_vec_col_major(vec![k, k], b.clone())?,
                    ))
                })
                .collect::<Result<_>>()?;
            let triangular = arm == "triangular-solve-loop";
            let one = |s: &mut dyn BackendSession, a: &Tensor, b: &Tensor| -> Result<Tensor> {
                Ok(if triangular {
                    a.triangular_solve(b, true, true, false, true, s)?
                } else {
                    a.solve(b, s)?
                })
            };
            backend.with_backend_session(|s| -> Result<(f64, Measurement, Vec<&str>)> {
                let mut worst = 0.0f64;
                for ((a, b), (ad, bd)) in tensors.iter().zip(&items) {
                    worst = worst.max(residual(ad, one(s, a, b)?.as_slice::<f64>()?, bd));
                }
                let error = check_tol(worst, 1e-10, "per-item solve residual")?;
                let measurement = measure(timing, bytes, || {
                    tensors
                        .iter()
                        .map(|(a, b)| one(s, a, b))
                        .collect::<Result<Vec<_>>>()
                })?;
                Ok((
                    error,
                    measurement,
                    vec!["batch x rank-2 solve calls", "output_allocation per item"],
                ))
            })??
        }
        "solve-batched" => {
            let mut a_all = Vec::with_capacity(bytes / 8);
            let mut b_all = Vec::with_capacity(bytes / 8);
            for (a, b) in &items {
                a_all.extend_from_slice(a);
                b_all.extend_from_slice(b);
            }
            let a = Tensor::from_vec_col_major(vec![k, k, batch], a_all)?;
            let b = Tensor::from_vec_col_major(vec![k, k, batch], b_all)?;
            backend.with_backend_session(|s| -> Result<(f64, Measurement, Vec<&str>)> {
                let x = a.solve(&b, s)?;
                let xs = x.as_slice::<f64>()?;
                let mut worst = 0.0f64;
                for (i, (ad, bd)) in items.iter().enumerate() {
                    worst = worst.max(residual(ad, &xs[i * k * k..(i + 1) * k * k], bd));
                }
                let error = check_tol(worst, 1e-10, "batched solve residual")?;
                let measurement = measure(timing, bytes, || Ok(a.solve(&b, s)?))?;
                Ok((
                    error,
                    measurement,
                    vec![
                        "one batched solve call ([k, k, batch])",
                        "output_allocation",
                    ],
                ))
            })??
        }
        _ => return Err(format!("unknown solve arm {arm}").into()),
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer,
        outside_timer: vec![
            "backend_construction",
            "input_construction (items pre-sliced)",
            "session_entry_exit",
            "correctness_check",
        ],
        extra: json!({}),
    })
}

// ---------------------------------------------------------------------------
// #1803 A: eager backward with N unrelated live gradient leaves
// ---------------------------------------------------------------------------

fn backward_live_leaves(params: &Value, timing: &Timing) -> Result<CaseOutcome> {
    let leaves = param_usize(params, "unrelated_leaves")?;
    let runtime = eager_runtime(CpuBackend::new())?;
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?,
        runtime.clone(),
    )?;
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![0.5, -1.0, 1.5, 2.0])?,
        runtime.clone(),
    )?;
    // Unrelated leaves stay alive for the whole case; they never reach the loss.
    let unrelated: Vec<EagerTensor> = (0..leaves)
        .map(|i| {
            EagerTensor::requires_grad_in(
                Tensor::from_vec_col_major(vec![2, 2], values(4, 100 + i as u64))?,
                runtime.clone(),
            )
            .map_err(Into::into)
        })
        .collect::<Result<_>>()?;
    let workflow = || -> Result<()> {
        let c = runtime.with_eager_session(|s| s.matmul(&a, &b))??;
        let loss = runtime.with_eager_session(|s| s.reduce_sum(&c, None))??;
        loss.backward()?;
        Ok(())
    };
    workflow()?;
    // d(sum(A B))/dA = 1 B^T, d/dB = A^T 1 (column-major 2x2).
    let ga = a.grad()?.ok_or("missing gradient for A")?;
    let gb = b.grad()?.ok_or("missing gradient for B")?;
    let expected_a = [2.0, 2.0, 1.0, 1.0];
    let expected_b = [3.0, 7.0, 3.0, 7.0];
    let mut worst = 0.0f64;
    for (got, want) in ga
        .as_slice::<f64>()?
        .iter()
        .zip(expected_a)
        .chain(gb.as_slice::<f64>()?.iter().zip(expected_b))
    {
        worst = worst.max(rel_error(*got, want));
    }
    let error = check_tol(worst, 1e-12, "matmul backward gradient")?;
    a.clear_grad()?;
    b.clear_grad()?;
    let measurement = measure(timing, 0, workflow)?;
    black_box(&unrelated);
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer: vec![
            "eager matmul forward",
            "reduce_sum",
            "backward (gradients accumulate across a batch)",
            "internal eager session entries",
        ],
        outside_timer: vec![
            "eager_runtime_construction",
            "leaf_construction (including the unrelated leaves)",
            "correctness_check",
            "gradient reset between cases",
        ],
        extra: json!({"unrelated_live_leaves": leaves}),
    })
}

// ---------------------------------------------------------------------------
// #1990: tanh chain, compiled fused region vs eager per-op
// ---------------------------------------------------------------------------

fn tanh_chain(params: &Value, timing: &Timing) -> Result<CaseOutcome> {
    let (rows, cols, chain) = (
        param_usize(params, "rows")?,
        param_usize(params, "cols")?,
        param_usize(params, "chain")?,
    );
    let arm = param_str(params, "arm")?;
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) * 0.0001).sin())
        .collect();
    let probes: Vec<usize> = (0..64).map(|i| (i * 1031) % (rows * cols)).collect();
    let check = |y: &[f32]| -> Result<f64> {
        let mut worst = 0.0f64;
        for &p in &probes {
            let mut v = data[p] as f64;
            for _ in 0..chain {
                v = v.tanh();
            }
            worst = worst.max(rel_error(y[p] as f64, v));
        }
        check_tol(worst, 1e-4, "tanh chain")
    };
    let bytes = rows * cols * 4;
    let x_t = Tensor::from_vec_col_major(vec![rows, cols], data.clone())?;
    let (error, measurement, timer, outside) = match arm {
        "compiled-prepared" => {
            let x_in = TracedTensor::input_concrete_shape(DType::F32, &[rows, cols])?;
            let mut y = x_in.clone();
            for _ in 0..chain {
                y = y.tanh()?;
            }
            let program = GraphCompiler::new()
                .compile_with_input_specs(&y, &[(&x_in, DType::F32, &[rows, cols])])?;
            let backend = CpuBackend::new();
            let mut builder = Runtime::builder();
            builder.register_engine(runtime_engine_registration(&backend)?)?;
            let runtime = builder.build()?;
            let prepared = runtime.prepare_compiled(&program, &[&x_t])?;
            let out = runtime.run_prepared(&prepared, &[&x_t])?;
            let error = check(out[0].as_slice::<f32>()?)?;
            let measurement = measure(timing, bytes, || {
                Ok(runtime.run_prepared(&prepared, &[&x_t])?)
            })?;
            let regions = format!("{:?}", prepared.elementwise_region_summary());
            let counts = format!("{:?}", prepared.elementwise_region_execution_counts());
            (
                error,
                measurement,
                vec![
                    "run_prepared (runtime-internal session entry)",
                    "fused region execution",
                    "output_allocation",
                ],
                json!({"elementwise_region_summary": regions, "elementwise_region_execution_counts": counts}),
            )
        }
        "eager-shared" => {
            let runtime = eager_runtime(CpuBackend::new())?;
            let x = eager_const(
                &runtime,
                Tensor::from_vec_col_major(vec![rows, cols], data.clone())?,
            )?;
            let (error, measurement) =
                runtime.with_eager_session(|s| -> Result<(f64, Measurement)> {
                    let mut chain_once = || -> Result<EagerTensor> {
                        let mut y = s.tanh(&x)?;
                        for _ in 1..chain {
                            y = s.tanh(&y)?;
                        }
                        Ok(y)
                    };
                    let y = chain_once()?;
                    let error = check(y.to_tensor()?.as_slice::<f32>()?)?;
                    let measurement = measure(timing, bytes, chain_once)?;
                    Ok((error, measurement))
                })??;
            (
                error,
                measurement,
                vec![
                    "chain of eager tanh ops",
                    "intermediate and output allocation",
                ],
                json!({}),
            )
        }
        _ => return Err(format!("unknown tanh chain arm {arm}").into()),
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer,
        outside_timer: vec![
            "backend/runtime construction",
            "trace and compile",
            "prepare_compiled",
            "input_construction",
            "eager session entry (eager arm)",
            "correctness_check",
        ],
        extra: outside,
    })
}

// ---------------------------------------------------------------------------
// #2006: composed layer_norm / rms_norm (today's ~10-op eager composition)
// ---------------------------------------------------------------------------

fn composed_norm(params: &Value, timing: &Timing) -> Result<CaseOutcome> {
    let (d, len, batch) = (
        param_usize(params, "d")?,
        param_usize(params, "len")?,
        param_usize(params, "batch")?,
    );
    let norm = param_str(params, "norm")?;
    let layer = match norm {
        "layer_norm" => true,
        "rms_norm" => false,
        _ => return Err(format!("unknown norm {norm}").into()),
    };
    let shape = [d, len, batch];
    let n = d * len * batch;
    let xd = values_f32(n, 30);
    let wd = values_f32(d, 31)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let bd = values_f32(d, 32);
    let eps = 1e-5f32;
    let runtime = eager_runtime(CpuBackend::new())?;
    let x = eager_const(
        &runtime,
        Tensor::from_vec_col_major(shape.to_vec(), xd.clone())?,
    )?;
    let weight = eager_const(&runtime, Tensor::from_vec_col_major(vec![d], wd.clone())?)?;
    let bias = eager_const(&runtime, Tensor::from_vec_col_major(vec![d], bd.clone())?)?;
    let eps_t = eager_const(
        &runtime,
        Tensor::from_vec_col_major(vec![len, batch], vec![eps; len * batch])?,
    )?;
    let inv_d = 1.0 / d as f64;
    let (error, measurement, ops) =
        runtime.with_eager_session(|s| -> Result<(f64, Measurement, usize)> {
            let mut forward = || -> Result<EagerTensor> {
                Ok(if layer {
                    let sum = s.reduce_sum(&x, Some(&[0]))?;
                    let mean = s.scale_real(&sum, inv_d)?;
                    let mean = s.broadcast_in_dim(&mean, &shape, &[1, 2])?;
                    let centered = s.sub(&x, &mean)?;
                    let sq = s.reduce_sum_squares(&centered, &[0])?;
                    let var = s.scale_real(&sq, inv_d)?;
                    let var = s.add(&var, &eps_t)?;
                    let inv = s.rsqrt(&var)?;
                    let inv = s.broadcast_in_dim(&inv, &shape, &[1, 2])?;
                    let normed = s.mul(&centered, &inv)?;
                    let w = s.broadcast_in_dim(&weight, &shape, &[0])?;
                    let scaled = s.mul(&normed, &w)?;
                    let b = s.broadcast_in_dim(&bias, &shape, &[0])?;
                    s.add(&scaled, &b)?
                } else {
                    let sq = s.reduce_sum_squares(&x, &[0])?;
                    let ms = s.scale_real(&sq, inv_d)?;
                    let ms = s.add(&ms, &eps_t)?;
                    let inv = s.rsqrt(&ms)?;
                    let inv = s.broadcast_in_dim(&inv, &shape, &[1, 2])?;
                    let normed = s.mul(&x, &inv)?;
                    let w = s.broadcast_in_dim(&weight, &shape, &[0])?;
                    s.mul(&normed, &w)?
                })
            };
            let y = forward()?;
            let ys = y.to_tensor()?;
            let ys = ys.as_slice::<f32>()?;
            let mut worst = 0.0f64;
            for col in 0..len * batch {
                let xs: Vec<f64> = (0..d).map(|i| xd[i + col * d] as f64).collect();
                let mean = if layer {
                    xs.iter().sum::<f64>() / d as f64
                } else {
                    0.0
                };
                let var = xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / d as f64;
                let inv = 1.0 / (var + eps as f64).sqrt();
                for i in 0..d {
                    let mut want = (xs[i] - mean) * inv * wd[i] as f64;
                    if layer {
                        want += bd[i] as f64;
                    }
                    worst = worst.max(rel_error(ys[i + col * d] as f64, want));
                }
            }
            let error = check_tol(worst, 1e-4, norm)?;
            let measurement = measure(timing, n * 4 * 4, forward)?;
            Ok((error, measurement, if layer { 13 } else { 8 }))
        })??;
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer: vec!["eager composition (reduce_sum, scale_real, broadcast_in_dim, sub/add/mul, reduce_sum_squares, rsqrt)", "intermediate and output allocation"],
        outside_timer: vec!["eager_runtime_construction", "input/weight/bias/eps construction", "eager session entry", "correctness_check"],
        extra: json!({"eager_ops_per_norm": ops}),
    })
}

// ---------------------------------------------------------------------------
// #1885 §4.1 addendum: one-shot small contraction, default pool vs 1 thread
// ---------------------------------------------------------------------------

fn small_contraction_pool_hop(params: &Value, timing: &Timing) -> Result<CaseOutcome> {
    let e = param_usize(params, "extent")?;
    let arm = param_str(params, "arm")?;
    // abcd,dbef->acef: b and d are contracted (lhs axes 1,3; rhs axes 1,0).
    let cfg = matmul_config(&[1, 3], &[1, 0]);
    let n = e * e * e * e;
    let (ad, bd) = (values(n, 40), values(n, 41));
    let a = Tensor::from_vec_col_major(vec![e; 4], ad.clone())?;
    let b = Tensor::from_vec_col_major(vec![e; 4], bd.clone())?;
    let idx = |i: [usize; 4]| i[0] + e * (i[1] + e * (i[2] + e * i[3]));
    let check = |out: &Tensor| -> Result<f64> {
        let o = out.as_slice::<f64>()?;
        let mut worst = 0.0f64;
        for av in 0..e {
            for c in 0..e {
                for ev in 0..e {
                    for f in 0..e {
                        let mut want = 0.0;
                        for bv in 0..e {
                            for dv in 0..e {
                                want += ad[idx([av, bv, c, dv])] * bd[idx([dv, bv, ev, f])];
                            }
                        }
                        worst = worst.max(rel_error(o[idx([av, c, ev, f])], want));
                    }
                }
            }
        }
        check_tol(worst, 1e-12, "abcd,dbef->acef")
    };
    let one = |s: &mut dyn BackendSession| -> Result<Tensor> {
        Ok(s.dot_general_read(
            TensorRead::from_tensor(&a),
            TensorRead::from_tensor(&b),
            &cfg,
        )?)
    };
    let bytes = n * 8;
    let (error, measurement, timer, construction) = match arm {
        "per-call-default-pool" | "per-call-threads1" => {
            let mut backend = if arm == "per-call-default-pool" {
                CpuBackend::new()
            } else {
                CpuBackend::with_threads(1)?
            };
            let error = check(&backend.with_backend_session(|s| one(s))??)?;
            let measurement = measure(timing, bytes, || {
                backend.with_backend_session(|s| one(s))?
            })?;
            (
                error,
                measurement,
                vec![
                    "with_backend_session entry/exit per call",
                    "dot_general_read",
                    "output_allocation",
                ],
                if arm == "per-call-default-pool" {
                    "CpuBackend::new()"
                } else {
                    "CpuBackend::with_threads(1)"
                },
            )
        }
        "shared-session-default-pool" => {
            let mut backend = CpuBackend::new();
            let (error, measurement) =
                backend.with_backend_session(|s| -> Result<(f64, Measurement)> {
                    let error = check(&one(s)?)?;
                    Ok((error, measure(timing, bytes, || one(s))?))
                })??;
            (
                error,
                measurement,
                vec!["dot_general_read", "output_allocation"],
                "CpuBackend::new()",
            )
        }
        _ => return Err(format!("unknown pool-hop arm {arm}").into()),
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer,
        outside_timer: vec![
            "backend_construction",
            "input_construction",
            "correctness_check",
        ],
        extra: json!({"backend_construction": construction}),
    })
}

// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let case_id = arg("--case").ok_or("--case is required")?;
    let params: Value = serde_json::from_str(&arg("--params").ok_or("--params is required")?)?;
    let threads = arg_usize("--threads", 1)?;
    let correctness_only = arg("--mode").as_deref() == Some("correctness-only");
    let timing = Timing {
        warmups: arg_usize("--warmups", 1)?,
        samples: if correctness_only {
            0
        } else {
            arg_usize("--samples", 3)?
        },
        target_ns: arg_usize("--target-ns", 1_000_000)? as u128,
    };
    let kind = param_str(&params, "kind")?;
    let outcome = match kind {
        "decode_projection" => decode_projection(&params, &timing, threads),
        "copy_volume" => copy_volume(&params),
        "gemm_mm256" => gemm_mm256(&params, &timing),
        "rank1_dot" => rank1_dot(&params, &timing),
        "small_solves" => small_solves(&params, &timing),
        "backward_live_leaves" => backward_live_leaves(&params, &timing),
        "tanh_chain" => tanh_chain(&params, &timing),
        "composed_norm" => composed_norm(&params, &timing),
        "small_contraction_pool_hop" => small_contraction_pool_hop(&params, &timing),
        _ => Err(format!("unknown case kind {kind}").into()),
    }?;
    let mut output = json!({
        "suite_id": "cpu/perf_issues",
        "case_id": case_id,
        "correctness_status": "passed",
        "max_rel_error": outcome.max_rel_error,
        "samples": Vec::<Sample>::new(),
        "scope": {"timer": outcome.timer, "outside_timer": outcome.outside_timer},
        "threads_requested": threads,
    });
    if let Value::Object(fields) = outcome.extra {
        for (key, value) in fields {
            output[key] = value;
        }
    }
    if let Some(m) = outcome.measurement {
        output["samples"] = serde_json::to_value(&m.samples)?;
        output["calibration"] = json!({"target_ns": timing.target_ns, "iterations": m.iterations, "elapsed_ns": m.calibrated_ns});
    }
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
