//! Executable case producer for gpu/perf_issues (CUDA).
//!
//! Each case reproduces the workload of an open tenferro-rs GPU performance
//! issue (the issue number is recorded next to the case in
//! `data/instances/gpu_perf_issues.json`), with its reference arm in the same
//! family. Reference arms that are not tenferro use cudarc directly on the same
//! device. Setup (backends, uploads of inputs, host buffers, warm calls) is
//! outside the timer; a timed batch ends with a device synchronization (never a
//! result download) unless the download is the declared operation. The
//! first-call layout case is an explicitly labelled first-call diagnostic.
//! Emits one JSON object on stdout.

use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use cudarc::driver::CudaContext;
use num_complex::Complex64;
use serde::Serialize;
use serde_json::{json, Value};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_gpu::cuda::cudarc::cublas::{result as cublas, sys as cublas_sys};
use tenferro_gpu::cuda::{
    download_tensor, upload_tensor, with_cuda_exec_session, CudaBackend, CudaDeviceId,
};
use tenferro_linalg::{EighDriver, EighOptions, SvdDriver, SvdOptions, TensorLinalgExt};
use tenferro_runtime::TensorSessionOpsExt;
use tenferro_tensor::{
    BackendSession, BackendSessionHost, DotGeneralConfig, GatherConfig, Tensor, TensorRead,
    TensorView, TensorViewCanonicalization, TensorWrite,
};

type BoxError = Box<dyn Error + Send + Sync>;
type Result<T> = std::result::Result<T, BoxError>;

const CALIBRATION_MAX_ITERATIONS: usize = 1 << 16;
const CALIBRATION_DEADLINE_NS: u128 = 30_000_000_000;
const RETENTION_BUDGET_BYTES: usize = 1 << 30;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn arg_usize(name: &str, default: usize) -> Result<usize> {
    Ok(arg(name).map(|v| v.parse()).transpose()?.unwrap_or(default))
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
    samples: Vec<Sample>,
}

/// `batch(iterations)` runs and times one batch itself (so a case can put a
/// device synchronization or a callback boundary at the batch edge) and
/// returns the elapsed nanoseconds.
fn measure(
    timing: &Timing,
    retained_bytes: usize,
    mut batch: impl FnMut(usize) -> Result<u128>,
) -> Result<Measurement> {
    for _ in 0..timing.warmups.max(1) {
        batch(1)?;
    }
    let cap = (RETENTION_BUDGET_BYTES / retained_bytes.max(1)).clamp(1, CALIBRATION_MAX_ITERATIONS);
    let start = Instant::now();
    let mut iterations = 1usize;
    loop {
        let elapsed = batch(iterations)?;
        if elapsed >= timing.target_ns || iterations >= cap {
            break;
        }
        if start.elapsed().as_nanos() >= CALIBRATION_DEADLINE_NS {
            return Err("calibration deadline exceeded".into());
        }
        iterations = (iterations * 2).min(cap);
    }
    let mut samples = Vec::with_capacity(timing.samples);
    for sample_index in 0..timing.samples {
        let elapsed_ns = batch(iterations)?;
        samples.push(Sample {
            sample_index,
            elapsed_ns,
            iterations,
        });
    }
    Ok(Measurement {
        iterations,
        samples,
    })
}

/// Time `iterations` calls of `op`, retaining outputs until after `sync`.
fn timed<O>(
    iterations: usize,
    op: &mut impl FnMut() -> Result<O>,
    sync: &mut impl FnMut() -> Result<()>,
) -> Result<u128> {
    let mut outputs = Vec::with_capacity(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        outputs.push(op()?);
    }
    sync()?;
    let elapsed = start.elapsed().as_nanos();
    black_box(&outputs);
    Ok(elapsed)
}

/// Time `iterations` session operations, then synchronize the device.
fn timed_in_session<O>(
    iterations: usize,
    s: &mut dyn BackendSession,
    mut op: impl FnMut(&mut dyn BackendSession) -> Result<O>,
) -> Result<u128> {
    let mut outputs = Vec::with_capacity(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        outputs.push(op(s)?);
    }
    device_sync(s)?;
    let elapsed = start.elapsed().as_nanos();
    black_box(&outputs);
    Ok(elapsed)
}

/// One isolated call per sample.
///
/// For the issue workloads whose precondition cannot be batched (a plan-cache
/// churn, a destination handle built per call) a single wall-clock interval
/// over many operations is not available, so the sample *is* one call and the
/// reported median is a per-call latency, not a per-operation average.
fn measure_isolated(
    timing: &Timing,
    mut op: impl FnMut() -> Result<u128>,
) -> Result<Measurement> {
    for _ in 0..timing.warmups.max(1) {
        black_box(op()?);
    }
    op()?;
    let mut samples = Vec::with_capacity(timing.samples);
    for sample_index in 0..timing.samples {
        let elapsed_ns = op()?;
        if elapsed_ns == 0 {
            return Err("timed sample elapsed zero nanoseconds".into());
        }
        samples.push(Sample {
            sample_index,
            elapsed_ns,
            iterations: 1,
        });
    }
    Ok(Measurement {
        iterations: 1,
        samples,
    })
}

struct CaseOutcome {
    max_rel_error: f64,
    measurement: Option<Measurement>,
    timer: Vec<&'static str>,
    outside_timer: Vec<&'static str>,
    extra: Value,
}

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

fn host(shape: Vec<usize>, data: Vec<f64>) -> Result<Tensor> {
    Ok(Tensor::from_vec_col_major(shape, data)?)
}

fn host_f32(shape: Vec<usize>, data: Vec<f32>) -> Result<Tensor> {
    Ok(Tensor::from_vec_col_major(shape, data)?)
}

fn host_c64(shape: Vec<usize>, data: Vec<Complex64>) -> Result<Tensor> {
    Ok(Tensor::from_vec_col_major(shape, data)?)
}

fn matmul_cfg(lhs_contract: &[usize], rhs_contract: &[usize]) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: lhs_contract.into(),
        rhs_contracting_dims: rhs_contract.into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    }
}

/// Cheap deterministic f32 fill; the `values` generator allocates an f64
/// buffer the size of the tensor, which is too much for the 1 GiB tables.
fn fill_f32(n: usize, seed: u64) -> Vec<f32> {
    (0..n as u64)
        .map(|i| (((i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97)) % 2001) as f32) / 1000.0 - 1.0)
        .collect()
}

fn complex_values(n: usize, seed: u64) -> Vec<Complex64> {
    let re = values(n, seed);
    let im = values(n, seed + 1);
    re.into_iter()
        .zip(im)
        .map(|(r, i)| Complex64::new(r, i))
        .collect()
}

/// Resident set size of this process, for the #1924 vendor-churn observable.
///
/// The documented leak is host memory (RSS grew 410 -> 3114 MB over 50
/// create/drop cycles), so device free bytes alone cannot see it. Linux only,
/// which is where the CUDA suite runs; an unreadable /proc reports zero.
fn process_rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|statm| {
            statm
                .split_whitespace()
                .nth(1)
                .and_then(|pages| pages.parse::<u64>().ok())
        })
        .map(|resident_pages| resident_pages * 4096)
        .unwrap_or(0)
}

/// Driver-reported free device bytes, for the state diagnostics (#2081, #1924).
///
/// `mem_get_info` is a driver call, so the driver must be initialized: the
/// runtime's own sessions normally do that, but the state cases must not depend
/// on having entered one first.
fn device_free_bytes(ctx: &CudaContext) -> Result<u64> {
    cudarc::driver::result::init()?;
    ctx.bind_to_thread()?;
    let (free, _total) = cudarc::driver::result::mem_get_info()?;
    Ok(free as u64)
}

fn backend(device: usize) -> Result<CudaBackend> {
    Ok(CudaBackend::new(CudaDeviceId::from_ordinal(
        u32::try_from(device)?,
    ))?)
}

fn device_sync(session: &mut dyn BackendSession) -> Result<()> {
    with_cuda_exec_session(session, |cs| cs.synchronize())
        .ok_or("not a CUDA execution session")??;
    Ok(())
}

/// Fold one error term into a running worst case.
///
/// `f64::max` returns the non-NaN operand when only one side is NaN, so a plain
/// `max` fold would silently discard a NaN result and leave a passing zero.
/// A non-finite term is reported as infinite instead.
fn worst_of(worst: f64, value: f64) -> f64 {
    if value.is_finite() {
        if value > worst {
            value
        } else {
            worst
        }
    } else {
        f64::INFINITY
    }
}

fn max_rel(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1.0))
        .fold(0.0, worst_of)
}

fn check_tol(error: f64, tol: f64, what: &str) -> Result<f64> {
    if error.is_finite() && error <= tol {
        Ok(error)
    } else {
        Err(format!("{what}: relative error {error:e} exceeds {tol:e}").into())
    }
}

// ---------------------------------------------------------------------------
// #2009: host<->device transfer size sweep vs cudarc
// ---------------------------------------------------------------------------

fn transfer(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let bytes = param_usize(params, "bytes")?;
    let up = param_str(params, "direction")? == "up";
    let arm = param_str(params, "arm")?;
    let n = (bytes / 8).max(1);
    let data = values(n, 1);
    let (error, measurement, timer) = match arm {
        "tenferro" => {
            let bk = backend(device)?;
            let rt = bk.runtime().clone();
            let host_t = host(vec![n], data.clone())?;
            let resident = upload_tensor(&rt, &host_t)?;
            rt.synchronize()?;
            let back = download_tensor(&rt, &resident)?;
            let error = check_tol(max_rel(back.as_slice::<f64>()?, &data), 0.0, "round trip")?;
            let m = if up {
                measure(timing, bytes, |it| {
                    timed(
                        it,
                        &mut || {
                            let t = upload_tensor(&rt, &host_t)?;
                            rt.synchronize()?;
                            Ok(t)
                        },
                        &mut || Ok(()),
                    )
                })?
            } else {
                measure(timing, bytes, |it| {
                    timed(
                        it,
                        &mut || Ok(download_tensor(&rt, &resident)?),
                        &mut || Ok(()),
                    )
                })?
            };
            (
                error,
                m,
                if up {
                    vec!["upload_tensor", "runtime.synchronize per call"]
                } else {
                    vec![
                        "download_tensor (synchronizes itself)",
                        "host output allocation",
                    ]
                },
            )
        }
        "cudarc-pageable" | "cudarc-pinned" => {
            let ctx = CudaContext::new(device)?;
            let stream = ctx.new_stream()?;
            let mut dev_buf = unsafe { stream.alloc::<f64>(n) }?;
            let pinned_arm = arm == "cudarc-pinned";
            let mut pageable = data.clone();
            let mut pinned = unsafe { ctx.alloc_pinned::<f64>(n) }?;
            pinned.as_mut_slice()?.copy_from_slice(&data);
            // Round trip check through the same path.
            if pinned_arm {
                stream.memcpy_htod(&pinned, &mut dev_buf)?;
                let mut back = unsafe { ctx.alloc_pinned::<f64>(n) }?;
                stream.memcpy_dtoh(&dev_buf, &mut back)?;
                stream.synchronize()?;
                check_tol(
                    max_rel(back.as_slice()?, &data),
                    0.0,
                    "cudarc pinned round trip",
                )?;
            } else {
                stream.memcpy_htod(&pageable, &mut dev_buf)?;
                let mut back = vec![0.0; n];
                stream.memcpy_dtoh(&dev_buf, &mut back)?;
                stream.synchronize()?;
                check_tol(max_rel(&back, &data), 0.0, "cudarc pageable round trip")?;
            }
            let m = measure(timing, 0, |it| {
                timed(
                    it,
                    &mut || -> Result<()> {
                        match (up, pinned_arm) {
                            (true, true) => stream.memcpy_htod(&pinned, &mut dev_buf)?,
                            (true, false) => stream.memcpy_htod(&pageable, &mut dev_buf)?,
                            (false, true) => stream.memcpy_dtoh(&dev_buf, &mut pinned)?,
                            (false, false) => stream.memcpy_dtoh(&dev_buf, &mut pageable)?,
                        }
                        stream.synchronize()?;
                        Ok(())
                    },
                    &mut || Ok(()),
                )
            })?;
            (
                0.0,
                m,
                vec![
                    "cudarc memcpy into a preallocated buffer",
                    "stream.synchronize per call",
                ],
            )
        }
        _ => return Err(format!("unknown transfer arm {arm}").into()),
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer,
        outside_timer: vec![
            "backend/context construction",
            "host and device buffer allocation",
            "round-trip check",
        ],
        extra: json!({"bytes_per_op": n * 8}),
    })
}

// ---------------------------------------------------------------------------
// #1887: alloc_zero_output within / across with_cubecl callbacks
// ---------------------------------------------------------------------------

fn alloc_zero(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let elements = param_usize(params, "elements")?;
    let arm = param_str(params, "arm")?;
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    let shape = [elements];
    let bytes = elements * 8;
    // Check: one zero output per arm route, downloaded outside timing.
    let zeros = host(vec![elements], vec![0.0; elements])?;
    let check = |t: &Tensor| -> Result<f64> {
        let back = download_tensor(&rt, t)?;
        if back.as_slice::<f64>()?.iter().any(|v| v.to_bits() != 0) {
            return Err("zero output is not +0.0".into());
        }
        Ok(0.0)
    };
    let alloc_zero_once = |s: &mut dyn BackendSession| -> Result<Tensor> {
        let typed = with_cuda_exec_session(s, |cs| {
            cs.with_cubecl("perf_issues.alloc_zero", |cub| {
                cub.alloc_zero_output::<f64>(&shape)
            })
        })
        .ok_or("not a CUDA execution session")??;
        Ok(Tensor::from(typed))
    };
    let alloc_fill_once = |s: &mut dyn BackendSession| -> Result<Tensor> {
        Ok(with_cuda_exec_session(s, |cs| {
            cs.with_cubecl("perf_issues.alloc_fill", |cub| {
                let mut out = Tensor::from(cub.alloc_output::<f64>(&shape)?);
                cub.fill_zero_write(TensorWrite::from_tensor(&mut out))?;
                Ok(out)
            })
        })
        .ok_or("not a CUDA execution session")??)
    };
    let first = match arm {
        "alloc-zero-within-callback" | "alloc-zero-per-callback" => {
            bk.with_backend_session(|s| alloc_zero_once(s))??
        }
        "alloc-fill-per-callback" => bk.with_backend_session(|s| alloc_fill_once(s))??,
        "host-zero-upload" => upload_tensor(&rt, &zeros)?,
        _ => return Err(format!("unknown alloc-zero arm {arm}").into()),
    };
    let error = check(&first)?;
    let (measurement, timer): (Measurement, Vec<&str>) = match arm {
        "alloc-zero-within-callback" => {
            // One with_cubecl callback per batch; the batch's allocations share it.
            let m = bk.with_backend_session(|s| {
                measure(timing, bytes, |it| {
                    let start = Instant::now();
                    let outputs = with_cuda_exec_session(s, |cs| {
                        cs.with_cubecl("perf_issues.alloc_zero_batch", |cub| {
                            (0..it)
                                .map(|_| cub.alloc_zero_output::<f64>(&shape))
                                .collect::<std::result::Result<Vec<_>, _>>()
                        })
                    })
                    .ok_or("not a CUDA execution session")??;
                    device_sync(s)?;
                    let elapsed = start.elapsed().as_nanos();
                    black_box(&outputs);
                    Ok(elapsed)
                })
            })??;
            (
                m,
                vec![
                    "one with_cubecl entry/exit per batch",
                    "alloc_zero_output x iterations",
                    "device synchronize at batch end",
                ],
            )
        }
        "alloc-zero-per-callback" | "alloc-fill-per-callback" => {
            let fill = arm == "alloc-fill-per-callback";
            let m = bk.with_backend_session(|s| {
                measure(timing, bytes, |it| {
                    let mut outputs = Vec::with_capacity(it);
                    let start = Instant::now();
                    for _ in 0..it {
                        outputs.push(if fill {
                            alloc_fill_once(s)?
                        } else {
                            alloc_zero_once(s)?
                        });
                    }
                    device_sync(s)?;
                    let elapsed = start.elapsed().as_nanos();
                    black_box(&outputs);
                    Ok(elapsed)
                })
            })??;
            (
                m,
                if fill {
                    vec![
                        "with_cubecl entry/exit per allocation",
                        "alloc_output + fill_zero_write (device fill)",
                        "device synchronize at batch end",
                    ]
                } else {
                    vec![
                        "with_cubecl entry/exit per allocation",
                        "alloc_zero_output",
                        "device synchronize at batch end",
                    ]
                },
            )
        }
        "host-zero-upload" => {
            let m = measure(timing, bytes, |it| {
                timed(it, &mut || Ok(upload_tensor(&rt, &zeros)?), &mut || {
                    Ok(rt.synchronize()?)
                })
            })?;
            (
                m,
                vec![
                    "upload_tensor of a host zero buffer",
                    "runtime synchronize at batch end",
                ],
            )
        }
        _ => unreachable!(),
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer,
        outside_timer: vec![
            "backend construction",
            "backend session entry/exit",
            "host zero buffer construction",
            "zero check (download)",
        ],
        extra: json!({"bytes_per_op": bytes}),
    })
}

// ---------------------------------------------------------------------------
// #1885 §3.1/§3.2: many small independent blocks vs one batched call
// ---------------------------------------------------------------------------

fn small_blocks(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let (n, count) = (param_usize(params, "n")?, param_usize(params, "count")?);
    let arm = param_str(params, "arm")?;
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    let (ad, bd) = (values(n * n * count, 3), values(n * n * count, 4));
    let block = |data: &[f64], i: usize| data[i * n * n..(i + 1) * n * n].to_vec();
    let reference = |i: usize| -> Vec<f64> {
        let (a, b) = (block(&ad, i), block(&bd, i));
        let mut c = vec![0.0; n * n];
        for j in 0..n {
            for k in 0..n {
                for r in 0..n {
                    c[r + j * n] += a[r + k * n] * b[k + j * n];
                }
            }
        }
        c
    };
    let gemm = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let batched = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [2].as_slice().into(),
        rhs_batch_dims: [2].as_slice().into(),
    };
    let bytes = n * n * count * 8;
    let (error, measurement, timer) = if arm == "loop" {
        let pairs: Vec<(Tensor, Tensor)> = (0..count)
            .map(|i| {
                Ok((
                    upload_tensor(&rt, &host(vec![n, n], block(&ad, i))?)?,
                    upload_tensor(&rt, &host(vec![n, n], block(&bd, i))?)?,
                ))
            })
            .collect::<Result<_>>()?;
        bk.with_backend_session(|s| -> Result<(f64, Measurement, Vec<&str>)> {
            let mut worst = 0.0f64;
            for i in [0, count / 2, count - 1] {
                let c = s.dot_general_read(
                    TensorRead::from_tensor(&pairs[i].0),
                    TensorRead::from_tensor(&pairs[i].1),
                    &gemm,
                )?;
                worst = worst.max(max_rel(
                    download_tensor(&rt, &c)?.as_slice::<f64>()?,
                    &reference(i),
                ));
            }
            let error = check_tol(worst, 1e-10, "small block gemm")?;
            let m = measure(timing, bytes, |it| {
                let mut outputs = Vec::with_capacity(it * count);
                let start = Instant::now();
                for _ in 0..it {
                    for (a, b) in &pairs {
                        outputs.push(s.dot_general_read(
                            TensorRead::from_tensor(a),
                            TensorRead::from_tensor(b),
                            &gemm,
                        )?);
                    }
                }
                device_sync(s)?;
                let elapsed = start.elapsed().as_nanos();
                black_box(&outputs);
                Ok(elapsed)
            })?;
            Ok((
                error,
                m,
                vec![
                    "count x dot_general_read (one submission per block)",
                    "device synchronize at batch end",
                ],
            ))
        })??
    } else {
        let a = upload_tensor(&rt, &host(vec![n, n, count], ad.clone())?)?;
        let b = upload_tensor(&rt, &host(vec![n, n, count], bd.clone())?)?;
        bk.with_backend_session(|s| -> Result<(f64, Measurement, Vec<&str>)> {
            let c = s.dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&b),
                &batched,
            )?;
            let cs = download_tensor(&rt, &c)?;
            let cs = cs.as_slice::<f64>()?;
            let mut worst = 0.0f64;
            for i in [0, count / 2, count - 1] {
                worst = worst.max(max_rel(&cs[i * n * n..(i + 1) * n * n], &reference(i)));
            }
            let error = check_tol(worst, 1e-10, "batched block gemm")?;
            let m = measure(timing, bytes, |it| {
                timed_in_session(it, s, |s| {
                    Ok(s.dot_general_read(
                        TensorRead::from_tensor(&a),
                        TensorRead::from_tensor(&b),
                        &batched,
                    )?)
                })
            })?;
            Ok((
                error,
                m,
                vec![
                    "one batched dot_general_read ([n, n, count])",
                    "device synchronize at batch end",
                ],
            ))
        })??
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer,
        outside_timer: vec![
            "backend construction",
            "uploads",
            "session entry/exit",
            "correctness check (download)",
        ],
        extra: json!({"blocks_per_op": count}),
    })
}

// ---------------------------------------------------------------------------
// #1885 §3.3: per-matrix QR / SVD loop vs the batched entry; §3.3 addendum:
// QR host cost vs the number of live device buffers
// ---------------------------------------------------------------------------

fn host_matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for j in 0..n {
        for p in 0..k {
            for i in 0..m {
                c[i + j * m] += a[i + p * m] * b[p + j * k];
            }
        }
    }
    c
}

/// Reconstruction error of a factorization of one `n x n` item.
fn reconstruct(op: &str, outs: &[Vec<f64>], a: &[f64], n: usize) -> f64 {
    let rebuilt = if op == "qr" {
        host_matmul(&outs[0], &outs[1], n, n, n)
    } else {
        let mut us = outs[0].clone();
        for j in 0..n {
            for i in 0..n {
                us[i + j * n] *= outs[1][j];
            }
        }
        host_matmul(&us, &outs[2], n, n, n)
    };
    max_rel(&rebuilt, a)
}

fn well_conditioned(n: usize, seed: u64) -> Vec<f64> {
    let mut a = values(n * n, seed);
    for i in 0..n {
        a[i + i * n] += n as f64;
    }
    a
}

fn factorize(op: &str, a: &Tensor, s: &mut dyn BackendSession) -> Result<Vec<Tensor>> {
    Ok(if op == "qr" {
        let (q, r) = a.qr(s)?;
        vec![q, r]
    } else {
        let (u, sv, vt) = a.svd(s)?;
        vec![u, sv, vt]
    })
}

fn batched_factorization(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let (n, batch) = (param_usize(params, "n")?, param_usize(params, "batch")?);
    let op = param_str(params, "op")?.to_string();
    let arm = param_str(params, "arm")?;
    let live = params["live_buffers"].as_u64().map(|v| v as usize);
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    let items: Vec<Vec<f64>> = (0..batch)
        .map(|i| well_conditioned(n, 50 + i as u64))
        .collect();
    let bytes = n * n * batch * 8 * 3;
    let item_outs = |outs: &[Tensor], i: usize, batched: bool| -> Result<Vec<Vec<f64>>> {
        outs.iter()
            .map(|t| {
                let h = download_tensor(&rt, t)?;
                let all = h.as_slice::<f64>()?;
                let len = if batched {
                    all.len() / batch
                } else {
                    all.len()
                };
                let off = if batched { i * len } else { 0 };
                Ok(all[off..off + len].to_vec())
            })
            .collect()
    };
    let (error, measurement, timer) = if arm == "loop" {
        let tensors: Vec<Tensor> = items
            .iter()
            .map(|a| upload_tensor(&rt, &host(vec![n, n], a.clone())?).map_err(Into::into))
            .collect::<Result<_>>()?;
        // Extra resident buffers alive for the whole case (§3.3 addendum).
        let resident: Vec<Tensor> = (0..live.unwrap_or(0))
            .map(|i| {
                upload_tensor(&rt, &host(vec![n, n], values(n * n, 900 + i as u64))?)
                    .map_err(Into::into)
            })
            .collect::<Result<_>>()?;
        let r = bk.with_backend_session(|s| -> Result<(f64, Measurement, Vec<&str>)> {
            let mut worst = 0.0f64;
            for i in [0, batch - 1] {
                let outs = factorize(&op, &tensors[i], s)?;
                worst = worst.max(reconstruct(&op, &item_outs(&outs, i, false)?, &items[i], n));
            }
            let error = check_tol(worst, 1e-8, "factorization reconstruction")?;
            let m = measure(timing, bytes, |it| {
                let mut outputs = Vec::with_capacity(it * batch);
                let start = Instant::now();
                for _ in 0..it {
                    for a in &tensors {
                        outputs.push(factorize(&op, a, s)?);
                    }
                }
                device_sync(s)?;
                let elapsed = start.elapsed().as_nanos();
                black_box(&outputs);
                Ok(elapsed)
            })?;
            Ok((
                error,
                m,
                vec![
                    "batch x rank-2 factorization (per-matrix submissions and info checks)",
                    "device synchronize at batch end",
                ],
            ))
        })??;
        black_box(&resident);
        r
    } else {
        let all: Vec<f64> = items.concat();
        let a = upload_tensor(&rt, &host(vec![n, n, batch], all)?)?;
        bk.with_backend_session(|s| -> Result<(f64, Measurement, Vec<&str>)> {
            let outs = factorize(&op, &a, s)?;
            let mut worst = 0.0f64;
            for i in [0, batch - 1] {
                worst = worst.max(reconstruct(&op, &item_outs(&outs, i, true)?, &items[i], n));
            }
            let error = check_tol(worst, 1e-8, "batched factorization reconstruction")?;
            let m = measure(timing, bytes, |it| {
                timed_in_session(it, s, |s| factorize(&op, &a, s))
            })?;
            Ok((
                error,
                m,
                vec![
                    "one batched factorization call ([n, n, batch])",
                    "device synchronize at batch end",
                ],
            ))
        })??
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer,
        outside_timer: vec![
            "backend construction",
            "uploads (and live resident buffers)",
            "session entry/exit",
            "reconstruction check (download)",
        ],
        extra: json!({
            "matrices_per_op": batch,
            "live_device_buffers": live.map(|l| l + batch),
            "unavailable": ["blocking host syncs per block (needs nsys or a driver-API hook; no public counter)"],
        }),
    })
}

// ---------------------------------------------------------------------------
// #1885 §3.4 addendum: first call per distinct copy layout (diagnostic)
// ---------------------------------------------------------------------------

fn first_call_layouts(params: &Value, device: usize) -> Result<CaseOutcome> {
    let max_rows = param_usize(params, "max_rows")?;
    let copy_into = match param_str(params, "arm")? {
        "to-contiguous" => false,
        "copy-into" => true,
        arm => return Err(format!("unknown first-call arm {arm}").into()),
    };
    let cols: Vec<usize> = params["cols"]
        .as_array()
        .ok_or("cols missing")?
        .iter()
        .filter_map(|v| v.as_u64().map(|c| c as usize))
        .collect();
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    let mut layouts = vec![(13usize, 3usize)]; // warm-up layout, reported separately
    for r in 2..=max_rows {
        for &c in &cols {
            layouts.push((r, c));
        }
    }
    let mut rows = Vec::new();
    for (index, &(r, c)) in layouts.iter().enumerate() {
        let data = values(r * c, 70 + index as u64);
        let t = upload_tensor(&rt, &host(vec![r, c], data.clone())?)?;
        rt.synchronize()?;
        // A compact destination for the copy_into route, allocated untimed.
        let mut dst = upload_tensor(&rt, &host(vec![c, r], vec![0.0; r * c])?)?;
        rt.synchronize()?;
        let mut call = |s: &mut dyn BackendSession| -> Result<(u128, Option<Tensor>)> {
            let typed = t.as_typed::<f64>().ok_or("device tensor is not f64")?;
            let view = typed.as_view().transpose_view([1, 0])?;
            if copy_into {
                let dst_typed = dst.as_typed_mut::<f64>().ok_or("destination is not f64")?;
                let mut dst_view = dst_typed.as_view_mut();
                let start = Instant::now();
                with_cuda_exec_session(s, |cs| cs.copy_into(&view, &mut dst_view))
                    .ok_or("not a CUDA execution session")??;
                device_sync(s)?;
                return Ok((start.elapsed().as_nanos(), None));
            }
            let start = Instant::now();
            let out = with_cuda_exec_session(s, |cs| cs.to_contiguous(&view))
                .ok_or("not a CUDA execution session")??;
            device_sync(s)?;
            Ok((start.elapsed().as_nanos(), Some(Tensor::from(out))))
        };
        let (first, second, out) =
            bk.with_backend_session(|s| -> Result<(u128, u128, Option<Tensor>)> {
                let (first, _) = call(s)?;
                let (second, out) = call(s)?;
                Ok((first, second, out))
            })??;
        let back = download_tensor(&rt, out.as_ref().unwrap_or(&dst))?;
        let mut expected = vec![0.0; r * c];
        for i in 0..r {
            for j in 0..c {
                expected[j + i * c] = data[i + j * r];
            }
        }
        check_tol(
            max_rel(back.as_slice::<f64>()?, &expected),
            0.0,
            "transpose copy",
        )?;
        rows.push(json!({"layout": format!("[{r}, {c}] -> transpose"), "warm_up_layout": index == 0,
                         "first_call_ms": first as f64 / 1e6, "second_call_ms": second as f64 / 1e6}));
    }
    let firsts: Vec<f64> = rows
        .iter()
        .skip(1)
        .map(|r| r["first_call_ms"].as_f64().unwrap())
        .collect();
    let seconds: Vec<f64> = rows
        .iter()
        .skip(1)
        .map(|r| r["second_call_ms"].as_f64().unwrap())
        .collect();
    let median = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        s[s.len() / 2]
    };
    Ok(CaseOutcome {
        max_rel_error: 0.0,
        measurement: None,
        timer: vec!["to_contiguous (or copy_into a compact destination) of a transposed device view + device synchronize, first and second call per distinct layout"],
        outside_timer: vec!["backend construction", "upload", "warm-up layout", "download check"],
        extra: json!({
            "first_call_layouts": rows,
            "median_first_call_ms": median(&firsts),
            "median_second_call_ms": median(&seconds),
            "distinct_layouts": firsts.len(),
        }),
    })
}

// ---------------------------------------------------------------------------
// #2078: entering a raw CUDA session is a host barrier
// ---------------------------------------------------------------------------

fn raw_scope(s: &mut dyn BackendSession) -> Result<()> {
    with_cuda_exec_session(s, |cs| {
        cs.with_raw("perf_issues.raw_scope_entry", |_raw| Ok(()))
    })
    .ok_or("not a CUDA execution session")??;
    Ok(())
}

fn raw_scope_entry(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let n = param_usize(params, "gemm_n")?;
    let queued = param_usize(params, "queued_gemms")?;
    let arm = param_str(params, "arm")?;
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    let data = values(n * n, 13);
    let a = upload_tensor(&rt, &host(vec![n, n], data.clone())?)?;
    let cfg = matmul_cfg(&[1], &[0]);
    // Correctness: the queued GEMM really ran (one probe entry against a host sum).
    let probe = bk.with_backend_session(|s| -> Result<Tensor> {
        Ok(a.dot_general(&a, cfg.clone(), s)?)
    })??;
    rt.synchronize()?;
    let (i, j) = (1usize, 2usize);
    let expected: f64 = (0..n).map(|p| data[i + p * n] * data[p + j * n]).sum();
    let back = download_tensor(&rt, &probe)?;
    let got = back.as_slice::<f64>()?;
    let error = check_tol(max_rel(&[got[i + j * n]], &[expected]), 1e-12, "queued gemm")?;
    let retained_bytes = queued * n * n * 8;
    let (measurement, timer) = match arm {
        "empty-idle" => {
            let m = bk.with_backend_session(|s| {
                measure(timing, 0, |it| {
                    let start = Instant::now();
                    for _ in 0..it {
                        raw_scope(s)?;
                    }
                    Ok(start.elapsed().as_nanos())
                })
            })??;
            (
                m,
                vec!["with_cuda_exec_session + empty with_raw closure on an idle device"],
            )
        }
        "empty-behind-queued-gemm" => {
            let m = bk.with_backend_session(|s| {
                measure(timing, retained_bytes, |it| {
                    let mut elapsed = 0u128;
                    for _ in 0..it {
                        let mut retained = Vec::with_capacity(queued);
                        for _ in 0..queued {
                            retained.push(a.dot_general(&a, cfg.clone(), s)?);
                        }
                        // The first empty scope after the queue returns immediately (that is
                        // what the issue measured); the second one is the timed scope.
                        raw_scope(s)?;
                        let start = Instant::now();
                        raw_scope(s)?;
                        elapsed += start.elapsed().as_nanos();
                        black_box(&retained);
                    }
                    Ok(elapsed)
                })
            })??;
            (
                m,
                vec!["the second empty with_raw scope after queued device work"],
            )
        }
        "sync-behind-queued-gemm" => {
            let m = bk.with_backend_session(|s| {
                measure(timing, retained_bytes, |it| {
                    let mut elapsed = 0u128;
                    for _ in 0..it {
                        let mut retained = Vec::with_capacity(queued);
                        for _ in 0..queued {
                            retained.push(a.dot_general(&a, cfg.clone(), s)?);
                        }
                        let start = Instant::now();
                        device_sync(s)?;
                        elapsed += start.elapsed().as_nanos();
                        black_box(&retained);
                    }
                    Ok(elapsed)
                })
            })??;
            (
                m,
                vec!["explicit device synchronize behind queued device work (the reference barrier)"],
            )
        }
        _ => return Err(format!("unknown raw-scope arm {arm}").into()),
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer,
        outside_timer: vec![
            "backend and runtime construction",
            "operand upload",
            "queued GEMM submissions and the first empty scope (the precondition of the timed scope)",
            "correctness probe (download)",
        ],
        extra: json!({
            "gemm_n": n,
            "queued_gemms": queued,
            "note": "timed operation is the second empty raw scope after the queued GEMMs",
        }),
    })
}

// ---------------------------------------------------------------------------
// #2079: eager per-op host cost for decode-sized tensors
// ---------------------------------------------------------------------------

fn eager_op(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let op = param_str(params, "op")?.to_string();
    let arm = param_str(params, "arm")?.to_string();
    let bk = backend(device)?;
    let rt = bk.runtime().clone();
    let runtime = EagerRuntime::with_cuda_backend(bk)?;
    let session = |s: &mut tenferro_ad::EagerSession<'_>| -> Result<(EagerTensor, EagerTensor, u64)> {
        match op.as_str() {
            "add" => {
                let (hidden, tokens) =
                    (param_usize(params, "hidden")?, param_usize(params, "tokens")?);
                let x = fill_f32(hidden * tokens, 7);
                let y = fill_f32(hidden * tokens, 8);
                let xt = s.constant_from_host(host_f32(vec![hidden, tokens], x)?)?;
                let yt = s.constant_from_host(host_f32(vec![hidden, tokens], y)?)?;
                let bytes = (hidden * tokens * 4) as u64;
                Ok((xt, yt, bytes))
            }
            "dot_general" => {
                let (din, dout, len) = (
                    param_usize(params, "in")?,
                    param_usize(params, "out")?,
                    param_usize(params, "len")?,
                );
                let w = fill_f32(dout * din, 5);
                let x = fill_f32(din * len, 6);
                let wt = s.constant_from_host(host_f32(vec![dout, din], w)?)?;
                let xt = s.constant_from_host(host_f32(vec![din, len], x)?)?;
                let bytes = (dout * len * 4) as u64;
                Ok((wt, xt, bytes))
            }
            other => Err(format!("unknown #2079 op {other}").into()),
        }
    };
    let (lhs, rhs, bytes) = runtime.with_eager_session(session)?;
    let run = |s: &mut tenferro_ad::EagerSession<'_>| -> Result<EagerTensor> {
        match op.as_str() {
            "add" => Ok(s.add(&lhs, &rhs)?),
            "dot_general" => Ok(s.dot_general(&lhs, &rhs, matmul_cfg(&[1], &[0]))?),
            other => Err(format!("unknown #2079 op {other}").into()),
        }
    };
    // Correctness: one result, downloaded outside timing and checked against a host reduction.
    let first = runtime.with_eager_session(run)?.to_tensor()?;
    rt.synchronize()?;
    let back = download_tensor(&rt, &first)?;
    let got = back.as_slice::<f32>()?;
    let error = match op.as_str() {
        "add" => {
            let len = param_usize(params, "hidden")? * param_usize(params, "tokens")?;
            let x = fill_f32(len, 7);
            let y = fill_f32(len, 8);
            let expected: Vec<f64> = x
                .iter()
                .zip(&y)
                .take(64)
                .map(|(a, b)| *a as f64 + *b as f64)
                .collect();
            check_tol(
                max_rel(
                    &got.iter().take(64).map(|v| *v as f64).collect::<Vec<_>>(),
                    &expected,
                ),
                1e-6,
                "decode add",
            )?
        }
        "dot_general" => {
            let (din, dout, len) = (
                param_usize(params, "in")?,
                param_usize(params, "out")?,
                param_usize(params, "len")?,
            );
            let w = fill_f32(dout * din, 5);
            let x = fill_f32(din * len, 6);
            let mut worst = 0.0f64;
            for j in 0..len.min(2) {
                for i in 0..dout.min(4) {
                    let expected: f64 = (0..din)
                        .map(|p| w[i + p * dout] as f64 * x[p + j * din] as f64)
                        .sum();
                    let actual = got[i + j * dout] as f64;
                    let scale = expected.abs().max(1.0);
                    worst = worst_of(worst, (actual - expected).abs() / scale);
                }
            }
            check_tol(worst, 1e-3, "decode dot_general")?
        }
        other => return Err(format!("unknown #2079 op {other}").into()),
    };
    let measurement = if arm == "eager-session" {
        runtime.with_eager_session(|s| {
            measure(timing, bytes as usize, |it| {
                timed(it, &mut || run(s), &mut || Ok(rt.synchronize()?))
            })
        })?
    } else if arm == "eager-per-call" {
        measure(timing, bytes as usize, |it| {
            timed(
                it,
                &mut || Ok(runtime.with_eager_session(run)?),
                &mut || Ok(rt.synchronize()?),
            )
        })?
    } else {
        return Err(format!("unknown eager-op arm {arm}").into());
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer: if arm == "eager-session" {
            vec![
                "one eager op in a reused with_eager_session",
                "one device synchronize at the batch edge",
            ]
        } else {
            vec![
                "one eager op with a with_eager_session entry/exit per operation",
                "one device synchronize at the batch edge",
            ]
        },
        outside_timer: vec![
            "backend, runtime and input construction (constant_from_host)",
            "correctness check (download)",
        ],
        extra: json!({"op": op, "output_bytes_per_op": bytes}),
    })
}

// ---------------------------------------------------------------------------
// #2080: gather of a few rows scales with the table
// ---------------------------------------------------------------------------

fn gather_table(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let (vocab, hidden, tokens) = (
        param_usize(params, "vocab")?,
        param_usize(params, "hidden")?,
        param_usize(params, "tokens")?,
    );
    let arm = param_str(params, "arm")?;
    let table = fill_f32(vocab * hidden, 21);
    let ids: Vec<i64> = (0..tokens)
        .map(|i| ((i * 37 + 2) % vocab) as i64)
        .collect();
    let cfg = GatherConfig {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, hidden],
    };
    let bytes = tokens * hidden * 4;
    let (error, measurement, timer) = if arm == "eager-gather" {
        let bk = backend(device)?;
        let rt = bk.runtime().clone();
        let runtime = EagerRuntime::with_cuda_backend(bk)?;
        let (table_t, index_t) = runtime.with_eager_session(|s| {
            Ok::<_, BoxError>((
                s.constant_from_host(host_f32(vec![vocab, hidden], table.clone())?)?,
                s.constant_from_host(Tensor::from_vec_col_major(vec![tokens], ids.clone())?)?,
            ))
        })?;
        let first = runtime.with_eager_session(|s| {
            Ok::<_, BoxError>(s.gather(&table_t, &index_t, cfg.clone())?)
        })?;
        rt.synchronize()?;
        let shape = first.shape().to_vec();
        let first = first.to_tensor()?;
        let back = download_tensor(&rt, &first)?;
        let got = back.as_slice::<f32>()?;
        // The issue's rows layout gathers (tokens, hidden): row t of the result is
        // row ids[t] of the table. Verify the first two rows against the host table.
        if shape != [tokens, hidden] {
            return Err(format!("unexpected gather output shape {shape:?}").into());
        }
        let mut worst = 0.0f64;
        for t in 0..tokens.min(2) {
            let row = ids[t] as usize;
            for h in 0..hidden.min(64) {
                // Both the table and the [tokens, hidden] result are column-major.
                let expected = table[row + h * vocab] as f64;
                let scale = expected.abs().max(1.0);
                worst = worst_of(worst, (got[t + h * tokens] as f64 - expected).abs() / scale);
            }
        }
        let error = check_tol(worst, 1e-6, "embedding gather")?;
        let m = runtime.with_eager_session(|s| {
            measure(timing, bytes, |it| {
                timed(
                    it,
                    &mut || Ok(s.gather(&table_t, &index_t, cfg.clone())?),
                    &mut || Ok(rt.synchronize()?),
                )
            })
        })?;
        (
            error,
            m,
            vec![
                "one eager gather of 8 indices from a [vocab, hidden] f32 device table",
                "one device synchronize at the batch edge",
            ],
        )
    } else if arm == "cudarc-row-copies" {
        let ctx = CudaContext::new(device)?;
        let stream = ctx.new_stream()?;
        // The reference copies contiguous rows, so it keeps a row-major copy of
        // the same table: table_rm[r * hidden + c] is table[r, c] in tenferro's
        // column-major [vocab, hidden].
        let mut table_rm = vec![0.0f32; vocab * hidden];
        for c in 0..hidden {
            for r in 0..vocab {
                table_rm[r * hidden + c] = table[r + c * vocab];
            }
        }
        let table_dev = stream.clone_htod(&table_rm)?;
        let mut out = unsafe { stream.alloc::<f32>(tokens * hidden) }?;
        for (t, id) in ids.iter().enumerate() {
            let row = *id as usize * hidden;
            stream.memcpy_dtod(
                &table_dev.slice(row..row + hidden),
                &mut out.slice_mut(t * hidden..(t + 1) * hidden),
            )?;
        }
        stream.synchronize()?;
        let back = stream.clone_dtoh(&out)?;
        let mut worst = 0.0f64;
        for (t, id) in ids.iter().enumerate() {
            let row = *id as usize;
            for h in 0..hidden.min(64) {
                let expected = table[row + h * vocab] as f64;
                let scale = expected.abs().max(1.0);
                worst = worst_of(worst, (back[t * hidden + h] as f64 - expected).abs() / scale);
            }
        }
        let error = check_tol(worst, 0.0, "cudarc row copies")?;
        let m = measure(timing, bytes, |it| {
            timed(
                it,
                // An allocation-returning operation, matching the eager arm's
                // output policy, so the two are comparable.
                &mut || -> Result<cudarc::driver::CudaSlice<f32>> {
                    let mut out = unsafe { stream.alloc::<f32>(tokens * hidden) }?;
                    for (t, id) in ids.iter().enumerate() {
                        let row = *id as usize * hidden;
                        stream.memcpy_dtod(
                            &table_dev.slice(row..row + hidden),
                            &mut out.slice_mut(t * hidden..(t + 1) * hidden),
                        )?;
                    }
                    Ok(out)
                },
                &mut || Ok(stream.synchronize()?),
            )
        })?;
        (
            error,
            m,
            vec![
                "8 device-to-device row copies (cudarc) into a freshly allocated destination",
                "stream.synchronize at the batch edge",
            ],
        )
    } else {
        return Err(format!("unknown gather arm {arm}").into());
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer,
        outside_timer: vec![
            "backend/runtime/context construction",
            "table generation and upload",
            "correctness check (download)",
        ],
        extra: json!({
            "table_bytes": vocab * hidden * 4,
            "output_bytes_per_op": bytes,
        }),
    })
}

// ---------------------------------------------------------------------------
// #2082: reshape of an uploaded constant copies the data
// ---------------------------------------------------------------------------

fn reshape_materialize(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let elements = param_usize(params, "elements")?;
    let (rows, cols) = (param_usize(params, "rows")?, param_usize(params, "cols")?);
    if rows * cols != elements {
        return Err("reshape target does not preserve the element count".into());
    }
    let arm = param_str(params, "arm")?;
    let bk = backend(device)?;
    let rt = bk.runtime().clone();
    let runtime = EagerRuntime::with_cuda_backend(bk)?;
    let data = fill_f32(elements, 33);
    let (constant, produced) = runtime.with_eager_session(|s| -> Result<(EagerTensor, EagerTensor)> {
        let constant = s.constant_from_host(host_f32(vec![elements], data.clone())?)?;
        let left = s.constant_from_host(host_f32(vec![elements], data.clone())?)?;
        let right = s.constant_from_host(host_f32(vec![elements], data.clone())?)?;
        let produced = s.add(&left, &right)?;
        Ok((constant, produced))
    })?;
    let input = if arm == "constant" {
        &constant
    } else if arm == "op-output" {
        &produced
    } else {
        return Err(format!("unknown reshape arm {arm}").into());
    };
    let target = vec![rows, cols];
    let first = runtime
        .with_eager_session(|s| Ok::<_, BoxError>(s.reshape(input, target.clone())?))?
        .to_tensor()?;
    rt.synchronize()?;
    let back = download_tensor(&rt, &first)?;
    let got = back.as_slice::<f32>()?;
    let factor = if arm == "op-output" { 2.0 } else { 1.0 };
    let mut worst = 0.0f64;
    for i in 0..elements.min(64) {
        let expected = factor * data[i] as f64;
        let scale = expected.abs().max(1.0);
        worst = worst_of(worst, (got[i] as f64 - expected).abs() / scale);
    }
    let error = check_tol(worst, 1e-6, "reshape values")?;
    let m = runtime.with_eager_session(|s| {
        measure(timing, elements * 4, |it| {
            timed(
                it,
                &mut || Ok(s.reshape(input, target.clone())?),
                &mut || Ok(rt.synchronize()?),
            )
        })
    })?;
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(m),
        timer: vec![
            "one reshape of a [2^26] f32 device tensor to [2^16, 2^10]",
            "one device synchronize at the batch edge",
        ],
        outside_timer: vec![
            "backend/runtime construction",
            "host fill, upload or elementwise production of the input",
            "correctness check (download)",
        ],
        extra: json!({"input_bytes": elements * 4}),
    })
}

// ---------------------------------------------------------------------------
// #2083: per-session cuBLAS handle
// ---------------------------------------------------------------------------

const VENDOR_OP: &str = "perf_issues.vendor_handle";

/// Run one cuBLAS dgemm on the session's stream from inside a raw scope.
///
/// `handle` is the reusable handle of the `reused` arm (None creates one), and
/// `keep` says whether the handle created or passed in survives the call. The
/// output is returned so the caller can keep it alive until its clock stops.
fn cublas_dgemm_raw(
    s: &mut dyn BackendSession,
    a: &Tensor,
    b: &Tensor,
    n: usize,
    keep: bool,
    handle: Option<cublas_sys::cublasHandle_t>,
) -> Result<(Option<cublas_sys::cublasHandle_t>, Tensor)> {
    use tenferro_tensor::Error;
    let dim = i32::try_from(n).map_err(|error| format!("{error}"))?;
    let mut out = None;
    let result = with_cuda_exec_session(s, |cs| {
        cs.with_raw(VENDOR_OP, |raw| {
            let ar = raw.tensor(
                a.as_typed::<f64>()
                    .ok_or_else(|| Error::invalid_argument(VENDOR_OP, "a", "not f64"))?,
            )?;
            let br = raw.tensor(
                b.as_typed::<f64>()
                    .ok_or_else(|| Error::invalid_argument(VENDOR_OP, "b", "not f64"))?,
            )?;
            let mut c = raw.alloc_output::<f64>(&[n, n])?;
            let cr = raw.tensor_mut(&mut c)?;
            let (alpha, beta) = (1.0_f64, 0.0_f64);
            let handle = match handle {
                Some(existing) => existing,
                None => cublas::create_handle().map_err(|e| Error::backend_source(VENDOR_OP, e))?,
            };
            // SAFETY: `raw_handle` is the live stream of this raw session, and the
            // three device pointers come from references that stay borrowed until
            // the synchronize below. A, B and C are all n x n in compact
            // column-major storage, which is cuBLAS's native layout.
            let launched = unsafe {
                cublas::set_stream(handle, raw.stream().raw_handle() as cublas_sys::cudaStream_t)
                    .and_then(|()| {
                        cublas::dgemm(
                            handle,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                            dim,
                            dim,
                            dim,
                            &alpha,
                            ar.raw_ptr().cast(),
                            dim,
                            br.raw_ptr().cast(),
                            dim,
                            &beta,
                            cr.raw_ptr().cast(),
                            dim,
                        )
                    })
            };
            let synchronized = raw.synchronize();
            // SAFETY: the barrier above means no work using the handle is pending.
            let destroyed = if keep {
                Ok(())
            } else {
                unsafe { cublas::destroy_handle(handle) }
            };
            launched.map_err(|e| Error::backend_source(VENDOR_OP, e))?;
            synchronized?;
            destroyed.map_err(|e| Error::backend_source(VENDOR_OP, e))?;
            out = Some(Tensor::from(c));
            Ok(if keep { Some(handle) } else { None })
        })
    });
    match result {
        Some(inner) => Ok((inner?, out.ok_or("the raw scope produced no output")?)),
        None => Err("not a CUDA execution session".into()),
    }
}

/// One dgemm through the handle the runtime already owns (#2083's ask), inside
/// a raw scope, with the same host barrier as the other arms.
fn cublas_dgemm_runtime_handle(
    bk: &mut CudaBackend,
    a: &Tensor,
    b: &Tensor,
    n: usize,
) -> Result<Tensor> {
    use tenferro_tensor::Error;
    let dim = i32::try_from(n).map_err(|error| format!("{error}"))?;
    let out = bk.with_backend_session(|s| -> Result<Tensor> {
        let out = with_cuda_exec_session(s, |cs| {
            cs.with_raw(VENDOR_OP, |raw| {
                let ar = raw.tensor(
                    a.as_typed::<f64>()
                        .ok_or_else(|| Error::invalid_argument(VENDOR_OP, "a", "not f64"))?,
                )?;
                let br = raw.tensor(
                    b.as_typed::<f64>()
                        .ok_or_else(|| Error::invalid_argument(VENDOR_OP, "b", "not f64"))?,
                )?;
                let mut c = raw.alloc_output::<f64>(&[n, n])?;
                let cr = raw.tensor_mut(&mut c)?;
                let (alpha, beta) = (1.0_f64, 0.0_f64);
                raw.with_cublas_handle(
                    VENDOR_OP,
                    cublas_sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
                    |handle| {
                        // SAFETY: the device pointers come from references that
                        // stay borrowed until the synchronize below; all three
                        // operands are n x n compact column-major, cuBLAS's
                        // native layout, with C the only buffer written.
                        unsafe {
                            cublas::dgemm(
                                handle,
                                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                dim,
                                dim,
                                dim,
                                &alpha,
                                ar.raw_ptr().cast(),
                                dim,
                                br.raw_ptr().cast(),
                                dim,
                                &beta,
                                cr.raw_ptr().cast(),
                                dim,
                            )
                        }
                        .map_err(|e| Error::backend_source(VENDOR_OP, e))
                    },
                )?;
                raw.synchronize()?;
                Ok(Tensor::from(c))
            })
        })
        .ok_or("not a CUDA execution session")??;
        Ok(out)
    })??;
    Ok(out)
}

fn destroy_raw_handle(bk: &mut CudaBackend, handle: cublas_sys::cublasHandle_t) -> Result<()> {
    use tenferro_tensor::Error;
    bk.with_backend_session(|s| -> Result<()> {
        with_cuda_exec_session(s, |cs| {
            cs.with_raw(VENDOR_OP, |_raw| {
                // SAFETY: the handle was created inside a raw session of this
                // runtime and is not used after this point.
                unsafe { cublas::destroy_handle(handle) }
                    .map_err(|e| Error::backend_source(VENDOR_OP, e))?;
                Ok(())
            })
        })
        .ok_or("not a CUDA execution session")??;
        Ok(())
    })??;
    Ok(())
}

fn vendor_handle(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let arm = param_str(params, "arm")?.to_string();
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    let n = 256usize;
    let a = upload_tensor(&rt, &host(vec![n, n], values(n * n, 41))?)?;
    let b = upload_tensor(&rt, &host(vec![n, n], values(n * n, 42))?)?;
    rt.synchronize()?;
    // Correctness: the raw cuBLAS route is verified against tenferro's own matmul
    // before either arm is timed.
    let (_, probe) = bk.with_backend_session(|s| cublas_dgemm_raw(s, &a, &b, n, false, None))??;
    rt.synchronize()?;
    let reference = bk.with_backend_session(|s| -> Result<Tensor> {
        Ok(a.dot_general(&b, matmul_cfg(&[1], &[0]), s)?)
    })??;
    rt.synchronize()?;
    let error = check_tol(
        max_rel(
            download_tensor(&rt, &probe)?.as_slice::<f64>()?,
            download_tensor(&rt, &reference)?.as_slice::<f64>()?,
        ),
        1e-12,
        "cuBLAS dgemm vs tenferro matmul",
    )?;
    // The `reused` arm keeps one handle for every operation; creating it is setup.
    let persistent = if arm == "reused" {
        bk.with_backend_session(|s| cublas_dgemm_raw(s, &a, &b, n, true, None))?
            .map(|(handle, _)| handle)?
    } else {
        None
    };
    let keep = arm == "reused";
    let bytes = n * n * 8;
    let measurement = if arm == "runtime-handle" {
        measure(timing, bytes, |it| {
            timed(
                it,
                &mut || cublas_dgemm_runtime_handle(&mut bk, &a, &b, n),
                &mut || Ok(()),
            )
        })?
    } else {
        bk.with_backend_session(|s| {
            measure(timing, bytes, |it| {
                timed(
                    it,
                    &mut || {
                        let (_, out) = cublas_dgemm_raw(s, &a, &b, n, keep, persistent)?;
                        Ok(out)
                    },
                    &mut || Ok(()),
                )
            })
        })??
    };
    if let Some(handle) = persistent {
        destroy_raw_handle(&mut bk, handle)?;
    }
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer: vec![
            match arm.as_str() {
                "runtime-handle" => {
                    "dgemm inside one with_raw scope per operation, on the runtime's own cuBLAS handle"
                }
                "reused" => "cublasSetStream + dgemm inside one with_raw scope per operation (handle reused)",
                _ => "cublasCreate + cublasSetStream + dgemm + cublasDestroy inside one with_raw scope per operation",
            },
            "host barrier (raw.synchronize) inside the scope",
        ],
        outside_timer: vec![
            "backend construction",
            "operand upload",
            "correctness probe",
            "persistent handle creation for the reused arm",
        ],
        extra: json!({"arm": arm, "gemm_n": n}),
    })
}
// #1924: vendor-library churn per backend cycle
// ---------------------------------------------------------------------------

fn vendor_churn(params: &Value, device: usize) -> Result<CaseOutcome> {
    let cycles = param_usize(params, "cycles")?;
    let n = param_usize(params, "gemm_n")?;
    let ctx = CudaContext::new(device)?;
    // Warm the process (driver, NVRTC, vendor libraries) before the first reading,
    // and verify the GEMM the cycles run against a host reference.
    let error = {
        let mut bk = backend(device)?;
        let rt = bk.runtime().clone();
        let host_a = values(n * n, 51);
        let host_b = values(n * n, 52);
        let a = upload_tensor(&rt, &host(vec![n, n], host_a.clone())?)?;
        let b = upload_tensor(&rt, &host(vec![n, n], host_b.clone())?)?;
        let out = bk.with_backend_session(|s| -> Result<Tensor> {
            Ok(a.dot_general(&b, matmul_cfg(&[1], &[0]), s)?)
        })??;
        rt.synchronize()?;
        let got = download_tensor(&rt, &out)?;
        let got = got.as_slice::<f64>()?;
        let mut worst = 0.0f64;
        for (i, j) in [(0usize, 0usize), (1, 2), (3, 5)] {
            let expected: f64 = (0..n).map(|p| host_a[i + p * n] * host_b[p + j * n]).sum();
            worst = worst_of(worst, max_rel(&[got[i + j * n]], &[expected]));
        }
        check_tol(worst, 1e-12, "churn-cycle gemm")?
    };
    let free_before = device_free_bytes(&ctx)?;
    let rss_before = process_rss_bytes();
    let mut per_cycle = Vec::new();
    let mut previous = free_before;
    for _ in 0..cycles {
        let mut bk = backend(device)?;
        let rt = bk.runtime().clone();
        let a = upload_tensor(&rt, &host(vec![n, n], values(n * n, 53))?)?;
        let b = upload_tensor(&rt, &host(vec![n, n], values(n * n, 54))?)?;
        bk.with_backend_session(|s| -> Result<Tensor> {
            Ok(a.dot_general(&b, matmul_cfg(&[1], &[0]), s)?)
        })??;
        rt.synchronize()?;
        drop(a);
        drop(b);
        drop(bk);
        drop(rt);
        let free = device_free_bytes(&ctx)?;
        let rss = process_rss_bytes();
        per_cycle.push(json!({
            "rss_bytes": rss,
            "free_bytes": free,
            "delta_bytes": free as i64 - previous as i64,
        }));
        previous = free;
    }
    let free_after = device_free_bytes(&ctx)?;
    let rss_after = process_rss_bytes();
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: None,
        timer: Vec::new(),
        outside_timer: vec!["everything (a state diagnostic, no timing)"],
        extra: json!({
            "cycles": cycles,
            "rss_bytes_before": rss_before,
            "rss_bytes_after": rss_after,
            "rss_growth_bytes": rss_after as i64 - rss_before as i64,
            "free_bytes_before": free_before,
            "free_bytes_after": free_after,
            "retained_bytes_over_cycles": free_before as i64 - free_after as i64,
            "per_cycle": per_cycle,
        }),
    })
}

// ---------------------------------------------------------------------------
// #2081: device memory of dropped tensors is never returned to the driver
// ---------------------------------------------------------------------------

fn pool_retention(params: &Value, device: usize) -> Result<CaseOutcome> {
    let gib = param_usize(params, "gib")?;
    let count = param_usize(params, "count")?;
    let probe_gib = param_usize(params, "probe_gib")?;
    let arm = param_str(params, "arm")?.to_string();
    let elements = gib * (1usize << 30) / 8;
    let ctx = CudaContext::new(device)?;
    // A non-zero deterministic payload, so the round-trip check has teeth.
    let host_data: Vec<f64> = (0..elements)
        .map(|i| ((i % 97) as f64) * 0.01 - 0.5)
        .collect();
    let host_tensor = host(vec![elements], host_data.clone())?;
    let free_start = device_free_bytes(&ctx)?;

    let bk = backend(device)?;
    let rt = bk.runtime().clone();
    // The issue's sequence: eight 1 GiB tensors alive at the high-water mark,
    // then all of them dropped.
    let mut allocated = Vec::with_capacity(count);
    for _ in 0..count {
        allocated.push(upload_tensor(&rt, &host_tensor)?);
    }
    rt.synchronize()?;
    // Verify one of them really holds the uploaded payload, outside any timing.
    let probe = download_tensor(&rt, &allocated[0])?;
    let got = probe.as_slice::<f64>()?;
    let mut worst = 0.0f64;
    for i in 0..8 {
        worst = worst_of(worst, (got[i] - host_data[i]).abs() / host_data[i].abs().max(1.0));
    }
    let error = check_tol(worst, 1e-12, "1 GiB tensor round trip")?;
    let free_while_alive = device_free_bytes(&ctx)?;
    allocated.clear();
    let free_after_drop;
    let free_after_recreate;
    if arm == "runtime-recreate" {
        // Drop the runtime as well. The pool belongs to the process-global
        // client of the device, so this reading is what a later runtime sees.
        drop(bk);
        drop(rt);
        free_after_drop = device_free_bytes(&ctx)?;
        let recreated = backend(device)?;
        free_after_recreate = device_free_bytes(&ctx)?;
        drop(recreated);
    } else {
        drop(bk);
        drop(rt);
        free_after_drop = device_free_bytes(&ctx)?;
        free_after_recreate = free_after_drop;
    }
    // `driver-alloc-after-free` adds the synchronous driver allocation the
    // issue's 6 GiB cuMemAlloc could not serve.
    let probe_status = if arm == "driver-alloc-after-free" {
        ctx.bind_to_thread()?;
        let probe_bytes = probe_gib * (1usize << 30);
        match unsafe { cudarc::driver::result::malloc_sync(probe_bytes) } {
            Ok(pointer) => {
                let freed = unsafe { cudarc::driver::result::free_sync(pointer) };
                match freed {
                    Ok(()) => format!("ok ({probe_bytes} bytes)"),
                    Err(free_error) => format!("allocated, free failed: {free_error}"),
                }
            }
            Err(alloc_error) => format!("failed: {alloc_error}"),
        }
    } else {
        format!("not requested ({arm} arm)")
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: None,
        timer: Vec::new(),
        outside_timer: vec!["everything (a state diagnostic, no timing)"],
        extra: json!({
            "arm": arm,
            "tensors": count,
            "tensor_gib": gib,
            "probe_gib": probe_gib,
            "probe_status": probe_status,
            "free_bytes_start": free_start,
            "free_bytes_while_alive": free_while_alive,
            "free_bytes_after_drop": free_after_drop,
            "free_bytes_after_recreate": free_after_recreate,
            "peak_used_bytes": free_start - free_while_alive,
            "retained_bytes_after_drop": free_start as i64 - free_after_drop as i64,
        }),
    })
}

// ---------------------------------------------------------------------------
// #1839 / #1849: cuSOLVER SVD driver
// ---------------------------------------------------------------------------

fn svd_driver_of(name: &str) -> Result<SvdDriver> {
    Ok(match name {
        "auto" => SvdDriver::Auto,
        "gesvdj" => SvdDriver::Gesvdj,
        "gesvd" => SvdDriver::Gesvd,
        "xgesvdp" => SvdDriver::Xgesvdp,
        other => return Err(format!("unknown SVD driver {other}").into()),
    })
}

fn host_matmul_c64(a: &[Complex64], b: &[Complex64], m: usize, k: usize, n: usize) -> Vec<Complex64> {
    let mut c = vec![Complex64::new(0.0, 0.0); m * n];
    for j in 0..n {
        for p in 0..k {
            for i in 0..m {
                c[i + j * m] += a[i + p * m] * b[p + j * k];
            }
        }
    }
    c
}

fn svd_driver(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let n = param_usize(params, "n")?;
    let driver = param_str(params, "driver")?;
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    // A DMRG-like complex f64 matrix: clustered, well-separated spectrum.
    let mut data = complex_values(n * n, 31);
    for i in 0..n {
        data[i + i * n] += Complex64::new(n as f64, 0.0);
    }
    let a = upload_tensor(&rt, &host_c64(vec![n, n], data.clone())?)?;
    rt.synchronize()?;
    let options = SvdOptions {
        driver: svd_driver_of(driver)?,
        ..SvdOptions::default()
    };
    let bytes = n * n * 16 * 3;
    let (error, outs) = bk.with_backend_session(|s| -> Result<(f64, Vec<Tensor>)> {
        let (u, sv, vh) = a.svd_with_options(options.clone(), s)?;
        let d = |t: &Tensor| -> Result<Vec<Complex64>> {
            Ok(download_tensor(&rt, t)?.as_slice::<Complex64>()?.to_vec())
        };
        // Singular values are real for a complex input.
        let sh = download_tensor(&rt, &sv)?.as_slice::<f64>()?.to_vec();
        let (uh, vhh) = (d(&u)?, d(&vh)?);
        let mut us = uh.clone();
        for j in 0..n {
            for i in 0..n {
                us[i + j * n] *= sh[j];
            }
        }
        let rebuilt = host_matmul_c64(&us, &vhh, n, n, n);
        let mut worst = 0.0f64;
        for (got, want) in rebuilt.iter().zip(&data) {
            let scale = want.norm().max(1.0);
            worst = worst_of(worst, (got - want).norm() / scale);
        }
        Ok((check_tol(worst, 1e-8, "svd reconstruction")?, vec![u, sv, vh]))
    })??;
    black_box(&outs);
    let m = bk.with_backend_session(|s| {
        measure(timing, bytes, |it| {
            timed_in_session(it, s, |s| {
                let (u, sv, vh) = a.svd_with_options(options.clone(), s)?;
                Ok(vec![u, sv, vh])
            })
        })
    })??;
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(m),
        timer: vec![
            "one thin SVD of an n x n c64 device matrix with an explicit cuSOLVER driver",
            "device synchronize at the batch edge",
        ],
        outside_timer: vec![
            "backend construction",
            "host matrix construction and upload",
            "reconstruction check (download)",
        ],
        extra: json!({"driver": driver, "n": n}),
    })
}

// ---------------------------------------------------------------------------
// #1846 / #1852 / #1923: batched cuSOLVER eigh driver
// ---------------------------------------------------------------------------

fn eigh_driver_of(name: &str) -> Result<EighDriver> {
    Ok(match name {
        "auto" => EighDriver::Auto,
        "syevd" => EighDriver::Syevd,
        "syevj" => EighDriver::Syevj,
        other => return Err(format!("unknown eigh driver {other}").into()),
    })
}

fn eigh_driver(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let n = param_usize(params, "n")?;
    let batch = param_usize(params, "batch")?;
    let driver = param_str(params, "driver")?;
    let dtype = param_str(params, "dtype")?;
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    let options = EighOptions {
        driver: eigh_driver_of(driver)?,
        ..EighOptions::default()
    };
    let bytes = batch * n * n * 16 * 2;
    let (error, measurement, timer) = if dtype == "f64" {
        let all: Vec<f64> = (0..batch)
            .flat_map(|b| {
                let mut item = well_conditioned(n, 60 + b as u64);
                let mut sym = vec![0.0; n * n];
                for j in 0..n {
                    for i in 0..n {
                        sym[i + j * n] = 0.5 * (item[i + j * n] + item[j + i * n]);
                    }
                }
                item = sym;
                item
            })
            .collect();
        let a = upload_tensor(&rt, &host(vec![n, n, batch], all.clone())?)?;
        rt.synchronize()?;
        bk.with_backend_session(|s| -> Result<(f64, Measurement, Vec<&str>)> {
            let (w, v) = a.eigh_with_options(options.clone(), s)?;
            let wh = download_tensor(&rt, &w)?.as_slice::<f64>()?.to_vec();
            let vh = download_tensor(&rt, &v)?.as_slice::<f64>()?.to_vec();
            let mut worst = 0.0f64;
            for b in [0usize, batch - 1] {
                let base = b * n * n;
                worst = worst.max(eigh_residual_f64(
                    &all[base..base + n * n],
                    &wh[b * n..(b + 1) * n],
                    &vh[base..base + n * n],
                    n,
                ));
            }
            let error = check_tol(worst, 1e-8, "batched eigh residual")?;
            let m = measure(timing, bytes, |it| {
                timed_in_session(it, s, |s| {
                    let (w, v) = a.eigh_with_options(options.clone(), s)?;
                    Ok(vec![w, v])
                })
            })?;
            Ok((error, m, vec!["one batched eigh call ([n, n, batch], batch trailing)", "device synchronize at the batch edge"]))
        })??
    } else {
        let all: Vec<Complex64> = (0..batch)
            .flat_map(|b| {
                let item: Vec<Complex64> = complex_values(n * n, 70 + b as u64);
                let mut sym = vec![Complex64::new(0.0, 0.0); n * n];
                for j in 0..n {
                    for i in 0..n {
                        sym[i + j * n] = (item[i + j * n] + item[j + i * n].conj()) * 0.5;
                    }
                }
                for i in 0..n {
                    sym[i + i * n] += Complex64::new(n as f64, 0.0);
                }
                sym
            })
            .collect();
        let a = upload_tensor(&rt, &host_c64(vec![n, n, batch], all.clone())?)?;
        rt.synchronize()?;
        bk.with_backend_session(|s| -> Result<(f64, Measurement, Vec<&str>)> {
            let (w, v) = a.eigh_with_options(options.clone(), s)?;
            let wh = download_tensor(&rt, &w)?.as_slice::<f64>()?.to_vec();
            let vh = download_tensor(&rt, &v)?.as_slice::<Complex64>()?.to_vec();
            let mut worst = 0.0f64;
            for b in [0usize, batch - 1] {
                let base = b * n * n;
                worst = worst.max(eigh_residual_c64(
                    &all[base..base + n * n],
                    &wh[b * n..(b + 1) * n],
                    &vh[base..base + n * n],
                    n,
                ));
            }
            let error = check_tol(worst, 1e-8, "batched eigh residual (c64)")?;
            let m = measure(timing, bytes, |it| {
                timed_in_session(it, s, |s| {
                    let (w, v) = a.eigh_with_options(options.clone(), s)?;
                    Ok(vec![w, v])
                })
            })?;
            Ok((error, m, vec!["one batched eigh call ([n, n, batch], batch trailing), c64 input", "device synchronize at the batch edge"]))
        })??
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer,
        outside_timer: vec![
            "backend construction",
            "host batch construction and upload",
            "residual check (download)",
        ],
        extra: json!({"driver": driver, "n": n, "batch": batch, "dtype": dtype}),
    })
}

/// Error of the eigendecomposition of one item: the residual `||A v_j - w_j v_j|| / ||A||`,
/// the deviation of every returned vector from unit norm, and the pairwise overlap
/// of the checked columns.
///
/// The residual alone accepts `V = 0` (and repeated copies of one valid vector),
/// because `A 0 - w 0 = 0`; the norm and overlap terms reject those.
fn eigh_residual_f64(a: &[f64], w: &[f64], v: &[f64], n: usize) -> f64 {
    let mut worst = 0.0f64;
    let norm = a.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);
    for j in 0..n {
        let vnorm = (0..n).map(|i| v[i + j * n] * v[i + j * n]).sum::<f64>().sqrt();
        worst = worst_of(worst, (vnorm - 1.0).abs());
        let residual = (0..n)
            .map(|i| {
                let av: f64 = (0..n).map(|p| a[i + p * n] * v[p + j * n]).sum();
                (av - w[j] * v[i + j * n]).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        worst = worst_of(worst, residual / norm);
    }
    for j in 0..n.min(3) {
        for k in (j + 1)..n.min(4) {
            let dot: f64 = (0..n).map(|i| v[i + j * n] * v[i + k * n]).sum();
            worst = worst_of(worst, dot.abs());
        }
    }
    worst
}

fn eigh_residual_c64(a: &[Complex64], w: &[f64], v: &[Complex64], n: usize) -> f64 {
    let mut worst = 0.0f64;
    let norm = a.iter().fold(0.0f64, |m, x| m.max(x.norm())).max(1.0);
    for j in 0..n {
        let vnorm = (0..n)
            .map(|i| v[i + j * n].norm_sqr())
            .sum::<f64>()
            .sqrt();
        worst = worst_of(worst, (vnorm - 1.0).abs());
        let residual = (0..n)
            .map(|i| {
                let av: Complex64 = (0..n).map(|p| a[i + p * n] * v[p + j * n]).sum();
                (av - v[i + j * n] * w[j]).norm_sqr()
            })
            .sum::<f64>()
            .sqrt();
        worst = worst_of(worst, residual / norm);
    }
    for j in 0..n.min(3) {
        for k in (j + 1)..n.min(4) {
            let dot: Complex64 = (0..n)
                .map(|i| v[i + j * n].conj() * v[i + k * n])
                .sum();
            worst = worst_of(worst, dot.norm());
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// #1898: a pure Hadamard product routed through dot_general
// ---------------------------------------------------------------------------

fn hadamard_dot_general(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let (rows, cols) = (param_usize(params, "rows")?, param_usize(params, "cols")?);
    let arm = param_str(params, "arm")?;
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    let x = fill_f32(rows * cols, 81);
    let y = fill_f32(rows * cols, 82);
    let a = upload_tensor(&rt, &host_f32(vec![rows, cols], x.clone())?)?;
    let b = upload_tensor(&rt, &host_f32(vec![rows, cols], y.clone())?)?;
    rt.synchronize()?;
    let bytes = rows * cols * 4;
    // Every axis is a batch axis and nothing is contracted: the elementwise product.
    let cfg = DotGeneralConfig {
        lhs_contracting_dims: [].as_slice().into(),
        rhs_contracting_dims: [].as_slice().into(),
        lhs_batch_dims: vec![0usize, 1].into(),
        rhs_batch_dims: vec![0usize, 1].into(),
    };
    let (error, measurement, timer) = bk.with_backend_session(|s| -> Result<(f64, Measurement, Vec<&str>)> {
        let out = if arm == "dot-general" {
            a.dot_general(&b, cfg.clone(), s)?
        } else {
            a.mul(&b, s)?
        };
        let got = download_tensor(&rt, &out)?;
        let got = got.as_slice::<f32>()?;
        let mut worst = 0.0f64;
        for i in 0..(rows * cols).min(64) {
            let expected = x[i] as f64 * y[i] as f64;
            let scale = expected.abs().max(1.0);
            worst = worst_of(worst, (got[i] as f64 - expected).abs() / scale);
        }
        let error = check_tol(worst, 1e-6, "hadamard product")?;
        let m = measure(timing, bytes, |it| {
            timed_in_session(it, s, |s| {
                Ok(if arm == "dot-general" {
                    a.dot_general(&b, cfg.clone(), s)?
                } else {
                    a.mul(&b, s)?
                })
            })
        })?;
        Ok((
            error,
            m,
            vec![if arm == "dot-general" {
                "one dot_general with every axis as a batch axis (pure Hadamard)"
            } else {
                "one elementwise multiply"
            }, "device synchronize at the batch edge"],
        ))
    })??;
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer,
        outside_timer: vec!["backend construction", "operand upload", "correctness check (download)"],
        extra: json!({"elements": rows * cols}),
    })
}

// ---------------------------------------------------------------------------
// #1873 / #1805: distinct contraction plans evict the plan cache
// ---------------------------------------------------------------------------

fn contraction_plan_churn(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let distinct = param_usize(params, "distinct")?;
    let n = param_usize(params, "n")?;
    let arm = param_str(params, "arm")?;
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    let cfg = matmul_cfg(&[1], &[0]);
    // Distinct contraction-plan keys come from distinct extents; the count must
    // exceed the backend's plan-cache capacity for the churn to evict anything.
    let mut plan_inputs = Vec::new();
    if arm == "distinct-plans" {
        for index in 0..distinct {
            let (m, k) = (n + index, n);
            let a = upload_tensor(&rt, &host(vec![m, k], values(m * k, 90 + index as u64))?)?;
            let b = upload_tensor(&rt, &host(vec![k, m], values(k * m, 190 + index as u64))?)?;
            plan_inputs.push((a, b));
        }
    } else {
        for index in 0..distinct {
            let a = upload_tensor(&rt, &host(vec![n, n], values(n * n, 90 + index as u64))?)?;
            let b = upload_tensor(&rt, &host(vec![n, n], values(n * n, 190 + index as u64))?)?;
            plan_inputs.push((a, b));
        }
    }
    rt.synchronize()?;
    let (error, measurement) = bk.with_backend_session(|s| -> Result<(f64, Measurement)> {
        let first = plan_inputs[0].0.dot_general(&plan_inputs[0].1, cfg.clone(), s)?;
        let got = download_tensor(&rt, &first)?;
        let got = got.as_slice::<f64>()?;
        let (m, k) = (n, n);
        let data = values(m * k, 90);
        let rhs = values(k * m, 190);
        let expected: f64 = (0..k).map(|p| data[1 + p * m] * rhs[p + 2 * k]).sum();
        let error = check_tol(max_rel(&[got[1 + 2 * m]], &[expected]), 1e-12, "contraction")?;
        // The timed operation is the re-run of the first shape alone, one
        // isolated call per sample. The churn that fills the plan cache is the
        // (untimed) precondition, and the device is drained before the clock
        // starts so its queued work cannot leak in.
        let m = measure_isolated(timing, || {
            let mut churn = Vec::with_capacity(plan_inputs.len());
            for (a, b) in &plan_inputs {
                churn.push(a.dot_general(b, cfg.clone(), s)?);
            }
            device_sync(s)?;
            let start = Instant::now();
            let out = plan_inputs[0].0.dot_general(&plan_inputs[0].1, cfg.clone(), s)?;
            device_sync(s)?;
            let elapsed = start.elapsed().as_nanos();
            black_box(&out);
            black_box(&churn);
            Ok(elapsed)
        })?;
        Ok((error, m))
    })??;
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer: vec![
            if arm == "distinct-plans" {
                "one re-run of the first contraction after the plan cache was filled by distinct shapes"
            } else {
                "one re-run of the same contraction"
            },
            "one isolated call per sample; device synchronize after the call",
        ],
        outside_timer: vec![
            "backend construction",
            "operand construction and upload",
            "the churn that fills (or repeats) the plan cache, and the synchronize after it",
            "correctness check (download)",
        ],
        extra: json!({"distinct_plans": distinct, "n": n}),
    })
}

// ---------------------------------------------------------------------------
// #1949: owned vs borrowed dot_general destination
// ---------------------------------------------------------------------------

fn owned_destination(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let n = param_usize(params, "n")?;
    let arm = param_str(params, "arm")?;
    let owned = arm == "owned-dst";
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    let host_a = values(n * n, 61);
    let host_b = values(n * n, 62);
    let a = upload_tensor(&rt, &host(vec![n, n], host_a.clone())?)?;
    let b = upload_tensor(&rt, &host(vec![n, n], host_b.clone())?)?;
    // One caller-provided destination, written by every operation.
    let mut dst = upload_tensor(&rt, &host(vec![n, n], vec![0.0; n * n])?)?;
    rt.synchronize()?;
    let cfg = matmul_cfg(&[1], &[0]);
    let expected: f64 = (0..n).map(|p| host_a[1 + p * n] * host_b[p + 2 * n]).sum();
    // `from_tensor` is the owned-tensor destination route (the write_device_ptr
    // path the #1949 memo covers); `from_view` is the borrowed-view route, which
    // keeps the resource round trip. Building the handle is untimed.
    let write = |s: &mut dyn BackendSession, dst: &mut Tensor| -> Result<u128> {
        let destination = if owned {
            TensorWrite::from_tensor(dst)
        } else {
            let view = dst
                .as_typed_mut::<f64>()
                .ok_or("destination is not f64")?
                .as_view_mut();
            TensorWrite::from_view(tenferro_tensor::TensorViewMut::F64(view))
        };
        let lhs = TensorRead::from_tensor(&a);
        let rhs = TensorRead::from_tensor(&b);
        let start = Instant::now();
        s.dot_general_read_into(lhs, rhs, &cfg, destination)?;
        device_sync(s)?;
        Ok(start.elapsed().as_nanos())
    };
    let (error, measurement) = bk.with_backend_session(|s| -> Result<(f64, Measurement)> {
        // Validate the destination the arm actually wrote, outside timing.
        write(s, &mut dst)?;
        let back = download_tensor(&rt, &dst)?;
        let got = back.as_slice::<f64>()?;
        let error = check_tol(
            max_rel(&[got[1 + 2 * n]], &[expected]),
            1e-12,
            "destination contents",
        )?;
        // The handle cannot be batched, so the sample is one isolated call.
        let m = measure_isolated(timing, || write(s, &mut dst))?;
        Ok((error, m))
    })??;
    black_box(&dst);
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer: vec![
            if owned {
                "dot_general_read_into a reused owned destination"
            } else {
                "dot_general_read_into a reused borrowed-view destination"
            },
            "one isolated call per sample; device synchronize after the call",
        ],
        outside_timer: vec![
            "backend construction",
            "operand and destination upload",
            "the destination handle built per call",
            "correctness check (download)",
        ],
        extra: json!({"n": n}),
    })
}

// ---------------------------------------------------------------------------
// #1925: borrowed read views reuse the memoized device address
// ---------------------------------------------------------------------------

fn borrowed_input_gemm(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let n = param_usize(params, "n")?;
    let arm = param_str(params, "arm")?;
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    let host_a = values(n * n, 71);
    let host_b = values(n * n, 72);
    let a = upload_tensor(&rt, &host(vec![n, n], host_a.clone())?)?;
    // `b` is the logical right operand. The view arm borrows it as a transposed
    // view of its own transpose, so both arms contract the same values and only
    // the operand representation differs.
    let b = upload_tensor(&rt, &host(vec![n, n], host_b.clone())?)?;
    let mut b_transposed = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            b_transposed[i + j * n] = host_b[j + i * n];
        }
    }
    let b_t = upload_tensor(&rt, &host(vec![n, n], b_transposed)?)?;
    rt.synchronize()?;
    let cfg = matmul_cfg(&[1], &[0]);
    let expected: f64 = (0..n).map(|p| host_a[1 + p * n] * host_b[p + 2 * n]).sum();
    let (error, measurement) = bk.with_backend_session(|s| -> Result<(f64, Measurement)> {
        // The borrowed view is a transposed view of its own transpose, so the
        // logical operand matches the owned arm and only the representation
        // differs. The view is metadata-only and is built outside the clock.
        let source = b_t.as_typed::<f64>().ok_or("operand is not f64")?.as_view();
        let operands = || -> Result<(TensorRead<'_>, TensorRead<'_>)> {
            Ok((
                TensorRead::from_tensor(&a),
                if arm == "view-input" {
                    // A borrowed read view reuses the buffer's memoized device
                    // address (the #1925 fix) instead of resolving a new one.
                    TensorRead::from_view(TensorView::F64(source.transpose_view([1, 0])?))
                } else {
                    TensorRead::from_tensor(&b)
                },
            ))
        };
        // Validate the route this arm actually runs, on its own result, outside timing.
        let (lhs, rhs) = operands()?;
        let checked = s.dot_general_read(lhs, rhs, &cfg)?;
        device_sync(s)?;
        let got = download_tensor(&rt, &checked)?;
        let got = got.as_slice::<f64>()?;
        let error = check_tol(
            max_rel(&[got[1 + 2 * n]], &[expected]),
            1e-12,
            "view-operand gemm",
        )?;
        // One isolated call per sample; the returned tensor stays alive until
        // the clock has stopped.
        let m = measure_isolated(timing, || {
            let (lhs, rhs) = operands()?;
            let start = Instant::now();
            // A plain local binding: no harness allocation inside the clock, and
            // the returned tensor lives until the clock has stopped.
            let result = s.dot_general_read(lhs, rhs, &cfg)?;
            device_sync(s)?;
            let elapsed = start.elapsed().as_nanos();
            black_box(&result);
            drop(result);
            Ok(elapsed)
        })?;
        Ok((error, m))
    })??;
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer: vec![
            if arm == "view-input" {
                "one small dot_general_read with a borrowed transposed input view"
            } else {
                "one small dot_general_read with owned inputs"
            },
            "one isolated call per sample; device synchronize after the call",
        ],
        outside_timer: vec![
            "backend construction",
            "operand upload",
            "correctness check (download)",
        ],
        extra: json!({"n": n}),
    })
}

fn main() -> Result<()> {
    let case_id = arg("--case").ok_or("--case is required")?;
    let params: Value = serde_json::from_str(&arg("--params").ok_or("--params is required")?)?;
    let device = arg_usize("--device", 0)?;
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
        "transfer" => transfer(&params, &timing, device),
        "alloc_zero" => alloc_zero(&params, &timing, device),
        "small_blocks" => small_blocks(&params, &timing, device),
        "batched_factorization" => batched_factorization(&params, &timing, device),
        "first_call_layouts" => first_call_layouts(&params, device),
        "raw_scope_entry" => raw_scope_entry(&params, &timing, device),
        "eager_op" => eager_op(&params, &timing, device),
        "gather_table" => gather_table(&params, &timing, device),
        "reshape_materialize" => reshape_materialize(&params, &timing, device),
        "vendor_handle" => vendor_handle(&params, &timing, device),
        "vendor_churn" => vendor_churn(&params, device),
        "pool_retention" => pool_retention(&params, device),
        "svd_driver" => svd_driver(&params, &timing, device),
        "eigh_driver" => eigh_driver(&params, &timing, device),
        "hadamard_dot_general" => hadamard_dot_general(&params, &timing, device),
        "contraction_plan_churn" => contraction_plan_churn(&params, &timing, device),
        "owned_destination" => owned_destination(&params, &timing, device),
        "borrowed_input_gemm" => borrowed_input_gemm(&params, &timing, device),
        _ => Err(format!("unknown case kind {kind}").into()),
    }?;
    let mut output = json!({
        "suite_id": "gpu/perf_issues",
        "case_id": case_id,
        "correctness_status": "passed",
        "max_rel_error": outcome.max_rel_error,
        "samples": Vec::<Sample>::new(),
        "scope": {"timer": outcome.timer, "outside_timer": outcome.outside_timer},
        "device_ordinal": device,
    });
    if let Value::Object(fields) = outcome.extra {
        for (key, value) in fields {
            output[key] = value;
        }
    }
    if let Some(m) = outcome.measurement {
        output["samples"] = serde_json::to_value(&m.samples)?;
        output["calibration"] = json!({"target_ns": timing.target_ns, "iterations": m.iterations});
    }
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
