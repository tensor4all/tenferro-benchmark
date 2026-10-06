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
use serde::Serialize;
use serde_json::{json, Value};
use tenferro_gpu::cuda::{
    download_tensor, upload_tensor, with_cuda_exec_session, CudaBackend, CudaDeviceId,
};
use tenferro_linalg::TensorLinalgExt;
use tenferro_tensor::{
    BackendSession, BackendSessionHost, DotGeneralConfig, Tensor, TensorRead,
    TensorViewCanonicalization, TensorWrite,
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

fn max_rel(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs() / e.abs().max(1.0))
        .fold(0.0, f64::max)
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
