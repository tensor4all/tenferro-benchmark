//! Stateful compact QR public-boundary workflow (#94).
use std::{error::Error, hint::black_box, time::Instant};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::TensorEinsumExt;
use tenferro_linalg::{EagerSessionLinalgExt, HouseholderQr, QrOptions, TensorLinalgExt};
use tenferro_runtime::TensorSessionOpsExt;
use tenferro_tensor::{Engine, Tensor, TensorRead};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
const BATCH: usize = 64;
const APPENDS: usize = 9;
enum Dispatch<'a> {
    Shared(&'a mut dyn Engine),
    Fresh(&'a mut CpuEngine),
}
impl Dispatch<'_> {
    fn call<O: Send>(&mut self, f: impl FnOnce(&mut dyn Engine) -> Result<O> + Send) -> Result<O> {
        match self {
            Self::Shared(s) => f(&mut **s),
            Self::Fresh(b) => f(&mut **b),
        }
    }
}
struct Output<T> {
    states: Vec<HouseholderQr<T>>,
    r: Vec<T>,
    q: Vec<T>,
}
impl<T> Output<T> {
    fn new() -> Self {
        Self {
            states: Vec::with_capacity(APPENDS),
            r: Vec::with_capacity(APPENDS),
            q: Vec::with_capacity(APPENDS),
        }
    }
}
fn timed<O>(attributed: bool, component: &mut u128, f: impl FnOnce() -> Result<O>) -> Result<O> {
    if attributed {
        let start = Instant::now();
        let out = f()?;
        *component += start.elapsed().as_nanos();
        Ok(out)
    } else {
        f()
    }
}
fn sequence(
    initial: &HouseholderQr<Tensor>,
    blocks: &[Tensor],
    out: &mut Output<Tensor>,
    dispatch: &mut Dispatch<'_>,
    attributed: bool,
) -> Result<[u128; 3]> {
    let mut sums = [0; 3];
    for (i, b) in blocks.iter().enumerate() {
        let prior = out.states.last().unwrap_or(initial);
        let next = timed(attributed, &mut sums[0], || {
            dispatch.call(|s| Ok(prior.append_columns(b, s)?))
        })?;
        out.states.push(next);
        let next = out.states.last().ok_or("missing state")?;
        let r = timed(attributed, &mut sums[1], || {
            dispatch.call(|s| Ok(next.r(QrOptions::default(), s)?))
        })?;
        let start = 5 + 3 * i;
        let q = timed(attributed, &mut sums[2], || {
            dispatch.call(|s| Ok(next.q_columns(start..start + 3, QrOptions::default(), s)?))
        })?;
        out.r.push(r);
        out.q.push(q);
    }
    Ok(sums)
}
fn eager_sequence(
    initial: &HouseholderQr<EagerTensor>,
    blocks: &[EagerTensor],
    out: &mut Output<EagerTensor>,
    runtime: &std::sync::Arc<EagerRuntime>,
    shared: Option<&mut tenferro_ad::EagerSession<'_>>,
    engine: Option<&mut CpuEngine>,
) -> Result<()> {
    match (shared, engine) {
        (Some(s), _) => {
            for (i, b) in blocks.iter().enumerate() {
                let next = out.states.last().unwrap_or(initial).append_columns(b, s)?;
                out.states.push(next);
                let state = out.states.last().ok_or("missing state")?;
                out.r.push(state.r(QrOptions::default(), s)?);
                out.q
                    .push(state.q_columns(5 + 3 * i..8 + 3 * i, QrOptions::default(), s)?);
            }
        }
        (None, Some(engine)) => {
            for (i, b) in blocks.iter().enumerate() {
                let next = runtime.with_eager_session(engine, |s| {
                    out.states.last().unwrap_or(initial).append_columns(b, s)
                })?;
                out.states.push(next);
                let state = out.states.last().ok_or("missing state")?;
                out.r.push(
                    runtime.with_eager_session(engine, |s| state.r(QrOptions::default(), s))?,
                );
                out.q.push(runtime.with_eager_session(engine, |s| {
                    state.q_columns(5 + 3 * i..8 + 3 * i, QrOptions::default(), s)
                })?);
            }
        }
        (None, None) => return Err("eager sequence needs an engine or an entered session".into()),
    }
    Ok(())
}
fn values(m: usize, n: usize, start: usize) -> Vec<f64> {
    (0..m * n)
        .map(|p| {
            let (i, j) = (p % m, p / m + start);
            if i == j {
                2.5 + j as f64 / 32.
            } else {
                ((i * 17 + j * 13) % 97) as f64 / 256. - 0.15
            }
        })
        .collect()
}
fn validation(m: usize, a: &[f64], q: &[f64], r: &[f64]) -> Result<(f64, f64)> {
    if a.len() != m * 32
        || q.len() != m * 32
        || r.len() != 32 * 32
        || a.iter().chain(q).chain(r).any(|v| !v.is_finite())
    {
        return Err("wrong shape or nonfinite QR validation input".into());
    }
    let mut residual: f64 = 0.;
    let mut ortho: f64 = 0.;
    for j in 0..32 {
        for i in 0..m {
            let product = (0..32).map(|k| q[i + m * k] * r[k + 32 * j]).sum::<f64>();
            residual = residual.max((product - a[i + m * j]).abs() / (1. + a[i + m * j].abs()));
        }
    }
    for j in 0..32 {
        for i in 0..32 {
            let dot = (0..m).map(|k| q[k + m * i] * q[k + m * j]).sum::<f64>();
            ortho = ortho.max((dot - if i == j { 1. } else { 0. }).abs());
        }
    }
    if residual > 1e-10 || ortho > 1e-10 {
        return Err(format!("QR validation residual={residual} orthogonality={ortho}").into());
    }
    Ok((residual, ortho))
}
/// Shape and f64 slice of a retained QR factor: an owned tensor on the direct
/// paths, or the eager value view when the engine is borrowed by an entered
/// eager session (validation only; it never changes the measured scope).
trait CheckedTensor {
    fn shape(&self) -> &[usize];
    fn as_f64(&self) -> Result<&[f64]>;
}

impl CheckedTensor for Tensor {
    fn shape(&self) -> &[usize] {
        Tensor::shape(self)
    }
    fn as_f64(&self) -> Result<&[f64]> {
        Ok(self.as_slice::<f64>()?)
    }
}

impl CheckedTensor for tenferro_ad::ValueGuard<'_> {
    fn shape(&self) -> &[usize] {
        tenferro_ad::ValueGuard::shape(self)
    }
    fn as_f64(&self) -> Result<&[f64]> {
        Ok(self.as_slice::<f64>()?)
    }
}

fn check_output<V: CheckedTensor>(
    m: usize,
    initial_q: &Tensor,
    output: &Output<V>,
    full: &[f64],
) -> Result<(f64, f64)> {
    if output.r.len() != 9 || output.q.len() != 9 {
        return Err("missing interleaved outputs".into());
    }
    let mut q = initial_q.as_slice::<f64>()?.to_vec();
    for (step, t) in output.q.iter().enumerate() {
        if t.shape() != [m, 3] || output.r[step].shape() != [8 + 3 * step, 8 + 3 * step] {
            return Err("wrong output shape".into());
        }
        q.extend_from_slice(t.as_f64()?);
        // Every step's R and accumulated selected Q must reconstruct that prefix.
        let rank = 8 + 3 * step;
        let r = output.r[step].as_f64()?;
        if q.iter().chain(r).any(|v| !v.is_finite()) {
            return Err("nonfinite interleaved output".into());
        }
        for j in 0..rank {
            for i in 0..m {
                let value = (0..rank)
                    .map(|k| q[i + m * k] * r[k + rank * j])
                    .sum::<f64>();
                if (value - full[i + m * j]).abs() > 1e-9 {
                    return Err("intermediate QR reconstruction failed".into());
                }
            }
        }
    }
    validation(m, full, &q, output.r.last().ok_or("missing R")?.as_f64()?)
}
// Explicit-Q BCGS2, using public einsum, QR, arithmetic and concatenation.
// All intermediate tensors are retained through the batch clock stop.
struct BcgsOutput {
    tensors: Vec<Tensor>,
    q: usize,
    r: usize,
    append_ns: u128,
}
fn bcgs(
    m: usize,
    blocks: &[Tensor],
    zeros: &[Tensor],
    out: &mut BcgsOutput,
    s: &mut dyn Engine,
    attributed: bool,
) -> Result<()> {
    for (i, b) in blocks.iter().enumerate() {
        let start = if attributed {
            Some(Instant::now())
        } else {
            None
        };
        let q = &out.tensors[out.q];
        let r = &out.tensors[out.r];
        let h1 = [q, b].einsum("ji,jk->ik", s)?;
        let p1 = [q, &h1].einsum("ij,jk->ik", s)?;
        let y1 = b.sub(&p1, s)?;
        let h2 = [q, &y1].einsum("ji,jk->ik", s)?;
        let p2 = [q, &h2].einsum("ij,jk->ik", s)?;
        let y2 = y1.sub(&p2, s)?;
        let (qb, rb) = y2.qr(s)?;
        let h = h1.add(&h2, s)?;
        let newq = Tensor::concatenate(&[q, &qb], 1, s)?;
        let top = Tensor::concatenate(&[r, &h], 1, s)?;
        let bottom = Tensor::concatenate(&[&zeros[i], &rb], 1, s)?;
        let newr = Tensor::concatenate(&[&top, &bottom], 0, s)?;
        if let Some(start) = start {
            out.append_ns += start.elapsed().as_nanos();
        }
        // Explicitly materialize equivalent snapshot outputs, diagnostic only.
        let observed_r = s.to_contiguous_read(TensorRead::from_tensor(&newr))?;
        let observed_q = s.to_contiguous_read(TensorRead::from_tensor(&qb))?;
        out.tensors.extend([
            h1, p1, y1, h2, p2, y2, qb, rb, h, top, bottom, observed_r, observed_q,
        ]);
        out.q = out.tensors.len();
        out.tensors.push(newq);
        out.r = out.tensors.len();
        out.tensors.push(newr);
    }
    let _ = m;
    Ok(())
}
fn arg(name: &str, default: &str) -> String {
    let args: Vec<_> = std::env::args().collect();
    args.windows(2)
        .find(|p| p[0] == name)
        .map(|p| p[1].clone())
        .unwrap_or(default.into())
}
fn main() -> Result<()> {
    let threads: usize = arg("--threads", "1").parse()?;
    tenferro_einsum_benchmark::thread_enforcement::enforce_thread_request(threads)?;
    let m: usize = arg("--rows", "64").parse()?;
    let path = arg("--path", "shared");
    let samples: usize = arg("--samples", "3").parse()?;
    let warmups: usize = arg("--warmups", "3").parse()?;
    if !matches!(m, 64 | 128 | 256)
        || samples == 0
        || !matches!(
            path.as_str(),
            "shared"
                | "fresh"
                | "eager-fresh"
                | "eager-shared"
                | "attributed"
                | "bcgs2"
                | "bcgs2-attributed"
        )
    {
        return Err("invalid case".into());
    }
    let setup = Instant::now();
    let input = Tensor::from_vec_col_major([m, 5], values(m, 5, 0))?;
    let blocks: Vec<Tensor> = (0..9)
        .map(|i| Tensor::from_vec_col_major([m, 3], values(m, 3, 5 + 3 * i)))
        .collect::<std::result::Result<_, _>>()?;
    let full = values(m, 32, 0);
    let mut backend = tenferro_einsum_benchmark::cpu_engine(None)?;
    tenferro_einsum_benchmark::thread_enforcement::verify_backend_threads(
        "CpuBackend",
        backend.num_threads(),
        threads,
    )?;
    let (initial, initial_q, initial_r) = {
        let s: &mut dyn Engine = &mut backend;
        let state = input.householder_qr(s)?;
        let q = state.q_columns(0..5, QrOptions::default(), s)?;
        let r = state.r(QrOptions::default(), s)?;
        (state, q, r)
    };
    let zeros: Vec<Tensor> = (0..9)
        .map(|i| Tensor::from_vec_col_major([3, 5 + 3 * i], vec![0.; 3 * (5 + 3 * i)]))
        .collect::<std::result::Result<_, _>>()?;
    let mut raw = Vec::with_capacity(samples);
    let mut attributed_raw = Vec::with_capacity(samples);
    let mut worst = (0f64, 0f64);
    let mut initialization_ms = setup.elapsed().as_secs_f64() * 1000.;
    let mut preparation_ns = 0u128;
    let mut validation_ns = 0u128;
    let mut warmup_ns = 0u128;
    let mut execute = |dispatch: &mut Dispatch<'_>| -> Result<()> {
        for pass in 0..1 + warmups + samples {
            // Per-workflow output containers and all input restoration are untimed.
            let preparation_clock = Instant::now();
            let mut outputs: Vec<_> = (0..BATCH).map(|_| Output::new()).collect();
            let mut parts = [0; 3];
            preparation_ns += preparation_clock.elapsed().as_nanos();
            let clock = Instant::now();
            for out in &mut outputs {
                let totals = sequence(&initial, &blocks, out, dispatch, path == "attributed")?;
                for (i, v) in totals.iter().enumerate() {
                    parts[i] += v;
                }
            }
            let elapsed = clock.elapsed().as_nanos();
            black_box(&outputs);
            if pass <= warmups {
                warmup_ns += elapsed;
            }
            let validation_clock = Instant::now();
            for out in &outputs {
                let (r, o) = check_output(m, &initial_q, out, &full)?;
                worst.0 = worst.0.max(r);
                worst.1 = worst.1.max(o);
            }
            validation_ns += validation_clock.elapsed().as_nanos();
            if pass > warmups {
                raw.push(elapsed);
                attributed_raw.push(parts);
            }
        }
        Ok(())
    };
    match path.as_str() {
        "shared" | "attributed" => execute(&mut Dispatch::Shared(&mut backend))?,
        "fresh" => execute(&mut Dispatch::Fresh(&mut backend))?,
        "bcgs2" | "bcgs2-attributed" => {
            let s: &mut dyn Engine = &mut backend;
            (|| -> Result<()> {
                for pass in 0..1 + warmups + samples {
                    let preparation_clock = Instant::now();
                    let mut outputs: Vec<_> = (0..BATCH)
                        .map(|_| {
                            Ok(BcgsOutput {
                                tensors: {
                                    let mut v = Vec::with_capacity(2 + 9 * 15);
                                    v.push(Tensor::from_vec_col_major(
                                        [m, 5],
                                        initial_q.as_slice::<f64>()?.to_vec(),
                                    )?);
                                    v.push(Tensor::from_vec_col_major(
                                        [5, 5],
                                        initial_r.as_slice::<f64>()?.to_vec(),
                                    )?);
                                    v
                                },
                                q: 0,
                                r: 1,
                                append_ns: 0,
                            })
                        })
                        .collect::<Result<_>>()?;
                    preparation_ns += preparation_clock.elapsed().as_nanos();
                    let clock = Instant::now();
                    for out in &mut outputs {
                        bcgs(m, &blocks, &zeros, out, s, path == "bcgs2-attributed")?;
                    }
                    let elapsed = clock.elapsed().as_nanos();
                    black_box(&outputs);
                    if pass <= warmups {
                        warmup_ns += elapsed;
                    }
                    let validation_clock = Instant::now();
                    for out in &outputs {
                        let copy = |t: &Tensor| -> Result<Tensor> {
                            Ok(Tensor::from_vec_col_major(
                                t.shape().to_vec(),
                                t.as_slice::<f64>()?.to_vec(),
                            )?)
                        };
                        let observed = Output {
                            states: vec![],
                            r: (0..9)
                                .map(|i| copy(&out.tensors[2 + 15 * i + 11]))
                                .collect::<Result<_>>()?,
                            q: (0..9)
                                .map(|i| copy(&out.tensors[2 + 15 * i + 12]))
                                .collect::<Result<_>>()?,
                        };
                        let (prefix_r, prefix_o) = check_output(m, &initial_q, &observed, &full)?;
                        worst.0 = worst.0.max(prefix_r);
                        worst.1 = worst.1.max(prefix_o);
                        let (r, o) = validation(
                            m,
                            &full,
                            out.tensors[out.q].as_slice::<f64>()?,
                            out.tensors[out.r].as_slice::<f64>()?,
                        )?;
                        worst.0 = worst.0.max(r);
                        worst.1 = worst.1.max(o);
                    }
                    validation_ns += validation_clock.elapsed().as_nanos();
                    if pass > warmups {
                        raw.push(elapsed);
                        attributed_raw.push([outputs.iter().map(|o| o.append_ns).sum(), 0, 0]);
                    }
                }
                Ok(())
            })()?
        }
        "eager-fresh" | "eager-shared" => {
            let runtime = EagerRuntime::new()?;
            let _guard = runtime.no_grad();
            let initial_input = EagerTensor::from_tensor_in(input, runtime.clone())?;
            if initial_input.tracks_grad() {
                return Err("tracked initial fixture".into());
            }
            let eager_initial =
                runtime.with_eager_session(&mut backend, |s| s.householder_qr(&initial_input))?;
            let eager_blocks: Vec<_> = blocks
                .into_iter()
                .map(|b| EagerTensor::from_tensor_in(b, runtime.clone()))
                .collect::<std::result::Result<_, _>>()?;
            if eager_blocks.iter().any(EagerTensor::tracks_grad) {
                return Err("tracked fixture".into());
            }
            initialization_ms = setup.elapsed().as_secs_f64() * 1000.;
            let mut batches = |mut shared: Option<&mut tenferro_ad::EagerSession<'_>>,
                               mut engine: Option<&mut CpuEngine>|
             -> Result<()> {
                for pass in 0..1 + warmups + samples {
                    let preparation_clock = Instant::now();
                    let mut outputs: Vec<_> = (0..BATCH).map(|_| Output::new()).collect();
                    preparation_ns += preparation_clock.elapsed().as_nanos();
                    let clock = Instant::now();
                    for out in &mut outputs {
                        eager_sequence(
                            &eager_initial,
                            &eager_blocks,
                            out,
                            &runtime,
                            shared.as_deref_mut(),
                            engine.as_deref_mut(),
                        )?;
                    }
                    let elapsed = clock.elapsed().as_nanos();
                    black_box(&outputs);
                    if pass <= warmups {
                        warmup_ns += elapsed;
                    }
                    let validation_clock = Instant::now();
                    for out in &outputs {
                        if out.q.iter().chain(&out.r).any(EagerTensor::tracks_grad) {
                            return Err("tracked result".into());
                        }
                        // The engine may be borrowed by the entered session,
                        // so validation reads the retained value view instead
                        // of materializing a second copy.
                        let concrete = Output {
                            states: vec![],
                            q: out
                                .q
                                .iter()
                                .map(|t| t.value())
                                .collect::<std::result::Result<_, _>>()?,
                            r: out
                                .r
                                .iter()
                                .map(|t| t.value())
                                .collect::<std::result::Result<_, _>>()?,
                        };
                        let (r, o) = check_output(m, &initial_q, &concrete, &full)?;
                        worst.0 = worst.0.max(r);
                        worst.1 = worst.1.max(o);
                    }
                    validation_ns += validation_clock.elapsed().as_nanos();
                    if pass > warmups {
                        raw.push(elapsed);
                        attributed_raw.push([0; 3]);
                    }
                }
                Ok(())
            };
            if path == "eager-shared" {
                runtime.with_eager_session(&mut backend, |s| batches(Some(s), None))?;
            } else {
                batches(None, Some(&mut backend))?;
            }
        }
        _ => unreachable!(),
    }
    println!(
        "{}",
        serde_json::json!({"suite_id":"cpu/stateful_linalg","case_id":format!("incremental_qr_m{m}_{path}"),"rows":m,"path":path,"dtype":"f64","initial_rank":5,"append_width":3,"appends":9,"final_rank":32,"operations_per_workflow":27,"workflows_per_sample":BATCH,"batch_retention_memory_policy":"64 workflows; all state, snapshot and BCGS2 intermediates retained; estimated below 192 MiB","warmups":warmups,"initialization_passes":1,"samples_ns":raw,"component_samples_ns":attributed_raw,"correctness":"passed","max_reconstruction_error":worst.0,"max_orthogonality_error":worst.1,"provider":tenferro_einsum_benchmark::compiled_cpu_provider(),"initialization_ms":initialization_ms,"phase_wall_ms":{"input_backend_initial_factor_setup":initialization_ms,"retention_preparation":preparation_ns as f64/1e6,"validation_and_output_materialization":validation_ns as f64/1e6,"initialization_and_warmup_execution":warmup_ns as f64/1e6,"measured_execution":raw.iter().sum::<u128>() as f64/1e6,"total_before_report":setup.elapsed().as_secs_f64()*1000.},"scope":{"timer":"nine interleaved append_columns/R/new Q-columns; intrinsic output allocation; fresh paths include per-operation session entry/exit diagnostic; attributed paths include per-call clocks and are diagnostic","outside_timer":["input generation","initial compact factorization","runtime/backend construction","output retention containers","shared session entry/exit","validation/materialization","destruction"]},"cache_policy":"explicit initialization pass plus three warmups; reusable runtime/backend; no capture_trace; no_grad","affinity":"none","requested_threads":threads,"worker_count":threads})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validation_rejects_nonfinite_wrong_shape_and_nonorthogonal_factors() {
        let m = 64;
        let mut q = vec![0.; m * 32];
        let mut r = vec![0.; 32 * 32];
        for i in 0..32 {
            q[i + m * i] = 1.;
            r[i + 32 * i] = 1.;
        }
        let a = q.clone();
        assert!(validation(m, &a, &q, &r).is_ok());
        q[0] = f64::NAN;
        assert!(validation(m, &a, &q, &r).is_err());
        assert!(validation(m, &a, &q[..1], &r).is_err());
        q[0] = 2.;
        assert!(validation(m, &a, &q, &r).is_err());
    }
}
