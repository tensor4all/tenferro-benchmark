//! Public CUDA operations, with one entered session spanning all samples.
//! Fixtures and independent reference outputs are prepared by the Python runner.
use num_complex::{Complex32, Complex64};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::HashMap, fs, hint::black_box, io::Write, path::Path, time::Instant};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_fft::{EagerSessionFftExt, FftExecutor, FftNorm};
use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaBackend, CudaDeviceId, CudaRuntime};
use tenferro_tensor::{BackendSession, BackendSessionHost, DType, Tensor};

#[path = "benchmark_gpu_public_api/linalg.rs"]
mod linalg;
#[path = "benchmark_gpu_public_api/multi.rs"]
mod multi;
#[path = "benchmark_gpu_public_api/reuse.rs"]
mod reuse;
#[path = "benchmark_gpu_public_api/runtime_ops.rs"]
mod runtime_ops;
#[path = "benchmark_gpu_public_api/state.rs"]
mod state;
#[path = "benchmark_gpu_public_api/structural.rs"]
mod structural;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Deserialize)]
struct Array {
    shape: Vec<usize>,
    dtype: String,
    file: String,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    op: String,
    n: usize,
    inputs: Vec<Array>,
    outputs: Vec<Array>,
    rtol: f64,
    atol: f64,
    reference: String,
    #[serde(default)]
    full_reduction: bool,
    #[serde(default)]
    compare_direction: String,
    #[serde(default)]
    output_shape: Vec<usize>,
    #[serde(default)]
    broadcast_dims: Vec<usize>,
    #[serde(default)]
    slice_sizes: Vec<usize>,
    #[serde(default)]
    gather_axis: usize,
    #[serde(default)]
    column_range: Vec<usize>,
    #[serde(default)]
    excluded: HashMap<String, String>,
    #[serde(default)]
    unavailable: HashMap<String, String>,
}
struct Settings {
    verify_only: bool,
    warmups: usize,
    runs: usize,
    target_ns: u128,
}

fn arg(name: &str) -> Result<String> {
    let args: Vec<_> = std::env::args().collect();
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].clone())
        .ok_or_else(|| format!("missing {name}").into())
}
fn env_usize(name: &str, default: usize) -> Result<usize> {
    Ok(std::env::var(name)
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(default))
}
fn read_array(root: &Path, a: &Array) -> Result<Tensor> {
    let bytes = fs::read(root.join(&a.file))?;
    macro_rules! real {
        ($ty:ty, $width:expr) => {{
            if bytes.len() != a.shape.iter().product::<usize>() * $width {
                return Err("fixture length mismatch".into());
            }
            Tensor::from_vec_col_major(
                a.shape.clone(),
                bytes
                    .chunks_exact($width)
                    .map(|b| <$ty>::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            )?
        }};
    }
    macro_rules! complex {
        ($ty:ty, $width:expr, $ct:ty) => {{
            if bytes.len() != a.shape.iter().product::<usize>() * $width * 2 {
                return Err("fixture length mismatch".into());
            }
            Tensor::from_vec_col_major(
                a.shape.clone(),
                bytes
                    .chunks_exact($width * 2)
                    .map(|b| {
                        <$ct>::new(
                            <$ty>::from_le_bytes(b[..$width].try_into().unwrap()),
                            <$ty>::from_le_bytes(b[$width..].try_into().unwrap()),
                        )
                    })
                    .collect::<Vec<_>>(),
            )?
        }};
    }
    Ok(match a.dtype.as_str() {
        "bool" => Tensor::from_vec_col_major(
            a.shape.clone(),
            bytes.iter().map(|&x| x != 0).collect::<Vec<_>>(),
        )?,
        "i64" => real!(i64, 8),
        "f32" => real!(f32, 4),
        "f64" => real!(f64, 8),
        "c32" => complex!(f32, 4, Complex32),
        "c64" => complex!(f64, 8, Complex64),
        _ => return Err(format!("unknown dtype {}", a.dtype).into()),
    })
}
fn numbers(t: &Tensor) -> Result<Vec<Complex64>> {
    Ok(match t.dtype() {
        DType::Bool => t
            .as_slice::<bool>()?
            .iter()
            .map(|&x| Complex64::new(if x { 1. } else { 0. }, 0.))
            .collect(),
        DType::F32 => t
            .as_slice::<f32>()?
            .iter()
            .map(|&x| Complex64::new(x as f64, 0.))
            .collect(),
        DType::F64 => t
            .as_slice::<f64>()?
            .iter()
            .map(|&x| Complex64::new(x, 0.))
            .collect(),
        DType::C32 => t
            .as_slice::<Complex32>()?
            .iter()
            .map(|x| Complex64::new(x.re as f64, x.im as f64))
            .collect(),
        DType::C64 => t.as_slice::<Complex64>()?.to_vec(),
        _ => return Err("unsupported verification dtype".into()),
    })
}
fn verify(rt: &CudaRuntime, actual: &Tensor, expected: &Tensor, case: &Case) -> Result<f64> {
    let device = actual
        .placement()
        .device
        .as_ref()
        .ok_or("CPU fallback: missing CUDA placement")?;
    if device.kind != tenferro_tensor::DeviceKind::Gpu(tenferro_tensor::GpuBackendKind::Cuda)
        || device.ordinal != env_usize("GPU_BENCH_DEVICE", 0)?
        || !actual.is_backend_buffer()
    {
        return Err("CPU fallback or incorrect CUDA allocation placement".into());
    }
    if actual.shape() != expected.shape() || actual.dtype() != expected.dtype() {
        return Err(format!(
            "verification shape/dtype mismatch: {:?}/{:?} vs {:?}/{:?}",
            actual.shape(),
            actual.dtype(),
            expected.shape(),
            expected.dtype()
        )
        .into());
    }
    let host = download_tensor(rt, actual)?;
    let mut worst = 0.0_f64;
    for (x, y) in numbers(&host)?.iter().zip(numbers(expected)?) {
        if *x == y {
            continue;
        }
        let error = (*x - y).norm() / (case.atol + case.rtol * y.norm());
        if !error.is_finite() {
            return Err("verification nonfinite output".into());
        }
        worst = worst.max(error);
    }
    if worst > 1. {
        return Err(format!("verification error / tolerance = {worst}").into());
    }
    Ok(worst)
}

/// Both calibration and measurement retain outputs until the clock stops.
/// The callback captures a session entered before invoking this function.
fn measure<O, A>(
    settings: &Settings,
    bytes: usize,
    rt: &CudaRuntime,
    mut prepare: impl FnMut() -> A,
    mut op: impl FnMut(A) -> Result<O>,
) -> Result<Value> {
    let batch = |count: usize| -> Result<u128> {
        let mut outputs = Vec::with_capacity(count);
        let mut arguments = (0..count).map(|_| Some(prepare())).collect::<Vec<_>>();
        rt.synchronize()?;
        let start = Instant::now();
        for args in &mut arguments {
            outputs.push(op(args.take().unwrap())?);
        }
        rt.synchronize()?;
        let elapsed = start.elapsed().as_nanos();
        black_box(&outputs);
        Ok(elapsed)
    };
    sample_batches(settings, bytes, batch)
}
fn sample_batches(
    settings: &Settings,
    bytes: usize,
    mut batch: impl FnMut(usize) -> Result<u128>,
) -> Result<Value> {
    // Explicit initialization is mandatory even for BENCH_WARMUPS=0.
    let first_ns = batch(1)?;
    for _ in 0..settings.warmups {
        batch(1)?;
    }
    let cap = ((256 << 20) / bytes.max(1)).clamp(1, 4096);
    let mut count = 1;
    while batch(count)? < settings.target_ns && count < cap {
        count = (count * 2).min(cap);
    }
    let mut samples = Vec::with_capacity(settings.runs);
    for _ in 0..settings.runs {
        samples.push(batch(count)? as f64 / 1e6);
    }
    Ok(json!({"operations_per_sample":count,"batch_ms":samples,
        "first_run_ms": first_ns as f64/1e6, "retention_budget_bytes":256<<20,
        "target_batch_ms":settings.target_ns as f64/1e6,"warmups":settings.warmups,
        "scope":"steady_state_host_api_plus_device_sync"}))
}
fn direct(
    case: &Case,
    inputs: &[Tensor],
    executor: &mut FftExecutor,
    session: &mut dyn BackendSession,
    args: structural::Arguments<'_>,
) -> Result<Tensor> {
    let input = &inputs[0];
    Ok(match case.op.as_str() {
        "fft" => executor.fft(input, None, -1, FftNorm::Backward, session)?,
        "ifft" => executor.ifft(input, None, -1, FftNorm::Backward, session)?,
        "rfft" => executor.rfft(input, None, -1, FftNorm::Backward, session)?,
        "irfft" => executor.irfft(input, Some(case.n), -1, FftNorm::Backward, session)?,
        _ => return runtime_ops::direct(case, inputs, session, args),
    })
}
fn eager(
    case: &Case,
    inputs: &[EagerTensor],
    session: &mut tenferro_ad::EagerSession<'_>,
    args: structural::Arguments<'_>,
) -> Result<EagerTensor> {
    let input = &inputs[0];
    Ok(match case.op.as_str() {
        "fft" => session.fft(input, None, -1, FftNorm::Backward)?,
        "ifft" => session.ifft(input, None, -1, FftNorm::Backward)?,
        "rfft" => session.rfft(input, None, -1, FftNorm::Backward)?,
        "irfft" => session.irfft(input, Some(case.n), -1, FftNorm::Backward)?,
        _ => return runtime_ops::eager(case, inputs, session, args),
    })
}
fn run(
    root: &Path,
    case: &Case,
    mode: &str,
    settings: &Settings,
    device: usize,
) -> Result<(Option<Value>, f64)> {
    if matches!(
        case.op.as_str(),
        "householder_from_factors"
            | "householder_append"
            | "householder_r"
            | "householder_q_columns"
    ) {
        return state::run(root, case, mode, settings, device);
    }
    if matches!(case.op.as_str(), "svd_full" | "slogdet" | "householder_qr") {
        return multi::run(root, case, mode, settings, device);
    }
    let input_host = case
        .inputs
        .iter()
        .map(|a| read_array(root, a))
        .collect::<Result<Vec<_>>>()?;
    let expected = read_array(root, &case.outputs[0])?;
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(device.try_into()?))?;
    let rt = backend.runtime().clone();
    let input = input_host
        .iter()
        .map(|a| Ok(upload_tensor(&rt, a)?))
        .collect::<Result<Vec<_>>>()?;
    let bytes = expected.shape().iter().product::<usize>()
        * match expected.dtype() {
            DType::C64 => 16,
            DType::C32 | DType::F64 => 8,
            _ => 4,
        };
    if mode == "tenferro-cuda-direct" {
        let mut executor = FftExecutor::default();
        backend.with_backend_session(|session| {
            if case.op.contains("_into") {
                return reuse::run(case, &input, &expected, &rt, session, settings, bytes);
            }
            let probe = direct(
                case,
                &input,
                &mut executor,
                session,
                structural::prepare_direct(case, &input),
            )?;
            let error = verify(&rt, &probe, &expected, case)?;
            drop(probe);
            if settings.verify_only || case.excluded.contains_key(mode) {
                return Ok((None, error));
            }
            let timing = measure(
                settings,
                bytes,
                &rt,
                || structural::prepare_direct(case, &input),
                |args| direct(case, &input, &mut executor, session, args),
            )?;
            let final_output = direct(
                case,
                &input,
                &mut executor,
                session,
                structural::prepare_direct(case, &input),
            )?;
            let error = error.max(verify(&rt, &final_output, &expected, case)?);
            Ok((Some(timing), error))
        })?
    } else {
        let runtime = EagerRuntime::with_cuda_backend(backend)?;
        let _guard = runtime.no_grad();
        runtime.with_eager_session(|session| {
            let input = input
                .into_iter()
                .map(|a| Ok(session.constant_from(a)?))
                .collect::<Result<Vec<_>>>()?;
            let probe = eager(case, &input, session, structural::prepare(case))?;
            let error = verify(
                &rt,
                probe
                    .tensor_read()
                    .as_tensor()
                    .ok_or("eager output is not materialized")?,
                &expected,
                case,
            )?;
            drop(probe);
            if settings.verify_only || case.excluded.contains_key(mode) {
                return Ok((None, error));
            }
            let timing = measure(
                settings,
                bytes,
                &rt,
                || structural::prepare(case),
                |args| eager(case, &input, session, args),
            )?;
            let final_output = eager(case, &input, session, structural::prepare(case))?;
            let error = error.max(verify(
                &rt,
                final_output
                    .tensor_read()
                    .as_tensor()
                    .ok_or("eager output is not materialized")?,
                &expected,
                case,
            )?);
            Ok((Some(timing), error))
        })
    }
}
fn main() -> Result<()> {
    let manifest = arg("--fixtures")?;
    let root = Path::new(&manifest)
        .parent()
        .ok_or("fixture parent missing")?;
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(&manifest)?)?;
    let output = arg("--output")?;
    let mode = arg("--backend")?;
    if !["tenferro-cuda-direct", "tenferro-cuda-eager"].contains(&mode.as_str()) {
        return Err("invalid backend".into());
    }
    let settings = Settings {
        verify_only: env_usize("GPU_API_VERIFY_ONLY", 0)? == 1,
        warmups: env_usize("BENCH_WARMUPS", 3)?,
        runs: env_usize("BENCH_RUNS", 15)?,
        target_ns: 2_000_000,
    };
    if settings.runs == 0 {
        return Err("BENCH_RUNS must be positive".into());
    }
    let device = env_usize("GPU_BENCH_DEVICE", 0)?;
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    for case in cases {
        eprintln!("{} {}", mode, case.id);
        let mut row = json!({"schema_version":1,"case_id":case.id,"op":case.op,"backend":mode,
            "gpu_role":std::env::var("GPU_API_ROLE").unwrap_or_else(|_| "benchmark".into()),
            "physical_device_index":env_usize("GPU_BENCH_PHYSICAL_DEVICE", device)?,
            "device_uuid":std::env::var("GPU_BENCH_DEVICE_UUID").ok(),
            "device_ordinal":device,"session_policy":"one entered session for initialization, calibration and samples",
            "setup_policy":"fixtures, H2D, handles, plan initialization and output validation outside timing",
            "synchronization":"CudaRuntime::synchronize; no D2H in timed region"});
        if case.op.contains("_into_accum") {
            row["input_restoration"] = json!("GPU seed copied into each destination before every batch; synchronized outside timing");
            row["accumulation"] = json!({"alpha":0.5,"beta":1.0});
        }
        if case.op == "dot_general_read_into" {
            row["accumulation"] = json!({"alpha":1.0,"beta":0.0});
        }
        if case.op == "dot_general_read_into_accum_cached" {
            row["cache_policy"] = json!("CUDA delegates to ordinary accumulation; session slot Some(0) is unused");
        }
        if let Some(reason) = case.unavailable.get(&mode) {
            row["status"] = json!("unsupported");
            row["reason"] = json!(reason);
            row["cuda_implementation_verified"] = json!(false);
            writeln!(file, "{}", row)?;
            continue;
        }
        match run(root, &case, &mode, &settings, device) {
            Ok((timing, error)) => {
                row["status"] = json!(if timing.is_some() {
                    "ok"
                } else if settings.verify_only && !case.excluded.contains_key(&mode) {
                    "verification_only"
                } else {
                    "unsupported"
                });
                row["timing"] = json!(timing);
                row["verification"] =
                    json!({"max_error_over_tolerance":error,"reference":case.reference});
                if let Some(reason) = case.excluded.get(&mode) {
                    row["reason"] = json!(reason);
                    row["cuda_implementation_verified"] = json!(true);
                }
            }
            Err(e) => {
                row["status"] = json!(if e.to_string().starts_with("verification") {
                    "verification_failed"
                } else {
                    "runtime_failed"
                });
                row["reason"] = json!(e.to_string());
            }
        }
        writeln!(file, "{}", row)?;
        file.flush()?;
    }
    Ok(())
}
