//! Public CUDA operations on one caller-held CUDA engine (tenferro-rs #2206).
//! Fixtures and independent reference outputs are prepared by the Python runner.
use num_complex::{Complex32, Complex64};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::HashMap, fs, hint::black_box, io::Write, path::Path, time::Instant};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_fft::{EagerSessionFftExt, FftExecutor, FftNorm};
use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaDeviceId, CudaEngine, CudaRuntime};
use tenferro_tensor::{DType, Engine, Tensor};

/// What this binary's `compile_time_ms` is, for a reader of a raw arm file.
///
/// The field stays present, and stays null: a Rust arm cannot time NVRTC on its
/// own, and the untimed case wall that surrounds a compile also holds that
/// case's fixture read, transfers and verification. `compile_modules` is the
/// compile diagnostic, and the untimed wall is published as a wall
/// (`overhead.untimed_ms`), never as a compile time.
const COMPILE_TIME_SCOPE: &str = "not measured: no compile interval is recorded; compile_modules counts the NVRTC modules this case compiled, and the case's untimed wall is published as overhead.untimed_ms, which also holds fixture read, transfers and verification";

/// Borrow the caller-held CUDA engine for one operation sequence.
///
/// tenferro-rs #2206 removed backend sessions: the engine the caller holds is
/// already the execution scope, so entering it is a plain borrow.
trait EngineScope {
    fn with_backend_session<R>(
        &mut self,
        f: impl FnOnce(&mut CudaEngine) -> R,
    ) -> tenferro_tensor::Result<R>;
}

impl EngineScope for CudaEngine {
    fn with_backend_session<R>(
        &mut self,
        f: impl FnOnce(&mut CudaEngine) -> R,
    ) -> tenferro_tensor::Result<R> {
        Ok(f(self))
    }
}

#[path = "benchmark_gpu_public_api/lifetime.rs"]
mod lifetime;
#[path = "benchmark_gpu_public_api/linalg.rs"]
mod linalg;
#[path = "benchmark_gpu_public_api/multi.rs"]
mod multi;
#[path = "benchmark_gpu_public_api/plan.rs"]
mod plan;
#[path = "benchmark_gpu_public_api/read_layout.rs"]
mod read_layout;
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
    normalization_weight: bool,
    #[serde(default)]
    normalization_bias: bool,
    #[serde(default)]
    padding: Vec<(i64, i64, i64)>,
    #[serde(default)]
    compare_direction: String,
    #[serde(default)]
    output_shape: Vec<usize>,
    #[serde(default)]
    axis: isize,
    #[serde(default)]
    broadcast_dims: Vec<usize>,
    #[serde(default)]
    slice_sizes: Vec<usize>,
    #[serde(default)]
    gather_axis: usize,
    #[serde(default)]
    column_range: Vec<usize>,
    #[serde(default)]
    read_op: String,
    #[serde(default)]
    alpha: Option<f64>,
    #[serde(default)]
    beta: Option<f64>,
    #[serde(default)]
    lhs_contracting_dims: Vec<usize>,
    #[serde(default)]
    rhs_contracting_dims: Vec<usize>,
    #[serde(default)]
    lhs_batch_dims: Vec<usize>,
    #[serde(default)]
    rhs_batch_dims: Vec<usize>,
    #[serde(default)]
    lhs_conj: bool,
    #[serde(default)]
    rhs_conj: bool,
    #[serde(default)]
    excluded: HashMap<String, String>,
    #[serde(default)]
    unavailable: HashMap<String, String>,
    /// Compulsory per operation traffic of the case, sized once by the fixture
    /// step from the suite shapes and dtype. It is a property of the case, not
    /// of the arm, so every arm copies the same block into its record.
    #[serde(default)]
    traffic: Option<Value>,
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
        "i32" => real!(i32, 4),
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
/// Total timed milliseconds of one row's samples, when the row sampled at all.
fn timed_ms_of(timing: Option<&Value>) -> Option<f64> {
    let samples = timing?.get("batch_ms")?.as_array()?;
    let total: f64 = samples.iter().filter_map(Value::as_f64).sum();
    (!samples.is_empty()).then_some(total)
}

/// The overhead line a case records when it spent more than `budget` of its timed
/// region outside it, or `None` when it did not.
///
/// The budget has no measured basis: it labels the untimed split for the report. It is
/// recorded in the row and printed, and nothing reads it back into a status or an exit
/// code, because a case that spends a long time setting up is still a measurement.
fn overhead_warning(
    case_id: &str,
    timed_ms: Option<f64>,
    case_wall_ms: f64,
    budget: f64,
) -> Option<String> {
    let timed = timed_ms.filter(|ms| *ms > 0.0)?;
    let untimed = (case_wall_ms - timed).max(0.0);
    (untimed > budget * timed).then(|| {
        format!(
            "{case_id} untimed {untimed:.1} ms is {:.0}% of the {timed:.1} ms timed region (budget {:.0}%)",
            100.0 * untimed / timed,
            100.0 * budget
        )
    })
}

/// The untimed work one case paid, as the harness measured it.
#[derive(Clone, Copy, Default)]
struct PhaseTimings {
    /// Reading the inputs and the reference (bytes and stored fingerprint).
    fixture_ms: f64,
    /// Verification, including the downloads it needs.
    verify_ms: f64,
    /// Fingerprint passes that ran and produced a diagnostic agreement answer.
    fingerprint_compares: usize,
    /// Of those, the passes whose invariants disagreed with the fixture.
    fingerprint_disagreements: usize,
    /// False for the families whose per-case path does not instrument the split.
    split_measured: bool,
}

/// The split of a path that verifies inside its own module: no phase numbers.
fn uninstrumented_phases() -> PhaseTimings {
    PhaseTimings::default()
}

/// Snapshot the phases measured so far for one case row.
fn phases_of(expected: &Expected<'_>, fixture_ms: f64) -> PhaseTimings {
    PhaseTimings {
        fixture_ms,
        verify_ms: expected.verify_ms,
        fingerprint_compares: expected.fingerprint_compares,
        fingerprint_disagreements: expected.fingerprint_disagreements,
        split_measured: true,
    }
}

/// One case's expected output: the stored fingerprint, and the reference bytes.
///
/// The fingerprint is a recorded diagnostic. The elementwise comparison against the
/// reference decides whether an output is correct, and runs for every verification.
struct Expected<'a> {
    case: &'a Case,
    output: usize,
    stored: Option<tenferro_einsum_benchmark::fingerprint::StoredFingerprint>,
    /// The reference values, read once when the case starts (inside the fixture window).
    tensor: Tensor,
    verify_ms: f64,
    /// Fingerprint passes that ran and produced a diagnostic agreement answer.
    fingerprint_compares: usize,
    /// Of those, the passes whose invariants disagreed with the stored ones.
    fingerprint_disagreements: usize,
}

impl<'a> Expected<'a> {
    /// Load the fingerprint document that sits next to the fixture.
    fn new(root: &'a Path, case: &'a Case, output: usize) -> Result<Self> {
        let path = root.join(
            tenferro_einsum_benchmark::fingerprint::StoredFingerprint::file_name(
                &case.outputs[output].file,
            ),
        );
        let stored = match fs::read(&path) {
            Ok(bytes) => {
                Some(tenferro_einsum_benchmark::fingerprint::StoredFingerprint::from_bytes(&bytes)?)
            }
            // A run against fixtures prepared before fingerprints existed keeps the
            // elementwise comparison instead of failing.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        Ok(Self {
            case,
            output,
            stored,
            tensor: read_array(root, &case.outputs[output])?,
            verify_ms: 0.0,
            fingerprint_compares: 0,
            fingerprint_disagreements: 0,
        })
    }

    /// The reference's declared shape and dtype.
    fn tensor_metadata(&self) -> (&[usize], DType) {
        let array = &self.case.outputs[self.output];
        (
            array.shape.as_slice(),
            match array.dtype.as_str() {
                "bool" => DType::Bool,
                "i32" => DType::I32,
                "i64" => DType::I64,
                "f32" => DType::F32,
                "f64" => DType::F64,
                "c32" => DType::C32,
                _ => DType::C64,
            },
        )
    }

    /// Verify `actual`, recording what the stored fingerprint says about it.
    fn verify(&mut self, rt: &CudaRuntime, actual: &Tensor) -> Result<f64> {
        let started = Instant::now();
        let outcome = self.verify_inner(rt, actual);
        self.verify_ms += started.elapsed().as_secs_f64() * 1e3;
        outcome
    }

    fn verify_inner(&mut self, rt: &CudaRuntime, actual: &Tensor) -> Result<f64> {
        // The fingerprint is a diagnostic and never the decision. An output can agree
        // with every stored invariant and still be elementwise wrong (one large element
        // pays the whole allowance of a sum), so the comparison below always runs.
        if let Some(stored) = self.stored.as_ref().and_then(|stored| {
            stored
                .fingerprint
                .as_ref()
                .map(|fingerprint| (stored, fingerprint))
        }) {
            let (stored, fingerprint) = stored;
            if !matches!(actual.dtype(), DType::Bool | DType::I32 | DType::I64)
                && actual.shape() == stored.shape.as_slice()
                && dtype_name(actual.dtype()) == stored.dtype
            {
                let host = download_tensor(rt, actual)?;
                let (real, imaginary) = real_and_imaginary(&host)?;
                let live =
                    tenferro_einsum_benchmark::fingerprint::Fingerprint::of(&real, &imaginary);
                self.fingerprint_compares += 1;
                if !fingerprint.matches(&live, self.case.atol, self.case.rtol) {
                    self.fingerprint_disagreements += 1;
                }
            }
        }
        let case = self.case;
        verify(rt, actual, &self.tensor, case)
    }
}

/// The harness name of a dtype, as the fixture manifest spells it.
fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::C32 => "c32",
        DType::C64 => "c64",
        _ => "unsupported",
    }
}

/// Split a host tensor into real and imaginary parts for the fingerprint pass.
fn real_and_imaginary(t: &Tensor) -> Result<(Vec<f64>, Vec<f64>)> {
    Ok(match t.dtype() {
        DType::F32 => (
            t.as_slice::<f32>()?.iter().map(|&x| x as f64).collect(),
            Vec::new(),
        ),
        DType::F64 => (t.as_slice::<f64>()?.to_vec(), Vec::new()),
        DType::C32 => {
            let values = t.as_slice::<Complex32>()?;
            (
                values.iter().map(|x| x.re as f64).collect(),
                values.iter().map(|x| x.im as f64).collect(),
            )
        }
        DType::C64 => {
            let values = t.as_slice::<Complex64>()?;
            (
                values.iter().map(|x| x.re).collect(),
                values.iter().map(|x| x.im).collect(),
            )
        }
        _ => return Err("fingerprint verification dtype".into()),
    })
}

/// `|miss| / (atol + rtol * |reference|)`, refused when the allowance cannot
/// score a miss at all.
///
/// An allowance that overflows from finite components (`atol + rtol * 1.7e308`)
/// divides every miss down to 0.0, so a wrong output reads as exact. An exact miss
/// over a zero allowance (`atol = 0`, an off-diagonal Gram entry) is 0 / 0 = NaN
/// and would drop the whole reduction. Both are decided here, before `max`.
fn scored(case: &Case, miss: f64, reference: f64) -> Result<f64> {
    // `f64::max` keeps the other operand when one side is NaN, so a NaN miss has
    // to be refused here or it disappears from `worst` without a trace.
    if miss.is_nan() {
        return Err("verification nonfinite output".into());
    }
    // Each component has to be a finite non-negative allowance on its own. The
    // sum alone is not enough: `(-1., 2.)` adds up to a positive budget and the
    // positive component hides the negative one, so the Python side (`budget()`)
    // and this one disagreed on the same case.
    if !case.rtol.is_finite() || !case.atol.is_finite() || case.rtol < 0. || case.atol < 0. {
        return Err(format!(
            "verification tolerance is not a finite non-negative allowance: rtol={} atol={}",
            case.rtol, case.atol
        )
        .into());
    }
    let allowance = case.atol + case.rtol * reference.abs();
    if !allowance.is_finite() {
        return Err(format!(
            "verification tolerance budget overflows: rtol={} atol={}",
            case.rtol, case.atol
        )
        .into());
    }
    if miss == 0. {
        return Ok(0.);
    }
    if allowance == 0. {
        return Ok(f64::INFINITY);
    }
    Ok(miss / allowance)
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
    if matches!(expected.dtype(), DType::I32 | DType::I64) {
        let equal = match expected.dtype() {
            DType::I32 => host.as_slice::<i32>()? == expected.as_slice::<i32>()?,
            DType::I64 => host.as_slice::<i64>()? == expected.as_slice::<i64>()?,
            _ => unreachable!(),
        };
        if !equal {
            return Err("verification exact integer mismatch".into());
        }
        return Ok(0.0);
    }
    let mut worst = 0.0_f64;
    for (x, y) in numbers(&host)?.iter().zip(numbers(expected)?) {
        if *x == y {
            continue;
        }
        let error = scored(case, (*x - y).norm(), y.norm())?;
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
    session: &mut dyn Engine,
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
    backend: &mut CudaEngine,
) -> Result<(Option<Value>, f64, PhaseTimings)> {
    if case.op.starts_with("einsum_plan") {
        return plan::run(root, case, settings, backend)
            .map(|(timing, error)| (timing, error, PhaseTimings::default()));
    }
    if matches!(
        case.op.as_str(),
        "householder_from_factors"
            | "householder_append"
            | "householder_r"
            | "householder_q_columns"
    ) {
        return state::run(root, case, mode, settings, backend)
            .map(|(timing, error)| (timing, error, PhaseTimings::default()));
    }
    if matches!(
        case.op.as_str(),
        "svd_full" | "slogdet" | "householder_qr" | "lu"
    ) {
        return multi::run(root, case, mode, settings, backend)
            .map(|(timing, error)| (timing, error, PhaseTimings::default()));
    }
    let fixtures_started = Instant::now();
    let input_host = case
        .inputs
        .iter()
        .map(|a| read_array(root, a))
        .collect::<Result<Vec<_>>>()?;
    let mut expected = Expected::new(root, case, 0)?;
    let fixture_ms = fixtures_started.elapsed().as_secs_f64() * 1e3;
    let rt = backend.runtime().clone();
    let input = input_host
        .iter()
        .map(|a| Ok(upload_tensor(&rt, a)?))
        .collect::<Result<Vec<_>>>()?;
    // Serialized reference length is the exact output size for every supported
    // dtype, including I64 and Bool; do not estimate it with a partial dtype table.
    let bytes: usize = fs::metadata(root.join(&case.outputs[0].file))?
        .len()
        .try_into()?;
    if mode == "tenferro-cuda-direct" {
        let mut executor = FftExecutor::default();
        backend.with_backend_session(|session| {
            if !case.read_op.is_empty() {
                return read_layout::run(
                    case,
                    &input,
                    &expected.tensor,
                    &rt,
                    session,
                    settings,
                    bytes,
                )
                .map(|(timing, error)| (timing, error, uninstrumented_phases()));
            }
            if case.op.contains("_into") {
                return reuse::run(
                    case,
                    &input,
                    &expected.tensor,
                    &rt,
                    session,
                    settings,
                    bytes,
                    None,
                    None,
                )
                .map(|(timing, error)| (timing, error, uninstrumented_phases()));
            }
            let probe = direct(
                case,
                &input,
                &mut executor,
                session,
                structural::prepare_direct(case, &input),
            )?;
            let error = expected.verify(&rt, &probe)?;
            structural::verify_materialized(
                case,
                &input.iter().collect::<Vec<_>>(),
                &probe,
                session,
            )?;
            drop(probe);
            if settings.verify_only || case.excluded.contains_key(mode) {
                return Ok((None, error, phases_of(&expected, fixture_ms)));
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
            let error = error.max(expected.verify(&rt, &final_output)?);
            structural::verify_materialized(
                case,
                &input.iter().collect::<Vec<_>>(),
                &final_output,
                session,
            )?;
            Ok((Some(timing), error, phases_of(&expected, fixture_ms)))
        })?
    } else {
        // The runtime stores no engine (tenferro-rs #2206); every eager call
        // borrows the caller-held CUDA engine.
        let runtime = EagerRuntime::new()?;
        let _guard = runtime.no_grad();
        runtime.with_eager_session(backend, |session| {
            let input = input
                .into_iter()
                .map(|a| Ok(session.constant_from(a)?))
                .collect::<Result<Vec<_>>>()?;
            let probe = eager(case, &input, session, structural::prepare(case))?;
            let error = verify_eager_inspected(&rt, session, &probe, &mut expected)?;
            drop(probe);
            if settings.verify_only || case.excluded.contains_key(mode) {
                return Ok((None, error, phases_of(&expected, fixture_ms)));
            }
            let timing = measure(
                settings,
                bytes,
                &rt,
                || structural::prepare(case),
                |args| eager(case, &input, session, args),
            )?;
            let final_output = eager(case, &input, session, structural::prepare(case))?;
            let error = error.max(verify_eager_inspected(
                &rt,
                session,
                &final_output,
                &mut expected,
            )?);
            Ok((Some(timing), error, phases_of(&expected, fixture_ms)))
        })
    }
}
/// Verify an eager output through the reference.
///
/// The placement and dtype guards run first: they are the statement that this row is a
/// CUDA row at all, and they cost no reference bytes.
fn verify_eager_inspected(
    rt: &CudaRuntime,
    session: &mut tenferro_ad::EagerSession<'_>,
    output: &EagerTensor,
    expected: &mut Expected<'_>,
) -> Result<f64> {
    let read = output.tensor_read();
    let reference = expected.tensor_metadata();
    let device = read
        .placement()
        .device
        .as_ref()
        .ok_or("verification eager output has no CUDA placement")?;
    if device.kind != tenferro_tensor::DeviceKind::Gpu(tenferro_tensor::GpuBackendKind::Cuda)
        || device.ordinal != env_usize("GPU_BENCH_DEVICE", 0)?
        || read.allocation_domain().is_none()
        || read.shape() != reference.0
        || read.dtype() != reference.1
    {
        return Err("verification eager original placement/shape/dtype mismatch".into());
    }
    if let Some(tensor) = read.as_tensor() {
        return expected.verify(rt, tensor);
    }
    // This public inspection copy is after clock stop (or before sampling).
    // It must never turn a metadata-only operation into a measured materialization.
    let inspected = session.duplicate_value(output)?;
    expected.verify(rt, &inspected)
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
    if env_usize("GPU_API_LIFETIME_COUNTERS", 0)? == 1 {
        if mode != "tenferro-cuda-direct" || std::env::var("GPU_API_ROLE").as_deref() != Ok("debug")
        {
            return Err("counter passes require direct API and debug GPU role".into());
        }
        return lifetime::run(root, &cases, Path::new(&output), device);
    }
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    // One engine for the whole process: the platform context, its stream and pool,
    // the module and artifact caches and the vendor handles are setup, not
    // measurement, and opening them once per case put them inside every case's
    // untimed wall clock.
    let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(device.try_into()?))?;
    let overhead_budget = std::env::var("GPU_API_OVERHEAD_BUDGET")
        .ok()
        .map(|value| value.parse::<f64>())
        .transpose()?
        .unwrap_or(0.2);
    for case in cases {
        eprintln!("{} {}", mode, case.id);
        let case_started = Instant::now();
        let mut row = json!({"schema_version":1,"case_id":case.id,"op":case.op,"backend":mode,
            "gpu_role":std::env::var("GPU_API_ROLE").unwrap_or_else(|_| "benchmark".into()),
            "physical_device_index":env_usize("GPU_BENCH_PHYSICAL_DEVICE", device)?,
            "device_uuid":std::env::var("GPU_BENCH_DEVICE_UUID").ok(),
            "traffic": case.traffic.clone().unwrap_or_else(|| json!({
                "bytes_per_operation": null,
                "unknown_reason": "fixture manifest carries no traffic model for this case"})),
            "device_ordinal":device,"session_policy":"one entered session for initialization, calibration and samples",
            "setup_policy":"fixtures, H2D, handles, plan initialization and output validation outside timing",
            "synchronization":"CudaRuntime::synchronize; no D2H in timed region"});
        if case.op.starts_with("einsum_plan") {
            row["plan_policy"] = json!("ConcreteEinsumPlan prepared once for ik,kj->ij outside execution and sampling; one entered session");
            row["input_layout_policy"] = json!(if case.op.contains("strided") {
                "transposed borrowed CUDA views and input handles prepared outside timing"
            } else {
                "compact CUDA inputs; input handles prepared outside timing"
            });
        }
        if matches!(case.op.as_str(), "dot_general_axes" | "dot_general_conj") {
            row["dot_configuration"] = json!({"lhs_contracting_dims":case.lhs_contracting_dims,"rhs_contracting_dims":case.rhs_contracting_dims,"lhs_batch_dims":case.lhs_batch_dims,"rhs_batch_dims":case.rhs_batch_dims,"lhs_conj":case.lhs_conj,"rhs_conj":case.rhs_conj});
            row["batch_output_policy"] = json!("native batch-last; reference axis normalization only during untimed CPU verification");
        }
        if !case.read_op.is_empty() {
            row["input_layout_policy"] = json!("transposed borrowed TensorRead over prepared CUDA storage; view and read-handle construction outside timing");
        }
        if case.op.contains("_into_accum") {
            row["input_restoration"] = json!("GPU seed copied into each destination before every batch; synchronized outside timing");
            row["accumulation"] =
                json!({"alpha":case.alpha.unwrap_or(0.5),"beta":case.beta.unwrap_or(1.0)});
        }
        if case.op == "dot_general_read_into" {
            row["accumulation"] = json!({"alpha":1.0,"beta":0.0});
        }
        if case.op == "dot_general_read_into_accum_cached" {
            row["cache_policy"] =
                json!("CUDA delegates to ordinary accumulation; session slot Some(0) is unused");
        }
        if let Some(reason) = case.unavailable.get(&mode) {
            row["status"] = json!("unsupported");
            row["reason"] = json!(reason);
            row["cuda_implementation_verified"] = json!(false);
            writeln!(file, "{}", row)?;
            continue;
        }
        let compiles_before = backend.runtime().compile_attempts();
        let outcome = run(root, &case, &mode, &settings, &mut backend);
        let case_wall_ms = case_started.elapsed().as_secs_f64() * 1e3;
        let compiled = backend.runtime().compile_attempts() - compiles_before;
        match outcome {
            Ok((timing, error, phases)) => {
                row["status"] = json!(if timing.is_some() {
                    "ok"
                } else if settings.verify_only && !case.excluded.contains_key(&mode) {
                    "verification_only"
                } else {
                    "unsupported"
                });
                let timed_ms = timed_ms_of(timing.as_ref());
                let untimed_ms = timed_ms.map(|timed| (case_wall_ms - timed).max(0.0));
                let warning = overhead_warning(&case.id, timed_ms, case_wall_ms, overhead_budget);
                if let Some(message) = warning.as_ref() {
                    eprintln!("overhead budget: {message}");
                }
                row["overhead"] = json!({
                    "timed_ms": timed_ms,
                    "case_wall_ms": case_wall_ms,
                    "untimed_ms": untimed_ms,
                    "fixture_ms": if phases.split_measured { Some(phases.fixture_ms) } else { None },
                    "verify_ms": if phases.split_measured { Some(phases.verify_ms) } else { None },
                    "fingerprint_compares": phases.fingerprint_compares,
                    "fingerprint_disagreements": phases.fingerprint_disagreements,
                    "split_measured": phases.split_measured,
                    "budget_ratio": overhead_budget,
                    "over_budget": warning.is_some(),
                    "warning": warning,
                    "scope": "case wall clock outside the timed region: fixtures, H2D, handles, plan initialization, validation, output destruction",
                });
                // miss means this case compiled a module and hit means it did not, so a
                // warm run cannot hide the compile it skipped. No compile interval is
                // measured here: a Rust arm cannot time NVRTC alone, and the untimed case
                // wall around a compile also holds that case's fixture read, transfers and
                // verification. The module count is therefore the compile diagnostic, and
                // the untimed wall is published as a wall (`overhead.untimed_ms`), never as
                // a compile time.
                row["compile_cache"] = json!(if compiled > 0 { "miss" } else { "hit" });
                row["compile_modules"] = json!(compiled);
                row["compile_time_ms"] = json!(null);
                row["compile_time_scope"] = json!(COMPILE_TIME_SCOPE);
                row["timing"] = match timing {
                    Some(mut timing) => {
                        // The block keeps the field, always null, so a reader of a raw arm
                        // file cannot mistake the module count for a measured interval.
                        timing["compile_time_ms"] = json!(null);
                        json!(timing)
                    }
                    None => json!(null),
                };
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
                // A case that failed after compiling still cost the run that module. Only
                // the module count is a compile diagnostic; its wall is the case's whole
                // untimed wall and is not a compile interval.
                if compiled > 0 {
                    row["compile_cache"] = json!("miss");
                    row["compile_modules"] = json!(compiled);
                    row["compile_time_ms"] = json!(null);
                    row["compile_time_scope"] = json!(COMPILE_TIME_SCOPE);
                }
            }
        }
        writeln!(file, "{}", row)?;
        file.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_over_budget_case_records_a_warning() {
        // The 20 % default against the measured 21.6 s wall clock of a 50 ms case:
        // the case is far outside it, which must be visible in its row.
        let warning = overhead_warning("cholesky_f64_32x32", Some(50.0), 21_600.0, 0.2)
            .expect("the case is outside the budget");
        assert!(
            warning.starts_with("cholesky_f64_32x32 untimed"),
            "{warning}"
        );
        assert!(warning.contains("43100%"), "{warning}");
        assert!(warning.contains("budget 20%"), "{warning}");
    }

    #[test]
    fn a_case_inside_the_budget_records_no_warning() {
        assert!(overhead_warning("case", Some(50.0), 55.0, 0.2).is_none());
    }

    #[test]
    fn a_case_that_sampled_nothing_records_no_warning() {
        assert!(overhead_warning("case", None, 55.0, 0.2).is_none());
        assert!(overhead_warning("case", Some(0.0), 55.0, 0.2).is_none());
    }

    #[test]
    fn a_longer_setup_only_changes_the_recorded_warning() {
        // Same timed region, growing untimed work: the row's verdict is the timing,
        // and the only thing that varies here is the recorded warning.
        for wall in [51.0, 60.0, 200.0, 21_600.0] {
            let warning = overhead_warning("case", Some(50.0), wall, 0.2);
            assert_eq!(warning.is_some(), wall > 60.0, "wall {wall}");
        }
    }

    fn case_with(rtol: f64, atol: f64) -> Case {
        let mut case: Case = serde_json::from_str(
            r#"{"id":"c","op":"exp","n":1,"inputs":[],"outputs":[],"rtol":0.0,"atol":0.0,"reference":"cpu"}"#,
        )
        .unwrap();
        case.rtol = rtol;
        case.atol = atol;
        case
    }

    /// Filtered by name in CI (`tests::budget`), so the guard cannot be
    /// satisfied by the unrelated overhead-warning tests in this module.
    mod budget {
        use super::*;

        #[test]
        fn an_allowance_that_overflows_from_finite_components_is_not_a_pass() {
            // #417 review third pass: 998 / (1.7e308 + 1.7e308) == 0.0, so a miss of 998
            // against a reference of 1 scored as exact and every public-API output
            // comparison accepted a wrong tensor.
            assert!(
                scored(&case_with(1.7e308, 1.7e308), 998., 1.).is_err(),
                "an overflowing allowance must be refused, not divide the miss to zero"
            );
        }

        #[test]
        fn an_exact_miss_over_a_zero_allowance_is_zero_and_not_nan() {
            // atol = 0 on an off-diagonal Gram entry: 0 / 0 must be an exact 0, and a
            // real miss over no allowance must stay infinite so `worst > 1` rejects it.
            let case = case_with(0., 0.);
            assert_eq!(scored(&case, 0., 0.).unwrap(), 0.);
            assert_eq!(scored(&case, 1., 0.).unwrap(), f64::INFINITY);
            assert!(scored(&case, f64::NAN, 1.).is_err());
            assert!(scored(&case_with(f64::NAN, 0.), 0., 0.).is_err());
            assert!(scored(&case_with(-1e-9, 0.), 1., 1.).is_err());
        }

        #[test]
        fn a_negative_component_is_not_hidden_by_the_other() {
            // rtol = -1 with atol = 2 sums to a positive budget, and swapping the two
            // does the same for atol.  Both must be refused component-wise, or the
            // Rust gate accepts a case the Python `budget()` rejects.
            assert!(scored(&case_with(-1., 2.), 0.5, 1.).is_err());
            assert!(scored(&case_with(2., -1.), 0.5, 1.).is_err());
            assert!(scored(&case_with(1., 1.), 0.5, 1.).is_ok());
        }
    }
}
