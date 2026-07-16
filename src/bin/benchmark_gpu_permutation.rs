//! GPU permutation / materialize-kernel benchmark suite (gpu/permutation).
//!
//! Ports `src/bin/benchmark_permutation.rs` (`cpu/permutation`) to CUDA per
//! `docs/gpu-permutation-suite.md`. Measures the cost of materializing a
//! strided/permuted `f64` tensor view into a compact column-major
//! destination on the GPU and compares:
//!
//! - `tenferro-cuda-transpose`: eager `TensorStructural::transpose` on
//!   `CudaBackend` (compact col-major source only)
//! - `tenferro-cuda-to-contiguous`: `TypedTensor::backend_region_view` (over
//!   the source layout) + `TypedTensorView::transpose_view(perm)` +
//!   `TensorViewCanonicalization::to_contiguous` (accepts arbitrary source
//!   strides), mirroring the CPU runner's view composition
//! - `cutensor`: direct cuTENSOR 2.x `cutensorPermute`, dlopen'd from this
//!   binary (tenferro's own cuTENSOR FFI binds only contraction)
//!
//! `pytorch-cuda`, `jax-cuda`, and `memcpy-d2d` are measured by
//! `scripts/benchmark_gpu_permutation_python.py` and are ignored here.
//!
//! Every participant's output is downloaded and compared elementwise against
//! a host-computed naive reference *outside* any timed region before any
//! timing; a mismatch is reported as `verification_failed` and is never
//! timed, per `AGENTS.md`'s GPU Timing Fairness policy. Timed regions cover
//! host API dispatch plus CUDA device synchronization only; no downloads
//! happen inside a timed closure.
//!
//! Environment variables:
//! - `PATTERN_ID`: run a single pattern id instead of the full suite.
//! - `BENCH_RUNS` / `BENCH_WARMUPS`: override the size-scaled iteration and
//!   warmup counts for every participant.
//! - `BENCH_OUTPUT`: path to write machine-readable JSON Lines results. When
//!   unset, only the human-readable table is printed to stdout.
//! - `GPU_BENCH_DEVICE`: CUDA device ordinal (default `0`).
//! - `TENFERRO_CUTENSOR_PATH`: colon-separated cuTENSOR library search paths
//!   (see `cutensor_ffi::DEFAULT_CUTENSOR_PATHS` for the fallback).

use std::env;
use std::ffi::c_void;
use std::fmt;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use tenferro_gpu::cuda_interop::raw_cuda_stream;
use tenferro_gpu::{device_ptr, download_tensor, gpu_available, upload_tensor, CudaBackend};
use tenferro_tensor::{Tensor, TensorStructural, TensorViewCanonicalization, TypedTensor};

const PATTERN_PATH: &str = "data/instances/permutation_patterns.json";
const SUITE_ID: &str = "gpu/permutation";

// ---------------------------------------------------------------------------
// Pattern schema (subset of data/instances/permutation_patterns.json; no
// `deny_unknown_fields` so CPU-only keys such as `participants` / `notes`
// are simply ignored here).
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct PatternSuite {
    version: u32,
    index_base: u32,
    semantics: String,
    data: String,
    patterns: Vec<PermutePattern>,
}

#[derive(Debug, Deserialize)]
struct PermutePattern {
    id: String,
    label: String,
    #[allow(dead_code)]
    dtype: String,
    shape: Vec<usize>,
    perm: Vec<usize>,
    src_layout: LayoutPattern,
    #[allow(dead_code)]
    dst_layout: LayoutPattern,
    #[serde(default)]
    participants_gpu: Vec<String>,
    #[allow(dead_code)]
    #[serde(default)]
    notes_gpu: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LayoutPattern {
    ColMajor,
    ExplicitStrides { strides: Vec<isize> },
}

fn load_pattern_suite() -> Result<PatternSuite, Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string(PATTERN_PATH)?;
    let suite: PatternSuite = serde_json::from_str(&json)?;
    if suite.version != 1 {
        return Err(Box::new(PatternError(format!(
            "unsupported pattern schema version {}",
            suite.version
        ))));
    }
    if suite.index_base != 0 {
        return Err(Box::new(PatternError(
            "permute patterns must use index_base = 0".into(),
        )));
    }
    if suite.semantics != "out[i0,...,ik] = src[i_perm0,...,i_permk]" {
        return Err(Box::new(PatternError(format!(
            "unsupported semantics {:?}",
            suite.semantics
        ))));
    }
    if suite.data != "deterministic_index_value" {
        return Err(Box::new(PatternError(format!(
            "unsupported data mode {:?}",
            suite.data
        ))));
    }
    Ok(suite)
}

#[derive(Debug)]
struct PatternError(String);

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PatternError {}

// ---------------------------------------------------------------------------
// Layout / data helpers (mirrors src/bin/benchmark_permutation.rs)
// ---------------------------------------------------------------------------

fn col_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }
    strides
}

fn layout_strides(shape: &[usize], layout: &LayoutPattern) -> Vec<isize> {
    match layout {
        LayoutPattern::ColMajor => col_major_strides(shape),
        LayoutPattern::ExplicitStrides { strides } => strides.clone(),
    }
}

fn output_shape(pattern: &PermutePattern) -> Vec<usize> {
    pattern
        .perm
        .iter()
        .map(|&axis| pattern.shape[axis])
        .collect()
}

fn deterministic_data(total: usize) -> Vec<f64> {
    (0..total).map(|i| i as f64 + 1.0).collect()
}

/// Generic strided copy using odometer iteration; the host-side correctness
/// reference for every GPU participant.
unsafe fn naive_strided_copy(
    src_ptr: *const f64,
    dst_ptr: *mut f64,
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) {
    let rank = dims.len();
    let total: usize = dims.iter().product();

    let mut idx = vec![0usize; rank];
    let mut src_off = 0isize;
    let mut dst_off = 0isize;

    for _ in 0..total {
        *dst_ptr.offset(dst_off) = *src_ptr.offset(src_off);
        for d in 0..rank {
            idx[d] += 1;
            src_off += src_strides[d];
            dst_off += dst_strides[d];
            if idx[d] < dims[d] {
                break;
            }
            src_off -= (idx[d] as isize) * src_strides[d];
            dst_off -= (idx[d] as isize) * dst_strides[d];
            idx[d] = 0;
        }
    }
}

struct PreparedPattern {
    src_data: Vec<f64>,
    src_strides: Vec<isize>,
    out_shape: Vec<usize>,
    dst_strides: Vec<isize>,
    reference: Vec<f64>,
}

fn prepare_pattern(pattern: &PermutePattern) -> PreparedPattern {
    let src_strides = layout_strides(&pattern.shape, &pattern.src_layout);
    let total: usize = pattern.shape.iter().product();
    let src_data = deterministic_data(total);

    let src_perm_strides: Vec<isize> = pattern.perm.iter().map(|&axis| src_strides[axis]).collect();
    let out_shape = output_shape(pattern);
    let dst_strides = col_major_strides(&out_shape);

    let mut reference = vec![0.0f64; out_shape.iter().product()];
    unsafe {
        naive_strided_copy(
            src_data.as_ptr(),
            reference.as_mut_ptr(),
            &out_shape,
            &src_perm_strides,
            &dst_strides,
        );
    }

    PreparedPattern {
        src_data,
        src_strides,
        out_shape,
        dst_strides,
        reference,
    }
}

fn verify_output(actual: &[f64], expected: &[f64]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "length mismatch {} != {}",
            actual.len(),
            expected.len()
        ));
    }
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if a != e {
            return Err(format!("mismatch at element {i}: {a} != {e}"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Timing (mirrors src/bin/benchmark_permutation.rs)
// ---------------------------------------------------------------------------

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort();
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        (samples[n / 2 - 1] + samples[n / 2]) / 2
    }
}

struct Timing {
    warmup: usize,
    iters: usize,
    median_ms: f64,
    p25_ms: f64,
    p75_ms: f64,
    gbps: f64,
}

fn bench_n(warmup: usize, iters: usize, bytes: usize, mut f: impl FnMut()) -> Timing {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed());
    }
    let med = median(&mut samples);
    let ms = med.as_secs_f64() * 1e3;
    let gbps = (bytes as f64) / med.as_secs_f64() / 1e9;
    samples.sort();
    let p25 = samples[samples.len() / 4].as_secs_f64() * 1e3;
    let p75 = samples[samples.len() * 3 / 4].as_secs_f64() * 1e3;
    Timing {
        warmup,
        iters,
        median_ms: ms,
        p25_ms: p25,
        p75_ms: p75,
        gbps,
    }
}

fn timing_counts(total: usize) -> (usize, usize) {
    if total >= 1 << 23 {
        (3, 15)
    } else {
        (5, 40)
    }
}

fn timing_counts_env(total: usize) -> (usize, usize) {
    let (default_warmup, default_iters) = timing_counts(total);
    let warmup = env::var("BENCH_WARMUPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_warmup);
    let iters = env::var("BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_iters);
    (warmup, iters.max(1))
}

// ---------------------------------------------------------------------------
// Result records
//
// Mirrors src/bin/benchmark_permutation.rs's ResultRecord, but `device` (GPU
// name string) replaces `threads`, and `gbps` is named `bandwidth_gbs` per
// docs/gpu-permutation-suite.md's timing policy.
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ResultRecord {
    schema_version: u32,
    suite_id: &'static str,
    runner: &'static str,
    pattern_id: String,
    label: String,
    backend: &'static str,
    shape: Vec<usize>,
    perm: Vec<usize>,
    dtype: &'static str,
    elems: usize,
    bytes_rw: usize,
    device: String,
    status: &'static str,
    correctness: &'static str,
    per_call_allocation: bool,
    warmup: Option<usize>,
    iters: Option<usize>,
    median_ms: Option<f64>,
    p25_ms: Option<f64>,
    p75_ms: Option<f64>,
    bandwidth_gbs: Option<f64>,
    notes: Option<String>,
}

impl ResultRecord {
    fn with_note(mut self, note: String) -> Self {
        self.notes = Some(note);
        self
    }
}

struct RecordSink {
    file: Option<File>,
    any_failed: bool,
}

impl RecordSink {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let file = match env::var("BENCH_OUTPUT") {
            Ok(path) => Some(File::create(path)?),
            Err(_) => None,
        };
        Ok(Self {
            file,
            any_failed: false,
        })
    }

    fn emit(&mut self, record: &ResultRecord) {
        if record.status == "verification_failed" {
            self.any_failed = true;
        }
        if let Some(file) = self.file.as_mut() {
            let line = serde_json::to_string(record).expect("record must serialize");
            writeln!(file, "{line}").expect("failed to write BENCH_OUTPUT record");
        }
    }
}

fn print_human_row(backend: &str, timing: Option<&Timing>, note: Option<&str>) {
    match timing {
        Some(t) => println!(
            "  {backend:30} {ms:8.3} ms  ({p25:.3} / {p75:.3})  {gbps:6.2} GB/s",
            backend = backend,
            ms = t.median_ms,
            p25 = t.p25_ms,
            p75 = t.p75_ms,
            gbps = t.gbps
        ),
        None => println!(
            "  {backend:30} skipped: {note}",
            backend = backend,
            note = note.unwrap_or("no timing")
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn base_record(
    pattern: &PermutePattern,
    backend: &'static str,
    elems: usize,
    bytes: usize,
    device: &str,
    status: &'static str,
    correctness: &'static str,
    per_call_allocation: bool,
) -> ResultRecord {
    ResultRecord {
        schema_version: 1,
        suite_id: SUITE_ID,
        runner: "rust",
        pattern_id: pattern.id.clone(),
        label: pattern.label.clone(),
        backend,
        shape: pattern.shape.clone(),
        perm: pattern.perm.clone(),
        dtype: "f64",
        elems,
        bytes_rw: bytes,
        device: device.to_string(),
        status,
        correctness,
        per_call_allocation,
        warmup: None,
        iters: None,
        median_ms: None,
        p25_ms: None,
        p75_ms: None,
        bandwidth_gbs: None,
        notes: None,
    }
}

fn finish_ok(mut record: ResultRecord, timing: Timing, sink: &mut RecordSink) {
    record.warmup = Some(timing.warmup);
    record.iters = Some(timing.iters);
    record.median_ms = Some(timing.median_ms);
    record.p25_ms = Some(timing.p25_ms);
    record.p75_ms = Some(timing.p75_ms);
    record.bandwidth_gbs = Some(timing.gbps);
    print_human_row(record.backend, Some(&timing), None);
    sink.emit(&record);
}

fn finish_skip(record: ResultRecord, sink: &mut RecordSink) {
    print_human_row(record.backend, None, record.notes.as_deref());
    sink.emit(&record);
}

// ---------------------------------------------------------------------------
// cuTENSOR permutation FFI (dlopen). tenferro-gpu's own cuTENSOR bindings
// only cover contraction (dot_general); permutation is bound directly here.
// Style mirrors
// extern/tenferro-rs/crates/tenferro-gpu/src/cubecl/ffi/cutensor.rs, with
// the extra function pointer surfaced by
// extern/tenferro-rs/docs/plans/2026-02-23-cuda-backend-impl-plan.md
// (`cutensorCreatePermutation` / `cutensorPermute`, ~line 385).
// ---------------------------------------------------------------------------

mod cutensor_ffi {
    use libloading::Library;
    use std::ffi::{c_void, CStr};
    use std::os::raw::c_char;

    pub type HandleRaw = *mut c_void;
    pub type TensorDescRaw = *mut c_void;
    pub type OpDescRaw = *mut c_void;
    pub type PlanPrefRaw = *mut c_void;
    pub type PlanRaw = *mut c_void;
    pub type ComputeDesc = *const c_void;
    pub type CudaStream = *mut c_void;
    type Status = i32;

    const STATUS_SUCCESS: Status = 0;

    /// Default cuTENSOR search paths. Override with `TENFERRO_CUTENSOR_PATH`
    /// (colon-separated), same convention as tenferro-gpu's own cuTENSOR FFI.
    pub const DEFAULT_CUTENSOR_PATHS: &[&str] = &[
        "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2",
        "libcutensor.so.2",
        "libcutensor.so",
    ];

    #[repr(i32)]
    #[derive(Clone, Copy)]
    pub enum DataType {
        R64F = 1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    enum Operator {
        Identity = 1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    enum Algo {
        Default = -1,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    enum JitMode {
        None = 0,
    }

    #[repr(i32)]
    #[derive(Clone, Copy)]
    enum WorkspacePref {
        Default = 2,
    }

    type FnCreate = unsafe extern "C" fn(*mut HandleRaw) -> Status;
    type FnDestroy = unsafe extern "C" fn(HandleRaw) -> Status;
    type FnCreateTensorDescriptor = unsafe extern "C" fn(
        HandleRaw,
        *mut TensorDescRaw,
        u32,
        *const i64,
        *const i64,
        DataType,
        u32,
    ) -> Status;
    type FnDestroyTensorDescriptor = unsafe extern "C" fn(TensorDescRaw) -> Status;
    type FnCreatePermutation = unsafe extern "C" fn(
        HandleRaw,
        *mut OpDescRaw,
        TensorDescRaw,
        *const i32,
        Operator,
        TensorDescRaw,
        *const i32,
        ComputeDesc,
    ) -> Status;
    type FnDestroyOperationDescriptor = unsafe extern "C" fn(OpDescRaw) -> Status;
    type FnCreatePlanPreference =
        unsafe extern "C" fn(HandleRaw, *mut PlanPrefRaw, Algo, JitMode) -> Status;
    type FnDestroyPlanPreference = unsafe extern "C" fn(PlanPrefRaw) -> Status;
    type FnEstimateWorkspaceSize = unsafe extern "C" fn(
        HandleRaw,
        OpDescRaw,
        PlanPrefRaw,
        WorkspacePref,
        *mut u64,
    ) -> Status;
    type FnCreatePlan =
        unsafe extern "C" fn(HandleRaw, *mut PlanRaw, OpDescRaw, PlanPrefRaw, u64) -> Status;
    type FnDestroyPlan = unsafe extern "C" fn(PlanRaw) -> Status;
    type FnPermute = unsafe extern "C" fn(
        HandleRaw,
        PlanRaw,
        *const c_void,
        *const c_void,
        *mut c_void,
        CudaStream,
    ) -> Status;
    type FnGetErrorString = unsafe extern "C" fn(Status) -> *const c_char;

    #[derive(Clone, Copy)]
    struct Vtable {
        create: FnCreate,
        destroy: FnDestroy,
        create_tensor_descriptor: FnCreateTensorDescriptor,
        destroy_tensor_descriptor: FnDestroyTensorDescriptor,
        create_permutation: FnCreatePermutation,
        destroy_operation_descriptor: FnDestroyOperationDescriptor,
        create_plan_preference: FnCreatePlanPreference,
        destroy_plan_preference: FnDestroyPlanPreference,
        estimate_workspace_size: FnEstimateWorkspaceSize,
        create_plan: FnCreatePlan,
        destroy_plan: FnDestroyPlan,
        permute: FnPermute,
        get_error_string: FnGetErrorString,
        compute_desc_64f: ComputeDesc,
    }

    unsafe fn load_symbol<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, String> {
        let symbol = lib.get::<T>(name).map_err(|err| {
            format!(
                "failed to load cuTENSOR symbol {}: {err}",
                String::from_utf8_lossy(name).trim_end_matches('\0')
            )
        })?;
        Ok(*symbol)
    }

    unsafe fn load_data_symbol<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, String> {
        let symbol_name = String::from_utf8_lossy(name).trim_end_matches('\0').to_owned();
        let symbol = lib
            .get::<*const T>(name)
            .map_err(|err| format!("failed to load cuTENSOR data symbol {symbol_name}: {err}"))?;
        let ptr = *symbol;
        if ptr.is_null() {
            return Err(format!(
                "cuTENSOR data symbol {symbol_name} resolved to a null pointer"
            ));
        }
        Ok(unsafe { std::ptr::read(ptr) })
    }

    fn library_search_paths() -> Vec<String> {
        if let Ok(val) = std::env::var("TENFERRO_CUTENSOR_PATH") {
            val.split(':').filter(|s| !s.is_empty()).map(String::from).collect()
        } else {
            DEFAULT_CUTENSOR_PATHS.iter().map(|s| s.to_string()).collect()
        }
    }

    fn status_message(vtable: &Vtable, status: Status) -> String {
        let ptr = unsafe { (vtable.get_error_string)(status) };
        if ptr.is_null() {
            return format!("status code {status}");
        }
        unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
    }

    fn check(vtable: &Vtable, status: Status, call: &'static str) -> Result<(), String> {
        if status == STATUS_SUCCESS {
            return Ok(());
        }
        Err(format!(
            "{call} failed with cuTENSOR {} ({status})",
            status_message(vtable, status)
        ))
    }

    /// Owns the dlopen'd library, the cuTENSOR handle, and the resolved
    /// function/data symbol vtable for the process lifetime of this
    /// benchmark binary.
    pub struct CutensorLib {
        _lib: Library,
        vtable: Vtable,
        handle: HandleRaw,
    }

    impl CutensorLib {
        pub fn load() -> Result<Self, String> {
            let paths = library_search_paths();
            let mut errors = Vec::new();
            for path in &paths {
                let lib = match unsafe { Library::new(path) } {
                    Ok(lib) => lib,
                    Err(err) => {
                        errors.push(format!("{path}: {err}"));
                        continue;
                    }
                };
                let vtable = unsafe {
                    Vtable {
                        create: load_symbol(&lib, b"cutensorCreate\0")?,
                        destroy: load_symbol(&lib, b"cutensorDestroy\0")?,
                        create_tensor_descriptor: load_symbol(
                            &lib,
                            b"cutensorCreateTensorDescriptor\0",
                        )?,
                        destroy_tensor_descriptor: load_symbol(
                            &lib,
                            b"cutensorDestroyTensorDescriptor\0",
                        )?,
                        create_permutation: load_symbol(&lib, b"cutensorCreatePermutation\0")?,
                        destroy_operation_descriptor: load_symbol(
                            &lib,
                            b"cutensorDestroyOperationDescriptor\0",
                        )?,
                        create_plan_preference: load_symbol(
                            &lib,
                            b"cutensorCreatePlanPreference\0",
                        )?,
                        destroy_plan_preference: load_symbol(
                            &lib,
                            b"cutensorDestroyPlanPreference\0",
                        )?,
                        estimate_workspace_size: load_symbol(
                            &lib,
                            b"cutensorEstimateWorkspaceSize\0",
                        )?,
                        create_plan: load_symbol(&lib, b"cutensorCreatePlan\0")?,
                        destroy_plan: load_symbol(&lib, b"cutensorDestroyPlan\0")?,
                        permute: load_symbol(&lib, b"cutensorPermute\0")?,
                        get_error_string: load_symbol(&lib, b"cutensorGetErrorString\0")?,
                        compute_desc_64f: load_data_symbol(&lib, b"CUTENSOR_COMPUTE_DESC_64F\0")?,
                    }
                };
                let mut handle = std::ptr::null_mut();
                let status = unsafe { (vtable.create)(&mut handle) };
                check(&vtable, status, "cutensorCreate")?;
                return Ok(Self {
                    _lib: lib,
                    vtable,
                    handle,
                });
            }
            Err(format!(
                "failed to load cuTENSOR library (tried {}): {}",
                paths.join(", "),
                errors.join("; ")
            ))
        }

        /// Build a permutation plan for `out[modes_b] = in[modes_a]`. Mirrors
        /// the cuTENSOR call sequence: create tensor descriptors for A/B,
        /// `cutensorCreatePermutation`, a default plan preference, then
        /// `cutensorCreatePlan`. Returns `Err` (never panics) if the library
        /// rejects the rank/mode configuration, so callers can record a
        /// `skipped` status instead of crashing (some cuTENSOR builds cap
        /// the number of modes below this suite's largest patterns).
        pub fn build_permutation_plan(
            &self,
            extents_a: &[i64],
            strides_a: &[i64],
            modes_a: &[i32],
            extents_b: &[i64],
            strides_b: &[i64],
            modes_b: &[i32],
        ) -> Result<Plan<'_>, String> {
            let desc_a = self.create_tensor_descriptor(extents_a, strides_a)?;
            let desc_b = self.create_tensor_descriptor(extents_b, strides_b)?;

            let mut op_raw = std::ptr::null_mut();
            let status = unsafe {
                (self.vtable.create_permutation)(
                    self.handle,
                    &mut op_raw,
                    desc_a.raw,
                    modes_a.as_ptr(),
                    Operator::Identity,
                    desc_b.raw,
                    modes_b.as_ptr(),
                    self.vtable.compute_desc_64f,
                )
            };
            check(&self.vtable, status, "cutensorCreatePermutation")?;
            let op = OpDesc {
                lib: self,
                raw: op_raw,
            };

            let mut pref_raw = std::ptr::null_mut();
            let status = unsafe {
                (self.vtable.create_plan_preference)(
                    self.handle,
                    &mut pref_raw,
                    Algo::Default,
                    JitMode::None,
                )
            };
            check(&self.vtable, status, "cutensorCreatePlanPreference")?;
            let pref = PlanPref {
                lib: self,
                raw: pref_raw,
            };

            let mut workspace_size = 0u64;
            let status = unsafe {
                (self.vtable.estimate_workspace_size)(
                    self.handle,
                    op.raw,
                    pref.raw,
                    WorkspacePref::Default,
                    &mut workspace_size,
                )
            };
            check(&self.vtable, status, "cutensorEstimateWorkspaceSize")?;

            let mut plan_raw = std::ptr::null_mut();
            let status = unsafe {
                (self.vtable.create_plan)(self.handle, &mut plan_raw, op.raw, pref.raw, workspace_size)
            };
            check(&self.vtable, status, "cutensorCreatePlan")?;
            Ok(Plan {
                lib: self,
                raw: plan_raw,
            })
        }

        fn create_tensor_descriptor(
            &self,
            extents: &[i64],
            strides: &[i64],
        ) -> Result<TensorDesc<'_>, String> {
            const CUDA_ALLOCATION_ALIGNMENT: u32 = 256;
            let num_modes = u32::try_from(extents.len())
                .map_err(|_| "tensor rank exceeds cuTENSOR u32 limit".to_string())?;
            let mut raw = std::ptr::null_mut();
            let status = unsafe {
                (self.vtable.create_tensor_descriptor)(
                    self.handle,
                    &mut raw,
                    num_modes,
                    extents.as_ptr(),
                    strides.as_ptr(),
                    DataType::R64F,
                    CUDA_ALLOCATION_ALIGNMENT,
                )
            };
            check(&self.vtable, status, "cutensorCreateTensorDescriptor")?;
            Ok(TensorDesc { lib: self, raw })
        }

        /// Execute `cutensorPermute(alpha=1.0, A, B, stream)` and return
        /// once the call is enqueued (the caller is responsible for device
        /// synchronization, matching this suite's timing-fairness policy).
        pub fn permute(
            &self,
            plan: &Plan<'_>,
            alpha: f64,
            a_ptr: *const c_void,
            b_ptr: *mut c_void,
            stream: CudaStream,
        ) -> Result<(), String> {
            let status = unsafe {
                (self.vtable.permute)(
                    self.handle,
                    plan.raw,
                    &alpha as *const f64 as *const c_void,
                    a_ptr,
                    b_ptr,
                    stream,
                )
            };
            check(&self.vtable, status, "cutensorPermute")
        }
    }

    impl Drop for CutensorLib {
        fn drop(&mut self) {
            let status = unsafe { (self.vtable.destroy)(self.handle) };
            if status != STATUS_SUCCESS {
                eprintln!(
                    "benchmark_gpu_permutation: cutensorDestroy failed with {} ({status})",
                    status_message(&self.vtable, status)
                );
            }
        }
    }

    struct TensorDesc<'a> {
        lib: &'a CutensorLib,
        raw: TensorDescRaw,
    }

    impl Drop for TensorDesc<'_> {
        fn drop(&mut self) {
            unsafe { (self.lib.vtable.destroy_tensor_descriptor)(self.raw) };
        }
    }

    struct OpDesc<'a> {
        lib: &'a CutensorLib,
        raw: OpDescRaw,
    }

    impl Drop for OpDesc<'_> {
        fn drop(&mut self) {
            unsafe { (self.lib.vtable.destroy_operation_descriptor)(self.raw) };
        }
    }

    struct PlanPref<'a> {
        lib: &'a CutensorLib,
        raw: PlanPrefRaw,
    }

    impl Drop for PlanPref<'_> {
        fn drop(&mut self) {
            unsafe { (self.lib.vtable.destroy_plan_preference)(self.raw) };
        }
    }

    pub struct Plan<'a> {
        lib: &'a CutensorLib,
        raw: PlanRaw,
    }

    impl Drop for Plan<'_> {
        fn drop(&mut self) {
            unsafe { (self.lib.vtable.destroy_plan)(self.raw) };
        }
    }
}

use cutensor_ffi::CutensorLib;

fn cuda_ptr_mut(addr: u64) -> *mut c_void {
    addr as usize as *mut c_void
}

fn cuda_ptr_const(addr: u64) -> *const c_void {
    addr as usize as *const c_void
}

// ---------------------------------------------------------------------------
// Participants
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_tenferro_cuda_transpose(
    pattern: &PermutePattern,
    prepared: &PreparedPattern,
    warmup: usize,
    iters: usize,
    bytes: usize,
    device_name: &str,
    backend: &mut CudaBackend,
    sink: &mut RecordSink,
) {
    let total = prepared.reference.len();
    let name: &'static str = "tenferro-cuda-transpose";
    if !matches!(pattern.src_layout, LayoutPattern::ColMajor) {
        finish_skip(
            base_record(pattern, name, total, bytes, device_name, "skipped", "skipped", true)
                .with_note(
                    "tenferro-cuda-transpose (eager op) requires a compact col-major source"
                        .into(),
                ),
            sink,
        );
        return;
    }

    let host = Tensor::from_vec_col_major(pattern.shape.clone(), prepared.src_data.clone())
        .expect("building source tensor must succeed");
    let gpu_in = upload_tensor(backend.runtime(), &host).expect("upload must succeed");

    let out = backend
        .transpose(&gpu_in, &pattern.perm)
        .expect("tenferro-cuda-transpose must succeed on a validated pattern");
    backend.runtime().synchronize().expect("device sync");
    let downloaded = download_tensor(backend.runtime(), &out).expect("download must succeed");
    let actual = downloaded
        .as_slice::<f64>()
        .expect("tenferro-cuda-transpose output must be f64");
    if let Err(msg) = verify_output(actual, &prepared.reference) {
        finish_skip(
            base_record(
                pattern,
                name,
                total,
                bytes,
                device_name,
                "verification_failed",
                "failed",
                true,
            )
            .with_note(msg),
            sink,
        );
        return;
    }

    let timing = bench_n(warmup, iters, bytes, || {
        let out = backend.transpose(&gpu_in, &pattern.perm).unwrap();
        backend.runtime().synchronize().unwrap();
        black_box(&out);
    });
    finish_ok(
        base_record(pattern, name, total, bytes, device_name, "ok", "passed", true).with_note(
            "allocates a fresh device tensor per call (eager transpose op); destination is not reused"
                .into(),
        ),
        timing,
        sink,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_tenferro_cuda_to_contiguous(
    pattern: &PermutePattern,
    prepared: &PreparedPattern,
    warmup: usize,
    iters: usize,
    bytes: usize,
    device_name: &str,
    backend: &mut CudaBackend,
    sink: &mut RecordSink,
) {
    let total = prepared.reference.len();
    let name: &'static str = "tenferro-cuda-to-contiguous";

    let flat_host = Tensor::from_vec_col_major(vec![total], prepared.src_data.clone())
        .expect("building flat source tensor must succeed");
    let flat_gpu = upload_tensor(backend.runtime(), &flat_host).expect("upload must succeed");
    let typed: &TypedTensor<f64> = match &flat_gpu {
        Tensor::F64(t) => t,
        _ => unreachable!("upload_tensor preserves dtype"),
    };
    // Mirror the CPU runner's composition exactly: build the strided view
    // over the SOURCE layout (pattern shape + pattern source strides -- for
    // the explicit-stride pattern these are the JSON strides), then apply
    // the permutation as a metadata-only `transpose_view(perm)`, then
    // materialize. Materializing the un-permuted source view would be an
    // identity copy, not the permutation task.
    let src_view = typed
        .backend_region_view(pattern.shape.clone(), prepared.src_strides.clone(), 0)
        .expect("building a device strided view must succeed for a validated pattern");
    let view = src_view
        .transpose_view(&pattern.perm)
        .expect("transpose_view must succeed on a validated permutation");
    debug_assert_eq!(view.shape(), prepared.out_shape.as_slice());

    let compact = backend
        .to_contiguous(&view)
        .expect("tenferro-cuda-to-contiguous must succeed on a validated pattern");
    backend.runtime().synchronize().expect("device sync");
    let downloaded = download_tensor(backend.runtime(), &Tensor::F64(compact))
        .expect("download must succeed");
    let actual = downloaded
        .as_slice::<f64>()
        .expect("tenferro-cuda-to-contiguous output must be f64");
    if let Err(msg) = verify_output(actual, &prepared.reference) {
        finish_skip(
            base_record(
                pattern,
                name,
                total,
                bytes,
                device_name,
                "verification_failed",
                "failed",
                true,
            )
            .with_note(msg),
            sink,
        );
        return;
    }

    let timing = bench_n(warmup, iters, bytes, || {
        let compact = backend.to_contiguous(&view).unwrap();
        backend.runtime().synchronize().unwrap();
        black_box(&compact);
    });
    finish_ok(
        base_record(pattern, name, total, bytes, device_name, "ok", "passed", true).with_note(
            "allocates a fresh device tensor per call (view -> contiguous materialize); \
             destination is not reused"
                .into(),
        ),
        timing,
        sink,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_cutensor(
    pattern: &PermutePattern,
    prepared: &PreparedPattern,
    warmup: usize,
    iters: usize,
    bytes: usize,
    device_name: &str,
    backend: &CudaBackend,
    cutensor: Option<&CutensorLib>,
    sink: &mut RecordSink,
) {
    let total = prepared.reference.len();
    let name: &'static str = "cutensor";

    let Some(lib) = cutensor else {
        finish_skip(
            base_record(pattern, name, total, bytes, device_name, "skipped", "skipped", false)
                .with_note("cuTENSOR library unavailable (see TENFERRO_CUTENSOR_PATH)".into()),
            sink,
        );
        return;
    };

    let rank = pattern.shape.len();
    let extents_a: Vec<i64> = pattern.shape.iter().map(|&d| d as i64).collect();
    let strides_a: Vec<i64> = prepared.src_strides.iter().map(|&s| s as i64).collect();
    let modes_a: Vec<i32> = (0..rank as i32).collect();
    let extents_b: Vec<i64> = prepared.out_shape.iter().map(|&d| d as i64).collect();
    let strides_b: Vec<i64> = prepared.dst_strides.iter().map(|&s| s as i64).collect();
    let modes_b: Vec<i32> = pattern.perm.iter().map(|&p| p as i32).collect();

    let plan = match lib.build_permutation_plan(
        &extents_a, &strides_a, &modes_a, &extents_b, &strides_b, &modes_b,
    ) {
        Ok(plan) => plan,
        Err(err) => {
            finish_skip(
                base_record(pattern, name, total, bytes, device_name, "skipped", "skipped", false)
                    .with_note(format!(
                        "cuTENSOR rejected this pattern (rank {rank}), recording skip instead of \
                         crashing: {err}"
                    )),
                sink,
            );
            return;
        }
    };

    let flat_src_host = Tensor::from_vec_col_major(vec![total], prepared.src_data.clone())
        .expect("building flat source tensor must succeed");
    let flat_src = upload_tensor(backend.runtime(), &flat_src_host).expect("upload must succeed");
    let flat_dst_host = Tensor::from_vec_col_major(vec![total], vec![0.0f64; total])
        .expect("building flat destination tensor must succeed");
    let flat_dst = upload_tensor(backend.runtime(), &flat_dst_host).expect("upload must succeed");

    let src_ptr = cuda_ptr_const(device_ptr(backend.runtime(), &flat_src).expect("device_ptr"));
    let dst_ptr = cuda_ptr_mut(device_ptr(backend.runtime(), &flat_dst).expect("device_ptr"));
    let stream = raw_cuda_stream(backend.runtime(), "benchmark_gpu_permutation::cutensor")
        .expect("raw_cuda_stream") as usize as cutensor_ffi::CudaStream;

    // A successful plan build does not guarantee execution succeeds; the
    // pattern JSON's notes_gpu promise a skip record instead of a crash, so
    // an execution failure here (before any timing) emits `runtime_failed`
    // and returns rather than panicking the whole suite. The `.unwrap()`
    // inside the timed closure below may stay: a failure after a verified
    // successful execution of the identical call is exceptional.
    if let Err(err) = lib.permute(&plan, 1.0, src_ptr, dst_ptr, stream) {
        finish_skip(
            base_record(
                pattern,
                name,
                total,
                bytes,
                device_name,
                "runtime_failed",
                "skipped",
                false,
            )
            .with_note(format!(
                "cutensorPermute failed at execution despite a successful plan build \
                 (rank {rank}), recording skip instead of crashing: {err}"
            )),
            sink,
        );
        return;
    }
    backend.runtime().synchronize().expect("device sync");
    let downloaded = download_tensor(backend.runtime(), &flat_dst).expect("download must succeed");
    let actual = downloaded.as_slice::<f64>().expect("cutensor output must be f64");
    if let Err(msg) = verify_output(actual, &prepared.reference) {
        finish_skip(
            base_record(
                pattern,
                name,
                total,
                bytes,
                device_name,
                "verification_failed",
                "failed",
                false,
            )
            .with_note(msg),
            sink,
        );
        return;
    }

    let timing = bench_n(warmup, iters, bytes, || {
        lib.permute(&plan, 1.0, src_ptr, dst_ptr, stream).unwrap();
        backend.runtime().synchronize().unwrap();
        black_box(dst_ptr);
    });
    finish_ok(
        base_record(pattern, name, total, bytes, device_name, "ok", "passed", false),
        timing,
        sink,
    );
}

// ---------------------------------------------------------------------------
// Suite driver
// ---------------------------------------------------------------------------

fn selected_patterns(suite: &PatternSuite) -> Vec<&PermutePattern> {
    match env::var("PATTERN_ID") {
        Ok(id) => suite.patterns.iter().filter(|p| p.id == id).collect(),
        Err(_) => suite.patterns.iter().collect(),
    }
}

fn run_pattern(
    pattern: &PermutePattern,
    device_name: &str,
    backend: &mut CudaBackend,
    cutensor: Option<&CutensorLib>,
    sink: &mut RecordSink,
) {
    let prepared = prepare_pattern(pattern);
    let total = prepared.reference.len();
    let bytes = total * std::mem::size_of::<f64>() * 2;
    let (warmup, iters) = timing_counts_env(total);

    println!(
        "=== {} ===\n  id={} elems={} bytes(r+w)={}",
        pattern.label, pattern.id, total, bytes
    );

    for participant in &pattern.participants_gpu {
        match participant.as_str() {
            "tenferro-cuda-transpose" => run_tenferro_cuda_transpose(
                pattern, &prepared, warmup, iters, bytes, device_name, backend, sink,
            ),
            "tenferro-cuda-to-contiguous" => run_tenferro_cuda_to_contiguous(
                pattern, &prepared, warmup, iters, bytes, device_name, backend, sink,
            ),
            "cutensor" => run_cutensor(
                pattern, &prepared, warmup, iters, bytes, device_name, backend, cutensor, sink,
            ),
            // pytorch-cuda / jax-cuda / memcpy-d2d are measured by
            // scripts/benchmark_gpu_permutation_python.py.
            _ => {}
        }
    }
    println!();
}

fn gpu_name(device_ordinal: usize) -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name",
            "--format=csv,noheader",
            "-i",
            &device_ordinal.to_string(),
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            let name = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if name.is_empty() {
                None
            } else {
                Some(name)
            }
        })
        .unwrap_or_else(|| format!("cuda:{device_ordinal}"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("tenferro-benchmark GPU permutation suite (gpu/permutation)");
    println!("============================================================");
    println!("Element size: {} bytes", std::mem::size_of::<f64>());
    println!("Format: label  median_ms  (p25 / p75)  bandwidth_GB/s");
    println!("Patterns: {PATTERN_PATH}");
    if let Ok(id) = env::var("PATTERN_ID") {
        println!("Pattern filter: {id}");
    }
    if let Ok(path) = env::var("BENCH_OUTPUT") {
        println!("JSON output: {path}");
    }

    if !gpu_available() {
        eprintln!("benchmark_gpu_permutation: no CUDA GPU found, skipping");
        if let Ok(path) = env::var("BENCH_OUTPUT") {
            std::fs::write(path, "")?;
        }
        return Ok(());
    }

    let device_ordinal: usize = env::var("GPU_BENCH_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let device_name = gpu_name(device_ordinal);
    println!("Device: cuda:{device_ordinal} ({device_name})");
    println!();

    let mut backend = CudaBackend::new(device_ordinal)
        .map_err(|e| PatternError(format!("failed to create CUDA backend: {e}")))?;

    let cutensor = match CutensorLib::load() {
        Ok(lib) => Some(lib),
        Err(err) => {
            eprintln!(
                "benchmark_gpu_permutation: cuTENSOR unavailable, cutensor participants will be \
                 recorded as skipped: {err}"
            );
            None
        }
    };

    let suite = load_pattern_suite()?;
    let patterns = selected_patterns(&suite);
    if patterns.is_empty() {
        return Err(Box::new(PatternError(format!(
            "PATTERN_ID did not match any pattern in {PATTERN_PATH}"
        ))));
    }

    let mut sink = RecordSink::new()?;
    println!("--- Correctness verification and benchmarks ---");
    for pattern in patterns {
        run_pattern(pattern, &device_name, &mut backend, cutensor.as_ref(), &mut sink);
    }
    if sink.any_failed {
        return Err(Box::new(PatternError(
            "one or more participants failed the correctness gate".into(),
        )));
    }

    println!("Done.");
    Ok(())
}
