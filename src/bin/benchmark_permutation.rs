//! CPU permutation / materialize-kernel benchmark suite.
//!
//! Ported from `extern/strided-rs-benchmark-suite`'s
//! `benchmarks/strided_benchmarks/permute/permute.rs` per
//! `docs/permutation-suite.md`. Measures the cost of materializing a
//! strided/permuted `f64` tensor view into a compact column-major
//! destination and compares:
//!
//! - `naive`: Rust odometer loop (also the correctness reference)
//! - `tenferro-transpose`: eager `CpuBackend::transpose` (compact col-major
//!   input only)
//! - `tenferro-to-contiguous`: `TypedTensorView::transpose_view` +
//!   `to_contiguous()` (accepts arbitrary source strides)
//! - `hptt` (feature `hptt`, contiguous src/dst only)
//! - `strided-rs` (feature `strided-rs`; `strided_perm::copy_into` /
//!   `copy_into_col_major`, see below)
//! - `memcpy` baseline for the identity-permutation pattern
//!
//! `julia-base` / `strided-jl` participants are measured by
//! `scripts/benchmark_permutation.jl` and are ignored here.
//!
//! Every participant's output is compared elementwise against the `naive`
//! reference before any timing; a mismatch is reported and fails the process
//! (nonzero exit code) without corrupting other patterns' measurements.
//!
//! Environment variables:
//! - `PATTERN_ID`: run a single pattern id instead of the full suite.
//! - `BENCH_RUNS` / `BENCH_WARMUPS`: override the size-scaled iteration and
//!   warmup counts for every participant.
//! - `BENCH_OUTPUT`: path to write machine-readable JSON Lines results. When
//!   unset, only the human-readable table is printed to stdout.
//!
//! ## `strided-rs` backend
//!
//! The `strided-rs` participant (feature `strided-rs`) depends on
//! `extern/strided-rs`'s `strided-perm` and `strided-view` crates as optional
//! path dependencies (see `scripts/setup_extern_deps.sh`, which clones
//! `extern/strided-rs` alongside `extern/tenferro-rs`). It builds a
//! `strided_view::StridedArray` directly from the pattern's source strides,
//! permutes it with `.view().permute(perm)`, and times both
//! `strided_perm::copy_into` and `copy_into_col_major` into a destination
//! `StridedArray` allocated once and reused across iterations
//! (`per_call_allocation: false`). The `strided-rs` feature also enables
//! `strided-perm/parallel`, so `copy_into_par` / `copy_into_col_major_par`
//! are timed too; whichever of the (up to four) variants is fastest is
//! reported as the single `strided-rs` cell, with the winning variant name
//! recorded in the JSONL `notes` field.

use std::env;
use std::fmt;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use tenferro_cpu::CpuBackend;
use tenferro_tensor::{Tensor, TensorStructural, TypedTensorView};

const PATTERN_PATH: &str = "data/instances/permutation_patterns.json";
const SUITE_ID: &str = "cpu/permutation";

// ---------------------------------------------------------------------------
// Pattern schema (mirrors data/instances/permutation_patterns.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PatternSuite {
    version: u32,
    index_base: u32,
    semantics: String,
    data: String,
    patterns: Vec<PermutePattern>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PermutePattern {
    id: String,
    label: String,
    dtype: String,
    shape: Vec<usize>,
    perm: Vec<usize>,
    src_layout: LayoutPattern,
    dst_layout: LayoutPattern,
    participants: Vec<Participant>,
    #[allow(dead_code)]
    notes: Option<String>,
    // The following two fields are consumed by the GPU permutation runner
    // (src/bin/benchmark_gpu_permutation.rs, docs/gpu-permutation-suite.md);
    // they are `#[serde(default)]` here purely so this CPU runner keeps
    // parsing the shared pattern file under `deny_unknown_fields`.
    #[allow(dead_code)]
    #[serde(default)]
    notes_gpu: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    participants_gpu: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LayoutPattern {
    ColMajor,
    ExplicitStrides { strides: Vec<isize> },
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Participant {
    Naive,
    TenferroTranspose,
    TenferroToContiguous,
    Hptt,
    StridedRs,
    JuliaBase,
    StridedJl,
    Memcpy,
}

impl Participant {
    fn as_str(self) -> &'static str {
        match self {
            Participant::Naive => "naive",
            Participant::TenferroTranspose => "tenferro-transpose",
            Participant::TenferroToContiguous => "tenferro-to-contiguous",
            Participant::Hptt => "hptt",
            Participant::StridedRs => "strided-rs",
            Participant::JuliaBase => "julia-base",
            Participant::StridedJl => "strided-jl",
            Participant::Memcpy => "memcpy",
        }
    }

    /// Whether this participant is measured by this Rust runner (as opposed
    /// to `scripts/benchmark_permutation.jl`).
    fn is_rust_participant(self) -> bool {
        !matches!(self, Participant::JuliaBase | Participant::StridedJl)
    }
}

#[derive(Debug)]
struct PatternError(String);

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PatternError {}

fn load_pattern_suite_from_str(json: &str) -> Result<PatternSuite, Box<dyn std::error::Error>> {
    let suite: PatternSuite = serde_json::from_str(json)?;
    validate_pattern_suite(&suite)?;
    Ok(suite)
}

fn validate_pattern_suite(suite: &PatternSuite) -> Result<(), PatternError> {
    if suite.version != 1 {
        return Err(PatternError(format!(
            "unsupported pattern schema version {}",
            suite.version
        )));
    }
    if suite.index_base != 0 {
        return Err(PatternError(
            "permute patterns must use index_base = 0".into(),
        ));
    }
    if suite.semantics != "out[i0,...,ik] = src[i_perm0,...,i_permk]" {
        return Err(PatternError(format!(
            "unsupported semantics {:?}",
            suite.semantics
        )));
    }
    if suite.data != "deterministic_index_value" {
        return Err(PatternError(format!(
            "unsupported data mode {:?}",
            suite.data
        )));
    }
    for pattern in &suite.patterns {
        validate_pattern(pattern)?;
    }
    Ok(())
}

fn validate_pattern(pattern: &PermutePattern) -> Result<(), PatternError> {
    if pattern.dtype != "f64" {
        return Err(PatternError(format!(
            "{} uses unsupported dtype {:?}",
            pattern.id, pattern.dtype
        )));
    }
    if pattern.shape.is_empty() {
        return Err(PatternError(format!("{} has empty shape", pattern.id)));
    }
    if pattern.shape.len() != pattern.perm.len() {
        return Err(PatternError(format!(
            "{} shape rank {} != perm rank {}",
            pattern.id,
            pattern.shape.len(),
            pattern.perm.len()
        )));
    }
    let mut seen = vec![false; pattern.perm.len()];
    for &axis in &pattern.perm {
        if axis >= pattern.perm.len() {
            return Err(PatternError(format!(
                "{} perm axis {} is out of range for rank {}",
                pattern.id,
                axis,
                pattern.perm.len()
            )));
        }
        if seen[axis] {
            return Err(PatternError(format!(
                "{} perm axis {} appears more than once",
                pattern.id, axis
            )));
        }
        seen[axis] = true;
    }
    validate_layout(
        &pattern.id,
        "src_layout",
        &pattern.src_layout,
        pattern.shape.len(),
    )?;
    validate_layout(
        &pattern.id,
        "dst_layout",
        &pattern.dst_layout,
        pattern.shape.len(),
    )?;
    Ok(())
}

fn validate_layout(
    id: &str,
    name: &str,
    layout: &LayoutPattern,
    rank: usize,
) -> Result<(), PatternError> {
    if let LayoutPattern::ExplicitStrides { strides } = layout {
        if strides.len() != rank {
            return Err(PatternError(format!(
                "{id} {name} stride rank {} != shape rank {rank}",
                strides.len()
            )));
        }
    }
    Ok(())
}

fn load_pattern_suite() -> Result<PatternSuite, Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string(PATTERN_PATH)?;
    load_pattern_suite_from_str(&json)
}

// ---------------------------------------------------------------------------
// Layout / data helpers
// ---------------------------------------------------------------------------

fn col_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }
    strides
}

fn layout_strides(pattern: &PermutePattern, layout: &LayoutPattern) -> Vec<isize> {
    match layout {
        LayoutPattern::ColMajor => col_major_strides(&pattern.shape),
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

/// Generic strided copy using odometer iteration. Also the correctness
/// reference and the `naive` benchmark participant.
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
    src_perm_strides: Vec<isize>,
    dst_strides: Vec<isize>,
    reference: Vec<f64>,
}

fn prepare_pattern(pattern: &PermutePattern) -> PreparedPattern {
    let src_strides = layout_strides(pattern, &pattern.src_layout);
    let total: usize = pattern.shape.iter().product();
    let src_data = deterministic_data(total);

    // Permuted source strides: src_perm_strides[k] = src_strides[perm[k]].
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
        src_perm_strides,
        dst_strides,
        reference,
    }
}

// ---------------------------------------------------------------------------
// Timing
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

/// The actual size of tenferro's global rayon pool. Rayon's global pool is
/// lazily initialized from `RAYON_NUM_THREADS` (falling back to
/// `std::thread::available_parallelism()`) on first use and is shared by the
/// whole process, so this reflects the same thread count tenferro-cpu's
/// parallel kernels use — not gated behind any Cargo feature.
fn current_thread_count() -> usize {
    rayon::current_num_threads()
}

// ---------------------------------------------------------------------------
// Result records
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
    threads: usize,
    status: &'static str,
    correctness: &'static str,
    per_call_allocation: bool,
    warmup: Option<usize>,
    iters: Option<usize>,
    median_ms: Option<f64>,
    p25_ms: Option<f64>,
    p75_ms: Option<f64>,
    gbps: Option<f64>,
    notes: Option<String>,
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

fn print_human_row(pattern: &PermutePattern, backend: &str, timing: Option<&Timing>, note: Option<&str>) {
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
    let _ = pattern;
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
// Participant execution
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_participant(
    pattern: &PermutePattern,
    prepared: &PreparedPattern,
    participant: Participant,
    warmup: usize,
    iters: usize,
    bytes: usize,
    threads: usize,
    sink: &mut RecordSink,
) {
    let total = prepared.reference.len();
    let base = |status: &'static str, correctness: &'static str, per_call_allocation: bool| ResultRecord {
        schema_version: 1,
        suite_id: SUITE_ID,
        runner: "rust",
        pattern_id: pattern.id.clone(),
        label: pattern.label.clone(),
        backend: participant.as_str(),
        shape: pattern.shape.clone(),
        perm: pattern.perm.clone(),
        dtype: "f64",
        elems: total,
        bytes_rw: bytes,
        threads,
        status,
        correctness,
        per_call_allocation,
        warmup: None,
        iters: None,
        median_ms: None,
        p25_ms: None,
        p75_ms: None,
        gbps: None,
        notes: None,
    };

    macro_rules! finish {
        ($record:expr, $timing:expr) => {{
            let mut record = $record;
            if let Some(timing) = $timing {
                record.warmup = Some(timing.warmup);
                record.iters = Some(timing.iters);
                record.median_ms = Some(timing.median_ms);
                record.p25_ms = Some(timing.p25_ms);
                record.p75_ms = Some(timing.p75_ms);
                record.gbps = Some(timing.gbps);
                print_human_row(pattern, participant.as_str(), Some(&timing), None);
            } else {
                print_human_row(pattern, participant.as_str(), None, record.notes.as_deref());
            }
            sink.emit(&record);
        }};
    }

    match participant {
        Participant::Naive => {
            let mut dst = vec![0.0f64; total];
            unsafe {
                naive_strided_copy(
                    prepared.src_data.as_ptr(),
                    dst.as_mut_ptr(),
                    &prepared.out_shape,
                    &prepared.src_perm_strides,
                    &prepared.dst_strides,
                );
            }
            if let Err(msg) = verify_output(&dst, &prepared.reference) {
                finish!(base("verification_failed", "failed", false).with_note(msg), None::<Timing>);
                return;
            }
            let timing = bench_n(warmup, iters, bytes, || {
                unsafe {
                    naive_strided_copy(
                        prepared.src_data.as_ptr(),
                        dst.as_mut_ptr(),
                        &prepared.out_shape,
                        &prepared.src_perm_strides,
                        &prepared.dst_strides,
                    )
                };
                black_box(dst.as_ptr());
            });
            finish!(base("ok", "passed", false), Some(timing));
        }
        Participant::Memcpy => {
            let identity = pattern.perm.iter().copied().eq(0..pattern.perm.len());
            let contiguous = matches!(pattern.src_layout, LayoutPattern::ColMajor)
                && matches!(pattern.dst_layout, LayoutPattern::ColMajor);
            if !identity || !contiguous {
                finish!(
                    base("skipped", "skipped", false)
                        .with_note("requires identity col-major pattern".into()),
                    None::<Timing>
                );
                return;
            }
            let mut dst = vec![0.0f64; total];
            unsafe {
                std::ptr::copy_nonoverlapping(prepared.src_data.as_ptr(), dst.as_mut_ptr(), total);
            }
            if let Err(msg) = verify_output(&dst, &prepared.reference) {
                finish!(base("verification_failed", "failed", false).with_note(msg), None::<Timing>);
                return;
            }
            let timing = bench_n(warmup, iters, bytes, || {
                unsafe {
                    std::ptr::copy_nonoverlapping(prepared.src_data.as_ptr(), dst.as_mut_ptr(), total)
                };
                black_box(dst.as_ptr());
            });
            finish!(base("ok", "passed", false), Some(timing));
        }
        Participant::TenferroTranspose => {
            if !matches!(pattern.src_layout, LayoutPattern::ColMajor) {
                finish!(
                    base("skipped", "skipped", true).with_note(
                        "tenferro-transpose (eager op) requires a compact col-major source"
                            .into()
                    ),
                    None::<Timing>
                );
                return;
            }
            let tensor = Tensor::from_vec_col_major(pattern.shape.clone(), prepared.src_data.clone())
                .expect("building source tensor must succeed");
            let mut backend = CpuBackend::new();
            let out = backend
                .transpose(&tensor, &pattern.perm)
                .expect("tenferro-transpose must succeed on a validated pattern");
            let actual = out
                .as_slice::<f64>()
                .expect("tenferro-transpose output must be f64");
            if let Err(msg) = verify_output(actual, &prepared.reference) {
                finish!(base("verification_failed", "failed", true).with_note(msg), None::<Timing>);
                return;
            }
            let timing = bench_n(warmup, iters, bytes, || {
                let out = backend.transpose(&tensor, &pattern.perm).unwrap();
                black_box(out.as_slice::<f64>().unwrap().as_ptr());
            });
            finish!(base("ok", "passed", true), Some(timing));
        }
        Participant::TenferroToContiguous => {
            let view = TypedTensorView::<f64>::from_slice(
                pattern.shape.clone(),
                prepared.src_strides.clone(),
                0,
                &prepared.src_data,
            )
            .expect("building tenferro source view must succeed");
            let transposed = view
                .transpose_view(&pattern.perm)
                .expect("transpose_view must succeed on a validated permutation");
            let compact = transposed
                .to_contiguous()
                .expect("to_contiguous must succeed");
            let actual = compact
                .as_slice()
                .expect("to_contiguous output must be host-contiguous");
            if let Err(msg) = verify_output(actual, &prepared.reference) {
                finish!(base("verification_failed", "failed", true).with_note(msg), None::<Timing>);
                return;
            }
            let timing = bench_n(warmup, iters, bytes, || {
                let compact = transposed.to_contiguous().unwrap();
                black_box(compact.as_slice().unwrap().as_ptr());
            });
            finish!(base("ok", "passed", true), Some(timing));
        }
        Participant::Hptt => run_hptt_participant(pattern, prepared, warmup, iters, bytes, threads, sink),
        Participant::StridedRs => {
            run_strided_rs_participant(pattern, prepared, warmup, iters, bytes, threads, sink)
        }
        Participant::JuliaBase | Participant::StridedJl => {
            // Measured by scripts/benchmark_permutation.jl.
        }
    }
}

impl ResultRecord {
    fn with_note(mut self, note: String) -> Self {
        self.notes = Some(note);
        self
    }
}

#[cfg(feature = "hptt")]
fn run_hptt_participant(
    pattern: &PermutePattern,
    prepared: &PreparedPattern,
    warmup: usize,
    iters: usize,
    bytes: usize,
    threads: usize,
    sink: &mut RecordSink,
) {
    let total = prepared.reference.len();
    let base = |status: &'static str, correctness: &'static str| ResultRecord {
        schema_version: 1,
        suite_id: SUITE_ID,
        runner: "rust",
        pattern_id: pattern.id.clone(),
        label: pattern.label.clone(),
        backend: Participant::Hptt.as_str(),
        shape: pattern.shape.clone(),
        perm: pattern.perm.clone(),
        dtype: "f64",
        elems: total,
        bytes_rw: bytes,
        threads,
        status,
        correctness,
        per_call_allocation: false,
        warmup: None,
        iters: None,
        median_ms: None,
        p25_ms: None,
        p75_ms: None,
        gbps: None,
        notes: None,
    };

    if !matches!(pattern.src_layout, LayoutPattern::ColMajor)
        || !matches!(pattern.dst_layout, LayoutPattern::ColMajor)
    {
        let record = base("skipped", "skipped")
            .with_note("hptt requires contiguous source and destination".into());
        print_human_row(pattern, "hptt", None, record.notes.as_deref());
        sink.emit(&record);
        return;
    }

    let mut dst = vec![0.0f64; total];
    hptt::transpose_f64(
        &pattern.perm,
        1.0,
        &prepared.src_data,
        &pattern.shape,
        0.0,
        &mut dst,
        1,
        hptt::MemoryOrder::ColumnMajor,
    )
    .expect("hptt correctness run must succeed");
    if let Err(msg) = verify_output(&dst, &prepared.reference) {
        let record = base("verification_failed", "failed").with_note(msg);
        print_human_row(pattern, "hptt", None, record.notes.as_deref());
        sink.emit(&record);
        return;
    }

    let timing = bench_n(warmup, iters, bytes, || {
        hptt::transpose_f64(
            &pattern.perm,
            1.0,
            &prepared.src_data,
            &pattern.shape,
            0.0,
            &mut dst,
            threads,
            hptt::MemoryOrder::ColumnMajor,
        )
        .unwrap();
        black_box(dst.as_ptr());
    });
    print_human_row(pattern, "hptt", Some(&timing), None);
    let mut record = base("ok", "passed");
    record.warmup = Some(timing.warmup);
    record.iters = Some(timing.iters);
    record.median_ms = Some(timing.median_ms);
    record.p25_ms = Some(timing.p25_ms);
    record.p75_ms = Some(timing.p75_ms);
    record.gbps = Some(timing.gbps);
    sink.emit(&record);
}

#[cfg(not(feature = "hptt"))]
fn run_hptt_participant(
    pattern: &PermutePattern,
    prepared: &PreparedPattern,
    _warmup: usize,
    _iters: usize,
    bytes: usize,
    threads: usize,
    sink: &mut RecordSink,
) {
    let total = prepared.reference.len();
    let record = ResultRecord {
        schema_version: 1,
        suite_id: SUITE_ID,
        runner: "rust",
        pattern_id: pattern.id.clone(),
        label: pattern.label.clone(),
        backend: Participant::Hptt.as_str(),
        shape: pattern.shape.clone(),
        perm: pattern.perm.clone(),
        dtype: "f64",
        elems: total,
        bytes_rw: bytes,
        threads,
        status: "skipped",
        correctness: "skipped",
        per_call_allocation: false,
        warmup: None,
        iters: None,
        median_ms: None,
        p25_ms: None,
        p75_ms: None,
        gbps: None,
        notes: Some("rebuild with --features hptt".into()),
    };
    print_human_row(pattern, "hptt", None, record.notes.as_deref());
    sink.emit(&record);
}

#[cfg(feature = "strided-rs")]
fn run_strided_rs_participant(
    pattern: &PermutePattern,
    prepared: &PreparedPattern,
    warmup: usize,
    iters: usize,
    bytes: usize,
    threads: usize,
    sink: &mut RecordSink,
) {
    let total = prepared.reference.len();
    let base = |status: &'static str, correctness: &'static str| ResultRecord {
        schema_version: 1,
        suite_id: SUITE_ID,
        runner: "rust",
        pattern_id: pattern.id.clone(),
        label: pattern.label.clone(),
        backend: Participant::StridedRs.as_str(),
        shape: pattern.shape.clone(),
        perm: pattern.perm.clone(),
        dtype: "f64",
        elems: total,
        bytes_rw: bytes,
        threads,
        status,
        correctness,
        per_call_allocation: false,
        warmup: None,
        iters: None,
        median_ms: None,
        p25_ms: None,
        p75_ms: None,
        gbps: None,
        notes: None,
    };

    let src_arr = strided_view::StridedArray::from_parts(
        prepared.src_data.clone(),
        &pattern.shape,
        &prepared.src_strides,
        0,
    )
    .expect("building strided-rs source array must succeed");
    let src_perm = src_arr
        .view()
        .permute(&pattern.perm)
        .expect("strided-rs permute must succeed on a validated pattern");
    let mut dst = strided_view::StridedArray::<f64>::col_major(&prepared.out_shape);

    // Same function-pointer signature once monomorphized at T = f64; the
    // `Send + Sync` bounds on the `_par` variants are compile-time only.
    type CopyFn = fn(
        &mut strided_view::StridedViewMut<f64>,
        &strided_view::StridedView<f64>,
    ) -> strided_view::Result<()>;

    // The `strided-rs` Cargo feature always enables strided-perm's own
    // `parallel` feature (see Cargo.toml), so the `_par` variants are always
    // linked in here; whether they win a given pattern/thread-count depends
    // on the actual rayon pool size (`threads`) at runtime.
    let candidates: [(&'static str, CopyFn); 4] = [
        ("copy_into", strided_perm::copy_into as CopyFn),
        ("copy_into_col_major", strided_perm::copy_into_col_major as CopyFn),
        ("copy_into_par", strided_perm::copy_into_par as CopyFn),
        ("copy_into_col_major_par", strided_perm::copy_into_col_major_par as CopyFn),
    ];

    // Correctness gate before any timing: run each variant once and keep
    // only the ones whose output matches the naive reference.
    let mut verified: Vec<(&'static str, CopyFn)> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    for (name, f) in candidates {
        f(&mut dst.view_mut(), &src_perm).expect("strided-rs correctness run must succeed");
        match verify_output(dst.data(), &prepared.reference) {
            Ok(()) => verified.push((name, f)),
            Err(msg) => failures.push(format!("{name}: {msg}")),
        }
    }

    if verified.is_empty() {
        let record = base("verification_failed", "failed").with_note(failures.join("; "));
        print_human_row(pattern, "strided-rs", None, record.notes.as_deref());
        sink.emit(&record);
        return;
    }

    // Time every correct variant against the reused destination and keep
    // the fastest as the single reported `strided-rs` cell.
    let mut best: Option<(&'static str, Timing)> = None;
    for (name, f) in verified.iter().copied() {
        let timing = bench_n(warmup, iters, bytes, || {
            f(&mut dst.view_mut(), &src_perm).unwrap();
            black_box(dst.data().as_ptr());
        });
        if best.as_ref().is_none_or(|(_, b)| timing.median_ms < b.median_ms) {
            best = Some((name, timing));
        }
    }
    let (winner, timing) = best.expect("verified is non-empty");

    let note = if verified.len() > 1 {
        format!("fastest of {}: {winner}", verified.len())
    } else {
        format!("fastest: {winner} (only correct variant)")
    };
    print_human_row(pattern, "strided-rs", Some(&timing), None);
    let mut record = base("ok", "passed");
    record.warmup = Some(timing.warmup);
    record.iters = Some(timing.iters);
    record.median_ms = Some(timing.median_ms);
    record.p25_ms = Some(timing.p25_ms);
    record.p75_ms = Some(timing.p75_ms);
    record.gbps = Some(timing.gbps);
    record.notes = Some(note);
    sink.emit(&record);
}

#[cfg(not(feature = "strided-rs"))]
fn run_strided_rs_participant(
    pattern: &PermutePattern,
    prepared: &PreparedPattern,
    _warmup: usize,
    _iters: usize,
    bytes: usize,
    threads: usize,
    sink: &mut RecordSink,
) {
    let total = prepared.reference.len();
    let record = ResultRecord {
        schema_version: 1,
        suite_id: SUITE_ID,
        runner: "rust",
        pattern_id: pattern.id.clone(),
        label: pattern.label.clone(),
        backend: Participant::StridedRs.as_str(),
        shape: pattern.shape.clone(),
        perm: pattern.perm.clone(),
        dtype: "f64",
        elems: total,
        bytes_rw: bytes,
        threads,
        status: "skipped",
        correctness: "skipped",
        per_call_allocation: false,
        warmup: None,
        iters: None,
        median_ms: None,
        p25_ms: None,
        p75_ms: None,
        gbps: None,
        notes: Some("rebuild with --features strided-rs".into()),
    };
    print_human_row(pattern, "strided-rs", None, record.notes.as_deref());
    sink.emit(&record);
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

fn run_pattern(pattern: &PermutePattern, sink: &mut RecordSink) {
    let prepared = prepare_pattern(pattern);
    let total = prepared.reference.len();
    let bytes = total * std::mem::size_of::<f64>() * 2;
    let (warmup, iters) = timing_counts_env(total);
    let threads = current_thread_count();

    println!(
        "=== {} ===\n  id={} elems={} bytes(r+w)={}",
        pattern.label, pattern.id, total, bytes
    );

    for &participant in &pattern.participants {
        if !participant.is_rust_participant() {
            continue;
        }
        run_participant(pattern, &prepared, participant, warmup, iters, bytes, threads, sink);
    }
    println!();
}

fn run_patterns(suite: &PatternSuite) -> Result<(), Box<dyn std::error::Error>> {
    let patterns = selected_patterns(suite);
    if patterns.is_empty() {
        return Err(Box::new(PatternError(format!(
            "PATTERN_ID did not match any pattern in {PATTERN_PATH}"
        ))));
    }

    let mut sink = RecordSink::new()?;
    println!("--- Correctness verification and benchmarks ---");
    for pattern in patterns {
        run_pattern(pattern, &mut sink);
    }
    if sink.any_failed {
        return Err(Box::new(PatternError(
            "one or more participants failed the correctness gate".into(),
        )));
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("tenferro-benchmark permutation suite (cpu/permutation)");
    println!("========================================================");
    println!("Element size: {} bytes", std::mem::size_of::<f64>());
    println!("Threads: {}", current_thread_count());
    println!("Format: label  median_ms  (p25 / p75)  bandwidth_GB/s");
    println!("Patterns: {PATTERN_PATH}");
    if let Ok(id) = env::var("PATTERN_ID") {
        println!("Pattern filter: {id}");
    }
    if let Ok(path) = env::var("BENCH_OUTPUT") {
        println!("JSON output: {path}");
    }
    println!();

    let suite = load_pattern_suite()?;
    run_patterns(&suite)?;

    println!("Done.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_patterns_are_valid() {
        let suite =
            load_pattern_suite_from_str(include_str!("../../data/instances/permutation_patterns.json"))
                .unwrap();
        assert!(suite.patterns.iter().any(|p| p.id == "transpose_2d_1024"));
        assert!(suite
            .patterns
            .iter()
            .any(|p| p.id == "tn_light_415_24d_scattered_to_colmajor"));
    }

    #[test]
    fn tenferro_columns_match_naive_reference() {
        let suite =
            load_pattern_suite_from_str(include_str!("../../data/instances/permutation_patterns.json"))
                .unwrap();
        for pattern in &suite.patterns {
            let prepared = prepare_pattern(pattern);

            if matches!(pattern.src_layout, LayoutPattern::ColMajor) {
                let tensor =
                    Tensor::from_vec_col_major(pattern.shape.clone(), prepared.src_data.clone())
                        .unwrap();
                let mut backend = CpuBackend::new();
                let out = backend.transpose(&tensor, &pattern.perm).unwrap();
                let actual = out.as_slice::<f64>().unwrap();
                assert_eq!(actual, prepared.reference.as_slice(), "{}", pattern.id);
            }

            let view = TypedTensorView::<f64>::from_slice(
                pattern.shape.clone(),
                prepared.src_strides.clone(),
                0,
                &prepared.src_data,
            )
            .unwrap();
            let compact = view.transpose_view(&pattern.perm).unwrap().to_contiguous().unwrap();
            assert_eq!(compact.as_slice().unwrap(), prepared.reference.as_slice(), "{}", pattern.id);
        }
    }
}
