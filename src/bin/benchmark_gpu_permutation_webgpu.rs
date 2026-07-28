//! wgpu/Metal permutation baseline for the `mac-gpu` profile.
//!
//! Correctness downloads are outside timed regions. Each timed operation is
//! followed by `WebGpuBackend::synchronize`, so measurements include host
//! dispatch and device completion but not host transfer.

use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tenferro_gpu::{webgpu_available, WebGpuBackend};
use tenferro_tensor::{
    Tensor, TensorDeviceTransfer, TensorStructural, TensorViewCanonicalization, TypedTensor,
};

const PATTERN_PATH: &str = "data/instances/gpu_permutation_mac_patterns.json";
const SUITE_ID: &str = "gpu/permutation";

#[derive(Debug, Deserialize)]
struct PatternSuite {
    patterns: Vec<Pattern>,
}

#[derive(Debug, Deserialize)]
struct Pattern {
    id: String,
    label: String,
    shape: Vec<usize>,
    perm: Vec<usize>,
    src_layout: Layout,
    participants_gpu: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Layout {
    ColMajor,
    ExplicitStrides { strides: Vec<isize> },
}

#[derive(Debug, Serialize)]
struct Record {
    schema_version: u32,
    suite_id: &'static str,
    target_profile: &'static str,
    runner: &'static str,
    runtime: &'static str,
    synchronization: &'static str,
    allocation: &'static str,
    tenferro_revision: &'static str,
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
    warmup: usize,
    iters: usize,
    median_ms: Option<f64>,
    p25_ms: Option<f64>,
    p75_ms: Option<f64>,
    bandwidth_gbs: Option<f64>,
    notes: Option<String>,
}

fn col_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1] * shape[axis - 1] as isize;
    }
    strides
}

fn source_strides(pattern: &Pattern) -> Vec<isize> {
    match &pattern.src_layout {
        Layout::ColMajor => col_major_strides(&pattern.shape),
        Layout::ExplicitStrides { strides } => strides.clone(),
    }
}

fn reference(pattern: &Pattern, data: &[f32]) -> Vec<f32> {
    let out_shape: Vec<_> = pattern
        .perm
        .iter()
        .map(|&axis| pattern.shape[axis])
        .collect();
    let out_strides = col_major_strides(&out_shape);
    let input_strides = source_strides(pattern);
    let total = data.len();
    let mut output = vec![0.0; total];
    let mut index = vec![0usize; out_shape.len()];
    for value in &mut output {
        let mut src_offset = 0isize;
        for out_axis in 0..out_shape.len() {
            src_offset += index[out_axis] as isize * input_strides[pattern.perm[out_axis]];
        }
        *value = data[src_offset as usize];
        for axis in 0..out_shape.len() {
            index[axis] += 1;
            if index[axis] < out_shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    debug_assert_eq!(out_strides.last().is_some(), !out_shape.is_empty());
    output
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[index]
}

fn measure(
    warmup: usize,
    iters: usize,
    bytes: usize,
    mut operation: impl FnMut(),
) -> (f64, f64, f64, f64) {
    for _ in 0..warmup {
        operation();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        operation();
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(f64::total_cmp);
    let median = quantile(&samples, 0.5);
    (
        median,
        quantile(&samples, 0.25),
        quantile(&samples, 0.75),
        bytes as f64 / (median * 1e6),
    )
}

#[allow(clippy::too_many_arguments)]
fn record(
    pattern: &Pattern,
    backend: &'static str,
    device: &str,
    warmup: usize,
    iters: usize,
    timing: Option<(f64, f64, f64, f64)>,
    correctness: &'static str,
    notes: Option<String>,
) -> Record {
    let elems = pattern.shape.iter().product::<usize>();
    Record {
        schema_version: 1,
        suite_id: SUITE_ID,
        target_profile: "mac-gpu",
        runner: "rust",
        runtime: "wgpu/Metal",
        synchronization: "backend.synchronize after each dispatch",
        allocation: "fresh destination per timed call",
        tenferro_revision: option_env!("TENFERRO_BENCH_REVISION").unwrap_or("unknown"),
        pattern_id: pattern.id.clone(),
        label: pattern.label.clone(),
        backend,
        shape: pattern.shape.clone(),
        perm: pattern.perm.clone(),
        dtype: "f32",
        elems,
        bytes_rw: elems * std::mem::size_of::<f32>() * 2,
        device: device.to_string(),
        status: if timing.is_some() {
            "ok"
        } else {
            "verification_failed"
        },
        correctness,
        per_call_allocation: true,
        warmup,
        iters,
        median_ms: timing.map(|value| value.0),
        p25_ms: timing.map(|value| value.1),
        p75_ms: timing.map(|value| value.2),
        bandwidth_gbs: timing.map(|value| value.3),
        notes,
    }
}

fn verify(actual: &[f32], expected: &[f32]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "length mismatch: {} != {}",
            actual.len(),
            expected.len()
        ));
    }
    actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
        .map_or(Ok(()), |index| {
            Err(format!(
                "mismatch at {index}: actual={} expected={}",
                actual[index], expected[index]
            ))
        })
}

fn run_transpose(
    backend: &mut WebGpuBackend,
    pattern: &Pattern,
    data: &[f32],
    expected: &[f32],
    device: &str,
    warmup: usize,
    iters: usize,
) -> Record {
    let name = "tenferro-webgpu-transpose-baseline";
    let host = Tensor::from_vec_col_major(pattern.shape.clone(), data.to_vec()).unwrap();
    let input = backend.upload_host_tensor(&host).unwrap();
    let output = backend.transpose(&input, &pattern.perm).unwrap();
    backend.synchronize().unwrap();
    let host_output = backend.download_to_host(&output).unwrap();
    if let Err(note) = verify(host_output.as_slice::<f32>().unwrap(), expected) {
        return record(
            pattern,
            name,
            device,
            warmup,
            iters,
            None,
            "failed",
            Some(note),
        );
    }
    let bytes = std::mem::size_of_val(data) * 2;
    let timing = measure(warmup, iters, bytes, || {
        let output = backend.transpose(&input, &pattern.perm).unwrap();
        backend.synchronize().unwrap();
        black_box(output);
    });
    record(
        pattern,
        name,
        device,
        warmup,
        iters,
        Some(timing),
        "passed",
        Some("pre-optimization native structural kernel baseline".into()),
    )
}

fn run_to_contiguous(
    backend: &mut WebGpuBackend,
    pattern: &Pattern,
    data: &[f32],
    expected: &[f32],
    device: &str,
    warmup: usize,
    iters: usize,
) -> Record {
    let name = "tenferro-webgpu-to-contiguous";
    let host = Tensor::from_vec_col_major(vec![data.len()], data.to_vec()).unwrap();
    let input = backend.upload_host_tensor(&host).unwrap();
    let Tensor::F32(typed) = &input else {
        unreachable!("f32 upload must preserve dtype")
    };
    let source = typed
        .backend_region_view(pattern.shape.clone(), source_strides(pattern), 0)
        .unwrap();
    let view = source.transpose_view(&pattern.perm).unwrap();
    let output: TypedTensor<f32> = backend.to_contiguous(&view).unwrap();
    backend.synchronize().unwrap();
    let host_output = backend.download_to_host(&Tensor::F32(output)).unwrap();
    if let Err(note) = verify(host_output.as_slice::<f32>().unwrap(), expected) {
        return record(
            pattern,
            name,
            device,
            warmup,
            iters,
            None,
            "failed",
            Some(note),
        );
    }
    let bytes = std::mem::size_of_val(data) * 2;
    let timing = measure(warmup, iters, bytes, || {
        let output = backend.to_contiguous(&view).unwrap();
        backend.synchronize().unwrap();
        black_box(output);
    });
    record(
        pattern,
        name,
        device,
        warmup,
        iters,
        Some(timing),
        "passed",
        Some("view metadata construction and correctness download excluded".into()),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !webgpu_available() {
        return Err("no WebGPU adapter is available".into());
    }
    let suite: PatternSuite = serde_json::from_str(&std::fs::read_to_string(PATTERN_PATH)?)?;
    let filter = env::var("PATTERN_ID").ok();
    let warmup = env::var("BENCH_WARMUPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);
    let iters = env::var("BENCH_RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(7);
    let device = env::var("GPU_BENCH_DEVICE_NAME").unwrap_or_else(|_| "Apple GPU".into());
    let mut output = env::var("BENCH_OUTPUT")
        .ok()
        .map(File::create)
        .transpose()?;
    let mut backend = WebGpuBackend::new_default()?;
    let mut failed = false;

    for pattern in suite
        .patterns
        .iter()
        .filter(|pattern| filter.as_ref().is_none_or(|filter| filter == &pattern.id))
    {
        let total = pattern.shape.iter().product();
        let data: Vec<f32> = (0..total).map(|index| (index % 65521) as f32).collect();
        let expected = reference(pattern, &data);
        for participant in &pattern.participants_gpu {
            let record = match participant.as_str() {
                "tenferro-webgpu-transpose-baseline" => run_transpose(
                    &mut backend,
                    pattern,
                    &data,
                    &expected,
                    &device,
                    warmup,
                    iters,
                ),
                "tenferro-webgpu-to-contiguous" => run_to_contiguous(
                    &mut backend,
                    pattern,
                    &data,
                    &expected,
                    &device,
                    warmup,
                    iters,
                ),
                _ => continue,
            };
            println!(
                "{:38} {:24} {:>10}",
                record.backend,
                record.pattern_id,
                record
                    .median_ms
                    .map_or_else(|| record.status.into(), |value| format!("{value:.3} ms"))
            );
            failed |= record.status == "verification_failed";
            if let Some(output) = output.as_mut() {
                writeln!(output, "{}", serde_json::to_string(&record)?)?;
            }
        }
    }
    if failed {
        Err("one or more WebGPU correctness checks failed".into())
    } else {
        Ok(())
    }
}
