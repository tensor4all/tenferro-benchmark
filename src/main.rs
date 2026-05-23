//! Benchmark runner for tenferro-einsum using einsum benchmark instances.
//!
//! Loads JSON metadata from data/instances/, builds ContractionTree from
//! pre-computed paths, compiles the compute graph once, and times execution.

use std::hint::black_box;
use std::panic;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::Deserialize;
use tenferro::exec::eval_exec_ir;
use tenferro_einsum::{ContractionTree, Subscripts};
use tenferro_einsum_benchmark::{compile_einsum, reorder_user_operands, unwrap_eval_result};
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{Tensor, TypedTensor};

// ---------------------------------------------------------------------------
// JSON schema
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct BenchmarkInstance {
    name: String,
    format_string_colmajor: String,
    shapes_colmajor: Vec<Vec<usize>>,
    dtype: String,
    num_tensors: usize,
    paths: PathInfo,
}

#[derive(Deserialize)]
struct PathInfo {
    opt_size: PathMeta,
    opt_flops: PathMeta,
}

#[derive(Deserialize)]
struct PathMeta {
    path: Vec<[usize; 2]>,
    log2_size: f64,
    log10_flops: f64,
}

// ---------------------------------------------------------------------------
// Contraction path conversion
// ---------------------------------------------------------------------------

/// Convert opt_einsum/cotengra path format to tenferro ContractionTree pairs.
fn path_to_pairs(n_inputs: usize, path: &[[usize; 2]]) -> Vec<(usize, usize)> {
    let mut available: Vec<usize> = (0..n_inputs).collect();
    let mut pairs = Vec::with_capacity(path.len());

    for (step_idx, &pair) in path.iter().enumerate() {
        let (i, j) = if pair[0] < pair[1] {
            (pair[0], pair[1])
        } else {
            (pair[1], pair[0])
        };
        let abs_j = available[j];
        let abs_i = available[i];
        pairs.push((abs_i, abs_j));

        available.remove(j);
        available.remove(i);
        let intermediate_idx = n_inputs + step_idx;
        available.push(intermediate_idx);
    }

    pairs
}

// ---------------------------------------------------------------------------
// Tensor creation
// ---------------------------------------------------------------------------

fn create_operand_tensors(shapes: &[Vec<usize>]) -> Vec<Tensor> {
    shapes
        .iter()
        .map(|shape| Tensor::F64(TypedTensor::<f64>::zeros(shape.clone())))
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

fn run_instance(
    instance: &BenchmarkInstance,
    path_meta: &PathMeta,
) -> Result<(Duration, Duration, Duration), String> {
    if instance.dtype == "complex128" {
        return Err("complex128 not supported".into());
    }

    let subs = Subscripts::parse(&instance.format_string_colmajor).map_err(|e| format!("{e}"))?;
    let shapes: Vec<&[usize]> = instance
        .shapes_colmajor
        .iter()
        .map(|s| s.as_slice())
        .collect();
    let pairs = path_to_pairs(instance.num_tensors, &path_meta.path);
    let tree = ContractionTree::from_pairs(&subs, &shapes, &pairs).map_err(|e| format!("{e}"))?;

    // Compile once
    let t_compile_start = Instant::now();
    let compiled = compile_einsum(&subs, &instance.shapes_colmajor, &tree)?;
    let compile_time = t_compile_start.elapsed();

    let mut backend = CpuBackend::new();

    // Warmup (execution only, graph already compiled)
    // Use catch_unwind to handle panics from unsupported layouts
    for _ in 0..3 {
        let operands = reorder_user_operands(
            &compiled.input_keys,
            create_operand_tensors(&instance.shapes_colmajor),
        )?;
        let exec_ref = &compiled.exec;
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            eval_exec_ir(&mut backend, exec_ref, operands)
        }));
        let _ = unwrap_eval_result(result, "panic during execution (unsupported layout?)")?;
    }

    // Timed runs
    let num_runs = 15;
    let mut durations = Vec::with_capacity(num_runs);
    for _ in 0..num_runs {
        let operands = reorder_user_operands(
            &compiled.input_keys,
            create_operand_tensors(&instance.shapes_colmajor),
        )?;
        let exec_ref = &compiled.exec;
        let t0 = Instant::now();
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            eval_exec_ir(&mut backend, exec_ref, operands)
        }));
        let elapsed = t0.elapsed();
        let eval = unwrap_eval_result(result, "panic during execution (unsupported layout?)")?;
        black_box(&eval);
        durations.push(elapsed);
    }

    durations.sort();
    let median = durations[num_runs / 2];
    let q1 = durations[num_runs / 4];
    let q3 = durations[3 * num_runs / 4];
    let iqr = q3.saturating_sub(q1);
    Ok((median, iqr, compile_time))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const BACKEND_NAME: &str = "tenferro-einsum";

fn load_instances() -> Vec<BenchmarkInstance> {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/instances");
    let mut paths: Vec<_> = std::fs::read_dir(&data_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", data_dir.display()))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    paths.sort();

    paths
        .iter()
        .filter_map(|path| {
            let json_str = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Warning: skip {} (read failed: {e})", path.display());
                    return None;
                }
            };
            match serde_json::from_str(&json_str) {
                Ok(instance) => Some(instance),
                Err(e) => {
                    eprintln!("Warning: skip {} (parse failed: {e})", path.display());
                    None
                }
            }
        })
        .collect()
}

fn main() {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/instances");
    let mut instances = load_instances();
    if let Ok(filter) = std::env::var("BENCH_INSTANCE") {
        instances.retain(|i| i.name == filter);
        if instances.is_empty() {
            eprintln!("BENCH_INSTANCE={filter:?}: no matching instance found");
            std::process::exit(1);
        }
    }

    let rayon_threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into());
    let omp_threads = std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "unset".into());

    println!("{BACKEND_NAME} benchmark suite");
    println!("==================================");
    println!(
        "Loaded {} instances from {}",
        instances.len(),
        data_dir.display()
    );
    println!("Backend: {BACKEND_NAME}");
    println!("RAYON_NUM_THREADS={rayon_threads}, OMP_NUM_THREADS={omp_threads}");
    println!("Timing: median ± IQR of 15 runs (3 warmup), graph compiled once");

    let strategies: &[(&str, fn(&PathInfo) -> &PathMeta)] = &[
        ("opt_flops", |p| &p.opt_flops),
        ("opt_size", |p| &p.opt_size),
    ];

    for &(strategy_name, get_path) in strategies {
        println!();
        println!("Strategy: {strategy_name}");
        println!(
            "{:<50} {:>8} {:>10} {:>12} {:>12} {:>10} {:>12}",
            "Instance",
            "Tensors",
            "log10FLOPS",
            "log2SIZE",
            "Median (ms)",
            "IQR (ms)",
            "Compile (ms)"
        );
        println!("{}", "-".repeat(120));

        for (i, instance) in instances.iter().enumerate() {
            eprintln!("  [{}/{}] {}...", i + 1, instances.len(), instance.name);
            let path_meta = get_path(&instance.paths);
            match run_instance(instance, path_meta) {
                Ok((median, iqr, compile_time)) => {
                    println!(
                        "{:<50} {:>8} {:>10.2} {:>12.2} {:>12.3} {:>10.3} {:>12.3}",
                        instance.name,
                        instance.num_tensors,
                        path_meta.log10_flops,
                        path_meta.log2_size,
                        median.as_secs_f64() * 1e3,
                        iqr.as_secs_f64() * 1e3,
                        compile_time.as_secs_f64() * 1e3,
                    );
                }
                Err(e) => {
                    eprintln!("  -> {} (error: {e})", instance.name);
                    println!(
                        "{:<50} {:>8} {:>10.2} {:>12.2} {:>12} {:>10} {:>12}",
                        instance.name,
                        instance.num_tensors,
                        path_meta.log10_flops,
                        path_meta.log2_size,
                        "SKIP",
                        "-",
                        "-",
                    );
                }
            }
        }
    }
}
