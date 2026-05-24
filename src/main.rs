//! Benchmark runner for tenferro-einsum using einsum benchmark instances.
//!
//! Loads JSON metadata from data/instances/, builds ContractionTree from
//! pre-computed paths, and times trace-mode or eager-mode execution.

use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::panic;
use std::path::Path;
use std::time::{Duration, Instant};

use serde::Deserialize;
use tenferro::{EagerRuntime, EagerTensor, GraphExecutor, TracedTensor};
use tenferro_einsum::eager_tensor;
use tenferro_einsum::{ContractionTree, Subscripts};
use tenferro_einsum_benchmark::{compile_einsum, unwrap_eval_result};
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

fn create_eager_operands(
    shapes: &[Vec<usize>],
    ctx: &std::sync::Arc<EagerRuntime>,
) -> Vec<EagerTensor> {
    create_operand_tensors(shapes)
        .into_iter()
        .map(|tensor| EagerTensor::from_tensor_in(tensor, std::sync::Arc::clone(ctx)))
        .collect()
}

fn bind_operands<'a>(
    inputs: &'a [TracedTensor],
    operands: &'a [Tensor],
) -> Result<Vec<(&'a TracedTensor, &'a Tensor)>, String> {
    if inputs.len() != operands.len() {
        return Err(format!(
            "operand count mismatch: graph expects {} inputs but runner created {}",
            inputs.len(),
            operands.len()
        ));
    }
    Ok(inputs.iter().zip(operands.iter()).collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TenferroMode {
    Trace,
    Eager,
}

impl TenferroMode {
    fn from_env() -> Self {
        match std::env::var("TENFERRO_MODE")
            .unwrap_or_else(|_| "trace".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "eager" => Self::Eager,
            "trace" | "traced" | "compile" | "compiled" => Self::Trace,
            other => {
                eprintln!("Unknown TENFERRO_MODE={other:?}; using trace");
                Self::Trace
            }
        }
    }

    fn backend_name(self) -> &'static str {
        match self {
            Self::Trace => "tenferro-trace",
            Self::Eager => "tenferro-eager",
        }
    }

    fn timing_note(self) -> &'static str {
        match self {
            Self::Trace => "graph compiled once",
            Self::Eager => "precomputed path, eager binary contractions",
        }
    }
}

fn append_unique(labels: &mut Vec<u32>, label: u32) {
    if !labels.contains(&label) {
        labels.push(label);
    }
}

fn intermediate_output_labels(
    subscripts: &[Vec<u32>],
    final_output: &[u32],
    lhs_index: usize,
    rhs_index: usize,
) -> Result<Vec<u32>, String> {
    if lhs_index >= subscripts.len() || rhs_index >= subscripts.len() {
        return Err("contraction index out of range".into());
    }
    if lhs_index == rhs_index {
        return Err("cannot contract an operand with itself".into());
    }

    let mut outside_counts: HashMap<u32, usize> = HashMap::new();
    for (i, labels) in subscripts.iter().enumerate() {
        if i == lhs_index || i == rhs_index {
            continue;
        }
        for &label in labels {
            *outside_counts.entry(label).or_default() += 1;
        }
    }

    let mut labels_in_order = Vec::new();
    for &label in &subscripts[lhs_index] {
        append_unique(&mut labels_in_order, label);
    }
    for &label in &subscripts[rhs_index] {
        append_unique(&mut labels_in_order, label);
    }

    let label_set: HashSet<u32> = labels_in_order.iter().copied().collect();
    let mut output = Vec::new();
    for &label in final_output {
        if label_set.contains(&label) {
            append_unique(&mut output, label);
        }
    }
    for &label in &labels_in_order {
        if outside_counts.get(&label).copied().unwrap_or(0) > 0 {
            append_unique(&mut output, label);
        }
    }
    Ok(output)
}

fn binary_subscripts(
    subscripts: &[Vec<u32>],
    final_output: &[u32],
    lhs_index: usize,
    rhs_index: usize,
) -> Result<Subscripts, String> {
    let output = intermediate_output_labels(subscripts, final_output, lhs_index, rhs_index)?;
    Ok(Subscripts::new(
        &[&subscripts[lhs_index], &subscripts[rhs_index]],
        &output,
    ))
}

fn contract_label_subscripts(
    subscripts: &[Vec<u32>],
    final_output: &[u32],
    lhs_index: usize,
    rhs_index: usize,
) -> Result<Vec<Vec<u32>>, String> {
    let output = intermediate_output_labels(subscripts, final_output, lhs_index, rhs_index)?;
    let mut next = subscripts.to_vec();
    let first_remove = lhs_index.max(rhs_index);
    let second_remove = lhs_index.min(rhs_index);
    next.remove(first_remove);
    next.remove(second_remove);
    next.push(output);
    Ok(next)
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

fn run_instance_trace(
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

    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .map_err(|e| format!("{e}"))?;

    // Warmup (execution only, graph already compiled)
    // Use catch_unwind to handle panics from unsupported layouts
    for _ in 0..3 {
        let operands = create_operand_tensors(&instance.shapes_colmajor);
        let bindings = bind_operands(&compiled.inputs, &operands)?;
        let program = &compiled.program;
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            executor.run_many_with_inputs(program, &bindings)
        }));
        let _ = unwrap_eval_result(result, "panic during execution (unsupported layout?)")?;
    }

    // Timed runs
    let num_runs = 15;
    let mut durations = Vec::with_capacity(num_runs);
    for _ in 0..num_runs {
        let operands = create_operand_tensors(&instance.shapes_colmajor);
        let bindings = bind_operands(&compiled.inputs, &operands)?;
        let program = &compiled.program;
        let t0 = Instant::now();
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            executor.run_many_with_inputs(program, &bindings)
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

fn contract_once_eager(
    ctx: &std::sync::Arc<EagerRuntime>,
    instance: &BenchmarkInstance,
    path_meta: &PathMeta,
) -> Result<Tensor, String> {
    let parsed = Subscripts::parse(&instance.format_string_colmajor).map_err(|e| format!("{e}"))?;
    let mut subscripts = parsed.inputs;
    let final_output = parsed.output;
    let mut operands = create_eager_operands(&instance.shapes_colmajor, ctx);

    for &[lhs_index, rhs_index] in &path_meta.path {
        let binary = binary_subscripts(&subscripts, &final_output, lhs_index, rhs_index)?;
        let first_remove = lhs_index.max(rhs_index);
        let second_remove = lhs_index.min(rhs_index);
        let rhs = operands.remove(first_remove);
        let lhs = operands.remove(second_remove);
        let ordered = if lhs_index < rhs_index {
            vec![lhs, rhs]
        } else {
            vec![rhs, lhs]
        };
        let input_refs: Vec<&EagerTensor> = ordered.iter().collect();
        let binary_subscripts = (&binary).into();
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            eager_tensor::einsum_subscripts(&input_refs, &binary_subscripts)
        }));
        let result = unwrap_eval_result(result, "panic during eager execution")?;
        operands.push(result);
        subscripts = contract_label_subscripts(&subscripts, &final_output, lhs_index, rhs_index)?;
    }

    if operands.len() != 1 {
        return Err("contraction did not reduce to one output tensor".into());
    }
    Ok(operands
        .pop()
        .expect("checked operands length")
        .data()
        .clone())
}

fn run_instance_eager(
    instance: &BenchmarkInstance,
    path_meta: &PathMeta,
) -> Result<(Duration, Duration, Duration), String> {
    if instance.dtype == "complex128" {
        return Err("complex128 not supported".into());
    }

    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());

    for _ in 0..3 {
        let eval = contract_once_eager(&ctx, instance, path_meta)?;
        black_box(&eval);
    }

    let num_runs = 15;
    let mut durations = Vec::with_capacity(num_runs);
    for _ in 0..num_runs {
        let t0 = Instant::now();
        let eval = contract_once_eager(&ctx, instance, path_meta)?;
        let elapsed = t0.elapsed();
        black_box(&eval);
        durations.push(elapsed);
    }

    durations.sort();
    let median = durations[num_runs / 2];
    let q1 = durations[num_runs / 4];
    let q3 = durations[3 * num_runs / 4];
    let iqr = q3.saturating_sub(q1);
    Ok((median, iqr, Duration::ZERO))
}

fn run_instance(
    mode: TenferroMode,
    instance: &BenchmarkInstance,
    path_meta: &PathMeta,
) -> Result<(Duration, Duration, Duration), String> {
    match mode {
        TenferroMode::Trace => run_instance_trace(instance, path_meta),
        TenferroMode::Eager => run_instance_eager(instance, path_meta),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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
    let mode = TenferroMode::from_env();
    let backend_name = mode.backend_name();
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

    println!("{backend_name} benchmark suite");
    println!("==================================");
    println!(
        "Loaded {} instances from {}",
        instances.len(),
        data_dir.display()
    );
    println!("Backend: {backend_name}");
    println!("RAYON_NUM_THREADS={rayon_threads}, OMP_NUM_THREADS={omp_threads}");
    println!(
        "Timing: median ± IQR of 15 runs (3 warmup), {}",
        mode.timing_note()
    );

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
            match run_instance(mode, instance, path_meta) {
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
