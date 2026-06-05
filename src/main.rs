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
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_einsum::eager_tensor;
use tenferro_einsum::{ContractionTree, Subscripts};
use tenferro_einsum_benchmark::{compile_einsum, unwrap_eval_result};
use tenferro_runtime::{GraphExecutor, TensorRead, TracedTensor};
use tenferro_tensor::{Tensor, TypedTensor};

const NUM_WARMUPS: usize = 3;
const NUM_RUNS: usize = 15;

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

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct StrategyCacheKey {
    instance_name: String,
    path: Vec<[usize; 2]>,
}

fn strategy_cache_key(instance: &BenchmarkInstance, path_meta: &PathMeta) -> StrategyCacheKey {
    StrategyCacheKey {
        instance_name: instance.name.clone(),
        path: path_meta.path.clone(),
    }
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

fn bind_operand_reads<'a>(
    inputs: &'a [TracedTensor],
    operands: &'a [Tensor],
) -> Result<Vec<(&'a TracedTensor, TensorRead<'a>)>, String> {
    if inputs.len() != operands.len() {
        return Err(format!(
            "operand count mismatch: graph expects {} inputs but runner created {}",
            inputs.len(),
            operands.len()
        ));
    }
    Ok(inputs
        .iter()
        .zip(operands.iter())
        .map(|(input, operand)| (input, TensorRead::from_tensor(operand)))
        .collect())
}

fn cpu_backend_from_env() -> Result<CpuBackend, String> {
    match std::env::var("TENFERRO_CPU_BACKEND_KIND")
        .unwrap_or_else(|_| "default".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "default" | "" => Ok(CpuBackend::new()),
        "blas" => CpuBackend::with_kind(CpuBackendKind::Blas).map_err(|e| e.to_string()),
        "faer" => CpuBackend::with_kind(CpuBackendKind::Faer).map_err(|e| e.to_string()),
        other => Err(format!(
            "unknown TENFERRO_CPU_BACKEND_KIND={other:?}; use default, blas, or faer"
        )),
    }
}

fn profile_bench_breakdown_enabled() -> bool {
    std::env::var_os("TENFERRO_PROFILE_BENCH_BREAKDOWN").is_some()
}

fn duration_stats(mut durations: Vec<Duration>) -> (Duration, Duration) {
    durations.sort();
    let median = durations[durations.len() / 2];
    let q1 = durations[durations.len() / 4];
    let q3 = durations[3 * durations.len() / 4];
    let iqr = q3.saturating_sub(q1);
    (median, iqr)
}

fn print_breakdown(
    backend: &str,
    instance: &BenchmarkInstance,
    strategy_name: &str,
    section: &str,
    durations: Vec<Duration>,
) {
    let (median, iqr) = duration_stats(durations);
    println!(
        "Breakdown: backend={backend} instance={} strategy={strategy_name} section={section} median_ms={:.6} iqr_ms={:.6}",
        instance.name,
        median.as_secs_f64() * 1e3,
        iqr.as_secs_f64() * 1e3,
    );
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
            Self::Trace => "graph compiled once, output TensorValue preserves final lazy views",
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
    strategy_name: &str,
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
    if std::env::var_os("TENFERRO_DUMP_PROGRAM").is_some() {
        eprintln!("{:#?}", compiled.program);
    }

    let mut executor = GraphExecutor::new(cpu_backend_from_env()?);
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .map_err(|e| format!("{e}"))?;

    let operands = create_operand_tensors(&instance.shapes_colmajor);
    let bindings = bind_operand_reads(&compiled.inputs, &operands)?;
    let program = &compiled.program;

    // Warmup (execution only, graph already compiled)
    // Use catch_unwind to handle panics from unsupported layouts
    for _ in 0..NUM_WARMUPS {
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            executor.run_many_values_with_input_reads(program, &bindings)
        }));
        let eval = unwrap_eval_result(result, "panic during execution (unsupported layout?)")?;
        black_box(&eval);
        executor.reclaim_value_outputs(eval);
    }

    // Timed runs
    let mut durations = Vec::with_capacity(NUM_RUNS);
    for _ in 0..NUM_RUNS {
        let t0 = Instant::now();
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            executor.run_many_values_with_input_reads(program, &bindings)
        }));
        let elapsed = t0.elapsed();
        let eval = unwrap_eval_result(result, "panic during execution (unsupported layout?)")?;
        black_box(&eval);
        executor.reclaim_value_outputs(eval);
        durations.push(elapsed);
    }

    let (median, iqr) = duration_stats(durations);

    if profile_bench_breakdown_enabled() {
        let mut input_create = Vec::with_capacity(NUM_RUNS);
        for _ in 0..NUM_RUNS {
            let started = Instant::now();
            let operands = create_operand_tensors(&instance.shapes_colmajor);
            input_create.push(started.elapsed());
            black_box(&operands);
        }
        print_breakdown(
            "tenferro-trace",
            instance,
            strategy_name,
            "trace.input_create",
            input_create,
        );

        let mut input_bind = Vec::with_capacity(NUM_RUNS);
        for _ in 0..NUM_RUNS {
            let started = Instant::now();
            let bindings = bind_operand_reads(&compiled.inputs, &operands)?;
            input_bind.push(started.elapsed());
            black_box(&bindings);
        }
        print_breakdown(
            "tenferro-trace",
            instance,
            strategy_name,
            "trace.input_bind",
            input_bind,
        );

        let mut executor_run = Vec::with_capacity(NUM_RUNS);
        for _ in 0..NUM_RUNS {
            let started = Instant::now();
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                executor.run_many_values_with_input_reads(program, &bindings)
            }));
            let eval = unwrap_eval_result(result, "panic during execution (unsupported layout?)")?;
            executor_run.push(started.elapsed());
            black_box(&eval);
            executor.reclaim_value_outputs(eval);
        }
        print_breakdown(
            "tenferro-trace",
            instance,
            strategy_name,
            "trace.executor_run",
            executor_run,
        );
    }

    Ok((median, iqr, compile_time))
}

#[derive(Default)]
struct EagerRunBreakdown {
    total: Duration,
    parse_subscripts: Duration,
    operand_handles: Duration,
    binary_setup: Duration,
    einsum_call: Duration,
    output_take: Duration,
}

fn contract_once_eager(
    instance: &BenchmarkInstance,
    path_meta: &PathMeta,
    source_operands: &[EagerTensor],
    mut profile: Option<&mut EagerRunBreakdown>,
) -> Result<EagerTensor, String> {
    let total_started = Instant::now();
    let started = Instant::now();
    let parsed = Subscripts::parse(&instance.format_string_colmajor).map_err(|e| format!("{e}"))?;
    if let Some(profile) = profile.as_deref_mut() {
        profile.parse_subscripts += started.elapsed();
    }

    let started = Instant::now();
    let mut subscripts = parsed.inputs;
    let final_output = parsed.output;
    let mut operands = source_operands.to_vec();
    if let Some(profile) = profile.as_deref_mut() {
        profile.operand_handles += started.elapsed();
    }

    for &[lhs_index, rhs_index] in &path_meta.path {
        let started = Instant::now();
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
        if let Some(profile) = profile.as_deref_mut() {
            profile.binary_setup += started.elapsed();
        }

        let started = Instant::now();
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            eager_tensor::einsum_subscripts(&input_refs, &binary_subscripts)
        }));
        let result = unwrap_eval_result(result, "panic during eager execution")?;
        if let Some(profile) = profile.as_deref_mut() {
            profile.einsum_call += started.elapsed();
        }
        operands.push(result);
        subscripts = contract_label_subscripts(&subscripts, &final_output, lhs_index, rhs_index)?;
    }

    if operands.len() != 1 {
        return Err("contraction did not reduce to one output tensor".into());
    }
    let started = Instant::now();
    let output = operands.pop().expect("checked operands length");
    if let Some(profile) = profile.as_deref_mut() {
        profile.output_take += started.elapsed();
        profile.total += total_started.elapsed();
    }
    Ok(output)
}

fn run_instance_eager(
    instance: &BenchmarkInstance,
    path_meta: &PathMeta,
    strategy_name: &str,
) -> Result<(Duration, Duration, Duration), String> {
    if instance.dtype == "complex128" {
        return Err("complex128 not supported".into());
    }

    let ctx = EagerRuntime::with_cpu_backend(cpu_backend_from_env()?);
    let source_operands = create_eager_operands(&instance.shapes_colmajor, &ctx);

    for _ in 0..NUM_WARMUPS {
        let eval = contract_once_eager(instance, path_meta, &source_operands, None)?;
        black_box(&eval);
    }

    let mut durations = Vec::with_capacity(NUM_RUNS);
    for _ in 0..NUM_RUNS {
        let t0 = Instant::now();
        let eval = contract_once_eager(instance, path_meta, &source_operands, None)?;
        let elapsed = t0.elapsed();
        black_box(&eval);
        durations.push(elapsed);
    }

    let (median, iqr) = duration_stats(durations);

    if profile_bench_breakdown_enabled() {
        let mut input_create = Vec::with_capacity(NUM_RUNS);
        for _ in 0..NUM_RUNS {
            let started = Instant::now();
            let operands = create_eager_operands(&instance.shapes_colmajor, &ctx);
            input_create.push(started.elapsed());
            black_box(&operands);
        }
        print_breakdown(
            "tenferro-eager",
            instance,
            strategy_name,
            "eager.input_create",
            input_create,
        );

        let mut total = Vec::with_capacity(NUM_RUNS);
        let mut parse_subscripts = Vec::with_capacity(NUM_RUNS);
        let mut operand_handles = Vec::with_capacity(NUM_RUNS);
        let mut binary_setup = Vec::with_capacity(NUM_RUNS);
        let mut einsum_call = Vec::with_capacity(NUM_RUNS);
        let mut output_take = Vec::with_capacity(NUM_RUNS);
        for _ in 0..NUM_RUNS {
            let mut breakdown = EagerRunBreakdown::default();
            let eval =
                contract_once_eager(instance, path_meta, &source_operands, Some(&mut breakdown))?;
            black_box(&eval);
            total.push(breakdown.total);
            parse_subscripts.push(breakdown.parse_subscripts);
            operand_handles.push(breakdown.operand_handles);
            binary_setup.push(breakdown.binary_setup);
            einsum_call.push(breakdown.einsum_call);
            output_take.push(breakdown.output_take);
        }
        print_breakdown(
            "tenferro-eager",
            instance,
            strategy_name,
            "eager.total",
            total,
        );
        print_breakdown(
            "tenferro-eager",
            instance,
            strategy_name,
            "eager.parse_subscripts",
            parse_subscripts,
        );
        print_breakdown(
            "tenferro-eager",
            instance,
            strategy_name,
            "eager.operand_handles",
            operand_handles,
        );
        print_breakdown(
            "tenferro-eager",
            instance,
            strategy_name,
            "eager.binary_setup",
            binary_setup,
        );
        print_breakdown(
            "tenferro-eager",
            instance,
            strategy_name,
            "eager.einsum_call",
            einsum_call,
        );
        print_breakdown(
            "tenferro-eager",
            instance,
            strategy_name,
            "eager.output_take",
            output_take,
        );
    }

    Ok((median, iqr, Duration::ZERO))
}

fn run_instance(
    mode: TenferroMode,
    instance: &BenchmarkInstance,
    path_meta: &PathMeta,
    strategy_name: &str,
) -> Result<(Duration, Duration, Duration), String> {
    match mode {
        TenferroMode::Trace => run_instance_trace(instance, path_meta, strategy_name),
        TenferroMode::Eager => run_instance_eager(instance, path_meta, strategy_name),
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
    let cpu_backend_kind =
        std::env::var("TENFERRO_CPU_BACKEND_KIND").unwrap_or_else(|_| "default".into());
    let dot_decomposer =
        std::env::var("TENFERRO_OPT_DOT_DECOMPOSER").unwrap_or_else(|_| "0".into());

    println!("{backend_name} benchmark suite");
    println!("==================================");
    println!(
        "Loaded {} instances from {}",
        instances.len(),
        data_dir.display()
    );
    println!("Backend: {backend_name}");
    println!("TENFERRO_CPU_BACKEND_KIND={cpu_backend_kind}");
    println!("TENFERRO_OPT_DOT_DECOMPOSER={dot_decomposer}");
    println!("RAYON_NUM_THREADS={rayon_threads}, OMP_NUM_THREADS={omp_threads}");
    println!(
        "Timing: median ± IQR of {NUM_RUNS} runs ({NUM_WARMUPS} warmup), {}",
        mode.timing_note()
    );

    let strategies: &[(&str, fn(&PathInfo) -> &PathMeta)] = &[
        ("opt_flops", |p| &p.opt_flops),
        ("opt_size", |p| &p.opt_size),
    ];
    let mut measured_by_path: HashMap<
        StrategyCacheKey,
        Result<(Duration, Duration, Duration), String>,
    > = HashMap::new();

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
            let cache_key = strategy_cache_key(instance, path_meta);
            let result = if let Some(cached) = measured_by_path.get(&cache_key) {
                eprintln!(
                    "  -> {} strategy={strategy_name}: reusing previous measurement for identical path",
                    instance.name
                );
                cached.clone()
            } else {
                let measured = run_instance(mode, instance, path_meta, strategy_name);
                measured_by_path.insert(cache_key, measured.clone());
                measured
            };
            match result {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_instance(name: &str) -> BenchmarkInstance {
        BenchmarkInstance {
            name: name.to_string(),
            format_string_colmajor: "ji,kj->ki".to_string(),
            shapes_colmajor: vec![vec![2, 2], vec![2, 2]],
            dtype: "float64".to_string(),
            num_tensors: 2,
            paths: PathInfo {
                opt_size: PathMeta {
                    path: vec![[0, 1]],
                    log2_size: 2.0,
                    log10_flops: 1.0,
                },
                opt_flops: PathMeta {
                    path: vec![[0, 1]],
                    log2_size: 2.0,
                    log10_flops: 1.0,
                },
            },
        }
    }

    #[test]
    fn strategy_cache_key_uses_instance_name_and_exact_path() {
        let instance = test_instance("bin_matmul_1024");
        let same_path_different_metadata = PathMeta {
            path: vec![[0, 1]],
            log2_size: 99.0,
            log10_flops: 88.0,
        };
        let different_path = PathMeta {
            path: vec![[1, 0]],
            log2_size: 2.0,
            log10_flops: 1.0,
        };

        assert_eq!(
            strategy_cache_key(&instance, &instance.paths.opt_flops),
            strategy_cache_key(&instance, &same_path_different_metadata)
        );
        assert_ne!(
            strategy_cache_key(&instance, &instance.paths.opt_flops),
            strategy_cache_key(&instance, &different_path)
        );
        assert_ne!(
            strategy_cache_key(&instance, &instance.paths.opt_flops),
            strategy_cache_key(&test_instance("other"), &instance.paths.opt_flops)
        );
    }
}
