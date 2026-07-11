//! GPU benchmark runner for tenferro-cuda-eager and tenferro-cuda-trace backends.
//!
//! # Usage
//!
//! ```text
//! benchmark_gpu_rust OUTPUT_JSONL DEVICE_ORDINAL PROBLEM_FILTER BACKEND... -- SUITE.yaml...
//! ```

use std::hint::black_box;
use std::panic;
use std::time::Instant;

use serde_json::{json, Value};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{EagerEinsumExt, GraphCompilerEinsumExt};
use tenferro_gpu::{download_tensor, gpu_available, upload_tensor, CudaBackend, CudaRuntime};
use tenferro_linalg::{EagerTensorLinalgExt, TracedTensorLinalgExt};
use tenferro_einsum_benchmark::tensornetwork::{
    build_cpu_eager_inputs as build_tensor_network_cpu_inputs, contract_tree_eager,
    contract_tree_trace, load_tensor_network, scalar_f32, scalar_f32_tensor,
    tensor_f32_col_major, TensorNetworkFile, TensorNetworkSpec,
};
use tenferro_runtime::{
    DotGeneralConfig, Error as TfError, GraphCompiler, GraphExecutor, Tensor, TracedTensor,
};
use tenferro_tensor::TypedTensor;

fn eager_from_tensor_in(tensor: Tensor, ctx: std::sync::Arc<EagerRuntime>) -> Result<EagerTensor, String> {
    EagerTensor::from_tensor_in(tensor, ctx).map_err(|e| format!("{e}"))
}

trait EagerTensorDataCompat {
    fn data(&self) -> Result<Tensor, String>;
}

impl EagerTensorDataCompat for EagerTensor {
    fn data(&self) -> Result<Tensor, String> {
        self.to_tensor().map_err(|e| format!("{e}"))
    }
}

mod traced_tensor {
    use super::*;

    pub fn matmul(lhs: &TracedTensor, rhs: &TracedTensor) -> Result<TracedTensor, TfError> {
        lhs.matmul(rhs)
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = &args[1];
    let device_ordinal: usize = args[2].parse().expect("DEVICE_ORDINAL must be usize");
    let problem_filter = &args[3];
    let sep = args.iter().position(|a| a == "--").expect("missing --");
    let backends: Vec<&str> = args[4..sep].iter().map(String::as_str).collect();
    let suite_paths: Vec<&str> = args[sep + 1..].iter().map(String::as_str).collect();

    if !gpu_available() {
        eprintln!("benchmark_gpu_rust: no CUDA GPU found, skipping");
        std::fs::write(out_path, "").expect("failed to write empty output");
        return;
    }

    let timestamp = utc_timestamp();
    let benchmark_commit = git_commit(".");
    let tenferro_commit = git_commit("extern/tenferro-rs");

    let mut lines: Vec<String> = Vec::new();

    for suite_path in suite_paths {
        let suite_text = std::fs::read_to_string(suite_path)
            .unwrap_or_else(|e| panic!("cannot read {suite_path}: {e}"));
        let suite: serde_yaml::Value =
            serde_yaml::from_str(&suite_text).expect("invalid suite YAML");
        let suite_id = suite["suite_id"].as_str().unwrap_or("unknown").to_string();
        let suite_backends = yaml_str_list(&suite["backends"]);

        for problem in suite["problems"].as_sequence().expect("problems list") {
            let pid = problem["id"].as_str().unwrap_or("");
            if !problem_filter.is_empty() && pid != problem_filter {
                continue;
            }
            let candidates = backend_candidates(problem, &suite_backends);

            for &backend in &backends {
                let rec = dispatch(
                    &suite_id,
                    problem,
                    backend,
                    device_ordinal,
                    &candidates,
                    &timestamp,
                    benchmark_commit.as_deref(),
                    tenferro_commit.as_deref(),
                );
                lines.push(serde_json::to_string(&rec).unwrap());
            }
        }
    }

    let jsonl = lines.join("\n") + if lines.is_empty() { "" } else { "\n" };
    std::fs::write(out_path, jsonl).expect("failed to write JSONL");
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn dispatch(
    suite_id: &str,
    problem: &serde_yaml::Value,
    backend: &str,
    device_ordinal: usize,
    candidates: &[String],
    ts: &str,
    bc: Option<&str>,
    tc: Option<&str>,
) -> Value {
    let pid = problem["id"].as_str().unwrap_or("");
    if !candidates.iter().any(|c| c == backend) {
        return stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            "unsupported",
            &format!("{backend} not listed for {pid}"),
            "phase2-runner",
            ts,
            bc,
            tc,
        );
    }
    match backend {
        "tenferro-cuda-eager" => run_eager(suite_id, problem, backend, device_ordinal, ts, bc, tc),
        "tenferro-cuda-trace" => run_trace(suite_id, problem, backend, device_ordinal, ts, bc, tc),
        _ => stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            "not_configured",
            &format!("{backend} not in Rust GPU runner"),
            "phase2-runner",
            ts,
            bc,
            tc,
        ),
    }
}

// ---------------------------------------------------------------------------
// Tensor network runner (TensorNetworkBenchmarks parity)
// ---------------------------------------------------------------------------

fn verify_tensor_network_scalar(
    gpu: f32,
    cpu: f32,
    rtol: f64,
    atol: f64,
) -> (String, Option<f64>, Option<f64>) {
    let max_abs = (gpu - cpu).abs() as f64;
    let max_rel = max_abs / cpu.abs().max(1e-10_f32) as f64;
    let norm = cpu.abs().max(1.0_f32) as f64;
    let passed = max_abs <= atol + rtol * norm;
    (
        if passed { "passed" } else { "failed" }.to_string(),
        Some(max_abs),
        Some(max_rel),
    )
}

fn build_gpu_eager_inputs(
    network: &TensorNetworkFile,
    spec: &TensorNetworkSpec,
    ctx: std::sync::Arc<EagerRuntime>,
    runtime: &CudaRuntime,
) -> Result<Vec<EagerTensor>, String> {
    network
        .inputs
        .iter()
        .map(|ixs| {
            let shape = vec![spec.bond_dim; ixs.len()];
            let cpu = tensor_f32_col_major(&shape, spec.fill_value)?;
            let gpu = upload_tensor(runtime, &cpu).map_err(|e| format!("upload: {e}"))?;
            Ok(eager_from_tensor_in(gpu, ctx.clone())?)
        })
        .collect()
}

fn build_trace_inputs(
    network: &TensorNetworkFile,
    spec: &TensorNetworkSpec,
    upload_rt: &CudaRuntime,
) -> Result<Vec<(TracedTensor, Tensor)>, String> {
    network
        .inputs
        .iter()
        .map(|ixs| {
            let shape = vec![spec.bond_dim; ixs.len()];
            let cpu = tensor_f32_col_major(&shape, spec.fill_value)?;
            let tensor = upload_tensor(upload_rt, &cpu).map_err(|e| format!("upload: {e}"))?;
            Ok((
                TracedTensor::from_tensor_concrete_shape(tensor.clone())
                    .map_err(|e| format!("trace input: {e}"))?,
                cpu,
            ))
        })
        .collect()
}

fn scalar_f32_from_eager_gpu(
    result: &EagerTensor,
    runtime: &CudaRuntime,
) -> Result<f32, String> {
    let downloaded = download_tensor(runtime, &result.data()?)
        .map_err(|e| format!("download: {e}"))?;
    scalar_f32_tensor(&downloaded)
}

fn run_eager_tensor_network(
    suite_id: &str,
    problem: &serde_yaml::Value,
    backend: &str,
    device_ordinal: usize,
    ts: &str,
    bc: Option<&str>,
    tc: Option<&str>,
) -> Value {
    let n_warmup = yaml_usize(&problem["run"]["warmups"], 3);
    let n_runs = yaml_usize(&problem["run"]["runs"], 10);
    let rtol = problem["verify"]["rtol"].as_f64().unwrap_or(1e-4);
    let atol = problem["verify"]["atol"].as_f64().unwrap_or(1e-5);
    let exec_path = "phase2-measured-tenferro-cuda-eager-tensornetwork";

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| -> Result<_, String> {
        let spec = TensorNetworkSpec::from_problem(problem, std::path::Path::new("."))?;
        let network = load_tensor_network(&spec.source)?;
        let transfer_bk =
            CudaBackend::new(device_ordinal).map_err(|e| format!("transfer backend: {e}"))?;
        let compute_bk =
            CudaBackend::new(device_ordinal).map_err(|e| format!("compute backend: {e}"))?;
        let ctx = EagerRuntime::with_cuda_backend(compute_bk);
        let gpu_inputs = build_gpu_eager_inputs(
            &network,
            &spec,
            ctx.clone(),
            transfer_bk.runtime(),
        )?;
        sync_cubecl_runtime(transfer_bk.runtime())?;
        let cpu_ctx = EagerRuntime::with_cpu_backend(CpuBackend::default());
        let cpu_inputs =
            build_tensor_network_cpu_inputs(&network, &spec, cpu_ctx.clone())?;

        for _ in 0..n_warmup {
            let _ = contract_tree_eager(&network.tree, &gpu_inputs)?;
            sync_eager_runtime(&ctx)?;
        }

        let mut times_ms = Vec::with_capacity(n_runs);
        let mut last_out: Option<EagerTensor> = None;
        for _ in 0..n_runs {
            let t0 = Instant::now();
            let out = contract_tree_eager(&network.tree, &gpu_inputs)?;
            sync_eager_runtime(&ctx)?;
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            last_out = Some(out);
        }

        let cpu_scalar = scalar_f32(&contract_tree_eager(&network.tree, &cpu_inputs)?)?;
        let last_out = last_out.ok_or("missing timed tensor network result".to_string())?;
        let gpu_scalar = scalar_f32_from_eager_gpu(&last_out, transfer_bk.runtime())?;
        let (ver_status, max_abs, max_rel) =
            verify_tensor_network_scalar(gpu_scalar, cpu_scalar, rtol, atol);

        Ok((times_ms, ver_status, max_abs, max_rel))
    }));

    match result {
        Ok(Ok((times_ms, ver_status, max_abs, max_rel))) => {
            if ver_status == "failed" {
                return stub(
                    suite_id,
                    problem,
                    backend,
                    device_ordinal,
                    "verification_failed",
                    &format!("max_abs={:.3e}", max_abs.unwrap_or(f64::NAN)),
                    exec_path,
                    ts,
                    bc,
                    tc,
                );
            }
            let stats = timing_stats(&times_ms);
            ok_record(
                suite_id,
                problem,
                backend,
                device_ordinal,
                exec_path,
                n_warmup,
                n_runs,
                stats,
                &ver_status,
                max_abs,
                max_rel,
                rtol,
                atol,
                ts,
                bc,
                tc,
            )
        }
        Ok(Err(msg)) => stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            "runtime_failed",
            &msg,
            exec_path,
            ts,
            bc,
            tc,
        ),
        Err(_) => stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            "runtime_failed",
            "panic during tensor network eager benchmark",
            exec_path,
            ts,
            bc,
            tc,
        ),
    }
}

fn run_trace_tensor_network(
    suite_id: &str,
    problem: &serde_yaml::Value,
    backend: &str,
    device_ordinal: usize,
    ts: &str,
    bc: Option<&str>,
    tc: Option<&str>,
) -> Value {
    let n_warmup = yaml_usize(&problem["run"]["warmups"], 3);
    let n_runs = yaml_usize(&problem["run"]["runs"], 10);
    let rtol = problem["verify"]["rtol"].as_f64().unwrap_or(1e-4);
    let atol = problem["verify"]["atol"].as_f64().unwrap_or(1e-5);
    let exec_path = "phase2-measured-tenferro-cuda-trace-tensornetwork";

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| -> Result<_, String> {
        let spec = TensorNetworkSpec::from_problem(problem, std::path::Path::new("."))?;
        let network = load_tensor_network(&spec.source)?;
        let transfer_bk =
            CudaBackend::new(device_ordinal).map_err(|e| format!("transfer backend: {e}"))?;
        let compute_bk =
            CudaBackend::new(device_ordinal).map_err(|e| format!("compute backend: {e}"))?;
        let trace_inputs = build_trace_inputs(&network, &spec, transfer_bk.runtime())?;
        let output = contract_tree_trace(&network.tree, &trace_inputs)?;
        sync_cubecl_runtime(transfer_bk.runtime())?;

        let mut compiler = GraphCompiler::new();
        let program = compiler
            .compile_many(&[&output])
            .map_err(|e| format!("compile: {e}"))?;

        let mut executor = GraphExecutor::new(compute_bk);
        executor
            .register_extension(tenferro_einsum::register_runtime)
            .map_err(|e| format!("register einsum: {e}"))?;

        for _ in 0..n_warmup {
            let out = executor
                .run_many(&program)
                .map_err(|e| format!("warmup: {e}"))?;
            sync_cubecl_runtime(executor.backend().runtime())?;
            black_box(out.len());
        }

        let mut times_ms = Vec::with_capacity(n_runs);
        let mut gpu_scalar = None;
        for _ in 0..n_runs {
            let t0 = Instant::now();
            let out = executor
                .run_many(&program)
                .map_err(|e| format!("run: {e}"))?;
            sync_cubecl_runtime(executor.backend().runtime())?;
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            let downloaded = download_tensor(transfer_bk.runtime(), &out[0])
                .map_err(|e| format!("download: {e}"))?;
            gpu_scalar = Some(scalar_f32_tensor(&downloaded)?);
        }

        let cpu_ctx = EagerRuntime::with_cpu_backend(CpuBackend::default());
        let cpu_inputs = build_tensor_network_cpu_inputs(&network, &spec, cpu_ctx)?;
        let cpu_scalar = scalar_f32(&contract_tree_eager(&network.tree, &cpu_inputs)?)?;
        let gpu_scalar = gpu_scalar.ok_or("missing timed tensor network result")?;
        let (ver_status, max_abs, max_rel) =
            verify_tensor_network_scalar(gpu_scalar, cpu_scalar, rtol, atol);

        Ok((times_ms, ver_status, max_abs, max_rel))
    }));

    match result {
        Ok(Ok((times_ms, ver_status, max_abs, max_rel))) => {
            if ver_status == "failed" {
                return stub(
                    suite_id,
                    problem,
                    backend,
                    device_ordinal,
                    "verification_failed",
                    &format!("max_abs={:.3e}", max_abs.unwrap_or(f64::NAN)),
                    exec_path,
                    ts,
                    bc,
                    tc,
                );
            }
            let stats = timing_stats(&times_ms);
            ok_record(
                suite_id,
                problem,
                backend,
                device_ordinal,
                exec_path,
                n_warmup,
                n_runs,
                stats,
                &ver_status,
                max_abs,
                max_rel,
                rtol,
                atol,
                ts,
                bc,
                tc,
            )
        }
        Ok(Err(msg)) => stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            "runtime_failed",
            &msg,
            exec_path,
            ts,
            bc,
            tc,
        ),
        Err(_) => stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            "runtime_failed",
            "panic during tensor network trace benchmark",
            exec_path,
            ts,
            bc,
            tc,
        ),
    }
}

// ---------------------------------------------------------------------------
// Eager runner
// ---------------------------------------------------------------------------

fn run_eager(
    suite_id: &str,
    problem: &serde_yaml::Value,
    backend: &str,
    device_ordinal: usize,
    ts: &str,
    bc: Option<&str>,
    tc: Option<&str>,
) -> Value {
    let op = problem["op"].as_str().unwrap_or("");
    if op == "tensor_network_contract" {
        return run_eager_tensor_network(suite_id, problem, backend, device_ordinal, ts, bc, tc);
    }
    let supported = [
        "matmul",
        "batched_matmul",
        "einsum",
        "qr",
        "solve",
        "svd",
        "eigh",
    ];
    if !supported.contains(&op) {
        return stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            "not_configured",
            &format!("tenferro-cuda-eager: op={op} not implemented"),
            "phase2-runner",
            ts,
            bc,
            tc,
        );
    }

    let n_warmup = yaml_usize(&problem["run"]["warmups"], 3);
    let n_runs = yaml_usize(&problem["run"]["runs"], 7);
    let rtol = problem["verify"]["rtol"].as_f64().unwrap_or(1e-5);
    let atol = problem["verify"]["atol"].as_f64().unwrap_or(1e-8);
    let exec_path = "phase2-measured-tenferro-cuda-eager";

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| -> Result<_, String> {
        // Use two backends for the same device ordinal.
        // CubeCL returns the same underlying device client for the same ordinal,
        // so both backends share the same CUDA stream.
        let transfer_bk =
            CudaBackend::new(device_ordinal).map_err(|e| format!("transfer backend: {e}"))?;
        let compute_bk =
            CudaBackend::new(device_ordinal).map_err(|e| format!("compute backend: {e}"))?;
        let ctx = EagerRuntime::with_cuda_backend(compute_bk);

        let seed = problem["data"]["seed"].as_u64().unwrap_or(0);
        let gen = problem["data"]["generator"].as_str().unwrap_or("normal");

        // Upload CPU tensors to GPU, then wrap in EagerTensor
        let gpu_inputs = build_and_upload_eager_inputs(
            op,
            problem,
            seed,
            gen,
            transfer_bk.runtime(),
            ctx.clone(),
        )?;
        sync_cubecl_runtime(transfer_bk.runtime())?;
        let cpu_inputs = build_cpu_eager_inputs(op, problem, seed, gen);

        // Warmup
        for _ in 0..n_warmup {
            let out = run_eager_op(op, problem, &gpu_inputs)?;
            sync_eager_runtime(&ctx)?;
            black_box(out.len());
        }

        // Timed runs
        let mut times_ms = Vec::with_capacity(n_runs);
        let mut last_out: Option<Vec<EagerTensor>> = None;
        for _ in 0..n_runs {
            let t0 = Instant::now();
            let out = run_eager_op(op, problem, &gpu_inputs)?;
            sync_eager_runtime(&ctx)?;
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            last_out = Some(out);
        }

        // Verify last result against CPU reference
        let (ver_status, max_abs, max_rel) = if let Some(ref g) = last_out {
            let cpu_out = if op == "solve" {
                Vec::new()
            } else {
                run_eager_op(op, problem, &cpu_inputs).unwrap_or_default()
            };
            verify_eager(
                op,
                g,
                &cpu_out,
                &cpu_inputs,
                transfer_bk.runtime(),
                rtol,
                atol,
            )
        } else {
            ("skipped".to_string(), None, None)
        };

        Ok((times_ms, ver_status, max_abs, max_rel))
    }));

    match result {
        Ok(Ok((times_ms, ver_status, max_abs, max_rel))) => {
            if ver_status == "failed" {
                return stub(
                    suite_id,
                    problem,
                    backend,
                    device_ordinal,
                    "verification_failed",
                    &format!("max_abs={:.3e}", max_abs.unwrap_or(f64::NAN)),
                    exec_path,
                    ts,
                    bc,
                    tc,
                );
            }
            let stats = timing_stats(&times_ms);
            ok_record(
                suite_id,
                problem,
                backend,
                device_ordinal,
                exec_path,
                n_warmup,
                n_runs,
                stats,
                &ver_status,
                max_abs,
                max_rel,
                rtol,
                atol,
                ts,
                bc,
                tc,
            )
        }
        Ok(Err(msg)) => stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            "runtime_failed",
            &msg,
            exec_path,
            ts,
            bc,
            tc,
        ),
        Err(_) => stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            "runtime_failed",
            "panic during eager benchmark",
            exec_path,
            ts,
            bc,
            tc,
        ),
    }
}

// ---------------------------------------------------------------------------
// Trace runner
// ---------------------------------------------------------------------------

fn run_trace(
    suite_id: &str,
    problem: &serde_yaml::Value,
    backend: &str,
    device_ordinal: usize,
    ts: &str,
    bc: Option<&str>,
    tc: Option<&str>,
) -> Value {
    let op = problem["op"].as_str().unwrap_or("");
    if op == "tensor_network_contract" {
        return run_trace_tensor_network(suite_id, problem, backend, device_ordinal, ts, bc, tc);
    }
    let supported = [
        "matmul",
        "batched_matmul",
        "einsum",
        "qr",
        "solve",
        "svd",
        "eigh",
    ];
    if !supported.contains(&op) {
        return stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            "not_configured",
            &format!("tenferro-cuda-trace: op={op} not implemented"),
            "phase2-runner",
            ts,
            bc,
            tc,
        );
    }

    let n_warmup = yaml_usize(&problem["run"]["warmups"], 3);
    let n_runs = yaml_usize(&problem["run"]["runs"], 7);
    let rtol = problem["verify"]["rtol"].as_f64().unwrap_or(1e-5);
    let atol = problem["verify"]["atol"].as_f64().unwrap_or(1e-8);
    let exec_path = "phase2-measured-tenferro-cuda-trace";

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| -> Result<_, String> {
        let seed = problem["data"]["seed"].as_u64().unwrap_or(0);
        let gen = problem["data"]["generator"].as_str().unwrap_or("normal");

        // Two backends sharing the same CubeCL device client.
        // Build the trace graph with GPU-uploaded tensors as embedded constants.
        let transfer_bk =
            CudaBackend::new(device_ordinal).map_err(|e| format!("transfer backend: {e}"))?;
        let compute_bk =
            CudaBackend::new(device_ordinal).map_err(|e| format!("compute backend: {e}"))?;

        let (outputs, cpu_inputs) =
            build_trace_graph_gpu(op, problem, seed, gen, transfer_bk.runtime())
                .map_err(|e| format!("graph build: {e}"))?;
        sync_cubecl_runtime(transfer_bk.runtime())?;
        let output_refs: Vec<&TracedTensor> = outputs.iter().collect();
        let mut compiler = GraphCompiler::new();
        let program = compiler
            .compile_many(&output_refs)
            .map_err(|e| format!("compile: {e}"))?;

        let mut executor = GraphExecutor::new(compute_bk);
        executor
            .register_extension(tenferro_einsum::register_runtime)
            .map_err(|e| format!("register einsum: {e}"))?;
        executor
            .register_extension(tenferro_linalg::register_runtime)
            .map_err(|e| format!("register linalg: {e}"))?;

        // Warmup (triggers JIT compilation on first run)
        for _ in 0..n_warmup {
            let out = executor
                .run_many(&program)
                .map_err(|e| format!("warmup: {e}"))?;
            sync_cubecl_runtime(executor.backend().runtime())?;
            black_box(out.len());
        }

        // Timed runs
        let mut times_ms = Vec::with_capacity(n_runs);
        let mut last_out: Option<Vec<Tensor>> = None;
        for _ in 0..n_runs {
            let t0 = Instant::now();
            let out = executor
                .run_many(&program)
                .map_err(|e| format!("run: {e}"))?;
            sync_cubecl_runtime(executor.backend().runtime())?;
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            last_out = Some(out);
        }

        // Verify via download + CPU reference
        let (ver_status, max_abs, max_rel) = if let Some(ref gpu_out) = last_out {
            let gpu_tensors: Vec<Tensor> = gpu_out
                .iter()
                .filter_map(|t| download_tensor(transfer_bk.runtime(), t).ok())
                .collect();
            let cpu_out = if op == "solve" {
                Vec::new()
            } else {
                run_cpu_trace(op, problem, seed, gen).unwrap_or_default()
            };
            let cpu_inputs: Vec<Tensor> = cpu_inputs.iter().map(|(_, t)| t.clone()).collect();
            verify_tensors(op, &gpu_tensors, &cpu_out, &cpu_inputs, rtol, atol)
        } else {
            ("skipped".to_string(), None, None)
        };

        Ok((times_ms, ver_status, max_abs, max_rel))
    }));

    match result {
        Ok(Ok((times_ms, ver_status, max_abs, max_rel))) => {
            if ver_status == "failed" {
                return stub(
                    suite_id,
                    problem,
                    backend,
                    device_ordinal,
                    "verification_failed",
                    &format!("max_abs={:.3e}", max_abs.unwrap_or(f64::NAN)),
                    exec_path,
                    ts,
                    bc,
                    tc,
                );
            }
            let stats = timing_stats(&times_ms);
            ok_record(
                suite_id,
                problem,
                backend,
                device_ordinal,
                exec_path,
                n_warmup,
                n_runs,
                stats,
                &ver_status,
                max_abs,
                max_rel,
                rtol,
                atol,
                ts,
                bc,
                tc,
            )
        }
        Ok(Err(msg)) => stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            "runtime_failed",
            &msg,
            exec_path,
            ts,
            bc,
            tc,
        ),
        Err(_) => stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            "runtime_failed",
            "panic during trace benchmark",
            exec_path,
            ts,
            bc,
            tc,
        ),
    }
}

// ---------------------------------------------------------------------------
// Eager op construction
// ---------------------------------------------------------------------------

struct EagerInputs {
    a: EagerTensor,
    b: Option<EagerTensor>,
    extra: Vec<EagerTensor>,
}

/// Upload CPU tensors to GPU and wrap in EagerTensor with the given GPU context.
fn build_and_upload_eager_inputs(
    op: &str,
    problem: &serde_yaml::Value,
    seed: u64,
    gen: &str,
    rt: &CudaRuntime,
    ctx: std::sync::Arc<EagerRuntime>,
) -> Result<EagerInputs, String> {
    let upload = |t: Tensor| -> Result<EagerTensor, String> {
        let gpu_t = upload_tensor(rt, &t).map_err(|e| format!("upload: {e}"))?;
        Ok(EagerTensor::from_tensor_in(gpu_t, ctx.clone()).map_err(|e| format!("{e}"))?)
    };

    match op {
        "matmul" => {
            let p = &problem["matmul"];
            let (m, n, k) = yshape3(p, "m", "n", "k");
            Ok(EagerInputs {
                a: upload(tensor_f64(&[m, k], normal_data(&[m, k], seed)))?,
                b: Some(upload(tensor_f64(&[k, n], normal_data(&[k, n], seed + 1)))?),
                extra: vec![],
            })
        }
        "batched_matmul" => {
            let p = &problem["batched_matmul"];
            let batch = yaml_usize(&p["batch"], 16);
            let (m, n, k) = yshape3(p, "m", "n", "k");
            Ok(EagerInputs {
                a: upload(tensor_f64(
                    &[m, k, batch],
                    normal_data(&[m, k, batch], seed),
                ))?,
                b: Some(upload(tensor_f64(
                    &[k, n, batch],
                    normal_data(&[k, n, batch], seed + 1),
                ))?),
                extra: vec![],
            })
        }
        "einsum" => {
            let p = &problem["einsum"];
            let shapes = yaml_shapes(p);
            let dummy = upload(tensor_f64(&[1], vec![0.0]))?;
            let extra = shapes
                .iter()
                .enumerate()
                .map(|(i, sh)| upload(tensor_f64(sh, normal_data(sh, seed + i as u64))))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(EagerInputs {
                a: dummy,
                b: None,
                extra,
            })
        }
        "qr" | "svd" => {
            let p = &problem["linalg"];
            let (m, n) = (yaml_usize(&p["m"], 64), yaml_usize(&p["n"], 64));
            let d = if gen == "well_conditioned" {
                well_conditioned(m, n, seed)
            } else {
                normal_data(&[m, n], seed)
            };
            Ok(EagerInputs {
                a: upload(tensor_f64(&[m, n], d))?,
                b: None,
                extra: vec![],
            })
        }
        "solve" => {
            let p = &problem["linalg"];
            let n = yaml_usize(&p["n"], 64);
            let rhs = yaml_usize(&p["rhs_cols"], 1);
            Ok(EagerInputs {
                a: upload(tensor_f64(&[n, n], solve_matrix_data(n, seed, gen)))?,
                b: Some(upload(tensor_f64(
                    &[n, rhs],
                    normal_data(&[n, rhs], seed + 100),
                ))?),
                extra: vec![],
            })
        }
        "eigh" => {
            let p = &problem["linalg"];
            let n = yaml_usize(&p["n"], 64);
            let d = if gen == "spd" {
                spd_data(n, seed)
            } else {
                normal_data(&[n, n], seed)
            };
            Ok(EagerInputs {
                a: upload(tensor_f64(&[n, n], d))?,
                b: None,
                extra: vec![],
            })
        }
        _ => Err(format!("unsupported op for eager GPU: {op}")),
    }
}

/// Build CPU EagerTensors for reference verification.
fn build_cpu_eager_inputs(
    op: &str,
    problem: &serde_yaml::Value,
    seed: u64,
    gen: &str,
) -> EagerInputs {
    let cpu_ctx = EagerRuntime::with_cpu_backend(CpuBackend::default());
    let cpu = |t: Tensor| {
        eager_from_tensor_in(t, cpu_ctx.clone()).expect("benchmark tensor should be valid")
    };

    match op {
        "matmul" => {
            let p = &problem["matmul"];
            let (m, n, k) = yshape3(p, "m", "n", "k");
            EagerInputs {
                a: cpu(tensor_f64(&[m, k], normal_data(&[m, k], seed))),
                b: Some(cpu(tensor_f64(&[k, n], normal_data(&[k, n], seed + 1)))),
                extra: vec![],
            }
        }
        "batched_matmul" => {
            let p = &problem["batched_matmul"];
            let batch = yaml_usize(&p["batch"], 16);
            let (m, n, k) = yshape3(p, "m", "n", "k");
            EagerInputs {
                a: cpu(tensor_f64(
                    &[m, k, batch],
                    normal_data(&[m, k, batch], seed),
                )),
                b: Some(cpu(tensor_f64(
                    &[k, n, batch],
                    normal_data(&[k, n, batch], seed + 1),
                ))),
                extra: vec![],
            }
        }
        "einsum" => {
            let p = &problem["einsum"];
            let shapes = yaml_shapes(p);
            let dummy = cpu(tensor_f64(&[1], vec![0.0]));
            let extra = shapes
                .iter()
                .enumerate()
                .map(|(i, sh)| cpu(tensor_f64(sh, normal_data(sh, seed + i as u64))))
                .collect();
            EagerInputs {
                a: dummy,
                b: None,
                extra,
            }
        }
        "qr" | "svd" => {
            let p = &problem["linalg"];
            let (m, n) = (yaml_usize(&p["m"], 64), yaml_usize(&p["n"], 64));
            let d = if gen == "well_conditioned" {
                well_conditioned(m, n, seed)
            } else {
                normal_data(&[m, n], seed)
            };
            EagerInputs {
                a: cpu(tensor_f64(&[m, n], d)),
                b: None,
                extra: vec![],
            }
        }
        "solve" => {
            let p = &problem["linalg"];
            let n = yaml_usize(&p["n"], 64);
            let rhs = yaml_usize(&p["rhs_cols"], 1);
            EagerInputs {
                a: cpu(tensor_f64(&[n, n], solve_matrix_data(n, seed, gen))),
                b: Some(cpu(tensor_f64(
                    &[n, rhs],
                    normal_data(&[n, rhs], seed + 100),
                ))),
                extra: vec![],
            }
        }
        "eigh" => {
            let p = &problem["linalg"];
            let n = yaml_usize(&p["n"], 64);
            let d = if gen == "spd" {
                spd_data(n, seed)
            } else {
                normal_data(&[n, n], seed)
            };
            EagerInputs {
                a: cpu(tensor_f64(&[n, n], d)),
                b: None,
                extra: vec![],
            }
        }
        _ => {
            let dummy = cpu(tensor_f64(&[1], vec![0.0]));
            EagerInputs {
                a: dummy,
                b: None,
                extra: vec![],
            }
        }
    }
}

fn run_eager_op(
    op: &str,
    problem: &serde_yaml::Value,
    inputs: &EagerInputs,
) -> Result<Vec<EagerTensor>, String> {
    match op {
        "matmul" => {
            let out = inputs
                .a
                .matmul(inputs.b.as_ref().unwrap())
                .map_err(|e| format!("matmul: {e}"))?;
            Ok(vec![out])
        }
        "batched_matmul" => {
            let cfg = batched_matmul_cfg();
            let out = inputs
                .a
                .dot_general(inputs.b.as_ref().unwrap(), cfg)
                .map_err(|e| format!("batched_matmul: {e}"))?;
            Ok(vec![out])
        }
        "einsum" => {
            let expr = problem["einsum"]["format_rowmajor"]
                .as_str()
                .unwrap_or("ij,jk->ik");
            let refs: Vec<&EagerTensor> = inputs.extra.iter().collect();
            let out = refs.as_slice().einsum(expr).map_err(|e| format!("einsum: {e}"))?;
            Ok(vec![out])
        }
        "qr" => {
            let (q, r) = inputs.a.qr().map_err(|e| format!("qr: {e}"))?;
            Ok(vec![q, r])
        }
        "solve" => {
            let x = inputs
                .a
                .solve(inputs.b.as_ref().unwrap())
                .map_err(|e| format!("solve: {e}"))?;
            Ok(vec![x])
        }
        "svd" => {
            let (u, s, vh) = inputs.a.svd().map_err(|e| format!("svd: {e}"))?;
            Ok(vec![u, s, vh])
        }
        "eigh" => {
            let (w, v) = inputs.a.eigh().map_err(|e| format!("eigh: {e}"))?;
            Ok(vec![w, v])
        }
        _ => Err(format!("unknown op: {op}")),
    }
}

// ---------------------------------------------------------------------------
// Trace graph builders
// ---------------------------------------------------------------------------

/// Build trace graph with CPU tensors as concrete-shape constants.
fn build_trace_graph(
    op: &str,
    problem: &serde_yaml::Value,
    seed: u64,
    gen: &str,
) -> Result<Vec<TracedTensor>, TfError> {
    let (outputs, _) = build_trace_graph_inner(op, problem, seed, gen, None)?;
    Ok(outputs)
}

/// Build trace graph with GPU-uploaded tensors embedded as concrete-shape constants.
/// When rt is Some, each input tensor is uploaded to GPU before embedding.
fn build_trace_graph_gpu(
    op: &str,
    problem: &serde_yaml::Value,
    seed: u64,
    gen: &str,
    rt: &CudaRuntime,
) -> Result<(Vec<TracedTensor>, Vec<(TracedTensor, Tensor)>), TfError> {
    build_trace_graph_inner(op, problem, seed, gen, Some(rt))
}

fn build_trace_graph_inner(
    op: &str,
    problem: &serde_yaml::Value,
    seed: u64,
    gen: &str,
    upload_rt: Option<&CudaRuntime>,
) -> Result<(Vec<TracedTensor>, Vec<(TracedTensor, Tensor)>), TfError> {
    // Helper: create a TracedTensor with concrete shape from tensor data.
    // If upload_rt is Some, the data is uploaded to GPU first.
    macro_rules! inp {
        ($shape:expr, $data:expr) => {{
            let cpu_d = tensor_f64($shape, $data);
            let tensor = if let Some(rt) = upload_rt {
                upload_tensor(rt, &cpu_d).map_err(|e| TfError::Internal(format!("upload: {e}")))?
            } else {
                cpu_d.clone()
            };
            let t = TracedTensor::from_tensor_concrete_shape(tensor)?;
            (t, cpu_d)
        }};
    }

    match op {
        "matmul" => {
            let p = &problem["matmul"];
            let (m, n, k) = yshape3(p, "m", "n", "k");
            let (a, a_data) = inp!(&[m, k], normal_data(&[m, k], seed));
            let (b, b_data) = inp!(&[k, n], normal_data(&[k, n], seed + 1));
            let out = traced_tensor::matmul(&a, &b)?;
            Ok((vec![out], vec![(a, a_data), (b, b_data)]))
        }
        "batched_matmul" => {
            let p = &problem["batched_matmul"];
            let batch = yaml_usize(&p["batch"], 16);
            let (m, n, k) = yshape3(p, "m", "n", "k");
            let (a, a_data) = inp!(&[m, k, batch], normal_data(&[m, k, batch], seed));
            let (b, b_data) = inp!(&[k, n, batch], normal_data(&[k, n, batch], seed + 1));
            let cfg = batched_matmul_cfg();
            Ok((vec![a.dot_general(&b, cfg)?], vec![(a, a_data), (b, b_data)]))
        }
        "einsum" => {
            let p = &problem["einsum"];
            let expr = p["format_rowmajor"].as_str().unwrap_or("ij,jk->ik");
            let shapes = yaml_shapes(p);
            let mut inputs: Vec<(TracedTensor, Tensor)> = Vec::new();
            for (i, sh) in shapes.iter().enumerate() {
                let (t, d) = inp!(sh.as_slice(), normal_data(sh, seed + i as u64));
                inputs.push((t, d));
            }
            let tensors: Vec<&TracedTensor> = inputs.iter().map(|(t, _)| t).collect();
            let mut compiler = GraphCompiler::new();
            let out = compiler.einsum(&tensors, expr)?;
            Ok((vec![out], inputs))
        }
        "qr" => {
            let p = &problem["linalg"];
            let (m, n) = (yaml_usize(&p["m"], 64), yaml_usize(&p["n"], 64));
            let d = if gen == "well_conditioned" {
                well_conditioned(m, n, seed)
            } else {
                normal_data(&[m, n], seed)
            };
            let (a, a_data) = inp!(&[m, n], d);
            let (q, r) = a.qr()?;
            Ok((vec![q, r], vec![(a, a_data)]))
        }
        "solve" => {
            let p = &problem["linalg"];
            let n = yaml_usize(&p["n"], 64);
            let rhs = yaml_usize(&p["rhs_cols"], 1);
            let (a, a_data) = inp!(&[n, n], solve_matrix_data(n, seed, gen));
            let (b, b_data) = inp!(&[n, rhs], normal_data(&[n, rhs], seed + 100));
            let out = a.solve(&b)?;
            Ok((vec![out], vec![(a, a_data), (b, b_data)]))
        }
        "svd" => {
            let p = &problem["linalg"];
            let (m, n) = (yaml_usize(&p["m"], 64), yaml_usize(&p["n"], 64));
            let d = if gen == "well_conditioned" {
                well_conditioned(m, n, seed)
            } else {
                normal_data(&[m, n], seed)
            };
            let (a, a_data) = inp!(&[m, n], d);
            let (u, s, vh) = a.svd()?;
            Ok((vec![u, s, vh], vec![(a, a_data)]))
        }
        "eigh" => {
            let p = &problem["linalg"];
            let n = yaml_usize(&p["n"], 64);
            let d = if gen == "spd" {
                spd_data(n, seed)
            } else {
                normal_data(&[n, n], seed)
            };
            let (a, a_data) = inp!(&[n, n], d);
            let (w, v) = a.eigh()?;
            Ok((vec![w, v], vec![(a, a_data)]))
        }
        _ => Err(TfError::Internal(format!("unsupported trace op: {op}"))),
    }
}

fn run_cpu_trace(
    op: &str,
    problem: &serde_yaml::Value,
    seed: u64,
    gen: &str,
) -> Result<Vec<Tensor>, String> {
    // Use None for upload_rt → CPU constants embedded in from_tensor_concrete_shape
    let outputs = build_trace_graph(op, problem, seed, gen).map_err(|e| format!("{e}"))?;
    let output_refs: Vec<&TracedTensor> = outputs.iter().collect();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_many(&output_refs)
        .map_err(|e| format!("{e}"))?;
    let mut executor = GraphExecutor::new(CpuBackend::default());
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .map_err(|e| format!("{e}"))?;
    executor
        .register_extension(tenferro_linalg::register_runtime)
        .map_err(|e| format!("{e}"))?;
    executor.run_many(&program).map_err(|e| format!("{e}"))
}

// ---------------------------------------------------------------------------
// Synchronization
// ---------------------------------------------------------------------------

fn sync_eager_runtime(ctx: &EagerRuntime) -> Result<(), String> {
    ctx.synchronize()
        .map_err(|e| format!("eager runtime synchronize: {e}"))
}

fn sync_cubecl_runtime(rt: &CudaRuntime) -> Result<(), String> {
    // Fair GPU timing must wait for queued device work without reading result
    // tensors back to the host. A full output download makes tenferro-rs look
    // worse for large outputs, while no synchronization makes async CUDA
    // launches look too fast. Use tenferro-rs explicit synchronization API so
    // benchmark timing follows the same runtime contract as production callers.
    rt.synchronize()
        .map_err(|e| format!("cubecl runtime synchronize: {e}"))
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

fn verify_eager(
    op: &str,
    gpu: &[EagerTensor],
    cpu_outputs: &[EagerTensor],
    cpu_inputs: &EagerInputs,
    rt: &CudaRuntime,
    rtol: f64,
    atol: f64,
) -> (String, Option<f64>, Option<f64>) {
    let gpu_tensors: Vec<Tensor> = gpu
        .iter()
        .filter_map(|t| t.data().ok().and_then(|d| download_tensor(rt, &d).ok()))
        .collect();
    let cpu_tensors: Vec<Tensor> = cpu_outputs
        .iter()
        .filter_map(|t| t.data().ok())
        .collect();
    let mut input_tensors = match cpu_inputs.a.data() {
        Ok(a) => vec![a],
        Err(_) => return ("skipped".to_string(), None, None),
    };
    if let Some(b) = &cpu_inputs.b {
        if let Ok(b_data) = b.data() {
            input_tensors.push(b_data);
        }
    }
    input_tensors.extend(cpu_inputs.extra.iter().filter_map(|t| t.data().ok()));
    verify_tensors(op, &gpu_tensors, &cpu_tensors, &input_tensors, rtol, atol)
}

fn verify_tensors(
    op: &str,
    gpu: &[Tensor],
    cpu_outputs: &[Tensor],
    cpu_inputs: &[Tensor],
    rtol: f64,
    atol: f64,
) -> (String, Option<f64>, Option<f64>) {
    let comparison = match op {
        "qr" => verify_qr(gpu, cpu_inputs),
        "solve" => verify_solve(gpu, cpu_inputs),
        "svd" => verify_svd(gpu, cpu_inputs),
        "eigh" => verify_eigh(gpu, cpu_inputs),
        _ => {
            let actual = gpu.first().and_then(tensor_f64_parts);
            let expected = cpu_outputs.first().and_then(tensor_f64_parts);
            match (actual, expected) {
                (Some((_, a)), Some((_, e))) => Some((a, e)),
                _ => None,
            }
        }
    };

    let Some((actual, expected)) = comparison else {
        return ("skipped".to_string(), None, None);
    };
    compare_vectors(&actual, &expected, rtol, atol)
}

fn verify_solve(gpu: &[Tensor], cpu_inputs: &[Tensor]) -> Option<(Vec<f64>, Vec<f64>)> {
    let (x_shape, x) = gpu.first().and_then(tensor_f64_parts)?;
    let (a_shape, a) = cpu_inputs.first().and_then(tensor_f64_parts)?;
    let (b_shape, b) = cpu_inputs.get(1).and_then(tensor_f64_parts)?;
    if x_shape.len() != 2 || a_shape.len() != 2 || b_shape.len() != 2 {
        return None;
    }
    let n = a_shape[0];
    let rhs = x_shape[1];
    if a_shape.as_slice() != [n, n]
        || x_shape.as_slice() != [n, rhs]
        || b_shape.as_slice() != [n, rhs]
    {
        return None;
    }
    let ax = matmul_col_major(&a, n, n, &x, n, rhs)?;
    Some((ax, b))
}

fn verify_qr(gpu: &[Tensor], cpu_inputs: &[Tensor]) -> Option<(Vec<f64>, Vec<f64>)> {
    let (q_shape, q) = gpu.first().and_then(tensor_f64_parts)?;
    let (r_shape, r) = gpu.get(1).and_then(tensor_f64_parts)?;
    let (_, a) = cpu_inputs.first().and_then(tensor_f64_parts)?;
    if q_shape.len() != 2 || r_shape.len() != 2 {
        return None;
    }
    let recon = matmul_col_major(&q, q_shape[0], q_shape[1], &r, r_shape[0], r_shape[1])?;
    Some((recon, a))
}

fn verify_svd(gpu: &[Tensor], cpu_inputs: &[Tensor]) -> Option<(Vec<f64>, Vec<f64>)> {
    let (u_shape, u) = gpu.first().and_then(tensor_f64_parts)?;
    let (s_shape, s) = gpu.get(1).and_then(tensor_f64_parts)?;
    let (vh_shape, vh) = gpu.get(2).and_then(tensor_f64_parts)?;
    let (_, a) = cpu_inputs.first().and_then(tensor_f64_parts)?;
    if u_shape.len() != 2 || s_shape.len() != 1 || vh_shape.len() != 2 {
        return None;
    }
    let (m, k) = (u_shape[0], u_shape[1]);
    if s_shape[0] != k || vh_shape[0] != k {
        return None;
    }
    let mut us = u.clone();
    for col in 0..k {
        for row in 0..m {
            us[row + m * col] *= s[col];
        }
    }
    let recon = matmul_col_major(&us, m, k, &vh, vh_shape[0], vh_shape[1])?;
    Some((recon, a))
}

fn verify_eigh(gpu: &[Tensor], cpu_inputs: &[Tensor]) -> Option<(Vec<f64>, Vec<f64>)> {
    let (w_shape, w) = gpu.first().and_then(tensor_f64_parts)?;
    let (v_shape, v) = gpu.get(1).and_then(tensor_f64_parts)?;
    let (a_shape, a) = cpu_inputs.first().and_then(tensor_f64_parts)?;
    if w_shape.len() != 1 || v_shape.len() != 2 || a_shape.len() != 2 {
        return None;
    }
    let n = w_shape[0];
    if v_shape.as_slice() != [n, n] || a_shape.as_slice() != [n, n] {
        return None;
    }
    let av = matmul_col_major(&a, n, n, &v, n, n)?;
    let mut vl = v.clone();
    for col in 0..n {
        for row in 0..n {
            vl[row + n * col] *= w[col];
        }
    }
    Some((av, vl))
}

fn tensor_f64_parts(t: &Tensor) -> Option<(Vec<usize>, Vec<f64>)> {
    if let Tensor::F64(typed) = t {
        typed
            .as_slice()
            .ok()
            .map(|data| (typed.shape().to_vec(), data.to_vec()))
    } else {
        None
    }
}

fn matmul_col_major(
    a: &[f64],
    a_rows: usize,
    a_cols: usize,
    b: &[f64],
    b_rows: usize,
    b_cols: usize,
) -> Option<Vec<f64>> {
    if a_cols != b_rows || a.len() != a_rows * a_cols || b.len() != b_rows * b_cols {
        return None;
    }
    let mut out = vec![0.0; a_rows * b_cols];
    for col in 0..b_cols {
        for row in 0..a_rows {
            let mut acc = 0.0;
            for k in 0..a_cols {
                acc += a[row + a_rows * k] * b[k + b_rows * col];
            }
            out[row + a_rows * col] = acc;
        }
    }
    Some(out)
}

fn compare_vectors(
    actual: &[f64],
    expected: &[f64],
    rtol: f64,
    atol: f64,
) -> (String, Option<f64>, Option<f64>) {
    if actual.len() != expected.len() {
        return ("skipped".to_string(), None, None);
    }
    let max_abs = actual
        .iter()
        .zip(expected.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    let max_rel = actual
        .iter()
        .zip(expected.iter())
        .map(|(x, y)| (x - y).abs() / y.abs().max(1e-10))
        .fold(0.0_f64, f64::max);
    let norm = expected.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let passed = max_abs <= atol + rtol * norm.max(1.0);
    (
        if passed { "passed" } else { "failed" }.to_string(),
        Some(max_abs),
        Some(max_rel),
    )
}

// ---------------------------------------------------------------------------
// Data helpers
// ---------------------------------------------------------------------------

fn tensor_f64(shape: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::F64(
        TypedTensor::from_vec_col_major(shape.to_vec(), data)
            .expect("benchmark shape/data length should match"),
    )
}

fn normal_data(shape: &[usize], seed: u64) -> Vec<f64> {
    let len: usize = shape.iter().product();
    (0..len)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed.wrapping_mul(1442695040888963407));
            ((x % 1024) as f64 - 512.0) / 512.0
        })
        .collect()
}

fn well_conditioned(m: usize, n: usize, seed: u64) -> Vec<f64> {
    let mut d = normal_data(&[m, n], seed);
    for j in 0..n {
        for i in 0..m {
            d[i + m * j] *= 0.05;
        }
        if j < m {
            d[j + m * j] += 1.0 + j as f64 / n.max(1) as f64;
        }
    }
    d
}

fn spd_data(n: usize, seed: u64) -> Vec<f64> {
    let base = well_conditioned(n, n, seed);
    let mut out = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut s: f64 = (0..n).map(|k| base[k + n * i] * base[k + n * j]).sum();
            if i == j {
                s += 1.0;
            }
            out[i + n * j] = s;
        }
    }
    out
}

fn solve_matrix_data(n: usize, seed: u64, gen: &str) -> Vec<f64> {
    match gen {
        "spd" => spd_data(n, seed),
        "well_conditioned" => well_conditioned(n, n, seed),
        _ => {
            let mut d = normal_data(&[n, n], seed);
            for j in 0..n {
                d[j + n * j] += n as f64;
            }
            d
        }
    }
}

fn batched_matmul_cfg() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![2],
        rhs_batch_dims: vec![2],
    }
}

// ---------------------------------------------------------------------------
// Timing statistics
// ---------------------------------------------------------------------------

fn timing_stats(times_ms: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = times_ms.len();
    let mut s = times_ms.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let first = times_ms[0];
    let median = if n % 2 == 0 {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    } else {
        s[n / 2]
    };
    let min_t = s[0];
    let p95 = s[(n as f64 * 0.95) as usize].min(*s.last().unwrap());
    let q1 = s[((n as f64 * 0.25) as usize).max(0)];
    let q3 = s[((n as f64 * 0.75) as usize).min(n - 1)];
    (first, median, min_t, p95, q3 - q1)
}

// ---------------------------------------------------------------------------
// JSONL builders
// ---------------------------------------------------------------------------

fn stub(
    suite_id: &str,
    problem: &serde_yaml::Value,
    backend: &str,
    device_ordinal: usize,
    status: &str,
    reason: &str,
    path: &str,
    _ts: &str,
    _bc: Option<&str>,
    _tc: Option<&str>,
) -> Value {
    let pid = problem["id"].as_str().unwrap_or("");
    let op = problem["op"].as_str().unwrap_or("");
    let ver_status = if status == "verification_failed" {
        "failed"
    } else {
        "skipped"
    };
    json!({
        "schema_version": 1,
        "suite_id": suite_id,
        "problem_id": pid,
        "op": op,
        "backend": backend,
        "status": status,
        "timing": {
            "warmup_runs": 0, "timed_runs": 0,
            "compile_time_ms": null, "first_run_ms": null, "median_ms": null,
            "min_ms": null, "p95_ms": null, "iqr_ms": null,
            "timing_scope": "steady_state_host_api_plus_device_sync"
        },
        "performance": { "tflops": null, "effective_bandwidth_gbps": null, "peak_memory_bytes": null },
        "verification": {
            "status": ver_status, "reference_backend": null,
            "max_abs_error": null, "max_rel_error": null, "residual": null,
            "rtol": null, "atol": null, "reason": reason
        },
        "execution": {
            "device": "cuda", "device_ordinal": device_ordinal,
            "execution_path": path, "synchronization": "not executed",
            "layout": layout_str(problem), "dtype": dtype_str(problem),
            "notes": null, "unsupported_reason": reason
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn ok_record(
    suite_id: &str,
    problem: &serde_yaml::Value,
    backend: &str,
    device_ordinal: usize,
    path: &str,
    n_warmup: usize,
    n_runs: usize,
    stats: (f64, f64, f64, f64, f64),
    ver_status: &str,
    max_abs: Option<f64>,
    max_rel: Option<f64>,
    rtol: f64,
    atol: f64,
    _ts: &str,
    _bc: Option<&str>,
    _tc: Option<&str>,
) -> Value {
    let (first, median, min_t, p95, iqr) = stats;
    let pid = problem["id"].as_str().unwrap_or("");
    let op = problem["op"].as_str().unwrap_or("");
    json!({
        "schema_version": 1,
        "suite_id": suite_id,
        "problem_id": pid,
        "op": op,
        "backend": backend,
        "status": "ok",
        "timing": {
            "warmup_runs": n_warmup, "timed_runs": n_runs,
            "compile_time_ms": null,
            "first_run_ms": first, "median_ms": median,
            "min_ms": min_t, "p95_ms": p95, "iqr_ms": iqr,
            "timing_scope": "steady_state_host_api_plus_device_sync"
        },
        "performance": { "tflops": null, "effective_bandwidth_gbps": null, "peak_memory_bytes": null },
        "verification": {
            "status": ver_status, "reference_backend": "cpu_fp64",
            "max_abs_error": max_abs, "max_rel_error": max_rel, "residual": null,
            "rtol": rtol, "atol": atol, "reason": null
        },
        "execution": {
            "device": "cuda", "device_ordinal": device_ordinal,
            "execution_path": path,
            "synchronization": "tenferro-rs explicit synchronize API",
            "layout": layout_str(problem), "dtype": dtype_str(problem),
            "notes": tenferro_notes(op),
            "unsupported_reason": null
        }
    })
}

fn tenferro_notes(op: &str) -> &'static str {
    match op {
        "svd" => concat!(
            "timing uses tenferro-rs explicit synchronization without result download; ",
            "verification downloads outputs after timing and reconstructs U*diag(S)*Vh; ",
            "tenferro-rs uses native column-major GPU tensors and its direct cuSOLVER path, ",
            "so compare SVD against row-major torch.linalg preferred-cusolver columns with caution"
        ),
        _ => concat!(
            "timing uses tenferro-rs explicit synchronization without result download; ",
            "verification downloads outputs after timing; tenferro-rs uses native column-major GPU tensors"
        ),
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn yaml_str_list(v: &serde_yaml::Value) -> Vec<String> {
    v.as_sequence()
        .map(|s| {
            s.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

fn backend_candidates(problem: &serde_yaml::Value, suite_backends: &[String]) -> Vec<String> {
    let only_backends = yaml_str_list(&problem["only_backends"]);
    let skip_backends = yaml_str_list(&problem["skip_backends"]);
    let base = if only_backends.is_empty() {
        suite_backends.to_vec()
    } else {
        only_backends
    };
    base.into_iter()
        .filter(|backend| !skip_backends.iter().any(|skip| skip == backend))
        .collect()
}

fn yaml_usize(v: &serde_yaml::Value, default: usize) -> usize {
    v.as_u64().unwrap_or(default as u64) as usize
}

fn yshape3(v: &serde_yaml::Value, a: &str, b: &str, c: &str) -> (usize, usize, usize) {
    (
        yaml_usize(&v[a], 64),
        yaml_usize(&v[b], 64),
        yaml_usize(&v[c], 64),
    )
}

fn yaml_shapes(p: &serde_yaml::Value) -> Vec<Vec<usize>> {
    p["shapes_rowmajor"]
        .as_sequence()
        .unwrap_or(&vec![])
        .iter()
        .map(|s| {
            s.as_sequence()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_u64().unwrap_or(1) as usize)
                .collect()
        })
        .collect()
}

fn layout_str(p: &serde_yaml::Value) -> String {
    serde_json::to_string(&serde_json::json!(p["layout"].as_str().unwrap_or("")))
        .unwrap_or_default()
}

fn dtype_str(p: &serde_yaml::Value) -> String {
    serde_json::to_string(&serde_json::json!(p["dtype"].as_str().unwrap_or(""))).unwrap_or_default()
}

fn git_commit(path: &str) -> Option<String> {
    std::process::Command::new("git")
        .args(["-C", path, "rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn utc_timestamp() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%S+00:00"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "1970-01-01T00:00:00+00:00".to_string())
}
