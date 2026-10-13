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
use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{EagerSessionEinsumExt, TraceContextEinsumExt};
use tenferro_einsum_benchmark::tensornetwork::{
    build_cpu_eager_inputs as build_tensor_network_cpu_inputs, contract_tree_eager,
    contract_tree_trace, load_tensor_network, scalar_f32, scalar_f32_tensor, tensor_f32_col_major,
    TensorNetworkFile, TensorNetworkSpec, TreeNode,
};
use tenferro_gpu::cuda::{
    cuda_runtime_engine_registration, download_tensor, gpu_available, upload_tensor, CudaBackend,
    CudaRuntime,
};
use tenferro_linalg::{EagerSessionLinalgExt, TracedTensorLinalgExt};
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{
    DType, DotGeneralConfig, Error as TfError, GraphCompiler, Runtime, TraceContext, TracedTensor,
};
use tenferro_tensor::{DeviceKind, GpuBackendKind, Tensor};

fn eager_from_tensor_in(
    tensor: Tensor,
    ctx: std::sync::Arc<EagerRuntime>,
) -> Result<EagerTensor, String> {
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

    let cuda_available = gpu_available();

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
                let rec = if !cuda_available && candidates.iter().any(|c| c == backend) {
                    stub(
                        &suite_id,
                        problem,
                        backend,
                        device_ordinal,
                        "not_configured",
                        "no CUDA GPU available",
                        "phase2-runner",
                        &timestamp,
                        benchmark_commit.as_deref(),
                        tenferro_commit.as_deref(),
                    )
                } else {
                    dispatch(
                        &suite_id,
                        problem,
                        backend,
                        device_ordinal,
                        &candidates,
                        &timestamp,
                        benchmark_commit.as_deref(),
                        tenferro_commit.as_deref(),
                    )
                };
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
        "tenferro-cuda-trace" if !matches!(suite_id, "gpu/public_path_latency" | "gpu/planning_latency") => stub(
            suite_id, problem, backend, device_ordinal, "unsupported",
            "Runtime::run_prepared enters backend sessions during execution and has no caller-borrowed session route; standard operation timing cannot exclude this setup. Public-call latency diagnostics are separate.",
            "trace-session-setup-unavailable", ts, bc, tc),
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

fn constant_tree_scalar(node: &TreeNode, spec: &TensorNetworkSpec) -> Result<f32, String> {
    fn logarithm(node: &TreeNode, spec: &TensorNetworkSpec) -> Result<f64, String> {
        if node.isleaf {
            return Ok((spec.fill_value as f32 as f64).log2());
        }
        let eins = node.eins.as_ref().ok_or("missing einsum metadata")?;
        let labels = eins
            .ixs
            .iter()
            .flatten()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let output = eins
            .iy
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let summed = labels.difference(&output).count();
        let mut value = summed as f64 * (spec.bond_dim as f64).log2();
        for child in node.args.as_ref().ok_or("missing tree children")? {
            value += logarithm(child, spec)?;
        }
        Ok(value)
    }
    Ok(logarithm(node, spec)?.exp2() as f32)
}

fn verification_only() -> bool {
    std::env::var("GPU_BENCH_VERIFY_ONLY").as_deref() == Ok("1")
}

fn apply_eager_sampling_record(record: &mut Value, sampling: &Value) {
    record["execution"]["notes"] = json!(sampling.to_string());
    if verification_only() {
        record["status"] = json!("verification_only");
        record["timing"]["timed_runs"] = json!(0);
        for name in ["first_run_ms", "median_ms", "min_ms", "p95_ms", "iqr_ms"] {
            record["timing"][name] = Value::Null;
        }
    }
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
) -> Result<Vec<(Tensor, Tensor)>, String> {
    network
        .inputs
        .iter()
        .map(|ixs| {
            let shape = vec![spec.bond_dim; ixs.len()];
            let cpu = tensor_f32_col_major(&shape, spec.fill_value)?;
            let gpu = upload_tensor(upload_rt, &cpu).map_err(|e| format!("upload: {e}"))?;
            Ok((gpu, cpu))
        })
        .collect()
}

fn cuda_runtime_with_extensions(backend: &CudaBackend, linalg: bool) -> Result<Runtime, String> {
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            cuda_runtime_engine_registration(
                backend,
                tenferro_runtime::EngineId::new("benchmark.cuda").map_err(|e| e.to_string())?,
            )
            .map_err(|e| format!("CUDA engine: {e}"))?,
        )
        .map_err(|e| format!("register CUDA engine: {e}"))?;
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<CudaBackend>(
                tenferro_runtime::EngineId::new("benchmark.cuda")
                    .map_err(|e| format!("CUDA engine id: {e}"))?,
            )
            .map_err(|e| format!("einsum extension: {e}"))?,
        )
        .map_err(|e| format!("install einsum extension: {e}"))?;
    if linalg {
        builder
            .install_extension_module(
                tenferro_linalg::extension_module::<CudaBackend>(
                    tenferro_runtime::EngineId::new("benchmark.cuda")
                        .map_err(|e| format!("CUDA engine id: {e}"))?,
                )
                .map_err(|e| format!("linalg extension: {e}"))?,
            )
            .map_err(|e| format!("install linalg extension: {e}"))?;
    }
    builder
        .build()
        .map_err(|e| format!("build CUDA runtime: {e}"))
}

fn cpu_runtime_with_extensions(linalg: bool) -> Result<Runtime, String> {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            tenferro_cpu::runtime_engine_registration(&backend)
                .map_err(|e| format!("CPU engine: {e}"))?,
        )
        .map_err(|e| format!("register CPU engine: {e}"))?;
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<CpuBackend>(
                tenferro_cpu::runtime_engine_id().map_err(|e| format!("CPU engine id: {e}"))?,
            )
            .map_err(|e| format!("einsum extension: {e}"))?,
        )
        .map_err(|e| format!("install einsum extension: {e}"))?;
    if linalg {
        builder
            .install_extension_module(
                tenferro_linalg::extension_module::<CpuBackend>(
                    tenferro_cpu::runtime_engine_id().map_err(|e| format!("CPU engine id: {e}"))?,
                )
                .map_err(|e| format!("linalg extension: {e}"))?,
            )
            .map_err(|e| format!("install linalg extension: {e}"))?;
    }
    builder
        .build()
        .map_err(|e| format!("build CPU runtime: {e}"))
}

fn scalar_f32_from_eager_gpu(result: &EagerTensor, runtime: &CudaRuntime) -> Result<f32, String> {
    let downloaded =
        download_tensor(runtime, &result.data()?).map_err(|e| format!("download: {e}"))?;
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
        let case_start = Instant::now();
        let spec = TensorNetworkSpec::from_problem(problem, std::path::Path::new("."))?;
        let network = load_tensor_network(&spec.source)?;
        let network_end = case_start.elapsed().as_nanos();
        // A clone shares the backend's runtime and allocation domain.
        // Independently constructed backends target the same CUDA device but own
        // distinct allocation domains, so tensors uploaded through the transfer
        // backend are rejected by the compute engine and by the traced runtime.
        let compute_bk = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(
            u32::try_from(device_ordinal).expect("device ordinal fits u32"),
        ))
        .map_err(|e| format!("compute backend: {e}"))?;
        let transfer_bk = compute_bk.clone();
        let ctx = EagerRuntime::with_cuda_backend(compute_bk).map_err(|e| e.to_string())?;
        let backend_end = case_start.elapsed().as_nanos();
        let gpu_inputs =
            build_gpu_eager_inputs(&network, &spec, ctx.clone(), transfer_bk.runtime())?;
        sync_cubecl_runtime(transfer_bk.runtime())?;
        let prepared_tree =
            tenferro_einsum_benchmark::tensornetwork::PreparedEagerTree::new(&network.tree)?;

        let setup_end = case_start.elapsed().as_nanos();
        let _guard = ctx.no_grad();
        let (times_ms, outputs, mut sampling) = sample_eager(
            &ctx,
            transfer_bk.runtime(),
            n_warmup,
            n_runs,
            device_ordinal,
            |session, _config| {
                prepared_tree
                    .execute_in(session, &gpu_inputs)
                    .map(|out| vec![out])
            },
        )?;
        let validation_start = case_start.elapsed().as_nanos();
        let cpu_scalar = constant_tree_scalar(&network.tree, &spec)?;
        let gpu_scalar = scalar_f32_from_eager_gpu(&outputs[0], transfer_bk.runtime())?;
        let (ver_status, max_abs, max_rel) =
            verify_tensor_network_scalar(gpu_scalar, cpu_scalar, rtol, atol);

        sampling["case_wall_phases_ns"] = json!({"network_loading":network_end,
            "backend_initialization":backend_end-network_end,
            "input_generation_transfer_and_preparation":setup_end-backend_end,
            "validation":case_start.elapsed().as_nanos()-validation_start,
            "case_body_before_record_construction":case_start.elapsed().as_nanos()});
        Ok((times_ms, ver_status, max_abs, max_rel, sampling))
    }));

    match result {
        Ok(Ok((times_ms, ver_status, max_abs, max_rel, sampling))) => {
            if ver_status != "passed" {
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
            let stats = if verification_only() {
                (0., 0., 0., 0., 0.)
            } else {
                timing_stats(&times_ms)
            };
            let mut record = ok_record(
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
            );
            apply_eager_sampling_record(&mut record, &sampling);
            record["verification"]["reference_backend"] =
                json!("analytic_constant_tree_fp64_from_exact_f32_inputs");
            record
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
        // A clone shares the backend's runtime and allocation domain.
        // Independently constructed backends target the same CUDA device but own
        // distinct allocation domains, so tensors uploaded through the transfer
        // backend are rejected by the compute engine and by the traced runtime.
        let compute_bk = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(
            u32::try_from(device_ordinal).expect("device ordinal fits u32"),
        ))
        .map_err(|e| format!("compute backend: {e}"))?;
        let transfer_bk = compute_bk.clone();
        let trace_inputs = build_trace_inputs(&network, &spec, transfer_bk.runtime())?;
        let mut trace = TraceContext::new();
        let values = network
            .inputs
            .iter()
            .map(|ixs| {
                trace
                    .input(ProgramInputSpec::new(
                        DType::F32,
                        vec![spec.bond_dim; ixs.len()].into_iter().map(Into::into),
                    ))
                    .map_err(|e| format!("trace input: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = contract_tree_trace(&mut trace, &network.tree, &values)?;
        let graph = trace
            .finish(&[output])
            .map_err(|e| format!("finish trace: {e}"))?;
        sync_cubecl_runtime(transfer_bk.runtime())?;

        let program = GraphCompiler::new()
            .compile_traced_graph(&graph)
            .map_err(|e| format!("compile: {e}"))?;
        let runtime = cuda_runtime_with_extensions(&compute_bk, false)?;
        let gpu_inputs: Vec<&Tensor> = trace_inputs.iter().map(|(gpu, _)| gpu).collect();
        let prepared = runtime
            .prepare_compiled(&program, &gpu_inputs)
            .map_err(|e| format!("prepare: {e}"))?;

        for _ in 0..n_warmup.max(1) {
            let out = runtime
                .run_prepared(&prepared, &gpu_inputs)
                .map_err(|e| format!("warmup: {e}"))?;
            sync_cubecl_runtime(compute_bk.runtime())?;
            black_box(out.len());
        }

        let mut times_ms = Vec::with_capacity(n_runs);
        let mut gpu_scalar = None;
        for _ in 0..n_runs {
            let t0 = Instant::now();
            let out = runtime
                .run_prepared(&prepared, &gpu_inputs)
                .map_err(|e| format!("run: {e}"))?;
            sync_cubecl_runtime(compute_bk.runtime())?;
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            let downloaded = download_tensor(transfer_bk.runtime(), &out[0])
                .map_err(|e| format!("download: {e}"))?;
            gpu_scalar = Some(scalar_f32_tensor(&downloaded)?);
        }

        let cpu_ctx =
            EagerRuntime::with_cpu_backend(CpuBackend::default()).expect("CPU reference runtime");
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
        "elementwise_chain",
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
        let case_start = Instant::now();
        // A clone shares the backend's runtime and allocation domain.
        // Independently constructed backends target the same CUDA device but own
        // distinct allocation domains, so tensors uploaded through the transfer
        // backend are rejected by the compute engine and by the traced runtime.
        let compute_bk = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(
            u32::try_from(device_ordinal).expect("device ordinal fits u32"),
        ))
        .map_err(|e| format!("compute backend: {e}"))?;
        let transfer_bk = compute_bk.clone();
        let ctx = EagerRuntime::with_cuda_backend(compute_bk).map_err(|e| e.to_string())?;
        let backend_end = case_start.elapsed().as_nanos();

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
        let expression = problem["einsum"]["format_rowmajor"]
            .as_str()
            .unwrap_or("ij,jk->ik");
        let parsed = tenferro_einsum::Subscripts::parse(expression).map_err(|e| e.to_string())?;
        let subscripts = (&parsed).into();

        let _guard = ctx.no_grad();
        let refs = gpu_inputs.extra.iter().collect::<Vec<_>>();
        let setup_end = case_start.elapsed().as_nanos();
        let (times_ms, last, mut sampling) = sample_eager(
            &ctx,
            transfer_bk.runtime(),
            n_warmup,
            n_runs,
            device_ordinal,
            |session, config| run_eager_op_in(session, op, &subscripts, config, &gpu_inputs, &refs),
        )?;
        let last_out = Some(last);
        let validation_start = case_start.elapsed().as_nanos();

        // Verify last result against CPU reference
        let (ver_status, max_abs, max_rel) = if let Some(ref g) = last_out {
            let cpu_out = if matches!(op, "qr" | "solve" | "svd" | "eigh") {
                Vec::new()
            } else {
                run_eager_op(op, &subscripts, batched_matmul_cfg(), &cpu_inputs).unwrap_or_default()
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

        sampling["case_wall_phases_ns"] = json!({"backend_initialization":backend_end,
            "input_generation_transfer_reference_and_preparation":setup_end-backend_end,
            "validation":case_start.elapsed().as_nanos()-validation_start,
            "case_body_before_record_construction":case_start.elapsed().as_nanos()});
        Ok((times_ms, ver_status, max_abs, max_rel, sampling))
    }));

    match result {
        Ok(Ok((times_ms, ver_status, max_abs, max_rel, sampling))) => {
            if ver_status != "passed" {
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
            let stats = if verification_only() {
                (0., 0., 0., 0., 0.)
            } else {
                timing_stats(&times_ms)
            };
            let mut record = ok_record(
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
            );
            apply_eager_sampling_record(&mut record, &sampling);
            record
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

fn run_semantic_trace_einsum(
    suite_id: &str,
    problem: &serde_yaml::Value,
    backend: &str,
    device_ordinal: usize,
    ts: &str,
    bc: Option<&str>,
    tc: Option<&str>,
) -> Value {
    let n_warmup = yaml_usize(&problem["run"]["warmups"], 3);
    let n_runs = yaml_usize(&problem["run"]["runs"], 7);
    let rtol = problem["verify"]["rtol"].as_f64().unwrap_or(1e-5);
    let atol = problem["verify"]["atol"].as_f64().unwrap_or(1e-8);
    let exec_path = "phase2-measured-tenferro-cuda-trace";

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| -> Result<_, String> {
        let seed = problem["data"]["seed"].as_u64().unwrap_or(0);
        let p = &problem["einsum"];
        let expr = p["format_rowmajor"].as_str().unwrap_or("ij,jk->ik");
        let shapes = yaml_shapes(p);

        let cpu_inputs: Vec<Tensor> = shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| tensor_f64(shape, normal_data(shape, seed + i as u64)))
            .collect();

        let mut trace = TraceContext::new();
        let values = shapes
            .iter()
            .map(|shape| {
                trace
                    .input(ProgramInputSpec::new(
                        DType::F64,
                        shape.iter().copied().map(Into::into),
                    ))
                    .map_err(|e| format!("trace input: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = trace
            .einsum(&values, expr)
            .map_err(|e| format!("einsum ({expr}): {e}"))?;
        let graph = trace
            .finish(&[output])
            .map_err(|e| format!("finish trace: {e}"))?;
        let program = GraphCompiler::new()
            .compile_traced_graph(&graph)
            .map_err(|e| format!("compile: {e}"))?;

        let compute_bk = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(
            u32::try_from(device_ordinal).expect("device ordinal fits u32"),
        ))
        .map_err(|e| format!("compute backend: {e}"))?;
        let gpu_inputs = cpu_inputs
            .iter()
            .map(|tensor| {
                upload_tensor(compute_bk.runtime(), tensor).map_err(|e| format!("upload: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        sync_cubecl_runtime(compute_bk.runtime())?;
        let gpu_input_refs: Vec<&Tensor> = gpu_inputs.iter().collect();
        let runtime = cuda_runtime_with_extensions(&compute_bk, false)?;
        let prepared = runtime
            .prepare_compiled(&program, &gpu_input_refs)
            .map_err(|e| format!("prepare: {e}"))?;

        for _ in 0..n_warmup.max(1) {
            let out = runtime
                .run_prepared(&prepared, &gpu_input_refs)
                .map_err(|e| format!("warmup: {e}"))?;
            sync_cubecl_runtime(compute_bk.runtime())?;
            black_box(out.len());
        }

        let mut times_ms = Vec::with_capacity(n_runs);
        let mut last_out = None;
        for _ in 0..n_runs {
            let t0 = Instant::now();
            let out = runtime
                .run_prepared(&prepared, &gpu_input_refs)
                .map_err(|e| format!("run: {e}"))?;
            sync_cubecl_runtime(compute_bk.runtime())?;
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            last_out = Some(out);
        }

        let gpu_outputs = last_out
            .ok_or("missing timed einsum result")?
            .iter()
            .map(|tensor| {
                download_tensor(compute_bk.runtime(), tensor)
                    .map_err(|e| format!("download result: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let cpu_input_refs: Vec<&Tensor> = cpu_inputs.iter().collect();
        let cpu_outputs = cpu_runtime_with_extensions(false)?
            .run_compiled(&program, &cpu_input_refs)
            .map_err(|e| format!("CPU reference: {e}"))?;
        let (ver_status, max_abs, max_rel) = verify_tensors(
            "einsum",
            &gpu_outputs,
            &cpu_outputs,
            &cpu_inputs,
            rtol,
            atol,
        );

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
            ok_record(
                suite_id,
                problem,
                backend,
                device_ordinal,
                exec_path,
                n_warmup,
                n_runs,
                timing_stats(&times_ms),
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
            "panic during semantic einsum trace benchmark",
            exec_path,
            ts,
            bc,
            tc,
        ),
    }
}

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
    if op == "einsum" {
        return run_semantic_trace_einsum(suite_id, problem, backend, device_ordinal, ts, bc, tc);
    }
    let supported = [
        "matmul",
        "batched_matmul",
        "elementwise_chain",
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
        // A clone shares the backend's runtime and allocation domain.
        // Independently constructed backends target the same CUDA device but own
        // distinct allocation domains, so tensors uploaded through the transfer
        // backend are rejected by the compute engine and by the traced runtime.
        let compute_bk = CudaBackend::new(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(
            u32::try_from(device_ordinal).expect("device ordinal fits u32"),
        ))
        .map_err(|e| format!("compute backend: {e}"))?;
        let transfer_bk = compute_bk.clone();

        let (outputs, cpu_inputs) =
            build_trace_graph_gpu(op, problem, seed, gen, transfer_bk.runtime())
                .map_err(|e| format!("graph build: {e}"))?;
        sync_cubecl_runtime(transfer_bk.runtime())?;
        let output_refs: Vec<&TracedTensor> = outputs.iter().collect();
        let mut compiler = GraphCompiler::new();
        let program = compiler
            .compile_many(&output_refs)
            .map_err(|e| format!("compile: {e}"))?;

        let runtime = cuda_runtime_with_extensions(&compute_bk, true)?;
        let prepared = runtime
            .prepare_compiled(&program, &[])
            .map_err(|e| format!("prepare: {e}"))?;
        // Warmup (triggers JIT compilation on first run)
        for _ in 0..n_warmup.max(1) {
            let out = runtime
                .run_prepared(&prepared, &[])
                .map_err(|e| format!("warmup: {e}"))?;
            sync_cubecl_runtime(compute_bk.runtime())?;
            black_box(out.len());
        }

        // Timed runs
        let mut times_ms = Vec::with_capacity(n_runs);
        let mut last_out: Option<Vec<Tensor>> = None;
        for _ in 0..n_runs {
            let t0 = Instant::now();
            let out = runtime
                .run_prepared(&prepared, &[])
                .map_err(|e| format!("run: {e}"))?;
            sync_cubecl_runtime(compute_bk.runtime())?;
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
            let cpu_inputs: Vec<Tensor> = cpu_inputs
                .iter()
                .map(|(_, t)| t.duplicate())
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?;
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
        "elementwise_chain" => {
            let p = &problem["elementwise_chain"];
            let n = yaml_usize(&p["n"], 1024);
            Ok(EagerInputs {
                a: upload(tensor_f64(&[n], normal_data(&[n], seed)))?,
                b: Some(upload(tensor_f64(&[n], normal_data(&[n], seed + 1)))?),
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
    let cpu_ctx =
        EagerRuntime::with_cpu_backend(CpuBackend::default()).expect("CPU reference runtime");
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
        "elementwise_chain" => {
            let p = &problem["elementwise_chain"];
            let n = yaml_usize(&p["n"], 1024);
            EagerInputs {
                a: cpu(tensor_f64(&[n], normal_data(&[n], seed))),
                b: Some(cpu(tensor_f64(&[n], normal_data(&[n], seed + 1)))),
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
    subscripts: &tenferro_einsum::EinsumSubscripts,
    batched_config: DotGeneralConfig,
    inputs: &EagerInputs,
) -> Result<Vec<EagerTensor>, String> {
    let anchor = if op == "einsum" {
        &inputs.extra[0]
    } else {
        &inputs.a
    };
    let refs = inputs.extra.iter().collect::<Vec<_>>();
    anchor
        .runtime()
        .with_eager_session(|session| {
            run_eager_op_in(session, op, subscripts, batched_config, inputs, &refs)
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })
        })
        .map_err(|e| e.to_string())
}

fn run_eager_op_in(
    session: &mut EagerSession<'_>,
    op: &str,
    subscripts: &tenferro_einsum::EinsumSubscripts,
    batched_config: DotGeneralConfig,
    inputs: &EagerInputs,
    refs: &[&EagerTensor],
) -> Result<Vec<EagerTensor>, String> {
    let result: Result<Vec<EagerTensor>, tenferro_ad::Error> = (|| {
        Ok(match op {
            "matmul" => vec![session.matmul(&inputs.a, inputs.b.as_ref().unwrap())?],
            "elementwise_chain" => {
                let a = &inputs.a;
                let b = inputs.b.as_ref().unwrap();
                let mut value = a.clone();
                for _ in 0..ELEMENTWISE_CHAIN_STEPS {
                    let product = session.mul(&value, a)?;
                    let sum = session.add(&product, b)?;
                    value = session.tanh(&sum)?;
                }
                vec![value]
            }
            "batched_matmul" => {
                vec![session.dot_general(&inputs.a, inputs.b.as_ref().unwrap(), batched_config)?]
            }
            "einsum" => vec![session
                .einsum_subscripts(refs, subscripts)
                .map_err(|e| TfError::Internal(e.to_string()))?],
            "qr" => {
                let (q, r) = session.qr(&inputs.a)?;
                vec![q, r]
            }
            "solve" => vec![session.solve(&inputs.a, inputs.b.as_ref().unwrap())?],
            "svd" => {
                let (u, s, vh) = session.svd(&inputs.a)?;
                vec![u, s, vh]
            }
            "eigh" => {
                let (w, v) = session.eigh(&inputs.a)?;
                vec![w, v]
            }
            _ => {
                return Err(tenferro_ad::Error::from(TfError::Internal(format!(
                    "unknown eager operation {op}"
                ))))
            }
        })
    })();
    result.map_err(|e| e.to_string())
}

fn sample_eager(
    ctx: &std::sync::Arc<EagerRuntime>,
    rt: &CudaRuntime,
    warmups: usize,
    runs: usize,
    device_ordinal: usize,
    mut op: impl FnMut(&mut EagerSession<'_>, DotGeneralConfig) -> Result<Vec<EagerTensor>, String>,
) -> Result<(Vec<f64>, Vec<EagerTensor>, Value), String> {
    let sampler_start = Instant::now();
    ctx.with_eager_session(|session| {
        let result = (|| -> Result<_, String> {
            if verification_only() {
                let output = op(session, batched_matmul_cfg())?;
                sync_cubecl_runtime(rt)?;
                assert_cuda_eager_outputs(&output, device_ordinal)?;
                return Ok((Vec::new(), output, json!({"sampling":null,"verification_only":true,
                    "wall_phases_ns":{"session_entry_and_initial_probe":sampler_start.elapsed().as_nanos()},
                    "device_uuid":std::env::var("CUDA_VISIBLE_DEVICES").ok()})));
            }
            let mut batch = |count: usize| -> Result<_, String> {
                let mut outputs = Vec::with_capacity(count);
                let mut configs = (0..count).map(|_| Some(batched_matmul_cfg())).collect::<Vec<_>>();
                sync_cubecl_runtime(rt)?;
                let start = Instant::now();
                for config in &mut configs {
                    outputs.push(op(session, config.take().unwrap())?);
                }
                sync_cubecl_runtime(rt)?;
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                black_box(&outputs);
                Ok((elapsed, outputs))
            };
            let (_, initial) = batch(1)?;
            assert_cuda_eager_outputs(&initial[0], device_ordinal)?;
            let bytes = initial[0].iter().map(|value| {
                let view = value.tensor_read();
                let width = match view.dtype() { DType::F32 => 4, DType::C64 => 16, _ => 8 };
                view.shape().iter().product::<usize>() * width
            }).sum::<usize>().max(1);
            drop(initial);
            let probe_end = sampler_start.elapsed().as_nanos();
            for _ in 0..warmups { drop(batch(1)?); }
            let warmup_end = sampler_start.elapsed().as_nanos();
            let cap = ((256 << 20) / bytes).clamp(1, 4096);
            let mut count = 1;
            while batch(count)?.0 < 2. && count < cap { count = (count * 2).min(cap); }
            let calibration_end = sampler_start.elapsed().as_nanos();
            let mut raw = Vec::with_capacity(runs);
            let mut last = None;
            for _ in 0..runs {
                drop(last.take());
                let (elapsed, mut outputs) = batch(count)?;
                raw.push(elapsed);
                last = outputs.pop();
            }
            let sampling_end = sampler_start.elapsed().as_nanos();
            let normalized = raw.iter().map(|time| time / count as f64).collect();
            let notes = json!({"sampling":{"operations_per_sample":count,"batch_ms":raw,
                "target_batch_ms":2,"retention_budget_bytes":256<<20},
                "setup":"one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks",
                "precision":"native declared dtype; no AD recording",
                "wall_phases_ns":{"session_entry_and_initial_probe":probe_end,
                    "warmup":warmup_end-probe_end,"calibration":calibration_end-warmup_end,
                    "sampling":sampling_end-calibration_end},
                "wall_phase_scope":"phase wall time includes untimed preparation/cleanup; outside operation clocks",
                "device_uuid":std::env::var("CUDA_VISIBLE_DEVICES").ok()});
            let output = last.ok_or("no timed eager samples")?;
            assert_cuda_eager_outputs(&output, device_ordinal)?;
            Ok((normalized, output, notes))
        })();
        result.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })
    }).map_err(|e| e.to_string())
}

fn assert_cuda_eager_outputs(outputs: &[EagerTensor], device_ordinal: usize) -> Result<(), String> {
    for output in outputs {
        let read = output.tensor_read();
        let tensor = read
            .as_tensor()
            .ok_or("CUDA operation returned a lazy view")?;
        let placement = tensor.placement();
        let device = placement
            .device
            .as_ref()
            .ok_or("CPU fallback: no CUDA placement")?;
        if device.kind != DeviceKind::Gpu(GpuBackendKind::Cuda)
            || device.ordinal != device_ordinal
            || !tensor.is_backend_buffer()
        {
            return Err("CPU fallback or wrong CUDA output allocation".into());
        }
    }
    Ok(())
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
                cpu_d.duplicate()?
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
            Ok((
                vec![a.dot_general(&b, cfg)?],
                vec![(a, a_data), (b, b_data)],
            ))
        }
        "einsum" => Err(TfError::Internal(
            "einsum uses the semantic TraceContext runner".to_string(),
        )),
        "elementwise_chain" => {
            let p = &problem["elementwise_chain"];
            let n = yaml_usize(&p["n"], 1024);
            let (a, a_data) = inp!(&[n], normal_data(&[n], seed));
            let (b, b_data) = inp!(&[n], normal_data(&[n], seed + 1));
            let mut t = a.clone();
            for _ in 0..ELEMENTWISE_CHAIN_STEPS {
                t = t.mul(&a)?.add(&b)?.tanh()?;
            }
            Ok((vec![t], vec![(a, a_data), (b, b_data)]))
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
    cpu_runtime_with_extensions(true)?
        .run_compiled(&program, &[])
        .map_err(|e| format!("{e}"))
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
    let cpu_tensors: Vec<Tensor> = cpu_outputs.iter().filter_map(|t| t.data().ok()).collect();
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
    let mut result = compare_vectors(&actual, &expected, rtol, atol);
    let checks = match op {
        "qr" => gpu.first().and_then(|q| orthogonality(q, false)).map(|x| vec![x]),
        "svd" => gpu.first().and_then(|u| orthogonality(u, false)).and_then(|u| {
            gpu.get(2).and_then(|v| orthogonality(v, true)).map(|v| vec![u, v])
        }),
        "eigh" => gpu.get(1).and_then(|v| orthogonality(v, false)).map(|x| vec![x]),
        _ => Some(Vec::new()),
    };
    let Some(checks) = checks else { return ("failed".into(), None, None); };
    for (actual, expected) in checks {
        let check = compare_vectors(&actual, &expected, rtol, atol);
        if check.0 != "passed" { result.0 = "failed".into(); }
        result.1 = result.1.zip(check.1).map(|(a,b)| a.max(b));
        result.2 = result.2.zip(check.2).map(|(a,b)| a.max(b));
    }
    if op == "qr" {
        let Some((shape,r)) = gpu.get(1).and_then(tensor_f64_parts) else { return ("failed".into(),None,None); };
        let (m,n) = (shape[0],shape[1]);
        let lower = (0..n).flat_map(|j| (j+1..m).map(move |i| i+m*j))
            .map(|index| r[index]).collect::<Vec<_>>();
        let check = compare_vectors(&lower,&vec![0.0;lower.len()],rtol,atol);
        if check.0 != "passed" { result.0 = "failed".into(); }
        result.1 = result.1.zip(check.1).map(|(a,b)| a.max(b));
        result.2 = result.2.zip(check.2).map(|(a,b)| a.max(b));
    }
    if matches!(op,"svd"|"eigh") {
        let index = if op == "svd" {1} else {0};
        let Some((_,values)) = gpu.get(index).and_then(tensor_f64_parts) else { return ("failed".into(),None,None); };
        if values.iter().any(|v| !v.is_finite() || (op == "svd" && *v < 0.0)) ||
            values.windows(2).any(|w| if op == "svd" {w[0]<w[1]}else{w[0]>w[1]}) {
            result.0 = "failed".into();
        }
    }
    result
}

fn orthogonality(tensor: &Tensor, rows_are_vectors: bool) -> Option<(Vec<f64>, Vec<f64>)> {
    let (shape, data) = tensor_f64_parts(tensor)?;
    if shape.len() != 2 || shape[0].checked_mul(shape[1])? != data.len() { return None; }
    let (m,n) = (shape[0],shape[1]);
    let mut transpose = vec![0.0; data.len()];
    for j in 0..n { for i in 0..m { transpose[j+n*i] = data[i+m*j]; } }
    let (gram,size) = if rows_are_vectors {
        (matmul_col_major(&data,m,n,&transpose,n,m)?,m)
    } else { (matmul_col_major(&transpose,n,m,&data,m,n)?,n) };
    let mut identity = vec![0.0; size.checked_mul(size)?];
    for i in 0..size { identity[i+size*i] = 1.0; }
    Some((gram,identity))
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
    let (a_shape, a) = cpu_inputs.first().and_then(tensor_f64_parts)?;
    if q_shape.len() != 2 || r_shape.len() != 2 {
        return None;
    }
    if a_shape.as_slice() != [q_shape[0],r_shape[1]] { return None; }
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
    t.as_slice::<f64>()
        .ok()
        .map(|data| (t.shape().to_vec(), data.to_vec()))
}

fn matmul_col_major(
    a: &[f64],
    a_rows: usize,
    a_cols: usize,
    b: &[f64],
    b_rows: usize,
    b_cols: usize,
) -> Option<Vec<f64>> {
    if a_cols != b_rows || a.len() != a_rows.checked_mul(a_cols)? || b.len() != b_rows.checked_mul(b_cols)? {
        return None;
    }
    let mut out = vec![0.0; a_rows.checked_mul(b_cols)?];
    let a_stride = isize::try_from(a_rows).ok()?;
    let b_stride = isize::try_from(b_rows).ok()?;
    if a_rows == 0 || b_cols == 0 || a_cols == 0 { return Some(out); }
    // Bounds were checked above. Inputs may alias each other, but the newly
    // allocated destination cannot alias either input. All layouts are compact
    // column-major. This independent blocked CPU product is verification only.
    unsafe {
        matrixmultiply::dgemm(a_rows, a_cols, b_cols, 1.0,
            a.as_ptr(),1,a_stride,b.as_ptr(),1,b_stride,0.0,out.as_mut_ptr(),1,a_stride);
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
    if actual.iter().chain(expected).any(|value| !value.is_finite()) {
        return ("failed".into(), None, None);
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

#[cfg(test)]
mod verification_tests {
    use super::*;
    fn matrix(data: Vec<f64>) -> Tensor { Tensor::from_vec_col_major(vec![2,2],data).unwrap() }
    fn copy(t: &Tensor) -> Tensor { Tensor::from_vec_col_major(t.shape().to_vec(),t.as_slice::<f64>().unwrap().to_vec()).unwrap() }
    fn identity() -> Tensor { matrix(vec![1.,0.,0.,1.]) }

    #[test]
    fn blocked_product_respects_rectangular_column_major_layout() {
        let result = matmul_col_major(&[1.,4.,2.,5.,3.,6.],2,3,&[7.,9.,11.,8.,10.,12.],3,2).unwrap();
        assert_eq!(result,vec![58.,139.,64.,154.]);
        assert!(matmul_col_major(&[],usize::MAX,2,&[],2,1).is_none());
    }
    #[test]
    fn nan_cannot_disappear_in_max_error_fold() {
        assert_eq!(compare_vectors(&[f64::NAN],&[0.],1e-8,1e-10).0,"failed");
    }
    #[test]
    fn qr_checks_orthogonality_and_triangularity() {
        let r = matrix(vec![2.,0.,1.,3.]);
        assert_eq!(verify_tensors("qr",&[identity(),copy(&r)],&[],&[r],1e-8,1e-10).0,"passed");
        let a = matrix(vec![2.,0.,1.,3.]);
        let scaled_q = matrix(vec![2.,0.,0.,2.]);
        let scaled_r = matrix(vec![1.,0.,0.5,1.5]);
        assert_eq!(verify_tensors("qr",&[scaled_q,scaled_r],&[],&[a],1e-8,1e-10).0,"failed");
        let dense = matrix(vec![2.,1.,1.,3.]);
        assert_eq!(verify_tensors("qr",&[identity(),copy(&dense)],&[],&[dense],1e-8,1e-10).0,"failed");
    }
    #[test]
    fn eigh_rejects_zero_vectors_and_unsorted_spectrum() {
        let a = matrix(vec![2.,0.,0.,3.]);
        let w = Tensor::from_vec_col_major(vec![2],vec![2.,3.]).unwrap();
        assert_eq!(verify_tensors("eigh",&[copy(&w),identity()],&[],&[copy(&a)],1e-8,1e-10).0,"passed");
        assert_eq!(verify_tensors("eigh",&[w,matrix(vec![0.;4])],&[],&[copy(&a)],1e-8,1e-10).0,"failed");
        let reverse = Tensor::from_vec_col_major(vec![2],vec![3.,2.]).unwrap();
        assert_eq!(verify_tensors("eigh",&[reverse,matrix(vec![0.,1.,1.,0.])],&[],&[a],1e-8,1e-10).0,"failed");
    }
    #[test]
    fn svd_rejects_negative_singular_values_despite_reconstruction() {
        let a = matrix(vec![3.,0.,0.,2.]);
        let s = Tensor::from_vec_col_major(vec![2],vec![3.,2.]).unwrap();
        assert_eq!(verify_tensors("svd",&[identity(),s,identity()],&[],&[copy(&a)],1e-8,1e-10).0,"passed");
        let negative = Tensor::from_vec_col_major(vec![2],vec![-3.,-2.]).unwrap();
        assert_eq!(verify_tensors("svd",&[matrix(vec![-1.,0.,0.,-1.]),negative,identity()],&[],&[a],1e-8,1e-10).0,"failed");
    }
}

// ---------------------------------------------------------------------------
// Data helpers
// ---------------------------------------------------------------------------

fn tensor_f64(shape: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), data)
        .expect("benchmark shape/data length should match")
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

/// Number of chained elementwise steps in `elementwise_chain` problems.
///
/// The chain is `t = tanh(t * a + b)` repeated this many times (three ops per
/// step). `scripts/benchmark_gpu_python.py` uses the same length.
const ELEMENTWISE_CHAIN_STEPS: usize = 8;

fn batched_matmul_cfg() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1].into(),
        rhs_contracting_dims: vec![0].into(),
        lhs_batch_dims: vec![2].into(),
        rhs_batch_dims: vec![2].into(),
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
        "tensor_network_contract" => concat!(
            "timing uses tenferro-rs explicit synchronization without result download; ",
            "verification downloads outputs after timing; tenferro-rs uses native column-major GPU tensors; ",
            "eager route: one borrowed EagerSession per tree execution (EagerSessionEinsumExt), ",
            "not one session per contraction node as before tenferro-rs #1946"
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
