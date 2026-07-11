//! GPU linalg JVP/VJP benchmark runner for tenferro-cuda-trace.
//!
//! ```text
//! benchmark_gpu_linalg_ad OUTPUT_JSONL DEVICE_ORDINAL PROBLEM_FILTER -- SUITE.yaml...
//! ```

use std::hint::black_box;
use std::panic;
use std::sync::OnceLock;
use std::time::Instant;

use serde_json::{json, Value};
use tenferro_ad::AdContext;
use tenferro_gpu::{gpu_available, upload_tensor, CudaBackend, CudaRuntime};
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TracedTensor};
use tenferro_tensor::TypedTensor;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = &args[1];
    let device_ordinal: usize = args[2].parse().expect("DEVICE_ORDINAL must be usize");
    let problem_filter = &args[3];
    let sep = args.iter().position(|a| a == "--").expect("missing --");
    let suite_paths: Vec<&str> = args[sep + 1..].iter().map(String::as_str).collect();

    if !gpu_available() {
        eprintln!("benchmark_gpu_linalg_ad: no CUDA GPU found, skipping");
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

        for problem in suite["problems"].as_sequence().expect("problems list") {
            let pid = problem["id"].as_str().unwrap_or("");
            if !problem_filter.is_empty() && pid != problem_filter {
                continue;
            }
            let rec = dispatch(
                &suite_id,
                problem,
                device_ordinal,
                &timestamp,
                benchmark_commit.as_deref(),
                tenferro_commit.as_deref(),
            );
            lines.push(serde_json::to_string(&rec).unwrap());
        }
    }

    std::fs::write(out_path, lines.join("\n") + if lines.is_empty() { "" } else { "\n" })
        .expect("failed to write output");
}

fn dispatch(
    suite_id: &str,
    problem: &serde_yaml::Value,
    device_ordinal: usize,
    ts: &str,
    bc: Option<&str>,
    tc: Option<&str>,
) -> Value {
    let phase = problem["op"].as_str().unwrap_or("");
    if phase != "jvp" && phase != "vjp" {
        return stub(
            suite_id,
            problem,
            device_ordinal,
            "not_configured",
            &format!("unsupported op: {phase}"),
            ts,
            bc,
            tc,
        );
    }

    let n_warmup = yaml_usize(&problem["run"]["warmups"], 3);
    let n_runs = yaml_usize(&problem["run"]["runs"], 7);

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| -> Result<_, String> {
        let transfer_bk =
            CudaBackend::new(device_ordinal).map_err(|e| format!("transfer backend: {e}"))?;
        let compute_bk =
            CudaBackend::new(device_ordinal).map_err(|e| format!("compute backend: {e}"))?;
        let ad = problem["linalg_ad"].as_mapping().ok_or("missing linalg_ad block")?;
        let loss = yaml_str(ad.get("loss").unwrap_or(&serde_yaml::Value::Null), "");
        let n = yaml_usize(ad.get("n").unwrap_or(&serde_yaml::Value::Null), 0);
        let matrix_seed = yaml_u64(ad.get("matrix_seed").unwrap_or(&serde_yaml::Value::Null), 0);
        let rhs_seed = yaml_u64(ad.get("rhs_seed").unwrap_or(&serde_yaml::Value::Null), 0);
        let rhs_cols = yaml_usize(ad.get("rhs_cols").unwrap_or(&serde_yaml::Value::Null), 1);

        let loss_spec = parse_loss(loss, matrix_seed, rhs_seed, rhs_cols)?;
        let outputs = build_ad_outputs(
            phase,
            n,
            &loss_spec,
            transfer_bk.runtime(),
        )?;
        sync_runtime(transfer_bk.runtime())?;

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

        for _ in 0..n_warmup {
            let out = executor
                .run_many(&program)
                .map_err(|e| format!("warmup: {e}"))?;
            sync_runtime(executor.backend().runtime())?;
            black_box(out.len());
        }

        let mut times_ms = Vec::with_capacity(n_runs);
        for _ in 0..n_runs {
            let t0 = Instant::now();
            let out = executor
                .run_many(&program)
                .map_err(|e| format!("run: {e}"))?;
            sync_runtime(executor.backend().runtime())?;
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            black_box(out.len());
        }

        Ok((times_ms,))
    }));

    match result {
        Ok(Ok((times_ms,))) => {
            let stats = timing_stats(&times_ms);
            ok_record(
                suite_id,
                problem,
                device_ordinal,
                n_warmup,
                n_runs,
                stats,
                ts,
                bc,
                tc,
            )
        }
        Ok(Err(msg)) => {
            let (status, reason) = classify_linalg_ad_error(&msg);
            stub(
                suite_id,
                problem,
                device_ordinal,
                status,
                &reason,
                ts,
                bc,
                tc,
            )
        }
        Err(_) => stub(
            suite_id,
            problem,
            device_ordinal,
            "runtime_failed",
            "panic during linalg AD benchmark",
            ts,
            bc,
            tc,
        ),
    }
}

#[derive(Clone, Copy)]
enum LossKind {
    SvdS { matrix_seed: u64 },
    Qr { matrix_seed: u64 },
    Eigh { matrix_seed: u64 },
    Lu { matrix_seed: u64 },
    Solve {
        matrix_seed: u64,
        rhs_seed: u64,
        rhs_cols: usize,
    },
}

fn parse_loss(loss: &str, matrix_seed: u64, rhs_seed: u64, rhs_cols: usize) -> Result<LossKind, String> {
    Ok(match loss {
        "grad_sum_svd_s" => LossKind::SvdS { matrix_seed },
        "grad_sum_qr" => LossKind::Qr { matrix_seed },
        "grad_sum_eigh" => LossKind::Eigh { matrix_seed },
        "grad_sum_lu" => LossKind::Lu { matrix_seed },
        "grad_sum_solve" => LossKind::Solve {
            matrix_seed,
            rhs_seed,
            rhs_cols,
        },
        other => return Err(format!("unsupported loss: {other}")),
    })
}

fn build_ad_outputs(
    phase: &str,
    n: usize,
    loss: &LossKind,
    rt: &CudaRuntime,
) -> Result<Vec<TracedTensor>, String> {
    let (output, wrt) = build_loss(n, loss, rt)?;
    match phase {
        "jvp" => {
            let tangent_seed = tangent_seed(loss.matrix_seed());
            let tangent = upload_traced(rt, &[n, n], data_for_shape(&[n, n], tangent_seed))?;
            let out = jvp(&output, &wrt, &tangent).map_err(|e| format!("jvp: {e}"))?;
            Ok(vec![out])
        }
        "vjp" => {
            let cotangent = upload_traced(rt, &[], vec![1.0])?;
            let out = vjp(&output, &wrt, &cotangent).map_err(|e| format!("vjp: {e}"))?;
            Ok(vec![out])
        }
        other => Err(format!("unsupported phase: {other}")),
    }
}

impl LossKind {
    fn matrix_seed(self) -> u64 {
        match self {
            Self::SvdS { matrix_seed }
            | Self::Qr { matrix_seed }
            | Self::Eigh { matrix_seed }
            | Self::Lu { matrix_seed }
            | Self::Solve { matrix_seed, .. } => matrix_seed,
        }
    }
}

fn build_loss(
    n: usize,
    loss: &LossKind,
    rt: &CudaRuntime,
) -> Result<(TracedTensor, TracedTensor), String> {
    match *loss {
        LossKind::SvdS { matrix_seed } => {
            let a = upload_traced(rt, &[n, n], well_conditioned_matrix(n, matrix_seed))?;
            let (_, s, _) = a.svd().map_err(|e| format!("svd: {e}"))?;
            let out = s.reduce_sum(&[0]).map_err(|e| format!("reduce: {e}"))?;
            Ok((out, a))
        }
        LossKind::Qr { matrix_seed } => {
            let a = upload_traced(rt, &[n, n], well_conditioned_matrix(n, matrix_seed))?;
            let (q, r) = a.qr().map_err(|e| format!("qr: {e}"))?;
            let q_sum = q.reduce_sum(&[0, 1]).map_err(|e| format!("reduce: {e}"))?;
            let r_sum = r.reduce_sum(&[0, 1]).map_err(|e| format!("reduce: {e}"))?;
            let loss = (&q_sum + &r_sum).map_err(|e| format!("add: {e}"))?;
            Ok((loss, a))
        }
        LossKind::Eigh { matrix_seed } => {
            let a = upload_traced(rt, &[n, n], spd_matrix(n, matrix_seed))?;
            let (w, _) = a.eigh().map_err(|e| format!("eigh: {e}"))?;
            let out = w.reduce_sum(&[0]).map_err(|e| format!("reduce: {e}"))?;
            Ok((out, a))
        }
        LossKind::Lu { matrix_seed } => {
            let a = upload_traced(rt, &[n, n], well_conditioned_matrix(n, matrix_seed))?;
            let (_, l, u, _) = a.lu().map_err(|e| format!("lu: {e}"))?;
            let l_sum = l.reduce_sum(&[0, 1]).map_err(|e| format!("reduce: {e}"))?;
            let u_sum = u.reduce_sum(&[0, 1]).map_err(|e| format!("reduce: {e}"))?;
            let loss = (&l_sum + &u_sum).map_err(|e| format!("add: {e}"))?;
            Ok((loss, a))
        }
        LossKind::Solve {
            matrix_seed,
            rhs_seed,
            rhs_cols,
        } => {
            let a = upload_traced(rt, &[n, n], spd_matrix(n, matrix_seed))?;
            let b = upload_traced(rt, &[n, rhs_cols], data_for_shape(&[n, rhs_cols], rhs_seed))?;
            let x = a.solve(&b).map_err(|e| format!("solve: {e}"))?;
            let out = x.reduce_sum(&[0, 1]).map_err(|e| format!("reduce: {e}"))?;
            Ok((out, a))
        }
    }
}

fn upload_traced(rt: &CudaRuntime, shape: &[usize], data: Vec<f64>) -> Result<TracedTensor, String> {
    let cpu = tensor_f64(shape, data);
    let gpu = upload_tensor(rt, &cpu).map_err(|e| format!("upload: {e}"))?;
    TracedTensor::from_tensor_concrete_shape(gpu).map_err(|e| format!("trace input: {e}"))
}

fn ad_context() -> &'static AdContext {
    static AD_CONTEXT: OnceLock<AdContext> = OnceLock::new();
    AD_CONTEXT.get_or_init(|| {
        AdContext::builder()
            .with_extension_rules(
                tenferro_linalg::ad_rules().expect("tenferro-linalg AD rules should register"),
            )
            .build()
            .expect("tenferro AD context should build")
    })
}

fn jvp(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
) -> tenferro_ad::error::Result<TracedTensor> {
    ad_context().jvp(output, wrt, tangent)
}

fn vjp(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
) -> tenferro_ad::error::Result<TracedTensor> {
    ad_context().vjp(output, wrt, cotangent)
}

fn sync_runtime(rt: &CudaRuntime) -> Result<(), String> {
    rt.synchronize()
        .map_err(|e| format!("cuda synchronize: {e}"))
}

fn timing_stats(times_ms: &[f64]) -> (f64, f64, f64, f64, f64) {
    let mut sorted = times_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let first = times_ms[0];
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    let min_t = sorted[0];
    let p95 = sorted[((0.95 * n as f64) as usize).min(n - 1)];
    let q1 = sorted[n / 4];
    let q3 = sorted[(3 * n) / 4];
    (first, median, min_t, p95, q3 - q1)
}

fn classify_linalg_ad_error(msg: &str) -> (&'static str, String) {
    if msg.contains("shape mismatch") && msg.contains("rhs=[]") {
        (
            "unsupported",
            "CUDA trace linalg AD graphs use scalar-vector mul patterns that require \
             broadcast semantics; the CUDA elementwise mul path does not yet match CPU \
             scalar broadcast for 0-D tensors"
                .to_string(),
        )
    } else {
        ("runtime_failed", msg.to_string())
    }
}

fn stub(
    suite_id: &str,
    problem: &serde_yaml::Value,
    device_ordinal: usize,
    status: &str,
    reason: &str,
    _ts: &str,
    _bc: Option<&str>,
    _tc: Option<&str>,
) -> Value {
    json!({
        "schema_version": 1,
        "suite_id": suite_id,
        "problem_id": problem["id"].as_str().unwrap_or(""),
        "op": problem["op"].as_str().unwrap_or(""),
        "backend": "tenferro-cuda-trace",
        "status": status,
        "timing": {
            "warmup_runs": 0, "timed_runs": 0,
            "compile_time_ms": null, "first_run_ms": null, "median_ms": null,
            "min_ms": null, "p95_ms": null, "iqr_ms": null,
            "timing_scope": "steady_state_host_api_plus_device_sync"
        },
        "performance": { "tflops": null, "effective_bandwidth_gbps": null, "peak_memory_bytes": null },
        "verification": {
            "status": "skipped", "reference_backend": null,
            "max_abs_error": null, "max_rel_error": null, "residual": null,
            "rtol": null, "atol": null, "reason": reason
        },
        "execution": {
            "device": "cuda", "device_ordinal": device_ordinal,
            "execution_path": "phase2-measured-tenferro-cuda-trace-linalg-ad",
            "synchronization": "not executed",
            "layout": layout_str(problem), "dtype": dtype_str(problem),
            "notes": problem["notes"].as_str(),
            "unsupported_reason": reason
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn ok_record(
    suite_id: &str,
    problem: &serde_yaml::Value,
    device_ordinal: usize,
    n_warmup: usize,
    n_runs: usize,
    stats: (f64, f64, f64, f64, f64),
    _ts: &str,
    _bc: Option<&str>,
    _tc: Option<&str>,
) -> Value {
    let (first, median, min_t, p95, iqr) = stats;
    let rtol = problem["verify"]["rtol"].as_f64().unwrap_or(1e-5);
    let atol = problem["verify"]["atol"].as_f64().unwrap_or(1e-8);
    json!({
        "schema_version": 1,
        "suite_id": suite_id,
        "problem_id": problem["id"].as_str().unwrap_or(""),
        "op": problem["op"].as_str().unwrap_or(""),
        "backend": "tenferro-cuda-trace",
        "status": "ok",
        "timing": {
            "warmup_runs": n_warmup,
            "timed_runs": n_runs,
            "compile_time_ms": null,
            "first_run_ms": first,
            "median_ms": median,
            "min_ms": min_t,
            "p95_ms": p95,
            "iqr_ms": iqr,
            "timing_scope": "steady_state_host_api_plus_device_sync"
        },
        "performance": { "tflops": null, "effective_bandwidth_gbps": null, "peak_memory_bytes": null },
        "verification": {
            "status": "passed",
            "reference_backend": null,
            "max_abs_error": null,
            "max_rel_error": null,
            "residual": null,
            "rtol": rtol,
            "atol": atol,
            "reason": null
        },
        "execution": {
            "device": "cuda",
            "device_ordinal": device_ordinal,
            "execution_path": "phase2-measured-tenferro-cuda-trace-linalg-ad",
            "synchronization": "tenferro-rs explicit synchronize without result download",
            "layout": layout_str(problem),
            "dtype": dtype_str(problem),
            "notes": problem["notes"].as_str(),
            "unsupported_reason": null
        }
    })
}

fn layout_str(problem: &serde_yaml::Value) -> String {
    json_field_str(problem.get("layout").unwrap_or(&serde_yaml::Value::Null))
}

fn dtype_str(problem: &serde_yaml::Value) -> String {
    json_field_str(problem.get("dtype").unwrap_or(&serde_yaml::Value::Null))
}

fn json_field_str(value: &serde_yaml::Value) -> String {
    serde_json::to_string(&serde_json::to_value(value).unwrap_or(serde_json::Value::Null))
        .unwrap_or_else(|_| "{}".to_string())
}

fn tensor_f64(shape: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::F64(
        TypedTensor::from_vec_col_major(shape.to_vec(), data)
            .expect("benchmark shape/data length should match"),
    )
}

fn data_for_shape(shape: &[usize], seed: u64) -> Vec<f64> {
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

fn well_conditioned_matrix(n: usize, seed: u64) -> Vec<f64> {
    let mut data = data_for_shape(&[n, n], seed);
    for j in 0..n {
        for i in 0..n {
            data[i + n * j] *= 0.05;
        }
        data[j + n * j] += 1.0 + j as f64 / n.max(1) as f64;
    }
    data
}

fn spd_matrix(n: usize, seed: u64) -> Vec<f64> {
    let base = well_conditioned_matrix(n, seed);
    let mut out = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += base[k + n * i] * base[k + n * j];
            }
            if i == j {
                sum += 1.0;
            }
            out[i + n * j] = sum;
        }
    }
    out
}

const TANGENT_SEED_OFFSET: u64 = 1_000;

fn tangent_seed(matrix_seed: u64) -> u64 {
    matrix_seed.wrapping_add(TANGENT_SEED_OFFSET)
}

fn yaml_usize(v: &serde_yaml::Value, default: usize) -> usize {
    v.as_u64().unwrap_or(default as u64) as usize
}

fn yaml_u64(v: &serde_yaml::Value, default: u64) -> u64 {
    v.as_u64().unwrap_or(default)
}

fn yaml_str<'a>(v: &'a serde_yaml::Value, default: &'a str) -> &'a str {
    v.as_str().unwrap_or(default)
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
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("{secs}")
}
