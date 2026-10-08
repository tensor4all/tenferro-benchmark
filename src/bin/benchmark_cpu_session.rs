//! Many independent small matrices in one entered CPU session, and the
//! one-operation batched routes of `cpu/session_matrix` (`--case <id>`).
#[path = "../bench_support/batch_route.rs"]
mod batch_route;
#[path = "../bench_support/shape_stream.rs"]
mod shape_stream;

use std::{error::Error, hint::black_box, time::Instant};
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::BackendSessionHost;
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorRead};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn fixture(n: usize, k: usize) -> Result<(Tensor, Tensor, Vec<f64>, Vec<f64>)> {
    let a: Vec<_> = (0..n * n)
        .map(|i| {
            let (r, c) = (i % n, i / n);
            if r == c {
                n as f64 + 1.0 + (k % 17) as f64 * 0.01
            } else {
                ((r + c + k) % 7) as f64 * 0.01
            }
        })
        .collect();
    let b: Vec<_> = (0..n * n)
        .map(|i| ((i + k) % 13) as f64 * 0.02 - 0.1)
        .collect();
    let product = (0..n * n)
        .map(|i| (0..n).map(|j| a[i % n + n * j] * b[j + n * (i / n)]).sum())
        .collect();
    Ok((
        Tensor::from_vec_col_major(vec![n, n], a)?,
        Tensor::from_vec_col_major(vec![n, n], b.clone())?,
        product,
        b,
    ))
}
/// Time one batched-route case: a single public call per operation, many
/// operations per wall-clock interval, one entered session around warmup,
/// calibration and all samples.
fn run_batched_case(args: &[String]) -> Result<()> {
    use batch_route::*;
    let get = |name: &str, default: &str| {
        args.windows(2)
            .find(|p| p[0] == name)
            .map(|p| p[1].clone())
            .unwrap_or(default.into())
    };
    let case_id = get("--case", "");
    let threads: usize = get("--threads", "1").parse()?;
    let warmups: usize = get("--warmups", "3").parse()?;
    let samples: usize = get("--samples", "15").parse()?;
    let target_ns: u128 = get("--target-ns", "1000000").parse()?;
    let instances = get("--instances", "data/instances/session_matrix.json");
    tenferro_einsum_benchmark::thread_enforcement::enforce_thread_request(threads)?;
    if let Some(stream) = shape_stream::load_stream(&instances, &case_id)? {
        return run_stream_case(&stream, threads, warmups, samples);
    }
    let spec = load_cases(&instances)?
        .into_iter()
        .find(|c| c.id == case_id)
        .ok_or_else(|| format!("unknown batched case {case_id}"))?;
    let mut row = serde_json::json!({
        "case_id": spec.id, "workload": spec.workload, "operation": spec.operation,
        "dtype": spec.dtype, "batch": spec.batch, "m": spec.m, "n": spec.n, "k": spec.k,
        "layout": spec.layout, "output": spec.output, "provider": "faer",
        "route": spec.route, "pair": spec.pair, "requested_threads": threads,
        "lane_cost_policy_api": LANE_COST_POLICY_API,
        "timing_scope": "many_operations_single_interval",
        "session_boundary": "one entered backend session around warmup, calibration and samples",
        "outside_timer": ["input generation", "view construction", "backend construction",
            "policy override", "session entry", "destination allocation (_into)",
            "first-call validation", "calibration", "output checks", "output destruction"],
        "inside_timer": ["public call", "intrinsic output allocation (allocating routes)"],
    });
    let emit_unsupported = |mut row: serde_json::Value, reason: String| {
        row["correctness"] = "unsupported".into();
        row["unsupported_reason"] = reason.into();
        row["samples_ns"] = serde_json::json!([]);
        println!("{row}");
    };
    let scope = match build_policy(&spec.policy) {
        Ok(scope) => scope,
        Err(reason) => {
            emit_unsupported(row, reason);
            return Ok(());
        }
    };
    let mut backend = backend_for(threads, scope, None)?;
    let info = backend.execution_info();
    tenferro_einsum_benchmark::thread_enforcement::verify_backend_threads(
        "CpuBackend",
        info.worker_count(),
        threads,
    )?;
    row["worker_count"] = info.worker_count().into();
    row["execution_mode"] = format!("{:?}", info.execution_mode()).into();
    row["effective_policy"] = match scope {
        PolicyScope::Scoped(p) => format!("scoped:{p:?}"),
        _ => format!("backend:{:?}", backend.batch_policy()),
    }
    .into();
    let fixture = Fixture::new(&spec)?;
    let reads = fixture.reads()?;
    let owned = fixture.owned();
    let mut destination = fixture.destination()?;
    let max_retained = (64usize << 20) / fixture.output_bytes().max(1);
    enum Outcome {
        Timed(Vec<u128>, usize, f64),
        Unsupported(String),
    }
    let outcome = backend.with_backend_session(|session| -> Result<Outcome> {
        in_policy_scope(session, scope, |session| -> Result<Outcome> {
            let into = spec.is_into();
            let call = |session: &mut dyn tenferro_tensor::BackendSession,
                        outputs: &mut Vec<tenferro_tensor::Tensor>,
                        destination: &mut tenferro_tensor::Tensor|
             -> std::result::Result<(), OpError> {
                if into {
                    run_into(session, &spec, &fixture, &reads, &owned, destination)
                } else {
                    outputs.push(run_alloc(session, &spec, &fixture, &reads, &owned)?);
                    Ok(())
                }
            };
            let check = |outputs: &[tenferro_tensor::Tensor],
                         destination: &tenferro_tensor::Tensor|
             -> Result<f64> {
                let mut worst = 0.0_f64;
                if into {
                    worst = fixture.check(destination)?;
                }
                for output in outputs {
                    worst = worst.max(fixture.check(output)?);
                }
                Ok(worst)
            };
            // Explicit lazy initialization and validation before any sample.
            let mut outputs = Vec::new();
            match call(session, &mut outputs, &mut destination) {
                Ok(()) => {}
                Err(OpError::Unsupported(reason)) => return Ok(Outcome::Unsupported(reason)),
                Err(OpError::Failed(reason)) => return Err(reason.into()),
            }
            check(&outputs, &destination)?;
            outputs.clear();
            // Calibrate operations per interval toward target_ns (outside stats).
            let start = Instant::now();
            call(session, &mut outputs, &mut destination).map_err(|e| e.to_string())?;
            let one = start.elapsed().as_nanos().max(1);
            outputs.clear();
            let reps = ((target_ns / one) as usize).clamp(1, max_retained.max(1));
            outputs.reserve(reps);
            let mut elapsed = Vec::with_capacity(samples);
            for sample in 0..warmups + samples {
                outputs.clear(); // Destruction is outside the clock.
                let start = Instant::now();
                for _ in 0..reps {
                    call(session, &mut outputs, &mut destination).map_err(|e| e.to_string())?;
                }
                let ns = start.elapsed().as_nanos();
                black_box(&outputs);
                if sample >= warmups {
                    elapsed.push(ns);
                }
            }
            let worst = check(&outputs, &destination)?;
            Ok(Outcome::Timed(elapsed, reps, worst))
        })?
    })??;
    match outcome {
        Outcome::Unsupported(reason) => emit_unsupported(row, reason),
        Outcome::Timed(elapsed, reps, worst) => {
            row["correctness"] = "passed".into();
            row["max_abs_error"] = worst.into();
            row["operations_per_sample"] = reps.into();
            row["warmups"] = warmups.into();
            row["samples_ns"] = serde_json::json!(elapsed);
            println!("{row}");
        }
    }
    Ok(())
}

/// Time a contraction-key stream: one sample is one whole call sequence.
fn run_stream_case(
    spec: &shape_stream::StreamSpec,
    threads: usize,
    warmups: usize,
    samples: usize,
) -> Result<()> {
    let mut backend = tenferro_einsum_benchmark::cpu_provider::configure(
        CpuBackend::with_threads_and_kind(threads, CpuBackendKind::Faer)?,
    )?;
    let info = backend.execution_info();
    tenferro_einsum_benchmark::thread_enforcement::verify_backend_threads(
        "CpuBackend",
        info.worker_count(),
        threads,
    )?;
    // Inputs for every call of every pass exist before the session is entered.
    let stream = shape_stream::Stream::new(spec, 1 + warmups + samples)?;
    let (elapsed, worst) = backend.with_backend_session(|session| {
        stream.measure(session, warmups).map_err(|e| e.to_string())
    })??;
    println!(
        "{}",
        serde_json::json!({
            "case_id": spec.id, "workload": "stream", "stream": spec.stream,
            "sequence_order": match spec.stream.as_str() {
                "mixed" | "strides" => format!("round-robin over {} keys", spec.keys),
                "fresh" => "every call a key unseen in this process".into(),
                _ => "one key".to_string(),
            },
            "operation": "einsum", "dtype": "f64", "provider": "faer", "route": spec.route,
            "requested_threads": threads, "worker_count": info.worker_count(),
            "execution_mode": format!("{:?}", info.execution_mode()),
            "operations_per_sample": spec.length, "warmups": warmups, "samples_ns": elapsed,
            "correctness": "passed", "max_abs_error": worst,
            "timing_scope": "whole_call_sequence_single_interval",
            "planning_in_timer": true,
            "cache_mode": "default (no bypass); pass 0 is an untimed initialization pass",
            "plan_counters": {"builds": "unavailable", "hits": "unavailable",
                              "misses": "unavailable", "evictions": "unavailable",
                              "cache_bytes": "unavailable"},
            "outside_timer": ["input generation for every call", "view construction",
                              "backend construction", "session entry", "output checks"],
        })
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.iter().any(|a| a == "--case") {
        return run_batched_case(&args);
    }
    let get = |name: &str, default: &str| {
        args.windows(2)
            .find(|p| p[0] == name)
            .map(|p| p[1].clone())
            .unwrap_or(default.into())
    };
    let n: usize = get("--n", "2").parse()?;
    let count: usize = get("--count", "1024").parse()?;
    let samples: usize = get("--samples", "15").parse()?;
    let warmups: usize = get("--warmups", "3").parse()?;
    let op = get("--op", "matmul");
    let provider = get("--provider", "blas");
    if n == 0 || count == 0 || samples == 0 || !["matmul", "solve"].contains(&op.as_str()) {
        return Err("invalid case".into());
    }
    let blas_built = cfg!(any(
        feature = "system-openblas",
        feature = "system-accelerate",
        feature = "system-mkl"
    ));
    if provider == "blas" && !blas_built {
        // A provider this build does not contain is unsupported, not failed.
        println!(
            "{}",
            serde_json::json!({"operation":op,"n":n,"provider":provider,"correctness":"unsupported",
                "unsupported_reason":"harness built without a system BLAS feature (cpu-blas)",
                "samples_ns":[]})
        );
        return Ok(());
    }
    let kind = match provider.as_str() {
        "blas" => CpuBackendKind::Blas,
        "faer" => CpuBackendKind::Faer,
        _ => return Err("invalid provider".into()),
    };
    let mut backend =
        tenferro_einsum_benchmark::cpu_provider::configure(CpuBackend::with_kind(kind)?)?;
    let execution_info = backend.execution_info();
    let execution_mode = format!("{:?}", execution_info.execution_mode());
    let worker_count = execution_info.worker_count();
    let fixtures = (0..count)
        .map(|k| fixture(n, k))
        .collect::<Result<Vec<_>>>()?;
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1].into(),
        rhs_contracting_dims: vec![0].into(),
        lhs_batch_dims: vec![].into(),
        rhs_batch_dims: vec![].into(),
    };
    // Output slots and every input are prepared before entering the session.
    let mut outputs = Vec::with_capacity(count);
    let mut elapsed_ns = Vec::with_capacity(samples);
    backend.with_backend_session(|session| -> Result<()> {
        for sample in 0..samples + warmups {
            outputs.clear(); // Destruction/reset is outside the clock.
            let start = Instant::now();
            for (a, b, _, _) in &fixtures {
                outputs.push(if op == "matmul" {
                    session.dot_general_read(
                        TensorRead::from_tensor(a),
                        TensorRead::from_tensor(b),
                        &config,
                    )?
                } else {
                    a.solve(b, session)?
                });
            }
            let elapsed = start.elapsed().as_nanos();
            black_box(&outputs);
            if sample >= warmups {
                elapsed_ns.push(elapsed);
            }
            // Check all outputs, including every warmup, after timer stop.
            for (output, (a, _, product, b)) in outputs.iter().zip(&fixtures) {
                let out = output.as_slice::<f64>()?;
                let av = a.as_slice::<f64>()?;
                for i in 0..n * n {
                    let (actual, expected) = if op == "matmul" {
                        (out[i], product[i])
                    } else {
                        (
                            (0..n)
                                .map(|j| av[i % n + n * j] * out[j + n * (i / n)])
                                .sum(),
                            b[i],
                        )
                    };
                    if !actual.is_finite()
                        || (actual - expected).abs() > 1e-11 * (1.0 + expected.abs())
                    {
                        return Err(
                            format!("incorrect {op} output {i}: {actual} != {expected}").into()
                        );
                    }
                }
            }
        }
        Ok(())
    })??;
    println!(
        "{}",
        serde_json::json!({"operation":op,"n":n,"operations_per_sample":count,"provider":provider,"route":"shared-session","session_count":1,"execution_mode":execution_mode,"worker_count":worker_count,"warmups":warmups,"samples_ns":elapsed_ns,"correctness":"passed"})
    );
    Ok(())
}
