//! #98: equal-output non-AD extension calls, with and without session reuse.
use super::*;
use tenferro_ad::EagerSession;
use tenferro_einsum::{
    EagerSessionEinsumExt, EinsumSubscripts, TensorEinsumExt, TensorReadEinsumExt,
};
use tenferro_linalg::{EagerSessionLinalgExt, TensorLinalgExt, TensorReadLinalgExt};
use tenferro_tensor::TensorValue;

fn fixture(n: usize, seed: usize) -> Vec<f64> {
    (0..n * n)
        .map(|p| {
            let (i, j) = (p % n, p / n);
            if i == j {
                3. + i as f64 / n as f64 + seed as f64 / 4.
            } else {
                ((i * 13 + j * 7 + seed * 11) % 31) as f64 / 128. - 0.1
            }
        })
        .collect()
}
fn transpose(n: usize, x: &[f64]) -> Vec<f64> {
    (0..n * n).map(|p| x[p / n + n * (p % n)]).collect()
}
fn orthogonality(n: usize, q: &[f64]) -> f64 {
    (0..n * n)
        .map(|p| {
            let (i, j) = (p % n, p / n);
            ((0..n).map(|k| q[k + n * i] * q[k + n * j]).sum::<f64>()
                - if i == j { 1. } else { 0. })
            .abs()
        })
        .fold(0., f64::max)
}
fn check(op: &str, n: usize, a: &[f64], expected: &[f64], out: &[Tensor]) -> Result<f64, BoxError> {
    let values: Vec<&[f64]> = out
        .iter()
        .map(|t| t.as_slice::<f64>())
        .collect::<Result<_, _>>()?;
    let mut ortho: f64 = 0.;
    let reconstructed = match op {
        "qr" => {
            ortho = orthogonality(n, values[0]);
            matmul_reference(n, values[0], values[1])
        }
        "svd" => {
            ortho = orthogonality(n, values[0]).max(orthogonality(n, &transpose(n, values[2])));
            let us: Vec<f64> = (0..n * n)
                .map(|p| values[0][p] * values[1][p / n])
                .collect();
            matmul_reference(n, &us, values[2])
        }
        "solve" => matmul_reference(n, a, values[0]),
        _ => values[0].to_vec(),
    };
    let reference = if matches!(op, "qr" | "svd") {
        a
    } else {
        expected
    };
    let residual = reconstructed
        .iter()
        .zip(reference)
        .map(|(x, y)| (x - y).abs() / (1. + y.abs()))
        .fold(0., f64::max);
    if residual > 1e-10 || ortho > 1e-10 {
        return Err(format!("{op}: residual={residual}, orthogonality={ortho}").into());
    }
    Ok(residual.max(ortho))
}

fn direct(
    op: &str,
    input: &[Tensor],
    views: Option<&[TensorValue]>,
    subs: &EinsumSubscripts,
    session: &mut dyn BackendSession,
) -> Result<Vec<Tensor>, BoxError> {
    if let Some(views) = views {
        let reads: Vec<_> = views.iter().map(TensorValue::tensor_read).collect();
        return Ok(match op {
            "einsum" | "einsum3" => vec![reads.as_slice().einsum_read_subscripts(subs, session)?],
            "qr" => {
                let (q, r) = reads[0].clone().qr_read(session)?;
                vec![q, r]
            }
            "svd" => {
                let (u, s, v) = reads[0].clone().svd_read(session)?;
                vec![u, s, v]
            }
            "solve" => vec![reads[0].clone().solve_read(reads[1].clone(), session)?],
            _ => return Err("unknown operation".into()),
        });
    }
    Ok(match op {
        "einsum" | "einsum3" => {
            let refs: Vec<_> = input.iter().collect();
            vec![refs.as_slice().einsum_subscripts(subs, session)?]
        }
        "qr" => {
            let (q, r) = input[0].qr(session)?;
            vec![q, r]
        }
        "svd" => {
            let (u, s, v) = input[0].svd(session)?;
            vec![u, s, v]
        }
        "solve" => vec![input[0].solve(&input[1], session)?],
        _ => return Err("unknown operation".into()),
    })
}
fn eager(
    op: &str,
    input: &[EagerTensor],
    subs: &EinsumSubscripts,
    session: &mut EagerSession<'_>,
) -> Result<Vec<EagerTensor>, BoxError> {
    Ok(match op {
        "einsum" | "einsum3" => {
            let refs: Vec<_> = input.iter().collect();
            vec![session.einsum_subscripts(&refs, subs)?]
        }
        "qr" => {
            let (q, r) = session.qr(&input[0])?;
            vec![q, r]
        }
        "svd" => {
            let (u, s, v) = session.svd(&input[0])?;
            vec![u, s, v]
        }
        "solve" => vec![session.solve(&input[0], &input[1])?],
        _ => return Err("unknown operation".into()),
    })
}

fn sample<O>(
    mode: &str,
    n: usize,
    warmups: usize,
    runs: usize,
    target: u128,
    mut operation: impl FnMut() -> Result<O, BoxError>,
) -> Result<serde_json::Value, BoxError> {
    #[cfg(feature = "allocation-diagnostics")]
    if mode == "measure" {
        return Err("instrumented build cannot collect latency".into());
    }
    // Explicit lazy initialization even with zero requested warmups.
    let first = Instant::now();
    let initialized = operation()?;
    let first_ns = first.elapsed().as_nanos();
    drop(initialized);
    for _ in 0..warmups {
        operation()?;
    }
    if mode == "correctness-only" {
        return Ok(serde_json::json!({"samples":[]}));
    }
    if mode == "allocations" {
        #[cfg(feature = "allocation-diagnostics")]
        {
            let mut counts = Vec::with_capacity(7);
            for _ in 0..7 {
                let (output, (allocations, bytes)) = super::allocations::capture(&mut operation);
                let output = output?;
                black_box(&output);
                drop(output);
                counts.push(serde_json::json!({"allocations":allocations,"bytes":bytes}));
            }
            return Ok(serde_json::json!({"samples":[],"allocation_samples":counts,
                "allocation_scope":"calling-thread Rust GlobalAlloc alloc/alloc_zeroed/realloc requests; excludes frees, foreign BLAS and worker-thread allocations; includes intrinsic result packaging"}));
        }
        #[cfg(not(feature = "allocation-diagnostics"))]
        return Err("allocations mode requires separate allocation-diagnostics build".into());
    }
    // Memory cap includes three retained matrix outputs plus wrapper headroom.
    let max_iterations = (64 * 1024 * 1024 / (3 * n * n * 8 + 4096))
        .min(1 << 16)
        .max(1);
    let mut batch = |count| -> Result<u128, BoxError> {
        let mut outputs = Vec::with_capacity(count);
        let clock = Instant::now();
        for _ in 0..count {
            outputs.push(operation()?);
        }
        let elapsed = clock.elapsed().as_nanos();
        black_box(&outputs);
        Ok(elapsed)
    };
    let mut count = 16usize.min(max_iterations);
    loop {
        let elapsed = batch(count)?;
        if elapsed >= target || count == max_iterations {
            break;
        }
        count = (count * 2).min(max_iterations);
    }
    let mut samples = Vec::with_capacity(runs);
    for index in 0..runs {
        samples.push(Sample {
            process_index: 0,
            sample_index: index,
            elapsed_ns: batch(count)?,
            iterations: count,
        });
    }
    Ok(
        serde_json::json!({"samples":samples,"initialization_after_validation_ns":first_ns,"initialization_scope":"untimed repeat initialization diagnostic, session policy of this route; never used for wrapper comparisons",
        "batch_memory_limit_bytes":64*1024*1024,"target_ns":target,"iterations":count}),
    )
}

pub fn run(case_id: &str) -> Result<(), BoxError> {
    let op = required_arg("--operation")?;
    let tier = required_arg("--api-tier")?;
    let mode = arg("--mode", "measure");
    let layout = arg("--layout", "col_major_contiguous");
    let size = parse_usize("--size", 4)?;
    let n = (size as f64).sqrt() as usize;
    if !matches!(n, 2 | 4 | 8 | 16 | 64)
        || n * n != size
        || !matches!(op.as_str(), "einsum" | "einsum3" | "qr" | "svd" | "solve")
        || !matches!(
            tier.as_str(),
            "concrete-fresh" | "concrete-shared" | "eager-no-ad" | "eager-shared"
        )
        || !matches!(
            mode.as_str(),
            "measure" | "correctness-only" | "allocations"
        )
        || !matches!(
            layout.as_str(),
            "col_major_contiguous" | "row_major_contiguous"
        )
    {
        return Err("invalid nonad selector".into());
    }
    let warmups = parse_usize("--warmups", 3)?;
    let runs = parse_usize("--samples", 15)?;
    let target = parse_usize("--target-ns", 1_000_000)? as u128;
    let a = fixture(n, 1);
    let b = fixture(n, 2);
    let c = fixture(n, 3);
    let expected = match op.as_str() {
        "einsum" => matmul_reference(n, &a, &b),
        "einsum3" => matmul_reference(n, &matmul_reference(n, &a, &b), &c),
        "solve" => b.clone(),
        _ => a.clone(),
    };
    let logical = match op.as_str() {
        "einsum3" => vec![a.clone(), b, c],
        "einsum" | "solve" => vec![a.clone(), b],
        _ => vec![a.clone()],
    };
    let input: Vec<Tensor> = logical
        .iter()
        .map(|v| {
            Tensor::from_vec_col_major(
                [n, n],
                if layout == "row_major_contiguous" {
                    transpose(n, v)
                } else {
                    v.clone()
                },
            )
        })
        .collect::<Result<_, _>>()?;
    let views = if layout == "row_major_contiguous" {
        Some(
            input
                .iter()
                .map(|t| {
                    TensorValue::from_tensor(Tensor::from_vec_col_major(
                        [n, n],
                        t.as_slice::<f64>()?.to_vec(),
                    )?)
                    .transpose_view([1, 0])
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };
    let subs = if op == "einsum3" {
        EinsumSubscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3])
    } else {
        EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2])
    };
    let mut backend = CpuBackend::new();
    let residual;
    let mut initial_public_call_ns = 0;
    let measured = if tier.starts_with("eager") {
        let runtime = EagerRuntime::with_cpu_backend(backend)?;
        let _no_grad = runtime.no_grad();
        // Fresh standalone process, no capture_trace guard is ever entered.
        let mut eager_input: Vec<_> = input
            .into_iter()
            .map(|t| EagerTensor::from_tensor_in(t, runtime.clone()))
            .collect::<Result<_, _>>()?;
        if views.is_some() {
            eager_input = runtime.with_eager_session(|s| {
                eager_input
                    .iter()
                    .map(|t| s.transpose(t, &[1, 0]))
                    .collect::<tenferro_ad::Result<Vec<_>>>()
            })?;
        }
        if eager_input.iter().any(EagerTensor::tracks_grad) {
            return Err("tracked eager input".into());
        }
        let validated = runtime.with_eager_session(|s| {
            let start = Instant::now();
            let result = eager(&op, &eager_input, &subs, s)?;
            initial_public_call_ns = start.elapsed().as_nanos();
            Ok::<_, BoxError>(result)
        })?;
        if validated.iter().any(EagerTensor::tracks_grad) {
            return Err("tracked eager output".into());
        }
        let tensors = validated
            .iter()
            .map(EagerTensor::to_tensor)
            .collect::<Result<Vec<_>, _>>()?;
        residual = check(&op, n, &a, &expected, &tensors)?;
        if tier == "eager-shared" {
            runtime.with_eager_session(|s| {
                sample(&mode, n, warmups, runs, target, || {
                    eager(&op, &eager_input, &subs, s)
                })
            })?
        } else {
            sample(&mode, n, warmups, runs, target, || {
                runtime.with_eager_session(|s| eager(&op, &eager_input, &subs, s))
            })?
        }
    } else {
        let validated = backend.session_scope(|s| {
            let start = Instant::now();
            let result = direct(&op, &input, views.as_deref(), &subs, s)?;
            initial_public_call_ns = start.elapsed().as_nanos();
            Ok(result)
        })?;
        residual = check(&op, n, &a, &expected, &validated)?;
        if tier == "concrete-shared" {
            backend.session_scope(|s| {
                sample(&mode, n, warmups, runs, target, || {
                    direct(&op, &input, views.as_deref(), &subs, s)
                })
            })?
        } else {
            sample(&mode, n, warmups, runs, target, || {
                backend.session_scope(|s| direct(&op, &input, views.as_deref(), &subs, s))
            })?
        }
    };
    let mut output = serde_json::json!({"suite_id":"cpu/small_work","case_id":case_id,"operation":op,"api_tier":tier,"dtype":"f64","shape":[n,n],"layout":layout,"workflow":"single","calls_per_workflow":1,"provider":tenferro_einsum_benchmark::compiled_cpu_provider(),"correctness_status":"passed","max_numerical_residual":residual,"capture_policy":"standalone fresh process; no capture_trace guard; no_grad; inputs and validated outputs tracks_grad=false","result_consumption":"borrowed/native owning results retained; no extra output materialization inside clock","scope":{"timer":["public extension call","intrinsic output allocation and result packaging",if tier.ends_with("shared") {"session entry excluded"} else {"per-call session entry/exit diagnostic"},if op.starts_with("einsum") {"ordinary einsum default Auto planning/cache lookup included equally; parsed integer notation"}else{"default linalg options"}],"outside_timer":["fixtures and view creation","backend/runtime construction","shared session entry/exit","initialization and warmups","validation/materialization","output destruction"]}});
    for (k, v) in measured.as_object().ok_or("invalid measurement")? {
        output[k] = v.clone();
    }
    output["initial_public_call_ns"] = serde_json::json!(initial_public_call_ns);
    output["initial_public_call_scope"] = serde_json::json!("first selected operation inside entered shared session; lazy operation initialization included; validation/destruction excluded; single-call diagnostic, no performance ratio");
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
