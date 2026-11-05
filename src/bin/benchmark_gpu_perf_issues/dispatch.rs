//! Public-operation regressions: prepared dispatch and batched GEMM routing.
use super::*;
use tenferro_gpu::cuda::cuda_runtime_engine_registration;
use tenferro_runtime::{DType, ExecutionEngine, GraphCompiler, Runtime, TracedTensor};

pub(super) fn decode(p: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let (din, dout, len) = (
        param_usize(p, "in")?,
        param_usize(p, "out")?,
        param_usize(p, "len")?,
    );
    let arm = param_str(p, "arm")?;
    let mut engine = backend(device)?;
    let rt = engine.runtime().clone();
    let xd = fill_f32(din * len, 1);
    let wd = fill_f32(din * dout, 2);
    let x = upload_tensor(&rt, &host_f32(vec![din, len], xd.clone())?)?;
    let w = upload_tensor(&rt, &host_f32(vec![din, dout], wd.clone())?)?;
    let cfg = matmul_cfg(&[0], &[0]);
    let check = |out: &Tensor| -> Result<f64> {
        let host = download_tensor(&rt, out)?;
        if host.shape() != [len, dout] {
            return Err("decode output shape".into());
        }
        let got = host.as_slice::<f32>()?;
        let mut worst = 0.0_f64;
        for j in 0..dout {
            for i in 0..len {
                let expected: f64 = (0..din)
                    .map(|k| xd[k + din * i] as f64 * wd[k + din * j] as f64)
                    .sum();
                worst = worst_of(
                    worst,
                    (got[i + len * j] as f64 - expected).abs() / expected.abs().max(1.0),
                );
            }
        }
        check_tol(worst, 1e-3, "decode scalar oracle")
    };
    let (error, measurement) = if arm == "prepared" {
        let tx = TracedTensor::input_concrete_shape(DType::F32, &[din, len])?;
        let tw = TracedTensor::input_concrete_shape(DType::F32, &[din, dout])?;
        let ty = tx.dot_general(&tw, cfg)?;
        let program = GraphCompiler::new().compile_with_input_specs(
            &ty,
            &[
                (&tx, DType::F32, &[din, len]),
                (&tw, DType::F32, &[din, dout]),
            ],
        )?;
        let mut builder = Runtime::builder();
        builder.register_engine(cuda_runtime_engine_registration(&engine)?)?;
        let runtime = builder.build()?;
        let reads = [TensorRead::from_tensor(&x), TensorRead::from_tensor(&w)];
        let prepared = runtime.prepare(&program, &reads, &[ExecutionEngine::target(&engine)])?;
        let output = runtime.run(&mut [&mut engine], &prepared, &reads)?;
        let tensor = output
            .into_iter()
            .next()
            .ok_or("missing decode output")?
            .into_tensor(&mut engine)?;
        let error = check(&tensor)?;
        rt.synchronize()?;
        let measurement = measure(timing, len * dout * 4, |it| {
            // Retain Runtime values; inspection/materialization is outside the clock.
            timed(
                it,
                &mut || Ok(runtime.run(&mut [&mut engine], &prepared, &reads)?),
                &mut || Ok(rt.synchronize()?),
            )
        })?;
        (error, measurement)
    } else if arm == "eager-session" {
        let runtime = EagerRuntime::new()?;
        let (ex, ew) = runtime.with_eager_session(&mut engine, |s| -> Result<_> {
            Ok((s.constant_from(x)?, s.constant_from(w)?))
        })?;
        let out =
            runtime.with_eager_session(&mut engine, |s| s.dot_general(&ex, &ew, cfg.clone()))?;
        let error = check(&out.to_tensor(&mut engine)?)?;
        rt.synchronize()?;
        runtime.with_eager_session(&mut engine, |s| -> Result<_> {
            let measurement = measure(timing, len * dout * 4, |it| {
                timed(
                    it,
                    &mut || Ok(s.dot_general(&ex, &ew, cfg.clone())?),
                    &mut || Ok(rt.synchronize()?),
                )
            })?;
            Ok((error, measurement))
        })?
    } else {
        return Err("unknown decode arm".into());
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer: vec![
            "one prepared Runtime::run or eager dot_general",
            "batch-edge device synchronization",
        ],
        outside_timer: vec![
            "fixture and upload",
            "engine and eager session creation",
            "trace compile and prepare",
            "validation and output inspection",
            "output destruction",
        ],
        extra: json!({"execution_path": arm, "output_shape": [len,dout]}),
    })
}

pub(super) fn batched(p: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let n = param_usize(p, "n")?;
    let batch = param_usize(p, "batch")?;
    let arm = param_str(p, "arm")?;
    let outer = match arm {
        "batch-trailing" => false,
        "batch-outer" => true,
        _ => return Err("unknown batch arm".into()),
    };
    let ad = values(n * n * batch, 31);
    let bd = values(n * n * batch, 32);
    let layout = |data: &[f64]| {
        if !outer {
            return data.to_vec();
        }
        let mut out = vec![0.0; data.len()];
        for b in 0..batch {
            for j in 0..n {
                for i in 0..n {
                    out[b + batch * (i + n * j)] = data[i + n * (j + n * b)];
                }
            }
        }
        out
    };
    let shape = if outer {
        vec![batch, n, n]
    } else {
        vec![n, n, batch]
    };
    let mut engine = backend(device)?;
    let rt = engine.runtime().clone();
    let a = upload_tensor(&rt, &host(shape.clone(), layout(&ad))?)?;
    let b = upload_tensor(&rt, &host(shape, layout(&bd))?)?;
    let cfg = DotGeneralConfig {
        lhs_contracting_dims: [if outer { 2 } else { 1 }].as_slice().into(),
        rhs_contracting_dims: [if outer { 1 } else { 0 }].as_slice().into(),
        lhs_batch_dims: [if outer { 0 } else { 2 }].as_slice().into(),
        rhs_batch_dims: [if outer { 0 } else { 2 }].as_slice().into(),
    };
    let ar = TensorRead::from_tensor(&a);
    let br = TensorRead::from_tensor(&b);
    let output = engine.dot_general_read(ar.clone(), br.clone(), &cfg)?;
    let host = download_tensor(&rt, &output)?;
    if host.shape() != [n, n, batch] {
        return Err("batched output shape".into());
    }
    let got = host.as_slice::<f64>()?;
    let mut worst = 0.0_f64;
    for b in 0..batch {
        for j in 0..n {
            for i in 0..n {
                let expected: f64 = (0..n)
                    .map(|k| ad[i + n * (k + n * b)] * bd[k + n * (j + n * b)])
                    .sum();
                worst = worst_of(
                    worst,
                    (got[i + n * (j + n * b)] - expected).abs() / expected.abs().max(1.0),
                );
            }
        }
    }
    let error = check_tol(worst, 1e-10, "batched scalar oracle")?;
    rt.synchronize()?;
    let measurement = measure(timing, n * n * batch * 8, |it| {
        timed(
            it,
            &mut || Ok(engine.dot_general_read(ar.clone(), br.clone(), &cfg)?),
            &mut || Ok(rt.synchronize()?),
        )
    })?;
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer: vec![
            "one allocating batched dot_general_read",
            "batch-edge device synchronization",
        ],
        outside_timer: vec![
            "input generation and layout conversion",
            "engine creation and upload",
            "full scalar oracle and download",
            "output destruction",
        ],
        extra: json!({"execution_path": arm, "expected_route": if outer {"cutensor"} else {"cublas-strided-batched"},
                     "route_evidence": "public operand layout; no private route counter", "output_shape": [n,n,batch]}),
    })
}
