//! #2436 / #2440 / #2445: factorization with intrinsic solver-info readback.
use super::*;
use tenferro_linalg::EagerSessionLinalgExt;

pub(super) fn run(p: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let n = param_usize(p, "n")?;
    let dtype = param_str(p, "dtype")?;
    let arm = param_str(p, "arm")?;
    // Symmetric strictly diagonally dominant positive definite matrix.
    let mut data = vec![0_f64; n * n];
    for j in 0..n {
        for i in 0..n {
            data[i + n * j] = if i == j {
                n as f64
            } else {
                ((i.min(j) * 7 + i.max(j) * 13) % 31) as f64 / 128. - 0.1
            };
        }
    }
    let input = match dtype {
        "f64" => Tensor::from_vec_col_major(vec![n, n], data.clone())?,
        "f32" => Tensor::from_vec_col_major(
            vec![n, n],
            data.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        )?,
        _ => return Err(format!("unsupported cholesky dtype {dtype}").into()),
    };
    // Validation uses precisely the rounded logical inputs of each dtype.
    if dtype == "f32" {
        for x in &mut data {
            *x = *x as f32 as f64;
        }
    }
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    let a = upload_tensor(&rt, &input)?;
    rt.synchronize()?;
    let check = |out: &Tensor| -> Result<f64> {
        let host = download_tensor(&rt, out)?;
        if host.shape() != [n, n] {
            return Err("wrong Cholesky output shape".into());
        }
        let l: Vec<f64> = if dtype == "f64" {
            host.as_slice::<f64>()?.to_vec()
        } else {
            host.as_slice::<f32>()?.iter().map(|&v| v as f64).collect()
        };
        let mut worst = 0_f64;
        for j in 0..n {
            for i in 0..n {
                let reconstructed: f64 = (0..=i.min(j)).map(|q| l[i + n * q] * l[j + n * q]).sum();
                worst = worst_of(
                    worst,
                    (reconstructed - data[i + n * j]).abs() / data[i + n * j].abs().max(1.),
                );
            }
        }
        check_tol(
            worst,
            if dtype == "f64" { 1e-10 } else { 1e-4 },
            "Cholesky reconstruction",
        )
    };
    let bytes = n * n * if dtype == "f64" { 8 } else { 4 };
    let (error, measurement) = if arm == "direct" {
        let first = a.cholesky(&mut bk)?;
        let error = check(&first)?;
        let measurement = measure(timing, bytes, |it| {
            timed_in_session(it, &mut bk, |s| Ok(a.cholesky(s)?))
        })?;
        (error, measurement)
    } else if arm == "eager" {
        let runtime = EagerRuntime::new()?;
        let eager = runtime.with_eager_session(&mut bk, |s| s.constant_from(a))?;
        let first = runtime.with_eager_session(&mut bk, |s| s.cholesky(&eager))?;
        let error = check(&first.to_tensor(&mut bk)?)?;
        let measurement = runtime.with_eager_session(&mut bk, |s| -> Result<Measurement> {
            measure(timing, bytes, |it| {
                let mut outputs = Vec::with_capacity(it);
                let start = Instant::now();
                for _ in 0..it {
                    outputs.push(s.cholesky(&eager)?);
                }
                rt.synchronize()?;
                let elapsed = start.elapsed().as_nanos();
                black_box(&outputs);
                Ok(elapsed)
            })
        })?;
        (error, measurement)
    } else {
        return Err(format!("unknown Cholesky arm {arm}").into());
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer: vec![
            "one public Cholesky call including intrinsic solver-info readback",
            "output allocation",
            "batch-edge device synchronization",
        ],
        outside_timer: vec![
            "HPD fixture and upload",
            "engine, eager runtime and session setup",
            "EagerTensor wrapping",
            "explicit first call and full reconstruction",
            "output download and destruction",
        ],
        extra: json!({"execution_path":arm,"dtype":dtype,"n":n}),
    })
}
