//! #2325 / #2416: allocating products, with identical-input controls.
use super::*;
use tenferro_einsum::{ConcreteEinsumPlan, EinsumSubscripts, TensorEinsumExt};

pub(super) fn run(p: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    let (m, k, n) = (
        param_usize(p, "m")?,
        param_usize(p, "k")?,
        param_usize(p, "n")?,
    );
    let complex = param_str(p, "dtype")? == "c64";
    let arm = param_str(p, "arm")?;
    let fmt = p["format"].as_str().unwrap_or("ij,jk->ik");
    let rank3 = fmt == "abc,cef->abef";
    let av = if complex {
        complex_values(m * k, 91)
    } else {
        values(m * k, 91)
            .into_iter()
            .map(|x| Complex64::new(x, 0.))
            .collect()
    };
    let bv = if complex {
        complex_values(k * n, 92)
    } else {
        values(k * n, 92)
            .into_iter()
            .map(|x| Complex64::new(x, 0.))
            .collect()
    };
    let mut expected = vec![Complex64::new(0., 0.); m * n];
    for j in 0..n {
        for i in 0..m {
            for q in 0..k {
                expected[i + m * j] += av[i + m * q] * bv[q + k * j];
            }
        }
    }
    let check = |actual: &[Complex64]| -> Result<f64> {
        if actual.len() != expected.len() {
            return Err("wrong product length".into());
        }
        let worst = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| rel_error_c64(*a, *b))
            .fold(0_f64, f64::max);
        check_tol(worst, 1e-10, "binary product scalar reference")
    };
    let elem = if complex { 16 } else { 8 };
    let (error, measurement) = if arm == "faer-direct" {
        use faer::{Accum, MatMut, MatRef, Par};
        let a = unsafe { MatRef::<Complex64>::from_raw_parts(av.as_ptr(), m, k, 1, m as isize) };
        let b = unsafe { MatRef::<Complex64>::from_raw_parts(bv.as_ptr(), k, n, 1, k as isize) };
        let op = || -> Result<Vec<Complex64>> {
            let mut output = vec![Complex64::new(0., 0.); m * n];
            let mut c =
                unsafe { MatMut::from_raw_parts_mut(output.as_mut_ptr(), m, n, 1, m as isize) };
            faer::linalg::matmul::matmul(
                &mut c,
                Accum::Replace,
                &a,
                &b,
                Complex64::new(1., 0.),
                Par::Seq,
            );
            Ok(output)
        };
        (check(&op()?)?, measure(timing, m * n * elem, op)?)
    } else {
        let ashape = if rank3 { vec![2, 4, 2] } else { vec![m, k] };
        let bshape = if rank3 { vec![2, 4, 4] } else { vec![k, n] };
        let a = if complex {
            Tensor::from_vec_col_major(ashape, av.clone())?
        } else {
            Tensor::from_vec_col_major(ashape, av.iter().map(|x| x.re).collect::<Vec<_>>())?
        };
        let b = if complex {
            Tensor::from_vec_col_major(bshape, bv.clone())?
        } else {
            Tensor::from_vec_col_major(bshape, bv.iter().map(|x| x.re).collect::<Vec<_>>())?
        };
        let subs = if rank3 {
            EinsumSubscripts::new(&[&[0, 1, 2], &[2, 3, 4]], &[0, 1, 3, 4])
        } else {
            EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2])
        };
        let plan = ConcreteEinsumPlan::prepare([&a, &b], fmt)?;
        let cfg = matmul_config(if rank3 { &[2] } else { &[1] }, &[0]);
        let mut bk = tenferro_einsum_benchmark::cpu_engine(Some(threads))?;
        bk.with_backend_session(|s| -> Result<(f64, Measurement)> {
            let mut op = || -> Result<Tensor> {
                Ok(match arm {
                    "string" => [&a, &b].einsum(fmt, s)?,
                    "subscripts" => [&a, &b].einsum_subscripts(&subs, s)?,
                    "prepared" => plan.execute([&a, &b], s)?,
                    "dot-general" => s.dot_general_read(
                        TensorRead::from_tensor(&a),
                        TensorRead::from_tensor(&b),
                        &cfg,
                    )?,
                    _ => return Err(format!("unknown product arm {arm}").into()),
                })
            };
            let output = op()?;
            let actual = if complex {
                output.as_slice::<Complex64>()?.to_vec()
            } else {
                output
                    .as_slice::<f64>()?
                    .iter()
                    .map(|&x| Complex64::new(x, 0.))
                    .collect()
            };
            let error = check(&actual)?;
            let shape = if rank3 { vec![2, 4, 4, 4] } else { vec![m, n] };
            if output.shape() != shape {
                return Err("wrong product shape".into());
            }
            Ok((error, measure(timing, m * n * elem, &mut op)?))
        })??
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer: vec![
            "declared allocating product API",
            "intrinsic output allocation",
        ],
        outside_timer: vec![
            "inputs and scalar reference",
            "engine and session",
            "subscripts and ConcreteEinsumPlan preparation",
            "explicit initialization and numerical validation",
            "output destruction",
        ],
        extra: json!({"execution_path":arm,"requested_threads":threads,"m":m,"k":k,"n":n}),
    })
}
