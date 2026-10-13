//! Isolated public linalg operations. Fixtures, sessions and validation are untimed.
use super::*;
use tenferro_linalg::{QrOptions, RankRevealingQrOptions, TensorReadLinalgExt};

pub(super) fn cases() -> Vec<Case> {
    let mut rows = Vec::new();
    for shape in ["256x128", "768x384"] {
        for name in [
            "householder_qr",
            "rank_revealing_qr",
            "rank_revealing_qr_deficient",
        ] {
            rows.push(Case {
                suite: "cpu/linalg_uncovered", benchmark: name, dtype: "f64", shape,
                notes: "setup=explicit; session=shared; output_retained_until_clock_stops; validation=untimed; QR gauge=Raw; RRQR rtol=1e-12 atol=0; RRQR outputs=Q/R/permutation/rank",
                run: unused,
            });
        }
    }
    for shape in ["256x256,rhs=16", "768x768,rhs=16"] {
        rows.push(Case {
            suite: "cpu/output_reuse", benchmark: "solve_read_into", dtype: "f64", shape,
            notes: "setup=explicit; session=shared; destination=preallocated; solve_includes_factorization; validation=untimed",
            run: unused,
        });
    }
    rows
}

fn unused(_: &mut CpuBackend) -> tenferro_tensor::Result<()> {
    Err(tenferro_tensor::Error::unsupported(
        "candidate",
        "use explicit setup sampler",
    ))
}

pub(super) fn handles(case: &Case) -> bool {
    matches!(
        case.benchmark,
        "householder_qr"
            | "rank_revealing_qr"
            | "rank_revealing_qr_deficient"
            | "solve_read_into"
            | "reduce_sum_squares_axis0"
            | "masked_log_softmax_axis1"
            | "where_select"
    )
}

/// The small dense perturbation and diagonal ensure full column rank. The
/// deficient variant has an exactly zero final quarter of its columns.
pub(super) fn fixture(
    rows: usize,
    cols: usize,
    deficient: bool,
) -> tenferro_tensor::Result<Tensor> {
    let mut data = data_f64(rows * cols, 1);
    for col in 0..cols {
        for row in 0..rows {
            let i = row + rows * col;
            data[i] = if deficient && col >= 3 * cols / 4 {
                0.0
            } else {
                data[i] * (0.125 / rows as f64)
                    + if row == col {
                        2.0 + col as f64 / cols as f64
                    } else {
                        0.0
                    }
            };
        }
    }
    Tensor::from_vec_col_major([rows, cols], data)
}

fn dims(case: &Case) -> (usize, usize) {
    let mut parts = case.shape.split(',').next().unwrap().split('x');
    (
        parts.next().unwrap().parse().unwrap(),
        parts.next().unwrap().parse().unwrap(),
    )
}

fn operation_count(case: &Case) -> usize {
    if case.suite == "cpu/neural_public_api" {
        return if case.shape.starts_with("1024x") {
            64
        } else {
            8
        };
    }
    // Sixteen retained dense QR outputs at the largest shape stay below
    // 64 MiB. Selection alone returns 256 MiB, so it is memory-limited to one.
    if case.benchmark == "where_select" {
        1
    } else {
        16
    }
}

/// Outputs, including opaque compact QR state, remain alive across clock stop.
/// An explicit initialization call is always performed even with zero warmups.
pub(super) fn sample_count<T, E: std::error::Error + 'static>(
    args: &Args,
    count: usize,
    mut op: impl FnMut() -> Result<T, E>,
) -> BenchResult<(f64, f64)> {
    drop(op()?);
    for _ in 0..args.warmups {
        drop(op()?);
    }
    let mut times = Vec::with_capacity(args.runs);
    for _ in 0..args.runs {
        let mut outputs = Vec::with_capacity(count);
        let start = Instant::now();
        for _ in 0..count {
            outputs.push(op()?);
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        black_box(&outputs);
        times.push(elapsed / count as f64);
        drop(outputs);
    }
    Ok(median_iqr(&times))
}

fn check_qr(a: &Tensor, q: &Tensor, r: &Tensor, permutation: Option<&Tensor>) -> BenchResult<()> {
    let m = a.shape()[0];
    let n = a.shape()[1];
    let k = m.min(n);
    let a = a.as_slice::<f64>()?;
    let q = q.as_slice::<f64>()?;
    let r = r.as_slice::<f64>()?;
    let p = permutation.map(|t| t.as_slice::<i64>()).transpose()?;
    let mut seen = vec![false; n];
    if let Some(p) = p {
        for &j in p {
            if j < 0 || j as usize >= n || seen[j as usize] {
                return Err("invalid QR permutation".into());
            }
            seen[j as usize] = true;
        }
    }
    let mut residual = 0.0_f64;
    for j in 0..n {
        for i in 0..m {
            let target_col = p.map_or(j, |p| p[j] as usize);
            let reconstructed: f64 = (0..k).map(|l| q[i + m * l] * r[l + k * j]).sum();
            residual = residual.max((reconstructed - a[i + m * target_col]).abs());
        }
    }
    let mut orthogonality = 0.0_f64;
    for j in 0..k {
        for i in 0..k {
            let value: f64 = (0..m).map(|l| q[l + m * i] * q[l + m * j]).sum();
            orthogonality = orthogonality.max((value - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    if !residual.is_finite()
        || !orthogonality.is_finite()
        || residual > 1e-9
        || orthogonality > 1e-9
    {
        return Err(format!(
            "QR validation: max_residual={residual}, orthogonality={orthogonality}"
        )
        .into());
    }
    Ok(())
}

fn check_solve(a: &Tensor, b: &Tensor, x: &Tensor) -> BenchResult<()> {
    let n = a.shape()[0];
    let nrhs = b.shape()[1];
    let a = a.as_slice::<f64>()?;
    let b = b.as_slice::<f64>()?;
    let x = x.as_slice::<f64>()?;
    let mut residual = 0.0_f64;
    for j in 0..nrhs {
        for i in 0..n {
            let value: f64 = (0..n).map(|l| a[i + n * l] * x[l + n * j]).sum();
            residual = residual.max((value - b[i + n * j]).abs());
        }
    }
    if !residual.is_finite() || residual > 1e-9 {
        return Err(format!("solve max_residual={residual}").into());
    }
    Ok(())
}

fn direct(args: &Args, backend: &mut CpuBackend, case: &Case) -> BenchResult<(f64, f64)> {
    let (m, n) = dims(case);
    let deficient = case.benchmark.ends_with("_deficient");
    let a = fixture(m, n, deficient)?;
    let options = RankRevealingQrOptions::default().rtol(1e-12);
    backend.with_backend_session(|s| -> BenchResult<_> {
        match case.benchmark {
            "householder_qr" => {
                let state = a.householder_qr(s)?;
                let q = state.q_columns(0..n, QrOptions::default(), s)?;
                let r = state.r(QrOptions::default(), s)?;
                check_qr(&a, &q, &r, None)?;
                sample_count(args, operation_count(case), || a.householder_qr(s))
            }
            "solve_read_into" => {
                let b = Tensor::from_vec_col_major([m, 16], data_f64(m * 16, 2))?;
                let mut out = Tensor::from_vec_col_major([m, 16], vec![f64::NAN; m * 16])?;
                TensorRead::from_tensor(&a).solve_read_into(
                    TensorRead::from_tensor(&b),
                    TensorWrite::from_tensor(&mut out),
                    s,
                )?;
                check_solve(&a, &b, &out)?;
                let stats = sample_count(args, operation_count(case), || {
                    TensorRead::from_tensor(&a).solve_read_into(
                        TensorRead::from_tensor(&b),
                        TensorWrite::from_tensor(&mut out),
                        s,
                    )
                })?;
                check_solve(&a, &b, &out)?;
                Ok(stats)
            }
            _ => {
                let output = a.rank_revealing_qr(options, s)?;
                check_qr(&a, &output.q, &output.r, Some(&output.column_permutation))?;
                let expected = if deficient { 3 * n / 4 } else { n };
                if output.rank.as_slice::<i64>()? != [expected as i64] {
                    return Err("incorrect RRQR numerical rank".into());
                }
                sample_count(args, operation_count(case), || {
                    a.rank_revealing_qr(options, s)
                })
            }
        }
    })?
}

fn eager(args: &Args, case: &Case) -> BenchResult<(f64, f64)> {
    let (m, n) = dims(case);
    let runtime = EagerRuntime::with_cpu_backend(cpu_backend()?)?;
    let _guard = runtime.no_grad();
    let original = fixture(m, n, case.benchmark.ends_with("_deficient"))?;
    let a = EagerTensor::from_tensor_in(original.duplicate()?, Arc::clone(&runtime))?;
    runtime.with_eager_session(|s| -> BenchResult<_> {
        if case.benchmark == "householder_qr" {
            let state = s.householder_qr(&a)?;
            let q = state.q_columns(0..n, QrOptions::default(), s)?;
            let r = state.r(QrOptions::default(), s)?;
            let q_read = q.tensor_read();
            let r_read = r.tensor_read();
            check_qr(
                &original,
                q_read.as_tensor().ok_or("eager Q is not materialized")?,
                r_read.as_tensor().ok_or("eager R is not materialized")?,
                None,
            )?;
            sample_count(args, operation_count(case), || s.householder_qr(&a))
        } else {
            let out = s.rank_revealing_qr(&a, RankRevealingQrOptions::default().rtol(1e-12))?;
            let q = out.q.tensor_read();
            let r = out.r.tensor_read();
            let p = out.column_permutation.tensor_read();
            check_qr(
                &original,
                q.as_tensor().ok_or("eager Q is not materialized")?,
                r.as_tensor().ok_or("eager R is not materialized")?,
                Some(
                    p.as_tensor()
                        .ok_or("eager permutation is not materialized")?,
                ),
            )?;
            let expected = if case.benchmark.ends_with("_deficient") {
                3 * n / 4
            } else {
                n
            };
            if out
                .rank
                .tensor_read()
                .as_tensor()
                .ok_or("eager rank is not materialized")?
                .as_slice::<i64>()?
                != [expected as i64]
            {
                return Err("incorrect eager RRQR rank".into());
            }
            sample_count(args, operation_count(case), || {
                s.rank_revealing_qr(&a, RankRevealingQrOptions::default().rtol(1e-12))
            })
        }
    })
}

pub(super) fn row(
    writer: &mut impl Write,
    args: &Args,
    case: &Case,
    backend: &str,
    result: BenchResult<(f64, f64)>,
) -> BenchResult<()> {
    let (median, iqr, status, notes) = match result {
        Ok((m,i)) => (format!("{m:.6}"), format!("{i:.6}"), "ok", format!("{}; setup=explicit; session=shared; validation=untimed; operations_per_sample={}; median_batch_ms={:.6}",case.notes,operation_count(case),m*operation_count(case) as f64)),
        Err(e) => (String::new(), String::new(), error_status(&e.to_string()), format!("{}; {}", case.notes, e)),
    };
    writeln!(
        writer,
        "{},{},{},{},\"{}\",{},{},{},{},\"{}\"",
        case.suite,
        case.benchmark,
        case.dtype,
        args.num_threads,
        case.shape,
        backend,
        median,
        iqr,
        status,
        csv_escape(&notes)
    )?;
    Ok(())
}

pub(super) fn emit(
    writer: &mut impl Write,
    args: &Args,
    backend: &mut CpuBackend,
    case: &Case,
) -> BenchResult<()> {
    if args.execution_filter.runs_direct() {
        let result = if session_case(case) {
            time_session_case(args, backend, case, None)
        } else {
            direct(args, backend, case)
        };
        row(writer, args, case, "tenferro-direct", result)?;
        if case.benchmark != "solve_read_into" {
            let result = if session_case(case) {
                eager_session_case(args, case)
            } else {
                eager(args, case)
            };
            row(writer, args, case, "tenferro-eager", result)?;
        } else {
            row(
                writer,
                args,
                case,
                "tenferro-eager",
                Err("unsupported: no public borrowed EagerSession caller-output solve API".into()),
            )?;
        }
    }
    if args.execution_filter.runs_trace() {
        let result = match case.benchmark {
            "householder_qr" => Err("unsupported: compiled trace cannot return opaque compact HouseholderQr state; Q/R extraction would add operations".into()),
            "solve_read_into" => Err("unsupported: compiled trace owns outputs; no caller-output solve API".into()),
            _ => Err("unsupported: Runtime::run_prepared has no public borrowed BackendSession API; per-call runtime session initialization is excluded from standard operation timing".into()),
        };
        row(writer, args, case, "tenferro-trace", result)?;
    }
    Ok(())
}

fn session_case(case: &Case) -> bool {
    matches!(
        case.benchmark,
        "reduce_sum_squares_axis0" | "masked_log_softmax_axis1" | "where_select"
    )
}

fn session_shape(case: &Case) -> Vec<usize> {
    match case.benchmark {
        "reduce_sum_squares_axis0" => vec![2048, 2048],
        "masked_log_softmax_axis1" => vec![1024, 64],
        _ => vec![EW_FAST_N],
    }
}

fn eager_session_case(args: &Args, case: &Case) -> BenchResult<(f64, f64)> {
    let shape = session_shape(case);
    let runtime = EagerRuntime::with_cpu_backend(cpu_backend()?)?;
    let _guard = runtime.no_grad();
    let input =
        EagerTensor::from_tensor_in(tensor_f64(&shape, 1).duplicate()?, Arc::clone(&runtime))?;
    let other = if case.benchmark == "where_select" {
        Some(EagerTensor::from_tensor_in(
            tensor_f64(&shape, 2).duplicate()?,
            Arc::clone(&runtime),
        )?)
    } else {
        None
    };
    let mask = if case.benchmark != "reduce_sum_squares_axis0" {
        Some(EagerTensor::from_tensor_in(
            tensor_bool(&shape).duplicate()?,
            Arc::clone(&runtime),
        )?)
    } else {
        None
    };
    let expected = session_expected(
        case.benchmark,
        tensor_f64(&shape, 1),
        if other.is_some() {
            Some(tensor_f64(&shape, 2))
        } else {
            None
        },
        if mask.is_some() {
            Some(tensor_bool(&shape))
        } else {
            None
        },
    )?;
    runtime.with_eager_session(|s| -> BenchResult<_> {
        let mut op = || match case.benchmark {
            "reduce_sum_squares_axis0" => s.reduce_sum_squares(&input, Some(&[0])),
            "masked_log_softmax_axis1" => s.masked_log_softmax(&input, mask.as_ref().unwrap(), 1),
            _ => s.where_select(mask.as_ref().unwrap(), &input, other.as_ref().unwrap()),
        };
        let checked = op()?;
        check_values(
            checked
                .tensor_read()
                .as_tensor()
                .ok_or("eager result is not materialized")?
                .as_slice::<f64>()?,
            &expected,
        )?;
        sample_count(args, operation_count(case), op)
    })
}

pub(super) fn time_session_case(
    args: &Args,
    backend: &mut CpuBackend,
    case: &Case,
    attribution: Option<&mut AttributionSink>,
) -> BenchResult<(f64, f64)> {
    // Allocate only this row's fixture; the previous implementation allocated
    // the 32M-element selection fixtures even for the small masked-softmax row.
    let shape: &[usize] = match case.benchmark {
        "reduce_sum_squares_axis0" => &[2048, 2048],
        "masked_log_softmax_axis1" => &[1024, 64],
        _ => &[EW_FAST_N],
    };
    let input = tensor_f64(shape, 1);
    let other = if case.benchmark == "where_select" {
        Some(tensor_f64(shape, 2))
    } else {
        None
    };
    let mask = if case.benchmark != "reduce_sum_squares_axis0" {
        Some(tensor_bool(shape))
    } else {
        None
    };
    let expected = session_expected(case.benchmark, input, other, mask)?;
    let stats = backend.with_backend_session(|session| -> BenchResult<_> {
        let mut op = || match case.benchmark {
            "reduce_sum_squares_axis0" => input.reduce_sum_squares(Some(&[0]), session),
            "masked_log_softmax_axis1" => input.masked_log_softmax(mask.unwrap(), 1, session),
            _ => mask.unwrap().where_select(input, other.unwrap(), session),
        };
        let output = op()?;
        check_values(output.as_slice::<f64>()?, &expected)?;
        sample_count(args, operation_count(case), op)
    })??;
    if let Some(attribution) = attribution {
        attribution.record(case, "direct", "steady_execute_median", 0, stats.0)?;
    }
    Ok(stats)
}

pub(super) fn check_values(actual: &[f64], expected: &[f64]) -> BenchResult<()> {
    if actual.len() != expected.len() {
        return Err("unexpected output length".into());
    }
    let mut residual = 0.0_f64;
    for (&a, &e) in actual.iter().zip(expected) {
        if a == e {
            continue;
        }
        if !a.is_finite() || !e.is_finite() {
            return Err("incorrect non-finite output".into());
        }
        residual = residual.max((a - e).abs() / (1.0 + e.abs()));
    }
    if residual > 1e-10 {
        return Err(format!("relative output residual={residual}").into());
    }
    Ok(())
}

fn session_expected(
    name: &str,
    input: &Tensor,
    other: Option<&Tensor>,
    mask: Option<&Tensor>,
) -> BenchResult<Vec<f64>> {
    let x = input.as_slice::<f64>()?;
    match name {
        "reduce_sum_squares_axis0" => {
            let m = input.shape()[0];
            Ok(x.chunks_exact(m)
                .map(|column| column.iter().map(|v| v * v).sum())
                .collect())
        }
        "masked_log_softmax_axis1" => {
            let m = input.shape()[0];
            let n = input.shape()[1];
            let mask = mask.unwrap().as_slice::<bool>()?;
            let mut output = vec![f64::NEG_INFINITY; x.len()];
            for i in 0..m {
                let maximum = (0..n)
                    .filter(|&j| mask[i + m * j])
                    .map(|j| x[i + m * j])
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = (0..n)
                    .filter(|&j| mask[i + m * j])
                    .map(|j| (x[i + m * j] - maximum).exp())
                    .sum();
                for j in 0..n {
                    if mask[i + m * j] {
                        output[i + m * j] = x[i + m * j] - maximum - sum.ln();
                    }
                }
            }
            Ok(output)
        }
        _ => {
            let y = other.unwrap().as_slice::<f64>()?;
            let mask = mask.unwrap().as_slice::<bool>()?;
            Ok(x.iter()
                .zip(y)
                .zip(mask)
                .map(|((&x, &y), &c)| if c { x } else { y })
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn public_linalg_outputs_reconstruct_the_fixture() -> BenchResult<()> {
        let mut backend = CpuBackend::with_threads(1)?;
        backend.with_backend_session(|s| -> BenchResult<()> {
            let a = fixture(12, 8, false)?;
            let state = a.householder_qr(s)?;
            check_qr(
                &a,
                &state.q_columns(0..8, QrOptions::default(), s)?,
                &state.r(QrOptions::default(), s)?,
                None,
            )?;
            for deficient in [false, true] {
                let a = fixture(12, 8, deficient)?;
                let out = a.rank_revealing_qr(RankRevealingQrOptions::default().rtol(1e-12), s)?;
                check_qr(&a, &out.q, &out.r, Some(&out.column_permutation))?;
                assert_eq!(out.rank.as_slice::<i64>()?, [if deficient { 6 } else { 8 }]);
            }
            let a = fixture(8, 8, false)?;
            let b = Tensor::from_vec_col_major([8, 2], data_f64(16, 2))?;
            let mut x = Tensor::from_vec_col_major([8, 2], vec![f64::NAN; 16])?;
            for _ in 0..2 {
                TensorRead::from_tensor(&a).solve_read_into(
                    TensorRead::from_tensor(&b),
                    TensorWrite::from_tensor(&mut x),
                    s,
                )?;
                check_solve(&a, &b, &x)?;
            }
            Ok(())
        })?
    }
}
