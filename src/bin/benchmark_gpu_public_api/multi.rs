use super::{Case, Result, Settings};
use num_complex::Complex64;
use serde_json::Value;
use std::path::Path;
use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor};
use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaBackend, CudaDeviceId, CudaRuntime};
use tenferro_linalg::{EagerSessionLinalgExt, HouseholderQr, QrOptions, TensorLinalgExt};
use tenferro_tensor::{BackendSession, BackendSessionHost, DType, Tensor};

// Stack-owned adapters add no output-retention allocation to the timed call.
enum Outputs<T> {
    Two([T; 2]),
    Three([T; 3]),
    Four([T; 4]),
    Compact(HouseholderQr<T>),
}
impl<T> Outputs<T> {
    fn values(&self) -> &[T] {
        match self {
            Self::Two(v) => v,
            Self::Three(v) => v,
            Self::Four(v) => v,
            Self::Compact(_) => panic!("opaque QR is inspected through public projections"),
        }
    }
}
fn direct(case: &Case, x: &Tensor, session: &mut dyn BackendSession) -> Result<Outputs<Tensor>> {
    Ok(match case.op.as_str() {
        "lu" => {
            let (p, l, u, parity) = x.lu(session)?;
            Outputs::Four([p, l, u, parity])
        }
        "householder_qr" => Outputs::Compact(x.householder_qr(session)?),
        "svd_full" => {
            let (u, s, vh) = x.svd_full(session)?;
            Outputs::Three([u, s, vh])
        }
        "slogdet" => {
            let (sign, logabs) = x.slogdet(session)?;
            Outputs::Two([sign, logabs])
        }
        _ => return Err("unknown multi-output operation".into()),
    })
}
fn eager(
    case: &Case,
    x: &EagerTensor,
    session: &mut EagerSession<'_>,
) -> Result<Outputs<EagerTensor>> {
    Ok(match case.op.as_str() {
        "lu" => {
            let (p, l, u, parity) = session.lu(x)?;
            Outputs::Four([p, l, u, parity])
        }
        "householder_qr" => Outputs::Compact(session.householder_qr(x)?),
        "svd_full" => {
            let (u, s, vh) = session.svd_full(x)?;
            Outputs::Three([u, s, vh])
        }
        "slogdet" => {
            let (sign, logabs) = session.slogdet(x)?;
            Outputs::Two([sign, logabs])
        }
        _ => return Err("unknown multi-output operation".into()),
    })
}
fn check(
    rt: &CudaRuntime,
    output: &[&Tensor],
    expected: &[Tensor],
    input: &Tensor,
    case: &Case,
) -> Result<f64> {
    if output.len() != expected.len() {
        return Err("verification output arity mismatch".into());
    }
    if case.op != "svd_full" {
        return output
            .iter()
            .zip(expected)
            .try_fold(0_f64, |worst, (a, e)| {
                Ok(worst.max(super::verify(rt, a, e, case)?))
            });
    }
    let mut host = Vec::with_capacity(output.len());
    for (actual, reference) in output.iter().zip(expected) {
        let placement = actual.placement();
        let device = placement
            .device
            .as_ref()
            .ok_or("CPU fallback: no CUDA placement")?;
        if device.kind != tenferro_tensor::DeviceKind::Gpu(tenferro_tensor::GpuBackendKind::Cuda)
            || device.ordinal != super::env_usize("GPU_BENCH_DEVICE", 0)?
            || !actual.is_backend_buffer()
        {
            return Err("CPU fallback or wrong CUDA output".into());
        }
        if actual.shape() != reference.shape() || actual.dtype() != reference.dtype() {
            return Err("verification full-SVD shape/dtype mismatch".into());
        }
        host.push(download_tensor(rt, actual)?);
    }
    let u = super::numbers(&host[0])?;
    let s = super::numbers(&host[1])?;
    let vh = super::numbers(&host[2])?;
    let a = super::numbers(input)?;
    if u.iter()
        .chain(&s)
        .chain(&vh)
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err("verification nonfinite full-SVD output".into());
    }
    if s.iter().any(|v| v.re < 0. || v.im != 0.) || s.windows(2).any(|v| v[1].re > v[0].re) {
        return Err("verification singular values are not sorted/nonnegative".into());
    }
    let m = input.shape()[0];
    let n = input.shape()[1];
    let k = m.min(n);
    let mut worst = super::verify(rt, output[1], &expected[1], case)?;
    for j in 0..n {
        for i in 0..m {
            let reconstructed = (0..k).fold(Complex64::new(0., 0.), |v, p| {
                v + u[i + m * p] * s[p] * vh[p + n * j]
            });
            worst = worst.max(
                (reconstructed - a[i + m * j]).norm()
                    / (case.atol + case.rtol * a[i + m * j].norm()),
            );
        }
    }
    // Check the complete square factors, including the nullspace columns/rows.
    for j in 0..m {
        for i in 0..m {
            let value = (0..m).fold(Complex64::new(0., 0.), |v, p| {
                v + u[p + m * i].conj() * u[p + m * j]
            });
            let target = if i == j { 1. } else { 0. };
            worst = worst.max((value - target).norm() / (case.atol + case.rtol * target));
        }
    }
    for j in 0..n {
        for i in 0..n {
            let value = (0..n).fold(Complex64::new(0., 0.), |v, p| {
                v + vh[i + n * p] * vh[j + n * p].conj()
            });
            let target = if i == j { 1. } else { 0. };
            worst = worst.max((value - target).norm() / (case.atol + case.rtol * target));
        }
    }
    if !worst.is_finite() || worst > 1. {
        return Err(format!("verification full-SVD error/tolerance = {worst}").into());
    }
    Ok(worst)
}
pub(super) fn check_qr(
    rt: &CudaRuntime,
    q: &Tensor,
    r: &Tensor,
    input: &Tensor,
    case: &Case,
) -> Result<f64> {
    let m = input.shape()[0];
    let n = input.shape()[1];
    let k = m.min(n);
    if q.shape() != [m, m]
        || r.shape() != [k, n]
        || q.dtype() != input.dtype()
        || r.dtype() != input.dtype()
    {
        return Err("verification compact-QR projection shape/dtype mismatch".into());
    }
    let q = super::numbers(&download_tensor(rt, q)?)?;
    let r = super::numbers(&download_tensor(rt, r)?)?;
    let a = super::numbers(input)?;
    if q.iter()
        .chain(&r)
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err("verification nonfinite compact-QR projection".into());
    }
    let mut worst = 0_f64;
    for j in 0..n {
        for i in 0..m {
            let value = (0..k).fold(Complex64::new(0., 0.), |v, p| {
                v + q[i + m * p] * r[p + k * j]
            });
            worst = worst
                .max((value - a[i + m * j]).norm() / (case.atol + case.rtol * a[i + m * j].norm()));
        }
    }
    for j in 0..m {
        for i in 0..m {
            let value = (0..m).fold(Complex64::new(0., 0.), |v, p| {
                v + q[p + m * i].conj() * q[p + m * j]
            });
            let target = if i == j { 1. } else { 0. };
            worst = worst.max((value - target).norm() / (case.atol + case.rtol * target));
        }
    }
    for j in 0..n {
        for i in (j + 1)..k {
            worst = worst.max(r[i + k * j].norm() / case.atol);
        }
    }
    if !worst.is_finite() || worst > 1. {
        return Err(format!("verification compact-QR error/tolerance = {worst}").into());
    }
    Ok(worst)
}
fn check_direct(
    rt: &CudaRuntime,
    out: &Outputs<Tensor>,
    expected: &[Tensor],
    input: &Tensor,
    case: &Case,
    session: &mut dyn BackendSession,
) -> Result<f64> {
    if let Outputs::Compact(state) = out {
        let q = state.q_columns(0..input.shape()[0], QrOptions::default(), session)?;
        let r = state.r(QrOptions::default(), session)?;
        return check_qr(rt, &q, &r, input, case);
    }
    check(
        rt,
        &out.values().iter().collect::<Vec<_>>(),
        expected,
        input,
        case,
    )
}
fn check_eager(
    rt: &CudaRuntime,
    outputs: &Outputs<EagerTensor>,
    expected: &[Tensor],
    input: &Tensor,
    case: &Case,
    session: &mut EagerSession<'_>,
) -> Result<f64> {
    if let Outputs::Compact(state) = outputs {
        let q = state.q_columns(0..input.shape()[0], QrOptions::default(), session)?;
        let r = state.r(QrOptions::default(), session)?;
        let qr = q.tensor_read();
        let rr = r.tensor_read();
        return check_qr(
            rt,
            qr.as_tensor().ok_or("QR projection not materialized")?,
            rr.as_tensor().ok_or("R projection not materialized")?,
            input,
            case,
        );
    }
    let reads = outputs
        .values()
        .iter()
        .map(|v| v.tensor_read())
        .collect::<Vec<_>>();
    let tensors = reads
        .iter()
        .map(|r| {
            r.as_tensor()
                .ok_or_else(|| "eager multi-output result is not materialized".into())
        })
        .collect::<Result<Vec<_>>>()?;
    check(rt, &tensors, expected, input, case)
}
pub(super) fn run(
    root: &Path,
    case: &Case,
    mode: &str,
    settings: &Settings,
    device: usize,
) -> Result<(Option<Value>, f64)> {
    let input = super::read_array(root, &case.inputs[0])?;
    let expected = case
        .outputs
        .iter()
        .map(|a| super::read_array(root, a))
        .collect::<Result<Vec<_>>>()?;
    let bytes = expected
        .iter()
        .map(|a| {
            a.shape().iter().product::<usize>()
                * match a.dtype() {
                    DType::C64 => 16,
                    DType::F64 | DType::C32 => 8,
                    _ => 4,
                }
        })
        .sum();
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(device.try_into()?))?;
    let rt = backend.runtime().clone();
    let x = upload_tensor(&rt, &input)?;
    if mode == "tenferro-cuda-direct" {
        backend.with_backend_session(|session| {
            let probe = direct(case, &x, session)?;
            let error = check_direct(&rt, &probe, &expected, &input, case, session)?;
            drop(probe);
            if settings.verify_only || case.excluded.contains_key(mode) {
                return Ok((None, error));
            }
            let timing =
                super::measure(settings, bytes, &rt, || (), |()| direct(case, &x, session))?;
            let final_output = direct(case, &x, session)?;
            let error = error.max(check_direct(
                &rt,
                &final_output,
                &expected,
                &input,
                case,
                session,
            )?);
            Ok((Some(timing), error))
        })?
    } else {
        let runtime = EagerRuntime::with_cuda_backend(backend)?;
        let _guard = runtime.no_grad();
        runtime.with_eager_session(|session| {
            let x = session.constant_from(x)?;
            let probe = eager(case, &x, session)?;
            let error = check_eager(&rt, &probe, &expected, &input, case, session)?;
            drop(probe);
            if settings.verify_only || case.excluded.contains_key(mode) {
                return Ok((None, error));
            }
            let timing =
                super::measure(settings, bytes, &rt, || (), |()| eager(case, &x, session))?;
            let final_output = eager(case, &x, session)?;
            let error = error.max(check_eager(
                &rt,
                &final_output,
                &expected,
                &input,
                case,
                session,
            )?);
            Ok((Some(timing), error))
        })
    }
}
