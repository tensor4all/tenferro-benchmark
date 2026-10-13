use super::{Case, Result, Settings};
use serde_json::Value;
use std::path::Path;
use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor};
use tenferro_gpu::cuda::{upload_tensor, CudaBackend, CudaDeviceId, CudaRuntime};
use tenferro_linalg::{EagerSessionLinalgExt, HouseholderQr, QrOptions, TensorLinalgExt};
use tenferro_tensor::{BackendSession, BackendSessionHost, DType, Tensor};

enum Output<T> {
    State(HouseholderQr<T>),
    Tensor(T),
}
fn direct(
    case: &Case,
    inputs: &[Tensor],
    state: Option<&HouseholderQr<Tensor>>,
    s: &mut dyn BackendSession,
) -> Result<Output<Tensor>> {
    Ok(match case.op.as_str() {
        "householder_from_factors" => Output::State(HouseholderQr::<Tensor>::from_factors(
            &inputs[1], &inputs[2], s,
        )?),
        "householder_append" => Output::State(
            state
                .ok_or("missing initial QR state")?
                .append_columns(&inputs[2], s)?,
        ),
        "householder_r" => Output::Tensor(
            state
                .ok_or("missing QR state")?
                .r(QrOptions::default(), s)?,
        ),
        "householder_q_columns" => Output::Tensor(state.ok_or("missing QR state")?.q_columns(
            case.column_range[0]..case.column_range[1],
            QrOptions::default(),
            s,
        )?),
        _ => return Err("unknown state operation".into()),
    })
}
fn eager(
    case: &Case,
    inputs: &[EagerTensor],
    state: Option<&HouseholderQr<EagerTensor>>,
    s: &mut EagerSession<'_>,
) -> Result<Output<EagerTensor>> {
    Ok(match case.op.as_str() {
        "householder_from_factors" => Output::State(HouseholderQr::<EagerTensor>::from_factors(
            &inputs[1], &inputs[2], s,
        )?),
        "householder_append" => Output::State(
            state
                .ok_or("missing initial QR state")?
                .append_columns(&inputs[2], s)?,
        ),
        "householder_r" => Output::Tensor(
            state
                .ok_or("missing QR state")?
                .r(QrOptions::default(), s)?,
        ),
        "householder_q_columns" => Output::Tensor(state.ok_or("missing QR state")?.q_columns(
            case.column_range[0]..case.column_range[1],
            QrOptions::default(),
            s,
        )?),
        _ => return Err("unknown state operation".into()),
    })
}
fn check_direct(
    rt: &CudaRuntime,
    output: &Output<Tensor>,
    expected: &[Tensor],
    input: &Tensor,
    case: &Case,
    s: &mut dyn BackendSession,
) -> Result<f64> {
    match output {
        Output::State(state) => {
            let q = state.q_columns(0..input.shape()[0], QrOptions::default(), s)?;
            let r = state.r(QrOptions::default(), s)?;
            super::multi::check_qr(rt, &q, &r, input, case)
        }
        Output::Tensor(t) => super::verify(rt, t, &expected[0], case),
    }
}
fn check_eager(
    rt: &CudaRuntime,
    output: &Output<EagerTensor>,
    expected: &[Tensor],
    input: &Tensor,
    case: &Case,
    s: &mut EagerSession<'_>,
) -> Result<f64> {
    match output {
        Output::State(state) => {
            let q = state.q_columns(0..input.shape()[0], QrOptions::default(), s)?;
            let r = state.r(QrOptions::default(), s)?;
            let qr = q.tensor_read();
            let rr = r.tensor_read();
            super::multi::check_qr(
                rt,
                qr.as_tensor().ok_or("Q projection not materialized")?,
                rr.as_tensor().ok_or("R projection not materialized")?,
                input,
                case,
            )
        }
        Output::Tensor(t) => super::verify(
            rt,
            t.tensor_read()
                .as_tensor()
                .ok_or("state query output not materialized")?,
            &expected[0],
            case,
        ),
    }
}
pub(super) fn run(
    root: &Path,
    case: &Case,
    mode: &str,
    settings: &Settings,
    device: usize,
) -> Result<(Option<Value>, f64)> {
    let hosts = case
        .inputs
        .iter()
        .map(|a| super::read_array(root, a))
        .collect::<Result<Vec<_>>>()?;
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
                    DType::C32 | DType::F64 => 8,
                    _ => 4,
                }
        })
        .sum();
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(device.try_into()?))?;
    let rt = backend.runtime().clone();
    let inputs = hosts
        .iter()
        .map(|t| Ok(upload_tensor(&rt, t)?))
        .collect::<Result<Vec<_>>>()?;
    if mode == "tenferro-cuda-direct" {
        backend.with_backend_session(|s| {
            let state = if case.op == "householder_from_factors" {
                None
            } else {
                Some(
                    inputs[if case.op == "householder_append" {
                        1
                    } else {
                        0
                    }]
                    .householder_qr(s)?,
                )
            };
            let probe = direct(case, &inputs, state.as_ref(), s)?;
            let error = check_direct(&rt, &probe, &expected, &hosts[0], case, s)?;
            drop(probe);
            if settings.verify_only || case.excluded.contains_key(mode) {
                return Ok((None, error));
            }
            let timing = super::measure(
                settings,
                bytes,
                &rt,
                || (),
                |()| direct(case, &inputs, state.as_ref(), s),
            )?;
            let final_out = direct(case, &inputs, state.as_ref(), s)?;
            let error = error.max(check_direct(
                &rt, &final_out, &expected, &hosts[0], case, s,
            )?);
            Ok((Some(timing), error))
        })?
    } else {
        let runtime = EagerRuntime::with_cuda_backend(backend)?;
        let _guard = runtime.no_grad();
        runtime.with_eager_session(|s| {
            let inputs = inputs
                .into_iter()
                .map(|t| Ok(s.constant_from(t)?))
                .collect::<Result<Vec<_>>>()?;
            let state = if case.op == "householder_from_factors" {
                None
            } else {
                Some(s.householder_qr(
                    &inputs[if case.op == "householder_append" {
                        1
                    } else {
                        0
                    }],
                )?)
            };
            let probe = eager(case, &inputs, state.as_ref(), s)?;
            let error = check_eager(&rt, &probe, &expected, &hosts[0], case, s)?;
            drop(probe);
            if settings.verify_only || case.excluded.contains_key(mode) {
                return Ok((None, error));
            }
            let timing = super::measure(
                settings,
                bytes,
                &rt,
                || (),
                |()| eager(case, &inputs, state.as_ref(), s),
            )?;
            let final_out = eager(case, &inputs, state.as_ref(), s)?;
            let error = error.max(check_eager(&rt, &final_out, &expected, &hosts[0], case, s)?);
            Ok((Some(timing), error))
        })
    }
}
