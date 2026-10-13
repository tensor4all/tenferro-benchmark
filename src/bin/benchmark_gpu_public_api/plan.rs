//! Concrete public plans are built once before execution and sampling.
use super::{measure, read_array, verify, Case, Result, Settings};
use serde_json::Value;
use std::path::Path;
use tenferro_einsum::ConcreteEinsumPlan;
use tenferro_gpu::cuda::{upload_tensor, CudaBackend, CudaDeviceId};
use tenferro_tensor::{BackendSessionHost, DType, Tensor, TensorRead};

pub(super) fn run(
    root: &Path,
    case: &Case,
    settings: &Settings,
    device: usize,
) -> Result<(Option<Value>, f64)> {
    let hosts = case
        .inputs
        .iter()
        .map(|spec| read_array(root, spec))
        .collect::<Result<Vec<_>>>()?;
    let expected = read_array(root, &case.outputs[0])?;
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(device.try_into()?))?;
    let rt = backend.runtime().clone();
    let inputs = hosts
        .iter()
        .map(|value| Ok(upload_tensor(&rt, value)?))
        .collect::<Result<Vec<_>>>()?;
    let strided = case.op.contains("strided");
    let views = if strided {
        Some(
            inputs[..2]
                .iter()
                .map(super::read_layout::transpose)
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };
    let reads = || match &views {
        Some(views) => [
            TensorRead::from_view(views[0].clone()),
            TensorRead::from_view(views[1].clone()),
        ],
        None => [
            TensorRead::from_tensor(&inputs[0]),
            TensorRead::from_tensor(&inputs[1]),
        ],
    };
    let owned = [&inputs[0], &inputs[1]];
    let plan = if case.op.contains("read") {
        ConcreteEinsumPlan::prepare_read(reads(), "ik,kj->ij")?
    } else {
        ConcreteEinsumPlan::prepare(owned, "ik,kj->ij")?
    };
    let bytes = expected.shape().iter().product::<usize>()
        * match expected.dtype() {
            DType::C64 => 16,
            DType::C32 | DType::F64 => 8,
            _ => 4,
        };
    backend.with_backend_session(|session| {
        if case.op.contains("_into") {
            return super::reuse::run(
                case,
                &inputs,
                &expected,
                &rt,
                session,
                settings,
                bytes,
                Some(&plan),
                views.as_deref(),
            );
        }
        let mut execute = |args: [TensorRead<'_>; 2]| -> Result<Tensor> {
            if case.op == "einsum_plan_owned" {
                Ok(plan.execute(owned, session)?)
            } else {
                Ok(plan.execute_read(args, session)?)
            }
        };
        let probe = execute(reads())?;
        let error = verify(&rt, &probe, &expected, case)?;
        drop(probe);
        if settings.verify_only {
            return Ok((None, error));
        }
        let timing = measure(settings, bytes, &rt, reads, |args| execute(args))?;
        let output = execute(reads())?;
        Ok((
            Some(timing),
            error.max(verify(&rt, &output, &expected, case)?),
        ))
    })?
}
