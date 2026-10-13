use super::{Case, Result};
use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_einsum::{EagerSessionEinsumExt, TensorDotAxes, TensorTensordotExt};
use tenferro_linalg::{EagerSessionLinalgExt, TensorLinalgExt};
use tenferro_tensor::TensorRead;
use tenferro_tensor::{BackendSession, Tensor};

pub(super) fn direct(case: &Case, inputs: &[Tensor], s: &mut dyn BackendSession) -> Result<Tensor> {
    let x = &inputs[0];
    Ok(match case.op.as_str() {
        "tensordot" => x.tensordot(&inputs[1], TensorDotAxes::Count(1), s)?,
        "vdot_read" => s.vdot_read(
            TensorRead::from_tensor(x),
            TensorRead::from_tensor(&inputs[1]),
        )?,
        "norm_squared_read" => s.norm_squared_read(TensorRead::from_tensor(x))?,
        "cholesky" => x.cholesky(s)?,
        "svdvals" => x.svdvals(s)?,
        "eigvalsh" => x.eigvalsh(s)?,
        "pinv" => x.pinv_with_rtol(1e-5, s)?,
        "inv" => x.inv(s)?,
        "det" => x.det(s)?,
        "norm" => x.norm(None, None, false, s)?,
        "triangular_solve" => x.triangular_solve(&inputs[1], true, true, false, false, s)?,
        _ => return Err(format!("unknown linalg operation {}", case.op).into()),
    })
}
pub(super) fn eager(
    case: &Case,
    inputs: &[EagerTensor],
    s: &mut EagerSession<'_>,
) -> Result<EagerTensor> {
    let x = &inputs[0];
    Ok(match case.op.as_str() {
        "tensordot" => s.tensordot(x, &inputs[1], TensorDotAxes::Count(1))?,
        "cholesky" => s.cholesky(x)?,
        "eigvalsh" => s.eigvalsh(x)?,
        "pinv" => s.pinv_with_rtol(x, 1e-5)?,
        "inv" => s.inv(x)?,
        "det" => s.det(x)?,
        "norm" => s.norm(x, None, None, false)?,
        "triangular_solve" => s.triangular_solve(x, &inputs[1], true, true, false, false)?,
        _ => return Err(format!("unknown eager linalg operation {}", case.op).into()),
    })
}
