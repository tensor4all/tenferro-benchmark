use super::{Case, Result};
use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_einsum::{EagerSessionEinsumExt, TensorDotAxes, TensorTensordotExt};
use tenferro_linalg::{EagerSessionLinalgExt, TensorLinalgExt};
use tenferro_tensor::{Engine, Tensor};

pub(super) fn direct(
    case: &Case,
    inputs: &[Tensor],
    s: &mut dyn Engine,
    args: super::structural::Arguments<'_>,
) -> Result<Tensor> {
    let x = &inputs[0];
    Ok(match case.op.as_str() {
        "dot_general_axes" | "dot_general_conj" => {
            let (a, b) = args.reads.ok_or("missing prepared dot reads")?;
            let config = args.dot.ok_or("missing prepared dot configuration")?;
            if case.op == "dot_general_conj" {
                s.dot_general_with_conj_read(a, b, &config, case.lhs_conj, case.rhs_conj)?
            } else {
                s.dot_general_read(a, b, &config)?
            }
        }
        "tensordot" => x.tensordot(&inputs[1], TensorDotAxes::Count(1), s)?,
        "vdot_read" => {
            let (a, b) = args.reads.ok_or("missing prepared read operands")?;
            s.vdot_read(a, b)?
        }
        "norm_squared_read" => {
            let (a, _) = args.reads.ok_or("missing prepared read operand")?;
            s.norm_squared_read(a)?
        }
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
    args: super::structural::Arguments<'_>,
) -> Result<EagerTensor> {
    let x = &inputs[0];
    Ok(match case.op.as_str() {
        "dot_general_axes" => s.dot_general(
            x,
            &inputs[1],
            args.dot.ok_or("missing prepared dot configuration")?,
        )?,
        "dot_general_conj" => s.dot_general_with_conj(
            x,
            &inputs[1],
            args.dot.ok_or("missing prepared dot configuration")?,
            case.lhs_conj,
            case.rhs_conj,
        )?,
        "tensordot" => s.tensordot(x, &inputs[1], TensorDotAxes::Count(1))?,
        "lstsq" => s.lstsq(x, &inputs[1])?,
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
