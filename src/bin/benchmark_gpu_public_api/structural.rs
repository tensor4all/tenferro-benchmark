use super::{Case, Result};
use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_runtime::TensorSessionOpsExt;
use tenferro_tensor::{BackendSession, DType, PadConfig, ScatterConfig, SliceConfig, Tensor};

fn pad() -> PadConfig {
    PadConfig {
        edge_padding_low: vec![0, 8],
        edge_padding_high: vec![0, 8],
        interior_padding: vec![0, 0],
    }
}
fn slice(shape: &[usize]) -> SliceConfig {
    SliceConfig {
        starts: vec![0, 0],
        limits: vec![shape[0], shape[1] / 2],
        strides: vec![1, 1],
    }
}
fn scatter() -> ScatterConfig {
    ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    }
}
pub(super) fn direct(case: &Case, inputs: &[Tensor], s: &mut dyn BackendSession) -> Result<Tensor> {
    let x = &inputs[0];
    Ok(match case.op.as_str() {
        "pad" => x.pad(pad(), s)?,
        "slice" => x.slice(slice(x.shape()), s)?,
        "reverse" => x.reverse(&[1], s)?,
        "concatenate" => Tensor::concatenate(&[x, &inputs[1]], 1, s)?,
        "extract_diag" => x.extract_diag(0, 1, s)?,
        "embed_diag" => x.embed_diag(0, 1, s)?,
        "tril" => x.tril(0, s)?,
        "triu" => x.triu(0, s)?,
        "take_along_axis" => x.take_along_axis(&inputs[1], 1, s)?,
        "scatter" => x.scatter(&inputs[1], &inputs[2], scatter(), s)?,
        "cast" => x.cast(
            if x.dtype() == DType::F32 {
                DType::F64
            } else {
                DType::F32
            },
            s,
        )?,
        _ => return super::linalg::direct(case, inputs, s),
    })
}
pub(super) fn eager(
    case: &Case,
    inputs: &[EagerTensor],
    s: &mut EagerSession<'_>,
) -> Result<EagerTensor> {
    let x = &inputs[0];
    Ok(match case.op.as_str() {
        "pad" => s.pad(x, pad())?,
        "slice" => s.slice(x, slice(x.shape()))?,
        "reverse" => s.reverse(x, &[1])?,
        "concatenate" => s.concatenate(&[x, &inputs[1]], 1)?,
        "extract_diag" => s.extract_diag(x, 0, 1)?,
        "embed_diag" => s.embed_diag(x, 0, 1)?,
        "tril" => s.tril(x, 0)?,
        "triu" => s.triu(x, 0)?,
        "take_along_axis" => s.take_along_axis(x, &inputs[1], 1)?,
        "scatter" => s.scatter(x, &inputs[1], &inputs[2], scatter())?,
        "cast" => s.cast(
            x,
            if x.dtype() == DType::F32 {
                DType::F64
            } else {
                DType::F32
            },
        )?,
        _ => return super::linalg::eager(case, inputs, s),
    })
}
