use super::{Case, Result};
use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_runtime::TensorSessionOpsExt;
use tenferro_tensor::{
    BackendSession, DType, GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor, TensorRead,
};

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
#[derive(Default)]
pub(super) struct Arguments<'a> {
    pub(super) reads: Option<(TensorRead<'a>, TensorRead<'a>)>,
    pad: Option<PadConfig>,
    slice: Option<SliceConfig>,
    scatter: Option<ScatterConfig>,
    gather: Option<GatherConfig>,
}

pub(super) fn prepare(case: &Case) -> Arguments<'static> {
    let mut args = Arguments::default();
    match case.op.as_str() {
        "pad" => args.pad = Some(pad()),
        "slice" => args.slice = Some(slice(&case.inputs[0].shape)),
        "scatter" => args.scatter = Some(scatter()),
        "gather" => {
            let axis = case.gather_axis;
            let mut sizes = case.inputs[0].shape.clone();
            sizes[axis] = 1;
            args.gather = Some(GatherConfig {
                offset_dims: vec![1 - axis],
                collapsed_slice_dims: vec![axis],
                start_index_map: vec![axis],
                index_vector_dim: 1,
                slice_sizes: sizes,
            });
        }
        _ => {}
    }
    args
}

pub(super) fn prepare_direct<'a>(case: &Case, inputs: &'a [Tensor]) -> Arguments<'a> {
    let mut args = prepare(case);
    if matches!(case.op.as_str(), "vdot_read" | "norm_squared_read") {
        args.reads = Some((
            TensorRead::from_tensor(&inputs[0]),
            TensorRead::from_tensor(inputs.get(1).unwrap_or(&inputs[0])),
        ));
    }
    args
}

pub(super) fn direct(
    case: &Case,
    inputs: &[Tensor],
    s: &mut dyn BackendSession,
    mut args: Arguments<'_>,
) -> Result<Tensor> {
    let x = &inputs[0];
    Ok(match case.op.as_str() {
        "pad" => x.pad(args.pad.take().ok_or("missing prepared pad config")?, s)?,
        "slice" => x.slice(args.slice.take().ok_or("missing prepared slice config")?, s)?,
        "reverse" => x.reverse(&[1], s)?,
        "concatenate" => Tensor::concatenate(&[x, &inputs[1]], 1, s)?,
        "broadcast_in_dim" => x.broadcast_in_dim(&case.output_shape, &case.broadcast_dims, s)?,
        "convert" => x.convert(DType::F64, s)?,
        "dynamic_slice" => x.dynamic_slice(&inputs[1], &case.slice_sizes, s)?,
        "gather" => x.gather(
            &inputs[1],
            args.gather.take().ok_or("missing prepared gather config")?,
            s,
        )?,
        "extract_diag" => x.extract_diag(0, 1, s)?,
        "embed_diag" => x.embed_diag(0, 1, s)?,
        "tril" => x.tril(0, s)?,
        "triu" => x.triu(0, s)?,
        "take_along_axis" => x.take_along_axis(&inputs[1], 1, s)?,
        "scatter" => x.scatter(
            &inputs[1],
            &inputs[2],
            args.scatter
                .take()
                .ok_or("missing prepared scatter config")?,
            s,
        )?,
        "cast" => x.cast(
            if x.dtype() == DType::F32 {
                DType::F64
            } else {
                DType::F32
            },
            s,
        )?,
        _ => return super::linalg::direct(case, inputs, s, args),
    })
}
pub(super) fn eager(
    case: &Case,
    inputs: &[EagerTensor],
    s: &mut EagerSession<'_>,
    mut args: Arguments<'_>,
) -> Result<EagerTensor> {
    let x = &inputs[0];
    Ok(match case.op.as_str() {
        "pad" => s.pad(x, args.pad.take().ok_or("missing prepared pad config")?)?,
        "slice" => s.slice(x, args.slice.take().ok_or("missing prepared slice config")?)?,
        "reverse" => s.reverse(x, &[1])?,
        "concatenate" => s.concatenate(&[x, &inputs[1]], 1)?,
        "broadcast_in_dim" => s.broadcast_in_dim(x, &case.output_shape, &case.broadcast_dims)?,
        "convert" => s.convert(x, DType::F64)?,
        "dynamic_slice" => s.dynamic_slice(x, &inputs[1], &case.slice_sizes)?,
        "gather" => s.gather(
            x,
            &inputs[1],
            args.gather.take().ok_or("missing prepared gather config")?,
        )?,
        "extract_diag" => s.extract_diag(x, 0, 1)?,
        "embed_diag" => s.embed_diag(x, 0, 1)?,
        "tril" => s.tril(x, 0)?,
        "triu" => s.triu(x, 0)?,
        "take_along_axis" => s.take_along_axis(x, &inputs[1], 1)?,
        "scatter" => s.scatter(
            x,
            &inputs[1],
            &inputs[2],
            args.scatter
                .take()
                .ok_or("missing prepared scatter config")?,
        )?,
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
