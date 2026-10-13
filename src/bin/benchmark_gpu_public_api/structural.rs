use super::{Case, Result};
use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_runtime::TensorSessionOpsExt;
use tenferro_tensor::{
    BackendSession, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
    Tensor, TensorRead,
};

fn pad(case: &Case) -> PadConfig {
    if !case.padding.is_empty() {
        return PadConfig {
            edge_padding_low: case.padding.iter().map(|p| p.0).collect(),
            edge_padding_high: case.padding.iter().map(|p| p.1).collect(),
            interior_padding: case.padding.iter().map(|p| p.2).collect(),
        };
    }
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
    pub(super) dot: Option<DotGeneralConfig>,
    pad: Option<PadConfig>,
    slice: Option<SliceConfig>,
    scatter: Option<ScatterConfig>,
    gather: Option<GatherConfig>,
    reshape: Option<Vec<usize>>,
}

pub(super) fn prepare(case: &Case) -> Arguments<'static> {
    let mut args = Arguments::default();
    match case.op.as_str() {
        "dot_general_axes" | "dot_general_conj" => {
            args.dot = Some(DotGeneralConfig {
                lhs_contracting_dims: case.lhs_contracting_dims.clone().into(),
                rhs_contracting_dims: case.rhs_contracting_dims.clone().into(),
                lhs_batch_dims: case.lhs_batch_dims.clone().into(),
                rhs_batch_dims: case.rhs_batch_dims.clone().into(),
            })
        }
        "reshape" => args.reshape = Some(case.output_shape.clone()),
        "pad" => args.pad = Some(pad(case)),
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
    // Direct reshape borrows the persistent shape; eager consumes a prepared vector.
    args.reshape = None;
    if matches!(
        case.op.as_str(),
        "vdot_read" | "norm_squared_read" | "dot_general_axes" | "dot_general_conj"
    ) {
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
        "reshape" => x.reshape(&case.output_shape, s)?,
        "stack" => Tensor::stack(&[x, &inputs[1]], case.axis, s)?,
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
        "reshape" => s.reshape(
            x,
            args.reshape
                .take()
                .ok_or("missing prepared reshape shape")?,
        )?,
        "stack" => s.stack(&[x, &inputs[1]], case.axis)?,
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
        _ => return super::linalg::eager(case, inputs, s, args),
    })
}

/// Inspect allocation identity only outside sampling, through the public raw seam.
pub(super) fn verify_materialized(
    case: &Case,
    inputs: &[&Tensor],
    output: &Tensor,
    session: &mut dyn BackendSession,
) -> Result<()> {
    if !matches!(case.op.as_str(), "reshape" | "stack") {
        return Ok(());
    }
    tenferro_gpu::cuda::with_cuda_exec_session(session, |cuda| {
        cuda.with_raw("verification.materialized", |raw| {
            macro_rules! check {
                ($ty:ty) => {{
                    let out = raw.tensor(output.as_typed::<$ty>().ok_or_else(|| {
                        tenferro_tensor::Error::unsupported(
                            "verification.materialized",
                            "output dtype mismatch",
                        )
                    })?)?;
                    for input in inputs {
                        let src = raw.tensor(input.as_typed::<$ty>().ok_or_else(|| {
                            tenferro_tensor::Error::unsupported(
                                "verification.materialized",
                                "input dtype mismatch",
                            )
                        })?)?;
                        // SAFETY: compare addresses only while both public raw references
                        // and their owning tensors remain alive; no pointer escapes the scope.
                        if unsafe { src.raw_ptr() == out.raw_ptr() } {
                            return Err(tenferro_tensor::Error::unsupported(
                                "verification.materialized",
                                "output aliases an input",
                            ));
                        }
                    }
                }};
            }
            match output.dtype() {
                DType::F32 => check!(f32),
                DType::F64 => check!(f64),
                DType::C32 => check!(num_complex::Complex32),
                DType::C64 => check!(num_complex::Complex64),
                DType::I32 => check!(i32),
                DType::I64 => check!(i64),
                _ => {
                    return Err(tenferro_tensor::Error::unsupported(
                        "verification.materialized",
                        "unselected dtype",
                    ))
                }
            }
            Ok(())
        })
    })
    .ok_or("verification missing CUDA session")??;
    Ok(())
}
