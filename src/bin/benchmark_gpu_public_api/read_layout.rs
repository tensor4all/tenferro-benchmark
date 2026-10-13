//! Transposed borrowed reads; view metadata and read handles are untimed.
use super::{measure, verify, Case, Result, Settings};
use num_complex::{Complex32, Complex64};
use serde_json::Value;
use tenferro_gpu::cuda::CudaRuntime;
use tenferro_tensor::{BackendSession, DType, Tensor, TensorRead, TensorView};

fn transpose(input: &Tensor) -> Result<TensorView<'_>> {
    macro_rules! view {
        ($ty:ty, $variant:ident) => {
            TensorView::$variant(
                input
                    .as_typed::<$ty>()
                    .ok_or("read dtype mismatch")?
                    .as_view()
                    .transpose_view([1, 0])?,
            )
        };
    }
    Ok(match input.dtype() {
        DType::F32 => view!(f32, F32),
        DType::F64 => view!(f64, F64),
        DType::C32 => view!(Complex32, C32),
        DType::C64 => view!(Complex64, C64),
        _ => return Err("unsupported strided fixture dtype".into()),
    })
}

fn execute(
    op: &str,
    reads: (TensorRead<'_>, Option<TensorRead<'_>>),
    s: &mut dyn BackendSession,
) -> Result<Tensor> {
    let (lhs, rhs) = reads;
    Ok(match op {
        "add" => s.add_read(lhs, rhs.ok_or("missing rhs")?)?,
        "mul" => s.mul_read(lhs, rhs.ok_or("missing rhs")?)?,
        "neg" => s.neg_read(lhs)?,
        "exp" => s.exp_read(lhs)?,
        "tanh" => s.tanh_read(lhs)?,
        "conj" => s.conj_read(lhs)?,
        "reduce_sum" => s.reduce_sum_read(lhs, &[1])?,
        _ => return Err(format!("unknown strided operation {op}").into()),
    })
}

pub(super) fn run(
    case: &Case,
    inputs: &[Tensor],
    expected: &Tensor,
    rt: &CudaRuntime,
    session: &mut dyn BackendSession,
    settings: &Settings,
    bytes: usize,
) -> Result<(Option<Value>, f64)> {
    let views = inputs.iter().map(transpose).collect::<Result<Vec<_>>>()?;
    let prepare = || {
        (
            TensorRead::from_view(views[0].clone()),
            views.get(1).cloned().map(TensorRead::from_view),
        )
    };
    let probe = execute(&case.read_op, prepare(), session)?;
    let error = verify(rt, &probe, expected, case)?;
    drop(probe);
    if settings.verify_only {
        return Ok((None, error));
    }
    let timing = measure(settings, bytes, rt, prepare, |reads| {
        execute(&case.read_op, reads, session)
    })?;
    let final_output = execute(&case.read_op, prepare(), session)?;
    Ok((
        Some(timing),
        error.max(verify(rt, &final_output, expected, case)?),
    ))
}
