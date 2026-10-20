//! Transposed borrowed reads; view metadata and read handles are untimed.
use super::{measure, verify, Case, Result, Settings};
use num_complex::{Complex32, Complex64};
use serde_json::Value;
use tenferro_gpu::cuda::CudaRuntime;
use tenferro_tensor::{DType, Engine, Tensor, TensorRead, TensorView};

pub(super) fn transpose(input: &Tensor) -> Result<TensorView<'_>> {
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
    s: &mut dyn Engine,
) -> Result<Tensor> {
    let (lhs, rhs) = reads;
    Ok(match op {
        "add" => s.add_read(lhs, rhs.ok_or("missing rhs")?)?,
        "mul" => s.mul_read(lhs, rhs.ok_or("missing rhs")?)?,
        "sub" => s.sub_read(lhs, rhs.ok_or("missing rhs")?)?,
        "div" => s.div_read(lhs, rhs.ok_or("missing rhs")?)?,
        "rem" => s.rem_read(lhs, rhs.ok_or("missing rhs")?)?,
        "pow" => s.pow_read(lhs, rhs.ok_or("missing rhs")?)?,
        "maximum" => s.maximum_read(lhs, rhs.ok_or("missing rhs")?)?,
        "minimum" => s.minimum_read(lhs, rhs.ok_or("missing rhs")?)?,
        "neg" => s.neg_read(lhs)?,
        "abs" => s.abs_read(lhs)?,
        "sign" => s.sign_read(lhs)?,
        "expm1" => s.expm1_read(lhs)?,
        "log" => s.log_read(lhs)?,
        "log1p" => s.log1p_read(lhs)?,
        "erf" => s.erf_read(lhs)?,
        "sin" => s.sin_read(lhs)?,
        "cos" => s.cos_read(lhs)?,
        "sqrt" => s.sqrt_read(lhs)?,
        "rsqrt" => s.rsqrt_read(lhs)?,
        "exp" => s.exp_read(lhs)?,
        "tanh" => s.tanh_read(lhs)?,
        "conj" => s.conj_read(lhs)?,
        "reduce_sum" => s.reduce_sum_read(lhs, &[1])?,
        "reduce_prod" => s.reduce_prod_read(lhs, &[1])?,
        "reduce_max" => s.reduce_max_read(lhs, &[1])?,
        "reduce_min" => s.reduce_min_read(lhs, &[1])?,
        "reduce_sum_squares" => s.reduce_sum_squares_read(lhs, &[1])?,
        _ => return Err(format!("unknown strided operation {op}").into()),
    })
}

pub(super) fn run(
    case: &Case,
    inputs: &[Tensor],
    expected: &Tensor,
    rt: &CudaRuntime,
    session: &mut dyn Engine,
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
