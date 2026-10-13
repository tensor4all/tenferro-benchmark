//! Output buffers and move-only write handles are prepared before every clock.
use super::{sample_batches, verify, Case, Result, Settings};
use num_complex::{Complex32, Complex64};
use serde_json::Value;
use std::{hint::black_box, time::Instant};
use tenferro_einsum::ConcreteEinsumPlan;
use tenferro_gpu::cuda::{upload_tensor, CudaRuntime};
use tenferro_tensor::{
    BackendSession, ContractionScalar, DType, DotGeneralAccumulation, DotGeneralConfig, Tensor,
    TensorRead, TensorView, TensorWrite,
};

fn execute(
    op: &str,
    inputs: &[Tensor],
    a: TensorRead<'_>,
    b: TensorRead<'_>,
    out: TensorWrite<'_>,
    s: &mut dyn BackendSession,
    scalars: (ContractionScalar, ContractionScalar),
    config: &DotGeneralConfig,
    dot_accumulation: DotGeneralAccumulation,
    plan: Option<&ConcreteEinsumPlan>,
) -> Result<()> {
    let x = &inputs[0];
    let y = inputs.get(1).unwrap_or(x);
    match op {
        "einsum_plan_into" => plan
            .ok_or("missing prepared plan")?
            .execute_into([x, y], s, out)?,
        "einsum_plan_read_into" | "einsum_plan_strided_read_into" => plan
            .ok_or("missing prepared plan")?
            .execute_read_into([a, b], s, out)?,
        "einsum_plan_read_into_accum" | "einsum_plan_strided_read_into_accum" => plan
            .ok_or("missing prepared plan")?
            .execute_read_into_accum([a, b], s, dot_accumulation, out)?,
        "dot_general_read_into" => s.dot_general_read_into(a, b, config, out)?,
        "dot_general_read_into_accum" => {
            s.dot_general_read_into_accum(a, b, config, dot_accumulation, out)?
        }
        "dot_general_read_into_accum_cached" => {
            s.dot_general_read_into_accum_cached(Some(0), a, b, config, dot_accumulation, out)?
        }
        "axpby_read_into_accum" => s.axpby_read_into_accum(scalars.0, a, scalars.1, out)?,
        "add_into" => s.add_into(x, y, out)?,
        "sub_into" => s.sub_into(x, y, out)?,
        "mul_into" => s.mul_into(x, y, out)?,
        "div_into" => s.div_into(x, y, out)?,
        "neg_into" => s.neg_into(x, out)?,
        "conj_into" => s.conj_into(x, out)?,
        "add_read_into" => s.add_read_into(a, b, out)?,
        "sub_read_into" => s.sub_read_into(a, b, out)?,
        "mul_read_into" => s.mul_read_into(a, b, out)?,
        "div_read_into" => s.div_read_into(a, b, out)?,
        "neg_read_into" => s.neg_read_into(a, out)?,
        "conj_read_into" => s.conj_read_into(a, out)?,
        _ => return Err(format!("unknown output reuse operation {op}").into()),
    }
    Ok(())
}
pub(super) fn run(
    case: &Case,
    inputs: &[Tensor],
    expected: &Tensor,
    rt: &CudaRuntime,
    s: &mut dyn BackendSession,
    settings: &Settings,
    bytes: usize,
    plan: Option<&ConcreteEinsumPlan>,
    views: Option<&[TensorView<'_>]>,
) -> Result<(Option<Value>, f64)> {
    let read = |index: usize| match views {
        Some(views) => TensorRead::from_view(views[index].clone()),
        None => TensorRead::from_tensor(&inputs[index]),
    };
    let n = expected.shape().iter().product::<usize>();
    let shape = expected.shape().to_vec();
    let alpha = case.alpha.unwrap_or(0.5);
    let beta = case.beta.unwrap_or(1.0);
    let scalars = match expected.dtype() {
        DType::F32 => (
            ContractionScalar::F32(alpha as f32),
            ContractionScalar::F32(beta as f32),
        ),
        DType::F64 => (ContractionScalar::F64(alpha), ContractionScalar::F64(beta)),
        DType::C32 => (
            ContractionScalar::C32(Complex32::new(alpha as f32, 0.)),
            ContractionScalar::C32(Complex32::new(beta as f32, 0.)),
        ),
        DType::C64 => (
            ContractionScalar::C64(Complex64::new(alpha, 0.)),
            ContractionScalar::C64(Complex64::new(beta, 0.)),
        ),
        _ => return Err("invalid reuse dtype".into()),
    };
    let accum = case.op.contains("_into_accum");
    let seed = if case.op.starts_with("dot_general") || case.op.starts_with("einsum_plan") {
        2
    } else {
        1
    };
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let dot_accumulation = if accum {
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: scalars.0,
            beta: scalars.1,
        }
    } else {
        DotGeneralAccumulation::overwrite(expected.dtype())?
    };
    let zero = match expected.dtype() {
        DType::F32 => Tensor::from_vec_col_major(shape, vec![0f32; n])?,
        DType::F64 => Tensor::from_vec_col_major(shape, vec![0f64; n])?,
        DType::C32 => Tensor::from_vec_col_major(shape, vec![Complex32::new(0., 0.); n])?,
        DType::C64 => Tensor::from_vec_col_major(shape, vec![Complex64::new(0., 0.); n])?,
        _ => return Err("invalid reuse dtype".into()),
    };
    let cap = if settings.verify_only {
        1
    } else {
        ((256 << 20) / bytes.max(1)).clamp(1, 4096)
    };
    let mut destinations = (0..cap)
        .map(|_| Ok(upload_tensor(rt, &zero)?))
        .collect::<Result<Vec<_>>>()?;
    if accum {
        s.copy_read_into(
            TensorRead::from_tensor(&inputs[seed]),
            TensorWrite::from_tensor(&mut destinations[0]),
        )?;
    }
    execute(
        &case.op,
        inputs,
        read(0),
        read(if inputs.len() > 1 { 1 } else { 0 }),
        TensorWrite::from_tensor(&mut destinations[0]),
        s,
        scalars,
        &config,
        dot_accumulation,
        plan,
    )?;
    let error = verify(rt, &destinations[0], expected, case)?;
    if settings.verify_only {
        return Ok((None, error));
    }
    let timing = sample_batches(settings, bytes, |count| {
        if accum {
            for destination in &mut destinations[..count] {
                s.copy_read_into(
                    TensorRead::from_tensor(&inputs[seed]),
                    TensorWrite::from_tensor(destination),
                )?;
            }
        }
        // Distinct destinations permit all move-only handles to exist before
        // the clock. The same allocations are reused by subsequent samples.
        let mut writes = destinations[..count]
            .iter_mut()
            .map(|t| {
                Some((
                    read(0),
                    read(if inputs.len() > 1 { 1 } else { 0 }),
                    TensorWrite::from_tensor(t),
                ))
            })
            .collect::<Vec<_>>();
        rt.synchronize()?;
        let start = Instant::now();
        for write in &mut writes {
            let (a, b, out) = write.take().unwrap();
            execute(
                &case.op,
                inputs,
                a,
                b,
                out,
                s,
                scalars,
                &config,
                dot_accumulation,
                plan,
            )?;
        }
        rt.synchronize()?;
        let elapsed = start.elapsed().as_nanos();
        black_box(&writes);
        Ok(elapsed)
    })?;
    let count = timing["operations_per_sample"].as_u64().unwrap() as usize;
    let error = error
        .max(verify(rt, &destinations[0], expected, case)?)
        .max(verify(rt, &destinations[count - 1], expected, case)?);
    Ok((Some(timing), error))
}
