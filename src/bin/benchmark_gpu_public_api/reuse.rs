//! Output buffers and move-only write handles are prepared before every clock.
use super::{sample_batches, verify, Case, Result, Settings};
use num_complex::{Complex32, Complex64};
use serde_json::Value;
use std::{hint::black_box, time::Instant};
use tenferro_gpu::cuda::{upload_tensor, CudaRuntime};
use tenferro_tensor::{BackendSession, DType, Tensor, TensorRead, TensorWrite};

fn execute(
    op: &str,
    inputs: &[Tensor],
    a: TensorRead<'_>,
    b: TensorRead<'_>,
    out: TensorWrite<'_>,
    s: &mut dyn BackendSession,
) -> Result<()> {
    let x = &inputs[0];
    let y = inputs.get(1).unwrap_or(x);
    match op {
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
) -> Result<(Option<Value>, f64)> {
    let n = expected.shape().iter().product::<usize>();
    let shape = expected.shape().to_vec();
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
    execute(
        &case.op,
        inputs,
        TensorRead::from_tensor(&inputs[0]),
        TensorRead::from_tensor(inputs.get(1).unwrap_or(&inputs[0])),
        TensorWrite::from_tensor(&mut destinations[0]),
        s,
    )?;
    let error = verify(rt, &destinations[0], expected, case)?;
    if settings.verify_only {
        return Ok((None, error));
    }
    let timing = sample_batches(settings, bytes, |count| {
        // Distinct destinations permit all move-only handles to exist before
        // the clock. The same allocations are reused by subsequent samples.
        let mut writes = destinations[..count]
            .iter_mut()
            .map(|t| {
                Some((
                    TensorRead::from_tensor(&inputs[0]),
                    TensorRead::from_tensor(inputs.get(1).unwrap_or(&inputs[0])),
                    TensorWrite::from_tensor(t),
                ))
            })
            .collect::<Vec<_>>();
        rt.synchronize()?;
        let start = Instant::now();
        for write in &mut writes {
            let (a, b, out) = write.take().unwrap();
            execute(&case.op, inputs, a, b, out, s)?;
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
