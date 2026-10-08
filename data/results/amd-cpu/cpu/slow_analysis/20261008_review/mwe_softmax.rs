use serde_json::{json, Value};
use std::{error::Error, hint::black_box, time::Instant};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::Tensor;
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
fn measure<O>(
    bytes: usize,
    warmups: usize,
    samples: usize,
    target_ns: u128,
    mut op: impl FnMut() -> Result<O>,
) -> Result<Value> {
    if samples == 0 {
        return Ok(json!({"samples": [], "calibration": {
            "iterations": 0, "elapsed_ns": 0, "target_ns": target_ns}}));
    }
    // Explicit priming also happens when warmups == 0.
    for _ in 0..warmups.max(1) {
        black_box(op()?);
    }
    let cap = (512 * 1024 * 1024 / bytes.max(1)).clamp(1, 65536);
    let mut batch = |count| -> Result<u128> {
        let mut outputs = Vec::with_capacity(count);
        let start = Instant::now();
        for _ in 0..count {
            outputs.push(op()?);
        }
        let ns = start.elapsed().as_nanos();
        black_box(&outputs); // destruction is after clock stop
        Ok(ns)
    };
    let mut count = 1;
    let calibrated = loop {
        let ns = batch(count)?;
        if ns >= target_ns || count == cap {
            break ns;
        }
        count = (count * 2).min(cap);
    };
    let mut rows = Vec::with_capacity(samples);
    for sample_index in 0..samples {
        rows.push(json!({"sample_index": sample_index, "iterations": count,
            "elapsed_ns": batch(count)?}));
    }
    Ok(json!({"samples": rows, "calibration": {"iterations": count,
        "elapsed_ns": calibrated, "target_ns": target_ns}}))
}

fn main() -> Result<()> {
    let operation = "softmax";
    let threads = std::env::args().nth(1).unwrap_or_else(|| "4".into()).parse()?;
    let backend = CpuBackend::with_threads(threads)?;
    let (warmups, samples, target_ns) = (3, 15, 2000000);
    let result = match operation {
        "gelu" | "softmax" => {
            let (shape, seed) = if operation == "gelu" {
                (vec![1024, 64], 50u64)
            } else {
                (vec![64, 64, 8], 60u64)
            };
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n as u64)
                .map(|i| {
                    let v = (i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) % 2001) as f64;
                    ((v / 1000.0 - 1.0) * 0.5) as f32 * 8.0
                })
                .collect();
            let mut want: Vec<f64> = data
                .iter()
                .map(|&x| {
                    let x = x as f64;
                    0.5 * x * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
                })
                .collect();
            if operation == "softmax" {
                for (xs, ys) in data.chunks_exact(64).zip(want.chunks_exact_mut(64)) {
                    let max = xs
                        .iter()
                        .map(|&x| x as f64)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let sum: f64 = xs.iter().map(|&x| (x as f64 - max).exp()).sum();
                    for (&x, y) in xs.iter().zip(ys.iter_mut()) {
                        *y = (x as f64 - max).exp() / sum;
                    }
                }
            }
            let runtime = EagerRuntime::with_cpu_backend(backend)?;
            let x = EagerTensor::from_tensor_in(
                Tensor::from_vec_col_major(shape, data)?,
                runtime.clone(),
            )?;
            runtime.with_eager_session(|s| -> tenferro_ad::Result<Value> {
                let output = if operation == "gelu" {
                    s.gelu(&x)?
                } else {
                    s.softmax(&x, 0)?
                };
                let got = s.duplicate_value(&output)?;
                assert!(got
                    .as_slice::<f32>()?
                    .iter()
                    .zip(&want)
                    .all(|(&x, &y)| (x as f64 - y).abs() <= 1e-5 * y.abs().max(1.0)));
                measure(n * 4, warmups, samples, target_ns, || {
                    Ok(if operation == "gelu" {
                        s.gelu(&x)?
                    } else {
                        s.softmax(&x, 0)?
                    })
                })
                .map_err(|e| {
                    tenferro_ad::Error::runtime_state(
                        "cpu_gap_mwe",
                        tenferro_runtime::ErrorPhase::Execution,
                        e.to_string(),
                    )
                })
            })?
        }
        _ => unreachable!(),
    };
    println!("{}", result);
    Ok(())
}
