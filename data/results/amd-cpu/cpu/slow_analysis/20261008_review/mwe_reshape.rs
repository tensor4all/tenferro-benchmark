//! Operation-only reproducers: `cargo run --release --no-default-features
//! --features system-mkl --example cpu_gap_mwe -- ifft 4`.
use num_complex::{Complex32, Complex64};
use serde_json::{json, Value};
use std::{error::Error, hint::black_box, time::Instant};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_fft::{FftExecutor, FftNorm};
use tenferro_linalg::{EagerSessionLinalgExt, TensorLinalgExt};
use tenferro_tensor::{BackendSessionHost, Tensor};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn measure<O>(
    bytes: usize,
    warmups: usize,
    samples: usize,
    target_ns: u128,
    mut op: impl FnMut() -> Result<O>,
) -> Result<Value> {
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

pub fn run(
    operation: &str,
    threads: usize,
    warmups: usize,
    samples: usize,
    target_ns: u128,
) -> Result<Value> {
    let mut backend = CpuBackend::with_threads(threads)?;
    let provider = format!("{:?}", backend.kind());
    let mut result = {
            let n = 33554432;
            let input =
                Tensor::from_vec_col_major(vec![n], (0..n).map(|i| (i % 17) as f64).collect())?;
            backend.with_backend_session(|s| -> Result<Value> {
                if operation == "cast" {
                    let output = s.cast(&input, tenferro_tensor::DType::F32)?;
                    assert!(output
                        .as_slice::<f32>()?
                        .iter()
                        .enumerate()
                        .all(|(i, &x)| x == (i % 17) as f32));
                    measure(n * 4, warmups, samples, target_ns, || {
                        Ok(s.cast(&input, tenferro_tensor::DType::F32)?)
                    })
                } else {
                    let output = s.reshape_read(
                        tenferro_tensor::TensorRead::from_tensor(&input),
                        &[8192, 4096],
                    )?;
                    assert!(output
                        .as_slice::<f64>()?
                        .iter()
                        .enumerate()
                        .all(|(i, &x)| x == (i % 17) as f64));
                    measure(n * 8, warmups, samples, target_ns, || {
                        Ok(s.reshape_read(
                            tenferro_tensor::TensorRead::from_tensor(&input),
                            &[8192, 4096],
                        )?)
                    })
                }
            })??
        };
    result["correctness_status"] = json!("passed");
    result["provider"] = json!(provider);
    result["threads_requested"] = json!(threads);
    result["scope"] = json!({"timer": [operation,"intrinsic output allocation"],
        "outside_timer": ["fixture construction","runtime/backend construction",
            "session entry","FFT plans and scratch priming","correctness validation",
            "retention vector allocation","output destruction"]});
    Ok(result)
}

#[allow(dead_code)]
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let op = args.get(1).map(String::as_str).unwrap_or("reshape");
    let threads = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1);
    println!("{}", run(op, threads, 3, 15, 2_000_000)?);
    Ok(())
}
