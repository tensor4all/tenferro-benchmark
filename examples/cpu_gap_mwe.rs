//! Operation-only reproducers: `cargo run --release --no-default-features
//! --features system-mkl --example cpu_gap_mwe -- ifft 4`.
use num_complex::Complex32;
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
    let mut result = match operation {
        "ifft" => {
            let n = 1048576;
            let input = Tensor::from_vec_col_major(vec![n], vec![Complex32::new(1.0, 0.0); n])?;
            let mut executor = FftExecutor::default();
            backend.with_backend_session(|s| {
                with_cpu_exec_session(s, |s| -> Result<Value> {
                    let output = executor.ifft(&input, None, -1, FftNorm::Backward, s)?;
                    let values = output.as_slice::<Complex32>()?;
                    for (i, value) in values.iter().enumerate() {
                        let want = if i == 0 { 1.0 } else { 0.0 };
                        assert!((value.re - want).abs() < 1e-5 && value.im.abs() < 1e-5);
                    }
                    measure(n * 8, warmups, samples, target_ns, || {
                        Ok(executor.ifft(&input, None, -1, FftNorm::Backward, s)?)
                    })
                })
                .expect("CPU session")
            })??
        }
        "diagonal" => {
            let n = 8388608;
            let mut data = vec![0.0; n * 4];
            data[..n].fill(1.0);
            data[3 * n..].fill(2.0);
            let input = Tensor::from_vec_col_major(vec![n, 2, 2], data)?;
            backend.with_backend_session(|s| -> Result<Value> {
                let output = s.extract_diagonal(&input, 1, 2)?;
                let got = output.as_slice::<f64>()?;
                assert!(got[..n].iter().all(|&x| x == 1.0));
                assert!(got[n..].iter().all(|&x| x == 2.0));
                measure(n * 16, warmups, samples, target_ns, || {
                    Ok(s.extract_diagonal(&input, 1, 2)?)
                })
            })??
        }
        "triangular" => {
            let n = 4096;
            let nrhs = 64;
            let mut data = vec![0.0; n * n];
            let mut sums = vec![1.0; n];
            for j in 0..n {
                data[j + n * j] = 1.0;
                for i in j + 1..n {
                    let v = (((i * 17 + j * 13) % 19) as f64 - 9.0) / (n as f64 * 100.0);
                    data[i + n * j] = v;
                    sums[i] += v;
                }
            }
            let a = Tensor::from_vec_col_major(vec![n, n], data)?;
            let rhs = Tensor::from_vec_col_major(vec![n, nrhs], sums.repeat(nrhs))?;
            backend.with_backend_session(|s| {
                with_cpu_exec_session(s, |s| -> Result<Value> {
                    let output = a.triangular_solve(&rhs, true, true, false, false, s)?;
                    assert!(output
                        .as_slice::<f64>()?
                        .iter()
                        .all(|x| (x - 1.0).abs() < 1e-10));
                    measure(n * nrhs * 8, warmups, samples, target_ns, || {
                        Ok(a.triangular_solve(&rhs, true, true, false, false, s)?)
                    })
                })
                .expect("CPU session")
            })??
        }
        "lstsq" => {
            let (m, n, nrhs) = (768, 384, 16);
            let mut data = vec![0.0; m * n];
            let mut sums = vec![0.0; m];
            for j in 0..n {
                for i in 0..m {
                    let v = if i == j {
                        2.0
                    } else {
                        (((i * 17 + j * 13) % 19) as f64 - 9.0) / (m as f64 * 10.0)
                    };
                    data[i + m * j] = v;
                    sums[i] += v;
                }
            }
            let runtime = EagerRuntime::with_cpu_backend(backend)?;
            let a = EagerTensor::from_tensor_in(
                Tensor::from_vec_col_major(vec![m, n], data)?,
                runtime.clone(),
            )?;
            let rhs = EagerTensor::from_tensor_in(
                Tensor::from_vec_col_major(vec![m, nrhs], sums.repeat(nrhs))?,
                runtime.clone(),
            )?;
            runtime.with_eager_session(|s| -> tenferro_ad::Result<Value> {
                let output = s.lstsq(&a, &rhs)?;
                let got = s.duplicate_value(&output)?;
                assert!(got
                    .as_slice::<f64>()?
                    .iter()
                    .all(|x| (x - 1.0).abs() < 1e-10));
                measure(n * nrhs * 8, warmups, samples, target_ns, || {
                    Ok(s.lstsq(&a, &rhs)?)
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
        _ => return Err(format!("unknown operation {operation}").into()),
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
    let op = args.get(1).map(String::as_str).unwrap_or("ifft");
    let threads = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1);
    println!("{}", run(op, threads, 3, 15, 2_000_000)?);
    Ok(())
}
