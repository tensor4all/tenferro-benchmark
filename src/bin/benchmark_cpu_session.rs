//! Many independent small matrices in one entered CPU session.
use std::{error::Error, hint::black_box, time::Instant};
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_linalg::TensorLinalgExt;
use tenferro_runtime::BackendSessionHost;
use tenferro_tensor::{DotGeneralConfig, Tensor};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn fixture(n: usize, k: usize) -> Result<(Tensor, Tensor, Vec<f64>, Vec<f64>)> {
    let a: Vec<_> = (0..n * n)
        .map(|i| {
            let (r, c) = (i % n, i / n);
            if r == c {
                n as f64 + 1.0 + (k % 17) as f64 * 0.01
            } else {
                ((r + c + k) % 7) as f64 * 0.01
            }
        })
        .collect();
    let b: Vec<_> = (0..n * n)
        .map(|i| ((i + k) % 13) as f64 * 0.02 - 0.1)
        .collect();
    let product = (0..n * n)
        .map(|i| (0..n).map(|j| a[i % n + n * j] * b[j + n * (i / n)]).sum())
        .collect();
    Ok((
        Tensor::from_vec_col_major(vec![n, n], a)?,
        Tensor::from_vec_col_major(vec![n, n], b.clone())?,
        product,
        b,
    ))
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let get = |name: &str, default: &str| {
        args.windows(2)
            .find(|p| p[0] == name)
            .map(|p| p[1].clone())
            .unwrap_or(default.into())
    };
    let n: usize = get("--n", "2").parse()?;
    let count: usize = get("--count", "1024").parse()?;
    let samples: usize = get("--samples", "15").parse()?;
    let op = get("--op", "matmul");
    let provider = get("--provider", "blas");
    if n == 0 || count == 0 || samples == 0 || !["matmul", "solve"].contains(&op.as_str()) {
        return Err("invalid case".into());
    }
    let kind = match provider.as_str() {
        "blas" => CpuBackendKind::Blas,
        "faer" => CpuBackendKind::Faer,
        _ => return Err("invalid provider".into()),
    };
    let mut backend = CpuBackend::with_kind(kind)?;
    let execution_info = backend.execution_info();
    let execution_mode = format!("{:?}", execution_info.execution_mode());
    let worker_count = execution_info.worker_count();
    let fixtures = (0..count)
        .map(|k| fixture(n, k))
        .collect::<Result<Vec<_>>>()?;
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    // Output slots and every input are prepared before entering the session.
    let mut outputs = Vec::with_capacity(count);
    let mut elapsed_ns = Vec::with_capacity(samples);
    backend.with_backend_session(|session| -> Result<()> {
        for sample in 0..samples + 3 {
            outputs.clear(); // Destruction/reset is outside the clock.
            let start = Instant::now();
            for (a, b, _, _) in &fixtures {
                outputs.push(if op == "matmul" {
                    session.dot_general(a, b, &config)?
                } else {
                    a.solve(b, session)?
                });
            }
            let elapsed = start.elapsed().as_nanos();
            black_box(&outputs);
            if sample >= 3 {
                elapsed_ns.push(elapsed);
            }
            // Check all outputs, including every warmup, after timer stop.
            for (output, (a, _, product, b)) in outputs.iter().zip(&fixtures) {
                let out = output.as_slice::<f64>()?;
                let av = a.as_slice::<f64>()?;
                for i in 0..n * n {
                    let (actual, expected) = if op == "matmul" {
                        (out[i], product[i])
                    } else {
                        (
                            (0..n)
                                .map(|j| av[i % n + n * j] * out[j + n * (i / n)])
                                .sum(),
                            b[i],
                        )
                    };
                    if !actual.is_finite()
                        || (actual - expected).abs() > 1e-11 * (1.0 + expected.abs())
                    {
                        return Err(
                            format!("incorrect {op} output {i}: {actual} != {expected}").into()
                        );
                    }
                }
            }
        }
        Ok(())
    })?;
    println!(
        "{}",
        serde_json::json!({"operation":op,"n":n,"operations_per_sample":count,"provider":provider,"route":"shared-session","session_count":1,"execution_mode":execution_mode,"worker_count":worker_count,"warmups":3,"samples_ns":elapsed_ns,"correctness":"passed"})
    );
    Ok(())
}
