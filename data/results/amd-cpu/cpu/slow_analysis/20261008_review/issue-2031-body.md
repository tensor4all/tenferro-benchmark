## Observed

Stable softmax over axis 0 of column-major f32 [64,64,8] attention scores, no mask and no AD. Rust calls s.softmax(&x,0); PyTorch uses native [8,64,64] and F.softmax(x,dim=-1). These are equivalent key/query/batch values and reduction axes. Every output is checked against a scalar f64 max-subtracted softmax oracle before sampling.

| Threads | Rust ms | PyTorch ms | Median paired ratio | Round range | A/A statistic | Max sample CoV |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.200 | 0.049 | 4.088x | 3.711–4.148 | 0.9877 | 5.16% |
| 4 | 0.200 | 0.025 | 7.905x | 7.665–9.482 | 1.0306 | 7.46% |

The original suite observation (~21x at 4T) is only a screening result; the operation-only confirmed ratio below is ~8x.

## Environment and timing

AMD Ryzen 9 9955HX, Linux CPU devcontainer, no affinity pinning, sequential 1T/4T processes, no CUDA. tenferro-rs `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`, Rust release `system-mkl` with detected Blas provider. PyTorch 2.12.0+cpu reports MKL. Versions differ: oneMKL 2026.0.1 versus PyTorch embedded MKL 2024.2, so this demonstrates an API-level same-host comparison, not an isolated kernel cause or revision regression.

Fixture generation, wrapping, runtime/backend creation, session entry, priming, numerical validation, output-retention containers and destruction are untimed. Intrinsic operation/output allocation is timed; outputs remain alive until the clock stops. Three warmups, fifteen samples, memory-bounded many-operation intervals (2 ms calibration target). Four balanced paired rounds (AB,BA,AB,BA) follow a same-build Rust/Rust A/A campaign. The idle guard remains enabled. The confirmation declaration predates the PyTorch samples: >20% and >500 ns, median A/A ratio within 10%, max sample CoV <=20%.

## Runnable MWE

The complete public-API Rust MWE is below. Place it in `examples/softmax_issue.rs` of tenferro-benchmark, whose Cargo manifest supplies the public dependencies. The maintained equivalent is the `softmax` arm of [cpu_gap_mwe.rs](https://github.com/tensor4all/tenferro-benchmark/blob/578d84e42aa1f745e00ac2a4416938112f0f6b48/examples/cpu_gap_mwe.rs), with [matched PyTorch fixture and batch timer](https://github.com/tensor4all/tenferro-benchmark/blob/578d84e42aa1f745e00ac2a4416938112f0f6b48/scripts/cpu_gap_mwe_python.py).

```bash
git clone https://github.com/tensor4all/tenferro-benchmark.git bench-mwe
cd bench-mwe
git checkout 578d84e42aa1f745e00ac2a4416938112f0f6b48
./scripts/setup_extern_deps.sh
git -C extern/tenferro-rs checkout 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash -lc '
  export CARGO_TARGET_DIR="$PWD/target/compatibility-mkl" CARGO_BUILD_JOBS=8
  export TENFERRO_CPU_FEATURES=system-mkl TENFERRO_CPU_BACKEND_KIND=blas
  export BENCHMARK_TARGET_PROFILE=amd-cpu
  export LD_LIBRARY_PATH="$MKLROOT/lib:/opt/intel/oneapi/compiler/2026.1/lib:${LD_LIBRARY_PATH:-}"
  cargo build --release --no-default-features --features system-mkl --example cpu_gap_mwe
  .venv/bin/python scripts/confirm_cpu_gaps.py data/results/amd-cpu/cpu/slow_analysis/repro-softmax softmax'
```

This collects both thread counts sequentially. A standalone arm prints JSON with `elapsed_ns` and `iterations`; normalize by iterations and take the median. For Rust use `target/compatibility-mkl/release/examples/cpu_gap_mwe softmax 4`; for PyTorch use `.venv/bin/python scripts/cpu_gap_mwe_python.py softmax 4` with `source scripts/thread_env.sh; configure_cpu_thread_env 4` and the idle guard before each timed process. [Raw A/A, paired samples and immutable declaration](https://github.com/tensor4all/tenferro-benchmark/tree/analysis/ryzen-cpu-slow-cases/data/results/amd-cpu/cpu/slow_analysis/20261008_activation_softmax) will be retained with the report.

```rust
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
```

## Existing coverage and related issues

#1976 requests the API and numerical contract. #2021 reports concrete-session broadcast materialization relative to eager, but this MWE uses the eager path and compares it with PyTorch; it does not attribute this gap to #2021. #2010 is an umbrella. Focused open/closed searches found no focused owner of this eager CPU/PyTorch gap.

The workload and PyTorch reference already exist as `softmax_f32_eager-single-call_len64_b8` in cpu/perf_issues. The new issue number will be recorded beside those existing cases when this issue opens. This does not claim that other activations, masked softmax or log_softmax have independently passed confirmation.
