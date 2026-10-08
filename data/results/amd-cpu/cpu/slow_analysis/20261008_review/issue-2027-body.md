## Observed

A well-conditioned full-column-rank 768×384 f64 matrix with 16 RHS columns. The requested output is the least-squares solution; its analytical value is all ones. Rust uses EagerSessionLinalgExt::lstsq in a borrowed eager session. PyTorch uses torch.linalg.lstsq(driver="gels") (its full returned result is retained).

| Threads | Rust median ms | PyTorch median ms | Median paired ratio | Round ratio range | A/A ratio | Max sample CoV |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.007 | 9.759 | 1.948x | 1.931–1.965 | 0.9998 | 2.43% |
| 4 | 7.984 | 4.399 | 1.838x | 1.787–2.141 | 0.9960 | 8.33% |

These are same-host cross-implementation results, not historical revision regressions. The gap reproduces at both 1T and 4T. The benchmark requests unpivoted full-rank least squares on both sides; JAX SVD-based least squares is not used as the reference.

## Environment and timing

- AMD Ryzen 9 9955HX, Linux CPU devcontainer, no CPU-affinity pinning, no CUDA.
- tenferro-rs `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`; benchmark MWE source `672df86a45e5b9fb621eea8a07313495955b1885`.
- tenferro-rs `system-mkl`, detected provider `Blas`; PyTorch 2.12.0+cpu reports MKL. Vendor versions differ: tenferro oneMKL 2026.0.1, PyTorch's embedded MKL 2024.2. Provider family is matched; this does not isolate an implementation-only or kernel-only cause.
- All input generation, dtype/layout wrapping, backend/runtime construction, session entry, lazy initialization, correctness checks, output-retention containers and cleanup are outside timing. Inputs remain unchanged, outputs survive clock stop. Intrinsic output allocation is timed.
- 3 warmups, 15 measured samples per process. Memory-bounded calibration targets 2 ms per interval. Four balanced paired rounds (Rust/PyTorch, PyTorch/Rust, Rust/PyTorch, PyTorch/Rust), with all processes sequential. An independent same-build Rust/Rust A/A campaign precedes the comparison.
- Threshold declaration was saved before the PyTorch candidate samples: >20% and >500 ns, A/A statistic within 10%, each arm/sample CoV ≤20%. Numerical assertions verify the analytical solutions before sampling. Neither compared path uses AD.

## Runnable MWE

The complete one-operation Rust MWE is included below. It runs in the benchmark repository because that repository already declares and pins the public tenferro dependencies. The same code is available as the `lstsq` arm of [cpu_gap_mwe.rs](https://github.com/tensor4all/tenferro-benchmark/blob/672df86a45e5b9fb621eea8a07313495955b1885/examples/cpu_gap_mwe.rs); the matched PyTorch fixture and timer are in [cpu_gap_mwe_python.py](https://github.com/tensor4all/tenferro-benchmark/blob/672df86a45e5b9fb621eea8a07313495955b1885/scripts/cpu_gap_mwe_python.py).

```bash
git clone https://github.com/tensor4all/tenferro-benchmark.git bench-mwe
cd bench-mwe
git checkout 672df86a45e5b9fb621eea8a07313495955b1885
./scripts/setup_extern_deps.sh
git -C extern/tenferro-rs checkout 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash -lc '
  export CARGO_TARGET_DIR="$PWD/target/compatibility-mkl"
  export CARGO_BUILD_JOBS=8 TENFERRO_CPU_FEATURES=system-mkl
  export TENFERRO_CPU_BACKEND_KIND=blas BENCHMARK_TARGET_PROFILE=amd-cpu
  export LD_LIBRARY_PATH="$MKLROOT/lib:/opt/intel/oneapi/compiler/2026.1/lib:${LD_LIBRARY_PATH:-}"
  source scripts/thread_env.sh
  configure_cpu_thread_env 4
  cargo build --release --no-default-features --features system-mkl --example cpu_gap_mwe
  source scripts/benchmark_host_idle.sh
  assert_benchmark_host_idle
  target/compatibility-mkl/release/examples/cpu_gap_mwe lstsq 4
  .venv/bin/python scripts/cpu_gap_mwe_python.py lstsq 4'
```

Each command prints sample duration and operations per sample as JSON. Normalize `elapsed_ns / iterations`, then take the median. Use thread 1 as the control. For the A/A plus balanced comparison, execute inside the same devcontainer:

```bash
.venv/bin/python scripts/confirm_cpu_gaps.py data/results/amd-cpu/cpu/slow_analysis/repro-lstsq lstsq
```

The guard remains enabled; an existing declaration is never overwritten. Raw paired/A/A JSON and the predeclared confirmation policy are [here](https://github.com/tensor4all/tenferro-benchmark/tree/analysis/ryzen-cpu-slow-cases/data/results/amd-cpu/cpu/slow_analysis/20261008_issues).

```rust
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
    let op = args.get(1).map(String::as_str).unwrap_or("lstsq");
    let threads = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1);
    println!("{}", run(op, threads, 3, 15, 2_000_000)?);
    Ok(())
}
```

## Existing issue search and requested investigation

Searched all 1,173 existing open/closed issues and operation-specific GitHub searches on 2026-10-08. #1956/#1953 track linalg extraction/provider injection, not this measured primal lstsq cost. #1328 is linalg AD overhead; no differentiation occurs here.

Please attribute the remaining cost with the same borrowed-session workload before choosing an optimization. Preserve the current public numerical/output/ownership contracts. The workload and PyTorch reference are being added to the existing `cpu/perf_issues` suite with this issue number; no additional audit/registry or exhaustive baseline campaign is requested.
