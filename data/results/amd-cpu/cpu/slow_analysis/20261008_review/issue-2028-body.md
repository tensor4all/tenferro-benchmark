## Observed

A dense, well-conditioned lower-triangular 4096×4096 f64 matrix with 64 RHS columns; left-side, non-transposed, non-unit-diagonal solve. The analytical solution is all ones. Both APIs return newly allocated solutions. Rust uses TensorLinalgExt::triangular_solve with a borrowed CPU execution session; PyTorch uses torch.linalg.solve_triangular(upper=False).

| Threads | Rust median ms | PyTorch median ms | Median paired ratio | Round ratio range | A/A ratio | Max sample CoV |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.913 | 35.722 | 1.031x | 0.982–1.059 | 1.0017 | 0.86% |
| 4 | 15.384 | 12.554 | 1.249x | 1.190–1.262 | 0.9877 | 2.13% |

These are same-host cross-implementation results, not historical revision regressions. The 1T triangular case is approximately at parity; the reported triangular gap is specific to 4T.

## Environment and timing

- AMD Ryzen 9 9955HX, Linux CPU devcontainer, no CPU-affinity pinning, no CUDA.
- tenferro-rs `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`; benchmark MWE source `672df86a45e5b9fb621eea8a07313495955b1885`.
- tenferro-rs `system-mkl`, detected provider `Blas`; PyTorch 2.12.0+cpu reports MKL. Vendor versions differ: tenferro oneMKL 2026.0.1, PyTorch's embedded MKL 2024.2. Provider family is matched; this does not isolate an implementation-only or kernel-only cause.
- All input generation, dtype/layout wrapping, backend/runtime construction, session entry, lazy initialization, correctness checks, output-retention containers and cleanup are outside timing. Inputs remain unchanged, outputs survive clock stop. Intrinsic output allocation is timed.
- 3 warmups, 15 measured samples per process. Memory-bounded calibration targets 2 ms per interval. Four balanced paired rounds (Rust/PyTorch, PyTorch/Rust, Rust/PyTorch, PyTorch/Rust), with all processes sequential. An independent same-build Rust/Rust A/A campaign precedes the comparison.
- Threshold declaration was saved before the PyTorch candidate samples: >20% and >500 ns, A/A statistic within 10%, each arm/sample CoV ≤20%. Numerical assertions verify the analytical solutions before sampling. Neither compared path uses AD.

## Runnable MWE

The complete one-operation Rust MWE is included below. It runs in the benchmark repository because that repository already declares and pins the public tenferro dependencies. The same code is available as the `triangular` arm of [cpu_gap_mwe.rs](https://github.com/tensor4all/tenferro-benchmark/blob/672df86a45e5b9fb621eea8a07313495955b1885/examples/cpu_gap_mwe.rs); the matched PyTorch fixture and timer are in [cpu_gap_mwe_python.py](https://github.com/tensor4all/tenferro-benchmark/blob/672df86a45e5b9fb621eea8a07313495955b1885/scripts/cpu_gap_mwe_python.py).

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
  target/compatibility-mkl/release/examples/cpu_gap_mwe triangular 4
  .venv/bin/python scripts/cpu_gap_mwe_python.py triangular 4'
```

Each command prints sample duration and operations per sample as JSON. Normalize `elapsed_ns / iterations`, then take the median. Use thread 1 as the control. For the A/A plus balanced comparison, execute inside the same devcontainer:

```bash
.venv/bin/python scripts/confirm_cpu_gaps.py data/results/amd-cpu/cpu/slow_analysis/repro-triangular triangular
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
    let op = args.get(1).map(String::as_str).unwrap_or("triangular");
    let threads = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1);
    println!("{}", run(op, threads, 3, 15, 2_000_000)?);
    Ok(())
}
```

## Existing issue search and requested investigation

Searched all 1,173 existing open/closed issues and operation-specific GitHub searches on 2026-10-08. #2007 is batched API support and #1878/#1884 are batched per-matrix/scaling costs. This reproducer is a single rank-2 4096×4096 system. #1328 concerns AD, which is absent here.

Please attribute the remaining cost with the same borrowed-session workload before choosing an optimization. Preserve the current public numerical/output/ownership contracts. The workload and PyTorch reference are being added to the existing `cpu/perf_issues` suite with this issue number; no additional audit/registry or exhaustive baseline campaign is requested.
