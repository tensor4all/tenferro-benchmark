## Observed

Materializing `BackendSession::reshape_read` of a borrowed contiguous f64 tensor (33,554,432 elements / 256 MiB) into a fresh owned `[8192,4096]` tensor has essentially no 1T→4T scaling. At 4T it is ~3.32x slower than PyTorch's equivalent allocation-returning materialization.

| Threads | Rust median ms | PyTorch median ms | Median paired ratio | Round ratio range | A/A ratio | Max CoV |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 69.561 | 62.364 | 1.114x | 1.102–1.127 | 0.9952 | 2.41% |
| 4 | 69.824 | 21.052 | 3.317x | 3.196–3.364 | 0.9934 | 3.54% |

This measures a fresh output copy in both implementations. It does **not** compare a materializing Rust operation to metadata-only `torch.reshape`, and it is not the `reshape_view` case from #1719. The input and output have identical logical values and shape. The PyTorch target view is built before timing (`x.reshape(4096,8192).T`) and `clone()` preserves its compact column-major layout, equivalent to the Rust result.

## Source observation

At tenferro-rs `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`, `reshape_read_with_pool` sends an owned input to `reshape` and then `typed_reshape`:

- [structural.rs:512](https://github.com/tensor4all/tenferro-rs/blob/5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102/crates/tenferro-cpu/src/structural.rs#L512)
- [typed_reshape:974](https://github.com/tensor4all/tenferro-rs/blob/5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102/crates/tenferro-cpu/src/structural.rs#L974)

`TypedTensor::from_vec_col_major(shape.to_vec(), tensor.host_data()?.to_vec())` performs the required fresh-owned duplication through a serial host copy. The adjacent TODO already proposes a parallel copy for large tensors and revisiting the engine-entry skip. This is a source observation, not an instruction-level attribution of every millisecond.

## Runnable MWE

Pinned tested benchmark source: `1b0e3aa`; tenferro-rs: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`.

[Complete Rust MWE](https://github.com/tensor4all/tenferro-benchmark/blob/1b0e3aa/examples/cpu_gap_mwe.rs), [matched PyTorch MWE](https://github.com/tensor4all/tenferro-benchmark/blob/1b0e3aa/scripts/cpu_gap_mwe_python.py).

```bash
git clone https://github.com/tensor4all/tenferro-benchmark.git bench-mwe
cd bench-mwe
git checkout 1b0e3aa
./scripts/setup_extern_deps.sh
git -C extern/tenferro-rs checkout 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash -lc '
  export CARGO_TARGET_DIR="$PWD/target/compatibility-mkl" CARGO_BUILD_JOBS=8
  export LD_LIBRARY_PATH="$MKLROOT/lib:/opt/intel/oneapi/compiler/2026.1/lib:${LD_LIBRARY_PATH:-}"
  source scripts/thread_env.sh
  configure_cpu_thread_env 4
  cargo build --release --no-default-features --features system-mkl --example cpu_gap_mwe
  source scripts/benchmark_host_idle.sh
  assert_benchmark_host_idle
  target/compatibility-mkl/release/examples/cpu_gap_mwe reshape 4
  .venv/bin/python scripts/cpu_gap_mwe_python.py reshape 4'
```

The emitted JSON contains `elapsed_ns` and `iterations`; take the median of `elapsed_ns / iterations`. Use 1T as the control. For same-build A/A followed by the balanced cross-implementation comparison, inside the devcontainer run:

```bash
.venv/bin/python scripts/confirm_cpu_gaps.py data/results/amd-cpu/cpu/slow_analysis/repro-reshape reshape
```

The complete operation-specific Rust program below runs in the benchmark repository's Cargo dependency environment. Inputs, backend/session creation, priming, correctness checks, retention containers and destruction are outside timing; only materialization and intrinsic output allocation are timed.

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
```

## Measurement record

AMD Ryzen 9 9955HX, Linux CPU devcontainer, no pinning/CUDA; system-mkl build (detected Blas provider), PyTorch 2.12.0+cpu MKL. This copy workload uses no BLAS math. 3 warmups +15 samples per process, memory-bounded batches toward 2 ms. Four balanced Rust/PyTorch paired rounds, preceded by four Rust/Rust A/A rounds, sequential processes and enabled idle-host guard. Predeclared threshold: >20% and >500 ns; A/A statistic within 10%, maximum sample CoV ≤20%.

[Raw A/A, paired samples and immutable declaration](https://github.com/tensor4all/tenferro-benchmark/tree/analysis/ryzen-cpu-slow-cases/data/results/amd-cpu/cpu/slow_analysis/20261008_reshape_matched).

## Existing ownership and requested investigation

Searched all 1,173 existing open/closed issues plus reshape/materialization searches. Related historical issues are #1001 (CPU kernel parallel wiring), #1480 (contiguous-read/transfer identity copies) and #1393 (arbitrary-stride view materialization); none explicitly tracks this borrowed-owned-input materializing reshape route and its serial duplicate. #1719's reshape case is metadata-only and must remain separate.

Please keep the fresh-owned-output contract, and compare a session-aware large-copy path for this exact route. Avoid changing reshape into a metadata-only API to satisfy the benchmark. The workload and matched PyTorch reference are being added to the existing `cpu/perf_issues` suite with this issue number.
