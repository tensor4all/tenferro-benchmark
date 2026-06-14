//! MWE: eager SVD sum(s) backward panics while trace grad succeeds.
//!
//! Run from tenferro-benchmark (with extern/tenferro-rs checked out):
//!   RUST_BACKTRACE=1 cargo run --example svd_backward_panic_mwe --features cpu-faer

use tenferro_ad::{AdContext, EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_linalg::eager_tensor as eager_linalg;
use tenferro_linalg as tenferro_linalg_crate;
use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TracedTensor, TypedTensor};

fn well_conditioned_matrix(n: usize, seed: u64) -> Vec<f64> {
    let len = n * n;
    let mut data: Vec<f64> = (0..len)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed.wrapping_mul(1442695040888963407));
            ((x % 1024) as f64 - 512.0) / 512.0
        })
        .collect();
    for j in 0..n {
        for i in 0..n {
            data[i + n * j] *= 0.05;
        }
        data[j + n * j] += 1.0 + j as f64 / n.max(1) as f64;
    }
    data
}

fn tensor(shape: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape.to_vec(), data))
}

fn ad_context() -> AdContext {
    AdContext::builder()
        .with_core_rules()
        .with_extension_rules(
            tenferro_linalg_crate::ad_rules().expect("tenferro-linalg AD rules should register"),
        )
        .build()
        .expect("tenferro AD context should build")
}

fn eager_svd_sum_backward(n: usize) {
    let ctx = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad_context());
    let a = EagerTensor::requires_grad_in(tensor(&[n, n], well_conditioned_matrix(n, 10)), ctx);
    let (_, s, _) = eager_linalg::svd(&a).expect("svd forward");
    let loss = s.reduce_sum(&[0]).expect("reduce_sum");
    loss.backward().expect("backward should not panic");
    let _ = a.grad().expect("grad should be populated");
}

fn trace_svd_sum_grad(n: usize) {
    let a = TracedTensor::from_tensor_concrete_shape(tensor(&[n, n], well_conditioned_matrix(n, 10)));
    let (_, s, _) = tenferro_linalg_crate::svd(&a).expect("svd forward");
    let loss = s.reduce_sum(&[0]);
    let grad = ad_context()
        .grad(&loss, &a)
        .expect("trace grad should succeed");

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_many(&[&grad])
        .expect("compile grad graph");
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_linalg_crate::register_runtime)
        .expect("register linalg runtime");
    let out = executor.run_many(&program).expect("execute grad graph");
    assert_eq!(out.len(), 1);
}

fn main() {
    let n = 4;
    println!("trace grad sum(svd(A)) on {n}x{n} ...");
    trace_svd_sum_grad(n);
    println!("ok");

    println!("eager backward sum(svd(A)) on {n}x{n} ...");
    eager_svd_sum_backward(n);
    println!("ok");
}
