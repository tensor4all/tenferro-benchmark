//! Focused CPU matmul diagnostic for tensor4all/tenferro-rs#975.
//!
//! This intentionally lives outside the main benchmark runner. It compares the
//! same square f64 GEMM-shaped workload across raw Accelerate, direct tenferro
//! `dot_general`, traced execution, and eager einsum in one process.

use std::hint::black_box;
use std::time::{Duration, Instant};

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_einsum::{
    eager_tensor, ContractionTree, EinsumOptimize, EinsumSubscripts, Subscripts,
};
use tenferro_runtime::{CompilerOptions, GraphCompiler, GraphExecutor, OptimizerConfig, TensorRead, TracedTensor};
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorBuffer, TensorDot};

const DEFAULT_N: usize = 1024;
const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_RUNS: usize = 15;

const CBLAS_COL_MAJOR: i32 = 102;
const CBLAS_NO_TRANS: i32 = 111;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_dgemm(
        layout: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
}

fn positive_env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn bool_env(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn matmul_ji_kj_ki_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![1],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

fn deterministic_matrix_data(n: usize, salt: usize) -> Vec<f64> {
    let len = n.checked_mul(n).expect("square matrix element count");
    (0..len)
        .map(|idx| {
            let value = ((idx.wrapping_mul(1_103).wrapping_add(salt * 97)) % 1009) as f64;
            (value + 1.0) / 1009.0
        })
        .collect()
}

fn tensor_from_data(n: usize, data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(vec![n, n], data)
}

fn duration_stats(mut durations: Vec<Duration>) -> (Duration, Duration) {
    durations.sort_unstable();
    let median = durations[durations.len() / 2];
    let q1 = durations[durations.len() / 4];
    let q3 = durations[3 * durations.len() / 4];
    (median, q3.saturating_sub(q1))
}

fn measure(
    name: &str,
    warmups: usize,
    runs: usize,
    mut f: impl FnMut() -> Result<(), String>,
) -> Result<(), String> {
    for _ in 0..warmups {
        f()?;
    }

    let mut durations = Vec::with_capacity(runs);
    for _ in 0..runs {
        let started = Instant::now();
        f()?;
        durations.push(started.elapsed());
    }
    let (median, iqr) = duration_stats(durations);
    println!(
        "{name:<34} median_ms={:>10.3} iqr_ms={:>10.3}",
        median.as_secs_f64() * 1e3,
        iqr.as_secs_f64() * 1e3
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn raw_accelerate_dgemm(n: usize, a: &[f64], b: &[f64], c: &mut [f64]) -> Result<(), String> {
    let n_i32 = i32::try_from(n).map_err(|_| format!("n={n} exceeds cblas i32 range"))?;
    // SAFETY: a, b, and c are dense n x n column-major matrices. c is uniquely
    // mutable for the duration of the call and does not alias a or b.
    unsafe {
        cblas_dgemm(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            n_i32,
            n_i32,
            n_i32,
            1.0,
            a.as_ptr(),
            n_i32,
            b.as_ptr(),
            n_i32,
            0.0,
            c.as_mut_ptr(),
            n_i32,
        );
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn raw_accelerate_dgemm(_n: usize, _a: &[f64], _b: &[f64], _c: &mut [f64]) -> Result<(), String> {
    Err("raw Accelerate is only available on macOS".to_string())
}

fn compile_trace_program(
    n: usize,
    dot_decomposer: bool,
) -> Result<(tenferro_runtime::GraphProgram, Vec<TracedTensor>), String> {
    let mut compiler = GraphCompiler::with_compiler_options(CompilerOptions {
        optimizer: OptimizerConfig {
            dot_decomposer,
            ..OptimizerConfig::default()
        },
    });
    let lhs = TracedTensor::input_concrete_shape(tenferro_runtime::DType::F64, &[n, n]);
    let rhs = TracedTensor::input_concrete_shape(tenferro_runtime::DType::F64, &[n, n]);
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[2, 0]], &[2, 1]);
    let subs = Subscripts::from(&subscripts);
    let tree = ContractionTree::from_pairs(&subs, &[&[n, n][..], &[n, n][..]], &[(0, 1)])
        .map_err(|err| err.to_string())?;
    let out = tenferro_einsum::einsum_subscripts_with(
        &mut compiler,
        &[&lhs, &rhs],
        &subscripts,
        EinsumOptimize::Tree(tree),
    )
    .map_err(|err| err.to_string())?;
    let program = compiler
        .compile_with_input_specs(
            &out,
            &[
                (&lhs, tenferro_runtime::DType::F64, &[n, n][..]),
                (&rhs, tenferro_runtime::DType::F64, &[n, n][..]),
            ],
        )
        .map_err(|err| err.to_string())?;
    Ok((program, vec![lhs, rhs]))
}

fn main() -> Result<(), String> {
    let n = positive_env_usize("MATMUL_DIAG_N", DEFAULT_N);
    let warmups = positive_env_usize("MATMUL_DIAG_WARMUPS", DEFAULT_WARMUPS);
    let runs = positive_env_usize("MATMUL_DIAG_RUNS", DEFAULT_RUNS);
    let dot_decomposer = bool_env("TENFERRO_OPT_DOT_DECOMPOSER");

    println!("cpu matmul diagnostic for tenferro-rs#975");
    println!("n={n} warmups={warmups} runs={runs}");
    println!(
        "threads: OMP_NUM_THREADS={} RAYON_NUM_THREADS={} VECLIB_MAXIMUM_THREADS={} VECLIB_NUM_THREADS={}",
        std::env::var("OMP_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
        std::env::var("VECLIB_MAXIMUM_THREADS").unwrap_or_else(|_| "unset".into()),
        std::env::var("VECLIB_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
    );
    println!("tenferro backend kind=blas dot_decomposer={dot_decomposer}");
    println!("einsum labels: ji,kj->ki");

    let lhs_data = deterministic_matrix_data(n, 1);
    let rhs_data = deterministic_matrix_data(n, 2);
    let lhs = tensor_from_data(n, lhs_data.clone());
    let rhs = tensor_from_data(n, rhs_data.clone());
    let config = matmul_ji_kj_ki_config();

    if cfg!(target_os = "macos") {
        let mut c = vec![0.0_f64; n * n];
        measure("raw_accelerate_dgemm", warmups, runs, || {
            raw_accelerate_dgemm(n, &rhs_data, &lhs_data, &mut c)?;
            black_box(&c);
            Ok(())
        })?;
    } else {
        println!(
            "{:<34} skipped: raw Accelerate is only available on macOS",
            "raw_accelerate_dgemm"
        );
    }

    measure("output_zero_alloc", warmups, runs, || {
        let c = vec![0.0_f64; n * n];
        black_box(c);
        Ok(())
    })?;

    let mut direct_backend = CpuBackend::with_kind(CpuBackendKind::Blas)
        .map_err(|err| format!("failed to create BLAS backend: {err}"))?;
    measure("tenferro_blas_dot_general", warmups, runs, || {
        let out = direct_backend
            .dot_general(&lhs, &rhs, &config)
            .map_err(|err| err.to_string())?;
        black_box(&out);
        direct_backend.reclaim_buffer(out);
        Ok(())
    })?;

    let mut read_backend = CpuBackend::with_kind(CpuBackendKind::Blas)
        .map_err(|err| format!("failed to create BLAS backend: {err}"))?;
    measure("tenferro_blas_dot_general_read", warmups, runs, || {
        let out = read_backend
            .dot_general_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
            )
            .map_err(|err| err.to_string())?;
        black_box(&out);
        read_backend.reclaim_buffer(out);
        Ok(())
    })?;

    let (program, traced_inputs) = compile_trace_program(n, dot_decomposer)?;
    let bindings = vec![
        (&traced_inputs[0], TensorRead::from_tensor(&lhs)),
        (&traced_inputs[1], TensorRead::from_tensor(&rhs)),
    ];
    let mut executor = GraphExecutor::new(
        CpuBackend::with_kind(CpuBackendKind::Blas)
            .map_err(|err| format!("failed to create BLAS backend: {err}"))?,
    );
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .map_err(|err| err.to_string())?;
    measure("tenferro_trace_input_reads", warmups, runs, || {
        let outputs = executor
            .run_many_with_input_reads(&program, &bindings)
            .map_err(|err| err.to_string())?;
        black_box(&outputs);
        executor.reclaim_outputs(outputs);
        Ok(())
    })?;

    let eager_ctx = EagerRuntime::with_cpu_backend(
        CpuBackend::with_kind(CpuBackendKind::Blas)
            .map_err(|err| format!("failed to create BLAS backend: {err}"))?,
    );
    let eager_lhs = EagerTensor::from_tensor_in(lhs.clone(), eager_ctx.clone());
    let eager_rhs = EagerTensor::from_tensor_in(rhs.clone(), eager_ctx);
    measure("tenferro_eager_einsum", warmups, runs, || {
        let out = eager_tensor::einsum_subscripts(
            &[&eager_lhs, &eager_rhs],
            &EinsumSubscripts::new(&[&[0, 1], &[2, 0]], &[2, 1]),
        )
        .map_err(|err| err.to_string())?;
        black_box(&out);
        Ok(())
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_config_matches_colmajor_einsum_labels() {
        let config = matmul_ji_kj_ki_config();

        assert_eq!(config.lhs_contracting_dims, vec![0]);
        assert_eq!(config.rhs_contracting_dims, vec![1]);
        assert!(config.lhs_batch_dims.is_empty());
        assert!(config.rhs_batch_dims.is_empty());
    }

    #[test]
    fn positive_env_usize_rejects_zero_and_invalid_values() {
        std::env::set_var("TENFERRO_TEST_POSITIVE_ENV_USIZE", "0");
        assert_eq!(positive_env_usize("TENFERRO_TEST_POSITIVE_ENV_USIZE", 7), 7);

        std::env::set_var("TENFERRO_TEST_POSITIVE_ENV_USIZE", "13");
        assert_eq!(
            positive_env_usize("TENFERRO_TEST_POSITIVE_ENV_USIZE", 7),
            13
        );

        std::env::set_var("TENFERRO_TEST_POSITIVE_ENV_USIZE", "not-a-number");
        assert_eq!(positive_env_usize("TENFERRO_TEST_POSITIVE_ENV_USIZE", 7), 7);

        std::env::remove_var("TENFERRO_TEST_POSITIVE_ENV_USIZE");
    }
}
