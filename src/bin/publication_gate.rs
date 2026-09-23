//! Publication-gate microbenchmarks for tenferro-rs.
//!
//! This runner covers the focused benchmark matrix from tensor4all/tenferro-rs#862:
//! small matrix latency, large matrix throughput, and batched small matrices.

use std::env;
use std::hint::black_box;
use std::io::Write;
use std::panic;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use tenferro_ad::{AdContext, EagerRuntime, EagerTensor};
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_einsum::{EagerEinsumExt, TraceContextEinsumExt};
use tenferro_einsum_benchmark::thread_enforcement::{
    enforce_thread_request, verify_backend_threads,
};
use tenferro_linalg::{EagerTensorLinalgExt, TracedTensorLinalgExt};
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{
    DotGeneralConfig, Error, ErrorPhase, GraphCompiler, Runtime, Tensor, TraceContext, TracedTensor,
};
use tenferro_tensor::TypedTensor;

const DEFAULT_WARMUPS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Profile {
    Quick,
    Full,
}

impl Profile {
    fn from_env() -> Self {
        match env::var("PUBLICATION_GATE_PROFILE")
            .unwrap_or_else(|_| "quick".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "full" => Self::Full,
            _ => Self::Quick,
        }
    }

    fn runs(self) -> usize {
        env::var("BENCH_RUNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(match self {
                Self::Quick => 7,
                Self::Full => 15,
            })
    }

    fn warmups(self) -> usize {
        env::var("BENCH_WARMUPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_WARMUPS)
    }
}

fn large_linalg_jvp_vjp_sizes(profile: Profile) -> &'static [usize] {
    match profile {
        // O(n^3) linalg AD: 256x256 is single-digit ms; 512x512 reaches tens of ms.
        Profile::Quick => &[256, 512],
        Profile::Full => &[256, 512, 1024],
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SuiteFilter {
    All,
    Small,
    Large,
    Batched,
}

fn eager_from_tensor_in(tensor: Tensor, ctx: Arc<EagerRuntime>) -> EagerTensor {
    EagerTensor::from_tensor_in(tensor, ctx).expect("benchmark tensor should be valid")
}

fn eager_requires_grad_in(tensor: Tensor, ctx: Arc<EagerRuntime>) -> EagerTensor {
    EagerTensor::requires_grad_in(tensor, ctx).expect("benchmark tensor should be valid")
}

mod eager_einsum_tensor {
    use super::*;
    pub fn einsum(
        inputs: &[&EagerTensor],
        _subscripts: &str,
    ) -> Result<EagerTensor, tenferro_ad::Error> {
        static SUBS: std::sync::OnceLock<tenferro_einsum::EinsumSubscripts> =
            std::sync::OnceLock::new();
        let subs = SUBS
            .get_or_init(|| tenferro_einsum::EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]));
        inputs
            .einsum_subscripts(subs)
            .map_err(|e| runtime_einsum_error(e, ErrorPhase::Execution))
    }
}

fn runtime_einsum_error(error: tenferro_einsum::Error, phase: ErrorPhase) -> Error {
    let kind = error.kind();
    Error::extension(
        "einsum",
        phase,
        tenferro_einsum::EINSUM_EXTENSION_FAMILY_ID,
        kind,
        error,
    )
}

mod eager_linalg_tensor {
    use super::*;

    pub fn svd(
        a: &EagerTensor,
    ) -> tenferro_ad::error::Result<(EagerTensor, EagerTensor, EagerTensor)> {
        a.svd()
    }

    pub fn qr(a: &EagerTensor) -> tenferro_ad::error::Result<(EagerTensor, EagerTensor)> {
        a.qr()
    }

    pub fn eigh(a: &EagerTensor) -> tenferro_ad::error::Result<(EagerTensor, EagerTensor)> {
        a.eigh()
    }

    pub fn solve(a: &EagerTensor, b: &EagerTensor) -> tenferro_ad::error::Result<EagerTensor> {
        a.solve(b)
    }
}

mod traced_tensor {
    use super::*;

    pub fn matmul(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
        lhs.matmul(rhs)
            .expect("benchmark matmul shapes should be valid")
    }
}

impl SuiteFilter {
    fn from_env() -> Self {
        match env::var("PUBLICATION_GATE_SUITE")
            .unwrap_or_else(|_| "all".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "small" => Self::Small,
            "large" => Self::Large,
            "batched" => Self::Batched,
            _ => Self::All,
        }
    }

    fn includes(self, suite: &str) -> bool {
        matches!(self, Self::All)
            || matches!(
                (self, suite),
                (Self::Small, "small") | (Self::Large, "large") | (Self::Batched, "batched")
            )
    }
}

struct BenchConfig {
    profile: Profile,
    suite: SuiteFilter,
    runs: usize,
    warmups: usize,
    backend: &'static str,
    include_eager: bool,
    include_trace: bool,
}

impl BenchConfig {
    fn from_env() -> Self {
        let profile = Profile::from_env();
        Self {
            profile,
            suite: SuiteFilter::from_env(),
            runs: profile.runs(),
            warmups: profile.warmups(),
            backend: backend_name(),
            include_eager: tenferro_mode_includes("eager"),
            include_trace: tenferro_mode_includes("trace"),
        }
    }
}

#[derive(Debug)]
struct Row {
    suite: &'static str,
    op: &'static str,
    phase: &'static str,
    dtype: &'static str,
    shape: String,
    backend: &'static str,
    median_ms: Option<f64>,
    iqr_ms: Option<f64>,
    status: String,
}

fn backend_name() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "system-openblas") {
        "system-openblas"
    } else if cfg!(feature = "system-accelerate") {
        "system-accelerate"
    } else if cfg!(feature = "system-mkl") {
        "system-mkl"
    } else {
        "cpu-faer"
    }
}

fn benchmark_name(op: &str, phase: &str) -> String {
    if phase.is_empty() || phase == "primal" {
        op.to_string()
    } else {
        format!("{op}_{phase}")
    }
}

fn benchmark_filter_matches(op: &str, phase: &str) -> bool {
    let Ok(filter) = env::var("CPU_OPS_BENCHMARK_FILTER") else {
        return true;
    };
    let filter = filter.trim();
    if filter.is_empty() {
        return true;
    }
    let name = benchmark_name(op, phase);
    filter
        .split(',')
        .map(str::trim)
        .any(|item| item == op || item == name)
}

fn filtered_row(
    suite: &'static str,
    op: &'static str,
    phase: &'static str,
    dtype: &'static str,
    shape: &str,
) -> Row {
    Row {
        suite,
        op,
        phase,
        dtype,
        shape: shape.to_string(),
        backend: "filtered",
        median_ms: None,
        iqr_ms: None,
        status: "filtered".to_string(),
    }
}

fn tenferro_mode_includes(mode: &str) -> bool {
    match env::var("PUBLICATION_GATE_TENFERRO_MODE")
        .or_else(|_| env::var("PUBLICATION_GATE_TENFERRO_MODES"))
        .unwrap_or_else(|_| "both".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "both" | "all" => true,
        selected => selected.split(',').any(|item| item.trim() == mode),
    }
}

fn requested_threads() -> usize {
    static REQUESTED: OnceLock<usize> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        env::var("RAYON_NUM_THREADS")
            .unwrap_or_else(|_| "1".into())
            .trim()
            .parse()
            .expect("RAYON_NUM_THREADS must be a positive integer")
    })
}

/// Fail before any timing when the thread environment disagrees with the
/// requested count or when the shared backend or its scope runs a different
/// number of Rayon workers.
fn enforce_threads_or_exit() {
    let requested = requested_threads();
    let check = enforce_thread_request(requested)
        .and_then(|()| {
            verify_backend_threads("CpuBackend", cpu_backend().num_threads(), requested)
        })
        .and_then(|()| {
            let scope_threads = cpu_backend()
                .with_execution_scope(rayon::current_num_threads)
                .map_err(|error| format!("enter CpuBackend execution scope: {error}"))?;
            verify_backend_threads("CpuBackend execution scope", scope_threads, requested)?;
            eprintln!(
                "publication_gate: requested_threads={requested} backend_threads={} scope_rayon_threads={scope_threads} global_rayon_threads={}",
                cpu_backend().num_threads(),
                rayon::current_num_threads(),
            );
            Ok(())
        });
    if let Err(error) = check {
        eprintln!("publication_gate: thread enforcement failed: {error}");
        std::process::exit(2);
    }
}

fn main() {
    let config = BenchConfig::from_env();
    enforce_threads_or_exit();
    println!("suite,op,phase,dtype,backend,profile,shape,warmups,runs,median_ms,iqr_ms,status");

    let rows = cpu_backend()
        .with_execution_scope(|| {
            record_cpu_runtime().expect("record effective CPU execution settings");
            run_all(&config)
        })
        .expect("enter shared CPU execution scope before benchmarking");
    for row in rows {
        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            row.suite,
            row.op,
            row.phase,
            row.dtype,
            row.backend,
            match config.profile {
                Profile::Quick => "quick",
                Profile::Full => "full",
            },
            csv_escape(&row.shape),
            config.warmups,
            config.runs,
            row.median_ms.map(|v| format!("{v:.6}")).unwrap_or_default(),
            row.iqr_ms.map(|v| format!("{v:.6}")).unwrap_or_default(),
            csv_escape(&row.status),
        );
    }
}

fn run_all(config: &BenchConfig) -> Vec<Row> {
    let mut rows = Vec::new();
    // Initialize the shared runtime before any timed samples, including zero-warmup runs.
    if config.include_eager {
        let _ = cpu_ctx();
    }
    if config.include_eager && config.suite.includes("small") {
        run_small_latency(config, &mut rows);
    }
    if config.include_eager && config.suite.includes("large") {
        run_large_throughput(config, &mut rows);
    }
    if config.include_eager && config.suite.includes("batched") {
        run_batched_small(config, &mut rows);
    }
    if config.include_trace && config.suite.includes("small") {
        run_small_latency_trace(config, &mut rows);
    }
    if config.include_trace && config.suite.includes("large") {
        run_large_throughput_trace(config, &mut rows);
    }
    if config.include_trace && config.suite.includes("batched") {
        run_batched_small_trace(config, &mut rows);
    }
    rows
}

fn run_small_latency(config: &BenchConfig, rows: &mut Vec<Row>) {
    let sizes: &[usize] = match config.profile {
        Profile::Quick => &[2, 4, 8],
        Profile::Full => &[2, 4, 8, 16, 32],
    };

    for &n in sizes {
        rows.push(bench_row(
            config,
            "small",
            "matmul",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let ctx = cpu_ctx();
                let a = tensor(&[n, n], data_for_shape(&[n, n], 1));
                let b = tensor(&[n, n], data_for_shape(&[n, n], 2));
                let a = eager_from_tensor_in(a, ctx.clone());
                let b = eager_from_tensor_in(b, ctx);
                Ok((a, b))
            },
            |(a, b)| {
                let out = a.matmul(&b)?;
                Ok(out)
            },
        ));

        rows.push(bench_row(
            config,
            "small",
            "einsum_ij_jk_ik",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let ctx = cpu_ctx();
                let a =
                    eager_from_tensor_in(tensor(&[n, n], data_for_shape(&[n, n], 1)), ctx.clone());
                let b = eager_from_tensor_in(tensor(&[n, n], data_for_shape(&[n, n], 2)), ctx);
                Ok((a, b))
            },
            |(a, b)| {
                let out = eager_einsum_tensor::einsum(&[&a, &b], "ij,jk->ik")?;
                Ok(out)
            },
        ));

        rows.push(bench_row(
            config,
            "small",
            "svd",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a =
                    eager_from_tensor_in(tensor(&[n, n], well_conditioned_matrix(n, 3)), cpu_ctx());
                Ok((a,))
            },
            |(a,)| {
                let (u, s, vh) = eager_linalg_tensor::svd(&a)?;
                Ok((u, s, vh))
            },
        ));

        rows.push(bench_row(
            config,
            "small",
            "qr",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a =
                    eager_from_tensor_in(tensor(&[n, n], well_conditioned_matrix(n, 4)), cpu_ctx());
                Ok((a,))
            },
            |(a,)| {
                let (q, r) = eager_linalg_tensor::qr(&a)?;
                Ok((q, r))
            },
        ));

        rows.push(bench_row(
            config,
            "small",
            "eigh",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = eager_from_tensor_in(tensor(&[n, n], spd_matrix(n, 5)), cpu_ctx());
                Ok((a,))
            },
            |(a,)| {
                let (w, v) = eager_linalg_tensor::eigh(&a)?;
                Ok((w, v))
            },
        ));

        for &rhs_cols in &[1, 4] {
            rows.push(bench_row(
                config,
                "small",
                "solve",
                "primal",
                "f64",
                &format!("{n}x{n},rhs={rhs_cols}"),
                || {
                    let ctx = cpu_ctx();
                    let a = eager_from_tensor_in(tensor(&[n, n], spd_matrix(n, 6)), ctx.clone());
                    let b = eager_from_tensor_in(
                        tensor(&[n, rhs_cols], data_for_shape(&[n, rhs_cols], 7)),
                        ctx,
                    );
                    Ok((a, b))
                },
                |(a, b)| {
                    let x = eager_linalg_tensor::solve(&a, &b)?;
                    Ok(x)
                },
            ));
        }

        rows.push(bench_row(
            config,
            "small",
            "grad_sum_matmul",
            "backward",
            "f64",
            &format!("{n}x{n}"),
            || {
                let ctx = cpu_ctx();
                let a = eager_requires_grad_in(
                    tensor(&[n, n], data_for_shape(&[n, n], 8)),
                    ctx.clone(),
                );
                let b = eager_requires_grad_in(tensor(&[n, n], data_for_shape(&[n, n], 9)), ctx);
                Ok((a, b))
            },
            |(a, b)| {
                let y = a.matmul(&b)?;
                let loss = y.reduce_sum(Some(&[0, 1]))?;
                let gradients = loss.backward()?;
                Ok((loss, y, gradients))
            },
        ));

        rows.push(bench_row(
            config,
            "small",
            "grad_sum_svd_s",
            "backward",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = eager_requires_grad_in(
                    tensor(&[n, n], well_conditioned_matrix(n, 10)),
                    cpu_ctx(),
                );
                Ok((a,))
            },
            |(a,)| {
                let (u, s, vt) = eager_linalg_tensor::svd(&a)?;
                let loss = s.reduce_sum(Some(&[0]))?;
                let gradients = loss.backward()?;
                Ok((loss, u, s, vt, gradients))
            },
        ));

        rows.push(bench_row(
            config,
            "small",
            "grad_sum_solve",
            "backward",
            "f64",
            &format!("{n}x{n},rhs=1"),
            || {
                let ctx = cpu_ctx();
                let a = eager_requires_grad_in(tensor(&[n, n], spd_matrix(n, 11)), ctx.clone());
                let b = eager_requires_grad_in(tensor(&[n, 1], data_for_shape(&[n, 1], 12)), ctx);
                Ok((a, b))
            },
            |(a, b)| {
                let x = eager_linalg_tensor::solve(&a, &b)?;
                let loss = x.reduce_sum(Some(&[0, 1]))?;
                let gradients = loss.backward()?;
                Ok((loss, x, gradients))
            },
        ));
    }
}

fn run_large_throughput(config: &BenchConfig, rows: &mut Vec<Row>) {
    let matmul_sizes: &[usize] = match config.profile {
        Profile::Quick => &[128, 256],
        Profile::Full => &[128, 256, 512, 1024],
    };
    for &n in matmul_sizes {
        rows.push(bench_row(
            config,
            "large",
            "matmul",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let ctx = cpu_ctx();
                let a =
                    eager_from_tensor_in(tensor(&[n, n], data_for_shape(&[n, n], 21)), ctx.clone());
                let b = eager_from_tensor_in(tensor(&[n, n], data_for_shape(&[n, n], 22)), ctx);
                Ok((a, b))
            },
            |(a, b)| {
                let out = a.matmul(&b)?;
                Ok(out)
            },
        ));
    }

    for &(m, k, n) in &[(1024, 256, 1024), (256, 1024, 256)] {
        if config.profile == Profile::Quick && m > 256 {
            continue;
        }
        rows.push(bench_row(
            config,
            "large",
            "matmul_rect",
            "primal",
            "f64",
            &format!("{m}x{k} * {k}x{n}"),
            || {
                let ctx = cpu_ctx();
                let a =
                    eager_from_tensor_in(tensor(&[m, k], data_for_shape(&[m, k], 23)), ctx.clone());
                let b = eager_from_tensor_in(tensor(&[k, n], data_for_shape(&[k, n], 24)), ctx);
                Ok((a, b))
            },
            |(a, b)| {
                let out = a.matmul(&b)?;
                Ok(out)
            },
        ));
    }

    let linalg_sizes: &[usize] = match config.profile {
        Profile::Quick => &[64],
        Profile::Full => &[64, 128, 256, 512, 1024],
    };
    for &n in linalg_sizes {
        rows.push(bench_row(
            config,
            "large",
            "svd",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = eager_from_tensor_in(
                    tensor(&[n, n], well_conditioned_matrix(n, 25)),
                    cpu_ctx(),
                );
                Ok((a,))
            },
            |(a,)| {
                let (u, s, vh) = eager_linalg_tensor::svd(&a)?;
                Ok((u, s, vh))
            },
        ));
        rows.push(bench_row(
            config,
            "large",
            "qr",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = eager_from_tensor_in(
                    tensor(&[n, n], well_conditioned_matrix(n, 26)),
                    cpu_ctx(),
                );
                Ok((a,))
            },
            |(a,)| {
                let (q, r) = eager_linalg_tensor::qr(&a)?;
                Ok((q, r))
            },
        ));
        rows.push(bench_row(
            config,
            "large",
            "eigh",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = eager_from_tensor_in(tensor(&[n, n], spd_matrix(n, 27)), cpu_ctx());
                Ok((a,))
            },
            |(a,)| {
                let (w, v) = eager_linalg_tensor::eigh(&a)?;
                Ok((w, v))
            },
        ));
        for &rhs_cols in &[1, 16, 64] {
            rows.push(bench_row(
                config,
                "large",
                "solve",
                "primal",
                "f64",
                &format!("{n}x{n},rhs={rhs_cols}"),
                || {
                    let ctx = cpu_ctx();
                    let a = eager_from_tensor_in(tensor(&[n, n], spd_matrix(n, 28)), ctx.clone());
                    let b = eager_from_tensor_in(
                        tensor(&[n, rhs_cols], data_for_shape(&[n, rhs_cols], 29)),
                        ctx,
                    );
                    Ok((a, b))
                },
                |(a, b)| {
                    let x = eager_linalg_tensor::solve(&a, &b)?;
                    Ok(x)
                },
            ));
        }
    }

    for &n in match config.profile {
        Profile::Quick => &[64][..],
        Profile::Full => &[64, 128, 256, 512, 1024][..],
    } {
        rows.push(bench_row(
            config,
            "large",
            "grad_sum_matmul",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let ctx = cpu_ctx();
                let a = eager_requires_grad_in(
                    tensor(&[n, n], data_for_shape(&[n, n], 30)),
                    ctx.clone(),
                );
                let b = eager_requires_grad_in(tensor(&[n, n], data_for_shape(&[n, n], 31)), ctx);
                Ok((a, b))
            },
            |(a, b)| {
                let y = a.matmul(&b)?;
                let loss = y.reduce_sum(Some(&[0, 1]))?;
                Ok(loss)
            },
        ));
        rows.push(bench_row(
            config,
            "large",
            "grad_sum_matmul",
            "backward",
            "f64",
            &format!("{n}x{n}"),
            || {
                let ctx = cpu_ctx();
                let a = eager_requires_grad_in(
                    tensor(&[n, n], data_for_shape(&[n, n], 32)),
                    ctx.clone(),
                );
                let b = eager_requires_grad_in(tensor(&[n, n], data_for_shape(&[n, n], 33)), ctx);
                Ok((a, b))
            },
            |(a, b)| {
                let y = a.matmul(&b)?;
                let loss = y.reduce_sum(Some(&[0, 1]))?;
                let gradients = loss.backward()?;
                Ok((loss, y, gradients))
            },
        ));
        rows.push(bench_row(
            config,
            "large",
            "grad_sum_svd_s",
            "backward",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = eager_requires_grad_in(
                    tensor(&[n, n], well_conditioned_matrix(n, 34)),
                    cpu_ctx(),
                );
                Ok((a,))
            },
            |(a,)| {
                let (u, s, vt) = eager_linalg_tensor::svd(&a)?;
                let loss = s.reduce_sum(Some(&[0]))?;
                let gradients = loss.backward()?;
                Ok((loss, u, s, vt, gradients))
            },
        ));
        rows.push(bench_row(
            config,
            "large",
            "grad_sum_solve",
            "backward",
            "f64",
            &format!("{n}x{n},rhs=1"),
            || {
                let ctx = cpu_ctx();
                let a = eager_requires_grad_in(tensor(&[n, n], spd_matrix(n, 35)), ctx.clone());
                let b = eager_requires_grad_in(tensor(&[n, 1], data_for_shape(&[n, 1], 36)), ctx);
                Ok((a, b))
            },
            |(a, b)| {
                let x = eager_linalg_tensor::solve(&a, &b)?;
                let loss = x.reduce_sum(Some(&[0, 1]))?;
                let gradients = loss.backward()?;
                Ok((loss, x, gradients))
            },
        ));
    }
}

fn run_batched_small(config: &BenchConfig, rows: &mut Vec<Row>) {
    let batches: &[usize] = match config.profile {
        Profile::Quick => &[16, 64],
        Profile::Full => &[16, 64, 256, 1024],
    };
    let sizes: &[usize] = match config.profile {
        Profile::Quick => &[2, 4],
        Profile::Full => &[2, 4, 8, 16],
    };

    for &b in batches {
        for &n in sizes {
            let shape = format!("{n}x{n}xbatch{b} (native batch layout)");
            rows.push(bench_row(
                config,
                "batched",
                "batched_matmul_ikb_kjb_ijb",
                "primal",
                "f64",
                &shape,
                || {
                    let ctx = cpu_ctx();
                    let a = eager_from_tensor_in(
                        tensor(&[n, n, b], data_for_shape(&[n, n, b], 41)),
                        ctx.clone(),
                    );
                    let rhs = eager_from_tensor_in(
                        tensor(&[n, n, b], data_for_shape(&[n, n, b], 42)),
                        ctx,
                    );
                    let config = batched_matmul_config();
                    Ok((a, rhs, Some(config)))
                },
                |(a, rhs, config)| {
                    let out = a.dot_general(&rhs, config.take().expect("prepared config"))?;
                    Ok(out)
                },
            ));
            rows.push(bench_row(
                config,
                "batched",
                "batched_svd",
                "primal",
                "f64",
                &shape,
                || {
                    let a = eager_from_tensor_in(
                        tensor(&[n, n, b], batched_well_conditioned(n, b, 43)),
                        cpu_ctx(),
                    );
                    Ok((a,))
                },
                |(a,)| {
                    let (u, s, vh) = eager_linalg_tensor::svd(&a)?;
                    Ok((u, s, vh))
                },
            ));
            rows.push(bench_row(
                config,
                "batched",
                "batched_qr",
                "primal",
                "f64",
                &shape,
                || {
                    let a = eager_from_tensor_in(
                        tensor(&[n, n, b], batched_well_conditioned(n, b, 44)),
                        cpu_ctx(),
                    );
                    Ok((a,))
                },
                |(a,)| {
                    let (q, r) = eager_linalg_tensor::qr(&a)?;
                    Ok((q, r))
                },
            ));
            rows.push(bench_row(
                config,
                "batched",
                "batched_eigh",
                "primal",
                "f64",
                &shape,
                || {
                    let a =
                        eager_from_tensor_in(tensor(&[n, n, b], batched_spd(n, b, 45)), cpu_ctx());
                    Ok((a,))
                },
                |(a,)| {
                    let (w, v) = eager_linalg_tensor::eigh(&a)?;
                    Ok((w, v))
                },
            ));
            rows.push(bench_row(
                config,
                "batched",
                "batched_solve",
                "primal",
                "f64",
                &format!("{shape},rhs=1"),
                || {
                    let ctx = cpu_ctx();
                    let a = eager_from_tensor_in(
                        tensor(&[n, n, b], batched_spd(n, b, 46)),
                        ctx.clone(),
                    );
                    let rhs = eager_from_tensor_in(
                        tensor(&[n, 1, b], data_for_shape(&[n, 1, b], 47)),
                        ctx,
                    );
                    Ok((a, rhs))
                },
                |(a, rhs)| {
                    let x = eager_linalg_tensor::solve(&a, &rhs)?;
                    Ok(x)
                },
            ));
            rows.push(bench_row(
                config,
                "batched",
                "grad_sum_batched_matmul",
                "backward",
                "f64",
                &shape,
                || {
                    let ctx = cpu_ctx();
                    let a = eager_requires_grad_in(
                        tensor(&[n, n, b], data_for_shape(&[n, n, b], 48)),
                        ctx.clone(),
                    );
                    let rhs = eager_requires_grad_in(
                        tensor(&[n, n, b], data_for_shape(&[n, n, b], 49)),
                        ctx,
                    );
                    let config = batched_matmul_config();
                    Ok((a, rhs, Some(config)))
                },
                |(a, rhs, config)| {
                    let out = a.dot_general(&rhs, config.take().expect("prepared config"))?;
                    let loss = out.reduce_sum(Some(&[0, 1, 2]))?;
                    let gradients = loss.backward()?;
                    Ok((loss, out, gradients))
                },
            ));
            rows.push(bench_row(
                config,
                "batched",
                "grad_sum_batched_solve",
                "backward",
                "f64",
                &format!("{shape},rhs=1"),
                || {
                    let ctx = cpu_ctx();
                    let a = eager_requires_grad_in(
                        tensor(&[n, n, b], batched_spd(n, b, 50)),
                        ctx.clone(),
                    );
                    let rhs = eager_requires_grad_in(
                        tensor(&[n, 1, b], data_for_shape(&[n, 1, b], 51)),
                        ctx,
                    );
                    Ok((a, rhs))
                },
                |(a, rhs)| {
                    let x = eager_linalg_tensor::solve(&a, &rhs)?;
                    let loss = x.reduce_sum(Some(&[0, 1, 2]))?;
                    let gradients = loss.backward()?;
                    Ok((loss, x, gradients))
                },
            ));
        }
    }
}

fn run_small_latency_trace(config: &BenchConfig, rows: &mut Vec<Row>) {
    let sizes: &[usize] = match config.profile {
        Profile::Quick => &[2, 4, 8],
        Profile::Full => &[2, 4, 8, 16, 32],
    };

    for &n in sizes {
        rows.push(bench_trace_row(
            config,
            "small",
            "matmul",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = traced_tensor(&[n, n], data_for_shape(&[n, n], 1));
                let b = traced_tensor(&[n, n], data_for_shape(&[n, n], 2));
                Ok(vec![traced_tensor::matmul(&a, &b)])
            },
        ));
        rows.push(bench_einsum_trace_row(config, n));
        rows.push(bench_trace_row(
            config,
            "small",
            "svd",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = traced_tensor(&[n, n], well_conditioned_matrix(n, 3));
                let (u, s, vh) = a.svd()?;
                Ok(vec![u, s, vh])
            },
        ));
        rows.push(bench_trace_row(
            config,
            "small",
            "qr",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = traced_tensor(&[n, n], well_conditioned_matrix(n, 4));
                let (q, r) = a.qr()?;
                Ok(vec![q, r])
            },
        ));
        rows.push(bench_trace_row(
            config,
            "small",
            "eigh",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = traced_tensor(&[n, n], spd_matrix(n, 5));
                let (w, v) = a.eigh()?;
                Ok(vec![w, v])
            },
        ));
        for &rhs_cols in &[1, 4] {
            rows.push(bench_trace_row(
                config,
                "small",
                "solve",
                "primal",
                "f64",
                &format!("{n}x{n},rhs={rhs_cols}"),
                || {
                    let a = traced_tensor(&[n, n], spd_matrix(n, 6));
                    let b = traced_tensor(&[n, rhs_cols], data_for_shape(&[n, rhs_cols], 7));
                    Ok(vec![a.solve(&b)?])
                },
            ));
        }
        rows.push(bench_trace_row(
            config,
            "small",
            "grad_sum_matmul",
            "backward",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = traced_tensor(&[n, n], data_for_shape(&[n, n], 8));
                let b = traced_tensor(&[n, n], data_for_shape(&[n, n], 9));
                let y = traced_tensor::matmul(&a, &b);
                let loss = y.reduce_sum(Some(&[0, 1]))?;
                Ok(vec![grad(&loss, &a)?, grad(&loss, &b)?, loss])
            },
        ));
        rows.push(bench_trace_row(
            config,
            "small",
            "grad_sum_svd_s",
            "backward",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = traced_tensor(&[n, n], well_conditioned_matrix(n, 10));
                let (_, s, _) = a.svd()?;
                let loss = s.reduce_sum(Some(&[0]))?;
                Ok(vec![grad(&loss, &a)?, loss])
            },
        ));
        rows.push(bench_trace_row(
            config,
            "small",
            "grad_sum_solve",
            "backward",
            "f64",
            &format!("{n}x{n},rhs=1"),
            || {
                let a = traced_tensor(&[n, n], spd_matrix(n, 11));
                let b = traced_tensor(&[n, 1], data_for_shape(&[n, 1], 12));
                let x = a.solve(&b)?;
                let loss = x.reduce_sum(Some(&[0, 1]))?;
                Ok(vec![grad(&loss, &a)?, grad(&loss, &b)?, loss])
            },
        ));
        push_linalg_jvp_vjp_trace_benches(
            config,
            rows,
            "small",
            &format!("{n}x{n}"),
            n,
            tangent_seed(10),
            LinalgAdLoss::SumSvdS { matrix_seed: 10 },
        );
        push_linalg_jvp_vjp_trace_benches(
            config,
            rows,
            "small",
            &format!("{n}x{n}"),
            n,
            tangent_seed(4),
            LinalgAdLoss::SumQr { matrix_seed: 4 },
        );
        push_linalg_jvp_vjp_trace_benches(
            config,
            rows,
            "small",
            &format!("{n}x{n}"),
            n,
            tangent_seed(5),
            LinalgAdLoss::SumEigh { matrix_seed: 5 },
        );
        push_linalg_jvp_vjp_trace_benches(
            config,
            rows,
            "small",
            &format!("{n}x{n}"),
            n,
            tangent_seed(52),
            LinalgAdLoss::SumLu { matrix_seed: 52 },
        );
        push_linalg_jvp_vjp_trace_benches(
            config,
            rows,
            "small",
            &format!("{n}x{n},rhs=1"),
            n,
            tangent_seed(11),
            LinalgAdLoss::SumSolveWrtA {
                matrix_seed: 11,
                rhs_seed: 12,
                rhs_cols: 1,
            },
        );
    }
}

fn run_large_throughput_trace(config: &BenchConfig, rows: &mut Vec<Row>) {
    let matmul_sizes: &[usize] = match config.profile {
        Profile::Quick => &[128, 256],
        Profile::Full => &[128, 256, 512, 1024],
    };
    for &n in matmul_sizes {
        rows.push(bench_trace_row(
            config,
            "large",
            "matmul",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = traced_tensor(&[n, n], data_for_shape(&[n, n], 21));
                let b = traced_tensor(&[n, n], data_for_shape(&[n, n], 22));
                Ok(vec![traced_tensor::matmul(&a, &b)])
            },
        ));
    }
    for &(m, k, n) in &[(1024, 256, 1024), (256, 1024, 256)] {
        if config.profile == Profile::Quick && m > 256 {
            continue;
        }
        rows.push(bench_trace_row(
            config,
            "large",
            "matmul_rect",
            "primal",
            "f64",
            &format!("{m}x{k} * {k}x{n}"),
            || {
                let a = traced_tensor(&[m, k], data_for_shape(&[m, k], 23));
                let b = traced_tensor(&[k, n], data_for_shape(&[k, n], 24));
                Ok(vec![traced_tensor::matmul(&a, &b)])
            },
        ));
    }
    let linalg_sizes: &[usize] = match config.profile {
        Profile::Quick => &[64],
        Profile::Full => &[64, 128, 256, 512, 1024],
    };
    for &n in linalg_sizes {
        rows.push(bench_trace_row(
            config,
            "large",
            "svd",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = traced_tensor(&[n, n], well_conditioned_matrix(n, 25));
                let (u, s, vh) = a.svd()?;
                Ok(vec![u, s, vh])
            },
        ));
        rows.push(bench_trace_row(
            config,
            "large",
            "qr",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = traced_tensor(&[n, n], well_conditioned_matrix(n, 26));
                let (q, r) = a.qr()?;
                Ok(vec![q, r])
            },
        ));
        rows.push(bench_trace_row(
            config,
            "large",
            "eigh",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = traced_tensor(&[n, n], spd_matrix(n, 27));
                let (w, v) = a.eigh()?;
                Ok(vec![w, v])
            },
        ));
        for &rhs_cols in &[1, 16, 64] {
            rows.push(bench_trace_row(
                config,
                "large",
                "solve",
                "primal",
                "f64",
                &format!("{n}x{n},rhs={rhs_cols}"),
                || {
                    let a = traced_tensor(&[n, n], spd_matrix(n, 28));
                    let b = traced_tensor(&[n, rhs_cols], data_for_shape(&[n, rhs_cols], 29));
                    Ok(vec![a.solve(&b)?])
                },
            ));
        }
    }
    for &n in match config.profile {
        Profile::Quick => &[64][..],
        Profile::Full => &[64, 128, 256, 512, 1024][..],
    } {
        rows.push(bench_trace_row(
            config,
            "large",
            "grad_sum_matmul",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = traced_tensor(&[n, n], data_for_shape(&[n, n], 30));
                let b = traced_tensor(&[n, n], data_for_shape(&[n, n], 31));
                Ok(vec![
                    traced_tensor::matmul(&a, &b).reduce_sum(Some(&[0, 1]))?
                ])
            },
        ));
        rows.push(bench_trace_row(
            config,
            "large",
            "grad_sum_matmul",
            "backward",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = traced_tensor(&[n, n], data_for_shape(&[n, n], 32));
                let b = traced_tensor(&[n, n], data_for_shape(&[n, n], 33));
                let loss = traced_tensor::matmul(&a, &b).reduce_sum(Some(&[0, 1]))?;
                Ok(vec![grad(&loss, &a)?, grad(&loss, &b)?, loss])
            },
        ));
        rows.push(bench_trace_row(
            config,
            "large",
            "grad_sum_svd_s",
            "backward",
            "f64",
            &format!("{n}x{n}"),
            || {
                let a = traced_tensor(&[n, n], well_conditioned_matrix(n, 34));
                let (_, s, _) = a.svd()?;
                let loss = s.reduce_sum(Some(&[0]))?;
                Ok(vec![grad(&loss, &a)?, loss])
            },
        ));
        rows.push(bench_trace_row(
            config,
            "large",
            "grad_sum_solve",
            "backward",
            "f64",
            &format!("{n}x{n},rhs=1"),
            || {
                let a = traced_tensor(&[n, n], spd_matrix(n, 35));
                let b = traced_tensor(&[n, 1], data_for_shape(&[n, 1], 36));
                let loss = a.solve(&b)?.reduce_sum(Some(&[0, 1]))?;
                Ok(vec![grad(&loss, &a)?, grad(&loss, &b)?, loss])
            },
        ));
    }
    for &n in large_linalg_jvp_vjp_sizes(config.profile) {
        push_linalg_jvp_vjp_trace_benches(
            config,
            rows,
            "large",
            &format!("{n}x{n}"),
            n,
            tangent_seed(34),
            LinalgAdLoss::SumSvdS { matrix_seed: 34 },
        );
        push_linalg_jvp_vjp_trace_benches(
            config,
            rows,
            "large",
            &format!("{n}x{n}"),
            n,
            tangent_seed(26),
            LinalgAdLoss::SumQr { matrix_seed: 26 },
        );
        push_linalg_jvp_vjp_trace_benches(
            config,
            rows,
            "large",
            &format!("{n}x{n}"),
            n,
            tangent_seed(27),
            LinalgAdLoss::SumEigh { matrix_seed: 27 },
        );
        push_linalg_jvp_vjp_trace_benches(
            config,
            rows,
            "large",
            &format!("{n}x{n}"),
            n,
            tangent_seed(54),
            LinalgAdLoss::SumLu { matrix_seed: 54 },
        );
        push_linalg_jvp_vjp_trace_benches(
            config,
            rows,
            "large",
            &format!("{n}x{n},rhs=1"),
            n,
            tangent_seed(35),
            LinalgAdLoss::SumSolveWrtA {
                matrix_seed: 35,
                rhs_seed: 36,
                rhs_cols: 1,
            },
        );
    }
}

fn run_batched_small_trace(config: &BenchConfig, rows: &mut Vec<Row>) {
    let batches: &[usize] = match config.profile {
        Profile::Quick => &[16, 64],
        Profile::Full => &[16, 64, 256, 1024],
    };
    let sizes: &[usize] = match config.profile {
        Profile::Quick => &[2, 4],
        Profile::Full => &[2, 4, 8, 16],
    };

    for &b in batches {
        for &n in sizes {
            let shape = format!("{n}x{n}xbatch{b} (native batch layout)");
            rows.push(bench_trace_row(
                config,
                "batched",
                "batched_matmul_ikb_kjb_ijb",
                "primal",
                "f64",
                &shape,
                || {
                    let a = traced_tensor(&[n, n, b], data_for_shape(&[n, n, b], 41));
                    let rhs = traced_tensor(&[n, n, b], data_for_shape(&[n, n, b], 42));
                    Ok(vec![a.dot_general(&rhs, batched_matmul_config())?])
                },
            ));
            rows.push(bench_trace_row(
                config,
                "batched",
                "batched_svd",
                "primal",
                "f64",
                &shape,
                || {
                    let a = traced_tensor(&[n, n, b], batched_well_conditioned(n, b, 43));
                    let (u, s, vh) = a.svd()?;
                    Ok(vec![u, s, vh])
                },
            ));
            rows.push(bench_trace_row(
                config,
                "batched",
                "batched_qr",
                "primal",
                "f64",
                &shape,
                || {
                    let a = traced_tensor(&[n, n, b], batched_well_conditioned(n, b, 44));
                    let (q, r) = a.qr()?;
                    Ok(vec![q, r])
                },
            ));
            rows.push(bench_trace_row(
                config,
                "batched",
                "batched_eigh",
                "primal",
                "f64",
                &shape,
                || {
                    let a = traced_tensor(&[n, n, b], batched_spd(n, b, 45));
                    let (w, v) = a.eigh()?;
                    Ok(vec![w, v])
                },
            ));
            rows.push(bench_trace_row(
                config,
                "batched",
                "batched_solve",
                "primal",
                "f64",
                &format!("{shape},rhs=1"),
                || {
                    let a = traced_tensor(&[n, n, b], batched_spd(n, b, 46));
                    let rhs = traced_tensor(&[n, 1, b], data_for_shape(&[n, 1, b], 47));
                    Ok(vec![a.solve(&rhs)?])
                },
            ));
            rows.push(bench_trace_row(
                config,
                "batched",
                "grad_sum_batched_matmul",
                "backward",
                "f64",
                &shape,
                || {
                    let a = traced_tensor(&[n, n, b], data_for_shape(&[n, n, b], 48));
                    let rhs = traced_tensor(&[n, n, b], data_for_shape(&[n, n, b], 49));
                    let loss = a
                        .dot_general(&rhs, batched_matmul_config())?
                        .reduce_sum(Some(&[0, 1, 2]))?;
                    Ok(vec![grad(&loss, &a)?, grad(&loss, &rhs)?, loss])
                },
            ));
            rows.push(bench_trace_row(
                config,
                "batched",
                "grad_sum_batched_solve",
                "backward",
                "f64",
                &format!("{shape},rhs=1"),
                || {
                    let a = traced_tensor(&[n, n, b], batched_spd(n, b, 50));
                    let rhs = traced_tensor(&[n, 1, b], data_for_shape(&[n, 1, b], 51));
                    let loss = a.solve(&rhs)?.reduce_sum(Some(&[0, 1, 2]))?;
                    Ok(vec![grad(&loss, &a)?, grad(&loss, &rhs)?, loss])
                },
            ));
        }
    }
}

const SAMPLE_TARGET_NS: u128 = 5_000_000;
const RETENTION_BUDGET_BYTES: usize = 64 * 1024 * 1024;

// Conservative logical-buffer reservation for this f64 matrix suite: sixteen
// tensors plus 8 KiB of handles/AD metadata per invocation. Multiplying all
// dimensions in the declared shape intentionally overestimates rectangular
// and multiple-RHS cases. This is not an RSS bound on backend scratch/cache.
fn retention_estimate(shape: &str) -> usize {
    shape
        .split(|c: char| !c.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .fold(1usize, |size, part| {
            size.saturating_mul(part.parse().unwrap())
        })
        .saturating_mul(16 * 8)
        .saturating_add(8192)
}

struct Measurement {
    operations: usize,
    estimated_bytes: usize,
    calibration_ns: u128,
    times: Vec<Duration>,
}

fn measure<I, O>(
    config: &BenchConfig,
    shape: &str,
    mut setup: impl FnMut() -> Result<I, Error>,
    mut execute: impl FnMut(&mut I) -> Result<O, Error>,
) -> Result<Measurement, Error> {
    let estimated_bytes = retention_estimate(shape);
    // A single operation remains permitted when its reservation exceeds the
    // batching budget; never multiply such a fixture's retained memory.
    let cap = (RETENTION_BUDGET_BYTES / estimated_bytes).clamp(1, 65_536);
    let mut batch = |operations| {
        let mut inputs = (0..operations)
            .map(|_| setup())
            .collect::<Result<Vec<_>, _>>()?;
        let mut outputs = Vec::with_capacity(operations);
        let start = Instant::now();
        for input in &mut inputs {
            outputs.push(execute(input)?);
        }
        let elapsed = start.elapsed();
        black_box(&outputs);
        // Both vectors (including gradients and output tensors) drop after stop.
        Ok::<_, Error>(elapsed)
    };
    for _ in 0..config.warmups.max(1) {
        batch(1)?;
    }
    let mut operations = 1;
    let calibration_ns = loop {
        let elapsed = batch(operations)?.as_nanos();
        if elapsed >= SAMPLE_TARGET_NS || operations == cap {
            break elapsed;
        }
        operations = (operations * 2).min(cap);
    };
    let times = (0..config.runs)
        .map(|_| batch(operations))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Measurement {
        operations,
        estimated_bytes,
        calibration_ns,
        times,
    })
}

fn record_raw(record: serde_json::Value) -> Result<(), Error> {
    if let Ok(path) = env::var("CPU_OPS_RAW_SAMPLES") {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| Error::Internal(format!("open raw samples {path}: {error}")))?;
        writeln!(file, "{record}")
            .map_err(|error| Error::Internal(format!("write raw samples {path}: {error}")))?;
    }
    Ok(())
}

fn summarize_measurement(
    measured: Measurement,
    suite: &str,
    op: &str,
    phase: &str,
    shape: &str,
    backend: &str,
) -> Result<(f64, f64), Error> {
    record_raw(serde_json::json!({
        "record_type": "samples", "suite": suite,
        "benchmark": benchmark_name(op, phase), "shape": shape,
        "backend": backend, "threads": cpu_backend().num_threads(),
        "operations_per_sample": measured.operations,
        "batch_elapsed_ns": measured.times.iter().map(Duration::as_nanos).collect::<Vec<_>>(),
        "per_operation_ns": measured.times.iter().map(|t| t.as_nanos() as f64 / measured.operations as f64).collect::<Vec<_>>(),
        "target_ns": SAMPLE_TARGET_NS, "calibration_ns": measured.calibration_ns,
        "estimated_retained_bytes_per_operation": measured.estimated_bytes,
        "retention_budget_bytes": RETENTION_BUDGET_BYTES,
        "execution_scope": "shared_cpu", "output_policy": "allocation_returning_retained_until_stop",
    }))?;
    let (median, iqr) = median_iqr_ms(measured.times);
    Ok((
        median / measured.operations as f64,
        iqr / measured.operations as f64,
    ))
}

fn cpu_backend() -> &'static CpuBackend {
    static BACKEND: OnceLock<CpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        CpuBackend::with_threads(requested_threads()).expect("configure explicit CPU thread count")
    })
}

fn record_cpu_runtime() -> Result<(), Error> {
    #[cfg(feature = "system-mkl")]
    let provider_threads = {
        extern "C" {
            fn MKL_Get_Max_Threads() -> i32;
        }
        // SAFETY: system-mkl links the oneMKL C ABI; this argument-free query
        // only reads the calling thread's effective provider configuration.
        let threads = unsafe { MKL_Get_Max_Threads() };
        assert_eq!(
            threads as usize,
            cpu_backend().num_threads(),
            "MKL and CPU thread budgets differ"
        );
        Some(threads)
    };
    #[cfg(not(feature = "system-mkl"))]
    let provider_threads: Option<i32> = None;
    let affinity = std::fs::read_to_string("/proc/thread-self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("Cpus_allowed_list:"))
                .map(str::to_owned)
        });
    let info = cpu_backend().execution_info();
    record_raw(serde_json::json!({
        "record_type": "runtime", "backend": "tenferro", "provider": backend_name(),
        "thread_budget": info.thread_budget(), "worker_count": info.worker_count(),
        "provider_max_threads": provider_threads, "scope_worker_affinity": affinity,
        "execution_info": format!("{info:?}"),
    }))
}

fn bench_row<I, O>(
    config: &BenchConfig,
    suite: &'static str,
    op: &'static str,
    phase: &'static str,
    dtype: &'static str,
    shape: &str,
    mut setup: impl FnMut() -> tenferro_ad::error::Result<I>,
    mut execute: impl FnMut(&mut I) -> tenferro_ad::error::Result<O>,
) -> Row {
    if !benchmark_filter_matches(op, phase) {
        return filtered_row(suite, op, phase, dtype, shape);
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let measured = measure(config, shape, &mut setup, &mut execute)?;
        summarize_measurement(measured, suite, op, phase, shape, "tenferro-eager")
    }));

    match result {
        Ok(Ok((median, iqr))) => Row {
            suite,
            op,
            phase,
            dtype,
            shape: shape.to_string(),
            backend: config.backend,
            median_ms: Some(median),
            iqr_ms: Some(iqr),
            status: "ok".to_string(),
        },
        Ok(Err(err)) => Row {
            suite,
            op,
            phase,
            dtype,
            shape: shape.to_string(),
            backend: config.backend,
            median_ms: None,
            iqr_ms: None,
            status: format!("error: {err}"),
        },
        Err(_) => Row {
            suite,
            op,
            phase,
            dtype,
            shape: shape.to_string(),
            backend: config.backend,
            median_ms: None,
            iqr_ms: None,
            status: "panic".to_string(),
        },
    }
}

fn bench_trace_row(
    config: &BenchConfig,
    suite: &'static str,
    op: &'static str,
    phase: &'static str,
    dtype: &'static str,
    shape: &str,
    mut build: impl FnMut() -> tenferro_ad::error::Result<Vec<TracedTensor>>,
) -> Row {
    if !benchmark_filter_matches(op, phase) {
        return filtered_row(suite, op, phase, dtype, shape);
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let outputs = build()?;
        let output_refs: Vec<&TracedTensor> = outputs.iter().collect();
        let mut compiler = GraphCompiler::new();
        let program = compiler.compile_many(&output_refs)?;
        let runtime = cpu_runtime_with_extensions()?;
        let prepared = runtime.prepare_compiled(&program, &[])?;
        let measured = measure(
            config,
            shape,
            || Ok(()),
            |_| runtime.run_prepared(&prepared, &[]),
        )?;
        summarize_measurement(measured, suite, op, phase, shape, "tenferro-trace")
    }));

    match result {
        Ok(Ok((median, iqr))) => Row {
            suite,
            op,
            phase,
            dtype,
            shape: shape.to_string(),
            backend: "tenferro-trace",
            median_ms: Some(median),
            iqr_ms: Some(iqr),
            status: "ok".to_string(),
        },
        Ok(Err(err)) => Row {
            suite,
            op,
            phase,
            dtype,
            shape: shape.to_string(),
            backend: "tenferro-trace",
            median_ms: None,
            iqr_ms: None,
            status: format!("error: {err}"),
        },
        Err(_) => Row {
            suite,
            op,
            phase,
            dtype,
            shape: shape.to_string(),
            backend: "tenferro-trace",
            median_ms: None,
            iqr_ms: None,
            status: "panic".to_string(),
        },
    }
}

fn cpu_runtime_with_extensions() -> Result<Runtime, Error> {
    let backend = cpu_backend();
    let engine_id = runtime_engine_id().map_err(|err| Error::Internal(err.to_string()))?;
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            runtime_engine_registration(backend).map_err(|err| Error::Internal(err.to_string()))?,
        )
        .map_err(|err| Error::Internal(err.to_string()))?;
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<CpuBackend>(engine_id.clone())
                .map_err(|err| Error::Internal(err.to_string()))?,
        )
        .map_err(|err| Error::Internal(err.to_string()))?;
    builder
        .install_extension_module(
            tenferro_linalg::extension_module::<CpuBackend>(engine_id)
                .map_err(|err| Error::Internal(err.to_string()))?,
        )
        .map_err(|err| Error::Internal(err.to_string()))?;
    builder
        .build()
        .map_err(|err| Error::Internal(err.to_string()))
}

fn bench_einsum_trace_row(config: &BenchConfig, n: usize) -> Row {
    let suite = "small";
    let op = "einsum_ij_jk_ik";
    let phase = "primal";
    let dtype = "f64";
    let shape = format!("{n}x{n}");
    if !benchmark_filter_matches(op, phase) {
        return filtered_row(suite, op, phase, dtype, &shape);
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let a = tensor(&[n, n], data_for_shape(&[n, n], 1));
        let b = tensor(&[n, n], data_for_shape(&[n, n], 2));
        let mut trace = TraceContext::new();
        let a_value = trace
            .input_with_default(ProgramInputSpec::new(a.dtype(), [n.into(), n.into()]), a)
            .map_err(|err| Error::Internal(err.to_string()))?;
        let b_value = trace
            .input_with_default(ProgramInputSpec::new(b.dtype(), [n.into(), n.into()]), b)
            .map_err(|err| Error::Internal(err.to_string()))?;
        let output = trace
            .einsum(&[a_value, b_value], "ij,jk->ik")
            .map_err(|error| runtime_einsum_error(error, ErrorPhase::GraphBuild))?;
        let graph = trace
            .finish(&[output])
            .map_err(|err| Error::Internal(err.to_string()))?;
        let program = GraphCompiler::new().compile_traced_graph(&graph)?;
        let runtime = cpu_runtime_with_extensions()?;
        let prepared = runtime.prepare_compiled(&program, &[])?;
        let measured = measure(
            config,
            &shape,
            || Ok(()),
            |_| runtime.run_prepared(&prepared, &[]),
        )?;
        summarize_measurement(measured, suite, op, phase, &shape, "tenferro-trace")
    }));

    match result {
        Ok(Ok((median, iqr))) => Row {
            suite,
            op,
            phase,
            dtype,
            shape,
            backend: "tenferro-trace",
            median_ms: Some(median),
            iqr_ms: Some(iqr),
            status: "ok".to_string(),
        },
        Ok(Err(err)) => Row {
            suite,
            op,
            phase,
            dtype,
            shape,
            backend: "tenferro-trace",
            median_ms: None,
            iqr_ms: None,
            status: format!("error: {err}"),
        },
        Err(_) => Row {
            suite,
            op,
            phase,
            dtype,
            shape,
            backend: "tenferro-trace",
            median_ms: None,
            iqr_ms: None,
            status: "panic".to_string(),
        },
    }
}

fn cpu_ctx() -> Arc<EagerRuntime> {
    static RUNTIME: OnceLock<Arc<EagerRuntime>> = OnceLock::new();
    Arc::clone(RUNTIME.get_or_init(|| {
        EagerRuntime::with_cpu_backend_and_ad_context(cpu_backend().clone(), ad_context())
            .expect("configured eager CPU runtime should initialize")
    }))
}

fn ad_context() -> &'static AdContext {
    static AD_CONTEXT: OnceLock<AdContext> = OnceLock::new();
    AD_CONTEXT.get_or_init(|| {
        AdContext::builder()
            .with_semantic_extension_rules(
                tenferro_linalg::semantic_ad_rules()
                    .expect("tenferro-linalg AD rules should register"),
            )
            .expect("tenferro-linalg semantic AD rules should merge")
            .build()
            .expect("tenferro AD context should build")
    })
}

fn grad(output: &TracedTensor, wrt: &TracedTensor) -> tenferro_ad::error::Result<TracedTensor> {
    ad_context().grad(output, wrt)
}

fn jvp(
    output: &TracedTensor,
    wrt: &TracedTensor,
    tangent: &TracedTensor,
) -> tenferro_ad::error::Result<TracedTensor> {
    ad_context().jvp(output, wrt, tangent)
}

fn vjp(
    output: &TracedTensor,
    wrt: &TracedTensor,
    cotangent: &TracedTensor,
) -> tenferro_ad::error::Result<TracedTensor> {
    ad_context().vjp(output, wrt, cotangent)
}

const TANGENT_SEED_OFFSET: u64 = 1_000;

fn tangent_seed(matrix_seed: u64) -> u64 {
    matrix_seed.wrapping_add(TANGENT_SEED_OFFSET)
}

fn scalar_one() -> TracedTensor {
    traced_tensor(&[], vec![1.0])
}

#[derive(Clone, Copy, Debug)]
enum LinalgAdLoss {
    SumSvdS {
        matrix_seed: u64,
    },
    SumQr {
        matrix_seed: u64,
    },
    SumEigh {
        matrix_seed: u64,
    },
    SumLu {
        matrix_seed: u64,
    },
    SumSolveWrtA {
        matrix_seed: u64,
        rhs_seed: u64,
        rhs_cols: usize,
    },
}

impl LinalgAdLoss {
    fn op_name(self) -> &'static str {
        match self {
            Self::SumSvdS { .. } => "grad_sum_svd_s",
            Self::SumQr { .. } => "grad_sum_qr",
            Self::SumEigh { .. } => "grad_sum_eigh",
            Self::SumLu { .. } => "grad_sum_lu",
            Self::SumSolveWrtA { .. } => "grad_sum_solve",
        }
    }

    fn build(self, n: usize) -> tenferro_ad::error::Result<(TracedTensor, TracedTensor)> {
        match self {
            Self::SumSvdS { matrix_seed } => {
                let a = traced_tensor(&[n, n], well_conditioned_matrix(n, matrix_seed));
                let (_, s, _) = a.svd()?;
                Ok((s.reduce_sum(Some(&[0]))?, a))
            }
            Self::SumQr { matrix_seed } => {
                let a = traced_tensor(&[n, n], well_conditioned_matrix(n, matrix_seed));
                let (q, r) = a.qr()?;
                let loss = (&q.reduce_sum(Some(&[0, 1]))? + &r.reduce_sum(Some(&[0, 1]))?)?;
                Ok((loss, a))
            }
            Self::SumEigh { matrix_seed } => {
                let a = traced_tensor(&[n, n], spd_matrix(n, matrix_seed));
                let (w, _) = a.eigh()?;
                Ok((w.reduce_sum(Some(&[0]))?, a))
            }
            Self::SumLu { matrix_seed } => {
                let a = traced_tensor(&[n, n], well_conditioned_matrix(n, matrix_seed));
                let (_, l, u, _) = a.lu()?;
                let loss = (&l.reduce_sum(Some(&[0, 1]))? + &u.reduce_sum(Some(&[0, 1]))?)?;
                Ok((loss, a))
            }
            Self::SumSolveWrtA {
                matrix_seed,
                rhs_seed,
                rhs_cols,
            } => {
                let a = traced_tensor(&[n, n], spd_matrix(n, matrix_seed));
                let b = traced_tensor(&[n, rhs_cols], data_for_shape(&[n, rhs_cols], rhs_seed));
                let x = a.solve(&b)?;
                Ok((x.reduce_sum(Some(&[0, 1]))?, a))
            }
        }
    }
}

fn push_linalg_jvp_vjp_trace_benches(
    config: &BenchConfig,
    rows: &mut Vec<Row>,
    suite: &'static str,
    shape: &str,
    n: usize,
    tangent_seed: u64,
    loss: LinalgAdLoss,
) {
    let op = loss.op_name();
    rows.push(bench_trace_row(
        config,
        suite,
        op,
        "jvp",
        "f64",
        shape,
        || {
            let (output, wrt) = loss.build(n)?;
            let tangent = traced_tensor(&[n, n], data_for_shape(&[n, n], tangent_seed));
            let derivative = jvp(&output, &wrt, &tangent)?;
            Ok(vec![output, derivative])
        },
    ));
    rows.push(bench_trace_row(
        config,
        suite,
        op,
        "vjp",
        "f64",
        shape,
        || {
            let (output, wrt) = loss.build(n)?;
            let derivative = vjp(&output, &wrt, &scalar_one())?;
            Ok(vec![output, derivative])
        },
    ));
}

fn tensor(shape: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::from_typed(
        TypedTensor::from_vec_col_major(shape.to_vec(), data)
            .expect("benchmark shape/data length should match"),
    )
}

fn traced_tensor(shape: &[usize], data: Vec<f64>) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(tensor(shape, data))
        .expect("benchmark tensor should produce a traced input")
}

fn data_for_shape(shape: &[usize], seed: u64) -> Vec<f64> {
    let len: usize = shape.iter().product();
    (0..len)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed.wrapping_mul(1442695040888963407));
            ((x % 1024) as f64 - 512.0) / 512.0
        })
        .collect()
}

fn well_conditioned_matrix(n: usize, seed: u64) -> Vec<f64> {
    let mut data = data_for_shape(&[n, n], seed);
    for j in 0..n {
        for i in 0..n {
            data[i + n * j] *= 0.05;
        }
        data[j + n * j] += 1.0 + j as f64 / n.max(1) as f64;
    }
    data
}

fn spd_matrix(n: usize, seed: u64) -> Vec<f64> {
    let base = well_conditioned_matrix(n, seed);
    let mut out = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += base[k + n * i] * base[k + n * j];
            }
            if i == j {
                sum += 1.0;
            }
            out[i + n * j] = sum;
        }
    }
    out
}

fn batched_well_conditioned(n: usize, batch: usize, seed: u64) -> Vec<f64> {
    let mut data = vec![0.0; n * n * batch];
    for b in 0..batch {
        let matrix = well_conditioned_matrix(n, seed + b as u64);
        let offset = n * n * b;
        data[offset..offset + n * n].copy_from_slice(&matrix);
    }
    data
}

fn batched_spd(n: usize, batch: usize, seed: u64) -> Vec<f64> {
    let mut data = vec![0.0; n * n * batch];
    for b in 0..batch {
        let matrix = spd_matrix(n, seed + b as u64);
        let offset = n * n * b;
        data[offset..offset + n * n].copy_from_slice(&matrix);
    }
    data
}

fn batched_matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![2],
        rhs_batch_dims: vec![2],
    }
}

fn median_iqr_ms(mut times: Vec<Duration>) -> (f64, f64) {
    times.sort();
    let median = percentile_ms(&times, 0.5);
    let q1 = percentile_ms(&times, 0.25);
    let q3 = percentile_ms(&times, 0.75);
    (median, q3 - q1)
}

fn percentile_ms(times: &[Duration], p: f64) -> f64 {
    if times.is_empty() {
        return 0.0;
    }
    let idx = ((times.len() - 1) as f64 * p).round() as usize;
    times[idx].as_secs_f64() * 1.0e3
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod timing_tests {
    use super::*;

    #[test]
    fn eager_runtime_is_reused_across_samples() {
        assert!(Arc::ptr_eq(&cpu_ctx(), &cpu_ctx()));
    }

    #[test]
    fn batch_reservation_counts_matrix_and_batch_dimensions() {
        assert_eq!(
            retention_estimate("2x2xbatch16 (native batch layout)"),
            16_384
        );
        assert_eq!(retention_estimate("4x4"), 10_240);
        assert!(SuiteFilter::All.includes("small"));
    }

    #[test]
    fn shared_benchmark_primal_and_both_gradients_match_scalar_references() {
        let ctx = cpu_ctx();
        let runtime = cpu_runtime_with_extensions().unwrap();
        cpu_backend()
            .with_execution_scope(|| {
                for (n, batch) in [(2, 16), (4, 3), (16, 1)] {
                    let shape = [n, n, batch];
                    let a = data_for_shape(&shape, 48);
                    let b = data_for_shape(&shape, 49);
                    let mut expected = vec![0.0; a.len()];
                    let mut da = vec![0.0; a.len()];
                    let mut db = vec![0.0; b.len()];
                    for z in 0..batch {
                        for j in 0..n {
                            for i in 0..n {
                                for k in 0..n {
                                    let ai = i + n * (k + n * z);
                                    let bi = k + n * (j + n * z);
                                    expected[i + n * (j + n * z)] += a[ai] * b[bi];
                                    da[ai] += b[bi];
                                    db[bi] += a[ai];
                                }
                            }
                        }
                    }
                    let check = |values: &[f64], expected: &[f64]| {
                        assert_eq!(values.len(), expected.len());
                        for (value, expected) in values.iter().zip(expected) {
                            assert!((value - expected).abs() < 1e-11);
                        }
                    };
                    let x = eager_requires_grad_in(tensor(&shape, a.clone()), ctx.clone());
                    let y = eager_requires_grad_in(tensor(&shape, b.clone()), ctx.clone());
                    let output = x.dot_general(&y, batched_matmul_config()).unwrap();
                    assert_eq!(output.shape(), shape);
                    check(
                        output.to_tensor().unwrap().as_slice::<f64>().unwrap(),
                        &expected,
                    );
                    output.reduce_sum(None).unwrap().backward().unwrap();
                    check(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &da);
                    check(y.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &db);

                    let x = traced_tensor(&shape, a);
                    let y = traced_tensor(&shape, b);
                    let output = x.dot_general(&y, batched_matmul_config()).unwrap();
                    let loss = output.reduce_sum(None).unwrap();
                    let gx = grad(&loss, &x).unwrap();
                    let gy = grad(&loss, &y).unwrap();
                    let program = GraphCompiler::new()
                        .compile_many(&[&output, &gx, &gy])
                        .unwrap();
                    let prepared = runtime.prepare_compiled(&program, &[]).unwrap();
                    let outputs = runtime.run_prepared(&prepared, &[]).unwrap();
                    assert_eq!(outputs.len(), 3);
                    for (actual, expected) in outputs.iter().zip([&expected, &da, &db]) {
                        check(actual.as_slice::<f64>().unwrap(), expected);
                    }
                }
            })
            .unwrap();
    }
}
