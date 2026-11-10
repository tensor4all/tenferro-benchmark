//! Publication-gate microbenchmarks for tenferro-rs.
//!
//! This runner covers the focused benchmark matrix from tensor4all/tenferro-rs#862:
//! small matrix latency, large matrix throughput, and batched small matrices.

use std::env;
use std::hint::black_box;
use std::io::Write;
use std::panic;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use tenferro_ad::{AdContext, EagerRuntime, EagerSession, EagerTensor};
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuEngine};
use tenferro_einsum::{EagerSessionEinsumExt, TraceContextEinsumExt};
use tenferro_einsum_benchmark::spd_from_base;
use tenferro_einsum_benchmark::thread_enforcement::{
    enforce_thread_request, verify_backend_threads,
};
use tenferro_linalg::{EagerSessionLinalgExt, TracedTensorLinalgExt};
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{
    DotGeneralConfig, Error, ErrorPhase, ExecutionEngine, GraphCompiler, Runtime, Tensor,
    TraceContext, TracedTensor,
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

/// Extra matmul-only sizes guarding the single small-matmul thread contract
/// (tenferro-rs #1898 follow-up: a 4-thread run must not be slower than 1T).
/// The full profile already covers 32 through the ordinary small sizes.
fn small_matmul_thread_contract_sizes(profile: Profile) -> &'static [usize] {
    match profile {
        Profile::Quick => &[32, 64],
        Profile::Full => &[64],
    }
}

/// Extra `(n, batch)` batched-matmul shapes guarding tenferro-rs #1898
/// (batched small GEMM thread scaling). The full profile already covers
/// n=4, batch=1024 through the ordinary batched grid.
fn batched_matmul_thread_contract_shapes(profile: Profile) -> &'static [(usize, usize)] {
    match profile {
        Profile::Quick => &[(4, 1024)],
        Profile::Full => &[],
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

// Operations use the borrowed session entered outside the sampling clock.
mod eager_einsum_tensor {
    use super::*;
    pub fn einsum(
        session: &mut EagerSession<'_>,
        inputs: &[&EagerTensor],
        _subscripts: &str,
    ) -> Result<EagerTensor, tenferro_ad::Error> {
        static SUBS: OnceLock<tenferro_einsum::EinsumSubscripts> = OnceLock::new();
        let subs = SUBS
            .get_or_init(|| tenferro_einsum::EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]));
        session
            .einsum_subscripts(inputs, subs)
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
        session: &mut EagerSession<'_>,
        a: &EagerTensor,
    ) -> tenferro_ad::error::Result<(EagerTensor, EagerTensor, EagerTensor)> {
        session.svd(a)
    }
    pub fn qr(
        session: &mut EagerSession<'_>,
        a: &EagerTensor,
    ) -> tenferro_ad::error::Result<(EagerTensor, EagerTensor)> {
        session.qr(a)
    }
    pub fn eigh(
        session: &mut EagerSession<'_>,
        a: &EagerTensor,
    ) -> tenferro_ad::error::Result<(EagerTensor, EagerTensor)> {
        session.eigh(a)
    }
    pub fn solve(
        session: &mut EagerSession<'_>,
        a: &EagerTensor,
        b: &EagerTensor,
    ) -> tenferro_ad::error::Result<EagerTensor> {
        session.solve(a, b)
    }
}
mod eager_session_ops {
    use super::*;
    pub fn matmul(
        session: &mut EagerSession<'_>,
        lhs: &EagerTensor,
        rhs: &EagerTensor,
    ) -> tenferro_ad::error::Result<EagerTensor> {
        session.matmul(lhs, rhs)
    }
    pub fn dot_general(
        session: &mut EagerSession<'_>,
        lhs: &EagerTensor,
        rhs: &EagerTensor,
        config: DotGeneralConfig,
    ) -> tenferro_ad::error::Result<EagerTensor> {
        session.dot_general(lhs, rhs, config)
    }
    pub fn reduce_sum(
        session: &mut EagerSession<'_>,
        input: &EagerTensor,
        axes: &[usize],
    ) -> tenferro_ad::error::Result<EagerTensor> {
        session.reduce_sum(input, Some(axes))
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
    } else if cfg!(feature = "blas-openblas") {
        "blas-openblas"
    } else if cfg!(feature = "blas-accelerate") {
        "blas-accelerate"
    } else if cfg!(feature = "blas-mkl") {
        "blas-mkl"
    } else {
        "native"
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
        .and_then(|()| verify_backend_threads("CpuBackend", cpu_engine().num_threads(), requested))
        .and_then(|()| {
            // tenferro-rs #2206: the caller-held engine is the execution scope,
            // so the scope width is the engine's own pool width.
            let scope_threads = cpu_engine().num_threads();
            verify_backend_threads("CpuBackend execution scope", scope_threads, requested)?;
            eprintln!(
                "publication_gate: requested_threads={requested} backend_threads={} scope_rayon_threads={scope_threads} global_rayon_threads={}",
                cpu_engine().num_threads(),
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

    record_cpu_runtime().expect("record effective CPU execution settings");
    let rows = run_all(&config);
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

    for &n in small_matmul_thread_contract_sizes(config.profile) {
        rows.push(bench_row(
            config,
            "small",
            "matmul",
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
            |session, (a, b)| {
                let out = eager_session_ops::matmul(session, &a, &b)?;
                Ok(out)
            },
        ));
    }

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
            |session, (a, b)| {
                let out = eager_session_ops::matmul(session, &a, &b)?;
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
            |session, (a, b)| {
                let out = eager_einsum_tensor::einsum(session, &[&a, &b], "ij,jk->ik")?;
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
            |session, (a,)| {
                let (u, s, vh) = eager_linalg_tensor::svd(session, &a)?;
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
            |session, (a,)| {
                let (q, r) = eager_linalg_tensor::qr(session, &a)?;
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
            |session, (a,)| {
                let (w, v) = eager_linalg_tensor::eigh(session, &a)?;
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
                |session, (a, b)| {
                    let x = eager_linalg_tensor::solve(session, &a, &b)?;
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
            |session, (a, b)| {
                let y = eager_session_ops::matmul(session, &a, &b)?;
                let loss = eager_session_ops::reduce_sum(session, &y, &[0, 1])?;
                // tenferro-rs #2206: the reverse pass needs the engine itself, which the
                // entered session already borrows, so the backward rows stay
                // unsupported (see `bench_row`) and only the forward pass is
                // expressible inside the session.
                Ok((loss, y))
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
            |session, (a,)| {
                let (u, s, vt) = eager_linalg_tensor::svd(session, &a)?;
                let loss = eager_session_ops::reduce_sum(session, &s, &[0])?;
                // tenferro-rs #2206: the reverse pass needs the engine itself, which the
                // entered session already borrows, so the backward rows stay
                // unsupported (see `bench_row`) and only the forward pass is
                // expressible inside the session.
                Ok((loss, u, s, vt))
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
            |session, (a, b)| {
                let x = eager_linalg_tensor::solve(session, &a, &b)?;
                let loss = eager_session_ops::reduce_sum(session, &x, &[0, 1])?;
                // tenferro-rs #2206: the reverse pass needs the engine itself, which the
                // entered session already borrows, so the backward rows stay
                // unsupported (see `bench_row`) and only the forward pass is
                // expressible inside the session.
                Ok((loss, x))
            },
        ));
        push_linalg_jvp_vjp_eager_benches(
            config,
            rows,
            "small",
            &format!("{n}x{n}"),
            n,
            tangent_seed(10),
            LinalgAdLoss::SumSvdS { matrix_seed: 10 },
        );
        push_linalg_jvp_vjp_eager_benches(
            config,
            rows,
            "small",
            &format!("{n}x{n}"),
            n,
            tangent_seed(4),
            LinalgAdLoss::SumQr { matrix_seed: 4 },
        );
        push_linalg_jvp_vjp_eager_benches(
            config,
            rows,
            "small",
            &format!("{n}x{n}"),
            n,
            tangent_seed(5),
            LinalgAdLoss::SumEigh { matrix_seed: 5 },
        );
        push_linalg_jvp_vjp_eager_benches(
            config,
            rows,
            "small",
            &format!("{n}x{n}"),
            n,
            tangent_seed(52),
            LinalgAdLoss::SumLu { matrix_seed: 52 },
        );
        push_linalg_jvp_vjp_eager_benches(
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
            |session, (a, b)| {
                let out = eager_session_ops::matmul(session, &a, &b)?;
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
            |session, (a, b)| {
                let out = eager_session_ops::matmul(session, &a, &b)?;
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
            |session, (a,)| {
                let (u, s, vh) = eager_linalg_tensor::svd(session, &a)?;
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
            |session, (a,)| {
                let (q, r) = eager_linalg_tensor::qr(session, &a)?;
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
            |session, (a,)| {
                let (w, v) = eager_linalg_tensor::eigh(session, &a)?;
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
                |session, (a, b)| {
                    let x = eager_linalg_tensor::solve(session, &a, &b)?;
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
            |session, (a, b)| {
                let y = eager_session_ops::matmul(session, &a, &b)?;
                let loss = eager_session_ops::reduce_sum(session, &y, &[0, 1])?;
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
            |session, (a, b)| {
                let y = eager_session_ops::matmul(session, &a, &b)?;
                let loss = eager_session_ops::reduce_sum(session, &y, &[0, 1])?;
                // tenferro-rs #2206: the reverse pass needs the engine itself, which the
                // entered session already borrows, so the backward rows stay
                // unsupported (see `bench_row`) and only the forward pass is
                // expressible inside the session.
                Ok((loss, y))
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
            |session, (a,)| {
                let (u, s, vt) = eager_linalg_tensor::svd(session, &a)?;
                let loss = eager_session_ops::reduce_sum(session, &s, &[0])?;
                // tenferro-rs #2206: the reverse pass needs the engine itself, which the
                // entered session already borrows, so the backward rows stay
                // unsupported (see `bench_row`) and only the forward pass is
                // expressible inside the session.
                Ok((loss, u, s, vt))
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
            |session, (a, b)| {
                let x = eager_linalg_tensor::solve(session, &a, &b)?;
                let loss = eager_session_ops::reduce_sum(session, &x, &[0, 1])?;
                // tenferro-rs #2206: the reverse pass needs the engine itself, which the
                // entered session already borrows, so the backward rows stay
                // unsupported (see `bench_row`) and only the forward pass is
                // expressible inside the session.
                Ok((loss, x))
            },
        ));
    }
    // The linalg AD cases use the trace arm's size grid so that both arms of
    // `cpu/linalg_jvp_vjp` record the same shapes.
    for &n in large_linalg_jvp_vjp_sizes(config.profile) {
        push_linalg_jvp_vjp_eager_benches(
            config,
            rows,
            "large",
            &format!("{n}x{n}"),
            n,
            tangent_seed(34),
            LinalgAdLoss::SumSvdS { matrix_seed: 34 },
        );
        push_linalg_jvp_vjp_eager_benches(
            config,
            rows,
            "large",
            &format!("{n}x{n}"),
            n,
            tangent_seed(26),
            LinalgAdLoss::SumQr { matrix_seed: 26 },
        );
        push_linalg_jvp_vjp_eager_benches(
            config,
            rows,
            "large",
            &format!("{n}x{n}"),
            n,
            tangent_seed(27),
            LinalgAdLoss::SumEigh { matrix_seed: 27 },
        );
        push_linalg_jvp_vjp_eager_benches(
            config,
            rows,
            "large",
            &format!("{n}x{n}"),
            n,
            tangent_seed(54),
            LinalgAdLoss::SumLu { matrix_seed: 54 },
        );
        push_linalg_jvp_vjp_eager_benches(
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

fn run_batched_small(config: &BenchConfig, rows: &mut Vec<Row>) {
    let batches: &[usize] = match config.profile {
        Profile::Quick => &[16, 64],
        Profile::Full => &[16, 64, 256, 1024],
    };
    let sizes: &[usize] = match config.profile {
        Profile::Quick => &[2, 4],
        Profile::Full => &[2, 4, 8, 16],
    };

    for &(n, b) in batched_matmul_thread_contract_shapes(config.profile) {
        rows.push(bench_row(
            config,
            "batched",
            "batched_matmul_ikb_kjb_ijb",
            "primal",
            "f64",
            &format!("{n}x{n}xbatch{b} (native batch layout)"),
            || {
                let ctx = cpu_ctx();
                let a = eager_from_tensor_in(
                    tensor(&[n, n, b], data_for_shape(&[n, n, b], 41)),
                    ctx.clone(),
                );
                let rhs =
                    eager_from_tensor_in(tensor(&[n, n, b], data_for_shape(&[n, n, b], 42)), ctx);
                Ok((a, rhs, Some(batched_matmul_config())))
            },
            |session, (a, rhs, config)| {
                let out = eager_session_ops::dot_general(
                    session,
                    &a,
                    &rhs,
                    config.take().expect("prepared config"),
                )?;
                Ok(out)
            },
        ));
    }

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
                |session, (a, rhs, config)| {
                    let out = eager_session_ops::dot_general(
                        session,
                        &a,
                        &rhs,
                        config.take().expect("prepared config"),
                    )?;
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
                |session, (a,)| {
                    let (u, s, vh) = eager_linalg_tensor::svd(session, &a)?;
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
                |session, (a,)| {
                    let (q, r) = eager_linalg_tensor::qr(session, &a)?;
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
                |session, (a,)| {
                    let (w, v) = eager_linalg_tensor::eigh(session, &a)?;
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
                |session, (a, rhs)| {
                    let x = eager_linalg_tensor::solve(session, &a, &rhs)?;
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
                |session, (a, rhs, config)| {
                    let out = eager_session_ops::dot_general(
                        session,
                        &a,
                        &rhs,
                        config.take().expect("prepared config"),
                    )?;
                    let loss = eager_session_ops::reduce_sum(session, &out, &[0, 1, 2])?;
                    // tenferro-rs #2206: the reverse pass needs the engine itself, which the
                    // entered session already borrows, so the backward rows stay
                    // unsupported (see `bench_row`) and only the forward pass is
                    // expressible inside the session.
                    Ok((loss, out))
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
                |session, (a, rhs)| {
                    let x = eager_linalg_tensor::solve(session, &a, &rhs)?;
                    let loss = eager_session_ops::reduce_sum(session, &x, &[0, 1, 2])?;
                    // tenferro-rs #2206: the reverse pass needs the engine itself, which the
                    // entered session already borrows, so the backward rows stay
                    // unsupported (see `bench_row`) and only the forward pass is
                    // expressible inside the session.
                    Ok((loss, x))
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

    for &n in small_matmul_thread_contract_sizes(config.profile) {
        rows.push(bench_trace_row(
            config,
            "small",
            "matmul",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            |trace| {
                let a = traced_tensor(trace, &[n, n], data_for_shape(&[n, n], 1));
                let b = traced_tensor(trace, &[n, n], data_for_shape(&[n, n], 2));
                Ok(vec![traced_tensor::matmul(&a, &b)])
            },
        ));
    }

    for &n in sizes {
        rows.push(bench_trace_row(
            config,
            "small",
            "matmul",
            "primal",
            "f64",
            &format!("{n}x{n}"),
            |trace| {
                let a = traced_tensor(trace, &[n, n], data_for_shape(&[n, n], 1));
                let b = traced_tensor(trace, &[n, n], data_for_shape(&[n, n], 2));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], well_conditioned_matrix(n, 3));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], well_conditioned_matrix(n, 4));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], spd_matrix(n, 5));
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
                |trace| {
                    let a = traced_tensor(trace, &[n, n], spd_matrix(n, 6));
                    let b = traced_tensor(trace, &[n, rhs_cols], data_for_shape(&[n, rhs_cols], 7));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], data_for_shape(&[n, n], 8));
                let b = traced_tensor(trace, &[n, n], data_for_shape(&[n, n], 9));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], well_conditioned_matrix(n, 10));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], spd_matrix(n, 11));
                let b = traced_tensor(trace, &[n, 1], data_for_shape(&[n, 1], 12));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], data_for_shape(&[n, n], 21));
                let b = traced_tensor(trace, &[n, n], data_for_shape(&[n, n], 22));
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
            |trace| {
                let a = traced_tensor(trace, &[m, k], data_for_shape(&[m, k], 23));
                let b = traced_tensor(trace, &[k, n], data_for_shape(&[k, n], 24));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], well_conditioned_matrix(n, 25));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], well_conditioned_matrix(n, 26));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], spd_matrix(n, 27));
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
                |trace| {
                    let a = traced_tensor(trace, &[n, n], spd_matrix(n, 28));
                    let b =
                        traced_tensor(trace, &[n, rhs_cols], data_for_shape(&[n, rhs_cols], 29));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], data_for_shape(&[n, n], 30));
                let b = traced_tensor(trace, &[n, n], data_for_shape(&[n, n], 31));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], data_for_shape(&[n, n], 32));
                let b = traced_tensor(trace, &[n, n], data_for_shape(&[n, n], 33));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], well_conditioned_matrix(n, 34));
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
            |trace| {
                let a = traced_tensor(trace, &[n, n], spd_matrix(n, 35));
                let b = traced_tensor(trace, &[n, 1], data_for_shape(&[n, 1], 36));
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

    for &(n, b) in batched_matmul_thread_contract_shapes(config.profile) {
        rows.push(bench_trace_row(
            config,
            "batched",
            "batched_matmul_ikb_kjb_ijb",
            "primal",
            "f64",
            &format!("{n}x{n}xbatch{b} (native batch layout)"),
            |trace| {
                let a = traced_tensor(trace, &[n, n, b], data_for_shape(&[n, n, b], 41));
                let rhs = traced_tensor(trace, &[n, n, b], data_for_shape(&[n, n, b], 42));
                Ok(vec![a.dot_general(&rhs, batched_matmul_config())?])
            },
        ));
    }

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
                |trace| {
                    let a = traced_tensor(trace, &[n, n, b], data_for_shape(&[n, n, b], 41));
                    let rhs = traced_tensor(trace, &[n, n, b], data_for_shape(&[n, n, b], 42));
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
                |trace| {
                    let a = traced_tensor(trace, &[n, n, b], batched_well_conditioned(n, b, 43));
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
                |trace| {
                    let a = traced_tensor(trace, &[n, n, b], batched_well_conditioned(n, b, 44));
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
                |trace| {
                    let a = traced_tensor(trace, &[n, n, b], batched_spd(n, b, 45));
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
                |trace| {
                    let a = traced_tensor(trace, &[n, n, b], batched_spd(n, b, 46));
                    let rhs = traced_tensor(trace, &[n, 1, b], data_for_shape(&[n, 1, b], 47));
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
                |trace| {
                    let a = traced_tensor(trace, &[n, n, b], data_for_shape(&[n, n, b], 48));
                    let rhs = traced_tensor(trace, &[n, n, b], data_for_shape(&[n, n, b], 49));
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
                |trace| {
                    let a = traced_tensor(trace, &[n, n, b], batched_spd(n, b, 50));
                    let rhs = traced_tensor(trace, &[n, 1, b], data_for_shape(&[n, 1, b], 51));
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

// Primal fixtures are read-only and reusable. Prepare them before entering
// the eager session; tensor wrapping itself may enter a backend session.
fn measure_eager<I: Clone + Send + Sync, O>(
    config: &BenchConfig,
    shape: &str,
    mut setup: impl FnMut() -> Result<I, Error>,
    mut execute: impl FnMut(&mut EagerSession<'_>, &mut I) -> Result<O, Error> + Send,
) -> Result<Measurement, Error> {
    let estimated_bytes = retention_estimate(shape);
    let cap = (RETENTION_BUDGET_BYTES / estimated_bytes).clamp(1, 65_536);
    let ctx = cpu_ctx();
    // One caller-held engine for every sample of this case (tenferro-rs #2206).
    let mut engine = cpu_engine();
    let mut inputs = vec![setup()?];
    let mut run = |inputs: &mut [I], samples: usize| {
        ctx.with_eager_session(&mut *engine, |session| {
            let mut times = Vec::with_capacity(samples);
            for _ in 0..samples {
                // Restore consumed descriptors and handles before this sample's timer.
                let mut sample_inputs = inputs.to_vec();
                let mut outputs = Vec::with_capacity(sample_inputs.len());
                let start = Instant::now();
                for input in &mut sample_inputs {
                    outputs.push(execute(session, input)?);
                }
                times.push(start.elapsed());
                black_box(&outputs);
                // Output destruction is outside the clock, while fixtures persist.
            }
            Ok::<_, Error>(times)
        })
    };
    run(&mut inputs, config.warmups.max(1))?;
    let calibration_ns = loop {
        let elapsed = run(&mut inputs, 1)?[0].as_nanos();
        if elapsed >= SAMPLE_TARGET_NS || inputs.len() == cap {
            break elapsed;
        }
        let next = (inputs.len() * 2).min(cap);
        while inputs.len() < next {
            inputs.push(setup()?);
        }
    };
    let operations = inputs.len();
    // One borrowed session encloses all measured batches, outside their clocks.
    let times = run(&mut inputs, config.runs)?;
    Ok(Measurement {
        operations,
        estimated_bytes,
        calibration_ns,
        times,
    })
}

// Eager AD samples cannot share one borrowed session the way the primal rows do:
// `EagerRuntime::jvp`/`vjp` take the caller-held engine themselves (tenferro-rs
// #2206), so one sample is one forward pass in a borrowed session plus one AD
// call. That call also carries the eager side's AD preparation, which the public
// API does not expose separately; the trace arm prepares outside the clock (see
// `docs/cpu-suite-arms.md`).
fn measure_eager_ad<I, O>(
    config: &BenchConfig,
    shape: &str,
    setup: impl FnMut() -> Result<I, Error>,
    mut execute: impl FnMut(
        &Arc<EagerRuntime>,
        &mut dyn ExecutionEngine,
        &mut I,
    ) -> tenferro_ad::error::Result<O>,
) -> Result<Measurement, Error> {
    let ctx = cpu_ctx();
    let mut engine = cpu_engine();
    measure(config, shape, setup, |input| {
        Ok(execute(&ctx, &mut *engine, input)?)
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
    eager_session_policy: &str,
) -> Result<(f64, f64), Error> {
    record_raw(serde_json::json!({
        "record_type": "samples", "suite": suite,
        "benchmark": benchmark_name(op, phase), "shape": shape,
        // Callers may still hold the engine lock here, so do not take it again;
        // enforce_threads_or_exit already verified the engine width matches.
        "backend": backend, "threads": requested_threads(),
        "operations_per_sample": measured.operations,
        "batch_elapsed_ns": measured.times.iter().map(Duration::as_nanos).collect::<Vec<_>>(),
        "per_operation_ns": measured.times.iter().map(|t| t.as_nanos() as f64 / measured.operations as f64).collect::<Vec<_>>(),
        "target_ns": SAMPLE_TARGET_NS, "calibration_ns": measured.calibration_ns,
        "estimated_retained_bytes_per_operation": measured.estimated_bytes,
        "retention_budget_bytes": RETENTION_BUDGET_BYTES,
        "execution_scope": "shared_cpu", "eager_session_policy": eager_session_policy, "output_policy": "allocation_returning_retained_until_stop",
    }))?;
    let (median, iqr) = median_iqr_ms(measured.times);
    Ok((
        median / measured.operations as f64,
        iqr / measured.operations as f64,
    ))
}

/// The one caller-held CPU engine of this process (tenferro-rs #2206).
///
/// The engine owns the pool, so the process holds it in a lock and borrows it
/// as `&mut` for one case at a time instead of rebuilding it per iteration.
fn cpu_backend() -> &'static Mutex<CpuEngine> {
    static BACKEND: OnceLock<Mutex<CpuEngine>> = OnceLock::new();
    BACKEND.get_or_init(|| {
        Mutex::new(
            tenferro_einsum_benchmark::cpu_engine(Some(requested_threads()))
                .expect("configure explicit CPU thread count"),
        )
    })
}

fn cpu_engine() -> MutexGuard<'static, CpuEngine> {
    cpu_backend()
        .lock()
        .expect("the publication-gate CPU engine lock is not poisoned")
}

fn record_cpu_runtime() -> Result<(), Error> {
    #[cfg(feature = "blas-mkl")]
    let provider_threads = {
        extern "C" {
            fn MKL_Get_Max_Threads() -> i32;
        }
        // SAFETY: blas-mkl links the oneMKL C ABI; this argument-free query
        // only reads the calling thread's effective provider configuration.
        let threads = unsafe { MKL_Get_Max_Threads() };
        assert_eq!(
            threads as usize,
            cpu_engine().num_threads(),
            "MKL and CPU thread budgets differ"
        );
        Some(threads)
    };
    #[cfg(not(feature = "blas-mkl"))]
    let provider_threads: Option<i32> = None;
    let affinity = std::fs::read_to_string("/proc/thread-self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("Cpus_allowed_list:"))
                .map(str::to_owned)
        });
    let backend = cpu_engine();
    record_raw(serde_json::json!({
        "record_type": "runtime", "backend": "tenferro", "provider": backend_name(),
        "thread_budget": backend.num_threads(), "worker_count": backend.num_threads(),
        "provider_max_threads": provider_threads, "scope_worker_affinity": affinity,
        "cpu_provider": tenferro_cpu::cpu_provider_id(),
    }))
}

fn bench_row<I: Clone + Send + Sync, O>(
    config: &BenchConfig,
    suite: &'static str,
    op: &'static str,
    phase: &'static str,
    dtype: &'static str,
    shape: &str,
    mut setup: impl FnMut() -> tenferro_ad::error::Result<I> + Send,
    mut execute: impl FnMut(&mut EagerSession<'_>, &mut I) -> tenferro_ad::error::Result<O> + Send,
) -> Row {
    if !benchmark_filter_matches(op, phase) {
        return filtered_row(suite, op, phase, dtype, shape);
    }

    if phase == "backward" {
        return Row { suite, op, phase, dtype, shape: shape.to_string(), backend: config.backend,
            median_ms: None, iqr_ms: None,
            status: "unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled".into() };
    }
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let measured = measure_eager(config, shape, &mut setup, &mut execute)?;
        summarize_measurement(
            measured,
            suite,
            op,
            phase,
            shape,
            "tenferro-eager",
            "borrowed_session_outside_timing_for_primal",
        )
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

/// One eager AD row: `jvp` or `vjp` measured by `measure_eager_ad`.
fn bench_eager_ad_row<I, O>(
    config: &BenchConfig,
    suite: &'static str,
    op: &'static str,
    phase: &'static str,
    dtype: &'static str,
    shape: &str,
    mut setup: impl FnMut() -> Result<I, Error>,
    mut execute: impl FnMut(
        &Arc<EagerRuntime>,
        &mut dyn ExecutionEngine,
        &mut I,
    ) -> tenferro_ad::error::Result<O>,
) -> Row {
    if !benchmark_filter_matches(op, phase) {
        return filtered_row(suite, op, phase, dtype, shape);
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let measured = measure_eager_ad(config, shape, &mut setup, &mut execute)?;
        // Unlike a primal row, the forward pass is inside this sample, and so is
        // the eager API's own AD preparation (tenferro-rs #2206 exposes no
        // separate prepare step). The record says that instead of claiming the
        // primal-only policy above.
        summarize_measurement(
            measured,
            suite,
            op,
            phase,
            shape,
            "tenferro-eager",
            "forward_and_derivative_inside_timing_with_eager_ad_prepare",
        )
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
    mut build: impl FnMut(&TraceContext) -> tenferro_ad::error::Result<Vec<TracedTensor>>,
) -> Row {
    if !benchmark_filter_matches(op, phase) {
        return filtered_row(suite, op, phase, dtype, shape);
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        // Every traced value of one row belongs to this row's context.
        let trace = TraceContext::new();
        let outputs = build(&trace)?;
        let output_refs: Vec<&TracedTensor> = outputs.iter().collect();
        let mut compiler = GraphCompiler::new();
        let program = compiler.compile_many(&output_refs)?;
        let runtime = cpu_runtime_with_extensions()?;
        let mut engine = cpu_engine();
        let target = ExecutionEngine::target(&*engine);
        let prepared = runtime.prepare(&program, &[], std::slice::from_ref(&target))?;
        let measured = measure(
            config,
            shape,
            || Ok(()),
            |_| runtime.run(&mut [&mut *engine], &prepared, &[]),
        )?;
        summarize_measurement(
            measured,
            suite,
            op,
            phase,
            shape,
            "tenferro-trace",
            "borrowed_session_outside_timing_for_trace_program",
        )
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
    let backend = cpu_engine();
    let engine_id = runtime_engine_id().map_err(|err| Error::Internal(err.to_string()))?;
    let mut builder = Runtime::builder();
    builder
        .register_engine(
            runtime_engine_registration(&backend)
                .map_err(|err| Error::Internal(err.to_string()))?,
        )
        .map_err(|err| Error::Internal(err.to_string()))?;
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<CpuEngine>(engine_id.clone())
                .map_err(|err| Error::Internal(err.to_string()))?,
        )
        .map_err(|err| Error::Internal(err.to_string()))?;
    builder
        .install_extension_module(
            tenferro_linalg::extension_module::<CpuEngine>(engine_id)
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
        let mut engine = cpu_engine();
        let target = ExecutionEngine::target(&*engine);
        let prepared = runtime.prepare(&program, &[], std::slice::from_ref(&target))?;
        let measured = measure(
            config,
            &shape,
            || Ok(()),
            |_| runtime.run(&mut [&mut *engine], &prepared, &[]),
        )?;
        summarize_measurement(
            measured,
            suite,
            op,
            phase,
            &shape,
            "tenferro-trace",
            "borrowed_session_outside_timing_for_trace_program",
        )
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
        // The runtime stores no engine (tenferro-rs #2206); every eager call
        // borrows the process's caller-held CPU engine.
        EagerRuntime::with_ad_context(ad_context())
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

fn scalar_one(trace: &TraceContext) -> TracedTensor {
    traced_tensor(trace, &[], vec![1.0])
}

/// The eager fixtures of one AD case: the differentiated matrix, plus the fixed
/// right-hand side that `solve` needs.
struct EagerLinalgAdFixture {
    wrt: EagerTensor,
    rhs: Option<EagerTensor>,
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

    fn build(
        self,
        trace: &TraceContext,
        n: usize,
    ) -> tenferro_ad::error::Result<(TracedTensor, TracedTensor)> {
        match self {
            Self::SumSvdS { matrix_seed } => {
                let a = traced_tensor(trace, &[n, n], well_conditioned_matrix(n, matrix_seed));
                let (_, s, _) = a.svd()?;
                Ok((s.reduce_sum(Some(&[0]))?, a))
            }
            Self::SumQr { matrix_seed } => {
                let a = traced_tensor(trace, &[n, n], well_conditioned_matrix(n, matrix_seed));
                let (q, r) = a.qr()?;
                let loss = (&q.reduce_sum(Some(&[0, 1]))? + &r.reduce_sum(Some(&[0, 1]))?)?;
                Ok((loss, a))
            }
            Self::SumEigh { matrix_seed } => {
                let a = traced_tensor(trace, &[n, n], spd_matrix(n, matrix_seed));
                let (w, _) = a.eigh()?;
                Ok((w.reduce_sum(Some(&[0]))?, a))
            }
            Self::SumLu { matrix_seed } => {
                let a = traced_tensor(trace, &[n, n], well_conditioned_matrix(n, matrix_seed));
                let (_, l, u, _) = a.lu()?;
                let loss = (&l.reduce_sum(Some(&[0, 1]))? + &u.reduce_sum(Some(&[0, 1]))?)?;
                Ok((loss, a))
            }
            Self::SumSolveWrtA {
                matrix_seed,
                rhs_seed,
                rhs_cols,
            } => {
                let a = traced_tensor(trace, &[n, n], spd_matrix(n, matrix_seed));
                let b = traced_tensor(
                    trace,
                    &[n, rhs_cols],
                    data_for_shape(&[n, rhs_cols], rhs_seed),
                );
                let x = a.solve(&b)?;
                Ok((x.reduce_sum(Some(&[0, 1]))?, a))
            }
        }
    }

    /// Build the eager inputs of this case. Tensor wrapping may enter a backend
    /// session, so this happens before a sample's clock.
    fn eager_fixture(self, n: usize) -> EagerLinalgAdFixture {
        let matrix = |data: Vec<f64>| eager_requires_grad_in(tensor(&[n, n], data), cpu_ctx());
        match self {
            Self::SumSvdS { matrix_seed }
            | Self::SumQr { matrix_seed }
            | Self::SumLu { matrix_seed } => EagerLinalgAdFixture {
                wrt: matrix(well_conditioned_matrix(n, matrix_seed)),
                rhs: None,
            },
            Self::SumEigh { matrix_seed } => EagerLinalgAdFixture {
                wrt: matrix(spd_matrix(n, matrix_seed)),
                rhs: None,
            },
            Self::SumSolveWrtA {
                matrix_seed,
                rhs_seed,
                rhs_cols,
            } => EagerLinalgAdFixture {
                wrt: matrix(spd_matrix(n, matrix_seed)),
                rhs: Some(eager_from_tensor_in(
                    tensor(&[n, rhs_cols], data_for_shape(&[n, rhs_cols], rhs_seed)),
                    cpu_ctx(),
                )),
            },
        }
    }

    /// The same loss as [`Self::build`], computed by one borrowed eager session.
    fn eager_forward(
        self,
        session: &mut EagerSession<'_>,
        fixture: &EagerLinalgAdFixture,
    ) -> tenferro_ad::error::Result<EagerTensor> {
        match self {
            Self::SumSvdS { .. } => {
                let (_, s, _) = session.svd(&fixture.wrt)?;
                session.reduce_sum(&s, Some(&[0]))
            }
            Self::SumQr { .. } => {
                let (q, r) = session.qr(&fixture.wrt)?;
                let q_sum = session.reduce_sum(&q, Some(&[0, 1]))?;
                let r_sum = session.reduce_sum(&r, Some(&[0, 1]))?;
                session.add(&q_sum, &r_sum)
            }
            Self::SumEigh { .. } => {
                let (w, _) = session.eigh(&fixture.wrt)?;
                session.reduce_sum(&w, Some(&[0]))
            }
            Self::SumLu { .. } => {
                let (_, l, u, _) = session.lu(&fixture.wrt)?;
                let l_sum = session.reduce_sum(&l, Some(&[0, 1]))?;
                let u_sum = session.reduce_sum(&u, Some(&[0, 1]))?;
                session.add(&l_sum, &u_sum)
            }
            Self::SumSolveWrtA { .. } => {
                let rhs = fixture
                    .rhs
                    .as_ref()
                    .expect("the solve AD case carries a right-hand side");
                let x = session.solve(&fixture.wrt, rhs)?;
                session.reduce_sum(&x, Some(&[0, 1]))
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
        |trace| {
            let (output, wrt) = loss.build(trace, n)?;
            let tangent = traced_tensor(trace, &[n, n], data_for_shape(&[n, n], tangent_seed));
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
        |trace| {
            let (output, wrt) = loss.build(trace, n)?;
            let derivative = vjp(&output, &wrt, &scalar_one(trace))?;
            Ok(vec![output, derivative])
        },
    ));
}

/// Register one linalg AD case on the eager arm under the trace arm's case ids.
///
/// `cpu/linalg_jvp_vjp` is derived from the CPU-ops CSV rows named `_jvp`/`_vjp`,
/// so the ids must match [`push_linalg_jvp_vjp_trace_benches`] exactly for both
/// arms to share a row.
fn push_linalg_jvp_vjp_eager_benches(
    config: &BenchConfig,
    rows: &mut Vec<Row>,
    suite: &'static str,
    shape: &str,
    n: usize,
    tangent_seed: u64,
    loss: LinalgAdLoss,
) {
    let op = loss.op_name();
    rows.push(bench_eager_ad_row(
        config,
        suite,
        op,
        "jvp",
        "f64",
        shape,
        || {
            Ok((
                loss.eager_fixture(n),
                eager_from_tensor_in(
                    tensor(&[n, n], data_for_shape(&[n, n], tangent_seed)),
                    cpu_ctx(),
                ),
            ))
        },
        |ctx, engine, (fixture, tangent)| {
            let output = ctx.with_eager_session(engine, |s| loss.eager_forward(s, fixture))?;
            let derivative = ctx.jvp(engine, &output, &fixture.wrt, tangent)?;
            Ok((output, derivative))
        },
    ));
    rows.push(bench_eager_ad_row(
        config,
        suite,
        op,
        "vjp",
        "f64",
        shape,
        || {
            Ok((
                loss.eager_fixture(n),
                eager_from_tensor_in(tensor(&[], vec![1.0]), cpu_ctx()),
            ))
        },
        |ctx, engine, (fixture, cotangent)| {
            let output = ctx.with_eager_session(engine, |s| loss.eager_forward(s, fixture))?;
            let derivative = ctx.vjp(engine, &output, &fixture.wrt, cotangent)?;
            Ok((output, derivative))
        },
    ));
}

fn tensor(shape: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::from_typed(
        TypedTensor::from_vec_col_major(shape.to_vec(), data)
            .expect("benchmark shape/data length should match"),
    )
}

fn traced_tensor(trace: &TraceContext, shape: &[usize], data: Vec<f64>) -> TracedTensor {
    trace
        .from_tensor_concrete_shape(tensor(shape, data))
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

/// The `spd` fixture: `BᵀB + I` for the well-conditioned `B`.
///
/// One BLAS-shaped GEMM instead of a triple loop: `BᵀB` is 2n³ flops, and the
/// batched rows build one matrix per batch element. See `spd_from_base`.
fn spd_matrix(n: usize, seed: u64) -> Vec<f64> {
    spd_from_base(&well_conditioned_matrix(n, seed), n)
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
        lhs_contracting_dims: [1].into_iter().collect(),
        rhs_contracting_dims: [0].into_iter().collect(),
        lhs_batch_dims: [2].into_iter().collect(),
        rhs_batch_dims: [2].into_iter().collect(),
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
    // These tests share the benchmark runtime and its live gradient slots.
    static EAGER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn eager_runtime_is_reused_across_samples() {
        assert!(Arc::ptr_eq(&cpu_ctx(), &cpu_ctx()));
    }

    /// Set the CPU-ops case filter for one test and restore it afterwards, so a
    /// panicking test cannot leave the process-wide filter set.
    struct CaseFilter(Option<std::ffi::OsString>);

    impl CaseFilter {
        fn only(filter: &str) -> Self {
            let previous = env::var_os("CPU_OPS_BENCHMARK_FILTER");
            env::set_var("CPU_OPS_BENCHMARK_FILTER", filter);
            Self(previous)
        }
    }

    impl Drop for CaseFilter {
        fn drop(&mut self) {
            match self.0.take() {
                Some(previous) => env::set_var("CPU_OPS_BENCHMARK_FILTER", previous),
                None => env::remove_var("CPU_OPS_BENCHMARK_FILTER"),
            }
        }
    }

    const LINALG_AD_CASES: [&str; 10] = [
        "grad_sum_svd_s_jvp",
        "grad_sum_svd_s_vjp",
        "grad_sum_qr_jvp",
        "grad_sum_qr_vjp",
        "grad_sum_eigh_jvp",
        "grad_sum_eigh_vjp",
        "grad_sum_lu_jvp",
        "grad_sum_lu_vjp",
        "grad_sum_solve_jvp",
        "grad_sum_solve_vjp",
    ];

    fn linalg_ad_config(suite: SuiteFilter, eager: bool, trace: bool) -> BenchConfig {
        BenchConfig {
            profile: Profile::Quick,
            suite,
            runs: 1,
            warmups: 1,
            backend: backend_name(),
            include_eager: eager,
            include_trace: trace,
        }
    }

    /// Every linalg AD JVP/VJP case runs on the eager arm as well as the trace
    /// arm, under the same case id (tenferro-benchmark #307).
    ///
    /// The case ids are derived from the CPU-ops CSV rows named `_jvp`/`_vjp`, so a
    /// lost eager registration would shrink `cpu/linalg_jvp_vjp` back to one arm
    /// without any other test noticing.
    #[test]
    fn linalg_ad_jvp_vjp_cases_are_registered_on_both_arms() {
        let _guard = EAGER_TEST_LOCK.lock().unwrap();
        // Measure only the linalg AD cases; every other small case is filtered.
        let _filter = CaseFilter::only(&LINALG_AD_CASES.join(","));
        let rows = run_all(&linalg_ad_config(SuiteFilter::Small, true, true));

        let mut ids: std::collections::BTreeMap<&str, std::collections::BTreeSet<String>> =
            std::collections::BTreeMap::new();
        for row in &rows {
            if row.phase != "jvp" && row.phase != "vjp" {
                continue;
            }
            let arm = if row.backend == "tenferro-trace" {
                "tenferro-trace"
            } else {
                "tenferro-eager"
            };
            let id = benchmark_name(row.op, row.phase);
            assert_eq!(row.status, "ok", "{arm} {id}: {}", row.status);
            ids.entry(arm).or_default().insert(id);
        }
        for arm in ["tenferro-eager", "tenferro-trace"] {
            let registered = ids.remove(arm).unwrap_or_default();
            assert_eq!(
                registered,
                LINALG_AD_CASES.iter().map(|id| (*id).to_string()).collect(),
                "{arm} linalg AD cases"
            );
        }
    }

    /// The large linalg AD cases are registered from the eager driver, and on the
    /// trace arm's size grid (`large_linalg_jvp_vjp_sizes`): off-grid shapes would
    /// never join, and a registration in the trace-gated driver would vanish in a
    /// `PUBLICATION_GATE_TENFERRO_MODE=eager` run.
    #[test]
    fn linalg_ad_large_cases_share_the_trace_size_grid() {
        let _guard = EAGER_TEST_LOCK.lock().unwrap();
        // No case matches, so both arms register their rows without measuring;
        // a filtered row keeps its case id and shape.
        let _filter = CaseFilter::only("__no_case_matches__");
        let eager_only = run_all(&linalg_ad_config(SuiteFilter::Large, true, false));
        let trace_only = run_all(&linalg_ad_config(SuiteFilter::Large, false, true));

        let shapes = |rows: &[Row]| {
            let mut shapes: std::collections::BTreeMap<(String, String), usize> =
                std::collections::BTreeMap::new();
            for row in rows {
                if row.phase == "jvp" || row.phase == "vjp" {
                    *shapes
                        .entry((benchmark_name(row.op, row.phase), row.shape.clone()))
                        .or_default() += 1;
                }
            }
            shapes
        };
        let mut expected = std::collections::BTreeMap::new();
        for id in LINALG_AD_CASES {
            for &n in large_linalg_jvp_vjp_sizes(Profile::Quick) {
                let shape = if id.starts_with("grad_sum_solve") {
                    format!("{n}x{n},rhs=1")
                } else {
                    format!("{n}x{n}")
                };
                expected.insert((id.to_string(), shape), 1);
            }
        }
        assert_eq!(shapes(&eager_only), expected, "eager-only run");
        assert_eq!(shapes(&trace_only), expected, "trace-only run");
        assert_eq!(eager_only.len(), trace_only.len());
    }

    #[test]
    fn primal_uses_borrowed_session_and_backward_is_explicitly_unsupported() {
        let _guard = EAGER_TEST_LOCK.lock().unwrap();
        let config = BenchConfig {
            profile: Profile::Quick,
            suite: SuiteFilter::Small,
            runs: 1,
            warmups: 0,
            backend: backend_name(),
            include_eager: true,
            include_trace: false,
        };
        let row = bench_row(
            &config,
            "small",
            "matmul",
            "primal",
            "f64",
            "2x2",
            || {
                Ok((eager_from_tensor_in(
                    tensor(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]),
                    cpu_ctx(),
                ),))
            },
            |session, (a,)| session.matmul(a, a),
        );
        assert_eq!(row.status, "ok");
        let row = bench_row(
            &config,
            "batched",
            "batched_matmul_ikb_kjb_ijb",
            "primal",
            "f64",
            "2x2xbatch1",
            || {
                let a =
                    eager_from_tensor_in(tensor(&[2, 2, 1], vec![1.0, 0.0, 0.0, 1.0]), cpu_ctx());
                Ok((a, Some(batched_matmul_config())))
            },
            |session, (a, descriptor)| {
                session.dot_general(
                    a,
                    a,
                    descriptor
                        .take()
                        .expect("descriptor restored outside timing"),
                )
            },
        );
        assert_eq!(row.status, "ok");
        let row = bench_row(
            &config,
            "small",
            "grad_sum_matmul",
            "backward",
            "f64",
            "2x2",
            || -> Result<(), Error> { panic!("unsupported operation must not prepare inputs") },
            |_session, _input| -> Result<(), Error> {
                panic!("unsupported operation must not execute")
            },
        );
        assert!(row.status.starts_with("unsupported:"));
        assert!(row.median_ms.is_none());
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
        let _guard = EAGER_TEST_LOCK.lock().unwrap();
        let ctx = cpu_ctx();
        let runtime = cpu_runtime_with_extensions().unwrap();
        // One caller-held engine for the whole reference check.
        let mut engine = cpu_engine();
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
            let output = ctx
                .with_eager_session(&mut *engine, |session| {
                    eager_session_ops::dot_general(session, &x, &y, batched_matmul_config())
                })
                .unwrap();
            assert_eq!(output.shape(), shape);
            check(
                output
                    .to_tensor(&mut *engine)
                    .unwrap()
                    .as_slice::<f64>()
                    .unwrap(),
                &expected,
            );
            ctx.with_eager_session(&mut *engine, |session| session.reduce_sum(&output, None))
                .unwrap()
                .backward(&mut *engine)
                .unwrap();
            check(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &da);
            check(y.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &db);

            let trace = TraceContext::new();
            let x = traced_tensor(&trace, &shape, a);
            let y = traced_tensor(&trace, &shape, b);
            let output = x.dot_general(&y, batched_matmul_config()).unwrap();
            let loss = output.reduce_sum(None).unwrap();
            let gx = grad(&loss, &x).unwrap();
            let gy = grad(&loss, &y).unwrap();
            let program = GraphCompiler::new()
                .compile_many(&[&output, &gx, &gy])
                .unwrap();
            let target = ExecutionEngine::target(&*engine);
            let prepared = runtime
                .prepare(&program, &[], std::slice::from_ref(&target))
                .unwrap();
            let outputs = runtime.run(&mut [&mut *engine], &prepared, &[]).unwrap();
            assert_eq!(outputs.len(), 3);
            for (actual, expected) in outputs.into_iter().zip([&expected, &da, &db]) {
                let actual = actual.into_tensor(&mut *engine).unwrap();
                check(actual.as_slice::<f64>().unwrap(), expected);
            }
        }
    }
}
