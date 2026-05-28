//! Publication-gate microbenchmarks for tenferro-rs.
//!
//! This runner covers the focused benchmark matrix from tensor4all/tenferro-rs#862:
//! small matrix latency, large matrix throughput, and batched small matrices.

use std::env;
use std::hint::black_box;
use std::panic;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use tenferro_ad::{AdContext, EagerRuntime, EagerTensor};
use tenferro_einsum::eager_tensor as eager_einsum_tensor;
use tenferro_linalg::eager_tensor as eager_linalg_tensor;
use tenferro_runtime::{
    traced_tensor, CpuBackend, DotGeneralConfig, Error, GraphCompiler, GraphExecutor, Tensor,
    TracedTensor, TypedTensor,
};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SuiteFilter {
    All,
    Small,
    Large,
    Batched,
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
    } else {
        "cpu-faer"
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

fn main() {
    let config = BenchConfig::from_env();
    println!("suite,op,phase,dtype,backend,profile,shape,warmups,runs,median_ms,iqr_ms,status");

    for row in run_all(&config) {
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
                let a = EagerTensor::from_tensor_in(a, ctx.clone());
                let b = EagerTensor::from_tensor_in(b, ctx);
                let out = a.matmul(&b)?;
                black_box(out.data());
                Ok(())
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
                let a = EagerTensor::from_tensor_in(
                    tensor(&[n, n], data_for_shape(&[n, n], 1)),
                    ctx.clone(),
                );
                let b =
                    EagerTensor::from_tensor_in(tensor(&[n, n], data_for_shape(&[n, n], 2)), ctx);
                let out = eager_einsum_tensor::einsum(&[&a, &b], "ij,jk->ik")?;
                black_box(out.data());
                Ok(())
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
                let a = EagerTensor::from_tensor_in(
                    tensor(&[n, n], well_conditioned_matrix(n, 3)),
                    cpu_ctx(),
                );
                let (u, s, vh) = eager_linalg_tensor::svd(&a)?;
                black_box((u.data(), s.data(), vh.data()));
                Ok(())
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
                let a = EagerTensor::from_tensor_in(
                    tensor(&[n, n], well_conditioned_matrix(n, 4)),
                    cpu_ctx(),
                );
                let (q, r) = eager_linalg_tensor::qr(&a)?;
                black_box((q.data(), r.data()));
                Ok(())
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
                let a = EagerTensor::from_tensor_in(tensor(&[n, n], spd_matrix(n, 5)), cpu_ctx());
                let (w, v) = eager_linalg_tensor::eigh(&a)?;
                black_box((w.data(), v.data()));
                Ok(())
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
                    let a =
                        EagerTensor::from_tensor_in(tensor(&[n, n], spd_matrix(n, 6)), ctx.clone());
                    let b = EagerTensor::from_tensor_in(
                        tensor(&[n, rhs_cols], data_for_shape(&[n, rhs_cols], 7)),
                        ctx,
                    );
                    let x = eager_linalg_tensor::solve(&a, &b)?;
                    black_box(x.data());
                    Ok(())
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
                let a = EagerTensor::requires_grad_in(
                    tensor(&[n, n], data_for_shape(&[n, n], 8)),
                    ctx.clone(),
                );
                let b =
                    EagerTensor::requires_grad_in(tensor(&[n, n], data_for_shape(&[n, n], 9)), ctx);
                let y = a.matmul(&b)?;
                let loss = y.reduce_sum(&[0, 1])?;
                let _ = loss.backward()?;
                black_box((a.grad(), b.grad()));
                Ok(())
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
                let a = EagerTensor::requires_grad_in(
                    tensor(&[n, n], well_conditioned_matrix(n, 10)),
                    cpu_ctx(),
                );
                let (_, s, _) = eager_linalg_tensor::svd(&a)?;
                let loss = s.reduce_sum(&[0])?;
                let _ = loss.backward()?;
                black_box(a.grad());
                Ok(())
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
                let a =
                    EagerTensor::requires_grad_in(tensor(&[n, n], spd_matrix(n, 11)), ctx.clone());
                let b = EagerTensor::requires_grad_in(
                    tensor(&[n, 1], data_for_shape(&[n, 1], 12)),
                    ctx,
                );
                let x = eager_linalg_tensor::solve(&a, &b)?;
                let loss = x.reduce_sum(&[0, 1])?;
                let _ = loss.backward()?;
                black_box((a.grad(), b.grad()));
                Ok(())
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
                let a = EagerTensor::from_tensor_in(
                    tensor(&[n, n], data_for_shape(&[n, n], 21)),
                    ctx.clone(),
                );
                let b =
                    EagerTensor::from_tensor_in(tensor(&[n, n], data_for_shape(&[n, n], 22)), ctx);
                let out = a.matmul(&b)?;
                black_box(out.data());
                Ok(())
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
                let a = EagerTensor::from_tensor_in(
                    tensor(&[m, k], data_for_shape(&[m, k], 23)),
                    ctx.clone(),
                );
                let b =
                    EagerTensor::from_tensor_in(tensor(&[k, n], data_for_shape(&[k, n], 24)), ctx);
                let out = a.matmul(&b)?;
                black_box(out.data());
                Ok(())
            },
        ));
    }

    let linalg_sizes: &[usize] = match config.profile {
        Profile::Quick => &[64],
        Profile::Full => &[64, 128, 256],
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
                let a = EagerTensor::from_tensor_in(
                    tensor(&[n, n], well_conditioned_matrix(n, 25)),
                    cpu_ctx(),
                );
                let (u, s, vh) = eager_linalg_tensor::svd(&a)?;
                black_box((u.data(), s.data(), vh.data()));
                Ok(())
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
                let a = EagerTensor::from_tensor_in(
                    tensor(&[n, n], well_conditioned_matrix(n, 26)),
                    cpu_ctx(),
                );
                let (q, r) = eager_linalg_tensor::qr(&a)?;
                black_box((q.data(), r.data()));
                Ok(())
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
                let a = EagerTensor::from_tensor_in(tensor(&[n, n], spd_matrix(n, 27)), cpu_ctx());
                let (w, v) = eager_linalg_tensor::eigh(&a)?;
                black_box((w.data(), v.data()));
                Ok(())
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
                    let a = EagerTensor::from_tensor_in(
                        tensor(&[n, n], spd_matrix(n, 28)),
                        ctx.clone(),
                    );
                    let b = EagerTensor::from_tensor_in(
                        tensor(&[n, rhs_cols], data_for_shape(&[n, rhs_cols], 29)),
                        ctx,
                    );
                    let x = eager_linalg_tensor::solve(&a, &b)?;
                    black_box(x.data());
                    Ok(())
                },
            ));
        }
    }

    for &n in match config.profile {
        Profile::Quick => &[64][..],
        Profile::Full => &[64, 128][..],
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
                let a = EagerTensor::requires_grad_in(
                    tensor(&[n, n], data_for_shape(&[n, n], 30)),
                    ctx.clone(),
                );
                let b = EagerTensor::requires_grad_in(
                    tensor(&[n, n], data_for_shape(&[n, n], 31)),
                    ctx,
                );
                let y = a.matmul(&b)?;
                let loss = y.reduce_sum(&[0, 1])?;
                black_box(loss.data());
                Ok(())
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
                let a = EagerTensor::requires_grad_in(
                    tensor(&[n, n], data_for_shape(&[n, n], 32)),
                    ctx.clone(),
                );
                let b = EagerTensor::requires_grad_in(
                    tensor(&[n, n], data_for_shape(&[n, n], 33)),
                    ctx,
                );
                let y = a.matmul(&b)?;
                let loss = y.reduce_sum(&[0, 1])?;
                let _ = loss.backward()?;
                black_box((a.grad(), b.grad()));
                Ok(())
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
                let a = EagerTensor::requires_grad_in(
                    tensor(&[n, n], well_conditioned_matrix(n, 34)),
                    cpu_ctx(),
                );
                let (_, s, _) = eager_linalg_tensor::svd(&a)?;
                let loss = s.reduce_sum(&[0])?;
                let _ = loss.backward()?;
                black_box(a.grad());
                Ok(())
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
                let a =
                    EagerTensor::requires_grad_in(tensor(&[n, n], spd_matrix(n, 35)), ctx.clone());
                let b = EagerTensor::requires_grad_in(
                    tensor(&[n, 1], data_for_shape(&[n, 1], 36)),
                    ctx,
                );
                let x = eager_linalg_tensor::solve(&a, &b)?;
                let loss = x.reduce_sum(&[0, 1])?;
                let _ = loss.backward()?;
                black_box((a.grad(), b.grad()));
                Ok(())
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
                    let a = EagerTensor::from_tensor_in(
                        tensor(&[n, n, b], data_for_shape(&[n, n, b], 41)),
                        ctx.clone(),
                    );
                    let rhs = EagerTensor::from_tensor_in(
                        tensor(&[n, n, b], data_for_shape(&[n, n, b], 42)),
                        ctx,
                    );
                    let out = a.dot_general(&rhs, batched_matmul_config())?;
                    black_box(out.data());
                    Ok(())
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
                    let a = EagerTensor::from_tensor_in(
                        tensor(&[n, n, b], batched_well_conditioned(n, b, 43)),
                        cpu_ctx(),
                    );
                    let (u, s, vh) = eager_linalg_tensor::svd(&a)?;
                    black_box((u.data(), s.data(), vh.data()));
                    Ok(())
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
                    let a = EagerTensor::from_tensor_in(
                        tensor(&[n, n, b], batched_well_conditioned(n, b, 44)),
                        cpu_ctx(),
                    );
                    let (q, r) = eager_linalg_tensor::qr(&a)?;
                    black_box((q.data(), r.data()));
                    Ok(())
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
                    let a = EagerTensor::from_tensor_in(
                        tensor(&[n, n, b], batched_spd(n, b, 45)),
                        cpu_ctx(),
                    );
                    let (w, v) = eager_linalg_tensor::eigh(&a)?;
                    black_box((w.data(), v.data()));
                    Ok(())
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
                    let a = EagerTensor::from_tensor_in(
                        tensor(&[n, n, b], batched_spd(n, b, 46)),
                        ctx.clone(),
                    );
                    let rhs = EagerTensor::from_tensor_in(
                        tensor(&[n, 1, b], data_for_shape(&[n, 1, b], 47)),
                        ctx,
                    );
                    let x = eager_linalg_tensor::solve(&a, &rhs)?;
                    black_box(x.data());
                    Ok(())
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
                    let a = EagerTensor::requires_grad_in(
                        tensor(&[n, n, b], data_for_shape(&[n, n, b], 48)),
                        ctx.clone(),
                    );
                    let rhs = EagerTensor::requires_grad_in(
                        tensor(&[n, n, b], data_for_shape(&[n, n, b], 49)),
                        ctx,
                    );
                    let out = a.dot_general(&rhs, batched_matmul_config())?;
                    let loss = out.reduce_sum(&[0, 1, 2])?;
                    let _ = loss.backward()?;
                    black_box((a.grad(), rhs.grad()));
                    Ok(())
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
                    let a = EagerTensor::requires_grad_in(
                        tensor(&[n, n, b], batched_spd(n, b, 50)),
                        ctx.clone(),
                    );
                    let rhs = EagerTensor::requires_grad_in(
                        tensor(&[n, 1, b], data_for_shape(&[n, 1, b], 51)),
                        ctx,
                    );
                    let x = eager_linalg_tensor::solve(&a, &rhs)?;
                    let loss = x.reduce_sum(&[0, 1, 2])?;
                    let _ = loss.backward()?;
                    black_box((a.grad(), rhs.grad()));
                    Ok(())
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
        rows.push(bench_trace_row(config, "small", "matmul", "primal", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], data_for_shape(&[n, n], 1));
            let b = traced_tensor(&[n, n], data_for_shape(&[n, n], 2));
            Ok(vec![traced_tensor::matmul(&a, &b)])
        }));
        rows.push(bench_trace_row(config, "small", "einsum_ij_jk_ik", "primal", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], data_for_shape(&[n, n], 1));
            let b = traced_tensor(&[n, n], data_for_shape(&[n, n], 2));
            let mut compiler = GraphCompiler::new();
            Ok(vec![tenferro_einsum::einsum(&mut compiler, &[&a, &b], "ij,jk->ik")?])
        }));
        rows.push(bench_trace_row(config, "small", "svd", "primal", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], well_conditioned_matrix(n, 3));
            let (u, s, vh) = tenferro_linalg::svd(&a)?;
            Ok(vec![u, s, vh])
        }));
        rows.push(bench_trace_row(config, "small", "qr", "primal", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], well_conditioned_matrix(n, 4));
            let (q, r) = tenferro_linalg::qr(&a)?;
            Ok(vec![q, r])
        }));
        rows.push(bench_trace_row(config, "small", "eigh", "primal", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], spd_matrix(n, 5));
            let (w, v) = tenferro_linalg::eigh(&a)?;
            Ok(vec![w, v])
        }));
        for &rhs_cols in &[1, 4] {
            rows.push(bench_trace_row(config, "small", "solve", "primal", "f64", &format!("{n}x{n},rhs={rhs_cols}"), || {
                let a = traced_tensor(&[n, n], spd_matrix(n, 6));
                let b = traced_tensor(&[n, rhs_cols], data_for_shape(&[n, rhs_cols], 7));
                Ok(vec![tenferro_linalg::solve(&a, &b)?])
            }));
        }
        rows.push(bench_trace_row(config, "small", "grad_sum_matmul", "backward", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], data_for_shape(&[n, n], 8));
            let b = traced_tensor(&[n, n], data_for_shape(&[n, n], 9));
            let y = traced_tensor::matmul(&a, &b);
            let loss = y.reduce_sum(&[0, 1]);
            Ok(vec![grad(&loss, &a)?, grad(&loss, &b)?])
        }));
        rows.push(bench_trace_row(config, "small", "grad_sum_svd_s", "backward", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], well_conditioned_matrix(n, 10));
            let (_, s, _) = tenferro_linalg::svd(&a)?;
            let loss = s.reduce_sum(&[0]);
            Ok(vec![grad(&loss, &a)?])
        }));
        rows.push(bench_trace_row(config, "small", "grad_sum_solve", "backward", "f64", &format!("{n}x{n},rhs=1"), || {
            let a = traced_tensor(&[n, n], spd_matrix(n, 11));
            let b = traced_tensor(&[n, 1], data_for_shape(&[n, 1], 12));
            let x = tenferro_linalg::solve(&a, &b)?;
            let loss = x.reduce_sum(&[0, 1]);
            Ok(vec![grad(&loss, &a)?, grad(&loss, &b)?])
        }));
    }
}

fn run_large_throughput_trace(config: &BenchConfig, rows: &mut Vec<Row>) {
    let matmul_sizes: &[usize] = match config.profile {
        Profile::Quick => &[128, 256],
        Profile::Full => &[128, 256, 512, 1024],
    };
    for &n in matmul_sizes {
        rows.push(bench_trace_row(config, "large", "matmul", "primal", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], data_for_shape(&[n, n], 21));
            let b = traced_tensor(&[n, n], data_for_shape(&[n, n], 22));
            Ok(vec![traced_tensor::matmul(&a, &b)])
        }));
    }
    for &(m, k, n) in &[(1024, 256, 1024), (256, 1024, 256)] {
        if config.profile == Profile::Quick && m > 256 {
            continue;
        }
        rows.push(bench_trace_row(config, "large", "matmul_rect", "primal", "f64", &format!("{m}x{k} * {k}x{n}"), || {
            let a = traced_tensor(&[m, k], data_for_shape(&[m, k], 23));
            let b = traced_tensor(&[k, n], data_for_shape(&[k, n], 24));
            Ok(vec![traced_tensor::matmul(&a, &b)])
        }));
    }
    let linalg_sizes: &[usize] = match config.profile {
        Profile::Quick => &[64],
        Profile::Full => &[64, 128, 256],
    };
    for &n in linalg_sizes {
        rows.push(bench_trace_row(config, "large", "svd", "primal", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], well_conditioned_matrix(n, 25));
            let (u, s, vh) = tenferro_linalg::svd(&a)?;
            Ok(vec![u, s, vh])
        }));
        rows.push(bench_trace_row(config, "large", "qr", "primal", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], well_conditioned_matrix(n, 26));
            let (q, r) = tenferro_linalg::qr(&a)?;
            Ok(vec![q, r])
        }));
        rows.push(bench_trace_row(config, "large", "eigh", "primal", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], spd_matrix(n, 27));
            let (w, v) = tenferro_linalg::eigh(&a)?;
            Ok(vec![w, v])
        }));
        for &rhs_cols in &[1, 16, 64] {
            rows.push(bench_trace_row(config, "large", "solve", "primal", "f64", &format!("{n}x{n},rhs={rhs_cols}"), || {
                let a = traced_tensor(&[n, n], spd_matrix(n, 28));
                let b = traced_tensor(&[n, rhs_cols], data_for_shape(&[n, rhs_cols], 29));
                Ok(vec![tenferro_linalg::solve(&a, &b)?])
            }));
        }
    }
    for &n in match config.profile {
        Profile::Quick => &[64][..],
        Profile::Full => &[64, 128][..],
    } {
        rows.push(bench_trace_row(config, "large", "grad_sum_matmul", "primal", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], data_for_shape(&[n, n], 30));
            let b = traced_tensor(&[n, n], data_for_shape(&[n, n], 31));
            Ok(vec![traced_tensor::matmul(&a, &b).reduce_sum(&[0, 1])])
        }));
        rows.push(bench_trace_row(config, "large", "grad_sum_matmul", "backward", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], data_for_shape(&[n, n], 32));
            let b = traced_tensor(&[n, n], data_for_shape(&[n, n], 33));
            let loss = traced_tensor::matmul(&a, &b).reduce_sum(&[0, 1]);
            Ok(vec![grad(&loss, &a)?, grad(&loss, &b)?])
        }));
        rows.push(bench_trace_row(config, "large", "grad_sum_svd_s", "backward", "f64", &format!("{n}x{n}"), || {
            let a = traced_tensor(&[n, n], well_conditioned_matrix(n, 34));
            let (_, s, _) = tenferro_linalg::svd(&a)?;
            let loss = s.reduce_sum(&[0]);
            Ok(vec![grad(&loss, &a)?])
        }));
        rows.push(bench_trace_row(config, "large", "grad_sum_solve", "backward", "f64", &format!("{n}x{n},rhs=1"), || {
            let a = traced_tensor(&[n, n], spd_matrix(n, 35));
            let b = traced_tensor(&[n, 1], data_for_shape(&[n, 1], 36));
            let loss = tenferro_linalg::solve(&a, &b)?.reduce_sum(&[0, 1]);
            Ok(vec![grad(&loss, &a)?, grad(&loss, &b)?])
        }));
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
            rows.push(bench_trace_row(config, "batched", "batched_matmul_ikb_kjb_ijb", "primal", "f64", &shape, || {
                let a = traced_tensor(&[n, n, b], data_for_shape(&[n, n, b], 41));
                let rhs = traced_tensor(&[n, n, b], data_for_shape(&[n, n, b], 42));
                Ok(vec![a.dot_general(&rhs, batched_matmul_config())])
            }));
            rows.push(bench_trace_row(config, "batched", "batched_svd", "primal", "f64", &shape, || {
                let a = traced_tensor(&[n, n, b], batched_well_conditioned(n, b, 43));
                let (u, s, vh) = tenferro_linalg::svd(&a)?;
                Ok(vec![u, s, vh])
            }));
            rows.push(bench_trace_row(config, "batched", "batched_qr", "primal", "f64", &shape, || {
                let a = traced_tensor(&[n, n, b], batched_well_conditioned(n, b, 44));
                let (q, r) = tenferro_linalg::qr(&a)?;
                Ok(vec![q, r])
            }));
            rows.push(bench_trace_row(config, "batched", "batched_eigh", "primal", "f64", &shape, || {
                let a = traced_tensor(&[n, n, b], batched_spd(n, b, 45));
                let (w, v) = tenferro_linalg::eigh(&a)?;
                Ok(vec![w, v])
            }));
            rows.push(bench_trace_row(config, "batched", "batched_solve", "primal", "f64", &format!("{shape},rhs=1"), || {
                let a = traced_tensor(&[n, n, b], batched_spd(n, b, 46));
                let rhs = traced_tensor(&[n, 1, b], data_for_shape(&[n, 1, b], 47));
                Ok(vec![tenferro_linalg::solve(&a, &rhs)?])
            }));
            rows.push(bench_trace_row(config, "batched", "grad_sum_batched_matmul", "backward", "f64", &shape, || {
                let a = traced_tensor(&[n, n, b], data_for_shape(&[n, n, b], 48));
                let rhs = traced_tensor(&[n, n, b], data_for_shape(&[n, n, b], 49));
                let loss = a.dot_general(&rhs, batched_matmul_config()).reduce_sum(&[0, 1, 2]);
                Ok(vec![grad(&loss, &a)?, grad(&loss, &rhs)?])
            }));
            rows.push(bench_trace_row(config, "batched", "grad_sum_batched_solve", "backward", "f64", &format!("{shape},rhs=1"), || {
                let a = traced_tensor(&[n, n, b], batched_spd(n, b, 50));
                let rhs = traced_tensor(&[n, 1, b], data_for_shape(&[n, 1, b], 51));
                let loss = tenferro_linalg::solve(&a, &rhs)?.reduce_sum(&[0, 1, 2]);
                Ok(vec![grad(&loss, &a)?, grad(&loss, &rhs)?])
            }));
        }
    }
}

fn bench_row(
    config: &BenchConfig,
    suite: &'static str,
    op: &'static str,
    phase: &'static str,
    dtype: &'static str,
    shape: &str,
    mut f: impl FnMut() -> tenferro_ad::error::Result<()>,
) -> Row {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        for _ in 0..config.warmups {
            f()?;
        }
        let mut times = Vec::with_capacity(config.runs);
        for _ in 0..config.runs {
            let start = Instant::now();
            f()?;
            times.push(start.elapsed());
        }
        Ok::<_, Error>(times)
    }));

    match result {
        Ok(Ok(times)) => {
            let (median, iqr) = median_iqr_ms(times);
            Row {
                suite,
                op,
                phase,
                dtype,
                shape: shape.to_string(),
                backend: config.backend,
                median_ms: Some(median),
                iqr_ms: Some(iqr),
                status: "ok".to_string(),
            }
        }
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
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let outputs = build()?;
        let output_refs: Vec<&TracedTensor> = outputs.iter().collect();
        let mut compiler = GraphCompiler::new();
        let program = compiler.compile_many(&output_refs)?;
        let mut executor = GraphExecutor::new(CpuBackend::new());
        executor
            .register_extension(tenferro_einsum::register_runtime)
            .map_err(|err| Error::Internal(err.to_string()))?;
        executor
            .register_extension(tenferro_linalg::register_runtime)
            .map_err(|err| Error::Internal(err.to_string()))?;

        for _ in 0..config.warmups {
            let out = executor.run_many(&program)?;
            black_box(out.len());
        }
        let mut times = Vec::with_capacity(config.runs);
        for _ in 0..config.runs {
            let start = Instant::now();
            let out = executor.run_many(&program)?;
            black_box(out.len());
            times.push(start.elapsed());
        }
        Ok::<_, Error>(times)
    }));

    match result {
        Ok(Ok(times)) => {
            let (median, iqr) = median_iqr_ms(times);
            Row {
                suite,
                op,
                phase,
                dtype,
                shape: shape.to_string(),
                backend: "tenferro-trace",
                median_ms: Some(median),
                iqr_ms: Some(iqr),
                status: "ok".to_string(),
            }
        }
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

fn cpu_ctx() -> Arc<EagerRuntime> {
    EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), ad_context())
}

fn ad_context() -> &'static AdContext {
    static AD_CONTEXT: OnceLock<AdContext> = OnceLock::new();
    AD_CONTEXT.get_or_init(|| {
        AdContext::builder()
            .with_core_rules()
            .with_extension_rules(
                tenferro_linalg::ad_rules().expect("tenferro-linalg AD rules should register"),
            )
            .build()
            .expect("tenferro AD context should build")
    })
}

fn grad(output: &TracedTensor, wrt: &TracedTensor) -> tenferro_ad::error::Result<TracedTensor> {
    ad_context().grad(output, wrt)
}

fn tensor(shape: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape.to_vec(), data))
}

fn traced_tensor(shape: &[usize], data: Vec<f64>) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(tensor(shape, data))
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
