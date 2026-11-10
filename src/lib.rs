pub mod fingerprint;
pub mod tensornetwork;
pub mod thread_enforcement;
pub mod verification_mode;

use std::fmt::Display;

use tenferro_einsum::{ContractionTree, EinsumOptimize, Subscripts, TraceContextEinsumExt};
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{
    CompiledGraph, CompilerOptions, DType, GraphCompiler, OptimizerConfig, TraceContext,
};

/// The compiled tenferro CPU backend as this harness names it in result rows:
/// `faer` for the `native` build, `blas` for a vendor BLAS build. tenferro-rs
/// #2004 made the backend a compile-time choice, so the compiled provider id is
/// the only authority for what a row measured.
pub fn compiled_cpu_provider() -> &'static str {
    if tenferro_cpu::cpu_provider_id() == "tenferro.cpu.blas" {
        "blas"
    } else {
        "faer"
    }
}

/// Number of workers a benchmark process runs on when the caller does not name
/// one: `RAYON_NUM_THREADS`, else the host's available parallelism.
pub fn cpu_thread_count() -> Result<usize, String> {
    match std::env::var("RAYON_NUM_THREADS") {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|err| format!("RAYON_NUM_THREADS={value:?}: {err}")),
        Err(_) => Ok(std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)),
    }
}

/// Build the caller-held CPU engine the harness measures on.
///
/// tenferro-rs #2206 removed backend sessions: the caller owns the engine and
/// passes it as `&mut` to every operation. Build it once per run, outside the
/// timed region, and reuse it across samples. `threads: None` sizes the pool
/// from [`cpu_thread_count`]; the pool is a private caller-owned strided pool,
/// built through tenferro's public placement API.
pub fn cpu_engine(threads: Option<usize>) -> Result<tenferro_cpu::CpuEngine, String> {
    let threads = threads.map_or_else(cpu_thread_count, Ok)?;
    if threads == 0 {
        return Err("requested thread count must be at least 1".into());
    }
    let pool = tenferro_cpu::Pool::builder()
        .threads(threads)
        .build()
        .map_err(|err| format!("build CPU engine pool: {err}"))?;
    Ok(tenferro_cpu::CpuEngine::new(pool))
}

pub struct CompiledEinsum {
    pub program: CompiledGraph,
    pub input_count: usize,
}

pub fn compile_einsum(
    subs: &Subscripts,
    shapes: &[Vec<usize>],
    tree: &ContractionTree,
) -> Result<CompiledEinsum, String> {
    let mut trace = TraceContext::new();
    let inputs = shapes
        .iter()
        .map(|shape| {
            trace.input(ProgramInputSpec::new(
                DType::F64,
                shape.iter().copied().map(Into::into),
            ))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{e}"))?;
    let einsum_subscripts = subs.into();
    let pairs: Vec<(usize, usize)> = (0..tree.step_count())
        .map(|idx| tree.step_pair(idx).expect("step index is in 0..step_count"))
        .collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let owned_tree =
        ContractionTree::from_pairs(subs, &shape_refs, &pairs).map_err(|e| format!("{e}"))?;
    let output = trace
        .einsum_subscripts_with(
            &inputs,
            &einsum_subscripts,
            EinsumOptimize::Tree(owned_tree),
        )
        .map_err(|e| format!("{e}"))?;
    let graph = trace.finish(&[output]).map_err(|e| format!("{e}"))?;
    let mut compiler = GraphCompiler::with_compiler_options(compiler_options_from_env());
    let program = compiler
        .compile_traced_graph(&graph)
        .map_err(|e| format!("{e}"))?;

    Ok(CompiledEinsum {
        program,
        input_count: inputs.len(),
    })
}

fn compiler_options_from_env() -> CompilerOptions {
    let mut optimizer = OptimizerConfig::default();
    if std::env::var("TENFERRO_OPT_DOT_DECOMPOSER")
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    {
        optimizer.dot_decomposer = true;
    }
    CompilerOptions { optimizer }
}

pub fn unwrap_eval_result<T, E>(
    result: std::thread::Result<Result<T, E>>,
    panic_message: &str,
) -> Result<T, String>
where
    E: Display,
{
    match result {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(err)) => Err(err.to_string()),
        Err(_) => Err(panic_message.to_string()),
    }
}

/// The `spd` fixture a GPU arm builds from a well-conditioned base: `BᵀB + I`.
///
/// `BᵀB` is 2n³ flops, so a hand-written triple loop is the wrong shape for the
/// matrices these suites run at GPU sizes: at n=4096 the fixture is 137 GFLOP
/// and cost ~66 s per call, which every arm of a case pays again, and the CPU
/// reference pays again. `faer` runs the same product as a blocked BLAS-shaped
/// GEMM: 0.6 s at n=4096 on 36 workers, 4.9 s on one. Fixture preparation is
/// untimed, so this moves no measurement; it moves wall time off the pod
/// budget, which is what made the n=4096 matrix unaffordable.
///
/// `base` is column-major and `n × n`, the layout the callers' generators
/// already build, so this is the fixture the Python arms build with
/// `base.T @ base + I`. It differs from the loop it replaced only in summation
/// order (max |Δ| ≈ 9e-16 at n=512).
pub fn spd_from_base(base: &[f64], n: usize) -> Vec<f64> {
    let cells = match n.checked_mul(n) {
        Some(cells) if cells == base.len() => cells,
        _ => panic!(
            "the spd base must be a column-major n x n matrix: n={n} needs {} cells, the base has {}",
            n.saturating_mul(n),
            base.len()
        ),
    };
    let mut out = vec![0.0; cells];
    {
        // Both views borrow the vectors directly: no second copy of an n²
        // fixture (134 MB at n=4096) is made to take them. Peak memory still
        // includes the GEMM's own packing scratch (about twice the detected L3)
        // and its workers, which the scalar loop did not allocate.
        let a = faer::MatRef::from_column_major_slice(base, n, n);
        let dst = faer::MatMut::from_column_major_slice_mut(&mut out, n, n);
        faer::linalg::matmul::matmul(
            dst,
            faer::Accum::Replace,
            a.transpose(),
            a,
            1.0f64,
            faer::Par::rayon(0),
        );
    }
    for i in 0..n {
        out[i + n * i] += 1.0;
    }
    out
}

#[cfg(test)]
mod spd_fixture_tests {
    use super::spd_from_base;

    /// The fixture the triple loop in each runner used to build, kept as the
    /// reference the GEMM has to agree with.
    fn reference(base: &[f64], n: usize) -> Vec<f64> {
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

    fn base(n: usize, seed: u64) -> Vec<f64> {
        // The production generator: `0.05 * noise` plus `1 + j/n` on the
        // diagonal, in the same column-major layout.
        let mut data: Vec<f64> = (0..n * n)
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

    #[test]
    fn the_gemm_fixture_is_the_reference_product() {
        for (n, seed) in [(0usize, 1u64), (1, 3), (4, 2), (8, 5), (17, 11)] {
            let base = base(n, seed);
            let expected = reference(&base, n);
            let actual = spd_from_base(&base, n);
            assert_eq!(actual.len(), expected.len(), "n={n}");
            for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "n={n} element {index}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn the_fixture_is_symmetric_with_a_positive_diagonal() {
        for n in [1usize, 5, 16, 33] {
            let fixture = spd_from_base(&base(n, 7), n);
            for j in 0..n {
                for i in 0..n {
                    let (upper, lower) = (fixture[i + n * j], fixture[j + n * i]);
                    assert!(
                        (upper - lower).abs() <= 1e-12 * upper.abs().max(1.0),
                        "n={n}: {upper} vs {lower}"
                    );
                }
                assert!(fixture[j + n * j] > 0.0, "n={n}: diagonal {j} is not positive");
            }
        }
    }

    #[test]
    #[should_panic(expected = "must be a column-major")]
    fn a_base_that_is_not_n_by_n_is_refused() {
        spd_from_base(&[1.0, 0.0, 0.0], 2);
    }

    #[test]
    #[should_panic(expected = "must be a column-major")]
    fn a_dimension_that_overflows_is_refused_by_name() {
        // `n * n` wraps in release; the caller gets the named refusal instead
        // of a slice length that happens to match a wrapped product.
        spd_from_base(&[], usize::MAX);
    }

    #[test]
    fn the_fixture_does_not_depend_on_the_worker_count() {
        // The fixture is untimed, but a benchmark compares rows across thread
        // counts: n=64 is past the small-matrix threshold, so the parallel
        // path is what runs. `Par::rayon(0)` takes the pool's size at call
        // time, so `install` is what changes it.
        let n = 64;
        let base = base(n, 13);
        let mut reference: Option<Vec<f64>> = None;
        for workers in [1usize, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .expect("test thread pool");
            let fixture = pool.install(|| {
                assert_eq!(rayon::current_num_threads(), workers);
                spd_from_base(&base, n)
            });
            match &reference {
                None => reference = Some(fixture),
                Some(want) => {
                    assert_eq!(fixture.len(), want.len(), "workers={workers}");
                    for (index, (got, expected)) in fixture.iter().zip(want).enumerate() {
                        assert!(
                            (got - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                            "workers={workers} element {index}: {got} vs {expected}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod cpu_engine_tests {
    use super::cpu_engine;
    use tenferro_tensor::{Tensor, TensorElementwise, TensorRead};

    #[test]
    fn requested_worker_counts_execute_on_reusable_engine() {
        for threads in [1, 4] {
            let mut engine = cpu_engine(Some(threads)).unwrap();
            assert_eq!(engine.num_threads(), threads);
            let x = Tensor::from_vec_col_major(vec![4096], vec![2.0_f64; 4096]).unwrap();
            for _ in 0..2 {
                let result = engine
                    .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
                    .unwrap();
                assert!(result.as_slice::<f64>().unwrap().iter().all(|&v| v == 4.0));
            }
        }
        assert!(cpu_engine(Some(0)).is_err());
    }
}
