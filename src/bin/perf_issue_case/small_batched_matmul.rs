//! tenferro-rs #2745 / tenferro-benchmark #543: many independent small matrix
//! products through the batched contract.
//!
//! `C[i, j, b] = sum_k A[i, k, b] * B[k, j, b]` on column-major `[N, N, batch]`
//! operands, as one `dot_general` with the last axis as the batch dimension.
//! Two arms run in this one binary under the one timing policy of
//! `perf_issue_case`:
//!
//! * `tenferro`: `Engine::dot_general_read_into_accum` into a preallocated
//!   output, with `beta = 0` (`overwrite`) or `beta = 1` (`add_to`).
//! * `direct-loop`: the naive fixed-size loop a user could write without the
//!   library. `N` is a const generic, one thread, safe Rust, the same
//!   preallocated output and the same update mode. It is a reference cell, not a
//!   competitor.
//!
//! Both arms are checked against a third, size-generic scalar implementation
//! before timing, so neither arm vouches for the other.
use super::*;
use std::ops::{Add, Mul};
use tenferro_tensor::{DotGeneralAccumulation, TensorScalar};

trait Elem: TensorScalar + Add<Output = Self> + Mul<Output = Self> + Default + PartialEq {
    const DTYPE: DType;
    fn data(n: usize, seed: u64) -> Vec<Self>;
    fn rel_err(actual: Self, expected: Self) -> f64;
}

impl Elem for f64 {
    const DTYPE: DType = DType::F64;
    fn data(n: usize, seed: u64) -> Vec<Self> {
        values(n, seed)
    }
    fn rel_err(actual: Self, expected: Self) -> f64 {
        rel_error(actual, expected)
    }
}

impl Elem for Complex64 {
    const DTYPE: DType = DType::C64;
    fn data(n: usize, seed: u64) -> Vec<Self> {
        complex_values(n, seed)
    }
    fn rel_err(actual: Self, expected: Self) -> f64 {
        rel_error_c64(actual, expected)
    }
}

/// The reference cell: a naive fixed-size loop over the batch, column-major
/// `[N, N]` slices, one thread.
#[inline(never)]
fn direct_loop<T: Elem, const N: usize>(a: &[T], b: &[T], c: &mut [T], accumulate: bool) {
    for ((a, b), c) in a
        .chunks_exact(N * N)
        .zip(b.chunks_exact(N * N))
        .zip(c.chunks_exact_mut(N * N))
    {
        for j in 0..N {
            for i in 0..N {
                let mut sum = T::default();
                for k in 0..N {
                    sum = sum + a[i + N * k] * b[k + N * j];
                }
                c[i + N * j] = if accumulate { c[i + N * j] + sum } else { sum };
            }
        }
    }
}

/// Size-generic scalar implementation used only to verify the two arms.
fn check_product<T: Elem>(n: usize, a: &[T], b: &[T], c0: &[T], accumulate: bool) -> Vec<T> {
    let mut out = c0.to_vec();
    for (batch, out) in out.chunks_exact_mut(n * n).enumerate() {
        let (a, b) = (&a[batch * n * n..], &b[batch * n * n..]);
        for j in 0..n {
            for i in 0..n {
                let mut sum = T::default();
                for k in 0..n {
                    sum = sum + a[i + n * k] * b[k + n * j];
                }
                out[i + n * j] = if accumulate { out[i + n * j] + sum } else { sum };
            }
        }
    }
    out
}

fn worst_error<T: Elem>(got: &[T], want: &[T], what: &str) -> Result<f64> {
    if got.len() != want.len() {
        return Err(format!("{what}: length {} != {}", got.len(), want.len()).into());
    }
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| fail_closed(T::rel_err(*g, *w)))
        .fold(0.0f64, f64::max);
    check_tol(worst, 1e-12, what)
}

type DirectFn<T> = fn(&[T], &[T], &mut [T], bool);

fn direct_for<T: Elem>(n: usize) -> Result<DirectFn<T>> {
    Ok(match n {
        1 => direct_loop::<T, 1>,
        2 => direct_loop::<T, 2>,
        3 => direct_loop::<T, 3>,
        4 => direct_loop::<T, 4>,
        5 => direct_loop::<T, 5>,
        8 => direct_loop::<T, 8>,
        _ => return Err(format!("no fixed-size direct loop for N={n}").into()),
    })
}

fn run_typed<T: Elem>(p: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    let n = param_usize(p, "n")?;
    let batch = param_usize(p, "batch")?;
    let arm = param_str(p, "arm")?;
    let accumulate = match param_str(p, "mode")? {
        "overwrite" => false,
        "accumulate" => true,
        other => return Err(format!("unknown update mode {other}").into()),
    };
    let len = n * n * batch;
    let (a, b, c0) = (T::data(len, 301), T::data(len, 302), T::data(len, 303));
    let want = check_product(n, &a, &b, &c0, accumulate);
    let shape = vec![n, n, batch];

    let (error, measurement, timer): (f64, Measurement, Vec<&'static str>) = match arm {
        "direct-loop" => {
            let direct = direct_for::<T>(n)?;
            let mut c = c0.clone();
            direct(&a, &b, &mut c, accumulate);
            let error = worst_error(&c, &want, "direct loop")?;
            c.copy_from_slice(&c0);
            let measurement = measure(timing, 1, || {
                direct(black_box(&a), black_box(&b), black_box(&mut c), accumulate);
                Ok(())
            })?;
            (
                error,
                measurement,
                vec!["naive fixed-size loop over the batch into the preallocated output, one thread"],
            )
        }
        "tenferro" => {
            let lhs = Tensor::from_vec_col_major(shape.clone(), a.clone())?;
            let rhs = Tensor::from_vec_col_major(shape.clone(), b.clone())?;
            let mut out = Tensor::from_vec_col_major(shape, c0.clone())?;
            let cfg = DotGeneralConfig {
                lhs_contracting_dims: [1].as_slice().into(),
                rhs_contracting_dims: [0].as_slice().into(),
                lhs_batch_dims: [2].as_slice().into(),
                rhs_batch_dims: [2].as_slice().into(),
            };
            let accumulation = if accumulate {
                DotGeneralAccumulation::add_to(T::DTYPE)?
            } else {
                DotGeneralAccumulation::overwrite(T::DTYPE)?
            };
            let mut backend = tenferro_einsum_benchmark::cpu_engine(Some(threads))?;
            let (error, measurement) =
                backend.with_backend_session(|s| -> Result<(f64, Measurement)> {
                    let one = |s: &mut dyn Engine, out: &mut Tensor| -> Result<()> {
                        Ok(s.dot_general_read_into_accum(
                            TensorRead::from_tensor(&lhs),
                            TensorRead::from_tensor(&rhs),
                            &cfg,
                            accumulation,
                            TensorWrite::from_tensor(out),
                        )?)
                    };
                    one(s, &mut out)?;
                    let error = worst_error(out.as_slice::<T>()?, &want, "tenferro dot_general")?;
                    out = Tensor::from_vec_col_major(vec![n, n, batch], c0.clone())?;
                    let measurement = measure(timing, 1, || one(s, &mut out))?;
                    Ok((error, measurement))
                })??;
            (
                error,
                measurement,
                vec!["Engine::dot_general_read_into_accum into the preallocated output on the borrowed caller-held session"],
            )
        }
        other => return Err(format!("unknown small_batched_matmul arm {other}").into()),
    };
    Ok(CaseOutcome {
        max_rel_error: error,
        measurement: Some(measurement),
        timer,
        outside_timer: vec![
            "backend/session construction",
            "input_construction",
            "correctness_check",
            "output_restore",
        ],
        extra: json!({"n": n, "batch": batch, "products_per_op": batch,
                      "mode": param_str(p, "mode")?}),
    })
}

pub(super) fn run(p: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    match param_str(p, "dtype")? {
        "f64" => run_typed::<f64>(p, timing, threads),
        "c64" => run_typed::<Complex64>(p, timing, threads),
        other => Err(format!("unsupported dtype {other}").into()),
    }
}
