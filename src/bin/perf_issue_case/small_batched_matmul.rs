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
//! (dynamic `N`, k-outermost summation order) before timing. It is a different
//! summation order and indexing path, not a different algorithm; the unit tests
//! below pin it, and the 2x2 and 3x3 fixed-size loops, to hand-computed products;
//! the other sizes are compared with the checker.
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

/// Size-generic scalar implementation used only to verify the two arms: dynamic
/// `N`, k-outermost rank-1 updates into a zeroed block, then the update mode.
fn check_product<T: Elem>(n: usize, a: &[T], b: &[T], c0: &[T], accumulate: bool) -> Vec<T> {
    let mut out = c0.to_vec();
    for (batch, out) in out.chunks_exact_mut(n * n).enumerate() {
        let (a, b) = (&a[batch * n * n..], &b[batch * n * n..]);
        let mut product = vec![T::default(); n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    product[i + n * j] = product[i + n * j] + a[i + n * k] * b[k + n * j];
                }
            }
        }
        for (o, x) in out.iter_mut().zip(product) {
            *o = if accumulate { *o + x } else { x };
        }
    }
    out
}

/// `beta = 0` is `overwrite`, `beta = 1` is `accumulate`; the one parse drives the
/// expected values, the direct loop and the tenferro accumulation.
fn accumulate_flag(mode: &str) -> Result<bool> {
    match mode {
        "overwrite" => Ok(false),
        "accumulate" => Ok(true),
        other => Err(format!("unknown update mode {other}").into()),
    }
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
    let accumulate = accumulate_flag(param_str(p, "mode")?)?;
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
            // Nothing is retained (the output is the preallocated buffer), so the
            // retention bound is irrelevant: pass 1 byte.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_modes_map_to_beta() {
        assert!(!accumulate_flag("overwrite").unwrap());
        assert!(accumulate_flag("accumulate").unwrap());
        assert!(accumulate_flag("add").is_err());
    }

    /// A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] (column-major), AB = [[19, 22], [43, 50]].
    #[test]
    fn checker_and_every_direct_loop_match_a_hand_computed_product() {
        let a = [1.0, 3.0, 2.0, 4.0];
        let b = [5.0, 7.0, 6.0, 8.0];
        let c0 = [100.0, 200.0, 300.0, 400.0];
        let ab = [19.0, 43.0, 22.0, 50.0];
        let acc: Vec<f64> = c0.iter().zip(ab).map(|(c, x)| c + x).collect();
        assert_eq!(check_product(2, &a, &b, &c0, false), ab);
        assert_eq!(check_product(2, &a, &b, &c0, true), acc);
        let mut c = c0;
        direct_for::<f64>(2).unwrap()(&a, &b, &mut c, false);
        assert_eq!(c, ab);
        let mut c = c0;
        direct_for::<f64>(2).unwrap()(&a, &b, &mut c, true);
        assert_eq!(c.to_vec(), acc);
    }

    /// A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], B = [[1, 0, 2], [0, 1, 0], [3, 0, 1]],
    /// AB = [[10, 2, 5], [22, 5, 14], [34, 8, 23]] (column-major below).
    #[test]
    fn three_by_three_matches_a_hand_computed_product() {
        let a = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
        let b = [1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0];
        let ab = [10.0, 22.0, 34.0, 2.0, 5.0, 8.0, 5.0, 14.0, 23.0];
        let c0 = [0.0; 9];
        assert_eq!(check_product(3, &a, &b, &c0, false), ab);
        let mut c = c0;
        direct_for::<f64>(3).unwrap()(&a, &b, &mut c, false);
        assert_eq!(c, ab);
    }

    #[test]
    fn every_fixed_size_loop_matches_the_checker_for_both_scalars_and_modes() {
        for &n in &[1usize, 2, 3, 4, 5, 8] {
            for &accumulate in &[false, true] {
                let len = n * n * 3;
                let (a, b, c0) = (f64::data(len, 1), f64::data(len, 2), f64::data(len, 3));
                let want = check_product(n, &a, &b, &c0, accumulate);
                let mut c = c0.clone();
                direct_for::<f64>(n).unwrap()(&a, &b, &mut c, accumulate);
                worst_error(&c, &want, "f64").unwrap();
                let (a, b, c0) = (
                    Complex64::data(len, 1),
                    Complex64::data(len, 2),
                    Complex64::data(len, 3),
                );
                let want = check_product(n, &a, &b, &c0, accumulate);
                let mut c = c0.clone();
                direct_for::<Complex64>(n).unwrap()(&a, &b, &mut c, accumulate);
                worst_error(&c, &want, "c64").unwrap();
            }
        }
    }
}
