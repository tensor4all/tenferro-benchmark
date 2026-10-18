//! #2178 item 8: allocation-free provider counters. Each case counts the Rust
//! allocations of one public call in its pooled steady state (outputs dropped
//! after each call so the pool checkout is a hit), next to a host loop writing
//! into a caller-owned buffer. Sizes follow the #2178 measurements; a case shows
//! 0 once the providers' allocation-free APIs are pinned.
//!
//! The cpu/overhead_smoke suite (tenferro-benchmark#197) reuses these fixtures
//! one at a time with `timed` set: the same public call is then also timed with
//! the standard retained-output batches, so each row carries ns/op and allocs/op.
use super::*;
use tenferro_tensor::{TensorRead, TypedTensorView};

const REPEATS: usize = 64;

/// Count one public call in its pooled steady state: warm the pool, then count
/// `REPEATS` calls whose outputs are dropped before the next one, so only the
/// call's own planning and metadata allocations remain.
fn pooled_counter<O>(name: &str, bytes: usize, mut op: impl FnMut() -> Result<O>) -> Result<Value> {
    for _ in 0..4 {
        drop(black_box(op()?)); // Explicit lazy initialization and pool warmup.
    }
    let (_, allocations, allocated) = count_allocations(|| {
        for _ in 0..REPEATS {
            drop(black_box(op()?));
        }
        Ok(())
    })?;
    Ok(json!({"op": name, "output_bytes": bytes,
        "host_allocations_per_op": allocations as f64 / REPEATS as f64,
        "host_allocated_bytes_per_op": allocated as f64 / REPEATS as f64,
        "allocated_bytes_over_output_bytes": if bytes == 0 { None } else { Some(allocated as f64 / REPEATS as f64 / bytes as f64) }}))
}

/// The pooled allocator pass, then (for overhead_smoke rows) the standard
/// retained-output timing of the same call.
fn probe<O>(
    name: &str,
    bytes: usize,
    timing: Option<&Timing>,
    mut op: impl FnMut() -> Result<O>,
) -> Result<(Value, Option<Measurement>)> {
    let row = pooled_counter(name, bytes, &mut op)?;
    let measurement = timing.map(|t| measure(t, bytes, &mut op)).transpose()?;
    Ok((row, measurement))
}

fn col_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut step = 1;
    for &d in shape {
        strides.push(step);
        step *= d;
    }
    strides
}

/// Column-major gather plan for a permuted read: output shape and, per output
/// axis, the input stride it walks. Built once, outside every counted call.
struct Gather {
    out_shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Gather {
    fn new(in_shape: &[usize], perm: &[usize]) -> Self {
        let in_strides = col_major_strides(in_shape);
        Gather {
            out_shape: perm.iter().map(|&p| in_shape[p]).collect(),
            strides: perm.iter().map(|&p| in_strides[p]).collect(),
        }
    }

    /// out[idx] = data[sum idx[k] * strides[k]] over the column-major output.
    fn into(&self, data: &[f64], out: &mut [f64]) {
        for (l, slot) in out.iter_mut().enumerate() {
            let (mut rest, mut at) = (l, 0);
            for (&d, &step) in self.out_shape.iter().zip(&self.strides) {
                at += (rest % d) * step;
                rest /= d;
            }
            *slot = data[at];
        }
    }
}

/// Sum over axis 1 of a column-major [d0, d1, d2] block.
fn reduce_axis1_into(data: &[f64], [d0, d1, d2]: [usize; 3], out: &mut [f64]) {
    for k in 0..d2 {
        for i in 0..d0 {
            out[i + d0 * k] = (0..d1).map(|j| data[i + d0 * j + d0 * d1 * k]).sum();
        }
    }
}

/// Concatenate two column-major [d0, d1, d2] blocks along axis 1.
fn concat_axis1_into(x: &[f64], y: &[f64], [d0, d1, d2]: [usize; 3], out: &mut [f64]) {
    for k in 0..d2 {
        for j in 0..2 * d1 {
            for i in 0..d0 {
                let (src, jj) = if j < d1 { (x, j) } else { (y, j - d1) };
                out[i + d0 * j + 2 * d0 * d1 * k] = src[i + d0 * jj + d0 * d1 * k];
            }
        }
    }
}

/// Column-major n x n matrix product.
fn matmul_into(x: &[f64], y: &[f64], n: usize, out: &mut [f64]) {
    for k in 0..n {
        for i in 0..n {
            out[i + n * k] = (0..n).map(|j| x[i + n * j] * y[j + n * k]).sum();
        }
    }
}

struct Fixture {
    label: &'static str,
    shape: Vec<usize>,
}

fn fixtures(op: &str) -> Result<Vec<Fixture>> {
    let f = |label, shape: &[usize]| Fixture {
        label,
        shape: shape.to_vec(),
    };
    Ok(match op {
        "bcast" => vec![
            f("add_rank0_scalar_n16", &[16]),
            f("add_rank0_scalar_8x8x8", &[8, 8, 8]),
        ],
        "permuted_view" => vec![
            f("add_read_permuted_lhs_8x8x8", &[8, 8, 8]),
            f("add_read_identity_view_lhs_8x8x8", &[8, 8, 8]),
        ],
        "permute_r4" => vec![
            f("transpose_r4_4x4x4x4", &[4, 4, 4, 4]),
            f("transpose_r4_8x8x8x8", &[8, 8, 8, 8]),
        ],
        "reduce_axis" => vec![
            f("reduce_sum_axis1_4x4", &[4, 4, 1]),
            f("reduce_sum_axis1_8x8x8", &[8, 8, 8]),
        ],
        "fused" => vec![
            f("run_prepared_mul_add_8x8x8", &[8, 8, 8]),
            f("run_prepared_mul_add_32x32x32", &[32, 32, 32]),
        ],
        "concat" => vec![f("concatenate_axis1_8x8x8", &[8, 8, 8])],
        "einsum_prepared" => vec![
            f("einsum_prepared_execute_into_n2", &[2, 2]),
            f("einsum_prepared_execute_into_n4", &[4, 4]),
        ],
        _ => return Err("unknown alloc counter op".into()),
    })
}

const PERMUTED_VIEW: [usize; 3] = [2, 0, 1];
const PERMUTE_R4: [usize; 4] = [3, 1, 0, 2];

fn perm_for(fx: &Fixture) -> Vec<usize> {
    match fx.label {
        "add_read_permuted_lhs_8x8x8" => PERMUTED_VIEW.to_vec(),
        l if l.starts_with("transpose_r4") => PERMUTE_R4.to_vec(),
        _ => (0..fx.shape.len()).collect(),
    }
}

/// The expected result and the allocation-free host loop for one fixture.
fn host_into(
    op: &str,
    fx: &Fixture,
    gather: &Gather,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    out: &mut [f64],
) {
    match op {
        "bcast" => out.iter_mut().zip(x).for_each(|(o, &a)| *o = a + y[0]),
        "permuted_view" => {
            gather.into(x, out);
            out.iter_mut().zip(y).for_each(|(o, &b)| *o += b);
        }
        "permute_r4" => gather.into(x, out),
        "reduce_axis" => reduce_axis1_into(x, [fx.shape[0], fx.shape[1], fx.shape[2]], out),
        "fused" => out
            .iter_mut()
            .zip(x.iter().zip(y).zip(z))
            .for_each(|(o, ((&a, &b), &c))| *o = a * b + c),
        "concat" => concat_axis1_into(x, y, [fx.shape[0], fx.shape[1], fx.shape[2]], out),
        "einsum_prepared" => matmul_into(x, y, fx.shape[0], out),
        _ => unreachable!(),
    }
}

fn out_len(op: &str, fx: &Fixture) -> usize {
    let n: usize = fx.shape.iter().product();
    match op {
        "reduce_axis" => n / fx.shape[1],
        "concat" => 2 * n,
        _ => n,
    }
}

/// `threads` is the backend thread budget (an explicit `budget` parameter when
/// the case sets one). `fixture` selects one fixture; `timed` also times it.
pub(super) fn run(p: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    let op = param_str(p, "op")?;
    let only = p.get("fixture").and_then(Value::as_str);
    let timing = p
        .get("timed")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        .then_some(timing);
    let entry_diagnostic = op == "fused" && param_str(p, "arm")? == "tenferro";
    if timing.is_some()
        && entry_diagnostic
        && p.get("timing_scope").and_then(Value::as_str)
            != Some("public_path_latency_including_internal_session_entry")
    {
        return Err("run_prepared timing requires explicit session-entry diagnostic scope".into());
    }
    let selected: Vec<_> = fixtures(op)?
        .into_iter()
        .filter(|fx| only.map_or(true, |label| fx.label == label))
        .collect();
    if selected.is_empty() || (timing.is_some() && selected.len() != 1) {
        return Err("a timed alloc counter case selects exactly one known fixture".into());
    }
    let mut measurement = None;
    let mut rows = Vec::new();
    let mut extra = json!({"requested_threads": threads,
        "output_retention": "tenferro: output dropped after each call (pooled checkout reuse); host: caller-owned buffer"});
    if op == "fused" {
        extra["fused_regions"] = json!({});
    }
    for fx in selected {
        let n: usize = fx.shape.iter().product();
        let x = values(n, 120);
        let y = if op == "bcast" {
            values(1, 121)
        } else {
            values(n, 121)
        };
        let z = values(n, 122);
        let gather = Gather::new(&fx.shape, &perm_for(&fx));
        let mut want = vec![0.; out_len(op, &fx)];
        host_into(op, &fx, &gather, &x, &y, &z, &mut want);
        let bytes = want.len() * 8;
        if param_str(p, "arm")? == "host" {
            let mut out = vec![0.; want.len()];
            let (row, m) = probe(&format!("host_loop_{}", fx.label), 0, timing, || {
                host_into(op, &fx, &gather, &x, &y, &z, &mut out);
                Ok(())
            })?;
            measurement = m;
            check(&out, &want)?;
            rows.push(row);
            continue;
        }
        let (row, m) = if op == "fused" {
            let (row, m, regions) = fused(&fx, &x, &y, &z, &want, timing, threads)?;
            extra["fused_regions"][fx.label] = regions;
            (row, m)
        } else {
            session_op(op, &fx, &x, &y, &want, bytes, timing, threads)?
        };
        measurement = m;
        rows.push(row);
    }
    let mut extra = merge(counters(rows), extra);
    extra["counter_source"] = json!(
        "separate Rust global allocator pass over 64 calls after pool warmup, \
        including this process's Rust worker allocations"
    );
    if op == "fused" {
        extra["unavailable"] = json!([
            "foreign BLAS/C allocator calls",
            "pooled-buffer reuse volume",
            "fused-lane allocations when the runtime declines the region (see elementwise_region_execution_counts)"
        ]);
        extra["session_entry"] = json!(
            "run_prepared enters a runtime-internal session per call; its entry allocations are counted"
        );
    }
    if entry_diagnostic {
        extra["timing_scope"] = json!("public_path_latency_including_internal_session_entry");
        extra["diagnostic_timing"] = json!(
            "run_prepared includes runtime-internal session entry/exit; not shared-session operation timing"
        );
    } else {
        extra["timing_scope"] = json!("steady_state_host_api");
    }
    let mut timer = if measurement.is_some() {
        vec![
            "declared public operation (counters[0].op)",
            "intrinsic output allocation or pooled checkout",
        ]
    } else {
        vec![]
    };
    if entry_diagnostic && measurement.is_some() {
        timer.push("runtime-internal session entry and exit (diagnostic)");
    }
    let mut result = outcome(measurement, extra, timer);
    if entry_diagnostic {
        result
            .outside_timer
            .retain(|item| !item.contains("entered held session"));
    }
    Ok(result)
}

fn merge(mut base: Value, extra: Value) -> Value {
    if let (Some(b), Value::Object(e)) = (base.as_object_mut(), extra) {
        b.extend(e);
    }
    base
}

fn session_op(
    op: &str,
    fx: &Fixture,
    x: &[f64],
    y: &[f64],
    want: &[f64],
    bytes: usize,
    timing: Option<&Timing>,
    threads: usize,
) -> Result<(Value, Option<Measurement>)> {
    let a = TypedTensor::<f64>::from_vec_col_major(fx.shape.clone(), x.to_vec())?;
    let rhs_shape: Vec<usize> = match op {
        "bcast" => vec![],
        "permuted_view" => perm_for(fx).iter().map(|&k| fx.shape[k]).collect(),
        _ => fx.shape.clone(),
    };
    let b = TypedTensor::<f64>::from_vec_col_major(rhs_shape.clone(), y.to_vec())?;
    let lhs_view = TypedTensorView::<f64>::from_slice(
        &fx.shape,
        col_major_strides(&fx.shape)
            .iter()
            .map(|&s| s as isize)
            .collect::<Vec<_>>(),
        0,
        a.host_data()?,
    )?
    .transpose_view(perm_for(fx))?;
    let lhs_read = TensorRead::from_view(f64::tensor_view(lhs_view));
    let perm = perm_for(fx);
    let (tx, ty) = (
        Tensor::from_vec_col_major(fx.shape.clone(), x.to_vec())?,
        Tensor::from_vec_col_major(rhs_shape, y.to_vec())?,
    );
    let plan = (op == "einsum_prepared")
        .then(|| ConcreteEinsumPlan::prepare([&tx, &ty], "ij,jk->ik"))
        .transpose()?;
    // The caller-owned destination of the prepared einsum step (n x n output).
    let out_shape = if op == "einsum_prepared" {
        fx.shape.clone()
    } else {
        vec![want.len()]
    };
    let mut out = Tensor::from_vec_col_major(out_shape, vec![0.; want.len()])?;
    let backend = checked_backend(threads)?;
    let mut held = backend.open_session()?;
    let row = held.with_session(|s| -> Result<(Value, Option<Measurement>)> {
        let label = fx.label;
        match op {
            "bcast" => {
                check(a.add(&b, s)?.host_data()?, want)?;
                probe(label, bytes, timing, || Ok(a.add(&b, s)?))
            }
            "permuted_view" => {
                check(
                    data(&s.add_read(lhs_read.clone(), f64::tensor_read(&b))?)?,
                    want,
                )?;
                probe(label, bytes, timing, || {
                    Ok(s.add_read(lhs_read.clone(), f64::tensor_read(&b))?)
                })
            }
            "permute_r4" => {
                check(a.transpose(&perm, s)?.host_data()?, want)?;
                probe(label, bytes, timing, || Ok(a.transpose(&perm, s)?))
            }
            "reduce_axis" => {
                check(a.reduce_sum(Some(&[1]), s)?.host_data()?, want)?;
                probe(label, bytes, timing, || Ok(a.reduce_sum(Some(&[1]), s)?))
            }
            "concat" => {
                let concat = |s: &mut dyn BackendSession| {
                    <Tensor as TensorSessionOpsExt>::concatenate(&[&tx, &ty], 1, s)
                };
                check(data(&concat(s)?)?, want)?;
                probe(label, bytes, timing, || Ok(concat(s)?))
            }
            "einsum_prepared" => {
                let plan = plan.as_ref().ok_or("missing einsum plan")?;
                let mut step = || -> Result<()> {
                    plan.execute_into([&tx, &ty], s, TensorWrite::Tensor(&mut out))?;
                    Ok(())
                };
                step()?;
                let probed = probe(label, 0, timing, &mut step)?;
                drop(step);
                check(data(&out)?, want)?;
                Ok(probed)
            }
            _ => unreachable!(),
        }
    })?;
    held.close()?;
    Ok(row)
}

/// A traced two-op program (a * b + c) run through the public prepared path on
/// a backend with the requested thread budget. The region counters say whether
/// the runtime executed it as one fused command or declined to per-op dispatch.
fn fused(
    fx: &Fixture,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    want: &[f64],
    timing: Option<&Timing>,
    threads: usize,
) -> Result<(Value, Option<Measurement>, Value)> {
    let shape = &fx.shape;
    let inputs: Vec<_> = (0..3)
        .map(|_| TracedTensor::input_concrete_shape(DType::F64, shape))
        .collect::<std::result::Result<_, _>>()?;
    let out = inputs[0].mul(&inputs[1])?.add(&inputs[2])?;
    let specs: Vec<_> = inputs
        .iter()
        .map(|t| (t, DType::F64, shape.as_slice()))
        .collect();
    let program = GraphCompiler::new().compile_with_input_specs(&out, &specs)?;
    let backend = checked_backend(threads)?;
    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration(&backend)?)?;
    let runtime = builder.build()?;
    let tensors = [x, y, z].map(|v| Tensor::from_vec_col_major(shape.clone(), v.to_vec()));
    let [tx, ty, tz] = tensors;
    let (tx, ty, tz) = (tx?, ty?, tz?);
    let args = [&tx, &ty, &tz];
    let prepared = runtime.prepare_compiled(&program, &args)?;
    check(data(&runtime.run_prepared(&prepared, &args)?[0])?, want)?;
    let (row, measurement) = probe(fx.label, want.len() * 8, timing, || {
        Ok(runtime.run_prepared(&prepared, &args)?)
    })?;
    let (fused, fallbacks) = prepared.elementwise_region_execution_counts();
    let (planned, covered) = prepared.elementwise_region_summary();
    Ok((
        row,
        measurement,
        json!({"elementwise_region_summary": {"regions": planned, "instructions": covered},
        "elementwise_region_execution_counts": {"fused": fused, "fallbacks": fallbacks},
        "execution_counts_scope": "cumulative over validation, warmup, the counted calls and any timed batches"}),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_timing_rejects_an_implicit_or_steady_session_scope() {
        let timing = Timing {
            warmups: 0,
            samples: 0,
            target_ns: 1,
        };
        for scope in [None, Some("steady_state_host_api")] {
            let mut params = json!({"op": "fused", "arm": "tenferro", "timed": true});
            if let Some(scope) = scope {
                params["timing_scope"] = json!(scope);
            }
            let error = run(&params, &timing, 1)
                .err()
                .expect("must reject standard timing");
            assert!(
                error
                    .to_string()
                    .contains("explicit session-entry diagnostic")
            );
        }
    }

    #[test]
    fn runtime_diagnostic_records_entry_inside_the_timer() {
        let timing = Timing {
            warmups: 0,
            samples: 0,
            target_ns: 1,
        };
        let params = json!({"op": "fused", "arm": "tenferro", "timed": true,
            "fixture": "run_prepared_mul_add_8x8x8",
            "timing_scope": "public_path_latency_including_internal_session_entry"});
        let result = run(&params, &timing, 1).expect("correctness-only diagnostic");
        assert_eq!(
            result.extra["timing_scope"],
            "public_path_latency_including_internal_session_entry"
        );
        assert!(
            result
                .timer
                .iter()
                .any(|item| item.contains("session entry and exit"))
        );
        assert!(
            !result
                .outside_timer
                .iter()
                .any(|item| item.contains("entered held session"))
        );
        assert!(
            result
                .measurement
                .expect("zero-sample measurement")
                .samples
                .is_empty()
        );
    }
}
