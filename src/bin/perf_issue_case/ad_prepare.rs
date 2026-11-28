//! #2678 / #2643: `Runtime::prepare` cost of AD derivative graphs.
//!
//! The regressions these cases track were in graph preparation, not in the
//! kernels: the layout-alias simplifier rescanned the whole program for every
//! identity transpose/reshape rewrite, conj-sinking copied every slot's
//! symbolic metadata on every pass, and a deep reverse-mode graph built an
//! exponential `DimExpr::Max` tree. A steady-state operation row cannot see any
//! of them, so the timed operation here is `Runtime::prepare` itself, with the
//! graph construction, tracing, AD transform, `compile_many`, engine/runtime
//! construction and the numerical oracle all outside the clock.

use super::*;

/// One row of `cpu/perf_issues` (`kind = "ad_prepare"`).
///
/// `graph` selects the traced workload; `arm` is `vjp` (value + gradient
/// program) or `forward` (the same graph without the AD transform), so the
/// difference between the two arms is the AD preparation cost.
pub(super) fn run(p: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    let graph = param_str(p, "graph")?;
    let arm = param_str(p, "arm")?;
    if !matches!(arm, "vjp" | "forward") {
        return Err(format!("unknown ad_prepare arm {arm}").into());
    }
    match graph {
        "identity_layout_chain" | "conj_chain" => run_chain(p, timing, threads, graph, arm),
        "deep_reverse_mode" => run_deep_reverse_mode(p, timing, threads, arm),
        "small_tn" => run_small_tn(p, timing, threads, arm),
        _ => Err(format!("unknown ad_prepare graph {graph}").into()),
    }
}

// ---------------------------------------------------------------------------
// #2678: identity layout chains and conj chains on one complex leaf.
// ---------------------------------------------------------------------------

fn run_chain(
    p: &Value,
    timing: &Timing,
    threads: usize,
    graph: &str,
    arm: &str,
) -> Result<CaseOutcome> {
    let length = param_usize(p, "length")?;
    let n = param_usize(p, "n")?;

    let data = complex_values(n * n, 5);
    let mut backend = tenferro_einsum_benchmark::cpu_engine(Some(threads))?;

    // Setup: trace, AD transform, compile. None of this is timed.
    let trace = TraceContext::new();
    let x =
        trace.from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![n, n], data.clone())?)?;
    let loss = build_chain_loss(graph, length, &x)?;
    let grad = if arm == "vjp" {
        let ad = AdContext::builder().build()?;
        let seed = trace.from_tensor_concrete_shape(Tensor::from_vec_col_major(
            Vec::<usize>::new(),
            vec![Complex64::new(1.0, 0.0)],
        )?)?;
        Some(ad.vjp(&loss, &x, &seed)?)
    } else {
        None
    };
    let roots: Vec<&TracedTensor> = match &grad {
        Some(gradient) => vec![&loss, gradient],
        None => vec![&loss],
    };
    let program = GraphCompiler::new().compile_many(&roots)?;
    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration(&backend)?)?;
    let runtime = builder.build()?;
    let target = ExecutionEngine::target(&backend);

    // Correctness (untimed): one prepared run, the scalar loss against an
    // independent scalar loop, and every gradient component against a
    // central-difference derivative of the same loop. The complex convention is
    // the one the harness's other complex AD checks use: the cotangent seed 1
    // differentiates the loss's real part, so `g.re` and `g.im` are the
    // derivatives along the real and imaginary axes.
    let prepared_check = runtime.prepare(&program, &[], std::slice::from_ref(&target))?;
    let mut outputs = runtime
        .run(&mut [&mut backend], &prepared_check, &[])?
        .into_iter();
    let loss_out = outputs
        .next()
        .ok_or("prepared program returned no loss")?
        .into_tensor(&mut backend)?;
    let got_loss = loss_out.as_slice::<Complex64>()?[0];
    let want_loss = chain_loss_value(&data, n, graph, length)?;
    let mut worst = check_tol(rel_error(got_loss.re, want_loss), 1e-10, "loss real part")?;

    if grad.is_some() {
        let grad_out = outputs
            .next()
            .ok_or("prepared program returned no gradient")?
            .into_tensor(&mut backend)?;
        let gradient = grad_out.as_slice::<Complex64>()?;
        if gradient.len() != n * n {
            return Err("prepared program returned a wrong-sized gradient".into());
        }
        const STEP: f64 = 1e-6;
        for index in 0..n * n {
            for (direction, component) in [
                (Complex64::new(1.0, 0.0), true),
                (Complex64::new(0.0, 1.0), false),
            ] {
                let mut plus = data.clone();
                let mut minus = data.clone();
                plus[index] += direction * STEP;
                minus[index] -= direction * STEP;
                let fd = (chain_loss_value(&plus, n, graph, length)?
                    - chain_loss_value(&minus, n, graph, length)?)
                    / (2.0 * STEP);
                let analytic = if component {
                    gradient[index].re
                } else {
                    gradient[index].im
                };
                worst = worst.max(check_tol(
                    rel_error(fd, analytic),
                    1e-5,
                    "directional derivative",
                )?);
            }
        }
    }

    timed_prepare(
        runtime,
        program,
        backend,
        target,
        timing,
        worst,
        graph,
        arm,
        json!({
            "length": length,
            "n": n,
        }),
    )
}

/// The traced graph whose preparation is measured.
fn build_chain_loss(graph: &str, length: usize, x: &TracedTensor) -> Result<TracedTensor> {
    let mut y = x.clone();
    match graph {
        // A chain of identity layout pairs: every pair cancels, so the
        // simplifier has to rewrite the whole chain back to the leaf.
        "identity_layout_chain" => {
            for _ in 0..length {
                y = y.transpose(&[1, 0])?.transpose(&[1, 0])?;
            }
        }
        // Conjugation that has to sink through a transpose and a product on
        // every step; the metadata of each slot is what the old pass copied.
        "conj_chain" => {
            for _ in 0..length {
                y = y.transpose(&[1, 0])?;
                y = y.conj()?;
                y = (&y * x)?;
            }
        }
        _ => return Err(format!("unknown ad_prepare graph {graph}").into()),
    }
    Ok(y.reduce_sum(None)?)
}

/// The same graph evaluated on a flat column-major buffer, for the oracle.
fn chain_loss_value(data: &[Complex64], n: usize, graph: &str, length: usize) -> Result<f64> {
    let mut y = data.to_vec();
    let mut scratch = vec![Complex64::new(0.0, 0.0); n * n];
    match graph {
        "identity_layout_chain" => {
            for _ in 0..length {
                transpose_col_major(&y, &mut scratch, n);
                transpose_col_major(&scratch, &mut y, n);
            }
        }
        "conj_chain" => {
            for _ in 0..length {
                transpose_col_major(&y, &mut scratch, n);
                for index in 0..n * n {
                    y[index] = scratch[index].conj() * data[index];
                }
            }
        }
        _ => return Err(format!("unknown ad_prepare graph {graph}").into()),
    }
    Ok(y.iter().map(|value| value.re).sum())
}

/// `out[i + n*j] = src[j + n*i]` for a column-major `[n, n]` matrix.
fn transpose_col_major(src: &[Complex64], dst: &mut [Complex64], n: usize) {
    for j in 0..n {
        for i in 0..n {
            dst[i + n * j] = src[j + n * i];
        }
    }
}

// ---------------------------------------------------------------------------
// #2643: an unrolled RK4 neural-ODE window, the deep reverse-mode graph whose
// prepare built an exponential DimExpr::Max tree.
// ---------------------------------------------------------------------------

fn run_deep_reverse_mode(
    p: &Value,
    timing: &Timing,
    threads: usize,
    arm: &str,
) -> Result<CaseOutcome> {
    let steps = param_usize(p, "steps")?;
    let batch = param_usize(p, "batch")?;
    let hidden = param_usize(p, "hidden")?;

    let x_data = values(batch * 2, 7);
    let w1_data = values(2 * hidden, 11);
    let b1_data = values(hidden, 12);
    let w2_data = values(hidden * 2, 13);
    let b2_data = values(2, 14);
    let mut backend = tenferro_einsum_benchmark::cpu_engine(Some(threads))?;

    // Setup: trace, AD transform, compile. None of this is timed.
    let trace = TraceContext::new();
    let x = trace
        .from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![batch, 2], x_data.clone())?)?;
    let w1 = trace.from_tensor_concrete_shape(Tensor::from_vec_col_major(
        vec![2, hidden],
        w1_data.clone(),
    )?)?;
    let b1 = trace
        .from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![hidden], b1_data.clone())?)?;
    let w2 = trace.from_tensor_concrete_shape(Tensor::from_vec_col_major(
        vec![hidden, 2],
        w2_data.clone(),
    )?)?;
    let b2 =
        trace.from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![2], b2_data.clone())?)?;
    let loss = build_rk4_loss(&x, &w1, &b1, &w2, &b2, steps, batch, hidden)?;
    let wrts = [&w1, &b1, &w2, &b2];
    let grads = if arm == "vjp" {
        let ad = AdContext::builder().build()?;
        let seed = trace.from_tensor_concrete_shape(Tensor::from_vec_col_major(
            Vec::<usize>::new(),
            vec![1.0_f64],
        )?)?;
        ad.vjp_many(&loss, &wrts, &seed)?
            .into_iter()
            .map(|product| product.ok_or_else(|| "missing gradient".into()))
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let mut roots: Vec<&TracedTensor> = vec![&loss];
    roots.extend(grads.iter());
    let program = GraphCompiler::new().compile_many(&roots)?;
    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration(&backend)?)?;
    let runtime = builder.build()?;
    let target = ExecutionEngine::target(&backend);

    // Correctness (untimed): the scalar loss and every parameter gradient
    // against a central difference of the same RK4 loop.
    let prepared_check = runtime.prepare(&program, &[], std::slice::from_ref(&target))?;
    let mut outputs = runtime
        .run(&mut [&mut backend], &prepared_check, &[])?
        .into_iter();
    let loss_out = outputs
        .next()
        .ok_or("prepared program returned no loss")?
        .into_tensor(&mut backend)?;
    let got_loss = loss_out.as_slice::<f64>()?[0];
    let want_loss = rk4_loss_value(
        &x_data, &w1_data, &b1_data, &w2_data, &b2_data, steps, batch, hidden,
    );
    let mut worst = check_tol(rel_error(got_loss, want_loss), 1e-10, "loss")?;

    if arm == "vjp" {
        let mut grad_tensors = Vec::with_capacity(grads.len());
        for output in outputs {
            grad_tensors.push(output.into_tensor(&mut backend)?);
        }
        const STEP: f64 = 1e-6;
        for (parameter, (values, grad)) in [
            (&w1_data, &grad_tensors[0]),
            (&b1_data, &grad_tensors[1]),
            (&w2_data, &grad_tensors[2]),
            (&b2_data, &grad_tensors[3]),
        ]
        .into_iter()
        .enumerate()
        {
            let gradient = grad.as_slice::<f64>()?;
            if gradient.len() != values.len() {
                return Err(format!("wrong-sized gradient for parameter {parameter}").into());
            }
            for index in 0..values.len() {
                let loss_at = |scale: f64| -> f64 {
                    let mut perturbed = values.clone();
                    perturbed[index] += scale * STEP;
                    let (w1, b1, w2, b2) = match parameter {
                        0 => (
                            perturbed.as_slice(),
                            b1_data.as_slice(),
                            w2_data.as_slice(),
                            b2_data.as_slice(),
                        ),
                        1 => (
                            w1_data.as_slice(),
                            perturbed.as_slice(),
                            w2_data.as_slice(),
                            b2_data.as_slice(),
                        ),
                        2 => (
                            w1_data.as_slice(),
                            b1_data.as_slice(),
                            perturbed.as_slice(),
                            b2_data.as_slice(),
                        ),
                        _ => (
                            w1_data.as_slice(),
                            b1_data.as_slice(),
                            w2_data.as_slice(),
                            perturbed.as_slice(),
                        ),
                    };
                    rk4_loss_value(x_data.as_slice(), w1, b1, w2, b2, steps, batch, hidden)
                };
                let fd = (loss_at(1.0) - loss_at(-1.0)) / (2.0 * STEP);
                worst = worst.max(check_tol(
                    rel_error(fd, gradient[index]),
                    1e-5,
                    "parameter derivative",
                )?);
            }
        }
    }

    timed_prepare(
        runtime,
        program,
        backend,
        target,
        timing,
        worst,
        "deep_reverse_mode",
        arm,
        json!({
            "steps": steps,
            "batch": batch,
            "hidden": hidden,
        }),
    )
}

/// The traced RK4 window whose preparation is measured.
fn build_rk4_loss(
    x: &TracedTensor,
    w1: &TracedTensor,
    b1: &TracedTensor,
    w2: &TracedTensor,
    b2: &TracedTensor,
    steps: usize,
    batch: usize,
    hidden: usize,
) -> Result<TracedTensor> {
    const H: f64 = 25.0 / 999.0;
    let mut state = x.clone();
    for _ in 0..steps {
        let k1 = rk4_rhs(&state, w1, b1, w2, b2, batch, hidden)?;
        let k2 = rk4_rhs(
            &offset(&state, &k1, 0.5 * H)?,
            w1,
            b1,
            w2,
            b2,
            batch,
            hidden,
        )?;
        let k3 = rk4_rhs(
            &offset(&state, &k2, 0.5 * H)?,
            w1,
            b1,
            w2,
            b2,
            batch,
            hidden,
        )?;
        let k4 = rk4_rhs(&offset(&state, &k3, H)?, w1, b1, w2, b2, batch, hidden)?;
        // H/6 (k1 + 2 k2 + 2 k3 + k4)
        let pair12 = add(&k1, &scale(&k2, 2.0)?)?;
        let pair34 = add(&scale(&k3, 2.0)?, &k4)?;
        let increment = scale(&add(&pair12, &pair34)?, H / 6.0)?;
        state = add(&state, &increment)?;
    }
    Ok(state.reduce_sum(None)?)
}

/// f(state) = tanh(state W1 + b1) W2 + b2, all column-major.
fn rk4_rhs(
    state: &TracedTensor,
    w1: &TracedTensor,
    b1: &TracedTensor,
    w2: &TracedTensor,
    b2: &TracedTensor,
    batch: usize,
    hidden: usize,
) -> Result<TracedTensor> {
    let hidden_in = state.matmul(w1)?;
    let b1_broadcast = b1.broadcast_in_dim(&[batch, hidden], &[1])?;
    let hidden_out = add(&hidden_in, &b1_broadcast)?.tanh()?;
    let out = hidden_out.matmul(w2)?;
    let b2_broadcast = b2.broadcast_in_dim(&[batch, 2], &[1])?;
    add(&out, &b2_broadcast)
}

/// `state + factor * direction` on traced tensors.
fn offset(state: &TracedTensor, direction: &TracedTensor, factor: f64) -> Result<TracedTensor> {
    add(state, &scale(direction, factor)?)
}

fn scale(tensor: &TracedTensor, factor: f64) -> Result<TracedTensor> {
    Ok((tensor * factor)?)
}

fn add(left: &TracedTensor, right: &TracedTensor) -> Result<TracedTensor> {
    Ok((left + right)?)
}

/// The same RK4 window on flat column-major buffers, for the oracle.
fn rk4_loss_value(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    steps: usize,
    batch: usize,
    hidden: usize,
) -> f64 {
    const H: f64 = 25.0 / 999.0;
    let mut state = x.to_vec();
    let mut k1 = vec![0.0; batch * 2];
    let mut k2 = vec![0.0; batch * 2];
    let mut k3 = vec![0.0; batch * 2];
    let mut k4 = vec![0.0; batch * 2];
    let mut temporary = vec![0.0; batch * 2];
    for _ in 0..steps {
        rk4_rhs_value(&state, w1, b1, w2, b2, batch, hidden, &mut k1);
        offset_value(&state, &k1, 0.5 * H, &mut temporary);
        rk4_rhs_value(&temporary, w1, b1, w2, b2, batch, hidden, &mut k2);
        offset_value(&state, &k2, 0.5 * H, &mut temporary);
        rk4_rhs_value(&temporary, w1, b1, w2, b2, batch, hidden, &mut k3);
        offset_value(&state, &k3, H, &mut temporary);
        rk4_rhs_value(&temporary, w1, b1, w2, b2, batch, hidden, &mut k4);
        for index in 0..batch * 2 {
            state[index] += H / 6.0 * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]);
        }
    }
    state.iter().sum()
}

fn rk4_rhs_value(
    state: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    batch: usize,
    hidden: usize,
    out: &mut [f64],
) {
    let mut inner = vec![0.0; batch * hidden];
    for j in 0..hidden {
        for row in 0..batch {
            let mut sum = b1[j];
            for k in 0..2 {
                sum += state[row + batch * k] * w1[k + 2 * j];
            }
            inner[row + batch * j] = sum.tanh();
        }
    }
    for i in 0..2 {
        for row in 0..batch {
            let mut sum = b2[i];
            for j in 0..hidden {
                sum += inner[row + batch * j] * w2[j + hidden * i];
            }
            out[row + batch * i] = sum;
        }
    }
}

fn offset_value(state: &[f64], direction: &[f64], factor: f64, out: &mut [f64]) {
    for index in 0..state.len() {
        out[index] = state[index] + factor * direction[index];
    }
}

// ---------------------------------------------------------------------------
// #2678: a small tensor-network energy — an MPS expectation value with conj,
// einsum and identity layout round trips, differentiated over every site.
// ---------------------------------------------------------------------------

fn run_small_tn(p: &Value, timing: &Timing, threads: usize, arm: &str) -> Result<CaseOutcome> {
    let n = param_usize(p, "sites")?;
    let d = param_usize(p, "d")?;
    let bond = param_usize(p, "bond")?;
    let center = n / 2;
    // Open boundaries: one bond index of size 1 at each end, `bond` in between.
    let shapes: Vec<[usize; 3]> = (0..n)
        .map(|k| {
            [
                if k == 0 { 1 } else { bond },
                d,
                if k + 1 == n { 1 } else { bond },
            ]
        })
        .collect();
    let data: Vec<Vec<Complex64>> = shapes
        .iter()
        .enumerate()
        .map(|(k, shape)| complex_values(shape[0] * shape[1] * shape[2], 101 + k as u64))
        .collect();
    let operator = complex_values(d * d, 97);

    let mut backend = tenferro_einsum_benchmark::cpu_engine(Some(threads))?;

    // Setup: trace, AD transform, compile. None of this is timed.
    let trace = TraceContext::new();
    let mut sites = Vec::with_capacity(n);
    for (shape, values) in shapes.iter().zip(&data) {
        sites.push(trace.from_tensor_concrete_shape(Tensor::from_vec_col_major(
            shape.to_vec(),
            values.clone(),
        )?)?);
    }
    let h = trace
        .from_tensor_concrete_shape(Tensor::from_vec_col_major(vec![d, d], operator.clone())?)?;
    let loss = build_tn_loss(&trace, &sites, &h, n, d, bond, center)?;
    let wrts: Vec<&TracedTensor> = sites.iter().collect();
    let grads = if arm == "vjp" {
        let ad = AdContext::builder().build()?;
        let seed = trace.from_tensor_concrete_shape(Tensor::from_vec_col_major(
            Vec::<usize>::new(),
            vec![Complex64::new(1.0, 0.0)],
        )?)?;
        ad.vjp_many(&loss, &wrts, &seed)?
            .into_iter()
            .map(|product| product.ok_or_else(|| "missing gradient".into()))
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let mut roots: Vec<&TracedTensor> = vec![&loss];
    roots.extend(grads.iter());
    let program = GraphCompiler::new().compile_many(&roots)?;
    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration(&backend)?)?;
    let runtime = builder.build()?;
    let target = ExecutionEngine::target(&backend);

    // Correctness (untimed): the scalar energy against an independent scalar
    // contraction, and every site gradient against a central difference of the
    // same contraction (real and imaginary directions, as in `run_chain`).
    let prepared_check = runtime.prepare(&program, &[], std::slice::from_ref(&target))?;
    let mut outputs = runtime
        .run(&mut [&mut backend], &prepared_check, &[])?
        .into_iter();
    let loss_out = outputs
        .next()
        .ok_or("prepared program returned no loss")?
        .into_tensor(&mut backend)?;
    let got_loss = loss_out.as_slice::<Complex64>()?[0];
    let want_loss = tn_loss_value(&data, &operator, n, d, bond, center);
    let mut worst = check_tol(rel_error(got_loss.re, want_loss), 1e-10, "energy")?;

    if arm == "vjp" {
        let mut grad_tensors = Vec::with_capacity(grads.len());
        for output in outputs {
            grad_tensors.push(output.into_tensor(&mut backend)?);
        }
        const STEP: f64 = 1e-6;
        for (k, (values, grad)) in data.iter().zip(&grad_tensors).enumerate() {
            let gradient = grad.as_slice::<Complex64>()?;
            if gradient.len() != values.len() {
                return Err(format!("wrong-sized gradient for site {k}").into());
            }
            for index in 0..values.len() {
                for (direction, component) in [
                    (Complex64::new(1.0, 0.0), true),
                    (Complex64::new(0.0, 1.0), false),
                ] {
                    let mut plus = data.clone();
                    let mut minus = data.clone();
                    plus[k][index] += direction * STEP;
                    minus[k][index] -= direction * STEP;
                    let fd = (tn_loss_value(&plus, &operator, n, d, bond, center)
                        - tn_loss_value(&minus, &operator, n, d, bond, center))
                        / (2.0 * STEP);
                    let analytic = if component {
                        gradient[index].re
                    } else {
                        gradient[index].im
                    };
                    worst = worst.max(check_tol(rel_error(fd, analytic), 1e-5, "site derivative")?);
                }
            }
        }
    }

    timed_prepare(
        runtime,
        program,
        backend,
        target,
        timing,
        worst,
        "small_tn",
        arm,
        json!({
            "sites": n,
            "d": d,
            "bond": bond,
        }),
    )
}

/// One site with identity transition and reshape round trips on top of it: the
/// layout aliases the simplifier rewrites, and the fan-out both the conjugated
/// and the plain copy of the site read.
fn tn_site_layout(
    a: &TracedTensor,
    bond_left: usize,
    d: usize,
    bond_right: usize,
) -> Result<TracedTensor> {
    let a_t = a.transpose(&[2, 1, 0])?.transpose(&[2, 1, 0])?;
    let a_r = a_t.reshape(&[bond_left * d, bond_right])?;
    Ok(a_r.reshape(&[bond_left, d, bond_right])?)
}

/// The traced MPS energy: a left-to-right transfer-matrix sweep over
/// `[bond_left, bond_right]` environments, with the local operator inserted at
/// `center`.
///
/// The contraction is `dot_general`, not `einsum`: at this revision
/// `tenferro.einsum.v1` has no VJP rule, and the timed graph is the AD
/// transform's output, so an einsum node would fail the transform instead of
/// measuring preparation.
fn build_tn_loss(
    trace: &TraceContext,
    sites: &[TracedTensor],
    h: &TracedTensor,
    n: usize,
    d: usize,
    bond: usize,
    center: usize,
) -> Result<TracedTensor> {
    let mut env = trace.from_tensor_concrete_shape(Tensor::from_vec_col_major(
        vec![1, 1],
        vec![Complex64::new(1.0, 0.0)],
    )?)?;
    for (k, a) in sites.iter().enumerate() {
        let bond_left = if k == 0 { 1 } else { bond };
        let bond_right = if k + 1 == n { 1 } else { bond };
        let site = tn_site_layout(a, bond_left, d, bond_right)?;
        let conj = site.conj()?;
        env = if k == center {
            let with_conj = env.dot_general(&conj, matmul_config(&[0], &[0]))?;
            let with_h = with_conj.dot_general(h, matmul_config(&[1], &[0]))?;
            with_h.dot_general(&site, matmul_config(&[0, 2], &[0, 1]))?
        } else {
            let with_conj = env.dot_general(&conj, matmul_config(&[0], &[0]))?;
            with_conj.dot_general(&site, matmul_config(&[0, 1], &[0, 1]))?
        };
    }
    Ok(env.reduce_sum(None)?)
}

/// The same energy on flat column-major buffers, for the oracle.
fn tn_loss_value(
    data: &[Vec<Complex64>],
    operator: &[Complex64],
    n: usize,
    d: usize,
    bond: usize,
    center: usize,
) -> f64 {
    let mut env = vec![Complex64::new(1.0, 0.0)];
    let mut left = 1usize;
    for k in 0..n {
        let bond_left = if k == 0 { 1 } else { bond };
        let bond_right = if k + 1 == n { 1 } else { bond };
        let a = &data[k];
        let mut next = vec![Complex64::new(0.0, 0.0); bond_right * bond_right];
        for c in 0..bond_right {
            for dd in 0..bond_right {
                let mut sum = Complex64::new(0.0, 0.0);
                for x in 0..bond_left {
                    for y in 0..bond_left {
                        let e = env[x + left * y];
                        if e == Complex64::new(0.0, 0.0) || e == Complex64::new(-0.0, -0.0) {
                            continue;
                        }
                        for s in 0..d {
                            let a_s = a[x + bond_left * (s + d * c)];
                            for t in 0..d {
                                let a_t = a[y + bond_left * (t + d * dd)];
                                let weight = if k == center {
                                    operator[s + d * t]
                                } else if s == t {
                                    Complex64::new(1.0, 0.0)
                                } else {
                                    continue;
                                };
                                sum += e * a_s.conj() * weight * a_t;
                            }
                        }
                    }
                }
                next[c + bond_right * dd] = sum;
            }
        }
        env = next;
        left = bond_right;
    }
    env[0].re
}

// ---------------------------------------------------------------------------
// Shared timer.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn timed_prepare(
    runtime: Runtime,
    program: tenferro_runtime::CompiledGraph,
    _backend: CpuEngine,
    target: tenferro_runtime::EngineTarget,
    timing: &Timing,
    worst: f64,
    graph: &str,
    arm: &str,
    shape: Value,
) -> Result<CaseOutcome> {
    let measurement = measure(timing, 1 << 20, || {
        Ok(runtime.prepare(&program, &[], std::slice::from_ref(&target))?)
    })?;
    Ok(CaseOutcome {
        max_rel_error: worst,
        measurement: Some(measurement),
        timer: vec!["Runtime::prepare of the traced graph"],
        outside_timer: vec![
            "engine/runtime construction",
            "trace and AD transform",
            "compile_many",
            "input construction",
            "finite-difference correctness check",
        ],
        extra: {
            let mut extra = json!({
                "graph": graph,
                "execution_path": arm,
            });
            if let (Value::Object(fields), Value::Object(more)) = (&mut extra, shape) {
                fields.extend(more);
            }
            extra
        },
    })
}
