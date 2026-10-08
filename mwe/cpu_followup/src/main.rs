//! Public-API CPU reproducers, independent of the legacy harness's optional providers.
//! Run from the repository root: cpu-followup-mwe CASE_ID PATH THREADS [RUNS].
use num_complex::{Complex32, Complex64};
use serde_json::{json, Value};
use std::{error::Error, hint::black_box, time::Instant};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_fft::{FftExecutor, FftNorm, TracedTensorFftExt};
use tenferro_linalg::{TensorLinalgExt, TracedTensorLinalgExt};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{
    BackendSessionHost, DType, GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor,
    TensorRead, TensorValue, TensorView, TypedTensorView,
};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn value(i: usize, seed: u64) -> f64 {
    (((i as u64).wrapping_mul(1837).wrapping_add(seed * 335) % 2048) as f64 - 1024.) / 1024.
}
fn data(n: usize, seed: u64) -> Vec<f64> {
    (0..n).map(|i| value(i, seed)).collect()
}
fn tensor(shape: Vec<usize>, seed: u64, positive: bool) -> Result<Tensor> {
    let xs = (0..shape.iter().product())
        .map(|i| {
            if positive {
                0.25 + value(i, seed).abs()
            } else {
                value(i, seed)
            }
        })
        .collect();
    Ok(Tensor::from_vec_col_major(shape, xs)?)
}
fn probe_indices(n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    let mut v = vec![0, n - 1, n / 2];
    v.extend((1..128).map(|i| (i * 1597334677usize) % n));
    v.sort_unstable();
    v.dedup();
    v
}
fn signature(t: &Tensor) -> Result<Value> {
    let mut sum = Complex64::new(0., 0.);
    let mut abs = 0.;
    let mut sq = 0.;
    let ids = probe_indices(t.shape().iter().product::<usize>());
    let mut probes = Vec::new();
    macro_rules! read_values {
        ($ty:ty,$convert:expr) => {{
            let xs = t.as_slice::<$ty>()?;
            for &x in xs {
                let z: Complex64 = ($convert)(x);
                assert!(z.re.is_finite() && z.im.is_finite());
                sum += z;
                abs += z.norm();
                sq += z.norm_sqr();
            }
            for &i in &ids {
                let z: Complex64 = ($convert)(xs[i]);
                probes.push(json!([i, z.re, z.im]));
            }
        }};
    }
    match t.dtype() {
        DType::F64 => read_values!(f64, |x| Complex64::new(x, 0.)),
        DType::F32 => read_values!(f32, |x| Complex64::new(x as f64, 0.)),
        DType::C64 => read_values!(Complex64, |x| x),
        DType::C32 => read_values!(Complex32, |x: Complex32| Complex64::new(
            x.re as f64,
            x.im as f64
        )),
        _ => return Err("unexpected output dtype".into()),
    }
    Ok(
        json!({"shape":t.shape(),"count":t.shape().iter().product::<usize>(),"sum":[sum.re,sum.im],"sum_abs":abs,"sum_sq":sq,"probes":probes}),
    )
}

// setup(count) is outside EVERY clock, including calibration and zero warmups.
fn sample<I, O>(
    bytes: usize,
    runs: usize,
    target: u128,
    mut setup: impl FnMut(usize) -> Result<Vec<I>>,
    mut op: impl FnMut(I) -> Result<O>,
) -> Result<Value> {
    if runs == 0 {
        return Ok(json!({"samples":[],"calibration":{"iterations":0}}));
    }
    for _ in 0..3 {
        let mut inputs = setup(1)?;
        black_box(op(inputs.pop().unwrap())?);
    }
    let cap = (512 * 1024 * 1024 / bytes.max(1)).clamp(1, 2_000_000);
    let mut batch = |n: usize| -> Result<u128> {
        let inputs = setup(n)?;
        let mut iter = inputs.into_iter();
        let mut outputs = Vec::with_capacity(n);
        let start = Instant::now();
        for input in iter.by_ref() {
            outputs.push(op(input)?);
        }
        let elapsed = start.elapsed().as_nanos();
        black_box(&outputs);
        drop(outputs);
        drop(iter);
        Ok(elapsed)
    };
    let mut count = 1;
    let calibrated = loop {
        let elapsed = batch(count)?;
        if elapsed >= target || count == cap {
            break elapsed;
        }
        count = (count * 2).min(cap);
    };
    let samples = (0..runs)
        .map(|i| Ok(json!({"sample_index":i,"iterations":count,"elapsed_ns":batch(count)?})))
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"samples":samples,"calibration":{"iterations":count,"elapsed_ns":calibrated,"target_ns":target,"memory_cap_bytes":512*1024*1024}}),
    )
}
fn units(n: usize) -> Result<Vec<()>> {
    Ok(vec![(); n])
}
fn reads<'a>(x: &'a Tensor, n: usize) -> Result<Vec<TensorRead<'a>>> {
    Ok((0..n).map(|_| TensorRead::from_tensor(x)).collect())
}
fn traced(x: &Tensor) -> Result<TracedTensor> {
    Ok(TracedTensor::from_tensor_concrete_shape(x.duplicate()?)?)
}
fn trace_measure(
    outputs: &[TracedTensor],
    threads: usize,
    bytes: usize,
    runs: usize,
    target: u128,
) -> Result<Value> {
    let refs: Vec<_> = outputs.iter().collect();
    let program = GraphCompiler::new().compile_many(&refs)?;
    let b = CpuBackend::with_threads(threads)?;
    let engine = tenferro_cpu::runtime_engine_id()?;
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&b)?)?;
    builder.install_extension_module(tenferro_linalg::extension_module::<CpuBackend>(
        engine.clone(),
    )?)?;
    builder.install_extension_module(tenferro_fft::extension_module::<CpuBackend>(
        engine.clone(),
    )?)?;
    let runtime = builder.build()?;
    let prepared = runtime.prepare_compiled(&program, &[])?;
    let output = runtime.run_prepared(&prepared, &[])?;
    let sig = output.iter().map(signature).collect::<Result<Vec<_>>>()?;
    drop(output);
    let mut row = sample(bytes, runs, target, units, |()| {
        Ok(runtime.run_prepared(&prepared, &[])?)
    })?;
    row["outputs"] = json!(sig);
    // Runtime does not expose a borrowed execution-session call. Its internal
    // admission/session work is included: these are GRAPH-CALL LATENCY diagnostics.
    row["timing_scope"] = json!(
        "prepared graph-call latency diagnostic; internal runtime admission/session entry included"
    );
    Ok(row)
}
fn tensor_measure<'a, I>(
    bytes: usize,
    runs: usize,
    target: u128,
    setup: impl FnMut(usize) -> Result<Vec<I>>,
    mut op: impl FnMut(I) -> Result<Tensor>,
) -> Result<Value> {
    let mut setup = setup;
    let input = setup(1)?.pop().unwrap();
    let output = op(input)?;
    let sig = signature(&output)?;
    drop(output);
    let mut row = sample(bytes, runs, target, setup, op)?;
    row["outputs"] = json!([sig]);
    Ok(row)
}

fn real(case: &Value, path: &str, threads: usize, runs: usize, target: u128) -> Result<Value> {
    let op = case["op"].as_str().unwrap();
    let reduction = op.starts_with("reduce_");
    let shape = if reduction {
        if op.ends_with("_all") {
            vec![8192, 4096]
        } else if op == "reduce_min_axis1" {
            vec![4096, 4096]
        } else {
            vec![2048, 2048]
        }
    } else if matches!(op, "pow" | "expm1" | "log1p") {
        vec![4194304]
    } else {
        vec![8388608]
    };
    let n = shape.iter().product();
    let mut x = tensor(shape.clone(), 1, matches!(op, "pow" | "log" | "log1p"))?;
    if op.contains("reduce_prod") {
        x = Tensor::from_vec_col_major(shape.clone(), vec![1.000001f64; n])?;
    }
    let exponent = if op == "pow" {
        Some(Tensor::from_vec_col_major(shape.clone(), vec![1.5f64; n])?)
    } else {
        None
    };
    let axes = if op.ends_with("axis0") {
        vec![0]
    } else if op.ends_with("axis1") {
        vec![1]
    } else {
        vec![0, 1]
    };
    let bytes = if reduction {
        if axes.len() == 2 {
            8
        } else {
            shape[1 - axes[0]] * 8
        }
    } else {
        n * 8
    };
    if path == "prepared-trace-call" {
        let tx = traced(&x)?;
        let y = match op {
            "exp" => tx.exp()?,
            "log" => tx.log()?,
            "sin" => tx.sin()?,
            "cos" => tx.cos()?,
            "tanh" => tx.tanh()?,
            "expm1" => tx.expm1()?,
            "log1p" => tx.log1p()?,
            "pow" => tx.pow(&traced(exponent.as_ref().unwrap())?)?,
            x if x.starts_with("reduce_max") => tx.reduce_max(Some(&axes))?,
            x if x.starts_with("reduce_min") => tx.reduce_min(Some(&axes))?,
            x if x.starts_with("reduce_sum") => tx.reduce_sum(Some(&axes))?,
            x if x.starts_with("reduce_prod") => tx.reduce_prod(Some(&axes))?,
            _ => return Err("unknown real op".into()),
        };
        return trace_measure(&[y], threads, bytes, runs, target);
    }
    let mut b = CpuBackend::with_threads(threads)?;
    b.with_backend_session(|s| -> Result<Value> {
        tensor_measure(
            bytes,
            runs,
            target,
            |count| {
                Ok((0..count)
                    .map(|_| {
                        (
                            TensorRead::from_tensor(&x),
                            exponent.as_ref().map(TensorRead::from_tensor),
                        )
                    })
                    .collect())
            },
            |(r, e)| {
                Ok(match op {
                    "exp" => s.exp_read(r)?,
                    "log" => s.log_read(r)?,
                    "sin" => s.sin_read(r)?,
                    "cos" => s.cos_read(r)?,
                    "tanh" => s.tanh_read(r)?,
                    "expm1" => s.expm1_read(r)?,
                    "log1p" => s.log1p_read(r)?,
                    "pow" => s.pow_read(r, e.unwrap())?,
                    x if x.starts_with("reduce_max") => s.reduce_max_read(r, &axes)?,
                    x if x.starts_with("reduce_min") => s.reduce_min_read(r, &axes)?,
                    x if x.starts_with("reduce_sum") => s.reduce_sum_read(r, &axes)?,
                    x if x.starts_with("reduce_prod") => s.reduce_prod_read(r, &axes)?,
                    _ => return Err("unknown real op".into()),
                })
            },
        )
    })?
}

fn activation(case: &Value, threads: usize, runs: usize, target: u128) -> Result<Value> {
    let op = case["op"].as_str().unwrap();
    let n = 1024 * 64;
    let data: Vec<f32> = (0..n as u64)
        .map(|i| (((i * 2654435761 + 50 * 97) % 2001) as f64 / 1000. - 1.) as f32 * 4.)
        .collect();
    // Cast after multiplication by 0.5 in the maintained fixture; multiplication
    // by 8 is exact in f32 and the formula above has the same binary rounding.
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(threads)?)?;
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1024, 64], data)?,
        runtime.clone(),
    )?;
    runtime
        .with_eager_session(|s| -> tenferro_ad::Result<Value> {
            let mut call = || match op {
                "erf" => s.erf(&x),
                "sigmoid" => s.sigmoid(&x),
                "silu" => s.silu(&x),
                "softplus" => s.softplus(&x),
                "gelu_tanh" => s.gelu_tanh(&x),
                _ => unreachable!(),
            };
            let initial = call()?;
            drop(call);
            let actual = s.duplicate_value(&initial)?;
            let scalar_error = validate_activation(&actual, op).map_err(ad_error)?;
            let sig = signature(&actual).map_err(ad_error)?;
            drop(actual);
            drop(initial);
            let row = sample(n * 4, runs, target, units, |()| {
                Ok(match op {
                    "erf" => s.erf(&x)?,
                    "sigmoid" => s.sigmoid(&x)?,
                    "silu" => s.silu(&x)?,
                    "softplus" => s.softplus(&x)?,
                    "gelu_tanh" => s.gelu_tanh(&x)?,
                    _ => unreachable!(),
                })
            })
            .map_err(ad_error)?;
            let mut row = row;
            row["outputs"] = json!([sig]);
            row["max_scaled_scalar_error"] = json!(scalar_error);
            Ok(row)
        })
        .map_err(Into::into)
}
fn validate_activation(actual: &Tensor, op: &str) -> Result<f64> {
    let mut error = 0f64;
    for (i, &got) in actual.as_slice::<f32>()?.iter().enumerate() {
        let x =
            ((((i as u64 * 2654435761 + 50 * 97) % 2001) as f64 / 1000. - 1.) as f32 * 4.) as f64;
        let sigmoid = 1. / (1. + (-x).exp());
        let expected = match op {
            "erf" => libm::erf(x),
            "sigmoid" => sigmoid,
            "silu" => x * sigmoid,
            "softplus" => x.max(0.) + (-x.abs()).exp().ln_1p(),
            "gelu_tanh" => {
                0.5 * x
                    * (1.
                        + ((2. / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
            }
            _ => unreachable!(),
        };
        error = error.max((got as f64 - expected).abs() / expected.abs().max(1.));
    }
    if error > 2e-5 {
        return Err(format!("scalar activation oracle error {error}").into());
    }
    Ok(error)
}

fn ad_error(e: Box<dyn Error + Send + Sync>) -> tenferro_ad::Error {
    tenferro_ad::Error::runtime_state(
        "cpu_followup",
        tenferro_runtime::ErrorPhase::Execution,
        e.to_string(),
    )
}

fn indexing(case: &Value, path: &str, threads: usize, runs: usize, target: u128) -> Result<Value> {
    let op = case["op"].as_str().unwrap();
    let n = match op {
        "gather" | "scatter" => 262144,
        "slice" | "dynamic_slice" => 4194304,
        "concatenate" => 1048576,
        _ => 2097152,
    };
    let x = tensor(vec![n], 1, false)?;
    let y = tensor(
        vec![if op == "dynamic_update_slice" {
            n / 2
        } else {
            n
        }],
        2,
        false,
    )?;
    let zeros = if op == "scatter" {
        Some(Tensor::from_vec_col_major(vec![n], vec![0.0f64; n])?)
    } else {
        None
    };
    let idx = Tensor::from_vec_col_major(
        if op == "scatter" { vec![n, 1] } else { vec![n] },
        (0..n).map(|i| ((i * 37 + 11) % n) as i64).collect(),
    )?;
    let start = Tensor::from_vec_col_major(vec![1], vec![1024i64])?;
    let slice = SliceConfig {
        starts: vec![1024],
        limits: vec![n - 1024],
        strides: vec![2],
    };
    let pad = PadConfig {
        edge_padding_low: vec![128],
        edge_padding_high: vec![128],
        interior_padding: vec![0],
    };
    let gather = GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };
    let scatter = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let bytes = match op {
        "slice" => (n - 2048) / 2 * 8,
        "dynamic_slice" => n / 2 * 8,
        "pad" => (n + 256) * 8,
        "concatenate" => n * 16,
        _ => n * 8,
    };
    if path == "prepared-trace-call" {
        let tx = traced(&x)?;
        let out = match op {
            "gather" => tx.gather(&traced(&idx)?, gather.clone())?,
            "scatter" => traced(zeros.as_ref().unwrap())?.scatter(
                &traced(&idx)?,
                &traced(&y)?,
                scatter.clone(),
            )?,
            "slice" => tx.slice(slice.clone())?,
            "dynamic_slice" => tx.dynamic_slice(&traced(&start)?, &[n / 2])?,
            "dynamic_update_slice" => {
                return Err("unsupported: no public TracedTensor dynamic_update_slice API".into())
            }
            "pad" => tx.pad(pad.clone())?,
            "concatenate" => TracedTensor::concatenate(&[&tx, &traced(&y)?], 0)?,
            "reverse" => tx.reverse(&[0])?,
            _ => unreachable!(),
        };
        return trace_measure(&[out], threads, bytes, runs, target);
    }
    let mut b = CpuBackend::with_threads(threads)?;
    b.with_backend_session(|s| -> Result<Value> {
        tensor_measure(bytes, runs, target, units, |()| {
            Ok(match op {
                "gather" => s.gather(&x, &idx, &gather)?,
                "scatter" => s.scatter(zeros.as_ref().unwrap(), &idx, &y, &scatter)?,
                "slice" => s.slice(&x, &slice)?,
                "dynamic_slice" => s.dynamic_slice(&x, &start, &[n / 2])?,
                "dynamic_update_slice" => s.dynamic_update_slice(&x, &y, &start)?,
                "pad" => s.pad(&x, &pad)?,
                "concatenate" => s.concatenate(&[&x, &y], 0)?,
                "reverse" => s.reverse(&x, &[0])?,
                _ => unreachable!(),
            })
        })
    })?
}

fn structural(
    case: &Value,
    path: &str,
    threads: usize,
    runs: usize,
    target: u128,
) -> Result<Value> {
    let op = case["op"].as_str().unwrap();
    let shape = match op {
        "cast_f64_f32" => vec![33554432],
        "extract_diagonal" => vec![8388608, 2, 2],
        "broadcast_in_dim" => vec![8192, 1],
        _ => vec![4096, 4096],
    };
    let x = tensor(shape.clone(), 1, false)?;
    let bytes = match op {
        "cast_f64_f32" => 33554432 * 4,
        "extract_diagonal" => 8388608 * 2 * 8,
        "broadcast_in_dim" => 8192 * 4096 * 8,
        _ => 4096 * 4096 * 8,
    };
    if path == "prepared-trace-call" {
        let tx = traced(&x)?;
        let out = match op {
            "cast_f64_f32" => tx.cast(DType::F32)?,
            "extract_diagonal" => tx.extract_diag(1, 2)?,
            "broadcast_in_dim" => tx.broadcast_in_dim(&[8192, 4096], &[0, 1])?,
            "transpose" => tx.transpose(&[1, 0])?,
            "tril" => tx.tril(0)?,
            "triu" => tx.triu(0)?,
            _ => unreachable!(),
        };
        return trace_measure(&[out], threads, bytes, runs, target);
    }
    let mut b = CpuBackend::with_threads(threads)?;
    b.with_backend_session(|s| -> Result<Value> {
        tensor_measure(
            bytes,
            runs,
            target,
            |n| reads(&x, n),
            |r| {
                Ok(match op {
                    "cast_f64_f32" => s.cast(&x, DType::F32)?,
                    "extract_diagonal" => s.extract_diagonal(&x, 1, 2)?,
                    "broadcast_in_dim" => s.broadcast_in_dim_read(r, &[8192, 4096], &[0, 1])?,
                    "transpose" => s.transpose_read(r, &[1, 0])?,
                    "tril" => s.tril(&x, 0)?,
                    "triu" => s.triu(&x, 0)?,
                    _ => unreachable!(),
                })
            },
        )
    })?
}

fn complex_or_linalg(
    case: &Value,
    path: &str,
    threads: usize,
    runs: usize,
    target: u128,
) -> Result<Value> {
    let op = case["op"].as_str().unwrap();
    let complex = case["category"] == "complex";
    let shape = if op == "norm_fro" {
        vec![2048, 1536]
    } else if complex {
        vec![4194304]
    } else {
        vec![1024, 1024]
    };
    let n = shape.iter().product();
    let x = if complex {
        Tensor::from_vec_col_major(
            shape.clone(),
            (0..n)
                .map(|i| Complex64::new(value(i, 1), value(i, 2)))
                .collect(),
        )?
    } else {
        let mut xs = data(n, 1);
        for i in 0..1024 {
            xs[i + i * 1024] += 2. + i as f64 / 1024.;
        }
        Tensor::from_vec_col_major(shape, xs)?
    };
    let bytes = if matches!(op, "norm_fro" | "slogdet") {
        16
    } else {
        n * 16
    };
    if path == "prepared-trace-call" {
        let tx = traced(&x)?;
        let outs = match op {
            "exp" => vec![tx.exp()?],
            "log" => vec![tx.log()?],
            "norm_fro" => vec![tx.norm(None, Some(&[0, 1]), false)?],
            "slogdet" => {
                let (s, l) = tx.slogdet()?;
                vec![s, l]
            }
            _ => unreachable!(),
        };
        return trace_measure(&outs, threads, bytes, runs, target);
    }
    let mut b = CpuBackend::with_threads(threads)?;
    b.with_backend_session(|s| -> Result<Value> {
        if matches!(op, "exp" | "log") {
            return tensor_measure(
                bytes,
                runs,
                target,
                |n| reads(&x, n),
                |r| {
                    Ok(if op == "exp" {
                        s.exp_read(r)?
                    } else {
                        s.log_read(r)?
                    })
                },
            );
        }
        with_cpu_exec_session(s, |s| -> Result<Value> {
            if op == "norm_fro" {
                tensor_measure(bytes, runs, target, units, |()| {
                    Ok(x.norm(None, Some(&[0, 1]), false, s)?)
                })
            } else {
                let initial = x.slogdet(s)?;
                let sig = vec![signature(&initial.0)?, signature(&initial.1)?];
                drop(initial);
                let mut row = sample(bytes, runs, target, units, |()| Ok(x.slogdet(s)?))?;
                row["outputs"] = json!(sig);
                Ok(row)
            }
        })
        .expect("CPU session")
    })?
}

fn fft(case: &Value, path: &str, threads: usize, runs: usize, target: u128) -> Result<Value> {
    let op = case["op"].as_str().unwrap();
    let n = 1048576usize;
    let x = if op == "rfft" {
        Tensor::from_vec_col_major(vec![n], (0..n).map(|i| value(i, 17) as f32).collect())?
    } else if case["dtype"] == "c64" {
        Tensor::from_vec_col_major(
            vec![if op == "irfft" { n / 2 + 1 } else { n }],
            (0..if op == "irfft" { n / 2 + 1 } else { n })
                .map(|i| Complex64::new(value(i, 17), value(i, 18)))
                .collect(),
        )?
    } else {
        Tensor::from_vec_col_major(
            vec![if op == "irfft" { n / 2 + 1 } else { n }],
            (0..if op == "irfft" { n / 2 + 1 } else { n })
                .map(|i| Complex32::new(value(i, 17) as f32, value(i, 18) as f32))
                .collect(),
        )?
    };
    let bytes = n * if case["dtype"] == "c64" { 16 } else { 8 };
    if path == "prepared-trace-call" {
        let tx = traced(&x)?;
        let out = match op {
            "fft" => tx.fft(None, -1, FftNorm::Backward)?,
            "ifft" => tx.ifft(None, -1, FftNorm::Backward)?,
            "rfft" => tx.rfft(None, -1, FftNorm::Backward)?,
            "irfft" => tx.irfft(Some(n), -1, FftNorm::Backward)?,
            _ => unreachable!(),
        };
        return trace_measure(&[out], threads, bytes, runs, target);
    }
    let mut b = CpuBackend::with_threads(threads)?;
    let mut executor = FftExecutor::default();
    b.with_backend_session(|s| -> Result<Value> {
        with_cpu_exec_session(s, |s| -> Result<Value> {
            tensor_measure(bytes, runs, target, units, |()| {
                Ok(match op {
                    "fft" => executor.fft(&x, None, -1, FftNorm::Backward, s)?,
                    "ifft" => executor.ifft(&x, None, -1, FftNorm::Backward, s)?,
                    "rfft" => executor.rfft(&x, None, -1, FftNorm::Backward, s)?,
                    "irfft" => executor.irfft(&x, Some(n), -1, FftNorm::Backward, s)?,
                    _ => unreachable!(),
                })
            })
        })
        .expect("CPU session")
    })?
}

fn metadata(case: &Value, runs: usize, target: u128) -> Result<Value> {
    let op = case["op"].as_str().unwrap();
    let shape = if op == "transpose_view" {
        vec![2, 2]
    } else if op == "reshape_view" {
        vec![1024]
    } else {
        vec![4096]
    };
    let n: usize = shape.iter().product();
    let setup = |count| -> Result<Vec<TensorValue>> {
        (0..count)
            .map(|_| Ok(TensorValue::from_tensor(tensor(shape.clone(), 1, false)?)))
            .collect()
    };
    let cfg = SliceConfig {
        starts: vec![128],
        limits: vec![3968],
        strides: vec![2],
    };
    let call = |input: TensorValue| -> Result<TensorValue> {
        Ok(match op {
            "reshape_view" => input.reshape_view([32, 32])?,
            "transpose_view" => input.transpose_view([1, 0])?,
            "slice_view" => input.slice_view(&cfg)?,
            _ => unreachable!(),
        })
    };
    let first = call(setup(1)?.pop().unwrap())?;
    let sig = json!({"kind":"metadata","shape":first.shape(),"strides":first.strides(),"offset":first.offset(),"metadata_only":true});
    drop(first);
    let retained_bytes = n * 8 + 2 * std::mem::size_of::<TensorValue>() + 256;
    let mut row = sample(retained_bytes, runs, target, setup, call)?;
    row["outputs"] = json!([sig]);
    row["scope_note"]=json!("metadata-only on small owned inputs, no session; extent reduced to permit memory-bounded long batches; input copies untimed");
    Ok(row)
}

mod metadata_host;
mod permutation;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let id = args.get(1).ok_or("CASE_ID required")?;
    let path = args.get(2).map(String::as_str).unwrap_or("direct-shared");
    let threads = args.get(3).map(|x| x.parse()).transpose()?.unwrap_or(1);
    let runs = args.get(4).map(|x| x.parse()).transpose()?.unwrap_or(15);
    let target = std::env::var("CPU_FOLLOWUP_TARGET_NS")
        .ok()
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(10_000_000);
    let cases: Vec<Value> = serde_json::from_str(include_str!("../cases.json"))?;
    let case = cases
        .iter()
        .find(|c| c["id"] == *id)
        .ok_or("unknown case")?;
    let mut row = match case["category"].as_str().unwrap() {
        "real" => real(case, path, threads, runs, target)?,
        "activation" => activation(case, threads, runs, target)?,
        "index" => indexing(case, path, threads, runs, target)?,
        "structural" => structural(case, path, threads, runs, target)?,
        "complex" | "linalg" => complex_or_linalg(case, path, threads, runs, target)?,
        "fft" => fft(case, path, threads, runs, target)?,
        "metadata" if path == "metadata" => metadata(case, runs, target)?,
        "metadata" => metadata_host::run(case, path, runs, target)?,
        "perm" => permutation::run(case, path, threads, runs, target)?,
        _ => unreachable!(),
    };
    row["case_id"] = json!(id);
    row["fixture"] = case["fixture"].clone();
    row["path"] = json!(path);
    row["threads"] = json!(threads);
    row["correctness_status"] = json!("awaiting cross-implementation signature check");
    row["scope"] = json!({"outside_timer":["fixtures","runtime/backend/session creation","input wrappers/descriptors","graph construction/compilation/preparation","FFT plans/scratch priming","validation/signatures","output retention allocation","input restoration","output destruction"],"inside_timer":["declared API call","intrinsic output allocation","native completion"]});
    println!("{row}");
    Ok(())
}
