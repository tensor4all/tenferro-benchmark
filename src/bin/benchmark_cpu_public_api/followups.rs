//! Standard operation rows for APIs previously measured only in perf_issues.
use super::*;

pub(super) const SUITE: &str = "cpu/neural_public_api";
const NAMES: &[&str] = &[
    "erf",
    "sigmoid",
    "silu",
    "softplus",
    "gelu",
    "gelu_tanh",
    "reduce_mean",
    "softmax",
    "log_softmax",
    "masked_softmax",
    "layer_norm",
    "rms_norm",
    "take_along_axis",
];

pub(super) fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for shape in ["1024x64", "4096x256"] {
        for &name in NAMES {
            out.push(Case {suite:SUITE,benchmark:name,dtype:"f32",shape,
                notes:"public session API; axis=1; normalization eps=1e-5; affine=none; gelu=exact gelu_tanh=approximate; capture=inactive; tracked=false",
                run:unused});
        }
    }
    out
}
fn unused(_: &mut CpuEngine) -> tenferro_tensor::Result<()> {
    Err(tenferro_tensor::Error::unsupported(
        "followup",
        "use explicit setup sampler",
    ))
}

struct Fixture {
    x: Tensor,
    mask: Tensor,
    indices: Tensor,
    m: usize,
    n: usize,
}
impl Fixture {
    fn new(m: usize, n: usize) -> tenferro_tensor::Result<Self> {
        let data = (0..m * n)
            .map(|p| {
                (0.75 * pseudo_value(p, 1) + ((p % m * 13 + p / m * 7) % 31) as f64 / 32.0 - 0.5)
                    as f32
            })
            .collect::<Vec<_>>();
        Ok(Self {
            x: Tensor::from_vec_col_major([m, n], data)?,
            mask: Tensor::from_vec_col_major([m, n], data_bool(m * n))?,
            indices: Tensor::from_vec_col_major([m, n], data_i64(m * n, n))?,
            m,
            n,
        })
    }
    fn expected(&self, name: &str) -> BenchResult<Vec<f64>> {
        let x = self.x.as_slice::<f32>()?;
        let mask = self.mask.as_slice::<bool>()?;
        let indices = self.indices.as_slice::<i64>()?;
        let m = self.m;
        let n = self.n;
        let scalar = |x: f64| match name {
            "erf" => libm::erf(x),
            "sigmoid" => 1.0 / (1.0 + (-x).exp()),
            "silu" => x / (1.0 + (-x).exp()),
            "softplus" => x.exp().ln_1p(),
            "gelu" => 0.5 * x * (1.0 + libm::erf(x / 2.0_f64.sqrt())),
            _ => {
                0.5 * x
                    * (1.0
                        + (std::f64::consts::FRAC_2_PI.sqrt() * (x + 0.044715 * x * x * x)).tanh())
            }
        };
        if matches!(
            name,
            "erf" | "sigmoid" | "silu" | "softplus" | "gelu" | "gelu_tanh"
        ) {
            return Ok(x.iter().map(|&v| scalar(f64::from(v))).collect());
        }
        if name == "take_along_axis" {
            return Ok(indices
                .iter()
                .enumerate()
                .map(|(p, &j)| f64::from(x[p % m + m * j as usize]))
                .collect());
        }
        let mut out = vec![0.0; if name == "reduce_mean" { m } else { m * n }];
        for i in 0..m {
            let mean = (0..n).map(|j| f64::from(x[i + m * j])).sum::<f64>() / n as f64;
            if name == "reduce_mean" {
                out[i] = mean;
                continue;
            }
            if matches!(name, "layer_norm" | "rms_norm") {
                let center = if name == "layer_norm" { mean } else { 0.0 };
                let variance = (0..n)
                    .map(|j| (f64::from(x[i + m * j]) - center).powi(2))
                    .sum::<f64>()
                    / n as f64;
                for j in 0..n {
                    out[i + m * j] = (f64::from(x[i + m * j]) - center) / (variance + 1e-5).sqrt();
                }
            } else {
                let included = |j: usize| name != "masked_softmax" || mask[i + m * j];
                let maximum = (0..n)
                    .filter(|&j| included(j))
                    .map(|j| f64::from(x[i + m * j]))
                    .fold(f64::NEG_INFINITY, f64::max);
                let denom = (0..n)
                    .filter(|&j| included(j))
                    .map(|j| (f64::from(x[i + m * j]) - maximum).exp())
                    .sum::<f64>();
                for j in 0..n {
                    out[i + m * j] = if !included(j) {
                        0.0
                    } else if name == "log_softmax" {
                        f64::from(x[i + m * j]) - maximum - denom.ln()
                    } else {
                        (f64::from(x[i + m * j]) - maximum).exp() / denom
                    };
                }
            }
        }
        Ok(out)
    }
}

fn validate(out: &Tensor, expected: &[f64]) -> BenchResult<()> {
    let actual = out.as_slice::<f32>()?;
    if actual.len() != expected.len() {
        return Err("unexpected neural output length".into());
    }
    let mut residual = 0.0_f64;
    for (&a, &e) in actual.iter().zip(expected) {
        if !a.is_finite() {
            return Err("non-finite neural output".into());
        }
        residual = residual.max((f64::from(a) - e).abs() / (1.0 + e.abs()));
    }
    if residual > 2e-5 {
        return Err(format!("neural relative residual={residual}").into());
    }
    Ok(())
}

fn direct_op(f: &Fixture, name: &str, s: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
    match name {
        "erf" => f.x.erf(s),
        "sigmoid" => f.x.sigmoid(s),
        "silu" => f.x.silu(s),
        "softplus" => f.x.softplus(s),
        "gelu" => f.x.gelu(s),
        "gelu_tanh" => f.x.gelu_tanh(s),
        "reduce_mean" => f.x.reduce_mean(Some(&[1]), s),
        "softmax" => f.x.softmax(1, s),
        "log_softmax" => f.x.log_softmax(1, s),
        "masked_softmax" => f.x.masked_softmax(&f.mask, 1, s),
        "layer_norm" => f.x.layer_norm(1, None, None, 1e-5, s),
        "rms_norm" => f.x.rms_norm(1, None, None, 1e-5, s),
        _ => f.x.take_along_axis(&f.indices, 1, s),
    }
}

fn eager_op(
    name: &str,
    x: &EagerTensor,
    mask: &EagerTensor,
    indices: &EagerTensor,
    s: &mut tenferro_ad::EagerSession<'_>,
) -> tenferro_ad::Result<EagerTensor> {
    match name {
        "erf" => s.erf(x),
        "sigmoid" => s.sigmoid(x),
        "silu" => s.silu(x),
        "softplus" => s.softplus(x),
        "gelu" => s.gelu(x),
        "gelu_tanh" => s.gelu_tanh(x),
        "reduce_mean" => s.reduce_mean(x, Some(&[1])),
        "softmax" => s.softmax(x, 1),
        "log_softmax" => s.log_softmax(x, 1),
        "masked_softmax" => s.masked_softmax(x, mask, 1),
        "layer_norm" => s.layer_norm(x, 1, None, None, 1e-5),
        "rms_norm" => s.rms_norm(x, 1, None, None, 1e-5),
        _ => s.take_along_axis(x, indices, 1),
    }
}

fn timed(
    args: &Args,
    backend: &mut CpuEngine,
    case: &Case,
    eager: bool,
) -> BenchResult<(f64, f64)> {
    let mut shape = case.shape.split('x').map(str::parse::<usize>);
    let m = shape.next().unwrap()?;
    let n = shape.next().unwrap()?;
    let fixture = Fixture::new(m, n)?;
    let expected = fixture.expected(case.benchmark)?;
    let count = if m == 1024 { 64 } else { 8 };
    if !eager {
        backend.with_backend_session(|s| -> BenchResult<_> {
            validate(&direct_op(&fixture, case.benchmark, s)?, &expected)?;
            candidates::sample_count(args, count, || direct_op(&fixture, case.benchmark, s))
        })?
    } else {
        let runtime = EagerRuntime::new()?;
        let mut engine = cpu_backend()?;
        let _guard = runtime.no_grad();
        let x = EagerTensor::from_tensor_in(fixture.x.duplicate()?, Arc::clone(&runtime))?;
        let mask = EagerTensor::from_tensor_in(fixture.mask.duplicate()?, Arc::clone(&runtime))?;
        let indices =
            EagerTensor::from_tensor_in(fixture.indices.duplicate()?, Arc::clone(&runtime))?;
        runtime.with_eager_session(&mut engine, |s| -> BenchResult<_> {
            let output = eager_op(case.benchmark, &x, &mask, &indices, s)?;
            validate(
                output
                    .tensor_read()
                    .as_tensor()
                    .ok_or("eager output not materialized")?,
                &expected,
            )?;
            candidates::sample_count(args, count, || {
                eager_op(case.benchmark, &x, &mask, &indices, s)
            })
        })
    }
}

pub(super) fn emit(
    writer: &mut impl Write,
    args: &Args,
    backend: &mut CpuEngine,
    case: &Case,
) -> BenchResult<()> {
    if args.execution_filter.runs_direct() {
        candidates::row(
            writer,
            args,
            case,
            "tenferro-direct",
            timed(args, backend, case, false),
        )?;
        candidates::row(
            writer,
            args,
            case,
            "tenferro-eager",
            timed(args, backend, case, true),
        )?;
    }
    if args.execution_filter.runs_trace() {
        candidates::row(writer,args,case,"tenferro-trace",Err("unsupported: compiled trace cannot borrow the entered session; per-call execution setup excluded".into()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn neural_public_operations_match_scalar_references() -> BenchResult<()> {
        let f = Fixture::new(7, 9)?;
        let mut backend = tenferro_einsum_benchmark::cpu_engine(Some(1))?;
        backend.with_backend_session(|s| -> BenchResult<()> {
            for &name in NAMES {
                validate(&direct_op(&f, name, s)?, &f.expected(name)?)?;
            }
            Ok(())
        })?
    }
}
