//! Executable public-boundary case producer for cpu/small_work.
//!
//! Reuses the 154-case producer from feat/95-small-work (38b9a83), without
//! affinity/provenance machinery. Numerical checks precede calibrated timing;
//! outputs are black-boxed and dropped. Emits one JSON object on stdout.

use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use num_complex::Complex64;
use serde::Serialize;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend, CpuBackendKind};
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{
    BackendSession, BackendSessionHost, CompiledGraph, DType, GraphCompiler, Runtime,
    TensorSessionOpsExt, TraceContext,
};
use tenferro_tensor::{Tensor, TensorRead, TypedTensorView};

const CALIBRATION_MAX_ITERATIONS: usize = 1 << 30;
const CALIBRATION_DEADLINE_NS: u128 = 10_000_000_000;

#[derive(Serialize)]
struct Sample {
    process_index: usize,
    sample_index: usize,
    elapsed_ns: u128,
    iterations: usize,
}

fn arg(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn required_arg(name: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
    let value = arg(name, "");
    if value.is_empty() {
        return Err(format!("missing required selector argument {name}").into());
    }
    Ok(value)
}

fn parse_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error + Send + Sync>> {
    Ok(arg(name, &default.to_string()).parse()?)
}

fn inputs(
    size: usize,
    calls: usize,
) -> Result<(Tensor, Tensor, Vec<f64>), Box<dyn Error + Send + Sync>> {
    let a: Vec<f64> = (0..size).map(|i| 1.0 + (i % 4) as f64).collect();
    let b: Vec<f64> = (0..size).map(|i| -0.5 + (i % 3) as f64).collect();
    let expected = a
        .iter()
        .zip(&b)
        .map(|(x, y)| x + calls as f64 * y)
        .collect();
    Ok((
        Tensor::from_vec_col_major(vec![size], a)?,
        Tensor::from_vec_col_major(vec![size], b)?,
        expected,
    ))
}

fn matrix_dimension(size: usize) -> Result<usize, Box<dyn Error + Send + Sync>> {
    match size {
        4 => Ok(2),
        16 => Ok(4),
        64 => Ok(8),
        256 => Ok(16),
        1024 => Ok(32),
        _ => Err("matrix dimension must be 2, 4, 8, 16 or 32".into()),
    }
}

fn einsum_inputs(size: usize) -> Result<(Tensor, Tensor, Vec<f64>), Box<dyn Error + Send + Sync>> {
    let n = matrix_dimension(size)?;
    let a: Vec<f64> = (0..size).map(|i| (i % 7) as f64 * 0.25 - 0.5).collect();
    let b: Vec<f64> = (0..size).map(|i| (i % 5) as f64 * 0.5 - 0.75).collect();
    let expected = matmul_reference(n, &a, &b);
    Ok((
        Tensor::from_vec_col_major(vec![n, n], a)?,
        Tensor::from_vec_col_major(vec![n, n], b)?,
        expected,
    ))
}

fn broadcast_einsum_inputs(
    size: usize,
) -> Result<(Tensor, Tensor, Vec<f64>), Box<dyn Error + Send + Sync>> {
    let n = matrix_dimension(size)?;
    let (lhs, rhs, _) = einsum_inputs(size)?;
    let a = lhs.as_slice::<f64>()?[..n].repeat(n);
    // Positive B column sums avoid an all-zero product for the n4 fixture.
    let b = rhs.as_slice::<f64>()?[..n]
        .iter()
        .map(|x| x + 1.0)
        .collect::<Vec<_>>()
        .repeat(n);
    let expected = matmul_reference(n, &a, &b);
    Ok((
        Tensor::from_vec_col_major(vec![n, n], a)?,
        Tensor::from_vec_col_major(vec![n, n], b)?,
        expected,
    ))
}

fn matmul_reference<T>(n: usize, a: &[T], b: &[T]) -> Vec<T>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let mut expected = vec![T::default(); n * n];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                expected[i + n * k] += a[i + n * j] * b[j + n * k];
            }
        }
    }
    expected
}

fn complex_einsum_inputs(
    size: usize,
) -> Result<(Tensor, Tensor, Tensor), Box<dyn Error + Send + Sync>> {
    let n = matrix_dimension(size)?;
    let a: Vec<Complex64> = (0..size)
        .map(|i| Complex64::new((i % 7) as f64 * 0.25 - 0.5, (i % 3) as f64 * 0.125 - 0.125))
        .collect();
    let b: Vec<Complex64> = (0..size)
        .map(|i| Complex64::new((i % 5) as f64 * 0.5 - 0.75, (i % 4) as f64 * 0.25 + 0.125))
        .collect();
    let expected = matmul_reference(n, &a, &b);
    Ok((
        Tensor::from_vec_col_major(vec![n, n], a)?,
        Tensor::from_vec_col_major(vec![n, n], b)?,
        Tensor::from_vec_col_major(vec![n, n], expected)?,
    ))
}

fn complex_broadcast_einsum_inputs(
    size: usize,
) -> Result<(Tensor, Tensor, Tensor), Box<dyn Error + Send + Sync>> {
    let n = matrix_dimension(size)?;
    let (lhs, rhs, _) = complex_einsum_inputs(size)?;
    let a = lhs.as_slice::<Complex64>()?[..n].repeat(n);
    let b = rhs.as_slice::<Complex64>()?[..n].repeat(n);
    let expected = matmul_reference(n, &a, &b);
    Ok((
        Tensor::from_vec_col_major(vec![n, n], a)?,
        Tensor::from_vec_col_major(vec![n, n], b)?,
        Tensor::from_vec_col_major(vec![n, n], expected)?,
    ))
}

fn reduce_sum_input(size: usize) -> Result<(Tensor, Tensor), Box<dyn Error + Send + Sync>> {
    let values: Vec<f64> = (0..size).map(|i| (i % 7) as f64 * 0.25 - 0.5).collect();
    let expected: f64 = values.iter().sum();
    Ok((
        Tensor::from_vec_col_major(vec![size], values)?,
        Tensor::from_vec_col_major(vec![], vec![expected])?,
    ))
}

fn gather_inputs(size: usize) -> Result<(Tensor, Tensor, Vec<f64>), Box<dyn Error + Send + Sync>> {
    let data: Vec<f64> = (0..size).map(|i| i as f64 / 8.0 - 3.0).collect();
    let indices: Vec<i64> = (0..size)
        .map(|i| ((3 * i + i / 2 + 1) % size) as i64)
        .collect();
    let expected = indices.iter().map(|&index| data[index as usize]).collect();
    Ok((
        Tensor::from_vec_col_major(vec![size], data)?,
        Tensor::from_vec_col_major(vec![size], indices)?,
        expected,
    ))
}

fn solve_inputs(size: usize) -> Result<(Tensor, Tensor, Vec<f64>), Box<dyn Error + Send + Sync>> {
    let n = matrix_dimension(size)?;
    let mut a = vec![0.0; size];
    let mut x = vec![0.0; size];
    for j in 0..n {
        for i in 0..n {
            a[i + n * j] = if i == j {
                n as f64 + 2.0 + i as f64 / 8.0
            } else {
                (1 + (3 * i + 5 * j) % 7) as f64 / 32.0
            };
            x[i + n * j] =
                ((3 * i + 5 * j) % 13) as f64 / 16.0 - 6.0 / 16.0 + (i as f64 - j as f64) / 64.0;
        }
    }
    let b = matmul_reference(n, &a, &x);
    Ok((
        Tensor::from_vec_col_major(vec![n, n], a)?,
        Tensor::from_vec_col_major(vec![n, n], b)?,
        x,
    ))
}

fn check_solve_result(
    output: &Tensor,
    a: &Tensor,
    b: &Tensor,
    expected: &Tensor,
) -> Result<f64, Box<dyn Error + Send + Sync>> {
    let value = check_tensor_reference(output, expected)?;
    let product = Tensor::from_vec_col_major(
        a.shape().to_vec(),
        matmul_reference(
            a.shape()[0],
            a.as_slice::<f64>()?,
            output.as_slice::<f64>()?,
        ),
    )?;
    check_tensor_reference(&product, b)?;
    Ok(value)
}

fn trace_compile_einsum(
    n: usize,
    dtype: DType,
) -> Result<CompiledGraph, Box<dyn Error + Send + Sync>> {
    use tenferro_einsum::TraceContextEinsumExt;
    let mut trace = TraceContext::new();
    let a = trace.input(ProgramInputSpec::new(dtype, [n.into(), n.into()]))?;
    let b = trace.input(ProgramInputSpec::new(dtype, [n.into(), n.into()]))?;
    let output = trace.einsum(&[a, b], "ij,jk->ik")?;
    let graph = trace.finish(&[output])?;
    Ok(GraphCompiler::new().compile_traced_graph(&graph)?)
}

fn compiled_einsum(
    backend: &CpuBackend,
    n: usize,
    dtype: DType,
) -> Result<(Runtime, CompiledGraph), Box<dyn Error + Send + Sync>> {
    let program = trace_compile_einsum(n, dtype)?;
    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration(backend)?)?;
    builder.install_extension_module(tenferro_einsum::extension_module::<CpuBackend>(
        runtime_engine_id()?,
    )?)?;
    Ok((builder.build()?, program))
}

fn check_compiled_einsum(
    runtime: &Runtime,
    program: &CompiledGraph,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<f64, Box<dyn Error + Send + Sync>> {
    let n = matrix_dimension(lhs.shape().iter().product())?;
    let mut checked = 0.0;
    for (a, b) in [(lhs, rhs), (rhs, lhs), (lhs, rhs)] {
        let expected = if a.dtype() == DType::F64 {
            Tensor::from_vec_col_major(
                vec![n, n],
                matmul_reference(n, a.as_slice::<f64>()?, b.as_slice::<f64>()?),
            )?
        } else {
            Tensor::from_vec_col_major(
                vec![n, n],
                matmul_reference(n, a.as_slice::<Complex64>()?, b.as_slice::<Complex64>()?),
            )?
        };
        let outputs = runtime.run_compiled(program, &[a, b])?;
        if outputs.len() != 1 {
            return Err("compiled einsum must return one output".into());
        }
        checked = check_tensor_reference(&outputs[0], &expected)?;
    }
    Ok(checked)
}

enum BorrowedEinsumViews<'a> {
    F64([TypedTensorView<'a, f64>; 2]),
    C64([TypedTensorView<'a, Complex64>; 2]),
}

struct BorrowedEinsumInputs {
    storage: [Tensor; 2],
    n: usize,
    strides: [isize; 2],
    offset: usize,
}

impl BorrowedEinsumInputs {
    fn new(lhs: &Tensor, rhs: &Tensor, layout: &str) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let n = matrix_dimension(lhs.shape().iter().product())?;
        let (strides, offset) = match layout {
            "col_major_contiguous" => ([1, n as isize], 0),
            "row_major_contiguous" => ([n as isize, 1], 0),
            "broadcast" => ([1, 0], 0),
            "strided" => ([2, (2 * n + 1) as isize], 1),
            _ => return Err(format!("unsupported borrowed layout: {layout}").into()),
        };
        fn store<T: tenferro_tensor::TensorScalar>(
            input: &Tensor,
            n: usize,
            strides: [isize; 2],
            offset: usize,
            padding: T,
        ) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
            let length = offset + (n - 1) * (strides[0] + strides[1]) as usize + 1;
            let mut destination = vec![padding; length];
            let values = input.as_slice::<T>()?;
            for j in 0..n {
                for i in 0..n {
                    destination[offset + i * strides[0] as usize + j * strides[1] as usize] =
                        values[i + n * j];
                }
            }
            Ok(Tensor::from_vec_col_major(vec![length], destination)?)
        }
        let copy = |input: &Tensor| match input.dtype() {
            DType::F64 => store(input, n, strides, offset, f64::NAN),
            DType::C64 => store(
                input,
                n,
                strides,
                offset,
                Complex64::new(f64::NAN, f64::NAN),
            ),
            _ => Err("unsupported borrowed dtype".into()),
        };
        let storage = [copy(lhs)?, copy(rhs)?];
        Ok(Self {
            storage,
            n,
            strides,
            offset,
        })
    }

    fn views(&self) -> Result<BorrowedEinsumViews<'_>, Box<dyn Error + Send + Sync>> {
        match self.storage[0].dtype() {
            DType::F64 => Ok(BorrowedEinsumViews::F64(self.typed_views()?)),
            DType::C64 => Ok(BorrowedEinsumViews::C64(self.typed_views()?)),
            _ => Err("unsupported borrowed dtype".into()),
        }
    }

    fn typed_views<T: tenferro_tensor::TensorScalar>(
        &self,
    ) -> Result<[TypedTensorView<'_, T>; 2], Box<dyn Error + Send + Sync>> {
        Ok([
            TypedTensorView::from_slice(
                vec![self.n, self.n],
                self.strides,
                self.offset as isize,
                self.storage[0].as_slice::<T>()?,
            )?,
            TypedTensorView::from_slice(
                vec![self.n, self.n],
                self.strides,
                self.offset as isize,
                self.storage[1].as_slice::<T>()?,
            )?,
        ])
    }
}

fn borrowed_einsum(
    inputs: &BorrowedEinsumViews<'_>,
    session: &mut dyn BackendSession,
) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
    use tenferro_einsum::TypedTensorReadEinsumExt;
    Ok(match inputs {
        BorrowedEinsumViews::F64(views) => views.einsum_read("ij,jk->ik", session)?.into(),
        BorrowedEinsumViews::C64(views) => views.einsum_read("ij,jk->ik", session)?.into(),
    })
}

fn concrete_operation(
    operation: &str,
    session: &mut dyn BackendSession,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
    use tenferro_einsum::TensorEinsumExt;
    use tenferro_linalg::TensorLinalgExt;
    static GATHER: std::sync::OnceLock<tenferro_tensor::GatherConfig> = std::sync::OnceLock::new();
    let gather = GATHER.get_or_init(|| tenferro_tensor::GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    });
    Ok(match operation {
        "add" => lhs.add(rhs, session)?,
        "einsum" => [lhs, rhs].einsum("ij,jk->ik", session)?,
        "solve" => lhs.solve(rhs, session)?,
        "reduce_sum" => session.reduce_sum(lhs, &[0])?,
        "gather" => session.gather(lhs, rhs, gather)?,
        _ => return Err(format!("unsupported operation: {operation}").into()),
    })
}

fn concrete_fresh(
    operation: &str,
    backend: &mut CpuBackend,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
    backend.with_backend_session(|session| concrete_operation(operation, session, lhs, rhs))
}

fn add_chain<T, F>(
    lhs: &T,
    rhs: &T,
    calls: usize,
    mut add: F,
) -> Result<T, Box<dyn Error + Send + Sync>>
where
    F: FnMut(&T, &T) -> Result<T, Box<dyn Error + Send + Sync>>,
{
    let mut output = add(lhs, rhs)?;
    for _ in 1..calls {
        output = add(&output, rhs)?;
    }
    Ok(output)
}

fn eager_operation(
    operation: &str,
    lhs: &EagerTensor,
    rhs: &EagerTensor,
) -> Result<EagerTensor, Box<dyn Error + Send + Sync>> {
    use tenferro_einsum::{EagerEinsumExt, EinsumSubscripts};
    static SUBS: std::sync::OnceLock<EinsumSubscripts> = std::sync::OnceLock::new();
    let subs = SUBS.get_or_init(|| EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]));
    Ok(match operation {
        "add" => lhs.add(rhs)?,
        "einsum" => [lhs, rhs].einsum_subscripts(subs)?,
        _ => return Err(format!("unsupported operation: {operation}").into()),
    })
}

fn case_descriptor(
    operation: &str,
    dtype: &str,
    api_tier: &str,
    workflow: &str,
    size: usize,
    calls: usize,
    provider: &str,
) -> Result<serde_json::Value, Box<dyn Error + Send + Sync>> {
    let supported_dtype = dtype == "f64"
        || (dtype == "c64"
            && operation == "einsum"
            && matches!(
                api_tier,
                "concrete-fresh"
                    | "concrete-shared"
                    | "prepared-setup"
                    | "prepared-repeat"
                    | "eager-no-ad"
                    | "eager-ad"
                    | "compiled-repeat"
                    | "compiled-setup"
                    | "borrowed-fresh"
                    | "borrowed-shared"
            ));
    if !matches!(
        operation,
        "add" | "einsum" | "solve" | "gather" | "reduce_sum"
    ) || !supported_dtype
        || !(matches!(size, 4 | 16 | 256) || (operation == "einsum" && matches!(size, 64 | 1024)))
    {
        return Err(format!(
            "unsupported small-work selector: operation={operation}, dtype={dtype}"
        )
        .into());
    }
    let expected_calls = match workflow {
        "single" => 1,
        "dependent10" => 10,
        _ => return Err(format!("unsupported small-work workflow: {workflow}").into()),
    };
    if calls != expected_calls {
        return Err(
            format!("workflow {workflow} requires calls_per_workflow={expected_calls}").into(),
        );
    }
    if matches!(operation, "einsum" | "solve" | "gather" | "reduce_sum") && workflow != "single" {
        return Err(format!("{operation} currently requires the single workflow").into());
    }
    if matches!(operation, "solve" | "gather" | "reduce_sum")
        && !matches!(api_tier, "concrete-fresh" | "concrete-shared")
    {
        return Err(format!("{operation} currently requires a concrete fresh/shared tier").into());
    }
    if api_tier.starts_with("borrowed-") && operation != "einsum" {
        return Err("borrowed tiers require einsum".into());
    }
    let (contract_id, family, surface, timer, outside_timer) = match api_tier {
        "concrete-fresh" | "borrowed-fresh" => (
            "core.add.ordinary.concrete",
            "core",
            "concrete",
            vec!["session_entry_exit", "add", "output_allocation"],
            vec![
                "backend_construction",
                "input_construction",
                "correctness_check",
            ],
        ),
        "concrete-shared" | "borrowed-shared" => (
            "core.add.ordinary.concrete",
            "core",
            "concrete",
            vec!["add", "output_allocation"],
            vec![
                "backend_construction",
                "input_construction",
                "session_entry_exit",
                "shared_admission_exit",
                "correctness_check",
            ],
        ),
        "eager-no-ad" => (
            "core.add.ordinary.eager",
            "core",
            "eager",
            vec!["add", "eager_dispatch", "output_allocation"],
            vec![
                "backend_construction",
                "input_construction",
                "eager_runtime_construction",
                "correctness_check",
            ],
        ),
        "eager-ad" => (
            "core.add.ordinary.eager",
            "core",
            "eager",
            vec![
                "add_forward_recording",
                "eager_dispatch",
                "output_allocation",
            ],
            vec![
                "backend_construction",
                "input_construction",
                "eager_runtime_construction",
                "backward",
                "correctness_check",
            ],
        ),
        "prepared-setup" if operation == "einsum" => (
            "einsum.einsum.prepare.concrete",
            "einsum",
            "concrete",
            vec!["prepare", "plan_allocation"],
            vec![
                "backend_construction",
                "input_construction",
                "correctness_check",
            ],
        ),
        "compiled-setup" if operation == "einsum" => (
            "einsum.einsum.prepare.traced",
            "einsum",
            "traced",
            vec!["trace_compile", "program_allocation"],
            vec![
                "backend_construction",
                "input_construction",
                "runtime_construction",
                "input_bindings",
                "correctness_check",
            ],
        ),
        "compiled-repeat" if operation == "einsum" => (
            "einsum.einsum.prepared.traced",
            "einsum",
            "traced",
            vec!["runtime_admission_execution", "output_allocation"],
            vec![
                "backend_construction",
                "input_construction",
                "trace_compile",
                "runtime_construction",
                "input_bindings",
                "correctness_check",
            ],
        ),
        "prepared-repeat" if operation == "einsum" => (
            "einsum.einsum.prepared.concrete",
            "einsum",
            "concrete",
            vec!["prepared_execute", "output_allocation"],
            vec![
                "backend_construction",
                "input_construction",
                "plan_preparation",
                "session_entry_exit",
                "shared_admission_exit",
                "correctness_check",
            ],
        ),
        _ => return Err(format!("unsupported small-work API tier: {api_tier}").into()),
    };
    let (contract_id, family, shape) = if operation == "einsum" {
        (
            if api_tier.starts_with("prepared-") || api_tier.starts_with("compiled-") {
                contract_id.to_string()
            } else {
                format!("einsum.einsum.ordinary.{surface}")
            },
            "einsum",
            vec![matrix_dimension(size)?, matrix_dimension(size)?],
        )
    } else if operation == "solve" {
        (
            "linalg.solve.ordinary.concrete".to_string(),
            "linalg",
            vec![matrix_dimension(size)?, matrix_dimension(size)?],
        )
    } else if operation == "reduce_sum" {
        (
            "core.reduce_sum.ordinary.concrete".to_string(),
            "core",
            vec![size],
        )
    } else if operation == "gather" {
        (
            "core.gather.ordinary.concrete".to_string(),
            "core",
            vec![size],
        )
    } else {
        (contract_id.to_string(), family, vec![size])
    };
    let mut outside_timer = outside_timer;
    outside_timer.push("output_destruction");
    let timer: Vec<String> = timer
        .iter()
        .map(|label| label.replace("add", operation))
        .collect();
    outside_timer.push("operation_config_construction");
    Ok(serde_json::json!({
        "contract_id": contract_id, "family": family, "surface": surface,
        "operation": operation, "phase": if api_tier.ends_with("-setup") { "setup" } else { "execution" }, "api_tier": api_tier,
        "backend": "tenferro-rs", "provider": if api_tier.ends_with("-setup") { "not-applicable" } else { provider }, "dtype": dtype,
        "layout": "col_major_contiguous", "shape": shape, "workflow": workflow,
        "calls_per_workflow": calls,
        "setup": {"includes": timer, "excludes": outside_timer},
        "scope": {"timer": timer, "outside_timer": outside_timer},
        "descriptor_source": "rust_case_selection"
    }))
}

fn check_tensor_shape(
    output: &Tensor,
    expected: &[f64],
    shape: &[usize],
) -> Result<f64, Box<dyn Error + Send + Sync>> {
    if output.shape() != shape {
        return Err("output shape does not match expected shape".into());
    }
    let values = TensorRead::from_tensor(output).as_slice::<f64>()?;
    if values.len() != expected.len()
        || values.iter().zip(expected).any(|(actual, expected)| {
            !actual.is_finite() || !expected.is_finite() || (*actual - *expected).abs() > 1e-12
        })
    {
        return Err("output values do not match expected elementwise result".into());
    }
    Ok(values.iter().copied().sum())
}

fn check_tensor_reference(
    output: &Tensor,
    expected: &Tensor,
) -> Result<f64, Box<dyn Error + Send + Sync>> {
    if expected.dtype() == DType::F64 {
        return check_tensor_shape(output, expected.as_slice::<f64>()?, expected.shape());
    }
    if output.shape() != expected.shape() {
        return Err("output shape does not match expected shape".into());
    }
    let actual = output.as_slice::<Complex64>()?;
    let reference = expected.as_slice::<Complex64>()?;
    if actual.len() != reference.len()
        || actual.iter().zip(reference).any(|(a, b)| {
            !a.re.is_finite()
                || !a.im.is_finite()
                || !b.re.is_finite()
                || !b.im.is_finite()
                || (a.re - b.re).abs() > 1e-12
                || (a.im - b.im).abs() > 1e-12
        })
    {
        return Err("complex output components do not match independent reference".into());
    }
    Ok(actual.iter().map(|z| z.re + z.im).sum())
}

fn check_tensor(output: &Tensor, expected: &[f64]) -> Result<f64, Box<dyn Error + Send + Sync>> {
    check_tensor_shape(output, expected, &[expected.len()])
}

fn check_einsum_ad(
    output: &EagerTensor,
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    a: &Tensor,
    b: &Tensor,
    expected: &[f64],
) -> Result<f64, Box<dyn Error + Send + Sync>> {
    let value = check_tensor_shape(&output.to_tensor()?, expected, a.shape())?;
    let loss = output.reduce_sum(Some(&[0, 1]))?;
    let _ = loss.backward()?;
    let av = TensorRead::from_tensor(a).as_slice::<f64>()?;
    let bv = TensorRead::from_tensor(b).as_slice::<f64>()?;
    let n = a.shape()[0];
    let mut da = vec![0.0; expected.len()];
    let mut db = vec![0.0; expected.len()];
    for j in 0..n {
        for i in 0..n {
            da[i + n * j] = (0..n).map(|k| bv[j + n * k]).sum();
            db[j + n * i] = (0..n).map(|k| av[k + n * j]).sum();
        }
    }
    for (input, expected_gradient) in [(lhs, da), (rhs, db)] {
        let gradient = input
            .grad()?
            .ok_or("active einsum AD gradient is missing")?;
        if gradient.shape() != a.shape()
            || gradient
                .as_slice::<f64>()?
                .iter()
                .zip(&expected_gradient)
                .any(|(x, y)| !x.is_finite() || (*x - y).abs() > 1e-12)
        {
            return Err("active AD gradient does not match analytic einsum gradient".into());
        }
        input.clear_grad()?;
    }
    Ok(value)
}

fn check_complex_einsum_ad(
    output: &EagerTensor,
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    a: &Tensor,
    b: &Tensor,
    expected: &Tensor,
) -> Result<f64, Box<dyn Error + Send + Sync>> {
    let value = check_tensor_reference(&output.to_tensor()?, expected)?;
    output.reduce_sum(Some(&[0, 1]))?.backward()?;
    let av = a.as_slice::<Complex64>()?;
    let bv = b.as_slice::<Complex64>()?;
    let n = a.shape()[0];
    // Unit complex seed represents Re(sum(A B)) under the Hermitian convention.
    let mut da = vec![Complex64::new(0.0, 0.0); n * n];
    let mut db = da.clone();
    for j in 0..n {
        for i in 0..n {
            da[i + n * j] = (0..n).map(|k| bv[j + n * k].conj()).sum();
            db[j + n * i] = (0..n).map(|k| av[k + n * j].conj()).sum();
        }
    }
    for (input, values) in [(lhs, da), (rhs, db)] {
        let gradient = input
            .grad()?
            .ok_or("active complex einsum AD gradient is missing")?;
        let expected = Tensor::from_vec_col_major(vec![n, n], values)?;
        check_tensor_reference(&gradient.to_tensor()?, &expected)?;
        input.clear_grad()?;
    }
    Ok(value)
}

fn execute_fresh(
    operation: &str,
    backend: &mut CpuBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    calls: usize,
) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
    add_chain(lhs, rhs, calls, |a, b| {
        concrete_fresh(operation, backend, a, b)
    })
}

fn execute_shared(
    operation: &str,
    session: &mut dyn BackendSession,
    lhs: &Tensor,
    rhs: &Tensor,
    calls: usize,
) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
    add_chain(lhs, rhs, calls, |a, b| {
        concrete_operation(operation, session, a, b)
    })
}

fn calibrate<F>(
    target_ns: u128,
    mut batch: F,
) -> Result<(usize, u128), Box<dyn Error + Send + Sync>>
where
    F: FnMut(usize) -> Result<u128, Box<dyn Error + Send + Sync>>,
{
    let calibration_start = Instant::now();
    let mut iterations = 1usize;
    loop {
        let elapsed = batch(iterations)?;
        if calibration_start.elapsed().as_nanos() >= CALIBRATION_DEADLINE_NS {
            return Err(
                format!("calibration deadline exceeded before target_ns={target_ns}").into(),
            );
        }
        if elapsed >= target_ns {
            return Ok((iterations, elapsed));
        }
        if iterations >= CALIBRATION_MAX_ITERATIONS {
            return Err("calibration iteration cap exceeded before target duration".into());
        }
        iterations = iterations
            .checked_mul(2)
            .ok_or("calibration iteration overflow")?;
    }
}

struct Measurement {
    iterations: usize,
    calibrated_ns: u128,
    samples: Vec<Sample>,
}

fn measure<F, O>(
    warmups: usize,
    samples_count: usize,
    target_ns: u128,
    process_index: usize,
    sample_start: usize,
    mut execute: F,
) -> Result<Measurement, Box<dyn Error + Send + Sync>>
where
    F: FnMut() -> Result<O, Box<dyn Error + Send + Sync>>,
{
    for _ in 0..warmups.max(1) {
        execute()?;
    }
    // Sum execution intervals: output destruction occurs between intervals.
    // This deliberately includes clock overhead, recorded in the report.
    let mut batch = |iterations| -> Result<u128, Box<dyn Error + Send + Sync>> {
        let mut elapsed = 0;
        for _ in 0..iterations {
            let start = Instant::now();
            let output = execute()?;
            elapsed += start.elapsed().as_nanos();
            black_box(output);
        }
        Ok(elapsed)
    };
    let (iterations, calibrated_ns) = calibrate(target_ns, &mut batch)?;
    let mut samples = Vec::with_capacity(samples_count);
    for sample_index in 0..samples_count {
        let elapsed_ns = batch(iterations)?;
        if elapsed_ns == 0 {
            return Err(
                "timed sample elapsed zero nanoseconds; raw duration not fabricated".into(),
            );
        }
        samples.push(Sample {
            process_index,
            sample_index: sample_start + sample_index,
            elapsed_ns,
            iterations,
        });
    }
    Ok(Measurement {
        iterations,
        calibrated_ns,
        samples,
    })
}

fn execute_eager(
    operation: &str,
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    calls: usize,
) -> Result<EagerTensor, Box<dyn Error + Send + Sync>> {
    add_chain(lhs, rhs, calls, |a, b| eager_operation(operation, a, b))
}

fn check_eager(
    output: &EagerTensor,
    expected: &[f64],
) -> Result<f64, Box<dyn Error + Send + Sync>> {
    let tensor = output.to_tensor()?;
    check_tensor(&tensor, expected)
}

fn check_active_ad(
    output: &EagerTensor,
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    expected: &[f64],
    rhs_gradient: f64,
) -> Result<f64, Box<dyn Error + Send + Sync>> {
    let value = check_eager(output, expected)?;
    let loss = output.reduce_sum(Some(&[0]))?;
    let _ = loss.backward()?;
    for (input, expected_gradient) in [(lhs, 1.0), (rhs, rhs_gradient)] {
        let gradient = input.grad()?.ok_or("active AD gradient is missing")?;
        if gradient.shape() != [expected.len()]
            || gradient.as_slice::<f64>()?.iter().any(|value| {
                !value.is_finite()
                    || !expected_gradient.is_finite()
                    || (*value - expected_gradient).abs() > 1e-12
            })
        {
            return Err("active AD gradient does not match analytic add gradient".into());
        }
    }
    lhs.clear_grad()?;
    rhs.clear_grad()?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn einsum_results_and_gradients_match_independent_reference() {
        for size in [4, 16, 256] {
            let (a, b, expected) = einsum_inputs(size).unwrap();
            if size == 4 {
                assert_eq!(expected, vec![0.375, 0.125, -0.125, 0.125]);
            }
            let mut backend = CpuBackend::new();
            let actual = concrete_fresh("einsum", &mut backend, &a, &b).unwrap();
            check_tensor_shape(&actual, &expected, a.shape()).unwrap();
            let runtime = EagerRuntime::with_cpu_backend(backend).unwrap();
            let (left, right, _) = einsum_inputs(size).unwrap();
            let lhs = EagerTensor::requires_grad_in(left, runtime.clone()).unwrap();
            let rhs = EagerTensor::requires_grad_in(right, runtime).unwrap();
            let output = eager_operation("einsum", &lhs, &rhs).unwrap();
            check_einsum_ad(&output, &lhs, &rhs, &a, &b, &expected).unwrap();
        }
    }

    #[test]
    fn complex_einsum_matches_both_components_in_fresh_and_shared_sessions() {
        for size in [4, 16, 256] {
            let (lhs, rhs, expected) = complex_einsum_inputs(size).unwrap();
            if size == 4 {
                assert_eq!(
                    expected.as_slice::<Complex64>().unwrap(),
                    &[
                        Complex64::new(0.34375, 0.0),
                        Complex64::new(0.171875, 0.09375),
                        Complex64::new(-0.15625, -0.25),
                        Complex64::new(0.234375, -0.03125),
                    ]
                );
            }
            let mut backend = CpuBackend::new();
            let fresh = concrete_fresh("einsum", &mut backend, &lhs, &rhs).unwrap();
            check_tensor_reference(&fresh, &expected).unwrap();
            backend
                .with_backend_session(|session| {
                    let output = concrete_operation("einsum", session, &lhs, &rhs)?;
                    check_tensor_reference(&output, &expected)
                })
                .unwrap();
            let values = expected.as_slice::<Complex64>().unwrap();
            let conjugated = Tensor::from_vec_col_major(
                expected.shape().to_vec(),
                values.iter().map(|z| z.conj()).collect(),
            )
            .unwrap();
            assert!(check_tensor_reference(&conjugated, &expected).is_err());
            let real_only = Tensor::from_vec_col_major(
                expected.shape().to_vec(),
                values.iter().map(|z| z.re).collect(),
            )
            .unwrap();
            assert!(check_tensor_reference(&real_only, &expected).is_err());
            let mut nonfinite = values.to_vec();
            nonfinite[0].im = f64::NAN;
            let invalid = Tensor::from_vec_col_major(expected.shape().to_vec(), nonfinite).unwrap();
            assert!(check_tensor_reference(&invalid, &expected).is_err());
            assert!(check_tensor_reference(&expected, &invalid).is_err());
        }
        for tier in ["borrowed-fresh", "borrowed-shared"] {
            assert!(case_descriptor("einsum", "c32", tier, "single", 4, 1, "faer").is_err());
        }
    }

    #[test]
    fn complex_prepared_einsum_executes_and_revalidates_rebound_inputs() {
        for size in [4, 16, 256] {
            let (lhs, rhs, expected) = complex_einsum_inputs(size).unwrap();
            let plan =
                tenferro_einsum::ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik").unwrap();
            let swapped = Tensor::from_vec_col_major(
                lhs.shape().to_vec(),
                matmul_reference(
                    matrix_dimension(size).unwrap(),
                    rhs.as_slice::<Complex64>().unwrap(),
                    lhs.as_slice::<Complex64>().unwrap(),
                ),
            )
            .unwrap();
            assert_ne!(
                expected.as_slice::<Complex64>().unwrap(),
                swapped.as_slice::<Complex64>().unwrap()
            );
            let mut backend = CpuBackend::new();
            backend
                .with_backend_session(|session| {
                    for (a, b, reference) in [
                        (&lhs, &rhs, &expected),
                        (&rhs, &lhs, &swapped),
                        (&lhs, &rhs, &expected),
                    ] {
                        check_tensor_reference(&plan.execute([a, b], session)?, reference)?;
                    }
                    let wrong_dtype =
                        Tensor::from_vec_col_major(lhs.shape().to_vec(), vec![1.0_f64; size])?;
                    assert!(plan.execute([&lhs, &wrong_dtype], session).is_err());
                    let wrong_shape = Tensor::from_vec_col_major(
                        vec![size],
                        vec![Complex64::new(1.0, 0.0); size],
                    )?;
                    assert!(plan.execute([&lhs, &wrong_shape], session).is_err());
                    Ok::<_, Box<dyn Error + Send + Sync>>(())
                })
                .unwrap();
            let setup =
                case_descriptor("einsum", "c64", "prepared-setup", "single", size, 1, "faer")
                    .unwrap();
            assert_eq!(setup["contract_id"], "einsum.einsum.prepare.concrete");
            assert_eq!(setup["phase"], "setup");
            assert_eq!(setup["provider"], "not-applicable");
            let repeat = case_descriptor(
                "einsum",
                "c64",
                "prepared-repeat",
                "single",
                size,
                1,
                "faer",
            )
            .unwrap();
            assert_eq!(repeat["contract_id"], "einsum.einsum.prepared.concrete");
            assert_eq!(repeat["phase"], "execution");
        }
    }

    #[test]
    fn reduce_sum_checks_scalar_result_and_preserves_input() {
        for size in [4, 16, 256] {
            let (input, expected) = reduce_sum_input(size).unwrap();
            let original = input.as_slice::<f64>().unwrap().to_vec();
            assert_eq!(expected.shape(), &[] as &[usize]);
            let mut backend = CpuBackend::new();
            let output = concrete_fresh("reduce_sum", &mut backend, &input, &input).unwrap();
            check_tensor_reference(&output, &expected).unwrap();
            assert_eq!(input.as_slice::<f64>().unwrap(), original);
            backend
                .with_backend_session(|session| {
                    for _ in 0..2 {
                        let output = concrete_operation("reduce_sum", session, &input, &input)?;
                        check_tensor_reference(&output, &expected)?;
                        assert_eq!(input.as_slice::<f64>()?, original);
                    }
                    assert!(session.reduce_sum(&input, &[1]).is_err());
                    Ok::<_, Box<dyn Error + Send + Sync>>(())
                })
                .unwrap();
            for tier in ["concrete-fresh", "concrete-shared"] {
                let descriptor =
                    case_descriptor("reduce_sum", "f64", tier, "single", size, 1, "faer").unwrap();
                assert_eq!(
                    descriptor["contract_id"],
                    "core.reduce_sum.ordinary.concrete"
                );
                assert_eq!(descriptor["shape"], serde_json::json!([size]));
            }
        }
        assert!(
            case_descriptor("reduce_sum", "f64", "eager-no-ad", "single", 4, 1, "faer").is_err()
        );
        assert!(case_descriptor(
            "reduce_sum",
            "c64",
            "concrete-fresh",
            "single",
            4,
            1,
            "faer"
        )
        .is_err());
    }

    #[test]
    fn gather_preserves_repeated_index_order_and_inputs() {
        for size in [4, 16, 256] {
            let (data, indices, expected) = gather_inputs(size).unwrap();
            let before_data = data.as_slice::<f64>().unwrap().to_vec();
            let before_indices = indices.as_slice::<i64>().unwrap().to_vec();
            assert_eq!(indices.dtype(), DType::I64);
            assert!(before_indices.windows(2).any(|pair| pair[0] > pair[1]));
            assert!(
                before_indices
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    < size
            );
            let mut backend = CpuBackend::new();
            let output = concrete_fresh("gather", &mut backend, &data, &indices).unwrap();
            check_tensor_shape(&output, &expected, &[size]).unwrap();
            backend
                .with_backend_session(|session| {
                    let output = concrete_operation("gather", session, &data, &indices)?;
                    check_tensor_shape(&output, &expected, &[size])?;
                    assert!(concrete_operation("gather", session, &data, &data).is_err());
                    Ok::<_, Box<dyn Error + Send + Sync>>(())
                })
                .unwrap();
            assert_eq!(data.as_slice::<f64>().unwrap(), &before_data);
            assert_eq!(indices.as_slice::<i64>().unwrap(), &before_indices);
            for tier in ["concrete-fresh", "concrete-shared"] {
                let descriptor =
                    case_descriptor("gather", "f64", tier, "single", size, 1, "faer").unwrap();
                assert_eq!(descriptor["contract_id"], "core.gather.ordinary.concrete");
                assert_eq!(descriptor["shape"], serde_json::json!([size]));
            }
        }
        assert!(case_descriptor("gather", "f64", "eager-no-ad", "single", 4, 1, "faer").is_err());
    }

    #[test]
    fn requested_binary_sizes_cover_real_complex_prepare_and_ordinary() {
        for (size, n) in [(64, 8), (1024, 32)] {
            for dtype in ["f64", "c64"] {
                let (a, b, expected) = if dtype == "f64" {
                    let (a, b, values) = einsum_inputs(size).unwrap();
                    (
                        a,
                        b,
                        Tensor::from_vec_col_major(vec![n, n], values).unwrap(),
                    )
                } else {
                    complex_einsum_inputs(size).unwrap()
                };
                let mut backend = CpuBackend::new();
                let output = concrete_fresh("einsum", &mut backend, &a, &b).unwrap();
                check_tensor_reference(&output, &expected).unwrap();
                let plan =
                    tenferro_einsum::ConcreteEinsumPlan::prepare([&a, &b], "ij,jk->ik").unwrap();
                backend
                    .with_backend_session(|session| {
                        check_tensor_reference(
                            &concrete_operation("einsum", session, &a, &b)?,
                            &expected,
                        )?;
                        check_tensor_reference(&plan.execute([&a, &b], session)?, &expected)?;
                        Ok::<_, Box<dyn Error + Send + Sync>>(())
                    })
                    .unwrap();
                for tier in ["concrete-fresh", "concrete-shared", "prepared-setup"] {
                    let descriptor =
                        case_descriptor("einsum", dtype, tier, "single", size, 1, "faer").unwrap();
                    assert_eq!(descriptor["shape"], serde_json::json!([n, n]));
                }
            }
        }
    }

    #[test]
    fn concrete_solve_matches_known_solution_and_residual_without_mutation() {
        for size in [4, 16, 256] {
            let n = matrix_dimension(size).unwrap();
            let (a, b, x) = solve_inputs(size).unwrap();
            let expected = Tensor::from_vec_col_major(vec![n, n], x).unwrap();
            let before_a = a.as_slice::<f64>().unwrap().to_vec();
            let before_b = b.as_slice::<f64>().unwrap().to_vec();
            assert_ne!(before_a[1], before_a[n]);
            for i in 0..n {
                let off_diagonal: f64 = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| before_a[i + n * j].abs())
                    .sum();
                assert!(before_a[i + n * i] > off_diagonal);
                assert!((0..n)
                    .filter(|&j| j != i)
                    .all(|j| before_a[i + n * j] != 0.0));
            }
            let mut backend = CpuBackend::new();
            let result = concrete_fresh("solve", &mut backend, &a, &b).unwrap();
            check_solve_result(&result, &a, &b, &expected).unwrap();
            assert_eq!(a.as_slice::<f64>().unwrap(), &before_a);
            assert_eq!(b.as_slice::<f64>().unwrap(), &before_b);
            backend
                .with_backend_session(|session| {
                    let result = concrete_operation("solve", session, &a, &b)?;
                    check_solve_result(&result, &a, &b, &expected)?;
                    Ok::<_, Box<dyn Error + Send + Sync>>(())
                })
                .unwrap();
            assert_eq!(a.as_slice::<f64>().unwrap(), &before_a);
            assert_eq!(b.as_slice::<f64>().unwrap(), &before_b);
            let mut wrong_b = before_b.clone();
            wrong_b[0] += 1.0;
            let wrong_b = Tensor::from_vec_col_major(vec![n, n], wrong_b).unwrap();
            assert!(check_solve_result(&expected, &a, &wrong_b, &expected).is_err());
            for tier in ["concrete-fresh", "concrete-shared"] {
                let descriptor =
                    case_descriptor("solve", "f64", tier, "single", size, 1, "faer").unwrap();
                assert_eq!(descriptor["contract_id"], "linalg.solve.ordinary.concrete");
                assert_eq!(descriptor["family"], "linalg");
                assert_eq!(descriptor["shape"], serde_json::json!([n, n]));
            }
        }
        assert!(case_descriptor("solve", "c64", "concrete-fresh", "single", 4, 1, "faer").is_err());
        assert!(case_descriptor("solve", "f64", "eager-no-ad", "single", 4, 1, "faer").is_err());
        assert!(case_descriptor(
            "solve",
            "f64",
            "concrete-fresh",
            "dependent10",
            4,
            10,
            "faer"
        )
        .is_err());
    }

    #[test]
    fn complex_einsum_ad_matches_hermitian_and_directional_oracles() {
        for size in [4, 16, 256] {
            let (a, b, expected) = complex_einsum_inputs(size).unwrap();
            let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
            let (left, right, _) = complex_einsum_inputs(size).unwrap();
            let lhs = EagerTensor::requires_grad_in(left, ctx.clone()).unwrap();
            let rhs = EagerTensor::requires_grad_in(right, ctx).unwrap();
            let output = eager_operation("einsum", &lhs, &rhs).unwrap();
            check_complex_einsum_ad(&output, &lhs, &rhs, &a, &b, &expected).unwrap();
            assert!(lhs.grad().unwrap().is_none());
            assert!(rhs.grad().unwrap().is_none());
            if size == 4 {
                output
                    .reduce_sum(Some(&[0, 1]))
                    .unwrap()
                    .backward()
                    .unwrap();
                let av = a.as_slice::<Complex64>().unwrap().to_vec();
                let bv = b.as_slice::<Complex64>().unwrap().to_vec();
                let loss = |a: &[Complex64], b: &[Complex64]| {
                    matmul_reference(2, a, b).iter().map(|z| z.re).sum::<f64>()
                };
                let epsilon = 1e-6;
                for (operand, input) in [(0, &lhs), (1, &rhs)] {
                    let gradient = input.grad().unwrap().unwrap();
                    for (index, g) in gradient.as_slice::<Complex64>().unwrap().iter().enumerate() {
                        for (direction, component) in [
                            (Complex64::new(1.0, 0.0), g.re),
                            (Complex64::new(0.0, 1.0), g.im),
                        ] {
                            let evaluate = |sign: f64| {
                                let (mut a, mut b) = (av.clone(), bv.clone());
                                if operand == 0 {
                                    a[index] += direction * (sign * epsilon);
                                } else {
                                    b[index] += direction * (sign * epsilon);
                                }
                                loss(&a, &b)
                            };
                            let derivative = (evaluate(1.0) - evaluate(-1.0)) / (2.0 * epsilon);
                            assert!((derivative - component).abs() < 1e-8);
                        }
                    }
                }
            }
            let descriptor =
                case_descriptor("einsum", "c64", "eager-ad", "single", size, 1, "faer").unwrap();
            assert_eq!(descriptor["contract_id"], "einsum.einsum.ordinary.eager");
        }
    }

    #[test]
    fn complex_eager_and_compiled_einsum_keep_complex_inputs() {
        for size in [4, 16, 256] {
            let (lhs, rhs, expected) = complex_einsum_inputs(size).unwrap();
            let backend = CpuBackend::new();
            let (runtime, program) =
                compiled_einsum(&backend, matrix_dimension(size).unwrap(), lhs.dtype()).unwrap();
            check_compiled_einsum(&runtime, &program, &lhs, &rhs).unwrap();
            let real_input =
                Tensor::from_vec_col_major(lhs.shape().to_vec(), vec![1.0_f64; size]).unwrap();
            assert!(runtime
                .run_compiled(&program, &[&lhs, &real_input])
                .is_err());
            let context = EagerRuntime::with_cpu_backend(backend).unwrap();
            let a = EagerTensor::from_tensor_in(lhs, context.clone()).unwrap();
            let b = EagerTensor::from_tensor_in(rhs, context).unwrap();
            let output = eager_operation("einsum", &a, &b)
                .unwrap()
                .to_tensor()
                .unwrap();
            check_tensor_reference(&output, &expected).unwrap();
            for (tier, surface, contract) in [
                ("eager-no-ad", "eager", "einsum.einsum.ordinary.eager"),
                ("compiled-repeat", "traced", "einsum.einsum.prepared.traced"),
            ] {
                let descriptor =
                    case_descriptor("einsum", "c64", tier, "single", size, 1, "faer").unwrap();
                assert_eq!(descriptor["surface"], surface);
                assert_eq!(descriptor["contract_id"], contract);
                assert_eq!(descriptor["dtype"], "c64");
            }
        }
    }

    #[test]
    fn compiled_einsum_rebinds_both_inputs_without_stale_results() {
        for size in [4, 16, 256] {
            let n = matrix_dimension(size).unwrap();
            let (lhs, rhs, expected) = einsum_inputs(size).unwrap();
            let swapped = matmul_reference(
                n,
                rhs.as_slice::<f64>().unwrap(),
                lhs.as_slice::<f64>().unwrap(),
            );
            assert_ne!(expected, swapped);
            let backend = CpuBackend::new();
            let (runtime, program) = compiled_einsum(&backend, n, lhs.dtype()).unwrap();
            check_compiled_einsum(&runtime, &program, &lhs, &rhs).unwrap();
            let descriptor = case_descriptor(
                "einsum",
                "f64",
                "compiled-repeat",
                "single",
                size,
                1,
                "faer",
            )
            .unwrap();
            assert_eq!(descriptor["contract_id"], "einsum.einsum.prepared.traced");
            assert_eq!(descriptor["surface"], "traced");
        }
    }

    #[test]
    fn complex_borrowed_layouts_preserve_components_and_physical_storage() {
        for size in [4, 16, 256] {
            let n = matrix_dimension(size).unwrap();
            for (layout, strides, offset) in [
                ("col_major_contiguous", [1, n as isize], 0),
                ("row_major_contiguous", [n as isize, 1], 0),
                ("strided", [2, (2 * n + 1) as isize], 1),
                ("broadcast", [1, 0], 0),
            ] {
                let (lhs, rhs, expected) = if layout == "broadcast" {
                    complex_broadcast_einsum_inputs(size).unwrap()
                } else {
                    complex_einsum_inputs(size).unwrap()
                };
                let values = expected.as_slice::<Complex64>().unwrap();
                assert!(values.iter().any(|x| x.re.abs() > 0.1));
                assert!(values.iter().any(|x| x.im.abs() > 0.1));
                let fixture = BorrowedEinsumInputs::new(&lhs, &rhs, layout).unwrap();
                let bits = |tensor: &Tensor| {
                    tensor
                        .as_slice::<Complex64>()
                        .unwrap()
                        .iter()
                        .map(|x| (x.re.to_bits(), x.im.to_bits()))
                        .collect::<Vec<_>>()
                };
                let before = [bits(&fixture.storage[0]), bits(&fixture.storage[1])];
                let views = fixture.views().unwrap();
                let BorrowedEinsumViews::C64(typed_views) = &views else {
                    panic!("expected C64 views")
                };
                for view in typed_views {
                    assert_eq!(view.shape(), &[n, n]);
                    assert_eq!(view.strides(), strides);
                    assert_eq!(view.offset(), offset);
                }
                for storage in &fixture.storage {
                    let data = storage.as_slice::<Complex64>().unwrap();
                    if layout == "broadcast" {
                        assert_eq!(data.len(), n);
                    }
                    if layout == "strided" {
                        assert!(data[0].re.is_nan() && data[0].im.is_nan());
                    }
                }
                let mut backend = CpuBackend::new();
                let fresh = backend
                    .with_backend_session(|session| borrowed_einsum(&views, session))
                    .unwrap();
                check_tensor_reference(&fresh, &expected).unwrap();
                backend
                    .with_backend_session(|session| {
                        for _ in 0..2 {
                            check_tensor_reference(&borrowed_einsum(&views, session)?, &expected)?;
                        }
                        Ok::<_, Box<dyn Error + Send + Sync>>(())
                    })
                    .unwrap();
                assert_eq!(
                    [bits(&fixture.storage[0]), bits(&fixture.storage[1])],
                    before
                );
                for tier in ["borrowed-fresh", "borrowed-shared"] {
                    assert!(
                        case_descriptor("einsum", "c64", tier, "single", size, 1, "faer").is_ok()
                    );
                }
            }
        }
    }

    #[test]
    fn borrowed_broadcast_uses_stride_zero_and_compact_physical_storage() {
        for size in [4, 16, 256] {
            let n = matrix_dimension(size).unwrap();
            let (lhs, rhs, expected) = broadcast_einsum_inputs(size).unwrap();
            assert!(expected.iter().any(|x| x.abs() > 0.1));
            let fixture = BorrowedEinsumInputs::new(&lhs, &rhs, "broadcast").unwrap();
            assert_eq!(fixture.storage[0].as_slice::<f64>().unwrap().len(), n);
            assert_eq!(fixture.storage[1].as_slice::<f64>().unwrap().len(), n);
            let views = fixture.views().unwrap();
            let BorrowedEinsumViews::F64(typed_views) = &views else {
                panic!("expected F64 views")
            };
            for view in typed_views {
                assert_eq!(view.shape(), &[n, n]);
                assert_eq!(view.strides(), &[1, 0]);
            }
            let mut backend = CpuBackend::new();
            backend
                .with_backend_session(|session| {
                    check_tensor_shape(&borrowed_einsum(&views, session)?, &expected, &[n, n])?;
                    Ok::<_, Box<dyn Error + Send + Sync>>(())
                })
                .unwrap();
            backend
                .with_backend_session(|session| {
                    check_tensor_shape(&borrowed_einsum(&views, session)?, &expected, &[n, n])?;
                    check_tensor_shape(&borrowed_einsum(&views, session)?, &expected, &[n, n])?;
                    Ok::<_, Box<dyn Error + Send + Sync>>(())
                })
                .unwrap();
        }
    }

    #[test]
    fn borrowed_einsum_layouts_match_reference_in_fresh_and_shared_sessions() {
        for size in [4, 16, 256] {
            let (lhs, rhs, expected) = einsum_inputs(size).unwrap();
            let n = matrix_dimension(size).unwrap();
            for (layout, strides, offset) in [
                ("col_major_contiguous", [1, n as isize], 0),
                ("row_major_contiguous", [n as isize, 1], 0),
                ("strided", [2, (2 * n + 1) as isize], 1),
            ] {
                let fixture = BorrowedEinsumInputs::new(&lhs, &rhs, layout).unwrap();
                let views = fixture.views().unwrap();
                let BorrowedEinsumViews::F64(typed_views) = &views else {
                    panic!("expected F64 views")
                };
                for view in typed_views {
                    assert_eq!(view.strides(), strides);
                    assert_eq!(view.offset(), offset);
                }
                if layout == "strided" {
                    assert!(fixture.storage[0].as_slice::<f64>().unwrap()[0].is_nan());
                }
                let mut backend = CpuBackend::new();
                let fresh = backend
                    .with_backend_session(|session| borrowed_einsum(&views, session))
                    .unwrap();
                check_tensor_shape(&fresh, &expected, &[n, n]).unwrap();
                backend
                    .with_backend_session(|session| {
                        for _ in 0..2 {
                            let output = borrowed_einsum(&views, session)?;
                            check_tensor_shape(&output, &expected, &[n, n])?;
                        }
                        Ok::<_, Box<dyn Error + Send + Sync>>(())
                    })
                    .unwrap();
            }
            assert!(BorrowedEinsumInputs::new(&lhs, &rhs, "unknown").is_err());
        }
        assert!(case_descriptor("add", "f64", "borrowed-fresh", "single", 4, 1, "faer").is_err());
    }

    #[test]
    fn einsum_selector_rejects_unsupported_work_and_keeps_matrix_shape() {
        let descriptor =
            case_descriptor("einsum", "f64", "concrete-fresh", "single", 4, 1, "faer").unwrap();
        assert_eq!(descriptor["shape"], serde_json::json!([2, 2]));
        assert_eq!(descriptor["contract_id"], "einsum.einsum.ordinary.concrete");
        assert!(case_descriptor(
            "einsum",
            "f64",
            "concrete-fresh",
            "dependent10",
            4,
            10,
            "faer"
        )
        .is_err());
        assert!(einsum_inputs(3).is_err());
    }

    #[test]
    fn calibration_uses_each_batch_duration() {
        let durations = [60, 70, 80, 120];
        let mut seen = Vec::new();
        let result = calibrate(100, |iterations| {
            seen.push(iterations);
            Ok(durations[seen.len() - 1])
        })
        .unwrap();
        assert_eq!(seen, [1, 2, 4, 8]);
        assert_eq!(result, (8, 120));
    }

    #[test]
    fn dependent_chain_invokes_the_requested_number_of_adds() {
        for (calls, expected) in [(1, 5), (10, 32)] {
            let mut invocations = 0;
            let output = add_chain(&2_i32, &3_i32, calls, |lhs, rhs| {
                invocations += 1;
                Ok::<_, Box<dyn Error + Send + Sync>>(lhs + rhs)
            })
            .unwrap();
            assert_eq!(invocations, calls);
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn check_tensor_rejects_equal_element_count_with_wrong_shape() {
        let output = Tensor::from_vec_col_major(vec![2, 2], vec![1.0; 4]).unwrap();
        assert!(check_tensor(&output, &[1.0; 4]).is_err());
    }

    #[test]
    fn check_tensor_rejects_nonfinite_values() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let output = Tensor::from_vec_col_major(vec![1], vec![value]).unwrap();
            assert!(check_tensor(&output, &[value]).is_err());
        }
    }

    #[test]
    fn check_tensor_accepts_valid_values() {
        let output = Tensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
        assert_eq!(check_tensor(&output, &[1.0, 2.0]).unwrap(), 3.0);
    }

    #[test]
    fn check_active_ad_rejects_nonfinite_expected_gradient() {
        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
        let lhs = EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major(vec![1], vec![1.0]).unwrap(),
            runtime.clone(),
        )
        .unwrap();
        let rhs = EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major(vec![1], vec![2.0]).unwrap(),
            runtime,
        )
        .unwrap();
        let output = eager_operation("add", &lhs, &rhs).unwrap();
        assert!(check_active_ad(&output, &lhs, &rhs, &[3.0], f64::NAN).is_err());
    }
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let case_id = required_arg("--case")?;
    let operation = required_arg("--operation")?;
    let dtype = required_arg("--dtype")?;
    let api_tier = required_arg("--api-tier")?;
    let workflow = required_arg("--workflow")?;
    let mode = arg("--mode", "measure");
    if mode != "measure" && mode != "correctness-only" {
        return Err(
            format!("unsupported mode: {mode}; expected measure or correctness-only").into(),
        );
    }
    let size = parse_usize("--size", 4)?;
    let warmups = parse_usize("--warmups", 1)?;
    let samples_count = parse_usize("--samples", 3)?;
    let target_ns = parse_usize("--target-ns", 1_000_000)? as u128;
    let process_index = parse_usize("--process-index", 0)?;
    let sample_start = parse_usize("--sample-start", 0)?;
    let calls = parse_usize("--calls", 1)?;
    if size == 0 || samples_count == 0 || calls == 0 || target_ns == 0 {
        return Err("size, samples, calls, and target-ns must be positive".into());
    }
    let expected_calls = match workflow.as_str() {
        "single" => 1,
        "dependent10" => 10,
        _ => return Err(format!("unsupported small-work workflow: {workflow}").into()),
    };
    if calls != expected_calls {
        return Err(
            format!("workflow {workflow} requires calls_per_workflow={expected_calls}").into(),
        );
    }

    let layout = arg("--layout", "col_major_contiguous");
    if !api_tier.starts_with("borrowed-") && layout != "col_major_contiguous" {
        return Err("owned-input tiers require col_major_contiguous layout".into());
    }

    // Construct exactly one CPU backend per process, and capture its actual provider
    // before moving it into an eager runtime where applicable.
    let backend = match std::env::var("TENFERRO_CPU_BACKEND_KIND").as_deref() {
        Ok("blas") => CpuBackend::with_kind(CpuBackendKind::Blas)?,
        Ok("faer") => CpuBackend::with_kind(CpuBackendKind::Faer)?,
        Ok("") | Ok("default") | Err(_) => CpuBackend::new(),
        Ok(other) => return Err(format!("unsupported CPU backend: {other}").into()),
    };
    let provider = format!("{:?}", backend.kind()).to_ascii_lowercase();
    let mut descriptor = case_descriptor(
        &operation, &dtype, &api_tier, &workflow, size, calls, &provider,
    )?;
    descriptor["layout"] = serde_json::json!(layout);
    let mut concrete = None;
    let mut runtime = None;
    match api_tier.as_str() {
        "concrete-fresh" | "concrete-shared" | "borrowed-fresh" | "borrowed-shared"
        | "prepared-setup" | "prepared-repeat" | "compiled-repeat" | "compiled-setup" => {
            concrete = Some(backend)
        }
        "eager-no-ad" | "eager-ad" => runtime = Some(EagerRuntime::with_cpu_backend(backend)?),
        _ => return Err(format!("unsupported small-work API tier: {api_tier}").into()),
    }
    let (lhs, rhs_storage, expected_values) = if operation == "reduce_sum" {
        let (input, expected) = reduce_sum_input(size)?;
        (input, None, expected)
    } else if dtype == "c64" {
        let (lhs, rhs, expected) = if layout == "broadcast" {
            complex_broadcast_einsum_inputs(size)?
        } else {
            complex_einsum_inputs(size)?
        };
        (lhs, Some(rhs), expected)
    } else {
        let (lhs, rhs, values) = if operation == "einsum" && layout == "broadcast" {
            broadcast_einsum_inputs(size)?
        } else if operation == "einsum" {
            einsum_inputs(size)?
        } else if operation == "solve" {
            solve_inputs(size)?
        } else if operation == "gather" {
            gather_inputs(size)?
        } else {
            inputs(size, calls)?
        };
        let expected = Tensor::from_vec_col_major(lhs.shape().to_vec(), values)?;
        (lhs, Some(rhs), expected)
    };
    // Unary dispatch ignores rhs: no second tensor allocation or ownership clone.
    let rhs = rhs_storage.as_ref().unwrap_or(&lhs);
    let compiled = if api_tier.starts_with("compiled-") {
        Some(compiled_einsum(
            concrete.as_ref().ok_or("concrete backend missing")?,
            matrix_dimension(size)?,
            lhs.dtype(),
        )?)
    } else {
        None
    };
    let compiled_bindings = [&lhs, rhs];
    let borrowed_storage = if api_tier.starts_with("borrowed-") {
        Some(BorrowedEinsumInputs::new(&lhs, rhs, &layout)?)
    } else {
        None
    };
    let borrowed = borrowed_storage
        .as_ref()
        .map(BorrowedEinsumInputs::views)
        .transpose()?;
    let prepared = if api_tier.starts_with("prepared-") {
        Some(tenferro_einsum::ConcreteEinsumPlan::prepare(
            [&lhs, rhs],
            "ij,jk->ik",
        )?)
    } else {
        None
    };
    let eager_pair = if let Some(ctx) = runtime.as_ref() {
        let (left, right) = if dtype == "c64" {
            let (left, right, _) = complex_einsum_inputs(size)?;
            (left, right)
        } else {
            let (left, right, _) = if operation == "einsum" {
                einsum_inputs(size)?
            } else {
                inputs(size, calls)?
            };
            (left, right)
        };
        let tracked = api_tier == "eager-ad";
        let left = if tracked {
            EagerTensor::requires_grad_in(left, ctx.clone())?
        } else {
            EagerTensor::from_tensor_in(left, ctx.clone())?
        };
        let right = if tracked {
            EagerTensor::requires_grad_in(right, ctx.clone())?
        } else {
            EagerTensor::from_tensor_in(right, ctx.clone())?
        };
        Some((left, right))
    } else {
        None
    };

    let check_concrete = |output: &Tensor| {
        if operation == "solve" {
            check_solve_result(output, &lhs, rhs, &expected_values)
        } else {
            check_tensor_reference(output, &expected_values)
        }
    };
    let after_correctness = match api_tier.as_str() {
        "concrete-fresh" => {
            let backend = concrete.as_mut().ok_or("concrete backend missing")?;
            check_concrete(&add_chain(&lhs, rhs, calls, |a, b| {
                concrete_fresh(&operation, backend, a, b)
            })?)?
        }
        "compiled-repeat" | "compiled-setup" => {
            let (runtime, program) = compiled.as_ref().ok_or("compiled program missing")?;
            check_compiled_einsum(runtime, program, &lhs, rhs)?
        }
        "borrowed-fresh" | "borrowed-shared" => {
            let backend = concrete.as_mut().ok_or("concrete backend missing")?;
            backend.with_backend_session(|session| {
                check_tensor_reference(
                    &borrowed_einsum(borrowed.as_ref().ok_or("borrowed inputs missing")?, session)?,
                    &expected_values,
                )
            })?
        }
        "prepared-setup" | "prepared-repeat" => {
            let backend = concrete.as_mut().ok_or("concrete backend missing")?;
            backend.with_backend_session(|session| {
                check_tensor_reference(
                    &prepared
                        .as_ref()
                        .ok_or("prepared plan missing")?
                        .execute([&lhs, rhs], session)?,
                    &expected_values,
                )
            })?
        }
        "concrete-shared" => {
            let backend = concrete.as_mut().ok_or("concrete backend missing")?;
            backend.with_backend_session(|session| {
                check_concrete(&add_chain(&lhs, rhs, calls, |a, b| {
                    concrete_operation(&operation, session, a, b)
                })?)
            })?
        }
        "eager-no-ad" => {
            let (left, right) = eager_pair.as_ref().ok_or("eager inputs missing")?;
            check_tensor_reference(
                &add_chain(left, right, calls, |a, b| eager_operation(&operation, a, b))?
                    .to_tensor()?,
                &expected_values,
            )?
        }
        "eager-ad" => {
            let (left, right) = eager_pair.as_ref().ok_or("eager inputs missing")?;
            let value = add_chain(left, right, calls, |a, b| eager_operation(&operation, a, b))?;
            if operation == "einsum" && dtype == "c64" {
                check_complex_einsum_ad(&value, left, right, &lhs, rhs, &expected_values)?
            } else if operation == "einsum" {
                check_einsum_ad(
                    &value,
                    left,
                    right,
                    &lhs,
                    rhs,
                    expected_values.as_slice::<f64>()?,
                )?
            } else {
                check_active_ad(
                    &value,
                    left,
                    right,
                    expected_values.as_slice::<f64>()?,
                    calls as f64,
                )?
            }
        }
        _ => return Err(format!("unsupported small-work API tier: {api_tier}").into()),
    };
    let correctness_status = if after_correctness.is_finite() {
        "passed"
    } else {
        "failed"
    };
    let mut output = serde_json::json!({
        "suite_id": "cpu/small_work", "case_id": case_id,
        "provider": provider, "calls_per_workflow": calls,
        "correctness_status": correctness_status, "samples": Vec::<Sample>::new(),
    });
    if let serde_json::Value::Object(fields) = descriptor {
        for (key, value) in fields {
            output[key] = value;
        }
    }
    if mode == "correctness-only" {
        println!("{}", serde_json::to_string(&output)?);
        return Ok(());
    }

    let setup_diagnostic = api_tier.ends_with("-setup")
        || api_tier.ends_with("-fresh")
        || (operation == "einsum"
            && matches!(api_tier.as_str(), "concrete-shared" | "borrowed-shared"));
    if setup_diagnostic && std::env::var("BENCH_INCLUDE_SETUP_DIAGNOSTICS").as_deref() != Ok("1") {
        return Err("setup-inclusive route requires BENCH_INCLUDE_SETUP_DIAGNOSTICS=1; use shared/prepared operation routes".into());
    }
    output["measurement_kind"] = serde_json::json!(if setup_diagnostic {
        "setup_diagnostic"
    } else {
        "operation"
    });

    let measurement = match api_tier.as_str() {
        "compiled-setup" => {
            let n = matrix_dimension(size)?;
            let dtype = lhs.dtype();
            measure(
                warmups,
                samples_count,
                target_ns,
                process_index,
                sample_start,
                || trace_compile_einsum(n, dtype),
            )?
        }
        "compiled-repeat" => {
            let (runtime, program) = compiled.as_ref().ok_or("compiled program missing")?;
            measure(
                warmups,
                samples_count,
                target_ns,
                process_index,
                sample_start,
                || Ok(runtime.run_compiled(program, &compiled_bindings)?),
            )?
        }
        "prepared-setup" => measure(
            warmups,
            samples_count,
            target_ns,
            process_index,
            sample_start,
            || {
                Ok(tenferro_einsum::ConcreteEinsumPlan::prepare(
                    [&lhs, rhs],
                    "ij,jk->ik",
                )?)
            },
        )?,
        "concrete-shared" | "borrowed-shared" | "prepared-repeat" => {
            let backend = concrete.as_mut().ok_or("concrete backend missing")?;
            backend.with_backend_session(|session| {
                measure(
                    warmups,
                    samples_count,
                    target_ns,
                    process_index,
                    sample_start,
                    || {
                        if let Some(views) = borrowed.as_ref() {
                            borrowed_einsum(views, session)
                        } else if let Some(plan) = prepared.as_ref() {
                            Ok(plan.execute([&lhs, rhs], session)?)
                        } else {
                            execute_shared(&operation, session, &lhs, rhs, calls)
                        }
                    },
                )
            })?
        }
        "concrete-fresh" | "borrowed-fresh" => {
            let backend = concrete.as_mut().ok_or("concrete backend missing")?;
            measure(
                warmups,
                samples_count,
                target_ns,
                process_index,
                sample_start,
                || {
                    if let Some(views) = borrowed.as_ref() {
                        backend.with_backend_session(|session| borrowed_einsum(views, session))
                    } else {
                        execute_fresh(&operation, backend, &lhs, rhs, calls)
                    }
                },
            )?
        }
        "eager-no-ad" | "eager-ad" => {
            let (left, right) = eager_pair.as_ref().ok_or("eager inputs missing")?;
            measure(
                warmups,
                samples_count,
                target_ns,
                process_index,
                sample_start,
                || execute_eager(&operation, left, right, calls),
            )?
        }
        _ => return Err(format!("unsupported small-work API tier: {api_tier}").into()),
    };
    output["samples"] = serde_json::to_value(&measurement.samples)?;
    output["calibration"] = serde_json::json!({"target_ns": target_ns, "iterations": measurement.iterations, "elapsed_ns": measurement.calibrated_ns});
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
