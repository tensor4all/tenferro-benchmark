//! Executable public-boundary case producer for cpu/small_work.
//!
//! The binary owns setup, calibrated aggregate timing, value-dependent output
//! consumption, and correctness. It emits exactly one JSON object on stdout.

use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use serde::Serialize;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{BackendSession, BackendSessionHost, TensorSessionOpsExt};
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

#[derive(Serialize)]
struct TaskAffinity {
    tid: usize,
    cpus: Vec<usize>,
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

fn affinity() -> Result<Vec<usize>, Box<dyn Error + Send + Sync>> {
    let text = std::fs::read_to_string("/proc/self/status")?;
    text.lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:\t"))
        .map(parse_cpu_list)
        .ok_or_else(|| "Cpus_allowed_list is missing from /proc/self/status".into())
}

fn task_vanished(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::NotFound || error.raw_os_error() == Some(3)
}

fn parse_cpu_list(value: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in value.trim().split(',') {
        let mut bounds = part.split('-');
        let Some(start) = bounds.next().and_then(|v| v.parse::<usize>().ok()) else {
            continue;
        };
        let end = bounds.next().and_then(|v| v.parse().ok()).unwrap_or(start);
        cpus.extend(start..=end);
    }
    cpus
}

fn thread_affinities(
    expected: Option<&[usize]>,
) -> Result<Vec<TaskAffinity>, Box<dyn Error + Send + Sync>> {
    let mut observed = Vec::new();
    for entry in std::fs::read_dir("/proc/self/task")? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(tid) = name.to_str().and_then(|value| value.parse::<usize>().ok()) else {
            continue;
        };
        let status = match std::fs::read_to_string(entry.path().join("status")) {
            Ok(status) => status,
            Err(error) if task_vanished(&error) => {
                // /proc/task is a live directory: a worker may exit between
                // enumeration and status read.  This is not failed telemetry;
                // retain the surviving-task observations and leave diagnostics
                // on stderr rather than contaminating the JSON receipt.
                eprintln!("affinity observation skipped vanished task {tid}");
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let cpus = status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:\t"))
            .map(parse_cpu_list)
            .ok_or_else(|| format!("Cpus_allowed_list missing for live task {tid}"))?;
        if let Some(expected) = expected {
            if cpus != expected {
                return Err(format!(
                    "live task affinity differs from selected CPUs (task {tid}, expected {:?}, observed {:?})",
                    expected, cpus
                ).into());
            }
        }
        observed.push(TaskAffinity { tid, cpus });
    }
    if observed.is_empty() {
        return Err("no live task affinity observations were available".into());
    }
    observed.sort_by_key(|task| task.tid);
    Ok(observed)
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
        256 => Ok(16),
        _ => Err("einsum requires square matrices of dimension 2, 4 or 16".into()),
    }
}

fn einsum_inputs(size: usize) -> Result<(Tensor, Tensor, Vec<f64>), Box<dyn Error + Send + Sync>> {
    let n = matrix_dimension(size)?;
    let a: Vec<f64> = (0..size).map(|i| (i % 7) as f64 * 0.25 - 0.5).collect();
    let b: Vec<f64> = (0..size).map(|i| (i % 5) as f64 * 0.5 - 0.75).collect();
    let mut expected = vec![0.0; size];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                expected[i + n * k] += a[i + n * j] * b[j + n * k];
            }
        }
    }
    Ok((
        Tensor::from_vec_col_major(vec![n, n], a)?,
        Tensor::from_vec_col_major(vec![n, n], b)?,
        expected,
    ))
}

struct BorrowedEinsumInputs {
    storage: [Vec<f64>; 2],
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
            "strided" => ([2, (2 * n + 1) as isize], 1),
            _ => return Err(format!("unsupported borrowed layout: {layout}").into()),
        };
        let length = offset + (n - 1) * (strides[0] + strides[1]) as usize + 1;
        let mut storage = [vec![f64::NAN; length], vec![f64::NAN; length]];
        for (destination, input) in storage.iter_mut().zip([lhs, rhs]) {
            let values = input.as_slice::<f64>()?;
            for j in 0..n {
                for i in 0..n {
                    destination[offset + i * strides[0] as usize + j * strides[1] as usize] =
                        values[i + n * j];
                }
            }
        }
        Ok(Self {
            storage,
            n,
            strides,
            offset,
        })
    }

    fn views(&self) -> Result<[TypedTensorView<'_, f64>; 2], Box<dyn Error + Send + Sync>> {
        Ok([
            TypedTensorView::from_slice(
                vec![self.n, self.n],
                self.strides,
                self.offset as isize,
                &self.storage[0],
            )?,
            TypedTensorView::from_slice(
                vec![self.n, self.n],
                self.strides,
                self.offset as isize,
                &self.storage[1],
            )?,
        ])
    }
}

fn borrowed_einsum(
    inputs: &[TypedTensorView<'_, f64>; 2],
    session: &mut dyn BackendSession,
) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
    use tenferro_einsum::TypedTensorReadEinsumExt;
    Ok(inputs.einsum_read("ij,jk->ik", session)?.into())
}

fn concrete_operation(
    operation: &str,
    session: &mut dyn BackendSession,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Tensor, Box<dyn Error + Send + Sync>> {
    use tenferro_einsum::TensorEinsumExt;
    Ok(match operation {
        "add" => lhs.add(rhs, session)?,
        "einsum" => [lhs, rhs].einsum("ij,jk->ik", session)?,
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
    use tenferro_einsum::EagerEinsumExt;
    Ok(match operation {
        "add" => lhs.add(rhs)?,
        "einsum" => [lhs, rhs].einsum("ij,jk->ik")?,
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
    if !matches!(operation, "add" | "einsum") || dtype != "f64" || !matches!(size, 4 | 16 | 256) {
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
    if operation == "einsum" && workflow != "single" {
        return Err("einsum currently requires the single workflow".into());
    }
    if api_tier.starts_with("borrowed-") && operation != "einsum" {
        return Err("borrowed tiers require einsum".into());
    }
    let (contract_id, family, surface, timer, outside_timer) = match api_tier {
        "concrete-fresh" | "borrowed-fresh" => (
            "core.add.ordinary.concrete",
            "core",
            "concrete",
            vec!["session_entry_exit", "add", "output_lifetime"],
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
            vec!["add", "output_lifetime"],
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
            vec!["add", "output_materialization"],
            vec![
                "backend_construction",
                "input_construction",
                "session_entry_exit",
                "correctness_check",
            ],
        ),
        "eager-ad" => (
            "core.add.ordinary.eager",
            "core",
            "eager",
            vec!["add_forward_recording", "output_materialization"],
            vec![
                "backend_construction",
                "input_construction",
                "session_entry_exit",
                "backward",
                "correctness_check",
            ],
        ),
        "prepared-setup" if operation == "einsum" => (
            "einsum.einsum.prepare.concrete",
            "einsum",
            "concrete",
            vec!["prepare", "plan_lifetime"],
            vec![
                "backend_construction",
                "input_construction",
                "correctness_check",
            ],
        ),
        "prepared-repeat" if operation == "einsum" => (
            "einsum.einsum.prepared.concrete",
            "einsum",
            "concrete",
            vec!["prepared_execute", "output_lifetime"],
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
            if api_tier.starts_with("prepared-") {
                contract_id.to_string()
            } else {
                format!("einsum.einsum.ordinary.{surface}")
            },
            "einsum",
            vec![matrix_dimension(size)?, matrix_dimension(size)?],
        )
    } else {
        (contract_id.to_string(), family, vec![size])
    };
    let timer: Vec<String> = timer
        .iter()
        .map(|label| label.replace("add", operation))
        .collect();
    Ok(serde_json::json!({
        "contract_id": contract_id, "family": family, "surface": surface,
        "operation": operation, "phase": if api_tier == "prepared-setup" { "setup" } else { "execution" }, "api_tier": api_tier,
        "backend": "tenferro-rs", "provider": if api_tier == "prepared-setup" { "not-applicable" } else { provider }, "dtype": dtype,
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

fn execute_fresh(
    operation: &str,
    backend: &mut CpuBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    calls: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    black_box(add_chain(lhs, rhs, calls, |a, b| {
        concrete_fresh(operation, backend, a, b)
    })?);
    Ok(())
}

fn execute_shared(
    operation: &str,
    session: &mut dyn BackendSession,
    lhs: &Tensor,
    rhs: &Tensor,
    calls: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    black_box(add_chain(lhs, rhs, calls, |a, b| {
        concrete_operation(operation, session, a, b)
    })?);
    Ok(())
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
    under_duration: bool,
    after_warmup: Vec<TaskAffinity>,
    after_timing: Vec<TaskAffinity>,
}

fn measure<F>(
    warmups: usize,
    samples_count: usize,
    target_ns: u128,
    process_index: usize,
    sample_start: usize,
    expected_cpus: Option<&[usize]>,
    mut execute: F,
) -> Result<Measurement, Box<dyn Error + Send + Sync>>
where
    F: FnMut() -> Result<(), Box<dyn Error + Send + Sync>>,
{
    for _ in 0..warmups {
        execute()?;
    }
    let after_warmup = thread_affinities(expected_cpus)?;
    let (iterations, calibrated_ns) = calibrate(target_ns, |iterations| {
        let start = Instant::now();
        for _ in 0..iterations {
            execute()?;
        }
        Ok(start.elapsed().as_nanos())
    })?;
    let mut samples = Vec::with_capacity(samples_count);
    let mut under_duration = false;
    for sample_index in 0..samples_count {
        let start = Instant::now();
        for _ in 0..iterations {
            execute()?;
        }
        let elapsed_ns = start.elapsed().as_nanos();
        if elapsed_ns == 0 {
            return Err(
                "timed sample elapsed zero nanoseconds; raw duration not fabricated".into(),
            );
        }
        under_duration |= elapsed_ns < target_ns;
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
        under_duration,
        after_warmup,
        after_timing: thread_affinities(expected_cpus)?,
    })
}

fn execute_eager(
    operation: &str,
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    calls: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    black_box(add_chain(lhs, rhs, calls, |a, b| {
        eager_operation(operation, a, b)
    })?);
    Ok(())
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
                for view in &views {
                    assert_eq!(view.strides(), strides);
                    assert_eq!(view.offset(), offset);
                }
                if layout == "strided" {
                    assert!(fixture.storage[0][0].is_nan());
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
    fn vanished_task_errors_are_recognized_without_masking_other_errors() {
        assert!(task_vanished(&std::io::Error::from_raw_os_error(3)));
        assert!(task_vanished(&std::io::Error::from(
            std::io::ErrorKind::NotFound
        )));
        assert!(!task_vanished(&std::io::Error::from(
            std::io::ErrorKind::PermissionDenied
        )));
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
    let declared_threads = parse_usize("--declared-threads", 1)?;
    let expected_cpus = {
        let value = arg("--expected-cpus", "");
        if value.is_empty() {
            None
        } else {
            Some(parse_cpu_list(&value))
        }
    };
    if size == 0 || samples_count == 0 || calls == 0 || target_ns == 0 || declared_threads == 0 {
        return Err(
            "size, samples, calls, target-ns, and declared-threads must be positive".into(),
        );
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
    let backend = CpuBackend::new();
    let provider = format!("{:?}", backend.kind()).to_ascii_lowercase();
    let mut descriptor = case_descriptor(
        &operation, &dtype, &api_tier, &workflow, size, calls, &provider,
    )?;
    descriptor["layout"] = serde_json::json!(layout);
    let mut concrete = None;
    let mut runtime = None;
    match api_tier.as_str() {
        "concrete-fresh" | "concrete-shared" | "borrowed-fresh" | "borrowed-shared"
        | "prepared-setup" | "prepared-repeat" => concrete = Some(backend),
        "eager-no-ad" | "eager-ad" => runtime = Some(EagerRuntime::with_cpu_backend(backend)?),
        _ => return Err(format!("unsupported small-work API tier: {api_tier}").into()),
    }
    let (lhs, rhs, expected_values) = if operation == "einsum" {
        einsum_inputs(size)?
    } else {
        inputs(size, calls)?
    };
    let borrowed_storage = if api_tier.starts_with("borrowed-") {
        Some(BorrowedEinsumInputs::new(&lhs, &rhs, &layout)?)
    } else {
        None
    };
    let borrowed = borrowed_storage
        .as_ref()
        .map(BorrowedEinsumInputs::views)
        .transpose()?;
    let prepared = if api_tier.starts_with("prepared-") {
        Some(tenferro_einsum::ConcreteEinsumPlan::prepare(
            [&lhs, &rhs],
            "ij,jk->ik",
        )?)
    } else {
        None
    };
    let eager_pair = if let Some(ctx) = runtime.as_ref() {
        let (left, right, _) = if operation == "einsum" {
            einsum_inputs(size)?
        } else {
            inputs(size, calls)?
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

    let expected_mask = match affinity() {
        Ok(mask) => mask,
        Err(error) => {
            eprintln!("affinity telemetry unavailable: {error}");
            Vec::new()
        }
    };
    let after_correctness = match api_tier.as_str() {
        "concrete-fresh" => {
            let backend = concrete.as_mut().ok_or("concrete backend missing")?;
            check_tensor_shape(
                &add_chain(&lhs, &rhs, calls, |a, b| {
                    concrete_fresh(&operation, backend, a, b)
                })?,
                &expected_values,
                lhs.shape(),
            )?
        }
        "borrowed-fresh" | "borrowed-shared" => {
            let backend = concrete.as_mut().ok_or("concrete backend missing")?;
            backend.with_backend_session(|session| {
                check_tensor_shape(
                    &borrowed_einsum(borrowed.as_ref().ok_or("borrowed inputs missing")?, session)?,
                    &expected_values,
                    lhs.shape(),
                )
            })?
        }
        "prepared-setup" | "prepared-repeat" => {
            let backend = concrete.as_mut().ok_or("concrete backend missing")?;
            backend.with_backend_session(|session| {
                check_tensor_shape(
                    &prepared
                        .as_ref()
                        .ok_or("prepared plan missing")?
                        .execute([&lhs, &rhs], session)?,
                    &expected_values,
                    lhs.shape(),
                )
            })?
        }
        "concrete-shared" => {
            let backend = concrete.as_mut().ok_or("concrete backend missing")?;
            backend.with_backend_session(|session| {
                check_tensor_shape(
                    &add_chain(&lhs, &rhs, calls, |a, b| {
                        concrete_operation(&operation, session, a, b)
                    })?,
                    &expected_values,
                    lhs.shape(),
                )
            })?
        }
        "eager-no-ad" => {
            let (left, right) = eager_pair.as_ref().ok_or("eager inputs missing")?;
            check_tensor_shape(
                &add_chain(left, right, calls, |a, b| eager_operation(&operation, a, b))?
                    .to_tensor()?,
                &expected_values,
                lhs.shape(),
            )?
        }
        "eager-ad" => {
            let (left, right) = eager_pair.as_ref().ok_or("eager inputs missing")?;
            let value = add_chain(left, right, calls, |a, b| eager_operation(&operation, a, b))?;
            if operation == "einsum" {
                check_einsum_ad(&value, left, right, &lhs, &rhs, &expected_values)?
            } else {
                check_active_ad(&value, left, right, &expected_values, calls as f64)?
            }
        }
        _ => return Err(format!("unsupported small-work API tier: {api_tier}").into()),
    };
    let (correctness_observed, telemetry_error) = match thread_affinities(expected_cpus.as_deref())
    {
        Ok(observed) => (observed, None),
        Err(error) => {
            eprintln!("post-correctness affinity telemetry unavailable: {error}");
            (Vec::new(), Some(error.to_string()))
        }
    };
    let correctness_status = if after_correctness.is_finite() {
        "passed"
    } else {
        "failed"
    };
    let mut output = serde_json::json!({
        "schema_version": 2, "suite_id": "cpu/small_work", "case_id": case_id,
        "provider": provider, "calls_per_workflow": calls, "correctness_status": correctness_status,
        "affinity": expected_mask, "declared_thread_budget": declared_threads,
        "thread_affinity_observations": {"after_correctness": correctness_observed, "after_warmup": [], "after_timing": []},
        "observed_thread_count": correctness_observed.len(), "samples": Vec::<Sample>::new(),
        "timing_validity": if mode == "correctness-only" { "inconclusive" } else { "invalid" },
        "timing_reasons": if mode == "correctness-only" { vec!["correctness-only mode"] } else { vec!["timing not run"] },
        "errors": telemetry_error.into_iter().collect::<Vec<_>>(), "process_index": process_index,
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

    let measurement = match api_tier.as_str() {
        "prepared-setup" => measure(
            warmups,
            samples_count,
            target_ns,
            process_index,
            sample_start,
            expected_cpus.as_deref(),
            || {
                black_box(tenferro_einsum::ConcreteEinsumPlan::prepare(
                    [&lhs, &rhs],
                    "ij,jk->ik",
                )?);
                Ok(())
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
                    expected_cpus.as_deref(),
                    || {
                        if let Some(views) = borrowed.as_ref() {
                            black_box(borrowed_einsum(views, session)?);
                            Ok(())
                        } else if let Some(plan) = prepared.as_ref() {
                            black_box(plan.execute([&lhs, &rhs], session)?);
                            Ok(())
                        } else {
                            execute_shared(&operation, session, &lhs, &rhs, calls)
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
                expected_cpus.as_deref(),
                || {
                    if let Some(views) = borrowed.as_ref() {
                        black_box(
                            backend
                                .with_backend_session(|session| borrowed_einsum(views, session))?,
                        );
                        Ok(())
                    } else {
                        execute_fresh(&operation, backend, &lhs, &rhs, calls)
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
                expected_cpus.as_deref(),
                || execute_eager(&operation, left, right, calls),
            )?
        }
        _ => return Err(format!("unsupported small-work API tier: {api_tier}").into()),
    };
    output["thread_affinity_observations"]["after_warmup"] =
        serde_json::to_value(&measurement.after_warmup)?;
    output["thread_affinity_observations"]["after_timing"] =
        serde_json::to_value(&measurement.after_timing)?;
    output["samples"] = serde_json::to_value(&measurement.samples)?;
    output["calibration"] = serde_json::json!({"target_ns": target_ns, "iterations": measurement.iterations, "elapsed_ns": measurement.calibrated_ns});
    output["timing_validity"] = serde_json::json!(if measurement.under_duration {
        "invalid"
    } else {
        "valid"
    });
    output["timing_reasons"] = serde_json::json!(if measurement.under_duration {
        vec!["at least one measured aggregate was shorter than target_ns"]
    } else {
        Vec::<&str>::new()
    });
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
