//! Public CPU umbrella routes (#177). Setup, counters and diagnostic clocks
//! are distinct; no private lane planner or provider object is inspected.
use super::*;
use tenferro_cpu::with_cpu_exec_session;
use tenferro_einsum::ConcreteEinsumPlan;
use tenferro_linalg::LinalgBackend;
use tenferro_runtime::TypedTensorSessionOpsExt;
use tenferro_tensor::{
    ContractionScalar, DynRank, Host, Rank, StridedSliceSpec, TensorScalar, TensorViewMut,
    TensorWrite, TypedTensor, TypedTensorViewMut,
};

fn outcome(
    measurement: Option<Measurement>,
    mut extra: Value,
    timer: Vec<&'static str>,
) -> CaseOutcome {
    // Validators check every value/alias but do not return an aggregate maximum.
    // Do not publish the CaseOutcome sentinel as an exact measured zero.
    extra["max_rel_error"] = Value::Null;
    extra["validation_method"] = json!("complete elementwise or metadata/alias checks before sampling; aggregate maximum not recorded");
    CaseOutcome {
        max_rel_error: 0.0,
        measurement,
        timer,
        outside_timer: vec![
            "backend/thread-pool initialization and explicit priming",
            "input/output preparation and per-sample restoration",
            "entered held session (except explicitly named entry diagnostics)",
            "validation, allocation counter queries, output destruction",
        ],
        extra,
    }
}

fn check(actual: &[f64], expected: &[f64]) -> Result<()> {
    if actual.len() != expected.len() {
        return Err("wrong output length".into());
    }
    for (&a, &b) in actual.iter().zip(expected) {
        check_tol(rel_error(a, b), 1e-10, "umbrella numerical check")?;
    }
    Ok(())
}

fn data(t: &Tensor) -> Result<&[f64]> {
    Ok(t.as_slice::<f64>()?)
}

/// A separate, untimed retained-output allocator pass. The output container
/// itself is preallocated. Does not count foreign C/BLAS allocator activity.
fn counter<O>(name: &str, bytes: usize, mut op: impl FnMut() -> Result<O>) -> Result<Value> {
    black_box(op()?); // Explicit lazy initialization before enabling counters.
    let repeats = 64;
    let mut outputs = Vec::with_capacity(repeats);
    let (_, allocations, allocated) = count_allocations(|| {
        for _ in 0..repeats {
            outputs.push(op()?);
        }
        Ok(())
    })?;
    black_box(&outputs);
    drop(outputs);
    Ok(json!({"op": name, "output_bytes": bytes,
        "host_allocations_per_op": allocations as f64 / repeats as f64,
        "host_allocated_bytes_per_op": allocated as f64 / repeats as f64,
        "allocated_bytes_over_output_bytes": if bytes == 0 { None } else { Some(allocated as f64 / repeats as f64 / bytes as f64) }}))
}

fn counters(rows: Vec<Value>) -> Value {
    json!({"counters": rows,
        "counter_source": "separate retained-output Rust global allocator pass, including this process's Rust worker allocations",
        "unavailable": ["foreign BLAS/C allocator calls", "pooled-buffer reuse volume"]})
}

pub(super) fn run(p: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    let mut row = match param_str(p, "kind")? {
        "umbrella_add" => add(p, timing, threads),
        "umbrella_matmul" | "umbrella_matmul_counter" => matmul(p, timing, threads),
        "umbrella_construct" => construct(p, timing),
        "umbrella_entry" => entry(p, timing, threads),
        "umbrella_neg" => neg(p, timing, threads),
        "umbrella_gemm" => gemm(p, timing, threads),
        "umbrella_prepared_counter" => prepared(p, threads),
        "umbrella_lu" | "umbrella_lane_counter" => lu(p, timing, threads),
        "umbrella_axpby" => axpby(p, timing, threads),
        "umbrella_view_counter" => views(p),
        "umbrella_pool_state" => pool_state(p, timing),
        _ => Err("unregistered umbrella kind".into()),
    }?;
    if matches!(
        param_str(p, "kind")?,
        "umbrella_construct" | "umbrella_view_counter" | "umbrella_pool_state"
    ) {
        row.extra["operation_requires_session"] = json!(false);
    } else if param_str(p, "arm")? == "host" {
        row.extra["host_reference_threads"] = json!(1);
    } else {
        row.extra["effective_backend_threads"] = json!(threads);
    }
    Ok(row)
}

fn checked_backend(threads: usize) -> Result<CpuBackend> {
    let backend = CpuBackend::with_threads(threads)?;
    if backend.num_threads() != threads {
        return Err("CPU backend thread budget differs from requested count".into());
    }
    Ok(backend)
}

fn add(p: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    let n = param_usize(p, "n")?;
    let route = param_str(p, "route")?;
    let a = TypedTensor::<f64>::from_vec_col_major([n], values(n, 110))?;
    let b = TypedTensor::<f64>::from_vec_col_major([n], values(n, 111))?;
    let want: Vec<_> = a
        .host_data()?
        .iter()
        .zip(b.host_data()?)
        .map(|(a, b)| a + b)
        .collect();
    let mut backend = checked_backend(threads)?;
    let measurement = if route == "scoped" {
        let got = backend.with_backend_session(|s| a.add(&b, s))??;
        check(got.host_data()?, &want)?;
        measure(timing, n * 8, || {
            Ok(backend.with_backend_session(|s| a.add(&b, s))??)
        })?
    } else {
        let mut held = backend.open_session()?;
        let m = held.with_session(|s| -> Result<Measurement> {
            if route == "into_held" {
                let mut out = TypedTensor::<f64>::from_vec_col_major([n], vec![0.; n])?;
                s.add_read_into(
                    f64::tensor_read(&a),
                    f64::tensor_read(&b),
                    f64::tensor_write(&mut out),
                )?;
                check(out.host_data()?, &want)?;
                measure(timing, 0, || {
                    s.add_read_into(
                        f64::tensor_read(&a),
                        f64::tensor_read(&b),
                        f64::tensor_write(&mut out),
                    )?;
                    Ok(())
                })
            } else {
                let got = a.add(&b, s)?;
                check(got.host_data()?, &want)?;
                measure(timing, n * 8, || Ok(a.add(&b, s)?))
            }
        })?;
        held.close()?;
        m
    };
    Ok(outcome(
        Some(measurement),
        json!({"route": route, "requested_threads": threads}),
        if route == "scoped" {
            vec!["with_backend_session entry/exit", "typed allocating add"]
        } else if route == "into_held" {
            vec!["add_read_into with typed reads and caller output"]
        } else {
            vec!["typed allocating add in held session"]
        },
    ))
}

fn scalar_mm(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.; n * n];
    for j in 0..n {
        for i in 0..n {
            for k in 0..n {
                c[i + n * j] += a[i + n * k] * b[k + n * j];
            }
        }
    }
    c
}

fn matmul(p: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    let n = param_usize(p, "n")?;
    let a = TypedTensor::<f64>::from_vec_col_major([n, n], values(n * n, 112))?;
    let b = TypedTensor::<f64>::from_vec_col_major([n, n], values(n * n, 113))?;
    let want = scalar_mm(a.host_data()?, b.host_data()?, n);
    if param_str(p, "arm")? == "host" {
        let row = counter("scalar_matmul", n * n * 8, || {
            Ok(scalar_mm(a.host_data()?, b.host_data()?, n))
        })?;
        return Ok(outcome(None, counters(vec![row]), vec![]));
    }
    let backend = checked_backend(threads)?;
    let mut held = backend.open_session()?;
    let result = held.with_session(|s| -> Result<CaseOutcome> {
        let got = a.matmul(&b, s)?;
        check(got.host_data()?, &want)?;
        if param_str(p, "kind")? == "umbrella_matmul_counter" {
            let row = counter("typed_matmul", n * n * 8, || Ok(a.matmul(&b, s)?))?;
            Ok(outcome(None, counters(vec![row]), vec![]))
        } else {
            Ok(outcome(
                Some(measure(timing, n * n * 8, || Ok(a.matmul(&b, s)?))?),
                json!({}),
                vec!["typed allocating matmul in held session"],
            ))
        }
    });
    held.close()?;
    result
}

struct HostContainer {
    shape: [usize; 2],
    data: Vec<f64>,
}

fn construct(p: &Value, timing: &Timing) -> Result<CaseOutcome> {
    let n = param_usize(p, "n")?;
    let shape = [8, n / 8];
    let input = values(n, 114);
    let host = param_str(p, "arm")? == "host";
    // Payload buffers are prepared BEFORE each construction batch. Constructors
    // consume those exact Vecs; no clone, payload allocation or drop is timed.
    macro_rules! construct_arm {
        ($make:expr, $check:expr, $size:expr) => {{
            let make = $make;
            let first = make(input.clone())?;
            ($check)(&first, &input)?;
            let batch = |count: usize| -> Result<u128> {
                let mut inputs: Vec<_> = (0..count).map(|_| input.clone()).collect();
                let mut outputs = Vec::with_capacity(count);
                let start = Instant::now();
                for data in &mut inputs { outputs.push(make(std::mem::take(data))?); }
                let elapsed = start.elapsed().as_nanos();
                black_box(&outputs);
                Ok(elapsed)
            };
            let measured = measure_batches(timing, n*8 + $size, batch)?;
            let count = 64;
            let mut inputs: Vec<_> = (0..count).map(|_| input.clone()).collect();
            let mut outputs = Vec::with_capacity(count);
            let (_, allocs, bytes) = count_allocations(|| {
                for data in &mut inputs { outputs.push(make(std::mem::take(data))?); }
                Ok(())
            })?;
            black_box(&outputs);
            let mut extra = counters(vec![json!({"op":"construct_from_preallocated_vec", "output_bytes":n*8,
                "host_allocations_per_op":allocs as f64/count as f64,
                "host_allocated_bytes_per_op":bytes as f64/count as f64,
                "allocated_bytes_over_output_bytes":bytes as f64/count as f64/(n*8) as f64})]);
            extra["size_of_bytes"] = json!($size);
            extra["diagnostic_timing"] = json!({"scope":"container construction only; prepared payload Vecs consumed, outputs retained",
                "samples": measured.samples, "iterations":measured.iterations});
            Ok(outcome(None, extra, vec![]))
        }};
    }
    if host {
        construct_arm!(
            |data| -> Result<_> { Ok(HostContainer { shape, data }) },
            |t: &HostContainer, v: &[f64]| -> Result<()> {
                if t.shape != shape {
                    return Err("host shape mismatch".into());
                }
                check(&t.data, v)
            },
            std::mem::size_of::<HostContainer>()
        )
    } else {
        construct_arm!(
            |data| -> Result<_> {
                Ok(TypedTensor::<f64, Rank<2>, Host>::from_host_vec_col_major(
                    shape, data,
                )?)
            },
            |t: &TypedTensor<f64, Rank<2>, Host>, v: &[f64]| -> Result<()> {
                if t.shape() != shape {
                    return Err("typed shape mismatch".into());
                }
                check(t.as_slice(), v)
            },
            std::mem::size_of::<TypedTensor<f64, Rank<2>, Host>>()
        )
    }
}

fn entry(p: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    let mut backend = checked_backend(threads)?;
    let host = param_str(p, "arm")? == "host";
    let mut call = || -> Result<()> {
        if host {
            black_box(());
        } else if param_str(p, "route")? == "scoped" {
            backend.with_backend_session(|_| black_box(()))?;
        } else {
            backend.open_session()?.close()?;
        }
        Ok(())
    };
    call()?;
    Ok(outcome(
        Some(measure(timing, 0, call)?),
        json!({"diagnostic":true}),
        vec!["explicit session entry/exit or direct host empty callback"],
    ))
}

fn neg(p: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    let n = param_usize(p, "n")?;
    let a = TypedTensor::<f64>::from_vec_col_major([n], values(n, 115))?;
    let want: Vec<_> = a.host_data()?.iter().map(|x| -x).collect();
    let backend = checked_backend(threads)?;
    let mut held = backend.open_session()?;
    let result = held.with_session(|s| -> Result<CaseOutcome> {
        check(a.neg(s)?.host_data()?, &want)?;
        Ok(outcome(
            Some(measure(timing, n * 8, || Ok(a.neg(s)?))?),
            json!({}),
            vec!["typed allocating neg"],
        ))
    });
    held.close()?;
    result
}

fn gemm(p: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    let n = param_usize(p, "n")?;
    let a = Tensor::from_vec_col_major([n, n], vec![1_f64; n * n])?;
    let b = Tensor::from_vec_col_major([n, n], vec![2_f64; n * n])?;
    let plan = ConcreteEinsumPlan::prepare([&a, &b], "ij,jk->ik")?;
    let backend = checked_backend(threads)?;
    let mut held = backend.open_session()?;
    let result = held.with_session(|s| -> Result<CaseOutcome> {
        check(
            data(&plan.execute([&a, &b], s)?)?,
            &vec![2. * n as f64; n * n],
        )?;
        Ok(outcome(
            Some(measure(timing, n * n * 8, || {
                Ok(plan.execute([&a, &b], s)?)
            })?),
            json!({"requested_threads":threads,"flops_per_op":2*n*n*n}),
            vec!["prepared allocating einsum execution"],
        ))
    });
    held.close()?;
    result
}

fn prepared(p: &Value, threads: usize) -> Result<CaseOutcome> {
    let n = param_usize(p, "n")?;
    let a = Tensor::from_vec_col_major([n, n], vec![1_f64; n * n])?;
    let b = Tensor::from_vec_col_major([n, n], vec![2_f64; n * n])?;
    let plan = ConcreteEinsumPlan::prepare([&a, &b], "ij,jk->ik")?;
    let config = matmul_config(&[1], &[0]);
    let mut out = Tensor::from_vec_col_major([n, n], vec![0_f64; n * n])?;
    let backend = checked_backend(threads)?;
    let mut held = backend.open_session()?;
    let result = held.with_session(|s| -> Result<CaseOutcome> {
        let direct = param_str(p,"arm")? == "direct";
        let mut op = || -> Result<()> {
            if direct { s.dot_general_read_into(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b), &config, TensorWrite::Tensor(&mut out))?; }
            else { plan.execute_into([&a,&b], s, TensorWrite::Tensor(&mut out))?; }
            Ok(())
        };
        op()?;
        let before = plan.cpu_cache_stats()?;
        let row = counter("execute_into", n*n*8, &mut op)?;
        let after = plan.cpu_cache_stats()?;
        drop(op);
        check(data(&out)?, &vec![2.*n as f64; n*n])?;
        let mut extra = counters(vec![row]);
        extra["descriptor_cache_before"] = json!({"entries":before.entries,"hits":before.hits,"misses":before.misses,"evictions":before.evictions,"retained_bytes":before.retained_bytes});
        extra["descriptor_cache_after"] = json!({"entries":after.entries,"hits":after.hits,"misses":after.misses,"evictions":after.evictions,"retained_bytes":after.retained_bytes});
        extra["binary_plan_builds_per_call"] = Value::Null;
        extra["unavailable"].as_array_mut().unwrap().push(json!("binary-dot plan rebuild observer; descriptor cache misses are not plan builds"));
        if !direct {
            extra["correctness_status"] = json!("unsupported");
            extra["unsupported_reason"] = json!("Operation verified; requested binary plan-build counter has no public observer. Allocation/cache observations are retained, not proof of zero rebuilds.");
        }
        Ok(outcome(None, extra, vec![]))
    });
    held.close()?;
    result
}

/// Independent partial-pivot scalar LU, column-major batches. LAPACK pivots
/// are one-based swap indices, not the final permutation.
fn scalar_lu(a: &[f64], n: usize, batch: usize) -> (Vec<f64>, Vec<i32>, Vec<f64>) {
    let mut lu = a.to_vec();
    let mut piv = vec![0; n * batch];
    let mut parity = vec![1.; batch];
    for b in 0..batch {
        let t = &mut lu[b * n * n..(b + 1) * n * n];
        for k in 0..n {
            let pivot = (k..n)
                .max_by(|&i, &j| t[i + n * k].abs().total_cmp(&t[j + n * k].abs()))
                .unwrap();
            piv[k + b * n] = (pivot + 1) as i32;
            if pivot != k {
                for j in 0..n {
                    t.swap(k + n * j, pivot + n * j);
                }
                parity[b] = -parity[b];
            }
            for i in k + 1..n {
                t[i + n * k] /= t[k + n * k];
            }
            for j in k + 1..n {
                for i in k + 1..n {
                    t[i + n * j] -= t[i + n * k] * t[k + n * j];
                }
            }
        }
    }
    (lu, piv, parity)
}

fn lu(p: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    let n = param_usize(p, "n")?;
    let batch = param_usize(p, "batch")?;
    let mut av = values(n * n * batch, 116);
    for b in 0..batch {
        for i in 0..n {
            av[b * n * n + i + n * i] += n as f64;
        }
    }
    // Row swap forces a nontrivial pivot/parity, rather than only diagonal LU.
    for b in 0..batch {
        for j in 0..n {
            av.swap(b * n * n + n * j, b * n * n + 1 + n * j);
        }
    }
    let expected = scalar_lu(&av, n, batch);
    let mut bv = vec![0.; n * batch];
    for b in 0..batch {
        for i in 0..n {
            bv[i + n * b] = (0..n).map(|j| av[b * n * n + i + n * j]).sum();
        }
    }
    let a = Tensor::from_vec_col_major([n, n, batch], av)?;
    let b = Tensor::from_vec_col_major([n, 1, batch], bv)?;
    let bytes = n * n * batch * 8 + n * batch * 4 + batch * 8;
    let lane = param_str(p, "kind")? == "umbrella_lane_counter";
    if param_str(p, "arm")? == "host" {
        let mut op = || Ok(scalar_lu(data(&a)?, n, batch));
        return if lane {
            let row = counter("serial_scalar_lu", bytes, &mut op)?;
            let mut extra = counters(vec![row]);
            extra["actual_lanes"] = json!(1);
            extra["actual_item_threads"] = json!(1);
            Ok(outcome(None, extra, vec![]))
        } else {
            Ok(outcome(
                Some(measure(timing, bytes, op)?),
                json!({}),
                vec!["independent scalar partial-pivot packed LU, pivots and parity"],
            ))
        };
    }
    let backend = checked_backend(threads)?;
    let mut held = backend.open_session()?;
    let result = held.with_session(|s| -> Result<CaseOutcome> {
        with_cpu_exec_session(s, |cpu| -> Result<CaseOutcome> {
            let factors = LinalgBackend::lu_factor(cpu,&a)?;
            if factors.len()!=3 { return Err("expected packed LU, pivots and parity".into()); }
            check(data(&factors[0])?,&expected.0)?;
            if factors[1].as_slice::<i32>()? != expected.1 { return Err("wrong LU pivots".into()); }
            check(data(&factors[2])?,&expected.2)?;
            if lane {
                let row = counter("batched_lu_factor",bytes,|| Ok(LinalgBackend::lu_factor(cpu,&a)?))?;
                let mut extra = counters(vec![row]);
                extra["requested_threads"] = json!(threads);
                extra["actual_lanes"] = Value::Null; extra["actual_item_threads"] = Value::Null;
                extra["correctness_status"] = json!("unsupported");
                extra["unsupported_reason"] = json!("LU verified; no public observer for the internal linalg lane plan. Requested threads and allocator counts do not establish lanes/item-threads.");
                return Ok(outcome(None,extra,vec![]));
            }
            if param_str(p,"op")? == "factor" {
                Ok(outcome(Some(measure(timing,bytes,|| Ok(LinalgBackend::lu_factor(cpu,&a)?))?),json!({}),vec!["public LinalgBackend packed LU, pivots and parity"]))
            } else {
                let got = LinalgBackend::lu_solve_prepared(cpu,&a,&factors[0],&factors[1],&b,false,false)?;
                check(data(&got)?,&vec![1.;n*batch])?;
                Ok(outcome(Some(measure(timing,n*batch*8,|| Ok(LinalgBackend::lu_solve_prepared(cpu,&a,&factors[0],&factors[1],&b,false,false)?))?),
                    json!({"prepared_factorization_outside_timing":true}),vec!["public LinalgBackend prepared LU solve only"]))
            }
        }).ok_or("CPU execution session unavailable")?
    });
    held.close()?;
    result
}

fn axpby(p: &Value, timing: &Timing, threads: usize) -> Result<CaseOutcome> {
    let n = param_usize(p, "n")?;
    let x = Tensor::from_vec_col_major([n, n], values(n * n, 117))?;
    let stride = n + 3;
    let offset = 2;
    let storage_len = offset + stride * n;
    let backend = checked_backend(threads)?;
    let mut held = backend.open_session()?;
    let result = held.with_session(|s| -> Result<CaseOutcome> {
        let host = param_str(p,"arm")? == "host";
        let mut batch = |count: usize| -> Result<u128> {
            let mut outputs = vec![vec![3_f64; storage_len];count];
            // Mutable strided view construction and seed restoration are untimed.
            let mut writes: Vec<_> = outputs.iter_mut().map(|v| {
                Ok(Some(TensorWrite::View(TensorViewMut::F64(TypedTensorViewMut::from_slice(
                    [n,n],[1,stride as isize],offset as isize,v)?))))
            }).collect::<Result<_>>()?;
            let start = Instant::now();
            for write in &mut writes { s.axpby_read_into_accum(ContractionScalar::F64(1.),TensorRead::from_tensor(&x),ContractionScalar::F64(1.),write.take().unwrap())?; }
            let elapsed = start.elapsed().as_nanos();
            drop(writes);
            validate_blocks(&outputs,data(&x)?,n,stride,offset)?;
            black_box(&outputs);
            Ok(elapsed)
        };
        let m = if host {
            let batch = |count: usize| -> Result<u128> {
                let mut outputs = vec![vec![3_f64; storage_len];count];
                let xv = data(&x)?;
                let start = Instant::now();
                for out in &mut outputs { for j in 0..n { for i in 0..n { out[offset+i+stride*j] += xv[i+n*j]; } } }
                let elapsed = start.elapsed().as_nanos();
                validate_blocks(&outputs,xv,n,stride,offset)?;
                black_box(&outputs);
                Ok(elapsed)
            };
            batch(1)?;
            measure_batches(timing,storage_len*8,batch)?
        } else {
            batch(1)?;
            measure_batches(timing,storage_len*8,&mut batch)?
        };
        Ok(outcome(Some(m),json!({"destination_shape":[n,n],"destination_strides":[1,stride],"destination_offset":offset,
            "alpha":1,"beta":1,"input_seed":3,"independent_destinations_per_batch":true}), vec!["strided alpha*x+beta*y, no staging"]))
    });
    held.close()?;
    result
}

fn validate_blocks(
    outputs: &[Vec<f64>],
    x: &[f64],
    n: usize,
    stride: usize,
    offset: usize,
) -> Result<()> {
    for out in outputs {
        for (k, &v) in out.iter().enumerate() {
            let q = k.checked_sub(offset);
            let expected = match q {
                Some(q) if q / stride < n && q % stride < n => {
                    3. + x[q % stride + n * (q / stride)]
                }
                _ => 3.,
            };
            check_tol(
                rel_error(v, expected),
                1e-12,
                "strided output or guard region",
            )?;
        }
    }
    Ok(())
}

/// tenferro-rs #2170: the buffer pool's acquisition cost against the number of
/// distinct capacities the run has seen. This is the issue's own minimal
/// reproducer at the published pool crate's public surface, and a state
/// diagnostic: the capacity history is built outside the timed batches and the
/// measured operation is the pool call itself.
fn pool_state(p: &Value, timing: &Timing) -> Result<CaseOutcome> {
    use tenferro_cpu_basic::buffer_pool::{BufferPool, PoolScalar};

    let route = param_str(p, "route")?;
    let bins = param_usize(p, "bins")?;
    let len = p["len"].as_u64().map(|v| v as usize).unwrap_or(1);
    let cap_bytes = p["cap_bytes"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(usize::MAX);

    let mut pool = BufferPool::with_max_retained_capacity_bytes(cap_bytes);
    let mut capacities = Vec::with_capacity(bins);
    for index in 0..bins {
        let buffer = pool.acquire_with_capacity::<f64>(1 + index);
        capacities.push(buffer.capacity());
        <f64 as PoolScalar>::pool_release(&mut pool, buffer);
    }

    // The reporter's reproducer checks every pooled buffer out and holds it, so
    // the capacity bins stay behind empty and a request nothing can serve walks
    // them. The other routes keep their bins served.
    let mut held: Vec<Vec<f64>> = Vec::new();
    if route == "miss" {
        held = capacities
            .iter()
            .rev()
            .map(|&capacity| pool.acquire_with_capacity::<f64>(capacity))
            .collect();
    }
    // The retained accounting must be identical before and after the timed
    // batches, so the rows measure a steady state rather than a state shift. The
    // snapshot is taken after each route's own warm-up probe.
    let steady;

    let (measurement, check) = match route {
        "miss" => {
            let served_before = pool.stats().buffers;
            let probe = pool.acquire_with_capacity::<f64>(len);
            let probe_capacity = probe.capacity();
            if probe_capacity < len {
                return Err("the unservable request returned a short buffer".into());
            }
            if pool.stats().buffers > served_before {
                return Err("the unservable request was served from the pool".into());
            }
            steady = pool.stats();
            let measurement =
                measure(timing, len * 8, || Ok(pool.acquire_with_capacity::<f64>(len)))?;
            (measurement, json!({"probe_capacity": probe_capacity}))
        }
        "hit" | "cycle" => {
            let capacity = if route == "hit" {
                *capacities
                    .first()
                    .ok_or("the hit route needs a retained capacity")?
            } else {
                len
            };
            let probe = pool.acquire_with_capacity::<f64>(capacity);
            let pointer = probe.as_ptr();
            <f64 as PoolScalar>::pool_release(&mut pool, probe);
            let again = pool.acquire_with_capacity::<f64>(capacity);
            if again.as_ptr() != pointer {
                return Err("the reuse control did not reuse a retained buffer".into());
            }
            <f64 as PoolScalar>::pool_release(&mut pool, again);
            steady = pool.stats();
            let measurement = measure(timing, capacity * 8, || {
                let buffer = pool.acquire_with_capacity::<f64>(capacity);
                <f64 as PoolScalar>::pool_release(&mut pool, buffer);
                Ok(())
            })?;
            (measurement, json!({"reused_capacity": capacity}))
        }
        "in_flight" => {
            // Acquire/release cycling across the capacity history: the tracked
            // path, so every cycle passes through the in-flight bookkeeping.
            let mut step = 0usize;
            // Warm-up probe before the measured closure takes the pool.
            let probe = pool.acquire_with_capacity::<f64>(len);
            <f64 as PoolScalar>::pool_release(&mut pool, probe);
            steady = pool.stats();
            let cycle = || {
                let capacity = capacities
                    .get(step % capacities.len().max(1))
                    .copied()
                    .unwrap_or(len);
                step += 1;
                let buffer = pool.acquire_with_capacity::<f64>(capacity);
                <f64 as PoolScalar>::pool_release(&mut pool, buffer);
                Ok(())
            };
            let measurement = measure(timing, 8, cycle)?;
            (measurement, json!({"cycled_capacities": capacities.len()}))
        }
        other => return Err(format!("unknown pool route {other}").into()),
    };

    let after = pool.stats();
    let cache = pool.cache_stats();
    if after.buffers != steady.buffers || after.capacity_bytes != steady.capacity_bytes {
        return Err("the measured pool route changed the retained accounting".into());
    }
    drop(held);
    Ok(outcome(
        Some(measurement),
        json!({
            "route": route,
            "history_capacities": capacities.len(),
            "retention_cap_bytes": if cap_bytes == usize::MAX { None } else { Some(cap_bytes) },
            "retained_buffers": after.buffers,
            "retained_capacity_bytes": after.capacity_bytes,
            "steady_buffers": steady.buffers,
            "steady_capacity_bytes": steady.capacity_bytes,
            "cache_entries": cache.entries,
            "cache_retained_bytes": cache.retained_bytes,
            "check": check,
            "pool_single_threaded": true,
            "unavailable": [
                "emptied-bin count and the resulting scan length inside the pool                  (tenferro-cpu-basic exposes no observation point outside its own tests)",
                "allocator-level buffer reuse behind the returned vectors",
            ],
        }),
        vec!["one pool acquisition, or one acquire/release cycle"],
    ))
}

fn views(p: &Value) -> Result<CaseOutcome> {
    let owner =
        TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major([64, 64], values(4096, 118))?;
    let view = owner.as_view();
    let op = param_str(p, "op")?;
    let (shape, strides, offset) = match op {
        "transpose" => ([64, 64], [64, 1], 0),
        "slice" => ([32, 64], [2, 64], 1),
        "reshape" => ([32, 128], [1, 32], 0),
        _ => return Err("unknown view op".into()),
    };
    if param_str(p, "arm")? == "host" {
        // A plain borrowing descriptor baseline; no tensor materialization.
        let call = || -> Result<_> {
            let (dims, steps, start) = ([64_usize, 64], [1_isize, 64], 0_isize);
            Ok(match op {
                "transpose" => (
                    [dims[1], dims[0]],
                    [steps[1], steps[0]],
                    start,
                    owner.as_slice(),
                ),
                "slice" => (
                    [(dims[0] - 1).div_ceil(2), dims[1]],
                    [2 * steps[0], steps[1]],
                    start + steps[0],
                    owner.as_slice(),
                ),
                "reshape" => (
                    [32, dims[0] * dims[1] / 32],
                    [1, 32],
                    start,
                    owner.as_slice(),
                ),
                _ => unreachable!(),
            })
        };
        let first = call()?;
        if first.0 != shape
            || first.1 != strides
            || first.2 != offset
            || !std::ptr::eq(first.3.as_ptr(), owner.as_slice().as_ptr())
        {
            return Err("incorrect host descriptor or owner alias".into());
        }
        let row = counter("host_view_descriptor", 0, call)?;
        return Ok(outcome(None, counters(vec![row]), vec![]));
    }
    let call = || -> Result<_> {
        Ok(match op {
            "transpose" => view.transpose_view([1, 0])?,
            "slice" => view.slice_axis_view(0, StridedSliceSpec::new(1, Some(64), 2))?,
            "reshape" => view.reshape_view(&shape)?,
            _ => unreachable!(),
        })
    };
    let got = call()?;
    if got.shape() != shape || got.strides() != strides || got.offset() != offset {
        return Err("incorrect view metadata".into());
    }
    for j in 0..shape[1] {
        for i in 0..shape[0] {
            let at = offset as usize + i * strides[0] as usize + j * strides[1] as usize;
            if !std::ptr::eq(
                got.get(&[i, j]).ok_or("view element missing")?,
                &owner.as_slice()[at],
            ) {
                return Err("view copied or indexed wrong storage".into());
            }
        }
    }
    let row = counter(op, 0, call)?;
    Ok(outcome(None, counters(vec![row]), vec![]))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validators_reject_nonfinite_values_and_guard_region_writes() {
        assert!(check(&[f64::NAN], &[1.]).is_err());
        assert!(check(&[f64::INFINITY], &[1.]).is_err());
        let mut out = vec![vec![3.; 8]];
        let x = vec![1.; 4];
        for at in [1, 2, 4, 5] {
            out[0][at] = 4.;
        }
        assert!(validate_blocks(&out, &x, 2, 3, 1).is_ok());
        out[0][0] = 4.;
        assert!(validate_blocks(&out, &x, 2, 3, 1).is_err());
    }
}
