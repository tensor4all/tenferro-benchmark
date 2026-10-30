//! Whole-sequence planning latency; counters run in a separate untimed pass.
use super::*;
use tenferro_tensor::TensorDot;

/// Public CUDA runtime counters, one snapshot per phase.
///
/// tenferro-gpu v0.8.0's kernel-catalogue cutover removed the cuTENSOR plan
/// cache, vendor workspace and workspace-retirement counters: the plan cache
/// now lives inside the engine's private cuTENSOR handle and has no public
/// counter or capacity knob. The runtime's own counters remain public, so a
/// counter pass reports the allocations, zero fills, launches, retirement state
/// and artifact/module cache activity each phase causes instead. `stats()` may
/// retire detached work, so these snapshots are observation-only and never a
/// synchronisation primitive (the timing rows never call them).
fn counters(b: &CudaEngine) -> Result<Value> {
    let rt = b.runtime();
    let s = rt.stats();
    Ok(json!({
        "device":{"live_allocations":s.live_allocations,"live_bytes":s.live_bytes,
            "pool_used_bytes":s.pool_used_bytes,"pool_reserved_bytes":s.pool_reserved_bytes,
            "idle_pinned_bytes":s.idle_pinned_bytes},
        "retirement":{"pending_bytes":s.retirement.pending_bytes,
            "module_bytes":s.retirement.module_bytes,
            "pending_records":s.retirement.pending_records,
            "epochs":s.retirement.epochs,"frontiers":s.retirement.frontiers,
            "reclaimed_epochs":s.retirement.reclaimed_epochs,
            "launches":s.retirement.launches,"allocations":s.retirement.allocations,
            "zero_fills":s.retirement.zero_fills,
            "events_recorded":s.retirement.events_recorded},
        "artifact_cache":{"entries":s.artifact_entries,"bytes":s.artifact_bytes},
        "module_cache":{"entries":s.module_entries,"bytes":s.module_bytes},
        "compile_attempts":rt.compile_attempts(),"module_loads":rt.module_loads(),
        "artifact_cache_hits":rt.artifact_cache_hits(),
    }))
}

pub(super) fn run(params: &Value, timing: &Timing, device: usize) -> Result<CaseOutcome> {
    let n = param_usize(params, "n")?;
    if ![16, 64].contains(&n) {
        return Err("shape-stream size must be 16 or 64".into());
    }
    let mode = param_str(params, "arm")?;
    let instrument = params["instrument"].as_bool().unwrap_or(false);
    let (keys, order): (usize, Vec<usize>) = match mode {
        "fixed-warm" => (1, vec![0; 32]),
        "mixed-warm" => (3, (0..32).map(|i| [0, 1, 2, 0, 2, 1][i % 6]).collect()),
        "fresh" => (8, (0..8).collect()),
        "capacity-thrash" => (72, (0..72).chain([0]).collect()),
        "equal-shape-strides" => (2, (0..32).map(|i| i % 2).collect()),
        _ => return Err("unknown shape-stream mode".into()),
    };
    let mut bk = backend(device)?;
    let rt = bk.runtime().clone();
    let cfg = DotGeneralConfig {
        lhs_contracting_dims: [2].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [1].as_slice().into(),
        rhs_batch_dims: [1].as_slice().into(),
    };
    let mut operands = Vec::with_capacity(keys);
    let mut references = Vec::with_capacity(keys);
    for key in 0..keys {
        let m = n + if mode == "equal-shape-strides" {
            0
        } else {
            key
        };
        let a = values(m * 2 * n, 90);
        let b = values(n * 2 * 3, 190);
        let strided = mode == "equal-shape-strides" && key == 1;
        let (shape, storage) = if strided {
            let mut transposed = vec![0.0; a.len()];
            for i in 0..m {
                for batch in 0..2 {
                    for k in 0..n {
                        transposed[k + n * (batch + 2 * i)] = a[i + m * (batch + 2 * k)];
                    }
                }
            }
            (vec![n, 2, m], transposed)
        } else {
            (vec![m, 2, n], a.clone())
        };
        operands.push((
            upload_tensor(&rt, &host(shape, storage)?)?,
            upload_tensor(&rt, &host(vec![n, 2, 3], b.clone())?)?,
        ));
        let mut expected = vec![0.0; 2 * m * 3];
        for j in 0..3 {
            for i in 0..m {
                for batch in 0..2 {
                    expected[i + m * (j + 3 * batch)] = (0..n)
                        .map(|k| a[i + m * (batch + 2 * k)] * b[k + n * (batch + 2 * j)])
                        .sum();
                }
            }
        }
        references.push(expected);
    }
    rt.synchronize()?;
    let retained_bytes = order
        .iter()
        .map(|&key| references[key].len() * 8)
        .sum::<usize>();
    if retained_bytes > 256 << 20 {
        return Err("shape-stream retention exceeds 256 MiB".into());
    }
    let reads = operands
        .iter()
        .enumerate()
        .map(|(key, (a, b))| -> Result<_> {
            let lhs = if mode == "equal-shape-strides" && key == 1 {
                TensorRead::from_view(TensorView::F64(
                    a.as_typed::<f64>()
                        .ok_or("not f64")?
                        .as_view()
                        .transpose_view([2, 1, 0])?,
                ))
            } else {
                TensorRead::from_tensor(a)
            };
            Ok((lhs, TensorRead::from_tensor(b)))
        })
        .collect::<Result<Vec<_>>>()?;
    let sequence = order
        .iter()
        .map(|&key| -> Result<_> {
            Ok(json!({"key":key,
        "lhs_shape":reads[key].0.shape(),"rhs_shape":reads[key].1.shape(),
            "lhs_strides":reads[key].0.strides()?,
            "rhs_strides":reads[key].1.strides()?,
        "lhs_layout":if mode == "equal-shape-strides" && key == 1 {"transposed"}else{"compact"}}))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut error = 0.0_f64;
    let mut observations = Vec::new();
    let mut enqueue_ns = Vec::new();
    let measurement = bk.with_backend_session(|s| -> Result<_> {
        let mut validate = |outputs: &[Tensor]| -> Result<()> {
            for (out, &key) in outputs.iter().zip(&order) {
                let placement = out
                    .placement()
                    .device
                    .as_ref()
                    .ok_or("shape stream CPU fallback")?;
                if placement.kind
                    != tenferro_tensor::DeviceKind::Gpu(tenferro_tensor::GpuBackendKind::Cuda)
                    || placement.ordinal != device
                    || !out.is_backend_buffer()
                {
                    return Err("shape stream wrong CUDA placement".into());
                }
                let m = reads[key].0.shape()[0];
                if out.shape() != [m, 3, 2]
                    || TensorRead::from_tensor(out).allocation_domain().is_none()
                {
                    return Err(format!("shape stream output shape/domain mismatch: shape={:?}, expected={:?}, domain={:?}",
                        out.shape(), [m,3,2], TensorRead::from_tensor(out).allocation_domain()).into());
                }
                let got = download_tensor(&rt, out)?;
                error = error.max(check_tol(
                    max_rel(got.as_slice::<f64>()?, &references[key]),
                    1e-10,
                    "shape stream full output",
                )?);
            }
            Ok(())
        };
        // Initialize the library/route explicitly before any interval, even
        // when the caller requests zero warmups. Planning is reset per sample.
        let mut initialized = Vec::with_capacity(order.len());
        for &key in &order {
            initialized.push(s.dot_general_read(
                reads[key].0.clone(),
                reads[key].1.clone(),
                &cfg,
            )?);
        }
        rt.synchronize()?;
        validate(&initialized)?;
        drop(initialized);
        rt.synchronize()?;
        let mut batch = || -> Result<u128> {
            // Clear only the CUDA extension-owned caches. Compiled kernels,
            // the cuTENSOR library handle and allocator pools remain
            // initialized. tenferro-gpu v0.8.0 owns the extension cache store
            // through `ExecutionEngine`; taking it out and dropping it is the
            // reset, and the engine-owned cuTENSOR plan cache has neither a
            // public reset nor a capacity knob (its cache bound is internal).
            drop(s.take_extension_caches());
            rt.synchronize()?;
            if mode.ends_with("warm") || mode == "equal-shape-strides" {
                for (a, b) in &reads {
                    let out = s.dot_general_read(a.clone(), b.clone(), &cfg)?;
                    rt.synchronize()?;
                    drop(out);
                }
            }
            let mut prepared = order
                .iter()
                .map(|&key| Some(reads[key].clone()))
                .collect::<Vec<_>>();
            let mut outputs = Vec::with_capacity(order.len());
            if instrument {
                observations.push(json!({"phase":"before","stats":counters(s)?}));
            }
            let start = (!instrument && timing.samples > 0).then(Instant::now);
            for (index, item) in prepared.iter_mut().enumerate() {
                let (a, b) = item.take().ok_or("missing prepared read")?;
                outputs.push(s.dot_general_read(a, b, &cfg)?);
                if instrument {
                    observations.push(json!({"phase":"enqueue","index":index,
                    "key":order[index],"stats":counters(s)?}));
                }
            }
            let queued = start.map(|t| t.elapsed().as_nanos());
            rt.synchronize()?;
            let elapsed = start.map(|t| t.elapsed().as_nanos()).unwrap_or(0);
            if let Some(ns) = queued {
                enqueue_ns.push(ns);
            }
            if instrument {
                observations.push(json!({"phase":"completion","stats":counters(s)?}));
            }
            validate(&outputs)?;
            drop(outputs);
            rt.synchronize()?;
            if instrument {
                observations.push(json!({"phase":"drop","stats":counters(s)?}));
            }
            Ok(elapsed)
        };
        if instrument || timing.samples == 0 {
            batch()?;
            Ok(None)
        } else {
            // Initialization was explicit above. Respect zero warmups and
            // sample whole sequences without adaptive per-operation batching.
            for _ in 0..timing.warmups { batch()?; }
            let mut samples = Vec::with_capacity(timing.samples);
            for sample_index in 0..timing.samples {
                let elapsed_ns = batch()?;
                if elapsed_ns == 0 { return Err("zero shape-sequence duration".into()); }
                samples.push(Sample { sample_index, elapsed_ns, iterations:1 });
            }
            Ok(Some(Measurement { iterations:1, samples }))
        }
    })??;
    let measured_enqueue = if instrument {
        Vec::new()
    } else {
        enqueue_ns
            .iter()
            .skip(timing.warmups)
            .copied()
            .collect::<Vec<_>>()
    };
    Ok(CaseOutcome { max_rel_error:error, measurement,
        timer:vec!["entire declared shape sequence, including required planning, then native synchronization"],
        outside_timer:vec!["backend/session/library initialization, input generation/upload and read-handle preparation",
            "controlled CUDA extension-cache reset and declared warm-key precondition",
            "full-output download/verification and output destruction"],
        extra:json!({"sequence":sequence,"operations_per_sequence":order.len(),"retained_output_bytes":retained_bytes,
            "cache_mode":mode,"cache_reset_scope":"CUDA extension-owned caches; excludes compiled kernels, cuTENSOR handle and output/allocator pools",
            "instrumentation_only":instrument,"counter_observations":observations,
            "host_enqueue_samples_ns":measured_enqueue,
            "host_enqueue_ns_including_warmups":enqueue_ns,
            "latency_classification":"whole-sequence workflow diagnostic; not steady-state operation throughput"}) })
}
