//! #107: whole public-call sequences with separately observed cache counters.
use std::{error::Error, hint::black_box, num::NonZeroUsize, time::Instant};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{EagerSessionEinsumExt, EinsumSubscripts};
use tenferro_runtime::{ExtensionCacheLimits, ExtensionCacheSelector};
use tenferro_tensor::Tensor;
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn get(args: &[String], key: &str, default: &str) -> String {
    args.windows(2)
        .find(|p| p[0] == key)
        .map(|p| p[1].clone())
        .unwrap_or(default.into())
}
fn key(stream: &str, pass: usize, index: usize, length: usize) -> (usize, usize, usize, bool) {
    match stream {
        "fixed" | "cold" => (8, 8, 8, false),
        "mixed" | "thrash" => {
            let n = [6, 8, 7, 9][index % 4];
            (n, n, n, false)
        }
        "strides" => (8, 8, 8, index % 2 == 1),
        "fresh" => {
            let id = pass * length + index;
            (2 + id / 169, 2 + id % 13, 2 + (id / 13) % 13, false)
        }
        _ => unreachable!(),
    }
}
struct Call {
    inputs: [EagerTensor; 3],
    expected: Vec<f64>,
    m: usize,
    key: [usize; 4],
}
fn call(
    runtime: &std::sync::Arc<EagerRuntime>,
    geometry: (usize, usize, usize, bool),
    seed: usize,
) -> Result<Call> {
    let (m, k, n, row) = geometry;
    let a: Vec<f64> = (0..m * k)
        .map(|p| ((p % m * 3 + p / m * 7 + seed) % 17) as f64 / 32. - 0.25)
        .collect();
    let b: Vec<f64> = (0..k * n)
        .map(|p| ((p % k * 5 + p / k * 3 + seed) % 19) as f64 / 32. - 0.2)
        .collect();
    let c: Vec<f64> = (0..n * 2)
        .map(|p| ((p % n * 7 + p / n * 11 + seed) % 13) as f64 / 32. - 0.15)
        .collect();
    let mut ab = vec![0.; m * n];
    let mut expected = vec![0.; m * 2];
    for j in 0..n {
        for p in 0..k {
            for i in 0..m {
                ab[i + m * j] += a[i + m * p] * b[p + k * j];
            }
        }
    }
    for j in 0..2 {
        for p in 0..n {
            for i in 0..m {
                expected[i + m * j] += ab[i + m * p] * c[p + n * j];
            }
        }
    }
    let input_a = if row {
        let storage = (0..m * k).map(|p| a[p / k + m * (p % k)]).collect();
        let base = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major([k, m], storage)?,
            runtime.clone(),
        )?;
        runtime.with_eager_session(|s| s.transpose(&base, &[1, 0]))?
    } else {
        EagerTensor::from_tensor_in(Tensor::from_vec_col_major([m, k], a)?, runtime.clone())?
    };
    let inputs = [
        input_a,
        EagerTensor::from_tensor_in(Tensor::from_vec_col_major([k, n], b)?, runtime.clone())?,
        EagerTensor::from_tensor_in(Tensor::from_vec_col_major([n, 2], c)?, runtime.clone())?,
    ];
    let expected_strides = if row {
        vec![k as isize, 1]
    } else {
        vec![1, m as isize]
    };
    if inputs[0].tensor_read().strides()? != expected_strides {
        return Err("eager input did not preserve declared layout".into());
    }
    if inputs.iter().any(EagerTensor::tracks_grad) {
        return Err("tracked input".into());
    }
    Ok(Call {
        inputs,
        expected,
        m,
        key: [m, k, n, usize::from(row)],
    })
}
fn snapshot(s: &mut tenferro_ad::EagerSession<'_>) -> Result<serde_json::Value> {
    let stats = s.with_extension_caches(|c| c.stats(ExtensionCacheSelector::All))?;
    Ok(
        serde_json::json!({"entries":stats.entries,"retained_bytes":stats.retained_bytes,"hits":stats.hits,"misses":stats.misses,"evictions":stats.evictions,"clears":stats.clears,"builds":"unavailable: no public build counter"}),
    )
}
pub fn run(args: &[String]) -> Result<()> {
    let id = get(args, "--case", "");
    let mode = get(args, "--mode", "measure");
    let requested = get(args, "--provider", "blas");
    let compiled = tenferro_einsum_benchmark::compiled_cpu_provider();
    if requested != compiled {
        println!(
            "{}",
            serde_json::json!({"case_id":id,"provider":requested,"correctness":"unsupported","unsupported_reason":format!("compiled CPU provider {compiled}, requested {requested}"),"samples_ns":[]})
        );
        return Ok(());
    }
    if !matches!(
        mode.as_str(),
        "measure" | "cache-diagnostic" | "correctness-only"
    ) {
        return Err("invalid stream mode".into());
    }
    let cases: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(get(
        args,
        "--instances",
        "data/instances/session_matrix.json",
    ))?)?;
    let spec = cases
        .iter()
        .find(|c| c["id"] == id)
        .ok_or("unknown eager stream")?;
    let stream = spec["stream"].as_str().ok_or("missing stream")?;
    if !matches!(
        stream,
        "fixed" | "mixed" | "fresh" | "strides" | "thrash" | "cold"
    ) {
        return Err("invalid stream".into());
    }
    let length = spec["length"].as_u64().ok_or("missing length")? as usize;
    let warmups: usize = get(args, "--warmups", "3").parse()?;
    let samples: usize = get(args, "--samples", "15").parse()?;
    let threads: usize = get(args, "--threads", "1").parse()?;
    if length == 0 || samples == 0 || length * (1 + warmups + samples) > 8192 {
        return Err("stream input memory/key budget exceeded".into());
    }
    tenferro_einsum_benchmark::thread_enforcement::enforce_thread_request(threads)?;
    let setup = Instant::now();
    let backend = CpuBackend::new();
    let workers = backend.num_threads();
    tenferro_einsum_benchmark::thread_enforcement::verify_backend_threads(
        "CpuBackend",
        workers,
        threads,
    )?;
    let runtime = EagerRuntime::with_cpu_backend(backend)?;
    let _no_grad = runtime.no_grad();
    if stream == "thrash" {
        runtime.set_extension_cache_limits(ExtensionCacheLimits::new(
            NonZeroUsize::new(1).ok_or("zero limit")?,
        ))?;
    }
    let limits = runtime.extension_cache_limits()?;
    let passes: Vec<Vec<Call>> = (0..1 + warmups + samples)
        .map(|p| {
            (0..length)
                .map(|i| call(&runtime, key(stream, p, i, length), p * length + i))
                .collect()
        })
        .collect::<Result<_>>()?;
    let setup_ns = setup.elapsed().as_nanos();
    let subs = EinsumSubscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let mut durations = Vec::with_capacity(samples);
    let mut counters = Vec::with_capacity(samples);
    let mut worst = 0f64;
    let mut verification_ns = 0u128;
    let mut warmup_ns = 0u128;
    runtime.with_eager_session(|s| -> Result<()> {
        for (p, pass) in passes.iter().enumerate() {
            // Reset only the declared eager extension caches. This is not bypass:
            // lookups and retention remain enabled inside the public call.
            if stream == "cold" {
                s.with_extension_caches(|c| c.clear())?;
            }
            let bindings: Vec<_> = pass
                .iter()
                .map(|c| [&c.inputs[0], &c.inputs[1], &c.inputs[2]])
                .collect();
            let mut outputs = Vec::with_capacity(length);
            let before = if mode == "cache-diagnostic" {
                Some(snapshot(s)?)
            } else {
                None
            };
            let clock = if mode == "measure" {
                Some(Instant::now())
            } else {
                None
            };
            for inputs in &bindings {
                outputs.push(s.einsum_subscripts(inputs, &subs)?);
            }
            let elapsed = clock.map(|c| c.elapsed().as_nanos());
            black_box(&outputs);
            let after = if mode == "cache-diagnostic" {
                Some(snapshot(s)?)
            } else {
                None
            };
            let verification = Instant::now();
            for (out, call) in outputs.iter().zip(pass) {
                if out.tracks_grad() || out.shape() != [call.m, 2] {
                    return Err("tracked or incorrectly shaped stream output".into());
                }
                let tensor = out.to_tensor()?;
                for (x, y) in tensor.as_slice::<f64>()?.iter().zip(&call.expected) {
                    let err = (x - y).abs();
                    if !err.is_finite() || err > 1e-10 * (1. + y.abs()) {
                        return Err("stream numerical mismatch".into());
                    }
                    worst = worst.max(err);
                }
            }
            verification_ns += verification.elapsed().as_nanos();
            if p > warmups {
                if let Some(ns) = elapsed {
                    durations.push(ns);
                }
                if let (Some(before), Some(after)) = (before, after) {
                    counters.push(serde_json::json!({"before":before,"after":after}));
                }
            } else if let Some(ns) = elapsed {
                warmup_ns += ns;
            }
        }
        Ok(())
    })?;
    let orders: Vec<_> = passes
        .iter()
        .map(|p| p.iter().map(|c| c.key).collect::<Vec<_>>())
        .collect();
    println!(
        "{}",
        serde_json::json!({"case_id":id,"workload":"eager_stream","operation":"einsum3","dtype":"f64","provider":compiled,"correctness":"passed","max_abs_error":worst,"samples_ns":durations,"cache_diagnostics":counters,"diagnostic_mode":mode,"requested_threads":threads,"worker_count":workers,"operations_per_sample":length,"warmups":warmups,"initialization_passes":1,"timing_scope":"whole public-call sequence, allocation-returning, retained output, borrowed eager session","planning_in_timer":true,"sequence_order":orders,"sequence_key_fields":["m","k","n","row-major-left-view"],"subscripts":"ij,jk,kl->il (last output extent=2)","cache_mode":if stream=="cold"{"eager extension caches cleared outside every sequence; lookup and retention enabled"}else if stream=="thrash"{"one-entry eager extension cache; four keys revisited; lookup and retention enabled"}else{"default eager extension cache; pass0 initializes before samples; no bypass"},"cache_limits":{"max_entries":limits.max_entries().get(),"max_retained_bytes":limits.max_retained_bytes().map(NonZeroUsize::get)},"cache_counter_scope":"all EagerRuntime extension caches; excludes provider GEMM-analysis, indexed-plan and runtime prepared caches; diagnostics run in a separate process; misses are not equated to plan builds","phase_wall_ms":{"input_and_runtime_setup":setup_ns as f64/1e6,"verification_and_materialization":verification_ns as f64/1e6,"initialization_and_warmup_execution":warmup_ns as f64/1e6,"measured_execution":durations.iter().sum::<u128>()as f64/1e6},"outside_timer":["all pass inputs/view creation and eager wrapping","runtime/backend/session creation","explicit cold reset","binding and retention containers","counter observations (separate process)","numerical checks","output destruction"]})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fresh_keys_never_repeat_and_stride_keys_preserve_shape() {
        let keys: std::collections::HashSet<_> = (0..19)
            .flat_map(|p| (0..128).map(move |i| key("fresh", p, i, 128)))
            .collect();
        assert_eq!(keys.len(), 19 * 128);
        assert_eq!(key("strides", 0, 0, 128), (8, 8, 8, false));
        assert_eq!(key("strides", 0, 1, 128), (8, 8, 8, true));
    }
}
