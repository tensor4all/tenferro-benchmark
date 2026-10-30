//! Separate, untimed public-API observations; never operation result rows.
use super::{Case, EngineScope, Result};
use serde_json::{json, Value};
use std::{fs, io::Write, path::Path};
use tenferro_fft::FftExecutor;
use tenferro_gpu::cuda::{CudaDeviceId, CudaEngine};
use tenferro_runtime::ExecutionEngine;

/// The engine's extension-owned caches, read and put back in place.
///
/// tenferro-gpu v0.8.0 borrows the cache store from the engine for the length
/// of one operation (`ExecutionEngine::take_extension_caches` /
/// `restore_extension_caches`), so a snapshot takes it out, reads its counters
/// and restores it unchanged. The store is moved, never re-planned.
fn extension_cache(backend: &mut CudaEngine) -> Value {
    let store = backend.take_extension_caches();
    let stats = store.stats(tenferro_runtime::ExtensionCacheSelector::All);
    backend.restore_extension_caches(store);
    json!({"entries":stats.entries,"retained_bytes":stats.retained_bytes,
           "hits":stats.hits,"misses":stats.misses,"evictions":stats.evictions,
           "clears":stats.clears})
}

/// Public counters of one untimed observation.
///
/// tenferro-gpu v0.8.0's kernel-catalogue cutover removed the cuTENSOR plan
/// cache, permutation-plan cache, vendor workspace and workspace-retirement
/// counters from the engine: a plan cache is now private to the engine's
/// cuTENSOR handle, has no public capacity knob and is not reachable from a
/// downstream crate. The public counters below replace them: the runtime's
/// device/pool/pending accounting, its artifact and module caches, and the
/// extension-owned cache store.
pub(super) fn snapshot(backend: &mut CudaEngine) -> Result<Value> {
    let rt = backend.runtime();
    let (free, total) = rt.with_current_context(
        "benchmark.counter_memory",
        cudarc::driver::result::mem_get_info,
    )??;
    let stats = rt.stats();
    let extensions = extension_cache(backend);
    let rt = backend.runtime();
    Ok(json!({"driver_free_bytes":free,"driver_total_bytes":total,
        "cuda_extensions":extensions,
        "device":{"live_allocations":stats.live_allocations,"live_bytes":stats.live_bytes,
            "pool_used_bytes":stats.pool_used_bytes,
            "pool_reserved_bytes":stats.pool_reserved_bytes,
            "idle_pinned_bytes":stats.idle_pinned_bytes},
        "retirement":{"pending_bytes":stats.retirement.pending_bytes,
            "module_bytes":stats.retirement.module_bytes,
            "pending_records":stats.retirement.pending_records,
            "epochs":stats.retirement.epochs,"frontiers":stats.retirement.frontiers,
            "reclaimed_epochs":stats.retirement.reclaimed_epochs,
            "launches":stats.retirement.launches,"allocations":stats.retirement.allocations,
            "zero_fills":stats.retirement.zero_fills,
            "events_recorded":stats.retirement.events_recorded},
        "artifact_cache":{"entries":stats.artifact_entries,"bytes":stats.artifact_bytes},
        "module_cache":{"entries":stats.module_entries,"bytes":stats.module_bytes},
        "compile_attempts":rt.compile_attempts(),"module_loads":rt.module_loads(),
        "artifact_cache_hits":rt.artifact_cache_hits(),
        "pending_transfer_retirements":rt.pending_transfer_retirements(),
        "output_pool":{"status":"unavailable","reason":"No public output-pool stats/clear API; driver free memory covers the whole device, not just that pool"}}))
}

pub(super) fn run(root: &Path, cases: &[Case], path: &Path, device: usize) -> Result<()> {
    let batches = super::env_usize("GPU_LIFETIME_BATCHES", 30)?;
    let operations = super::env_usize("GPU_LIFETIME_OPS", 128)?;
    if batches == 0 || operations == 0 {
        return Err("counter counts must be positive".into());
    }
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    for case in cases {
        if case.op != "compare" {
            return Err("lifetime pass currently requires compare fixtures".into());
        }
        let expected = super::read_array(root, &case.outputs[0])?;
        let bytes = expected.shape().iter().product::<usize>();
        if operations
            .checked_mul(bytes)
            .ok_or("counter memory overflow")?
            > 256 << 20
        {
            return Err("counter batch exceeds logical retention bound".into());
        }
        let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(device.try_into()?))?;
        let rt = backend.runtime().clone();
        let inputs = case
            .inputs
            .iter()
            .map(|spec| {
                Ok(tenferro_gpu::cuda::upload_tensor(
                    &rt,
                    &super::read_array(root, spec)?,
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut executor = FftExecutor::default();
        backend.with_backend_session(|session| -> Result<()> {
            rt.synchronize()?;
            writeln!(file,"{}",json!({"schema_version":1,"case_id":case.id,
                "status":"instrumentation_only","timing":null,
                "measurement_scope":"separate_untimed_public_counter_pass",
                "gpu_role":"debug","physical_device_index":super::env_usize("GPU_BENCH_PHYSICAL_DEVICE",1)?,
                "device_uuid":std::env::var("GPU_BENCH_DEVICE_UUID").ok(),
                "session_policy":"one entered session per case; process-global allocator state may survive cases",
                "batches":batches,"operations_per_batch":operations,"logical_output_bytes":bytes,
                "phase":"before","snapshot":snapshot(session)?}))?;
            for batch in 0..batches {
                let mut outputs = Vec::with_capacity(operations);
                for _ in 0..operations {
                    outputs.push(super::direct(case,&inputs,&mut executor,session,
                        super::structural::prepare_direct(case,&inputs))?);
                }
                let queued = snapshot(session)?;
                rt.synchronize()?;
                let completed = snapshot(session)?;
                let error = super::verify(&rt,&outputs[0],&expected,case)?
                    .max(super::verify(&rt,&outputs[operations-1],&expected,case)?);
                drop(outputs);
                rt.synchronize()?;
                writeln!(file,"{}",json!({"schema_version":1,"case_id":case.id,
                    "status":"instrumentation_only","timing":null,
                    "measurement_scope":"separate_untimed_public_counter_pass",
                    "phase":"batch","batch":batch,"operations_per_batch":operations,
                    "max_error_over_tolerance":error,"after_enqueue":queued,
                    "after_completion":completed,"after_drop_and_barrier":snapshot(session)?}))?;
                file.flush()?;
            }
            Ok(())
        })??;
    }
    Ok(())
}
