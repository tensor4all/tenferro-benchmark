//! Mechanism diagnostics for the `cpu/session_matrix` batched routes
//! (tenferro-rs #1946 B3). Never a timing run.
//!
//! For each case ID it installs a provider spy (a `CpuGemmProvider` +
//! `CpuUninitGemmProvider` that forwards unchanged to faer, the public
//! provider-bundle mechanism used by tenferro-cpu's own route-parity test),
//! runs the case's public call once after one untracked warm call, and
//! prints one JSON object per case with:
//!
//! - provider calls: initialized vs uninitialized entry, batch count per call,
//!   whether the call ran on an outer fan-out lane;
//! - effective policy (backend default or scoped override) and worker count;
//! - process-wide allocation count/bytes during the call, from a counting
//!   global allocator in this binary (so they include tenferro-internal
//!   allocations on every thread and are only observable here, not in timing
//!   runs);
//! - counters with no public observation point, reported as the literal
//!   string "unavailable", never 0.
//!
//! The route-contract verdict (allocating vs `_into` twins honoring the same
//! lane decision) is computed from these rows by `scripts/route_contract.py`.
//!
//! This binary uses only APIs present both at the #1946 audit baseline
//! (5a4e7fd84) and in the repair, so the same harness shows the baseline
//! failure and the repaired behavior.
#[path = "../bench_support/batch_route.rs"]
mod batch_route;

use std::alloc::{GlobalAlloc, Layout, System};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use batch_route::*;
use tenferro_cpu::provider::{
    CpuGemmProvider, CpuGemmRequest, CpuGemmUninitRequest, CpuGroupedGemmRequest,
    CpuProviderOutcome, CpuUninitGemmProvider, FaerGemmProvider, StridedLayoutTransformProvider,
};
use tenferro_cpu::{CpuExecutionContext, CpuProviderBundle, CpuProviderExecutionCapabilities};
use tenferro_tensor::{BackendSessionHost, Tensor};

struct CountingAllocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards every request unchanged to the system allocator and only
// updates relaxed atomic counters.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        // SAFETY: the caller's layout contract is forwarded unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        // SAFETY: forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Debug)]
struct Call {
    uninitialized: bool,
    lane: bool,
    batch: usize,
    grouped: bool,
}

/// Records provider calls into a preallocated buffer, so recording itself does
/// not allocate while the allocation counter is active.
#[derive(Debug)]
struct Spy(Mutex<Vec<Call>>);

impl Spy {
    fn new() -> Self {
        Self(Mutex::new(Vec::with_capacity(1 << 16)))
    }

    fn record(&self, call: Call) {
        let mut calls = self.0.lock().unwrap();
        if calls.len() < calls.capacity() {
            calls.push(call);
        }
    }

    fn drain(&self) -> Vec<Call> {
        let mut calls = self.0.lock().unwrap();
        let taken = calls.clone();
        calls.clear();
        taken
    }
}

impl CpuGemmProvider for Spy {
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
        FaerGemmProvider.execution_capabilities()
    }

    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.record(Call {
            uninitialized: false,
            lane: context.is_outer_fan_out_lane(),
            batch: 1,
            grouped: false,
        });
        FaerGemmProvider.gemm(context, request)
    }

    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.record(Call {
            uninitialized: false,
            lane: context.is_outer_fan_out_lane(),
            batch: request.batch_count(),
            grouped: false,
        });
        FaerGemmProvider.strided_batched_gemm(context, request)
    }

    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.record(Call {
            uninitialized: false,
            lane: context.is_outer_fan_out_lane(),
            batch: 0,
            grouped: true,
        });
        FaerGemmProvider.grouped_gemm(context, request)
    }

    fn uninit_provider(&self) -> Option<&dyn CpuUninitGemmProvider> {
        Some(self)
    }
}

// SAFETY: forwards the full-overwrite contract unchanged to the standard
// provider, which implements it.
unsafe impl CpuUninitGemmProvider for Spy {
    unsafe fn gemm_into_uninit(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmUninitRequest<'_, '_>,
        output: &mut [MaybeUninit<u8>],
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.record(Call {
            uninitialized: true,
            lane: context.is_outer_fan_out_lane(),
            batch: request.batch_count(),
            grouped: false,
        });
        // SAFETY: the caller's contract is forwarded unchanged.
        unsafe { FaerGemmProvider.gemm_into_uninit(context, request, output) }
    }
}

const UNAVAILABLE: &str = "unavailable";

fn diagnose(spec: &CaseSpec, threads: usize) -> Result<serde_json::Value, BoxError> {
    let mut row = serde_json::json!({
        "case_id": spec.id, "workload": spec.workload, "operation": spec.operation,
        "dtype": spec.dtype, "batch": spec.batch, "m": spec.m, "n": spec.n, "k": spec.k,
        "layout": spec.layout, "output": spec.output, "pair": spec.pair,
        "policy_request": {"strategy": spec.policy.strategy, "override": spec.policy.scope,
                           "thresholds": spec.policy.thresholds},
        "route": spec.route, "requested_threads": threads, "provider": "faer (spy-forwarded)",
        "lane_cost_policy_api": LANE_COST_POLICY_API,
        "measurement_kind": "mechanism_diagnostic",
        "counters": {
            "session_entries": UNAVAILABLE,
            "executor_entries": UNAVAILABLE,
            "zero_fill_bytes": UNAVAILABLE,
            "copy_bytes": UNAVAILABLE,
            "output_pool_reuse": UNAVAILABLE,
        },
        "counter_notes": "provider calls via the public CpuProviderBundle spy; allocations via \
            a counting global allocator in this process (all threads, the one measured call); \
            session/executor entry, zero-fill/copy volume and output-pool reuse have no \
            public observation point",
    });
    let scope = match build_policy(&spec.policy) {
        Ok(scope) => scope,
        Err(reason) => {
            row["status"] = "unsupported".into();
            row["unsupported_reason"] = reason.into();
            return Ok(row);
        }
    };
    let spy = Arc::new(Spy::new());
    let bundle = CpuProviderBundle::custom_builder()
        .gemm_provider(spy.clone())
        .layout_transform_provider(Arc::new(StridedLayoutTransformProvider))
        .build()?;
    let mut backend = backend_for(threads, scope, Some(bundle))?;
    let info = backend.execution_info();
    row["worker_count"] = info.worker_count().into();
    row["execution_mode"] = format!("{:?}", info.execution_mode()).into();
    row["effective_policy"] = match scope {
        PolicyScope::Scoped(p) => format!("scoped:{p:?}"),
        _ => format!("backend:{:?}", backend.batch_policy()),
    }
    .into();
    let fixture = Fixture::new(spec)?;
    let reads = fixture.reads()?;
    let owned = fixture.owned();
    let mut destination = fixture.destination()?;
    type Observed = (
        Result<Option<Tensor>, OpError>,
        Vec<Call>,
        Option<(u64, u64)>,
    );
    let observed: Observed = backend.with_backend_session(|session| {
        in_policy_scope(session, scope, |session| -> Observed {
            let call = |session: &mut dyn tenferro_tensor::BackendSession,
                        destination: &mut Tensor| {
                if spec.is_into() {
                    run_into(session, spec, &fixture, &reads, &owned, destination).map(|()| None)
                } else {
                    run_alloc(session, spec, &fixture, &reads, &owned).map(Some)
                }
            };
            // One untracked warm call so one-time plan/cache setup is not
            // attributed to the observed call.
            let warm = call(session, &mut destination);
            let _ = spy.drain();
            if warm.is_err() {
                return (warm, Vec::new(), None);
            }
            drop(warm);
            let (a0, b0) = (
                ALLOCATIONS.load(Ordering::SeqCst),
                ALLOCATED_BYTES.load(Ordering::SeqCst),
            );
            COUNTING.store(true, Ordering::SeqCst);
            let result = call(session, &mut destination);
            COUNTING.store(false, Ordering::SeqCst);
            let (a1, b1) = (
                ALLOCATIONS.load(Ordering::SeqCst),
                ALLOCATED_BYTES.load(Ordering::SeqCst),
            );
            (result, spy.drain(), Some((a1 - a0, b1 - b0)))
        })
    })??;
    let (result, calls, allocations) = observed;
    match result {
        Err(OpError::Unsupported(reason)) => {
            row["status"] = "unsupported".into();
            row["unsupported_reason"] = reason.into();
        }
        Err(OpError::Failed(reason)) => {
            row["status"] = "failed".into();
            row["error"] = reason.into();
        }
        Ok(output) => {
            let checked = match &output {
                Some(tensor) => fixture.check(tensor),
                None => fixture.check(&destination),
            };
            match checked {
                Ok(err) => {
                    row["status"] = "observed".into();
                    row["numerical_check"] = "passed".into();
                    row["max_abs_error"] = err.into();
                }
                Err(e) => {
                    row["status"] = "failed".into();
                    row["numerical_check"] = "failed".into();
                    row["error"] = e.to_string().into();
                }
            }
        }
    }
    let provider_calls: Vec<_> = calls
        .iter()
        .map(|c| {
            serde_json::json!({
                "entry": if c.grouped { "grouped" } else if c.uninitialized { "uninitialized" } else { "initialized" },
                "outer_fan_out_lane": c.lane, "batch_count": c.batch,
            })
        })
        .collect();
    row["provider_calls"] = serde_json::json!(provider_calls);
    row["provider_call_count"] = calls.len().into();
    row["initialized_calls"] = calls.iter().filter(|c| !c.uninitialized).count().into();
    row["uninitialized_calls"] = calls.iter().filter(|c| c.uninitialized).count().into();
    row["lane_calls"] = calls.iter().filter(|c| c.lane).count().into();
    row["lane_used"] = calls.iter().any(|c| c.lane).into();
    row["batch_total"] = calls.iter().map(|c| c.batch).sum::<usize>().into();
    row["batch_sizes"] = serde_json::json!(calls.iter().map(|c| c.batch).collect::<Vec<_>>());
    // No observed call (the warm call already failed): nothing was counted.
    let (count, bytes) = match allocations {
        Some((count, bytes)) => (count.into(), bytes.into()),
        None => (UNAVAILABLE.into(), UNAVAILABLE.into()),
    };
    row["counters"]["allocations"] = count;
    row["counters"]["allocated_bytes"] = bytes;
    Ok(row)
}

fn main() -> Result<(), BoxError> {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str, default: &str| {
        args.windows(2)
            .find(|p| p[0] == name)
            .map(|p| p[1].clone())
            .unwrap_or(default.into())
    };
    let threads: usize = get("--threads", "1").parse()?;
    let instances = get("--instances", "data/instances/session_matrix.json");
    let selected = get("--cases", "");
    tenferro_einsum_benchmark::thread_enforcement::enforce_thread_request(threads)?;
    let cases = load_cases(&instances)?;
    let wanted: Vec<&str> = selected.split(',').filter(|s| !s.is_empty()).collect();
    for id in &wanted {
        if !cases.iter().any(|c| c.id == *id) {
            return Err(format!("unknown case ID {id}").into());
        }
    }
    for spec in cases
        .iter()
        .filter(|c| wanted.is_empty() || wanted.contains(&c.id.as_str()))
    {
        let row = diagnose(spec, threads).unwrap_or_else(|e| {
            serde_json::json!({"case_id": spec.id, "requested_threads": threads,
                               "pair": spec.pair, "output": spec.output,
                               "status": "failed", "error": e.to_string()})
        });
        println!("{row}");
    }
    Ok(())
}
