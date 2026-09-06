# Small-work thread evidence

The producer already calls CpuBackend::new. This resolves its budget from
RAYON_NUM_THREADS or available parallelism, and can create backend-pinned workers.
It is not an explicit-unpinned-budget constructor. A4T correctness reproduction
showed the old equality check rejecting a legitimate worker mask[0] within the
outer mask[0,1,2,3].

Keep exact equality for the outer process mask. Each observed worker must have a
nonempty mask contained in that same selected set; backend narrowing is allowed,
escape is not. This is confinement evidence at observation points, not exclusive
reservation, proof of all-time affinity, or authority to change backend domains.

The caller chooses required observation stages: correctness-only checks
post-correctness, while timing still requires post-correctness, post-warmup and
post-timing. Never infer this relaxation from the child's reported timing status
or manufacture observations for phases that did not run.

Record effective_thread_budget from CpuBackend::num_threads before moving the
backend into the runtime, plus the unchanged constructor and RAYON_NUM_THREADS
input. The requested declaration must match the runner's request; timing also
requires effective budget equality. Correctness-only can report a different
actual budget when no eligible placement exists, without pretending it is a1T
performance result. OS task count is not the backend budget.

Also record `backend_execution_mode` from
`backend.execution_info().execution_mode()` before moving the backend. This is
observed diagnostic information, not a new acceptance gate: constructor spelling
alone does not prove the resolved mode, since default construction can fall back.
Do not require `Managed` indiscriminately: BLAS may report
`ProviderDefaultExclusive`.

For the future explicit budget-one control, `CpuBackend::with_threads(1)` is NOT
an unpinned alias on Linux: it uses the managed construction path. The existing
`CpuBackend::from_context(Arc::new(CpuContext::with_threads(1)?))` uses a context
with no pinned CPUs and no Rayon pool at budget one. Its backend mode is
`Compatibility` for native/Faer, or `ProviderDefaultExclusive` for BLAS. Outer
harness affinity remains a separate constraint. This control is not yet wired
into the runner; do not relabel default records as that control.

Do not change library constructors, placement/admission, default thread policy,
worker pools or provider behavior. Existing1T/default correctness remains valid;
new4T correctness proves the corrected observation path, not performance gains.
A separately identified explicit-unpinned control and valid paired measurements
remain required by #1762/#96.
