# Which CPU suites measure an eager arm

The CPU dashboard draws a `tenferro-rs eager mode` bar beside the trace and reference bars for
every case it finds. Where the eager bar is absent, that is a property of the **measurement**, not
of the rendering: no eager row was recorded. This page is the one place that says, per suite, why.
A case whose eager arm is missing here is not a lost measurement and does not need chasing.

Evidence for every row is one of three things: the suite declaration in `benchmarks/cpu/<suite>.yaml`,
the registration in `src/bin/publication_gate.rs` / `src/bin/benchmark_cpu_public_api.rs`, or the
backend set of a committed `result/<profile>/cpu/<suite>.summary.json`.

## Matrix

| suite (page) | declared backends (`*.yaml`) | eager rows in the committed summaries | eager status |
|---|---|---|---|
| `cpu/einsum` | `tenferro-trace, tenferro-eager, pytorch-cpu, jax-cpu, omeinsum-jl` | zen5 144 / amd 96 / mac 96 (every case) | **measured** |
| `cpu/cpu_ops` | `tenferro-eager, tenferro-trace, pytorch-cpu, jax-cpu` | zen5 657 / amd 248 / mac 248; amd and mac have both arms on every case (28/28), zen5 has 18 and **10 trace only** (the JVP/VJP cases, which its last collection predates) | **partial by design** (see below) |
| `cpu/linalg_jvp_vjp` (derived from the CPU-ops CSV) | — (not a suite file; `render_run.py` selects the `_jvp`/`_vjp` rows) | zen5 **0** eager / 240 trace; amd and mac **100 / 100** (all ten cases on both arms) | **measured** on amd and mac; zen5 awaits its next collection |
| `cpu/public_api` | `tenferro-direct, tenferro-eager, tenferro-trace, pytorch-cpu, jax-cpu, julia-base, strided-jl` | zen5 117 / amd 78 / mac 78; 22 cases have both arms, **96 have trace only** | **partial by design** |
| `cpu/permutation` | `tenferro-rs, hptt, strided-rs, julia-base, strided-jl, memcpy` | 0 (no `tenferro-eager`/`-trace` arm exists) | **not applicable**: the axis is different arms |
| `cpu/fft` | `tenferro-fft-immediate, tenferro-fft-executor-cached, pytorch-cpu` | 0 | **not applicable** |
| `cpu/session_matrix` | `tenferro-blas-shared-session, tenferro-faer-shared-session, pytorch-python-loop` | 0 | **not applicable** |
| `cpu/small_work`, `cpu/overhead_smoke` | `tenferro-rs` | 0 | **not applicable** |
| `cpu/perf_issues` | per-case references (`benchmarks/cpu/perf_issues.yaml`) | 0 | **not applicable** |

The eager/trace axis exists only in `cpu/einsum`, `cpu/cpu_ops` (+ its derived `linalg_jvp_vjp`
page) and `cpu/public_api`. In the other six the "missing eager bar" is not a missing arm: those
suite pages measure other arms (`tenferro-rs`, `tenferro-fft-*`, `*-shared-session`, or a per-case
reference), and the dashboard's default eager+trace rule does not describe them.

## Why `linalg_jvp_vjp` gained an eager arm

`src/bin/publication_gate.rs` registered the AD cases only through
`push_linalg_jvp_vjp_trace_benches`. That was a **registration gap, not a capability gap**: every op
the ten cases differentiate (`svd`, `qr`, `eigh`, `lu`, `solve`) has an eager entry point in
`tenferro-linalg` (`EagerSessionLinalgExt`), the derivatives come from `EagerRuntime::jvp` / `vjp`,
and `tenferro-linalg`'s own integration tests assert gradients on that surface. Since
tenferro-benchmark #307 the same ten case ids are also registered on the eager arm
(`push_linalg_jvp_vjp_eager_benches`, called from the eager drivers `run_small_latency` and
`run_large_throughput`, the latter on the `large_linalg_jvp_vjp_sizes` grid the trace arm uses), so
both arms of `cpu/linalg_jvp_vjp` come from the same CPU-ops CSV rows.
Sharing the ids matters: `render_run.py` derives that page from the rows named `_jvp`/`_vjp`, so a
renamed eager case would add no arm.

Eager AD samples cannot share one borrowed session the way the primal rows do:
`EagerRuntime::jvp`/`vjp` take the caller-held engine themselves (tenferro-rs #2206). One eager AD
sample is therefore one forward pass in a borrowed session plus one `jvp`/`vjp` call, which is the
eager counterpart of the trace arm's forward-and-derivative program run. Fixtures and tensor
wrapping stay outside the clock, as for every other suite.

The eager call also carries the AD preparation it performs itself (the eager VJP caches its prepared
derivative, the eager JVP re-prepares per call), because the public eager API exposes no separate
prepare step to move outside the clock; the trace arm compiles and prepares outside it. Read the
eager number as what a program pays per `jvp`/`vjp` call, not as a prepared-program steady state.
`scripts/format_linalg_ad_results.py` states both facts beside the table and heads the eager column
`tenferro-rs eager per call, incl. AD prepare (ms)`, so the two columns are not read as the same
measurement; `scripts/ci/tests/test_linalg_ad_backends.py` keeps that note and that heading.
The raw sample record says it too: an eager AD row writes
`eager_session_policy: forward_and_derivative_inside_timing_with_eager_ad_prepare`, where a primal
row writes `borrowed_session_outside_timing_for_primal`. The row is therefore self-describing
rather than setup-inclusive by accident (`src/bin/publication_gate.rs`).

The amd-cpu and mac-cpu summaries now carry all ten eager JVP/VJP cases (100 rows each); they were
collected after the registration, and `result/amd-cpu/cpu/linalg_jvp_vjp.md` and its mac counterpart
show the column. The zen5-cpu summary still has none, because its last collection predates #311, so
its next collection is the one that fills it. This page states no eager-vs-trace numbers of its own.

The `_backward` cases (`grad_sum_matmul_backward`, `grad_sum_*_backward`) are the eager arm's
reverse-mode counterpart. They record `unsupported: eager backward has no borrowed-session API`:
since tenferro-rs #2206 the reverse pass needs the engine that the entered session already borrows,
so those cases are declared but not timed.

## Why `public_api` has eager for 22 cases and direct for the rest

`src/bin/benchmark_cpu_public_api.rs` has separate execution filters. The direct/API-time route
labels its rows `tenferro-direct`, except `lstsq` and `svd_full`, which are labelled
`tenferro-eager` (lines ~1022 and ~1044). The eager filter registers a small set of ops
(`erf`, `gelu`, `gelu_tanh`, `householder_qr`, `layer_norm`, `log_softmax`, `lstsq`,
`masked_log_softmax_axis1`, `masked_softmax`, `rank_revealing_qr`, `rank_revealing_qr_deficient`,
`reduce_mean`, ...). The other 96 ops (views, `abs`, `add`, `batched_*`, `cast_*`, ...) have only
a direct/API-time route, so their eager bar cannot exist.

## Rule for the reader and for the dashboard

1. An absent eager bar means "this suite or case declares no eager arm". It never means "the run
   dropped a measurement" — a dropped measurement is a defect and would be visible as a row with a
   `status` other than `ok`, not as a missing backend.
2. When eager *is* declared and a case still lacks it, that is a bug: report it against the suite,
   not against the page.
3. `scripts/format_linalg_ad_results.py` keeps every backend it is given (declared order first, then
   any other), so a future eager measurement for JVP/VJP renders instead of being silently dropped.
   That behaviour is covered by `scripts/ci/tests/test_linalg_ad_backends.py`.
