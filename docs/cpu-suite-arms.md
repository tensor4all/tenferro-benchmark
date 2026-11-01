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
| `cpu/cpu_ops` | `tenferro-eager, tenferro-trace, pytorch-cpu, jax-cpu` | zen5 657 / amd 148 / mac 148; 18 cases have both arms, **10 have trace only** | **partial by design** (see below) |
| `cpu/linalg_jvp_vjp` (derived from the CPU-ops CSV) | — (not a suite file; `render_run.py` selects the `_jvp`/`_vjp` rows) | **0** eager, 240 trace (zen5) / 100 (amd, mac) | **not measured by design** |
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

## Why `cpu_ops` and `linalg_jvp_vjp` are trace-only for JVP/VJP

`src/bin/publication_gate.rs` registers the AD cases through
`push_linalg_jvp_vjp_trace_benches` (called from `run_all` at the `include_trace` branches, lines
~1238-1487). There is no eager registration for `grad_sum_{eigh,lu,qr,solve,svd_s}_{jvp,vjp}`:
the eager runtime has no reverse-mode `AdContext` path for them, so the cases are declared on the
trace arm only. `run_cpu_ops.sh` states the same split in its header ("primal linalg, trace-mode
JVP/VJP (tenferro-trace), and eager backward").

`linalg_jvp_vjp` is not a suite declaration: `scripts/render_run.py` derives that page from the CPU-ops
CSV rows whose benchmark matches `_jvp`/`_vjp`, so it inherits exactly the ten trace-only cases. A
page with no eager column is therefore the expected output, and the "eager" arm for those cases
must not be described as missing data.

The `_backward` cases (`grad_sum_matmul_backward`, `grad_sum_*_backward`) are the eager arm's
counterpart and are measured on eager; they appear on the `cpu_ops` page.

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
