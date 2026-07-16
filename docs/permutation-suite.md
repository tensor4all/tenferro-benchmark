# CPU Permutation Suite Specification

Status: implemented. This document specifies the suite, ported from
[`extern/strided-rs-benchmark-suite`](../extern/strided-rs-benchmark-suite/)'s
permutation benchmark into the tenferro-benchmark schema and result policy.

## Purpose

Measure the cost of materializing a strided/permuted `f64` tensor view into a
contiguous column-major destination, and compare tenferro-rs against
established transpose/permutation implementations on identical semantics:

- naive baseline (Rust odometer loop)
- tenferro-rs transpose/materialize paths
- HPTT
- Julia Base (`permutedims!` / generic `copyto!`)
- Strided.jl (`@strided` materialization)
- strided-rs `strided_perm` (optional reference column)

This is a **materialize/copy kernel benchmark**, not an einsum benchmark. The
measured object is the copy kernel that turns a permuted view into contiguous
output, so the suite is kept separate from `cpu/einsum` and its
trace/eager/planner comparisons.

## Scope

In scope:

- Forward materialization of `transpose` / `permute` on dense `f64` tensors.
- Copy from a strided source layout (permuted col-major or explicit strides)
  into a compact col-major destination.
- Same-semantics comparison against HPTT, Julia Base, and Strided.jl with
  correctness verification before timing.

Out of scope:

- AD in any form: trace graphs, linearize, transpose rules,
  backward/gradient, JVP/VJP. This suite never touches the trace machinery.
- Einsum/contraction planner comparisons (`cpu/einsum` covers those).
- GPU permutation (possible future `gpu/permutation` suite).
- Row-major destinations. All destinations are compact col-major, matching
  tenferro's layout convention.

## Suite Identity and Result Policy

Following [Result Layout and Metadata](results.md) and AGENTS.md:

- `suite_id`: `cpu/permutation`. Workload identity only; no hardware name.
- Suite definition: `benchmarks/cpu/permutation.yaml`, valid against
  `schemas/benchmark-suite.schema.json` and checked by
  `scripts/validate_benchmark_suite.py`.
- Pattern data: `data/instances/permutation_patterns.json` — the single
  source of truth for benchmark patterns, read by both the Rust and Julia
  runners. Ported from
  `extern/strided-rs-benchmark-suite/benchmarks/strided_benchmarks/permute/patterns.json`.
- Raw runs: `data/results/<target_profile>/cpu/permutation/<timestamp>/`
  with the standard `run.yaml` metadata.
- Latest reports (tracked, overwritten per profile):
  - `result/mac-cpu/cpu/permutation.md`
  - `result/amd-cpu/cpu/permutation.md`
- No result tables in README.md or this document. Git history serves as the
  archive of older reports.

## Pattern Schema

`data/instances/permutation_patterns.json` keeps the source schema:

- top-level: `version`, `index_base: 0`,
  `semantics: "out[i0,...,ik] = src[i_perm0,...,i_permk]"`,
  `data: "deterministic_index_value"` (no random seed).
- per pattern: `id`, `label`, `dtype` (`f64`), `shape`, `perm` (0-indexed),
  `src_layout` (`col_major` or explicit strides), `dst_layout` (`col_major`),
  `participants`.

`participants` is the per-pattern allowlist of backend columns. It is the
mechanism that keeps comparisons honest: a backend appears in a pattern's row
only when it can express exactly the pattern's semantics (see
[HPTT eligibility](#hptt-eligibility)).

### Initial Pattern Set

Ported unchanged from the source suite:

| id | description |
|---|---|
| `memcpy_24d_contiguous` | memcpy baseline, 24D 2^24 identity permutation |
| `transpose_2d_256` / `_1024` / `_2048` | 2D square transpose `[1,0]` |
| `transpose_3d_256_201` | 3D 256^3 permutation `[2,0,1]` |
| `transpose_3d_256_102` | 3D 256^3 permutation `[1,0,2]` |
| `rotation_6d_32_32_32_32_16_16` | 6D rotation `[5,0,4,1,3,2]` |
| `reverse_20d_2` | 20D 2^20 axis reversal |
| `reverse_13d_2` | 13D 2^13 axis reversal |
| `cyclic_13d_2` | 13D 2^13 cyclic shift |
| `tn_light_415_24d_scattered_to_colmajor` | 24D explicit scattered source strides → col-major |
| `tn_light_415_24d_contiguous_same_perm` | same 24D permutation, contiguous source |

The scattered/contiguous 24D pair is derived from the
`tensornetwork_permutation_light_415` einsum instance and links this suite to
the permutation-heavy cases observed in `cpu/einsum`.

## Backends

Backend column names follow [architecture terminology](architecture.md).

| backend | runner | measured path |
|---|---|---|
| `naive` | Rust | odometer index loop, baseline and correctness reference |
| `tenferro-transpose` | Rust | eager op `tenferro_cpu::structural::transpose` (`TensorStructural::transpose` on `CpuBackend`); internally `StridedView::permute` + `strided_kernel::copy_into` |
| `tenferro-to-contiguous` | Rust | view API `TypedTensorView::transpose_view(perm)` followed by `to_contiguous()` |
| `hptt` | Rust (`hptt` crate, feature-gated) | HPTT tensor transpose, contiguous cases only |
| `strided-rs` (Rust `strided-rs` feature, on by default) | Rust | `strided_perm::copy_into` / `copy_into_col_major`, serial and parallel (`copy_into_par` / `copy_into_col_major_par`); the fastest of the (up to four) variants is reported |
| `julia-base` | Julia | `permutedims!` for contiguous sources, generic `copyto!` for explicit-stride sources |
| `strided-jl` | Julia | Strided.jl `@strided` materialization |

Notes:

- Neither tenferro column goes through trace, einsum, or AD machinery.
  `tenferro-transpose` is the eager structural op; `tenferro-to-contiguous` is
  the user-facing lazy-view-then-materialize path. Both are reported so the
  wrapper overhead over the shared `strided_kernel::copy_into` kernel is
  visible.
- `strided-rs` is kept as a reference column because tenferro's CPU kernel is
  built on the same `strided-kernel` family; a divergence between
  `tenferro-transpose` and `strided-rs` on the same pattern indicates wrapper
  overhead, not kernel regression. It depends on `extern/strided-rs`'s
  `strided-perm` / `strided-view` crates as optional path dependencies (Cargo
  feature `strided-rs`, on by default in `scripts/run_permutation.sh`);
  `scripts/setup_extern_deps.sh` clones `extern/strided-rs` unconditionally
  (alongside `extern/tenferro-rs`) because Cargo resolves every path
  dependency's manifest to build the unit graph even when the feature
  enabling it is disabled, so a checkout without `extern/strided-rs` would
  fail `cargo build` regardless of `--features strided-rs`.
- LibTorch/PyTorch is intentionally not part of this suite; the CPU backend
  policy comparison set for einsum does not apply to this copy-kernel suite.

### Allocation Semantics

All backend columns measure **materialization including destination reuse,
excluding destination allocation**: the destination buffer is allocated once
per pattern outside the timed region, and each timed iteration overwrites it.
If a tenferro API forces an allocation per call (e.g. `to_contiguous`
returning a new owned tensor), the report must say so in a footnote for that
column, because it is then not directly comparable to in-place columns on
small patterns.

### Correctness Gate

Before timing, every participant's output is compared elementwise against the
`naive` reference for the pattern's deterministic input. A mismatch fails the
run for that pattern; incorrect implementations are never timed.

### HPTT Eligibility

The public HPTT API cannot express arbitrary source strides. HPTT is listed
in `participants` only for patterns where both source and destination are
contiguous (col-major) layouts. Explicit-stride patterns such as
`tn_light_415_24d_scattered_to_colmajor` intentionally exclude HPTT; reports
render `-` for non-participating backends rather than reusing a different
semantics.

## Runner Design

One runner per implementation family, all consuming
`data/instances/permutation_patterns.json`:

- Rust: `src/bin/benchmark_permutation.rs`, ported from the source suite's
  `permute.rs` with the two tenferro backends added. A Cargo feature gates
  `hptt` (requires cmake and a C++ toolchain; builds on both macOS and Linux;
  not on by default, opt in via `PERMUTATION_EXTRA_FEATURES=hptt`). Another
  Cargo feature, `strided-rs` (on by default in `scripts/run_permutation.sh`),
  gates the `strided-rs` participant, which builds a `strided_view::StridedArray`
  from the pattern's source strides and times `strided_perm::copy_into` /
  `copy_into_col_major` plus their `_par` counterparts (via strided-perm's own
  `parallel` feature, always enabled alongside `strided-rs`), reporting the
  fastest variant. Thread-count reporting always uses
  `rayon::current_num_threads()`, not gated behind any feature.
- Julia: `scripts/benchmark_permutation.jl`, ported from `permute.jl`.
  Strided.jl becomes a dependency of the repository Julia project.
- Formatting: `scripts/format_permutation_results.py` aggregates per-runner
  JSON outputs from `data/results/.../cpu/permutation/<timestamp>/` into the
  latest report, following the existing `format_*_results.py` scripts.
- Orchestration: `scripts/run_permutation.sh <NUM_THREADS...>` runs the Rust
  and Julia runners sequentially for one or more thread counts (looped
  sequentially within one run directory/timestamp so the latest report has
  one section per thread count), records `run.yaml` via
  `scripts/collect_run_metadata.py`, and regenerates
  `result/<target_profile>/cpu/permutation.md`. `scripts/run_all.sh` gains an
  opt-in hook (or the script is documented as a standalone entry point) so
  the latest report is reproducible from one command.

Environment variables follow existing conventions: `BENCHMARK_TARGET_PROFILE`,
`BENCH_RUNS`, `BENCH_WARMUPS`, and `PATTERN_ID` to run a single pattern while
investigating.

## Threading and Measurement Policy

- Thread-count variants (1 thread and 4 threads) are measured **sequentially,
  in separate runs**. Benchmark processes are never run concurrently, per
  AGENTS.md timing discipline.
- Thread control per runner: `RAYON_NUM_THREADS` (Rust/rayon, including
  tenferro via `CpuBackend::with_threads`), `OMP_NUM_THREADS` (HPTT), and
  `JULIA_NUM_THREADS` (Julia). One report section per thread count; the same
  HPTT thread count as the Rust rayon count.
- Timing: median with p25/p75 over per-pattern iteration counts scaled to the
  pattern size (as in the source suite), reported in milliseconds; bandwidth
  in GB/s may be included as a derived column.
- CPU pinning: on macOS, CPU pinning is unavailable — `run.yaml` and the
  report state this explicitly. On Linux devcontainer runs, follow the
  existing devcontainer conventions and record the pinning policy actually
  used in `run.yaml`.
- `run.yaml` records the standard fields (target profile, suite id, tenferro
  commit/dirty/features, host metadata, controlled thread environment
  variables) plus the pattern file path.

## Fair-Comparison Guards

Supervision checklist, restated as suite requirements:

1. tenferro columns are eager materialize/copy paths only; no trace, no AD.
2. HPTT rows exist only where the public HPTT API expresses the identical
   semantics (contiguous source and destination).
3. Pattern definitions live in one JSON file consumed by all runners; no
   per-runner hardcoded shapes.
4. Correctness is verified against the naive reference before any timing.
5. Input data is deterministic (`deterministic_index_value`); no RNG state in
   the comparison.
6. Allocation semantics are uniform across columns or explicitly footnoted.
7. Cross-language caveats (JIT warmup for Julia, GC) are handled by warmup
   iterations and noted in the report when they affect small patterns.

## Future Work

- `gpu/permutation` suite on CUDA.
- Row-major destination variants if tenferro grows first-class row-major
  output support.
- Coupling with `cpu/einsum` diagnostics: patterns derived from
  permutation-heavy einsum instances can be added when planner analysis
  identifies new hot permutations.
