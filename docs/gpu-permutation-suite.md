# GPU Permutation Suite Specification

Status: implemented. This document defines `gpu/permutation`, the CUDA port
of [`cpu/permutation`](permutation-suite.md). Read that document first; this
one only restates what differs on CUDA.

## Purpose

Measure the cost of materializing a strided/permuted `f64` tensor view into
a contiguous column-major destination **on the GPU**, and compare tenferro-rs
against cuTENSOR and PyTorch CUDA on identical semantics:

- tenferro-rs CUDA transpose/materialize paths
- cuTENSOR (`cutensorPermute`)
- PyTorch CUDA
- a device-to-device `memcpy` bandwidth baseline

This is a **materialize/copy kernel benchmark**, not an einsum or AD
benchmark, exactly like `cpu/permutation`. It stays separate from
`gpu/dense`, `gpu/einsum`, and `gpu/linalg_jvp_vjp`.

## Scope

In scope:

- Forward materialization of `transpose` / `permute` on dense `f64` tensors
  resident on a CUDA device.
- Copy from a strided source layout (permuted col-major or explicit
  strides) into a compact col-major destination.
- Same-semantics comparison against cuTENSOR and PyTorch CUDA,
  with correctness verification before timing.

Out of scope:

- AD in any form: trace graphs, linearize, transpose rules,
  backward/gradient, JVP/VJP. `gpu/linalg_jvp_vjp` covers CUDA AD.
- Einsum/contraction planner comparisons (`gpu/einsum` covers those).
- Row-major destinations. All destinations are compact col-major, matching
  tenferro's layout convention and `cpu/permutation`.
- CPU backends. `cpu/permutation` covers the CPU-only comparison
  (`naive`/HPTT/Julia/strided-rs); none of those participate here.

## Suite Identity and Result Policy

Following [Result Layout and Metadata](results.md) and `AGENTS.md`:

- `suite_id`: `gpu/permutation`. Workload identity only; no hardware name.
- Suite definition: `benchmarks/gpu/permutation.yaml`, valid against
  `schemas/benchmark-suite.schema.json` and checked by
  `scripts/validate_benchmark_suite.py`.
- Pattern data: `data/instances/permutation_patterns.json` -- the same file
  `cpu/permutation` uses. Every pattern carries a `participants_gpu` array
  (parallel to the CPU-only `participants` array) so the two suites share
  one source of truth for shapes/perms/strides/semantics while keeping
  independent backend eligibility lists.
- Raw runs: `data/results/nvidia-gpu/gpu/permutation/<timestamp>/` with the
  standard `run.yaml` metadata.
- Latest report (tracked, overwritten per profile): `result/nvidia-gpu/gpu/permutation.md`.
- No result tables in README.md or this document. Git history serves as the
  archive of older reports.

## Pattern Schema

Unchanged from `cpu/permutation`: `data/instances/permutation_patterns.json`
still has one `version`, `index_base: 0`,
`semantics: "out[i0,...,ik] = src[i_perm0,...,i_permk]"`, and
`data: "deterministic_index_value"` at the top level, with one entry per
pattern (`id`, `label`, `dtype`, `shape`, `perm`, `src_layout`, `dst_layout`).
Two fields are new, and only consumed by this suite's runners:

- `participants_gpu`: the per-pattern allowlist of GPU backend columns,
  exactly like `participants` gates CPU backend columns. A backend appears
  in a pattern's row only when it can express exactly the pattern's
  semantics (see [Participant Eligibility](#participant-eligibility)
  below).
- `notes_gpu` (optional): free-text rationale for a pattern's GPU-specific
  exclusions or risks, parallel to the existing `notes` field.

The pattern set is identical to `cpu/permutation`'s (see that document's
"Pattern Set" table): `memcpy_24d_contiguous`, `transpose_2d_2048`,
`transpose_3d_256_201`, `transpose_3d_256_102`,
`rotation_6d_32_32_32_32_16_16`, `reverse_23d_2`, `reverse_15d_3`,
`cyclic_15d_3`, `tn_light_415_24d_scattered_to_colmajor`, and
`tn_light_415_24d_contiguous_same_perm`.

## Backends

Backend column names follow [architecture terminology](architecture.md).

| backend | runner | measured path |
|---|---|---|
| `tenferro-cuda-transpose` | Rust | eager `TensorStructural::transpose` on `CudaBackend` (compact col-major source only) |
| `tenferro-cuda-to-contiguous` | Rust | `TypedTensor::backend_region_view` (source layout) + `TypedTensorView::transpose_view(perm)` + `TensorViewCanonicalization::to_contiguous` (accepts arbitrary source strides) |
| `cutensor` | Rust | direct cuTENSOR 2.x `cutensorPermute`, dlopen'd from this benchmark repo |
| `pytorch-cuda` | Python | `dst.copy_(src_view)` + `torch.cuda.synchronize()` |
| `memcpy-d2d` | Python (torch) | `dst.copy_(src)` + `torch.cuda.synchronize()`, contiguous identity permutation only |

Notes:

- Neither tenferro column goes through trace, einsum, or AD machinery, same
  as `cpu/permutation`'s policy. `tenferro-cuda-transpose` is the eager
  structural op; `tenferro-cuda-to-contiguous` is the user-facing
  lazy-view-then-materialize path.
- Use `tenferro-cuda-to-contiguous` as the primary tenferro-rs column when
  comparing against framework-level APIs: it matches PyTorch's
  view/`permute`-then-materialize path most closely. Use
  `tenferro-cuda-transpose` when comparing direct structural
  permutation primitives or kernels, especially against cuTENSOR. The report
  retains both columns because they answer these different questions; they
  should not be treated as interchangeable tenferro-rs modes.
- tenferro's own cuTENSOR FFI bindings
  (`extern/tenferro-rs/crates/tenferro-gpu/src/cubecl/ffi/cutensor.rs`) only
  cover contraction (`dot_general`), because that is all the production
  backend needs today. This suite dlopens the same vendor cuTENSOR library
  a second time, from a small self-contained FFI module inside
  `src/bin/benchmark_gpu_permutation.rs` (`mod cutensor_ffi`), binding just
  the permutation entry points (`cutensorCreateTensorDescriptor`,
  `cutensorCreatePermutation`, `cutensorCreatePlanPreference`,
  `cutensorCreatePlan`, `cutensorPermute`). This mirrors the style of
  tenferro's own FFI (dlopen via `libloading`, `TENFERRO_CUTENSOR_PATH`
  search path with the same default fallback path, RAII wrappers with
  `Drop`-based cleanup) without depending on `tenferro-gpu` internals that
  are not part of its public API.
- LibTorch/PyTorch C++ is intentionally not part of this suite; `AGENTS.md`'s
  CPU Backend Policy (no LibTorch) also applies here -- only the Python
  PyTorch wheel is used, as `gpu/dense`/`gpu/einsum`/`gpu/linalg_jvp_vjp`
  already do.

### Column-Major Destination Layout Mapping (PyTorch)

PyTorch tensors are always row-major (C-contiguous) at the storage level;
there is no native column-major dense allocation. A compact column-major
tensor of logical shape `out_shape` has exactly the same physical byte
layout as a C-contiguous tensor of shape `reversed(out_shape)` with its axes
fully reversed. `scripts/benchmark_gpu_permutation_python.py`'s
`run_pytorch` therefore allocates the destination once as
`torch.empty(reversed(out_shape), dtype=torch.float64, device=...)` (a real
allocation) and builds a metadata-only `.permute(*reversed_axes)` view over
it; every timed `copy_` targets that view. The buffer itself is allocated
once per pattern, outside the timed region, and reused across every timed
iteration -- the same allocation semantics as `cpu/permutation`'s non-tenferro
columns. Verification compares the view's *logical* values (via `.cpu()`,
which correctly materializes non-contiguous strided views) against the
host-computed reference; physical byte layout only matters for the timing
measurement, not for correctness.

### Why JAX Is Not Collected

JAX is intentionally excluded from maintained GPU benchmark reports. During
the 2026-07-28 A100 80GB collection, the permutation runner remained inside
the JAX/XLA path for more than 20 minutes while using one host CPU core and
about 61 GiB of device memory, without completing the backend run. Earlier
rank-24 testing had already shown compilation taking more than 40 minutes.
This makes routine full-suite refreshes impractical. JAX also cannot request
the compact column-major destination used by the other participants.

The required framework comparison is therefore tenferro-rs versus PyTorch.
Vendor backends such as cuTENSOR remain optional references. The Python JAX
implementation remains available for ad hoc experiments, but standard
collection scripts and suite backend lists do not invoke it.

### Allocation Semantics

Unlike the uniform allocation-inclusive `cpu/permutation` policy, GPU backend
columns generally measure
**materialization including destination reuse, excluding destination
allocation**, with exceptions called out per column:

- `tenferro-cuda-transpose` and `tenferro-cuda-to-contiguous` both allocate
  a fresh device tensor on every call (their tenferro-rs public APIs return
  owned tensors); this is footnoted in the report, same as the CPU pair.
- `cutensor`, `pytorch-cuda`, and `memcpy-d2d` reuse a destination buffer
  allocated once per pattern outside the timed region.

### Correctness Gate

Before timing, every participant's output is downloaded to the host and
compared elementwise against a host-computed naive reference for the
pattern's deterministic input -- **exactly**, `rtol=0, atol=0` (pure data
movement of `f64` must be bit-exact). This download happens once, after the
correctness run and before the timed loop; it is never inside a timed
closure, per `AGENTS.md`'s GPU Timing Fairness policy. A mismatch fails that
pattern/backend cell (`status: verification_failed`) and is never timed; it
does not abort the rest of the suite.

## Participant Eligibility

`participants_gpu` is the per-pattern allowlist, mirroring how
`participants` gates `cpu/permutation`'s HPTT column:

- **All contiguous-source (`col_major`) patterns**: all four permutation
  backends -- `tenferro-cuda-transpose`, `tenferro-cuda-to-contiguous`,
  `cutensor`, and `pytorch-cuda`.
- **`tn_light_415_24d_scattered_to_colmajor`** (explicit source strides):
  `tenferro-cuda-to-contiguous`, `cutensor`, `pytorch-cuda` only.
  `tenferro-cuda-transpose` is excluded because the eager op only accepts a
  compact col-major `Tensor`. The CPU suite's view-based `tenferro-rs`
  participant remains eligible for this pattern, while HPTT is excluded.
- **`memcpy_24d_contiguous`**: `memcpy-d2d` only, mirroring the CPU
  precedent exactly -- the CPU pattern's `participants` is
  `["memcpy", "strided-rs"]`, i.e. only memcpy-family columns, never
  `tenferro-rs`/HPTT/Julia. The CPU odometer is an internal untimed
  correctness reference rather than a participant. This
  pattern is a pure bandwidth baseline kept isolated from the
  materialize-kernel backends on both CPU and GPU, even though the other
  backends could trivially express an identity permutation.
- **cuTENSOR rank limits**: some cuTENSOR builds cap the number of tensor
  modes below this suite's largest patterns (rank 23 `reverse_23d_2`, rank
  24 `tn_light_415_24d_contiguous_same_perm`/`tn_light_415_24d_scattered_to_colmajor`).
  Rather than excluding `cutensor` from these patterns' `participants_gpu`
  up front, the JSON still lists it as a participant, and
  `src/bin/benchmark_gpu_permutation.rs` verifies rank support **at
  runtime**: if `cutensorCreateTensorDescriptor` / `cutensorCreatePermutation`
  / `cutensorCreatePlan` reject the configuration, the runner records
  `status: skipped` with a note instead of crashing the whole binary.

Every exclusion is documented with a `notes_gpu` entry in
`data/instances/permutation_patterns.json`, the same convention the CPU
suite uses for its `notes` field.

## Runner Design

Two runners, both consuming `data/instances/permutation_patterns.json`:

- Rust: `src/bin/benchmark_gpu_permutation.rs`
  (`required-features = ["cuda"]`), covering `tenferro-cuda-transpose`,
  `tenferro-cuda-to-contiguous`, and `cutensor`. Builds a single
  `CudaBackend` for the whole run; uploads the pattern's deterministic data
  once per pattern/backend combination via `tenferro_gpu::upload_tensor`,
  runs the correctness gate, then times the steady-state op with
  `tenferro_gpu`'s explicit `CudaRuntime::synchronize()` as the
  timing-region boundary (see [Timing Policy](#timing-policy)).
- Python: `scripts/benchmark_gpu_permutation_python.py`, covering
  `pytorch-cuda` and `memcpy-d2d`. Follows the argument/env handling and
  synchronization discipline of `scripts/benchmark_gpu_python.py`;
  `torch.cuda.synchronize()` defines the timing boundary.
- Formatting: `scripts/format_gpu_permutation_results.py` aggregates the
  two runners' JSONL outputs into the latest report, modeled closely on
  `scripts/format_permutation_results.py` but embedding GPU info (via
  `scripts/collect_gpu_info.py`, the same helper `format_gpu_results.py`
  uses) instead of CPU info, and rendering a single table (no per-thread-count
  sections, since this suite has no CPU-thread dimension) with column order
  `tenferro-cuda-transpose`, `tenferro-cuda-to-contiguous`, `cutensor`,
  and `pytorch-cuda`. The `memcpy-d2d`
  measurement is rendered once as a device-bandwidth baseline rather than as
  a mostly-empty per-pattern table column.
- Orchestration: `scripts/run_gpu_permutation.sh` runs the Rust runner and
  then each Python backend **sequentially** (never concurrently, per
  `AGENTS.md` timing discipline and GPU Timing Fairness), records `run.yaml`
  via `scripts/collect_run_metadata.py`, and regenerates
  `result/nvidia-gpu/gpu/permutation.md`. Each Python backend
  (`pytorch-cuda`, `memcpy-d2d`) runs as its **own process** so its CUDA
  allocator releases device memory before the next backend. It requires
  `BENCHMARK_TARGET_PROFILE=nvidia-gpu` (rejects any other value) and is a
  **standalone entry point**, exactly like `scripts/run_permutation.sh` is
  for `cpu/permutation`: it is intentionally not wired into
  `scripts/run_gpu_suite.sh`.

Environment variables follow existing conventions:
`BENCHMARK_TARGET_PROFILE`, `BENCH_RUNS`, `BENCH_WARMUPS`, `PATTERN_ID` (run
a single pattern while investigating), and `GPU_BENCH_DEVICE` (CUDA device
ordinal, default `0`, same variable name `scripts/run_gpu_suite.sh` and
`scripts/run_gpu_linalg_jvp_vjp.sh` use). `TENFERRO_CUTENSOR_PATH` selects
the cuTENSOR library search path(s) (colon-separated), defaulting to the
same path baked into `.devcontainer/cuda/devcontainer.json`
(`/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2`).

## Timing Policy

- Timing scope: `steady_state_host_api_plus_device_sync` (see
  `benchmarks/gpu/dense.yaml` for the same scope on other GPU suites). The
  timed region is host API dispatch plus backend-native device
  synchronization only -- CUDA stream/device sync
  (`tenferro_gpu::CudaRuntime::synchronize`, `cutensorPermute` followed by
  the same runtime sync, `torch.cuda.synchronize()`). No D2H downloads happen inside any
  timed region; downloads for correctness verification always happen once,
  before the timed loop, per `AGENTS.md`'s GPU Timing Fairness policy.
- All setup happens **outside** the timed region -- this is a
  fairness-relevant choice, made explicit here the same way the CPU spec
  documents HPTT's contract: the cuTENSOR plan (`cutensorCreateTensorDescriptor`
  / `cutensorCreatePermutation` / `cutensorCreatePlanPreference` /
  `cutensorCreatePlan`) is built **once per pattern, outside the timed
  region**, so the `cutensor` column measures steady-state kernel execution
  with planning cost excluded; likewise torch's `torch.as_strided` /
  `.permute` view construction and every H2D upload
  (`upload_tensor`, `torch...to(device)`) are performed once per pattern/backend before any timed
  iteration. Timed regions contain only dispatch of the already-planned
  materialize op plus device synchronization.
- Per-pattern iteration counts scale with pattern size, matching
  `cpu/permutation`'s Rust runner exactly: patterns with `elems >= 2^23`
  use `warmup=3, iters=15`; smaller patterns use `warmup=5, iters=40`.
  `BENCH_RUNS` / `BENCH_WARMUPS` override both runners identically.
- Reported as median with p25/p75 in milliseconds, plus bandwidth in GB/s
  (`bytes = 2 * elems * 8`, i.e. one read plus one write of `f64`, the same
  formula `cpu/permutation` uses).
- Hardware sizing sanity: the largest pattern
  (`rotation_6d_32_32_32_32_16_16`, 2^24 elements) is 2 GiB source + 2 GiB
  destination = 4 GiB, well within a 12 GiB RTX 3060's memory. `f64`
  permutation is a pure memory-bound copy kernel; consumer-card `f64` FLOP
  throttling (RTX 3060's `f64` throughput is a small fraction of its `f32`
  throughput) is irrelevant here, since no arithmetic happens beyond the
  data movement itself.

## Record Shape

Both runners emit the same JSONL record shape (see
`src/bin/benchmark_gpu_permutation.rs::ResultRecord` and
`scripts/benchmark_gpu_permutation_python.py::base_record`), mirroring
`cpu/permutation`'s Rust/Julia record shape with two differences: `device`
(a GPU name string, e.g. `"NVIDIA GeForce RTX 3060"`) replaces `threads`
(there is no CPU-thread dimension), and the bandwidth field is named
`bandwidth_gbs` rather than `gbps`.

Fields: `schema_version`, `suite_id`, `runner` (`"rust"` or `"python"`),
`pattern_id`, `label`, `backend`, `shape`, `perm`, `dtype`, `elems`,
`bytes_rw`, `device`, `status` (`ok` / `verification_failed` / `skipped` /
`not_configured` / `runtime_failed`), `correctness`, `per_call_allocation`,
`warmup`, `iters`, `median_ms`, `p25_ms`, `p75_ms`, `bandwidth_gbs`,
`notes`.

## Fair-Comparison Guards

Supervision checklist, restated as suite requirements (parallel to
`cpu/permutation`'s):

1. tenferro columns are eager materialize/copy paths only; no trace, no AD.
2. `cutensor` rows exist only where the public cuTENSOR API expresses the
   identical semantics; rank rejections are recorded as `skipped`, not
   silently dropped or crashed past.
3. Pattern definitions live in one JSON file consumed by both suites (CPU
   and GPU) and both runners within this suite; no per-runner hardcoded
   shapes.
4. Correctness is verified against a host-computed reference, downloaded
   outside the timed region, before any timing.
5. Input data is deterministic (`deterministic_index_value`); no RNG state
   in the comparison.
6. Allocation semantics are uniform across columns or explicitly footnoted
   (`tenferro-cuda-transpose`, `tenferro-cuda-to-contiguous`).
7. Downloads never happen inside a timed region (`AGENTS.md` GPU Timing
   Fairness); the Rust and Python runtimes never run concurrently
   (`AGENTS.md` Benchmark Timing Discipline).

## Future Work

- Row-major destination variants if tenferro grows first-class row-major
  output support (same caveat as `cpu/permutation`).
- Coupling with `gpu/einsum` diagnostics: patterns derived from
  permutation-heavy einsum instances can be added when planner analysis
  identifies new hot permutations on CUDA, mirroring
  `cpu/permutation`'s `tn_light_415_24d_*` pair.
- A `cusparse`/`ginkgo`-style vendor comparison is not applicable here
  (permutation is a dense operation), but a second dense vendor library
  beyond cuTENSOR (e.g. a raw CUDA kernel baseline) could be added if a
  concrete regression investigation needs it.
