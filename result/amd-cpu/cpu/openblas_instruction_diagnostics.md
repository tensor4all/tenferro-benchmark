# OpenBLAS CPU work attribution (instruction counts, not timings)

Historical pre-optimization diagnosis, retained with the revisions below.
Subsequent implementation/evidence is in `../blas-full-overwrite/`,
`../eager-read-forwarding/`, and `../canonical-copy-dispatch/`; native results
in `../native-1t/` are inconclusive. The original raw diagnostic directories
are retained in `../openblas-attribution/raw.tar.gz` (extract under
`data/results/amd-cpu/diagnostics/`). No new measurement was collected for PRs.

## Scope and limits

AMD EPYC 7713P; benchmark source `976c4b9a459f94086c714ec0f6836978a7bbdfc0`,
tenferro `d8759f4320a337d2399f4a87dfec55af51d2ebf1`, strided-rs
`fbd10fa5b70bb462b961cfd9e02faadb6bb95be0`. Docker image
`sha256:228bf2e4baf4ef4117a65bb8a5a2d58247ce0c0985aec8a8fb53b7a58193893f`,
with Valgrind 3.22 installed in the analysis container. Optimized Rust build,
line debug information, 16 build jobs. The generated dependency lock is retained
with the raw artifacts.

Rust and PyTorch 2.12.0+openblas resolve the same OpenBLAS 0.3.26 shared library
under `/opt/openblas/lib/`. Runtime diagnostics report a BLAS CPU backend,
thread budget **1**, worker count **1**, and OpenBLAS threads **1**.

Three other-user Julia calculations were active. **No native timing comparison
or speedup claim is made.** These are sequential Callgrind `Ir` (executed
instruction) profiles. Instruction shares are not time shares, memory bandwidth,
or an estimate of an achievable speedup. SIMD instructions, REP instructions,
cache behavior, waiting and vendor dispatch prevent that conversion. The
benchmark's elapsed-time output under Valgrind must not be used as performance
results. The initial pass profiled Rust only; the paired follow-up below also
profiles warmed PyTorch execution, still without native timing comparisons.

Full-process profiles include loading, input creation, graph setup, one warmup
and one sampled invocation per distinct contraction path. To distinguish setup
from repeated execution in the elementwise case, a second profile uses three
sampled invocations with the same one warmup. The difference contains two extra
warmed operations plus loop/output-cleanup work. Identical paths are deduplicated
by the existing runner. The language-model case has two distinct paths, thus
four total contractions in its full-process profile.

Raw artifacts: `data/results/amd-cpu/diagnostics/openblas-instructions/`.
An initial caller-thread-filtered profile missed worker execution; it is named
`discarded-main-thread-only.callgrind` and is excluded from every conclusion.

## 1. Eager elementwise execution copies both inputs

Case: `bin_elementwise_mul_2048x2048`, float64.

| Instructions per additional warmed operation | Eager | Trace |
|---|---:|---:|
| Whole-process differential / 2 | 40,452,214.5 | 34,726,087 |
| Shared multiply kernel (self instructions) | 34,603,035 | 34,603,035 |
| Input-copy kernel (self instructions) | 5,767,494 | 0 |

The multiply kernel work is identical. The extra work is in materialization,
not a different multiplication kernel or BLAS. Callgrind records four calls to
`concrete_tensor_read`'s materialization path for two operations and eight for
four operations: two input copies per operation. Setup copies in `new_leaf`
are separate and are not the source of this scaling.

Source chain (paths relative to tenferro-rs):

- `crates/tenferro-ad/src/eager_exec.rs:473`: materializes `concrete_inputs`
  before dispatching standard operations, including `Mul` at line 490.
- `eager_exec.rs:170-180`: `TensorRead::View` becomes an owned compact tensor
  through `to_contiguous_read`, including compact views.
- `crates/tenferro-cpu/src/lib.rs:365-373,436-458`: materializes a view through
  `typed_materialize_view_with_pool`.

Each input is 32 MiB. Copying both inputs adds 64 MiB of temporary payload and
**128 MiB of logical read+write traffic per multiplication**, beyond the
96 MiB logically required to read two inputs and write one output. These are
shape-derived traffic counts, not measured DRAM traffic.

**First optimization target:** avoid repeated view-to-owned copies for already
compact inputs while preserving storage ownership/lifetime and backend input
contracts. The current boundary explicitly requires compact owned tensors;
simply deleting it is not a validated fix.

## 2. Multi-contraction work spends substantial effort on packing and zeroing

Case: `lm_batch_likelihood_sentence_3_12d`: 38 inputs, 37 contraction steps per
path (36 dot steps and one broadcast multiplication), maximum intermediate
1,900,800 elements. Two paths × (one warmup + one sample) were profiled.

| Full-process observation | Executed instructions | Share of process Ir |
|---|---:|---:|
| OpenBLAS `dgemm_kernel_HASWELL` (self) | 1,778,603,368 | 44.45% |
| `memset` (self, all callers) | 1,026,933,616 | 25.66% |
| `materialize_canonical_operand` (inclusive) | 714,933,050 | 17.87% |

The inclusive row and all other categories must not be summed as independent
costs. In particular, the `memset` row includes vendor work:

- 522,521,244 instructions were called from OpenBLAS `dgemm_beta_HASWELL`.
- `execute_dot_allocated` called `pool_acquire_zeroed` **144 times**, once for
  each of the 36 dot steps across four contractions; that allocation subtree
  accounts for about 500.5 million instructions.
- Canonical operand materialization occurred **204 times**. This is actual
  tensor traversal/packing, not just subscripts parsing.
- There were **61,688 `cblas_dgemm` calls** across those four contractions:
  high-rank batching decomposes some logical dot operations into many GEMMs.

Source locations:

- `crates/tenferro-cpu/src/exec_session.rs:485-560`: allocation-returning dot
  uses beta=0; without a provider full-overwrite witness it allocates initialized
  output before calling the provider.
- `crates/tenferro-cpu/src/provider.rs:1891-1947`: `BlasGemmProvider` does not
  override the default `uninit_provider` capability.
- `crates/tenferro-cpu/src/dot_runtime.rs:639-673,1043-1086`: fallback constructs
  canonical layouts and materializes both operands before provider GEMM.
- `crates/tenferro-cpu/src/gemm/blas_gemm.rs:34-59`: sequential grouped-GEMM
  fallback; batching can generate many individual calls.

**Next targets:** reduce avoidable canonical repacking and investigate a properly
proven full-overwrite BLAS output path. Never bypass the uninitialized-memory
safety contract. These profiles establish CPU work worth investigating, not
whether each pack or zero-fill can safely be removed.

## 3. Large GEMM is mostly vendor compute, not eager dispatch

For `bin_matmul_1024`, the OpenBLAS GEMM kernel accounts for **96.17% of
full-process instructions** (1,024,773,264 of 1,065,532,635). This case gives no
instruction-count reason to prioritize eager dispatch over the BLAS kernel.

For `bin_batched_matmul_b32_m128_n128_k128`, the same GEMM kernel executes
66,547,072 instructions in both eager and trace profiles. Trace has additional
copy/layout work. This does not establish a new eager/PyTorch slowdown: a quiet
native run is still required.

## Paired follow-up: warmed PyTorch versus eager/trace

Additional local artifacts:
`data/results/amd-cpu/diagnostics/openblas-instructions-continued/`.
`summary.json` is generated by `summarize.py` from **100%-threshold self-cost**
annotations (not inclusive trees or truncated top-function listings).

The PyTorch diagnostic uses the same instance JSON, float64 zero inputs,
`opt_einsum.contract_expression`, and stored paths as `benchmark_python.py`.
Imports, operand construction, expression construction and one warmup precede
`CALLGRIND_START_INSTRUMENTATION` / `CALLGRIND_ZERO_STATS`. The counted window
contains expression execution and output destruction, then explicitly dumps
stats before stopping instrumentation. Its `.callgrind.1` file, not the tiny
exit dump, is the relevant artifact. PyTorch intra/inter-op and OpenBLAS thread
counts are asserted to be 1. Rust still uses the N=3 minus N=1 differential,
which includes both worker-thread work and output cleanup. These are comparable
operation-work diagnostics, **not identical native timing boundaries**.

For elementwise and batched matmul, PyTorch also uses a (3−1)/2 differential;
for large matmul it uses a single warmed window. For the language-model case,
Rust differences are divided by **4**, because there are two additional runs
for each of two distinct paths. PyTorch's two individually profiled paths are
averaged. Allocator state and small harness costs can affect these estimates.

| Million instructions per warmed operation | Eager | Trace | PyTorch |
|---|---:|---:|---:|
| Elementwise multiply 2048×2048 | 40.45 | 34.73 | 4.84 |
| Batched matmul 32×128×128 | 43.89 | 46.94 | 39.42 |
| Matmul 1024×1024 | 533.35 | not profiled | 523.95 |
| `lm_batch_likelihood_sentence_3_12d` (path mean) | 1002.91 | 903.72 | 774.46 |

These numbers do **not** predict elapsed-time ratios. In particular, contiguous
multiply can be memory-bandwidth-bound despite very different instruction counts.

### A. The shared multiply kernel has a second, independent issue

After removing eager-only copies, the multiply kernel itself uses **34,603,035
instructions** per operation versus **4,718,656** in PyTorch. Both are AVX2
implementations; this is not scalar fallback or a different BLAS provider.

Pinned strided-rs `strided-basic/src/simd.rs:32-113` generates this kernel.
Its full-vector loop at lines 61-71 uses `partial_load_f64s` twice and
`mask_store_ptr_f64s` once, even when all four lanes are present. The retained
Rust disassembly shows **three indirect jumps per four elements** into the
load/store machinery, plus bounds/loop control. The resulting kernel count is
approximately 33 instructions per four elements. The retained PyTorch assembly
has an ordinary unrolled full-vector load/multiply/store loop; its count is
approximately nine instructions per eight elements.

**Improvement location: `strided-rs`, not a duplicated tenferro SIMD kernel.**
Investigate a full-vector body with partial-lane handling only for the tail.
Preserve unaligned access, bounds and uninitialized-destination safety: do not
create initialized Rust references over unwritten output storage. This is an
optimization candidate, not an implemented or timed speedup.

### B. Same GEMM kernel work, additional zeroing and copies above it

Batched matmul uses exactly **33,273,536 GEMM-kernel instructions per operation**
in all three implementations. Large matmul likewise uses **512,386,632** in
both profiled implementations. This controls for the vendor compute kernel
much more directly than a whole-process instruction share.

For batched matmul, `memset` self-cost is approximately **8.39M** instructions
in both Rust modes versus **4.20M** in PyTorch. Identified copy/map kernels add
**0.72M** in eager and **3.06M** in trace; no corresponding identified copy
kernel appears in this PyTorch window. The eager call tree records eight
`concrete_tensor_read` materializations across four contractions—two per
contraction—separate from setup in `new_leaf`.

For large matmul, `memset` is **16.78M** in eager versus **8.39M** in PyTorch;
eager also has **1.44M** identified copy/map instructions. Thus the small
instruction-count difference is primarily outside the shared GEMM kernel,
consistent with the allocation/copy source paths above, not a slower vendor
matrix multiply.

### C. Trace does not eliminate contraction zeroing

| LM path-mean self-cost, million instructions | Eager | Trace | PyTorch |
|---|---:|---:|---:|
| GEMM compute kernel | 444.65 | 446.18 | 426.68 |
| `memset`, all callers | 267.37 | 303.88 | 131.47 |
| Identified copy/map kernels | 229.27 | 94.63 | 32.91 |

Copy/map rows aggregate identified strided identity-copy and PyTorch direct-copy
symbols, not all possible copying or measured memory traffic. The kernels and
layouts are not identical in this network case, unlike the simple GEMM controls.

Trace substantially reduces the identified copy work, but **still performs
pooled zero initialization**. Its retained call tree attributes large `memset`
subtrees to `PoolScalar::pool_acquire_zeroed`, in addition to OpenBLAS's
`dgemm_beta_HASWELL`. Therefore trace is not a zero-copy/zero-initialization-free
reference. A trace/eager difference alone cannot isolate dispatch overhead.

### Follow-up validation and priority

- Explicit PyTorch windows exclude the roughly 5.2-billion-instruction Python
  startup/full-process profile; it is retained only as a setup-cost diagnostic.
- One-versus-three windows give exactly linear multiply-kernel counts; the
  summary check asserts the Rust eager/trace equality and PyTorch kernel count.
- Same-GEMM-kernel checks pass for the batched control. Output shapes and zero
  outputs are checked outside PyTorch windows; these are smoke checks, not
  general nonzero numerical oracles.
- No numerical implementation was changed. Backward/AD-specific profiling was
  not performed in this pass; the newly identified shared SIMD issue took
  priority over broadening the case list.
- Next optimization candidates: **strided-rs full-vector multiply loop**, eager
  view materialization, then safe BLAS full-overwrite output and contraction
  layout planning. Native timing remains necessary to rank their actual impact.

## Benchmark-contract finding

The CPU operation runner (`src/bin/publication_gate.rs:277`) enters a shared
CPU execution scope. The einsum runner's eager loop
(`src/main.rs:678-703`) reuses its runtime but does **not** enter an outer
`with_execution_scope`. Each contraction can still enter backend sessions.
Therefore the historical shared-scope CPU-ops conclusion must not be applied
wholesale to einsum, especially for the deferred 4T investigation.

## Verification and remaining work

- Rust einsum and publication-gate binaries built with `system-openblas`.
- PyTorch provider, BLAS/LAPACK operations and effective 1T verified.
- Three publication-gate unit tests passed, including scalar-reference primal
  and both-gradient checks under the shared OpenBLAS execution scope.
- No tensor/kernel implementation changes were made.
- Full `test_cpu_ops_linalg_ad.sh` and native eager/trace/PyTorch timings remain
  deferred to a quiet timing window; its missing-checkout build blocker is fixed.
- Next: quiet 1T paired comparisons and native profiling of the identified paths;
  only then evaluate fixes and investigate Rayon/BLAS 4T interactions. No historic
  MKL-to-new-OpenBLAS result is treated as an implementation speedup.
