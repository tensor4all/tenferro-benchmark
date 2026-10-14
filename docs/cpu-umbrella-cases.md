# CPU umbrella regression cases (#177)

The generator registers 58 additional arms in `cpu/perf_issues`, manifest v6.
Each family has a separately labelled PyTorch, direct public CPU operation or
scalar host reference. Existing folded workloads remain registered under their
original issue IDs. No LibTorch runner is used.

| Umbrella | Added family | Reference | Thread counts |
|---|---|---|---|
| #2139 | Typed allocating add, held/scoped, n=16/256/4096 | PyTorch add | 1 |
| #2139 | Typed reads into caller output | PyTorch add(out=) | 1 |
| #2139 | Rank-2 Host container construction, 64/4096/65536 values | Owning host descriptor | 1 |
| #2139 | Allocating typed 2x2 matmul and separate allocation counters | PyTorch / scalar product | 1 |
| #2139 | Empty scoped entry and held open/close | Host empty callback | 1 |
| #2139 | Allocating neg across 8/32 MiB inputs (#2095) | PyTorch neg | 1 |
| #2141 | Repeated prepared execute_into (#1899) | Public dot_general_read_into | 1 |
| #2141 | Prepared GEMM-bound 1536 cubed contraction (#1863) | PyTorch matmul | 1, 8 |
| #2142 | Packed LU factor and prepared LU solve, batch=1024, n=8/16 (#1884) | Scalar LU (all outputs) / PyTorch lu_solve | 1, 4 |
| #2142 | Separate lane-policy diagnostic, batch=16, n=16 (#2000) | Serial scalar LU | 1, 4 |
| #2143 | Offset, padded-stride 64x64 accumulate destination (#1838) | Direct host strided update | 1 |
| #2143 | Host typed transpose/slice/reshape allocation counters (#2040/#1917) | Borrowing host descriptor | 1 |

`thread_counts` is part of each new case contract. The runner and per-thread
metadata filter expected/selected IDs using the same thread domain. Existing
cases without that field retain their prior behavior. For a focused run, use
explicit IDs from the generated manifest with `BENCH_INSTANCE`, selecting
`BENCH_COVERAGE=full` when diagnostics are required. Run 1T, 4T and 8T
sequentially, never as concurrent benchmark processes.

The allocating typed add uses `TypedTensorSessionOpsExt::add`. The caller-output
arm uses public `add_read_into` with typed reads; it does not assert that the
typed extension has an `add_into` method. The held session is entered before
sampling; scoped entry is a separately labelled session-entry diagnostic.
`session_entry_cpu_held_empty` measures opening and closing a held session,
not lending a view of an already-open one.

Inputs, wrappers, output destinations, plans, factorization for prepared solve,
backends and thread pools are prepared outside timing. Every short operation
uses a batch clock with retained outputs and a memory bound. Constructors
consume preallocated payload Vecs and retain the resulting owners through clock
stop; payload allocation is excluded. Their clocks are diagnostic and recorded
under `diagnostic_timing`, separately from the untimed allocation pass and
standard operation samples. `size_of_bytes` reports the actual compiled type.
Strided accumulation uses alpha=1, beta=1 and an independent, seed-restored output
for each operation. View setup/restoration precedes every batch clock; all
updated values and untouched guard regions are checked afterward.

The LU factor arm uses the publicly callable `LinalgBackend::lu_factor` through
`with_cpu_exec_session`, as the existing public-API suite does. It retains
packed LU, one-based swap pivots and parity. The independent scalar reference
returns the same outputs and runs serially, even when requested thread metadata
is 4T. The input forces nontrivial row swaps, so validation exercises pivot and
parity behavior. The solve arm prepares its own factors before sampling and
checks the complete analytical all-ones solution.

Two requested observations have no public observer at the tested revision:

- Binary-dot plan builds per `execute_into`: descriptor-cache entries/hits/misses
  are available but do not count binary plan construction. The counter case
  retains allocation/cache observations, reports the build count as null, and
  is explicitly unsupported for that requested observation.
- Internal linalg lanes and item-threads: the operation and its Rust allocations
  can be checked, but requested worker count is not the actual lane plan.
  Those two fields remain null and the target counter case is unsupported.

Unsupported rows remain listed, cannot count as covered and never receive
invented zeros. This registers the public instrumentation gaps for a future
upstream fix. A new observer should make these same IDs executable; it should
not require another case registry. Rust allocator counts include Rust workers
in this process, but exclude foreign BLAS/C allocator activity and cannot
measure pooled-buffer reuse volume. Counter passes do not supply operation
latencies or automatic performance verdicts.

Registration does not impose a baseline campaign. A performance-fix PR should
collect its focused IDs using the existing CPU devcontainer/MKL workflow and
report the declared thread counts. A scan is screening only; performance
claims use the existing paired-confirmation protocol.
