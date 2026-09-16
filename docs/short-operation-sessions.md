# Short operations and session reuse

The standard small-work measurement is now many operations in one clock
interval, inside one already-created backend session. Inputs, session construction and
entry, plans, warmup, output-container allocation and cleanup are outside the
clock. Intrinsic output allocation remains inside. All final outputs survive
until timer stop. A normalized ns/op is the batch duration divided by its
operation count; it is not standalone call latency.

`./scripts/run_cpu_session.sh 1 4` compares 1024 distinct f64 matmul/solve pairs
at matrix sizes 2, 4, 8, 16 and 32. The same build explicitly selects Accelerate
or faer at runtime. Both use tenferro's shared BackendSession; the 4-thread faer
case uses the managed Rayon domain and faer's internal parallelism. It does not
add an outer Rayon parallel loop over the independent matrices. PyTorch calls
the same operations in a Python loop over the same logical values. The Python
loop cost is included and the route is labeled accordingly. Batched tensor
operators are a different workload. Results are generated in
[result/mac-cpu/cpu/session_matrix.md](../result/mac-cpu/cpu/session_matrix.md).

`cpu/small_work` extends this rule to add, dependent add chains, gather,
reduce_sum, prepared real/complex einsum and solve. Calibration starts at 1024
operations and uses one interval for the entire batch. Every sample records
its iteration count and total elapsed time; the report also shows normalized
time. A session spans all warmups, calibration and samples. The report retains
noisy rows. At most 65536 outputs are retained; inability to reach the calibration
target is an error, not a fabricated duration.

Metadata-only TensorValue views use 16 prepared owners per interval, capped
at 4 GiB of input storage for the largest existing case. They do not need an
execution session. PyTorch uses the same operation count. Small cached FFTs
(length <= 16384) use 128 calls per interval in one session; larger FFTs retain
their existing operation scope. CSV notes record operation count and median
batch duration, while the existing latency columns remain normalized per call.

## CPU eager, AD and prepared trace

Since tenferro-rs PR #1802 (merge `d8759f4320a337d2399f4a87dfec55af51d2ebf1`),
`publication_gate` uses `CpuBackend::with_execution_scope` outside all timing.
Its eager runtime and prepared trace runtimes share clones of that exact backend
witness. This is **not** a borrowed BackendSession and does not bypass the
operation re-entry guard. The `cpu_ops` small suite is now included by default;
its former isolated-call results remain diagnostics, not new scope measurements.

Rust, PyTorch and JAX calibrate a batch from one operation toward 5 ms. Batch
size is also capped by a 64 MiB logical-retention reservation: sixteen f64
tensors at the product of the dimensions in the declared shape, plus 8 KiB of
handles/AD metadata per invocation. This deliberately overestimates rectangular
and multiple-RHS shapes; it is an estimate, **not** a bound on backend scratch,
allocator caches or process RSS. A single operation is allowed if its estimate
already exceeds the budget. A memory-limited batch may remain below 5 ms; its
actual duration is preserved. Each batch prepares all fixtures and retention
containers before timing, and destroys inputs/outputs afterward. Fresh eager
and PyTorch AD leaves prevent gradient accumulation; prepared trace uses its
immutable default inputs. Allocation-returning APIs remain allocation-returning
(no preallocated output/reuse claim). Intrinsic eager tape/backward work remains
part of the declared forward-plus-backward operation.

`run_cpu_ops.sh` writes `cpu_ops_samples_t<N>_<timestamp>.jsonl` beside the CSV.
It contains raw batch durations, counts, normalized ns/op, memory estimates and
runtime/provider observations. CSV `sampling_policy` distinguishes new batches
from historical isolated-call data. Python fixtures now match Rust's logical
column-major values while retaining native contiguous Python storage, including
an untimed [n,n,batch] to [batch,n,n] axis mapping. Old fixture values differed;
old/new timings are not a controlled execution-scope speedup experiment.
These shared Python fixture helpers also affect future GPU AD runs; existing
GPU data and reports are unchanged.
Independent scalar-reference tests cover batched matmul primal and both input
gradients at 2x2 batch16, 4x4 batch3 and 16x16 batch1. Other successful timing rows
are execution evidence, not blanket numerical verification.

The separate `cpu/small_work` eager/compiled and short FFT diagnostic routes
still require `BENCH_INCLUDE_SETUP_DIAGNOSTICS=1`. This change does not silently
relabel those unadapted runners; ordinary eager APIs still cannot be entered
inside a borrowed BackendSession.

CPU and GPU trace runners now prepare execution plans with `prepare_compiled`
before warmup and call `run_prepared` inside timing. Graph compilation alone
did not exclude the metadata/ingress preparation performed by `run_compiled`.

A shared BackendSession is distinct from one executor entry. Since tenferro-rs
PR #1796, managed BLAS sessions in `ProviderDefaultExclusive` mode retain the
entered context across operations, just as faer sessions do. Older results on
`a793c2e9` include per-operation BLAS executor entry and remain historical
baselines. The permutation runner also uses one pre-entered session for its
warmups and samples; every borrowed input descriptor is prepared before timing.

JAX thread configuration exports `PJRT_NPROC` and Eigen/XLA settings before
importing JAX, but environment requests alone do not prove a worker count.
The CPU ops runner records observed Linux `tf_XLAEigen` workers and affinity;
Rust records its worker/thread budget and queries oneMKL's maximum threads
inside the entered scope, and PyTorch records both thread APIs. The issue-1801
Linux study uses verified idle, explicitly selected physical cores (one at 1T,
four at 4T), as requested for that study. This differs from the repository's
usual unpinned protocol. Affinity is not a substitute for explicit provider
thread configuration. Earlier unverified JAX rows cannot substantiate 1T/4T
comparisons. A configured maximum does not mean every kernel uses every worker.

The sampling rule is fixed in [AGENTS.md](../AGENTS.md). Earlier timing-policy-2
raw data are preserved, not relabeled with the new measurement scope. See the
[root-cause investigation](m5-short-operation-root-causes.md) for why a reused
runtime is insufficient and which historical ratios were misleading.
