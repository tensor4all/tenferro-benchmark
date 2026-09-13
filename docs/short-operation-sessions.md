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

The single-matrix `cpu_ops` small suite is diagnostic: default `all` excludes it;
`PUBLICATION_GATE_SUITE=small` or `BENCH_INCLUDE_SINGLE_CALL_DIAGNOSTICS=1`
explicitly enables it. Old small-work eager/compiled routes, and short FFT
routes that enter internal sessions, require
`BENCH_INCLUDE_SETUP_DIAGNOSTICS=1`. Their ordinary APIs cannot be called from
inside the borrowed shared CPU session: the backend intentionally rejects
re-entry. They are not relabeled as shared-session measurements. Supporting
EagerTensor/compiled trace in that same scope requires a tenferro API/runtime
change; this benchmark change does not bypass the guard.

CPU and GPU trace runners now prepare execution plans with `prepare_compiled`
before warmup and call `run_prepared` inside timing. Graph compilation alone
did not exclude the metadata/ingress preparation performed by `run_compiled`.

A shared BackendSession is distinct from one executor entry. On this revision,
BLAS resolves to `ProviderDefaultExclusive`; its session callback does not
retain an entered executor context, so each operation still performs an
internal executor entry. Faer does retain that context. The benchmark records
this remaining provider cost rather than bypassing tenferro's exclusivity guard.

JAX thread configuration now additionally exports `PJRT_NPROC` before importing
JAX. The earlier Eigen/XLA flag alone did not constrain the CPU client's thread
pool on this machine. Earlier JAX rows cannot substantiate 1-thread/4-thread
comparisons; rerun each affected suite with this correction. This does not
invalidate the independent PyTorch or Julia measurements in those runs.

The sampling rule is fixed in [AGENTS.md](../AGENTS.md). Earlier timing-policy-2
raw data are preserved, not relabeled with the new measurement scope. See the
[root-cause investigation](m5-short-operation-root-causes.md) for why a reused
runtime is insufficient and which historical ratios were misleading.
