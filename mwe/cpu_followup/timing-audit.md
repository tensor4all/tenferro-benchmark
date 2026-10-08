# Public-API timing audit at b3f47296

The timer encloses calls on prepared inputs. The enclosing backend/eager
session spans validation, calibration and all samples. CPU session admission,
managed placement, resource lending and `CpuExecSession` construction occur in
`CpuBackend::run_backend_session_cached` before its callback. The operations
reach `CpuExecSession` with its already-entered context. Native Rayon/provider
dispatch within an operation remains part of execution. No runner `taskset`
or `numactl` is used; public CPU backend managed placement remains active.

| Path | Prepared outside timer | Timed execution |
|---|---|---|
| Analytic/reduction `_read` | tensor, each borrowed TensorRead descriptor, entered session, initial completed call | public operation, allocated output |
| Indexing | operands, i64 index/start tensors, configs, entered session, initial cached plan | public operation including implementation-owned index workspace |
| Structural/materialization | tensors/views and descriptors, entered session | copy/cast/materialization and fresh output |
| Activation | input EagerTensor/runtime/session, completed call and full scalar-oracle validation | declared eager operation with no gradients, intrinsic intermediates/output |
| Linalg | input tensors and borrowed CPU execution session, provider lazy initialization | full public norm/slogdet operation; for slogdet both public outputs retained |
| FFT | input tensor, reusable FftExecutor, entered CPU session, first completed transform to prime plans/scratch | cached executor transform and output |
| Metadata | fresh owned inputs for each call, SliceConfig and retention storage | consumed-input metadata view construction; no data copy or session |
| Permutation | original physical source, permuted borrowed view descriptors, session, full independent odometer verification | materialize fresh column-major output |

`try_index_tensor` in upstream's indexing implementation internally copies i64
indices into an `IndexTensor` workspace on each public call. The MWE has no
harness index allocation or conversion inside timing. Report these as the
cost of the **whole public API operation**, including that implementation
workspace, rather than claiming isolated gather/scatter kernel throughput.
Plans are cached and primed outside sampling. Scatter is additive in tenferro;
the reference uses `scatter_add` / JAX `.at[].add`. The indices form a permutation
so the updates have no collisions. Output allocation is matched in both arms.

The prepared trace call performs internal admission/session work on every
`Runtime::run_prepared` invocation and does not expose a public borrowed
execution-session call at this revision. Its selected observations remain
scope-ineligible for standard operation claims. The separately named binary
trace diagnostic is not used in this campaign's performance ratios.

Every batch allocates its input iterator and output-retention storage before
timing, holds every output until the interval ends, then drops outputs/inputs.
Python JAX uses dynamic input arguments, compiles and completes an initial
execution outside timing, and calls native completion inside each batch call.
Julia specializes the batch on the operation closure type; owned inputs,
typed retention storage, the type probe and explicit GC are outside timing.
Signature calculation and all inspection copies are untimed in every arm.
