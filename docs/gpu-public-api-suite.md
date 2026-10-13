# CUDA public API suites

These suites implement the missing-operation workstream in
[issue #155](https://github.com/tensor4all/tenferro-benchmark/issues/155).
Each family has a YAML selection in `benchmarks/gpu/` and explicit operation,
dtype, shape and unsupported-arm records in `data/instances/gpu_<family>.json`.
An existing einsum, AD or fused-chain measurement does not count as an isolated
forward-operation measurement.

| Family / suite suffix | Public operations | Latest report |
| --- | --- | --- |
| `fft` | `FftExecutor::{fft,ifft,rfft,irfft}`, borrowed eager FFT | [FFT](../result/nvidia-gpu/gpu/fft.md) |
| `nn` | sigmoid, silu, softplus, exact/tanh GELU, softmax/log-softmax including masks, layer/RMS normalization | [NN](../result/nvidia-gpu/gpu/nn.md) |
| `isolated_ops` | arithmetic, analytic functions including erf, compare/select/clamp, axis-wise reductions | [Isolated operations](../result/nvidia-gpu/gpu/isolated_ops.md) |
| `structural` | scatter, take_along_axis, pad, materialized slice, reverse, concatenate, diagonal extraction/embedding, triangular masks, cast | [Structural](../result/nvidia-gpu/gpu/structural.md) |
| `linalg_public_api` | Cholesky, values-only SVD/eigh, pinv with explicit rtol, inverse, determinant, Frobenius norm, triangular solve | [Linear algebra](../result/nvidia-gpu/gpu/linalg_public_api.md) |
| `reuse_contraction` | add/sub/mul/div/neg/conj `_into` and `_read_into`, tensordot, vdot_read, real norm_squared_read | [Output reuse and contractions](../result/nvidia-gpu/gpu/reuse_contraction.md) |

The [coverage inventory](../benchmarks/gpu/public_api_coverage.yaml) maps selected,
partially covered and deferred routes. The operation/route source of truth for
selected cases is the case manifests, including their
`excluded` and `unavailable` maps. These are coverage selections, not measured
result tables. Reports are generated only under `result/nvidia-gpu/gpu/`.

## Run in the repository CUDA devcontainer

```bash
devcontainer up --workspace-folder . --config .devcontainer/cuda/devcontainer.json
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=1 ./scripts/run_gpu_public_api.sh fft --verify-only'
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3 ./scripts/run_gpu_public_api.sh fft'
```

Repeat verification and collection sequentially with `nn`, `isolated_ops`,
`structural`, `linalg_public_api`, and `reuse_contraction`. Do not overlap timing commands or
other GPU benchmark processes. The runner requires a CUDA container, fetches a
`main` tenferro checkout while preserving pinned checkouts, and rebuilds its
Rust binary before collection. Public GitHub fetches use HTTPS inside the image.

On the two-RTX-3060 host, physical GPU 0 is reserved for benchmark collection;
GPU 1 is for debugging and numerical verification. `--verify-only` defaults to
GPU 1 and performs no calibration or timing samples. Normal collection defaults
to GPU 0. `GPU_BENCH_DEVICE` selects the physical `nvidia-smi` index; the runner
isolates its UUID with `CUDA_VISIBLE_DEVICES`, so all frameworks see that card as
process-local CUDA ordinal 0. Both indices, the UUID and the role are recorded.
Do not run heavy compilation or CPU-intensive validation alongside collection,
even when the GPU differs, because the host API timing shares CPU resources.

Verification outputs go under
`data/verification/nvidia-gpu/gpu/<family>/<timestamp>/`, including a
`verification.json` summary and per-backend records. They cannot replace a latest
timing report: the formatter rejects `verification_only` as a measured row.

The maintained defaults are 3 warmups, 15 samples, one CPU thread, GPU ordinal 0,
a 2 ms target batch duration, a maximum of 4096 operations/sample and 256 MiB of
retained outputs. The memory limit may stop calibration below 2 ms; the raw
batch durations and counts make this visible. `BENCH_WARMUPS=0` still performs
explicit initialization before calibration/sampling. Each backend runs in its
own process and completes before the next starts.

Raw runs live in `data/results/nvidia-gpu/gpu/<family>/<timestamp>/`. They contain
`run.yaml`, per-backend JSONL and `fixtures/manifest.json` with typed binary
inputs/reference outputs. Metadata records the tenferro revision, framework and
CUDA versions, GPU, environment, exact command, devcontainer configuration and
source/binary hashes. Failed runs are retained; publication rejects failures,
missing rows, duplicates and invalid samples without replacing the latest report.

## Timing and reference contracts

* Direct uses immediate public operations on one borrowed `BackendSession`.
  FFT uses `FftExecutor` so its plans survive across samples. Eager uses one
  borrowed `EagerSession` with inactive AD recording. Input wrapping and uploads
  happen before sampling. Compiled tenferro trace is not silently called direct:
  `Runtime::run_prepared` does not expose a borrowed-session execution route and
  its per-call session setup cannot satisfy this suite's timing contract.
* PyTorch eager, JAX eager and compiled JAX are distinct backend names. JAX
  compilation occurs outside timing. JAX enables x64 and `highest` matmul
  precision; PyTorch disables TF32. Outputs must remain on the selected GPU.
* Logical values match exactly across backends. Rust uses native column-major
  storage; Python uses native row-major storage. Both are constructed outside
  timing. `rem` compares truncation-based `fmod`, not floor-based remainder.
  Conjugation uses materializing `torch.conj_physical`; diagonal/slice comparisons
  use PyTorch's copy APIs, not metadata-only views. Eager slice returns a lazy
  metadata view and is excluded from that materializing comparison. Scatter is
  additive and matches `torch.scatter_add` / `jax.lax.scatter_add`.
* The timer covers declared operation dispatch and native device completion.
  It never includes output downloads. Returned outputs stay alive until the
  clock stops. Returning APIs include intrinsic output allocation. `_into`
  cases prepare distinct destinations and move-only write handles before each
  batch and reuse the same destination allocations across samples; PyTorch's
  `out=` arm uses the same destination schedule. JAX donation is not output reuse.
  Rust pad/slice/scatter configurations and vdot/norm borrowed read handles are
  also prepared before each batch, including initialization with zero warmups.
  Earlier structural measurements constructed their move-only configurations
  inside the timer and are noncompliant; rerun them before using their ratios.
* References use NumPy/SciPy double precision on the exact rounded inputs and
  check every output element before and after measurement. Shapes and dtypes
  must match; NaNs fail. Matched masked log-softmax negative infinities are
  accepted. FFT absolute tolerance scales as
  `32 * eps * ceil(log2(n)) * norm2(input) / normalization`, with a `1e-9`
  floor, to account for roundoff in cancellation-heavy frequency bins; the
  per-case manifest records the actual tolerances.
  Full sum reductions use an absolute roundoff budget of
  `max(base_atol, 32 * eps * ceil(log2(n)) * norm2(input))`; full products use
  `max(base_rtol, 32 * eps * sqrt(n))` for relative error. These are explicitly
  size-dependent verification policies, rather than elementwise error budgets
  or formal worst-case IEEE error bounds. Both preserve the original fixture
  values and double-precision reference. The initial failed verification run
  `20261013_124057` remains available in raw verification data.

## Explicit limitations

`excluded` means the API can be called for numerical verification, but its
timing contract cannot be matched. `unavailable` means the selected public
operation is absent for that arm. Both are displayed as unsupported, with a
reason, and never count as measured coverage.

The CUDA direct/eager NN composites other than softplus create scalar constants
and upload them per call (`composite/session.rs::scalar` and
`eager/composite.rs::scalar`). This also affects `reduce_mean` and the
batch-varying-index form of `take_along_axis`. The shared-index form avoids the
coordinate-constant construction and has its own case. These internal transfers
cannot be moved outside the timer through the current public API. Pinv/inverse
composites likewise construct scalar/identity operands internally. The concrete
`TensorLinalgExt::{inv,pinv_with_rtol}` CUDA routes actually reject those host
operands at this revision (reproduced in raw run `20261013_094906`); these routes
are marked unavailable. Eager execution succeeds numerically but its internal
setup remains excluded from standard timing. Unexpected numerical execution
failures still prevent publication rather than becoming performance measurements.

Small direct `compare` cases (F32/F64, n=1024) pass numerical verification, but
their consecutive batch times increase in both raw runs `20261013_101359` and
`20261013_102038`. Those records are retained as diagnostics; the latest standard
report excludes these two arms because steady-state sampling has not been
established. Eager and large compare cases remain selected. Source inspection
found that `alloc_bool_output` requests fresh CubeCL handles while dropped
buffers enter the bounded output pool; its causal role in the drift still needs
allocator/lifetime investigation. No upstream regression is asserted.

JAX has no equivalent output-reuse arm. Axis-wise sum-of-squares lacks a single
matching PyTorch/JAX operation; compositions are not timed as substitutes.
Masked softmax lacks a matching stable PyTorch operation. JAX RMS normalization
and eager tenferro values-only SVD are similarly explicit unsupported arms.
Normalization uses no affine parameters, `eps=1e-5`, and nonempty finite inputs;
pinv uses `atol=0`, `rtol=1e-5`, with framework tolerance tensors prepared and
uploaded before timing; triangular solve is left/lower/non-transposed/non-unit.

This first set covers compact F32/F64 paths, complex FFT/conjugation/contractions,
Bool masks and I64 indices. Non-contiguous `TensorRead` routes are not aliases of
compact routes. Stateful Householder updates, RRQR/rank outputs, LU output
contracts, full-SVD gauge-invariant verification, dynamic slicing, cached
accumulation/einsum destinations and new AD transformations require dedicated
fixtures/contracts. Existing QR/SVD/eigh/solve, permutation, AD and perf-issue
suites remain separate coverage; their presence does not count these remaining
routes as measured. CUDA-unsupported eig/eigvals/full-pivot LU, grouped GEMM and
unsupported read-linalg hooks remain excluded as documented in issue #155.

## Existing suites and JAX references

`gpu/dense`, `gpu/einsum`, `gpu/elementwise`, and `gpu/tensornetwork`
select JAX references. The Python runner prepares plans, explicitly compiles JAX
functions, initializes libraries, asserts CUDA placement, and verifies outputs
before bounded batched sampling. It verifies again after sampling. Skipped
verification cannot become a successful measurement. Frameworks execute in
separate processes, releasing their GPU allocator before the next starts.

Use the complete declared shapes for numerical verification on physical GPU 1:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_VERIFY_ONLY=1 GPU_BENCH_DEVICE=1 \
    GPU_BENCH_BACKENDS=tenferro-cuda-eager,pytorch-cuda,jax-cuda,jax-cuda-tree-jit \
    GPU_BENCH_SUITE=benchmarks/gpu/dense.yaml,benchmarks/gpu/einsum.yaml,benchmarks/gpu/elementwise.yaml,benchmarks/gpu/tensornetwork.yaml \
    ./scripts/run_gpu_suite.sh'
```

Verification-only records have no timing samples and cannot replace reports.
`jax-cuda` means whole-operation compiled JIT. `jax-cuda-tree-jit` is a distinct
tensor-network path: the complete original tree is preserved, each contraction
is planned/compiled/initialized before timing, and the Python tree dispatches
those kernels. All inputs and requested outputs match the PyTorch tree arm.
Ordinary binary nodes use `jax.lax.dot_general` directly: shared output labels
become batch axes and shared non-output labels become contraction axes. The
eager tree retains the native intermediate axis order and remaps downstream
labels; only the root is restored to the requested order. Repeated labels or
unilateral reductions retain the general einsum implementation. This preserves
the full tree and avoids invoking JAX's general einsum compiler for every
ordinary binary node. Full-size CUDA verification remains required before
publishing this path.

The tensor-network node-JIT arm disables XLA GPU autotuning and Triton GEMM
through the suite's explicit `tensor_network.jax_compiler_options`. Its default
compiler configuration exceeded the 32 GiB guard (`20261013_125730`); the
configured path compiled all 549 nodes and passed CUDA verification
(`20261013_131210`). Result notes record the options actually passed to JIT
compilation. `GPU_BENCH_JAX_COMPILER_OPTIONS` can override them with a JSON
object; eager JAX and PyTorch do not apply these JIT options.

The previous general-einsum whole-tree lowering exhausted roughly 91 GiB of
host memory and also failed guarded 32 GiB attempts. Those failed raw runs are
retained. The compact native lowering and recorded compiler policy now pass
whole-tree JIT CUDA verification (`compiler_probe_native_no_autotune.json`).
Whole-tree JIT, node-JIT and eager execution remain separate report arms.
The CUDA devcontainer now caps host memory at 32 GiB (34 GiB including swap) to
prevent a compiler process from exhausting the workstation. Per-arm notes record
the active cgroup memory limit and any nondefault JAX compiler options.

The constant-fill tensor-network fixture has an independent analytic reference:
each contraction multiplies child constants and the number of summed index
assignments. It uses exact rounded F32 input values and evaluates the result in
F64, without allocating giant CPU intermediates. Small summed/free-axis trees
are checked against dense NumPy contractions, and prepared JAX tree kernels are
tested on nonconstant inputs.

Tenferro eager sampling reuses one entered session for initialization, calibration
and samples, with inactive AD recording. Standard tenferro trace rows are
unsupported because `Runtime::run_prepared` still enters sessions during execution
and exposes no caller-borrowed session route. Trace public-call latency belongs
in separately named diagnostic suites, including the work in issue #107.

## Checks

Run inside the same devcontainer before collecting:

```bash
cargo check --features cuda --bin benchmark_gpu_public_api -j 4
uv run python -m unittest tests/test_gpu_public_api.py
uv run python scripts/validate_benchmark_suite.py benchmarks/gpu/fft.yaml \
  benchmarks/gpu/nn.yaml benchmarks/gpu/isolated_ops.yaml \
  benchmarks/gpu/structural.yaml benchmarks/gpu/linalg_public_api.yaml \
  benchmarks/gpu/reuse_contraction.yaml
bash -n scripts/run_gpu_public_api.sh
bash tests/test_suite_result_layout.sh
bash tests/test_run_all_docs_outputs.sh
```

The full real-vector sum-of-squares cases use the single native `vdot(x,x)`
PyTorch/JAX operation; axis-wise cases remain unsupported in those references.
Full mean uses an input-size-dependent absolute roundoff policy so that the
near-zero mean cannot pass as an all-zero result. The underlying tenferro mean
API still constructs/uploads a scalar per call and stays timing-excluded.

The structural additions distinguish checked lossless F32-to-F64 `convert`
from general `cast`. Materialized rank-two singleton broadcast uses concrete
`broadcast_in_dim`, `torch.expand_copy`, and `jax.lax.broadcast_in_dim`.
Eager broadcast returns a lazy broadcast view and is excluded from this
materializing comparison. Case selection alone is not measured coverage;
only validated cells in the linked latest reports count.

Dynamic-slice fixtures use I64 device-resident start vectors, half-sized windows,
and separate in-bounds and negative/high clamping cases. JAX start scalars are
prepared from the index vector before compilation and sampling; negative-index
wrapping is disabled. No composed PyTorch indexing substitute is measured.
Gather fixtures select unique in-bounds rows or columns. Rust `GatherConfig`
values and PyTorch index views are prepared before timing; JAX uses native
`lax.gather` with the same index values and output extents.

Multiple-output factorizations retain every output through the stopped clock and
use the sum of their allocation sizes to bound sampling. Full SVD checks shapes,
dtypes, finiteness, sorted/nonnegative singular values, reconstruction and the
entire square-factor orthogonality, including nullspaces. Vector signs/phases
are not compared elementwise. Slogdet verifies both sign and real log magnitude,
including complex determinant phases.

Compact Householder construction returns an opaque public state. Its private
buffers are never accessed: public `r` and full-width `q_columns` projections
are inspected after timing for reconstruction, upper-trapezoidal structure and
unitarity. Torch `geqrf` and JAX raw QR return equivalent compact factors;
JAX's native transposed reflector layout is normalized only during untimed
inspection. State projections used for validation do not count as independently
measured `r`/`q_columns` cases.

Householder state operations have independent cases. `from_factors` receives
pre-generated Q/R factors; `append_columns` receives a pre-generated initial
block and appended columns. Both return opaque states retained through the
stopped clock and verified by untimed public projections. Initial state
construction for `r`/`q_columns` is outside timing. R uses native `triu` on
prepared compact factors as its reference. Prefix Q uses native
`householder_product`; the complement uses Torch `ormqr` with a selector
prepared outside timing. No refactorization or full-Q-plus-slice composition
is substituted for a missing state-query/update operation.

CUDA append/Q application internally constructs and uploads BLAS scalar
operands per call. Those paths are numerically verified but excluded from
standard operation timing. From-factors uses host BLAS scalar arguments and
has a separately verified standard path. Unsupported reference arms and
unselected/failed cells never count as measured coverage.

The `reuse_contraction` suite also selects `axpby_read_into_accum` on
compact F32/F64/C32/C64 vectors of length 1024 and 1048576. Its contract is
`y <- 0.5*x + y`, with distinct prepared inputs and caller-owned destinations.
The native PyTorch counterpart is `y.add_(x, alpha=0.5)`; JAX and the eager
wrapper have no matching public output-reuse arm. Restore every destination
from the same GPU seed before each initialization, calibration, warmup and
measured batch, and synchronize restoration before starting the clock.
Restoration is device-to-device and is excluded from operation time. The
compact CUDA path passes coefficients as host cuBLAS arguments; the strided
source path uploads coefficients internally and is a separate contract.

Matrix output reuse selects public `dot_general_read_into`,
`dot_general_read_into_accum`, and
`SessionCachedDot::dot_general_read_into_accum_cached(Some(0), ...)` on
square 16/256 F32/F64/C32/C64 matrices. All configurations and move-only
operand/output handles are prepared outside the clock. Overwrite computes
`C = A @ B`; accumulation computes `C = 0.5 * A @ B + C` and restores C
from a prepared GPU seed before every batch. PyTorch uses one native
`torch.addmm(C, A, B, alpha=..., beta=..., out=C)` call with matching
output ownership. Eager and JAX have no equivalent caller-owned output API.
CUDA's cached accumulation wrapper currently delegates to ordinary
accumulation and ignores the session-local slot; these rows identify public
routes and do not assert distinct cache behavior or a cache speedup. Cache
miss streams and instrumented counters remain the separate #107 workload.

The `reuse_contraction` suite includes distinct transposed borrowed-read cases
for `add_read`, `mul_read`, `neg_read`, `exp_read`, `tanh_read`,
`reduce_sum_read` (logical axis 1), and materialized `conj_read`.
Rectangular logical shapes 16x64 and 512x2048 prevent swapping logical axes
from passing unnoticed. Each fixture stores the transposed physical matrix;
tenferro creates a transposed `TensorView` on its CUDA allocation and prepares
each move-only `TensorRead` before the clock. PyTorch creates its native
transposed view before timing. Both receive the same logical values and retain
their native layouts; no input materialization is charged to just one arm.
JAX prepares the same logical values outside compilation/timing and compares
the operation result, without claiming an equivalent public strided-view
contract. Eager owned tensors are not substitutes for borrowed reads and stay
unsupported. These cases do not alias other read operations or layouts.

Full public LU selects 24 square/tall/wide cases (side 32/256,
F32/F64/C32/C64) through dynamic `TensorLinalgExt::lu` and
`EagerSessionLinalgExt::lu`. All four outputs P/L/U/parity are retained
through synchronization and checked against independent double-precision
SciPy factorization. tenferro uses `P A = L U`; the fixture transposes
SciPy's `A = P L U` permutation and derives parity from pivot swaps.
The dynamic CUDA implementation returns parity in the input dtype; these
cases do not establish the typed complex API's real-parity contract.
PyTorch/JAX candidates return three outputs and have no verified matching
single-call P/L/U/parity operation, so those cells remain unsupported.
The native cuSOLVER diagnostic-status read is intrinsic to the public LU
call and remains included; benchmark output inspection downloads are untimed.

Least-squares contract checks are a separate entrypoint:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=1 ./scripts/run_gpu_public_api.sh lstsq --verify-only'
```

`gpu/lstsq` selects full-column-rank tall F32/F64/C32/C64 matrices with
32/256 columns and eight RHS columns. Its independent NumPy SVD-based
least-squares oracle has a nonzero residual; fixture tests check the normal
equations and full column rank. The selected public API is
`EagerSessionLinalgExt::lstsq`, whose QR-based implementation creates and
uploads scalar operands internally. It can be numerically verified on CUDA,
but standard setup-free timing is excluded. The current direct
`TensorLinalgExt` has no `lstsq` method. PyTorch/JAX native candidates return
extra residual/rank/singular-value outputs; no matching single-call
solution-only contract is established here. These explicit unsupported cells
are not measured coverage. Wide and rank-deficient problems are outside the
current eager QR-based API contract, not smaller replacements for these cases.

Additional compact AXPBY contracts select `(alpha,beta)=(0.5,0)` (overwrite,
PyTorch `mul(..., out=...)`), `(0.25,0.75)` (PyTorch
`lerp(y,x,0.25,out=y)`), and `(1,1)` (native in-place addition). Each has
F32/F64/C32/C64 cases at 1024/1048576 elements with independent, nonconstant
GPU seeds; complex seeds also have nonzero imaginary values. Parameter
construction, output allocation and seed restoration remain untimed.

Unit coefficients have genuine public JAX output-reuse counterparts:
`jax.ref.addupdate` on preallocated caller-owned Refs, with eager and AOT
compiled paths reported separately. This is a mutable Ref API, not immutable
array donation. In JAX 0.10.1, `block_until_ready(ref)` is a no-op and
`effects_barrier()` does not finish queued Ref GPU writes. Full `ref.get`
snapshots create different buffers and are used only for untimed inspection.
For completion, a separately AOT-compiled helper reads one scalar from every
updated Ref and waits on those device tokens. Prepare and warm each batch-size
helper outside the clock, retain tokens through clock stop, and include their
logical bytes in the 256 MiB retention limit. This synchronization has a fixed
scalar-read cost per destination, which is recorded in raw sampling metadata;
it performs no full output copy or D2H transfer inside timing. GPU1 checks
confirmed all destinations keep their buffer addresses through restoration
and updates, and validated completion against native CUDA context waits.
Ref.addupdate has no scaled-alpha argument; pre-scaling its inputs would
change the timed operation and is not used for other coefficient cases.

Generalized dot adds non-leading batch axes (`lhs[m,b,k]`, `rhs[k,n,b]`)
and multiple contraction axes (`lhs[m,k,l]`, `rhs[l,n,k]`). Direct borrowed
read handles and DotGeneralConfig values are prepared before every batch;
eager uses its public owned dot-general API with prepared configurations.
The native tenferro batch output is `[m,n,b]`, while native PyTorch `bmm`
and JAX `lax.dot_general` return `[b,m,n]`. Prepare PyTorch input permutation
views before timing; preserve every backend's output layout. Normalize batch
axes only on downloaded CPU arrays for numerical verification, with a view
and no timed output transpose/copy. No einsum planning is added to these arms.

Complex vector cases distinguish no conjugation, lhs-only, rhs-only, and
both-conjugated public dot-general calls. Independent real/imaginary fixtures
make lhs-only and rhs-only results observably different. Native dot and vdot
are single-call references for the first three contracts (swap arguments for
rhs-only scalar vdot). Both-conjugated references remain unsupported because
an explicit conjugation composition would change the operation.
