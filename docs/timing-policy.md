# Benchmark timing policy, revision 2

Standard CPU and GPU results measure steady-state operation execution. Prepare
input values, wrapping, layout conversions, transfers, sessions, runtimes,
thread pools, vendor handles, contraction plans and compiled graphs before the
clock starts. Per-sample gradient reset and replacement inputs are also untimed.
Run at least one untimed priming invocation, including when `BENCH_WARMUPS=0`.

Stop the clock after the operation and completion synchronization, while its
returned outputs are still alive. Release outputs and validate/download values
afterwards. Intrinsic output allocation belongs to allocation-returning APIs;
caller-owned output reuse is a separate comparison. AD eager forward recording
is part of a declared forward/backward operation; building reusable compiled AD
graphs is preparation. AD rows request the primal value and the declared
derivatives; matmul and solve backward rows differentiate both operands, while
linalg JVP/VJP rows differentiate the matrix with the RHS held fixed.

The repository's `AGENTS.md` fixes this rule for future changes. New `run.yaml`
files record `timing_policy.version: 2`. A policy marker records the runner's
contract; it is not a substitute for hardware verification.

## Corrections

- CPU ops: fresh inputs and gradient state are prepared separately from timing
  in Rust, PyTorch and JAX. Runtime reuse is explicit. JAX AD functions are
  prepared/JIT-compiled before sampling. Return all decomposition outputs;
  exclude validation-only EagerTensor materialization.
- Public API: concrete linalg sessions span all samples; outputs survive timer
  stop. Report concrete calls as `tenferro-direct`; the EagerTensor fallbacks for
  `lstsq` and `svd_full` remain `tenferro-eager`. Metadata views prepare replacement
  owners outside the clock.
- FFT: keep direct CPU execution sessions and cached executors outside timing.
  One-shot Tensor/TensorRead APIs create a fresh plan internally, so they are
  skipped by default. The explicit diagnostic opt-in writes `cpu/fft_setup` rows.
- Einsum: prepare contraction paths, expressions and integer-label mappings
  before sampling. Keep runtime inputs separate from compiled programs.
- GPU vendors: keep Ginkgo executor/CSR/input wrappers and CUTLASS input layout
  conversions outside timing. cuTENSOR uses a scoped raw CUDA session around
  planning, verification and all samples; the timer covers execution plus sync.
  The standalone C++ CUTLASS runner reuses initialized launch parameters and
  records host wall time plus device synchronization. Its destination-reuse
  results have their own backend identity and table.
- Permutation: retain outputs until timing ends; GPU formatters separate output
  allocation from destination reuse. Unknown legacy policies stay separate.
  The HPTT crate's one-shot API builds a plan per call and cannot supply an
  equivalent prepared operation; it is recorded as skipped with a reason.
- Small-work: ordinary operation runs exclude fresh-session and setup-only
  routes. `BENCH_INCLUDE_SETUP_DIAGNOSTICS=1` explicitly enables these diagnostic
  routes, which are reported separately. Per-call clocks exclude final output
  destruction; their sum includes clock overhead and is not a single batch
  wall-clock measurement. Intermediate lifetime work within a dependent chain
  remains part of that declared workflow.

## Existing results

The [2026-09-13 M5 Max CPU rerun](m5-cpu-refresh-20260913.md) regenerated all
seven CPU reports with revision 2 at 1 and 4 threads. Its raw runs record
`timing_policy.version: 2` and tenferro `a793c2e9`.

Results collected before this correction, including the earlier M5 refresh
on the same date, use earlier timing scopes. Other target reports remain historical.
Do not use their ratios to assess tenferro operation performance or regressions
against revision 2 measurements. Rerun the affected participants before making
those comparisons; preserve the original raw data for provenance.

## Verification

Run `python tests/test_timing_boundaries.py` in the benchmark Python environment,
the public-API harness tests, and the repository's suite/formatter contract tests.
Build CPU, CUDA and WebGPU runners against the recorded tenferro checkout.
CUDA/vendor execution still requires the NVIDIA devcontainer and GPU hardware;
a successful `cargo check` alone does not validate kernels or vendor bindings.

The correction was checked on M5 Max with tenferro `a793c2e9` (2026-09-13):
34 Rust unit tests; Python timing/lifetime, fixture and selection tests; CPU and
GPU suite/formatter contracts; CPU linalg AD integration; CPU/CUDA/WebGPU builds.
Zero-optional-warmup smoke runs passed all 90 CPU ops rows, supported selected
public-API rows, and cached/eager/trace FFT rows. M5 WebGPU transpose and
`to_contiguous`, plus PyTorch MPS, passed output verification for
`mac_transpose_2d`. These smoke timings are not published benchmark results.
NVIDIA CUDA kernels and the Ginkgo/CUTLASS vendor bindings require follow-up
execution on CUDA hardware; their correctness is not established by Mac builds.
