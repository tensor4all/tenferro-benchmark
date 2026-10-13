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
  bash -lc 'GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3 ./scripts/run_gpu_public_api.sh fft'
```

Repeat the second command sequentially with `nn`, `isolated_ops`, `structural`,
`linalg_public_api`, and `reuse_contraction`. Do not overlap these commands or
other GPU benchmark processes. The runner requires a CUDA container, fetches a
`main` tenferro checkout while preserving pinned checkouts, and rebuilds its
Rust binary before collection. Public GitHub fetches use HTTPS inside the image.

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
* References use NumPy/SciPy double precision on the exact rounded inputs and
  check every output element before and after measurement. Shapes and dtypes
  must match; NaNs fail. Matched masked log-softmax negative infinities are
  accepted. FFT absolute tolerance scales as
  `32 * eps * ceil(log2(n)) * norm2(input) / normalization`, with a `1e-9`
  floor, to account for roundoff in cancellation-heavy frequency bins; the
  per-case manifest records the actual tolerances.

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

JAX has no equivalent output-reuse arm. Axis-wise sum-of-squares lacks a single
matching PyTorch/JAX operation; compositions are not timed as substitutes.
Masked softmax lacks a matching stable PyTorch operation. JAX RMS normalization
and eager tenferro values-only SVD are similarly explicit unsupported arms.
Normalization uses no affine parameters, `eps=1e-5`, and nonempty finite inputs;
pinv uses `rtol=1e-5`; triangular solve is left/lower/non-transposed/non-unit.

This first set covers compact F32/F64 paths, complex FFT/conjugation/contractions,
Bool masks and I64 indices. Non-contiguous `TensorRead` routes are not aliases of
compact routes. Stateful Householder updates, RRQR/rank outputs, LU output
contracts, full-SVD gauge-invariant verification, dynamic slicing, cached
accumulation/einsum destinations and new AD transformations require dedicated
fixtures/contracts. Existing QR/SVD/eigh/solve, permutation, AD and perf-issue
suites remain separate coverage; their presence does not count these remaining
routes as measured. CUDA-unsupported eig/eigvals/full-pivot LU, grouped GEMM and
unsupported read-linalg hooks remain excluded as documented in issue #155.

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
