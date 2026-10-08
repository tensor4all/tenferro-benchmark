# Latest-main CPU operation MWEs

This independent crate reproduces the operation groups selected from the
2026-10-08 inventory of slow observations without an open issue, at the user's
1.2× reporting threshold. `cases.json` retains the original shape, dtype,
reference, execution paths and observation rows. Those historical ratios are
screening evidence only. The current reproducers use the public APIs of
`tenferro-rs b3f47296244ff7b7c55ac0a75f782cb0835418c1` (main when this campaign
started), after its CPU provider feature/API changes.

Use the Linux CPU devcontainer, which supplies oneMKL and the MKL-backed PyTorch
wheel. No CUDA or CPU affinity is used. First inspect/pull `extern/tenferro-rs`
when it is on main. Preserve a deliberately pinned checkout. Build sequentially
before timing; stop other benchmarks, compilers and tests.

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash -lc '
  export BENCHMARK_TARGET_PROFILE=amd-cpu UV_INDEX_STRATEGY=unsafe-best-match
  source scripts/python_venv.sh
  reset_benchmark_python_venv "$PWD"
  prepare_cpu_benchmark_python_venv "$PWD"'
devcontainer exec --workspace-folder . bash -lc '
  export CARGO_TARGET_DIR="$PWD/target/followup-b3f47296"
  cargo build --release --locked --manifest-path mwe/cpu_followup/Cargo.toml'
```

For a standalone sigmoid comparison at four threads:

```bash
devcontainer exec --workspace-folder . bash -lc '
  source scripts/thread_env.sh
  configure_cpu_thread_env 4
  source scripts/benchmark_host_idle.sh
  assert_benchmark_host_idle
  export LD_LIBRARY_PATH="$MKLROOT/lib:/opt/intel/oneapi/compiler/2026.1/lib:${LD_LIBRARY_PATH:-}"
  target/followup-b3f47296/release/cpu-followup-mwe activation.sigmoid eager-shared 4 15'
devcontainer exec --workspace-folder . bash -lc '
  source scripts/thread_env.sh
  configure_cpu_thread_env 4
  source scripts/benchmark_host_idle.sh
  assert_benchmark_host_idle
  .venv/bin/python mwe/cpu_followup/reference.py activation.sigmoid pytorch 4 15'
```

Change the case ID and use its declared path from `cases.json`. The Rust
`strided-rs` path runs the independent strided-rs materialization reference.
`reference.py CASE_ID jax THREADS RUNS` runs a compiled JAX graph with **dynamic
input arguments**, with compilation and the first completed execution untimed.
`julia --project=. mwe/cpu_followup/reference.jl CASE_ID THREADS RUNS` runs the
Julia reference. Python fixtures preserve the same logical coordinates in
native row-major layout; Rust and Julia use native column-major layout.
Materializing operations produce allocated outputs in both arms. A missing
one-call equivalent is unsupported; for example, dynamic-update-slice uses
JAX rather than a synthesized PyTorch clone-plus-write operation.

The Rust crate uses its `system-mkl` feature to enable upstream's new `blas`
features and link installed oneMKL. This is independent of the root harness's
older feature/path dependencies. It does not establish root harness
compatibility with the new upstream provider structure. `system-openblas` is
available for an explicitly different provider experiment.

All fixtures, sessions, input descriptors, graph construction/compilation,
FFT priming, correctness checks, retention containers and output destruction
are outside the timer. Direct/eager calls reuse an already-entered session.
A sample is one interval containing many operations, targeting 10 ms, bounded
by 512 MiB of retained outputs (and retained owned inputs for metadata).
Intrinsic output allocation and required native completion stay inside.
JSON contains batch durations, operation counts, signatures and the scope.
The 512 MiB budget refers to retained tensor payloads; one operation is still
required when a single output is larger (the 2 GiB rotation). Metadata also
budgets its owned inputs and descriptor storage. Activation outputs are
additionally checked element-by-element against scalar f64 formulas, and
permutation outputs against a full independent odometer oracle.
The collector checks shape, aggregates and about 130 deterministic output
probes against the independent implementation; metadata descriptors must
match exactly. This is a cross-implementation numerical check, not an exhaustive
floating-point equality proof.

Metadata view reproducers use smaller extents to allow memory-bounded long
batches of owned inputs. Their inputs are allocated before each interval and
outputs retain ownership until it ends. They measure metadata only; their
numbers do not confirm the original large-fixture measurements.

`Runtime::run_prepared` includes internal admission/session work and exposes
no public borrowed execution-session entry point at the pinned revision.
The collector explicitly records the selected trace paths as scope-ineligible
for standard operation performance evidence. The binary retains a separately
labelled graph-call diagnostic path; it is not mixed into operation ratios.

For the whole sequential campaign, the host-side driver writes an immutable
configuration **before any timing**, then checks source revisions on restart:

```bash
python3 mwe/cpu_followup/collect.py --container CONTAINER_ID \
  --output data/results/amd-cpu/cpu/followup_12x/RUN_TIMESTAMP --phase scan
python3 mwe/cpu_followup/collect.py --container CONTAINER_ID \
  --output data/results/amd-cpu/cpu/followup_12x/RUN_TIMESTAMP --phase confirm
```

The scan uses three samples at 1T/4T. For each operation that remains a suspect,
confirmation chooses its strongest screened thread condition, uses 15 samples
per process, four A/A pairs, then four balanced comparison pairs ordered AB,
BA, BA, AB. The statistic is the median of round median ratios. A claim requires
at least 1.2×, A/A median ratio within 10% of unity, and every sample set CoV
at most 20%; otherwise the result is below threshold or inconclusive. The
relative threshold comes from the user; absolute threshold is zero so that
short metadata operations are not filtered by an arbitrary time floor.

Raw campaign evidence belongs under `data/results/amd-cpu/cpu/`; the maintained
report belongs under `result/amd-cpu/cpu/`. Read the report for confirmed issue
links and negative/inconclusive dispositions. Never interpret a scan ratio as
a confirmed defect.

CPU PyTorch FFT is not an operation-only reference: at its recorded source
revision `_exec_fft` constructs and commits a DFTI descriptor on every call
([source](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/mkl/SpectralOps.cpp#L490)).
The maintained FFT MWE instead uses `reference.py CASE_ID mkl-dfti THREADS RUNS`.
It calls public oneMKL DFTI through ctypes, prepares/commits one descriptor and
completes a priming transform before sampling. Only a fresh host output
allocation (`torch.empty`) and DftiComputeForward/Backward are timed. This is
a **cached oneMKL reference**, not a PyTorch FFT row. Native completion is
synchronous. Its explicit thread limit is recorded in the JSON provider field.

Julia primes the exact typed batch-loop specialization, including output
assignment, before every interval. Each output escapes to an observable root
after the clock, held until the next untimed setup. This prevents JIT setup
from selecting a one-operation batch for very short metadata calls and avoids
dead-result elimination.

For stricter confirmation or noisy cases, `--balanced-aa` uses AB/BA/BA/AB
for A/A as well as the comparison, and additionally requires **every** A/A
round ratio within 10% of unity. `--target-ns 50000000` selects a 50 ms batch
target. These choices are fixed in a new immutable declaration before that
phase's measurements; the 1.2× reporting and 20% sample-CoV bounds do not change.
Never overwrite an earlier declaration when changing a phase.

The transpose metadata repro uses 2×2 owned inputs. Metadata batches may retain
up to two million operations, still bounded by the 512 MiB input/descriptor
budget, to give Julia a longer interval for its specialized fixed permutation.
