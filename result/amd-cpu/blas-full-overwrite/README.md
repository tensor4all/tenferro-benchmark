# BLAS full-overwrite: matched-provider instruction evidence

**Local candidate, not a native-speedup claim or merge-ready dependency chain.**

## Results

Same migrated baseline, strided revision, compiler, OpenBLAS and 1T settings;
only tenferro's linked-BLAS full-overwrite implementation differs.

| Workload | Baseline Ir / operation | Candidate Ir / operation | Paired reduction |
|---|---:|---:|---:|
| b32 × 128 × 128 × 128 | 43,895,074 | 39,696,305 | 9.560–9.569% |
| GEMM 1024 | 533,358,520 | 524,962,594 | 1.573–1.574% |
| LM sentence 3, 12d | 1,009,589,326.5 | 968,164,607.75 | 4.103%, **one exploratory pair** |

Binary cases report medians of three independent N3/N1 pairs. Every pair passes
the predeclared instruction gate: BMM reduction >=5%, GEMM control regression
<=1%. LM was an additional check, not a predeclared acceptance workload; its
single pair is not a repeated-measurement performance claim. GEMM kernel self-Ir
is identical in every baseline/candidate pair (33,273,536 for BMM, 512,386,632
for GEMM, and path-mean 444,650,842 for LM).

The candidate removes redundant Rust output initialization when BLAS beta=0
fully overwrites the result. It does not change the GEMM math kernel, add a
batched BLAS API, remove canonical packing, or eliminate eager input copies.
Empty contractions still initialize zeros; nonzero beta stays initialized.
Injected providers do **not** advertise the new witness: their existing unsafe
registration contract promises ABI compatibility rather than full overwrite.

Julia workloads were active, including in CPU16's L3 domain (CPUs16–23).
**No native timings, bandwidth figures, time shares or predicted speedups are
reported.** Timing fields in diagnostic/profiler stdout are intentionally ignored.

## PyTorch and batched GEMM findings

Exact wheel source: PyTorch `0d62256a2b23365f8e1604297eb23a6545102aa8`:

- [LinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/aten/src/ATen/native/LinearAlgebra.cpp): CPU `bmm_out_or_baddbmm_` chooses the native small kernel when `M*N*K < 400`.
- [CPUBlas.cpp](https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/aten/src/ATen/native/CPUBlas.cpp): eligible float64 batches can use MKL batching; the non-MKL implementation loops over ordinary GEMM. This source-built OpenBLAS wheel is non-MKL.
- Installed OpenBLAS **0.3.26** exports no `gemm_batch` symbol. This is a finding
  about this exact provider, not all newer OpenBLAS releases. Tenferro already
  contains feature-gated grouped batching; adding a batch-shaped wrapper around
  the current system-OpenBLAS loop is not vendor batching acceleration.
- `raw/torch-shapes.jsonl` records both LM paths at actual `mm`/`bmm` boundaries.
  Batches of 1100 include 12×12×12, 11×1×11, and 1×1×11 products. The latter two
  are below PyTorch's threshold. Observed BMM operands have nonzero batch
  strides, not shared-operand broadcast strides; they cannot simply be folded
  into one ordinary GEMM without extra work.
- `raw/rust-gemm.log.gz` contains **61,688 calls for the entire diagnostic invocation**,
  not a claimed count per contraction. This older shape trace uses tenferro
  d8759f / strided fbd10f; it is not the candidate performance measurement.

This justifies investigating small independent products separately, but not
copying PyTorch's threshold into Rust without matching-kernel evidence.

## Provenance and validation

- Benchmark source: `976c4b9a459f94086c714ec0f6836978a7bbdfc0`.
- Baseline tenferro: `03ec9804beca51dce6355d1751e10251ff2972d4`.
- Candidate tenferro: `57bee76de1a752782bdd985f332050422f885372`.
- Fixed strided: `17e05ffb168d0f826529ea2c3aceeb4ec851448b`.
- OpenBLAS 0.3.26, LP64/pthreads; Rust 1.98.1 / LLVM22.1.8 for Docker builds;
  Valgrind3.22. CPU16 affinity and explicitly configured 1T, not inferred from
  affinity. `verify_threads.c` separately queried `openblas_get_num_threads()`
  **inside every intercepted dgemm call**, aborting unless it was 1. It was never
  preloaded into measurements. Environment, image/library/binary hashes and
  Cargo features are in `raw/environment.txt` and `raw/final-build-features.log.gz`.
- The preserved pre-edit baseline diff matches committed 03ec980 exactly.
- Combined faer/BLAS: **564 library tests passed**. Faer-only: **543 passed**.
- BLAS-only optimized build: **6 focused tests passed**, covering four scalar
  types, alpha=0, beta rejection, nonzero products, transpose, views, empty
  contractions/extents, batches, invalid storage, and untouched unsupported output.
- Inject-only build checked; its no-uninit-witness regression test passed.
- Optimized focused tests under Memcheck: **zero errors**; leak checking disabled.
  Earlier debug Memcheck also reported zero errors. This is not a leak-free claim.
- Existing pooled-output/nonzero-beta regression tests remain in the library run.
  Benchmark fixture runs themselves are zero-input smoke checks, not the nonzero
  numerical oracle; the focused tests provide that oracle.
- Normal dependency pins still need the unpublished upstream strided commits.
  Tests/builds use the retained local overrides. No push or PR was performed.
  Quiet-host native validation and the broader integration gate remain pending;
  pre-existing strict-Clippy failures from the migration have not been resolved.

## Reproduction and retained evidence

Use the existing OpenBLAS image identified in `raw/environment.txt`, mount the
workspace at the same absolute path (the harness embeds `CARGO_MANIFEST_DIR`),
and point `extern/tenferro-rs` at separate baseline/candidate worktrees and
`extern/strided-rs` at the fixed strided worktree. Do not overwrite a dirty checkout.
`raw/blas-bench-patch.toml` records all six local strided overrides;
`raw/benchmark-Cargo.lock` records dependency resolution. Sequential build command:

```sh
RUSTC_WRAPPER= CARGO_PROFILE_RELEASE_DEBUG=1 cargo build -j 16 \
  --config /tmp/local-patch.toml --release --no-default-features \
  --features system-openblas --bin tenferro-einsum-benchmark
```

Preserve each binary before rebuilding. The build container used
`CARGO_HOME=/tmp/bench-cargo` (mounted host Cargo cache),
`RUSTUP_HOME=/home/vscode/.rustup`, and `HOME=/tmp/bench-home`.
The archived patch is mounted as `/tmp/local-patch.toml`.

Run `reproduce.sh` in the benchmark worktree inside the image, supplying the two
binaries and output directory. Baseline then candidate, never concurrent. It
uses `configure_cpu_thread_env 1`, explicit eager/BLAS, CPU16, one fixed warmup,
and N1/N3. The whole-process difference cancels fixed preparation; small
per-repeat harness/statistics overhead remains, so these are **warmed eager
operation counts, not kernel-only windows**. Identical binary paths share the
harness's strategy cache: divide by 2. LM has two distinct paths: divide by 4.

```sh
python3 result/amd-cpu/blas-full-overwrite/summarize.py
# To inspect a compressed raw profile:
gzip -dc raw/final/PROFILE.callgrind.gz > /tmp/profile.callgrind
callgrind_annotate --auto=no --inclusive=no --threshold=100 /tmp/profile.callgrind
```

Raw profiles, annotations, logs and the source patch are gzip-compressed without
changing their bytes (including original tool-output whitespace).
`summary.json` is generated from the raw profiles; the parser was checked against
both uncompressed and compressed evidence with identical output. Raw logs include
the initial candidate 133ca79 trial (not used for the final table) and a rejected
missing-fixtures launch. The final table uses only `raw/final/`. Non-einsum JSON
fixture warnings are loader skips, identical across variants. Auto source
annotation produced a Valgrind/Perl warning; final annotations disable automatic
source annotation and retain complete function self-cost tables.
