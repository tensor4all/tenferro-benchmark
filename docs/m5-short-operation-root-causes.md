# M5 CPU slow-case investigation

This investigation uses tenferro a793c2e95693f053722fbff8d0db25722336c21f.
The original operation timings used isolated public calls. They should not be
interpreted as steady-state shared-session throughput; the replacement contract
is documented in [short-operation-sessions.md](short-operation-sessions.md).

The tiny 4-thread GEMM penalty is predominantly execution-domain admission and
synchronous Rayon worker handoff. Changing Rayon from 1 to 4 threads with BLAS
thread settings held fixed reproduces it; changing only the BLAS settings does
not. `CpuBackend::run_backend_session_cached` constructs/enters a session;
`CpuOperationEntry::enter` and `CpuContext::install` run work through the domain.
Many independent calls made on the caller thread repeat that boundary. A shared
session supplies an already-entered context and amortizes it across operations.
This is not evidence that splitting a 2×2 multiplication across four cores is
useful. PyTorch's installed mm path reaches CPUBlas/BLAS without an explicit
2×2-specific threading threshold in that path; the tiny-call activity probe
shows no appreciable additional worker CPU. Accelerate controls its own BLAS
threads, while tenferro's Rayon domain also hosts copies, elementwise work,
reductions and faer kernels.

QR scalar-loss VJP expands into 27 semantic operations at 2×2. Besides the QR
factorization, it executes transposes, dot products, reductions, triangular
solves and elementwise rules. The 4-thread sampled stacks show repeated waits
below CPU operation entry in these nodes. The 1-thread profile exposes an additional benchmark error: 1527 of 2309
samples below the QR benchmark fall under run_compiled preparation, including
input signatures and metadata. Reusing a CompiledGraph was insufficient; the
runner now calls prepare_compiled outside timing and run_prepared inside.
Allocation, tensor/view metadata and dispatch work also remain. The expansion and per-node
runtime work explain why one fast LAPACK call is not a useful proxy for this
VJP. Profiles do not prove a unique percentage of total cost for each cause.

Small batched solve has an additional kernel implementation cost: the LAPACK
helper loops over batches serially, copies values into small tensor wrappers,
allocates per-matrix pivot/output storage, and calls tiny factorizations/solves.
Trace solve already uses LuFactor plus LuSolvePrepared: it is incorrect to say
trace always refactorizes for the backward pass. Its prepared solve path still
performs copies, pivot application and two triangular solves. The ordinary
Solve transpose rule used by eager AD can reconstruct primal/adjoint solves;
PyTorch's ordinary backward reuses saved LU factors. The sampled batch profile
shows descriptor, copy and allocation work in the solve subtree, consistent
with that source-level explanation.

Maximum reduction over the contiguous first axis reaches strided-rs's generic
sequential reduce_axis traversal (actual pinned strided revision fbd10fa5).
There is no output-axis parallel partitioner on that path. The contiguous-output
fast path is not taken for this input layout, and the inner reduction uses one
NaN-propagating accumulator. Increasing the Rayon budget therefore does not
parallelize this reduction. The much faster Julia result remains a meaningful
reference; the older JAX thread labels do not.

The tanh comparison exposed a separate benchmark error. Even with the old
“1-thread” flags, JAX's optimized HLO partitioned the operation five ways and
used about 4.7 CPU-seconds per wall-second. `PJRT_NPROC=1` eliminates that
partitioning; `PJRT_NPROC=4` produces four partitions. With the exact published
fixture, three corrected process runs put JAX 1-thread near 8.57 ms rather than
1.82 ms. Thus the previously quoted 11.9× gap was not a fair 1-thread comparison.
The residual gap is about 2.5× against the earlier tenferro timing. Tenferro maps
scalar f64 tanh; vectorized compiled math is a plausible remaining factor, but
this investigation has not isolated its contribution with an equivalent-kernel
experiment. It must not be presented as fully proven.

Diagnostic scripts, raw thread sweeps, corrected JAX samples, HLO and sampled
stacks are preserved under
[data/results/mac-cpu/cpu/rootcause/20260913](../data/results/mac-cpu/cpu/rootcause/20260913/).
These are diagnostic experiments, not replacement publication rows. The
`examples/rootcause_graph.rs` executable records semantic operation counts.

Primary implementation references: tenferro's local `crates/tenferro-cpu/src/`
context, provider, backend and exec_session modules; linalg `cpu/linalg/`
LAPACK batch helpers and `ad/rules/mod.rs`; pinned strided-basic `reduce_view.rs`;
[PyTorch LinearAlgebra.cpp at the installed wheel revision](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/aten/src/ATen/native/LinearAlgebra.cpp),
[PyTorch solve backward](https://github.com/pytorch/pytorch/blob/7661cd9c6b841b62b7f411aa52ec51f05457263b/torch/csrc/autograd/FunctionsManual.cpp),
and [OpenXLA PJRT thread-pool sizing](https://github.com/openxla/xla/blob/main/xla/pjrt/utils.cc).
