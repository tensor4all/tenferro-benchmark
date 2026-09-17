# Diagnostic: vector-shaped GEMM through the same OpenBLAS GEMV

Hypothesis recorded before performance measurement: the LM call trace contains
35,200 vector-shaped calls among 61,688 GEMM calls (whole historical invocation,
not one contraction and not a time fraction). Test exact m=1 or n=1 routing to
CBLAS GEMV, with no arbitrary size threshold, new numerical kernel or batching
wrapper. The inspected OpenBLAS v0.3.26 `interface/gemm.c` has no corresponding
general GEMV dispatch. This is an LD_PRELOAD diagnostic, not production code.

Baseline: tenferro636cd499 + strided78d5190, `/tmp/canonical-copy-candidate`.
Candidate: identical binary plus `gemv_probe.so` built with cc -O3, same shared
OpenBLAS0.3.26. Explicit provider/kernel1T and CPU16. No MKL changes.

Before collection: one N1/N3 instruction pair for LM3_12d, multiply2048² and
GEMM1024; compare against the completed r1-candidate profiles of
`canonical-copy-dispatch`. LM hypothesis threshold >=5% reduction, both controls
<=1% regression. Prior baseline profiles are reused, not contemporaneous native
samples. No extra repetition series. Intrinsic operation allocations remain in
whole-process differencing; fixed initialization cancels. Vendor GEMM self-Ir
is expected to change because vector cases now execute GEMV. Output smoke tests
alone do not establish correctness: `check_probe.py` independently checks96
nonzero, padded-leading-dimension, transpose, alpha/beta cases, untouched holes,
source preservation, and actual OpenBLAS thread count1.

Native acceptance is separate and requires quiet-host measurements. Julia jobs
are still active; do not interpret these diagnostics as measured native speedup.
If the hypothesis fails, retain the negative evidence without production edits.

## Outcome

The96 nonzero checks passed. One instruction pair gave LM -4.3004%, multiply
-0.0637%, and GEMM1024 effectively unchanged. The >=5% instruction hypothesis
therefore **failed**; no production dispatch edit was made. Subsequent native
experiments are in `../native-1t/`: apparent incremental GEMV savings were about
2.4–3.1%, but both complete suites were invalidated by observed L3 interference.
Neither result is accepted native-speedup evidence.
