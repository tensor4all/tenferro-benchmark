# Canonical-copy instruction experiment — shortened at user's request

Original three-pair protocol remains in `protocol.json`. Only repeat1 completed;
the user stopped additional repetitions as unnecessary. Repeat2 was interrupted,
and its owned profiling descendants were explicitly terminated. Partial repeat2
artifacts are preserved but excluded. The original three-pair acceptance gate
was **not completed** and must not be called passed.

Repeat1 whole-process N3−N1 normalization:

| Case | baseline Ir | candidate Ir | reduction |
|---|---:|---:|---:|
| multiply2048² |5,315,180|5,310,838.5|0.0817%|
| GEMM1024 |523,494,572.5|523,494,508.5|0.000012%|
| LM3_12d path mean |950,488,172|837,405,324.5|11.8973%|

These are provisional instruction results, not native speedups. Baseline is
2fd8163/strided17e05ff; candidate636cd499/strided78d5190; shared compiler/provider
and explicit1T as documented by the protocol. Fixtures and baseline provenance
are also recorded in sibling `eager-read-forwarding`. Follow-up native results
in `../native-1t/` are inconclusive due observed L3 interference.

The raw directory additionally preserves CPU/AD integration logs, the initial
BLAS UI-link failure, the successful linked run and Docker's final exitCode0
event after the outer client's timeout. The test linker is included. No UI
snapshots were changed; the worktree manifest was restored after local patches.
