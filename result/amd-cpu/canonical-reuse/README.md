# Rejected experiment: borrow already-canonical operands

The draft against tenferro `2bc31a8` borrowed compact non-conjugated canonical
operands instead of repacking them. It preserved the existing two-attempt
provider limit and materialized borrowed inputs if the first canonical request
was unsupported. No new kernel or thread policy was introduced.

A separate diagnostic LM N1 invocation (two paths, one warmup) found 60 compact
operands among 204 packing calls, representing about 5.56 MB per path mean.
The diagnostic patch/log and complete candidate patch (including two new tests
and its pre-collection protocol) are retained in `raw/`.

## Decision: reject, source restored

Three candidate N1/N3 instruction pairs were collected for multiply, GEMM1024
and LM using the same Docker/compiler/provider/strided revision, CPU16, and
explicit 1T settings as `../eager-read-forwarding`. Its three frozen candidate
profiles serve as this experiment's baseline through relative symlinks: this
is **not** a contemporaneous native timing comparison. Baseline code is
`2fd8163`; the candidate patch is against its documentation-only successor.

| Workload | Paired Ir reduction range |
|---|---:|
| multiply control | −0.049% to +0.024% |
| GEMM1024 control | +0.0002% to +0.0019% |
| LM3 12d path mean | +0.020% to +0.030% |

The required LM reduction was >=1%; it failed. Controls passed their <=1%
regression bounds. The complete numerical results and protocol are in
`summary.json`. Do not infer native speedups, bandwidth, or kernel time shares
from these counts, nor conclude that native benefits are impossible.

The corrected draft passed all 543 existing CPU library tests and both new
focused tests (borrow/copy combinations and provider retry outcomes/pool
reclamation). The first draft exposed the existing exactly-one-retry contract;
that was fixed before collection. Broader candidate validation was not completed
because the performance gate rejected it. Both production changes and new tests
were removed from the source worktree; only experimental evidence is retained.

Use the collection loops in `../eager-read-forwarding/reproduce.sh`, with the
saved baseline binary and a binary built from `raw/candidate.patch.gz`, and the
same fixtures and build flags. The actual candidate-only collection used three
repeats, cases listed in `protocol.json`, runs 1/3, warmup 1, CPU16, and
`timeout 150s taskset -c 16 valgrind --tool=callgrind`. The diagnostic interceptor
was absent from measurements. `raw/binaries-threads.txt` records binary hashes
and separately verified OpenBLAS 1T calls for the candidate. Baseline hashes and
thread verification are retained in the referenced earlier evidence directory.

Regenerate with:

```sh
python3 ../eager-read-forwarding/summarize.py .
```

The parser produced identical results before and after lossless compression.
Formal integration and all outstanding goal validation remain pending.
