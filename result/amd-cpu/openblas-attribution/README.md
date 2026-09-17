# Historical OpenBLAS instruction attribution

`raw.tar.gz` preserves the original `openblas-instructions/` and
`openblas-instructions-continued/` diagnostic directories, including profiles,
commands, build/source identities, parsers and logs. Compiled `.so` helpers are
excluded; their source and reproduction instructions are retained.

From the repository root:

```sh
mkdir -p data/results/amd-cpu/diagnostics
tar -xzf result/amd-cpu/openblas-attribution/raw.tar.gz \
  -C data/results/amd-cpu/diagnostics
```

See `../cpu/openblas_instruction_diagnostics.md` for the historical analysis.
Later implementation experiments have their own raw data beside their reports.
Instruction counts are not elapsed-time evidence; the subsequent native1T
comparisons were inconclusive and no new measurements were made for PRs.
