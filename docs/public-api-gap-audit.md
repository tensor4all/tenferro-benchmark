# Public API gap audit

The public session surface is checked against benchmark source evidence with:

```bash
uv run python scripts/audit_public_api_gaps.py
uv run python scripts/audit_public_api_gaps.py --check
```

The audit reads `session_ext.rs` and `typed_session_ext.rs` from the current
`extern/tenferro-rs` checkout. It deduplicates typed/untyped spellings and
searches the benchmark manifests, Rust runner, and PyTorch/JAX/Julia runners.
`--check` fails when a newly added public session method has no evidence.

Evidence is only the inventory guard. A real comparison still needs a case in
the Rust runner and equivalent PyTorch, JAX, and Julia arms, with setup outside
the timed region. The current newly discovered cases are
`reduce_sum_squares_axis0`, `masked_log_softmax_axis1`, and `where_select` in
`cpu/elementwise_reduction`.

Compact Householder QR, four-output rank-revealing QR (full and deficient
rank), and caller-output solve now have dedicated cases. Collect these and
the three session cases as a focused report inside the current CPU
devcontainer image:

```bash
npx --yes @devcontainers/cli exec --workspace-folder . bash -lc '
  export TENFERRO_CPU_FEATURES=blas-mkl
  export BENCHMARK_TARGET_PROFILE=amd-cpu
  export PUBLICATION_GATE_PROFILE=full
  export PUBLIC_API_CANDIDATES_ONLY=1
  ./scripts/run_cpu_public_api.sh 1 4'
```

Focused collection writes `cpu/public_api_candidates` raw runs and
`result/<target_profile>/cpu/public_api_candidates.md`; it does not overwrite
the complete Public API report. Candidate rows batch sixteen operations per
sample, with retained outputs and normalized per-operation durations; selection
is limited to one operation because its output already occupies 256 MiB.
Fixture creation, session entry, JIT initialization and numerical
validation remain outside every clock, including zero-warmup runs.

Missing direct operations remain explicit unsupported rows: square-plus-sum
and mask-plus-log-softmax compositions are not standard operation comparisons.
JAX pivoted QR has three outputs while tenferro RRQR also returns numerical
rank. Compiled trace currently has no public borrowed-session execution API;
its per-call session creation is excluded from these standard operation rows.
The initial follow-ups are tracked individually in issues #133–#138.

The thirteen APIs tracked by #139–#151 now have standard F32 direct/eager cases
at 1024x64 and 4096x256 and native PyTorch/JAX counterparts where available.
They are also included in an ordinary full Public API collection. Their
focused refresh uses the same command above with
`PUBLIC_API_FOLLOWUPS_ONLY=1` in place of `PUBLIC_API_CANDIDATES_ONLY=1` and writes
`result/<target_profile>/cpu/public_api_followups.md`. Existing AMD perf_issues
results remain intact. Normalization is explicitly non-affine: JAX's public
`standardize` provides LayerNorm and, with `mean=0` and stable variance, RMSNorm.
The selected softplus fixture stays below PyTorch's threshold transition.
