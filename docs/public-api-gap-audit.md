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
