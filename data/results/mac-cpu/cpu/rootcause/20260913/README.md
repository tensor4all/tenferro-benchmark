# Diagnostic evidence provenance

Host: native Apple M5 Max. CPU only. tenferro-rs:
`a793c2e95693f053722fbff8d0db25722336c21f`.

The historical measurements here used benchmark code
`c886a7bcd4e7572b7a53db454987361b1f1a839a`, plus the saved
`examples/rootcause_graph.rs` diagnostic. In particular, those historical trace
samples include `run_compiled` preparation; do not interpret them as measurements
of the newer `run_prepared` runner. Check out that benchmark revision in a
separate worktree when reproducing the historical thread sweep or profiles.
The scripts ran from the benchmark root and wrote temporary files under
`/tmp/tenferro-rootcause`; the outputs have been copied here without editing.

- `thread_sweep.py`: three process repetitions, independently varying Rayon and
  BLAS thread settings at 1/4; three tiny operations and multiple matrix sizes.
- `profile_qr*.py`, `profile_batch.py`: macOS sampled stacks during repeated
  operations. These runs are instrumentation diagnostics, not publication rows.
- `python_probe.py`: process/caller CPU-time diagnostic for PyTorch and JAX.
- `jax_threads*.py`, `run_jax*.sh`: optimized HLO partition and CPU-time evidence
  before/after controlling the PJRT client with `PJRT_NPROC`.
- `final_diagnostics.sh`: exact public-API fixtures for tanh and reduce_max_axis0,
  1/4 threads, three process repetitions, 10 warmups and 51 samples each.
- `graphs.txt`: semantic graph operation counts and empty execution-domain
  entry timing. Zero at 1 thread is below clock resolution, not zero real cost.

The documented residual tanh SIMD explanation is a hypothesis, not a separately
isolated kernel experiment. No rows were removed because of noise. Original
run files may contain filtered rows for unselected operations.
