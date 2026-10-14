# GPU A/A and paired confirmation

Run `scripts/benchmark_gpu_paired.py` inside the repository CUDA devcontainer.
It builds the public performance-issue runner with explicit `native,cuda`
features into a revision-specific target directory, records binary/lockfile
hashes, then collects sequential processes in balanced AB/BA rounds. Physical
GPU0 is the default timing card; each process sees its UUID as CUDA ordinal zero.
One CPU worker is configured and no affinity is pinned. The GPU must be idle at
arm boundaries; busy-device bypass is not supported.

Start with A/A using the same library build in both arms. The default selection
is the ten routine latency controls: n16/64 fixed/mixed/fresh whole sequences
and 16/32768-element shared/per-call allocation. Source suite omissions remain
explicit in each raw status file; this focused campaign does not establish
complete coverage of all GPU performance-issue cases.

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc '.venv/bin/python scripts/benchmark_gpu_paired.py aa \
    --baseline extern/tenferro-rs --rounds 4 --warmups 3 --runs 15'
```

Use a worktree whose `extern/tenferro-rs` is a symlink when building different
library checkouts. The builder temporarily repoints that symlink and restores
it; real checkouts are never replaced. Main library checkouts are refreshed
before building; detached baseline/candidate pins are preserved. Do not run
other benchmark/compiler processes while samples are collected.

After inspecting A/A noise, copy `benchmarks/gpu/confirmation.yaml` to a file
beside the raw runs or outside tracked source. Fill every required value and
set `status: declared` **before any candidate collection**. Declare the same
harness, case IDs/order, manifest, coverage, repetitions and GPU identity used
for A/A. `host.cuda` contains device_uuid, device_name, driver_version and
runtime_version from the A/A arm metadata. Choose effect/noise thresholds from
A/A and the change that matters, never candidate results. Keep the harness
unchanged between A/A and confirmation; do not commit the declaration into it
between those campaigns.

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc '.venv/bin/python scripts/benchmark_gpu_paired.py confirm \
    --baseline extern/baseline-pin --candidate extern/candidate-pin \
    --config /tmp/gpu-confirmation.yaml \
    --aa-dir data/results/nvidia-gpu/gpu/perf_issues_aa/<timestamp>'
```

Confirmation freezes the declaration and records its hash before sampling.
Both arms reuse their freshly built immutable binaries. Setup, tracing/JIT,
fixtures, native handles and numerical validation remain outside operation
clocks. Shape-stream samples time the entire controlled sequence with required
planning and native synchronization; counter runs are excluded. Outputs remain
alive through clock stop. Every arm retains its samples, exact sequence/cache
contract, full metadata and global selection status. Per-case process/validation
wall time is additional workflow metadata outside operation clocks.

The detector uses the existing median of round ratios with predeclared relative
and absolute thresholds, arm CoV and A/A-spread limits. It does not report a
p-value or statistical significance. Missing/failed/unmeasured cases, changed
hardware/features or operation/cache/sequence contracts invalidate comparison.
Noisy, incomplete or insufficiently characterized comparisons are inconclusive;
a scan is never a final performance conclusion. A same-revision confirmation
is a control, not a speedup claim.

Raw campaigns live under `data/results/nvidia-gpu/gpu/perf_issues_{aa,paired}/`;
latest reports are `result/nvidia-gpu/gpu/perf_issues_{aa,paired}.md`. Expected
confirmation cases are the declared subset. Unselected global suite cases are
never rewritten as covered, and any missing declared subset case is a failure.
