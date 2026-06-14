# Result Layout and Metadata

`suite_id` identifies the workload. `target_profile` identifies the hardware
profile where the suite was run.

Current target profiles:

- `mac-cpu`
- `amd-cpu`
- `nvidia-gpu`

Latest human-facing reports are kept under `result/<target_profile>/...`:

```text
result/
  mac-cpu/
    cpu/einsum.md
    cpu/cpu_ops.md
    cpu/linalg_jvp_vjp.md
  amd-cpu/
    cpu/einsum.md
    cpu/cpu_ops.md
    cpu/linalg_jvp_vjp.md
  nvidia-gpu/
    gpu/dense.md
    gpu/einsum.md
    gpu/sparse.md
```

Raw run data is written under `data/results/<target_profile>/<suite_id>/<timestamp>/`:

```text
data/results/mac-cpu/cpu/einsum/<timestamp>/run.yaml
data/results/amd-cpu/cpu/einsum/<timestamp>/run.yaml
data/results/nvidia-gpu/gpu/dense/<timestamp>/run.yaml
```

Historical report files are not maintained. Use git history to inspect older
latest reports.

`run.yaml` records:

- `target_profile`
- `suite_id`
- suite file path
- timestamp
- tenferro-rs path, commit, dirty state, and features
- host CPU/OS metadata
- controlled thread environment variables
- BLAS provider for tenferro
- PyTorch/JAX provider metadata
- CUDA metadata for GPU runs where available

The benchmark repository commit is intentionally not stored in `run.yaml`; the
result file's git history already records it. A dirty `extern/tenferro-rs`
checkout is valid for benchmark runs. `tenferro_rs.dirty` records that state
alongside the measured tenferro commit.
