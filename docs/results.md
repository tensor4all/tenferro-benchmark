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
    cpu/permutation.md
  amd-cpu/
    cpu/einsum.md
    cpu/cpu_ops.md
    cpu/linalg_jvp_vjp.md
    cpu/permutation.md
  nvidia-gpu/
    gpu/dense.md
    gpu/einsum.md
    gpu/elementwise.md
    gpu/sparse.md
    gpu/linalg_jvp_vjp.md
    gpu/linalg_ad_latency.md
    gpu/permutation.md
```

`gpu/linalg_jvp_vjp` holds GPU-sized AD rows; `gpu/linalg_ad_latency` holds the
n=2, 4, 8 single-call latency/overhead rows (device kernel time is a small
fraction there, so it is not a throughput report).

Raw run data is written under `data/results/<target_profile>/<suite_id>/<timestamp>/`:

```text
data/results/mac-cpu/cpu/einsum/<timestamp>/run.yaml
data/results/mac-cpu/cpu/permutation/<timestamp>/run.yaml
data/results/amd-cpu/cpu/einsum/<timestamp>/run.yaml
data/results/nvidia-gpu/gpu/dense/<timestamp>/run.yaml
data/results/nvidia-gpu/gpu/permutation/<timestamp>/run.yaml
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

## Machine-readable summaries

Each maintained report has a machine-readable sidecar written by the same
formatter that writes the Markdown, from the same records:

```text
result/nvidia-gpu/gpu/dense.summary.json
result/mac-cpu/cpu/einsum.summary.json
```

The sidecar uses schema version 1 (`summary-schema-v1`):

```json
{
  "schema_version": 1,
  "target_profile": "nvidia-gpu",
  "suite_id": "gpu/dense",
  "timestamp": "2026-10-16T23:51:16.689453+00:00",
  "tenferro_commit": "2754c8c85dcc35c1b63cb7bbdd94ddfd7ea5746e",
  "hardware": {"gpu": "NVIDIA A100 80GB PCIe"},
  "unit": "ms",
  "rows": [
    {"case": "dense_matmul_f64_3072", "group": "matmul",
     "backend": "tenferro-cuda-eager", "median": 1.23,
     "iqr": 0.01, "status": "ok", "verified": true}
  ]
}
```

Backends keep their report column ids (`tenferro-cuda-eager`,
`tenferro-cuda-trace`, `pytorch-cuda`, `jax-cuda`, `cublaslt`, ...; CPU suites
their own ids and thread counts). `unit` is `ms` for the dense/linalg/permutation
suites and `ns/op` for `*/perf_issues`. Rebuild the sidecars from the local raw
runs (gitignored) with:

```bash
uv run python scripts/regenerate_report_summaries.py
uv run python scripts/regenerate_report_summaries.py --profile nvidia-gpu
uv run python scripts/regenerate_report_summaries.py --dry-run
```

Reports whose raw run is not available locally keep the Markdown report and are
listed without a summary; the dashboard shows a report link only.

## Dashboard

`scripts/generate_dashboard.py` renders the static GitHub Pages dashboard
(`index.html` plus one page per report). It reads the summaries, computes
geometric-mean tenferro/reference ratios per suite, and flags per-case anomalies
(`eager/torch >= 1.5`, `eager/jax >= 3`, `trace/eager >= 1.3`, a non-`ok`
tenferro status, or a `>= 1.25x` regression against the previous committed
summary for the same suite and hardware). The Pages publish workflow checks out
with `fetch-depth: 0` so the regression comparison can walk `git log`.

```bash
uv run python scripts/generate_dashboard.py --result-dir result --output-dir _site
```

The GPU sections publish the repository's A100 results under `result/nvidia-gpu`.
The legacy step that overlaid RunPod artifacts onto the Pages tree has been
removed; RunPod results remain available as workflow artifacts.
