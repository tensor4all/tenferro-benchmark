# Result Layout and Metadata

`suite_id` identifies the workload. `target_profile` identifies the hardware
profile where the suite was run.

Current target profiles:

- `zen5-cpu` (latest CPU generation; pinned 1/4/8-thread rows)
- `nvidia-gpu` (latest GPU generation)
- `mac-cpu`, `amd-cpu`, `linux-cpu`, `mac-gpu` (older generations: kept in `result/`
  and git history, not linked from the README or the Pages site)

Latest human-facing reports are kept under `result/<target_profile>/...`:

```text
result/
  zen5-cpu/
    cpu/einsum.md
    cpu/cpu_ops.md
    cpu/linalg_jvp_vjp.md
    cpu/permutation.md
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

A row may also carry `"reason": "..."` when a backend is `unsupported` for
a recorded harness-policy reason (for example a backend with no equivalent
public operation). The dashboard treats those as neutral
"not measured" rows rather than anomalies; an `unsupported` row without a
recorded reason stays an anomaly.

Backends keep their report column ids (`tenferro-cuda-eager`,
`tenferro-cuda-trace`, `pytorch-cuda`, `jax-cuda`, `cublaslt`, ...; CPU suites
their own ids and thread counts). `unit` is `ms` for the dense/linalg/permutation
suites and `ns/op` for `*/perf_issues`. CPU pages and their sidecars are
published from explicitly selected run directories, and every maintained page is
checked against them:

```bash
uv run python scripts/regenerate_report_summaries.py --publish \
    data/results/zen5-cpu/cpu/perf_issues/20261028_101500
uv run python scripts/regenerate_report_summaries.py --check
```

`--publish` renders the selected runs with `scripts/render_run.py` and writes
`result/<profile>/cpu/<suite>.md` plus its `.summary.json`; the summary names the
source runs and their recorded conditions. `--check` re-renders every page of the
maintained profiles from those sources and fails on a byte difference; pages
whose raw records are not in git yet are listed in
`benchmarks/unverified-pages.txt` and reported as UNVERIFIED.

GPU sidecars are rebuilt from the local raw runs (gitignored) with:

```bash
uv run python scripts/regenerate_report_summaries.py
uv run python scripts/regenerate_report_summaries.py --profile nvidia-gpu
uv run python scripts/regenerate_report_summaries.py --dry-run
```

Reports whose raw run is not available locally keep the Markdown report and are
listed without a summary; the dashboard shows a report link only.

## Dashboard

`scripts/generate_dashboard.py` renders the static GitHub Pages dashboard
(`index.html` plus one page per report). It reads the summaries and computes
geometric-mean tenferro/reference ratios from one per-suite mapping table:

- Operation suites default to `vs PyTorch` and `vs JAX` over the tenferro
eager/trace arms.
- `gpu/permutation` uses `vs PyTorch` and `vs cuTENSOR` over the
`tenferro-cuda-transpose`, `to_contiguous`, and `destination-reuse` arms.
- `gpu/perf_issues` uses `vs reference`, pairing each tenferro case with its
`reference_for` arm from `data/instances/gpu_perf_issues.json`; the suite page
lists the worst ratios.
- `gpu/sparse` has no tenferro column and stays `-`.

Anomaly flags are `eager/torch >= 1.5`, `eager/jax >= 3`, `trace/eager >= 1.3`,
a genuine tenferro status anomaly (failed/error/verification failure/missing or
an `unsupported` row without a recorded reason), or a `>= 1.25x` regression
against the previous committed summary for the same suite and hardware. Policy
`unsupported` rows are counted separately in the neutral "not measured" column.
The Pages publish workflow checks out with `fetch-depth: 0` so the regression
comparison can walk `git log`.

```bash
uv run --no-project --python 3.12 --with-requirements scripts/dashboard-requirements.txt \
  python scripts/generate_dashboard.py --result-dir result --output-dir _site --git-repo .
```

The collection workflows upload raw bundles. `ingest-results.yml` validates one
explicit successful main run and commits accepted reports, summary sidecars
and provenance. `publish-results.yml` only renders these committed results and
deploys Pages; it has no artifact download or run-discovery stage. Renderer-only
changes publish directly. Ingestion explicitly triggers Pages after its bot
commit. The Linux nightly keeps the canonical `amd-cpu` profile; it is not
copied to `linux-cpu`.

Suite pages link the exact accepted run and raw artifact, show its API-recorded
expiry, and expose the provenance/digests JSON. Raw artifacts can expire without
changing the committed report or causing a fallback to another run. Legacy
reports remain readable with an explicit missing-provenance label. Rendering
uses neither the network nor filesystem modification dates for result data.
See [the ingestion contract and recovery procedure](result-ingestion.md).
