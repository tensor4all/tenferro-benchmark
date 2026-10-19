# Benchmark rows format (`tenferro-bench-row/1`)

Each suite keeps its own raw files (CSV tables, per-sample JSONL, nested GPU
records). For reports and automated audits, `scripts/ci/bench_rows.py`
flattens raw runs into one JSONL row per measurement, so a reader handles a
single format:

```bash
uv run python -m scripts.ci.bench_rows <artifact-or-checkout> \
  --profile amd-cpu --since 20261016_174851 --output rows.jsonl
uv run python -m scripts.ci.slow_rows rows.jsonl \
  --json slow-rows.json --markdown slow-rows.md
```

`--since` selects raw runs whose timestamp directory (UTC) is at or after the
given value. Raw runs whose format has no adapter are reported on stderr and
skipped.

When the root carries an `ingestion.json` manifest for the profile (a
collection bundle, see `docs/result-ingestion.md`), only the raw files the
manifest lists are read, with the metadata of their source. Several suites of
one collection can list the same raw file (`cpu/cpu_ops` and
`cpu/linalg_jvp_vjp` share the `cpu_ops` tables); a file with the same digest
is exported once, under the first suite that lists it. Without a manifest,
every `data/results/<profile>/<suite>/<timestamp>/` directory is scanned; an
older artifact also carries the repository's earlier raw runs, so pass the
collecting run's start time.

## Row fields

| field | meaning |
|---|---|
| `schema` | always `tenferro-bench-row/1` |
| `target_profile` | `amd-cpu`, `mac-cpu`, `nvidia-gpu` |
| `suite_id` | workload identity, for example `cpu/public_api` |
| `problem_id` | problem within the suite; unique per suite together with `threads` and `backend` |
| `op` | operation name, used to group related problems |
| `dtype`, `shape` | as recorded by the suite, or `null` |
| `threads` | CPU thread count; `null` on GPU |
| `backend` | for example `tenferro-trace`, `pytorch-cpu`, `hptt`, `tenferro-cuda-eager` |
| `role` | `tenferro` (any `tenferro*` backend), `reference` (another implementation), or `bound` (a lower bound such as `memcpy`, never a reference) |
| `execution_path` | route or API tier when the suite records one, else `null` |
| `status` | `ok`, `unsupported`, `failed` or `skipped` |
| `median_ms`, `iqr_ms` | steady-state time per declared operation; `null` unless `status` is `ok` |
| `note` | unsupported/skip reason or suite note |
| `raw_run`, `timestamp` | raw-run directory relative to the root and its timestamp |
| `tenferro_commit`, `harness_commit` | from the run metadata |
| `blas`, `device` | BLAS implementation and GPU name from the run metadata, or `null` |

A row is `ok` only when the suite reported success and verification passed.
Unsupported, failed and skipped rows are kept so coverage can be reported,
and they are never timing evidence.

## Suite sources

| suite | raw files | notes |
|---|---|---|
| `cpu/einsum` (cpu_ops), `cpu/fft`, `cpu/public_api` | `cpu_ops_t*.csv`, `cpu_fft_t*.csv`, `cpu_public_api_t*.csv` | `problem_id` is `<table suite>/<benchmark>/<dtype>/<shape>` |
| `cpu/permutation` | `*_output_t*.jsonl` | IQR is p75 − p25 |
| `cpu/session_matrix` | `samples_t*.jsonl` | `blas`/`faer` providers become `tenferro-blas`/`tenferro-faer`; `pytorch` becomes `pytorch-cpu`; time is per sample |
| `cpu/small_work` | `samples_t*.jsonl` | per-operation time (`elapsed_ns / iterations`); tenferro-rs only, so no comparison |
| GPU suites | `records.jsonl` | `gpu/perf_issues` and `gpu/permutation` have no adapter yet |

## Slow rows

`scripts/ci/slow_rows.py` compares each `ok` tenferro row with the fastest
`ok` reference row of the same profile, suite, problem and thread count. It
reports a row when the median ratio reaches the threshold (default 1.5) and
`median - IQR` of the tenferro row exceeds `median + IQR` of the reference.
A row whose IQR exceeds its median is below the timer's useful resolution and
is not compared. The markdown output lists the suites with their status
counts, a table grouped by `(suite, op, backend)`, and the highest-ratio rows;
the JSON output carries every finding.
