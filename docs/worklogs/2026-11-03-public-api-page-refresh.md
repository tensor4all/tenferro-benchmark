# GPU public-API pages: republish with provenance (2026-11-03)

## Why

`result/nvidia-gpu/gpu/{fft,nn,isolated_ops,structural,linalg_public_api,reuse_contraction,lstsq}.md`
named raw runs from 2026-10-13 (`20261013_*`). Those run directories are not in
the repository (`data/results/*` is gitignored for `nvidia-gpu`) and the pages
recorded no commit, so no published number was tied to a tenferro-rs revision.
The `isolated_ops` reduce rows in particular predated `#2134`/`#2152`, merged
2026-10-14, which split the remaining reduction families over the axis: they
published 119–155× behind PyTorch for a gap that no longer existed at that size.

## Blocker

`scripts/run_gpu_public_api.sh` finishes by folding the family's raw rows into
the routine record contract through `scripts/normalize_gpu_records.py`, which
read `row["suite_id"]`. The public-API writers emit `case_id` rows and never
wrote `suite_id`, so every publish attempt stopped with `KeyError: 'suite_id'`
**after** the measurement had completed. Without `records.jsonl` and
`report.summary.json` no page can be republished from a run, for any of the
seven families.

The suite identity belongs to the run, and `run.yaml` already carries it, so
`fold` now falls back to the run's `suite_id` when a row has none, and a row
that carries one keeps it. No raw file changes shape, no writer or runner
change is needed, and the run record stays the single source of the identity.
`tests.test_normalize_gpu_records` covers the fallback and the rejection when
neither the row nor the run names a suite, and CI now runs that test file.

## Set

One collection, one environment, all seven families in order (`all`), physical
GPU 0 only, host CPU pinned to the free L3 domain `0-7`, 22m00s:

```bash
# inside the repository CUDA devcontainer, idle host, one measurement at a time
BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 \
BENCH_RUNS=15 BENCH_WARMUPS=3 \
  ./scripts/run_gpu_public_api.sh all
uv run python scripts/regenerate_report_summaries.py --profile nvidia-gpu
```

| family | run directory | measured cells |
| --- | --- | ---: |
| `fft` | `data/results/nvidia-gpu/gpu/fft/20261102_205658` | 80 |
| `nn` | `data/results/nvidia-gpu/gpu/nn/20261102_205922` | — |
| `isolated_ops` | `data/results/nvidia-gpu/gpu/isolated_ops/20261102_210120` | — |
| `structural` | `data/results/nvidia-gpu/gpu/structural/20261102_210608` | — |
| `linalg_public_api` | `data/results/nvidia-gpu/gpu/linalg_public_api/20261102_210935` | 616 (204 unsupported, 56 verified-but-excluded) |
| `reuse_contraction` | `data/results/nvidia-gpu/gpu/reuse_contraction/20261102_211407` | — |
| `lstsq` | `data/results/nvidia-gpu/gpu/lstsq/20261102_211841` | — |

## Provenance

* tenferro-rs `433121a7` — `pins/tenferro-rs.rev` is `58fff234`; the collected
  checkout `433121a7` is the pre-merge commit of that pin whose tree is
  identical (`git diff 433121a7 58fff234` is empty).
* tenferro-benchmark `8987033` at collection time.
* NVIDIA A100 80GB PCIe, driver 580.178.04, CUDA runtime 13.0, torch
  2.12.0+cu130, jax/jaxlib 0.10.1.
* Every published `<family>.summary.json` records `tenferro_commit`,
  `target_profile`, `suite_id`, the run timestamp and the hardware, so the
  page's `Raw run:` line resolves to a commit even though the raw directory
  itself stays local-only.

## `isolated_ops` reduce/compare, before and after

Microseconds per operation, median (IQR); published page rows, 15 samples, 3
warmups, 2 ms target batch, 256 MiB retained-output cap.

| case | 20261013 direct | 20261102 direct | factor | PyTorch now | direct ÷ PyTorch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `reduce_max_full_f32_n1048576` | 9844.627 (5.935) | 79.248 (3.172) | 124× | 10.262 | 525.0 → 7.72 |
| `reduce_min_full_f32_n1048576` | 9847.470 (7.957) | 81.678 (0.283) | 121× | 10.255 | 521.8 → 7.96 |
| `reduce_sum_full_f32_n1048576` | 2546.618 (5.038) | 56.374 (3.193) | 45× | 9.652 | 137.9 → 5.84 |
| `reduce_max_full_f64_n1048576` | 12521.546 (6.782) | 85.185 (0.320) | 147× | 11.538 | 280.6 → 7.38 |
| `compare_eq_f32_n1048576` | 33.337 (0.110) | 19.123 (0.152) | 1.7× | 5.414 | 1.1 → 3.53 |

The 100–500× claim in the previous pages was the pre-`#2134` state. The
remaining 5.7–8.0× (52–98 GB/s against PyTorch's 409–727 GB/s) is a real gap and
is filed as `tensor4all/tenferro-rs#2406`, with the issue number recorded on the
eight `reduce_{max,min,sum,prod}_full_{f32,f64}_n1048576` cases in
`data/instances/gpu_isolated_ops.json`.

Other GPU pages (`dense`, `einsum`, `sparse`, `permutation`, `perf_issues`,
`elementwise`, `linalg_*`, throughput) are unchanged: their raw runs are not on
this host, so `regenerate_report_summaries.py --profile nvidia-gpu` skipped them
and left their pages byte-identical.
