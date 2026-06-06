#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cd "$ROOT"

cat > "$TMP/report.md" <<'MD'
#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) |
|---|---:|---:|---:|
| path_case | 120.000 ± 1.000 | 150.000 ± 1.000 | 80.000 ± 1.000 |
| kernel_case | 25.000 ± 1.000 | 30.000 ± 1.000 | 10.000 ± 1.000 |
| small_gap_case | 0.300 ± 0.010 | 0.350 ± 0.010 | 0.200 ± 0.010 |
| ok_case | **9.000 ± 1.000** | 11.000 ± 1.000 | 10.000 ± 1.000 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-rs trace mode (ms) | tenferro-rs eager mode (ms) | PyTorch Python (ms) |
|---|---:|---:|---:|
| path_case | 79.000 ± 1.000 | 90.000 ± 1.000 | 80.000 ± 1.000 |
| kernel_case | 25.000 ± 1.000 | 30.000 ± 1.000 | 10.000 ± 1.000 |
| small_gap_case | 0.300 ± 0.010 | 0.350 ± 0.010 | 0.200 ± 0.010 |
| ok_case | **9.000 ± 1.000** | 11.000 ± 1.000 | 10.000 ± 1.000 |
MD

mkdir -p "$TMP/instances"
cat > "$TMP/instances/path_case.json" <<'JSON'
{
  "name": "path_case",
  "format_string_colmajor": "ab,ac,ad,ae,af,ag,ah,ai,aj,ak,al,am,an,ao,ap,aq->abcdefghijklmnopq",
  "shapes_colmajor": [[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]],
  "paths": {
    "opt_flops": {"path": [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]], "log2_size": 27.0, "log10_flops": 9.0},
    "opt_size": {"path": [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]], "log2_size": 27.0, "log10_flops": 9.0}
  }
}
JSON
cat > "$TMP/instances/kernel_case.json" <<'JSON'
{
  "name": "kernel_case",
  "format_string_colmajor": "ab,bc->ac",
  "shapes_colmajor": [[2,2],[2,2]],
  "paths": {
    "opt_flops": {"path": [[0,1]], "log2_size": 2.0, "log10_flops": 1.0},
    "opt_size": {"path": [[0,1]], "log2_size": 2.0, "log10_flops": 1.0}
  }
}
JSON
cat > "$TMP/instances/ok_case.json" <<'JSON'
{
  "name": "ok_case",
  "format_string_colmajor": "ab,bc->ac",
  "shapes_colmajor": [[2,2],[2,2]],
  "paths": {
    "opt_flops": {"path": [[0,1]], "log2_size": 2.0, "log10_flops": 1.0},
    "opt_size": {"path": [[0,1]], "log2_size": 2.0, "log10_flops": 1.0}
  }
}
JSON
cat > "$TMP/instances/small_gap_case.json" <<'JSON'
{
  "name": "small_gap_case",
  "format_string_colmajor": "ab,bc->ac",
  "shapes_colmajor": [[2,2],[2,2]],
  "paths": {
    "opt_flops": {"path": [[0,1]], "log2_size": 2.0, "log10_flops": 1.0},
    "opt_size": {"path": [[0,1]], "log2_size": 2.0, "log10_flops": 1.0}
  }
}
JSON

uv run python scripts/analyze_einsum_gaps.py \
  --report "$TMP/report.md" \
  --instances "$TMP/instances" \
  --threshold 1.15 > "$TMP/out.md"

grep -q '| opt_flops | path_case | 1.500 | path_intermediate |' "$TMP/out.md"
grep -q '| opt_flops | kernel_case | 2.500 | kernel_or_executor |' "$TMP/out.md"
grep -q '| opt_flops | small_gap_case | 1.500 | small_absolute_gap |' "$TMP/out.md"
grep -q '| opt_size | ok_case | 0.900 | ok_or_faster |' "$TMP/out.md"
