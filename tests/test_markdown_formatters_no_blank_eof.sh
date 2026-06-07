#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if command -v uv >/dev/null 2>&1; then
  PY=(uv run python)
else
  PY=(python3)
fi

assert_clean_eof() {
  local path="$1"
  python3 - "$path" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
data = path.read_bytes()
if not data.endswith(b"\n"):
    raise SystemExit(f"{path} does not end with a newline")
if data.endswith(b"\n\n"):
    raise SystemExit(f"{path} ends with a blank line")
PY
}

cat > "$TMP/cpu.log" <<'LOG'
tenferro-trace benchmark suite
Backend: tenferro-trace
OMP_NUM_THREADS=1
RAYON_NUM_THREADS=1
Strategy: opt_flops
Instance Tensors log10FLOPS log2SIZE Median (ms) IQR (ms) Compile (ms)
bin_matmul_256 2 0 0 1.0 0.1 0.0
LOG

"${PY[@]}" "$ROOT/scripts/format_results.py" "$TMP/cpu.log" > "$TMP/cpu.md"
assert_clean_eof "$TMP/cpu.md"

cat > "$TMP/cpu_ops.csv" <<'CSV'
suite,benchmark,dtype,threads,shape,backend,median_ms,iqr_ms,status
matmul,f64_square,f64,1,2x2,tenferro-eager,1.000,0.100,ok
CSV

"${PY[@]}" "$ROOT/scripts/format_cpu_ops_results.py" "$TMP/cpu_ops.csv" > "$TMP/cpu_ops.md"
assert_clean_eof "$TMP/cpu_ops.md"

cat > "$TMP/gpu.jsonl" <<'JSONL'
{"schema_version":1,"suite_id":"gpu/dense","problem_id":"dense_matmul_f64_16","op":"matmul","backend":"pytorch-cuda","status":"ok","timing":{"median_ms":1.25},"verification":{"status":"passed"},"execution":{"device":"cuda"}}
JSONL

cat > "$TMP/run.yaml" <<'YAML'
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/dense
suite_file: benchmarks/gpu/dense.yaml
timestamp: "1999-01-01T00:00:00+00:00"
YAML

"${PY[@]}" "$ROOT/scripts/format_gpu_results.py" \
  "$TMP/gpu.jsonl" --run-metadata "$TMP/run.yaml" --output "$TMP/gpu.md"
assert_clean_eof "$TMP/gpu.md"

"${PY[@]}" "$ROOT/scripts/format_gpu_results.py" \
  "$TMP/gpu.jsonl" --run-metadata "$TMP/run.yaml" > "$TMP/gpu_stdout.md"
assert_clean_eof "$TMP/gpu_stdout.md"
