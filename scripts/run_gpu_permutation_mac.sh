#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

profile="${BENCHMARK_TARGET_PROFILE:-mac-gpu}"
if [[ "$profile" != "mac-gpu" ]]; then
  echo "BENCHMARK_TARGET_PROFILE must be mac-gpu" >&2
  exit 2
fi

tenferro_dir="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
if [[ ! -d "$tenferro_dir" ]]; then
  echo "TENFERRO_RS_DIR is missing: $tenferro_dir" >&2
  exit 2
fi

timestamp="${BENCHMARK_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="$PROJECT_DIR/data/results/mac-gpu/gpu/permutation/$timestamp"
mkdir -p "$run_dir" "$PROJECT_DIR/result/mac-gpu/gpu"
records="$run_dir/records.jsonl"
: > "$records"

device_name="${GPU_BENCH_DEVICE_NAME:-$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip:/ {print $2; exit}')}"
device_name="${device_name:-Apple GPU}"
tenferro_revision="$(git -C "$tenferro_dir" rev-parse HEAD)"
tenferro_dirty=false
if [[ -n "$(git -C "$tenferro_dir" status --porcelain)" ]]; then
  tenferro_dirty=true
fi
export GPU_BENCH_DEVICE_NAME="$device_name"
export TENFERRO_BENCH_REVISION="$tenferro_revision"

uv run python scripts/validate_benchmark_suite.py benchmarks/gpu/permutation-mac.yaml

cat > "$run_dir/run.yaml" <<EOF
schema_version: 1
target_profile: mac-gpu
suite_id: gpu/permutation
suite_file: benchmarks/gpu/permutation-mac.yaml
timestamp: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
tenferro_rs:
  path: "$tenferro_dir"
  commit: "$tenferro_revision"
  dirty: $tenferro_dirty
  features: [webgpu, cpu-faer]
environment:
  hostname: "$(hostname)"
  os: "$(sw_vers -productName) $(sw_vers -productVersion)"
  arch: "$(uname -m)"
  cpu: "$device_name"
  env:
    BENCHMARK_TARGET_PROFILE: mac-gpu
metal:
  device_name: "$device_name"
  runtime: wgpu/Metal
  os_version: "$(sw_vers -productVersion)"
  metal_version: "system Metal runtime"
EOF
uv run python scripts/validate_benchmark_suite.py --kind run "$run_dir/run.yaml"

TENFERRO_BENCH_REVISION="$tenferro_revision" cargo build --release \
  --features webgpu --bin benchmark_gpu_permutation_webgpu
BENCH_OUTPUT="$records" target/release/benchmark_gpu_permutation_webgpu \
  >"$run_dir/rust.stdout" 2>"$run_dir/rust.stderr"

for backend in pytorch-mps jax-metal memcpy-metal-d2d; do
  BENCH_OUTPUT="$records" uv run python scripts/benchmark_gpu_permutation_metal.py "$backend" \
    >"$run_dir/$backend.stdout" 2>"$run_dir/$backend.stderr"
done

uv run python scripts/format_gpu_permutation_results.py "$records" \
  --run-metadata "$run_dir/run.yaml" --output "$run_dir/report.md"
cp "$run_dir/report.md" "$PROJECT_DIR/result/mac-gpu/gpu/permutation.md"
echo "$run_dir"
