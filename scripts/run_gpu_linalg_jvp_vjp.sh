#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
TIMESTAMP="${GPU_BENCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-nvidia-gpu}"
SUITE_FILE="$PROJECT_DIR/benchmarks/gpu/linalg_jvp_vjp.yaml"
DEVICE_ORDINAL="${GPU_BENCH_DEVICE:-0}"
RUST_MIN_STACK="${RUST_MIN_STACK:-67108864}"

case "$BENCHMARK_TARGET_PROFILE" in
    nvidia-gpu)
        ;;
    *)
        echo "ERROR: GPU linalg JVP/VJP collection requires BENCHMARK_TARGET_PROFILE=nvidia-gpu." >&2
        exit 1
        ;;
esac

RUN_TIMESTAMP_RFC3339="$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

cd "$PROJECT_DIR"

VALIDATOR=(python3 "$SCRIPT_DIR/validate_benchmark_suite.py")
METADATA=(python3 "$SCRIPT_DIR/collect_run_metadata.py")
FORMATTER=(python3 "$SCRIPT_DIR/format_gpu_linalg_ad_results.py")
PYTHON=(python3)
if command -v uv >/dev/null 2>&1; then
    VALIDATOR=(uv run python "$SCRIPT_DIR/validate_benchmark_suite.py")
    METADATA=(uv run python "$SCRIPT_DIR/collect_run_metadata.py")
    FORMATTER=(uv run python "$SCRIPT_DIR/format_gpu_linalg_ad_results.py")
    PYTHON=(uv run python)
fi

"${PYTHON[@]}" "$SCRIPT_DIR/generate_gpu_linalg_jvp_vjp_yaml.py" >/dev/null
"${VALIDATOR[@]}" "$SUITE_FILE"

suite_id="$(awk '$1 == "suite_id:" { print $2; exit }' "$SUITE_FILE")"
suite_name="${suite_id#gpu/}"
RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/gpu/$suite_name/$TIMESTAMP"
RUN_YAML="$RUN_DIR/run.yaml"
RESULT_JSONL="$RUN_DIR/records.jsonl"
RUST_JSONL="$RUN_DIR/rust_records.jsonl"
PYTHON_JSONL="$RUN_DIR/python_records.jsonl"
REPORT_OUT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/gpu/$suite_name.md"
GPU_BENCH_BACKEND_SLEEP="${GPU_BENCH_BACKEND_SLEEP:-2}"

mkdir -p "$RUN_DIR" "$(dirname "$REPORT_OUT")"
: > "$RESULT_JSONL"

tenferro_dir="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
tenferro_commit=""
if [[ -d "$tenferro_dir/.git" ]] && command -v git >/dev/null 2>&1; then
    tenferro_commit="$(git -C "$tenferro_dir" rev-parse HEAD 2>/dev/null || true)"
fi
metadata_args=(
    --suite-id "$suite_id"
    --target-profile "$BENCHMARK_TARGET_PROFILE"
    --suite-file "$SUITE_FILE"
    --timestamp "$RUN_TIMESTAMP_RFC3339"
    --tenferro-dir "$tenferro_dir"
    --features cuda
    --blas none
    --cuda-device-ordinal "$DEVICE_ORDINAL"
    --output "$RUN_YAML"
)
if [[ -n "$tenferro_commit" ]]; then
    metadata_args+=(--tenferro-commit "$tenferro_commit")
fi
"${METADATA[@]}" "${metadata_args[@]}"

echo "============================================"
echo " GPU linalg JVP/VJP benchmark suite"
echo "============================================"
echo "Suite:    $SUITE_FILE"
echo "Device:   cuda:$DEVICE_ORDINAL"
echo "Target:   $BENCHMARK_TARGET_PROFILE"
echo "Timestamp: $TIMESTAMP"
echo ""

RUST_BIN="$PROJECT_DIR/target/release/benchmark_gpu_linalg_ad"
echo "Building benchmark_gpu_linalg_ad..."
CUBECL_DEBUG_LOG=0 \
CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
RUST_MIN_STACK="$RUST_MIN_STACK" \
LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    cargo build --release --features cuda --bin benchmark_gpu_linalg_ad

echo "Running tenferro-cuda-trace linalg AD benchmarks..."
: > "$RUST_JSONL"
CUBECL_DEBUG_LOG=0 \
CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
RUST_MIN_STACK="$RUST_MIN_STACK" \
LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "$RUST_BIN" "$RUST_JSONL" "$DEVICE_ORDINAL" "" -- "$SUITE_FILE"
cat "$RUST_JSONL" >> "$RESULT_JSONL"
sleep "$GPU_BENCH_BACKEND_SLEEP"

for backend in pytorch-cuda jax-cuda; do
    echo "Running $backend linalg AD benchmarks..."
    chunk="$RUN_DIR/chunk_${backend}.jsonl"
    : > "$chunk"
    "${PYTHON[@]}" "$SCRIPT_DIR/benchmark_gpu_linalg_ad.py" \
        "$chunk" "$DEVICE_ORDINAL" "" "$backend" -- "$SUITE_FILE"
    if [[ -s "$chunk" ]]; then
        cat "$chunk" >> "$RESULT_JSONL"
    fi
    sleep "$GPU_BENCH_BACKEND_SLEEP"
done

if [[ ! -s "$RESULT_JSONL" ]]; then
    echo "No records generated for $suite_id; skipping report." >&2
    exit 1
fi

"${VALIDATOR[@]}" --kind run "$RUN_YAML"
"${VALIDATOR[@]}" --kind result "$RESULT_JSONL"
"${FORMATTER[@]}" "$RESULT_JSONL" --run-metadata "$RUN_YAML" --output "$REPORT_OUT"

echo ""
echo "GPU linalg JVP/VJP benchmark suite complete: $suite_id"
echo "Run YAML: $RUN_YAML"
echo "JSONL:    $RESULT_JSONL"
echo "Report:   $REPORT_OUT"
