#!/usr/bin/env bash
# Run gpu/tensornetwork once per backend (sequential) and emit a combined report.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-nvidia-gpu}"
DEVICE_ORDINAL="${GPU_BENCH_DEVICE:-0}"
TIMESTAMP="${GPU_BENCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_FILE="${GPU_BENCH_SUITE:-benchmarks/gpu/tensornetwork.yaml}"
BACKENDS="${GPU_BENCH_BACKENDS:-tenferro-cuda-eager,tenferro-cuda-trace,pytorch-cuda,jax-cuda}"
RUST_MIN_STACK="${RUST_MIN_STACK:-67108864}"

RESULTS_ROOT="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
SUITE_ID="$(awk '$1 == "suite_id:" { print $2; exit }' "$SUITE_FILE")"
SUITE_NAME="${SUITE_ID#gpu/}"
RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/gpu/$SUITE_NAME/$TIMESTAMP"
RUN_YAML="$RUN_DIR/run.yaml"
RESULT_JSONL="$RUN_DIR/records.jsonl"
REPORT_OUT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/gpu/$SUITE_NAME.md"
MARKDOWN_OUT="$RUN_DIR/report.md"

PYTHON=(python3)
VALIDATOR=(python3 "$SCRIPT_DIR/validate_benchmark_suite.py")
FORMATTER=(python3 "$SCRIPT_DIR/format_gpu_results.py")
METADATA=(python3 "$SCRIPT_DIR/collect_run_metadata.py")
if command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run python)
    VALIDATOR=(uv run python "$SCRIPT_DIR/validate_benchmark_suite.py")
    FORMATTER=(uv run python "$SCRIPT_DIR/format_gpu_results.py")
    METADATA=(uv run python "$SCRIPT_DIR/collect_run_metadata.py")
fi

mkdir -p "$RUN_DIR" "$(dirname "$REPORT_OUT")"
: > "$RESULT_JSONL"

"${VALIDATOR[@]}" "$SUITE_FILE"

resolve_git_commit() {
    local checkout_dir="$1"
    if [[ -d "$checkout_dir/.git" ]] && command -v git >/dev/null 2>&1; then
        git -C "$checkout_dir" rev-parse HEAD 2>/dev/null || true
    fi
}

tenferro_dir="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
tenferro_commit="$(resolve_git_commit "$tenferro_dir")"
meta_args=(
    --suite-id "$SUITE_ID"
    --target-profile "$BENCHMARK_TARGET_PROFILE"
    --suite-file "$SUITE_FILE"
    --timestamp "$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"
    --tenferro-dir "$tenferro_dir"
    --features cuda
    --blas none
    --output "$RUN_YAML"
)
if [[ -n "$tenferro_commit" ]]; then
    meta_args+=(--tenferro-commit "$tenferro_commit")
fi
"${METADATA[@]}" "${meta_args[@]}"

IFS=',' read -r -a BACKEND_LIST <<< "$BACKENDS"
RUST_BIN="$PROJECT_DIR/target/release/benchmark_gpu_rust"
RUST_BUILT=0

for backend in "${BACKEND_LIST[@]}"; do
    echo ""
    echo ">>> Backend: $backend"
    chunk="$RUN_DIR/chunk_${backend}.jsonl"
    : > "$chunk"

    if [[ "$backend" == "tenferro-cuda-eager" || "$backend" == "tenferro-cuda-trace" ]]; then
        if [[ "$RUST_BUILT" == "0" ]]; then
            echo "  Building benchmark_gpu_rust..."
            CUBECL_DEBUG_LOG=0 \
            CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
            RUST_MIN_STACK="$RUST_MIN_STACK" \
            LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
                cargo build --release --features cuda --bin benchmark_gpu_rust
            RUST_BUILT=1
        fi
        CUBECL_DEBUG_LOG=0 \
        CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
        RUST_MIN_STACK="$RUST_MIN_STACK" \
        LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
            "$RUST_BIN" "$chunk" "$DEVICE_ORDINAL" "" "$backend" -- "$SUITE_FILE"
    else
        "${PYTHON[@]}" "$SCRIPT_DIR/benchmark_gpu_python.py" \
            "$chunk" "$DEVICE_ORDINAL" "" "$backend" -- "$SUITE_FILE"
    fi

    if [[ -s "$chunk" ]]; then
        cat "$chunk" >> "$RESULT_JSONL"
    fi

    # Give the GPU time to release memory before the next backend.
    sleep 2
done

if [[ ! -s "$RESULT_JSONL" ]]; then
    echo "No records generated." >&2
    exit 1
fi

"${VALIDATOR[@]}" --kind run "$RUN_YAML"
"${VALIDATOR[@]}" --kind result "$RESULT_JSONL"
"${FORMATTER[@]}" "$RESULT_JSONL" --run-metadata "$RUN_YAML" --output "$MARKDOWN_OUT"
cp "$MARKDOWN_OUT" "$REPORT_OUT"

echo ""
echo "Combined tensornetwork benchmark complete"
echo "Run YAML: $RUN_YAML"
echo "JSONL:    $RESULT_JSONL"
echo "Report:   $REPORT_OUT"
