#!/usr/bin/env bash
set -euo pipefail

# GPU throughput-mode collection for tenferro-rs CUDA trace (tenferro-rs #2087).
#
# Throughput mode uses the deferred prepared API: each timed sample enqueues K
# deferred runs (`Runtime::run_prepared_deferred`) in one caller-held CUDA
# session and pays one retirement barrier in a single `finish_prepared`, so a
# sample's per-call time is `sample_ms / K`. It is a separate regime from every
# per-call latency suite: the rows carry their own backend identity
# (`tenferro-cuda-trace-deferred-batch`), their own suite ids
# (`gpu/dense_throughput`, `gpu/linalg_ad_throughput`) and their own reports.
# Never average, ratio or summarise a throughput row together with a latency row.
#
# Usage:
#   BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_throughput.sh
#   GPU_BENCH_DEFERRED_BATCH_K=16 ./scripts/run_gpu_throughput.sh

[[ -f /.dockerenv && -d /usr/local/cuda ]] || {
    echo 'Use the repository CUDA devcontainer for GPU throughput collection.' >&2
    exit 1
}
export GPU_BENCH_VERIFY_ONLY="${GPU_BENCH_VERIFY_ONLY:-0}"
default_device=0
[[ "$GPU_BENCH_VERIFY_ONLY" == 1 ]] && default_device=1
export GPU_BENCH_PHYSICAL_DEVICE="${GPU_BENCH_DEVICE:-$default_device}"
export GPU_BENCH_DEFERRED_BATCH_K="${GPU_BENCH_DEFERRED_BATCH_K:-64}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
GPU_BENCH_DEVICE_UUID="$(nvidia-smi -i "$GPU_BENCH_PHYSICAL_DEVICE" --query-gpu=uuid --format=csv,noheader)"
[[ "$GPU_BENCH_DEVICE_UUID" == GPU-* && "$GPU_BENCH_DEVICE_UUID" != *$'\n'* ]] || exit 1
export CUDA_VISIBLE_DEVICES="$GPU_BENCH_DEVICE_UUID" GPU_BENCH_DEVICE=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="$PROJECT_DIR/data/results"
[[ "$GPU_BENCH_VERIFY_ONLY" == 1 ]] && RESULTS_ROOT="$PROJECT_DIR/data/verification"
REPORTS_DIR="$PROJECT_DIR/result"
TIMESTAMP="${GPU_BENCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-nvidia-gpu}"
DENSE_SUITE="${GPU_BENCH_DENSE_THROUGHPUT_SUITE:-$PROJECT_DIR/benchmarks/gpu/dense_throughput.yaml}"
AD_SUITE="${GPU_BENCH_AD_THROUGHPUT_SUITE:-$PROJECT_DIR/benchmarks/gpu/linalg_ad_throughput.yaml}"
RUST_MIN_STACK="${RUST_MIN_STACK:-67108864}"

case "$BENCHMARK_TARGET_PROFILE" in
    nvidia-gpu) ;;
    *)
        echo "ERROR: GPU throughput collection requires BENCHMARK_TARGET_PROFILE=nvidia-gpu." >&2
        exit 1
        ;;
esac

RUN_TIMESTAMP_RFC3339="$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

cd "$PROJECT_DIR"
if [[ "$(git -C extern/tenferro-rs branch --show-current)" == main ]]; then
    git -C extern/tenferro-rs -c 'url.https://github.com/.insteadOf=git@github.com:' pull --ff-only
fi

VALIDATOR=(python3 "$SCRIPT_DIR/validate_benchmark_suite.py")
METADATA=(python3 "$SCRIPT_DIR/collect_run_metadata.py")
DENSE_FORMATTER=(python3 "$SCRIPT_DIR/format_gpu_results.py")
AD_FORMATTER=(python3 "$SCRIPT_DIR/format_gpu_linalg_ad_results.py")
PYTHON=(python3)
if command -v uv >/dev/null 2>&1; then
    VALIDATOR=(uv run python "$SCRIPT_DIR/validate_benchmark_suite.py")
    METADATA=(uv run python "$SCRIPT_DIR/collect_run_metadata.py")
    DENSE_FORMATTER=(uv run python "$SCRIPT_DIR/format_gpu_results.py")
    AD_FORMATTER=(uv run python "$SCRIPT_DIR/format_gpu_linalg_ad_results.py")
    PYTHON=(uv run python)
fi

"${VALIDATOR[@]}" "$DENSE_SUITE" "$AD_SUITE"

echo "============================================"
echo " GPU throughput mode suite (deferred batch)"
echo "============================================"
echo "Dense suite: $DENSE_SUITE"
echo "AD suite:    $AD_SUITE"
echo "Batch K:     $GPU_BENCH_DEFERRED_BATCH_K"
echo "Device:      cuda:$GPU_BENCH_DEVICE (physical $GPU_BENCH_PHYSICAL_DEVICE)"
echo "Target:      $BENCHMARK_TARGET_PROFILE"
echo "Timestamp:   $TIMESTAMP"
echo ""

tenferro_dir="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
tenferro_commit=""
if [[ -d "$tenferro_dir/.git" ]] && command -v git >/dev/null 2>&1; then
    tenferro_commit="$(git -C "$tenferro_dir" rev-parse HEAD 2>/dev/null || true)"
fi

write_run_metadata() {
    local suite_file="$1"
    local suite_id="$2"
    local run_yaml="$3"
    local mode_json="$4"
    local metadata_args=(
        --suite-id "$suite_id"
        --target-profile "$BENCHMARK_TARGET_PROFILE"
        --suite-file "$suite_file"
        --timestamp "$RUN_TIMESTAMP_RFC3339"
        --tenferro-dir "$tenferro_dir"
        --features cuda
        --blas none
        --cuda-device-ordinal "$GPU_BENCH_PHYSICAL_DEVICE"
        --collection-json "$mode_json"
        --output "$run_yaml"
    )
    if [[ -n "$tenferro_commit" ]]; then
        metadata_args+=(--tenferro-commit "$tenferro_commit")
    fi
    "${METADATA[@]}" "${metadata_args[@]}"
}

MODE_JSON="$(printf '{"mode":"throughput","execution_mode":"deferred-batch","deferred_batch_k":%s,"regime_note":"throughput rows are separate from per-call latency rows and are never averaged with them"}' "$GPU_BENCH_DEFERRED_BATCH_K")"

RUST_BIN="${CARGO_TARGET_DIR:-$PROJECT_DIR/target}/release/benchmark_gpu_rust"
AD_BIN="${CARGO_TARGET_DIR:-$PROJECT_DIR/target}/release/benchmark_gpu_linalg_ad"
echo "Building benchmark_gpu_rust and benchmark_gpu_linalg_ad..."
CUBECL_DEBUG_LOG=0 \
CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
RUST_MIN_STACK="$RUST_MIN_STACK" \
LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    cargo build --release --features cuda \
        --bin benchmark_gpu_rust --bin benchmark_gpu_linalg_ad

publish_report() {
    local markdown="$1"
    local report="$2"
    [[ "${GPU_BENCH_SKIP_LATEST:-0}" == 1 ]] && return 0
    cp "$markdown" "$report"
    if [[ -f "${markdown%.md}.summary.json" ]]; then
        cp "${markdown%.md}.summary.json" "${report%.md}.summary.json"
    fi
}

# --- Dense throughput suite --------------------------------------------------
dense_id="$(awk '$1 == "suite_id:" { print $2; exit }' "$DENSE_SUITE")"
dense_name="${dense_id#gpu/}"
DENSE_RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/gpu/$dense_name/$TIMESTAMP"
DENSE_YAML="$DENSE_RUN_DIR/run.yaml"
DENSE_JSONL="$DENSE_RUN_DIR/records.jsonl"
DENSE_MARKDOWN="$DENSE_RUN_DIR/report.md"
DENSE_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/gpu/$dense_name.md"
mkdir -p "$DENSE_RUN_DIR" "$(dirname "$DENSE_REPORT")"
write_run_metadata "$DENSE_SUITE" "$dense_id" "$DENSE_YAML" "$MODE_JSON"

echo "Running $dense_id (deferred batch, K=$GPU_BENCH_DEFERRED_BATCH_K)..."
CUBECL_DEBUG_LOG=0 \
CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
RUST_MIN_STACK="$RUST_MIN_STACK" \
LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "$RUST_BIN" "$DENSE_JSONL" 0 "" tenferro-cuda-trace-deferred-batch -- "$DENSE_SUITE"

if [[ ! -s "$DENSE_JSONL" ]]; then
    echo "No records generated for $dense_id." >&2
    exit 1
fi
"${VALIDATOR[@]}" --kind run "$DENSE_YAML"
"${VALIDATOR[@]}" --kind result "$DENSE_JSONL"
"${DENSE_FORMATTER[@]}" "$DENSE_JSONL" --run-metadata "$DENSE_YAML" --output "$DENSE_MARKDOWN"
publish_report "$DENSE_MARKDOWN" "$DENSE_REPORT"
echo ""
echo "Dense throughput report: $DENSE_REPORT"

# --- Linalg AD throughput suite ---------------------------------------------
ad_id="$(awk '$1 == "suite_id:" { print $2; exit }' "$AD_SUITE")"
ad_name="${ad_id#gpu/}"
AD_RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/gpu/$ad_name/$TIMESTAMP"
AD_YAML="$AD_RUN_DIR/run.yaml"
AD_JSONL="$AD_RUN_DIR/records.jsonl"
AD_UNVERIFIED="$AD_RUN_DIR/rust_records_unverified.jsonl"
AD_MARKDOWN="$AD_RUN_DIR/report.md"
AD_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/gpu/$ad_name.md"
mkdir -p "$AD_RUN_DIR" "$(dirname "$AD_REPORT")"
write_run_metadata "$AD_SUITE" "$ad_id" "$AD_YAML" "$MODE_JSON"

echo "Running $ad_id (deferred batch, K=$GPU_BENCH_DEFERRED_BATCH_K)..."
CUBECL_DEBUG_LOG=0 \
CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
RUST_MIN_STACK="$RUST_MIN_STACK" \
LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "$AD_BIN" "$AD_UNVERIFIED" 0 "" -- "$AD_SUITE"
"${PYTHON[@]}" "$SCRIPT_DIR/verify_gpu_linalg_ad_rust.py" \
    "$AD_UNVERIFIED" "$AD_JSONL" "$AD_SUITE"

if [[ "$GPU_BENCH_VERIFY_ONLY" == 1 ]]; then
    "${PYTHON[@]}" - "$DENSE_JSONL" "$AD_JSONL" <<'PYDEBUG'
import json, sys
for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    assert all(r['status'] in {'verification_only', 'unsupported'} for r in rows), rows
    assert all(r['verification']['status'] == 'passed' for r in rows if r['status'] == 'verification_only')
    assert all(r['timing']['median_ms'] is None for r in rows)
print('AD CUDA throughput verification complete; no timing report updated')
PYDEBUG
    exit 0
fi

"${VALIDATOR[@]}" --kind run "$AD_YAML"
"${VALIDATOR[@]}" --kind result "$AD_JSONL"
"${AD_FORMATTER[@]}" "$AD_JSONL" --run-metadata "$AD_YAML" --output "$AD_MARKDOWN"
publish_report "$AD_MARKDOWN" "$AD_REPORT"
echo ""
echo "AD throughput report: $AD_REPORT"

echo ""
echo "GPU throughput-mode suites complete: $dense_id, $ad_id"
echo "Dense run: $DENSE_RUN_DIR"
echo "AD run:    $AD_RUN_DIR"
