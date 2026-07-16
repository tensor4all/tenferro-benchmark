#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# GPU permutation / materialize-kernel benchmark suite (gpu/permutation):
#   - tenferro-cuda-transpose, tenferro-cuda-to-contiguous, cutensor (Rust)
#   - pytorch-cuda, jax-cuda, memcpy-d2d (Python)
#
# Usage: BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_permutation.sh
#
# Ports scripts/run_permutation.sh (cpu/permutation) to CUDA, minus the
# thread-count loop (this suite has no CPU-thread dimension). Runs the Rust
# runner, then EACH Python backend as its own process, all SEQUENTIALLY
# (never concurrently), per AGENTS.md timing discipline and GPU Timing
# Fairness. The per-backend process split follows scripts/run_gpu_suite.sh
# so PyTorch/JAX CUDA allocators release device memory between backends.
#
# This is a STANDALONE entry point, exactly like scripts/run_permutation.sh
# is for cpu/permutation: it is intentionally not wired into
# scripts/run_gpu_suite.sh.
#
# Compares permutation-kernel performance for the patterns listed in
# benchmarks/gpu/permutation.yaml / data/instances/permutation_patterns.json
# (`participants_gpu` per pattern) and writes
# result/nvidia-gpu/gpu/permutation.md.
#
# Environment variables:
# - PATTERN_ID / BENCH_RUNS / BENCH_WARMUPS: forwarded to both runners, as
#   in src/bin/benchmark_gpu_permutation.rs /
#   scripts/benchmark_gpu_permutation_python.py.
# - GPU_BENCH_DEVICE: CUDA device ordinal (default 0).
# - GPU_BENCH_BACKEND_SLEEP: pause in seconds between per-backend Python
#   processes (default 2, same as scripts/run_gpu_suite.sh).
# - TENFERRO_CUTENSOR_PATH: cuTENSOR library search path(s), colon-separated;
#   defaults below to the same path baked into the CUDA devcontainer.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"

BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-nvidia-gpu}"
case "$BENCHMARK_TARGET_PROFILE" in
    nvidia-gpu)
        export BENCHMARK_TARGET_PROFILE
        ;;
    *)
        echo "ERROR: BENCHMARK_TARGET_PROFILE must be nvidia-gpu for gpu/permutation." >&2
        exit 1
        ;;
esac

SUITE_ID="gpu/permutation"
SUITE_FILE="$PROJECT_DIR/benchmarks/gpu/permutation.yaml"
DEVICE_ORDINAL="${GPU_BENCH_DEVICE:-0}"
RUST_MIN_STACK="${RUST_MIN_STACK:-67108864}"

# Same cuTENSOR search path baked into .devcontainer/cuda/devcontainer.json;
# exported here too so this script also works outside that devcontainer as
# long as the same vendor package layout is present (mirrors how
# scripts/run_gpu_suite.sh exports CUTLASS_DIR / GINKGO_DIR defaults).
export TENFERRO_CUTENSOR_PATH="${TENFERRO_CUTENSOR_PATH:-/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2}"

TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TIMESTAMP_RFC3339="$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/gpu/permutation/$TIMESTAMP"
RUN_YAML="$RUN_DIR/run.yaml"
LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/gpu/permutation.md"

mkdir -p "$RUN_DIR" "$(dirname "$LATEST_REPORT")"

PYTHON=(python3)
VALIDATOR=(python3 "$SCRIPT_DIR/validate_benchmark_suite.py")
METADATA=(python3 "$SCRIPT_DIR/collect_run_metadata.py")
FORMATTER=(python3 "$SCRIPT_DIR/format_gpu_permutation_results.py")
if command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run python)
    VALIDATOR=(uv run python "$SCRIPT_DIR/validate_benchmark_suite.py")
    METADATA=(uv run python "$SCRIPT_DIR/collect_run_metadata.py")
    FORMATTER=(uv run python "$SCRIPT_DIR/format_gpu_permutation_results.py")
fi

"${VALIDATOR[@]}" "$SUITE_FILE"

tenferro_dir="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
tenferro_commit=""
if [[ -d "$tenferro_dir/.git" ]] && command -v git >/dev/null 2>&1; then
    tenferro_commit="$(git -C "$tenferro_dir" rev-parse HEAD 2>/dev/null || true)"
fi

metadata_args=(
    --suite-id "$SUITE_ID"
    --target-profile "$BENCHMARK_TARGET_PROFILE"
    --suite-file "${SUITE_FILE#$PROJECT_DIR/}"
    --timestamp "$RUN_TIMESTAMP_RFC3339"
    --tenferro-dir "$tenferro_dir"
    --features cuda
    --blas none
    --cuda-device-ordinal "$DEVICE_ORDINAL"
    --output "$RUN_YAML"
)
[[ -n "$tenferro_commit" ]] && metadata_args+=(--tenferro-commit "$tenferro_commit")
"${METADATA[@]}" "${metadata_args[@]}"

echo "============================================"
echo " GPU permutation benchmark suite"
echo "============================================"
echo "Suite:        $SUITE_ID"
echo "Suite file:   ${SUITE_FILE#$PROJECT_DIR/}"
echo "Target:       $BENCHMARK_TARGET_PROFILE"
echo "Device:       cuda:$DEVICE_ORDINAL"
echo "Timestamp:    $TIMESTAMP"
echo "Run dir:      ${RUN_DIR#$PROJECT_DIR/}"
echo ""

echo "Building benchmark_gpu_permutation..."
CUBECL_DEBUG_LOG=0 \
CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
RUST_MIN_STACK="$RUST_MIN_STACK" \
LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    cargo build --release --features cuda --bin benchmark_gpu_permutation

RUST_JSONL="$RUN_DIR/rust_output.jsonl"
GPU_BENCH_BACKEND_SLEEP="${GPU_BENCH_BACKEND_SLEEP:-2}"

echo "Running Rust GPU permutation benchmarks (tenferro-cuda-transpose, tenferro-cuda-to-contiguous, cutensor)..."
CUBECL_DEBUG_LOG=0 \
CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
RUST_MIN_STACK="$RUST_MIN_STACK" \
LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
GPU_BENCH_DEVICE="$DEVICE_ORDINAL" \
BENCH_OUTPUT="$RUST_JSONL" \
    "$PROJECT_DIR/target/release/benchmark_gpu_permutation"

INPUTS=("$RUST_JSONL")

# One process per Python backend, strictly sequential, mirroring
# scripts/run_gpu_suite.sh: "Separate processes so PyTorch/JAX CUDA
# allocators release device memory." In a shared process, JAX's default
# XLA preallocation (~75% of the card) starves torch of memory for the
# 2 GiB rotation_6d pattern; process exit releases it.
for backend in pytorch-cuda jax-cuda memcpy-d2d; do
    echo "Running Python GPU permutation benchmarks ($backend)..."
    BACKEND_JSONL="$RUN_DIR/python_output_${backend}.jsonl"
    GPU_BENCH_DEVICE="$DEVICE_ORDINAL" \
    BENCH_OUTPUT="$BACKEND_JSONL" \
        "${PYTHON[@]}" "$SCRIPT_DIR/benchmark_gpu_permutation_python.py" "$backend"
    [[ -s "$BACKEND_JSONL" ]] && INPUTS+=("$BACKEND_JSONL")
    sleep "$GPU_BENCH_BACKEND_SLEEP"
done

echo ""
echo "Formatting gpu/permutation results as markdown..."
"${FORMATTER[@]}" \
    "${INPUTS[@]}" \
    --run-metadata "$RUN_YAML" \
    --output "$LATEST_REPORT"

echo ""
echo "GPU permutation benchmark suite complete: $SUITE_ID"
echo "Run YAML:  ${RUN_YAML#$PROJECT_DIR/}"
for path in "${INPUTS[@]}"; do
    echo "JSONL:     ${path#$PROJECT_DIR/}"
done
echo "Report:    ${LATEST_REPORT#$PROJECT_DIR/}"
