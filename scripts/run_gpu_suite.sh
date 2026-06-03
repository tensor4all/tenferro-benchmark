#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
TIMESTAMP="${GPU_BENCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TIMESTAMP_RFC3339="$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

SUITES_VALUE="${GPU_BENCH_SUITE:-benchmarks/gpu/dense.yaml,benchmarks/gpu/einsum.yaml,benchmarks/gpu/sparse.yaml}"
BACKENDS_VALUE="${GPU_BENCH_BACKENDS:-tenferro-cuda-trace,tenferro-cuda-eager,pytorch-cuda,libtorch-cuda,jax-cuda,cublaslt,cutlass,cusolver,cusparse,ginkgo}"
DEVICE_ORDINAL="${GPU_BENCH_DEVICE:-0}"
PROBLEM_FILTER="${GPU_BENCH_PROBLEM:-}"

cd "$PROJECT_DIR"

mkdir -p "$RESULTS_ROOT" "$REPORTS_DIR"

IFS=',' read -r -a SUITES <<< "$SUITES_VALUE"
IFS=',' read -r -a BACKENDS <<< "$BACKENDS_VALUE"

echo "============================================"
echo " GPU benchmark contract suite"
echo "============================================"
echo "Suites:   $SUITES_VALUE"
echo "Backends: $BACKENDS_VALUE"
echo "Device:   cuda:$DEVICE_ORDINAL"
echo "Timestamp: $TIMESTAMP"
echo ""

# Vendor library locations for JIT-compiled backends (CUTLASS, Ginkgo).
# Override via env if installed elsewhere.
export CUTLASS_DIR="${CUTLASS_DIR:-/opt/cutlass}"
export GINKGO_DIR="${GINKGO_DIR:-/opt/ginkgo}"

VALIDATOR=(python3 "$SCRIPT_DIR/validate_benchmark_suite.py")
FORMATTER=(python3 "$SCRIPT_DIR/format_gpu_results.py")
METADATA=(python3 "$SCRIPT_DIR/collect_run_metadata.py")
PYTHON=(python3)
if command -v uv >/dev/null 2>&1; then
    VALIDATOR=(uv run python "$SCRIPT_DIR/validate_benchmark_suite.py")
    FORMATTER=(uv run python "$SCRIPT_DIR/format_gpu_results.py")
    METADATA=(uv run python "$SCRIPT_DIR/collect_run_metadata.py")
    PYTHON=(uv run python)
fi

"${VALIDATOR[@]}" "${SUITES[@]}"

# Split backends: Rust-native vs Python
RUST_BACKENDS=()
PYTHON_BACKENDS=()
for b in "${BACKENDS[@]}"; do
    if [[ "$b" == "tenferro-cuda-trace" || "$b" == "tenferro-cuda-eager" ]]; then
        RUST_BACKENDS+=("$b")
    else
        PYTHON_BACKENDS+=("$b")
    fi
done

resolve_git_commit() {
    local checkout_dir="$1"
    if [[ -d "$checkout_dir/.git" ]] && command -v git >/dev/null 2>&1; then
        git -C "$checkout_dir" rev-parse HEAD 2>/dev/null || true
    fi
}

suite_id_for_file() {
    awk '$1 == "suite_id:" { print $2; exit }' "$1"
}

write_run_metadata() {
    local suite_id="$1"
    local suite_file="$2"
    local run_yaml="$3"
    local tenferro_dir="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
    local tenferro_commit
    tenferro_commit="$(resolve_git_commit "$tenferro_dir")"
    local args=(
        --suite-id "$suite_id"
        --suite-file "$suite_file"
        --timestamp "$RUN_TIMESTAMP_RFC3339"
        --tenferro-dir "$tenferro_dir"
        --features cuda
        --blas none
        --output "$run_yaml"
    )
    if [[ -n "$tenferro_commit" ]]; then
        args+=(--tenferro-commit "$tenferro_commit")
    fi
    "${METADATA[@]}" "${args[@]}"
}

RUST_BIN="$PROJECT_DIR/target/release/benchmark_gpu_rust"
RUST_BUILT=0

for suite in "${SUITES[@]}"; do
    suite_id="$(suite_id_for_file "$suite")"
    if [[ ! "$suite_id" =~ ^gpu/[a-z0-9][a-z0-9_.-]*$ ]]; then
        echo "ERROR: unsupported GPU suite_id '$suite_id' in $suite" >&2
        exit 1
    fi

    suite_name="${suite_id#gpu/}"
    RUN_DIR="$RESULTS_ROOT/gpu/$suite_name/$TIMESTAMP"
    RUN_YAML="$RUN_DIR/run.yaml"
    RESULT_JSONL="$RUN_DIR/records.jsonl"
    RUST_JSONL="$RUN_DIR/rust_records.jsonl"
    MARKDOWN_OUT="$RUN_DIR/report.md"
    REPORT_OUT="$REPORTS_DIR/gpu/$suite_name.md"

    mkdir -p "$RUN_DIR" "$(dirname "$REPORT_OUT")"
    : > "$RESULT_JSONL"
    write_run_metadata "$suite_id" "$suite" "$RUN_YAML"

    # Run Python benchmark (pytorch-cuda, jax-cuda, vendor backends)
    if [[ ${#PYTHON_BACKENDS[@]} -gt 0 ]]; then
        echo "Running Python benchmarks for $suite_id: ${PYTHON_BACKENDS[*]}"
        "${PYTHON[@]}" "$SCRIPT_DIR/benchmark_gpu_python.py" \
            "$RESULT_JSONL" "$DEVICE_ORDINAL" "$PROBLEM_FILTER" \
            "${PYTHON_BACKENDS[@]}" -- "$suite"
    fi

    # Run Rust benchmark (tenferro-cuda-eager, tenferro-cuda-trace)
    if [[ ${#RUST_BACKENDS[@]} -gt 0 ]]; then
        echo "Running Rust GPU benchmarks for $suite_id: ${RUST_BACKENDS[*]}"
        if [[ "$RUST_BUILT" == "0" ]]; then
            # Always ask Cargo to build before measuring so source changes that affect
            # synchronization or timing metadata cannot be hidden by a stale release
            # binary from an earlier benchmark run.
            echo "  Building benchmark_gpu_rust..."
            CUBECL_DEBUG_LOG=0 \
            CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
            LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
                cargo build --release --features cuda --bin benchmark_gpu_rust 2>&1
            RUST_BUILT=1
        fi
        CUBECL_DEBUG_LOG=0 \
        CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
        LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
            "$RUST_BIN" "$RUST_JSONL" "$DEVICE_ORDINAL" "$PROBLEM_FILTER" \
            "${RUST_BACKENDS[@]}" -- "$suite"
        cat "$RUST_JSONL" >> "$RESULT_JSONL"
    fi

    if [[ ! -s "$RESULT_JSONL" ]]; then
        echo "No records generated for $suite_id; skipping report."
        continue
    fi

    "${VALIDATOR[@]}" --kind run "$RUN_YAML"
    "${VALIDATOR[@]}" --kind result "$RESULT_JSONL"

    "${FORMATTER[@]}" "$RESULT_JSONL" --run-metadata "$RUN_YAML" --output "$MARKDOWN_OUT"
    cp "$MARKDOWN_OUT" "$REPORT_OUT"

    echo ""
    echo "GPU benchmark suite complete: $suite_id"
    echo "Run YAML: $RUN_YAML"
    echo "JSONL:    $RESULT_JSONL"
    echo "Markdown: $MARKDOWN_OUT"
    echo "Report:   $REPORT_OUT"
done

echo ""
echo "GPU benchmark suites complete"
