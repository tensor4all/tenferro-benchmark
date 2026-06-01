#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
TIMESTAMP="${GPU_BENCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

SUITES_VALUE="${GPU_BENCH_SUITE:-benchmarks/gpu/dense.yaml,benchmarks/gpu/einsum.yaml,benchmarks/gpu/sparse.yaml}"
BACKENDS_VALUE="${GPU_BENCH_BACKENDS:-tenferro-cuda-trace,tenferro-cuda-eager,pytorch-cuda,libtorch-cuda,jax-cuda,cublaslt,cutlass,cusolver,cusparse,ginkgo}"
DEVICE_ORDINAL="${GPU_BENCH_DEVICE:-0}"
PROBLEM_FILTER="${GPU_BENCH_PROBLEM:-}"

cd "$PROJECT_DIR"

mkdir -p "$RESULTS_DIR" "$REPORTS_DIR"

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
PYTHON=(python3)
if command -v uv >/dev/null 2>&1; then
    VALIDATOR=(uv run python "$SCRIPT_DIR/validate_benchmark_suite.py")
    FORMATTER=(uv run python "$SCRIPT_DIR/format_gpu_results.py")
    PYTHON=(uv run python)
fi

"${VALIDATOR[@]}" "${SUITES[@]}"

RESULT_JSONL="$RESULTS_DIR/gpu_contract_${TIMESTAMP}.jsonl"

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

# Run Python benchmark (pytorch-cuda, jax-cuda, vendor backends)
if [[ ${#PYTHON_BACKENDS[@]} -gt 0 ]]; then
    echo "Running Python benchmarks: ${PYTHON_BACKENDS[*]}"
    "${PYTHON[@]}" "$SCRIPT_DIR/benchmark_gpu_python.py" \
        "$RESULT_JSONL" "$DEVICE_ORDINAL" "$PROBLEM_FILTER" \
        "${PYTHON_BACKENDS[@]}" -- "${SUITES[@]}"
fi

# Run Rust benchmark (tenferro-cuda-eager, tenferro-cuda-trace)
RUST_BIN="$PROJECT_DIR/target/release/benchmark_gpu_rust"
if [[ ${#RUST_BACKENDS[@]} -gt 0 ]]; then
    echo "Running Rust GPU benchmarks: ${RUST_BACKENDS[*]}"
    RUST_JSONL="$RESULTS_DIR/gpu_rust_${TIMESTAMP}.jsonl"
    # Always ask Cargo to build before measuring so source changes that affect
    # synchronization or timing metadata cannot be hidden by a stale release
    # binary from an earlier benchmark run.
    echo "  Building benchmark_gpu_rust..."
    CUBECL_DEBUG_LOG=0 \
    CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
    LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        cargo build --release --features cuda --bin benchmark_gpu_rust 2>&1
    CUBECL_DEBUG_LOG=0 \
    CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
    LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        "$RUST_BIN" "$RUST_JSONL" "$DEVICE_ORDINAL" "$PROBLEM_FILTER" \
        "${RUST_BACKENDS[@]}" -- "${SUITES[@]}"
    # Append Rust results to the main JSONL
    if [[ ${#PYTHON_BACKENDS[@]} -gt 0 ]]; then
        cat "$RUST_JSONL" >> "$RESULT_JSONL"
    else
        cp "$RUST_JSONL" "$RESULT_JSONL"
    fi
fi

"${VALIDATOR[@]}" --kind result "$RESULT_JSONL"

MARKDOWN_OUT="$RESULTS_DIR/gpu_results_${TIMESTAMP}.md"
REPORT_OUT="$REPORTS_DIR/gpu-benchmark-results.md"
"${FORMATTER[@]}" "$RESULT_JSONL" --output "$MARKDOWN_OUT"
cp "$MARKDOWN_OUT" "$REPORT_OUT"

echo ""
echo "GPU benchmark suite complete"
echo "JSONL:    $RESULT_JSONL"
echo "Markdown: $MARKDOWN_OUT"
echo "Report:   $REPORT_OUT"
