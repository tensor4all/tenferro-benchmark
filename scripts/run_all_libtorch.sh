#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# LibTorch CPU benchmark runner.
#
# Usage: ./scripts/run_all_libtorch.sh [NUM_THREADS]
#
# Requires:
#   - Torch_DIR or CMAKE_PREFIX_PATH pointing at an OpenBLAS-linked LibTorch
#   - OPENBLAS_ROOT pointing at the same OpenBLAS installation used for Rust
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-$PROJECT_DIR/data/results}"
BUILD_DIR="$PROJECT_DIR/build/cpp-libtorch"

# shellcheck source=scripts/thread_env.sh
source "$SCRIPT_DIR/thread_env.sh"
configure_cpu_thread_env "$NUM_THREADS"

mkdir -p "$RESULTS_DIR"

TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${OPENBLAS_ROOT:-}" ]]; then
    if command -v brew >/dev/null 2>&1 && brew --prefix openblas >/dev/null 2>&1; then
        export OPENBLAS_ROOT="$(brew --prefix openblas)"
    else
        echo "ERROR: OPENBLAS_ROOT is required for a fair OpenBLAS comparison." >&2
        echo "Set OPENBLAS_ROOT=/path/to/openblas." >&2
        exit 1
    fi
fi

if [[ -z "${Torch_DIR:-}" && -z "${CMAKE_PREFIX_PATH:-}" ]]; then
    echo "ERROR: Torch_DIR or CMAKE_PREFIX_PATH must point to an OpenBLAS-linked LibTorch." >&2
    echo "Example: Torch_DIR=/path/to/libtorch/share/cmake/Torch" >&2
    exit 1
fi

echo "============================================"
echo " C++ benchmark: libtorch-cpu (threads=${NUM_THREADS})"
echo "============================================"
print_cpu_thread_env
echo "  OPENBLAS_ROOT=$OPENBLAS_ROOT"
[[ -n "${Torch_DIR:-}" ]] && echo "  Torch_DIR=$Torch_DIR"
[[ -n "${CMAKE_PREFIX_PATH:-}" ]] && echo "  CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"
echo ""

rm -rf "$BUILD_DIR"
CMAKE_ARGS=(
    -DBUILD_LIBTORCH_BENCHMARK=ON
    -DOPENBLAS_ROOT="$OPENBLAS_ROOT"
)
if [[ -n "${Torch_DIR:-}" ]]; then
    CMAKE_ARGS+=(-DTorch_DIR="$Torch_DIR")
fi

cmake -S "$PROJECT_DIR/cpp" -B "$BUILD_DIR" \
    -U Torch_DIR \
    "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --target benchmark_libtorch --config Release

SEARCH_ROOTS=("$BUILD_DIR")
if [[ -n "${Torch_DIR:-}" ]]; then
    SEARCH_ROOTS+=("$Torch_DIR")
    TORCH_PREFIX="$(cd "$Torch_DIR/../../.." 2>/dev/null && pwd || true)"
    [[ -n "$TORCH_PREFIX" ]] && SEARCH_ROOTS+=("$TORCH_PREFIX")
fi
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
    IFS=':' read -r -a CMAKE_PREFIX_ROOTS <<< "$CMAKE_PREFIX_PATH"
    SEARCH_ROOTS+=("${CMAKE_PREFIX_ROOTS[@]}")
fi

LIBTORCH_CPU="$(find "${SEARCH_ROOTS[@]}" \
    -name 'libtorch_cpu.*' -print 2>/dev/null | head -n 1 || true)"
if [[ -n "$LIBTORCH_CPU" ]]; then
    if command -v otool >/dev/null 2>&1; then
        if ! otool -L "$LIBTORCH_CPU" | grep -qi openblas; then
            echo "ERROR: $LIBTORCH_CPU does not appear to link OpenBLAS." >&2
            echo "Use a LibTorch build linked against OpenBLAS for a fair comparison." >&2
            exit 1
        fi
    elif command -v ldd >/dev/null 2>&1; then
        if ! ldd "$LIBTORCH_CPU" | grep -qi openblas; then
            echo "ERROR: $LIBTORCH_CPU does not appear to link OpenBLAS." >&2
            echo "Use a LibTorch build linked against OpenBLAS for a fair comparison." >&2
            exit 1
        fi
    fi
else
    echo "WARNING: could not locate libtorch_cpu for OpenBLAS linkage inspection." >&2
fi

LIBTORCH_LOG="$RESULTS_DIR/libtorch_cpu_t${NUM_THREADS}_${TIMESTAMP}.log"

echo "Running libtorch-cpu benchmark..."
"$BUILD_DIR/benchmark_libtorch" 2>&1 | tee "$LIBTORCH_LOG"

echo ""
echo "libtorch-cpu results saved to: $LIBTORCH_LOG"
