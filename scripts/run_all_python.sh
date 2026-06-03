#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Python benchmark runner (PyTorch CPU + JAX CPU)
#
# Usage: ./scripts/run_all_python.sh [NUM_THREADS]
#
# NUM_THREADS (default: 1) controls:
#   - OMP_NUM_THREADS  (JAX/XLA CPU threading environment)
#
# Requires uv or another Python environment with the project's Python deps.
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-$PROJECT_DIR/data/results}"

# shellcheck source=scripts/thread_env.sh
source "$SCRIPT_DIR/thread_env.sh"
configure_cpu_thread_env "$NUM_THREADS"

mkdir -p "$RESULTS_DIR"

TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

echo "============================================"
echo " Python benchmark: pytorch-cpu + jax-cpu (threads=${NUM_THREADS})"
echo "============================================"
print_cpu_thread_env
echo ""

PYTHON_LOGS=()

run_python_backend() {
    local backend="$1"
    local log="$2"

    echo "============================================"
    echo " Python: ${backend}-cpu"
    echo "============================================"
    echo "Running ${backend}-cpu benchmark..."
    if command -v uv >/dev/null 2>&1; then
        uv run python "$SCRIPT_DIR/benchmark_python.py" \
            --backend "$backend" \
            --num-threads "$NUM_THREADS" 2>&1 | tee "$log" \
            || python3 "$SCRIPT_DIR/benchmark_python.py" \
                --backend "$backend" \
                --num-threads "$NUM_THREADS" 2>&1 | tee "$log"
    else
        python3 "$SCRIPT_DIR/benchmark_python.py" \
            --backend "$backend" \
            --num-threads "$NUM_THREADS" 2>&1 | tee "$log"
    fi

    echo ""
    echo "${backend}-cpu results saved to: $log"
    echo ""
    PYTHON_LOGS+=("$log")
}

PYTORCH_LOG="$RESULTS_DIR/pytorch_cpu_t${NUM_THREADS}_${TIMESTAMP}.log"
JAX_LOG="$RESULTS_DIR/jax_cpu_t${NUM_THREADS}_${TIMESTAMP}.log"

run_python_backend pytorch "$PYTORCH_LOG"
run_python_backend jax "$JAX_LOG"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo " Python benchmark complete"
echo "============================================"
echo "Results:"
for log in "${PYTHON_LOGS[@]}"; do
    echo "  $log"
done
