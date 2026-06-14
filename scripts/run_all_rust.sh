#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Rust einsum benchmark runner (tenferro-rs trace + eager)
#
# Usage: ./scripts/run_all_rust.sh [NUM_THREADS]
#
# NUM_THREADS (default: 1) controls the shared CPU thread environment
# used by OpenBLAS, Accelerate/vecLib, and Rust rayon.
#
# Requires tenferro-rs at extern/tenferro-rs.
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-$PROJECT_DIR/data/results}"

# shellcheck source=scripts/cpu_blas_provider.sh
source "$SCRIPT_DIR/cpu_blas_provider.sh"

TENFERRO_CPU_FEATURES="$(normalize_cpu_blas_features "${TENFERRO_CPU_FEATURES:-}")"
export TENFERRO_CPU_FEATURES
case "${TENFERRO_CPU_BACKEND_KIND:-}" in
    "")
        case "$TENFERRO_CPU_FEATURES" in
            system-openblas|system-accelerate)
                export TENFERRO_CPU_BACKEND_KIND=blas
                ;;
            *)
                export TENFERRO_CPU_BACKEND_KIND=default
                ;;
        esac
        ;;
    default|faer|blas)
        export TENFERRO_CPU_BACKEND_KIND
        ;;
    *)
        echo "ERROR: TENFERRO_CPU_BACKEND_KIND must be default, faer, or blas." >&2
        exit 1
        ;;
esac

# shellcheck source=scripts/thread_env.sh
source "$SCRIPT_DIR/thread_env.sh"
configure_cpu_thread_env "$NUM_THREADS"

mkdir -p "$RESULTS_DIR"

TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"

echo "============================================"
echo " Rust benchmark (threads=${NUM_THREADS})"
echo "============================================"
print_cpu_thread_env
[[ -n "${OPENBLAS_ROOT:-}" ]] && echo "  OPENBLAS_ROOT=$OPENBLAS_ROOT"
echo "  tenferro features=$TENFERRO_CPU_FEATURES"
echo "  TENFERRO_CPU_BACKEND_KIND=$TENFERRO_CPU_BACKEND_KIND"
echo ""

RUST_LOGS=()

# ---------------------------------------------------------------------------
# tenferro trace/eager
# ---------------------------------------------------------------------------
echo "============================================"
echo " Rust: tenferro trace/eager"
echo "============================================"

echo "Building tenferro benchmark ($TENFERRO_CPU_FEATURES, release)..."
cargo build --release --no-default-features --features "$TENFERRO_CPU_FEATURES" \
    --bin tenferro-einsum-benchmark \
    --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1

for TENFERRO_MODE in trace eager; do
    TENFERRO_LOG="$RESULTS_DIR/tenferro_${TENFERRO_MODE}_t${NUM_THREADS}_${TIMESTAMP}.log"
    echo "Running tenferro ${TENFERRO_MODE} benchmark..."
    TENFERRO_MODE="$TENFERRO_MODE" \
    TENFERRO_CPU_BACKEND_KIND="$TENFERRO_CPU_BACKEND_KIND" \
        cargo run --release --no-default-features --features "$TENFERRO_CPU_FEATURES" \
        --bin tenferro-einsum-benchmark \
        --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1 | tee "$TENFERRO_LOG"

    echo ""
    echo "tenferro ${TENFERRO_MODE} results saved to: $TENFERRO_LOG"
    echo ""
    RUST_LOGS+=("$TENFERRO_LOG")
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo " Rust benchmark complete"
echo "============================================"
echo "Results:"
for log in "${RUST_LOGS[@]}"; do
    echo "  $log"
done
