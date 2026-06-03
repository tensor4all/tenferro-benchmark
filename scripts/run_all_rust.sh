#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Rust benchmark runner (tenferro trace/eager system OpenBLAS + strided-rs faer)
#
# Usage: ./scripts/run_all_rust.sh [NUM_THREADS]
#
# NUM_THREADS (default: 1) controls:
#   - OMP_NUM_THREADS   (OpenBLAS internal threading)
#   - RAYON_NUM_THREADS  (Rust rayon parallelism)
#
# Requires:
#   - tenferro-rs at extern/tenferro-rs
#   - strided-rs-benchmark-suite at ../strided-rs-benchmark-suite (optional)
#     └─ strided-rs at ../strided-rs (dependency of strided-rs-benchmark-suite)
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

export OMP_NUM_THREADS="$NUM_THREADS"
export RAYON_NUM_THREADS="$NUM_THREADS"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STRIDED_DIR="$(cd "$PROJECT_DIR/../strided-rs-benchmark-suite" 2>/dev/null && pwd || true)"
RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-$PROJECT_DIR/data/results}"

mkdir -p "$RESULTS_DIR"

TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${OPENBLAS_ROOT:-}" ]]; then
    if command -v brew >/dev/null 2>&1 && brew --prefix openblas >/dev/null 2>&1; then
        export OPENBLAS_ROOT="$(brew --prefix openblas)"
    else
        echo "ERROR: OPENBLAS_ROOT is required for tenferro system OpenBLAS." >&2
        echo "Set OPENBLAS_ROOT=/path/to/openblas." >&2
        exit 1
    fi
fi

echo "============================================"
echo " Rust benchmark (threads=${NUM_THREADS})"
echo "============================================"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
echo "  OPENBLAS_ROOT=$OPENBLAS_ROOT"
echo "  tenferro features=system-openblas"
echo ""

RUST_LOGS=()

# ---------------------------------------------------------------------------
# tenferro trace/eager
# ---------------------------------------------------------------------------
echo "============================================"
echo " Rust: tenferro trace/eager"
echo "============================================"

echo "Building tenferro benchmark (OpenBLAS, release)..."
cargo build --release --no-default-features --features system-openblas \
    --bin tenferro-einsum-benchmark \
    --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1

for TENFERRO_MODE in trace eager; do
    TENFERRO_LOG="$RESULTS_DIR/tenferro_${TENFERRO_MODE}_t${NUM_THREADS}_${TIMESTAMP}.log"
    echo "Running tenferro ${TENFERRO_MODE} benchmark..."
    TENFERRO_MODE="$TENFERRO_MODE" \
        cargo run --release --no-default-features --features system-openblas \
        --bin tenferro-einsum-benchmark \
        --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1 | tee "$TENFERRO_LOG"

    echo ""
    echo "tenferro ${TENFERRO_MODE} results saved to: $TENFERRO_LOG"
    echo ""
    RUST_LOGS+=("$TENFERRO_LOG")
done

# ---------------------------------------------------------------------------
# strided-rs (faer)
# ---------------------------------------------------------------------------
if [[ -z "$STRIDED_DIR" ]] || [[ ! -d "$STRIDED_DIR" ]]; then
    echo "NOTE: strided-rs-benchmark-suite not found at ../strided-rs-benchmark-suite"
    echo "  Skipping strided-rs comparison."
    echo "  To enable: git clone https://github.com/tensor4all/strided-rs-benchmark-suite ../strided-rs-benchmark-suite"
    echo ""
else
    echo "============================================"
    echo " Rust: strided-opteinsum (faer)"
    echo "============================================"

    STRIDED_FAER_LOG="$RESULTS_DIR/strided_faer_t${NUM_THREADS}_${TIMESTAMP}.log"

    echo "Building strided-rs (faer, release)..."
    if cargo build --release --no-default-features --features faer,parallel \
            --manifest-path="$STRIDED_DIR/Cargo.toml" 2>&1; then
        echo "Running strided-rs (faer) benchmark..."
        cargo run --release --no-default-features --features faer,parallel \
            --bin strided-rs-benchmark-suite \
            --manifest-path="$STRIDED_DIR/Cargo.toml" 2>&1 | tee "$STRIDED_FAER_LOG"
        echo ""
        echo "strided-rs (faer) results saved to: $STRIDED_FAER_LOG"
        RUST_LOGS+=("$STRIDED_FAER_LOG")
    else
        echo "WARNING: strided-rs (faer) build failed. Skipping."
    fi
    echo ""
fi

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
