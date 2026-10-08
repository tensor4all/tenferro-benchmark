#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THREADS="${1:-1}"
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
PROFILE="${PUBLICATION_GATE_PROFILE:-quick}"
SUITE="${PUBLICATION_GATE_SUITE:-all}"
RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-$ROOT/data/results}"

# shellcheck source=scripts/cpu_blas_provider.sh
source "$ROOT/scripts/cpu_blas_provider.sh"

FEATURES="$(normalize_cpu_blas_features "${PUBLICATION_GATE_FEATURES:-}")"
export PUBLICATION_GATE_FEATURES="$FEATURES"

# shellcheck source=scripts/thread_env.sh
source "$ROOT/scripts/thread_env.sh"
configure_cpu_thread_env "$THREADS"
export PUBLICATION_GATE_PROFILE="$PROFILE"
export PUBLICATION_GATE_SUITE="$SUITE"

mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/publication_gate_${FEATURES}_t${THREADS}_${PROFILE}_${SUITE}_${TIMESTAMP}.csv"

ensure_blas_env_for_features "$FEATURES"

echo "Running publication-gate benchmarks"
echo "  features: $FEATURES"
echo "  threads:  $THREADS"
echo "  profile:  $PROFILE"
echo "  suite:    $SUITE"
echo "  output:   $LOG"
print_cpu_thread_env

case "$FEATURES" in
  native|blas-openblas|blas-accelerate|blas-mkl|cuda) ;;
  *)
    echo "Unsupported PUBLICATION_GATE_FEATURES=$FEATURES (use native, blas-openblas, blas-accelerate, blas-mkl, or cuda)" >&2
    exit 1
    ;;
esac
cargo run --release --bin publication_gate --no-default-features --features "$FEATURES" > "$LOG"

echo "Wrote $LOG"
