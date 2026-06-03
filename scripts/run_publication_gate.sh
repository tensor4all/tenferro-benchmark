#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THREADS="${1:-1}"
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
PROFILE="${PUBLICATION_GATE_PROFILE:-quick}"
SUITE="${PUBLICATION_GATE_SUITE:-all}"
FEATURES="${PUBLICATION_GATE_FEATURES:-system-openblas}"
RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-$ROOT/data/results}"

# shellcheck source=scripts/thread_env.sh
source "$ROOT/scripts/thread_env.sh"
configure_cpu_thread_env "$THREADS"
export PUBLICATION_GATE_PROFILE="$PROFILE"
export PUBLICATION_GATE_SUITE="$SUITE"

mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/publication_gate_${FEATURES}_t${THREADS}_${PROFILE}_${SUITE}_${TIMESTAMP}.csv"

if [[ "$FEATURES" == "system-openblas" && -z "${OPENBLAS_ROOT:-}" ]]; then
  echo "OPENBLAS_ROOT must be set for PUBLICATION_GATE_FEATURES=system-openblas" >&2
  exit 1
fi

echo "Running publication-gate benchmarks"
echo "  features: $FEATURES"
echo "  threads:  $THREADS"
echo "  profile:  $PROFILE"
echo "  suite:    $SUITE"
echo "  output:   $LOG"
print_cpu_thread_env

case "$FEATURES" in
  cpu-faer)
    cargo run --release --bin publication_gate --no-default-features --features cpu-faer > "$LOG"
    ;;
  system-openblas)
    cargo run --release --bin publication_gate --no-default-features --features system-openblas > "$LOG"
    ;;
  cuda)
    cargo run --release --bin publication_gate --no-default-features --features cuda > "$LOG"
    ;;
  *)
    echo "Unsupported PUBLICATION_GATE_FEATURES=$FEATURES (use cpu-faer, system-openblas, or cuda)" >&2
    exit 1
    ;;
esac

echo "Wrote $LOG"
