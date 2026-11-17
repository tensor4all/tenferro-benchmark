#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THREADS="${1:-1}"
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
PROFILE="${PUBLICATION_GATE_PROFILE:-quick}"
SUITE="${PUBLICATION_GATE_SUITE:-all}"
RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-$ROOT/data/results}"



# shellcheck source=scripts/thread_env.sh
source "$ROOT/scripts/thread_env.sh"
configure_cpu_thread_env "$THREADS"
export PUBLICATION_GATE_PROFILE="$PROFILE"
export PUBLICATION_GATE_SUITE="$SUITE"

mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/publication_gate_native_t${THREADS}_${PROFILE}_${SUITE}_${TIMESTAMP}.csv"

echo "Running publication-gate benchmarks"
echo "  features: native"
echo "  threads:  $THREADS"
echo "  profile:  $PROFILE"
echo "  suite:    $SUITE"
echo "  output:   $LOG"
print_cpu_thread_env

cargo run --release --bin publication_gate --no-default-features --features native > "$LOG"

echo "Wrote $LOG"
