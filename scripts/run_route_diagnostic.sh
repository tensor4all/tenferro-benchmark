#!/usr/bin/env bash
# cpu/route_contract: mechanism diagnostics for the cpu/session_matrix batched
# routes (provider spy + counting allocator). Never a timing run, so it does
# not need an idle host and must not be mixed into timing tables.
#
# Usage: scripts/run_route_diagnostic.sh [threads...]        (default: 1 4)
#
# Environment:
#   TENFERRO_RS_DIR           tenferro-rs checkout to build against
#                             (default: the current extern/tenferro-rs target)
#   BENCHMARK_TARGET_PROFILE  mac-cpu | amd-cpu (default amd-cpu)
#   BENCH_COVERAGE            quick | full (default full; counters are cheap)
#   BENCH_INSTANCE            optional comma-separated case-ID filter
#   RUN_LABEL                 free-form label recorded in metadata (baseline, repair, ...)
#
# Writes data/results/<profile>/cpu/route_contract/<timestamp>/ and
# result/<profile>/cpu/route_contract.md. Exit status 3 means a deterministic
# route-contract FAIL (or INCOMPLETE pair); see scripts/route_contract.py.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
source "$SCRIPT_DIR/thread_env.sh"
if [[ $# -gt 0 ]]; then THREAD_COUNTS=("$@"); else THREAD_COUNTS=(1 4); fi
export BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-amd-cpu}"
COVERAGE="${BENCH_COVERAGE:-full}"
TENFERRO_DIR="${TENFERRO_RS_DIR:-$(readlink -f extern/tenferro-rs)}"
PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3
"$PYTHON" -c 'from scripts.benchmark_layout import safe_target_profile; import sys; safe_target_profile(sys.argv[1])' "$BENCHMARK_TARGET_PROFILE"
COMMAND="$(printf '%q ' "$0" "$@")"
COMMAND="TENFERRO_RS_DIR=$TENFERRO_DIR BENCHMARK_TARGET_PROFILE=$BENCHMARK_TARGET_PROFILE BENCH_COVERAGE=$COVERAGE ${BENCH_INSTANCE:+BENCH_INSTANCE=$BENCH_INSTANCE }${RUN_LABEL:+RUN_LABEL=$RUN_LABEL }$COMMAND"

VARS="$(mktemp)"
trap 'rm -f "$VARS"' EXIT
"$SCRIPT_DIR/build_for_tenferro_rev.sh" "$TENFERRO_DIR" "$VARS" cpu_route_diagnostic
# shellcheck disable=SC1090
source "$VARS"

CASES="$("$PYTHON" "$SCRIPT_DIR/benchmark_cpu_session.py" --list-diagnostic-cases --coverage "$COVERAGE")"
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="$PROJECT_DIR/data/results/$BENCHMARK_TARGET_PROFILE/cpu/route_contract/$TIMESTAMP"
mkdir -p "$RUN_DIR"
for threads in "${THREAD_COUNTS[@]}"; do
    configure_cpu_thread_env "$threads"
    "$PYTHON" "$SCRIPT_DIR/collect_run_metadata.py" --suite-id cpu/route_contract \
        --target-profile "$BENCHMARK_TARGET_PROFILE" --suite-file benchmarks/cpu/session_matrix.yaml \
        --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --tenferro-dir "$TENFERRO_DIR" \
        --features "$BUILD_FEATURES" --command "$COMMAND" \
        --collection-json "{\"coverage\":\"$COVERAGE\",\"effort\":\"scan\",\"measurement_kind\":\"mechanism_diagnostic\",\"label\":\"${RUN_LABEL:-}\",\"binary\":\"$BIN_DIR/cpu_route_diagnostic\"}" \
        --output "$RUN_DIR/run_t${threads}.yaml"
    "$BIN_DIR/cpu_route_diagnostic" --threads "$threads" --cases "$CASES" \
        > "$RUN_DIR/diagnostics_t${threads}.jsonl"
done
status=0
"$PYTHON" "$SCRIPT_DIR/route_contract.py" --run-dir "$RUN_DIR" \
    --target-profile "$BENCHMARK_TARGET_PROFILE" || status=$?
echo "$RUN_DIR"
exit "$status"
