#!/usr/bin/env bash
set -euo pipefail

# Reproduce the local Linux CPU linalg JVP/VJP report requested for
# result/linux-cpu/cpu/linalg_jvp_jvp.md.
#
# The benchmark runner's maintained Linux CPU target profile is amd-cpu. This
# script runs that standard CPU path sequentially for the requested thread
# counts, then mirrors the linalg AD reports to the requested linux-cpu report
# alias. By default it runs both 1T and 4T.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "$#" -eq 0 ]]; then
    THREAD_COUNTS=(1 4)
else
    THREAD_COUNTS=("$@")
fi

RUN_TARGET_PROFILE="${BENCHMARK_RUN_TARGET_PROFILE:-amd-cpu}"
REPORT_TARGET_PROFILE="${BENCHMARK_REPORT_TARGET_PROFILE:-linux-cpu}"

case "$RUN_TARGET_PROFILE" in
    mac-cpu|amd-cpu)
        ;;
    *)
        echo "BENCHMARK_RUN_TARGET_PROFILE must be mac-cpu or amd-cpu for this CPU script." >&2
        exit 2
        ;;
esac

export BENCHMARK_TARGET_PROFILE="$RUN_TARGET_PROFILE"
export TENFERRO_CPU_FEATURES="${TENFERRO_CPU_FEATURES:-cpu-faer}"
export PUBLICATION_GATE_FEATURES="${PUBLICATION_GATE_FEATURES:-$TENFERRO_CPU_FEATURES}"
if [[ "$TENFERRO_CPU_FEATURES" == "cpu-faer" ]]; then
    export TENFERRO_CPU_BACKEND_KIND="${TENFERRO_CPU_BACKEND_KIND:-faer}"
fi

export BENCH_INSTANCE="${BENCH_INSTANCE:-bin_matmul_256}"
export BENCH_RUNS="${BENCH_RUNS:-15}"
export BENCH_WARMUPS="${BENCH_WARMUPS:-3}"
export PUBLICATION_GATE_PROFILE="${PUBLICATION_GATE_PROFILE:-quick}"
export PUBLICATION_GATE_SUITE="${PUBLICATION_GATE_SUITE:-all}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

alias_report="$ROOT/result/$REPORT_TARGET_PROFILE/cpu/linalg_jvp_jvp.md"
mkdir -p "$(dirname "$alias_report")"

validate_threads() {
    local threads="$1"
    case "$threads" in
        ''|*[!0-9]*)
            echo "Usage: $0 [NUM_THREADS ...]" >&2
            exit 2
            ;;
    esac
    if [[ "$threads" -lt 1 ]]; then
        echo "NUM_THREADS must be >= 1" >&2
        exit 2
    fi
}

script_invocation() {
    local cmd="./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh"
    if [[ "$#" -gt 0 ]]; then
        printf '%s %s\n' "$cmd" "$*"
    else
        printf '%s\n' "$cmd"
    fi
}

rewrite_report() {
    local source_report="$1"
    local output_report="$2"
    local reproduce="$3"
    awk \
        -v run_profile="$RUN_TARGET_PROFILE" \
        -v report_profile="$REPORT_TARGET_PROFILE" \
        -v script="$reproduce" '
{
    gsub("Target profile: `" run_profile "`", "Target profile: `" report_profile "`")
    gsub("data/results/" run_profile "/", "data/results/" report_profile "/")
    print
    if ($0 ~ /^Latest run:/) {
        print ""
        print "Reproduce: `" script "`."
    }
}
    ' "$source_report" > "$output_report"
}

sectionize_report() {
    local report="$1"
    awk '
        /^#/ {
            print "##" $0
            next
        }
        { print }
    ' "$report"
}

for threads in "${THREAD_COUNTS[@]}"; do
    validate_threads "$threads"
done

reproduce_cmd="$(script_invocation "${THREAD_COUNTS[@]}")"

TMP_REPORTS=()
THREADS_DONE=()
TIMESTAMPS_DONE=()
RUN_DIRS_DONE=()

cleanup() {
    for path in "${TMP_REPORTS[@]:-}"; do
        [[ -n "$path" ]] && rm -f "$path"
    done
}
trap cleanup EXIT

for threads in "${THREAD_COUNTS[@]}"; do
    "$SCRIPT_DIR/run_all.sh" "$threads"

    source_report="$ROOT/result/$RUN_TARGET_PROFILE/cpu/linalg_jvp_vjp.md"
    if [[ ! -f "$source_report" ]]; then
        echo "Expected source report was not generated: $source_report" >&2
        exit 1
    fi

    timestamp="$(
        awk -F'`' '/^- Timestamp:/ { print $2; exit }' "$source_report"
    )"
    if [[ -z "$timestamp" ]]; then
        echo "Could not determine benchmark timestamp from $source_report" >&2
        exit 1
    fi

    source_run_dir="$ROOT/data/results/$RUN_TARGET_PROFILE/cpu/einsum/$timestamp"
    alias_run_dir="$ROOT/data/results/$REPORT_TARGET_PROFILE/cpu/einsum/$timestamp"
    if [[ -d "$source_run_dir" ]]; then
        mkdir -p "$(dirname "$alias_run_dir")"
        rm -rf "$alias_run_dir"
        cp -a "$source_run_dir" "$alias_run_dir"
    fi

    tmp_report="$(mktemp "${TMPDIR:-/tmp}/linux-cpu-linalg-jvp-jvp.${threads}.XXXXXX.md")"
    rewrite_report "$source_report" "$tmp_report" "$reproduce_cmd"
    TMP_REPORTS+=("$tmp_report")
    THREADS_DONE+=("$threads")
    TIMESTAMPS_DONE+=("$timestamp")
    RUN_DIRS_DONE+=("${alias_run_dir#$ROOT/}")
done

{
    echo "# Linux CPU Linalg JVP/VJP Benchmark Results"
    echo ""
    echo "- Suite: \`cpu/linalg_jvp_vjp\`"
    echo "- Target profile: \`$REPORT_TARGET_PROFILE\`"
    echo "- Thread runs: \`${THREADS_DONE[*]}\`"
    echo ""
    echo "Reproduce: \`$reproduce_cmd\`."
    echo ""
    echo "This report is generated from sequential CPU runs. Do not compare it with"
    echo "measurements collected while another CPU benchmark process was running."
    echo ""
    echo "## Run Inputs"
    echo ""
    for i in "${!THREADS_DONE[@]}"; do
        echo "- Threads ${THREADS_DONE[$i]}: timestamp \`${TIMESTAMPS_DONE[$i]}\`, raw run \`${RUN_DIRS_DONE[$i]}\`"
    done
    for i in "${!TMP_REPORTS[@]}"; do
        echo ""
        echo "## Threads: ${THREADS_DONE[$i]}"
        echo ""
        sectionize_report "${TMP_REPORTS[$i]}"
    done
} > "$alias_report"

echo "Wrote $alias_report"
for run_dir in "${RUN_DIRS_DONE[@]}"; do
    echo "Mirrored raw run to $ROOT/$run_dir"
done
