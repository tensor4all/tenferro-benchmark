#!/usr/bin/env bash
set -euo pipefail

# Mirror the amd-cpu cpu/einsum and cpu/cpu_ops reports to the linux-cpu
# report alias, following the same convention as
# reproduce_linux_cpu_linalg_jvp_jvp.sh: the benchmark runner's maintained
# Linux CPU target profile is amd-cpu, and linux-cpu reports are rewritten
# aliases of amd-cpu runs (no re-measurement).
#
# Usage:
#   ./scripts/mirror_linux_cpu_cpu_reports.sh THREADS:TIMESTAMP [THREADS:TIMESTAMP ...]
# Example:
#   ./scripts/mirror_linux_cpu_cpu_reports.sh 1:20260716_075234 4:20260716_081717
#
# Each TIMESTAMP must name an existing full run under
# data/results/<run_profile>/cpu/einsum/<TIMESTAMP>/ produced by
# `BENCHMARK_TARGET_PROFILE=<run_profile> ./scripts/run_all.sh <THREADS>`.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_TARGET_PROFILE="${BENCHMARK_RUN_TARGET_PROFILE:-amd-cpu}"
REPORT_TARGET_PROFILE="${BENCHMARK_REPORT_TARGET_PROFILE:-linux-cpu}"

if [[ "$#" -eq 0 ]]; then
    echo "Usage: $0 THREADS:TIMESTAMP [THREADS:TIMESTAMP ...]" >&2
    exit 2
fi

THREADS_LIST=()
TIMESTAMP_LIST=()
for arg in "$@"; do
    threads="${arg%%:*}"
    timestamp="${arg#*:}"
    case "$threads" in
        ''|*[!0-9]*)
            echo "Bad THREADS in '$arg' (expected THREADS:TIMESTAMP)" >&2
            exit 2
            ;;
    esac
    if [[ -z "$timestamp" || "$timestamp" == "$arg" ]]; then
        echo "Bad TIMESTAMP in '$arg' (expected THREADS:TIMESTAMP)" >&2
        exit 2
    fi
    if [[ ! -d "$ROOT/data/results/$RUN_TARGET_PROFILE/cpu/einsum/$timestamp" ]]; then
        echo "Run directory not found: data/results/$RUN_TARGET_PROFILE/cpu/einsum/$timestamp" >&2
        exit 2
    fi
    THREADS_LIST+=("$threads")
    TIMESTAMP_LIST+=("$timestamp")
done

mirror_invocation="./scripts/mirror_linux_cpu_cpu_reports.sh $*"

rewrite_report() {
    local source_report="$1"
    awk \
        -v run_profile="$RUN_TARGET_PROFILE" \
        -v report_profile="$REPORT_TARGET_PROFILE" '
{
    gsub("Target profile: `" run_profile "`", "Target profile: `" report_profile "`")
    gsub("data/results/" run_profile "/", "data/results/" report_profile "/")
    print
}
    ' "$source_report"
}

sectionize_report() {
    awk '
        /^#/ {
            print "##" $0
            next
        }
        { print }
    '
}

# Mirror raw run dirs so the alias reports point at existing paths.
for timestamp in "${TIMESTAMP_LIST[@]}"; do
    source_run_dir="$ROOT/data/results/$RUN_TARGET_PROFILE/cpu/einsum/$timestamp"
    alias_run_dir="$ROOT/data/results/$REPORT_TARGET_PROFILE/cpu/einsum/$timestamp"
    mkdir -p "$(dirname "$alias_run_dir")"
    rm -rf "$alias_run_dir"
    cp -a "$source_run_dir" "$alias_run_dir"
done

write_alias_report() {
    local suite="$1"        # e.g. cpu/einsum
    local title="$2"        # report H1 title
    local source_name="$3"  # per-run source report filename inside the run dir
    local output="$4"       # output path

    {
        echo "# $title"
        echo ""
        echo "- Suite: \`$suite\`"
        echo "- Target profile: \`$REPORT_TARGET_PROFILE\`"
        echo "- Thread runs: \`${THREADS_LIST[*]}\`"
        echo ""
        echo "Mirrored from \`$RUN_TARGET_PROFILE\` runs (\`BENCHMARK_TARGET_PROFILE=$RUN_TARGET_PROFILE ./scripts/run_all.sh <THREADS>\`);"
        echo "regenerate with \`$mirror_invocation\`."
        echo ""
        echo "This report is generated from sequential CPU runs. Do not compare it with"
        echo "measurements collected while another CPU benchmark process was running."
        echo ""
        echo "## Run Inputs"
        echo ""
        for i in "${!THREADS_LIST[@]}"; do
            echo "- Threads ${THREADS_LIST[$i]}: timestamp \`${TIMESTAMP_LIST[$i]}\`, raw run \`data/results/$REPORT_TARGET_PROFILE/cpu/einsum/${TIMESTAMP_LIST[$i]}\`"
        done
        for i in "${!THREADS_LIST[@]}"; do
            source_report="$ROOT/data/results/$RUN_TARGET_PROFILE/cpu/einsum/${TIMESTAMP_LIST[$i]}/$source_name"
            if [[ ! -f "$source_report" ]]; then
                echo "Missing per-run report: $source_report" >&2
                exit 1
            fi
            echo ""
            echo "## Threads: ${THREADS_LIST[$i]}"
            echo ""
            rewrite_report "$source_report" | sectionize_report
        done
    } > "$output"
    echo "Wrote $output"
}

mkdir -p "$ROOT/result/$REPORT_TARGET_PROFILE/cpu"
write_alias_report "cpu/einsum" "Linux CPU Einsum Benchmark Results" \
    "report.md" "$ROOT/result/$REPORT_TARGET_PROFILE/cpu/einsum.md"
write_alias_report "cpu/cpu_ops" "Linux CPU Ops Benchmark Results" \
    "cpu_ops_report.md" "$ROOT/result/$REPORT_TARGET_PROFILE/cpu/cpu_ops.md"
