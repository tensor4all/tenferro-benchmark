#!/usr/bin/env bash
# Publication hooks shared by the CPU runners.
#
# A run directory is the record of a measurement; a page is derived from it.
#
#   finalize_benchmark_run <project_dir> <run_dir>
#       close the run record after collection: merge the per-thread case
#       statuses and record the command wall. Collection facts only.
#
#   render_run_record <project_dir> <run_dir>...
#       render each run directory's own report and summary into the record.
#
#   publish_cpu_runs <project_dir> <run_dir>...
#       render the records, then publish
#       result/<profile>/cpu/<page>.{md,summary.json} from a rendering of the
#       selected run directories. The page is never a copy of a run report.
#
# BENCHMARK_RAW_ONLY=1 ends the run at finalize: nothing is rendered and
# nothing under result/ changes.

run_benchmark_python() {
    local project_dir="$1"
    shift
    if [[ -x "$project_dir/.venv/bin/python" ]]; then
        "$project_dir/.venv/bin/python" "$@"
    elif command -v uv >/dev/null 2>&1; then
        uv run --project "$project_dir" python "$@" || python3 "$@"
    else
        python3 "$@"
    fi
}

benchmark_raw_only() {
    [[ "${BENCHMARK_RAW_ONLY:-0}" == "1" ]]
}

finalize_benchmark_run() {
    local project_dir="$1"
    local run_dir="$2"
    run_benchmark_python "$project_dir" "$project_dir/scripts/finalize_run.py" \
        --run-dir "$run_dir"
}

render_run_record() {
    local project_dir="$1"
    shift
    local run_dir
    for run_dir in "$@"; do
        run_benchmark_python "$project_dir" "$project_dir/scripts/render_run.py" "$run_dir"
    done
}

publish_cpu_runs() {
    local project_dir="$1"
    shift
    local run_dirs=("$@")
    if benchmark_raw_only; then
        echo "BENCHMARK_RAW_ONLY=1: the run record ends here; no report, summary or result/ page is written." >&2
        return 0
    fi
    render_run_record "$project_dir" "${run_dirs[@]}"
    run_benchmark_python "$project_dir" \
        "$project_dir/scripts/regenerate_report_summaries.py" --publish "${run_dirs[@]}"
}
