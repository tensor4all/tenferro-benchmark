#!/usr/bin/env bash
# The link preflight (campaign-toolkit `link_preflight`, declared in benchmarks/link-preflight.yaml):
# every library a build or a run is about to link has to link and load on this host, with the build's
# own search paths and the runner's loader path, before anything is built or timed. A missing
# `-liomp5` otherwise surfaces at the follow-up MWE build, hours into a full CPU run.
#
#   source scripts/link_preflight.sh
#   require_link_preflight <arm>...                    # exit status 1 names the missing library and the fix
#   require_link_preflight_for_run <features> [perf_issues=1]   # the arms a run_all.sh run will link
#
# LINK_PREFLIGHT_JSON, when set, receives the JSON report (the evidence a campaign keeps).

_link_preflight_project_dir() { cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd; }

# The interpreter whose campaign-toolkit has the preflight: the project's venv (the pinned toolkit),
# then python3. No silent skip: a host that cannot run the check cannot be trusted to link either.
_link_preflight_python() {
    local project_dir="$1" candidate
    for candidate in "${LINK_PREFLIGHT_PYTHON:-}" "$project_dir/.venv/bin/python" python3; do
        [[ -n "$candidate" ]] || continue
        if "$candidate" -c 'import campaign_toolkit.link_preflight' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "ERROR: no Python here can import campaign_toolkit.link_preflight (the link preflight)." >&2
    echo "       Run \`uv sync\` in $project_dir (pyproject.toml pins campaign-toolkit), or set LINK_PREFLIGHT_PYTHON." >&2
    return 1
}

require_link_preflight() {
    [[ $# -gt 0 ]] || return 0
    local project_dir python arm
    project_dir="$(_link_preflight_project_dir)"
    python="$(_link_preflight_python "$project_dir")" || return 1
    local args=(--config "$project_dir/benchmarks/link-preflight.yaml")
    for arm in "$@"; do args+=(--arm "$arm"); done
    [[ -n "${BENCHMARK_TARGET_PROFILE:-}" ]] && args+=(--profile "$BENCHMARK_TARGET_PROFILE")
    [[ -n "${LINK_PREFLIGHT_JSON:-}" ]] && args+=(--json "$LINK_PREFLIGHT_JSON")
    "$python" -m campaign_toolkit.link_preflight "${args[@]}"
}

# The arms a full run links: the harness's vendor arm, and the follow-up MWE (system-mkl) whenever
# perf_issues runs with MKLROOT set -- run_perf_issues.sh builds it then, at the end of the run.
link_preflight_arms_for_run() {
    local features="$1" perf_issues="${2:-1}"
    case "$features" in
        blas-mkl) printf '%s\n' blas-mkl ;;
        blas-openblas) printf '%s\n' blas-openblas ;;
    esac
    if [[ "$perf_issues" == 1 && -n "${MKLROOT:-}" ]]; then
        printf '%s\n' followup-mwe
    fi
}

require_link_preflight_for_run() {
    local arms=() arm
    while IFS= read -r arm; do
        [[ -n "$arm" ]] && arms+=("$arm")
    done < <(link_preflight_arms_for_run "$@")
    [[ ${#arms[@]} -gt 0 ]] || return 0
    require_link_preflight "${arms[@]}"
}
