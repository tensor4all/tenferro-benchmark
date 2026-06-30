#!/usr/bin/env bash

reset_benchmark_python_venv() {
    local project_dir="$1"
    local venv="$project_dir/.venv"

    if [[ -d "$venv" ]]; then
        echo "Removing Python virtualenv before benchmark: ${venv#$project_dir/}"
        rm -r -- "$venv"
    fi
}

prepare_cpu_benchmark_python_venv() {
    local project_dir="$1"

    if ! command -v uv >/dev/null 2>&1; then
        return 0
    fi

    case "$(uname -s)" in
        Linux) ;;
        *) return 0 ;;
    esac

    if [[ "${BENCHMARK_TARGET_PROFILE:-}" == "nvidia-gpu" ]]; then
        return 0
    fi

    echo "Preparing CPU Python benchmark environment"
    (
        cd "$project_dir"
        UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" uv sync
        uv pip install --index-url "https://download.pytorch.org/whl/cpu" --force-reinstall "torch==2.12.0+cpu"
    )

    # Keep subsequent `uv run` invocations from syncing the lockfile back to a
    # non-CPU PyTorch wheel during benchmark collection.
    export UV_NO_SYNC=1
}
