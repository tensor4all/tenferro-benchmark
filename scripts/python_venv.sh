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
        if [[ -n "${BENCHMARK_TORCH_WHEEL:-}" ]]; then
            # The OpenBLAS image supplies a source-built wheel. Do not replace
            # it with the lockfile's provider-mismatched binary distribution.
            local wheel
            wheel="$(realpath -e -- "$BENCHMARK_TORCH_WHEEL")" || exit 1
            uv venv --allow-existing .venv || exit 1
            # Prune wheel-only CUDA dependencies too, while retaining the
            # lockfile versions for the rest of the benchmark environment.
            set -o pipefail
            uv export --frozen --no-hashes --no-emit-project --prune torch |
                uv pip install --reinstall-package torch -r - "$wheel" || exit 1
        else
            UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" uv sync || exit 1
            uv pip install --index-url "https://download.pytorch.org/whl/cpu" --force-reinstall "torch==2.12.0+cpu" || exit 1
        fi
    ) || return 1

    # Keep subsequent `uv run` invocations from syncing the lockfile back to a
    # non-CPU PyTorch wheel during benchmark collection.
    export UV_NO_SYNC=1
}
