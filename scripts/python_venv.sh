#!/usr/bin/env bash

reset_benchmark_python_venv() {
    local project_dir="$1"
    local venv="$project_dir/.venv"

    if [[ -d "$venv" ]]; then
        echo "Removing Python virtualenv before benchmark: ${venv#$project_dir/}"
        rm -r -- "$venv"
    fi
}

# Print the torch version pinned in uv.lock (for example `2.12.0`).
locked_torch_version() {
    local project_dir="$1"

    awk '
        /^name = "torch"$/ { in_torch = 1; next }
        in_torch && /^version = "/ {
            gsub(/^version = "|"$/, "", $0)
            print
            exit
        }
        in_torch && /^\[\[package\]\]$/ { exit }
    ' "$project_dir/uv.lock"
}

prepare_cpu_benchmark_python_venv() {
    local project_dir="$1"

    if ! command -v uv >/dev/null 2>&1; then
        return 0
    fi

    case "$(uname -s)" in
        Linux|Darwin) ;;
        *) return 0 ;;
    esac

    if [[ "${BENCHMARK_TARGET_PROFILE:-}" == "nvidia-gpu" ]]; then
        return 0
    fi

    echo "Preparing CPU Python benchmark environment"
    (
        cd "$project_dir"
        if [[ "$(uname -s)" == "Darwin" ]]; then
            # macOS uses the regular PyPI wheels.  In particular, the Linux
            # provider-matched wheel used by the OpenBLAS image cannot be
            # installed on this platform.
            uv sync --frozen || uv sync || exit 1
        elif [[ -n "${BENCHMARK_TORCH_WHEEL:-}" ]]; then
            # The OpenBLAS image supplies a source-built wheel. Do not replace
            # it with the lockfile's provider-mismatched binary distribution.
            local wheel
            [[ -e "$BENCHMARK_TORCH_WHEEL" ]] || exit 1
            wheel="$(realpath -- "$BENCHMARK_TORCH_WHEEL")" || exit 1
            uv venv --allow-existing .venv || exit 1
            # Prune wheel-only CUDA dependencies too, while retaining the
            # lockfile versions for the rest of the benchmark environment.
            set -o pipefail
            uv export --frozen --no-hashes --no-emit-project --prune torch |
                uv pip install --reinstall-package torch -r - "$wheel" || exit 1
        else
            # A plain `uv sync` resolves torch from PyPI, where the Linux wheel
            # is the CUDA build and drags in roughly 2.5 GB of nvidia-* and
            # triton packages that a CPU profile never loads, which is what
            # filled the hosted runner's disk (#342).  Install the lockfile
            # without torch, then the CPU wheel at the pinned version.
            local torch_version
            torch_version="$(locked_torch_version "$project_dir")"
            if [[ -z "$torch_version" ]]; then
                echo "cannot read the locked torch version from uv.lock" >&2
                exit 1
            fi
            uv venv --allow-existing .venv || exit 1
            set -o pipefail
            uv export --frozen --no-hashes --no-emit-project --prune torch |
                uv pip install -r - || exit 1
            uv pip install --index-url "https://download.pytorch.org/whl/cpu" \
                --force-reinstall "torch==${torch_version}+cpu" || exit 1
        fi
    ) || return 1

    # Keep subsequent `uv run` invocations from syncing the lockfile back to a
    # non-CPU PyTorch wheel during benchmark collection.
    export UV_NO_SYNC=1
}
