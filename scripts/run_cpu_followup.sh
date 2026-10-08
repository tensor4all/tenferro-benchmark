#!/usr/bin/env bash
# Independent latest-main MWEs for registered cpu/perf_issues follow-up cases.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" == Linux && ! -e /.dockerenv ]]; then
    echo 'Run Linux CPU collection inside the devcontainer.' >&2
    exit 1
fi
source scripts/thread_env.sh
source scripts/benchmark_host_idle.sh
if [[ "${1:-}" == --prepare ]]; then
    source scripts/python_venv.sh
    export BENCHMARK_TARGET_PROFILE=amd-cpu
    # Both indexes are the established official PyPI/PyTorch sources. The CPU
    # index also carries older unrelated packages such as tqdm.
    export UV_INDEX_STRATEGY=unsafe-best-match
    reset_benchmark_python_venv "$PWD"
    prepare_cpu_benchmark_python_venv "$PWD"
    if [[ "$(git -C extern/tenferro-rs branch --show-current)" == main ]]; then
        git -C extern/tenferro-rs pull --ff-only
    fi
    test -d "$MKLROOT/include"
    export CARGO_TARGET_DIR="$PWD/target/followup-b3f47296"
    cargo build --release --locked --manifest-path mwe/cpu_followup/Cargo.toml
    exit 0
fi
THREADS="${1:?thread count required}"; shift
configure_cpu_thread_env "$THREADS"
assert_benchmark_host_idle
export LD_LIBRARY_PATH="$MKLROOT/lib:/opt/intel/oneapi/compiler/2026.1/lib:${LD_LIBRARY_PATH:-}"
exec .venv/bin/python scripts/cpu_followup_cases.py --threads "$THREADS" "$@"
