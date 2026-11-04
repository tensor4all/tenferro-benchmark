#!/usr/bin/env bash
# Independent latest-main MWEs for registered cpu/perf_issues follow-up cases.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/tenferro_checkout.sh
source scripts/tenferro_checkout.sh
source scripts/thread_env.sh
source scripts/benchmark_host_idle.sh
# Build the independent Rust MWE into the directory the collector defaults to
# ($PWD/target/followup-current, see DEFAULT_MWE_BINARY in
# scripts/cpu_followup_cases.py). --prepare does the full environment setup and
# then calls this; --prepare-mwe only builds, for callers that already prepared
# the environment (the nightly perf_issues path).
build_followup_mwe() {
    : "${MKLROOT:?MKLROOT is required to build mwe/cpu_followup (system-mkl)}"
    test -d "$MKLROOT/include"
    export CARGO_TARGET_DIR="$PWD/target/followup-current"
    cargo build --release --locked --manifest-path mwe/cpu_followup/Cargo.toml
    # The binary just built is pinned to this revision, which the collection step
    # may no longer see if the checkout moves before it runs.
    tenferro_checkout_commit "$PWD/extern/tenferro-rs" > "$CARGO_TARGET_DIR/.tenferro-commit"
}
if [[ "${1:-}" == --prepare ]]; then
    source scripts/python_venv.sh
    export BENCHMARK_TARGET_PROFILE=amd-cpu
    # Both indexes are the established official PyPI/PyTorch sources. The CPU
    # index also carries older unrelated packages such as tqdm.
    export UV_INDEX_STRATEGY=unsafe-best-match
    reset_benchmark_python_venv "$PWD"
    prepare_cpu_benchmark_python_venv "$PWD"
    update_tenferro_checkout_if_main "$PWD/extern/tenferro-rs"
    build_followup_mwe
    exit 0
fi
if [[ "${1:-}" == --prepare-mwe ]]; then
    export BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-amd-cpu}"
    build_followup_mwe
    exit 0
fi
THREADS="${1:?thread count required}"; shift
configure_cpu_thread_env "$THREADS"
assert_benchmark_host_idle
source scripts/cpu_blas_provider.sh
ensure_blas_env_for_features blas-mkl

# Record what this run collected with, in the same run-metadata shape as every
# other suite: host + thread environment and `environment.container` (the
# collector runs inside the MKL devcontainer or on a host without one). The
# results file keeps its own --output contract; the record is a sibling
# `<output>.run.yaml`.
OUTPUT_PATH=""
previous=""
for argument in "$@"; do
    [[ "$previous" == --output ]] && OUTPUT_PATH="$argument"
    previous="$argument"
done
if [[ -n "$OUTPUT_PATH" ]]; then
    metadata_args=(
        --suite-id cpu/followup
        --target-profile "${BENCHMARK_TARGET_PROFILE:-amd-cpu}"
        --suite-file benchmarks/cpu/perf_issues.yaml
        --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        --features blas-mkl --blas mkl
        --output "$OUTPUT_PATH.run.yaml"
    )
    # Prefer the revision --prepare built the MWE from over whatever the
    # checkout holds now.
    prepared_commit_file="${CARGO_TARGET_DIR:-$PWD/target/followup-current}/.tenferro-commit"
    if [[ -s "$prepared_commit_file" ]]; then
        metadata_args+=(--tenferro-commit "$(cat "$prepared_commit_file")")
    else
        metadata_args+=(--tenferro-dir "$PWD/extern/tenferro-rs")
    fi
    .venv/bin/python scripts/collect_run_metadata.py "${metadata_args[@]}"
fi
exec .venv/bin/python scripts/cpu_followup_cases.py --threads "$THREADS" "$@"
