#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# CPU permutation / materialize-kernel benchmark suite (cpu/permutation):
#   - naive odometer baseline, tenferro-rs transpose paths (Rust)
#   - strided-rs (Rust, `strided-rs` Cargo feature, enabled by default here;
#     requires extern/strided-rs, cloned by scripts/setup_extern_deps.sh)
#   - HPTT (Rust, `hptt` Cargo feature, NOT enabled by default; requires cmake
#     and a C++ toolchain. Builds on both macOS, e.g. `brew install cmake`,
#     and Linux; opt in via PERMUTATION_EXTRA_FEATURES=hptt below)
#   - Julia Base + Strided.jl (Julia)
#
# Usage: ./scripts/run_permutation.sh <NUM_THREADS...>
#   e.g. ./scripts/run_permutation.sh 1
#        ./scripts/run_permutation.sh 1 4
#
# Each thread count is run as a fresh Rust process and a fresh Julia process
# (RAYON_NUM_THREADS / OMP_NUM_THREADS / JULIA_NUM_THREADS are only read at
# process start, e.g. rayon's global pool is sized once, lazily, from
# RAYON_NUM_THREADS on first use), so multiple thread counts are looped over
# SEQUENTIALLY within one run directory/timestamp rather than run
# concurrently or reusing a single long-lived process. Per-thread-count JSONL
# outputs (rust_output_t<N>.jsonl, julia_output_t<N>.jsonl) and run metadata
# (run_t<N>.yaml) are written under one
# data/results/<target_profile>/cpu/permutation/<timestamp>/ directory, and
# the formatter runs once at the end over all of them so the latest report
# has one table per thread count, per docs/permutation-suite.md.
#
# This is a STANDALONE entry point. It is intentionally not wired into
# scripts/run_all.sh's default path; run it explicitly when you want
# cpu/permutation results.
#
# Compares permutation-kernel performance for the patterns listed in
# benchmarks/cpu/permutation.yaml / data/instances/permutation_patterns.json
# and writes result/<target_profile>/cpu/permutation.md.
#
# Environment variables:
# - PERMUTATION_EXTRA_FEATURES: comma-separated extra Cargo features appended
#   to benchmark_permutation's build (e.g. `hptt` once cmake and a C++
#   toolchain are available). Empty by default; `strided-rs` is already on by
#   default (see CARGO_FEATURES below) since extern/strided-rs is fetched by
#   scripts/setup_extern_deps.sh the same as extern/tenferro-rs.
# - PATTERN_ID / BENCH_RUNS / BENCH_WARMUPS: forwarded to both runners, as in
#   src/bin/benchmark_permutation.rs / scripts/benchmark_permutation.jl.
#
# Runs the Rust and Julia runners SEQUENTIALLY (never concurrently), and runs
# each requested thread count SEQUENTIALLY (never concurrently), per
# AGENTS.md timing discipline.
# ---------------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <NUM_THREADS...>  (e.g. $0 1  or  $0 1 4)" >&2
    exit 1
fi
THREAD_COUNTS=("$@")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=scripts/cpu_blas_provider.sh
source "$SCRIPT_DIR/cpu_blas_provider.sh"

TENFERRO_CPU_FEATURES="$(normalize_cpu_blas_features "${TENFERRO_CPU_FEATURES:-}")"
export TENFERRO_CPU_FEATURES

PERMUTATION_EXTRA_FEATURES="${PERMUTATION_EXTRA_FEATURES:-}"
# `strided-rs` is on by default: extern/strided-rs is fetched unconditionally
# by setup_extern_deps.sh (below), so the strided-rs participant is available
# out of the box. `hptt` stays opt-in via PERMUTATION_EXTRA_FEATURES since it
# additionally requires cmake and a C++ toolchain.
CARGO_FEATURES="$TENFERRO_CPU_FEATURES,strided-rs"
[[ -n "$PERMUTATION_EXTRA_FEATURES" ]] && CARGO_FEATURES="$CARGO_FEATURES,$PERMUTATION_EXTRA_FEATURES"

default_target_profile() {
    case "$(benchmark_host_os)" in
        Darwin) printf '%s\n' "mac-cpu" ;;
        *) printf '%s\n' "amd-cpu" ;;
    esac
}

BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-$(default_target_profile)}"
case "$BENCHMARK_TARGET_PROFILE" in
    mac-cpu|amd-cpu)
        export BENCHMARK_TARGET_PROFILE
        ;;
    *)
        echo "ERROR: BENCHMARK_TARGET_PROFILE must be mac-cpu or amd-cpu for cpu/permutation." >&2
        exit 1
        ;;
esac

# shellcheck source=scripts/thread_env.sh
source "$SCRIPT_DIR/thread_env.sh"

if [[ "${SKIP_EXTERN_SETUP:-0}" != "1" ]]; then
    # shellcheck source=scripts/setup_extern_deps.sh
    source "$SCRIPT_DIR/setup_extern_deps.sh"
fi

ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"

RESULTS_ROOT="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
SUITE_ID="cpu/permutation"
SUITE_FILE="$PROJECT_DIR/benchmarks/cpu/permutation.yaml"
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TIMESTAMP_RFC3339="$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/cpu/permutation/$TIMESTAMP"
RUN_YAML="$RUN_DIR/run.yaml"
LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/permutation.md"

mkdir -p "$RUN_DIR" "$(dirname "$LATEST_REPORT")"

PYTHON=(python3)
if command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run python)
fi

"${PYTHON[@]}" "$SCRIPT_DIR/validate_benchmark_suite.py" "$SUITE_FILE"

blas_impl="$(blas_impl_for_features "$TENFERRO_CPU_FEATURES")"
tenferro_dir="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
tenferro_commit=""
if [[ -d "$tenferro_dir/.git" ]] && command -v git >/dev/null 2>&1; then
    tenferro_commit="$(git -C "$tenferro_dir" rev-parse HEAD 2>/dev/null || true)"
fi

write_run_metadata() {
    local output="$1"
    local args=(
        --suite-id "$SUITE_ID"
        --target-profile "$BENCHMARK_TARGET_PROFILE"
        --suite-file "${SUITE_FILE#$PROJECT_DIR/}"
        --timestamp "$RUN_TIMESTAMP_RFC3339"
        --tenferro-dir "$tenferro_dir"
        --features "$CARGO_FEATURES"
        --blas "$blas_impl"
        --output "$output"
    )
    [[ -n "$tenferro_commit" ]] && args+=(--tenferro-commit "$tenferro_commit")
    "${PYTHON[@]}" "$SCRIPT_DIR/collect_run_metadata.py" "${args[@]}"
}

# Canonical run.yaml for the whole invocation (report header metadata is
# thread-count independent; per-thread-count environment snapshots are
# written separately below as run_t<N>.yaml).
write_run_metadata "$RUN_YAML"

echo "============================================"
echo " CPU permutation benchmark suite"
echo "============================================"
echo "Suite:        $SUITE_ID"
echo "Suite file:   ${SUITE_FILE#$PROJECT_DIR/}"
echo "Target:       $BENCHMARK_TARGET_PROFILE"
echo "Thread counts: ${THREAD_COUNTS[*]}"
echo "Timestamp:    $TIMESTAMP"
echo "Run dir:      ${RUN_DIR#$PROJECT_DIR/}"
echo "Cargo features: $CARGO_FEATURES"
echo ""

echo "Building benchmark_permutation (features: $CARGO_FEATURES)..."
cargo build --release --features "$CARGO_FEATURES" --bin benchmark_permutation

HAVE_JULIA=0
if command -v julia >/dev/null 2>&1; then
    HAVE_JULIA=1
    echo "Instantiating Julia project..."
    (cd "$PROJECT_DIR" && julia --project="$PROJECT_DIR" -e 'import Pkg; Pkg.instantiate()')
else
    echo "WARNING: julia not found on PATH; skipping julia-base / strided-jl participants." >&2
fi

INPUTS=()

for NUM_THREADS in "${THREAD_COUNTS[@]}"; do
    echo ""
    echo "--- Thread count: $NUM_THREADS ---"

    # Fresh process per thread count below: rayon's global pool (Rust) and
    # Julia's thread pool are both sized once at process start from these
    # environment variables, so re-exporting them here only takes effect
    # because each benchmark invocation is a new process. Do not turn this
    # loop into a single long-lived process.
    configure_cpu_thread_env "$NUM_THREADS"
    export JULIA_NUM_THREADS="$NUM_THREADS"
    print_cpu_thread_env

    RUST_JSONL="$RUN_DIR/rust_output_t${NUM_THREADS}.jsonl"
    RUN_T_YAML="$RUN_DIR/run_t${NUM_THREADS}.yaml"
    write_run_metadata "$RUN_T_YAML"

    echo "Running Rust permutation benchmarks (threads=$NUM_THREADS)..."
    BENCH_OUTPUT="$RUST_JSONL" "$PROJECT_DIR/target/release/benchmark_permutation"
    INPUTS+=("$RUST_JSONL")

    JULIA_JSONL="$RUN_DIR/julia_output_t${NUM_THREADS}.jsonl"
    : > "$JULIA_JSONL"
    if [[ "$HAVE_JULIA" == "1" ]]; then
        echo "Running Julia permutation benchmarks (threads=$NUM_THREADS)..."
        (
            cd "$PROJECT_DIR"
            JULIA_PROJECT="$PROJECT_DIR" BENCH_OUTPUT="$JULIA_JSONL" \
                julia --project="$PROJECT_DIR" "$SCRIPT_DIR/benchmark_permutation.jl"
        )
        [[ -s "$JULIA_JSONL" ]] && INPUTS+=("$JULIA_JSONL")
    fi
done

# ---------------------------------------------------------------------------
# Format the latest report once, over every thread count's JSONL output.
# ---------------------------------------------------------------------------
echo ""
echo "Formatting cpu/permutation results as markdown..."
"${PYTHON[@]}" "$SCRIPT_DIR/format_permutation_results.py" \
    "${INPUTS[@]}" \
    --run-metadata "$RUN_YAML" \
    --output "$LATEST_REPORT"

echo ""
echo "CPU permutation benchmark suite complete: $SUITE_ID"
echo "Run YAML:  ${RUN_YAML#$PROJECT_DIR/}"
for path in "${INPUTS[@]}"; do
    echo "JSONL:     ${path#$PROJECT_DIR/}"
done
echo "Report:    ${LATEST_REPORT#$PROJECT_DIR/}"
