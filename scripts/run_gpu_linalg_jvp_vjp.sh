#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/gpu_preflight.sh
source "$SCRIPT_DIR/gpu_preflight.sh"
# This collector owns its arm loop, so it writes the wall records the generic
# driver would have written: one row per arm process, and the non-arm stages
# (build, separate CPU verification pass, report) in `wall_phases.json`. Without
# them the suite published an empty wall file and could only be described as
# "collection time".
wall_now() { python3 -c 'import time; print(time.monotonic_ns())'; }
wall_arm() {
    python3 "$SCRIPT_DIR/gpu_wall_phase.py" arm --output "$1/wall_backends.jsonl" \
        --backend "$2" --start-ns "$3" --end-ns "$4" --build-seconds "$5"
}
wall_phase() {
    python3 "$SCRIPT_DIR/gpu_wall_phase.py" phase --output "$1/wall_phases.json" \
        --phase "$2" --start-ns "$3" --end-ns "$4"
}

export GPU_BENCH_VERIFY_ONLY="${GPU_BENCH_VERIFY_ONLY:-0}"
default_device=0
[[ "$GPU_BENCH_VERIFY_ONLY" == 1 ]] && default_device=1
configure_gpu_bench_env "$default_device"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RESULTS_ROOT="$PROJECT_DIR/data/results"
[[ "$GPU_BENCH_VERIFY_ONLY" == 1 ]] && RESULTS_ROOT="$PROJECT_DIR/data/verification"
REPORTS_DIR="$PROJECT_DIR/result"
TIMESTAMP="${GPU_BENCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-nvidia-gpu}"
# `LINALG_AD_SUITE_FILE` selects the suite document; the report path and
# suite_id are derived from it. Defaults to the GPU-sized JVP/VJP suite; the
# n=2/4/8 latency suite lives in `benchmarks/gpu/linalg_ad_latency.yaml` and is
# normally driven by `scripts/run_gpu_linalg_ad_latency.sh`.
SUITE_FILE="${LINALG_AD_SUITE_FILE:-$PROJECT_DIR/benchmarks/gpu/linalg_jvp_vjp.yaml}"
DEVICE_ORDINAL="${GPU_BENCH_DEVICE:-0}"
RUST_MIN_STACK="${RUST_MIN_STACK:-67108864}"

case "$BENCHMARK_TARGET_PROFILE" in
    nvidia-gpu)
        ;;
    *)
        echo "ERROR: GPU linalg JVP/VJP collection requires BENCHMARK_TARGET_PROFILE=nvidia-gpu." >&2
        exit 1
        ;;
esac

# Same rule as run_gpu_suite.sh: the run metadata timestamp identifies the run
# directory, so it is derived from that stamp rather than the wall clock.
RUN_TIMESTAMP_RFC3339="${GPU_BENCH_RUN_TIMESTAMP:-$(python3 - "$TIMESTAMP" <<'PY'
from datetime import datetime, timezone
import sys
print(datetime.strptime(sys.argv[1], "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).isoformat())
PY
)}"

cd "$PROJECT_DIR"
tenferro_dir="$PROJECT_DIR/extern/tenferro-rs"
update_tenferro_checkout_if_main "$tenferro_dir"

VALIDATOR=(python3 "$SCRIPT_DIR/validate_benchmark_suite.py")
METADATA=(python3 "$SCRIPT_DIR/collect_run_metadata.py")
FORMATTER=(python3 "$SCRIPT_DIR/format_gpu_linalg_ad_results.py")
PYTHON=(python3)
if command -v uv >/dev/null 2>&1; then
    VALIDATOR=(uv run python "$SCRIPT_DIR/validate_benchmark_suite.py")
    METADATA=(uv run python "$SCRIPT_DIR/collect_run_metadata.py")
    FORMATTER=(uv run python "$SCRIPT_DIR/format_gpu_linalg_ad_results.py")
    PYTHON=(uv run python)
fi

# Regenerate the repository's own suites from their seeds, but never a suite
# file handed in by the routine selection (that one is already curated).
for repo_suite in "$PROJECT_DIR/benchmarks/gpu/linalg_jvp_vjp.yaml" \
                  "$PROJECT_DIR/benchmarks/gpu/linalg_ad_latency.yaml"; do
    if [[ "$SUITE_FILE" == "$repo_suite" ]]; then
        "${PYTHON[@]}" "$SCRIPT_DIR/generate_gpu_linalg_jvp_vjp_yaml.py" >/dev/null
        break
    fi
done
"${VALIDATOR[@]}" "$SUITE_FILE"

suite_id="$(awk '$1 == "suite_id:" { print $2; exit }' "$SUITE_FILE")"
suite_name="${suite_id#gpu/}"
RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/gpu/$suite_name/$TIMESTAMP"
RUN_YAML="$RUN_DIR/run.yaml"
RESULT_JSONL="$RUN_DIR/records.jsonl"
RUST_JSONL="$RUN_DIR/rust_records.jsonl"
PYTHON_JSONL="$RUN_DIR/python_records.jsonl"
MARKDOWN_OUT="$RUN_DIR/report.md"
REPORT_OUT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/gpu/$suite_name.md"
GPU_BENCH_BACKEND_SLEEP="${GPU_BENCH_BACKEND_SLEEP:-2}"

mkdir -p "$RUN_DIR" "$(dirname "$REPORT_OUT")"
: > "$RESULT_JSONL"

tenferro_commit="$(tenferro_checkout_commit "$tenferro_dir")"
metadata_args=(
    --suite-id "$suite_id"
    --target-profile "$BENCHMARK_TARGET_PROFILE"
    --suite-file "$SUITE_FILE"
    --timestamp "$RUN_TIMESTAMP_RFC3339"
    --tenferro-dir "$tenferro_dir"
    --features cuda
    --blas none
    --cuda-device-ordinal "$GPU_BENCH_PHYSICAL_DEVICE"
    --output "$RUN_YAML"
)
if [[ -n "$tenferro_commit" ]]; then
    metadata_args+=(--tenferro-commit "$tenferro_commit")
fi
"${METADATA[@]}" "${metadata_args[@]}"

echo "============================================"
echo " $suite_id benchmark suite"
echo "============================================"
echo "Suite:    $SUITE_FILE"
echo "Device:   cuda:$DEVICE_ORDINAL"
echo "Target:   $BENCHMARK_TARGET_PROFILE"
echo "Timestamp: $TIMESTAMP"
echo ""

# An arm that dies must not cost this suite the rows the other arms measured:
# everything that was written is still validated, formatted and published, and
# the failure is reported by the exit status after the report is written.
ARM_FAILED=0
RUST_BIN="${CARGO_TARGET_DIR:-$PROJECT_DIR/target}/release/benchmark_gpu_linalg_ad"
# The routine run hands in the backend arm set through GPU_BENCH_BACKENDS;
# without a filter the suite's own declared arms all run.
mapfile -t AD_BACKENDS < <("${PYTHON[@]}" - "$SUITE_FILE" "${GPU_BENCH_BACKENDS:-}" <<'PY'
import sys, yaml
declared = yaml.safe_load(open(sys.argv[1])).get("backends") or []
requested = [name for name in sys.argv[2].split(",") if name] or declared
for name in requested:
    if name in declared:
        print(name)
PY
)
[[ ${#AD_BACKENDS[@]} -gt 0 ]] || { echo "No linalg AD backends selected" >&2; exit 1; }
RUN_RUST=0
RUN_RUST_EAGER=0
for backend in "${AD_BACKENDS[@]}"; do
    case "$backend" in
        tenferro-cuda-trace) RUN_RUST=1 ;;
        tenferro-cuda-eager) RUN_RUST_EAGER=1 ;;
    esac
done
rust_build_seconds=0.0
# A run with a dead arm still renders the rows that ran; the exit status below
# is what tells the caller the matrix is incomplete.
PARTIAL_ARGS=()
if [[ "$RUN_RUST" == 1 || "$RUN_RUST_EAGER" == 1 ]]; then
    rust_build_started="$(wall_now)"
    if [[ "${GPU_BENCH_PREBUILT_RUST:-0}" == "1" ]]; then
        # The hosted build job ships this binary; the benchmark pod has no Rust toolchain.
        python3 "$SCRIPT_DIR/ci/gpu_artifact.py" "$RUST_BIN"
    else
        echo "Building benchmark_gpu_linalg_ad..."
        CUBECL_DEBUG_LOG=0 \
        RUST_MIN_STACK="$RUST_MIN_STACK" \
            cargo build --release --features cuda --bin benchmark_gpu_linalg_ad
    fi
    wall_phase "$RUN_DIR" rust_build "$rust_build_started" "$(wall_now)"
    rust_build_seconds="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["phases_seconds"]["rust_build"])' "$RUN_DIR/wall_phases.json")"

    # Every Rust arm is one process behind the same binary; the arm name is the
    # process-wide selection. Each arm's raw outputs are checked by the
    # backend-agnostic CPU reference pass before they reach the result stream.
    run_rust_arm() {
        local backend="$1" raw="$2" verified="$3" rc=0 started
        echo "Running $backend linalg AD benchmarks..."
        : > "$raw"
        started="$(wall_now)"
        GPU_BENCH_AD_ARM="$backend" \
        CUBECL_DEBUG_LOG=0 \
        RUST_MIN_STACK="$RUST_MIN_STACK" \
            "$RUST_BIN" "$raw" "$DEVICE_ORDINAL" "" -- "$SUITE_FILE" || rc=$?
        wall_arm "$RUN_DIR" "$backend" "$started" "$(wall_now)" "$rust_build_seconds"
        if [[ "$rc" != 0 ]]; then
            echo "ARM FAILED: $backend exited $rc; its rows stay in $raw" >&2
            ARM_FAILED=1
            return 0
        fi
        # The verification pass is a stage of the suite, not part of any arm's
        # clock; the raw file stays beside the verified one when it fails.
        local verify_started
        verify_started="$(wall_now)"
        if ! "${PYTHON[@]}" "$SCRIPT_DIR/verify_gpu_linalg_ad_rust.py" \
                "$raw" "$verified" "$SUITE_FILE"; then
            echo "ARM FAILED: $backend verification failed; its rows stay unverified" >&2
            ARM_FAILED=1
            return 0
        fi
        wall_phase "$RUN_DIR" rust_verification "$verify_started" "$(wall_now)"
        [[ ! -s "$verified" ]] || cat "$verified" >> "$RESULT_JSONL"
    }

    [[ "$RUN_RUST" != 1 ]] || run_rust_arm tenferro-cuda-trace \
        "$RUN_DIR/rust_records_unverified.jsonl" "$RUST_JSONL"
    [[ "$RUN_RUST_EAGER" != 1 ]] || run_rust_arm tenferro-cuda-eager \
        "$RUN_DIR/rust_eager_records_unverified.jsonl" "$RUN_DIR/rust_eager_records.jsonl"
    sleep "$GPU_BENCH_BACKEND_SLEEP"
fi

# One Python arm. Its rows stay and are published even when the arm exits
# non-zero: the failure is reported after the report exists, not instead of it.
run_python_arm() {
    local backend="$1" chunk="$2" rc=0 started
    started="$(wall_now)"
    "${PYTHON[@]}" "$SCRIPT_DIR/benchmark_gpu_linalg_ad.py" \
        "$chunk" "$DEVICE_ORDINAL" "" "$backend" -- "$SUITE_FILE" || rc=$?
    wall_arm "$RUN_DIR" "$backend" "$started" "$(wall_now)" 0.0
    if [[ "$rc" != 0 ]]; then
        echo "ARM FAILED: $backend exited $rc; its rows stay in $chunk" >&2
        ARM_FAILED=1
    fi
    [[ -s "$chunk" ]] || return 0
    cat "$chunk" >> "$RESULT_JSONL"
}

for backend in "${AD_BACKENDS[@]}"; do
    case "$backend" in
        tenferro-cuda-trace|tenferro-cuda-eager) continue ;;
    esac
    echo "Running $backend linalg AD benchmarks..."
    chunk="$RUN_DIR/chunk_${backend}.jsonl"
    : > "$chunk"
    run_python_arm "$backend" "$chunk"
    sleep "$GPU_BENCH_BACKEND_SLEEP"
done

if [[ ! -s "$RESULT_JSONL" ]]; then
    echo "No records generated for $suite_id; skipping report." >&2
    exit 1
fi

if [[ "$GPU_BENCH_VERIFY_ONLY" == 1 ]]; then
    "${PYTHON[@]}" - "$RESULT_JSONL" <<'PYDEBUG'
import json,sys
rows=[json.loads(line) for line in open(sys.argv[1])]
assert all(r['status'] in {'verification_only','unsupported'} for r in rows), rows
assert all(r['verification']['status']=='passed' for r in rows if r['status']=='verification_only')
assert all(r['timing']['median_ms'] is None for r in rows)
print('AD CUDA verification complete:',len(rows),'cells; no timing report updated')
PYDEBUG
    [[ "$ARM_FAILED" == 0 ]] || exit 1
    exit 0
fi
"${VALIDATOR[@]}" --kind run "$RUN_YAML"
"${VALIDATOR[@]}" --kind result "$RESULT_JSONL"
report_started="$(wall_now)"
[[ "$ARM_FAILED" == 0 ]] || PARTIAL_ARGS+=(--allow-partial)
"${FORMATTER[@]}" "$RESULT_JSONL" --run-metadata "$RUN_YAML" \
    "${PARTIAL_ARGS[@]+"${PARTIAL_ARGS[@]}"}" --output "$MARKDOWN_OUT"
if [[ "${GPU_BENCH_SKIP_LATEST:-0}" != 1 ]]; then
    cp "$MARKDOWN_OUT" "$REPORT_OUT"
    if [[ -f "${MARKDOWN_OUT%.md}.summary.json" ]]; then
        cp "${MARKDOWN_OUT%.md}.summary.json" "${REPORT_OUT%.md}.summary.json"
        "${PYTHON[@]}" "$SCRIPT_DIR/ci/result_ingestion.py" "$REPORT_OUT"
    fi
fi

wall_phase "$RUN_DIR" report "$report_started" "$(wall_now)"
echo ""
echo "GPU linalg JVP/VJP benchmark suite complete: $suite_id"
echo "Run YAML: $RUN_YAML"
echo "JSONL:    $RESULT_JSONL"
echo "Report:   $REPORT_OUT"
if [[ "$ARM_FAILED" != 0 ]]; then
    echo "SUITE INCOMPLETE: at least one AD arm failed; the rows that did run are published" >&2
    exit 1
fi
