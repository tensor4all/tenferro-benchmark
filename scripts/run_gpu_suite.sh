#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
wall_now() { python3 -c 'import time; print(time.monotonic_ns())'; }
COMMAND_START_NS="$(wall_now)"
RESULTS_ROOT="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
TIMESTAMP="${GPU_BENCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-nvidia-gpu}"
case "$BENCHMARK_TARGET_PROFILE" in
    mac-cpu|amd-cpu|nvidia-gpu)
        export BENCHMARK_TARGET_PROFILE
        ;;
    *)
        echo "ERROR: BENCHMARK_TARGET_PROFILE must be mac-cpu, amd-cpu, or nvidia-gpu." >&2
        exit 1
        ;;
esac
RUN_TIMESTAMP_RFC3339="$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

SUITES_VALUE="${GPU_BENCH_SUITE:-benchmarks/gpu/dense.yaml,benchmarks/gpu/einsum.yaml,benchmarks/gpu/sparse.yaml,benchmarks/gpu/tensornetwork.yaml}"
BACKENDS_VALUE="${GPU_BENCH_BACKENDS:-tenferro-cuda-trace,tenferro-cuda-eager,pytorch-cuda,jax-cuda,jax-cuda-tree-jit,jax-cuda-eager,cublaslt,cutlass,cusolver,cusparse,ginkgo}"
DEVICE_ORDINAL="${GPU_BENCH_DEVICE:-0}"
if [[ "${GPU_BENCH_VERIFY_ONLY:-0}" == 1 ]]; then
    DEVICE_ORDINAL="${GPU_BENCH_DEVICE:-1}"
    BACKENDS_VALUE="${GPU_BENCH_BACKENDS:-tenferro-cuda-eager,pytorch-cuda,jax-cuda}"
    RESULTS_ROOT="$PROJECT_DIR/data/verification"
fi
# Physical nvidia-smi index -> one visible UUID -> process-local ordinal zero.
# Contract tests can supply fake runners on a host without nvidia-smi.
PHYSICAL_DEVICE_ORDINAL="$DEVICE_ORDINAL"
if command -v nvidia-smi >/dev/null 2>&1; then
    export GPU_BENCH_PHYSICAL_DEVICE="$PHYSICAL_DEVICE_ORDINAL"
    CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$PHYSICAL_DEVICE_ORDINAL" --query-gpu=uuid --format=csv,noheader)"
    [[ "$CUDA_VISIBLE_DEVICES" == GPU-* && "$CUDA_VISIBLE_DEVICES" != *$'\n'* ]] || exit 1
    export CUDA_VISIBLE_DEVICES
    DEVICE_ORDINAL=0
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export MATMUL_NUM_THREADS="${MATMUL_NUM_THREADS:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PROBLEM_FILTER="${GPU_BENCH_PROBLEM:-}"
RUST_MIN_STACK="${RUST_MIN_STACK:-67108864}"

cd "$PROJECT_DIR"

if [[ -d extern/tenferro-rs/.git && "$(git -C extern/tenferro-rs branch --show-current)" == main ]]; then
    git -C extern/tenferro-rs -c 'url.https://github.com/.insteadOf=git@github.com:' pull --ff-only
fi

mkdir -p "$RESULTS_ROOT" "$REPORTS_DIR"

IFS=',' read -r -a SUITES <<< "$SUITES_VALUE"
IFS=',' read -r -a BACKENDS <<< "$BACKENDS_VALUE"

echo "============================================"
echo " GPU benchmark contract suite"
echo "============================================"
echo "Suites:   $SUITES_VALUE"
echo "Backends: $BACKENDS_VALUE"
echo "Device:   cuda:$DEVICE_ORDINAL"
echo "Target:   $BENCHMARK_TARGET_PROFILE"
echo "Timestamp: $TIMESTAMP"
echo ""

# Vendor library locations for JIT-compiled backends (CUTLASS, Ginkgo).
# Override via env if installed elsewhere.
export CUTLASS_DIR="${CUTLASS_DIR:-/opt/cutlass}"
export GINKGO_DIR="${GINKGO_DIR:-/opt/ginkgo}"

VALIDATOR=(python3 "$SCRIPT_DIR/validate_benchmark_suite.py")
FORMATTER=(python3 "$SCRIPT_DIR/format_gpu_results.py")
METADATA=(python3 "$SCRIPT_DIR/collect_run_metadata.py")
PYTHON=(python3)
if command -v uv >/dev/null 2>&1; then
    VALIDATOR=(uv run python "$SCRIPT_DIR/validate_benchmark_suite.py")
    FORMATTER=(uv run python "$SCRIPT_DIR/format_gpu_results.py")
    METADATA=(uv run python "$SCRIPT_DIR/collect_run_metadata.py")
    PYTHON=(uv run python)
fi

if [[ -n "${BENCH_COVERAGE:-}" ]]; then
    COVERAGE="$BENCH_COVERAGE"
elif [[ -n "${GPU_BENCH_SUITE:-}${GPU_BENCH_BACKENDS:-}${GPU_BENCH_PROBLEM:-}" ]]; then
    COVERAGE=full
else
    COVERAGE=quick
fi
SELECTION_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/gpu/routine_selection/$TIMESTAMP"
mapfile -t SUITES < <("${PYTHON[@]}" "$SCRIPT_DIR/gpu_routine_selection.py" \
    --coverage "$COVERAGE" --effort "${BENCH_EFFORT:-standard}" \
    --backends "${GPU_BENCH_BACKENDS:-}" --problem "$PROBLEM_FILTER" \
    --output "$SELECTION_DIR" "${SUITES[@]}")
[[ ${#SUITES[@]} -gt 0 ]] || { echo "No GPU suites selected" >&2; exit 1; }
"${VALIDATOR[@]}" "${SUITES[@]}"

# Split backends: Rust-native vs Python
RUST_BACKENDS=()
PYTHON_BACKENDS=()
for b in "${BACKENDS[@]}"; do
    if [[ "$b" == "tenferro-cuda-trace" || "$b" == "tenferro-cuda-eager" ]]; then
        RUST_BACKENDS+=("$b")
    else
        PYTHON_BACKENDS+=("$b")
    fi
done

resolve_git_commit() {
    local checkout_dir="$1"
    if [[ -d "$checkout_dir/.git" ]] && command -v git >/dev/null 2>&1; then
        git -C "$checkout_dir" rev-parse HEAD 2>/dev/null || true
    fi
}

suite_id_for_file() {
    awk '$1 == "suite_id:" { print $2; exit }' "$1"
}

suite_backends_for_file() {
    "${PYTHON[@]}" - "$1" <<'PY'
import sys, yaml
for backend in yaml.safe_load(open(sys.argv[1]))["backends"]:
    print(backend)
PY
}

backends_for_suite() {
    local suite="$1"
    local -n _out="$2"
    local suite_backend
    local requested
    _out=()
    mapfile -t _suite_backends < <(suite_backends_for_file "$suite")
    for requested in "${BACKENDS[@]}"; do
        for suite_backend in "${_suite_backends[@]}"; do
            if [[ "$requested" == "$suite_backend" ]]; then
                _out+=("$requested")
                break
            fi
        done
    done
}

write_run_metadata() {
    local suite_id="$1"
    local suite_file="$2"
    local run_yaml="$3"
    local tenferro_dir="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
    local tenferro_commit
    tenferro_commit="$(resolve_git_commit "$tenferro_dir")"
    local args=(
        --suite-id "$suite_id"
        --target-profile "$BENCHMARK_TARGET_PROFILE"
        --suite-file "$suite_file"
        --timestamp "$RUN_TIMESTAMP_RFC3339"
        --tenferro-dir "$tenferro_dir"
        --features cuda
        --blas none
        --cuda-device-ordinal "$PHYSICAL_DEVICE_ORDINAL"
        --output "$run_yaml"
    )
    if [[ -n "$tenferro_commit" ]]; then
        args+=(--tenferro-commit "$tenferro_commit")
    fi
    "${METADATA[@]}" "${args[@]}"
    "${PYTHON[@]}" - "$RUN_YAML" "$SELECTION_DIR/selection.json" <<'PY'
import json, sys, yaml
from pathlib import Path
path = Path(sys.argv[1]); metadata = yaml.safe_load(path.read_text())
plan = json.loads(Path(sys.argv[2]).read_text())
metadata.setdefault('collection',{}).update({'coverage':plan['coverage'],'effort':plan['effort'],
    'gpu_routine_manifest_version':plan['manifest_version'],
    'gpu_routine_selection_file':sys.argv[2],
    'expected_case_ids':plan['expected'],'selected_case_ids':plan['selected'],
    'unselected_full_case_ids':plan['unselected']})
path.write_text(yaml.safe_dump(metadata,sort_keys=False))
PY
}

RUST_BIN="${CARGO_TARGET_DIR:-$PROJECT_DIR/target}/release/benchmark_gpu_rust"
RUST_BUILT=0
RUST_BUILD_NS=0
GPU_BENCH_BACKEND_SLEEP="${GPU_BENCH_BACKEND_SLEEP:-2}"

build_rust_benchmark_if_needed() {
    if [[ "$RUST_BUILT" != "0" ]]; then
        return 0
    fi
    if [[ "${GPU_BENCH_PREBUILT_RUST:-0}" == "1" ]]; then
        python3 "$SCRIPT_DIR/ci/gpu_artifact.py" "$RUST_BIN"
        RUST_BUILT=1
        return 0
    fi
    echo "  Building benchmark_gpu_rust..."
    local build_start
    build_start="$(wall_now)"
    CUBECL_DEBUG_LOG=0 \
    CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
    RUST_MIN_STACK="$RUST_MIN_STACK" \
    LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        cargo build --release --features cuda --bin benchmark_gpu_rust 2>&1
    RUST_BUILD_NS=$(( $(wall_now) - build_start ))
    RUST_BUILT=1
}

run_rust_backend() {
    local suite="$1"
    local chunk_jsonl="$2"
    shift 2
    build_rust_benchmark_if_needed
    CUBECL_DEBUG_LOG=0 \
    CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
    RUST_MIN_STACK="$RUST_MIN_STACK" \
    LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        "$RUST_BIN" "$chunk_jsonl" "$DEVICE_ORDINAL" "$PROBLEM_FILTER" \
        "$@" -- "$suite"
}

run_python_backend() {
    local suite="$1"
    local chunk_jsonl="$2"
    shift 2
    "${PYTHON[@]}" "$SCRIPT_DIR/benchmark_gpu_python.py" \
        "$chunk_jsonl" "$DEVICE_ORDINAL" "$PROBLEM_FILTER" \
        "$@" -- "$suite"
}

run_suite_serial_by_backend() {
    local suite="$1"
    local suite_id="$2"
    local run_dir="$3"
    local result_jsonl="$4"
    local backends_to_run=()

    backends_for_suite "$suite" backends_to_run
    if [[ ${#backends_to_run[@]} -eq 0 ]]; then
        echo "No backends from GPU_BENCH_BACKENDS are listed in $suite; skipping runs."
        return 0
    fi

    echo "Running $suite_id serially (one backend per process): ${backends_to_run[*]}"
    for backend in "${backends_to_run[@]}"; do
        echo ""
        echo ">>> Backend: $backend"
        local chunk="$run_dir/chunk_${backend}.jsonl"
        : > "$chunk"
        local backend_start build_before backend_end
        backend_start="$(wall_now)"
        build_before="$RUST_BUILD_NS"

        if [[ "$backend" == "tenferro-cuda-eager" || "$backend" == "tenferro-cuda-trace" ]]; then
            run_rust_backend "$suite" "$chunk" "$backend"
        else
            run_python_backend "$suite" "$chunk" "$backend"
        fi
        backend_end="$(wall_now)"
        "${PYTHON[@]}" - "$run_dir/wall_backends.jsonl" "$backend" "$backend_start" "$backend_end" "$build_before" "$RUST_BUILD_NS" <<'PY'
import json, sys
path, backend = sys.argv[1:3]
start, end, before, after = map(int, sys.argv[3:])
with open(path, 'a') as output:
    output.write(json.dumps({'backend':backend, 'build_seconds':(after-before)/1e9,
        'process_seconds':(end-start-after+before)/1e9,
        'scope':'backend process wall time includes startup, preparation, validation and samples; build separated'})+'\n')
PY

        if [[ -s "$chunk" ]]; then
            cat "$chunk" >> "$result_jsonl"
        fi

        # Separate processes so framework CUDA allocators release device memory.
        sleep "$GPU_BENCH_BACKEND_SLEEP"
    done
}

RESULT_PATHS=()
for suite in "${SUITES[@]}"; do
    suite_id="$(suite_id_for_file "$suite")"
    if [[ ! "$suite_id" =~ ^gpu/[a-z0-9][a-z0-9_.-]*$ ]]; then
        echo "ERROR: unsupported GPU suite_id '$suite_id' in $suite" >&2
        exit 1
    fi

    suite_name="${suite_id#gpu/}"
    RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/gpu/$suite_name/$TIMESTAMP"
    RUN_YAML="$RUN_DIR/run.yaml"
    RESULT_JSONL="$RUN_DIR/records.jsonl"
    RUST_JSONL="$RUN_DIR/rust_records.jsonl"
    MARKDOWN_OUT="$RUN_DIR/report.md"
    REPORT_OUT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/gpu/$suite_name.md"

    mkdir -p "$RUN_DIR" "$(dirname "$REPORT_OUT")"
    : > "$RESULT_JSONL"
    RESULT_PATHS+=("$RESULT_JSONL")
    : > "$RUN_DIR/wall_backends.jsonl"
    write_run_metadata "$suite_id" "$suite" "$RUN_YAML"

    # Each framework releases its CUDA allocator before the next process starts.
    COLLECTION_START_NS="$(wall_now)"
    run_suite_serial_by_backend "$suite" "$suite_id" "$RUN_DIR" "$RESULT_JSONL"
    COLLECTION_END_NS="$(wall_now)"

    if [[ ! -s "$RESULT_JSONL" ]]; then
        echo "No records generated for $suite_id; skipping report."
        continue
    fi
    if [[ "${GPU_BENCH_VERIFY_ONLY:-0}" == 1 ]]; then
        "${PYTHON[@]}" - "$RESULT_JSONL" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
failures = [row for row in rows if row["status"] not in {"verification_only", "unsupported"}]
if failures:
    raise SystemExit(f"GPU numerical verification failed: {failures}")
print(f"Verified {sum(row['status']=='verification_only' for row in rows)} rows; no timing report published")
PY
        continue
    fi

    "${VALIDATOR[@]}" --kind run "$RUN_YAML"
    "${VALIDATOR[@]}" --kind result "$RESULT_JSONL"

    REPORT_START_NS="$(wall_now)"
    "${FORMATTER[@]}" "$RESULT_JSONL" --run-metadata "$RUN_YAML" --output "$MARKDOWN_OUT"
    if [[ "${GPU_BENCH_SKIP_LATEST:-0}" != 1 ]]; then
        cp "$MARKDOWN_OUT" "$REPORT_OUT"
    fi
    "${PYTHON[@]}" - "$RUN_DIR/wall_command.json" "$COMMAND_START_NS" "$COLLECTION_START_NS" "$COLLECTION_END_NS" "$REPORT_START_NS" "$(wall_now)" <<'PY'
import json, sys
from pathlib import Path
start, collection, collected, report, end = map(int, sys.argv[2:])
Path(sys.argv[1]).write_text(json.dumps({
    'command_seconds_through_this_suite':(end-start)/1e9,
    'collection_seconds_including_build_and_backend_release_delays':(collected-collection)/1e9,
    'report_generation_and_copy_seconds':(end-report)/1e9,
    'scope':'script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded',
},indent=2)+'\n')
PY

    echo ""
    echo "GPU benchmark suite complete: $suite_id"
    echo "Run YAML: $RUN_YAML"
    echo "JSONL:    $RESULT_JSONL"
    echo "Markdown: $MARKDOWN_OUT"
    echo "Report:   $REPORT_OUT"
done

"${PYTHON[@]}" "$SCRIPT_DIR/gpu_routine_selection.py" --output "$SELECTION_DIR" --records "${RESULT_PATHS[@]}"
"${PYTHON[@]}" "$SCRIPT_DIR/format_gpu_routine_coverage.py" "$SELECTION_DIR/case_status.json" \
    --output "$SELECTION_DIR/report.md"
if [[ "${GPU_BENCH_SKIP_LATEST:-0}" != 1 && "${GPU_BENCH_VERIFY_ONLY:-0}" != 1 ]]; then
    mkdir -p "$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/gpu"
    cp "$SELECTION_DIR/report.md" "$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/gpu/routine_coverage.md"
fi

CONTROLS="${GPU_BENCH_ROUTINE_CONTROLS:-}"
if [[ -z "$CONTROLS" ]]; then
    if [[ -z "${GPU_BENCH_SUITE:-}${GPU_BENCH_BACKENDS:-}${GPU_BENCH_PROBLEM:-}" ]]; then
        CONTROLS=1
    else
        CONTROLS=0
    fi
fi
[[ "$CONTROLS" == 0 || "$CONTROLS" == 1 ]] || { echo "Invalid GPU_BENCH_ROUTINE_CONTROLS" >&2; exit 1; }
if [[ "$CONTROLS" == 1 ]]; then
    BENCH_COVERAGE="$COVERAGE" GPU_BENCH_TIMESTAMP="$TIMESTAMP" \
        bash "$SCRIPT_DIR/run_gpu_routine_controls.sh"
fi
COMMAND_COMPLETE_NS="$(wall_now)"
"${PYTHON[@]}" - "$SELECTION_DIR/wall_command_complete.json" "$COMMAND_START_NS" "$COMMAND_COMPLETE_NS" "$CONTROLS" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    'script_seconds_before_final_wall_record_write': (int(sys.argv[3])-int(sys.argv[2]))/1e9,
    'routine_controls_selected': sys.argv[4] == '1',
    'scope': 'script clock read through core and control collection, validation, accounting and report writes; excludes devcontainer CLI startup and final wall metadata/report formatting'
}, indent=2)+'\n')
PY
if [[ "${GPU_BENCH_VERIFY_ONLY:-0}" != 1 ]]; then
    "${PYTHON[@]}" "$SCRIPT_DIR/format_gpu_routine_wall.py" "$SELECTION_DIR" --output "$SELECTION_DIR/wall_report.md"
    if [[ "${GPU_BENCH_SKIP_LATEST:-0}" != 1 ]]; then
        cp "$SELECTION_DIR/wall_report.md" "$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/gpu/routine_workflow.md"
    fi
fi

echo ""
echo "GPU benchmark suites complete"
