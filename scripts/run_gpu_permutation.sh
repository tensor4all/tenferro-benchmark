#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# GPU permutation / materialize-kernel benchmark suite (gpu/permutation):
#   - tenferro-cuda-transpose, tenferro-cuda-to-contiguous,
#     tenferro-cuda-destination-reuse, cutensor (Rust)
#   - pytorch-cuda, memcpy-d2d (Python)
#
# Usage: BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_permutation.sh
#
# Ports scripts/run_permutation.sh (cpu/permutation) to CUDA, minus the
# thread-count loop (this suite has no CPU-thread dimension). Runs the Rust
# runner, then EACH Python backend as its own process, all SEQUENTIALLY
# (never concurrently), per AGENTS.md timing discipline and GPU Timing
# Fairness. The per-backend process split follows scripts/run_gpu_suite.sh
# so framework CUDA allocators release device memory between backends.
#
# This is a STANDALONE entry point, exactly like scripts/run_permutation.sh
# is for cpu/permutation: it is intentionally not wired into
# scripts/run_gpu_suite.sh.
#
# Compares permutation-kernel performance for the patterns listed in
# benchmarks/gpu/permutation.yaml / data/instances/gpu_permutation_patterns.json
# (`participants_gpu` per pattern) and writes
# result/nvidia-gpu/gpu/permutation.md.
#
# Environment variables:
# - PATTERN_ID / BENCH_RUNS / BENCH_WARMUPS: forwarded to both runners, as
#   in src/bin/benchmark_gpu_permutation.rs /
#   scripts/benchmark_gpu_permutation_python.py.
# - PERMUTATION_FULL_VERIFY=1: keep the legacy full host reference +
#   elementwise compare. The default (`sampled`) correctness gate checks a
#   fixed set of corners plus roughly 1M seeded output elements against the
#   source index mapping, in both runners; each backend gathers/copies only
#   those sampled elements on the device. The mode is recorded in run.yaml and
#   the report.
# - GPU_BENCH_DEVICE: CUDA device ordinal (default 0).
# - GPU_BENCH_BACKEND_SLEEP: pause in seconds between per-backend Python
#   processes (default 2, same as scripts/run_gpu_suite.sh).
# - TENFERRO_CUTENSOR_PATH: cuTENSOR library search path(s), colon-separated;
#   defaults below to the same path baked into the CUDA devcontainer.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/tenferro_checkout.sh
source "$SCRIPT_DIR/tenferro_checkout.sh"
RESULTS_ROOT="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"

BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-nvidia-gpu}"
case "$BENCHMARK_TARGET_PROFILE" in
    nvidia-gpu)
        export BENCHMARK_TARGET_PROFILE
        ;;
    *)
        echo "ERROR: BENCHMARK_TARGET_PROFILE must be nvidia-gpu for gpu/permutation." >&2
        exit 1
        ;;
esac

SUITE_ID="gpu/permutation"
SUITE_FILE="$PROJECT_DIR/benchmarks/gpu/permutation.yaml"
DEVICE_ORDINAL="${GPU_BENCH_DEVICE:-0}"
RUST_MIN_STACK="${RUST_MIN_STACK:-67108864}"

# Correctness gate recorded in run.yaml and the generated report. Both runners
# read the same PERMUTATION_FULL_VERIFY environment variable.
PERMUTATION_VERIFICATION_MODE="sampled"
if [[ "${PERMUTATION_FULL_VERIFY:-0}" == "1" ]]; then
    PERMUTATION_VERIFICATION_MODE="full"
fi

# cuTENSOR search path. An explicit TENFERRO_CUTENSOR_PATH always wins. The
# devcontainer path baked into .devcontainer/cuda/devcontainer.json is used only
# when that file is actually present; otherwise the variable stays unset so the
# runner falls back to its own DEFAULT_CUTENSOR_PATHS, which include the bare
# sonames and therefore find a library reachable through LD_LIBRARY_PATH (a
# module install, or one unpacked under $HOME). Exporting the devcontainer path
# unconditionally replaced that fallback with a file that exists on no other
# host, so every cuTENSOR cell was recorded as skipped off the devcontainer.
CUTENSOR_DEVCONTAINER_PATH="/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2"
if [[ -z "${TENFERRO_CUTENSOR_PATH:-}" && -f "$CUTENSOR_DEVCONTAINER_PATH" ]]; then
    export TENFERRO_CUTENSOR_PATH="$CUTENSOR_DEVCONTAINER_PATH"
fi

TIMESTAMP="${BENCHMARK_TIMESTAMP:-${GPU_BENCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}}"
# Same rule as run_gpu_suite.sh: the run metadata timestamp identifies the run
# directory, so it is derived from that stamp rather than the wall clock.
RUN_TIMESTAMP_RFC3339="${GPU_BENCH_RUN_TIMESTAMP:-$(python3 - "$TIMESTAMP" <<'PY'
from datetime import datetime, timezone
import sys
print(datetime.strptime(sys.argv[1], "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).isoformat())
PY
)}"

RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/gpu/permutation/$TIMESTAMP"
RUN_YAML="$RUN_DIR/run.yaml"
LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/gpu/permutation.md"

mkdir -p "$RUN_DIR" "$(dirname "$LATEST_REPORT")"

PYTHON=(python3)
VALIDATOR=(python3 "$SCRIPT_DIR/validate_benchmark_suite.py")
METADATA=(python3 "$SCRIPT_DIR/collect_run_metadata.py")
FORMATTER=(python3 "$SCRIPT_DIR/format_gpu_permutation_results.py")
if command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run python)
    VALIDATOR=(uv run python "$SCRIPT_DIR/validate_benchmark_suite.py")
    METADATA=(uv run python "$SCRIPT_DIR/collect_run_metadata.py")
    FORMATTER=(uv run python "$SCRIPT_DIR/format_gpu_permutation_results.py")
fi

"${VALIDATOR[@]}" "$SUITE_FILE"

tenferro_dir="$PROJECT_DIR/extern/tenferro-rs"
tenferro_commit="$(tenferro_checkout_commit "$tenferro_dir")"

metadata_args=(
    --suite-id "$SUITE_ID"
    --target-profile "$BENCHMARK_TARGET_PROFILE"
    --suite-file "${SUITE_FILE#$PROJECT_DIR/}"
    --timestamp "$RUN_TIMESTAMP_RFC3339"
    --tenferro-dir "$tenferro_dir"
    --features cuda
    --blas none
    --cuda-device-ordinal "$DEVICE_ORDINAL"
    --output "$RUN_YAML"
)
[[ -n "$tenferro_commit" ]] && metadata_args+=(--tenferro-commit "$tenferro_commit")
"${METADATA[@]}" "${metadata_args[@]}"

# Record the correctness-gate mode with the run metadata so sampled and full
# runs are distinguishable from the raw run alone.
"${PYTHON[@]}" - "$RUN_YAML" "$PERMUTATION_VERIFICATION_MODE" <<'PY'
import sys
import yaml
from pathlib import Path
path = Path(sys.argv[1])
doc = yaml.safe_load(path.read_text())
doc.setdefault("collection", {})["permutation_verification_mode"] = sys.argv[2]
path.write_text(yaml.safe_dump(doc, sort_keys=False))
PY

echo "============================================"
echo " GPU permutation benchmark suite"
echo "============================================"
echo "Suite:        $SUITE_ID"
echo "Suite file:   ${SUITE_FILE#$PROJECT_DIR/}"
echo "Target:       $BENCHMARK_TARGET_PROFILE"
echo "Device:       cuda:$DEVICE_ORDINAL"
echo "Timestamp:    $TIMESTAMP"
echo "Run dir:      ${RUN_DIR#$PROJECT_DIR/}"
echo ""

PERMUTATION_BIN="${CARGO_TARGET_DIR:-$PROJECT_DIR/target}/release/benchmark_gpu_permutation"
RUST_JSONL="$RUN_DIR/rust_output.jsonl"
GPU_BENCH_BACKEND_SLEEP="${GPU_BENCH_BACKEND_SLEEP:-2}"

# Every failure below is recorded and the collection continues: one dead arm (or
# a build that never produced the binary) must never cost the run the rows the
# other arms did measure. The script reports red at the very end, after the page,
# the summary and the records are written. The arm bodies return explicitly: the
# caller runs them under `||`, where `set -e` is off.
run_rust_arm() {
    echo "Building benchmark_gpu_permutation..."
    if [[ "${GPU_BENCH_PREBUILT_RUST:-0}" == 1 ]]; then
        # The hosted build job ships this binary; the benchmark pod has no Rust toolchain.
        python3 "$SCRIPT_DIR/ci/gpu_artifact.py" "$PERMUTATION_BIN" || return $?
    else
        CUBECL_DEBUG_LOG=0 \
        CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
        RUST_MIN_STACK="$RUST_MIN_STACK" \
        LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
            cargo build --release --features cuda --bin benchmark_gpu_permutation || return $?
    fi
    echo "Running Rust GPU permutation benchmarks (tenferro-cuda-transpose, tenferro-cuda-to-contiguous, tenferro-cuda-destination-reuse, cutensor)..."
    CUBECL_DEBUG_LOG=0 \
    CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
    RUST_MIN_STACK="$RUST_MIN_STACK" \
    LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    GPU_BENCH_DEVICE="$DEVICE_ORDINAL" \
    BENCH_OUTPUT="$RUST_JSONL" \
        "$PERMUTATION_BIN"
}

COLLECTION_FAILED=0
rust_rc=0
run_rust_arm || rust_rc=$?
if [[ "$rust_rc" != 0 ]]; then
    echo "ARM FAILED: $SUITE_ID rust collector exited $rust_rc; keeping the rows it wrote and continuing" >&2
    COLLECTION_FAILED=1
fi

INPUTS=()
[[ -s "$RUST_JSONL" ]] && INPUTS+=("$RUST_JSONL")

# One process per Python backend, strictly sequential, mirroring
# scripts/run_gpu_suite.sh: separate processes so framework CUDA allocators
# release device memory before the next backend starts.
for backend in pytorch-cuda memcpy-d2d; do
    echo "Running Python GPU permutation benchmarks ($backend)..."
    BACKEND_JSONL="$RUN_DIR/python_output_${backend}.jsonl"
    arm_rc=0
    GPU_BENCH_DEVICE="$DEVICE_ORDINAL" \
    BENCH_OUTPUT="$BACKEND_JSONL" \
        "${PYTHON[@]}" "$SCRIPT_DIR/benchmark_gpu_permutation_python.py" "$backend" || arm_rc=$?
    if [[ "$arm_rc" != 0 ]]; then
        echo "ARM FAILED: $SUITE_ID $backend exited $arm_rc; keeping the rows it wrote and continuing" >&2
        COLLECTION_FAILED=1
    fi
    if [[ -s "$BACKEND_JSONL" ]]; then
        INPUTS+=("$BACKEND_JSONL")
    fi
    sleep "$GPU_BENCH_BACKEND_SLEEP"
done

if [[ ${#INPUTS[@]} -eq 0 ]]; then
    echo "No arm produced a row for $SUITE_ID; nothing to format." >&2
    COLLECTION_FAILED=1
fi

echo ""
# The routine path reads one record contract: fold this suite's rows and build
# its summary, then render the permutation page into the run directory.
echo "Formatting gpu/permutation results as markdown..."
formatter_rc=0
if [[ ${#INPUTS[@]} -gt 0 ]]; then
    "${FORMATTER[@]}" \
        "${INPUTS[@]}" \
        --run-metadata "$RUN_YAML" \
        --output "$RUN_DIR/report.md" || formatter_rc=$?
fi
if [[ "$formatter_rc" != 0 ]]; then
    echo "FORMAT FAILED: $SUITE_ID formatter exited $formatter_rc; its rows stay in the raw run" >&2
    COLLECTION_FAILED=1
fi
# One record contract for the routine accounting, the bundle row gate and the
# dashboard summary. It is written after the page so the shared summary replaces
# this suite's own sidecar rather than the other way round. It runs even when the
# page could not be rendered: the rows it measured stay countable either way.
normalize_rc=0
if [[ ${#INPUTS[@]} -gt 0 ]]; then
    "${PYTHON[@]}" "$SCRIPT_DIR/normalize_gpu_records.py" --output "$RUN_DIR/records.jsonl" \
        --summary "$RUN_DIR/report.summary.json" --run-metadata "$RUN_YAML" "${INPUTS[@]}" || normalize_rc=$?
fi
if [[ "$normalize_rc" != 0 ]]; then
    echo "NORMALIZE FAILED: $SUITE_ID normalize exited $normalize_rc; its rows stay in the raw run" >&2
    COLLECTION_FAILED=1
fi
if [[ "${GPU_BENCH_SKIP_LATEST:-0}" != 1 ]]; then
    if [[ ! -s "$RUN_DIR/report.md" || ! -s "$RUN_DIR/report.summary.json" ]]; then
        echo "PUBLISH SKIPPED: $SUITE_ID has no page or summary to publish" >&2
    else
        cp "$RUN_DIR/report.md" "$LATEST_REPORT"
        # Record the correctness-gate mode in the maintained report text, before the
        # receipt is written: the receipt holds the digest of the page as published.
        "${PYTHON[@]}" - "$LATEST_REPORT" "$PERMUTATION_VERIFICATION_MODE" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text().rstrip()
text += (
    "\n\nCorrectness gate: exact **%s** verification. `sampled` checks a fixed set of "
    "per-axis corners plus roughly 1M seeded output elements against the host source index mapping; "
    "`full` downloads the whole output and compares it elementwise against a full host "
    "transpose. Set `PERMUTATION_FULL_VERIFY=1` to request the full gate.\n"
) % sys.argv[2]
path.write_text(text)
PY
        cp "$RUN_DIR/report.summary.json" "${LATEST_REPORT%.md}.summary.json"
        "${PYTHON[@]}" "$SCRIPT_DIR/ci/result_ingestion.py" "$LATEST_REPORT"
    fi
fi

echo ""
if [[ "$COLLECTION_FAILED" != 0 ]]; then
    # Everything measured is written and packaged by the caller's always-step;
    # this exit code only makes the collection red so someone looks at it.
    echo "GPU permutation collection finished with failures; the rows it measured and the" >&2
    echo "reasons it recorded are in ${RUN_DIR#$PROJECT_DIR/}." >&2
    exit 1
fi
echo "GPU permutation benchmark suite complete: $SUITE_ID"
echo "Run YAML:  ${RUN_YAML#$PROJECT_DIR/}"
for path in "${INPUTS[@]}"; do
    echo "JSONL:     ${path#$PROJECT_DIR/}"
done
echo "Report:    ${LATEST_REPORT#$PROJECT_DIR/}"
