#!/usr/bin/env bash
# Public CUDA latency controls on the timing card; untimed counters/lifetime
# diagnostics on the diagnostic card. Never overlap their collection.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3
STAMP="${GPU_BENCH_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
EFFORT="${BENCH_EFFORT:-standard}"
VERIFY_ONLY="${GPU_BENCH_VERIFY_ONLY:-0}"
TIMING_DEVICE="${GPU_BENCH_DEVICE:-0}"
STATE_DEVICE="${GPU_BENCH_DIAGNOSTIC_DEVICE:-}"
if [[ -z "$STATE_DEVICE" ]]; then
    if nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader >/dev/null 2>&1; then
        STATE_DEVICE=1
    else
        STATE_DEVICE="$TIMING_DEVICE"
    fi
fi
ROOT_KIND=results
[[ "$VERIFY_ONLY" != 1 ]] || ROOT_KIND=verification
RUN_DIR="$PROJECT_DIR/data/$ROOT_KIND/nvidia-gpu/gpu/routine_controls/$STAMP"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}" MATMUL_NUM_THREADS="${MATMUL_NUM_THREADS:-1}"
RECORDS=()
for lane in timing state; do
    case_ids="$("$PYTHON" scripts/gpu_routine_controls.py \
        --effort "$EFFORT" --output "$RUN_DIR" --lane "$lane")"
    device="$TIMING_DEVICE"
    correctness=0
    if [[ "$lane" == state || "$VERIFY_ONLY" == 1 ]]; then
        device="$STATE_DEVICE"
        correctness=1
    fi
    child_stamp="${STAMP}_${lane}"
    GPU_BENCH_DEVICE="$device" GPU_BENCH_SKIP_LATEST=1 BENCH_COVERAGE=full \
        BENCH_EFFORT="$EFFORT" BENCH_INSTANCE="$case_ids" BENCHMARK_TIMESTAMP="$child_stamp" \
        PERF_ISSUES_CORRECTNESS_ONLY="$correctness" \
        bash scripts/run_gpu_perf_issues.sh
    RECORDS+=("$PROJECT_DIR/data/results/nvidia-gpu/gpu/perf_issues/$child_stamp/samples_t1.jsonl")
done
EXTRA=()
[[ "$VERIFY_ONLY" != 1 ]] || EXTRA+=(--verify-only)
"$PYTHON" scripts/gpu_routine_controls.py --output "$RUN_DIR" --records "${RECORDS[@]}" "${EXTRA[@]}"
if [[ "$VERIFY_ONLY" != 1 && "${GPU_BENCH_SKIP_LATEST:-0}" != 1 ]]; then
    mkdir -p "$PROJECT_DIR/result/nvidia-gpu/gpu"
    cp "$RUN_DIR/report.md" "$PROJECT_DIR/result/nvidia-gpu/gpu/routine_controls.md"
fi
echo "Routine controls: $RUN_DIR/report.md"
