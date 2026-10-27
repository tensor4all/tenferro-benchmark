#!/usr/bin/env bash
# RunPod collection for #247. A single device, correctness first, then timing.
set -euo pipefail
cd "$(dirname "$0")/../.."
export GPU_BENCH_DEVICE=0 BENCHMARK_TARGET_PROFILE=nvidia-gpu
export BENCH_COVERAGE=full BENCH_EFFORT=standard GPU_BENCH_CLOCK_RAMP_MS=300
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
device_name="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)"
[[ "$device_name" == NVIDIA\ A100* ]] || { echo "A100 required, got $device_name" >&2; exit 1; }
export BENCH_INSTANCE="$(python3 scripts/ci/gpu_readback.py)"
timestamp="$(date -u +%Y%m%d_%H%M%S)"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "readback_timestamp=$timestamp" >> "$GITHUB_OUTPUT"
fi
root=data/results/nvidia-gpu/gpu/perf_issues
PERF_ISSUES_CORRECTNESS_ONLY=1 GPU_BENCH_SKIP_LATEST=1 BENCHMARK_TIMESTAMP="${timestamp}_verify" \
    ./scripts/run_gpu_perf_issues.sh
python3 scripts/ci/gpu_readback.py --correctness-only --verify "$root/${timestamp}_verify/samples_t1.jsonl"
PERF_ISSUES_CORRECTNESS_ONLY=0 GPU_BENCH_SKIP_LATEST=0 BENCHMARK_TIMESTAMP="$timestamp" \
    ./scripts/run_gpu_perf_issues.sh
python3 scripts/ci/gpu_readback.py --verify "$root/$timestamp/samples_t1.jsonl"
