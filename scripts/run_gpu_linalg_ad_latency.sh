#!/usr/bin/env bash
set -euo pipefail

# GPU linalg AD single-call latency / per-op overhead suite (n=2, 4, 8).
#
# These rows are latency diagnostics: device kernel time is a small fraction of
# each measurement, so they must not be read as GPU throughput. GPU-sized AD
# results live in the `gpu/linalg_jvp_vjp` suite.
#
# Usage: BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_linalg_ad_latency.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export LINALG_AD_SUITE_FILE="$PROJECT_DIR/benchmarks/gpu/linalg_ad_latency.yaml"
exec "$SCRIPT_DIR/run_gpu_linalg_jvp_vjp.sh" "$@"
