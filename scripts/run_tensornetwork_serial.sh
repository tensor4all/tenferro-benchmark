#!/usr/bin/env bash
# Run gpu/tensornetwork via run_gpu_suite.sh (serial backend execution).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export GPU_BENCH_SUITE="${GPU_BENCH_SUITE:-benchmarks/gpu/tensornetwork.yaml}"
exec "$SCRIPT_DIR/run_gpu_suite.sh"
