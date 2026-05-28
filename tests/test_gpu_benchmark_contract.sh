#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

JSONL="data/results/gpu_contract_19990101_000000.jsonl"
MARKDOWN="data/results/gpu_results_19990101_000000.md"
REPORT="result/gpu-benchmark-results.md"
TMP="$(mktemp -d)"
HAD_REPORT=0

if [[ -e "$REPORT" ]]; then
  HAD_REPORT=1
  cp "$REPORT" "$TMP/gpu-benchmark-results.md"
fi

cleanup() {
  rm -f "$JSONL" "$MARKDOWN"
  if [[ "$HAD_REPORT" -eq 1 ]]; then
    cp "$TMP/gpu-benchmark-results.md" "$REPORT"
  else
    rm -f "$REPORT"
  fi
  rm -rf "$TMP"
}
trap cleanup EXIT

uv run python scripts/validate_benchmark_suite.py \
  benchmarks/gpu/dense.yaml \
  benchmarks/gpu/einsum.yaml \
  benchmarks/gpu/sparse.yaml

GPU_BENCH_TIMESTAMP=19990101_000000 \
GPU_BENCH_BACKENDS=tenferro-cuda-trace,pytorch-cuda \
GPU_BENCH_PROBLEM=dense_matmul_f64_256 \
  bash scripts/run_gpu_suite.sh

test -s "$JSONL"
test -s "$MARKDOWN"
test -s "$REPORT"

uv run python scripts/validate_benchmark_suite.py --kind result "$JSONL"

rg -n "GPU Benchmark Results" "$REPORT"
rg -n "dense_matmul_f64_256" "$REPORT"
rg -n "tenferro-rs CUDA trace" "$REPORT"
rg -n "PyTorch CUDA" "$REPORT"
rg -n "not configured" "$REPORT"
