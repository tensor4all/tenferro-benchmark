#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

JSONL="data/results/gpu_contract_19990101_000000.jsonl"
MARKDOWN="data/results/gpu_results_19990101_000000.md"
REPORT="result/gpu-benchmark-results.md"
TMP="$(mktemp -d)"
ARTIFACTS=("$JSONL" "$MARKDOWN" "$REPORT")

mkdir -p "$TMP/originals"
for artifact in "${ARTIFACTS[@]}"; do
  if [[ -e "$artifact" ]]; then
    mkdir -p "$TMP/originals/$(dirname "$artifact")"
    cp "$artifact" "$TMP/originals/$artifact"
  fi
done

cleanup() {
  for artifact in "${ARTIFACTS[@]}"; do
    if [[ -e "$TMP/originals/$artifact" ]]; then
      cp "$TMP/originals/$artifact" "$artifact"
    else
      rm -f "$artifact"
    fi
  done
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
