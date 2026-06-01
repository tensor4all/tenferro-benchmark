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

cat > "$TMP/bad_ok_result.jsonl" <<'JSON'
{"schema_version":1,"suite_id":"gpu_bad","problem_id":"bad_ok","op":"matmul","backend":"pytorch-cuda","status":"ok","timing":{"warmup_runs":0,"timed_runs":0,"compile_time_ms":null,"first_run_ms":null,"median_ms":null,"min_ms":null,"p95_ms":null,"iqr_ms":null,"timing_scope":"steady_state_host_api_plus_device_sync"},"verification":{"status":"skipped","reference_backend":null,"rtol":null,"atol":null},"environment":{"timestamp_utc":"not-a-date"},"execution":{"device":"cuda","device_ordinal":0,"execution_path":"contract-test","synchronization":"none","layout":"{}","dtype":"{}"}}
JSON
if uv run python scripts/validate_benchmark_suite.py --kind result "$TMP/bad_ok_result.jsonl" >/dev/null 2>&1; then
  echo "invalid ok result record unexpectedly passed validation" >&2
  exit 1
fi

cat > "$TMP/bad_verification_failed_result.jsonl" <<'JSON'
{"schema_version":1,"suite_id":"gpu_bad","problem_id":"bad_verification","op":"matmul","backend":"pytorch-cuda","status":"verification_failed","timing":{"warmup_runs":0,"timed_runs":1,"compile_time_ms":null,"first_run_ms":1.0,"median_ms":1.0,"min_ms":1.0,"p95_ms":1.0,"iqr_ms":0.0,"timing_scope":"steady_state_host_api_plus_device_sync"},"verification":{"status":"passed","reference_backend":"cpu_fp64","rtol":1.0e-8,"atol":1.0e-10},"environment":{"timestamp_utc":"1999-01-01T00:00:00+00:00"},"execution":{"device":"cuda","device_ordinal":0,"execution_path":"contract-test","synchronization":"none","layout":"{}","dtype":"{}"}}
JSON
if uv run python scripts/validate_benchmark_suite.py --kind result "$TMP/bad_verification_failed_result.jsonl" >/dev/null 2>&1; then
  echo "verification_failed result with passed verification unexpectedly passed validation" >&2
  exit 1
fi

cat > "$TMP/duplicate_problem_ids.yaml" <<'YAML'
schema_version: 1
suite_id: duplicate_problem_ids
description: duplicate problem IDs should fail semantic validation
problems:
  - id: duplicate
    family: dense
    op: matmul
    dtype: {a: f64, b: f64, c: f64}
    data: {generator: normal, seed: 1}
    run: {warmups: 0, runs: 1}
    verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    backend_candidates: [pytorch-cuda]
    matmul: {m: 1, n: 1, k: 1}
  - id: duplicate
    family: sparse
    op: spmm
    dtype: {values: f64, x: f64, y: f64}
    data: {generator: suitesparse, seed: 0}
    run: {warmups: 0, runs: 1}
    verify: {reference: cpu_sparse_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    backend_candidates: [cusparse]
    sparse: {source: suitesparse, storage: csr, rows: 1, cols: 1, nnz: 1, rhs_cols: 1}
YAML
if uv run python scripts/validate_benchmark_suite.py "$TMP/duplicate_problem_ids.yaml" >/dev/null 2>&1; then
  echo "duplicate problem IDs unexpectedly passed validation" >&2
  exit 1
fi

cat > "$TMP/bad_family_op.yaml" <<'YAML'
schema_version: 1
suite_id: bad_family_op
description: family/op mismatch should fail validation
problems:
  - id: dense_sparse_mismatch
    family: dense
    op: spmv
    dtype: {values: f64, x: f64, y: f64}
    data: {generator: suitesparse, seed: 0}
    run: {warmups: 0, runs: 1}
    verify: {reference: cpu_sparse_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    backend_candidates: [cusparse]
    sparse: {source: suitesparse, storage: csr, rows: 1, cols: 1, nnz: 1}
YAML
if uv run python scripts/validate_benchmark_suite.py "$TMP/bad_family_op.yaml" >/dev/null 2>&1; then
  echo "family/op mismatch unexpectedly passed validation" >&2
  exit 1
fi

cat > "$TMP/bad_extra_op_block.yaml" <<'YAML'
schema_version: 1
suite_id: bad_extra_op_block
description: unrelated op block should fail validation
problems:
  - id: matmul_with_sparse
    family: dense
    op: matmul
    dtype: {a: f64, b: f64, c: f64}
    data: {generator: normal, seed: 1}
    run: {warmups: 0, runs: 1}
    verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    backend_candidates: [pytorch-cuda]
    matmul: {m: 1, n: 1, k: 1}
    sparse: {source: suitesparse, storage: csr, rows: 1, cols: 1, nnz: 1}
YAML
if uv run python scripts/validate_benchmark_suite.py "$TMP/bad_extra_op_block.yaml" >/dev/null 2>&1; then
  echo "unrelated op block unexpectedly passed validation" >&2
  exit 1
fi

GPU_BENCH_TIMESTAMP=19990101_000000 \
GPU_BENCH_BACKENDS=tenferro-cuda-trace,pytorch-cuda,cusolver,cutlass \
GPU_BENCH_PROBLEM=dense_matmul_f64_3072 \
  bash scripts/run_gpu_suite.sh

test -s "$JSONL"
test -s "$MARKDOWN"
test -s "$REPORT"

uv run python scripts/validate_benchmark_suite.py --kind result "$JSONL"

rg -n "GPU Benchmark Results" "$REPORT"
rg -n "dense_matmul_f64_3072" "$REPORT"
rg -n "tenferro-rs CUDA trace" "$REPORT"
rg -n "PyTorch CUDA" "$REPORT"
rg -n "cuSOLVER" "$REPORT"
rg -n "unsupported" "$REPORT"
rg -n "stream synchronization without downloading result tensors" "$REPORT"
