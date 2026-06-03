#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_DIR="data/results/gpu/dense/19990101_000000"
RUN_YAML="$RUN_DIR/run.yaml"
JSONL="$RUN_DIR/records.jsonl"
MARKDOWN="$RUN_DIR/report.md"
REPORT="result/gpu/dense.md"
TMP="$(mktemp -d)"
ARTIFACTS=("$RUN_YAML" "$JSONL" "$MARKDOWN" "$RUN_DIR/rust_records.jsonl" "$REPORT")

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
{"schema_version":1,"suite_id":"gpu/dense","problem_id":"bad_ok","op":"matmul","backend":"pytorch-cuda","status":"ok","timing":{"warmup_runs":0,"timed_runs":0,"compile_time_ms":null,"first_run_ms":null,"median_ms":null,"min_ms":null,"p95_ms":null,"iqr_ms":null,"timing_scope":"steady_state_host_api_plus_device_sync"},"verification":{"status":"skipped","reference_backend":null,"rtol":null,"atol":null},"environment":{"timestamp_utc":"not-a-date"},"execution":{"device":"cuda","device_ordinal":0,"execution_path":"contract-test","synchronization":"none","layout":"{}","dtype":"{}"}}
JSON
if uv run python scripts/validate_benchmark_suite.py --kind result "$TMP/bad_ok_result.jsonl" >/dev/null 2>&1; then
  echo "invalid ok result record unexpectedly passed validation" >&2
  exit 1
fi

cat > "$TMP/bad_verification_failed_result.jsonl" <<'JSON'
{"schema_version":1,"suite_id":"gpu/dense","problem_id":"bad_verification","op":"matmul","backend":"pytorch-cuda","status":"verification_failed","timing":{"warmup_runs":0,"timed_runs":1,"compile_time_ms":null,"first_run_ms":1.0,"median_ms":1.0,"min_ms":1.0,"p95_ms":1.0,"iqr_ms":0.0,"timing_scope":"steady_state_host_api_plus_device_sync"},"verification":{"status":"passed","reference_backend":"cpu_fp64","rtol":1.0e-8,"atol":1.0e-10},"environment":{"timestamp_utc":"1999-01-01T00:00:00+00:00"},"execution":{"device":"cuda","device_ordinal":0,"execution_path":"contract-test","synchronization":"none","layout":"{}","dtype":"{}"}}
JSON
if uv run python scripts/validate_benchmark_suite.py --kind result "$TMP/bad_verification_failed_result.jsonl" >/dev/null 2>&1; then
  echo "verification_failed result with passed verification unexpectedly passed validation" >&2
  exit 1
fi

cat > "$TMP/duplicate_problem_ids.yaml" <<'YAML'
schema_version: 1
suite_id: gpu/dense
title: Duplicate Problem ID Suite
description: duplicate problem IDs should fail semantic validation
defaults:
  run: {warmups: 0, runs: 1}
  verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
backends: [pytorch-cuda, cusparse]
problems:
  - id: duplicate
    family: dense
    op: matmul
    dtype: {a: f64, b: f64, c: f64}
    data: {generator: normal, seed: 1}
    run: {warmups: 0, runs: 1}
    verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    only_backends: [pytorch-cuda]
    matmul: {m: 1, n: 1, k: 1}
  - id: duplicate
    family: sparse
    op: spmm
    dtype: {values: f64, x: f64, y: f64}
    data: {generator: suitesparse, seed: 0}
    run: {warmups: 0, runs: 1}
    verify: {reference: cpu_sparse_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    only_backends: [cusparse]
    sparse: {source: suitesparse, storage: csr, rows: 1, cols: 1, nnz: 1, rhs_cols: 1}
YAML
if uv run python scripts/validate_benchmark_suite.py "$TMP/duplicate_problem_ids.yaml" >/dev/null 2>&1; then
  echo "duplicate problem IDs unexpectedly passed validation" >&2
  exit 1
fi

cat > "$TMP/bad_family_op.yaml" <<'YAML'
schema_version: 1
suite_id: gpu/sparse
title: Bad Family Operation Suite
description: family/op mismatch should fail validation
defaults:
  run: {warmups: 0, runs: 1}
  verify: {reference: cpu_sparse_fp64, rtol: 1.0e-8, atol: 1.0e-10}
backends: [cusparse]
problems:
  - id: dense_sparse_mismatch
    family: dense
    op: spmv
    dtype: {values: f64, x: f64, y: f64}
    data: {generator: suitesparse, seed: 0}
    run: {warmups: 0, runs: 1}
    verify: {reference: cpu_sparse_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    sparse: {source: suitesparse, storage: csr, rows: 1, cols: 1, nnz: 1}
YAML
if uv run python scripts/validate_benchmark_suite.py "$TMP/bad_family_op.yaml" >/dev/null 2>&1; then
  echo "family/op mismatch unexpectedly passed validation" >&2
  exit 1
fi

cat > "$TMP/bad_extra_op_block.yaml" <<'YAML'
schema_version: 1
suite_id: gpu/dense
title: Bad Extra Operation Block Suite
description: unrelated op block should fail validation
defaults:
  run: {warmups: 0, runs: 1}
  verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
backends: [pytorch-cuda]
problems:
  - id: matmul_with_sparse
    family: dense
    op: matmul
    dtype: {a: f64, b: f64, c: f64}
    data: {generator: normal, seed: 1}
    run: {warmups: 0, runs: 1}
    verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
    matmul: {m: 1, n: 1, k: 1}
    sparse: {source: suitesparse, storage: csr, rows: 1, cols: 1, nnz: 1}
YAML
if uv run python scripts/validate_benchmark_suite.py "$TMP/bad_extra_op_block.yaml" >/dev/null 2>&1; then
  echo "unrelated op block unexpectedly passed validation" >&2
  exit 1
fi

uv run python - <<'PY'
import importlib.util
import sys
import types
from pathlib import Path

root = Path.cwd()
spec = importlib.util.spec_from_file_location(
    "benchmark_gpu_python", root / "scripts" / "benchmark_gpu_python.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

fake_jax = types.ModuleType("jax")
fake_jnp = types.ModuleType("jax.numpy")

class FakeConfig:
    def update(self, *args, **kwargs):
        pass

def devices(backend=None):
    if backend == "cuda":
        raise RuntimeError("Unknown backend cuda. Available backends are ['cpu']")
    return []

fake_jax.config = FakeConfig()
fake_jax.devices = devices
fake_jax.numpy = fake_jnp
sys.modules["jax"] = fake_jax
sys.modules["jax.numpy"] = fake_jnp

suite = {
    "suite_id": "gpu/dense",
    "backends": ["jax-cuda"],
    "problems": [
        {
            "id": "jax_without_cuda",
            "op": "matmul",
            "only_backends": ["jax-cuda"],
            "dtype": {"a": "f64", "b": "f64", "c": "f64"},
            "layout": {},
        }
    ],
}
problem = dict(suite["problems"][0])
rec = mod._run_one(
    suite["suite_id"],
    problem,
    "jax-cuda",
    0,
    suite_backends=suite["backends"],
    ts="1999-01-01T00:00:00+00:00",
    bc="benchmark",
    tc="tenferro",
)
assert rec["status"] == "not_configured", rec
assert "Unknown backend cuda" in rec["execution"]["unsupported_reason"], rec
PY

GPU_BENCH_TIMESTAMP=19990101_000000 \
GPU_BENCH_SUITE=benchmarks/gpu/dense.yaml \
GPU_BENCH_BACKENDS=tenferro-cuda-trace,pytorch-cuda,cusolver,cutlass \
GPU_BENCH_PROBLEM=dense_matmul_f64_3072 \
  bash scripts/run_gpu_suite.sh

test -s "$RUN_YAML"
test -s "$JSONL"
test -s "$MARKDOWN"
test -s "$REPORT"

uv run python scripts/validate_benchmark_suite.py --kind run "$RUN_YAML"
uv run python scripts/validate_benchmark_suite.py --kind result "$JSONL"
if rg -q '"environment"' "$JSONL"; then
  echo "result records unexpectedly contained run-level environment" >&2
  exit 1
fi

rg -n "GPU Benchmark Results" "$REPORT"
rg -n "Suite: \`gpu/dense\`" "$REPORT"
rg -n "dense_matmul_f64_3072" "$REPORT"
rg -n "tenferro-rs CUDA trace" "$REPORT"
rg -n "PyTorch CUDA" "$REPORT"
rg -n "cuSOLVER" "$REPORT"
rg -n "unsupported" "$REPORT"
rg -n "explicit tenferro-rs synchronize API" "$REPORT"
rg -n "QR-based cuSOLVER comparison" "$REPORT"
rg -n "same deterministic benchmark generator" "$REPORT"
