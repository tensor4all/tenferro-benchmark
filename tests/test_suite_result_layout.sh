#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

assert_valid() {
  local kind="$1"
  local path="$2"
  local label="$3"

  if ! uv run python scripts/validate_benchmark_suite.py --kind "$kind" "$path" >"$TMP/out" 2>&1; then
    echo "$label did not pass validation" >&2
    cat "$TMP/out" >&2
    exit 1
  fi
}

assert_invalid() {
  local kind="$1"
  local path="$2"
  local label="$3"

  if uv run python scripts/validate_benchmark_suite.py --kind "$kind" "$path" >"$TMP/out" 2>&1; then
    echo "$label unexpectedly passed validation" >&2
    exit 1
  fi
}

assert_suite_id() {
  local path="$1"
  local expected="$2"
  local actual

  if ! actual="$(uv run python - "$path" 2>"$TMP/out" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
with path.open() as fh:
    suite = yaml.safe_load(fh)
if not isinstance(suite, dict) or not isinstance(suite.get("suite_id"), str):
    print(f"{path}: missing top-level suite_id", file=sys.stderr)
    raise SystemExit(1)
print(suite["suite_id"])
PY
)"; then
    echo "$path did not expose a scalar suite_id" >&2
    cat "$TMP/out" >&2
    exit 1
  fi
  if [[ "$actual" != "$expected" ]]; then
    echo "$path suite_id was '$actual', expected '$expected'" >&2
    exit 1
  fi
}

assert_top_level_backends() {
  local path="$1"

  if ! uv run python - "$path" >"$TMP/out" 2>&1 <<'PY'; then
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
with path.open() as fh:
    suite = yaml.safe_load(fh)
backends = suite.get("backends") if isinstance(suite, dict) else None
if not isinstance(backends, list) or not backends:
    print(f"{path}: missing non-empty top-level backends block", file=sys.stderr)
    raise SystemExit(1)
PY
    cat "$TMP/out" >&2
    exit 1
  fi
}

assert_valid suite benchmarks/cpu/einsum.yaml "bundled CPU einsum suite"
assert_valid suite benchmarks/gpu/dense.yaml "bundled GPU dense suite"
assert_valid suite benchmarks/gpu/einsum.yaml "bundled GPU einsum suite"
assert_valid suite benchmarks/gpu/sparse.yaml "bundled GPU sparse suite"

assert_suite_id benchmarks/cpu/einsum.yaml cpu/einsum
assert_suite_id benchmarks/gpu/dense.yaml gpu/dense
assert_suite_id benchmarks/gpu/einsum.yaml gpu/einsum
assert_suite_id benchmarks/gpu/sparse.yaml gpu/sparse

assert_top_level_backends benchmarks/gpu/dense.yaml
assert_top_level_backends benchmarks/gpu/einsum.yaml
assert_top_level_backends benchmarks/gpu/sparse.yaml

cat > "$TMP/cpu_selector_suite.yaml" <<'YAML'
schema_version: 1
suite_id: cpu/einsum
title: CPU einsum selector suite
description: CPU selector suite using shared einsum problem definitions.
defaults:
  run: {warmups: 1, runs: 3, timing_scope: steady_state_host_api}
  verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
backends: [tenferro-cpu-eager, libtorch-cpu, pytorch-cpu]
problems:
  source: benchmarks/cpu/einsum.yaml
  include: [einsum_bin_matmul_256_f64]
  exclude: [einsum_skip_me]
YAML
assert_valid suite "$TMP/cpu_selector_suite.yaml" "valid CPU selector suite"

cat > "$TMP/legacy_suite_id.yaml" <<'YAML'
schema_version: 1
suite_id: gpu_einsum_contract_v1
title: Legacy GPU einsum suite
description: Legacy underscore suite IDs should be rejected.
defaults:
  run: {warmups: 1, runs: 1, timing_scope: steady_state_host_api_plus_device_sync}
  verify: {reference: cpu_fp64, rtol: 1.0e-8, atol: 1.0e-10}
backends: [pytorch-cuda]
problems:
  source: benchmarks/gpu/einsum.yaml
  include: [einsum_bin_matmul_3072_f64]
YAML
assert_invalid suite "$TMP/legacy_suite_id.yaml" "legacy underscore suite ID"

cat > "$TMP/run.yaml" <<'YAML'
schema_version: 1
suite_id: cpu/einsum
suite_file: benchmarks/cpu/einsum.yaml
timestamp: "2026-06-03T12:34:56+09:00"
tenferro_rs:
  path: extern/tenferro-rs
  commit: abcdef1
  features: [openblas]
environment:
  hostname: ci-host
  os: macos
  arch: arm64
  env: {OMP_NUM_THREADS: "8", RAYON_NUM_THREADS: "8"}
blas:
  implementation: openblas
  version: 0.3.26
  root: /opt/OpenBLAS
  library: /opt/OpenBLAS/lib/libopenblas.dylib
YAML
assert_valid run "$TMP/run.yaml" "valid run metadata"

cat > "$TMP/run_with_benchmark_repo_commit.yaml" <<'YAML'
schema_version: 1
suite_id: cpu/einsum
suite_file: benchmarks/cpu/einsum.yaml
timestamp: "2026-06-03T12:34:56+09:00"
benchmark_repo_commit: 0123456789abcdef
tenferro_rs:
  path: extern/tenferro-rs
  commit: abcdef1
  features: [openblas]
environment:
  hostname: ci-host
  os: macos
blas:
  implementation: openblas
  version: 0.3.26
  root: /opt/OpenBLAS
  library: /opt/OpenBLAS/lib/libopenblas.dylib
YAML
assert_invalid run "$TMP/run_with_benchmark_repo_commit.yaml" "run metadata with benchmark_repo_commit"

cat > "$TMP/run_with_environment_benchmark_repo_commit.yaml" <<'YAML'
schema_version: 1
suite_id: cpu/einsum
suite_file: benchmarks/cpu/einsum.yaml
timestamp: "2026-06-03T12:34:56+09:00"
tenferro_rs:
  path: extern/tenferro-rs
  commit: abcdef1
  features: [openblas]
environment:
  hostname: ci-host
  os: macos
  benchmark_repo_commit: 0123456789abcdef
blas:
  implementation: openblas
  version: 0.3.26
  root: /opt/OpenBLAS
  library: /opt/OpenBLAS/lib/libopenblas.dylib
YAML
assert_invalid run "$TMP/run_with_environment_benchmark_repo_commit.yaml" "run metadata with environment.benchmark_repo_commit"

cat > "$TMP/cpu_result.jsonl" <<'JSON'
{"schema_version":1,"suite_id":"cpu/einsum","problem_id":"einsum_bin_matmul_256_f64","op":"einsum","backend":"tenferro-cpu-eager","status":"ok","timing":{"warmup_runs":1,"timed_runs":3,"compile_time_ms":0.0,"first_run_ms":1.2,"median_ms":1.0,"min_ms":0.9,"p95_ms":1.1,"iqr_ms":0.1,"timing_scope":"steady_state_host_api"},"performance":{"tflops":null,"effective_bandwidth_gbps":null,"peak_memory_bytes":null},"verification":{"status":"passed","reference_backend":"cpu_fp64","max_abs_error":0.0,"max_rel_error":0.0,"residual":null,"rtol":1.0e-8,"atol":1.0e-10,"reason":null},"execution":{"device":"cpu","device_ordinal":0,"execution_path":"tenferro-rs eager cpu","synchronization":"none","layout":"col_major","dtype":"f64","notes":null,"unsupported_reason":null}}
JSON
assert_valid result "$TMP/cpu_result.jsonl" "valid CPU result record"

cat > "$TMP/result_with_legacy_suite_id.jsonl" <<'JSON'
{"schema_version":1,"suite_id":"gpu_bad","problem_id":"einsum_bin_matmul_256_f64","op":"einsum","backend":"tenferro-cpu-eager","status":"ok","timing":{"warmup_runs":1,"timed_runs":3,"compile_time_ms":0.0,"first_run_ms":1.2,"median_ms":1.0,"min_ms":0.9,"p95_ms":1.1,"iqr_ms":0.1,"timing_scope":"steady_state_host_api"},"performance":{"tflops":null,"effective_bandwidth_gbps":null,"peak_memory_bytes":null},"verification":{"status":"passed","reference_backend":"cpu_fp64","max_abs_error":0.0,"max_rel_error":0.0,"residual":null,"rtol":1.0e-8,"atol":1.0e-10,"reason":null},"execution":{"device":"cpu","device_ordinal":0,"execution_path":"tenferro-rs eager cpu","synchronization":"none","layout":"col_major","dtype":"f64","notes":null,"unsupported_reason":null}}
JSON
assert_invalid result "$TMP/result_with_legacy_suite_id.jsonl" "result record with legacy suite ID"

cat > "$TMP/cpu_result_with_environment.jsonl" <<'JSON'
{"schema_version":1,"suite_id":"cpu/einsum","problem_id":"einsum_bin_matmul_256_f64","op":"einsum","backend":"tenferro-cpu-eager","status":"ok","timing":{"warmup_runs":1,"timed_runs":3,"compile_time_ms":0.0,"first_run_ms":1.2,"median_ms":1.0,"min_ms":0.9,"p95_ms":1.1,"iqr_ms":0.1,"timing_scope":"steady_state_host_api"},"performance":{"tflops":null,"effective_bandwidth_gbps":null,"peak_memory_bytes":null},"verification":{"status":"passed","reference_backend":"cpu_fp64","max_abs_error":0.0,"max_rel_error":0.0,"residual":null,"rtol":1.0e-8,"atol":1.0e-10,"reason":null},"environment":{"timestamp_utc":"2026-06-03T03:34:56+00:00"},"execution":{"device":"cpu","device_ordinal":0,"execution_path":"tenferro-rs eager cpu","synchronization":"none","layout":"col_major","dtype":"f64","notes":null,"unsupported_reason":null}}
JSON
assert_invalid result "$TMP/cpu_result_with_environment.jsonl" "CPU result record with top-level environment"
