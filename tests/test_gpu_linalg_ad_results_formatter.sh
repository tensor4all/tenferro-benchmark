#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if command -v uv >/dev/null 2>&1; then
    PY=(uv run python)
else
    PY=(python3)
fi

cat > "$TMP/records.jsonl" <<'JSONL'
{"schema_version":1,"suite_id":"gpu/linalg_jvp_vjp","problem_id":"linalg_ad_small_grad_sum_qr_jvp_f64_2","op":"jvp","backend":"tenferro-cuda-trace","status":"ok","timing":{"warmup_runs":3,"timed_runs":7,"compile_time_ms":null,"first_run_ms":0.12,"median_ms":0.12,"min_ms":0.12,"p95_ms":0.12,"iqr_ms":0.01,"timing_scope":"steady_state_host_api_plus_device_sync"},"performance":{"tflops":null,"effective_bandwidth_gbps":null,"peak_memory_bytes":null},"verification":{"status":"skipped","reference_backend":null,"max_abs_error":null,"max_rel_error":null,"residual":null,"rtol":1e-5,"atol":1e-8,"reason":null},"execution":{"device":"cuda","device_ordinal":0,"execution_path":"test","synchronization":"sync","layout":null,"dtype":null,"notes":"shape=2x2","unsupported_reason":null}}
{"schema_version":1,"suite_id":"gpu/linalg_jvp_vjp","problem_id":"linalg_ad_small_grad_sum_qr_jvp_f64_2","op":"jvp","backend":"pytorch-cuda","status":"ok","timing":{"warmup_runs":3,"timed_runs":7,"compile_time_ms":null,"first_run_ms":0.20,"median_ms":0.20,"min_ms":0.20,"p95_ms":0.20,"iqr_ms":0.02,"timing_scope":"steady_state_host_api_plus_device_sync"},"performance":{"tflops":null,"effective_bandwidth_gbps":null,"peak_memory_bytes":null},"verification":{"status":"skipped","reference_backend":null,"max_abs_error":null,"max_rel_error":null,"residual":null,"rtol":1e-5,"atol":1e-8,"reason":null},"execution":{"device":"cuda","device_ordinal":0,"execution_path":"test","synchronization":"sync","layout":null,"dtype":null,"notes":"shape=2x2","unsupported_reason":null}}
{"schema_version":1,"suite_id":"gpu/linalg_jvp_vjp","problem_id":"linalg_ad_small_grad_sum_solve_vjp_f64_8","op":"vjp","backend":"jax-cuda","status":"ok","timing":{"warmup_runs":3,"timed_runs":7,"compile_time_ms":null,"first_run_ms":1.5,"median_ms":1.5,"min_ms":1.5,"p95_ms":1.5,"iqr_ms":0.1,"timing_scope":"steady_state_host_api_plus_device_sync"},"performance":{"tflops":null,"effective_bandwidth_gbps":null,"peak_memory_bytes":null},"verification":{"status":"skipped","reference_backend":null,"max_abs_error":null,"max_rel_error":null,"residual":null,"rtol":1e-5,"atol":1e-8,"reason":null},"execution":{"device":"cuda","device_ordinal":0,"execution_path":"test","synchronization":"sync","layout":null,"dtype":null,"notes":"shape=8x8,rhs=1","unsupported_reason":null}}
JSONL

OUT="$TMP/report.md"
"${PY[@]}" "$ROOT/scripts/format_gpu_linalg_ad_results.py" "$TMP/records.jsonl" > "$OUT"

grep -q '## Linalg JVP/VJP Benchmark Items' "$OUT"
grep -q 'grad_sum_qr_jvp' "$OUT"
grep -q 'tenferro-rs CUDA trace' "$OUT"
grep -q 'PyTorch CUDA' "$OUT"
grep -q 'JAX CUDA' "$OUT"
grep -q 'loss = sum(solve(A, b))' "$OUT"
! grep -q 'grad_sum_matmul' "$OUT"

tail -c1 "$OUT" | od -An -tx1 | grep -q '0a'

echo "format_gpu_linalg_ad_results.sh: ok"
