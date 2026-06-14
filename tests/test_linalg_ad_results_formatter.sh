#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PY=()
if command -v uv >/dev/null 2>&1; then
  PY=(uv run python)
else
  PY=(python3)
fi

cat > "$TMP/cpu_ops.csv" <<'CSV'
suite,benchmark,dtype,threads,shape,backend,median_ms,iqr_ms,status
small,grad_sum_svd_s_jvp,f64,4,2x2,tenferro-trace,0.020,0.005,ok
small,grad_sum_svd_s_jvp,f64,4,2x2,pytorch-cpu,0.066,0.005,ok
small,grad_sum_svd_s_vjp,f64,4,2x2,jax-cpu,0.438,0.193,ok
small,matmul,f64,4,2x2,tenferro-trace,0.010,0.001,ok
large,grad_sum_qr_jvp,f64,4,64x64,tenferro-trace,0.214,0.016,ok
CSV

"${PY[@]}" "$ROOT/scripts/format_linalg_ad_results.py" "$TMP/cpu_ops.csv" > "$TMP/linalg_jvp_vjp.md"

grep -q "## Linalg JVP/VJP Benchmark Items" "$TMP/linalg_jvp_vjp.md"
grep -q 'grad_sum_svd_s_jvp' "$TMP/linalg_jvp_vjp.md"
grep -q 'grad_sum_qr_jvp' "$TMP/linalg_jvp_vjp.md"
grep -q 'tenferro-rs trace mode' "$TMP/linalg_jvp_vjp.md"
grep -q 'PyTorch Python' "$TMP/linalg_jvp_vjp.md"
grep -q 'JAX Python' "$TMP/linalg_jvp_vjp.md"
grep -q 'loss = sum(singular values)' "$TMP/linalg_jvp_vjp.md"
! grep -q '`matmul`' "$TMP/linalg_jvp_vjp.md"

tail -c1 "$TMP/linalg_jvp_vjp.md" | od -An -tx1 | grep -q '0a'

echo "linalg ad results formatter test passed"
