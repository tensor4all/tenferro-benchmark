#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export BENCHMARK_RESULTS_DIR="$TMP/results"
export BENCHMARK_TIMESTAMP="linalg_ad_test"
export SKIP_EXTERN_SETUP=1
export PUBLICATION_GATE_PROFILE=quick
export PUBLICATION_GATE_SUITE=small
export PUBLICATION_GATE_TENFERRO_MODE=trace
export BENCH_RUNS=2
export BENCH_WARMUPS=1

bash "$ROOT/scripts/run_cpu_ops.sh" 1

CSV="$BENCHMARK_RESULTS_DIR/cpu_ops_t1_${BENCHMARK_TIMESTAMP}.csv"
test -s "$CSV"

for benchmark in \
    grad_sum_svd_s_jvp \
    grad_sum_svd_s_vjp \
    grad_sum_qr_jvp \
    grad_sum_qr_vjp \
    grad_sum_eigh_jvp \
    grad_sum_eigh_vjp \
    grad_sum_lu_jvp \
    grad_sum_lu_vjp \
    grad_sum_solve_jvp \
    grad_sum_solve_vjp; do
    grep -q ",${benchmark}," "$CSV"
done

grep -q ',tenferro-trace,.*,ok$' "$CSV"
grep -q ',pytorch-cpu,.*,ok$' "$CSV"
grep -q ',jax-cpu,.*,ok$' "$CSV"

MD="$TMP/cpu_ops.md"
python3 "$ROOT/scripts/format_cpu_ops_results.py" "$CSV" > "$MD"
grep -q 'grad_sum_svd_s_jvp' "$MD"

LINALG_MD="$TMP/linalg_jvp_vjp.md"
python3 "$ROOT/scripts/format_linalg_ad_results.py" "$CSV" > "$LINALG_MD"
grep -q 'grad_sum_svd_s_jvp' "$LINALG_MD"
grep -q 'grad_sum_solve_vjp' "$LINALG_MD"
grep -q 'loss = sum(singular values)' "$LINALG_MD"
! grep -q 'grad_sum_matmul_backward' "$LINALG_MD"

echo "cpu_ops linalg JVP/VJP integration test passed"
