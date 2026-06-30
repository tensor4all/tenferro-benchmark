#!/usr/bin/env bash
set -euo pipefail

# Runs the focused CPU benchmark items where this repository has a native
# runner. Includes primal linalg, trace-mode JVP/VJP (tenferro-trace), and
# eager backward. Output is normalized to the backend-comparison CSV consumed by
# format_cpu_ops_results.py.

NUM_THREADS="${1:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-$PROJECT_DIR/data/results}"
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

# shellcheck source=scripts/python_venv.sh
source "$SCRIPT_DIR/python_venv.sh"
reset_benchmark_python_venv "$PROJECT_DIR"
prepare_cpu_benchmark_python_venv "$PROJECT_DIR"

# shellcheck source=scripts/cpu_blas_provider.sh
source "$SCRIPT_DIR/cpu_blas_provider.sh"

PUBLICATION_GATE_FEATURES="$(normalize_cpu_blas_features "${PUBLICATION_GATE_FEATURES:-}")"
export PUBLICATION_GATE_FEATURES
export PUBLICATION_GATE_TENFERRO_MODE="${PUBLICATION_GATE_TENFERRO_MODE:-both}"

# shellcheck source=scripts/thread_env.sh
source "$SCRIPT_DIR/thread_env.sh"
configure_cpu_thread_env "$NUM_THREADS"

mkdir -p "$RESULTS_DIR"

RAW_LOG="$RESULTS_DIR/publication_gate_${PUBLICATION_GATE_FEATURES}_t${NUM_THREADS}_${PUBLICATION_GATE_PROFILE:-quick}_${PUBLICATION_GATE_SUITE:-all}_${TIMESTAMP}.csv"
CPU_OPS_LOG="$RESULTS_DIR/cpu_ops_t${NUM_THREADS}_${TIMESTAMP}.csv"
LINALG_AD_MD="$RESULTS_DIR/linalg_jvp_vjp_t${NUM_THREADS}_${TIMESTAMP}.md"

ensure_blas_env_for_features "$PUBLICATION_GATE_FEATURES"

"$SCRIPT_DIR/run_publication_gate.sh" "$NUM_THREADS"

python3 - "$RAW_LOG" "$CPU_OPS_LOG" "$NUM_THREADS" <<'PY'
import csv
import sys

raw_path, out_path, threads = sys.argv[1:]
with open(raw_path, newline="") as f:
    rows = list(csv.DictReader(f))

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "suite",
            "benchmark",
            "dtype",
            "threads",
            "shape",
            "backend",
            "median_ms",
            "iqr_ms",
            "status",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        benchmark = row["op"]
        phase = row.get("phase", "")
        if phase and phase != "primal":
            benchmark = f"{benchmark}_{phase}"
        backend = row["backend"]
        if backend in {"cpu-faer", "system-openblas", "system-accelerate", "cuda"}:
            backend = "tenferro-eager"
        writer.writerow(
            {
                "suite": row["suite"],
                "benchmark": benchmark,
                "dtype": row["dtype"],
                "threads": threads,
                "shape": row["shape"],
                "backend": backend,
                "median_ms": row["median_ms"],
                "iqr_ms": row["iqr_ms"],
                "status": row["status"],
            }
        )
PY

if command -v uv >/dev/null 2>&1; then
    uv run python "$SCRIPT_DIR/benchmark_cpu_ops_python.py" \
        --backend pytorch-cpu \
        --num-threads "$NUM_THREADS" \
        --output "$CPU_OPS_LOG"
    uv run python "$SCRIPT_DIR/benchmark_cpu_ops_python.py" \
        --backend jax-cpu \
        --num-threads "$NUM_THREADS" \
        --output "$CPU_OPS_LOG"
else
    python3 "$SCRIPT_DIR/benchmark_cpu_ops_python.py" \
        --backend pytorch-cpu \
        --num-threads "$NUM_THREADS" \
        --output "$CPU_OPS_LOG"
    python3 "$SCRIPT_DIR/benchmark_cpu_ops_python.py" \
        --backend jax-cpu \
        --num-threads "$NUM_THREADS" \
        --output "$CPU_OPS_LOG"
fi

echo "CPU ops results saved to: $CPU_OPS_LOG"
if [[ -f "$CPU_OPS_LOG" ]]; then
    linalg_ad_count="$(grep -Ec '_jvp,|_vjp,' "$CPU_OPS_LOG" || true)"
    if [[ -z "$linalg_ad_count" || "$linalg_ad_count" == "0" ]]; then
        echo "WARNING: no linalg JVP/VJP rows found in $CPU_OPS_LOG" >&2
    else
        echo "Linalg JVP/VJP rows: $linalg_ad_count"
        if command -v uv >/dev/null 2>&1; then
            uv run python "$SCRIPT_DIR/format_linalg_ad_results.py" "$CPU_OPS_LOG" > "$LINALG_AD_MD" \
                || python3 "$SCRIPT_DIR/format_linalg_ad_results.py" "$CPU_OPS_LOG" > "$LINALG_AD_MD"
        else
            python3 "$SCRIPT_DIR/format_linalg_ad_results.py" "$CPU_OPS_LOG" > "$LINALG_AD_MD"
        fi
        echo "Linalg JVP/VJP markdown saved to: $LINALG_AD_MD"
    fi
fi
