#!/usr/bin/env bash
set -euo pipefail

# Runs the focused CPU benchmark items introduced in tensor4all/tenferro-rs#884
# where this repository has a native runner. The output is normalized to the
# backend-comparison CSV consumed by format_cpu_ops_results.py.

NUM_THREADS="${1:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-$PROJECT_DIR/data/results}"
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

# shellcheck source=scripts/thread_env.sh
source "$SCRIPT_DIR/thread_env.sh"
configure_cpu_thread_env "$NUM_THREADS"

mkdir -p "$RESULTS_DIR"

RAW_LOG="$RESULTS_DIR/publication_gate_system-openblas_t${NUM_THREADS}_${PUBLICATION_GATE_PROFILE:-quick}_${PUBLICATION_GATE_SUITE:-all}_${TIMESTAMP}.csv"
CPU_OPS_LOG="$RESULTS_DIR/cpu_ops_t${NUM_THREADS}_${TIMESTAMP}.csv"
BUILD_DIR="$PROJECT_DIR/build/cpp-libtorch"

if [[ -z "${OPENBLAS_ROOT:-}" ]]; then
    if command -v brew >/dev/null 2>&1 && brew --prefix openblas >/dev/null 2>&1; then
        export OPENBLAS_ROOT="$(brew --prefix openblas)"
    fi
fi

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
        if backend in {"cpu-faer", "system-openblas", "cuda"}:
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

if [[ -n "${Torch_DIR:-}" ]]; then
    cmake -S "$PROJECT_DIR/cpp" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTorch_DIR="$Torch_DIR" \
        -DOPENBLAS_ROOT="${OPENBLAS_ROOT:-}" \
        -DREQUIRE_TORCH_OPENBLAS=ON
    cmake --build "$BUILD_DIR" --target benchmark_cpu_ops_libtorch --config Release
    "$BUILD_DIR/benchmark_cpu_ops_libtorch" \
        --num-threads "$NUM_THREADS" \
        --output "$CPU_OPS_LOG"
else
    echo "WARNING: Torch_DIR is not set; skipping Torch C++ CPU ops." >&2
fi

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
