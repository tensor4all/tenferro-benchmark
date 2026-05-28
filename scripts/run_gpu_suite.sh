#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
TIMESTAMP="${GPU_BENCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

SUITES_VALUE="${GPU_BENCH_SUITE:-benchmarks/gpu/dense.yaml,benchmarks/gpu/einsum.yaml,benchmarks/gpu/sparse.yaml}"
BACKENDS_VALUE="${GPU_BENCH_BACKENDS:-tenferro-cuda-trace,tenferro-cuda-eager,pytorch-cuda,libtorch-cuda,jax-cuda,cublaslt,cutlass,cusolver,cusparse,ginkgo}"
DEVICE_ORDINAL="${GPU_BENCH_DEVICE:-0}"
PROBLEM_FILTER="${GPU_BENCH_PROBLEM:-}"

mkdir -p "$RESULTS_DIR" "$REPORTS_DIR"

IFS=',' read -r -a SUITES <<< "$SUITES_VALUE"
IFS=',' read -r -a BACKENDS <<< "$BACKENDS_VALUE"

echo "============================================"
echo " GPU benchmark contract suite"
echo "============================================"
echo "Suites:   $SUITES_VALUE"
echo "Backends: $BACKENDS_VALUE"
echo "Device:   cuda:$DEVICE_ORDINAL"
echo "Timestamp: $TIMESTAMP"
echo ""

VALIDATOR=(python3 "$SCRIPT_DIR/validate_benchmark_suite.py")
FORMATTER=(python3 "$SCRIPT_DIR/format_gpu_results.py")
PYTHON=(python3)
if command -v uv >/dev/null 2>&1; then
    VALIDATOR=(uv run python "$SCRIPT_DIR/validate_benchmark_suite.py")
    FORMATTER=(uv run python "$SCRIPT_DIR/format_gpu_results.py")
    PYTHON=(uv run python)
fi

"${VALIDATOR[@]}" "${SUITES[@]}"

RESULT_JSONL="$RESULTS_DIR/gpu_contract_${TIMESTAMP}.jsonl"
"${PYTHON[@]}" - "$RESULT_JSONL" "$DEVICE_ORDINAL" "$PROBLEM_FILTER" "${BACKENDS[@]}" -- "${SUITES[@]}" <<'PY'
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

output = Path(sys.argv[1])
device_ordinal = int(sys.argv[2])
problem_filter = sys.argv[3]
separator = sys.argv.index("--")
backends = sys.argv[4:separator]
suites = [Path(p) for p in sys.argv[separator + 1:]]


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


benchmark_commit = git_commit(Path.cwd())
tenferro_commit = git_commit(Path.cwd() / "extern" / "tenferro-rs")
timestamp = datetime.now(timezone.utc).isoformat()

with output.open("w") as fh:
    for suite_path in suites:
        suite = yaml.safe_load(suite_path.read_text())
        for problem in suite["problems"]:
            if problem_filter and problem["id"] != problem_filter:
                continue
            for backend in backends:
                if backend not in problem["backend_candidates"]:
                    status = "unsupported"
                    reason = f"{backend} is not listed for {problem['id']}"
                else:
                    status = "not_configured"
                    reason = "Phase 1 contract smoke does not execute GPU kernels"
                record = {
                    "schema_version": 1,
                    "suite_id": suite["suite_id"],
                    "problem_id": problem["id"],
                    "op": problem["op"],
                    "backend": backend,
                    "status": status,
                    "timing": {
                        "warmup_runs": 0,
                        "timed_runs": 0,
                        "compile_time_ms": None,
                        "first_run_ms": None,
                        "median_ms": None,
                        "min_ms": None,
                        "p95_ms": None,
                        "iqr_ms": None,
                        "timing_scope": "steady_state_host_api_plus_device_sync",
                    },
                    "performance": {
                        "tflops": None,
                        "effective_bandwidth_gbps": None,
                        "peak_memory_bytes": None,
                    },
                    "verification": {
                        "status": "skipped",
                        "reference_backend": None,
                        "max_abs_error": None,
                        "max_rel_error": None,
                        "residual": None,
                        "rtol": problem["verify"].get("rtol"),
                        "atol": problem["verify"].get("atol"),
                        "reason": reason,
                    },
                    "environment": {
                        "hostname": socket.gethostname(),
                        "timestamp_utc": timestamp,
                        "os": platform.platform(),
                        "gpu_name": None,
                        "gpu_uuid": None,
                        "gpu_memory_bytes": None,
                        "driver_version": None,
                        "cuda_version": None,
                        "cudnn_version": None,
                        "framework_version": None,
                        "tenferro_rs_commit": tenferro_commit,
                        "benchmark_repo_commit": benchmark_commit,
                        "env": {
                            "CUDA_PATH": os.environ.get("CUDA_PATH"),
                            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
                        },
                    },
                    "execution": {
                        "device": "cuda",
                        "device_ordinal": device_ordinal,
                        "execution_path": "phase1-contract-smoke",
                        "synchronization": "not executed",
                        "layout": json.dumps(problem.get("layout", {}), sort_keys=True),
                        "dtype": json.dumps(problem.get("dtype", {}), sort_keys=True),
                        "notes": "Generated by scripts/run_gpu_suite.sh Phase 1 smoke path.",
                        "unsupported_reason": reason,
                    },
                }
                fh.write(json.dumps(record, sort_keys=True) + "\n")
PY

"${VALIDATOR[@]}" --kind result "$RESULT_JSONL"

MARKDOWN_OUT="$RESULTS_DIR/gpu_results_${TIMESTAMP}.md"
REPORT_OUT="$REPORTS_DIR/gpu-benchmark-results.md"
"${FORMATTER[@]}" "$RESULT_JSONL" --output "$MARKDOWN_OUT"
cp "$MARKDOWN_OUT" "$REPORT_OUT"

echo ""
echo "GPU contract smoke complete"
echo "JSONL:    $RESULT_JSONL"
echo "Markdown: $MARKDOWN_OUT"
echo "Report:   $REPORT_OUT"
