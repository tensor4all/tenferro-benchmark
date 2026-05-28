#!/usr/bin/env python3
"""Format GPU benchmark JSONL results as Markdown."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PREFERRED_BACKENDS = [
    "tenferro-cuda-eager",
    "tenferro-cuda-trace",
    "pytorch-cuda",
    "libtorch-cuda",
    "jax-cuda",
    "cublaslt",
    "cutlass",
    "cusolver",
    "cusparse",
    "ginkgo",
]

BACKEND_LABELS = {
    "tenferro-cuda-eager": "tenferro-rs CUDA eager",
    "tenferro-cuda-trace": "tenferro-rs CUDA trace",
    "pytorch-cuda": "PyTorch CUDA",
    "libtorch-cuda": "LibTorch CUDA",
    "jax-cuda": "JAX CUDA",
    "cublaslt": "cuBLASLt",
    "cutlass": "CUTLASS",
    "cusolver": "cuSOLVER",
    "cusparse": "cuSPARSE",
    "ginkgo": "Ginkgo",
}


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
    return records


def backend_order(records: list[dict[str, Any]]) -> list[str]:
    found = {record["backend"] for record in records}
    ordered = [backend for backend in PREFERRED_BACKENDS if backend in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def format_cell(record: dict[str, Any] | None) -> str:
    if record is None:
        return "-"
    status = record["status"]
    if status == "ok":
        median = record["timing"]["median_ms"]
        if median is None:
            return "ok"
        return f"{median:.3f}"
    if status == "not_configured":
        return "not configured"
    if status == "verification_failed":
        return "verification failed"
    if status == "cpu_fallback":
        return "CPU fallback"
    return status.replace("_", " ")


def format_markdown(records: list[dict[str, Any]]) -> str:
    if not records:
        return "# GPU Benchmark Results\n\nNo GPU benchmark records found.\n"

    backends = backend_order(records)
    by_suite_op: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_suite_op[(record["suite_id"], record["op"])].append(record)

    lines = ["# GPU Benchmark Results", ""]
    lines.append("Median time is reported in milliseconds for `ok` records.")
    lines.append("Non-`ok` cells show the structured backend status.")
    lines.append("")

    for (suite_id, op), group in sorted(by_suite_op.items()):
        lines.append(f"## {suite_id} / {op}")
        lines.append("")
        header = "| Problem | " + " | ".join(BACKEND_LABELS.get(b, b) for b in backends) + " |"
        separator = "|---|" + "|".join("---:" for _ in backends) + "|"
        lines.append(header)
        lines.append(separator)

        by_problem_backend: dict[tuple[str, str], dict[str, Any]] = {}
        problem_ids = sorted({record["problem_id"] for record in group})
        for record in group:
            by_problem_backend[(record["problem_id"], record["backend"])] = record

        for problem_id in problem_ids:
            row = [problem_id]
            for backend in backends:
                row.append(format_cell(by_problem_backend.get((problem_id, backend))))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    markdown = format_markdown(load_records(args.results))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
