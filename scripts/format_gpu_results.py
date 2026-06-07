#!/usr/bin/env python3
"""Format GPU benchmark JSONL results as Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from collect_cpu_info import collect_cpu_info, markdown as cpu_info_markdown
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

PREFERRED_BACKENDS = [
    "tenferro-cuda-trace",
    "tenferro-cuda-eager",
    "pytorch-cuda",
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
    "jax-cuda": "JAX CUDA",
    "cublaslt": "cuBLASLt",
    "cutlass": "CUTLASS",
    "cusolver": "cuSOLVER",
    "cusparse": "cuSPARSE",
    "ginkgo": "Ginkgo",
}


def clean_markdown_eof(markdown: str) -> str:
    return markdown.rstrip() + "\n"


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
            return "ok (missing median)"
        return f"{median:.3f}"
    if status == "not_configured":
        return "not configured"
    if status == "verification_failed":
        return "verification failed"
    if status == "cpu_fallback":
        return "CPU fallback"
    return status.replace("_", " ")


def markdown_cell(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|")


def choose_duplicate_record(existing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return candidate


def load_run_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open() as fh:
        metadata = yaml.safe_load(fh)
    return metadata if isinstance(metadata, dict) else None


def run_metadata_lines(metadata: dict[str, Any] | None) -> list[str]:
    if metadata is None:
        return []
    lines = [
        f"- Target profile: `{metadata.get('target_profile', 'unknown')}`",
        f"- Suite: `{metadata.get('suite_id', 'unknown')}`",
        f"- Suite file: `{metadata.get('suite_file', 'unknown')}`",
        f"- Timestamp: `{metadata.get('timestamp', 'unknown')}`",
    ]
    tenferro = metadata.get("tenferro_rs")
    if isinstance(tenferro, dict) and tenferro.get("commit"):
        lines.append(f"- tenferro-rs commit: `{tenferro['commit']}`")
    return lines + [""]


def format_markdown(records: list[dict[str, Any]], run_metadata: dict[str, Any] | None = None) -> str:
    if not records:
        return clean_markdown_eof("# GPU Benchmark Results\n\nNo GPU benchmark records found.")

    backends = backend_order(records)
    by_suite_op: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_suite_op[(record["suite_id"], record["op"])].append(record)

    lines = ["# GPU Benchmark Results", ""]
    lines.extend(run_metadata_lines(run_metadata))
    lines.append(cpu_info_markdown(collect_cpu_info()).rstrip())
    lines.append("")
    lines.append("Median time is reported in milliseconds for `ok` records.")
    lines.append("Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.")
    lines.append("Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.")
    lines.append("Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.")
    lines.append("tenferro-rs uses native column-major GPU tensors; Torch, JAX, and vendor-wrapper columns use their native row-major framework tensors unless noted.")
    lines.append("The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently JAX-compatible gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.")
    lines.append("Non-`ok` cells show the structured backend status.")
    lines.append("")

    for (suite_id, op), group in sorted(by_suite_op.items()):
        lines.append(f"## {markdown_cell(suite_id)} / {markdown_cell(op)}")
        lines.append("")
        if op == "svd":
            lines.append("> **SVD note:** SVD rows use synchronized timed regions and matched Rust/Python input generators. tenferro-rs CUDA uses its backend default driver policy, currently JAX-compatible gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise. The cuSOLVER column pins torch.linalg.svd driver=gesvd as a QR-based cuSOLVER comparison; compare PyTorch/JAX default rows separately because they may use different SVD drivers and row-major framework layouts.")
            lines.append("")
        header = "| Problem | " + " | ".join(markdown_cell(BACKEND_LABELS.get(b, b)) for b in backends) + " |"
        separator = "|---|" + "|".join("---:" for _ in backends) + "|"
        lines.append(header)
        lines.append(separator)

        by_problem_backend: dict[tuple[str, str], dict[str, Any]] = {}
        problem_ids = sorted({record["problem_id"] for record in group})
        for record in group:
            key = (record["problem_id"], record["backend"])
            if key in by_problem_backend:
                by_problem_backend[key] = choose_duplicate_record(by_problem_backend[key], record)
            else:
                by_problem_backend[key] = record

        for problem_id in problem_ids:
            row = [markdown_cell(problem_id)]
            for backend in backends:
                row.append(markdown_cell(format_cell(by_problem_backend.get((problem_id, backend)))))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return clean_markdown_eof("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    markdown = format_markdown(load_records(args.results), load_run_metadata(args.run_metadata))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
