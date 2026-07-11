#!/usr/bin/env python3
"""Format GPU linalg JVP/VJP JSONL records as a backend comparison table."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from collect_cpu_info import collect_cpu_info, markdown as cpu_info_markdown
from collect_gpu_info import markdown as gpu_info_markdown, resolve_gpu_info

BACKEND_ORDER = [
    "tenferro-cuda-trace",
    "pytorch-cuda",
    "jax-cuda",
]

BACKEND_LABELS = {
    "tenferro-cuda-trace": "tenferro-rs CUDA trace",
    "pytorch-cuda": "PyTorch CUDA",
    "jax-cuda": "JAX CUDA",
}

LOSS_NOTES = {
    "grad_sum_svd_s": "loss = sum(singular values); w.r.t. input matrix A",
    "grad_sum_qr": "loss = sum(Q) + sum(R); w.r.t. input matrix A",
    "grad_sum_eigh": "loss = sum(eigenvalues); w.r.t. SPD input matrix A",
    "grad_sum_lu": "loss = sum(L) + sum(U); w.r.t. input matrix A",
    "grad_sum_solve": "loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)",
}


def clean_markdown_eof(markdown: str) -> str:
    return markdown.rstrip() + "\n"


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def load_run_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open() as fh:
        metadata = yaml.safe_load(fh)
    return metadata if isinstance(metadata, dict) else None


def format_cell(record: dict[str, Any] | None) -> str:
    if record is None:
        return "-"
    status = record["status"]
    if status == "ok":
        median = record["timing"]["median_ms"]
        iqr = record["timing"]["iqr_ms"]
        if median is None:
            return "ok (missing median)"
        if iqr is None:
            return f"{median:.3f}"
        return f"{median:.3f} ± {iqr:.3f}"
    if status == "not_configured":
        return "not configured"
    return status.replace("_", " ")


def shape_from_problem_id(problem_id: str, loss: str) -> str:
    parts = problem_id.split("_")
    n = parts[-1]
    if loss == "grad_sum_solve":
        return f"{n}x{n},rhs=1"
    return f"{n}x{n}"


def problem_meta(problem_id: str) -> tuple[str, str, str]:
    # linalg_ad_{suite}_{loss}_{phase}_f64_{n}
    parts = problem_id.split("_")
    suite = parts[2]
    loss = "_".join(parts[3:-3])
    phase = parts[-3]
    benchmark = f"{loss}_{phase}"
    shape = shape_from_problem_id(problem_id, loss)
    return suite, benchmark, shape


def format_markdown(
    records: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None = None,
) -> str:
    by_key: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        suite, benchmark, shape = problem_meta(record["problem_id"])
        key = (suite, benchmark, "f64", shape)
        backend = record["backend"]
        if backend in BACKEND_ORDER:
            by_key[key][backend] = record

    lines = ["# GPU Linalg JVP/VJP Benchmark Results", ""]
    if run_metadata:
        lines.extend(
            [
                f"- Target profile: `{run_metadata.get('target_profile', 'nvidia-gpu')}`",
                f"- Suite: `{run_metadata.get('suite_id', 'gpu/linalg_jvp_vjp')}`",
                f"- Suite file: `{run_metadata.get('suite_file', 'benchmarks/gpu/linalg_jvp_vjp.yaml')}`",
                f"- Timestamp: `{run_metadata.get('timestamp', 'unknown')}`",
            ]
        )
        tenferro = run_metadata.get("tenferro_rs")
        if isinstance(tenferro, dict) and tenferro.get("commit"):
            lines.append(f"- tenferro-rs commit: `{tenferro['commit']}`")
        lines.append("")

    gpu_info = resolve_gpu_info(records, run_metadata)
    gpu_markdown = gpu_info_markdown(gpu_info)
    if gpu_markdown:
        lines.append(gpu_markdown.rstrip())
        lines.append("")
    lines.append(cpu_info_markdown(collect_cpu_info()).rstrip())
    lines.append("")
    lines.append("Median ± IQR (ms). Missing backends are shown as `-`.")
    lines.append("")
    lines.append(
        "tenferro-rs JVP/VJP use trace-mode `AdContext` on CUDA; PyTorch uses "
        "`torch.func.jvp` / `vjp` on CUDA; JAX uses `jax.jvp` / `jax.vjp` on CUDA."
    )
    tenferro_records = [r for r in records if r.get("backend") == "tenferro-cuda-trace"]
    if tenferro_records and all(r.get("status") == "unsupported" for r in tenferro_records):
        lines.append(
            "tenferro-rs CUDA trace linalg JVP/VJP is currently unsupported: compiled AD "
            "graphs hit scalar-vector `mul` shape mismatches because the CUDA backend does "
            "not yet implement CPU-style 0-D scalar broadcast."
        )
    elif any(
        r.get("backend") == "tenferro-cuda-trace" and r.get("status") == "unsupported"
        for r in records
    ):
        lines.append(
            "tenferro-rs CUDA trace linalg JVP/VJP is partially unsupported: `grad_sum_qr` "
            "hits scalar-vector `mul` shape mismatches on CUDA because the backend does not "
            "yet implement CPU-style 0-D scalar broadcast."
        )
    lines.append(
        "Inputs are uploaded to the GPU before timed runs; initial host-to-device "
        "transfer is outside the timed region."
    )
    lines.append(
        "Timed runs include the host API call and backend-native device synchronization "
        "without downloading AD outputs in the timed region."
    )
    lines.append("")

    lines.extend(
        [
            "## Linalg JVP/VJP Benchmark Items",
            "",
            "| suite | benchmark | dtype | shape | "
            + " | ".join(BACKEND_LABELS[b] for b in BACKEND_ORDER)
            + " |",
            "|---|---|---:|---|"
            + "|".join("---:" for _ in BACKEND_ORDER)
            + "|",
        ]
    )

    for suite, benchmark, dtype, shape in sorted(by_key):
        backends = by_key[(suite, benchmark, dtype, shape)]
        row = [
            suite,
            f"`{benchmark}`",
            dtype,
            f"`{shape}`",
            *(format_cell(backends.get(backend)) for backend in BACKEND_ORDER),
        ]
        lines.append("| " + " | ".join(row) + " |")

    covered = sorted({benchmark.rsplit("_", 1)[0] for _, benchmark, _, _ in by_key})
    if covered:
        lines.extend(["", "## Loss Definitions", ""])
        for op in covered:
            note = LOSS_NOTES.get(op)
            if note:
                lines.append(f"- `{op}`: {note}")

    return clean_markdown_eof("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    markdown = format_markdown(
        load_records(args.results),
        load_run_metadata(args.run_metadata),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
