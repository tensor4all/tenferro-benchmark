#!/usr/bin/env python3
"""Format linalg JVP/VJP rows from a cpu_ops CSV as a backend comparison table."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import report_summary
from format_cpu_ops_results import BACKEND_LABELS, clean_markdown_eof, normalize_row, row_value

LINALG_AD_BACKEND_ORDER = [
    "tenferro-trace",
    "pytorch-cpu",
    "jax-cpu",
]

LOSS_NOTES = {
    "grad_sum_svd_s": "loss = sum(singular values); w.r.t. input matrix A",
    "grad_sum_qr": "loss = sum(Q) + sum(R); w.r.t. input matrix A",
    "grad_sum_eigh": "loss = sum(eigenvalues); w.r.t. SPD input matrix A",
    "grad_sum_lu": "loss = sum(L) + sum(U); w.r.t. input matrix A",
    "grad_sum_solve": "loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)",
}


def is_linalg_ad_benchmark(benchmark: str) -> bool:
    return benchmark.endswith("_jvp") or benchmark.endswith("_vjp")


def base_op(benchmark: str) -> str:
    for suffix in ("_jvp", "_vjp"):
        if benchmark.endswith(suffix):
            return benchmark[: -len(suffix)]
    return benchmark


def format_table(path: Path) -> str:
    by_key: dict[tuple[str, str, str, str, str], dict[str, str]] = defaultdict(dict)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key, backend, value = normalize_row(row)
            _, benchmark, _, _, _ = key
            if not is_linalg_ad_benchmark(benchmark):
                continue
            if backend not in LINALG_AD_BACKEND_ORDER:
                continue
            by_key[key][backend] = value

    lines = [
        "## Linalg JVP/VJP Benchmark Items",
        "",
        "Median ± IQR (ms). Missing backends are shown as `-`.",
        "",
        "tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /",
        "`vjp`; JAX uses `jax.jvp` / `jax.vjp`.",
        "",
        "| suite | benchmark | dtype | threads | shape | "
        + " | ".join(BACKEND_LABELS[b] for b in LINALG_AD_BACKEND_ORDER)
        + " |",
        "|---|---|---:|---:|---|"
        + "|".join("---:" for _ in LINALG_AD_BACKEND_ORDER)
        + "|",
    ]

    for key in sorted(by_key):
        suite, benchmark, dtype, threads, shape = key
        values = by_key[key]
        row = [
            suite,
            f"`{benchmark}`",
            dtype,
            threads,
            f"`{shape}`",
            *(values.get(backend, "-") for backend in LINALG_AD_BACKEND_ORDER),
        ]
        lines.append("| " + " | ".join(row) + " |")

    covered = sorted({base_op(key[1]) for key in by_key})
    if covered:
        lines.extend(["", "## Loss Definitions", ""])
        for op in covered:
            note = LOSS_NOTES.get(op)
            if note:
                lines.append(f"- `{op}`: {note}")

    return clean_markdown_eof("\n".join(lines))


def build_summary(
    path: Path,
    run_metadata: dict | None = None,
    *,
    suite_id: str | None = None,
    run_dir: Path | None = None,
    cpu_model: str | None = None,
) -> dict:
    """Machine-readable sidecar for the same CSV rows the table is built from."""
    header = report_summary.metadata_from_run_metadata(run_metadata)
    hardware = dict(header["hardware"])
    if cpu_model:
        hardware["cpu"] = cpu_model
    if not hardware.get("cpu"):
        hardware["cpu"] = "unknown"
    rows = []
    threads_seen: set[int] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key, backend, _value = normalize_row(row)
            if not backend or backend not in LINALG_AD_BACKEND_ORDER:
                continue
            _suite, benchmark, dtype, threads, shape = key
            if not is_linalg_ad_benchmark(benchmark):
                continue
            status = row_value(row, "status") or "ok"
            median_ms = row_value(row, "median_ms")
            if not median_ms:
                mean_us = row_value(row, "mean_us")
                if mean_us:
                    median_ms = str(float(mean_us) / 1000.0)
            median = float(median_ms) if (median_ms and status == "ok" and float(median_ms) > 0) else None
            if status == "ok" and median is None:
                status = "missing"
            iqr_ms = row_value(row, "iqr_ms")
            if threads.isdigit():
                threads_seen.add(int(threads))
            rows.append(
                report_summary.make_row(
                    benchmark,
                    backend,
                    median,
                    status=status,
                    iqr=float(iqr_ms) if iqr_ms else None,
                    group=f"dtype={dtype} / threads={threads} / shape={shape}",
                    verified=status == "ok" and median is not None,
                )
            )
    if threads_seen:
        hardware["threads"] = sorted(threads_seen)
    resolved_suite = suite_id or header["suite_id"]
    if resolved_suite in (None, "", "cpu/einsum"):
        resolved_suite = "cpu/linalg_jvp_vjp"
    return report_summary.build_document(
        rows,
        target_profile=header["target_profile"],
        suite_id=resolved_suite,
        timestamp=header["timestamp"],
        tenferro_commit=header["tenferro_commit"],
        hardware=hardware,
        unit="ms",
        **(report_summary.v2_provenance(run_dir, run_metadata) if run_dir is not None else {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvfile", type=Path)
    parser.add_argument("--run-metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    markdown = format_table(args.csvfile)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
    else:
        sys.stdout.write(markdown)
    run_metadata = (
        report_summary.load_run_metadata(args.run_metadata)
        if args.run_metadata
        else report_summary.env_run_metadata()
    )
    run_dir = args.run_metadata.parent if args.run_metadata else None
    report_summary.maybe_write(
        build_summary(
            args.csvfile,
            run_metadata,
            run_dir=run_dir,
            cpu_model=report_summary.cpu_model_from_run_dir(run_dir),
        ),
        summary_path=report_summary.summary_path_for_report(args.output) if args.output else None,
        env=os.environ,
    )


if __name__ == "__main__":
    main()
