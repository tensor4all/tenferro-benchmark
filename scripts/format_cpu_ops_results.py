#!/usr/bin/env python3
"""Format PR884 CPU benchmark CSV rows as a backend comparison table."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


BACKEND_ORDER = [
    "tenferro-eager",
    "tenferro-trace",
    "libtorch-cpu",
    "pytorch-cpu",
    "jax-cpu",
]

BACKEND_LABELS = {
    "tenferro-eager": "tenferro-rs eager mode (ms)",
    "tenferro-trace": "tenferro-rs trace mode (ms)",
    "libtorch-cpu": "Torch C++ (ms)",
    "pytorch-cpu": "PyTorch Python (ms)",
    "jax-cpu": "JAX Python (ms)",
}


def row_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value != "":
            return value
    return ""


def normalize_backend(row: dict[str, str]) -> str:
    backend = row_value(row, "backend").strip()
    if backend in {"cpu-faer", "system-openblas"}:
        return "tenferro-eager"
    return backend


def normalize_row(row: dict[str, str]) -> tuple[tuple[str, str, str, str, str], str, str]:
    suite = row_value(row, "suite")
    benchmark = row_value(row, "benchmark", "name", "op")
    phase = row_value(row, "phase")
    if phase and phase != "primal":
        benchmark = f"{benchmark}_{phase}"
    dtype = row_value(row, "dtype")
    threads = row_value(row, "threads")
    shape = row_value(row, "shape")
    backend = normalize_backend(row)

    median_ms = row_value(row, "median_ms")
    iqr_ms = row_value(row, "iqr_ms")
    if not median_ms:
        mean_us = row_value(row, "mean_us")
        if mean_us:
            median_ms = f"{float(mean_us) / 1000.0:.6f}"
    if not median_ms:
        status = row_value(row, "status") or "-"
        value = status if status != "ok" else "-"
    elif iqr_ms:
        value = f"{float(median_ms):.3f} ± {float(iqr_ms):.3f}"
    else:
        value = f"{float(median_ms):.3f}"

    return (suite, benchmark, dtype, threads, shape), backend, value


def format_table(path: Path) -> str:
    by_key: dict[tuple[str, str, str, str, str], dict[str, str]] = defaultdict(dict)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key, backend, value = normalize_row(row)
            if backend:
                by_key[key][backend] = value

    lines = [
        "## PR884 CPU Benchmark Items",
        "",
        "Median ± IQR (ms). Missing backends are shown as `-`.",
        "",
        "| suite | benchmark | dtype | threads | shape | "
        + " | ".join(BACKEND_LABELS[b] for b in BACKEND_ORDER)
        + " |",
        "|---|---|---:|---:|---|"
        + "|".join("---:" for _ in BACKEND_ORDER)
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
            *(values.get(backend, "-") for backend in BACKEND_ORDER),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu-ops.csv>", file=sys.stderr)
        sys.exit(1)
    print(format_table(Path(sys.argv[1])))


if __name__ == "__main__":
    main()
