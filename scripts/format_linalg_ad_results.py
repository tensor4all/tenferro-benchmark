#!/usr/bin/env python3
"""Format linalg JVP/VJP rows from a cpu_ops CSV as a backend comparison table."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from format_cpu_ops_results import BACKEND_LABELS, clean_markdown_eof, normalize_row

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


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cpu-ops.csv>", file=sys.stderr)
        sys.exit(1)
    sys.stdout.write(format_table(Path(sys.argv[1])))


if __name__ == "__main__":
    main()
