#!/usr/bin/env python3
"""Format CPU FFT benchmark CSV rows as a backend comparison table."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


BACKEND_ORDER = [
    "tenferro-fft-immediate",
    "tenferro-fft-executor-cached",
    "pytorch-cpu",
]

BACKEND_LABELS = {
    "tenferro-fft-immediate": "tenferro-rs immediate (ms)",
    "tenferro-fft-executor-cached": "tenferro-rs FftExecutor cached (ms)",
    "pytorch-cpu": "PyTorch torch.fft (ms)",
}


def row_value(row: dict[str, str], name: str) -> str:
    return row.get(name, "").strip()


def format_value(row: dict[str, str]) -> str:
    median_ms = row_value(row, "median_ms")
    iqr_ms = row_value(row, "iqr_ms")
    if not median_ms:
        status = row_value(row, "status") or "-"
        return status if status != "ok" else "-"
    if iqr_ms:
        return f"{float(median_ms):.3f} ± {float(iqr_ms):.3f}"
    return f"{float(median_ms):.3f}"


def format_table(paths: list[Path]) -> str:
    by_key: dict[tuple[str, str, str, str, str], dict[str, str]] = defaultdict(dict)
    notes: set[str] = set()
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                key = (
                    row_value(row, "suite"),
                    row_value(row, "benchmark"),
                    row_value(row, "dtype"),
                    row_value(row, "threads"),
                    row_value(row, "shape"),
                )
                backend = row_value(row, "backend")
                if backend:
                    by_key[key][backend] = format_value(row)
                note = row_value(row, "notes")
                if note:
                    notes.add(f"{backend}: {note}")

    lines = [
        "## CPU FFT Benchmark Items",
        "",
        "Median ± IQR (ms). Missing backends are shown as `-`.",
        "",
        "Timing scope: input tensors are created outside the timed region; each timed call creates the FFT output tensor. "
        "Rows are limited to 1D transforms so tenferro-rs column-major layout and PyTorch row-major layout do not change the measured transform axis.",
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
        lines.append(
            "| "
            + " | ".join(
                [
                    suite,
                    f"`{benchmark}`",
                    dtype,
                    threads,
                    f"`{shape}`",
                    *(values.get(backend, "-") for backend in BACKEND_ORDER),
                ]
            )
            + " |"
        )

    if notes:
        lines.extend(["", "Notes:", ""])
        for note in sorted(notes):
            lines.append(f"- {note}")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cpu-fft.csv>...", file=sys.stderr)
        sys.exit(1)
    sys.stdout.write(format_table([Path(value) for value in sys.argv[1:]]))


if __name__ == "__main__":
    main()
