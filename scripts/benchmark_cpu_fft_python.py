#!/usr/bin/env python3
"""Run CPU FFT benchmark items for PyTorch and append normalized CSV rows."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path


def runs_from_env() -> tuple[int, int]:
    profile = os.environ.get("PUBLICATION_GATE_PROFILE", "quick").lower()
    default_runs = 15 if profile == "full" else 7
    runs = int(os.environ.get("BENCH_RUNS", default_runs))
    warmups = int(os.environ.get("BENCH_WARMUPS", 3))
    return runs, warmups


def parse_lengths(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def lengths_from_env() -> list[int]:
    value = os.environ.get("FFT_BENCH_LENGTHS", "1048576")
    return parse_lengths(value)


def pseudo_value(i: int, seed: int) -> float:
    x = (i * 6364136223846793005 + seed * 1442695040888963407) & ((1 << 64) - 1)
    return ((x % 2048) - 1024) / 1024.0


def torch_input(op: str, dtype: str, n: int):
    import torch

    if op in {"fft", "ifft"}:
        if dtype == "c32":
            real = [pseudo_value(i, 17) for i in range(n)]
            imag = [pseudo_value(i, 18) for i in range(n)]
            return torch.complex(
                torch.tensor(real, dtype=torch.float32),
                torch.tensor(imag, dtype=torch.float32),
            )
        if dtype == "c64":
            real = [pseudo_value(i, 17) for i in range(n)]
            imag = [pseudo_value(i, 18) for i in range(n)]
            return torch.complex(
                torch.tensor(real, dtype=torch.float64),
                torch.tensor(imag, dtype=torch.float64),
            )
    if op == "rfft":
        values = [pseudo_value(i, 29) for i in range(n)]
        if dtype == "f32":
            return torch.tensor(values, dtype=torch.float32)
        if dtype == "f64":
            return torch.tensor(values, dtype=torch.float64)
    if op == "irfft":
        spectrum_len = n // 2 + 1
        real = [pseudo_value(i, 31) for i in range(spectrum_len)]
        imag = [pseudo_value(i, 32) for i in range(spectrum_len)]
        if dtype == "c32":
            return torch.complex(
                torch.tensor(real, dtype=torch.float32),
                torch.tensor(imag, dtype=torch.float32),
            )
        if dtype == "c64":
            return torch.complex(
                torch.tensor(real, dtype=torch.float64),
                torch.tensor(imag, dtype=torch.float64),
            )
    raise ValueError(f"unsupported FFT case op={op} dtype={dtype}")


def median_iqr(times: list[float]) -> tuple[float, float]:
    values = sorted(times)
    return statistics.median(values), values[(3 * len(values)) // 4] - values[len(values) // 4]


def bench(fn: Callable[[], object], runs: int, warmups: int) -> tuple[float, float]:
    for _ in range(warmups):
        value = fn()
        consume(value)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        value = fn()
        consume(value)
        times.append((time.perf_counter() - start) * 1000.0)
    return median_iqr(times)


def consume(value: object) -> None:
    try:
        _ = value.numel()  # type: ignore[attr-defined]
    except AttributeError:
        pass


def emit_case(writer: csv.DictWriter[str], args, op: str, dtype: str, n: int) -> None:
    import torch

    x = torch_input(op, dtype, n)
    if op == "fft":
        fn = lambda: torch.fft.fft(x, dim=-1, norm=None)
    elif op == "ifft":
        fn = lambda: torch.fft.ifft(x, dim=-1, norm=None)
    elif op == "rfft":
        fn = lambda: torch.fft.rfft(x, dim=-1, norm=None)
    elif op == "irfft":
        fn = lambda: torch.fft.irfft(x, n=n, dim=-1, norm=None)
    else:
        raise ValueError(op)

    try:
        median_ms, iqr_ms = bench(fn, args.runs, args.warmups)
        writer.writerow(
            {
                "suite": "cpu/fft",
                "benchmark": op,
                "dtype": dtype,
                "threads": args.num_threads,
                "shape": f"1d_n{n}",
                "backend": "pytorch-cpu",
                "median_ms": f"{median_ms:.6f}",
                "iqr_ms": f"{iqr_ms:.6f}",
                "status": "ok",
                "notes": "torch.fft warmup before measured runs; input allocation outside timed region",
            }
        )
    except Exception as exc:  # noqa: BLE001
        writer.writerow(
            {
                "suite": "cpu/fft",
                "benchmark": op,
                "dtype": dtype,
                "threads": args.num_threads,
                "shape": f"1d_n{n}",
                "backend": "pytorch-cpu",
                "median_ms": "",
                "iqr_ms": "",
                "status": "failed",
                "notes": str(exc),
            }
        )


def configure_torch_threads(num_threads: int) -> None:
    import torch

    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-threads", required=True, type=int)
    parser.add_argument("--lengths", default=",".join(str(v) for v in lengths_from_env()))
    args = parser.parse_args()
    args.lengths = parse_lengths(args.lengths)
    args.runs, args.warmups = runs_from_env()

    configure_torch_threads(args.num_threads)

    append = args.output.exists()
    with args.output.open("a", newline="") as f:
        fieldnames = [
            "suite",
            "benchmark",
            "dtype",
            "threads",
            "shape",
            "backend",
            "median_ms",
            "iqr_ms",
            "status",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        if not append:
            writer.writeheader()
        for n in args.lengths:
            for op, dtype in (
                ("fft", "c32"),
                ("fft", "c64"),
                ("ifft", "c32"),
                ("ifft", "c64"),
                ("rfft", "f32"),
                ("rfft", "f64"),
                ("irfft", "c32"),
                ("irfft", "c64"),
            ):
                emit_case(writer, args, op, dtype, n)


if __name__ == "__main__":
    main()
