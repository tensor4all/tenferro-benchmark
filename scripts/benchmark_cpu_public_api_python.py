#!/usr/bin/env python3
"""Run PyTorch CPU rows for the public API coverage benchmark suite."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path


FIELDNAMES = [
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


def runs_from_env() -> tuple[int, int]:
    profile = os.environ.get("PUBLICATION_GATE_PROFILE", "quick").lower()
    return int(os.environ.get("BENCH_RUNS", 15 if profile == "full" else 7)), int(
        os.environ.get("BENCH_WARMUPS", 3)
    )


def pseudo_value(i: int, seed: int) -> float:
    x = (i * 6364136223846793005 + seed * 1442695040888963407) & ((1 << 64) - 1)
    return ((x % 2048) - 1024) / 1024.0


def values(n: int, seed: int) -> list[float]:
    return [pseudo_value(i, seed) for i in range(n)]


def positive_values(n: int, seed: int) -> list[float]:
    return [0.25 + abs(pseudo_value(i, seed)) for i in range(n)]


def i64_values(n: int, modulus: int) -> list[int]:
    return [(i * 37 + 11) % modulus for i in range(n)]


def bool_values(n: int) -> list[bool]:
    return [i % 3 == 0 for i in range(n)]


def median_iqr(times: list[float]) -> tuple[float, float]:
    sorted_times = sorted(times)
    return (
        statistics.median(sorted_times),
        sorted_times[(3 * len(sorted_times)) // 4] - sorted_times[len(sorted_times) // 4],
    )


def consume(value: object) -> None:
    if isinstance(value, tuple):
        for item in value:
            consume(item)
        return
    try:
        _ = value.numel()  # type: ignore[attr-defined]
    except AttributeError:
        pass


def bench(fn: Callable[[], object], runs: int, warmups: int) -> tuple[float, float]:
    for _ in range(warmups):
        consume(fn())
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        consume(fn())
        times.append((time.perf_counter() - start) * 1000.0)
    return median_iqr(times)


def configure_torch_threads(num_threads: int) -> None:
    import torch

    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def tensor_f64(shape: tuple[int, ...], seed: int):
    import torch

    return torch.tensor(values(math_prod(shape), seed), dtype=torch.float64).reshape(shape)


def tensor_f64_positive(shape: tuple[int, ...], seed: int):
    import torch

    return torch.tensor(positive_values(math_prod(shape), seed), dtype=torch.float64).reshape(shape)


def tensor_c64(shape: tuple[int, ...], seed: int):
    import torch

    real = torch.tensor(values(math_prod(shape), seed), dtype=torch.float64).reshape(shape)
    imag = torch.tensor(values(math_prod(shape), seed + 1), dtype=torch.float64).reshape(shape)
    return torch.complex(real, imag)


def math_prod(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def well_conditioned(n: int, seed: int):
    import torch

    x = torch.tensor(values(n * n, seed), dtype=torch.float64).reshape((n, n))
    x = x.clone()
    x.diagonal().add_(torch.linspace(2.0, 3.0, n, dtype=torch.float64))
    return x


def lower_triangular(n: int, seed: int):
    import torch

    x = torch.zeros((n, n), dtype=torch.float64)
    for col in range(n):
        for row in range(col, n):
            x[row, col] = 2.0 + row / n if row == col else 0.05 * pseudo_value(row + col * n, seed)
    return x


def spd(n: int, seed: int):
    import torch

    x = torch.zeros((n, n), dtype=torch.float64)
    for col in range(n):
        x[col, col] = 2.0 + col / n
        for row in range(col + 1, n):
            value = 0.01 * pseudo_value(row + col * n, seed)
            x[row, col] = value
            x[col, row] = value
    return x


def hpd_c64(n: int, seed: int):
    import torch

    x = torch.zeros((n, n), dtype=torch.complex128)
    for col in range(n):
        x[col, col] = complex(2.0 + col / n, 0.0)
        for row in range(col + 1, n):
            value = complex(0.01 * pseudo_value(row + col * n, seed), 0.01 * pseudo_value(row + col * n, seed + 1))
            x[row, col] = value
            x[col, row] = value.conjugate()
    return x


def make_cases() -> list[tuple[str, str, str, str, str, Callable[[], object] | None]]:
    import torch
    import torch.nn.functional as F

    x = tensor_f64((262_144,), 1)
    y = tensor_f64((262_144,), 2)
    yp = tensor_f64_positive((262_144,), 2)
    xp = tensor_f64_positive((262_144,), 1)
    cond = torch.tensor(bool_values(262_144), dtype=torch.bool)
    lower = torch.full((262_144,), -0.5, dtype=torch.float64)
    upper = torch.full((262_144,), 0.5, dtype=torch.float64)
    matrix = tensor_f64((256, 1024), 1)
    prod_matrix = torch.full((256, 1024), 1.000001, dtype=torch.float64)
    base = tensor_f64((65_536,), 1)
    updates = tensor_f64((65_536,), 2)
    gather_idx = torch.tensor(i64_values(65_536, 65_536), dtype=torch.int64)
    scatter_idx = torch.tensor(i64_values(65_536, 65_536), dtype=torch.int64)
    part_a = tensor_f64((32_768,), 1)
    part_b = tensor_f64((32_768,), 2)
    a128 = well_conditioned(128, 1)
    spd128 = spd(128, 1)
    a64 = well_conditioned(64, 1)
    l128 = lower_triangular(128, 1)
    rhs128x16 = tensor_f64((128, 16), 2)
    rhs64x8 = tensor_f64((64, 8), 2)
    rect128x64 = tensor_f64((128, 64), 1)
    norm256 = tensor_f64((256, 256), 1)
    z = tensor_c64((65_536,), 1)
    z2 = tensor_c64((65_536,), 2)
    zden = torch.full((65_536,), complex(1.5, 0.25), dtype=torch.complex128)
    zlog = torch.full((65_536,), complex(1.5, 0.25), dtype=torch.complex128)
    z128a = tensor_c64((128, 128), 1)
    z128b = tensor_c64((128, 128), 2)
    z32 = tensor_c64((32, 32), 1)
    z32_rhs = tensor_c64((32, 4), 2)
    hpd32 = hpd_c64(32, 1)
    z64 = tensor_c64((64, 64), 1)

    return [
        ("cpu/elementwise_reduction", "add", "f64", "262144", "binary elementwise", lambda: x + y),
        ("cpu/elementwise_reduction", "sub", "f64", "262144", "binary elementwise", lambda: x - y),
        ("cpu/elementwise_reduction", "mul", "f64", "262144", "binary elementwise", lambda: x * y),
        ("cpu/elementwise_reduction", "div", "f64", "262144", "binary elementwise", lambda: x / yp),
        ("cpu/elementwise_reduction", "rem", "f64", "262144", "binary elementwise", lambda: torch.remainder(x, yp)),
        ("cpu/elementwise_reduction", "neg", "f64", "262144", "unary elementwise", lambda: -x),
        ("cpu/elementwise_reduction", "abs", "f64", "262144", "unary elementwise", lambda: torch.abs(x)),
        ("cpu/elementwise_reduction", "sign", "f64", "262144", "unary elementwise", lambda: torch.sign(x)),
        ("cpu/elementwise_reduction", "maximum", "f64", "262144", "binary elementwise", lambda: torch.maximum(x, y)),
        ("cpu/elementwise_reduction", "minimum", "f64", "262144", "binary elementwise", lambda: torch.minimum(x, y)),
        ("cpu/elementwise_reduction", "compare_lt", "f64", "262144", "ordered compare", lambda: x < y),
        ("cpu/elementwise_reduction", "select", "f64", "262144", "ternary select", lambda: torch.where(cond, x, y)),
        ("cpu/elementwise_reduction", "clamp", "f64", "262144", "clamp with tensor bounds", lambda: torch.minimum(torch.maximum(x, lower), upper)),
        ("cpu/elementwise_reduction", "exp", "f64", "262144", "analytic unary", lambda: torch.exp(x)),
        ("cpu/elementwise_reduction", "log", "f64", "262144", "analytic unary", lambda: torch.log(xp)),
        ("cpu/elementwise_reduction", "sin", "f64", "262144", "analytic unary", lambda: torch.sin(x)),
        ("cpu/elementwise_reduction", "cos", "f64", "262144", "analytic unary", lambda: torch.cos(x)),
        ("cpu/elementwise_reduction", "tanh", "f64", "262144", "analytic unary", lambda: torch.tanh(x)),
        ("cpu/elementwise_reduction", "sqrt", "f64", "262144", "analytic unary", lambda: torch.sqrt(xp)),
        ("cpu/elementwise_reduction", "rsqrt", "f64", "262144", "analytic unary", lambda: torch.rsqrt(xp)),
        ("cpu/elementwise_reduction", "pow", "f64", "262144", "binary analytic", lambda: torch.pow(xp, 1.5)),
        ("cpu/elementwise_reduction", "expm1", "f64", "262144", "analytic unary", lambda: torch.expm1(x)),
        ("cpu/elementwise_reduction", "log1p", "f64", "262144", "analytic unary", lambda: torch.log1p(xp)),
        ("cpu/elementwise_reduction", "chain_log1p_exp_mul", "f64", "262144", "short elementwise chain", lambda: torch.exp(torch.log1p(xp)) * y),
        ("cpu/elementwise_reduction", "reduce_sum_all", "f64", "256x1024", "full reduction", lambda: torch.sum(matrix)),
        ("cpu/elementwise_reduction", "reduce_prod_all", "f64", "256x1024", "full reduction", lambda: torch.prod(prod_matrix)),
        ("cpu/elementwise_reduction", "reduce_max_axis0", "f64", "256x1024", "axis reduction", lambda: torch.max(matrix, dim=0).values),
        ("cpu/elementwise_reduction", "reduce_min_axis1", "f64", "256x1024", "axis reduction", lambda: torch.min(matrix, dim=1).values),
        ("cpu/indexing_layout", "gather", "f64", "65536", "1D gather", lambda: torch.gather(base, 0, gather_idx)),
        ("cpu/indexing_layout", "scatter", "f64", "65536", "1D scatter", lambda: torch.zeros_like(base).scatter(0, scatter_idx, updates)),
        ("cpu/indexing_layout", "slice", "f64", "65536", "static slice", lambda: base[1024 : 65_536 - 1024 : 2]),
        ("cpu/indexing_layout", "dynamic_slice", "f64", "65536", "runtime-start slice", lambda: base[1024 : 1024 + 32_768]),
        ("cpu/indexing_layout", "dynamic_update_slice", "f64", "65536", "runtime-start update", lambda: dynamic_update(base, tensor_f64((32_768,), 2))),
        ("cpu/indexing_layout", "pad", "f64", "65536", "edge padding", lambda: F.pad(base, (128, 128))),
        ("cpu/indexing_layout", "concatenate", "f64", "32768+32768", "concatenate along axis 0", lambda: torch.cat((part_a, part_b), dim=0)),
        ("cpu/indexing_layout", "reverse", "f64", "65536", "reverse axis 0", lambda: torch.flip(base, dims=(0,))),
        ("cpu/linalg_uncovered", "cholesky", "f64", "128x128", "SPD input", lambda: torch.linalg.cholesky(spd128)),
        ("cpu/linalg_uncovered", "eig", "f64", "64x64", "general input", lambda: torch.linalg.eig(a64)),
        ("cpu/linalg_uncovered", "eigvals", "f64", "64x64", "general input values only", lambda: torch.linalg.eigvals(a64)),
        ("cpu/linalg_uncovered", "eigvalsh", "f64", "128x128", "SPD input values only", lambda: torch.linalg.eigvalsh(spd128)),
        ("cpu/linalg_uncovered", "triangular_solve", "f64", "128x128,rhs=16", "lower-triangular solve", lambda: torch.linalg.solve_triangular(l128, rhs128x16, upper=False, left=True, unitriangular=False)),
        ("cpu/linalg_uncovered", "det", "f64", "128x128", "well-conditioned input", lambda: torch.linalg.det(a128)),
        ("cpu/linalg_uncovered", "slogdet", "f64", "128x128", "well-conditioned input", lambda: torch.linalg.slogdet(a128)),
        ("cpu/linalg_uncovered", "inv", "f64", "128x128", "well-conditioned input", lambda: torch.linalg.inv(a128)),
        ("cpu/linalg_uncovered", "pinv", "f64", "128x64", "rectangular input", lambda: torch.linalg.pinv(rect128x64)),
        ("cpu/linalg_uncovered", "norm_fro", "f64", "256x256", "Frobenius norm", lambda: torch.linalg.norm(norm256, ord="fro")),
        ("cpu/linalg_uncovered", "full_piv_lu_solve", "f64", "64x64,rhs=8", "PyTorch direct solve", lambda: torch.linalg.solve(a64, rhs64x8)),
        ("cpu/complex", "conj", "c64", "65536", "complex elementwise", lambda: torch.conj(z)),
        ("cpu/complex", "mul", "c64", "65536", "complex elementwise", lambda: z * z2),
        ("cpu/complex", "div", "c64", "65536", "complex elementwise", lambda: z / zden),
        ("cpu/complex", "exp", "c64", "65536", "complex analytic", lambda: torch.exp(z)),
        ("cpu/complex", "log", "c64", "65536", "complex analytic", lambda: torch.log(zlog)),
        ("cpu/complex", "dot_general_conj", "c64", "128x128", "complex matrix multiply", lambda: z128a @ z128b),
        ("cpu/complex", "svd", "c64", "32x32", "complex SVD", lambda: torch.linalg.svd(z32, full_matrices=True)),
        ("cpu/complex", "qr", "c64", "32x32", "complex QR", lambda: torch.linalg.qr(z32, mode="reduced")),
        ("cpu/complex", "eig", "c64", "32x32", "complex eig", lambda: torch.linalg.eig(z32)),
        ("cpu/complex", "solve", "c64", "32x32,rhs=4", "complex solve", lambda: torch.linalg.solve(z32, z32_rhs)),
        ("cpu/complex", "cholesky", "c64", "32x32", "Hermitian positive definite", lambda: torch.linalg.cholesky(hpd32)),
        ("cpu/complex", "norm_fro", "c64", "64x64", "complex Frobenius norm", lambda: torch.linalg.norm(z64, ord="fro")),
    ]


def dynamic_update(base, update):
    output = base.clone()
    output[1024 : 1024 + 32_768] = update
    return output


def emit_case(
    writer: csv.DictWriter[str],
    args,
    suite: str,
    benchmark: str,
    dtype: str,
    shape: str,
    notes: str,
    fn: Callable[[], object] | None,
) -> None:
    try:
        median_ms, iqr_ms = bench(fn, args.runs, args.warmups)
        writer.writerow(
            {
                "suite": suite,
                "benchmark": benchmark,
                "dtype": dtype,
                "threads": args.num_threads,
                "shape": shape,
                "backend": "pytorch-cpu",
                "median_ms": f"{median_ms:.6f}",
                "iqr_ms": f"{iqr_ms:.6f}",
                "status": "ok",
                "notes": f"{notes}; input allocation outside timed region",
            }
        )
    except Exception as exc:  # noqa: BLE001
        writer.writerow(
            {
                "suite": suite,
                "benchmark": benchmark,
                "dtype": dtype,
                "threads": args.num_threads,
                "shape": shape,
                "backend": "pytorch-cpu",
                "median_ms": "",
                "iqr_ms": "",
                "status": "failed",
                "notes": str(exc),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-threads", required=True, type=int)
    args = parser.parse_args()
    args.runs, args.warmups = runs_from_env()

    configure_torch_threads(args.num_threads)

    append = args.output.exists()
    with args.output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        if not append:
            writer.writeheader()
        for case in make_cases():
            emit_case(writer, args, *case)


if __name__ == "__main__":
    main()
