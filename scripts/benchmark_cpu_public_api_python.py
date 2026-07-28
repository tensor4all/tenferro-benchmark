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


class LazyTensor:
    """Materialize a fixture on first warmup use, never during timed runs."""

    def __init__(self, factory: Callable[[], object]):
        self.factory = factory
        self.value = None

    def get(self):
        if self.value is None:
            self.value = self.factory()
        return self.value

    @staticmethod
    def unwrap(value):
        if isinstance(value, LazyTensor):
            return value.get()
        if isinstance(value, tuple):
            return tuple(LazyTensor.unwrap(item) for item in value)
        if isinstance(value, list):
            return [LazyTensor.unwrap(item) for item in value]
        if isinstance(value, dict):
            return {key: LazyTensor.unwrap(item) for key, item in value.items()}
        return value

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types
        return func(*cls.unwrap(args), **cls.unwrap(kwargs or {}))

    def __getattr__(self, name):
        return getattr(self.get(), name)

    def __getitem__(self, key):
        return self.get()[key]

    def __add__(self, other):
        return self.get() + self.unwrap(other)

    def __sub__(self, other):
        return self.get() - self.unwrap(other)

    def __mul__(self, other):
        return self.get() * self.unwrap(other)

    def __truediv__(self, other):
        return self.get() / self.unwrap(other)

    def __neg__(self):
        return -self.get()

    def __lt__(self, other):
        return self.get() < self.unwrap(other)

    def __matmul__(self, other):
        return self.get() @ self.unwrap(other)


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
    def build():
        import torch

        indices = torch.arange(math_prod(shape), dtype=torch.int64)
        result = ((indices * 37 + seed * 11).remainder(2048).to(torch.float64) - 1024.0) / 1024.0
        return result.reshape(shape)

    return LazyTensor(build)


def tensor_f64_positive(shape: tuple[int, ...], seed: int):
    return LazyTensor(lambda: 0.25 + tensor_f64(shape, seed).get().abs())


def tensor_c64(shape: tuple[int, ...], seed: int):
    return LazyTensor(
        lambda: __import__("torch").complex(
            tensor_f64(shape, seed).get(), tensor_f64(shape, seed + 1).get()
        )
    )


def math_prod(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def well_conditioned(n: int, seed: int):
    def build():
        import torch

        x = tensor_f64((n, n), seed).get().clone()
        x.diagonal().add_(torch.linspace(2.0, 3.0, n, dtype=torch.float64))
        return x

    return LazyTensor(build)


def well_conditioned_c64(n: int, seed: int):
    def build():
        import torch

        x = tensor_c64((n, n), seed).get().clone()
        x.diagonal().add_(torch.linspace(3.0, 4.0, n, dtype=torch.float64))
        return x

    return LazyTensor(build)


def lower_triangular(n: int, seed: int):
    def build():
        import torch

        x = torch.tril(0.05 * tensor_f64((n, n), seed).get())
        x.diagonal().copy_(torch.linspace(2.0, 3.0, n, dtype=torch.float64))
        return x

    return LazyTensor(build)


def spd(n: int, seed: int):
    def build():
        import torch

        return torch.diag(torch.linspace(2.0, 3.0, n, dtype=torch.float64))

    del seed
    return LazyTensor(build)


def hpd_c64(n: int, seed: int):
    def build():
        import torch

        diagonal = torch.linspace(2.0, 3.0, n, dtype=torch.float64).to(torch.complex128)
        return torch.diag(diagonal)

    del seed
    return LazyTensor(build)


def make_cases() -> list[tuple[str, str, str, str, str, Callable[[], object] | None]]:
    import torch
    import torch.nn.functional as F

    fast_n = 33_554_432
    ew_n = 8_388_608
    slow_n = 4_194_304
    x_fast = tensor_f64((fast_n,), 1)
    y_fast = tensor_f64((fast_n,), 2)
    yp_fast = tensor_f64_positive((fast_n,), 2)
    xp_fast = tensor_f64_positive((fast_n,), 1)
    cond_fast = LazyTensor(lambda: torch.arange(fast_n, dtype=torch.int64).remainder(3) == 0)
    x = tensor_f64((ew_n,), 1)
    yp = tensor_f64_positive((ew_n,), 2)
    xp = tensor_f64_positive((ew_n,), 1)
    lower = LazyTensor(lambda: torch.full((ew_n,), -0.5, dtype=torch.float64))
    upper = LazyTensor(lambda: torch.full((ew_n,), 0.5, dtype=torch.float64))
    x_slow = tensor_f64((slow_n,), 1)
    y_slow = tensor_f64((slow_n,), 2)
    xp_slow = tensor_f64_positive((slow_n,), 1)
    matrix_sum = tensor_f64((8192, 4096), 1)
    prod_matrix = LazyTensor(lambda: torch.full((8192, 4096), 1.000001, dtype=torch.float64))
    matrix_max = tensor_f64((2048, 2048), 1)
    matrix_min = tensor_f64((4096, 4096), 1)
    gather_n = 262_144
    base_gather = tensor_f64((gather_n,), 1)
    updates_gather = tensor_f64((gather_n,), 2)
    gather_idx = LazyTensor(
        lambda: (torch.arange(gather_n, dtype=torch.int64) * 37 + 11).remainder(gather_n)
    )
    scatter_idx = gather_idx
    base_slice = tensor_f64((4_194_304,), 1)
    base_update = tensor_f64((2_097_152,), 1)
    update_half = tensor_f64((1_048_576,), 2)
    part_a = tensor_f64((1_048_576,), 1)
    part_b = tensor_f64((1_048_576,), 2)
    spd1536 = spd(1536, 1)
    a160 = well_conditioned(160, 1)
    a192 = well_conditioned(192, 1)
    spd512 = spd(512, 1)
    l4096 = lower_triangular(4096, 1)
    rhs4096x64 = tensor_f64((4096, 64), 2)
    a1024 = well_conditioned(1024, 1)
    a768 = well_conditioned(768, 1)
    a256 = well_conditioned(256, 1)
    rhs256x16 = tensor_f64((256, 16), 2)
    rect512x256 = tensor_f64((512, 256), 1)
    norm2048 = tensor_f64((2048, 2048), 1)
    z_conj = tensor_c64((16_777_216,), 1)
    z_mul = tensor_c64((8_388_608,), 1)
    z_mul2 = tensor_c64((8_388_608,), 2)
    zden = LazyTensor(
        lambda: torch.full((8_388_608,), complex(1.5, 0.25), dtype=torch.complex128)
    )
    z_exp = tensor_c64((4_194_304,), 1)
    zlog = LazyTensor(
        lambda: torch.full((4_194_304,), complex(1.5, 0.25), dtype=torch.complex128)
    )
    z640a = tensor_c64((640, 640), 1)
    z640b = tensor_c64((640, 640), 2)
    z160 = tensor_c64((160, 160), 1)
    z256 = tensor_c64((256, 256), 1)
    z112 = tensor_c64((112, 112), 1)
    z384 = well_conditioned_c64(384, 1)
    z384_rhs = tensor_c64((384, 8), 2)
    hpd448 = hpd_c64(448, 1)
    z_norm = tensor_c64((2048, 1536), 1)

    return [
        ("cpu/elementwise_reduction", "add", "f64", "33554432", "binary elementwise", lambda: x_fast + y_fast),
        ("cpu/elementwise_reduction", "sub", "f64", "33554432", "binary elementwise", lambda: x_fast - y_fast),
        ("cpu/elementwise_reduction", "mul", "f64", "33554432", "binary elementwise", lambda: x_fast * y_fast),
        ("cpu/elementwise_reduction", "div", "f64", "33554432", "binary elementwise", lambda: x_fast / yp_fast),
        ("cpu/elementwise_reduction", "rem", "f64", "8388608", "binary elementwise", lambda: torch.remainder(x, yp)),
        ("cpu/elementwise_reduction", "neg", "f64", "33554432", "unary elementwise", lambda: -x_fast),
        ("cpu/elementwise_reduction", "abs", "f64", "33554432", "unary elementwise", lambda: torch.abs(x_fast)),
        ("cpu/elementwise_reduction", "sign", "f64", "33554432", "unary elementwise", lambda: torch.sign(x_fast)),
        ("cpu/elementwise_reduction", "maximum", "f64", "33554432", "binary elementwise", lambda: torch.maximum(x_fast, y_fast)),
        ("cpu/elementwise_reduction", "minimum", "f64", "33554432", "binary elementwise", lambda: torch.minimum(x_fast, y_fast)),
        ("cpu/elementwise_reduction", "compare_lt", "f64", "33554432", "ordered compare", lambda: x_fast < y_fast),
        ("cpu/elementwise_reduction", "select", "f64", "33554432", "ternary select", lambda: torch.where(cond_fast, x_fast, y_fast)),
        ("cpu/elementwise_reduction", "clamp", "f64", "8388608", "clamp with tensor bounds", lambda: torch.minimum(torch.maximum(x, lower), upper)),
        ("cpu/elementwise_reduction", "exp", "f64", "8388608", "analytic unary", lambda: torch.exp(x)),
        ("cpu/elementwise_reduction", "log", "f64", "8388608", "analytic unary", lambda: torch.log(xp)),
        ("cpu/elementwise_reduction", "sin", "f64", "8388608", "analytic unary", lambda: torch.sin(x)),
        ("cpu/elementwise_reduction", "cos", "f64", "8388608", "analytic unary", lambda: torch.cos(x)),
        ("cpu/elementwise_reduction", "tanh", "f64", "8388608", "analytic unary", lambda: torch.tanh(x)),
        ("cpu/elementwise_reduction", "sqrt", "f64", "33554432", "analytic unary", lambda: torch.sqrt(xp_fast)),
        ("cpu/elementwise_reduction", "rsqrt", "f64", "33554432", "analytic unary", lambda: torch.rsqrt(xp_fast)),
        ("cpu/elementwise_reduction", "pow", "f64", "4194304", "binary analytic", lambda: torch.pow(xp_slow, 1.5)),
        ("cpu/elementwise_reduction", "expm1", "f64", "4194304", "analytic unary", lambda: torch.expm1(x_slow)),
        ("cpu/elementwise_reduction", "log1p", "f64", "4194304", "analytic unary", lambda: torch.log1p(xp_slow)),
        ("cpu/elementwise_reduction", "chain_log1p_exp_mul", "f64", "4194304", "short elementwise chain", lambda: torch.exp(torch.log1p(xp_slow)) * y_slow),
        ("cpu/elementwise_reduction", "reduce_sum_all", "f64", "8192x4096", "full reduction", lambda: torch.sum(matrix_sum)),
        ("cpu/elementwise_reduction", "reduce_prod_all", "f64", "8192x4096", "full reduction", lambda: torch.prod(prod_matrix)),
        ("cpu/elementwise_reduction", "reduce_max_axis0", "f64", "2048x2048", "axis reduction", lambda: torch.max(matrix_max, dim=0).values),
        ("cpu/elementwise_reduction", "reduce_min_axis1", "f64", "4096x4096", "axis reduction", lambda: torch.min(matrix_min, dim=1).values),
        ("cpu/indexing_layout", "gather", "f64", "262144", "1D gather", lambda: torch.gather(base_gather, 0, gather_idx)),
        ("cpu/indexing_layout", "scatter", "f64", "262144", "1D scatter", lambda: torch.zeros_like(base_gather).scatter(0, scatter_idx, updates_gather)),
        ("cpu/indexing_layout", "slice", "f64", "4194304", "static slice", lambda: base_slice[1024 : 4_194_304 - 1024 : 2]),
        ("cpu/indexing_layout", "dynamic_slice", "f64", "4194304", "runtime-start slice", lambda: base_slice[1024 : 1024 + 2_097_152]),
        ("cpu/indexing_layout", "dynamic_update_slice", "f64", "2097152", "runtime-start update", lambda: dynamic_update(base_update, update_half)),
        ("cpu/indexing_layout", "pad", "f64", "2097152", "edge padding", lambda: F.pad(base_update, (128, 128))),
        ("cpu/indexing_layout", "concatenate", "f64", "1048576+1048576", "concatenate along axis 0", lambda: torch.cat((part_a, part_b), dim=0)),
        ("cpu/indexing_layout", "reverse", "f64", "2097152", "reverse axis 0", lambda: torch.flip(base_update, dims=(0,))),
        ("cpu/linalg_uncovered", "cholesky", "f64", "1536x1536", "SPD input", lambda: torch.linalg.cholesky(spd1536)),
        ("cpu/linalg_uncovered", "eig", "f64", "160x160", "general input", lambda: torch.linalg.eig(a160)),
        ("cpu/linalg_uncovered", "eigvals", "f64", "192x192", "general input values only", lambda: torch.linalg.eigvals(a192)),
        ("cpu/linalg_uncovered", "eigvalsh", "f64", "512x512", "SPD input values only", lambda: torch.linalg.eigvalsh(spd512)),
        ("cpu/linalg_uncovered", "triangular_solve", "f64", "4096x4096,rhs=64", "lower-triangular solve", lambda: torch.linalg.solve_triangular(l4096, rhs4096x64, upper=False, left=True, unitriangular=False)),
        ("cpu/linalg_uncovered", "det", "f64", "1024x1024", "well-conditioned input", lambda: torch.linalg.det(a1024)),
        ("cpu/linalg_uncovered", "slogdet", "f64", "1024x1024", "well-conditioned input", lambda: torch.linalg.slogdet(a1024)),
        ("cpu/linalg_uncovered", "inv", "f64", "768x768", "well-conditioned input", lambda: torch.linalg.inv(a768)),
        ("cpu/linalg_uncovered", "pinv", "f64", "512x256", "rectangular input", lambda: torch.linalg.pinv(rect512x256)),
        ("cpu/linalg_uncovered", "norm_fro", "f64", "2048x2048", "Frobenius norm", lambda: torch.linalg.norm(norm2048, ord="fro")),
        ("cpu/linalg_uncovered", "full_piv_lu_solve", "f64", "256x256,rhs=16", "PyTorch direct solve", lambda: torch.linalg.solve(a256, rhs256x16)),
        ("cpu/complex", "conj", "c64", "16777216", "complex elementwise", lambda: torch.conj(z_conj)),
        ("cpu/complex", "mul", "c64", "8388608", "complex elementwise", lambda: z_mul * z_mul2),
        ("cpu/complex", "div", "c64", "8388608", "complex elementwise", lambda: z_mul / zden),
        ("cpu/complex", "exp", "c64", "4194304", "complex analytic", lambda: torch.exp(z_exp)),
        ("cpu/complex", "log", "c64", "4194304", "complex analytic", lambda: torch.log(zlog)),
        ("cpu/complex", "dot_general_conj", "c64", "640x640", "complex matrix multiply", lambda: z640a @ z640b),
        ("cpu/complex", "svd", "c64", "160x160", "complex SVD", lambda: torch.linalg.svd(z160, full_matrices=True)),
        ("cpu/complex", "qr", "c64", "256x256", "complex QR", lambda: torch.linalg.qr(z256, mode="reduced")),
        ("cpu/complex", "eig", "c64", "112x112", "complex eig", lambda: torch.linalg.eig(z112)),
        ("cpu/complex", "solve", "c64", "384x384,rhs=8", "complex solve", lambda: torch.linalg.solve(z384, z384_rhs)),
        ("cpu/complex", "cholesky", "c64", "448x448", "Hermitian positive definite", lambda: torch.linalg.cholesky(hpd448)),
        ("cpu/complex", "norm_fro", "c64", "2048x1536", "complex Frobenius norm", lambda: torch.linalg.norm(z_norm, ord="fro")),
    ]


def dynamic_update(base, update):
    output = base.clone()
    output[1024 : 1024 + update.numel()] = update
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
        cases = make_cases()
        # Pop each closure after use so its large fixture tensors can be
        # released before the next API family is measured. A normal list
        # iterator retains every already-measured closure until loop exit.
        while cases:
            emit_case(writer, args, *cases.pop(0))


if __name__ == "__main__":
    main()
