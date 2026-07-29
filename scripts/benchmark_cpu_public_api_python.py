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
        # Match the low 11 bits of the Rust LCG exactly.  The Rust fixture is
        # loaded as column-major, so reverse/permute dimensions to preserve the
        # same logical values while keeping a native row-major PyTorch tensor.
        result = (
            (indices * 1837 + seed * 335).remainder(2048).to(torch.float64) - 1024.0
        ) / 1024.0
        if len(shape) < 2:
            return result.reshape(shape)
        return result.reshape(tuple(reversed(shape))).permute(tuple(reversed(range(len(shape))))).contiguous()

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
        x.diagonal().add_(2.0 + torch.arange(n, dtype=torch.float64) / n)
        return x

    return LazyTensor(build)


def well_conditioned_c64(n: int, seed: int):
    def build():
        import torch

        x = tensor_c64((n, n), seed).get().clone()
        x.diagonal().add_(3.0 + torch.arange(n, dtype=torch.float64) / n)
        return x

    return LazyTensor(build)


def lower_triangular(n: int, seed: int):
    def build():
        import torch

        x = torch.tril(0.05 * tensor_f64((n, n), seed).get())
        x.diagonal().copy_(2.0 + torch.arange(n, dtype=torch.float64) / n)
        return x

    return LazyTensor(build)


def spd(n: int, seed: int):
    def build():
        import torch

        source = tensor_f64((n, n), seed).get()
        matrix = (0.125 / n) * (source + source.T)
        matrix.diagonal().add_(2.0 + torch.arange(n, dtype=torch.float64) / n)
        return matrix

    return LazyTensor(build)


def hpd_c64(n: int, seed: int):
    def build():
        import torch

        diagonal = (2.0 + torch.arange(n, dtype=torch.float64) / n).to(torch.complex128)
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
    exponent_slow = LazyTensor(lambda: torch.full((slow_n,), 1.5, dtype=torch.float64))
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
    structural_matrix = tensor_f64((4096, 4096), 1)
    reshape_input = tensor_f64((33_554_432,), 1)
    broadcast_input = tensor_f64((8192, 1), 1)
    batched_diagonal_input = tensor_f64((8_388_608, 2, 2), 1)
    diagonal_input = tensor_f64((8192,), 1)
    reuse_vector_out = LazyTensor(lambda: torch.empty((fast_n,), dtype=torch.float64))
    reuse_complex_out = LazyTensor(lambda: torch.empty((16_777_216,), dtype=torch.complex128))
    dot_reuse_a = tensor_f64((1024, 1024), 1)
    dot_reuse_b = tensor_f64((1024, 1024), 2)
    dot_reuse_out = LazyTensor(lambda: torch.zeros((1024, 1024), dtype=torch.float64))
    spd1536 = spd(1536, 1)
    a160 = well_conditioned(160, 1)
    a192 = well_conditioned(192, 1)
    spd512 = spd(512, 1)
    l4096 = lower_triangular(4096, 1)
    rhs4096x64 = tensor_f64((4096, 64), 2)
    a1024 = well_conditioned(1024, 1)
    a768 = well_conditioned(768, 1)
    rect512x256 = tensor_f64((512, 256), 1)
    lu1024 = well_conditioned(1024, 1)
    lstsq_a = tensor_f64((768, 384), 1)
    lstsq_rhs = tensor_f64((768, 16), 2)
    svd_full_a = tensor_f64((768, 384), 1)
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
        ("cpu/elementwise_reduction", "clamp", "f64", "8388608", "clamp with tensor bounds", lambda: torch.clamp(x, min=lower, max=upper)),
        ("cpu/elementwise_reduction", "exp", "f64", "8388608", "analytic unary", lambda: torch.exp(x)),
        ("cpu/elementwise_reduction", "log", "f64", "8388608", "analytic unary", lambda: torch.log(xp)),
        ("cpu/elementwise_reduction", "sin", "f64", "8388608", "analytic unary", lambda: torch.sin(x)),
        ("cpu/elementwise_reduction", "cos", "f64", "8388608", "analytic unary", lambda: torch.cos(x)),
        ("cpu/elementwise_reduction", "tanh", "f64", "8388608", "analytic unary", lambda: torch.tanh(x)),
        ("cpu/elementwise_reduction", "sqrt", "f64", "33554432", "analytic unary", lambda: torch.sqrt(xp_fast)),
        ("cpu/elementwise_reduction", "rsqrt", "f64", "33554432", "analytic unary", lambda: torch.rsqrt(xp_fast)),
        ("cpu/elementwise_reduction", "pow", "f64", "4194304", "binary analytic with tensor exponent", lambda: torch.pow(xp_slow, exponent_slow)),
        ("cpu/elementwise_reduction", "expm1", "f64", "4194304", "analytic unary", lambda: torch.expm1(x_slow)),
        ("cpu/elementwise_reduction", "log1p", "f64", "4194304", "analytic unary", lambda: torch.log1p(xp_slow)),
        ("cpu/elementwise_reduction", "chain_log1p_exp_mul", "f64", "4194304", "short elementwise chain", lambda: torch.exp(torch.log1p(xp_slow)) * y_slow),
        ("cpu/elementwise_reduction", "reduce_sum_all", "f64", "8192x4096", "full reduction", lambda: torch.sum(matrix_sum)),
        ("cpu/elementwise_reduction", "reduce_prod_all", "f64", "8192x4096", "full reduction", lambda: torch.prod(prod_matrix)),
        ("cpu/elementwise_reduction", "reduce_max_axis0", "f64", "2048x2048", "axis reduction", lambda: torch.max(matrix_max, dim=0).values),
        ("cpu/elementwise_reduction", "reduce_min_axis1", "f64", "4096x4096", "axis reduction", lambda: torch.min(matrix_min, dim=1).values),
        ("cpu/indexing_layout", "gather", "f64", "262144", "1D gather", lambda: torch.gather(base_gather, 0, gather_idx)),
        ("cpu/indexing_layout", "scatter", "f64", "262144", "1D scatter", lambda: torch.zeros_like(base_gather).scatter(0, scatter_idx, updates_gather)),
        ("cpu/indexing_layout", "slice", "f64", "4194304 -> 2096128", "static slice materialized to owned output", lambda: base_slice[1024 : 4_194_304 - 1024 : 2].clone()),
        ("cpu/indexing_layout", "dynamic_slice", "f64", "4194304 -> 2097152", "runtime-start slice materialized to owned output", lambda: base_slice[1024 : 1024 + 2_097_152].clone()),
        ("cpu/indexing_layout", "dynamic_update_slice", "f64", "2097152", "runtime-start update", lambda: dynamic_update(base_update, update_half)),
        ("cpu/indexing_layout", "pad", "f64", "2097152", "edge padding", lambda: F.pad(base_update, (128, 128))),
        ("cpu/indexing_layout", "concatenate", "f64", "1048576+1048576", "concatenate along axis 0", lambda: torch.cat((part_a, part_b), dim=0)),
        ("cpu/indexing_layout", "reverse", "f64", "2097152", "reverse axis 0", lambda: torch.flip(base_update, dims=(0,))),
        ("cpu/structural_shape", "transpose", "f64", "4096x4096", "materialized matrix transpose", lambda: structural_matrix.transpose(0, 1).contiguous()),
        ("cpu/structural_shape", "reshape", "f64", "33554432 -> 8192x4096", "materialized reshape; clone makes PyTorch perform the same output-sized write", lambda: reshape_input.reshape(8192, 4096).clone()),
        ("cpu/structural_shape", "broadcast_in_dim", "f64", "8192x1 -> 8192x4096", "materialized broadcast", lambda: broadcast_input.expand(8192, 4096).clone()),
        ("cpu/structural_shape", "cast_f64_f32", "f64->f32", "33554432", "dtype cast", lambda: reshape_input.to(torch.float32)),
        ("cpu/structural_shape", "extract_diagonal", "f64", "8388608x2x2 -> 8388608x2", "batched matrix diagonal extraction", lambda: batched_diagonal_input.diagonal(dim1=1, dim2=2).clone()),
        ("cpu/structural_shape", "embed_diagonal", "f64", "8192 -> 8192x8192", "embed vector as matrix diagonal", lambda: torch.diag(diagonal_input)),
        ("cpu/structural_shape", "tril", "f64", "4096x4096", "lower triangle", lambda: torch.tril(structural_matrix)),
        ("cpu/structural_shape", "triu", "f64", "4096x4096", "upper triangle", lambda: torch.triu(structural_matrix)),
        ("cpu/view_metadata", "reshape_view", "f64", "33554432 -> 8192x4096", "metadata-only view reshape; no output-sized copy", lambda: reshape_input.reshape(8192, 4096)),
        ("cpu/view_metadata", "transpose_view", "f64", "4096x4096", "metadata-only transpose view; no output-sized copy", lambda: structural_matrix.transpose(0, 1)),
        ("cpu/view_metadata", "slice_view", "f64", "4194304 -> 2096128", "metadata-only strided slice view; no output-sized copy", lambda: base_slice[1024:-1024:2]),
        ("cpu/view_metadata", "broadcast_in_dim_view", "f64", "8192x1 -> 8192x4096", "metadata-only zero-stride broadcast view; no output-sized copy", lambda: broadcast_input.expand(8192, 4096)),
        ("cpu/output_reuse", "add_into", "f64", "33554432", "torch.add out= caller-owned output", lambda: torch.add(x_fast, y_fast, out=reuse_vector_out)),
        ("cpu/output_reuse", "sub_into", "f64", "33554432", "torch.sub out= caller-owned output", lambda: torch.sub(x_fast, y_fast, out=reuse_vector_out)),
        ("cpu/output_reuse", "mul_into", "f64", "33554432", "torch.mul out= caller-owned output", lambda: torch.mul(x_fast, y_fast, out=reuse_vector_out)),
        ("cpu/output_reuse", "div_into", "f64", "33554432", "torch.div out= caller-owned output", lambda: torch.div(x_fast, yp_fast, out=reuse_vector_out)),
        ("cpu/output_reuse", "neg_into", "f64", "33554432", "torch.neg out= caller-owned output", lambda: torch.neg(x_fast, out=reuse_vector_out)),
        ("cpu/output_reuse", "conj_into", "c64", "16777216", "torch.conj_physical out= caller-owned output", lambda: torch.conj_physical(z_conj, out=reuse_complex_out)),
        ("cpu/output_reuse", "copy_read_into", "f64", "33554432", "Tensor.copy_ caller-owned output", lambda: reuse_vector_out.copy_(x_fast)),
        ("cpu/output_reuse", "dot_general_read_into", "f64", "1024x1024", "torch.mm out= caller-owned output", lambda: torch.mm(dot_reuse_a, dot_reuse_b, out=dot_reuse_out)),
        ("cpu/output_reuse", "dot_general_read_into_accum", "f64", "1024x1024", "torch.addmm out = lhs @ rhs + out", lambda: torch.addmm(dot_reuse_out, dot_reuse_a, dot_reuse_b, beta=1.0, alpha=1.0, out=dot_reuse_out)),
        ("cpu/einsum_concrete", "einsum_ij_jk_ik", "f64", "1024x1024", "torch.einsum allocation-returning API", lambda: torch.einsum("ij,jk->ik", dot_reuse_a, dot_reuse_b)),
        ("cpu/linalg_uncovered", "cholesky", "f64", "1536x1536", "SPD input", lambda: torch.linalg.cholesky(spd1536)),
        ("cpu/linalg_uncovered", "eig", "f64", "160x160", "general input", lambda: torch.linalg.eig(a160)),
        ("cpu/linalg_uncovered", "eigvals", "f64", "192x192", "general input values only", lambda: torch.linalg.eigvals(a192)),
        ("cpu/linalg_uncovered", "eigvalsh", "f64", "512x512", "SPD input values only", lambda: torch.linalg.eigvalsh(spd512)),
        ("cpu/linalg_uncovered", "triangular_solve", "f64", "4096x4096,rhs=64", "lower-triangular solve", lambda: torch.linalg.solve_triangular(l4096, rhs4096x64, upper=False, left=True, unitriangular=False)),
        ("cpu/linalg_uncovered", "det", "f64", "1024x1024", "well-conditioned input", lambda: torch.linalg.det(a1024)),
        ("cpu/linalg_uncovered", "slogdet", "f64", "1024x1024", "well-conditioned input", lambda: torch.linalg.slogdet(a1024)),
        ("cpu/linalg_uncovered", "inv", "f64", "768x768", "well-conditioned input", lambda: torch.linalg.inv(a768)),
        ("cpu/linalg_uncovered", "pinv", "f64", "512x256", "rectangular input", lambda: torch.linalg.pinv(rect512x256)),
        ("cpu/linalg_uncovered", "pinv_with_rtol", "f64", "512x256", "rectangular input; rtol=1e-12", lambda: torch.linalg.pinv(rect512x256, rtol=1e-12)),
        ("cpu/linalg_uncovered", "lu", "f64", "1024x1024", "partial-pivot LU", lambda: torch.linalg.lu(lu1024)),
        ("cpu/linalg_uncovered", "lstsq", "f64", "768x384,rhs=16", "tall full-column-rank QR least-squares solve; GEL driver; solution output", lambda: torch.linalg.lstsq(lstsq_a, lstsq_rhs, driver="gels").solution),
        ("cpu/linalg_uncovered", "svd_full", "f64", "768x384", "full-matrices SVD", lambda: torch.linalg.svd(svd_full_a, full_matrices=True)),
        ("cpu/linalg_uncovered", "norm_fro", "f64", "2048x2048", "Frobenius norm", lambda: torch.linalg.norm(norm2048, ord="fro")),
        ("cpu/complex", "conj", "c64", "16777216", "physical complex conjugate output", lambda: torch.conj_physical(z_conj)),
        ("cpu/complex", "mul", "c64", "8388608", "complex elementwise", lambda: z_mul * z_mul2),
        ("cpu/complex", "div", "c64", "8388608", "complex elementwise", lambda: z_mul / zden),
        ("cpu/complex", "exp", "c64", "4194304", "complex analytic", lambda: torch.exp(z_exp)),
        ("cpu/complex", "log", "c64", "4194304", "complex analytic", lambda: torch.log(zlog)),
        ("cpu/complex", "dot_general", "c64", "640x640", "complex matrix multiply", lambda: z640a @ z640b),
        ("cpu/complex", "dot_general_with_conj", "c64", "640x640", "conjugated-lhs matrix multiply using PyTorch's lazy conjugate view", lambda: torch.conj(z640a) @ z640b),
        ("cpu/complex", "tensordot", "c64", "640x640", "complex matrix contraction over one axis", lambda: torch.tensordot(z640a, z640b, dims=1)),
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


def selected(case: tuple[str, str, str, str, str, Callable[[], object] | None]) -> bool:
    suite, benchmark, *_ = case
    suite_filter = os.environ.get("PUBLIC_API_SUITE_FILTER", "")
    benchmark_filter = os.environ.get("PUBLIC_API_BENCHMARK_FILTER", "")
    return (not suite_filter or suite == suite_filter) and (
        not benchmark_filter or benchmark in benchmark_filter.split(",")
    )


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
        cases = [case for case in make_cases() if selected(case)]
        # Pop each closure after use so its large fixture tensors can be
        # released before the next API family is measured. A normal list
        # iterator retains every already-measured closure until loop exit.
        while cases:
            emit_case(writer, args, *cases.pop(0))


if __name__ == "__main__":
    main()
