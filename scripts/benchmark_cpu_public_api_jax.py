#!/usr/bin/env python3
"""Run JAX/XLA CPU rows for the public API coverage benchmark suite."""

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

Case = tuple[str, str, str, str, str, Callable[[], object]]


class LazyArray:
    """Materialize a fixture on first warmup use, never during timed runs."""

    def __init__(self, factory: Callable[[], object]):
        self.factory = factory
        self.value = None

    def get(self):
        if self.value is None:
            self.value = self.factory()
        return self.value


def runs_from_env() -> tuple[int, int]:
    profile = os.environ.get("PUBLICATION_GATE_PROFILE", "quick").lower()
    return int(os.environ.get("BENCH_RUNS", 15 if profile == "full" else 7)), int(
        os.environ.get("BENCH_WARMUPS", 3)
    )


def math_prod(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def median_iqr(times: list[float]) -> tuple[float, float]:
    values = sorted(times)
    return (
        statistics.median(values),
        values[(3 * len(values)) // 4] - values[len(values) // 4],
    )


def bench(fn: Callable[[], object], runs: int, warmups: int) -> tuple[float, float]:
    import jax

    # The first warmup performs XLA compilation. All warmups, including that
    # compilation, are outside the measured region.
    for _ in range(max(warmups, 1)):
        jax.block_until_ready(fn())
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        output = jax.block_until_ready(fn())
        times.append((time.perf_counter() - start) * 1000.0)
        del output
    return median_iqr(times)


def tensor_f64(shape: tuple[int, ...], seed: int) -> LazyArray:
    def build():
        import jax.numpy as jnp

        indices = jnp.arange(math_prod(shape), dtype=jnp.int64)
        # Match the Rust fixture's LCG low bits and column-major logical order.
        values = ((indices * 1837 + seed * 335) % 2048).astype(jnp.float64)
        return jnp.reshape((values - 1024.0) / 1024.0, shape, order="F")

    return LazyArray(build)


def tensor_f64_positive(shape: tuple[int, ...], seed: int) -> LazyArray:
    import jax.numpy as jnp

    return LazyArray(lambda: 0.25 + jnp.abs(tensor_f64(shape, seed).get()))


def tensor_c64(shape: tuple[int, ...], seed: int) -> LazyArray:
    import jax.numpy as jnp

    return LazyArray(
        lambda: jnp.asarray(
            tensor_f64(shape, seed).get()
            + 1j * tensor_f64(shape, seed + 1).get(),
            dtype=jnp.complex128,
        )
    )


def well_conditioned(n: int, seed: int) -> LazyArray:
    import jax.numpy as jnp

    return LazyArray(
        lambda: tensor_f64((n, n), seed)
        .get()
        .at[jnp.diag_indices(n)]
        .add(2.0 + jnp.arange(n, dtype=jnp.float64) / n)
    )


def well_conditioned_c64(n: int, seed: int) -> LazyArray:
    import jax.numpy as jnp

    return LazyArray(
        lambda: tensor_c64((n, n), seed)
        .get()
        .at[jnp.diag_indices(n)]
        .add(3.0 + jnp.arange(n, dtype=jnp.float64) / n)
    )


def lower_triangular(n: int, seed: int) -> LazyArray:
    import jax.numpy as jnp

    def build():
        value = jnp.tril(0.05 * tensor_f64((n, n), seed).get())
        return value.at[jnp.diag_indices(n)].set(
            2.0 + jnp.arange(n, dtype=jnp.float64) / n
        )

    return LazyArray(build)


def spd(n: int, seed: int) -> LazyArray:
    import jax.numpy as jnp

    def build():
        source = tensor_f64((n, n), seed).get()
        matrix = (0.125 / n) * (source + source.T)
        return matrix.at[jnp.diag_indices(n)].add(
            2.0 + jnp.arange(n, dtype=jnp.float64) / n
        )

    return LazyArray(build)


def hpd_c64(n: int) -> LazyArray:
    import jax.numpy as jnp

    return LazyArray(
        lambda: jnp.diag(
            (2.0 + jnp.arange(n, dtype=jnp.float64) / n).astype(jnp.complex128)
        )
    )


def compiled(fn: Callable[..., object], *inputs: LazyArray) -> Callable[[], object]:
    import jax

    fn_jit = jax.jit(fn)
    return lambda: fn_jit(*(value.get() for value in inputs))


def make_cases() -> list[Case]:
    import jax.numpy as jnp
    from jax import lax
    from jax.scipy.linalg import lu, solve_triangular

    fast_n = 33_554_432
    ew_n = 8_388_608
    slow_n = 4_194_304
    x_fast = tensor_f64((fast_n,), 1)
    y_fast = tensor_f64((fast_n,), 2)
    yp_fast = tensor_f64_positive((fast_n,), 2)
    xp_fast = tensor_f64_positive((fast_n,), 1)
    cond_fast = LazyArray(lambda: jnp.arange(fast_n, dtype=jnp.int64) % 3 == 0)
    x = tensor_f64((ew_n,), 1)
    yp = tensor_f64_positive((ew_n,), 2)
    xp = tensor_f64_positive((ew_n,), 1)
    lower = LazyArray(lambda: jnp.full((ew_n,), -0.5, dtype=jnp.float64))
    upper = LazyArray(lambda: jnp.full((ew_n,), 0.5, dtype=jnp.float64))
    x_slow = tensor_f64((slow_n,), 1)
    y_slow = tensor_f64((slow_n,), 2)
    xp_slow = tensor_f64_positive((slow_n,), 1)
    exponent = LazyArray(lambda: jnp.full((slow_n,), 1.5, dtype=jnp.float64))
    matrix_sum = tensor_f64((8192, 4096), 1)
    matrix_prod = LazyArray(lambda: jnp.full((8192, 4096), 1.000001, dtype=jnp.float64))
    matrix_max = tensor_f64((2048, 2048), 1)
    matrix_min = tensor_f64((4096, 4096), 1)

    gather_n = 262_144
    gather_base = tensor_f64((gather_n,), 1)
    gather_updates = tensor_f64((gather_n,), 2)
    gather_idx = LazyArray(
        lambda: (jnp.arange(gather_n, dtype=jnp.int64) * 37 + 11) % gather_n
    )
    slice_base = tensor_f64((4_194_304,), 1)
    update_base = tensor_f64((2_097_152,), 1)
    update_half = tensor_f64((1_048_576,), 2)
    start = LazyArray(lambda: jnp.asarray([1024], dtype=jnp.int64))
    part_a = tensor_f64((1_048_576,), 1)
    part_b = tensor_f64((1_048_576,), 2)
    structural_matrix = tensor_f64((4096, 4096), 1)
    reshape_input = tensor_f64((33_554_432,), 1)
    broadcast_input = tensor_f64((8192, 1), 1)
    batched_diagonal_input = tensor_f64((8_388_608, 2, 2), 1)
    diagonal_input = tensor_f64((8192,), 1)

    spd1536 = spd(1536, 1)
    a160 = well_conditioned(160, 1)
    a192 = well_conditioned(192, 1)
    spd512 = spd(512, 1)
    l4096 = lower_triangular(4096, 1)
    rhs4096 = tensor_f64((4096, 64), 2)
    a1024 = well_conditioned(1024, 1)
    a768 = well_conditioned(768, 1)
    rect = tensor_f64((512, 256), 1)
    lu1024 = well_conditioned(1024, 1)
    lstsq_a = tensor_f64((768, 384), 1)
    lstsq_rhs = tensor_f64((768, 16), 2)
    svd_full_a = tensor_f64((768, 384), 1)
    norm = tensor_f64((2048, 2048), 1)

    z_conj = tensor_c64((16_777_216,), 1)
    z_mul = tensor_c64((8_388_608,), 1)
    z_mul2 = tensor_c64((8_388_608,), 2)
    z_den = LazyArray(
        lambda: jnp.full((8_388_608,), 1.5 + 0.25j, dtype=jnp.complex128)
    )
    z_exp = tensor_c64((4_194_304,), 1)
    z_log = LazyArray(
        lambda: jnp.full((4_194_304,), 1.5 + 0.25j, dtype=jnp.complex128)
    )
    z640a = tensor_c64((640, 640), 1)
    z640b = tensor_c64((640, 640), 2)
    z160 = tensor_c64((160, 160), 1)
    z256 = tensor_c64((256, 256), 1)
    z112 = tensor_c64((112, 112), 1)
    z384 = well_conditioned_c64(384, 1)
    z384_rhs = tensor_c64((384, 8), 2)
    hpd448 = hpd_c64(448)
    z_norm = tensor_c64((2048, 1536), 1)

    elem = "cpu/elementwise_reduction"
    idx = "cpu/indexing_layout"
    lin = "cpu/linalg_uncovered"
    cplx = "cpu/complex"
    return [
        (elem, "add", "f64", "33554432", "binary elementwise", compiled(lambda a, b: a + b, x_fast, y_fast)),
        (elem, "sub", "f64", "33554432", "binary elementwise", compiled(lambda a, b: a - b, x_fast, y_fast)),
        (elem, "mul", "f64", "33554432", "binary elementwise", compiled(lambda a, b: a * b, x_fast, y_fast)),
        (elem, "div", "f64", "33554432", "binary elementwise", compiled(lambda a, b: a / b, x_fast, yp_fast)),
        (elem, "rem", "f64", "8388608", "binary elementwise", compiled(jnp.remainder, x, yp)),
        (elem, "neg", "f64", "33554432", "unary elementwise", compiled(lambda a: -a, x_fast)),
        (elem, "abs", "f64", "33554432", "unary elementwise", compiled(jnp.abs, x_fast)),
        (elem, "sign", "f64", "33554432", "unary elementwise", compiled(jnp.sign, x_fast)),
        (elem, "maximum", "f64", "33554432", "binary elementwise", compiled(jnp.maximum, x_fast, y_fast)),
        (elem, "minimum", "f64", "33554432", "binary elementwise", compiled(jnp.minimum, x_fast, y_fast)),
        (elem, "compare_lt", "f64", "33554432", "ordered compare", compiled(lambda a, b: a < b, x_fast, y_fast)),
        (elem, "select", "f64", "33554432", "ternary select", compiled(jnp.where, cond_fast, x_fast, y_fast)),
        (elem, "clamp", "f64", "8388608", "clamp with tensor bounds", compiled(jnp.clip, x, lower, upper)),
        (elem, "exp", "f64", "8388608", "analytic unary", compiled(jnp.exp, x)),
        (elem, "log", "f64", "8388608", "analytic unary", compiled(jnp.log, xp)),
        (elem, "sin", "f64", "8388608", "analytic unary", compiled(jnp.sin, x)),
        (elem, "cos", "f64", "8388608", "analytic unary", compiled(jnp.cos, x)),
        (elem, "tanh", "f64", "8388608", "analytic unary", compiled(jnp.tanh, x)),
        (elem, "sqrt", "f64", "33554432", "analytic unary", compiled(jnp.sqrt, xp_fast)),
        (elem, "rsqrt", "f64", "33554432", "analytic unary", compiled(lambda a: lax.rsqrt(a), xp_fast)),
        (elem, "pow", "f64", "4194304", "binary analytic with tensor exponent", compiled(jnp.power, xp_slow, exponent)),
        (elem, "expm1", "f64", "4194304", "analytic unary", compiled(jnp.expm1, x_slow)),
        (elem, "log1p", "f64", "4194304", "analytic unary", compiled(jnp.log1p, xp_slow)),
        (elem, "chain_log1p_exp_mul", "f64", "4194304", "short elementwise chain", compiled(lambda a, b: jnp.exp(jnp.log1p(a)) * b, xp_slow, y_slow)),
        (elem, "reduce_sum_all", "f64", "8192x4096", "full reduction", compiled(jnp.sum, matrix_sum)),
        (elem, "reduce_prod_all", "f64", "8192x4096", "full reduction", compiled(jnp.prod, matrix_prod)),
        (elem, "reduce_max_axis0", "f64", "2048x2048", "axis reduction", compiled(lambda a: jnp.max(a, axis=0), matrix_max)),
        (elem, "reduce_min_axis1", "f64", "4096x4096", "axis reduction", compiled(lambda a: jnp.min(a, axis=1), matrix_min)),
        (idx, "gather", "f64", "262144", "1D gather", compiled(lambda a, i: jnp.take(a, i), gather_base, gather_idx)),
        (idx, "scatter", "f64", "262144", "1D scatter", compiled(lambda a, i, u: jnp.zeros_like(a).at[i].set(u), gather_base, gather_idx, gather_updates)),
        (idx, "slice", "f64", "4194304 -> 2096128", "static slice materialized output", compiled(lambda a: a[1024 : 4_194_304 - 1024 : 2], slice_base)),
        (idx, "dynamic_slice", "f64", "4194304 -> 2097152", "runtime-start slice", compiled(lambda a, s: lax.dynamic_slice(a, (s[0],), (2_097_152,)), slice_base, start)),
        (idx, "dynamic_update_slice", "f64", "2097152", "runtime-start update", compiled(lambda a, u, s: lax.dynamic_update_slice(a, u, (s[0],)), update_base, update_half, start)),
        (idx, "pad", "f64", "2097152", "edge padding", compiled(lambda a: jnp.pad(a, (128, 128)), update_base)),
        (idx, "concatenate", "f64", "1048576+1048576", "concatenate along axis 0", compiled(lambda a, b: jnp.concatenate((a, b)), part_a, part_b)),
        (idx, "reverse", "f64", "2097152", "reverse axis 0", compiled(jnp.flip, update_base)),
        ("cpu/structural_shape", "transpose", "f64", "4096x4096", "materialized matrix transpose", compiled(jnp.transpose, structural_matrix)),
        ("cpu/structural_shape", "reshape", "f64", "33554432 -> 8192x4096", "materialized reshape; JAX arrays have value semantics and no public strided-view contract", compiled(lambda a: jnp.reshape(a, (8192, 4096)), reshape_input)),
        ("cpu/structural_shape", "broadcast_in_dim", "f64", "8192x1 -> 8192x4096", "materialized broadcast", compiled(lambda a: jnp.broadcast_to(a, (8192, 4096)), broadcast_input)),
        ("cpu/structural_shape", "cast_f64_f32", "f64->f32", "33554432", "dtype cast", compiled(lambda a: a.astype(jnp.float32), reshape_input)),
        ("cpu/structural_shape", "extract_diagonal", "f64", "8388608x2x2 -> 8388608x2", "batched matrix diagonal extraction", compiled(lambda a: jnp.diagonal(a, axis1=1, axis2=2), batched_diagonal_input)),
        ("cpu/structural_shape", "embed_diagonal", "f64", "8192 -> 8192x8192", "embed vector as matrix diagonal", compiled(jnp.diag, diagonal_input)),
        ("cpu/structural_shape", "tril", "f64", "4096x4096", "lower triangle", compiled(jnp.tril, structural_matrix)),
        ("cpu/structural_shape", "triu", "f64", "4096x4096", "upper triangle", compiled(jnp.triu, structural_matrix)),
        ("cpu/einsum_concrete", "einsum_ij_jk_ik", "f64", "1024x1024", "jnp.einsum allocation-returning API", compiled(lambda a, b: jnp.einsum("ij,jk->ik", a, b), tensor_f64((1024, 1024), 1), tensor_f64((1024, 1024), 2))),
        (lin, "cholesky", "f64", "1536x1536", "SPD input", compiled(jnp.linalg.cholesky, spd1536)),
        (lin, "eig", "f64", "160x160", "general input", compiled(jnp.linalg.eig, a160)),
        (lin, "eigvals", "f64", "192x192", "general input values only", compiled(jnp.linalg.eigvals, a192)),
        (lin, "eigvalsh", "f64", "512x512", "SPD input values only", compiled(jnp.linalg.eigvalsh, spd512)),
        (lin, "triangular_solve", "f64", "4096x4096,rhs=64", "lower-triangular solve", compiled(lambda a, b: solve_triangular(a, b, lower=True), l4096, rhs4096)),
        (lin, "det", "f64", "1024x1024", "well-conditioned input", compiled(jnp.linalg.det, a1024)),
        (lin, "slogdet", "f64", "1024x1024", "well-conditioned input", compiled(jnp.linalg.slogdet, a1024)),
        (lin, "inv", "f64", "768x768", "well-conditioned input", compiled(jnp.linalg.inv, a768)),
        (lin, "pinv", "f64", "512x256", "rectangular input", compiled(jnp.linalg.pinv, rect)),
        (lin, "pinv_with_rtol", "f64", "512x256", "rectangular input; rtol=1e-12", compiled(lambda a: jnp.linalg.pinv(a, rtol=1e-12), rect)),
        (lin, "lu", "f64", "1024x1024", "partial-pivot LU", compiled(lu, lu1024)),
        (lin, "lstsq", "f64", "768x384,rhs=16", "tall full-column-rank least-squares solve; solution output; JAX chooses its native solver", compiled(lambda a, b: jnp.linalg.lstsq(a, b)[0], lstsq_a, lstsq_rhs)),
        (lin, "svd_full", "f64", "768x384", "full-matrices SVD", compiled(lambda a: jnp.linalg.svd(a, full_matrices=True), svd_full_a)),
        (lin, "norm_fro", "f64", "2048x2048", "Frobenius norm", compiled(lambda a: jnp.linalg.norm(a, ord="fro"), norm)),
        (cplx, "conj", "c64", "16777216", "physical complex conjugate output", compiled(jnp.conj, z_conj)),
        (cplx, "mul", "c64", "8388608", "complex elementwise", compiled(lambda a, b: a * b, z_mul, z_mul2)),
        (cplx, "div", "c64", "8388608", "complex elementwise", compiled(lambda a, b: a / b, z_mul, z_den)),
        (cplx, "exp", "c64", "4194304", "complex analytic", compiled(jnp.exp, z_exp)),
        (cplx, "log", "c64", "4194304", "complex analytic", compiled(jnp.log, z_log)),
        (cplx, "dot_general", "c64", "640x640", "complex matrix multiply", compiled(lambda a, b: a @ b, z640a, z640b)),
        (cplx, "dot_general_with_conj", "c64", "640x640", "conjugated-lhs complex matrix multiply", compiled(lambda a, b: jnp.conj(a) @ b, z640a, z640b)),
        (cplx, "tensordot", "c64", "640x640", "complex matrix contraction over one axis", compiled(lambda a, b: jnp.tensordot(a, b, axes=1), z640a, z640b)),
        (cplx, "svd", "c64", "160x160", "complex SVD", compiled(lambda a: jnp.linalg.svd(a, full_matrices=True), z160)),
        (cplx, "qr", "c64", "256x256", "complex QR", compiled(lambda a: jnp.linalg.qr(a, mode="reduced"), z256)),
        (cplx, "eig", "c64", "112x112", "complex eig", compiled(jnp.linalg.eig, z112)),
        (cplx, "solve", "c64", "384x384,rhs=8", "complex solve", compiled(jnp.linalg.solve, z384, z384_rhs)),
        (cplx, "cholesky", "c64", "448x448", "Hermitian positive definite", compiled(jnp.linalg.cholesky, hpd448)),
        (cplx, "norm_fro", "c64", "2048x1536", "complex Frobenius norm", compiled(lambda a: jnp.linalg.norm(a, ord="fro"), z_norm)),
    ]


def selected(case: Case) -> bool:
    suite_filter = os.environ.get("PUBLIC_API_SUITE_FILTER")
    benchmark_filter = os.environ.get("PUBLIC_API_BENCHMARK_FILTER", "")
    suite, benchmark = case[0], case[1]
    return (not suite_filter or suite == suite_filter) and (
        not benchmark_filter or benchmark in benchmark_filter.split(",")
    )


def emit_case(writer: csv.DictWriter[str], args, case: Case) -> None:
    suite, benchmark, dtype, shape, notes, fn = case
    try:
        median_ms, iqr_ms = bench(fn, args.runs, args.warmups)
        status, median, iqr, result_notes = (
            "ok",
            f"{median_ms:.6f}",
            f"{iqr_ms:.6f}",
            f"{notes}; inputs and XLA compilation outside timed region",
        )
    except Exception as exc:  # noqa: BLE001
        status, median, iqr, result_notes = "failed", "", "", str(exc)
    writer.writerow(
        {
            "suite": suite,
            "benchmark": benchmark,
            "dtype": dtype,
            "threads": args.num_threads,
            "shape": shape,
            "backend": "jax-cpu",
            "median_ms": median,
            "iqr_ms": iqr,
            "status": status,
            "notes": result_notes,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-threads", required=True, type=int)
    args = parser.parse_args()
    args.runs, args.warmups = runs_from_env()
    os.environ["PJRT_NPROC"] = str(args.num_threads)

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    append = args.output.exists()
    with args.output.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, lineterminator="\n")
        if not append:
            writer.writeheader()
        cases = [case for case in make_cases() if selected(case)]
        while cases:
            emit_case(writer, args, cases.pop(0))


if __name__ == "__main__":
    main()
